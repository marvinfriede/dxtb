# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public packed Result adapter and candidate E6broute mixed-System evaluator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from dxtb import OutputHandler, labels
from dxtb._src.calculators.canonical_padding import (
    _base_signature,
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.legacy_composition import (
    LegacySCFOptions,
    LegacySystemRoute,
    classify_legacy_system,
    compose_legacy_systems,
)
from dxtb._src.calculators.model import System
from dxtb._src.calculators.result import ChargeResult, PotentialResult, Result
from dxtb._src.calculators.singlepoint import _call_tensor
from dxtb._src.constants import defaults
from dxtb._src.scf.functional.eager import _raise_failure_code
from dxtb._src.scf.functional.padded import (
    evaluate_padded_systems,
    validate_padded_charge_spin,
)
from dxtb._src.scf.status import SCFStatus
from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)

__all__ = [
    "evaluate_packed_systems",
    "pack_system_results",
]

_KNOWN_PUBLIC_SINGLEPOINT_KWARGS = frozenset(
    {
        "write_overlap",
        "write_dipole",
        "write_quadrupole",
        "write_hcore",
        "cuda_sync_in_scf",
    }
)

_KNOWN_LEGACY_SCF_DATA_KWARGS = frozenset(
    {
        "fwd_options",
        "bck_options",
        "eigen_options",
    }
)


def _pad_last_dim(tensors: Sequence[Tensor], target_len: int) -> Tensor:
    """Right-pad each tensor along its last dimension and stack along dim 0."""
    padded = [
        (
            t
            if t.shape[-1] == target_len
            else F.pad(t, (0, target_len - t.shape[-1]))
        )
        for t in tensors
    ]
    return torch.stack(padded, dim=0)


def _pad_atom_multipole(
    tensors: Sequence[Tensor | None],
    target_atoms: int,
    components: int,
    *,
    like: Tensor,
) -> Tensor | None:
    """Right-pad optional (nat_i, components) multipole tensors and stack."""
    if all(t is None for t in tensors):
        return None
    padded: list[Tensor] = []
    for t in tensors:
        if t is None:
            padded.append(
                like.new_zeros((*like.shape[:-1], target_atoms, components))
            )
        elif t.shape[-2] == target_atoms:
            padded.append(t)
        else:
            padded.append(F.pad(t, (0, 0, 0, target_atoms - t.shape[-2])))
    return torch.stack(padded, dim=0)


def _pad_matrix_2d(
    tensors: Sequence[Tensor | None],
    target_rows: int,
    target_cols: int,
    *,
    like: Tensor,
) -> Tensor | None:
    """Right-pad optional (..., rows_i, cols_i) matrices and stack along dim 0."""
    if all(t is None for t in tensors):
        return None
    sample = next(t for t in tensors if t is not None)
    lead_shape = sample.shape[:-2]
    padded: list[Tensor] = []
    for t in tensors:
        if t is None:
            padded.append(
                like.new_zeros((*lead_shape, target_rows, target_cols))
            )
        else:
            pad_r = target_rows - t.shape[-2]
            pad_c = target_cols - t.shape[-1]
            if pad_r == 0 and pad_c == 0:
                padded.append(t)
            else:
                padded.append(F.pad(t, (0, pad_c, 0, pad_r)))
    return torch.stack(padded, dim=0)


def _orbital_extent(r: Result) -> int:
    """Infer atomic-orbital extent from any present orbital-indexed field."""
    for mat in (
        r.overlap,
        r.hcore,
        r.density,
        r.hamiltonian,
        r.dipole_integrals,
        r.quadrupole_integrals,
    ):
        if mat is not None:
            return int(mat.shape[-1])
    if r.coefficients is not None:
        return int(r.coefficients.shape[-2])
    if r.charges is not None:
        return int(r.charges.mono.shape[-1])
    if r.potential is not None and r.potential.mono is not None:
        return int(r.potential.mono.shape[-1])
    if r.overlap_norm is not None:
        return int(r.overlap_norm.shape[-1])
    return 0


def _mo_extent(r: Result, default_norb: int) -> int:
    """Infer molecular-orbital extent from present MO-indexed fields."""
    if r.emo is not None:
        return int(r.emo.shape[-1])
    if r.occupation is not None:
        return int(r.occupation.shape[-1])
    if r.coefficients is not None:
        return int(r.coefficients.shape[-1])
    return default_norb


def pack_system_results(
    results: Sequence[Result],
    *,
    legacy_compat: bool = False,
) -> Result:
    """Pack per-System physical Results into the public batched Result layout.

    Parameters
    ----------
    results : Sequence[Result]
        Non-empty ordered sequence of single-System physical ``Result`` values.
    legacy_compat : bool, default False
        When ``True``, emit legacy ``Calculator`` batch metadata (scalar
        maximum ``iterations`` and ``None`` for ``converged``, ``residual``,
        and ``status``). When ``False``, preserve per-lane ``iterations``,
        ``converged``, ``residual``, and ``SCFStatus`` tensors.

    Returns
    -------
    Result
        A single packed ``Result`` with leading batch dimension ``B`` and
        right-zero-padded atom/orbital extents matching ``Calculator`` batch
        mode 1/2.
    """
    if len(results) == 0:
        raise ValueError("pack_system_results requires at least one Result.")

    ref = results[0].energy
    max_atoms = max(int(r.energy.shape[-1]) for r in results)
    max_orbitals = max(_orbital_extent(r) for r in results)
    max_mos = max(_mo_extent(r, max_orbitals) for r in results)

    energy = _pad_last_dim([r.energy for r in results], max_atoms)
    scf = (
        None
        if all(r.scf is None for r in results)
        else _pad_last_dim(
            [
                r.scf if r.scf is not None else ref.new_zeros((max_atoms,))
                for r in results
            ],
            max_atoms,
        )
    )
    fenergy = (
        None
        if all(r.fenergy is None for r in results)
        else _pad_last_dim(
            [
                (
                    r.fenergy
                    if r.fenergy is not None
                    else ref.new_zeros((max_atoms,))
                )
                for r in results
            ],
            max_atoms,
        )
    )

    # Classical contributions in deterministic first-seen occurrence order
    classical_slots: list[tuple[str, int]] = []
    classical_maps: list[dict[tuple[str, int], Tensor]] = []
    for r in results:
        counts: dict[str, int] = {}
        rmap: dict[tuple[str, int], Tensor] = {}
        for label, val in r.classical:
            occ = counts.get(label, 0)
            counts[label] = occ + 1
            slot = (label, occ)
            rmap[slot] = val
            if slot not in classical_slots:
                classical_slots.append(slot)
        classical_maps.append(rmap)
    classical = tuple(
        (
            label,
            _pad_last_dim(
                [
                    rmap.get((label, occ), ref.new_zeros((max_atoms,)))
                    for rmap in classical_maps
                ],
                max_atoms,
            ),
        )
        for label, occ in classical_slots
    )

    # Charges
    if all(r.charges is None for r in results):
        charges = None
    else:
        mono = _pad_last_dim(
            [
                (
                    r.charges.mono
                    if r.charges is not None
                    else ref.new_zeros((max_orbitals,))
                )
                for r in results
            ],
            max_orbitals,
        )
        dipole = _pad_atom_multipole(
            [None if r.charges is None else r.charges.dipole for r in results],
            max_atoms,
            3,
            like=ref,
        )
        quadrupole = _pad_atom_multipole(
            [
                None if r.charges is None else r.charges.quadrupole
                for r in results
            ],
            max_atoms,
            6,
            like=ref,
        )
        charges = ChargeResult(mono=mono, dipole=dipole, quadrupole=quadrupole)

    # Potential
    if all(r.potential is None for r in results):
        potential = None
    else:
        pmono = (
            None
            if all(
                r.potential is None or r.potential.mono is None for r in results
            )
            else _pad_last_dim(
                [
                    (
                        r.potential.mono
                        if (
                            r.potential is not None
                            and r.potential.mono is not None
                        )
                        else ref.new_zeros((max_orbitals,))
                    )
                    for r in results
                ],
                max_orbitals,
            )
        )
        pdipole = _pad_atom_multipole(
            [
                None if r.potential is None else r.potential.dipole
                for r in results
            ],
            max_atoms,
            3,
            like=ref,
        )
        pquadrupole = _pad_atom_multipole(
            [
                None if r.potential is None else r.potential.quadrupole
                for r in results
            ],
            max_atoms,
            6,
            like=ref,
        )
        potential = PotentialResult(
            mono=pmono, dipole=pdipole, quadrupole=pquadrupole
        )

    density = _pad_matrix_2d(
        [r.density for r in results], max_orbitals, max_orbitals, like=ref
    )
    coefficients = _pad_matrix_2d(
        [r.coefficients for r in results], max_orbitals, max_mos, like=ref
    )
    emo = (
        None
        if all(r.emo is None for r in results)
        else _pad_last_dim(
            [
                r.emo if r.emo is not None else ref.new_zeros((max_mos,))
                for r in results
            ],
            max_mos,
        )
    )
    if all(r.occupation is None for r in results):
        occupation = None
    else:
        occ_prefix = next(
            r.occupation.shape[:-1] for r in results if r.occupation is not None
        )
        occupation = _pad_last_dim(
            [
                (
                    r.occupation
                    if r.occupation is not None
                    else ref.new_zeros((*occ_prefix, max_mos))
                )
                for r in results
            ],
            max_mos,
        )
    hamiltonian = _pad_matrix_2d(
        [r.hamiltonian for r in results], max_orbitals, max_orbitals, like=ref
    )
    overlap = _pad_matrix_2d(
        [r.overlap for r in results], max_orbitals, max_orbitals, like=ref
    )
    hcore = _pad_matrix_2d(
        [r.hcore for r in results], max_orbitals, max_orbitals, like=ref
    )
    dipole_integrals = _pad_matrix_2d(
        [r.dipole_integrals for r in results],
        max_orbitals,
        max_orbitals,
        like=ref,
    )
    quadrupole_integrals = _pad_matrix_2d(
        [r.quadrupole_integrals for r in results],
        max_orbitals,
        max_orbitals,
        like=ref,
    )
    overlap_norm = (
        None
        if all(r.overlap_norm is None for r in results)
        else _pad_last_dim(
            [
                (
                    r.overlap_norm
                    if r.overlap_norm is not None
                    else ref.new_zeros((max_orbitals,))
                )
                for r in results
            ],
            max_orbitals,
        )
    )

    lane_iters = torch.stack([r.iterations.reshape(()) for r in results], dim=0)
    if legacy_compat:
        iterations = lane_iters.max()
        converged = None
        residual = None
        status = None
    else:
        iterations = lane_iters
        can_fill_non_scf = all(
            r.converged is not None or r.scf is None or r.density is None
            for r in results
        )
        if any(r.converged is not None for r in results) and can_fill_non_scf:
            converged = torch.stack(
                [
                    (
                        r.converged.reshape(())
                        if r.converged is not None
                        else torch.tensor(
                            True, device=ref.device, dtype=torch.bool
                        )
                    )
                    for r in results
                ],
                dim=0,
            )
        else:
            converged = None

        can_fill_residual = all(
            r.residual is not None or r.scf is None or r.density is None
            for r in results
        )
        if any(r.residual is not None for r in results) and can_fill_residual:
            sample_res = next(
                r.residual for r in results if r.residual is not None
            )
            if all(
                r.residual is None or r.residual.numel() == 1 for r in results
            ):
                residual = torch.stack(
                    [
                        (
                            r.residual.reshape(sample_res.shape)
                            if r.residual is not None
                            else sample_res.new_zeros(sample_res.shape)
                        )
                        for r in results
                    ],
                    dim=0,
                )
            else:
                max_res_len = max(
                    1 if r.residual is None else int(r.residual.shape[-1])
                    for r in results
                )
                residual = _pad_last_dim(
                    [
                        (
                            r.residual.reshape(-1)
                            if r.residual is not None
                            else sample_res.new_zeros((max_res_len,))
                        )
                        for r in results
                    ],
                    max_res_len,
                )
        else:
            residual = None

        can_fill_status = all(
            r.status is not None or r.scf is None or r.density is None
            for r in results
        )
        if any(r.status is not None for r in results) and can_fill_status:
            zero_code = torch.tensor(0, device=ref.device, dtype=torch.int64)
            status = SCFStatus(
                trajectory_failure=torch.stack(
                    [
                        (
                            r.status.trajectory_failure.reshape(())
                            if r.status is not None
                            else zero_code
                        )
                        for r in results
                    ],
                    dim=0,
                ),
                final_failure=torch.stack(
                    [
                        (
                            r.status.final_failure.reshape(())
                            if r.status is not None
                            else zero_code
                        )
                        for r in results
                    ],
                    dim=0,
                ),
            )
        else:
            status = None

    return Result(
        energy=energy,
        scf=scf,
        classical=classical,
        fenergy=fenergy,
        iterations=iterations,
        charges=charges,
        density=density,
        coefficients=coefficients,
        emo=emo,
        occupation=occupation,
        potential=potential,
        hamiltonian=hamiltonian,
        overlap=overlap,
        hcore=hcore,
        dipole_integrals=dipole_integrals,
        quadrupole_integrals=quadrupole_integrals,
        overlap_norm=overlap_norm,
        converged=converged,
        residual=residual,
        status=status,
    )


def _scalar_to_0d_tensor(
    value: Tensor | float | int,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert a scalar int/float or 0D/1-element Tensor to a 0D Tensor."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.tensor(float(value), device=device, dtype=dtype)
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.to(device=device, dtype=dtype)
        if value.ndim == 1 and value.numel() == 1:
            return value.reshape(()).to(device=device, dtype=dtype)
        raise ValueError(
            f"{name} element must be scalar, got shape {tuple(value.shape)}."
        )
    raise TypeError(f"Unsupported {name} input type {type(value)}.")


def _normalize_packed_charges(
    chrg: Tensor | float | int | Sequence[Tensor | float | int],
    lane_positions: Sequence[Tensor],
    *,
    legacy_compat: bool,
) -> list[Tensor]:
    """Normalize batched or per-lane charges under functional or legacy rules."""
    count = len(lane_positions)
    if isinstance(chrg, (int, float)) and not isinstance(chrg, bool):
        if legacy_compat and count > 1 and float(chrg) != float(defaults.CHRG):
            raise ValueError(
                "Cannot set charge for batched calculations with a single "
                "float or integer. Please provide a 1D tensor of charges."
            )
        return [
            torch.tensor(float(chrg), device=pos.device, dtype=pos.dtype)
            for pos in lane_positions
        ]
    if isinstance(chrg, Tensor):
        if chrg.ndim == 0:
            if legacy_compat and count > 1:
                raise ValueError(
                    "Cannot set charge for batched calculations with a 0D "
                    "scalar tensor. Please provide a 1D tensor of charges."
                )
            return [
                chrg.to(device=pos.device, dtype=pos.dtype)
                for pos in lane_positions
            ]
        if chrg.ndim == 1:
            if chrg.numel() == count:
                return [
                    chrg[idx].reshape(()).to(device=pos.device, dtype=pos.dtype)
                    for idx, pos in enumerate(lane_positions)
                ]
            if chrg.numel() == 1 and not legacy_compat:
                return [
                    chrg.reshape(()).to(device=pos.device, dtype=pos.dtype)
                    for pos in lane_positions
                ]
            raise ValueError(
                f"Batched charges tensor must have length {count}, got shape "
                f"{tuple(chrg.shape)}."
            )
        if chrg.ndim == 2 and chrg.shape == (count, 1):
            return [
                chrg[idx, 0].reshape(()).to(device=pos.device, dtype=pos.dtype)
                for idx, pos in enumerate(lane_positions)
            ]
        raise ValueError(
            f"Batched charges tensor has invalid shape {tuple(chrg.shape)} "
            f"for {count} Systems."
        )
    if isinstance(chrg, Sequence):
        if len(chrg) != count:
            raise ValueError(
                f"Charges sequence length {len(chrg)} does not match "
                f"{count} Systems."
            )
        return [
            _scalar_to_0d_tensor(
                item, name=f"charge[{idx}]", device=pos.device, dtype=pos.dtype
            )
            for idx, (item, pos) in enumerate(zip(chrg, lane_positions))
        ]
    raise TypeError(f"Unsupported charge input type {type(chrg)}.")


def _normalize_packed_spins(
    spin: Tensor | float | int | None | Sequence[Tensor | float | int | None],
    lane_positions: Sequence[Tensor],
) -> list[Tensor | None]:
    """Normalize batched or per-lane spin inputs while preserving None."""
    count = len(lane_positions)
    if spin is None:
        return [None for _ in lane_positions]
    if isinstance(spin, (int, float)) and not isinstance(spin, bool):
        return [
            torch.tensor(float(spin), device=pos.device, dtype=pos.dtype)
            for pos in lane_positions
        ]
    if isinstance(spin, Tensor):
        if spin.ndim == 0:
            return [
                spin.to(device=pos.device, dtype=pos.dtype)
                for pos in lane_positions
            ]
        if spin.ndim == 1:
            if spin.numel() == count:
                return [
                    spin[idx].reshape(()).to(device=pos.device, dtype=pos.dtype)
                    for idx, pos in enumerate(lane_positions)
                ]
            if spin.numel() == 1:
                return [
                    spin.reshape(()).to(device=pos.device, dtype=pos.dtype)
                    for pos in lane_positions
                ]
            raise ValueError(
                f"Batched spin tensor must have length {count} or 1, got "
                f"shape {tuple(spin.shape)}."
            )
        if spin.ndim == 2 and spin.shape in ((count, 1), (1, 1)):
            return [
                (spin[idx, 0] if spin.shape[0] == count else spin[0, 0])
                .reshape(())
                .to(device=pos.device, dtype=pos.dtype)
                for idx, pos in enumerate(lane_positions)
            ]
        raise ValueError(
            f"Batched spin tensor has invalid shape {tuple(spin.shape)} "
            f"for {count} Systems."
        )
    if isinstance(spin, Sequence):
        if len(spin) != count:
            raise ValueError(
                f"Spin sequence length {len(spin)} does not match "
                f"{count} Systems."
            )
        result: list[Tensor | None] = []
        for idx, (item, pos) in enumerate(zip(spin, lane_positions)):
            if item is None:
                result.append(None)
            else:
                result.append(
                    _scalar_to_0d_tensor(
                        item,
                        name=f"spin[{idx}]",
                        device=pos.device,
                        dtype=pos.dtype,
                    )
                )
        return result
    raise TypeError(f"Unsupported spin input type {type(spin)}.")


def _normalize_packed_fields(
    name: str,
    value: Tensor | Sequence[Tensor | None] | None,
    lane_positions: Sequence[Tensor],
    expected_shape: tuple[int, ...],
    *,
    legacy_compat: bool,
) -> list[Tensor | None]:
    """Normalize shared or per-lane perturbation tensors."""
    count = len(lane_positions)
    if value is None:
        return [None for _ in lane_positions]
    if isinstance(value, Tensor):
        if value.shape == expected_shape:
            return [
                _call_tensor(name, value, pos, expected_shape)
                for pos in lane_positions
            ]
        if not legacy_compat and value.shape == (count, *expected_shape):
            return [
                _call_tensor(name, value[idx], pos, expected_shape)
                for idx, pos in enumerate(lane_positions)
            ]
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(value.shape)}."
        )
    if not legacy_compat and isinstance(value, Sequence):
        if len(value) != count:
            raise ValueError(
                f"{name} sequence length {len(value)} does not match "
                f"{count} Systems."
            )
        return [
            _call_tensor(name, item, pos, expected_shape)
            for item, pos in zip(value, lane_positions)
        ]
    raise TypeError(f"Unsupported {name} input type {type(value)}.")


def _normalize_system_for_e6bcompat(
    system: System,
    *,
    strict_e6bcompat: bool,
) -> System:
    """Apply E6bcompat GFN2 Fock-mode policy to a single-System instance."""
    if (
        system.config.method == labels.GFN2_XTB
        and system.config.scf.scp_mode != labels.SCP_MODE_FOCK
    ):
        if strict_e6bcompat or system.config.strict or system.config.scf.strict:
            raise NotImplementedError(
                "GFN2-xTB mapped batch evaluation requires Fock-mode SCF "
                "(scp_mode='fock')."
            )
        OutputHandler.warn(
            "Resetting 'scp_mode' to 'fock' because GFN2-xTB requires Fock "
            "matrix mixing.",
            UserWarning,
        )
        new_scf = replace(system.config.scf, scp_mode=labels.SCP_MODE_FOCK)
        new_config = replace(system.config, scf=new_scf)
        new_core = (
            None
            if system.exact_core is None
            else replace(system.exact_core, config=new_config)
        )
        return replace(system, config=new_config, exact_core=new_core)
    return system


def _enforce_decoded_eager_policy(
    result: Result,
    system: System,
    source_id: int,
) -> None:
    """Enforce eager convergence and Fermi-status policy on a decoded Result."""
    if (
        result.energy is not None
        and torch._C._functorch.is_functorch_wrapped_tensor(result.energy)
    ):
        return
    if result.status is not None:
        if bool((result.status.trajectory_failure.detach() != 0).any()):
            try:
                _raise_failure_code(result.status.trajectory_failure.detach())
            except Exception as error:
                setattr(error, "source_id", source_id)
                raise
        if bool((result.status.final_failure.detach() != 0).any()):
            try:
                _raise_failure_code(result.status.final_failure.detach())
            except Exception as error:
                setattr(error, "source_id", source_id)
                raise
    config = system.config.scf
    if (
        config.maxiter > 0
        and result.converged is not None
        and not bool(result.converged.detach())
    ):
        mixer_label = (
            "Simple" if config.mixer == labels.MIXER_LINEAR else "Anderson"
        )
        message = (
            f"SCF does not converge after {config.maxiter} cycles using "
            f"{mixer_label} mixing with a damping factor of {config.damp}."
        )
        if config.force_convergence:
            error = SCFConvergenceError(message)
            setattr(error, "source_id", source_id)
            raise error
        OutputHandler.warn(message, SCFConvergenceWarning)


def evaluate_packed_systems(
    systems: Sequence[System],
    positions: Tensor | Sequence[Tensor],
    chrg: Tensor | float | int | Sequence[Tensor | float | int] = defaults.CHRG,
    spin: (
        Tensor | float | int | None | Sequence[Tensor | float | int | None]
    ) = defaults.SPIN,
    field: Tensor | Sequence[Tensor | None] | None = None,
    field_grad: Tensor | Sequence[Tensor | None] | None = None,
    *,
    legacy_options: LegacySCFOptions | None = None,
    legacy_compat: bool = False,
    strict_e6bcompat: bool = False,
    ignore_unknown_kwargs: bool = False,
    **kwargs: Any,
) -> Result:
    """Evaluate a sequence of Systems and return a single packed Result.

    This function implements the candidate ``E6broute`` batch evaluation path:
    - Classifies each ``System`` via :func:`classify_legacy_system`.
    - Buckets all ``MAPPED`` lanes by canonical exact-core structure via
      :func:`bucket_canonical_exact_cores` and evaluates each bucket with
      :func:`evaluate_padded_systems`.
    - Evaluates ``LEGACY_COMPOSED`` and ``SEPARATE_ROUTE`` lanes via
      :func:`compose_legacy_systems`.
    - Reassembles all decoded per-System ``Result`` objects in original source
      order and packs them into a single batched ``Result`` via
      :func:`pack_system_results`.
    """
    count = len(systems)
    if count == 0:
        raise ValueError(
            "evaluate_packed_systems requires at least one System."
        )

    removed_store = sorted(k for k in kwargs if k.startswith("store_"))
    if removed_store:
        raise TypeError(
            f"Removed store_* keyword arguments are not supported: {removed_store}"
        )
    unknown_public = sorted(set(kwargs) - _KNOWN_PUBLIC_SINGLEPOINT_KWARGS)
    if unknown_public and (strict_e6bcompat or not ignore_unknown_kwargs):
        raise TypeError(
            f"evaluate_packed_systems() got unexpected keyword arguments "
            f"{unknown_public}: unexpected keyword argument {unknown_public[0]!r}"
        )

    effective_legacy_options = legacy_options
    if legacy_options is not None and legacy_options.kwargs:
        removed_legacy_store = sorted(
            k for k in legacy_options.kwargs if k.startswith("store_")
        )
        if removed_legacy_store:
            raise TypeError(
                "Removed store_* keyword arguments are not supported: "
                f"{removed_legacy_store}"
            )
        unknown_legacy = sorted(
            set(legacy_options.kwargs) - _KNOWN_LEGACY_SCF_DATA_KWARGS
        )
        if unknown_legacy:
            if strict_e6bcompat or not ignore_unknown_kwargs:
                raise TypeError(
                    f"evaluate_packed_systems() got unexpected keyword arguments "
                    f"{unknown_legacy}: unexpected keyword argument {unknown_legacy[0]!r}"
                )
            filtered_kwargs = {
                k: v
                for k, v in legacy_options.kwargs.items()
                if k in _KNOWN_LEGACY_SCF_DATA_KWARGS
            }
            effective_legacy_options = (
                LegacySCFOptions(
                    args=legacy_options.args,
                    kwargs=filtered_kwargs or None,
                )
                if (legacy_options.args or filtered_kwargs)
                else None
            )

    normalized_systems = [
        _normalize_system_for_e6bcompat(sys, strict_e6bcompat=strict_e6bcompat)
        for sys in systems
    ]

    # Normalize per-lane positions
    lane_positions: list[Tensor] = []
    if isinstance(positions, Tensor):
        if positions.ndim == 2 and count == 1:
            lane_positions.append(positions)
        elif positions.ndim == 3 and positions.shape[0] == count:
            for idx, sys in enumerate(normalized_systems):
                nat = int(sys.numbers.numel())
                lane_positions.append(positions[idx, :nat])
        else:
            raise ValueError(
                f"Batched positions shape {tuple(positions.shape)} does not "
                f"match {count} Systems."
            )
    else:
        if len(positions) != count:
            raise ValueError("Systems and positions must have equal length.")
        lane_positions = list(positions)

    lane_charges = _normalize_packed_charges(
        chrg, lane_positions, legacy_compat=legacy_compat
    )
    lane_spins = _normalize_packed_spins(spin, lane_positions)
    lane_fields = _normalize_packed_fields(
        "field",
        field,
        lane_positions,
        (3,),
        legacy_compat=legacy_compat,
    )
    lane_field_grads = _normalize_packed_fields(
        "field_grad",
        field_grad,
        lane_positions,
        (3, 3),
        legacy_compat=legacy_compat,
    )

    routes = [
        classify_legacy_system(sys, legacy_options=effective_legacy_options)
        for sys in normalized_systems
    ]
    rejected = [
        idx
        for idx, route in enumerate(routes)
        if route is LegacySystemRoute.REJECTED
    ]
    if rejected:
        raise ValueError(
            f"Systems at source indices {rejected} are rejected by E6b routing."
        )

    ordered_results: dict[int, Result] = {}

    # 1. Evaluate MAPPED lanes via canonical bucketing + evaluate_padded_systems
    mapped_indices = [
        idx
        for idx, route in enumerate(routes)
        if route is LegacySystemRoute.MAPPED
    ]
    if mapped_indices:
        groups: dict[tuple[object, ...], list[int]] = {}
        for source_id in mapped_indices:
            core = normalized_systems[source_id].exact_core
            assert core is not None
            d4_cutoffs = tuple(
                float(descriptor.setup.cutoff.cn_eeq)
                for descriptor in core.classical_terms
                if descriptor.kind == "d4"
            )
            group_key = (
                _base_signature(core),
                d4_cutoffs,
                lane_fields[source_id] is not None,
                lane_field_grads[source_id] is not None,
            )
            groups.setdefault(group_key, []).append(source_id)

        for group_source_ids in groups.values():
            group_cores = tuple(
                normalized_systems[sid].exact_core
                for sid in group_source_ids
                if normalized_systems[sid].exact_core is not None
            )
            bucket = bucket_canonical_exact_cores(group_cores)
            padded_cores = tuple(
                pad_canonical_exact_core(core, bucket, local_idx)
                for local_idx, core in enumerate(group_cores)
            )
            padded_pos = tuple(
                canonicalize_positions(lane_positions[sid], bucket, local_idx)
                for local_idx, sid in enumerate(group_source_ids)
            )
            padded_chrg = tuple(lane_charges[sid] for sid in group_source_ids)
            padded_spin_list: list[Tensor] = []
            for p_core, sid in zip(padded_cores, group_source_ids):
                s_opt = lane_spins[sid]
                nab = validate_padded_charge_spin(
                    p_core, lane_charges[sid], s_opt
                )
                if s_opt is None:
                    padded_spin_list.append(
                        (nab[0] - nab[1])
                        .reshape(())
                        .to(
                            device=lane_charges[sid].device,
                            dtype=lane_charges[sid].dtype,
                        )
                    )
                else:
                    padded_spin_list.append(s_opt)
            padded_spin = tuple(padded_spin_list)

            first_field = lane_fields[group_source_ids[0]]
            if first_field is None:
                bucket_field = None
            elif all(
                lane_fields[sid] is first_field for sid in group_source_ids
            ):
                bucket_field = first_field
            else:
                bucket_field = torch.stack(
                    [
                        lane_fields[sid]
                        for sid in group_source_ids
                        if lane_fields[sid] is not None
                    ],
                    dim=0,
                )

            first_fg = lane_field_grads[group_source_ids[0]]
            if first_fg is None:
                bucket_field_grad = None
            elif all(
                lane_field_grads[sid] is first_fg for sid in group_source_ids
            ):
                bucket_field_grad = first_fg
            else:
                bucket_field_grad = torch.stack(
                    [
                        lane_field_grads[sid]
                        for sid in group_source_ids
                        if lane_field_grads[sid] is not None
                    ],
                    dim=0,
                )

            _, decoded = evaluate_padded_systems(
                padded_cores,
                bucket,
                padded_pos,
                padded_chrg,
                padded_spin,
                field=bucket_field,
                field_grad=bucket_field_grad,
            )
            for source_id, result in zip(group_source_ids, decoded):
                if legacy_compat:
                    _enforce_decoded_eager_policy(
                        result, normalized_systems[source_id], source_id
                    )
                ordered_results[source_id] = result

    # 2. Evaluate LEGACY_COMPOSED and SEPARATE_ROUTE lanes via compose_legacy_systems
    retained_indices = [
        idx
        for idx, route in enumerate(routes)
        if route
        in (LegacySystemRoute.LEGACY_COMPOSED, LegacySystemRoute.SEPARATE_ROUTE)
    ]
    if retained_indices:
        composed = compose_legacy_systems(
            tuple(normalized_systems[idx] for idx in retained_indices),
            tuple(lane_positions[idx] for idx in retained_indices),
            tuple(lane_charges[idx] for idx in retained_indices),
            tuple(lane_spins[idx] for idx in retained_indices),
            source_ids=tuple(retained_indices),
            field=tuple(lane_fields[idx] for idx in retained_indices),
            field_grad=tuple(lane_field_grads[idx] for idx in retained_indices),
            legacy_options=effective_legacy_options,
        )
        for source_id, result in zip(composed.source_ids, composed.results):
            ordered_results[source_id] = result

    final_results = [ordered_results[idx] for idx in range(count)]
    return pack_system_results(final_results, legacy_compat=legacy_compat)
