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
from dxtb._src.constants import defaults
from dxtb._src.scf.functional.eager import _raise_failure_code
from dxtb._src.scf.functional.padded import evaluate_padded_systems
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
    max_orbitals = max(
        (
            int(r.overlap.shape[-1])
            if r.overlap is not None
            else (
                int(r.charges.mono.shape[-1])
                if r.charges is not None
                else (int(r.hcore.shape[-1]) if r.hcore is not None else 0)
            )
        )
        for r in results
    )
    max_mos = max(
        int(r.emo.shape[-1]) if r.emo is not None else max_orbitals
        for r in results
    )

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

    # Classical contributions in deterministic first-seen order
    classical_labels: list[str] = []
    for r in results:
        for label, _ in r.classical:
            if label not in classical_labels:
                classical_labels.append(label)
    classical_maps = [dict(r.classical) for r in results]
    classical = tuple(
        (
            label,
            _pad_last_dim(
                [
                    cmap.get(label, ref.new_zeros((max_atoms,)))
                    for cmap in classical_maps
                ],
                max_atoms,
            ),
        )
        for label in classical_labels
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
        converged = (
            torch.stack([r.converged.reshape(()) for r in results], dim=0)
            if all(r.converged is not None for r in results)
            else None
        )
        residual = (
            torch.stack([r.residual for r in results], dim=0)
            if all(r.residual is not None for r in results)
            else None
        )
        status = (
            SCFStatus(
                trajectory_failure=torch.stack(
                    [
                        r.status.trajectory_failure.reshape(())
                        for r in results
                        if r.status is not None
                    ],
                    dim=0,
                ),
                final_failure=torch.stack(
                    [
                        r.status.final_failure.reshape(())
                        for r in results
                        if r.status is not None
                    ],
                    dim=0,
                ),
            )
            if all(r.status is not None for r in results)
            else None
        )

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


def _normalize_scalar_lane_input(
    value: Tensor | float | int | None,
    index: int,
    *,
    default: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Normalize one lane's scalar charge or spin input to a 0D Tensor."""
    if value is None:
        return torch.tensor(default, device=device, dtype=dtype)
    if isinstance(value, (int, float)):
        return torch.tensor(float(value), device=device, dtype=dtype)
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.to(device=device, dtype=dtype)
        if value.ndim == 1 and value.numel() == 1:
            return value.reshape(()).to(device=device, dtype=dtype)
        if value.ndim >= 1 and value.shape[0] > index:
            selected = value[index]
            if selected.numel() == 1:
                return selected.reshape(()).to(device=device, dtype=dtype)
        raise ValueError(
            f"Lane {index} scalar input has invalid shape {tuple(value.shape)}."
        )
    raise TypeError(f"Unsupported scalar input type {type(value)}.")


def _normalize_system_for_e6bcompat(
    system: System,
    *,
    strict_e6bcompat: bool,
) -> System:
    """Reject strategies unsupported by the retained packed GFN2 route."""
    if (
        system.config.method == labels.GFN2_XTB
        and system.config.scf._legacy_representation_code
        != labels.SCP_MODE_FOCK
    ):
        raise NotImplementedError(
            "GFN2-xTB mapped batch evaluation requires an explicit Fock "
            "strategy stage. Set representation=SCFRepresentation.FOCK."
        )
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
        strategy_label = " -> ".join(
            type(stage.method).__name__.removesuffix("Method").lower()
            for stage in config.strategy.stages
        )
        message = (
            f"SCF did not certify after {config.maxiter} iterations using "
            f"strategy '{strategy_label}'."
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
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    legacy_options: LegacySCFOptions | None = None,
    legacy_compat: bool = False,
    strict_e6bcompat: bool = False,
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

    if legacy_options is not None and legacy_options.kwargs:
        if strict_e6bcompat:
            unknown = sorted(legacy_options.kwargs.keys())
            raise TypeError(
                f"singlepoint() got unexpected keyword arguments: {unknown}"
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

    lane_charges: list[Tensor] = []
    lane_spins: list[Tensor | None] = []
    for idx, (sys, pos) in enumerate(zip(normalized_systems, lane_positions)):
        c_raw = chrg[idx] if isinstance(chrg, (tuple, list)) else chrg
        s_raw = spin[idx] if isinstance(spin, (tuple, list)) else spin
        lane_charges.append(
            _normalize_scalar_lane_input(
                c_raw,
                idx,
                default=float(defaults.CHRG),
                device=pos.device,
                dtype=pos.dtype,
            )
        )
        if s_raw is None:
            lane_spins.append(None)
        else:
            lane_spins.append(
                _normalize_scalar_lane_input(
                    s_raw,
                    idx,
                    default=0.0,
                    device=pos.device,
                    dtype=pos.dtype,
                )
            )

    routes = [
        classify_legacy_system(sys, legacy_options=legacy_options)
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
            group_key = (_base_signature(core), d4_cutoffs)
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
            from dxtb._src.scf.functional.padded import (
                validate_padded_charge_spin,
            )

            padded_chrg = tuple(lane_charges[sid] for sid in group_source_ids)
            padded_spin_list: list[Tensor] = []
            for p_core, sid in zip(padded_cores, group_source_ids):
                s_opt = lane_spins[sid]
                if s_opt is None:
                    nab = validate_padded_charge_spin(
                        p_core, lane_charges[sid], None
                    )
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
            _, decoded = evaluate_padded_systems(
                padded_cores,
                bucket,
                padded_pos,
                padded_chrg,
                padded_spin,
                field=field,
                field_grad=field_grad,
            )
            for source_id, result in zip(group_source_ids, decoded):
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
            field=field,
            field_grad=field_grad,
            legacy_options=legacy_options,
        )
        for source_id, result in zip(composed.source_ids, composed.results):
            ordered_results[source_id] = result

    final_results = [ordered_results[idx] for idx in range(count)]
    return pack_system_results(final_results, legacy_compat=legacy_compat)
