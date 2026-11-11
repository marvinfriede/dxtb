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
"""Packed evaluation of heterogeneous single-System batches.

Batched ``Calculator`` input is split into ordinary single-System lanes. Lanes
execute through :mod:`dxtb._src.calculators.lane_evaluation` (grouped ``vmap``
for lanes sharing one System, independent evaluation otherwise), and the
per-lane Results are then packed into the public batched Result.

Padding happens only in :func:`pack_system_results`, to represent Results of
different atom/orbital extents in one batched container. The physics of each
lane is never evaluated on a padded or remapped structure.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.io.checks import content_checks, shape_checks

from dxtb import OutputHandler, labels
from dxtb._src.calculators.lane_evaluation import (
    evaluate_lane_group,
    group_lanes,
)
from dxtb._src.calculators.model import Model, System
from dxtb._src.calculators.result import ChargeResult, PotentialResult, Result
from dxtb._src.calculators.singlepoint import _call_tensor
from dxtb._src.calculators.types.base import (
    _convert_eval_system_dtype,
    _convert_system_dtype,
)
from dxtb._src.calculators.types.energy import check_singlepoint_options
from dxtb._src.constants import defaults
from dxtb._src.scf.eager import _raise_failure_code
from dxtb._src.scf.status import SCFStatus
from dxtb._src.timing import timer
from dxtb._src.typing import Tensor
from dxtb._src.utils.tensors import disable_functorch
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)

__all__ = [
    "build_calculator_lane_systems",
    "evaluate_calculator_packed_singlepoint",
    "evaluate_packed_systems",
    "pack_system_results",
]


@dataclass(frozen=True)
class _AtomPlacement:
    """Per-lane positions of real atoms on the packed atom axis.

    ``source[i]`` (shape ``(nat_i,)``) lists the packed slots holding lane
    ``i``'s atoms. ``gather[i]`` (shape ``(target,)``) maps every packed slot
    to a lane atom, or to ``nat_i`` for a padding slot; appending one zero to
    a lane tensor and gathering with it places the lane values for prefix
    and non-contiguous padding alike.
    """

    source: tuple[Tensor, ...]
    gather: tuple[Tensor, ...]


def _resolve_atom_placement(
    atom_masks: Sequence[Tensor], device: torch.device
) -> _AtomPlacement:
    """Resolve structural atom masks with one transfer in each direction."""
    masks = torch.stack([mask.detach() for mask in atom_masks]).cpu()
    target = int(masks.shape[-1])
    host = torch.zeros((len(atom_masks), 2 * target), dtype=torch.long)
    counts: list[int] = []
    for lane, mask in enumerate(masks):
        source = torch.nonzero(mask).flatten()
        nat = int(source.numel())
        gather = torch.full((target,), nat, dtype=torch.long)
        gather[source] = torch.arange(nat)
        host[lane, :nat] = source
        host[lane, target:] = gather
        counts.append(nat)

    index = host.to(device)
    return _AtomPlacement(
        source=tuple(index[lane, :nat] for lane, nat in enumerate(counts)),
        gather=tuple(index[lane, target:] for lane in range(len(counts))),
    )


def _place_atoms(t: Tensor, gather: Tensor, dim: int) -> Tensor:
    """Place a lane's atom-resolved tensor on the packed atom axis ``dim``."""
    pad = [0, 0] * (-dim - 1) + [0, 1]
    return F.pad(t, pad).index_select(dim, gather)


def _pad_last_dim(
    tensors: Sequence[Tensor],
    target_len: int,
    *,
    atom_gather: Sequence[Tensor] | None = None,
) -> Tensor:
    """Right-pad or mask-place each tensor along its last dim and stack."""
    if atom_gather is not None:
        return torch.stack(
            [_place_atoms(t, idx, -1) for t, idx in zip(tensors, atom_gather)],
            dim=0,
        )

    return torch.stack(
        [
            t
            if t.shape[-1] == target_len
            else F.pad(t, (0, target_len - t.shape[-1]))
            for t in tensors
        ],
        dim=0,
    )


def _pad_optional_last_dim(
    tensors: Sequence[Tensor | None],
    target_len: int,
    *,
    zero_shapes: Sequence[tuple[int, ...]],
    like: Tensor,
    atom_gather: Sequence[Tensor] | None = None,
) -> Tensor | None:
    """Right-pad optional 1D/ND tensors along their last dim when any is present."""
    if all(t is None for t in tensors):
        return None
    filled = [
        t if t is not None else like.new_zeros(shape)
        for t, shape in zip(tensors, zero_shapes)
    ]
    return _pad_last_dim(filled, target_len, atom_gather=atom_gather)


def _pad_atom_multipole(
    tensors: Sequence[Tensor | None],
    target_atoms: int,
    components: int,
    *,
    like: Tensor,
    atom_gather: Sequence[Tensor] | None = None,
) -> Tensor | None:
    """Right-pad or mask-place optional (nat_i, components) multipoles."""
    if all(t is None for t in tensors):
        return None
    padded: list[Tensor] = []
    for idx, t in enumerate(tensors):
        if t is None:
            padded.append(
                like.new_zeros((*like.shape[:-1], target_atoms, components))
            )
        elif atom_gather is not None:
            padded.append(_place_atoms(t, atom_gather[idx], -2))
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
    target_atoms: int | None = None,
    atom_masks: Sequence[Tensor] | None = None,
) -> Result:
    """Pack per-System physical Results into the public batched Result layout.

    Parameters
    ----------
    results : Sequence[Result]
        Non-empty ordered sequence of single-System physical ``Result`` values.
    target_atoms : int | None, default None
        Optional explicit packed atom extent (e.g. ``numbers.shape[1]`` when
        all lanes have trailing zero padding).
    atom_masks : Sequence[Tensor] | None, default None
        Optional per-lane boolean masks of length ``target_atoms`` indicating
        real-atom positions when ``numbers`` contains non-contiguous padding.

    Returns
    -------
    Result
        A single packed ``Result`` with leading batch dimension ``B`` and
        zero-padded atom/orbital extents matching ``Calculator`` batch mode 1/2.
    """
    if len(results) == 0:
        raise ValueError("pack_system_results requires at least one Result.")
    return _pack_lane_results(
        results,
        target_atoms=target_atoms,
        atom_gather=(
            None
            if atom_masks is None
            else _resolve_atom_placement(
                atom_masks, results[0].energy.device
            ).gather
        ),
    )


def _pack_lane_results(
    results: Sequence[Result],
    *,
    target_atoms: int | None,
    atom_gather: Sequence[Tensor] | None,
) -> Result:
    """Pack Results; ``atom_gather`` is the resolved non-contiguous placement."""

    ref = results[0].energy
    max_atoms = max(int(r.energy.shape[-1]) for r in results)
    if target_atoms is not None:
        max_atoms = max(max_atoms, int(target_atoms))
    max_orbitals = max(_orbital_extent(r) for r in results)
    max_mos = max(_mo_extent(r, max_orbitals) for r in results)

    atom_shapes = [(int(r.energy.shape[-1]),) for r in results]
    orb_shapes = [(max_orbitals,) for _ in results]
    mo_shapes = [(max_mos,) for _ in results]

    energy = _pad_last_dim(
        [r.energy for r in results], max_atoms, atom_gather=atom_gather
    )
    scf = _pad_optional_last_dim(
        [r.scf for r in results],
        max_atoms,
        zero_shapes=atom_shapes,
        like=ref,
        atom_gather=atom_gather,
    )
    fenergy = _pad_optional_last_dim(
        [r.fenergy for r in results],
        max_atoms,
        zero_shapes=atom_shapes,
        like=ref,
        atom_gather=atom_gather,
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
                    rmap.get(
                        (label, occ),
                        ref.new_zeros((int(r.energy.shape[-1]),)),
                    )
                    for r, rmap in zip(results, classical_maps)
                ],
                max_atoms,
                atom_gather=atom_gather,
            ),
        )
        for label, occ in classical_slots
    )

    # Charges
    if all(r.charges is None for r in results):
        charges = None
    else:
        mono = _pad_optional_last_dim(
            [None if r.charges is None else r.charges.mono for r in results],
            max_orbitals,
            zero_shapes=orb_shapes,
            like=ref,
        )
        assert mono is not None
        dipole = _pad_atom_multipole(
            [None if r.charges is None else r.charges.dipole for r in results],
            max_atoms,
            3,
            like=ref,
            atom_gather=atom_gather,
        )
        quadrupole = _pad_atom_multipole(
            [
                None if r.charges is None else r.charges.quadrupole
                for r in results
            ],
            max_atoms,
            6,
            like=ref,
            atom_gather=atom_gather,
        )
        charges = ChargeResult(mono=mono, dipole=dipole, quadrupole=quadrupole)

    # Potential
    if all(r.potential is None for r in results):
        potential = None
    else:
        pmono = _pad_optional_last_dim(
            [
                None if r.potential is None else r.potential.mono
                for r in results
            ],
            max_orbitals,
            zero_shapes=orb_shapes,
            like=ref,
        )
        pdipole = _pad_atom_multipole(
            [
                None if r.potential is None else r.potential.dipole
                for r in results
            ],
            max_atoms,
            3,
            like=ref,
            atom_gather=atom_gather,
        )
        pquadrupole = _pad_atom_multipole(
            [
                None if r.potential is None else r.potential.quadrupole
                for r in results
            ],
            max_atoms,
            6,
            like=ref,
            atom_gather=atom_gather,
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
    emo = _pad_optional_last_dim(
        [r.emo for r in results],
        max_mos,
        zero_shapes=mo_shapes,
        like=ref,
    )
    if all(r.occupation is None for r in results):
        occupation = None
    else:
        occ_prefix = next(
            r.occupation.shape[:-1] for r in results if r.occupation is not None
        )
        occupation = _pad_optional_last_dim(
            [r.occupation for r in results],
            max_mos,
            zero_shapes=[(*occ_prefix, max_mos) for _ in results],
            like=ref,
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
    overlap_norm = _pad_optional_last_dim(
        [r.overlap_norm for r in results],
        max_orbitals,
        zero_shapes=orb_shapes,
        like=ref,
    )

    iterations = torch.stack(
        [r.iterations.reshape(()) for r in results], dim=0
    )
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


def _validate_tensor_call_input_dd(
    value: Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Validate device and dtype of a Tensor call input (charge or spin)."""
    if value.dtype == torch.bool:
        raise TypeError(f"Unsupported boolean tensor for {name}.")
    if value.device != device:
        raise DeviceError(
            f"Device mismatch: positions are on '{device}', "
            f"but {name} is on '{value.device}'."
        )
    if value.is_floating_point() and value.dtype != dtype:
        raise DtypeError(
            f"Dtype mismatch: positions are of type '{dtype}', "
            f"but {name} is of type '{value.dtype}'."
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
        _validate_tensor_call_input_dd(
            value,
            name=name,
            device=device,
            dtype=dtype,
        )
        if value.ndim == 0:
            return value.to(device=device, dtype=dtype)
        if value.ndim == 1 and value.numel() == 1:
            return value.reshape(()).to(device=device, dtype=dtype)
        raise ValueError(
            f"{name} element must be scalar, got shape {tuple(value.shape)}."
        )
    raise TypeError(f"Unsupported {name} input type {type(value)}.")


def _shared_scalar_lanes(
    value: float, lane_positions: Sequence[Tensor]
) -> list[Tensor]:
    """Broadcast a Python scalar call input as one tensor per device/dtype."""
    created: dict[tuple[torch.device, torch.dtype], Tensor] = {}
    lanes: list[Tensor] = []
    for pos in lane_positions:
        key = (pos.device, pos.dtype)
        if key not in created:
            created[key] = torch.tensor(
                value, device=pos.device, dtype=pos.dtype
            )
        lanes.append(created[key])
    return lanes


def _normalize_packed_charges(
    chrg: Tensor | float | int | Sequence[Tensor | float | int],
    lane_positions: Sequence[Tensor],
) -> list[Tensor]:
    """Normalize batched or per-lane charges."""
    count = len(lane_positions)
    if isinstance(chrg, (int, float)) and not isinstance(chrg, bool):
        return _shared_scalar_lanes(float(chrg), lane_positions)
    if isinstance(chrg, Tensor):
        for pos in lane_positions:
            _validate_tensor_call_input_dd(
                chrg,
                name="charge",
                device=pos.device,
                dtype=pos.dtype,
            )
        if chrg.ndim == 0:
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
            if chrg.numel() == 1:
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
                item,
                name=f"charge[{idx}]",
                device=pos.device,
                dtype=pos.dtype,
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
        return _shared_scalar_lanes(float(spin), lane_positions)
    if isinstance(spin, Tensor):
        for pos in lane_positions:
            _validate_tensor_call_input_dd(
                spin,
                name="spin",
                device=pos.device,
                dtype=pos.dtype,
            )
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
        if value.shape == (count, *expected_shape):
            return [
                _call_tensor(name, value[idx], pos, expected_shape)
                for idx, pos in enumerate(lane_positions)
            ]
        raise ValueError(
            f"{name} must have shape {expected_shape} or "
            f"{(count, *expected_shape)}, got {tuple(value.shape)}."
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
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


def _default_integral_write_path(key: str, system: System) -> str:
    """Return the default `<label.casefold()>.pt` filename of an integral."""
    if key == "write_hcore":
        method = system.config.method
        if method == labels.GFN0_XTB:
            return "gfn0hamiltonian.pt"
        if method == labels.GFN1_XTB:
            return "gfn1hamiltonian.pt"
        if method == labels.GFN2_XTB:
            return "gfn2hamiltonian.pt"
        raise RuntimeError(f"No default filename is available for {key}.")
    driver = system.config.ints.driver
    suffix = "libcint" if driver == labels.INTDRIVER_LIBCINT else "pytorch"
    prefix_map = {
        "write_overlap": "overlap",
        "write_dipole": "dipole",
        "write_quadrupole": "quadrupole",
    }
    if key not in prefix_map:
        raise RuntimeError(f"No default filename is available for {key}.")
    return f"{prefix_map[key]}{suffix}.pt"


def _write_packed_integrals(
    result: Result,
    system: System,
    kwargs: dict[str, Any],
) -> None:
    """Persist requested integral matrices using the Calculator write_* contract."""
    for key, matrix in (
        ("write_overlap", result.overlap),
        ("write_dipole", result.dipole_integrals),
        ("write_quadrupole", result.quadrupole_integrals),
        ("write_hcore", result.hcore),
    ):
        path = kwargs.get(key, False)
        if path is False or matrix is None:
            continue
        if path is None or path is True:
            path = _default_integral_write_path(key, system)
        torch.save(matrix, path)


def _enforce_eager_status_policy(
    results: Sequence[Result],
    system: System,
    source_ids: Sequence[int],
) -> None:
    """Raise eager convergence and Fermi-status diagnostics for a mapped group.

    The status flags of all lanes are moved to the host in one transfer. The
    lanes are then checked in group order with the eager precedence per lane:
    trajectory failure, final failure, convergence.
    """
    first = results[0]
    if first.energy is not None and (
        torch._C._functorch.is_functorch_wrapped_tensor(first.energy)
    ):
        return

    config = system.config.scf
    has_status = first.status is not None
    check_converged = config.maxiter > 0 and first.converged is not None
    flags: list[Tensor] = []
    if has_status:
        flags.append(
            torch.stack(
                [
                    (r.status.trajectory_failure.detach() != 0).any()
                    for r in results
                ]
            )
        )
        flags.append(
            torch.stack(
                [(r.status.final_failure.detach() != 0).any() for r in results]
            )
        )
    if check_converged:
        flags.append(
            torch.stack(
                [~r.converged.detach().reshape(()).bool() for r in results]
            )
        )
    if not flags:
        return
    host = torch.stack(flags).tolist()
    trajectory_failed = host.pop(0) if has_status else None
    final_failed = host.pop(0) if has_status else None
    not_converged = host.pop(0) if check_converged else None

    for lane, (result, source_id) in enumerate(zip(results, source_ids)):
        if trajectory_failed is not None and trajectory_failed[lane]:
            try:
                _raise_failure_code(result.status.trajectory_failure.detach())
            except Exception as error:
                setattr(error, "source_id", source_id)
                raise
        if final_failed is not None and final_failed[lane]:
            try:
                _raise_failure_code(result.status.final_failure.detach())
            except Exception as error:
                setattr(error, "source_id", source_id)
                raise
        if not_converged is not None and not_converged[lane]:
            strategy_label = " -> ".join(
                type(stage.converger).__name__.removesuffix("Converger").lower()
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
    field: Tensor | Sequence[Tensor | None] | None = None,
    field_grad: Tensor | Sequence[Tensor | None] | None = None,
    *,
    eager_status_policy: bool = False,
    target_atoms: int | None = None,
    atom_masks: Sequence[Tensor] | None = None,
    **kwargs: Any,
) -> Result:
    """Evaluate a sequence of Systems and return a single packed Result.

    1. Validate options and normalize per-lane positions, charges, spins and
       fields.
    2. Group lanes with :func:`group_lanes`: lanes referencing the same
       ``System`` object (with the same optional inputs) are mapped with
       ``torch.func.vmap``; every other lane is evaluated independently.
    3. Restore source order and pack the per-lane Results with
       :func:`pack_system_results`.

    With ``eager_status_policy``, mapped Lane groups raise the same
    convergence and Fermi diagnostics as Independent evaluation (the
    batched ``Calculator`` policy); otherwise they are reported in the
    per-Lane status tensors only.
    """
    count = len(systems)
    if count == 0:
        raise ValueError(
            "evaluate_packed_systems requires at least one System."
        )

    # Stage 1: Public Calculator option validation.
    check_singlepoint_options(kwargs, "evaluate_packed_systems")

    # Stage 2: Normalize and validate per-lane positions, charges, spins, fields.
    # Structural atom masks are resolved once for positions and packing.
    placement = None
    if atom_masks is not None:
        placement = _resolve_atom_placement(
            atom_masks,
            (
                positions if isinstance(positions, Tensor) else positions[0]
            ).device,
        )
    effective_target_atoms = target_atoms
    lane_positions: list[Tensor] = []
    if isinstance(positions, Tensor):
        if positions.ndim == 2 and count == 1:
            lane_positions.append(positions)
        elif positions.ndim == 3 and positions.shape[0] == count:
            if effective_target_atoms is None:
                effective_target_atoms = int(positions.shape[1])
            for idx, sys in enumerate(systems):
                if placement is not None:
                    lane_positions.append(
                        positions[idx].index_select(0, placement.source[idx])
                    )
                else:
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

    for sys, pos in zip(systems, lane_positions):
        if sys.batch_mode != 0 or sys.numbers.ndim != 1:
            raise ValueError(
                "evaluate_packed_systems expects single-System (batch_mode=0) "
                "instances."
            )
        nat = int(sys.numbers.numel())
        if pos.shape != (nat, 3):
            raise ValueError(
                f"Core singlepoint expects positions with shape (nat, 3), "
                f"got {tuple(pos.shape)} for System with {nat} atoms."
            )
        if pos.device != sys.numbers.device:
            raise DeviceError(
                f"Device mismatch: System is on '{sys.numbers.device}', "
                f"but positions are on '{pos.device}'."
            )
        expected_dtype = (
            sys.exact_core.effective_dtype
            if sys.exact_core is not None
            else sys.dd["dtype"]
        )
        if pos.dtype != expected_dtype:
            raise DtypeError(
                f"Dtype mismatch: System floating setup is of type "
                f"'{expected_dtype}', but positions are of type '{pos.dtype}'."
            )

    lane_charges = _normalize_packed_charges(chrg, lane_positions)
    lane_spins = _normalize_packed_spins(spin, lane_positions)
    lane_fields = _normalize_packed_fields(
        "field", field, lane_positions, (3,)
    )
    lane_field_grads = _normalize_packed_fields(
        "field_grad", field_grad, lane_positions, (3, 3)
    )

    # Stage 3: Field integral-level and custom-label conflict checks.
    for sys, f_val, fg_val in zip(systems, lane_fields, lane_field_grads):
        if f_val is not None and sys.config.ints.level < labels.INTLEVEL_DIPOLE:
            raise RuntimeError(
                "An electric field requires dipole integrals. Construct the "
                "System with an integral level of at least "
                f"{labels.INTLEVEL_DIPOLE}."
            )
        if (
            fg_val is not None
            and sys.config.ints.level < labels.INTLEVEL_QUADRUPOLE
        ):
            raise RuntimeError(
                "An electric field gradient requires quadrupole integrals. "
                "Construct the System with an integral level of at least "
                f"{labels.INTLEVEL_QUADRUPOLE}."
            )

    # Stage 4: Partition lanes into execution groups. Lanes sharing one
    # System run together under vmap; all others are evaluated independently.
    # Different molecules are never padded to a common layout for physics.
    groups = group_lanes(systems, lane_spins, lane_fields, lane_field_grads)

    ordered_results: dict[int, Result] = {}
    old_cuda_sync = timer.cuda_sync
    if "cuda_sync_in_scf" in kwargs:
        timer.cuda_sync = kwargs.get(
            "cuda_sync_in_scf",
            False if lane_positions[0].device.type == "cpu" else True,
        )
    try:
        for group in groups:
            group_results = evaluate_lane_group(
                group,
                lane_positions,
                lane_charges,
                lane_spins,
                lane_fields,
                lane_field_grads,
            )

            # Independent evaluation raises its own convergence diagnostics;
            # mapped Lanes report them through tensor status unless the
            # caller asks for the same error policy.
            if eager_status_policy and group.use_vmap:
                _enforce_eager_status_policy(
                    group_results, group.system, group.source_ids
                )
            for source_id, result in zip(group.source_ids, group_results):
                ordered_results[source_id] = result
    finally:
        timer.cuda_sync = old_cuda_sync

    # Stage 5: Restore source order and pad heterogeneous shapes into the
    # public batched Result layout.
    final_results = [ordered_results[idx] for idx in range(count)]
    packed_result = _pack_lane_results(
        final_results,
        target_atoms=effective_target_atoms,
        atom_gather=None if placement is None else placement.gather,
    )
    _write_packed_integrals(packed_result, systems[0], kwargs)
    return packed_result


def _sync_lane_system_labels(
    lane_sys: System,
    source_classicals: Sequence[Any],
    source_interactions: Sequence[Any],
    caller_owned_ids: set[int],
) -> System:
    """Restore source classical/interaction labels on a lane System without mutating caller components."""
    classical_changed = False
    interaction_changed = False
    if len(lane_sys.classicals.components) == len(source_classicals):
        new_classicals = list(lane_sys.classicals.components)
        for idx, (lane_comp, orig_comp) in enumerate(
            zip(new_classicals, source_classicals)
        ):
            if lane_comp.label != orig_comp.label:
                if id(lane_comp) in caller_owned_ids:
                    lane_comp = copy.copy(lane_comp)
                    new_classicals[idx] = lane_comp
                lane_comp.label = orig_comp.label
                classical_changed = True
        if classical_changed:
            lane_sys.classicals.components = new_classicals

    if len(lane_sys.interactions.components) == len(source_interactions):
        new_interactions = list(lane_sys.interactions.components)
        for idx, (lane_inter, orig_inter) in enumerate(
            zip(new_interactions, source_interactions)
        ):
            if lane_inter.label != orig_inter.label:
                if id(lane_inter) in caller_owned_ids:
                    lane_inter = copy.copy(lane_inter)
                    new_interactions[idx] = lane_inter
                lane_inter.label = orig_inter.label
                interaction_changed = True
        if interaction_changed:
            lane_sys.interactions.components = new_interactions

    if not (classical_changed or interaction_changed):
        return lane_sys

    exact_core = lane_sys.exact_core
    if exact_core is not None:
        new_c_terms = tuple(
            replace(
                term,
                label=lane_sys.classicals.components[term.position].label,
            )
            for term in exact_core.classical_terms
        )
        new_i_terms = tuple(
            replace(
                term,
                label=lane_sys.interactions.components[term.position].label,
            )
            for term in exact_core.interaction_terms
        )
        exact_core = exact_core.replace(
            classical_terms=new_c_terms,
            interaction_terms=new_i_terms,
        )

    return replace(
        lane_sys,
        exact_core=exact_core,
    )


def build_calculator_lane_systems(calc: Any) -> tuple[System, ...]:
    """Construct deduplicated per-lane single-System instances from a Calculator.

    Identical unpadded lane compositions (such as all conformers in
    ``batch_mode == 2`` or repeated compositions in ``batch_mode == 1``) reuse
    a single ``Model.setup(..., batch_mode=0)`` call within the evaluation and
    share the resulting immutable ``System`` value.
    """
    if calc.numbers.ndim == 1:
        if hasattr(calc, "_prepare_eval_system"):
            return (calc._prepare_eval_system(),)
        return (calc.system,)

    from dxtb._src.calculators.types.base import _convert_eval_system_dtype

    batch_size = int(calc.numbers.shape[0])
    source_classicals = tuple(calc.classicals.components)
    source_interactions = tuple(calc.interactions.components)
    model_classicals = tuple(calc.model.classical)
    model_interactions = tuple(calc.model.interaction)
    caller_owned_ids = {
        id(c)
        for c in (
            *model_classicals,
            *model_interactions,
            *source_classicals,
            *source_interactions,
        )
    }

    current_labels = [c.label for c in source_classicals]
    has_label_collision = len(set(current_labels)) < len(current_labels)

    # Atomic numbers are structural metadata: resolve them on the host with
    # one transfer per tensor instead of one scalar transfer per atom. The
    # device tensors are still the ones passed to ``Model.setup``.
    with disable_functorch():
        host_numbers = calc.numbers.detach().cpu().tolist()
        host_component_numbers = [
            (id(c), c.numbers.detach().cpu().tolist())
            for c in model_classicals
            if getattr(c, "numbers", None) is not None
            and isinstance(c.numbers, Tensor)
            and c.numbers.ndim == 2
        ]

    lane_memo: dict[tuple[object, ...], System] = {}
    lane_systems: list[System] = []
    for idx in range(batch_size):
        host_row = host_numbers[idx]
        real_slots = [slot for slot, z in enumerate(host_row) if z > 0]
        comp_nums_key = tuple(
            (comp_id, tuple(int(rows[idx][slot]) for slot in real_slots))
            for comp_id, rows in host_component_numbers
        )
        dedup_key = (
            tuple(int(host_row[slot]) for slot in real_slots),
            comp_nums_key,
            calc.opts,
            calc.dtype,
            calc.device,
            tuple((id(c), c.label) for c in source_classicals),
            tuple((id(c), c.label) for c in source_interactions),
        )
        lane_sys = lane_memo.get(dedup_key)
        if lane_sys is None:
            # The real atom slots are known on the host; select them with an
            # index tensor instead of a dynamic Boolean mask.
            slots = torch.tensor(
                real_slots, dtype=torch.long, device=calc.numbers.device
            )
            row_nums = calc.numbers[idx].index_select(0, slots)
            lane_classical_list = []
            for c_idx, comp in enumerate(model_classicals):
                has_2d_nums = (
                    getattr(comp, "numbers", None) is not None
                    and isinstance(comp.numbers, Tensor)
                    and comp.numbers.ndim == 2
                )
                # Caller-owned Components are never mutated in place.
                c_copy = copy.copy(comp)
                if has_2d_nums:
                    c_copy.numbers = comp.numbers[idx].index_select(
                        0, slots.to(comp.numbers.device)
                    )
                if has_label_collision:
                    c_copy.label = f"__dxtb_lane_setup_{c_idx}_{comp.label}"
                lane_classical_list.append(c_copy)

            lane_interaction_list = [
                copy.copy(inter) for inter in model_interactions
            ]

            lane_model = Model(
                par=calc.model.par,
                config=calc.opts,
                classical=tuple(lane_classical_list),
                interaction=tuple(lane_interaction_list),
                auto_int_level=False,
            )
            setup_dd = (
                calc.model.par.dd
                if calc.dtype != calc.model.par.dtype
                else calc.dd
            )
            lane_sys = lane_model.setup(row_nums, batch_mode=0, dd=setup_dd)
            lane_sys = _sync_lane_system_labels(
                lane_sys,
                source_classicals,
                source_interactions,
                caller_owned_ids,
            )
            if calc.dtype != calc.model.par.dtype:
                lane_sys = _convert_eval_system_dtype(lane_sys, calc.dtype)
            lane_memo[dedup_key] = lane_sys
        lane_systems.append(lane_sys)

    return tuple(lane_systems)


def evaluate_calculator_packed_singlepoint(
    calc: Any,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | Sequence[Tensor | None] | None = None,
    field_grad: Tensor | Sequence[Tensor | None] | None = None,
    **kwargs: Any,
) -> Result:
    """Evaluate a batched Calculator Lane by Lane over its Systems.

    Builds deduplicated lane Systems (repeated compositions share one System
    and therefore one ``vmap`` group), evaluates them with
    :func:`evaluate_packed_systems`, and returns the packed Result for batched
    ``Calculator.singlepoint`` calls.
    """
    check_singlepoint_options(kwargs, "singlepoint")
    resolved_field, resolved_field_grad = calc._check_fields(
        field, field_grad
    )
    assert shape_checks(calc.numbers, positions, allow_batched=True)
    assert content_checks(
        calc.numbers,
        positions,
        calc.opts.max_element,
        allow_batched=True,
    )
    if calc.numbers.ndim == 1:
        return calc.singlepoint(
            positions,
            chrg=chrg,
            spin=spin,
            field=resolved_field,
            field_grad=resolved_field_grad,
            **kwargs,
        )

    lane_systems = build_calculator_lane_systems(calc)
    # The Calculator synchronizes CUDA in its SCF timer unless told otherwise.
    kwargs.setdefault("cuda_sync_in_scf", positions.device.type != "cpu")
    atom_masks = tuple(calc.numbers[idx] > 0 for idx in range(len(lane_systems)))
    target_atoms = int(calc.numbers.shape[1])
    result = evaluate_packed_systems(
        lane_systems,
        positions,
        chrg=chrg,
        spin=spin,
        field=resolved_field,
        field_grad=resolved_field_grad,
        eager_status_policy=True,
        target_atoms=target_atoms,
        atom_masks=atom_masks,
        **kwargs,
    )
    calc._ncalcs += 1
    return result
