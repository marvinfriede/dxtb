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
"""Mapped numerical integral evaluation for canonical padded exact cores."""

from __future__ import annotations

import torch
from tad_mctc.tree import Node, child

from dxtb import labels
from dxtb._src.calculators.padding import PaddedExactSystemCore
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.evaluation import IntegralSetup, build_integral_matrices
from dxtb._src.typing import Tensor

__all__ = ["PaddedIntegralEvaluation", "evaluate_padded_integrals"]


class PaddedIntegralEvaluation(Node):
    """Tensor result of one fixed-layout padded integral evaluation."""

    hcore: Tensor = child()
    overlap: Tensor = child()
    dipole: Tensor | None = child()
    quadrupole: Tensor | None = child()
    reference_occupations: Tensor = child()
    overlap_norm: Tensor = child()


def _validate_inputs(
    padded_core: PaddedExactSystemCore,
    positions: Tensor,
    charge: Tensor,
) -> None:
    """Validate the fixed structural contract before numerical evaluation."""
    core = padded_core.core
    integral = core.integral
    if integral is None:
        raise ValueError("Padded integral evaluation requires integral setup.")
    if integral.driver_type != labels.INTDRIVER_PYTORCH:
        raise TypeError(
            "Padded integral evaluation requires the PyTorch driver."
        )
    if integral.intlevel < labels.INTLEVEL_HCORE:
        raise ValueError("Padded H0 evaluation requires HCORE integral level.")
    backend = integral.backend
    if backend is None:
        raise ValueError("Padded PyTorch integral backend is missing.")
    if core.h0_setup is None:
        raise ValueError("Padded integral evaluation requires H0 setup.")
    if core.h0_setup.method != "gfn1":
        raise TypeError(
            "Canonical padded integral evaluation currently supports GFN1 only."
        )

    validity = padded_core.validity
    n_atoms = core.numbers.numel()
    n_shells = validity.shells.numel()
    n_orbitals = validity.orbitals.numel()
    if validity.shells.shape != (n_shells,):
        raise ValueError("Padded shell validity has an incompatible shape.")
    if validity.unique_elements.ndim != 1:
        raise ValueError("Padded element validity must be one-dimensional.")
    if validity.unique_shell_types.ndim != 1:
        raise ValueError("Padded shell-type validity must be one-dimensional.")
    if core.h0_setup is not None and validity.unique_elements.shape != (
        core.h0_setup.unique.numel(),
    ):
        raise ValueError("Padded element validity does not match H0 setup.")
    if validity.unique_shell_types.shape != (
        len(backend.pair_plan.unique_angular),
    ):
        raise ValueError(
            "Padded shell-type validity does not match integral setup."
        )
    if positions.shape != (n_atoms, 3):
        raise ValueError("Positions do not match the padded atom layout.")
    if positions.device != core.numbers.device:
        raise ValueError("Positions and padded core must share a device.")
    if positions.dtype != core.effective_dtype:
        raise TypeError("Positions must use the padded core effective dtype.")
    if charge.ndim != 0:
        raise ValueError("Padded integral evaluation expects scalar charge.")
    if charge.device != positions.device or charge.dtype != positions.dtype:
        raise ValueError("Charge must match the positions dtype and device.")
    if validity.atoms.shape != (n_atoms,):
        raise ValueError("Padded atom validity has an incompatible shape.")
    for name, mask in (
        ("atom", validity.atoms),
        ("shell", validity.shells),
        ("orbital", validity.orbitals),
        ("element", validity.unique_elements),
        ("shell-type", validity.unique_shell_types),
    ):
        if mask.dtype != torch.bool:
            raise TypeError(f"Padded {name} validity must be Boolean.")
        if mask.device != positions.device:
            raise ValueError(f"Padded {name} validity is on the wrong device.")
    if validity.orbitals.shape != (n_orbitals,):
        raise ValueError("Padded orbital validity has an incompatible shape.")
    if backend.index.shells_to_atom.shape != (n_shells,):
        raise ValueError("Padded shell map does not match shell validity.")
    if backend.index.orbitals_to_shell.shape != (n_orbitals,):
        raise ValueError("Padded AO map does not match orbital validity.")
    if backend.pair_plan.nao != n_orbitals:
        raise ValueError("Padded PairPlan AO size does not match validity.")
    if len(validity.pair_entries) != len(backend.pair_plan.classes):
        raise ValueError("Padded PairPlan validity has an incompatible layout.")
    for mask, pair_class in zip(
        validity.pair_entries, backend.pair_plan.classes
    ):
        if mask.shape != pair_class.ib.shape:
            raise ValueError("Padded PairPlan entry mask has an invalid shape.")
        if mask.dtype != torch.bool:
            raise TypeError("Padded PairPlan entry masks must be Boolean.")
        if mask.device != positions.device:
            raise ValueError(
                "Padded PairPlan entry mask is on the wrong device."
            )
    if core.h0_setup.numbers.shape != core.numbers.shape:
        raise ValueError("Padded H0 numbers do not match the atom layout.")
    if core.h0_setup.index.shells_to_atom.shape != (n_shells,):
        raise ValueError("Padded H0 shell map does not match validity.")
    if core.h0_setup.index.orbitals_to_shell.shape != (n_orbitals,):
        raise ValueError("Padded H0 AO map does not match validity.")
    for index in (
        backend.index.shells_to_atom,
        backend.index.orbitals_to_shell,
        backend.pair_plan.atom,
        core.h0_setup.index.atom_to_unique,
        core.h0_setup.index.shells_to_ushell,
        core.h0_setup.index.unique_angular,
        core.h0_setup.index.shells_to_atom,
        core.h0_setup.index.orbitals_to_shell,
    ):
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "Padded integral/H0 indices must be integer tensors."
            )
        if index.device != positions.device:
            raise ValueError("Padded integral/H0 index is on the wrong device.")
    for value in (*backend.alphas, *backend.coeffs):
        if value.dtype != positions.dtype or value.device != positions.device:
            raise ValueError(
                "Padded basis parameters must match position dtype/device."
            )


def _mask_integrals(
    matrices: IntegralMatrices,
    reference_occupations: Tensor,
    overlap_norm: Tensor,
    padded_core: PaddedExactSystemCore,
) -> PaddedIntegralEvaluation:
    """Zero inactive output rows and columns after finite kernel evaluation."""
    validity = padded_core.validity
    orbital_mask = validity.orbitals
    matrix_mask = orbital_mask.unsqueeze(-1) & orbital_mask.unsqueeze(-2)
    masked = IntegralMatrices(
        hcore=torch.where(matrix_mask, matrices.hcore, 0),
        overlap=torch.where(matrix_mask, matrices.overlap, 0),
        dipole=(
            None
            if matrices.dipole is None
            else torch.where(matrix_mask.unsqueeze(-3), matrices.dipole, 0)
        ),
        quadrupole=(
            None
            if matrices.quadrupole is None
            else torch.where(matrix_mask.unsqueeze(-3), matrices.quadrupole, 0)
        ),
    )
    if reference_occupations.shape != validity.unique_shell_types.shape:
        raise ValueError(
            "Reference occupations do not match canonical shell types."
        )
    occupations = torch.where(
        validity.unique_shell_types, reference_occupations, 0
    )
    norm = torch.where(orbital_mask, overlap_norm, 0)
    return PaddedIntegralEvaluation(
        hcore=masked.hcore,
        overlap=masked.overlap,
        dipole=masked.dipole,
        quadrupole=masked.quadrupole,
        reference_occupations=occupations,
        overlap_norm=norm,
    )


def evaluate_padded_integrals(
    padded_core: PaddedExactSystemCore,
    positions: Tensor,
    charge: Tensor,
) -> PaddedIntegralEvaluation:
    """Evaluate existing PyTorch overlap and H0 kernels for one padded lane.

    Setup and canonical bucketing happen before this call.  The function is
    designed to be mapped over stacked ``PaddedExactSystemCore`` values; it
    does not construct a System façade or perform per-System dispatch.
    """
    _validate_inputs(padded_core, positions, charge)
    core = padded_core.core
    assert core.integral is not None and core.h0_setup is not None
    backend = core.integral.backend
    assert backend is not None
    setup = IntegralSetup(
        driver_type=core.integral.driver_type,
        intlevel=core.integral.intlevel,
        pytorch=backend,
    )
    positions = torch.where(
        padded_core.validity.atoms.unsqueeze(-1), positions, 0
    )
    matrices, refocc, overlap_norm = build_integral_matrices(
        setup,
        core.h0_setup,
        positions,
        charge,
        pair_validity=padded_core.validity.pair_entries,
    )
    return _mask_integrals(matrices, refocc, overlap_norm, padded_core)
