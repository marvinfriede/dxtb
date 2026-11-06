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

from dxtb._src.calculators.padding import PaddedExactSystemCore
from dxtb._src.integral.evaluation import build_integral_matrices
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


def evaluate_padded_integrals(
    padded_core: PaddedExactSystemCore,
    positions: Tensor,
    charge: Tensor,
) -> PaddedIntegralEvaluation:
    """Evaluate PyTorch overlap, multipole, and H0 integrals for one padded core.

    Setup and canonical bucketing happen before this call. The function is
    designed to be mapped over stacked ``PaddedExactSystemCore`` values; it
    does not construct a System façade or perform per-System dispatch.
    """
    core = padded_core.core
    validity = padded_core.validity
    assert core.integral_setup is not None and core.h0_setup is not None

    # Zero inactive coordinates before kernel evaluation so arbitrary caller
    # padding values cannot produce nonfinite distances or leak into gradients.
    positions = torch.where(validity.atoms.unsqueeze(-1), positions, 0.0)
    matrices, refocc, overlap_norm = build_integral_matrices(
        core.integral_setup,
        core.h0_setup,
        positions,
        charge,
        pair_validity=validity.pair_entries,
    )

    # Zero inactive AO and shell-type blocks after pair-masked kernel assembly
    # so downstream spectral and multipole contractions see exact zeros.
    orbital_mask = validity.orbitals
    matrix_mask = orbital_mask.unsqueeze(-1) & orbital_mask.unsqueeze(-2)
    multipole_mask = matrix_mask.unsqueeze(-3)

    return PaddedIntegralEvaluation(
        hcore=torch.where(matrix_mask, matrices.hcore, 0.0),
        overlap=torch.where(matrix_mask, matrices.overlap, 0.0),
        dipole=(
            None
            if matrices.dipole is None
            else torch.where(multipole_mask, matrices.dipole, 0.0)
        ),
        quadrupole=(
            None
            if matrices.quadrupole is None
            else torch.where(multipole_mask, matrices.quadrupole, 0.0)
        ),
        reference_occupations=torch.where(
            validity.unique_shell_types, refocc, 0.0
        ),
        overlap_norm=torch.where(orbital_mask, overlap_norm, 0.0),
    )
