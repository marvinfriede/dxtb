# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""
Implementation: Multipole (PyTorch)
===================================

Shared implementation of the PyTorch dipole and quadrupole integrals, built
from the interchangeable ``compute_1d`` kernels (see
:mod:`dxtb._src.integral.driver.pytorch.impls.kernels`).
"""

from __future__ import annotations

from dxtb._src.constants import labels
from dxtb._src.integral.types.dipole import shift_dipole_origin
from dxtb._src.integral.types.quadrupole import (
    make_quadrupole_traceless,
    reduce_quadrupole_9_to_6,
    shift_quadrupole_origin,
)
from dxtb._src.typing import Tensor

from .impls.kernels import get_kernel
from .impls.pairs import assemble_matrix
from .impls.pipeline import DIPOLE_COMPONENTS, QUADRUPOLE_COMPONENTS
from .overlap import build_overlap
from .setup import PytorchIntegralSetup

__all__ = ["build_dipole", "build_quadrupole", "evaluate_moments"]


def _build_multipole(
    setup: PytorchIntegralSetup,
    positions: Tensor,
    components: tuple[tuple[int, int, int], ...],
) -> Tensor:
    """Build a raw multipole matrix from setup data and current positions."""
    return assemble_matrix(
        get_kernel(setup.algorithm),
        None,
        setup.alphas,
        setup.coeffs,
        positions,
        components,
        plan=setup.pair_plan,
    )


def build_dipole(
    setup: PytorchIntegralSetup,
    positions: Tensor,
) -> Tensor:
    """Build the standalone origin-centered three-component dipole matrix."""
    return _build_multipole(setup, positions, DIPOLE_COMPONENTS)


def build_quadrupole(
    setup: PytorchIntegralSetup,
    positions: Tensor,
) -> Tensor:
    """Build the standalone origin-centered nine-component quadrupole matrix."""
    return _build_multipole(setup, positions, QUADRUPOLE_COMPONENTS)


def evaluate_moments(
    setup: PytorchIntegralSetup,
    positions: Tensor,
    intlevel: int,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Evaluate normalized overlap and internal xTB ket-centered moments."""
    overlap = build_overlap(setup, positions)
    dipole = None
    quadrupole = None

    if intlevel >= labels.INTLEVEL_DIPOLE:
        raw_dipole = build_dipole(setup, positions)
        orbital_positions = setup.index.spread_atom_to_orbital(
            positions, dim=-2, extra=True
        )
        dipole = shift_dipole_origin(raw_dipole, overlap, orbital_positions)
        if intlevel >= labels.INTLEVEL_QUADRUPOLE:
            raw_quadrupole = build_quadrupole(setup, positions)
            quadrupole = shift_quadrupole_origin(
                reduce_quadrupole_9_to_6(raw_quadrupole),
                raw_dipole,
                overlap,
                orbital_positions,
            )
            quadrupole = make_quadrupole_traceless(quadrupole)

    return overlap, dipole, quadrupole
