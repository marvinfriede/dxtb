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


from dxtb._src.typing import Tensor

from .impls.kernels import get_kernel
from .impls.pairs import assemble_matrix
from .impls.pipeline import DIPOLE_COMPONENTS, QUADRUPOLE_COMPONENTS
from .setup import PytorchIntegralSetup, PytorchIntegralSetupNode

__all__ = ["build_dipole", "build_quadrupole"]


def _build_multipole(
    setup: PytorchIntegralSetup | PytorchIntegralSetupNode,
    positions: Tensor,
    components: tuple[tuple[int, int, int], ...],
) -> Tensor:
    """Build a raw multipole matrix from setup data and current positions."""
    ihelp = None if isinstance(setup, PytorchIntegralSetupNode) else setup.ihelp
    return assemble_matrix(
        get_kernel(setup.algorithm),
        ihelp,
        list(setup.alphas),
        list(setup.coeffs),
        positions,
        components,
        plan=setup.pair_plan,
    )


def build_dipole(
    setup: PytorchIntegralSetup | PytorchIntegralSetupNode,
    positions: Tensor,
) -> Tensor:
    """Build the raw, origin-centered three-component dipole matrix."""
    return _build_multipole(setup, positions, DIPOLE_COMPONENTS)


def build_quadrupole(
    setup: PytorchIntegralSetup | PytorchIntegralSetupNode,
    positions: Tensor,
) -> Tensor:
    """Build the raw, origin-centered nine-component quadrupole matrix."""
    return _build_multipole(setup, positions, QUADRUPOLE_COMPONENTS)
