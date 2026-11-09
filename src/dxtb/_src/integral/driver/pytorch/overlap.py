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
Implementation: Overlap
=======================

PyTorch-based overlap implementations.
"""

from __future__ import annotations


from dxtb._src.basis.bas import Basis
from dxtb._src.typing import Tensor

from .impls.kernels import DEFAULT_ALGORITHM, get_kernel
from .impls.pairs import assemble_matrix
from .setup import (
    PytorchIntegralSetup,
    PytorchIntegralSetupNode,
    setup_integrals,
)

__all__ = [
    "PytorchIntegralSetup",
    "PytorchOverlapSetup",
    "setup_overlap",
    "build_overlap",
]


PytorchOverlapSetup = PytorchIntegralSetup


def setup_overlap(
    basis: Basis,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
) -> PytorchIntegralSetup:
    """Build explicit overlap data from an already constructed basis."""
    return setup_integrals(basis, algorithm=algorithm)


def build_overlap(
    setup: PytorchIntegralSetup | PytorchIntegralSetupNode,
    positions: Tensor,
) -> Tensor:
    """Build an overlap matrix from immutable setup data and current positions."""
    ihelp = None if isinstance(setup, PytorchIntegralSetupNode) else setup.ihelp
    raw = assemble_matrix(
        get_kernel(setup.algorithm),
        ihelp,
        list(setup.alphas),
        list(setup.coeffs),
        positions,
        plan=setup.pair_plan,
    )[0]
    return raw
