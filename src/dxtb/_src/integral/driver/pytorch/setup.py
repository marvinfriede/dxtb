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
Integral Setup
==============

Composition-dependent setup Node for PyTorch integral construction.
"""

from __future__ import annotations

from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import Tensor

from .impls.kernels import DEFAULT_ALGORITHM, get_kernel
from .impls.pairs import PairPlan, prepare

__all__ = ["PytorchIntegralSetup", "setup_integrals"]


class PytorchIntegralSetup(Node):
    """Immutable, transform-visible setup for pure PyTorch integrals."""

    index: IndexHelper = child()
    alphas: tuple[Tensor, ...] = child()
    coeffs: tuple[Tensor, ...] = child()
    pair_plan: PairPlan = child()
    algorithm: str = context()

    def _validate(self) -> None:
        if self.index.batch_mode != 0:
            raise ValueError(
                "PyTorch integral setup requires one single System."
            )


def setup_integrals(
    basis: Basis,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
) -> PytorchIntegralSetup:
    """Build the authoritative PyTorch integral setup Node from ``basis``."""
    if basis.ihelp.batch_mode != 0:
        raise ValueError("PyTorch integral setup requires one single System.")
    get_kernel(algorithm)
    alphas, coeffs = basis.create_cgtos()
    dev = basis.device
    index = (
        basis.ihelp
        if basis.ihelp.device == dev
        else basis.ihelp.to(device=dev)
    )
    pair_plan = prepare(index, dev, dtype=alphas[0].dtype)
    return PytorchIntegralSetup(
        index=index,
        alphas=tuple(alphas),
        coeffs=tuple(coeffs),
        pair_plan=pair_plan,
        algorithm=algorithm.casefold(),
    )
