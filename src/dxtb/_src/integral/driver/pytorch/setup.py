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
    """
    Immutable, transform-visible setup for pure PyTorch integrals.

    Stores setup-time CGTO ``alphas`` and ``coeffs`` expanded from the active
    parametrization. Reusing one instance across geometries supports repeated
    independent backward passes with respect to ``positions`` (``retain_graph=False``).
    Independent repeated backward passes with respect to basis parameters
    (such as ``element.*.slater``) require rebuilding a fresh setup per
    evaluation so setup-time ``slater_to_gauss``/``orthogonalize`` saved
    tensors are not consumed across calls.
    """

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
    dev = basis.device
    index = basis.ihelp
    if index.batch_mode != 0:
        raise ValueError("PyTorch integral setup requires one single System.")
    if index.device != dev:
        raise ValueError(
            f"Basis IndexHelper device ({index.device}) must match Basis "
            f"device ({dev})."
        )
    # Validate the kernel before creating the setup.
    _ = get_kernel(algorithm)
    expansion = basis.expand()
    dtype = (
        expansion.alphas[0].dtype if expansion.alphas else basis.slater.dtype
    )
    pair_plan = prepare(index, dev, dtype=dtype)
    return PytorchIntegralSetup(
        index=index,
        alphas=expansion.alphas,
        coeffs=expansion.coeffs,
        pair_plan=pair_plan,
        algorithm=algorithm.casefold(),
    )
