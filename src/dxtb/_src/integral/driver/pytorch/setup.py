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

Composition-dependent data for PyTorch integral construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import Tensor

from .impls.kernels import DEFAULT_ALGORITHM, get_kernel
from .impls.pairs import PairPlan, prepare

__all__ = ["PytorchIntegralSetup", "setup_integrals"]


@dataclass(frozen=True, eq=False)
class PytorchIntegralSetup:
    """Composition- and parameter-dependent data for integral construction."""

    ihelp: IndexHelper
    alphas: tuple[Tensor, ...]
    coeffs: tuple[Tensor, ...]
    pair_plan: PairPlan
    algorithm: str = DEFAULT_ALGORITHM

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PytorchIntegralSetup:
        """Return setup tensors converted to the requested dtype/device."""
        ihelp = self.ihelp.to(device=device)
        return replace(
            self,
            ihelp=ihelp,
            alphas=tuple(
                alpha.to(device=device, dtype=dtype) for alpha in self.alphas
            ),
            coeffs=tuple(
                coeff.to(device=device, dtype=dtype) for coeff in self.coeffs
            ),
            pair_plan=prepare(ihelp, ihelp.device),
        )


def setup_integrals(
    basis: Basis,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
) -> PytorchIntegralSetup:
    """Build explicit structural integral data from a constructed basis."""
    get_kernel(algorithm)
    alphas, coeffs = basis.create_cgtos()
    return PytorchIntegralSetup(
        ihelp=basis.ihelp,
        alphas=tuple(alphas),
        coeffs=tuple(coeffs),
        pair_plan=prepare(basis.ihelp, basis.numbers.device),
        algorithm=algorithm.casefold(),
    )
