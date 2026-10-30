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
"""Generalized symmetric eigensolve for the retained GFN0 route."""

from __future__ import annotations

import torch
from tad_mctc import storch

from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import Tensor

__all__ = ["diagonalize"]


@timer_decorator("Diagonalize", "SCF")
def diagonalize(hamiltonian: Tensor, overlap: Tensor) -> tuple[Tensor, Tensor]:
    """Solve the padded generalized problem ``H C = S C e``.

    ``storch.eighb`` handles zero-overlap padding by moving ghost orbitals to
    the end and returning their eigenvalues as zero. This is the ordering
    required by the occupation mask in the retained non-self-consistent GFN0
    path, so no artificial Hamiltonian shift is needed.
    """
    differentiable = hamiltonian.requires_grad or overlap.requires_grad
    broadening_method = "lorn" if differentiable else None
    factor = torch.finfo(hamiltonian.dtype).eps ** 0.5

    return storch.eighb(
        hamiltonian,
        overlap,
        sort_out=True,
        aux=True,
        is_posdef=False,
        broadening_method=broadening_method,
        factor=factor,
    )
