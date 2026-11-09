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
Integrals: Utility Functions
============================

Integral-related utility functions.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Tensor

__all__ = ["normalize_integral_matrix", "snorm"]


def snorm(ovlp: Tensor) -> Tensor:
    d = ovlp.diagonal(dim1=-1, dim2=-2)
    zero = torch.tensor(0.0, dtype=d.dtype, device=d.device)
    return torch.where(d == 0.0, zero, torch.pow(d, -0.5))


def normalize_integral_matrix(
    matrix: Tensor, norm: Tensor | None = None
) -> Tensor:
    """Return a normalized integral matrix without changing its owner."""
    if norm is None:
        norm = snorm(matrix)

    if norm.ndim == 1:
        einsum_str = "...ij,i,j->...ij"
    elif norm.ndim == 2:
        einsum_str = "b...ij,bi,bj->b...ij"
    else:
        raise ValueError(f"Invalid norm shape: {norm.shape}")

    return einsum(einsum_str, matrix, norm, norm)
