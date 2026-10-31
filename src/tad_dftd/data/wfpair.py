# This file is part of tad-dftd.
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
Pairwise Gaussian weighting factors for D3S and D4S (from ``dftd/data/wfpair_*.f90``).
"""
from __future__ import annotations

from pathlib import Path

import torch
from tad_mctc.typing import Tensor, get_default_dtype

__all__ = ["WFPAIR_D3S", "WFPAIR_D4S", "get_wfpair_d3s", "get_wfpair_d4s"]

_DATA_DIR = Path(__file__).resolve().parent
_WFPAIR_D3S_CPU = torch.load(
    _DATA_DIR / "wfpair-d3s.pt", map_location="cpu", weights_only=True
)
_WFPAIR_D4S_CPU = torch.load(
    _DATA_DIR / "wfpair-d4s.pt", map_location="cpu", weights_only=True
)


def WFPAIR_D3S(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pairwise weighting factors ``(95, 95)`` for D3S (Z=0..94)."""
    dt = get_default_dtype() if dtype is None else dtype
    return _WFPAIR_D3S_CPU.to(device=device, dtype=dt)


def WFPAIR_D4S(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pairwise weighting factors ``(119, 119)`` for D4S (Z=0..118)."""
    dt = get_default_dtype() if dtype is None else dtype
    return _WFPAIR_D4S_CPU.to(device=device, dtype=dt)


def get_wfpair_d3s(
    numbers_i: Tensor,
    numbers_j: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up pairwise D3S weighting factors for pair atomic numbers."""
    dev = numbers_i.device if device is None else device
    table = WFPAIR_D3S(device=dev, dtype=dtype)
    zi = torch.clamp(numbers_i, min=0, max=table.shape[0] - 1)
    zj = torch.clamp(numbers_j, min=0, max=table.shape[1] - 1)
    out = table[zi, zj]
    in_bounds = (
        (numbers_i > 0)
        & (numbers_i < table.shape[0])
        & (numbers_j > 0)
        & (numbers_j < table.shape[1])
    )
    return torch.where(in_bounds, out, torch.zeros_like(out))


def get_wfpair_d4s(
    numbers_i: Tensor,
    numbers_j: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up pairwise D4S weighting factors for pair atomic numbers."""
    dev = numbers_i.device if device is None else device
    table = WFPAIR_D4S(device=dev, dtype=dtype)
    zi = torch.clamp(numbers_i, min=0, max=table.shape[0] - 1)
    zj = torch.clamp(numbers_j, min=0, max=table.shape[1] - 1)
    out = table[zi, zj]
    in_bounds = (
        (numbers_i > 0)
        & (numbers_i < table.shape[0])
        & (numbers_j > 0)
        & (numbers_j < table.shape[1])
    )
    return torch.where(in_bounds, out, torch.zeros_like(out))
