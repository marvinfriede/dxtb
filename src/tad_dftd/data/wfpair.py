# This file is part of tad-dftd.
#
# SPDX-Identifier: LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
"""
Pairwise Gaussian weighting factors for D3S and D4S (from ``dftd/data/wfpair_*.f90``).
"""
from __future__ import annotations

from importlib.resources import as_file, files

import torch
from tad_mctc.typing import Tensor, get_default_dtype

__all__ = ["WFPAIR_D3S", "WFPAIR_D4S", "get_wfpair_d3s", "get_wfpair_d4s"]


def _load_pkg_tensor(filename: str) -> Tensor:
    res = files(__package__).joinpath(filename)
    with as_file(res) as path:
        return torch.load(path, map_location="cpu", weights_only=True)


_WFPAIR_D3S_CPU = _load_pkg_tensor("wfpair-d3s.pt")
_WFPAIR_D4S_CPU = _load_pkg_tensor("wfpair-d4s.pt")


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
