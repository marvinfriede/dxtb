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
Atomic van-der-Waals radii and :math:`C_6` coefficients for the DFT-D1 model
(from ``dftd/data/d1.f90`` and ``dftd/model/d1/reference.f90``).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
from tad_mctc.typing import Tensor, get_default_dtype

__all__ = [
    "D1_C6",
    "D1_RADII",
    "D1_RADII_PAIRWISE",
    "get_d1_c6",
    "get_d1_radius",
]

# CODATA 2018 conversion factors from mctc_io_convert.f90
_H = 6.62607015e-34
_C = 299792458.0
_NA = 6.02214076e23
_ALPHA = 7.2973525693e-3
_ME = 9.1093837015e-31
_HBAR = _H / (2.0 * math.pi)
_BOHR = _HBAR / (_ME * _C * _ALPHA)
_HARTREE = _ME * (_C**2) * (_ALPHA**2)
_AATOAU = 1.0 / (_BOHR * 1.0e10)
_KJTOAU = 1.0 / (_HARTREE * _NA * 1.0e-3)
_C6CONV = 1.0e-3 * _KJTOAU * ((10.0 * _AATOAU) ** 6)


def _lookup(
    table_fn: Callable[..., Tensor],
    numbers: Tensor,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> Tensor:
    """Entries of the table `table_fn` for the atomic `numbers`, on the
    device of `numbers` unless `device` is given."""
    dev = numbers.device if device is None else device
    return table_fn(device=dev, dtype=dtype)[numbers]


def _scaled_table(
    raw: list[float],
    factor: float,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> Tensor:
    """`raw * factor`, multiplied in double precision and then cast."""
    dt = get_default_dtype() if dtype is None else dtype
    return (torch.tensor(raw, device=device, dtype=torch.float64) * factor).to(
        dtype=dt
    )


def _pairwise_radii(raw_aa: list[float]) -> Tensor:
    """Pairwise damping radii ``r_i + r_j`` in Bohr, zero where either atom
    has no radius, in double precision on the CPU."""
    rad = torch.tensor(raw_aa, dtype=torch.float64) * _AATOAU
    valid = (rad.unsqueeze(-1) > 0) & (rad.unsqueeze(-2) > 0)
    return torch.where(
        valid, rad.unsqueeze(-1) + rad.unsqueeze(-2), torch.zeros_like(rad)
    )


def _cast(
    table: Tensor, device: torch.device | None, dtype: torch.dtype | None
) -> Tensor:
    """A precomputed CPU `table` on `device`, in `dtype` (default dtype if
    ``None``)."""
    dt = get_default_dtype() if dtype is None else dtype
    return table.to(device=device, dtype=dt)

# Tabulated for Z=0..10 (padded to Z=0..118 with zeros)
_D1_RADIUS_AA: list[float] = [
    0.0,
    1.110, 0.0,
    0.0, 0.0, 0.0, 1.610, 1.550, 1.490, 1.430, 1.380,
] + [0.0] * 108

_D1_C6_JNM6: list[float] = [
    0.0,
    0.160, 0.0,
    0.0, 0.0, 0.0, 1.650, 1.110, 0.700, 0.570, 0.450,
] + [0.0] * 108


def D1_RADII(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Atomic van-der-Waals radii of D1 in Bohr for Z=0..118."""
    return _scaled_table(_D1_RADIUS_AA, _AATOAU, device, dtype)


_D1_RADII_PAIRWISE_CPU: Tensor = _pairwise_radii(_D1_RADIUS_AA)


def D1_RADII_PAIRWISE(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pairwise D1 damping radii ``r_i + r_j`` in Bohr of shape ``(119, 119)``."""
    return _cast(_D1_RADII_PAIRWISE_CPU, device, dtype)


def D1_C6(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Atomic C6 coefficients of D1 in Hartree * Bohr^6 for Z=0..118."""
    return _scaled_table(_D1_C6_JNM6, _C6CONV, device, dtype)


def get_d1_radius(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up D1 van-der-Waals radii in Bohr for given atomic numbers."""
    return _lookup(D1_RADII, numbers, device, dtype)


def get_d1_c6(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up D1 atomic C6 coefficients in Hartree * Bohr^6 for given atomic numbers."""
    return _lookup(D1_C6, numbers, device, dtype)
