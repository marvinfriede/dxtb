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
Atomic van-der-Waals radii and :math:`C_6` coefficients for the DFT-D2 model
(from ``dftd/data/d2.f90`` and ``dftd/model/d2/reference.f90``).
"""
from __future__ import annotations

import torch
from tad_mctc.typing import Tensor, get_default_dtype

from .d1 import _AATOAU, _C6CONV

__all__ = [
    "D2_C6",
    "D2_RADII",
    "D2_RADII_PAIRWISE",
    "get_d2_c6",
    "get_d2_radius",
]

# Tabulated for Z=0..86 (padded to Z=0..118 with zeros)
_D2_RADIUS_AA: list[float] = [
    0.0,
    0.910, 0.920,
    0.750, 1.280, 1.350, 1.320, 1.270, 1.220, 1.170, 1.130,
    1.040, 1.240, 1.490, 1.560, 1.550, 1.530, 1.490, 1.450,
    1.350, 1.340,
    1.420, 1.420, 1.420, 1.420, 1.420, 1.420, 1.420, 1.420, 1.420, 1.420,
    1.500, 1.570, 1.600, 1.610, 1.590, 1.570,
    1.480, 1.460,
    1.490, 1.490, 1.490, 1.490, 1.490, 1.490, 1.490, 1.490, 1.490, 1.490,
    1.520, 1.640, 1.710, 1.720, 1.720, 1.710,
    1.638, 1.602, 1.564,
    1.594, 1.594, 1.594, 1.594, 1.594, 1.594, 1.594,
    1.594, 1.594, 1.594, 1.594, 1.594, 1.594, 1.594,
    1.625, 1.611, 1.611, 1.611, 1.611, 1.611, 1.611, 1.611,
    1.598, 1.805, 1.767, 1.725, 1.823, 1.810, 1.749,
] + [0.0] * 32

_D2_C6_JNM6: list[float] = [
    0.0,
    0.140, 0.080,
    1.610, 1.610, 3.130, 1.750, 1.230, 0.700, 0.750, 0.630,
    5.710, 5.710, 10.790, 9.230, 7.840, 5.570, 5.070, 4.610,
    10.800, 10.800,
    10.800, 10.800, 10.800, 10.800, 10.800, 10.800, 10.800, 10.800, 10.800, 10.800,
    16.990, 17.100, 16.370, 12.640, 12.470, 12.010,
    24.670, 24.670,
    24.670, 24.670, 24.670, 24.670, 24.670, 24.670, 24.670, 24.670, 24.670, 24.670,
    37.320, 38.710, 38.440, 31.740, 31.500, 29.990,
    315.275, 226.994, 176.252,
    140.680, 140.680, 140.680, 140.680, 140.680, 140.680, 140.680,
    140.680, 140.680, 140.680, 140.680, 140.680, 140.680, 140.680,
    105.112, 81.240, 81.240, 81.240, 81.240, 81.240, 81.240, 81.240,
    57.364, 57.254, 63.162, 63.540, 55.283, 57.171, 56.640,
] + [0.0] * 32


def D2_RADII(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Atomic van-der-Waals radii of D2 in Bohr for Z=0..118."""
    dt = get_default_dtype() if dtype is None else dtype
    rad_64 = torch.tensor(_D2_RADIUS_AA, device=device, dtype=torch.float64) * _AATOAU
    return rad_64.to(dtype=dt)


def D2_RADII_PAIRWISE(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pairwise D2 damping radii ``r_i + r_j`` in Bohr of shape ``(104, 104)``."""
    rad = D2_RADII(device=device, dtype=dtype)[:104]
    valid = (rad.unsqueeze(-1) > 0) & (rad.unsqueeze(-2) > 0)
    return torch.where(valid, rad.unsqueeze(-1) + rad.unsqueeze(-2), 0.0)


def D2_C6(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Atomic C6 coefficients of D2 in Hartree * Bohr^6 for Z=0..118."""
    dt = get_default_dtype() if dtype is None else dtype
    c6_64 = torch.tensor(_D2_C6_JNM6, device=device, dtype=torch.float64) * _C6CONV
    return c6_64.to(dtype=dt)


def get_d2_radius(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up D2 van-der-Waals radii in Bohr for given atomic numbers."""
    dev = numbers.device if device is None else device
    table = D2_RADII(device=dev, dtype=dtype)
    return table[numbers]


def get_d2_c6(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up D2 atomic C6 coefficients in Hartree * Bohr^6 for given atomic numbers."""
    dev = numbers.device if device is None else device
    table = D2_C6(device=dev, dtype=dtype)
    return table[numbers]
