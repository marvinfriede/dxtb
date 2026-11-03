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
Atomic :math:`\\sqrt{0.5 \\cdot \\langle r^4 \\rangle / \\langle r^2 \\rangle \\cdot \\sqrt{Z}}`
expectation values for elements Z=0..118 (from ``dftd/data/r4r2.f90``).
"""
from __future__ import annotations

import torch
from tad_mctc.typing import Tensor, get_default_dtype

__all__ = ["R4R2", "R4_OVER_R2", "SQRT_Z_R4_OVER_R2", "get_r4r2_val"]

_R4_OVER_R2_RAW: list[float] = [
    0.0,
    8.0589, 3.4698,
    29.0974, 14.8517, 11.8799, 7.8715, 5.5588, 4.7566, 3.8025, 3.1036,
    26.1552, 17.2304, 17.7210, 12.7442, 9.5361, 8.1652, 6.7463, 5.6004,
    29.2012, 22.3934,
    19.0598, 16.8590, 15.4023, 12.5589, 13.4788,
    12.2309, 11.2809, 10.5569, 10.1428, 9.4907,
    13.4606, 10.8544, 8.9386, 8.1350, 7.1251, 6.1971,
    30.0162, 24.4103,
    20.3537, 17.4780, 13.5528, 11.8451, 11.0355,
    10.1997, 9.5414, 9.0061, 8.6417, 8.9975,
    14.0834, 11.8333, 10.0179, 9.3844, 8.4110, 7.5152,
    32.7622, 27.5708,
    23.1671, 21.6003, 20.9615, 20.4562, 20.1010, 19.7475, 19.4828,
    15.6013, 19.2362, 17.4717, 17.8321, 17.4237, 17.1954, 17.1631,
    14.5716, 15.8758, 13.8989, 12.4834, 11.4421,
    10.2671, 8.3549, 7.8496, 7.3278, 7.4820,
    13.5124, 11.6554, 10.0959, 9.7340, 8.8584, 8.0125,
    29.8135, 26.3157,
    19.1885, 15.8542, 16.1305, 15.6161, 15.1226, 16.1576, 14.6510,
    14.7178, 13.9108, 13.5623, 13.2326, 12.9189, 12.6133, 12.3142,
    14.8326, 12.3771, 10.6378, 9.3638, 8.2297,
    7.5667, 6.9456, 6.3946, 5.9159, 5.4929,
    6.7286, 6.5144, 10.9169, 10.3600, 9.4723, 8.6641,
]


def R4_OVER_R2(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Raw ``<r4>/<r2>`` expectation values for Z=0..118."""
    return torch.tensor(
        _R4_OVER_R2_RAW,
        device=device,
        dtype=get_default_dtype() if dtype is None else dtype,
    )


def R4R2(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
     Scaled expectation values ``sqrt(0.5 * r4_over_r2 * sqrt(Z))`` for Z=0..118.
    """
    dt = get_default_dtype() if dtype is None else dtype
    r4r2_64 = torch.tensor(_R4_OVER_R2_RAW, device=device, dtype=torch.float64)
    z_64 = torch.arange(len(_R4_OVER_R2_RAW), device=device, dtype=torch.float64)
    val = torch.sqrt(0.5 * r4r2_64 * torch.sqrt(z_64))
    return val.to(dtype=dt)


SQRT_Z_R4_OVER_R2 = R4R2


def get_r4r2_val(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up ``sqrt_z_r4_over_r2`` for given atomic numbers."""
    dev = numbers.device if device is None else device
    table = R4R2(device=dev, dtype=dtype)
    return table[numbers]
