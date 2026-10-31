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
Pauling electronegativities for Z=0..118 (from ``dftd/data/en.f90``).
"""
from __future__ import annotations

import torch
from tad_mctc.typing import Tensor, get_default_dtype

__all__ = ["EN", "PAULING_EN", "get_electronegativity"]

_PAULING_EN_RAW: list[float] = [
    0.0,
    2.20, 3.00,
    0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 4.50,
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 3.50,
    0.82, 1.00,
    1.36, 1.54, 1.63, 1.66, 1.55,
    1.83, 1.88, 1.91, 1.90, 1.65,
    1.81, 2.01, 2.18, 2.55, 2.96, 3.00,
    0.82, 0.95,
    1.22, 1.33, 1.60, 2.16, 1.90,
    2.20, 2.28, 2.20, 1.93, 1.69,
    1.78, 1.96, 2.05, 2.10, 2.66, 2.60,
    0.79, 0.89,
    1.10, 1.12, 1.13, 1.14, 1.15, 1.17, 1.18,
    1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26,
    1.27, 1.30, 1.50, 2.36, 1.90,
    2.20, 2.20, 2.28, 2.54, 2.00,
    1.62, 2.33, 2.02, 2.00, 2.20, 2.20,
    0.79, 0.90,
    1.10, 1.30, 1.50, 1.38, 1.36, 1.28, 1.30,
    1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30,
    1.30,
    1.50, 1.50, 1.50, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50, 1.50,
]


def PAULING_EN(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pauling electronegativities for Z=0..118."""
    return torch.tensor(
        _PAULING_EN_RAW,
        device=device,
        dtype=get_default_dtype() if dtype is None else dtype,
    )


EN = PAULING_EN


def get_electronegativity(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Look up Pauling electronegativities for given atomic numbers."""
    dev = numbers.device if device is None else device
    table = PAULING_EN(device=dev, dtype=dtype)
    return table[numbers]
