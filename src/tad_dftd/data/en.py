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
from tad_mctc.data import PAULING
from tad_mctc.typing import Tensor, get_default_dtype

from .d1 import _lookup

__all__ = ["EN", "PAULING_EN", "get_electronegativity"]


def PAULING_EN(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pauling electronegativities for Z=0..118."""
    return PAULING(
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
    return _lookup(PAULING_EN, numbers, device, dtype)
