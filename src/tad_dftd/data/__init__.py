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
Atomic and pairwise data tables for ``tad_dftd``.
"""
from __future__ import annotations

from .d1 import D1_C6, D1_RADII, D1_RADII_PAIRWISE, get_d1_c6, get_d1_radius
from .d2 import D2_C6, D2_RADII, D2_RADII_PAIRWISE, get_d2_c6, get_d2_radius
from .en import EN, PAULING_EN, get_electronegativity
from .hardness import GAM, HARDNESS, get_hardness
from .r4r2 import R4_OVER_R2, R4R2, SQRT_Z_R4_OVER_R2, get_r4r2_val
from .radii import (
    COV_D3,
    EEQBC_COV_RADII,
    VDW_D3,
    VDW_PAIRWISE,
    DampingRadii,
    get_damping_radii,
    get_damping_radius,
)
from .table import TABLES, element_table
from .tanh import TANH_PARAMS, get_tanh_params
from .wfpair import WFPAIR_D3S, WFPAIR_D4S, get_wfpair_d3s, get_wfpair_d4s
from .zeff import ZEFF, get_effective_charge

__all__ = [
    "COV_D3",
    "D1_C6",
    "D1_RADII",
    "D1_RADII_PAIRWISE",
    "D2_C6",
    "D2_RADII",
    "D2_RADII_PAIRWISE",
    "DampingRadii",
    "EEQBC_COV_RADII",
    "EN",
    "GAM",
    "HARDNESS",
    "PAULING_EN",
    "R4R2",
    "R4_OVER_R2",
    "SQRT_Z_R4_OVER_R2",
    "TABLES",
    "TANH_PARAMS",
    "VDW_D3",
    "VDW_PAIRWISE",
    "WFPAIR_D3S",
    "WFPAIR_D4S",
    "ZEFF",
    "element_table",
    "get_d1_c6",
    "get_d1_radius",
    "get_d2_c6",
    "get_d2_radius",
    "get_damping_radii",
    "get_damping_radius",
    "get_effective_charge",
    "get_electronegativity",
    "get_hardness",
    "get_r4r2_val",
    "get_tanh_params",
    "get_wfpair_d3s",
    "get_wfpair_d4s",
]
