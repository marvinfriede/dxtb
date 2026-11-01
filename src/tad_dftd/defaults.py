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
Default values and physical constants for the ``tad_dftd`` library.
"""
from __future__ import annotations

__all__ = [
    "A1",
    "A2",
    "ALP",
    "BET",
    "CSO_A2",
    "CSO_RS6",
    "CSO_RS8",
    "D1_ALP",
    "D1_DISP2_CUTOFF",
    "D1_RS6",
    "D2_ALP",
    "D2_DISP2_CUTOFF",
    "D2_RS6",
    "D3_ALP",
    "D3_CN_CUTOFF",
    "D3_DISP_CUTOFF",
    "D3_DISP2_CUTOFF",
    "D3_DISP2_WIDTH",
    "D3_DISP3_CUTOFF",
    "D3_DISP3_WIDTH",
    "D3_K1",
    "D3_K3",
    "D3_KCN",
    "D3_RS6",
    "D3_RS8",
    "D3_RS9",
    "D3_S6",
    "D3_S8",
    "D3_S9",
    "D3_S10",
    "D4_ALP",
    "D4_CN_CUTOFF",
    "D4_CN_EEQ_CUTOFF",
    "D4_CN_EEQ_LMAX",
    "D4_DISP2_CUTOFF",
    "D4_DISP3_CUTOFF",
    "D4_GA",
    "D4_GC",
    "D4_K1",
    "D4_K3",
    "D4_K4",
    "D4_K5",
    "D4_K6",
    "D4_RS6",
    "D4_RS8",
    "D4_RS9",
    "D4_S6",
    "D4_S8",
    "D4_S9",
    "D4_S10",
    "D4_WF",
    "DISP2_CUTOFF_WIDTH",
    "DISP3_CUTOFF_WIDTH",
    "MAX_ELEM",
    "MAX_ELEMENT",
    "MAX_ELEM_D1",
    "MAX_ELEM_D2",
    "MAX_ELEM_D3",
    "MAX_ELEM_D4",
    "RATIONAL_A1",
    "RATIONAL_A2",
    "RS6",
    "RS8",
    "RS9",
    "S6",
    "S8",
    "S9",
    "S10",
]

# Maximum element numbers supported by each model
MAX_ELEM = 118
MAX_ELEMENT = 104
MAX_ELEM_D1 = 10
MAX_ELEM_D2 = 86
MAX_ELEM_D3 = 103
MAX_ELEM_D4 = 118

# Cutoffs in Bohr (matching Grimme-lab dftd cutoff.f90)
D1_DISP2_CUTOFF = 90.0
D2_DISP2_CUTOFF = 90.0

D3_DISP2_CUTOFF = 60.0
D3_DISP3_CUTOFF = 40.0
D3_DISP_CUTOFF = 50.0
D3_CN_CUTOFF = 40.0
D3_DISP2_WIDTH = 0.0
D3_DISP3_WIDTH = 0.0

DISP2_CUTOFF_WIDTH = 0.0
DISP3_CUTOFF_WIDTH = 0.0

D4_DISP2_CUTOFF = 60.0
D4_DISP3_CUTOFF = 40.0
D4_CN_CUTOFF = 30.0
D4_CN_EEQ_CUTOFF = 25.0
D4_CN_EEQ_LMAX = 8.0

# D1 & D2 model defaults
D1_RS6 = 1.0
D1_ALP = 23.0
D2_RS6 = 1.1
D2_ALP = 20.0

# D3 model defaults
D3_K1 = 16.0
D3_K3 = -4.0
D3_KCN = 16.0

D3_S6 = 1.0
D3_S8 = 1.0
D3_S9 = 1.0
D3_S10 = 0.0
D3_RS6 = 1.0
D3_RS8 = 1.0
D3_RS9 = 4.0 / 3.0
D3_ALP = 14.0

# D4 model defaults
D4_K1 = 16.0
D4_K3 = -4.0
D4_K4 = 4.10451
D4_K5 = 19.08857
D4_K6 = 2 * 11.28174**2  # 254.5553140552

D4_GA = 3.0
D4_GC = 2.0
D4_WF = 6.0

D4_S6 = 1.0
D4_S8 = 1.0
D4_S9 = 1.0
D4_S10 = 0.0
D4_RS6 = 1.0
D4_RS8 = 1.0
D4_RS9 = 1.0
D4_ALP = 16.0

# CSO damping defaults
CSO_A2 = 2.5
CSO_RS6 = 0.0
CSO_RS8 = 6.25

# Damping parameter defaults (compatibility aliases)
S6 = 1.0
S8 = 1.0
S9 = 1.0
S10 = 0.0
A1 = 0.4
A2 = 5.0
RATIONAL_A1 = 0.4
RATIONAL_A2 = 5.0
ALP = 14.0
RS6 = 1.0
RS8 = 1.0
RS9 = 4.0 / 3.0
BET = 0.0
