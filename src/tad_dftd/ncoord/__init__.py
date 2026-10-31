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
Coordination Numbers
====================

Coordination number counting functions and models for D3, D3S, D4, D4S, D4Srev,
D5, and EEQ/EEQBC charge models, built on :mod:`tad_mctc.ncoord`.
"""
from __future__ import annotations

from functools import partial

from tad_mctc.ncoord import (
    CNModel,
    cn_d3,
    cn_d4,
    cn_eeq,
    cn_eeq_en,
    cn_eeqbc,
    cn_eeqbc_en,
    cn_gfn2,
    cut_coordination_number,
    d4_en_weight,
    en_difference,
    erf_count,
    exp_count,
    gfn2_count,
)

from .. import data, defaults

__all__ = [
    "CNModel",
    "cn_d3",
    "cn_d4",
    "cn_d4srev",
    "cn_d5",
    "cn_eeq",
    "cn_eeq_en",
    "cn_eeqbc",
    "cn_eeqbc_en",
    "cn_gfn2",
    "cut_coordination_number",
    "d4_en_weight",
    "en_difference",
    "erf_count",
    "exp_count",
    "gfn2_count",
]

#: Coordination number for DFT-D4Srev (erf counting without EN weight, cutoff 30.0 Bohr)
cn_d4srev = CNModel(
    count=partial(erf_count, kcn=7.5),
    cutoff=defaults.D4_CN_CUTOFF,
    rcov=data.COV_D3,
)

#: Coordination number for DFT-D5 (exponential counting, cutoff 30.0 Bohr)
cn_d5 = CNModel(
    count=partial(exp_count, kcn=16.0),
    cutoff=defaults.D5_CN_CUTOFF,
    rcov=data.COV_D3,
)
