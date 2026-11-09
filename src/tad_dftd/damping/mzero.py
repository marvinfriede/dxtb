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
r"""
Modified zero damping
=====================

The zero damping with an offset :math:`\beta` on the distance
(Smith et al., J. Phys. Chem. Lett. 2016, 7, 2197) for the two-body term.
"""
from __future__ import annotations

from typing import ClassVar

from tad_mctc.typing import Tensor

from .. import defaults
from .base import DFTD_RADIUS_DEFAULTS, PairData, TwoBodyDamping, scaled_radius
from .param import DampingParam
from .zero import zero_like

__all__ = ["ModifiedZeroTwoBody"]


class ModifiedZeroTwoBody(TwoBodyDamping):
    """
    Modified zero damping of the two-body term. Needs `s8`, `rs6` and `bet`;
    `s6` defaults to 1.0, `rs8` to 1.0, `alp` to
    :data:`tad_dftd.defaults.ALP`, `a1` to 1.0 and `a2` to 0.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("rs8", defaults.RS8),
        ("alp", defaults.ALP),
        *DFTD_RADIUS_DEFAULTS,
    )
    label: ClassVar[str] = "Modified zero damping"
    reads_rvdw: ClassVar[bool] = True
    reads_znum: ClassVar[bool] = False

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        r0 = scaled_radius(self, param, pairs.rvdw)
        alp = self.value(param, "alp")
        offset = self.value(param, "bet") * r0
        t6 = (r / (self.value(param, "rs6") * r0) + offset) ** (-alp)
        t8 = (r / (self.value(param, "rs8") * r0) + offset) ** (-(alp + 2.0))
        return zero_like(self, pairs, param, t6, t8)
