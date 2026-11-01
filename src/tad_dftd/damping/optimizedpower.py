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
r"""
Optimized power damping
=======================

The rational damping with an additional zero-damping like power
:math:`\beta` (Witte et al., J. Chem. Theory Comput. 2017, 13, 2043) for the
two-body term.
"""
from __future__ import annotations

from typing import ClassVar

from tad_mctc.typing import Tensor

from .. import defaults
from .base import PairData, TwoBodyDamping, pow6_pow8, scaled_radius
from .param import DampingParam

__all__ = ["OptimizedPowerTwoBody"]


class OptimizedPowerTwoBody(TwoBodyDamping):
    """
    Optimized power damping of the two-body term. Needs `s8`, `a1`, `a2` and
    `bet`; `s6` defaults to 1.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (("s6", defaults.S6),)
    label: ClassVar[str] = "Optimized power damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        bet = self.value(param, "bet")
        r0 = scaled_radius(self, param, pairs.rdamp, qq=pairs.qq)
        rb = r**bet
        ab = r0**bet
        r6, r8 = pow6_pow8(r)
        r0_6, r0_8 = pow6_pow8(r0)
        return self.value(param, "s6") * rb / (
            rb * r6 + ab * r0_6
        ) + self.value(param, "s8") * pairs.qq * rb / (rb * r8 + ab * r0_8)
