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

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        r0 = scaled_radius(self, param, pairs.rvdw)
        alp = self.value(param, "alp")
        offset = self.value(param, "bet") * r0
        t6 = (r / (self.value(param, "rs6") * r0) + offset) ** (-alp)
        t8 = (r / (self.value(param, "rs8") * r0) + offset) ** (-(alp + 2.0))
        return zero_like(self, pairs, param, t6, t8)
