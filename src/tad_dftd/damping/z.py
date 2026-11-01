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
Z damping
=========

The atomic number damping of the XDM(Z) model for the two-body term.
"""
from __future__ import annotations

from typing import ClassVar

from tad_mctc.typing import Tensor

from .. import defaults
from .base import PairData, TwoBodyDamping, pow6_pow8
from .param import DampingParam

__all__ = ["ZTwoBody"]


class ZTwoBody(TwoBodyDamping):
    """
    Z damping of the two-body term. Needs `a1`; `s6` and `s8` default to 1.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("s8", defaults.S8),
    )
    label: ClassVar[str] = "Z damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        r0c6 = self.value(param, "a1") / pairs.znum * pairs.c6
        r6, r8 = pow6_pow8(r)
        return self.value(param, "s6") / (r6 + r0c6) + self.value(
            param, "s8"
        ) * pairs.qq / (r8 + r0c6 * pairs.qq)
