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
