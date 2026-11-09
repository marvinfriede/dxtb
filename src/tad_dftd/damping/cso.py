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
CSO damping
===========

The C6-scaled only damping (Schröder et al., J. Chem. Theory Comput. 2015,
11, 3163) for the two-body term.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc.typing import Tensor

from .. import defaults
from .base import PairData, TwoBodyDamping, pow6
from .param import DampingParam

__all__ = ["CSOTwoBody"]


class CSOTwoBody(TwoBodyDamping):
    """
    CSO damping of the two-body term. Needs `a1`; `s6` defaults to 1.0, `a2`
    to 2.5, `rs6` (``a3``) to 0.0 and `rs8` (``a4``) to 6.25. `s8` is not
    used.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("a2", defaults.CSO_A2),
        ("rs6", defaults.CSO_RS6),
        ("rs8", defaults.CSO_RS8),
    )
    label: ClassVar[str] = "CSO damping"
    reads_rvdw: ClassVar[bool] = False
    reads_znum: ClassVar[bool] = False
    has_d8: ClassVar[bool] = False

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        r0 = (
            pairs.rdamp if pairs.rdamp is not None else torch.sqrt(pairs.qq)
        )

        if param.a3 is not None and param.rs6 is None and param.rs8 is None:
            # Grimme-lab dftd convention: a3 is sigmoid amplitude, a4 is
            # sigmoid radius scale (default 2.5), a2 is denominator shift
            # (default 6.25).
            amp: Tensor | float = param.a3
            rad_scale: Tensor | float = (
                param.a4 if param.a4 is not None else defaults.CSO_A2
            )
            d6_val: Tensor | float = (
                param.a2 if param.a2 is not None else defaults.CSO_RS8
            )
            d6 = torch.as_tensor(d6_val, device=r.device, dtype=r.dtype)
        else:
            # s-dftd3 / tad-dftd3 convention: a1 is sigmoid amplitude, a2 is
            # sigmoid radius scale, rs6 is denominator radius scale, rs8 is
            # denominator constant shift.
            amp = self.value(param, "a1")
            rad_scale = self.value(param, "a2")
            d6 = self.value(param, "rs6") * r0 + self.value(param, "rs8")

        # 1 / (1 + exp(r - rad_scale * r0)), without exponential overflow
        sigmoid = torch.sigmoid(rad_scale * r0 - r)
        scale = self.value(param, "s6") + amp * sigmoid
        return scale / (pow6(r) + pow6(d6))
