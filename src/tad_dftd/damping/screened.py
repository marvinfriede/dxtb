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
Screened damping
================

The screened rational damping for the two- and three-body terms, ported from
dftd's ``dftd_damping_screened``.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc.typing import Tensor

from .. import defaults
from .base import (
    DFTD_RADIUS_DEFAULTS,
    PairData,
    ThreeBodyDamping,
    TripleData,
    TwoBodyDamping,
    _Damping,
    pow6_pow8,
    scaled_radius,
    scaled_radius3,
)
from .param import DampingParam

__all__ = ["ScreenedThreeBody", "ScreenedTwoBody"]


def _effective_radius(
    damping: _Damping,
    param: DampingParam,
    r: Tensor,
    rdamp: Tensor,
    *,
    three_body: bool = False,
) -> Tensor:
    """The critical radius switched on by the error function of `r`."""
    rpoly = (
        scaled_radius3(damping, param, rdamp)
        if three_body
        else scaled_radius(damping, param, rdamp)
    )
    a3 = damping.value(param, "a3")
    a4 = damping.value(param, "a4")
    screen = 0.5 * (1.0 + torch.erf(-a3 * (r - a4 * rdamp)))
    return rpoly * screen


class ScreenedTwoBody(TwoBodyDamping):
    """
    Screened damping of the two-body term, as dftd. Needs `s8`, `a3` and
    `a4`; `s6` defaults to 1, `a1` to 1 and `a2` to 0. Reads the damping
    radius of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        *DFTD_RADIUS_DEFAULTS,
    )
    label: ClassVar[str] = "Screened damping"
    reads_rvdw: ClassVar[bool] = False
    reads_znum: ClassVar[bool] = False

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        rdamp = (
            pairs.rdamp if pairs.rdamp is not None else torch.sqrt(pairs.qq)
        )
        reff = _effective_radius(self, param, r, rdamp)
        d6, d8 = pow6_pow8(r + reff)
        return (
            self.value(param, "s6") / d6
            + self.value(param, "s8") * pairs.qq / d8
        )


class ScreenedThreeBody(ThreeBodyDamping):
    """
    Screened damping of the three-body term, as dftd. Needs `s9`, `a3` and
    `a4`; `a1` and `a2` default to 1 and 0. Reads the damping radii of the
    model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Screened damping of the three-body term"
    reads_rdamp: ClassVar[bool] = True
    reads_rvdw: ClassVar[bool] = False

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        damp = torch.ones_like(triples.r)
        for r2, rdamp in triples.legs:
            r = torch.sqrt(r2)
            reff = _effective_radius(self, param, r, rdamp, three_body=True)
            ratio = r / (r + reff)
            damp = damp * (ratio * ratio * ratio)
        return self.value(param, "s9") * damp
