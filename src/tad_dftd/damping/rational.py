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
Rational (Becke-Johnson) damping
================================

The Becke-Johnson damping for the two-body term and the three-body
(Axilrod-Teller-Muto) term.
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
    pow6_pow8,
    scaled_radius,
    scaled_radius3,
)
from .param import DampingParam

__all__ = ["RationalThreeBody", "RationalTwoBody", "rational_damping"]


class RationalTwoBody(TwoBodyDamping):
    """
    Rational (Becke-Johnson) damping of the two-body term. Needs `s8`, `a1`
    and `a2`; `s6` defaults to 1.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (("s6", defaults.S6),)
    label: ClassVar[str] = "Rational damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r0 = scaled_radius(self, param, pairs.rdamp, qq=pairs.qq)
        r6, r8 = pow6_pow8(pairs.distances)
        r0_6, r0_8 = pow6_pow8(r0)
        return self.value(param, "s6") / (r6 + r0_6) + self.value(
            param, "s8"
        ) * pairs.qq / (r8 + r0_8)


class RationalThreeBody(ThreeBodyDamping):
    """
    Rational (Becke-Johnson) damping of the three-body term, as dftd. Needs
    `s9`; `a1` and `a2` default to 1 and 0. Reads the damping radii of the
    model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Rational damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        damp = torch.ones_like(triples.r)
        for r2, rdamp in (
            (triples.r2ij, triples.rdampij),
            (triples.r2ik, triples.rdampik),
            (triples.r2jk, triples.rdampjk),
        ):
            r3 = r2 * torch.sqrt(r2)
            rpoly = scaled_radius3(self, param, rdamp)
            rpoly3 = rpoly * rpoly * rpoly
            damp = damp * (r3 / (r3 + rpoly3))
        return self.value(param, "s9") * damp


def rational_damping(
    order: int,
    distances: Tensor,
    qq: Tensor,
    *,
    a1: Tensor | float = torch.tensor(defaults.A1),
    a2: Tensor | float = torch.tensor(defaults.A2),
) -> Tensor:
    """
    Rational (Becke-Johnson) damping function (from s-dftd3).

    Kept as a helper of the tests; :func:`tad_dftd.disp.dispersion2` calls
    :class:`RationalTwoBody`.
    """
    r0 = a1 * torch.sqrt(qq) + a2
    return 1.0 / (distances**order + r0**order)
