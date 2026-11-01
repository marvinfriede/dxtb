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
Zero damping
============

The Chai-Head-Gordon damping for the two-body term, for the three-body
(Axilrod-Teller-Muto) term on the product of the three distances (in the D3
and D4 forms), and as a product over the three pairs.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc import storch
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

__all__ = [
    "ZeroProductThreeBody",
    "ZeroThreeBodyD3",
    "ZeroThreeBodyD4",
    "ZeroTwoBody",
    "averaged_zero_damping",
    "zero_damping",
    "zero_like",
]


def zero_like(
    damping: TwoBodyDamping,
    pairs: PairData,
    param: DampingParam,
    t6: Tensor,
    t8: Tensor,
) -> Tensor:
    r"""
    Two-body term for any damping of the form :math:`1 / (1 + 6 t_n)`:
    zero damping and modified zero damping.
    """
    r6, r8 = pow6_pow8(pairs.distances)
    d6 = 1.0 / (1.0 + 6.0 * t6)
    d8 = 1.0 / (1.0 + 6.0 * t8)
    return (
        damping.value(param, "s6") * d6 / r6
        + damping.value(param, "s8") * pairs.qq * d8 / r8
    )


class ZeroTwoBody(TwoBodyDamping):
    """
    Zero damping of the two-body term. Needs `s8` and `rs6`; `s6` defaults to
    1.0, `rs8` to 1.0, `alp` to :data:`tad_dftd.defaults.ALP`, `a1` to 1.0
    and `a2` to 0.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("rs8", defaults.RS8),
        ("alp", defaults.ALP),
        *DFTD_RADIUS_DEFAULTS,
    )
    label: ClassVar[str] = "Zero damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        r0 = scaled_radius(self, param, pairs.rvdw)
        alp = self.value(param, "alp")
        t6 = (self.value(param, "rs6") * r0 / r) ** alp
        t8 = (self.value(param, "rs8") * r0 / r) ** (alp + 2.0)
        return zero_like(self, pairs, param, t6, t8)


def averaged_zero_damping(
    r0: Tensor, r: Tensor, s9: Tensor | float, exponent: Tensor | float
) -> Tensor:
    r"""
    :math:`s_9 / \left(1 + 6 (R_0 / R)^\text{exponent}\right)`, the zero
    damping on the product :math:`R` of the three distances and the product
    :math:`R_0` of the three scaled radii, shared by :class:`ZeroThreeBodyD3`
    and :class:`ZeroThreeBodyD4`.
    """
    return s9 / (1.0 + 6.0 * (r0 / r) ** exponent)


class ZeroThreeBodyD3(ThreeBodyDamping):
    """
    Zero damping of the three-body term of D3 (and s-dftd3). Needs `s9`;
    `rs9` defaults to ``4/3`` and `alp` to :data:`tad_dftd.defaults.ALP`.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("rs9", defaults.D3_RS9),
        ("alp", defaults.ALP),
    )
    label: ClassVar[str] = "Zero damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rs9 = self.value(param, "rs9")
        alp3 = (
            param.alp3
            if param.alp3 is not None
            else self.value(param, "alp") + 2.0
        )
        r0 = (
            (rs9 * triples.rvdwij)
            * (rs9 * triples.rvdwik)
            * (rs9 * triples.rvdwjk)
        )
        return averaged_zero_damping(
            r0,
            triples.r,
            self.value(param, "s9"),
            alp3 / 3.0,
        )


class ZeroThreeBodyD4(ThreeBodyDamping):
    """
    Zero damping of the three-body term of D4, as in dftd's ``zero_avg``:
    the averaged zero damping on the damping radii of the model, with
    exponent ``alp / 3``. Needs `s9`; `alp` defaults to ``16.0``, `rs9` to
    ``1.0``, `a1` to ``1.0`` and `a2` to ``0.0``.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        *DFTD_RADIUS_DEFAULTS,
        ("rs9", defaults.D4_RS9),
        ("alp", defaults.D4_ALP),
    )
    label: ClassVar[str] = "D4 zero damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rs9 = self.value(param, "rs9")
        alp = param.alp3 if param.alp3 is not None else self.value(param, "alp")
        rpoly = (
            scaled_radius3(self, param, triples.rdampij)
            * scaled_radius3(self, param, triples.rdampik)
            * scaled_radius3(self, param, triples.rdampjk)
        )
        return averaged_zero_damping(
            (rs9 * rs9 * rs9) * rpoly,
            triples.r,
            self.value(param, "s9"),
            alp / 3.0,
        )


class ZeroProductThreeBody(ThreeBodyDamping):
    """
    Product-of-pairs zero damping of the three-body term, as dftd's
    ``zero_damping_threebody``. Needs `s9`, `rs9` and `alp` (or `alp3`); `a1`
    and `a2` default to 1 and 0. Reads the damping radii of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Product zero damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rs9 = self.value(param, "rs9")
        alp = param.alp3 if param.alp3 is not None else self.value(param, "alp")
        damp = torch.ones_like(triples.r)
        for r2, rdamp in (
            (triples.r2ij, triples.rdampij),
            (triples.r2ik, triples.rdampik),
            (triples.r2jk, triples.rdampjk),
        ):
            rpoly = scaled_radius3(self, param, rdamp)
            t3 = (rs9 * rpoly / torch.sqrt(r2)) ** alp
            damp = damp * (1.0 / (1.0 + 6.0 * t3))
        return self.value(param, "s9") * damp


def zero_damping(
    order: int,
    distances: Tensor,
    r0: Tensor,
    *,
    rsn: Tensor | float = torch.tensor(defaults.RS6),
    alp: Tensor | float = torch.tensor(defaults.ALP),
) -> Tensor:
    """
    Zero-damping function (from s-dftd3).

    Kept as a helper of the tests; :func:`tad_dftd.disp.dispersion2` calls
    :class:`ZeroTwoBody`.
    """
    alpn = alp + order - 6
    return 1.0 / (1.0 + 6.0 * (rsn * storch.safe_divide(r0, distances)) ** alpn)
