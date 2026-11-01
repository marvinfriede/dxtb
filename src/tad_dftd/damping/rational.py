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

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

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

__all__ = [
    "CallableTwoBodyDamping",
    "RationalThreeBody",
    "RationalTwoBody",
    "rational_damping",
]


class RationalTwoBody(TwoBodyDamping):
    """
    Rational (Becke-Johnson) damping of the two-body term. Needs `s8`, `a1`
    and `a2`; `s6` defaults to 1.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
    )
    label: ClassVar[str] = "Rational damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r0 = scaled_radius(self, param, pairs.rdamp, qq=pairs.qq)
        r6, r8 = pow6_pow8(pairs.distances)
        r0_6, r0_8 = pow6_pow8(r0)
        res = self.value(param, "s6") / (r6 + r0_6) + self.value(
            param, "s8"
        ) * pairs.qq / (r8 + r0_8)
        if param.s10 is not None:
            r2 = pairs.distances * pairs.distances
            r0_2 = r0 * r0
            res = res + param.s10 * (49.0 / 40.0) * (pairs.qq * pairs.qq) / (
                r8 * r2 + r0_8 * r0_2
            )
        return res


class CallableTwoBodyDamping(TwoBodyDamping):
    """
    Adapter wrapping a legacy ``damping_function(order, r, qq, param)``
    callable into a :class:`TwoBodyDamping`.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("s8", defaults.S8),
        ("a1", defaults.A1),
        ("a2", defaults.A2),
    )
    label: ClassVar[str] = "Custom callable two-body damping"

    def __init__(
        self,
        fn: Callable[..., Tensor],
        raw_param: Mapping[str, Any] | None = None,
        *,
        radius_type: str | None = None,
    ) -> None:
        self.fn = fn
        self.raw_param = raw_param
        self.radius_type = (
            radius_type
            if radius_type is not None
            else getattr(fn, "radius_type", "r4r2")
        )

    def _eval_order(
        self, order: int, distances: Tensor, radii: Tensor, p: Any
    ) -> Tensor:
        if hasattr(self.fn, "radius_type"):
            kw = p if isinstance(p, Mapping) else p.to_dict()
            return self.fn(distances, radii, order, **kw)
        return self.fn(order, distances, radii, p)

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        p = self.raw_param if self.raw_param is not None else param
        radii = pairs.rvdw if self.radius_type == "rvdw" else pairs.qq
        t6 = self._eval_order(6, pairs.distances, radii, p)
        t8 = self._eval_order(8, pairs.distances, radii, p)
        res = (
            self.value(param, "s6") * t6
            + self.value(param, "s8") * pairs.qq * t8
        )
        if param.s10 is not None:
            t10 = self._eval_order(10, pairs.distances, radii, p)
            res = res + param.s10 * (49.0 / 40.0) * (pairs.qq * pairs.qq) * t10
        return res


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
    param: Mapping[str, Any] | None = None,
    *,
    a1: Tensor | float | None = None,
    a2: Tensor | float | None = None,
) -> Tensor:
    """
    Rational (Becke-Johnson) damping function (from s-dftd3).

    Accepts damping parameters either via a `param` mapping or as keyword
    arguments `a1` and `a2`.
    """
    dd = {"device": distances.device, "dtype": distances.dtype}
    if param is not None:
        if a1 is None:
            a1 = param.get("a1", torch.tensor(defaults.A1, **dd))
        if a2 is None:
            a2 = param.get("a2", torch.tensor(defaults.A2, **dd))
    if a1 is None:
        a1 = torch.tensor(defaults.A1, **dd)
    if a2 is None:
        a2 = torch.tensor(defaults.A2, **dd)
    r0 = a1 * torch.sqrt(qq) + a2
    return 1.0 / (distances.pow(order) + r0.pow(order))
