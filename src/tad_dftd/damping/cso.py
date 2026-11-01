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
from .param import DampingParam, ParamInput, as_damping_param

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

    def damp(
        self, param: ParamInput, r2: Tensor, rdamp: Tensor
    ) -> tuple[Tensor, Tensor]:
        p = as_damping_param(param)
        mask = r2 > 0
        r = torch.sqrt(torch.where(mask, r2, torch.ones_like(r2)))
        one = torch.ones_like(r)
        zero = torch.zeros_like(r)
        rdamp_safe = torch.where(rdamp > 0, rdamp, one)
        pairs = PairData(
            distances=r,
            qq=zero,
            c6=one,
            rvdw=rdamp_safe,
            znum=one,
            rdamp=rdamp_safe,
        )
        return torch.where(mask, self(pairs, p), zero), zero
