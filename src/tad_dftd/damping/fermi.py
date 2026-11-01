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
Fermi damping
=============

The Fermi-type damping function used in DFT-D1 and DFT-D2 (from
``dftd/damping/fermi.f90``):

.. math::

    d_6(r) = \frac{s_6}{\left(1 + \exp\left(-\alpha (r / R_0 - 1)\right)\right) r^6},
    \quad R_0 = s_{r,6} (a_1 R_{\text{damp}} + a_2), \quad d_8(r) = 0.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc.typing import Tensor

from .. import defaults
from .base import (
    DFTD_RADIUS_DEFAULTS,
    PairData,
    TwoBodyDamping,
    pow6,
    scaled_radius,
)
from .param import DampingParam, ParamInput, as_damping_param

__all__ = ["FermiTwoBody"]


class FermiTwoBody(TwoBodyDamping):
    """
    Fermi-type damping of the two-body term (used in D1 and D2). Needs `rs6`
    and `alp`; `s6` defaults to 1.0, `a1` to 1.0 and `a2` to 0.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        *DFTD_RADIUS_DEFAULTS,
    )
    label: ClassVar[str] = "Fermi damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        rdamp = (
            pairs.rdamp if pairs.rdamp is not None else pairs.rvdw
        )
        r0 = self.value(param, "rs6") * scaled_radius(self, param, rdamp)
        alp = self.value(param, "alp")
        f6 = torch.sigmoid(alp * (r / r0 - 1.0))
        return self.value(param, "s6") * f6 / pow6(r)

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
