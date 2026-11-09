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
from .param import DampingParam

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
    reads_rvdw: ClassVar[bool] = False
    reads_znum: ClassVar[bool] = False
    has_d8: ClassVar[bool] = False

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        rdamp = (
            pairs.rdamp if pairs.rdamp is not None else pairs.rvdw
        )
        r0 = self.value(param, "rs6") * scaled_radius(self, param, rdamp)
        alp = self.value(param, "alp")
        f6 = torch.sigmoid(alp * (r / r0 - 1.0))
        return self.value(param, "s6") * f6 / pow6(r)
