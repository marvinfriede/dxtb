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
Koide damping
=============

The spherical-wave expanded (Koide) damping with an optional
exchange-correlation screening from the overlap of Slater densities, ported
from dftd's ``dftd_damping_koide``.
"""
from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc.typing import Tensor

from .base import (
    DFTD_RADIUS_DEFAULTS,
    PairData,
    ThreeBodyDamping,
    TripleData,
    TwoBodyDamping,
    scaled_radius,
    scaled_radius3,
)
from .param import DampingParam

__all__ = ["KoideThreeBody", "KoideTwoBody"]


_P2 = (
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1699.0 / 207360.0,
    259.0 / 207360.0,
    197.0 / 1451520.0,
    19.0 / 2177280.0,
    1.0 / 4354560.0,
)
_P0 = (
    89.0 / 13824.0,
    89.0 / 13824.0,
    119.0 / 41472.0,
    5.0 / 6912.0,
    11.0 / 103680.0,
    1.0 / 124416.0,
    1.0 / 4354560.0,
)
_P3 = (
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    109.0 / 552960.0,
    13.0 / 552960.0,
    211.0 / 96768000.0,
    57.0 / 435456000.0,
    1.0 / 290304000.0,
)
_P1 = (
    47.0 / 552960.0,
    47.0 / 552960.0,
    7.0 / 184320.0,
    1.0 / 103680.0,
    209.0 / 145152000.0,
    11.0 / 96768000.0,
    1.0 / 290304000.0,
)


def _horner(x: Tensor, coefficients: tuple[float, ...]) -> Tensor:
    """``c0 + x * (c1 + x * (c2 + ...))``, nested as in dftd."""
    value = torch.full_like(x, coefficients[-1])
    for c in reversed(coefficients[:-1]):
        value = c + x * value
    return value


def _chi11(x: Tensor) -> Tensor:
    """Dipole-dipole attenuation function of the Koide expansion."""
    exp_x = torch.exp(-x)
    phi2 = 1.0 - exp_x * _horner(x, _P2)
    phi0 = exp_x * (x * x * x * _horner(x, _P0))
    return phi2 * phi2 + 0.5 * phi0 * phi0


def _chi12(x: Tensor) -> Tensor:
    """Dipole-quadrupole attenuation function of the Koide expansion."""
    exp_x = torch.exp(-x)
    phi3 = 1.0 - exp_x * _horner(x, _P3)
    phi1 = exp_x * (x * x * x * x * x * _horner(x, _P1))
    return phi3 * phi3 + (2.0 / 3.0) * phi1 * phi1


def _slater_overlap(r: Tensor, rslater: Tensor | float) -> Tensor:
    """Normalized overlap of two Slater densities of radius `rslater`."""
    y = r / rslater
    return torch.exp(-y) * (1.0 + y + y * y / 3.0)


def _screened(param: DampingParam) -> bool:
    """Whether the XC screening is evaluated, a static choice."""
    sxc = param.sxc
    return sxc is not None and (isinstance(sxc, Tensor) or sxc != 0.0)


class KoideTwoBody(TwoBodyDamping):
    """
    Koide damping of the two-body term, as dftd. Needs `s6`, `s8`, `rs6` and
    `rs8`; `a1` and `a2` default to 1 and 0, and `rsxc` to 1 for the
    screening by `sxc`. Reads the damping radius of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        *DFTD_RADIUS_DEFAULTS,
        ("rsxc", 1.0),
    )
    label: ClassVar[str] = "Koide damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        rpoly = scaled_radius(self, param, pairs.rdamp, qq=pairs.qq)

        r2 = r * r
        r6 = r2 * r2 * r2
        r8 = r6 * r2

        chi_11 = _chi11(self.value(param, "rs6") * r / rpoly)
        chi_12 = _chi12(self.value(param, "rs8") * r / rpoly)

        d6 = self.value(param, "s6") * chi_11 / r6
        d8 = self.value(param, "s8") * chi_12 / r8
        kernel = d6 + pairs.qq * d8

        if _screened(param):
            overlap = _slater_overlap(r, self.value(param, "rsxc") * rpoly)
            kernel = kernel * (1.0 - self.value(param, "sxc") * overlap)

        return kernel


class KoideThreeBody(ThreeBodyDamping):
    """
    Koide damping of the three-body term, as dftd. Needs `s9` and `rs9`; `a1`
    and `a2` default to 1 and 0, and `rsxc` to 1 for the screening by `sxc`.
    Reads the damping radii of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        *DFTD_RADIUS_DEFAULTS,
        ("rsxc", 1.0),
    )
    label: ClassVar[str] = "Koide damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rs9 = self.value(param, "rs9")
        screened = _screened(param)

        damp = torch.ones_like(triples.r)
        overlap = torch.ones_like(triples.r)
        positive = torch.ones_like(triples.r, dtype=torch.bool)
        for r2, rdamp in (
            (triples.r2ij, triples.rdampij),
            (triples.r2ik, triples.rdampik),
            (triples.r2jk, triples.rdampjk),
        ):
            r = torch.sqrt(r2)
            rpoly = scaled_radius3(self, param, rdamp)
            chi = _chi11(rs9 * r / rpoly)

            positive = positive & (chi > 0.0)
            damp = damp * torch.sqrt(torch.where(chi > 0.0, chi, 1.0))

            if screened:
                rsxc = self.value(param, "rsxc")
                overlap = overlap * _slater_overlap(r, rsxc * rpoly)

        d9 = self.value(param, "s9") * damp
        if screened:
            d9 = d9 * (1.0 - self.value(param, "sxc") * overlap)

        return torch.where(positive, d9, torch.zeros_like(d9))
