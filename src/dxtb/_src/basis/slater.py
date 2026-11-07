# This file is part of dxtb.
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
"""
Basis: Slater Expansion
=======================

Expansion coefficients for Slater functions into primitive Gaussian functions
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from dxtb._src.typing import Literal, Tensor
from dxtb._src.typing.exceptions import (
    CGTOAzimuthalQuantumNumberError,
    CGTOPrimitivesError,
    CGTOPrincipalQuantumNumberError,
    CGTOQuantumNumberError,
    CGTOSlaterExponentsError,
)

__all__ = ["slater_to_gauss"]

base = Path(__file__).parent / "sto-ng"
sto_ng = [
    torch.from_numpy(np.load(base / f"sto-{n}g.npy")).type(torch.double)
    for n in range(1, 7)
]


# Two over pi
top = 2.0 / math.pi

dfactorial = torch.tensor(
    [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]
)
"""
Double factorial up to 7!! for normalization of the Gaussian basis functions.

See `OEIS A001147 <https://oeis.org/A001147>`__.
"""

MAX_PRINCIPAL = 6
"""Maximum principal quantum number."""

MAX_AZIMUTHAL = 4
"""Maximum azimuthal quantum number."""

# GFN1-xTB 4sp and 5sp expansions with shared primitive exponents (tblite #423)
# 4sp: W. J. Pietro et al., Inorg. Chem. 19, 2225-2229 (1980). DOI: 10.1021/ic50210a005
# 5sp: W. J. Pietro et al., Inorg. Chem. 20, 3650-3654 (1981). DOI: 10.1021/ic50225a013
pAlpha6sp: dict[int, Tensor] = {
    4: torch.tensor(
        [
            1.365346e00,
            4.393213e-01,
            1.877069e-01,
            9.360270e-02,
            5.052263e-02,
            2.809354e-02,
        ],
        dtype=torch.double,
    ),
    5: torch.tensor(
        [
            7.701420258e-01,
            2.756268915e-01,
            1.301847480e-01,
            6.953441940e-02,
            4.002545502e-02,
            2.348388309e-02,
        ],
        dtype=torch.double,
    ),
}

pCoeff6sp_s: dict[int, Tensor] = {
    4: torch.tensor(
        [
            3.775056e-03,
            -5.585965e-02,
            -3.192946e-01,
            -2.764780e-02,
            9.049199e-01,
            3.406258e-01,
        ],
        dtype=torch.double,
    ),
    5: torch.tensor(
        [
            1.267447151e-02,
            3.266734789e-03,
            -4.307553999e-01,
            -3.231998963e-01,
            1.104322879e00,
            4.368498703e-01,
        ],
        dtype=torch.double,
    ),
}

pCoeff6sp_p: dict[int, Tensor] = {
    4: torch.tensor(
        [
            -7.052075e-03,
            -5.259505e-02,
            -3.773450e-02,
            3.874773e-01,
            5.791672e-01,
            1.221817e-01,
        ],
        dtype=torch.double,
    ),
    5: torch.tensor(
        [
            -1.105673292e-3,
            -6.243132446e-2,
            -1.628476766e-1,
            3.210328714e-01,
            6.964579592e-01,
            1.493146125e-01,
        ],
        dtype=torch.double,
    ),
}


def slater_to_gauss(
    ng: Tensor | int,
    n: Tensor | int,
    l: Tensor | int,
    zeta: Tensor,
    norm: bool = True,
    *,
    sto: Literal["gfn1", "gfn2"] = "gfn2",
) -> tuple[Tensor, Tensor]:
    """
    Expand Slater function in primitive gaussian functions.

    Parameters
    ----------
    ng : int
        Number of Gaussian functions for the expansion.
    n : int
        Principal quantum number of shell.
    l : int
        Azimuthal quantum number of shell.
    zeta : Tensor
        Exponent of Slater function to expand.
    norm : bool, optional
        Include normalization in contraction coefficients.
        Defaults to ``True``.
    sto : Literal["gfn1", "gfn2"], optional
        STO-NG expansion set. Defaults to ``"gfn2"``.

    Returns
    -------
    (Tensor, Tensor):
        Contraction coefficients of primitive gaussians, can contain
        normalization, and exponents of primitive gaussian functions.
    """
    if sto not in ("gfn1", "gfn2"):
        raise ValueError(
            f"Unknown STO-NG expansion '{sto}'. Expected 'gfn1' or 'gfn2'."
        )

    if not 1 <= ng <= 6:
        raise CGTOPrimitivesError()
    if zeta <= 0:
        raise CGTOSlaterExponentsError()
    if l > MAX_AZIMUTHAL:
        raise CGTOAzimuthalQuantumNumberError(MAX_AZIMUTHAL)
    if n > MAX_PRINCIPAL:
        raise CGTOPrincipalQuantumNumberError(MAX_PRINCIPAL)
    if n <= l:  # l ∊ [n-1, n-2, ..., 1, 0]
        raise CGTOQuantumNumberError()

    int_ng = int(ng)
    int_n = int(n)
    int_l = int(l)

    if sto == "gfn1" and int_ng == 6 and int_n in (4, 5) and int_l in (0, 1):
        _alpha = pAlpha6sp[int_n]
        _coeff = pCoeff6sp_s[int_n] if int_l == 0 else pCoeff6sp_p[int_n]
    else:
        # we have to use a little hack here,
        # if you pass n and l correctly, everything is fine
        # ityp: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
        #    n: 1 2 3 4 5 2 3 4 5  3  4  5  4  5  5  6  6
        #    l: 0 0 0 0 0 1 1 1 1  2  2  2  3  3  4  0  1
        itype: Tensor | int = (
            n + torch.tensor([0, 4, 7, 9, 10], device=zeta.device)[l] - 1
        )
        if n == 6 and ng == 6:
            itype = 15 + l

        _coeff, _alpha = sto_ng[int_ng - 1][:, itype, :]

    alpha = _alpha.type(zeta.dtype).to(zeta.device) * zeta.unsqueeze(-1) ** 2
    coeff = _coeff.type(zeta.dtype).to(zeta.device)

    # normalize the gaussian if requested
    # <φ|φ> = (2i-1)!!(2j-1)!!(2k-1)!!/(4α)^(i+j+k) · sqrt(π/2α)³
    # N² = (4α)^(i+j+k)/((2i-1)!!(2j-1)!!(2k-1)!!)  · sqrt(2α/π)³
    # N = (4α)^((i+j+k)/2) / sqrt((2i-1)!!(2j-1)!!(2k-1)!!) · (2α/π)^(3/4)
    if norm is True:
        coeff = coeff * (
            (top * alpha) ** 0.75
            * torch.sqrt(4 * alpha) ** l
            / torch.sqrt(dfactorial[l])
        )

        # normalize the contracted shell (the primitives alone leave cross
        # terms, i.e., deviations of up to ~1e-8 from unit self-overlap)
        # S = sum_kl c_k c_l (π/(α_k+α_l))^(3/2) (2l-1)!! / (2(α_k+α_l))^l
        p = alpha.unsqueeze(-1) + alpha.unsqueeze(-2)
        pref = (math.pi / p) ** 1.5 * dfactorial[l] / (2 * p) ** l
        self_overlap = (coeff.unsqueeze(-1) * coeff.unsqueeze(-2) * pref).sum(
            (-2, -1)
        )
        coeff = coeff / torch.sqrt(self_overlap)

    return (alpha, coeff)
