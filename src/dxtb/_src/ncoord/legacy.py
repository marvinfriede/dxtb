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
Legacy coordination number functions (temporary shim)
=====================================================

tad-mctc 0.8 replaced the functional coordination number API
(``coordination_number(numbers, positions, ...)``, ``cn_d3``, ``cn_d3_gradient``,
...) by ``CNModel`` objects acting on a ``Structure``. dxtb and the not yet
released tad-dftd3/tad-dftd4 still use the old call signatures, so they are
vendored here from tad-mctc 0.7.0 (including the analytical derivatives).

This module is temporary: it goes away once dxtb moves to the ``CNModel``
interface.
"""

from __future__ import annotations

from math import pi, sqrt

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.data import en as eneg
from tad_mctc.data import radii
from tad_mctc.ncoord import defaults
from tad_mctc.ncoord.common import cut_coordination_number
from tad_mctc.ncoord.count import erf_count, exp_count, gfn2_count

from dxtb._src.typing import DD, Any, CountingFunction, Tensor

__all__ = [
    "coordination_number",
    "cn_d3",
    "cn_d4",
    "cn_gfn2",
    "cn_d3_gradient",
    "dexp_count",
    "derf_count",
    "dgfn2_count",
]


def coordination_number(
    numbers: Tensor,
    positions: Tensor,
    *,
    counting_function: CountingFunction,
    rcov: Tensor | None = None,
    cutoff: Tensor | float | int | None = None,
    cn_max: Tensor | float | int | None = None,
    pair_weight: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Generic coordination number evaluator for molecules (batched or not).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates (shape: ``(..., nat, 3)``).
    counting_function : CountingFunction
        Pair counting function.
    rcov : Tensor | None, optional
        Covalent radii of each atom (shape: ``(..., nat)``).
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff.
    cn_max : Tensor | float | int | None, optional
        Optional upper bound for the coordination number.
    pair_weight : Tensor | None, optional
        Optional per-pair weighting factor.
    kwargs : dict[str, Any]
        Additional keyword arguments for the counting function.

    Returns
    -------
    Tensor
        Coordination numbers (shape: ``(..., nat)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd)

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    mask = real_pairs(numbers, mask_diagonal=True)
    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    counts = counting_function(distances, rc, **kwargs)

    if pair_weight is not None:
        counts = pair_weight.to(**dd) * counts

    if cutoff is not None:
        cutoff_tensor = (
            cutoff.to(**dd)
            if isinstance(cutoff, torch.Tensor)
            else torch.tensor(cutoff, **dd)
        )
        valid = mask & (distances <= cutoff_tensor)
    else:
        valid = mask

    zero = torch.tensor(0.0, **dd)
    cn = torch.sum(torch.where(valid, counts, zero), dim=-1)

    if cn_max is None:
        return cn

    return cut_coordination_number(cn, cn_max)


def cn_d3(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = exp_count,
) -> Tensor:
    """D3 fractional coordination number."""
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    return coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=radii.COV_D3(**dd)[numbers],
        cutoff=torch.tensor(defaults.CUTOFF_D3, **dd),
    )


def cn_gfn2(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = gfn2_count,
) -> Tensor:
    """Double-exponential (GFN2-xTB) coordination number."""
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    return coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=radii.COV_D3(**dd)[numbers],
        cutoff=torch.tensor(defaults.CUTOFF_GFN2, **dd),
    )


def cn_d4(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = erf_count,
) -> Tensor:
    """D4 fractional coordination number."""
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    en = eneg.PAULING(**dd)[numbers]

    endiff = torch.abs(en.unsqueeze(-2) - en.unsqueeze(-1))
    weight = defaults.D4_K4 * torch.exp(
        -((endiff + defaults.D4_K5) ** 2.0) / defaults.D4_K6
    )

    return coordination_number(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=radii.COV_D3(**dd)[numbers],
        cutoff=torch.tensor(defaults.CUTOFF_D4, **dd),
        pair_weight=weight,
    )


def cn_d3_gradient(
    numbers: Tensor,
    positions: Tensor,
    *,
    dcounting_function: CountingFunction | None = None,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Derivative of the D3 coordination number with respect to the positions
    (shape: ``(..., nat, nat, 3)``).
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if dcounting_function is None:
        dcounting_function = dexp_count
    if cutoff is None:
        cutoff = torch.tensor(defaults.CUTOFF_D3, **dd)

    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd)

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(mask, storch.cdist(positions, positions, p=2), eps)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    dcf = torch.where(
        mask * (distances <= cutoff),
        dcounting_function(distances, rc, **kwargs),
        torch.tensor(0.0, **dd),
    )

    # (..., nat, nat, 3)
    rij = positions.unsqueeze(-3) - positions.unsqueeze(-2)
    return (dcf / distances).unsqueeze(-1) * rij


# analytical derivatives of the counting functions


def dexp_count(
    r: Tensor, r0: Tensor, kcn: Tensor | float | int = defaults.KCN_D3
) -> Tensor:
    """Derivative of the exponential counting function."""
    expterm = torch.exp(-kcn * (storch.safe_divide(r0, r) - 1.0))
    return (-kcn * r0 * expterm) / (r**2 * ((expterm + 1.0) ** 2))


def derf_count(
    r: Tensor, r0: Tensor, kcn: Tensor | float | int = defaults.KCN_D4
) -> Tensor:
    """Derivative of the error function counting function."""
    div = storch.safe_divide(-(kcn**2) * (r - r0) ** 2, r0**2)
    return -kcn / sqrt(pi) / r0 * torch.exp(div)


def dgfn2_count(
    r: Tensor,
    r0: Tensor,
    ka: Tensor | float | int = defaults.KA,
    kb: Tensor | float | int = defaults.KB,
    r_shift: Tensor | float | int = defaults.R_SHIFT,
) -> Tensor:
    """Derivative of the GFN2-xTB counting function."""
    cnt1 = exp_count(r, r0, ka)
    cnt2 = exp_count(r, r0 + r_shift, kb)
    return dexp_count(r, r0, ka) * cnt2 + cnt1 * dexp_count(r, r0 + r_shift, kb)
