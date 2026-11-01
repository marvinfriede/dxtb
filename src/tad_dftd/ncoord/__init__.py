# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
"""
Coordination Numbers
====================

Coordination number counting functions and models for D3, D3S, D4, D4S, D4Srev,
and EEQ charge models, built on :mod:`tad_mctc.ncoord`.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.data import radii
from tad_mctc.ncoord import (
    CNModel,
    cn_d3,
    cn_d4,
    cn_eeq,
    cn_eeq_en,
    cn_gfn2,
    cut_coordination_number,
    d4_en_weight,
    en_difference,
    erf_count,
    exp_count,
    gfn2_count,
)
from tad_mctc.typing import DD, CountingFunction, Tensor

from .. import data, defaults

__all__ = [
    "CNModel",
    "cn_d3",
    "cn_d4",
    "cn_d4srev",
    "cn_eeq",
    "cn_eeq_en",
    "cn_gfn2",
    "coordination_number",
    "cut_coordination_number",
    "d4_en_weight",
    "en_difference",
    "erf_count",
    "exp_count",
    "gfn2_count",
]

#: Coordination number for DFT-D4Srev (erf counting without EN weight, cutoff 30.0 Bohr)
cn_d4srev = CNModel(
    count=partial(erf_count, kcn=7.5),
    cutoff=defaults.D4_CN_CUTOFF,
    rcov=data.COV_D3,
)


def coordination_number(
    numbers: Tensor,
    positions: Tensor,
    *,
    counting_function: CountingFunction = exp_count,
    rcov: Tensor | None = None,
    cutoff: Tensor | float | int | None = None,
    cn_max: Tensor | float | int | None = None,
    pair_weight: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Evaluate fractional coordination numbers from per-atom covalent radii
    `rcov` (shape ``(..., nat)``) and `positions` (shape ``(..., nat, 3)``).
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
