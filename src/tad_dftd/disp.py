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
"""
Dispersion energy evaluation
============================

High-level entrypoints for evaluating dispersion energies (:func:`dftd`,
:func:`dftd3`, :func:`dftd3s`, :func:`dispersion`, :func:`dispersion2`,
:func:`dispersion3`).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from tad_mctc import Structure
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.typing import CountingFunction, TableFunction, Tensor

from . import ncoord
from .cutoff import Cutoff
from .damping import (
    Damping,
    DampingParam,
    ThreeBodyDamping,
    TwoBodyDamping,
    ZeroThreeBodyD3,
    as_damping_param,
)
from .model.c6 import AtomicC6
from .model.catalogue import (
    DispersionModelId,
    get_model_metadata,
    resolve_model_id,
)
from .model.d3 import (
    D3Model,
    D3SModel,
    Environment,
    _eval_dispersion2,
    _eval_dispersion3,
    _matrix,
    default_damping,
)
from .model.weights import WeightingFunction, gaussian_log_weight
from .pairs import DensePairs, Pairs, TermPairs
from .reference.d3 import Reference

__all__ = [
    "D3Model",
    "D3SModel",
    "Environment",
    "Param",
    "default_damping",
    "dftd",
    "dftd3",
    "dftd3s",
    "dispersion",
    "dispersion2",
    "dispersion3",
]

Param: TypeAlias = DampingParam | Mapping[str, Any]


def dftd3(
    structure: Structure,
    param: Param,
    *,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    ref: Reference | None = None,
    rcov_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    counting_function: CountingFunction = ncoord.exp_count,
    weighting_function: WeightingFunction = gaussian_log_weight,
) -> Tensor:
    """
    Evaluate atom-resolved DFT-D3 dispersion energy for a structure or batch.
    """
    return D3Model(
        rcov_table=rcov_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        ref=ref,
        cutoff=cutoff,
        counting_function=counting_function,
        weighting_function=weighting_function,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )(structure, param, damping=damping, pairs=pairs)


def dftd3s(
    structure: Structure,
    param: Param,
    *,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    ref: Reference | None = None,
    rcov_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    wfpair_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    counting_function: CountingFunction = ncoord.exp_count,
) -> Tensor:
    """
    Evaluate atom-resolved DFT-D3S dispersion energy for a structure or batch.
    """
    return D3SModel(
        rcov_table=rcov_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        wfpair_table=wfpair_table,
        ref=ref,
        cutoff=cutoff,
        counting_function=counting_function,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )(structure, param, damping=damping, pairs=pairs)


def dftd(
    structure: Structure,
    param: Param,
    *,
    model: str | int | DispersionModelId = "d3",
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    cutoff: Cutoff | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    **kwargs: Any,
) -> Tensor:
    """
    Unified dispersion energy dispatcher across the DFT-D model catalogue.
    """
    model_id = resolve_model_id(model)
    cut = (
        get_model_metadata(model_id).default_cutoff
        if cutoff is None
        else cutoff
    )
    if model_id == DispersionModelId.D3:
        return dftd3(
            structure,
            param,
            damping=damping,
            pairs=pairs,
            cutoff=cut,
            max_triples=max_triples,
            checkpoint=checkpoint,
            **kwargs,
        )
    if model_id == DispersionModelId.D3S:
        return dftd3s(
            structure,
            param,
            damping=damping,
            pairs=pairs,
            cutoff=cut,
            max_triples=max_triples,
            checkpoint=checkpoint,
            **kwargs,
        )
    raise NotImplementedError(
        f"Dispersion model '{model}' is not yet wired into dftd()."
    )


def dispersion(
    structure: Structure,
    param: Param,
    c6: Tensor | AtomicC6,
    *,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Dispersion energy from given C6 coefficients: the two-body term, and the
    three-body term if `damping` has one.
    """
    d3 = D3Model(
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        cutoff=cutoff,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )
    if pairs is None:
        pairs = DensePairs()
    param, damping, pairs = d3._prepare(structure, param, damping, pairs)

    if isinstance(pairs, DensePairs):
        c6 = _matrix(c6)

    energy = d3.dispersion2(structure, param, c6, pairs.disp2, damping.two)
    if damping.three is not None:
        energy = energy + d3.dispersion3(
            structure, param, c6, pairs.disp3, damping.three
        )
    return energy


def dispersion2(
    structure: Structure,
    param: Param,
    c6: Tensor | AtomicC6,
    *,
    pairs: PeriodicShifts | NeighborList | None = None,
    damping: TwoBodyDamping | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    checkpoint: bool = False,
) -> Tensor:
    """
    Two-body dispersion energy from given C6 coefficients.
    """
    param = as_damping_param(param)
    if damping is None:
        damping = default_damping(param).two

    return _eval_dispersion2(
        structure,
        param,
        c6,
        pairs,
        damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        rdamp_table=rdamp_table,
        cutoff=cutoff,
        checkpoint=checkpoint,
    )


def dispersion3(
    structure: Structure,
    param: Param,
    c6: Tensor | AtomicC6,
    *,
    pairs: TermPairs = None,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Three-body (Axilrod-Teller-Muto) dispersion term from given C6 coefficients.
    """
    param = as_damping_param(param)
    return _eval_dispersion3(
        structure,
        param,
        c6,
        pairs,
        damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        rdamp_table=rdamp_table,
        cutoff=cutoff,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )
