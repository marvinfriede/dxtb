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
:func:`dftd1`, :func:`dftd2`, :func:`dftd3`, :func:`dftd3s`, :func:`dftd4`,
:func:`dftd4s`, :func:`dftd4srev`, :func:`get_properties`, :func:`dispersion`,
:func:`dispersion2`, :func:`dispersion3`).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from tad_mctc import Structure
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.typing import CountingFunction, TableFunction, Tensor

from . import defaults, ncoord
from .cutoff import Cutoff
from .damping import (
    Damping,
    DampingParam,
    ThreeBodyDamping,
    TwoBodyDamping,
    ZeroThreeBodyD3,
    as_damping_param,
)
from .model.base import DispersionProperties
from .model.c6 import AtomicC6
from .model.catalogue import (
    DispersionModelId,
    get_model_metadata,
    resolve_model_id,
)
from .model.d1 import D1Model, _default_d1_cutoff
from .model.d2 import D2Model, _default_d2_cutoff
from .model.d3 import (
    D3Model,
    D3SModel,
    Environment,
    _eval_dispersion2,
    _eval_dispersion3,
    _matrix,
    default_damping,
)
from .model.d4 import D4Model, D4SModel, D4SrevModel, _default_d4_cutoff
from .model.weights import WeightingFunction, gaussian_log_weight
from .pairs import DensePairs, Pairs, TermPairs
from .reference.d3 import Reference
from .reference.d4 import D4Reference

__all__ = [
    "D1Model",
    "D2Model",
    "D3Model",
    "D3SModel",
    "D4Model",
    "D4SModel",
    "D4SrevModel",
    "Environment",
    "Param",
    "default_damping",
    "dftd",
    "dftd1",
    "dftd2",
    "dftd3",
    "dftd3s",
    "dftd4",
    "dftd4s",
    "dftd4srev",
    "dispersion",
    "dispersion2",
    "dispersion3",
    "get_properties",
]

Param: TypeAlias = DampingParam | Mapping[str, Any]


def dftd1(
    structure: Structure,
    param: Param,
    *,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    c6_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff | None = None,
) -> Tensor:
    """
    Evaluate atom-resolved DFT-D1 dispersion energy for a structure or batch.
    """
    cut = _default_d1_cutoff() if cutoff is None else cutoff
    return D1Model(
        c6_table=c6_table,
        rdamp_table=rdamp_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        cutoff=cut,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )(structure, param, damping=damping, pairs=pairs)


def dftd2(
    structure: Structure,
    param: Param,
    *,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    c6_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff | None = None,
) -> Tensor:
    """
    Evaluate atom-resolved DFT-D2 dispersion energy for a structure or batch.
    """
    cut = _default_d2_cutoff() if cutoff is None else cutoff
    return D2Model(
        c6_table=c6_table,
        rdamp_table=rdamp_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        cutoff=cut,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )(structure, param, damping=damping, pairs=pairs)


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


def dftd4(
    structure: Structure,
    param: Param,
    *,
    q: Tensor | None = None,
    qmodel: str = "eeq",
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    wf: Tensor | float = defaults.D4_WF,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    ref: D4Reference | None = None,
    rcov_table: Tensor | TableFunction | None = None,
    en_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    zeff_table: Tensor | TableFunction | None = None,
    gam_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff | None = None,
    counting_function: CountingFunction = ncoord.erf_count,
) -> Tensor:
    """
    Evaluate atom-resolved DFT-D4 dispersion energy for a structure or batch.
    """
    cut = _default_d4_cutoff() if cutoff is None else cutoff
    return D4Model(
        ga=ga,
        gc=gc,
        wf=wf,
        qmodel=qmodel,
        rcov_table=rcov_table,
        en_table=en_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        zeff_table=zeff_table,
        gam_table=gam_table,
        ref=ref,
        cutoff=cut,
        counting_function=counting_function,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )(structure, param, q=q, damping=damping, pairs=pairs)


def dftd4s(
    structure: Structure,
    param: Param,
    *,
    q: Tensor | None = None,
    qmodel: str = "eeq",
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    ref: D4Reference | None = None,
    rcov_table: Tensor | TableFunction | None = None,
    en_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    wfpair_table: Tensor | TableFunction | None = None,
    zeff_table: Tensor | TableFunction | None = None,
    gam_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff | None = None,
    counting_function: CountingFunction = ncoord.erf_count,
) -> Tensor:
    """
    Evaluate atom-resolved DFT-D4S dispersion energy for a structure or batch.
    """
    cut = _default_d4_cutoff() if cutoff is None else cutoff
    return D4SModel(
        ga=ga,
        gc=gc,
        qmodel=qmodel,
        rcov_table=rcov_table,
        en_table=en_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        wfpair_table=wfpair_table,
        zeff_table=zeff_table,
        gam_table=gam_table,
        ref=ref,
        cutoff=cut,
        counting_function=counting_function,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )(structure, param, q=q, damping=damping, pairs=pairs)


def dftd4srev(
    structure: Structure,
    param: Param,
    *,
    q: Tensor | None = None,
    qmodel: str = "eeq",
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    ref: D4Reference | None = None,
    rcov_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    wfpair_table: Tensor | TableFunction | None = None,
    tanh_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff | None = None,
    counting_function: CountingFunction = ncoord.erf_count,
) -> Tensor:
    """
    Evaluate atom-resolved DFT-D4Srev dispersion energy for a structure or
    batch.
    """
    cut = _default_d4_cutoff() if cutoff is None else cutoff
    return D4SrevModel(
        qmodel=qmodel,
        rcov_table=rcov_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        wfpair_table=wfpair_table,
        tanh_table=tanh_table,
        ref=ref,
        cutoff=cut,
        counting_function=counting_function,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )(structure, param, q=q, damping=damping, pairs=pairs)


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
    if model_id == DispersionModelId.D1:
        return dftd1(
            structure,
            param,
            damping=damping,
            pairs=pairs,
            cutoff=cut,
            max_triples=max_triples,
            checkpoint=checkpoint,
            **kwargs,
        )
    if model_id == DispersionModelId.D2:
        return dftd2(
            structure,
            param,
            damping=damping,
            pairs=pairs,
            cutoff=cut,
            max_triples=max_triples,
            checkpoint=checkpoint,
            **kwargs,
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
    if model_id == DispersionModelId.D4:
        return dftd4(
            structure,
            param,
            damping=damping,
            pairs=pairs,
            cutoff=cut,
            max_triples=max_triples,
            checkpoint=checkpoint,
            **kwargs,
        )
    if model_id == DispersionModelId.D4S:
        return dftd4s(
            structure,
            param,
            damping=damping,
            pairs=pairs,
            cutoff=cut,
            max_triples=max_triples,
            checkpoint=checkpoint,
            **kwargs,
        )
    if model_id == DispersionModelId.D4SREV:
        return dftd4srev(
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


def get_properties(
    structure: Structure,
    *,
    model: (
        str
        | int
        | DispersionModelId
        | D1Model
        | D2Model
        | D3Model
        | D3SModel
        | D4Model
        | D4SModel
        | D4SrevModel
    ) = "d3",
    pairs: Pairs | None = None,
    cutoff: Cutoff | None = None,
    cn: Tensor | None = None,
    q: Tensor | None = None,
    **kwargs: Any,
) -> DispersionProperties:
    """
    Evaluate dispersion model properties (`cn`, `c6`, `c8`, and optional `q`,
    `alphadd`, `alphaqq`) matching ``get_properties`` in ``dftd/disp.f90``.
    """
    if isinstance(model, (D1Model, D2Model, D3Model, D3SModel)):
        return model.get_properties(structure, pairs=pairs, cn=cn)
    if isinstance(model, (D4Model, D4SModel, D4SrevModel)):
        return model.get_properties(structure, pairs=pairs, cn=cn, q=q)

    model_id = resolve_model_id(model)
    cut = (
        get_model_metadata(model_id).default_cutoff
        if cutoff is None
        else cutoff
    )
    if model_id == DispersionModelId.D1:
        return D1Model(cutoff=cut, **kwargs).get_properties(
            structure, pairs=pairs, cn=cn
        )
    if model_id == DispersionModelId.D2:
        return D2Model(cutoff=cut, **kwargs).get_properties(
            structure, pairs=pairs, cn=cn
        )
    if model_id == DispersionModelId.D3:
        return D3Model(cutoff=cut, **kwargs).get_properties(
            structure, pairs=pairs, cn=cn
        )
    if model_id == DispersionModelId.D3S:
        return D3SModel(cutoff=cut, **kwargs).get_properties(
            structure, pairs=pairs, cn=cn
        )
    if model_id == DispersionModelId.D4:
        return D4Model(cutoff=cut, **kwargs).get_properties(
            structure, pairs=pairs, cn=cn, q=q
        )
    if model_id == DispersionModelId.D4S:
        return D4SModel(cutoff=cut, **kwargs).get_properties(
            structure, pairs=pairs, cn=cn, q=q
        )
    if model_id == DispersionModelId.D4SREV:
        return D4SrevModel(cutoff=cut, **kwargs).get_properties(
            structure, pairs=pairs, cn=cn, q=q
        )
    raise NotImplementedError(
        f"Dispersion model '{model}' is not yet wired into get_properties()."
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
