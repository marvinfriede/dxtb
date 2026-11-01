# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
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
    CallableTwoBodyDamping,
    Damping,
    DampingParam,
    RationalTwoBody,
    ThreeBodyDamping,
    TwoBodyDamping,
    ZeroThreeBodyD3,
    as_damping_param,
    rational_damping,
)
from .model.base import DispersionProperties
from .model.c6 import AtomicC6
from .model.catalogue import (
    DispersionModelId,
    get_model_metadata,
    resolve_model_id,
)
from .model.d1 import D1Model, _default_d1_cutoff, _validate_d1_numbers
from .model.d2 import D2Model, _default_d2_cutoff, _validate_d2_numbers
from .model.d3 import (
    D3Model,
    D3SModel,
    Environment,
    _eval_dispersion2,
    _eval_dispersion3,
    _matrix,
    _validate_d3_numbers,
    _validate_d3s_numbers,
    default_damping,
)
from .model.d4 import (
    D4Model,
    D4SModel,
    D4SrevModel,
    _default_d4_cutoff,
    _validate_d4_numbers,
    _validate_d4srev_numbers,
)
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
    "validate_numbers",
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


def _legacy_damping_param(
    raw_param: Param,
    *,
    allow_unknown: bool = False,
) -> DampingParam:
    """
    Convert a legacy parameter mapping or :class:`DampingParam` to a
    :class:`DampingParam`, filling default ``s6``, ``s8``, ``a1``, and ``a2``
    values when omitted so sparse legacy dicts remain supported.
    """
    if isinstance(raw_param, DampingParam):
        return raw_param.replace(
            s6=defaults.S6 if raw_param.s6 is None else raw_param.s6,
            s8=defaults.S8 if raw_param.s8 is None else raw_param.s8,
            a1=defaults.A1 if raw_param.a1 is None else raw_param.a1,
            a2=defaults.A2 if raw_param.a2 is None else raw_param.a2,
        )
    merged: dict[str, Any] = {
        "s6": defaults.S6,
        "s8": defaults.S8,
        "a1": defaults.A1,
        "a2": defaults.A2,
        **raw_param,
    }
    if allow_unknown:
        return DampingParam.from_dict(merged)
    return as_damping_param(merged)


def dispersion(
    structure: Structure | Tensor,
    param: Param | Tensor,
    c6: Tensor | AtomicC6 | Param,
    c6_pos: Tensor | AtomicC6 | None = None,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    damping_function: Any = None,
    *,
    damping: Damping | None = None,
    pairs: Pairs | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Dispersion energy from given C6 coefficients: the two-body term, and the
    three-body term if `damping` has one.

    Supports both :class:`~tad_mctc.Structure` inputs and legacy
    ``(numbers, positions, param, c6, rvdw, r4r2, damping_function)`` calls.
    """
    if isinstance(structure, Structure):
        mol = structure
        raw_param: Param = param  # type: ignore[assignment]
        c6_val: Tensor | AtomicC6 = c6  # type: ignore[assignment]
        rvdw_val = rvdw
        r4r2_val = r4r2
        cut = Cutoff() if cutoff is None else cutoff
    else:
        assert isinstance(param, Tensor)
        assert c6_pos is not None
        mol = Structure(numbers=structure, positions=param)
        raw_param = c6  # type: ignore[assignment]
        c6_val = c6_pos
        rvdw_val = rvdw
        r4r2_val = r4r2
        cut = (
            Cutoff(
                disp2=defaults.D3_DISP_CUTOFF,
                disp3=defaults.D3_DISP_CUTOFF,
            )
            if cutoff is None
            else cutoff
        )
        custom_fn = (
            damping is None
            and damping_function is not None
            and damping_function is not rational_damping
        )
        dparam_legacy = _legacy_damping_param(
            raw_param, allow_unknown=custom_fn
        )
        if custom_fn:
            def_damp = default_damping(dparam_legacy)
            two = (
                damping_function
                if isinstance(damping_function, TwoBodyDamping)
                else CallableTwoBodyDamping(
                    damping_function,
                    raw_param if isinstance(raw_param, Mapping) else None,
                )
            )
            damping = Damping(two=two, three=def_damp.three)
        raw_param = dparam_legacy

    d3 = D3Model(
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        cutoff=cut,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )
    if pairs is None:
        pairs = DensePairs()
    dparam, damp, prs = d3._prepare(mol, raw_param, damping, pairs)

    if isinstance(prs, DensePairs):
        c6_val = _matrix(c6_val)

    energy = _eval_dispersion2(
        mol,
        dparam,
        c6_val,
        prs.disp2,
        damp.two,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        rvdw=rvdw_val,
        r4r2=r4r2_val,
        cutoff=cut,
        checkpoint=checkpoint,
    )
    if damp.three is not None:
        energy = energy + _eval_dispersion3(
            mol,
            dparam,
            c6_val,
            prs.disp3,
            damp.three,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            rvdw=rvdw_val,
            r4r2=r4r2_val,
            cutoff=cut,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )
    return energy


def dispersion2(
    structure: Structure | Tensor,
    param: Param | Tensor,
    c6: Tensor | AtomicC6 | Param,
    r4r2_or_c6: Tensor | AtomicC6 | None = None,
    r4r2: Tensor | None = None,
    damping_function: Any = None,
    cutoff_pos: Cutoff | Tensor | float | None = None,
    *,
    pairs: PeriodicShifts | NeighborList | None = None,
    damping: TwoBodyDamping | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    rvdw: Tensor | None = None,
    cutoff: Cutoff | Tensor | float | None = None,
    checkpoint: bool = False,
    as_matrix: bool = False,
) -> Tensor:
    """
    Two-body dispersion energy from given C6 coefficients.
    """
    if isinstance(structure, Structure):
        mol = structure
        raw_param: Param = param  # type: ignore[assignment]
        c6_val: Tensor | AtomicC6 = c6  # type: ignore[assignment]
        r4r2_val = r4r2
        cut_arg = cutoff
        cut: Cutoff | Tensor | float = Cutoff() if cut_arg is None else cut_arg
        dparam = as_damping_param(raw_param)
    else:
        assert isinstance(param, Tensor)
        assert r4r2_or_c6 is not None
        mol = Structure(numbers=structure, positions=param)
        raw_param = c6  # type: ignore[assignment]
        c6_val = r4r2_or_c6
        r4r2_val = r4r2
        cut_arg = cutoff if cutoff is not None else cutoff_pos
        cut = (
            Cutoff(disp2=defaults.D3_DISP_CUTOFF)
            if cut_arg is None
            else cut_arg
        )
        custom_fn = isinstance(damping, CallableTwoBodyDamping) or (
            damping is None
            and damping_function is not None
            and damping_function is not rational_damping
        )
        dparam = _legacy_damping_param(raw_param, allow_unknown=custom_fn)
        if damping is None:
            if damping_function is None or damping_function is rational_damping:
                damping = RationalTwoBody()
            elif isinstance(damping_function, TwoBodyDamping):
                damping = damping_function
            else:
                damping = CallableTwoBodyDamping(
                    damping_function,
                    raw_param if isinstance(raw_param, Mapping) else None,
                )

    if damping is None:
        damping = default_damping(dparam).two

    return _eval_dispersion2(
        mol,
        dparam,
        c6_val,
        pairs,
        damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        rdamp_table=rdamp_table,
        rvdw=rvdw,
        r4r2=r4r2_val,
        cutoff=cut,
        checkpoint=checkpoint,
        as_matrix=as_matrix,
    )


def dispersion3(
    structure: Structure | Tensor,
    param: Param | Tensor,
    c6: Tensor | AtomicC6 | Param,
    rvdw_or_c6: Tensor | AtomicC6 | None = None,
    rvdw: Tensor | None = None,
    cutoff_pos: Cutoff | Tensor | float | None = None,
    *,
    pairs: TermPairs = None,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    r4r2: Tensor | None = None,
    cutoff: Cutoff | Tensor | float | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Three-body (Axilrod-Teller-Muto) dispersion term from given C6 coefficients.
    """
    if isinstance(structure, Structure):
        mol = structure
        raw_param: Param = param  # type: ignore[assignment]
        c6_val: Tensor | AtomicC6 = c6  # type: ignore[assignment]
        rvdw_val = rvdw
        cut_arg = cutoff
        cut: Cutoff | Tensor | float = Cutoff() if cut_arg is None else cut_arg
        dparam = as_damping_param(raw_param)
    else:
        assert isinstance(param, Tensor)
        assert rvdw_or_c6 is not None
        mol = Structure(numbers=structure, positions=param)
        raw_param = c6  # type: ignore[assignment]
        c6_val = rvdw_or_c6
        rvdw_val = rvdw
        cut_arg = cutoff if cutoff is not None else cutoff_pos
        cut = (
            Cutoff(disp3=defaults.D3_DISP_CUTOFF)
            if cut_arg is None
            else cut_arg
        )
        dparam = _legacy_damping_param(raw_param)

    return _eval_dispersion3(
        mol,
        dparam,
        c6_val,
        pairs,
        damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        rdamp_table=rdamp_table,
        rvdw=rvdw_val,
        r4r2=r4r2,
        cutoff=cut,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )


def validate_numbers(
    numbers: Tensor,
    model: str | int | DispersionModelId = "d3",
    *,
    q: Tensor | None = None,
    qmodel: str = "eeq",
) -> Tensor:
    """
    Validate atomic numbers for `model` at the structure boundary (`ValueError`
    in eager mode, graph-connected out-of-bounds guard under `torch.compile`).
    """
    import torch

    from .charge import MAX_ELEM_EEQ
    from .data.d1 import D1_C6
    from .data.d2 import D2_C6
    from .data.table import guard_numbers

    model_id = resolve_model_id(model)
    like = torch.zeros(1, device=numbers.device, dtype=torch.float64)
    if model_id == DispersionModelId.D1:
        return _validate_d1_numbers(numbers, D1_C6(device=like.device, dtype=like.dtype))
    if model_id == DispersionModelId.D2:
        return _validate_d2_numbers(numbers, D2_C6(device=like.device, dtype=like.dtype))
    if model_id == DispersionModelId.D3:
        return _validate_d3_numbers(numbers)
    if model_id == DispersionModelId.D3S:
        return _validate_d3s_numbers(numbers)
    if model_id in (DispersionModelId.D4, DispersionModelId.D4S):
        numbers = _validate_d4_numbers(numbers)
        if q is None and qmodel.lower().strip() in ("eeq", "eeq2019"):
            invalid = (numbers < 0) | (numbers > MAX_ELEM_EEQ)
            numbers = guard_numbers(
                numbers,
                invalid,
                f"Structure contains unsupported element for 'eeq' charge model "
                f"(valid range: 1..{MAX_ELEM_EEQ}).",
            )
        return numbers
    if model_id == DispersionModelId.D4SREV:
        return _validate_d4srev_numbers(numbers)
    raise NotImplementedError(
        f"Dispersion model '{model}' is not supported by validate_numbers()."
    )
