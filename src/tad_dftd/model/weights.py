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
Reference Weighting and Charge Scaling Functions
================================================

Differentiable weighting and interpolation kernels shared across D3, D3S, D4,
D4S, and D4Srev dispersion models, following ``dftd/model/utils.f90`` and
``dftd/model/d4/{orig,smooth,srev}.f90``.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, Protocol, overload

import torch
from tad_mctc.typing import Tensor

from .. import defaults
from ..data.hardness import GAM
from ..data.tanh import TANH_PARAMS
from ..data.zeff import ZEFF

__all__ = [
    "WeightingFunction",
    "dzeta",
    "dzeta_tanh",
    "gaussian_log_weight",
    "gaussian_weight",
    "limit_avg_deriv",
    "quintic_hermite_basis",
    "quintic_hermite_weight",
    "smooth_weight",
    "weight_cn",
    "weight_references",
    "weight_references_d3s",
    "weight_references_d4",
    "weight_references_d4s",
    "zeta",
    "zeta_tanh",
]


class _HasCN(Protocol):
    cn: Tensor


class _HasD4Ref(Protocol):
    refn: Tensor
    ngw: Tensor
    cn: Tensor
    q: Tensor


WeightingFunction = Callable[..., Tensor]
"""
Function that weights the reference systems by the difference of their
coordination number to the one of the atom, as the logarithm of the
(unnormalized) weight, so that :func:`weight_references` can normalize the
weights without them underflowing.
"""


def gaussian_log_weight(dcn: Tensor, factor: Tensor | float = 4.0) -> Tensor:
    """
    Logarithm of the Gaussian weight of a reference system,
    ``-factor * dcn**2``, i.e. the weight ``exp(-factor * dcn**2)`` of D3.
    """
    return -factor * dcn * dcn


def weight_references(
    numbers: Tensor,
    cn: Tensor,
    reference: _HasCN,
    weighting_function: WeightingFunction = gaussian_log_weight,
    **kwargs: Any,
) -> Tensor:
    """
    Normalized weights of the reference systems of each atom using softmax of
    their logarithms.
    """
    refcn = reference.cn[numbers]
    mask = refcn >= 0
    dcn = refcn - cn.unsqueeze(-1)

    if (
        weighting_function is gaussian_log_weight
        or weighting_function is gaussian_weight
    ):
        factor = kwargs.get("factor", kwargs.get("wf", 4.0))
        log_w = gaussian_log_weight(dcn, factor=factor)
        log_weights = torch.where(mask, log_w, -torch.inf)

        has_reference = mask.any(dim=-1, keepdim=True)
        log_weights = torch.where(has_reference, log_weights, 0.0)

        weights = torch.softmax(log_weights, dim=-1)
        return torch.where(mask, weights, torch.zeros_like(weights))

    dcn_double = dcn.to(torch.double)
    raw = torch.where(
        mask,
        weighting_function(dcn_double, **kwargs),
        torch.zeros_like(dcn_double),
    )
    small = torch.tensor(1e-300, device=cn.device, dtype=torch.double)
    norm = torch.where(mask, torch.sum(raw, dim=-1, keepdim=True), small)
    safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
    gw_temp = (raw / safe_norm).to(cn.dtype)
    maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]
    exceptional = (norm == 0.0) | ~torch.isfinite(gw_temp)
    gw = torch.where(
        exceptional,
        torch.where(
            refcn == maxcn,
            torch.ones_like(gw_temp),
            torch.zeros_like(gw_temp),
        ),
        gw_temp,
    )
    return torch.where(mask, gw, torch.zeros_like(gw))


def weight_references_d3s(
    numbers: Tensor,
    cn: Tensor,
    reference: _HasCN,
    wfpair: Tensor,
    elements: Tensor | None = None,
) -> Tensor:
    """
    Normalized partner-element-dependent weights of the reference systems for
    DFT-D3S, of shape ``(..., nat, nel, nref)``.
    """
    if elements is None:
        elements = torch.arange(wfpair.shape[0], device=numbers.device)

    refcn = reference.cn[numbers]  # (..., nat, nref)
    mask_ref = refcn >= 0  # (..., nat, nref)

    wf = wfpair[numbers.unsqueeze(-1), elements]  # (..., nat, nel)
    valid_pair = (numbers.unsqueeze(-1) > 0) & (elements > 0) & (wf > 0)

    mask = mask_ref.unsqueeze(-2) & valid_pair.unsqueeze(-1)  # (..., nat, nel, nref)
    dcn = refcn.unsqueeze(-2) - cn.unsqueeze(-1).unsqueeze(-1)  # (..., nat, 1, nref)
    log_weights = torch.where(
        mask,
        -wf.unsqueeze(-1) * dcn * dcn,
        -torch.inf,
    )

    has_reference = mask.any(dim=-1, keepdim=True)
    log_weights = torch.where(has_reference, log_weights, 0.0)

    weights = torch.softmax(log_weights, dim=-1)
    return torch.where(mask, weights, torch.zeros_like(weights))


def _d4_cngw(
    cn_diff2: Tensor,
    wf: Tensor | float,
    ngw: Tensor,
    refcn: Tensor,
    valid_ref: Tensor,
    *,
    dcn: Tensor | None = None,
    with_dgwdcn: bool = False,
    orig_dtype: torch.dtype | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Evaluate normalized multi-Gaussian CN weights ``gwk`` (and optional
    derivative ``dgwk/d(cn)``) for D4, D4S, and D4Srev.

    Parameters
    ----------
    cn_diff2 : Tensor
        Squared CN difference ``(cn - refcn)^2`` of shape ``(..., nref)``.
    wf : Tensor | float
        Weighting factor broadcastable against ``cn_diff2``.
    ngw : Tensor
        Number of Gaussian weights per reference of shape ``(..., nref)``.
    refcn : Tensor
        Reference coordination numbers of shape ``(..., nref)``.
    valid_ref : Tensor
        Boolean mask of valid references of shape ``(..., nref)``.
    dcn : Tensor | None, optional
        Signed CN difference ``cn - refcn`` (required when ``with_dgwdcn=True``).
    with_dgwdcn : bool, optional
        Whether to also return ``dgwk/d(cn)``.
    orig_dtype : torch.dtype | None, optional
        Output floating-point dtype. Defaults to ``cn_diff2.dtype``.
    """
    out_dtype = cn_diff2.dtype if orig_dtype is None else orig_dtype
    cn_diff2_64 = cn_diff2.to(torch.float64)
    refcn_64 = refcn.to(torch.float64)
    igw = torch.arange(1, 7, device=cn_diff2.device, dtype=torch.float64)
    igw_mask = valid_ref.unsqueeze(-1) & (igw <= ngw.unsqueeze(-1))

    wf_t = (
        wf.to(torch.float64).unsqueeze(-1)
        if isinstance(wf, Tensor)
        else float(wf)
    )
    exponent = -wf_t * cn_diff2_64.unsqueeze(-1) * igw
    expw_igw = torch.where(
        igw_mask,
        torch.exp(exponent),
        torch.zeros_like(exponent),
    )
    expw = expw_igw.sum(dim=-1)

    norm = expw.sum(dim=-1, keepdim=True)
    eps_norm = math.sqrt(torch.finfo(torch.float64).tiny)
    norm_ok = norm > eps_norm

    # Fallback when norm underflows: 1.0 at max(refcn), 0.0 elsewhere (zero derivative)
    refcn_masked = torch.where(
        valid_ref, refcn_64, torch.full_like(refcn_64, -torch.inf)
    )
    max_rcn = torch.amax(refcn_masked, dim=-1, keepdim=True)
    fallback = torch.where(
        valid_ref & (torch.abs(refcn_64 - max_rcn) < 1e-12),
        torch.ones_like(expw),
        torch.zeros_like(expw),
    )

    safe_norm = torch.where(norm_ok, norm, torch.ones_like(norm))
    gwk_64 = torch.where(norm_ok, expw / safe_norm, fallback)
    gwk = torch.where(
        valid_ref,
        gwk_64.to(out_dtype),
        torch.zeros(cn_diff2.shape, device=cn_diff2.device, dtype=out_dtype),
    )
    if not with_dgwdcn:
        return gwk

    assert dcn is not None
    dcn_64 = dcn.to(torch.float64)
    dexpw = (-2.0 * wf_t * dcn_64.unsqueeze(-1) * igw * expw_igw).sum(dim=-1)
    dnorm = dexpw.sum(dim=-1, keepdim=True)
    dgwk_64 = torch.where(
        norm_ok,
        (dexpw - (expw / safe_norm) * dnorm) / safe_norm,
        torch.zeros_like(dexpw),
    )
    dgwk = torch.where(
        valid_ref & torch.isfinite(dgwk_64),
        dgwk_64.to(out_dtype),
        torch.zeros_like(gwk),
    )
    return gwk, dgwk


def _d4_charge_scale(
    numbers: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    valid_ref: Tensor,
    *,
    is_srev: bool = False,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Evaluate charge-scaling factor ``z`` (and optional ``dz/dq``) of shape
    ``(..., nat, 7)`` for D4/D4S or D4Srev.
    """
    dev = q.device
    dt = q.dtype
    ref_q = reference.q[numbers]  # (..., nat, 7)

    if is_srev:
        t_tab = (
            TANH_PARAMS(device=dev, dtype=dt)
            if tanh_table is None
            else tanh_table.to(device=dev, dtype=dt)
        )
        a_i = t_tab[numbers, 0].unsqueeze(-1)
        b_i = t_tab[numbers, 1].unsqueeze(-1)
        c_i = t_tab[numbers, 2].unsqueeze(-1)
        d_i = t_tab[numbers, 3].unsqueeze(-1)
        q_i = q.unsqueeze(-1)
        z = zeta_tanh(a_i, b_i, c_i, d_i, ref_q, q_i, mask=valid_ref)
        if not with_dgwdq:
            return z
        dz = dzeta_tanh(a_i, b_i, c_i, d_i, ref_q, q_i, mask=valid_ref)
        return z, dz

    z_tab = (
        ZEFF(device=dev, dtype=dt)
        if zeff_table is None
        else zeff_table.to(device=dev, dtype=dt)
    )
    g_tab = (
        GAM(device=dev, dtype=dt)
        if gam_table is None
        else gam_table.to(device=dev, dtype=dt)
    )
    zeff_i = z_tab[numbers].unsqueeze(-1)
    gi = (g_tab[numbers] * gc).unsqueeze(-1)
    qref_eff = ref_q + zeff_i
    qmod_eff = q.unsqueeze(-1) + zeff_i
    z = zeta(ga, gi, qref_eff, qmod_eff, mask=valid_ref)
    if not with_dgwdq:
        return z
    dz = dzeta(ga, gi, qref_eff, qmod_eff, mask=valid_ref)
    return z, dz


@overload
def weight_references_d4(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    *,
    wf: Tensor | float = defaults.D4_WF,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: Literal[False] = False,
    with_dgwdcn: Literal[False] = False,
) -> Tensor: ...


@overload
def weight_references_d4(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    *,
    wf: Tensor | float = defaults.D4_WF,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: Literal[True],
    with_dgwdcn: Literal[False] = False,
) -> tuple[Tensor, Tensor]: ...


@overload
def weight_references_d4(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    *,
    wf: Tensor | float = defaults.D4_WF,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: Literal[False] = False,
    with_dgwdcn: Literal[True],
) -> tuple[Tensor, Tensor]: ...


@overload
def weight_references_d4(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    *,
    wf: Tensor | float = defaults.D4_WF,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: Literal[True],
    with_dgwdcn: Literal[True],
) -> tuple[Tensor, Tensor, Tensor]: ...


def weight_references_d4(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    *,
    wf: Tensor | float = defaults.D4_WF,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: bool = False,
    with_dgwdcn: bool = False,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """
    Reference weights ``gw`` of shape ``(..., nat, 7)`` (and optional
    derivatives ``dgwdcn`` and/or ``dgwdq``) for DFT-D4 (or diagonal
    polarizability weights for DFT-D4S / DFT-D4Srev).
    """
    refn = reference.refn[numbers]  # (..., nat)
    ngw = reference.ngw[numbers]  # (..., nat, 7)
    refcn = reference.cn[numbers]  # (..., nat, 7)

    ir_idx = torch.arange(refcn.shape[-1], device=numbers.device)
    valid_ref = (numbers.unsqueeze(-1) > 0) & (ir_idx < refn.unsqueeze(-1))

    dcn = cn.to(torch.float64).unsqueeze(-1) - refcn.to(torch.float64)
    cn_diff2 = dcn * dcn
    wf_arg = wf.unsqueeze(-1) if isinstance(wf, Tensor) and wf.ndim == cn.ndim else wf

    if with_dgwdcn:
        gwk, dgwk = _d4_cngw(
            cn_diff2,
            wf_arg,
            ngw,
            refcn,
            valid_ref,
            dcn=dcn,
            with_dgwdcn=True,
            orig_dtype=cn.dtype,
        )
    else:
        gwk = _d4_cngw(
            cn_diff2, wf_arg, ngw, refcn, valid_ref, orig_dtype=cn.dtype
        )
        dgwk = None

    if with_dgwdq:
        z, dz = _d4_charge_scale(
            numbers,
            q,
            reference,
            valid_ref,
            is_srev=is_srev,
            ga=ga,
            gc=gc,
            zeff_table=zeff_table,
            gam_table=gam_table,
            tanh_table=tanh_table,
            with_dgwdq=True,
        )
        if dgwk is not None:
            return gwk * z, dgwk * z, gwk * dz
        return gwk * z, gwk * dz

    z = _d4_charge_scale(
        numbers,
        q,
        reference,
        valid_ref,
        is_srev=is_srev,
        ga=ga,
        gc=gc,
        zeff_table=zeff_table,
        gam_table=gam_table,
        tanh_table=tanh_table,
        with_dgwdq=False,
    )
    if dgwk is not None:
        return gwk * z, dgwk * z
    return gwk * z


@overload
def weight_references_d4s(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    wfpair: Tensor,
    elements: Tensor | None = None,
    *,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: Literal[False] = False,
) -> Tensor: ...


@overload
def weight_references_d4s(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    wfpair: Tensor,
    elements: Tensor | None = None,
    *,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: Literal[True],
) -> tuple[Tensor, Tensor]: ...


def weight_references_d4s(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: _HasD4Ref,
    wfpair: Tensor,
    elements: Tensor | None = None,
    *,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
    with_dgwdq: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Partner-element-dependent reference weights ``gw`` of shape
    ``(..., nat, nel, 7)`` (and optional ``dgwdq``) for DFT-D4S and DFT-D4Srev.
    """
    if elements is None:
        elements = torch.arange(wfpair.shape[0], device=numbers.device)

    refn = reference.refn[numbers]  # (..., nat)
    ngw = reference.ngw[numbers]  # (..., nat, 7)
    refcn = reference.cn[numbers]  # (..., nat, 7)

    ir_idx = torch.arange(refcn.shape[-1], device=numbers.device)
    valid_ref = (numbers.unsqueeze(-1) > 0) & (ir_idx < refn.unsqueeze(-1))

    wf = wfpair[numbers.unsqueeze(-1), elements]  # (..., nat, nel)
    valid_pair = (numbers.unsqueeze(-1) > 0) & (elements > 0) & (wf > 0)
    mask = valid_ref.unsqueeze(-2) & valid_pair.unsqueeze(-1)  # (..., nat, nel, 7)

    cn_diff2 = (cn.unsqueeze(-1).unsqueeze(-1) - refcn.unsqueeze(-2)) ** 2  # (..., nat, 1, 7)
    gwk = _d4_cngw(
        cn_diff2.expand_as(mask.to(cn.dtype)),
        wf.unsqueeze(-1),
        ngw.unsqueeze(-2).expand_as(mask),
        refcn.unsqueeze(-2).expand_as(mask.to(cn.dtype)),
        mask,
    )

    if with_dgwdq:
        z, dz = _d4_charge_scale(
            numbers,
            q,
            reference,
            valid_ref,
            is_srev=is_srev,
            ga=ga,
            gc=gc,
            zeff_table=zeff_table,
            gam_table=gam_table,
            tanh_table=tanh_table,
            with_dgwdq=True,
        )
        return gwk * z.unsqueeze(-2), gwk * dz.unsqueeze(-2)

    z = _d4_charge_scale(
        numbers,
        q,
        reference,
        valid_ref,
        is_srev=is_srev,
        ga=ga,
        gc=gc,
        zeff_table=zeff_table,
        gam_table=gam_table,
        tanh_table=tanh_table,
        with_dgwdq=False,
    )
    return gwk * z.unsqueeze(-2)


def gaussian_weight(
    dcn: Tensor,
    factor: Tensor | float = 4.0,
    *,
    wf: Tensor | float | None = None,
) -> Tensor:
    r"""
    Gaussian reference coordination number weight:

    .. math::

        w(\Delta \mathrm{CN}) = \exp\!\left(-w_f \, (\Delta \mathrm{CN})^2\right)
    """
    w = factor if wf is None else wf
    return torch.exp(-w * dcn * dcn)


def weight_cn(
    wf: Tensor | float,
    cn: Tensor,
    cnref: Tensor,
) -> Tensor:
    r"""
    Evaluate :math:`\exp(-w_f (\mathrm{CN} - \mathrm{CN}_{\mathrm{ref}})^2)`.
    """
    return gaussian_weight(cn - cnref, wf=wf)


def smooth_weight(
    cn: Tensor,
    cnref: Tensor,
    wf: Tensor,
    mask: Tensor,
) -> Tensor:
    r"""
    Normalized LogSumExp (softmax) smooth reference coordination number weights.
    """
    dcn = cn - cnref
    log_w = -wf * dcn * dcn
    neg_inf = torch.full_like(log_w, -torch.inf)
    log_w_masked = torch.where(mask, log_w, neg_inf)

    max_log_w = torch.amax(log_w_masked, dim=-1, keepdim=True)
    max_log_w = torch.where(
        torch.isfinite(max_log_w), max_log_w, torch.zeros_like(max_log_w)
    )

    exp_w = torch.where(
        mask, torch.exp(log_w - max_log_w), torch.zeros_like(log_w)
    )
    norm = torch.sum(exp_w, dim=-1, keepdim=True)
    safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
    return torch.where(mask, exp_w / safe_norm, torch.zeros_like(exp_w))


def zeta(
    a: Tensor | float,
    c: Tensor | float,
    qref: Tensor,
    qmod: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    r"""
    Charge scaling function :math:`\zeta(a, c, q_{\mathrm{ref}}, q_{\mathrm{mod}})`
    for D4 and D4S (``dftd/model/utils.f90``).
    """
    pos = qmod > 0.0
    if mask is not None:
        pos = pos & mask
    safe_qmod = torch.where(pos, qmod, torch.ones_like(qmod))
    exponent = c * (1.0 - qref / safe_qmod)
    val = torch.exp(a * (1.0 - torch.exp(torch.clamp(exponent, max=50.0))))
    fallback = torch.exp(
        torch.as_tensor(a, device=qmod.device, dtype=qmod.dtype)
    )
    out = torch.where(pos, val, fallback)
    if mask is not None:
        out = torch.where(mask, out, torch.zeros_like(out))
    return out


def dzeta(
    a: Tensor | float,
    c: Tensor | float,
    qref: Tensor,
    qmod: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    r"""
    First derivative of :func:`zeta` with respect to :math:`q_{\mathrm{mod}}`
    (``dftd/model/utils.f90``).
    """
    pos = qmod > 0.0
    if mask is not None:
        pos = pos & mask
    safe_qmod = torch.where(pos, qmod, torch.ones_like(qmod))
    exponent = c * (1.0 - qref / safe_qmod)
    exp_term = torch.exp(torch.clamp(exponent, max=50.0))
    z_val = torch.exp(a * (1.0 - exp_term))
    dz = -a * c * exp_term * z_val * qref / (safe_qmod * safe_qmod)
    return torch.where(pos, dz, torch.zeros_like(dz))


def zeta_tanh(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    qref: Tensor,
    qmod: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    r"""
    Hyperbolic-tangent charge scaling function for D4Srev
    (``dftd/model/utils.f90``).
    """
    zeta_mod = a + b * torch.tanh(c * qmod + d)
    zeta_ref = a + b * torch.tanh(c * qref + d)
    valid = zeta_ref != 0.0
    if mask is not None:
        valid = valid & mask
    safe_ref = torch.where(valid, zeta_ref, torch.ones_like(zeta_ref))
    return torch.where(valid, zeta_mod / safe_ref, torch.zeros_like(zeta_mod))


def dzeta_tanh(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    qref: Tensor,
    qmod: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    r"""
    First derivative of :func:`zeta_tanh` with respect to :math:`q_{\mathrm{mod}}`
    (``dftd/model/utils.f90``).
    """
    sech = 1.0 / torch.cosh(torch.clamp(c * qmod + d, min=-50.0, max=50.0))
    dzeta_mod = c * b * sech * sech
    zeta_ref = a + b * torch.tanh(c * qref + d)
    valid = zeta_ref != 0.0
    if mask is not None:
        valid = valid & mask
    safe_ref = torch.where(valid, zeta_ref, torch.ones_like(zeta_ref))
    return torch.where(valid, dzeta_mod / safe_ref, torch.zeros_like(dzeta_mod))


def limit_avg_deriv(dleft: Tensor, dright: Tensor) -> Tensor:
    r"""
    Monotonicity-preserving limited average nodal derivative for two adjacent
    intervals (``dftd/model/utils.f90``).
    """
    eps = torch.finfo(dleft.dtype).eps
    qlim = (15.0 / 7.0) * (1.0 - 100.0 * eps)

    same_sign = (dleft * dright) > 0.0
    avg_sec = 0.5 * (dleft + dright)
    max_sec = qlim * torch.minimum(torch.abs(dleft), torch.abs(dright))
    limited = torch.copysign(torch.minimum(torch.abs(avg_sec), max_sec), avg_sec)
    return torch.where(same_sign, limited, torch.zeros_like(avg_sec))


def quintic_hermite_basis(t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Quintic Hermite basis functions :math:`(h_0(t), h_1(t), h_2(t))` on
    :math:`t \in [0, 1]` (``dftd/model/utils.f90``).
    """
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    h0 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    h1 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    h2 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
    return h0, h1, h2


def quintic_hermite_weight(t: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""
    Quintic Hermite interpolation weight :math:`h_0(t) + a \, h_1(t) + b \, h_2(t)`.
    """
    h0, h1, h2 = quintic_hermite_basis(t)
    return h0 + a * h1 + b * h2
