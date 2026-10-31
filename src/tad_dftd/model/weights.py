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
D4S, D4Srev, and D5 dispersion models, following ``dftd/model/utils.f90``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from tad_mctc.typing import Tensor

__all__ = [
    "WeightingFunction",
    "gaussian_log_weight",
    "gaussian_weight",
    "limit_avg_deriv",
    "quintic_hermite_basis",
    "quintic_hermite_weight",
    "smooth_weight",
    "weight_cn",
    "weight_references",
    "weight_references_d3s",
    "zeta",
    "zeta_tanh",
]


class _HasCN(Protocol):
    cn: Tensor


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

    if weighting_function is gaussian_weight:
        factor = kwargs.get("factor", kwargs.get("wf", 4.0))
        log_w = gaussian_log_weight(dcn, factor=factor)
    else:
        log_w = weighting_function(dcn, **kwargs)

    log_weights = torch.where(mask, log_w, -torch.inf)

    has_reference = mask.any(dim=-1, keepdim=True)
    log_weights = torch.where(has_reference, log_weights, 0.0)

    weights = torch.softmax(log_weights, dim=-1)
    return torch.where(mask, weights, torch.zeros_like(weights))


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

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of shape ``(..., nat)``.
    cn : Tensor
        Coordination numbers of shape ``(..., nat)``.
    reference : _HasCN
        D3S reference systems (see :meth:`Reference.load_d3s`).
    wfpair : Tensor
        Pairwise weighting factor table ``wf(Z_A, Z_B)`` of shape ``(95, 95)``.
    elements : Tensor | None, optional
        1D tensor of partner element atomic numbers ``(nel,)``. Defaults to
        all elements ``0..wfpair.shape[0]-1``.

    Returns
    -------
    Tensor
        Normalized weights of shape ``(..., nat, nel, nref)``.
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
    Normalized LogSumExp (softmax) smooth reference coordination number weights
    for D3S, D4S, and D4Srev:

    .. math::

        w_r = \frac{\exp\!\left(-w_f \, (\mathrm{CN} - \mathrm{CN}_{\mathrm{ref},r})^2\right)}
                   {\sum_{r'} \exp\!\left(-w_f \, (\mathrm{CN} - \mathrm{CN}_{\mathrm{ref},r'})^2\right)}

    Evaluated in log-space with a shift by the maximum valid log-weight so that
    exponential underflow never causes ``0 / 0 = NaN`` at large ``wf`` or large
    ``|CN - CN_ref|``.

    Parameters
    ----------
    cn : Tensor
        Coordination numbers broadcastable against ``cnref`` and ``wf``.
    cnref : Tensor
        Reference coordination numbers with the reference axis at ``dim=-1``.
    wf : Tensor
        Weighting factor :math:`w_f` broadcastable against ``cnref``.
    mask : Tensor
        Boolean mask of valid reference slots along ``dim=-1``.

    Returns
    -------
    Tensor
        Normalized reference weights along ``dim=-1`` (zero for masked slots).
    """
    dcn = cn - cnref
    log_w = -wf * dcn * dcn
    neg_inf = torch.full_like(log_w, -torch.inf)
    log_w_masked = torch.where(mask, log_w, neg_inf)

    max_log_w = torch.amax(log_w_masked, dim=-1, keepdim=True)
    # Replace -inf (for padding atoms where all mask entries are False) with 0.0
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
) -> Tensor:
    r"""
    Charge scaling function :math:`\zeta(a, c, q_{\mathrm{ref}}, q_{\mathrm{mod}})`
    for D4 and D4S (``dftd/model/utils.f90``):

    .. math::

        \zeta(a, c, q_{\mathrm{ref}}, q_{\mathrm{mod}}) =
        \begin{cases}
            \exp(a) & q_{\mathrm{mod}} \le 0, \\
            \exp\!\left(a \left(1 - \exp\!\left(c \left(1 - \frac{q_{\mathrm{ref}}}{q_{\mathrm{mod}}}\right)\right)\right)\right) & q_{\mathrm{mod}} > 0.
        \end{cases}

    Differentiated safely via ``torch.where`` with a positive dummy ``qmod`` on
    the non-positive branch so that ``dzeta = 0`` and ``d2zeta = 0`` for
    :math:`q_{\mathrm{mod}} \le 0` without ``NaN`` gradients.
    """
    pos = qmod > 0.0
    safe_qmod = torch.where(pos, qmod, torch.ones_like(qmod))
    exponent = c * (1.0 - qref / safe_qmod)
    # Clamp exponent to avoid exp overflow on masked/dummy entries
    val = torch.exp(a * (1.0 - torch.exp(torch.clamp(exponent, max=50.0))))
    fallback = torch.exp(torch.as_tensor(a, device=qmod.device, dtype=qmod.dtype))
    return torch.where(pos, val, fallback)


def zeta_tanh(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    qref: Tensor,
    qmod: Tensor,
) -> Tensor:
    r"""
    Hyperbolic-tangent charge scaling function for D4Srev
    (``dftd/model/utils.f90``):

    .. math::

        \zeta_{\tanh}(a, b, c, d, q_{\mathrm{ref}}, q_{\mathrm{mod}}) =
        \frac{a + b \tanh(c \, q_{\mathrm{mod}} + d)}
             {a + b \tanh(c \, q_{\mathrm{ref}} + d)}.
    """
    zeta_mod = a + b * torch.tanh(c * qmod + d)
    zeta_ref = a + b * torch.tanh(c * qref + d)
    safe_ref = torch.where(zeta_ref != 0.0, zeta_ref, torch.ones_like(zeta_ref))
    return torch.where(
        zeta_ref != 0.0, zeta_mod / safe_ref, torch.zeros_like(zeta_mod)
    )


def limit_avg_deriv(dleft: Tensor, dright: Tensor) -> Tensor:
    r"""
    Monotonicity-preserving limited average nodal derivative for two adjacent
    intervals (``dftd/model/utils.f90``):

    .. math::

        d_{\mathrm{node}} =
        \begin{cases}
            0 & d_{\mathrm{left}} \, d_{\mathrm{right}} \le 0, \\
            \mathrm{sign}(d_{\mathrm{avg}}) \min\!\left(|d_{\mathrm{avg}}|,\, q_{\mathrm{lim}} \min(|d_{\mathrm{left}}|, |d_{\mathrm{right}}|)\right) & \text{otherwise},
        \end{cases}

    where :math:`d_{\mathrm{avg}} = \tfrac{1}{2}(d_{\mathrm{left}} + d_{\mathrm{right}})`
    and :math:`q_{\mathrm{lim}} = \frac{15}{7}(1 - 100 \epsilon)`.
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
    :math:`t \in [0, 1]` (``dftd/model/utils.f90``):

    .. math::

        h_0(t) &= 10 t^3 - 15 t^4 + 6 t^5, \\
        h_1(t) &= t - 6 t^3 + 8 t^4 - 3 t^5, \\
        h_2(t) &= -4 t^3 + 7 t^4 - 3 t^5.
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
