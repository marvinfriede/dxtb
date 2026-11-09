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
Model: Atomic C6
================

Computation of atomic C6 dispersion coefficients for D3 and D3S, factored per
atom over the element axis so that neither dense ``(..., nat, nat, 7, 7)``
intermediates nor quadratic ``O(nat^2)`` memory in sparse mode is ever needed.
"""
from __future__ import annotations

import torch
from tad_mctc.autograd import is_functorch_tensor, is_vmapped
from tad_mctc.math import einsum
from tad_mctc.tools import is_compiling
from tad_mctc.tree import Node, child
from tad_mctc.typing import Tensor

from .. import defaults
from ..reference.d3 import Reference
from ..reference.d4 import D4Reference
from .weights import weight_references_d3s, weight_references_d4s

__all__ = [
    "AtomicC6",
    "atomic_c6",
    "atomic_c6_d3s",
    "atomic_c6_d4s",
    "contract_rc6",
    "factored_c6",
    "factored_c6_d3s",
    "factored_c6_d4s",
]


def contract_rc6(
    rc6: Tensor, gw_i: Tensor, gw_j: Tensor | None = None
) -> Tensor:
    """
    Contract a pre-expanded pairwise reference C6 tensor ``(..., nat, nat, nref, nref)``
    with reference weights ``gw_i`` and ``gw_j`` (defaults to ``gw_i``).
    """
    if gw_j is None:
        gw_j = gw_i
    return einsum(
        "...ijab,...ia,...jb->...ij",
        *(rc6, gw_i, gw_j),
        optimize=[(0, 1), (0, 1)],
    )


class AtomicC6(Node):
    """
    The C6 coefficients of all pairs of atoms, factored per atom.

    Supports both D3 (where `weights` has shape ``(..., nat, nref)``) and D3S
    (where `weights` has partner-element-dependent shape
    ``(..., nat, nel, nref)``).

    Parameters
    ----------
    partial : Tensor
        ``sum_a w[i, (u,) a] * c6ref[Z_i, u, a, b]`` of every atom ``i`` with
        every element ``u`` of the element axis, ``(..., nat, nel, nref)``.
    weights : Tensor
        Weights of the reference systems of every atom, ``(..., nat, nref)``
        for D3 or ``(..., nat, nel, nref)`` for D3S.
    element : Tensor
        Index of the element of every atom on the element axis of `partial`,
        ``(..., nat)``.
    """

    partial: Tensor = child()
    weights: Tensor = child()
    element: Tensor = child()

    def dense(self) -> Tensor:
        """
        The C6 coefficients of all pairs of atoms, ``(..., nat, nat)``.
        """
        nel = self.partial.shape[-2]
        elements = torch.arange(nel, device=self.element.device)
        onehot = (self.element.unsqueeze(-1) == elements).type(
            self.weights.dtype
        )

        if self.weights.ndim == self.partial.ndim - 1:
            # D3: weights has shape (..., nat, nref)
            wu = self.weights.unsqueeze(-2) * onehot[..., None]
            return (
                self.partial.flatten(-2) @ wu.flatten(-2).transpose(-1, -2)
            )

        # D3S: weights has shape (..., nat, nel, nref)
        if nel <= 16:
            p_mat = (
                onehot[..., :, :, None, None]
                * self.partial[..., :, None, :, :]
            )
            q_mat = (
                self.weights[..., :, :, None, :]
                * onehot[..., :, None, :, None]
            )
            return p_mat.flatten(-3) @ q_mat.flatten(-3).transpose(-1, -2)

        nat = self.element.shape[-1]
        nref = self.partial.shape[-1]
        idx = self.element[..., None, :, None].expand(
            *self.partial.shape[:-2], nat, nref
        )
        g_ij = torch.take_along_dim(self.partial, idx, dim=-2)
        w_ji = torch.take_along_dim(self.weights, idx, dim=-2)
        return torch.sum(g_ij * w_ji.transpose(-2, -3), dim=-1)

    def pair(self, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """
        The C6 coefficients of the pairs ``(idx_i, idx_j)``, with the atoms of
        a batch numbered as one flat system, atom ``i`` of system ``b`` being
        ``b * nat + i``.
        """
        nel, nref = self.partial.shape[-2:]
        partial = self.partial.reshape(-1, nref)
        elem_flat = self.element.reshape(-1)
        element_j = elem_flat.index_select(0, idx_j)
        rows = partial.index_select(0, idx_i * nel + element_j)

        if self.weights.ndim == self.partial.ndim - 1:
            weights_j = self.weights.reshape(-1, nref).index_select(0, idx_j)
        else:
            element_i = elem_flat.index_select(0, idx_i)
            weights_j = self.weights.reshape(-1, nref).index_select(
                0, idx_j * nel + element_i
            )
        return torch.sum(rows * weights_j, dim=-1)


def factored_c6(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference | D4Reference,
) -> AtomicC6:
    """
    Atomic dispersion coefficients for D3 or D4, factored per atom.
    """
    elements, element, width = _element_axis(
        numbers, reference, reference.c6.shape[0]
    )
    return _factors(
        numbers, weights, reference, elements, element, width, _SUBSCRIPTS_D3
    )


def atomic_c6(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference | D4Reference,
) -> Tensor:
    """Calculate dense ``(..., nat, nat)`` atomic C6 coefficients for D3 or D4."""
    return factored_c6(numbers, weights, reference).dense()


def factored_c6_d3s(
    numbers: Tensor,
    cn: Tensor,
    reference: Reference | D4Reference,
    wfpair: Tensor,
    weights: Tensor | None = None,
) -> AtomicC6:
    """
    Atomic dispersion coefficients for D3S, factored per atom and partner
    element.
    """
    if weights is not None:
        nel = weights.shape[-2]
        elements, element, width = _element_axis(
            numbers,
            reference,
            nel,
            static=nel in (wfpair.shape[0], reference.c6.shape[0]),
        )
        return _factors(
            numbers,
            weights,
            reference,
            elements,
            element,
            width,
            _SUBSCRIPTS_D3S,
        )

    elements, element, width = _element_axis(numbers, reference, wfpair.shape[0])
    w = weight_references_d3s(numbers, cn, reference, wfpair, elements)
    return _factors(
        numbers, w, reference, elements, element, width, _SUBSCRIPTS_D3S
    )


def atomic_c6_d3s(
    numbers: Tensor,
    cn: Tensor,
    reference: Reference | D4Reference,
    wfpair: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Calculate dense ``(..., nat, nat)`` atomic C6 coefficients for D3S."""
    return factored_c6_d3s(numbers, cn, reference, wfpair, weights).dense()


def factored_c6_d4s(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: D4Reference,
    wfpair: Tensor,
    weights: Tensor | None = None,
    *,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
) -> AtomicC6:
    """
    Atomic dispersion coefficients for D4S and D4Srev, factored per atom and
    partner element.
    """
    if weights is not None:
        return factored_c6_d3s(
            numbers, cn, reference, wfpair, weights=weights
        )

    elements, element, width = _element_axis(numbers, reference, wfpair.shape[0])
    w = weight_references_d4s(
        numbers,
        cn,
        q,
        reference,
        wfpair,
        elements,
        ga=ga,
        gc=gc,
        is_srev=is_srev,
        zeff_table=zeff_table,
        gam_table=gam_table,
        tanh_table=tanh_table,
    )
    return _factors(
        numbers, w, reference, elements, element, width, _SUBSCRIPTS_D3S
    )


def atomic_c6_d4s(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    reference: D4Reference,
    wfpair: Tensor,
    weights: Tensor | None = None,
    *,
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    is_srev: bool = False,
    zeff_table: Tensor | None = None,
    gam_table: Tensor | None = None,
    tanh_table: Tensor | None = None,
) -> Tensor:
    """Calculate dense ``(..., nat, nat)`` atomic C6 coefficients for D4S or D4Srev."""
    return factored_c6_d4s(
        numbers,
        cn,
        q,
        reference,
        wfpair,
        weights=weights,
        ga=ga,
        gc=gc,
        is_srev=is_srev,
        zeff_table=zeff_table,
        gam_table=gam_table,
        tanh_table=tanh_table,
    ).dense()


def _reference_width(
    reference: Reference | D4Reference, elements: Tensor
) -> int:
    """
    The number of leading references that hold a non-zero C6 for any pair of
    `elements`.
    """
    nref = reference.c6.shape[-1]
    if is_compiling() or is_functorch_tensor(reference.c6):
        return nref

    block = reference.c6[elements.unsqueeze(-1), elements]
    used = (block != 0).flatten(0, 1).any(0)
    used = used.any(0) | used.any(1)

    last = torch.arange(1, nref + 1, device=used.device) * used
    return max(int(last.max()), 1)


_SUBSCRIPTS_D3 = "...ia,...iuab->...iub"
"""Contraction of D3/D4 weights ``(..., nat, nref)`` with the references."""

_SUBSCRIPTS_D3S = "...iua,...iuab->...iub"
"""Contraction of D3S/D4S weights ``(..., nat, nel, nref)``."""


def _element_axis(
    numbers: Tensor,
    reference: Reference | D4Reference,
    nel: int,
    *,
    static: bool = False,
) -> tuple[Tensor, Tensor, int | None]:
    """
    The element axis of the factored coefficients: ``(elements, element,
    width)``, with the elements on the axis, the index of the element of every
    atom on it, and the number of leading references to keep (``None`` for all).

    Under tracing (``torch.compile``, ``vmap``, functorch) or if `static`, the
    axis holds all `nel` elements and no data-dependent shape is derived.
    Otherwise it holds only the elements present in `numbers`.
    """
    if (
        static
        or is_compiling()
        or is_vmapped(numbers)
        or is_functorch_tensor(reference.c6)
    ):
        return torch.arange(nel, device=numbers.device), numbers, None

    elements = torch.unique(numbers)
    element = torch.searchsorted(elements, numbers.contiguous())
    return elements, element, _reference_width(reference, elements)


def _factors(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference | D4Reference,
    elements: Tensor,
    element: Tensor,
    width: int | None,
    subscripts: str,
) -> AtomicC6:
    table = reference.c6
    if width is not None:
        table = table[..., :width, :width]
        weights = weights[..., :width]
    rc6 = table[numbers.unsqueeze(-1), elements]
    g = torch.einsum(subscripts, weights, rc6)
    return AtomicC6(partial=g, weights=weights, element=element)
