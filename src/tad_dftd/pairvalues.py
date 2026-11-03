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
Pair values
===========

What the two- and three-body terms read of a pair of atoms besides its
distance: its C6 coefficient and its radii (:class:`PairRadii`).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor import pair_distance_squared, split_lattice
from tad_mctc.typing import TableFunction, Tensor

from .data.table import element_table

__all__ = [
    "PairRadii",
    "SlotValues",
    "nonzero",
    "pair_radii",
    "pair_radii_of",
    "slot_values",
]


class PairRadii(NamedTuple):
    """The values of :class:`~tad_dftd.damping.PairData` that follow from
    the elements of the two atoms alone, all finite and positive."""

    qq: Tensor
    """``3 * r4r2[i] * r4r2[j]``."""

    rdamp: Tensor
    """``sqrt(qq)``, the damping radius of the model, as in D4."""

    rvdw: Tensor
    """Van-der-Waals radius of the pair."""

    znum: Tensor
    """Sum of the atomic numbers of the pair."""

    def map(self, fn: Callable[[Tensor], Tensor]) -> PairRadii:
        """`fn` applied to every value, e.g. to index or broadcast them."""
        return PairRadii(*(fn(value) for value in self))


def nonzero(x: Tensor) -> Tensor:
    """`x`, with ones in place of zeros."""
    return torch.where(x != 0, x, torch.ones_like(x))


def pair_radii_of(
    z_i: Tensor,
    z_j: Tensor,
    rvdw: Tensor,
    r4r2: Tensor,
    *,
    rdamp: Tensor | None = None,
) -> PairRadii:
    """
    The element-dependent values of the pairs ``(z_i, z_j)``.

    Parameters
    ----------
    z_i, z_j : Tensor
        Atomic numbers of the two atoms of each pair, zero for padding.
    rvdw : Tensor
        Van-der-Waals radii, ``(nelem, nelem)``.
    r4r2 : Tensor
        Expectation values ``<r4>/<r2>``, ``(nelem,)``.
    rdamp : Tensor | None, optional
        Explicit pairwise damping radius table ``(nelem, nelem)`` (e.g. for
        D1, D2). Defaults to ``None`` (``sqrt(qq)``).

    Returns
    -------
    PairRadii
        Shaped like the broadcast of `z_i` and `z_j`, in the dtype of the
        tables.
    """
    r4r2_i = nonzero(r4r2[z_i])
    r4r2_j = nonzero(r4r2[z_j])
    qq = 3 * r4r2_i * r4r2_j
    rdamp_val = (
        torch.sqrt(qq) if rdamp is None else nonzero(rdamp[z_i, z_j])
    )
    return PairRadii(
        qq=qq,
        rdamp=rdamp_val,
        rvdw=nonzero(rvdw[z_i, z_j]),
        znum=nonzero(z_i + z_j).to(qq.dtype),
    )


def pair_radii(
    structure: Structure,
    rvdw_table: Tensor | TableFunction | None,
    r4r2_table: Tensor | TableFunction | None,
    *,
    rdamp_table: Tensor | TableFunction | None = None,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    rdamp: Tensor | None = None,
) -> PairRadii:
    """
    The radii of all pairs of atoms of `structure`, ``(..., nat, nat)``.

    Optional pre-gathered per-pair `rvdw` / `rdamp` (shape ``(..., nat, nat)``)
    and per-atom `r4r2` (shape ``(..., nat)``) may be supplied directly to
    avoid re-indexing element tables when setup data has already been gathered.
    """
    numbers, positions = structure.numbers, structure.positions
    rdamp_tbl = (
        element_table(rdamp_table, "rdamp_table", positions)
        if rdamp_table is not None
        else None
    )
    if rvdw is None and r4r2 is None and rdamp is None:
        return pair_radii_of(
            numbers.unsqueeze(-1),
            numbers.unsqueeze(-2),
            element_table(rvdw_table, "rvdw_table", positions),
            element_table(r4r2_table, "r4r2_table", positions),
            rdamp=rdamp_tbl,
        )

    z_i = numbers.unsqueeze(-1)
    z_j = numbers.unsqueeze(-2)

    if r4r2 is not None:
        r4r2_atom = nonzero(
            r4r2.to(device=positions.device, dtype=positions.dtype)
        )
        qq = 3 * r4r2_atom.unsqueeze(-1) * r4r2_atom.unsqueeze(-2)
    else:
        r4r2_tbl = element_table(r4r2_table, "r4r2_table", positions)
        qq = 3 * nonzero(r4r2_tbl[z_i]) * nonzero(r4r2_tbl[z_j])

    if rvdw is not None:
        rvdw_val = nonzero(
            rvdw.to(device=positions.device, dtype=positions.dtype)
        )
    else:
        rvdw_tbl = element_table(rvdw_table, "rvdw_table", positions)
        rvdw_val = nonzero(rvdw_tbl[z_i, z_j])

    if rdamp is not None:
        rdamp_val = nonzero(
            rdamp.to(device=positions.device, dtype=positions.dtype)
        )
    elif rdamp_tbl is not None:
        rdamp_val = nonzero(rdamp_tbl[z_i, z_j])
    else:
        rdamp_val = torch.sqrt(qq)

    return PairRadii(
        qq=qq,
        rdamp=rdamp_val,
        rvdw=rvdw_val,
        znum=nonzero(z_i + z_j).to(qq.dtype),
    )


class SlotValues(NamedTuple):
    """What the terms read of each slot of a list, all ``(n,)``."""

    distance_squared: Tensor
    """Squared distance of the pair, with the shift of the slot."""

    c6: Tensor
    """C6 coefficient of the pair."""

    radii: PairRadii
    """Radii of the pair."""


def slot_values(
    structure: Structure,
    c6: Tensor | Any,
    rvdw_table: Tensor | TableFunction | None,
    r4r2_table: Tensor | TableFunction | None,
    *,
    rdamp_table: Tensor | TableFunction | None = None,
    c8: Tensor | Any | None = None,
) -> Callable[[Tensor, Tensor, Tensor], SlotValues]:
    """
    The values of the slots ``(idx_i, idx_j, shift)`` of a list, for any
    batch of its slots (a chunk of the list, or all its slots at once).
    """
    numbers, positions = structure.numbers, structure.positions
    rvdw = element_table(rvdw_table, "rvdw_table", positions)
    r4r2 = element_table(r4r2_table, "r4r2_table", positions)
    rdamp = (
        element_table(rdamp_table, "rdamp_table", positions)
        if rdamp_table is not None
        else None
    )

    nat = positions.shape[-2]
    numbers_flat = numbers.reshape(-1)
    total_atoms = numbers_flat.shape[0]

    flat_positions = positions.reshape(-1, 3)
    padded_positions = torch.cat(
        [flat_positions, flat_positions.new_zeros(1, 3)]
    )
    shared_lattice, system_lattices = split_lattice(structure.lattice)
    if system_lattices is not None:
        system_lattices = torch.cat(
            [system_lattices, system_lattices.new_zeros(1, 3, 3)]
        )

    def values(idx_i: Tensor, idx_j: Tensor, shift: Tensor) -> SlotValues:
        distance_squared = pair_distance_squared(
            idx_i,
            idx_j,
            shift,
            padded_positions,
            shared_lattice=shared_lattice,
            system_lattices=system_lattices,
            atoms_per_system=nat,
        )
        real_i = idx_i.clamp(max=total_atoms - 1)
        real_j = idx_j.clamp(max=total_atoms - 1)
        radii = pair_radii_of(
            numbers_flat[real_i],
            numbers_flat[real_j],
            rvdw,
            r4r2,
            rdamp=rdamp,
        )
        c6_pair = _pair_c6(c6, real_i, real_j, nat)
        if c8 is not None:
            c8_pair = _pair_c6(c8, real_i, real_j, nat)
            qq = torch.where(
                c6_pair != 0, c8_pair / nonzero(c6_pair), radii.qq
            )
            radii = radii._replace(qq=qq)
        return SlotValues(distance_squared, c6_pair, radii)

    return values


def _pair_c6(
    c6: Tensor | Any, idx_i: Tensor, idx_j: Tensor, nat: int
) -> Tensor:
    """
    The C6 coefficients of the pairs ``(idx_i, idx_j)`` of a list, in its flat
    numbering of the atoms of a batch.
    """
    if hasattr(c6, "pair"):
        return c6.pair(idx_i, idx_j)
    return c6.reshape(-1, nat)[idx_i, idx_j % nat]
