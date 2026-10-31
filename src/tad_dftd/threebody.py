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
r"""
Three-body (Axilrod-Teller-Muto) term
=====================================

The three-body dispersion energy, one function per way of enumerating the
triples:

- :func:`dispersion_atm`: all triples of a molecule, ``O(nat**3)`` memory;
- :func:`dispersion_atm_periodic`: a cell, every atom of the cell with all
  pairs of periodic images within the cutoff;
- :func:`dispersion_atm_sparse`: the triangles of a neighbour list, for a
  molecule or a cell.

Supports both geometric-mean :math:`C_9 = \sqrt{|C_{6,ij} C_{6,ik} C_{6,jk}|}`
(used by D3, D3S, D4, D4S, D4Srev) and Casimir-Polder quadrature of dynamic
polarizabilities :math:`\alpha_{dd}(i\omega)` (used by D5).
"""
from __future__ import annotations

import math
from typing import Any

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import _periodic_images
from tad_mctc.neighbor import gather_index
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.neighbor.triples import TripleList
from tad_mctc.typing import DD, TableFunction, Tensor
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from . import defaults
from .cutoff import smooth_cutoff
from .damping.base import ThreeBodyDamping, TripleData
from .damping.param import DampingParam
from .damping.zero import ZeroThreeBodyD3
from .pairvalues import PairRadii, pair_radii, slot_values

__all__ = [
    "atm_term",
    "dispersion_atm",
    "dispersion_atm_periodic",
    "dispersion_atm_sparse",
]


def _sides(table: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The pairs ij, ik and jk of all triples ``(i, j, k)`` of a pair table
    ``(..., nat, nat)``, as the three sides of a broadcast."""
    return table.unsqueeze(-1), table.unsqueeze(-2), table.unsqueeze(-3)


def atm_term(
    r2ij: Tensor,
    r2ik: Tensor,
    r2jk: Tensor,
    rdamp: tuple[Tensor, Tensor, Tensor],
    rvdw: tuple[Tensor, Tensor, Tensor],
    damping: ThreeBodyDamping,
    param: DampingParam,
    cutoff: float,
    width: float,
    valid: Tensor | None = None,
) -> Tensor:
    r"""
    Angular part, damping (with `s9`) and switch of triples, without the
    :math:`C_9`, from their squared side lengths and the two radii a
    three-body damping reads of their pairs ij, ik and jk (`rdamp` and
    `rvdw`), which must be positive. Zero for a triple that is not `valid` or
    has a side beyond `cutoff`.
    """
    cutoff2 = cutoff * cutoff
    mask = (r2ij <= cutoff2) & (r2ik <= cutoff2) & (r2jk <= cutoff2)
    if valid is not None:
        mask = mask & valid

    one = torch.ones_like(r2ij)
    r2ij = torch.where(mask, r2ij, one)
    r2ik = torch.where(mask, r2ik, one)
    r2jk = torch.where(mask, r2jk, one)

    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    r3 = r1 * r2
    r5 = r2 * r3

    triples = TripleData(r1, r2ij, r2ik, r2jk, *rdamp, *rvdw)
    fdamp = damping(triples, param)

    s = (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik)
    ang = torch.where(mask, 0.375 * s / r5 + 1.0 / r3, torch.zeros_like(r5))

    if width > 0.0:
        ang = (
            ang
            * smooth_cutoff(torch.sqrt(r2ij), cutoff, width)
            * smooth_cutoff(torch.sqrt(r2ik), cutoff, width)
            * smooth_cutoff(torch.sqrt(r2jk), cutoff, width)
        )

    return ang * fdamp


def dispersion_atm(
    structure: Structure,
    c6: Tensor,
    param: DampingParam,
    *,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    c9: Tensor | None = None,
    adiw: Tensor | None = None,
    gl_weights: Tensor | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    width: float = defaults.D3_DISP3_WIDTH,
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term of a molecule, over all triples of
    atoms.
    """
    if structure.lattice is not None:
        raise ValueError(
            "'structure' is a periodic cell; use 'dispersion_atm_periodic' "
            "with its periodic image shifts (see 'DensePairs.build')."
        )

    numbers, positions = structure.numbers, structure.positions
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    radii = pair_radii(
        structure, rvdw_table, r4r2_table, rdamp_table=rdamp_table
    )

    mask_pairs = real_pairs(numbers, mask_diagonal=True)
    mask_triples = real_triples(numbers, mask_self=True)

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)

    if c9 is None:
        if adiw is not None and gl_weights is not None:
            w = gl_weights.to(device=adiw.device, dtype=adiw.dtype)
            c9 = (3.0 / math.pi) * torch.einsum(
                "...iw,...jw,...kw,w->...ijk", adiw, adiw, adiw, w
            )
        else:
            c9 = storch.safe_sqrt(
                torch.abs(
                    c6.unsqueeze(-1) * c6.unsqueeze(-2) * c6.unsqueeze(-3)
                )
            )

    distances = torch.pow(
        torch.where(
            mask_pairs,
            storch.cdist(positions, positions, p=2),
            eps,
        ),
        2.0,
    )

    energy = atm_term(
        distances.unsqueeze(-1),
        distances.unsqueeze(-2),
        distances.unsqueeze(-3),
        _sides(radii.rdamp),
        _sides(radii.rvdw),
        damping,
        param,
        cutoff,
        width,
        mask_triples,
    )
    return torch.sum(energy * c9, dim=(-2, -1)) / 6.0


def dispersion_atm_periodic(
    structure: Structure,
    c6: Tensor,
    param: DampingParam,
    shifts: PeriodicShifts,
    *,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    adiw: Tensor | None = None,
    gl_weights: Tensor | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    width: float = defaults.D3_DISP3_WIDTH,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term of a periodic cell: every atom ``i``
    of the cell with every pair of atoms or images of atoms within `cutoff`
    of it (also images of ``i`` itself).
    """
    numbers, positions = structure.numbers, structure.positions
    lattice, periodic = structure.lattice, structure.periodic
    if lattice is None or periodic is None:
        raise ValueError("'structure' has no lattice; use 'dispersion_atm'.")

    shifts.check_compatible(structure, cutoff)

    radii = pair_radii(
        structure, rvdw_table, r4r2_table, rdamp_table=rdamp_table
    )

    images = _periodic_images(
        numbers, positions, lattice, shifts.shifts, periodic
    )
    assert images.translations is not None

    nat, n_shift = positions.shape[-2], shifts.shifts.shape[0]
    folded = images.positions.reshape(-1, nat, 3)
    n_system = folded.shape[0]
    translations = images.translations.reshape(-1, n_shift, 3)
    translations = translations.expand(n_system, n_shift, 3)
    valid = images.valid.reshape(n_system, nat, nat, n_shift)
    valid = valid.expand(n_system, nat, nat, n_shift)

    c6_systems = c6.reshape(n_system, nat, nat)
    radii_systems = radii.map(lambda r: r.reshape(n_system, nat, nat))
    adiw_systems = (
        adiw.reshape(n_system, nat, -1)
        if adiw is not None
        else None
    )

    cutoff2 = cutoff * cutoff
    eps = torch.finfo(positions.dtype).eps
    zero = torch.zeros((), device=positions.device, dtype=positions.dtype)

    energies = []
    for system in range(n_system):
        c6_sys = c6_systems[system]
        radii_sys = radii_systems.map(lambda r: r[system])
        adiw_sys = adiw_systems[system] if adiw_systems is not None else None

        for i in range(nat):
            vectors = (
                folded[system].unsqueeze(1)
                - folded[system, i]
                + translations[system].unsqueeze(0)
            )
            r2 = vectors.pow(2).sum(-1)

            with torch.no_grad():
                keep = valid[system, i] & (r2 <= cutoff2) & (r2 >= eps)
                atoms, shift_idx = torch.nonzero(keep, as_tuple=True)

            legs = vectors.reshape(-1, 3).index_select(
                0, atoms * n_shift + shift_idx
            )
            c6_i = c6_sys[i].index_select(0, atoms)
            radii_i = radii_sys.map(lambda r: r[i].index_select(0, atoms))

            n_leg = atoms.shape[0]
            energy = zero
            block = max(1, max_triples // max(n_leg, 1))
            for start in range(0, max(n_leg, 1), block):
                rows = slice(start, start + block)
                args = (
                    legs[rows],
                    legs,
                    atoms[rows],
                    atoms,
                    c6_i[rows],
                    c6_i,
                    radii_i.map(lambda r: r[rows]),
                    radii_i,
                    c6_sys,
                    radii_sys,
                    damping,
                    param,
                    cutoff,
                    width,
                    i,
                    adiw_sys,
                    gl_weights,
                )
                if checkpoint:
                    energy = energy + _torch_checkpoint(
                        _periodic_block_energy, *args, use_reentrant=False
                    )
                else:
                    energy = energy + _periodic_block_energy(*args)

            energies.append(energy)

    return torch.stack(energies).reshape(numbers.shape)


def _periodic_block_energy(
    vec_a: Tensor,
    vec_b: Tensor,
    atom_a: Tensor,
    atom_b: Tensor,
    c6_a: Tensor,
    c6_b: Tensor,
    radii_a: PairRadii,
    radii_b: PairRadii,
    c6: Tensor,
    radii: PairRadii,
    damping: ThreeBodyDamping,
    param: DampingParam,
    cutoff: float,
    width: float,
    atom_i: int = 0,
    adiw_sys: Tensor | None = None,
    gl_weights: Tensor | None = None,
) -> Tensor:
    """
    Sum of the ATM energy of the triples of one atom ``i`` of the cell with
    each of the ``A`` images in ``a`` and each of the ``B`` in ``b``, over
    the ``A * B`` combinations, divided by six.
    """
    vec_jk = vec_b.unsqueeze(0) - vec_a.unsqueeze(1)

    r2ij = vec_a.pow(2).sum(-1).unsqueeze(1)
    r2ik = vec_b.pow(2).sum(-1).unsqueeze(0)
    r2jk = vec_jk.pow(2).sum(-1)

    jk = (atom_a.unsqueeze(1), atom_b.unsqueeze(0))
    c6_jk = c6[jk]

    def sides(
        of_a: Tensor, of_b: Tensor, table: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return of_a.unsqueeze(1), of_b.unsqueeze(0), table[jk]

    if adiw_sys is not None and gl_weights is not None:
        w = gl_weights.to(device=adiw_sys.device, dtype=adiw_sys.dtype)
        ai = adiw_sys[atom_i]  # (ngrid,)
        aj = adiw_sys.index_select(0, atom_a)  # (A, ngrid)
        ak = adiw_sys.index_select(0, atom_b)  # (B, ngrid)
        c9 = (3.0 / math.pi) * torch.einsum(
            "w,aw,bw,w->ab", ai, aj, ak, w
        )
    else:
        c9 = storch.safe_sqrt(
            torch.abs(c6_a.unsqueeze(1) * c6_b.unsqueeze(0) * c6_jk)
        )

    eps = torch.finfo(vec_a.dtype).eps
    term = atm_term(
        r2ij,
        r2ik,
        r2jk,
        sides(radii_a.rdamp, radii_b.rdamp, radii.rdamp),
        sides(radii_a.rvdw, radii_b.rvdw, radii.rvdw),
        damping,
        param,
        cutoff,
        width,
        r2jk >= eps,
    )

    return torch.sum(term * c9) / 6.0


def dispersion_atm_sparse(
    structure: Structure,
    c6: Tensor | Any,
    param: DampingParam,
    nbl: NeighborList | TripleList,
    *,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    adiw: Tensor | None = None,
    gl_weights: Tensor | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    width: float = defaults.D3_DISP3_WIDTH,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term over the triangles of a neighbour
    list (see :class:`~tad_mctc.neighbor.triples.TripleList`), for a
    molecule or a cell, in chunks of at most `max_triples` triples.
    """
    if isinstance(nbl, TripleList):
        triples = nbl
        triples.check_compatible(structure, cutoff)
    else:
        nbl.check_compatible(structure, cutoff)
        triples = TripleList.from_neighborlist(nbl, chunk_size=max_triples)

    numbers, positions = structure.numbers, structure.positions
    flat_positions = positions.reshape(-1, 3)
    total_atoms = flat_positions.shape[0]

    pairs = triples.pairs
    r2, c6_pairs, radii = slot_values(
        structure, c6, rvdw_table, r4r2_table, rdamp_table=rdamp_table
    )(gather_index(pairs.idx_i), gather_index(pairs.idx_j), pairs.shift)

    flat_adiw: Tensor | None = None
    if adiw is not None and gl_weights is not None:
        flat_adiw = adiw.reshape(total_atoms, -1)

    total = triples.idx_i.shape[0]
    size = max(max_triples, 1)
    if torch.compiler.is_compiling():
        size = max(total, 1)

    energy = flat_positions.new_zeros(total_atoms)
    for start in range(0, max(total, 1), size):
        rows = slice(start, start + size)
        idx_i = gather_index(triples.idx_i[rows])
        idx_j = gather_index(triples.idx_j[rows])
        idx_k = gather_index(triples.idx_k[rows])
        args = (
            r2,
            c6_pairs,
            radii,
            triples.side_ij[rows],
            triples.side_jk[rows],
            triples.side_ik[rows],
            damping,
            param,
            cutoff,
            width,
            idx_i,
            idx_j,
            idx_k,
            flat_adiw,
            gl_weights,
        )
        if checkpoint:
            share = _torch_checkpoint(
                _triple_energy, *args, use_reentrant=False
            )
        else:
            share = _triple_energy(*args)

        energy = energy.index_add(0, idx_i, share)
        energy = energy.index_add(0, idx_j, share)
        energy = energy.index_add(0, idx_k, share)

    return energy.reshape(numbers.shape)


def _triple_energy(
    r2: Tensor,
    c6: Tensor,
    radii: PairRadii,
    side_ij: Tensor,
    side_jk: Tensor,
    side_ik: Tensor,
    damping: ThreeBodyDamping,
    param: DampingParam,
    cutoff: float,
    width: float,
    idx_i: Tensor | None = None,
    idx_j: Tensor | None = None,
    idx_k: Tensor | None = None,
    flat_adiw: Tensor | None = None,
    gl_weights: Tensor | None = None,
) -> Tensor:
    """
    A third of the ATM energy of each triple, which the caller adds to each
    of its three atoms.
    """

    def of(table: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            table.index_select(0, side_ij),
            table.index_select(0, side_ik),
            table.index_select(0, side_jk),
        )

    r2ij, r2ik, r2jk = of(r2)
    term = atm_term(
        r2ij,
        r2ik,
        r2jk,
        of(radii.rdamp),
        of(radii.rvdw),
        damping,
        param,
        cutoff,
        width,
    )
    if (
        flat_adiw is not None
        and gl_weights is not None
        and idx_i is not None
        and idx_j is not None
        and idx_k is not None
    ):
        w = gl_weights.to(device=flat_adiw.device, dtype=flat_adiw.dtype)
        ai = flat_adiw.index_select(0, idx_i)
        aj = flat_adiw.index_select(0, idx_j)
        ak = flat_adiw.index_select(0, idx_k)
        c9 = (3.0 / math.pi) * torch.sum(ai * aj * ak * w, dim=-1)
    else:
        c6ij, c6ik, c6jk = of(c6)
        c9 = storch.safe_sqrt(torch.abs(c6ij * c6ik * c6jk))

    return term * c9 / 3.0
