# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
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
polarizabilities :math:`\alpha_{dd}(i\omega)`.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from tad_mctc import storch
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.batch import real_pairs, real_triples
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import _periodic_images
from tad_mctc.neighbor import gather_index
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.neighbor.triples import TripleList
from tad_mctc.tools import is_compiling
from tad_mctc.typing import DD, TableFunction, Tensor
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from . import defaults
from .cutoff import smooth_cutoff
from .damping.base import ThreeBodyDamping, TripleData
from .damping.param import DampingParam
from .damping.zero import ZeroThreeBodyD3
from .pairvalues import PairRadii, pair_radii, slot_values

__all__ = [
    "_periodic_tile_shape",
    "atm_term",
    "dispersion_atm",
    "dispersion_atm_periodic",
    "dispersion_atm_sparse",
]


def _periodic_tile_shape(n_leg: int, max_triples: int) -> tuple[int, int]:
    """
    Bounded 2D tile shape ``(row_block, col_block)`` for ``n_leg`` candidate
    periodic legs such that ``row_block * col_block <= max_triples``.

    Parameters
    ----------
    n_leg : int
        Number of candidate periodic legs ``(j, shift)`` for one central atom.
    max_triples : int
        Maximum number of candidate leg pairs per tile (``>= 1``).

    Returns
    -------
    tuple[int, int]
        ``(row_block, col_block)`` satisfying
        ``1 <= row_block <= n_leg``, ``1 <= col_block <= n_leg``, and
        ``row_block * col_block <= max_triples`` when ``n_leg >= 1``, or
        ``(0, 0)`` when ``n_leg < 1``.
    """
    if max_triples < 1:
        raise ValueError(f"'max_triples' must be >= 1, got {max_triples}.")
    if n_leg < 1:
        return 0, 0
    row_block = min(n_leg, max(1, math.isqrt(max_triples)))
    col_block = min(n_leg, max(1, max_triples // row_block))
    return row_block, col_block


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
    cutoff: float | Tensor,
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
    if isinstance(cutoff2, Tensor) and cutoff2.ndim > 0:
        while cutoff2.ndim < r2ij.ndim:
            cutoff2 = cutoff2.unsqueeze(-1)
    eps = torch.finfo(r2ij.dtype).eps
    mask = (
        (r2ij <= cutoff2)
        & (r2ik <= cutoff2)
        & (r2jk <= cutoff2)
        & (r2ij >= eps)
        & (r2ik >= eps)
        & (r2jk >= eps)
    )
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
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    rdamp: Tensor | None = None,
    c9: Tensor | None = None,
    adiw: Tensor | None = None,
    gl_weights: Tensor | None = None,
    cutoff: float | Tensor = defaults.D3_DISP3_CUTOFF,
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
        structure,
        rvdw_table,
        r4r2_table,
        rdamp_table=rdamp_table,
        rvdw=rvdw,
        r4r2=r4r2,
        rdamp=rdamp,
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

    Candidate leg pairs ``(j, shift_j) x (k, shift_k)`` around each central
    atom ``i`` are evaluated in 2D tiles of shape ``(tile_rows, tile_cols)``
    satisfying ``tile_rows * tile_cols <= max_triples`` so no full
    ``(n_leg, n_leg)`` intermediate is materialized when ``n_leg**2 > max_triples``.
    """
    if max_triples < 1:
        raise ValueError(f"'max_triples' must be >= 1, got {max_triples}.")

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
    valid = images.valid.reshape(-1, nat, nat, n_shift)
    valid = valid.expand(n_system, nat, nat, n_shift)

    c6_systems = c6.reshape(-1, nat, nat).expand(n_system, nat, nat)
    radii_systems = PairRadii(
        qq=radii.qq.reshape(-1, nat, nat).expand(n_system, nat, nat),
        rdamp=radii.rdamp.reshape(-1, nat, nat).expand(n_system, nat, nat),
        rvdw=radii.rvdw.reshape(-1, nat, nat).expand(n_system, nat, nat),
        znum=radii.znum.reshape(-1, nat, nat).expand(n_system, nat, nat),
    )
    adiw_systems = (
        adiw.reshape(-1, nat, adiw.shape[-1]).expand(
            n_system, nat, adiw.shape[-1]
        )
        if adiw is not None
        else None
    )

    cutoff2 = cutoff * cutoff
    eps = torch.finfo(positions.dtype).eps

    use_static_legs = (
        is_compiling()
        or is_functorch_tensor(positions)
        or is_functorch_tensor(lattice)
        or is_functorch_tensor(c6)
        or torch._C._functorch.get_interpreter_stack() is not None
    )
    static_atoms = (
        torch.arange(nat, device=positions.device)
        .unsqueeze(1)
        .expand(nat, n_shift)
        .reshape(-1)
        if use_static_legs
        else None
    )

    energies = []
    for system in range(n_system):
        c6_sys = c6_systems[system]
        radii_sys = PairRadii(
            qq=radii_systems.qq[system],
            rdamp=radii_systems.rdamp[system],
            rvdw=radii_systems.rvdw[system],
            znum=radii_systems.znum[system],
        )
        adiw_sys = adiw_systems[system] if adiw_systems is not None else None

        for i in range(nat):
            vectors = (
                folded[system].unsqueeze(1)
                - folded[system, i]
                + translations[system].unsqueeze(0)
            )
            r2 = vectors.pow(2).sum(-1)
            keep = valid[system, i] & (r2 <= cutoff2) & (r2 >= eps)

            if use_static_legs:
                assert static_atoms is not None
                atoms = static_atoms
                legs = vectors.reshape(-1, 3)
                c6_i = c6_sys[i].unsqueeze(1).expand(nat, n_shift).reshape(-1)
                radii_i = PairRadii(
                    qq=radii_sys.qq[i]
                    .unsqueeze(1)
                    .expand(nat, n_shift)
                    .reshape(-1),
                    rdamp=radii_sys.rdamp[i]
                    .unsqueeze(1)
                    .expand(nat, n_shift)
                    .reshape(-1),
                    rvdw=radii_sys.rvdw[i]
                    .unsqueeze(1)
                    .expand(nat, n_shift)
                    .reshape(-1),
                    znum=radii_sys.znum[i]
                    .unsqueeze(1)
                    .expand(nat, n_shift)
                    .reshape(-1),
                )
                keep_legs: Tensor | None = keep.reshape(-1)
            else:
                with torch.no_grad():
                    atoms, shift_idx = torch.nonzero(keep, as_tuple=True)

                legs = vectors.reshape(-1, 3).index_select(
                    0, atoms * n_shift + shift_idx
                )
                c6_i = c6_sys[i].index_select(0, atoms)
                radii_i = PairRadii(
                    qq=radii_sys.qq[i].index_select(0, atoms),
                    rdamp=radii_sys.rdamp[i].index_select(0, atoms),
                    rvdw=radii_sys.rvdw[i].index_select(0, atoms),
                    znum=radii_sys.znum[i].index_select(0, atoms),
                )
                keep_legs = None

            n_leg = atoms.shape[0]
            if n_leg == 0:
                energy = (
                    folded[system, i].sum()
                    + translations[system].sum()
                    + c6_sys[i].sum()
                ) * 0.0
                energies.append(energy)
                continue

            row_block, col_block = _periodic_tile_shape(n_leg, max_triples)
            energy = folded.new_zeros(())
            for row_start in range(0, n_leg, row_block):
                rows = slice(row_start, min(row_start + row_block, n_leg))
                legs_a = legs[rows]
                atoms_a = atoms[rows]
                c6_a = c6_i[rows]
                radii_a = PairRadii(
                    qq=radii_i.qq[rows],
                    rdamp=radii_i.rdamp[rows],
                    rvdw=radii_i.rvdw[rows],
                    znum=radii_i.znum[rows],
                )
                keep_rows = keep_legs[rows] if keep_legs is not None else None
                for col_start in range(0, n_leg, col_block):
                    cols = slice(col_start, min(col_start + col_block, n_leg))
                    legs_b = legs[cols]
                    atoms_b = atoms[cols]
                    c6_b = c6_i[cols]
                    radii_b = PairRadii(
                        qq=radii_i.qq[cols],
                        rdamp=radii_i.rdamp[cols],
                        rvdw=radii_i.rvdw[cols],
                        znum=radii_i.znum[cols],
                    )
                    keep_cols = (
                        keep_legs[cols] if keep_legs is not None else None
                    )
                    args = (
                        legs_a,
                        legs_b,
                        atoms_a,
                        atoms_b,
                        c6_a,
                        c6_b,
                        radii_a,
                        radii_b,
                        c6_sys,
                        radii_sys,
                        damping,
                        param,
                        cutoff,
                        width,
                        i,
                        adiw_sys,
                        gl_weights,
                        keep_rows,
                        keep_cols,
                    )
                    if checkpoint:
                        energy = energy + _torch_checkpoint(
                            _periodic_block_energy, *args, use_reentrant=False
                        )
                    else:
                        energy = energy + _periodic_block_energy(*args)

            energies.append(energy)

    return torch.stack(energies).reshape(
        torch.broadcast_shapes(numbers.shape, positions.shape[:-1])
    )


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
    keep_a: Tensor | None = None,
    keep_b: Tensor | None = None,
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

    eps = torch.finfo(vec_a.dtype).eps
    valid_jk = (r2ij >= eps) & (r2ik >= eps) & (r2jk >= eps)
    if keep_a is not None:
        valid_jk = valid_jk & keep_a.unsqueeze(1)
    if keep_b is not None:
        valid_jk = valid_jk & keep_b.unsqueeze(0)

    if adiw_sys is not None and gl_weights is not None:
        w = gl_weights.to(device=adiw_sys.device, dtype=adiw_sys.dtype)
        ai = adiw_sys[atom_i]  # (ngrid,)
        aj = adiw_sys.index_select(0, atom_a)  # (A, ngrid)
        ak = adiw_sys.index_select(0, atom_b)  # (B, ngrid)
        c9 = (3.0 / math.pi) * torch.einsum(
            "w,aw,bw,w->ab", ai, aj, ak, w
        )
    else:
        c6_prod = torch.where(
            valid_jk,
            torch.abs(c6_a.unsqueeze(1) * c6_b.unsqueeze(0) * c6_jk),
            torch.zeros_like(c6_jk),
        )
        c9 = storch.safe_sqrt(c6_prod)

    rdamp_sides = (
        radii_a.rdamp.unsqueeze(1),
        radii_b.rdamp.unsqueeze(0),
        radii.rdamp[jk],
    )
    rvdw_sides = (
        radii_a.rvdw.unsqueeze(1),
        radii_b.rvdw.unsqueeze(0),
        radii.rvdw[jk],
    )

    term = atm_term(
        r2ij,
        r2ik,
        r2jk,
        rdamp_sides,
        rvdw_sides,
        damping,
        param,
        cutoff,
        width,
        valid_jk,
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
    size = max(1, min(max_triples, max(total, 1)))

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
