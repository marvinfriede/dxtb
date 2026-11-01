# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
"""
Two-body term
=============

The two-body dispersion energy, one function per way of enumerating the
pairs:

- :func:`_dispersion2_dense`: all pairs of a molecule, or of a cell with every
  periodic image of its shifts within the cutoff;
- :func:`dispersion2_sparse`: the pairs of a neighbour list, for a molecule
  or a cell.
"""
from __future__ import annotations

from typing import Any

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import _periodic_images, sum_over_neighborlist
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.typing import DD, TableFunction, Tensor

from . import defaults
from .cutoff import smooth_cutoff
from .damping.base import PairData, TwoBodyDamping
from .damping.param import DampingParam
from .pairvalues import PairRadii, nonzero, pair_radii, slot_values

__all__ = ["_dispersion2_dense", "dispersion2_sparse", "pair_term"]


def pair_term(
    distances: Tensor,
    keep: Tensor,
    c6: Tensor,
    radii: PairRadii,
    damping: TwoBodyDamping,
    param: DampingParam,
    cutoff: float,
    width: float,
) -> Tensor:
    """
    Damping and switch of pairs, without the factor ``-c6`` (and the half of
    each atom): the energy of a pair is ``-c6`` times this. Zero for a pair
    that is not kept.
    """
    pairs = PairData(
        distances, radii.qq, c6, radii.rvdw, radii.znum, radii.rdamp
    )
    kernel = damping(pairs, param)
    if width > 0.0:
        kernel = smooth_cutoff(distances, cutoff, width) * kernel
    return torch.where(keep, kernel, torch.zeros_like(kernel))


def _dispersion2_dense(
    structure: Structure,
    param: DampingParam,
    c6: Tensor,
    shifts: PeriodicShifts | None,
    *,
    damping: TwoBodyDamping,
    rvdw_table: Tensor | TableFunction | None,
    r4r2_table: Tensor | TableFunction | None,
    rdamp_table: Tensor | TableFunction | None = None,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    c8: Tensor | None = None,
    cutoff: float | Tensor,
    width: float,
    as_matrix: bool = False,
) -> Tensor:
    """
    Two-body energy over all pairs of a molecule, ``(..., nat, nat)``, or of
    a cell with every periodic image of its `shifts`,
    ``(..., nat, nat, n_shift)``.
    """
    if shifts is not None:
        shifts.check_compatible(structure, cutoff)
    elif structure.lattice is not None:
        raise ValueError(
            "'structure' is a periodic cell, but no periodic image shifts "
            "were given (see 'DensePairs.build')."
        )

    numbers, positions = structure.numbers, structure.positions

    radii = pair_radii(
        structure,
        rvdw_table,
        r4r2_table,
        rdamp_table=rdamp_table,
        rvdw=rvdw,
        r4r2=r4r2,
    )
    if c8 is not None:
        qq = torch.where(c6 != 0, c8 / nonzero(c6), radii.qq)
        radii = radii._replace(qq=qq)
    c6_pairs = c6

    if shifts is None:
        distances, keep = _molecular_distances(numbers, positions, cutoff)
    else:
        distances, keep = _periodic_distances(structure, shifts, cutoff)

        radii = radii.map(lambda value: value.unsqueeze(-1))
        c6_pairs = c6.unsqueeze(-1)

    kernel = pair_term(
        distances, keep, c6_pairs, radii, damping, param, cutoff, width
    )

    if shifts is not None:
        kernel = kernel.sum(dim=-1)

    if as_matrix:
        return -c6 * kernel
    return -0.5 * torch.sum(c6 * kernel, dim=-1)


def _molecular_distances(
    numbers: Tensor, positions: Tensor, cutoff: float | Tensor
) -> tuple[Tensor, Tensor]:
    """
    Distances between all atoms of a molecule, ``(..., nat, nat)``, and
    which of them are pairs within `cutoff`.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(1.0, **dd),
    )

    cut = cutoff
    if isinstance(cut, Tensor) and cut.ndim > 0:
        while cut.ndim < distances.ndim:
            cut = cut.unsqueeze(-1)
    keep = mask * (distances <= cut)
    return distances, keep


def _periodic_distances(
    cell: Structure, shifts: PeriodicShifts, cutoff: float
) -> tuple[Tensor, Tensor]:
    """
    Distances from every atom of a cell to every periodic image of every
    atom, ``(..., nat, nat, n_shift)``, and which of them are pairs within
    `cutoff`.
    """
    assert cell.lattice is not None and cell.periodic is not None

    images = _periodic_images(
        cell.numbers, cell.positions, cell.lattice, shifts.shifts, cell.periodic
    )
    assert images.translations is not None
    positions = images.positions

    pair_vectors = positions.unsqueeze(-3) - positions.unsqueeze(-2)

    image_translations = images.translations[..., None, None, :, :]
    image_vectors = pair_vectors.unsqueeze(-2) + image_translations
    distance_squared = torch.sum(image_vectors * image_vectors, dim=-1)

    keep = images.valid & (distance_squared <= cutoff * cutoff)

    distances = torch.sqrt(torch.where(keep, distance_squared, 1.0))

    return distances, keep


def dispersion2_sparse(
    structure: Structure,
    param: DampingParam,
    c6: Tensor | Any,
    nbl: NeighborList,
    *,
    damping: TwoBodyDamping,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    c8: Tensor | Any | None = None,
    cutoff: float = defaults.D3_DISP2_CUTOFF,
    width: float = defaults.D3_DISP2_WIDTH,
    checkpoint: bool = False,
) -> Tensor:
    """
    Two-body dispersion energy summed over a neighbour list.
    """
    nbl.check_compatible(structure, cutoff)

    values_of = slot_values(
        structure,
        c6,
        rvdw_table,
        r4r2_table,
        rdamp_table=rdamp_table,
        c8=c8,
    )

    def pair_contributions(
        idx_i: Tensor, idx_j: Tensor, mask: Tensor, shift: Tensor
    ) -> tuple[Tensor, Tensor]:
        distance_squared, c6_pair, radii = values_of(idx_i, idx_j, shift)

        mask = mask & (distance_squared <= cutoff * cutoff)
        distances = torch.sqrt(torch.where(mask, distance_squared, 1.0))

        kernel = pair_term(
            distances, mask, c6_pair, radii, damping, param, cutoff, width
        )
        contribution = -0.5 * c6_pair * kernel
        return contribution, contribution

    energy = sum_over_neighborlist(
        nbl,
        pair_contributions,
        structure.positions.reshape(-1, 3),
        mode="recompute" if checkpoint else "graph",
    )
    return energy.reshape(structure.numbers.shape)
