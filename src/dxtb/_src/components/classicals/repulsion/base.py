# This file is part of dxtb.
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
Classical repulsion energy contribution
=======================================

This module implements the classical repulsion energy term.

Note
----
Numbers-only parameter values are resolved into a frozen setup. Positions are
supplied explicitly to ``get_energy`` and are never stored in that setup.

Example
-------

.. code-block:: python

    import torch
    from dxtb import IndexHelper
    from dxtb.classical import new_repulsion
    from dxtb import GFN1_XTB

    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, 0.00000000000000, 0.00000000000000],
        [1.61768389755830, 1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [1.61768389755830, -1.61768389755830, 1.61768389755830],
        [-1.61768389755830, 1.61768389755830, 1.61768389755830],
    ])

    rep = new_repulsion(numbers, positions, GFN1_XTB)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    cache = rep.get_cache(numbers, ihelp)
    energy = rep.get_energy(positions, cache)

    print(energy.sum(-1))  # Output: tensor(0.0303)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import any_to_tensor

from dxtb import IndexHelper
from dxtb._src.constants import xtb
from dxtb._src.typing import Any, Tensor, override

from ..base import Classical

__all__ = [
    "BaseRepulsion",
    "RepulsionSetup",
    "repulsion_energy",
    "setup_repulsion",
]


@dataclass(frozen=True, eq=False)
class RepulsionSetup:
    """Numbers-only, parameter-derived values for repulsion evaluation."""

    arep: Tensor
    """Pair-specific screening parameters."""

    zeff: Tensor
    """Pair-specific effective nuclear-charge products."""

    kexp: Tensor
    """
    Scaling of the interatomic distance in the exponential damping function
    of the repulsion energy.
    """

    mask: Tensor
    """Mask for real atom pairs, excluding padding and the diagonal."""

    cutoff: Tensor
    """Real-space cutoff captured at setup."""


def setup_repulsion(
    numbers: Tensor,
    ihelp: IndexHelper,
    arep: Tensor,
    zeff: Tensor,
    kexp: Tensor,
    cutoff: Tensor,
    *,
    klight: Tensor | None = None,
    en: Tensor | None = None,
    enscale: Tensor | None = None,
) -> RepulsionSetup:
    """Resolve repulsion parameters and structural masks for one System."""
    dd = {"device": arep.device, "dtype": arep.dtype}
    atom_arep = ihelp.spread_uspecies_to_atom(arep)
    atom_zeff = ihelp.spread_uspecies_to_atom(zeff)
    atom_kexp = ihelp.spread_uspecies_to_atom(
        kexp.expand(torch.unique(numbers).shape)
    )
    mask = real_pairs(numbers, mask_diagonal=True)

    # Keep the established epsilon workaround for the pairwise geometric mean.
    eps = torch.finfo(atom_arep.dtype).tiny
    pair_arep = torch.where(
        mask,
        torch.sqrt(atom_arep.unsqueeze(-1) * atom_arep.unsqueeze(-2) + eps),
        torch.tensor(0.0, **dd),
    )
    if en is not None and enscale is not None:
        atom_en = ihelp.spread_uspecies_to_atom(en)
        den2 = (atom_en.unsqueeze(-1) - atom_en.unsqueeze(-2)) ** 2
        pair_arep = pair_arep * (1.0 + (0.01 * den2 + 0.01 * den2**2) * enscale)

    pair_zeff = atom_zeff.unsqueeze(-1) * atom_zeff.unsqueeze(-2) * mask
    pair_kexp = (
        atom_kexp.unsqueeze(-1)
        * atom_kexp.new_ones(atom_kexp.shape).unsqueeze(-2)
        * mask
    )
    if klight is not None:
        light_mask = ~real_pairs(numbers <= 2)
        pair_kexp = torch.where(light_mask, pair_kexp, klight) * mask

    # Keep the compatibility object's mutable cutoff tensor from aliasing
    # System setup while preserving a possible autograd edge.
    return RepulsionSetup(pair_arep, pair_zeff, pair_kexp, mask, cutoff.clone())


class BaseRepulsion(Classical):
    """
    Representation of the classical repulsion.
    """

    arep: Tensor
    """Atom-specific screening parameters for unique species."""

    zeff: Tensor
    """Effective nuclear charges for unique species."""

    kexp: Tensor
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy.
    """

    klight: Tensor | None
    """
    Scaling of the interatomic distance in the exponential damping function of
    the repulsion energy for light elements, i.e., H and He (only GFN2).
    """

    en: Tensor | None
    """Element-specific electronegativities used by optional EN scaling."""

    enscale: Tensor | None
    """Optional electronegativity-difference scaling of pair screening."""

    cutoff: Tensor | float | int
    """
    Real space cutoff for repulsion interactions.

    :default: :data:`xtb.DEFAULT_REPULSION_CUTOFF`
    """

    __slots__ = [
        "arep",
        "zeff",
        "kexp",
        "klight",
        "en",
        "enscale",
        "cutoff",
    ]

    def __init__(
        self,
        arep: Tensor,
        zeff: Tensor,
        kexp: Tensor,
        klight: Tensor | None = None,
        cutoff: Tensor | float | int = xtb.DEFAULT_REPULSION_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        en: Tensor | None = None,
        enscale: Tensor | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.arep = arep.to(**self.dd)
        self.zeff = zeff.to(**self.dd)
        self.kexp = kexp.to(**self.dd)
        self.cutoff = any_to_tensor(cutoff, **self.dd)
        self.klight = None if klight is None else klight.to(**self.dd)
        self.en = None if en is None else en.to(**self.dd)
        self.enscale = None if enscale is None else enscale.to(**self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> RepulsionSetup:
        """
        Create numbers-only setup values for energy and gradient calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        RepulsionSetup
            Frozen setup for repulsion.

        Note
        ----
        This compatibility method returns a fresh setup. It does not store or
        reuse evaluation state on the Repulsion object.
        """
        if ihelp is None:
            raise ValueError("IndexHelper must be passed for repulsion.")

        return setup_repulsion(
            numbers,
            ihelp,
            self.arep,
            self.zeff,
            self.kexp,
            self.cutoff,
            klight=self.klight,
            en=self.en,
            enscale=self.enscale,
        )

    @abstractmethod
    def get_energy(
        self, positions: Tensor, cache: RepulsionSetup, **kwargs: Any
    ) -> Tensor:
        """
        Get repulsion energy.

        Parameters
        ----------
        cache : Repulsion.Cache
            Cache for repulsion.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        atom_resolved : bool
            Whether to return atom-resolved energy (True) or full matrix
            (False).

        Returns
        -------
        Tensor
            (Atom-resolved) repulsion energy.
        """


def repulsion_energy(
    positions: Tensor,
    mask: Tensor,
    arep: Tensor,
    kexp: Tensor,
    zeff: Tensor,
    cutoff: Tensor | float | int = xtb.DEFAULT_REPULSION_CUTOFF,
) -> Tensor:
    """
    Clasical repulsion energy.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    mask : Tensor
        Mask for padding.
    arep : Tensor
        Atom-specific screening parameters.
    kexp : Tensor
        Scaling of the interatomic distance in the exponential damping function
        of the repulsion energy.
    zeff : Tensor
        Effective nuclear charges.
    cutoff : Tensor | float | int, optional
        Real-space cutoff. Defaults to :data:`xtb.DEFAULT_REPULSION_CUTOFF`.

    Returns
    -------
    Tensor
        Atom-resolved repulsion energy
    """
    eps = torch.tensor(
        torch.finfo(positions.dtype).eps,
        dtype=positions.dtype,
        device=positions.device,
    )
    zero = torch.tensor(
        0.0,
        dtype=positions.dtype,
        device=positions.device,
    )
    _cutoff = any_to_tensor(
        cutoff,
        dtype=positions.dtype,
        device=positions.device,
    )

    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        eps,
    )

    # Eq.13: R_AB ** k_f
    r1k = torch.pow(distances, kexp)

    # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
    exp_term = torch.exp(-arep * r1k)

    # Eq.13: repulsion energy
    return torch.where(
        mask * (distances <= _cutoff),
        storch.safe_divide(zeff * exp_term, distances),
        zero,
    )
