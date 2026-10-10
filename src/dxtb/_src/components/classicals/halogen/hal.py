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
"""Halogen bond correction using explicit setup and fixed-shape tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tad_mctc.batch import pack
from tad_mctc.data.radii import ATOMIC_RADII
from tad_mctc.typing import Any, Tensor, override

from dxtb import IndexHelper
from dxtb._src.constants import xtb

from ..base import Classical, ComponentCache

__all__ = ["Halogen", "HalogenSetup", "LABEL_HALOGEN", "halogen_energy"]


LABEL_HALOGEN = "Halogen"
"""Label for the :class:`.Halogen` component, coinciding with the class name."""


@dataclass(frozen=True, eq=False)
class HalogenSetup:
    """Numbers- and parameter-derived data for one halogen term."""

    numbers: Tensor
    xbond: Tensor
    atomic_radii: Tensor
    damp: Tensor
    cutoff: Tensor
    halogen_mask: Tensor
    base_mask: Tensor
    valid_atom_mask: Tensor
    pair_type_mask: Tensor
    neighbor_type_mask: Tensor


def setup_halogen(
    halogen: Halogen, numbers: Tensor, ihelp: IndexHelper
) -> HalogenSetup:
    """Gather element parameters and fixed-shape masks for halogen bonding."""
    xbond = ihelp.spread_uspecies_to_atom(halogen.bond_strength)
    halogen_mask = torch.zeros_like(numbers, dtype=torch.bool)
    for atomic_number in halogen.halogens:
        halogen_mask = halogen_mask | (numbers == atomic_number)
    base_mask = torch.zeros_like(numbers, dtype=torch.bool)
    for atomic_number in halogen.bases:
        base_mask = base_mask | (numbers == atomic_number)
    valid_atom_mask = numbers != 0
    return HalogenSetup(
        numbers=numbers,
        xbond=xbond,
        atomic_radii=ATOMIC_RADII(device=halogen.device, dtype=halogen.dtype)[
            numbers
        ]
        * halogen.rscale,
        damp=halogen.damp.clone(),
        cutoff=halogen.cutoff.clone(),
        halogen_mask=halogen_mask,
        base_mask=base_mask,
        valid_atom_mask=valid_atom_mask,
        pair_type_mask=halogen_mask.unsqueeze(-1) & base_mask.unsqueeze(-2),
        neighbor_type_mask=halogen_mask.unsqueeze(-1)
        & valid_atom_mask.unsqueeze(-2),
    )


def halogen_energy(setup: HalogenSetup, positions: Tensor) -> Tensor:
    """Evaluate atom-resolved halogen bonding with fixed-size tensor masks."""
    if setup.numbers.ndim != 1:
        raise ValueError("halogen_energy accepts one System at a time.")
    if positions.shape != (setup.numbers.numel(), 3):
        raise ValueError("positions must have shape (nat, 3).")
    if positions.device != setup.numbers.device:
        raise RuntimeError("Halogen setup and positions must share a device.")
    if positions.dtype != setup.atomic_radii.dtype:
        raise RuntimeError("Halogen setup and positions must share a dtype.")

    # delta[x, j] is the vector from halogen x to candidate base j.
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)
    r2 = torch.sum(delta * delta, dim=-1)

    # The former nearest-neighbor loop ignored padding and every zero-distance
    # candidate, including coincident atoms with different indices.
    neighbor_valid = setup.neighbor_type_mask & (r2 > 0.0)
    masked_r2 = torch.where(neighbor_valid, r2, torch.full_like(r2, torch.inf))
    nearest_index = torch.argmin(masked_r2, dim=-1)
    nearest_position = torch.gather(
        positions.unsqueeze(0).expand(positions.shape[0], -1, -1),
        1,
        nearest_index[:, None, None].expand(-1, 1, 3),
    ).squeeze(1)

    nearest_valid = neighbor_valid.any(dim=-1)
    dxk = nearest_position - positions
    d2xk = torch.sum(dxk * dxk, dim=-1)
    safe_d2xk = torch.where(nearest_valid, d2xk, torch.ones_like(d2xk))

    active_pair = setup.pair_type_mask & (r2 <= setup.cutoff.square())
    safe_d2xj = torch.where(active_pair, r2, torch.ones_like(r2))
    rxj = torch.sqrt(safe_d2xj)
    xy = torch.sqrt(safe_d2xk[:, None] * safe_d2xj)

    dkj = nearest_position[:, None, :] - positions[None, :, :]
    d2kj = torch.sum(dkj * dkj, dim=-1)
    cosine = (safe_d2xk[:, None] + safe_d2xj - d2kj) / xy

    r0xj = setup.atomic_radii[:, None] + setup.atomic_radii[None, :]
    lj6 = torch.pow(r0xj / rxj, 6.0)
    lj12 = torch.pow(lj6, 2.0)
    lj = (lj12 - setup.damp * lj6) / (1.0 + lj12)
    angle_damping = torch.pow(0.5 - 0.25 * cosine, 6.0)
    pair_energy = torch.where(
        active_pair,
        lj * angle_damping * setup.xbond[:, None],
        torch.zeros_like(lj),
    )
    return torch.sum(pair_energy, dim=-1)


class Halogen(Classical):
    """Representation of the halogen bond correction."""

    halogens: list[int]
    bases: list[int]
    damp: Tensor
    rscale: Tensor
    bond_strength: Tensor
    cutoff: Tensor

    __slots__ = ["damp", "rscale", "bond_strength", "cutoff"]

    def __init__(
        self,
        damp: Tensor,
        rscale: Tensor,
        bond_strength: Tensor,
        cutoff: Tensor | float | int = xtb.DEFAULT_XB_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.damp = damp.to(**self.dd)
        self.rscale = rscale.to(**self.dd)
        self.bond_strength = bond_strength.to(**self.dd)
        self.cutoff = torch.as_tensor(cutoff, **self.dd)
        self.halogens = [17, 35, 53, 85]
        self.bases = [7, 8, 15, 16]

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> HalogenSetup:
        """Build fresh halogen setup for the compatibility API."""
        if ihelp is None:
            raise ValueError("IndexHelper is required for halogen bonding.")
        return setup_halogen(self, numbers, ihelp)

    @override
    def get_energy(
        self, positions: Tensor, cache: ComponentCache, **_: Any
    ) -> Tensor:
        """Evaluate through setup; the legacy batch adapter loops structures."""
        if not isinstance(cache, HalogenSetup):
            raise TypeError(
                f"Setup in {self.label} is not of type 'HalogenSetup'."
            )
        if cache.numbers.ndim > 1:
            return pack(
                [
                    halogen_energy(_slice_setup(cache, batch), positions[batch])
                    for batch in range(cache.numbers.shape[0])
                ]
            )
        return halogen_energy(cache, positions)

    def update(self, **kwargs: Any) -> None:
        """Reject in-place updates for exact setup-migrated halogen terms."""
        if type(self) is Halogen:
            raise RuntimeError(
                "Halogen parameters are setup-derived and cannot be updated. "
                "Create a new Model/System/Calculator with changed parameters."
            )
        super().update(**kwargs)

    def reset(self) -> None:
        """Reject in-place reset for exact setup-migrated halogen terms."""
        if type(self) is Halogen:
            raise RuntimeError(
                "Halogen parameters are setup-derived and cannot be reset. "
                "Create a new Model/System/Calculator with changed parameters."
            )
        super().reset()


def _slice_setup(setup: HalogenSetup, index: int) -> HalogenSetup:
    """Select one legacy packed structure for the single-System kernel."""
    return HalogenSetup(
        numbers=setup.numbers[index],
        xbond=setup.xbond[index],
        atomic_radii=setup.atomic_radii[index],
        damp=setup.damp,
        cutoff=setup.cutoff,
        halogen_mask=setup.halogen_mask[index],
        base_mask=setup.base_mask[index],
        valid_atom_mask=setup.valid_atom_mask[index],
        pair_type_mask=setup.pair_type_mask[index],
        neighbor_type_mask=setup.neighbor_type_mask[index],
    )
