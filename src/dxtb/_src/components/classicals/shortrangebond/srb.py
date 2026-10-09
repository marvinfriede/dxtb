# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Short-Range Bond Correction
===========================

This module provides explicit numbers-only setup and plain-PyTorch energy
evaluation for the short-range bond correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import any_to_tensor
from tad_mctc.data import en as element_en

from dxtb import IndexHelper
from dxtb._src.ncoord import coordination_number
from dxtb._src.typing import Any, CountingFunction, Tensor, override

from ..base import Classical, ComponentCache

__all__ = [
    "LABEL_SRB",
    "ShortRangeBond",
    "ShortRangeBondSetup",
    "setup_srb",
    "short_range_bond_energy",
]


LABEL_SRB = "ShortRangeBond"
"""Label for the :class:`.ShortRangeBond` component."""


@dataclass(frozen=True, eq=False)
class ShortRangeBondSetup:
    """Numbers-only SRB values used by one or more geometry evaluations."""

    numbers: Tensor
    r0: Tensor
    cnfak: Tensor
    en: Tensor
    pauling: Tensor
    rcov: Tensor
    counting_function: CountingFunction
    shift: Tensor
    prefactor: Tensor
    steepness: Tensor
    enscale: Tensor
    enpoly: Tensor
    pair_cutoff2: Tensor
    cn_cutoff: Tensor
    cn_max: Tensor
    cn_kcn: Tensor


def setup_srb(
    component: ShortRangeBond,
    numbers: Tensor,
    ihelp: IndexHelper,
) -> ShortRangeBondSetup:
    """Gather all geometry-independent SRB data into a frozen setup."""

    def clone(value: Tensor) -> Tensor:
        return value.clone()

    return ShortRangeBondSetup(
        numbers=numbers,
        r0=clone(ihelp.spread_uspecies_to_atom(component.r0)),
        cnfak=clone(ihelp.spread_uspecies_to_atom(component.cnfak)),
        en=clone(ihelp.spread_uspecies_to_atom(component.en)),
        pauling=clone(element_en.PAULING(**component.dd)[numbers]),
        rcov=clone(component.rcov[numbers]),
        counting_function=component.counting_function,
        shift=clone(component.shift),
        prefactor=clone(component.prefactor),
        steepness=clone(component.steepness),
        enscale=clone(component.enscale),
        enpoly=clone(component.enpoly),
        pair_cutoff2=clone(component.pair_cutoff2),
        cn_cutoff=clone(component.cn_cutoff),
        cn_max=clone(component.cn_max),
        cn_kcn=clone(component.cn_kcn),
    )


def short_range_bond_energy(
    setup: ShortRangeBondSetup, positions: Tensor
) -> Tensor:
    """Evaluate the atom-partitioned SRB energy from setup and positions."""
    cn = coordination_number(
        setup.numbers,
        positions,
        counting_function=setup.counting_function,
        rcov=setup.rcov,
        cutoff=setup.cn_cutoff,
        cn_max=setup.cn_max,
        kcn=setup.cn_kcn,
    )

    eligible = (setup.numbers >= 5) & (setup.numbers <= 9)
    pair_mask = (
        real_pairs(setup.numbers, mask_diagonal=True)
        & eligible.unsqueeze(-1)
        & eligible.unsqueeze(-2)
        & (setup.numbers.unsqueeze(-1) != setup.numbers.unsqueeze(-2))
    )

    eps = positions.new_tensor(torch.finfo(positions.dtype).eps)
    distances = torch.where(
        pair_mask,
        storch.cdist(positions, positions, p=2),
        eps,
    )
    pair_mask = pair_mask & (distances**2 < setup.pair_cutoff2)

    radius = setup.r0 + setup.cnfak * cn + setup.shift
    fitted_difference = torch.abs(
        setup.en.unsqueeze(-1) - setup.en.unsqueeze(-2)
    )
    orders = torch.arange(
        1,
        setup.enpoly.shape[0] + 1,
        device=fitted_difference.device,
        dtype=fitted_difference.dtype,
    ).reshape((-1,) + (1,) * fitted_difference.ndim)
    powers = fitted_difference.unsqueeze(0) ** orders
    coefficients = setup.enpoly.reshape((-1,) + (1,) * fitted_difference.ndim)
    factor = 1.0 - (coefficients * powers).sum(0)
    reference_distance = (radius.unsqueeze(-1) + radius.unsqueeze(-2)) * factor

    pauling_difference = setup.pauling.unsqueeze(-1) - setup.pauling.unsqueeze(
        -2
    )
    width = setup.steepness * (1.0 + setup.enscale * pauling_difference**2)
    pair_energy = setup.prefactor * torch.exp(
        -width * (distances - reference_distance) ** 2
    )
    pair_energy = torch.where(
        pair_mask, pair_energy, torch.zeros_like(pair_energy)
    )
    return 0.5 * pair_energy.sum(-1)


class ShortRangeBond(Classical):
    """Short-range correction for heteronuclear B--F pairs."""

    __slots__ = [
        "r0",
        "cnfak",
        "en",
        "rcov",
        "counting_function",
        "shift",
        "prefactor",
        "steepness",
        "enscale",
        "enpoly",
        "pair_cutoff2",
        "cn_cutoff",
        "cn_max",
        "cn_kcn",
    ]

    def __init__(
        self,
        r0: Tensor,
        cnfak: Tensor,
        en: Tensor,
        rcov: Tensor,
        counting_function: CountingFunction,
        shift: Tensor | float | int,
        prefactor: Tensor | float | int,
        steepness: Tensor | float | int,
        enscale: Tensor | float | int,
        enpoly: Tensor,
        pair_cutoff2: Tensor | float | int,
        cn_cutoff: Tensor | float | int,
        cn_max: Tensor | float | int,
        cn_kcn: Tensor | float | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.r0 = r0.to(**self.dd)
        self.cnfak = cnfak.to(**self.dd)
        self.en = en.to(**self.dd)
        self.rcov = rcov.to(**self.dd)
        self.counting_function = counting_function
        self.shift = any_to_tensor(shift, **self.dd)
        self.prefactor = any_to_tensor(prefactor, **self.dd)
        self.steepness = any_to_tensor(steepness, **self.dd)
        self.enscale = any_to_tensor(enscale, **self.dd)
        self.enpoly = enpoly.to(**self.dd)
        self.pair_cutoff2 = any_to_tensor(pair_cutoff2, **self.dd)
        self.cn_cutoff = any_to_tensor(cn_cutoff, **self.dd)
        self.cn_max = any_to_tensor(cn_max, **self.dd)
        self.cn_kcn = any_to_tensor(cn_kcn, **self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> ShortRangeBondSetup:
        """Build fresh numbers-only SRB setup data."""
        if ihelp is None:
            raise ValueError(
                "IndexHelper is required for short-range bond correction."
            )
        return setup_srb(self, numbers, ihelp)

    @override
    def get_energy(
        self, positions: Tensor, cache: ComponentCache, **_: Any
    ) -> Tensor:
        """Return the symmetrically half-partitioned atomwise SRB energy."""
        if not isinstance(cache, ShortRangeBondSetup):
            raise TypeError(
                f"Data in {self.label} is not of type 'ShortRangeBondSetup'."
            )
        return short_range_bond_energy(cache, positions)

    def update(self, **kwargs: Any) -> None:
        """Reject in-place changes for the exact setup-migrated SRB term."""
        if type(self) is ShortRangeBond:
            raise RuntimeError(
                "ShortRangeBond parameters are setup-derived and cannot be "
                "updated. Create a new Model/System/Calculator with changed "
                "parameters."
            )
        super().update(**kwargs)

    def reset(self) -> None:
        """Reject in-place reset for the exact setup-migrated SRB term."""
        if type(self) is ShortRangeBond:
            raise RuntimeError(
                "ShortRangeBond parameters are setup-derived and cannot be "
                "reset. Create a new Model/System/Calculator with changed "
                "parameters."
            )
        super().reset()
