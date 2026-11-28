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

from functools import partial

import torch
from tad_mctc import Structure, storch
from tad_mctc.batch import real_pairs
from tad_mctc.data import en as element_en
from tad_mctc.ncoord import CNModel
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.components.base import numbers_to_tensors
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


def _resolve_cn_cutoff(cutoff: Tensor | float | int) -> float:
    """Resolve an SRB coordination-number cutoff to a static Python float."""
    if isinstance(cutoff, Tensor):
        if cutoff.ndim != 0 and cutoff.numel() != 1:
            raise ValueError(
                "SRB coordination-number cutoff must be a scalar value."
            )
        return float(cutoff.detach())
    if isinstance(cutoff, (int, float)) and not isinstance(cutoff, bool):
        return float(cutoff)
    raise TypeError(
        "SRB coordination-number cutoff must be a scalar float or Tensor, "
        f"got {type(cutoff).__name__}."
    )


class ShortRangeBondSetup(Node):
    """Numbers-only SRB values used by one or more geometry evaluations."""

    numbers: Tensor = child(keep_dtype=True)
    r0: Tensor = child()
    cnfak: Tensor = child()
    en: Tensor = child()
    pauling: Tensor = child()
    rcov: Tensor = child()
    counting_function: CountingFunction = context()
    cn_cutoff: float = context()
    shift: Tensor = child()
    prefactor: Tensor = child()
    steepness: Tensor = child()
    enscale: Tensor = child()
    enpoly: Tensor = child()
    pair_cutoff2: Tensor = child()
    cn_max: Tensor = child()
    cn_kcn: Tensor = child()


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
        rcov=clone(component.rcov),
        counting_function=component.counting_function,
        shift=clone(component.shift),
        prefactor=clone(component.prefactor),
        steepness=clone(component.steepness),
        enscale=clone(component.enscale),
        enpoly=clone(component.enpoly),
        pair_cutoff2=clone(component.pair_cutoff2),
        cn_cutoff=_resolve_cn_cutoff(component.cn_cutoff),
        cn_max=clone(component.cn_max),
        cn_kcn=clone(component.cn_kcn),
    )


def short_range_bond_energy(
    setup: ShortRangeBondSetup, positions: Tensor
) -> Tensor:
    """Evaluate the atom-partitioned SRB energy from setup and positions."""
    cn_model = CNModel(
        count=partial(setup.counting_function, kcn=setup.cn_kcn),
        cutoff=setup.cn_cutoff,
        cn_max=setup.cn_max,
        rcov=setup.rcov,
    )
    cn = cn_model(
        Structure(
            numbers=setup.numbers.expand(positions.shape[:-1]),
            positions=positions,
        )
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

    r0: Tensor = child()
    """Reference bond lengths of the unique elements."""

    cnfak: Tensor = child()
    """Coordination-number scaling of the reference bond lengths."""

    en: Tensor = child()
    """Electronegativities of the unique elements."""

    rcov: Tensor = child()
    """Covalent radii of all elements."""

    counting_function: CountingFunction = context()
    """Counting function of the coordination number."""

    shift: Tensor = child()
    """Shift of the bond-length correction."""

    prefactor: Tensor = child()
    """Prefactor of the correction energy."""

    steepness: Tensor = child()
    """Steepness of the correction potential."""

    enscale: Tensor = child()
    """Electronegativity scaling."""

    enpoly: Tensor = child()
    """Polynomial coefficients of the electronegativity scaling."""

    pair_cutoff2: Tensor = child()
    """Squared cutoff of the atom pairs."""

    cn_cutoff: Tensor = child()
    """Cutoff of the coordination number."""

    cn_max: Tensor = child()
    """Maximum coordination number."""

    cn_kcn: Tensor = child()
    """Steepness of the coordination-number counting function."""

    def _normalize(self) -> dict[str, Any]:
        return {
            **super()._normalize(),
            **numbers_to_tensors(
                self,
                (
                    "shift",
                    "prefactor",
                    "steepness",
                    "enscale",
                    "pair_cutoff2",
                    "cn_cutoff",
                    "cn_max",
                    "cn_kcn",
                ),
                self.r0,
            ),
        }

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
