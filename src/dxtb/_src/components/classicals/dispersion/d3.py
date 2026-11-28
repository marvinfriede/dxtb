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
Dispersion: D3
==============

The DFT-D3 dispersion model.
"""

from __future__ import annotations

import tad_dftd as d3
import torch
from tad_mctc.data import radii
from tad_mctc.ncoord import exp_count
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.typing import Any, CountingFunction, Tensor, override

from .base import Dispersion

__all__ = [
    "LABEL_DISPERSIOND3",
    "DispersionD3",
    "DispersionD3Setup",
    "dispersion_d3_energy",
]

_DXTB_D3_CN_CUTOFF = 25.0
_DXTB_D3_DISPERSION_CUTOFF = 50.0


LABEL_DISPERSIOND3 = "DispersionD3"


class DispersionD3Setup(Node):
    """Numbers-only values required by one D3 evaluation."""

    numbers: Tensor = child(keep_dtype=True)
    """Atomic numbers of the dispersion component."""

    param: dict[str, Tensor] = child()
    """Dispersion parameters."""

    ref: d3.reference.Reference = child()
    """Reference C6 coefficients and coordination numbers."""

    rcov: Tensor = child()
    """Covalent radii of all atoms."""

    rvdw: Tensor = child()
    """Pairwise van der Waals radii."""

    r4r2: Tensor = child()
    """R4/R2 ratio of all atoms."""

    cutoff: Tensor = child()
    """Real-space cutoff of the coordination number."""

    counting_function: CountingFunction = context()
    """Counting function of the coordination number."""

    weighting_function: d3.typing.WeightingFunction = context()
    """Weighting function of the reference C6 coefficients."""

    damping_function: d3.typing.DampingFunction = context()
    """Damping function of the dispersion energy."""


def dispersion_d3_energy(
    setup: DispersionD3Setup,
    positions: Tensor,
) -> Tensor:
    """Evaluate D3 using only captured setup values and call-local geometry."""
    # Keep the fixed atomic numbers aligned with geometry batch dimensions.
    # The upstream AtomicC6 vmap rule expects its numbers input to carry the
    # same mapped axis as its reference weights; without this zero-valued
    # structural broadcast, conformer vmap can mistake the atom axis for the
    # conformer axis (or reject the call when the dimensions differ).
    numbers = setup.numbers + torch.zeros_like(
        positions[..., 0], dtype=setup.numbers.dtype
    )
    reference = setup.ref
    cn = d3.ncoord.coordination_number(
        numbers,
        positions,
        counting_function=setup.counting_function,
        rcov=setup.rcov,
        cutoff=setup.cutoff,
    )
    weights = d3.model.weight_references(
        numbers, cn, reference, setup.weighting_function
    )
    c6 = d3.model.atomic_c6(numbers, weights, reference)
    return d3.disp.dispersion(
        numbers,
        positions,
        setup.param,
        c6,
        setup.rvdw,
        setup.r4r2,
        setup.damping_function,
        cutoff=d3.Cutoff(
            disp2=_DXTB_D3_DISPERSION_CUTOFF,
            disp3=_DXTB_D3_DISPERSION_CUTOFF,
        ),
    )


class DispersionD3(Dispersion):
    """
    Representation of the DFT-D3(BJ) dispersion correction
    (:class:`.DispersionD3`).
    """

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **kwargs: Any
    ) -> DispersionD3Setup:
        """
        Obtain cache for storage of settings.

        Settings can be passed as ``kwargs``. The available optional parameters
        are the same as in :func:`tad_dftd3.dftd3`, i.e., "ref", "rcov",
        "rvdw", and "r4r2".

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).

        Returns
        -------
        DispersionD3Setup
            Setup for the D3 dispersion.
        """
        ref = kwargs.pop(
            "ref",
            d3.reference.Reference.load(**self.dd),
        ).to(**self.dd)

        rcov = kwargs.pop(
            "rcov",
            radii.COV_D3(**self.dd)[numbers],
        ).to(**self.dd)

        rvdw = kwargs.pop(
            "rvdw",
            radii.VDW_PAIRWISE(**self.dd)[
                numbers.unsqueeze(-1), numbers.unsqueeze(-2)
            ],
        ).to(**self.dd)

        r4r2 = kwargs.pop(
            "r4r2",
            d3.data.R4R2(**self.dd)[numbers],
        ).to(**self.dd)

        cutoff = kwargs.pop(
            "cutoff",
            torch.tensor(_DXTB_D3_CN_CUTOFF, **self.dd),
        ).to(**self.dd)

        cf = kwargs.pop("counting_function", exp_count)
        wf = kwargs.pop("weighting_function", d3.model.gaussian_weight)
        df = kwargs.pop("damping_function", d3.damping.rational_damping)

        return DispersionD3Setup(
            numbers=self.numbers.clone(),
            param={key: value.clone() for key, value in self.param.items()},
            ref=ref,
            rcov=rcov,
            rvdw=rvdw,
            r4r2=r4r2,
            cutoff=cutoff,
            counting_function=cf,
            weighting_function=wf,
            damping_function=df,
        )

    @override
    def get_energy(
        self, positions: Tensor, cache: DispersionD3Setup, **kwargs: Any
    ) -> Tensor:
        """
        Get D3 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : DispersionD3Setup
            Numbers-only D3 setup.

        Returns
        -------
        Tensor
            Atom-resolved D3 dispersion energy.
        """
        if not isinstance(cache, DispersionD3Setup):
            raise TypeError(
                f"Setup in {self.label} is not of type 'DispersionD3Setup'."
            )
        return dispersion_d3_energy(cache, positions)
