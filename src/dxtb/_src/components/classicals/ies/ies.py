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
Isotropic Electrostatics: Class
==============================

This module implements the isotropic electrostatics class. The
:class:`dxtb.components.IES` class is constructed similar to the
:class:`dxtb.components.Halogen` class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tad_mctc import Structure
from tad_mctc.convert import any_to_tensor
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper
from dxtb._src.ncoord import coordination_number, erf_count
from dxtb._src.typing import Any, Tensor, override
from dxtb._src.utils.tensors import structure_charge

from ..base import Classical

__all__ = ["IES", "IESSetup", "LABEL_IES", "ies_energy", "setup_ies"]


LABEL_IES = "IES"
"""Label for the :class:`.IES` component, coinciding with the class name."""


@dataclass(frozen=True, eq=False)
class IESSetup:
    """Frozen numbers/parameter data for one IES term.

    ``EEQModel`` is an immutable tad-multicharge Node. Its ``solve`` method
    constructs all geometry-dependent values locally and does not update the
    model, so it is safe as static System setup.
    """

    numbers: Tensor
    eeq: EEQModel
    rcov: Tensor
    cutoff: Tensor
    cn_max: Tensor
    cn_kcn: Tensor


def setup_ies(ies: IES, numbers: Tensor) -> IESSetup:
    """Gather all numbers-only IES values into immutable System setup."""
    dd = ies.dd
    eeq = EEQModel(
        chi=ies.chi.to(**dd).clone(),
        kcn=ies.eeq_kcn.to(**dd).clone(),
        eta=ies.eta.to(**dd).clone(),
        rad=ies.rad.to(**dd).clone(),
    )
    return IESSetup(
        numbers=numbers,
        eeq=eeq,
        rcov=ies.rcov[numbers].clone(),
        cutoff=ies.cutoff.clone(),
        cn_max=ies.cn_max.clone(),
        cn_kcn=ies.cn_kcn.clone(),
    )


def ies_energy(
    setup: IESSetup,
    positions: Tensor,
    charge: Tensor | float | int,
) -> Tensor:
    """Evaluate IES from frozen setup and explicit geometry/total charge."""
    if positions.device != setup.numbers.device:
        raise RuntimeError(
            "IES setup and positions must be on the same device."
        )
    if positions.dtype != setup.rcov.dtype:
        raise RuntimeError("IES setup and positions must have the same dtype.")

    total_charge = any_to_tensor(
        charge, device=positions.device, dtype=positions.dtype
    )
    cn = coordination_number(
        setup.numbers,
        positions,
        counting_function=erf_count,
        rcov=setup.rcov,
        cutoff=setup.cutoff,
        cn_max=setup.cn_max,
        kcn=setup.cn_kcn,
    )
    structure = Structure(
        numbers=setup.numbers,
        positions=positions,
        charge=structure_charge(total_charge, setup.numbers),
    )
    _charges, energy = setup.eeq.solve(structure, cn, return_energy=True)
    return energy


class IES(Classical):
    """
    Representation of the isotropic electrostatics (IES) component in a tight-binding model.
    """

    chi: Tensor
    """Element-resolved electronegativity lookup table."""

    eeq_kcn: Tensor
    """Element-resolved coordination-number dependence."""

    eta: Tensor
    """Element-resolved chemical hardness."""

    rad: Tensor
    """Element-resolved Gaussian charge width."""

    rcov: Tensor
    """D3 covalent-radius lookup table from tad-mctc."""

    cutoff: Tensor
    """Coordination-number real-space cutoff."""

    cn_max: Tensor
    """Smooth coordination-number cap."""

    cn_kcn: Tensor
    """Steepness of the erf counting function."""

    __slots__ = [
        "chi",
        "eeq_kcn",
        "eta",
        "rad",
        "rcov",
        "cutoff",
        "cn_max",
        "cn_kcn",
    ]

    def __init__(
        self,
        chi: Tensor,
        eeq_kcn: Tensor,
        eta: Tensor,
        rad: Tensor,
        rcov: Tensor,
        cutoff: Tensor | float | int,
        cn_max: Tensor | float | int,
        cn_kcn: Tensor | float | int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.chi = chi.to(**self.dd)
        self.eeq_kcn = eeq_kcn.to(**self.dd)
        self.eta = eta.to(**self.dd)
        self.rad = rad.to(**self.dd)
        self.rcov = rcov.to(**self.dd)
        self.cutoff = any_to_tensor(cutoff, **self.dd)
        self.cn_max = any_to_tensor(cn_max, **self.dd)
        self.cn_kcn = any_to_tensor(cn_kcn, **self.dd)

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> IESSetup:
        """
        Build fresh coordinate-independent data for the EEQ solve.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of the system (shape: ``(natom,)``).
        ihelp : IndexHelper | None
            Helper class for indexing.

        Returns
        -------
        IESSetup
            Frozen setup containing coordinate-independent data.
        """
        return setup_ies(self, numbers)

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: IESSetup,
        charge: Tensor | float | int | None = None,
        **_: Any,
    ) -> Tensor:
        """
        Calculate the isotropic electrostatics energy using the EEQ model.

        Parameters
        ----------
        positions : Tensor
            Atomic positions of the system (shape: ``(natom, 3)``).
        cache : IESSetup
            Frozen coordinate-independent data for the EEQ solve.
        charge : Tensor | float | int | None
            Total molecular charge. If None, the energy is not computed.

        Returns
        -------
        Tensor
            Isotropic electrostatics energy of the system (shape: ``()``).
        """
        if not isinstance(cache, IESSetup):
            raise TypeError(f"Setup in {self.label} is not of type 'IESSetup'.")
        if charge is None:
            raise ValueError("Total molecular charge is required for IES.")
        return ies_energy(cache, positions, charge)

    def update(self, **kwargs: Any) -> None:
        """Reject setup-authority changes for exact migrated IES."""
        if type(self) is IES:
            raise RuntimeError(
                "Exact IES is setup-derived and immutable. Create a new "
                "Model/System after changing its parameters."
            )
        super().update(**kwargs)

    def reset(self) -> None:
        """Reject reset of exact migrated IES parameters."""
        if type(self) is IES:
            raise RuntimeError(
                "Exact IES is setup-derived and immutable. Create a new "
                "Model/System after changing its parameters."
            )
        super().reset()
