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

from functools import partial

from tad_mctc import Structure
from tad_mctc.convert import any_to_tensor
from tad_mctc.ncoord import CNModel, erf_count
from tad_mctc.tree import Node, child, context
from tad_multicharge.model.base import SolveMode
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper
from dxtb._src.components.base import numbers_to_tensors
from dxtb._src.typing import Any, Tensor, override
from dxtb._src.utils.tensors import structure_charge

from ..base import Classical

__all__ = [
    "IES",
    "IESSetup",
    "LABEL_IES",
    "ies_energy",
    "setup_ies",
]


LABEL_IES = "IES"
"""Label for the :class:`.IES` component, coinciding with the class name."""


def _resolve_cutoff(cutoff: Tensor | float | int) -> float:
    """Resolve an IES coordination-number cutoff to a static Python float."""
    if isinstance(cutoff, Tensor):
        if cutoff.ndim != 0 and cutoff.numel() != 1:
            raise ValueError(
                "IES coordination-number cutoff must be a scalar value."
            )
        return float(cutoff.detach())
    if isinstance(cutoff, (int, float)) and not isinstance(cutoff, bool):
        return float(cutoff)
    raise TypeError(
        "IES coordination-number cutoff must be a scalar float or Tensor, "
        f"got {type(cutoff).__name__}."
    )


class IESSetup(Node):
    """Numbers- and parameter-derived data for one IES term.

    ``EEQModel`` is an immutable tad-multicharge Node. Its ``solve`` method
    constructs all geometry-dependent values locally and does not update the
    model.
    """

    numbers: Tensor = child(keep_dtype=True)
    eeq: EEQModel = child()
    rcov: Tensor = child()
    cutoff: float = context()
    cn_max: Tensor = child()
    cn_kcn: Tensor = child()


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
        rcov=ies.rcov.clone(),
        cutoff=_resolve_cutoff(ies.cutoff),
        cn_max=ies.cn_max.clone(),
        cn_kcn=ies.cn_kcn.clone(),
    )


def ies_energy(
    setup: IESSetup,
    positions: Tensor,
    charge: Tensor | float | int,
    *,
    solve_mode: SolveMode = "schur",
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
    numbers = setup.numbers.expand(positions.shape[:-1])
    structure = Structure(
        numbers=numbers,
        positions=positions,
        charge=structure_charge(total_charge, numbers),
    )
    cn_model = CNModel(
        count=partial(erf_count, kcn=setup.cn_kcn),
        cutoff=setup.cutoff,
        cn_max=setup.cn_max,
        rcov=setup.rcov,
    )
    cn = cn_model(structure)
    _charges, energy = setup.eeq.solve(
        structure, cn, return_energy=True, solve_mode=solve_mode
    )
    return energy


class IES(Classical):
    """
    Isotropic electrostatics (IES) Classical of a tight-binding Method.
    """

    chi: Tensor = child()
    """Element-resolved electronegativity lookup table."""

    eeq_kcn: Tensor = child()
    """Element-resolved coordination-number dependence."""

    eta: Tensor = child()
    """Element-resolved chemical hardness."""

    rad: Tensor = child()
    """Element-resolved Gaussian charge width."""

    rcov: Tensor = child()
    """D3 covalent-radius lookup table from tad-mctc."""

    cutoff: Tensor = child()
    """Coordination-number real-space cutoff."""

    cn_max: Tensor = child()
    """Smooth coordination-number cap."""

    cn_kcn: Tensor = child()
    """Steepness of the erf counting function."""

    def _normalize(self) -> dict[str, Any]:
        return {
            **super()._normalize(),
            **numbers_to_tensors(
                self, ("cutoff", "cn_max", "cn_kcn"), self.chi
            ),
        }

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **_: Any
    ) -> IESSetup:
        """
        Build fresh coordinate-independent data for the EEQ solve.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of the Structure (shape: ``(natom,)``).
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
        *,
        solve_mode: SolveMode = "schur",
        **_: Any,
    ) -> Tensor:
        """
        Calculate the isotropic electrostatics energy using the EEQ model.

        Parameters
        ----------
        positions : Tensor
            Atomic positions of the Structure (shape: ``(natom, 3)``).
        cache : IESSetup
            Frozen coordinate-independent data for the EEQ solve.
        charge : Tensor | float | int | None
            Total molecular charge. If None, the energy is not computed.
        solve_mode : SolveMode, optional
            Linear solver backend for the EEQ solve. Defaults to ``"schur"``.

        Returns
        -------
        Tensor
            Isotropic electrostatics energy of the Structure (shape: ``()``).
        """
        if not isinstance(cache, IESSetup):
            raise TypeError(f"Setup in {self.label} is not of type 'IESSetup'.")
        if charge is None:
            raise ValueError("Total molecular charge is required for IES.")
        return ies_energy(cache, positions, charge, solve_mode=solve_mode)
