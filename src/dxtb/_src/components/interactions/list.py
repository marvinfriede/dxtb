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
Container for interactions.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.typing import (
    Any,
    Literal,
    Slicers,
    Tensor,
    TensorOrTensors,
    overload,
    override,
)

from ..list import ComponentList, ComponentListCache
from .base import Interaction
from .container import Charges, Potential
from .coulomb.multipole import AES2, LABEL_AES2
from .coulomb.secondorder import ES2, LABEL_ES2
from .coulomb.thirdorder import ES3, LABEL_ES3
from .dispersion.d4sc import LABEL_DISPERSIOND4SC, DispersionD4SC
from .field.efield import LABEL_EFIELD, ElectricField
from .field.efieldgrad import LABEL_EFIELD_GRAD, ElectricFieldGrad
from .solvation.alpb import LABEL_GENERALIZED_BORN, GeneralizedBorn

__all__ = ["InteractionList", "InteractionListCache"]


class InteractionListCache(ComponentListCache):
    """
    Restart data for individual interactions, extended by subclasses as
    needed.
    """

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        """
        Cull all interaction caches.

        Parameters
        ----------
        conv : Tensor
            Mask of converged systems.
        """
        for cache in self.values():
            cache.cull(conv, slicers)

    def restore(self) -> None:
        """
        Restore all interaction caches.
        """
        for cache in self.values():
            cache.restore()


class InteractionList(ComponentList[Interaction]):
    """
    List of interactions.
    """

    def __init__(
        self,
        *components: Interaction | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create a list and reject duplicate migrated interaction labels."""
        for label in (
            LABEL_ES2,
            LABEL_ES3,
            LABEL_AES2,
            LABEL_DISPERSIOND4SC,
            LABEL_GENERALIZED_BORN,
            LABEL_EFIELD,
            LABEL_EFIELD_GRAD,
        ):
            matching = [
                component
                for component in components
                if component is not None and component.label == label
            ]
            if len({id(component) for component in matching}) > 1:
                raise ValueError(
                    f"Multiple interactions use the label '{label}'. "
                    "Duplicate migrated interaction labels are ambiguous."
                )
        super().__init__(*components, device=device, dtype=dtype)

    def update(self, name: str, **kwargs: Any) -> Interaction:
        """Update an interaction, preserving legacy subclass behavior."""
        return super().update(name, **kwargs)

    def reset(self, name: str) -> Interaction:
        """Reset an interaction, preserving legacy subclass behavior."""
        return super().reset(name)

    def reset_all(self) -> None:
        """Reset unmigrated interactions, leaving setup terms untouched."""
        for component in self.components:
            if type(component) not in (
                ES2,
                ES3,
                AES2,
                DispersionD4SC,
                GeneralizedBorn,
                ElectricField,
                ElectricFieldGrad,
            ):
                component.reset()

    @override
    def get_energy(
        self,
        charges: Charges | Tensor,
        cache: InteractionListCache,
        ihelp: IndexHelper,
    ) -> Tensor:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Charges | Tensor
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : InteractionListCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-resolved energy vector for orbital partial charges.
        """
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges)

        if len(self.components) <= 0:
            return ihelp.reduce_orbital_to_atom(torch.zeros_like(charges.mono))

        return torch.stack(
            [
                interaction.get_energy(cache[interaction.label], charges, ihelp)
                for interaction in self.components
            ]
        ).sum(dim=0)

    def get_energy_as_dict(
        self, charges: Charges, cache: InteractionListCache, ihelp: IndexHelper
    ) -> dict[str, Tensor]:
        """
        Compute the energy for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : InteractionListCache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Energy vector for each orbital partial charge.
        """
        if len(self.components) <= 0:
            return {"none": torch.zeros_like(charges.mono)}

        return {
            interaction.label: interaction.get_energy(
                cache[interaction.label], charges, ihelp
            )
            for interaction in self.components
        }

    @override
    def get_gradient(
        self,
        charges: Charges,
        positions: Tensor,
        cache: InteractionListCache,
        ihelp: IndexHelper,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Calculate gradient for a list of interactions.

        Parameters
        ----------
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : InteractionListCache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Nuclear gradient of all interactions.
        """
        if len(self.components) <= 0:
            return torch.zeros_like(positions)

        return torch.stack(
            [
                interaction.get_gradient(
                    charges,
                    positions,
                    cache[interaction.label],
                    ihelp,
                    grad_outputs=grad_outputs,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                )
                for interaction in self.components
            ]
        ).sum(dim=0)

    @override
    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> InteractionListCache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        InteractionListCache
            Restart data for the interactions.
        """
        cache = InteractionListCache()
        cache.update(
            **{
                interaction.label: interaction.get_cache(
                    numbers=numbers, positions=positions, ihelp=ihelp
                )
                for interaction in self.components
            }
        )
        return cache

    def get_potential(
        self, cache: InteractionListCache, charges: Charges, ihelp: IndexHelper
    ) -> Potential:
        """
        Compute the potential for a list of interactions.

        Parameters
        ----------
        cache : InteractionListCache
            Restart data for the interactions.
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        # create empty potential
        pot = Potential(
            torch.zeros_like(charges.mono),
            dipole=None,
            quad=None,
            batch_mode=ihelp.batch_mode,
        )

        # exit with empty potential if no interactions present
        if len(self.components) <= 0:
            return pot

        # add up potentials from all interactions
        for interaction in self.components:
            p = interaction.get_potential(
                cache[interaction.label], charges, ihelp
            )
            pot += p

        return pot

    ###########################################################################

    @overload
    def get_interaction(
        self, name: Literal["DispersionD4SC"]
    ) -> DispersionD4SC: ...

    @overload
    def get_interaction(
        self, name: Literal["ElectricField"]
    ) -> ElectricField: ...

    @overload
    def get_interaction(
        self, name: Literal["ElectricFieldGrad"]
    ) -> ElectricFieldGrad: ...

    @overload
    def get_interaction(self, name: Literal["ES2"]) -> ES2: ...

    @overload
    def get_interaction(self, name: Literal["ES3"]) -> ES3: ...

    @override  # generic implementation for typing
    def get_interaction(self, name: str) -> Interaction:
        return super().get_interaction(name)
