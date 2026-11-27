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
# pylint: disable=assignment-from-none
"""
Provides base class for interactions in the extended tight-binding Hamiltonian.
The `Interaction` class is not purely abstract as its methods return zero.
"""

# pylint: disable=unused-argument

from __future__ import annotations

import functools
import inspect
from collections.abc import Sequence

import torch

from dxtb import IndexHelper
from dxtb._src.typing import Any, Tensor, TensorOrTensors, final

from ...components.base import Component, ComponentCache
from .container import Charges, Potential

__all__ = [
    "Interaction",
    "InteractionCache",
    "call_monopole_hook",
    "validate_unique_interaction_labels",
]


@functools.lru_cache(maxsize=None)
def _monopole_multipole_flags(func: Any) -> tuple[bool, bool]:
    """Return ``(accepts_qdp, accepts_qqp)`` for an Interaction monopole hook."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return (True, True)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return (True, True)
    accepts_qdp = (
        "qdp" in params
        and params["qdp"].kind != inspect.Parameter.POSITIONAL_ONLY
    )
    accepts_qqp = (
        "qqp" in params
        and params["qqp"].kind != inspect.Parameter.POSITIONAL_ONLY
    )
    return (accepts_qdp, accepts_qqp)


def call_monopole_hook(
    method: Any,
    cache: Any,
    q: Tensor,
    qdp: Tensor | None,
    qqp: Tensor | None,
) -> Tensor:
    """Invoke a monopole potential or energy hook with optional multipoles."""
    accepts_qdp, accepts_qqp = _monopole_multipole_flags(
        getattr(method, "__func__", method)
    )
    if accepts_qdp and accepts_qqp:
        return method(cache, q, qdp=qdp, qqp=qqp)
    if accepts_qdp:
        return method(cache, q, qdp=qdp)
    if accepts_qqp:
        return method(cache, q, qqp=qqp)
    return method(cache, q)


class InteractionCache(ComponentCache):
    """
    Restart data for individual interactions, extended by subclasses as
    needed.
    """

    __slots__: list[str] = []


class Interaction(Component):
    """
    Base class for defining interactions with the charge density.

    Every charge-dependent energy contribution should inherit from this class
    as it conveniently handles the different resolutions (atom, shell, orbital)
    required for all subclasses. Depending on the multipole order and
    resolution, subclasses implement one or more pairs of
    ``get_<monopole/dipole/quadrupole>_<atom/shell>_<potential/energy>`` hooks;
    hooks that are not overridden automatically evaluate to zero.

    .. warning::

        Never overwrite the ``get_potential``, ``get_energy`` and
        ``get_gradient`` methods in the subclass. These methods collect the
        respective variable for their (internally-handled) evaluation.

    Additionally, a ``get_cache`` method is required that precalculates and
    returns per-geometry, charge-independent setup data to avoid repeated
    calculations during the SCF iterations.

    Example
    -------
    The third-order electrostatics :class:`~dxtb.components.coulomb.ES3`
    implements ``get_monopole_atom_energy``, ``get_monopole_shell_energy``,
    ``get_monopole_atom_potential``, and ``get_monopole_shell_potential``
    alongside ``get_cache``. Nuclear and parameter derivatives are obtained
    automatically via PyTorch autograd / ``torch.func``.
    """

    # Already defined in `Component` parent class
    label: str
    """Label for the interaction."""

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the interaction."""
        super().__init__(device, dtype)

    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> InteractionCache:
        """
        Create restart data for individual interactions.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty :class:`.Interaction` by returning an empty
        :class:`.InteractionCache`.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        InteractionCache
            Restart data for the interaction.
        """
        return InteractionCache()

    @final
    def get_potential(
        self,
        cache: InteractionCache,
        charges: Charges,
        ihelp: IndexHelper,
    ) -> Potential:
        """
        Compute the potential from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        charges : Charges
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Potential
            Orbital- and multipole-resolved potentials.
        """

        # monopole potential: shell-resolved
        qsh = ihelp.reduce_orbital_to_shell(charges.mono)
        vsh = call_monopole_hook(
            self.get_monopole_shell_potential,
            cache,
            qsh,
            charges.dipole,
            charges.quad,
        )

        # monopole potential: atom-resolved
        qat = ihelp.reduce_shell_to_atom(qsh)
        vat = call_monopole_hook(
            self.get_monopole_atom_potential,
            cache,
            qat,
            charges.dipole,
            charges.quad,
        )

        # spread to orbital-resolution
        vsh = vsh + ihelp.spread_atom_to_shell(vat)
        vmono = ihelp.spread_shell_to_orbital(vsh)

        # multipole potentials
        vdipole = self.get_dipole_atom_potential(
            cache, qat, charges.dipole, charges.quad
        )
        vquad = self.get_quadrupole_atom_potential(
            cache, qat, charges.dipole, charges.quad
        )

        return Potential(vmono, dipole=vdipole, quad=vquad, label=self.label)

    def get_monopole_shell_potential(
        self,
        cache: InteractionCache,
        qsh: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qsh : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """
        return torch.zeros_like(qsh)

    def get_monopole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved potential vector for each atom partial charge.
        """
        return torch.zeros_like(qat)

    def get_dipole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor | None:
        """
        Compute the dipole potential. All quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning None.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor | None
            Atom-resolved potential vector for each atom or None if not needed.
        """
        return None

    def get_quadrupole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor | None:
        """
        Compute the quadrupole potential. All quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning None.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor | None
            Atom-resolved potential vector for each atom or None if not needed.
        """
        return None

    ##########################################################################

    @final
    def get_energy(
        self, cache: InteractionCache, charges: Charges, ihelp: IndexHelper
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are
        orbital-resolved.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        charges : Charges
            Collection of charges. Monopolar partial charges are
            orbital-resolved.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Atom-resolved energy vector.

        Note
        ----
        The subclasses of :class:`dxtb.components.base.Interaction` should
        implement the `get_<type>_energy` methods. If they are not implemented
        in the subclass, they will evaluate to zero.
        """
        if charges.mono is None:
            raise RuntimeError(
                "Charge collection is empty. At least monopolar partial "
                "charges are required."
            )

        qsh = ihelp.reduce_orbital_to_shell(charges.mono)
        esh = call_monopole_hook(
            self.get_monopole_shell_energy,
            cache,
            qsh,
            charges.dipole,
            charges.quad,
        )

        qat = ihelp.reduce_shell_to_atom(qsh)
        eat = call_monopole_hook(
            self.get_monopole_atom_energy,
            cache,
            qat,
            charges.dipole,
            charges.quad,
        )

        e = eat + ihelp.reduce_shell_to_atom(esh)

        if charges.dipole is not None:
            edp = self.get_dipole_atom_energy(
                cache, qat=qat, qdp=charges.dipole, qqp=charges.quad
            )
            e = e + edp

        if charges.quad is not None:
            eqp = self.get_quadrupole_atom_energy(
                cache, qat=qat, qdp=charges.dipole, qqp=charges.quad
            )
            e = e + eqp

        return e

    def get_monopole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor | None, optional
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor | None, optional
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Energy vector for each atom partial charge.
        """
        return torch.zeros_like(qat)

    def get_monopole_shell_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        qat : Tensor
            Shell-resolved partial charges.
        qdp : Tensor | None, optional
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor | None, optional
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Energy vector for each shell partial charge.
        """
        return torch.zeros_like(qat)

    def get_dipole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the energy from the atomic dipole moments, all quantities are
        atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Energy vector for each atomic dipole moment.
        """
        return torch.zeros_like(qat)

    def get_quadrupole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the energy from the atomic quadrupole moments, all quantities
        are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        cache : InteractionCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Energy vector for each atomic quadrupole moment.
        """
        return torch.zeros_like(qat)

    ##########################################################################

    @final
    def get_gradient(
        self,
        charges: Charges,
        positions: Tensor,
        cache: InteractionCache,
        ihelp: IndexHelper,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Compute the nuclear gradient using orbital-resolved charges.

        Note
        ----
        This method calls both :meth:`.get_atom_gradient` and
        :meth:`.get_shell_gradient` and adds up both gradients. Hence, one of
        the contributions must be zero.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : InteractionCache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """
        qao = charges.mono.detach()

        qsh = ihelp.reduce_orbital_to_shell(qao)
        gsh = self.get_shell_gradient(
            qsh, positions, cache, grad_outputs, retain_graph, create_graph
        )

        qat = ihelp.reduce_shell_to_atom(qsh)
        gat = self.get_atom_gradient(
            qat, positions, cache, grad_outputs, retain_graph, create_graph
        )

        return gsh + gat

    def get_atom_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: InteractionCache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Return zero gradient.

        This method should be implemented by the subclass.
        However, returning zeros here serves three purposes:

        - the interaction can (theoretically) be empty
        - the gradient of the interaction is indeed zero and thus requires no
          gradient implementation (one can, however, implement a method that
          returns zeros to make this more obvious)
        - the interaction always uses shell-resolved charges and atom-resolved
          charges are never required

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates (for shape of gradient).

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """
        return torch.zeros_like(positions)

    def get_shell_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: InteractionCache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Return zero gradient.

        This method should be implemented by the subclass.
        However, returning zeros here serves three purposes:

        - the interaction can (theoretically) be empty
        - the gradient of the interaction is indeed zero and thus requires no
          gradient implementation (one can, however, implement a method that
          returns zeros to make this more obvious)
        - the interaction always uses atom-resolved charges and shell-resolved
          charges are never required

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates (for shape of gradient).

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """
        return torch.zeros_like(positions)


def validate_unique_interaction_labels(
    interactions: Sequence[Interaction],
) -> None:
    """Reject distinct ``Interaction`` instances that share the same label."""
    seen_by_label: dict[str, Interaction] = {}
    for interaction in interactions:
        previous = seen_by_label.get(interaction.label)
        if previous is not None and previous is not interaction:
            raise ValueError(
                "Distinct interactions in a single System share the "
                f"label '{interaction.label}'."
            )
        seen_by_label[interaction.label] = interaction
