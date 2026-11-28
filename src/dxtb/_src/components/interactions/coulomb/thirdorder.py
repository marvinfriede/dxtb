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
Coulomb: On-site third-order electrostatic energy (ES3)
=======================================================

This module implements the third-order electrostatic energy for GFN1-xTB.

Example
-------

.. code-block:: python

    import torch
    from dxtb import GFN1_XTB, IndexHelper
    from dxtb.components.coulomb import new_es3

    # Define atomic numbers and atomic charges
    numbers = torch.tensor([14, 1, 1, 1, 1])
    qat = torch.tensor([
        -8.41282505804719e-2,
        2.10320626451180e-2,
        2.10320626451178e-2,
        2.10320626451179e-2,
        2.10320626451179e-2,
    ])

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    es = new_es3(torch.unique(numbers), GFN1_XTB)
    cache = es.get_cache(numbers=numbers, ihelp=ihelp)
    e = es.get_monopole_atom_energy(cache, qat)

    # Print the summed energy
    torch.set_printoptions(precision=7)
    print(torch.sum(e, dim=-1))  # tensor(-2.9771152e-05)
"""

from __future__ import annotations

import torch
from tad_mctc.exceptions import DeviceError
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.param import Param, ParamModule, as_param_module
from dxtb._src.typing import DD, Any, Tensor, override
from dxtb._src.utils.tensors import normalize_device

from ..base import Interaction, InteractionCache

__all__ = [
    "ES3",
    "ES3Setup",
    "LABEL_ES3",
    "build_es3_cache",
    "new_es3",
    "setup_es3",
]


LABEL_ES3 = "ES3"
"""Label for the :class:`.ES3` interaction, coinciding with the class name."""


class ES3Setup(Node):
    """Numbers-only parameter data for one ES3 evaluation."""

    hubbard_derivs: Tensor = child()
    shell_resolved: bool = context()


class ES3Cache(InteractionCache):
    """
    Restart data for the :class:`.ES3` interaction.
    """

    hd: Tensor = child()
    """Spread Hubbard derivatives of all atoms (not only unique)."""

    shell_resolved: bool = context(default=False)
    """Whether the third-order electrostatics are shell-resolved."""


class ES3(Interaction):
    """
    On-site third-order electrostatic energy (:class:`.ES3`).
    """

    hubbard_derivs: Tensor = child()
    """Hubbard derivatives of all atoms."""

    shell_scale: Tensor | None = child(default=None)
    """
    Scaling factors for shell-resolved third-order electrostatics.

    In GFN2-xTB, this is a tensor of shape ``(3,)`` containing the scaling
    factors for the s, p, and d shells.

    :default: ``None``
    """

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> ES3Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        ES3Cache
            Restart data for the interaction.

        Note
        ----
        If the :class:`.ES3` interaction is evaluated within the
        :class:`dxtb.components. InteractionList`, ``positions`` will be
        passed as an argument, too. Hence, it is necessary to absorb
        the ``positions`` in the signature of the function (also see
        :meth:`dxtb.components.Interaction.get_cache`).
        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for ES3 cache.")
        if ihelp is None:
            raise ValueError("IndexHelper is required for ES3 cache.")

        setup = setup_es3(
            self.hubbard_derivs, ihelp, shell_scale=self.shell_scale
        )
        return build_es3_cache(setup)

    @override
    def get_monopole_atom_energy(
        self, cache: ES3Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Implements Eq.30 of the following paper:

        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht,
          J. Seibert, S. Spicher and S. Grimme, *WIREs Computational Molecular
          Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493
          <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        cache : ES3Cache
            Restart data for the interaction.
        charges : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atom-wise third-order Coulomb interaction energies.
        """
        return (
            cache.hd * torch.pow(qat, 3.0) / 3.0
            if not cache.shell_resolved
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_shell_energy(
        self, cache: ES3Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Parameters
        ----------
        cache : ES3Cache
            Restart data for the interaction.
        qat : Tensor
            Shell charges of all atoms.

        Returns
        -------
        Tensor
            Shell-wise third-order Coulomb interaction energy.
        """
        return (
            torch.zeros_like(qat)
            if not cache.shell_resolved
            else cache.hd * torch.pow(qat, 3.0) / 3.0
        )

    @override
    def get_monopole_atom_potential(
        self,
        cache: ES3Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate the third-order electrostatic potential.
        Zero if this interaction is shell-resolved.

        Parameters
        ----------
        qat : ES3Cache
            Restart data for the interaction.
        charges : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atom-wise third-order Coulomb interaction potential.
        """
        return (
            cache.hd * torch.pow(qat, 2.0)
            if not cache.shell_resolved
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_shell_potential(
        self, cache: ES3Cache, qsh: Tensor, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the third-order electrostatic potential.
        Zero if this interaction is atom-resolved.

        Parameters
        ----------
        qsh : Tensor
            Shell charges of all atoms.
        cache : ES3Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Shell-wise third-order Coulomb interaction potential.
        """
        return (
            torch.zeros_like(qsh)
            if not cache.shell_resolved
            else cache.hd * torch.pow(qsh, 2.0)
        )


def setup_es3(
    hubbard_derivs: Tensor,
    ihelp: IndexHelper,
    *,
    shell_scale: Tensor | None = None,
) -> ES3Setup:
    """Resolve ES3 parameter data to atoms or shells for one System."""
    if shell_scale is None:
        resolved = ihelp.spread_uspecies_to_atom(hubbard_derivs)
    else:
        scale = ihelp.spread_ushell_to_shell(shell_scale[ihelp.unique_angular])
        resolved = ihelp.spread_uspecies_to_shell(hubbard_derivs) * scale

    return ES3Setup(
        hubbard_derivs=resolved, shell_resolved=shell_scale is not None
    )


def build_es3_cache(setup: ES3Setup) -> ES3Cache:
    """Build the ES3 cache; it does not depend on the geometry."""
    return ES3Cache(
        hd=setup.hubbard_derivs, shell_resolved=setup.shell_resolved
    )


def new_es3(
    unique: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ES3 | None:
    """
    Create new instance of :class:`.ES3`.

    Parameters
    ----------
    unique : Tensor
        Unique elements in the Structure (shape: ``(nunique,)``).
    par : Param | ParamModule
        Parametrization of an extended tight-binding (xTB) Method.

    Returns
    -------
    ES3 | None
        Instance of the :class:`.ES3` class or ``None`` if no :class:`.ES3` is
        used.
    """
    par = as_param_module(par, device=device, dtype=dtype)
    dd: DD = {"device": device, "dtype": dtype or par.dtype}

    if "thirdorder" not in par or par.is_none("thirdorder"):
        return None

    if device is not None:
        if normalize_device(device) != unique.device:
            raise DeviceError(
                f"Passed device ({device}) and device of `unique` tensor "
                f"({unique.device}) do not match."
            )

    hubbard_derivs = par.get_elem_param(unique, "gam3")

    shell_scale = (
        None
        if par.is_false("thirdorder", "shell")
        else torch.cat(
            [
                torch.atleast_1d(par.get("thirdorder.shell.s")),
                torch.atleast_1d(par.get("thirdorder.shell.p")),
                torch.atleast_1d(par.get("thirdorder.shell.d")),
            ],
            dim=0,
        )
    )

    return ES3(
        hubbard_derivs=hubbard_derivs.to(**dd),
        shell_scale=None if shell_scale is None else shell_scale.to(**dd),
    )
