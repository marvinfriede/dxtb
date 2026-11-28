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
Interactions: Self-consistent D4 Dispersion
===========================================

Self-consistent D4 dispersion correction.
"""

from __future__ import annotations

from typing import Any

import torch
from tad_dftd import defaults as d4_defaults
from tad_dftd.model.d4 import (
    d4sc_atom_energy,
    d4sc_atom_potential,
    d4sc_dispersion_matrix,
)
from tad_dftd.ncoord import coordination_number
from tad_mctc.data import PAULING
from tad_mctc.exceptions import DeviceError
from tad_mctc.ncoord import erf_count
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import DD, CountingFunction, Tensor, override

from dxtb import IndexHelper
from dxtb._src.components.classicals.dispersion import _d4_adapter as d4
from dxtb._src.param import Param, ParamModule, as_param_module
from dxtb._src.utils.tensors import normalize_device

from ..base import Interaction, InteractionCache

__all__ = [
    "D4SCSetup",
    "DispersionD4SC",
    "DispersionD4SCCache",
    "LABEL_DISPERSIOND4SC",
    "build_d4sc_cache",
    "new_d4sc",
    "setup_d4sc",
]


LABEL_DISPERSIOND4SC = "DispersionD4SC"
"""Label for the :class:`.DispersionD4SC` interaction, coinciding with the class name."""


class D4SCSetup(Node):
    """Static numbers and parameter data for self-consistent D4."""

    numbers: Tensor = child(keep_dtype=True)
    param: d4.Param = child()
    rcov: Tensor = child()
    r4r2: Tensor = child()
    cn_cutoff: Tensor = child()
    pair_weight: Tensor = child()
    model: d4.model.D4Model = child()
    counting_function: CountingFunction = context()


def setup_d4sc(component: DispersionD4SC, numbers: Tensor) -> D4SCSetup:
    """Gather D4SC parameter and model tables for one System."""
    if type(component.damping_function) is not d4.damping.RationalDamping:
        raise TypeError(
            "DispersionD4SC setup requires RationalDamping; "
            "custom damping strategies are only supported via a "
            "DispersionD4SC subclass overriding get_cache."
        )
    en = PAULING(device=numbers.device, dtype=component.rcov.dtype)[numbers]
    endiff = torch.abs(en.unsqueeze(-2) - en.unsqueeze(-1))
    pair_weight = d4_defaults.D4_K4 * torch.exp(
        -((endiff + d4_defaults.D4_K5) ** 2.0) / d4_defaults.D4_K6
    )

    model = component.model
    return D4SCSetup(
        numbers=numbers.clone(),
        param={name: value.clone() for name, value in component.param.items()},
        rcov=component.rcov.clone(),
        r4r2=component.r4r2.clone(),
        cn_cutoff=component.cutoff.cn.clone(),
        pair_weight=pair_weight,
        model=model.replace(
            numbers=numbers.clone(), wf=model.wf.clone(), rc6=model.rc6.clone()
        ),
        counting_function=component.counting_function,
    )


def build_d4sc_cache(
    setup: D4SCSetup, positions: Tensor
) -> DispersionD4SCCache:
    """Build fresh geometry-dependent D4SC data for one evaluation."""
    numbers = setup.numbers
    cn = coordination_number(
        numbers,
        positions,
        counting_function=setup.counting_function,
        rcov=setup.rcov,
        cutoff=setup.cn_cutoff,
        pair_weight=setup.pair_weight,
    )
    model = setup.model
    dispmat = d4sc_dispersion_matrix(
        numbers,
        positions,
        setup.param,
        setup.r4r2,
        model.rc6,
        cutoff=d4_defaults.D4_DISP2_CUTOFF,
    )
    return DispersionD4SCCache(cn=cn, dispmat=dispmat, model=model)


class DispersionD4SCCache(InteractionCache):
    """
    Restart data for the :class:`.DispersionD4SC` interaction.

    Note
    ----
    The dispersion parameters (a1, a2, ...) are given in the dispersion
    class constructor.
    """

    cn: Tensor = child()
    """Coordination number of every atom."""

    dispmat: Tensor = child()
    """
    Dispersion matrix. This quantity is almost equal to the dispersion energy,
    except for multiplication with C6 and C8.
    """

    model: d4.model.D4Model = child()
    """D4 model of the setup."""


class DispersionD4SC(Interaction):
    """
    Self-consistent D4 dispersion correction (:class:`.DispersionD4SC`).
    """

    param: d4.Param = child()
    """Dispersion parameters."""

    model: d4.model.D4Model = child()
    """Model for the D4 dispersion correction."""

    rcov: Tensor = child()
    """Covalent radii of all atoms."""

    r4r2: Tensor = child()
    """R4/R2 ratio of all atoms."""

    cutoff: d4.cutoff.Cutoff = child()
    """Real-space cutoff for the D4 dispersion correction."""

    counting_function: CountingFunction = context(default=erf_count)
    """
    Counting function for the coordination number.

    :default: :func:`tad_mctc.ncoord.erf_count`
    """

    damping_function: d4.damping.Damping = context(
        default=d4.damping.RationalDamping()
    )
    """
    Damping function for the dispersion correction.

    :default: :func:`d4.damping.RationalDamping`
    """

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
        **_,
    ) -> DispersionD4SCCache:
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
        DispersionD4SCCache
            Restart data for the interaction.

        Note
        ----
        If the :class:`.DispersionD4SC` interaction is evaluated within the
        :class:`dxtb.components.InteractionList`, ``positions`` will be passed
        as an argument, too. Hence, it is necessary to absorb the ``positions``
        in the signature of the function (also see
        :meth:`dxtb.components.Interaction.get_cache`).
        """
        if numbers is None:
            raise ValueError(
                "Atomic numbers are required for DispersionD4SC data."
            )
        if positions is None:
            raise ValueError("Positions are required for D4SC data.")
        return build_d4sc_cache(setup_d4sc(self, numbers), positions)

    @override
    def get_monopole_atom_energy(
        self, cache: InteractionCache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the D4 dispersion correction energy.

        Parameters
        ----------
        cache : DispersionD4SCCache
            Restart data for the interaction.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise D4 dispersion correction energies.
        """
        if not isinstance(cache, DispersionD4SCCache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'DispersionD4SCCache'."
            )

        weights = cache.model.weight_references(cache.cn, qat)

        return d4sc_atom_energy(cache.dispmat, weights)

    @override
    def get_monopole_atom_potential(
        self, cache: InteractionCache, qat: Tensor, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the D4 dispersion correction potential.

        Parameters
        ----------
        cache : DispersionD4SCCache
            Restart data for the interaction.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise dispersion correction potential.
        """
        if not isinstance(cache, DispersionD4SCCache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'DispersionD4SCCache'."
            )

        weights, dgwdq = cache.model.weight_references(
            cache.cn, qat, with_dgwdq=True
        )

        return d4sc_atom_potential(cache.dispmat, weights, dgwdq)


def new_d4sc(
    numbers: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DispersionD4SC | None:
    """
    Create new instance of :class:`.DispersionD4SC`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).
    par : Param | ParamModule
        Parametrization of an extended tight-binding (xTB) Method.

    Returns
    -------
    DispersionD4SC | None
        Instance of the :class:`.DispersionD4SC` class or ``None`` if
        no :class:`.DispersionD4SC` is used.
    """
    target_device = (
        numbers.device if device is None else normalize_device(device)
    )
    par = as_param_module(par, device=target_device, dtype=dtype)
    if (
        "dispersion" not in par
        or par.is_none("dispersion")
        or par.is_none("dispersion.d4")
        or par.is_false("dispersion.d4.sc")
    ):
        return None

    if device is not None and target_device != numbers.device:
        raise DeviceError(
            f"Passed device ({device}) and device of `numbers` tensor "
            f"({numbers.device}) do not match."
        )
    dd: DD = {"device": target_device, "dtype": dtype or par.dtype}

    param = d4.Param(
        **{
            key: (
                val.to(**dd)
                if isinstance(val := par.get(f"dispersion.d4.{key}"), Tensor)
                else val
            )
            for key in ("a1", "a2", "s6", "s8", "s9", "s10")
        }
    )

    rcov = d4.data.COV_D3(**dd)[numbers]
    r4r2 = d4.data.R4R2(**dd)[numbers]
    model = d4.model.D4Model.create(numbers, ref_charges="gfn2", **dd)
    cutoff = d4.cutoff.Cutoff.create(disp2=50.0, disp3=25.0, **dd)

    return DispersionD4SC(
        param=param, model=model, rcov=rcov, r4r2=r4r2, cutoff=cutoff
    )
