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
Coulomb: Isotropic second-order electrostatic energy (ES2)
==========================================================

This module implements the second-order electrostatic energy for GFN1-xTB.

Example
-------
.. code-block:: python

    import torch
    import dxtb.coulomb.secondorder as es2
    from dxtb.coulomb.average import harmonic_average as average
    from dxtb import GFN1_XTB, get_element_param

    # Define atomic numbers, positions, and charges
    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [+0.00000000000000, -0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ])

    q = torch.tensor([
        -8.41282505804719e-2,
        2.10320626451180e-2,
        2.10320626451178e-2,
        2.10320626451179e-2,
        2.10320626451179e-2,
    ])

    # Initialize the ES2 energy calculator with parameters
    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
    hubbard = get_element_param(GFN1_XTB.element, "gam")
    es = es2.ES2(hubbard=hubbard, average=average, gexp=gexp)

    # Calculate energy using the provided atomic charges and positions
    cache = es.get_cache(numbers=numbers, positions=positions)
    e = es.get_energy(q, cache)

    torch.set_printoptions(precision=7)
    print(torch.sum(e, dim=-1))  # Output: tensor(0.0005078)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.exceptions import DeviceError
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.constants import xtb
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import (
    DD,
    Any,
    Slicers,
    Tensor,
    TensorLike,
    TensorOrTensors,
    get_default_dtype,
    override,
)
from dxtb._src.utils.scattergather import wrap_gather
from dxtb._src.utils.tensors import grad_key, normalize_device

from ..base import Interaction, InteractionCache
from .average import AveragingFunction, averaging_function, harmonic_average

__all__ = [
    "ES2",
    "ES2Setup",
    "LABEL_ES2",
    "build_es2_coulomb",
    "new_es2",
    "setup_es2",
]


LABEL_ES2 = "ES2"
"""Label for the 'ES2' interaction, coinciding with the class name."""


@dataclass(frozen=True, eq=False)
class ES2Setup:
    """Numbers-only structural and parameter data for one ES2 evaluation."""

    atom_pair_mask: Tensor
    atom_mask: Tensor
    atom_hubbard: Tensor | None
    shell_pair_mask: Tensor | None
    shell_mask: Tensor | None
    shell_hubbard: Tensor | None
    shells_to_atom: Tensor | None
    gexp: Tensor
    average: AveragingFunction
    shell_resolved: bool


class ES2Cache(InteractionCache, TensorLike):
    """
    Cache for Coulomb matrix in ES2.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    mat: Tensor
    """Coulomb matrix."""

    shell_resolved: bool
    """Electrostatics is shell-resolved (default: ``True``)."""

    __slots__ = ["__store", "mat", "shell_resolved"]

    def __init__(
        self,
        mat: Tensor,
        shell_resolved: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else mat.device,
            dtype=dtype if dtype is None else mat.dtype,
        )
        self.mat = mat
        self.shell_resolved = shell_resolved
        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        mat: Tensor
        """Coulomb matrix"""

        def __init__(self, mat: Tensor) -> None:
            self.mat = mat

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.mat)

        _slicer = slicers["shell"] if self.shell_resolved else slicers["atom"]
        slicer = tuple([~conv, *_slicer, *_slicer])
        self.mat = self.mat[slicer]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.mat = self.__store.mat


class ES2(Interaction):
    """
    Isotropic second-order electrostatic energy (ES2).
    """

    hubbard: Tensor
    """Hubbard parameters of all elements."""

    lhubbard: Tensor | None
    """
    Shell-resolved scaling factors for Hubbard parameters.

    :default: ``None`` (i.e., no shell resolution).
    """

    average: AveragingFunction
    """
    Function to use for averaging the Hubbard parameters.

    :default: :func:`dxtb._src.components.interactions.average.harmonic_average`
    """

    gexp: Tensor
    """
    Exponent of the second-order Coulomb interaction.

    :default: 2.0
    """

    shell_resolved: bool
    """
    Whether electrostatics is shell-resolved.

    :default: ``True``
    """

    __slots__ = [
        "hubbard",
        "lhubbard",
        "average",
        "gexp",
        "shell_resolved",
    ]

    def __init__(
        self,
        hubbard: Tensor,
        lhubbard: Tensor | None = None,
        average: AveragingFunction = harmonic_average,
        gexp: Tensor = torch.tensor(xtb.DEFAULT_ES2_GEXP),
        shell_resolved: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.hubbard = hubbard.to(**self.dd)
        self.lhubbard = lhubbard if lhubbard is None else lhubbard.to(**self.dd)
        self.gexp = gexp.to(**self.dd)
        self.average = average

        self.shell_resolved = shell_resolved and lhubbard is not None

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> ES2Cache:
        """
        Obtain the cache object containing the Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        ES2Cache
            Cache object for second order electrostatics.

        Note
        ----
        The cache of an interaction requires ``positions`` as they do not change
        during the self-consistent charge iterations.
        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for ES2 cache.")
        if positions is None:
            raise ValueError("Positions are required for ES2 cache.")
        if ihelp is None:
            raise ValueError("IndexHelper is required for ES2 cache creation.")

        cachvars = (numbers.detach().clone(), positions.detach().clone())

        if self.cache_is_latest(cachvars, grad=(positions,)) is True:
            if not isinstance(self.cache, ES2Cache):
                raise TypeError(
                    f"Cache in {self.label} is not of type '{self.label}."
                    "Cache'. This can only happen if you manually manipulate "
                    "the cache."
                )
            return self.cache

        # if the cache is built, store the cachvar for validation
        self._cachevars = cachvars
        self._cachegrad = grad_key(positions)

        setup = setup_es2(
            numbers,
            self.hubbard,
            ihelp,
            lhubbard=self.lhubbard,
            gexp=self.gexp,
            average=self.average,
            shell_resolved=self.shell_resolved,
        )
        self.cache = ES2Cache(
            build_es2_coulomb(setup, positions),
            shell_resolved=setup.shell_resolved,
        )

        return self.cache

    def get_atom_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> Tensor:
        """
        Calculate the atom-resolved Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Coulomb matrix.
        """
        setup = setup_es2(
            numbers,
            self.hubbard,
            ihelp,
            gexp=self.gexp,
            average=self.average,
            shell_resolved=False,
        )
        return build_es2_coulomb(setup, positions)

    def get_shell_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> Tensor:
        """
        Calculate the Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Coulomb matrix.
        """
        setup = setup_es2(
            numbers,
            self.hubbard,
            ihelp,
            lhubbard=self.lhubbard,
            gexp=self.gexp,
            average=self.average,
            shell_resolved=True,
        )
        return build_es2_coulomb(setup, positions)

    @override
    def get_monopole_atom_energy(
        self, cache: ES2Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        return (
            0.5 * qat * self.get_monopole_atom_potential(cache, qat)
            if not self.shell_resolved
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_shell_energy(
        self, cache: ES2Cache, qat: Tensor, **_: Any
    ) -> Tensor:
        return (
            0.5 * qat * self.get_monopole_shell_potential(cache, qat)
            if self.shell_resolved
            else torch.zeros_like(qat)
        )

    @override
    def get_monopole_atom_potential(
        self,
        cache: ES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved potential. Zero if this interaction is
        shell-resolved.

        Parameters
        ----------
        cache : ES2Cache
            Cache object for second order electrostatics.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).

        Returns
        -------
        Tensor
            Atom-resolved potential.
        """
        return (
            torch.zeros_like(qat)
            if self.shell_resolved
            else einsum("...ik,...k->...i", cache.mat, qat)
        )

    @override
    def get_monopole_shell_potential(
        self,
        cache: ES2Cache,
        qsh: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate shell-resolved potential. Zero if this interaction is only
        atom-resolved.

        Parameters
        ----------
        cache : ES2Cache
            Cache object for second order electrostatics.
        qsh : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Shell-resolved potential.
        """
        return (
            einsum("...ik,...k->...i", cache.mat, qsh)
            if self.shell_resolved
            else torch.zeros_like(qsh)
        )

    @override
    def get_atom_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: ES2Cache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Calculates nuclear gradient of an second order electrostatic energy
        contribution via PyTorch's autograd engine.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        positions : Tensor
            Nuclear positions. Needs ``requires_grad=True``.
        cache : ES2Cache
            Cache object for second order electrostatics.
        grad_outputs : Tensor | None
            Gradient of previous computation, i.e., "vector" in VJP of this
            gradient computation. Defaults to ``None``.

        Returns
        -------
        Tensor
            Nuclear gradient of energy.

        Raises
        ------
        RuntimeError
            ``positions`` tensor does not have ``requires_grad=True``.
        """
        if self.shell_resolved:
            return torch.zeros_like(positions)

        energy = self.get_monopole_atom_energy(cache, charges)
        return self._gradient(
            energy,
            positions,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )

    @override
    def get_shell_gradient(
        self,
        charges: Tensor,
        positions: Tensor,
        cache: ES2Cache,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Calculates nuclear gradient of an second order electrostatic energy
        contribution via PyTorch's autograd engine.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.
        positions : Tensor
            Nuclear positions. Needs ``requires_grad=True``.
        cache : ES2Cache
            Cache object for second order electrostatics.
        grad_out : Tensor | None
            Gradient of previous computation, i.e., "vector" in VJP of this
            gradient computation.

        Returns
        -------
        Tensor
            Nuclear gradient of energy.

        Raises
        ------
        RuntimeError
            ``positions`` tensor does not have ``requires_grad=True``.
        """
        if not self.shell_resolved:
            return torch.zeros_like(positions)

        energy = self.get_monopole_shell_energy(cache, charges)
        return self._gradient(
            energy,
            positions,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )

    def _gradient(
        self,
        energy: Tensor,
        positions: Tensor,
        grad_outputs: TensorOrTensors | None = None,
        retain_graph: bool | None = True,
        create_graph: bool | None = None,
    ) -> Tensor:
        """
        Calculates nuclear gradient of an second order electrostatic energy
        contribution via PyTorch's autograd engine.

        Parameters
        ----------
        energy : Tensor
            Shell-resolved energy.
        positions : Tensor
            Nuclear positions. Needs ``requires_grad=True``.
        grad_out : Tensor | None
            Gradient of previous computation, i.e., "vector" in VJP of this
            gradient computation.

        Returns
        -------
        Tensor
            Nuclear gradient of energy.

        Raises
        ------
        RuntimeError
            ``positions`` tensor does not have ``requires_grad=True``.
        """
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs ``requires_grad=True``.")

        # avoid autograd call if energy is zero (autograd fails anyway)
        if torch.equal(energy, torch.zeros_like(energy)):
            return torch.zeros_like(positions)

        if create_graph is None:
            create_graph = torch.is_grad_enabled()

        if grad_outputs is None:
            grad_outputs = torch.ones_like(energy)

        (gradient,) = torch.autograd.grad(
            energy,
            positions,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )
        return gradient


def setup_es2(
    numbers: Tensor,
    hubbard: Tensor,
    ihelp: IndexHelper,
    *,
    gexp: Tensor,
    average: AveragingFunction,
    lhubbard: Tensor | None = None,
    shell_resolved: bool = True,
) -> ES2Setup:
    """Gather number- and parameter-dependent ES2 data for evaluation."""
    resolve_shells = shell_resolved and lhubbard is not None
    if shell_resolved and lhubbard is None:
        raise ValueError("No 'lhubbard' parameters set.")

    atom_pair_mask = real_pairs(numbers, mask_diagonal=True)
    atom_mask = atom_pair_mask | torch.diag_embed(
        torch.ones_like(numbers, dtype=torch.bool)
    )

    if resolve_shells:
        assert lhubbard is not None
        shells_to_atom = ihelp.shells_to_atom
        shell_pair_mask = wrap_gather(atom_pair_mask, (-2, -1), shells_to_atom)
        shell_mask = wrap_gather(atom_mask, (-2, -1), shells_to_atom)
        shell_hubbard = ihelp.spread_ushell_to_shell(
            lhubbard
        ) * ihelp.spread_uspecies_to_shell(hubbard)
        atom_hubbard = None
    else:
        shells_to_atom = None
        shell_pair_mask = None
        shell_mask = None
        shell_hubbard = None
        atom_hubbard = ihelp.spread_uspecies_to_atom(hubbard)

    # The setup deliberately retains only gathered tensors and structural maps,
    # not the ES2 interaction or its cross-call geometry cache.
    return ES2Setup(
        atom_pair_mask=atom_pair_mask,
        atom_mask=atom_mask,
        atom_hubbard=atom_hubbard,
        shell_pair_mask=shell_pair_mask,
        shell_mask=shell_mask,
        shell_hubbard=shell_hubbard,
        shells_to_atom=shells_to_atom,
        gexp=gexp,
        average=average,
        shell_resolved=resolve_shells,
    )


def build_es2_coulomb(setup: ES2Setup, positions: Tensor) -> Tensor:
    """Build the atom- or shell-resolved ES2 matrix for current positions."""
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    distances = storch.cdist(positions, positions, p=2)

    if setup.shell_resolved:
        if (
            setup.shell_pair_mask is None
            or setup.shell_mask is None
            or setup.shell_hubbard is None
            or setup.shells_to_atom is None
        ):
            raise RuntimeError("Shell-resolved ES2 setup is incomplete.")
        shell_distances = wrap_gather(distances, (-2, -1), setup.shells_to_atom)
        dist_gexp = torch.where(
            setup.shell_pair_mask,
            torch.pow(shell_distances + eps, setup.gexp),
            eps,
        )
        avg = torch.where(
            setup.shell_mask,
            setup.average(setup.shell_hubbard + eps),
            eps,
        )
        tmp = dist_gexp + torch.where(
            setup.shell_mask, torch.pow(avg, -setup.gexp), eps
        )
        return torch.where(
            setup.shell_mask, 1.0 / torch.pow(tmp, 1.0 / setup.gexp), zero
        )

    if setup.atom_pair_mask is None or setup.atom_mask is None:
        raise RuntimeError("Atom-resolved ES2 setup is incomplete.")
    if setup.atom_hubbard is None:
        raise RuntimeError("Atom-resolved ES2 setup has no Hubbard values.")

    dist_gexp = torch.where(
        setup.atom_pair_mask,
        torch.pow(distances + eps, setup.gexp),
        eps,
    )
    avg = torch.where(
        setup.atom_mask,
        setup.average(setup.atom_hubbard + eps),
        eps,
    )
    tmp = dist_gexp + torch.where(
        setup.atom_mask, torch.pow(avg, -setup.gexp), eps
    )
    return torch.where(
        setup.atom_mask, 1.0 / torch.pow(tmp, 1.0 / setup.gexp), zero
    )


def new_es2(
    unique: Tensor,
    par: Param | ParamModule,
    shell_resolved: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ES2 | None:
    """
    Create new instance of :class:`.ES2`.

    Parameters
    ----------
    unique : Tensor
        Unique elements in the system (shape: ``(nunique,)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.
    shell_resolved: bool
        Electrostatics is shell-resolved.

    Returns
    -------
    ES2 | None
        Instance of the ES2 class or ``None`` if no ES2 is used.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    # compatibility with previous version based on `Param`
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "charge" not in par or par.is_none("charge.effective"):
        return None

    if device is not None:
        if normalize_device(device) != unique.device:
            raise DeviceError(
                f"Passed device ({device}) and device of `unique` tensor "
                f"({unique.device}) do not match."
            )

    hubbard = par.get_elem_param(unique, "gam")
    lhubbard = (
        par.get_elem_param(unique, "lgam") if shell_resolved is True else None
    )

    return ES2(
        hubbard,
        lhubbard,
        average=averaging_function[par.get("charge.effective.average")],
        gexp=par.get("charge.effective.gexp"),
        **dd,
    )
