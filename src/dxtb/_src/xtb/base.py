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
xTB Hamiltonians: Base
======================

Base class for xTB Hamiltonians.
"""

from __future__ import annotations

import torch
from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.typing import PathLike, Tensor, TensorLike

from dxtb import IndexHelper
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import CNFunc
from dxtb._src.xtb.h0 import H0Setup, build_hcore, setup_h0

from .abc import HamiltonianABC

__all__ = ["BaseHamiltonian"]

PAD = -1


class BaseHamiltonian(HamiltonianABC, TensorLike):
    """
    Base class for GFN Hamiltonians.

    For the Hamiltonians, no integral driver is needed. Therefore, the
    signatures are different from the integrals over atomic orbitals. The most
    important difference is the `build` method, which does not require the
    driver anymore and only takes the positions (and the overlap integral).
    """

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""
    unique: Tensor
    """Unique species of the system."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    hscale: Tensor
    """Off-site scaling factor for the Hamiltonian."""
    kcn: Tensor
    """Coordination number dependent shift of the self energy."""
    kpair: Tensor
    """Element-pair-specific parameters for scaling the Hamiltonian."""
    refocc: Tensor
    """Reference occupation numbers."""
    selfenergy: Tensor
    """Self-energy of each species."""
    shpoly: Tensor
    """Polynomial parameters for the distant dependent scaling."""
    valence: Tensor
    """
    Whether the shell belongs to the valence shell.
    Only requried for GFN1-xTB (second s-function for H).
    """

    en: Tensor
    """Pauling electronegativity of each species."""
    enscale: Tensor
    """Electronegativity scaling factor."""
    rad: Tensor
    """Van-der-Waals radius of each species."""

    cn: CNFunc | None
    """Coordination number function."""

    __slots__ = [
        "numbers",
        "unique",
        "ihelp",
        "hscale",
        "kcn",
        "kpair",
        "refocc",
        "selfenergy",
        "shpoly",
        "valence",
        "en",
        "enscale",
        "rad",
        "setup",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        setup: H0Setup | None = None,
        **_,
    ) -> None:
        super().__init__(device, dtype)

        # check device of input tensors
        if any(tensor.device != self.device for tensor in (numbers, ihelp)):
            raise ValueError("All input tensors must be on the same device")

        if not isinstance(par, ParamModule):
            par = ParamModule(par, **self.dd)

        if par.is_none("hamiltonian"):
            raise RuntimeError("Parametrization does not specify Hamiltonian.")

        setup = (
            setup_h0(numbers, par, ihelp, device=device, dtype=dtype)
            if setup is None
            else setup
        )
        self.setup = setup
        self.numbers = setup.numbers
        self.unique = setup.unique
        self.ihelp = setup.ihelp

        self.label = self.__class__.__name__
        self._matrix = None

        # Initialize Hamiltonian parameters

        self.rad = setup.rad
        self.en = setup.en
        self.enscale = setup.enscale
        self.kcn = setup.kcn
        self.selfenergy = setup.selfenergy
        self.shpoly = setup.shpoly
        self.refocc = setup.refocc
        self.valence = setup.valence
        self.hscale = setup.hscale
        self.kpair = setup.kpair
        self.cn = setup.cn

        tensors = [
            ("hscale", self.hscale),
            ("kcn", self.kcn),
            ("kpair", self.kpair),
            ("refocc", self.refocc),
            ("selfenergy", self.selfenergy),
            ("shpoly", self.shpoly),
            ("en", self.en),
            ("rad", self.rad),
        ]

        for name, tensor in tensors:
            if tensor.dtype != self.dtype:
                raise DtypeError(
                    f"Tensor '{name}' has dtype '{tensor.dtype}'; "
                    f"expected '{self.dtype}'."
                )

        # For device checking, include an extra tensor 'valence'
        tensors_device = tensors + [("valence", self.valence)]
        for name, tensor in tensors_device:
            if tensor.device != self.device:
                raise DeviceError(
                    f"Tensor '{name}' is on device '{tensor.device}'; "
                    f"expected '{self.device}'."
                )

    @property
    def matrix(self) -> Tensor | None:
        """Hamiltonian matrix."""
        return self._matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        self._matrix = mat

    def _clone_tensorlike(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BaseHamiltonian:
        """Clone setup tensors with structural indices kept integral."""
        new = super()._clone_tensorlike(device=device, dtype=dtype)
        new.setup = self.setup.to(device=device, dtype=dtype)
        return new

    def clear(self) -> None:
        """Clear the integral matrix."""
        self._matrix = None

    @property
    def requires_grad(self) -> bool:
        """Whether the Hamiltonian matrix will be differentiated."""
        if self._matrix is None:
            return False

        return self._matrix.requires_grad

    def _get_elem_valence(self, par: ParamModule) -> Tensor:
        """
        Obtain a mask for valence and non-valence shells. This is only required
        for GFN1-xTB's second hydrogen s-function. For GFN2-xTB, this is a
        dummy method, i.e., the mask is always ``True``.

        Returns
        -------
        Tensor
            Mask indicating valence shells for each unique species.
        """
        return torch.ones(
            len(self.ihelp.unique_angular), device=self.device, dtype=torch.bool
        )

    def get_occupation(self) -> Tensor:
        """
        Obtain the reference occupation numbers for each orbital.
        """

        refocc = self.ihelp.spread_ushell_to_orbital(self.refocc)
        orb_per_shell = self.ihelp.spread_shell_to_orbital(
            self.ihelp.orbitals_per_shell
        )

        return torch.where(
            orb_per_shell != 0,
            refocc / orb_per_shell,
            torch.tensor(0, **self.dd),
        )

    def to_pt(self, path: PathLike | None = None) -> None:
        """
        Save the integral matrix to a file.

        Parameters
        ----------
        path : PathLike | None
            Path to the file where the integral matrix should be saved. If
            ``None``, the matrix is saved to the default location.
        """
        if path is None:
            path = f"{self.label.casefold()}.pt"

        torch.save(self.matrix, path)

    def build(
        self,
        positions: Tensor,
        overlap: Tensor | None = None,
        charge: Tensor | float | int | None = None,
    ) -> Tensor:
        """
        Build the xTB Hamiltonian.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        overlap : Tensor | None, optional
            Overlap matrix. If ``None``, the true xTB Hamiltonian is *not*
            built. Defaults to ``None``.
        charge : Tensor | float | int | None, optional
            Total molecular charge. Consumed by GFN0 and ignored by the
            existing GFN1/GFN2 Hamiltonians.

        Returns
        -------
        Tensor
            Hamiltonian (always symmetric).
        """
        h0, _ = build_hcore(self.setup, positions, overlap, charge)
        self.matrix = h0
        return h0
