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
Driver: Libcint
===============

Base class for a `libcint`-based integral implementation
Calculation and modification of multipole integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tad_mctc.batch import deflate
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis, BasisSetup
from dxtb._src.basis.ortho import orthogonalize
from dxtb._src.basis.slater import slater_to_gauss
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional.core import CoreIndexData
from dxtb._src.typing import Any, Tensor
from dxtb._src.utils import is_basis_list

from ...base import IntDriver
from .base import LibcintImplementation

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = [
    "BaseIntDriverLibcint",
    "IntDriverLibcint",
    "LibcintCallData",
    "LibcintBasisNode",
    "LibcintIntegralSetupNode",
    "LibcintIntegralSetup",
    "LegacyLibcintSetup",
    "core_libcint_setup",
    "setup_libcint",
    "setup_libcint_legacy",
]


@dataclass(frozen=True, eq=False)
class LibcintIntegralSetup:
    """Narrow, composition-dependent setup for pure libcint construction."""

    numbers: Tensor
    ihelp: IndexHelper
    basis_setups: tuple[BasisSetup, ...]
    force_cpu: bool


class LibcintBasisNode(Node):
    """Captured basis and shell-order data for pure libcint evaluation."""

    numbers: Tensor = child(keep_dtype=True)
    unique: Tensor = child(keep_dtype=True)
    ngauss: Tensor = child(keep_dtype=True)
    pqn: Tensor = child(keep_dtype=True)
    slater: Tensor = child()
    valence: Tensor = child(keep_dtype=True)
    shells_per_atom: Tensor = child(keep_dtype=True)
    shells_to_ushell: Tensor = child(keep_dtype=True)
    angular: Tensor = child(keep_dtype=True)
    ushells_per_unique: Tensor = child(keep_dtype=True)
    unique_angular: Tensor = child(keep_dtype=True)

    def create_libcint_cgtos(
        self,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Construct differentiable CGTO values from captured Slater data."""
        alphas: list[Tensor] = []
        coeffs: list[Tensor] = []
        shell = 0

        for unique_index in range(self.unique.size(0)):
            shell_count = self.ushells_per_unique[unique_index]
            if shell_count == 0:
                zero = torch.tensor(
                    0.0, dtype=self.slater.dtype, device=self.slater.device
                )
                alphas.append(zero)
                coeffs.append(zero)
                shell += 1
                continue

            for _ in range(shell_count):
                alpha, coeff = slater_to_gauss(
                    self.ngauss[shell],
                    self.pqn[shell],
                    self.unique_angular[shell],
                    self.slater[shell],
                )
                if self.valence[shell].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[shell - 1], alpha),
                        (coeffs[shell - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)
                shell += 1
        return tuple(alphas), tuple(coeffs)

    def create_libcint(self, positions: Tensor) -> list[libcint.AtomCGTOBasis]:
        """Create ordered, geometry-local atom basis wrappers."""
        from dxtb._src.exlibs import libcint

        alphas, coeffs = self.create_libcint_cgtos()
        atom_basis: list[libcint.AtomCGTOBasis] = []
        shell = 0
        for atom, number in enumerate(self.numbers):
            bases = []
            for _ in range(self.shells_per_atom[atom]):
                index = self.shells_to_ushell[shell]
                bases.append(
                    libcint.CGTOBasis(
                        angmom=int(self.angular[shell]),
                        alphas=alphas[index],
                        coeffs=coeffs[index],
                        normalized=True,
                    )
                )
                shell += 1
            atom_basis.append(
                libcint.AtomCGTOBasis(
                    atomz=number, bases=bases, pos=positions[atom, :]
                )
            )
        return atom_basis


class LibcintIntegralSetupNode(Node):
    """Transform-visible setup for an exact single-System libcint path."""

    numbers: Tensor = child(keep_dtype=True)
    basis_setups: tuple[LibcintBasisNode, ...] = child()
    index: CoreIndexData = child()
    force_cpu: bool = context()

    def _convert_child(
        self,
        name: str,
        value: object,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> object:
        if self.force_cpu:
            device = torch.device("cpu")
        return super()._convert_child(name, value, device, dtype)


def core_libcint_setup(
    setup: LibcintIntegralSetup,
) -> LibcintIntegralSetupNode:
    """Snapshot a single-system legacy setup as structural Node data."""
    if setup.ihelp.batch_mode != 0 or len(setup.basis_setups) != 1:
        raise ValueError("Core libcint setup requires one single System.")

    basis = setup.basis_setups[0]
    ihelp = basis.ihelp

    def snapshot(value: Tensor) -> Tensor:
        return value.clone()

    return LibcintIntegralSetupNode(
        numbers=snapshot(setup.numbers),
        basis_setups=(
            LibcintBasisNode(
                numbers=snapshot(basis.numbers),
                unique=snapshot(basis.unique),
                ngauss=snapshot(basis.ngauss),
                pqn=snapshot(basis.pqn),
                slater=snapshot(basis.slater),
                valence=snapshot(basis.valence),
                shells_per_atom=snapshot(ihelp.shells_per_atom),
                shells_to_ushell=snapshot(ihelp.shells_to_ushell),
                angular=snapshot(ihelp.angular),
                ushells_per_unique=snapshot(ihelp.ushells_per_unique),
                unique_angular=snapshot(ihelp.unique_angular),
            ),
        ),
        index=CoreIndexData(
            shells_to_atom=snapshot(ihelp.shells_to_atom),
            orbitals_to_shell=snapshot(ihelp.orbitals_to_shell),
        ),
        force_cpu=setup.force_cpu,
    )


@dataclass(frozen=True, eq=False)
class LegacyLibcintSetup:
    """Calculator-local inputs used by the transitional libcint driver."""

    numbers: Tensor
    par: ParamModule
    ihelp: IndexHelper
    force_cpu: bool


@dataclass(frozen=True, eq=False)
class LibcintCallData:
    """Call-local libcint wrappers for one set of positions."""

    drivers: tuple[Any, ...]
    ihelp: IndexHelper | CoreIndexData

    @property
    def batch_mode(self) -> int:
        """Batch mode of the structural index helper."""
        return getattr(self.ihelp, "batch_mode", 0)


def setup_libcint(
    setup: LibcintIntegralSetup | LibcintIntegralSetupNode,
    positions: Tensor,
    *,
    mask: Tensor | None = None,
) -> LibcintCallData:
    """Build call-local libcint wrappers from setup data and positions."""
    from dxtb._src.exlibs import libcint

    if setup.force_cpu and positions.device.type != "cpu":
        positions = positions.to(device=torch.device("cpu"))

    if isinstance(setup, LibcintIntegralSetupNode):
        if len(setup.basis_setups) != 1:
            raise RuntimeError("Single-system setup needs one basis value.")
        basis = setup.basis_setups[0]
        atom_basis = basis.create_libcint(positions)
        return LibcintCallData(
            drivers=(libcint.LibcintWrapper(atom_basis),),
            ihelp=setup.index,
        )

    if setup.ihelp.batch_mode == 0:
        if len(setup.basis_setups) != 1:
            raise RuntimeError("Single-system setup needs one basis value.")
        basis = setup.basis_setups[0]
        atom_basis = basis.create_libcint(positions, mask=mask)
        return LibcintCallData(
            drivers=(libcint.LibcintWrapper(atom_basis, basis.ihelp),),
            ihelp=setup.ihelp,
        )

    if setup.ihelp.batch_mode not in (1, 2):
        raise ValueError(f"Unknown batch mode '{setup.ihelp.batch_mode}'.")
    if len(setup.basis_setups) != positions.shape[0]:
        raise RuntimeError("Batched setup count does not match positions.")

    drivers = []
    for batch_index, basis in enumerate(setup.basis_setups):
        if setup.ihelp.batch_mode == 1:
            if mask is not None:
                position = torch.masked_select(
                    positions[batch_index], mask[batch_index]
                ).reshape((-1, 3))
            else:
                position = positions[batch_index, : basis.numbers.shape[-1]]
        else:
            position = positions[batch_index]
        atom_basis = basis.create_libcint(position)
        drivers.append(libcint.LibcintWrapper(atom_basis, basis.ihelp))

    return LibcintCallData(drivers=tuple(drivers), ihelp=setup.ihelp)


def setup_libcint_legacy(
    setup: LegacyLibcintSetup,
    positions: Tensor,
    *,
    mask: Tensor | None = None,
) -> LibcintCallData:
    """Rebuild Calculator-local libcint wrappers and basis values per call."""
    from dxtb._src.exlibs import libcint

    if setup.force_cpu and positions.device.type != "cpu":
        positions = positions.to(device=torch.device("cpu"))

    if setup.ihelp.batch_mode == 0:
        # Build parameter-dependent CGTO values for this call. Keeping them
        # on the persistent driver would retain an autograd graph and could
        # become stale after parameter updates.
        basis = Basis(
            setup.numbers,
            setup.par,
            setup.ihelp,
            device=positions.device,
            dtype=positions.dtype,
        )
        atombases = basis.create_libcint(positions)
        if not is_basis_list(atombases):
            raise RuntimeError("Single-system libcint basis is not a list.")
        return LibcintCallData(
            drivers=(libcint.LibcintWrapper(atombases, setup.ihelp),),
            ihelp=setup.ihelp,
        )

    if setup.ihelp.batch_mode not in (1, 2):
        raise ValueError(f"Unknown batch mode '{setup.ihelp.batch_mode}'.")

    drivers = []
    for batch_index, numbers in enumerate(setup.numbers):
        if setup.ihelp.batch_mode == 1:
            numbers = deflate(numbers)
            if mask is not None:
                position = torch.masked_select(
                    positions[batch_index], mask[batch_index]
                ).reshape((-1, 3))
            else:
                position = positions[batch_index, : numbers.shape[-1]]
        else:
            position = positions[batch_index]

        ihelp = IndexHelper.from_numbers(numbers, setup.par)
        basis = Basis(
            numbers,
            setup.par,
            ihelp,
            device=positions.device,
            dtype=positions.dtype,
        )
        atombases = basis.create_libcint(position)
        if not is_basis_list(atombases):
            raise RuntimeError("Batched libcint basis entry is not a list.")
        drivers.append(libcint.LibcintWrapper(atombases, ihelp))

    return LibcintCallData(drivers=tuple(drivers), ihelp=setup.ihelp)


class BaseIntDriverLibcint(LibcintImplementation, IntDriver):
    """
    Implementation of `libcint`-based integral driver.
    """

    __slots__ = ["_integral_setup"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        force_cpu = kwargs.pop("force_cpu", False)
        super().__init__(*args, **kwargs)
        self._integral_setup = LegacyLibcintSetup(
            numbers=self.numbers,
            par=self.par,
            ihelp=self.ihelp,
            force_cpu=force_cpu,
        )

    @property
    def integral_setup(self) -> LegacyLibcintSetup:
        """Legacy Calculator-local libcint setup data."""
        return self._integral_setup

    def setup(self, positions: Tensor, **kwargs: Any) -> LibcintCallData:
        """Create call-local libcint wrappers for ``positions``."""
        return setup_libcint_legacy(
            self.integral_setup, positions, mask=kwargs.get("mask")
        )


class IntDriverLibcint(BaseIntDriverLibcint):
    """
    Implementation of ``libcint``-based integral driver.
    """
