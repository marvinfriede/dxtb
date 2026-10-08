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

import torch
from tad_mctc.batch import deflate

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis, BasisSetup
from dxtb._src.param import ParamModule
from dxtb._src.typing import Any, Tensor
from dxtb._src.utils import is_basis_list

from ...base import IntDriver
from .base import LibcintImplementation

__all__ = [
    "BaseIntDriverLibcint",
    "IntDriverLibcint",
    "LibcintCallData",
    "LibcintIntegralSetup",
    "LegacyLibcintSetup",
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
    ihelp: IndexHelper

    @property
    def batch_mode(self) -> int:
        """Batch mode of the structural index helper."""
        return self.ihelp.batch_mode


def setup_libcint(
    setup: LibcintIntegralSetup,
    positions: Tensor,
    *,
    mask: Tensor | None = None,
) -> LibcintCallData:
    """Build call-local libcint wrappers from setup data and positions."""
    from dxtb._src.exlibs import libcint

    if setup.force_cpu and positions.device.type != "cpu":
        positions = positions.to(device=torch.device("cpu"))

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
