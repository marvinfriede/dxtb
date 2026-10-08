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
from dxtb._src.basis.bas import Basis
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
    "setup_libcint",
]


@dataclass(frozen=True, eq=False)
class LibcintIntegralSetup:
    """Composition-dependent data for libcint integral construction."""

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
        self._integral_setup = LibcintIntegralSetup(
            numbers=self.numbers,
            par=self.par,
            ihelp=self.ihelp,
            force_cpu=force_cpu,
        )

    @property
    def integral_setup(self) -> LibcintIntegralSetup:
        """Composition-dependent libcint setup data."""
        return self._integral_setup

    def setup(self, positions: Tensor, **kwargs: Any) -> LibcintCallData:
        """Create call-local libcint wrappers for ``positions``."""
        return setup_libcint(
            self.integral_setup, positions, mask=kwargs.get("mask")
        )


class IntDriverLibcint(BaseIntDriverLibcint):
    """
    Implementation of ``libcint``-based integral driver.
    """
