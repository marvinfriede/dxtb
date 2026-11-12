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

Setup Node and call-local inputs for `libcint`-based integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.basis.bas import (
    Basis,
    create_libcint_cgtos_from_data,
    create_libcint_from_data,
)
from dxtb._src.typing import Any, Literal, Tensor

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = [
    "LibcintCallData",
    "LibcintIntegralSetup",
    "prepare_libcint",
    "setup_libcint",
]


class LibcintIntegralSetup(Node):
    """Transform-visible setup for single-System libcint evaluation."""

    numbers: Tensor = child(keep_dtype=True)
    unique: Tensor = child(keep_dtype=True)
    ngauss: Tensor = child(keep_dtype=True)
    pqn: Tensor = child(keep_dtype=True)
    slater: Tensor = child()
    valence: Tensor = child(keep_dtype=True)
    index: IndexHelper = child()
    sto: Literal["gfn1", "gfn2"] = context(default="gfn2")
    force_cpu: bool = context(default=True)

    def _validate(self) -> None:
        if self.index.batch_mode != 0:
            raise ValueError("Libcint setup requires one single System.")
        if self.force_cpu and self.numbers.device.type != "cpu":
            raise ValueError(
                "force_cpu=True requires CPU-resident libcint setup tensors."
            )

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

    def create_libcint_cgtos(
        self,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Construct differentiable CGTO values from captured Slater data."""
        return create_libcint_cgtos_from_data(
            self.unique,
            self.index.ushells_per_unique,
            self.ngauss,
            self.pqn,
            self.index.unique_angular,
            self.slater,
            self.valence,
            self.sto,
        )

    def create_libcint(
        self, positions: Tensor
    ) -> tuple[libcint.Basis, libcint.Structure]:
        """Create a call-local basis and structure from captured data."""
        alphas, coeffs = self.create_libcint_cgtos()
        return create_libcint_from_data(
            self.numbers,
            self.unique,
            self.index.ushells_per_unique,
            self.index.unique_angular,
            alphas,
            coeffs,
            positions,
        )


def setup_libcint(
    basis: Basis,
    *,
    force_cpu: bool = True,
) -> LibcintIntegralSetup:
    """Build the authoritative libcint setup Node from a single-System Basis."""
    ihelp = basis.ihelp
    if ihelp.batch_mode != 0:
        raise ValueError("Libcint setup requires one single System.")

    source_tensors = (
        basis.numbers,
        basis.unique,
        basis.ngauss,
        basis.pqn,
        basis.slater,
        basis.valence,
        ihelp.shells_per_atom,
        ihelp.shells_to_ushell,
        ihelp.angular,
        ihelp.ushells_per_unique,
        ihelp.unique_angular,
        ihelp.shells_to_atom,
        ihelp.orbitals_to_shell,
    )
    if force_cpu and any(
        tensor.device.type != "cpu" for tensor in source_tensors
    ):
        raise ValueError(
            "force_cpu=True requires CPU-resident libcint setup tensors."
        )

    return LibcintIntegralSetup(
        numbers=basis.numbers.clone(),
        unique=basis.unique.clone(),
        ngauss=basis.ngauss.clone(),
        pqn=basis.pqn.clone(),
        slater=basis.slater.clone(),
        valence=basis.valence.clone(),
        index=ihelp,
        sto=basis.sto,
        force_cpu=force_cpu,
    )


@dataclass(frozen=True, eq=False)
class LibcintCallData:
    """Call-local libcint basis, structure, and index helper for one system."""

    basis: Any
    structure: Any
    index: IndexHelper


def prepare_libcint(
    setup: LibcintIntegralSetup,
    positions: Tensor,
) -> LibcintCallData:
    """Build call-local libcint basis and structure from setup and positions."""
    if setup.force_cpu and positions.device.type != "cpu":
        positions = positions.to(device=torch.device("cpu"))

    libcint_basis, structure = setup.create_libcint(positions)
    return LibcintCallData(
        basis=libcint_basis,
        structure=structure,
        index=setup.index,
    )
