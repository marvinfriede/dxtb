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

Setup Node and evaluation-boundary basis construction for `libcint` integrals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.basis.bas import (
    Basis,
    ShellLayout,
    build_shell_layout,
    create_libcint_from_layout,
)
from dxtb._src.typing import Tensor

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = [
    "LibcintIntegralSetup",
    "setup_libcint",
]


class LibcintIntegralSetup(Node):
    """Immutable, transform-visible setup for single-System libcint evaluation."""

    index: IndexHelper = child()
    numbers: Tensor = child(keep_dtype=True)
    alphas: tuple[Tensor, ...] = child()
    coeffs: tuple[Tensor, ...] = child()
    active_atoms: Tensor | None = child(default=None, keep_dtype=True)
    shell_layout: ShellLayout = context()
    force_cpu: bool = context(default=True)

    def _validate(self) -> None:
        if self.index.batch_mode != 0:
            raise ValueError("Libcint setup requires one single System.")
        if self.force_cpu:
            tensors: list[Tensor] = [
                self.numbers,
                *self.alphas,
                *self.coeffs,
                self.index.shells_per_atom,
                self.index.shells_to_ushell,
                self.index.angular,
                self.index.ushells_per_unique,
                self.index.unique_angular,
                self.index.shells_to_atom,
                self.index.orbitals_to_shell,
            ]
            if self.active_atoms is not None:
                tensors.append(self.active_atoms)
            if any(tensor.device.type != "cpu" for tensor in tensors):
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

    def create_libcint(
        self, positions: Tensor
    ) -> tuple[libcint.Basis, libcint.Structure]:
        """Construct call-local libcint basis and structure for ``positions``."""
        if self.force_cpu and positions.device.type != "cpu":
            positions = positions.to(device=torch.device("cpu"))

        numbers = self.numbers
        if self.active_atoms is not None:
            numbers = numbers.index_select(0, self.active_atoms)
            positions = positions.index_select(-2, self.active_atoms)

        return create_libcint_from_layout(
            numbers,
            self.shell_layout,
            self.alphas,
            self.coeffs,
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

    alphas, coeffs = basis.create_cgtos()
    shell_layout = build_shell_layout(
        basis.unique, ihelp.ushells_per_unique, ihelp.unique_angular
    )

    numbers = basis.numbers.clone()
    active_atoms: Tensor | None = None
    if numbers.device.type != "meta":
        num_list = [int(v) for v in numbers.detach().cpu().reshape(-1).tolist()]
        if any(z <= 0 for z in num_list):
            active_atoms = torch.tensor(
                [idx for idx, z in enumerate(num_list) if z > 0],
                dtype=torch.long,
                device=numbers.device,
            )

    return LibcintIntegralSetup(
        index=ihelp,
        numbers=numbers,
        alphas=tuple(alphas),
        coeffs=tuple(coeffs),
        active_atoms=active_atoms,
        shell_layout=shell_layout,
        force_cpu=force_cpu,
    )
