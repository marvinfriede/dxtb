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

Setup data and call-local inputs for `libcint`-based integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.basis.bas import (
    Basis,
    BasisSetup,
    create_libcint_cgtos_from_data,
    create_libcint_from_data,
)
from dxtb._src.typing import Any, Literal, Tensor

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = [
    "LibcintCallData",
    "LibcintBasisNode",
    "LibcintIntegralSetupNode",
    "LibcintIntegralSetup",
    "core_libcint_setup",
    "setup_libcint",
    "setup_libcint_core",
]


@dataclass(frozen=True, eq=False)
class LibcintIntegralSetup:
    """Narrow, composition-dependent setup for pure libcint construction."""

    numbers: Tensor
    ihelp: IndexHelper
    basis_setups: tuple[BasisSetup, ...]
    force_cpu: bool

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LibcintIntegralSetup:
        """Return setup tensors converted to the requested dtype/device."""
        target_device = torch.device("cpu") if self.force_cpu else device
        return LibcintIntegralSetup(
            numbers=self.numbers.to(device=target_device),
            ihelp=self.ihelp.to(device=target_device),
            basis_setups=tuple(
                bs.to(device=target_device, dtype=dtype)
                for bs in self.basis_setups
            ),
            force_cpu=self.force_cpu,
        )


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
    sto: Literal["gfn1", "gfn2"] = context(default="gfn2")

    def create_libcint_cgtos(
        self,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Construct differentiable CGTO values from captured Slater data."""
        return create_libcint_cgtos_from_data(
            self.unique,
            self.ushells_per_unique,
            self.ngauss,
            self.pqn,
            self.unique_angular,
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
            self.ushells_per_unique,
            self.unique_angular,
            alphas,
            coeffs,
            positions,
        )


class LibcintIntegralSetupNode(Node):
    """Transform-visible setup for an exact single-System libcint path."""

    numbers: Tensor = child(keep_dtype=True)
    basis_setups: tuple[LibcintBasisNode, ...] = child()
    index: IndexHelper = child()
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


def _build_libcint_setup_node(
    numbers: Tensor,
    basis: Basis | BasisSetup,
    ihelp: IndexHelper,
    *,
    force_cpu: bool,
    copy_index: bool,
) -> LibcintIntegralSetupNode:
    """Build a LibcintIntegralSetupNode from gathered basis and index data."""
    source_tensors = (
        numbers,
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

    def snapshot(value: Tensor) -> Tensor:
        return value.clone()

    index = (
        IndexHelper(
            unique_angular=snapshot(ihelp.unique_angular),
            angular=snapshot(ihelp.angular),
            atom_to_unique=snapshot(ihelp.atom_to_unique),
            ushells_to_unique=snapshot(ihelp.ushells_to_unique),
            ushells_per_unique=snapshot(ihelp.ushells_per_unique),
            shells_to_ushell=snapshot(ihelp.shells_to_ushell),
            shells_per_atom=snapshot(ihelp.shells_per_atom),
            shell_index=snapshot(ihelp.shell_index),
            shells_to_atom=snapshot(ihelp.shells_to_atom),
            orbitals_per_shell=snapshot(ihelp.orbitals_per_shell),
            orbital_index=snapshot(ihelp.orbital_index),
            orbitals_to_shell=snapshot(ihelp.orbitals_to_shell),
            batch_mode=0,
        )
        if copy_index
        else ihelp
    )

    return LibcintIntegralSetupNode(
        numbers=snapshot(numbers),
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
                sto=basis.sto,
            ),
        ),
        index=index,
        force_cpu=force_cpu,
    )


def core_libcint_setup(
    setup: LibcintIntegralSetup,
    *,
    copy_index: bool = True,
) -> LibcintIntegralSetupNode:
    """Snapshot a single-System driver setup as structural Node data."""
    if setup.ihelp.batch_mode != 0 or len(setup.basis_setups) != 1:
        raise ValueError("Core libcint setup requires one single System.")

    basis = setup.basis_setups[0]
    return _build_libcint_setup_node(
        setup.numbers,
        basis,
        basis.ihelp,
        force_cpu=setup.force_cpu,
        copy_index=copy_index,
    )


def setup_libcint_core(
    basis: Basis,
    *,
    force_cpu: bool = True,
    copy_index: bool = False,
) -> LibcintIntegralSetupNode:
    """Build the authoritative Node setup directly from a single-System Basis."""
    if basis.ihelp.batch_mode != 0:
        raise ValueError("Core libcint setup requires one single System.")
    return _build_libcint_setup_node(
        basis.numbers,
        basis,
        basis.ihelp,
        force_cpu=force_cpu,
        copy_index=copy_index,
    )


@dataclass(frozen=True, eq=False)
class LibcintInput:
    """One call-local new-API basis/structure pair."""

    basis: Any
    structure: Any


@dataclass(frozen=True, eq=False)
class LibcintCallData:
    """Call-local libcint inputs for one or more independent structures."""

    inputs: tuple[LibcintInput, ...]
    ihelp: IndexHelper

    @property
    def batch_mode(self) -> int:
        """Batch mode of the structural index helper."""
        return self.ihelp.batch_mode


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
        libcint_basis, structure = basis.create_libcint(positions)
        return LibcintCallData(
            inputs=(LibcintInput(libcint_basis, structure),),
            ihelp=setup.index,
        )

    if setup.ihelp.batch_mode == 0:
        if len(setup.basis_setups) != 1:
            raise RuntimeError("Single-system setup needs one basis value.")
        basis = setup.basis_setups[0]
        libcint_basis, structure = basis.create_libcint(positions, mask=mask)
        return LibcintCallData(
            inputs=(LibcintInput(libcint_basis, structure),),
            ihelp=setup.ihelp,
        )

    if setup.ihelp.batch_mode not in (1, 2):
        raise ValueError(f"Unknown batch mode '{setup.ihelp.batch_mode}'.")
    if len(setup.basis_setups) != positions.shape[0]:
        raise RuntimeError("Batched setup count does not match positions.")

    inputs = []
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
        libcint_basis, structure = basis.create_libcint(position)
        inputs.append(LibcintInput(libcint_basis, structure))

    return LibcintCallData(inputs=tuple(inputs), ihelp=setup.ihelp)
