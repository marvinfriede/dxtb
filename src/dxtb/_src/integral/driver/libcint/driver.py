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
from dxtb._src.basis.bas import Basis, _to_int_list
from dxtb._src.typing import Tensor

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = [
    "LibcintIntegralSetup",
    "ShellLayout",
    "setup_libcint",
]

ShellLayout = tuple[tuple[int, tuple[tuple[int, int], ...]], ...]


def _build_shell_layout(
    unique: Tensor,
    ushells_per_unique: Tensor,
    unique_angular: Tensor,
) -> ShellLayout:
    """Extract static per-element ``(angmom, unique_shell_idx)`` metadata."""
    unique_numbers = _to_int_list(unique)
    shell_counts = _to_int_list(ushells_per_unique)
    angular = _to_int_list(unique_angular)
    n_slots = len(angular)

    layout: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    shell = 0
    for number, count in zip(unique_numbers, shell_counts):
        if count == 0:
            if shell < n_slots and angular[shell] < 0:
                shell += 1
            continue

        element_shells: list[tuple[int, int]] = []
        for _ in range(count):
            angmom = angular[shell]
            if angmom >= 0:
                element_shells.append((angmom, shell))
            shell += 1

        if number > 0:
            if element_shells:
                layout.append((number, tuple(element_shells)))
        elif element_shells:
            raise ValueError("Padding atomic number zero cannot own shells.")

    return tuple(layout)


class LibcintIntegralSetup(Node):
    """
    Immutable, transform-visible setup for single-System libcint evaluation.

    Stores setup-time CGTO ``alphas`` and ``coeffs`` expanded from the active
    parametrization. Reusing one instance across geometries supports repeated
    independent backward passes with respect to ``positions`` (``retain_graph=False``).
    Independent repeated backward passes with respect to basis parameters
    (such as ``element.*.slater``) require rebuilding a fresh setup per
    evaluation so setup-time ``slater_to_gauss``/``orthogonalize`` saved
    tensors are not consumed across calls.
    """

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
        if self.numbers.shape != self.index.shells_per_atom.shape:
            raise ValueError(
                "Libcint setup numbers must match IndexHelper atom shape."
            )
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
        self,
        positions: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[libcint.Basis, libcint.Structure]:
        """Construct call-local libcint basis and structure for ``positions``."""
        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import libcint

        if positions.shape != (*self.numbers.shape, 3):
            raise ValueError(
                f"Expected positions of shape {(*self.numbers.shape, 3)}, "
                f"got {tuple(positions.shape)}."
            )
        if self.force_cpu and positions.device.type != "cpu":
            positions = positions.to(device=torch.device("cpu"))

        shells_by_element: dict[int, list[libcint.Shell]] = {}
        for number, element_shells in self.shell_layout:
            # ``slater_to_gauss`` normalizes stretched 3D Cartesian Gaussians
            # ``x^l exp(-alpha r^2)``, whereas ``tad-libcint`` combines the
            # radial contraction ``sum_k c_k r^l exp(-alpha_k r^2)`` with real
            # spherical harmonics of unit norm on the sphere, leaving an
            # unnormalized shell self-overlap of ``(2l + 1) / (4 * pi)``.
            # ``Shell.normalized()`` rescales the 1D shell coefficients by
            # their exact analytical radial self-overlap ``overlap.rsqrt()``
            # (differentiably w.r.t. ``alphas`` and ``coeffs``) rather than
            # multiplying a second normalization factor on top, so do not
            # remove ``.normalized()``.
            shells_by_element[number] = [
                libcint.Shell(
                    angmom=angmom,
                    alphas=self.alphas[shell_idx],
                    coeffs=self.coeffs[shell_idx],
                ).normalized()
                for angmom, shell_idx in element_shells
            ]

        numbers = self.numbers
        if mask is not None:
            if mask.shape != numbers.shape:
                raise ValueError(
                    f"Expected mask of shape {tuple(numbers.shape)}, "
                    f"got {tuple(mask.shape)}."
                )
            if self.force_cpu and mask.device.type != "cpu":
                mask = mask.to(device=torch.device("cpu"))
            if self.active_atoms is not None:
                mask = mask & (numbers > 0)
            numbers = torch.masked_select(numbers, mask)
            positions = torch.masked_select(
                positions, mask.unsqueeze(-1)
            ).reshape((-1, 3))
        elif self.active_atoms is not None:
            numbers = numbers.index_select(0, self.active_atoms)
            positions = positions.index_select(-2, self.active_atoms)

        return (
            libcint.Basis(shells=shells_by_element),
            libcint.Structure(numbers=numbers, positions=positions),
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
    if (
        basis.numbers.ndim != 1
        or basis.numbers.shape != ihelp.shells_per_atom.shape
    ):
        raise ValueError(
            "Libcint setup requires 1D atomic numbers matching IndexHelper."
        )

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
    shell_layout = _build_shell_layout(
        basis.unique, ihelp.ushells_per_unique, ihelp.unique_angular
    )

    numbers = basis.numbers.clone()
    active_atoms: Tensor | None = None
    if numbers.device.type != "meta":
        num_list = _to_int_list(basis.numbers)
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
