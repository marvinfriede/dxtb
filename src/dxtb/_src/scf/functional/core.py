# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Frozen structural and interaction values for the single-System SCF core."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from tad_mctc.tree import Node, child

from dxtb._src.typing import Tensor
from dxtb._src.utils import wrap_gather, wrap_scatter_reduce


class CoreIndexData(Node):
    """Minimal immutable mappings needed by a single-System SCF map."""

    shells_to_atom: Tensor = child(keep_dtype=True)
    orbitals_to_shell: Tensor = child(keep_dtype=True)

    def reduce_orbital_to_shell(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        return wrap_scatter_reduce(
            values, dim, self.orbitals_to_shell, reduce, extra=extra
        )

    def reduce_shell_to_atom(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        return wrap_scatter_reduce(
            values, dim, self.shells_to_atom, reduce, extra=extra
        )

    def reduce_orbital_to_atom(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        shell_values = self.reduce_orbital_to_shell(
            values, dim=dim, reduce=reduce, extra=extra
        )
        return self.reduce_shell_to_atom(
            shell_values, dim=dim, reduce=reduce, extra=extra
        )

    def spread_atom_to_shell(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        return wrap_gather(values, dim, self.shells_to_atom, extra=extra)

    def spread_shell_to_orbital(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        return wrap_gather(values, dim, self.orbitals_to_shell, extra=extra)

    def spread_atom_to_orbital(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        shell_values = self.spread_atom_to_shell(values, dim=dim, extra=extra)
        return self.spread_shell_to_orbital(shell_values, dim=dim, extra=extra)


@dataclass(frozen=True, eq=False)
class FunctionalInteractionEntry:
    """Immutable exact-built-in interaction kind, tensor data, and setup."""

    kind: str
    data: tuple[tuple[str, Tensor], ...]
    static: tuple[tuple[str, Any], ...] = ()

    def tensor(self, name: str) -> Tensor:
        for key, value in self.data:
            if key == name:
                return value
        raise KeyError(name)

    def option(self, name: str) -> Any:
        for key, value in self.static:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, eq=False)
class FunctionalInteractionSet:
    """Ordered immutable interaction descriptors for the functional core."""

    entries: tuple[FunctionalInteractionEntry, ...]


def tensor_leaves(value: Any, path: str = "") -> tuple[tuple[str, Tensor], ...]:
    """Return tensor leaves from an immutable dataclass/tuple tree."""
    if isinstance(value, Tensor):
        return ((path, value),)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            leaf
            for field in fields(value)
            for leaf in tensor_leaves(
                getattr(value, field.name),
                f"{path}.{field.name}" if path else field.name,
            )
        )
    if isinstance(value, tuple):
        return tuple(
            leaf
            for index, item in enumerate(value)
            for leaf in tensor_leaves(item, f"{path}[{index}]")
        )
    return ()


def replace_tensor_leaves(
    value: Any, values: dict[str, Tensor], path: str = ""
) -> Any:
    """Replace tensor leaves while retaining the immutable tree structure."""
    if isinstance(value, Tensor):
        return values.get(path, value)
    if is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: replace_tensor_leaves(
                getattr(value, field.name),
                values,
                f"{path}.{field.name}" if path else field.name,
            )
            for field in fields(value)
        }
        return replace(value, **changes)
    if isinstance(value, tuple):
        return tuple(
            replace_tensor_leaves(item, values, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return value
