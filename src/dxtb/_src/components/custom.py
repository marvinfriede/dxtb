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
"""
Components: Custom
==================

Setup data for user-defined :class:`~dxtb.components.base.Classical` and
:class:`~dxtb.components.base.Interaction` components.

Components and their caches are :class:`~tad_mctc.tree.Node` objects, so a
custom component together with its cache is a pytree: it passes through
:mod:`torch.func` transforms as it is, and its tensors are found and replaced
by flattening and unflattening the pair.
"""

from __future__ import annotations

from tad_mctc.tree import Node, child
from torch.utils import _pytree as pytree

from dxtb._src.scf.core import SCFInteractionEntry
from dxtb._src.typing import Any, Tensor

from .classicals.base import Classical
from .interactions.base import Interaction

__all__ = [
    "CustomClassicalCoreSetup",
    "prepare_custom_interaction_entry",
    "reconstruct_custom_interaction",
    "validate_setup_data",
]


def _is_setup_data(value: Any) -> bool:
    if value is None or isinstance(value, (Tensor, Node)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_setup_data(v) for v in value)
    if isinstance(value, dict):
        return all(_is_setup_data(v) for v in value.values())
    return False


def validate_setup_data(component: Classical | Interaction, data: Any) -> None:
    """
    Check that ``get_cache`` of a custom component returned a
    :class:`~tad_mctc.tree.Node`, a tensor, ``None``, or a list, tuple or dict
    of them, so that its tensors are found by transforms and conversions.

    Raises
    ------
    TypeError
        The setup data has another type, or a container holds another type.
    """
    if not _is_setup_data(data):
        raise TypeError(
            f"'{component.label}.get_cache' must return a Node (such as a "
            "ComponentCache), a tensor, None, or a list, tuple or dict of "
            f"them; got {type(data).__name__}."
        )


class CustomClassicalCoreSetup(Node):
    """
    A custom :class:`~dxtb.components.base.Classical` together with the
    setup data its ``get_cache`` returned.

    The setup data must be a :class:`~tad_mctc.tree.Node` (such as a
    :class:`~dxtb.components.base.ClassicalCache`), a tensor, a container of
    them, or ``None``.
    """

    component: Classical = child()
    """The custom classical component."""

    cache: Any = child(keep_dtype=True)
    """
    Setup data of the component for one System. Its floating-point dtype may
    differ from the component's, so it is converted explicitly.
    """

    def evaluate_energy(self, positions: Tensor, charge: Tensor) -> Tensor:
        """Evaluate the atom-resolved energy of the component."""
        energy: Tensor = self.component.get_energy(
            positions, self.cache, charge=charge
        )
        return energy


class _TreeSpec:
    """
    Holder of a pytree structure in the static part of an SCF entry; not a
    dataclass or tuple, so the SCF tensor-leaf walkers do not descend into
    it.
    """

    __slots__ = ("spec",)

    def __init__(self, spec: pytree.TreeSpec) -> None:
        self.spec = spec


def prepare_custom_interaction_entry(
    component: Interaction, cache: Any
) -> SCFInteractionEntry:
    """
    Store a custom interaction and its cache as an SCF entry.

    The tensors of the pair ``(component, cache)`` become the entry's data,
    and the tree structure is kept to rebuild the pair from (possibly
    transformed) tensors.
    """
    validate_setup_data(component, cache)
    leaves, treespec = pytree.tree_flatten((component, cache))
    return SCFInteractionEntry(
        kind="custom",
        data=tuple((f"leaf_{i}", leaf) for i, leaf in enumerate(leaves)),
        static=(("treespec", _TreeSpec(treespec)), ("label", component.label)),
    )


def reconstruct_custom_interaction(
    entry: SCFInteractionEntry,
) -> tuple[Interaction, Any]:
    """Rebuild ``(component, cache)`` from a custom SCF entry."""
    treespec = entry.option("treespec").spec
    component, cache = pytree.tree_unflatten(
        [leaf for _, leaf in entry.data], treespec
    )
    return component, cache
