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
Components: Base Class
======================

Base classes for all tight-binding components and their caches.

A component is an immutable :class:`~tad_mctc.tree.Node`: a frozen dataclass
whose fields are declared with :func:`~tad_mctc.tree.child` (tensors, nodes,
containers of them, or static hashable values) and
:func:`~tad_mctc.tree.context` (static hashable values). Being a registered
pytree, a component passes directly through :mod:`torch.func` transforms and
``torch.compile``, and ``.to()``/``.type()`` convert it field by field.

A component is never changed in place. To change a parameter, create a new
component with :meth:`~tad_mctc.tree.Node.replace` and build a new
:class:`~dxtb.Calculator` (or System) from it.

Example
-------

.. code-block:: python

    import torch
    from dxtb.components.base import Classical, ClassicalCache
    from dxtb.typing import Tensor
    from tad_mctc.tree import child


    class RestraintCache(ClassicalCache):
        force_constant: Tensor = child()


    class Restraint(Classical):
        force_constant: Tensor = child()

        def get_cache(self, numbers, ihelp=None, **kwargs):
            return RestraintCache(force_constant=self.force_constant)

        def get_energy(self, positions, cache, **kwargs):
            return 0.5 * cache.force_constant * positions.square().sum(-1)


    restraint = Restraint(force_constant=torch.tensor(0.05))
    stiffer = restraint.replace(force_constant=torch.tensor(0.10))
"""

from __future__ import annotations

import torch
from tad_mctc.tree import Node, context

from dxtb._src.typing import Any, Tensor

__all__ = ["Component", "ComponentCache", "numbers_to_tensors"]


def numbers_to_tensors(
    node: Node, names: tuple[str, ...], like: Tensor
) -> dict[str, Tensor]:
    """
    Return the fields ``names`` of ``node`` that hold a Python number,
    converted to tensors on the device and with the dtype of ``like``. Meant
    for ``_normalize``, so that scalar parameters may be given as numbers.
    """
    return {
        name: torch.tensor(value, device=like.device, dtype=like.dtype)
        for name in names
        if isinstance(value := getattr(node, name), (int, float))
        and not isinstance(value, bool)
    }


class ComponentCache(Node):
    """
    Base class for the setup data of a component.

    Subclasses declare their fields with :func:`~tad_mctc.tree.child` and
    :func:`~tad_mctc.tree.context`, like any :class:`~tad_mctc.tree.Node`.
    """


class Component(Node):
    """
    Base class for all tight-binding terms.

    The ``label`` identifies the component in a Model and in the results; it
    defaults to the class name. Labels of the components of one Model must be
    unique.
    """

    label: str = context(default="")
    """Label of the component (defaults to the class name)."""

    def _normalize(self) -> dict[str, Any]:
        return {} if self.label else {"label": type(self).__name__}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"
