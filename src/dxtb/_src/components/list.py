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
Components: List
================

Base class for all collections (lists) of tight-binding components.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import torch

from dxtb._src.typing import Any, Generic, Self, Tensor, TypeVar

from .base import Component

__all__ = ["ComponentList", "ComponentListCache"]


class ComponentListCacheABC(ABC, dict):
    """
    List of component caches.

    Note
    ----
    The cache class inherits from `dict`.
    """


class ComponentListCache(ComponentListCacheABC):
    """
    List of restart data for individual components.
    """

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({list(self.keys())})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


################################################################################


class ComponentListABC(ABC):
    """
    Abstract base class for component lists.
    """

    @abstractmethod
    def get_energy(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Compute the energy for a list of components.
        """

    @abstractmethod
    def get_gradient(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Compute the gradient for a list of components.
        """

    @abstractmethod
    def get_cache(self, *args: Any, **kwargs: Any) -> ComponentListCacheABC:
        """
        Create restart data for all components.
        """


C = TypeVar("C", bound=Component)


class ComponentList(ComponentListABC, Generic[C]):
    """
    List of tight-binding components.

    A plain container: the components themselves are immutable
    :class:`~tad_mctc.tree.Node` objects, and :meth:`to`/:meth:`type` return
    a new list of converted components.
    """

    components: list[C]
    """List of tight-binding component instances."""

    __slots__ = ("components",)

    def __init__(self, *components: C | None) -> None:
        """
        Instantiate a collection of tight-binding components.

        Parameters
        ----------
        components : C | None
            Components; ``None`` entries are skipped and repeated objects are
            kept once, in input order.
        """
        self.components = list(
            dict.fromkeys(c for c in components if c is not None)
        )

    @property
    def labels(self) -> list[str]:
        """Alphabetically sorted list of all components labels."""
        return sorted([component.label for component in self.components])

    def get_interaction(self, name: str) -> C:
        """
        Obtain an component from the list of components by its label.

        Parameters
        ----------
        name : str
            Label of the component.

        Returns
        -------
        Component
            Instance of the component as present in the list.

        Raises
        ------
        ValueError
            Unknown component name given or component is not in the list.
        """
        for component in self.components:
            if name == component.label:
                return component
        raise ValueError(f"The component named '{name}' is not in the list.")

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return a new list with all components converted."""
        new = copy.copy(self)
        new.components = [
            c.to(device=device, dtype=dtype) for c in self.components
        ]
        return new

    def type(self, dtype: torch.dtype) -> Self:
        """Return a new list with all components converted to ``dtype``."""
        return self.to(dtype=dtype)

    def __len__(self) -> int:
        return len(self.components)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.labels})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
