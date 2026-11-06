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

Base class for all tight-binding components.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Any, Self, Tensor, TensorLike, override
from dxtb._src.utils.misc import get_all_slots

__all__ = ["Component", "ComponentCache"]


class ComponentCache(TensorLike):
    """Cache of a component."""

    def __len__(self) -> int:
        slots = get_all_slots(self)
        return len([s for s in slots if not s.startswith("_")])

    def __str__(self) -> str:  # pragma: no cover
        slots = get_all_slots(self)
        s = ", ".join(s for s in slots if not s.startswith("_"))
        return f"{self.__class__.__name__}({s})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


class Component(TensorLike):
    """
    Base class for all tight-binding terms.
    """

    label: str
    """Label for the tight-binding component."""

    __slots__ = ["label"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.label = self.__class__.__name__

    ############################################################################

    def update(self, **kwargs: Any) -> None:
        """
        Update the attributes of the :class:`.Component` instance.

        This method updates the attributes of the :class:`.Component`
        instance based on the provided keyword arguments. Only the attributes
        defined in ``__slots__`` can be updated.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Keyword arguments where keys are attribute names and values are the
            new values for those attributes.
            Valid keys are those defined in ``__slots__`` of this class.

        Raises
        ------
        AttributeError
            If any key in kwargs is not an attribute defined in ``__slots__``.

        Examples
        --------

        .. code-block:: python

            component = SomeMutableComponent(parameter=initial_value)
            component.update(parameter=new_value)
        """
        for key, value in kwargs.items():
            if key is None:
                continue

            if key in self.__slots__:
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"Cannot update '{key}' of the '{self.__class__.__name__}' "
                    "interaction. Invalid attribute.\nPossible attributes are: "
                    f"{', '.join(self.__slots__)}."
                )

    def reset(self) -> None:
        """
        Reset the tensor attributes of the
        :class:`dxtb.components.base.Component` instance to their original
        states or to specified values.

        This method iterates through the attributes defined in ``__slots__`` and
        resets any tensor attributes to a detached clone of their original
        state. The `requires_grad` status of each tensor is preserved.

        Examples
        --------

        .. code-block:: python

            component = SomeMutableComponent(parameter=initial_value)
            component.reset()

        Notes
        -----
        Only tensor attributes defined in ``__slots__`` are reset. Non-tensor
        attributes are ignored. Attempting to reset an attribute not defined in
        ``__slots__`` or providing a non-tensor value in `kwargs` will not raise
        an error; the method will simply ignore these cases and proceed with
        the reset operation for valid tensor attributes.
        """
        for slot in self.__slots__:
            attr = getattr(self, slot)
            if isinstance(attr, Tensor):
                reset = attr.detach().clone()
                reset.requires_grad = attr.requires_grad

                setattr(self, slot, reset)

    ############################################################################

    @override
    def type(self, dtype: torch.dtype) -> Self:
        new = super().type(dtype)
        new.label = getattr(self, "label", self.__class__.__name__)
        return new

    ############################################################################

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"

    def __repr__(self) -> str:
        return str(self)
