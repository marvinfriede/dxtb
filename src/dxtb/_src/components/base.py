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

import copy
import dataclasses
import functools
from typing import TypeVar
from weakref import WeakKeyDictionary

import torch
from tad_mctc.exceptions import DtypeError
from tad_mctc.tree import Node

from dxtb._src.typing import Any, Self, Tensor, TensorLike, override

__all__ = ["Component", "ComponentCache"]

_COMPONENT_MUTATION_EPOCHS: WeakKeyDictionary[Component, int] = (
    WeakKeyDictionary()
)


def _iter_mro_slots(obj: object) -> tuple[str, ...]:
    """Return instance slot attribute names across ``type(obj).__mro__`` in parent-first order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for cls in reversed(type(obj).__mro__):
        raw_slots = cls.__dict__.get("__slots__", ())
        slots = (raw_slots,) if isinstance(raw_slots, str) else raw_slots
        for slot in slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            if cls is TensorLike and slot in ("__device", "__dtype"):
                continue
            if slot.startswith("__") and not slot.endswith("__"):
                prefix = cls.__name__.lstrip("_")
                attr_name = f"_{prefix}{slot}" if prefix else slot
            else:
                attr_name = slot
            if attr_name in seen:
                continue
            seen.add(attr_name)
            ordered.append(attr_name)
    return tuple(ordered)


def _convert_component_value(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Convert floating tensors and nested containers/nodes to ``(device, dtype)``."""
    if isinstance(value, Tensor):
        if value.is_floating_point():
            if value.device == device and value.dtype == dtype:
                return value
            return value.to(device=device, dtype=dtype)
        if value.device == device:
            return value
        return value.to(device=device)

    if isinstance(value, TensorLike):
        allowed = getattr(value, "allowed_dtypes", None)
        if allowed is not None and dtype not in allowed:
            if value.device == device:
                return value
            return value.to(device=device)
        return value.to(device=device, dtype=dtype)

    if isinstance(value, Node):
        return value.to(device=device, dtype=dtype)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        params = getattr(value, "__dataclass_params__", None)
        is_frozen = bool(params is not None and params.frozen)
        changes: dict[str, Any] = {}
        for f in fields:
            if is_frozen and not f.init:
                continue
            old_v = getattr(value, f.name)
            new_v = _convert_component_value(old_v, device=device, dtype=dtype)
            if new_v is not old_v:
                changes[f.name] = new_v
        if not changes:
            return value
        if is_frozen:
            return dataclasses.replace(value, **changes)
        dc_copy = copy.copy(value)
        for name, new_v in changes.items():
            setattr(dc_copy, name, new_v)
        return dc_copy

    if isinstance(value, dict):
        converted_pairs = [
            (k, _convert_component_value(v, device=device, dtype=dtype))
            for k, v in value.items()
        ]
        if all(v is value[k] for k, v in converted_pairs):
            return value
        if type(value) is dict:
            return dict(converted_pairs)
        dict_copy = copy.copy(value)
        dict_copy.clear()
        dict_copy.update(converted_pairs)
        return dict_copy

    if isinstance(value, tuple):
        converted_tuple = tuple(
            _convert_component_value(v, device=device, dtype=dtype)
            for v in value
        )
        if all(a is b for a, b in zip(converted_tuple, value)):
            return value
        if type(value) is tuple:
            return converted_tuple
        if hasattr(type(value), "_fields"):
            return type(value)(*converted_tuple)
        return type(value)(converted_tuple)

    if isinstance(value, list):
        converted_list = [
            _convert_component_value(v, device=device, dtype=dtype)
            for v in value
        ]
        if all(a is b for a, b in zip(converted_list, value)):
            return value
        if type(value) is list:
            return converted_list
        list_copy = copy.copy(value)
        list_copy[:] = converted_list
        return list_copy

    return value


def _value_needs_conversion(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    seen: set[int] | None = None,
) -> bool:
    """Return ``True`` if ``value`` holds any tensor/node differing from ``(device, dtype)``."""
    if isinstance(value, Tensor):
        if dtype is not None and value.is_floating_point():
            return bool(value.device != device or value.dtype != dtype)
        return bool(value.device != device)

    if isinstance(value, TensorLike):
        if seen is None:
            seen = set()
        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)
        allowed = getattr(value, "allowed_dtypes", None)
        target_dt = (
            value.dtype
            if dtype is None or (allowed is not None and dtype not in allowed)
            else dtype
        )
        if value.device != device or value.dtype != target_dt:
            return True
        if isinstance(value, (Component, ComponentCache)):
            return _component_like_needs_conversion(
                value, device=device, dtype=target_dt, seen=seen
            )
        return False

    if isinstance(value, Node):
        if seen is None:
            seen = set()
        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)
        return any(
            _value_needs_conversion(
                getattr(value, name),
                device=device,
                dtype=(None if name in value._keep_dtype_names else dtype),
                seen=seen,
            )
            for name in value._child_names
        )

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if seen is None:
            seen = set()
        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)
        params = getattr(value, "__dataclass_params__", None)
        is_frozen = bool(params is not None and params.frozen)
        return any(
            _value_needs_conversion(
                getattr(value, f.name), device=device, dtype=dtype, seen=seen
            )
            for f in dataclasses.fields(value)
            if not is_frozen or f.init
        )

    if isinstance(value, dict):
        if seen is None:
            seen = set()
        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)
        return any(
            _value_needs_conversion(v, device=device, dtype=dtype, seen=seen)
            for v in value.values()
        )

    if isinstance(value, (tuple, list)):
        if seen is None:
            seen = set()
        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)
        return any(
            _value_needs_conversion(v, device=device, dtype=dtype, seen=seen)
            for v in value
        )

    return False


def _component_like_needs_conversion(
    obj: TensorLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seen: set[int] | None = None,
) -> bool:
    if obj.device != device or obj.dtype != dtype:
        return True
    if seen is None:
        seen = set()
    slots = _iter_mro_slots(obj)
    for slot in slots:
        if hasattr(obj, slot) and _value_needs_conversion(
            getattr(obj, slot), device=device, dtype=dtype, seen=seen
        ):
            return True
    extra_attrs = getattr(obj, "__dict__", None)
    if extra_attrs is not None:
        for key, value in extra_attrs.items():
            if key in slots:
                continue
            if _value_needs_conversion(
                value, device=device, dtype=dtype, seen=seen
            ):
                return True
    return False


_TL = TypeVar("_TL", bound=TensorLike)


def _clone_component_like(
    obj: _TL,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> _TL:
    """Clone a ``Component`` or ``ComponentCache`` across all MRO slots and ``__dict__``."""
    target_device = device if device is not None else obj.device
    target_dtype = dtype if dtype is not None else obj.dtype

    new_obj = obj.__class__.__new__(obj.__class__)
    for slot in _iter_mro_slots(obj):
        if not hasattr(obj, slot):
            continue
        attr = getattr(obj, slot)
        converted = _convert_component_value(
            attr, device=target_device, dtype=target_dtype
        )
        setattr(new_obj, slot, converted)

    new_obj.override_device(target_device)
    new_obj.override_dtype(target_dtype)

    extra_attrs = getattr(obj, "__dict__", None)
    if extra_attrs is not None:
        for key, value in extra_attrs.items():
            setattr(
                new_obj,
                key,
                _convert_component_value(
                    value, device=target_device, dtype=target_dtype
                ),
            )

    return new_obj


def component_mutation_epoch(component: Component) -> int:
    """Return the monotonic ``update``/``reset`` epoch recorded for ``component``."""
    try:
        return _COMPONENT_MUTATION_EPOCHS.get(component, 0)
    except TypeError:
        return 0


def _bump_component_mutation_epoch(component: Component) -> None:
    try:
        _COMPONENT_MUTATION_EPOCHS[component] = (
            _COMPONENT_MUTATION_EPOCHS.get(component, 0) + 1
        )
    except TypeError:
        pass


class ComponentCache(TensorLike):
    """Cache of a component."""

    @override
    def _clone_tensorlike(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        return _clone_component_like(self, device=device, dtype=dtype)

    @override
    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        if device is None and dtype is None:
            return self
        target_device = device if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.dtype
        if target_dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{target_dtype}')."
            )
        if not _component_like_needs_conversion(
            self, device=target_device, dtype=target_dtype
        ):
            return self
        return self._clone_tensorlike(device=target_device, dtype=target_dtype)

    @override
    def type(self, dtype: torch.dtype) -> Self:
        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )
        if not _component_like_needs_conversion(
            self, device=self.device, dtype=dtype
        ):
            return self
        return self._clone_tensorlike(device=None, dtype=dtype)

    def __len__(self) -> int:
        slots = [s for s in _iter_mro_slots(self) if not s.startswith("_")]
        extra = [
            k
            for k in getattr(self, "__dict__", ())
            if not k.startswith("_") and k not in slots
        ]
        return len(slots) + len(extra)

    def __str__(self) -> str:  # pragma: no cover
        slots = [s for s in _iter_mro_slots(self) if not s.startswith("_")]
        extra = [
            k
            for k in getattr(self, "__dict__", ())
            if not k.startswith("_") and k not in slots
        ]
        s = ", ".join([*slots, *extra])
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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for method_name in ("update", "reset"):
            raw = cls.__dict__.get(method_name)
            if raw is None or not callable(raw):
                continue

            def _make_wrapped(fn: Any) -> Any:
                @functools.wraps(fn)
                def _wrapped(
                    self: Component, *m_args: Any, **m_kwargs: Any
                ) -> Any:
                    result = fn(self, *m_args, **m_kwargs)
                    _bump_component_mutation_epoch(self)
                    return result

                return _wrapped

            setattr(cls, method_name, _make_wrapped(raw))

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
        instance based on the provided keyword arguments. Attributes defined in
        ``__slots__`` across the class hierarchy or already present on the
        instance can be updated.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Keyword arguments where keys are attribute names and values are the
            new values for those attributes.

        Raises
        ------
        AttributeError
            If any key in kwargs is not a valid attribute of this component.

        Examples
        --------

        .. code-block:: python

            component = SomeMutableComponent(parameter=initial_value)
            component.update(parameter=new_value)
        """
        slots = _iter_mro_slots(self)
        extra_attrs = getattr(self, "__dict__", {})
        valid_keys = (*slots, *(k for k in extra_attrs if k not in slots))
        updated = False
        for key, value in kwargs.items():
            if key in valid_keys:
                setattr(self, key, value)
                updated = True
            else:
                raise AttributeError(
                    f"Cannot update '{key}' of the '{self.__class__.__name__}' "
                    "interaction. Invalid attribute.\nPossible attributes are: "
                    f"{', '.join(valid_keys)}."
                )
        if updated:
            _bump_component_mutation_epoch(self)

    def reset(self) -> None:
        """
        Reset the tensor attributes of the
        :class:`dxtb.components.base.Component` instance to their original
        states or to specified values.

        This method iterates through the component attributes and resets any
        tensor attributes to a detached clone of their current state. The
        `requires_grad` status of each tensor is preserved.

        Examples
        --------

        .. code-block:: python

            component = SomeMutableComponent(parameter=initial_value)
            component.reset()
        """
        slots = _iter_mro_slots(self)
        for slot in slots:
            if not hasattr(self, slot):
                continue
            attr = getattr(self, slot)
            if isinstance(attr, Tensor):
                reset = attr.detach().clone()
                reset.requires_grad = attr.requires_grad
                setattr(self, slot, reset)

        extra_attrs = getattr(self, "__dict__", None)
        if extra_attrs is not None:
            for key, attr in list(extra_attrs.items()):
                if key in slots:
                    continue
                if isinstance(attr, Tensor):
                    reset = attr.detach().clone()
                    reset.requires_grad = attr.requires_grad
                    setattr(self, key, reset)
        _bump_component_mutation_epoch(self)

    ############################################################################

    @override
    def _clone_tensorlike(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        return _clone_component_like(self, device=device, dtype=dtype)

    @override
    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        if device is None and dtype is None:
            return self
        target_device = device if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.dtype
        if target_dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{target_dtype}')."
            )
        if not _component_like_needs_conversion(
            self, device=target_device, dtype=target_dtype
        ):
            return self
        new = self._clone_tensorlike(device=target_device, dtype=target_dtype)
        new.label = getattr(self, "label", self.__class__.__name__)
        return new

    @override
    def type(self, dtype: torch.dtype) -> Self:
        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )
        if not _component_like_needs_conversion(
            self, device=self.device, dtype=dtype
        ):
            return self
        new = self._clone_tensorlike(device=None, dtype=dtype)
        new.label = getattr(self, "label", self.__class__.__name__)
        return new

    ############################################################################

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"

    def __repr__(self) -> str:
        return str(self)
