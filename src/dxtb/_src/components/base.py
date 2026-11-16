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
import types
from enum import Enum
from typing import TypeVar
from weakref import WeakKeyDictionary

import torch
from tad_mctc.exceptions import DtypeError
from tad_mctc.tree import Node

from dxtb._src.typing import Any, Self, Tensor, TensorLike, override
from dxtb._src.utils.tensors import normalize_device

__all__ = ["Component", "ComponentCache"]

ConversionMemo = dict[tuple[int, Any, Any], tuple[Any, Any]]

_PRIMITIVE_STATIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    slice,
    torch.dtype,
    torch.device,
)

_NON_TRAVERSABLE_TYPES = (
    *_PRIMITIVE_STATIC_TYPES,
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
)

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


def _contains_tensor_value(obj: Any, seen: set[int] | None = None) -> bool:
    """Return ``True`` if any ``Tensor`` is reachable from ``obj``."""
    if isinstance(obj, Tensor):
        return True
    if isinstance(obj, (*_NON_TRAVERSABLE_TYPES, Enum)):
        return False
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return False
    seen.add(oid)
    if isinstance(obj, dict):
        return any(
            _contains_tensor_value(k, seen) or _contains_tensor_value(v, seen)
            for k, v in obj.items()
        )
    if isinstance(obj, (tuple, list)):
        return any(_contains_tensor_value(v, seen) for v in obj)
    if isinstance(obj, Node) or (
        dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    ):
        return any(
            _contains_tensor_value(getattr(obj, f.name), seen)
            for f in dataclasses.fields(obj)
        )
    slots = _iter_mro_slots(obj)
    for slot in slots:
        if hasattr(obj, slot) and _contains_tensor_value(
            getattr(obj, slot), seen
        ):
            return True
    extra = getattr(obj, "__dict__", None)
    if isinstance(extra, dict):
        return any(
            _contains_tensor_value(v, seen)
            for k, v in extra.items()
            if k not in slots
        )
    return False


def _validate_supported_dataclass(dc: Any) -> None:
    """Reject frozen dataclasses (excluding ``Node``) with tensor-valued ``init=False`` fields."""
    if isinstance(dc, Node):
        return
    params = getattr(dc, "__dataclass_params__", None)
    if not (params is not None and params.frozen):
        return
    for f in dataclasses.fields(dc):
        if not f.init and _contains_tensor_value(getattr(dc, f.name)):
            raise TypeError(
                f"Frozen dataclass '{type(dc).__qualname__}' stores a Tensor "
                f"in init=False field '{f.name}', which cannot be converted "
                "or differentiated safely. Use an init=True field, a mutable "
                "dataclass, a NamedTuple, or tad_mctc.tree.Node instead."
            )


def _uses_default_component_conversion(
    obj: Component | ComponentCache, *, has_device: bool
) -> bool:
    """Return ``True`` when ``obj`` has not overridden default ``.to``/``.type``."""
    cls = type(obj)
    if isinstance(obj, Component):
        return (
            (cls.to is Component.to)
            if has_device
            else (cls.type is Component.type)
        )
    return (
        (cls.to is ComponentCache.to)
        if has_device
        else (cls.type is ComponentCache.type)
    )


def _uses_default_node_conversion(node: Node) -> bool:
    """Return ``True`` when ``node`` uses unmodified ``Node._convert_child`` and ``Node.to``."""
    cls = type(node)
    return cls._convert_child is Node._convert_child and cls.to is Node.to


def _convert_component_value(
    value: Any,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
    memo: ConversionMemo | None = None,
) -> Any:
    """Convert floating tensors and nested containers/nodes to ``(device, dtype)`` with alias preservation."""
    if memo is None:
        memo = {}

    if isinstance(value, Tensor):
        target_dev = value.device if device is None else device
        target_dt = (
            value.dtype
            if (dtype is None or not value.is_floating_point())
            else dtype
        )
        mkey: tuple[int, Any, Any] = (id(value), target_dev, target_dt)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        if value.device == target_dev and value.dtype == target_dt:
            converted_t = value
        elif value.is_floating_point():
            converted_t = value.to(device=target_dev, dtype=target_dt)
        else:
            converted_t = value.to(device=target_dev)
        memo[mkey] = (value, converted_t)
        return converted_t

    if isinstance(value, (Component, ComponentCache)):
        allowed = value.allowed_dtypes
        target_dev = value.device if device is None else device
        target_dt = (
            value.dtype if (dtype is None or dtype not in allowed) else dtype
        )
        mkey = (id(value), target_dev, target_dt)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        if _uses_default_component_conversion(
            value, has_device=device is not None
        ):
            converted_comp = _convert_component_like(
                value, device=target_dev, dtype=target_dt, memo=memo
            )
        elif dtype is not None and dtype not in allowed:
            converted_comp = (
                value
                if value.device == target_dev
                else value.to(device=target_dev)
            )
        elif device is None:
            converted_comp = value.type(target_dt)
        else:
            converted_comp = value.to(device=target_dev, dtype=target_dt)
        memo[mkey] = (value, converted_comp)
        return converted_comp

    if isinstance(value, TensorLike):
        allowed_tl = getattr(value, "allowed_dtypes", None)
        target_dev = value.device if device is None else device
        disallowed_dt = (
            dtype is not None
            and allowed_tl is not None
            and dtype not in allowed_tl
        )
        target_dt = value.dtype if (dtype is None or disallowed_dt) else dtype
        mkey = (id(value), target_dev, target_dt)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        if disallowed_dt:
            converted_tl = (
                value
                if value.device == target_dev
                else value.to(device=target_dev)
            )
        elif device is None:
            converted_tl = value.type(target_dt)
        else:
            converted_tl = value.to(device=target_dev, dtype=target_dt)
        memo[mkey] = (value, converted_tl)
        return converted_tl

    if isinstance(value, Node):
        mkey = (id(value), device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        if not _uses_default_node_conversion(value):
            converted_node = value.to(device=device, dtype=dtype)
        else:
            node_changes: dict[str, Any] = {}
            for name in value._child_names:
                old_child = getattr(value, name)
                child_dt = None if name in value._keep_dtype_names else dtype
                new_child = _convert_component_value(
                    old_child, device=device, dtype=child_dt, memo=memo
                )
                if new_child is not old_child:
                    node_changes[name] = new_child
            converted_node = (
                value.replace(**node_changes)
                if node_changes
                else value.to(device=device, dtype=dtype)
            )
        memo[mkey] = (value, converted_node)
        return converted_node

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _validate_supported_dataclass(value)
        mkey = (id(value), device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        fields = dataclasses.fields(value)
        params = getattr(value, "__dataclass_params__", None)
        is_frozen = bool(params is not None and params.frozen)
        changes: dict[str, Any] = {}
        for f in fields:
            if is_frozen and not f.init:
                continue
            old_v = getattr(value, f.name)
            new_v = _convert_component_value(
                old_v, device=device, dtype=dtype, memo=memo
            )
            if new_v is not old_v:
                changes[f.name] = new_v
        if not changes:
            memo[mkey] = (value, value)
            return value
        if is_frozen:
            converted_dc = dataclasses.replace(value, **changes)
        else:
            converted_dc = copy.copy(value)
            for name, new_v in changes.items():
                setattr(converted_dc, name, new_v)
        memo[mkey] = (value, converted_dc)
        return converted_dc

    if isinstance(value, dict):
        mkey = (id(value), device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        converted_pairs = [
            (
                _convert_component_value(
                    k, device=device, dtype=dtype, memo=memo
                ),
                _convert_component_value(
                    v, device=device, dtype=dtype, memo=memo
                ),
            )
            for k, v in value.items()
        ]
        if all(
            k_c is k and v_c is v
            for (k, v), (k_c, v_c) in zip(value.items(), converted_pairs)
        ):
            memo[mkey] = (value, value)
            return value
        if type(value) is dict:
            converted_dict: Any = dict(converted_pairs)
        else:
            converted_dict = copy.copy(value)
            converted_dict.clear()
            converted_dict.update(converted_pairs)
        memo[mkey] = (value, converted_dict)
        return converted_dict

    if isinstance(value, tuple):
        mkey = (id(value), device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        converted_tuple = tuple(
            _convert_component_value(v, device=device, dtype=dtype, memo=memo)
            for v in value
        )
        if all(a is b for a, b in zip(converted_tuple, value)):
            memo[mkey] = (value, value)
            return value
        if type(value) is tuple:
            converted_tup: Any = converted_tuple
        elif hasattr(type(value), "_fields"):
            converted_tup = type(value)(*converted_tuple)
        else:
            converted_tup = type(value)(converted_tuple)
        memo[mkey] = (value, converted_tup)
        return converted_tup

    if isinstance(value, list):
        mkey = (id(value), device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        converted_list = [
            _convert_component_value(v, device=device, dtype=dtype, memo=memo)
            for v in value
        ]
        if all(a is b for a, b in zip(converted_list, value)):
            memo[mkey] = (value, value)
            return value
        if type(value) is list:
            converted_lst: Any = converted_list
        else:
            converted_lst = copy.copy(value)
            converted_lst[:] = converted_list
        memo[mkey] = (value, converted_lst)
        return converted_lst

    return value


def _value_needs_conversion(
    value: Any,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
    seen: set[int] | None = None,
) -> bool:
    """Return ``True`` if ``value`` holds any tensor/node differing from ``(device, dtype)``."""
    if isinstance(value, Tensor):
        dev_diff = device is not None and value.device != device
        dt_diff = (
            dtype is not None
            and value.is_floating_point()
            and value.dtype != dtype
        )
        return bool(dev_diff or dt_diff)

    if isinstance(value, TensorLike):
        if seen is None:
            seen = set()
        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)
        allowed = getattr(value, "allowed_dtypes", None)
        target_dev = value.device if device is None else device
        target_dt = (
            value.dtype
            if dtype is None or (allowed is not None and dtype not in allowed)
            else dtype
        )
        if isinstance(value, (Component, ComponentCache)):
            return _component_like_needs_conversion(
                value, device=target_dev, dtype=target_dt, seen=seen
            )
        return bool(value.device != target_dev or value.dtype != target_dt)

    if isinstance(value, Node):
        if seen is None:
            seen = set()
        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)
        if not _uses_default_node_conversion(value):
            return value.to(device=device, dtype=dtype) is not value
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
        _validate_supported_dataclass(value)
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
            _value_needs_conversion(k, device=device, dtype=dtype, seen=seen)
            or _value_needs_conversion(v, device=device, dtype=dtype, seen=seen)
            for k, v in value.items()
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
    # Always traverse attributes first so unsupported frozen dataclass fields
    # are rejected even when obj.device/dtype already match the target.
    if seen is None:
        seen = set()
    needs = obj.device != device or obj.dtype != dtype
    slots = _iter_mro_slots(obj)
    for slot in slots:
        if hasattr(obj, slot) and _value_needs_conversion(
            getattr(obj, slot), device=device, dtype=dtype, seen=seen
        ):
            needs = True
    extra_attrs = getattr(obj, "__dict__", None)
    if extra_attrs is not None:
        for key, value in extra_attrs.items():
            if key in slots:
                continue
            if _value_needs_conversion(
                value, device=device, dtype=dtype, seen=seen
            ):
                needs = True
    return needs


_TL = TypeVar("_TL", bound=TensorLike)


def _clone_component_like(
    obj: _TL,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    memo: ConversionMemo | None = None,
) -> _TL:
    """Clone a ``Component`` or ``ComponentCache`` across all MRO slots and ``__dict__``."""
    if memo is None:
        memo = {}
    target_device = obj.device if device is None else device
    target_dtype = obj.dtype if dtype is None else dtype

    new_obj = obj.__class__.__new__(obj.__class__)
    memo[(id(obj), target_device, target_dtype)] = (obj, new_obj)
    for slot in _iter_mro_slots(obj):
        if not hasattr(obj, slot):
            continue
        attr = getattr(obj, slot)
        converted = _convert_component_value(
            attr, device=target_device, dtype=target_dtype, memo=memo
        )
        setattr(new_obj, slot, converted)

    new_obj.override_device(target_device)
    new_obj.override_dtype(target_dtype)

    extra_attrs = getattr(obj, "__dict__", None)
    if extra_attrs is not None:
        slots_set = set(_iter_mro_slots(obj))
        for key, value in extra_attrs.items():
            if key in slots_set:
                continue
            setattr(
                new_obj,
                key,
                _convert_component_value(
                    value, device=target_device, dtype=target_dtype, memo=memo
                ),
            )

    if isinstance(obj, Component) and isinstance(new_obj, Component):
        new_obj.label = getattr(obj, "label", obj.__class__.__name__)

    return new_obj


def _convert_component_like(
    obj: _TL,
    *,
    device: torch.device,
    dtype: torch.dtype,
    memo: ConversionMemo | None = None,
) -> _TL:
    """Convert ``obj`` to ``(device, dtype)`` when needed, preserving aliases via ``memo``."""
    if memo is None:
        memo = {}
    mkey = (id(obj), device, dtype)
    cached = memo.get(mkey)
    if cached is not None:
        return cached[1]
    if not _component_like_needs_conversion(obj, device=device, dtype=dtype):
        memo[mkey] = (obj, obj)
        return obj
    return _clone_component_like(obj, device=device, dtype=dtype, memo=memo)


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
        resolved_device = (
            self.device if device is None else normalize_device(device)
        )
        target_device = (
            self.device if resolved_device is None else resolved_device
        )
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
        reset_memo: dict[int, Tensor] = {}
        slots = _iter_mro_slots(self)
        for slot in slots:
            if not hasattr(self, slot):
                continue
            attr = getattr(self, slot)
            if isinstance(attr, Tensor):
                reset = reset_memo.get(id(attr))
                if reset is None:
                    reset = attr.detach().clone()
                    reset.requires_grad = attr.requires_grad
                    reset_memo[id(attr)] = reset
                setattr(self, slot, reset)

        extra_attrs = getattr(self, "__dict__", None)
        if extra_attrs is not None:
            for key, attr in list(extra_attrs.items()):
                if key in slots:
                    continue
                if isinstance(attr, Tensor):
                    reset = reset_memo.get(id(attr))
                    if reset is None:
                        reset = attr.detach().clone()
                        reset.requires_grad = attr.requires_grad
                        reset_memo[id(attr)] = reset
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
        resolved_device = (
            self.device if device is None else normalize_device(device)
        )
        target_device = (
            self.device if resolved_device is None else resolved_device
        )
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
