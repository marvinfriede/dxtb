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
from collections import deque
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from enum import Enum
from typing import NoReturn, TypeVar

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


def _raise_cycle_error(obj: object) -> NoReturn:
    """Raise a deterministic ValueError for cyclic component/cache object graphs."""
    raise ValueError(
        "Circular reference detected in component object graph involving "
        f"'{type(obj).__qualname__}'; cyclic component or cache graphs are "
        "not supported."
    )


def _contains_tensor_value(
    obj: Any,
    *,
    active: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
) -> bool:
    """Return ``True`` if any ``Tensor`` is reachable from ``obj``, rejecting cycles."""
    if isinstance(obj, Tensor):
        return True
    if isinstance(obj, (*_NON_TRAVERSABLE_TYPES, Enum)):
        return False
    if active is None:
        active = set()
    if tensor_memo is None:
        tensor_memo = {}
    oid = id(obj)
    if oid in active:
        _raise_cycle_error(obj)
    cached = tensor_memo.get(oid)
    if cached is not None:
        return cached
    active.add(oid)
    try:
        has_tensor = False
        if isinstance(obj, functools.partial):
            if _contains_tensor_value(
                obj.func, active=active, tensor_memo=tensor_memo
            ):
                has_tensor = True
            for arg in obj.args:
                if _contains_tensor_value(
                    arg, active=active, tensor_memo=tensor_memo
                ):
                    has_tensor = True
            if obj.keywords:
                for val in obj.keywords.values():
                    if _contains_tensor_value(
                        val, active=active, tensor_memo=tensor_memo
                    ):
                        has_tensor = True
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                if _contains_tensor_value(
                    k, active=active, tensor_memo=tensor_memo
                ):
                    has_tensor = True
                if _contains_tensor_value(
                    v, active=active, tensor_memo=tensor_memo
                ):
                    has_tensor = True
        elif isinstance(obj, (tuple, list, deque, AbstractSet)):
            for v in obj:
                if _contains_tensor_value(
                    v, active=active, tensor_memo=tensor_memo
                ):
                    has_tensor = True
        elif isinstance(obj, Node) or (
            dataclasses.is_dataclass(obj) and not isinstance(obj, type)
        ):
            for f in dataclasses.fields(obj):
                if _contains_tensor_value(
                    getattr(obj, f.name),
                    active=active,
                    tensor_memo=tensor_memo,
                ):
                    has_tensor = True
        else:
            slots = _iter_mro_slots(obj)
            for slot in slots:
                if hasattr(obj, slot) and _contains_tensor_value(
                    getattr(obj, slot),
                    active=active,
                    tensor_memo=tensor_memo,
                ):
                    has_tensor = True
            extra = getattr(obj, "__dict__", None)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k not in slots and _contains_tensor_value(
                        v, active=active, tensor_memo=tensor_memo
                    ):
                        has_tensor = True
        tensor_memo[oid] = has_tensor
        return has_tensor
    finally:
        active.discard(oid)


def _validate_supported_dataclass(
    dc: Any,
    *,
    active: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
) -> None:
    """Reject frozen dataclasses (excluding ``Node``) with tensor-valued ``init=False`` fields."""
    if isinstance(dc, Node):
        return
    params = getattr(dc, "__dataclass_params__", None)
    if not (params is not None and params.frozen):
        return
    for f in dataclasses.fields(dc):
        if not f.init and _contains_tensor_value(
            getattr(dc, f.name), active=active, tensor_memo=tensor_memo
        ):
            raise TypeError(
                f"Frozen dataclass '{type(dc).__qualname__}' stores a Tensor "
                f"in init=False field '{f.name}', which cannot be converted "
                "or differentiated safely. Use an init=True field, a mutable "
                "dataclass, a NamedTuple, or tad_mctc.tree.Node instead."
            )


def _validate_opaque_object(
    obj: Any,
    *,
    active: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
) -> None:
    """Ensure an opaque external object contains neither cycles nor hidden ``Tensor`` state."""
    if isinstance(obj, (*_NON_TRAVERSABLE_TYPES, Enum)):
        return
    if _contains_tensor_value(obj, active=active, tensor_memo=tensor_memo):
        raise TypeError(
            f"Opaque object of type '{type(obj).__qualname__}' stores a Tensor "
            "outside supported container types (Component, ComponentCache, "
            "TensorLike, tad_mctc.tree.Node, dataclass, tuple, NamedTuple, "
            "list, dict), so its tensors cannot be converted or differentiated "
            "safely."
        )


def _validate_node_contexts(
    node: Node,
    *,
    active: set[int] | None = None,
    visited: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
) -> None:
    """Ensure ``Node`` context fields contain neither cycles nor ``Tensor``s."""
    for ctx_name in node._context_names:
        ctx_val = getattr(node, ctx_name)
        _validate_component_object_graph(
            ctx_val,
            active=active,
            visited=visited,
            tensor_memo=tensor_memo,
        )
        if _contains_tensor_value(
            ctx_val, active=active, tensor_memo=tensor_memo
        ):
            raise TypeError(
                f"Node '{type(node).__qualname__}' stores a Tensor inside "
                f"context() field '{ctx_name}', which cannot be converted or "
                "differentiated safely. Declare tensor-bearing fields with "
                "child() instead of context()."
            )


def _validate_component_object_graph(
    obj: Any,
    *,
    active: set[int] | None = None,
    visited: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
) -> None:
    """Validate that ``obj`` has no cycles, frozen ``init=False`` tensors, or opaque tensors."""
    if isinstance(obj, (Tensor, *_NON_TRAVERSABLE_TYPES, Enum)):
        return
    if active is None:
        active = set()
    if visited is None:
        visited = set()
    if tensor_memo is None:
        tensor_memo = {}
    oid = id(obj)
    if oid in active:
        _raise_cycle_error(obj)
    if oid in visited:
        return

    if not isinstance(obj, (dict, tuple, list, Node, TensorLike)) and not (
        dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    ):
        _validate_opaque_object(obj, active=active, tensor_memo=tensor_memo)
        visited.add(oid)
        return

    active.add(oid)
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _validate_component_object_graph(
                    k,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                )
                _validate_component_object_graph(
                    v,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                )
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                _validate_component_object_graph(
                    item,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                )
        elif isinstance(obj, Node):
            _validate_node_contexts(
                obj, active=active, visited=visited, tensor_memo=tensor_memo
            )
            for child_name in obj._child_names:
                _validate_component_object_graph(
                    getattr(obj, child_name),
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                )
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            _validate_supported_dataclass(
                obj, active=active, tensor_memo=tensor_memo
            )
            for f in dataclasses.fields(obj):
                _validate_component_object_graph(
                    getattr(obj, f.name),
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                )
        else:
            assert isinstance(obj, TensorLike)
            slots = _iter_mro_slots(obj)
            for slot in slots:
                if hasattr(obj, slot):
                    _validate_component_object_graph(
                        getattr(obj, slot),
                        active=active,
                        visited=visited,
                        tensor_memo=tensor_memo,
                    )
            extra = getattr(obj, "__dict__", None)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k not in slots:
                        _validate_component_object_graph(
                            v,
                            active=active,
                            visited=visited,
                            tensor_memo=tensor_memo,
                        )
        visited.add(oid)
    finally:
        active.discard(oid)


def _uses_default_tensorlike_conversion(
    obj: TensorLike, *, use_to: bool
) -> bool:
    """Return ``True`` when ``obj`` has not overridden default ``.to``/``.type``."""
    cls = type(obj)
    if isinstance(obj, Component):
        return (
            (
                cls.to is Component.to
                and cls._clone_tensorlike is Component._clone_tensorlike
            )
            if use_to
            else (
                cls.type is Component.type
                and cls._clone_tensorlike is Component._clone_tensorlike
            )
        )
    if isinstance(obj, ComponentCache):
        return (
            (
                cls.to is ComponentCache.to
                and cls._clone_tensorlike is ComponentCache._clone_tensorlike
            )
            if use_to
            else (
                cls.type is ComponentCache.type
                and cls._clone_tensorlike is ComponentCache._clone_tensorlike
            )
        )
    return (
        (
            cls.to is TensorLike.to
            and cls._clone_tensorlike is TensorLike._clone_tensorlike
        )
        if use_to
        else (
            cls.type is TensorLike.type
            and cls._clone_tensorlike is TensorLike._clone_tensorlike
        )
    )


def _uses_default_node_conversion(
    node: Node, *, use_to: bool | None = None
) -> bool:
    """Return ``True`` when ``node`` uses unmodified ``Node`` conversion hooks."""
    cls = type(node)
    if use_to is True:
        return cls._convert_child is Node._convert_child and cls.to is Node.to
    return (
        cls._convert_child is Node._convert_child
        and cls.to is Node.to
        and cls.type is Node.type
    )


def _convert_component_value(
    value: Any,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
    memo: ConversionMemo | None = None,
    active: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
    use_to: bool | None = None,
    explicit_device: bool | None = None,
) -> Any:
    """Convert floating tensors and nested containers/nodes to ``(device, dtype)`` with alias preservation."""
    if memo is None:
        memo = {}
    if active is None:
        active = set()
    if tensor_memo is None:
        tensor_memo = {}
    via_to = (device is not None) if use_to is None else use_to
    has_dev = (
        (device is not None) if explicit_device is None else explicit_device
    )

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

    if isinstance(value, (*_NON_TRAVERSABLE_TYPES, Enum)):
        return value

    oid = id(value)
    if oid in active:
        _raise_cycle_error(value)

    if isinstance(value, TensorLike):
        allowed_tl = getattr(value, "allowed_dtypes", None)
        target_dev = value.device if device is None else device
        disallowed_dt = (
            dtype is not None
            and allowed_tl is not None
            and dtype not in allowed_tl
        )
        target_dt = value.dtype if (dtype is None or disallowed_dt) else dtype
        mkey = (oid, target_dev, target_dt)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        if _uses_default_tensorlike_conversion(value, use_to=via_to):
            return _convert_component_like(
                value,
                device=target_dev,
                dtype=target_dt,
                memo=memo,
                active=active,
                tensor_memo=tensor_memo,
                use_to=via_to,
                explicit_device=has_dev,
            )
        _validate_component_object_graph(
            value, active=active, tensor_memo=tensor_memo
        )
        active.add(oid)
        try:
            if disallowed_dt:
                converted_tl = (
                    value
                    if value.device == target_dev
                    else value.to(device=target_dev)
                )
            elif not via_to:
                converted_tl = value.type(target_dt)
            else:
                converted_tl = value.to(
                    device=target_dev if has_dev else None, dtype=target_dt
                )
        finally:
            active.discard(oid)
        memo[mkey] = (value, converted_tl)
        return converted_tl

    if isinstance(value, Node):
        mkey = (oid, device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        if not _uses_default_node_conversion(value, use_to=via_to):
            _validate_component_object_graph(
                value, active=active, tensor_memo=tensor_memo
            )
            active.add(oid)
            try:
                if not via_to:
                    converted_node = (
                        value if dtype is None else value.type(dtype)
                    )
                else:
                    converted_node = value.to(
                        device=device if has_dev else None, dtype=dtype
                    )
            finally:
                active.discard(oid)
        else:
            active.add(oid)
            try:
                _validate_node_contexts(
                    value, active=active, tensor_memo=tensor_memo
                )
                node_changes: dict[str, Any] = {}
                for name in value._child_names:
                    old_child = getattr(value, name)
                    child_dt = (
                        None if name in value._keep_dtype_names else dtype
                    )
                    new_child = _convert_component_value(
                        old_child,
                        device=device,
                        dtype=child_dt,
                        memo=memo,
                        active=active,
                        tensor_memo=tensor_memo,
                        use_to=via_to,
                        explicit_device=has_dev,
                    )
                    if new_child is not old_child:
                        node_changes[name] = new_child
                converted_node = (
                    value.replace(**node_changes) if node_changes else value
                )
            finally:
                active.discard(oid)
        memo[mkey] = (value, converted_node)
        return converted_node

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        active.add(oid)
        try:
            _validate_supported_dataclass(
                value, active=active, tensor_memo=tensor_memo
            )
            mkey = (oid, device, dtype)
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
                    old_v,
                    device=device,
                    dtype=dtype,
                    memo=memo,
                    active=active,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                )
                if new_v is not old_v:
                    changes[f.name] = new_v
        finally:
            active.discard(oid)
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
        mkey = (oid, device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        active.add(oid)
        try:
            converted_pairs = [
                (
                    _convert_component_value(
                        k,
                        device=device,
                        dtype=dtype,
                        memo=memo,
                        active=active,
                        tensor_memo=tensor_memo,
                        use_to=via_to,
                        explicit_device=has_dev,
                    ),
                    _convert_component_value(
                        v,
                        device=device,
                        dtype=dtype,
                        memo=memo,
                        active=active,
                        tensor_memo=tensor_memo,
                        use_to=via_to,
                        explicit_device=has_dev,
                    ),
                )
                for k, v in value.items()
            ]
        finally:
            active.discard(oid)
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
        mkey = (oid, device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        active.add(oid)
        try:
            converted_tuple = tuple(
                _convert_component_value(
                    v,
                    device=device,
                    dtype=dtype,
                    memo=memo,
                    active=active,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                )
                for v in value
            )
        finally:
            active.discard(oid)
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
        mkey = (oid, device, dtype)
        cached = memo.get(mkey)
        if cached is not None:
            return cached[1]
        active.add(oid)
        try:
            converted_list = [
                _convert_component_value(
                    v,
                    device=device,
                    dtype=dtype,
                    memo=memo,
                    active=active,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                )
                for v in value
            ]
        finally:
            active.discard(oid)
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

    _validate_opaque_object(value, active=active, tensor_memo=tensor_memo)
    return value


def _value_needs_conversion(
    value: Any,
    *,
    device: torch.device | None,
    dtype: torch.dtype | None,
    active: set[int] | None = None,
    visited: dict[tuple[int, Any, Any], bool] | None = None,
    tensor_memo: dict[int, bool] | None = None,
    use_to: bool | None = None,
    explicit_device: bool | None = None,
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

    if isinstance(value, (*_NON_TRAVERSABLE_TYPES, Enum)):
        return False

    if active is None:
        active = set()
    if visited is None:
        visited = {}
    if tensor_memo is None:
        tensor_memo = {}
    via_to = (device is not None) if use_to is None else use_to
    has_dev = (
        (device is not None) if explicit_device is None else explicit_device
    )
    oid = id(value)
    if oid in active:
        _raise_cycle_error(value)
    vkey: tuple[int, Any, Any] = (oid, device, dtype)
    if vkey in visited:
        return visited[vkey]

    if isinstance(value, TensorLike):
        allowed = getattr(value, "allowed_dtypes", None)
        target_dev = value.device if device is None else device
        target_dt = (
            value.dtype
            if dtype is None or (allowed is not None and dtype not in allowed)
            else dtype
        )
        if _uses_default_tensorlike_conversion(value, use_to=via_to):
            needs = _component_like_needs_conversion(
                value,
                device=target_dev,
                dtype=target_dt,
                active=active,
                visited=visited,
                tensor_memo=tensor_memo,
                use_to=via_to,
                explicit_device=has_dev,
            )
            visited[vkey] = needs
            return needs
        _validate_component_object_graph(
            value, active=active, tensor_memo=tensor_memo
        )
        needs = bool(value.device != target_dev or value.dtype != target_dt)
        visited[vkey] = needs
        return needs

    if isinstance(value, Node):
        if not _uses_default_node_conversion(value, use_to=via_to):
            _validate_component_object_graph(
                value, active=active, tensor_memo=tensor_memo
            )
            active.add(oid)
            try:
                if not via_to:
                    needs = (
                        False
                        if dtype is None
                        else (value.type(dtype) is not value)
                    )
                else:
                    needs = (
                        value.to(
                            device=device if has_dev else None, dtype=dtype
                        )
                        is not value
                    )
            finally:
                active.discard(oid)
            visited[vkey] = needs
            return needs

        active.add(oid)
        try:
            _validate_node_contexts(
                value, active=active, tensor_memo=tensor_memo
            )
            needs = False
            for name in value._child_names:
                if _value_needs_conversion(
                    getattr(value, name),
                    device=device,
                    dtype=(None if name in value._keep_dtype_names else dtype),
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                ):
                    needs = True
            visited[vkey] = needs
            return needs
        finally:
            active.discard(oid)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        active.add(oid)
        try:
            _validate_supported_dataclass(
                value, active=active, tensor_memo=tensor_memo
            )
            params = getattr(value, "__dataclass_params__", None)
            is_frozen = bool(params is not None and params.frozen)
            needs = False
            for f in dataclasses.fields(value):
                if is_frozen and not f.init:
                    continue
                if _value_needs_conversion(
                    getattr(value, f.name),
                    device=device,
                    dtype=dtype,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                ):
                    needs = True
            visited[vkey] = needs
            return needs
        finally:
            active.discard(oid)

    if isinstance(value, dict):
        active.add(oid)
        try:
            needs = False
            for k, v in value.items():
                if _value_needs_conversion(
                    k,
                    device=device,
                    dtype=dtype,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                ):
                    needs = True
                if _value_needs_conversion(
                    v,
                    device=device,
                    dtype=dtype,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                ):
                    needs = True
            visited[vkey] = needs
            return needs
        finally:
            active.discard(oid)

    if isinstance(value, (tuple, list)):
        active.add(oid)
        try:
            needs = False
            for v in value:
                if _value_needs_conversion(
                    v,
                    device=device,
                    dtype=dtype,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                ):
                    needs = True
            visited[vkey] = needs
            return needs
        finally:
            active.discard(oid)

    _validate_opaque_object(value, active=active, tensor_memo=tensor_memo)
    visited[vkey] = False
    return False


def _component_like_needs_conversion(
    obj: TensorLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
    active: set[int] | None = None,
    visited: dict[tuple[int, Any, Any], bool] | None = None,
    tensor_memo: dict[int, bool] | None = None,
    use_to: bool | None = None,
    explicit_device: bool | None = None,
) -> bool:
    # Always traverse attributes so cycles, unsupported frozen dataclass fields,
    # and opaque tensor holders fail even when obj.device/dtype match target.
    if active is None:
        active = set()
    if visited is None:
        visited = {}
    if tensor_memo is None:
        tensor_memo = {}
    via_to = True if use_to is None else use_to
    has_dev = True if explicit_device is None else explicit_device
    oid = id(obj)
    if oid in active:
        _raise_cycle_error(obj)
    vkey: tuple[int, Any, Any] = (oid, device, dtype)
    if vkey in visited:
        return visited[vkey]
    active.add(oid)
    try:
        needs = obj.device != device or obj.dtype != dtype
        slots = _iter_mro_slots(obj)
        for slot in slots:
            if hasattr(obj, slot) and _value_needs_conversion(
                getattr(obj, slot),
                device=device,
                dtype=dtype,
                active=active,
                visited=visited,
                tensor_memo=tensor_memo,
                use_to=via_to,
                explicit_device=has_dev,
            ):
                needs = True
        extra_attrs = getattr(obj, "__dict__", None)
        if extra_attrs is not None:
            for key, value in extra_attrs.items():
                if key in slots:
                    continue
                if _value_needs_conversion(
                    value,
                    device=device,
                    dtype=dtype,
                    active=active,
                    visited=visited,
                    tensor_memo=tensor_memo,
                    use_to=via_to,
                    explicit_device=has_dev,
                ):
                    needs = True
        visited[vkey] = needs
        return needs
    finally:
        active.discard(oid)


_TL = TypeVar("_TL", bound=TensorLike)


def _clone_component_like(
    obj: _TL,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    memo: ConversionMemo | None = None,
    active: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
    use_to: bool | None = None,
    explicit_device: bool | None = None,
) -> _TL:
    """Clone a ``TensorLike`` across all MRO slots and ``__dict__``."""
    if memo is None:
        memo = {}
    if active is None:
        active = set()
    if tensor_memo is None:
        tensor_memo = {}
    via_to = (device is not None) if use_to is None else use_to
    has_dev = (
        (device is not None) if explicit_device is None else explicit_device
    )
    oid = id(obj)
    if oid in active:
        _raise_cycle_error(obj)
    target_device = obj.device if device is None else device
    target_dtype = obj.dtype if dtype is None else dtype

    new_obj = obj.__class__.__new__(obj.__class__)
    active.add(oid)
    try:
        slots = _iter_mro_slots(obj)
        for slot in slots:
            if not hasattr(obj, slot):
                continue
            attr = getattr(obj, slot)
            converted = _convert_component_value(
                attr,
                device=target_device,
                dtype=target_dtype,
                memo=memo,
                active=active,
                tensor_memo=tensor_memo,
                use_to=via_to,
                explicit_device=has_dev,
            )
            setattr(new_obj, slot, converted)

        new_obj.override_device(target_device)
        new_obj.override_dtype(target_dtype)

        extra_attrs = getattr(obj, "__dict__", None)
        if extra_attrs is not None:
            slots_set = set(slots)
            for key, value in extra_attrs.items():
                if key in slots_set:
                    continue
                setattr(
                    new_obj,
                    key,
                    _convert_component_value(
                        value,
                        device=target_device,
                        dtype=target_dtype,
                        memo=memo,
                        active=active,
                        tensor_memo=tensor_memo,
                        use_to=via_to,
                        explicit_device=has_dev,
                    ),
                )
    finally:
        active.discard(oid)

    if isinstance(obj, Component) and isinstance(new_obj, Component):
        new_obj.label = getattr(obj, "label", obj.__class__.__name__)

    memo[(oid, target_device, target_dtype)] = (obj, new_obj)
    return new_obj


def _convert_component_like(
    obj: _TL,
    *,
    device: torch.device,
    dtype: torch.dtype,
    memo: ConversionMemo | None = None,
    active: set[int] | None = None,
    tensor_memo: dict[int, bool] | None = None,
    use_to: bool | None = None,
    explicit_device: bool | None = None,
) -> _TL:
    """Convert ``obj`` to ``(device, dtype)`` when needed, preserving aliases via ``memo``."""
    if memo is None:
        memo = {}
    if active is None:
        active = set()
    if tensor_memo is None:
        tensor_memo = {}
    via_to = True if use_to is None else use_to
    has_dev = True if explicit_device is None else explicit_device
    oid = id(obj)
    if oid in active:
        _raise_cycle_error(obj)
    mkey = (oid, device, dtype)
    cached = memo.get(mkey)
    if cached is not None:
        return cached[1]
    if not _component_like_needs_conversion(
        obj,
        device=device,
        dtype=dtype,
        active=active,
        tensor_memo=tensor_memo,
        use_to=via_to,
        explicit_device=has_dev,
    ):
        memo[mkey] = (obj, obj)
        return obj
    return _clone_component_like(
        obj,
        device=device,
        dtype=dtype,
        memo=memo,
        active=active,
        tensor_memo=tensor_memo,
        use_to=via_to,
        explicit_device=has_dev,
    )


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
            _validate_component_object_graph(self)
            return self
        has_dev = device is not None
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
            self,
            device=target_device,
            dtype=target_dtype,
            use_to=True,
            explicit_device=has_dev,
        ):
            return self
        if type(self)._clone_tensorlike is ComponentCache._clone_tensorlike:
            return _clone_component_like(
                self,
                device=target_device if has_dev else None,
                dtype=target_dtype,
                use_to=True,
                explicit_device=has_dev,
            )
        return self._clone_tensorlike(
            device=target_device if has_dev else None, dtype=target_dtype
        )

    @override
    def type(self, dtype: torch.dtype) -> Self:
        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )
        if not _component_like_needs_conversion(
            self,
            device=self.device,
            dtype=dtype,
            use_to=False,
            explicit_device=False,
        ):
            return self
        if type(self)._clone_tensorlike is ComponentCache._clone_tensorlike:
            return _clone_component_like(
                self,
                device=None,
                dtype=dtype,
                use_to=False,
                explicit_device=False,
            )
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

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.label = self.__class__.__name__

    ############################################################################

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
            _validate_component_object_graph(self)
            return self
        has_dev = device is not None
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
            self,
            device=target_device,
            dtype=target_dtype,
            use_to=True,
            explicit_device=has_dev,
        ):
            return self
        if type(self)._clone_tensorlike is Component._clone_tensorlike:
            new = _clone_component_like(
                self,
                device=target_device if has_dev else None,
                dtype=target_dtype,
                use_to=True,
                explicit_device=has_dev,
            )
        else:
            new = self._clone_tensorlike(
                device=target_device if has_dev else None, dtype=target_dtype
            )
        new.label = getattr(self, "label", self.__class__.__name__)
        return new

    @override
    def type(self, dtype: torch.dtype) -> Self:
        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )
        if not _component_like_needs_conversion(
            self,
            device=self.device,
            dtype=dtype,
            use_to=False,
            explicit_device=False,
        ):
            return self
        if type(self)._clone_tensorlike is Component._clone_tensorlike:
            new = _clone_component_like(
                self,
                device=None,
                dtype=dtype,
                use_to=False,
                explicit_device=False,
            )
        else:
            new = self._clone_tensorlike(device=None, dtype=dtype)
        new.label = getattr(self, "label", self.__class__.__name__)
        return new

    ############################################################################

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"

    def __repr__(self) -> str:
        return str(self)
