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
Components: Custom Tree Helpers
===============================

Pure tensor-tree extraction, reconstruction, and mutation tracking for custom
:class:`~dxtb._src.components.classicals.base.Classical` and
:class:`~dxtb._src.components.interactions.base.Interaction` components.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import types
from enum import Enum
from typing import Any

import torch
from tad_mctc.tree import Node, child, context

from dxtb._src.scf.core import SCFInteractionEntry
from dxtb._src.typing import DD, Tensor, TensorLike
from dxtb._src.utils.tensors import normalize_device

from .base import (
    Component,
    ComponentCache,
    _iter_mro_slots,
    component_mutation_epoch,
)

__all__ = [
    "ComponentMutationSnapshot",
    "CustomClassicalCoreSetup",
    "CustomClassicalTemplate",
    "CustomInteractionTemplate",
    "build_custom_classical_core_setup",
    "can_build_custom_classical_core_setup",
    "is_tensor_only_object",
    "prepare_custom_interaction_entry",
    "reconstruct_custom_interaction",
    "snapshot_component_state",
]

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


def _resolve_dd_device(device: torch.device | str | None) -> torch.device:
    """Normalize optional ``DD['device']`` values to a ``torch.device``."""
    resolved = normalize_device(device)
    if resolved is None:
        return torch.empty(0).device
    return resolved


def _static_key(val: Any) -> Any:
    """Build a Python-3.10-hashable, NaN-safe, type-tagged structural key."""
    if isinstance(val, slice):
        return (
            slice,
            _static_key(val.start),
            _static_key(val.stop),
            _static_key(val.step),
        )
    if isinstance(val, float):
        return (float, "nan") if math.isnan(val) else (float, val)
    if isinstance(val, complex):
        real_k: Any = "nan" if math.isnan(val.real) else val.real
        imag_k: Any = "nan" if math.isnan(val.imag) else val.imag
        return (complex, real_k, imag_k)
    if isinstance(val, torch.device):
        return (torch.device, normalize_device(val))
    if isinstance(val, (*_PRIMITIVE_STATIC_TYPES, Enum)):
        return (type(val), val)
    return ("id", type(val), id(val))


def _safe_tensor_version(tensor: Tensor) -> int | None:
    """Return ``tensor._version`` when available outside functorch wrappers."""
    try:
        if torch._C._functorch.is_functorch_wrapped_tensor(tensor):
            return None
        return int(tensor._version)
    except (AttributeError, RuntimeError):
        return None


def _collect_component_state(
    obj: Any,
    path: str = "",
    seen: set[int] | None = None,
    tensor_records: list[tuple[str, int, int | None]] | None = None,
    static_records: list[tuple[str, Any]] | None = None,
) -> tuple[
    list[tuple[str, int, int | None]],
    list[tuple[str, Any]],
]:
    """Record reachable tensor identities/versions and non-tensor attribute state."""
    if seen is None:
        seen = set()
    if tensor_records is None:
        tensor_records = []
    if static_records is None:
        static_records = []

    if isinstance(obj, Tensor):
        tid = id(obj)
        tensor_records.append((path, tid, _safe_tensor_version(obj)))
        return tensor_records, static_records

    if isinstance(obj, (*_NON_TRAVERSABLE_TYPES, Enum)):
        static_records.append((path, _static_key(obj)))
        return tensor_records, static_records

    oid = id(obj)
    if oid in seen:
        static_records.append((path, ("seen", type(obj))))
        return tensor_records, static_records
    seen.add(oid)

    if isinstance(obj, dict):
        static_records.append(
            (
                path,
                ("dict", type(obj), tuple(_static_key(k) for k in obj.keys())),
            )
        )
        for key, val in obj.items():
            _collect_component_state(
                val, f"{path}[{key!r}]", seen, tensor_records, static_records
            )
        return tensor_records, static_records

    if isinstance(obj, (tuple, list)):
        static_records.append((path, ("seq", type(obj), len(obj))))
        for idx, item in enumerate(obj):
            _collect_component_state(
                item, f"{path}[{idx}]", seen, tensor_records, static_records
            )
        return tensor_records, static_records

    if isinstance(obj, Node) or (
        dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    ):
        static_records.append((path, ("dc", type(obj))))
        for field_info in dataclasses.fields(obj):
            val = getattr(obj, field_info.name)
            sub = f"{path}.{field_info.name}" if path else field_info.name
            _collect_component_state(
                val, sub, seen, tensor_records, static_records
            )
        return tensor_records, static_records

    ext_id = None if isinstance(obj, (Component, ComponentCache)) else oid
    static_records.append((path, ("obj", type(obj), ext_id)))
    slots = _iter_mro_slots(obj)
    for slot in slots:
        if not hasattr(obj, slot):
            static_records.append((f"{path}.{slot}" if path else slot, "unset"))
            continue
        sub = f"{path}.{slot}" if path else slot
        _collect_component_state(
            getattr(obj, slot), sub, seen, tensor_records, static_records
        )

    extra_attrs = getattr(obj, "__dict__", None)
    if isinstance(extra_attrs, dict):
        static_records.append(
            (
                path,
                ("dict_keys", tuple(k for k in extra_attrs if k not in slots)),
            )
        )
        for key, val in extra_attrs.items():
            if key in slots:
                continue
            sub = f"{path}.{key}" if path else key
            _collect_component_state(
                val, sub, seen, tensor_records, static_records
            )

    return tensor_records, static_records


class ComponentMutationSnapshot:
    """Snapshot of a component's mutation epoch, tensors, and static attributes."""

    __slots__ = ("component_id", "epoch", "tensor_records", "static_records")

    component_id: int
    epoch: int
    tensor_records: tuple[tuple[str, int, int | None], ...]
    static_records: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        component_id: int,
        epoch: int,
        tensor_records: tuple[tuple[str, int, int | None], ...],
        static_records: tuple[tuple[str, Any], ...],
    ) -> None:
        self.component_id = component_id
        self.epoch = epoch
        self.tensor_records = tensor_records
        self.static_records = static_records

    def matches(self, component: Component) -> bool:
        """Return ``True`` iff ``component`` has not been mutated since capture."""
        if id(component) != self.component_id:
            return False
        if component_mutation_epoch(component) != self.epoch:
            return False
        cur_tensors, cur_statics = _collect_component_state(component)
        if len(cur_tensors) != len(self.tensor_records):
            return False
        if tuple(cur_statics) != self.static_records:
            return False
        for (path_a, id_a, ver_a), (path_b, id_b, ver_b) in zip(
            self.tensor_records, cur_tensors
        ):
            if path_a != path_b or id_a != id_b:
                return False
            if ver_a is not None and ver_b is not None and ver_a != ver_b:
                return False
        return True

    def sync_post_eval_static_state(self, component: Component) -> None:
        """Advance ``static_records`` when ``get_energy`` mutated only Python counters."""
        if id(component) != self.component_id:
            return
        if component_mutation_epoch(component) != self.epoch:
            return
        cur_tensors, cur_statics = _collect_component_state(component)
        if len(cur_tensors) != len(self.tensor_records):
            return
        for (path_a, id_a, ver_a), (path_b, id_b, ver_b) in zip(
            self.tensor_records, cur_tensors
        ):
            if path_a != path_b or id_a != id_b:
                return
            if ver_a is not None and ver_b is not None and ver_a != ver_b:
                return
        self.static_records = tuple(cur_statics)


def snapshot_component_state(component: Component) -> ComponentMutationSnapshot:
    """Capture a mutation-detection token for ``component``."""
    tensor_records, static_records = _collect_component_state(component)
    return ComponentMutationSnapshot(
        component_id=id(component),
        epoch=component_mutation_epoch(component),
        tensor_records=tuple(tensor_records),
        static_records=tuple(static_records),
    )


def _contains_tensor(obj: Any, seen: set[int] | None = None) -> bool:
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
        return any(_contains_tensor(v, seen) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return any(_contains_tensor(v, seen) for v in obj)
    if isinstance(obj, Node) or (
        dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    ):
        return any(
            _contains_tensor(getattr(obj, f.name), seen)
            for f in dataclasses.fields(obj)
        )
    if isinstance(obj, (Component, ComponentCache)):
        for slot in _iter_mro_slots(obj):
            if hasattr(obj, slot) and _contains_tensor(
                getattr(obj, slot), seen
            ):
                return True
        extra = getattr(obj, "__dict__", None)
        if isinstance(extra, dict):
            return any(_contains_tensor(v, seen) for v in extra.values())
    return False


def is_tensor_only_object(
    obj: Any,
    *,
    seen: set[int] | None = None,
) -> bool:
    """Return ``True`` iff ``obj`` recursively holds only tensors and static primitives.

    External stateful model objects (such as ``D4Model`` or ``Reference``) and
    Python callables return ``False`` so ``can_vmap_system`` routes them to
    independent lane evaluation instead of ``torch.func.vmap``.
    """
    if isinstance(obj, (Tensor, *_PRIMITIVE_STATIC_TYPES, Enum)):
        return True

    if callable(obj) or isinstance(obj, _NON_TRAVERSABLE_TYPES):
        return False

    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return True
    seen.add(oid)

    if isinstance(obj, (tuple, list)):
        return all(is_tensor_only_object(item, seen=seen) for item in obj)

    if isinstance(obj, dict):
        return all(
            is_tensor_only_object(k, seen=seen)
            and is_tensor_only_object(v, seen=seen)
            for k, v in obj.items()
        )

    if isinstance(obj, Node):
        return all(
            is_tensor_only_object(getattr(obj, f.name), seen=seen)
            for f in dataclasses.fields(obj)
        )

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        params = getattr(obj, "__dataclass_params__", None)
        is_frozen = bool(params is not None and params.frozen)
        for f in dataclasses.fields(obj):
            f_val = getattr(obj, f.name)
            if is_frozen and not f.init and _contains_tensor(f_val):
                return False
            if not is_tensor_only_object(f_val, seen=seen):
                return False
        return True

    if isinstance(obj, (Component, ComponentCache)):
        for slot in _iter_mro_slots(obj):
            if hasattr(obj, slot) and not is_tensor_only_object(
                getattr(obj, slot), seen=seen
            ):
                return False
        extra_attrs = getattr(obj, "__dict__", None)
        if isinstance(extra_attrs, dict):
            for val in extra_attrs.values():
                if not is_tensor_only_object(val, seen=seen):
                    return False
        return True

    return False


def _spec_struct_key(spec: Any) -> Any:
    """Return a hashable structural key for any spec node or static value."""
    if isinstance(
        spec,
        (
            _LeafRef,
            _TupleSpec,
            _ListSpec,
            _DictSpec,
            _DataclassSpec,
            _ObjectSpec,
        ),
    ):
        return spec._struct_key()
    return _static_key(spec)


class _LeafRef:
    """Placeholder pointing into extracted floating or kept-dtype tensor tuples."""

    __slots__ = ("bucket", "index")

    bucket: int  # 0 -> primary (floating), 1 -> secondary (kept_dtype)
    index: int

    def __init__(self, bucket: int, index: int) -> None:
        self.bucket = bucket
        self.index = index

    def _struct_key(self) -> tuple[Any, ...]:
        return (_LeafRef, self.bucket, self.index)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _LeafRef)
            and self.bucket == other.bucket
            and self.index == other.index
        )

    def __hash__(self) -> int:
        return hash((_LeafRef, self.bucket, self.index))


class _TupleSpec:
    __slots__ = (
        "orig",
        "tuple_cls",
        "is_namedtuple",
        "items",
        "has_dynamic",
        "_key_cache",
        "_hash_cache",
    )

    def __init__(
        self,
        orig: tuple[Any, ...],
        items: tuple[Any, ...],
        has_dynamic: bool,
    ) -> None:
        self.orig = orig
        self.tuple_cls = type(orig)
        self.is_namedtuple = self.tuple_cls is not tuple and hasattr(
            self.tuple_cls, "_fields"
        )
        self.items = items
        self.has_dynamic = has_dynamic
        self._key_cache: Any = None
        self._hash_cache: int | None = None

    def _struct_key(self) -> Any:
        if self._key_cache is None:
            self._key_cache = (
                _TupleSpec,
                self.tuple_cls,
                tuple(_spec_struct_key(item) for item in self.items),
            )
        return self._key_cache

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TupleSpec) and (
            self is other or self._struct_key() == other._struct_key()
        )

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(self._struct_key())
        return self._hash_cache


class _ListSpec:
    __slots__ = (
        "orig",
        "list_cls",
        "items",
        "has_dynamic",
        "_key_cache",
        "_hash_cache",
    )

    def __init__(
        self,
        orig: list[Any],
        items: tuple[Any, ...],
        has_dynamic: bool,
    ) -> None:
        self.orig = orig
        self.list_cls = type(orig)
        self.items = items
        self.has_dynamic = has_dynamic
        self._key_cache: Any = None
        self._hash_cache: int | None = None

    def _struct_key(self) -> Any:
        if self._key_cache is None:
            self._key_cache = (
                _ListSpec,
                self.list_cls,
                tuple(_spec_struct_key(item) for item in self.items),
            )
        return self._key_cache

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ListSpec) and (
            self is other or self._struct_key() == other._struct_key()
        )

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(self._struct_key())
        return self._hash_cache


class _DictSpec:
    __slots__ = (
        "orig",
        "dict_cls",
        "entries",
        "has_dynamic",
        "_key_cache",
        "_hash_cache",
    )

    def __init__(
        self,
        orig: dict[Any, Any],
        entries: tuple[tuple[Any, Any], ...],
        has_dynamic: bool,
    ) -> None:
        self.orig = orig
        self.dict_cls = type(orig)
        self.entries = entries
        self.has_dynamic = has_dynamic
        self._key_cache: Any = None
        self._hash_cache: int | None = None

    def _struct_key(self) -> Any:
        if self._key_cache is None:
            self._key_cache = (
                _DictSpec,
                self.dict_cls,
                tuple(
                    (_spec_struct_key(k), _spec_struct_key(v))
                    for k, v in self.entries
                ),
            )
        return self._key_cache

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DictSpec) and (
            self is other or self._struct_key() == other._struct_key()
        )

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(self._struct_key())
        return self._hash_cache


class _DataclassSpec:
    __slots__ = (
        "orig",
        "is_node",
        "is_frozen",
        "fields",
        "non_init_keys",
        "has_dynamic",
        "_key_cache",
        "_hash_cache",
    )

    def __init__(
        self,
        orig: Any,
        is_node: bool,
        is_frozen: bool,
        fields: tuple[tuple[str, Any], ...],
        non_init_keys: tuple[tuple[str, Any], ...],
        has_dynamic: bool,
    ) -> None:
        self.orig = orig
        self.is_node = is_node
        self.is_frozen = is_frozen
        self.fields = fields
        self.non_init_keys = non_init_keys
        self.has_dynamic = has_dynamic
        self._key_cache: Any = None
        self._hash_cache: int | None = None

    def _struct_key(self) -> Any:
        if self._key_cache is None:
            self._key_cache = (
                _DataclassSpec,
                type(self.orig),
                tuple((name, _spec_struct_key(s)) for name, s in self.fields),
                self.non_init_keys,
            )
        return self._key_cache

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DataclassSpec) and (
            self is other or self._struct_key() == other._struct_key()
        )

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(self._struct_key())
        return self._hash_cache


class _ObjectSpec:
    __slots__ = (
        "orig",
        "slot_entries",
        "dict_entries",
        "has_dynamic",
        "_key_cache",
        "_hash_cache",
    )

    def __init__(
        self,
        orig: Any,
        slot_entries: tuple[tuple[str, Any], ...],
        dict_entries: tuple[tuple[str, Any], ...],
        has_dynamic: bool,
    ) -> None:
        self.orig = orig
        self.slot_entries = slot_entries
        self.dict_entries = dict_entries
        self.has_dynamic = has_dynamic
        self._key_cache: Any = None
        self._hash_cache: int | None = None

    def _struct_key(self) -> Any:
        if self._key_cache is None:
            self._key_cache = (
                _ObjectSpec,
                type(self.orig),
                tuple((s, _spec_struct_key(v)) for s, v in self.slot_entries),
                tuple((k, _spec_struct_key(v)) for k, v in self.dict_entries),
            )
        return self._key_cache

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ObjectSpec) and (
            self is other or self._struct_key() == other._struct_key()
        )

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(self._struct_key())
        return self._hash_cache


def _spec_has_dynamic(spec: Any) -> bool:
    if isinstance(spec, _LeafRef):
        return True
    if isinstance(
        spec,
        (_TupleSpec, _ListSpec, _DictSpec, _DataclassSpec, _ObjectSpec),
    ):
        return spec.has_dynamic
    return False


def _extract_spec(
    obj: Any,
    *,
    floating_list: list[Tensor],
    kept_list: list[Tensor],
    tensor_refs_by_id: dict[int, _LeafRef],
    target_dtype: torch.dtype | None,
    active_obj_ids: set[int],
    memo_specs: dict[int, Any],
    snapshot_mutable: bool = False,
) -> Any:
    """Build a tensor-free structural spec and populate ``floating_list``/``kept_list``."""
    if isinstance(obj, Tensor):
        tid = id(obj)
        existing = tensor_refs_by_id.get(tid)
        if existing is not None:
            return existing
        use_primary = obj.is_floating_point() and (
            target_dtype is None or obj.dtype == target_dtype
        )
        if use_primary:
            ref = _LeafRef(0, len(floating_list))
            floating_list.append(obj)
        else:
            ref = _LeafRef(1, len(kept_list))
            kept_list.append(obj)
        tensor_refs_by_id[tid] = ref
        return ref

    if isinstance(obj, (*_NON_TRAVERSABLE_TYPES, Enum)):
        return obj

    oid = id(obj)
    if oid in memo_specs:
        return memo_specs[oid]
    if oid in active_obj_ids:
        return obj

    active_obj_ids.add(oid)
    try:
        if isinstance(obj, tuple):
            sub_items = tuple(
                _extract_spec(
                    item,
                    floating_list=floating_list,
                    kept_list=kept_list,
                    tensor_refs_by_id=tensor_refs_by_id,
                    target_dtype=target_dtype,
                    active_obj_ids=active_obj_ids,
                    memo_specs=memo_specs,
                    snapshot_mutable=snapshot_mutable,
                )
                for item in obj
            )
            has_dyn = any(_spec_has_dynamic(s) for s in sub_items)
            result: Any = _TupleSpec(obj, sub_items, has_dyn)
            memo_specs[oid] = result
            return result

        if isinstance(obj, list):
            sub_items = tuple(
                _extract_spec(
                    item,
                    floating_list=floating_list,
                    kept_list=kept_list,
                    tensor_refs_by_id=tensor_refs_by_id,
                    target_dtype=target_dtype,
                    active_obj_ids=active_obj_ids,
                    memo_specs=memo_specs,
                    snapshot_mutable=snapshot_mutable,
                )
                for item in obj
            )
            has_dyn = any(_spec_has_dynamic(s) for s in sub_items)
            orig_list = copy.copy(obj) if snapshot_mutable else obj
            result = _ListSpec(orig_list, sub_items, has_dyn)
            memo_specs[oid] = result
            return result

        if isinstance(obj, dict):
            entries = tuple(
                (
                    _extract_spec(
                        key,
                        floating_list=floating_list,
                        kept_list=kept_list,
                        tensor_refs_by_id=tensor_refs_by_id,
                        target_dtype=target_dtype,
                        active_obj_ids=active_obj_ids,
                        memo_specs=memo_specs,
                        snapshot_mutable=snapshot_mutable,
                    ),
                    _extract_spec(
                        val,
                        floating_list=floating_list,
                        kept_list=kept_list,
                        tensor_refs_by_id=tensor_refs_by_id,
                        target_dtype=target_dtype,
                        active_obj_ids=active_obj_ids,
                        memo_specs=memo_specs,
                        snapshot_mutable=snapshot_mutable,
                    ),
                )
                for key, val in obj.items()
            )
            has_dyn = any(
                _spec_has_dynamic(k_s) or _spec_has_dynamic(v_s)
                for k_s, v_s in entries
            )
            orig_dict = copy.copy(obj) if snapshot_mutable else obj
            result = _DictSpec(orig_dict, entries, has_dyn)
            memo_specs[oid] = result
            return result

        if isinstance(obj, Node) or (
            dataclasses.is_dataclass(obj) and not isinstance(obj, type)
        ):
            is_node = isinstance(obj, Node)
            params = getattr(obj, "__dataclass_params__", None)
            is_frozen = is_node or bool(params is not None and params.frozen)
            field_specs: list[tuple[str, Any]] = []
            non_init_keys: list[tuple[str, Any]] = []
            for f in dataclasses.fields(obj):
                f_val = getattr(obj, f.name)
                if not is_node and is_frozen and not f.init:
                    non_init_keys.append((f.name, _static_key(f_val)))
                    continue
                field_specs.append(
                    (
                        f.name,
                        _extract_spec(
                            f_val,
                            floating_list=floating_list,
                            kept_list=kept_list,
                            tensor_refs_by_id=tensor_refs_by_id,
                            target_dtype=target_dtype,
                            active_obj_ids=active_obj_ids,
                            memo_specs=memo_specs,
                            snapshot_mutable=snapshot_mutable,
                        ),
                    )
                )
            has_dyn = any(_spec_has_dynamic(s) for _, s in field_specs)
            orig_dc = (
                copy.copy(obj) if snapshot_mutable and not is_frozen else obj
            )
            result = _DataclassSpec(
                orig_dc,
                is_node,
                is_frozen,
                tuple(field_specs),
                tuple(non_init_keys),
                has_dyn,
            )
            memo_specs[oid] = result
            return result

        if not snapshot_mutable and not isinstance(
            obj, (Component, ComponentCache)
        ):
            # Preserve opaque external models in custom Interaction entries when
            # they hold no tensors; wrap them only if a tensor is nested inside.
            slots = _iter_mro_slots(obj)
            slot_entries_ext: list[tuple[str, Any]] = []
            for slot in slots:
                if not hasattr(obj, slot):
                    continue
                slot_entries_ext.append(
                    (
                        slot,
                        _extract_spec(
                            getattr(obj, slot),
                            floating_list=floating_list,
                            kept_list=kept_list,
                            tensor_refs_by_id=tensor_refs_by_id,
                            target_dtype=target_dtype,
                            active_obj_ids=active_obj_ids,
                            memo_specs=memo_specs,
                            snapshot_mutable=False,
                        ),
                    )
                )
            dict_entries_ext: list[tuple[str, Any]] = []
            extra_ext = getattr(obj, "__dict__", None)
            if isinstance(extra_ext, dict):
                for key, val in extra_ext.items():
                    if key in slots:
                        continue
                    dict_entries_ext.append(
                        (
                            key,
                            _extract_spec(
                                val,
                                floating_list=floating_list,
                                kept_list=kept_list,
                                tensor_refs_by_id=tensor_refs_by_id,
                                target_dtype=target_dtype,
                                active_obj_ids=active_obj_ids,
                                memo_specs=memo_specs,
                                snapshot_mutable=False,
                            ),
                        )
                    )
            has_dyn_ext = any(
                _spec_has_dynamic(s) for _, s in slot_entries_ext
            ) or any(_spec_has_dynamic(s) for _, s in dict_entries_ext)
            if has_dyn_ext:
                result = _ObjectSpec(
                    obj,
                    tuple(slot_entries_ext),
                    tuple(dict_entries_ext),
                    True,
                )
                memo_specs[oid] = result
                return result
            memo_specs[oid] = obj
            return obj

        slots = _iter_mro_slots(obj)
        slot_entries: list[tuple[str, Any]] = []
        for slot in slots:
            if not hasattr(obj, slot):
                continue
            slot_entries.append(
                (
                    slot,
                    _extract_spec(
                        getattr(obj, slot),
                        floating_list=floating_list,
                        kept_list=kept_list,
                        tensor_refs_by_id=tensor_refs_by_id,
                        target_dtype=target_dtype,
                        active_obj_ids=active_obj_ids,
                        memo_specs=memo_specs,
                        snapshot_mutable=snapshot_mutable,
                    ),
                )
            )

        dict_entries: list[tuple[str, Any]] = []
        extra_attrs = getattr(obj, "__dict__", None)
        if isinstance(extra_attrs, dict):
            for key, val in extra_attrs.items():
                if key in slots:
                    continue
                dict_entries.append(
                    (
                        key,
                        _extract_spec(
                            val,
                            floating_list=floating_list,
                            kept_list=kept_list,
                            tensor_refs_by_id=tensor_refs_by_id,
                            target_dtype=target_dtype,
                            active_obj_ids=active_obj_ids,
                            memo_specs=memo_specs,
                            snapshot_mutable=snapshot_mutable,
                        ),
                    )
                )

        has_dyn = any(_spec_has_dynamic(s) for _, s in slot_entries) or any(
            _spec_has_dynamic(s) for _, s in dict_entries
        )
        orig_obj = copy.copy(obj) if snapshot_mutable else obj
        result = _ObjectSpec(
            orig_obj, tuple(slot_entries), tuple(dict_entries), has_dyn
        )
        memo_specs[oid] = result
        return result
    finally:
        active_obj_ids.discard(oid)


def _rebuild_spec(
    spec: Any,
    primary: tuple[Tensor, ...],
    secondary: tuple[Tensor, ...],
    target_dd: DD | None,
    rebuilt_memo: dict[int, Any],
) -> Any:
    """Reconstruct shallow object views when any leaf tensor or dd changed."""
    if isinstance(spec, _LeafRef):
        return (
            primary[spec.index] if spec.bucket == 0 else secondary[spec.index]
        )

    if not isinstance(
        spec,
        (_TupleSpec, _ListSpec, _DictSpec, _DataclassSpec, _ObjectSpec),
    ):
        return spec

    sid = id(spec)
    if sid in rebuilt_memo:
        return rebuilt_memo[sid]

    if isinstance(spec, _TupleSpec):
        if not spec.has_dynamic:
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        items = tuple(
            _rebuild_spec(item, primary, secondary, target_dd, rebuilt_memo)
            for item in spec.items
        )
        if all(a is b for a, b in zip(items, spec.orig)):
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        out: Any = (
            spec.tuple_cls(*items)
            if spec.is_namedtuple
            else (items if spec.tuple_cls is tuple else spec.tuple_cls(items))
        )
        rebuilt_memo[sid] = out
        return out

    if isinstance(spec, _ListSpec):
        if not spec.has_dynamic:
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        items_list = [
            _rebuild_spec(item, primary, secondary, target_dd, rebuilt_memo)
            for item in spec.items
        ]
        if all(a is b for a, b in zip(items_list, spec.orig)):
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        if spec.list_cls is list:
            out_list: Any = items_list
        else:
            out_list = copy.copy(spec.orig)
            out_list[:] = items_list
        rebuilt_memo[sid] = out_list
        return out_list

    if isinstance(spec, _DictSpec):
        if not spec.has_dynamic:
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        built_pairs = [
            (
                _rebuild_spec(
                    key_spec, primary, secondary, target_dd, rebuilt_memo
                ),
                _rebuild_spec(
                    val_spec, primary, secondary, target_dd, rebuilt_memo
                ),
            )
            for key_spec, val_spec in spec.entries
        ]
        if all(
            k is orig_k and val is spec.orig[orig_k]
            for (k, val), orig_k in zip(built_pairs, spec.orig.keys())
        ):
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        if spec.dict_cls is dict:
            out = dict(built_pairs)
        else:
            out = copy.copy(spec.orig)
            out.clear()
            out.update(built_pairs)
        rebuilt_memo[sid] = out
        return out

    if isinstance(spec, _DataclassSpec):
        if not spec.has_dynamic:
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        changes: dict[str, Any] = {}
        for name, field_spec in spec.fields:
            new_val = _rebuild_spec(
                field_spec, primary, secondary, target_dd, rebuilt_memo
            )
            if new_val is not getattr(spec.orig, name):
                changes[name] = new_val
        if not changes:
            rebuilt_memo[sid] = spec.orig
            return spec.orig
        if spec.is_node:
            out = spec.orig.replace(**changes)
        elif spec.is_frozen:
            out = dataclasses.replace(spec.orig, **changes)
        else:
            out = copy.copy(spec.orig)
            for name, new_val in changes.items():
                setattr(out, name, new_val)
        rebuilt_memo[sid] = out
        return out

    assert isinstance(spec, _ObjectSpec)
    resolved_target_device = (
        _resolve_dd_device(target_dd["device"])
        if target_dd is not None
        else None
    )
    dd_changed = (
        target_dd is not None
        and resolved_target_device is not None
        and isinstance(spec.orig, TensorLike)
        and (
            spec.orig.device != resolved_target_device
            or (
                target_dd["dtype"] in spec.orig.allowed_dtypes
                and spec.orig.dtype != target_dd["dtype"]
            )
        )
    )
    if not spec.has_dynamic and not dd_changed:
        rebuilt_memo[sid] = spec.orig
        return spec.orig

    slot_updates: list[tuple[str, Any]] = []
    changed = False
    for slot, slot_spec in spec.slot_entries:
        new_val = _rebuild_spec(
            slot_spec, primary, secondary, target_dd, rebuilt_memo
        )
        if new_val is not getattr(spec.orig, slot):
            changed = True
        slot_updates.append((slot, new_val))

    dict_updates: list[tuple[str, Any]] = []
    orig_dict = getattr(spec.orig, "__dict__", None)
    for key, val_spec in spec.dict_entries:
        new_val = _rebuild_spec(
            val_spec, primary, secondary, target_dd, rebuilt_memo
        )
        if orig_dict is None or new_val is not orig_dict.get(key):
            changed = True
        dict_updates.append((key, new_val))

    if not changed and not dd_changed:
        rebuilt_memo[sid] = spec.orig
        return spec.orig

    view = copy.copy(spec.orig)
    rebuilt_memo[sid] = view
    for slot, new_val in slot_updates:
        setattr(view, slot, new_val)
    if dict_updates:
        view_dict = vars(view)
        for key, new_val in dict_updates:
            view_dict[key] = new_val
    if target_dd is not None and isinstance(view, TensorLike):
        if resolved_target_device is not None:
            view.override_device(resolved_target_device)
        if target_dd["dtype"] in view.allowed_dtypes:
            view.override_dtype(target_dd["dtype"])
    return view


def _bind_leaves_from_value(
    spec: Any,
    val: Any,
    primary_overrides: dict[int, Tensor],
    secondary_overrides: dict[int, Tensor],
) -> None:
    """Record leaf tensors from ``val`` matching ``spec``'s ``_LeafRef`` paths."""
    if isinstance(spec, _LeafRef):
        if isinstance(val, Tensor):
            if spec.bucket == 0:
                primary_overrides[spec.index] = val
            else:
                secondary_overrides[spec.index] = val
        return
    if isinstance(spec, _TupleSpec) and isinstance(val, tuple):
        for s_item, v_item in zip(spec.items, val):
            _bind_leaves_from_value(
                s_item, v_item, primary_overrides, secondary_overrides
            )
    elif isinstance(spec, _ListSpec) and isinstance(val, list):
        for s_item, v_item in zip(spec.items, val):
            _bind_leaves_from_value(
                s_item, v_item, primary_overrides, secondary_overrides
            )
    elif isinstance(spec, _DictSpec) and isinstance(val, dict):
        for (_, v_spec), v_item in zip(spec.entries, val.values()):
            _bind_leaves_from_value(
                v_spec, v_item, primary_overrides, secondary_overrides
            )
    elif isinstance(spec, _DataclassSpec) and (
        isinstance(val, Node)
        or (dataclasses.is_dataclass(val) and not isinstance(val, type))
    ):
        for name, f_spec in spec.fields:
            if hasattr(val, name):
                _bind_leaves_from_value(
                    f_spec,
                    getattr(val, name),
                    primary_overrides,
                    secondary_overrides,
                )
    elif isinstance(spec, _ObjectSpec):
        for slot, s_spec in spec.slot_entries:
            if hasattr(val, slot):
                _bind_leaves_from_value(
                    s_spec,
                    getattr(val, slot),
                    primary_overrides,
                    secondary_overrides,
                )
        val_dict = getattr(val, "__dict__", None)
        if isinstance(val_dict, dict):
            for key, d_spec in spec.dict_entries:
                if key in val_dict:
                    _bind_leaves_from_value(
                        d_spec,
                        val_dict[key],
                        primary_overrides,
                        secondary_overrides,
                    )


class CustomClassicalTemplate:
    """Tensor-free structural template for ``CustomClassicalCoreSetup``."""

    __slots__ = (
        "component",
        "cache",
        "pair_spec",
        "primary_ids",
        "secondary_ids",
        "initial_dd",
        "mutation_snapshot",
        "_hash_cache",
    )

    component: Any
    cache: Any
    pair_spec: Any
    primary_ids: tuple[int, ...]
    secondary_ids: tuple[int, ...]
    initial_dd: DD
    mutation_snapshot: ComponentMutationSnapshot
    _hash_cache: int | None

    def __init__(
        self,
        *,
        component: Any,
        cache: Any,
        pair_spec: Any,
        primary_ids: tuple[int, ...],
        secondary_ids: tuple[int, ...],
        initial_dd: DD,
        mutation_snapshot: ComponentMutationSnapshot,
    ) -> None:
        self.component = component
        self.cache = cache
        self.pair_spec = pair_spec
        self.primary_ids = primary_ids
        self.secondary_ids = secondary_ids
        self.initial_dd = initial_dd
        self.mutation_snapshot = mutation_snapshot
        self._hash_cache = None

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, CustomClassicalTemplate):
            return False
        return (
            self.initial_dd["dtype"] == other.initial_dd["dtype"]
            and _resolve_dd_device(self.initial_dd["device"])
            == _resolve_dd_device(other.initial_dd["device"])
            and self.pair_spec == other.pair_spec
        )

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(
                (
                    CustomClassicalTemplate,
                    _resolve_dd_device(self.initial_dd["device"]),
                    self.initial_dd["dtype"],
                    _spec_struct_key(self.pair_spec),
                )
            )
        return self._hash_cache


class CustomClassicalCoreSetup(Node):
    """Immutable core setup Node wrapping a tensor-only custom ``Classical``."""

    anchor: Tensor = child()
    floating_tensors: tuple[Tensor, ...] = child()
    kept_dtype_tensors: tuple[Tensor, ...] = child(keep_dtype=True)
    template: CustomClassicalTemplate = context()

    def matches_component_state(self, component: Component) -> bool:
        """Return ``True`` iff ``component`` matches its setup-time state."""
        return self.template.mutation_snapshot.matches(component)

    def reconstruct(self) -> tuple[Any, Any]:
        """Materialize ``(component_view, cache_view)`` for the current leaf tensors."""
        target_device = self.anchor.device
        target_dd: DD = {
            "device": target_device,
            "dtype": self.anchor.dtype,
        }
        same_primary = len(self.floating_tensors) == len(
            self.template.primary_ids
        ) and all(
            id(t) == orig_id
            for t, orig_id in zip(
                self.floating_tensors, self.template.primary_ids
            )
        )
        same_secondary = len(self.kept_dtype_tensors) == len(
            self.template.secondary_ids
        ) and all(
            id(t) == orig_id
            for t, orig_id in zip(
                self.kept_dtype_tensors, self.template.secondary_ids
            )
        )
        same_dd = (
            target_device
            == _resolve_dd_device(self.template.initial_dd["device"])
            and target_dd["dtype"] == self.template.initial_dd["dtype"]
        )
        in_functorch = torch._C._functorch.peek_interpreter_stack() is not None
        if same_primary and same_secondary and same_dd and not in_functorch:
            return self.template.component, self.template.cache

        rebuilt = _rebuild_spec(
            self.template.pair_spec,
            self.floating_tensors,
            self.kept_dtype_tensors,
            target_dd,
            {},
        )
        if isinstance(rebuilt, tuple) and len(rebuilt) == 2:
            comp_view, cache_view = rebuilt
            if comp_view is self.template.component and (
                in_functorch or not same_dd
            ):
                comp_view = copy.copy(comp_view)
                if not same_dd and isinstance(comp_view, TensorLike):
                    comp_view.override_device(target_device)
                    if target_dd["dtype"] in comp_view.allowed_dtypes:
                        comp_view.override_dtype(target_dd["dtype"])
            return comp_view, cache_view
        return self.template.component, self.template.cache

    def evaluate_energy(self, positions: Tensor, charge: Tensor) -> Tensor:
        """Evaluate ``get_energy`` and sync non-tensor call counters updated inside it."""
        comp_view, cache_view = self.reconstruct()
        was_live_clean = (
            comp_view is self.template.component
            and torch._C._functorch.peek_interpreter_stack() is None
            and self.template.mutation_snapshot.matches(comp_view)
        )
        energy: Tensor = comp_view.get_energy(
            positions, cache_view, charge=charge
        )
        if was_live_clean:
            self.template.mutation_snapshot.sync_post_eval_static_state(
                comp_view
            )
        return energy

    def with_converted_component(
        self,
        component: Component,
        mutation_snapshot: ComponentMutationSnapshot | None = None,
    ) -> CustomClassicalCoreSetup:
        """Rebind the mutation snapshot and shared leaves after ``component`` conversion."""
        primary_overrides: dict[int, Tensor] = {}
        secondary_overrides: dict[int, Tensor] = {}
        if (
            isinstance(self.template.pair_spec, _TupleSpec)
            and len(self.template.pair_spec.items) == 2
        ):
            _bind_leaves_from_value(
                self.template.pair_spec.items[0],
                component,
                primary_overrides,
                secondary_overrides,
            )
        merged_primary = tuple(
            primary_overrides.get(idx, t)
            for idx, t in enumerate(self.floating_tensors)
        )
        merged_secondary = tuple(
            secondary_overrides.get(idx, t)
            for idx, t in enumerate(self.kept_dtype_tensors)
        )
        target_dd: DD = {
            "device": self.anchor.device,
            "dtype": self.anchor.dtype,
        }
        rebuilt = _rebuild_spec(
            self.template.pair_spec,
            merged_primary,
            merged_secondary,
            target_dd,
            {},
        )
        cache_view = (
            rebuilt[1]
            if isinstance(rebuilt, tuple) and len(rebuilt) == 2
            else self.template.cache
        )

        floating_list: list[Tensor] = []
        kept_list: list[Tensor] = []
        pair_spec = _extract_spec(
            (component, cache_view),
            floating_list=floating_list,
            kept_list=kept_list,
            tensor_refs_by_id={},
            target_dtype=self.anchor.dtype,
            active_obj_ids=set(),
            memo_specs={},
            snapshot_mutable=True,
        )
        new_template = CustomClassicalTemplate(
            component=component,
            cache=cache_view,
            pair_spec=pair_spec,
            primary_ids=tuple(id(t) for t in floating_list),
            secondary_ids=tuple(id(t) for t in kept_list),
            initial_dd={
                "device": self.anchor.device,
                "dtype": self.anchor.dtype,
            },
            mutation_snapshot=(
                snapshot_component_state(component)
                if mutation_snapshot is None
                else mutation_snapshot
            ),
        )
        return CustomClassicalCoreSetup(
            anchor=self.anchor,
            floating_tensors=tuple(floating_list),
            kept_dtype_tensors=tuple(kept_list),
            template=new_template,
        )


def can_build_custom_classical_core_setup(
    component: Component,
    cache: Any,
    dd: DD,
) -> bool:
    """Return ``True`` when ``(component, cache)`` is tensor-only on ``dd['device']``."""
    if not (is_tensor_only_object(component) and is_tensor_only_object(cache)):
        return False
    floating_list: list[Tensor] = []
    kept_list: list[Tensor] = []
    _extract_spec(
        (component, cache),
        floating_list=floating_list,
        kept_list=kept_list,
        tensor_refs_by_id={},
        target_dtype=dd["dtype"],
        active_obj_ids=set(),
        memo_specs={},
    )
    target_device = _resolve_dd_device(dd["device"])
    return all(t.device == target_device for t in (*floating_list, *kept_list))


def build_custom_classical_core_setup(
    component: Component,
    cache: Any,
    dd: DD,
    *,
    mutation_snapshot: ComponentMutationSnapshot | None = None,
) -> CustomClassicalCoreSetup:
    """Wrap a tensor-only custom ``Classical`` and its setup into a ``Node``."""
    target_device = _resolve_dd_device(dd["device"])
    floating_list: list[Tensor] = []
    kept_list: list[Tensor] = []
    pair_spec = _extract_spec(
        (component, cache),
        floating_list=floating_list,
        kept_list=kept_list,
        tensor_refs_by_id={},
        target_dtype=dd["dtype"],
        active_obj_ids=set(),
        memo_specs={},
        snapshot_mutable=True,
    )
    template = CustomClassicalTemplate(
        component=component,
        cache=cache,
        pair_spec=pair_spec,
        primary_ids=tuple(id(t) for t in floating_list),
        secondary_ids=tuple(id(t) for t in kept_list),
        initial_dd={"device": target_device, "dtype": dd["dtype"]},
        mutation_snapshot=(
            snapshot_component_state(component)
            if mutation_snapshot is None
            else mutation_snapshot
        ),
    )
    return CustomClassicalCoreSetup(
        anchor=torch.empty(0, device=target_device, dtype=dd["dtype"]),
        floating_tensors=tuple(floating_list),
        kept_dtype_tensors=tuple(kept_list),
        template=template,
    )


class CustomInteractionTemplate:
    """Non-dataclass template for custom ``SCFInteractionEntry`` reconstruction."""

    __slots__ = (
        "component",
        "cache",
        "pair_spec",
        "primary_count",
        "primary_ids",
        "secondary_ids",
    )

    component: Any
    cache: Any
    pair_spec: Any
    primary_count: int
    primary_ids: tuple[int, ...]
    secondary_ids: tuple[int, ...]

    def __init__(
        self,
        *,
        component: Any,
        cache: Any,
        pair_spec: Any,
        primary_ids: tuple[int, ...],
        secondary_ids: tuple[int, ...],
    ) -> None:
        self.component = component
        self.cache = cache
        self.pair_spec = pair_spec
        self.primary_count = len(primary_ids)
        self.primary_ids = primary_ids
        self.secondary_ids = secondary_ids


def prepare_custom_interaction_entry(
    component: Component,
    cache: Any,
) -> SCFInteractionEntry:
    """Extract all tensors from ``(component, cache)`` into an ``SCFInteractionEntry``."""
    floating_list: list[Tensor] = []
    kept_list: list[Tensor] = []
    pair_spec = _extract_spec(
        (component, cache),
        floating_list=floating_list,
        kept_list=kept_list,
        tensor_refs_by_id={},
        target_dtype=None,
        active_obj_ids=set(),
        memo_specs={},
        snapshot_mutable=False,
    )
    template = CustomInteractionTemplate(
        component=component,
        cache=cache,
        pair_spec=pair_spec,
        primary_ids=tuple(id(t) for t in floating_list),
        secondary_ids=tuple(id(t) for t in kept_list),
    )
    data = tuple(
        [
            *(
                (f"float_{idx}", tensor)
                for idx, tensor in enumerate(floating_list)
            ),
            *(
                (f"static_tensor_{idx}", tensor)
                for idx, tensor in enumerate(kept_list)
            ),
        ]
    )
    return SCFInteractionEntry(
        kind="custom",
        data=data,
        static=(("template", template), ("label", component.label)),
    )


def reconstruct_custom_interaction(
    entry: SCFInteractionEntry,
) -> tuple[Any, Any]:
    """Reconstruct ``(component_view, cache_view)`` from a custom ``SCFInteractionEntry``."""
    template = entry.option("template")
    assert isinstance(template, CustomInteractionTemplate)
    n_primary = template.primary_count
    primary = tuple(value for _, value in entry.data[:n_primary])
    secondary = tuple(value for _, value in entry.data[n_primary:])

    same_primary = len(primary) == len(template.primary_ids) and all(
        id(t) == orig_id for t, orig_id in zip(primary, template.primary_ids)
    )
    same_secondary = len(secondary) == len(template.secondary_ids) and all(
        id(t) == orig_id
        for t, orig_id in zip(secondary, template.secondary_ids)
    )
    if same_primary and same_secondary:
        return template.component, template.cache

    rebuilt = _rebuild_spec(template.pair_spec, primary, secondary, None, {})
    if isinstance(rebuilt, tuple) and len(rebuilt) == 2:
        return rebuilt[0], rebuilt[1]
    return template.component, template.cache
