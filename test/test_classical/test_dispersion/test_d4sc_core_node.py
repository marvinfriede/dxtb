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
"""Tests for the System-owned D4SC Node representation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from typing import Any

import tad_dftd4 as d4
import torch
import pytest
from tad_mctc.tree import Node, stack
from torch.utils import _pytree

from dxtb import GFN2_XTB, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.interactions.dispersion.d4sc import (
    D4SCCoreSetup,
    D4SCSetup,
    DispersionD4SC,
    _StaticList,
    _StaticMapping,
    _build_d4sc_model,
    _extra_model_state,
    _restore_static,
    build_d4sc_data,
    core_d4sc_setup,
    new_d4sc,
    setup_d4sc,
)
from dxtb._src.typing import Tensor
from dxtb.config import Config

from .samples import samples

DEVICE = torch.device("cpu")
DD = {"device": DEVICE, "dtype": torch.double}


def _sample(name: str = "LiH") -> tuple[Tensor, Tensor]:
    sample = samples[name]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _interaction(numbers: Tensor) -> DispersionD4SC:
    par = ParamModule(GFN2_XTB, **DD)
    interaction = new_d4sc(numbers, par, **DD)
    assert interaction is not None
    return interaction


def _model() -> Model:
    return Model(
        par=ParamModule(GFN2_XTB, **DD),
        config=Config.create(exclude=("es2", "es3", "aes2")),
        auto_int_level=False,
    )


def _tensor_occurrences(value: Any) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, _StaticMapping):
        return [
            tensor
            for key, item in value.items
            for tensor in (_tensor_occurrences(key) + _tensor_occurrences(item))
        ]
    if isinstance(value, _StaticList):
        return [
            tensor
            for item in value.items
            for tensor in _tensor_occurrences(item)
        ]
    if isinstance(value, (tuple, list)):
        return [
            tensor for item in value for tensor in _tensor_occurrences(item)
        ]
    return []


def _core_tensor_values(value: Any) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [
            tensor for item in value for tensor in _core_tensor_values(item)
        ]
    if isinstance(value, dict):
        return [
            tensor
            for item in value.values()
            for tensor in _core_tensor_values(item)
        ]
    if isinstance(value, _StaticMapping):
        return [
            tensor
            for key, item in value.items
            for tensor in (_core_tensor_values(key) + _core_tensor_values(item))
        ]
    if isinstance(value, _StaticList):
        return [
            tensor
            for item in value.items
            for tensor in _core_tensor_values(item)
        ]
    return []


def _setup_tensors(setup: D4SCSetup) -> list[Tensor]:
    return [
        setup.numbers,
        *_tensor_occurrences(setup.parameters),
        setup.rcov,
        setup.r4r2,
        setup.cn_cutoff,
        setup.pair_weight,
        setup.model_wf,
        setup.model_rc6,
        *_tensor_occurrences(setup.model_extra),
    ]


def test_core_node_roundtrip_has_complete_dynamic_tensor_leaves() -> None:
    """D4SC core round-trips every tensor as one dynamic leaf."""
    numbers, _ = _sample()
    legacy = setup_d4sc(_interaction(numbers), numbers)
    core = core_d4sc_setup(legacy)

    assert isinstance(core, Node)
    assert type(core) is D4SCCoreSetup
    assert [field.name for field in fields(core)] == [
        "numbers",
        "parameter_entries",
        "rcov",
        "r4r2",
        "cn_cutoff",
        "pair_weight",
        "counting_function",
        "model_type",
        "model_ga",
        "model_gc",
        "model_wf",
        "model_ref_charges",
        "model_rc6",
        "model_extra_entries",
    ]

    leaves, spec = _pytree.tree_flatten(core)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is D4SCCoreSetup
    assert all(isinstance(leaf, Tensor) for leaf in leaves)
    assert len(leaves) == len(_setup_tensors(legacy))
    assert all(new is old for new, old in zip(leaves, _setup_tensors(legacy)))
    path_leaves, path_spec = _pytree.tree_flatten_with_path(core)
    repeat_paths, repeat_spec = _pytree.tree_flatten_with_path(core)
    assert path_spec == repeat_spec
    assert [path for path, _ in path_leaves] == [
        path for path, _ in repeat_paths
    ]
    rendered_paths = [str(path) for path, _ in path_leaves]
    assert any("numbers" in path for path in rendered_paths)
    assert any("rcov" in path for path in rendered_paths)
    assert any("model_wf" in path for path in rendered_paths)
    assert any("parameter_entries" in path for path in rendered_paths)
    assert rebuilt.parameters == core.parameters
    assert rebuilt.model_extra == core.model_extra
    assert rebuilt.counting_function is legacy.counting_function
    assert rebuilt.model_type is legacy.model_type
    assert rebuilt.model_ga == legacy.model_ga
    assert rebuilt.model_gc == legacy.model_gc
    assert rebuilt.model_ref_charges == legacy.model_ref_charges
    assert core.to() is core
    assert not any(
        isinstance(getattr(core, field.name), (D4SCSetup, DispersionD4SC))
        for field in fields(core)
    )


def test_sparse_parameters_roundtrip_without_filling_missing_values() -> None:
    """Sparse D4 parameter mappings retain their exact ordered keys."""
    numbers, _ = _sample()
    legacy = setup_d4sc(_interaction(numbers), numbers)
    for omitted in ("s9", "s10"):
        sparse = replace(
            legacy,
            parameters=tuple(
                (name, value)
                for name, value in legacy.parameters
                if name != omitted
            ),
        )
        core = core_d4sc_setup(sparse)
        assert tuple(name for name, _ in core.parameters) == tuple(
            name for name, _ in sparse.parameters
        )
        assert tuple(core.as_param().keys()) == tuple(
            name for name, _ in sparse.as_param().items()
        )
        rebuilt_params = dict(core.as_param().items())
        original_params = dict(sparse.as_param().items())
        for name, value in original_params.items():
            rebuilt_value = rebuilt_params[name]
            if isinstance(value, Tensor):
                torch.testing.assert_close(rebuilt_value, value)
            else:
                assert type(rebuilt_value) is type(value)
                assert rebuilt_value == value
        for (core_name, core_value), (old_name, old_value) in zip(
            core.parameters, sparse.parameters
        ):
            assert core_name == old_name
            if isinstance(old_value, Tensor):
                assert core_value is old_value
            else:
                assert type(core_value) is type(old_value)
                assert core_value == old_value


def test_mixed_parameter_dtype_is_preserved_until_explicit_conversion() -> None:
    """Mixed-dtype setup tensors remain dynamic without implicit casting."""
    numbers, _ = _sample()
    legacy = setup_d4sc(_interaction(numbers), numbers)
    parameters = tuple(
        (
            name,
            (
                value.to(dtype=torch.float32)
                if name == "a1" and isinstance(value, Tensor)
                else value
            ),
        )
        for name, value in legacy.parameters
    )
    mixed = replace(legacy, parameters=parameters)
    core = core_d4sc_setup(mixed)
    values = dict(core.parameters)
    assert values["a1"].dtype == torch.float32
    assert values["a2"].dtype == torch.float64
    assert core.to() is core
    converted = core.to(dtype=torch.float32)
    assert all(
        not isinstance(value, Tensor) or value.dtype == torch.float32
        for _, value in converted.parameters
    )


def test_nested_model_state_is_encoded_as_dynamic_nodes() -> None:
    """Nested dict, slot, tuple and list model tensors remain dynamic."""
    numbers, _ = _sample()
    interaction = _interaction(numbers)

    class NestedD4Model(d4.model.D4Model):
        __slots__ = ("slot_state",)

        def weight_references(self, *args, **kwargs):
            weights = super().weight_references(*args, **kwargs)
            if isinstance(weights, tuple):
                return tuple(self.scale * value for value in weights)
            return self.scale * weights

    original_model = interaction.model
    model = NestedD4Model(
        numbers,
        ga=original_model.ga,
        gc=original_model.gc,
        wf=original_model.wf,
        ref_charges=original_model.ref_charges,
        rc6=original_model.rc6,
        **DD,
    )
    scale = torch.tensor(1.03, **DD, requires_grad=True)
    direct = torch.tensor(2.0, **DD, requires_grad=True)
    tuple_tensor = torch.tensor(3.0, **DD, requires_grad=True)
    list_a = torch.tensor(4.0, **DD, requires_grad=True)
    list_b = torch.tensor(5.0, **DD, requires_grad=True)
    mapping_tensor = torch.tensor(6.0, **DD, requires_grad=True)
    slot_tensor = torch.tensor(7.0, **DD, requires_grad=True)
    model.scale = scale
    model.direct_state = direct
    model.tuple_state = (tuple_tensor, "tuple-mode", 9)
    model.list_state = [list_a, (list_b, "nested-list")]
    model.mapping_state = {
        "weight": mapping_tensor,
        "metadata": {"enabled": True},
    }
    model.slot_state = slot_tensor
    interaction.model = model

    legacy = setup_d4sc(interaction, numbers)
    core = core_d4sc_setup(legacy)
    source = _setup_tensors(legacy)
    leaves, spec = _pytree.tree_flatten(core)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert len(leaves) == len(source)
    assert all(new is old for new, old in zip(leaves, source))
    assert tuple(name for name, _ in rebuilt.model_extra) == (
        "scale",
        "direct_state",
        "tuple_state",
        "list_state",
        "mapping_state",
        "slot_state",
    )
    state = dict(rebuilt.model_extra)
    assert state["tuple_state"] == (tuple_tensor.clone(), "tuple-mode", 9)
    assert isinstance(state["list_state"], _StaticList)
    assert isinstance(state["mapping_state"], _StaticMapping)
    restored = {
        name: _restore_static(value) for name, value in rebuilt.model_extra
    }
    assert isinstance(restored["list_state"], list)
    assert isinstance(restored["mapping_state"], dict)
    assert torch.equal(restored["list_state"][0], list_a)
    assert torch.equal(restored["list_state"][1][0], list_b)
    assert tuple(restored["mapping_state"]) == ("weight", "metadata")
    assert torch.equal(restored["mapping_state"]["weight"], mapping_tensor)
    assert restored["mapping_state"]["metadata"] == {"enabled": True}
    custom_paths = [
        str(path) for path, _ in _pytree.tree_flatten_with_path(core)[0]
    ]
    assert any("model_extra_entries" in path for path in custom_paths)
    dtype_converted = core.to(dtype=torch.float32)
    converted_state_tensors = _core_tensor_values(dtype_converted.model_extra)
    assert converted_state_tensors
    assert all(
        tensor.dtype == torch.float32 for tensor in converted_state_tensors
    )
    assert dtype_converted.model_type is core.model_type
    assert dtype_converted.counting_function is core.counting_function
    meta_converted = core.to(device="meta")
    assert all(
        tensor.device.type == "meta"
        for tensor in _core_tensor_values(meta_converted.model_extra)
    )
    call_model = _build_d4sc_model(rebuilt)
    assert type(call_model) is NestedD4Model
    assert call_model.slot_state is not slot_tensor
    assert torch.equal(call_model.slot_state, slot_tensor)

    _, positions = _sample()
    data = build_d4sc_data(core, positions)
    charges = torch.tensor([0.1, -0.1], **DD)
    energy = interaction.get_monopole_atom_energy(data, charges).sum()
    (gradient,) = torch.autograd.grad(energy, scale)
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0


def test_core_node_conversion_and_compatible_stack_vmap() -> None:
    """Core values convert dtype/device and map over matching structures."""
    numbers_a, _ = _sample("LiH")
    first = core_d4sc_setup(setup_d4sc(_interaction(numbers_a), numbers_a))
    second = first.replace(numbers=torch.tensor([11, 1]))

    converted = first.to(dtype=torch.float32)
    assert converted.numbers.dtype == first.numbers.dtype
    assert converted.rcov.dtype == torch.float32
    assert all(
        not isinstance(value, Tensor) or value.dtype == torch.float32
        for _, value in converted.parameters
    )
    assert converted.model_type is first.model_type
    assert converted.counting_function is first.counting_function
    meta = first.to(device="meta")
    assert meta.numbers.device.type == "meta"
    assert meta.model_rc6.device.type == "meta"
    assert all(
        value.device.type == "meta"
        for _, value in meta.parameters
        if isinstance(value, Tensor)
    )
    with pytest.raises(FrozenInstanceError):
        first.rcov = torch.ones_like(first.rcov)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.model_type = d4.model.D4Model  # type: ignore[misc]

    stacked = stack((first, second))
    assert type(stacked) is D4SCCoreSetup
    assert stacked.numbers.shape[0] == 2
    assert stacked.rcov.shape[0] == 2
    torch.testing.assert_close(
        torch.func.vmap(lambda node: node.rcov.sum())(stacked),
        torch.stack((first.rcov.sum(), second.rcov.sum())),
    )


def test_system_owns_core_node_and_core_builder_matches_legacy() -> None:
    """Model setup stores one Node and both setup forms share D4 physics."""
    numbers, positions = _sample()
    system = _model().setup(numbers)
    assert type(system.d4sc_setup) is D4SCCoreSetup
    assert system.d4sc_interaction is not None
    setup = system.d4sc_setup
    assert setup is not None
    legacy = setup_d4sc(system.d4sc_interaction, numbers)
    assert setup.parameters == legacy.parameters
    assert setup.model_type is legacy.model_type
    legacy_data = build_d4sc_data(legacy, positions)
    core_data = build_d4sc_data(setup, positions)
    torch.testing.assert_close(core_data.cn, legacy_data.cn)
    torch.testing.assert_close(core_data.dispmat, legacy_data.dispmat)
    torch.testing.assert_close(core_data.model.rc6, legacy_data.model.rc6)
    q = torch.tensor([0.1, -0.1], **DD)
    torch.testing.assert_close(
        system.d4sc_interaction.get_monopole_atom_energy(core_data, q),
        system.d4sc_interaction.get_monopole_atom_energy(legacy_data, q),
    )
