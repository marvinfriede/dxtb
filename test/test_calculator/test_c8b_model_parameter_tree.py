# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""C8b Model parameter-tree and functional setup acceptance."""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event, Lock, local
from weakref import ref

import pytest
import torch
from tad_mctc.tree import ModuleNode, combine, partition
from torch.utils._pytree import tree_flatten, tree_unflatten

import dxtb._src.calculators.model as model_module
from dxtb import GFN1_XTB, IndexHelper, ParamModule, labels
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import (
    Model,
    ModelParameterNode,
    System,
    _ModelSetupAdapter,
    _parameter_setup_lock,
)
from dxtb._src.components.classicals import Repulsion, new_repulsion
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.libcint.driver import LibcintIntegralSetupNode
from dxtb._src.integral.driver.pytorch.setup import PytorchIntegralSetupNode


def _model(par: ParamModule) -> Model:
    """Build GFN1 with the dependency-independent PyTorch integral route."""
    return Model(
        par=par,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    )


def _replace_parameter(
    tree: ModelParameterNode, name: str, value: torch.Tensor
) -> ModelParameterNode:
    selected, remainder = partition(tree, lambda path, _: ".params" in path)
    path = next(key for key in selected if key.endswith(f"['{name}']"))
    selected[path] = value
    result = combine(selected, remainder)
    assert isinstance(result, ModelParameterNode)
    return result


def _functional_energy(
    model: Model,
    tree: ModelParameterNode,
    name: str,
    value: torch.Tensor,
    numbers: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, System]:
    system = model.setup_functional(
        numbers, _replace_parameter(tree, name, value)
    )
    return system.singlepoint(positions).energy.sum(), system


def _replace_tree_value(
    tree: ModelParameterNode, name: str, value: torch.Tensor
) -> ModelParameterNode:
    selected, remainder = partition(tree, lambda path, _: ".params" in path)
    path = next(key for key in selected if key.endswith(f"['{name}']"))
    selected[path] = value
    result = combine(selected, remainder)
    assert isinstance(result, ModelParameterNode)
    return result


def _state_snapshot(par: ParamModule):
    parameters = dict(par.named_parameters())
    buffers = dict(par.named_buffers())
    return (
        {name: value for name, value in parameters.items()},
        {name: value for name, value in buffers.items()},
        {name: value.detach().clone() for name, value in parameters.items()},
        {name: value.detach().clone() for name, value in buffers.items()},
        par.dd,
    )


def _assert_state_snapshot(par: ParamModule, snapshot) -> None:
    parameters, buffers, parameter_values, buffer_values, dd = snapshot
    assert par.dd == dd
    current_parameters = dict(par.named_parameters())
    current_buffers = dict(par.named_buffers())
    assert current_parameters.keys() == parameters.keys()
    assert current_buffers.keys() == buffers.keys()
    for name, original in parameters.items():
        assert current_parameters[name] is original
        torch.testing.assert_close(
            current_parameters[name], parameter_values[name]
        )
    for name, original in buffers.items():
        assert current_buffers[name] is original
        torch.testing.assert_close(current_buffers[name], buffer_values[name])


class _ObservedParameterSetupLock:
    """Report entry to a Model-mediated owner-lock acquisition."""

    def __init__(self, lock, attempting: Event) -> None:
        self.lock = lock
        self.attempting = attempting

    def __enter__(self):
        self.attempting.set()
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.lock.release()


def _observe_parameter_setup_attempts(
    monkeypatch: pytest.MonkeyPatch,
    role: local,
    attempts: dict[str, Event],
) -> None:
    """Instrument the actual owner-lock boundary for selected workers."""
    original = model_module._parameter_setup_lock

    def observed(par):
        lock = original(par)
        worker = getattr(role, "worker", None)
        if worker in attempts:
            return _ObservedParameterSetupLock(lock, attempts[worker])
        return lock

    monkeypatch.setattr(model_module, "_parameter_setup_lock", observed)


def test_model_parameter_node_exposes_named_parameters_and_buffers() -> None:
    """The registered Model tree dynamically owns its module state."""
    model = Model(par=ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    named = dict(model.par.named_parameters())
    named.update(dict(model.par.named_buffers()))

    leaves, spec = tree_flatten(tree)
    assert isinstance(tree, ModelParameterNode)
    assert tree.parameters.module is model.par
    assert set(tree.parameters.params) == set(named)
    assert all(
        tree.parameters.params[name] is value for name, value in named.items()
    )
    assert all(
        any(leaf is value for leaf in leaves) for value in named.values()
    )

    selected, remainder = partition(tree, lambda path, _: ".params" in path)
    rebuilt = combine(selected, remainder)
    assert isinstance(rebuilt, ModelParameterNode)
    assert rebuilt.parameters.module is model.par
    assert set(rebuilt.parameters.params) == set(named)

    flattened, rebuilt_spec = tree_flatten(tree_unflatten(leaves, spec))
    assert rebuilt_spec == spec
    assert len(flattened) == len(leaves)
    assert all(
        torch.equal(left, right) for left, right in zip(flattened, leaves)
    )

    converted = tree.to(dtype=torch.float32)
    assert converted.parameters.module is model.par
    assert converted.config is model.config
    for name, value in tree.parameters.params.items():
        converted_value = converted.parameters.params[name]
        if value.is_floating_point():
            assert converted_value.dtype == torch.float32
        else:
            assert converted_value.dtype == value.dtype


def test_functional_repulsion_parameter_reaches_setup_and_energy() -> None:
    """Substituted Repulsion parameters control full evaluation and gradients."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    model = _model(par)
    tree = model.parameter_tree
    name = "parameter_tree.element.H.arep.param"
    original = tree.parameters.params[name]
    original_values = {
        key: value.detach().clone()
        for key, value in dict(par.named_parameters()).items()
    }
    original_buffers = {
        key: value.detach().clone()
        for key, value in dict(par.named_buffers()).items()
    }
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    replacement = original.clone().requires_grad_()
    changed = replacement + 0.1

    energy_a, system_a = _functional_energy(
        model, tree, name, replacement, numbers, positions
    )
    energy_b, system_b = _functional_energy(
        model, tree, name, changed, numbers, positions
    )
    energy_a2, system_a2 = _functional_energy(
        model, tree, name, replacement, numbers, positions
    )

    assert system_a.exact_core is not None
    assert system_b.exact_core is not None
    assert system_a.repulsion_setup is not None
    assert system_b.repulsion_setup is not None
    torch.testing.assert_close(system_a.repulsion_setup.arep[0, 1], replacement)
    torch.testing.assert_close(system_b.repulsion_setup.arep[0, 1], changed)
    repulsion_descriptor = next(
        term
        for term in system_a.exact_core.classical_terms
        if term.kind == "repulsion"
    )
    assert repulsion_descriptor.setup is system_a.repulsion_setup
    assert not hasattr(system_a.exact_core, "par")
    assert system_a.exact_core.integral is not None
    assert system_a.exact_core.integral.backend is not None
    assert (
        system_a.exact_core.index is system_a.exact_core.integral.backend.index
    )
    assert not torch.equal(energy_a, energy_b)
    torch.testing.assert_close(energy_a2, energy_a)
    torch.testing.assert_close(
        system_a2.repulsion_setup.arep[0, 1], replacement
    )

    # An independent ParamModule using its established get() API is the
    # value reference for the substituted setup.
    reference_gradients = []
    reference_position_gradients = []
    for value, expected_energy in (
        (replacement, energy_a),
        (changed, energy_b),
    ):
        reference_par = deepcopy(par)
        reference_parameter = torch.nn.Parameter(value.detach().clone())
        reference_par.get("element.H.arep", unwrapped=False).param = (
            reference_parameter
        )
        reference_system = Model(
            par=reference_par,
            config=model.config,
            classical=model.classical,
            interaction=model.interaction,
            auto_int_level=model.auto_int_level,
        ).setup(numbers)
        reference_positions = positions.detach().clone().requires_grad_(True)
        reference_energy = reference_system.singlepoint(
            reference_positions
        ).energy.sum()
        torch.testing.assert_close(reference_energy, expected_energy)
        reference_parameter_gradient, reference_position_gradient = (
            torch.autograd.grad(
                reference_energy, (reference_parameter, reference_positions)
            )
        )
        reference_gradients.append(reference_parameter_gradient)
        reference_position_gradients.append(reference_position_gradient)

    functional_gradient = torch.func.grad(
        lambda value: _functional_energy(
            model, tree, name, value, numbers, positions
        )[0]
    )(replacement)
    autograd_energy, _ = _functional_energy(
        model, tree, name, replacement, numbers, positions
    )
    (autograd_gradient,) = torch.autograd.grad(autograd_energy, replacement)
    torch.testing.assert_close(functional_gradient, autograd_gradient)
    torch.testing.assert_close(functional_gradient, reference_gradients[0])
    changed_gradient = torch.func.grad(
        lambda value: _functional_energy(
            model, tree, name, value, numbers, positions
        )[0]
    )(changed)
    assert not torch.equal(changed_gradient, functional_gradient)
    torch.testing.assert_close(changed_gradient, reference_gradients[1])
    assert torch.isfinite(functional_gradient)
    (position_gradient,) = torch.autograd.grad(
        energy_a, positions, create_graph=True, retain_graph=True
    )
    assert torch.isfinite(position_gradient).all()
    (candidate_position_gradient,) = torch.autograd.grad(
        energy_b, positions, retain_graph=True
    )
    torch.testing.assert_close(
        candidate_position_gradient, reference_position_gradients[1]
    )
    (parameter_gradient,) = torch.autograd.grad(
        energy_a, replacement, create_graph=True
    )
    (mixed_gradient,) = torch.autograd.grad(
        parameter_gradient, positions, retain_graph=True
    )
    assert torch.isfinite(mixed_gradient).all()
    (position_parameter_gradient,) = torch.autograd.grad(
        position_gradient.sum(), replacement
    )
    torch.testing.assert_close(
        mixed_gradient.sum(), position_parameter_gradient
    )

    assert par.get("element.H.arep") is original
    for key, value in dict(par.named_parameters()).items():
        torch.testing.assert_close(value, original_values[key])
    for key, value in dict(par.named_buffers()).items():
        torch.testing.assert_close(value, original_buffers[key])


def test_functional_es2_parameter_reaches_setup_and_energy() -> None:
    """Substituted ES2 parameters remain active in captured interaction data."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    name = "parameter_tree.charge.effective.gexp.param"
    original = tree.parameters.params[name]
    numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.float64,
    )
    replacement = original.clone().requires_grad_()
    changed = replacement + 0.2

    energy_a, system_a = _functional_energy(
        model, tree, name, replacement, numbers, positions
    )
    energy_b, system_b = _functional_energy(
        model, tree, name, changed, numbers, positions
    )
    assert system_a.es2_setup is not None
    assert system_b.es2_setup is not None
    torch.testing.assert_close(system_a.es2_setup.gexp, replacement)
    torch.testing.assert_close(system_b.es2_setup.gexp, changed)
    assert not torch.equal(energy_a, energy_b)
    gradient = torch.func.grad(
        lambda value: _functional_energy(
            model, tree, name, value, numbers, positions
        )[0]
    )(replacement)
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0
    assert model.par.get("charge.effective.gexp") is original


def test_functional_setup_rejects_a_foreign_parameter_owner() -> None:
    """A functional tree is tied to its source Model architecture."""
    first = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    second = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    with pytest.raises(ValueError, match="does not belong"):
        first.setup_functional(
            torch.tensor([1, 1], dtype=torch.int64), second.parameter_tree
        )


@pytest.mark.parametrize(
    ("driver", "expected_type"),
    [
        (labels.INTDRIVER_PYTORCH, PytorchIntegralSetupNode),
        (labels.INTDRIVER_LIBCINT, LibcintIntegralSetupNode),
    ],
)
def test_integral_backend_node_matches_driver(
    driver: int, expected_type: type
) -> None:
    """Backend metadata and concrete setup Node types stay coherent."""
    if driver == labels.INTDRIVER_LIBCINT and not has_libcint:
        pytest.skip("tad-libcint is unavailable")
    config = Config.create(method=labels.GFN1_XTB, int_driver=driver)
    model = Model(par=ParamModule(GFN1_XTB, dtype=torch.float64), config=config)
    system = model.setup(torch.tensor([1, 1], dtype=torch.int64))
    assert system.exact_core is not None
    assert system.exact_core.integral is not None
    backend = system.exact_core.integral.backend
    assert isinstance(backend, expected_type)
    assert system.exact_core.integral.logical_device == system.numbers.device
    assert system.exact_core.index is backend.index
    assert system._active_exact_core() is system.exact_core


def test_integral_backend_node_rejects_crossed_backend_type() -> None:
    """A selected backend cannot be relabeled as the other driver."""
    if not has_libcint:
        pytest.skip("cross-backend proof requires tad-libcint")
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    config = Config.create(
        method=labels.GFN1_XTB, int_driver=labels.INTDRIVER_PYTORCH
    )
    pytorch_system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=config,
    ).setup(numbers)
    libcint_config = Config.create(
        method=labels.GFN1_XTB, int_driver=labels.INTDRIVER_LIBCINT
    )
    libcint_system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=libcint_config,
    ).setup(numbers)
    assert pytorch_system.exact_core is not None
    assert libcint_system.exact_core is not None
    assert pytorch_system.exact_core.integral is not None
    assert libcint_system.exact_core.integral is not None
    with pytest.raises(TypeError, match="does not match driver"):
        pytorch_system.exact_core.integral.replace(
            backend=libcint_system.exact_core.integral.backend
        )
    with pytest.raises(TypeError, match="does not match driver"):
        libcint_system.exact_core.integral.replace(
            backend=pytorch_system.exact_core.integral.backend
        )


def test_exact_core_rejects_inconsistent_logical_device() -> None:
    """Node construction prevents a logical System-device mismatch."""
    system = Model(par=ParamModule(GFN1_XTB)).setup(
        torch.tensor([1, 1], dtype=torch.int64)
    )
    assert system.exact_core is not None
    assert system.exact_core.integral is not None
    with pytest.raises(RuntimeError, match="different devices"):
        system.exact_core.replace(
            integral=system.exact_core.integral.replace(
                logical_device=torch.device("meta")
            )
        )
    assert system._active_exact_core() is system.exact_core

    incomplete_core = system.exact_core.replace(
        integral=system.exact_core.integral.replace(backend=None)
    )
    compatibility_system = replace(system, exact_core=incomplete_core)
    assert compatibility_system._active_exact_core() is None


def test_no_integral_setup_keeps_backend_child_absent() -> None:
    """An intentionally integral-free Model has no backend wrapper child."""
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=0,
    )
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=config,
        auto_int_level=False,
    )
    system = model.setup(torch.tensor([1, 1], dtype=torch.int64))
    assert system.exact_core is not None
    assert system.integral_setup is None
    assert system.exact_core.integral is None
    assert system._active_exact_core() is system.exact_core


def test_functional_setup_preserves_explicit_dtype_override() -> None:
    """Functional setup forwards the existing explicit dd override."""
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=0,
    )
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=config,
        auto_int_level=False,
    )
    system = model.setup_functional(
        torch.tensor([1, 1], dtype=torch.int64),
        model.parameter_tree,
        dd={"device": torch.device("cpu"), "dtype": torch.float32},
    )
    assert system.exact_core is not None
    assert system.exact_core.effective_dtype is torch.float32
    assert system.dd["dtype"] is torch.float32
    assert system.integral_setup is None


def test_functional_parameter_tree_dtype_conversion_drives_numerics() -> None:
    """A converted ModuleNode controls setup, energy, and gradients."""
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float32
    )
    model64 = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree64 = model64.parameter_tree
    snapshot = _state_snapshot(model64.par)
    tree32 = tree64.to(dtype=torch.float32)

    system32 = model64.setup_functional(numbers, tree32)
    assert system32.exact_core is not None
    assert system32.exact_core.effective_dtype is torch.float32
    assert system32.dd["dtype"] is torch.float32
    assert system32.repulsion_setup is not None
    assert system32.h0_setup is not None
    assert system32.repulsion_setup.arep.dtype is torch.float32
    assert system32.h0_setup.hscale.dtype is torch.float32
    converted_parameter = tree32.parameters.params[
        "parameter_tree.element.H.arep.param"
    ]
    torch.testing.assert_close(
        system32.repulsion_setup.arep[0, 1], converted_parameter
    )
    functional_energy = system32.singlepoint(positions).energy.sum()

    direct32 = _model(ParamModule(GFN1_XTB, dtype=torch.float32))
    direct_tree = direct32.parameter_tree
    direct_system = direct32.setup(numbers)
    assert direct_system.repulsion_setup is not None
    torch.testing.assert_close(
        system32.repulsion_setup.arep, direct_system.repulsion_setup.arep
    )
    direct_energy = direct_system.singlepoint(positions).energy.sum()
    torch.testing.assert_close(functional_energy, direct_energy)

    name = "parameter_tree.element.H.arep.param"
    differentiable = converted_parameter.detach().clone().requires_grad_()
    tree32_differentiable = _replace_tree_value(tree32, name, differentiable)
    functional_loss = (
        model64.setup_functional(numbers, tree32_differentiable)
        .singlepoint(positions)
        .energy.sum()
    )
    (functional_gradient,) = torch.autograd.grad(
        functional_loss, differentiable
    )

    direct_parameter = direct_tree.parameters.params[name]
    direct_parameter = direct_parameter.detach().clone().requires_grad_()
    direct_differentiable_tree = _replace_tree_value(
        direct_tree, name, direct_parameter
    )
    direct_differentiable_system = direct32.setup_functional(
        numbers, direct_differentiable_tree
    )
    direct_loss = direct_differentiable_system.singlepoint(
        positions
    ).energy.sum()
    (direct_gradient,) = torch.autograd.grad(direct_loss, direct_parameter)
    assert torch.isfinite(functional_gradient)
    assert functional_gradient.abs() > 0
    torch.testing.assert_close(functional_gradient, direct_gradient)
    _assert_state_snapshot(model64.par, snapshot)


@pytest.mark.parametrize("invalid_kind", ["config", "auto_int_level", "child"])
def test_functional_parameter_tree_rejects_invalid_metadata(
    invalid_kind: str,
) -> None:
    """Functional tree metadata must match the target Model."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    if invalid_kind == "config":
        invalid = tree.replace(config=replace(model.config))
    elif invalid_kind == "auto_int_level":
        invalid = tree.replace(auto_int_level=not model.auto_int_level)
    else:
        invalid = tree.replace(parameters=object())
    snapshot = _state_snapshot(model.par)

    expected = TypeError if invalid_kind == "child" else ValueError
    with pytest.raises(expected):
        model.setup_functional(torch.tensor([1, 1]), invalid)
    _assert_state_snapshot(model.par, snapshot)


@pytest.mark.parametrize("state_error", ["missing", "unexpected", "shape"])
def test_functional_parameter_tree_rejects_invalid_state(
    state_error: str,
) -> None:
    """Malformed ModuleNode mappings fail before mutating module state."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    values = dict(tree.parameters.params)
    name = "parameter_tree.element.H.arep.param"
    if state_error == "missing":
        values.pop(name)
    elif state_error == "unexpected":
        values["unexpected.parameter"] = torch.tensor(1.0, dtype=torch.float64)
    else:
        values[name] = values[name].reshape(1)
    malformed = tree.replace(parameters=tree.parameters.replace(params=values))
    snapshot = _state_snapshot(model.par)

    with pytest.raises(ValueError):
        model.setup_functional(torch.tensor([1, 1]), malformed)
    _assert_state_snapshot(model.par, snapshot)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("parameter_tree.element.H.arep.param", torch.int64),
        ("parameter_tree.element.H.arep.param", torch.bool),
        ("parameter_tree.element.H.arep.param", torch.complex64),
        ("parameter_tree.element.H.ngauss.param", torch.int64),
        ("parameter_tree.element.H.ngauss.param", torch.bool),
        ("parameter_tree.element.H.ngauss.param", torch.float64),
        ("parameter_tree.element.H.ngauss.param", torch.complex64),
    ],
)
def test_functional_parameter_tree_rejects_dtype_category_changes(
    name: str, dtype: torch.dtype
) -> None:
    """Functional substitutions preserve floating and structural categories."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    original = tree.parameters.params[name]
    replacement = torch.ones_like(original, dtype=dtype)
    malformed = _replace_tree_value(tree, name, replacement)
    snapshot = _state_snapshot(model.par)

    with pytest.raises(TypeError, match="dtype"):
        model.setup_functional(torch.tensor([1, 1]), malformed)
    _assert_state_snapshot(model.par, snapshot)


@pytest.mark.parametrize(
    "name",
    [
        "parameter_tree.element.H.arep.param",
        "dummy",
    ],
)
def test_functional_parameter_tree_rejects_inconsistent_float_dtypes(
    name: str,
) -> None:
    """Direct dict mutation cannot bypass the ModuleNode dtype invariant."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    assert tree.parameters.params[name].dtype is torch.float64
    assert all(
        not value.is_floating_point() or value.dtype is torch.float64
        for value in tree.parameters.params.values()
    )
    snapshot = _state_snapshot(model.par)

    # Deliberately mutate the existing child dictionary. Reconstructing the
    # Node would reject this state before it reached setup_functional().
    tree.parameters.params[name] = tree.parameters.params[name].to(
        dtype=torch.float32
    )

    with pytest.raises(
        TypeError, match="floating state tensors must share one dtype"
    ):
        model.setup_functional(torch.tensor([1, 1]), tree)
    _assert_state_snapshot(model.par, snapshot)


def test_functional_parameter_tree_rejects_inconsistent_state_devices() -> None:
    """Device-mixed mappings fail metadata validation before setup."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    name = "parameter_tree.element.H.arep.param"
    snapshot = _state_snapshot(model.par)

    # Meta is used only to exercise the metadata guard; setup/evaluation is
    # never attempted with a meta tensor and this is not CUDA evidence.
    tree.parameters.params[name] = tree.parameters.params[name].to(
        device=torch.device("meta")
    )

    with pytest.raises(ValueError, match="state tensors must share one device"):
        model.setup_functional(torch.tensor([1, 1]), tree)
    _assert_state_snapshot(model.par, snapshot)


def test_functional_parameter_tree_accepts_structural_integer_dtype() -> None:
    """A structural integer leaf remains valid in its registered dtype."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    name = "parameter_tree.element.H.ngauss.param"
    value = tree.parameters.params[name]
    assert value.dtype is torch.uint8
    valid = _replace_tree_value(tree, name, value.clone())
    system = model.setup_functional(torch.tensor([1, 1]), valid)
    assert system.repulsion_setup is not None


def test_functional_setup_uses_one_shallow_state_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation and functional_call use one mapping of original leaves."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    name = "parameter_tree.element.H.arep.param"
    original = tree.parameters.params[name]
    replacement = (original + 0.3).requires_grad_()
    alternate = original + 0.8

    class ChangingStateDict(dict):
        """Expose changed values if setup rereads the mutable input mapping."""

        def __init__(self, values, key, later_value) -> None:
            super().__init__(values)
            self.key = key
            self.later_value = later_value
            self.reads = 0
            self.armed = False

        def items(self):
            self.reads += 1
            result = dict.items(self)
            if self.armed and self.reads == 1:

                def iterate():
                    yield from result
                    self[self.key] = self.later_value

                return iterate()
            return result

    values = dict(tree.parameters.params)
    values[name] = replacement
    source = ChangingStateDict(values, name, alternate)
    state_tree = tree.replace(parameters=tree.parameters.replace(params=source))
    source.reads = 0
    source.armed = True
    original_functional_call = torch.func.functional_call
    forwarded = {}

    def inspect_functional_call(module, state, *args, **kwargs):
        forwarded.update(state)
        return original_functional_call(module, state, *args, **kwargs)

    monkeypatch.setattr(torch.func, "functional_call", inspect_functional_call)
    system = model.setup_functional(torch.tensor([1, 1]), state_tree)
    assert system.repulsion_setup is not None
    torch.testing.assert_close(system.repulsion_setup.arep[0, 1], replacement)
    assert source[name] is replacement
    assert forwarded[f"par.{name}"] is replacement
    assert source.reads == 0
    loss = system.repulsion_setup.arep[0, 1]
    (gradient,) = torch.autograd.grad(loss, replacement)
    torch.testing.assert_close(gradient, torch.ones_like(replacement))


def test_functional_parameter_tree_accepts_partition_reconstruction() -> None:
    """A structurally valid partition/combine round-trip remains usable."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    selected, remainder = partition(tree, lambda path, _: ".params" in path)
    rebuilt = combine(selected, remainder)
    assert isinstance(rebuilt, ModelParameterNode)
    system = model.setup_functional(torch.tensor([1, 1]), rebuilt)
    assert system.repulsion_setup is not None


def test_functional_setup_failure_restores_module_state() -> None:
    """An exception from Model.setup restores parameters and buffers."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    tree = model.parameter_tree
    name = "parameter_tree.element.H.arep.param"
    replacement = tree.parameters.params[name] + 0.7
    substituted = _replace_tree_value(tree, name, replacement)
    snapshot = _state_snapshot(model.par)

    with pytest.raises(ValueError, match="Invalid batch mode"):
        model.setup_functional(numbers, substituted, batch_mode=99)

    _assert_state_snapshot(model.par, snapshot)
    ordinary = model.setup(numbers)
    successful = model.setup_functional(numbers, substituted)
    ordinary_again = model.setup(numbers)
    assert ordinary.repulsion_setup is not None
    assert successful.repulsion_setup is not None
    assert ordinary_again.repulsion_setup is not None
    torch.testing.assert_close(
        ordinary.repulsion_setup.arep, ordinary_again.repulsion_setup.arep
    )
    torch.testing.assert_close(
        ordinary.repulsion_setup.arep[0, 1], tree.parameters.params[name]
    )
    torch.testing.assert_close(
        successful.repulsion_setup.arep[0, 1], replacement
    )
    _assert_state_snapshot(model.par, snapshot)


def test_functional_setup_serializes_shared_parammodule_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent Model façades sharing one owner see their own values."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    first = _model(par)
    second = _model(par)
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    name = "parameter_tree.element.H.arep.param"
    first_tree = first.parameter_tree
    second_tree = second.parameter_tree
    base = first_tree.parameters.params[name]
    first_value, second_value = base + 0.3, base + 0.9
    first_substitution = _replace_tree_value(first_tree, name, first_value)
    second_substitution = _replace_tree_value(second_tree, name, second_value)
    sequential_first = first.setup_functional(numbers, first_substitution)
    sequential_second = second.setup_functional(numbers, second_substitution)
    assert sequential_first.repulsion_setup is not None
    assert sequential_second.repulsion_setup is not None
    expected_first = sequential_first.repulsion_setup.arep[0, 1].clone()
    expected_second = sequential_second.repulsion_setup.arep[0, 1].clone()
    ordinary_reference = first.setup(numbers)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    ordinary_reference_energy = ordinary_reference.singlepoint(positions).energy
    snapshot = _state_snapshot(par)

    first_entered = Event()
    second_entered = Event()
    second_started = Event()
    release_first = Event()
    original_forward = _ModelSetupAdapter.forward
    call_count = 0
    count_guard = Lock()

    def overlapping_forward(adapter, *args, **kwargs):
        nonlocal call_count
        with count_guard:
            call_count += 1
            ordinal = call_count
        if ordinal == 1:
            first_entered.set()
            # Bounded wait: the test always releases this call, even if a
            # regression prevents the second call from entering.
            release_first.wait(timeout=10.0)
        elif ordinal == 2:
            second_entered.set()
        return original_forward(adapter, *args, **kwargs)

    monkeypatch.setattr(_ModelSetupAdapter, "forward", overlapping_forward)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                first.setup_functional, numbers, first_substitution
            )
            assert first_entered.wait(timeout=10.0)

            def second_setup():
                second_started.set()
                return second.setup_functional(numbers, second_substitution)

            second_future = executor.submit(second_setup)
            assert second_started.wait(timeout=10.0)
            # An owner-scoped lock prevents the second adapter from entering.
            second_overlapped = second_entered.wait(timeout=0.5)
            release_first.set()
            first_system = first_future.result(timeout=10.0)
            second_system = second_future.result(timeout=10.0)
    finally:
        release_first.set()

    assert not second_overlapped
    assert first_system.repulsion_setup is not None
    assert second_system.repulsion_setup is not None
    torch.testing.assert_close(
        first_system.repulsion_setup.arep[0, 1], expected_first
    )
    torch.testing.assert_close(
        second_system.repulsion_setup.arep[0, 1], expected_second
    )
    _assert_state_snapshot(par, snapshot)
    ordinary = first.setup(numbers)
    assert ordinary.repulsion_setup is not None
    torch.testing.assert_close(ordinary.repulsion_setup.arep[0, 1], base)
    torch.testing.assert_close(
        ordinary.singlepoint(positions).energy, ordinary_reference_energy
    )


def test_parameter_setup_lock_registry_is_owner_scoped_and_weak() -> None:
    """A shared ParamModule shares one lock without being retained by it."""
    first_par = ParamModule(GFN1_XTB, dtype=torch.float64)
    second_par = ParamModule(GFN1_XTB, dtype=torch.float64)
    first_lock = _parameter_setup_lock(first_par)
    assert _parameter_setup_lock(first_par) is first_lock
    assert _parameter_setup_lock(second_par) is not first_lock

    collected_owner = ParamModule(GFN1_XTB, dtype=torch.float64)
    _parameter_setup_lock(collected_owner)
    owner_ref = ref(collected_owner)
    del collected_owner
    gc.collect()
    assert owner_ref() is None


def test_ordinary_setup_waits_for_functional_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary Model.setup cannot read a shared temporary substitution."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    tree = model.parameter_tree
    name = "parameter_tree.element.H.arep.param"
    baseline = tree.parameters.params[name]
    replacement = baseline + 0.6
    substituted = _replace_tree_value(tree, name, replacement)
    entered = Event()
    release = Event()
    ordinary_attempting = Event()
    ordinary_entered = Event()
    role = local()
    _observe_parameter_setup_attempts(
        monkeypatch, role, {"ordinary": ordinary_attempting}
    )
    original_forward = _ModelSetupAdapter.forward
    original_from_numbers = IndexHelper.from_numbers

    def paused_forward(adapter, *args, **kwargs):
        entered.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("bounded functional setup wait expired")
        return original_forward(adapter, *args, **kwargs)

    def tracked_from_numbers(cls, *args, **kwargs):
        if getattr(role, "worker", None) == "ordinary":
            ordinary_entered.set()
        return original_from_numbers(*args, **kwargs)

    monkeypatch.setattr(_ModelSetupAdapter, "forward", paused_forward)
    monkeypatch.setattr(
        IndexHelper, "from_numbers", classmethod(tracked_from_numbers)
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:

            def functional_setup():
                role.worker = "functional"
                try:
                    return model.setup_functional(numbers, substituted)
                finally:
                    role.worker = None

            def ordinary_setup():
                role.worker = "ordinary"
                try:
                    return model.setup(numbers)
                finally:
                    role.worker = None

            functional_future = executor.submit(functional_setup)
            assert entered.wait(timeout=10.0)
            ordinary_future = executor.submit(ordinary_setup)
            assert ordinary_attempting.wait(timeout=10.0)
            ordinary_overlapped = ordinary_entered.wait(timeout=0.5)
            release.set()
            functional_system = functional_future.result(timeout=10.0)
            ordinary_system = ordinary_future.result(timeout=10.0)
    finally:
        release.set()

    assert not ordinary_overlapped
    assert functional_system.repulsion_setup is not None
    assert ordinary_system.repulsion_setup is not None
    torch.testing.assert_close(
        functional_system.repulsion_setup.arep[0, 1], replacement
    )
    torch.testing.assert_close(
        ordinary_system.repulsion_setup.arep[0, 1], baseline
    )


def test_parameter_tree_snapshot_waits_for_functional_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parameter snapshots wait until temporary module state is restored."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    tree = model.parameter_tree
    name = "parameter_tree.element.H.arep.param"
    baseline = tree.parameters.params[name]
    substituted = _replace_tree_value(tree, name, baseline + 0.6)
    entered = Event()
    release = Event()
    snapshot_attempting = Event()
    snapshot_entered = Event()
    role = local()
    _observe_parameter_setup_attempts(
        monkeypatch, role, {"snapshot": snapshot_attempting}
    )
    original_forward = _ModelSetupAdapter.forward
    original_from_module = ModuleNode.from_module

    def paused_forward(adapter, *args, **kwargs):
        entered.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("bounded functional setup wait expired")
        return original_forward(adapter, *args, **kwargs)

    def tracked_from_module(module):
        if getattr(role, "worker", None) == "snapshot":
            snapshot_entered.set()
        return original_from_module(module)

    monkeypatch.setattr(_ModelSetupAdapter, "forward", paused_forward)
    monkeypatch.setattr(
        ModuleNode, "from_module", staticmethod(tracked_from_module)
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:

            def functional_setup():
                role.worker = "functional"
                try:
                    return model.setup_functional(numbers, substituted)
                finally:
                    role.worker = None

            def parameter_snapshot():
                role.worker = "snapshot"
                try:
                    return model.parameter_tree
                finally:
                    role.worker = None

            functional_future = executor.submit(functional_setup)
            assert entered.wait(timeout=10.0)
            snapshot_future = executor.submit(parameter_snapshot)
            assert snapshot_attempting.wait(timeout=10.0)
            snapshot_overlapped = snapshot_entered.wait(timeout=0.5)
            release.set()
            functional_system = functional_future.result(timeout=10.0)
            restored_tree = snapshot_future.result(timeout=10.0)
    finally:
        release.set()

    assert not snapshot_overlapped
    assert functional_system.repulsion_setup is not None
    torch.testing.assert_close(
        functional_system.repulsion_setup.arep[0, 1], baseline + 0.6
    )
    assert restored_tree.parameters.params[name] is baseline


def test_functional_h0_parameter_reaches_setup_energy_and_gradient() -> None:
    """The substituted GFN1 kpol parameter controls H0 setup and energy."""
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    tree = model.parameter_tree
    name = "parameter_tree.hamiltonian.xtb.kpol.param"
    parameter = tree.parameters.params[name]
    initial = parameter.clone().requires_grad_()
    changed = initial + 0.1
    snapshot = _state_snapshot(model.par)

    def evaluate(value: torch.Tensor):
        candidate = _replace_tree_value(tree, name, value)
        system = model.setup_functional(numbers, candidate)
        assert system.h0_setup is not None
        return system.singlepoint(positions).energy.sum(), system

    energy_p, system_p = evaluate(initial)
    energy_q, system_q = evaluate(changed)
    energy_p2, system_p2 = evaluate(initial)
    assert system_p.h0_setup is not None
    assert system_q.h0_setup is not None
    assert system_p2.h0_setup is not None
    assert not torch.equal(system_p.h0_setup.hscale, system_q.h0_setup.hscale)
    torch.testing.assert_close(
        system_p2.h0_setup.hscale, system_p.h0_setup.hscale
    )
    assert not torch.equal(energy_p, energy_q)
    torch.testing.assert_close(energy_p2, energy_p)

    reference_par = deepcopy(model.par)
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    reference_par.get("hamiltonian.xtb.kpol", unwrapped=False).param = (
        reference_parameter
    )
    reference_model = _model(reference_par)
    reference_system = reference_model.setup(numbers)
    assert reference_system.h0_setup is not None
    torch.testing.assert_close(
        system_p.h0_setup.hscale, reference_system.h0_setup.hscale
    )
    reference_energy = reference_system.singlepoint(positions).energy.sum()
    torch.testing.assert_close(energy_p, reference_energy)

    changed_reference_par = deepcopy(model.par)
    changed_reference_parameter = torch.nn.Parameter(
        parameter.detach().clone() + 0.1
    )
    changed_reference_par.get("hamiltonian.xtb.kpol", unwrapped=False).param = (
        changed_reference_parameter
    )
    changed_reference_system = _model(changed_reference_par).setup(numbers)
    assert changed_reference_system.h0_setup is not None
    torch.testing.assert_close(
        system_q.h0_setup.hscale, changed_reference_system.h0_setup.hscale
    )
    changed_reference_energy = changed_reference_system.singlepoint(
        positions
    ).energy.sum()
    torch.testing.assert_close(energy_q, changed_reference_energy)

    functional_gradient = torch.func.grad(lambda value: evaluate(value)[0])(
        initial
    )
    (reference_gradient,) = torch.autograd.grad(
        reference_energy, reference_parameter
    )
    assert torch.isfinite(functional_gradient)
    assert functional_gradient.abs() > 0
    torch.testing.assert_close(functional_gradient, reference_gradient)
    ordinary_after_gradient = model.setup(numbers)
    assert ordinary_after_gradient.h0_setup is not None
    torch.testing.assert_close(
        ordinary_after_gradient.h0_setup.hscale, system_p.h0_setup.hscale
    )
    _assert_state_snapshot(model.par, snapshot)


def test_functional_setup_preserves_packed_mode_compatibility() -> None:
    """Functional parameters reach the existing mode-1 façade route."""
    model = _model(ParamModule(GFN1_XTB, dtype=torch.float64))
    numbers = torch.tensor([[1, 1], [1, 1]], dtype=torch.int64)
    tree = model.parameter_tree
    name = "parameter_tree.element.H.arep.param"
    baseline = model.setup(numbers, batch_mode=1)
    assert baseline.repulsion_setup is not None
    parameter = tree.parameters.params[name]
    replacement = parameter + 0.4
    functional = model.setup_functional(
        numbers,
        _replace_tree_value(tree, name, replacement),
        batch_mode=1,
    )
    assert functional.batch_mode == 1
    assert functional.exact_core is None
    assert functional._packed_classical_cache is not None
    assert functional.classical_cache is functional._packed_classical_cache
    assert functional.repulsion_setup is not None
    expected_delta = torch.zeros_like(functional.repulsion_setup.arep)
    expected_delta[..., 0, 1] = 0.4
    expected_delta[..., 1, 0] = 0.4
    torch.testing.assert_close(
        functional.repulsion_setup.arep - baseline.repulsion_setup.arep,
        expected_delta,
    )
    torch.testing.assert_close(
        functional.repulsion_setup.arep[0, 0, 1], replacement
    )
    assert model.par.get("element.H.arep") is tree.parameters.params[name]


def test_functional_setup_preserves_custom_classical_component() -> None:
    """Functional setup retains legacy extension identity and cache hooks."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    base = new_repulsion(torch.tensor([1]), par, dtype=torch.float64)
    assert base is not None
    calls = []

    class RepulsionExtension(Repulsion):
        def get_cache(self, numbers, ihelp=None, **kwargs):
            calls.append("get_cache")
            return super().get_cache(numbers, ihelp=ihelp, **kwargs)

    extension = RepulsionExtension(
        base.arep,
        base.zeff,
        base.kexp,
        base.klight,
        base.cutoff,
        device=torch.device("cpu"),
        dtype=torch.float64,
        en=base.en,
        enscale=base.enscale,
    )
    model = Model(
        par=par,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            exclude="rep,disp",
        ),
        classical=(extension,),
    )
    tree = model.parameter_tree
    h0_name = "parameter_tree.hamiltonian.xtb.kpol.param"
    changed_h0 = tree.parameters.params[h0_name] + 0.1
    system = model.setup_functional(
        numbers,
        _replace_tree_value(tree, h0_name, changed_h0),
    )
    assert any(
        component is extension for component in system.classicals.components
    )
    assert system.classicals.components[-1] is extension
    assert "get_cache" in calls
    assert system.classical_cache[extension.label] is not None
    assert system.exact_core is not None
    assert all(
        term.label != extension.label
        for term in system.exact_core.classical_terms
    )
