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

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from tad_mctc.tree import combine, partition
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import (
    Model,
    ModelParameterNode,
    System,
)
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
