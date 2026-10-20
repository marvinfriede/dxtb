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
"""Executable C8b exact-core, alias, partition, and stacking preflight."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import FunctionType

import pytest
import torch
from tad_mctc.tree import (
    ModuleNode,
    Node,
    child,
    combine,
    context,
    partition,
    stack,
)
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import (
    ClassicalTermDescriptor,
    ExactSystemCore,
    InteractionTermDescriptor,
    Model,
    System,
)
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.libcint.driver import LibcintIntegralSetupNode
from dxtb._src.integral.driver.pytorch.setup import PytorchIntegralSetupNode


class _IntegralCoreNode(Node):
    """Expose only the selected backend Node under static driver metadata."""

    driver_type: int = context()
    intlevel: int = context()
    backend: PytorchIntegralSetupNode | LibcintIntegralSetupNode | None = (
        child()
    )


class _ExactCoreNode(Node):
    """Test-local enclosing tree; production ExactSystemCore stays opaque."""

    numbers: torch.Tensor = child(keep_dtype=True)
    effective_dtype: torch.dtype = context()
    config: Config = context()
    integral: _IntegralCoreNode | None = child()
    h0: Node | None = child()
    classical_terms: tuple[ClassicalTermDescriptor, ...] = child()
    interaction_terms: tuple[InteractionTermDescriptor, ...] = child()

    @property
    def index(self):
        """Derive the SCF index alias from its selected backend owner."""
        if self.integral is None or self.integral.backend is None:
            return None
        return self.integral.backend.index

    def to(self, device=None, dtype=None):
        """Convert children and keep the effective dtype context coherent."""
        result = super().to(device=device, dtype=dtype)
        return (
            result if dtype is None else replace(result, effective_dtype=dtype)
        )


class _ModelParameterTree(Node):
    """Preflight the Model boundary without placing tables in each System."""

    parameters: ModuleNode = child()
    config: Config = context()
    auto_int_level: bool = context()


def _system(
    method: str = GFN2_XTB,
    numbers: tuple[int, ...] = (1, 1),
    driver: int = labels.INTDRIVER_PYTORCH,
    dtype: torch.dtype = torch.float64,
) -> System:
    method_label = labels.GFN1_XTB if method == GFN1_XTB else labels.GFN2_XTB
    config = Config.create(method=method_label, int_driver=driver)
    model = Model(par=ParamModule(method, dtype=dtype), config=config)
    return model.setup(torch.tensor(numbers, dtype=torch.int64))


def _prototype(system: System) -> _ExactCoreNode:
    core = system.exact_core
    assert core is not None
    setup = core.integral_setup
    backend = None
    driver = system.config.ints.driver
    level = system.config.ints.level
    if setup is not None:
        if driver == labels.INTDRIVER_PYTORCH:
            backend = setup.pytorch
        else:
            backend = setup.libcint
    return _ExactCoreNode(
        numbers=core.numbers,
        effective_dtype=core.effective_dtype,
        config=core.config,
        integral=_IntegralCoreNode(
            driver_type=driver, intlevel=level, backend=backend
        ),
        h0=core.h0_setup,
        classical_terms=core.classical_terms,
        interaction_terms=core.interaction_terms,
    )


def _partition_signature(core: _ExactCoreNode) -> tuple:
    """Return a value-based bucket key without repr or Python identities."""
    leaves, spec = tree_flatten(core)
    backend = None if core.integral is None else core.integral.backend
    return (
        _tree_signature(spec),
        core.config,
        core.effective_dtype,
        (
            None
            if core.integral is None
            else (core.integral.driver_type, core.integral.intlevel)
        ),
        None if backend is None else getattr(backend, "force_cpu", False),
        tuple(
            (term.kind, term.label, term.position, type(term.setup).__name__)
            for term in core.classical_terms
        ),
        tuple(
            (term.kind, term.label, term.position, type(term.setup).__name__)
            for term in core.interaction_terms
        ),
        tuple(
            (tuple(leaf.shape), leaf.dtype, leaf.device.type, leaf.device.index)
            for leaf in leaves
        ),
    )


def _core_repulsion_energy(
    core: ExactSystemCore, positions: torch.Tensor
) -> torch.Tensor:
    """Evaluate the existing Repulsion kernel from a complete core tree."""
    from dxtb._src.calculators.singlepoint import _evaluate_classical_descriptor

    descriptor = next(
        term for term in core.classical_terms if term.kind == "repulsion"
    )
    return _evaluate_classical_descriptor(
        descriptor, positions, positions.new_zeros((1,))
    ).sum()


def _static_signature(value):
    """Canonicalize supported static Node context without object IDs/repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return ("path", value.as_posix())
    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    if isinstance(value, torch.device):
        return ("device", value.type, value.index)
    if isinstance(value, Enum):
        return (type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, type) or isinstance(value, FunctionType):
        return (value.__module__, value.__qualname__)
    if is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (item.name, _static_signature(getattr(value, item.name)))
                for item in fields(value)
            ),
        )
    if isinstance(value, tuple):
        return tuple(_static_signature(item) for item in value)
    raise TypeError(f"Unsupported static signature value: {type(value)!r}")


def _tree_signature(spec):
    """Collect each PyTree node type/context, excluding dynamic leaves."""
    node_type = getattr(spec, "type", None)
    type_key = (
        None
        if node_type is None
        else (node_type.__module__, node_type.__qualname__)
    )
    children = (
        tuple(spec.child(index) for index in range(spec.num_children))
        if hasattr(spec, "child") and hasattr(spec, "num_children")
        else tuple(getattr(spec, "children_specs", ()))
    )
    return (
        type_key,
        _static_signature(getattr(spec, "context", None)),
        tuple(_tree_signature(child_spec) for child_spec in children),
    )


def _contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if is_dataclass(value):
        return any(
            _contains_tensor(getattr(value, item.name))
            for item in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    return False


def _assert_immutable_static(value) -> None:
    """Require nested config values to be frozen scalar/container data."""
    assert not isinstance(value, torch.Tensor)
    if is_dataclass(value):
        assert type(value).__dataclass_params__.frozen
        for item in fields(value):
            _assert_immutable_static(getattr(value, item.name))
        return
    if isinstance(value, tuple):
        for item in value:
            _assert_immutable_static(item)
        return
    assert value is None or isinstance(
        value, (bool, int, float, str, Path, Enum, torch.dtype)
    )


def _walk_spec(spec):
    yield spec
    children = (
        (spec.child(i) for i in range(spec.num_children))
        if hasattr(spec, "child") and hasattr(spec, "num_children")
        else iter(getattr(spec, "children_specs", ()))
    )
    for child_spec in children:
        yield from _walk_spec(child_spec)


@pytest.mark.parametrize(
    "driver",
    [
        labels.INTDRIVER_PYTORCH,
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            marks=pytest.mark.skipif(
                not has_libcint, reason="tad-libcint unavailable"
            ),
        ),
    ],
)
def test_production_exact_core_and_backend_are_registered(
    driver: int,
) -> None:
    """The exact core exposes its selected backend and setup leaves."""
    system = _system(driver=driver)
    assert system.exact_core is not None
    core = system.exact_core
    core_leaves, core_spec = tree_flatten(core)
    assert core_leaves and all(isinstance(x, torch.Tensor) for x in core_leaves)
    assert not any(
        _contains_tensor(node.context) for node in _walk_spec(core_spec)
    )
    assert tree_flatten(system.integral_setup)[0] == [system.integral_setup]
    assert system.integral_setup is not None
    backend = (
        system.integral_setup.pytorch
        if driver == labels.INTDRIVER_PYTORCH
        else system.integral_setup.libcint
    )
    assert isinstance(
        backend, (PytorchIntegralSetupNode, LibcintIntegralSetupNode)
    )
    backend_leaves, _ = tree_flatten(backend)
    outer_leaves, _ = tree_flatten(system.integral_setup)
    assert outer_leaves == [system.integral_setup]
    assert backend_leaves and all(
        isinstance(x, torch.Tensor) for x in backend_leaves
    )
    assert core.integral is not None
    assert core.integral.backend is backend
    legacy_view = core.integral_setup
    assert legacy_view is not None
    assert legacy_view.driver_type == system.integral_setup.driver_type
    assert legacy_view.intlevel == system.integral_setup.intlevel
    assert (
        legacy_view.pytorch is backend
        if isinstance(backend, PytorchIntegralSetupNode)
        else legacy_view.libcint is backend
    )
    assert core.index is backend.index
    expected = [core.numbers]
    expected.extend(tree_flatten(core.integral)[0])
    if core.h0_setup is not None:
        expected.extend(tree_flatten(core.h0_setup)[0])
    for term in core.classical_terms + core.interaction_terms:
        expected.extend(tree_flatten(term)[0])
    assert len(core_leaves) == len(expected)
    assert all(
        actual is wanted for actual, wanted in zip(core_leaves, expected)
    )

    rebuilt = tree_unflatten(core_leaves, core_spec)
    assert type(rebuilt.integral.backend) is type(backend)
    assert rebuilt.index is rebuilt.integral.backend.index
    rebuilt_leaves, _ = tree_flatten(rebuilt)
    assert all(
        actual is original
        for actual, original in zip(rebuilt_leaves, core_leaves)
    )


def test_optional_integral_and_h0_children_round_trip_as_none() -> None:
    """The schema does not synthesize setup data for excluded integrals."""
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=Config.create(method=labels.GFN1_XTB, int_level=0),
        auto_int_level=False,
    )
    system = model.setup(torch.tensor([1, 1]))
    proto = _prototype(system)
    assert system.exact_core is not None
    assert system.exact_core.integral is None
    assert proto.integral.backend is None
    assert proto.h0 is None
    leaves, spec = tree_flatten(proto)
    rebuilt = tree_unflatten(leaves, spec)
    assert rebuilt.integral.backend is None
    assert rebuilt.h0 is None
    assert [term.kind for term in rebuilt.classical_terms] == [
        term.kind for term in proto.classical_terms
    ]
    assert [term.label for term in rebuilt.classical_terms] == [
        term.label for term in proto.classical_terms
    ]
    assert [type(term.setup) for term in rebuilt.classical_terms] == [
        type(term.setup) for term in proto.classical_terms
    ]


@pytest.mark.parametrize(
    "driver",
    [
        labels.INTDRIVER_PYTORCH,
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            marks=pytest.mark.skipif(
                not has_libcint, reason="tad-libcint unavailable"
            ),
        ),
    ],
)
def test_candidate_core_flattens_backend_h0_and_descriptor_setup_nodes(
    driver: int,
) -> None:
    """Both backends and all nested setup Nodes are dynamic children."""
    system = _system(driver=driver)
    assert system.exact_core is not None
    production_core = system.exact_core
    production_leaves, production_spec = tree_flatten(production_core)
    assert production_leaves and all(
        isinstance(leaf, torch.Tensor) for leaf in production_leaves
    )
    assert not any(
        _contains_tensor(node.context) for node in _walk_spec(production_spec)
    )
    assert production_core.integral is not None
    converted_core = production_core.to(dtype=torch.float32)
    assert converted_core.effective_dtype is torch.float32
    assert converted_core.config is production_core.config
    assert converted_core.integral.backend.index is converted_core.index
    for original, converted_leaf in zip(
        production_leaves, tree_flatten(converted_core)[0]
    ):
        if original.dtype == torch.bool or not original.is_floating_point():
            assert converted_leaf.dtype is original.dtype
        else:
            assert converted_leaf.dtype is torch.float32

    proto = _prototype(system)
    leaves, spec = tree_flatten(proto)
    assert leaves and all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    assert not any(_contains_tensor(node.context) for node in _walk_spec(spec))
    expected_leaves = [proto.numbers]
    if proto.integral.backend is not None:
        expected_leaves.extend(tree_flatten(proto.integral.backend)[0])
    if proto.h0 is not None:
        expected_leaves.extend(tree_flatten(proto.h0)[0])
    for term in proto.classical_terms + proto.interaction_terms:
        expected_leaves.extend(tree_flatten(term)[0])
    assert len(leaves) == len(expected_leaves)
    assert all(
        actual is expected for actual, expected in zip(leaves, expected_leaves)
    )
    assert proto.index is proto.integral.backend.index
    assert proto.integral.backend is (
        system.integral_setup.pytorch
        if driver == labels.INTDRIVER_PYTORCH
        else system.integral_setup.libcint
    )
    assert proto.h0 is system.exact_core.h0_setup
    assert proto.classical_terms is system.exact_core.classical_terms
    assert proto.interaction_terms is system.exact_core.interaction_terms
    assert (
        proto.classical_terms[0].setup
        is system.exact_core.classical_terms[0].setup
    )

    rebuilt = tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, _ExactCoreNode)
    assert rebuilt.config == proto.config
    assert rebuilt.effective_dtype is proto.effective_dtype
    assert rebuilt.integral.driver_type == proto.integral.driver_type
    assert rebuilt.integral.intlevel == proto.integral.intlevel
    assert type(rebuilt.integral.backend) is type(proto.integral.backend)
    assert rebuilt.index is rebuilt.integral.backend.index
    assert [term.kind for term in rebuilt.classical_terms] == [
        term.kind for term in proto.classical_terms
    ]
    assert [term.label for term in rebuilt.interaction_terms] == [
        term.label for term in proto.interaction_terms
    ]

    converted = proto.to(dtype=torch.float32)
    assert converted.effective_dtype is torch.float32
    assert converted.config == proto.config
    converted_leaves, _ = tree_flatten(converted)
    if driver == labels.INTDRIVER_LIBCINT:
        assert converted.integral.backend.force_cpu is True
        assert all(leaf.device.type == "cpu" for leaf in converted_leaves)
    assert [term.kind for term in converted.classical_terms] == [
        term.kind for term in proto.classical_terms
    ]
    assert converted.integral.backend is not proto.integral.backend
    assert converted.index is converted.integral.backend.index
    for original, converted_leaf in zip(leaves, converted_leaves):
        if original.dtype == torch.bool or not original.is_floating_point():
            assert converted_leaf.dtype is original.dtype
        else:
            assert converted_leaf.dtype is torch.float32
    cpu_converted = proto.to(device=torch.device("cpu"))
    assert cpu_converted.index is cpu_converted.integral.backend.index
    assert all(
        leaf.device.type == "cpu" for leaf in tree_flatten(cpu_converted)[0]
    )


def test_core_stack_positive_different_composition_and_shape_rejection() -> (
    None
):
    """Equal atom count alone is insufficient; equal leaves can stack."""
    gfn2_h2_a = _system(method=GFN2_XTB, numbers=(1, 1)).exact_core
    gfn2_h2_b = _system(method=GFN2_XTB, numbers=(1, 1)).exact_core
    assert gfn2_h2_a is not None and gfn2_h2_b is not None
    h2_leaves, h2_spec = tree_flatten(gfn2_h2_a)
    h2_b_leaves, h2_b_spec = tree_flatten(gfn2_h2_b)
    assert h2_spec == h2_b_spec
    assert [(x.shape, x.dtype) for x in h2_leaves] == [
        (x.shape, x.dtype) for x in h2_b_leaves
    ]
    same_stacked = stack((gfn2_h2_a, gfn2_h2_b))
    assert same_stacked.numbers.shape == (2, 2)
    assert same_stacked.index is same_stacked.integral.backend.index

    # Carbon and nitrogen have different compositions but compatible GFN1
    # shell/pair structures in this exact-size bucket.
    c2 = _system(method=GFN1_XTB, numbers=(6, 6)).exact_core
    n2 = _system(method=GFN1_XTB, numbers=(7, 7)).exact_core
    assert c2 is not None and n2 is not None
    c2_leaves, c2_spec = tree_flatten(c2)
    n2_leaves, n2_spec = tree_flatten(n2)
    assert c2_spec == n2_spec
    assert _partition_signature(c2) == _partition_signature(n2)
    assert [(x.shape, x.dtype) for x in c2_leaves] == [
        (x.shape, x.dtype) for x in n2_leaves
    ]
    stacked = stack((c2, n2))
    assert stacked.numbers.shape == (2, 2)
    assert stacked.index is stacked.integral.backend.index
    assert stacked.numbers.dtype == torch.int64
    c2_repulsion = next(
        term for term in c2.classical_terms if term.kind == "repulsion"
    )
    n2_repulsion = next(
        term for term in n2.classical_terms if term.kind == "repulsion"
    )
    assert (
        c2_repulsion.setup.arep.data_ptr() != n2_repulsion.setup.arep.data_ptr()
    )

    # A supported family kernel can map over the structurally compatible
    # descriptor batch, independently of full System.singlepoint transforms.
    from dxtb._src.calculators.singlepoint import _evaluate_classical_descriptor

    repulsion_batch = stack((c2_repulsion, n2_repulsion))
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
        ],
        dtype=torch.float64,
    )
    charges = torch.zeros((2, 1), dtype=torch.float64)
    mapped = torch.func.vmap(_evaluate_classical_descriptor)(
        repulsion_batch, positions, charges
    )
    serial = torch.stack(
        [
            _evaluate_classical_descriptor(
                c2_repulsion, positions[0], charges[0]
            ),
            _evaluate_classical_descriptor(
                n2_repulsion, positions[1], charges[1]
            ),
        ]
    )
    torch.testing.assert_close(mapped, serial)

    # Map the complete production core, selecting its existing descriptor
    # inside the pure function instead of mapping descriptor-only inputs.
    core_batch = stack((c2, n2))
    mapped_core = torch.func.vmap(_core_repulsion_energy)(core_batch, positions)
    serial_core = torch.stack(
        [
            _core_repulsion_energy(c2, positions[0]),
            _core_repulsion_energy(n2, positions[1]),
        ]
    )
    torch.testing.assert_close(mapped_core, serial_core)
    mapped_core_grad = torch.func.vmap(
        torch.func.grad(_core_repulsion_energy, argnums=1)
    )(core_batch, positions)
    serial_core_grad = torch.stack(
        [
            torch.func.grad(_core_repulsion_energy, argnums=1)(
                c2, positions[0]
            ),
            torch.func.grad(_core_repulsion_energy, argnums=1)(
                n2, positions[1]
            ),
        ]
    )
    torch.testing.assert_close(mapped_core_grad, serial_core_grad)
    mapped_pos_grad = torch.func.vmap(
        torch.func.grad(
            lambda term, pos, charge: _evaluate_classical_descriptor(
                term, pos, charge
            ).sum(),
            argnums=1,
        ),
        in_dims=(0, 0, 0),
    )(repulsion_batch, positions, charges)
    serial_pos_grad = torch.stack(
        [
            torch.func.grad(
                lambda pos: _evaluate_classical_descriptor(
                    c2_repulsion, pos, charges[0]
                ).sum()
            )(positions[0]),
            torch.func.grad(
                lambda pos: _evaluate_classical_descriptor(
                    n2_repulsion, pos, charges[1]
                ).sum()
            )(positions[1]),
        ]
    )
    torch.testing.assert_close(mapped_pos_grad, serial_pos_grad)

    h2 = _system(method=GFN1_XTB, numbers=(1, 1)).exact_core
    lih = _system(method=GFN1_XTB, numbers=(3, 1)).exact_core
    assert h2 is not None and lih is not None
    with pytest.raises((ValueError, RuntimeError)):
        stack((h2, lih))

    # Same-sized GFN2 inputs can still disagree in shell-resolved leaves.
    gfn2_h2 = _prototype(_system(method=GFN2_XTB, numbers=(1, 1)))
    gfn2_lih = _prototype(_system(method=GFN2_XTB, numbers=(3, 1)))
    shapes_h2 = [x.shape for x in tree_flatten(gfn2_h2)[0]]
    shapes_lih = [x.shape for x in tree_flatten(gfn2_lih)[0]]
    assert shapes_h2 != shapes_lih
    assert _partition_signature(gfn2_h2) != _partition_signature(gfn2_lih)
    with pytest.raises((ValueError, RuntimeError)):
        stack((gfn2_h2, gfn2_lih))


def test_model_module_partition_keeps_one_parameter_owner_and_gradients() -> (
    None
):
    """ModuleNode exposes original ParamModule leaves to partition/combine."""
    model = Model(par=ParamModule(GFN1_XTB, dtype=torch.float64))
    floating_parameter = next(
        value for value in model.par.parameters() if value.is_floating_point()
    )
    floating_parameter.requires_grad_(True)
    module_node = ModuleNode.from_module(model.par)
    model_tree = _ModelParameterTree(
        parameters=module_node,
        config=model.config,
        auto_int_level=model.auto_int_level,
    )
    assert set(module_node.params) == {
        name for name, _ in model.par.named_parameters()
    } | {name for name, _ in model.par.named_buffers()}
    original = dict(model.par.named_parameters())
    assert all(
        module_node.params[name] is value for name, value in original.items()
    )

    params, remainder = partition(
        model_tree, lambda path, value: ".params" in path
    )
    assert params
    assert all(
        value is module_node.params[key.split("['", 1)[-1].split("']", 1)[0]]
        for key, value in params.items()
        if ".params" in key
    )
    reconstructed = combine(params, remainder)
    assert isinstance(reconstructed, _ModelParameterTree)
    assert reconstructed.parameters.module is model.par

    selected = next(key for key, value in params.items() if value.requires_grad)
    gradient = torch.autograd.grad(params[selected].sum(), params[selected])[0]
    torch.testing.assert_close(gradient, torch.ones_like(params[selected]))


def test_parameter_to_setup_gradient_survives_core_wrapping() -> None:
    """Registered descriptor children preserve gathered setup autograd edges."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    trainable = tuple(
        value for value in par.parameters() if value.is_floating_point()
    )
    for value in trainable:
        value.requires_grad_(True)
    system = Model(par=par).setup(torch.tensor([1, 1]))
    core = _prototype(system)
    leaves, spec = tree_flatten(core)
    reconstructed = tree_unflatten(leaves, spec)
    repulsion = next(
        term
        for term in reconstructed.classical_terms
        if term.kind == "repulsion"
    )
    assert repulsion.setup.arep.requires_grad
    grads = torch.autograd.grad(
        repulsion.setup.arep.sum(), trainable, allow_unused=True
    )
    assert any(grad is not None and torch.any(grad != 0) for grad in grads)


def test_config_static_values_are_immutable_and_tensor_free() -> None:
    """Config can be static context; runtime numeric inputs remain dynamic."""
    config = Config.create(method=labels.GFN2_XTB)
    assert hash(config)
    assert not _contains_tensor(config)
    _assert_immutable_static(config)
    assert isinstance(config.exclude, tuple)
    assert isinstance(config.file, (str, Path, type(None)))
