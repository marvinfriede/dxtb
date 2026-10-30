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
"""Audit the pytree boundary of the staged C6 descriptors."""

from __future__ import annotations

from dataclasses import replace
from typing import get_args, get_type_hints

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import (
    ClassicalTermDescriptor,
    InteractionTermDescriptor,
    Model,
    System,
)
from dxtb._src.calculators.singlepoint import (
    _evaluate_classical_descriptor,
    _interaction_data,
)
from dxtb._src.components.classicals import Classical
from dxtb._src.loader.lazy.lazy_param import LazyLoaderParam
from dxtb._src.scf.functional.core import tensor_leaves
from dxtb._src.scf.functional.interactions import (
    prepare_functional_interactions,
)


def _system(method: LazyLoaderParam, label: int, numbers: list[int]) -> System:
    return Model(
        par=ParamModule(method, dtype=torch.float64),
        config=Config.create(method=label),
    ).setup(torch.tensor(numbers))


def test_c6_descriptor_children_expose_setup_nodes_to_standard_pytree() -> None:
    """Registered descriptors expose their captured setup Node leaves."""
    system = _system(GFN2_XTB, labels.GFN2_XTB, [1, 1])
    assert system.exact_core is not None
    core = system.exact_core
    descriptor = system.exact_core.classical_terms[0]
    setup = descriptor.setup
    assert isinstance(descriptor, Node)

    for term in core.classical_terms:
        component = system.classicals.components[term.position]
        assert term.setup is system.classical_setup(component)
    for term in core.interaction_terms:
        component = system.interactions.components[term.position]
        assert term.setup is system.interaction_setup(component)

    core_leaves, core_spec = tree_flatten(core)
    assert isinstance(core, Node)
    assert core_leaves and all(
        isinstance(leaf, torch.Tensor) for leaf in core_leaves
    )
    expected_core_leaves = [core.numbers]
    if core.integral is not None:
        expected_core_leaves.extend(tree_flatten(core.integral)[0])
    if core.h0_setup is not None:
        expected_core_leaves.extend(tree_flatten(core.h0_setup)[0])
    for term in core.classical_terms + core.interaction_terms:
        expected_core_leaves.extend(tree_flatten(term)[0])
    assert len(core_leaves) == len(expected_core_leaves)
    assert all(
        actual is expected
        for actual, expected in zip(core_leaves, expected_core_leaves)
    )
    rebuilt_core = tree_unflatten(core_leaves, core_spec)
    assert isinstance(rebuilt_core, type(core))
    assert rebuilt_core.index is rebuilt_core.integral.backend.index
    assert [term.kind for term in rebuilt_core.classical_terms] == [
        term.kind for term in core.classical_terms
    ]

    descriptor_leaves, descriptor_spec = tree_flatten(descriptor)
    setup_leaves, setup_spec = tree_flatten(setup)
    assert descriptor_leaves
    assert all(isinstance(leaf, torch.Tensor) for leaf in descriptor_leaves)
    assert len(descriptor_leaves) == len(setup_leaves)
    assert all(
        left is right for left, right in zip(descriptor_leaves, setup_leaves)
    )
    assert descriptor.setup is setup

    rebuilt_descriptor = tree_unflatten(descriptor_leaves, descriptor_spec)
    assert isinstance(rebuilt_descriptor, type(descriptor))
    assert rebuilt_descriptor is not descriptor
    assert rebuilt_descriptor.kind == descriptor.kind
    assert rebuilt_descriptor.label == descriptor.label
    assert rebuilt_descriptor.position == descriptor.position
    assert rebuilt_descriptor.charge_dependent is descriptor.charge_dependent
    assert rebuilt_descriptor.geometry_inputs == descriptor.geometry_inputs
    assert type(rebuilt_descriptor.setup) is type(setup)
    assert tree_flatten(rebuilt_descriptor.setup)[1] == setup_spec
    for actual, expected in zip(
        tree_flatten(rebuilt_descriptor.setup)[0], setup_leaves
    ):
        torch.testing.assert_close(actual, expected)

    interaction = core.interaction_terms[0]
    assert isinstance(interaction, Node)
    interaction_leaves, interaction_spec = tree_flatten(interaction)
    interaction_tensor_leaves, _ = tree_flatten(interaction.setup)
    assert len(interaction_leaves) == len(interaction_tensor_leaves)
    assert all(
        left is right
        for left, right in zip(interaction_leaves, interaction_tensor_leaves)
    )
    rebuilt_interaction = tree_unflatten(interaction_leaves, interaction_spec)
    assert rebuilt_interaction.kind == interaction.kind
    assert rebuilt_interaction.label == interaction.label
    assert rebuilt_interaction.position == interaction.position
    assert rebuilt_interaction.geometry_inputs == interaction.geometry_inputs
    assert type(rebuilt_interaction.setup) is type(interaction.setup)

    # The implicit adapter and standard pytree now reach the same tensor objects.
    nested = tensor_leaves(descriptor)
    assert nested
    assert all(path.startswith("setup.") for path, _ in nested)
    setup_values = {id(value) for _, value in tensor_leaves(setup)}
    assert {id(value) for _, value in nested} == setup_values
    assert {id(value) for value in descriptor_leaves} == setup_values

    def assert_no_tensor_context(spec) -> None:
        def contains_tensor(value) -> bool:
            if isinstance(value, torch.Tensor):
                return True
            if isinstance(value, (tuple, list)):
                return any(contains_tensor(item) for item in value)
            if isinstance(value, dict):
                return any(contains_tensor(item) for item in value.values())
            return False

        assert not contains_tensor(spec.context)
        children = (
            (spec.child(index) for index in range(spec.num_children))
            if hasattr(spec, "child") and hasattr(spec, "num_children")
            else iter(spec.children_specs)
        )
        for child_spec in children:
            assert_no_tensor_context(child_spec)

    assert_no_tensor_context(descriptor_spec)
    wrapped = (descriptor, None, ())
    wrapped_leaves, wrapped_spec = tree_flatten(wrapped)
    rebuilt_wrapped = tree_unflatten(wrapped_leaves, wrapped_spec)
    assert rebuilt_wrapped[1:] == (None, ())
    assert type(rebuilt_wrapped[0].setup) is type(descriptor.setup)
    assert all(
        left is right
        for left, right in zip(
            tree_flatten(rebuilt_wrapped[0].setup)[0], setup_leaves
        )
    )


def test_core_setup_union_contains_all_registered_families() -> None:
    """The descriptor setup annotation names all eleven family Nodes."""
    expected = {
        "RepulsionCoreSetup",
        "ShortRangeBondCoreSetup",
        "DispersionD3CoreSetup",
        "DispersionD4CoreSetup",
        "IESCoreSetup",
        "HalogenCoreSetup",
        "ES2CoreSetup",
        "ES3CoreSetup",
        "AES2CoreSetup",
        "D4SCCoreSetup",
        "GeneralizedBornCoreSetup",
    }
    for descriptor_type in (ClassicalTermDescriptor, InteractionTermDescriptor):
        setup_type = get_type_hints(descriptor_type)["setup"]
        assert {member.__name__ for member in get_args(setup_type)} == expected


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB])
def test_singlepoint_coherence_checks_do_not_construct_descriptors(
    monkeypatch: pytest.MonkeyPatch, method: LazyLoaderParam
) -> None:
    """Persistent descriptors are reused during repeated evaluations."""
    from dxtb._src.calculators.singlepoint import singlepoint

    if method is GFN1_XTB:
        method_label = labels.GFN1_XTB
    elif method is GFN2_XTB:
        method_label = labels.GFN2_XTB
    else:
        raise ValueError(f"Unsupported parameter loader: {method!r}")
    positions_a = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    positions_b = positions_a.clone()
    positions_b[1, 2] += 0.05

    system = _system(method, method_label, [1, 1])
    assert system.config.method == method_label
    expected_h0_method = "gfn1" if method is GFN1_XTB else "gfn2"
    assert system.h0_setup.method == expected_h0_method
    assert system.exact_core is not None
    persistent_classical = system.exact_core.classical_terms
    persistent_interactions = system.exact_core.interaction_terms
    assert persistent_classical or persistent_interactions

    counts = {ClassicalTermDescriptor: 0, InteractionTermDescriptor: 0}
    for descriptor_type in counts:
        original_init = descriptor_type.__init__

        def counted_init(
            self, *args, __init=original_init, __type=descriptor_type, **kwargs
        ):
            counts[__type] += 1
            __init(self, *args, **kwargs)

        monkeypatch.setattr(descriptor_type, "__init__", counted_init)

    for positions in (positions_a, positions_b, positions_a):
        singlepoint(system, positions)

    assert counts == {ClassicalTermDescriptor: 0, InteractionTermDescriptor: 0}
    assert system._active_exact_core() is system.exact_core
    assert system.exact_core.classical_terms is persistent_classical
    assert system.exact_core.interaction_terms is persistent_interactions


def test_full_and_implicit_coherence_checks_do_not_construct_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FULL and implicit calls keep using setup-time descriptor identities."""
    system = _system(GFN1_XTB, labels.GFN1_XTB, [1, 1])
    assert system.exact_core is not None
    persistent_classical = system.exact_core.classical_terms
    persistent_interactions = system.exact_core.interaction_terms
    counts = {ClassicalTermDescriptor: 0, InteractionTermDescriptor: 0}
    for descriptor_type in counts:
        original_init = descriptor_type.__init__

        def counted_init(
            self, *args, __init=original_init, __type=descriptor_type, **kwargs
        ):
            counts[__type] += 1
            __init(self, *args, **kwargs)

        monkeypatch.setattr(descriptor_type, "__init__", counted_init)

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    full_result = system.singlepoint(positions)
    assert full_result.energy is not None

    scf = replace(
        system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        mixer=labels.MIXER_LINEAR,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    config = replace(system.config, scf=scf)
    assert system.exact_core is not None
    implicit_core = replace(system.exact_core, config=config)
    implicit_system = replace(system, config=config, exact_core=implicit_core)
    implicit_result = implicit_system.singlepoint(positions)
    assert implicit_result.energy is not None

    assert counts == {ClassicalTermDescriptor: 0, InteractionTermDescriptor: 0}
    assert system.exact_core.classical_terms is persistent_classical
    assert system.exact_core.interaction_terms is persistent_interactions
    assert implicit_system._active_exact_core() is implicit_core


def test_c6_descriptor_method_compositions_have_distinct_trees() -> None:
    """Different exact method compositions remain structurally distinct."""
    gfn1 = _system(GFN1_XTB, labels.GFN1_XTB, [1, 1]).exact_core
    gfn2 = _system(GFN2_XTB, labels.GFN2_XTB, [1, 1]).exact_core
    assert gfn1 is not None and gfn2 is not None

    assert [term.kind for term in gfn1.classical_terms] == [
        "halogen",
        "d3",
        "repulsion",
    ]
    assert [term.kind for term in gfn1.interaction_terms] == ["es2", "es3"]
    assert [term.kind for term in gfn2.classical_terms] == ["d4", "repulsion"]
    assert [term.kind for term in gfn2.interaction_terms] == [
        "es2",
        "aes2",
        "es3",
        "d4sc",
    ]

    gfn1_leaves, gfn1_spec = tree_flatten(gfn1.classical_terms)
    gfn2_leaves, gfn2_spec = tree_flatten(gfn2.classical_terms)
    assert gfn1_spec != gfn2_spec
    assert gfn1_leaves and gfn2_leaves
    assert all(isinstance(leaf, torch.Tensor) for leaf in gfn1_leaves)
    assert all(isinstance(leaf, torch.Tensor) for leaf in gfn2_leaves)


def test_c6_descriptor_trees_stack_only_for_compatible_setup_shapes() -> None:
    """Stack matching descriptor trees without sharing setup storage."""
    first = _system(GFN2_XTB, labels.GFN2_XTB, [1, 1]).exact_core
    second = _system(GFN2_XTB, labels.GFN2_XTB, [1, 1]).exact_core
    assert first is not None and second is not None

    first_tree = (first.classical_terms, first.interaction_terms)
    second_tree = (second.classical_terms, second.interaction_terms)
    first_leaves, first_spec = tree_flatten(first_tree)
    second_leaves, second_spec = tree_flatten(second_tree)
    assert first_spec == second_spec
    assert len(first_leaves) == len(second_leaves)
    assert [leaf.shape for leaf in first_leaves] == [
        leaf.shape for leaf in second_leaves
    ]
    assert [leaf.dtype for leaf in first_leaves] == [
        leaf.dtype for leaf in second_leaves
    ]
    for lhs, rhs in zip(first_leaves, second_leaves):
        assert lhs is not rhs
        if lhs.numel() and rhs.numel():
            assert (
                lhs.untyped_storage().data_ptr()
                != rhs.untyped_storage().data_ptr()
            )

    stacked = stack((first_tree, second_tree))
    stacked_leaves, stacked_spec = tree_flatten(stacked)
    assert stacked_spec == first_spec
    assert all(
        value.shape[0] == 2 and value.dtype == original.dtype
        for value, original in zip(stacked_leaves, first_leaves)
    )

    lih = _system(GFN2_XTB, labels.GFN2_XTB, [3, 1]).exact_core
    assert lih is not None
    lih_tree = (lih.classical_terms, lih.interaction_terms)
    lih_leaves, lih_spec = tree_flatten(lih_tree)
    assert lih_spec == first_spec
    assert any(a.shape != b.shape for a, b in zip(first_leaves, lih_leaves))
    with pytest.raises((RuntimeError, ValueError)):
        stack((first_tree, lih_tree))


@pytest.mark.parametrize(
    "method,label,kind",
    [
        (GFN1_XTB, labels.GFN1_XTB, "d3"),
        (GFN2_XTB, labels.GFN2_XTB, "aes2"),
    ],
)
def test_c6_descriptor_conversion_preserves_metadata_and_structure(
    method: str, label: int, kind: str
) -> None:
    """Node conversion changes floating leaves and keeps structural dtypes."""
    core = _system(method, label, [1, 1]).exact_core
    assert core is not None
    descriptor = next(
        term
        for term in (*core.classical_terms, *core.interaction_terms)
        if term.kind == kind
    )
    leaves, _ = tree_flatten(descriptor)
    converted = descriptor.to(dtype=torch.float32)
    converted_leaves, _ = tree_flatten(converted)
    assert type(converted) is type(descriptor)
    assert (converted.kind, converted.label, converted.position) == (
        descriptor.kind,
        descriptor.label,
        descriptor.position,
    )
    assert converted.geometry_inputs == descriptor.geometry_inputs
    if hasattr(descriptor, "charge_dependent"):
        assert converted.charge_dependent is descriptor.charge_dependent
    assert len(leaves) == len(converted_leaves)
    assert any(
        old.is_floating_point() and new.dtype == torch.float32
        for old, new in zip(leaves, converted_leaves)
    )
    if kind == "d3":
        assert any(old.dtype in (torch.int32, torch.int64) for old in leaves)
    if kind == "aes2":
        assert any(old.dtype == torch.bool for old in leaves)
    for old, new in zip(leaves, converted_leaves):
        if old.is_floating_point():
            assert new.dtype == torch.float32
        else:
            assert new.dtype == old.dtype
    roundtrip = converted.to(dtype=torch.float64)
    for old, new in zip(leaves, tree_flatten(roundtrip)[0]):
        if old.is_floating_point():
            torch.testing.assert_close(new, old)
        else:
            assert new.dtype == old.dtype
            torch.testing.assert_close(new, old)
    cpu_copy = descriptor.to(device="cpu")
    assert type(cpu_copy) is type(descriptor)
    assert all(leaf.device.type == "cpu" for leaf in tree_flatten(cpu_copy)[0])


def test_c6_repulsion_descriptor_preserves_parameter_and_mixed_gradients() -> (
    None
):
    """The registered child keeps the original parameter graph intact."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    par.set_differentiable("element.H.arep")
    numbers = torch.tensor([1, 1])
    system = Model(
        par=par,
        config=Config.create(method=labels.GFN1_XTB, exclude="scf"),
    ).setup(numbers)
    assert system.exact_core is not None
    descriptor = next(
        term
        for term in system.exact_core.classical_terms
        if term.kind == "repulsion"
    )
    component = system.repulsion_classical
    legacy_setup = component.get_cache(system.numbers, system.ihelp)
    parameter = par.get("element.H.arep")
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )

    core_positions = positions.clone().requires_grad_(True)
    core_energy = _evaluate_classical_descriptor(
        descriptor, core_positions, torch.tensor(0.0, dtype=torch.float64)
    ).sum()
    core_pos_grad, core_param_grad = torch.autograd.grad(
        core_energy, (core_positions, parameter), create_graph=True
    )
    core_mixed = torch.autograd.grad(core_pos_grad.sum(), parameter)[0]

    legacy_positions = positions.clone().requires_grad_(True)
    legacy_energy = component.get_energy(legacy_positions, legacy_setup).sum()
    legacy_pos_grad, legacy_param_grad = torch.autograd.grad(
        legacy_energy, (legacy_positions, parameter), create_graph=True
    )
    legacy_mixed = torch.autograd.grad(legacy_pos_grad.sum(), parameter)[0]

    torch.testing.assert_close(core_energy, legacy_energy)
    torch.testing.assert_close(core_pos_grad, legacy_pos_grad)
    torch.testing.assert_close(core_param_grad, legacy_param_grad)
    torch.testing.assert_close(core_mixed, legacy_mixed)
    assert torch.count_nonzero(core_param_grad) > 0

    direction = torch.ones_like(positions) * 0.1
    core_jvp = torch.func.jvp(
        lambda geometry: _evaluate_classical_descriptor(
            descriptor, geometry, torch.tensor(0.0, dtype=torch.float64)
        ).sum(),
        (positions,),
        (direction,),
    )[1]
    legacy_jvp = torch.func.jvp(
        lambda geometry: component.get_energy(geometry, legacy_setup).sum(),
        (positions,),
        (direction,),
    )[1]
    torch.testing.assert_close(core_jvp, legacy_jvp)


def test_c6_same_size_compositions_can_have_incompatible_dynamic_shapes() -> (
    None
):
    """Show same-size compositions with incompatible dynamic shell shapes."""
    h2 = _system(GFN2_XTB, labels.GFN2_XTB, [1, 1]).exact_core
    lih = _system(GFN2_XTB, labels.GFN2_XTB, [3, 1]).exact_core
    assert h2 is not None and lih is not None

    assert [
        (term.kind, term.label, term.position) for term in h2.classical_terms
    ] == [
        (term.kind, term.label, term.position) for term in lih.classical_terms
    ]
    assert [
        (term.kind, term.label, term.position) for term in h2.interaction_terms
    ] == [
        (term.kind, term.label, term.position) for term in lih.interaction_terms
    ]

    h2_es2 = h2.interaction_terms[0].setup
    lih_es2 = lih.interaction_terms[0].setup
    h2_shapes = {
        path: tuple(value.shape) for path, value in tensor_leaves(h2_es2)
    }
    lih_shapes = {
        path: tuple(value.shape) for path, value in tensor_leaves(lih_es2)
    }
    assert h2_shapes["shell_hubbard"] != lih_shapes["shell_hubbard"]
    assert h2_shapes["shells_to_atom"] != lih_shapes["shells_to_atom"]


def test_functional_interaction_set_uses_explicit_tensor_tree_adapter() -> None:
    """Check the per-call interaction tree and its explicit tensor walker."""
    system = _system(GFN2_XTB, labels.GFN2_XTB, [1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float64
    )
    call_data = _interaction_data(system, positions)
    interactions = prepare_functional_interactions(
        system.interactions,
        call_data,
        system.numbers,
        d4sc_setup=call_data.d4sc_setup,
        kind_by_label=call_data.functional_kinds,
    )

    leaves, spec = tree_flatten(interactions)
    assert leaves == [interactions]
    assert tree_unflatten(leaves, spec) is interactions
    assert tensor_leaves(interactions)


def test_post_setup_extension_label_mutation_is_not_a_supported_update() -> (
    None
):
    """The setup-time legacy cache key does not follow a mutated label."""

    class Extension(Classical):
        __slots__ = ("scale",)

        def __init__(self) -> None:
            super().__init__(device="cpu", dtype=torch.float64)
            self.label = "extension:old"
            self.scale = torch.tensor(0.25, dtype=torch.float64)

        def get_cache(self, numbers, ihelp=None, **kwargs):
            return self.scale

        def get_energy(self, positions, cache, **kwargs):
            return (
                torch.ones(positions.shape[-2], dtype=positions.dtype) * cache
            )

    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    extension = Extension()
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=Config.create(method=labels.GFN1_XTB, exclude="scf"),
        classical=(extension,),
    ).setup(numbers)
    extension.label = "extension:new"

    with pytest.raises(KeyError, match="extension:new"):
        system.singlepoint(positions)
