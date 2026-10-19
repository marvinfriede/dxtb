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

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model, System
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.classicals import Classical
from dxtb._src.scf.functional.core import tensor_leaves
from dxtb._src.scf.functional.interactions import (
    prepare_functional_interactions,
)


def _system(method: str, label: int, numbers: list[int]) -> System:
    return Model(
        par=ParamModule(method, dtype=torch.float64),
        config=Config.create(method=label),
    ).setup(torch.tensor(numbers))


def test_c6_descriptor_nested_setup_is_opaque_to_standard_pytree() -> None:
    """The current dataclass descriptor hides its registered setup Node."""
    system = _system(GFN2_XTB, labels.GFN2_XTB, [1, 1])
    assert system.exact_core is not None
    core = system.exact_core
    descriptor = system.exact_core.classical_terms[0]
    setup = descriptor.setup

    for term in core.classical_terms:
        component = system.classicals.components[term.position]
        assert term.setup is system.classical_setup(component)
    for term in core.interaction_terms:
        component = system.interactions.components[term.position]
        assert term.setup is system.interaction_setup(component)

    core_leaves, core_spec = tree_flatten(core)
    assert core_leaves == [core]
    assert tree_unflatten(core_leaves, core_spec) is core

    descriptor_leaves, descriptor_spec = tree_flatten(descriptor)
    assert descriptor_leaves == [descriptor]
    assert tree_unflatten(descriptor_leaves, descriptor_spec) is descriptor

    interaction = core.interaction_terms[0]
    interaction_leaves, interaction_spec = tree_flatten(interaction)
    assert interaction_leaves == [interaction]
    assert tree_unflatten(interaction_leaves, interaction_spec) is interaction
    interaction_tensor_leaves, _ = tree_flatten(interaction.setup)
    assert interaction_tensor_leaves
    assert all(
        isinstance(leaf, torch.Tensor) for leaf in interaction_tensor_leaves
    )

    setup_leaves, setup_spec = tree_flatten(setup)
    assert setup_leaves
    assert all(isinstance(leaf, torch.Tensor) for leaf in setup_leaves)
    rebuilt_setup = tree_unflatten(setup_leaves, setup_spec)
    rebuilt_leaves, rebuilt_spec = tree_flatten(rebuilt_setup)
    assert rebuilt_setup is not setup
    assert rebuilt_spec == setup_spec
    assert all(
        left is right for left, right in zip(rebuilt_leaves, setup_leaves)
    )

    # The accepted implicit tensor-tree adapter recurses through dataclasses,
    # so it sees the same setup tensors even though standard pytree does not.
    nested = tensor_leaves(descriptor)
    assert nested
    assert all(path.startswith("setup.") for path, _ in nested)
    assert {path[len("setup.") :] for path, _ in nested} == {
        path for path, _ in tensor_leaves(setup)
    }
    setup_values = {id(value) for _, value in tensor_leaves(setup)}
    assert {id(value) for _, value in nested} == setup_values


def test_c6_system_descriptor_tree_is_not_ready_for_standard_stack() -> None:
    """Different exact method compositions have different opaque trees."""
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
    assert len(gfn1_leaves) == len(gfn1.classical_terms)
    assert len(gfn2_leaves) == len(gfn2.classical_terms)
    assert all(not isinstance(leaf, torch.Tensor) for leaf in gfn1_leaves)
    assert all(not isinstance(leaf, torch.Tensor) for leaf in gfn2_leaves)


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
