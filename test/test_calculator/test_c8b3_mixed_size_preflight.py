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
"""C8b.3 mixed-size structural padding preflight."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import ExactSystemCore, Model


def _core(numbers: tuple[int, ...]) -> ExactSystemCore:
    """Construct an independent production exact core."""
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    )
    system = model.setup(torch.tensor(numbers, dtype=torch.int64))
    assert system.exact_core is not None
    return system.exact_core


def _shape_inventory(core: ExactSystemCore) -> dict[str, object]:
    """Return the structural sizes that mixed-size padding must reconcile."""
    assert core.integral is not None
    backend = core.integral.backend
    assert backend is not None
    pair_plan = backend.pair_plan
    h0 = core.h0_setup
    assert h0 is not None
    repulsion = next(
        term.setup for term in core.classical_terms if term.kind == "repulsion"
    )
    d3 = next(term.setup for term in core.classical_terms if term.kind == "d3")
    es2 = next(
        term.setup for term in core.interaction_terms if term.kind == "es2"
    )
    return {
        "atoms": core.numbers.numel(),
        "shells": backend.index.shells_to_atom.numel(),
        "orbitals": backend.index.orbitals_to_shell.numel(),
        "ao_capacity": pair_plan.nao,
        "unique_shell_types": len(pair_plan.unique_angular),
        "pair_classes": len(pair_plan.classes),
        "pair_class_contexts": tuple(
            (item.ub, item.uk, item.lb, item.lk) for item in pair_plan.classes
        ),
        "pair_counts": tuple(item.ib.numel() for item in pair_plan.classes),
        "pair_index_shapes": tuple(
            tuple(item.idx.shape) for item in pair_plan.classes
        ),
        "alpha_shapes": tuple(tuple(item.shape) for item in backend.alphas),
        "h0_shell_matrix": tuple(h0.hscale.shape),
        "h0_unique_shell_matrix": tuple(h0.kpair.shape),
        "repulsion_atom_pairs": tuple(repulsion.arep.shape),
        "d3_atom_map": tuple(d3.setup_numbers.shape),
        "es2_atom_mask": tuple(es2.atom_pair_mask.shape),
        "es2_shell_mask": tuple(es2.shell_pair_mask.shape),
    }


@pytest.mark.parametrize(
    ("first_numbers", "second_numbers", "first_expected", "second_expected"),
    [
        (
            (1, 1),
            (1, 1, 1),
            (2, 4, 4, 4, 2, 4),
            (3, 6, 6, 6, 2, 4),
        ),
        (
            (1, 1),
            (8, 1, 1),
            (2, 4, 4, 4, 2, 4),
            (3, 6, 8, 8, 4, 11),
        ),
    ],
    ids=["H2-H3", "H2-H2O"],
)
def test_mixed_size_cores_require_role_aware_padding(
    first_numbers: tuple[int, ...],
    second_numbers: tuple[int, ...],
    first_expected: tuple[int, ...],
    second_expected: tuple[int, ...],
) -> None:
    """Record production-tree dimensions and reject unpadded stacking."""
    first = _core(first_numbers)
    second = _core(second_numbers)
    assert first is not second
    assert first.config == second.config
    assert first.integral is not None and second.integral is not None
    assert first.integral.backend is not second.integral.backend

    first_inventory = _shape_inventory(first)
    second_inventory = _shape_inventory(second)
    first_dims = tuple(
        first_inventory[key]
        for key in (
            "atoms",
            "shells",
            "orbitals",
            "ao_capacity",
            "unique_shell_types",
            "pair_classes",
        )
    )
    second_dims = tuple(
        second_inventory[key]
        for key in (
            "atoms",
            "shells",
            "orbitals",
            "ao_capacity",
            "unique_shell_types",
            "pair_classes",
        )
    )
    assert first_dims == first_expected
    assert second_dims == second_expected

    first_leaves, first_spec = tree_flatten(first)
    second_leaves, second_spec = tree_flatten(second)
    assert first_leaves and second_leaves
    assert first_spec != second_spec
    with pytest.raises((ValueError, RuntimeError)):
        stack((first, second))

    # H2/H3 share pair-class metadata, but pair entry counts, AO offsets,
    # and atom-indexed setup leaves differ. H2/H2O additionally needs a
    # canonical unique-shell type and pair-class map.
    assert first_inventory["pair_class_contexts"] != ()
    assert second_inventory["pair_class_contexts"] != ()
    assert first_inventory["pair_counts"] != second_inventory["pair_counts"]
    assert (
        first_inventory["repulsion_atom_pairs"]
        != second_inventory["repulsion_atom_pairs"]
    )
    assert (
        first_inventory["es2_shell_mask"] != second_inventory["es2_shell_mask"]
    )
