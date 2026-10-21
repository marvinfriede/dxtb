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
"""Executable C8b.3a mixed-size ExactSystemCore padding controls."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import ExactSystemCore, Model
from dxtb._src.calculators.padding import (
    PaddedExactSystemCore,
    bucket_exact_cores,
    pad_exact_core,
    pad_positions,
)
from dxtb._src.calculators.singlepoint import _evaluate_classical_descriptor
from dxtb._src.integral.driver.pytorch.impls.kernels import get_kernel
from dxtb._src.integral.driver.pytorch.impls.pairs import assemble_matrix


def _system(
    numbers: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float64,
    differentiable_repulsion: bool = False,
) -> tuple[Model, ExactSystemCore]:
    """Build an independent GFN1 Model/System with the PyTorch backend."""
    par = ParamModule(GFN1_XTB, dtype=dtype)
    if differentiable_repulsion:
        par.set_differentiable("element", "H", "arep")
    model = Model(
        par=par,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    )
    system = model.setup(torch.tensor(numbers, dtype=torch.int64))
    assert system.exact_core is not None
    return model, system.exact_core


def _repulsion(core: ExactSystemCore, positions: torch.Tensor) -> torch.Tensor:
    descriptor = next(
        term for term in core.classical_terms if term.kind == "repulsion"
    )
    return _evaluate_classical_descriptor(
        descriptor, positions, positions.new_zeros(())
    ).sum()


def _repulsion_atomwise(
    core: ExactSystemCore, positions: torch.Tensor
) -> torch.Tensor:
    descriptor = next(
        term for term in core.classical_terms if term.kind == "repulsion"
    )
    return _evaluate_classical_descriptor(
        descriptor, positions, positions.new_zeros(())
    )


def _coordinates(natoms: int) -> torch.Tensor:
    base = torch.tensor(
        [[0.1, 0.2, 0.3], [0.1, 0.2, 1.6], [1.4, 0.0, 2.5]],
        dtype=torch.float64,
    )
    return base[:natoms].clone().requires_grad_()


def test_h2_h3_complete_cores_pad_and_stack_with_validity() -> None:
    """The full selected-backend/H0/descriptor trees stack after padding."""
    model_h2, h2 = _system((1, 1))
    _, h3 = _system((1, 1, 1))
    source_leaves, _ = tree_flatten(h2)
    source_values = [leaf.clone() for leaf in source_leaves]
    source_ids = [id(leaf) for leaf in source_leaves]
    parameter_ids = [id(value) for value in model_h2.par.parameters()]
    parameter_values = [value.clone() for value in model_h2.par.parameters()]
    bucket = bucket_exact_cores((h2, h3))[0]
    assert bucket.indices == (0, 1)
    assert (bucket.atoms, bucket.shells, bucket.orbitals) == (3, 6, 6)
    assert (bucket.unique_elements, bucket.unique_shell_types) == (1, 2)
    assert bucket.pair_entries == (6, 3, 6, 6)

    padded_h2 = pad_exact_core(h2, bucket, h3)
    padded_h3 = pad_exact_core(h3, bucket, h3)
    stacked = stack((padded_h2, padded_h3))
    assert isinstance(stacked, PaddedExactSystemCore)
    assert stacked.core.numbers.shape == (2, 3)
    assert stacked.core.integral is not None
    assert stacked.core.integral.backend is not None
    assert stacked.core.integral.backend.pair_plan.nao == 6
    assert stacked.core.h0_setup is not None
    assert tuple(t.kind for t in stacked.core.classical_terms) == tuple(
        t.kind for t in h3.classical_terms
    )
    assert tuple(t.kind for t in stacked.core.interaction_terms) == tuple(
        t.kind for t in h3.interaction_terms
    )
    assert stacked.core.index is stacked.core.integral.backend.index
    torch.testing.assert_close(
        stacked.core.integral.backend.pair_plan.atom,
        stacked.core.integral.backend.index.shells_to_atom,
    )
    leaves, tree_spec = tree_flatten(stacked)
    assert leaves and all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    restored = tree_unflatten(leaves, tree_spec)
    assert isinstance(restored, PaddedExactSystemCore)
    torch.testing.assert_close(restored.validity.atoms, stacked.validity.atoms)
    torch.testing.assert_close(
        restored.core.integral.backend.index.shells_to_atom,
        stacked.core.integral.backend.index.shells_to_atom,
    )

    torch.testing.assert_close(
        padded_h2.core.integral.backend.index.shells_to_atom[:4],
        h2.integral.backend.index.shells_to_atom,
    )
    torch.testing.assert_close(
        padded_h2.core.integral.backend.index.orbitals_to_shell[:4],
        h2.integral.backend.index.orbitals_to_shell,
    )
    torch.testing.assert_close(
        padded_h2.core.h0_setup.index.atom_to_unique[:2],
        h2.h0_setup.index.atom_to_unique,
    )
    assert padded_h2.validity.atoms.tolist() == [True, True, False]
    assert padded_h2.validity.shells.tolist() == [True] * 4 + [False] * 2
    assert padded_h2.validity.orbitals.tolist() == [True] * 4 + [False] * 2
    assert padded_h2.validity.unique_elements.tolist() == [True]
    assert padded_h2.validity.unique_shell_types.tolist() == [True, True]
    assert all(
        mask.shape == (n,)
        for mask, n in zip(padded_h2.validity.pair_entries, bucket.pair_entries)
    )

    # Source values and setup objects remain untouched by padding.
    assert h2.numbers.shape == (2,)
    assert h2.integral.backend.pair_plan.nao == 4
    assert h2.integral.backend.index.shells_to_atom.shape == (4,)
    assert h2.classical_terms[-1].setup.arep.shape == (2, 2)
    after_leaves, _ = tree_flatten(h2)
    assert [id(leaf) for leaf in after_leaves] == source_ids
    for actual, expected in zip(after_leaves, source_values):
        torch.testing.assert_close(actual, expected)
    assert [id(value) for value in model_h2.par.parameters()] == parameter_ids
    for actual, expected in zip(model_h2.par.parameters(), parameter_values):
        torch.testing.assert_close(actual, expected)

    converted = stacked.to(dtype=torch.float32)
    assert converted.core.numbers.dtype == torch.int64
    assert converted.validity.atoms.dtype == torch.bool
    assert (
        converted.core.integral.backend.index.shells_to_atom.dtype
        == torch.int64
    )
    assert converted.core.classical_terms[-1].setup.arep.dtype == torch.float32


def test_pair_plan_reindexes_ao_stride_without_scatter_collisions() -> None:
    """H2 real AO entries survive a padded stride and overlap assembly."""
    _, h2 = _system((1, 1))
    _, h3 = _system((1, 1, 1))
    bucket = bucket_exact_cores((h2, h3))[0]
    padded = pad_exact_core(h2, bucket, h3)
    old_plan = h2.integral.backend.pair_plan
    new_plan = padded.core.integral.backend.pair_plan
    assert old_plan.nao == 4 and new_plan.nao == 6

    real_targets = []
    invalid_targets = []
    for old, new, valid in zip(
        old_plan.classes, new_plan.classes, padded.validity.pair_entries
    ):
        assert torch.equal(old.ib, new.ib[valid])
        assert torch.equal(old.jk, new.jk[valid])
        old_rows, old_cols = old.idx // old_plan.nao, old.idx % old_plan.nao
        expected = old_rows * new_plan.nao + old_cols
        assert torch.equal(expected, new.idx[valid])
        torch.testing.assert_close(old.weight, new.weight[valid])
        real_targets.append(new.idx[valid].reshape(-1))
        invalid_targets.append(new.idx[~valid].reshape(-1))

    real_flat = torch.cat(real_targets)
    invalid_flat = torch.cat(invalid_targets)
    assert torch.unique(real_flat).numel() == real_flat.numel()
    assert torch.unique(invalid_flat).numel() == invalid_flat.numel()
    assert not torch.isin(invalid_flat, real_flat).any()
    assert (
        (invalid_flat // new_plan.nao >= old_plan.nao)
        | (invalid_flat % new_plan.nao >= old_plan.nao)
    ).all()

    positions = _coordinates(2)
    padded_positions = pad_positions(positions, padded.validity)
    ref = assemble_matrix(
        get_kernel(h2.integral.backend.algorithm),
        None,
        list(h2.integral.backend.alphas),
        list(h2.integral.backend.coeffs),
        positions,
        plan=old_plan,
    )[0]
    actual = assemble_matrix(
        get_kernel(padded.core.integral.backend.algorithm),
        None,
        list(padded.core.integral.backend.alphas),
        list(padded.core.integral.backend.coeffs),
        padded_positions,
        plan=new_plan,
    )[0]
    torch.testing.assert_close(actual[:4, :4], ref, atol=0.0, rtol=0.0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("padding_kind", ["coincident", "zero"])
def test_mapped_repulsion_energy_and_position_gradients(
    padding_kind: str,
) -> None:
    """Mapped complete-core Repulsion matches serial values and gradients."""
    model_h2, h2 = _system((1, 1), differentiable_repulsion=True)
    _, h3 = _system((1, 1, 1))
    bucket = bucket_exact_cores((h2, h3))[0]
    padded = (
        pad_exact_core(h2, bucket, h3),
        pad_exact_core(h3, bucket, h3),
    )
    stacked = stack(padded)
    p2, p3 = _coordinates(2), _coordinates(3)
    p2_padded = pad_positions(p2, padded[0].validity)
    if padding_kind == "coincident":
        p2_padded = torch.cat((p2, p2[:1]), dim=0)
    else:
        p2_padded = torch.cat((p2, p2.new_zeros((1, 3))), dim=0)
    stacked_positions = torch.stack((p2_padded, p3))

    def mapped_energy(
        core: PaddedExactSystemCore, pos: torch.Tensor
    ) -> torch.Tensor:
        return _repulsion(core.core, pos)

    mapped = torch.func.vmap(mapped_energy)(stacked, stacked_positions)
    serial = torch.stack((_repulsion(h2, p2), _repulsion(h3, p3)))
    torch.testing.assert_close(mapped, serial, atol=1.0e-13, rtol=1.0e-13)
    assert torch.isfinite(mapped).all()
    atomwise_h2 = _repulsion_atomwise(padded[0].core, p2_padded)
    torch.testing.assert_close(atomwise_h2[2], torch.zeros((), dtype=p2.dtype))

    mapped_gradient = torch.func.vmap(
        torch.func.grad(mapped_energy, argnums=1)
    )(stacked, stacked_positions)
    serial_gradient_h2 = torch.autograd.grad(
        _repulsion(h2, p2), p2, retain_graph=True
    )[0]
    serial_gradient_h3 = torch.autograd.grad(_repulsion(h3, p3), p3)[0]
    torch.testing.assert_close(
        mapped_gradient[0, :2], serial_gradient_h2, atol=1.0e-13, rtol=1.0e-13
    )
    torch.testing.assert_close(
        mapped_gradient[1], serial_gradient_h3, atol=1.0e-13, rtol=1.0e-13
    )
    torch.testing.assert_close(
        mapped_gradient[0, 2], torch.zeros(3, dtype=p2.dtype)
    )
    assert torch.isfinite(mapped_gradient).all()

    # Parameter-derived setup tensors remain connected through padding.
    arep = dict(model_h2.par.named_parameters())[
        "parameter_tree.element.H.arep.param"
    ]
    padded_grad = torch.autograd.grad(
        _repulsion(padded[0].core, p2_padded), arep, retain_graph=True
    )[0]
    reference_grad = torch.autograd.grad(_repulsion(h2, p2), arep)[0]
    torch.testing.assert_close(
        padded_grad, reference_grad, atol=1.0e-13, rtol=1.0e-13
    )


def test_buckets_are_stable_and_keep_incompatible_structures() -> None:
    """Composition-compatible size buckets preserve every input position."""
    cores = tuple(
        _system(numbers)[1]
        for numbers in (
            (1, 1),
            (1, 1, 1),
            (6, 6),
            (7, 7),
            (8, 1, 1),
        )
    )
    first = bucket_exact_cores(cores)
    second = bucket_exact_cores(cores)
    assert [(b.indices, b.atoms, b.shells, b.orbitals) for b in first] == [
        (b.indices, b.atoms, b.shells, b.orbitals) for b in second
    ]
    assert first[0].indices == (0, 1)
    assert all(
        4 not in bucket.indices for bucket in first if 0 in bucket.indices
    )
    assert sorted(i for bucket in first for i in bucket.indices) == list(
        range(5)
    )
    assert any(bucket.indices == (2, 3) for bucket in first)
    cn_bucket = next(bucket for bucket in first if bucket.indices == (2, 3))
    cn_padded = tuple(
        pad_exact_core(cores[i], cn_bucket, cores[2]) for i in cn_bucket.indices
    )
    cn_stacked = stack(cn_padded)
    assert cn_stacked.core.numbers.shape == (2, 2)
    assert all(mask.all() for mask in cn_stacked.validity.pair_entries)

    dtype_cores = (_system((1, 1))[1], _system((1, 1), dtype=torch.float32)[1])
    dtype_buckets = bucket_exact_cores(dtype_cores)
    assert tuple(bucket.indices for bucket in dtype_buckets) == ((0,), (1,))
