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

from dataclasses import replace as dataclass_replace

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
    compatibility_signature,
    pad_exact_core,
    pad_positions,
)
from dxtb._src.calculators.singlepoint import _evaluate_classical_descriptor
from dxtb._src.components.classicals.dispersion.d3 import (
    DispersionD3CoreSetup,
)
from dxtb._src.components.interactions.coulomb.average import arithmetic_average
from dxtb._src.components.interactions.coulomb.thirdorder import (
    ES3CoreSetup,
    core_es3_setup,
    setup_es3,
)
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


def _shell_resolved_es3_core(
    numbers: tuple[int, ...],
) -> tuple[Model, ExactSystemCore]:
    """Create an exact core with ES3 setup from the production resolver."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    model = Model(
        par=par,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    )
    system = model.setup(torch.tensor(numbers, dtype=torch.int64))
    assert system.exact_core is not None and system.h0_setup is not None
    gam3 = par.get_elem_param(system.h0_setup.unique, "gam3")
    shell_scale = torch.ones_like(
        system.h0_setup.index.unique_angular, dtype=gam3.dtype
    )
    resolved = core_es3_setup(
        setup_es3(gam3, system.ihelp, shell_scale=shell_scale)
    )
    terms = tuple(
        term.replace(setup=resolved) if term.kind == "es3" else term
        for term in system.exact_core.interaction_terms
    )
    return model, system.exact_core.replace(interaction_terms=terms)


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


def _realize_buckets(
    cores: tuple[ExactSystemCore, ...],
) -> tuple[frozenset[tuple[int, ...]], ...]:
    """Pad and stack every bucket returned by the partitioner."""
    buckets = bucket_exact_cores(cores)
    recovered: list[int] = []
    memberships = []
    for bucket in buckets:
        assert bucket.template_index in bucket.indices
        template = cores[bucket.template_index]
        padded = tuple(
            pad_exact_core(cores[index], bucket, template)
            for index in bucket.indices
        )
        stacked = stack(padded)
        assert isinstance(stacked, PaddedExactSystemCore)
        assert stacked.core.numbers.shape[1] == bucket.atoms
        assert stacked.core.integral is not None
        assert stacked.core.integral.backend is not None
        assert stacked.core.integral.backend.pair_plan.nao == bucket.orbitals
        assert (
            stacked.core.integral.backend.index.shells_to_atom.shape[-1]
            == bucket.shells
        )
        assert (
            stacked.core.integral.backend.index.orbitals_to_shell.shape[-1]
            == bucket.orbitals
        )
        assert (
            tuple(
                pair.ib.shape[-1]
                for pair in stacked.core.integral.backend.pair_plan.classes
            )
            == bucket.pair_entries
        )
        assert (
            0
            if stacked.core.h0_setup is None
            else stacked.core.h0_setup.unique.shape[-1]
        ) == bucket.unique_elements
        assert (
            len(stacked.core.integral.backend.pair_plan.unique_angular)
            == bucket.unique_shell_types
        )
        recovered.extend(bucket.indices)
        memberships.append(
            frozenset(
                tuple(int(value) for value in cores[i].numbers)
                for i in bucket.indices
            )
        )
    assert sorted(recovered) == list(range(len(cores)))
    return tuple(memberships)


def test_buckets_separate_unrealizable_c2_n3_layouts_independent_of_order() -> (
    None
):
    """Buckets only promise layouts that their representative can realize."""
    c2, n3, c3, n2 = tuple(
        _system(numbers)[1]
        for numbers in ((6, 6), (7, 7, 7), (6, 6, 6), (7, 7))
    )
    assert compatibility_signature(c2) == compatibility_signature(n3)
    forward = (c2, n3)
    reverse = (n3, c2)
    forward_groups = _realize_buckets(forward)
    reverse_groups = _realize_buckets(reverse)
    assert (
        set(forward_groups)
        == set(reverse_groups)
        == {
            frozenset({(6, 6)}),
            frozenset({(7, 7, 7)}),
        }
    )

    mixed = (c2, c3, n2, n3)
    reverse_mixed = tuple(reversed(mixed))
    mixed_groups = _realize_buckets(mixed)
    reverse_groups = _realize_buckets(reverse_mixed)
    assert set(mixed_groups) == set(reverse_groups)
    assert set().union(*mixed_groups) == {
        (6, 6),
        (6, 6, 6),
        (7, 7),
        (7, 7, 7),
    }


def test_es3_atom_and_shell_resolved_dimensions_pad_correctly() -> None:
    """ES3 follows its registered atom- or shell-resolved extent."""
    _, atom_h2 = _system((1, 1))
    _, atom_h3 = _system((1, 1, 1))
    atom_bucket = bucket_exact_cores((atom_h2, atom_h3))[0]
    atom_setup = next(
        term.setup for term in atom_h2.interaction_terms if term.kind == "es3"
    )
    padded_atom = pad_exact_core(atom_h2, atom_bucket, atom_h3)
    padded_atom_setup = next(
        term.setup
        for term in padded_atom.core.interaction_terms
        if term.kind == "es3"
    )
    assert isinstance(atom_setup, ES3CoreSetup)
    assert atom_setup.shell_resolved is False
    assert atom_setup.hubbard_derivs.shape == (2,)
    assert padded_atom_setup.hubbard_derivs.shape == (3,)
    torch.testing.assert_close(
        padded_atom_setup.hubbard_derivs[:2], atom_setup.hubbard_derivs
    )

    _, shell_h2 = _shell_resolved_es3_core((1, 1))
    _, shell_h3 = _shell_resolved_es3_core((1, 1, 1))
    shell_bucket = bucket_exact_cores((shell_h2, shell_h3))[0]
    source = next(
        term.setup for term in shell_h2.interaction_terms if term.kind == "es3"
    )
    assert isinstance(source, ES3CoreSetup)
    assert source.shell_resolved is True
    assert source.hubbard_derivs.shape == (4,)
    source_identity = id(source.hubbard_derivs)
    source_values = source.hubbard_derivs.clone()
    padded_shell = pad_exact_core(shell_h2, shell_bucket, shell_h3)
    actual = next(
        term.setup
        for term in padded_shell.core.interaction_terms
        if term.kind == "es3"
    )
    assert actual.hubbard_derivs.shape == (6,)
    torch.testing.assert_close(actual.hubbard_derivs[:4], source_values)
    assert id(source.hubbard_derivs) == source_identity
    torch.testing.assert_close(source.hubbard_derivs, source_values)

    padded_shell_h3 = pad_exact_core(shell_h3, shell_bucket, shell_h3)
    stacked_shell = stack((padded_shell, padded_shell_h3))
    assert isinstance(stacked_shell, PaddedExactSystemCore)
    assert type(stacked_shell.core) is type(shell_h2)
    assert stacked_shell.core.integral is not None
    assert stacked_shell.core.integral.backend is not None
    assert stacked_shell.core.h0_setup is not None
    assert any(
        term.kind == "es3" for term in stacked_shell.core.interaction_terms
    )
    stacked_es3 = next(
        term.setup
        for term in stacked_shell.core.interaction_terms
        if term.kind == "es3"
    )
    assert isinstance(stacked_es3, ES3CoreSetup)
    assert stacked_es3.hubbard_derivs.shape == (2, 6)
    torch.testing.assert_close(stacked_es3.hubbard_derivs[0, :4], source_values)
    torch.testing.assert_close(
        stacked_es3.hubbard_derivs[0, 4:],
        torch.zeros(2, dtype=source_values.dtype),
    )
    assert stacked_shell.validity.shells.shape == (2, 6)
    assert stacked_shell.validity.shells[0].tolist() == [True] * 4 + [False] * 2
    assert stacked_shell.validity.shells[1].all()
    torch.testing.assert_close(source.hubbard_derivs, source_values)

    malformed = source.replace(hubbard_derivs=source.hubbard_derivs[:-1])
    bad_core = shell_h2.replace(
        interaction_terms=tuple(
            term.replace(setup=malformed) if term.kind == "es3" else term
            for term in shell_h2.interaction_terms
        )
    )
    with pytest.raises(ValueError, match="shell-resolved"):
        bucket_exact_cores((bad_core, shell_h3))


def test_bucket_signature_tracks_retained_dynamic_leaf_shapes() -> None:
    """Only dimensions normalized by padding may vary within a bucket."""
    _, core = _system((1, 1))
    term = next(item for item in core.classical_terms if item.kind == "d3")
    setup = term.setup
    assert isinstance(setup, DispersionD3CoreSetup)
    assert tuple(setup.reference_cn.shape) == (104, 7)
    assert tuple(setup.reference_c6.shape) == (104, 104, 7, 7)

    malformed_setup = setup.replace(
        reference_cn=torch.cat((setup.reference_cn, setup.reference_cn[:1]))
    )
    malformed = core.replace(
        classical_terms=tuple(
            item.replace(setup=malformed_setup) if item.kind == "d3" else item
            for item in core.classical_terms
        )
    )
    assert malformed.numbers.shape == core.numbers.shape
    with pytest.raises(ValueError, match="D3 reference_cn and reference_c6"):
        compatibility_signature(malformed)
    with pytest.raises(ValueError, match="D3 reference_cn and reference_c6"):
        bucket_exact_cores((core, malformed))

    # A shape-consistent alternate table is not malformed, but it is not a
    # stack-compatible dynamic state because the adapter leaves it unchanged.
    alternate_setup = setup.replace(
        reference_cn=setup.reference_cn[:, :-1],
        reference_c6=setup.reference_c6[:, :, :-1, :-1],
    )
    alternate = core.replace(
        classical_terms=tuple(
            item.replace(setup=alternate_setup) if item.kind == "d3" else item
            for item in core.classical_terms
        )
    )
    assert compatibility_signature(core) != compatibility_signature(alternate)
    alternate_buckets = bucket_exact_cores((core, alternate))
    assert tuple(bucket.indices for bucket in alternate_buckets) == ((0,), (1,))
    for bucket, value in zip(alternate_buckets, (core, alternate)):
        assert bucket.template_index == bucket.indices[0]
        padded = pad_exact_core(value, bucket, value)
        stacked = stack((padded,))
        assert isinstance(stacked, PaddedExactSystemCore)


def test_descriptor_static_metadata_and_callable_policy_are_in_signature() -> (
    None
):
    """Static descriptor context is keyed and custom closures are rejected."""
    _, core = _system((1, 1))
    term = core.classical_terms[0]
    altered_term = term.replace(charge_dependent=not term.charge_dependent)
    altered = core.replace(
        classical_terms=tuple(
            altered_term if item is term else item
            for item in core.classical_terms
        )
    )
    assert compatibility_signature(core) != compatibility_signature(altered)
    assert tree_flatten(core)[1] != tree_flatten(altered)[1]

    def make_average(scale: float):
        def average(hubbard: torch.Tensor) -> torch.Tensor:
            return scale * hubbard.unsqueeze(-1) * hubbard.unsqueeze(-2)

        return average

    average_a = make_average(1.0)
    average_b = make_average(2.0)
    assert average_a.__module__ == average_b.__module__
    assert average_a.__qualname__ == average_b.__qualname__
    assert average_a is not average_b
    probe = torch.tensor([2.0], dtype=torch.float64)
    assert not torch.equal(average_a(probe), average_b(probe))
    es2 = next(
        item.setup for item in core.interaction_terms if item.kind == "es2"
    )
    core_a = core.replace(
        interaction_terms=tuple(
            (
                item.replace(setup=es2.replace(average_context=average_a))
                if item.kind == "es2"
                else item
            )
            for item in core.interaction_terms
        )
    )
    core_b = core.replace(
        interaction_terms=tuple(
            (
                item.replace(setup=es2.replace(average_context=average_b))
                if item.kind == "es2"
                else item
            )
            for item in core.interaction_terms
        )
    )
    with pytest.raises(TypeError, match="custom or stateful callable"):
        compatibility_signature(core_a)
    with pytest.raises(TypeError, match="custom or stateful callable"):
        bucket_exact_cores((core_a, core_b))

    built_in_average = es2.replace(average_context=arithmetic_average)
    alternate_average_core = core.replace(
        interaction_terms=tuple(
            item.replace(setup=built_in_average) if item.kind == "es2" else item
            for item in core.interaction_terms
        )
    )
    assert compatibility_signature(core) != compatibility_signature(
        alternate_average_core
    )
    assert tuple(
        bucket.indices
        for bucket in bucket_exact_cores((core, alternate_average_core))
    ) == ((0,), (1,))

    assert core.integral is not None and core.integral.backend is not None
    alternate_algorithm = core.integral.backend.replace(
        algorithm=f"{core.integral.backend.algorithm}-other"
    )
    alternate_backend_core = core.replace(
        integral=core.integral.replace(backend=alternate_algorithm)
    )
    assert compatibility_signature(core) != compatibility_signature(
        alternate_backend_core
    )
    assert tuple(
        bucket.indices
        for bucket in bucket_exact_cores((core, alternate_backend_core))
    ) == ((0,), (1,))
    alternate_method_core = core.replace(
        config=dataclass_replace(core.config, method=labels.GFN2_XTB)
    )
    with pytest.raises(TypeError, match="GFN1 cores only"):
        compatibility_signature(alternate_method_core)


def test_pad_positions_requires_prefix_validity_and_preserves_gradients() -> (
    None
):
    """Position rows map only to a prefix of the padded atom dimension."""
    _, h2 = _system((1, 1))
    _, h3 = _system((1, 1, 1))
    bucket = bucket_exact_cores((h2, h3))[0]
    padded = pad_exact_core(h2, bucket, h3)
    positions = _coordinates(2)
    output = pad_positions(positions, padded.validity)
    assert output.shape == (3, 3)
    torch.testing.assert_close(output[:2], positions)
    torch.testing.assert_close(output[2], torch.zeros(3, dtype=positions.dtype))
    assert pad_positions(
        _coordinates(3), pad_exact_core(h3, bucket, h3).validity
    ).shape == (3, 3)

    loss = output.square().sum()
    gradient = torch.autograd.grad(loss, positions)[0]
    torch.testing.assert_close(gradient, 2 * positions)
    malformed = padded.validity.replace(atoms=torch.tensor([True, False, True]))
    with pytest.raises(ValueError, match="prefix atom-validity"):
        pad_positions(positions, malformed)
    with pytest.raises(ValueError, match="Position count"):
        pad_positions(_coordinates(1), padded.validity)


def test_padded_core_remains_outside_ordinary_system_dispatch() -> None:
    """The padding adapter is not an ordinary System or exact core."""
    _, h2 = _system((1, 1))
    _, h3 = _system((1, 1, 1))
    bucket = bucket_exact_cores((h2, h3))[0]
    padded = pad_exact_core(h2, bucket, h3)
    assert isinstance(padded, PaddedExactSystemCore)
    assert not isinstance(padded, ExactSystemCore)
    assert not hasattr(padded, "singlepoint")
