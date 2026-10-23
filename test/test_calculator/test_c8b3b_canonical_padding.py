# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

"""C8b.3b different-element canonical exact-core padding controls."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, IndexHelper, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import ExactSystemCore, Model
from dxtb._src.calculators.singlepoint import _evaluate_classical_descriptor
from dxtb._src.components.interactions.coulomb.secondorder import (
    core_es2_setup,
    new_es2,
    setup_es2,
)
from dxtb._src.integral.driver.pytorch.impls.kernels import get_kernel
from dxtb._src.integral.driver.pytorch.impls.pairs import assemble_matrix


def _system(
    numbers: tuple[int, ...],
    *,
    differentiable_repulsion: bool = False,
) -> tuple[Model, ExactSystemCore]:
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
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


def _coordinates(numbers: tuple[int, ...]) -> torch.Tensor:
    # Deliberately non-degenerate, with deterministic geometry per atom count.
    coords = {
        (1, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        (8, 1, 1): [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.6],
            [-0.8, 0.0, 0.6],
        ],
        (1, 1, 8): [
            [0.8, 0.0, 0.6],
            [-0.8, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ],
    }
    return torch.tensor(
        coords[numbers], dtype=torch.float64, requires_grad=True
    )


def _repulsion(core: ExactSystemCore, positions: torch.Tensor) -> torch.Tensor:
    descriptor = next(
        term for term in core.classical_terms if term.kind == "repulsion"
    )
    return _evaluate_classical_descriptor(
        descriptor, positions, positions.new_zeros(())
    ).sum()


def _with_es2_resolution(
    model: Model, core: ExactSystemCore, shell_resolved: bool
) -> ExactSystemCore:
    """Install a production-derived ES2 setup at the requested resolution."""
    es2 = new_es2(
        torch.unique(core.numbers),
        model.par,
        shell_resolved=shell_resolved,
        dtype=core.effective_dtype,
        device=core.numbers.device,
    )
    assert es2 is not None
    ihelp = IndexHelper.from_numbers(core.numbers, GFN1_XTB)
    legacy = setup_es2(
        core.numbers,
        es2.hubbard,
        ihelp,
        lhubbard=es2.lhubbard,
        gexp=es2.gexp,
        average=es2.average,
        shell_resolved=shell_resolved,
    )
    setup = core_es2_setup(legacy)
    terms = tuple(
        term.replace(setup=setup) if term.kind.lower() == "es2" else term
        for term in core.interaction_terms
    )
    assert any(term.setup is setup for term in terms)
    return core.replace(interaction_terms=terms)


@pytest.mark.parametrize(
    "numbers",
    [((1, 1), (8, 1, 1)), ((1, 1), (1, 1, 8))],
    ids=["oxygen-first", "hydrogen-first"],
)
def test_h2_h2o_canonical_complete_core_stack(numbers) -> None:
    """Different element and atom counts map into one complete core tree."""
    model_h2, h2 = _system(numbers[0], differentiable_repulsion=True)
    model_water, water = _system(numbers[1])
    source_snapshots = []
    for source in (h2, water):
        source_leaves, _ = tree_flatten(source)
        source_snapshots.append(
            (
                tuple(id(value) for value in source_leaves),
                tuple(value.clone() for value in source_leaves),
            )
        )
    parameter_snapshots = []
    for model in (model_h2, model_water):
        parameters = tuple(model.par.parameters())
        parameter_snapshots.append(
            (
                tuple(id(value) for value in parameters),
                tuple(value.clone() for value in parameters),
            )
        )
    layout = bucket_canonical_exact_cores((h2, water))
    assert layout.elements == (1, 8)
    assert layout.atoms == 3
    assert layout.shell_types == ((1, 0, 0), (1, 1, 0), (8, 0, 0), (8, 1, 1))

    padded_h2 = pad_canonical_exact_core(h2, layout, 0)
    padded_water = pad_canonical_exact_core(water, layout, 1)
    stacked = stack((padded_h2, padded_water))
    assert stacked.core.numbers.shape == (2, 3)
    assert stacked.core.integral is not None
    assert stacked.core.h0_setup is not None
    assert stacked.core.index is stacked.core.integral.backend.index
    assert stacked.core.integral.backend.pair_plan.nao == layout.orbitals
    assert padded_h2.core.h0_setup.unique.tolist() == [1, 8]
    assert stacked.core.h0_setup.unique.tolist() == [[1, 8], [1, 8]]
    assert stacked.validity.atoms.shape == (2, layout.atoms)
    stacked_leaves, stacked_spec = tree_flatten(stacked)
    reconstructed = tree_unflatten(stacked_leaves, stacked_spec)
    assert reconstructed.core.index is reconstructed.core.integral.backend.index
    assert stacked.core.integral.backend is not None
    assert tuple(term.kind for term in padded_h2.core.classical_terms) == tuple(
        term.kind for term in h2.classical_terms
    )
    assert tuple(
        term.kind for term in padded_h2.core.interaction_terms
    ) == tuple(term.kind for term in h2.interaction_terms)

    # Every source shell/AO map is represented by the explicit canonical map.
    maps = dict(layout.source_maps)
    for source_i, source in enumerate((h2, water)):
        mapping = maps[source_i]
        original_backend = source.integral.backend
        mapped_backend = (padded_h2, padded_water)[
            source_i
        ].core.integral.backend
        assert original_backend is not None and mapped_backend is not None
        assert [layout.atom_elements[i] for i in mapping.atoms] == list(
            source.numbers.tolist()
        )
        assert [layout.shells_to_atom[i] for i in mapping.shells] == [
            mapping.atoms[i]
            for i in original_backend.index.shells_to_atom.tolist()
        ]
        assert [layout.orbitals_to_shell[i] for i in mapping.orbitals] == [
            mapping.shells[i]
            for i in original_backend.index.orbitals_to_shell.tolist()
        ]
        assert (
            mapped_backend.alphas[mapping.shell_types[0]]
            is original_backend.alphas[0]
        )
        source_h0 = source.h0_setup
        target_h0 = (padded_h2, padded_water)[source_i].core.h0_setup
        assert source_h0 is not None and target_h0 is not None
        assert target_h0.unique.tolist() == [1, 8]
        assert target_h0.index.atom_to_unique.tolist() == [0, 0, 1]
        assert target_h0.index.shells_to_ushell.index_select(
            0, torch.tensor(mapping.shells, dtype=torch.int64)
        ).tolist() == [
            mapping.shell_types[index]
            for index in source_h0.index.shells_to_ushell.tolist()
        ]
        torch.testing.assert_close(
            target_h0.hscale.index_select(
                0, torch.tensor(mapping.shell_types, dtype=torch.int64)
            ).index_select(
                1, torch.tensor(mapping.shell_types, dtype=torch.int64)
            ),
            source_h0.hscale,
        )
        torch.testing.assert_close(
            target_h0.kpair.index_select(
                0, torch.tensor(mapping.elements, dtype=torch.int64)
            ).index_select(
                1, torch.tensor(mapping.elements, dtype=torch.int64)
            ),
            source_h0.kpair,
        )
        for name in ("kcn", "refocc", "selfenergy", "shpoly", "valence"):
            torch.testing.assert_close(
                getattr(target_h0, name).index_select(
                    0, torch.tensor(mapping.shell_types, dtype=torch.int64)
                ),
                getattr(source_h0, name),
            )
        for name in ("en", "rad"):
            torch.testing.assert_close(
                getattr(target_h0, name).index_select(
                    0, torch.tensor(mapping.elements, dtype=torch.int64)
                ),
                getattr(source_h0, name),
            )
        for source_type, target_type in enumerate(mapping.shell_types):
            assert (padded_h2, padded_water)[
                source_i
            ].core.integral.backend.alphas[
                target_type
            ] is original_backend.alphas[
                source_type
            ]

        # The canonical pair plan has unique lower-triangle destinations;
        # invalid entries are marked separately, never inferred from weight.
        destinations = [
            int(value)
            for pair_class in mapped_backend.pair_plan.classes
            for value in pair_class.idx.reshape(-1).tolist()
        ]
        assert len(destinations) == len(set(destinations))
        validity = (padded_h2, padded_water)[source_i].validity
        assert len(validity.pair_entries) == len(
            mapped_backend.pair_plan.classes
        )
        for mask, pair_class in zip(
            validity.pair_entries, mapped_backend.pair_plan.classes
        ):
            assert mask.shape == pair_class.ib.shape
            assert mask.dtype is torch.bool
        real_destinations = {
            int(value)
            for mask, pair_class in zip(
                validity.pair_entries, mapped_backend.pair_plan.classes
            )
            for value in pair_class.idx[mask].reshape(-1).tolist()
        }
        padding_destinations = {
            int(value)
            for mask, pair_class in zip(
                validity.pair_entries, mapped_backend.pair_plan.classes
            )
            for value in pair_class.idx[~mask].reshape(-1).tolist()
        }
        assert real_destinations.isdisjoint(padding_destinations)

        expected_pairs = set()
        for source_class in original_backend.pair_plan.classes:
            for old_i, old_j in zip(
                source_class.ib.tolist(), source_class.jk.tolist()
            ):
                new_i, new_j = mapping.shells[old_i], mapping.shells[old_j]
                if new_i < new_j:
                    new_i, new_j = new_j, new_i
                expected_pairs.add((new_i, new_j))
        actual_pairs = {
            (int(i), int(j))
            for mask, pair_class in zip(
                validity.pair_entries, mapped_backend.pair_plan.classes
            )
            for i, j in zip(
                pair_class.ib[mask].tolist(), pair_class.jk[mask].tolist()
            )
        }
        assert actual_pairs == expected_pairs
        for pair_class, mask in zip(
            mapped_backend.pair_plan.classes, validity.pair_entries
        ):
            for ib, jk, idx in zip(
                pair_class.ib[mask].tolist(),
                pair_class.jk[mask].tolist(),
                pair_class.idx[mask],
            ):
                rows = layout.ao_offsets[ib] + torch.arange(
                    2 * pair_class.lb + 1, dtype=torch.int64
                )
                cols = layout.ao_offsets[jk] + torch.arange(
                    2 * pair_class.lk + 1, dtype=torch.int64
                )
                expected_idx = rows[:, None] * layout.orbitals + cols[None, :]
                torch.testing.assert_close(idx, expected_idx)

    # Full-core mapped Repulsion values and real-coordinate gradients.
    serial = []
    serial_grad = []
    for core, source_numbers in ((h2, numbers[0]), (water, numbers[1])):
        pos = _coordinates(source_numbers)
        energy = _repulsion(core, pos)
        serial.append(energy.detach())
        serial_grad.append(torch.autograd.grad(energy, pos)[0].detach())

    coords = tuple(
        canonicalize_positions(_coordinates(n), layout, i)
        for i, n in enumerate(numbers)
    )
    stacked_positions = torch.stack(coords)

    def one(padded_core, positions):
        return _repulsion(padded_core.core, positions)

    mapped = torch.vmap(one)(stacked, stacked_positions)
    torch.testing.assert_close(mapped, torch.stack(serial))
    mapped_grad = torch.vmap(torch.func.grad(one, argnums=1))(
        stacked, stacked_positions
    )
    for i, (core, source_numbers) in enumerate(
        ((h2, numbers[0]), (water, numbers[1]))
    ):
        mapping = maps[i]
        real_grad = mapped_grad[i].index_select(
            0, torch.tensor(mapping.atoms, dtype=torch.int64)
        )
        torch.testing.assert_close(real_grad, serial_grad[i])
        assert (
            torch.count_nonzero(mapped_grad[i][~stacked.validity.atoms[i]]) == 0
        )

    # Parameter-derived Repulsion setup values retain the original graph.
    arep = model_h2.par.get("element.H.arep")
    (mapped_parameter_gradient,) = torch.autograd.grad(
        mapped[0], arep, retain_graph=True
    )
    serial_position = _coordinates(numbers[0])
    serial_energy = _repulsion(h2, serial_position)
    (serial_parameter_gradient,) = torch.autograd.grad(serial_energy, arep)
    torch.testing.assert_close(
        mapped_parameter_gradient, serial_parameter_gradient
    )
    assert torch.isfinite(mapped_parameter_gradient).all()

    # Selected PyTorch overlap assembly is diagnostic only: compare the real
    # AO submatrix through the explicit source-to-canonical AO map.
    source_backend = h2.integral.backend
    padded_backend = padded_h2.core.integral.backend
    assert source_backend is not None and padded_backend is not None
    source_overlap = assemble_matrix(
        get_kernel(source_backend.algorithm),
        None,
        list(source_backend.alphas),
        list(source_backend.coeffs),
        _coordinates(numbers[0]),
        plan=source_backend.pair_plan,
    )
    canonical_positions = canonicalize_positions(
        _coordinates(numbers[0]), layout, 0
    )
    padded_overlap = assemble_matrix(
        get_kernel(padded_backend.algorithm),
        None,
        list(padded_backend.alphas),
        list(padded_backend.coeffs),
        canonical_positions,
        plan=padded_backend.pair_plan,
    )
    ao_map = torch.tensor(
        dict(layout.source_maps)[0].orbitals, dtype=torch.int64
    )
    real_ao_block = padded_overlap.index_select(1, ao_map).index_select(
        2, ao_map
    )
    torch.testing.assert_close(real_ao_block, source_overlap)

    # Independently created Model parameter owners remain distinct and intact.
    assert tuple(model_water.par.parameters())
    for source, snapshot in zip((h2, water), source_snapshots):
        current_leaves, _ = tree_flatten(source)
        identities, values = snapshot
        assert tuple(id(value) for value in current_leaves) == identities
        for current, before in zip(current_leaves, values):
            torch.testing.assert_close(current, before)
    for model, snapshot in zip((model_h2, model_water), parameter_snapshots):
        parameters = tuple(model.par.parameters())
        identities, values = snapshot
        assert tuple(id(value) for value in parameters) == identities
        for current, before in zip(parameters, values):
            torch.testing.assert_close(current, before)

    converted = padded_h2.to(dtype=torch.float32)
    assert converted.core.numbers.dtype is torch.int64
    assert converted.validity.atoms.dtype is torch.bool
    assert converted.core.integral.backend.alphas[0].dtype is torch.float32
    assert tuple(model_water.par.parameters())


def test_canonical_bucket_is_input_order_independent_for_schema() -> None:
    _, h2 = _system((1, 1))
    _, water = _system((8, 1, 1))
    first = bucket_canonical_exact_cores((h2, water))
    reverse = bucket_canonical_exact_cores((water, h2))
    assert first.elements == reverse.elements
    assert first.shell_types == reverse.shell_types
    assert first.atom_elements == reverse.atom_elements
    assert first.pair_classes == reverse.pair_classes
    assert first.pair_shells == reverse.pair_shells
    reversed_padded = tuple(
        pad_canonical_exact_core(core, reverse, index)
        for index, core in enumerate((water, h2))
    )
    reversed_stack = stack(reversed_padded)
    assert reversed_stack.core.numbers.tolist()[0] == [1, 1, 8]
    assert reversed_stack.core.numbers.tolist()[1] == [1, 1, 0]


def test_three_independent_sources_share_canonical_core_schema() -> None:
    _, h2 = _system((1, 1))
    _, water_oxygen_first = _system((8, 1, 1))
    _, water_hydrogen_first = _system((1, 1, 8))
    cores = (h2, water_oxygen_first, water_hydrogen_first)
    bucket = bucket_canonical_exact_cores(cores)
    assert bucket.indices == (0, 1, 2)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, i)
        for i, core in enumerate(cores)
    )
    stacked = stack(padded)
    assert stacked.core.numbers.shape == (3, 3)
    assert stacked.validity.atoms.tolist() == [
        [True, True, False],
        [True, True, True],
        [True, True, True],
    ]


def test_source_index_cannot_be_reused_for_another_core() -> None:
    """The bucket's index map is checked against each actual core structure."""
    _, h2 = _system((1, 1))
    _, n2 = _system((7, 7))
    bucket = bucket_canonical_exact_cores((h2, n2))
    with pytest.raises(ValueError, match="does not describe this source core"):
        pad_canonical_exact_core(n2, bucket, 0)


def test_different_methods_cannot_share_a_canonical_bucket() -> None:
    from dxtb import GFN2_XTB

    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    model = Model(
        par=par,
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    )
    system = model.setup(torch.tensor([1, 1], dtype=torch.int64))
    assert system.exact_core is not None
    _, gfn1 = _system((1, 1))
    with pytest.raises(TypeError, match="matching static configuration"):
        bucket_canonical_exact_cores((gfn1, system.exact_core))


def test_malformed_structural_and_dtype_metadata_is_rejected() -> None:
    """Do not admit invalid H0 maps or leaf dtypes into a bucket."""
    _, core = _system((1, 1))
    assert core.integral is not None and core.h0_setup is not None

    backend = core.integral.backend
    assert backend is not None
    bad_alphas = (backend.alphas[0].to(torch.float32), *backend.alphas[1:])
    with pytest.raises(TypeError, match="different dtypes"):
        backend.replace(alphas=bad_alphas)

    bad_h0_index = core.h0_setup.index.replace(
        atom_to_unique=torch.tensor([9, 0], dtype=torch.int64)
    )
    bad_h0 = core.h0_setup.replace(index=bad_h0_index)
    bad_indices = core.replace(h0_setup=bad_h0)
    with pytest.raises(ValueError, match="invalid index"):
        bucket_canonical_exact_cores((bad_indices,))

    bad_shell_map = core.h0_setup.index.replace(
        shells_to_atom=torch.tensor([1, 0, 1, 1], dtype=torch.int64)
    )
    with pytest.raises(ValueError, match="shell-to-atom mappings disagree"):
        bucket_canonical_exact_cores(
            (core.replace(h0_setup=core.h0_setup.replace(index=bad_shell_map)),)
        )

    class0 = backend.pair_plan.classes[0]
    bad_idx = class0.idx.clone()
    bad_idx[0, 0, 0] = -1
    bad_pair_class = class0.replace(idx=bad_idx)
    bad_plan = backend.pair_plan.replace(
        classes=(bad_pair_class, *backend.pair_plan.classes[1:])
    )
    bad_backend = backend.replace(pair_plan=bad_plan)
    bad_pair_core = core.replace(
        integral=core.integral.replace(backend=bad_backend)
    )
    with pytest.raises(ValueError, match="destination is outside"):
        bucket_canonical_exact_cores((bad_pair_core,))


def test_meta_device_leaf_mismatch_is_metadata_rejected_only() -> None:
    """Meta tensors probe structural validation, never numerical device support."""
    _, core = _system((1, 1))
    assert core.integral is not None and core.integral.backend is not None
    backend = core.integral.backend
    meta_alpha = backend.alphas[0].to(device="meta")
    with pytest.raises(RuntimeError, match="different devices"):
        backend.replace(alphas=(meta_alpha, *backend.alphas[1:]))


def test_canonical_position_mapping_rejects_wrong_source_shape() -> None:
    _, h2 = _system((1, 1))
    _, water = _system((8, 1, 1))
    bucket = bucket_canonical_exact_cores((h2, water))
    with pytest.raises(ValueError, match="source atom mapping"):
        canonicalize_positions(torch.zeros((1, 3)), bucket, 0)


def test_unsupported_integral_algorithm_is_rejected() -> None:
    _, core = _system((1, 1))
    assert core.integral is not None and core.integral.backend is not None
    bad_backend = core.integral.backend.replace(algorithm="not-a-kernel")
    bad_core = core.replace(integral=core.integral.replace(backend=bad_backend))
    with pytest.raises((KeyError, ValueError)):
        bucket_canonical_exact_cores((bad_core,))


@pytest.mark.parametrize("shell_resolved", [False, True])
def test_es2_optional_children_are_preserved_when_canonicalized(
    shell_resolved: bool,
) -> None:
    """Preserve production ES2 optional-child contracts through padding."""
    model_h2, h2 = _system((1, 1))
    model_water, water = _system((8, 1, 1))
    parameter_snapshots = tuple(
        (
            tuple(id(value) for value in model.par.parameters()),
            tuple(value.detach().clone() for value in model.par.parameters()),
        )
        for model in (model_h2, model_water)
    )
    h2 = _with_es2_resolution(model_h2, h2, shell_resolved)
    water = _with_es2_resolution(model_water, water, shell_resolved)
    source_setups = tuple(
        next(
            term.setup
            for term in core.interaction_terms
            if term.kind.lower() == "es2"
        )
        for core in (h2, water)
    )

    if shell_resolved:
        assert all(setup.shell_resolved for setup in source_setups)
        assert all(setup.shells_to_atom is not None for setup in source_setups)
        assert all(setup.shell_pair_mask is not None for setup in source_setups)
        assert all(setup.shell_mask is not None for setup in source_setups)
        assert all(setup.shell_hubbard is not None for setup in source_setups)
        assert all(setup.atom_hubbard is None for setup in source_setups)
    else:
        assert all(not setup.shell_resolved for setup in source_setups)
        assert all(setup.shells_to_atom is None for setup in source_setups)
        assert all(setup.shell_pair_mask is None for setup in source_setups)
        assert all(setup.shell_mask is None for setup in source_setups)
        assert all(setup.shell_hubbard is None for setup in source_setups)
        assert all(setup.atom_hubbard is not None for setup in source_setups)

    bucket = bucket_canonical_exact_cores((h2, water))
    if shell_resolved:
        source_map = source_setups[0].shells_to_atom.clone()
        source_map[0] = 1
        bad_setup = source_setups[0].replace(shells_to_atom=source_map)
        bad_core = h2.replace(
            interaction_terms=tuple(
                (
                    term.replace(setup=bad_setup)
                    if term.setup is source_setups[0]
                    else term
                )
                for term in h2.interaction_terms
            )
        )
        with pytest.raises(ValueError, match="ES2 shell-to-atom mapping"):
            pad_canonical_exact_core(bad_core, bucket, 0)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, index)
        for index, core in enumerate((h2, water))
    )
    stacked = stack(padded)
    assert isinstance(stacked.core, ExactSystemCore)
    assert stacked.core.integral is not None
    assert stacked.core.h0_setup is not None
    assert len(stacked.core.interaction_terms) == len(h2.interaction_terms)

    padded_setups = tuple(
        next(
            term.setup
            for term in padded_core.core.interaction_terms
            if term.kind.lower() == "es2"
        )
        for padded_core in padded
    )
    if shell_resolved:
        for setup in padded_setups:
            assert setup.shells_to_atom is not None
            assert setup.shells_to_atom.shape == (bucket.shells,)
            assert setup.shell_mask is not None
            assert setup.shell_pair_mask is not None
            assert setup.shell_hubbard is not None
            assert torch.equal(
                setup.shells_to_atom,
                padded[0].core.integral.backend.index.shells_to_atom,
            )
    else:
        assert all(setup.shells_to_atom is None for setup in padded_setups)
        assert all(setup.shell_pair_mask is None for setup in padded_setups)
        assert all(setup.shell_mask is None for setup in padded_setups)
        assert all(setup.shell_hubbard is None for setup in padded_setups)
        assert all(setup.atom_hubbard is not None for setup in padded_setups)

    for core, source_setup in zip((h2, water), source_setups):
        current_setup = next(
            term.setup
            for term in core.interaction_terms
            if term.kind.lower() == "es2"
        )
        assert current_setup is source_setup
    for model, snapshot in zip((model_h2, model_water), parameter_snapshots):
        parameters = tuple(model.par.parameters())
        identities, values = snapshot
        assert tuple(id(value) for value in parameters) == identities
        for value, original in zip(parameters, values):
            torch.testing.assert_close(value, original)


def test_retained_dynamic_shapes_are_validated_before_bucket_creation() -> None:
    """Reject malformed fixed-shape leaves while allowing value updates."""
    _, h2 = _system((1, 1))
    _, water = _system((8, 1, 1))
    assert h2.h0_setup is not None
    malformed_h0 = h2.replace(
        h0_setup=h2.h0_setup.replace(enscale=h2.h0_setup.enscale.reshape(1))
    )
    with pytest.raises(ValueError, match="scalar H0 enscale"):
        bucket_canonical_exact_cores((malformed_h0, water))

    model_h2, h2 = _system((1, 1))
    model_water, water = _system((8, 1, 1))
    h2 = _with_es2_resolution(model_h2, h2, False)
    water = _with_es2_resolution(model_water, water, False)
    descriptor = next(
        term for term in h2.interaction_terms if term.kind.lower() == "es2"
    )
    malformed_setup = descriptor.setup.replace(
        gexp=descriptor.setup.gexp.reshape(1)
    )
    malformed_es2 = h2.replace(
        interaction_terms=tuple(
            term.replace(setup=malformed_setup) if term is descriptor else term
            for term in h2.interaction_terms
        )
    )
    with pytest.raises(ValueError, match="scalar ES2 gexp"):
        bucket_canonical_exact_cores((malformed_es2, water))

    bucket = bucket_canonical_exact_cores((h2, water))
    updated_value = descriptor.setup.gexp + 0.25
    updated_setup = descriptor.setup.replace(gexp=updated_value)
    updated_core = h2.replace(
        interaction_terms=tuple(
            term.replace(setup=updated_setup) if term is descriptor else term
            for term in h2.interaction_terms
        )
    )
    padded = pad_canonical_exact_core(updated_core, bucket, 0)
    updated_descriptor = next(
        term
        for term in padded.core.interaction_terms
        if term.kind.lower() == "es2"
    )
    torch.testing.assert_close(updated_descriptor.setup.gexp, updated_value)


def test_stale_bucket_revalidates_source_pairplan() -> None:
    """Do not replace changed source pair data with captured bucket values."""
    _, h2 = _system((1, 1))
    _, water = _system((8, 1, 1))
    bucket = bucket_canonical_exact_cores((h2, water))
    assert h2.integral is not None and h2.integral.backend is not None
    backend = h2.integral.backend
    first = backend.pair_plan.classes[0]
    changed_idx = first.idx.clone()
    changed_idx[0, 0, 0] += 1
    changed_class = first.replace(idx=changed_idx)
    changed_plan = backend.pair_plan.replace(
        classes=(changed_class, *backend.pair_plan.classes[1:])
    )
    changed_core = h2.replace(
        integral=h2.integral.replace(
            backend=backend.replace(pair_plan=changed_plan)
        )
    )
    with pytest.raises(ValueError, match="Mapped PairPlan AO destinations"):
        pad_canonical_exact_core(changed_core, bucket, 0)


@pytest.mark.parametrize("numbers", [(8, 1, 1), (1, 1, 8)])
def test_h2o_real_ao_overlap_block_matches_source(
    numbers: tuple[int, ...],
) -> None:
    """Compare the physical H2O AO block in both source atom orderings."""
    _, h2 = _system((1, 1))
    _, water = _system(numbers)
    bucket = bucket_canonical_exact_cores((h2, water))
    padded = pad_canonical_exact_core(water, bucket, 1)
    source_backend = water.integral.backend
    padded_backend = padded.core.integral.backend
    assert source_backend is not None and padded_backend is not None
    source_positions = _coordinates(numbers)
    canonical_positions = canonicalize_positions(source_positions, bucket, 1)
    source_overlap = assemble_matrix(
        get_kernel(source_backend.algorithm),
        None,
        list(source_backend.alphas),
        list(source_backend.coeffs),
        source_positions,
        plan=source_backend.pair_plan,
    )
    padded_overlap = assemble_matrix(
        get_kernel(padded_backend.algorithm),
        None,
        list(padded_backend.alphas),
        list(padded_backend.coeffs),
        canonical_positions,
        plan=padded_backend.pair_plan,
    )
    ao_map = torch.tensor(
        dict(bucket.source_maps)[1].orbitals, dtype=torch.int64
    )
    real_block = padded_overlap.index_select(1, ao_map).index_select(2, ao_map)
    torch.testing.assert_close(real_block, source_overlap)
