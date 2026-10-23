# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

"""Canonical GFN2 exact-core setup and padded integral parity."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.padded_integrals import evaluate_padded_integrals
from dxtb._src.calculators.padding import compatibility_signature
from dxtb._src.integral.evaluation import build_integral_matrices

DTYPE = torch.float64

_GEOMETRIES = {
    (1, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]],
    (3, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.62]],
    (8, 1, 1): [
        [0.0, 0.0, 0.0],
        [0.82, 0.0, 0.61],
        [-0.82, 0.0, 0.61],
    ],
    (6, 1, 1, 1, 1): [
        [0.0, 0.0, 0.0],
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ],
}


def _model(*, level: int, differentiable_h0: bool = False) -> Model:
    parameters = ParamModule(GFN2_XTB, dtype=DTYPE)
    if differentiable_h0:
        parameters.set_differentiable("hamiltonian", "xtb", "wexp")
    return Model(
        par=parameters,
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=level,
        ),
        auto_int_level=False,
    )


def _sources(model: Model, compositions: tuple[tuple[int, ...], ...]):
    systems = tuple(
        model.setup(torch.tensor(numbers, dtype=torch.int64))
        for numbers in compositions
    )
    assert all(system.exact_core is not None for system in systems)
    return systems


def _positions(compositions: tuple[tuple[int, ...], ...]):
    return tuple(
        torch.tensor(_GEOMETRIES[numbers], dtype=DTYPE)
        for numbers in compositions
    )


def _mapped(systems, positions):
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, index)
        for index, core in enumerate(cores)
    )
    canonical_positions = tuple(
        canonicalize_positions(position, bucket, index)
        for index, position in enumerate(positions)
    )
    stacked = stack(padded)
    result = torch.func.vmap(evaluate_padded_integrals)(
        stacked,
        torch.stack(canonical_positions),
        torch.zeros(len(systems), dtype=DTYPE),
    )
    return bucket, padded, canonical_positions, stacked, result


def _serial(system, positions):
    assert system.integral_setup is not None
    assert system.h0_setup is not None
    return build_integral_matrices(
        system.integral_setup,
        system.h0_setup,
        positions,
        torch.zeros((), dtype=DTYPE),
    )


def _select_matrix(value: torch.Tensor, indices: tuple[int, ...]):
    index = torch.tensor(indices, dtype=torch.int64)
    return value.index_select(-2, index).index_select(-1, index)


def _assert_vector_mapping(actual, expected, indices) -> None:
    index = torch.tensor(indices, dtype=torch.int64)
    torch.testing.assert_close(actual.index_select(0, index), expected)


def _assert_square_mapping(actual, expected, indices) -> None:
    torch.testing.assert_close(_select_matrix(actual, indices), expected)


def _assert_descriptor_mapping(source_core, padded_core, mapping, bucket):
    """Check all default GFN2 atom/shell setup domains explicitly."""
    for source_term, padded_term in zip(
        source_core.classical_terms + source_core.interaction_terms,
        padded_core.classical_terms + padded_core.interaction_terms,
    ):
        assert source_term.kind == padded_term.kind
        source = source_term.setup
        actual = padded_term.setup
        kind = source_term.kind
        if kind == "d4":
            for name in ("numbers", "model_numbers", "rcov", "r4r2"):
                _assert_vector_mapping(
                    getattr(actual, name), getattr(source, name), mapping.atoms
                )
            torch.testing.assert_close(
                actual.model_rc6.index_select(
                    0, torch.tensor(mapping.atoms)
                ).index_select(1, torch.tensor(mapping.atoms)),
                source.model_rc6,
            )
        elif kind == "repulsion":
            for name in ("arep", "zeff", "kexp", "mask"):
                _assert_square_mapping(
                    getattr(actual, name), getattr(source, name), mapping.atoms
                )
        elif kind == "es2":
            for name in ("atom_pair_mask", "atom_mask"):
                _assert_square_mapping(
                    getattr(actual, name), getattr(source, name), mapping.atoms
                )
            assert source.shell_resolved
            _assert_vector_mapping(
                actual.shell_hubbard, source.shell_hubbard, mapping.shells
            )
            torch.testing.assert_close(
                actual.shells_to_atom,
                torch.tensor(bucket.shells_to_atom),
            )
            _assert_square_mapping(
                actual.shell_pair_mask, source.shell_pair_mask, mapping.shells
            )
            _assert_square_mapping(
                actual.shell_mask, source.shell_mask, mapping.shells
            )
        elif kind == "aes2":
            for name in ("numbers", "dkernel", "qkernel", "rad", "vcn"):
                _assert_vector_mapping(
                    getattr(actual, name), getattr(source, name), mapping.atoms
                )
            _assert_square_mapping(
                actual.pair_mask, source.pair_mask, mapping.atoms
            )
        elif kind == "es3":
            assert source.shell_resolved
            _assert_vector_mapping(
                actual.hubbard_derivs,
                source.hubbard_derivs,
                mapping.shells,
            )
        elif kind == "d4sc":
            for name in ("numbers", "rcov", "r4r2"):
                _assert_vector_mapping(
                    getattr(actual, name), getattr(source, name), mapping.atoms
                )
            _assert_square_mapping(
                actual.pair_weight, source.pair_weight, mapping.atoms
            )
            torch.testing.assert_close(
                actual.model_rc6.index_select(
                    0, torch.tensor(mapping.atoms)
                ).index_select(1, torch.tensor(mapping.atoms)),
                source.model_rc6,
            )
            assert actual.model_extra_entries == ()
        else:
            raise AssertionError(f"Unexpected default GFN2 term {kind!r}.")


def _assert_h0_mapping(source, padded, mapping, bucket):
    """Check GFN2 element/shell tables and exact built-in CN metadata."""
    source_h0 = source.h0_setup
    padded_h0 = padded.h0_setup
    assert source_h0 is not None and padded_h0 is not None
    assert source_h0.method == padded_h0.method == "gfn2"
    assert source_h0.cn_kind == padded_h0.cn_kind == "cn_d3(gfn2_count)"
    assert source_h0.cn is not None and padded_h0.cn is not None
    _assert_vector_mapping(padded_h0.numbers, source_h0.numbers, mapping.atoms)
    assert padded_h0.index.atom_to_unique.tolist() == [
        bucket.elements.index(number) for number in bucket.atom_elements
    ]
    assert padded_h0.index.unique_angular.tolist() == list(
        bucket.shell_type_angular
    )
    assert padded_h0.unique.tolist() == list(bucket.elements)
    assert padded_h0.index.shells_to_atom.tolist() == list(
        bucket.shells_to_atom
    )
    assert padded_h0.index.shells_to_ushell.tolist() == list(
        bucket.shells_to_type
    )
    assert padded_h0.index.orbitals_to_shell.tolist() == list(
        bucket.orbitals_to_shell
    )
    for name in ("kcn", "refocc", "selfenergy", "shpoly", "valence"):
        _assert_vector_mapping(
            getattr(padded_h0, name),
            getattr(source_h0, name),
            mapping.shell_types,
        )
    for name in ("en", "rad"):
        _assert_vector_mapping(
            getattr(padded_h0, name), getattr(source_h0, name), mapping.elements
        )
    _assert_square_mapping(
        padded_h0.hscale, source_h0.hscale, mapping.shell_types
    )
    _assert_square_mapping(padded_h0.kpair, source_h0.kpair, mapping.elements)
    assert padded_h0.enscale is source_h0.enscale
    for name in (
        "kq",
        "kqat",
        "h0rad",
        "kdiff",
        "enshell",
        "enscale4",
        "eeq_model",
        "cn_radii",
        "cn_cutoff",
        "cn_max",
        "cn_kcn",
    ):
        assert getattr(source_h0, name) is None
        assert getattr(padded_h0, name) is None


@pytest.mark.parametrize(
    "compositions",
    [
        ((1, 1), (1, 1)),
        ((1, 1), (3, 1)),
        ((1, 1), (8, 1, 1)),
        ((8, 1, 1), (1, 1)),
        ((8, 1, 1), (6, 1, 1, 1, 1)),
    ],
    ids=["h2-h2", "h2-lih", "h2-water", "water-h2", "water-methane"],
)
def test_gfn2_complete_default_cores_stack_and_evaluate(
    compositions: tuple[tuple[int, ...], ...],
) -> None:
    """Default GFN2 descriptors remain present in mapped numerical setup."""
    model = _model(level=labels.INTLEVEL_HCORE)
    systems = _sources(model, compositions)
    positions = _positions(compositions)
    bucket, padded, canonical_positions, stacked, mapped = _mapped(
        systems, positions
    )

    leaves, _ = tree_flatten(stacked)
    assert all(leaf.shape[0] == len(systems) for leaf in leaves)
    assert len(tree_flatten(padded[0])[0]) == len(tree_flatten(padded[1])[0])
    assert bucket.atoms >= max(map(len, compositions))
    assert tuple(systems[0].exact_core.config.method for _ in systems) == (
        labels.GFN2_XTB,
    ) * len(systems)

    for lane, (system, position) in enumerate(zip(systems, positions)):
        core = system.exact_core
        assert core is not None and core.h0_setup is not None
        assert core.h0_setup.cn_kind == "cn_d3(gfn2_count)"
        assert tuple(term.kind for term in core.classical_terms) == (
            "d4",
            "repulsion",
        )
        assert tuple(term.kind for term in core.interaction_terms) == (
            "es2",
            "aes2",
            "es3",
            "d4sc",
        )
        serial = _serial(system, position)
        mapping = dict(bucket.source_maps)[lane]
        _assert_h0_mapping(core, padded[lane].core, mapping, bucket)
        _assert_descriptor_mapping(core, padded[lane].core, mapping, bucket)
        pair_source = dict(bucket.source_pair_shells)[lane]
        for pair_class, source_pairs, validity in zip(
            padded[lane].core.integral.backend.pair_plan.classes,
            pair_source,
            padded[lane].validity.pair_entries,
        ):
            actual_pairs = tuple(
                (int(i), int(j))
                for i, j, is_valid in zip(
                    pair_class.ib, pair_class.jk, validity
                )
                if is_valid
            )
            assert actual_pairs == source_pairs
        torch.testing.assert_close(
            _select_matrix(mapped.overlap[lane], mapping.orbitals),
            serial[0].overlap,
            atol=2e-13,
            rtol=2e-13,
        )
        torch.testing.assert_close(
            _select_matrix(mapped.hcore[lane], mapping.orbitals),
            serial[0].hcore,
            atol=2e-13,
            rtol=2e-13,
        )
        torch.testing.assert_close(
            mapped.reference_occupations[lane].index_select(
                0, torch.tensor(mapping.shell_types)
            ),
            serial[1],
            atol=0,
            rtol=0,
        )
        assert torch.isfinite(mapped.hcore[lane]).all()
        assert torch.isfinite(mapped.overlap[lane]).all()
        assert (
            torch.count_nonzero(
                mapped.hcore[lane][~padded[lane].validity.orbitals]
            )
            == 0
        )
        assert (
            torch.count_nonzero(
                canonical_positions[lane][~padded[lane].validity.atoms]
            )
            == 0
        )


@pytest.mark.parametrize(
    ("level", "field"),
    [
        (labels.INTLEVEL_DIPOLE, "dipole"),
        (labels.INTLEVEL_QUADRUPOLE, "quadrupole"),
    ],
)
def test_gfn2_multipole_integrals_match_serial(level: int, field: str) -> None:
    """Optional integral levels retain physical GFN2 AO blocks."""
    compositions = ((1, 1), (8, 1, 1))
    systems = _sources(_model(level=level), compositions)
    positions = _positions(compositions)
    bucket, _, _, _, mapped = _mapped(systems, positions)
    for lane, (system, position) in enumerate(zip(systems, positions)):
        serial = _serial(system, position)[0]
        mapping = dict(bucket.source_maps)[lane]
        actual = getattr(mapped, field)[lane]
        expected = getattr(serial, field)
        assert actual is not None and expected is not None
        index = torch.tensor(mapping.orbitals)
        selected = actual.index_select(-2, index).index_select(-1, index)
        torch.testing.assert_close(selected, expected, atol=2e-13, rtol=2e-13)


def test_gfn2_inactive_coordinates_do_not_change_physical_integrals() -> None:
    """Padded coordinates and noncontiguous physical AO maps are inert."""
    compositions = ((1, 1), (8, 1, 1))
    systems = _sources(_model(level=labels.INTLEVEL_HCORE), compositions)
    positions = _positions(compositions)
    bucket, padded, canonical_positions, stacked, baseline = _mapped(
        systems, positions
    )
    water_map = dict(bucket.source_maps)[1]
    assert water_map.orbitals != tuple(range(len(water_map.orbitals)))
    changed = torch.stack(canonical_positions).clone()
    inactive = ~padded[0].validity.atoms
    changed[0, inactive] = torch.tensor([50.0, -70.0, 90.0], dtype=DTYPE)
    changed_result = torch.func.vmap(evaluate_padded_integrals)(
        stacked, changed, torch.zeros(2, dtype=DTYPE)
    )
    for name in ("hcore", "overlap"):
        torch.testing.assert_close(
            getattr(changed_result, name),
            getattr(baseline, name),
            atol=0,
            rtol=0,
        )

    def lane_h0(position_batch):
        result = torch.func.vmap(evaluate_padded_integrals)(
            stacked, position_batch, torch.zeros(2, dtype=DTYPE)
        )
        return result.hcore.square().sum(dim=(-2, -1))

    jacobian = torch.func.jacrev(lane_h0)(changed)
    assert torch.count_nonzero(jacobian[0, 0, inactive]) == 0
    assert torch.count_nonzero(jacobian[0, 1]) == 0


def test_gfn1_and_gfn2_never_share_a_canonical_bucket() -> None:
    """Matching tensor capacities cannot merge different xtb methods."""
    gfn2 = _sources(_model(level=labels.INTLEVEL_HCORE), ((1, 1),))[0]
    gfn1_parameters = ParamModule(GFN1_XTB, dtype=DTYPE)
    gfn1_model = Model(
        par=gfn1_parameters,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_HCORE,
        ),
        auto_int_level=False,
    )
    gfn1 = gfn1_model.setup(torch.tensor([1, 1], dtype=torch.int64))
    assert gfn1.exact_core is not None and gfn2.exact_core is not None
    assert compatibility_signature(gfn2.exact_core)
    assert compatibility_signature(gfn1.exact_core) != compatibility_signature(
        gfn2.exact_core
    )
    with pytest.raises(TypeError, match="matching static configuration"):
        bucket_canonical_exact_cores((gfn1.exact_core, gfn2.exact_core))


def test_gfn2_h0_parameter_and_position_derivatives_are_lane_local() -> None:
    """Mapped H0 derivatives retain the shared Model owner and source lanes."""
    compositions = ((1, 1), (3, 1))
    model = _model(level=labels.INTLEVEL_HCORE, differentiable_h0=True)
    systems = _sources(model, compositions)
    positions = tuple(
        value.requires_grad_() for value in _positions(compositions)
    )
    bucket, padded, canonical_positions, stacked, _ = _mapped(
        systems, positions
    )
    positions_batch = torch.stack(canonical_positions)
    charges = torch.zeros(len(systems), dtype=DTYPE)
    coefficients = (
        torch.arange(bucket.orbitals * bucket.orbitals, dtype=DTYPE).reshape(
            bucket.orbitals, bucket.orbitals
        )
        / 17.0
    )

    def lane_value(position_batch):
        result = torch.func.vmap(evaluate_padded_integrals)(
            stacked, position_batch, charges
        )
        return (result.hcore * coefficients).sum(dim=(-2, -1))

    forward = torch.func.jacfwd(lane_value)(positions_batch)
    reverse = torch.func.jacrev(lane_value)(positions_batch)
    assert forward.shape == (2, 2, bucket.atoms, 3)
    torch.testing.assert_close(forward, reverse, atol=2e-10, rtol=2e-10)
    direction = torch.linspace(
        -0.4, 0.5, positions_batch.numel(), dtype=DTYPE
    ).reshape_as(positions_batch)
    _, tangent = torch.func.jvp(lane_value, (positions_batch,), (direction,))
    torch.testing.assert_close(
        tangent,
        torch.einsum("bijc,bjc->b", forward, direction),
        atol=2e-10,
        rtol=2e-10,
    )
    for lane, (system, source_position) in enumerate(zip(systems, positions)):
        mapping = dict(bucket.source_maps)[lane]
        indices = torch.tensor(mapping.orbitals)
        source_coefficients = coefficients.index_select(
            0, indices
        ).index_select(1, indices)

        def serial_value(value):
            matrices = _serial(system, value)[0]
            return (matrices.hcore * source_coefficients).sum()

        serial_gradient = torch.func.grad(serial_value)(source_position)
        decoded = forward[lane, lane].index_select(
            0, torch.tensor(mapping.atoms)
        )
        torch.testing.assert_close(
            decoded, serial_gradient, atol=2e-10, rtol=2e-10
        )
        assert torch.count_nonzero(decoded) > 0
        off_lane = forward[lane, 1 - lane]
        assert torch.count_nonzero(off_lane) == 0
        inactive = ~padded[lane].validity.atoms
        assert torch.count_nonzero(forward[lane, lane][inactive]) == 0

    parameter = model.par.get("hamiltonian.xtb.wexp")
    assert isinstance(parameter, torch.Tensor) and parameter.requires_grad
    parameter_identity = id(parameter)
    parameter_snapshot = parameter.detach().clone()
    serial_gradients = []
    for lane, (system, position) in enumerate(zip(systems, positions)):
        mapping = dict(bucket.source_maps)[lane]
        indices = torch.tensor(mapping.orbitals)
        source_coefficients = coefficients.index_select(
            0, indices
        ).index_select(1, indices)
        serial_hcore = _serial(system, position)[0].hcore
        serial_gradients.append(
            torch.autograd.grad(
                (serial_hcore * source_coefficients).sum(),
                parameter,
                retain_graph=True,
            )[0]
        )
    serial_gradient = torch.stack(tuple(serial_gradients)).sum()
    mapped_result = torch.func.vmap(evaluate_padded_integrals)(
        stacked, positions_batch, charges
    )
    mapped_gradient = torch.autograd.grad(
        (mapped_result.hcore * coefficients).sum(), parameter
    )[0]
    torch.testing.assert_close(
        mapped_gradient, serial_gradient, atol=2e-10, rtol=2e-10
    )
    assert torch.count_nonzero(mapped_gradient) > 0
    assert id(model.par.get("hamiltonian.xtb.wexp")) == parameter_identity
    torch.testing.assert_close(
        parameter.detach(), parameter_snapshot, atol=0, rtol=0
    )
