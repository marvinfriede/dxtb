# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

"""Batch-critical padding safety regressions for mapped GFN1 and GFN2."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch.utils._pytree import tree_flatten

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.scf.functional import energy as functional_energy
from dxtb._src.scf.functional.padded import (
    _prepare_padded_interactions,
    evaluate_padded_systems,
)
from dxtb._src.scf.functional.spectral import _solve_positive_definite

DTYPE = torch.float64

_POSITIONS = {
    (1, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]],
    (3, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.62]],
    (8, 1, 1): [
        [0.0, 0.0, 0.0],
        [0.82, 0.0, 0.61],
        [-0.82, 0.0, 0.61],
    ],
}


def _build_systems(
    method: int,
    compositions: Sequence[tuple[int, ...]],
    *,
    maxiter: int = 1,
    temperature: float = 0.0,
):
    is_gfn2 = method == labels.GFN2_XTB
    parameters = ParamModule(GFN2_XTB if is_gfn2 else GFN1_XTB, dtype=DTYPE)
    model = Model(
        par=parameters,
        config=Config.create(
            method=method,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=(
                labels.INTLEVEL_QUADRUPOLE if is_gfn2 else labels.INTLEVEL_HCORE
            ),
            mixer=labels.MIXER_LINEAR,
            maxiter=maxiter,
            fermi_etemp=temperature,
            fermi_partition=labels.FERMI_PARTITION_EQUAL,
        ),
        auto_int_level=False,
    )
    systems = tuple(
        model.setup(torch.tensor(numbers, dtype=torch.int64))
        for numbers in compositions
    )
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, index)
        for index, core in enumerate(cores)
    )
    source_positions = tuple(
        torch.tensor(_POSITIONS[numbers], dtype=DTYPE)
        for numbers in compositions
    )
    positions = tuple(
        canonicalize_positions(position, bucket, index)
        for index, position in enumerate(source_positions)
    )
    charges = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    return systems, bucket, padded, source_positions, positions, charges, spins


def _direction(position: torch.Tensor) -> torch.Tensor:
    values = torch.arange(1, position.numel() + 1, dtype=position.dtype)
    return values.reshape_as(position) / values.norm()


def _third_directional(function, point, first, second=None, third=None):
    second = first if second is None else second
    third = first if third is None else third

    def hessian_contraction(value):
        hessian_vector = torch.func.jvp(
            torch.func.grad(function), (value,), (second,)
        )[1]
        hessian_leaves, _ = tree_flatten(hessian_vector)
        direction_leaves, _ = tree_flatten(third)
        return torch.stack(
            [
                (hessian * tangent).sum()
                for hessian, tangent in zip(hessian_leaves, direction_leaves)
            ]
        ).sum()

    return torch.func.jvp(hessian_contraction, (point,), (first,))[1]


def _third_directional_forward(function, point, first, second, third):
    def first_derivative(value):
        return torch.func.jvp(function, (value,), (third,))[1]

    def second_derivative(value):
        return torch.func.jvp(first_derivative, (value,), (second,))[1]

    return torch.func.jvp(second_derivative, (point,), (first,))[1]


def test_spectral_positive_definite_solve_supports_forward_second_derivatives() -> (
    None
):
    """The Cholesky solve keeps forward and reverse Hessians consistent."""
    dtype = DTYPE
    matrix = torch.tensor(
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]],
        dtype=dtype,
    )
    direction = torch.tensor(
        [[0.1, 0.2, -0.1], [0.2, -0.15, 0.05], [-0.1, 0.05, 0.08]],
        dtype=dtype,
    )
    rhs = torch.tensor([[1.0], [-0.5], [0.25]], dtype=dtype)
    weight = torch.tensor([[0.5], [1.25], [-0.75]], dtype=dtype)

    def function(value):
        solved = _solve_positive_definite(matrix + value * direction, rhs)
        return (solved * weight).sum()

    zero = torch.zeros((), dtype=dtype)
    tangent = torch.ones_like(zero)
    forward_second = torch.func.jvp(
        lambda value: torch.func.jvp(function, (value,), (tangent,))[1],
        (zero,),
        (tangent,),
    )[1]
    reverse_second = torch.func.jvp(
        torch.func.grad(function), (zero,), (tangent,)
    )[1]
    step = 1.0e-3
    finite_difference = (
        torch.func.grad(function)(zero + step)
        - torch.func.grad(function)(zero - step)
    ) / (2.0 * step)

    torch.testing.assert_close(
        forward_second, reverse_second, atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        forward_second, finite_difference, atol=1.0e-6, rtol=1.0e-6
    )


@pytest.mark.parametrize(
    "method,compositions",
    [
        (labels.GFN1_XTB, ((1, 1), (8, 1, 1))),
        (labels.GFN2_XTB, ((1, 1), (8, 1, 1))),
    ],
    ids=["gfn1", "gfn2"],
)
def test_inactive_coordinate_perturbation_is_safe_through_full_scf(
    method: int, compositions: tuple[tuple[int, ...], ...]
) -> None:
    """Coincident and displaced padding cannot affect full mapped results."""
    _, bucket, padded, _, positions, charges, spins = _build_systems(
        method, compositions
    )
    perturbed = tuple(
        position.clone().requires_grad_() for position in positions
    )
    adversarial = []
    for lane, (position, core) in enumerate(zip(perturbed, padded)):
        changed = position.detach().clone()
        inactive = ~core.validity.atoms
        if lane == 0:
            changed[inactive] = position.detach()[0]
        else:
            changed[inactive] = torch.tensor([17.0, -23.0, 31.0], dtype=DTYPE)
        adversarial.append(changed.requires_grad_())

    expected, _ = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    actual, _ = evaluate_padded_systems(
        padded, bucket, tuple(adversarial), charges, spins
    )
    for field in ("energy", "scf", "density", "hcore", "overlap"):
        torch.testing.assert_close(
            getattr(actual, field),
            getattr(expected, field),
            atol=2e-10,
            rtol=2e-10,
        )
    gradient = torch.autograd.grad(actual.energy.sum(), adversarial)
    for value, core in zip(gradient, padded):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value[~core.validity.atoms]) == 0
        assert torch.count_nonzero(value[core.validity.atoms]) > 0


def test_finite_temperature_trace_is_padding_safe_through_mixed_gfn1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The equal-partition trace path composes with canonical SCF padding."""
    compositions = ((1, 1), (8, 1, 1))
    systems, bucket, padded, source, positions, charges, spins = _build_systems(
        labels.GFN1_XTB, compositions, maxiter=2, temperature=300.0
    )
    positions = tuple(value.requires_grad_() for value in positions)

    trace_calls = 0
    trace = functional_energy.fermi_free_energy_trace

    def record_trace(*args, **kwargs):
        nonlocal trace_calls
        trace_calls += 1
        return trace(*args, **kwargs)

    monkeypatch.setattr(
        functional_energy, "fermi_free_energy_trace", record_trace
    )

    def mapped_energies(values):
        result, _ = evaluate_padded_systems(
            padded, bucket, values, charges, spins
        )
        return result.energy.sum(-1) + result.fenergy.sum(-1)

    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    mapped_trace_calls = trace_calls
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source, charges, spins
        )
    )
    mapped_energy = mapped.energy.sum(-1) + mapped.fenergy.sum(-1)
    assert torch.isfinite(mapped_energy).all()
    assert torch.isfinite(mapped.fenergy).all()
    assert mapped_trace_calls > 0
    assert all(
        system.config.scf.fermi.partition == labels.FERMI_PARTITION_EQUAL
        for system in systems
    )

    source_maps = dict(bucket.source_maps)
    has_noncontiguous_orbitals = False
    for lane, (actual, expected, core) in enumerate(
        zip(decoded, serial, padded)
    ):
        torch.testing.assert_close(
            actual.energy,
            expected.energy,
            atol=3e-9,
            rtol=3e-9,
        )
        torch.testing.assert_close(
            actual.fenergy,
            expected.fenergy,
            atol=3e-9,
            rtol=3e-9,
        )
        torch.testing.assert_close(
            actual.energy.sum() + actual.fenergy.sum(),
            expected.energy.sum() + expected.fenergy.sum(),
            atol=3e-9,
            rtol=3e-9,
        )
        assert torch.isfinite(actual.energy).all()
        assert torch.isfinite(actual.fenergy).all()
        atom_indices = torch.tensor(
            source_maps[bucket.indices[lane]].atoms, dtype=torch.int64
        )
        orbital_indices = torch.tensor(
            source_maps[bucket.indices[lane]].orbitals, dtype=torch.int64
        )
        has_noncontiguous_orbitals |= not torch.equal(
            orbital_indices, torch.arange(orbital_indices.numel())
        )
        assert (
            torch.count_nonzero(mapped.energy[lane, ~core.validity.atoms]) == 0
        )
        assert (
            torch.count_nonzero(mapped.fenergy[lane, ~core.validity.atoms]) == 0
        )
        assert mapped.occupation is not None

        # Occupations are energy-sorted molecular-orbital values. Canonical AO
        # masks can be noncontiguous and must not index this sorted spectrum.
        valid_orbitals = int(core.validity.orbitals.sum())
        assert (
            torch.count_nonzero(mapped.occupation[lane, ..., valid_orbitals:])
            == 0
        )
    assert has_noncontiguous_orbitals

    jacobian = torch.func.jacrev(mapped_energies)(positions)
    for output_lane, (system, source_position, core) in enumerate(
        zip(systems, source, padded)
    ):
        source_map = source_maps[bucket.indices[output_lane]]
        atom_indices = torch.tensor(source_map.atoms, dtype=torch.int64)

        def serial_lane_energy(value):
            result = system.singlepoint(
                value,
                chrg=charges[output_lane],
                spin=spins[output_lane],
            )
            return result.energy.sum() + result.fenergy.sum()

        serial_gradient = torch.func.grad(serial_lane_energy)(source_position)
        torch.testing.assert_close(
            jacobian[output_lane][output_lane].index_select(0, atom_indices),
            serial_gradient,
            atol=3e-8,
            rtol=3e-8,
        )
        torch.testing.assert_close(
            jacobian[output_lane][output_lane][~core.validity.atoms],
            torch.zeros_like(
                jacobian[output_lane][output_lane][~core.validity.atoms]
            ),
            atol=0,
            rtol=0,
        )
        other_lane = 1 - output_lane
        torch.testing.assert_close(
            jacobian[other_lane][output_lane],
            torch.zeros_like(jacobian[other_lane][output_lane]),
            atol=0,
            rtol=0,
        )

    lane = 0
    source_direction = _direction(source[lane])
    atom_indices = torch.tensor(
        source_maps[bucket.indices[lane]].atoms, dtype=torch.int64
    )
    padded_direction = torch.zeros_like(positions[lane]).index_copy(
        0, atom_indices, source_direction
    )
    directions = (padded_direction, torch.zeros_like(positions[1]))

    def lane_energy(values):
        return mapped_energies(values)[lane]

    _, mapped_hvp = torch.func.jvp(
        torch.func.grad(lane_energy), (positions,), (directions,)
    )

    def serial_energy(value):
        result = systems[lane].singlepoint(
            value, chrg=charges[lane], spin=spins[lane]
        )
        return result.energy.sum() + result.fenergy.sum()

    _, serial_hvp = torch.func.jvp(
        torch.func.grad(serial_energy),
        (source[lane],),
        (source_direction,),
    )
    assert torch.isfinite(mapped_hvp[0]).all()
    torch.testing.assert_close(
        mapped_hvp[0].index_select(0, atom_indices),
        serial_hvp,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        mapped_hvp[0][~padded[0].validity.atoms],
        torch.zeros_like(mapped_hvp[0][~padded[0].validity.atoms]),
        atol=0,
        rtol=0,
    )


def test_aes2_invalid_pairs_are_zero_before_multipole_contraction() -> None:
    """Invalid AES2 pairs cannot couple adversarial padded multipoles."""
    _, bucket, padded, _, positions, _, _ = _build_systems(
        labels.GFN2_XTB, ((1, 1), (8, 1, 1))
    )
    core = padded[0]
    inactive = ~core.validity.atoms
    adversarial = positions[0].clone()
    adversarial[inactive] = torch.tensor([11.0, -9.0, 5.0], dtype=DTYPE)
    adversarial.requires_grad_()

    # Nonzero values in all padded multipole slots must not enter the physical
    # contraction or create padded-coordinate derivatives.
    def contraction(position):
        interactions = _prepare_padded_interactions(core, position)
        aes2 = next(
            entry for entry in interactions.entries if entry.kind == "aes2"
        )
        return (
            aes2.tensor("amat_dd") * torch.ones_like(aes2.tensor("amat_dd"))
        ).sum()

    interactions = _prepare_padded_interactions(core, adversarial)
    aes2 = next(entry for entry in interactions.entries if entry.kind == "aes2")
    for name in ("amat_sd", "amat_dd", "amat_sq"):
        matrix = aes2.tensor(name)
        assert torch.isfinite(matrix).all()
        assert torch.count_nonzero(matrix[~core.validity.atoms]) == 0
        assert torch.count_nonzero(matrix[:, ~core.validity.atoms]) == 0

    gradient = torch.func.grad(contraction)(adversarial)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[inactive]) == 0
    assert torch.count_nonzero(gradient[core.validity.atoms]) > 0
    direction = torch.ones_like(adversarial)
    _, tangent = torch.func.jvp(contraction, (adversarial,), (direction,))
    torch.testing.assert_close(tangent, (gradient * direction).sum())
    torch.testing.assert_close(
        torch.func.jacfwd(contraction)(adversarial), gradient
    )


@pytest.mark.parametrize(
    "method,compositions",
    [
        (labels.GFN1_XTB, ((1, 1), (8, 1, 1))),
        (labels.GFN2_XTB, ((1, 1), (3, 1))),
    ],
    ids=["gfn1", "gfn2"],
)
def test_mixed_full_energy_third_directional_derivative_is_lane_local(
    method: int, compositions: tuple[tuple[int, ...], ...]
) -> None:
    """Third derivatives of full mapped energies match serial physical lanes."""
    systems, bucket, padded, source, positions, charges, spins = _build_systems(
        method, compositions
    )
    positions = tuple(position.requires_grad_() for position in positions)
    source_maps = dict(bucket.source_maps)
    mismatches = []

    def mapped_lane_energy(lane: int, values):
        result, _ = evaluate_padded_systems(
            padded, bucket, values, charges, spins
        )
        return result.energy.sum(-1)[lane]

    for lane, (system, source_position) in enumerate(zip(systems, source)):
        source_map = source_maps[bucket.indices[lane]]
        atom_index = torch.tensor(source_map.atoms, dtype=torch.int64)
        source_direction = _direction(source_position)
        padded_direction = torch.zeros_like(positions[lane]).index_copy(
            0, atom_index, source_direction
        )
        lane_directions = tuple(
            padded_direction if index == lane else torch.zeros_like(value)
            for index, value in enumerate(positions)
        )

        def lane_function(values):
            return mapped_lane_energy(lane, values)

        mapped_third = _third_directional(
            lane_function, positions, lane_directions
        )
        mapped_forward = _third_directional_forward(
            lane_function,
            positions,
            lane_directions,
            lane_directions,
            lane_directions,
        )
        serial_function = lambda value: system.singlepoint(
            value, chrg=charges[lane], spin=spins[lane]
        ).energy.sum()
        serial_third = _third_directional(
            serial_function, source_position, source_direction
        )
        serial_forward = _third_directional_forward(
            serial_function,
            source_position,
            source_direction,
            source_direction,
            source_direction,
        )
        assert torch.isfinite(mapped_third)
        assert torch.isfinite(serial_third)
        assert torch.abs(serial_third) > 1.0e-8
        torch.testing.assert_close(
            mapped_forward, mapped_third, atol=3e-5, rtol=3e-5
        )
        torch.testing.assert_close(
            serial_forward, serial_third, atol=3e-5, rtol=3e-5
        )
        if not torch.isclose(mapped_third, serial_third, atol=3e-5, rtol=3e-5):
            mismatches.append(
                (lane, mapped_third.detach(), serial_third.detach())
            )

        # Directional derivatives involving only another System's geometry
        # or this lane's inactive coordinates must be exactly absent.
        other_directions = tuple(
            _direction(value) if index != lane else torch.zeros_like(value)
            for index, value in enumerate(positions)
        )
        cross_third = _third_directional(
            lane_function, positions, other_directions
        )
        torch.testing.assert_close(
            cross_third,
            torch.zeros_like(cross_third),
            atol=1e-10,
            rtol=0,
        )
        inactive_directions = tuple(
            (
                torch.where(
                    core.validity.atoms.unsqueeze(-1),
                    torch.zeros_like(value),
                    _direction(value),
                )
                if index == lane
                else torch.zeros_like(value)
            )
            for index, (value, core) in enumerate(zip(positions, padded))
        )
        inactive_third = _third_directional(
            lane_function, positions, inactive_directions
        )
        torch.testing.assert_close(
            inactive_third,
            torch.zeros_like(inactive_third),
            atol=1e-10,
            rtol=0,
        )

    assert not mismatches, (
        f"{method=} third-directional derivative mismatch by lane: "
        f"{mismatches}"
    )


@pytest.mark.parametrize(
    "method,compositions,reverse",
    [
        (labels.GFN1_XTB, ((1, 1), (8, 1, 1)), False),
        (labels.GFN1_XTB, ((8, 1, 1), (1, 1)), True),
        (labels.GFN2_XTB, ((1, 1), (3, 1)), False),
        (labels.GFN2_XTB, ((3, 1), (1, 1)), True),
    ],
    ids=["gfn1", "gfn1-reversed", "gfn2", "gfn2-reversed"],
)
def test_mixed_full_energy_third_derivative_uses_distinct_directions(
    method: int, compositions: tuple[tuple[int, ...], ...], reverse: bool
) -> None:
    """Mixed third derivatives match serial evaluation for three directions."""
    systems, bucket, padded, source, positions, charges, spins = _build_systems(
        method, compositions
    )
    lane = 0 if reverse else 1
    source_map = dict(bucket.source_maps)[bucket.indices[lane]]
    atom_index = torch.tensor(source_map.atoms, dtype=torch.int64)
    source_position = source[lane]

    first = _direction(source_position)
    second = _direction(torch.flip(source_position, dims=(0,)))
    third = _direction(torch.roll(source_position, shifts=1, dims=1))
    source_directions = (first, second, third)
    padded_directions = tuple(
        torch.zeros_like(positions[lane]).index_copy(0, atom_index, value)
        for value in source_directions
    )
    for source_direction, padded_direction in zip(
        source_directions, padded_directions
    ):
        torch.testing.assert_close(
            padded_direction.index_select(0, atom_index), source_direction
        )
        assert (
            torch.count_nonzero(padded_direction[~padded[lane].validity.atoms])
            == 0
        )

    def mapped_energy(values):
        result, _ = evaluate_padded_systems(
            padded, bucket, values, charges, spins
        )
        return result.energy.sum(-1)[lane]

    mapped_directions = tuple(
        tuple(
            direction if index == lane else torch.zeros_like(position)
            for index, position in enumerate(positions)
        )
        for direction in padded_directions
    )
    mapped_third = _third_directional(
        mapped_energy, positions, *mapped_directions
    )
    mapped_forward = _third_directional_forward(
        mapped_energy, positions, *mapped_directions
    )

    def serial_energy(value):
        return (
            systems[lane]
            .singlepoint(value, chrg=charges[lane], spin=spins[lane])
            .energy.sum()
        )

    serial_third = _third_directional(
        serial_energy, source_position, *source_directions
    )
    serial_forward = _third_directional_forward(
        serial_energy, source_position, *source_directions
    )
    assert torch.isfinite(mapped_third)
    assert torch.isfinite(serial_third)
    assert torch.abs(serial_third) > 1.0e-8
    torch.testing.assert_close(
        mapped_forward, mapped_third, atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        serial_forward, serial_third, atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(mapped_third, serial_third, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "method,compositions",
    [
        (labels.GFN1_XTB, ((1, 1), (8, 1, 1))),
        (labels.GFN2_XTB, ((1, 1), (3, 1))),
    ],
    ids=["gfn1", "gfn2"],
)
def test_mixed_full_energy_third_derivative_has_both_lane_tangents(
    method: int, compositions: tuple[tuple[int, ...], ...]
) -> None:
    """Combined directions perturb both independent physical Systems."""
    systems, bucket, padded, source, positions, charges, spins = _build_systems(
        method, compositions
    )
    source_maps = dict(bucket.source_maps)
    source_direction_sets = []
    padded_direction_sets = []
    for transform in (
        lambda value: value,
        lambda value: torch.flip(value, dims=(0,)),
        lambda value: torch.roll(value, shifts=1, dims=1),
    ):
        physical_directions = []
        lane_directions = []
        for lane, source_position in enumerate(source):
            source_map = source_maps[bucket.indices[lane]]
            atom_index = torch.tensor(source_map.atoms, dtype=torch.int64)
            physical_direction = _direction(transform(source_position))
            assert torch.count_nonzero(physical_direction) > 0
            physical_directions.append(physical_direction)
            lane_directions.append(
                torch.zeros_like(positions[lane]).index_copy(
                    0, atom_index, physical_direction
                )
            )
        source_direction_sets.append(tuple(physical_directions))
        padded_direction_sets.append(tuple(lane_directions))
    for lane_directions in zip(*padded_direction_sets):
        assert all(torch.count_nonzero(value) > 0 for value in lane_directions)

    def mapped_energy(values):
        result, _ = evaluate_padded_systems(
            padded, bucket, values, charges, spins
        )
        return result.energy.sum()

    def serial_energy(values):
        return torch.stack(
            [
                system.singlepoint(value, chrg=charge, spin=spin).energy.sum()
                for system, value, charge, spin in zip(
                    systems, values, charges, spins
                )
            ]
        ).sum()

    mapped_third = _third_directional(
        mapped_energy, positions, *padded_direction_sets
    )
    serial_third = _third_directional(
        serial_energy, source, *source_direction_sets
    )
    torch.testing.assert_close(mapped_third, serial_third, atol=3e-5, rtol=3e-5)
