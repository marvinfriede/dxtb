# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

"""Batch-critical padding safety regressions for mapped GFN1 and GFN2."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.scf.functional.padded import (
    _prepare_padded_interactions,
    evaluate_padded_systems,
)

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


def _build_systems(method: int, compositions: Sequence[tuple[int, ...]]):
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
            maxiter=1,
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


def _third_directional(function, point, direction):
    def first(value):
        return torch.func.jvp(function, (value,), (direction,))[1]

    def second(value):
        return torch.func.jvp(first, (value,), (direction,))[1]

    return torch.func.jvp(second, (point,), (direction,))[1]


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
        serial_function = lambda value: system.singlepoint(
            value, chrg=charges[lane], spin=spins[lane]
        ).energy.sum()
        serial_third = _third_directional(
            serial_function, source_position, source_direction
        )
        assert torch.isfinite(mapped_third)
        assert torch.isfinite(serial_third)
        assert torch.abs(serial_third) > 1.0e-8
        if not torch.isclose(mapped_third, serial_third, atol=3e-5, rtol=3e-5):
            mismatches.append(
                (lane, mapped_third.detach(), serial_third.detach())
            )

        # Directional derivatives involving only another System's geometry
        # or this lane's inactive coordinates must be exactly absent.
        if lane == 0:
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
            inactive = ~padded[lane].validity.atoms
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
