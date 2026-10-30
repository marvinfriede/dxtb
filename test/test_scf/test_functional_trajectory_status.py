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
"""Functional SCF trajectory-status compatibility tests."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.func import vmap

from dxtb._src.scf.functional import (
    SCFCharges,
    SCFMapResult,
    SCFPotential,
    SCFSolverSettings,
    SCFSpectralResult,
    SCFTrajectoryStatus,
    functional_converged_and_valid,
    functional_subsolves_ok,
)
from dxtb._src.scf.functional.method import LinearMethod
from dxtb._src.scf.functional.solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_INVALID_TEMPERATURE,
    FERMI_FAILURE_NONCONVERGENCE,
    FERMI_FAILURE_NONE,
    _solve_map_control,
    accumulate_trajectory_status,
    fermi_failure_code,
)
from dxtb._src.wavefunction.filling import FermiStatus


def _toy_result(mapped: torch.Tensor, valid: bool) -> SCFMapResult:
    flag = torch.tensor(valid, dtype=torch.bool, device=mapped.device)
    status = FermiStatus(
        converged=flag,
        residual=mapped.new_zeros(()),
        invalid_temperature=~flag,
        too_many_electrons=torch.zeros_like(flag),
        electron_count_ok=torch.ones_like(flag),
    )
    return SCFMapResult(
        mapped_x=mapped,
        driving_potential=SCFPotential(mono=mapped),
        spectral=SCFSpectralResult(
            hamiltonian=mapped,
            density=mapped,
            emo=mapped,
            coefficients=mapped,
            occupation=mapped,
            charges=SCFCharges(mono=mapped),
            orbital_energy=mapped,
            fermi_status=status,
        ),
    )


def test_early_fermi_failure_survives_later_valid_selected_map() -> None:
    """Keep an early raw-map failure despite a valid selected final map."""
    calls = 0

    def map_fn(x: torch.Tensor) -> SCFMapResult:
        nonlocal calls
        calls += 1
        result = _toy_result(torch.ones_like(x), valid=calls > 1)
        status = result.fermi_status
        assert status is not None
        status = FermiStatus(
            converged=status.converged,
            residual=status.residual,
            invalid_temperature=torch.zeros_like(status.invalid_temperature),
            too_many_electrons=status.too_many_electrons,
            electron_count_ok=torch.ones_like(status.electron_count_ok),
        )
        return replace(
            result,
            spectral=replace(result.spectral, fermi_status=status),
        )

    settings = SCFSolverSettings(
        maxiter=1,
        mix_guess=False,
        damp=0.5,
        x_atol=0.1,
        x_atol_max=0.1,
    )
    outcome = _solve_map_control(
        torch.zeros((), dtype=torch.float64),
        maxiter=settings.maxiter,
        mix_guess=settings.mix_guess,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        map_fn=map_fn,
        method=LinearMethod(),
        initial_damp=1.0,
    )
    spectral = outcome.map_result.spectral
    assert calls == 2
    assert outcome.converged.item() is True
    assert spectral.fermi_status is not None
    assert spectral.fermi_status.converged.item() is True

    # Build the public result to check trajectory validity separately from
    # the selected map's successful Fermi status.
    from dxtb._src.scf.functional import SCFSolveResult

    result = SCFSolveResult(
        map_result=outcome.map_result,
        final_evaluation=spectral,
        root_x=outcome.root_x,
        final_charges=SCFCharges(mono=torch.ones(1, dtype=torch.float64)),
        final_potential=SCFPotential(mono=torch.ones(1, dtype=torch.float64)),
        energy=torch.zeros(1, dtype=torch.float64),
        fenergy=torch.zeros(1, dtype=torch.float64),
        iterations=outcome.iterations,
        converged=outcome.converged,
        residual=outcome.residual,
        trajectory_status=outcome.trajectory_status,
    )
    assert result.trajectory_status is not None
    assert result.trajectory_status.fermi_converged.item() is False
    assert result.trajectory_status.invalid_temperature.item() is False
    assert functional_subsolves_ok(result).item() is False
    assert functional_converged_and_valid(result).item() is False


def test_newly_converged_candidate_contributes_first_failure() -> None:
    """The physical map accepted by outer convergence still contributes status."""
    calls = 0

    def map_fn(x: torch.Tensor) -> SCFMapResult:
        nonlocal calls
        calls += 1
        result = _toy_result(torch.ones_like(x), valid=calls == 1)
        status = result.fermi_status
        assert status is not None
        recoverable_status = FermiStatus(
            converged=status.converged,
            residual=status.residual,
            invalid_temperature=torch.zeros_like(status.invalid_temperature),
            too_many_electrons=status.too_many_electrons,
            electron_count_ok=torch.ones_like(status.electron_count_ok),
        )
        return replace(
            result,
            spectral=replace(result.spectral, fermi_status=recoverable_status),
        )

    outcome = _solve_map_control(
        torch.zeros((), dtype=torch.float64),
        maxiter=1,
        mix_guess=False,
        x_atol=0.1,
        x_atol_max=0.1,
        map_fn=map_fn,
        method=LinearMethod(),
        initial_damp=1.0,
    )
    assert outcome.converged.item()
    assert outcome.map_result.fermi_status is not None
    assert outcome.map_result.fermi_status.converged.item() is False
    assert outcome.trajectory_status is not None
    assert (
        outcome.trajectory_status.first_failure.item() == FERMI_FAILURE_NONCONVERGENCE
    )


def test_trajectory_status_combines_failure_categories() -> None:
    """Each trajectory failure category remains visible after later success."""
    good = _toy_result(torch.ones(()), valid=True).fermi_status
    assert good is not None
    active = torch.tensor(True)
    initial = SCFTrajectoryStatus(
        fermi_converged=torch.tensor(True),
        electron_count_ok=torch.tensor(True),
        invalid_temperature=torch.tensor(False),
        too_many_electrons=torch.tensor(False),
        first_failure=torch.tensor(FERMI_FAILURE_NONE),
    )

    for field in (
        "fermi_converged",
        "electron_count_ok",
        "invalid_temperature",
        "too_many_electrons",
    ):
        values = {
            "converged": torch.tensor(True),
            "residual": torch.tensor(0.0),
            "invalid_temperature": torch.tensor(False),
            "too_many_electrons": torch.tensor(False),
            "electron_count_ok": torch.tensor(True),
        }
        map_field = {
            "fermi_converged": "converged",
            "electron_count_ok": "electron_count_ok",
            "invalid_temperature": "invalid_temperature",
            "too_many_electrons": "too_many_electrons",
        }[field]
        values[map_field] = torch.tensor(
            field
            in (
                "invalid_temperature",
                "too_many_electrons",
            )
        )
        failed = type(good)(**values)
        accumulated = accumulate_trajectory_status(initial, failed, active)
        assert accumulated is not None
        recovered = accumulate_trajectory_status(accumulated, good, active)
        assert recovered is not None
        if field == "fermi_converged":
            assert not recovered.fermi_converged.item()
        elif field == "electron_count_ok":
            assert not recovered.electron_count_ok.item()
        elif field == "invalid_temperature":
            assert recovered.invalid_temperature.item()
        else:
            assert recovered.too_many_electrons.item()


def test_first_failure_is_chronological_and_category_specific() -> None:
    """An early post-smearing failure remains before later Fermi failure."""

    def status(*, converged: bool, electron_count_ok: bool) -> FermiStatus:
        return FermiStatus(
            converged=torch.tensor(converged),
            residual=torch.tensor(0.0),
            invalid_temperature=torch.tensor(False),
            too_many_electrons=torch.tensor(False),
            electron_count_ok=torch.tensor(electron_count_ok),
        )

    active = torch.tensor(True)
    trajectory = accumulate_trajectory_status(
        None, status(converged=True, electron_count_ok=False), active
    )
    trajectory = accumulate_trajectory_status(
        trajectory, status(converged=False, electron_count_ok=True), active
    )
    assert trajectory is not None
    assert trajectory.first_failure.item() == FERMI_FAILURE_ELECTRON_COUNT
    assert not trajectory.fermi_converged.item()
    assert not trajectory.electron_count_ok.item()

    reverse = accumulate_trajectory_status(
        None, status(converged=False, electron_count_ok=True), active
    )
    reverse = accumulate_trajectory_status(
        reverse, status(converged=True, electron_count_ok=False), active
    )
    assert reverse is not None
    assert reverse.first_failure.item() == FERMI_FAILURE_NONCONVERGENCE


def test_one_spectrum_priority_matches_eager_fermi_order() -> None:
    """Search failure outranks post-smearing count failure in one spectrum."""
    status = FermiStatus(
        converged=torch.tensor([True, False]),
        residual=torch.zeros(2),
        invalid_temperature=torch.tensor([False, False]),
        too_many_electrons=torch.tensor([False, False]),
        electron_count_ok=torch.tensor([False, True]),
    )
    code = fermi_failure_code(status, like=torch.zeros(()))
    assert code.item() == FERMI_FAILURE_NONCONVERGENCE


def test_first_failure_vmap_is_lane_local() -> None:
    """Each mapped lane retains its own earliest category and ignores dummies."""

    def combine(lane: torch.Tensor) -> tuple[torch.Tensor, ...]:
        step_one = FermiStatus(
            converged=lane != 2,
            residual=lane.new_zeros(()),
            invalid_temperature=torch.zeros_like(lane, dtype=torch.bool),
            too_many_electrons=torch.zeros_like(lane, dtype=torch.bool),
            electron_count_ok=lane != 1,
        )
        step_two = FermiStatus(
            converged=lane != 1,
            residual=lane.new_zeros(()),
            invalid_temperature=torch.zeros_like(lane, dtype=torch.bool),
            too_many_electrons=torch.zeros_like(lane, dtype=torch.bool),
            electron_count_ok=lane != 2,
        )
        first = accumulate_trajectory_status(
            None, step_one, torch.tensor(True, device=lane.device)
        )
        active = lane != 3
        combined = accumulate_trajectory_status(first, step_two, active)
        assert combined is not None
        return (
            combined.first_failure,
            combined.fermi_converged,
            combined.electron_count_ok,
            active,
        )

    lanes = torch.arange(4)
    actual = vmap(combine)(lanes)
    expected = tuple(
        torch.stack(values) for values in zip(*(combine(i) for i in lanes))
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)
    assert actual[0].tolist() == [0, 4, 3, 0]


def test_inactive_dummy_status_does_not_pollute_trajectory() -> None:
    """Inactive fixed-trip map statuses do not change accumulated validity."""
    valid = _toy_result(torch.ones(()), valid=True).fermi_status
    invalid = _toy_result(torch.ones(()), valid=False).fermi_status
    assert valid is not None and invalid is not None
    initial = accumulate_trajectory_status(None, valid, torch.tensor(True))
    accumulated = accumulate_trajectory_status(initial, invalid, torch.tensor(False))
    assert accumulated is not None
    assert accumulated.fermi_converged.item()
    assert accumulated.electron_count_ok.item()
    assert not accumulated.invalid_temperature.item()
    assert accumulated.first_failure.item() == FERMI_FAILURE_NONE


def test_trajectory_status_vmap_matches_python_loop() -> None:
    """Mapped toy lanes retain independent trajectory validity and status."""

    def solve_one(guess: torch.Tensor) -> tuple[torch.Tensor, ...]:
        def map_fn(x: torch.Tensor) -> SCFMapResult:
            invalid = (x > 0.5) & (x < 1.5)
            mapped = torch.where(x > 1.5, x + 1.0, torch.zeros_like(x))
            value = torch.ones_like(x)
            status = FermiStatus(
                converged=~invalid,
                residual=torch.zeros_like(x),
                invalid_temperature=torch.zeros_like(invalid),
                too_many_electrons=torch.zeros_like(invalid),
                electron_count_ok=torch.ones_like(invalid),
            )
            return SCFMapResult(
                mapped_x=mapped,
                driving_potential=SCFPotential(mono=value),
                spectral=SCFSpectralResult(
                    hamiltonian=value,
                    density=value,
                    emo=value,
                    coefficients=value,
                    occupation=value,
                    charges=SCFCharges(mono=value),
                    orbital_energy=value,
                    fermi_status=status,
                ),
            )

        outcome = _solve_map_control(
            guess,
            maxiter=2,
            mix_guess=False,
            x_atol=0.1,
            x_atol_max=0.1,
            map_fn=map_fn,
            method=LinearMethod(),
            initial_damp=0.5,
        )
        status = outcome.trajectory_status
        assert status is not None
        final_status = outcome.map_result.fermi_status
        assert final_status is not None
        subsolves_ok = (
            status.fermi_converged.all()
            & status.electron_count_ok.all()
            & ~status.invalid_temperature.any()
            & ~status.too_many_electrons.any()
            & final_status.converged.all()
        )
        return (
            outcome.map_result.mapped_x,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
            status.fermi_converged,
            status.electron_count_ok,
            status.invalid_temperature,
            status.too_many_electrons,
            subsolves_ok,
        )

    guesses = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    mapped = vmap(solve_one)(guesses)
    loop = tuple(
        torch.stack(values) for values in zip(*(solve_one(g) for g in guesses))
    )
    for actual, expected in zip(mapped, loop):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert mapped[2].tolist() == [True, True, False]
    assert mapped[-1].tolist() == [True, False, True]
