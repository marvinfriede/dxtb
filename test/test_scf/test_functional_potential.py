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
"""Functional potential-representation solver parity and transforms."""

from __future__ import annotations

from dataclasses import replace
from math import prod
from types import SimpleNamespace

import pytest
import torch
from tad_mctc.units.energy import KELVIN2AU
from torch.func import vmap

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.constants import defaults
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional import (
    SCFCharges,
)
from dxtb._src.scf.functional import anderson as functional_anderson
from dxtb._src.scf.functional import (
    functional_converged_and_valid,
    functional_subsolves_ok,
)
from dxtb._src.scf.functional import map as functional_map
from dxtb._src.scf.functional import (
    potential_initial_guess,
    setup_anderson_solver_settings,
    setup_simple_solver_settings,
    simple_mix,
    solve_scf_anderson,
    solve_scf_potential_anderson,
    solve_scf_potential_simple,
)
from dxtb._src.scf.functional import solver as functional_solver
from dxtb._src.scf.functional.map import potential_map as apply_potential_map

from .test_functional_map import (
    _assert_directional_fd,
    _assert_tensor_equal,
    _prepare_case,
)


def _initial_guess(case: SimpleNamespace) -> tuple[torch.Tensor, object]:
    layout = case.inputs.charge_layout
    initial = SCFCharges(
        mono=case.initial_charges,
        dipole=(
            None
            if layout.dipole_shape is None
            else case.initial_charges.new_zeros(layout.dipole_shape)
        ),
        quad=(
            None
            if layout.quadrupole_shape is None
            else case.initial_charges.new_zeros(layout.quadrupole_shape)
        ),
    )
    return potential_initial_guess(initial, case.inputs)


def _assert_public_parity(
    actual: object,
    expected: dict[str, object],
) -> None:
    for key in (
        "density",
        "hamiltonian",
        "emo",
        "occupation",
        "energy",
        "fenergy",
    ):
        _assert_tensor_equal(getattr(actual, key), expected[key])
    actual_projector = (
        actual.coefficients
        @ torch.diag_embed(actual.occupation.sum(-2))
        @ actual.coefficients.mT
    )
    expected_projector = (
        expected["coefficients"]
        @ torch.diag_embed(expected["occupation"].sum(-2))
        @ expected["coefficients"].mT
    )
    _assert_tensor_equal(actual_projector, expected_projector)
    for key in ("mono", "dipole", "quad"):
        want = getattr(expected["charges"], key)
        got = getattr(actual.final_charges, key)
        if want is None:
            assert got is None
        else:
            assert got is not None
            _assert_tensor_equal(got, want)
        want_p = getattr(expected["potential"], key)
        got_p = getattr(actual.final_potential, key)
        if want_p is None:
            assert got_p is None
        else:
            assert got_p is not None
            _assert_tensor_equal(got_p, want_p)
    assert int(actual.iterations) == expected["iterations"]


def _run_legacy_and_functional(
    options: dict[str, object], *, method: int = GFN1_XTB
) -> tuple[SimpleNamespace, object, dict[str, object]]:
    options = {"scp_mode": labels.SCP_MODE_POTENTIAL, **options}
    legacy = _prepare_case(method, scf_options=options, setup_functional=False)
    functional = _prepare_case(method, scf_options=options)
    guess, layout = _initial_guess(functional)
    _assert_tensor_equal(guess, legacy.x)
    mapped_states: list[torch.Tensor] = []
    legacy_map = legacy.engine.iterate_potential

    def record_map(value: torch.Tensor) -> torch.Tensor:
        mapped = legacy_map(value)
        mapped_states.append(mapped.clone())
        return mapped

    legacy.engine.iterate_potential = record_map
    legacy_result = legacy.engine(legacy.initial_charges.clone())
    config = functional.system.config.scf
    if config.mixer == labels.MIXER_LINEAR:
        settings = setup_simple_solver_settings(config, guess.dtype)
        result = solve_scf_potential_simple(
            guess, functional.inputs, layout, settings
        )
    else:
        settings = setup_anderson_solver_settings(config, guess.dtype)
        result = solve_scf_potential_anderson(
            guess, functional.inputs, layout, settings
        )
    expected = dict(legacy_result)
    expected["mapped_states"] = mapped_states
    return functional, result, expected


@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_potential_simple_finite_budget_matches_legacy(
    maxiter: int, mix_guess: bool
) -> None:
    """Potential Simple retains legacy finite-budget public values."""
    _, result, legacy = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_LINEAR,
            "maxiter": maxiter,
            "mix_guess": mix_guess,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        }
    )
    _assert_public_parity(result, legacy)
    assert legacy["iterations"] == len(legacy["mapped_states"])
    assert len(legacy["mapped_states"]) == maxiter + 1
    _assert_tensor_equal(
        result.map_result.mapped_x, legacy["mapped_states"][-1]
    )
    assert not bool(result.converged)
    assert result.final_evaluation is not result.map_result.spectral


def test_potential_maxiter_zero_runs_one_final_spectrum(monkeypatch) -> None:
    """The final representation conversion is one spectrum, not an SCF step."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 0,
        },
    )
    guess, layout = _initial_guess(case)
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)
    calls = 0
    original = functional_map._transitional_diagonalize

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(functional_map, "_transitional_diagonalize", counted)
    result = solve_scf_potential_simple(guess, case.inputs, layout, settings)
    assert int(result.iterations) == 1
    assert not bool(result.converged)
    assert calls == int(result.iterations) + 1


def test_potential_maxiter_zero_tracks_raw_and_final_fermi_status() -> None:
    """One-shot potential mode tracks both spectra without outer convergence."""
    _, result, _ = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 0,
            "fermi_etemp": 300.0,
        }
    )
    assert not bool(result.converged)
    assert result.trajectory_status is not None
    assert result.map_result.fermi_status is not None
    assert result.final_evaluation.fermi_status is not None
    assert bool(result.trajectory_status.fermi_converged.all())
    assert bool(result.final_evaluation.fermi_status.converged.all())
    assert bool(functional_subsolves_ok(result))
    assert not bool(functional_converged_and_valid(result))


def test_potential_initial_guess_has_no_spectral_evaluation(
    monkeypatch,
) -> None:
    """Initial representation conversion only evaluates interaction data."""
    case = _prepare_case(
        GFN2_XTB,
        scf_options={"scp_mode": labels.SCP_MODE_POTENTIAL},
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("initial potential guess performed an eigensolve")

    monkeypatch.setattr(functional_map, "_transitional_diagonalize", forbidden)
    guess, _ = _initial_guess(case)
    _assert_tensor_equal(guess, case.x)


@pytest.mark.parametrize(
    "mixer,options",
    [
        (
            labels.MIXER_LINEAR,
            {"maxiter": 80, "x_atol": 1e-8, "x_atol_max": 1e-8},
        ),
        (
            labels.MIXER_ANDERSON,
            {
                "maxiter": 12,
                "damp_generations": 2,
                "damp_soft_start": False,
                "x_atol": 1e-8,
                "x_atol_max": 1e-8,
            },
        ),
    ],
    ids=["simple-converged", "anderson-accelerated"],
)
def test_potential_solver_converged_matches_legacy(
    mixer: int, options: dict[str, object]
) -> None:
    """Potential Simple/Anderson finalization matches legacy at convergence."""
    _, result, legacy = _run_legacy_and_functional(
        {"mixer": mixer, "mix_guess": True, **options}
    )
    _assert_public_parity(result, legacy)
    assert bool(result.converged)
    assert legacy["iterations"] == len(legacy["mapped_states"])
    _assert_tensor_equal(
        result.map_result.mapped_x, legacy["mapped_states"][-1]
    )


def test_potential_anderson_gfn2_multipole_finite_budget() -> None:
    """Potential Anderson keeps all GFN2 multipole rows through solving."""
    _, result, legacy = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_ANDERSON,
            "maxiter": 2,
            "mix_guess": False,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
        method=GFN2_XTB,
    )
    _assert_public_parity(result, legacy)
    assert result.final_charges.dipole is not None
    assert result.final_charges.quad is not None
    assert result.final_potential.dipole is not None
    assert result.final_potential.quad is not None


def test_gfn2_potential_padding_is_invariant_under_mixing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PADNZ potential tails have zero residual and survive both mixers."""
    case = _prepare_case(
        GFN2_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "mixer": labels.MIXER_ANDERSON,
            "maxiter": 2,
            "mix_guess": True,
            "damp_generations": 2,
            "damp_soft_start": False,
        },
    )
    guess, layout = _initial_guess(case)
    mapped = apply_potential_map(guess, case.inputs, layout).mapped_x
    row_lengths = [
        prod(layout.mono_shape),
        prod(layout.dipole_shape),
        prod(layout.quadrupole_shape),
    ]
    for row, length in enumerate(row_lengths):
        torch.testing.assert_close(
            guess[row, length:],
            torch.full_like(guess[row, length:], defaults.PADNZ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            mapped[row, length:] - guess[row, length:],
            torch.zeros_like(mapped[row, length:]),
            rtol=0.0,
            atol=0.0,
        )
    simple = simple_mix(guess, mapped, 0.25)
    torch.testing.assert_close(
        simple, guess + 0.25 * (mapped - guess), rtol=0.0, atol=0.0
    )
    for row, length in enumerate(row_lengths):
        torch.testing.assert_close(
            simple[row, length:],
            torch.full_like(simple[row, length:], defaults.PADNZ),
            rtol=0.0,
            atol=0.0,
        )

    mapped_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = functional_map.potential_map

    def record(value: torch.Tensor, inputs: object, layout: object):
        result = original(value, inputs, layout)
        mapped_pairs.append((value.clone(), result.mapped_x.clone()))
        return result

    monkeypatch.setattr(functional_anderson, "potential_map", record)
    settings = setup_anderson_solver_settings(
        case.system.config.scf, guess.dtype
    )
    result = solve_scf_potential_anderson(guess, case.inputs, layout, settings)
    assert result.iterations.item() == 3
    assert len(mapped_pairs) == 3  # initial, warm-up, accelerated transition
    for value, mapped_value in mapped_pairs:
        for row, length in enumerate(row_lengths):
            torch.testing.assert_close(
                value[row, length:],
                torch.full_like(value[row, length:], defaults.PADNZ),
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                mapped_value[row, length:] - value[row, length:],
                torch.zeros_like(mapped_value[row, length:]),
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.parametrize(
    "partition",
    [labels.FERMI_PARTITION_EQUAL, labels.FERMI_PARTITION_ATOMIC],
    ids=["equal", "atomic"],
)
def test_potential_final_finite_temperature_partition(partition: int) -> None:
    """Potential final conversion owns partitioned finite-T free energy."""
    _, result, legacy = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "mix_guess": False,
            "fermi_etemp": 300.0,
            "fermi_partition": partition,
            "x_atol": 1.0,
            "x_atol_max": 1.0,
        }
    )
    _assert_public_parity(result, legacy)
    assert result.final_evaluation.fermi_status is not None
    assert functional_subsolves_ok(result).shape == ()
    assert result.final_evaluation is not result.map_result.spectral

    failed_status = replace(
        result.final_evaluation.fermi_status,
        converged=torch.zeros_like(
            result.final_evaluation.fermi_status.converged
        ),
    )
    failed_final = replace(
        result,
        final_evaluation=replace(
            result.final_evaluation, fermi_status=failed_status
        ),
    )
    assert bool(result.map_result.fermi_status.converged.all())
    assert bool(result.converged)
    assert not bool(functional_subsolves_ok(failed_final))


def test_potential_trajectory_failure_survives_valid_final_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid physical map remains visible after valid later spectra."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "mix_guess": False,
            "fermi_etemp": 300.0,
            "x_atol": 1.0e6,
            "x_atol_max": 1.0e6,
        },
    )
    guess, layout = _initial_guess(case)
    original = functional_solver.potential_map
    map_calls = 0

    def fail_initial_map(value: torch.Tensor, inputs: object, layout_: object):
        nonlocal map_calls
        result = original(value, inputs, layout_)
        map_calls += 1
        if map_calls == 1:
            assert result.fermi_status is not None
            failed_status = replace(
                result.fermi_status,
                converged=torch.zeros_like(result.fermi_status.converged),
                electron_count_ok=torch.zeros_like(
                    result.fermi_status.electron_count_ok
                ),
            )
            result = replace(
                result,
                spectral=replace(result.spectral, fermi_status=failed_status),
            )
        return result

    monkeypatch.setattr(functional_solver, "potential_map", fail_initial_map)
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)
    result = solve_scf_potential_simple(guess, case.inputs, layout, settings)
    assert map_calls == 2
    assert result.map_result.fermi_status is not None
    assert result.final_evaluation.fermi_status is not None
    assert bool(result.map_result.fermi_status.converged.all())
    assert bool(result.final_evaluation.fermi_status.converged.all())
    assert result.trajectory_status is not None
    assert not bool(result.trajectory_status.fermi_converged.all())
    assert not bool(result.trajectory_status.electron_count_ok.all())
    assert not bool(functional_subsolves_ok(result))


def test_potential_solver_vmap_over_fixed_system_guesses() -> None:
    """Potential solver maps independent guesses for one fixed System."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    guess, layout = _initial_guess(case)
    guesses = torch.stack((guess, guess.clone(), guess.clone()))
    guesses[1, 0, 0] += 0.01
    guesses[2, 0, 0] -= 0.015
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)

    def solve(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = solve_scf_potential_simple(
            value, case.inputs, layout, settings
        )
        return result.energy, result.iterations

    mapped = vmap(solve)(guesses)
    loop = tuple(
        torch.stack(values) for values in zip(*(solve(g) for g in guesses))
    )
    _assert_tensor_equal(mapped[0], loop[0])
    _assert_tensor_equal(mapped[1], loop[1])


def test_potential_atomic_partition_vmap() -> None:
    """Atomic entropy partition maps over guesses for one finite-T System."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "fermi_etemp": 300.0,
            "fermi_partition": labels.FERMI_PARTITION_ATOMIC,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    guess, layout = _initial_guess(case)
    guesses = torch.stack((guess, guess.clone()))
    guesses[1, 0, 0] += 0.01
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)

    def solve(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = solve_scf_potential_simple(
            value, case.inputs, layout, settings
        )
        return result.energy + result.fenergy, result.iterations

    mapped = vmap(solve)(guesses)
    loop = tuple(
        torch.stack(values) for values in zip(*(solve(g) for g in guesses))
    )
    _assert_tensor_equal(mapped[0], loop[0])
    _assert_tensor_equal(mapped[1], loop[1])


def test_potential_position_parameter_and_field_gradients() -> None:
    """Potential solving preserves geometry, parameter, and field graphs."""
    options = {
        "scp_mode": labels.SCP_MODE_POTENTIAL,
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 1,
        "mix_guess": False,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    positions = _prepare_case(GFN1_XTB).positions.clone().requires_grad_(True)
    functional = _prepare_case(
        GFN1_XTB, positions=positions, scf_options=options
    )
    guess, layout = _initial_guess(functional)
    settings = setup_simple_solver_settings(
        functional.system.config.scf, positions.dtype
    )
    energy = solve_scf_potential_simple(
        guess, functional.inputs, layout, settings
    ).energy.sum()
    gradient = torch.autograd.grad(energy, positions, create_graph=True)[0]
    hessian_projection = torch.autograd.grad(
        gradient.square().sum(), positions
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian_projection).all()

    def energy_at_positions(value: torch.Tensor) -> torch.Tensor:
        case = _prepare_case(GFN1_XTB, positions=value, scf_options=options)
        initial, initial_layout = _initial_guess(case)
        local_settings = setup_simple_solver_settings(
            case.system.config.scf, value.dtype
        )
        return solve_scf_potential_simple(
            initial, case.inputs, initial_layout, local_settings
        ).energy.sum()

    _assert_directional_fd(energy_at_positions, positions, gradient)

    par = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    leaf = par.get("element.O.levels")
    leaf.requires_grad_(True)
    finite_options = {**options, "fermi_etemp": 300.0}
    parameter_case = _prepare_case(
        GFN1_XTB, par=par, scf_options=finite_options
    )
    parameter_guess, parameter_layout = _initial_guess(parameter_case)
    parameter_settings = setup_simple_solver_settings(
        parameter_case.system.config.scf, torch.float64
    )
    parameter_result = solve_scf_potential_simple(
        parameter_guess,
        parameter_case.inputs,
        parameter_layout,
        parameter_settings,
    )
    parameter_total = (parameter_result.energy + parameter_result.fenergy).sum()
    parameter_gradient = torch.autograd.grad(parameter_total, leaf)[0]
    assert torch.isfinite(parameter_gradient).all()
    assert torch.count_nonzero(parameter_gradient) > 0

    field = torch.tensor(
        [0.001, -0.0007, 0.0004], dtype=torch.float64, requires_grad=True
    )
    field_case = _prepare_case(
        GFN1_XTB, field=field, scf_options=finite_options
    )
    field_guess, field_layout = _initial_guess(field_case)
    field_settings = setup_simple_solver_settings(
        field_case.system.config.scf, torch.float64
    )
    field_result = solve_scf_potential_simple(
        field_guess, field_case.inputs, field_layout, field_settings
    )
    field_total = (field_result.energy + field_result.fenergy).sum()
    field_gradient = torch.autograd.grad(field_total, field)[0]
    assert torch.isfinite(field_gradient).all()
    assert torch.count_nonzero(field_gradient) > 0

    def total_at_field(value: torch.Tensor) -> torch.Tensor:
        case = _prepare_case(GFN1_XTB, field=value, scf_options=finite_options)
        initial, initial_layout = _initial_guess(case)
        local_settings = setup_simple_solver_settings(
            case.system.config.scf, value.dtype
        )
        result = solve_scf_potential_simple(
            initial, case.inputs, initial_layout, local_settings
        )
        return (result.energy + result.fenergy).sum()

    _assert_directional_fd(total_at_field, field, field_gradient)


@pytest.mark.parametrize(
    "temperature",
    [300.0, 0.5 * 3.0e-7 / KELVIN2AU],
    ids=["smearing", "positive-subcutoff"],
)
def test_potential_fractional_charge_gradient_at_finite_temperature(
    temperature: float,
) -> None:
    """Potential final-spectrum free energy preserves fractional-charge AD."""
    charge = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    options = {
        "scp_mode": labels.SCP_MODE_POTENTIAL,
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 1,
        "mix_guess": False,
        "fermi_etemp": temperature,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    functional = _prepare_case(GFN1_XTB, charge=charge, scf_options=options)
    guess, layout = _initial_guess(functional)
    settings = setup_simple_solver_settings(
        functional.system.config.scf, torch.float64
    )
    total = solve_scf_potential_simple(
        guess, functional.inputs, layout, settings
    )
    if temperature < 3.0e-7 / KELVIN2AU:
        assert torch.count_nonzero(total.fenergy) > 0
    functional_gradient = torch.autograd.grad(
        (total.energy + total.fenergy).sum(), charge
    )[0]

    def total_at_charge(value: torch.Tensor) -> torch.Tensor:
        case = _prepare_case(GFN1_XTB, charge=value, scf_options=options)
        initial, initial_layout = _initial_guess(case)
        local_settings = setup_simple_solver_settings(
            case.system.config.scf, torch.float64
        )
        result = solve_scf_potential_simple(
            initial, case.inputs, initial_layout, local_settings
        )
        return (result.energy + result.fenergy).sum()

    _assert_directional_fd(
        total_at_charge,
        charge,
        functional_gradient,
        direction=torch.ones_like(charge),
        one_sided=-1,
    )


def test_potential_map_reverse_jvp_and_jacfwd() -> None:
    """Potential map packing and spectral outputs retain derivative graphs."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={"scp_mode": labels.SCP_MODE_POTENTIAL},
    )
    guess, layout = _initial_guess(case)

    def mapped(value: torch.Tensor) -> torch.Tensor:
        return apply_potential_map(value, case.inputs, layout).mapped_x

    reverse = torch.autograd.grad(
        mapped(guess.requires_grad_(True)).sum(), guess
    )[0]
    primal, tangent = torch.func.jvp(
        mapped, (guess,), (torch.ones_like(guess),)
    )
    forward_jacobian = torch.func.jacfwd(mapped)(guess)
    assert torch.isfinite(reverse).all()
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(forward_jacobian).all()
    assert torch.isfinite(primal).all()

    from dxtb._src.scf.functional.data import unpack_potential
    from dxtb._src.scf.functional.map import potential_to_hamiltonian

    direction = torch.linspace(
        -0.7, 0.9, guess.numel(), dtype=guess.dtype
    ).reshape_as(guess)
    weights = torch.linspace(
        0.3,
        1.1,
        case.inputs.integrals.hcore.numel(),
        dtype=guess.dtype,
    ).reshape_as(case.inputs.integrals.hcore)

    def smooth_spectral_input(value: torch.Tensor) -> torch.Tensor:
        potential = unpack_potential(value, layout)
        hamiltonian = potential_to_hamiltonian(
            case.inputs.integrals.hcore,
            case.inputs.integrals.overlap,
            case.inputs.integrals.dipole,
            case.inputs.integrals.quadrupole,
            case.inputs.index,
            potential,
        )
        return (hamiltonian * weights).sum()

    smooth_input = guess.detach().requires_grad_(True)
    reverse_gradient = torch.autograd.grad(
        smooth_spectral_input(smooth_input), smooth_input
    )[0]
    _, smooth_tangent = torch.func.jvp(
        smooth_spectral_input, (guess,), (direction,)
    )
    smooth_jacfwd = torch.func.jacfwd(smooth_spectral_input)(guess)
    _assert_tensor_equal(smooth_tangent, (reverse_gradient * direction).sum())
    _assert_tensor_equal(smooth_jacfwd, reverse_gradient)
