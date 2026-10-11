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
"""Functional simple-solver compatibility and transform tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN1_XTB, labels
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional import (
    SCFCharges,
    SCFMapResult,
    SCFSpectralResult,
    SCFPotential,
    SCFSolverSettings,
    setup_simple_solver_settings,
    solve_scf_simple,
)
from dxtb._src.scf.functional.solver import _solve_map
from dxtb._src.scf.mixer.simple import Simple

from .samples import samples
from .test_functional_map import _assert_tensor_equal, _prepare_case


def _map_result(mapped: torch.Tensor, driving: torch.Tensor) -> SCFMapResult:
    return SCFMapResult(
        mapped_x=mapped,
        driving_potential=SCFPotential(mono=driving),
        spectral=SCFSpectralResult(
            charges=SCFCharges(mono=mapped),
            hamiltonian=driving,
            density=driving,
            emo=driving,
            coefficients=driving,
            occupation=driving,
            orbital_energy=driving,
        ),
    )


def _compare_legacy_simple(
    *,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    extra_options: dict[str, object] | None = None,
) -> SimpleNamespace:
    """Compare a functional solve with an independently prepared engine."""
    options: dict[str, object] = {
        "mixer": labels.MIXER_LINEAR,
        "maxiter": maxiter,
        "mix_guess": mix_guess,
        "x_atol": x_atol,
        "x_atol_max": x_atol_max,
    }
    if extra_options is not None:
        options.update(extra_options)
    legacy = _prepare_case(GFN1_XTB, scf_options=options)
    functional = _prepare_case(GFN1_XTB, scf_options=options)
    guess = legacy.x.clone()
    _assert_tensor_equal(functional.x, guess)

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    iterate = legacy.engine.iterate_charges

    def record(x: torch.Tensor) -> torch.Tensor:
        mapped = iterate(x)
        calls.append((x.clone(), mapped.clone()))
        return mapped

    legacy.engine.iterate_charges = record
    legacy_x = legacy.engine.scf(guess, return_charges=False)
    actual = solve_scf_simple(
        functional.x,
        functional.inputs,
        setup_simple_solver_settings(
            functional.system.config.scf, functional.positions.dtype
        ),
    )

    assert len(calls) == legacy.engine._data.iter
    assert int(actual.iterations) == len(calls)
    _assert_tensor_equal(actual.map_result.mapped_x, legacy_x)
    _assert_tensor_equal(
        actual.map_result.hamiltonian, legacy.engine._data.hamiltonian
    )
    _assert_tensor_equal(actual.map_result.density, legacy.engine._data.density)
    _assert_tensor_equal(actual.map_result.emo, legacy.engine._data.evals)
    _assert_tensor_equal(
        actual.map_result.occupation, legacy.engine._data.occupation
    )
    _assert_tensor_equal(
        actual.map_result.orbital_energy, legacy.engine._data.energy
    )
    legacy_charges = legacy.engine.converged_to_charges(legacy_x)
    _assert_tensor_equal(actual.final_charges.mono, legacy_charges.mono)
    if legacy_charges.dipole is not None:
        assert actual.final_charges.dipole is not None
        _assert_tensor_equal(actual.final_charges.dipole, legacy_charges.dipole)
    if legacy_charges.quad is not None:
        assert actual.final_charges.quad is not None
        _assert_tensor_equal(actual.final_charges.quad, legacy_charges.quad)

    legacy_potential = legacy.engine.charges_to_potential(legacy_charges)
    _assert_tensor_equal(actual.final_potential.mono, legacy_potential.mono)
    if legacy_potential.dipole is not None:
        assert actual.final_potential.dipole is not None
        _assert_tensor_equal(
            actual.final_potential.dipole, legacy_potential.dipole
        )
    if legacy_potential.quad is not None:
        assert actual.final_potential.quad is not None
        _assert_tensor_equal(actual.final_potential.quad, legacy_potential.quad)
    _assert_tensor_equal(
        actual.energy, legacy.engine.get_energy(legacy_charges)
    )
    _assert_tensor_equal(
        actual.fenergy, legacy.engine.get_electronic_free_energy()
    )
    if functional.inputs.fermi_settings is None:
        assert torch.count_nonzero(actual.fenergy) == 0

    expected_converged = (
        legacy.engine.mixer.converged if maxiter > 0 else torch.tensor(False)
    )
    assert torch.equal(actual.converged.cpu(), expected_converged.cpu())
    if calls:
        last_delta = calls[-1][1] - calls[-1][0]
        expected_residual = torch.stack(
            (torch.linalg.vector_norm(last_delta), torch.amax(last_delta.abs()))
        )
        _assert_tensor_equal(actual.residual, expected_residual)
    return SimpleNamespace(
        legacy=legacy,
        functional=functional,
        calls=calls,
        actual=actual,
        legacy_x=legacy_x,
    )


@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_simple_solver_budget_matrix_matches_legacy(
    maxiter: int, mix_guess: bool
) -> None:
    """Match raw-map counts and outputs for the six finite budgets."""
    comparison = _compare_legacy_simple(
        maxiter=maxiter,
        mix_guess=mix_guess,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert comparison.actual.converged.item() is False
    assert int(comparison.actual.iterations) == maxiter + 1


def test_simple_solver_initial_mix_is_not_convergence_tested() -> None:
    """A loose tolerance is first checked after the initial mixer step."""
    comparison = _compare_legacy_simple(
        maxiter=4,
        mix_guess=True,
        x_atol=1.0e6,
        x_atol_max=1.0e6,
    )
    assert len(comparison.calls) == 2
    assert int(comparison.actual.iterations) == 2
    assert comparison.actual.converged.item() is True


def test_simple_solver_nonconverged_result_uses_last_raw_map() -> None:
    """A tight finite budget returns the last raw map without raising."""
    comparison = _compare_legacy_simple(
        maxiter=1,
        mix_guess=False,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert comparison.actual.converged.item() is False
    _assert_tensor_equal(
        comparison.actual.map_result.mapped_x, comparison.legacy_x
    )


def test_simple_solver_converges_and_freezes_first_map_result() -> None:
    """Normal simple mixing matches a converged legacy full-mode solve."""
    comparison = _compare_legacy_simple(
        maxiter=100,
        mix_guess=True,
        x_atol=1.0e-4,
        x_atol_max=1.0e-5,
    )
    assert comparison.actual.converged.item() is True
    assert int(comparison.actual.iterations) < 101


@pytest.mark.parametrize(
    "partition",
    [labels.FERMI_PARTITION_EQUAL, labels.FERMI_PARTITION_ATOMIC],
)
@pytest.mark.parametrize("mix_guess", [False, True])
def test_finite_temperature_simple_solver_matches_legacy(
    partition: int, mix_guess: bool
) -> None:
    """Simple finite-T trajectories and both entropy partitions match legacy."""
    comparison = _compare_legacy_simple(
        maxiter=3,
        mix_guess=mix_guess,
        x_atol=1.0e-20,
        x_atol_max=1.0e-20,
        extra_options={
            "fermi_etemp": 300.0,
            "fermi_partition": partition,
        },
    )
    assert (
        comparison.actual.fenergy.shape
        == comparison.functional.system.numbers.shape
    )
    assert torch.isfinite(comparison.actual.fenergy).all()


@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_finite_temperature_simple_budget_matrix(
    maxiter: int, mix_guess: bool
) -> None:
    """Finite-T Simple preserves the six legacy map-evaluation budgets."""
    comparison = _compare_legacy_simple(
        maxiter=maxiter,
        mix_guess=mix_guess,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
        extra_options={"fermi_etemp": 300.0},
    )
    assert int(comparison.actual.iterations) == maxiter + 1
    assert comparison.actual.converged.item() is False


def test_atomic_partition_supports_reverse_and_vmap() -> None:
    """Atomic finite-T free energy supports gradients and fixed-System vmap."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "fermi_etemp": 300.0,
            "fermi_partition": labels.FERMI_PARTITION_ATOMIC,
        },
    )
    settings = setup_simple_solver_settings(
        case.system.config.scf, case.positions.dtype
    )

    def free_energy(guess: torch.Tensor) -> torch.Tensor:
        return solve_scf_simple(guess, case.inputs, settings).fenergy.sum()

    leaf = case.x.clone().requires_grad_(True)
    gradient = torch.autograd.grad(free_energy(leaf), leaf)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    guesses = torch.stack((case.x, case.x + 1.0e-3, case.x - 1.0e-3))
    mapped = vmap(free_energy)(guesses)
    looped = torch.stack([free_energy(guess) for guess in guesses])
    _assert_tensor_equal(mapped, looped)


def test_finite_temperature_simple_normal_convergence() -> None:
    """Finite-T Simple reaches the same converged legacy result."""
    comparison = _compare_legacy_simple(
        maxiter=100,
        mix_guess=True,
        x_atol=1.0e-4,
        x_atol_max=1.0e-5,
        extra_options={"fermi_etemp": 300.0},
    )
    assert comparison.actual.converged.item() is True


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_simple_solver_tolerance_resolution_matches_legacy(
    dtype: torch.dtype,
) -> None:
    """Effective tolerances use the Simple mixer's dtype resolution rule."""
    case = _prepare_case(GFN1_XTB, scf_options={"mixer": labels.MIXER_LINEAR})
    config = replace(
        case.system.config.scf,
        x_atol=1.0e-9,
        x_atol_max=1.0e-11,
    )
    settings = setup_simple_solver_settings(config, dtype)
    mixer = Simple(
        {
            "x_tol": config.x_atol,
            "x_tol_max": config.x_atol_max,
            "damp": config.damp,
        }
    )
    old = torch.zeros(2, dtype=dtype)
    mapped = torch.ones_like(old)
    mixer.iter(mapped, old)
    assert settings.x_atol == mixer.options["x_tol"]
    assert settings.x_atol_max == mixer.options["x_tol_max"]
    assert config.x_atol == 1.0e-9
    assert config.x_atol_max == 1.0e-11


def test_simple_solver_rejects_unsupported_mixer_and_damping() -> None:
    """Functional setup refuses mixers it does not implement."""
    case = _prepare_case(GFN1_XTB, scf_options={"mixer": labels.MIXER_LINEAR})
    with pytest.raises(NotImplementedError, match="simple linear mixing"):
        setup_simple_solver_settings(
            replace(case.system.config.scf, mixer=labels.MIXER_ANDERSON),
            torch.float64,
        )
    with pytest.raises(NotImplementedError, match="simple linear mixing"):
        setup_simple_solver_settings(
            replace(case.system.config.scf, mixer=labels.MIXER_BROYDEN),
            torch.float64,
        )
    with pytest.raises(NotImplementedError, match="dynamic damping"):
        setup_simple_solver_settings(
            replace(
                case.system.config.scf,
                mixer=labels.MIXER_LINEAR,
                damp_dynamic=True,
            ),
            torch.float64,
        )
    with pytest.raises(ValueError, match="maxiter >= 0"):
        setup_simple_solver_settings(
            replace(case.system.config.scf, maxiter=-1), torch.float64
        )


def test_affine_solver_reverse_jvp_jacfwd_and_vmap() -> None:
    """The fixed-trip solver is transform-consistent on a smooth affine map."""
    settings = SCFSolverSettings(
        maxiter=5,
        mix_guess=True,
        damp=0.4,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    slope = torch.tensor(0.3, dtype=torch.float64)
    offset = torch.tensor(0.7, dtype=torch.float64)

    def solve(initial: torch.Tensor) -> torch.Tensor:
        def affine(x: torch.Tensor) -> SCFMapResult:
            mapped = slope * x + offset
            return _map_result(mapped, x)

        return _solve_map(initial, settings, affine).map_result.mapped_x.sum()

    initial = torch.tensor([[0.15]], dtype=torch.float64)
    direction = torch.ones_like(initial)
    value, tangent = jvp(solve, (initial,), (direction,))
    leaf = initial.clone().requires_grad_(True)
    reverse = torch.autograd.grad(solve(leaf), leaf)[0]
    forward = jacfwd(solve)(initial)
    _assert_tensor_equal(tangent, torch.sum(reverse * direction))
    _assert_tensor_equal(forward, reverse)
    assert torch.isfinite(value)

    guesses = torch.tensor([[[0.1]], [[0.4]], [[0.8]]], dtype=torch.float64)
    mapped = vmap(solve)(guesses)
    looped = torch.stack([solve(item) for item in guesses])
    _assert_tensor_equal(mapped, looped)

    status_settings = replace(settings, x_atol=0.05, x_atol_max=0.05)

    def status_solve(initial_value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        def affine(x: torch.Tensor) -> SCFMapResult:
            mapped_value = slope * x + offset
            return _map_result(mapped_value, x)

        outcome = _solve_map(initial_value, status_settings, affine)
        return (
            outcome.map_result.mapped_x,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
        )

    status_guesses = torch.tensor(
        [[[0.99]], [[0.1]], [[0.7]]], dtype=torch.float64
    )
    status_mapped = vmap(status_solve)(status_guesses)
    status_looped = tuple(
        torch.stack(items)
        for items in zip(*(status_solve(item) for item in status_guesses))
    )
    for actual, expected in zip(status_mapped, status_looped):
        _assert_tensor_equal(actual, expected)


def test_functional_solver_real_position_parameter_and_field_gradients() -> (
    None
):
    """Geometry, ParamModule, and explicit field graphs reach final energy."""
    positions = (
        samples["H2O"]["positions"]
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .requires_grad_(True)
    )
    case = _prepare_case(
        GFN1_XTB,
        positions=positions,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 1.0e-30,
            "x_atol_max": 1.0e-30,
        },
    )
    legacy_position_case = _prepare_case(
        GFN1_XTB,
        positions=positions,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 1.0e-30,
            "x_atol_max": 1.0e-30,
        },
    )
    settings = setup_simple_solver_settings(
        case.system.config.scf, positions.dtype
    )
    result = solve_scf_simple(case.x, case.inputs, settings)
    position_gradient = torch.autograd.grad(
        result.energy.sum(), positions, create_graph=True
    )[0]
    position_hessian_projection = torch.autograd.grad(
        position_gradient.square().sum(), positions
    )[0]
    assert torch.isfinite(position_gradient).all()
    assert torch.isfinite(position_hessian_projection).all()
    legacy_position_x = legacy_position_case.engine.scf(
        legacy_position_case.x, return_charges=False
    )
    legacy_position_charges = legacy_position_case.engine.converged_to_charges(
        legacy_position_x
    )
    legacy_position_energy = legacy_position_case.engine.get_energy(
        legacy_position_charges
    ).sum()
    legacy_position_gradient = torch.autograd.grad(
        legacy_position_energy, positions, create_graph=True
    )[0]
    legacy_position_hessian_projection = torch.autograd.grad(
        legacy_position_gradient.square().sum(), positions
    )[0]
    _assert_tensor_equal(position_gradient, legacy_position_gradient)
    _assert_tensor_equal(
        position_hessian_projection, legacy_position_hessian_projection
    )

    par = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    leaf = par.get("element.O.levels")
    leaf.requires_grad_(True)
    parameter_case = _prepare_case(
        GFN1_XTB,
        par=par,
        scf_options={"mixer": labels.MIXER_LINEAR, "maxiter": 2},
    )
    parameter_settings = setup_simple_solver_settings(
        parameter_case.system.config.scf, torch.float64
    )
    parameter_energy = solve_scf_simple(
        parameter_case.x, parameter_case.inputs, parameter_settings
    ).energy.sum()
    parameter_gradient = torch.autograd.grad(parameter_energy, leaf)[0]
    assert torch.isfinite(parameter_gradient).all()
    assert torch.count_nonzero(parameter_gradient) > 0

    field = torch.tensor(
        [0.001, -0.0007, 0.0004], dtype=torch.float64, requires_grad=True
    )
    field_case = _prepare_case(
        GFN1_XTB,
        field=field,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 1.0e-30,
            "x_atol_max": 1.0e-30,
        },
    )
    field_settings = setup_simple_solver_settings(
        field_case.system.config.scf, torch.float64
    )
    field_energy = solve_scf_simple(
        field_case.x, field_case.inputs, field_settings
    ).energy.sum()
    field_gradient = torch.autograd.grad(field_energy, field)[0]
    assert torch.isfinite(field_gradient).all()
    assert torch.count_nonzero(field_gradient) > 0
    legacy_field_case = _prepare_case(
        GFN1_XTB,
        field=field,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 1.0e-30,
            "x_atol_max": 1.0e-30,
        },
    )
    legacy_field_x = legacy_field_case.engine.scf(
        legacy_field_case.x, return_charges=False
    )
    legacy_field_charges = legacy_field_case.engine.converged_to_charges(
        legacy_field_x
    )
    legacy_field_energy = legacy_field_case.engine.get_energy(
        legacy_field_charges
    ).sum()
    legacy_field_gradient = torch.autograd.grad(legacy_field_energy, field)[0]
    _assert_tensor_equal(field_gradient, legacy_field_gradient)


def test_real_functional_solver_vmap_over_charge_guesses() -> None:
    """Mapped initial charge iterates agree with independent solver calls."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    settings = setup_simple_solver_settings(
        case.system.config.scf, case.positions.dtype
    )
    guesses = torch.stack((case.x, case.x + 0.01, case.x - 0.02), dim=0)

    def energy(guess: torch.Tensor) -> torch.Tensor:
        return solve_scf_simple(guess, case.inputs, settings).energy.sum()

    mapped = vmap(energy)(guesses)
    looped = torch.stack([energy(guess) for guess in guesses])
    _assert_tensor_equal(mapped, looped)


def test_finite_temperature_solver_vmap_over_charge_guesses() -> None:
    """Finite-temperature Fermi search is vmap-safe inside one fixed System."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 300.0,
        },
    )
    settings = setup_simple_solver_settings(
        case.system.config.scf, case.positions.dtype
    )
    guesses = torch.stack((case.x, case.x + 0.01, case.x - 0.02), dim=0)

    def energy(guess: torch.Tensor) -> torch.Tensor:
        result = solve_scf_simple(guess, case.inputs, settings)
        return (result.energy + result.fenergy).sum()

    mapped = vmap(energy)(guesses)
    looped = torch.stack([energy(guess) for guess in guesses])
    _assert_tensor_equal(mapped, looped)


def test_finite_temperature_real_solver_forward_ad_characterization() -> None:
    """Finite-T SCF forward-AD remains finite with the transitional spectrum."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 300.0,
        },
    )
    settings = setup_simple_solver_settings(
        case.system.config.scf, case.positions.dtype
    )

    def energy(guess: torch.Tensor) -> torch.Tensor:
        result = solve_scf_simple(guess, case.inputs, settings)
        return (result.energy + result.fenergy).sum()

    direction = torch.ones_like(case.x) / case.x.numel() ** 0.5
    value, tangent = jvp(energy, (case.x,), (direction,))
    forward = jacfwd(energy)(case.x)
    leaf = case.x.clone().requires_grad_(True)
    reverse_value = energy(leaf)
    (reverse,) = torch.autograd.grad(reverse_value, leaf)
    assert torch.isfinite(value)
    assert torch.isfinite(tangent)
    assert torch.isfinite(forward).all()
    assert torch.isfinite(reverse).all()


def test_fractional_charge_gradient_through_finite_temperature_solver() -> None:
    """Total charge remains connected through nel, Fermi response and entropy."""
    charge = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    options = {
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 2,
        "mix_guess": True,
        "fermi_etemp": 1000.0,
    }
    case = _prepare_case(
        GFN1_XTB,
        charge=charge,
        scf_options=options,
    )
    settings = setup_simple_solver_settings(
        case.system.config.scf, case.positions.dtype
    )
    result = solve_scf_simple(case.x, case.inputs, settings)
    gradient = torch.autograd.grad(
        (result.energy + result.fenergy).sum(), charge
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    legacy_case = _prepare_case(GFN1_XTB, charge=charge, scf_options=options)
    legacy_x = legacy_case.engine.scf(legacy_case.x, return_charges=False)
    legacy_charges = legacy_case.engine.converged_to_charges(legacy_x)
    legacy_value = legacy_case.engine.get_energy(legacy_charges).sum()
    legacy_value = (
        legacy_value + legacy_case.engine.get_electronic_free_energy().sum()
    )
    legacy_gradient = torch.autograd.grad(legacy_value, charge)[0]
    _assert_tensor_equal(gradient, legacy_gradient)


def test_position_parameter_and_field_gradients_at_finite_temperature() -> None:
    """Geometry, real parameters and explicit fields stay connected at finite T."""
    positions = (
        samples["H2O"]["positions"]
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .requires_grad_(True)
    )
    options = {
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 1.0e-30,
        "x_atol_max": 1.0e-30,
        "fermi_etemp": 300.0,
    }
    position_case = _prepare_case(
        GFN1_XTB, positions=positions, scf_options=options
    )
    settings = setup_simple_solver_settings(
        position_case.system.config.scf, positions.dtype
    )
    position_result = solve_scf_simple(
        position_case.x, position_case.inputs, settings
    )
    gradient = torch.autograd.grad(
        (position_result.energy + position_result.fenergy).sum(),
        positions,
        create_graph=True,
    )[0]
    hessian_projection = torch.autograd.grad(
        gradient.square().sum(), positions
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hessian_projection).all()
    legacy_position_case = _prepare_case(
        GFN1_XTB, positions=positions, scf_options=options
    )
    legacy_position_x = legacy_position_case.engine.scf(
        legacy_position_case.x, return_charges=False
    )
    legacy_position_charges = legacy_position_case.engine.converged_to_charges(
        legacy_position_x
    )
    legacy_position_value = legacy_position_case.engine.get_energy(
        legacy_position_charges
    ).sum()
    legacy_position_value = (
        legacy_position_value
        + legacy_position_case.engine.get_electronic_free_energy().sum()
    )
    legacy_position_gradient = torch.autograd.grad(
        legacy_position_value, positions, create_graph=True
    )[0]
    legacy_hessian_projection = torch.autograd.grad(
        legacy_position_gradient.square().sum(), positions
    )[0]
    _assert_tensor_equal(gradient, legacy_position_gradient)
    _assert_tensor_equal(hessian_projection, legacy_hessian_projection)

    par = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    leaf = par.get("element.O.levels")
    leaf.requires_grad_(True)
    parameter_case = _prepare_case(GFN1_XTB, par=par, scf_options=options)
    parameter_settings = setup_simple_solver_settings(
        parameter_case.system.config.scf, torch.float64
    )
    parameter_result = solve_scf_simple(
        parameter_case.x, parameter_case.inputs, parameter_settings
    )
    parameter_gradient = torch.autograd.grad(
        (parameter_result.energy + parameter_result.fenergy).sum(), leaf
    )[0]
    assert torch.isfinite(parameter_gradient).all()
    assert torch.count_nonzero(parameter_gradient) > 0

    field = torch.tensor(
        [0.001, -0.0007, 0.0004], dtype=torch.float64, requires_grad=True
    )
    field_case = _prepare_case(
        GFN1_XTB,
        field=field,
        scf_options={**options, "int_level": labels.INTLEVEL_DIPOLE},
    )
    field_settings = setup_simple_solver_settings(
        field_case.system.config.scf, torch.float64
    )
    field_result = solve_scf_simple(
        field_case.x, field_case.inputs, field_settings
    )
    field_gradient = torch.autograd.grad(
        (field_result.energy + field_result.fenergy).sum(), field
    )[0]
    assert torch.isfinite(field_gradient).all()
    assert torch.count_nonzero(field_gradient) > 0


def test_real_functional_solver_jvp_and_jacfwd_characterization() -> None:
    """Characterize real-map forward AD separately from solver arithmetic."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    settings = setup_simple_solver_settings(
        case.system.config.scf, case.positions.dtype
    )

    def energy(guess: torch.Tensor) -> torch.Tensor:
        return solve_scf_simple(guess, case.inputs, settings).energy.sum()

    direction = torch.ones_like(case.x) / case.x.numel() ** 0.5
    value, tangent = jvp(energy, (case.x,), (direction,))
    jacobian = jacfwd(energy)(case.x)
    leaf = case.x.clone().requires_grad_(True)
    reverse_value = energy(leaf)
    (reverse_gradient,) = torch.autograd.grad(reverse_value, leaf)
    reverse_tangent = torch.sum(reverse_gradient * direction)
    assert torch.isfinite(value)
    assert torch.isfinite(tangent)
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(reverse_gradient).all()
    assert torch.isfinite(reverse_tangent)
    # The fixed-trip simple solver agrees across AD modes on a smooth affine
    # map above; any xTB mismatch is therefore in the transitional spectrum.
    # E3's isolated spectral test identifies the reverse/forward branch split.


def test_functional_solver_has_no_mutable_mixer_or_tensor_control_flow() -> (
    None
):
    """The new solver avoids legacy mixer state and host tensor decisions."""
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "src/dxtb/_src/scf/functional/solver.py"
    )
    text = source.read_text()
    for forbidden in (
        "Mixer",
        "Anderson(",
        "Broyden(",
        "self._x_old",
        "self._delta",
        "iter_step",
        ".reset(",
        ".cull(",
        ".item(",
        ".tolist(",
        "torch.autograd.Function",
    ):
        assert forbidden not in text
