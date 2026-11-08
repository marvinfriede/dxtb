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
    SCFPotential,
    SCFSolverSettings,
    SCFSpectralResult,
    setup_solver_settings,
    simple_mix,
    solve_scf_simple,
)
from dxtb._src.scf.functional.solver import _solve_map

from ..utils import scf_strategy
from .samples import samples
from .test_functional_map import (
    _assert_directional_fd,
    _assert_tensor_equal,
    _prepare_case,
)


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


def _run_simple_solver(
    *,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    extra_options: dict[str, object] | None = None,
) -> SimpleNamespace:
    """Run a functional simple-solver configuration and verify core invariants."""
    options: dict[str, object] = {
        "strategy": scf_strategy("linear", "charge"),
        "maxiter": maxiter,
        "mix_guess": mix_guess,
        "x_atol": x_atol,
        "x_atol_max": x_atol_max,
    }
    if extra_options is not None:
        options.update(extra_options)
    functional = _prepare_case(GFN1_XTB, scf_options=options)
    actual = solve_scf_simple(
        functional.x,
        functional.inputs,
        setup_solver_settings(
            functional.system.config.scf, functional.positions.dtype
        ),
    )
    if functional.inputs.fermi_settings is None:
        assert torch.count_nonzero(actual.fenergy) == 0
    return SimpleNamespace(
        functional=functional,
        actual=actual,
    )


@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_simple_solver_budget_matrix(
    maxiter: int, mix_guess: bool
) -> None:
    """Verify raw-map counts and outputs for the six finite budgets."""
    comparison = _run_simple_solver(
        maxiter=maxiter,
        mix_guess=mix_guess,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert comparison.actual.converged.item() is False
    assert int(comparison.actual.iterations) == maxiter + 1


def test_simple_solver_initial_mix_is_not_convergence_tested() -> None:
    """The initial raw map seeds mix_guess but is not a root candidate."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 4,
            "mix_guess": True,
            "x_atol": 1.0e6,
            "x_atol_max": 1.0e6,
        },
    )
    result = solve_scf_simple(
        case.x,
        case.inputs,
        SCFSolverSettings(4, True, 0.2, 1.0e6, 1.0e6),
        record_trace=True,
    )
    assert result.trace is not None
    expected_first_iterate = simple_mix(
        case.x, result.trace.raw_mapped[0], case.x.new_tensor(0.2)
    )
    torch.testing.assert_close(
        result.trace.map_inputs[1], expected_first_iterate
    )
    assert int(result.iterations) > 1


def test_simple_solver_nonconverged_result_uses_last_raw_map() -> None:
    """A tight finite budget returns the last raw map without raising."""
    comparison = _run_simple_solver(
        maxiter=1,
        mix_guess=False,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert comparison.actual.converged.item() is False
    assert int(comparison.actual.iterations) == 2


def test_simple_solver_converges_and_freezes_first_map_result() -> None:
    """Normal simple mixing converges before the iteration budget."""
    comparison = _run_simple_solver(
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
def test_finite_temperature_simple_solver(
    partition: int, mix_guess: bool
) -> None:
    """Simple finite-T trajectories and both entropy partitions are finite."""
    comparison = _run_simple_solver(
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
    """Finite-T Simple preserves the six map-evaluation budgets."""
    comparison = _run_simple_solver(
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
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
            "fermi_etemp": 300.0,
            "fermi_partition": labels.FERMI_PARTITION_ATOMIC,
        },
    )
    settings = setup_solver_settings(
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
    """Finite-T Simple reaches convergence."""
    comparison = _run_simple_solver(
        maxiter=100,
        mix_guess=True,
        x_atol=1.0e-4,
        x_atol_max=1.0e-5,
        extra_options={"fermi_etemp": 300.0},
    )
    assert comparison.actual.converged.item() is True


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_simple_solver_tolerance_resolution(
    dtype: torch.dtype,
) -> None:
    """Effective tolerances clamp to machine epsilon for the target dtype."""
    case = _prepare_case(
        GFN1_XTB, scf_options={"strategy": scf_strategy("linear", "charge")}
    )
    config = replace(
        case.system.config.scf,
        x_atol=1.0e-9,
        x_atol_max=1.0e-11,
    )
    settings = setup_solver_settings(config, dtype)
    resolution = float(torch.finfo(dtype).resolution)
    assert settings.x_atol == (
        resolution * 50 if resolution > config.x_atol else config.x_atol
    )
    assert settings.x_atol_max == (
        resolution * 50 if resolution > config.x_atol_max else config.x_atol_max
    )
    assert config.x_atol == 1.0e-9
    assert config.x_atol_max == 1.0e-11


def test_common_solver_settings_resolve_independent_of_converger() -> None:
    """Controller settings are shared and independent of method choice."""
    from dxtb.scf import strategy_preset

    case = _prepare_case(
        GFN1_XTB, scf_options={"strategy": scf_strategy("linear", "charge")}
    )
    linear = setup_solver_settings(case.system.config.scf, torch.float64)
    anderson = setup_solver_settings(
        replace(case.system.config.scf, strategy=strategy_preset("anderson")),
        torch.float64,
    )
    broyden = setup_solver_settings(
        replace(case.system.config.scf, strategy=strategy_preset("broyden")),
        torch.float64,
    )
    assert linear.maxiter == anderson.maxiter == broyden.maxiter
    dynamic = setup_solver_settings(
        replace(
            case.system.config.scf,
            strategy=strategy_preset("linear"),
            damp_dynamic=True,
        ),
        torch.float64,
    )
    assert dynamic.dynamic_damping.enabled
    with pytest.raises(ValueError, match="maxiter >= 0"):
        setup_solver_settings(
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
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 1.0e-30,
            "x_atol_max": 1.0e-30,
        },
    )
    settings = setup_solver_settings(
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
    position_options = {
        "strategy": scf_strategy("linear", "charge"),
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 1.0e-30,
        "x_atol_max": 1.0e-30,
    }

    def energy_at_positions(value: torch.Tensor) -> torch.Tensor:
        local_case = _prepare_case(
            GFN1_XTB, positions=value, scf_options=position_options
        )
        local_settings = setup_solver_settings(
            local_case.system.config.scf, value.dtype
        )
        local_result = solve_scf_simple(
            local_case.x, local_case.inputs, local_settings
        )
        return local_result.energy.sum()

    _assert_directional_fd(energy_at_positions, positions, position_gradient)

    par = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    leaf = par.get("element.O.levels")
    leaf.requires_grad_(True)
    parameter_case = _prepare_case(
        GFN1_XTB,
        par=par,
        scf_options={
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
        },
    )
    parameter_settings = setup_solver_settings(
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
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 1.0e-30,
            "x_atol_max": 1.0e-30,
        },
    )
    field_settings = setup_solver_settings(
        field_case.system.config.scf, torch.float64
    )
    field_energy = solve_scf_simple(
        field_case.x, field_case.inputs, field_settings
    ).energy.sum()
    field_gradient = torch.autograd.grad(field_energy, field)[0]
    assert torch.isfinite(field_gradient).all()
    assert torch.count_nonzero(field_gradient) > 0

    def energy_at_field(value: torch.Tensor) -> torch.Tensor:
        local_case = _prepare_case(
            GFN1_XTB, field=value, scf_options=position_options
        )
        local_settings = setup_solver_settings(
            local_case.system.config.scf, value.dtype
        )
        return solve_scf_simple(
            local_case.x, local_case.inputs, local_settings
        ).energy.sum()

    _assert_directional_fd(energy_at_field, field, field_gradient)


def test_real_functional_solver_vmap_over_charge_guesses() -> None:
    """Mapped initial charge iterates agree with independent solver calls."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    settings = setup_solver_settings(
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
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 300.0,
        },
    )
    settings = setup_solver_settings(
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
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 300.0,
        },
    )
    settings = setup_solver_settings(
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
        "strategy": scf_strategy("linear", "charge"),
        "maxiter": 2,
        "mix_guess": True,
        "fermi_etemp": 1000.0,
    }
    case = _prepare_case(
        GFN1_XTB,
        charge=charge,
        scf_options=options,
    )
    settings = setup_solver_settings(
        case.system.config.scf, case.positions.dtype
    )
    result = solve_scf_simple(case.x, case.inputs, settings)
    gradient = torch.autograd.grad(
        (result.energy + result.fenergy).sum(), charge
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    def free_energy_at_charge(value: torch.Tensor) -> torch.Tensor:
        local_case = _prepare_case(GFN1_XTB, charge=value, scf_options=options)
        local_settings = setup_solver_settings(
            local_case.system.config.scf, local_case.positions.dtype
        )
        local_result = solve_scf_simple(
            local_case.x, local_case.inputs, local_settings
        )
        return (local_result.energy + local_result.fenergy).sum()

    _assert_directional_fd(free_energy_at_charge, charge, gradient)


def test_position_parameter_and_field_gradients_at_finite_temperature() -> None:
    """Geometry, real parameters and explicit fields stay connected at finite T."""
    positions = (
        samples["H2O"]["positions"]
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .requires_grad_(True)
    )
    options = {
        "strategy": scf_strategy("linear", "charge"),
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 1.0e-30,
        "x_atol_max": 1.0e-30,
        "fermi_etemp": 300.0,
    }
    position_case = _prepare_case(
        GFN1_XTB, positions=positions, scf_options=options
    )
    settings = setup_solver_settings(
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

    def free_energy_at_positions(value: torch.Tensor) -> torch.Tensor:
        local_case = _prepare_case(
            GFN1_XTB, positions=value, scf_options=options
        )
        local_settings = setup_solver_settings(
            local_case.system.config.scf, value.dtype
        )
        local_result = solve_scf_simple(
            local_case.x, local_case.inputs, local_settings
        )
        return (local_result.energy + local_result.fenergy).sum()

    _assert_directional_fd(free_energy_at_positions, positions, gradient)

    par = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    leaf = par.get("element.O.levels")
    leaf.requires_grad_(True)
    parameter_case = _prepare_case(GFN1_XTB, par=par, scf_options=options)
    parameter_settings = setup_solver_settings(
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
    field_settings = setup_solver_settings(
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

    def free_energy_at_field(value: torch.Tensor) -> torch.Tensor:
        local_case = _prepare_case(GFN1_XTB, field=value, scf_options=options)
        local_settings = setup_solver_settings(
            local_case.system.config.scf, value.dtype
        )
        local_result = solve_scf_simple(
            local_case.x, local_case.inputs, local_settings
        )
        return (local_result.energy + local_result.fenergy).sum()

    _assert_directional_fd(free_energy_at_field, field, field_gradient)


def test_real_functional_solver_jvp_and_jacfwd_characterization() -> None:
    """Characterize real-map forward AD separately from solver arithmetic."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy("linear", "charge"),
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    settings = setup_solver_settings(
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
    # This is a transform smoke characterization for the new invariant
    # spectral-density path; the output stays finite in each AD mode.


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
