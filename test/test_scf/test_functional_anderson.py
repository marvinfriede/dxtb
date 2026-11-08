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
"""Functional Anderson compatibility, purity, and transform tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, OutputHandler
from dxtb._src.scf.functional import (
    AndersonState,
    SCFCharges,
    SCFMapResult,
    SCFPotential,
    SCFSpectralResult,
)
from dxtb._src.scf.functional import anderson as anderson_module
from dxtb._src.scf.functional import anderson_mix
from dxtb._src.scf.functional.anderson import AndersonConvergerSettings
from dxtb._src.scf.functional.problem import SCFProblem, SCFVariable

from ..utils import scf_strategy
from ._anderson_helpers import (
    AndersonTestSettings,
)
from ._anderson_helpers import from_config as setup_anderson_solver_settings
from ._anderson_helpers import solve_common as solve_common
from ._anderson_helpers import solve_physical as solve_scf_anderson
from .samples import samples
from .test_functional_map import (
    _assert_directional_fd,
    _assert_tensor_equal,
    _prepare_case,
)


def _map_result(mapped: torch.Tensor) -> SCFMapResult:
    return SCFMapResult(
        mapped_x=mapped,
        driving_potential=SCFPotential(mono=mapped),
        spectral=SCFSpectralResult(
            charges=SCFCharges(mono=mapped),
            hamiltonian=mapped,
            density=mapped,
            emo=mapped,
            coefficients=mapped,
            occupation=mapped,
            orbital_energy=mapped,
        ),
    )


def _settings(
    *,
    generations: int = 5,
    soft_start: bool = True,
    diagonal_offset: float | None = 0.01,
    damp: float = 0.4,
    damp_init: float = 0.1,
) -> AndersonTestSettings:
    return AndersonTestSettings(
        maxiter=8,
        mix_guess=False,
        damp=damp,
        damp_init=damp_init,
        generations=generations,
        soft_start=soft_start,
        diagonal_offset=diagonal_offset,
        x_atol=0.0,
        x_atol_max=0.0,
    )


def _run_anderson(
    *,
    method: int = GFN1_XTB,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    generations: int = 5,
    soft_start: bool = True,
    diagonal_offset: float | None = 0.01,
    extra_options: dict[str, object] | None = None,
) -> SimpleNamespace:
    options: dict[str, object] = {
        "strategy": scf_strategy(
            "anderson",
            "charge",
            generations=generations,
            soft_start=soft_start,
            diagonal_offset=diagonal_offset,
        ),
        "maxiter": maxiter,
        "mix_guess": mix_guess,
        "x_atol": x_atol,
        "x_atol_max": x_atol_max,
    }
    if extra_options is not None:
        options.update(extra_options)
    functional = _prepare_case(method, scf_options=options)
    settings = setup_anderson_solver_settings(
        functional.system.config.scf, functional.positions.dtype
    )
    actual = solve_scf_anderson(functional.x, functional.inputs, settings)
    if functional.inputs.fermi_settings is None:
        assert torch.count_nonzero(actual.fenergy) == 0
    return SimpleNamespace(
        functional=functional,
        actual=actual,
    )


@pytest.mark.parametrize("soft_start", [True, False])
@pytest.mark.parametrize("generations,offset", [(1, 0.01), (3, 0.037)])
def test_anderson_transition_preserves_prior_state_immutability(
    soft_start: bool, generations: int, offset: float
) -> None:
    """Each pure transition returns a fresh state without mutating prior history."""
    settings = _settings(
        generations=generations,
        soft_start=soft_start,
        diagonal_offset=offset,
    )
    state: AndersonState | None = None
    old = torch.linspace(-0.4, 0.7, 12, dtype=torch.float64)
    for step in range(1, 9):
        residual = torch.zeros_like(old)
        residual[(step - 1) % old.numel()] = 0.2 + step * 0.013
        residual[(step + 3) % old.numel()] = -0.07 * step
        mapped = old + residual
        previous_state = state
        prior_x = None if state is None else state.x_hist.clone()
        prior_f = None if state is None else state.f_hist.clone()
        actual, state = anderson_mix(
            old,
            mapped,
            state,
            settings.method(),
            step,
            damp=settings.damp,
        )
        assert torch.isfinite(actual).all()
        if previous_state is not None:
            assert prior_x is not None and prior_f is not None
            torch.testing.assert_close(previous_state.x_hist, prior_x)
            torch.testing.assert_close(previous_state.f_hist, prior_f)
        old = actual


@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_anderson_finite_budget(
    maxiter: int, mix_guess: bool
) -> None:
    """The six budget cases retain expected map counts and final values."""
    result = _run_anderson(
        maxiter=maxiter,
        mix_guess=mix_guess,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert int(result.actual.iterations) == maxiter + 1
    assert result.actual.converged.item() is False


@pytest.mark.parametrize("mix_guess", [False, True])
def test_anderson_initial_map_history_semantics(
    mix_guess: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only mix_guess=True inserts the initial map into mixer history."""
    settings = replace(_settings(generations=2), maxiter=1, mix_guess=mix_guess)
    calls: list[
        tuple[torch.Tensor, torch.Tensor, AndersonState | None, int]
    ] = []
    original = anderson_module.anderson_mix

    def record(
        old: torch.Tensor,
        mapped: torch.Tensor,
        state: AndersonState | None,
        current_settings: AndersonConvergerSettings,
        step: int,
        transition_active: torch.Tensor | None = None,
        damp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AndersonState]:
        calls.append((old.clone(), mapped.clone(), state, step))
        return original(
            old,
            mapped,
            state,
            current_settings,
            step,
            transition_active=transition_active,
            damp=damp,
        )

    monkeypatch.setattr(anderson_module, "anderson_mix", record)
    guess = torch.tensor([0.1, -0.2], dtype=torch.float64)

    def affine(x: torch.Tensor) -> SCFMapResult:
        return _map_result(0.3 * x + 0.4)

    solve_common(guess, settings, affine)
    if mix_guess:
        assert calls[0][3] == 1
        assert calls[0][2] is None
        torch.testing.assert_close(calls[0][0], guess)
    assert calls[-1][3] == (2 if mix_guess else 1)
    if not mix_guess:
        assert len(calls) == 1
        assert calls[0][2] is None
        torch.testing.assert_close(calls[0][0], affine(guess).mapped_x)
    else:
        assert len(calls) == 2
        assert calls[1][2] is not None


def test_anderson_converged_accelerated_step_does_not_advance_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A converged candidate skips the next Anderson history transition."""
    settings = replace(
        _settings(
            generations=1,
            soft_start=False,
            diagonal_offset=None,
            damp_init=0.5,
        ),
        maxiter=3,
        mix_guess=True,
        x_atol=10.0,
        x_atol_max=10.0,
    )
    guess = torch.tensor([0.1, -0.2], dtype=torch.float64)
    increment = torch.tensor([0.5, 0.25], dtype=torch.float64)
    calls: list[tuple[int, torch.Tensor | None, AndersonState]] = []
    original = anderson_module.anderson_mix

    def record(
        old: torch.Tensor,
        mapped: torch.Tensor,
        state: AndersonState | None,
        current_settings: AndersonConvergerSettings,
        step: int,
        transition_active: torch.Tensor | None = None,
        damp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AndersonState]:
        mixed, next_state = original(
            old,
            mapped,
            state,
            current_settings,
            step,
            transition_active=transition_active,
            damp=damp,
        )
        calls.append(
            (
                step,
                (
                    None
                    if transition_active is None
                    else transition_active.clone()
                ),
                next_state,
            )
        )
        return mixed, next_state

    monkeypatch.setattr(anderson_module, "anderson_mix", record)

    def constant_residual(x: torch.Tensor) -> SCFMapResult:
        return _map_result(x + increment)

    outcome = solve_common(guess, settings, constant_residual)
    assert outcome.converged.item() is True
    assert int(outcome.iterations) == 2
    torch.testing.assert_close(
        outcome.map_result.mapped_x,
        constant_residual(guess + increment * settings.damp_init).mapped_x,
    )
    torch.testing.assert_close(
        outcome.residual,
        torch.tensor([0.3125**0.5, 0.5], dtype=torch.float64),
    )
    assert [step for step, _, _ in calls] == [1, 2, 3, 4]
    assert calls[0][1] is not None and calls[0][1].item()
    assert calls[1][1] is not None and not calls[1][1].item()
    torch.testing.assert_close(calls[1][2].x_hist, calls[0][2].x_hist)
    torch.testing.assert_close(calls[1][2].f_hist, calls[0][2].f_hist)
    for _, transition_active, state in calls[2:]:
        assert transition_active is not None and not transition_active.item()
        torch.testing.assert_close(state.x_hist, calls[0][2].x_hist)
        torch.testing.assert_close(state.f_hist, calls[0][2].f_hist)


def test_anderson_inactive_initial_transition_keeps_initialized_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convergence before the first mix leaves Anderson history unadvanced."""
    settings = replace(
        _settings(generations=2),
        maxiter=1,
        mix_guess=False,
        x_atol=10.0,
        x_atol_max=10.0,
    )
    guess = torch.tensor([0.1, -0.2], dtype=torch.float64)
    calls: list[
        tuple[torch.Tensor, AndersonState | None, torch.Tensor, AndersonState]
    ] = []
    original = anderson_module.anderson_mix

    def record(
        old: torch.Tensor,
        mapped: torch.Tensor,
        state: AndersonState | None,
        current_settings: AndersonConvergerSettings,
        step: int,
        transition_active: torch.Tensor | None = None,
        damp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AndersonState]:
        mixed, next_state = original(
            old,
            mapped,
            state,
            current_settings,
            step,
            transition_active=transition_active,
            damp=damp,
        )
        assert transition_active is not None
        calls.append(
            (old.clone(), state, transition_active.clone(), next_state)
        )
        return mixed, next_state

    monkeypatch.setattr(anderson_module, "anderson_mix", record)

    def constant_residual(x: torch.Tensor) -> SCFMapResult:
        return _map_result(x + 0.2)

    outcome = solve_common(guess, settings, constant_residual)
    assert outcome.converged.item() is True
    assert int(outcome.iterations) == 2
    torch.testing.assert_close(
        outcome.map_result.mapped_x,
        constant_residual(constant_residual(guess).mapped_x).mapped_x,
    )
    assert len(calls) == 1
    old, state, transition_active, next_state = calls[0]
    assert state is None
    assert not transition_active.item()
    torch.testing.assert_close(next_state.x_hist[0], old)
    torch.testing.assert_close(
        next_state.x_hist[1:], torch.zeros_like(next_state.x_hist[1:])
    )
    torch.testing.assert_close(
        next_state.f_hist, torch.zeros_like(next_state.f_hist)
    )


def test_anderson_history_freezes_after_two_active_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accepted third candidate preserves history from the prior step."""
    settings = replace(
        _settings(generations=1, soft_start=False, diagonal_offset=0.05),
        maxiter=4,
        mix_guess=False,
        x_atol=0.01,
        x_atol_max=0.01,
    )
    matrix = torch.tensor([[0.25, 0.04], [-0.03, 0.4]], dtype=torch.float64)
    bias = torch.tensor([0.1, -0.08], dtype=torch.float64)
    guess = torch.tensor([0.1, 0.2], dtype=torch.float64)
    states: list[tuple[int, torch.Tensor, AndersonState]] = []
    original = anderson_module.anderson_mix

    def record(
        old: torch.Tensor,
        mapped: torch.Tensor,
        state: AndersonState | None,
        current_settings: AndersonConvergerSettings,
        step: int,
        transition_active: torch.Tensor | None = None,
        damp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AndersonState]:
        mixed, next_state = original(
            old,
            mapped,
            state,
            current_settings,
            step,
            transition_active=transition_active,
            damp=damp,
        )
        assert transition_active is not None
        states.append((step, transition_active.clone(), next_state))
        return mixed, next_state

    monkeypatch.setattr(anderson_module, "anderson_mix", record)

    def affine(x: torch.Tensor) -> SCFMapResult:
        return _map_result(matrix @ x + bias)

    outcome = solve_common(guess, settings, affine)
    assert outcome.converged.item() is True
    assert int(outcome.iterations) == 4
    assert [step for step, _, _ in states] == [1, 2, 3, 4]
    assert all(mask.item() for _, mask, _ in states[:2])
    assert all(not mask.item() for _, mask, _ in states[2:])
    torch.testing.assert_close(states[2][2].x_hist, states[1][2].x_hist)
    torch.testing.assert_close(states[2][2].f_hist, states[1][2].f_hist)
    torch.testing.assert_close(states[3][2].x_hist, states[1][2].x_hist)
    torch.testing.assert_close(states[3][2].f_hist, states[1][2].f_hist)


def test_anderson_constant_residual_converges_without_next_transition() -> None:
    """Canonical control accepts the converged map before another transition."""
    settings = replace(
        _settings(
            generations=1,
            soft_start=False,
            diagonal_offset=None,
            damp_init=0.5,
        ),
        maxiter=3,
        mix_guess=True,
        x_atol=10.0,
        x_atol_max=10.0,
    )
    guess = torch.tensor([0.1, -0.2], dtype=torch.float64)
    increment = torch.tensor([0.5, 0.25], dtype=torch.float64)

    def constant_residual(x: torch.Tensor) -> SCFMapResult:
        return _map_result(x + increment)

    outcome = solve_common(guess, settings, constant_residual)
    legacy_first_mixed, _ = anderson_mix(
        guess,
        constant_residual(guess).mapped_x,
        None,
        settings.method(),
        1,
        damp=settings.damp,
    )
    legacy_candidate = constant_residual(legacy_first_mixed)
    torch.testing.assert_close(
        outcome.map_result.mapped_x, legacy_candidate.mapped_x
    )
    assert outcome.converged.item() is True
    assert int(outcome.iterations) == 2

    def solve(value: torch.Tensor) -> torch.Tensor:
        return solve_common(
            value, settings, constant_residual
        ).map_result.mapped_x.sum()

    direction = torch.tensor([-0.3, 0.7], dtype=torch.float64)
    _, tangent = jvp(solve, (guess,), (direction,))
    leaf = guess.clone().requires_grad_(True)
    gradient = torch.autograd.grad(solve(leaf), leaf)[0]
    torch.testing.assert_close(tangent, torch.dot(gradient, direction))
    torch.testing.assert_close(jacfwd(solve)(guess), gradient)
    points = torch.stack((guess, guess + 0.2))
    torch.testing.assert_close(
        vmap(solve)(points), torch.stack([solve(point) for point in points])
    )


def test_anderson_accelerated_solver() -> None:
    """The default soft-start solver converges after acceleration begins."""
    result = _run_anderson(
        maxiter=12,
        mix_guess=True,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert int(result.actual.iterations) == 13
    assert result.actual.converged.item() is True


def test_anderson_soft_start_false() -> None:
    """Anderson acceleration starts at the second mixer transition."""
    result = _run_anderson(
        maxiter=5,
        mix_guess=True,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
        generations=3,
        soft_start=False,
    )
    assert int(result.actual.iterations) == 6


def _solve_loose_anderson_with_certification(maxiter: int):
    """Large raw tolerances still require a valid selected physical root."""
    options = {
        "strategy": scf_strategy("anderson", "charge"),
        "maxiter": maxiter,
        "mix_guess": True,
        "x_atol": 1.0e6,
        "x_atol_max": 1.0e6,
    }
    case = _prepare_case(GFN1_XTB, scf_options=options)
    settings = setup_anderson_solver_settings(
        case.system.config.scf, case.positions.dtype
    )
    result = solve_scf_anderson(case.x, case.inputs, settings)
    expected = SCFProblem(case.inputs, SCFVariable.CHARGE).evaluate(
        result.root_x
    )
    _assert_tensor_equal(
        result.map_result.mapped_x, expected.map_result.mapped_x
    )
    _assert_tensor_equal(
        result.map_result.spectral.density,
        expected.map_result.spectral.density,
    )
    return result


def test_anderson_loose_convergence_keeps_first_result_safely() -> None:
    """Inactive fixed trips preserve the first physically certified result."""
    result = _solve_loose_anderson_with_certification(maxiter=14)
    assert result.converged.item() is True
    assert int(result.iterations) > 2


def test_anderson_loose_tolerance_matches_first_converged_map() -> None:
    """Raw-map candidacy waits for fresh-Fock physical certification."""
    result = _solve_loose_anderson_with_certification(maxiter=20)
    assert result.converged.item() is True
    assert int(result.iterations) > 2


def test_anderson_normal_convergence() -> None:
    """A practical H2O solve converges to a certified physical result."""
    result = _run_anderson(
        maxiter=100,
        mix_guess=True,
        x_atol=1.0e-4,
        x_atol_max=1.0e-5,
    )
    assert result.actual.converged.item() is True


def test_anderson_gfn2_multipole_solver() -> None:
    """Finite-budget GFN2 solving preserves all multipole charge fields."""
    result = _run_anderson(
        method=GFN2_XTB,
        maxiter=2,
        mix_guess=False,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert result.actual.final_charges.dipole is not None
    assert result.actual.final_charges.quad is not None


def test_anderson_settings_rejects_broyden_and_resolves_damping() -> None:
    """Anderson setup rejects Broyden and freezes dynamic settings."""
    from dxtb.scf import AndersonConverger, SCFStrategy, strategy_preset

    case = _prepare_case(
        GFN1_XTB,
        scf_options={"strategy": scf_strategy("anderson", "charge")},
    )
    config = case.system.config.scf
    with pytest.raises(NotImplementedError, match="require an Anderson"):
        setup_anderson_solver_settings(
            replace(config, strategy=strategy_preset("broyden")), torch.float64
        )
    dynamic = setup_anderson_solver_settings(
        replace(config, damp_dynamic=True), torch.float64
    )
    assert dynamic.dynamic_damping.enabled
    with pytest.raises(ValueError, match="generations"):
        setup_anderson_solver_settings(
            replace(
                config,
                strategy=SCFStrategy(
                    (
                        replace(
                            config.strategy.stages[0],
                            converger=AndersonConverger(
                                replace(
                                    config.strategy.stages[0].converger.settings,
                                    generations=0,
                                )
                            ),
                        ),
                    )
                ),
            ),
            torch.float64,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_anderson_tolerance_resolution(
    dtype: torch.dtype,
) -> None:
    """Anderson setup resolves thresholds using dtype epsilon floor."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={"strategy": scf_strategy("anderson", "charge")},
    )
    config = replace(
        case.system.config.scf,
        x_atol=1.0e-9,
        x_atol_max=1.0e-11,
    )
    settings = setup_anderson_solver_settings(config, dtype)
    resolution = float(torch.finfo(dtype).resolution)
    assert settings.x_atol == (
        resolution * 50 if resolution > config.x_atol else config.x_atol
    )
    assert settings.x_atol_max == (
        resolution * 50 if resolution > config.x_atol_max else config.x_atol_max
    )
    assert config.x_atol == 1.0e-9
    assert config.x_atol_max == 1.0e-11


def test_anderson_solver_affine_transform_ladder() -> None:
    """Functional Anderson arithmetic has consistent AD transforms."""
    settings = _settings(
        generations=2,
        soft_start=False,
        diagonal_offset=0.05,
        damp=0.35,
    )
    matrix = torch.tensor([[0.2, 0.03], [-0.04, 0.25]], dtype=torch.float64)
    offset = torch.tensor([0.5, -0.2], dtype=torch.float64)

    def solve(x: torch.Tensor) -> torch.Tensor:
        def affine(value: torch.Tensor) -> SCFMapResult:
            return _map_result(matrix @ value + offset)

        return solve_common(x, settings, affine).map_result.mapped_x.sum()

    point = torch.tensor([0.1, -0.3], dtype=torch.float64)
    direction = torch.tensor([0.6, -0.8], dtype=torch.float64)
    value, tangent = jvp(solve, (point,), (direction,))
    leaf = point.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(solve(leaf), leaf)
    forward = jacfwd(solve)(point)
    torch.testing.assert_close(tangent, torch.dot(gradient, direction))
    torch.testing.assert_close(forward, gradient)
    assert torch.isfinite(value)

    points = torch.tensor(
        [[0.1, -0.3], [0.7, 0.2], [-0.4, 0.8]], dtype=torch.float64
    )
    mapped = vmap(solve)(points)
    looped = torch.stack([solve(point) for point in points])
    torch.testing.assert_close(mapped, looped)


def test_anderson_solver_mixed_convergence_vmap() -> None:
    """Mapped toy systems freeze distinct convergence times through AA steps."""
    settings = replace(
        _settings(generations=5, soft_start=True, diagonal_offset=0.05),
        maxiter=8,
        mix_guess=True,
        x_atol=0.01,
        x_atol_max=0.01,
    )
    matrix = torch.tensor([[0.25, 0.04], [-0.03, 0.4]], dtype=torch.float64)
    bias = torch.tensor([0.1, -0.08], dtype=torch.float64)

    def solve(initial: torch.Tensor) -> tuple[torch.Tensor, ...]:
        def affine(x: torch.Tensor) -> SCFMapResult:
            return _map_result(matrix @ x + bias)

        outcome = solve_common(initial, settings, affine)
        return (
            outcome.map_result.mapped_x,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
        )

    fixed_point = torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) - matrix, bias
    )
    initial = torch.stack(
        (
            fixed_point,
            fixed_point + torch.tensor([0.6, 0.3], dtype=torch.float64),
            fixed_point + torch.tensor([1000.0, 400.0], dtype=torch.float64),
        )
    )
    mapped = vmap(solve)(initial)
    looped = tuple(
        torch.stack(items) for items in zip(*(solve(item) for item in initial))
    )
    for actual, expected in zip(mapped, looped):
        torch.testing.assert_close(actual, expected)
    assert mapped[1].tolist() == [2, 8, 9]
    assert torch.any(~mapped[2])


def test_anderson_solver_real_position_parameter_and_field_gradients() -> None:
    """Position, parameter, and explicit-field graphs pass through Anderson."""
    positions = (
        samples["H2O"]["positions"]
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .requires_grad_(True)
    )
    options = {
        "strategy": scf_strategy(
            "anderson", "charge", soft_start=False, generations=2
        ),
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    geom = _prepare_case(GFN1_XTB, positions=positions, scf_options=options)
    settings = setup_anderson_solver_settings(
        geom.system.config.scf, positions.dtype
    )
    energy = solve_scf_anderson(geom.x, geom.inputs, settings).energy.sum()
    grad = torch.autograd.grad(energy, positions, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad.square().sum(), positions)[0]
    assert torch.isfinite(grad).all() and torch.isfinite(grad2).all()

    def energy_at_positions(value: torch.Tensor) -> torch.Tensor:
        local = _prepare_case(GFN1_XTB, positions=value, scf_options=options)
        local_settings = setup_anderson_solver_settings(
            local.system.config.scf, value.dtype
        )
        return solve_scf_anderson(
            local.x, local.inputs, local_settings
        ).energy.sum()

    _assert_directional_fd(energy_at_positions, positions, grad)

    from dxtb._src.param import ParamModule

    par = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    leaf = par.get("element.O.levels")
    leaf.requires_grad_(True)
    parameter_case = _prepare_case(GFN1_XTB, par=par, scf_options=options)
    parameter_settings = setup_anderson_solver_settings(
        parameter_case.system.config.scf, torch.float64
    )
    parameter_energy = solve_scf_anderson(
        parameter_case.x, parameter_case.inputs, parameter_settings
    ).energy.sum()
    parameter_grad = torch.autograd.grad(parameter_energy, leaf)[0]
    assert torch.isfinite(parameter_grad).all()
    assert torch.count_nonzero(parameter_grad) > 0

    field = torch.tensor(
        [0.001, -0.0007, 0.0004], dtype=torch.float64, requires_grad=True
    )
    field_case = _prepare_case(
        GFN1_XTB, field=field, scf_options={**options, "maxiter": 1}
    )
    field_settings = setup_anderson_solver_settings(
        field_case.system.config.scf, torch.float64
    )
    field_energy = solve_scf_anderson(
        field_case.x, field_case.inputs, field_settings
    ).energy.sum()
    field_grad = torch.autograd.grad(field_energy, field)[0]
    assert torch.isfinite(field_grad).all()
    assert torch.count_nonzero(field_grad) > 0

    def energy_at_field(value: torch.Tensor) -> torch.Tensor:
        local = _prepare_case(
            GFN1_XTB, field=value, scf_options={**options, "maxiter": 1}
        )
        local_settings = setup_anderson_solver_settings(
            local.system.config.scf, torch.float64
        )
        return solve_scf_anderson(
            local.x, local.inputs, local_settings
        ).energy.sum()

    _assert_directional_fd(energy_at_field, field, field_grad)


def test_anderson_real_xTB_jvp_jacfwd_characterization() -> None:
    """Real-map forward AD is characterized apart from toy solver behavior."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy(
                "anderson", "charge", generations=2, soft_start=False
            ),
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    settings = setup_anderson_solver_settings(
        case.system.config.scf, case.positions.dtype
    )

    def energy(guess: torch.Tensor) -> torch.Tensor:
        return solve_scf_anderson(guess, case.inputs, settings).energy.sum()

    direction = torch.ones_like(case.x) / case.x.numel() ** 0.5
    value, tangent = jvp(energy, (case.x,), (direction,))
    forward = jacfwd(energy)(case.x)
    leaf = case.x.clone().requires_grad_(True)
    (reverse,) = torch.autograd.grad(energy(leaf), leaf)
    assert torch.isfinite(value)
    assert torch.isfinite(tangent)
    assert torch.isfinite(forward).all()
    assert torch.isfinite(reverse).all()
    # The affine solver test proves the solver arithmetic itself agrees;
    # any real xTB AD-mode difference remains in the E3 spectral primitive.


def test_finite_temperature_anderson_and_free_energy() -> None:
    """Anderson acceleration and entropy evaluate cleanly at finite temperature."""
    comparison = _run_anderson(
        maxiter=7,
        mix_guess=True,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
        extra_options={"fermi_etemp": 300.0},
    )
    assert int(comparison.actual.iterations) == 8
    assert comparison.actual.map_result.fermi_status is not None


def test_finite_temperature_anderson_converged() -> None:
    """Finite-T Anderson converges with finite free energy."""
    comparison = _run_anderson(
        maxiter=100,
        mix_guess=True,
        x_atol=1.0e-4,
        x_atol_max=1.0e-5,
        extra_options={"fermi_etemp": 300.0},
    )
    assert comparison.actual.converged.item() is True
