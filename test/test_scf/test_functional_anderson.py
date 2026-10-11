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

from dxtb import GFN1_XTB, GFN2_XTB, OutputHandler, labels
from dxtb._src.scf.functional import (
    AndersonSettings,
    AndersonState,
    SCFCharges,
    SCFMapResult,
    SCFPotential,
)
from dxtb._src.scf.functional import anderson as anderson_module
from dxtb._src.scf.functional import (
    anderson_mix,
    setup_anderson_solver_settings,
    solve_scf_anderson,
)
from dxtb._src.scf.functional.anderson import _solve_map_anderson
from dxtb._src.scf.mixer import Anderson

from .samples import samples
from .test_functional_map import _assert_tensor_equal, _prepare_case


def _map_result(mapped: torch.Tensor) -> SCFMapResult:
    return SCFMapResult(
        mapped_x=mapped,
        charges=SCFCharges(mono=mapped),
        driving_potential=SCFPotential(mono=mapped),
        hamiltonian=mapped,
        density=mapped,
        emo=mapped,
        coefficients=mapped,
        occupation=mapped,
        orbital_energy=mapped,
    )


def _settings(
    *,
    generations: int = 5,
    soft_start: bool = True,
    diagonal_offset: float | None = 0.01,
    damp: float = 0.4,
    damp_init: float = 0.1,
) -> AndersonSettings:
    return AndersonSettings(
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


def _compare_legacy_anderson(
    *,
    method: int = GFN1_XTB,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    damp_generations: int = 5,
    damp_soft_start: bool = True,
    damp_diagonal_offset: float | None = 0.01,
) -> SimpleNamespace:
    options: dict[str, object] = {
        "mixer": labels.MIXER_ANDERSON,
        "maxiter": maxiter,
        "mix_guess": mix_guess,
        "x_atol": x_atol,
        "x_atol_max": x_atol_max,
        "damp_generations": damp_generations,
        "damp_soft_start": damp_soft_start,
        "damp_diagonal_offset": damp_diagonal_offset,
    }
    legacy = _prepare_case(method, scf_options=options)
    functional = _prepare_case(method, scf_options=options)
    torch.testing.assert_close(functional.x, legacy.x, rtol=0.0, atol=0.0)

    map_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    iterate = legacy.engine.iterate_charges

    def record(x: torch.Tensor) -> torch.Tensor:
        mapped = iterate(x)
        map_calls.append((x.clone(), mapped.clone()))
        return mapped

    legacy.engine.iterate_charges = record
    legacy_x = legacy.engine.scf(legacy.x.clone(), return_charges=False)
    settings = setup_anderson_solver_settings(
        functional.system.config.scf, functional.positions.dtype
    )
    actual = solve_scf_anderson(functional.x, functional.inputs, settings)

    assert len(map_calls) == legacy.engine._data.iter
    assert int(actual.iterations) == len(map_calls)
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
    charges = legacy.engine.converged_to_charges(legacy_x)
    _assert_tensor_equal(actual.final_charges.mono, charges.mono)
    if charges.dipole is not None:
        assert actual.final_charges.dipole is not None
        _assert_tensor_equal(actual.final_charges.dipole, charges.dipole)
    if charges.quad is not None:
        assert actual.final_charges.quad is not None
        _assert_tensor_equal(actual.final_charges.quad, charges.quad)
    potential = legacy.engine.charges_to_potential(charges)
    _assert_tensor_equal(actual.final_potential.mono, potential.mono)
    if potential.dipole is not None:
        assert actual.final_potential.dipole is not None
        _assert_tensor_equal(actual.final_potential.dipole, potential.dipole)
    if potential.quad is not None:
        assert actual.final_potential.quad is not None
        _assert_tensor_equal(actual.final_potential.quad, potential.quad)
    _assert_tensor_equal(actual.energy, legacy.engine.get_energy(charges))
    expected_converged = (
        legacy.engine.mixer.converged if maxiter > 0 else torch.tensor(False)
    )
    assert torch.equal(actual.converged.cpu(), expected_converged.cpu())
    if map_calls:
        delta = map_calls[-1][1] - map_calls[-1][0]
        expected_residual = torch.stack(
            (torch.linalg.vector_norm(delta), torch.amax(delta.abs()))
        )
        _assert_tensor_equal(actual.residual, expected_residual)
    return SimpleNamespace(
        legacy=legacy,
        functional=functional,
        actual=actual,
        calls=map_calls,
        legacy_x=legacy_x,
    )


@pytest.mark.parametrize("soft_start", [True, False])
@pytest.mark.parametrize("generations,offset", [(1, 0.01), (3, 0.037)])
def test_anderson_transition_matches_legacy_history(
    soft_start: bool, generations: int, offset: float
) -> None:
    """Each pure transition matches legacy output and rolled histories."""
    settings = _settings(
        generations=generations,
        soft_start=soft_start,
        diagonal_offset=offset,
    )
    options = {
        "damp": settings.damp,
        "damp_init": settings.damp_init,
        "damp_generations": settings.generations,
        "damp_soft_start": settings.soft_start,
        "damp_diagonal_offset": settings.diagonal_offset,
        "x_tol": 0.0,
        "x_tol_max": 0.0,
    }
    legacy = Anderson(options)
    state: AndersonState | None = None
    old = torch.linspace(-0.4, 0.7, 12, dtype=torch.float64)
    for step in range(1, 9):
        residual = torch.zeros_like(old)
        residual[(step - 1) % old.numel()] = 0.2 + step * 0.013
        residual[(step + 3) % old.numel()] = -0.07 * step
        mapped = old + residual
        expected = legacy.iter(mapped, old)
        previous_state = state
        prior_x = None if state is None else state.x_hist.clone()
        prior_f = None if state is None else state.f_hist.clone()
        actual, state = anderson_mix(old, mapped, state, settings, step)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        if previous_state is not None:
            assert prior_x is not None and prior_f is not None
            torch.testing.assert_close(previous_state.x_hist, prior_x)
            torch.testing.assert_close(previous_state.f_hist, prior_f)
        assert legacy._x_hist is not None
        assert legacy._f is not None
        torch.testing.assert_close(state.x_hist, legacy._x_hist)
        torch.testing.assert_close(state.f_hist, legacy._f)
        old = expected


@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_anderson_finite_budget_matches_legacy(
    maxiter: int, mix_guess: bool
) -> None:
    """The six budget cases retain legacy map counts and final values."""
    result = _compare_legacy_anderson(
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
        current_settings: AndersonSettings,
        step: int,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AndersonState]:
        calls.append((old.clone(), mapped.clone(), state, step))
        return original(old, mapped, state, current_settings, step, active)

    monkeypatch.setattr(anderson_module, "anderson_mix", record)
    guess = torch.tensor([0.1, -0.2], dtype=torch.float64)

    def affine(x: torch.Tensor) -> SCFMapResult:
        return _map_result(0.3 * x + 0.4)

    _solve_map_anderson(guess, settings, affine)
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


def test_anderson_accelerated_solver_matches_legacy() -> None:
    """The default soft-start solver matches after acceleration begins."""
    result = _compare_legacy_anderson(
        maxiter=12,
        mix_guess=True,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert int(result.actual.iterations) == 13
    assert result.actual.converged.item() is True


def test_anderson_soft_start_false_matches_legacy() -> None:
    """Anderson acceleration starts at the second mixer transition."""
    result = _compare_legacy_anderson(
        maxiter=5,
        mix_guess=True,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
        damp_generations=3,
        damp_soft_start=False,
    )
    assert int(result.actual.iterations) == 6


def test_anderson_loose_convergence_keeps_first_result_safely() -> None:
    """Inactive fixed trips do not replace results or solve singular history."""
    result = _compare_legacy_anderson(
        maxiter=14,
        mix_guess=True,
        x_atol=1.0e6,
        x_atol_max=1.0e6,
    )
    assert result.actual.converged.item() is True
    assert int(result.actual.iterations) == 2


def test_anderson_loose_tolerance_matches_first_converged_map() -> None:
    """A loose criterion freezes the first eligible raw map result."""
    result = _compare_legacy_anderson(
        maxiter=20,
        mix_guess=True,
        x_atol=1.0e6,
        x_atol_max=1.0e6,
    )
    assert result.actual.converged.item() is True
    assert int(result.actual.iterations) == 2


def test_anderson_normal_convergence_matches_legacy() -> None:
    """A practical H2O solve converges to the same physical result."""
    result = _compare_legacy_anderson(
        maxiter=100,
        mix_guess=True,
        x_atol=1.0e-4,
        x_atol_max=1.0e-5,
    )
    assert result.actual.converged.item() is True


def test_anderson_gfn2_multipole_solver_matches_legacy() -> None:
    """Finite-budget GFN2 solving preserves all multipole charge fields."""
    result = _compare_legacy_anderson(
        method=GFN2_XTB,
        maxiter=2,
        mix_guess=False,
        x_atol=1.0e-30,
        x_atol_max=1.0e-30,
    )
    assert result.actual.final_charges.dipole is not None
    assert result.actual.final_charges.quad is not None


def test_anderson_settings_reject_unsupported_modes() -> None:
    """Anderson setup rejects Broyden and dynamic damping explicitly."""
    case = _prepare_case(GFN1_XTB, scf_options={"mixer": labels.MIXER_ANDERSON})
    config = case.system.config.scf
    with pytest.raises(NotImplementedError, match="effective Anderson"):
        setup_anderson_solver_settings(
            replace(config, mixer=labels.MIXER_BROYDEN), torch.float64
        )
    with pytest.raises(NotImplementedError, match="dynamic damping"):
        setup_anderson_solver_settings(
            replace(config, damp_dynamic=True), torch.float64
        )
    with pytest.raises(ValueError, match="generations"):
        setup_anderson_solver_settings(
            replace(config, damp_generations=0), torch.float64
        )
    invalid_legacy = Anderson(
        {
            "damp_generations": 0,
            "x_tol": 1.0e-8,
            "x_tol_max": 1.0e-8,
        }
    )
    with pytest.raises(IndexError, match="index 1"):
        invalid_legacy.iter(torch.ones(2), torch.zeros(2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_anderson_tolerance_resolution_matches_legacy(
    dtype: torch.dtype,
) -> None:
    """Anderson setup resolves thresholds like the legacy mixer wrapper."""
    case = _prepare_case(GFN1_XTB, scf_options={"mixer": labels.MIXER_ANDERSON})
    config = replace(
        case.system.config.scf,
        x_atol=1.0e-9,
        x_atol_max=1.0e-11,
    )
    settings = setup_anderson_solver_settings(config, dtype)
    mixer = Anderson(
        {
            "x_tol": config.x_atol,
            "x_tol_max": config.x_atol_max,
            "damp": config.damp,
            "damp_init": config.damp_init,
            "damp_generations": config.damp_generations,
            "damp_soft_start": config.damp_soft_start,
            "damp_diagonal_offset": config.damp_diagonal_offset,
        }
    )
    old = torch.zeros(2, dtype=dtype)
    mixer.iter(torch.ones_like(old), old)
    assert settings.x_atol == mixer.options["x_tol"]
    assert settings.x_atol_max == mixer.options["x_tol_max"]
    assert config.x_atol == 1.0e-9
    assert config.x_atol_max == 1.0e-11


def test_legacy_full_mixer_resolution_contract() -> None:
    """Legacy full mode resolves Broyden to Anderson or rejects strictly."""
    linear = _prepare_case(GFN1_XTB, scf_options={"mixer": labels.MIXER_LINEAR})
    assert type(linear.engine.mixer).__name__ == "Simple"

    anderson = _prepare_case(
        GFN1_XTB, scf_options={"mixer": labels.MIXER_ANDERSON}
    )
    assert type(anderson.engine.mixer).__name__ == "Anderson"

    OutputHandler.clear_warnings()
    fallback = _prepare_case(
        GFN1_XTB, scf_options={"mixer": labels.MIXER_BROYDEN}
    )
    assert type(fallback.engine.mixer).__name__ == "Anderson"
    warning_messages = [message for message, _ in OutputHandler.warnings]
    assert len(warning_messages) == 1
    assert warning_messages[0] == (
        "Broyden mixer is not implemented for SCF with full gradient tracking. "
        "Using Anderson mixer instead."
    )

    OutputHandler.clear_warnings()
    with pytest.raises(
        NotImplementedError, match="Broyden mixer is not implemented"
    ):
        _prepare_case(
            GFN1_XTB,
            scf_options={"mixer": labels.MIXER_BROYDEN, "strict": True},
        )
    assert OutputHandler.warnings == []


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

        return _solve_map_anderson(
            x, settings, affine
        ).map_result.mapped_x.sum()

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

        outcome = _solve_map_anderson(initial, settings, affine)
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
            fixed_point + torch.tensor([0.02, 0.008], dtype=torch.float64),
            fixed_point + torch.tensor([1000.0, 400.0], dtype=torch.float64),
        )
    )
    mapped = vmap(solve)(initial)
    looped = tuple(
        torch.stack(items) for items in zip(*(solve(item) for item in initial))
    )
    for actual, expected in zip(mapped, looped):
        torch.testing.assert_close(actual, expected)
    assert torch.unique(mapped[1]).numel() > 1
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
        "mixer": labels.MIXER_ANDERSON,
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
        "damp_soft_start": False,
        "damp_generations": 2,
    }
    geom = _prepare_case(GFN1_XTB, positions=positions, scf_options=options)
    settings = setup_anderson_solver_settings(
        geom.system.config.scf, positions.dtype
    )
    energy = solve_scf_anderson(geom.x, geom.inputs, settings).energy.sum()
    grad = torch.autograd.grad(energy, positions, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad.square().sum(), positions)[0]
    assert torch.isfinite(grad).all() and torch.isfinite(grad2).all()
    legacy_geom = _prepare_case(
        GFN1_XTB, positions=positions, scf_options=options
    )
    legacy_x = legacy_geom.engine.scf(legacy_geom.x, return_charges=False)
    legacy_charges = legacy_geom.engine.converged_to_charges(legacy_x)
    legacy_energy = legacy_geom.engine.get_energy(legacy_charges).sum()
    legacy_grad = torch.autograd.grad(
        legacy_energy, positions, create_graph=True
    )[0]
    legacy_grad2 = torch.autograd.grad(legacy_grad.square().sum(), positions)[0]
    _assert_tensor_equal(grad, legacy_grad)
    _assert_tensor_equal(grad2, legacy_grad2)

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
    legacy_parameter_case = _prepare_case(
        GFN1_XTB, par=par, scf_options=options
    )
    legacy_parameter_x = legacy_parameter_case.engine.scf(
        legacy_parameter_case.x, return_charges=False
    )
    legacy_parameter_charges = (
        legacy_parameter_case.engine.converged_to_charges(legacy_parameter_x)
    )
    legacy_parameter_energy = legacy_parameter_case.engine.get_energy(
        legacy_parameter_charges
    ).sum()
    legacy_parameter_grad = torch.autograd.grad(legacy_parameter_energy, leaf)[
        0
    ]
    _assert_tensor_equal(parameter_grad, legacy_parameter_grad)

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
    legacy_field_case = _prepare_case(
        GFN1_XTB, field=field, scf_options={**options, "maxiter": 1}
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
    legacy_field_grad = torch.autograd.grad(legacy_field_energy, field)[0]
    _assert_tensor_equal(field_grad, legacy_field_grad)


def test_anderson_real_xTB_jvp_jacfwd_characterization() -> None:
    """Real-map forward AD is characterized apart from toy solver behavior."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_ANDERSON,
            "maxiter": 2,
            "mix_guess": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "damp_generations": 2,
            "damp_soft_start": False,
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


def test_functional_anderson_has_no_legacy_mixer_or_host_control() -> None:
    """The production-style functional module stays state-value based."""
    from pathlib import Path

    source = (
        Path(__file__).parents[2] / "src/dxtb/_src/scf/functional/anderson.py"
    )
    text = source.read_text()
    for forbidden in (
        "from dxtb._src.scf.mixer",
        "Mixer(",
        "Anderson(",
        "Broyden(",
        "self._x_hist",
        "self._f",
        "iter_step",
        ".item(",
        ".tolist(",
        "bool(",
        "torch.autograd.Function",
        ".cull(",
        ".reset(",
    ):
        assert forbidden not in text
