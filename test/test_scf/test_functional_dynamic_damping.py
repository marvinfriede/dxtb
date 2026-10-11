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
"""Functional dynamic-damping transition and transform tests."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.func import vmap

from dxtb._src.scf.functional import (
    AndersonSettings,
    SCFCharges,
    SCFMapResult,
    SCFSolverSettings,
    SCFSpectralResult,
    anderson_mix,
)
from dxtb._src.scf.functional.anderson import (
    _solve_map_anderson,
    setup_anderson_solver_settings,
)
from dxtb._src.scf.functional.solver import (
    DynamicDampingSettings,
    _solve_map_control,
    setup_simple_solver_settings,
)
from dxtb import GFN1_XTB, labels

from .test_functional_map import _prepare_case


def _map_result(mapped: torch.Tensor) -> SCFMapResult:
    spectral = SCFSpectralResult(
        hamiltonian=mapped,
        density=mapped,
        emo=mapped,
        coefficients=mapped,
        occupation=mapped,
        charges=SCFCharges(mono=mapped),
        orbital_energy=mapped,
    )
    return SCFMapResult(mapped_x=mapped, spectral=spectral)


def _run_simple(
    initial: torch.Tensor, offset: torch.Tensor, slope: torch.Tensor
):
    def map_fn(value: torch.Tensor) -> SCFMapResult:
        return _map_result(slope * value + offset)

    return _solve_map_control(
        initial,
        maxiter=5,
        mix_guess=False,
        x_atol=0.0,
        x_atol_max=0.0,
        map_fn=map_fn,
        mix_fn=lambda old, mapped, state, step, active, damp: (
            old + (mapped - old) * damp,
            None,
        ),
        initial_damp=0.2,
        damping_settings=DynamicDampingSettings(True, 0.8),
    )


def _run_simple_values(
    initial: torch.Tensor, offset: torch.Tensor, slope: torch.Tensor
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return tensor leaves so the solver outcome can be mapped with vmap."""
    outcome = _run_simple(initial, offset, slope)
    return (
        outcome.map_result.mapped_x,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.damping_state.value,
    )


def test_dynamic_damping_switches_after_transition_for_next_step() -> None:
    """A raw residual switches only the following Simple transition."""
    initial = torch.zeros((), dtype=torch.float64)
    offset = torch.tensor(0.04, dtype=torch.float64)
    slope = torch.tensor(0.5, dtype=torch.float64)
    seen: list[torch.Tensor] = []

    def map_fn(value: torch.Tensor) -> SCFMapResult:
        seen.append(value.clone())
        return _map_result(slope * value + offset)

    outcome = _solve_map_control(
        initial,
        maxiter=3,
        mix_guess=False,
        x_atol=0.0,
        x_atol_max=0.0,
        map_fn=map_fn,
        mix_fn=lambda old, mapped, state, step, active, damp: (
            old + (mapped - old) * damp,
            None,
        ),
        initial_damp=0.2,
        damping_settings=DynamicDampingSettings(True, 0.8),
    )
    torch.testing.assert_close(seen[0], torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(seen[1], torch.tensor(0.04, dtype=torch.float64))
    torch.testing.assert_close(
        seen[2], torch.tensor(0.044, dtype=torch.float64)
    )
    torch.testing.assert_close(
        outcome.damping_state.value, torch.tensor(0.8, dtype=torch.float64)
    )


def test_mix_guess_does_not_switch_and_convergence_freezes_damping() -> None:
    """The initial transition is exempt and a converged lane does not switch."""
    initial = torch.zeros((), dtype=torch.float64)
    offset = torch.tensor(0.04, dtype=torch.float64)
    slope = torch.tensor(0.5, dtype=torch.float64)
    seen: list[torch.Tensor] = []

    def map_fn(value: torch.Tensor) -> SCFMapResult:
        seen.append(value.clone())
        return _map_result(slope * value + offset)

    outcome = _solve_map_control(
        initial,
        maxiter=2,
        mix_guess=True,
        x_atol=1.0,
        x_atol_max=1.0,
        map_fn=map_fn,
        mix_fn=lambda old, mapped, state, step, active, damp: (
            old + (mapped - old) * damp,
            None,
        ),
        initial_damp=0.2,
        damping_settings=DynamicDampingSettings(True, 0.8),
    )
    torch.testing.assert_close(
        seen[1], torch.tensor(0.008, dtype=torch.float64)
    )
    assert bool(outcome.converged)
    torch.testing.assert_close(
        outcome.damping_state.value, torch.tensor(0.2, dtype=torch.float64)
    )


def test_dynamic_damping_is_lane_local_under_vmap() -> None:
    """Different residual threshold crossings retain independent damping."""
    initials = torch.zeros(3, dtype=torch.float64)
    offsets = torch.tensor([0.25, 0.18, 0.30], dtype=torch.float64)
    slopes = torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64)
    mapped = vmap(_run_simple_values)(initials, offsets, slopes)
    looped = [
        _run_simple_values(initials[i], offsets[i], slopes[i]) for i in range(3)
    ]
    for mapped_value, index in zip(mapped, range(5)):
        torch.testing.assert_close(
            mapped_value, torch.stack([value[index] for value in looped])
        )
    assert torch.equal(
        mapped[-1],
        torch.tensor([0.2, 0.8, 0.8], dtype=torch.float64),
    )


def test_anderson_dynamic_damping_is_lane_local_under_vmap() -> None:
    """Mapped Anderson lanes switch ordinary damping independently."""
    settings = AndersonSettings(
        maxiter=3,
        mix_guess=False,
        damp=0.2,
        damp_init=0.05,
        generations=1,
        soft_start=True,
        diagonal_offset=0.01,
        x_atol=0.0,
        x_atol_max=0.0,
        dynamic_damping=DynamicDampingSettings(True, 0.8),
    )
    matrix = torch.tensor([[0.92, 0.02], [0.03, 0.90]], dtype=torch.float64)
    offsets = torch.tensor(
        [[4.0, 1.2], [0.105, 0.0315], [0.15, 0.045]],
        dtype=torch.float64,
    )
    initials = torch.zeros_like(offsets)

    def run(initial: torch.Tensor, offset: torch.Tensor):
        def map_fn(value: torch.Tensor) -> SCFMapResult:
            return _map_result(matrix @ value + offset)

        outcome = _solve_map_anderson(initial, settings, map_fn)
        return (
            outcome.map_result.mapped_x,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
            outcome.damping_state.value,
        )

    mapped = vmap(run)(initials, offsets)
    looped = [run(initials[i], offsets[i]) for i in range(3)]
    for mapped_value, index in zip(mapped, range(5)):
        torch.testing.assert_close(
            mapped_value, torch.stack([value[index] for value in looped])
        )
    assert torch.equal(
        mapped[-1],
        torch.tensor([0.2, 0.8, 0.8], dtype=torch.float64),
    )


def test_dynamic_damping_setup_resolves_simple_and_anderson() -> None:
    """Both functional mixers freeze the configured dynamic factor."""
    case = _prepare_case(GFN1_XTB)
    config = replace(
        case.system.config.scf,
        mixer=labels.MIXER_LINEAR,
        damp_dynamic=True,
        damp_dynamic_factor=0.73,
    )
    simple = setup_simple_solver_settings(config, torch.float64)
    anderson = setup_anderson_solver_settings(
        replace(config, mixer=labels.MIXER_ANDERSON), torch.float64
    )
    assert simple.dynamic_damping.enabled
    assert anderson.dynamic_damping.enabled
    assert simple.dynamic_damping.factor == 0.73
    assert anderson.dynamic_damping.factor == 0.73
    assert simple.dynamic_damping.threshold == 0.1
    assert anderson.dynamic_damping.threshold == 0.1


def test_anderson_warmup_keeps_damp_init_after_dynamic_switch() -> None:
    """Dynamic ordinary damping does not change Anderson warm-up damping."""
    settings = AndersonSettings(
        maxiter=3,
        mix_guess=False,
        damp=0.25,
        damp_init=0.1,
        generations=1,
        soft_start=True,
        diagonal_offset=0.01,
        x_atol=0.0,
        x_atol_max=0.0,
        dynamic_damping=DynamicDampingSettings(True, 0.8),
    )
    initial = torch.zeros(2, dtype=torch.float64)
    offset = torch.tensor([0.04, 0.03], dtype=torch.float64)
    matrix = torch.tensor([[0.2, 0.03], [0.04, 0.25]], dtype=torch.float64)

    def map_fn(value: torch.Tensor) -> SCFMapResult:
        return _map_result(matrix @ value + offset)

    outcome = _solve_map_anderson(initial, settings, map_fn)
    # Two raw transition residuals are below 0.1 and switch ordinary damp.
    torch.testing.assert_close(
        outcome.damping_state.value, torch.tensor(0.8, dtype=torch.float64)
    )
    assert torch.isfinite(outcome.map_result.mapped_x).all()


def test_anderson_warmup_uses_damp_init_not_runtime_damping() -> None:
    """Warm-up transitions retain damp_init after ordinary damping changes."""
    settings = AndersonSettings(
        maxiter=3,
        mix_guess=False,
        damp=0.25,
        damp_init=0.1,
        generations=3,
        soft_start=True,
        diagonal_offset=0.01,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    old = torch.zeros(2, dtype=torch.float64)
    mapped = torch.ones_like(old)
    first, state = anderson_mix(old, mapped, None, settings, 1, damp=0.8)
    torch.testing.assert_close(first, torch.full_like(old, 0.1))
    second, _ = anderson_mix(first, mapped, state, settings, 2, damp=0.8)
    torch.testing.assert_close(
        second, first + (mapped - first) * settings.damp_init
    )
