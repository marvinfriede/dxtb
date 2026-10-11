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
"""Functional Anderson mixing and fixed-trip SCF solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch

from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor

from .data import PotentialLayout, SCFInputs, SCFMapResult
from .map import fock_map, potential_map, scf_map
from .solver import (
    DynamicDampingSettings,
    SCFSolveResult,
    _project_potential_padding,
    _resolve_tolerance,
    _solve_map_control,
    _SolverOutcome,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
)

__all__ = [
    "AndersonState",
    "AndersonSettings",
    "anderson_mix",
    "setup_anderson_solver_settings",
    "solve_scf_anderson",
    "solve_scf_potential_anderson",
    "solve_scf_fock_anderson",
]


@dataclass(frozen=True, eq=False)
class AndersonSettings:
    """Static root-finding settings for the functional Anderson solver."""

    maxiter: int
    mix_guess: bool
    damp: float
    damp_init: float
    generations: int
    soft_start: bool
    diagonal_offset: float | None
    x_atol: float
    x_atol_max: float
    dynamic_damping: DynamicDampingSettings = DynamicDampingSettings()


@dataclass(frozen=True, eq=False)
class AndersonState:
    """Immutable, fixed-size history for one single-System mixer."""

    x_hist: Tensor
    f_hist: Tensor


def setup_anderson_solver_settings(
    config: ConfigSCF, dtype: torch.dtype
) -> AndersonSettings:
    """Resolve Anderson-only settings without mutating the configuration."""
    if config.maxiter < 0:
        raise ValueError("The functional SCF solver requires maxiter >= 0.")
    if config.mixer != labels.MIXER_ANDERSON:
        raise NotImplementedError(
            "Functional Anderson settings require the effective Anderson mixer."
        )
    if config.damp_generations < 1:
        raise ValueError("Anderson generations must be at least 1.")

    return AndersonSettings(
        maxiter=config.maxiter,
        mix_guess=config.mix_guess,
        damp=config.damp,
        damp_init=config.damp_init,
        generations=config.damp_generations,
        soft_start=config.damp_soft_start,
        diagonal_offset=config.damp_diagonal_offset,
        x_atol=_resolve_tolerance(config.x_atol, dtype),
        x_atol_max=_resolve_tolerance(config.x_atol_max, dtype),
        dynamic_damping=DynamicDampingSettings(
            enabled=config.damp_dynamic,
            factor=config.damp_dynamic_factor,
        ),
    )


def _select_state(
    active: Tensor, new: AndersonState, old: AndersonState
) -> AndersonState:
    return AndersonState(
        x_hist=torch.where(active, new.x_hist, old.x_hist),
        f_hist=torch.where(active, new.f_hist, old.f_hist),
    )


def anderson_mix(
    old: Tensor,
    mapped: Tensor,
    state: AndersonState | None,
    settings: AndersonSettings,
    step: int,
    transition_active: Tensor | None = None,
    damp: Tensor | float | None = None,
) -> tuple[Tensor, AndersonState]:
    """Apply one out-of-place transition matching the legacy Anderson mixer."""
    flat_old = old.reshape(-1)
    flat_mapped = mapped.reshape(-1)
    if state is None:
        x_hist = torch.cat(
            (
                flat_old.unsqueeze(0),
                flat_old.new_zeros((settings.generations, flat_old.numel())),
            ),
            dim=0,
        )
        f_hist = flat_old.new_zeros(
            (settings.generations + 1, flat_old.numel())
        )
        previous = AndersonState(x_hist=x_hist, f_hist=f_hist)
    else:
        previous = state

    raw = flat_mapped - flat_old
    f_current = torch.cat((raw.unsqueeze(0), previous.f_hist[1:]), dim=0)
    x_current = torch.cat((flat_old.unsqueeze(0), previous.x_hist[1:]), dim=0)

    accelerate = step > settings.generations or (
        step > 1 and not settings.soft_start
    )
    if accelerate:
        n = min(step - 1, settings.generations)
        df = f_current[0] - f_current[1 : n + 1]
        a = torch.einsum("iv,jv->ij", df, df)
        b = torch.einsum("hv,v->h", df, f_current[0])
        if settings.diagonal_offset is not None:
            eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
            one = a.new_tensor(1.0)
            scale = torch.where(
                eye != 0,
                eye + settings.diagonal_offset**2,
                one,
            )
            a = a * scale

        if transition_active is not None:
            eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
            a = torch.where(transition_active, a, eye)
            b = torch.where(transition_active, b, torch.zeros_like(b))
        theta = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        x_bar = torch.einsum(
            "h,hv->v", theta, x_current[1 : n + 1] - x_current[0]
        )
        f_bar = torch.einsum("h,hv->v", theta, -df)
        x_bar = x_bar + x_current[0]
        f_bar = f_bar + f_current[0]
        mixed = x_bar + (settings.damp if damp is None else damp) * f_bar
    else:
        mixed = x_current[0] + f_current[0] * settings.damp_init

    rolled_x = torch.roll(x_current, 1, 0)
    next_state = AndersonState(
        x_hist=torch.cat((mixed.unsqueeze(0), rolled_x[1:]), dim=0),
        f_hist=torch.roll(f_current, 1, 0),
    )
    if transition_active is not None:
        next_state = _select_state(transition_active, next_state, previous)
    return mixed.reshape_as(mapped), next_state


def _solve_map_anderson(
    guess: Tensor,
    settings: AndersonSettings,
    map_fn,
    state_projection: Callable[[Tensor], Tensor] | None = None,
) -> _SolverOutcome:
    """Run the Anderson fixed-trip solver around a raw-map callback."""

    def transition(
        old: Tensor,
        mapped: Tensor,
        state: object | None,
        step: int,
        transition_active: Tensor,
        damp: Tensor,
    ) -> tuple[Tensor, object | None]:
        mixed, next_state = anderson_mix(
            old,
            mapped,
            cast(AndersonState | None, state),
            settings,
            step,
            transition_active,
            damp,
        )
        return mixed, next_state

    return _solve_map_control(
        guess,
        maxiter=settings.maxiter,
        mix_guess=settings.mix_guess,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        map_fn=map_fn,
        mix_fn=transition,
        state_projection=state_projection,
        initial_damp=settings.damp,
        damping_settings=settings.dynamic_damping,
    )


def solve_scf_anderson(
    guess: Tensor, inputs: SCFInputs, settings: AndersonSettings
) -> SCFSolveResult:
    """Solve the single-System charge map with functional Anderson mixing."""
    outcome = _solve_map_anderson(guess, settings, lambda x: scf_map(x, inputs))
    return finalize_scf(
        outcome.map_result,
        inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
    )


def solve_scf_potential_anderson(
    guess: Tensor,
    inputs: SCFInputs,
    layout: PotentialLayout,
    settings: AndersonSettings,
) -> SCFSolveResult:
    """Solve the potential representation with functional Anderson mixing."""
    outcome = _solve_map_anderson(
        guess,
        settings,
        lambda x: potential_map(x, inputs, layout),
        state_projection=lambda x: _project_potential_padding(x, layout),
    )
    return finalize_scf_potential(
        outcome.map_result,
        inputs,
        layout,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
    )


def solve_scf_fock_anderson(
    guess: Tensor, inputs: SCFInputs, settings: AndersonSettings
) -> SCFSolveResult:
    """Solve single-System Fock mode with functional Anderson mixing."""
    outcome = _solve_map_anderson(
        guess, settings, lambda x: fock_map(x, inputs)
    )
    return finalize_scf_fock(
        outcome.map_result,
        inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
    )
