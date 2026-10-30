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
"""Anderson method and immutable method-local history for functional SCF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor

from .method import (
    SCFMethodContext,
    SCFMethodProposal,
    SCFMethodState,
)
from .problem import SCFPhysicalState, SCFProblem

__all__ = [
    "AndersonMethod",
    "AndersonMethodSettings",
    "AndersonState",
    "anderson_mix",
    "setup_anderson_method_settings",
]


@dataclass(frozen=True, eq=False)
class AndersonMethodSettings:
    """Static numerical policy owned specifically by the Anderson method."""

    damp_init: float
    generations: int
    soft_start: bool
    diagonal_offset: float | None


@dataclass(frozen=True, eq=False)
class AndersonState:
    """Immutable, fixed-size history for one SCF method transition."""

    x_hist: Tensor
    f_hist: Tensor


@dataclass(frozen=True, eq=False)
class AndersonMethod:
    """Anderson proposal rule with explicit immutable history."""

    settings: AndersonMethodSettings

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state: SCFMethodState | None,
        context: SCFMethodContext,
    ) -> SCFMethodProposal:
        """Propose one Anderson transition from the paired physical state."""
        del problem
        mixed, next_state = anderson_mix(
            physical_state.x,
            context.proposal_x,
            cast(AndersonState | None, method_state),
            self.settings,
            context.step,
            transition_active=context.transition_active,
            damp=context.damp,
        )
        return SCFMethodProposal(x=mixed, state=next_state)


def setup_anderson_method_settings(
    config: ConfigSCF,
) -> AndersonMethodSettings:
    """Resolve and validate only the effective Anderson method options."""
    if config.mixer != labels.MIXER_ANDERSON:
        raise NotImplementedError(
            "Anderson settings require the effective Anderson mixer."
        )
    if config.damp_generations < 1:
        raise ValueError("Anderson generations must be at least 1.")
    return AndersonMethodSettings(
        damp_init=config.damp_init,
        generations=config.damp_generations,
        soft_start=config.damp_soft_start,
        diagonal_offset=config.damp_diagonal_offset,
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
    settings: AndersonMethodSettings,
    step: int,
    transition_active: Tensor | None = None,
    *,
    damp: Tensor | float,
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
        f_hist = flat_old.new_zeros((settings.generations + 1, flat_old.numel()))
        previous = AndersonState(x_hist=x_hist, f_hist=f_hist)
    else:
        previous = state

    raw = flat_mapped - flat_old
    f_current = torch.cat((raw.unsqueeze(0), previous.f_hist[1:]), dim=0)
    x_current = torch.cat((flat_old.unsqueeze(0), previous.x_hist[1:]), dim=0)

    accelerate = step > settings.generations or (step > 1 and not settings.soft_start)
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
        x_bar = torch.einsum("h,hv->v", theta, x_current[1 : n + 1] - x_current[0])
        f_bar = torch.einsum("h,hv->v", theta, -df)
        x_bar = x_bar + x_current[0]
        f_bar = f_bar + f_current[0]
        mixed = x_bar + damp * f_bar
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
