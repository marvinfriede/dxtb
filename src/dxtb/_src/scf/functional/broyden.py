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
"""Limited-memory Broyden's first method for functional SCF."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import cast

import torch

from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor

from .method import SCFMethodContext, SCFMethodProposal, SCFMethodState
from .problem import SCFPhysicalState, SCFProblem

__all__ = [
    "BroydenMethod",
    "BroydenMethodSettings",
    "BroydenState",
    "setup_broyden_method_settings",
]


@dataclass(frozen=True, eq=False)
class BroydenMethodSettings:
    """Static numerical policy owned specifically by Broyden's first method."""

    initial_inverse_scale: float = 0.5
    max_rank: int = 8
    secant_quality_tolerance: float | None = None

    def __post_init__(self) -> None:
        if (
            not isfinite(self.initial_inverse_scale)
            or self.initial_inverse_scale <= 0.0
        ):
            raise ValueError(
                "Broyden initial inverse scale must be finite and positive."
            )
        if self.max_rank < 1:
            raise ValueError("Broyden history rank must be at least 1.")
        tolerance = self.secant_quality_tolerance
        if tolerance is not None and not (0.0 <= tolerance < 1.0):
            raise ValueError(
                "Broyden secant quality tolerance must be in [0, 1)."
            )


@dataclass(frozen=True, eq=False)
class BroydenState:
    """Immutable fixed-size secant history and update-health diagnostics."""

    previous_x: Tensor
    previous_f: Tensor
    c_history: Tensor
    d_history: Tensor
    valid_history: Tensor
    history_count: Tensor
    restart_count: Tensor
    last_update_accepted: Tensor


@dataclass(frozen=True, eq=False)
class BroydenMethod:
    """Broyden's first inverse update with bounded low-rank history."""

    settings: BroydenMethodSettings = BroydenMethodSettings()

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state: SCFMethodState | None,
        context: SCFMethodContext,
    ) -> SCFMethodProposal:
        """Propose a damped step using the controller-projected map value."""
        del problem
        x = physical_state.x
        flat_x = x.reshape(-1)
        flat_proposal = context.proposal_x.reshape(-1)

        # The controller residual is r=G(x)-x, while the legacy rootfinder
        # solves F=x-G(x) and applies -H F. Thus the undamped proposal is H r.
        residual_f = flat_x - flat_proposal
        old = cast(BroydenState | None, method_state)
        if old is None:
            state = _empty_state(flat_x, self.settings.max_rank)
            state = _store_observation(state, flat_x, residual_f)
        else:
            state = _update_inverse(old, flat_x, residual_f, self.settings)

        step = -_apply_inverse(
            state, residual_f, self.settings.initial_inverse_scale
        )
        finite_step = torch.isfinite(step).all()
        reset_state = _clear_corrections(state, increment_restart=True)
        state = _select_state(finite_step, state, reset_state)
        safe_step = torch.where(
            finite_step,
            step,
            -self.settings.initial_inverse_scale * residual_f,
        )
        output = (flat_x + context.damp * safe_step).reshape_as(x)

        if old is not None:
            state = _select_state(context.transition_active, state, old)
        return SCFMethodProposal(x=output, state=state)


def setup_broyden_method_settings(config: ConfigSCF) -> BroydenMethodSettings:
    """Validate that the effective configuration selects native Broyden."""
    if config.mixer != labels.MIXER_BROYDEN:
        raise NotImplementedError("Broyden settings require the Broyden mixer.")
    return BroydenMethodSettings()


def _empty_state(vector: Tensor, max_rank: int) -> BroydenState:
    """Create a fixed-shape zero-correction state for one vector shape."""
    return BroydenState(
        previous_x=vector,
        previous_f=torch.zeros_like(vector),
        c_history=vector.new_zeros((max_rank, vector.numel())),
        d_history=vector.new_zeros((max_rank, vector.numel())),
        valid_history=torch.zeros(
            (max_rank,), dtype=torch.bool, device=vector.device
        ),
        history_count=torch.zeros((), dtype=torch.int64, device=vector.device),
        restart_count=torch.zeros((), dtype=torch.int64, device=vector.device),
        last_update_accepted=torch.zeros(
            (), dtype=torch.bool, device=vector.device
        ),
    )


def _store_observation(
    state: BroydenState, x: Tensor, f: Tensor
) -> BroydenState:
    """Store the current raw-map input and its rootfinder residual."""
    return BroydenState(
        previous_x=x,
        previous_f=f,
        c_history=state.c_history,
        d_history=state.d_history,
        valid_history=state.valid_history,
        history_count=state.history_count,
        restart_count=state.restart_count,
        last_update_accepted=state.last_update_accepted,
    )


def _apply_inverse(
    state: BroydenState, vector: Tensor, initial_inverse_scale: float
) -> Tensor:
    """Apply H=h0 I+sum(c_i d_i.T) without materializing H."""
    coefficients = torch.sum(state.d_history * vector.unsqueeze(0), dim=-1)
    coefficients = torch.where(
        state.valid_history, coefficients, torch.zeros_like(coefficients)
    )
    correction = torch.sum(state.c_history * coefficients.unsqueeze(-1), dim=0)
    return initial_inverse_scale * vector + correction


def _apply_inverse_transpose(
    state: BroydenState, vector: Tensor, initial_inverse_scale: float
) -> Tensor:
    """Apply H.T=h0 I+sum(d_i c_i.T) in low-rank storage."""
    coefficients = torch.sum(state.c_history * vector.unsqueeze(0), dim=-1)
    coefficients = torch.where(
        state.valid_history, coefficients, torch.zeros_like(coefficients)
    )
    correction = torch.sum(state.d_history * coefficients.unsqueeze(-1), dim=0)
    return initial_inverse_scale * vector + correction


def _select_state(
    mask: Tensor, new: BroydenState, old: BroydenState
) -> BroydenState:
    """Select every tensor leaf without Python inspection of lane state."""
    return BroydenState(
        previous_x=torch.where(mask, new.previous_x, old.previous_x),
        previous_f=torch.where(mask, new.previous_f, old.previous_f),
        c_history=torch.where(mask, new.c_history, old.c_history),
        d_history=torch.where(mask, new.d_history, old.d_history),
        valid_history=torch.where(mask, new.valid_history, old.valid_history),
        history_count=torch.where(mask, new.history_count, old.history_count),
        restart_count=torch.where(mask, new.restart_count, old.restart_count),
        last_update_accepted=torch.where(
            mask, new.last_update_accepted, old.last_update_accepted
        ),
    )


def _clear_corrections(
    state: BroydenState, *, increment_restart: bool
) -> BroydenState:
    """Keep the previous observation and clear all low-rank corrections."""
    increment = 1 if increment_restart else 0
    return BroydenState(
        previous_x=state.previous_x,
        previous_f=state.previous_f,
        c_history=torch.zeros_like(state.c_history),
        d_history=torch.zeros_like(state.d_history),
        valid_history=torch.zeros_like(state.valid_history),
        history_count=torch.zeros_like(state.history_count),
        restart_count=state.restart_count + increment,
        last_update_accepted=torch.zeros_like(state.last_update_accepted),
    )


def _update_inverse(
    state: BroydenState,
    current_x: Tensor,
    current_f: Tensor,
    settings: BroydenMethodSettings,
) -> BroydenState:
    """Apply Broyden's first inverse update or restart safely at H0."""
    full = state.history_count >= settings.max_rank
    empty_c = torch.zeros_like(state.c_history)
    empty_d = torch.zeros_like(state.d_history)
    base_c = torch.where(full, empty_c, state.c_history)
    base_d = torch.where(full, empty_d, state.d_history)
    base_valid = torch.where(
        full, torch.zeros_like(state.valid_history), state.valid_history
    )
    base_count = torch.where(
        full, torch.zeros_like(state.history_count), state.history_count
    )
    base = BroydenState(
        previous_x=state.previous_x,
        previous_f=state.previous_f,
        c_history=base_c,
        d_history=base_d,
        valid_history=base_valid,
        history_count=base_count,
        restart_count=state.restart_count,
        last_update_accepted=state.last_update_accepted,
    )

    # For F=x-G(x), the inverse update is
    # H_new = H + (s-Hy) (H.T s).T / (y.T H.T s).
    s = current_x - state.previous_x
    y = current_f - state.previous_f
    h_y = _apply_inverse(base, y, settings.initial_inverse_scale)
    v = _apply_inverse_transpose(base, s, settings.initial_inverse_scale)
    c = s - h_y
    denominator = torch.dot(y, v)
    norm_y = torch.linalg.vector_norm(y)
    norm_v = torch.linalg.vector_norm(v)
    scale = norm_y * norm_v
    quality = denominator.abs() / torch.where(
        scale > 0.0, scale, torch.ones_like(scale)
    )
    quality_tolerance = settings.secant_quality_tolerance
    if quality_tolerance is None:
        quality_tolerance = sqrt(torch.finfo(current_x.dtype).eps)

    update_valid = (
        (norm_y > 0.0)
        & (norm_v > 0.0)
        & torch.isfinite(denominator)
        & (denominator != 0.0)
        & torch.isfinite(scale)
        & torch.isfinite(quality)
        & (quality >= quality_tolerance)
        & torch.isfinite(s).all()
        & torch.isfinite(y).all()
        & torch.isfinite(h_y).all()
        & torch.isfinite(v).all()
        & torch.isfinite(c).all()
    )
    safe_denominator = torch.where(
        update_valid, denominator, torch.ones_like(denominator)
    )
    d = v / safe_denominator
    update_valid = update_valid & torch.isfinite(d).all()

    insertion = (
        torch.arange(settings.max_rank, device=current_x.device) == base_count
    )
    insert = insertion.unsqueeze(-1) & update_valid
    candidate_c = torch.where(insert, c.unsqueeze(0), base.c_history)
    candidate_d = torch.where(insert, d.unsqueeze(0), base.d_history)
    candidate_valid = base.valid_history | (insertion & update_valid)
    candidate_count = base_count + update_valid.to(torch.int64)
    return BroydenState(
        previous_x=current_x,
        previous_f=current_f,
        c_history=torch.where(update_valid, candidate_c, empty_c),
        d_history=torch.where(update_valid, candidate_d, empty_d),
        valid_history=torch.where(
            update_valid, candidate_valid, torch.zeros_like(base_valid)
        ),
        history_count=torch.where(
            update_valid, candidate_count, torch.zeros_like(base_count)
        ),
        restart_count=state.restart_count
        + (full | ~update_valid).to(torch.int64),
        last_update_accepted=update_valid,
    )
