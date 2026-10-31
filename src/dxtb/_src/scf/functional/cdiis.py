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
"""Fock-space commutator DIIS for functional SCF."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, cast

import torch

from dxtb._src.typing import Tensor

from .method import (
    SCFMethodContext,
    SCFMethodProposal,
    SCFMethodState,
    simple_mix,
)
from .problem import SCFPhysicalState, SCFProblem, SCFRepresentation
from .stationarity import (
    ao_stationarity_matrix,
    orthogonalized_stationarity_matrix,
)

__all__ = ["CDIISMethod", "CDIISMethodSettings", "CDIISState"]


@dataclass(frozen=True, eq=False)
class CDIISMethodSettings:
    """Static numerical policy for bounded Fock-space CDIIS history."""

    max_history: int = 8
    condition_limit: float | None = None
    relative_regularization: float | None = None

    def __post_init__(self) -> None:
        if self.max_history < 2:
            raise ValueError("CDIIS history must contain at least two slots.")
        if self.condition_limit is not None and (
            not isfinite(self.condition_limit) or self.condition_limit <= 1.0
        ):
            raise ValueError("CDIIS condition limit must be finite and > 1.")
        if self.relative_regularization is not None and (
            not isfinite(self.relative_regularization)
            or self.relative_regularization <= 0.0
        ):
            raise ValueError(
                "CDIIS regularization must be finite and positive."
            )

    def resolved_condition_limit(self, dtype: torch.dtype) -> float:
        """Use a dtype-aware limit for normalized history dependence."""
        if self.condition_limit is not None:
            return self.condition_limit
        return torch.finfo(dtype).eps ** -0.5

    def resolved_regularization(self, dtype: torch.dtype) -> float:
        """Use a small dtype-aware relative Gram regularization."""
        if self.relative_regularization is not None:
            return self.relative_regularization
        return torch.finfo(dtype).eps ** 0.75


@dataclass(frozen=True, eq=False)
class CDIISState:
    """Fixed-shape immutable Fock/error history and Pulay diagnostics."""

    fock_history: Tensor
    error_history: Tensor
    history_count: Tensor
    restart_count: Tensor
    prune_count: Tensor
    last_condition: Tensor
    last_regularization: Tensor
    last_coeff_max: Tensor
    last_solve_ok: Tensor


@dataclass(frozen=True, eq=False)
class CDIISMethod:
    """Conventional commutator CDIIS operating on the Fock representation."""

    supports_fixed_trip: ClassVar[bool] = True

    required_representation: ClassVar[SCFRepresentation] = (
        SCFRepresentation.FOCK
    )
    settings: CDIISMethodSettings = CDIISMethodSettings()

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state: SCFMethodState | None,
        context: SCFMethodContext,
    ) -> SCFMethodProposal:
        """Insert the fresh mapped Fock/error and make a damped proposal."""
        if problem.representation is not SCFRepresentation.FOCK:
            raise ValueError("CDIIS requires a Fock working representation.")

        old = cast(CDIISState | None, method_state)
        if old is None:
            old = _empty_state(physical_state.x, self.settings.max_history)

        inputs = problem.inputs
        pair_mask = (
            inputs.orbital_mask[..., :, None]
            & inputs.orbital_mask[..., None, :]
        )
        fresh_fock = context.proposal_x
        fresh_fock = 0.5 * (fresh_fock + fresh_fock.transpose(-2, -1))
        fresh_fock = torch.where(pair_mask, fresh_fock, 0.0)
        ao_error = ao_stationarity_matrix(
            fresh_fock,
            physical_state.map_result.spectral.density,
            inputs.integrals.overlap,
            inputs.orbital_mask,
        )
        fresh_error = orthogonalized_stationarity_matrix(
            ao_error, inputs.inverse_cholesky, inputs.orbital_mask
        )

        history, count, observation_ok, restart = _insert_observation(
            old, fresh_fock, fresh_error, self.settings.max_history
        )
        history, count, condition, pruned, conditioning_restart = (
            _prune_history(history, count, self.settings)
        )
        restart = restart | conditioning_restart
        coefficients, regularization, solve_ok = _solve_pulay_coefficients(
            history.error_history, count, self.settings
        )

        extrapolated = torch.einsum(
            "i,ijk->jk", coefficients, history.fock_history
        )
        extrapolated = 0.5 * (extrapolated + extrapolated.transpose(-2, -1))
        extrapolated = torch.where(pair_mask, extrapolated, 0.0)
        target = torch.where(solve_ok, extrapolated, fresh_fock)
        proposal_x = simple_mix(physical_state.x, target, context.damp)

        solve_restart = (count >= 2) & ~solve_ok
        restart = restart | solve_restart
        restart_count = observation_ok.to(torch.int64)
        final_count = torch.where(restart, restart_count, count)
        final_history = _truncate_history(history, final_count)
        final_history = _keep_current_only(
            final_history,
            fresh_fock,
            fresh_error,
            restart & observation_ok,
        )
        final_history = CDIISState(
            fock_history=final_history.fock_history,
            error_history=final_history.error_history,
            history_count=final_history.history_count,
            restart_count=old.restart_count + restart.to(torch.int64),
            prune_count=old.prune_count + pruned,
            last_condition=condition,
            last_regularization=regularization,
            last_coeff_max=coefficients.abs().amax(),
            last_solve_ok=solve_ok,
        )
        selected = _select_state(context.transition_active, final_history, old)
        proposal_x = torch.where(
            context.transition_active, proposal_x, physical_state.x
        )
        return SCFMethodProposal(x=proposal_x, state=selected)


def _empty_state(fock: Tensor, max_history: int) -> CDIISState:
    """Create zeroed fixed-shape history for one Fock matrix shape."""
    return CDIISState(
        fock_history=fock.new_zeros((max_history, *fock.shape)),
        error_history=fock.new_zeros((max_history, *fock.shape)),
        history_count=torch.zeros((), dtype=torch.int64, device=fock.device),
        restart_count=torch.zeros((), dtype=torch.int64, device=fock.device),
        prune_count=torch.zeros((), dtype=torch.int64, device=fock.device),
        last_condition=fock.new_zeros(()),
        last_regularization=fock.new_zeros(()),
        last_coeff_max=fock.new_zeros(()),
        last_solve_ok=torch.zeros((), dtype=torch.bool, device=fock.device),
    )


def _insert_observation(
    state: CDIISState, fock: Tensor, error: Tensor, max_history: int
) -> tuple[CDIISState, Tensor, Tensor, Tensor]:
    """Insert the current finite, resolvable observation as newest."""
    fock_finite = torch.isfinite(fock).all()
    error_finite = torch.isfinite(error).all()
    safe_fock = torch.where(fock_finite, fock, torch.zeros_like(fock))
    safe_error = torch.where(error_finite, error, torch.zeros_like(error))
    error_norm = torch.linalg.vector_norm(safe_error)
    fock_scale = torch.linalg.vector_norm(safe_fock).clamp_min(1.0)
    resolvable = error_norm > torch.finfo(error.dtype).eps * fock_scale
    observation_ok = fock_finite & error_finite & resolvable

    slots = torch.arange(max_history, device=fock.device)
    old_active = slots < state.history_count
    fock_mask = old_active.reshape((-1,) + (1,) * fock.ndim)
    error_mask = old_active.reshape((-1,) + (1,) * error.ndim)
    old_focks = torch.where(fock_mask, state.fock_history, 0.0)
    old_errors = torch.where(error_mask, state.error_history, 0.0)
    shifted_focks = torch.cat((safe_fock.unsqueeze(0), old_focks[:-1]), dim=0)
    shifted_errors = torch.cat(
        (safe_error.unsqueeze(0), old_errors[:-1]), dim=0
    )
    focks = torch.where(observation_ok, shifted_focks, 0.0)
    errors = torch.where(observation_ok, shifted_errors, 0.0)
    count = torch.minimum(
        state.history_count + 1,
        torch.full_like(state.history_count, max_history),
    )
    count = torch.where(observation_ok, count, torch.zeros_like(count))
    restart = ~observation_ok & (state.history_count > 0)
    return (
        CDIISState(
            fock_history=focks,
            error_history=errors,
            history_count=count,
            restart_count=state.restart_count,
            prune_count=state.prune_count,
            last_condition=state.last_condition,
            last_regularization=state.last_regularization,
            last_coeff_max=state.last_coeff_max,
            last_solve_ok=state.last_solve_ok,
        ),
        count,
        observation_ok,
        restart,
    )


def _normalized_error_gram(errors: Tensor, count: Tensor) -> Tensor:
    """Build a fixed-size normalized Gram with identity inactive slots."""
    max_history = errors.shape[0]
    flat = errors.reshape(max_history, -1)
    norms = torch.linalg.vector_norm(flat, dim=-1)
    safe_norms = torch.where(norms > 0.0, norms, torch.ones_like(norms))
    normalized = flat / safe_norms.unsqueeze(-1)
    gram = normalized @ normalized.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    active = torch.arange(max_history, device=errors.device) < count
    pairs = active[:, None] & active[None, :]
    identity = torch.eye(max_history, dtype=errors.dtype, device=errors.device)
    return (
        torch.where(pairs, gram, 0.0)
        + torch.diag((~active).to(errors.dtype)) * identity
    )


def _history_condition(errors: Tensor, count: Tensor) -> tuple[Tensor, Tensor]:
    """Return the normalized-history condition and finite validity."""
    gram = _normalized_error_gram(errors, count)
    eigenvalues = torch.linalg.eigvalsh(gram)
    minimum = eigenvalues.amin()
    maximum = eigenvalues.amax()
    condition = maximum / minimum.clamp_min(torch.finfo(errors.dtype).tiny)
    valid = torch.isfinite(eigenvalues).all() & torch.isfinite(condition)
    return condition, valid


def _prune_history(
    history: CDIISState,
    count: Tensor,
    settings: CDIISMethodSettings,
) -> tuple[CDIISState, Tensor, Tensor, Tensor, Tensor]:
    """Drop oldest vectors until conditioning is acceptable or restart."""
    maximum = settings.resolved_condition_limit(history.error_history.dtype)
    pruned = torch.zeros((), dtype=torch.int64, device=count.device)
    condition = history.error_history.new_ones(())
    condition_valid = torch.ones((), dtype=torch.bool, device=count.device)

    for _ in range(settings.max_history):
        condition, condition_valid = _history_condition(
            history.error_history, count
        )
        healthy = condition_valid & (condition <= maximum)
        remove_oldest = (count > 2) & ~healthy
        count = torch.where(remove_oldest, count - 1, count)
        pruned = pruned + remove_oldest.to(torch.int64)

    condition, condition_valid = _history_condition(
        history.error_history, count
    )
    healthy = condition_valid & (condition <= maximum)
    restart = (count >= 2) & ~healthy
    return _truncate_history(history, count), count, condition, pruned, restart


def _truncate_history(history: CDIISState, count: Tensor) -> CDIISState:
    """Zero all entries outside the active newest-first prefix."""
    slots = torch.arange(history.fock_history.shape[0], device=count.device)
    active = slots < count
    fock_mask = active.reshape((-1,) + (1,) * (history.fock_history.ndim - 1))
    error_mask = active.reshape((-1,) + (1,) * (history.error_history.ndim - 1))
    return CDIISState(
        fock_history=torch.where(
            fock_mask,
            history.fock_history,
            torch.zeros_like(history.fock_history),
        ),
        error_history=torch.where(
            error_mask,
            history.error_history,
            torch.zeros_like(history.error_history),
        ),
        history_count=count,
        restart_count=history.restart_count,
        prune_count=history.prune_count,
        last_condition=history.last_condition,
        last_regularization=history.last_regularization,
        last_coeff_max=history.last_coeff_max,
        last_solve_ok=history.last_solve_ok,
    )


def _keep_current_only(
    history: CDIISState,
    fock: Tensor,
    error: Tensor,
    restart: Tensor,
) -> CDIISState:
    """After an unhealthy solve, retain only the current valid observation."""
    count = torch.where(
        restart, torch.ones_like(history.history_count), history.history_count
    )
    focks = torch.cat(
        (fock.unsqueeze(0), torch.zeros_like(history.fock_history[1:])), dim=0
    )
    errors = torch.cat(
        (error.unsqueeze(0), torch.zeros_like(history.error_history[1:])), dim=0
    )
    return CDIISState(
        fock_history=torch.where(restart, focks, history.fock_history),
        error_history=torch.where(restart, errors, history.error_history),
        history_count=count,
        restart_count=history.restart_count,
        prune_count=history.prune_count,
        last_condition=history.last_condition,
        last_regularization=history.last_regularization,
        last_coeff_max=history.last_coeff_max,
        last_solve_ok=history.last_solve_ok,
    )


def _solve_pulay_coefficients(
    errors: Tensor,
    count: Tensor,
    settings: CDIISMethodSettings,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve the regularized affine Pulay system without dynamic slicing."""
    max_history = errors.shape[0]
    flat = errors.reshape(max_history, -1)
    gram_raw = flat @ flat.transpose(0, 1)
    gram_finite = torch.isfinite(gram_raw).all()
    gram = torch.where(torch.isfinite(gram_raw), gram_raw, 0.0)
    diagonal = torch.diagonal(gram)
    active = torch.arange(max_history, device=errors.device) < count
    active_count = count.clamp_min(1).to(errors.dtype)
    scale_raw = (diagonal * active.to(errors.dtype)).sum() / active_count
    scale_valid = torch.isfinite(scale_raw) & (scale_raw > 0.0)
    scale = torch.where(scale_valid, scale_raw, torch.ones_like(scale_raw))
    regularization = settings.resolved_regularization(errors.dtype) * scale

    active_pairs = active[:, None] & active[None, :]
    regularized = gram + torch.diag(active.to(errors.dtype)) * regularization
    regularized = torch.where(active_pairs, regularized, 0.0)
    regularized = regularized + torch.diag((~active).to(errors.dtype))
    regularized_finite = torch.isfinite(regularized).all()
    safe_regularized = torch.where(
        torch.isfinite(regularized), regularized, 0.0
    )
    condition, condition_valid = _history_condition(errors, count)
    usable = (
        (count >= 2)
        & gram_finite
        & scale_valid
        & regularized_finite
        & condition_valid
        & (condition <= settings.resolved_condition_limit(errors.dtype))
    )
    identity = torch.eye(max_history, dtype=errors.dtype, device=errors.device)
    solve_matrix = torch.where(usable, safe_regularized, identity)
    cholesky, info = torch.linalg.cholesky_ex(solve_matrix)
    factor_valid = (info == 0) & torch.isfinite(cholesky).all()
    safe_cholesky = torch.where(factor_valid, cholesky, identity)
    rhs = active.to(errors.dtype)
    rhs = torch.where(usable & factor_valid, rhs, torch.zeros_like(rhs))
    z = torch.cholesky_solve(rhs.unsqueeze(-1), safe_cholesky).squeeze(-1)
    denominator = z.sum()
    denominator_valid = torch.isfinite(denominator) & (
        denominator.abs() > torch.finfo(errors.dtype).tiny
    )
    safe_denominator = torch.where(
        denominator_valid, denominator, torch.ones_like(denominator)
    )
    coefficients = z / safe_denominator
    solve_ok = (
        usable
        & factor_valid
        & denominator_valid
        & torch.isfinite(coefficients).all()
    )
    coefficients = torch.where(
        solve_ok & active, coefficients, torch.zeros_like(coefficients)
    )
    return coefficients, regularization, solve_ok


def _select_state(mask: Tensor, new: CDIISState, old: CDIISState) -> CDIISState:
    """Select every state leaf and preserve inactive lanes exactly."""
    return CDIISState(
        fock_history=torch.where(mask, new.fock_history, old.fock_history),
        error_history=torch.where(mask, new.error_history, old.error_history),
        history_count=torch.where(mask, new.history_count, old.history_count),
        restart_count=torch.where(mask, new.restart_count, old.restart_count),
        prune_count=torch.where(mask, new.prune_count, old.prune_count),
        last_condition=torch.where(
            mask, new.last_condition, old.last_condition
        ),
        last_regularization=torch.where(
            mask, new.last_regularization, old.last_regularization
        ),
        last_coeff_max=torch.where(
            mask, new.last_coeff_max, old.last_coeff_max
        ),
        last_solve_ok=torch.where(mask, new.last_solve_ok, old.last_solve_ok),
    )
