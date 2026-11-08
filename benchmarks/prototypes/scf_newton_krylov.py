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
"""Research-only prototype for matrix-free physical SCF Newton refinement.

This module is intentionally outside ``src/dxtb``. S9's benchmark matrix did
not justify adopting Newton--Krylov at the current S2 tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, cast

import torch
from torch.func import jvp

from dxtb._src.scf.functional.method import (
    SCFConvergerContext,
    SCFConvergerProposal,
    SCFConvergerState,
)
from dxtb._src.scf.functional.problem import (
    SCFPhysicalState,
    SCFProblem,
    SCFVariable,
)
from dxtb._src.scf.functional.spectral import eager_primal_fermi_scope
from dxtb._src.typing import Tensor

__all__ = [
    "NewtonKrylovConverger",
    "NewtonKrylovSettings",
    "NewtonKrylovState",
    "fixed_point_jvp",
]


@dataclass(frozen=True, eq=False)
class NewtonKrylovSettings:
    """Internal controls for eager matrix-free physical-root refinement."""

    max_linear_iterations: int = 30
    linear_rtol: float = 1.0e-3
    linear_atol_factor: float = 10.0
    max_step_ratio: float = 1.0
    step_scale_floor: float = 1.0e-8
    fallback_damp: float = 0.1

    def __post_init__(self) -> None:
        if self.max_linear_iterations < 1:
            raise ValueError("Newton--Krylov iteration limit must be positive.")
        if not isfinite(self.linear_rtol) or not 0.0 < self.linear_rtol < 1.0:
            raise ValueError("Newton--Krylov linear_rtol must be in (0, 1).")
        if (
            not isfinite(self.linear_atol_factor)
            or self.linear_atol_factor < 0.0
        ):
            raise ValueError(
                "Newton--Krylov linear_atol_factor must be finite and nonnegative."
            )
        if not isfinite(self.max_step_ratio) or self.max_step_ratio <= 0.0:
            raise ValueError("Newton--Krylov max_step_ratio must be positive.")
        if not isfinite(self.step_scale_floor) or self.step_scale_floor <= 0.0:
            raise ValueError(
                "Newton--Krylov step_scale_floor must be positive."
            )
        if (
            not isfinite(self.fallback_damp)
            or not 0.0 < self.fallback_damp <= 1.0
        ):
            raise ValueError("Newton--Krylov fallback_damp must be in (0, 1].")


@dataclass(frozen=True, eq=False)
class NewtonKrylovState:
    """Immutable diagnostics from the most recent Newton--Krylov transition."""

    last_residual_norm: Tensor
    last_linear_iterations: Tensor
    last_linear_residual: Tensor
    last_linear_converged: Tensor
    last_step_norm: Tensor
    last_step_scale: Tensor
    last_jvp_count: Tensor
    used_fixed_point_fallback: Tensor
    fallback_count: Tensor


@dataclass(frozen=True, eq=False)
class _KrylovResult:
    solution: Tensor
    residual_norm: Tensor
    iterations: Tensor
    jvp_count: Tensor
    converged: Tensor


def _project_direction(problem: SCFProblem, value: Tensor) -> Tensor:
    """Apply structural projection and the symmetric Fock tangent constraint."""
    projected = problem.project_residual(value)
    if problem.variable is SCFVariable.FOCK:
        projected = 0.5 * (projected + projected.transpose(-2, -1))
        projected = problem.project_residual(projected)
    return projected


def fixed_point_jvp(
    problem: SCFProblem, x: Tensor, direction: Tensor
) -> Tensor:
    """Evaluate ``(I - G_x) direction`` for the projected physical map.

    The W2c10 adaptive Fermi schedule is explicitly disabled around the JVP.
    This keeps the derivative on the static matrix-Fermi path even when a
    detached eager root preparation has enabled its thread-local scope.
    """
    tangent = _project_direction(problem, direction)

    def mapped_value(value: Tensor) -> Tensor:
        return problem.map(value).mapped_x

    with eager_primal_fermi_scope(False):
        _, mapped_tangent = jvp(mapped_value, (x,), (tangent,))
    result = tangent - _project_direction(problem, mapped_tangent)
    return _project_direction(problem, result)


def _bicgstab(
    operator,
    rhs: Tensor,
    *,
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: Tensor,
) -> _KrylovResult:
    """Solve one nonsymmetric system and expose all matvec work."""
    shape = rhs.shape
    b = rhs.reshape(-1)
    zero = torch.zeros_like(b)
    x = zero.clone()
    r = b.clone()
    shadow = r.clone()
    norm_b = torch.linalg.vector_norm(b)
    threshold = torch.maximum(absolute_tolerance, relative_tolerance * norm_b)
    residual_norm = torch.linalg.vector_norm(r)
    iteration_value = torch.zeros((), dtype=torch.int64, device=rhs.device)
    jvp_count = torch.zeros_like(iteration_value)
    converged = residual_norm <= threshold
    if bool(converged):
        return _KrylovResult(
            x.reshape(shape),
            residual_norm,
            iteration_value,
            jvp_count,
            converged,
        )

    eps = torch.finfo(rhs.dtype).eps
    tiny = torch.finfo(rhs.dtype).tiny
    rho_old = rhs.new_ones(())
    alpha = rhs.new_ones(())
    omega = rhs.new_ones(())
    v = zero.clone()
    p = zero.clone()
    finite = torch.isfinite(b).all()

    for index in range(max_iterations):
        if not bool(finite) or bool(converged):
            break
        rho = torch.dot(shadow, r)
        rho_scale = torch.linalg.vector_norm(shadow) * torch.linalg.vector_norm(
            r
        )
        if not bool(torch.isfinite(rho)) or bool(
            rho.abs() <= torch.maximum(eps * rho_scale, rho.new_tensor(tiny))
        ):
            break
        if index == 0:
            p = r
        else:
            beta = (rho / rho_old) * (alpha / omega)
            if not bool(torch.isfinite(beta)):
                break
            p = r + beta * (p - omega * v)

        v = operator(p.reshape(shape)).reshape(-1)
        jvp_count = jvp_count + 1
        finite = torch.isfinite(v).all()
        if not bool(finite):
            break
        vdot = torch.dot(shadow, v)
        vdot_scale = torch.linalg.vector_norm(
            shadow
        ) * torch.linalg.vector_norm(v)
        if bool(
            vdot.abs() <= torch.maximum(eps * vdot_scale, vdot.new_tensor(tiny))
        ):
            break
        alpha = rho / vdot
        s = r - alpha * v
        s_norm = torch.linalg.vector_norm(s)
        if bool(torch.isfinite(s_norm)) and bool(s_norm <= threshold):
            x = x + alpha * p.reshape(-1)
            residual_norm = s_norm
            converged = torch.ones_like(converged)
            iteration_value = torch.tensor(
                index + 1, dtype=torch.int64, device=rhs.device
            )
            break

        t = operator(s.reshape(shape)).reshape(-1)
        jvp_count = jvp_count + 1
        finite = torch.isfinite(t).all()
        if not bool(finite):
            break
        t_dot_t = torch.dot(t, t)
        t_scale = torch.linalg.vector_norm(t).square()
        if bool(
            t_dot_t <= torch.maximum(eps * t_scale, t_dot_t.new_tensor(tiny))
        ):
            break
        omega = torch.dot(t, s) / t_dot_t
        if not bool(torch.isfinite(omega)):
            break
        x = x + alpha * p.reshape(-1) + omega * s
        r = s - omega * t
        residual_norm = torch.linalg.vector_norm(r)
        iteration_value = torch.tensor(
            index + 1, dtype=torch.int64, device=rhs.device
        )
        finite = torch.isfinite(x).all() & torch.isfinite(residual_norm)
        converged = finite & (residual_norm <= threshold)
        if bool(converged):
            break
        omega_scale = torch.linalg.vector_norm(s) * torch.linalg.vector_norm(t)
        if bool(
            omega.abs()
            <= torch.maximum(eps * omega_scale, omega.new_tensor(tiny))
        ):
            break
        rho_old = rho

    return _KrylovResult(
        x.reshape(shape), residual_norm, iteration_value, jvp_count, converged
    )


def _initial_state(x: Tensor) -> NewtonKrylovState:
    zero = x.new_zeros(())
    zero_i = torch.zeros((), dtype=torch.int64, device=x.device)
    return NewtonKrylovState(
        last_residual_norm=zero,
        last_linear_iterations=zero_i,
        last_linear_residual=zero,
        last_linear_converged=torch.zeros(
            (), dtype=torch.bool, device=x.device
        ),
        last_step_norm=zero,
        last_step_scale=zero,
        last_jvp_count=zero_i,
        used_fixed_point_fallback=torch.zeros(
            (), dtype=torch.bool, device=x.device
        ),
        fallback_count=zero_i,
    )


@dataclass(frozen=True, eq=False)
class NewtonKrylovConverger:
    """Eager, active-smearing Newton refinement in any SCF variable."""

    supports_fixed_trip: ClassVar[bool] = False
    settings: NewtonKrylovSettings = NewtonKrylovSettings()

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        converger_state: SCFConvergerState | None,
        context: SCFConvergerContext,
    ) -> SCFConvergerProposal:
        """Take a bounded Newton step or an explicitly diagnosed fixed-point step."""
        fermi = problem.inputs.fermi_settings
        if fermi is None or not fermi.smear:
            raise NotImplementedError(
                "NewtonKrylovConverger currently requires Fermi smearing at nonzero "
                "electronic temperature."
            )
        if not bool(context.transition_active):
            return SCFConvergerProposal(physical_state.x, converger_state)

        previous = cast(NewtonKrylovState | None, converger_state)
        old = _initial_state(physical_state.x) if previous is None else previous
        x = physical_state.x
        residual = _project_direction(problem, context.proposal_x - x)
        residual_norm = torch.linalg.vector_norm(residual)
        info = torch.finfo(x.dtype)
        absolute_tolerance = (
            self.settings.linear_atol_factor
            * info.eps
            * residual_norm.clamp_min(1.0)
        )

        def operator(direction: Tensor) -> Tensor:
            return fixed_point_jvp(problem, x, direction)

        linear = _bicgstab(
            operator,
            residual,
            max_iterations=self.settings.max_linear_iterations,
            relative_tolerance=self.settings.linear_rtol,
            absolute_tolerance=absolute_tolerance,
        )
        used_fallback = not bool(linear.converged)
        if used_fallback:
            step = _project_direction(problem, context.proposal_x - x)
            scale = torch.minimum(
                context.damp.new_tensor(self.settings.fallback_damp),
                context.damp,
            )
        else:
            step = _project_direction(problem, linear.solution)
            step_norm = torch.linalg.vector_norm(step)
            state_norm = torch.linalg.vector_norm(x).clamp_min(
                self.settings.step_scale_floor
            )
            maximum_step = self.settings.max_step_ratio * state_norm
            trust_scale = maximum_step / step_norm.clamp_min(info.tiny)
            scale = torch.minimum(context.damp, trust_scale).clamp(max=1.0)

        accepted_step = _project_direction(problem, scale * step)
        proposed_x = x + accepted_step
        step_norm = torch.linalg.vector_norm(accepted_step)
        fallback_count = old.fallback_count + torch.tensor(
            int(used_fallback), dtype=torch.int64, device=x.device
        )
        new_state = NewtonKrylovState(
            last_residual_norm=residual_norm,
            last_linear_iterations=linear.iterations,
            last_linear_residual=linear.residual_norm,
            last_linear_converged=linear.converged,
            last_step_norm=step_norm,
            last_step_scale=scale,
            last_jvp_count=linear.jvp_count,
            used_fixed_point_fallback=torch.tensor(
                used_fallback, dtype=torch.bool, device=x.device
            ),
            fallback_count=fallback_count,
        )
        return SCFConvergerProposal(proposed_x, new_state)
