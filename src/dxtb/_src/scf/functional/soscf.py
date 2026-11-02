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
"""Eager fixed-occupation second-order SCF in Fock representation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar, cast

import torch

from dxtb._src.typing import Tensor

from .method import SCFMethodContext, SCFMethodProposal, SCFMethodState
from .orbital import SOSCFOrbitalModel, build_orbital_model
from .problem import SCFPhysicalState, SCFProblem, SCFRepresentation

__all__ = ["SOSCFMethod", "SOSCFMethodSettings", "SOSCFState"]


@dataclass(frozen=True, eq=False)
class SOSCFMethodSettings:
    """Provisional internal controls for eager fixed-occupation SOSCF."""

    max_microiterations: int = 40
    micro_rtol: float = 1.0e-3
    micro_atol_factor: float = 100.0
    stationary_atol_factor: float = 32.0
    initial_level_shift: float = 0.05
    level_shift_growth: float = 2.0
    max_shift_attempts: int = 8
    initial_trust_radius: float = 0.2
    minimum_trust_radius: float = 1.0e-5
    maximum_trust_radius: float = 0.5
    max_backtracks: int = 8
    armijo_constant: float = 1.0e-4
    occupation_tolerance: float = 1.0e-10
    gap_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.max_microiterations < 1 or self.max_shift_attempts < 1:
            raise ValueError(
                "SOSCF iteration and retry limits must be positive."
            )
        if (
            self.max_backtracks < 1
            or not isfinite(self.level_shift_growth)
            or self.level_shift_growth <= 1.0
        ):
            raise ValueError("SOSCF backtracking/shift settings are invalid.")
        if not isfinite(self.micro_rtol) or not 0.0 < self.micro_rtol < 1.0:
            raise ValueError("SOSCF micro_rtol must be in (0, 1).")
        if not isfinite(self.micro_atol_factor) or self.micro_atol_factor < 0.0:
            raise ValueError(
                "SOSCF micro_atol_factor must be finite and nonnegative."
            )
        if (
            not isfinite(self.stationary_atol_factor)
            or self.stationary_atol_factor <= 0.0
        ):
            raise ValueError(
                "SOSCF stationary_atol_factor must be finite and positive."
            )
        if (
            not isfinite(self.initial_level_shift)
            or self.initial_level_shift <= 0.0
        ):
            raise ValueError(
                "SOSCF initial_level_shift must be finite and positive."
            )
        if not all(
            isfinite(value)
            for value in (
                self.initial_trust_radius,
                self.minimum_trust_radius,
                self.maximum_trust_radius,
            )
        ) or not (
            0.0
            < self.minimum_trust_radius
            <= self.initial_trust_radius
            <= self.maximum_trust_radius
        ):
            raise ValueError("SOSCF trust radii are inconsistent.")
        if (
            not isfinite(self.armijo_constant)
            or not 0.0 < self.armijo_constant < 1.0
        ):
            raise ValueError("SOSCF Armijo constant must be in (0, 1).")
        for name, value in (
            ("occupation_tolerance", self.occupation_tolerance),
            ("gap_tolerance", self.gap_tolerance),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"SOSCF {name} must be finite and positive.")


@dataclass(frozen=True, eq=False)
class SOSCFState:
    """Immutable macroiteration diagnostics and trust/step-radius controls.

    ``last_predicted_reduction`` is the linear directional decrease
    ``-gradient.dot(step)`` used by the Armijo test, not a quadratic model
    reduction.
    """

    trust_radius: Tensor
    level_shift: Tensor
    last_gradient_norm: Tensor
    last_step_norm: Tensor
    last_step_max_abs: Tensor
    last_microiterations: Tensor
    last_inner_residual: Tensor
    last_hvp_count: Tensor
    last_line_search_alpha: Tensor
    last_line_search_trials: Tensor
    last_predicted_reduction: Tensor
    last_actual_reduction: Tensor
    last_negative_curvature: Tensor
    last_gradient_fallback: Tensor
    last_step_accepted: Tensor
    fallback_count: Tensor


class _SOSCFHessianOperator:
    """Matrix-free Hessian-vector products of the fixed-occupation energy."""

    def __init__(self, model: SOSCFOrbitalModel, origin: Tensor) -> None:
        self.model = model
        self.origin = origin
        self.gradient = torch.func.grad(model.energy_at)

    def __call__(self, vector: Tensor) -> Tensor:
        return torch.func.jvp(self.gradient, (self.origin,), (vector,))[1]


def _initial_state(
    reference: Tensor, settings: SOSCFMethodSettings
) -> SOSCFState:
    """Create tensor diagnostics on the working device."""
    scalar = reference.new_zeros(())
    integer = torch.zeros((), dtype=torch.int64, device=reference.device)
    boolean = torch.zeros((), dtype=torch.bool, device=reference.device)
    return SOSCFState(
        trust_radius=scalar.new_tensor(settings.initial_trust_radius),
        level_shift=scalar,
        last_gradient_norm=scalar,
        last_step_norm=scalar,
        last_step_max_abs=scalar,
        last_microiterations=integer,
        last_inner_residual=scalar,
        last_hvp_count=integer,
        last_line_search_alpha=scalar,
        last_line_search_trials=integer,
        last_predicted_reduction=scalar,
        last_actual_reduction=scalar,
        last_negative_curvature=boolean,
        last_gradient_fallback=boolean,
        last_step_accepted=boolean,
        fallback_count=integer,
    )


def _positive_preconditioner(
    model: SOSCFOrbitalModel, fresh_fock: Tensor, settings: SOSCFMethodSettings
) -> tuple[Tensor, Tensor]:
    """Estimate positive orbital gaps from the fresh physical Fock."""
    fock_mo = model.coefficients.mT @ fresh_fock @ model.coefficients
    energies = torch.diagonal(fock_mo)
    denominator = (
        2.0
        * model.occupation_difference
        * (energies[model.pair_a] - energies[model.pair_i])
    )
    denominator_scale = denominator.abs().amax().clamp_min(1.0)
    floor = torch.finfo(fresh_fock.dtype).eps ** 0.5 * denominator_scale
    fock_scale = fresh_fock.detach().abs().amax().clamp_min(1.0)
    gap_floor = (
        max(settings.gap_tolerance, torch.finfo(fresh_fock.dtype).eps ** 0.5)
        * fock_scale
    )
    return denominator.clamp_min(floor), gap_floor


def _stationary_gradient_floor(
    fresh_fock: Tensor, nrot: int, settings: SOSCFMethodSettings
) -> Tensor:
    """Return an eps-scale floor for the numerical-zero stationary shortcut."""
    scale = fresh_fock.detach().abs().amax().clamp_min(1.0)
    return (
        settings.stationary_atol_factor
        * torch.finfo(fresh_fock.dtype).eps
        * scale
        * max(nrot, 1) ** 0.5
    )


def _occupation_sector_spectra(
    model: SOSCFOrbitalModel, fock_mo: Tensor
) -> tuple[tuple[float, Tensor], ...]:
    """Return eigenvalue ranges of the active equal-occupation Fock blocks."""
    total_occupation = model.occupation.sum(dim=-2).detach().cpu().tolist()
    orbital_mask = model.problem.inputs.orbital_mask.detach().cpu().tolist()
    sectors = sorted(
        {
            float(value)
            for value, active in zip(total_occupation, orbital_mask)
            if active
        },
        reverse=True,
    )
    spectra: list[tuple[float, Tensor]] = []
    detached_fock = 0.5 * (fock_mo.detach() + fock_mo.detach().mT)
    for sector in sectors:
        indices = [
            index
            for index, (value, active) in enumerate(
                zip(total_occupation, orbital_mask)
            )
            if active and float(value) == sector
        ]
        index_tensor = torch.tensor(
            indices, dtype=torch.int64, device=fock_mo.device
        )
        block = detached_fock.index_select(0, index_tensor).index_select(
            1, index_tensor
        )
        spectra.append((sector, torch.linalg.eigvalsh(block)))
    return tuple(spectra)


def _occupation_sector_order_valid(
    model: SOSCFOrbitalModel, fock_mo: Tensor, gap_floor: Tensor
) -> bool:
    """Check gauge-invariant spectral separation of occupied sectors."""
    spectra = _occupation_sector_spectra(model, fock_mo)
    return all(
        bool(high_values[-1] + gap_floor < low_values[0])
        for (_, high_values), (_, low_values) in zip(spectra, spectra[1:])
    )


def _pcg_newton_step(
    operator: _SOSCFHessianOperator,
    gradient: Tensor,
    preconditioner: Tensor,
    initial_shift: Tensor,
    settings: SOSCFMethodSettings,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
    """Solve the shifted Newton system with restarted preconditioned CG."""
    eps = torch.finfo(gradient.dtype).eps
    target = torch.maximum(
        gradient.new_tensor(settings.micro_atol_factor * eps),
        settings.micro_rtol * torch.linalg.vector_norm(gradient),
    )
    floor = eps**0.5 * preconditioner.abs().amax().clamp_min(1.0)
    shift = initial_shift
    hvps = torch.zeros((), dtype=torch.int64, device=gradient.device)
    negative = torch.tensor(False, device=gradient.device)
    last_residual = torch.linalg.vector_norm(gradient)
    last_iterations = torch.zeros_like(hvps)

    for _ in range(settings.max_shift_attempts):
        step = torch.zeros_like(gradient)
        residual = -gradient
        z = residual / preconditioner.clamp_min(floor)
        direction = z
        rz = torch.dot(residual, z)
        failed = False
        inner_converged = False
        if not bool(torch.isfinite(rz)) or not bool(
            torch.isfinite(direction).all()
        ):
            failed = True
        for iteration in range(settings.max_microiterations):
            if failed:
                break
            h_direction = operator(direction)
            hvps = hvps + 1
            if not bool(torch.isfinite(h_direction).all()):
                failed = True
                break
            shifted = h_direction + shift * direction
            curvature = torch.dot(direction, shifted)
            curvature_scale = torch.linalg.vector_norm(
                direction
            ) * torch.linalg.vector_norm(shifted)
            curvature_floor = eps * curvature_scale
            if not bool(torch.isfinite(curvature)) or bool(
                curvature <= curvature_floor
            ):
                negative = torch.tensor(True, device=gradient.device)
                failed = True
                break
            alpha = rz / curvature
            if not bool(torch.isfinite(alpha)):
                failed = True
                break
            step = step + alpha * direction
            residual = residual - alpha * shifted
            last_residual = torch.linalg.vector_norm(residual)
            if not bool(torch.isfinite(last_residual)):
                failed = True
                break
            last_iterations = torch.tensor(
                iteration + 1, dtype=torch.int64, device=gradient.device
            )
            if bool(last_residual <= target):
                inner_converged = True
                break
            z = residual / preconditioner.clamp_min(floor)
            rz_new = torch.dot(residual, z)
            if not bool(torch.isfinite(rz_new)) or bool(rz <= 0.0):
                failed = True
                break
            direction = (
                z
                + (rz_new / rz.clamp_min(torch.finfo(gradient.dtype).tiny))
                * direction
            )
            if not bool(torch.isfinite(direction).all()):
                failed = True
                break
            rz = rz_new
        if (
            not failed
            and inner_converged
            and bool(torch.dot(gradient, step) < 0.0)
            and bool(torch.isfinite(step).all())
        ):
            return (
                step,
                shift,
                last_iterations,
                last_residual,
                hvps,
                negative,
                False,
            )
        shift = shift * settings.level_shift_growth

    fallback = -gradient / preconditioner.clamp_min(floor)
    if not bool(torch.isfinite(fallback).all()) or not bool(
        torch.dot(gradient, fallback) < 0.0
    ):
        raise RuntimeError("SOSCF could not construct a finite descent step.")
    return fallback, shift, last_iterations, last_residual, hvps, negative, True


def _build_fock_seed(
    model: SOSCFOrbitalModel,
    rotation: Tensor,
    fresh_fock: Tensor,
    gap_floor: Tensor | None = None,
) -> Tensor:
    """Build the next input from fresh Fock blocks in accepted orbitals."""
    coefficients = model.coefficients_at(rotation)
    fock_mo = coefficients.mT @ fresh_fock @ coefficients
    if gap_floor is None:
        gap_floor = torch.finfo(
            fresh_fock.dtype
        ).eps ** 0.5 * fock_mo.detach().abs().amax().clamp_min(1.0)
    if not _occupation_sector_order_valid(model, fock_mo, gap_floor):
        raise ValueError(
            "SOSCF cannot build a Fock seed with overlapping occupation-sector "
            "spectra."
        )
    occupations = model.occupation.sum(dim=-2)
    same_sector = occupations[:, None] == occupations[None, :]

    # Equal-occupation rotations are gauge directions. Retaining their fresh
    # Fock blocks preserves the represented density while avoiding a second
    # eigensolve; at a stationary point only cross-sector blocks vanish.
    fock_blocks = torch.where(same_sector, fock_mo, 0.0)
    overlap = model.problem.inputs.integrals.overlap
    seed = overlap @ coefficients @ fock_blocks @ coefficients.mT @ overlap
    return 0.5 * (seed + seed.mT)


def _backtracking_step(
    model: SOSCFOrbitalModel,
    step: Tensor,
    energy0: Tensor,
    gap_floor: Tensor,
    initial_alpha: Tensor,
    linear_predicted_reduction: Tensor,
    settings: SOSCFMethodSettings,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """Accept the first Armijo step with separated occupation spectra."""
    eps = torch.finfo(step.dtype).eps
    alpha = initial_alpha
    for line_search_trials in range(1, settings.max_backtracks + 1):
        trial = alpha * step
        trial_energy = model.energy_at(trial)
        trial_fock = model.fock_at(trial)
        if not bool(torch.isfinite(trial_energy)) or not bool(
            torch.isfinite(trial_fock).all()
        ):
            alpha = alpha * 0.5
            continue

        coefficients = model.coefficients_at(trial)
        trial_mo = coefficients.mT @ trial_fock @ coefficients
        order_ok = _occupation_sector_order_valid(model, trial_mo, gap_floor)
        slack = 32.0 * eps * energy0.abs().clamp_min(1.0)
        armijo_ok = bool(
            trial_energy
            <= energy0
            - settings.armijo_constant * alpha * linear_predicted_reduction
            + slack
        )
        if order_ok and armijo_ok:
            return trial, trial_fock, trial_energy, alpha, line_search_trials
        alpha = alpha * 0.5

    raise RuntimeError(
        "SOSCF energy line search could not accept a fixed-occupation "
        "orbital step with separated occupation-sector spectra."
    )


@dataclass(frozen=True, eq=False)
class SOSCFMethod:
    """Eager fixed-occupation orbital Newton method in Fock space."""

    required_representation: ClassVar[SCFRepresentation] = (
        SCFRepresentation.FOCK
    )
    supports_fixed_trip: ClassVar[bool] = False
    settings: SOSCFMethodSettings = SOSCFMethodSettings()

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state: SCFMethodState | None,
        context: SCFMethodContext,
    ) -> SCFMethodProposal:
        """Compute one damped, energy-accepted orbital Newton proposal."""
        if problem.representation is not SCFRepresentation.FOCK:
            raise ValueError("SOSCF requires a Fock working representation.")
        if not bool(context.transition_active):
            return SCFMethodProposal(
                physical_state.x,
                (
                    None
                    if method_state is None
                    else cast(SOSCFState, method_state)
                ),
            )

        model = build_orbital_model(
            problem,
            physical_state.map_result.spectral,
            occupation_tolerance=self.settings.occupation_tolerance,
        )
        fresh_fock = context.proposal_x
        zero = physical_state.x.new_zeros(model.nrot)
        gradient = model.physical_gradient_at(zero, fresh_fock)
        if not bool(torch.isfinite(gradient).all()) or not bool(
            torch.isfinite(fresh_fock).all()
        ):
            raise RuntimeError("SOSCF received a non-finite physical state.")
        if not bool(torch.isfinite(context.damp)) or not bool(
            (context.damp > 0.0) & (context.damp <= 1.0)
        ):
            raise RuntimeError(
                "SOSCF requires finite controller damping in (0, 1]."
            )

        previous = (
            _initial_state(physical_state.x, self.settings)
            if method_state is None
            else cast(SOSCFState, method_state)
        )
        preconditioner, gap_floor = _positive_preconditioner(
            model, fresh_fock, self.settings
        )
        if not bool(torch.isfinite(preconditioner).all()):
            raise RuntimeError("SOSCF orbital preconditioner is non-finite.")
        fresh_mo = model.coefficients.mT @ fresh_fock @ model.coefficients
        if not _occupation_sector_order_valid(model, fresh_mo, gap_floor):
            raise RuntimeError(
                "SOSCF starting state has overlapping differently occupied "
                "Fock-block spectra outside the smooth fixed-Aufbau sector."
            )

        # If the current orbitals are stationary to working precision, an
        # orbital step cannot improve them. The input Fock can still have
        # stale eigenvalues, so rebuild the density-consistent seed and let
        # the shared controller perform the next physical map/certification.
        gradient_norm = torch.linalg.vector_norm(gradient)
        stationary_floor = _stationary_gradient_floor(
            fresh_fock, model.nrot, self.settings
        )
        if bool(gradient_norm <= stationary_floor):
            seed = _build_fock_seed(model, zero, fresh_fock, gap_floor)
            if not bool(torch.isfinite(seed).all()):
                raise RuntimeError("SOSCF constructed a non-finite Fock seed.")
            next_state = SOSCFState(
                trust_radius=previous.trust_radius,
                level_shift=previous.level_shift,
                last_gradient_norm=gradient_norm,
                last_step_norm=gradient_norm.new_zeros(()),
                last_step_max_abs=gradient_norm.new_zeros(()),
                last_microiterations=torch.zeros_like(
                    previous.last_microiterations
                ),
                last_inner_residual=gradient_norm,
                last_hvp_count=torch.zeros_like(previous.last_hvp_count),
                last_line_search_alpha=gradient_norm.new_zeros(()),
                last_line_search_trials=torch.zeros_like(
                    previous.last_line_search_trials
                ),
                last_predicted_reduction=gradient_norm.new_zeros(()),
                last_actual_reduction=gradient_norm.new_zeros(()),
                last_negative_curvature=torch.zeros_like(
                    previous.last_negative_curvature
                ),
                last_gradient_fallback=torch.zeros_like(
                    previous.last_gradient_fallback
                ),
                last_step_accepted=torch.ones_like(previous.last_step_accepted),
                fallback_count=previous.fallback_count,
            )
            return SCFMethodProposal(seed, next_state)

        shift = previous.level_shift
        if not bool(shift > 0.0):
            shift = self.settings.initial_level_shift * (
                preconditioner.abs().amax().clamp_min(1.0)
            )
        operator = _SOSCFHessianOperator(model, zero)
        (
            step,
            shift,
            microiterations,
            inner_residual,
            hvp_count,
            negative,
            fallback,
        ) = _pcg_newton_step(
            operator, gradient, preconditioner, shift, self.settings
        )
        if fallback:
            shift = self.settings.initial_level_shift * (
                preconditioner.abs().amax().clamp_min(1.0)
            )
        eps = torch.finfo(step.dtype).eps
        radius = previous.trust_radius.clamp(
            min=self.settings.minimum_trust_radius,
            max=self.settings.maximum_trust_radius,
        )
        step = step * torch.clamp(
            radius / torch.linalg.vector_norm(step).clamp_min(eps), max=1.0
        )
        descent = torch.dot(gradient, step)
        if not bool(torch.isfinite(step).all()) or not bool(descent < 0.0):
            raise RuntimeError(
                "SOSCF Newton step is not a finite descent step."
            )

        energy0 = model.energy_at(zero)
        if not bool(torch.isfinite(energy0)):
            raise RuntimeError(
                "SOSCF current fixed-occupation energy is non-finite."
            )
        # This is the linear directional decrease used by Armijo.
        predicted = -descent
        (
            accepted_rotation,
            accepted_fock,
            accepted_energy,
            alpha,
            line_search_trials,
        ) = _backtracking_step(
            model,
            step,
            energy0,
            gap_floor,
            context.damp,
            predicted,
            self.settings,
        )

        seed = _build_fock_seed(
            model, accepted_rotation, accepted_fock, gap_floor
        )
        if not bool(torch.isfinite(seed).all()):
            raise RuntimeError("SOSCF constructed a non-finite Fock seed.")
        actual = energy0 - accepted_energy
        # A successful macrostep permits the next Newton solve to retry at
        # the previous lower shift. Curvature failures will increase it again.
        next_shift = torch.maximum(
            shift / self.settings.level_shift_growth,
            self.settings.initial_level_shift
            * preconditioner.abs().amax().clamp_min(1.0),
        )
        next_state = SOSCFState(
            trust_radius=radius,
            level_shift=next_shift,
            last_gradient_norm=torch.linalg.vector_norm(gradient),
            last_step_norm=torch.linalg.vector_norm(accepted_rotation),
            last_step_max_abs=accepted_rotation.abs().amax(),
            last_microiterations=microiterations,
            last_inner_residual=inner_residual,
            last_hvp_count=hvp_count,
            last_line_search_alpha=alpha,
            last_line_search_trials=torch.tensor(
                line_search_trials, dtype=torch.int64, device=seed.device
            ),
            last_predicted_reduction=predicted,
            last_actual_reduction=actual,
            last_negative_curvature=negative,
            last_gradient_fallback=torch.tensor(fallback, device=seed.device),
            last_step_accepted=torch.tensor(True, device=seed.device),
            fallback_count=previous.fallback_count
            + torch.tensor(fallback, dtype=torch.int64, device=seed.device),
        )
        return SCFMethodProposal(seed, next_state)
