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
"""Pure physical SCF certification and progress diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from dxtb._src.typing import Tensor

from .convergence import SCFConvergence, SCFResidual
from .energy import electronic_energy_from_spectral, free_energy_from_spectral
from .problem import SCFPhysicalState, SCFProblem, SCFRepresentation
from .stationarity import (
    ao_stationarity_matrix,
    orthogonalized_stationarity_matrix,
)

__all__ = [
    "SCFCertificationMetrics",
    "SCFCertificationCoreMetrics",
    "SCFCertificationCoreResult",
    "SCFInternalCertificationProfile",
    "SCFCertificationResult",
    "SCFCertificationThresholds",
    "SCF_CERTIFICATION_STANDARD",
    "SCF_CERTIFICATION_TIGHT",
    "SCF_CERTIFICATION_HIGHER_ORDER_REFERENCE",
    "SCFProgressMetrics",
    "certification_metrics",
    "certification_core_metrics",
    "certify_scf_core_state",
    "certify_scf_state",
    "resolve_certification_thresholds",
    "terminal_failure_code",
    "SCFCoreFailure",
    "stationarity_matrix",
]


class SCFCoreFailure(IntEnum):
    """Internal physical-certification failure categories."""

    NONE = 0
    RAW_ROOT = 1
    STATIONARITY = 2
    ELECTRON_COUNT = 3
    FERMI = 4
    NONFINITE = 5
    INVALID_TEMPERATURE = 6
    TOO_MANY_ELECTRONS = 7


@dataclass(frozen=True, eq=False)
class SCFCertificationThresholds:
    """Explicit representation-specific thresholds for diagnostic verdicts."""

    representation: SCFRepresentation
    fixed_point_l2: float
    fixed_point_max: float
    stationarity_orthogonalized_per_sqrt_orbital: float
    electron_count: float
    requested_fixed_point_l2: float | None = None
    requested_fixed_point_max: float | None = None


@dataclass(frozen=True, eq=False)
class SCFCertificationCoreResult:
    """Core metrics, component verdicts, and an internal failure code."""

    metrics: SCFCertificationCoreMetrics
    fixed_point_ok: Tensor
    stationarity_ok: Tensor
    electron_count_ok: Tensor
    finite_ok: Tensor
    fermi_ok: Tensor
    certified: Tensor
    terminal_failed: Tensor
    failure_code: Tensor
    requested_fixed_point_l2_tolerance: Tensor
    requested_fixed_point_max_tolerance: Tensor
    effective_fixed_point_l2_tolerance: Tensor
    effective_fixed_point_max_tolerance: Tensor
    effective_stationarity_tolerance: Tensor
    effective_electron_count_tolerance: Tensor


@dataclass(frozen=True)
class SCFInternalCertificationProfile:
    """Explicit internal accuracy profile proposed for S2b integration."""

    name: str
    fixed_point_l2: float
    fixed_point_max: float
    stationarity_orthogonalized_per_sqrt_orbital: float
    inherit_solver_root_tolerances: bool = False
    stationarity_float32_floor: float = 1.0e-5

    def stationarity_tolerance(self, dtype: torch.dtype) -> float:
        """Resolve the explicit stationarity floor for the numerical dtype."""
        if dtype is torch.float32:
            return max(
                self.stationarity_orthogonalized_per_sqrt_orbital,
                self.stationarity_float32_floor,
            )
        return self.stationarity_orthogonalized_per_sqrt_orbital


SCF_CERTIFICATION_STANDARD = SCFInternalCertificationProfile(
    "standard",
    fixed_point_l2=1.0e-4,
    fixed_point_max=1.0e-5,
    stationarity_orthogonalized_per_sqrt_orbital=1.0e-5,
    inherit_solver_root_tolerances=True,
)
SCF_CERTIFICATION_TIGHT = SCFInternalCertificationProfile(
    "tight",
    fixed_point_l2=1.0e-10,
    fixed_point_max=1.0e-10,
    stationarity_orthogonalized_per_sqrt_orbital=1.0e-9,
)
SCF_CERTIFICATION_HIGHER_ORDER_REFERENCE = SCFInternalCertificationProfile(
    "higher_order_reference",
    fixed_point_l2=1.0e-12,
    fixed_point_max=1.0e-12,
    stationarity_orthogonalized_per_sqrt_orbital=1.0e-12,
)


@dataclass(frozen=True, eq=False)
class SCFCertificationCoreMetrics:
    """Certification metrics that do not require energy or entropy work."""

    representation: SCFRepresentation
    fixed_point: SCFResidual
    stationarity_ao_l2: Tensor
    stationarity_ao_max: Tensor
    stationarity_orthogonalized_l2: Tensor
    stationarity_orthogonalized_max: Tensor
    stationarity_orthogonalized_l2_per_sqrt_orbital: Tensor
    electron_count: Tensor
    electron_count_error: Tensor
    finite_ok: Tensor
    fermi_ok: Tensor
    fermi_converged: Tensor
    fermi_electron_count_ok: Tensor
    fermi_invalid_temperature: Tensor
    fermi_too_many_electrons: Tensor


@dataclass(frozen=True, eq=False)
class SCFCertificationMetrics:
    """Continuous physical metrics and independent validity flags."""

    representation: SCFRepresentation
    fixed_point: SCFResidual
    stationarity_ao_l2: Tensor
    stationarity_ao_max: Tensor
    stationarity_orthogonalized_l2: Tensor
    stationarity_orthogonalized_max: Tensor
    stationarity_orthogonalized_l2_per_sqrt_orbital: Tensor
    electron_count: Tensor
    electron_count_error: Tensor
    electronic_energy: Tensor
    electronic_free_energy: Tensor
    finite_ok: Tensor
    fermi_ok: Tensor
    fermi_converged: Tensor
    fermi_electron_count_ok: Tensor
    fermi_invalid_temperature: Tensor
    fermi_too_many_electrons: Tensor


@dataclass(frozen=True, eq=False)
class SCFCertificationResult:
    """Metrics paired with explicit-threshold component verdicts."""

    metrics: SCFCertificationMetrics
    fixed_point_ok: Tensor
    stationarity_ok: Tensor
    electron_count_ok: Tensor
    finite_ok: Tensor
    fermi_ok: Tensor
    certified: Tensor


@dataclass(frozen=True, eq=False)
class SCFProgressMetrics:
    """Energy/free-energy values and changes between paired states."""

    previous_energy: Tensor
    current_energy: Tensor
    previous_free_energy: Tensor
    current_free_energy: Tensor
    energy_change: Tensor
    free_energy_change: Tensor
    absolute_energy_change: Tensor
    absolute_free_energy_change: Tensor


def _finite_all(values: tuple[Tensor, ...], *, like: Tensor) -> Tensor:
    finite = torch.ones((), dtype=torch.bool, device=like.device)
    for value in values:
        finite = finite & torch.isfinite(value).all()
    return finite


def _spectral_tensors(state: SCFPhysicalState) -> tuple[Tensor, ...]:
    spectral = state.map_result.spectral
    charges = spectral.charges
    values = [
        state.x,
        state.map_result.mapped_x,
        state.delta,
        spectral.hamiltonian,
        spectral.density,
        spectral.emo,
        spectral.coefficients,
        spectral.occupation,
        spectral.orbital_energy,
        charges.mono,
    ]
    if charges.dipole is not None:
        values.append(charges.dipole)
    if charges.quad is not None:
        values.append(charges.quad)
    potential = state.map_result.driving_potential
    if potential is not None:
        if potential.mono is not None:
            values.append(potential.mono)
        if potential.dipole is not None:
            values.append(potential.dipole)
        if potential.quad is not None:
            values.append(potential.quad)
    if spectral.fermi_status is not None:
        values.append(spectral.fermi_status.residual)
    fermi_state = spectral._fermi_state
    if fermi_state is not None:
        values.extend(
            (
                fermi_state.result.occupation,
                fermi_state.result.status.residual,
                fermi_state.chemical_potential_ref,
                fermi_state.chemical_potential_offset,
                fermi_state.gap,
            )
        )
    return tuple(values)


def _fermi_metrics(
    state: SCFPhysicalState,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    status = state.map_result.fermi_status
    like = state.x
    if status is None:
        valid = torch.ones((), dtype=torch.bool, device=like.device)
        invalid = torch.zeros((), dtype=torch.bool, device=like.device)
        return valid, valid, invalid, invalid, valid
    converged = status.converged.all()
    electron_count_ok = status.electron_count_ok.all()
    invalid_temperature = status.invalid_temperature.any()
    too_many_electrons = status.too_many_electrons.any()
    valid = (
        converged
        & electron_count_ok
        & ~invalid_temperature
        & ~too_many_electrons
    )
    return (
        converged,
        electron_count_ok,
        invalid_temperature,
        too_many_electrons,
        valid,
    )


def stationarity_matrix(
    problem: SCFProblem,
    state: SCFPhysicalState,
) -> Tensor:
    """Return the masked nonorthogonal AO commutator using a fresh Fock."""
    fresh_hamiltonian = problem.rebuild_hamiltonian(
        state.map_result.spectral.charges
    )
    return _stationarity_matrix(problem, state, fresh_hamiltonian)


def _stationarity_matrix(
    problem: SCFProblem,
    state: SCFPhysicalState,
    fresh_hamiltonian: Tensor,
) -> Tensor:
    inputs = problem.inputs
    density = state.map_result.spectral.density
    return ao_stationarity_matrix(
        fresh_hamiltonian,
        density,
        inputs.integrals.overlap,
        inputs.orbital_mask,
    )


def _stationarity_norms(
    ao_residual: Tensor,
    inverse_sqrt_overlap: Tensor,
    orbital_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return AO/orthogonalized norms and the size-aware orthogonal norm."""
    orthogonalized = orthogonalized_stationarity_matrix(
        ao_residual, inverse_sqrt_overlap, orbital_mask
    )
    active_orbitals = orbital_mask.sum().to(ao_residual.dtype)
    active_pairs = (
        orbital_mask[..., :, None] & orbital_mask[..., None, :]
    ).sum()
    ao_l2 = torch.linalg.vector_norm(ao_residual)
    ao_max = ao_residual.abs().amax()
    orthogonalized_l2 = torch.linalg.vector_norm(orthogonalized)
    orthogonalized_max = orthogonalized.abs().amax()
    per_sqrt_orbital = orthogonalized_l2 / active_orbitals.sqrt()
    per_sqrt_pair = (
        orthogonalized_l2 / active_pairs.to(ao_residual.dtype).sqrt()
    )
    return (
        ao_l2,
        ao_max,
        orthogonalized_l2,
        orthogonalized_max,
        per_sqrt_orbital,
        per_sqrt_pair,
    )


def certification_core_metrics(
    problem: SCFProblem,
    state: SCFPhysicalState,
    *,
    fixed_point_residual: SCFResidual | None = None,
) -> SCFCertificationCoreMetrics:
    """Measure core physical validity without energy/entropy observables."""
    if state.map_result.mapped_x.shape != state.x.shape:
        raise ValueError("Paired SCF state input and map output shapes differ.")

    inputs = problem.inputs
    spectral = state.map_result.spectral
    overlap = inputs.integrals.overlap
    if fixed_point_residual is None:
        projected_delta = problem.project_residual(state.delta)
        fixed_point = SCFConvergence.residual(projected_delta)
    else:
        fixed_point = fixed_point_residual

    fresh_hamiltonian = problem.rebuild_hamiltonian(spectral.charges)
    stationarity = _stationarity_matrix(problem, state, fresh_hamiltonian)
    pair_mask = (
        inputs.orbital_mask[..., :, None] & inputs.orbital_mask[..., None, :]
    )
    density = spectral.density
    physical_density = torch.where(pair_mask, density, 0.0)
    electron_count = torch.diagonal(
        physical_density @ overlap, dim1=-2, dim2=-1
    ).sum()
    target_count = inputs.nel.sum()
    electron_count_error = electron_count - target_count

    (
        stationarity_l2,
        stationarity_max,
        stationarity_orthogonalized_l2,
        stationarity_orthogonalized_max,
        stationarity_scaled,
        stationarity_per_sqrt_pair,
    ) = _stationarity_norms(
        stationarity, inputs.inverse_cholesky, inputs.orbital_mask
    )

    (
        fermi_converged,
        fermi_electron_ok,
        invalid_temperature,
        too_many,
        fermi_ok,
    ) = _fermi_metrics(state)
    finite_values = _spectral_tensors(state) + (
        fresh_hamiltonian,
        fixed_point.l2,
        fixed_point.max_abs,
        stationarity,
        stationarity_l2,
        stationarity_max,
        stationarity_orthogonalized_l2,
        stationarity_orthogonalized_max,
        stationarity_scaled,
        stationarity_per_sqrt_pair,
        electron_count,
        electron_count_error,
    )
    finite_ok = _finite_all(finite_values, like=state.x)

    return SCFCertificationCoreMetrics(
        representation=problem.representation,
        fixed_point=fixed_point,
        stationarity_ao_l2=stationarity_l2,
        stationarity_ao_max=stationarity_max,
        stationarity_orthogonalized_l2=stationarity_orthogonalized_l2,
        stationarity_orthogonalized_max=stationarity_orthogonalized_max,
        stationarity_orthogonalized_l2_per_sqrt_orbital=stationarity_scaled,
        electron_count=electron_count,
        electron_count_error=electron_count_error,
        finite_ok=finite_ok,
        fermi_ok=fermi_ok,
        fermi_converged=fermi_converged,
        fermi_electron_count_ok=fermi_electron_ok,
        fermi_invalid_temperature=invalid_temperature,
        fermi_too_many_electrons=too_many,
    )


def resolve_certification_thresholds(
    problem: SCFProblem,
    *,
    x_atol: float,
    x_atol_max: float,
    dtype: torch.dtype,
    profile: SCFInternalCertificationProfile = SCF_CERTIFICATION_STANDARD,
    requested_x_atol: float | None = None,
    requested_x_atol_max: float | None = None,
) -> SCFCertificationThresholds:
    """Resolve a profile while preserving effective caller root tolerances.

    Standard certification inherits the already-resolved controller settings.
    Explicit tight/reference profiles use their declared root thresholds and
    the existing dtype-resolution floor.
    """
    if profile.inherit_solver_root_tolerances:
        fixed_point_l2 = x_atol
        fixed_point_max = x_atol_max
    else:
        resolution_floor = 50.0 * torch.finfo(dtype).resolution
        fixed_point_l2 = max(profile.fixed_point_l2, resolution_floor)
        fixed_point_max = max(profile.fixed_point_max, resolution_floor)
    return SCFCertificationThresholds(
        representation=problem.representation,
        fixed_point_l2=fixed_point_l2,
        fixed_point_max=fixed_point_max,
        stationarity_orthogonalized_per_sqrt_orbital=(
            profile.stationarity_tolerance(dtype)
        ),
        electron_count=fixed_point_l2,
        requested_fixed_point_l2=(
            (x_atol if requested_x_atol is None else requested_x_atol)
            if profile.inherit_solver_root_tolerances
            else profile.fixed_point_l2
        ),
        requested_fixed_point_max=(
            (
                x_atol_max
                if requested_x_atol_max is None
                else requested_x_atol_max
            )
            if profile.inherit_solver_root_tolerances
            else profile.fixed_point_max
        ),
    )


def certify_scf_core_state(
    problem: SCFProblem,
    state: SCFPhysicalState,
    thresholds: SCFCertificationThresholds,
    *,
    fixed_point_residual: SCFResidual | None = None,
) -> SCFCertificationCoreResult:
    """Apply explicit thresholds without computing energy or entropy."""
    if thresholds.representation is not problem.representation:
        raise ValueError(
            "Certification thresholds do not match the SCF representation."
        )
    metrics = certification_core_metrics(
        problem, state, fixed_point_residual=fixed_point_residual
    )
    fixed_point_ok = (metrics.fixed_point.l2 < thresholds.fixed_point_l2) & (
        metrics.fixed_point.max_abs < thresholds.fixed_point_max
    )
    stationarity_ok = (
        metrics.stationarity_orthogonalized_l2_per_sqrt_orbital
        < thresholds.stationarity_orthogonalized_per_sqrt_orbital
    )
    electron_count_ok = (
        metrics.electron_count_error.abs() < thresholds.electron_count
    )
    certified = (
        fixed_point_ok
        & stationarity_ok
        & electron_count_ok
        & metrics.finite_ok
        & metrics.fermi_ok
    )
    code = torch.zeros((), dtype=torch.int64, device=state.x.device)
    code = torch.where(
        ~fixed_point_ok,
        torch.full_like(code, int(SCFCoreFailure.RAW_ROOT)),
        code,
    )
    code = torch.where(
        ~stationarity_ok,
        torch.full_like(code, int(SCFCoreFailure.STATIONARITY)),
        code,
    )
    code = torch.where(
        ~electron_count_ok,
        torch.full_like(code, int(SCFCoreFailure.ELECTRON_COUNT)),
        code,
    )
    code = torch.where(
        ~metrics.fermi_ok,
        torch.full_like(code, int(SCFCoreFailure.FERMI)),
        code,
    )
    code = torch.where(
        metrics.fermi_too_many_electrons,
        torch.full_like(code, int(SCFCoreFailure.TOO_MANY_ELECTRONS)),
        code,
    )
    code = torch.where(
        metrics.fermi_invalid_temperature,
        torch.full_like(code, int(SCFCoreFailure.INVALID_TEMPERATURE)),
        code,
    )
    code = torch.where(
        ~metrics.finite_ok,
        torch.full_like(code, int(SCFCoreFailure.NONFINITE)),
        code,
    )
    terminal_failed = (
        ~metrics.finite_ok
        | metrics.fermi_invalid_temperature
        | metrics.fermi_too_many_electrons
    )
    return SCFCertificationCoreResult(
        metrics=metrics,
        fixed_point_ok=fixed_point_ok,
        stationarity_ok=stationarity_ok,
        electron_count_ok=electron_count_ok,
        finite_ok=metrics.finite_ok,
        fermi_ok=metrics.fermi_ok,
        certified=certified,
        terminal_failed=terminal_failed,
        failure_code=code,
        requested_fixed_point_l2_tolerance=state.x.new_tensor(
            thresholds.requested_fixed_point_l2
            if thresholds.requested_fixed_point_l2 is not None
            else thresholds.fixed_point_l2
        ),
        requested_fixed_point_max_tolerance=state.x.new_tensor(
            thresholds.requested_fixed_point_max
            if thresholds.requested_fixed_point_max is not None
            else thresholds.fixed_point_max
        ),
        effective_fixed_point_l2_tolerance=state.x.new_tensor(
            thresholds.fixed_point_l2
        ),
        effective_fixed_point_max_tolerance=state.x.new_tensor(
            thresholds.fixed_point_max
        ),
        effective_stationarity_tolerance=state.x.new_tensor(
            thresholds.stationarity_orthogonalized_per_sqrt_orbital
        ),
        effective_electron_count_tolerance=state.x.new_tensor(
            thresholds.electron_count
        ),
    )


def terminal_failure_code(state: SCFPhysicalState) -> Tensor:
    """Return terminal finite/Fermi failure without rebuilding a Fock."""
    code = torch.zeros((), dtype=torch.int64, device=state.x.device)
    finite = _finite_all(_spectral_tensors(state), like=state.x)
    code = torch.where(
        ~finite,
        torch.full_like(code, int(SCFCoreFailure.NONFINITE)),
        code,
    )
    status = state.map_result.fermi_status
    if status is not None:
        code = torch.where(
            status.too_many_electrons.any(),
            torch.full_like(code, int(SCFCoreFailure.TOO_MANY_ELECTRONS)),
            code,
        )
        code = torch.where(
            status.invalid_temperature.any(),
            torch.full_like(code, int(SCFCoreFailure.INVALID_TEMPERATURE)),
            code,
        )
    return code


def certification_metrics(
    problem: SCFProblem,
    state: SCFPhysicalState,
    *,
    core_metrics: SCFCertificationCoreMetrics | None = None,
) -> SCFCertificationMetrics:
    """Return core metrics and optional physical energy observables."""
    core = (
        certification_core_metrics(problem, state)
        if core_metrics is None
        else core_metrics
    )
    spectral = state.map_result.spectral
    electronic_energy = electronic_energy_from_spectral(
        spectral, problem.inputs
    ).sum()
    electronic_free_energy = (
        electronic_energy
        + free_energy_from_spectral(spectral, problem.inputs).sum()
    )
    return SCFCertificationMetrics(
        representation=core.representation,
        fixed_point=core.fixed_point,
        stationarity_ao_l2=core.stationarity_ao_l2,
        stationarity_ao_max=core.stationarity_ao_max,
        stationarity_orthogonalized_l2=core.stationarity_orthogonalized_l2,
        stationarity_orthogonalized_max=core.stationarity_orthogonalized_max,
        stationarity_orthogonalized_l2_per_sqrt_orbital=(
            core.stationarity_orthogonalized_l2_per_sqrt_orbital
        ),
        electron_count=core.electron_count,
        electron_count_error=core.electron_count_error,
        electronic_energy=electronic_energy,
        electronic_free_energy=electronic_free_energy,
        finite_ok=(
            core.finite_ok
            & torch.isfinite(electronic_energy)
            & torch.isfinite(electronic_free_energy)
        ),
        fermi_ok=core.fermi_ok,
        fermi_converged=core.fermi_converged,
        fermi_electron_count_ok=core.fermi_electron_count_ok,
        fermi_invalid_temperature=core.fermi_invalid_temperature,
        fermi_too_many_electrons=core.fermi_too_many_electrons,
    )


def certify_scf_state(
    problem: SCFProblem,
    state: SCFPhysicalState,
    thresholds: SCFCertificationThresholds,
) -> SCFCertificationResult:
    """Apply caller-supplied thresholds to one method-independent state."""
    core = certify_scf_core_state(problem, state, thresholds)
    metrics = certification_metrics(problem, state, core_metrics=core.metrics)
    certified = (
        core.fixed_point_ok
        & core.stationarity_ok
        & core.electron_count_ok
        & metrics.finite_ok
        & core.fermi_ok
    )
    return SCFCertificationResult(
        metrics=metrics,
        fixed_point_ok=core.fixed_point_ok,
        stationarity_ok=core.stationarity_ok,
        electron_count_ok=core.electron_count_ok,
        finite_ok=metrics.finite_ok,
        fermi_ok=core.fermi_ok,
        certified=certified,
    )


def _state_energy_values(
    problem: SCFProblem, state: SCFPhysicalState
) -> tuple[Tensor, Tensor]:
    spectral = state.map_result.spectral
    energy = electronic_energy_from_spectral(spectral, problem.inputs).sum()
    free_energy = (
        energy + free_energy_from_spectral(spectral, problem.inputs).sum()
    )
    return energy, free_energy


def scf_progress_metrics(
    problem: SCFProblem,
    previous: SCFPhysicalState,
    current: SCFPhysicalState,
) -> SCFProgressMetrics:
    """Compare existing paired states without evaluating another SCF map."""
    previous_energy, previous_free_energy = _state_energy_values(
        problem, previous
    )
    current_energy, current_free_energy = _state_energy_values(problem, current)
    energy_change = current_energy - previous_energy
    free_energy_change = current_free_energy - previous_free_energy
    return SCFProgressMetrics(
        previous_energy=previous_energy,
        current_energy=current_energy,
        previous_free_energy=previous_free_energy,
        current_free_energy=current_free_energy,
        energy_change=energy_change,
        free_energy_change=free_energy_change,
        absolute_energy_change=energy_change.abs(),
        absolute_free_energy_change=free_energy_change.abs(),
    )
