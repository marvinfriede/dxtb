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

import torch

from dxtb._src.typing import Tensor

from .convergence import SCFConvergence, SCFResidual
from .energy import electronic_energy_from_spectral, free_energy_from_spectral
from .problem import SCFPhysicalState, SCFProblem, SCFRepresentation

__all__ = [
    "SCFCertificationMetrics",
    "SCFCertificationCoreMetrics",
    "SCFInternalCertificationProfile",
    "SCFCertificationResult",
    "SCFCertificationThresholds",
    "SCF_CERTIFICATION_STANDARD",
    "SCF_CERTIFICATION_TIGHT",
    "SCF_CERTIFICATION_HIGHER_ORDER_REFERENCE",
    "SCFProgressMetrics",
    "certification_metrics",
    "certification_core_metrics",
    "certify_scf_state",
    "stationarity_matrix",
]


@dataclass(frozen=True, eq=False)
class SCFCertificationThresholds:
    """Explicit representation-specific thresholds for diagnostic verdicts."""

    representation: SCFRepresentation
    fixed_point_l2: float
    fixed_point_max: float
    stationarity_orthogonalized_per_sqrt_orbital: float
    electron_count: float


@dataclass(frozen=True)
class SCFInternalCertificationProfile:
    """Explicit internal accuracy profile proposed for S2b integration."""

    name: str
    fixed_point_l2: float
    fixed_point_max: float
    stationarity_orthogonalized_per_sqrt_orbital: float
    electron_count_uses_effective_root_tolerance: bool = True
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
    overlap = inputs.integrals.overlap
    left = fresh_hamiltonian @ density @ overlap
    right = overlap @ density @ fresh_hamiltonian
    residual = left - right
    pair_mask = (
        inputs.orbital_mask[..., :, None] & inputs.orbital_mask[..., None, :]
    )
    return torch.where(pair_mask, residual, 0.0)


def _stationarity_norms(
    ao_residual: Tensor,
    inverse_sqrt_overlap: Tensor,
    orbital_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return AO/orthogonalized norms and the size-aware orthogonal norm."""
    orthogonalized = (
        inverse_sqrt_overlap.transpose(-2, -1)
        @ ao_residual
        @ inverse_sqrt_overlap
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
) -> SCFCertificationCoreMetrics:
    """Measure core physical validity without energy/entropy observables."""
    if state.map_result.mapped_x.shape != state.x.shape:
        raise ValueError("Paired SCF state input and map output shapes differ.")

    inputs = problem.inputs
    spectral = state.map_result.spectral
    overlap = inputs.integrals.overlap
    projected_delta = problem.project_residual(state.delta)
    fixed_point = SCFConvergence.residual(projected_delta)

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


def certification_metrics(
    problem: SCFProblem,
    state: SCFPhysicalState,
) -> SCFCertificationMetrics:
    """Return core metrics and optional physical energy observables."""
    core = certification_core_metrics(problem, state)
    spectral = state.map_result.spectral
    electronic_energy = electronic_energy_from_spectral(
        spectral, problem.inputs
    ).sum()
    electronic_free_energy = (
        electronic_energy
        + free_energy_from_spectral(spectral, problem.inputs).sum()
    )
    finite_ok = core.finite_ok & torch.isfinite(electronic_energy)
    finite_ok = finite_ok & torch.isfinite(electronic_free_energy)
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
        finite_ok=finite_ok,
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
    if thresholds.representation is not problem.representation:
        raise ValueError(
            "Certification thresholds do not match the SCF representation."
        )
    metrics = certification_metrics(problem, state)
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
    return SCFCertificationResult(
        metrics=metrics,
        fixed_point_ok=fixed_point_ok,
        stationarity_ok=stationarity_ok,
        electron_count_ok=electron_count_ok,
        finite_ok=metrics.finite_ok,
        fermi_ok=metrics.fermi_ok,
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
