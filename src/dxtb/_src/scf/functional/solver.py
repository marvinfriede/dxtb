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
"""Functional simple-mixing SCF solver, kept parallel to production SCF."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import ClassVar, cast

import torch
from tad_mctc.math import einsum
from tad_mctc.tree import Node, child

from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.constants import labels
from dxtb._src.scf.status import SCFFailureCode
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction import filling
from dxtb._src.wavefunction.filling import (
    FermiOccupationResult,
    FermiStatus,
    _FermiCoreState,
)

from .certification import (
    SCF_CERTIFICATION_STANDARD,
    SCFCertificationCoreResult,
    SCFCertificationThresholds,
    SCFCoreFailure,
    SCFInternalCertificationProfile,
    certify_scf_core_state,
    resolve_certification_thresholds,
    terminal_failure_code,
)
from .convergence import SCFConvergence, SCFResidual
from .data import (
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFMapResult,
    SCFPotential,
    SCFSpectralResult,
    pack_potential,
    unpack_potential,
)
from .energy import free_energy_from_spectral, interaction_energy_from_charges
from .map import (
    evaluate_hamiltonian,
    evaluate_potential,
    potential_from_charges,
)
from .method import (
    LinearMethod,
    SCFMethod,
    SCFMethodContext,
    SCFMethodProposal,
    simple_mix,
)
from .problem import SCFPhysicalState, SCFProblem, SCFRepresentation
from .strategy import SCFStage, SCFStrategy

__all__ = [
    "DynamicDampingSettings",
    "DampingState",
    "SCFSolverDiagnostics",
    "SCFPhysicalCertificationDiagnostics",
    "SCFResidual",
    "SCFSolveResult",
    "SCFTrajectoryStatus",
    "SCFSolverSettings",
    "finalize_scf",
    "finalize_scf_fock",
    "functional_subsolves_ok",
    "functional_converged_and_valid",
    "functional_physical_converged",
    "functional_trajectory_ok",
    "functional_finalization_ok",
    "functional_return_valid",
    "trajectory_status_from_map",
    "accumulate_trajectory_status",
    "fermi_status_ok",
    "interaction_energy_from_charges",
    "setup_simple_solver_settings",
    "simple_mix",
    "solve_scf_simple",
    "solve_scf_potential_simple",
    "solve_scf_fock_simple",
    "finalize_scf_potential",
]


@dataclass(frozen=True, eq=False)
class DynamicDampingSettings:
    """Static configuration for legacy-compatible dynamic damping."""

    threshold: ClassVar[float] = 0.1
    enabled: bool = False
    factor: float = 0.99


@dataclass(frozen=True, eq=False)
class DampingState:
    """Tensor-valued effective ordinary damping during a solve."""

    value: Tensor
    switched: Tensor


@dataclass(frozen=True, eq=False)
class SCFSolverDiagnostics:
    """Tensor-valued numerical diagnostics for eager compatibility."""

    final_damp: Tensor
    dynamic_damping_switched: Tensor
    terminal_failed: Tensor
    certification: SCFPhysicalCertificationDiagnostics | None = None


@dataclass(frozen=True, eq=False)
class SCFPhysicalCertificationDiagnostics:
    """Core physical verdict and effective thresholds for the selected root."""

    result: SCFCertificationCoreResult


class SCFSolverTrace(Node):
    """Optional tensor trace of the shared fixed-trip solver transitions."""

    initial_guess: Tensor = child()
    map_inputs: Tensor = child()
    raw_mapped: Tensor = child()
    iterates: Tensor = child()
    hamiltonians: Tensor = child()
    densities: Tensor = child()
    charges: Tensor = child()
    driving_potentials: Tensor | None = child()
    residuals: Tensor = child()
    active_before: Tensor = child(keep_dtype=True)
    active_after: Tensor = child(keep_dtype=True)
    iterations: Tensor = child()
    damping: Tensor = child()
    fermi_converged: Tensor | None = child(keep_dtype=True)
    fermi_electron_count_ok: Tensor | None = child(keep_dtype=True)
    fermi_invalid_temperature: Tensor | None = child(keep_dtype=True)
    fermi_too_many_electrons: Tensor | None = child(keep_dtype=True)
    fermi_residual: Tensor | None = child()
    mixer_state: tuple[Tensor, ...] = child()


@dataclass(frozen=True, eq=False)
class SCFSolverSettings:
    """Static root-finding settings for the E4.1 simple solver."""

    maxiter: int
    mix_guess: bool
    damp: float
    x_atol: float
    x_atol_max: float
    dynamic_damping: DynamicDampingSettings = DynamicDampingSettings()
    requested_x_atol: float | None = None
    requested_x_atol_max: float | None = None


@dataclass(frozen=True, eq=False)
class SCFSolveResult:
    """Finalized result and tensor-valued solver status.

    ``energy`` contains the atom-resolved SCF electronic/interactions energy
    excluding entropy. ``fenergy`` is the separately partitioned electronic
    free-energy contribution, so the total electronic term is their sum.
    """

    map_result: SCFMapResult
    final_evaluation: SCFSpectralResult
    final_charges: SCFCharges
    final_potential: SCFPotential
    energy: Tensor
    fenergy: Tensor
    root_x: Tensor
    iterations: Tensor
    converged: Tensor
    residual: Tensor
    trajectory_status: SCFTrajectoryStatus | None
    diagnostics: SCFSolverDiagnostics | None = None
    trace: SCFSolverTrace | None = None

    @property
    def hamiltonian(self) -> Tensor:
        """Public final Hamiltonian phase."""
        return self.final_evaluation.hamiltonian

    @property
    def density(self) -> Tensor:
        """Public final density phase."""
        return self.final_evaluation.density

    @property
    def emo(self) -> Tensor:
        """Public final orbital energies."""
        return self.final_evaluation.emo

    @property
    def coefficients(self) -> Tensor:
        """Public final orbital coefficients."""
        return self.final_evaluation.coefficients

    @property
    def occupation(self) -> Tensor:
        """Public final occupation values."""
        return self.final_evaluation.occupation


@dataclass(frozen=True, eq=False)
class SCFTrajectoryStatus:
    """Validity accumulated across physically executed raw-map spectra."""

    fermi_converged: Tensor
    electron_count_ok: Tensor
    invalid_temperature: Tensor
    too_many_electrons: Tensor
    first_failure: Tensor


FERMI_FAILURE_NONE = int(SCFFailureCode.NONE)
FERMI_FAILURE_INVALID_TEMPERATURE = int(SCFFailureCode.INVALID_TEMPERATURE)
FERMI_FAILURE_TOO_MANY_ELECTRONS = int(SCFFailureCode.TOO_MANY_ELECTRONS)
FERMI_FAILURE_NONCONVERGENCE = int(SCFFailureCode.FERMI_NONCONVERGENCE)
FERMI_FAILURE_ELECTRON_COUNT = int(SCFFailureCode.ELECTRON_COUNT)


def fermi_failure_code(status: FermiStatus | None, *, like: Tensor) -> Tensor:
    """Return the first eager failure category for one spectral evaluation."""
    if status is None:
        return torch.zeros((), dtype=torch.int64, device=like.device)
    code = torch.zeros((), dtype=torch.int64, device=status.converged.device)
    code = torch.where(
        ~status.electron_count_ok.all(),
        torch.full_like(code, FERMI_FAILURE_ELECTRON_COUNT),
        code,
    )
    code = torch.where(
        ~status.converged.all(),
        torch.full_like(code, FERMI_FAILURE_NONCONVERGENCE),
        code,
    )
    code = torch.where(
        status.too_many_electrons.any(),
        torch.full_like(code, FERMI_FAILURE_TOO_MANY_ELECTRONS),
        code,
    )
    return torch.where(
        status.invalid_temperature.any(),
        torch.full_like(code, FERMI_FAILURE_INVALID_TEMPERATURE),
        code,
    )


def trajectory_status_from_map(
    status: FermiStatus | None,
) -> SCFTrajectoryStatus | None:
    """Initialize trajectory validity from its first physical map status."""
    if status is None:
        return None
    return SCFTrajectoryStatus(
        fermi_converged=status.converged,
        electron_count_ok=status.electron_count_ok,
        invalid_temperature=status.invalid_temperature,
        too_many_electrons=status.too_many_electrons,
        first_failure=fermi_failure_code(status, like=status.converged),
    )


def accumulate_trajectory_status(
    old: SCFTrajectoryStatus | None,
    new_status: FermiStatus | None,
    physical_active: Tensor,
) -> SCFTrajectoryStatus | None:
    """Accumulate status only for lanes executing a physical raw map."""
    if new_status is None:
        return old
    if old is None:
        initial = trajectory_status_from_map(new_status)
        assert initial is not None
        return SCFTrajectoryStatus(
            fermi_converged=torch.where(
                physical_active,
                initial.fermi_converged,
                torch.ones_like(initial.fermi_converged),
            ),
            electron_count_ok=torch.where(
                physical_active,
                initial.electron_count_ok,
                torch.ones_like(initial.electron_count_ok),
            ),
            invalid_temperature=torch.where(
                physical_active,
                initial.invalid_temperature,
                torch.zeros_like(initial.invalid_temperature),
            ),
            too_many_electrons=torch.where(
                physical_active,
                initial.too_many_electrons,
                torch.zeros_like(initial.too_many_electrons),
            ),
            first_failure=torch.where(
                physical_active,
                initial.first_failure,
                torch.zeros_like(initial.first_failure),
            ),
        )
    new_failure = fermi_failure_code(new_status, like=new_status.converged)
    return SCFTrajectoryStatus(
        fermi_converged=torch.where(
            physical_active,
            old.fermi_converged & new_status.converged,
            old.fermi_converged,
        ),
        electron_count_ok=torch.where(
            physical_active,
            old.electron_count_ok & new_status.electron_count_ok,
            old.electron_count_ok,
        ),
        invalid_temperature=torch.where(
            physical_active,
            old.invalid_temperature | new_status.invalid_temperature,
            old.invalid_temperature,
        ),
        too_many_electrons=torch.where(
            physical_active,
            old.too_many_electrons | new_status.too_many_electrons,
            old.too_many_electrons,
        ),
        first_failure=torch.where(
            physical_active & (old.first_failure == FERMI_FAILURE_NONE),
            torch.where(
                new_failure != FERMI_FAILURE_NONE,
                new_failure,
                old.first_failure,
            ),
            old.first_failure,
        ),
    )


def _trajectory_status_ok(status: SCFTrajectoryStatus) -> Tensor:
    return (
        status.fermi_converged.all()
        & status.electron_count_ok.all()
        & ~status.invalid_temperature.any()
        & ~status.too_many_electrons.any()
    )


def fermi_status_ok(status: FermiStatus | None, *, like: Tensor) -> Tensor:
    """Return one-spectrum validity, with structural absence treated as valid."""
    if status is None:
        return torch.ones((), dtype=torch.bool, device=like.device)
    return (
        status.converged.all()
        & status.electron_count_ok.all()
        & ~status.invalid_temperature.any()
        & ~status.too_many_electrons.any()
    )


def functional_subsolves_ok(result: SCFSolveResult) -> Tensor:
    """Return historical trajectory and final-spectrum validity only.

    This operational diagnostic is deliberately separate from physical root
    certification. It retains the compatibility rule that a missing
    zero-budget trajectory is structurally valid.
    """
    trajectory_ok = (
        torch.ones((), dtype=torch.bool, device=result.converged.device)
        if result.trajectory_status is None
        else _trajectory_status_ok(result.trajectory_status)
    )
    final_ok = fermi_status_ok(
        result.final_evaluation.fermi_status, like=result.converged
    )
    return trajectory_ok & final_ok


def functional_physical_converged(result: SCFSolveResult) -> Tensor:
    """Return the authoritative physical-root certification verdict."""
    return result.converged


def functional_trajectory_ok(result: SCFSolveResult) -> Tensor:
    """Return historical raw-map Fermi validity without finalization status."""
    if result.trajectory_status is None:
        return torch.ones((), dtype=torch.bool, device=result.converged.device)
    return _trajectory_status_ok(result.trajectory_status)


def functional_finalization_ok(result: SCFSolveResult) -> Tensor:
    """Return validity of the selected public final-spectrum evaluation."""
    return fermi_status_ok(
        result.final_evaluation.fermi_status, like=result.converged
    )


def functional_return_valid(result: SCFSolveResult) -> Tensor:
    """Return physical convergence and compatibility validity for output."""
    return functional_physical_converged(result) & functional_subsolves_ok(
        result
    )


def functional_converged_and_valid(result: SCFSolveResult) -> Tensor:
    """Compatibility alias for :func:`functional_return_valid`."""
    return functional_return_valid(result)


@dataclass(frozen=True, eq=False)
class _SolverOutcome:
    """Transform-friendly outcome before interaction finalization.

    ``root_x`` is the input paired with ``map_result`` when the convergence
    residual was tested; it is distinct from ``map_result.mapped_x``.
    """

    root_x: Tensor
    map_result: SCFMapResult
    iterations: Tensor
    converged: Tensor
    residual: Tensor
    trajectory_status: SCFTrajectoryStatus | None
    damping_state: DampingState
    terminal_failed: Tensor
    certification: SCFCertificationCoreResult | None = None
    trace: SCFSolverTrace | None = None


@dataclass(frozen=True, eq=False)
class _SolverControlState:
    """Shared immutable state advanced by fixed-trip and eager controllers.

    ``map_input_x`` pairs with the selected ``map_result`` and advances only
    when an active physical map evaluation is selected.
    """

    current_x: Tensor
    map_input_x: Tensor
    method_state: object | None
    damping_state: DampingState
    map_result: SCFMapResult
    iterations: Tensor
    residual: Tensor
    active: Tensor
    trajectory_status: SCFTrajectoryStatus | None
    certified: Tensor
    terminal_failed: Tensor
    certification: SCFCertificationCoreResult | None


def _resolve_tolerance(value: float, dtype: torch.dtype) -> float:
    """Match the legacy Simple mixer's first-call resolution downgrade."""
    resolution = torch.finfo(dtype).resolution
    return resolution * 50 if resolution > value else value


def setup_simple_solver_settings(
    config: ConfigSCF, dtype: torch.dtype
) -> SCFSolverSettings:
    """Resolve supported simple-solver settings without mutating config."""
    if config.maxiter < 0:
        raise ValueError("The functional SCF solver requires maxiter >= 0.")
    if config.mixer != labels.MIXER_LINEAR:
        raise NotImplementedError(
            "The E4.1 functional solver supports only simple linear mixing."
        )
    return SCFSolverSettings(
        maxiter=config.maxiter,
        mix_guess=config.mix_guess,
        damp=config.damp,
        x_atol=_resolve_tolerance(config.x_atol, dtype),
        x_atol_max=_resolve_tolerance(config.x_atol_max, dtype),
        dynamic_damping=DynamicDampingSettings(
            enabled=config.damp_dynamic,
            factor=config.damp_dynamic_factor,
        ),
        requested_x_atol=config.x_atol,
        requested_x_atol_max=config.x_atol_max,
    )


def _update_damping(
    current: DampingState,
    residual_l2: Tensor,
    transition_active: Tensor,
    settings: DynamicDampingSettings,
) -> DampingState:
    """Select damping for the next transition using the legacy threshold."""
    if not settings.enabled:
        return current
    switch = transition_active & (residual_l2 < settings.threshold)
    factor = current.value.new_tensor(settings.factor)
    return DampingState(
        torch.where(switch, factor, current.value),
        current.switched | switch,
    )


def _residual(
    delta: Tensor,
    projection: Callable[[Tensor], Tensor] | None = None,
) -> SCFResidual:
    return SCFConvergence.residual(delta, projection)


def _linear_transition_adapter(
    problem: SCFProblem,
    method: SCFMethod,
    old: Tensor,
    map_result: SCFMapResult,
    proposal_x: Tensor,
    method_state: object | None,
    step: int,
    transition_active: Tensor,
    damp: Tensor,
) -> SCFMethodProposal:
    """Adapt the physical-state method contract to shared controller steps."""
    physical_state = SCFPhysicalState(x=old, map_result=map_result)
    context = SCFMethodContext(
        step=step,
        transition_active=transition_active,
        damp=damp,
        proposal_x=proposal_x,
    )
    return method.propose(problem, physical_state, method_state, context)


def _transition_proposal(
    old: Tensor,
    map_result: SCFMapResult,
    proposal_x: Tensor,
    method_state: object | None,
    step: int,
    transition_active: Tensor,
    damp: Tensor,
    *,
    mix_fn: (
        Callable[
            [Tensor, Tensor, object | None, int, Tensor, Tensor],
            tuple[Tensor, object | None],
        ]
        | None
    ),
    problem: SCFProblem | None,
    method: SCFMethod | None,
) -> tuple[Tensor, object | None]:
    if method is not None:
        if problem is None:
            raise ValueError("A physical SCFProblem is required by SCFMethod.")
        proposal = _linear_transition_adapter(
            problem,
            method,
            old,
            map_result,
            proposal_x,
            method_state,
            step,
            transition_active,
            damp,
        )
        return proposal.x, proposal.state
    if mix_fn is None:
        raise ValueError("The controller requires a method or transition hook.")
    return mix_fn(
        old,
        proposal_x,
        method_state,
        step,
        transition_active,
        damp,
    )


def _simple_mixer_controller_adapter(
    old: Tensor,
    mapped: Tensor,
    state: object | None,
    step: int,
    transition_active: Tensor,
    damp: Tensor,
) -> tuple[Tensor, object | None]:
    """Transitional callback for synthetic/private map-control callers."""
    del step, transition_active
    if state is not None:
        raise TypeError("The simple mixer has no method history.")
    return simple_mix(old, mapped, damp), None


def _select_tensor(mask: Tensor, new: Tensor, old: Tensor) -> Tensor:
    return torch.where(mask, new, old)


def _select_core_certification(
    mask: Tensor,
    new: SCFCertificationCoreResult,
    old: SCFCertificationCoreResult,
) -> SCFCertificationCoreResult:
    """Select tensor leaves of immutable certification diagnostics."""

    def select_value(new_value, old_value):
        if isinstance(new_value, Tensor):
            return torch.where(mask, new_value, old_value)
        if is_dataclass(new_value):
            return replace(
                new_value,
                **{
                    field.name: select_value(
                        getattr(new_value, field.name),
                        getattr(old_value, field.name),
                    )
                    for field in fields(new_value)
                },
            )
        return new_value

    return cast(SCFCertificationCoreResult, select_value(new, old))


def _project_potential_padding(
    value: Tensor, layout: PotentialLayout
) -> Tensor:
    """Restore structural PADNZ tails after affine mixer arithmetic.

    Anderson's affine combination can introduce roundoff into a row whose
    entries are all the same PADNZ sentinel. Packing the structural components
    again preserves those tails exactly without changing physical values.
    """
    return pack_potential(unpack_potential(value, layout), layout)


def _select_map_result(
    mask: Tensor, new: SCFMapResult, old: SCFMapResult
) -> SCFMapResult:
    """Select map leaves without selecting Python structure dynamically."""

    def select_optional(
        new_value: Tensor | None, old_value: Tensor | None
    ) -> Tensor | None:
        if new_value is None and old_value is None:
            return None
        if new_value is None or old_value is None:
            raise ValueError("SCF map result structures must match.")
        return _select_tensor(mask, new_value, old_value)

    def select_fermi_state(
        new_state: _FermiCoreState | None,
        old_state: _FermiCoreState | None,
    ) -> _FermiCoreState | None:
        if new_state is None and old_state is None:
            return None
        if new_state is None or old_state is None:
            raise ValueError("SCF map result structures must match.")
        new_result, old_result = new_state.result, old_state.result
        return _FermiCoreState(
            result=FermiOccupationResult(
                occupation=_select_tensor(
                    mask, new_result.occupation, old_result.occupation
                ),
                status=FermiStatus(
                    converged=_select_tensor(
                        mask,
                        new_result.status.converged,
                        old_result.status.converged,
                    ),
                    residual=_select_tensor(
                        mask,
                        new_result.status.residual,
                        old_result.status.residual,
                    ),
                    invalid_temperature=_select_tensor(
                        mask,
                        new_result.status.invalid_temperature,
                        old_result.status.invalid_temperature,
                    ),
                    too_many_electrons=_select_tensor(
                        mask,
                        new_result.status.too_many_electrons,
                        old_result.status.too_many_electrons,
                    ),
                    electron_count_ok=_select_tensor(
                        mask,
                        new_result.status.electron_count_ok,
                        old_result.status.electron_count_ok,
                    ),
                ),
            ),
            chemical_potential_ref=_select_tensor(
                mask,
                new_state.chemical_potential_ref,
                old_state.chemical_potential_ref,
            ),
            chemical_potential_offset=_select_tensor(
                mask,
                new_state.chemical_potential_offset,
                old_state.chemical_potential_offset,
            ),
            gap=_select_tensor(mask, new_state.gap, old_state.gap),
            below=_select_tensor(mask, new_state.below, old_state.below),
            active=_select_tensor(mask, new_state.active, old_state.active),
            full=_select_tensor(mask, new_state.full, old_state.full),
            empty=_select_tensor(mask, new_state.empty, old_state.empty),
            too_many=_select_tensor(
                mask, new_state.too_many, old_state.too_many
            ),
            kt_positive=_select_tensor(
                mask, new_state.kt_positive, old_state.kt_positive
            ),
            orbital_mask=select_optional(
                new_state.orbital_mask, old_state.orbital_mask
            ),
        )

    if new.fermi_status is None and old.fermi_status is None:
        fermi_status = None
    elif new.fermi_status is None or old.fermi_status is None:
        raise ValueError("SCF map result structures must match.")
    else:
        fermi_status = type(new.fermi_status)(
            converged=_select_tensor(
                mask, new.fermi_status.converged, old.fermi_status.converged
            ),
            residual=_select_tensor(
                mask, new.fermi_status.residual, old.fermi_status.residual
            ),
            invalid_temperature=_select_tensor(
                mask,
                new.fermi_status.invalid_temperature,
                old.fermi_status.invalid_temperature,
            ),
            too_many_electrons=_select_tensor(
                mask,
                new.fermi_status.too_many_electrons,
                old.fermi_status.too_many_electrons,
            ),
            electron_count_ok=_select_tensor(
                mask,
                new.fermi_status.electron_count_ok,
                old.fermi_status.electron_count_ok,
            ),
        )

    spectral = SCFSpectralResult(
        hamiltonian=_select_tensor(
            mask, new.spectral.hamiltonian, old.spectral.hamiltonian
        ),
        density=_select_tensor(
            mask, new.spectral.density, old.spectral.density
        ),
        emo=_select_tensor(mask, new.spectral.emo, old.spectral.emo),
        coefficients=_select_tensor(
            mask, new.spectral.coefficients, old.spectral.coefficients
        ),
        occupation=_select_tensor(
            mask, new.spectral.occupation, old.spectral.occupation
        ),
        charges=SCFCharges(
            mono=_select_tensor(
                mask, new.spectral.charges.mono, old.spectral.charges.mono
            ),
            dipole=select_optional(
                new.spectral.charges.dipole, old.spectral.charges.dipole
            ),
            quad=select_optional(
                new.spectral.charges.quad, old.spectral.charges.quad
            ),
        ),
        orbital_energy=_select_tensor(
            mask, new.spectral.orbital_energy, old.spectral.orbital_energy
        ),
        fermi_status=fermi_status,
        _fermi_state=select_fermi_state(
            new.spectral._fermi_state, old.spectral._fermi_state
        ),
    )
    if new.driving_potential is None and old.driving_potential is None:
        driving_potential = None
    elif new.driving_potential is None or old.driving_potential is None:
        raise ValueError("SCF map result structures must match.")
    else:
        driving_potential = SCFPotential(
            mono=select_optional(
                new.driving_potential.mono, old.driving_potential.mono
            ),
            dipole=select_optional(
                new.driving_potential.dipole, old.driving_potential.dipole
            ),
            quad=select_optional(
                new.driving_potential.quad, old.driving_potential.quad
            ),
        )
    return SCFMapResult(
        mapped_x=_select_tensor(mask, new.mapped_x, old.mapped_x),
        spectral=spectral,
        driving_potential=driving_potential,
    )


def _solve_map(
    guess: Tensor,
    settings: SCFSolverSettings,
    map_fn: Callable[[Tensor], SCFMapResult],
    state_projection: Callable[[Tensor], Tensor] | None = None,
    *,
    eager: bool = False,
    record_trace: bool = False,
    residual_projection: Callable[[Tensor], Tensor] | None = None,
    project_initial_state: bool = False,
    problem: SCFProblem | None = None,
    strategy: SCFStrategy | None = None,
    certification_profile: SCFInternalCertificationProfile = (
        SCF_CERTIFICATION_STANDARD
    ),
) -> _SolverOutcome:
    """Run simple mixing through fixed-trip or eager single-System control."""
    method = None
    if strategy is not None:
        if problem is None:
            raise ValueError("An SCFStrategy requires an SCFProblem.")
        method = strategy.single_stage().method

    controller = _solve_map_control_eager if eager else _solve_map_control
    common = dict(
        maxiter=settings.maxiter,
        mix_guess=settings.mix_guess,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        map_fn=map_fn,
        state_projection=state_projection,
        damping_settings=settings.dynamic_damping,
        initial_damp=settings.damp,
        mix_fn=(
            None if method is not None else _simple_mixer_controller_adapter
        ),
        residual_projection=residual_projection,
        project_initial_state=project_initial_state,
        problem=problem,
        method=method,
        certification_profile=certification_profile,
        requested_x_atol=settings.requested_x_atol,
        requested_x_atol_max=settings.requested_x_atol_max,
    )
    if eager:
        if (
            record_trace
            or residual_projection is not None
            or project_initial_state
        ):
            raise ValueError(
                "Solver trace/projection requires fixed-trip execution."
            )
        common.pop("residual_projection")
        common.pop("project_initial_state")
        return controller(guess, **common)
    return controller(
        guess,
        **common,
        record_trace=record_trace,
    )


def _solve_map_control(
    guess: Tensor,
    *,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    map_fn: Callable[[Tensor], SCFMapResult],
    mix_fn: (
        Callable[
            [Tensor, Tensor, object | None, int, Tensor, Tensor],
            tuple[Tensor, object | None],
        ]
        | None
    ) = None,
    problem: SCFProblem | None = None,
    method: SCFMethod | None = None,
    initial_damp: float = 0.4,
    damping_settings: DynamicDampingSettings = DynamicDampingSettings(),
    state_projection: Callable[[Tensor], Tensor] | None = None,
    record_trace: bool = False,
    residual_projection: Callable[[Tensor], Tensor] | None = None,
    project_initial_state: bool = False,
    certification_profile: SCFInternalCertificationProfile = (
        SCF_CERTIFICATION_STANDARD
    ),
    requested_x_atol: float | None = None,
    requested_x_atol_max: float | None = None,
) -> _SolverOutcome:
    """Run transform-compatible fixed-trip control around a raw map."""
    state = _initialize_solver_control(
        guess,
        maxiter=maxiter,
        mix_guess=mix_guess,
        initial_damp=initial_damp,
        map_fn=map_fn,
        mix_fn=mix_fn,
        problem=problem,
        method=method,
        state_projection=state_projection,
        record_initial_status=maxiter > 0,
        residual_projection=residual_projection,
        project_initial_state=project_initial_state,
    )

    observations = (
        (
            (
                state.map_input_x,
                state.map_result,
                state.current_x,
                state.residual,
                state.active,
                state.active,
                state.iterations,
                state.damping_state.value,
                state.method_state,
            ),
        )
        if record_trace
        else ()
    )

    if maxiter == 0:
        return _solver_outcome(
            state,
            converged=torch.zeros_like(state.active),
            trace=(
                _materialize_solver_trace(guess, observations)
                if record_trace
                else None
            ),
        )

    thresholds = (
        None
        if problem is None
        else resolve_certification_thresholds(
            problem,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            dtype=guess.dtype,
            profile=certification_profile,
            requested_x_atol=requested_x_atol,
            requested_x_atol_max=requested_x_atol_max,
        )
    )

    for index in range(maxiter):
        map_input = state.current_x
        active_before = state.active
        candidate = map_fn(state.current_x)
        candidate_certification = None
        if problem is not None and thresholds is not None:
            candidate_x = (
                state_projection(candidate.mapped_x)
                if project_initial_state and state_projection is not None
                else candidate.mapped_x
            )
            candidate_residual = _residual(
                candidate_x - state.current_x, residual_projection
            )
            candidate_certification = certify_scf_core_state(
                problem,
                SCFPhysicalState(state.current_x, candidate),
                thresholds,
                fixed_point_residual=candidate_residual,
            )
        state = _solver_control_step(
            state,
            candidate,
            mixer_step=index + (2 if mix_guess else 1),
            mix_fn=mix_fn,
            problem=problem,
            method=method,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            damping_settings=damping_settings,
            state_projection=state_projection,
            residual_projection=residual_projection,
            project_initial_state=project_initial_state,
            certification=candidate_certification,
        )
        if record_trace:
            observations += (
                (
                    map_input,
                    candidate,
                    state.current_x,
                    state.residual,
                    active_before,
                    state.active,
                    state.iterations,
                    state.damping_state.value,
                    state.method_state,
                ),
            )

    if problem is not None and thresholds is not None:
        state = replace(
            state,
            certification=certify_scf_core_state(
                problem,
                SCFPhysicalState(state.map_input_x, state.map_result),
                thresholds,
                fixed_point_residual=SCFResidual(
                    state.residual[0], state.residual[1]
                ),
            ),
        )

    return _solver_outcome(
        state,
        converged=state.certified,
        trace=(
            _materialize_solver_trace(guess, observations)
            if record_trace
            else None
        ),
    )


def _solve_map_control_eager(
    guess: Tensor,
    *,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    map_fn: Callable[[Tensor], SCFMapResult],
    mix_fn: (
        Callable[
            [Tensor, Tensor, object | None, int, Tensor, Tensor],
            tuple[Tensor, object | None],
        ]
        | None
    ) = None,
    problem: SCFProblem | None = None,
    method: SCFMethod | None = None,
    initial_damp: float = 0.4,
    damping_settings: DynamicDampingSettings = DynamicDampingSettings(),
    state_projection: Callable[[Tensor], Tensor] | None = None,
    residual_projection: Callable[[Tensor], Tensor] | None = None,
    project_initial_state: bool = False,
    certification_profile: SCFInternalCertificationProfile = (
        SCF_CERTIFICATION_STANDARD
    ),
    requested_x_atol: float | None = None,
    requested_x_atol_max: float | None = None,
) -> _SolverOutcome:
    """Run eager single-System control, stopping after accepted convergence.

    This host-controlled controller is for production eager execution only.
    Transform and vmap users must use :func:`_solve_map_control`, whose trip
    count is static and which contains no tensor-dependent Python control.
    Both controllers call the same pure initialization and transition helpers.
    """
    initial = map_fn(guess)
    state = _initialize_solver_control(
        guess,
        maxiter=maxiter,
        mix_guess=mix_guess,
        initial_damp=initial_damp,
        map_fn=lambda _: initial,
        mix_fn=mix_fn,
        problem=problem,
        method=method,
        state_projection=state_projection,
        initial_result=initial,
    )

    if maxiter == 0:
        return _solver_outcome(state, converged=torch.zeros_like(state.active))

    thresholds = (
        None
        if problem is None
        else resolve_certification_thresholds(
            problem,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            dtype=guess.dtype,
            profile=certification_profile,
            requested_x_atol=requested_x_atol,
            requested_x_atol_max=requested_x_atol_max,
        )
    )
    if (
        problem is not None
        and thresholds is not None
        and bool(state.terminal_failed)
    ):
        state = replace(
            state,
            certification=certify_scf_core_state(
                problem,
                SCFPhysicalState(state.map_input_x, state.map_result),
                thresholds,
                fixed_point_residual=SCFResidual(
                    state.residual[0], state.residual[1]
                ),
            ),
        )

    for index in range(maxiter):
        # The tensor-to-bool read is deliberately isolated to this eager,
        # single-System controller. The fixed-trip controller never reads it.
        if not bool(state.active):
            break
        candidate = map_fn(state.current_x)
        candidate_certification = None
        if problem is not None and thresholds is not None:
            candidate_x = (
                state_projection(candidate.mapped_x)
                if project_initial_state and state_projection is not None
                else candidate.mapped_x
            )
            candidate_residual = _residual(
                candidate_x - state.current_x, residual_projection
            )
            raw_candidate = SCFConvergence(x_atol, x_atol_max).converged(
                candidate_residual
            )
            physical_state = SCFPhysicalState(state.current_x, candidate)
            terminal_code = terminal_failure_code(physical_state)
            should_certify = bool(raw_candidate) or bool(
                terminal_code != int(SCFCoreFailure.NONE)
            )
            if should_certify:
                candidate_certification = certify_scf_core_state(
                    problem,
                    physical_state,
                    thresholds,
                    fixed_point_residual=candidate_residual,
                )
        state = _solver_control_step(
            state,
            candidate,
            mixer_step=index + (2 if mix_guess else 1),
            mix_fn=mix_fn,
            problem=problem,
            method=method,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            damping_settings=damping_settings,
            state_projection=state_projection,
            residual_projection=residual_projection,
            project_initial_state=project_initial_state,
            certification=candidate_certification,
        )

    if problem is not None and thresholds is not None and bool(state.active):
        state = replace(
            state,
            certification=certify_scf_core_state(
                problem,
                SCFPhysicalState(
                    state.map_input_x,
                    state.map_result,
                ),
                thresholds,
                fixed_point_residual=SCFResidual(
                    state.residual[0], state.residual[1]
                ),
            ),
        )

    return _solver_outcome(state, converged=state.certified)


def _initialize_solver_control(
    guess: Tensor,
    *,
    maxiter: int,
    mix_guess: bool,
    initial_damp: float,
    map_fn: Callable[[Tensor], SCFMapResult],
    mix_fn: (
        Callable[
            [Tensor, Tensor, object | None, int, Tensor, Tensor],
            tuple[Tensor, object | None],
        ]
        | None
    ),
    problem: SCFProblem | None,
    method: SCFMethod | None,
    state_projection: Callable[[Tensor], Tensor] | None,
    initial_result: SCFMapResult | None = None,
    record_initial_status: bool = True,
    residual_projection: Callable[[Tensor], Tensor] | None = None,
    project_initial_state: bool = False,
) -> _SolverControlState:
    """Initialize shared solver state and the optional first mix transition."""
    initial_guess = (
        state_projection(guess)
        if project_initial_state and state_projection is not None
        else guess
    )
    initial = (
        map_fn(initial_guess) if initial_result is None else initial_result
    )
    trajectory_status = (
        trajectory_status_from_map(initial.fermi_status)
        if record_initial_status
        else None
    )
    initial_terminal = terminal_failure_code(
        SCFPhysicalState(initial_guess, initial)
    ) != int(SCFCoreFailure.NONE)
    initial_x = (
        state_projection(initial.mapped_x)
        if project_initial_state and state_projection is not None
        else initial.mapped_x
    )
    initial_residual = _residual(
        initial_x - initial_guess, residual_projection
    ).as_tensor()
    active = ~initial_terminal
    damping = DampingState(
        guess.new_tensor(initial_damp),
        torch.zeros((), dtype=torch.bool, device=guess.device),
    )
    method_state: object | None = None
    current_x = _select_tensor(initial_terminal, initial_guess, initial_x)
    if maxiter > 0 and mix_guess:
        current_x, method_state = _transition_proposal(
            initial_guess,
            _finite_map_result(initial),
            current_x,
            method_state,
            1,
            active,
            damping.value,
            mix_fn=mix_fn,
            problem=problem,
            method=method,
        )
        if state_projection is not None:
            current_x = state_projection(current_x)
    return _SolverControlState(
        current_x=current_x,
        map_input_x=initial_guess,
        method_state=method_state,
        damping_state=damping,
        map_result=initial,
        iterations=torch.ones((), dtype=torch.int64, device=guess.device),
        residual=initial_residual,
        active=active,
        trajectory_status=trajectory_status,
        certified=torch.zeros_like(active),
        terminal_failed=initial_terminal,
        certification=None,
    )


def _finite_map_result(result: SCFMapResult) -> SCFMapResult:
    """Replace non-finite map leaves before an inactive initial transition."""

    def finite_tree(value):
        if isinstance(value, Tensor):
            if value.is_floating_point() or value.is_complex():
                return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return value
        if is_dataclass(value):
            return replace(
                value,
                **{
                    item.name: finite_tree(getattr(value, item.name))
                    for item in fields(value)
                },
            )
        if isinstance(value, tuple):
            return tuple(finite_tree(item) for item in value)
        return value

    return cast(SCFMapResult, finite_tree(result))


def _solver_control_step(
    state: _SolverControlState,
    candidate: SCFMapResult,
    *,
    mixer_step: int,
    mix_fn: (
        Callable[
            [Tensor, Tensor, object | None, int, Tensor, Tensor],
            tuple[Tensor, object | None],
        ]
        | None
    ),
    problem: SCFProblem | None,
    method: SCFMethod | None,
    x_atol: float,
    x_atol_max: float,
    damping_settings: DynamicDampingSettings,
    state_projection: Callable[[Tensor], Tensor] | None,
    residual_projection: Callable[[Tensor], Tensor] | None,
    project_initial_state: bool,
    certification: SCFCertificationCoreResult | None = None,
) -> _SolverControlState:
    """Advance one raw map using canonical result/mixer/status ordering."""
    trajectory_status = accumulate_trajectory_status(
        state.trajectory_status, candidate.fermi_status, state.active
    )
    candidate_x = (
        state_projection(candidate.mapped_x)
        if project_initial_state and state_projection is not None
        else candidate.mapped_x
    )
    candidate_residual = _residual(
        candidate_x - state.current_x, residual_projection
    )
    is_converged = SCFConvergence(x_atol, x_atol_max).converged(
        candidate_residual
    )
    if certification is None:
        certified_event = state.active & is_converged
        terminal_event = torch.zeros_like(state.active)
    else:
        certified_event = state.active & certification.certified
        terminal_event = state.active & certification.terminal_failed
    map_result = _select_map_result(state.active, candidate, state.map_result)
    residual = _select_tensor(
        state.active, candidate_residual.as_tensor(), state.residual
    )
    iterations = torch.where(
        state.active, state.iterations + 1, state.iterations
    )
    transition_active = state.active & ~certified_event & ~terminal_event
    transition_candidate_x = _select_tensor(
        terminal_event, state.current_x, candidate_x
    )
    transition_map_result = _select_map_result(
        terminal_event, state.map_result, candidate
    )
    mixed_x, method_state = _transition_proposal(
        state.current_x,
        transition_map_result,
        transition_candidate_x,
        state.method_state,
        mixer_step,
        transition_active,
        state.damping_state.value,
        mix_fn=mix_fn,
        problem=problem,
        method=method,
    )
    damping = _update_damping(
        state.damping_state,
        candidate_residual.l2,
        transition_active,
        damping_settings,
    )
    if state_projection is not None:
        mixed_x = state_projection(mixed_x)
    current_x = _select_tensor(transition_active, mixed_x, state.current_x)
    return _SolverControlState(
        current_x=current_x,
        map_input_x=_select_tensor(
            state.active, state.current_x, state.map_input_x
        ),
        method_state=method_state,
        damping_state=damping,
        map_result=map_result,
        iterations=iterations,
        residual=residual,
        active=transition_active,
        trajectory_status=trajectory_status,
        certified=state.certified | certified_event,
        terminal_failed=state.terminal_failed | terminal_event,
        certification=(
            state.certification
            if certification is None
            else (
                certification
                if state.certification is None
                else _select_core_certification(
                    state.active, certification, state.certification
                )
            )
        ),
    )


def _solver_outcome(
    state: _SolverControlState,
    *,
    converged: Tensor,
    trace: SCFSolverTrace | None = None,
) -> _SolverOutcome:
    """Materialize solver status from shared controller state."""
    return _SolverOutcome(
        root_x=state.map_input_x,
        map_result=state.map_result,
        iterations=state.iterations,
        converged=converged,
        residual=state.residual,
        trajectory_status=state.trajectory_status,
        damping_state=state.damping_state,
        terminal_failed=state.terminal_failed,
        certification=state.certification,
        trace=trace,
    )


def _mixer_tensors(state: object | None) -> tuple[Tensor, ...]:
    """Flatten tensor-valued mixer history in its stable dataclass order."""
    if state is None:
        return ()
    if isinstance(state, Tensor):
        return (state,)
    if isinstance(state, tuple):
        return tuple(value for item in state for value in _mixer_tensors(item))
    if is_dataclass(state):
        return tuple(
            value
            for item in fields(state)
            for value in _mixer_tensors(getattr(state, item.name))
        )
    return ()


def _materialize_solver_trace(
    initial_guess: Tensor,
    observations: tuple[tuple[object, ...], ...],
) -> SCFSolverTrace:
    """Stack optional observations from the existing solver transitions."""
    map_inputs = tuple(cast(Tensor, row[0]) for row in observations)
    mapped = tuple(cast(SCFMapResult, row[1]) for row in observations)
    iterates = tuple(cast(Tensor, row[2]) for row in observations)
    statuses = tuple(item.spectral.fermi_status for item in mapped)

    def stack_optional(name: str) -> Tensor | None:
        values = tuple(
            None if status is None else getattr(status, name)
            for status in statuses
        )
        if values[0] is None:
            return None
        return torch.stack(cast(tuple[Tensor, ...], values))

    mixer_rows = tuple(_mixer_tensors(row[8]) for row in observations)
    if mixer_rows and mixer_rows[0]:
        mixer_state = tuple(
            torch.stack(tuple(row[index] for row in mixer_rows))
            for index in range(len(mixer_rows[0]))
        )
    else:
        mixer_state = ()
    driving = tuple(item.driving_potential for item in mapped)
    driving_mono = tuple(
        None if item is None else item.mono for item in driving
    )
    return SCFSolverTrace(
        initial_guess=initial_guess,
        map_inputs=torch.stack(map_inputs),
        raw_mapped=torch.stack(tuple(item.mapped_x for item in mapped)),
        iterates=torch.stack(iterates),
        hamiltonians=torch.stack(
            tuple(item.spectral.hamiltonian for item in mapped)
        ),
        densities=torch.stack(tuple(item.spectral.density for item in mapped)),
        charges=torch.stack(tuple(item.charges.mono for item in mapped)),
        driving_potentials=(
            None
            if driving_mono[0] is None
            else torch.stack(cast(tuple[Tensor, ...], driving_mono))
        ),
        residuals=torch.stack(
            tuple(cast(Tensor, row[3]) for row in observations)
        ),
        active_before=torch.stack(
            tuple(cast(Tensor, row[4]) for row in observations)
        ),
        active_after=torch.stack(
            tuple(cast(Tensor, row[5]) for row in observations)
        ),
        iterations=torch.stack(
            tuple(cast(Tensor, row[6]) for row in observations)
        ),
        damping=torch.stack(
            tuple(cast(Tensor, row[7]) for row in observations)
        ),
        fermi_converged=stack_optional("converged"),
        fermi_electron_count_ok=stack_optional("electron_count_ok"),
        fermi_invalid_temperature=stack_optional("invalid_temperature"),
        fermi_too_many_electrons=stack_optional("too_many_electrons"),
        fermi_residual=stack_optional("residual"),
        mixer_state=mixer_state,
    )


def finalize_scf(
    map_result: SCFMapResult,
    inputs: SCFInputs,
    iterations: Tensor,
    converged: Tensor,
    residual: Tensor,
    trajectory_status: SCFTrajectoryStatus | None,
    damping_state: DampingState | None = None,
    trace: SCFSolverTrace | None = None,
    certification: SCFCertificationCoreResult | None = None,
    terminal_failed: Tensor | None = None,
    root_x: Tensor | None = None,
) -> SCFSolveResult:
    """Finalize energy/potential from the selected raw map output charges."""
    final_charges = SCFCharges(
        mono=map_result.charges.mono,
        dipole=map_result.charges.dipole,
        quad=map_result.charges.quad,
    )
    final_potential = potential_from_charges(final_charges, inputs)
    h0_energy = inputs.index.reduce_orbital_to_atom(map_result.orbital_energy)
    energy = h0_energy + interaction_energy_from_charges(final_charges, inputs)
    fenergy = free_energy_from_spectral(map_result.spectral, inputs)
    return SCFSolveResult(
        map_result=map_result,
        final_evaluation=map_result.spectral,
        final_charges=final_charges,
        final_potential=final_potential,
        energy=energy,
        fenergy=fenergy,
        root_x=(map_result.mapped_x if root_x is None else root_x),
        iterations=iterations,
        converged=converged,
        residual=residual,
        trajectory_status=trajectory_status,
        diagnostics=SCFSolverDiagnostics(
            final_damp=(
                map_result.mapped_x.new_tensor(0.0)
                if damping_state is None
                else damping_state.value
            ),
            dynamic_damping_switched=(
                torch.zeros(
                    (), dtype=torch.bool, device=map_result.mapped_x.device
                )
                if damping_state is None
                else damping_state.switched
            ),
            terminal_failed=(
                torch.zeros(
                    (), dtype=torch.bool, device=map_result.mapped_x.device
                )
                if terminal_failed is None
                else terminal_failed
            ),
            certification=(
                None
                if certification is None
                else SCFPhysicalCertificationDiagnostics(certification)
            ),
        ),
        trace=trace,
    )


def finalize_scf_potential(
    map_result: SCFMapResult,
    inputs: SCFInputs,
    layout: PotentialLayout,
    iterations: Tensor,
    converged: Tensor,
    residual: Tensor,
    trajectory_status: SCFTrajectoryStatus | None,
    damping_state: DampingState | None = None,
    trace: SCFSolverTrace | None = None,
    certification: SCFCertificationCoreResult | None = None,
    terminal_failed: Tensor | None = None,
    root_x: Tensor | None = None,
) -> SCFSolveResult:
    """Finalize potential mode from one extra representation conversion."""
    final_potential_state = unpack_potential(map_result.mapped_x, layout)
    final_evaluation = evaluate_potential(final_potential_state, inputs)
    final_charges = final_evaluation.charges
    final_potential = potential_from_charges(final_charges, inputs)
    h0_energy = inputs.index.reduce_orbital_to_atom(
        final_evaluation.orbital_energy
    )
    energy = h0_energy + interaction_energy_from_charges(final_charges, inputs)
    fenergy = free_energy_from_spectral(final_evaluation, inputs)
    return SCFSolveResult(
        map_result=map_result,
        final_evaluation=final_evaluation,
        final_charges=final_charges,
        final_potential=final_potential,
        energy=energy,
        fenergy=fenergy,
        root_x=(map_result.mapped_x if root_x is None else root_x),
        iterations=iterations,
        converged=converged,
        residual=residual,
        trajectory_status=trajectory_status,
        diagnostics=SCFSolverDiagnostics(
            final_damp=(
                map_result.mapped_x.new_tensor(0.0)
                if damping_state is None
                else damping_state.value
            ),
            dynamic_damping_switched=(
                torch.zeros(
                    (), dtype=torch.bool, device=map_result.mapped_x.device
                )
                if damping_state is None
                else damping_state.switched
            ),
            terminal_failed=(
                torch.zeros(
                    (), dtype=torch.bool, device=map_result.mapped_x.device
                )
                if terminal_failed is None
                else terminal_failed
            ),
            certification=(
                None
                if certification is None
                else SCFPhysicalCertificationDiagnostics(certification)
            ),
        ),
        trace=trace,
    )


def finalize_scf_fock(
    map_result: SCFMapResult,
    inputs: SCFInputs,
    iterations: Tensor,
    converged: Tensor,
    residual: Tensor,
    trajectory_status: SCFTrajectoryStatus | None,
    damping_state: DampingState | None = None,
    trace: SCFSolverTrace | None = None,
    certification: SCFCertificationCoreResult | None = None,
    terminal_failed: Tensor | None = None,
    root_x: Tensor | None = None,
) -> SCFSolveResult:
    """Finalize Fock mode from one spectrum of the selected mapped matrix."""
    final_evaluation = evaluate_hamiltonian(map_result.mapped_x, inputs)
    final_charges = final_evaluation.charges
    final_potential = potential_from_charges(final_charges, inputs)
    h0_energy = inputs.index.reduce_orbital_to_atom(
        final_evaluation.orbital_energy
    )
    energy = h0_energy + interaction_energy_from_charges(final_charges, inputs)
    fenergy = free_energy_from_spectral(final_evaluation, inputs)
    return SCFSolveResult(
        map_result=map_result,
        final_evaluation=final_evaluation,
        final_charges=final_charges,
        final_potential=final_potential,
        energy=energy,
        fenergy=fenergy,
        root_x=(map_result.mapped_x if root_x is None else root_x),
        iterations=iterations,
        converged=converged,
        residual=residual,
        trajectory_status=trajectory_status,
        diagnostics=SCFSolverDiagnostics(
            final_damp=(
                map_result.mapped_x.new_tensor(0.0)
                if damping_state is None
                else damping_state.value
            ),
            dynamic_damping_switched=(
                torch.zeros(
                    (), dtype=torch.bool, device=map_result.mapped_x.device
                )
                if damping_state is None
                else damping_state.switched
            ),
            terminal_failed=(
                torch.zeros(
                    (), dtype=torch.bool, device=map_result.mapped_x.device
                )
                if terminal_failed is None
                else terminal_failed
            ),
            certification=(
                None
                if certification is None
                else SCFPhysicalCertificationDiagnostics(certification)
            ),
        ),
        trace=trace,
    )


def solve_scf_simple(
    guess: Tensor,
    inputs: SCFInputs,
    settings: SCFSolverSettings,
    *,
    eager: bool = False,
    record_trace: bool = False,
    state_projection: Callable[[Tensor], Tensor] | None = None,
    residual_projection: Callable[[Tensor], Tensor] | None = None,
    project_initial_state: bool = False,
    certification_profile: SCFInternalCertificationProfile = (
        SCF_CERTIFICATION_STANDARD
    ),
) -> SCFSolveResult:
    """Solve charge mode with simple mixing and selected eager/fixed control."""
    problem = SCFProblem(inputs, SCFRepresentation.CHARGE)
    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        state_projection=state_projection,
        eager=eager,
        record_trace=record_trace,
        residual_projection=residual_projection,
        project_initial_state=project_initial_state,
        problem=problem,
        strategy=SCFStrategy((SCFStage(LinearMethod()),)),
        certification_profile=certification_profile,
    )
    return finalize_scf(
        outcome.map_result,
        inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
        outcome.trace,
        certification=outcome.certification,
        terminal_failed=outcome.terminal_failed,
        root_x=outcome.root_x,
    )


def solve_scf_potential_simple(
    guess: Tensor,
    inputs: SCFInputs,
    layout: PotentialLayout,
    settings: SCFSolverSettings,
    *,
    eager: bool = False,
    record_trace: bool = False,
    state_projection: Callable[[Tensor], Tensor] | None = None,
    residual_projection: Callable[[Tensor], Tensor] | None = None,
    project_initial_state: bool = False,
    certification_profile: SCFInternalCertificationProfile = (
        SCF_CERTIFICATION_STANDARD
    ),
) -> SCFSolveResult:
    """Solve potential mode with selected simple-mixing control."""
    problem = SCFProblem(
        inputs, SCFRepresentation.POTENTIAL, potential_layout=layout
    )
    project = lambda value: _project_potential_padding(value, layout)
    if state_projection is not None:
        project = lambda value: state_projection(
            _project_potential_padding(value, layout)
        )
    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        state_projection=project,
        eager=eager,
        record_trace=record_trace,
        residual_projection=residual_projection,
        project_initial_state=project_initial_state,
        problem=problem,
        strategy=SCFStrategy((SCFStage(LinearMethod()),)),
        certification_profile=certification_profile,
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
        outcome.trace,
        certification=outcome.certification,
        terminal_failed=outcome.terminal_failed,
        root_x=outcome.root_x,
    )


def solve_scf_fock_simple(
    guess: Tensor,
    inputs: SCFInputs,
    settings: SCFSolverSettings,
    *,
    eager: bool = False,
    record_trace: bool = False,
    state_projection: Callable[[Tensor], Tensor] | None = None,
    residual_projection: Callable[[Tensor], Tensor] | None = None,
    project_initial_state: bool = False,
    certification_profile: SCFInternalCertificationProfile = (
        SCF_CERTIFICATION_STANDARD
    ),
) -> SCFSolveResult:
    """Solve Fock mode with simple mixing and selected eager/fixed control."""
    problem = SCFProblem(inputs, SCFRepresentation.FOCK)
    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        state_projection=state_projection,
        eager=eager,
        record_trace=record_trace,
        residual_projection=residual_projection,
        project_initial_state=project_initial_state,
        problem=problem,
        strategy=SCFStrategy((SCFStage(LinearMethod()),)),
        certification_profile=certification_profile,
    )
    return finalize_scf_fock(
        outcome.map_result,
        inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
        outcome.trace,
        certification=outcome.certification,
        terminal_failed=outcome.terminal_failed,
        root_x=outcome.root_x,
    )
