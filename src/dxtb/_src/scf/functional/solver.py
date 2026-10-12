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
from dataclasses import dataclass
from typing import ClassVar

import torch
from tad_mctc.batch import real_atoms
from tad_mctc.math import einsum

from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.components.interactions.container import Charges
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction import filling, mulliken
from dxtb._src.wavefunction.filling import FermiStatus

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
from .map import (
    evaluate_hamiltonian,
    evaluate_potential,
    fock_map,
    potential_from_charges,
    potential_map,
    scf_map,
)

__all__ = [
    "DynamicDampingSettings",
    "DampingState",
    "SCFSolverDiagnostics",
    "SCFResidual",
    "SCFSolveResult",
    "SCFTrajectoryStatus",
    "SCFSolverSettings",
    "finalize_scf",
    "finalize_scf_fock",
    "functional_subsolves_ok",
    "functional_converged_and_valid",
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


@dataclass(frozen=True, eq=False)
class SCFSolverSettings:
    """Static root-finding settings for the E4.1 simple solver."""

    maxiter: int
    mix_guess: bool
    damp: float
    x_atol: float
    x_atol_max: float
    dynamic_damping: DynamicDampingSettings = DynamicDampingSettings()


@dataclass(frozen=True, eq=False)
class SCFResidual:
    """L2 and max-absolute norms of the raw physical residual."""

    l2: Tensor
    max_abs: Tensor

    def as_tensor(self) -> Tensor:
        """Return norms in the public residual order."""
        return torch.stack((self.l2, self.max_abs))


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
    iterations: Tensor
    converged: Tensor
    residual: Tensor
    trajectory_status: SCFTrajectoryStatus | None
    diagnostics: SCFSolverDiagnostics | None = None

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


FERMI_FAILURE_NONE = 0
FERMI_FAILURE_INVALID_TEMPERATURE = 1
FERMI_FAILURE_TOO_MANY_ELECTRONS = 2
FERMI_FAILURE_NONCONVERGENCE = 3
FERMI_FAILURE_ELECTRON_COUNT = 4


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
    """Return raw-trajectory and final-public-spectrum validity only.

    This is not an eager return policy and does not include outer SCF
    convergence. A future adapter must handle ``maxiter=0`` separately.
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


def functional_converged_and_valid(result: SCFSolveResult) -> Tensor:
    """Combine outer convergence and Fermi subsolve validity explicitly."""
    return result.converged & functional_subsolves_ok(result)


@dataclass(frozen=True, eq=False)
class _SolverOutcome:
    """Transform-friendly outcome before interaction finalization."""

    map_result: SCFMapResult
    iterations: Tensor
    converged: Tensor
    residual: Tensor
    trajectory_status: SCFTrajectoryStatus | None
    damping_state: DampingState


@dataclass(frozen=True, eq=False)
class _SolverControlState:
    """Shared immutable state advanced by fixed-trip and eager controllers."""

    current_x: Tensor
    mixer_state: object | None
    damping_state: DampingState
    map_result: SCFMapResult
    iterations: Tensor
    residual: Tensor
    active: Tensor
    trajectory_status: SCFTrajectoryStatus | None


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
    )


def simple_mix(old: Tensor, mapped: Tensor, damp: Tensor | float) -> Tensor:
    """Apply one out-of-place linear mixing transition."""
    return old + (mapped - old) * damp


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


def _residual(delta: Tensor) -> SCFResidual:
    return SCFResidual(
        l2=torch.linalg.vector_norm(delta),
        max_abs=torch.amax(torch.abs(delta)),
    )


def _select_tensor(mask: Tensor, new: Tensor, old: Tensor) -> Tensor:
    return torch.where(mask, new, old)


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
) -> _SolverOutcome:
    """Run simple mixing through fixed-trip or eager single-System control."""
    controller = _solve_map_control_eager if eager else _solve_map_control
    return controller(
        guess,
        maxiter=settings.maxiter,
        mix_guess=settings.mix_guess,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        map_fn=map_fn,
        state_projection=state_projection,
        damping_settings=settings.dynamic_damping,
        initial_damp=settings.damp,
        mix_fn=lambda old, mapped, state, step, transition_active, damp: (
            simple_mix(old, mapped, damp),
            None,
        ),
    )


def _solve_map_control(
    guess: Tensor,
    *,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    map_fn: Callable[[Tensor], SCFMapResult],
    mix_fn: Callable[
        [Tensor, Tensor, object | None, int, Tensor, Tensor],
        tuple[Tensor, object | None],
    ],
    initial_damp: float = 0.4,
    damping_settings: DynamicDampingSettings = DynamicDampingSettings(),
    state_projection: Callable[[Tensor], Tensor] | None = None,
) -> _SolverOutcome:
    """Run transform-compatible fixed-trip control around a raw map."""
    state = _initialize_solver_control(
        guess,
        maxiter=maxiter,
        mix_guess=mix_guess,
        initial_damp=initial_damp,
        map_fn=map_fn,
        mix_fn=mix_fn,
        state_projection=state_projection,
    )

    if maxiter == 0:
        return _solver_outcome(state, converged=torch.zeros_like(state.active))

    for index in range(maxiter):
        candidate = map_fn(state.current_x)
        state = _solver_control_step(
            state,
            candidate,
            mixer_step=index + (2 if mix_guess else 1),
            mix_fn=mix_fn,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            damping_settings=damping_settings,
            state_projection=state_projection,
        )

    return _solver_outcome(state, converged=~state.active)


def _solve_map_control_eager(
    guess: Tensor,
    *,
    maxiter: int,
    mix_guess: bool,
    x_atol: float,
    x_atol_max: float,
    map_fn: Callable[[Tensor], SCFMapResult],
    mix_fn: Callable[
        [Tensor, Tensor, object | None, int, Tensor, Tensor],
        tuple[Tensor, object | None],
    ],
    initial_damp: float = 0.4,
    damping_settings: DynamicDampingSettings = DynamicDampingSettings(),
    state_projection: Callable[[Tensor], Tensor] | None = None,
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
        state_projection=state_projection,
        initial_result=initial,
    )

    if maxiter == 0:
        return _solver_outcome(state, converged=torch.zeros_like(state.active))

    for index in range(maxiter):
        # The tensor-to-bool read is deliberately isolated to this eager,
        # single-System controller. The fixed-trip controller never reads it.
        if not bool(state.active):
            break
        candidate = map_fn(state.current_x)
        state = _solver_control_step(
            state,
            candidate,
            mixer_step=index + (2 if mix_guess else 1),
            mix_fn=mix_fn,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            damping_settings=damping_settings,
            state_projection=state_projection,
        )

    return _solver_outcome(state, converged=~state.active)


def _initialize_solver_control(
    guess: Tensor,
    *,
    maxiter: int,
    mix_guess: bool,
    initial_damp: float,
    map_fn: Callable[[Tensor], SCFMapResult],
    mix_fn: Callable[
        [Tensor, Tensor, object | None, int, Tensor, Tensor],
        tuple[Tensor, object | None],
    ],
    state_projection: Callable[[Tensor], Tensor] | None,
    initial_result: SCFMapResult | None = None,
) -> _SolverControlState:
    """Initialize shared solver state and the optional first mix transition."""
    initial = map_fn(guess) if initial_result is None else initial_result
    trajectory_status = trajectory_status_from_map(initial.fermi_status)
    initial_residual = _residual(initial.mapped_x - guess).as_tensor()
    active = torch.ones((), dtype=torch.bool, device=guess.device)
    damping = DampingState(
        guess.new_tensor(initial_damp),
        torch.zeros((), dtype=torch.bool, device=guess.device),
    )
    mixer_state: object | None = None
    current_x = initial.mapped_x
    if maxiter > 0 and mix_guess:
        current_x, mixer_state = mix_fn(
            guess, initial.mapped_x, mixer_state, 1, active, damping.value
        )
        if state_projection is not None:
            current_x = state_projection(current_x)
    return _SolverControlState(
        current_x=current_x,
        mixer_state=mixer_state,
        damping_state=damping,
        map_result=initial,
        iterations=torch.ones((), dtype=torch.int64, device=guess.device),
        residual=initial_residual,
        active=active,
        trajectory_status=trajectory_status,
    )


def _solver_control_step(
    state: _SolverControlState,
    candidate: SCFMapResult,
    *,
    mixer_step: int,
    mix_fn: Callable[
        [Tensor, Tensor, object | None, int, Tensor, Tensor],
        tuple[Tensor, object | None],
    ],
    x_atol: float,
    x_atol_max: float,
    damping_settings: DynamicDampingSettings,
    state_projection: Callable[[Tensor], Tensor] | None,
) -> _SolverControlState:
    """Advance one raw map using canonical result/mixer/status ordering."""
    trajectory_status = accumulate_trajectory_status(
        state.trajectory_status, candidate.fermi_status, state.active
    )
    candidate_residual = _residual(candidate.mapped_x - state.current_x)
    is_converged = (candidate_residual.l2 < x_atol) & (
        candidate_residual.max_abs < x_atol_max
    )
    map_result = _select_map_result(state.active, candidate, state.map_result)
    residual = _select_tensor(
        state.active, candidate_residual.as_tensor(), state.residual
    )
    iterations = torch.where(
        state.active, state.iterations + 1, state.iterations
    )
    transition_active = state.active & ~is_converged
    mixed_x, mixer_state = mix_fn(
        state.current_x,
        candidate.mapped_x,
        state.mixer_state,
        mixer_step,
        transition_active,
        state.damping_state.value,
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
        mixer_state=mixer_state,
        damping_state=damping,
        map_result=map_result,
        iterations=iterations,
        residual=residual,
        active=transition_active,
        trajectory_status=trajectory_status,
    )


def _solver_outcome(
    state: _SolverControlState, *, converged: Tensor
) -> _SolverOutcome:
    """Materialize solver status from shared controller state."""
    return _SolverOutcome(
        map_result=state.map_result,
        iterations=state.iterations,
        converged=converged,
        residual=state.residual,
        trajectory_status=state.trajectory_status,
        damping_state=state.damping_state,
    )


def interaction_energy_from_charges(
    charges: SCFCharges, inputs: SCFInputs
) -> Tensor:
    """Return atom-resolved interaction energy for explicit final charges."""
    local_charges = Charges(
        mono=charges.mono, dipole=charges.dipole, quad=charges.quad
    )
    return inputs.interactions.get_energy(
        local_charges, inputs.interaction_data, inputs.ihelp
    )


def finalize_scf(
    map_result: SCFMapResult,
    inputs: SCFInputs,
    iterations: Tensor,
    converged: Tensor,
    residual: Tensor,
    trajectory_status: SCFTrajectoryStatus | None,
    damping_state: DampingState | None = None,
) -> SCFSolveResult:
    """Finalize energy/potential from the selected raw map output charges."""
    final_charges = SCFCharges(
        mono=map_result.charges.mono,
        dipole=map_result.charges.dipole,
        quad=map_result.charges.quad,
    )
    final_potential = potential_from_charges(final_charges, inputs)
    h0_energy = inputs.ihelp.reduce_orbital_to_atom(map_result.orbital_energy)
    energy = h0_energy + interaction_energy_from_charges(final_charges, inputs)
    fenergy = free_energy_from_spectral(map_result.spectral, inputs)
    return SCFSolveResult(
        map_result=map_result,
        final_evaluation=map_result.spectral,
        final_charges=final_charges,
        final_potential=final_potential,
        energy=energy,
        fenergy=fenergy,
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
        ),
    )


def free_energy_from_spectral(
    spectral: SCFSpectralResult, inputs: SCFInputs
) -> Tensor:
    """Partition the entropy contribution from one selected spectral phase."""
    if inputs.fermi_settings is None:
        return inputs.integrals.hcore.new_zeros(inputs.numbers.shape)
    settings = inputs.fermi_settings
    if settings.smear:
        entropy = filling.get_fermi_entropy_finite(
            spectral.occupation,
            spectral.emo,
            settings.kt,
            inputs.orbital_mask,
        )
    else:
        entropy = filling.get_fermi_entropy_low_temperature(
            spectral.occupation, settings.kt
        )
    if settings.partition == labels.FERMI_PARTITION_EQUAL:
        real = real_atoms(inputs.numbers)
        count = real.count_nonzero(dim=-1).unsqueeze(-1)
        equal = entropy.sum(-1, keepdim=True) / count
        return torch.where(real, equal.expand_as(real), 0.0)
    if settings.partition == labels.FERMI_PARTITION_ATOMIC:
        entropy_density = einsum(
            "...ik,...k,...jk->...ij",
            spectral.coefficients,
            entropy,
            spectral.coefficients,
        )
        return mulliken.get_atomic_populations(
            inputs.integrals.overlap, entropy_density, inputs.ihelp
        )
    raise ValueError(
        f"Unknown Fermi free-energy partition {settings.partition}."
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
) -> SCFSolveResult:
    """Finalize potential mode from one extra representation conversion."""
    final_potential_state = unpack_potential(map_result.mapped_x, layout)
    final_evaluation = evaluate_potential(final_potential_state, inputs)
    final_charges = final_evaluation.charges
    final_potential = potential_from_charges(final_charges, inputs)
    h0_energy = inputs.ihelp.reduce_orbital_to_atom(
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
        ),
    )


def finalize_scf_fock(
    map_result: SCFMapResult,
    inputs: SCFInputs,
    iterations: Tensor,
    converged: Tensor,
    residual: Tensor,
    trajectory_status: SCFTrajectoryStatus | None,
    damping_state: DampingState | None = None,
) -> SCFSolveResult:
    """Finalize Fock mode from one spectrum of the selected mapped matrix."""
    final_evaluation = evaluate_hamiltonian(map_result.mapped_x, inputs)
    final_charges = final_evaluation.charges
    final_potential = potential_from_charges(final_charges, inputs)
    h0_energy = inputs.ihelp.reduce_orbital_to_atom(
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
        ),
    )


def solve_scf_simple(
    guess: Tensor,
    inputs: SCFInputs,
    settings: SCFSolverSettings,
    *,
    eager: bool = False,
) -> SCFSolveResult:
    """Solve charge mode with simple mixing and selected eager/fixed control."""
    outcome = _solve_map(
        guess, settings, lambda x: scf_map(x, inputs), eager=eager
    )
    return finalize_scf(
        outcome.map_result,
        inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
    )


def solve_scf_potential_simple(
    guess: Tensor,
    inputs: SCFInputs,
    layout: PotentialLayout,
    settings: SCFSolverSettings,
    *,
    eager: bool = False,
) -> SCFSolveResult:
    """Solve potential mode with selected simple-mixing control."""
    outcome = _solve_map(
        guess,
        settings,
        lambda x: potential_map(x, inputs, layout),
        state_projection=lambda x: _project_potential_padding(x, layout),
        eager=eager,
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


def solve_scf_fock_simple(
    guess: Tensor,
    inputs: SCFInputs,
    settings: SCFSolverSettings,
    *,
    eager: bool = False,
) -> SCFSolveResult:
    """Solve Fock mode with simple mixing and selected eager/fixed control."""
    outcome = _solve_map(
        guess, settings, lambda x: fock_map(x, inputs), eager=eager
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
