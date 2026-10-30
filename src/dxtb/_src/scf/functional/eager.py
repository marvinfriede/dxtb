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
"""Eager policy adapter for the parallel functional full-SCF implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from dxtb import IndexHelper, OutputHandler
from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.components.interactions import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.constants import labels
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.scf.nonselfconsistent import solve_nonselfconsistent
from dxtb._src.scf.status import SCFStatus
from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)
from dxtb._src.wavefunction.filling import FermiStatus

from .anderson import AndersonMethod, setup_anderson_method_settings
from .broyden import BroydenMethod, setup_broyden_method_settings
from .core import CoreIndexData
from .data import (
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFPotential,
    functional_scf_interactions_eligible,
    pack_charges,
    pack_potential,
    setup_scf_inputs,
    unpack_charges,
)
from .implicit import _prepare_scf_root, _SCFMapTensorTree
from .map import (
    evaluate_hamiltonian,
    evaluate_potential,
    fock_initial_guess,
    potential_initial_guess,
    unpack_potential,
)
from .method import LinearMethod
from .problem import SCFProblem, SCFRepresentation
from .solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_INVALID_TEMPERATURE,
    FERMI_FAILURE_NONCONVERGENCE,
    FERMI_FAILURE_NONE,
    FERMI_FAILURE_TOO_MANY_ELECTRONS,
    SCFSolveResult,
    SCFSolverSettings,
    SCFTrajectoryStatus,
    _project_potential_padding,
    _solve_map,
    fermi_failure_code,
    fermi_status_ok,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
    setup_solver_settings,
    solve_scf_problem,
)
from .strategy import SCFStage, SCFStrategy

__all__ = [
    "FullSCFRoute",
    "FunctionalCoreResult",
    "EagerFullResult",
    "resolve_full_scf_route",
    "apply_functional_full_policy",
    "functional_core_result",
    "run_functional_full_eager",
    "run_prepared_functional_implicit",
    "run_system_fixed_trip",
]


@dataclass(frozen=True)
class FullSCFRoute:
    """Static decision made before entering either SCF implementation."""

    kind: str
    representation: int
    effective_mixer: int
    warning: str | None = None
    reason: str | None = None


@dataclass(frozen=True, eq=False)
class FunctionalCoreResult:
    """Core-compatible functional values with tensor-valued solver status."""

    charges: SCFCharges
    coefficients: Tensor
    density: Tensor
    emo: Tensor
    energy: Tensor
    fenergy: Tensor
    hamiltonian: Tensor
    occupation: Tensor
    potential: SCFPotential
    iterations: Tensor
    converged: Tensor
    residual: Tensor
    solve_result: SCFSolveResult


@dataclass(frozen=True, eq=False)
class EagerFullResult:
    """Parallel eager execution result, retaining the chosen route."""

    route: FullSCFRoute
    functional: FunctionalCoreResult | None
    legacy: dict[str, object] | None


def resolve_full_scf_route(
    config: ConfigSCF,
    interactions: InteractionList,
    ihelp: IndexHelper,
    numbers: Tensor | None = None,
    integrals: IntegralMatrices | None = None,
    legacy_engine_overrides: bool = False,
) -> FullSCFRoute:
    """Resolve a supported functional route or its explicit batch boundary."""
    if config.scf_mode != labels.SCF_MODE_FULL:
        raise ValueError("FULL route resolution requires SCF_MODE_FULL.")
    if not config.requires_iterations:
        raise ValueError("FULL route resolution requires iterative SCF.")
    if config.maxiter < 0:
        raise ValueError("The functional SCF solver requires maxiter >= 0.")
    if config.fermi.etemp < 0.0:
        raise ValueError("Electronic Temperature must be non-negative.")
    if config.fermi.maxiter < 0:
        raise ValueError("Fermi maxiter must be non-negative.")
    if integrals is not None and (
        integrals.hcore is None or integrals.overlap is None
    ):
        raise ValueError("Functional SCF requires hcore and overlap integrals.")
    if legacy_engine_overrides:
        raise NotImplementedError(
            "Legacy SCF engine positional/keyword overrides are no longer "
            "supported by functional SCF."
        )

    eligible, names = functional_scf_interactions_eligible(interactions)
    if not eligible:
        raise NotImplementedError(
            "Functional SCF does not support interaction extension(s): "
            f"{names}."
        )

    if ihelp.batch_mode != 0 or (numbers is not None and numbers.ndim != 1):
        return FullSCFRoute(
            "batch_compat",
            config.scp_mode,
            config.mixer,
            reason="public packed batch FULL compatibility route",
        )
    if config.mixer == labels.MIXER_BROYDEN:
        return FullSCFRoute("functional", config.scp_mode, config.mixer)
    if config.mixer not in (labels.MIXER_LINEAR, labels.MIXER_ANDERSON):
        raise ValueError(f"Unknown mixer '{config.mixer}'.")
    return FullSCFRoute("functional", config.scp_mode, config.mixer)


def functional_core_result(result: SCFSolveResult) -> FunctionalCoreResult:
    """Expose core values without converting tensor status to Python scalars."""
    return FunctionalCoreResult(
        charges=result.final_charges,
        coefficients=result.coefficients,
        density=result.density,
        emo=result.emo,
        energy=result.energy,
        fenergy=result.fenergy,
        hamiltonian=result.hamiltonian,
        occupation=result.occupation,
        potential=result.final_potential,
        iterations=result.iterations,
        converged=result.converged,
        residual=result.residual,
        solve_result=result,
    )


def public_scf_status(
    trajectory_status: SCFTrajectoryStatus | None,
    final_fermi_status: FermiStatus | None,
    *,
    like: Tensor,
) -> SCFStatus:
    """Project internal trajectory and final-spectrum failures publicly."""
    trajectory_failure = (
        torch.zeros_like(like, dtype=torch.int64)
        if trajectory_status is None
        else trajectory_status.first_failure
    )
    final_failure = (
        torch.zeros_like(like, dtype=torch.int64)
        if final_fermi_status is None
        else fermi_failure_code(final_fermi_status, like=like)
    )
    return SCFStatus(
        trajectory_failure=trajectory_failure,
        final_failure=final_failure,
    )


def _check_subsolve_status(result: SCFSolveResult) -> None:
    """Interpret trajectory failures before the later public spectrum."""
    if result.trajectory_status is not None:
        failure = result.trajectory_status.first_failure
        if bool((failure != FERMI_FAILURE_NONE).any()):
            _raise_failure_code(failure)
    status = result.final_evaluation.fermi_status
    if status is not None:
        from .solver import fermi_failure_code

        code = fermi_failure_code(status, like=result.converged)
        if bool((code != FERMI_FAILURE_NONE).any()):
            _raise_failure_code(code)


def _raise_failure_code(code: Tensor) -> None:
    """Raise the eager error family encoded by the earliest failure code."""
    if bool((code == FERMI_FAILURE_INVALID_TEMPERATURE).any()):
        raise ValueError("Electronic Temperature must be non-negative.")
    if bool((code == FERMI_FAILURE_TOO_MANY_ELECTRONS).any()):
        raise ValueError("Number of electrons exceeds the number of orbitals.")
    if bool((code == FERMI_FAILURE_NONCONVERGENCE).any()):
        raise RuntimeError("Fermi energy failed to converge.")
    if bool((code == FERMI_FAILURE_ELECTRON_COUNT).any()):
        raise RuntimeError("Number of electrons changed during Fermi smearing.")


def _nonconvergence_message(
    config: ConfigSCF, route: FullSCFRoute, switched: bool
) -> str:
    """Format the legacy warning from configured Python values."""
    mixer_label = {
        labels.MIXER_LINEAR: "Simple",
        labels.MIXER_ANDERSON: "Anderson",
        labels.MIXER_BROYDEN: "Broyden",
    }[route.effective_mixer]
    damp = config.damp_dynamic_factor if switched else config.damp
    return (
        f"SCF does not converge after {config.maxiter} cycles using "
        f"{mixer_label} mixing with a damping factor of {damp}."
    )


def apply_functional_full_policy(
    result: SCFSolveResult,
    config: ConfigSCF,
    route: FullSCFRoute,
) -> FunctionalCoreResult:
    """Apply eager Fermi and convergence policy to a completed solve."""
    _check_subsolve_status(result)

    # maxiter=0 is an accepted one-shot even when force_convergence is set.
    if config.maxiter > 0 and not bool(result.converged):
        assert result.diagnostics is not None
        message = _nonconvergence_message(
            config,
            route,
            bool(result.diagnostics.dynamic_damping_switched),
        )
        if config.force_convergence:
            raise SCFConvergenceError(
                f"{message} The physical SCF root was not certified."
            )
        OutputHandler.warn(message, SCFConvergenceWarning)

    return functional_core_result(result)


def _prepare_solver(
    config: ConfigSCF, dtype: torch.dtype
) -> tuple[SCFSolverSettings, SCFStrategy]:
    """Translate the effective configuration into common settings and method."""
    settings = setup_solver_settings(config, dtype)
    if config.mixer == labels.MIXER_LINEAR:
        method = LinearMethod()
    elif config.mixer == labels.MIXER_ANDERSON:
        method = AndersonMethod(setup_anderson_method_settings(config))
    elif config.mixer == labels.MIXER_BROYDEN:
        method = BroydenMethod(setup_broyden_method_settings(config))
    else:
        raise NotImplementedError(
            f"Unsupported functional mixer {config.mixer}."
        )
    return settings, SCFStrategy((SCFStage(method),))


def _representation_label(representation: SCFRepresentation) -> int:
    """Translate the internal working representation to public config labels."""
    return {
        SCFRepresentation.CHARGE: labels.SCP_MODE_CHARGE,
        SCFRepresentation.POTENTIAL: labels.SCP_MODE_POTENTIAL,
        SCFRepresentation.FOCK: labels.SCP_MODE_FOCK,
    }[representation]


def _solve_representation(
    guess: Tensor,
    inputs: SCFInputs,
    config: ConfigSCF,
    *,
    eager: bool = True,
    record_trace: bool = False,
    padded: bool = False,
) -> SCFSolveResult:
    """Run one representation with eager or transform-fixed control."""
    settings, strategy = _prepare_solver(config, guess.dtype)
    mode = config.scp_mode
    padded_state: Callable[[Tensor], Tensor] | None = None
    padded_residual: Callable[[Tensor], Tensor] | None = None
    if mode == labels.SCP_MODE_CHARGE:
        x = pack_charges(
            SCFCharges(
                mono=guess,
                dipole=(
                    None
                    if inputs.charge_layout.dipole_shape is None
                    else guess.new_zeros(inputs.charge_layout.dipole_shape)
                ),
                quad=(
                    None
                    if inputs.charge_layout.quadrupole_shape is None
                    else guess.new_zeros(inputs.charge_layout.quadrupole_shape)
                ),
            ),
            inputs.charge_layout,
        )
        if padded:
            padded_state, padded_residual = _padded_charge_projections(inputs)
        return solve_scf_problem(
            x,
            SCFProblem(inputs, SCFRepresentation.CHARGE),
            settings,
            strategy,
            eager=eager,
            record_trace=record_trace,
            state_projection=padded_state,
            residual_projection=padded_residual,
            project_initial_state=padded,
        )
    charges = SCFCharges(
        mono=guess,
        dipole=(
            None
            if inputs.charge_layout.dipole_shape is None
            else guess.new_zeros(inputs.charge_layout.dipole_shape)
        ),
        quad=(
            None
            if inputs.charge_layout.quadrupole_shape is None
            else guess.new_zeros(inputs.charge_layout.quadrupole_shape)
        ),
    )
    if mode == labels.SCP_MODE_POTENTIAL:
        packed, layout = potential_initial_guess(charges, inputs)
        if padded:
            padded_state, padded_residual = _padded_potential_projections(
                inputs, layout
            )
        return solve_scf_problem(
            packed,
            SCFProblem(inputs, SCFRepresentation.POTENTIAL, layout),
            settings,
            strategy,
            eager=eager,
            record_trace=record_trace,
            state_projection=padded_state,
            residual_projection=padded_residual,
            project_initial_state=padded,
        )
    if mode == labels.SCP_MODE_FOCK:
        fock = fock_initial_guess(charges, inputs)
        if padded:
            matrix_mask = inputs.orbital_mask.unsqueeze(
                -1
            ) & inputs.orbital_mask.unsqueeze(-2)
            padded_state = lambda value: torch.where(
                matrix_mask, value, inputs.integrals.hcore
            )
            problem = SCFProblem(inputs, SCFRepresentation.FOCK)
            padded_residual = problem.project_residual
        return solve_scf_problem(
            fock,
            SCFProblem(inputs, SCFRepresentation.FOCK),
            settings,
            strategy,
            eager=eager,
            record_trace=record_trace,
            state_projection=padded_state,
            residual_projection=padded_residual,
            project_initial_state=padded,
        )
    raise ValueError(f"Unknown convergence target (SCP mode) '{mode}'.")


def _padded_potential_projections(
    inputs: SCFInputs, layout: PotentialLayout
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Mask fictitious orbital/atom potential entries and PADNZ residuals."""
    orbital_mask = inputs.orbital_mask
    atom_mask = inputs.numbers != 0

    def project_state(value: Tensor) -> Tensor:
        potential = unpack_potential(value, layout)
        return pack_potential(
            SCFPotential(
                mono=torch.where(orbital_mask, potential.mono, 0.0),
                dipole=(
                    None
                    if potential.dipole is None
                    else torch.where(
                        atom_mask.unsqueeze(-1), potential.dipole, 0.0
                    )
                ),
                quad=(
                    None
                    if potential.quad is None
                    else torch.where(
                        atom_mask.unsqueeze(-1), potential.quad, 0.0
                    )
                ),
            ),
            layout,
        )

    problem = SCFProblem(
        inputs, SCFRepresentation.POTENTIAL, potential_layout=layout
    )
    return project_state, problem.project_residual


def _padded_charge_projections(
    inputs: SCFInputs,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Mask inactive orbital and atom multipoles in charge-SCP state."""
    orbital_mask = inputs.orbital_mask
    atom_mask = inputs.numbers != 0

    def project_state(value: Tensor) -> Tensor:
        charges = unpack_charges(value, inputs.charge_layout)
        return pack_charges(
            SCFCharges(
                mono=torch.where(orbital_mask, charges.mono, 0.0),
                dipole=(
                    None
                    if charges.dipole is None
                    else torch.where(
                        atom_mask.unsqueeze(-1), charges.dipole, 0.0
                    )
                ),
                quad=(
                    None
                    if charges.quad is None
                    else torch.where(atom_mask.unsqueeze(-1), charges.quad, 0.0)
                ),
            ),
            inputs.charge_layout,
        )

    problem = SCFProblem(inputs, SCFRepresentation.CHARGE)
    return project_state, problem.project_residual


def run_functional_full_eager(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    spin: Tensor | None,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    refocc: Tensor,
    *args: object,
    **kwargs: object,
) -> EagerFullResult:
    """Run the functional FULL calculation or retained batch boundary.

    Tensor-status inspection and warning/error behavior stay outside the
    fixed-trip numerical solver.
    """
    if args or kwargs:
        raise NotImplementedError(
            "Legacy SCF engine positional/keyword overrides are no longer "
            "supported by functional SCF."
        )
    n0, occupation, nel = get_refocc(refocc, chrg, spin, ihelp)
    if not config.requires_iterations:
        route = FullSCFRoute(
            "non_scf", config.scp_mode, config.mixer, reason="non-SCF method"
        )
        legacy = solve_nonselfconsistent(
            numbers,
            interactions,
            cache,
            ihelp,
            config,
            integrals,
            n0,
            occupation,
            **kwargs,
        )
        return EagerFullResult(route, None, legacy)

    guess = get_guess(numbers, positions, chrg, ihelp, config.guess)
    legacy_engine_overrides = bool(args) or bool(kwargs)
    route = resolve_full_scf_route(
        config,
        interactions,
        ihelp,
        numbers,
        integrals,
        legacy_engine_overrides,
    )
    if route.kind == "batch_compat":
        from dxtb._src.scf.unrolling import SelfConsistentFieldFull

        legacy = SelfConsistentFieldFull(
            interactions,
            occupation,
            n0,
            *args,
            numbers=numbers,
            ihelp=ihelp,
            cache=cache,
            integrals=integrals,
            config=config,
            nel=nel,
            **kwargs,
        )(guess)
        return EagerFullResult(route, None, legacy)

    if route.warning is not None:
        OutputHandler.warn(route.warning)
    result = run_prepared_functional_full(
        numbers,
        guess,
        n0,
        occupation,
        nel,
        interactions,
        cache,
        ihelp,
        config,
        integrals,
        route,
    )
    functional = apply_functional_full_policy(result, config, route)
    return EagerFullResult(route, functional, None)


def run_prepared_functional_full(
    numbers: Tensor,
    guess: Tensor,
    n0: Tensor,
    occupation: Tensor,
    nel: Tensor,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    route: FullSCFRoute,
    *,
    core_index: CoreIndexData | None = None,
    core_index_owner: object | None = None,
    eager: bool = True,
) -> SCFSolveResult:
    """Run only functional physics after common SCF preparation is complete."""
    if route.kind != "functional":
        raise ValueError(
            "The prepared functional runner requires a functional route."
        )
    if route.representation != config.scp_mode:
        raise ValueError(
            "The prepared functional route representation does not match "
            "the SCF configuration."
        )
    effective_config = replace(config, mixer=route.effective_mixer)
    inputs = setup_scf_inputs(
        numbers,
        integrals,
        n0,
        occupation,
        nel,
        interactions,
        cache,
        ihelp,
        effective_config,
        core_index=core_index,
        core_index_owner=core_index_owner,
    )
    return _solve_representation(guess, inputs, effective_config, eager=eager)


def run_system_fixed_trip(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    spin: Tensor | None,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    refocc: Tensor,
    *,
    core_index: CoreIndexData | None = None,
    core_index_owner: object | None = None,
) -> FunctionalCoreResult | dict[str, object]:
    """Run fixed trips for supported functional single-System SCF routes."""
    if config.scf_mode == labels.SCF_MODE_EXPERIMENTAL:
        raise NotImplementedError(
            "Experimental single-shot SCF derivatives were not exact and the "
            "legacy implementation has been retired; use FULL or IMPLICIT."
        )
    if not config.requires_iterations:
        n0, occupation, nel = get_refocc(refocc, chrg, spin, ihelp)
        return solve_nonselfconsistent(
            numbers,
            interactions,
            cache,
            ihelp,
            config,
            integrals,
            n0,
            occupation,
        )

    route = resolve_full_scf_route(
        config, interactions, ihelp, numbers, integrals
    )
    if route.kind != "functional":
        raise NotImplementedError(
            "The fixed-trip System API supports functional single-System "
            "SCF only."
        )

    n0, occupation, nel = get_refocc(
        refocc, chrg, spin, ihelp, validate_spin=False
    )
    guess = get_guess(
        numbers, positions, chrg, ihelp, config.guess, validate=False
    )
    result = run_prepared_functional_full(
        numbers,
        guess,
        n0,
        occupation,
        nel,
        interactions,
        cache,
        ihelp,
        config,
        integrals,
        route,
        core_index=core_index,
        core_index_owner=core_index_owner,
        eager=False,
    )
    return functional_core_result(result)


def run_prepared_functional_implicit(
    numbers: Tensor,
    guess: Tensor,
    n0: Tensor,
    occupation: Tensor,
    nel: Tensor,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    *,
    core_index: CoreIndexData | None = None,
    core_index_owner: object | None = None,
) -> SCFSolveResult:
    """Acquire a functional primal root, attach its raw-map IFT, then finalize.

    This entry point is intentionally single-System and eager. It rejects an
    invalid primal before calling the transform-capable root attachment.
    """
    if config.maxiter == 0:
        raise SCFConvergenceError(
            "Implicit physical-root differentiation requires maxiter > 0."
        )
    inputs = setup_scf_inputs(
        numbers,
        integrals,
        n0,
        occupation,
        nel,
        interactions,
        cache,
        ihelp,
        config,
        core_index=core_index,
        core_index_owner=core_index_owner,
    )
    settings, strategy = _prepare_solver(config, guess.dtype)
    charges = SCFCharges(
        mono=guess,
        dipole=(
            None
            if inputs.charge_layout.dipole_shape is None
            else guess.new_zeros(inputs.charge_layout.dipole_shape)
        ),
        quad=(
            None
            if inputs.charge_layout.quadrupole_shape is None
            else guess.new_zeros(inputs.charge_layout.quadrupole_shape)
        ),
    )
    initial_representation = config.scp_mode
    if initial_representation == labels.SCP_MODE_CHARGE:
        state = pack_charges(charges, inputs.charge_layout)
        initial_layout = None
        problem = SCFProblem(inputs, SCFRepresentation.CHARGE)
    elif initial_representation == labels.SCP_MODE_POTENTIAL:
        state, initial_layout = potential_initial_guess(charges, inputs)
        problem = SCFProblem(
            inputs,
            SCFRepresentation.POTENTIAL,
            potential_layout=initial_layout,
        )
    elif initial_representation == labels.SCP_MODE_FOCK:
        state = fock_initial_guess(charges, inputs)
        initial_layout = None
        problem = SCFProblem(inputs, SCFRepresentation.FOCK)
    else:
        raise ValueError(
            "Unknown convergence target (SCP mode) "
            f"'{initial_representation}'."
        )

    outcome = _solve_map(
        state,
        settings,
        problem.map,
        state_projection=(
            None
            if initial_layout is None
            else lambda value: _project_potential_padding(value, initial_layout)
        ),
        eager=True,
        problem=problem,
        strategy=strategy,
    )

    # The strategy may change representation. The selected physical problem is
    # the owner of the final response map and potential layout.
    final_problem = outcome.final_problem
    if final_problem is None:
        raise RuntimeError("The SCF strategy did not return its final problem.")
    final_representation = final_problem.representation
    final_mode = _representation_label(final_representation)
    final_inputs = final_problem.inputs
    final_layout = final_problem.potential_layout

    # Check the selected raw result and the representation's final spectrum
    # before any transform sees this root. The accepted solver input remains
    # outcome.root_x; the paired map result is not promoted to the root.
    if final_representation is SCFRepresentation.CHARGE:
        final_status = outcome.map_result.fermi_status
    elif final_representation is SCFRepresentation.POTENTIAL:
        assert final_layout is not None
        final_state = unpack_potential(
            outcome.map_result.mapped_x, final_layout
        )
        final_status = evaluate_potential(
            final_state, final_inputs
        ).fermi_status
    else:
        final_status = evaluate_hamiltonian(
            outcome.map_result.mapped_x, final_inputs
        ).fermi_status

    # Preserve final-spectrum failure mapping. Historical trajectory status is
    # diagnostic and does not decide whether this physical root is acceptable.
    _raise_failure_code(
        fermi_failure_code(final_status, like=outcome.converged)
    )

    if not bool(outcome.converged.all()):
        raise SCFConvergenceError(
            "Cannot prepare implicit differentiation for an SCF state that "
            "did not pass physical certification."
        )

    adapter, parameters = _SCFMapTensorTree.prepare(
        final_inputs, final_mode, final_layout
    )
    prepared = _prepare_scf_root(
        outcome,
        maxiter=config.maxiter,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        final_valid=fermi_status_ok(final_status, like=outcome.converged),
        raw_map=adapter,
        parameters=parameters,
    )
    attached_root = prepared.attach()
    attached_map_result = adapter.evaluate(attached_root, *parameters)

    if final_representation is SCFRepresentation.CHARGE:
        return finalize_scf(
            attached_map_result,
            final_inputs,
            prepared.iterations,
            prepared.converged,
            prepared.residual,
            prepared.trajectory_status,
            prepared.damping_state,
            root_x=attached_root,
            certification=prepared.certification,
            terminal_failed=torch.zeros_like(prepared.converged),
        )
    if final_representation is SCFRepresentation.POTENTIAL:
        assert final_layout is not None
        return finalize_scf_potential(
            attached_map_result,
            final_inputs,
            final_layout,
            prepared.iterations,
            prepared.converged,
            prepared.residual,
            prepared.trajectory_status,
            prepared.damping_state,
            root_x=attached_root,
            certification=prepared.certification,
            terminal_failed=torch.zeros_like(prepared.converged),
        )
    return finalize_scf_fock(
        attached_map_result,
        final_inputs,
        prepared.iterations,
        prepared.converged,
        prepared.residual,
        prepared.trajectory_status,
        prepared.damping_state,
        root_x=attached_root,
        certification=prepared.certification,
        terminal_failed=torch.zeros_like(prepared.converged),
    )
