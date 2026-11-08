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

from dataclasses import dataclass, replace

import torch

from dxtb import IndexHelper, OutputHandler
from dxtb._src.calculators.config.scf import ConfigSCF, SCFDifferentiation
from dxtb._src.components.interactions import (
    InteractionList,
    InteractionListCache,
)
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

from .anderson import AndersonConverger, setup_anderson_converger_settings
from .broyden import BroydenConverger, setup_broyden_converger_settings
from .certification import SCFInternalCertificationProfile
from .data import (
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFPotential,
    functional_scf_interactions_eligible,
    pack_charges,
    setup_scf_inputs,
)
from .implicit import _prepare_scf_root, _SCFMapTensorTree
from .map import (
    evaluate_hamiltonian,
    evaluate_potential,
    fock_initial_guess,
    potential_initial_guess,
    unpack_potential,
)
from .converger import LinearConverger
from .problem import SCFProblem, SCFVariable
from .cpscf import CPSCFSettings, prepare_response_root
from .solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_INVALID_TEMPERATURE,
    FERMI_FAILURE_NONCONVERGENCE,
    FERMI_FAILURE_NONE,
    FERMI_FAILURE_TOO_MANY_ELECTRONS,
    FixedTripControl,
    SCFSolveResult,
    SCFSolverSettings,
    SCFTrajectoryStatus,
    _project_charge_padding,
    _project_potential_padding,
    _solve_map,
    fermi_failure_code,
    fermi_status_ok,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
    fixed_trip_problem_control,
    setup_solver_settings,
    solve_scf_problem,
)
from .strategy import SCFStage, SCFStrategy

__all__ = [
    "SCFPlan",
    "FunctionalCoreResult",
    "EagerSCFResult",
    "resolve_scf_plan",
    "apply_functional_full_policy",
    "functional_core_result",
    "run_functional_full_eager",
    "fixed_trip_control",
    "prepare_system_fixed_trip",
    "run_prepared_functional_implicit",
    "run_system_fixed_trip",
]


@dataclass(frozen=True)
class SCFPlan:
    """Static decision whether and how the SCF iterates, made before it runs.

    ``kind`` is ``"scf"`` for a self-consistent Method and ``"non_scf"``
    for a non-self-consistent one.
    """

    kind: str
    variable: SCFVariable
    strategy_label: str
    strategy: SCFStrategy | None = None
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
class EagerSCFResult:
    """Eager SCF result together with the plan that produced it."""

    route: SCFPlan
    functional: FunctionalCoreResult | None
    non_scf: dict[str, object] | None


def resolve_scf_plan(
    config: ConfigSCF,
    interactions: InteractionList,
    ihelp: IndexHelper,
    numbers: Tensor | None = None,
    integrals: IntegralMatrices | None = None,
    allow_implicit: bool = False,
) -> SCFPlan:
    """Resolve the SCF plan of a single System, or reject a batch."""
    if (
        config.differentiation is SCFDifferentiation.IMPLICIT
        and not allow_implicit
    ):
        raise ValueError(
            "SCF plan resolution requires unrolled "
            "differentiation."
        )
    if not config.requires_iterations:
        raise ValueError("SCF plan resolution requires an iterative SCF.")
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

    eligible, names = functional_scf_interactions_eligible(interactions)
    if not eligible:
        raise NotImplementedError(
            "Functional SCF does not support interaction extension(s): "
            f"{names}."
        )

    variable = config.initial_variable
    strategy_label = " -> ".join(
        type(item.converger).__name__.removesuffix("Converger").lower()
        for item in config.strategy.stages
    )
    if ihelp.batch_mode != 0 or (numbers is not None and numbers.ndim != 1):
        raise NotImplementedError(
            "The functional SCF solver supports single-System SCF only; "
            "use evaluate_packed_systems or Calculator batch evaluation for batches."
        )
    return SCFPlan(
        "scf", variable, strategy_label, config.strategy
    )


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
    config: ConfigSCF, route: SCFPlan, switched: bool
) -> str:
    """Format the non-convergence warning from configured Python values."""
    damp = config.damp_dynamic_factor if switched else config.damp
    strategy = " -> ".join(
        type(stage.converger).__name__.removesuffix("Converger").lower()
        for stage in config.strategy.stages
    )
    return (
        f"SCF did not certify after {config.maxiter} iterations using "
        f"strategy '{strategy}' (damping {damp})."
    )


def apply_functional_full_policy(
    result: SCFSolveResult,
    config: ConfigSCF,
    route: SCFPlan,
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
    return settings, config.strategy


def _certification_profile(
    config: ConfigSCF,
) -> SCFInternalCertificationProfile:
    """Resolve profile stationarity while honoring explicit root tolerances."""
    stationarity = {
        "standard": 1.0e-5,
        "tight": 1.0e-9,
        "higher_order": 1.0e-12,
    }[config.convergence.value]
    return SCFInternalCertificationProfile(
        name=config.convergence.value,
        fixed_point_l2=float(config.x_atol),
        fixed_point_max=float(config.x_atol_max),
        stationarity_orthogonalized_per_sqrt_orbital=stationarity,
        inherit_solver_root_tolerances=True,
    )


def _variable_problem(
    guess: Tensor, inputs: SCFInputs, config: ConfigSCF
) -> tuple[Tensor, SCFProblem]:
    """Build the initial state and problem of the configured variable."""
    mode = config.initial_variable
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
    if mode == SCFVariable.CHARGE:
        return (
            pack_charges(charges, inputs.charge_layout),
            SCFProblem(inputs, SCFVariable.CHARGE),
        )
    if mode == SCFVariable.POTENTIAL:
        packed, layout = potential_initial_guess(charges, inputs)
        return packed, SCFProblem(inputs, SCFVariable.POTENTIAL, layout)
    if mode == SCFVariable.FOCK:
        return (
            fock_initial_guess(charges, inputs),
            SCFProblem(inputs, SCFVariable.FOCK),
        )
    raise ValueError(f"Unknown SCF variable '{mode}'.")


def _solve_variable(
    guess: Tensor,
    inputs: SCFInputs,
    config: ConfigSCF,
    *,
    eager: bool = True,
    record_trace: bool = False,
) -> SCFSolveResult:
    """Run one variable with eager or transform-fixed control."""
    settings, strategy = _prepare_solver(config, guess.dtype)
    x, problem = _variable_problem(guess, inputs, config)
    return solve_scf_problem(
        x,
        problem,
        settings,
        strategy,
        eager=eager,
        record_trace=record_trace,
        certification_profile=_certification_profile(config),
    )


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
) -> EagerSCFResult:
    """Run the functional FULL calculation or retained batch boundary.

    Tensor-status inspection and warning/error behavior stay outside the
    fixed-trip numerical solver.
    """
    n0, occupation, nel = get_refocc(refocc, chrg, spin, ihelp)
    if not config.requires_iterations:
        route = SCFPlan(
            "non_scf",
            config.initial_variable,
            "none",
            config.strategy,
            reason="non-SCF method",
        )
        non_scf = solve_nonselfconsistent(
            numbers,
            interactions,
            cache,
            ihelp,
            config,
            integrals,
            n0,
            occupation,
        )
        return EagerSCFResult(route, None, non_scf)

    guess = get_guess(numbers, positions, chrg, ihelp, config.guess)
    route = resolve_scf_plan(
        config,
        interactions,
        ihelp,
        numbers,
        integrals,
    )
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
    return EagerSCFResult(route, functional, None)


def _prepared_functional_inputs(
    numbers: Tensor,
    n0: Tensor,
    occupation: Tensor,
    nel: Tensor,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    route: SCFPlan,
    *,
    core_index: IndexHelper | None = None,
    core_index_owner: object | None = None,
) -> SCFInputs:
    """Check an SCF plan and build its immutable SCF inputs."""
    if route.kind != "scf":
        raise ValueError(
            "The prepared SCF runner requires an iterative SCF plan."
        )
    if route.variable != config.initial_variable:
        raise ValueError(
            "The SCF variable of the prepared SCF plan does not match "
            "the SCF configuration."
        )
    return setup_scf_inputs(
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
    route: SCFPlan,
    *,
    core_index: IndexHelper | None = None,
    core_index_owner: object | None = None,
    eager: bool = True,
) -> SCFSolveResult:
    """Run only functional physics after common SCF preparation is complete."""
    inputs = _prepared_functional_inputs(
        numbers,
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
    )
    return _solve_variable(guess, inputs, config, eager=eager)


def prepare_system_fixed_trip(
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
    core_index: IndexHelper | None = None,
    core_index_owner: object | None = None,
) -> tuple[Tensor, SCFProblem]:
    """Build the initial state and problem of a System for the Fixed-trip controller.

    Requires an iterative, unrolled functional single-System SCF.
    """
    if config.differentiation.value == "implicit":
        raise ValueError(
            "The Fixed-trip controller requires differentiation='unrolled'."
        )
    route = resolve_scf_plan(
        config, interactions, ihelp, numbers, integrals
    )
    if route.kind != "scf":
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
    inputs = _prepared_functional_inputs(
        numbers,
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
    )
    return _variable_problem(guess, inputs, config)


def fixed_trip_control(
    x: Tensor, problem: SCFProblem, config: ConfigSCF
) -> FixedTripControl:
    """Return the resumable Fixed-trip controller of a System."""
    settings, strategy = _prepare_solver(config, x.dtype)
    return fixed_trip_problem_control(
        x,
        problem,
        settings,
        strategy,
        certification_profile=_certification_profile(config),
    )


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
    core_index: IndexHelper | None = None,
    core_index_owner: object | None = None,
) -> FunctionalCoreResult | dict[str, object]:
    """Run the Fixed-trip controller for a supported single-System SCF."""
    if config.differentiation.value == "implicit":
        raise ValueError(
            "The Fixed-trip controller requires differentiation='unrolled'."
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

    x, problem = prepare_system_fixed_trip(
        numbers,
        positions,
        chrg,
        spin,
        interactions,
        cache,
        ihelp,
        config,
        integrals,
        refocc,
        core_index=core_index,
        core_index_owner=core_index_owner,
    )
    settings, strategy = _prepare_solver(config, x.dtype)
    result = solve_scf_problem(
        x,
        problem,
        settings,
        strategy,
        eager=False,
        certification_profile=_certification_profile(config),
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
    core_index: IndexHelper | None = None,
    core_index_owner: object | None = None,
    response_settings: CPSCFSettings | None = None,
    strategy_override: SCFStrategy | None = None,
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
    settings, configured_strategy = _prepare_solver(config, guess.dtype)
    strategy = (
        configured_strategy if strategy_override is None else strategy_override
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
    initial_variable = config.initial_variable
    if initial_variable == SCFVariable.CHARGE:
        state = pack_charges(charges, inputs.charge_layout)
        initial_layout = None
        problem = SCFProblem(inputs, SCFVariable.CHARGE)
    elif initial_variable == SCFVariable.POTENTIAL:
        state, initial_layout = potential_initial_guess(charges, inputs)
        problem = SCFProblem(
            inputs,
            SCFVariable.POTENTIAL,
            potential_layout=initial_layout,
        )
    elif initial_variable == SCFVariable.FOCK:
        state = fock_initial_guess(charges, inputs)
        initial_layout = None
        problem = SCFProblem(inputs, SCFVariable.FOCK)
    else:
        raise ValueError(f"Unknown SCF variable '{initial_variable}'.")

    if initial_variable == SCFVariable.CHARGE:
        state_projection: Callable[[Tensor], Tensor] | None = (
            lambda value: _project_charge_padding(value, inputs.charge_layout)
        )
    elif initial_variable == SCFVariable.POTENTIAL:
        assert initial_layout is not None
        state_projection = lambda value: _project_potential_padding(
            value, initial_layout
        )
    else:
        state_projection = None

    outcome = _solve_map(
        state,
        settings,
        problem.map,
        state_projection=state_projection,
        eager=True,
        problem=problem,
        strategy=strategy,
        certification_profile=_certification_profile(config),
    )

    # The strategy may change variable. The selected physical problem is
    # the owner of the final response map and potential layout.
    final_problem = outcome.final_problem
    if final_problem is None:
        raise RuntimeError("The SCF strategy did not return its final problem.")
    final_variable = final_problem.variable
    final_inputs = final_problem.inputs
    final_layout = final_problem.potential_layout

    # Check the selected raw result and the variable's final spectrum
    # before any transform sees this root. The accepted solver input remains
    # outcome.root_x; the paired map result is not promoted to the root.
    if final_variable is SCFVariable.CHARGE:
        final_status = outcome.map_result.fermi_status
    elif final_variable is SCFVariable.POTENTIAL:
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

    response_settings = (
        config.cpscf if response_settings is None else response_settings
    )
    response_root = prepare_response_root(outcome, settings, response_settings)
    final_problem = response_root.problem
    final_variable = final_problem.variable
    final_inputs = final_problem.inputs
    final_layout = final_problem.potential_layout
    adapter = response_root.adapter
    parameters = response_root.parameters
    response_settings = response_root.settings
    response_outcome = response_root.outcome
    prepared = _prepare_scf_root(
        response_outcome,
        maxiter=config.maxiter,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        final_valid=fermi_status_ok(
            response_root.map_result.fermi_status,
            like=response_outcome.converged,
        ),
        raw_map=adapter,
        parameters=parameters,
        response_rtol=response_settings.rtol,
        response_atol=response_settings.atol,
        response_maxiter=response_settings.maxiter,
        independent_map_result=response_root.independent_map_result,
    )
    attached_root = prepared.attach()
    attached_map_result = adapter.evaluate(attached_root, *parameters)

    if final_variable is SCFVariable.CHARGE:
        finalized = finalize_scf(
            attached_map_result,
            final_inputs,
            prepared.iterations,
            prepared.converged,
            prepared.residual,
            prepared.trajectory_status,
            prepared.damping_state,
            root_x=attached_root,
            certification=prepared.certification,
            strategy_diagnostics=outcome.strategy_diagnostics,
            terminal_failed=torch.zeros_like(prepared.converged),
        )
    elif final_variable is SCFVariable.POTENTIAL:
        assert final_layout is not None
        finalized = finalize_scf_potential(
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
            strategy_diagnostics=outcome.strategy_diagnostics,
            terminal_failed=torch.zeros_like(prepared.converged),
        )
    else:
        finalized = finalize_scf_fock(
            attached_map_result,
            final_inputs,
            prepared.iterations,
            prepared.converged,
            prepared.residual,
            prepared.trajectory_status,
            prepared.damping_state,
            root_x=attached_root,
            certification=prepared.certification,
            strategy_diagnostics=outcome.strategy_diagnostics,
            terminal_failed=torch.zeros_like(prepared.converged),
        )

    # Response-coordinate revalidation is separate from the primal solver
    # trajectory. Keep the public solve counters tied to the primal controller.
    return replace(
        finalized,
        iterations=outcome.iterations,
        residual=outcome.residual,
    )
