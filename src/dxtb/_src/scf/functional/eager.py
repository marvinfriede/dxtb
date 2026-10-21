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

from .anderson import (
    AndersonSettings,
    _solve_map_anderson,
    setup_anderson_solver_settings,
    solve_scf_anderson,
    solve_scf_fock_anderson,
    solve_scf_potential_anderson,
)
from .core import CoreIndexData
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
    fock_map,
    potential_initial_guess,
    potential_map,
    scf_map,
    unpack_potential,
)
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
    setup_simple_solver_settings,
    solve_scf_fock_simple,
    solve_scf_potential_simple,
    solve_scf_simple,
)

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
    """Resolve full-mode eligibility and compatibility warnings statically."""
    if config.scf_mode != labels.SCF_MODE_FULL:
        return FullSCFRoute(
            "legacy", config.scp_mode, config.mixer, reason="non-full SCF mode"
        )
    if not config.requires_iterations:
        return FullSCFRoute(
            "legacy", config.scp_mode, config.mixer, reason="non-SCF method"
        )
    if ihelp.batch_mode != 0 or (numbers is not None and numbers.ndim != 1):
        return FullSCFRoute(
            "legacy", config.scp_mode, config.mixer, reason="batched System"
        )
    if legacy_engine_overrides:
        return FullSCFRoute(
            "legacy",
            config.scp_mode,
            config.mixer,
            reason="legacy SCF engine options were supplied",
        )
    eligible, names = functional_scf_interactions_eligible(interactions)
    if not eligible:
        return FullSCFRoute(
            "legacy",
            config.scp_mode,
            config.mixer,
            reason=f"unsupported interaction extension(s): {names}",
        )
    if config.fermi.etemp < 0.0:
        return FullSCFRoute(
            "legacy",
            config.scp_mode,
            config.mixer,
            reason="negative electronic temperature retains legacy eager behavior",
        )
    if config.fermi.maxiter < 0:
        return FullSCFRoute(
            "legacy",
            config.scp_mode,
            config.mixer,
            reason="negative Fermi maxiter retains legacy behavior",
        )
    if integrals is not None and (
        integrals.hcore is None or integrals.overlap is None
    ):
        return FullSCFRoute(
            "legacy",
            config.scp_mode,
            config.mixer,
            reason="missing hcore or overlap integral retains legacy behavior",
        )
    if config.maxiter < 0:
        raise ValueError("The functional SCF solver requires maxiter >= 0.")
    if config.mixer == labels.MIXER_BROYDEN:
        message = (
            "Broyden mixer is not implemented for SCF with full gradient "
            "tracking."
        )
        if config.strict:
            raise NotImplementedError(message)
        return FullSCFRoute(
            "functional",
            config.scp_mode,
            labels.MIXER_ANDERSON,
            warning=message + " Using Anderson mixer instead.",
        )
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
    mixer_label = (
        "Simple" if route.effective_mixer == labels.MIXER_LINEAR else "Anderson"
    )
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
            raise SCFConvergenceError(message)
        OutputHandler.warn(message, SCFConvergenceWarning)

    return functional_core_result(result)


def _solver_settings(config: ConfigSCF, dtype: torch.dtype):
    if config.mixer == labels.MIXER_LINEAR:
        return setup_simple_solver_settings(config, dtype)
    return setup_anderson_solver_settings(config, dtype)


def _solve_representation(
    guess: Tensor,
    inputs: SCFInputs,
    config: ConfigSCF,
    *,
    eager: bool = True,
) -> SCFSolveResult:
    """Run one representation with eager or transform-fixed control."""
    settings = _solver_settings(config, guess.dtype)
    mode = config.scp_mode
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
        if isinstance(settings, SCFSolverSettings):
            return solve_scf_simple(x, inputs, settings, eager=eager)
        return solve_scf_anderson(x, inputs, settings, eager=eager)
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
        if isinstance(settings, SCFSolverSettings):
            return solve_scf_potential_simple(
                packed, inputs, layout, settings, eager=eager
            )
        return solve_scf_potential_anderson(
            packed, inputs, layout, settings, eager=eager
        )
    if mode == labels.SCP_MODE_FOCK:
        fock = fock_initial_guess(charges, inputs)
        if isinstance(settings, SCFSolverSettings):
            return solve_scf_fock_simple(fock, inputs, settings, eager=eager)
        return solve_scf_fock_anderson(fock, inputs, settings, eager=eager)
    raise ValueError(f"Unknown convergence target (SCP mode) '{mode}'.")


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
    """Run a side-by-side functional or legacy FULL calculation eagerly.

    Production dispatch uses ``scf.iterator.solve`` for common preparation
    and its prepared functional runner for eligible FULL routes. This wrapper
    remains a direct compatibility-test boundary. Tensor-status inspection
    and warning/error behavior stay outside the fixed-trip numerical solver.
    """
    n0, occupation, nel = get_refocc(refocc, chrg, spin, ihelp)
    if not config.requires_iterations:
        route = resolve_full_scf_route(
            config, interactions, ihelp, numbers, integrals
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
    if route.kind == "legacy":
        if not isinstance(config.scf_mode, int):
            raise ValueError(
                "SCF mode must be an integer within `solve`. This can only "
                "happen if you explicitly change the configuration object."
            )
        if config.scf_mode == labels.SCF_MODE_IMPLICIT:
            from dxtb._src.scf.implicit import SelfConsistentFieldImplicit

            engine_type = SelfConsistentFieldImplicit
        elif config.scf_mode == labels.SCF_MODE_EXPERIMENTAL:
            from dxtb._src.scf.unrolling import SelfConsistentFieldSingleShot

            engine_type = SelfConsistentFieldSingleShot
        elif config.scf_mode == labels.SCF_MODE_FULL:
            from dxtb._src.scf.unrolling import SelfConsistentFieldFull

            engine_type = SelfConsistentFieldFull
        else:
            raise ValueError(f"Unknown SCF mode '{config.scf_mode}'.")

        legacy = engine_type(
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
    """Run fixed trips when eligible; retain legacy-owned System routes."""
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
        eager_result = run_functional_full_eager(
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
        )
        if eager_result.legacy is None:
            raise RuntimeError(
                "Legacy System fallback did not return a result."
            )
        return eager_result.legacy

    n0, occupation, nel = get_refocc(refocc, chrg, spin, ihelp)
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
    settings = _solver_settings(config, guess.dtype)
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
    representation = config.scp_mode
    if representation == labels.SCP_MODE_CHARGE:
        state = pack_charges(charges, inputs.charge_layout)
        layout = None
        map_fn = lambda x: scf_map(x, inputs)
    elif representation == labels.SCP_MODE_POTENTIAL:
        state, layout = potential_initial_guess(charges, inputs)
        map_fn = lambda x: potential_map(x, inputs, layout)
    elif representation == labels.SCP_MODE_FOCK:
        state = fock_initial_guess(charges, inputs)
        layout = None
        map_fn = lambda x: fock_map(x, inputs)
    else:
        raise ValueError(
            f"Unknown convergence target (SCP mode) '{representation}'."
        )

    if isinstance(settings, SCFSolverSettings):
        outcome = _solve_map(
            state,
            settings,
            map_fn,
            state_projection=(
                None
                if layout is None
                else lambda value: _project_potential_padding(value, layout)
            ),
            eager=True,
        )
    else:
        outcome = _solve_map_anderson(
            state,
            settings,
            map_fn,
            state_projection=(
                None
                if layout is None
                else lambda value: _project_potential_padding(value, layout)
            ),
            eager=True,
        )

    # Check the selected raw result and the representation's final spectrum
    # before any transform sees this root. The accepted solver input remains
    # outcome.root_x; the paired map result is not promoted to the root.
    if representation == labels.SCP_MODE_CHARGE:
        final_status = outcome.map_result.fermi_status
    elif representation == labels.SCP_MODE_POTENTIAL:
        assert layout is not None
        final_state = unpack_potential(outcome.map_result.mapped_x, layout)
        final_status = evaluate_potential(final_state, inputs).fermi_status
    else:
        final_status = evaluate_hamiltonian(
            outcome.map_result.mapped_x, inputs
        ).fermi_status

    # Preserve the established eager Fermi failure mapping before rejecting
    # any remaining outer-root nonconvergence in the prepared-root boundary.
    if outcome.trajectory_status is not None:
        _raise_failure_code(outcome.trajectory_status.first_failure)
    _raise_failure_code(
        fermi_failure_code(final_status, like=outcome.converged)
    )

    adapter, parameters = _SCFMapTensorTree.prepare(
        inputs, representation, layout
    )
    prepared = _prepare_scf_root(
        outcome,
        maxiter=config.maxiter,
        final_valid=fermi_status_ok(final_status, like=outcome.converged),
        raw_map=adapter,
        parameters=parameters,
    )
    attached_root = prepared.attach()
    attached_map_result = adapter.evaluate(attached_root, *parameters)

    if representation == labels.SCP_MODE_CHARGE:
        return finalize_scf(
            attached_map_result,
            inputs,
            prepared.iterations,
            prepared.converged,
            prepared.residual,
            prepared.trajectory_status,
            prepared.damping_state,
        )
    if representation == labels.SCP_MODE_POTENTIAL:
        assert layout is not None
        return finalize_scf_potential(
            attached_map_result,
            inputs,
            layout,
            prepared.iterations,
            prepared.converged,
            prepared.residual,
            prepared.trajectory_status,
            prepared.damping_state,
        )
    return finalize_scf_fock(
        attached_map_result,
        inputs,
        prepared.iterations,
        prepared.converged,
        prepared.residual,
        prepared.trajectory_status,
        prepared.damping_state,
    )
