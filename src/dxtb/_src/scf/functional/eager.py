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
from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)

from .data import (
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFPotential,
    functional_scf_interactions_eligible,
    pack_charges,
    setup_scf_inputs,
)
from .solver import (
    SCFSolveResult,
    SCFSolverSettings,
    setup_simple_solver_settings,
    solve_scf_fock_simple,
    solve_scf_potential_simple,
    solve_scf_simple,
)
from .anderson import (
    AndersonSettings,
    setup_anderson_solver_settings,
    solve_scf_anderson,
    solve_scf_fock_anderson,
    solve_scf_potential_anderson,
)
from .map import fock_initial_guess, potential_initial_guess

__all__ = [
    "FullSCFRoute",
    "FunctionalCoreResult",
    "EagerFullResult",
    "resolve_full_scf_route",
    "functional_core_result",
    "run_functional_full_eager",
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
    eligible, names = functional_scf_interactions_eligible(interactions)
    if not eligible:
        return FullSCFRoute(
            "legacy",
            config.scp_mode,
            config.mixer,
            reason=f"unsupported interaction extension(s): {names}",
        )
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


def _check_subsolve_status(result: SCFSolveResult) -> None:
    """Apply legacy-compatible eager error priority to tensor statuses."""
    statuses = []
    if result.trajectory_status is not None:
        status = result.trajectory_status
        statuses.append(
            (
                status.fermi_converged,
                status.electron_count_ok,
                status.invalid_temperature,
                status.too_many_electrons,
            )
        )
    status = result.final_evaluation.fermi_status
    if status is not None:
        statuses.append(
            (
                status.converged,
                status.electron_count_ok,
                status.invalid_temperature,
                status.too_many_electrons,
            )
        )
    if not statuses:
        return
    invalid = any(bool(value[2].any()) for value in statuses)
    if invalid:
        raise ValueError("Electronic Temperature must be non-negative.")
    too_many = any(bool(value[3].any()) for value in statuses)
    if too_many:
        raise ValueError("Number of electrons exceeds the number of orbitals.")
    if any(not bool(value[0].all()) for value in statuses):
        raise RuntimeError("Fermi energy failed to converge.")
    if any(not bool(value[1].all()) for value in statuses):
        raise RuntimeError("Number of electrons changed during Fermi smearing.")


def _solver_settings(config: ConfigSCF, dtype: torch.dtype):
    if config.mixer == labels.MIXER_LINEAR:
        return setup_simple_solver_settings(config, dtype)
    return setup_anderson_solver_settings(config, dtype)


def _solve_representation(
    guess: Tensor,
    inputs: SCFInputs,
    config: ConfigSCF,
) -> SCFSolveResult:
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
            return solve_scf_simple(x, inputs, settings)
        return solve_scf_anderson(x, inputs, settings)
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
            return solve_scf_potential_simple(packed, inputs, layout, settings)
        return solve_scf_potential_anderson(packed, inputs, layout, settings)
    if mode == labels.SCP_MODE_FOCK:
        fock = fock_initial_guess(charges, inputs)
        if isinstance(settings, SCFSolverSettings):
            return solve_scf_fock_simple(fock, inputs, settings)
        return solve_scf_fock_anderson(fock, inputs, settings)
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
) -> EagerFullResult:
    """Run the parallel functional route with legacy eager policy handling.

    This adapter is intentionally not called by production dispatch yet.
    Tensor-status inspection and warning/error behavior live here, outside
    the fixed-trip numerical solver.
    """
    route = resolve_full_scf_route(config, interactions, ihelp, numbers)
    if route.kind == "legacy":
        n0, occupation, nel = get_refocc(refocc, chrg, spin, ihelp)
        if not config.requires_iterations:
            legacy = solve_nonselfconsistent(
                numbers,
                interactions,
                cache,
                ihelp,
                config,
                integrals,
                n0,
                occupation,
            )
            return EagerFullResult(route, None, legacy)
        guess = get_guess(numbers, positions, chrg, ihelp, config.guess)
        if config.scf_mode == labels.SCF_MODE_IMPLICIT:
            from dxtb._src.scf.implicit import SelfConsistentFieldImplicit

            engine_type = SelfConsistentFieldImplicit
        elif config.scf_mode == labels.SCF_MODE_EXPERIMENTAL:
            from dxtb._src.scf.unrolling import SelfConsistentFieldSingleShot

            engine_type = SelfConsistentFieldSingleShot
        else:
            from dxtb._src.scf.unrolling import SelfConsistentFieldFull

            engine_type = SelfConsistentFieldFull

        legacy = engine_type(
            interactions,
            occupation,
            n0,
            numbers=numbers,
            ihelp=ihelp,
            cache=cache,
            integrals=integrals,
            config=config,
            nel=nel,
        )(guess)
        return EagerFullResult(route, None, legacy)

    if route.warning is not None:
        OutputHandler.warn(route.warning)
    n0, occupation, nel = get_refocc(refocc, chrg, spin, ihelp)
    guess = get_guess(numbers, positions, chrg, ihelp, config.guess)
    functional_config = replace(config, mixer=route.effective_mixer)
    inputs = setup_scf_inputs(
        numbers,
        integrals,
        n0,
        occupation,
        nel,
        interactions,
        cache,
        ihelp,
        functional_config,
    )
    result = _solve_representation(guess, inputs, functional_config)
    _check_subsolve_status(result)

    # maxiter=0 is the explicitly accepted one-shot policy, even under
    # force_convergence; only positive budgets apply outer convergence policy.
    if config.maxiter > 0 and not bool(result.converged):
        mixer_label = (
            "Simple"
            if route.effective_mixer == labels.MIXER_LINEAR
            else "Anderson"
        )
        damp = float(result.diagnostics.final_damp)
        message = (
            f"SCF does not converge after {config.maxiter} cycles using "
            f"{mixer_label} mixing with a damping factor of {damp}."
        )
        if config.force_convergence:
            raise SCFConvergenceError(message)
        OutputHandler.warn(message, SCFConvergenceWarning)

    return EagerFullResult(route, functional_core_result(result), None)
