# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""Functional SCF building blocks, not yet used by the production solver."""

from .eager import (
    EagerFullResult,
    FullSCFRoute,
    FunctionalCoreResult,
    functional_core_result,
    resolve_full_scf_route,
    run_functional_full_eager,
)

from .anderson import (
    AndersonSettings,
    AndersonState,
    anderson_mix,
    setup_anderson_solver_settings,
    solve_scf_anderson,
    solve_scf_fock_anderson,
    solve_scf_potential_anderson,
)
from .data import (
    ChargeLayout,
    FermiSettings,
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFMapResult,
    SCFPotential,
    SCFSpectralResult,
    pack_charges,
    pack_potential,
    potential_layout,
    setup_scf_inputs,
    unpack_charges,
    unpack_potential,
    validate_functional_scf_interactions,
    functional_scf_interactions_eligible,
)
from .map import (
    evaluate_hamiltonian,
    evaluate_potential,
    fock_initial_guess,
    fock_map,
    potential_from_charges,
    potential_initial_guess,
    potential_map,
    potential_to_hamiltonian,
    scf_map,
    validate_fock_state,
)
from .solver import (
    DampingState,
    DynamicDampingSettings,
    SCFSolverDiagnostics,
    SCFResidual,
    SCFSolveResult,
    SCFSolverSettings,
    SCFTrajectoryStatus,
    accumulate_trajectory_status,
    fermi_status_ok,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
    functional_converged_and_valid,
    functional_subsolves_ok,
    interaction_energy_from_charges,
    setup_simple_solver_settings,
    simple_mix,
    solve_scf_fock_simple,
    solve_scf_potential_simple,
    solve_scf_simple,
    trajectory_status_from_map,
)

__all__ = [
    "AndersonSettings",
    "EagerFullResult",
    "FullSCFRoute",
    "FunctionalCoreResult",
    "DampingState",
    "DynamicDampingSettings",
    "AndersonState",
    "ChargeLayout",
    "FermiSettings",
    "PotentialLayout",
    "SCFCharges",
    "SCFInputs",
    "SCFMapResult",
    "SCFResidual",
    "SCFSolveResult",
    "SCFTrajectoryStatus",
    "SCFSolverSettings",
    "SCFSolverDiagnostics",
    "SCFPotential",
    "SCFSpectralResult",
    "pack_charges",
    "pack_potential",
    "potential_layout",
    "finalize_scf",
    "finalize_scf_fock",
    "functional_subsolves_ok",
    "functional_converged_and_valid",
    "trajectory_status_from_map",
    "accumulate_trajectory_status",
    "fermi_status_ok",
    "anderson_mix",
    "evaluate_hamiltonian",
    "evaluate_potential",
    "interaction_energy_from_charges",
    "potential_from_charges",
    "fock_initial_guess",
    "fock_map",
    "validate_fock_state",
    "potential_initial_guess",
    "potential_map",
    "potential_to_hamiltonian",
    "scf_map",
    "setup_scf_inputs",
    "setup_anderson_solver_settings",
    "setup_simple_solver_settings",
    "simple_mix",
    "solve_scf_simple",
    "solve_scf_potential_simple",
    "solve_scf_potential_anderson",
    "solve_scf_fock_anderson",
    "solve_scf_fock_simple",
    "solve_scf_anderson",
    "unpack_charges",
    "unpack_potential",
    "validate_functional_scf_interactions",
    "functional_scf_interactions_eligible",
    "functional_core_result",
    "resolve_full_scf_route",
    "run_functional_full_eager",
]
