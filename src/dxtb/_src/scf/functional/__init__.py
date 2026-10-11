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

from .anderson import (
    AndersonSettings,
    AndersonState,
    anderson_mix,
    setup_anderson_solver_settings,
    solve_scf_anderson,
)
from .data import (
    ChargeLayout,
    FermiSettings,
    SCFCharges,
    SCFInputs,
    SCFMapResult,
    SCFPotential,
    pack_charges,
    setup_scf_inputs,
    unpack_charges,
    validate_functional_scf_interactions,
)
from .map import potential_from_charges, potential_to_hamiltonian, scf_map
from .solver import (
    SCFResidual,
    SCFSolveResult,
    SCFSolverSettings,
    finalize_scf,
    interaction_energy_from_charges,
    setup_simple_solver_settings,
    simple_mix,
    functional_result_ok,
    solve_scf_simple,
)

__all__ = [
    "AndersonSettings",
    "AndersonState",
    "ChargeLayout",
    "FermiSettings",
    "SCFCharges",
    "SCFInputs",
    "SCFMapResult",
    "SCFResidual",
    "SCFSolveResult",
    "SCFSolverSettings",
    "SCFPotential",
    "pack_charges",
    "finalize_scf",
    "functional_result_ok",
    "anderson_mix",
    "interaction_energy_from_charges",
    "potential_from_charges",
    "potential_to_hamiltonian",
    "scf_map",
    "setup_scf_inputs",
    "setup_anderson_solver_settings",
    "setup_simple_solver_settings",
    "simple_mix",
    "solve_scf_simple",
    "solve_scf_anderson",
    "unpack_charges",
    "validate_functional_scf_interactions",
]
