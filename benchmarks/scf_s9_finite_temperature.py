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
"""Characterize finite-temperature physical Newton--Krylov SCF refinement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from test.test_scf.test_functional_map import _prepare_case  # noqa: E402

import torch

from benchmarks.prototypes.scf_newton_krylov import (  # noqa: E402
    NewtonKrylovConverger,
    NewtonKrylovSettings,
    NewtonKrylovState,
)
from dxtb import GFN1_XTB, GFN2_XTB  # noqa: E402
from dxtb._src.scf.functional.anderson import (  # noqa: E402
    AndersonConverger,
    setup_anderson_converger_settings,
)
from dxtb._src.scf.functional.broyden import BroydenConverger  # noqa: E402
from dxtb._src.scf.functional.cdiis import CDIISConverger  # noqa: E402
from dxtb._src.scf.functional.data import unpack_charges  # noqa: E402
from dxtb._src.scf.functional.energy import (  # noqa: E402
    electronic_energy_from_spectral,
    free_energy_from_spectral,
)
from dxtb._src.scf.functional.map import (  # noqa: E402
    fock_initial_guess,
    potential_initial_guess,
)
from dxtb._src.scf.functional.method import (  # noqa: E402
    SCFConvergerContext,
    SCFConvergerProposal,
)
from dxtb._src.scf.functional.problem import (  # noqa: E402
    SCFPhysicalState,
    SCFProblem,
    SCFVariable,
)
from dxtb._src.scf.functional.solver import (  # noqa: E402
    SCFSolverSettings,
    _solve_map,
)
from dxtb._src.scf.functional.strategy import (  # noqa: E402
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)
from dxtb.scf import strategy_preset  # noqa: E402

CASES = {
    "h2o-300k": (GFN2_XTB, "H2O", 0.0, 0.0, 300.0),
    "no2-2000k": (GFN2_XTB, "NO2", 0.0, 1.0, 2000.0),
    "no2-5000k": (GFN2_XTB, "NO2", 0.0, 1.0, 5000.0),
    "h2o-frac": (GFN1_XTB, "H2O", 0.25, 0.0, 2000.0),
}


@dataclass(frozen=True, eq=False)
class _RecordingNewton:
    """Benchmark-only method wrapper retaining per-proposal diagnostics."""

    events: list[dict[str, float | int | bool]]
    settings: NewtonKrylovSettings = NewtonKrylovSettings()
    supports_fixed_trip: ClassVar[bool] = False

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        converger_state,
        context: SCFConvergerContext,
    ) -> SCFConvergerProposal:
        proposal = NewtonKrylovConverger(self.settings).propose(
            problem, physical_state, converger_state, context
        )
        state = proposal.state
        assert isinstance(state, NewtonKrylovState)
        self.events.append(
            {
                "residual_norm": float(state.last_residual_norm),
                "linear_iterations": int(state.last_linear_iterations),
                "linear_residual": float(state.last_linear_residual),
                "linear_converged": bool(state.last_linear_converged),
                "step_norm": float(state.last_step_norm),
                "step_scale": float(state.last_step_scale),
                "jvps": int(state.last_jvp_count),
                "fixed_point_fallback": bool(state.used_fixed_point_fallback),
            }
        )
        return proposal


def _case_problem(case_name: str, variable: SCFVariable):
    method, sample_name, charge, spin, temperature = CASES[case_name]
    case = _prepare_case(
        method,
        sample_name=sample_name,
        charge=charge,
        spin=spin,
        dtype=torch.float64,
        scf_options={"fermi_etemp": temperature},
    )
    assert case.inputs.fermi_settings is not None
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    if variable in (SCFVariable.CHARGE, SCFVariable.FOCK):
        problem = SCFProblem(case.inputs, variable)
        guess = (
            case.x
            if variable is SCFVariable.CHARGE
            else fock_initial_guess(charges, case.inputs)
        )
    else:
        guess, layout = potential_initial_guess(charges, case.inputs)
        problem = SCFProblem(case.inputs, variable, layout)
    return case, problem, guess


def condition_case(case_name: str) -> dict[str, object]:
    """Solve and measure dense small-system fixed-point conditioning."""
    records = []
    for variable in (
        SCFVariable.CHARGE,
        SCFVariable.POTENTIAL,
        SCFVariable.FOCK,
    ):
        case, problem, guess = _case_problem(case_name, variable)
        settings = SCFSolverSettings(120, False, 0.5, 1.0e-9, 1.0e-9)
        outcome = _solve_map(
            guess,
            settings,
            problem.map,
            problem=problem,
            strategy=SCFStrategy(
                (SCFStage(BroydenConverger(), variable=variable),)
            ),
            eager=True,
        )
        root = outcome.root_x.detach()
        shape = root.shape

        def flat_map(flat: torch.Tensor) -> torch.Tensor:
            state = flat.reshape(shape)
            return problem.project_residual(
                problem.map(state).mapped_x
            ).reshape(-1)

        jacobian = torch.func.jacfwd(flat_map)(root.reshape(-1))
        operator = torch.eye(root.numel(), dtype=root.dtype) - jacobian
        singular_values = torch.linalg.svdvals(operator)
        records.append(
            {
                "variable": variable.value,
                "dimension": int(root.numel()),
                "certified": bool(outcome.converged),
                "root_residual_l2": float(outcome.residual[0]),
                "condition_number": float(
                    singular_values.max() / singular_values.min()
                ),
                "minimum_singular_value": float(singular_values.min()),
                "maximum_singular_value": float(singular_values.max()),
            }
        )
    return {"case": case_name, "conditioning": records}


def _method(case, name: str):
    if name == "broyden":
        return BroydenConverger()
    if name == "anderson":
        return AndersonConverger(
            setup_anderson_converger_settings(
                replace(
                    case.system.config.scf,
                    strategy=strategy_preset("anderson"),
                )
            )
        )
    if name == "cdiis":
        return CDIISConverger()
    raise ValueError(f"Unknown method {name!r}.")


def run_case(
    case_name: str,
    strategy_name: str,
    switch_threshold: float = 1.0e-2,
) -> dict[str, object]:
    variable = (
        SCFVariable.FOCK
        if strategy_name.startswith("cdiis")
        else SCFVariable.POTENTIAL
    )
    case, problem, guess = _case_problem(case_name, variable)
    method_name = strategy_name.split("-")[0]
    first_method = _method(case, method_name)
    events: list[dict[str, float | int | bool]] = []
    if strategy_name.endswith("-newton"):
        final_method = _RecordingNewton(events)
        stages = (
            SCFStage(
                first_method,
                variable=variable,
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=2,
                    fixed_point_l2=switch_threshold,
                ),
            ),
            SCFStage(
                final_method,
                variable=variable,
                initial_damp=1.0,
            ),
        )
    else:
        stages = (SCFStage(first_method, variable=variable),)
    strategy = SCFStrategy(stages)
    settings = SCFSolverSettings(120, False, 0.5, 1.0e-7, 1.0e-7)
    start = time.perf_counter()
    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        problem=problem,
        strategy=strategy,
        eager=True,
    )
    wall = time.perf_counter() - start
    spectral = outcome.map_result.spectral
    occupation = spectral.occupation
    fractional = (occupation > 1.0e-8) & (occupation < 1.0 - 1.0e-8)
    overlap = case.inputs.integrals.overlap
    electron_count = torch.einsum("ij,ji->", spectral.density, overlap)
    energy = electronic_energy_from_spectral(spectral, case.inputs).sum()
    free_energy = free_energy_from_spectral(spectral, case.inputs).sum()
    diagnostics = outcome.strategy_diagnostics
    return {
        "case": case_name,
        "strategy": strategy_name,
        "switch_threshold": (
            switch_threshold if strategy_name.endswith("-newton") else None
        ),
        "variable": variable.value,
        "nao": case.inputs.integrals.hcore.shape[-1],
        "state_dimension": guess.numel(),
        "kt": float(case.inputs.fermi_settings.kt),
        "smear": case.inputs.fermi_settings.smear,
        "fractional_orbital_channels": int(fractional.sum()),
        "minimum_fractional_occupation": (
            float(occupation[fractional].min())
            if bool(fractional.any())
            else None
        ),
        "maximum_fractional_hole": (
            float((1.0 - occupation[fractional]).max())
            if bool(fractional.any())
            else None
        ),
        "iterations": int(outcome.iterations),
        "stage_iterations": (
            []
            if diagnostics is None
            else [int(value) for value in diagnostics.stage_iterations]
        ),
        "switch_count": (
            0 if diagnostics is None else int(diagnostics.switch_count)
        ),
        "certified": bool(outcome.converged),
        "fixed_point_l2": float(outcome.residual[0]),
        "fixed_point_max": float(outcome.residual[1]),
        "stationarity": (
            None
            if outcome.certification is None
            else float(
                outcome.certification.metrics.stationarity_orthogonalized_l2_per_sqrt_orbital
            )
        ),
        "electron_count_error": float(electron_count - case.inputs.nel.sum()),
        "energy": float(energy),
        "free_energy": float(free_energy),
        "newton_events": events,
        "wall_seconds": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--conditioning", action="store_true")
    parser.add_argument("--switch-threshold", type=float, action="append")
    parser.add_argument(
        "--strategy",
        choices=(
            "broyden",
            "broyden-newton",
            "anderson",
            "anderson-newton",
            "cdiis",
            "cdiis-newton",
        ),
        action="append",
    )
    args = parser.parse_args()
    cases = args.case or tuple(CASES)
    strategies = args.strategy or (
        "broyden",
        "broyden-newton",
    )
    thresholds = args.switch_threshold or [1.0e-2]
    records = (
        [condition_case(case_name) for case_name in cases]
        if args.conditioning
        else [
            run_case(case_name, strategy_name, threshold)
            for case_name in cases
            for strategy_name in strategies
            for threshold in (
                thresholds if strategy_name.endswith("-newton") else [1.0e-2]
            )
        ]
    )
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
