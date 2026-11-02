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
"""Reproducible eager S10 startup strategy comparison."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from test.test_scf.test_functional_map import _prepare_case

import torch

from benchmarks.prototypes.scf_energy_diis import ResearchDIISMethod
from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.scf.functional import unpack_charges
from dxtb._src.scf.functional.anderson import (
    AndersonMethod,
    AndersonMethodSettings,
)
from dxtb._src.scf.functional.broyden import BroydenMethod
from dxtb._src.scf.functional.cdiis import CDIISMethod
from dxtb._src.scf.functional.energy import electronic_energy_from_spectral
from dxtb._src.scf.functional.map import fock_initial_guess
from dxtb._src.scf.functional.method import LinearMethod
from dxtb._src.scf.functional.problem import SCFProblem, SCFRepresentation
from dxtb._src.scf.functional.solver import SCFSolverSettings, _solve_map
from dxtb._src.scf.functional.soscf import SOSCFMethod
from dxtb._src.scf.functional.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="H2O")
    parser.add_argument("--model", choices=("gfn1", "gfn2"), default="gfn2")
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument("--stretch", type=float)
    parser.add_argument("--issue187", action="store_true")
    parser.add_argument(
        "--method",
        choices=(
            "linear",
            "anderson",
            "broyden",
            "cdiis",
            "broyden-cdiis",
            "anderson-cdiis",
            "exact-cdiis",
            "exact-cdiis-soscf",
            "classic-cdiis",
            "quadratic-cdiis",
            "adiis-cdiis",
        ),
        default="exact-cdiis",
    )
    parser.add_argument(
        "--initial", choices=("normal", "zero", "perturbed"), default="normal"
    )
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--switch", type=float, default=1.0e-2)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    dtype = torch.float64
    model = GFN1_XTB if args.model == "gfn1" else GFN2_XTB
    positions = None
    if args.issue187:
        if args.case != "H2O":
            raise ValueError("--issue187 uses the H2O issue geometry.")
        positions = torch.tensor(
            [
                [-2.95915993, 1.40005084, 0.24966306],
                [-2.1362031, 1.4795743, -1.38758999],
                [-2.40235213, 2.84218589, 1.24419946],
            ],
            dtype=dtype,
        )
    if args.stretch is not None:
        if args.case != "H2":
            raise ValueError("--stretch is only defined for H2.")
        positions = torch.tensor(
            [[0.0, 0.0, -0.5 * args.stretch], [0.0, 0.0, 0.5 * args.stretch]],
            dtype=dtype,
        )
    case = _prepare_case(
        model,
        sample_name=args.case,
        charge=args.charge,
        spin=args.spin,
        positions=positions,
        dtype=dtype,
    )
    inputs = case.inputs
    if inputs.fermi_settings is not None and inputs.fermi_settings.smear:
        raise ValueError(
            "This fixed-occupation benchmark requires no active smearing."
        )
    problem = SCFProblem(inputs, SCFRepresentation.FOCK)
    guess = fock_initial_guess(
        unpack_charges(case.x, inputs.charge_layout), inputs
    )
    if args.initial == "zero":
        guess = torch.zeros_like(guess)
    elif args.initial == "perturbed":
        torch.manual_seed(20261031)
        perturbation = torch.randn_like(guess)
        perturbation = 0.5 * (perturbation + perturbation.mT)
        active = (
            inputs.orbital_mask[..., :, None]
            & inputs.orbital_mask[..., None, :]
        )
        guess = guess + torch.where(active, 0.35 * perturbation, 0.0)

    research_instance = None
    if args.method == "linear":
        stages = (SCFStage(LinearMethod(), SCFRepresentation.FOCK),)
    elif args.method == "anderson":
        config = case.system.config.scf
        method_instance = AndersonMethod(
            AndersonMethodSettings(
                config.damp_init,
                config.damp_generations,
                config.damp_soft_start,
                config.damp_diagonal_offset,
            )
        )
        stages = (SCFStage(method_instance, SCFRepresentation.FOCK),)
    elif args.method == "broyden":
        stages = (SCFStage(BroydenMethod(), SCFRepresentation.FOCK),)
    elif args.method == "cdiis":
        stages = (SCFStage(CDIISMethod(), SCFRepresentation.FOCK),)
    elif args.method == "broyden-cdiis":
        stages = (
            SCFStage(
                BroydenMethod(),
                SCFRepresentation.FOCK,
                SCFSwitchPolicy(
                    min_stage_iterations=3,
                    fixed_point_l2=args.switch,
                    max_stage_iterations=30,
                ),
            ),
            SCFStage(CDIISMethod(), SCFRepresentation.FOCK, initial_damp=1.0),
        )
    elif args.method == "anderson-cdiis":
        config = case.system.config.scf
        method_instance = AndersonMethod(
            AndersonMethodSettings(
                config.damp_init,
                config.damp_generations,
                config.damp_soft_start,
                config.damp_diagonal_offset,
            )
        )
        stages = (
            SCFStage(
                method_instance,
                SCFRepresentation.FOCK,
                SCFSwitchPolicy(
                    min_stage_iterations=3,
                    fixed_point_l2=args.switch,
                    max_stage_iterations=30,
                ),
            ),
            SCFStage(CDIISMethod(), SCFRepresentation.FOCK, initial_damp=1.0),
        )
    else:
        use_soscf = args.method == "exact-cdiis-soscf"
        mode = "exact" if use_soscf else args.method.removesuffix("-cdiis")
        research_instance = ResearchDIISMethod(
            mode=mode, capacity=args.capacity
        )
        stages = (
            SCFStage(
                research_instance,
                SCFRepresentation.FOCK,
                SCFSwitchPolicy(
                    min_stage_iterations=3,
                    fixed_point_l2=args.switch,
                    max_stage_iterations=30,
                ),
                initial_damp=1.0,
            ),
            SCFStage(
                CDIISMethod(),
                SCFRepresentation.FOCK,
                switch_policy=(
                    SCFSwitchPolicy(
                        min_stage_iterations=3,
                        fixed_point_l2=args.switch,
                        stationarity=args.switch,
                        max_stage_iterations=20,
                    )
                    if use_soscf
                    else None
                ),
                initial_damp=1.0,
            ),
        )
        if use_soscf:
            stages += (
                SCFStage(
                    SOSCFMethod(),
                    SCFRepresentation.FOCK,
                    initial_damp=1.0,
                ),
            )

    settings = SCFSolverSettings(args.maxiter, False, 0.5, 1.0e-7, 1.0e-7)
    map_diagnostics: list[dict[str, float]] = []
    map_diagnostic_seconds = 0.0
    physical_map_seconds = 0.0

    def recorded_map(value: torch.Tensor):
        nonlocal map_diagnostic_seconds, physical_map_seconds
        map_started = time.perf_counter()
        mapped = problem.map(value)
        physical_map_seconds += time.perf_counter() - map_started
        diagnostic_started = time.perf_counter()
        residual = problem.project_residual(mapped.mapped_x - value)
        energy = electronic_energy_from_spectral(mapped.spectral, inputs).sum()
        map_diagnostics.append(
            {
                "energy": float(energy),
                "fixed_point_l2": float(torch.linalg.vector_norm(residual)),
                "fixed_point_max": float(torch.amax(residual.abs())),
            }
        )
        map_diagnostic_seconds += time.perf_counter() - diagnostic_started
        return mapped

    started = time.perf_counter()
    outcome = _solve_map(
        guess,
        settings,
        recorded_map,
        problem=problem,
        strategy=SCFStrategy(stages),
        eager=True,
    )
    instrumented_elapsed = time.perf_counter() - started
    elapsed = instrumented_elapsed - map_diagnostic_seconds
    final_spectral = outcome.map_result.spectral
    record = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_tree_dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "src/dxtb",
                    "benchmarks/prototypes",
                    "benchmarks/scf_s10_energy_diis.py",
                    "test/test_scf/test_functional_ediis.py",
                ],
                text=True,
            ).strip()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dtype": str(dtype),
        "threads": torch.get_num_threads(),
        "case": args.case,
        "model": args.model,
        "charge": args.charge,
        "spin": args.spin,
        "stretch": args.stretch,
        "issue187_geometry": args.issue187,
        "initial": args.initial,
        "strategy": args.method,
        "capacity": args.capacity,
        "switch_l2": args.switch,
        "iterations": int(outcome.iterations),
        "physical_maps": len(map_diagnostics),
        "certified": bool(outcome.converged),
        "residual_l2": float(outcome.residual[0]),
        "residual_max": float(outcome.residual[1]),
        "energy": float(
            electronic_energy_from_spectral(final_spectral, inputs).sum()
        ),
        "wall_seconds": elapsed,
        "instrumented_wall_seconds": instrumented_elapsed,
        "physical_map_seconds": physical_map_seconds,
        "map_diagnostic_seconds_removed": map_diagnostic_seconds,
        "physical_map_trajectory": map_diagnostics,
        "physical_maps_recorded": len(map_diagnostics),
        "coefficient_optimizer_seconds": sum(
            record["optimizer_seconds"]
            for record in (
                () if research_instance is None else research_instance.records
            )
        ),
        "coefficient_optimizer_evaluations": sum(
            record["optimizer_evaluations"]
            for record in (
                () if research_instance is None else research_instance.records
            )
        ),
        "coefficient_optimization": (
            None if research_instance is None else research_instance.records
        ),
        "stage_iterations": (
            None
            if outcome.strategy_diagnostics is None
            else [
                int(value)
                for value in outcome.strategy_diagnostics.stage_iterations
            ]
        ),
    }
    rendered = json.dumps(record, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
