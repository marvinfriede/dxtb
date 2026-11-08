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
"""Compare eager single-stage SCF methods under one-thread CPU execution."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
from test.test_scf import test_functional_root_derivatives as root_reference

import torch

from dxtb import GFN2_XTB
from dxtb._src.scf.anderson import (
    AndersonConverger,
    AndersonConvergerSettings,
)
from dxtb._src.scf.broyden import BroydenConverger
from dxtb._src.scf.cdiis import CDIISConverger
from dxtb._src.scf.energy import (
    electronic_energy_from_spectral,
    free_energy_from_spectral,
)
from dxtb._src.scf.converger import LinearConverger, SCFConvergerContext
from dxtb._src.scf.problem import SCFProblem, SCFVariable
from dxtb._src.scf.solver import _solve_map, setup_solver_settings
from dxtb._src.scf.strategy import SCFStage, SCFStrategy


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def _prepare_case(name: str, method_name: str, tolerance: float, maxiter: int):
    if name == "C4H5NCS":
        from test.molecules import mols

        root_reference.samples = {
            **root_reference.samples,
            "C4H5NCS": mols["C4H5NCS"],
        }

    variable = (
        SCFVariable.FOCK
        if method_name == "CDIIS"
        else SCFVariable.POTENTIAL
    )
    converger = {
        "Linear": LinearConverger,
        "Anderson": AndersonConverger,
        "Broyden": BroydenConverger,
        "CDIIS": LinearConverger,
    }[method_name]
    return root_reference._case(
        name,
        method=GFN2_XTB,
        variable=variable,
        converger=converger,
        tolerance=tolerance,
        maxiter=maxiter,
        temperature=0.0,
    )


def _build_method(name: str, config) -> object:
    if name == "Linear":
        return LinearConverger()
    if name == "Anderson":
        configured = config.strategy.stages[0].converger
        return AndersonConverger(
            configured.settings
            if isinstance(configured, AndersonConverger)
            else AndersonConvergerSettings()
        )
    if name == "Broyden":
        return BroydenConverger()
    if name == "CDIIS":
        return CDIISConverger()
    raise ValueError(f"Unknown method {name!r}")


def _state_bytes(method, problem, guess: torch.Tensor) -> int:
    """Estimate persistent state size from one initialized method transition."""
    physical = problem.evaluate(guess)
    context = SCFConvergerContext(
        step=1,
        transition_active=torch.ones((), dtype=torch.bool, device=guess.device),
        damp=guess.new_tensor(0.5),
        proposal_x=physical.map_result.mapped_x,
    )
    state = method.propose(problem, physical, None, context).state
    if state is None:
        return 0
    leaves, _ = torch.utils._pytree.tree_flatten(state)
    return sum(
        value.numel() * value.element_size()
        for value in leaves
        if isinstance(value, torch.Tensor)
    )


def _run(
    name: str,
    method_name: str,
    *,
    tolerance: float,
    maxiter: int,
    fixed_trip: bool,
) -> dict[str, object]:
    case = _prepare_case(name, method_name, tolerance, maxiter)
    guess, layout = root_reference._initial_state(case, case.inputs)
    variable = (
        SCFVariable.FOCK
        if method_name == "CDIIS"
        else SCFVariable.POTENTIAL
    )
    problem = SCFProblem(case.inputs, variable, potential_layout=layout)
    method = _build_method(method_name, case.system.config.scf)
    strategy = SCFStrategy((SCFStage(method),))
    settings = setup_solver_settings(case.system.config.scf, guess.dtype)
    map_count = 0

    def count_map(value: torch.Tensor):
        nonlocal map_count
        map_count += 1
        return problem.map(value)

    started = time.perf_counter()
    outcome = _solve_map(
        guess,
        settings,
        count_map,
        eager=not fixed_trip,
        problem=problem,
        strategy=strategy,
    )
    elapsed = time.perf_counter() - started
    spectral = outcome.map_result.spectral
    electronic_energy = electronic_energy_from_spectral(
        spectral, case.inputs
    ).sum()
    free_energy = free_energy_from_spectral(spectral, case.inputs).sum()
    certification = outcome.certification
    return {
        "case": name,
        "method": method_name,
        "variable": variable.value,
        "controller": "fixed_trip" if fixed_trip else "eager",
        "maxiter": maxiter,
        "requested_tolerance": tolerance,
        "physical_maps": map_count,
        "iterations": int(outcome.iterations),
        "certified": bool(outcome.converged),
        "final_residual_l2": float(torch.linalg.vector_norm(outcome.residual)),
        "final_residual_max": float(outcome.residual.abs().amax()),
        "stationarity": (
            None
            if certification is None
            else float(certification.metrics.stationarity_orthogonalized_l2)
        ),
        "energy": float(electronic_energy),
        "free_energy": float(free_energy),
        "wall_seconds": elapsed,
        "method_state_bytes": _state_bytes(method, problem, guess),
        "process_peak_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["H2O", "NO2"],
        choices=["H2O", "NO2", "C4H5NCS"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Linear", "Anderson", "Broyden", "CDIIS"],
        choices=["Linear", "Anderson", "Broyden", "CDIIS"],
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--fixed-trip", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runs = [
        _run(
            case,
            method,
            tolerance=args.tolerance,
            maxiter=args.maxiter,
            fixed_trip=args.fixed_trip,
        )
        for case in args.cases
        for method in args.methods
    ]
    result = {
        "source_sha": _git_value("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git_value("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "intraop_threads": torch.get_num_threads(),
        "interoperability_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": "set by caller; expected 1",
        "maxiter": args.maxiter,
        "tolerance": args.tolerance,
        "runs": runs,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
