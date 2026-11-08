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
"""Current-head default-method decision matrix for SCF S13.

Run with one CPU thread and the repository test environment.  The benchmark
uses the same functional physical-map/controller boundary as the accepted
S12 method probe, but applies the current public iteration and convergence
defaults.  It is research tooling and does not change production defaults.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from test.molecules import mols
from test.test_baseline.molecules import SYSTEMS
from test.test_scf import test_functional_root_derivatives as reference
from test.test_scf.samples import samples

import torch

from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.scf.anderson import (
    AndersonConverger,
    AndersonConvergerSettings,
)
from dxtb._src.scf.broyden import BroydenConverger
from dxtb._src.scf.cdiis import CDIISConverger
from dxtb._src.scf.energy import electronic_energy_from_spectral
from dxtb._src.scf.converger import SCFConvergerContext
from dxtb._src.scf.problem import SCFProblem, SCFVariable
from dxtb._src.scf.solver import _solve_map, setup_solver_settings
from dxtb._src.scf.strategy import SCFStage, SCFStrategy

METHODS = ("Anderson", "Broyden", "CDIIS")


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def _prepare_custom_samples() -> None:
    """Add explicitly described charge and geometry variants to test samples."""
    samples["C4H5NCS"] = mols["C4H5NCS"]
    water = SYSTEMS["H2O"]
    numbers = water.numbers.detach().cpu()
    positions = water.positions.detach().cpu()
    samples["S13_H2O_plus"] = {
        "numbers": numbers,
        "positions": positions,
        "charge": 1.0,
        "spin": 1.0,
    }
    samples["S13_H2O_fractional"] = {
        "numbers": numbers,
        "positions": positions,
        "charge": 0.25,
        "spin": 0.0,
    }
    water_issue = torch.tensor(
        [
            [-2.95915993, 1.40005084, 0.24966306],
            [-2.13620310, 1.47957430, -1.38758999],
            [-2.40235213, 2.84218589, 1.24419946],
        ],
        dtype=positions.dtype,
    )
    samples["S13_issue187"] = {
        "numbers": numbers,
        "positions": water_issue,
        "charge": 0.0,
        "spin": 0.0,
    }
    hydrogen = samples["H2"]
    h2_positions = hydrogen["positions"].detach().cpu().clone()
    center = h2_positions.mean(dim=0, keepdim=True)
    samples["S13_stretched_H2"] = {
        **hydrogen,
        "positions": center + 1.8 * (h2_positions - center),
    }


def _case_specifications() -> list[dict[str, object]]:
    """Return 15 cases, later crossed with the three candidate methods."""
    _prepare_custom_samples()
    return [
        {"name": "H2", "method": GFN1_XTB, "temperature": 0.0},
        {"name": "H2O", "method": GFN1_XTB, "temperature": 0.0},
        {"name": "H2O", "method": GFN2_XTB, "temperature": 0.0},
        {"name": "NO2", "method": GFN2_XTB, "temperature": 0.0},
        {"name": "OH-", "method": GFN2_XTB, "temperature": 0.0},
        {"name": "S13_H2O_plus", "method": GFN2_XTB, "temperature": 0.0},
        {"name": "C4H5NCS", "method": GFN2_XTB, "temperature": 0.0},
        {
            "name": "S13_stretched_H2",
            "method": GFN1_XTB,
            "temperature": 0.0,
        },
        {"name": "S13_issue187", "method": GFN1_XTB, "temperature": 0.0},
        {
            "name": "S13_H2O_fractional",
            "method": GFN2_XTB,
            "temperature": 0.0,
        },
        {"name": "H2O", "method": GFN2_XTB, "temperature": 300.0},
        {"name": "NO2", "method": GFN2_XTB, "temperature": 2000.0},
        {
            "name": "H2O",
            "method": GFN2_XTB,
            "temperature": 0.0,
            "initial": "zero",
        },
        {
            "name": "NO2",
            "method": GFN2_XTB,
            "temperature": 0.0,
            "initial": "scaled",
        },
        {
            "name": "S13_issue187",
            "method": GFN1_XTB,
            "temperature": 0.0,
            "initial": "perturbed",
        },
    ]


def _build_method(name: str, config):
    if name == "Anderson":
        configured = config.strategy.stages[0].converger
        if isinstance(configured, AndersonConverger):
            return AndersonConverger(configured.settings)
        return AndersonConverger(AndersonConvergerSettings())
    if name == "Broyden":
        return BroydenConverger()
    if name == "CDIIS":
        return CDIISConverger()
    raise ValueError(f"Unknown method {name!r}.")


def _state_bytes(method, problem, guess: torch.Tensor) -> int:
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
        leaf.numel() * leaf.element_size()
        for leaf in leaves
        if isinstance(leaf, torch.Tensor)
    )


def _run(spec: dict[str, object], method_name: str, maxiter: int) -> dict:
    variable = (
        SCFVariable.FOCK
        if method_name == "CDIIS"
        else SCFVariable.POTENTIAL
    )
    case = reference._case(
        str(spec["name"]),
        method=spec["method"],
        variable=variable,
        converger=(
            AndersonConverger if method_name == "Anderson" else BroydenConverger
        ),
        tolerance=1.0e-5,
        maxiter=maxiter,
        temperature=float(spec["temperature"]),
    )
    # Match the extant public default pair, rather than the helper's single
    # tolerance convenience parameter.
    config = replace(
        case.system.config.scf,
        x_atol=1.0e-4,
        x_atol_max=1.0e-5,
        maxiter=maxiter,
    )
    guess, layout = reference._initial_state(case, case.inputs)
    rep = (
        SCFVariable.FOCK
        if method_name == "CDIIS"
        else SCFVariable.POTENTIAL
    )
    problem = SCFProblem(case.inputs, rep, potential_layout=layout)
    initial_variant = str(spec.get("initial", "physical"))
    if initial_variant == "zero":
        if rep is SCFVariable.FOCK:
            guess = torch.diag(
                torch.linspace(-0.5, 0.5, guess.shape[-1], dtype=guess.dtype)
            )
        else:
            guess = torch.zeros_like(guess)
    elif initial_variant == "scaled":
        guess = 1.5 * guess if rep is SCFVariable.FOCK else 4.0 * guess
    elif initial_variant == "perturbed":
        if rep is SCFVariable.FOCK:
            perturbation = torch.diag(
                torch.linspace(-0.25, 0.25, guess.shape[-1], dtype=guess.dtype)
            )
            guess = guess + perturbation
        else:
            guess = 3.0 * guess
    method = _build_method(method_name, config)
    strategy = SCFStrategy((SCFStage(method),))
    settings = setup_solver_settings(config, guess.dtype)
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
        eager=True,
        problem=problem,
        strategy=strategy,
    )
    wall = time.perf_counter() - started
    spectral = outcome.map_result.spectral
    energy = electronic_energy_from_spectral(spectral, case.inputs).sum()
    certification = outcome.certification
    return {
        "case": str(spec["name"]),
        "initial_variant": initial_variant,
        "method": method_name,
        "tight_binding_method": (
            "GFN1_XTB" if spec["method"] is GFN1_XTB else "GFN2_XTB"
        ),
        "variable": rep.value,
        "temperature_kelvin": float(spec["temperature"]),
        "maxiter": maxiter,
        "x_atol": config.x_atol,
        "x_atol_max": config.x_atol_max,
        "physical_maps": map_count,
        "iterations": int(outcome.iterations),
        "certified": bool(outcome.converged),
        "residual_l2": float(torch.linalg.vector_norm(outcome.residual)),
        "residual_max": float(outcome.residual.abs().amax()),
        "stationarity": (
            None
            if certification is None
            else float(certification.metrics.stationarity_orthogonalized_l2)
        ),
        "energy": float(energy),
        "wall_seconds": wall,
        "method_state_bytes": _state_bytes(method, problem, guess),
        "process_peak_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runs = [
        _run(spec, method, args.maxiter)
        for spec in _case_specifications()
        for method in METHODS
    ]
    result = {
        "source_sha": _git_value("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git_value("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "dtype": "torch.float64",
        "device": "cpu",
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": "expected 1 from caller",
        "num_workers": 1,
        "maxiter": args.maxiter,
        "physical_case_count": len(_case_specifications()),
        "method_system_case_count": len(_case_specifications()) * len(METHODS),
        "methods": list(METHODS),
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
