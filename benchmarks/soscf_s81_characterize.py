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
"""Characterize fixed-occupation CDIIS to SOSCF on a selected molecule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from test.test_scf.test_functional_map import _prepare_case  # noqa: E402

import torch

from dxtb import GFN2_XTB  # noqa: E402
from dxtb._src.scf.functional import unpack_charges  # noqa: E402
from dxtb._src.scf.functional.cdiis import CDIISMethod  # noqa: E402
from dxtb._src.scf.functional.map import fock_initial_guess  # noqa: E402
from dxtb._src.scf.functional.method import (  # noqa: E402
    SCFMethodContext,
    SCFMethodProposal,
)
from dxtb._src.scf.functional.problem import (  # noqa: E402
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
)
from dxtb._src.scf.functional.solver import (  # noqa: E402
    SCFSolverSettings,
    solve_scf_problem,
)
from dxtb._src.scf.functional.soscf import (  # noqa: E402
    SOSCFMethod,
    SOSCFMethodSettings,
    SOSCFState,
    _occupation_sector_spectra,
)
from dxtb._src.scf.functional.strategy import (  # noqa: E402
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)


@dataclass(frozen=True, eq=False)
class _RecordingSOSCF:
    """Benchmark-only adapter that records method-local diagnostics."""

    events: list[dict[str, float | int | bool]]
    settings: SOSCFMethodSettings = SOSCFMethodSettings()
    required_representation: ClassVar[SCFRepresentation] = (
        SCFRepresentation.FOCK
    )
    supports_fixed_trip: ClassVar[bool] = False

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state,
        context: SCFMethodContext,
    ) -> SCFMethodProposal:
        proposal = SOSCFMethod(self.settings).propose(
            problem, physical_state, method_state, context
        )
        state = proposal.state
        assert isinstance(state, SOSCFState)
        if bool(context.transition_active):
            self.events.append(
                {
                    "gradient_norm": float(state.last_gradient_norm),
                    "step_norm": float(state.last_step_norm),
                    "hvp_count": int(state.last_hvp_count),
                    "microiterations": int(state.last_microiterations),
                    "line_search_trials": int(state.last_line_search_trials),
                    "negative_curvature": bool(state.last_negative_curvature),
                    "gradient_fallback": bool(state.last_gradient_fallback),
                    "level_shift": float(state.level_shift),
                }
            )
        return proposal


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="C4H5NCS")
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=180)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--switch-stationarity", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    case = _prepare_case(
        GFN2_XTB,
        sample_name=args.sample,
        charge=args.charge,
        spin=args.spin,
        dtype=torch.float64,
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    guess = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    events: list[dict[str, float | int | bool]] = []
    strategy = SCFStrategy(
        (
            SCFStage(
                CDIISMethod(),
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=2,
                    stationarity=args.switch_stationarity,
                ),
                initial_damp=1.0,
            ),
            SCFStage(_RecordingSOSCF(events), initial_damp=1.0),
        )
    )
    settings = SCFSolverSettings(
        args.maxiter,
        True,
        1.0,
        args.tolerance,
        args.tolerance,
    )
    started = time.perf_counter()
    outcome = solve_scf_problem(guess, problem, settings, strategy, eager=True)
    elapsed = time.perf_counter() - started
    diagnostics = outcome.diagnostics
    assert diagnostics is not None and diagnostics.certification is not None
    assert diagnostics.strategy is not None
    n_orbitals = int(case.inputs.orbital_mask.sum())
    metrics = diagnostics.certification.result.metrics
    result = {
        "sample": args.sample,
        "charge": args.charge,
        "spin": args.spin,
        "method": "GFN2-xTB",
        "temperature": 0.0,
        "dtype": str(case.positions.dtype),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "threads": torch.get_num_threads(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "maxiter": args.maxiter,
        "tolerance": args.tolerance,
        "switch_stationarity": args.switch_stationarity,
        "converged": bool(outcome.converged),
        "iterations": int(outcome.iterations),
        "stage_iterations": [
            int(item) for item in diagnostics.strategy.stage_iterations
        ],
        "switch_count": int(diagnostics.strategy.switch_count),
        "n_orbitals": n_orbitals,
        "soscf_macroiterations": len(events),
        "soscf_events": events,
        "total_hvp_calls": sum(item["hvp_count"] for item in events),
        "total_microiterations": sum(
            item["microiterations"] for item in events
        ),
        "total_line_search_trials": sum(
            item["line_search_trials"] for item in events
        ),
        "fallback_count": sum(item["gradient_fallback"] for item in events),
        "negative_curvature_events": sum(
            item["negative_curvature"] for item in events
        ),
        "fixed_point_l2": float(metrics.fixed_point.l2),
        "fixed_point_max": float(metrics.fixed_point.max_abs),
        "stationarity": float(
            metrics.stationarity_orthogonalized_l2_per_sqrt_orbital
        ),
        "electron_count_error": float(metrics.electron_count_error),
        "energy": float(outcome.energy.sum()),
        "free_energy": float(outcome.fenergy.sum()),
        "wall_seconds": elapsed,
        "peak_rss_mib": _peak_rss_mib(),
        "ao_matrix_float64_bytes": n_orbitals * n_orbitals * 8,
        "largest_persistent_orbital_hessian": None,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    # The orbital coordinate count is read from the actual final occupation
    # pattern, without constructing a dense Hessian.
    from dxtb._src.scf.functional.orbital import build_orbital_model

    final_model = build_orbital_model(problem, outcome.map_result.spectral)
    result["nrot"] = final_model.nrot
    final_gradient = final_model.physical_gradient_at(
        outcome.root_x.new_zeros(final_model.nrot), outcome.map_result.mapped_x
    )
    final_fock_mo = (
        final_model.coefficients.mT
        @ outcome.map_result.mapped_x
        @ final_model.coefficients
    )
    sector_spectra = _occupation_sector_spectra(final_model, final_fock_mo)
    result["final_orbital_gradient_norm"] = float(
        torch.linalg.vector_norm(final_gradient)
    )
    result["occupation_sector_spectra"] = [
        {
            "occupation": sector,
            "minimum": float(values[0]),
            "maximum": float(values[-1]),
        }
        for sector, values in sector_spectra
    ]
    result["adjacent_sector_gaps"] = [
        float(low_values[0] - high_values[-1])
        for (_, high_values), (_, low_values) in zip(
            sector_spectra, sector_spectra[1:]
        )
    ]
    result["switch_metrics"] = [
        {
            "fixed_point_l2": float(item.fixed_point_l2),
            "fixed_point_max": float(item.fixed_point_max),
            "stationarity": float(item.stationarity),
        }
        for item in diagnostics.strategy.switch_metrics
    ]
    result["worktree_dirty"] = (
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
        != ""
    )
    result["tracked_diff_sha256"] = hashlib.sha256(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    ).hexdigest()
    if args.output is None:
        print(json.dumps(result, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
