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
"""Measure one tmpda implicit response with experimental matrix shifts.

This private diagnostic substitutes the W2c5a matrix-density correction for
one fresh process. It restores the original function on every exit path and
never changes production Calculator routing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import benchmarks.e6c1_w2c2_implicit as w2c2  # noqa: E402
import dxtb  # noqa: E402
from benchmarks.e6c1_w2c5_fermi_shift import _candidate_density  # noqa: E402
from dxtb._src.scf.functional import map as map_module  # noqa: E402

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
DTYPE = torch.float64


def _emit(stream: Any, event: str, **payload: Any) -> None:
    """Write one flushed JSON Lines diagnostic event."""
    stream.write(
        json.dumps(
            {"event": event, "time": time.time(), **payload},
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    stream.flush()


def _read_reference(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load energy/force reference records from a baseline run."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    root = next(
        row for row in records if row["event"] == "physical_root_status"
    )
    response = next(
        row for row in records if row["event"] == "implicit_response_completed"
    )
    complete = next(
        row for row in records if row["event"] == "physical_case_complete"
    )
    return {**root, **complete}, response


def run(output: Path, baseline_path: Path, mode: str) -> None:
    """Run one implementation in the matched physical response harness."""
    if mode not in ("production", "matrix-shift"):
        raise ValueError(f"Unknown implementation mode: {mode}")
    mismatches = {
        key: (os.environ.get(key), value)
        for key, value in THREAD_ENV.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Thread environment mismatch: {mismatches}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Expected one intra-op and inter-op thread")
    expected_source = (ROOT / "src/dxtb/__init__.py").resolve()
    if Path(dxtb.__file__).resolve() != expected_source:
        raise RuntimeError(
            "The benchmark imported a different dxtb source tree"
        )

    numbers, positions = w2c2.construct_workload(0)
    model, system = w2c2.setup_model_system(
        numbers,
        maxiter=300,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        gradient_mode="positions-only",
    )
    positions = positions.detach().clone().requires_grad_(True)
    case = w2c2.PhysicalCase(
        model=model,
        system=system,
        numbers=numbers,
        positions=positions,
        charge=torch.zeros(1, dtype=DTYPE),
        spin=None,
    )
    prepared = w2c2.prepare_call_inputs(case)
    baseline_root, baseline_response = _read_reference(baseline_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    original_density = map_module.fermi_density_matrix
    original_emit = w2c2.emit
    started = time.perf_counter()

    with output.open("w", encoding="utf-8") as stream:

        def emit(event: str, **payload: Any) -> None:
            _emit(
                stream,
                event,
                elapsed_s=time.perf_counter() - started,
                **payload,
            )

        def shifted_density(
            hamiltonian: torch.Tensor,
            nel: torch.Tensor,
            kt: torch.Tensor,
            state: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            kwargs.pop("diff_steps", None)
            return _candidate_density(
                hamiltonian,
                nel,
                kt,
                state,
                use_trace_square=True,
                **kwargs,
            )

        def diagnostic_emit(event: str, **payload: Any) -> None:
            emit(event, **payload)

        if mode == "matrix-shift":
            map_module.fermi_density_matrix = shifted_density
        w2c2.emit = diagnostic_emit
        try:
            emit(
                "environment",
                implementation=mode,
                case="tmpda-conformer-0-frozen-W2",
                python=platform.python_version(),
                torch=torch.__version__,
                source=str(expected_source),
                torch_intra_threads=torch.get_num_threads(),
                torch_inter_threads=torch.get_num_interop_threads(),
                thread_environment=THREAD_ENV,
                scf={
                    "scp_mode": "charge",
                    "scf_mode": "full",
                    "mixer": "anderson",
                    "temperature_kelvin": 300.0,
                    "maxiter": 300,
                    "x_atol": 1.0e-8,
                    "x_atol_max": 1.0e-5,
                    "f_atol": 1.0e-8,
                },
            )
            outcome, map_evaluations, root_seconds = w2c2.prepare_primal_root(
                prepared
            )
            adapter, tensor_parameters, residual, root = (
                w2c2.validate_physical_root(prepared, outcome)
            )
            if not bool(outcome.converged):
                raise RuntimeError("Candidate primal SCF did not converge")
            emit(
                "candidate_root_validated",
                iterations=int(outcome.iterations.detach()),
                map_evaluations=map_evaluations,
                residual_l2=float(torch.linalg.vector_norm(residual)),
                residual_max=float(residual.abs().max()),
                root_seconds=root_seconds,
                root_finite=bool(torch.isfinite(outcome.root_x).all()),
            )

            finalization_started = time.perf_counter()
            implicit_root = w2c2.attach_implicit_root(root)
            mapped = adapter.evaluate(implicit_root, *tensor_parameters)
            result, _ = w2c2.finalize_physical_result(prepared, outcome, mapped)
            finalization_seconds = time.perf_counter() - finalization_started
            energy = result.energy.sum()
            free_energy = (
                result.fenergy.sum()
                if result.fenergy is not None
                else energy.new_zeros(())
            )
            emit(
                "candidate_finalized",
                energy=float(energy.detach()),
                free_energy=float(free_energy.detach()),
                density_finite=bool(torch.isfinite(result.density).all()),
                finalization_elapsed_s=finalization_seconds,
            )

            _, gradients = w2c2.differentiate_positions(
                result,
                positions,
                (),
                observable="energy",
            )
            gradient = gradients[0]
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Position response is nonfinite")
            candidate_energy = float(energy.detach())
            candidate_free_energy = float(free_energy.detach())
            torch.testing.assert_close(
                energy.detach(),
                torch.tensor(baseline_root["energy_sum"], dtype=DTYPE),
                atol=2.0e-9,
                rtol=2.0e-9,
                msg="candidate physical energy versus accepted production reference",
            )
            torch.testing.assert_close(
                free_energy.detach(),
                torch.tensor(baseline_root["free_energy_sum"], dtype=DTYPE),
                atol=2.0e-9,
                rtol=2.0e-9,
                msg="candidate Fermi free energy versus accepted production reference",
            )
            reference_gradient = torch.tensor(
                baseline_response["position_gradient"], dtype=DTYPE
            ).reshape_as(gradient)
            torch.testing.assert_close(
                gradient,
                reference_gradient,
                atol=2.0e-8,
                rtol=2.0e-8,
                msg="candidate energy-only position gradient versus production reference",
            )
            emit(
                "matched_implementation_completed",
                implementation=mode,
                energy=candidate_energy,
                free_energy=candidate_free_energy,
                energy_abs_difference=abs(
                    candidate_energy - baseline_root["energy_sum"]
                ),
                free_energy_abs_difference=abs(
                    candidate_free_energy - baseline_root["free_energy_sum"]
                ),
                gradient_max_abs_difference=float(
                    (gradient - reference_gradient).abs().max()
                ),
                gradient_norm=float(torch.linalg.vector_norm(gradient)),
                baseline_gradient_norm=baseline_response["gradient_norm"],
                baseline_response_seconds=baseline_response["duration_s"],
                process_peak_rss_mib=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024.0,
                total_elapsed_s=time.perf_counter() - started,
            )
        finally:
            map_module.fermi_density_matrix = original_density
            w2c2.emit = original_emit


def main() -> None:
    """Run the bounded private response comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("production", "matrix-shift"),
        required=True,
    )
    arguments = parser.parse_args()
    run(arguments.output, arguments.baseline, arguments.mode)


if __name__ == "__main__":
    main()
