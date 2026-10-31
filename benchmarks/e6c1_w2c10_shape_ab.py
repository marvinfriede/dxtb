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
"""Physical first-map spectral A/B at representative orbital dimensions."""

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
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(
    os.environ.get("DXTB_SOURCE_ROOT", Path(__file__).resolve().parents[1])
)
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks import e6c1_w2c2_implicit as w2c2  # noqa: E402
from benchmarks.baseline.workloads import water_cluster  # noqa: E402
from dxtb._src.scf.functional.map import scf_map  # noqa: E402
from dxtb._src.scf.functional.solver import (  # noqa: E402
    free_energy_from_spectral,
)


def run(water_count: int) -> dict[str, object]:
    """Measure the same physical first-map point with independent revisions."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    numbers, positions = water_cluster(water_count)
    model, system = w2c2.setup_model_system(
        numbers,
        maxiter=300,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        gradient_mode="positions-only",
    )
    case = w2c2.PhysicalCase(
        model=model,
        system=system,
        numbers=numbers,
        positions=positions,
        charge=torch.zeros((), dtype=torch.float64),
        spin=None,
    )
    prepared = w2c2.prepare_call_inputs(case)
    root = prepared.initial_state.detach()
    inputs = prepared.inputs

    def observable(value: torch.Tensor) -> torch.Tensor:
        mapped = scf_map(value, inputs)
        free_energy = free_energy_from_spectral(mapped.spectral, inputs)
        return mapped.mapped_x.square().sum() + free_energy.sum()

    with torch.no_grad():
        observable(root)
    counts = {}
    with torch.no_grad(), profile(
        activities=[ProfilerActivity.CPU], record_shapes=True
    ) as profiler:
        observable(root)
    for event in profiler.key_averages():
        if event.key in ("aten::linalg_solve", "aten::mm", "aten::matmul"):
            counts[event.key] = event.count

    timings = {}

    def forward():
        return observable(root.detach().requires_grad_(True))

    def backward():
        value = root.detach().requires_grad_(True)
        return torch.autograd.grad(observable(value), value)[0]

    for label, function in (
        ("primal_no_grad", lambda: _primal(observable, root)),
        ("forward", forward),
        ("backward", backward),
    ):
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            function()
            samples.append(time.perf_counter() - started)
        timings[label] = samples

    direction = torch.ones_like(root) / max(1, root.numel()) ** 0.5
    started = time.perf_counter()
    jvp_value = torch.func.jvp(observable, (root,), (direction,))[1]
    timings["jvp"] = [time.perf_counter() - started]
    started = time.perf_counter()
    hvp = torch.func.jvp(torch.func.grad(observable), (root,), (direction,))[1]
    timings["second_directional"] = [time.perf_counter() - started]
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    return {
        "water_count": water_count,
        "dimension": int(inputs.integrals.hcore.shape[-1]),
        "root_sha256": hashlib.sha256(
            root.contiguous().cpu().numpy().tobytes()
        ).hexdigest(),
        "operation_counts": counts,
        "timings_s": timings,
        "observable": float(observable(root)),
        "jvp_norm": float(torch.linalg.vector_norm(jvp_value)),
        "hvp_norm": float(torch.linalg.vector_norm(hvp)),
        "peak_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        / 1024.0,
        "environment": {
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "benchmark_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": platform.processor(),
            "omp": os.environ.get("OMP_NUM_THREADS"),
            "mkl": os.environ.get("MKL_NUM_THREADS"),
            "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }


def _primal(function, root):
    with torch.no_grad():
        return function(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.water_count)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
