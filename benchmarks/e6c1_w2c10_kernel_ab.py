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
"""Fresh-process A/B measurements for the W2c9 spectral changes."""

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
from dxtb._src.scf.functional.map import scf_map  # noqa: E402
from dxtb._src.scf.functional.solver import (  # noqa: E402
    free_energy_from_spectral,
)


def _metadata() -> dict[str, object]:
    """Return exact source and process metadata for this fresh run."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    return {
        "source_root": str(ROOT),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_dirty": bool(status),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def _construct():
    """Build identical frozen tmpda inputs in each isolated source tree."""
    numbers, position = w2c2.construct_workload(0)
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
        positions=position,
        charge=torch.zeros(1, dtype=torch.float64),
        spin=None,
    )
    return numbers, system, position, w2c2.prepare_call_inputs(case)


def _root(prepared, root_file: Path) -> tuple[torch.Tensor, float]:
    if root_file.exists():
        return (
            torch.load(root_file, map_location="cpu", weights_only=True),
            0.0,
        )
    started = time.perf_counter()
    outcome, _, _ = w2c2.prepare_primal_root(prepared)
    elapsed = time.perf_counter() - started
    root = outcome.root_x.detach().clone()
    root_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(root, root_file)
    return root, elapsed


def _observable(root: torch.Tensor, inputs) -> torch.Tensor:
    mapped = scf_map(root, inputs)
    free_energy = free_energy_from_spectral(mapped.spectral, inputs)
    return mapped.mapped_x.sum() + free_energy.sum()


def _time_call(function, repeats: int) -> tuple[list[float], object]:
    times = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - started)
    return times, result


def run(mode: str, root_file: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _numbers, system, _position, prepared = _construct()
    root, root_preparation_seconds = _root(prepared, root_file)
    inputs = prepared.inputs
    direction = torch.ones_like(root) / max(1, root.numel()) ** 0.5
    scalar = lambda x: _observable(x, inputs)
    timings: dict[str, object] = {}

    if mode in ("all", "no-grad"):

        def primal():
            with torch.no_grad():
                return _observable(root, inputs)

        timings["primal_no_grad"], _ = _time_call(primal, 3)
        with torch.no_grad(), profile(
            activities=[ProfilerActivity.CPU], record_shapes=True
        ) as profiler:
            _observable(root, inputs)
        timings["operation_counts"] = {
            event.key: event.count
            for event in profiler.key_averages()
            if event.key in ("aten::linalg_solve", "aten::mm", "aten::matmul")
        }

    if mode in ("all", "forward", "backward"):

        def forward():
            x = root.detach().clone().requires_grad_(True)
            return scalar(x)

        timings["differentiable_forward"], _ = _time_call(forward, 3)

        def backward():
            x = root.detach().clone().requires_grad_(True)
            return torch.autograd.grad(scalar(x), x)[0]

        timings["backward"], gradient = _time_call(backward, 3)
        timings["gradient_norm"] = float(torch.linalg.vector_norm(gradient))

    if mode in ("all", "jvp"):
        timings["jvp"], tangent = _time_call(
            lambda: torch.func.jvp(scalar, (root,), (direction,))[1], 1
        )
        timings["jvp_norm"] = float(torch.linalg.vector_norm(tangent))

    if mode in ("all", "second"):
        timings["second_directional"], hvp = _time_call(
            lambda: torch.func.jvp(
                torch.func.grad(scalar), (root,), (direction,)
            )[1],
            1,
        )
        timings["hvp_norm"] = float(torch.linalg.vector_norm(hvp))

    return {
        "stage": "w2c10_kernel_ab",
        "mode": mode,
        "dimension": int(inputs.integrals.hcore.shape[-1]),
        "root_x_sha256": hashlib.sha256(
            root.contiguous().numpy().tobytes()
        ).hexdigest(),
        "physical": {
            "method": system.config.method,
            "temperature_kelvin": 300.0,
            "conformer": "tmpda[0]",
        },
        "timings_s": timings,
        "root_preparation_seconds": root_preparation_seconds,
        "peak_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        / 1024.0,
        "environment": _metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "no-grad", "forward", "backward", "jvp", "second"),
        default="all",
    )
    parser.add_argument("--root-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.mode, args.root_file)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
