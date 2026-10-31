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
"""Compare fixed and adaptive detached tmpda primal-root preparation."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks import e6c1_w2c2_implicit as w2c2  # noqa: E402
from dxtb._src.calculators.batch_implicit import (  # noqa: E402
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.scf.functional import spectral as spectral_module  # noqa: E402


def run(adaptive: bool) -> dict[str, object]:
    """Prepare, independently evaluate, and differentiate one tmpda root."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    numbers, position = w2c2.construct_workload(0)
    _model, system = w2c2.setup_model_system(
        numbers,
        maxiter=300,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        gradient_mode="positions-only",
    )
    positions = position.unsqueeze(0).to(dtype=torch.float64)
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    source_files = (
        ROOT / "src/dxtb/_src/calculators/batch_implicit.py",
        ROOT / "src/dxtb/_src/scf/functional/map.py",
        ROOT / "src/dxtb/_src/scf/functional/spectral.py",
    )
    source_hash = hashlib.sha256()
    for source_file in source_files:
        source_hash.update(source_file.name.encode())
        source_hash.update(source_file.read_bytes())
    source_metadata = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "production_source_sha256": source_hash.hexdigest(),
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    solve_count = [0]
    original_dense_solve = spectral_module.dense_solve

    def counted_dense_solve(*args, **kwargs):
        solve_count[0] += 1
        return original_dense_solve(*args, **kwargs)

    spectral_module.dense_solve = counted_dense_solve
    started = time.perf_counter()
    try:
        prepared = prepare_converged_root_batch(
            system,
            positions,
            strict=True,
            eager_primal_fermi=adaptive,
        )
    finally:
        spectral_module.dense_solve = original_dense_solve
    preparation_seconds = time.perf_counter() - started

    differentiable_positions = positions.detach().requires_grad_(True)
    evaluation_started = time.perf_counter()
    result = evaluate_converged_root_batch(
        prepared, positions=differentiable_positions, strict=True
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    gradient = torch.autograd.grad(
        result.energy.sum(), differentiable_positions
    )[0]
    root = prepared.lanes[0].root_x.detach().contiguous()
    return {
        "adaptive": adaptive,
        "conformer": "tmpda[0]",
        "dimension": int(root.shape[-1]),
        "root_sha256": hashlib.sha256(root.cpu().numpy().tobytes()).hexdigest(),
        "iterations": int(prepared.lanes[0].iterations),
        "dense_solve_calls": solve_count[0],
        "root_residual_l2": float(prepared.lanes[0].raw_residual_l2),
        "root_residual_max": float(prepared.lanes[0].raw_residual_max),
        "certified": bool(result.converged.all()),
        "energy_eh": result.energy.detach().cpu().tolist(),
        "position_gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "preparation_seconds": preparation_seconds,
        "independent_evaluation_seconds": evaluation_seconds,
        "peak_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        / 1024.0,
        "environment": {
            **source_metadata,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": platform.processor(),
            "omp": os.environ.get("OMP_NUM_THREADS"),
            "mkl": os.environ.get("MKL_NUM_THREADS"),
            "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.adaptive)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
