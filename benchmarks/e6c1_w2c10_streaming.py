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
"""Measure simultaneous, retained-chunk, and streaming W2 implicit gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.baseline.workloads import conformers  # noqa: E402
from benchmarks.e6c1_w2c9_batch_implicit import (  # noqa: E402
    current_rss_mb,
    peak_rss_mb,
    setup_w2_model_and_system,
)
from dxtb._src.calculators.batch_implicit import (  # noqa: E402
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.calculators.batch_streaming import (  # noqa: E402
    stream_batch_value_and_grad,
)


class RSSMonitor:
    """Sample process RSS and attribute peaks to explicit benchmark phases."""

    def __init__(self) -> None:
        self.phase = "setup"
        self.peaks: dict[str, float] = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.025):
            rss = current_rss_mb()
            if rss is not None:
                self.peaks[self.phase] = max(
                    self.peaks.get(self.phase, 0.0), rss
                )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> dict[str, float]:
        self.stop.set()
        self.thread.join()
        return dict(self.peaks)


def run(nconformers: int, mode: str, chunk_size: int, gradient_mode: str):
    rss_monitor = RSSMonitor()
    rss_monitor.start()
    all_numbers, all_positions = conformers("tmpda", 8, sigma=0.1, seed=0)
    numbers = all_numbers[0]
    positions = (
        all_positions[:nconformers]
        .to(dtype=torch.float64)
        .clone()
        .requires_grad_(True)
    )
    model, system = setup_w2_model_and_system(
        numbers, gradient_mode=gradient_mode
    )
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    source_files = (
        ROOT / "src/dxtb/_src/calculators/batch_implicit.py",
        ROOT / "src/dxtb/_src/calculators/batch_streaming.py",
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
    rss_monitor.phase = "root_preparation"
    root_started = time.perf_counter()
    prepared = prepare_converged_root_batch(
        system, positions.detach(), strict=True
    )
    root_seconds = time.perf_counter() - root_started
    rss_monitor.phase = "post_root"
    rss_after_root = current_rss_mb()
    model_parameters = tuple(
        p
        for p in model.par.parameters()
        if p.is_floating_point() and p.requires_grad
    )
    weights = torch.linspace(
        0.5, 1.5, nconformers, dtype=positions.dtype, device=positions.device
    )
    started = time.perf_counter()
    retained = []
    if mode == "simultaneous":
        rss_monitor.phase = "forward"
        forward_started = time.perf_counter()
        result = evaluate_converged_root_batch(
            prepared, positions=positions, strict=True
        )
        forward_seconds = time.perf_counter() - forward_started
        rss_after_forward = current_rss_mb()
        lane_energy = result.energy.reshape(nconformers, -1).sum(-1)
        lane_free_energy = (
            result.fenergy.reshape(nconformers, -1).sum(-1)
            if result.fenergy is not None
            else lane_energy
        )
        objective = (weights * lane_energy).sum()
        backward_started = time.perf_counter()
        rss_monitor.phase = "backward"
        gradients = torch.autograd.grad(
            objective,
            (positions, *model_parameters),
            allow_unused=True,
        )
        energy = lane_energy.detach()
        free_energy = lane_free_energy.detach()
        position_gradient = gradients[0]
        parameter_gradients = gradients[1:]
        backward_seconds = time.perf_counter() - backward_started
    elif mode == "naive-chunks":
        if gradient_mode != "positions-only":
            raise ValueError("naive-chunks currently measures positions only.")
        rss_monitor.phase = "forward"
        for first in range(0, nconformers, chunk_size):
            last = min(first + chunk_size, nconformers)
            chunk = replace(
                prepared,
                positions=prepared.positions[first:last],
                lanes=prepared.lanes[first:last],
                positions_snapshot=prepared.positions_snapshot[first:last],
                input_tensor_versions=prepared.input_tensor_versions[
                    first:last
                ],
                lane_input_snapshots=prepared.lane_input_snapshots[first:last],
                parameter_snapshots=prepared.parameter_snapshots[first:last],
                root_snapshots=prepared.root_snapshots[first:last],
                valid_lane_mask=prepared.valid_lane_mask[first:last],
                all_roots_valid=bool(
                    prepared.valid_lane_mask[first:last].all()
                ),
            )
            retained.append(
                evaluate_converged_root_batch(
                    chunk, positions=positions[first:last], strict=True
                )
            )
        forward_seconds = time.perf_counter() - started
        rss_after_forward = current_rss_mb()
        energy = torch.cat(
            [
                result.energy.reshape(result.energy.shape[0], -1).sum(-1)
                for result in retained
            ]
        )
        free_energy = torch.cat(
            [
                (
                    result.fenergy.reshape(result.fenergy.shape[0], -1).sum(-1)
                    if result.fenergy is not None
                    else result.energy.reshape(result.energy.shape[0], -1).sum(
                        -1
                    )
                )
                for result in retained
            ]
        )
        objective = (weights * energy).sum()
        rss_monitor.phase = "backward"
        position_gradient = torch.autograd.grad(objective, positions)[0]
        parameter_gradients = ()
        backward_seconds = time.perf_counter() - started - forward_seconds
    elif mode == "streaming":
        rss_monitor.phase = "streaming"
        streamed = stream_batch_value_and_grad(
            prepared,
            model,
            positions.detach(),
            weights=weights,
            chunk_size=chunk_size,
            strict=True,
        )
        rss_after_forward = current_rss_mb()
        energy = streamed.energy
        free_energy = streamed.free_energy
        position_gradient = streamed.position_gradient
        parameter_gradients = streamed.model_parameter_gradients
        forward_seconds = None
        backward_seconds = time.perf_counter() - started
    else:
        raise ValueError(f"Unknown execution mode '{mode}'.")

    rss_monitor.phase = "finalize"
    phase_peaks = rss_monitor.close()
    return {
        "mode": mode,
        "nconformers": nconformers,
        "chunk_size": chunk_size,
        "gradient_mode": gradient_mode,
        "iterations": [int(lane.iterations.item()) for lane in prepared.lanes],
        "root_residual_l2": [
            float(lane.raw_residual_l2) for lane in prepared.lanes
        ],
        "energy_eh": energy.detach().cpu().tolist(),
        "free_energy_eh": free_energy.detach().cpu().tolist(),
        "position_gradient_norm": float(
            torch.linalg.vector_norm(position_gradient)
        ),
        "model_gradient_connected": sum(
            g is not None for g in parameter_gradients
        ),
        "timings_s": {
            "root_preparation": root_seconds,
            "forward": forward_seconds,
            "backward_or_stream": backward_seconds,
        },
        "memory_mib": {
            "rss_after_root": rss_after_root,
            "rss_after_forward": rss_after_forward,
            "peak_rss": peak_rss_mb(),
            "phase_peak_rss": phase_peaks,
        },
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nconformers", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("simultaneous", "naive-chunks", "streaming"),
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument(
        "--gradient-mode",
        choices=("positions-only", "positions-and-model"),
        default="positions-only",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    report = run(
        args.nconformers,
        args.mode,
        args.chunk_size,
        args.gradient_mode,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
