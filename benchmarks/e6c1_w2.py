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
"""Measure the E6c1-W2 tmpda conformer workload on one CPU process.

Run one route per process for isolated peak-RSS measurements, for example::

    python benchmarks/e6c1_w2.py --route mapped --nconformers 8 --threads 1

The script writes one JSON object to stdout. It deliberately keeps the
historical workload generator and production calculators as the numerical
owners; it does not provide an alternate energy implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from molecules import mols  # noqa: E402

from dxtb import labels  # noqa: E402
from dxtb import GFN1_XTB, Calculator, ParamModule  # noqa: E402
from dxtb._src.calculators.model import Model  # noqa: E402
from dxtb.config import Config  # noqa: E402

DTYPE = torch.float64
METHOD = GFN1_XTB
OPTIONS = {
    "verbosity": 0,
    "scf_mode": "full",
    "mixer": "anderson",
    "maxiter": 300,
    "f_atol": 1.0e-8,
    "x_atol": 1.0e-8,
    "int_driver": "pytorch",
    "fermi_etemp": 300.0,
}


def conformers(nconformers: int, sigma: float = 0.1, seed: int = 0):
    """Reproduce the T0.8 tmpda geometry generator exactly."""
    numbers = mols["tmpda"]["numbers"]
    positions = mols["tmpda"]["positions"].double()
    generator = torch.Generator().manual_seed(seed)
    displacement = (
        sigma
        * torch.randn(
            (nconformers, *positions.shape), generator=generator
        ).double()
    )
    displacement[0] = 0.0
    return numbers, positions.unsqueeze(0) + displacement


def _rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _package(name: str) -> dict[str, str | None]:
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return {"version": None, "wheel": None}
    direct_url = dist.read_text("direct_url.json")
    return {"version": version(name), "wheel": direct_url}


def _cpu_model() -> str | None:
    """Return the Linux CPU model name when procfs exposes it."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def _available_memory_mb() -> float | None:
    """Return currently available physical memory from procfs."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _build(
    args: argparse.Namespace, numbers: torch.Tensor, options: dict[str, Any]
):
    parameters = ParamModule(METHOD, dtype=DTYPE)
    system = None
    calculator = None
    if args.route != "legacy":
        positions_config = Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            mixer=labels.MIXER_ANDERSON,
            scp_mode=labels.SCP_MODE_CHARGE,
            scf_mode=labels.SCF_MODE_FULL,
            maxiter=args.maxiter,
            force_convergence=False,
            f_atol=1.0e-8,
            x_atol=1.0e-8,
            fermi_etemp=300.0,
        )
        system = Model(par=parameters, config=positions_config).setup(numbers)
    else:
        batched_numbers = (
            numbers.unsqueeze(0).expand(args.nconformers, -1).clone()
        )
        calculator = Calculator(
            batched_numbers,
            parameters,
            opts=dict(options, batch_mode=2),
            dtype=DTYPE,
            device="cpu",
            auto_int_level=True,
        )
    return system, calculator


def _evaluate(
    args: argparse.Namespace,
    system: Any,
    calculator: Calculator,
    positions: torch.Tensor,
) -> Any:
    charges = torch.zeros(positions.shape[0], dtype=DTYPE)
    if args.route == "legacy":
        assert calculator is not None
        return calculator.singlepoint(positions, charges)
    if args.route == "mapped":
        assert system is not None
        return torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    assert system is not None
    return tuple(system.singlepoint(position) for position in positions)


def _total_energy(results: Any, route: str) -> torch.Tensor:
    if route == "serial":
        return torch.stack([result.energy.sum() for result in results]).sum()
    assert results is not None
    return results.energy.sum()


def _physical_summary(result: Any, lane: int, route: str) -> dict[str, Any]:
    if route == "serial":
        value = result[lane]
        index: Any = ...
    else:
        value = result
        index = lane

    def leaf(tensor: torch.Tensor | None) -> Any:
        if tensor is None:
            return None
        item = (
            tensor
            if index is ... or (index is not ... and tensor.ndim == 0)
            else tensor[index]
        )
        return item.detach().cpu().tolist()

    def norm(tensor: torch.Tensor | None) -> dict[str, Any] | None:
        if tensor is None:
            return None
        item = tensor if index is ... or tensor.ndim == 0 else tensor[index]
        item = item.detach()
        return {
            "shape": list(item.shape),
            "trace": (
                float(item.diagonal(dim1=-2, dim2=-1).sum())
                if item.ndim >= 2 and item.shape[-1] == item.shape[-2]
                else None
            ),
            "frobenius_norm": float(torch.linalg.vector_norm(item)),
        }

    charges = value.charges
    classical = []
    for label, energy in value.classical:
        classical.append([label, leaf(energy)])
    return {
        "energy_total": float(
            (value.energy if index is ... else value.energy[index])
            .detach()
            .sum()
        ),
        "energy_atomwise": leaf(value.energy),
        "scf": leaf(value.scf),
        "fenergy": leaf(value.fenergy),
        "classical": classical,
        "charges_mono": None if charges is None else leaf(charges.mono),
        "density": norm(value.density),
        "iterations": leaf(value.iterations),
        "converged": leaf(value.converged),
        "residual": leaf(value.residual),
        "status": (
            None
            if value.status is None
            else {
                "trajectory_failure": leaf(value.status.trajectory_failure),
                "final_failure": leaf(value.status.final_failure),
            }
        ),
    }


def _measure_forward(
    args: argparse.Namespace,
    system: Any,
    calculator: Calculator,
    base_positions: torch.Tensor,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = base_positions.detach().clone().requires_grad_(True)
    start = time.perf_counter()
    results = _evaluate(args, system, calculator, positions)
    energy = _total_energy(results, args.route)
    forward_seconds = time.perf_counter() - start
    return results, energy, positions, torch.tensor(forward_seconds)


def _full_call(
    args: argparse.Namespace,
    system: Any,
    calculator: Calculator,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any, float, float]:
    results, energy, pos, forward_tensor = _measure_forward(
        args, system, calculator, positions
    )
    start = time.perf_counter()
    (gradient,) = torch.autograd.grad(energy, pos)
    backward_seconds = time.perf_counter() - start
    return (
        energy.detach(),
        gradient.detach(),
        results,
        float(forward_tensor),
        backward_seconds,
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    values_tensor = torch.tensor(values, dtype=torch.float64)
    q1, q3 = torch.quantile(
        values_tensor, torch.tensor([0.25, 0.75], dtype=torch.float64)
    )
    return {
        "values_seconds": values,
        "median_seconds": float(torch.median(values_tensor)),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "iqr_seconds": float(q3 - q1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route", choices=("legacy", "serial", "mapped"), required=True
    )
    parser.add_argument("--nconformers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    numbers, positions = conformers(args.nconformers)
    numbers = numbers.to(dtype=torch.int64, device="cpu")
    positions = positions.to(dtype=DTYPE, device="cpu")
    options = dict(OPTIONS, maxiter=args.maxiter)

    process_start_rss = _rss_mb()
    setup_start = time.perf_counter()
    system, calculator = _build(args, numbers, options)
    setup_seconds = time.perf_counter() - setup_start

    # One untimed warm-up verifies the exact selected API and initializes
    # backend/library paths before measured repetitions.
    cold_start = time.perf_counter()
    with torch.enable_grad():
        warm_energy, warm_gradient, warm_result, _, _ = _full_call(
            args, system, calculator, positions
        )
    cold_energy_force_seconds = time.perf_counter() - cold_start
    del warm_result, warm_gradient

    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    parity_result: Any = None
    parity_energy: torch.Tensor | None = None
    parity_gradient: torch.Tensor | None = None

    profile_rows: list[dict[str, Any]] = []
    for repetition in range(args.repeats):
        with torch.enable_grad():
            results, energy, pos, forward = _measure_forward(
                args, system, calculator, positions
            )
            forward_times.append(float(forward))
            start = time.perf_counter()
            (gradient,) = torch.autograd.grad(energy, pos)
            backward = time.perf_counter() - start
            backward_times.append(backward)
            total_times.append(float(forward) + backward)
            if repetition == 0:
                parity_result = results
                parity_energy = energy.detach()
                parity_gradient = gradient.detach()
        profile_rows.append({"repetition": repetition})

    profiler: list[dict[str, Any]] = []
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with torch.enable_grad():
                _full_call(args, system, calculator, positions)
        profiler = [
            {
                "operation": row.key,
                "self_cpu_seconds": row.self_cpu_time_total / 1e6,
            }
            for row in sorted(
                prof.key_averages(),
                key=lambda item: item.self_cpu_time_total,
                reverse=True,
            )[:30]
        ]

    assert parity_energy is not None and parity_gradient is not None
    per_lane = [
        _physical_summary(parity_result, lane, args.route)
        for lane in range(args.nconformers)
    ]
    cpu_affinity = None
    if hasattr(os, "sched_getaffinity"):
        cpu_affinity = sorted(os.sched_getaffinity(0))
    result = {
        "workload": "E6c1-W2",
        "route": args.route,
        "nconformers": args.nconformers,
        "natoms": int(numbers.numel()),
        "sigma_bohr": 0.1,
        "seed": 0,
        "dtype": str(DTYPE),
        "device": "cpu",
        "scf": {
            "method": "GFN1-xTB",
            "int_driver": "pytorch",
            "fermi_etemp_kelvin": 300.0,
            "mode": "full",
            "mixer": "anderson",
            "maxiter": args.maxiter,
            "f_atol": 1.0e-8,
            "x_atol": 1.0e-8,
            "controller": (
                "legacy-eager" if args.route == "legacy" else "fixed-trip"
            ),
        },
        "timing": {
            "setup_seconds": setup_seconds,
            "cold_setup_plus_energy_force_seconds": (
                setup_seconds + cold_energy_force_seconds
            ),
            "forward": _quantiles(forward_times),
            "backward": _quantiles(backward_times),
            "energy_plus_force": _quantiles(total_times),
            "warmup_energy": float(warm_energy),
        },
        "result": {
            "lane_total_energies": [row["energy_total"] for row in per_lane],
            "lane_forces": (-parity_gradient).cpu().tolist(),
            "lane_results": per_lane,
        },
        "memory_mb": {
            "starting_rss": process_start_rss,
            "peak_rss": _rss_mb(),
            "peak_incremental_rss": _rss_mb() - process_start_rss,
            "final_rss": _rss_mb(),
            "ru_maxrss_is_process_lifetime_peak": True,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "cpu_count_logical": os.cpu_count(),
            "cpu_affinity": cpu_affinity,
            "available_memory_mb_at_start": _available_memory_mb(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            "dxtb_git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
            ).stdout.strip(),
            "tad_mctc": _package("tad-mctc"),
            "tad_dftd3": _package("tad-dftd3"),
            "tad_dftd4": _package("tad-dftd4"),
        },
        "profiler_top_self_cpu": profiler,
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
