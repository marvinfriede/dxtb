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
"""Benchmark: grouped vmap schedules vs. independent eager lanes.

One homogeneous group (one shared ``System``, ``B`` conformers) is evaluated

- ``vmap-static``:   as one mapped group with the full static fixed-trip
  schedule (``maxiter`` map trips; the route used inside transforms),
- ``vmap-adaptive``: as one mapped group advanced in chunks of
  ``--chunk`` trips that stops once no lane is active (the default route of
  eager/autograd calls), and
- ``eager``: lane by lane with the eager single-System route that stops at
  convergence (the route used for singleton groups).

Per case the physical per-lane iterations, the number of raw SCF map
evaluations (one per mapped trip for a group, one per lane iteration for
eager lanes, including initial and final evaluations) and the number of
adaptive chunks (one host check each) are recorded.

Both strategies use the same System and geometries. Modes:

- ``energy``:   no autograd (``torch.no_grad``);
- ``energy_ad``: positions require grad, energy only (tape recorded);
- ``forces``:   energy and position gradient;
- ``param``:    energy and the gradient of a shared trainable Model
  parameter (GFN1 ``element.<heavy>.gam``; setup rebuilt per repeat,
  setup time excluded).

Every (molecule, mode, strategy, size) runs in its own subprocess so peak RSS
is attributable to it; a watchdog kills a case whose resident memory exceeds
``--mem-gib``. On CUDA (``--device cuda``) timings are synchronized and peak
allocated/reserved CUDA memory is recorded.

Usage::

    OMP_NUM_THREADS=1 python benchmarks/vmap_vs_eager_groups.py \
        --output benchmarks/results/vmap-vs-eager-cpu.json
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "benchmarks"))

DTYPE = torch.float64
MOLECULES = ("H2O", "CH4", "SiH4", "C4H5NCS")
SIZES = (1, 2, 4, 8, 16, 32)
MODES = ("energy", "energy_ad", "forces", "param")
STRATEGIES = ("vmap-static", "vmap-adaptive", "eager")


def _setup(
    name: str, device: torch.device, *, differentiable: bool, maxiter: int
):
    from molecules import mols

    from dxtb import GFN1_XTB, ParamModule, labels
    from dxtb._src.calculators.model import Model
    from dxtb.config import Config

    par = ParamModule(GFN1_XTB, dtype=DTYPE, device=device)
    numbers = mols[name]["numbers"].to(device)
    leaf = None
    if differentiable:
        heavy = {1: "H", 6: "C", 7: "N", 8: "O", 14: "Si", 16: "S"}[
            int(numbers.max())
        ]
        par.set_differentiable("element", heavy, "gam")
        leaf = par.get(f"element.{heavy}.gam")
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        maxiter=maxiter,
    )
    model = Model(par=par, config=config, auto_int_level=False)
    return model.setup(numbers), leaf


def _geometries(name: str, size: int, device: torch.device) -> list:
    from molecules import mols

    base = mols[name]["positions"].to(device=device, dtype=DTYPE)
    generator = torch.Generator().manual_seed(11)
    return [
        base
        + 0.03
        * torch.randn(base.shape, generator=generator, dtype=DTYPE).to(device)
        for _ in range(size)
    ]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(
    name: str,
    mode: str,
    strategy: str,
    size: int,
    repeats: int,
    device: str,
    maxiter: int,
    chunk: int,
) -> dict[str, object]:
    from dxtb._src.calculators import lane_evaluation
    from dxtb._src.calculators.lane_evaluation import (
        LaneGroup,
        can_vmap_system,
        evaluate_lane_group,
        evaluate_systems_independently,
    )
    from dxtb._src.scf.problem import SCFProblem

    counts = {"maps": 0, "chunks": 0}
    original_map = SCFProblem.map
    original_advance = lane_evaluation._group_advance

    def counting_map(self, x):
        counts["maps"] += 1
        return original_map(self, x)

    def counting_advance(*args):
        counts["chunks"] += 1
        return original_advance(*args)

    SCFProblem.map = counting_map
    lane_evaluation._group_advance = counting_advance
    lane_evaluation._ADAPTIVE_CHUNK_TRIPS = chunk
    lane_evaluation._requires_static_schedule = lambda: (
        strategy == "vmap-static"
    )

    dev = torch.device(device)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    geometries = _geometries(name, size, dev)
    charges = [torch.tensor(0.0, dtype=DTYPE, device=dev)] * size
    nones = [None] * size

    timings: list[float] = []
    iterations: list[int] = []
    converged: list[bool] = []
    energy_sum = 0.0
    system = None
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(repeats):
        system, leaf = _setup(
            name, dev, differentiable=mode == "param", maxiter=maxiter
        )
        assert can_vmap_system(system)
        positions = [
            g.clone().requires_grad_(mode in ("energy_ad", "forces"))
            for g in geometries
        ]

        _sync(dev)
        counts.update(maps=0, chunks=0)
        start = time.perf_counter()
        context = torch.no_grad() if mode == "energy" else _null()
        with context:
            if strategy != "eager":
                group = LaneGroup(
                    system=system,
                    source_ids=tuple(range(size)),
                    use_vmap=True,
                )
                results = evaluate_lane_group(
                    group, positions, charges, nones, nones, nones
                )
            else:
                results = evaluate_systems_independently(
                    (system,) * size, positions, charges
                )
            energy = torch.stack([r.energy.sum() for r in results]).sum()
            if mode == "forces":
                torch.autograd.grad(energy, positions)
            elif mode == "param":
                torch.autograd.grad(energy, leaf)
        _sync(dev)
        timings.append(time.perf_counter() - start)
        energy_sum = float(energy.detach())
        iterations = [int(r.iterations) for r in results]
        converged = [
            bool(r.converged) if r.converged is not None else True
            for r in results
        ]
        del results, energy, positions

    assert system is not None
    info: dict[str, object] = {
        "molecule": name,
        "mode": mode,
        "strategy": strategy,
        "group_size": size,
        "nat": int(system.numbers.numel()),
        "nao": int(system.ihelp.nao),
        "maxiter": int(system.config.scf.maxiter),
        "iterations": iterations,
        "converged": converged,
        "map_evaluations": counts["maps"],
        "chunks": counts["chunks"],
        "chunk_trips": chunk if strategy == "vmap-adaptive" else None,
        "energy_sum": energy_sum,
        "wall_s": timings,
        "wall_s_min": min(timings),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "rss_before_run_mib": baseline_rss / 1024.0,
    }
    if dev.type == "cuda":
        info["cuda_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(dev) / 1024**2
        )
        info["cuda_peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved(dev) / 1024**2
        )
    return info


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> None:
    from grouped_vmap_batching import _run_with_rss_limit

    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", choices=MOLECULES)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--size", type=int)
    parser.add_argument("--molecules", nargs="+", default=list(MOLECULES))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="SCF maxiter (fixed trips of the vmap route); default as Config",
    )
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument(
        "--chunk",
        type=int,
        nargs="+",
        default=[1],
        help="map trips per adaptive vmap call (vmap-adaptive only)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mem-gib", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.molecule is not None:
        result = run_case(
            args.molecule,
            args.mode,
            args.strategy,
            args.size,
            args.repeats,
            args.device,
            args.maxiter,
            args.chunk[0],
        )
        print(json.dumps(result))
        return

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    report: dict[str, object] = {
        "commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "gpu": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else None
        ),
        "cuda": torch.version.cuda if args.device.startswith("cuda") else None,
        "dtype": str(DTYPE),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mem_limit_gib": args.mem_gib,
        "repeats": args.repeats,
        "maxiter": args.maxiter,
        "chunks": args.chunk,
        "cases": [],
    }

    def save() -> None:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    variants = [
        (strategy, chunk)
        for strategy in args.strategies
        for chunk in (args.chunk if strategy == "vmap-adaptive" else [0])
    ]
    for name in args.molecules:
        for mode in args.modes:
            for strategy, chunk in variants:
                killed_at = None
                for size in args.sizes:
                    label = (
                        f"{strategy}[{chunk}]"
                        if strategy == "vmap-adaptive"
                        else strategy
                    )
                    tag = f"{name}/{mode}/{label}/B={size}"
                    if killed_at is not None:
                        entry = {"status": "skipped", "after_kill": killed_at}
                    else:
                        code, out, err, killed = _run_with_rss_limit(
                            [
                                sys.executable,
                                __file__,
                                "--molecule",
                                name,
                                "--mode",
                                mode,
                                "--strategy",
                                strategy,
                                "--size",
                                str(size),
                                "--repeats",
                                str(args.repeats),
                                "--device",
                                args.device,
                                "--maxiter",
                                str(args.maxiter),
                                "--chunk",
                                str(max(chunk, 1)),
                            ],
                            int(args.mem_gib * 1024**3),
                        )
                        if code == 0:
                            entry = json.loads(out.strip().splitlines()[-1])
                            entry["status"] = "ok"
                            print(
                                f"{tag}: {entry['wall_s_min']:.3f} s, "
                                f"{entry['peak_rss_mib']:.0f} MiB, "
                                f"iter {max(entry['iterations'])}, "
                                f"maps {entry['map_evaluations']}, "
                                f"chunks {entry['chunks']}, "
                                f"converged {all(entry['converged'])}",
                                flush=True,
                            )
                        else:
                            reason = (
                                f"killed above {args.mem_gib} GiB"
                                if killed
                                else (err.strip().splitlines() or ["?"])[-1]
                            )
                            entry = {"status": "failed", "error": reason}
                            if killed:
                                killed_at = size
                            print(f"{tag}: FAILED ({reason})", flush=True)
                    entry.update(
                        molecule=name,
                        mode=mode,
                        strategy=strategy,
                        group_size=size,
                        chunk_trips=(
                            chunk if strategy == "vmap-adaptive" else None
                        ),
                    )
                    report["cases"].append(entry)
                    save()
    save()


if __name__ == "__main__":
    main()
