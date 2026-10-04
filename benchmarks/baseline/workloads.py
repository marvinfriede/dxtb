# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""
Performance and memory baseline (T0.8 in ``docs/plan/01-T0-baseline.md``).

Workloads with fixed inputs:

==== ========================================================================
W1   one large molecule (water cluster, 510 atoms): energy and forces
W2   conformer ensemble (tmpda, 69 atoms, displaced copies): energy and
     forces with ``batch_mode=2`` and with a plain loop
W3   training-like step: padded batch of 32 mixed molecules; energy, forces
     and the gradient of a loss with respect to all parameters
W4   Hessian of a medium molecule (LYS_xao, 33 atoms)
W5   hyperpolarizability of a small molecule (glycine)
==== ========================================================================

Usage (repository root)::

    python benchmarks/baseline/workloads.py W1 --method gfn1 --driver pytorch
    python benchmarks/baseline/workloads.py all --quick   # small sizes, smoke run

Each run records the wall time of the stages (dxtb timer), the derivative
passes, the top operations of ``torch.profiler`` (optional, ``--profile``)
and the peak memory (resident set size; CUDA allocator on GPU). Results are
appended to ``benchmarks/baseline/results/<host>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import socket
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import torch
from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols
from tad_mctc.units import AA2AU

import dxtb
from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule
from dxtb.components.field import new_efield

RESULTS = Path(__file__).parent / "results"
PARAMS = {"gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
OPTS = {
    "verbosity": 0,
    "scf_mode": "full",
    "mixer": "anderson",
    "maxiter": 300,
    "f_atol": 1e-8,
    "x_atol": 1e-8,
}


###############################################################################
# inputs


def water_cluster(
    nmol: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Water molecules on a cubic grid (3.1 A) with random orientations."""
    gen = torch.Generator().manual_seed(seed)
    w = mols["H2O"]
    wpos = w["positions"].double()
    wpos = wpos - wpos.mean(0)
    side = math.ceil(nmol ** (1 / 3))
    numbers, positions = [], []
    for k in range(nmol):
        i, j, l = k % side, (k // side) % side, k // side**2
        q, _ = torch.linalg.qr(torch.randn((3, 3), generator=gen).double())
        positions.append(
            wpos @ q.T + torch.tensor([i, j, l]).double() * 3.1 * AA2AU
        )
        numbers.append(w["numbers"])
    return torch.cat(numbers), torch.cat(positions)


def conformers(name: str, n: int, sigma: float = 0.1, seed: int = 0):
    numbers = mols[name]["numbers"]
    positions = mols[name]["positions"].double()
    gen = torch.Generator().manual_seed(seed)
    disp = sigma * torch.randn((n, *positions.shape), generator=gen).double()
    disp[0] = 0.0
    return numbers.unsqueeze(0).expand(n, -1).clone(), positions + disp


TRAINING_SET = [
    "H2O",
    "CH4",
    "NH3",
    "CO2",
    "SiH4",
    "LiH",
    "C4H5NCS",
    "NO2",
    "MB16_43_01",
    "MB16_43_02",
    "MB16_43_03",
    "MB16_43_07",
    "MB16_43_08",
    "PbH4-BiH3",
    "C6H5I-CH3SH",
    "LYS_xao",
    "ZnOOH-",
    "S2",
    "SCl",
    "finch",
    "br2nh3",
    "br2och2",
    "br2nh2o",
    "C2H4F+",
    "NH3-dimer",
    "Li2",
    "HC",
    "HLi",
    "Ag2Cl22-",
    "Al3+Ar6",
    "HHe",
    "tmpda",
]


###############################################################################
# measurement


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Recorder:
    def __init__(self, profile: bool) -> None:
        self.stages: dict[str, float] = {}
        self.profile = profile
        self.top_ops: list[tuple[str, float]] = []

    def run(self, label: str, fn: Callable[[], Any]) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stages[label] = time.perf_counter() - start
        return out

    def profiled(self, fn: Callable[[], Any]) -> Any:
        if not self.profile:
            return fn()
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        with profile(activities=acts) as prof:
            out = fn()
        ops = defaultdict(float)
        for ev in prof.key_averages():
            ops[ev.key] += ev.self_cpu_time_total / 1e6
        self.top_ops = sorted(ops.items(), key=lambda x: -x[1])[:25]
        return out


def _dxtb_stages() -> dict[str, float]:
    times = dxtb.timer.get_times()
    return {k: round(v["value"], 4) for k, v in times.items()}


###############################################################################
# workloads


def w1(args, rec: Recorder) -> dict:
    nmol = 20 if args.quick else 170
    numbers, positions = water_cluster(nmol)
    dd = {"dtype": torch.float64, "device": args.device}
    numbers, positions = numbers.to(args.device), positions.to(**dd)

    calc = rec.run(
        "setup",
        lambda: Calculator(
            numbers,
            PARAMS[args.method],
            opts=dict(OPTS, int_driver=args.driver),
            **dd,
        ),
    )
    pos = positions.clone().requires_grad_(True)
    energy = rec.run("energy", lambda: rec.profiled(lambda: calc.energy(pos)))
    rec.run("forces (backward)", lambda: torch.autograd.grad(energy, pos))
    return {"natoms": int(numbers.numel())}


def w2(args, rec: Recorder) -> dict:
    n = 8 if args.quick else args.nconf
    numbers, positions = conformers("tmpda", n)
    dd = {"dtype": torch.float64, "device": args.device}
    numbers, positions = numbers.to(args.device), positions.to(**dd)
    opts = dict(OPTS, int_driver=args.driver)

    def batched():
        calc = Calculator(
            numbers, PARAMS[args.method], opts=dict(opts, batch_mode=2), **dd
        )
        pos = positions.clone().requires_grad_(True)
        e = calc.energy(pos, torch.zeros(n, **dd))
        return torch.autograd.grad(e.sum(), pos)[0]

    def loop():
        calc = Calculator(numbers[0], PARAMS[args.method], opts=opts, **dd)
        out = []
        for p in positions:
            pos = p.clone().requires_grad_(True)
            e = Calculator(
                numbers[0], PARAMS[args.method], opts=opts, **dd
            ).energy(pos)
            out.append(torch.autograd.grad(e, pos)[0])
        del calc
        return torch.stack(out)

    gb = rec.run("batch_mode=2 (energy+forces)", lambda: rec.profiled(batched))
    gl = rec.run("loop (energy+forces)", loop)
    return {
        "nconformers": n,
        "natoms": int(numbers.shape[-1]),
        "max_diff_batch_vs_loop": float((gb - gl).abs().max()),
    }


def w3(args, rec: Recorder) -> dict:
    names = TRAINING_SET[: 8 if args.quick else args.nsys]
    dd = {"dtype": torch.float64, "device": args.device}
    numbers = pack([mols[n]["numbers"] for n in names]).to(args.device)
    positions = pack([mols[n]["positions"].double() for n in names]).to(**dd)
    charges = torch.tensor(
        [
            (
                -1.0
                if n.endswith("-") and not n.endswith("2-")
                else (
                    -2.0
                    if n.endswith("2-")
                    else 1.0 if n.endswith("+") else 3.0 if "Al3+" in n else 0.0
                )
            )
            for n in names
        ],
        **dd,
    )

    par = ParamModule(PARAMS[args.method], **dd)
    leaves = [p for p in par.parameters() if p.is_floating_point()]
    for p in leaves:
        p.requires_grad_(True)

    calc = rec.run(
        "setup",
        lambda: Calculator(
            numbers,
            par,
            opts=dict(OPTS, int_driver=args.driver, batch_mode=1),
            **dd,
        ),
    )
    pos = positions.clone().requires_grad_(True)

    def forward():
        e = calc.energy(pos, charges)
        (g,) = torch.autograd.grad(e.sum(), pos, create_graph=True)
        return e, g

    e, g = rec.run(
        "energy+forces (create_graph)", lambda: rec.profiled(forward)
    )
    loss = (e**2).sum() + (g**2).sum()
    grads = rec.run(
        "loss backward (parameters)",
        lambda: torch.autograd.grad(loss, leaves, allow_unused=True),
    )
    nz = sum(1 for x in grads if x is not None and bool((x != 0).any()))
    return {
        "nsystems": len(names),
        "max_atoms": int(numbers.shape[-1]),
        "parameter_leaves": len(leaves),
        "leaves_with_gradient": nz,
    }


def w4(args, rec: Recorder) -> dict:
    name = "H2O" if args.quick else "LYS_xao"
    dd = {"dtype": torch.float64, "device": args.device}
    numbers = mols[name]["numbers"].to(args.device)
    positions = mols[name]["positions"].to(**dd)
    calc = Calculator(
        numbers,
        PARAMS[args.method],
        opts=dict(OPTS, int_driver=args.driver),
        **dd,
    )
    pos = positions.clone().requires_grad_(True)
    rec.run(
        "hessian (autograd)", lambda: rec.profiled(lambda: calc.hessian(pos))
    )
    return {"natoms": int(numbers.numel())}


def w5(args, rec: Recorder) -> dict:
    name = "H2O" if args.quick else "glycine"
    dd = {"dtype": torch.float64, "device": args.device}
    if name == "glycine":
        import sys

        sys.path.insert(0, str(Path(__file__).parents[2]))
        from test.test_baseline.molecules import get_system

        s = get_system("glycine")
        numbers, positions = s.numbers, s.positions
    else:
        numbers, positions = (
            mols[name]["numbers"],
            mols[name]["positions"].double(),
        )
    numbers, positions = numbers.to(args.device), positions.to(**dd)
    field = torch.zeros(3, **dd, requires_grad=True)
    calc = Calculator(
        numbers,
        PARAMS[args.method],
        interaction=[new_efield(field)],
        opts=dict(OPTS, int_driver=args.driver),
        **dd,
    )
    rec.run(
        "hyperpolarizability (autograd)",
        lambda: rec.profiled(lambda: calc.hyperpolarizability(positions)),
    )
    return {"natoms": int(numbers.numel())}


WORKLOADS = {"W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5}


def main() -> None:
    parser = argparse.ArgumentParser(description="dxtb performance baseline")
    parser.add_argument("workload", choices=[*WORKLOADS, "all"])
    parser.add_argument("--method", default="gfn1", choices=list(PARAMS))
    parser.add_argument(
        "--driver", default="pytorch", choices=["pytorch", "libcint"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nconf", type=int, default=100)
    parser.add_argument("--nsys", type=int, default=len(TRAINING_SET))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    dxtb.timer.enable()
    names = list(WORKLOADS) if args.workload == "all" else [args.workload]
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{socket.gethostname()}.json"
    data = json.loads(path.read_text()) if path.exists() else []

    for name in names:
        dxtb.timer.reset()
        rec = Recorder(args.profile)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        info = WORKLOADS[name](args, rec)
        entry = {
            "workload": name,
            "method": args.method,
            "driver": args.driver,
            "device": args.device,
            "threads": torch.get_num_threads(),
            "quick": args.quick,
            "info": info,
            "total_seconds": round(time.perf_counter() - start, 3),
            "stages_seconds": {k: round(v, 3) for k, v in rec.stages.items()},
            "dxtb_timer_seconds": _dxtb_stages(),
            "peak_rss_mb": round(_rss_mb(), 1),
            "peak_cuda_mb": (
                round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available()
                else None
            ),
            "top_ops_self_cpu_seconds": [
                (k, round(v, 3)) for k, v in rec.top_ops
            ],
            "versions": {
                "dxtb": dxtb.__version__,
                "torch": torch.__version__,
                "tad-mctc": version("tad-mctc"),
            },
            "cpu": platform.processor() or platform.machine(),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        data.append(entry)
        print(json.dumps(entry, indent=1))
    path.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
