#!/usr/bin/env python3
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
"""WP6 torch.compile evaluation on LiH higher-order derivatives."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from dxtb import GFN1_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb.config import Config
from test.utils import scf_strategy

DD: dict[str, Any] = {"device": torch.device("cpu"), "dtype": torch.float64}


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _setup_lih(maxiter: int = 5):
    opts = {
        "method": labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": labels.INTLEVEL_DIPOLE,
        "differentiation": "unrolled",
        "strategy": scf_strategy("anderson", "charge"),
        "maxiter": maxiter,
        "x_atol": 1.0e-10,
        "x_atol_max": 1.0e-10,
        "fermi_diff_order": 0,
    }
    cfg = Config.create(**opts)
    par = ParamModule(GFN1_XTB, **DD)
    model = Model(par=par, config=cfg)
    numbers = torch.tensor([3, 1], dtype=torch.int64)
    system = model.setup(numbers)
    calc = Calculator(numbers, par, opts=opts | {"verbosity": 0}, **DD)
    positions = torch.tensor(
        [[0.0, 0.0, -1.50796689], [0.0, 0.0, 1.50796689]], **DD
    )
    field = torch.zeros((3,), **DD)
    return system, calc, positions, field


def _run_worker(mode: str, maxiter: int) -> dict[str, Any]:
    system, calc, positions, field = _setup_lih(maxiter=maxiter)
    rss_before = _peak_rss_mb()

    if mode == "eager_system_hyperpol":
        t0 = time.perf_counter()
        out1 = system.hyperpolarizability(positions, field=field)
        t1 = time.perf_counter()
        rss_first = _peak_rss_mb()
        t2 = time.perf_counter()
        out2 = system.hyperpolarizability(positions, field=field)
        t3 = time.perf_counter()
        rss_warm = _peak_rss_mb()
        return {
            "mode": mode,
            "maxiter": maxiter,
            "status": "ok",
            "first_call_s": round(t1 - t0, 4),
            "warm_call_s": round(t3 - t2, 4),
            "rss_before_mb": round(rss_before, 2),
            "rss_after_first_mb": round(rss_first, 2),
            "rss_after_warm_mb": round(rss_warm, 2),
            "compile_additional_rss_mb": round(rss_first - rss_before, 2),
            "xxz": float(out2[0, 0, 2].item()),
            "first_vs_warm_max_abs": float((out1 - out2).abs().max().item()),
        }

    if mode == "explain_singlepoint":
        def sp_fn(pos: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
            return system.energy(pos, field=f)

        t0 = time.perf_counter()
        explanation = torch._dynamo.explain(sp_fn)(positions, field)
        t1 = time.perf_counter()
        return {
            "mode": mode,
            "maxiter": maxiter,
            "status": "ok",
            "explain_time_s": round(t1 - t0, 4),
            "graph_count": int(explanation.graph_count),
            "graph_break_count": int(explanation.graph_break_count),
            "break_reasons": [str(r) for r in explanation.break_reasons[:5]],
        }

    if mode in ("compile_singlepoint_aot_eager", "compile_singlepoint_inductor"):
        backend = "aot_eager" if mode.endswith("aot_eager") else "inductor"
        eager_out = system.energy(positions, field=field)

        def sp_fn(pos: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
            return system.energy(pos, field=f)

        compiled_fn = torch.compile(sp_fn, backend=backend)
        t0 = time.perf_counter()
        c_out1 = compiled_fn(positions, field)
        t1 = time.perf_counter()
        rss_first = _peak_rss_mb()
        t2 = time.perf_counter()
        c_out2 = compiled_fn(positions, field)
        t3 = time.perf_counter()
        rss_warm = _peak_rss_mb()
        return {
            "mode": mode,
            "backend": backend,
            "maxiter": maxiter,
            "status": "ok",
            "first_call_s": round(t1 - t0, 4),
            "warm_call_s": round(t3 - t2, 4),
            "rss_before_mb": round(rss_before, 2),
            "rss_after_first_mb": round(rss_first, 2),
            "rss_after_warm_mb": round(rss_warm, 2),
            "compile_additional_rss_mb": round(rss_first - rss_before, 2),
            "max_abs_err_vs_eager": float(
                (c_out2 - eager_out).abs().max().item()
            ),
        }

    if mode in ("compile_hyperpol_aot_eager", "compile_hyperpol_inductor"):
        backend = "aot_eager" if mode.endswith("aot_eager") else "inductor"
        eager_out = system.hyperpolarizability(positions, field=field)
        rss_after_eager = _peak_rss_mb()

        def hyp_fn(pos: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
            return system.hyperpolarizability(pos, field=f)

        try:
            compiled_fn = torch.compile(hyp_fn, backend=backend)
            t0 = time.perf_counter()
            c_out1 = compiled_fn(positions, field)
            t1 = time.perf_counter()
            rss_first = _peak_rss_mb()
            t2 = time.perf_counter()
            c_out2 = compiled_fn(positions, field)
            t3 = time.perf_counter()
            rss_warm = _peak_rss_mb()
            return {
                "mode": mode,
                "backend": backend,
                "maxiter": maxiter,
                "status": "ok",
                "first_call_s": round(t1 - t0, 4),
                "warm_call_s": round(t3 - t2, 4),
                "rss_before_mb": round(rss_after_eager, 2),
                "rss_after_first_mb": round(rss_first, 2),
                "rss_after_warm_mb": round(rss_warm, 2),
                "compile_additional_rss_mb": round(
                    rss_first - rss_after_eager, 2
                ),
                "xxz": float(c_out2[0, 0, 2].item()),
                "max_abs_err_vs_eager": float(
                    (c_out2 - eager_out).abs().max().item()
                ),
            }
        except Exception as exc:
            return {
                "mode": mode,
                "backend": backend,
                "maxiter": maxiter,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if mode == "compile_calc_autograd_hyperpol":
        eager_out = calc.hyperpolarizability(
            positions.clone().requires_grad_(True),
            field=field.clone().requires_grad_(True),
            use_functorch=False,
        )
        rss_after_eager = _peak_rss_mb()

        def calc_hyp_fn(pos: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
            return calc.hyperpolarizability(pos, field=f, use_functorch=False)

        try:
            compiled_fn = torch.compile(calc_hyp_fn, backend="aot_eager")
            t0 = time.perf_counter()
            c_out1 = compiled_fn(
                positions.clone().requires_grad_(True),
                field.clone().requires_grad_(True),
            )
            t1 = time.perf_counter()
            rss_first = _peak_rss_mb()
            t2 = time.perf_counter()
            c_out2 = compiled_fn(
                positions.clone().requires_grad_(True),
                field.clone().requires_grad_(True),
            )
            t3 = time.perf_counter()
            rss_warm = _peak_rss_mb()
            return {
                "mode": mode,
                "backend": "aot_eager",
                "maxiter": maxiter,
                "status": "ok",
                "first_call_s": round(t1 - t0, 4),
                "warm_call_s": round(t3 - t2, 4),
                "rss_before_mb": round(rss_after_eager, 2),
                "rss_after_first_mb": round(rss_first, 2),
                "rss_after_warm_mb": round(rss_warm, 2),
                "compile_additional_rss_mb": round(
                    rss_first - rss_after_eager, 2
                ),
                "xxz": float(c_out2[0, 0, 2].item()),
                "max_abs_err_vs_eager": float(
                    (c_out2 - eager_out).abs().max().item()
                ),
            }
        except Exception as exc:
            return {
                "mode": mode,
                "backend": "aot_eager",
                "maxiter": maxiter,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", type=str, default="eager_system_hyperpol")
    parser.add_argument("--maxiter", type=int, default=5)
    args = parser.parse_args()

    if args.worker:
        res = _run_worker(args.mode, args.maxiter)
        print("WORKER_JSON:" + json.dumps(res))
        return

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    modes = [
        "eager_system_hyperpol",
        "explain_singlepoint",
        "compile_singlepoint_aot_eager",
        "compile_singlepoint_inductor",
        "compile_hyperpol_aot_eager",
        "compile_hyperpol_inductor",
        "compile_calc_autograd_hyperpol",
    ]
    results = []
    for mode in modes:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--mode",
                    mode,
                    "--maxiter",
                    str(args.maxiter),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            parsed = None
            for line in proc.stdout.splitlines():
                if line.startswith("WORKER_JSON:"):
                    parsed = json.loads(line[len("WORKER_JSON:") :])
            if parsed is None:
                parsed = {
                    "mode": mode,
                    "maxiter": args.maxiter,
                    "status": "crash",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[-1000:],
                }
        except subprocess.TimeoutExpired:
            parsed = {
                "mode": mode,
                "maxiter": args.maxiter,
                "status": "timeout_90s",
                "note": "Compilation exceeded 90s wall-clock budget",
            }
        results.append(parsed)
        print(json.dumps(parsed, indent=2), flush=True)

    out_path = Path("benchmarks/results/hod_p1_wp6_compile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved WP6 compile evaluation to {out_path}")


if __name__ == "__main__":
    main()
