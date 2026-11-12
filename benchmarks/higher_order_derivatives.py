#!/usr/bin/env python3
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
Subprocess-isolated diagnostic benchmark for higher-order derivatives in dxtb.

Measures wall-clock execution time, process resident memory (baseline RSS, peak
RSS via VmHWM / RUSAGE_SELF, and peak additional RSS), CPU utilization, SCF
iteration counts, AD evaluation counts, and numerical accuracy across derivative
orders, integral drivers, SCF budgets, and AD transformation compositions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "HOD_BENCHMARK_JSON:"

DEFAULT_CASES = [
    "system_energy",
    "calculator_energy",
    "system_dipole",
    "calculator_dipole",
    "system_polarizability",
    "calculator_polarizability",
    "system_hyperpol",
    "calculator_hyperpol_autograd",
    "calculator_hyperpol_autograd_terminal",
    "calculator_hyperpol_functorch_energy",
    "calculator_hyperpol_functorch_dipole",
    "calculator_hyperpol_functorch_pol",
    "calculator_hyperpol_numerical",
    "system_hyperpol_numerical",
]

WP2_AD_CASES = [
    "ad_A0_jacfwd_jacfwd_jacrev",
    "ad_A_jacfwd_jacfwd_grad",
    "ad_B_jacrev_jacrev_grad",
    "ad_C1_jacfwd_jacrev_grad",
    "ad_C2_jacrev_jacfwd_grad",
    "ad_D1_jvp_directional_9_grad",
    "ad_D2_jvp_directional_sym6_grad",
    "ad_D3_jvp_outer3_jacfwd_grad",
    "ad_E1_chunk1_jacfwd_jacfwd_grad",
    "ad_E2_chunk1_jacrev_jacrev_grad",
]

ALL_CASES = DEFAULT_CASES + WP2_AD_CASES


def _read_proc_status_mb(pid: int | None = None) -> tuple[float, float]:
    """Return (VmRSS_MB, VmHWM_MB) from /proc/<pid>/status."""
    target = f"/proc/{pid}/status" if pid is not None else "/proc/self/status"
    rss_kb = 0.0
    hwm_kb = 0.0
    try:
        with open(target, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kb = float(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm_kb = float(line.split()[1])
    except OSError:
        pass
    return rss_kb / 1024.0, hwm_kb / 1024.0


def _try_reset_hwm() -> bool:
    """Attempt to reset /proc/self/status VmHWM to current VmRSS."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="utf-8") as handle:
            handle.write("5\n")
        return True
    except OSError:
        return False


def _ru_maxrss_mb() -> float:
    """Return resource.getrusage(RUSAGE_SELF).ru_maxrss in MiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _get_git_sha() -> str:
    """Return the current 40-character git commit SHA."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _get_molecule_geometry(name: str, torch_mod: Any) -> tuple[Any, Any, float, int]:
    """Return (numbers, positions, charge, uhf) in float64 for standard fixtures."""
    dtype = torch_mod.float64
    key = name.upper()
    if key == "LIH":
        numbers = torch_mod.tensor([3, 1], dtype=torch_mod.long)
        positions = torch_mod.tensor(
            [
                [0.0, 0.0, -1.50796743897235],
                [0.0, 0.0, +1.50796743897235],
            ],
            dtype=dtype,
        )
        return numbers, positions, 0.0, 0
    if key == "H2":
        numbers = torch_mod.tensor([1, 1], dtype=torch_mod.long)
        positions = torch_mod.tensor(
            [
                [0.0, 0.0, -0.70000000000000],
                [0.0, 0.0, +0.70000000000000],
            ],
            dtype=dtype,
        )
        return numbers, positions, 0.0, 0
    if key == "H2O":
        numbers = torch_mod.tensor([8, 1, 1], dtype=torch_mod.long)
        positions = torch_mod.tensor(
            [
                [0.00000000000000, 0.00000000000000, -0.22192154000000],
                [0.00000000000000, 1.43092809000000, +0.88768615000000],
                [0.00000000000000, -1.43092809000000, +0.88768615000000],
            ],
            dtype=dtype,
        )
        return numbers, positions, 0.0, 0
    if key == "NO2":
        numbers = torch_mod.tensor([7, 8, 8], dtype=torch_mod.long)
        positions = torch_mod.tensor(
            [
                [0.00000000000000, 0.00000000000000, 0.24423438000000],
                [0.00000000000000, 2.07855666000000, -0.89552605000000],
                [0.00000000000000, -2.07855666000000, -0.89552605000000],
            ],
            dtype=dtype,
        )
        return numbers, positions, 0.0, 1
    raise ValueError(f"Unsupported molecule '{name}'.")


def _run_worker(args: argparse.Namespace) -> int:
    """Execute a single benchmark case inside an isolated child process."""
    if args.mem_limit_gb > 0:
        limit_bytes = int(args.mem_limit_gb * (1024**3))
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            new_soft = limit_bytes if hard == resource.RLIM_INFINITY else min(limit_bytes, hard)
            resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        except (ValueError, OSError):
            pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import torch

    torch.set_num_threads(1)

    import dxtb
    import dxtb._src.scf.energy as scf_energy_mod
    import dxtb._src.scf.iterator as scf_iterator_mod
    import dxtb._src.scf.map as scf_map_mod
    import dxtb._src.scf.spectral as scf_spectral_mod
    from dxtb import Calculator, GFN1_XTB, GFN2_XTB
    from dxtb._src.calculators.model import System

    # Lightweight call counters for SCF / spectral internals
    counters = {
        "evaluate_hamiltonian": 0,
        "diagonalize": 0,
        "matrix_fermi": 0,
        "dense_solve": 0,
    }
    orig_eval_h = scf_map_mod.evaluate_hamiltonian
    orig_diag = scf_map_mod._transitional_diagonalize
    orig_fermi = scf_spectral_mod._matrix_fermi_impl
    orig_solve = scf_spectral_mod.dense_solve

    def counted_eval_h(*a: Any, **kw: Any) -> Any:
        counters["evaluate_hamiltonian"] += 1
        return orig_eval_h(*a, **kw)

    def counted_diag(*a: Any, **kw: Any) -> Any:
        counters["diagonalize"] += 1
        return orig_diag(*a, **kw)

    def counted_fermi(*a: Any, **kw: Any) -> Any:
        counters["matrix_fermi"] += 1
        return orig_fermi(*a, **kw)

    def counted_solve(*a: Any, **kw: Any) -> Any:
        counters["dense_solve"] += 1
        return orig_solve(*a, **kw)

    scf_map_mod.evaluate_hamiltonian = counted_eval_h
    scf_iterator_mod.evaluate_hamiltonian = counted_eval_h
    scf_energy_mod.evaluate_hamiltonian = counted_eval_h
    scf_map_mod._transitional_diagonalize = counted_diag
    scf_spectral_mod._matrix_fermi_impl = counted_fermi
    scf_spectral_mod.dense_solve = counted_solve

    dtype = torch.float64
    param = GFN1_XTB if args.method.upper() == "GFN1" else GFN2_XTB
    numbers, positions, charge_val, uhf_val = _get_molecule_geometry(args.molecule, torch)
    charge = torch.tensor(charge_val, dtype=dtype)
    uhf = torch.tensor(uhf_val, dtype=dtype) if uhf_val != 0 else None
    field = torch.zeros((3,), dtype=dtype)

    opts = {
        "int_level": dxtb.labels.INTLEVEL_DIPOLE,
        "int_driver": args.driver,
        "maxiter": args.maxiter,
        "strategy": "anderson",
        "differentiation": args.differentiation,
        "verbosity": 0,
        "x_atol": 1e-10,
        "fermi_diff_order": args.fermi_diff_order,
    }

    case = args.worker_case
    t_setup_start = time.perf_counter()

    system_obj: System | None = None
    calc_obj: Calculator | None = None

    if case.startswith("system_") or case.startswith("ad_"):
        system_obj = Calculator(
            numbers,
            param,
            opts=opts,
            dtype=dtype,
        ).system
    elif case.startswith("calculator_"):
        calc_obj = Calculator(
            numbers,
            param,
            opts=opts,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown case '{case}'.")

    t_setup_end = time.perf_counter()
    setup_time_s = t_setup_end - t_setup_start

    def _invoke_case() -> torch.Tensor:
        assert torch is not None
        if case == "system_energy":
            assert system_obj is not None
            return system_obj.energy(
                positions, chrg=charge, spin=uhf, field=field
            )
        if case == "calculator_energy":
            assert calc_obj is not None
            return calc_obj.energy(
                positions.clone(), chrg=charge, spin=uhf, field=field.clone()
            )
        if case == "system_dipole":
            assert system_obj is not None
            return system_obj.dipole(
                positions, chrg=charge, spin=uhf, field=field
            )
        if case == "calculator_dipole":
            assert calc_obj is not None
            f_req = field.clone().requires_grad_(True)
            return calc_obj.dipole(
                positions.clone(), chrg=charge, spin=uhf, field=f_req
            )
        if case == "system_polarizability":
            assert system_obj is not None
            return system_obj.polarizability(
                positions, chrg=charge, spin=uhf, field=field
            )
        if case == "calculator_polarizability":
            assert calc_obj is not None
            f_req = field.clone().requires_grad_(True)
            return calc_obj.polarizability(
                positions.clone(), chrg=charge, spin=uhf, field=f_req
            )
        if case == "system_hyperpol":
            assert system_obj is not None
            return system_obj.hyperpolarizability(
                positions, chrg=charge, spin=uhf, field=field
            )
        if case == "calculator_hyperpol_autograd":
            assert calc_obj is not None
            f_req = field.clone().requires_grad_(True)
            return calc_obj.hyperpolarizability(
                positions.clone(),
                chrg=charge,
                spin=uhf,
                use_functorch=False,
                field=f_req,
                create_graph=True,
            )
        if case == "calculator_hyperpol_autograd_terminal":
            assert calc_obj is not None
            f_req = field.clone().requires_grad_(True)
            return calc_obj.hyperpolarizability(
                positions.clone(),
                chrg=charge,
                spin=uhf,
                use_functorch=False,
                field=f_req,
                create_graph=False,
            )
        if case == "calculator_hyperpol_functorch_energy":
            assert calc_obj is not None
            return calc_obj.hyperpolarizability(
                positions.clone(),
                chrg=charge,
                spin=uhf,
                use_functorch=True,
                derived_quantity="energy",
                field=field.clone(),
            )
        if case == "calculator_hyperpol_functorch_dipole":
            assert calc_obj is not None
            return calc_obj.hyperpolarizability(
                positions.clone(),
                chrg=charge,
                spin=uhf,
                use_functorch=True,
                derived_quantity="dipole",
                field=field.clone(),
            )
        if case == "calculator_hyperpol_functorch_pol":
            assert calc_obj is not None
            return calc_obj.hyperpolarizability(
                positions.clone(),
                chrg=charge,
                spin=uhf,
                use_functorch=True,
                derived_quantity="pol",
                field=field.clone(),
            )
        if case == "calculator_hyperpol_numerical":
            assert calc_obj is not None
            return calc_obj.hyperpolarizability_numerical(
                positions.clone(),
                chrg=charge,
                spin=uhf,
                field=field.clone(),
            )
        if case == "system_hyperpol_numerical":
            assert system_obj is not None
            step = 1.0e-4
            cols = []
            for k in range(3):
                df = torch.zeros((3,), dtype=dtype)
                df[k] = step
                pol_p = system_obj.polarizability(
                    positions, chrg=charge, spin=uhf, field=field + df
                )
                pol_m = system_obj.polarizability(
                    positions, chrg=charge, spin=uhf, field=field - df
                )
                cols.append(0.5 * (pol_p - pol_m) / step)
            return torch.stack(cols, dim=-1)

        # WP2 AD composition candidates on energy_of_field
        assert system_obj is not None

        def energy_of_field(f_vec: torch.Tensor) -> torch.Tensor:
            return system_obj.energy(
                positions,
                chrg=charge,
                spin=uhf,
                field=f_vec,
            ).reshape(())

        if case == "ad_A0_jacfwd_jacfwd_jacrev":
            grad_fn = torch.func.jacrev(energy_of_field)
            hess_fn = torch.func.jacfwd(grad_fn)
            third_fn = torch.func.jacfwd(hess_fn)
            return -third_fn(field)

        if case == "ad_A_jacfwd_jacfwd_grad":
            grad_fn = torch.func.grad(energy_of_field)
            hess_fn = torch.func.jacfwd(grad_fn)
            third_fn = torch.func.jacfwd(hess_fn)
            return -third_fn(field)

        if case == "ad_B_jacrev_jacrev_grad":
            grad_fn = torch.func.grad(energy_of_field)
            hess_fn = torch.func.jacrev(grad_fn)
            third_fn = torch.func.jacrev(hess_fn)
            return -third_fn(field)

        if case == "ad_C1_jacfwd_jacrev_grad":
            grad_fn = torch.func.grad(energy_of_field)
            hess_fn = torch.func.jacrev(grad_fn)
            third_fn = torch.func.jacfwd(hess_fn)
            return -third_fn(field)

        if case == "ad_C2_jacrev_jacfwd_grad":
            grad_fn = torch.func.grad(energy_of_field)
            hess_fn = torch.func.jacfwd(grad_fn)
            third_fn = torch.func.jacrev(hess_fn)
            return -third_fn(field)

        if case == "ad_D1_jvp_directional_9_grad":
            grad_fn = torch.func.grad(energy_of_field)
            basis = torch.eye(3, dtype=dtype, device=field.device)
            beta = torch.empty((3, 3, 3), dtype=dtype, device=field.device)
            for i in range(3):
                e_i = basis[i]
                for j in range(3):
                    e_j = basis[j]

                    def jvp_j(f_in: torch.Tensor) -> torch.Tensor:
                        _, dg_dj = torch.func.jvp(grad_fn, (f_in,), (e_j,))
                        return dg_dj

                    _, d2g_didj = torch.func.jvp(jvp_j, (field,), (e_i,))
                    beta[:, j, i] = -d2g_didj
            return beta

        if case == "ad_D2_jvp_directional_sym6_grad":
            grad_fn = torch.func.grad(energy_of_field)
            basis = torch.eye(3, dtype=dtype, device=field.device)
            beta = torch.empty((3, 3, 3), dtype=dtype, device=field.device)
            for i in range(3):
                e_i = basis[i]
                for j in range(i, 3):
                    e_j = basis[j]

                    def jvp_j(f_in: torch.Tensor) -> torch.Tensor:
                        _, dg_dj = torch.func.jvp(grad_fn, (f_in,), (e_j,))
                        return dg_dj

                    _, d2g_didj = torch.func.jvp(jvp_j, (field,), (e_i,))
                    vec = -d2g_didj
                    beta[:, j, i] = vec
                    if i != j:
                        beta[:, i, j] = vec
            return beta

        if case == "ad_D3_jvp_outer3_jacfwd_grad":
            grad_fn = torch.func.grad(energy_of_field)
            hess_fn = torch.func.jacfwd(grad_fn)
            basis = torch.eye(3, dtype=dtype, device=field.device)
            cols = []
            for k in range(3):
                e_k = basis[k]
                _, dh_dk = torch.func.jvp(hess_fn, (field,), (e_k,))
                cols.append(-dh_dk)
            return torch.stack(cols, dim=-1)

        if case == "ad_E1_chunk1_jacfwd_jacfwd_grad":
            grad_fn = torch.func.grad(energy_of_field)
            hess_fn = torch.func.jacfwd(grad_fn, chunk_size=1)
            third_fn = torch.func.jacfwd(hess_fn, chunk_size=1)
            return -third_fn(field)

        if case == "ad_E2_chunk1_jacrev_jacrev_grad":
            grad_fn = torch.func.grad(energy_of_field)
            hess_fn = torch.func.jacrev(grad_fn, chunk_size=1)
            third_fn = torch.func.jacrev(hess_fn, chunk_size=1)
            return -third_fn(field)

        raise ValueError(f"Unhandled case '{case}'.")

    baseline_rss_mb, baseline_hwm_mb = _read_proc_status_mb()
    hwm_reset_ok = _try_reset_hwm()
    if hwm_reset_ok:
        baseline_rss_mb, baseline_hwm_mb = _read_proc_status_mb()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    out = _invoke_case()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    ru_after = resource.getrusage(resource.RUSAGE_SELF)

    first_call_time_s = t1 - t0
    cpu_user_s = float(ru_after.ru_utime - ru_before.ru_utime)
    cpu_sys_s = float(ru_after.ru_stime - ru_before.ru_stime)
    cpu_util_pct = (
        100.0 * (cpu_user_s + cpu_sys_s) / first_call_time_s
        if first_call_time_s > 0.0
        else 0.0
    )

    post_rss_mb, post_hwm_mb = _read_proc_status_mb()
    ru_peak_mb = _ru_maxrss_mb()
    peak_rss_mb = max(post_hwm_mb, ru_peak_mb)
    peak_additional_rss_mb = max(0.0, peak_rss_mb - baseline_rss_mb)
    retained_additional_rss_mb = max(0.0, post_rss_mb - baseline_rss_mb)
    graph_retained = bool(out.requires_grad or out.grad_fn is not None)

    first_counters = dict(counters)

    repeat_call_time_s: float | None = None
    repeat_peak_rss_mb: float | None = None
    if args.repeat:
        t_rep0 = time.perf_counter()
        _ = _invoke_case()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_rep1 = time.perf_counter()
        repeat_call_time_s = t_rep1 - t_rep0
        _, rep_hwm_mb = _read_proc_status_mb()
        repeat_peak_rss_mb = max(rep_hwm_mb, _ru_maxrss_mb())

    out_det = out.detach().cpu().to(torch.float64)
    del out
    gc.collect()
    post_free_rss_mb, _ = _read_proc_status_mb()
    flat_vals = [float(x) for x in out_det.reshape(-1).tolist()]
    val_bytes = out_det.numpy().tobytes()
    checksum = hashlib.sha256(val_bytes).hexdigest()[:16]

    xxz_val: float | None = None
    zzz_val: float | None = None
    if out_det.shape == (3, 3, 3):
        xxz_val = float(out_det[0, 0, 2].item())
        zzz_val = float(out_det[2, 2, 2].item())

    # Run a detached primal singlepoint at the very end to record SCF convergence metrics
    primal_calc = Calculator(
        numbers,
        param,
        opts=opts,
        dtype=dtype,
    )
    sp_res = primal_calc.singlepoint(
        positions.detach().clone(), chrg=charge, spin=uhf, field=field.clone()
    )
    primal_scf_iters = int(sp_res.iterations.item())
    primal_converged = bool(sp_res.converged.item())
    primal_res_norm = float(torch.linalg.norm(sp_res.residual).item())
    is_eager_schedule = case.startswith("calculator_")
    scf_schedule = "eager" if is_eager_schedule else "fixed_trip"
    executed_scf_iters = primal_scf_iters if is_eager_schedule else int(args.maxiter)

    cuda_avail = bool(torch.cuda.is_available())
    cuda_alloc_mb = (
        float(torch.cuda.max_memory_allocated()) / (1024.0**2) if cuda_avail else 0.0
    )
    cuda_res_mb = (
        float(torch.cuda.max_memory_reserved()) / (1024.0**2) if cuda_avail else 0.0
    )
    gpu_model = torch.cuda.get_device_name(0) if cuda_avail else "CPU-only"

    payload = {
        "status": "OK",
        "case": case,
        "molecule": args.molecule,
        "method": args.method.upper(),
        "driver": args.driver,
        "differentiation": args.differentiation,
        "fermi_diff_order": args.fermi_diff_order,
        "maxiter": args.maxiter,
        "scf_schedule": scf_schedule,
        "executed_scf_iterations": executed_scf_iters,
        "graph_retained": graph_retained,
        "setup_time_s": round(setup_time_s, 6),
        "first_call_time_s": round(first_call_time_s, 6),
        "repeat_call_time_s": (
            round(repeat_call_time_s, 6) if repeat_call_time_s is not None else None
        ),
        "baseline_rss_mb": round(baseline_rss_mb, 3),
        "post_rss_mb": round(post_rss_mb, 3),
        "post_free_rss_mb": round(post_free_rss_mb, 3),
        "retained_additional_rss_mb": round(retained_additional_rss_mb, 3),
        "peak_rss_mb": round(peak_rss_mb, 3),
        "peak_additional_rss_mb": round(peak_additional_rss_mb, 3),
        "repeat_peak_rss_mb": (
            round(repeat_peak_rss_mb, 3) if repeat_peak_rss_mb is not None else None
        ),
        "hwm_reset_supported": hwm_reset_ok,
        "cpu_user_s": round(cpu_user_s, 6),
        "cpu_sys_s": round(cpu_sys_s, 6),
        "cpu_util_pct": round(cpu_util_pct, 2),
        "primal_scf_iterations": primal_scf_iters,
        "primal_converged": primal_converged,
        "primal_residual_norm": primal_res_norm,
        "ad_hamiltonian_evals": first_counters["evaluate_hamiltonian"],
        "ad_diagonalize_evals": first_counters["diagonalize"],
        "ad_matrix_fermi_evals": first_counters["matrix_fermi"],
        "ad_dense_solve_evals": first_counters["dense_solve"],
        "shape": list(out_det.shape),
        "dtype": str(out_det.dtype),
        "all_finite": bool(torch.isfinite(out_det).all().item()),
        "max_abs": float(out_det.abs().max().item()),
        "frobenius_norm": float(torch.linalg.norm(out_det).item()),
        "xxz": xxz_val,
        "zzz": zzz_val,
        "checksum_sha256": checksum,
        "values": flat_vals,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
        },
        "cuda_available": cuda_avail,
        "cuda_max_allocated_mb": round(cuda_alloc_mb, 3),
        "cuda_max_reserved_mb": round(cuda_res_mb, 3),
        "gpu_model": gpu_model,
        "commit_sha": _get_git_sha(),
    }
    print(f"{RESULT_PREFIX}{json.dumps(payload)}", flush=True)
    return 0


def _run_subprocess_case(
    case: str,
    maxiter: int,
    driver: str,
    differentiation: str,
    fermi_diff_order: int,
    molecule: str,
    method: str,
    repeat: bool,
    timeout_s: float,
    mem_limit_gb: float,
) -> dict[str, Any]:
    """Launch a fresh Python subprocess for a single benchmark configuration."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        case,
        "--maxiter",
        str(maxiter),
        "--driver",
        driver,
        "--differentiation",
        differentiation,
        "--fermi-diff-order",
        str(fermi_diff_order),
        "--molecule",
        molecule,
        "--method",
        method,
        "--mem-limit-gb",
        str(mem_limit_gb),
    ]
    if repeat:
        cmd.append("--repeat")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    sampled_max_rss_mb = 0.0
    sampled_max_hwm_mb = 0.0
    stop_event = threading.Event()

    def _sampler() -> None:
        nonlocal sampled_max_rss_mb, sampled_max_hwm_mb
        while not stop_event.is_set():
            rss_mb, hwm_mb = _read_proc_status_mb(proc.pid)
            if rss_mb > sampled_max_rss_mb:
                sampled_max_rss_mb = rss_mb
            if hwm_mb > sampled_max_hwm_mb:
                sampled_max_hwm_mb = hwm_mb
            stop_event.wait(0.01)

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        stop_event.set()
        sampler_thread.join(timeout=1.0)

    wall_s = time.perf_counter() - t0

    if not timed_out and proc.returncode == 0:
        for line in stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                data = json.loads(line[len(RESULT_PREFIX) :])
                data["sampled_peak_rss_mb"] = round(
                    max(sampled_max_rss_mb, sampled_max_hwm_mb), 3
                )
                data["peak_rss_mb"] = round(
                    max(
                        float(data.get("peak_rss_mb", 0.0)),
                        sampled_max_rss_mb,
                        sampled_max_hwm_mb,
                    ),
                    3,
                )
                return data

    status = "TIMEOUT" if timed_out else f"EXIT_{proc.returncode}"
    err_tail = (stderr or stdout or "").strip().splitlines()[-5:]
    return {
        "status": status,
        "case": case,
        "molecule": molecule,
        "method": method.upper(),
        "driver": driver,
        "differentiation": differentiation,
        "fermi_diff_order": fermi_diff_order,
        "maxiter": maxiter,
        "setup_time_s": None,
        "first_call_time_s": round(wall_s, 6),
        "repeat_call_time_s": None,
        "baseline_rss_mb": None,
        "post_rss_mb": None,
        "peak_rss_mb": round(max(sampled_max_rss_mb, sampled_max_hwm_mb), 3),
        "peak_additional_rss_mb": None,
        "sampled_peak_rss_mb": round(max(sampled_max_rss_mb, sampled_max_hwm_mb), 3),
        "primal_scf_iterations": None,
        "primal_converged": None,
        "primal_residual_norm": None,
        "ad_hamiltonian_evals": None,
        "ad_diagonalize_evals": None,
        "ad_matrix_fermi_evals": None,
        "ad_dense_solve_evals": None,
        "shape": None,
        "dtype": None,
        "all_finite": False,
        "max_abs": None,
        "frobenius_norm": None,
        "xxz": None,
        "zzz": None,
        "checksum_sha256": None,
        "error_tail": " | ".join(err_tail),
        "commit_sha": _get_git_sha(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-case", type=str, default=None)
    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(DEFAULT_CASES),
        help="Comma-separated list of benchmark cases (or 'default', 'wp2', 'all').",
    )
    parser.add_argument(
        "--maxiters",
        type=str,
        default="2,5,10,20,40,100",
        help="Comma-separated list of SCF maxiter budgets.",
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument(
        "--drivers",
        type=str,
        default="libcint,pytorch",
        help="Comma-separated list of integral drivers.",
    )
    parser.add_argument("--driver", type=str, default="libcint")
    parser.add_argument(
        "--differentiations",
        type=str,
        default="unrolled",
        help="Comma-separated list of differentiation modes.",
    )
    parser.add_argument("--differentiation", type=str, default="unrolled")
    parser.add_argument("--fermi-diff-order", type=int, default=0)
    parser.add_argument("--molecule", type=str, default="LiH")
    parser.add_argument("--method", type=str, default="GFN1")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--mem-limit-gb", type=float, default=32.0)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    if args.worker_case is not None:
        return _run_worker(args)

    if args.cases == "default":
        case_list = list(DEFAULT_CASES)
    elif args.cases == "wp2":
        case_list = list(WP2_AD_CASES)
    elif args.cases == "all":
        case_list = list(ALL_CASES)
    else:
        case_list = [c.strip() for c in args.cases.split(",") if c.strip()]

    maxiter_list = [int(x.strip()) for x in args.maxiters.split(",") if x.strip()]
    driver_list = [d.strip() for d in args.drivers.split(",") if d.strip()]
    diff_list = [d.strip() for d in args.differentiations.split(",") if d.strip()]

    results: list[dict[str, Any]] = []
    header = (
        f"{'Driver':<8} {'Diff':<9} {'MaxIt':>5} {'Case':<36} "
        f"{'Status':<8} {'Time(s)':>9} {'PeakRSS(MB)':>11} {'AddRSS(MB)':>10} "
        f"{'SCFIt':>5} {'HamEval':>7} {'xxz':>12}"
    )
    print(header)
    print("-" * len(header))

    for driver in driver_list:
        for diff_mode in diff_list:
            for maxiter in maxiter_list:
                for case in case_list:
                    rec = _run_subprocess_case(
                        case=case,
                        maxiter=maxiter,
                        driver=driver,
                        differentiation=diff_mode,
                        fermi_diff_order=args.fermi_diff_order,
                        molecule=args.molecule,
                        method=args.method,
                        repeat=args.repeat,
                        timeout_s=args.timeout,
                        mem_limit_gb=args.mem_limit_gb,
                    )
                    results.append(rec)
                    t_str = (
                        f"{rec['first_call_time_s']:.4f}"
                        if rec.get("first_call_time_s") is not None
                        else "N/A"
                    )
                    p_str = (
                        f"{rec['peak_rss_mb']:.1f}"
                        if rec.get("peak_rss_mb") is not None
                        else "N/A"
                    )
                    a_str = (
                        f"{rec['peak_additional_rss_mb']:.1f}"
                        if rec.get("peak_additional_rss_mb") is not None
                        else "N/A"
                    )
                    it_str = (
                        str(rec["primal_scf_iterations"])
                        if rec.get("primal_scf_iterations") is not None
                        else "N/A"
                    )
                    he_str = (
                        str(rec["ad_hamiltonian_evals"])
                        if rec.get("ad_hamiltonian_evals") is not None
                        else "N/A"
                    )
                    xxz_str = (
                        f"{rec['xxz']:.4f}" if rec.get("xxz") is not None else "N/A"
                    )
                    print(
                        f"{driver:<8} {diff_mode:<9} {maxiter:>5d} {case:<36} "
                        f"{rec['status']:<8} {t_str:>9} {p_str:>11} {a_str:>10} "
                        f"{it_str:>5} {he_str:>7} {xxz_str:>12}",
                        flush=True,
                    )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    if args.output_csv and results:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_fields = [
            "status",
            "driver",
            "differentiation",
            "fermi_diff_order",
            "maxiter",
            "case",
            "molecule",
            "method",
            "setup_time_s",
            "first_call_time_s",
            "repeat_call_time_s",
            "baseline_rss_mb",
            "post_rss_mb",
            "peak_rss_mb",
            "peak_additional_rss_mb",
            "cpu_user_s",
            "cpu_sys_s",
            "cpu_util_pct",
            "primal_scf_iterations",
            "primal_converged",
            "primal_residual_norm",
            "ad_hamiltonian_evals",
            "ad_diagonalize_evals",
            "ad_matrix_fermi_evals",
            "ad_dense_solve_evals",
            "shape",
            "dtype",
            "all_finite",
            "max_abs",
            "frobenius_norm",
            "xxz",
            "zzz",
            "checksum_sha256",
            "python_version",
            "torch_version",
            "commit_sha",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
