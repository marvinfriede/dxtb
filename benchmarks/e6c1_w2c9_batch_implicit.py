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
"""Process-isolated E6c1-W2c9 tmpda scaling and kernel/response benchmark.

Usage::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -u benchmarks/e6c1_w2c9_batch_implicit.py \\
        --stage batch --nconformers 2 --gradient-mode positions-only \\
        --verify-parity --output benchmarks/results/e6c1-w2c9/n2_pos.json
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
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.baseline.workloads import conformers as frozen_conformers  # noqa: E402
from dxtb import GFN1_XTB, ParamModule, labels  # noqa: E402
from dxtb._src.calculators.model import Model  # noqa: E402
from dxtb.config import Config  # noqa: E402
from dxtb._src.calculators.batch_implicit import (  # noqa: E402
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)

DTYPE = torch.float64
DEVICE = torch.device("cpu")


def current_rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        return None


def environment_metadata() -> dict[str, Any]:
    return {
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_model": cpu_model(),
        "hostname": platform.node(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def setup_w2_model_and_system(
    numbers: torch.Tensor, *, gradient_mode: str
) -> tuple[Model, Any]:
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        mixer=labels.MIXER_ANDERSON,
        scf_mode=labels.SCF_MODE_FULL,
        scp_mode=labels.SCP_MODE_CHARGE,
        maxiter=300,
        f_atol=1.0e-8,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        fermi_etemp=300.0,
        force_convergence=True,
    )
    model = Model(
        par=par, config=config, auto_int_level=True
    )
    trainable = gradient_mode == "positions-and-model"
    for p in model.par.parameters():
        if p.is_floating_point():
            p.requires_grad_(trainable)
    system = model.setup(numbers, batch_mode=0)
    return model, system


def _instrument_implicit_solve():
    import tad_mctc.storch.implicit.root as implicit_root_module

    original = implicit_root_module.implicit_solve
    records: list[dict[str, Any]] = []
    solve_summaries: list[dict[str, Any]] = []
    backward_start = [0.0]

    def wrap_op(op, name: str):
        if op is None:
            return None

        def measured(*args, **kwargs):
            t0 = time.perf_counter()
            out = op(*args, **kwargs)
            records.append(
                {"operator": name, "seconds": time.perf_counter() - t0}
            )
            return out

        return measured

    def wrap_factory(factory, name: str):
        if factory is None:
            return None

        def make(*args, **kwargs):
            return wrap_op(factory(*args, **kwargs), name)

        return make

    def traced_solve(matvec, rhs, *args, **kwargs):
        t0 = time.perf_counter()
        pre_s = t0 - backward_start[0]
        kwargs["linearize"] = wrap_factory(kwargs.get("linearize"), "linearize")
        kwargs["rlinearize"] = wrap_factory(
            kwargs.get("rlinearize"), "rlinearize"
        )
        res = original(wrap_op(matvec, "matvec"), rhs, *args, **kwargs)
        dur = time.perf_counter() - t0
        rhs_det = rhs.detach()
        res_det = res.residual.detach()
        if rhs_det.ndim >= 2:
            dims = tuple(range(1, rhs_det.ndim))
            rhs_norm = torch.linalg.vector_norm(rhs_det, dim=dims)
        else:
            rhs_norm = torch.linalg.vector_norm(rhs_det)
        if res_det.ndim == rhs_det.ndim and res_det.ndim >= 2:
            res_norm = torch.linalg.vector_norm(
                res_det, dim=tuple(range(1, res_det.ndim))
            )
        else:
            res_norm = res_det.abs()
        rel_res = (
            res_norm / rhs_norm.clamp_min(torch.finfo(rhs.dtype).tiny)
        ).cpu().tolist()
        solve_summaries.append(
            {
                "pre_solver_s": pre_s,
                "solver_duration_s": dur,
                "status": res.status.detach().cpu().tolist(),
                "iterations": res.iterations.detach().cpu().tolist(),
                "rhs_norm": rhs_norm.cpu().tolist(),
                "residual_norm": res_norm.cpu().tolist(),
                "relative_residual": rel_res,
            }
        )
        return res

    implicit_root_module.implicit_solve = traced_solve
    return implicit_root_module, original, records, solve_summaries, backward_start


def run_kernel_ab_benchmark() -> dict[str, Any]:
    """Level 1 & Level 2 isolated kernel and response A/B benchmark on tmpda conformer 0."""
    import dxtb._src.scf.functional.spectral as spectral_mod
    from dxtb._src.scf.functional.map import scf_map
    from dxtb._src.scf.functional.solver import free_energy_from_spectral

    all_numbers, all_positions = frozen_conformers(
        "tmpda", 8, sigma=0.1, seed=0
    )
    numbers = all_numbers[0].to(device=DEVICE)
    pos0 = all_positions[0:1].to(device=DEVICE, dtype=DTYPE)
    _, system = setup_w2_model_and_system(
        numbers, gradient_mode="positions-only"
    )
    prepared = prepare_converged_root_batch(system, pos0)
    lane = prepared.lanes[0]
    root_x = lane.root_x
    inputs = lane.inputs

    # Warmup once
    with torch.no_grad():
        _ = scf_map(root_x, inputs)

    # Measure optimized (current W2c9) scf_map + free_energy_from_spectral forward & backward
    opt_no_grad_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            m = scf_map(root_x, inputs)
            _ = free_energy_from_spectral(m.spectral, inputs)
        opt_no_grad_times.append(time.perf_counter() - t0)

    opt_fwd_times = []
    opt_bwd_times = []
    for _ in range(3):
        x_req = root_x.detach().clone().requires_grad_(True)
        t0 = time.perf_counter()
        m = scf_map(x_req, inputs)
        fe = free_energy_from_spectral(m.spectral, inputs)
        loss = m.mapped_x.sum() + fe.sum()
        opt_fwd_times.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        g_opt = torch.autograd.grad(loss, x_req)[0]
        opt_bwd_times.append(time.perf_counter() - t1)

    # Measure baseline (without _tracks_derivatives primal fast-path)
    orig_tracks = spectral_mod._tracks_derivatives
    try:
        spectral_mod._tracks_derivatives = lambda _x: True
        base_no_grad_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                m = scf_map(root_x, inputs)
                _ = free_energy_from_spectral(m.spectral, inputs)
            base_no_grad_times.append(time.perf_counter() - t0)

        base_fwd_times = []
        base_bwd_times = []
        for _ in range(3):
            x_req = root_x.detach().clone().requires_grad_(True)
            t0 = time.perf_counter()
            m = scf_map(x_req, inputs)
            fe = free_energy_from_spectral(m.spectral, inputs)
            loss = m.mapped_x.sum() + fe.sum()
            base_fwd_times.append(time.perf_counter() - t0)
            t1 = time.perf_counter()
            g_base = torch.autograd.grad(loss, x_req)[0]
            base_bwd_times.append(time.perf_counter() - t1)
    finally:
        spectral_mod._tracks_derivatives = orig_tracks

    grad_max_diff = float((g_opt - g_base).abs().max().item())
    return {
        "stage": "kernel_ab",
        "environment": environment_metadata(),
        "baseline_no_grad_median_s": sorted(base_no_grad_times)[1],
        "optimized_no_grad_median_s": sorted(opt_no_grad_times)[1],
        "baseline_fwd_grad_median_s": sorted(base_fwd_times)[1],
        "optimized_fwd_grad_median_s": sorted(opt_fwd_times)[1],
        "baseline_bwd_grad_median_s": sorted(base_bwd_times)[1],
        "optimized_bwd_grad_median_s": sorted(opt_bwd_times)[1],
        "gradient_max_abs_diff": grad_max_diff,
        "rss_peak_mb": peak_rss_mb(),
    }


def run_batch_benchmark(
    nconformers: int,
    gradient_mode: str,
    *,
    verify_parity: bool = False,
    reverse_order: bool = False,
) -> dict[str, Any]:
    """Level 3 end-to-end physical tmpda conformer batch benchmark."""
    total_t0 = time.perf_counter()
    all_numbers, all_positions = frozen_conformers(
        "tmpda", 8, sigma=0.1, seed=0
    )
    indices = list(range(nconformers))
    if reverse_order:
        indices.reverse()
    numbers = all_numbers[0].to(device=DEVICE)
    positions = (
        all_positions[indices]
        .to(device=DEVICE, dtype=DTYPE)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    setup_t0 = time.perf_counter()
    model, system = setup_w2_model_and_system(
        numbers, gradient_mode=gradient_mode
    )
    setup_s = time.perf_counter() - setup_t0
    rss_after_setup = current_rss_mb()

    root_t0 = time.perf_counter()
    prepared = prepare_converged_root_batch(system, positions, strict=True)
    root_prep_s = time.perf_counter() - root_t0
    rss_after_root_prep = current_rss_mb()

    fwd_t0 = time.perf_counter()
    result = evaluate_converged_root_batch(prepared, strict=True)
    attach_finalize_s = time.perf_counter() - fwd_t0
    rss_after_forward = current_rss_mb()

    model_param_items = tuple(
        (name, p)
        for name, p in model.par.named_parameters()
        if p.is_floating_point() and p.requires_grad
    )
    diff_targets = (positions, *(p for _, p in model_param_items))

    (
        imp_mod,
        orig_solve,
        op_records,
        solve_summaries,
        bwd_start_ref,
    ) = _instrument_implicit_solve()
    bwd_t0 = time.perf_counter()
    bwd_start_ref[0] = bwd_t0
    try:
        grads = torch.autograd.grad(
            result.energy.sum(),
            diff_targets,
            allow_unused=True,
            retain_graph=verify_parity,
        )
    finally:
        imp_mod.implicit_solve = orig_solve
    backward_s = time.perf_counter() - bwd_t0
    total_s = time.perf_counter() - total_t0
    rss_after_backward = current_rss_mb()
    peak_mb = peak_rss_mb()

    pos_grad = grads[0]
    param_grads = grads[1:]
    assert pos_grad is not None

    floating_total = sum(
        1 for p in model.par.parameters() if p.is_floating_point()
    )
    connected_nonzero = sum(
        g is not None and bool((g.abs() > 0.0).any()) for g in param_grads
    )
    connected_zero = sum(
        g is not None and bool((g == 0.0).all()) for g in param_grads
    )
    disconnected = sum(g is None for g in param_grads)

    report: dict[str, Any] = {
        "stage": "batch",
        "nconformers": nconformers,
        "conformer_indices": indices,
        "gradient_mode": gradient_mode,
        "environment": environment_metadata(),
        "timings_s": {
            "setup_s": setup_s,
            "root_prep_s": root_prep_s,
            "attach_finalize_s": attach_finalize_s,
            "backward_s": backward_s,
            "total_s": total_s,
        },
        "memory_mb": {
            "rss_after_setup_mb": rss_after_setup,
            "rss_after_root_prep_mb": rss_after_root_prep,
            "rss_after_forward_mb": rss_after_forward,
            "rss_after_backward_mb": rss_after_backward,
            "peak_rss_mb": peak_mb,
        },
        "lanes": [
            {
                "lane_index": lane.lane_index,
                "conformer_index": indices[lane.lane_index],
                "converged": bool(lane.solver_converged.item()),
                "iterations": int(lane.iterations.item()),
                "solver_residual": lane.solver_residual.detach().cpu().tolist(),
                "raw_residual_l2": float(lane.raw_residual_l2.item()),
                "raw_residual_max": float(lane.raw_residual_max.item()),
                "electron_count_error": float(
                    lane.electron_count_error.item()
                ),
                "energy_eh": float(result.energy[lane.lane_index].sum().item()),
                "scf_eh": float(result.scf[lane.lane_index].sum().item()),
                "fenergy_eh": float(
                    result.fenergy[lane.lane_index].sum().item()
                ),
                "pos_grad_norm": float(
                    torch.linalg.vector_norm(
                        pos_grad[lane.lane_index].detach()
                    ).item()
                ),
                "pos_grad_max_abs": float(
                    pos_grad[lane.lane_index].detach().abs().max().item()
                ),
            }
            for lane in prepared.lanes
        ],
        "linear_response": {
            "solves": solve_summaries,
            "operator_counts": {
                name: sum(r["operator"] == name for r in op_records)
                for name in ("matvec", "linearize", "rlinearize")
            },
            "operator_seconds": {
                name: sum(
                    r["seconds"] for r in op_records if r["operator"] == name
                )
                for name in ("matvec", "linearize", "rlinearize")
            },
        },
        "model_parameters": {
            "floating_total": floating_total,
            "trainable_total": len(model_param_items),
            "connected_nonzero": connected_nonzero,
            "connected_zero": connected_zero,
            "disconnected": disconnected,
        },
    }

    if verify_parity:
        from dataclasses import replace

        cross_lane_max_abs = 0.0
        for idx in range(nconformers):
            lane_pos_grad = torch.autograd.grad(
                result.energy[idx].sum(),
                positions,
                retain_graph=True,
            )[0].detach()
            for other_idx in range(nconformers):
                if other_idx != idx:
                    c_max = float(lane_pos_grad[other_idx].abs().max().item())
                    if c_max > cross_lane_max_abs:
                        cross_lane_max_abs = c_max

        serial_energies = []
        serial_fenergies = []
        serial_pos_grads = []
        serial_param_accum = [None] * len(model_param_items)
        for idx in range(nconformers):
            serial_system = model.setup(numbers, batch_mode=0)
            s_pos = (
                positions[idx : idx + 1].detach().clone().requires_grad_(True)
            )
            base_lane = prepared.lanes[idx]
            s_lane = replace(base_lane, lane_index=0, positions=s_pos[0])
            s_prepared = replace(
                prepared,
                system=serial_system,
                positions=s_pos,
                lanes=(s_lane,),
                valid_lane_mask=prepared.valid_lane_mask[idx : idx + 1],
            )
            s_res = evaluate_converged_root_batch(
                s_prepared, positions=s_pos, strict=True
            )
            s_grads = torch.autograd.grad(
                s_res.energy.sum(),
                (s_pos, *(p for _, p in model_param_items)),
                allow_unused=True,
            )
            serial_energies.append(s_res.energy[0].sum().detach())
            serial_fenergies.append(s_res.fenergy[0].sum().detach())
            serial_pos_grads.append(s_grads[0][0].detach())
            for p_idx, g in enumerate(s_grads[1:]):
                if g is not None:
                    serial_param_accum[p_idx] = (
                        g.detach().clone()
                        if serial_param_accum[p_idx] is None
                        else serial_param_accum[p_idx] + g.detach()
                    )

        s_energy_t = torch.stack(serial_energies)
        s_fenergy_t = torch.stack(serial_fenergies)
        s_pos_grad_t = torch.stack(serial_pos_grads)

        energy_max_diff = float(
            (result.energy.sum(dim=-1).detach() - s_energy_t)
            .abs()
            .max()
            .item()
        )
        fenergy_max_diff = float(
            (result.fenergy.sum(dim=-1).detach() - s_fenergy_t)
            .abs()
            .max()
            .item()
        )
        pos_grad_max_diff = float(
            (pos_grad.detach() - s_pos_grad_t).abs().max().item()
        )

        param_max_diff = 0.0
        param_meta_matches = True
        for b_g, s_g in zip(param_grads, serial_param_accum):
            if (b_g is None) != (s_g is None):
                param_meta_matches = False
            elif b_g is not None and s_g is not None:
                diff = float((b_g.detach() - s_g).abs().max().item())
                if diff > param_max_diff:
                    param_max_diff = diff

        report["parity_vs_serial"] = {
            "energy_max_abs_diff": energy_max_diff,
            "fenergy_max_abs_diff": fenergy_max_diff,
            "pos_grad_max_abs_diff": pos_grad_max_diff,
            "cross_lane_pos_grad_max_abs": cross_lane_max_abs,
            "model_param_grad_max_abs_diff": param_max_diff,
            "model_param_connected_metadata_matches": param_meta_matches,
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("kernel-ab", "batch"), default="batch"
    )
    parser.add_argument("--nconformers", type=int, default=2)
    parser.add_argument(
        "--gradient-mode",
        choices=("positions-only", "positions-and-model"),
        default="positions-only",
    )
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if args.stage == "kernel-ab":
        result = run_kernel_ab_benchmark()
    else:
        result = run_batch_benchmark(
            args.nconformers,
            args.gradient_mode,
            verify_parity=args.verify_parity,
            reverse_order=args.reverse_order,
        )

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
