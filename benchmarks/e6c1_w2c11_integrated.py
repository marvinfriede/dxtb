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
"""Process-isolated E6c1-W2c11 integrated batch SCF benchmark and audit runner."""

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
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.baseline.workloads import conformers  # noqa: E402
from dxtb import GFN1_XTB, Calculator, ParamModule, labels  # noqa: E402
from dxtb._src.calculators.batch_implicit import (  # noqa: E402
    ResponseDiagnosticCollector,
    classify_batch_failure_category,
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.calculators.batch_streaming import (  # noqa: E402
    stream_batch_value_and_grad,
)
from dxtb._src.calculators.model import Model  # noqa: E402
from dxtb.config import Config  # noqa: E402

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


class RSSMonitor:
    """Sample process RSS and attribute peaks to explicit benchmark phases."""

    def __init__(self) -> None:
        self.phase = "setup"
        self.peaks: dict[str, float] = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.02):
            rss = current_rss_mb()
            if rss is not None:
                self.peaks[self.phase] = max(
                    self.peaks.get(self.phase, 0.0), rss
                )

    def start(self) -> None:
        rss = current_rss_mb()
        if rss is not None:
            self.peaks[self.phase] = rss
        self.thread.start()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        rss = current_rss_mb()
        if rss is not None:
            self.peaks[phase] = max(self.peaks.get(phase, 0.0), rss)

    def close(self) -> dict[str, float]:
        rss = current_rss_mb()
        if rss is not None:
            self.peaks[self.phase] = max(self.peaks.get(self.phase, 0.0), rss)
        self.stop.set()
        self.thread.join()
        return dict(self.peaks)


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def get_environment_metadata() -> dict[str, Any]:
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    source_files = (
        ROOT / "src/dxtb/_src/calculators/batch_implicit.py",
        ROOT / "src/dxtb/_src/calculators/batch_streaming.py",
        ROOT / "src/dxtb/_src/scf/functional/spectral.py",
    )
    source_hash = hashlib.sha256()
    for source_file in source_files:
        source_hash.update(source_file.name.encode())
        source_hash.update(source_file.read_bytes())
    return {
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
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_model": _cpu_model(),
        "hostname": platform.node(),
        "torch_num_threads": torch.get_num_threads(),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def setup_w2_model_and_system(
    numbers: torch.Tensor,
    *,
    gradient_mode: str = "positions-only",
    scp_mode: str = "charge",
) -> tuple[Model, Any]:
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    scp_label = (
        labels.SCP_MODE_CHARGE
        if scp_mode == "charge"
        else labels.SCP_MODE_POTENTIAL
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        mixer=labels.MIXER_ANDERSON,
        scf_mode=labels.SCF_MODE_FULL,
        scp_mode=scp_label,
        maxiter=300,
        f_atol=1.0e-8,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        fermi_etemp=300.0,
        force_convergence=True,
    )
    model = Model(par=par, config=config, auto_int_level=True)
    trainable = gradient_mode == "positions-and-model"
    for p in model.par.parameters():
        if p.is_floating_point():
            p.requires_grad_(trainable)
    system = model.setup(numbers, batch_mode=0)
    return model, system


def _make_weights(nconformers: int, weights_mode: str) -> torch.Tensor:
    if weights_mode == "unit":
        return torch.ones(nconformers, dtype=DTYPE, device=DEVICE)
    if weights_mode == "linspace":
        if nconformers == 1:
            return torch.tensor([1.0], dtype=DTYPE, device=DEVICE)
        return torch.linspace(0.5, 1.5, nconformers, dtype=DTYPE, device=DEVICE)
    if weights_mode == "mixed-sign-zero":
        pattern = [1.25, -0.75, 0.0, 0.5, -1.0, 0.25, 0.0, 0.8]
        return torch.tensor(
            pattern[:nconformers], dtype=DTYPE, device=DEVICE
        )
    raise ValueError(f"Unknown weights_mode '{weights_mode}'.")


def run_root_profile(fermi_mode: str) -> dict[str, Any]:
    """Profile physical root preparation phases on tmpda conformer 0."""
    import dxtb._src.scf.functional.anderson as anderson_mod
    import dxtb._src.scf.functional.map as map_mod
    import dxtb._src.scf.functional.solver as solver_mod
    import dxtb._src.scf.functional.spectral as spectral_mod

    all_numbers, all_positions = conformers("tmpda", 8, sigma=0.1, seed=0)
    numbers = all_numbers[0].to(device=DEVICE)
    pos0 = all_positions[0:1].to(device=DEVICE, dtype=DTYPE)
    _, system = setup_w2_model_and_system(numbers)

    phase_timers = {
        "scf_map_total_s": 0.0,
        "scf_map_calls": 0,
        "fermi_density_matrix_s": 0.0,
        "fermi_density_calls": 0,
        "anderson_mix_s": 0.0,
        "anderson_mix_calls": 0,
        "dense_solve_calls": 0,
        "eigh_calls": 0,
    }

    orig_eval_pot = map_mod.evaluate_potential
    orig_fermi = map_mod.fermi_density_matrix
    orig_mix = anderson_mod.anderson_mix
    orig_dense_solve = spectral_mod.dense_solve
    orig_eigh = torch.linalg.eigh

    def timed_eval_pot(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_eval_pot(*args, **kwargs)
        phase_timers["scf_map_total_s"] += time.perf_counter() - t0
        phase_timers["scf_map_calls"] += 1
        return out

    def timed_fermi(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_fermi(*args, **kwargs)
        phase_timers["fermi_density_matrix_s"] += time.perf_counter() - t0
        phase_timers["fermi_density_calls"] += 1
        return out

    def timed_mix(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_mix(*args, **kwargs)
        phase_timers["anderson_mix_s"] += time.perf_counter() - t0
        phase_timers["anderson_mix_calls"] += 1
        return out

    def counted_dense_solve(*args, **kwargs):
        phase_timers["dense_solve_calls"] += 1
        return orig_dense_solve(*args, **kwargs)

    def counted_eigh(*args, **kwargs):
        phase_timers["eigh_calls"] += 1
        return orig_eigh(*args, **kwargs)

    map_mod.evaluate_potential = timed_eval_pot
    map_mod.fermi_density_matrix = timed_fermi
    anderson_mod.anderson_mix = timed_mix
    spectral_mod.dense_solve = counted_dense_solve
    torch.linalg.eigh = counted_eigh
    try:
        t_start = time.perf_counter()
        prepared = prepare_converged_root_batch(
            system,
            pos0,
            strict=True,
            eager_primal_fermi=fermi_mode,
        )
        total_root_s = time.perf_counter() - t_start
    finally:
        map_mod.evaluate_potential = orig_eval_pot
        map_mod.fermi_density_matrix = orig_fermi
        anderson_mod.anderson_mix = orig_mix
        spectral_mod.dense_solve = orig_dense_solve
        torch.linalg.eigh = orig_eigh

    # Also measure static validation time in isolation on the converged root
    lane = prepared.lanes[0]
    t_val0 = time.perf_counter()
    with torch.no_grad():
        _ = map_mod.scf_map(lane.root_x, lane.inputs)
    static_validation_s = time.perf_counter() - t_val0

    fock_and_charges_s = (
        phase_timers["scf_map_total_s"] - phase_timers["fermi_density_matrix_s"]
    )
    setup_and_validation_s = (
        total_root_s
        - phase_timers["scf_map_total_s"]
        - phase_timers["anderson_mix_s"]
    )
    root_sha256 = hashlib.sha256(
        lane.root_x.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()

    return {
        "stage": "root-profile",
        "fermi_mode": fermi_mode,
        "iterations": int(lane.iterations.item()),
        "raw_residual_l2": float(lane.raw_residual_l2.item()),
        "raw_residual_max": float(lane.raw_residual_max.item()),
        "electron_count_error": float(lane.electron_count_error.item()),
        "root_x_sha256": root_sha256,
        "root_x_values": lane.root_x.detach().cpu().tolist(),
        "timings_s": {
            "total_root_prep_s": total_root_s,
            "scf_map_loop_total_s": phase_timers["scf_map_total_s"],
            "fermi_density_matrix_s": phase_timers["fermi_density_matrix_s"],
            "fock_and_charges_s": fock_and_charges_s,
            "anderson_mix_s": phase_timers["anderson_mix_s"],
            "static_validation_single_map_s": static_validation_s,
            "setup_and_validation_s": setup_and_validation_s,
        },
        "counts": {
            "scf_map_loop_calls": phase_timers["scf_map_calls"],
            "fermi_density_calls": phase_timers["fermi_density_calls"],
            "anderson_mix_calls": phase_timers["anderson_mix_calls"],
            "dense_solve_calls": phase_timers["dense_solve_calls"],
            "eigh_calls": phase_timers["eigh_calls"],
        },
        "peak_rss_mb": peak_rss_mb(),
        "environment": get_environment_metadata(),
    }


def run_serial_reference(
    nconformers: int,
    gradient_mode: str,
    weights_mode: str,
    fermi_mode: str,
) -> dict[str, Any]:
    """Compute independent single-System serial reference for tmpda conformers."""
    all_numbers, all_positions = conformers("tmpda", 8, sigma=0.1, seed=0)
    numbers = all_numbers[0].to(device=DEVICE)
    positions = all_positions[:nconformers].to(device=DEVICE, dtype=DTYPE)
    weights = _make_weights(nconformers, weights_mode)

    model, _ = setup_w2_model_and_system(numbers, gradient_mode=gradient_mode)
    model_param_items = tuple(
        (name, p)
        for name, p in model.par.named_parameters()
        if p.is_floating_point() and p.requires_grad
    )

    t0 = time.perf_counter()
    lane_records = []
    pos_grads = []
    root_xs = []
    energies = []
    fenergies = []
    param_accum: list[torch.Tensor | None] = [None] * len(model_param_items)

    for idx in range(nconformers):
        lane_t0 = time.perf_counter()
        sys_i = model.setup(numbers, batch_mode=0)
        pos_i = (
            positions[idx : idx + 1].detach().clone().requires_grad_(True)
        )
        prep_i = prepare_converged_root_batch(
            sys_i,
            pos_i.detach(),
            strict=True,
            eager_primal_fermi=fermi_mode,
        )
        lane_prep_s = time.perf_counter() - lane_t0
        diff_t0 = time.perf_counter()
        collector = ResponseDiagnosticCollector()
        res_i = evaluate_converged_root_batch(
            prep_i,
            positions=pos_i,
            strict=True,
            response_diagnostics=collector,
        )
        e_i = res_i.energy.reshape(-1).sum()
        fe_i = res_i.fenergy.reshape(-1).sum()
        w_i = weights[idx]
        if float(w_i.item()) == 0.0:
            g_pos_i = torch.zeros_like(pos_i[0])
            g_params_i = (None,) * len(model_param_items)
        else:
            grads_i = torch.autograd.grad(
                e_i * w_i,
                (pos_i, *(p for _, p in model_param_items)),
                allow_unused=True,
            )
            g_pos_i = grads_i[0][0].detach()
            g_params_i = grads_i[1:]
        lane_diff_s = time.perf_counter() - diff_t0

        for p_idx, g in enumerate(g_params_i):
            if g is not None:
                gd = g.detach()
                param_accum[p_idx] = (
                    gd.clone()
                    if param_accum[p_idx] is None
                    else param_accum[p_idx] + gd
                )

        diag_i = collector.finalize(
            [0] if float(w_i.item()) != 0.0 else [], 1, DEVICE, DTYPE
        )
        lane = prep_i.lanes[0]
        energies.append(float(e_i.detach().item()))
        fenergies.append(float(fe_i.detach().item()))
        pos_grads.append(g_pos_i.cpu().tolist())
        root_xs.append(lane.root_x.detach().cpu().tolist())
        lane_records.append(
            {
                "conformer_index": idx,
                "iterations": int(lane.iterations.item()),
                "raw_residual_l2": float(lane.raw_residual_l2.item()),
                "raw_residual_max": float(lane.raw_residual_max.item()),
                "electron_count_error": float(lane.electron_count_error.item()),
                "energy_eh": float(e_i.detach().item()),
                "free_energy_eh": float(fe_i.detach().item()),
                "unweighted_pos_grad_norm": float(
                    (
                        torch.linalg.vector_norm(g_pos_i / w_i).item()
                        if float(w_i.item()) != 0.0
                        else 0.0
                    )
                ),
                "weighted_pos_grad_norm": float(
                    torch.linalg.vector_norm(g_pos_i).item()
                ),
                "response_iterations": int(diag_i.iterations[0].item()),
                "response_residual": float(diag_i.residual[0].item()),
                "response_status": int(diag_i.status[0].item()),
                "prep_s": lane_prep_s,
                "diff_s": lane_diff_s,
            }
        )

    total_s = time.perf_counter() - t0
    connected_nonzero = sum(
        g is not None and bool((g.abs() > 0.0).any()) for g in param_accum
    )
    connected_zero = sum(
        g is not None and bool((g == 0.0).all()) for g in param_accum
    )
    disconnected = sum(g is None for g in param_accum)

    return {
        "stage": "serial-reference",
        "nconformers": nconformers,
        "gradient_mode": gradient_mode,
        "weights_mode": weights_mode,
        "fermi_mode": fermi_mode,
        "weights": weights.cpu().tolist(),
        "total_s": total_s,
        "peak_rss_mb": peak_rss_mb(),
        "energy_eh": energies,
        "free_energy_eh": fenergies,
        "root_xs": root_xs,
        "position_gradient": pos_grads,
        "lanes": lane_records,
        "model_parameters": {
            "total_leaves": len(model_param_items),
            "connected_nonzero": connected_nonzero,
            "connected_zero": connected_zero,
            "disconnected": disconnected,
            "leaf_names": [name for name, _ in model_param_items],
            "leaf_connected": [g is not None for g in param_accum],
            "leaf_grads": [
                None if g is None else g.cpu().tolist() for g in param_accum
            ],
        },
        "environment": get_environment_metadata(),
    }


def run_integrated(
    nconformers: int,
    fermi_mode: str,
    diff_mode: str,
    chunk_size: int,
    num_workers: int,
    gradient_mode: str,
    weights_mode: str,
    reference_json: Path | None,
) -> dict[str, Any]:
    """Run an isolated integrated physical-root + differentiation benchmark."""
    rss_monitor = RSSMonitor()
    rss_monitor.start()

    all_numbers, all_positions = conformers("tmpda", 8, sigma=0.1, seed=0)
    numbers = all_numbers[0].to(device=DEVICE)
    positions = (
        all_positions[:nconformers]
        .to(device=DEVICE, dtype=DTYPE)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    weights = _make_weights(nconformers, weights_mode)

    setup_t0 = time.perf_counter()
    model, system = setup_w2_model_and_system(
        numbers, gradient_mode=gradient_mode
    )
    setup_s = time.perf_counter() - setup_t0
    model_param_items = tuple(
        (name, p)
        for name, p in model.par.named_parameters()
        if p.is_floating_point() and p.requires_grad
    )

    rss_monitor.set_phase("root_preparation")
    root_t0 = time.perf_counter()
    prepared = prepare_converged_root_batch(
        system,
        positions.detach(),
        strict=True,
        eager_primal_fermi=fermi_mode,
        num_workers=num_workers,
    )
    root_prep_s = time.perf_counter() - root_t0
    rss_after_root = current_rss_mb()

    prep_diag = prepared.preparation_diagnostics
    prep_diag_dict = (
        None
        if prep_diag is None
        else {
            "num_workers": prep_diag.num_workers,
            "wall_time_s": prep_diag.wall_time_s,
            "parent_cpu_time_s": prep_diag.parent_cpu_time_s,
            "worker_cpu_time_s": prep_diag.worker_cpu_time_s,
            "total_cpu_time_s": prep_diag.total_cpu_time_s,
            "parent_peak_rss_mb": prep_diag.parent_peak_rss_mb,
            "worker_peak_rss_sum_mb": prep_diag.worker_peak_rss_sum_mb,
            "process_tree_peak_rss_mb": prep_diag.process_tree_peak_rss_mb,
            "process_tree_pss_mb": prep_diag.process_tree_pss_mb,
            "worker_pids": list(prep_diag.worker_pids),
        }
    )

    diff_t0 = time.perf_counter()
    if diff_mode == "simultaneous":
        rss_monitor.set_phase("forward")
        fwd_t0 = time.perf_counter()
        collector = ResponseDiagnosticCollector()
        result = evaluate_converged_root_batch(
            prepared,
            positions=positions,
            strict=True,
            response_diagnostics=collector,
        )
        forward_s = time.perf_counter() - fwd_t0
        rss_after_forward = current_rss_mb()

        lane_energy = result.energy.reshape(nconformers, -1).sum(-1)
        lane_free_energy = (
            result.fenergy.reshape(nconformers, -1).sum(-1)
            if result.fenergy is not None
            else lane_energy
        )
        objective = (weights * lane_energy).sum()

        rss_monitor.set_phase("backward")
        bwd_t0 = time.perf_counter()
        grads = torch.autograd.grad(
            objective,
            (positions, *(p for _, p in model_param_items)),
            allow_unused=True,
        )
        backward_s = time.perf_counter() - bwd_t0
        diff_or_stream_s = time.perf_counter() - diff_t0

        energy = lane_energy.detach()
        free_energy = lane_free_energy.detach()
        pos_grad = grads[0].detach()
        param_grads = tuple(
            None if g is None else g.detach() for g in grads[1:]
        )
        param_connected = tuple(g is not None for g in param_grads)
        diag = collector.finalize(
            list(range(nconformers)), nconformers, DEVICE, DTYPE
        )
        resp_converged = diag.converged.cpu().tolist()
        resp_status = diag.status.cpu().tolist()
        resp_iterations = diag.iterations.cpu().tolist()
        resp_residual = diag.residual.cpu().tolist()
        final_failure = [int(lane.failure_code) for lane in prepared.lanes]
        failure_category = list(
            classify_batch_failure_category(
                torch.tensor(final_failure, dtype=torch.int64, device=DEVICE)
            )
        )
    elif diff_mode == "streaming":
        rss_monitor.set_phase("streaming")
        streamed = stream_batch_value_and_grad(
            prepared,
            model,
            positions.detach(),
            weights=weights,
            chunk_size=chunk_size,
            strict=True,
        )
        diff_or_stream_s = time.perf_counter() - diff_t0
        forward_s = None
        backward_s = diff_or_stream_s
        rss_after_forward = current_rss_mb()

        energy = streamed.energy.detach()
        free_energy = streamed.free_energy.detach()
        pos_grad = streamed.position_gradient.detach()
        param_grads = streamed.model_parameter_gradients
        param_connected = streamed.model_parameter_connected
        resp_converged = streamed.response_converged.cpu().tolist()
        resp_status = streamed.response_status.cpu().tolist()
        resp_iterations = streamed.response_iterations.cpu().tolist()
        resp_residual = streamed.response_residual.cpu().tolist()
        final_failure = streamed.final_failure.cpu().tolist()
        failure_category = list(streamed.failure_category)
    else:
        raise ValueError(f"Unknown diff_mode '{diff_mode}'.")

    total_s = setup_s + root_prep_s + diff_or_stream_s
    rss_monitor.set_phase("finalize")
    phase_peaks = rss_monitor.close()
    proc_peak_rss = peak_rss_mb()
    overall_peak_rss = max(
        proc_peak_rss,
        0.0 if prep_diag is None else prep_diag.process_tree_peak_rss_mb,
    )

    connected_nonzero = sum(
        g is not None and bool((g.abs() > 0.0).any()) for g in param_grads
    )
    connected_zero = sum(
        g is not None and bool((g == 0.0).all()) for g in param_grads
    )
    disconnected = sum(g is None for g in param_grads)

    lanes_summary = []
    for idx, lane in enumerate(prepared.lanes):
        lanes_summary.append(
            {
                "conformer_index": idx,
                "converged": bool(lane.solver_converged.item()),
                "iterations": int(lane.iterations.item()),
                "raw_residual_l2": float(lane.raw_residual_l2.item()),
                "raw_residual_max": float(lane.raw_residual_max.item()),
                "electron_count_error": float(lane.electron_count_error.item()),
                "energy_eh": float(energy[idx].item()),
                "free_energy_eh": float(free_energy[idx].item()),
                "pos_grad_norm": float(
                    torch.linalg.vector_norm(pos_grad[idx]).item()
                ),
                "pos_grad_max_abs": float(pos_grad[idx].abs().max().item()),
                "response_converged": bool(resp_converged[idx]),
                "response_status": int(resp_status[idx]),
                "response_iterations": int(resp_iterations[idx]),
                "response_residual": float(resp_residual[idx]),
                "final_failure": int(final_failure[idx]),
                "failure_category": failure_category[idx],
            }
        )

    parity_summary = None
    if reference_json is not None and reference_json.exists():
        ref = json.loads(reference_json.read_text(encoding="utf-8"))
        ref_e = torch.tensor(
            ref["energy_eh"][:nconformers], dtype=DTYPE, device=DEVICE
        )
        ref_fe = torch.tensor(
            ref["free_energy_eh"][:nconformers], dtype=DTYPE, device=DEVICE
        )
        ref_pg = torch.tensor(
            ref["position_gradient"][:nconformers], dtype=DTYPE, device=DEVICE
        )
        ref_roots = [
            torch.tensor(r, dtype=DTYPE, device=DEVICE)
            for r in ref["root_xs"][:nconformers]
        ]
        root_diffs = [
            float((prepared.lanes[i].root_x.detach() - ref_roots[i]).abs().max().item())
            for i in range(nconformers)
        ]
        per_lane_e_diff = (energy - ref_e).abs().cpu().tolist()
        per_lane_pg_diff = [
            float((pos_grad[i] - ref_pg[i]).abs().max().item())
            for i in range(nconformers)
        ]
        parity_summary = {
            "root_charge_max_abs_diff": max(root_diffs),
            "per_lane_root_charge_max_abs_diff": root_diffs,
            "energy_max_abs_diff": float((energy - ref_e).abs().max().item()),
            "per_lane_energy_abs_diff": per_lane_e_diff,
            "free_energy_max_abs_diff": float(
                (free_energy - ref_fe).abs().max().item()
            ),
            "pos_grad_max_abs_diff": max(per_lane_pg_diff),
            "per_lane_pos_grad_max_abs_diff": per_lane_pg_diff,
        }
        if (
            gradient_mode == "positions-and-model"
            and ref.get("model_parameters", {}).get("leaf_grads")
            and ref.get("nconformers") == nconformers
        ):
            ref_pgrads = ref["model_parameters"]["leaf_grads"]
            param_max_abs = 0.0
            param_diff_sq = 0.0
            ref_norm_sq = 0.0
            meta_matches = True
            for g, rg_list in zip(param_grads, ref_pgrads):
                if (g is None) != (rg_list is None):
                    meta_matches = False
                elif g is not None and rg_list is not None:
                    rg = torch.tensor(rg_list, dtype=DTYPE, device=DEVICE)
                    d = (g - rg).abs()
                    param_max_abs = max(param_max_abs, float(d.max().item()))
                    param_diff_sq += float((d * d).sum().item())
                    ref_norm_sq += float((rg * rg).sum().item())
            parity_summary["model_param_connected_matches"] = meta_matches
            parity_summary["model_param_grad_max_abs_diff"] = param_max_abs
            parity_summary["model_param_grad_rel_l2_diff"] = (
                (param_diff_sq**0.5) / max(ref_norm_sq**0.5, 1e-30)
            )

    return {
        "stage": "integrated",
        "nconformers": nconformers,
        "fermi_mode": fermi_mode,
        "diff_mode": diff_mode,
        "chunk_size": chunk_size,
        "num_workers": num_workers,
        "gradient_mode": gradient_mode,
        "weights_mode": weights_mode,
        "weights": weights.cpu().tolist(),
        "timings_s": {
            "setup_s": setup_s,
            "root_prep_s": root_prep_s,
            "forward_s": forward_s,
            "backward_s": backward_s,
            "diff_or_stream_s": diff_or_stream_s,
            "total_s": total_s,
        },
        "preparation_diagnostics": prep_diag_dict,
        "memory_mib": {
            "rss_after_root": rss_after_root,
            "rss_after_forward": rss_after_forward,
            "process_peak_rss": proc_peak_rss,
            "overall_peak_tree_rss": overall_peak_rss,
            "phase_peak_rss": phase_peaks,
        },
        "energy_eh": energy.cpu().tolist(),
        "free_energy_eh": free_energy.cpu().tolist(),
        "root_xs": [
            lane.root_x.detach().cpu().tolist() for lane in prepared.lanes
        ],
        "position_gradient": pos_grad.cpu().tolist(),
        "position_gradient_norm": float(
            torch.linalg.vector_norm(pos_grad).item()
        ),
        "lanes": lanes_summary,
        "model_parameters": {
            "total_leaves": len(model_param_items),
            "connected_total": sum(param_connected),
            "connected_nonzero": connected_nonzero,
            "connected_zero": connected_zero,
            "disconnected": disconnected,
        },
        "parity_vs_reference": parity_summary,
        "environment": get_environment_metadata(),
    }


def run_legacy_audit(
    nconformers: int,
    scp_mode: str,
    route: str,
    reference_json: Path | None,
) -> dict[str, Any]:
    """Run legacy Calculator(batch_mode=2) or serial loop under explicit scp_mode."""
    rss_monitor = RSSMonitor()
    rss_monitor.start()

    all_numbers, all_positions = conformers("tmpda", 8, sigma=0.1, seed=0)
    numbers = all_numbers[:nconformers].to(device=DEVICE)
    positions = (
        all_positions[:nconformers]
        .to(device=DEVICE, dtype=DTYPE)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    opts = {
        "verbosity": 0,
        "scf_mode": "full",
        "mixer": "anderson",
        "scp_mode": scp_mode,
        "maxiter": 300,
        "f_atol": 1.0e-8,
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-5,
        "int_driver": "pytorch",
        "fermi_etemp": 300.0,
    }

    setup_t0 = time.perf_counter()
    if route == "legacy-packed":
        calc = Calculator(
            numbers,
            GFN1_XTB,
            opts=dict(opts, batch_mode=2),
            dtype=DTYPE,
            device=DEVICE,
        )
        setup_s = time.perf_counter() - setup_t0
        rss_monitor.set_phase("forward")
        fwd_t0 = time.perf_counter()
        charges = torch.zeros(nconformers, dtype=DTYPE, device=DEVICE)
        res = calc.singlepoint(positions, charges)
        forward_s = time.perf_counter() - fwd_t0
        rss_after_forward = current_rss_mb()

        lane_energy = res.energy.reshape(nconformers, -1).sum(-1)
        lane_fenergy = res.fenergy.reshape(nconformers, -1).sum(-1)
        rss_monitor.set_phase("backward")
        bwd_t0 = time.perf_counter()
        pos_grad = torch.autograd.grad(lane_energy.sum(), positions)[0].detach()
        backward_s = time.perf_counter() - bwd_t0
        iterations = int(res.iterations.item()) if res.iterations.ndim == 0 else res.iterations.cpu().tolist()
    elif route == "legacy-serial":
        setup_s = 0.0
        forward_s = 0.0
        backward_s = 0.0
        energies_list = []
        fenergies_list = []
        grads_list = []
        iters_list = []
        for idx in range(nconformers):
            s_t0 = time.perf_counter()
            calc_i = Calculator(
                numbers[0],
                GFN1_XTB,
                opts=opts,
                dtype=DTYPE,
                device=DEVICE,
            )
            setup_s += time.perf_counter() - s_t0
            pos_i = (
                positions[idx]
                .detach()
                .clone()
                .requires_grad_(True)
            )
            rss_monitor.set_phase("forward")
            f_t0 = time.perf_counter()
            res_i = calc_i.singlepoint(pos_i)
            forward_s += time.perf_counter() - f_t0
            e_i = res_i.energy.sum()
            fe_i = res_i.fenergy.sum()
            rss_monitor.set_phase("backward")
            b_t0 = time.perf_counter()
            g_i = torch.autograd.grad(e_i, pos_i)[0].detach()
            backward_s += time.perf_counter() - b_t0
            energies_list.append(e_i.detach())
            fenergies_list.append(fe_i.detach())
            grads_list.append(g_i)
            iters_list.append(int(res_i.iterations.item()))
        rss_after_forward = current_rss_mb()
        lane_energy = torch.stack(energies_list)
        lane_fenergy = torch.stack(fenergies_list)
        pos_grad = torch.stack(grads_list)
        iterations = iters_list
    else:
        raise ValueError(f"Unknown route '{route}'.")

    total_s = setup_s + forward_s + backward_s
    phase_peaks = rss_monitor.close()
    peak_rss = peak_rss_mb()

    parity_summary = None
    if reference_json is not None and reference_json.exists():
        ref = json.loads(reference_json.read_text(encoding="utf-8"))
        ref_e = torch.tensor(
            ref["energy_eh"][:nconformers], dtype=DTYPE, device=DEVICE
        )
        ref_fe = torch.tensor(
            ref["free_energy_eh"][:nconformers], dtype=DTYPE, device=DEVICE
        )
        ref_pg = torch.tensor(
            ref["position_gradient"][:nconformers], dtype=DTYPE, device=DEVICE
        )
        e_max_diff = float((lane_energy.detach() - ref_e).abs().max().item())
        fe_max_diff = float(
            (lane_fenergy.detach() - ref_fe).abs().max().item()
        )
        pg_max_diff = float((pos_grad - ref_pg).abs().max().item())
        pg_norm_diff = float(
            abs(
                torch.linalg.vector_norm(pos_grad).item()
                - torch.linalg.vector_norm(ref_pg).item()
            )
        )
        parity_summary = {
            "energy_max_abs_diff": e_max_diff,
            "energy_max_abs_diff_vs_functional": e_max_diff,
            "free_energy_max_abs_diff": fe_max_diff,
            "pos_grad_max_abs_diff_vs_functional": pg_max_diff,
            "position_gradient_norm_abs_diff": pg_norm_diff,
        }

    return {
        "stage": "legacy-audit",
        "route": route,
        "scp_mode": scp_mode,
        "nconformers": nconformers,
        "iterations": iterations,
        "timings_s": {
            "setup_s": setup_s,
            "forward_s": forward_s,
            "backward_s": backward_s,
            "total_s": total_s,
        },
        "memory_mib": {
            "rss_after_forward": rss_after_forward,
            "peak_rss": peak_rss,
            "phase_peak_rss": phase_peaks,
        },
        "energy_eh": lane_energy.detach().cpu().tolist(),
        "free_energy_eh": lane_fenergy.detach().cpu().tolist(),
        "position_gradient_norm": float(
            torch.linalg.vector_norm(pos_grad).item()
        ),
        "parity_vs_functional": parity_summary,
        "environment": get_environment_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("root-profile", "serial-reference", "integrated", "legacy-audit"),
        required=True,
    )
    parser.add_argument("--nconformers", type=int, default=1)
    parser.add_argument(
        "--fermi-mode",
        choices=("static", "adaptive", "spectral"),
        default="spectral",
    )
    parser.add_argument(
        "--diff-mode",
        choices=("simultaneous", "streaming"),
        default="streaming",
    )
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--gradient-mode",
        choices=("positions-only", "positions-and-model"),
        default="positions-only",
    )
    parser.add_argument(
        "--weights-mode",
        choices=("unit", "linspace", "mixed-sign-zero"),
        default="unit",
    )
    parser.add_argument(
        "--scp-mode",
        choices=("charge", "potential"),
        default="charge",
    )
    parser.add_argument(
        "--legacy-route",
        choices=("legacy-packed", "legacy-serial"),
        default="legacy-packed",
    )
    parser.add_argument("--reference-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if args.stage == "root-profile":
        report = run_root_profile(args.fermi_mode)
    elif args.stage == "serial-reference":
        report = run_serial_reference(
            args.nconformers,
            args.gradient_mode,
            args.weights_mode,
            args.fermi_mode,
        )
    elif args.stage == "integrated":
        report = run_integrated(
            args.nconformers,
            args.fermi_mode,
            args.diff_mode,
            args.chunk_size,
            args.num_workers,
            args.gradient_mode,
            args.weights_mode,
            args.reference_json,
        )
    elif args.stage == "legacy-audit":
        report = run_legacy_audit(
            args.nconformers,
            args.scp_mode,
            args.legacy_route,
            args.reference_json,
        )
    else:
        raise ValueError(f"Unsupported stage '{args.stage}'.")

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
