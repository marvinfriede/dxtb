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
"""Profile the public one-System SCF route at bounded fixed trip counts.

Run each route/mode in a fresh process for isolated RSS measurements::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python benchmarks/e6c1_w2c8_phase_profile.py --route vmap \
      --mode backward --maxiter 1

This diagnostic intentionally permits unconverged roots. It does not claim
physical validity or meaningful forces for the one-trip cases.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import dxtb  # noqa: E402
from benchmarks.baseline.workloads import conformers  # noqa: E402
from dxtb import GFN1_XTB, ParamModule, labels  # noqa: E402
from dxtb._src.calculators.model import Model  # noqa: E402
from dxtb._src.calculators.result import Result  # noqa: E402
from dxtb.config import Config  # noqa: E402

DTYPE = torch.float64


def emit(event: str, **payload: Any) -> None:
    """Write one machine-readable progress event."""
    print(json.dumps({"event": event, **payload}, default=str), flush=True)


def current_rss_mb() -> float | None:
    """Read current resident memory from procfs when available."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def peak_rss_mb() -> float:
    """Return the process lifetime RSS high-water mark in MiB."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / 1024**2


def cpu_model() -> str | None:
    """Return the CPU model string from procfs when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def build_system(maxiter: int) -> tuple[Any, torch.Tensor, int]:
    """Build the frozen conformer-zero functional System."""
    numbers, positions = conformers("tmpda", 8, sigma=0.1, seed=0)
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        mixer=labels.MIXER_ANDERSON,
        scf_mode=labels.SCF_MODE_FULL,
        scp_mode=labels.SCP_MODE_CHARGE,
        maxiter=maxiter,
        f_atol=1.0e-8,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        fermi_etemp=300.0,
        force_convergence=False,
    )
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=DTYPE),
        config=config,
        auto_int_level=True,
    )
    for parameter in model.par.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(False)
    trainable_parameters = sum(
        parameter.is_floating_point() and parameter.requires_grad
        for parameter in model.par.parameters()
    )
    system = model.setup(numbers[0], batch_mode=0)
    return system, positions[0].to(dtype=DTYPE), trainable_parameters


@contextmanager
def timed_patches() -> Iterator[dict[str, dict[str, float | int]]]:
    """Time call-local production boundaries and restore all patches."""
    import dxtb._src.calculators.singlepoint as singlepoint_module
    import dxtb._src.scf.functional.anderson as anderson_module
    import dxtb._src.scf.functional.eager as eager_module
    import dxtb._src.scf.functional.map as map_module
    import dxtb._src.scf.functional.solver as solver_module
    from dxtb._src.scf.functional import spectral as spectral_module
    from dxtb._src.wavefunction import filling

    rows: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"seconds": 0.0, "calls": 0}
    )
    patches: list[tuple[Any, str, Any]] = []
    raw_map_count = 0

    def patch(owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        def measured(*args: Any, **kwargs: Any) -> Any:
            nonlocal raw_map_count
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                row_label = label
                if label == "raw_scf_map":
                    row_label = (
                        "initial_raw_scf_map"
                        if raw_map_count == 0
                        else "iteration_raw_scf_map"
                    )
                    raw_map_count += 1
                rows[row_label]["seconds"] += time.perf_counter() - started
                rows[row_label]["calls"] += 1

        patches.append((owner, name, original))
        setattr(owner, name, measured)

    patch(singlepoint_module, "build_integral_matrices", "integral_preparation")
    patch(singlepoint_module, "_interaction_data", "interaction_preparation")
    patch(
        singlepoint_module, "_evaluate_classical_descriptor", "classical_energy"
    )
    patch(
        eager_module,
        "run_system_fixed_trip",
        "scf_solver_and_finalization_inclusive",
    )
    patch(anderson_module, "scf_map", "raw_scf_map")
    patch(map_module, "potential_from_charges", "interaction_potential")
    patch(map_module, "potential_to_hamiltonian", "hamiltonian_build")
    patch(map_module, "_transitional_diagonalize", "spectral_diagonalization")
    (
        patch(
            map_module,
            "_get_fermi_occupation_core_state",
            "fermi_occupation_state",
        )
        if hasattr(map_module, "_get_fermi_occupation_core_state")
        else None
    )
    patch(
        filling,
        "_get_fermi_occupation_core_state",
        "fermi_occupation_state",
    )
    patch(map_module, "fermi_density_matrix", "fermi_density")
    patch(
        solver_module, "free_energy_from_spectral", "free_energy_construction"
    )
    patch(solver_module, "fermi_free_energy_trace", "fermi_free_energy_trace")
    patch(solver_module, "fermi_free_energy_matrix", "fermi_free_energy_matrix")
    patch(spectral_module, "_binary_entropy_matrix", "binary_entropy_matrix")
    snapshot = Result.snapshot.__func__

    def measured_snapshot(
        cls: type[Result], *args: Any, **kwargs: Any
    ) -> Result:
        started = time.perf_counter()
        try:
            return snapshot(cls, *args, **kwargs)
        finally:
            rows["result_assembly"]["seconds"] += time.perf_counter() - started
            rows["result_assembly"]["calls"] += 1

    patches.append((Result, "snapshot", classmethod(snapshot)))
    Result.snapshot = classmethod(measured_snapshot)
    try:
        yield rows
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def run_call(
    system: Any,
    positions: torch.Tensor,
    route: str,
    mode: str,
) -> dict[str, Any]:
    """Execute direct or one-lane-vmapped Result and optional gradient."""
    input_positions = (
        positions.detach().clone().requires_grad_(mode == "backward")
    )
    started = time.perf_counter()
    if route == "direct":
        result = system.singlepoint(input_positions)
        energy = result.energy.sum()
    else:
        batch = input_positions.unsqueeze(0)
        result = torch.func.vmap(system.singlepoint, in_dims=(0,))(batch)
        energy = result.energy.sum()
    forward_seconds = time.perf_counter() - started
    backward_seconds = None
    gradient_summary = None
    if mode == "backward":
        emit("backward_started", route=route, rss_mb=current_rss_mb())
        started = time.perf_counter()
        (gradient,) = torch.autograd.grad(energy, input_positions)
        backward_seconds = time.perf_counter() - started
        gradient_summary = {
            "finite": bool(torch.isfinite(gradient).all()),
            "norm": float(torch.linalg.vector_norm(gradient.detach())),
            "max_abs": float(gradient.detach().abs().max()),
            "values": gradient.detach().cpu().tolist(),
        }
        emit(
            "backward_completed",
            route=route,
            seconds=backward_seconds,
            rss_mb=current_rss_mb(),
        )
    if route == "vmap":
        energy_value = energy.detach()
        density = result.density.detach()[0]
        iterations = result.iterations.detach()[0]
        converged = (
            None if result.converged is None else result.converged.detach()[0]
        )
        residual = (
            None if result.residual is None else result.residual.detach()[0]
        )
        fenergy = result.fenergy.detach()[0]
    else:
        energy_value = energy.detach()
        density = result.density.detach()
        iterations = result.iterations.detach()
        converged = (
            None if result.converged is None else result.converged.detach()
        )
        residual = None if result.residual is None else result.residual.detach()
        fenergy = result.fenergy.detach()
    return {
        "route": route,
        "mode": mode,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "energy_hartree": float(energy_value),
        "fenergy_field_hartree": float(fenergy.sum()),
        "density_finite": bool(torch.isfinite(density).all()),
        "density_norm": float(torch.linalg.vector_norm(density)),
        "iterations": int(iterations),
        "converged": None if converged is None else bool(converged),
        "residual": None if residual is None else residual.cpu().tolist(),
        "gradient": gradient_summary,
    }


def main() -> None:
    """Run one bounded public-route timing experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("direct", "vmap"), required=True)
    parser.add_argument(
        "--mode", choices=("forward", "backward"), required=True
    )
    parser.add_argument("--maxiter", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be explicitly set to 1")
    if args.threads != 1:
        raise RuntimeError("W2c8 profiling uses one PyTorch intra-op thread")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    source = (ROOT / "src/dxtb/__init__.py").resolve()
    if Path(dxtb.__file__).resolve() != source:
        raise RuntimeError("dxtb import does not resolve to this worktree")
    from dxtb._src.wavefunction import filling

    emit(
        "environment",
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        source=str(Path(dxtb.__file__).resolve()),
        python=platform.python_version(),
        torch=torch.__version__,
        cpu_model=cpu_model(),
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        thread_environment={
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        dependencies={
            name: importlib.metadata.version(name)
            for name in (
                "tad-mctc",
                "tad-multicharge",
                "tad-libcint",
                "tad-dftd3",
                "tad-dftd4",
            )
        },
    )
    setup_started = time.perf_counter()
    system, positions, trainable_parameters = build_system(args.maxiter)
    emit(
        "setup_completed",
        seconds=time.perf_counter() - setup_started,
        rss_mb=current_rss_mb(),
        maxiter=system.config.scf.maxiter,
        scp_mode=system.config.scf.scp_mode,
        mixer=system.config.scf.mixer,
        x_atol=system.config.scf.x_atol,
        x_atol_max=system.config.scf.x_atol_max,
        f_atol=system.config.scf.f_atol,
        smear=system.config.scf.fermi.etemp > 0,
        partition=system.config.scf.fermi.partition,
        temperature_kelvin=system.config.scf.fermi.etemp,
        diff_order=system.config.scf.fermi.diff_order,
        diff_steps=filling._diff_steps(system.config.scf.fermi.diff_order),
        fermi_partition_name=labels.FERMI_PARTITION_MAP[
            system.config.scf.fermi.partition
        ],
        trainable_floating_model_parameters=trainable_parameters,
    )
    start_rss = current_rss_mb()
    with timed_patches() as timings:
        result = run_call(system, positions, args.route, args.mode)
    total = result["forward_seconds"] + (result["backward_seconds"] or 0.0)
    emit(
        "profile_completed",
        **result,
        total_seconds=total,
        stages=dict(timings),
        start_rss_mb=start_rss,
        final_rss_mb=current_rss_mb(),
        peak_rss_mb=peak_rss_mb(),
        note="maxiter below convergence is diagnostic only; no physical-root claim",
        binary_entropy_matrix_calls=int(
            timings.get("binary_entropy_matrix", {}).get("calls", 0)
        ),
    )


if __name__ == "__main__":
    main()
