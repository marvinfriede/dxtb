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
"""Stage-level diagnostic for the accepted E6c1-W2 System path.

This script emits flushed JSON Lines to stdout. Run it in a bounded fresh
process and redirect stdout/stderr to a persistent file so the last stage
start survives a timeout. It instruments the production call with temporary,
process-local wrappers; all numerical functions and inputs remain unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

SEQUENCE = 0
RUN_START = time.perf_counter()
ROUTE = "unknown"
BACKWARD_REQUESTED = False
GRAD_MODE_REQUESTED = "enabled"
DETAILS = False
JSON_STREAM = sys.stdout


def _memory() -> dict[str, float | None]:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    try:
        fields = Path("/proc/self/statm").read_text().split()
        current = int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2
    except (OSError, ValueError, IndexError):
        current = None
    return {"rss_current_mb": current, "rss_peak_mb": peak}


def emit(stage: str, event: str, **fields: Any) -> None:
    global SEQUENCE
    SEQUENCE += 1
    record = {
        "sequence": SEQUENCE,
        "elapsed_from_process_start_s": time.perf_counter() - RUN_START,
        "route": ROUTE,
        "backward_requested": BACKWARD_REQUESTED,
        "grad_mode_requested": GRAD_MODE_REQUESTED,
        "stage": stage,
        "event": event,
        **_memory(),
        **fields,
    }
    print(
        json.dumps(record, default=str, separators=(",", ":")),
        file=JSON_STREAM,
        flush=True,
    )


@contextlib.contextmanager
def measured(stage: str, **fields: Any):
    start = time.perf_counter()
    emit(stage, "start", **fields)
    try:
        yield
    except BaseException as error:
        emit(
            stage,
            "failed",
            duration_s=time.perf_counter() - start,
            exception_type=type(error).__name__,
            exception=str(error),
        )
        raise
    else:
        emit(stage, "complete", duration_s=time.perf_counter() - start)


def _wrap(target: object, name: str, stage: str, details: dict[str, Any]):
    original = getattr(target, name)

    def wrapped(*args: Any, **kwargs: Any):
        with measured(stage, **details):
            return original(*args, **kwargs)

    setattr(target, name, wrapped)
    return original


@contextlib.contextmanager
def production_instrumentation():
    """Add temporary timers around actual production call boundaries."""
    from dxtb._src.calculators import singlepoint
    from dxtb._src.scf.functional import eager, map, solver, spectral
    from dxtb._src.wavefunction import filling

    restorers: list[tuple[object, str, object]] = []
    operator_totals: dict[str, dict[str, float | int]] = {}

    def patch(target: object, name: str, stage: str):
        original = _wrap(target, name, stage, {})
        restorers.append((target, name, original))

    patch(singlepoint, "build_integral_matrices", "integral_and_h0_build")
    patch(singlepoint, "_interaction_data", "initial_interaction_setup")
    patch(singlepoint, "_evaluate_classical_descriptor", "classical_component")
    patch(eager, "run_system_fixed_trip", "functional_scf_boundary")

    patch(eager, "get_refocc", "initial_scf_reference_occupations")
    patch(eager, "get_guess", "initial_scf_guess")
    patch(eager, "setup_scf_inputs", "initial_scf_input_setup")
    patch(eager, "_solve_representation", "forward_scf_solver")
    patch(eager, "functional_core_result", "functional_result_materialization")
    patch(map, "evaluate_hamiltonian", "spectral_evaluation")
    patch(map, "fermi_density_matrix", "fermi_density_matrix")
    patch(filling, "_get_fermi_occupation_core_state", "fermi_state_and_mu")
    patch(solver, "finalize_scf", "solver_finalization")
    patch(solver, "free_energy_from_spectral", "fermi_free_energy_finalization")
    patch(solver, "fermi_free_energy_trace", "fermi_free_energy_trace")
    fermi_setup = spectral._fermi_free_energy_setup

    def timed_fermi_setup(*args: Any, **kwargs: Any):
        with measured("fermi_free_energy_setup"):
            result = fermi_setup(*args, **kwargs)
        if DETAILS and ROUTE == "serial":
            emit(
                "fermi_free_energy_setup_output",
                "snapshot",
                x=_finite_stats(result.x),
                occupation=_finite_stats(result.occupation),
                shift=_finite_stats(result.shift),
                spectral=_finite_stats(result.spectral),
                ref=_finite_stats(result.ref),
                offset=_finite_stats(result.offset),
                target=_finite_stats(result.target),
                empty=result.empty.detach().tolist(),
                full=result.full.detach().tolist(),
                too_many=result.too_many.detach().tolist(),
                gap_mask=result.gap_mask.detach().tolist(),
                shift_values=result.shift.detach().tolist(),
                electron_count_values=result.target.detach().tolist(),
            )
        return result

    spectral._fermi_free_energy_setup = timed_fermi_setup
    restorers.append((spectral, "_fermi_free_energy_setup", fermi_setup))
    matrix_fermi = spectral._matrix_fermi

    def timed_matrix_fermi(matrix, shift, kt, active):
        start = time.perf_counter()
        try:
            result = matrix_fermi(matrix, shift, kt, active)
        finally:
            totals = operator_totals.setdefault(
                "matrix_fermi", {"calls": 0, "duration_s": 0.0}
            )
            totals["calls"] += 1
            totals["duration_s"] += time.perf_counter() - start
        if DETAILS and ROUTE == "serial":
            count = result.diagonal(dim1=-2, dim2=-1).sum(-1)
            slope = (result - result @ result).diagonal(dim1=-2, dim2=-1).sum(
                -1
            ) / kt
            emit(
                "matrix_fermi_output",
                "snapshot",
                matrix=_finite_stats(matrix),
                shift=_finite_stats(shift),
                shift_values=shift.detach().tolist(),
                kt=_finite_stats(kt),
                count_values=count.detach().tolist(),
                slope_values=slope.detach().tolist(),
                result=_finite_stats(result),
            )
        return result

    spectral._matrix_fermi = timed_matrix_fermi
    restorers.append((spectral, "_matrix_fermi", matrix_fermi))
    patch(spectral, "_gap_electron_response", "gap_electron_response")
    patch(spectral, "attach_gradient", "attach_gradient")

    trace_logcosh = spectral._trace_logcosh

    def timed_trace_logcosh(matrix):
        details: dict[str, Any] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "device": str(matrix.device),
        }
        if DETAILS and ROUTE == "serial":
            details["finite"] = bool(torch.isfinite(matrix).all())
            details["max_abs"] = float(matrix.abs().amax())
            details["frobenius_norm"] = float(torch.linalg.vector_norm(matrix))
            details["symmetry_error"] = float(
                (matrix - matrix.transpose(-2, -1)).abs().amax()
            )
        emit("trace_logcosh_input", "snapshot", **details)
        with measured("trace_logcosh"):
            return trace_logcosh(matrix)

    spectral._trace_logcosh = timed_trace_logcosh
    restorers.append((spectral, "_trace_logcosh", trace_logcosh))

    matrix_exp = spectral.torch.matrix_exp

    def timed_matrix_exp(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            return matrix_exp(*args, **kwargs)
        finally:
            totals = operator_totals.setdefault(
                "torch.matrix_exp", {"calls": 0, "duration_s": 0.0}
            )
            totals["calls"] += 1
            totals["duration_s"] += time.perf_counter() - start

    spectral.torch.matrix_exp = timed_matrix_exp
    restorers.append((spectral.torch, "matrix_exp", matrix_exp))

    dense_solve = spectral.dense_solve

    def timed_dense_solve(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            return dense_solve(*args, **kwargs)
        finally:
            totals = operator_totals.setdefault(
                "dense_solve", {"calls": 0, "duration_s": 0.0}
            )
            totals["calls"] += 1
            totals["duration_s"] += time.perf_counter() - start

    spectral.dense_solve = timed_dense_solve
    restorers.append((spectral, "dense_solve", dense_solve))

    snapshot_descriptor = singlepoint.Result.__dict__["snapshot"]
    snapshot = snapshot_descriptor.__func__

    def timed_snapshot(cls, **kwargs):
        with measured("result_materialization"):
            return snapshot(cls, **kwargs)

    singlepoint.Result.snapshot = classmethod(timed_snapshot)
    restorers.append((singlepoint.Result, "snapshot", snapshot_descriptor))

    try:
        yield
    finally:
        for operator, totals in sorted(operator_totals.items()):
            emit(operator, "aggregate", **totals)
        for target, name, original in reversed(restorers):
            if target is singlepoint.Result:
                setattr(target, name, original)
            else:
                setattr(target, name, original)


def _tensor_summary(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "finite": bool(torch.isfinite(detached).all()),
        "sum": float(detached.sum()),
        "norm": (
            float(torch.linalg.vector_norm(detached))
            if detached.is_floating_point() or detached.is_complex()
            else None
        ),
        "values": (
            detached.cpu().tolist() if detached.numel() <= 1024 else None
        ),
    }


def _finite_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "shape": list(detached.shape),
        "finite_count": int(finite.sum()),
        "element_count": detached.numel(),
        "finite_abs_max": (
            float(finite_values.abs().amax()) if finite_values.numel() else None
        ),
    }


def main() -> None:
    global BACKWARD_REQUESTED, DETAILS, GRAD_MODE_REQUESTED, ROUTE, torch

    with measured("python_and_torch_import"):
        import torch

    with measured("dxtb_import"):
        from molecules import mols

        from benchmarks.e6c1_w2 import DTYPE, conformers
        from dxtb import GFN1_XTB, ParamModule, labels
        from dxtb._src.calculators import singlepoint
        from dxtb._src.calculators.model import Model
        from dxtb._src.scf.functional import eager, spectral
        from dxtb.config import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("serial", "mapped"), required=True)
    parser.add_argument("--nconformers", type=int, default=1)
    parser.add_argument("--molecule", choices=("tmpda", "H2O"), default="tmpda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--no-grad", action="store_true")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    if args.no_grad and args.backward:
        parser.error("--backward cannot be combined with --no-grad")
    ROUTE = args.route
    BACKWARD_REQUESTED = args.backward
    GRAD_MODE_REQUESTED = "disabled" if args.no_grad else "enabled"
    DETAILS = args.details

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.molecule == "tmpda":
        numbers, positions = conformers(args.nconformers)
    else:
        molecule = mols[args.molecule]
        numbers = molecule["numbers"]
        reference_positions = molecule["positions"].double()
        generator = torch.Generator().manual_seed(0)
        displacements = (
            0.1
            * torch.randn(
                (args.nconformers, *reference_positions.shape),
                generator=generator,
            ).double()
        )
        displacements[0] = 0.0
        positions = reference_positions.unsqueeze(0) + displacements
    numbers = numbers.to(dtype=torch.int64, device="cpu")
    positions = positions.to(dtype=DTYPE, device="cpu")
    emit(
        "environment",
        "snapshot",
        python=platform.python_version(),
        torch=torch.__version__,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        omp_num_threads=os.getenv("OMP_NUM_THREADS"),
        mctc=importlib.metadata.version("tad-mctc"),
        multicharge=importlib.metadata.version("tad-multicharge"),
        libcint=importlib.metadata.version("tad-libcint"),
        d3=importlib.metadata.version("tad-dftd3"),
        d4=importlib.metadata.version("tad-dftd4"),
        natoms=int(numbers.numel()),
        molecule=args.molecule,
        nconformers=args.nconformers,
        maxiter=args.maxiter,
    )

    config = Config.create(
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

    with measured("model_parameter_construction"):
        parameters = ParamModule(GFN1_XTB, dtype=DTYPE)
    with contextlib.redirect_stdout(sys.stderr):
        with measured("system_exact_core_setup", natoms=int(numbers.numel())):
            system = Model(par=parameters, config=config).setup(numbers)

    position_input = positions.detach().clone().requires_grad_(args.backward)
    grad_context = torch.no_grad() if args.no_grad else torch.enable_grad()
    with production_instrumentation(), contextlib.redirect_stdout(sys.stderr):
        with grad_context:
            with measured(
                "singlepoint_forward",
                positions_shape=list(position_input.shape),
            ):
                if args.route == "mapped":
                    results = torch.func.vmap(system.singlepoint, in_dims=(0,))(
                        position_input
                    )
                    energy = results.energy.sum()
                    energy_by_lane = results.energy.sum(-1)
                    fenergy_by_lane = results.fenergy.sum(-1)
                    iterations = results.iterations
                    converged = results.converged
                    residual = results.residual
                else:
                    results = tuple(
                        system.singlepoint(position)
                        for position in position_input
                    )
                    energy_by_lane = torch.stack(
                        tuple(result.energy.sum() for result in results)
                    )
                    fenergy_by_lane = torch.stack(
                        tuple(result.fenergy.sum() for result in results)
                    )
                    energy = energy_by_lane.sum()
                    iterations = torch.stack(
                        tuple(result.iterations for result in results)
                    )
                    converged = torch.stack(
                        tuple(result.converged for result in results)
                    )
                    residual = torch.stack(
                        tuple(result.residual for result in results)
                    )

            emit(
                "physical_result",
                "snapshot",
                energy_requires_grad=energy.requires_grad,
                positions_requires_grad=position_input.requires_grad,
                energy=_tensor_summary(energy_by_lane),
                free_energy=_tensor_summary(fenergy_by_lane),
                iterations=_tensor_summary(iterations),
                converged=_tensor_summary(converged),
                residual=_tensor_summary(residual),
            )
            if args.backward:
                with measured("backward_position_gradient"):
                    (gradient,) = torch.autograd.grad(energy, position_input)
                emit(
                    "position_gradient",
                    "snapshot",
                    summary=_tensor_summary(gradient),
                )


if __name__ == "__main__":
    main()
