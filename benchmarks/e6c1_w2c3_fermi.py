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
"""Profile matrix-Fermi scaling and solve work on frozen W2 states.

The diagnostic captures states from the accepted eager W2 SCF controller,
then profiles the same raw SCF map at the initial, intermediate, and
converged states. It does not modify production SCF behavior.

Example
-------
Run the bounded tmpda profile in a fresh process::

    OMP_NUM_THREADS=1 python -u benchmarks/e6c1_w2c3_fermi.py \
        --mode tmpda-profile
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import dxtb._src.scf.functional.spectral as spectral  # noqa: E402
from benchmarks.baseline.workloads import conformers  # noqa: E402
from benchmarks.e6c1_w2c2_implicit import (  # noqa: E402
    DTYPE,
    PhysicalCase,
    prepare_call_inputs,
    setup_model_system,
)
from dxtb._src.scf.functional.anderson import _solve_map_anderson  # noqa: E402
from dxtb._src.scf.functional.eager import _solver_settings  # noqa: E402
from dxtb._src.scf.functional.map import scf_map  # noqa: E402

STARTED = time.perf_counter()


def rss_mib() -> float | None:
    """Return current process RSS in MiB when procfs is available."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def peak_rss_mib() -> float:
    """Return process lifetime peak RSS in MiB."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def emit(stage: str, **values: Any) -> None:
    """Write a flushed JSONL event with monotonic timing and memory."""
    print(
        json.dumps(
            {
                "stage": stage,
                "elapsed_s": time.perf_counter() - STARTED,
                "rss_current_mib": rss_mib(),
                "rss_peak_mib": peak_rss_mib(),
                **values,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def matrix_scale_metadata(
    matrix: torch.Tensor,
    shift: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, Any]:
    """Reproduce the production detached scale decisions for valid lanes."""
    identity = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    )
    difference = matrix - shift[..., None, None] * identity
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    exponent = torch.where(pair, difference / kt[..., None, None], 0.0)
    bound = exponent.detach().abs().sum(-1).amax(-1)
    required = torch.ceil(torch.log2(bound.clamp_min(1.0) / 4.0)).to(torch.long)
    base = torch.where(kt < 5.0e-4, 22, 11)
    selected = torch.maximum(base, required).clamp(min=0, max=30)
    scaled = (
        2.0
        * exponent
        / torch.pow(exponent.new_tensor(2.0), selected[..., None, None])
    )
    inner_norm = scaled.detach().abs().sum(-1).amax(-1)
    inner_power = (
        torch.ceil(torch.log2(inner_norm.clamp_min(0.25) / 0.25))
        .to(torch.long)
        .clamp(0, 8)
    )

    def host(value: torch.Tensor) -> Any:
        return value.detach().cpu().tolist()

    return {
        "shape": list(matrix.shape),
        "active_orbitals": host(active.sum(-1)),
        "exponent_inf_bound": host(bound),
        "required_power": host(required),
        "base_power": host(base),
        "selected_power": host(selected),
        "outer_active_doublings": host((selected - 1).clamp_min(0)),
        "outer_masked_doublings": host((30 - selected).clamp_min(0)),
        "inner_expm1_inf_bound": host(inner_norm),
        "inner_selected_power": host(inner_power),
    }


def profile_map(prepared: Any, state: torch.Tensor, label: str) -> None:
    """Run one no-grad SCF map and measure Fermi scale and dense solves."""
    from torch.profiler import ProfilerActivity, profile

    original_fermi = spectral._matrix_fermi
    original_solve = spectral.dense_solve
    calls: list[dict[str, Any]] = []
    current: list[dict[str, Any] | None] = [None]

    def measured_fermi(
        matrix: torch.Tensor,
        shift: torch.Tensor,
        kt: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        record = matrix_scale_metadata(matrix, shift, kt, active)
        record["dense_solve_calls"] = 0
        record["dense_solve_seconds"] = 0.0
        record["active_expm1_doublings"] = record["inner_selected_power"]
        record["masked_expm1_doublings"] = (
            [8 - int(value) for value in record["inner_selected_power"]]
            if isinstance(record["inner_selected_power"], list)
            else None
        )
        calls.append(record)
        current[0] = record
        return original_fermi(matrix, shift, kt, active)

    def measured_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        value = original_solve(matrix, rhs)
        record = current[0]
        if record is not None:
            record["dense_solve_calls"] += 1
            record["dense_solve_seconds"] += time.perf_counter() - started
        return value

    spectral._matrix_fermi = measured_fermi
    spectral.dense_solve = measured_solve
    try:
        started = time.perf_counter()
        with torch.no_grad(), profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            result = scf_map(state, prepared.inputs)
        duration = time.perf_counter() - started
    finally:
        spectral._matrix_fermi = original_fermi
        spectral.dense_solve = original_solve

    averages = profiler.key_averages()
    selected = [
        ("aten::bmm", "matrix_product"),
        ("aten::mm", "matrix_product"),
        ("aten::matmul", "matrix_product"),
        ("aten::linalg_lu_solve", "lu_solve"),
        ("aten::linalg_lu_factor_ex", "lu_factor"),
        ("aten::linalg_cholesky_ex", "cholesky_factor"),
    ]
    operations = {}
    for key, category in selected:
        matching = [event for event in averages if event.key == key]
        if matching:
            event = matching[0]
            entry = operations.setdefault(
                category, {"calls": 0, "self_cpu_seconds": 0.0}
            )
            entry["calls"] += event.count
            entry["self_cpu_seconds"] += event.self_cpu_time_total / 1.0e6
    emit(
        "matrix_fermi_profile",
        label=label,
        map_seconds=duration,
        density_finite=bool(torch.isfinite(result.density).all()),
        fermi_calls=calls,
        operation_totals=operations,
        dense_solve_total=sum(item["dense_solve_calls"] for item in calls),
        dense_solve_seconds_total=sum(
            item["dense_solve_seconds"] for item in calls
        ),
        profiler_total_matrix_product_calls=operations.get(
            "matrix_product", {}
        ).get("calls", 0),
    )


def run_tmpda_profile(args: argparse.Namespace) -> None:
    """Capture initial, intermediate, and converged frozen-W2 states."""
    numbers_batch, positions_batch = conformers("tmpda", 8, sigma=0.1, seed=0)
    numbers = numbers_batch[0]
    positions = positions_batch[0]
    model, system = setup_model_system(
        numbers,
        maxiter=300,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        gradient_mode="positions-only",
    )
    emit(
        "system_setup_completed",
        batch_mode=system.batch_mode,
        integral_level=system.config.ints.level,
        integral_setup=system.integral_setup is not None,
    )
    case = PhysicalCase(
        model=model,
        system=system,
        numbers=numbers,
        positions=positions,
        charge=torch.zeros(1, dtype=DTYPE),
        spin=None,
    )
    emit("call_input_preparation_started")
    prepared = prepare_call_inputs(case)
    emit(
        "call_input_preparation_completed",
        hcore_shape=list(prepared.integrals.hcore.shape),
        overlap_shape=list(prepared.integrals.overlap.shape),
    )
    snapshots: dict[str, torch.Tensor] = {
        "initial": prepared.initial_state.detach().clone()
    }
    count = 0

    def observed_map(state: torch.Tensor):
        nonlocal count
        count += 1
        if count == 1 or count % 20 == 0:
            emit("primal_map_progress", map_evaluations=count)
        if count in (25, 50, 75, 100, 125, 150):
            snapshots[f"map_evaluation_{count}"] = state.detach().clone()
        return scf_map(state, prepared.inputs)

    settings = _solver_settings(system.config.scf, DTYPE)
    started = time.perf_counter()
    with torch.no_grad():
        outcome = _solve_map_anderson(
            prepared.initial_state,
            settings,
            observed_map,
            eager=True,
        )
    emit(
        "primal_root_completed",
        duration_s=time.perf_counter() - started,
        map_evaluations=count,
        iterations=int(outcome.iterations.detach()),
        converged=bool(outcome.converged),
        residual=outcome.residual.detach().cpu().tolist(),
    )
    snapshots["converged_root"] = outcome.root_x.detach().clone()
    for label, state in snapshots.items():
        profile_map(prepared, state, label)


def run_solver_microbench(args: argparse.Namespace) -> None:
    """Compare the accepted LU solve with the existing Cholesky helper."""
    from torch.func import grad, jvp

    powers = tuple(int(item) for item in args.powers.split(","))
    sizes = tuple(int(item) for item in args.sizes.split(","))
    for size in sizes:
        for power in powers:
            bound = 4.0 * 2.0 ** (power - 0.5)
            spectrum = torch.linspace(-bound, bound, size, dtype=DTYPE)
            spectrum[:3] = torch.tensor([-0.7, 0.13, 1.3], dtype=DTYPE)
            direction = torch.linspace(-0.5, 0.5, size, dtype=DTYPE)
            direction[:3] = torch.tensor([0.8, -0.4, 0.3], dtype=DTYPE)
            weights = torch.linspace(-1.0, 1.0, size, dtype=DTYPE)
            active = torch.ones(size, dtype=torch.bool)
            shift = torch.tensor(0.0, dtype=DTYPE)
            kt = torch.tensor(1.0, dtype=DTYPE)

            def objective(theta: torch.Tensor) -> torch.Tensor:
                matrix = torch.diag(spectrum + theta * direction)
                density = spectral._matrix_fermi(matrix, shift, kt, active)
                return (density.diagonal() * weights).sum()

            def third_derivative() -> torch.Tensor:
                theta = torch.tensor(0.0, dtype=DTYPE)
                return grad(grad(grad(objective)))(theta)

            fermi = 1.0 / (1.0 + torch.exp(spectrum.clamp(-700.0, 700.0)))
            scalar_first = -fermi * (1.0 - fermi)
            scalar_second = fermi * (1.0 - fermi) * (1.0 - 2.0 * fermi)
            scalar_third = (
                -fermi
                * (1.0 - fermi)
                * (1.0 - 6.0 * fermi + 6.0 * fermi.square())
            )
            oracle_value = (weights * fermi).sum()
            oracle_first = (weights * direction * scalar_first).sum()
            oracle_second = (weights * direction.square() * scalar_second).sum()
            oracle_third = (weights * direction.pow(3) * scalar_third).sum()
            variants = (
                ("dense_solve", original_solver),
                ("cholesky_solve", spectral._solve_positive_definite),
            )
            for name, solver in variants:
                original = spectral.dense_solve
                spectral.dense_solve = solver
                try:
                    theta = torch.tensor(0.0, dtype=DTYPE)
                    started = time.perf_counter()
                    value = objective(theta)
                    value_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    first = grad(objective)(theta)
                    first_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    second = grad(grad(objective))(theta)
                    second_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    third = third_derivative()
                    third_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    _, tangent = jvp(
                        objective,
                        (theta,),
                        (torch.ones_like(theta),),
                    )
                    jvp_seconds = time.perf_counter() - started
                    times = []
                    with torch.no_grad():
                        objective(theta)
                        for _ in range(args.repeats):
                            started = time.perf_counter()
                            objective(theta)
                            times.append(time.perf_counter() - started)
                finally:
                    spectral.dense_solve = original
                torch.testing.assert_close(
                    value,
                    oracle_value,
                    atol=1.0e-10,
                    rtol=2.0e-8,
                )
                torch.testing.assert_close(
                    first,
                    oracle_first,
                    atol=1.0e-10,
                    rtol=2.0e-8,
                )
                torch.testing.assert_close(
                    second,
                    oracle_second,
                    atol=1.0e-10,
                    rtol=2.0e-8,
                )
                torch.testing.assert_close(
                    third,
                    oracle_third,
                    atol=1.0e-10,
                    rtol=2.0e-8,
                )
                torch.testing.assert_close(
                    tangent,
                    first,
                    atol=1.0e-12,
                    rtol=1.0e-10,
                )
                emit(
                    "solver_microbenchmark",
                    size=size,
                    target_power=power,
                    variant=name,
                    median_forward_s=statistics.median(times),
                    min_forward_s=min(times),
                    max_forward_s=max(times),
                    value_seconds=value_seconds,
                    first_derivative_seconds=first_seconds,
                    second_derivative_seconds=second_seconds,
                    third_derivative_seconds=third_seconds,
                    jvp_seconds=jvp_seconds,
                    value=float(value.detach()),
                    first=float(first.detach()),
                    second=float(second.detach()),
                    third=float(third.detach()),
                    jvp=float(tangent.detach()),
                    oracle_value=float(oracle_value.detach()),
                    oracle_first=float(oracle_first.detach()),
                    oracle_second=float(oracle_second.detach()),
                    oracle_third=float(oracle_third.detach()),
                    third_abs_error=float((third - oracle_third).abs()),
                    finite=bool(
                        torch.isfinite(value)
                        & torch.isfinite(first)
                        & torch.isfinite(second)
                        & torch.isfinite(third)
                        & torch.isfinite(tangent)
                    ),
                )


def original_solver(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Call the currently accepted upstream dense solver."""
    from tad_mctc.storch.linsolve import dense_solve

    return dense_solve(matrix, rhs)


def run_control_flow_probe() -> None:
    """Check eager/transform availability of tensor-dependent loop controls."""
    from torch.func import grad, jvp, vmap

    matrix = torch.eye(2, dtype=DTYPE)
    count = torch.tensor(2, dtype=torch.int64)

    def conditional(value: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        for index in range(3):
            value = torch.cond(
                steps > index,
                lambda item: item @ item,
                lambda item: item,
                (value,),
            )
        return value.sum()

    def dynamic_loop(value: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        index = torch.zeros((), dtype=torch.int64, device=value.device)

        def condition(i: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
            del carry
            return i < steps

        def body(
            i: torch.Tensor, carry: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return i + 1, carry @ carry

        return torch.while_loop(condition, body, (index, value))[1].sum()

    cases = (
        ("cond-eager", lambda: conditional(matrix, count)),
        ("cond-grad", lambda: grad(lambda x: conditional(x, count))(matrix)),
        (
            "cond-jvp",
            lambda: jvp(
                lambda x: conditional(x, count),
                (matrix,),
                (torch.ones_like(matrix),),
            ),
        ),
        (
            "cond-vmap",
            lambda: vmap(conditional)(
                matrix.expand(2, -1, -1), torch.tensor([1, 2])
            ),
        ),
        ("while-eager", lambda: dynamic_loop(matrix, count)),
        ("while-grad", lambda: grad(lambda x: dynamic_loop(x, count))(matrix)),
        (
            "while-jvp",
            lambda: jvp(
                lambda x: dynamic_loop(x, count),
                (matrix,),
                (torch.ones_like(matrix),),
            ),
        ),
        (
            "while-vmap",
            lambda: vmap(dynamic_loop)(
                matrix.expand(2, -1, -1), torch.tensor([1, 2])
            ),
        ),
    )
    for label, operation in cases:
        try:
            value = operation()
            emit(
                "control_flow_probe",
                case=label,
                result="pass",
                value=str(value),
            )
        except Exception as error:
            emit(
                "control_flow_probe",
                case=label,
                result="unsupported",
                error_type=type(error).__name__,
                error=str(error).splitlines()[0],
            )


def run_mixed_power_vmap_check() -> None:
    """Check mixed adaptive powers and cross-lane derivative isolation."""
    from torch.func import jacrev, vmap

    powers = torch.tensor([14, 22, 30])
    bounds = 4.0 * torch.pow(2.0, powers.to(DTYPE) - 0.5)
    base = torch.linspace(-1.0, 1.0, 32, dtype=DTYPE).expand(3, -1).clone()
    base[:, 0] = -bounds
    base[:, 1] = 0.13
    direction = torch.linspace(-0.5, 0.5, 32, dtype=DTYPE)
    weights = torch.linspace(-1.0, 1.0, 32, dtype=DTYPE)
    active = torch.ones(32, dtype=torch.bool)

    def evaluate_lane(
        theta: torch.Tensor, spectrum: torch.Tensor
    ) -> torch.Tensor:
        density = spectral._matrix_fermi(
            torch.diag(spectrum + theta * direction),
            torch.tensor(0.0, dtype=DTYPE),
            torch.tensor(1.0, dtype=DTYPE),
            active,
        )
        return (density.diagonal() * weights).sum()

    for name, solver in (
        ("dense_solve", original_solver),
        ("cholesky_solve", spectral._solve_positive_definite),
    ):
        original = spectral.dense_solve
        spectral.dense_solve = solver
        try:
            mapped_function = lambda theta: vmap(evaluate_lane)(theta, base)
            theta = torch.zeros(3, dtype=DTYPE)
            mapped_values = mapped_function(theta)
            serial_values = torch.stack(
                [
                    evaluate_lane(torch.tensor(0.0, dtype=DTYPE), base[i])
                    for i in range(3)
                ]
            )
            jacobian = jacrev(mapped_function)(theta)
            off_diagonal = jacobian - torch.diag(jacobian.diagonal())
            torch.testing.assert_close(
                mapped_values,
                serial_values,
                atol=5.0e-11,
                rtol=2.0e-8,
            )
            assert torch.isfinite(jacobian).all()
            assert off_diagonal.abs().max() == 0
        finally:
            spectral.dense_solve = original
        emit(
            "mixed_power_vmap",
            solver=name,
            selected_powers=powers.tolist(),
            values=mapped_values.detach().tolist(),
            jacobian=jacobian.detach().tolist(),
            cross_lane_max_abs=float(off_diagonal.abs().max()),
            scalar_comparison="pass",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "tmpda-profile",
            "solver-microbench",
            "control-flow-probe",
            "mixed-power-vmap",
        ),
        default="tmpda-profile",
    )
    parser.add_argument("--sizes", default="32,128,255")
    parser.add_argument("--powers", default="16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    emit(
        "environment",
        mode=args.mode,
        python=sys.version.split()[0],
        torch=torch.__version__,
        threads=torch.get_num_threads(),
    )
    if args.mode == "tmpda-profile":
        run_tmpda_profile(args)
    elif args.mode == "solver-microbench":
        run_solver_microbench(args)
    elif args.mode == "control-flow-probe":
        run_control_flow_probe()
    else:
        run_mixed_power_vmap_check()


if __name__ == "__main__":
    main()
