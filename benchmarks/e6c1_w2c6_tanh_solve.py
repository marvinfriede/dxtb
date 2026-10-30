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
"""Compare equivalent matrix-Fermi tanh-doubling solve formulations.

Run each method in a fresh process with identical arguments to keep the RSS
high-water mark attributable to that method::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/e6c1_w2c6_tanh_solve.py \
        --method direct-rhs --dimension 255 --power 16 \
        --output benchmarks/results/e6c1-w2c6/direct-rhs-n255-p16.jsonl
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from tad_mctc.storch.linsolve import dense_solve

import dxtb  # noqa: E402
from dxtb._src.scf.functional.spectral import (  # noqa: E402
    _FERMI_MAX_SCALE_POWER,
    _FERMI_MAX_SCALED_NORM,
    _matrix_expm1,
    _matrix_fermi,
)

DTYPE = torch.float64
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _matrix_fermi_with_rhs(
    matrix: torch.Tensor,
    shift: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
    *,
    direct_rhs: bool,
) -> torch.Tensor:
    """Evaluate matrix-Fermi density with a selected doubling RHS form."""
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    eye = eye.expand(matrix.shape)
    active_pair = active.unsqueeze(-1) & active.unsqueeze(-2)

    active_matrix = torch.where(active_pair, matrix, torch.zeros_like(matrix))
    active_any = active.any(-1)
    finite_matrix = torch.where(
        active_pair, torch.isfinite(matrix), torch.ones_like(active_pair)
    ).all(dim=(-1, -2))
    finite_shift_kt = torch.isfinite(shift) & torch.isfinite(kt) & (kt > 0.0)
    finite_inputs = finite_matrix & torch.where(
        active_any, finite_shift_kt, torch.ones_like(active_any)
    )
    safe_matrix = torch.where(
        finite_inputs[..., None, None], active_matrix, torch.zeros_like(matrix)
    )
    safe_shift = torch.where(
        finite_inputs & active_any, shift, torch.zeros_like(shift)
    )
    safe_kt = torch.where(finite_inputs & active_any, kt, torch.ones_like(kt))

    diagonal = safe_matrix.diagonal(dim1=-2, dim2=-1)
    shift_diagonal = safe_shift.unsqueeze(-1)
    opposite_signs = (diagonal < 0.0) & (shift_diagonal > 0.0)
    opposite_signs = opposite_signs | (
        (diagonal > 0.0) & (shift_diagonal < 0.0)
    )
    maximum_value = torch.finfo(matrix.dtype).max
    subtraction_overflow = (
        active
        & opposite_signs
        & (diagonal.abs() > maximum_value - safe_shift.abs().unsqueeze(-1))
    ).any(-1)
    safe_inputs = finite_inputs & ~subtraction_overflow
    safe_matrix = torch.where(
        safe_inputs[..., None, None], safe_matrix, torch.zeros_like(matrix)
    )
    safe_shift = torch.where(
        safe_inputs, safe_shift, torch.zeros_like(safe_shift)
    )
    safe_kt = torch.where(safe_inputs, safe_kt, torch.ones_like(safe_kt))

    difference = safe_matrix - safe_shift[..., None, None] * eye
    maximum_exponent = _FERMI_MAX_SCALED_NORM * float(2**_FERMI_MAX_SCALE_POWER)
    entry_in_range = (
        difference.abs().amax(dim=(-1, -2)) / maximum_exponent <= safe_kt
    )
    safe_inputs = safe_inputs & (entry_in_range | ~active_any)
    difference = torch.where(
        safe_inputs[..., None, None], difference, torch.zeros_like(difference)
    )
    safe_kt = torch.where(safe_inputs, safe_kt, torch.ones_like(safe_kt))
    exponent = difference / safe_kt[..., None, None]
    low_temperature = safe_kt < 5.0e-4

    exponent = torch.where(active_pair, exponent, torch.zeros_like(exponent))
    exponent_bound = exponent.detach().abs().sum(-1).amax(-1)
    required_power = torch.ceil(
        torch.log2(exponent_bound.clamp_min(1.0) / _FERMI_MAX_SCALED_NORM)
    ).to(torch.long)

    base_power = torch.where(low_temperature, 22, 11)
    in_range = safe_inputs & (required_power <= _FERMI_MAX_SCALE_POWER)
    exponent = torch.where(
        in_range[..., None, None], exponent, torch.zeros_like(exponent)
    )
    scale_power = torch.maximum(base_power, required_power).clamp(
        min=0, max=_FERMI_MAX_SCALE_POWER
    )
    scale = torch.pow(exponent.new_tensor(2.0), scale_power)
    exp_argument = 2.0 * exponent / scale[..., None, None]
    expm1_scaled = _matrix_expm1(exp_argument)
    tangent = dense_solve(expm1_scaled + 2.0 * eye, expm1_scaled)

    for index in range(_FERMI_MAX_SCALE_POWER - 1):
        denominator = eye + tangent @ tangent
        if direct_rhs:
            doubled = 2.0 * dense_solve(denominator, tangent)
        else:
            doubled = 2.0 * tangent @ dense_solve(denominator, eye)
        update = index < (scale_power - 1)
        tangent = torch.where(update[..., None, None], doubled, tangent)

    density = 0.5 * (eye - tangent)
    density = 0.5 * (density + density.transpose(-2, -1))
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    density = torch.where(pair, density, torch.zeros_like(density))
    nan = torch.full_like(density, float("nan"))
    return torch.where(in_range[..., None, None], density, nan)


def _matrix_fermi_direct_rhs(
    matrix: torch.Tensor,
    shift: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Evaluate matrix-Fermi density using the direct-RHS doubling solve."""
    return _matrix_fermi_with_rhs(
        matrix, shift, kt, active, direct_rhs=True
    )


def _matrix_fermi_identity_rhs(
    matrix: torch.Tensor,
    shift: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the pre-W2c6 identity-RHS recurrence for regression tests."""
    return _matrix_fermi_with_rhs(
        matrix, shift, kt, active, direct_rhs=False
    )


def _fixture(dimension: int, power: int) -> tuple[torch.Tensor, ...]:
    """Create a rotated spectrum selecting the requested outer power."""
    generator = torch.Generator(device="cpu").manual_seed(8317 + dimension)
    raw = torch.randn((dimension, dimension), dtype=DTYPE, generator=generator)
    rotation, _ = torch.linalg.qr(raw)
    values = torch.zeros(dimension, dtype=DTYPE)
    values[0] = -0.17
    if dimension > 1:
        values[1] = 0.13
    if dimension > 3:
        values[2:-1] = torch.linspace(-0.31, 0.29, dimension - 3)
    values[-1] = 0.0
    projector = rotation[:, -1:] @ rotation[:, -1:].T
    base = rotation @ torch.diag(values) @ rotation.T
    desired_bound = 0.75 * _FERMI_MAX_SCALED_NORM * float(2**power)
    projector_norm = projector.abs().sum(-1).amax()
    base_norm = base.abs().sum(-1).amax()
    amplitude = (desired_bound - base_norm) / projector_norm
    matrix = base + amplitude * projector
    shift = torch.zeros((), dtype=DTYPE)
    kt = torch.ones((), dtype=DTYPE)
    active = torch.ones(dimension, dtype=torch.bool)
    bound = matrix.abs().sum(-1).amax()
    selected_power = int(
        torch.ceil(torch.log2(bound.clamp_min(1.0) / _FERMI_MAX_SCALED_NORM))
    )
    if selected_power != power:
        raise AssertionError(
            f"Fixture selected power {selected_power}; requested {power}"
        )
    return matrix, shift, kt, active


def _selected_powers(
    matrix: torch.Tensor,
    shift: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
) -> list[int]:
    """Calculate detached lane-local powers for benchmark metadata."""
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    exponent = (matrix - shift[..., None, None] * eye) / kt
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    exponent = torch.where(pair, exponent, torch.zeros_like(exponent))
    bound = exponent.abs().sum(-1).amax(-1)
    required = torch.ceil(
        torch.log2(bound.clamp_min(1.0) / _FERMI_MAX_SCALED_NORM)
    ).to(torch.long)
    base = torch.where(kt < 5.0e-4, 22, 11)
    selected = torch.maximum(base, required).clamp(
        min=0, max=_FERMI_MAX_SCALE_POWER
    )
    return [int(value) for value in selected.reshape(-1).tolist()]


def _density_correction(
    function: Callable[..., torch.Tensor],
    matrix: torch.Tensor,
    target: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
    diff_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the production two-step count correction and final Fermi call."""
    shift = torch.zeros_like(target)
    density = function(matrix, shift, kt, active)
    for _ in range(2):
        count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
        slope = (density - density @ density).diagonal(dim1=-2, dim2=-1).sum(
            -1
        ) / kt
        safe_slope = torch.where(
            slope.abs() > torch.finfo(matrix.dtype).tiny,
            slope,
            torch.ones_like(slope),
        )
        correction = torch.where(
            diff_mask, (target - count) / safe_slope, torch.zeros_like(target)
        )
        shift = shift + correction
        density = function(matrix, shift, kt, active)
    return density


def _rss_mib() -> float:
    """Read process lifetime peak RSS in MiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cpu_model() -> str:
    """Return the CPU model reported by Linux, when available."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", maxsplit=1)[-1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _source_revision() -> tuple[str, bool]:
    """Return repository HEAD and whether the worktree contains changes."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return revision, dirty


def _measure(
    function: Callable[[], torch.Tensor], repeats: int
) -> dict[str, float]:
    """Measure repeated wall time after one warm-up evaluation."""
    function()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        timings.append(time.perf_counter() - started)
    return {
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "max_s": max(timings),
    }


def _profile_counts(function: Callable[[], torch.Tensor]) -> dict[str, int]:
    """Count matrix products and dense solves in one call."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        function()
    counts = {event.key: event.count for event in profiler.key_averages()}
    selected = {
        key: count
        for key, count in counts.items()
        if key in ("aten::mm", "aten::bmm", "aten::matmul")
        or "solve" in key.lower()
    }
    return selected


def _run(args: argparse.Namespace) -> None:
    """Run one method and emit machine-readable timing and operator records."""
    for name, expected in THREAD_ENV.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must be explicitly set to {expected}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (1, 1):
        raise RuntimeError("Expected one intra-op and inter-op thread")
    if (
        Path(dxtb.__file__).resolve()
        != (ROOT / "src/dxtb/__init__.py").resolve()
    ):
        raise RuntimeError("dxtb import does not resolve to this worktree")

    if args.capture is None:
        if args.dimension is None:
            raise ValueError("--dimension is required without --capture")
        matrix, shift, kt, active = _fixture(args.dimension, args.power)
        input_source = "synthetic-rotated-spectrum"
        requested_power = args.power
        target = None
        diff_mask = None
    else:
        captured = torch.load(
            args.capture, map_location="cpu", weights_only=False
        )
        matrix = captured["matrix"]
        shift = captured["shifts"][args.correction_index]
        kt = captured["kt"]
        active = captured["active"]
        target = captured["target"]
        diff_mask = captured["diff_mask"]
        if args.dimension is not None and args.dimension != matrix.shape[-1]:
            raise ValueError("--dimension does not match the captured matrix")
        input_source = str(args.capture.resolve())
        requested_power = None
    selected_powers = _selected_powers(matrix, shift, kt, active)
    source_revision, source_dirty = _source_revision()
    functions = {
        "production": _matrix_fermi,
        "identity-rhs": _matrix_fermi_identity_rhs,
        "direct-rhs": _matrix_fermi_direct_rhs,
    }
    function = functions[args.method]
    if args.density_correction and (target is None or diff_mask is None):
        raise ValueError("--density-correction requires a physical capture")

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        if args.density_correction:
            assert target is not None and diff_mask is not None
            return _density_correction(
                function, value, target, kt, active, diff_mask
            )
        return function(value, shift, kt, active)

    direction = torch.randn_like(matrix)
    direction = 0.5 * (direction + direction.mT)
    cotangent = torch.randn_like(matrix)

    def primal_call() -> torch.Tensor:
        return evaluate(matrix)

    operation: Callable[[], torch.Tensor]
    if args.mode == "forward":

        def operation() -> torch.Tensor:
            return evaluate(matrix)

    elif args.mode == "jvp":

        def function_of_matrix(value: torch.Tensor) -> torch.Tensor:
            return evaluate(value)

        def operation() -> torch.Tensor:
            return torch.func.jvp(function_of_matrix, (matrix,), (direction,))[
                1
            ]

    else:

        def scalar(value: torch.Tensor) -> torch.Tensor:
            return (evaluate(value) * cotangent).sum()

        if args.mode == "reverse":

            def operation() -> torch.Tensor:
                return torch.func.grad(scalar)(matrix)

        elif args.mode == "second":

            def operation() -> torch.Tensor:
                gradient = torch.func.grad(scalar)
                return torch.func.jvp(gradient, (matrix,), (direction,))[1]

        else:

            def first_directional(value: torch.Tensor) -> torch.Tensor:
                return torch.func.jvp(scalar, (value,), (direction,))[1]

            def second_directional(value: torch.Tensor) -> torch.Tensor:
                return torch.func.jvp(
                    first_directional, (value,), (direction,)
                )[1]

            def operation() -> torch.Tensor:
                return torch.func.jvp(
                    second_directional, (matrix,), (direction,)
                )[1]

    elapsed = _measure(operation, args.repeats)
    rss_before_profile = _rss_mib()
    counts = _profile_counts(primal_call)
    record = {
        "event": "tanh_solve_benchmark",
        "method": args.method,
        "mode": args.mode,
        "dimension": matrix.shape[-1],
        "input_source": input_source,
        "operation": (
            "fermi_density_matrix-correction-core"
            if args.density_correction
            else "matrix_fermi"
        ),
        "requested_power": requested_power,
        "selected_power_by_lane": selected_powers,
        "dtype": str(matrix.dtype),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": _cpu_model(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "dxtb_source": str(Path(dxtb.__file__).resolve()),
        "pytorch_intra_threads": torch.get_num_threads(),
        "pytorch_inter_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "tad-mctc",
                "tad-multicharge",
                "tad-libcint",
                "tad-dftd3",
                "tad-dftd4",
            )
        },
        "repeats": args.repeats,
        "timing": elapsed,
        "operation_counts": counts,
        "process_peak_rss_before_profiler_mib": rss_before_profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    """Parse bounded benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("production", "identity-rhs", "direct-rhs"),
        required=True,
    )
    parser.add_argument(
        "--dimension",
        type=int,
        choices=(1, 2, 8, 32, 128, 255),
    )
    parser.add_argument(
        "--power", type=int, choices=(11, 16, 22, 29, 30), default=16
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--correction-index", type=int, default=0)
    parser.add_argument("--density-correction", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("forward", "reverse", "jvp", "second", "third"),
        default="forward",
    )
    parser.add_argument("--repeats", type=int, default=3)
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
