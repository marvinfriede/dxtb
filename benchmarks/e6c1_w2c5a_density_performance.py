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
"""Benchmark physical and synthetic matrix-Fermi correction sequences.

This compares the accepted three-evaluation correction sequence with the
diagnostic trace-squared and gauge-invariant matrix-shift candidates. The
physical fixture is a detached 255-orbital SCF input captured from tmpda.
The benchmark measures the matrix-Fermi correction core; it excludes the
SCF diagonalization, AO transform, and result construction.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dxtb  # noqa: E402
from benchmarks.e6c1_w2c5_fermi_shift import (  # noqa: E402
    _shift_density,
)
from dxtb._src.scf.functional.spectral import _matrix_fermi  # noqa: E402

DTYPE = torch.float64
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _matmul_slope(density: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
    """Return the original cubic trace-square response slope."""
    count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
    square = (density @ density).diagonal(dim1=-2, dim2=-1).sum(-1)
    return (count - square) / kt


def _trace_square_slope(
    density: torch.Tensor, kt: torch.Tensor
) -> torch.Tensor:
    """Return the equivalent elementwise trace-square response slope."""
    count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
    square = (density * density.transpose(-2, -1)).sum((-2, -1))
    return (count - square) / kt


def _correction_density(
    matrix: torch.Tensor,
    target: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
    diff_mask: torch.Tensor,
    *,
    use_shift: bool,
    use_trace_square: bool,
) -> torch.Tensor:
    """Run two count corrections with selected slope/shift implementations."""
    shift = torch.zeros_like(target)
    density = _matrix_fermi(matrix, shift, kt, active)
    for _ in range(2):
        slope_function = (
            _trace_square_slope if use_trace_square else _matmul_slope
        )
        slope = slope_function(density, kt)
        safe_slope = torch.where(
            slope.abs() > torch.finfo(DTYPE).tiny,
            slope,
            torch.ones_like(slope),
        )
        count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
        delta = torch.where(diff_mask, (target - count) / safe_slope, 0.0)
        if use_shift:
            density = _shift_density(density, delta, kt)
        else:
            shift = shift + delta
            density = _matrix_fermi(matrix, shift, kt, active)
    return density


def _synthetic_input(size: int) -> dict[str, torch.Tensor | str]:
    """Build a deterministic smooth spectrum for a requested matrix size."""
    orbital = torch.linspace(-0.05, 0.05, size, dtype=DTYPE)
    matrix = torch.diag(orbital).expand(1, 2, size, size).clone()
    target = torch.full((1, 2), 0.43 * size, dtype=DTYPE)
    active = torch.ones((1, 2, size), dtype=torch.bool)
    return {
        "system_id": f"synthetic-{size}",
        "matrix": matrix,
        "target": target,
        "kt": torch.full((), 0.008, dtype=DTYPE),
        "active": active,
        "diff_mask": torch.ones((1, 2), dtype=torch.bool),
        "reference_densities": torch.empty(0, dtype=DTYPE),
    }


def _load_physical_input(path: Path) -> dict[str, Any]:
    """Load one detached physical correction input captured from tmpda."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "matrix",
        "kt",
        "active",
        "target",
        "diff_mask",
        "reference_densities",
        "system_id",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"Capture is missing fields: {sorted(missing)}")
    return value


def _direction(matrix: torch.Tensor) -> torch.Tensor:
    """Return a deterministic symmetric direction with unit Frobenius norm."""
    index = torch.arange(matrix.shape[-1], dtype=DTYPE)
    raw = torch.sin(index[:, None] + 0.5 * index[None, :])
    raw = 0.5 * (raw + raw.T)
    raw = raw / torch.linalg.vector_norm(raw)
    return raw.expand_as(matrix)


def _timed(function: Callable[[], torch.Tensor], repeats: int) -> list[float]:
    """Warm the function once and return repeated wall-clock durations."""
    function()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return values


def _directional_derivative(
    function: Callable[[torch.Tensor], torch.Tensor],
    matrix: torch.Tensor,
    direction: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Evaluate a forward directional derivative through the given order."""

    def scalar(parameter: torch.Tensor) -> torch.Tensor:
        return function(matrix + parameter * direction)

    derivative = scalar
    for _ in range(order):
        previous = derivative

        def derivative(parameter: torch.Tensor, fn=previous) -> torch.Tensor:
            return torch.func.jvp(
                fn,
                (parameter,),
                (torch.ones_like(parameter),),
            )[1]

    return derivative(torch.zeros((), dtype=DTYPE))


def _measure_case(
    values: dict[str, Any], *, repeats: int
) -> list[dict[str, Any]]:
    """Compare four correction sequences and assert physical value parity."""
    matrix = values["matrix"].to(dtype=DTYPE)
    target = values["target"].to(dtype=DTYPE)
    kt = values["kt"].to(dtype=DTYPE)
    active = values["active"].to(dtype=torch.bool)
    diff_mask = values["diff_mask"].to(dtype=torch.bool)
    direction = _direction(matrix)
    methods = {
        "production": (False, False),
        "trace_squared": (False, True),
        "matrix_shift": (True, False),
        "combined": (True, True),
    }

    def make_function(use_shift: bool, use_trace_square: bool):
        def evaluate(spectrum: torch.Tensor) -> torch.Tensor:
            result = _correction_density(
                spectrum,
                target,
                kt,
                active,
                diff_mask,
                use_shift=use_shift,
                use_trace_square=use_trace_square,
            )
            return result.sum()

        return evaluate

    reference_output = None
    reference_densities = values.get("reference_densities")
    if (
        isinstance(reference_densities, torch.Tensor)
        and reference_densities.numel()
    ):
        reference_output = reference_densities[-1].to(dtype=DTYPE)

    records = []
    for name, (use_shift, use_trace_square) in methods.items():
        output_function = lambda spectrum: _correction_density(
            spectrum,
            target,
            kt,
            active,
            diff_mask,
            use_shift=use_shift,
            use_trace_square=use_trace_square,
        )
        output = output_function(matrix)
        if reference_output is not None:
            torch.testing.assert_close(
                output,
                reference_output,
                atol=2.0e-11,
                rtol=2.0e-10,
                msg=f"{values['system_id']} {name} versus accepted physical density",
            )

        scalar = make_function(use_shift, use_trace_square)
        functions: dict[str, Callable[[], torch.Tensor]] = {
            "forward": lambda: output_function(matrix),
            "reverse": lambda: torch.func.grad(scalar)(matrix),
            "jvp": lambda: torch.func.jvp(scalar, (matrix,), (direction,))[1],
        }
        derivative_orders = (1, 2, 3) if matrix.shape[-1] <= 32 else (1,)
        higher_order_seconds = {}
        if matrix.shape[-1] <= 32:
            for order in (2, 3):
                started = time.perf_counter()
                _directional_derivative(scalar, matrix, direction, order)
                higher_order_seconds[str(order)] = time.perf_counter() - started
        timing = {
            metric: _timed(function, repeats)
            for metric, function in functions.items()
        }
        derivative_values = {
            order: _directional_derivative(
                scalar, matrix, direction, order
            ).detach()
            for order in derivative_orders
        }
        record = {
            "case": values["system_id"],
            "n_orbitals": matrix.shape[-1],
            "method": name,
            "median_seconds": {
                key: statistics.median(samples)
                for key, samples in timing.items()
            },
            "range_seconds": {
                key: [min(samples), max(samples)]
                for key, samples in timing.items()
            },
            "output_finite": bool(torch.isfinite(output).all()),
            "density_reference_max_abs": (
                None
                if reference_output is None
                else float((output - reference_output).abs().max())
            ),
            "directional_derivatives": {
                str(key): float(value)
                for key, value in derivative_values.items()
            },
            "higher_order_seconds_single_sample": higher_order_seconds,
            "process_peak_rss_mib_cumulative": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            "expected_dense_solve_count": (32 if use_shift else 90),
            "matrix_multiply_slope_calls": (0 if use_trace_square else 2),
        }
        records.append(record)

    baseline = records[0]
    for record in records[1:]:
        for order in baseline["directional_derivatives"]:
            torch.testing.assert_close(
                torch.tensor(record["directional_derivatives"][order]),
                torch.tensor(baseline["directional_derivatives"][order]),
                atol={"1": 2.0e-9, "2": 2.0e-7, "3": 2.0e-4}[order],
                rtol=2.0e-8,
                msg=f"{values['system_id']} {record['method']} derivative {order}",
            )
    return records


def _configure() -> dict[str, Any]:
    """Assert all thread settings and report the benchmark environment."""
    mismatches = {
        key: (os.environ.get(key), value)
        for key, value in THREAD_ENV.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Thread environment mismatch: {mismatches}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Expected single-thread PyTorch settings")
    expected_source = (ROOT / "src/dxtb/__init__.py").resolve()
    actual_source = Path(dxtb.__file__).resolve()
    if actual_source != expected_source:
        raise RuntimeError(
            f"Imported {actual_source}; expected {expected_source}"
        )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "source": str(actual_source),
        "torch_intra_threads": torch.get_num_threads(),
        "torch_inter_threads": torch.get_num_interop_threads(),
        "thread_environment": dict(THREAD_ENV),
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
    }


def main() -> None:
    """Run bounded matched matrix-Fermi correction measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--physical-only", action="store_true")
    args = parser.parse_args()

    environment = _configure()
    physical = _load_physical_input(args.capture)
    cases = (
        [physical]
        if args.physical_only
        else [_synthetic_input(size) for size in (32, 128)] + [physical]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "environment", **environment}) + "\n")
        for case in cases:
            for record in _measure_case(case, repeats=args.repeats):
                stream.write(
                    json.dumps({"event": "density_benchmark", **record}) + "\n"
                )
                stream.flush()


if __name__ == "__main__":
    main()
