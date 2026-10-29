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
"""Assess invariant matrix-Fermi reuse under scalar chemical-potential shifts.

This diagnostic compares the accepted matrix-Fermi correction path with an
experimental gauge-invariant shift identity. It does not alter production
SCF code.

Example
-------
Run in one of the controlled CPU environments::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /path/to/python benchmarks/e6c1_w2c5_fermi_shift.py \
        --output benchmarks/results/e6c1-w2c5/py311.jsonl
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from tad_mctc import storch
from tad_mctc.storch.linsolve import dense_solve

import dxtb  # noqa: E402
from dxtb._src.scf.functional import spectral as spectral_module  # noqa: E402
from dxtb._src.scf.functional.spectral import (  # noqa: E402
    _gap_electron_response,
    _matrix_fermi,
    _prepare_matrix,
    fermi_density_matrix,
    fermi_free_energy_trace,
)
from dxtb._src.wavefunction import filling  # noqa: E402

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
DTYPE = torch.float64


def _version(name: str) -> str:
    """Return an installed package version or an explicit absence marker."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _cpu_model() -> str:
    """Return the host CPU model where Linux exposes it."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _emit(path: Path, event: str, **values: Any) -> None:
    """Persist and print one diagnostic record."""
    record = {"event": event, "time": time.time(), **values}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


def _configure(path: Path) -> None:
    """Require and record deterministic one-thread CPU settings."""
    mismatches = {
        key: (os.environ.get(key), value)
        for key, value in THREAD_ENV.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Thread environment mismatch: {mismatches}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (1, 1):
        raise RuntimeError("Expected one intra-op and inter-op thread")

    source = str(Path(dxtb.__file__).resolve())
    expected = str((ROOT / "src/dxtb/__init__.py").resolve())
    if source != expected:
        raise RuntimeError(f"Imported {source}; expected {expected}")
    _emit(
        path,
        "environment",
        python=sys.version.split()[0],
        torch=torch.__version__,
        cpu=_cpu_model(),
        dxtb_source=source,
        pytorch_intra_threads=torch.get_num_threads(),
        pytorch_inter_threads=torch.get_num_interop_threads(),
        thread_environment={key: os.environ[key] for key in THREAD_ENV},
        dependencies={
            name: _version(name)
            for name in (
                "tad-mctc",
                "tad-multicharge",
                "tad-libcint",
                "tad-dftd3",
                "tad-dftd4",
            )
        },
    )


def _state(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    mask: torch.Tensor | None = None,
    l_inv: torch.Tensor | None = None,
) -> Any:
    """Construct the same detached-primal, differentiable filling state."""
    values, _ = storch.eighb(
        a=hamiltonian,
        b=None,
        l_inv=l_inv,
        is_posdef=True,
        factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
        broadening_method=None,
    )
    channel_energies = values.unsqueeze(-2).expand(
        *electrons.shape, values.shape[-1]
    )
    channel_mask = None
    if mask is not None:
        active_count = mask.sum(-1)
        prefix = torch.arange(values.shape[-1], device=values.device)
        channel_mask = (prefix < active_count.unsqueeze(-1)).unsqueeze(-2)
        channel_mask = channel_mask.expand_as(channel_energies)
    return filling._get_fermi_occupation_core_state(
        electrons,
        channel_energies,
        kt,
        mask=channel_mask,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
    )


def _trace_square_slope(
    density: torch.Tensor, kt: torch.Tensor
) -> torch.Tensor:
    """Compute Tr(D-D²)/kT without a matrix-matrix product."""
    count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_square = (density * density.transpose(-2, -1)).sum(dim=(-2, -1))
    return (count - trace_square) / kt


def _matmul_slope(density: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
    """Reference slope using the current matrix-product expression."""
    count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_square = (density @ density).diagonal(dim1=-2, dim2=-1).sum(-1)
    return (count - trace_square) / kt


def _shift_density(
    density: torch.Tensor, delta: torch.Tensor, kt: torch.Tensor
) -> torch.Tensor:
    """Apply a scalar chemical-potential shift to an invariant Fermi matrix."""
    max_ratio = -0.25 * torch.log(
        density.new_tensor(torch.finfo(density.dtype).eps)
    )
    ratio = -delta / kt
    valid = (
        torch.isfinite(density).all(dim=(-1, -2))
        & torch.isfinite(ratio)
        & torch.isfinite(kt)
        & (kt > 0.0)
        & (ratio.abs() <= max_ratio)
    )
    safe_density = torch.where(
        valid[..., None, None], density, torch.zeros_like(density)
    )
    safe_ratio = torch.where(valid, ratio, torch.zeros_like(ratio))
    eye = torch.eye(
        density.shape[-1], dtype=density.dtype, device=density.device
    ).expand(density.shape)
    coefficient = torch.expm1(safe_ratio)
    denominator = eye + coefficient[..., None, None] * (eye - safe_density)
    updated = dense_solve(denominator, safe_density)
    updated = 0.5 * (updated + updated.transpose(-2, -1))
    nan = torch.full_like(updated, float("nan"))
    return torch.where(valid[..., None, None], updated, nan)


def _raw_shift_density(
    density: torch.Tensor, delta: torch.Tensor, kt: torch.Tensor
) -> torch.Tensor:
    """Run the unguarded algebra only for a labeled negative conditioning probe."""
    eye = torch.eye(
        density.shape[-1], dtype=density.dtype, device=density.device
    )
    coefficient = torch.expm1(-delta / kt)
    denominator = eye + coefficient * (eye - density)
    updated = dense_solve(denominator, density)
    return 0.5 * (updated + updated.T)


def _candidate_free_energy_trace(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    state: Any,
) -> torch.Tensor:
    """Evaluate the trace path with call-local matrix-shift reuse.

    The scoped substitution is confined to this diagnostic invocation and is
    restored even if evaluation raises. It is not a production backend.
    """
    original = spectral_module._matrix_fermi
    previous_density: torch.Tensor | None = None
    previous_shift: torch.Tensor | None = None

    def shifted_matrix_fermi(
        matrix: torch.Tensor,
        shift: torch.Tensor,
        temperature: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal previous_density, previous_shift
        if previous_density is None or previous_shift is None:
            density = original(matrix, shift, temperature, active)
        else:
            density = _shift_density(
                previous_density,
                shift - previous_shift,
                temperature,
            )
        previous_density = density
        previous_shift = shift
        return density

    spectral_module._matrix_fermi = shifted_matrix_fermi
    try:
        identity = torch.eye(
            hamiltonian.shape[-1],
            dtype=hamiltonian.dtype,
            device=hamiltonian.device,
        )
        return fermi_free_energy_trace(
            hamiltonian,
            electrons,
            kt,
            state,
            l_inv=identity,
            diff_steps=2,
        )
    finally:
        spectral_module._matrix_fermi = original


def _candidate_density(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    state: Any,
    *,
    mask: torch.Tensor | None = None,
    l_inv: torch.Tensor | None = None,
    use_trace_square: bool = False,
) -> torch.Tensor:
    """Mirror production count corrections using two invariant matrix shifts."""
    dtype = torch.float64
    h = hamiltonian.to(dtype=dtype)
    structural_mask = mask
    if structural_mask is None:
        structural_mask = state.orbital_mask
        if structural_mask is not None and structural_mask.ndim == h.ndim:
            structural_mask = structural_mask.select(-2, 0)
    li = None if l_inv is None else l_inv.to(device=h.device, dtype=dtype)
    h, spectral, active = _prepare_matrix(h, li, structural_mask)
    h = h.unsqueeze(-3).expand(*electrons.shape, *h.shape[-2:])
    spectral = spectral.unsqueeze(-3).expand_as(h)
    active = active.unsqueeze(-2).expand(*electrons.shape, active.shape[-1])
    if li is not None:
        li = li.unsqueeze(-3).expand_as(h)
        pair = active.unsqueeze(-1) & active.unsqueeze(-2)
        li = torch.where(pair, li, torch.zeros_like(li))

    kt = kt.to(device=h.device, dtype=dtype)
    safe_kt = torch.where(
        state.kt_positive.to(h.device), kt, torch.ones_like(kt)
    )
    ref = state.chemical_potential_ref.to(h.device, dtype=dtype).squeeze(-1)
    offset = state.chemical_potential_offset.to(h.device, dtype=dtype).squeeze(
        -1
    )
    eye = torch.eye(h.shape[-1], dtype=dtype, device=h.device)
    channel_shape = (*electrons.shape, 1, 1)
    relative = spectral - ref.reshape(channel_shape) * eye
    relative = relative - offset.reshape(channel_shape) * eye

    target = electrons.to(device=h.device, dtype=dtype)
    empty = state.empty.to(h.device)
    full = state.full.to(h.device)
    too_many = state.too_many.to(h.device)
    diff_mask = (
        state.active.to(h.device)
        & ~state.gap.to(h.device, dtype=torch.bool)
        & ~empty
        & ~full
        & ~too_many
    )
    shift = torch.zeros_like(target)
    density = _matrix_fermi(relative, shift, safe_kt, active)
    for _ in range(2):
        if use_trace_square:
            slope = _trace_square_slope(density, safe_kt)
        else:
            slope = _matmul_slope(density, safe_kt)
        safe_slope = torch.where(
            slope.abs() > torch.finfo(dtype).tiny, slope, 1.0
        )
        correction = torch.where(
            diff_mask,
            (target - density.diagonal(dim1=-2, dim2=-1).sum(-1)) / safe_slope,
            0.0,
        )
        density = _shift_density(density, correction, safe_kt)

    eye_channel = eye * active.unsqueeze(-1).to(dtype)
    density = torch.where(empty[..., None, None], 0.0, density)
    density = torch.where(too_many[..., None, None], 0.0, density)
    density = torch.where(full[..., None, None], eye_channel, density)
    gap_mask = state.gap.to(h.device, dtype=torch.bool)
    response = _gap_electron_response(
        h,
        spectral,
        active,
        state.below.to(h.device, dtype=torch.bool),
        gap_mask,
        safe_kt,
    )
    density = density + (target - target.detach())[..., None, None] * response
    if li is not None:
        density = li.transpose(-2, -1) @ density @ li
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    density = torch.where(pair, density, torch.zeros_like(density))
    return density.sum(-3).to(dtype=hamiltonian.dtype)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> float:
    """Assert finite numerical parity and return maximum absolute error."""
    assert torch.isfinite(actual).all(), f"{label}: actual is nonfinite"
    assert torch.isfinite(expected).all(), f"{label}: expected is nonfinite"
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, msg=label
    )
    return float((actual - expected).abs().max().detach())


def _directional_value(
    function: Any, point: torch.Tensor, order: int
) -> torch.Tensor:
    """Evaluate a scalar function's nested forward directional derivative."""
    derivative = function
    for _ in range(order):
        previous = derivative

        def derivative(value: torch.Tensor, fn=previous) -> torch.Tensor:
            return torch.func.jvp(fn, (value,), (torch.ones_like(value),))[1]

    return derivative(point)


def _check_transform_parity(
    output: Path,
    name: str,
    candidate: Any,
    reference: Any,
    point: torch.Tensor,
) -> None:
    """Assert forward, reverse, mixed, and third transform parity."""
    tangent = torch.ones_like(point)
    scalar_transforms = {
        "value": (candidate, reference),
        "grad": (
            torch.func.grad(candidate),
            torch.func.grad(reference),
        ),
        "jacfwd": (
            torch.func.jacfwd(candidate),
            torch.func.jacfwd(reference),
        ),
        "jacrev": (
            torch.func.jacrev(candidate),
            torch.func.jacrev(reference),
        ),
        "forward_second": (
            torch.func.jacfwd(torch.func.jacfwd(candidate)),
            torch.func.jacfwd(torch.func.jacfwd(reference)),
        ),
        "reverse_second": (
            torch.func.jacrev(torch.func.jacrev(candidate)),
            torch.func.jacrev(torch.func.jacrev(reference)),
        ),
        "forward_third": (
            torch.func.jacfwd(torch.func.jacfwd(torch.func.jacfwd(candidate))),
            torch.func.jacfwd(torch.func.jacfwd(torch.func.jacfwd(reference))),
        ),
        "reverse_third": (
            torch.func.grad(torch.func.grad(torch.func.grad(candidate))),
            torch.func.grad(torch.func.grad(torch.func.grad(reference))),
        ),
        "reverse_over_forward_third": (
            torch.func.jacrev(torch.func.jacfwd(torch.func.jacfwd(candidate))),
            torch.func.jacrev(torch.func.jacfwd(torch.func.jacfwd(reference))),
        ),
    }
    errors = {}
    for transform_name, (actual_fn, expected_fn) in scalar_transforms.items():
        actual = actual_fn(point)
        expected = expected_fn(point)
        errors[transform_name] = _assert_close(
            actual,
            expected,
            atol=3e-4 if "third" in transform_name else 3e-7,
            rtol=5e-7,
            label=f"{name} {transform_name}",
        )

    actual_jvp = torch.func.jvp(candidate, (point,), (tangent,))[1]
    expected_jvp = torch.func.jvp(reference, (point,), (tangent,))[1]
    errors["jvp"] = _assert_close(
        actual_jvp,
        expected_jvp,
        atol=3e-7,
        rtol=5e-7,
        label=f"{name} jvp",
    )
    _emit(output, "transform_parity", name=name, max_abs=errors)


def _check_physical_fixture(
    output: Path,
    name: str,
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt_value: float,
    mask: torch.Tensor | None = None,
    l_inv: torch.Tensor | None = None,
) -> None:
    """Compare candidate and accepted density values and matrix derivatives."""
    kt = torch.tensor(kt_value, dtype=hamiltonian.dtype)
    state = _state(hamiltonian, electrons, kt, mask, l_inv)
    expected = fermi_density_matrix(
        hamiltonian,
        electrons,
        kt,
        state,
        l_inv=l_inv,
        mask=mask,
        diff_steps=2,
    )
    actual = _candidate_density(
        hamiltonian, electrons, kt, state, mask=mask, l_inv=l_inv
    )
    trace_square_candidate = _candidate_density(
        hamiltonian,
        electrons,
        kt,
        state,
        mask=mask,
        l_inv=l_inv,
        use_trace_square=True,
    )
    _assert_close(
        trace_square_candidate,
        actual,
        atol=2e-10,
        rtol=2e-10,
        label=f"{name} trace-square combined candidate",
    )
    if l_inv is None:
        count = actual.diagonal(dim1=-2, dim2=-1).sum()
    else:
        factor = torch.linalg.inv(l_inv)
        overlap = factor @ factor.T
        count = torch.diagonal(actual @ overlap).sum()
    _assert_close(
        count,
        electrons.sum(),
        atol=3e-10,
        rtol=3e-10,
        label=f"{name} metric electron count",
    )
    error = _assert_close(
        actual,
        expected,
        atol=2e-9,
        rtol=2e-9,
        label=f"{name} density values",
    )
    direction = torch.arange(
        hamiltonian.numel(), dtype=hamiltonian.dtype
    ).reshape_as(hamiltonian)
    direction = 0.03 * (direction + direction.T + 1.0)

    def objective(parameter: torch.Tensor, use_candidate: bool) -> torch.Tensor:
        h = hamiltonian + parameter * direction
        current_state = _state(h, electrons, kt, mask, l_inv)
        current = (
            _candidate_density(
                h,
                electrons,
                kt,
                current_state,
                mask=mask,
                l_inv=l_inv,
                use_trace_square=True,
            )
            if use_candidate
            else fermi_density_matrix(
                h,
                electrons,
                kt,
                current_state,
                l_inv=l_inv,
                mask=mask,
                diff_steps=2,
            )
        )
        return (current * direction).sum()

    point = torch.zeros((), dtype=hamiltonian.dtype)
    derivative_errors = {}
    for order, atol in ((1, 3e-8), (2, 3e-6), (3, 3e-4)):
        expected_derivative = _directional_value(
            lambda p: objective(p, False), point, order
        )
        actual_derivative = _directional_value(
            lambda p: objective(p, True), point, order
        )
        derivative_errors[str(order)] = _assert_close(
            actual_derivative,
            expected_derivative,
            atol=atol,
            rtol=2e-7,
            label=f"{name} Hamiltonian derivative order {order}",
        )

    parameter_errors: dict[str, dict[str, float]] = {}
    for variable, atol in (("temperature", 2e-5), ("electron_count", 2e-4)):

        def scalar_parameter(
            parameter: torch.Tensor, use_candidate: bool
        ) -> torch.Tensor:
            if variable == "temperature":
                h = hamiltonian
                e = electrons
                current_kt = kt + 0.015 * parameter
            else:
                h = hamiltonian
                e = electrons + parameter * torch.tensor(
                    [0.07, -0.03], dtype=hamiltonian.dtype
                )
                current_kt = kt
            current_state = _state(h, e, current_kt, mask, l_inv)
            current = (
                _candidate_density(
                    h,
                    e,
                    current_kt,
                    current_state,
                    mask=mask,
                    l_inv=l_inv,
                    use_trace_square=True,
                )
                if use_candidate
                else fermi_density_matrix(
                    h,
                    e,
                    current_kt,
                    current_state,
                    l_inv=l_inv,
                    mask=mask,
                    diff_steps=2,
                )
            )
            return (current * direction).sum()

        parameter_errors[variable] = {}
        for order in (1, 2, 3):
            expected_parameter = _directional_value(
                lambda p: scalar_parameter(p, False), point, order
            )
            actual_parameter = _directional_value(
                lambda p: scalar_parameter(p, True), point, order
            )
            parameter_errors[variable][str(order)] = _assert_close(
                actual_parameter,
                expected_parameter,
                atol=atol,
                rtol=3e-7,
                label=f"{name} {variable} derivative order {order}",
            )

    _emit(
        output,
        "physical_fixture",
        name=name,
        density_max_abs=error,
        derivative_max_abs=derivative_errors,
        parameter_derivative_max_abs=parameter_errors,
        electron_count=float(count.detach()),
        target_electrons=float(electrons.sum().detach()),
        active_mask=None if mask is None else mask.tolist(),
    )


def _check_lane_independence(output: Path) -> None:
    """Check lane-local matrix shifts for distinct spectra and masks."""
    angle = torch.tensor(0.31, dtype=DTYPE)
    rotation = torch.stack(
        [
            torch.stack([angle.cos(), -angle.sin()]),
            torch.stack([angle.sin(), angle.cos()]),
        ]
    )
    matrices = torch.stack(
        [
            rotation
            @ torch.diag(torch.tensor([-0.3, 0.4], dtype=DTYPE))
            @ rotation.T,
            torch.diag(torch.tensor([-3.0, 1.5], dtype=DTYPE)),
        ]
    )
    active = torch.ones(2, 2, dtype=torch.bool)
    deltas = torch.tensor([0.011, -0.17], dtype=DTYPE)
    temperatures = torch.tensor([0.2, 0.35], dtype=DTYPE)

    def lane(
        matrix: torch.Tensor, delta: torch.Tensor, kt: torch.Tensor
    ) -> torch.Tensor:
        density = _matrix_fermi(matrix, torch.zeros_like(delta), kt, active[0])
        return _shift_density(density, delta, kt)

    mapped = torch.func.vmap(lane)(matrices, deltas, temperatures)
    serial = torch.stack(
        [lane(matrices[i], deltas[i], temperatures[i]) for i in range(2)]
    )
    error = _assert_close(
        mapped, serial, atol=2e-12, rtol=2e-12, label="mapped shift lanes"
    )

    def batched(value: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(lane)(value, deltas, temperatures)

    jacobian = torch.func.jacrev(batched)(matrices)
    cross = torch.maximum(
        jacobian[0, :, :, 1, :, :].abs().amax(),
        jacobian[1, :, :, 0, :, :].abs().amax(),
    )
    assert torch.isfinite(jacobian).all(), "mapped shift Jacobian is nonfinite"
    assert float(cross) == 0.0, f"cross-lane shift derivative is {cross}"
    _emit(
        output,
        "mapped_lanes",
        max_abs=error,
        cross_lane_max_abs=float(cross),
        lane_shift=deltas.tolist(),
        lane_temperature=temperatures.tolist(),
    )

    physical_h = torch.stack(
        [
            torch.diag(torch.tensor([-0.3, 0.4, 1.0], dtype=DTYPE)),
            torch.diag(torch.tensor([-0.3, 80.0, 0.5], dtype=DTYPE)),
        ]
    )
    physical_electrons = torch.tensor([[1.2, 0.8], [1.1, 0.7]], dtype=DTYPE)
    physical_kt = torch.tensor([0.2, 0.17], dtype=DTYPE)
    physical_mask = torch.tensor(
        [[True, True, True], [True, False, True]], dtype=torch.bool
    )

    def physical_lane(
        matrix: torch.Tensor,
        target: torch.Tensor,
        temperature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        lane_state = _state(matrix, target, temperature, mask)
        return _candidate_density(
            matrix, target, temperature, lane_state, mask=mask
        )

    mapped_physical = torch.func.vmap(physical_lane)(
        physical_h, physical_electrons, physical_kt, physical_mask
    )
    serial_physical = torch.stack(
        [
            physical_lane(
                physical_h[index],
                physical_electrons[index],
                physical_kt[index],
                physical_mask[index],
            )
            for index in range(2)
        ]
    )
    physical_error = _assert_close(
        mapped_physical,
        serial_physical,
        atol=3e-10,
        rtol=3e-10,
        label="mapped candidate densities",
    )

    def mapped_physical_from_matrices(value: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(physical_lane)(
            value, physical_electrons, physical_kt, physical_mask
        )

    physical_jacobian = torch.func.jacrev(mapped_physical_from_matrices)(
        physical_h
    )
    physical_cross = torch.maximum(
        physical_jacobian[0, :, :, 1, :, :].abs().amax(),
        physical_jacobian[1, :, :, 0, :, :].abs().amax(),
    )
    assert torch.isfinite(
        physical_jacobian
    ).all(), "mapped physical candidate Jacobian is nonfinite"
    assert (
        float(physical_cross) == 0.0
    ), f"mapped physical cross-lane derivative is {physical_cross}"
    reversed_physical = torch.func.vmap(physical_lane)(
        physical_h.flip(0),
        physical_electrons.flip(0),
        physical_kt.flip(0),
        physical_mask.flip(0),
    )
    _assert_close(
        reversed_physical,
        mapped_physical.flip(0),
        atol=3e-10,
        rtol=3e-10,
        label="mapped lane order permutation",
    )
    _emit(
        output,
        "mapped_candidate_physical_lanes",
        max_abs=physical_error,
        cross_lane_max_abs=float(physical_cross),
        masks=physical_mask.tolist(),
    )

    invalid_delta = torch.tensor([0.01, 20.0], dtype=DTYPE)
    invalid_mixed = torch.func.vmap(lane)(matrices, invalid_delta, temperatures)
    assert torch.isfinite(
        invalid_mixed[0]
    ).all(), "valid lane was contaminated by an unsupported shift"
    assert torch.isnan(
        invalid_mixed[1]
    ).all(), "unsupported shift did not propagate NaNs lane-locally"
    _emit(
        output,
        "invalid_lane_isolation",
        valid_lane_finite=bool(torch.isfinite(invalid_mixed[0]).all()),
        invalid_lane_nan=bool(torch.isnan(invalid_mixed[1]).all()),
    )


def _check_shift_derivatives(output: Path) -> None:
    """Compare shift derivatives directly at smooth and degenerate spectra."""
    angle = torch.tensor(0.51, dtype=DTYPE)
    rotation = torch.stack(
        [
            torch.stack([angle.cos(), -angle.sin()]),
            torch.stack([angle.sin(), angle.cos()]),
        ]
    )
    fixtures = {
        "rotated": rotation
        @ torch.diag(torch.tensor([-0.4, 0.25], dtype=DTYPE))
        @ rotation.T,
        "exact-degenerate": torch.eye(2, dtype=DTYPE) * -0.2,
    }
    kt = torch.tensor(0.18, dtype=DTYPE)
    delta = torch.tensor(0.0, dtype=DTYPE)
    direction = torch.tensor([[0.2, 0.3], [0.3, -0.1]], dtype=DTYPE)
    reports: dict[str, dict[str, float]] = {}

    for name, hamiltonian in fixtures.items():
        reports[name] = {}
        for variable in ("matrix", "shift", "temperature"):

            def evaluate(parameter: torch.Tensor, reuse: bool) -> torch.Tensor:
                h = (
                    hamiltonian + parameter * direction
                    if variable == "matrix"
                    else hamiltonian
                )
                shift = parameter if variable == "shift" else delta
                temperature = (
                    kt + parameter * 0.02 if variable == "temperature" else kt
                )
                active = torch.ones(2, dtype=torch.bool)
                if reuse:
                    initial = _matrix_fermi(
                        h, torch.zeros_like(shift), temperature, active
                    )
                    value = _shift_density(initial, shift, temperature)
                else:
                    value = _matrix_fermi(h, shift, temperature, active)
                return (value * direction).sum()

            point = torch.zeros((), dtype=DTYPE)
            for order, atol in ((1, 2e-8), (2, 2e-6), (3, 2e-4)):
                reference = _directional_value(
                    lambda p: evaluate(p, False), point, order
                )
                actual = _directional_value(
                    lambda p: evaluate(p, True), point, order
                )
                reports[name][f"{variable}_{order}"] = _assert_close(
                    actual,
                    reference,
                    atol=atol,
                    rtol=3e-7,
                    label=f"{name} matrix-shift {variable} derivative {order}",
                )
        active = torch.ones(2, dtype=torch.bool)

        def shift_reference(parameter: torch.Tensor) -> torch.Tensor:
            return (
                _matrix_fermi(hamiltonian, parameter, kt, active) * direction
            ).sum()

        def shift_candidate(parameter: torch.Tensor) -> torch.Tensor:
            initial = _matrix_fermi(
                hamiltonian, torch.zeros_like(parameter), kt, active
            )
            return (_shift_density(initial, parameter, kt) * direction).sum()

        _check_transform_parity(
            output,
            f"{name} explicit chemical-potential shift",
            shift_candidate,
            shift_reference,
            delta,
        )
    _emit(output, "matrix_shift_derivatives", max_abs=reports)


def _check_shift_conditioning(output: Path) -> None:
    """Record finite behavior and conditioning over increasingly large shifts."""
    hamiltonian = torch.diag(torch.tensor([-4.0, 0.1, 5.0], dtype=DTYPE))
    active = torch.ones(3, dtype=torch.bool)
    kt = torch.tensor(0.2, dtype=DTYPE)
    density = _matrix_fermi(
        hamiltonian, torch.zeros((), dtype=DTYPE), kt, active
    )
    entries = []
    max_ratio = -0.25 * torch.log(
        torch.tensor(torch.finfo(DTYPE).eps, dtype=DTYPE)
    )
    for delta_value in (
        -200.0,
        -20.0,
        -4.0,
        -0.2,
        -1e-8,
        0.0,
        1e-8,
        0.2,
        4.0,
        20.0,
        200.0,
    ):
        delta = torch.tensor(delta_value, dtype=DTYPE)
        ratio = -delta / kt
        coefficient = torch.expm1(ratio)
        eye = torch.eye(3, dtype=DTYPE)
        denominator = eye + coefficient * (eye - density)
        updated = _shift_density(density, delta, kt)
        raw_updated = _raw_shift_density(density, delta, kt)
        direct = _matrix_fermi(hamiltonian, delta, kt, active)
        denominator_is_finite = bool(torch.isfinite(denominator).all())
        reference_difference = (
            float((updated - direct).abs().max())
            if torch.isfinite(updated).all() and torch.isfinite(direct).all()
            else float("inf")
        )
        raw_reference_difference = (
            float((raw_updated - direct).abs().max())
            if torch.isfinite(raw_updated).all()
            and torch.isfinite(direct).all()
            else float("inf")
        )
        entries.append(
            {
                "delta": delta_value,
                "delta_over_kt": float(delta / kt),
                "finite": bool(torch.isfinite(updated).all()),
                "condition": (
                    float(torch.linalg.cond(denominator))
                    if denominator_is_finite
                    else float("inf")
                ),
                "occupations": updated.diagonal().tolist(),
                "unguarded_finite": bool(torch.isfinite(raw_updated).all()),
                "unguarded_occupations": raw_updated.diagonal().tolist(),
                "direct_matrix_fermi_max_abs": reference_difference,
                "unguarded_direct_max_abs": raw_reference_difference,
                "outside_candidate_guard": abs(delta_value / 0.2)
                > float(max_ratio),
            }
        )
    _emit(output, "shift_conditioning", cases=entries)

    boundary = float(max_ratio)
    boundary_cases = []
    for sign in (-1.0, 1.0):
        for multiplier, expected_valid in (
            (1.0 - 1e-8, True),
            (1.0, True),
            (1.0 + 1e-8, False),
        ):
            ratio = sign * max_ratio * multiplier
            delta = -ratio * kt
            updated = _shift_density(density, delta, kt)
            direct = _matrix_fermi(hamiltonian, delta, kt, active)
            is_valid = bool(torch.isfinite(updated).all())
            assert (
                is_valid is expected_valid
            ), f"shift guard boundary misclassified ratio={float(ratio)}"
            error = None
            if expected_valid:
                error = _assert_close(
                    updated,
                    direct,
                    atol=2e-10,
                    rtol=2e-10,
                    label="matrix shift guard boundary parity",
                )
            boundary_cases.append(
                {
                    "ratio": float(ratio),
                    "expected_valid": expected_valid,
                    "finite": is_valid,
                    "direct_max_abs": error,
                }
            )
    _emit(
        output,
        "shift_guard_boundary",
        max_abs_ratio=boundary,
        worst_case_condition_bound=float(torch.exp(max_ratio)),
        density_spectrum_assumption="symmetric D with eigenvalues in [0, 1]",
        cases=boundary_cases,
    )


def _time_operation(function: Any, repeats: int = 7) -> dict[str, float]:
    """Return median and range of a warmed CPU operation in seconds."""
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def _benchmark_trace_square(output: Path) -> None:
    """Measure the matrix-multiply versus elementwise slope identities."""
    records = []
    for size in (32, 128, 255):
        seed = 2100 + size
        generator = torch.Generator().manual_seed(seed)
        raw = torch.randn(size, size, dtype=DTYPE, generator=generator)
        rotation, _ = torch.linalg.qr(raw)
        occupations = torch.linspace(1e-8, 1.0 - 1e-8, size, dtype=DTYPE)
        density = rotation @ torch.diag(occupations) @ rotation.T
        temperatures = torch.tensor(0.2, dtype=DTYPE)
        reference_value = _matmul_slope(density, temperatures)
        candidate_value = _trace_square_slope(density, temperatures)
        value_error = _assert_close(
            candidate_value,
            reference_value,
            atol=2e-11,
            rtol=2e-11,
            label=f"trace-squared slope parity at dimension {size}",
        )

        reference_grad = torch.func.grad(
            lambda matrix: _matmul_slope(matrix, temperatures).sum()
        )
        candidate_grad = torch.func.grad(
            lambda matrix: _trace_square_slope(matrix, temperatures).sum()
        )
        gradient_error = _assert_close(
            candidate_grad(density),
            reference_grad(density),
            atol=2e-11,
            rtol=2e-11,
            label=f"trace-squared slope gradient at dimension {size}",
        )

        reference_jvp = lambda: torch.func.jvp(
            lambda matrix: _matmul_slope(matrix, temperatures).sum(),
            (density,),
            (torch.ones_like(density),),
        )[1]
        candidate_jvp = lambda: torch.func.jvp(
            lambda matrix: _trace_square_slope(matrix, temperatures).sum(),
            (density,),
            (torch.ones_like(density),),
        )[1]
        jvp_error = _assert_close(
            candidate_jvp(),
            reference_jvp(),
            atol=2e-11,
            rtol=2e-11,
            label=f"trace-squared slope JVP at dimension {size}",
        )

        repeats = 9 if size <= 128 else 5
        reference_time = _time_operation(
            lambda: _matmul_slope(density, temperatures), repeats
        )
        candidate_time = _time_operation(
            lambda: _trace_square_slope(density, temperatures), repeats
        )
        reference_grad_time = _time_operation(
            lambda: reference_grad(density), repeats
        )
        candidate_grad_time = _time_operation(
            lambda: candidate_grad(density), repeats
        )
        record = {
            "dimension": size,
            "value_max_abs": value_error,
            "gradient_max_abs": gradient_error,
            "jvp_max_abs": jvp_error,
            "matrix_multiply_forward": reference_time,
            "trace_square_forward": candidate_time,
            "matrix_multiply_reverse": reference_grad_time,
            "trace_square_reverse": candidate_grad_time,
        }
        records.append(record)
        _emit(output, "trace_square_performance", **record)
    _emit(output, "trace_square_performance_summary", dimensions=records)


def _run(output: Path) -> None:
    """Run algebraic identity, correction, derivative, and slope checks."""
    _configure(output)
    dtype = DTYPE
    kt = torch.tensor(0.2, dtype=dtype)
    electrons = torch.tensor([1.3, 0.8], dtype=dtype)
    angle = torch.tensor(0.43, dtype=dtype)
    rotation = torch.stack(
        [
            torch.stack([angle.cos(), -angle.sin()]),
            torch.stack([angle.sin(), angle.cos()]),
        ]
    )
    base = torch.diag(torch.tensor([-0.15, 0.22], dtype=dtype))
    hamiltonian = rotation @ base @ rotation.T
    delta = torch.tensor([0.013, -0.021], dtype=dtype)
    active = torch.ones(2, dtype=torch.bool)
    density = _matrix_fermi(
        hamiltonian.expand(2, 2, 2),
        torch.zeros(2, dtype=dtype),
        kt.expand(2),
        active.expand(2, 2),
    )
    shifted = _shift_density(density, delta, kt.expand(2))
    direct = _matrix_fermi(
        hamiltonian.expand(2, 2, 2), delta, kt.expand(2), active.expand(2, 2)
    )
    shift_error = _assert_close(
        shifted, direct, atol=3e-12, rtol=3e-12, label="matrix shift values"
    )
    _emit(output, "matrix_shift_identity", max_abs=shift_error)

    # The full correction candidate is compared with the accepted path using
    # identical filling state and physical inputs.
    state = _state(hamiltonian, electrons, kt)
    accepted = fermi_density_matrix(
        hamiltonian, electrons, kt, state, diff_steps=2
    )
    candidate = _candidate_density(hamiltonian, electrons, kt, state)
    correction_error = _assert_close(
        candidate,
        accepted,
        atol=2e-10,
        rtol=2e-10,
        label="two matrix-shift correction density",
    )
    count = candidate.diagonal(dim1=-2, dim2=-1).sum(-1)
    _assert_close(
        count,
        electrons.sum(),
        atol=2e-11,
        rtol=2e-11,
        label="candidate total electron count",
    )
    _emit(
        output,
        "candidate_density",
        max_abs=correction_error,
        electron_count=float(count.detach()),
    )

    zero = torch.zeros((), dtype=dtype)
    direction_for_slope = torch.tensor(
        [[0.1, -0.13], [-0.13, 0.07]], dtype=dtype
    )

    def free_energy_value(
        parameter: torch.Tensor, variable: str, reuse: bool
    ) -> torch.Tensor:
        h = hamiltonian
        e = electrons
        temperature = kt
        if variable == "matrix":
            h = hamiltonian + parameter * direction_for_slope
        elif variable == "temperature":
            temperature = kt + parameter * 0.015
        else:
            e = electrons + parameter * torch.tensor([0.07, -0.03], dtype=dtype)
        current_state = _state(h, e, temperature)
        value = (
            _candidate_free_energy_trace(h, e, temperature, current_state)
            if reuse
            else fermi_free_energy_trace(
                h,
                e,
                temperature,
                current_state,
                l_inv=torch.eye(h.shape[-1], dtype=dtype),
                diff_steps=2,
            )
        )
        return value.sum()

    free_energy_errors: dict[str, dict[str, float]] = {}
    for variable in ("matrix", "temperature", "electron_count"):
        free_energy_errors[variable] = {}
        for order, atol in ((1, 2e-7), (2, 2e-5), (3, 2e-3)):
            expected_value = _directional_value(
                lambda p: free_energy_value(p, variable, False), zero, order
            )
            actual_value = _directional_value(
                lambda p: free_energy_value(p, variable, True), zero, order
            )
            free_energy_errors[variable][str(order)] = _assert_close(
                actual_value,
                expected_value,
                atol=atol,
                rtol=5e-7,
                label=f"free energy {variable} derivative order {order}",
            )
    _emit(
        output,
        "free_energy_matrix_shift",
        max_abs=free_energy_errors,
    )

    # Compare the independent trace-squared identity to the existing slope.
    slope_reference = _matmul_slope(density, kt.expand(2))
    slope_candidate = _trace_square_slope(density, kt.expand(2))
    slope_error = _assert_close(
        slope_candidate,
        slope_reference,
        atol=3e-14,
        rtol=3e-14,
        label="trace-square slope values",
    )

    def slope_objective(parameter: torch.Tensor, fast: bool) -> torch.Tensor:
        trial_h = hamiltonian + parameter * direction_for_slope
        trial = _matrix_fermi(
            trial_h.expand(2, 2, 2),
            torch.zeros(2, dtype=dtype),
            kt.expand(2),
            active.expand(2, 2),
        )
        return (
            _trace_square_slope(trial, kt.expand(2))
            if fast
            else _matmul_slope(trial, kt.expand(2))
        ).sum()

    slope_derivatives: dict[str, float] = {}
    for name, derivative in (
        ("grad", torch.func.grad(lambda p: slope_objective(p, True))),
        (
            "grad_reference",
            torch.func.grad(lambda p: slope_objective(p, False)),
        ),
        (
            "second",
            torch.func.grad(
                torch.func.grad(lambda p: slope_objective(p, True))
            ),
        ),
        (
            "second_reference",
            torch.func.grad(
                torch.func.grad(lambda p: slope_objective(p, False))
            ),
        ),
        (
            "third",
            torch.func.grad(
                torch.func.grad(
                    torch.func.grad(lambda p: slope_objective(p, True))
                )
            ),
        ),
        (
            "third_reference",
            torch.func.grad(
                torch.func.grad(
                    torch.func.grad(lambda p: slope_objective(p, False))
                )
            ),
        ),
    ):
        slope_derivatives[name] = float(derivative(zero))
    _assert_close(
        torch.tensor(slope_derivatives["grad"], dtype=dtype),
        torch.tensor(slope_derivatives["grad_reference"], dtype=dtype),
        atol=2e-12,
        rtol=2e-12,
        label="trace-square first derivative",
    )
    _assert_close(
        torch.tensor(slope_derivatives["second"], dtype=dtype),
        torch.tensor(slope_derivatives["second_reference"], dtype=dtype),
        atol=2e-12,
        rtol=2e-12,
        label="trace-square second derivative",
    )
    _assert_close(
        torch.tensor(slope_derivatives["third"], dtype=dtype),
        torch.tensor(slope_derivatives["third_reference"], dtype=dtype),
        atol=2e-11,
        rtol=2e-11,
        label="trace-square third derivative",
    )
    _emit(
        output,
        "trace_square_slope",
        max_abs=slope_error,
        derivative_values=slope_derivatives,
        matrix_size=2,
    )

    # Hamiltonian derivatives compare the complete correction sequences. A
    # mismatch raises immediately and rejects the candidate.
    direction = torch.tensor([[0.3, 0.4], [0.4, -0.2]], dtype=dtype)

    def directional(order: int, use_candidate: bool) -> torch.Tensor:
        def scalar(t: torch.Tensor) -> torch.Tensor:
            h = hamiltonian + t * direction
            fermi_state = _state(h, electrons, kt)
            rho = (
                _candidate_density(h, electrons, kt, fermi_state)
                if use_candidate
                else fermi_density_matrix(h, electrons, kt, fermi_state)
            )
            return (rho * direction).sum()

        derivative = scalar
        for _ in range(order):
            previous = derivative

            def derivative(t: torch.Tensor, fn=previous) -> torch.Tensor:
                return torch.func.jvp(fn, (t,), (torch.ones_like(t),))[1]

        return derivative(zero)

    derivative_diffs = {}
    for order in (1, 2, 3):
        reference_value = directional(order, False)
        candidate_value = directional(order, True)
        error = _assert_close(
            candidate_value,
            reference_value,
            atol={1: 2e-9, 2: 2e-7, 3: 2e-4}[order],
            rtol=2e-8,
            label=f"candidate Hamiltonian directional derivative order {order}",
        )
        derivative_diffs[str(order)] = error
    _emit(output, "candidate_derivatives", max_abs=derivative_diffs)

    def corrected_candidate(parameter: torch.Tensor) -> torch.Tensor:
        h = hamiltonian + parameter * direction_for_slope
        current_state = _state(h, electrons, kt)
        density_matrix = _candidate_density(
            h,
            electrons,
            kt,
            current_state,
            use_trace_square=True,
        )
        return (density_matrix * direction_for_slope).sum()

    def corrected_reference(parameter: torch.Tensor) -> torch.Tensor:
        h = hamiltonian + parameter * direction_for_slope
        current_state = _state(h, electrons, kt)
        density_matrix = fermi_density_matrix(
            h, electrons, kt, current_state, diff_steps=2
        )
        return (density_matrix * direction_for_slope).sum()

    _check_transform_parity(
        output,
        "complete two-correction density",
        corrected_candidate,
        corrected_reference,
        zero,
    )

    _check_physical_fixture(
        output,
        "near-degenerate",
        torch.diag(torch.tensor([-0.3, -0.29999999, 0.5], dtype=dtype)),
        torch.tensor([1.2, 0.8], dtype=dtype),
        0.2,
    )
    _check_physical_fixture(
        output,
        "exact-degenerate",
        torch.diag(torch.tensor([-0.3, -0.3, 0.5], dtype=dtype)),
        torch.tensor([1.2, 0.8], dtype=dtype),
        0.2,
    )
    _check_physical_fixture(
        output,
        "padded-noncontiguous",
        torch.diag(torch.tensor([-0.3, 80.0, 0.5], dtype=dtype)),
        torch.tensor([1.2, 0.8], dtype=dtype),
        0.2,
        torch.tensor([True, False, True]),
    )
    _check_physical_fixture(
        output,
        "wide-gap",
        torch.diag(torch.tensor([-8.0, 8.0, 40.0], dtype=dtype)),
        torch.tensor([1.0, 0.0], dtype=dtype),
        0.2,
    )
    _check_physical_fixture(
        output,
        "low-temperature-partial",
        torch.diag(torch.tensor([-0.01, 0.015, 0.4], dtype=dtype)),
        torch.tensor([0.8, 0.6], dtype=dtype),
        0.002,
    )
    overlap = torch.tensor([[1.2, 0.1], [0.1, 0.9]], dtype=dtype)
    factor = torch.linalg.cholesky(overlap)
    l_inv = torch.linalg.inv(factor)
    generalized_rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=dtype)
    orthogonal_h = (
        generalized_rotation
        @ torch.diag(torch.tensor([-0.15, 0.22], dtype=dtype))
        @ generalized_rotation.T
    )
    generalized_h = factor @ orthogonal_h @ factor.T
    _check_physical_fixture(
        output,
        "generalized-overlap",
        generalized_h,
        electrons,
        0.2,
        l_inv=l_inv,
    )
    _check_lane_independence(output)
    _check_shift_derivatives(output)
    _check_shift_conditioning(output)
    _benchmark_trace_square(output)


def main() -> None:
    """Parse diagnostic options and execute the bounded comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _run(args.output)


if __name__ == "__main__":
    main()
