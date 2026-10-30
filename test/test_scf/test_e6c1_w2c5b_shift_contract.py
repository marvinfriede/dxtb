# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract checks for the experimental W2c5 matrix-shift candidate."""

from __future__ import annotations

import pytest
import torch

from benchmarks import e6c1_w2c5_fermi_shift as shift_candidate
from benchmarks import e6c1_w2c5a_candidate_response as response_candidate
from dxtb._src.scf.functional.spectral import (
    _matrix_fermi,
    fermi_density_matrix,
    fermi_free_energy_trace,
)
from dxtb._src.wavefunction import filling

DTYPE = torch.float64


def _case(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    temperature: torch.Tensor,
    diff_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the shared fixed-trip filling state for a test case."""
    energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        energies,
        temperature,
        thr=1.0e-14,
        maxiter=300,
        diff_order=diff_order,
    )
    return state, energies


def _directional(
    function: object, point: torch.Tensor, order: int
) -> torch.Tensor:
    """Evaluate a scalar repeated forward directional derivative."""
    derivative = function
    for _ in range(order):
        previous = derivative
        derivative = lambda value, fn=previous: torch.func.jvp(
            fn, (value,), (torch.ones_like(value),)
        )[1]
    return derivative(point)


@pytest.mark.parametrize(
    ("diff_steps", "diff_order"),
    [(0, 0), (1, 1), (2, 3), (3, 7), (4, 15)],
)
def test_matrix_shift_candidate_respects_diff_steps(
    diff_steps: int, diff_order: int
) -> None:
    """The candidate performs and differentiates the requested corrections."""
    angle = torch.tensor(0.37, dtype=DTYPE)
    rotation = torch.stack(
        (
            torch.stack((angle.cos(), -angle.sin())),
            torch.stack((angle.sin(), angle.cos())),
        )
    )
    hamiltonian = (
        rotation
        @ torch.diag(torch.tensor([-0.23, 0.41], dtype=DTYPE))
        @ rotation.T
    )
    electrons = torch.tensor([1.27, 0.61], dtype=DTYPE)
    temperature = torch.tensor(0.19, dtype=DTYPE)

    state, _ = _case(hamiltonian, electrons, temperature, diff_order)
    expected = fermi_density_matrix(
        hamiltonian, electrons, temperature, state, diff_steps=diff_steps
    )
    actual = response_candidate._shifted_density(
        hamiltonian,
        electrons,
        temperature,
        state,
        diff_steps=diff_steps,
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-9, rtol=2.0e-9)
    torch.testing.assert_close(
        actual.diagonal().sum(), electrons.sum(), atol=2.0e-9, rtol=2.0e-9
    )

    h_direction = torch.tensor([[0.2, -0.11], [-0.11, 0.31]], dtype=DTYPE)
    scalar_zero = torch.zeros((), dtype=DTYPE)

    def evaluate(
        h_parameter: torch.Tensor,
        t_parameter: torch.Tensor,
        n_parameter: torch.Tensor,
        candidate: bool,
    ) -> torch.Tensor:
        h = hamiltonian + h_parameter * h_direction
        kt = temperature + 0.01 * t_parameter
        nel = electrons + n_parameter * torch.tensor([0.03, -0.02], dtype=DTYPE)
        current_state, _ = _case(h, nel, kt, diff_order)
        if candidate:
            density = response_candidate._shifted_density(
                h, nel, kt, current_state, diff_steps=diff_steps
            )
        else:
            density = fermi_density_matrix(
                h, nel, kt, current_state, diff_steps=diff_steps
            )
        return (density * h_direction).sum()

    for variable, parameters in (
        ("hamiltonian", (scalar_zero, scalar_zero, scalar_zero)),
        ("temperature", (scalar_zero, scalar_zero, scalar_zero)),
        ("electron_count", (scalar_zero, scalar_zero, scalar_zero)),
    ):
        index = {
            "hamiltonian": 0,
            "temperature": 1,
            "electron_count": 2,
        }[variable]

        def scalar(parameter: torch.Tensor, candidate: bool) -> torch.Tensor:
            current = list(parameters)
            current[index] = parameter
            return evaluate(*current, candidate)

        expected_grad = torch.func.grad(lambda p: scalar(p, False))(scalar_zero)
        actual_grad = torch.func.grad(lambda p: scalar(p, True))(scalar_zero)
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            atol=2.0e-7,
            rtol=2.0e-7,
            msg=f"{variable} derivative at diff_steps={diff_steps}",
        )

    exact_order = 2**diff_steps - 1
    for order in range(2, min(3, exact_order) + 1):
        expected_derivative = _directional(
            lambda p: evaluate(p, scalar_zero, scalar_zero, False),
            scalar_zero,
            order,
        )
        actual_derivative = _directional(
            lambda p: evaluate(p, scalar_zero, scalar_zero, True),
            scalar_zero,
            order,
        )
        torch.testing.assert_close(
            actual_derivative,
            expected_derivative,
            atol=2.0e-4 if order == 3 else 2.0e-6,
            rtol=2.0e-7,
            msg=f"Hamiltonian order {order} at diff_steps={diff_steps}",
        )


def test_zero_diff_steps_does_not_apply_a_shift_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero requested corrections do not execute the candidate shift solve."""
    hamiltonian = torch.diag(torch.tensor([-0.2, 0.3], dtype=DTYPE))
    electrons = torch.tensor([1.2, 0.7], dtype=DTYPE)
    temperature = torch.tensor(0.2, dtype=DTYPE)
    state, _ = _case(hamiltonian, electrons, temperature, diff_order=0)
    calls = 0
    original = shift_candidate._shift_density

    def count_calls(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(shift_candidate, "_shift_density", count_calls)
    actual = shift_candidate._candidate_density(
        hamiltonian, electrons, temperature, state, diff_steps=0
    )
    expected = fermi_density_matrix(
        hamiltonian, electrons, temperature, state, diff_steps=0
    )

    assert calls == 0
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_shift_guard_narrows_a_supported_direct_matrix_fermi_case() -> None:
    """A saturated rounded density loses information needed by a large shift."""
    kt = torch.tensor(1.0, dtype=DTYPE)
    active = torch.tensor([True], dtype=torch.bool)
    hamiltonian = torch.tensor([[[1000.0]]], dtype=DTYPE)
    zero_shift = torch.zeros(1, dtype=DTYPE)
    large_shift = torch.tensor([990.0], dtype=DTYPE)

    initial = _matrix_fermi(hamiltonian, zero_shift, kt.expand(1), active)
    candidate = shift_candidate._shift_density(
        initial, large_shift, kt.expand(1)
    )
    direct = _matrix_fermi(hamiltonian, large_shift, kt.expand(1), active)

    assert torch.isfinite(initial).all()
    assert torch.isfinite(direct).all()
    assert initial[0, 0, 0] == 0.0
    assert torch.isnan(candidate).all()
    assert 0.0 < float(direct[0, 0, 0]) < 1.0e-4


@pytest.mark.parametrize("diff_steps", [0, 1, 2, 3, 4])
def test_free_energy_probe_respects_diff_steps(diff_steps: int) -> None:
    """The experimental free-energy probe forwards the correction schedule."""
    hamiltonian = torch.tensor([[-0.13, 0.04], [0.04, 0.21]], dtype=DTYPE)
    electrons = torch.tensor([1.24, 0.66], dtype=DTYPE)
    temperature = torch.tensor(0.16, dtype=DTYPE)
    diff_order = (1 << diff_steps) - 1
    state, _ = _case(hamiltonian, electrons, temperature, diff_order)
    identity = torch.eye(2, dtype=DTYPE)
    expected = fermi_free_energy_trace(
        hamiltonian,
        electrons,
        temperature,
        state,
        l_inv=identity,
        diff_steps=diff_steps,
    )
    actual = shift_candidate._candidate_free_energy_trace(
        hamiltonian,
        electrons,
        temperature,
        state,
        diff_steps=diff_steps,
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-9, rtol=2.0e-9)
