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
"""Tests for the direct-RHS matrix-Fermi tanh-doubling candidate."""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from benchmarks.e6c1_w2c6_tanh_solve import (
    _fixture,
    _matrix_fermi_direct_rhs,
    _matrix_fermi_identity_rhs,
)
from dxtb._src.scf.functional.spectral import (
    _FERMI_MAX_SCALED_NORM,
    _matrix_fermi,
    _matrix_fermi_eager,
    _prepare_matrix,
)

DTYPE = torch.float64


def _assert_pair(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2.0e-11,
    rtol: float = 2.0e-11,
    label: str,
) -> None:
    """Assert finite parity or the same lane-local NaN pattern."""
    torch.testing.assert_close(
        torch.isfinite(actual),
        torch.isfinite(expected),
        msg=f"{label} finiteness",
    )
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if finite.any():
        torch.testing.assert_close(
            actual[finite],
            expected[finite],
            atol=atol,
            rtol=rtol,
            msg=label,
        )


@pytest.mark.parametrize("dimension", [1, 2, 8, 32])
def test_direct_rhs_matches_production_on_rotated_spectra(
    dimension: int,
) -> None:
    """The direct-RHS recurrence preserves rotated matrix-Fermi values."""
    matrix, shift, kt, active = _fixture(dimension, 16)
    actual = _matrix_fermi_direct_rhs(matrix, shift, kt, active)
    expected = _matrix_fermi_identity_rhs(matrix, shift, kt, active)
    _assert_pair(actual, expected, label=f"rotated N={dimension}")
    torch.testing.assert_close(
        _matrix_fermi(matrix, shift, kt, active),
        actual,
        atol=2.0e-11,
        rtol=2.0e-11,
        msg="production path uses the direct-RHS recurrence",
    )
    with torch.no_grad():
        eager = _matrix_fermi_eager(matrix, shift, kt, active)
    torch.testing.assert_close(
        eager, _matrix_fermi(matrix, shift, kt, active), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(actual, actual.T, atol=2.0e-14, rtol=0.0)


@pytest.mark.parametrize("power", [11, 14, 15, 16, 21, 22, 23, 29, 30])
def test_direct_rhs_preserves_supported_scaling_powers(power: int) -> None:
    """The candidate matches production across accepted scaling boundaries."""
    # Keep the partially occupied state decoupled from the very remote
    # frontier used to select the outer power. A rotated matrix at p30 has
    # precision loss in its input construction that dominates this comparison.
    matrix = torch.diag(
        torch.tensor(
            [-0.17, 0.75 * _FERMI_MAX_SCALED_NORM * (2.0**power)],
            dtype=DTYPE,
        )
    )
    bound = matrix.abs().sum(-1).amax()
    selected_power = torch.ceil(
        torch.log2(bound.clamp_min(1.0) / _FERMI_MAX_SCALED_NORM)
    )
    assert int(selected_power) == power
    shift = torch.tensor(0.0, dtype=DTYPE)
    kt = torch.tensor(1.0, dtype=DTYPE)
    active = torch.ones(2, dtype=torch.bool)
    actual = _matrix_fermi_direct_rhs(matrix, shift, kt, active)
    expected = _matrix_fermi_identity_rhs(matrix, shift, kt, active)
    with torch.no_grad():
        eager = _matrix_fermi_eager(matrix, shift, kt, active)
    assert torch.isfinite(actual).all(), f"p{power} must be supported"
    assert torch.isfinite(_matrix_fermi(matrix, shift, kt, active)).all()
    _assert_pair(
        eager, _matrix_fermi(matrix, shift, kt, active), label=f"eager p{power}"
    )
    _assert_pair(
        actual, expected, atol=1.0e-10, rtol=1.0e-10, label=f"p{power}"
    )


def test_eager_adaptive_rejects_transformed_or_differentiable_inputs() -> None:
    """The data-dependent schedule cannot enter differentiable production paths."""
    matrix, shift, kt, active = _fixture(2, 11)
    with pytest.raises(RuntimeError, match="detached eager primal"):
        _matrix_fermi_eager(matrix.requires_grad_(True), shift, kt, active)

    with torch.no_grad(), pytest.raises(RuntimeError, match="torch.func"):
        torch.func.jvp(
            lambda value: _matrix_fermi_eager(value, shift, kt, active),
            (matrix,),
            (torch.ones_like(matrix),),
        )


@pytest.mark.parametrize("power", [14, 15, 21, 22, 23, 29, 30])
def test_direct_rhs_frontier_derivatives_at_supported_powers(
    power: int,
) -> None:
    """A partially occupied frontier keeps derivatives nontrivial at each scale."""
    frontier = torch.tensor(-0.17, dtype=DTYPE)
    remote = torch.tensor(
        0.75 * _FERMI_MAX_SCALED_NORM * (2.0**power), dtype=DTYPE
    )
    shift = torch.tensor(0.0, dtype=DTYPE)
    kt = torch.tensor(1.0, dtype=DTYPE)
    active = torch.ones(2, dtype=torch.bool)

    def occupation(
        value: torch.Tensor,
        implementation: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        matrix = torch.diag(torch.stack((value, remote)))
        return implementation(matrix, shift, kt, active)[0, 0]

    def actual(value: torch.Tensor) -> torch.Tensor:
        return occupation(value, _matrix_fermi_direct_rhs)

    def expected(value: torch.Tensor) -> torch.Tensor:
        return occupation(value, _matrix_fermi_identity_rhs)

    point = frontier

    torch.testing.assert_close(
        actual(point), expected(point), atol=1.0e-10, rtol=1.0e-10
    )
    for order in (1, 2, 3):
        actual_derivative = actual
        expected_derivative = expected
        for _ in range(order):
            actual_derivative = torch.func.jacfwd(actual_derivative)
            expected_derivative = torch.func.jacfwd(expected_derivative)
        torch.testing.assert_close(
            actual_derivative(point),
            expected_derivative(point),
            atol=2.0e-8 if order < 3 else 2.0e-5,
            rtol=2.0e-8,
            msg=f"frontier derivative order {order}, p{power}",
        )


def test_direct_rhs_matches_independent_diagonal_scalar_oracle() -> None:
    """Diagonal occupations agree with scalar Fermi-Dirac values."""
    energies = torch.tensor([-0.25, 0.25], dtype=DTYPE)
    matrix = torch.diag(energies)
    shift = torch.tensor(0.04, dtype=DTYPE)
    kt = torch.tensor(0.3, dtype=DTYPE)
    active = torch.ones(2, dtype=torch.bool)
    expected = torch.diag(torch.sigmoid(-(energies - shift) / kt))
    actual = _matrix_fermi_direct_rhs(matrix, shift, kt, active)
    torch.testing.assert_close(actual, expected, atol=2.0e-13, rtol=2.0e-13)

    def candidate_scalar(energy: torch.Tensor) -> torch.Tensor:
        orbital = torch.diag(energy.reshape(1))
        return _matrix_fermi_direct_rhs(orbital, shift, kt, active[:1])[0, 0]

    def scalar_oracle(energy: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-(energy - shift) / kt)

    for order in (1, 2, 3):
        candidate_derivative = candidate_scalar
        oracle_derivative = scalar_oracle
        for _ in range(order):
            candidate_derivative = torch.func.jacfwd(candidate_derivative)
            oracle_derivative = torch.func.jacfwd(oracle_derivative)
        torch.testing.assert_close(
            candidate_derivative(energies[:1]).reshape(-1),
            oracle_derivative(energies[:1]).reshape(-1),
            atol=2.0e-8 if order < 3 else 2.0e-5,
            rtol=2.0e-8,
            msg=f"independent scalar derivative order {order}",
        )


@pytest.mark.parametrize("frontier_gap", [0.0, 1.0e-12])
def test_direct_rhs_exact_and_near_degeneracy(frontier_gap: float) -> None:
    """Exact and near-degenerate rotated spectra preserve density values."""
    angle = torch.tensor(0.29, dtype=DTYPE)
    rotation = torch.stack(
        (
            torch.stack((angle.cos(), -angle.sin())),
            torch.stack((angle.sin(), angle.cos())),
        )
    )
    spectrum = torch.diag(
        torch.tensor([-0.2, -0.2 + frontier_gap], dtype=DTYPE)
    )
    matrix = rotation @ spectrum @ rotation.T
    shift = torch.tensor(0.03, dtype=DTYPE)
    kt = torch.tensor(0.18, dtype=DTYPE)
    active = torch.ones(2, dtype=torch.bool)
    _assert_pair(
        _matrix_fermi_direct_rhs(matrix, shift, kt, active),
        _matrix_fermi_identity_rhs(matrix, shift, kt, active),
        atol=2.0e-12,
        rtol=2.0e-12,
        label=f"frontier gap {frontier_gap}",
    )


def test_direct_rhs_float32_parity() -> None:
    """The algebraic change remains within the float32 kernel precision."""
    matrix = torch.tensor([[-0.2, 0.04], [0.04, 0.31]], dtype=torch.float32)
    shift = torch.tensor(0.02, dtype=torch.float32)
    kt = torch.tensor(0.25, dtype=torch.float32)
    active = torch.ones(2, dtype=torch.bool)
    actual = _matrix_fermi_direct_rhs(matrix, shift, kt, active)
    expected = _matrix_fermi_identity_rhs(matrix, shift, kt, active)
    _assert_pair(actual, expected, atol=2.0e-6, rtol=2.0e-6, label="float32")


def test_direct_rhs_preserves_cancellation_and_invalid_range_behavior() -> None:
    """The candidate retains overflow checks and p30 unsupported behavior."""
    active = torch.ones(1, dtype=torch.bool)
    hamiltonian = torch.tensor([[1000.0]], dtype=DTYPE)
    chemical_potential = torch.tensor(990.0, dtype=DTYPE)
    kt = torch.tensor(1.0, dtype=DTYPE)
    expected_occupation = torch.sigmoid(
        -(hamiltonian[0, 0] - chemical_potential) / kt
    )
    direct_occupation = _matrix_fermi_direct_rhs(
        hamiltonian, chemical_potential, kt, active
    )[0, 0]
    torch.testing.assert_close(
        direct_occupation,
        expected_occupation,
        atol=2.0e-13,
        rtol=2.0e-13,
        msg="accepted H=1000, shift=990 cancellation case",
    )
    _assert_pair(
        _matrix_fermi_direct_rhs(hamiltonian, chemical_potential, kt, active),
        _matrix_fermi_identity_rhs(hamiltonian, chemical_potential, kt, active),
        label="accepted H=1000, shift=990 production parity",
    )

    matrix = torch.tensor([[1.0e300]], dtype=DTYPE)
    shift = torch.tensor(1.0e300, dtype=DTYPE)
    _assert_pair(
        _matrix_fermi_direct_rhs(matrix, shift, kt, active),
        _matrix_fermi_identity_rhs(matrix, shift, kt, active),
        label="large cancellation",
    )

    unsupported = torch.tensor([[4.0 * 2.0**31]], dtype=DTYPE)
    unsupported_result = _matrix_fermi_direct_rhs(
        unsupported, torch.tensor(0.0, dtype=DTYPE), kt, active
    )
    assert torch.isnan(unsupported_result).all(), "p31 must be rejected"
    production_unsupported = _matrix_fermi(
        unsupported, torch.tensor(0.0, dtype=DTYPE), kt, active
    )
    assert torch.isnan(production_unsupported).all()
    with torch.no_grad():
        eager_unsupported = _matrix_fermi_eager(
            unsupported, torch.tensor(0.0, dtype=DTYPE), kt, active
        )
    assert torch.isnan(eager_unsupported).all()
    _assert_pair(
        unsupported_result,
        _matrix_fermi_identity_rhs(
            unsupported, torch.tensor(0.0, dtype=DTYPE), kt, active
        ),
        label="p31 unsupported boundary",
    )


def test_direct_rhs_preserves_padding_invalid_lanes_and_masks() -> None:
    """Padded nonfinite entries and invalid lanes remain lane-local."""
    matrix = torch.stack(
        (
            torch.tensor([[0.1, 0.02], [0.02, -0.15]], dtype=DTYPE),
            torch.tensor(
                [[0.3, float("nan")], [float("nan"), float("nan")]], dtype=DTYPE
            ),
            torch.eye(2, dtype=DTYPE),
        )
    )
    shift = torch.tensor([0.01, 0.02, float("nan")], dtype=DTYPE)
    kt = torch.tensor([0.2, 0.3, 0.4], dtype=DTYPE)
    active = torch.tensor([[True, True], [True, False], [True, True]])
    expected = _matrix_fermi_identity_rhs(matrix, shift, kt, active)
    actual = _matrix_fermi_direct_rhs(matrix, shift, kt, active)
    with torch.no_grad():
        eager = _matrix_fermi_eager(matrix, shift, kt, active)
    _assert_pair(actual, expected, label="masked mixed lanes")
    _assert_pair(eager, actual, label="adaptive masked mixed lanes")
    assert torch.isfinite(actual[0]).all()
    assert torch.isfinite(actual[1]).all()
    assert torch.isnan(actual[2]).all()

    zero_temperature = _matrix_fermi_direct_rhs(
        matrix[:1],
        torch.zeros(1, dtype=DTYPE),
        torch.zeros(1, dtype=DTYPE),
        active[:1],
    )
    assert torch.isnan(zero_temperature).all()


def test_direct_rhs_generalized_overlap_preparation() -> None:
    """The candidate agrees after the accepted overlap preparation path."""
    overlap = torch.tensor([[1.2, 0.1], [0.1, 0.9]], dtype=DTYPE)
    factor = torch.linalg.cholesky(overlap)
    l_inv = torch.linalg.inv(factor)
    hamiltonian = torch.tensor([[-0.2, 0.03], [0.03, 0.26]], dtype=DTYPE)
    _, spectral, active = _prepare_matrix(hamiltonian, l_inv, None)
    shift = torch.tensor(0.015, dtype=DTYPE)
    kt = torch.tensor(0.2, dtype=DTYPE)
    _assert_pair(
        _matrix_fermi_direct_rhs(spectral, shift, kt, active),
        _matrix_fermi_identity_rhs(spectral, shift, kt, active),
        label="generalized overlap prepared spectrum",
    )


def test_direct_rhs_first_through_third_order_transform_parity() -> None:
    """Forward and reverse transforms agree through third order."""
    matrix, shift, kt, active = _fixture(2, 14)
    direction = torch.tensor([[0.13, -0.08], [-0.08, 0.17]], dtype=DTYPE)
    cotangent = torch.tensor([[0.4, -0.2], [-0.2, 0.3]], dtype=DTYPE)

    def candidate(value: torch.Tensor) -> torch.Tensor:
        return _matrix_fermi_direct_rhs(value, shift, kt, active)

    def reference(value: torch.Tensor) -> torch.Tensor:
        return _matrix_fermi_identity_rhs(value, shift, kt, active)

    _assert_pair(candidate(matrix), reference(matrix), label="value")
    torch.testing.assert_close(
        torch.func.grad(lambda value: (candidate(value) * cotangent).sum())(
            matrix
        ),
        torch.func.grad(lambda value: (reference(value) * cotangent).sum())(
            matrix
        ),
        atol=2.0e-9,
        rtol=2.0e-9,
        msg="reverse gradient",
    )
    torch.testing.assert_close(
        torch.func.jvp(candidate, (matrix,), (direction,))[1],
        torch.func.jvp(reference, (matrix,), (direction,))[1],
        atol=2.0e-9,
        rtol=2.0e-9,
        msg="JVP",
    )
    torch.testing.assert_close(
        torch.func.jacfwd(candidate)(matrix),
        torch.func.jacrev(candidate)(matrix),
        atol=2.0e-8,
        rtol=2.0e-8,
        msg="candidate forward/reverse Jacobian consistency",
    )
    torch.testing.assert_close(
        torch.func.jacfwd(candidate)(matrix),
        torch.func.jacfwd(reference)(matrix),
        atol=2.0e-8,
        rtol=2.0e-8,
        msg="jacfwd parity",
    )
    torch.testing.assert_close(
        torch.func.jacrev(candidate)(matrix),
        torch.func.jacrev(reference)(matrix),
        atol=2.0e-8,
        rtol=2.0e-8,
        msg="jacrev parity",
    )

    def candidate_scalar(parameter: torch.Tensor) -> torch.Tensor:
        current = matrix + parameter * direction
        return (candidate(current) * cotangent).sum()

    def reference_scalar(parameter: torch.Tensor) -> torch.Tensor:
        current = matrix + parameter * direction
        return (reference(current) * cotangent).sum()

    zero = torch.zeros((), dtype=DTYPE)

    def directional(
        function: Callable[[torch.Tensor], torch.Tensor], order: int
    ):
        current = function
        for _ in range(order):
            previous = current

            def current(value: torch.Tensor, fn=previous) -> torch.Tensor:
                return torch.func.jvp(fn, (value,), (torch.ones_like(value),))[
                    1
                ]

        return current

    for order in (2, 3):
        torch.testing.assert_close(
            directional(candidate_scalar, order)(zero),
            directional(reference_scalar, order)(zero),
            atol=2.0e-4 if order == 3 else 2.0e-7,
            rtol=2.0e-8,
            msg=f"forward directional order {order}",
        )

    candidate_reverse = candidate_scalar
    reference_reverse = reference_scalar
    for order in range(3):
        candidate_reverse = torch.func.grad(candidate_reverse)
        reference_reverse = torch.func.grad(reference_reverse)
    torch.testing.assert_close(
        candidate_reverse(zero),
        reference_reverse(zero),
        atol=2.0e-4,
        rtol=2.0e-8,
        msg="reverse third derivative",
    )

    def candidate_scalar_matrix(value: torch.Tensor) -> torch.Tensor:
        return (candidate(value) * cotangent).sum()

    def reference_scalar_matrix(value: torch.Tensor) -> torch.Tensor:
        return (reference(value) * cotangent).sum()

    candidate_gradient = torch.func.grad(candidate_scalar_matrix)
    reference_gradient = torch.func.grad(reference_scalar_matrix)
    torch.testing.assert_close(
        torch.func.jvp(candidate_gradient, (matrix,), (direction,))[1],
        torch.func.jvp(reference_gradient, (matrix,), (direction,))[1],
        atol=2.0e-8,
        rtol=2.0e-8,
        msg="Hessian-vector product",
    )

    def candidate_first_directional(value: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(candidate_scalar_matrix, (value,), (direction,))[
            1
        ]

    def reference_first_directional(value: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(reference_scalar_matrix, (value,), (direction,))[
            1
        ]

    def candidate_second_directional(value: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(
            candidate_first_directional, (value,), (direction,)
        )[1]

    def reference_second_directional(value: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(
            reference_first_directional, (value,), (direction,)
        )[1]

    torch.testing.assert_close(
        torch.func.grad(
            lambda value: candidate_second_directional(value).sum()
        )(matrix),
        torch.func.grad(
            lambda value: reference_second_directional(value).sum()
        )(matrix),
        atol=2.0e-4,
        rtol=2.0e-8,
        msg="reverse-over-forward third derivative",
    )

    def candidate_shift(parameter: torch.Tensor) -> torch.Tensor:
        return (
            _matrix_fermi_direct_rhs(matrix, parameter, kt, active) * cotangent
        ).sum()

    def reference_shift(parameter: torch.Tensor) -> torch.Tensor:
        return (
            _matrix_fermi_identity_rhs(matrix, parameter, kt, active)
            * cotangent
        ).sum()

    def candidate_temperature(parameter: torch.Tensor) -> torch.Tensor:
        return (
            _matrix_fermi_direct_rhs(matrix, shift, parameter, active)
            * cotangent
        ).sum()

    def reference_temperature(parameter: torch.Tensor) -> torch.Tensor:
        return (
            _matrix_fermi_identity_rhs(matrix, shift, parameter, active)
            * cotangent
        ).sum()

    for candidate_fn, reference_fn, value in (
        (candidate_shift, reference_shift, shift),
        (candidate_temperature, reference_temperature, kt),
    ):
        for order in (1, 2, 3):
            torch.testing.assert_close(
                directional(candidate_fn, order)(value),
                directional(reference_fn, order)(value),
                atol=2.0e-4 if order == 3 else 2.0e-7,
                rtol=2.0e-8,
                msg=f"scalar input order {order}",
            )


def test_direct_rhs_vmap_mixed_powers_and_lane_independence() -> None:
    """Different scaling powers remain lane-local under nested vmap."""
    lanes = [_fixture(2, 11), _fixture(2, 16)]
    matrices = torch.stack([lane[0] for lane in lanes])
    shifts = torch.stack([lane[1] for lane in lanes])
    temperatures = torch.stack([lane[2] for lane in lanes])
    masks = torch.stack([lane[3] for lane in lanes])

    def evaluate(function: Callable[..., torch.Tensor], values: torch.Tensor):
        return torch.func.vmap(function)(values, shifts, temperatures, masks)

    mapped_candidate = evaluate(_matrix_fermi_direct_rhs, matrices)
    mapped_reference = evaluate(_matrix_fermi_identity_rhs, matrices)
    _assert_pair(mapped_candidate, mapped_reference, label="mixed-power vmap")
    serial_candidate = torch.stack(
        [
            _matrix_fermi_direct_rhs(
                matrices[index],
                shifts[index],
                temperatures[index],
                masks[index],
            )
            for index in range(2)
        ]
    )
    _assert_pair(mapped_candidate, serial_candidate, label="mapped/serial")

    jacobian = torch.func.jacrev(
        lambda values: evaluate(_matrix_fermi_direct_rhs, values)
    )(matrices)
    assert torch.isfinite(jacobian).all()
    assert jacobian[0, :, :, 1, :, :].abs().amax() == 0.0
    assert jacobian[1, :, :, 0, :, :].abs().amax() == 0.0

    reversed_result = evaluate(_matrix_fermi_direct_rhs, matrices.flip(0)).flip(
        0
    )
    _assert_pair(mapped_candidate, reversed_result, label="lane permutation")

    def single(
        matrix: torch.Tensor,
        shift: torch.Tensor,
        temperature: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        return _matrix_fermi_direct_rhs(matrix, shift, temperature, active)

    def inner(
        matrix_lanes: torch.Tensor,
        shift_lanes: torch.Tensor,
        temperature_lanes: torch.Tensor,
        active_lanes: torch.Tensor,
    ) -> torch.Tensor:
        return torch.func.vmap(single)(
            matrix_lanes, shift_lanes, temperature_lanes, active_lanes
        )

    nested_matrices = torch.stack((matrices, matrices.flip(0)))
    nested_shifts = torch.stack((shifts, shifts.flip(0)))
    nested_temperatures = torch.stack((temperatures, temperatures.flip(0)))
    nested_masks = torch.stack((masks, masks.flip(0)))
    nested = torch.func.vmap(inner)(
        nested_matrices, nested_shifts, nested_temperatures, nested_masks
    )
    expected_nested = torch.stack((mapped_candidate, mapped_candidate.flip(0)))
    _assert_pair(nested, expected_nested, label="nested vmap")
