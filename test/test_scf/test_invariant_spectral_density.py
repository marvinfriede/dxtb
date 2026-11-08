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
"""Focused tests for dxtb's invariant functional spectral density."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tad_mctc import storch
from tad_mctc.units import KELVIN2AU
from torch.func import jacfwd, jacrev, jvp

from dxtb._src.scf import spectral as spectral_module
from dxtb._src.scf.spectral import (
    _matrix_expm1,
    _matrix_fermi,
    _prepare_matrix,
    _projector,
    _solve_positive_definite,
    aufbau_density_matrix,
    fermi_density_matrix,
    fermi_free_energy_matrix,
    fermi_free_energy_trace,
)
from dxtb._src.scf.utils import get_density
from dxtb._src.wavefunction import filling


def _fermi_density(hamiltonian: torch.Tensor) -> torch.Tensor:
    """Evaluate the invariant Fermi matrix from dxtb's shared primal solve."""
    electrons = hamiltonian.new_tensor([1.0, 0.0])
    kt = hamiltonian.new_tensor(9.5e-4)
    energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        energies,
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
    )
    return fermi_density_matrix(
        hamiltonian,
        electrons,
        kt,
        state,
        diff_steps=filling._diff_steps(3),
    )


def test_electron_response_guard_matches_always_response_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primal-only response skip preserves mixed and higher derivatives."""
    hamiltonian = torch.tensor(
        [
            [-0.012, 0.002, 0.001, 0.0],
            [0.002, -0.001, 0.0004, 0.0],
            [0.001, 0.0004, 0.0002, 0.001],
            [0.0, 0.0, 0.001, 0.018],
        ],
        dtype=torch.float64,
    )
    electrons = torch.tensor([1.2, 0.8], dtype=torch.float64)
    temperature = torch.tensor(0.004, dtype=torch.float64)
    energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        energies,
        temperature,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
    )
    weight = torch.tensor(
        [[0.7, -0.1, 0.2, 0.3], [-0.2, 0.6, 0.1, -0.4]],
        dtype=torch.float64,
    )

    def observable(
        h: torch.Tensor, nel: torch.Tensor, kt: torch.Tensor
    ) -> torch.Tensor:
        density = fermi_density_matrix(
            h, nel, kt, state, diff_steps=filling._diff_steps(3)
        )
        return (density.diagonal(dim1=-2, dim2=-1) * weight).sum()

    original_guard = spectral_module._tracks_derivatives

    def evaluate(guard):
        monkeypatch.setattr(spectral_module, "_tracks_derivatives", guard)
        value = observable(hamiltonian, electrons, temperature)
        grad_h, grad_nel, grad_kt = torch.func.grad(
            observable, argnums=(0, 1, 2)
        )(hamiltonian, electrons, temperature)
        tangent_h = torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, -0.2, 0.0, 0.0],
                [0.0, 0.0, 0.15, 0.0],
                [0.0, 0.0, 0.0, -0.05],
            ],
            dtype=torch.float64,
        )
        tangent_nel = torch.tensor([0.3, -0.2], dtype=torch.float64)
        _, tangent = jvp(
            observable,
            (hamiltonian, electrons, temperature),
            (tangent_h, tangent_nel, torch.tensor(0.2, dtype=torch.float64)),
        )
        jacfwd_nel = jacfwd(observable, argnums=1)(
            hamiltonian, electrons, temperature
        )
        jacrev_nel = jacrev(observable, argnums=1)(
            hamiltonian, electrons, temperature
        )
        _, second_mixed = jvp(
            lambda h: torch.func.grad(observable, argnums=1)(
                h, electrons, temperature
            ),
            (hamiltonian,),
            (tangent_h,),
        )

        def along_count(alpha: torch.Tensor) -> torch.Tensor:
            return observable(
                hamiltonian,
                electrons + alpha * tangent_nel,
                temperature,
            )

        first = torch.func.grad(along_count)
        second = torch.func.grad(first)
        third = torch.func.grad(second)
        alpha = torch.tensor(0.0, dtype=torch.float64)
        derivatives = (first(alpha), second(alpha), third(alpha))

        batch_alphas = torch.tensor([-0.02, 0.0, 0.02], dtype=torch.float64)
        vmapped_first = torch.func.vmap(first)(batch_alphas)

        compiled_obs = torch.compile(along_count, backend="eager")
        alpha_leaf = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        compiled_val = compiled_obs(alpha_leaf)
        compiled_grad = torch.autograd.grad(compiled_val, alpha_leaf)[0]

        return (
            value,
            grad_h,
            grad_nel,
            grad_kt,
            tangent,
            jacfwd_nel,
            jacrev_nel,
            second_mixed,
            *derivatives,
            vmapped_first,
            compiled_val,
            compiled_grad,
        )

    reference = evaluate(lambda _: True)
    optimized = evaluate(original_guard)
    monkeypatch.setattr(spectral_module, "_tracks_derivatives", original_guard)
    for actual, expected in zip(optimized, reference):
        torch.testing.assert_close(actual, expected, atol=2.0e-10, rtol=2.0e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "spectrum,kt,shift,active",
    [
        ([-0.3, -0.3, 0.4, 0.0], 2.0e-4, -0.3, [True, True, True, False]),
        ([-1.0, -0.02, 0.021, 2.0], 0.004, 0.0, [True, False, True, True]),
        ([-0.1, -1.0e-7, 1.0e-7, 0.1], 0.02, 0.0, [True, True, True, True]),
        ([-0.08, -0.02, 0.03, 0.09], 0.06, 0.01, [True, True, True, True]),
    ],
    ids=(
        "saturated-degenerate",
        "wide-gap-padded",
        "near-degenerate",
        "high-temp-partial",
    ),
)
def test_trace_squared_slope_matches_legacy_density_product(
    dtype: torch.dtype,
    spectrum: list[float],
    kt: float,
    shift: float,
    active: list[bool],
) -> None:
    """Trace-squared Newton slope matches Tr(D-D²), including derivatives."""
    values = torch.tensor(spectrum, dtype=dtype)
    matrix = torch.diag(values)
    matrix[0, 2] = matrix[2, 0] = 0.15 * min(kt, 0.05)
    active_mask = torch.tensor(active, dtype=torch.bool)
    direction = torch.tensor(
        [
            [0.2, 0.0, -0.1, 0.0],
            [0.0, -0.3, 0.0, 0.0],
            [-0.1, 0.0, 0.15, 0.0],
            [0.0, 0.0, 0.0, -0.05],
        ],
        dtype=dtype,
    )
    kt_tensor = torch.tensor(kt, dtype=dtype)
    shift_tensor = torch.tensor(shift, dtype=dtype)
    _, spectral, prepared_active = _prepare_matrix(matrix, None, active_mask)

    def slopes(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = spectral + alpha * direction
        density = _matrix_fermi(
            candidate, shift_tensor, kt_tensor, prepared_active
        )
        old = (density - density @ density).diagonal().sum() / kt_tensor
        new = (density.diagonal().sum() - (density * density).sum()) / kt_tensor
        return old, new

    alpha = torch.tensor(0.0, dtype=dtype)
    old, new = slopes(alpha)
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(old, new, atol=64.0 * eps / kt, rtol=64.0 * eps)
    assert bool(torch.isfinite(torch.stack((old, new))).all())
    assert float(old) >= -64.0 * eps / kt
    assert float(new) >= -64.0 * eps / kt

    old_first = torch.func.grad(lambda value: slopes(value)[0])(alpha)
    new_first = torch.func.grad(lambda value: slopes(value)[1])(alpha)
    old_second = torch.func.grad(
        torch.func.grad(lambda value: slopes(value)[0])
    )(alpha)
    new_second = torch.func.grad(
        torch.func.grad(lambda value: slopes(value)[1])
    )(alpha)
    old_third = torch.func.grad(
        torch.func.grad(torch.func.grad(lambda value: slopes(value)[0]))
    )(alpha)
    new_third = torch.func.grad(
        torch.func.grad(torch.func.grad(lambda value: slopes(value)[1]))
    )(alpha)
    torch.testing.assert_close(
        old_first, new_first, atol=512.0 * eps / kt, rtol=512.0 * eps
    )
    torch.testing.assert_close(
        old_second, new_second, atol=2048.0 * eps / kt, rtol=2048.0 * eps
    )
    torch.testing.assert_close(
        old_third, new_third, atol=8192.0 * eps / kt, rtol=8192.0 * eps
    )


def test_aufbau_fractional_density_preserves_electron_response() -> None:
    """Fractional Aufbau filling remains differentiable in electron count."""
    hamiltonian = torch.diag(
        torch.tensor([-0.2, 0.1, 0.4], dtype=torch.float64)
    )
    electrons = torch.tensor(
        [1.25, 0.0], dtype=torch.float64, requires_grad=True
    )
    density = aufbau_density_matrix(hamiltonian, electrons)
    expected = torch.diag(torch.tensor([1.0, 0.25, 0.0], dtype=torch.float64))
    torch.testing.assert_close(density, expected, atol=2.0e-12, rtol=2.0e-12)
    derivative = torch.autograd.grad(density[1, 1], electrons)[0]
    torch.testing.assert_close(
        derivative, torch.tensor([1.0, 0.0], dtype=torch.float64)
    )
    count_jacobian = jacfwd(
        lambda count: aufbau_density_matrix(hamiltonian, count)[1, 1]
    )(electrons.detach())
    torch.testing.assert_close(count_jacobian, derivative, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "count, expected_slope",
    [(-0.5, 0.0), (0.0, 1.0), (0.25, 1.0), (3.0, 0.0), (4.0, 0.0)],
)
def test_aufbau_saturated_counts_match_authoritative_filling(
    count: float, expected_slope: float
) -> None:
    """Invariant projectors preserve Aufbau saturation and count response."""
    hamiltonian = torch.diag(
        torch.tensor([-0.3, 0.1, 0.7], dtype=torch.float64)
    )
    electrons = torch.tensor([count], dtype=torch.float64, requires_grad=True)
    density = aufbau_density_matrix(hamiltonian, electrons)
    occupation = filling.get_aufbau_occupation(
        torch.tensor(3), electrons.detach()
    )
    torch.testing.assert_close(
        density.diagonal(), occupation.squeeze(0), atol=2.0e-10, rtol=2.0e-10
    )
    expected_count = min(max(count, 0.0), 3.0)
    torch.testing.assert_close(
        density.trace(), torch.tensor(expected_count, dtype=torch.float64)
    )
    slope = torch.autograd.grad(density.trace(), electrons)[0]
    torch.testing.assert_close(
        slope, torch.tensor([expected_slope], dtype=torch.float64)
    )


@pytest.mark.parametrize("energy", [1.0, -4.0])
@pytest.mark.parametrize(
    "count, expected_density, expected_slope",
    [(-0.5, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.5, 1.0, 0.0)],
)
def test_aufbau_one_orbital_boundary_ranks(
    energy: float, count: float, expected_density: float, expected_slope: float
) -> None:
    """Empty and full one-orbital projectors avoid singular sign solves."""
    hamiltonian = torch.tensor([[energy]], dtype=torch.float64)
    electrons = torch.tensor([count], dtype=torch.float64, requires_grad=True)
    density = aufbau_density_matrix(hamiltonian, electrons)
    torch.testing.assert_close(
        density,
        torch.tensor([[expected_density]], dtype=torch.float64),
        atol=1.0e-14,
        rtol=0.0,
    )
    reverse = torch.autograd.grad(density.sum(), electrons)[0]
    torch.testing.assert_close(
        reverse,
        torch.tensor([expected_slope], dtype=torch.float64),
        atol=1.0e-14,
        rtol=0.0,
    )
    tangent = jvp(
        lambda x: aufbau_density_matrix(hamiltonian, x).sum(),
        (electrons.detach(),),
        (torch.ones_like(electrons),),
    )[1]
    forward = jacfwd(lambda x: aufbau_density_matrix(hamiltonian, x).sum())(
        electrons.detach()
    )
    torch.testing.assert_close(
        tangent, reverse.squeeze(0), atol=1.0e-14, rtol=0.0
    )
    torch.testing.assert_close(
        forward.squeeze(0), reverse.squeeze(0), atol=1.0e-14, rtol=0.0
    )


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("energy", [1.0, -4.0, 7.5])
def test_aufbau_degenerate_boundary_projectors_are_constant(
    energy: float, rank: int
) -> None:
    """Boundary projectors on cI are finite and independent of energy origin."""
    hamiltonian = torch.eye(3, dtype=torch.float64) * energy
    active = torch.ones(3, dtype=torch.bool)
    projector = _projector(hamiltonian, hamiltonian, active, torch.tensor(rank))
    expected = (
        torch.eye(3, dtype=torch.float64)
        if rank == 3
        else torch.zeros_like(hamiltonian)
    )
    torch.testing.assert_close(projector, expected, atol=1.0e-14, rtol=0.0)
    direction = torch.tensor(
        [[0.3, 0.1, 0.2], [0.1, -0.2, 0.0], [0.2, 0.0, 0.4]],
        dtype=torch.float64,
    )

    def evaluate(parameter: torch.Tensor) -> torch.Tensor:
        perturbed = hamiltonian + parameter * direction
        return _projector(perturbed, perturbed, active, torch.tensor(rank))

    parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(evaluate(parameter).sum(), parameter)[0]
    tangent = jvp(
        evaluate, (parameter.detach(),), (torch.ones_like(parameter),)
    )[1]
    jacobian = jacfwd(evaluate)(parameter.detach())
    torch.testing.assert_close(derivative, torch.zeros_like(derivative))
    torch.testing.assert_close(tangent, torch.zeros_like(tangent))
    torch.testing.assert_close(jacobian, torch.zeros_like(jacobian))


def test_fermi_float64_near_cutoff_is_finite_and_projector_limited() -> None:
    """The supported low-smearing cutoff remains stable over a 2 Eh gap."""
    energies = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    kt = torch.tensor(3.1e-7, dtype=torch.float64)

    def density(hamiltonian: torch.Tensor) -> torch.Tensor:
        channel_energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
        state = filling._get_fermi_occupation_core_state(
            electrons,
            channel_energies,
            kt,
            thr=1.0e-10,
            maxiter=300,
            diff_order=2,
        )
        assert state.result.status.converged.all()
        return fermi_density_matrix(
            hamiltonian, electrons, kt, state, diff_steps=2
        )

    coupling = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)

    def evaluate(parameter: torch.Tensor) -> torch.Tensor:
        return density(torch.diag(energies) + parameter * coupling)

    parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
    value = evaluate(parameter)
    reverse = torch.autograd.grad(value[0, 1], parameter)[0]
    _, tangent_density = jvp(
        evaluate, (parameter.detach(),), (torch.ones_like(parameter),)
    )
    expected = torch.tensor(-0.5, dtype=torch.float64)
    for item in (value, value.trace(), reverse, tangent_density):
        assert torch.isfinite(item).all()
    torch.testing.assert_close(
        value.trace(), torch.tensor(1.0, dtype=torch.float64)
    )
    torch.testing.assert_close(reverse, expected, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        tangent_density[0, 1], expected, atol=2.0e-6, rtol=2.0e-6
    )


@pytest.mark.parametrize(
    "spectrum, counts, temperature, threshold",
    [
        (
            [-1.0, -0.5, -0.5, 0.4, 0.5],
            [2.5, 1.5],
            3.1e-7,
            1.0e-4,
        ),
        (
            [
                -4.4926987,
                -4.4926891,
                -4.4926815,
                -4.4926419,
                -4.4926395,
                -2.30302,
            ],
            [5.0, 3.0],
            300.0 * KELVIN2AU,
            1.0e-4,
        ),
    ],
    ids=("tiny-temperature", "near-degenerate-300k"),
)
def test_float32_fermi_density_preserves_split_root_precision(
    spectrum: list[float],
    counts: list[float],
    temperature: float,
    threshold: float,
) -> None:
    """Integrated density preserves float32 Fermi roots at precision limits."""
    energies = torch.tensor(spectrum, dtype=torch.float32)
    hamiltonian = torch.diag(energies)
    electrons = torch.tensor(counts, dtype=torch.float32)
    kt = torch.tensor(temperature, dtype=torch.float32)
    channel_energies = energies.expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        channel_energies,
        kt,
        thr=threshold,
        maxiter=300,
        diff_order=1,
    )
    assert state.result.status.converged.all()
    assert state.result.status.electron_count_ok.all()
    collapsed_root = (
        state.chemical_potential_ref + state.chemical_potential_offset
    )
    split_root = state.chemical_potential_ref.to(
        torch.float64
    ) + state.chemical_potential_offset.to(torch.float64)
    assert not torch.equal(collapsed_root.to(torch.float64), split_root)
    density = fermi_density_matrix(
        hamiltonian, electrons, kt, state, diff_steps=1
    )
    torch.testing.assert_close(
        density.trace(),
        state.result.occupation.sum(),
        atol=threshold,
        rtol=0.0,
    )
    coefficients = torch.eye(energies.numel(), dtype=torch.float32)
    legacy = get_density(coefficients, state.result.occupation.sum(-2))
    torch.testing.assert_close(density, legacy, atol=5.0e-4, rtol=5.0e-5)


@pytest.mark.parametrize(
    "count, expected_trace, too_many",
    [(1.0, 1.0, False), (0.0, 0.0, False), (2.0, 2.0, False), (3.0, 0.0, True)],
    ids=("active", "empty", "full", "too-many"),
)
def test_fermi_density_uses_authoritative_channel_status(
    count: float, expected_trace: float, too_many: bool
) -> None:
    """Fermi density matches the shared worker on normal and invalid lanes."""
    energies = torch.tensor([-0.1, 0.2], dtype=torch.float64)
    hamiltonian = torch.diag(energies)
    electrons = torch.tensor([count], dtype=torch.float64)
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        energies.unsqueeze(0),
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=2,
    )
    assert bool(state.too_many[0]) is too_many
    density = fermi_density_matrix(hamiltonian, electrons, kt, state)
    torch.testing.assert_close(
        density.trace(), torch.tensor(expected_trace, dtype=torch.float64)
    )
    torch.testing.assert_close(
        density.trace(),
        state.result.occupation.sum(),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_aufbau_small_gap_has_analytic_and_energy_shift_invariant_response() -> (
    None
):
    """Aufbau projector derivatives approach the exact inverse-gap response."""
    count = torch.tensor([1.0], dtype=torch.float64)
    coupling = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    for gap in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        base = torch.tensor(
            [[-gap / 2.0, 0.0], [0.0, gap / 2.0]], dtype=torch.float64
        )
        responses = []
        for shift in (0.0, -5.0, 10.0):
            offset = torch.eye(2, dtype=torch.float64) * shift

            def projected(parameter: torch.Tensor) -> torch.Tensor:
                return aufbau_density_matrix(
                    base + offset + parameter * coupling, count
                )

            parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
            density = projected(parameter)
            reverse = torch.autograd.grad(density[0, 1], parameter)[0]
            _, tangent = jvp(
                projected,
                (parameter.detach(),),
                (torch.ones_like(parameter),),
            )
            forward = jacfwd(projected)(parameter.detach())[0, 1]
            expected = torch.tensor(-1.0 / gap, dtype=torch.float64)
            torch.testing.assert_close(reverse, expected, atol=2e-8, rtol=2e-9)
            torch.testing.assert_close(
                tangent[0, 1], expected, atol=2e-8, rtol=2e-9
            )
            torch.testing.assert_close(forward, expected, atol=2e-8, rtol=2e-9)
            responses.append((density.detach(), reverse.detach()))
        for density, derivative in responses[1:]:
            torch.testing.assert_close(
                density, responses[0][0], atol=2e-10, rtol=2e-10
            )
            torch.testing.assert_close(
                derivative, responses[0][1], atol=2e-8, rtol=2e-9
            )


def test_aufbau_internal_degeneracy_is_gauge_invariant() -> None:
    """An equally occupied exact degenerate block has a finite invariant response."""
    hamiltonian = torch.diag(
        torch.tensor([-0.2, -0.2, 0.3], dtype=torch.float64)
    )
    direction = torch.tensor(
        [[0.2, 0.3, 0.1], [0.3, -0.1, -0.2], [0.1, -0.2, 0.4]],
        dtype=torch.float64,
    )
    electrons = torch.tensor([2.0, 2.0], dtype=torch.float64)
    density, tangent = jvp(
        lambda matrix: aufbau_density_matrix(matrix, electrons),
        (hamiltonian,),
        (direction,),
    )
    torch.testing.assert_close(
        density,
        torch.diag(torch.tensor([2.0, 2.0, 0.0], dtype=torch.float64)),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    assert torch.isfinite(tangent).all()
    torch.testing.assert_close(tangent, tangent.mT, atol=1.0e-12, rtol=1.0e-12)

    parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
    probe = torch.tensor(
        [[0.3, 0.1, -0.2], [0.1, -0.1, 0.25], [-0.2, 0.25, 0.4]],
        dtype=torch.float64,
    )

    def weighted_density(value: torch.Tensor) -> torch.Tensor:
        density_value = aufbau_density_matrix(
            hamiltonian + value * direction, electrons
        )
        return torch.sum(probe * density_value)

    value = weighted_density(parameter)
    first = torch.autograd.grad(value, parameter, create_graph=True)[0]
    second = torch.autograd.grad(first, parameter, create_graph=True)[0]
    third = torch.autograd.grad(second, parameter)[0]
    forward = jacfwd(weighted_density)(parameter.detach())
    assert torch.isfinite(torch.stack((value, first, second, third))).all()
    torch.testing.assert_close(first, forward, atol=2.0e-8, rtol=2.0e-8)


def test_aufbau_degenerate_block_rotation_preserves_nested_responses() -> None:
    """Orthogonal basis rotations leave invariant density responses unchanged."""
    h0 = torch.diag(torch.tensor([-0.2, -0.2, 0.3], dtype=torch.float64))
    direction = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.2, 0.05, 0.1], [0.3, 0.1, -0.1]],
        dtype=torch.float64,
    )
    weight = torch.tensor(
        [[0.2, 0.4, -0.1], [0.4, -0.3, 0.25], [-0.1, 0.25, 0.5]],
        dtype=torch.float64,
    )
    electrons = torch.tensor([2.0, 2.0], dtype=torch.float64)

    def rotated_responses(rotation: torch.Tensor) -> torch.Tensor:
        h = rotation @ h0 @ rotation.mT
        d = rotation @ direction @ rotation.mT
        w = rotation @ weight @ rotation.mT

        def scalar(parameter: torch.Tensor) -> torch.Tensor:
            matrix = h + parameter * d
            density = aufbau_density_matrix(matrix, electrons)
            return torch.sum(w * density) + torch.sum(matrix * density.mT)

        parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
        value = scalar(parameter)
        first = torch.autograd.grad(value, parameter, create_graph=True)[0]
        second = torch.autograd.grad(first, parameter, create_graph=True)[0]
        third = torch.autograd.grad(second, parameter)[0]
        _, tangent = jvp(
            scalar,
            (parameter.detach(),),
            (torch.ones_like(parameter),),
        )
        forward = jacfwd(scalar)(parameter.detach())
        torch.testing.assert_close(first, tangent, atol=2.0e-8, rtol=2.0e-8)
        torch.testing.assert_close(first, forward, atol=2.0e-8, rtol=2.0e-8)
        return torch.stack(
            (value.detach(), first.detach(), second.detach(), third.detach())
        )

    reference = rotated_responses(torch.eye(3, dtype=torch.float64))
    for angle in (0.37, 1.13, 2.41):
        cosine = torch.cos(torch.tensor(angle, dtype=torch.float64))
        sine = torch.sin(torch.tensor(angle, dtype=torch.float64))
        rotation = torch.eye(3, dtype=torch.float64)
        rotation[:2, :2] = torch.stack(
            (torch.stack((cosine, -sine)), torch.stack((sine, cosine)))
        )
        torch.testing.assert_close(
            rotated_responses(rotation), reference, atol=2.0e-8, rtol=2.0e-8
        )


def test_zero_temperature_occupied_virtual_crossing_is_rejected() -> None:
    """An exact unequal-occupation boundary has no fabricated projector."""
    hamiltonian = torch.zeros((2, 2), dtype=torch.float64)
    electrons = torch.tensor([0.5, 0.0], dtype=torch.float64)
    with pytest.raises(torch.linalg.LinAlgError):
        aufbau_density_matrix(hamiltonian, electrons)


def test_fermi_exact_degeneracy_has_shared_reverse_forward_response() -> None:
    """The 300 K fixed-count density is smooth at a degeneracy at its chemical potential."""
    hamiltonian = torch.diag(
        torch.tensor([0.0, 0.0, 0.08], dtype=torch.float64)
    )
    direction = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
    weight = torch.tensor(
        [[0.3, 0.2, 0.1], [0.2, -0.1, 0.4], [0.1, 0.4, 0.5]],
        dtype=torch.float64,
    )

    def observable(matrix: torch.Tensor) -> torch.Tensor:
        return torch.sum(weight * _fermi_density(matrix))

    primal, tangent = jvp(observable, (hamiltonian,), (direction,))
    reverse = torch.autograd.grad(
        observable(hamiltonian.requires_grad_()), hamiltonian
    )[0]
    reverse_direction = torch.sum(reverse * direction)
    forward = jacfwd(observable)(hamiltonian)
    torch.testing.assert_close(
        tangent, reverse_direction, atol=2.0e-8, rtol=2.0e-8
    )
    torch.testing.assert_close(forward, reverse, atol=2.0e-8, rtol=2.0e-8)
    assert torch.isfinite(tangent)
    steps = (1.0e-5, 5.0e-6, 2.0e-6, 1.0e-6)
    finite_differences = []
    for step in steps:
        delta = hamiltonian.new_tensor(step)
        finite_differences.append(
            (
                observable(hamiltonian + delta * direction)
                - observable(hamiltonian - delta * direction)
            )
            / (2.0 * delta)
        )
    plateau = torch.stack(finite_differences[-3:])
    assert plateau.max() - plateau.min() < 3.0e-4
    torch.testing.assert_close(
        tangent, plateau.mean(), atol=2.0e-5, rtol=2.0e-5
    )


def test_fermi_density_forward_matches_eigensystem_density_at_300k() -> None:
    """The invariant matrix preserves ordinary well-conditioned density values."""
    hamiltonian = torch.tensor(
        [[-0.03, 0.002, 0.0], [0.002, 0.02, 0.001], [0.0, 0.001, 0.11]],
        dtype=torch.float64,
    )
    overlap = torch.tensor(
        [[1.0, 0.03, 0.0], [0.03, 1.1, 0.01], [0.0, 0.01, 0.9]],
        dtype=torch.float64,
    )
    l_inv = storch.inv_cholesky_factor(overlap, is_posdef=True)
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    kt = hamiltonian.new_tensor(9.5e-4)
    energies, coefficients = storch.eighb(
        a=hamiltonian,
        b=None,
        l_inv=l_inv,
        is_posdef=True,
        broadening_method=None,
    )
    state = filling._get_fermi_occupation_core_state(
        electrons,
        energies.expand(2, -1),
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
    )
    legacy_forward_density = get_density(
        coefficients, state.result.occupation.sum(-2)
    )
    invariant_density = fermi_density_matrix(
        hamiltonian,
        electrons,
        kt,
        state,
        l_inv=l_inv,
        diff_steps=filling._diff_steps(3),
    )
    torch.testing.assert_close(
        invariant_density,
        legacy_forward_density,
        atol=2.0e-10,
        rtol=2.0e-10,
    )


@pytest.mark.parametrize("kt", (9.5e-4, 2.0e-4, 2.0e-2))
def test_fermi_small_gap_density_is_continuous_in_temperature(
    kt: float,
) -> None:
    """Two-level matrix Fermi values approach the analytic degenerate limit."""
    thermal_energy = torch.tensor(kt, dtype=torch.float64)
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    for gap in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        hamiltonian = torch.diag(
            torch.tensor([-gap / 2.0, gap / 2.0], dtype=torch.float64)
        )
        energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
        state = filling._get_fermi_occupation_core_state(
            electrons,
            energies,
            thermal_energy,
            thr=1.0e-14,
            maxiter=300,
            diff_order=3,
        )
        density = fermi_density_matrix(
            hamiltonian,
            electrons,
            thermal_energy,
            state,
            diff_steps=filling._diff_steps(3),
        )
        expected_lower = torch.sigmoid(
            torch.tensor(gap / (2.0 * kt), dtype=torch.float64)
        )
        torch.testing.assert_close(
            density.diagonal(),
            torch.stack((expected_lower, 1.0 - expected_lower)),
            atol=2.0e-10,
            rtol=2.0e-10,
        )


def test_fermi_matrix_handles_large_positive_core_spectrum() -> None:
    """Large unoccupied core levels do not break the SPD matrix solve."""
    generator = torch.Generator().manual_seed(61)
    n_orbitals = 8
    rotation, _ = torch.linalg.qr(
        torch.randn(
            n_orbitals, n_orbitals, dtype=torch.float64, generator=generator
        )
    )
    energies = torch.cat(
        (
            torch.linspace(-1.24, -0.01, 3, dtype=torch.float64),
            torch.tensor([-5.0e-4, 5.0e-4], dtype=torch.float64),
            torch.linspace(0.01, 70.1, 3, dtype=torch.float64),
        )
    )
    hamiltonian = rotation @ torch.diag(energies) @ rotation.mT
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    active = torch.ones(n_orbitals, dtype=torch.bool)

    density = _matrix_fermi(
        hamiltonian,
        torch.tensor(0.0, dtype=torch.float64),
        kt,
        active,
    )
    expected = (
        rotation @ torch.diag(torch.sigmoid(-energies / kt)) @ rotation.mT
    )

    assert torch.isfinite(density).all()
    torch.testing.assert_close(density, expected, atol=5.0e-11, rtol=2.0e-10)

    direction_eigen = torch.linspace(-0.2, 0.2, n_orbitals, dtype=torch.float64)
    weight_eigen = torch.linspace(0.2, 0.9, n_orbitals, dtype=torch.float64)
    direction = rotation @ torch.diag(direction_eigen) @ rotation.mT
    weight = rotation @ torch.diag(weight_eigen) @ rotation.mT

    def response(parameter: torch.Tensor) -> torch.Tensor:
        matrix = hamiltonian + parameter * direction
        result = _matrix_fermi(
            matrix,
            torch.tensor(0.0, dtype=torch.float64),
            kt,
            active,
        )
        return torch.sum(weight * result)

    parameter = torch.zeros((), dtype=torch.float64)
    reverse = torch.func.grad(response)(parameter)
    forward = jacfwd(response)(parameter)
    reverse_jacobian = jacrev(response)(parameter)
    _, tangent = jvp(response, (parameter,), (torch.ones_like(parameter),))
    occupations = torch.sigmoid(-energies / kt)
    sigmoid_product = occupations * (1.0 - occupations)
    expected_derivative = torch.sum(
        weight_eigen * (-sigmoid_product / kt) * direction_eigen
    )
    assert torch.abs(expected_derivative) > 0.0
    torch.testing.assert_close(
        reverse, expected_derivative, atol=2.0e-10, rtol=2.0e-10
    )
    torch.testing.assert_close(
        forward, expected_derivative, atol=2.0e-10, rtol=2.0e-10
    )
    torch.testing.assert_close(
        reverse_jacobian, expected_derivative, atol=2.0e-10, rtol=2.0e-10
    )
    torch.testing.assert_close(
        tangent, expected_derivative, atol=2.0e-10, rtol=2.0e-10
    )

    index = torch.arange(n_orbitals, dtype=torch.float64)
    first_index = index.unsqueeze(-1)
    second_index = index.unsqueeze(-2)
    direction_basis = torch.sin((first_index + 1.0) * (second_index + 2.0))
    direction_basis = 0.5 * (direction_basis + direction_basis.mT)
    weight_basis = torch.cos((first_index + 2.0) * (second_index + 1.0))
    weight_basis = 0.5 * (weight_basis + weight_basis.mT)
    direction_basis.fill_diagonal_(0.0)
    direction_noncommuting = rotation @ direction_basis @ rotation.mT
    weight_noncommuting = rotation @ weight_basis @ rotation.mT

    def noncommuting_response(parameter: torch.Tensor) -> torch.Tensor:
        matrix = hamiltonian + parameter * direction_noncommuting
        result = _matrix_fermi(
            matrix,
            torch.tensor(0.0, dtype=torch.float64),
            kt,
            active,
        )
        return torch.sum(weight_noncommuting * result)

    divided_difference = (occupations.unsqueeze(-1) - occupations) / (
        energies.unsqueeze(-1) - energies
    )
    divided_difference.diagonal().copy_(-occupations * (1.0 - occupations) / kt)
    expected_noncommuting = torch.sum(
        weight_basis * divided_difference * direction_basis
    )
    _, noncommuting_tangent = jvp(
        noncommuting_response,
        (parameter,),
        (torch.ones_like(parameter),),
    )
    torch.testing.assert_close(
        noncommuting_tangent,
        expected_noncommuting,
        atol=2.0e-10,
        rtol=2.0e-10,
    )

    first = torch.func.grad(response)
    second = torch.func.grad(first)
    third = torch.func.grad(second)
    expected_second = torch.sum(
        weight_eigen
        * sigmoid_product
        * (1.0 - 2.0 * occupations)
        / kt.square()
        * direction_eigen.square()
    )
    expected_third = torch.sum(
        -weight_eigen
        * sigmoid_product
        * (1.0 - 6.0 * occupations + 6.0 * occupations.square())
        / kt.pow(3)
        * direction_eigen.pow(3)
    )
    actual_second = second(parameter)
    actual_third = third(parameter)
    torch.testing.assert_close(
        actual_second, expected_second, atol=2.0e-7, rtol=2.0e-8
    )
    torch.testing.assert_close(
        actual_third, expected_third, atol=2.0e-4, rtol=2.0e-8
    )


@pytest.mark.parametrize(
    ("maximum_energy", "temperature"),
    (
        (2.0, 9.5e-4),
        (12.0, 9.5e-4),
        (70.1, 9.5e-4),
        (70.1, 2.0e-4),
        (70.1, 3.1e-7),
    ),
)
def test_fermi_matrix_scaling_matches_spectral_oracle(
    maximum_energy: float, temperature: float
) -> None:
    """Adaptive scaling preserves Fermi values across spectral widths."""
    generator = torch.Generator().manual_seed(73)
    n_orbitals = 12
    rotation, _ = torch.linalg.qr(
        torch.randn(
            n_orbitals, n_orbitals, dtype=torch.float64, generator=generator
        )
    )
    energies = torch.linspace(
        -0.8, maximum_energy, n_orbitals, dtype=torch.float64
    )
    hamiltonian = rotation @ torch.diag(energies) @ rotation.mT
    kt = torch.tensor(temperature, dtype=torch.float64)
    active = torch.ones(n_orbitals, dtype=torch.bool)
    density = _matrix_fermi(hamiltonian, torch.zeros_like(kt), kt, active)
    occupations = torch.sigmoid(-energies / kt)
    expected = rotation @ torch.diag(occupations) @ rotation.mT

    assert torch.isfinite(density).all()
    torch.testing.assert_close(density, density.mT, atol=2.0e-13, rtol=0.0)
    torch.testing.assert_close(density, expected, atol=5.0e-11, rtol=2.0e-10)
    torch.testing.assert_close(
        density.trace(), occupations.sum(), atol=5.0e-11, rtol=2.0e-10
    )


@pytest.mark.parametrize(("maximum_energy", "seed"), ((70.1, 61), (100.0, 97)))
def test_fermi_matrix_cutoff_envelope_is_rotation_independent(
    maximum_energy: float, seed: int
) -> None:
    """The cutoff envelope covers wider spectra across independent rotations."""
    generator = torch.Generator().manual_seed(seed)
    n_orbitals = 8
    rotation, _ = torch.linalg.qr(
        torch.randn(
            n_orbitals, n_orbitals, dtype=torch.float64, generator=generator
        )
    )
    energies = torch.cat(
        (
            torch.linspace(-1.24, -0.01, 3, dtype=torch.float64),
            torch.tensor([-5.0e-4, 5.0e-4], dtype=torch.float64),
            torch.linspace(0.01, maximum_energy, 3, dtype=torch.float64),
        )
    )
    hamiltonian = rotation @ torch.diag(energies) @ rotation.mT
    kt = torch.tensor(3.1e-7, dtype=torch.float64)
    active = torch.ones(n_orbitals, dtype=torch.bool)
    density = _matrix_fermi(hamiltonian, torch.zeros_like(kt), kt, active)
    expected = (
        rotation @ torch.diag(torch.sigmoid(-energies / kt)) @ rotation.mT
    )

    assert torch.isfinite(density).all()
    torch.testing.assert_close(density, density.mT, atol=2.0e-13, rtol=0.0)
    torch.testing.assert_close(density, expected, atol=5.0e-11, rtol=2.0e-10)
    torch.testing.assert_close(
        density.trace(), expected.trace(), atol=5.0e-11, rtol=2.0e-10
    )


def test_fermi_matrix_unsupported_and_nonfinite_bounds_fail_closed() -> None:
    """Unsupported and non-finite exponent bounds never return finite values."""
    active = torch.ones(2, dtype=torch.bool)
    kt = torch.tensor(3.1e-7, dtype=torch.float64)
    for matrix in (
        torch.diag(torch.tensor([0.0, 5000.0], dtype=torch.float64)),
        torch.tensor(
            [[0.0, float("inf")], [float("inf"), 0.0]], dtype=torch.float64
        ),
        torch.tensor(
            [[0.0, float("nan")], [float("nan"), 0.0]], dtype=torch.float64
        ),
    ):
        result = _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)
        assert torch.isnan(result).all()

    hamiltonian = torch.diag(torch.tensor([0.0, 5000.0], dtype=torch.float64))
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons, energies, kt, thr=1.0e-10, maxiter=300, diff_order=2
    )
    assert state.result.status.converged.all()
    density = fermi_density_matrix(hamiltonian, electrons, kt, state)
    free_energy = fermi_free_energy_matrix(hamiltonian, electrons, kt, state)
    free_energy_trace = fermi_free_energy_trace(
        hamiltonian,
        electrons,
        kt,
        state,
        l_inv=torch.eye(2, dtype=torch.float64),
    )
    assert torch.isnan(density).all()
    assert torch.isnan(free_energy).all()
    assert torch.isnan(free_energy_trace).all()


def test_fermi_matrix_ignores_inactive_spectral_padding() -> None:
    """Large inactive padding does not enter the active scaling bound."""
    matrix = torch.diag(torch.tensor([-0.2, 0.1, 1.0e10], dtype=torch.float64))
    active = torch.tensor([True, True, False])
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    actual = _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)
    expected = torch.diag(
        torch.tensor(
            [
                torch.sigmoid(torch.tensor(0.2 / 9.5e-4)),
                torch.sigmoid(torch.tensor(-0.1 / 9.5e-4)),
                0.0,
            ],
            dtype=torch.float64,
        )
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2.0e-13, rtol=2.0e-13)


@pytest.mark.parametrize(
    ("kt_value", "lower_power"),
    ((9.5e-4, 11), (3.1e-7, 26)),
)
def test_fermi_matrix_scale_transition_preserves_derivatives(
    kt_value: float, lower_power: int
) -> None:
    """Detached scale transitions preserve value and response smoothness."""
    kt = torch.tensor(kt_value, dtype=torch.float64)
    active = torch.ones(3, dtype=torch.bool)
    threshold_energy = 4.0 * (2.0**lower_power) * kt_value
    near_gap = 0.3 * kt_value
    direction = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    def response(spectator_energy: float) -> torch.Tensor:
        hamiltonian = torch.diag(
            torch.tensor(
                [-near_gap, near_gap, spectator_energy], dtype=torch.float64
            )
        )

        def evaluate(parameter: torch.Tensor) -> torch.Tensor:
            density = _matrix_fermi(
                hamiltonian + parameter * direction,
                torch.zeros_like(kt),
                kt,
                active,
            )
            return density[0, 1]

        parameter = torch.zeros((), dtype=torch.float64)
        value = evaluate(parameter)
        reverse = torch.func.grad(evaluate)(parameter)
        forward = jacfwd(evaluate)(parameter)
        _, tangent = jvp(evaluate, (parameter,), (torch.ones_like(parameter),))
        second = torch.func.grad(torch.func.grad(evaluate))(parameter)
        third = torch.func.grad(torch.func.grad(torch.func.grad(evaluate)))(
            parameter
        )
        torch.testing.assert_close(reverse, forward, atol=2.0e-8, rtol=2.0e-8)
        torch.testing.assert_close(reverse, tangent, atol=2.0e-8, rtol=2.0e-8)
        return torch.stack((value, reverse, forward, tangent, second, third))

    epsilon = threshold_energy * 1.0e-8
    below = response(threshold_energy - epsilon)
    above = response(threshold_energy + epsilon)
    assert torch.isfinite(below).all() and torch.isfinite(above).all()
    torch.testing.assert_close(below, above, atol=2.0e-8, rtol=2.0e-8)


def test_fermi_density_public_path_supports_cutoff_temperature() -> None:
    """Public Fermi density retains count and JVP at the smear cutoff."""
    kt = torch.tensor(3.1e-7, dtype=torch.float64)
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    energies = torch.tensor([-0.8, 0.8, 100.0], dtype=torch.float64)
    hamiltonian = torch.diag(energies)
    direction = torch.tensor(
        [[0.0, 0.2, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    def density(parameter: torch.Tensor) -> torch.Tensor:
        matrix = hamiltonian + parameter * direction
        orbital_energies = torch.linalg.eigvalsh(matrix).expand(2, -1)
        state = filling._get_fermi_occupation_core_state(
            electrons,
            orbital_energies,
            kt,
            thr=1.0e-10,
            maxiter=300,
            diff_order=2,
        )
        assert state.result.status.converged.all()
        return fermi_density_matrix(matrix, electrons, kt, state, diff_steps=2)

    parameter = torch.zeros((), dtype=torch.float64)
    value = density(parameter)
    _, tangent = jvp(density, (parameter,), (torch.ones_like(parameter),))
    assert torch.isfinite(value).all() and torch.isfinite(tangent).all()
    torch.testing.assert_close(
        value.trace(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=2.0e-10,
        rtol=0,
    )
    torch.testing.assert_close(
        value,
        torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)),
        atol=2.0e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(
        tangent[0, 1],
        torch.tensor(-0.125, dtype=torch.float64),
        atol=2.0e-6,
        rtol=2e-6,
    )


def test_fermi_matrix_scaling_is_lane_local_under_vmap() -> None:
    """Batched Fermi scaling selects independent lane schedules."""
    generator = torch.Generator().manual_seed(79)
    n_orbitals = 10
    rotation, _ = torch.linalg.qr(
        torch.randn(
            n_orbitals, n_orbitals, dtype=torch.float64, generator=generator
        )
    )
    energies = torch.stack(
        (
            torch.linspace(-0.8, 1.2, n_orbitals, dtype=torch.float64),
            torch.linspace(-0.8, 70.1, n_orbitals, dtype=torch.float64),
        )
    )
    hamiltonians = rotation @ torch.diag_embed(energies) @ rotation.mT
    temperatures = torch.full((2,), 9.5e-4, dtype=torch.float64)
    active = torch.ones(n_orbitals, dtype=torch.bool)

    def evaluate(
        matrix: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        return _matrix_fermi(
            matrix, torch.zeros_like(temperature), temperature, active
        )

    batched = torch.func.vmap(evaluate)(hamiltonians, temperatures)
    explicit = torch.stack(
        [
            evaluate(hamiltonians[index], temperatures[index])
            for index in range(2)
        ]
    )

    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, explicit, atol=1.0e-13, rtol=1.0e-13)


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan"), 1.01])
def test_fermi_matrix_direct_batch_isolates_unsupported_lane(
    bad_value: float,
) -> None:
    """An unsupported direct-batch lane is sanitized before matrix solves."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    supported = torch.diag(torch.tensor([-2.0e-4, 3.0e-4], dtype=torch.float64))
    unsupported = torch.diag(
        torch.tensor(
            [0.1, bad_value * (4.0 * float(2**30) * float(kt))],
            dtype=torch.float64,
        )
    )
    matrices = torch.stack((supported, unsupported))
    active = torch.ones((2, 2), dtype=torch.bool)
    shifts = torch.zeros(2, dtype=torch.float64)
    temperatures = torch.full((2,), float(kt), dtype=torch.float64)

    batched = _matrix_fermi(matrices, shifts, temperatures, active)
    expected = _matrix_fermi(
        supported,
        torch.zeros_like(kt),
        kt,
        torch.ones(2, dtype=torch.bool),
    )

    assert torch.isfinite(batched[0]).all()
    torch.testing.assert_close(batched[0], expected, atol=1.0e-13, rtol=1.0e-13)
    assert torch.isnan(batched[1]).all()

    direction = torch.diag(torch.tensor([1.0, -0.5], dtype=torch.float64))

    def batched_response(parameter: torch.Tensor) -> torch.Tensor:
        perturbed = matrices.clone()
        perturbed[0] = perturbed[0] + parameter * direction
        result = _matrix_fermi(perturbed, shifts, temperatures, active)
        return result[0, 0, 0]

    def standalone_response(parameter: torch.Tensor) -> torch.Tensor:
        return _matrix_fermi(
            supported + parameter * direction,
            torch.zeros_like(kt),
            kt,
            torch.ones(2, dtype=torch.bool),
        )[0, 0]

    parameter = torch.zeros((), dtype=torch.float64)
    batched_jvp = jvp(
        batched_response, (parameter,), (torch.ones_like(parameter),)
    )[1]
    standalone_jvp = jvp(
        standalone_response, (parameter,), (torch.ones_like(parameter),)
    )[1]
    torch.testing.assert_close(
        batched_jvp, standalone_jvp, atol=1.0e-12, rtol=1.0e-12
    )


def test_fermi_matrix_vmap_isolates_unsupported_lane_and_jvp() -> None:
    """A bad vmapped lane does not change its supported neighbor's JVP."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    active = torch.ones(2, dtype=torch.bool)
    supported = torch.diag(torch.tensor([-2.0e-4, 3.0e-4], dtype=torch.float64))
    over_limit = torch.diag(
        torch.tensor(
            [0.1, 1.01 * 4.0 * float(2**30) * float(kt)],
            dtype=torch.float64,
        )
    )
    matrices = torch.stack((supported, over_limit))
    direction = torch.diag(torch.tensor([1.0, -0.5], dtype=torch.float64))

    def response(matrix: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return _matrix_fermi(
            matrix + delta * direction,
            torch.zeros_like(kt),
            kt,
            active,
        )[0, 0]

    parameters = torch.zeros(2, dtype=torch.float64)
    values = torch.func.vmap(response)(matrices, parameters)

    def jvp_one(matrix: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return jvp(
            lambda value: response(matrix, value),
            (delta,),
            (torch.ones_like(delta),),
        )[1]

    vmapped_jvp = torch.func.vmap(jvp_one)(matrices, parameters)
    standalone_jvp = jvp_one(supported, parameters[0])

    assert torch.isfinite(values[0])
    assert torch.isnan(values[1])
    torch.testing.assert_close(
        vmapped_jvp[0], standalone_jvp, atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.isfinite(vmapped_jvp[0])


def test_fermi_matrix_quadrature_shaped_batch_matches_explicit_lanes() -> None:
    """A heterogeneous 32-lane Fermi batch matches individual evaluations."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    pattern = torch.tensor([0.5, 2.0, 12.0, 70.1], dtype=torch.float64)
    upper = pattern.repeat(8)
    matrices = torch.diag_embed(
        torch.stack((torch.full_like(upper, -0.2), upper), dim=-1)
    )
    shifts = torch.zeros(32, dtype=torch.float64)
    temperatures = torch.full((32,), float(kt), dtype=torch.float64)
    active = torch.ones((32, 2), dtype=torch.bool)

    batched = _matrix_fermi(matrices, shifts, temperatures, active)
    explicit = torch.stack(
        [
            _matrix_fermi(
                matrices[index],
                torch.zeros_like(kt),
                kt,
                torch.ones(2, dtype=torch.bool),
            )
            for index in range(32)
        ]
    )

    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, explicit, atol=1.0e-13, rtol=1.0e-13)


def test_fermi_matrix_scaling_thresholds_preserve_nested_derivatives() -> None:
    """Values and derivatives remain stable across every scale boundary."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    frontier = 0.4 * kt
    active = torch.ones(2, dtype=torch.bool)
    high_energies = []
    # The accepted 70.1-Eh spectrum at 300 K selects at most power 15.
    # The low-temperature wide-spectrum oracle follows the fixed power-22
    # schedule and is covered separately below.
    for power in range(12, 15):
        threshold = float(2 ** (power + 2)) * float(kt)
        delta = threshold * 1.0e-10
        high_energies.extend((threshold - delta, threshold, threshold + delta))
    high = torch.tensor(high_energies, dtype=torch.float64)
    parameters = torch.zeros_like(high)

    def response(parameter: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        energies = torch.stack((frontier + parameter, upper))
        matrix = torch.diag(energies)
        density = _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)
        return density[0, 0]

    first_reverse = torch.func.grad(response, argnums=0)
    second_reverse = torch.func.grad(first_reverse, argnums=0)
    third_reverse = torch.func.grad(second_reverse, argnums=0)
    first_forward = jacfwd(response, argnums=0)
    second_forward = jacfwd(first_forward, argnums=0)
    third_forward = jacfwd(second_forward, argnums=0)
    third_reverse_forward = jacrev(second_forward, argnums=0)

    actual_value = torch.func.vmap(response)(parameters, high)
    actual_first_reverse = torch.func.vmap(first_reverse)(parameters, high)
    actual_first_forward = torch.func.vmap(first_forward)(parameters, high)
    actual_second_reverse = torch.func.vmap(second_reverse)(parameters, high)
    actual_third_reverse = torch.func.vmap(third_reverse)(parameters, high)
    actual_third_forward = torch.func.vmap(third_forward)(parameters, high)
    actual_third_reverse_forward = torch.func.vmap(third_reverse_forward)(
        parameters, high
    )

    expected_value, expected_first, expected_second, expected_third = (
        _analytic_fermi_derivatives(frontier, kt, high)
    )

    torch.testing.assert_close(
        actual_value, expected_value, atol=5.0e-11, rtol=2.0e-10
    )
    torch.testing.assert_close(
        actual_first_reverse, expected_first, atol=2.0e-9, rtol=2.0e-10
    )
    torch.testing.assert_close(
        actual_first_forward, expected_first, atol=2.0e-9, rtol=2.0e-10
    )
    torch.testing.assert_close(
        actual_second_reverse, expected_second, atol=2.0e-7, rtol=2.0e-8
    )
    torch.testing.assert_close(
        actual_third_reverse, expected_third, atol=2.0e-4, rtol=2.0e-8
    )
    torch.testing.assert_close(
        actual_third_forward, expected_third, atol=2.0e-4, rtol=2.0e-8
    )
    torch.testing.assert_close(
        actual_third_reverse_forward,
        expected_third,
        atol=2.0e-4,
        rtol=2.0e-8,
    )


def _analytic_fermi_derivatives(
    frontier: torch.Tensor, kt: torch.Tensor, template: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return analytic Fermi occupation and 1st-3rd energy derivatives."""
    occupation = torch.sigmoid(-frontier / kt)
    product = occupation * (1.0 - occupation)
    first = -product / kt
    second = product * (1.0 - 2.0 * occupation) / kt.square()
    third = (
        -product
        * (1.0 - 6.0 * occupation + 6.0 * occupation.square())
        / kt.pow(3)
    )
    return (
        occupation.expand_as(template),
        first.expand_as(template),
        second.expand_as(template),
        third.expand_as(template),
    )


def test_rotated_fermi_matrix_p14_p15_boundary_derivatives() -> None:
    """A rotated symmetric spectrum retains third derivatives at p14/p15."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    frontier = 0.4 * kt
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=torch.float64)
    frontier_vector = rotation[:, 0]
    frontier_projector = torch.outer(frontier_vector, frontier_vector)

    # For this rotation and upper > frontier, the largest absolute row sum is
    # 1.12 * upper - 0.12 * frontier. The boundary selects p14 at equality.
    threshold = (2.0**16 * kt + 0.12 * frontier) / 1.12
    upper = torch.stack(
        (threshold * (1.0 - 1.0e-10), threshold * (1.0 + 1.0e-10))
    )
    parameter = torch.zeros_like(upper)
    hamiltonian = (
        rotation
        @ torch.diag_embed(torch.stack((frontier + parameter, upper), dim=-1))
        @ rotation.mT
    )
    bound = (hamiltonian / kt).abs().sum(-1).amax(-1)
    selected_power = torch.ceil(torch.log2(bound / 4.0)).to(torch.long)
    torch.testing.assert_close(selected_power, torch.tensor([14, 15]))

    def response(delta: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stack((frontier + delta, high))
        matrix = rotation @ torch.diag(spectrum) @ rotation.mT
        density = _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)
        return (density * frontier_projector).sum()

    active = torch.ones(2, dtype=torch.bool)
    first_reverse = torch.func.grad(response, argnums=0)
    second_reverse = torch.func.grad(first_reverse, argnums=0)
    third_reverse = torch.func.grad(second_reverse, argnums=0)
    third_forward = jacfwd(
        jacfwd(jacfwd(response, argnums=0), argnums=0), argnums=0
    )
    third_reverse_forward = jacrev(
        jacfwd(jacfwd(response, argnums=0), argnums=0), argnums=0
    )

    actual_value = torch.func.vmap(response)(parameter, upper)
    actual_first = torch.func.vmap(first_reverse)(parameter, upper)
    actual_second = torch.func.vmap(second_reverse)(parameter, upper)
    actual_third_reverse = torch.func.vmap(third_reverse)(parameter, upper)
    actual_third_forward = torch.func.vmap(third_forward)(parameter, upper)
    actual_third_reverse_forward = torch.func.vmap(third_reverse_forward)(
        parameter, upper
    )

    expected = _analytic_fermi_derivatives(frontier, kt, upper)
    checks = (
        ("occupation", actual_value, expected[0], 5.0e-11),
        ("first derivative", actual_first, expected[1], 2.0e-9),
        ("second derivative", actual_second, expected[2], 2.0e-7),
        ("reverse third derivative", actual_third_reverse, expected[3], 2.0e-4),
        ("forward third derivative", actual_third_forward, expected[3], 2.0e-4),
        (
            "reverse-over-forward third derivative",
            actual_third_reverse_forward,
            expected[3],
            2.0e-4,
        ),
    )
    for label, value, reference, atol in checks:
        torch.testing.assert_close(
            value,
            reference,
            atol=atol,
            rtol=2.0e-8,
            msg=f"{label} disagrees across the rotated p14/p15 boundary",
        )


@pytest.mark.parametrize(("lower_power", "upper_power"), ((29, 30), (30, 31)))
def test_fermi_matrix_high_scale_boundaries(
    lower_power: int, upper_power: int
) -> None:
    """High outer-scale transitions are lane-local and explicitly bounded."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    frontier = 0.4 * kt
    active = torch.ones(2, dtype=torch.bool)
    threshold_exponent = 4.0 * float(2**lower_power)
    factors = torch.tensor(
        [1.0 - 1.0e-9, 1.0, 1.0 + 1.0e-9], dtype=torch.float64
    )
    upper = threshold_exponent * factors * kt
    parameters = torch.zeros_like(upper)
    matrix_bounds = torch.maximum(
        (frontier / kt).abs().expand_as(upper), (upper / kt).abs()
    )
    required = torch.ceil(torch.log2(matrix_bounds / 4.0)).to(torch.long)
    assert required.tolist() == [lower_power, lower_power, upper_power]

    def response(parameter: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        matrix = torch.diag(torch.stack((frontier + parameter, high)))
        return _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)[0, 0]

    third_reverse = torch.func.grad(
        torch.func.grad(torch.func.grad(response, argnums=0), argnums=0),
        argnums=0,
    )
    third_forward = jacfwd(
        jacfwd(jacfwd(response, argnums=0), argnums=0), argnums=0
    )
    third_mixed = jacrev(
        jacfwd(jacfwd(response, argnums=0), argnums=0), argnums=0
    )
    values = torch.func.vmap(response)(parameters, upper)
    derivatives_reverse = torch.func.vmap(third_reverse)(parameters, upper)
    derivatives_forward = torch.func.vmap(third_forward)(parameters, upper)
    derivatives_mixed = torch.func.vmap(third_mixed)(parameters, upper)
    valid = required <= 30

    assert torch.isfinite(values[valid]).all()
    assert torch.isfinite(derivatives_reverse[valid]).all()
    assert torch.isfinite(derivatives_forward[valid]).all()
    assert torch.isfinite(derivatives_mixed[valid]).all()
    if upper_power == 31:
        assert torch.isnan(values[~valid]).all()

    expected_occ, _, _, expected_third = _analytic_fermi_derivatives(
        frontier, kt, values[valid]
    )
    torch.testing.assert_close(
        values[valid],
        expected_occ,
        atol=5.0e-11,
        rtol=2.0e-10,
    )
    for label, derivatives in (
        ("reverse", derivatives_reverse),
        ("forward", derivatives_forward),
        ("reverse-over-forward", derivatives_mixed),
    ):
        torch.testing.assert_close(
            derivatives[valid],
            expected_third,
            atol=2.0e-4,
            rtol=2.0e-8,
            msg=f"{label} third derivative disagrees at p{lower_power}/p{upper_power}",
        )


def test_fermi_matrix_masks_inactive_inputs_before_matrix_function() -> None:
    """Inactive noncontiguous orbital values cannot destabilize the kernel."""
    matrix = torch.diag(
        torch.tensor(
            [0.2, float("nan"), -0.3, float("inf")], dtype=torch.float64
        )
    )
    active = torch.tensor([True, False, True, False])
    kt = torch.tensor(0.1, dtype=torch.float64)

    density = _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)
    expected = torch.zeros_like(matrix)
    expected[0, 0] = torch.sigmoid(-matrix[0, 0] / kt)
    expected[2, 2] = torch.sigmoid(-matrix[2, 2] / kt)

    def active_trace(value: torch.Tensor) -> torch.Tensor:
        return _matrix_fermi(value, torch.zeros_like(kt), kt, active).trace()

    gradient = torch.func.grad(active_trace)(matrix)

    inactive_direction = torch.diag(
        torch.tensor([0.0, 1.0e8, 0.0, -1.0e8], dtype=torch.float64)
    )

    def inactive_response(parameter: torch.Tensor) -> torch.Tensor:
        perturbed = matrix + parameter * inactive_direction
        return active_trace(perturbed)

    first = torch.func.grad(inactive_response)
    second = torch.func.grad(first)
    third = torch.func.grad(second)
    parameter = torch.zeros((), dtype=torch.float64)

    assert torch.isfinite(density).all()
    torch.testing.assert_close(density, expected, atol=2.0e-12, rtol=2.0e-12)
    assert torch.count_nonzero(gradient[~active, :]) == 0
    assert torch.count_nonzero(gradient[:, ~active]) == 0
    assert first(parameter) == 0.0
    assert second(parameter) == 0.0
    assert third(parameter) == 0.0


def test_fermi_matrix_marks_out_of_range_spectrum_as_numerical_failure() -> (
    None
):
    """The bounded expm1 approximation does not silently accept huge inputs."""
    matrix = torch.diag(torch.tensor([0.1, 1.0e10], dtype=torch.float64))
    active = torch.ones(2, dtype=torch.bool)
    kt = torch.tensor(1.0e-3, dtype=torch.float64)

    density = _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)

    assert not torch.isfinite(density).all()


def test_matrix_expm1_sanitizes_over_limit_and_nonfinite_lanes() -> None:
    """The inner expm1 helper rejects unsafe lanes before polynomial work."""
    matrices = torch.stack(
        (
            torch.diag(torch.tensor([0.1, -0.2], dtype=torch.float64)),
            torch.tensor([[64.0, 64.0], [64.0, 64.0]], dtype=torch.float64),
            torch.tensor(
                [[float("nan"), 0.0], [0.0, 0.0]], dtype=torch.float64
            ),
        )
    )
    values = torch.func.vmap(_matrix_expm1)(matrices)

    assert torch.isfinite(values[0]).all()
    assert torch.isnan(values[1]).all()
    assert torch.isnan(values[2]).all()


def test_fermi_matrix_scaling_is_bounded_through_power_30() -> None:
    """High scaling powers retain the scalar Fermi response and derivatives."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    frontier = 0.4 * kt
    active = torch.ones(2, dtype=torch.bool)
    powers = torch.tensor([20, 25, 30], dtype=torch.long)
    signs = torch.where(
        powers.remainder(2) == 0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
    )
    upper_exponents = (
        signs
        * 4.0
        * torch.pow(torch.tensor(2.0, dtype=torch.float64), powers - 0.5)
    )
    upper = upper_exponents * kt
    parameters = torch.zeros_like(upper)

    induced_bounds = torch.maximum(
        (frontier / kt).abs().expand_as(upper_exponents),
        upper_exponents.abs(),
    )
    required = torch.ceil(torch.log2(induced_bounds / 4.0)).to(torch.long)
    assert torch.equal(required, powers)
    # For this construction exp_argument has norm in (4, 8), so the inner
    # expm1 helper uses five halvings. The outer Cayley reconstruction uses
    # one fewer doubling than its scale power.
    inner_power = torch.ceil(
        torch.log2((2.0 * induced_bounds / 2.0**powers) / 0.25)
    )
    assert torch.equal(inner_power.to(torch.long), torch.full_like(powers, 5))
    assert torch.equal(powers - 1, torch.tensor([19, 24, 29], dtype=torch.long))

    def response(parameter: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        matrix = torch.diag(torch.stack((frontier + parameter, high)))
        density = _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)
        return density[0, 0]

    first = torch.func.grad(response, argnums=0)
    second = torch.func.grad(first, argnums=0)
    third_reverse = torch.func.grad(second, argnums=0)
    third_forward = jacfwd(
        jacfwd(jacfwd(response, argnums=0), argnums=0), argnums=0
    )
    third_mixed = jacrev(
        jacfwd(jacfwd(response, argnums=0), argnums=0), argnums=0
    )

    def full_density(high: torch.Tensor) -> torch.Tensor:
        matrix = torch.diag(torch.stack((frontier, high)))
        return _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)

    value = torch.func.vmap(response)(parameters, upper)
    density = torch.func.vmap(full_density)(upper)
    derivative1 = torch.func.vmap(first)(parameters, upper)
    derivative2 = torch.func.vmap(second)(parameters, upper)
    derivative3_reverse = torch.func.vmap(third_reverse)(parameters, upper)
    derivative3_forward = torch.func.vmap(third_forward)(parameters, upper)
    derivative3_mixed = torch.func.vmap(third_mixed)(parameters, upper)

    occupation = torch.sigmoid(-frontier / kt)
    product = occupation * (1.0 - occupation)
    expected = (
        occupation,
        -product / kt,
        product * (1.0 - 2.0 * occupation) / kt.square(),
        -product
        * (1.0 - 6.0 * occupation + 6.0 * occupation.square())
        / kt.pow(3),
    )
    torch.testing.assert_close(
        value, expected[0].expand_as(value), atol=5.0e-11, rtol=2.0e-10
    )
    expected_upper_occupation = torch.sigmoid(-upper / kt)
    torch.testing.assert_close(
        density[:, 1, 1],
        expected_upper_occupation,
        atol=5.0e-11,
        rtol=2.0e-10,
        msg="high positive/negative spectral occupations disagree",
    )
    torch.testing.assert_close(
        derivative1, expected[1].expand_as(value), atol=2.0e-9, rtol=2.0e-10
    )
    torch.testing.assert_close(
        derivative2, expected[2].expand_as(value), atol=2.0e-7, rtol=2.0e-8
    )
    for label, actual in (
        ("reverse third", derivative3_reverse),
        ("forward third", derivative3_forward),
        ("reverse-over-forward third", derivative3_mixed),
    ):
        torch.testing.assert_close(
            actual,
            expected[3].expand_as(value),
            atol=2.0e-4,
            rtol=2.0e-8,
            msg=f"{label} disagrees at scale powers 20 through 30",
        )


def test_fermi_matrix_scaling_limit_and_nonfinite_lanes_are_isolated() -> None:
    """Unsupported and nonfinite lanes return NaNs without harming neighbors."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    frontier = 0.4 * kt
    threshold = 4.0 * float(2**30) * kt
    upper = torch.tensor(
        [threshold * (1.0 - 1.0e-10), threshold * (1.0 + 1.0e-10)],
        dtype=torch.float64,
    )
    matrices = torch.diag_embed(
        torch.stack((frontier.expand_as(upper), upper), dim=-1)
    )
    matrices = torch.cat(
        (
            matrices,
            torch.diag_embed(
                torch.tensor([[frontier, float("nan")]], dtype=torch.float64)
            ),
        ),
        dim=0,
    )
    active = torch.ones(2, dtype=torch.bool)

    def evaluate(matrix: torch.Tensor) -> torch.Tensor:
        return _matrix_fermi(matrix, torch.zeros_like(kt), kt, active)

    values = torch.func.vmap(evaluate)(matrices)
    assert torch.isfinite(values[0]).all()
    assert torch.isnan(values[1]).all()
    assert torch.isnan(values[2]).all()
    torch.testing.assert_close(
        values[0, 0, 0], torch.sigmoid(-frontier / kt), atol=5e-11, rtol=2e-10
    )

    def lane_response(
        parameter: torch.Tensor, matrix: torch.Tensor
    ) -> torch.Tensor:
        direction = torch.diag(torch.tensor([1.0, 0.0], dtype=matrix.dtype))
        return evaluate(matrix + parameter * direction)[0, 0]

    lane_gradient = torch.func.vmap(torch.func.grad(lane_response, argnums=0))(
        torch.zeros(3, dtype=torch.float64), matrices
    )
    assert torch.isfinite(lane_gradient[0])


def test_fermi_matrix_range_check_uses_shifted_spectrum_safely() -> None:
    """Large cancelling energies remain valid; overflowing shifts fail early."""
    maximum = torch.finfo(torch.float64).max
    matrices = torch.tensor([[[1.0e300]], [[maximum]]], dtype=torch.float64)
    shifts = torch.tensor([1.0e300, -maximum], dtype=torch.float64)
    kt = torch.tensor(0.1, dtype=torch.float64)
    active = torch.ones(1, dtype=torch.bool)

    def evaluate(matrix: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return _matrix_fermi(matrix, shift, kt, active)

    values = torch.func.vmap(evaluate)(matrices, shifts)

    assert torch.isfinite(values[0]).all()
    torch.testing.assert_close(
        values[0, 0, 0], torch.tensor(0.5, dtype=torch.float64)
    )
    assert torch.isnan(values[1]).all()


def test_fermi_invalid_range_propagates_through_density_and_free_energy() -> (
    None
):
    """Downstream spectral APIs preserve the matrix-Fermi failure signal."""
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    hamiltonian = torch.diag(torch.tensor([0.0, 1.0e10], dtype=torch.float64))
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        energies,
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
    )
    # Suppress the explicit wide-gap closure for this low-level propagation
    # check so the intentionally invalid matrix-Fermi lane reaches entropy.
    state = replace(state, gap=torch.zeros_like(state.gap))
    inverse = torch.eye(2, dtype=torch.float64)

    density = fermi_density_matrix(
        hamiltonian, electrons, kt, state, l_inv=inverse
    )
    free_matrix = fermi_free_energy_matrix(
        hamiltonian, electrons, kt, state, l_inv=inverse
    )
    free_trace = fermi_free_energy_trace(
        hamiltonian, electrons, kt, state, l_inv=inverse
    )

    assert not torch.isfinite(density).all()
    assert not torch.isfinite(free_matrix).all()
    assert not torch.isfinite(free_trace).all()


def test_fermi_small_gap_jacfwd_matches_reverse_and_jvp() -> None:
    """jacfwd shares the physical density response at an xTB-scale gap."""
    h0 = torch.diag(torch.tensor([-5.0e-4, 5.0e-4], dtype=torch.float64))
    direction = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    weight = torch.tensor([[0.2, 0.4], [0.4, -0.3]], dtype=torch.float64)
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    kt = torch.tensor(9.5e-4, dtype=torch.float64)

    def value(parameter: torch.Tensor) -> torch.Tensor:
        hamiltonian = h0 + parameter * direction
        energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
        state = filling._get_fermi_occupation_core_state(
            electrons, energies, kt, thr=1.0e-14, maxiter=300, diff_order=3
        )
        density = fermi_density_matrix(
            hamiltonian,
            electrons,
            kt,
            state,
            diff_steps=filling._diff_steps(3),
        )
        return torch.sum(weight * density)

    parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
    reverse = torch.autograd.grad(value(parameter), parameter)[0]
    _, tangent = jvp(
        value, (parameter.detach(),), (torch.ones_like(parameter),)
    )
    forward = jacfwd(value)(parameter.detach())
    torch.testing.assert_close(reverse, tangent, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(reverse, forward, atol=2e-8, rtol=2e-8)


def test_fermi_exact_degeneracy_supports_nested_third_reverse() -> None:
    """The matrix Fermi response remains differentiable through third order."""
    hamiltonian = torch.diag(
        torch.tensor([0.0, 0.0, 0.08], dtype=torch.float64)
    )
    direction = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
    weight = torch.tensor(
        [[0.3, 0.2, 0.1], [0.2, -0.1, 0.4], [0.1, 0.4, 0.5]],
        dtype=torch.float64,
    )

    def response(parameter: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            weight * _fermi_density(hamiltonian + parameter * direction)
        )

    parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
    value = response(parameter)
    first = torch.autograd.grad(value, parameter, create_graph=True)[0]
    second = torch.autograd.grad(first, parameter, create_graph=True)[0]
    third = torch.autograd.grad(second, parameter)[0]
    assert torch.isfinite(torch.stack((value, first, second, third))).all()


def test_fermi_wide_gap_keeps_fractional_electron_response() -> None:
    """The invariant density retains dxtb's stable wide-gap count response."""
    hamiltonian = torch.diag(torch.tensor([-0.5, 0.5], dtype=torch.float64))
    electrons = torch.tensor(
        [1.0, 0.0], dtype=torch.float64, requires_grad=True
    )
    kt = hamiltonian.new_tensor(9.5e-4)
    energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        energies,
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
    )
    density = fermi_density_matrix(
        hamiltonian,
        electrons,
        kt,
        state,
        diff_steps=filling._diff_steps(3),
    )
    derivative = torch.autograd.grad(density[0, 0], electrons)[0]
    torch.testing.assert_close(
        derivative,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
        atol=2.0e-10,
        rtol=2.0e-10,
    )


def test_fermi_wide_gap_hamiltonian_response_and_trace_conservation() -> None:
    """A wide-gap density has projector response and fixed channel trace."""
    h0 = torch.diag(torch.tensor([-0.5, 0.5], dtype=torch.float64))
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    kt0 = torch.tensor(9.5e-4, dtype=torch.float64)
    off_diagonal = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)

    def density(h: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
        energies = torch.linalg.eigvalsh(h).expand(2, -1)
        state = filling._get_fermi_occupation_core_state(
            electrons, energies, kt, thr=1.0e-14, maxiter=300, diff_order=3
        )
        return fermi_density_matrix(
            h,
            electrons,
            kt,
            state,
            diff_steps=filling._diff_steps(3),
        )

    _, h_tangent = jvp(lambda h: density(h, kt0)[0, 1], (h0,), (off_diagonal,))
    step = h0.new_tensor(1.0e-5)
    h_fd = (
        density(h0 + step * off_diagonal, kt0)[0, 1]
        - density(h0 - step * off_diagonal, kt0)[0, 1]
    ) / (2.0 * step)
    torch.testing.assert_close(h_tangent, h_fd, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        h_tangent, h0.new_tensor(-1.0), atol=2.0e-6, rtol=2.0e-6
    )

    _, kt_tangent = jvp(
        lambda kt: density(h0, kt).trace(),
        (kt0,),
        (torch.ones_like(kt0),),
    )
    assert torch.isfinite(kt_tangent)
    torch.testing.assert_close(
        kt_tangent, torch.zeros_like(kt_tangent), atol=1.0e-12, rtol=0.0
    )


def test_fermi_temperature_response_matches_physical_fd() -> None:
    """Temperature response includes the fixed-count chemical-potential motion."""
    hamiltonian = torch.diag(torch.tensor([-0.015, 0.015], dtype=torch.float64))
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    weight = torch.tensor([[1.0, 0.3], [0.3, -0.4]], dtype=torch.float64)

    def observable(kt: torch.Tensor) -> torch.Tensor:
        energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
        state = filling._get_fermi_occupation_core_state(
            electrons,
            energies,
            kt,
            thr=1.0e-14,
            maxiter=300,
            diff_order=3,
        )
        density = fermi_density_matrix(
            hamiltonian,
            electrons,
            kt,
            state,
            diff_steps=filling._diff_steps(3),
        )
        return torch.sum(weight * density)

    kt0 = torch.tensor(2.0e-2, dtype=torch.float64)
    _, tangent = jvp(observable, (kt0,), (torch.ones_like(kt0),))
    steps = (2.0e-5, 1.0e-5, 5.0e-6, 2.0e-6)
    finite_differences = []
    for step in steps:
        delta = kt0.new_tensor(step)
        finite_differences.append(
            (observable(kt0 + delta) - observable(kt0 - delta)) / (2.0 * delta)
        )
    values = torch.stack(finite_differences)
    plateau = values[:3]
    assert values.abs().max() > 0.1
    assert plateau.max() - plateau.min() < 2.0e-5
    torch.testing.assert_close(
        tangent, plateau.mean(), atol=2.0e-5, rtol=2.0e-5
    )


def test_fermi_density_conserves_channel_electrons_and_overlap_response() -> (
    None
):
    """Generalized density responds to both Hamiltonian and metric inputs."""
    h0 = torch.diag(torch.tensor([-0.03, 0.02, 0.11], dtype=torch.float64))
    li0 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.05, 1.1, 0.0], [0.02, -0.03, 0.9]],
        dtype=torch.float64,
    )
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    kt = torch.tensor(9.5e-4, dtype=torch.float64)
    energies = torch.linalg.eigvalsh(li0 @ h0 @ li0.mT).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons, energies, kt, thr=1e-14, maxiter=300, diff_order=3
    )

    def density_trace(h: torch.Tensor, li: torch.Tensor) -> torch.Tensor:
        density = fermi_density_matrix(
            h,
            electrons,
            kt,
            state,
            l_inv=li,
            diff_steps=filling._diff_steps(3),
        )
        overlap = torch.linalg.inv(li.mT @ li)
        return torch.sum(overlap * density.mT)

    dh = torch.tensor(
        [[0.1, 0.02, 0.0], [0.02, -0.1, 0.01], [0.0, 0.01, 0.0]],
        dtype=torch.float64,
    )
    dli = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, -0.01, 0.0]],
        dtype=torch.float64,
    )
    _, tangent = jvp(density_trace, (h0, li0), (dh, dli))
    assert torch.isfinite(tangent).all()
    torch.testing.assert_close(
        tangent,
        torch.tensor(0.0, dtype=torch.float64),
        atol=1.0e-9,
        rtol=0.0,
    )


@pytest.mark.parametrize("perturb", ("hamiltonian", "overlap", "combined"))
def test_fermi_density_generalized_response_matches_physical_fd(
    perturb: str,
) -> None:
    """H and metric tangents agree with FD of the shared-root density."""
    h0 = torch.tensor(
        [[-0.08, 0.012, 0.0], [0.012, 0.01, 0.006], [0.0, 0.006, 0.09]],
        dtype=torch.float64,
    )
    li0 = torch.tensor(
        [[1.02, 0.0, 0.0], [0.04, 0.96, 0.0], [-0.02, 0.03, 1.04]],
        dtype=torch.float64,
    )
    dh = torch.tensor(
        [[0.2, -0.03, 0.01], [-0.03, -0.1, 0.02], [0.01, 0.02, 0.07]],
        dtype=torch.float64,
    )
    dli = torch.tensor(
        [[0.03, 0.0, 0.0], [-0.02, -0.01, 0.0], [0.01, 0.04, -0.03]],
        dtype=torch.float64,
    )
    if perturb == "hamiltonian":
        dli = torch.zeros_like(dli)
    elif perturb == "overlap":
        dh = torch.zeros_like(dh)

    electrons = torch.tensor([1.25, 0.0], dtype=torch.float64)
    kt = torch.tensor(2.0e-2, dtype=torch.float64)
    weight = torch.tensor(
        [[0.3, -0.2, 0.1], [-0.2, 0.4, 0.15], [0.1, 0.15, -0.1]],
        dtype=torch.float64,
    )

    def observable(h: torch.Tensor, li: torch.Tensor) -> torch.Tensor:
        energies = torch.linalg.eigvalsh(li @ h @ li.mT).expand(2, -1)
        state = filling._get_fermi_occupation_core_state(
            electrons, energies, kt, thr=1.0e-14, maxiter=300, diff_order=3
        )
        density = fermi_density_matrix(
            h,
            electrons,
            kt,
            state,
            l_inv=li,
            diff_steps=filling._diff_steps(3),
        )
        return torch.sum(weight * density)

    _, tangent = jvp(observable, (h0, li0), (dh, dli))
    parameter = h0.new_zeros((), requires_grad=True)

    def along_direction(value: torch.Tensor) -> torch.Tensor:
        return observable(h0 + value * dh, li0 + value * dli)

    forward = jacfwd(along_direction)(parameter.detach())
    torch.testing.assert_close(tangent, forward, atol=2.0e-8, rtol=2.0e-8)
    steps = (2.0e-4, 1.0e-4, 5.0e-5, 2.0e-5)
    finite_differences = []
    for step in steps:
        delta = h0.new_tensor(step)
        finite_differences.append(
            (
                observable(h0 + delta * dh, li0 + delta * dli)
                - observable(h0 - delta * dh, li0 - delta * dli)
            )
            / (2.0 * delta)
        )
    values = torch.stack(finite_differences)
    assert torch.isfinite(values).all()
    # The final three central differences form the independently recomputed
    # fixed-electron physical-density window for this smooth finite-T case.
    plateau = values[-3:]
    spread = plateau.max() - plateau.min()
    assert spread < 2.0e-5
    torch.testing.assert_close(
        tangent, plateau.mean(), atol=2.0e-5, rtol=2.0e-5
    )


def test_structural_mask_excludes_padded_density_and_input_derivatives() -> (
    None
):
    """Inactive orbital rows and columns have exactly zero density response."""
    hamiltonian = torch.diag(
        torch.tensor([-0.2, 0.1, 99.0], dtype=torch.float64)
    )
    mask = torch.tensor([True, True, False])
    electrons = torch.tensor([1.0, 0.0], dtype=torch.float64)
    density, tangent = jvp(
        lambda matrix: aufbau_density_matrix(matrix, electrons, mask=mask),
        (hamiltonian,),
        (torch.ones_like(hamiltonian),),
    )
    assert torch.equal(density[2], torch.zeros_like(density[2]))
    assert torch.equal(density[:, 2], torch.zeros_like(density[:, 2]))
    assert torch.equal(tangent[2], torch.zeros_like(tangent[2]))
    assert torch.equal(tangent[:, 2], torch.zeros_like(tangent[:, 2]))


def test_aufbau_vmap_preserves_lane_specific_orbital_masks() -> None:
    """Batched masks define independent fixed-shape spectral problems."""
    hamiltonians = torch.stack(
        (
            torch.diag(torch.tensor([-0.2, 0.1, 90.0], dtype=torch.float64)),
            torch.diag(torch.tensor([-0.1, 0.2, 0.3], dtype=torch.float64)),
        )
    )
    masks = torch.tensor([[True, True, False], [True, True, True]])
    electrons = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)

    batched = torch.func.vmap(
        lambda h, nel, mask: aufbau_density_matrix(h, nel, mask=mask)
    )(hamiltonians, electrons, masks)
    looped = torch.stack(
        [
            aufbau_density_matrix(hamiltonians[i], electrons[i], mask=masks[i])
            for i in range(hamiltonians.shape[0])
        ]
    )
    torch.testing.assert_close(batched, looped, atol=2.0e-10, rtol=2.0e-10)
    assert torch.equal(batched[0, 2], torch.zeros_like(batched[0, 2]))
    assert torch.equal(batched[0, :, 2], torch.zeros_like(batched[0, :, 2]))

    inverse_cholesky = torch.stack(
        (
            torch.eye(3, dtype=torch.float64),
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.2, 0.3, 1.0]],
                dtype=torch.float64,
            ),
        )
    )
    h_tangent = torch.ones_like(hamiltonians)
    li_tangent = torch.ones_like(inverse_cholesky)

    def masked_density(h: torch.Tensor, li: torch.Tensor, mask: torch.Tensor):
        return aufbau_density_matrix(h, electrons[0], l_inv=li, mask=mask)

    mapped_tangent = torch.func.vmap(
        lambda h, li, mask, dh, dli: jvp(
            lambda h_, li_: masked_density(h_, li_, mask),
            (h, li),
            (dh, dli),
        )
    )(hamiltonians, inverse_cholesky, masks, h_tangent, li_tangent)[1]
    for lane in range(2):
        expected = jvp(
            lambda h, li: masked_density(h, li, masks[lane]),
            (hamiltonians[lane], inverse_cholesky[lane]),
            (h_tangent[lane], li_tangent[lane]),
        )[1]
        torch.testing.assert_close(
            mapped_tangent[lane], expected, atol=2e-10, rtol=2e-10
        )
        inactive = ~masks[lane]
        assert torch.equal(
            mapped_tangent[lane][inactive],
            torch.zeros_like(mapped_tangent[lane][inactive]),
        )
        assert torch.equal(
            mapped_tangent[lane][:, inactive],
            torch.zeros_like(mapped_tangent[lane][:, inactive]),
        )

    changed_h = hamiltonians.clone()
    changed_h[0, 2, :] = -123.0
    changed_h[0, :, 2] = 456.0
    changed_li = inverse_cholesky.clone()
    changed_li[0, 2, :] = 789.0
    changed_li[0, :, 2] = -321.0
    changed = torch.func.vmap(
        lambda h, li, mask: aufbau_density_matrix(
            h, electrons[0], l_inv=li, mask=mask
        )
    )(changed_h, changed_li, masks)
    torch.testing.assert_close(changed[0], batched[0], atol=0.0, rtol=0.0)

    inactive_h = torch.zeros_like(hamiltonians[0])
    inactive_h[2, :] = 1.0
    inactive_h[:, 2] = 1.0
    inactive_li = torch.zeros_like(inverse_cholesky[0])
    inactive_li[2, :] = 1.0
    inactive_li[:, 2] = 1.0
    _, inactive_tangent = jvp(
        lambda h, li: aufbau_density_matrix(
            h, electrons[0], l_inv=li, mask=masks[0]
        ),
        (hamiltonians[0], inverse_cholesky[0]),
        (inactive_h, inactive_li),
    )
    assert torch.equal(inactive_tangent, torch.zeros_like(inactive_tangent))


@pytest.mark.parametrize(
    "case_name,evals,active",
    [
        (
            "well-separated",
            [-0.45, -0.18, 0.03, 0.27, 0.61],
            [True, True, True, True, True],
        ),
        (
            "exact-3fold",
            [-0.40, 0.02, 0.02, 0.02, 0.50],
            [True, True, True, True, True],
        ),
        (
            "near-1e-12",
            [-0.30, -0.01, -0.01 + 1.0e-12, 0.15, 0.45],
            [True, True, True, True, True],
        ),
        (
            "noncontiguous-padded",
            [-0.38, 12.0, -0.04, -0.04, 15.0, 0.29],
            [True, False, True, True, False, True],
        ),
    ],
)
@pytest.mark.parametrize("etemp_k", [50.0, 3000.0])
def test_detached_spectral_fermi_matches_static_across_degeneracy_and_temperature(
    case_name: str,
    evals: list[float],
    active: list[bool],
    etemp_k: float,
) -> None:
    """Detached spectral matrix-Fermi matches static matrix-Fermi across degeneracies and temperatures."""
    del case_name
    gen = torch.Generator().manual_seed(42)
    n = len(evals)
    active_mask = torch.tensor(active, dtype=torch.bool)
    n_act = int(active_mask.sum().item())
    q_act, _ = torch.linalg.qr(
        torch.randn((n_act, n_act), generator=gen, dtype=torch.float64)
    )
    u = torch.eye(n, dtype=torch.float64)
    act_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    u[act_idx.unsqueeze(-1), act_idx.unsqueeze(0)] = q_act
    diag_e = torch.diag(torch.tensor(evals, dtype=torch.float64))
    h_orth = u @ diag_e @ u.mT

    # Generalized overlap with nontrivial lower-triangular L^-1 on active indices.
    l_inv = torch.eye(n, dtype=torch.float64)
    l_inv[act_idx[1], act_idx[0]] = 0.08
    l_inv[act_idx[2], act_idx[0]] = -0.04
    l_inv[act_idx[2], act_idx[1]] = 0.06
    l_factor = torch.linalg.inv(l_inv)
    h_ao = l_factor @ h_orth @ l_factor.mT

    kt = torch.tensor(etemp_k * KELVIN2AU, dtype=torch.float64)
    electrons = torch.tensor([2.0, 1.0], dtype=torch.float64)
    _, spectral_base, prepared_active = _prepare_matrix(
        h_ao, l_inv, active_mask
    )
    act_evals = torch.tensor(
        [e for e, a in zip(evals, active) if a], dtype=torch.float64
    )
    padded_evals = torch.cat(
        (
            act_evals,
            torch.zeros(n - n_act, dtype=torch.float64),
        )
    ).expand(2, -1)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        padded_evals,
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
        mask=(torch.arange(n) < n_act).expand(2, -1),
    )

    with torch.no_grad():
        d_static = fermi_density_matrix(
            h_ao,
            electrons,
            kt,
            state,
            l_inv=l_inv,
            mask=active_mask,
            eager_primal="static",
        )
        d_spectral = fermi_density_matrix(
            h_ao,
            electrons,
            kt,
            state,
            l_inv=l_inv,
            mask=active_mask,
            eager_primal="spectral",
        )
        f_static = fermi_free_energy_matrix(
            h_ao,
            electrons,
            kt,
            state,
            l_inv=l_inv,
            mask=active_mask,
            eager_primal="static",
        )
        f_spectral = fermi_free_energy_matrix(
            h_ao,
            electrons,
            kt,
            state,
            l_inv=l_inv,
            mask=active_mask,
            eager_primal="spectral",
        )
        tr_static = fermi_free_energy_trace(
            h_ao,
            electrons,
            kt,
            state,
            l_inv=l_inv,
            mask=active_mask,
            eager_primal="static",
        )
        tr_spectral = fermi_free_energy_trace(
            h_ao,
            electrons,
            kt,
            state,
            l_inv=l_inv,
            mask=active_mask,
            eager_primal="spectral",
        )
    torch.testing.assert_close(d_spectral, d_static, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(f_spectral, f_static, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(
        tr_spectral, tr_static, atol=1.0e-11, rtol=1.0e-11
    )

    # Also check direct _matrix_fermi_spectral vs _matrix_fermi with basis rotation of degenerate subspace.
    shift = state.chemical_potential_ref
    with torch.no_grad():
        m_static = _matrix_fermi(
            spectral_base.expand(2, -1, -1), shift, kt, prepared_active
        )
        m_spectral = spectral_module._matrix_fermi_spectral(
            spectral_base.expand(2, -1, -1), shift, kt, prepared_active
        )
    torch.testing.assert_close(m_spectral, m_static, atol=1.0e-11, rtol=1.0e-11)


def test_detached_spectral_fermi_degenerate_subspace_rotation_invariance() -> (
    None
):
    """Random orthogonal rotations within a 3-fold degenerate subspace leave D invariant."""
    evals = torch.tensor([-0.4, 0.05, 0.05, 0.05, 0.6], dtype=torch.float64)
    active = torch.ones(5, dtype=torch.bool)
    kt = torch.tensor(300.0 * KELVIN2AU, dtype=torch.float64)
    shift = torch.tensor([0.05], dtype=torch.float64)

    gen = torch.Generator().manual_seed(7)
    q1, _ = torch.linalg.qr(
        torch.randn((5, 5), generator=gen, dtype=torch.float64)
    )
    rot_sub, _ = torch.linalg.qr(
        torch.randn((3, 3), generator=gen, dtype=torch.float64)
    )
    rot_full = torch.eye(5, dtype=torch.float64)
    rot_full[1:4, 1:4] = rot_sub
    q2 = q1 @ rot_full

    h1 = q1 @ torch.diag(evals) @ q1.mT
    h2 = q2 @ torch.diag(evals) @ q2.mT
    with torch.no_grad():
        d1 = spectral_module._matrix_fermi_spectral(
            h1.unsqueeze(0), shift, kt, active
        )
        d2 = spectral_module._matrix_fermi_spectral(
            h2.unsqueeze(0), shift, kt, active
        )
        d_ref = _matrix_fermi(h1.unsqueeze(0), shift, kt, active)
    torch.testing.assert_close(d1, d2, atol=1.0e-13, rtol=1.0e-13)
    torch.testing.assert_close(d1, d_ref, atol=1.0e-12, rtol=1.0e-12)


def test_detached_spectral_fermi_wide_spectra_saturation_s01b_and_nonfinite() -> (
    None
):
    """Cover wide spectra, saturated 0/1 occupations, S0.1b powers 30/31, and NaN/Inf inputs."""
    active = torch.tensor([True, True, True, False], dtype=torch.bool)
    kt = torch.tensor(1.0e-3, dtype=torch.float64)

    # 1. Wide spectrum + large positive/negative chemical potential shifts (saturated 1.0 and 0.0).
    wide_evals = torch.tensor([-25.0, 0.0, 25.0, 0.0], dtype=torch.float64)
    _, wide_spec, prep_act = _prepare_matrix(
        torch.diag(wide_evals), None, active
    )
    for shift_val in (-50.0, 0.0, 50.0):
        shift = torch.tensor(shift_val, dtype=torch.float64)
        with torch.no_grad():
            d_spec = spectral_module._matrix_fermi_spectral(
                wide_spec, shift, kt, prep_act
            )
            d_stat = _matrix_fermi(wide_spec, shift, kt, prep_act)
        torch.testing.assert_close(d_spec, d_stat, atol=1.0e-12, rtol=1.0e-12)
    # Saturated 0.0 at shift = -50.0 and saturated 1.0 on active block at shift = +50.0
    with torch.no_grad():
        d_zero = spectral_module._matrix_fermi_spectral(
            wide_spec, torch.tensor(-50.0, dtype=torch.float64), kt, prep_act
        )
        d_one = spectral_module._matrix_fermi_spectral(
            wide_spec, torch.tensor(50.0, dtype=torch.float64), kt, prep_act
        )
    torch.testing.assert_close(
        d_zero, torch.zeros_like(d_zero), atol=1.0e-14, rtol=0.0
    )
    expected_one = torch.diag(
        torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    )
    torch.testing.assert_close(d_one, expected_one, atol=1.0e-14, rtol=0.0)

    # 2. S0.1b regime boundaries: required_power == 30 (in range) vs required_power == 31 (out of range -> NaN).
    max_norm = float(spectral_module._FERMI_MAX_SCALED_NORM)
    for power in (30, 31):
        target_scaled_norm = max_norm * (2.0 ** (power - 0.5))
        e_max = float(kt.item()) * target_scaled_norm
        s01b_evals = torch.tensor(
            [-e_max, 0.0, 0.5 * e_max, 0.0], dtype=torch.float64
        )
        _, s01b_spec, s01b_act = _prepare_matrix(
            torch.diag(s01b_evals), None, active
        )
        act_pair = s01b_act.unsqueeze(-1) & s01b_act.unsqueeze(-2)
        exp_mat = torch.where(act_pair, s01b_spec / kt, 0.0)
        exp_bound = exp_mat.abs().sum(-1).amax(-1)
        req_pow = int(
            torch.ceil(torch.log2(exp_bound.clamp_min(1.0) / max_norm)).item()
        )
        assert req_pow == power
        with torch.no_grad():
            d_spec_p = spectral_module._matrix_fermi_spectral(
                s01b_spec,
                torch.tensor(0.0, dtype=torch.float64),
                kt,
                s01b_act,
            )
            d_stat_p = _matrix_fermi(
                s01b_spec,
                torch.tensor(0.0, dtype=torch.float64),
                kt,
                s01b_act,
            )
        if power == 30:
            assert bool(torch.isfinite(d_spec_p).all())
            torch.testing.assert_close(
                d_spec_p, d_stat_p, atol=1.0e-12, rtol=1.0e-12
            )
        else:
            assert bool(torch.isnan(d_spec_p).all())
            assert bool(torch.isnan(d_stat_p).all())

    # 3. Invalid active inputs (NaN / Inf) return NaN matching _matrix_fermi.
    for bad_val in (float("nan"), float("inf")):
        bad_h = torch.diag(
            torch.tensor([-0.2, 0.1, 0.3, 0.0], dtype=torch.float64)
        )
        bad_h[0, 0] = bad_val
        _, bad_spec, bad_act = _prepare_matrix(bad_h, None, active)
        with torch.no_grad():
            d_bad = spectral_module._matrix_fermi_spectral(
                bad_spec,
                torch.tensor(0.0, dtype=torch.float64),
                kt,
                bad_act,
            )
            d_bad_stat = _matrix_fermi(
                bad_spec,
                torch.tensor(0.0, dtype=torch.float64),
                kt,
                bad_act,
            )
        assert bool(torch.isnan(d_bad).all())
        assert bool(torch.isnan(d_bad_stat).all())

    # 4. Guard rejects grad-enabled, functorch, forward-AD on l_inv, or compile calls to spectral primal mode, and supports float32.
    h_grad = torch.diag(
        torch.tensor([-0.2, 0.1, 0.3, 0.0], dtype=torch.float64)
    ).requires_grad_(True)
    with pytest.raises(RuntimeError, match="detached eager"):
        spectral_module._matrix_fermi_spectral(
            h_grad, torch.tensor(0.0, dtype=torch.float64), kt, active
        )

    h_det = h_grad.detach()
    shift_det = torch.tensor(0.0, dtype=torch.float64)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="torch.func"):
            torch.func.jvp(
                lambda x: spectral_module._matrix_fermi_spectral(
                    x, shift_det, kt, active
                ),
                (h_det,),
                (torch.ones_like(h_det),),
            )
        with pytest.raises(RuntimeError, match="torch.func"):
            torch.func.vmap(
                lambda x: spectral_module._matrix_fermi_spectral(
                    x, shift_det, kt, active
                )
            )(h_det.unsqueeze(0))
        from dxtb._src.wavefunction.filling import (
            _get_fermi_occupation_core_state,
        )

        l_inv_eye = torch.eye(4, dtype=torch.float64)
        st_det = _get_fermi_occupation_core_state(
            torch.tensor([1.5], dtype=torch.float64),
            torch.tensor([[-0.2, 0.1, 0.3, 0.0]], dtype=torch.float64),
            kt,
            mask=active.unsqueeze(0),
        )
        with torch.autograd.forward_ad.dual_level():
            l_inv_dual = torch.autograd.forward_ad.make_dual(
                l_inv_eye, torch.ones_like(l_inv_eye)
            )
            with pytest.raises(RuntimeError, match="forward-AD dual"):
                fermi_density_matrix(
                    h_det,
                    torch.tensor([1.5], dtype=torch.float64),
                    kt,
                    st_det,
                    l_inv=l_inv_dual,
                    mask=active,
                    eager_primal="spectral",
                )

        # Verify float32 detached-primal spectral parity.
        h_f32 = h_det.to(torch.float32)
        kt_f32 = kt.to(torch.float32)
        shift_f32 = torch.tensor(0.0, dtype=torch.float32)
        d_spec_f32 = spectral_module._matrix_fermi_spectral(
            h_f32, shift_f32, kt_f32, active
        )
        d_stat_f32 = _matrix_fermi(h_f32, shift_f32, kt_f32, active)
        torch.testing.assert_close(
            d_spec_f32, d_stat_f32, atol=5.0e-6, rtol=5.0e-5
        )


def test_cold_gap_newton_step_is_bounded_against_roundoff_division() -> None:
    """Roundoff-level slope in a cold ungapped channel cannot cause a bogus step."""
    # Construct a rotated 4-orbital spectrum with a ~80 kT gap where
    # filling._get_fermi_occupation_core_state still marks state.gap = False
    # (tail exp(-40) ~ 4.2e-18 > diff_floor 3.16e-73), while in matrix space
    # 1.0 - 4.2e-18 rounds to 1.0 so Tr(D - D^2) is pure float64 roundoff.
    kt = torch.tensor(300.0 * KELVIN2AU, dtype=torch.float64)
    gap = 80.0 * kt
    evals = torch.stack(
        (
            -1.2 * gap,
            -0.5 * gap,
            0.5 * gap,
            1.3 * gap,
        )
    )
    gen = torch.Generator().manual_seed(19)
    q_rot, _ = torch.linalg.qr(
        torch.randn((4, 4), generator=gen, dtype=torch.float64)
    )
    h_rot = q_rot @ torch.diag(evals) @ q_rot.mT
    h_rot = 0.5 * (h_rot + h_rot.mT)
    electrons = torch.tensor([2.0, 2.0], dtype=torch.float64)
    state = filling._get_fermi_occupation_core_state(
        electrons,
        evals.expand(2, -1),
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
    )
    assert not bool(state.gap.any())
    # Perturb the chemical potential offset by a tiny roundoff-scale amount
    # (5e-15) to mimic rotated matrix roundoff in target - count.
    perturbed_state = replace(
        state,
        chemical_potential_offset=state.chemical_potential_offset + 5.0e-15,
    )
    setup_static = spectral_module._fermi_free_energy_setup(
        h_rot,
        electrons,
        kt,
        perturbed_state,
        None,
        None,
        diff_steps=2,
        eager_primal="static",
    )
    with torch.no_grad():
        setup_spectral = spectral_module._fermi_free_energy_setup(
            h_rot,
            electrons,
            kt,
            perturbed_state,
            None,
            None,
            diff_steps=2,
            eager_primal="spectral",
        )
    assert bool(torch.isfinite(setup_static.shift).all())
    assert bool(torch.isfinite(setup_spectral.shift).all())
    assert bool(setup_static.gap_mask.all())
    assert bool(setup_spectral.gap_mask.all())
    assert float(setup_static.shift.abs().max()) <= float(3.0 * kt)
    assert float(setup_spectral.shift.abs().max()) <= float(3.0 * kt)

    nel_grad = electrons.clone().requires_grad_(True)
    density = fermi_density_matrix(
        h_rot, nel_grad, kt, perturbed_state, diff_steps=2
    )
    assert bool(torch.isfinite(density).all())
    torch.testing.assert_close(
        density.trace(),
        electrons.sum(),
        atol=1.0e-12,
        rtol=0.0,
    )
    (grad_nel,) = torch.autograd.grad(density.trace(), nel_grad)
    torch.testing.assert_close(
        grad_nel,
        torch.ones_like(electrons),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_daleckii_krein_fermi_jvp_matches_matrix_fermi_and_is_rotation_invariant() -> (
    None
):
    """Daleckii-Krein JVP matches _matrix_fermi across exact and near degeneracies."""
    kt = torch.tensor(300.0 * KELVIN2AU, dtype=torch.float64)
    shift = torch.tensor([0.01, -0.02], dtype=torch.float64)
    tangent_shift = torch.tensor([0.05, -0.03], dtype=torch.float64)
    tangent_kt = torch.tensor(1.5e-4, dtype=torch.float64)
    active = torch.tensor([True, True, True, True, False], dtype=torch.bool)

    evals = torch.tensor(
        [-0.25, 0.01, 0.01 + 1.0e-9, 0.01, 10.0], dtype=torch.float64
    )
    gen = torch.Generator().manual_seed(31)
    q_act, _ = torch.linalg.qr(
        torch.randn((4, 4), generator=gen, dtype=torch.float64)
    )
    u = torch.eye(5, dtype=torch.float64)
    u[:4, :4] = q_act
    h0 = u @ torch.diag(evals) @ u.mT
    h0 = 0.5 * (h0 + h0.mT)
    dh0 = torch.randn((5, 5), generator=gen, dtype=torch.float64)
    dh0 = 0.5 * (dh0 + dh0.mT)

    h_batch = h0.unsqueeze(0).expand(2, -1, -1)
    dh_batch = dh0.unsqueeze(0).expand(2, -1, -1)
    active_batch = active.unsqueeze(0).expand(2, -1)

    dk_val, dk_tan = spectral_module._fermi_daleckii_krein_jvp(
        h_batch,
        dh_batch,
        shift,
        tangent_shift,
        kt.expand(2),
        tangent_kt.expand(2),
        active_batch,
    )
    mf_val, mf_tan = jvp(
        lambda m, s, t: _matrix_fermi(m, s, t, active_batch),
        (h_batch, shift, kt.expand(2)),
        (dh_batch, tangent_shift, tangent_kt.expand(2)),
    )
    torch.testing.assert_close(dk_val, mf_val, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(dk_tan, mf_tan, atol=1.0e-9, rtol=1.0e-9)

    # Orthogonal rotation within the exact 2-fold degenerate subspace (indices 1 and 3)
    angle = torch.tensor(0.93, dtype=torch.float64)
    c, s = torch.cos(angle), torch.sin(angle)
    rot_deg = torch.eye(5, dtype=torch.float64)
    rot_deg[1, 1] = c
    rot_deg[1, 3] = -s
    rot_deg[3, 1] = s
    rot_deg[3, 3] = c
    u2 = u @ rot_deg
    h0_rot = u2 @ torch.diag(evals) @ u2.mT
    dk_val2, dk_tan2 = spectral_module._fermi_daleckii_krein_jvp(
        h0_rot.unsqueeze(0).expand(2, -1, -1),
        dh_batch,
        shift,
        tangent_shift,
        kt.expand(2),
        tangent_kt.expand(2),
        active_batch,
    )
    torch.testing.assert_close(dk_val2, dk_val, atol=1.0e-13, rtol=1.0e-13)
    torch.testing.assert_close(dk_tan2, dk_tan, atol=1.0e-12, rtol=1.0e-12)


def test_spectral_positive_definite_solve_supports_forward_second_derivatives() -> (
    None
):
    """The Cholesky solve keeps forward and reverse Hessians consistent."""
    dtype = torch.float64
    matrix = torch.tensor(
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]],
        dtype=dtype,
    )
    direction = torch.tensor(
        [[0.1, 0.2, -0.1], [0.2, -0.15, 0.05], [-0.1, 0.05, 0.08]],
        dtype=dtype,
    )
    rhs = torch.tensor([[1.0], [-0.5], [0.25]], dtype=dtype)
    weight = torch.tensor([[0.5], [1.25], [-0.75]], dtype=dtype)

    def function(value):
        solved = _solve_positive_definite(matrix + value * direction, rhs)
        return (solved * weight).sum()

    zero = torch.zeros((), dtype=dtype)
    tangent = torch.ones_like(zero)
    forward_second = torch.func.jvp(
        lambda value: torch.func.jvp(function, (value,), (tangent,))[1],
        (zero,),
        (tangent,),
    )[1]
    reverse_second = torch.func.jvp(
        torch.func.grad(function), (zero,), (tangent,)
    )[1]
    step = 1.0e-3
    finite_difference = (
        torch.func.grad(function)(zero + step)
        - torch.func.grad(function)(zero - step)
    ) / (2.0 * step)

    torch.testing.assert_close(
        forward_second, reverse_second, atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        forward_second, finite_difference, atol=1.0e-6, rtol=1.0e-6
    )
