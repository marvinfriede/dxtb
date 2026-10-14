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

import torch
import pytest
from torch.func import jacfwd, jvp
from tad_mctc import storch
from tad_mctc.units import KELVIN2AU

from dxtb._src.scf.functional.spectral import (
    _projector,
    aufbau_density_matrix,
    fermi_density_matrix,
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
