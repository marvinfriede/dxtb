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
"""Focused acceptance tests for invariant functional SCF free energy."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.func import jacfwd, jacrev, jvp, vmap
from torch.utils import _pytree

import dxtb._src.scf.functional.map as functional_map
from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.scf.functional.map import evaluate_hamiltonian
from dxtb._src.scf.functional.solver import free_energy_from_spectral
from dxtb._src.scf.functional.spectral import (
    aufbau_free_energy_matrix,
    fermi_free_energy_matrix,
    fermi_free_energy_trace,
)
from dxtb._src.wavefunction import filling, mulliken

from .test_functional_map import _prepare_case


def _fermi_state(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    energies = torch.linalg.eigvalsh(hamiltonian)
    channel_energies = energies.expand(*electrons.shape, energies.shape[-1])
    channel_mask = None if mask is None else mask.expand_as(channel_energies)
    return filling._get_fermi_occupation_core_state(
        electrons,
        channel_energies,
        kt,
        mask=channel_mask,
        thr=1.0e-13,
        maxiter=300,
        diff_order=3,
    )


def _fermi_matrix_trace(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    l_inv: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the overlap-weighted trace of the shared-root matrix."""
    state = _fermi_state(
        hamiltonian if l_inv is None else l_inv @ hamiltonian @ l_inv.mT,
        electrons,
        kt,
        mask,
    )
    matrix = fermi_free_energy_matrix(
        hamiltonian,
        electrons,
        kt,
        state,
        l_inv=l_inv,
        mask=mask,
        diff_steps=filling._diff_steps(3),
    )
    return matrix.trace()


def test_fermi_free_energy_exact_degeneracy_transforms_and_rotation() -> None:
    """Exact degenerate entropy has invariant reverse and forward responses."""
    dtype = torch.float64
    h0 = torch.diag(torch.tensor([-0.2, 0.0, 0.0, 0.4], dtype=dtype))
    electrons = torch.tensor([1.5, 1.5], dtype=dtype)
    kt = torch.tensor(0.02, dtype=dtype)
    direction = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.0],
            [0.1, 0.1, 0.3, 0.0],
            [0.2, 0.3, -0.1, 0.0],
            [0.0, 0.0, 0.0, 0.2],
        ],
        dtype=dtype,
    )
    angle = torch.tensor(0.73, dtype=dtype)
    rotation = torch.eye(4, dtype=dtype)
    rotation[1:3, 1:3] = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )

    def response(base: torch.Tensor, tangent: torch.Tensor):
        def scalar(parameter: torch.Tensor) -> torch.Tensor:
            return _fermi_matrix_trace(
                base + parameter * tangent, electrons, kt
            )

        parameter = torch.zeros((), dtype=dtype, requires_grad=True)
        value = scalar(parameter)
        first = torch.autograd.grad(value, parameter, create_graph=True)[0]
        second = torch.autograd.grad(first, parameter, create_graph=True)[0]
        third = torch.autograd.grad(second, parameter)[0]
        _, tangent_value = jvp(
            scalar, (parameter.detach(),), (torch.ones_like(parameter),)
        )
        forward = jacfwd(scalar)(parameter.detach())
        return (
            value.detach(),
            first.detach(),
            second.detach(),
            third,
            tangent_value,
            forward,
        )

    ordinary = response(h0, direction)
    rotated = response(
        rotation @ h0 @ rotation.mT, rotation @ direction @ rotation.mT
    )
    for left, right in zip(ordinary, rotated, strict=True):
        torch.testing.assert_close(left, right, atol=2.0e-8, rtol=2.0e-8)
    torch.testing.assert_close(
        ordinary[1], ordinary[4], atol=2.0e-8, rtol=2.0e-8
    )
    torch.testing.assert_close(
        ordinary[1], ordinary[5], atol=2.0e-8, rtol=2.0e-8
    )

    steps = (2.0e-4, 1.0e-4, 5.0e-5)
    physical_fd = torch.stack(
        [
            (
                _fermi_matrix_trace(h0 + h * direction, electrons, kt)
                - _fermi_matrix_trace(h0 - h * direction, electrons, kt)
            )
            / (2.0 * h)
            for h in steps
        ]
    )
    torch.testing.assert_close(
        ordinary[1], physical_fd.mean(), atol=2.0e-6, rtol=2.0e-6
    )


@pytest.mark.parametrize(
    "partition",
    [labels.FERMI_PARTITION_EQUAL, labels.FERMI_PARTITION_ATOMIC],
)
def test_exact_degenerate_public_partitions_ignore_reporting_rotation(
    partition: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal and atomic free energy ignore rotations of degenerate columns."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "fermi_etemp": 300.0,
            "fermi_diff_order": 3,
            "fermi_partition": partition,
        },
    )
    inputs = replace(
        case.inputs,
        nel=torch.tensor([2.5, 2.5], dtype=torch.float64),
    )
    energies = torch.tensor(
        [-0.4, -0.2, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8],
        dtype=torch.float64,
    )
    orthogonal = torch.diag(energies)
    inverse = torch.linalg.inv(inputs.inverse_cholesky)
    h0 = inverse @ orthogonal @ inverse.mT
    direction = torch.zeros_like(h0)
    direction[2, 3] = direction[3, 2] = 0.03
    angle = torch.tensor(0.41, dtype=torch.float64)
    rotation = torch.eye(energies.numel(), dtype=torch.float64)
    rotation[2:4, 2:4] = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    original = functional_map._transitional_diagonalize

    def evaluate(parameter: torch.Tensor, rotate: bool) -> torch.Tensor:
        def diagonalize(hamiltonian: torch.Tensor, scf_inputs: object):
            orbital_energies, coefficients = original(hamiltonian, scf_inputs)
            if rotate:
                coefficients = coefficients @ rotation
            return orbital_energies, coefficients

        monkeypatch.setattr(
            functional_map, "_transitional_diagonalize", diagonalize
        )
        spectral = evaluate_hamiltonian(h0 + parameter * direction, inputs)
        return free_energy_from_spectral(spectral, inputs).sum()

    derivatives = []
    for rotate in (False, True):
        parameter = torch.zeros((), dtype=torch.float64, requires_grad=True)
        value = evaluate(parameter, rotate)
        first = torch.autograd.grad(value, parameter, create_graph=True)[0]
        second = torch.autograd.grad(first, parameter, create_graph=True)[0]
        third = torch.autograd.grad(second, parameter)[0]
        _, tangent = jvp(
            lambda p: evaluate(p, rotate),
            (parameter.detach(),),
            (torch.ones_like(parameter),),
        )
        forward = jacfwd(lambda p: evaluate(p, rotate))(parameter.detach())
        derivatives.append(
            torch.stack(
                (
                    value.detach(),
                    first.detach(),
                    second.detach(),
                    third,
                    tangent,
                    forward,
                )
            )
        )
    torch.testing.assert_close(
        derivatives[0], derivatives[1], atol=2.0e-8, rtol=2.0e-8
    )

    parameters = torch.tensor([-1.0e-5, 0.0, 1.0e-5], dtype=torch.float64)
    for rotate in (False, True):
        vmapped = torch.func.vmap(
            lambda parameter: evaluate(parameter, rotate)
        )(parameters)
        looped = torch.stack(
            [evaluate(parameter, rotate) for parameter in parameters]
        )
        torch.testing.assert_close(vmapped, looped, atol=2.0e-10, rtol=2.0e-10)


@pytest.mark.parametrize("temperature", [9.5e-4, 5.0e-2])
def test_fermi_free_energy_small_gap_is_continuous(temperature: float) -> None:
    """Finite-temperature matrix entropy approaches its degenerate limit."""
    dtype = torch.float64
    electrons = torch.tensor([1.5, 1.5], dtype=dtype)
    kt = torch.tensor(temperature, dtype=dtype)
    values = []
    for gap in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 0.0):
        hamiltonian = torch.diag(
            torch.tensor([-0.2, -gap / 2, gap / 2, 0.4], dtype=dtype)
        )
        values.append(_fermi_matrix_trace(hamiltonian, electrons, kt))
    assert torch.isfinite(torch.stack(values)).all()
    assert abs(values[-2] - values[-1]) < abs(values[0] - values[-1])
    assert abs(values[-1]) < 4.0 * kt * torch.log(
        torch.tensor(2.0, dtype=dtype)
    )


def test_fermi_free_energy_generalized_overlap_response_matches_fd() -> None:
    """Free-energy matrix responds to H and inverse-Cholesky perturbations."""
    dtype = torch.float64
    h0 = torch.tensor(
        [[-0.08, 0.012, 0.0], [0.012, 0.01, 0.006], [0.0, 0.006, 0.09]],
        dtype=dtype,
    )
    li0 = torch.tensor(
        [[1.02, 0.0, 0.0], [0.04, 0.96, 0.0], [-0.02, 0.03, 1.04]],
        dtype=dtype,
    )
    dh = torch.tensor(
        [[0.2, -0.03, 0.01], [-0.03, -0.1, 0.02], [0.01, 0.02, 0.07]],
        dtype=dtype,
    )
    dli = torch.tensor(
        [[0.03, 0.0, 0.0], [-0.02, -0.01, 0.0], [0.01, 0.04, -0.03]],
        dtype=dtype,
    )
    electrons = torch.tensor([1.25, 0.0], dtype=dtype)
    kt = torch.tensor(2.0e-2, dtype=dtype)
    weight = torch.tensor(
        [[0.3, -0.2, 0.1], [-0.2, 0.4, 0.15], [0.1, 0.15, -0.1]],
        dtype=dtype,
    )

    for mode in ("hamiltonian", "overlap", "combined"):
        local_h = torch.zeros_like(dh) if mode == "overlap" else dh
        local_li = torch.zeros_like(dli) if mode == "hamiltonian" else dli

        def observable(h: torch.Tensor, li: torch.Tensor) -> torch.Tensor:
            state = _fermi_state(li @ h @ li.mT, electrons, kt)
            matrix = fermi_free_energy_matrix(
                h, electrons, kt, state, l_inv=li, diff_steps=2
            )
            return torch.sum(weight * matrix)

        def along(parameter: torch.Tensor) -> torch.Tensor:
            return observable(
                h0 + parameter * local_h, li0 + parameter * local_li
            )

        parameter = torch.zeros((), dtype=dtype)
        _, tangent = jvp(
            lambda h, li: observable(h, li),
            (h0, li0),
            (local_h, local_li),
        )
        forward = jacfwd(along)(parameter)
        torch.testing.assert_close(tangent, forward, atol=2.0e-7, rtol=2.0e-7)
        steps = (2.0e-4, 1.0e-4, 5.0e-5)
        fd = torch.stack([(along(h) - along(-h)) / (2.0 * h) for h in steps])
        torch.testing.assert_close(tangent, fd.mean(), atol=3.0e-5, rtol=3.0e-5)


def test_fermi_partitions_share_matrix_trace_and_authoritative_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal and atomic partitions conserve one selected Fermi matrix."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "fermi_etemp": 300.0,
            "fermi_diff_order": 3,
            "fermi_partition": labels.FERMI_PARTITION_EQUAL,
        },
    )
    calls = 0
    original = filling._get_fermi_occupation_core_state

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(filling, "_get_fermi_occupation_core_state", counted)
    spectral = evaluate_hamiltonian(case.inputs.integrals.hcore, case.inputs)
    state = spectral._fermi_state
    assert state is not None
    assert spectral.fermi_status is state.result.status
    equal = free_energy_from_spectral(spectral, case.inputs)
    atomic_inputs = replace(
        case.inputs,
        fermi_settings=replace(
            case.inputs.fermi_settings,
            partition=labels.FERMI_PARTITION_ATOMIC,
        ),
    )
    atomic = free_energy_from_spectral(spectral, atomic_inputs)

    entropy = filling.get_fermi_entropy_finite(
        spectral.occupation,
        spectral.emo,
        case.inputs.fermi_settings.kt,
        case.inputs.orbital_mask,
    )
    old_equal = entropy.sum()
    old_matrix = torch.einsum(
        "ik,k,jk->ij",
        spectral.coefficients,
        entropy,
        spectral.coefficients,
    )
    old_atomic = mulliken.get_atomic_populations(
        case.inputs.integrals.overlap, old_matrix, case.inputs.index
    ).sum()

    assert calls == 1
    assert state.result.status.converged.all()
    torch.testing.assert_close(
        equal.sum(), atomic.sum(), atol=2.0e-12, rtol=2.0e-12
    )
    torch.testing.assert_close(
        equal.sum(), old_equal, atol=2.0e-12, rtol=2.0e-9
    )
    torch.testing.assert_close(
        atomic.sum(), old_atomic, atol=2.0e-12, rtol=2.0e-9
    )
    assert torch.isfinite(atomic).all()


def test_invalid_fermi_channels_do_not_produce_valid_entropy_matrices() -> None:
    """Invalid filling states retain the authoritative zero diagnostic."""
    dtype = torch.float64
    hamiltonian = torch.diag(torch.tensor([-0.2, 0.1], dtype=dtype))
    kt = torch.tensor(2.0e-2, dtype=dtype)
    mask = torch.ones(2, dtype=torch.bool)
    too_many_nel = torch.tensor([3.0], dtype=dtype)
    too_many = _fermi_state(hamiltonian, too_many_nel, kt, mask)
    assert too_many.result.status.too_many_electrons.any()
    too_many_matrix = fermi_free_energy_matrix(
        hamiltonian, too_many_nel, kt, too_many, mask=mask
    )
    assert torch.equal(too_many_matrix, torch.zeros_like(too_many_matrix))

    energies = hamiltonian.diagonal().expand(1, -1)
    unconverged = filling._get_fermi_occupation_core_state(
        torch.tensor([0.5], dtype=dtype),
        energies,
        kt,
        mask=mask.expand_as(energies),
        thr=1.0e-13,
        maxiter=0,
        diff_order=3,
    )
    assert not unconverged.result.status.converged.all()
    invalid_matrix = fermi_free_energy_matrix(
        hamiltonian,
        torch.tensor([0.5], dtype=dtype),
        kt,
        unconverged,
        mask=mask,
    )
    assert torch.equal(invalid_matrix, torch.zeros_like(invalid_matrix))


@pytest.mark.parametrize("electrons", [0.0, 2.0])
def test_fermi_saturated_channels_preserve_entropy_floor(
    electrons: float,
) -> None:
    """Valid saturated channels retain the legacy finite-temperature floor."""
    dtype = torch.float64
    hamiltonian = torch.diag(torch.tensor([2.0, 3.0], dtype=dtype))
    nel = torch.tensor([electrons], dtype=dtype)
    kt = torch.tensor(2.0e-2, dtype=dtype)
    state = _fermi_state(hamiltonian, nel, kt)
    assert state.result.status.converged.all()
    matrix = fermi_free_energy_matrix(hamiltonian, nel, kt, state)
    legacy = filling.get_fermi_entropy_finite(
        state.result.occupation, torch.linalg.eigvalsh(hamiltonian), kt
    ).sum()
    torch.testing.assert_close(
        matrix.trace(), legacy, atol=1.0e-16, rtol=1.0e-12
    )

    parameter = torch.zeros((), dtype=dtype, requires_grad=True)
    direction = torch.eye(2, dtype=dtype)

    def response(p: torch.Tensor) -> torch.Tensor:
        return fermi_free_energy_matrix(
            hamiltonian + p * direction, nel, kt, state
        ).trace()

    derivative = torch.autograd.grad(
        response(parameter), parameter, create_graph=True
    )[0]
    _, tangent = jvp(
        response, (parameter.detach(),), (torch.ones_like(parameter),)
    )
    forward = jacfwd(response)(parameter.detach())
    torch.testing.assert_close(derivative, torch.zeros_like(derivative))
    torch.testing.assert_close(tangent, torch.zeros_like(tangent))
    torch.testing.assert_close(forward, torch.zeros_like(forward))

    thermal = torch.tensor(2.0e-2, dtype=dtype, requires_grad=True)
    thermal_value = fermi_free_energy_matrix(
        hamiltonian, nel, thermal, state
    ).trace()
    thermal_derivative = torch.autograd.grad(thermal_value, thermal)[0]
    floor_slope = (
        2
        * torch.finfo(dtype).eps
        * torch.log(torch.tensor(torch.finfo(dtype).eps, dtype=dtype))
    )
    torch.testing.assert_close(thermal_derivative, floor_slope)

    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "fermi_etemp": 300.0,
            "fermi_partition": labels.FERMI_PARTITION_EQUAL,
        },
    )
    inputs = case.inputs
    electrons_empty = torch.zeros_like(inputs.nel)
    state_empty = _fermi_state(
        inputs.inverse_cholesky
        @ inputs.integrals.hcore
        @ inputs.inverse_cholesky.mT,
        electrons_empty,
        inputs.fermi_settings.kt,
        inputs.orbital_mask,
    )
    empty_matrix = fermi_free_energy_matrix(
        inputs.integrals.hcore,
        electrons_empty,
        inputs.fermi_settings.kt,
        state_empty,
        l_inv=inputs.inverse_cholesky,
        mask=inputs.orbital_mask,
    )
    equal_total = torch.trace(inputs.integrals.overlap @ empty_matrix)
    atomic_total = mulliken.get_atomic_populations(
        inputs.integrals.overlap, empty_matrix, inputs.index
    ).sum()
    torch.testing.assert_close(equal_total, atomic_total)


@pytest.mark.parametrize(
    "partition",
    [labels.FERMI_PARTITION_EQUAL, labels.FERMI_PARTITION_ATOMIC],
)
def test_no2_open_shell_free_energy_uses_channelwise_root(
    partition: int,
) -> None:
    """GFN2/NO2 free energy preserves both Fermi channels and transforms."""
    case = _prepare_case(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        scf_options={
            "fermi_etemp": 300.0,
            "fermi_diff_order": 3,
            "fermi_partition": partition,
        },
    )
    inputs = case.inputs
    h0 = inputs.integrals.hcore
    direction = torch.zeros_like(h0)
    direction[0, 1] = direction[1, 0] = 0.1

    def response(parameter: torch.Tensor) -> torch.Tensor:
        local = replace(
            inputs,
            integrals=replace(
                inputs.integrals, hcore=h0 + parameter * direction
            ),
        )
        spectral = evaluate_hamiltonian(local.integrals.hcore, local)
        return free_energy_from_spectral(spectral, local).sum()

    spectral = evaluate_hamiltonian(h0, inputs)
    state = spectral._fermi_state
    assert state is not None
    assert spectral.fermi_status is state.result.status
    torch.testing.assert_close(
        spectral.occupation.sum(-1), inputs.nel, atol=2.0e-9, rtol=0.0
    )
    old_entropy = filling.get_fermi_entropy_finite(
        spectral.occupation,
        spectral.emo,
        inputs.fermi_settings.kt,
        inputs.orbital_mask,
    )
    if partition == labels.FERMI_PARTITION_EQUAL:
        old_value = old_entropy.sum()
    else:
        old_matrix = torch.einsum(
            "ik,k,jk->ij",
            spectral.coefficients,
            old_entropy,
            spectral.coefficients,
        )
        old_value = mulliken.get_atomic_populations(
            inputs.integrals.overlap, old_matrix, inputs.index
        ).sum()
    new_value = free_energy_from_spectral(spectral, inputs).sum()
    torch.testing.assert_close(new_value, old_value, atol=2.0e-12, rtol=2.0e-9)
    value = response(torch.zeros((), dtype=h0.dtype))
    parameter = torch.zeros((), dtype=h0.dtype, requires_grad=True)
    reverse = torch.autograd.grad(response(parameter), parameter)[0]
    _, tangent = jvp(
        response, (parameter.detach(),), (torch.ones_like(parameter),)
    )
    forward = jacfwd(response)(parameter.detach())
    assert torch.isfinite(value)
    assert torch.isfinite(torch.stack((reverse, tangent, forward))).all()
    torch.testing.assert_close(reverse, tangent, atol=3.0e-7, rtol=3.0e-7)
    torch.testing.assert_close(reverse, forward, atol=3.0e-7, rtol=3.0e-7)


@pytest.mark.parametrize(
    "partition",
    [labels.FERMI_PARTITION_EQUAL, labels.FERMI_PARTITION_ATOMIC],
)
def test_low_temperature_partition_matches_legacy_forward_value(
    partition: int,
) -> None:
    """Below-cutoff Aufbau entropy keeps both old partition forward values."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "fermi_etemp": 1.0e-7,
            "fermi_partition": partition,
        },
    )
    spectral = evaluate_hamiltonian(case.inputs.integrals.hcore, case.inputs)
    entropy = filling.get_fermi_entropy_low_temperature(
        spectral.occupation, case.inputs.fermi_settings.kt
    )
    if partition == labels.FERMI_PARTITION_EQUAL:
        expected = entropy.sum() / case.inputs.numbers.numel()
        expected = expected.expand_as(case.inputs.numbers)
    else:
        legacy = torch.einsum(
            "ik,k,jk->ij",
            spectral.coefficients,
            entropy,
            spectral.coefficients,
        )
        expected = mulliken.get_atomic_populations(
            case.inputs.integrals.overlap, legacy, case.inputs.index
        )
    new_value = free_energy_from_spectral(spectral, case.inputs)
    torch.testing.assert_close(new_value, expected, atol=2.0e-14, rtol=2.0e-9)


@pytest.mark.parametrize(
    ("energies", "electrons"),
    (
        ([1.0], 0.0),
        ([1.0], 1.0),
        ([-4.0], 0.0),
        ([-4.0], 1.0),
        ([2.0, 3.0, 5.0], 0.0),
        ([2.0, 3.0, 5.0], 3.0),
    ),
)
def test_low_temperature_saturated_free_energy_matches_legacy(
    energies: list[float], electrons: float
) -> None:
    """Empty/full Aufbau entropy is finite and independent of H."""
    dtype = torch.float64
    hamiltonian = torch.diag(torch.tensor(energies, dtype=dtype))
    nel = torch.tensor([electrons], dtype=dtype)
    kt = torch.tensor(1.0e-7, dtype=dtype)

    def observable(matrix: torch.Tensor) -> torch.Tensor:
        return aufbau_free_energy_matrix(matrix, nel, kt).trace()

    actual = observable(hamiltonian)
    occupation = filling.get_aufbau_occupation(torch.tensor(len(energies)), nel)
    expected = filling.get_fermi_entropy_low_temperature(occupation, kt).sum()
    torch.testing.assert_close(actual, expected, atol=1.0e-18, rtol=1.0e-12)

    parameter = torch.zeros((), dtype=dtype, requires_grad=True)
    direction = torch.ones_like(hamiltonian)
    value = observable(hamiltonian + parameter * direction)
    derivative = torch.autograd.grad(value, parameter, create_graph=True)[0]
    _, tangent = jvp(
        lambda p: observable(hamiltonian + p * direction),
        (parameter.detach(),),
        (torch.ones_like(parameter),),
    )
    forward = jacfwd(lambda p: observable(hamiltonian + p * direction))(
        parameter.detach()
    )
    torch.testing.assert_close(derivative, torch.zeros_like(derivative))
    torch.testing.assert_close(tangent, torch.zeros_like(tangent))
    torch.testing.assert_close(forward, torch.zeros_like(forward))


@pytest.mark.parametrize("origin", [2.0, -4.0])
@pytest.mark.parametrize("electrons", [0.0, 3.0])
def test_low_temperature_degenerate_boundary_rank_is_exact(
    origin: float, electrons: float
) -> None:
    """Boundary projectors of a constant spectrum avoid sign solves."""
    dtype = torch.float64
    hamiltonian = origin * torch.eye(3, dtype=dtype)
    matrix = aufbau_free_energy_matrix(
        hamiltonian,
        torch.tensor([electrons], dtype=dtype),
        torch.tensor(1.0e-7, dtype=dtype),
    )
    assert torch.isfinite(matrix).all()
    assert torch.isfinite(matrix.trace())


def test_aufbau_free_energy_matrix_vmap_respects_lane_masks() -> None:
    """Low-temperature free-energy matrices preserve distinct lane masks."""
    hamiltonians = torch.stack(
        (
            torch.diag(torch.tensor([-0.2, 0.1, 20.0], dtype=torch.float64)),
            torch.diag(torch.tensor([-0.1, 0.2, 0.3], dtype=torch.float64)),
        )
    )
    electrons = torch.tensor([[1.25, 0.0], [2.0, 0.0]], dtype=torch.float64)
    masks = torch.tensor([[True, True, False], [True, True, True]])
    kt = torch.tensor(1.0e-7, dtype=torch.float64)
    mapped = torch.func.vmap(
        lambda h, nel, mask: aufbau_free_energy_matrix(h, nel, kt, mask=mask)
    )(hamiltonians, electrons, masks)
    looped = torch.stack(
        [
            aufbau_free_energy_matrix(
                hamiltonians[index], electrons[index], kt, mask=masks[index]
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(mapped, looped, atol=1.0e-12, rtol=1.0e-12)
    assert torch.equal(mapped[0, 2], torch.zeros_like(mapped[0, 2]))
    assert torch.equal(mapped[0, :, 2], torch.zeros_like(mapped[0, :, 2]))


def test_fermi_free_energy_vmap_preserves_lane_specific_masks() -> None:
    """Finite-temperature free-energy matrices vmap distinct masks safely."""
    hamiltonians = torch.stack(
        (
            torch.diag(torch.tensor([-0.2, 0.1, 20.0], dtype=torch.float64)),
            torch.diag(torch.tensor([-0.1, 0.2, 0.3], dtype=torch.float64)),
        )
    )
    electrons = torch.tensor([[1.0, 0.0], [1.5, 0.0]], dtype=torch.float64)
    masks = torch.tensor([[True, True, False], [True, True, True]])
    inverse_cholesky = torch.eye(3, dtype=torch.float64).expand(2, 3, 3).clone()
    kt = torch.tensor(2.0e-2, dtype=torch.float64)
    states = [
        _fermi_state(
            inverse_cholesky[i] @ hamiltonians[i] @ inverse_cholesky[i].mT,
            electrons[i],
            kt,
            masks[i],
        )
        for i in range(2)
    ]
    batched_state = _pytree.tree_map(
        lambda *values: torch.stack(values), *states
    )

    def evaluate(h, li, nel, state):
        return fermi_free_energy_matrix(
            h, nel, kt, state, l_inv=li, diff_steps=2
        )

    vmapped = torch.func.vmap(evaluate)(
        hamiltonians, inverse_cholesky, electrons, batched_state
    )
    looped = torch.stack(
        [
            evaluate(
                hamiltonians[i], inverse_cholesky[i], electrons[i], states[i]
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(vmapped, looped, atol=2.0e-10, rtol=2.0e-10)
    assert torch.equal(vmapped[0, 2], torch.zeros_like(vmapped[0, 2]))
    assert torch.equal(vmapped[0, :, 2], torch.zeros_like(vmapped[0, :, 2]))

    h_direction = torch.zeros_like(hamiltonians[0])
    h_direction[0, 2] = h_direction[2, 0] = 1.0
    li_direction = torch.zeros_like(inverse_cholesky[0])
    li_direction[1, 2] = li_direction[2, 1] = 1.0
    _, tangent = jvp(
        lambda h, li: evaluate(h, li, electrons[0], states[0]),
        (hamiltonians[0], inverse_cholesky[0]),
        (h_direction, li_direction),
    )
    assert torch.equal(tangent, torch.zeros_like(tangent))


def _trace_cases():
    dtype = torch.float64
    li0 = torch.tensor(
        [[1.02, 0.0, 0.0], [0.04, 0.96, 0.0], [-0.02, 0.03, 1.04]],
        dtype=dtype,
    )
    dli = torch.tensor(
        [[0.03, 0.0, 0.0], [-0.02, -0.01, 0.0], [0.01, 0.04, -0.03]],
        dtype=dtype,
    )
    h3 = torch.tensor(
        [[-0.08, 0.012, 0.0], [0.012, 0.01, 0.006], [0.0, 0.006, 0.09]],
        dtype=dtype,
    )
    dh3 = torch.tensor(
        [[0.2, -0.03, 0.01], [-0.03, -0.1, 0.02], [0.01, 0.02, 0.07]],
        dtype=dtype,
    )
    d4 = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.0],
            [0.1, 0.1, 0.3, 0.0],
            [0.2, 0.3, -0.1, 0.0],
            [0.0, 0.0, 0.0, 0.2],
        ],
        dtype=dtype,
    )
    eye4, eye2 = torch.eye(4, dtype=dtype), torch.eye(2, dtype=dtype)
    t = lambda *v: torch.tensor(v, dtype=dtype)  # noqa: E731
    return {
        "degenerate": (
            torch.diag(t(-0.2, 0.0, 0.0, 0.4)),
            d4,
            eye4,
            None,
            t(1.5, 1.5),
            t(0.02),
        ),
        "small_gap_300K": (
            torch.diag(t(-0.2, -5.0e-5, 5.0e-5, 0.4)),
            d4,
            eye4,
            None,
            t(1.5, 1.5),
            t(9.5e-4),
        ),
        "overlap": (h3, dh3, li0, dli, t(1.25, 0.0), t(2.0e-2)),
        "saturated": (
            torch.diag(t(2.0, 3.0)),
            eye2,
            eye2,
            None,
            t(2.0),
            t(2.0e-2),
        ),
        "wide_gap": (
            torch.diag(t(-0.5, 0.5)),
            t(0.0, 0.1, 0.1, 0.05).reshape(2, 2),
            eye2,
            None,
            t(1.0),
            t(1.0e-5),
        ),
    }


@pytest.mark.parametrize("name", list(_trace_cases()))
def test_fermi_free_energy_trace_matches_matrix_to_third_order(
    name: str,
) -> None:
    """The trace path reproduces the overlap-weighted matrix trace."""
    h0, dh, li0, dli, electrons, kt = _trace_cases()[name]
    dli = torch.zeros_like(li0) if dli is None else dli
    state = _fermi_state(li0 @ h0 @ li0.mT, electrons, kt)

    def matrix_path(p: torch.Tensor) -> torch.Tensor:
        h, li = h0 + p * dh, li0 + p * dli
        s = torch.linalg.inv(li) @ torch.linalg.inv(li).mT
        matrix = fermi_free_energy_matrix(
            h, electrons, kt, state, l_inv=li, diff_steps=2
        )
        return torch.trace(s @ matrix)

    def trace_path(p: torch.Tensor) -> torch.Tensor:
        return fermi_free_energy_trace(
            h0 + p * dh,
            electrons,
            kt,
            state,
            l_inv=li0 + p * dli,
            diff_steps=2,
        )

    def reverse(f, p):
        p = p.detach().requires_grad_(True)
        out = [f(p)]
        for order in range(3):
            out.append(
                torch.autograd.grad(out[-1], p, create_graph=order < 2)[0]
            )
        return torch.stack([o.detach() for o in out])

    p0 = torch.tensor(0.01, dtype=torch.float64)
    old, new = reverse(matrix_path, p0), reverse(trace_path, p0)
    scale = float(old.abs().max().clamp_min(kt))
    torch.testing.assert_close(new, old, atol=1.0e-10 * scale, rtol=1.0e-8)

    fwd3 = jacfwd(jacfwd(jacfwd(trace_path)))(p0)
    mixed = jacfwd(jacrev(jacfwd(trace_path)))(p0)
    torch.testing.assert_close(fwd3, new[3], atol=1.0e-9 * scale, rtol=1e-9)
    torch.testing.assert_close(mixed, new[3], atol=1.0e-9 * scale, rtol=1e-9)

    points = torch.stack((p0, -p0))
    batched = vmap(jacfwd(jacrev(trace_path)))(points)
    looped = torch.stack([jacfwd(jacrev(trace_path))(p) for p in points])
    torch.testing.assert_close(batched, looped, atol=1.0e-12, rtol=1.0e-10)


@pytest.mark.parametrize("name", list(_trace_cases()))
def test_fermi_free_energy_trace_electron_count_response(name: str) -> None:
    """Electron-count derivatives, including the gap response, agree."""
    h0, _, li0, _, electrons, kt = _trace_cases()[name]
    state = _fermi_state(li0 @ h0 @ li0.mT, electrons, kt)
    s = torch.linalg.inv(li0) @ torch.linalg.inv(li0).mT

    def matrix_path(nel: torch.Tensor) -> torch.Tensor:
        matrix = fermi_free_energy_matrix(
            h0, nel, kt, state, l_inv=li0, diff_steps=2
        )
        return torch.trace(s @ matrix)

    def trace_path(nel: torch.Tensor) -> torch.Tensor:
        return fermi_free_energy_trace(
            h0, nel, kt, state, l_inv=li0, diff_steps=2
        )

    old = torch.func.hessian(matrix_path)(electrons)
    new = torch.func.hessian(trace_path)(electrons)
    first_old = torch.func.grad(matrix_path)(electrons)
    first_new = torch.func.grad(trace_path)(electrons)
    scale = float(first_old.abs().max().clamp_min(kt))
    torch.testing.assert_close(
        first_new, first_old, atol=1.0e-10 * scale, rtol=1.0e-8
    )
    torch.testing.assert_close(new, old, atol=1.0e-10 * scale, rtol=1.0e-8)
