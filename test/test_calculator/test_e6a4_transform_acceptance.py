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
"""Candidate-backed public transform acceptance for E6a.4."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb.config import Config


def _system(
    method: object,
    numbers: tuple[int, ...],
    *,
    mixer: int | None = None,
    scp_mode: int | None = None,
    maxiter: int = 2,
    temperature: float = 0.0,
):
    """Build a production FULL System with its Model parameter owner."""
    is_gfn2 = method is GFN2_XTB
    config = Config.create(
        method=labels.GFN2_XTB if is_gfn2 else labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        mixer=(
            mixer
            if mixer is not None
            else labels.MIXER_ANDERSON if is_gfn2 else labels.MIXER_LINEAR
        ),
        scp_mode=(
            scp_mode
            if scp_mode is not None
            else labels.SCP_MODE_FOCK if is_gfn2 else labels.SCP_MODE_CHARGE
        ),
        maxiter=maxiter,
        fermi_etemp=temperature,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    parameters = ParamModule(method, dtype=torch.float64)
    model = Model(par=parameters, config=config)
    atomic_numbers = torch.tensor(numbers, dtype=torch.int64)
    return model, atomic_numbers, model.setup(atomic_numbers)


def _water_positions(count: int) -> torch.Tensor:
    """Return distinct, nondegenerate water conformers."""
    base = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]],
        dtype=torch.float64,
    )
    return torch.stack(
        [
            base
            + base.new_tensor(
                [[0.0, 0.0, 0.0], [delta, 0.0, 0.025], [-delta, 0.0, -0.025]]
            )
            for delta in torch.linspace(-0.035, 0.045, count)
        ]
    )


def _h2_positions(count: int) -> torch.Tensor:
    """Return distinct H2 conformers."""
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]], dtype=torch.float64
    )
    return torch.stack(
        [
            base + base.new_tensor([[0.0, 0.0, 0.0], [0.0, 0.0, delta]])
            for delta in torch.linspace(-0.04, 0.05, count)
        ]
    )


def _assert_result_tree_matches(mapped, serial: list[object]) -> None:
    """Check complete registered Result structure and every tensor leaf."""
    assert type(mapped) is type(serial[0])
    mapped_leaves, mapped_spec = _pytree.tree_flatten(mapped)
    serial_leaves = [_pytree.tree_flatten(result) for result in serial]
    assert all(spec == mapped_spec for _, spec in serial_leaves)
    for leaf_index, mapped_leaf in enumerate(mapped_leaves):
        assert isinstance(mapped_leaf, torch.Tensor)
        values = [leaves[leaf_index] for leaves, _ in serial_leaves]
        assert all(isinstance(value, torch.Tensor) for value in values)
        assert mapped_leaf.shape == (len(serial), *values[0].shape)
        assert mapped_leaf.dtype == values[0].dtype
        assert mapped_leaf.device == values[0].device
        torch.testing.assert_close(mapped_leaf, torch.stack(values))
    assert tuple(label for label, _ in mapped.classical) == tuple(
        label for label, _ in serial[0].classical
    )


@pytest.mark.parametrize("numbers", [(1, 1), (8, 1, 1)])
def test_gfn2_public_forward_and_mapped_derivatives(
    numbers: tuple[int, ...],
) -> None:
    """Full GFN2 public calls support forward and lane-local derivatives."""
    _, atomic_numbers, system = _system(GFN2_XTB, numbers)
    positions = _h2_positions(2) if numbers == (1, 1) else _water_positions(2)
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]
    _assert_result_tree_matches(mapped, serial)
    assert tuple(term.kind for term in system.exact_core.classical_terms) == (
        "d4",
        "repulsion",
    )
    assert tuple(term.kind for term in system.exact_core.interaction_terms) == (
        "es2",
        "aes2",
        "es3",
        "d4sc",
    )

    def energy(geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(geometry).energy.sum()

    tangent = torch.linspace(
        -0.1, 0.1, positions[0].numel(), dtype=positions.dtype
    ).reshape_as(positions[0])
    _, tangent_derivative = torch.func.jvp(energy, (positions[0],), (tangent,))
    forward = torch.func.jacfwd(energy)(positions[0])
    reverse = torch.func.jacrev(energy)(positions[0])
    torch.testing.assert_close(forward, reverse)
    torch.testing.assert_close(tangent_derivative, (reverse * tangent).sum())

    def batch_energy(batch: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(system.singlepoint, in_dims=(0,))(
            batch
        ).energy.sum(dim=-1)

    jacobian = torch.func.jacfwd(batch_energy)(positions)
    assert jacobian.shape == (2, 2, *positions.shape[1:])
    for out_index in range(2):
        for in_index in range(2):
            if out_index == in_index:
                expected = torch.func.grad(energy)(positions[in_index])
                torch.testing.assert_close(
                    jacobian[out_index, in_index], expected
                )
            else:
                torch.testing.assert_close(
                    jacobian[out_index, in_index],
                    torch.zeros_like(jacobian[out_index, in_index]),
                    atol=1.0e-12,
                    rtol=0.0,
                )
    assert atomic_numbers.tolist() == list(numbers)


@pytest.mark.parametrize(
    ("method", "mixer", "scp_mode"),
    [
        (method, mixer, scp_mode)
        for method in (GFN1_XTB, GFN2_XTB)
        for mixer in (labels.MIXER_LINEAR, labels.MIXER_ANDERSON)
        for scp_mode in (
            labels.SCP_MODE_CHARGE,
            labels.SCP_MODE_POTENTIAL,
            labels.SCP_MODE_FOCK,
        )
    ],
    ids=[
        f"{method}-{mixer}-{scp_mode}"
        for method in (GFN1_XTB, GFN2_XTB)
        for mixer in (labels.MIXER_LINEAR, labels.MIXER_ANDERSON)
        for scp_mode in (
            labels.SCP_MODE_CHARGE,
            labels.SCP_MODE_POTENTIAL,
            labels.SCP_MODE_FOCK,
        )
    ],
)
def test_h2_public_vmap_mixer_and_scp_modes(
    method: object, mixer: int, scp_mode: int
) -> None:
    """Supported GFN1/GFN2 mixer and SCP modes preserve serial parity."""
    _, _, system = _system(
        method, (1, 1), mixer=mixer, scp_mode=scp_mode, maxiter=1
    )
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.48]],
        ],
        dtype=torch.float64,
    )
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]
    _assert_result_tree_matches(mapped, serial)


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB])
def test_public_hessian_matches_finite_difference_of_gradients(
    method: object,
) -> None:
    """The full production Hessian agrees with independent gradient differences."""
    _, _, system = _system(method, (8, 1, 1))
    positions = _water_positions(3)[1]

    def energy(flat_geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(
            flat_geometry.reshape_as(positions)
        ).energy.sum()

    gradient = torch.func.grad(energy)
    origin = positions.flatten()
    hessian = torch.func.jacrev(gradient)(origin)
    basis = torch.eye(origin.numel(), dtype=origin.dtype)
    step = 1.0e-4
    finite_difference = torch.stack(
        [
            (gradient(origin + step * axis) - gradient(origin - step * axis))
            / (2.0 * step)
            for axis in basis
        ],
        dim=1,
    )
    torch.testing.assert_close(
        hessian, finite_difference, atol=1.0e-7, rtol=1.0e-7
    )


def test_gfn1_d3_public_third_directional_derivative_matches_finite_difference() -> (
    None
):
    """A full public GFN1 third derivative agrees with Hessian differences."""
    _, _, system = _system(GFN1_XTB, (8, 1, 1))
    positions = _water_positions(3)[1]
    dispersion = dict(system.singlepoint(positions).classical)["DispersionD3"]
    assert dispersion.abs().sum() > 0

    origin = positions.flatten()
    direction = torch.linspace(-0.4, 0.5, origin.numel(), dtype=origin.dtype)
    direction = direction / torch.linalg.vector_norm(direction)

    def directional_energy(offset: torch.Tensor) -> torch.Tensor:
        geometry = (origin + offset * direction).reshape_as(positions)
        return system.singlepoint(geometry).energy.sum()

    first = torch.func.grad(directional_energy)
    second = torch.func.grad(first)
    third = torch.func.grad(second)(origin.new_zeros(()))

    step = 1.0e-3
    second_values = [
        second(origin.new_tensor(multiplier * step))
        for multiplier in (-2.0, -1.0, 1.0, 2.0)
    ]
    finite_difference = (
        second_values[0]
        - 8.0 * second_values[1]
        + 8.0 * second_values[2]
        - second_values[3]
    ) / (12.0 * step)
    torch.testing.assert_close(
        third, finite_difference, atol=1.0e-8, rtol=1.0e-8
    )
