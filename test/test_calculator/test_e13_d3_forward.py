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
"""E1.3 end-to-end D3 transform regressions for composite AtomicC6."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import is_batched

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals.dispersion.d3 import (
    DispersionD3,
    _D3ParameterValue,
    core_dispersion_d3_setup,
    dispersion_d3_energy,
)
from dxtb.config import Config


def _system(
    numbers: tuple[int, ...], *, differentiable_s9: bool = False
) -> tuple[Model, torch.Tensor, object]:
    """Build a production GFN1/PyTorch System and Model owner."""
    params = ParamModule(GFN1_XTB, dtype=torch.float64)
    if differentiable_s9:
        params.set_differentiable("dispersion.d3.s9")
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        mixer=labels.MIXER_LINEAR,
        scp_mode=labels.SCP_MODE_CHARGE,
        maxiter=2,
        fermi_etemp=0.0,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    model = Model(par=params, config=config)
    numbers_tensor = torch.tensor(numbers, dtype=torch.int64)
    return model, numbers_tensor, model.setup(numbers_tensor)


def _positions(numbers: tuple[int, ...]) -> torch.Tensor:
    if numbers == (1, 1):
        return torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
        )
    if numbers == (8, 1, 1):
        return torch.tensor(
            [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]],
            dtype=torch.float64,
        )
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("numbers", [(1, 1), (8, 1, 1), (6, 1, 1, 1, 1)])
def test_public_system_jvp_and_jacfwd_match_reverse(
    numbers: tuple[int, ...],
) -> None:
    """Full GFN1 with D3 supports public position JVP/Jacobian transforms."""
    model, atomic_numbers, system = _system(numbers)
    assert model is not None
    positions = _positions(numbers)

    def energy(geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(geometry).energy.sum()

    tangent = torch.linspace(
        -0.08, 0.11, positions.numel(), dtype=positions.dtype
    ).reshape_as(positions)
    _, jvp = torch.func.jvp(energy, (positions,), (tangent,))
    reverse = torch.func.grad(energy)(positions)
    forward = torch.func.jacfwd(energy)(positions)

    assert atomic_numbers.shape == (len(numbers),)
    assert torch.isfinite(forward).all()
    torch.testing.assert_close(forward, reverse)
    torch.testing.assert_close(jvp, (reverse * tangent).sum())


def test_public_system_vmap_and_mapped_gradients_when_batch_differs_from_atoms() -> (
    None
):
    """Fixed numbers remain aligned when B and nat are different."""
    _, _, system = _system((1, 1))
    positions = torch.stack(
        [
            _positions((1, 1)),
            _positions((1, 1))
            + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
            _positions((1, 1))
            + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -0.08]]),
        ]
    )
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(p) for p in positions]
    torch.testing.assert_close(
        mapped.energy,
        torch.stack([result.energy for result in serial]),
    )

    def energy(geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(geometry).energy.sum()

    mapped_gradient = torch.func.vmap(torch.func.grad(energy))(positions)
    serial_gradient = torch.stack(
        [torch.func.grad(energy)(p) for p in positions]
    )
    torch.testing.assert_close(mapped_gradient, serial_gradient)


def test_public_system_third_reverse_derivative_is_finite() -> None:
    """The production ATM term supports third reverse derivatives on H2O."""
    _, _, system = _system((8, 1, 1), differentiable_s9=True)
    positions = _positions((8, 1, 1))

    def energy(flat: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(flat.reshape_as(positions)).energy.sum()

    third = torch.func.jacrev(torch.func.jacrev(torch.func.grad(energy)))(
        positions.flatten()
    )
    assert third.shape == (9, 9, 9)
    assert torch.isfinite(third).all()


def test_public_system_differentiable_zero_s9_has_three_body_derivative() -> (
    None
):
    """The actual GFN1 D3 path keeps the ATM derivative at s9 equal to zero."""
    model, _, system = _system((8, 1, 1), differentiable_s9=True)
    positions = _positions((8, 1, 1)).requires_grad_()
    s9 = model.par.get("dispersion.d3.s9", unwrapped=False).param
    assert s9 is not None and s9.requires_grad
    torch.testing.assert_close(s9.detach(), torch.zeros_like(s9))
    energy = system.singlepoint(positions).energy.sum()
    gradient = torch.autograd.grad(energy, s9, create_graph=True)[0]
    assert torch.isfinite(gradient)
    assert gradient.abs() > 1.0e-12

    mixed = torch.autograd.grad(gradient, positions)[0]
    assert mixed.shape == positions.shape
    assert torch.isfinite(mixed).all()


@pytest.mark.parametrize("name", ["s6", "s8", "s9", "a1", "a2"])
def test_mapped_public_d3_parameter_gradient_matches_serial(name: str) -> None:
    """Each active GFN1 D3 parameter keeps its Model-owned gradient."""
    model, _, system = _system((8, 1, 1), differentiable_s9=(name == "s9"))
    if name != "s9":
        model.par.set_differentiable(f"dispersion.d3.{name}")
        system = model.setup(torch.tensor([8, 1, 1], dtype=torch.int64))
    parameter = model.par.get(f"dispersion.d3.{name}", unwrapped=False).param
    assert parameter is not None and parameter.requires_grad
    positions = torch.stack(
        (
            _positions((8, 1, 1)),
            _positions((8, 1, 1))
            + torch.tensor(
                [[0.0, 0.0, 0.0], [0.015, 0.0, 0.0], [-0.015, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
    )

    mapped_energy = torch.func.vmap(system.singlepoint, in_dims=(0,))(
        positions
    ).energy.sum()
    mapped_gradient = torch.autograd.grad(
        mapped_energy, parameter, retain_graph=True
    )[0]
    serial_energy = torch.stack(
        [system.singlepoint(p).energy.sum() for p in positions]
    ).sum()
    serial_gradient = torch.autograd.grad(serial_energy, parameter)[0]
    torch.testing.assert_close(mapped_gradient, serial_gradient)
    assert torch.isfinite(mapped_gradient)
    assert mapped_gradient.abs() > 0


def test_public_system_vmap_jacfwd_has_one_conformer_axis() -> None:
    """Mapped D3 energy derivatives agree with serial fixed-System calls."""
    _, _, system = _system((8, 1, 1))
    base = _positions((8, 1, 1))
    positions = torch.stack(
        (
            base,
            base
            + torch.tensor(
                [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]]
            ),
        )
    )

    def mapped_energy(batch: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(system.singlepoint, in_dims=(0,))(
            batch
        ).energy.sum(dim=-1)

    jacobian = torch.func.jacfwd(mapped_energy)(positions)
    assert jacobian.shape == (2, 2, 3, 3)
    assert torch.isfinite(jacobian).all()
    for output in range(2):
        for input_ in range(2):
            if output != input_:
                torch.testing.assert_close(
                    jacobian[output, input_],
                    torch.zeros_like(jacobian[output, input_]),
                    atol=1.0e-12,
                    rtol=0.0,
                )


def test_official_atomic_c6_supports_nested_transforms() -> None:
    """The released composite C6 path handles padding and nested transforms."""
    import tad_dftd3 as d3
    from tad_mctc.data import radii

    numbers = torch.tensor([8, 1, 1, 0], dtype=torch.int64)
    base = _positions((8, 1, 1))
    padded = torch.cat((base, torch.zeros((1, 3), dtype=base.dtype)), dim=0)
    positions = torch.stack((padded, padded + 0.03, padded - 0.02))
    reference = d3.reference.Reference(dtype=positions.dtype)
    rcov = radii.COV_D3(dtype=positions.dtype)[numbers]

    def atomic_c6(geometry: torch.Tensor) -> torch.Tensor:
        aligned_numbers = numbers + torch.zeros_like(
            geometry[..., 0], dtype=numbers.dtype
        )
        cn = d3.ncoord.coordination_number(
            aligned_numbers,
            geometry,
            counting_function=d3.ncoord.exp_count,
            rcov=rcov,
            cutoff=positions.new_tensor(25.0),
        )
        weights = d3.model.weight_references(aligned_numbers, cn, reference)
        return d3.model.atomic_c6(aligned_numbers, weights, reference)

    mapped = torch.func.vmap(atomic_c6)(positions)
    assert mapped.shape == (3, 4, 4)
    torch.testing.assert_close(
        mapped[:, 3, :], torch.zeros_like(mapped[:, 3, :])
    )
    torch.testing.assert_close(
        mapped[:, :, 3], torch.zeros_like(mapped[:, :, 3])
    )

    def scalar(geometry: torch.Tensor) -> torch.Tensor:
        return atomic_c6(geometry).sum()

    jacobian = torch.func.jacfwd(scalar)(padded)
    hessian = torch.func.jacrev(torch.func.jacrev(scalar))(padded)
    assert jacobian.shape == padded.shape
    assert hessian.shape == (4, 3, 4, 3)
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(hessian).all()

    nested = torch.func.vmap(
        torch.func.jacrev(
            lambda geometry: geometry * float(is_batched(geometry))
        )
    )(positions)
    eye = torch.eye(12, dtype=positions.dtype).reshape(4, 3, 4, 3)
    torch.testing.assert_close(nested, eye.expand(3, -1, -1, -1, -1))


def test_zero_trainable_s9_derivative_matches_finite_difference() -> None:
    """Both legacy and System-owned D3 routes retain ATM dE/ds9 at zero."""
    numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
    positions = _positions((8, 1, 1))
    dd = {"dtype": positions.dtype, "device": positions.device}
    params = {
        "s6": torch.tensor(1.0, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "a1": torch.tensor(0.49484001, **dd),
        "a2": torch.tensor(5.73083694, **dd),
        "s9": torch.tensor(0.0, requires_grad=True, **dd),
    }
    dispersion = DispersionD3(numbers, params, **dd)
    cache = dispersion.get_cache(numbers)
    core = core_dispersion_d3_setup(cache, numbers, numbers, params)

    legacy = dispersion.get_energy(positions, cache).sum()
    exact = dispersion_d3_energy(core, positions).sum()
    legacy_gradient = torch.autograd.grad(
        legacy, params["s9"], create_graph=True, retain_graph=True
    )[0]
    exact_gradient = torch.autograd.grad(
        exact, params["s9"], create_graph=True
    )[0]
    assert legacy_gradient.abs() > 1.0e-12
    torch.testing.assert_close(legacy_gradient, exact_gradient)

    step = 1.0e-4
    with torch.no_grad():
        params["s9"].fill_(step)
    plus = dispersion.get_energy(positions, cache).sum()
    with torch.no_grad():
        params["s9"].fill_(-step)
    minus = dispersion.get_energy(positions, cache).sum()
    with torch.no_grad():
        params["s9"].zero_()
    finite_difference = (plus - minus) / (2.0 * step)
    torch.testing.assert_close(
        legacy_gradient, finite_difference, atol=1.0e-10, rtol=1.0e-6
    )

    def exact_energy(s9: torch.Tensor) -> torch.Tensor:
        entries = tuple(
            (
                entry.replace(value=_D3ParameterValue.encode(s9))
                if entry.name == "s9"
                else entry
            )
            for entry in core.parameter_entries
        )
        return dispersion_d3_energy(
            core.replace(parameter_entries=entries), positions
        ).sum()

    forward_s9 = torch.func.jacfwd(exact_energy)(
        torch.tensor(0.0, dtype=positions.dtype)
    )
    torch.testing.assert_close(
        forward_s9, exact_gradient, atol=1.0e-10, rtol=1.0e-6
    )
