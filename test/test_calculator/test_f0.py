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
"""Early functional parameter-training invariant (F0)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from dxtb import GFN1_XTB, ParamModule
from dxtb._src.calculators.model import Model
from dxtb.config import Config
from test.test_baseline.molecules import get_system

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


class _TrainingEnergy(nn.Module):
    """Evaluate the full GFN1 core from a fresh parameterized System."""

    def __init__(self, par: ParamModule, numbers: torch.Tensor) -> None:
        super().__init__()
        self.par = par
        self.register_buffer("numbers", numbers.clone())

    def forward(
        self, positions: torch.Tensor, charge: torch.Tensor, spin: int | None
    ) -> torch.Tensor:
        model = Model(
            par=self.par,
            config=Config.create(
                method="gfn1-xtb",
                int_driver="pytorch",
                maxiter=150,
                f_atol=1.0e-10,
                x_atol=1.0e-10,
            ),
        )
        system = model.setup(self.numbers)
        return system.singlepoint(
            positions, chrg=charge, spin=spin
        ).energy.sum()


def _parameter_names(module: nn.Module) -> dict[str, str]:
    """Find stable ParamModule paths independent of the wrapper prefix."""
    suffixes = (
        "parameter_tree.element.O.arep.param",
        "parameter_tree.element.O.gam.param",
        "parameter_tree.repulsion.effective.kexp.param",
    )
    found = {}
    for suffix in suffixes:
        matches = [
            name
            for name, _ in module.named_parameters()
            if name.endswith(suffix)
        ]
        assert (
            len(matches) == 1
        ), f"Expected one ParamModule leaf ending in {suffix}"
        found[suffix] = matches[0]
    return found


def _energy(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    charge: torch.Tensor,
    spin: int | None,
) -> torch.Tensor:
    return torch.func.functional_call(
        module, dict(state), (positions, charge, spin), strict=False
    )


def test_functional_parameter_training_invariant() -> None:
    """Real ParamModule values train through fresh Systems without mutation."""
    system_data = get_system("H2O")
    numbers = system_data.numbers.clone().to(device=DEVICE)
    positions = system_data.positions.to(**DD).clone()
    charge = system_data.charge.to(**DD).clone()
    spin = system_data.spin

    par = ParamModule(GFN1_XTB, **DD)
    module = _TrainingEnergy(par, numbers)
    names = _parameter_names(module)
    originals = dict(module.named_parameters())
    selected = {
        name: originals[name].detach().clone().requires_grad_(True)
        for name in names.values()
    }
    assert all(value.ndim == 0 for value in selected.values())
    unselected_name = next(
        name
        for name in originals
        if name.endswith("parameter_tree.element.H.gam.param")
    )
    snapshots = {
        name: originals[name].detach().clone()
        for name in (*names.values(), unselected_name)
    }
    assert all(
        selected[name].data_ptr() != originals[name].data_ptr()
        for name in selected
    )

    energy_of = lambda state: _energy(module, state, positions, charge, spin)
    initial_energy = energy_of(selected)
    original_energy = module(positions, charge, spin)
    torch.testing.assert_close(initial_energy, original_energy)

    energy_grad = torch.func.grad(energy_of)(selected)
    assert all(torch.isfinite(value).all() for value in energy_grad.values())
    assert all(torch.count_nonzero(value) > 0 for value in energy_grad.values())

    for name, value in selected.items():
        step = 1.0e-5 * torch.maximum(
            torch.ones_like(value), value.detach().abs()
        )
        plus = dict(selected)
        minus = dict(selected)
        plus[name] = value.detach() + step
        minus[name] = value.detach() - step
        finite_difference = (energy_of(plus) - energy_of(minus)) / (2 * step)
        torch.testing.assert_close(
            energy_grad[name], finite_difference, atol=1.0e-6, rtol=1.0e-5
        )

    target = initial_energy.detach() + initial_energy.new_tensor(1.0e-3)

    def loss_of(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (energy_of(state) - target).square()

    initial_loss = loss_of(selected)
    loss_gradient = torch.func.grad(loss_of)(selected)
    assert torch.isfinite(initial_loss)
    assert all(torch.isfinite(value).all() for value in loss_gradient.values())
    assert all(
        torch.count_nonzero(value) > 0 for value in loss_gradient.values()
    )

    learning_rate = 5.0
    state = selected
    losses = [initial_loss]
    for _ in range(4):
        gradient = torch.func.grad(loss_of)(state)
        state = {
            name: value - learning_rate * gradient[name]
            for name, value in state.items()
        }
        losses.append(loss_of(state))

    final_energy = energy_of(state)
    assert torch.isfinite(final_energy)
    assert all(torch.isfinite(loss) for loss in losses)
    assert all(later < earlier for earlier, later in zip(losses, losses[1:]))
    assert losses[-1] < losses[0]
    assert torch.abs(final_energy - target) < torch.abs(initial_energy - target)
    assert any(not torch.equal(state[name], selected[name]) for name in state)

    for name, snapshot in snapshots.items():
        assert torch.equal(originals[name].detach(), snapshot)
        assert originals[name].grad is None
    assert torch.equal(module(positions, charge, spin), original_energy)
    assert not torch.equal(final_energy, original_energy)

    direction = torch.arange(positions.numel(), **DD).reshape_as(positions)
    direction = direction / torch.linalg.vector_norm(direction)
    force_position = positions.clone().requires_grad_(True)
    force_energy = _energy(module, selected, force_position, charge, spin)
    (position_gradient,) = torch.autograd.grad(
        force_energy, force_position, create_graph=True
    )
    force_projection = (position_gradient * direction).sum()
    (mixed,) = torch.autograd.grad(
        force_projection,
        selected[names["parameter_tree.element.O.arep.param"]],
    )
    assert torch.isfinite(mixed)
    assert torch.count_nonzero(mixed) > 0
