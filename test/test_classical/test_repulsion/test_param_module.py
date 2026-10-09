# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""Repulsion gradients through gathered ParamModule leaves."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import new_repulsion
from dxtb._src.typing import DD
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize(
    "method,name,path",
    [
        ("gfn1", "H2O", "element.H.arep"),
        ("gfn1", "H2O", "element.O.zeff"),
        ("gfn1", "H2O", "repulsion.effective.kexp"),
        ("gfn2", "H2O", "element.H.arep"),
        ("gfn2", "H2O", "element.O.zeff"),
        ("gfn2", "H2O", "repulsion.effective.kexp"),
        ("gfn2", "H2", "repulsion.effective.klight"),
    ],
)
def test_factory_parameter_gradients(method: str, name: str, path: str) -> None:
    """Gradients reach active repulsion leaves through factory setup."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    parameter_set = GFN1_XTB if method == "gfn1" else GFN2_XTB
    par = ParamModule(parameter_set, **dd)
    par.set_differentiable(*path.split("."))
    parameter = par.get(path)

    repulsion = new_repulsion(torch.unique(numbers), par, **dd)
    assert repulsion is not None
    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = repulsion.get_cache(numbers, ihelp)
    energy = repulsion.get_energy(positions, cache).sum()
    (gradient,) = torch.autograd.grad(energy, parameter)

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_mixed_position_parameter_derivative() -> None:
    """Core setup preserves mixed position/parameter differentiation."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(**dd).requires_grad_(True)
    par = ParamModule(GFN1_XTB, **dd)
    par.set_differentiable("element.H.arep")
    arep = par.get("element.H.arep")

    system = Model(par=par, config=Config.create(exclude="scf")).setup(numbers)
    energy = dict(system.singlepoint(positions).classical)["Repulsion"].sum()
    (position_gradient,) = torch.autograd.grad(
        energy, positions, create_graph=True
    )
    direction = torch.arange(
        positions.numel(), dtype=positions.dtype, device=positions.device
    ).reshape_as(positions)
    projection = (position_gradient * direction).sum()
    (mixed_gradient,) = torch.autograd.grad(projection, arep)

    assert torch.isfinite(mixed_gradient).all()
    assert torch.count_nonzero(mixed_gradient) > 0


def test_geometry_history_after_backward() -> None:
    """One setup yields history-independent repulsion energies and gradients."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions_a = samples["H2O"]["positions"].to(**dd)
    positions_b = positions_a.clone()
    positions_b[1, 0] += 0.08
    system = Model(
        par=ParamModule(GFN1_XTB, **dd),
        config=Config.create(exclude="scf"),
    ).setup(numbers)

    pos_a = positions_a.clone().requires_grad_(True)
    energy_a = dict(system.singlepoint(pos_a).classical)["Repulsion"].sum()
    (gradient_a,) = torch.autograd.grad(energy_a, pos_a)

    pos_b = positions_b.clone().requires_grad_(True)
    energy_b = dict(system.singlepoint(pos_b).classical)["Repulsion"].sum()
    (gradient_b,) = torch.autograd.grad(energy_b, pos_b)

    pos_a_again = positions_a.clone().requires_grad_(True)
    energy_a_again = dict(system.singlepoint(pos_a_again).classical)[
        "Repulsion"
    ].sum()
    (gradient_a_again,) = torch.autograd.grad(energy_a_again, pos_a_again)

    assert not torch.isclose(energy_a, energy_b)
    torch.testing.assert_close(energy_a, energy_a_again)
    torch.testing.assert_close(gradient_a, gradient_a_again)
    assert torch.isfinite(gradient_b).all()


@pytest.mark.parametrize(
    "method,name,path",
    [
        ("gfn1", "H2O", "element.H.arep"),
        ("gfn1", "H2O", "element.O.zeff"),
        ("gfn1", "H2O", "repulsion.effective.kexp"),
        ("gfn2", "H2O", "element.H.arep"),
        ("gfn2", "H2O", "element.O.zeff"),
        ("gfn2", "H2O", "repulsion.effective.kexp"),
        ("gfn2", "H2", "repulsion.effective.klight"),
    ],
)
def test_system_setup_parameter_gradients(
    method: str, name: str, path: str
) -> None:
    """Gradients through System RepulsionSetup reach original parameters."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    parameter_set = GFN1_XTB if method == "gfn1" else GFN2_XTB
    par = ParamModule(parameter_set, **dd)
    par.set_differentiable(*path.split("."))
    parameter = par.get(path)
    model = Model(par=par, config=Config.create(exclude="scf"))

    def loss() -> torch.Tensor:
        system = model.setup(numbers)
        result = system.singlepoint(positions)
        return dict(result.classical)["Repulsion"].sum()

    (gradient1,) = torch.autograd.grad(loss(), parameter)
    (gradient2,) = torch.autograd.grad(loss(), parameter)
    assert torch.isfinite(gradient1).all()
    assert torch.isfinite(gradient2).all()
    assert torch.count_nonzero(gradient1) > 0
    torch.testing.assert_close(gradient1, gradient2)


def test_removed_analytical_factory_option_is_rejected() -> None:
    """The deleted analytical-gradient option is not silently ignored."""
    numbers = samples["H2"]["numbers"].to(DEVICE)
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.double)
    with pytest.raises(TypeError, match="with_analytical_gradient"):
        new_repulsion(
            torch.unique(numbers),
            par,
            with_analytical_gradient=True,  # type: ignore[call-arg]
        )
