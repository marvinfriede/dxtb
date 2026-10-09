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
"""Explicit ES3 setup and per-call data tests."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.interactions.list import InteractionList
from dxtb._src.components.interactions.coulomb.thirdorder import (
    ES3,
    ES3Cache,
    ES3Setup,
    setup_es3,
)
from dxtb._src.param import Param
from dxtb._src.components.interactions.container import Charges

from ..molecules import mols
from ..conftest import DEVICE


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = mols["H2O"]["numbers"].clone().to(device=DEVICE)
    positions = mols["H2O"]["positions"].to(device=DEVICE, dtype=torch.float64)
    return numbers, positions


def _energy(system, charges: Charges) -> torch.Tensor:
    assert system.es3_setup is not None
    assert system.es3_interaction is not None
    cache = ES3Cache(
        system.es3_setup.hubbard_derivs,
        shell_resolved=system.es3_setup.shell_resolved,
    )
    orbital_charges = Charges(
        mono=system.ihelp.spread_atom_to_orbital(charges.mono)
    )
    return system.es3_interaction.get_energy(
        cache, orbital_charges, system.ihelp
    ).sum()


def test_es3_setup_and_get_cache_are_call_local() -> None:
    """ES3 stores resolved setup only and every direct cache is fresh."""
    numbers, _ = _inputs()
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.float64)
    system = Model(par=par).setup(numbers)

    assert isinstance(system.es3_setup, ES3Setup)
    assert system.es3_interaction is not None
    assert {field.name for field in fields(system.es3_setup)} == {
        "hubbard_derivs",
        "shell_resolved",
    }
    assert isinstance(system.es3_setup.hubbard_derivs, torch.Tensor)
    assert system.es3_setup.hubbard_derivs.shape == numbers.shape
    forbidden = (Param, ParamModule, Model, ES3, ES3Cache, InteractionList)
    for field in fields(system.es3_setup):
        assert not isinstance(getattr(system.es3_setup, field.name), forbidden)

    interaction = system.es3_interaction
    first = interaction.get_cache(numbers=numbers, ihelp=system.ihelp)
    second = interaction.get_cache(numbers=numbers, ihelp=system.ihelp)
    assert first is not second
    assert interaction.cache is None
    assert interaction._cachevars is None
    assert interaction._cachegrad is None
    torch.testing.assert_close(first.hd, system.es3_setup.hubbard_derivs)

    interaction.cache_disable()
    third = interaction.get_cache(numbers=numbers, ihelp=system.ihelp)
    assert third is not second
    torch.testing.assert_close(third.hd, second.hd)


def test_es3_setup_parameter_gradient_rebuilds_system() -> None:
    """Fresh System setup preserves the original gam3 parameter graph."""
    numbers, _ = _inputs()
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.float64)
    parameter = dict(par.named_parameters())[
        "parameter_tree.element.O.gam3.param"
    ]
    parameter.requires_grad_(True)
    model = Model(par=par)
    orbital_charges = torch.linspace(
        -0.2, 0.3, numbers.numel(), device=DEVICE, dtype=torch.float64
    )
    charges = Charges(mono=orbital_charges)

    system1 = model.setup(numbers)
    loss1 = _energy(system1, charges)
    (grad1,) = torch.autograd.grad(loss1, parameter)

    system2 = model.setup(numbers)
    loss2 = _energy(system2, charges)
    (grad2,) = torch.autograd.grad(loss2, parameter)

    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()
    assert torch.count_nonzero(grad1) > 0
    torch.testing.assert_close(grad1, grad2)


@pytest.mark.parametrize(
    "parameter_name",
    [
        "parameter_tree.thirdorder.shell.s.param",
        "parameter_tree.thirdorder.shell.p.param",
    ],
)
def test_gfn2_shell_scale_parameter_gradient_rebuilds_system(
    parameter_name: str,
) -> None:
    """Active GFN2 shell scaling stays connected through fresh setup."""
    numbers, positions = _inputs()
    par = ParamModule(GFN2_XTB, device=DEVICE, dtype=positions.dtype)
    parameter = dict(par.named_parameters())[parameter_name]
    parameter.requires_grad_(True)
    model = Model(par=par)

    def loss() -> torch.Tensor:
        system = model.setup(numbers)
        assert system.es3_setup is not None
        assert system.es3_setup.shell_resolved
        return system.singlepoint(positions).scf.sum()

    (grad1,) = torch.autograd.grad(loss(), parameter)
    (grad2,) = torch.autograd.grad(loss(), parameter)

    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()
    assert torch.count_nonzero(grad1) > 0
    torch.testing.assert_close(grad1, grad2)


def test_setup_es3_atom_and_shell_resolution() -> None:
    """ES3 setup resolves Hubbard derivatives before per-call evaluation."""
    numbers, _ = _inputs()
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.float64)
    system = Model(par=par).setup(numbers)
    interaction = system.es3_interaction
    assert interaction is not None

    atom_setup = setup_es3(interaction.hubbard_derivs, system.ihelp)
    torch.testing.assert_close(
        atom_setup.hubbard_derivs,
        system.ihelp.spread_uspecies_to_atom(interaction.hubbard_derivs),
    )
    assert not atom_setup.shell_resolved

    scale = torch.tensor([1.0, 1.1, 1.2], device=DEVICE, dtype=torch.float64)
    shell_setup = setup_es3(
        interaction.hubbard_derivs, system.ihelp, shell_scale=scale
    )
    assert shell_setup.shell_resolved
    assert shell_setup.hubbard_derivs.shape == (
        system.ihelp.shells_to_atom.shape[-1],
    )
    assert not torch.allclose(
        shell_setup.hubbard_derivs,
        system.ihelp.spread_uspecies_to_shell(interaction.hubbard_derivs),
    )


def test_system_core_bypasses_es3_get_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-system SCF creates ES3 call data directly from System setup."""
    numbers, positions = _inputs()
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=positions.dtype)
    system = Model(par=par).setup(numbers)
    assert system.es3_setup is not None

    def fail_cache(*_: object, **__: object) -> ES3Cache:
        raise AssertionError("single-system core called ES3.get_cache")

    monkeypatch.setattr(ES3, "get_cache", fail_cache)
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()
