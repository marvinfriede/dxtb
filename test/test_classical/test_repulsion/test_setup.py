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
"""Numbers-only Repulsion setup and System ownership tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    Repulsion,
    RepulsionSetup,
    new_repulsion,
)
from dxtb._src.param import Param
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_system_owns_frozen_repulsion_setup(method) -> None:
    """System setup contains resolved tensors but no component or model."""
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    system = Model(
        par=ParamModule(method, device=DEVICE, dtype=torch.double)
    ).setup(numbers)

    assert system.repulsion_classical is not None
    assert isinstance(system.repulsion_setup, RepulsionSetup)
    assert system.classical_cache["Repulsion"] is system.repulsion_setup
    assert {field.name for field in fields(system.repulsion_setup)} == {
        "arep",
        "zeff",
        "kexp",
        "mask",
        "cutoff",
    }
    forbidden = (Param, ParamModule, Model, Repulsion)
    assert all(
        not isinstance(getattr(system.repulsion_setup, field.name), forbidden)
        for field in fields(system.repulsion_setup)
    )
    with pytest.raises(FrozenInstanceError):
        system.repulsion_setup.cutoff = torch.tensor(1.0)  # type: ignore[misc]
    first = system.repulsion_classical.get_cache(numbers, system.ihelp)
    second = system.repulsion_classical.get_cache(numbers, system.ihelp)
    assert first is not second
    assert system.repulsion_classical.cache is None
    assert system.repulsion_classical._cachevars is None
    system.repulsion_classical.cache_disable()
    third = system.repulsion_classical.get_cache(numbers, system.ihelp)
    assert third is not second


def test_core_repulsion_ignores_legacy_object_parameter_mutation() -> None:
    """The exact Repulsion object's mutable fields do not control core output."""
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(
        device=DEVICE, dtype=torch.double
    )
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.double)
    model = Model(par=par, config=Config.create(exclude="scf"))
    system = model.setup(numbers)

    before = system.singlepoint(positions)
    before_repulsion = dict(before.classical)["Repulsion"]
    assert system.repulsion_classical is not None
    assert system.repulsion_setup is not None
    setup_cutoff = system.repulsion_setup.cutoff.clone()
    with torch.no_grad():
        system.repulsion_classical.cutoff.fill_(0.0)
    after = system.singlepoint(positions.clone())
    after_repulsion = dict(after.classical)["Repulsion"]
    torch.testing.assert_close(after_repulsion, before_repulsion)
    torch.testing.assert_close(system.repulsion_setup.cutoff, setup_cutoff)

    parameter = dict(par.named_parameters())[
        "parameter_tree.element.H.arep.param"
    ]
    with torch.no_grad():
        parameter.mul_(1.3)
    fresh = model.setup(numbers)
    fresh_result = fresh.singlepoint(positions)
    fresh_repulsion = dict(fresh_result.classical)["Repulsion"]
    assert not torch.allclose(fresh_repulsion, before_repulsion)


def test_exact_repulsion_update_and_reset_are_rejected() -> None:
    """Exact migrated Repulsion cannot mutate independently of its setup."""
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    calc = Calculator(
        numbers,
        GFN1_XTB,
        dtype=torch.double,
        opts={"verbosity": 0},
    )
    assert calc.system.repulsion_classical is not None
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.system.repulsion_classical.update(cutoff=torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.system.repulsion_classical.reset()
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.classicals.update("Repulsion", cutoff=torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.classicals.reset("Repulsion")


def test_calculator_reset_preserves_repulsion_setup() -> None:
    """Calculator.reset leaves exact Repulsion setup and parameter graph alone."""
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(
        device=DEVICE, dtype=torch.double
    )
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.double)
    parameter = dict(par.named_parameters())[
        "parameter_tree.element.H.arep.param"
    ]
    parameter.requires_grad_(True)
    calc = Calculator(numbers, par, dtype=torch.double, opts={"verbosity": 0})
    setup = calc.system.repulsion_setup
    first = calc.singlepoint(positions)
    calc.reset()
    second = calc.singlepoint(positions.clone())

    assert calc.system.repulsion_setup is setup
    torch.testing.assert_close(first.energy, second.energy)
    assert setup is not None
    (gradient,) = torch.autograd.grad(setup.arep.sum(), parameter)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_custom_exact_repulsion_replaces_excluded_builtin() -> None:
    """An exact user Repulsion supplies the setup when the built-in is excluded."""
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(
        device=DEVICE, dtype=torch.double
    )
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.double)
    custom = new_repulsion(
        torch.unique(numbers),
        par,
        cutoff=1.0,
        device=DEVICE,
        dtype=torch.double,
    )
    assert custom is not None
    system = Model(
        par=par,
        config=Config.create(exclude="rep"),
        classical=(custom,),
    ).setup(numbers)

    assert system.repulsion_classical is custom
    assert system.repulsion_setup is not None
    torch.testing.assert_close(
        system.repulsion_setup.cutoff,
        torch.tensor(1.0, device=DEVICE, dtype=torch.double),
    )
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()
    assert torch.count_nonzero(dict(result.classical)["Repulsion"]) == 0


def test_repulsion_subclass_uses_legacy_path() -> None:
    """Repulsion subclasses keep their legacy setup and energy hooks."""

    class RepulsionExtension(Repulsion):
        __slots__ = (
            "get_cache_calls",
            "get_energy_calls",
            "reset_calls",
            "extension_value",
            "original_arep",
        )

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.get_cache_calls = 0
            self.get_energy_calls = 0
            self.reset_calls = 0
            self.extension_value = torch.tensor(1.0)
            self.original_arep = self.arep.clone()

        def get_cache(self, *args, **kwargs):
            self.get_cache_calls += 1
            return super().get_cache(*args, **kwargs)

        def get_energy(self, *args, **kwargs):
            self.get_energy_calls += 1
            return self.extension_value * super().get_energy(*args, **kwargs)

        def update(self, **kwargs) -> None:
            arep = kwargs.pop("arep", None)
            if arep is not None:
                self.arep = arep
                self.cache_invalidate()
            super().update(**kwargs)

        def reset(self) -> None:
            self.reset_calls += 1
            self.arep = self.original_arep.clone()
            self.cache_invalidate()
            super().reset()

    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(
        device=DEVICE, dtype=torch.double
    )
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.double)
    base = new_repulsion(torch.unique(numbers), par)
    assert base is not None
    custom = RepulsionExtension(
        base.arep,
        base.zeff,
        base.kexp,
        klight=base.klight,
        cutoff=base.cutoff,
        en=base.en,
        enscale=base.enscale,
        dtype=torch.double,
    )
    custom.label = "Repulsion"
    calc = Calculator(
        numbers,
        par,
        opts={"verbosity": 0, "exclude": "rep"},
        classical=(custom,),
        device=DEVICE,
        dtype=torch.double,
    )

    assert calc.system.repulsion_setup is None
    result = calc.singlepoint(positions)
    assert custom.get_cache_calls == 2  # Model setup + this evaluation
    assert custom.get_energy_calls == 1
    assert torch.isfinite(result.energy).all()
    initial_repulsion = dict(result.classical)["Repulsion"]
    calc.classicals.update(
        "Repulsion",
        arep=2.0 * custom.arep,
        extension_value=torch.tensor(2.0),
    )
    changed = calc.singlepoint(positions.clone())
    assert custom.get_cache_calls == 3
    assert not torch.allclose(
        dict(changed.classical)["Repulsion"], initial_repulsion
    )
    calc.classicals.reset("Repulsion")
    reset = calc.singlepoint(positions.clone())
    torch.testing.assert_close(
        dict(reset.classical)["Repulsion"], 2.0 * initial_repulsion
    )
    calc.reset()
    assert custom.extension_value.item() == 2.0
    assert custom.reset_calls == 2


def test_duplicate_repulsion_labels_are_rejected() -> None:
    """Built-in and additional Repulsion labels cannot overwrite setup data."""
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.double)
    custom = new_repulsion(torch.unique(numbers), par)
    assert custom is not None
    with pytest.raises(ValueError, match="Duplicate repulsion labels"):
        Model(par=par, classical=(custom,)).setup(numbers)

    class RepulsionExtension(Repulsion):
        pass

    extension = RepulsionExtension(
        custom.arep,
        custom.zeff,
        custom.kexp,
        cutoff=custom.cutoff,
        dtype=torch.double,
    )
    extension.label = "Repulsion"
    with pytest.raises(ValueError, match="Duplicate repulsion labels"):
        Model(par=par, classical=(extension,)).setup(numbers)
