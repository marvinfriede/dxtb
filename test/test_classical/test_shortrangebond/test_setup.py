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
"""Tests for explicit short-range bond setup and evaluation."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN0_XTB, Calculator, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    LABEL_SRB,
    ShortRangeBond,
    ShortRangeBondSetup,
    new_srb,
    short_range_bond_energy,
)
from dxtb._src.typing import DD, Tensor
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _system(par: ParamModule | None = None) -> tuple[Model, Tensor, Tensor]:
    numbers = samples["NO2"]["numbers"].to(DEVICE)
    model = Model(
        par=ParamModule(GFN0_XTB, **DD64) if par is None else par,
        config=Config.create(exclude="scf"),
    )
    return model, numbers, samples["NO2"]["positions"].to(**DD64)


def _srb_energy(result: object) -> Tensor:
    classical = getattr(result, "classical")
    return dict(classical)[LABEL_SRB].sum()


def test_setup_is_frozen_and_contains_only_resolved_values() -> None:
    """System SRB setup has explicit values and no legacy component state."""
    model, numbers, _ = _system()
    system = model.setup(numbers)
    setup = system.srb_setup

    assert isinstance(setup, ShortRangeBondSetup)
    assert set(field.name for field in fields(setup)) == {
        "numbers",
        "r0",
        "cnfak",
        "en",
        "pauling",
        "rcov",
        "counting_function",
        "shift",
        "prefactor",
        "steepness",
        "enscale",
        "enpoly",
        "pair_cutoff2",
        "cn_cutoff",
        "cn_max",
        "cn_kcn",
    }
    for value in vars(setup).values():
        assert not isinstance(value, (ParamModule, Model, ShortRangeBond))
    assert setup.r0.shape == numbers.shape
    assert setup.cnfak.shape == numbers.shape
    assert setup.en.shape == numbers.shape
    with pytest.raises(FrozenInstanceError):
        setup.prefactor = torch.ones_like(setup.prefactor)  # type: ignore[misc]

    source = inspect.getsource(short_range_bond_energy)
    assert "self." not in source
    assert "ShortRangeBondCache" not in source


@pytest.mark.parametrize(
    "path",
    [
        "element.N.srb_r0",
        "element.N.srb_cnfak",
        "element.N.srb_en",
        "short_range.srb.shift",
        "short_range.srb.prefactor",
        "short_range.srb.steepness",
        "short_range.srb.enscale",
        "short_range.srb.enpoly",
    ],
)
def test_param_module_gradients(path: str) -> None:
    """SRB setup preserves gradients to active GFN0 parameter leaves."""
    par = ParamModule(GFN0_XTB, **DD64)
    par.set_differentiable(*path.split("."))
    parameter = par.get(path)
    model, numbers, positions = _system(par)
    system = model.setup(numbers)
    result = system.singlepoint(positions)
    (gradient,) = torch.autograd.grad(_srb_energy(result), parameter)

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_fresh_setup_repeats_parameter_gradient() -> None:
    """Rebuilding SRB setup from current leaves repeats its gradient."""
    par = ParamModule(GFN0_XTB, **DD64)
    par.set_differentiable("short_range.srb.prefactor")
    parameter = par.get("short_range.srb.prefactor")
    model, numbers, positions = _system(par)

    system1 = model.setup(numbers)
    grad1 = torch.autograd.grad(
        _srb_energy(system1.singlepoint(positions)), parameter
    )[0]
    system2 = model.setup(numbers)
    grad2 = torch.autograd.grad(
        _srb_energy(system2.singlepoint(positions)), parameter
    )[0]

    torch.testing.assert_close(grad1, grad2)
    assert torch.isfinite(grad1).all()


def test_mixed_position_parameter_derivative() -> None:
    """The setup path composes geometry and SRB parameter derivatives."""
    par = ParamModule(GFN0_XTB, **DD64)
    par.set_differentiable("short_range.srb.prefactor")
    parameter = par.get("short_range.srb.prefactor")
    model, numbers, positions = _system(par)
    positions = positions.clone().requires_grad_(True)
    system = model.setup(numbers)
    energy = _srb_energy(system.singlepoint(positions))
    (gradient,) = torch.autograd.grad(energy, positions, create_graph=True)
    direction = torch.arange(positions.numel(), **DD64).reshape_as(positions)
    projection = (gradient * direction).sum()
    (mixed,) = torch.autograd.grad(projection, parameter)

    assert torch.isfinite(mixed).all()
    assert torch.count_nonzero(mixed) > 0


def test_srb_transform_suite() -> None:
    """SRB setup supports reverse, forward, and mapped geometry transforms."""
    from torch.func import jacfwd, jvp, vmap

    model, numbers, positions = _system()
    setup = model.setup(numbers).srb_setup
    assert setup is not None
    moved = positions.clone()
    moved[1, 1] += 0.04
    conformers = torch.stack((positions, moved))

    def scalar(pos: Tensor) -> Tensor:
        return short_range_bond_energy(setup, pos).sum()

    leaf = positions.clone().requires_grad_(True)
    value = scalar(leaf)
    (first,) = torch.autograd.grad(value, leaf, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), leaf, create_graph=True)
    (third,) = torch.autograd.grad(second.sum(), leaf)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.isfinite(third).all()

    direction = torch.arange(positions.numel(), **DD64).reshape_as(positions)
    _, tangent = jvp(scalar, (positions,), (direction,))
    reverse = torch.autograd.grad(scalar(leaf), leaf)[0]
    torch.testing.assert_close(tangent, (reverse * direction).sum())
    torch.testing.assert_close(jacfwd(scalar)(positions), reverse)

    mapped = vmap(lambda pos: short_range_bond_energy(setup, pos))(conformers)
    looped = torch.stack(
        [short_range_bond_energy(setup, pos) for pos in conformers]
    )
    torch.testing.assert_close(mapped, looped)
    assert not torch.isclose(mapped[0].sum(), mapped[1].sum())


def test_padded_batch_energy_and_gradient_are_finite() -> None:
    """Padded legacy setup keeps ghost-atom SRB values and gradients zero."""
    par = ParamModule(GFN0_XTB, **DD64)
    no2, lih = samples["NO2"], samples["LiH"]
    numbers = pack((no2["numbers"].to(DEVICE), lih["numbers"].to(DEVICE)))
    positions = pack(
        (no2["positions"].to(**DD64), lih["positions"].to(**DD64))
    ).requires_grad_(True)
    system = Model(
        par=par,
        config=Config.create(exclude="scf"),
    ).setup(numbers, batch_mode=1)
    assert system.srb_setup is not None

    energy = short_range_bond_energy(system.srb_setup, positions)
    (gradient,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(energy[1, 2:]) == 0
    assert torch.count_nonzero(gradient[1, 2:]) == 0


def test_core_history_and_setup_authority() -> None:
    """SRB values and gradients stay stable across geometry calls."""
    model, numbers, positions_a = _system()
    system = model.setup(numbers)
    assert system.srb_classical is not None
    assert system.srb_setup is not None
    positions_b = positions_a.clone()
    positions_b[1, 1] += 0.04

    prior_leaf = positions_a.clone().requires_grad_(True)
    _srb_energy(system.singlepoint(prior_leaf)).backward()
    assert torch.isfinite(prior_leaf.grad).all()  # type: ignore[union-attr]

    result_a = system.singlepoint(positions_a)
    value_a = _srb_energy(result_a)
    value_b = _srb_energy(system.singlepoint(positions_b))
    value_a_again = _srb_energy(system.singlepoint(positions_a.clone()))
    assert not torch.isclose(value_a, value_b)
    torch.testing.assert_close(value_a, value_a_again)

    position_a_leaf = positions_a.clone().requires_grad_(True)
    energy_a = _srb_energy(system.singlepoint(position_a_leaf))
    (gradient_a,) = torch.autograd.grad(energy_a, position_a_leaf)

    position_b_leaf = positions_b.clone().requires_grad_(True)
    energy_b = _srb_energy(system.singlepoint(position_b_leaf))
    (gradient_b,) = torch.autograd.grad(energy_b, position_b_leaf)
    assert not torch.isclose(energy_a, energy_b)
    assert not torch.allclose(gradient_a, gradient_b)

    position_a_again = positions_a.clone().requires_grad_(True)
    energy_a_again = _srb_energy(system.singlepoint(position_a_again))
    (gradient_a_again,) = torch.autograd.grad(energy_a_again, position_a_again)
    torch.testing.assert_close(energy_a, energy_a_again)
    torch.testing.assert_close(gradient_a, gradient_a_again)

    before = system.srb_setup.prefactor.clone()
    with torch.no_grad():
        system.srb_classical.prefactor.add_(1.0)
    torch.testing.assert_close(system.srb_setup.prefactor, before)
    torch.testing.assert_close(
        _srb_energy(system.singlepoint(positions_a)), value_a
    )

    parameter = model.par.get("short_range.srb.prefactor")
    with torch.no_grad():
        parameter.mul_(1.5)
    fresh = model.setup(numbers)
    assert not torch.isclose(
        _srb_energy(fresh.singlepoint(positions_a)), value_a
    )


def test_exact_replacement_and_update_reset_contract() -> None:
    """An exact replacement is set up; in-place exact mutation is rejected."""
    par = ParamModule(GFN0_XTB, **DD64)
    numbers = samples["NO2"]["numbers"].to(DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, par)
    replacement = new_srb(torch.unique(numbers), par, **DD64)
    assert replacement is not None
    replacement.prefactor = replacement.prefactor * 1.5
    model = Model(
        par=par,
        config=Config.create(exclude=("scf", "srb")),
        classical=(replacement,),
    )
    system = model.setup(numbers)
    assert system.srb_classical is replacement
    assert system.srb_setup is not None
    positions = samples["NO2"]["positions"].to(**DD64)
    replacement_energy = _srb_energy(system.singlepoint(positions))
    assert torch.isfinite(replacement_energy)
    default = Model(
        par=par,
        config=Config.create(exclude="scf"),
    ).setup(numbers)
    assert not torch.isclose(
        replacement_energy, _srb_energy(default.singlepoint(positions))
    )

    with pytest.raises(RuntimeError, match="setup-derived"):
        replacement.update(prefactor=torch.ones_like(replacement.prefactor))
    with pytest.raises(RuntimeError, match="setup-derived"):
        replacement.reset()
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.update(
            LABEL_SRB, prefactor=torch.ones_like(replacement.prefactor)
        )
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.reset(LABEL_SRB)
    system.classicals.reset_all()
    assert system.srb_classical is replacement
    assert system.srb_setup is not None

    # The setup helper is also available directly for compatibility callers.
    first = replacement.get_cache(numbers, ihelp)
    second = replacement.get_cache(numbers, ihelp)
    assert isinstance(first, ShortRangeBondSetup)
    assert first is not second
    assert replacement.cache is None
    assert replacement._cachevars is None


def test_calculator_reset_preserves_srb_setup_and_parameter_graph() -> None:
    """Calculator.reset leaves the exact SRB setup and graph untouched."""
    numbers = samples["NO2"]["numbers"].to(DEVICE)
    positions = samples["NO2"]["positions"].to(**DD64)
    par = ParamModule(GFN0_XTB, **DD64)
    par.set_differentiable("short_range.srb.prefactor")
    parameter = par.get("short_range.srb.prefactor")
    calc = Calculator(
        numbers,
        par,
        dtype=torch.double,
        opts={"exclude": "scf", "verbosity": 0},
    )
    setup = calc.system.srb_setup
    first = calc.singlepoint(positions)
    calc.reset()
    second = calc.singlepoint(positions.clone())

    assert calc.system.srb_setup is setup
    torch.testing.assert_close(_srb_energy(first), _srb_energy(second))
    (gradient,) = torch.autograd.grad(setup.prefactor.sum(), parameter)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_subclass_uses_legacy_hooks_and_remains_mutable() -> None:
    """SRB subclasses retain their legacy cache, energy, and mutation hooks."""

    class SRBExtension(ShortRangeBond):
        def get_cache(self, numbers, ihelp=None, **kwargs):
            self.cache_calls = getattr(self, "cache_calls", 0) + 1
            return super().get_cache(numbers, ihelp, **kwargs)

        def get_energy(self, positions, cache, **kwargs):
            self.energy_calls = getattr(self, "energy_calls", 0) + 1
            return super().get_energy(positions, cache, **kwargs)

        def reset(self):
            self.reset_calls = getattr(self, "reset_calls", 0) + 1
            super().reset()

    par = ParamModule(GFN0_XTB, **DD64)
    numbers = samples["NO2"]["numbers"].to(DEVICE)
    base = new_srb(torch.unique(numbers), par, **DD64)
    assert base is not None
    extension = SRBExtension(
        **{slot: getattr(base, slot) for slot in ShortRangeBond.__slots__},
        dtype=torch.double,
    )
    extension.label = LABEL_SRB
    model = Model(
        par=par,
        config=Config.create(exclude=("scf", "srb")),
        classical=(extension,),
    )
    system = model.setup(numbers)
    assert system.srb_setup is None

    positions = samples["NO2"]["positions"].to(**DD64)
    before = _srb_energy(system.singlepoint(positions))
    assert extension.cache_calls >= 2
    assert extension.energy_calls == 1

    extension.update(prefactor=extension.prefactor * 1.2)
    after = _srb_energy(system.singlepoint(positions))
    assert not torch.isclose(before, after)
    extension.reset()
    assert extension.energy_calls == 2
    assert extension.cache_calls >= 3

    calc = Calculator(
        numbers,
        par,
        classical=extension,
        dtype=torch.double,
        opts={"exclude": ["srb", "scf"], "verbosity": 0},
    )
    calc.reset()
    assert extension.reset_calls == 2


@pytest.mark.parametrize(
    "extra_kind", ["builtin_exact", "builtin_subclass", "two_custom"]
)
def test_duplicate_srb_labels_rejected(extra_kind: str) -> None:
    """An ambiguous built-in and additional SRB label fails at setup."""
    numbers = samples["NO2"]["numbers"].to(DEVICE)
    par = ParamModule(GFN0_XTB, **DD64)
    extra = new_srb(torch.unique(numbers), par, **DD64)
    assert extra is not None
    if extra_kind == "builtin_subclass":
        extra = type("SRBExtension", (ShortRangeBond,), {})(
            **{slot: getattr(extra, slot) for slot in ShortRangeBond.__slots__},
            dtype=torch.double,
        )
        extra.label = LABEL_SRB

    classical = (extra,)
    if extra_kind == "two_custom":
        other = new_srb(torch.unique(numbers), par, **DD64)
        assert other is not None
        classical = (extra, other)

    config = Config.create(exclude="scf")
    if extra_kind == "two_custom":
        config = Config.create(exclude=("srb", "scf"))

    with pytest.raises(ValueError, match="Duplicate ShortRangeBond labels"):
        Model(
            par=par,
            config=config,
            classical=classical,
        ).setup(numbers)
