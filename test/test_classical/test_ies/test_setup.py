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
"""IES immutable setup and single-system ownership tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import getsource

import pytest
import torch
from tad_multicharge.model.eeq import EEQModel

from dxtb import GFN0_XTB, Calculator, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import IES, IESSetup, new_ies
from dxtb._src.components.classicals.ies import ies_energy
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples

DD = {"device": DEVICE, "dtype": torch.double}
EXCLUDE = ("scf", "rep", "srb", "hal", "disp")


def _geometry() -> tuple[torch.Tensor, torch.Tensor]:
    sample = samples["SiH4"]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _module() -> ParamModule:
    return ParamModule(GFN0_XTB.model_copy(deep=True), **DD)


def _model(par: ParamModule) -> Model:
    return Model(par=par, config=Config.create(exclude=EXCLUDE))


def _energy(system: Model, positions: torch.Tensor, charge=0.0) -> torch.Tensor:
    return dict(system.singlepoint(positions, chrg=charge).classical)[
        "IES"
    ].sum()


def test_ies_setup_is_frozen_and_cache_compatibility_is_fresh() -> None:
    """IES setup has explicit values and no persistent component cache."""
    numbers, _ = _geometry()
    system = _model(_module()).setup(numbers)
    setup = system.ies_setup

    assert isinstance(setup, IESSetup)
    assert system.ies_classical is not None
    assert system.classical_cache["IES"] is setup
    assert {field.name for field in fields(setup)} == {
        "numbers",
        "eeq",
        "rcov",
        "cutoff",
        "cn_max",
        "cn_kcn",
    }
    assert type(setup.eeq) is EEQModel
    for name in ("numbers", "rcov", "cutoff", "cn_max", "cn_kcn"):
        assert isinstance(getattr(setup, name), torch.Tensor)
    assert not any(
        isinstance(getattr(setup, field.name), (ParamModule, Model, IES))
        for field in fields(setup)
    )
    assert system.ies_classical.get_cache(numbers) is not setup
    assert system.ies_classical.get_cache(numbers) is not setup
    assert system.ies_classical.cache is None
    assert system.ies_classical._cachevars is None
    with pytest.raises(FrozenInstanceError):
        setup.cutoff = torch.tensor(1.0, **DD)  # type: ignore[misc]


def test_ies_has_no_persistent_cache_or_mutation_authority() -> None:
    """IES direct compatibility setup never populates Component cache state."""
    source = getsource(IES)
    for forbidden in (
        "cache_is_latest",
        "self.cache =",
        "self._cachevars",
        "self._cachegrad",
    ):
        assert forbidden not in source


def test_core_ies_does_not_call_legacy_get_cache(monkeypatch) -> None:
    """Single-system IES construction uses System setup directly."""
    numbers, positions = _geometry()
    system = _model(_module()).setup(numbers)
    assert system.ies_classical is not None

    def fail(*args, **kwargs):
        raise AssertionError("core called IES.get_cache")

    monkeypatch.setattr(IES, "get_cache", fail)
    result = system.singlepoint(positions)
    assert torch.isfinite(dict(result.classical)["IES"]).all()


def test_ies_charge_gradient_and_a_b_charge_history() -> None:
    """IES uses each explicit charge and preserves its charge gradient."""
    numbers, positions = _geometry()
    system = _model(_module()).setup(numbers)
    neutral_a = _energy(system, positions, 0.0)
    moved = positions.clone()
    moved[1, 0] += 0.04
    _energy(system, moved, 0.0)
    charged = _energy(system, positions, torch.tensor([0.2], **DD))
    neutral_again = _energy(system, positions.clone(), 0.0)

    charge = torch.tensor(0.2, **DD, requires_grad=True)
    charged_scalar = _energy(system, positions, charge)
    (gradient,) = torch.autograd.grad(charged_scalar, charge)

    assert not torch.isclose(neutral_a, charged)
    torch.testing.assert_close(neutral_a, neutral_again)
    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    "parameter",
    [
        "eeq_chi",
        "eeq_kcn",
        "eeq_eta",
        "eeq_rad",
        "eeq.kcn",
        "eeq.cn_max",
    ],
)
def test_ies_parameter_gradients_reach_param_module(parameter: str) -> None:
    """IES setup retains graph connections to active GFN0 parameters."""
    numbers, positions = _geometry()
    par = _module()
    if parameter.startswith("eeq_"):
        leaf_name = f"parameter_tree.element.Si.{parameter}.param"
    else:
        leaf_name = f"parameter_tree.{parameter}.param"
    leaf = dict(par.named_parameters())[leaf_name]
    leaf.requires_grad_(True)
    system = _model(par).setup(numbers)

    energy = _energy(system, positions)
    (gradient,) = torch.autograd.grad(energy, leaf)
    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) > 0


def test_ies_geometry_transforms_and_charge_are_consistent() -> None:
    """IES ordinary PyTorch energy composes with forward and reverse AD."""
    from torch.func import jacfwd, jvp, vmap

    numbers, positions = _geometry()
    setup = _model(_module()).setup(numbers).ies_setup
    assert setup is not None

    def energy(pos: torch.Tensor) -> torch.Tensor:
        return ies_energy(setup, pos, 0.0).sum()

    leaf = positions.clone().requires_grad_(True)
    reverse = torch.autograd.grad(energy(leaf), leaf)[0]
    direction = torch.linspace(-0.2, 0.3, positions.numel(), **DD).reshape_as(
        positions
    )
    _, tangent = jvp(energy, (positions,), (direction,))
    torch.testing.assert_close(tangent, (reverse * direction).sum())
    torch.testing.assert_close(jacfwd(energy)(positions), reverse)

    second = positions.clone()
    second[1, 0] += 0.04
    conformers = torch.stack((positions, second))
    torch.testing.assert_close(
        vmap(energy)(conformers),
        torch.stack([energy(pos) for pos in conformers]),
    )

    # Exercise reverse orders 1-3 through the same setup function.
    pos = positions.clone().requires_grad_(True)
    grad = torch.autograd.grad(energy(pos), pos, create_graph=True)[0]
    assert torch.isfinite(grad).all()
    grad = torch.autograd.grad(grad.sum(), pos, create_graph=True)[0]
    assert torch.isfinite(grad).all()
    grad = torch.autograd.grad(grad.sum(), pos)[0]
    assert torch.isfinite(grad).all()


def test_ies_mixed_geometry_parameter_derivative_and_fresh_setup() -> None:
    """Position and parameter derivatives compose through fresh IES setup."""
    numbers, positions = _geometry()
    par = _module()
    leaf = dict(par.named_parameters())[
        "parameter_tree.element.Si.eeq_chi.param"
    ]
    leaf.requires_grad_(True)
    model = _model(par)

    def mixed_derivative() -> tuple[torch.Tensor, torch.Tensor]:
        system = model.setup(numbers)
        pos = positions.clone().requires_grad_(True)
        value = _energy(system, pos)
        (gradient,) = torch.autograd.grad(value, pos, create_graph=True)
        direction = torch.arange(1, pos.numel() + 1, **DD).reshape_as(pos)
        (mixed,) = torch.autograd.grad((gradient * direction).sum(), leaf)
        return value, mixed

    energy1, mixed1 = mixed_derivative()
    energy2, mixed2 = mixed_derivative()
    torch.testing.assert_close(energy1, energy2)
    torch.testing.assert_close(mixed1, mixed2)
    assert torch.isfinite(mixed1)
    assert torch.count_nonzero(mixed1) > 0


def test_ies_model_state_and_setup_survive_calls_and_reset() -> None:
    """Repeated calls and Calculator.reset leave IES setup authoritative."""
    numbers, positions = _geometry()
    par = _module()
    leaf = dict(par.named_parameters())[
        "parameter_tree.element.Si.eeq_chi.param"
    ]
    leaf.requires_grad_(True)
    system = _model(par).setup(numbers)
    setup = system.ies_setup
    assert setup is not None and system.ies_classical is not None

    before_model = tuple(
        getattr(setup.eeq, name).clone()
        for name in ("chi", "kcn", "eta", "rad")
    )
    before = _energy(system, positions)
    system.ies_classical.cache_disable()
    disabled = _energy(system, positions.clone())
    system.ies_classical.cache_enable()
    torch.testing.assert_close(disabled, before)

    pos = positions.clone().requires_grad_(True)
    value = _energy(system, pos)
    (position_gradient,) = torch.autograd.grad(value, pos)
    assert torch.isfinite(position_gradient).all()
    _energy(system, positions.clone(), 0.3)
    after = _energy(system, positions.clone())
    torch.testing.assert_close(before, after)
    for name, value_before in zip(("chi", "kcn", "eta", "rad"), before_model):
        torch.testing.assert_close(getattr(setup.eeq, name), value_before)

    # Raw mutation of the legacy object cannot change existing System output.
    with torch.no_grad():
        system.ies_classical.eeq_kcn.fill_(9.0)
    torch.testing.assert_close(_energy(system, positions), before)

    calc = Calculator(
        numbers,
        par,
        dtype=torch.double,
        opts={"exclude": EXCLUDE, "verbosity": 0},
        auto_int_level=False,
    )
    calc_setup = calc.system.ies_setup
    calc_before = _energy(calc.system, positions)
    calc.reset()
    assert calc.system.ies_setup is calc_setup
    torch.testing.assert_close(
        _energy(calc.system, positions.clone()), calc_before
    )
    assert calc_setup is not None
    (parameter_gradient,) = torch.autograd.grad(calc_setup.eeq.chi.sum(), leaf)
    assert torch.isfinite(parameter_gradient).all()
    assert torch.count_nonzero(parameter_gradient) > 0

    with torch.no_grad():
        leaf.add_(0.2)
    fresh = _model(par).setup(numbers)
    assert not torch.isclose(_energy(fresh, positions), calc_before)


def test_exact_custom_ies_replaces_excluded_builtin_and_mutation_is_rejected() -> (
    None
):
    """One exact custom IES supplies setup when the built-in is excluded."""
    numbers, positions = _geometry()
    par = _module()
    custom = new_ies(numbers, par, **DD)
    assert custom is not None
    custom.cutoff = custom.cutoff.new_tensor(0.5)
    system = Model(
        par=par,
        config=Config.create(exclude=("ies", *EXCLUDE)),
        classical=(custom,),
    ).setup(numbers)
    assert system.ies_classical is custom
    assert system.ies_setup is not None
    energy = _energy(system, positions)
    assert not torch.isclose(
        energy,
        _energy(_model(_module()).setup(numbers), positions),
    )
    with pytest.raises(RuntimeError, match="setup-derived"):
        custom.update(cutoff=torch.tensor(1.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        custom.reset()
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.update("IES", cutoff=torch.tensor(1.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.reset("IES")


def test_ies_subclass_refreshes_legacy_cache_and_reset() -> None:
    """IES subclasses remain cache-driven mutable extensions."""
    numbers, positions = _geometry()
    par = _module()
    base = new_ies(numbers, par, **DD)
    assert base is not None

    class IESExtension(IES):
        __slots__ = ("scale", "initial_scale", "cache_calls", "reset_calls")

        def __init__(self) -> None:
            super().__init__(
                base.chi,
                base.eeq_kcn,
                base.eta,
                base.rad,
                base.rcov,
                base.cutoff,
                base.cn_max,
                base.cn_kcn,
                **DD,
            )
            self.scale = torch.tensor(1.0, **DD)
            self.initial_scale = self.scale.clone()
            self.cache_calls = 0
            self.reset_calls = 0
            self.label = "IES"

        def get_cache(self, numbers, ihelp=None, **kwargs):
            self.cache_calls += 1
            return self.scale.clone()

        def get_energy(self, positions, cache, **kwargs):
            return positions.new_ones(positions.shape[-2]) * cache

        def reset(self) -> None:
            self.reset_calls += 1
            self.scale = self.initial_scale.clone()

    extension = IESExtension()
    calc = Calculator(
        numbers,
        par,
        classical=(extension,),
        opts={"exclude": ("ies", *EXCLUDE), "verbosity": 0},
        dtype=torch.double,
        auto_int_level=False,
    )
    first = dict(calc.system.singlepoint(positions).classical)["IES"].sum()
    calc.classicals.update("IES", scale=torch.tensor(2.0, **DD))
    updated = dict(calc.system.singlepoint(positions).classical)["IES"].sum()
    assert updated == 2 * first
    calc.reset()
    reset = dict(calc.system.singlepoint(positions).classical)["IES"].sum()
    torch.testing.assert_close(reset, first)
    assert extension.cache_calls == 4
    assert extension.reset_calls == 1


def test_ies_duplicate_labels_are_rejected() -> None:
    """Ambiguous IES labels fail before label-keyed data can overwrite."""
    numbers, _ = _geometry()
    par = _module()
    custom1 = new_ies(numbers, par, **DD)
    custom2 = new_ies(numbers, par, **DD)
    assert custom1 is not None and custom2 is not None

    class IESExtension(IES):
        pass

    extension = IESExtension(
        custom1.chi,
        custom1.eeq_kcn,
        custom1.eta,
        custom1.rad,
        custom1.rcov,
        custom1.cutoff,
        custom1.cn_max,
        custom1.cn_kcn,
        **DD,
    )
    extension.label = "IES"

    with pytest.raises(ValueError, match="IES.*ambiguous"):
        Model(
            par=par,
            config=Config.create(exclude=EXCLUDE),
            classical=(custom1,),
        ).setup(numbers)
    with pytest.raises(ValueError, match="IES.*ambiguous"):
        Model(
            par=par,
            config=Config.create(exclude=EXCLUDE),
            classical=(extension,),
        ).setup(numbers)
    with pytest.raises(ValueError, match="IES.*ambiguous"):
        Model(
            par=par,
            config=Config.create(exclude=("ies", *EXCLUDE)),
            classical=(custom1, custom2),
        ).setup(numbers)
