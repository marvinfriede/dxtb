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
"""Tests for explicit classical D4 setup and evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest
import tad_dftd4 as d4
import torch

from dxtb import GFN2_XTB, Calculator, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import DispersionD4, DispersionD4Setup
from dxtb._src.components.classicals.dispersion.d4 import (
    dispersion_d4_energy,
)
from dxtb._src.param import Param
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples

DD = {"device": DEVICE, "dtype": torch.double}


def _parameters(*, self_consistent: bool) -> ParamModule:
    """Create a fresh GFN2 ParamModule in the requested D4 mode."""
    par = GFN2_XTB.model_copy(deep=True)
    par.dispersion.d4.sc = self_consistent  # type: ignore[union-attr]
    return ParamModule(par, **DD)


def _geometry(name: str = "C4H5NCS") -> tuple[Tensor, Tensor]:
    sample = samples[name]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _system(
    par: ParamModule,
    *,
    exclude: tuple[str, ...] = ("scf", "d4sc"),
) -> Model:
    return Model(par=par, config=Config.create(exclude=exclude))


def _classical_energy(result: Any) -> Tensor:
    return dict(result.classical)["DispersionD4"].sum()


def test_setup_is_frozen_and_uses_immutable_parameter_structure() -> None:
    """D4 setup stores resolved data, not mutable component/cache objects."""
    numbers, _ = _geometry()
    system = _system(_parameters(self_consistent=False)).setup(numbers)
    setup = system.d4_setup

    assert isinstance(setup, DispersionD4Setup)
    assert system.d4_classical is not None
    assert system.classical_cache["DispersionD4"] is setup
    assert {field.name for field in fields(setup)} == {
        "numbers",
        "parameters",
        "model",
        "rcov",
        "r4r2",
        "cutoff",
        "counting_function",
        "damping_function",
        "ref_charges",
        "compatibility_q",
    }
    assert isinstance(setup.parameters, tuple)
    assert all(isinstance(entry, tuple) for entry in setup.parameters)
    forbidden = (Param, ParamModule, Model, DispersionD4)
    assert all(
        not isinstance(value, forbidden)
        for field in fields(setup)
        for value in (getattr(setup, field.name),)
    )
    assert all(
        not isinstance(value, forbidden) for _, value in setup.parameters
    )
    assert not isinstance(setup.model, DispersionD4)
    assert setup.compatibility_q is None
    assert not any("positions" in field.name for field in fields(setup))
    assert not any("energy" in field.name for field in fields(setup))
    assert not any("charge" == field.name for field in fields(setup))
    with pytest.raises(FrozenInstanceError):
        setup.ref_charges = "gfn2"  # type: ignore[misc]

    first = system.d4_classical.get_cache(numbers)
    second = system.d4_classical.get_cache(numbers)
    assert first is not second
    assert system.d4_classical.cache is None
    assert system.d4_classical._cachevars is None
    system.d4_classical.cache_disable()
    assert system.d4_classical.get_cache(numbers) is not second


def test_explicit_q_remains_a_legacy_get_cache_override() -> None:
    """Direct callers can still provide static atom charges to the adapter."""
    numbers, positions = _geometry()
    dispersion = DispersionD4(
        numbers,
        {"a1": torch.tensor(0.5, **DD), "a2": torch.tensor(5.0, **DD)},
        charge=torch.tensor(0.0, **DD),
        **DD,
    )
    q = torch.linspace(-0.2, 0.2, numbers.numel(), **DD)
    setup = dispersion.get_cache(numbers, q=q)
    adapter_energy = dispersion.get_energy(positions, setup)
    explicit_energy = dispersion_d4_energy(setup, positions, 0.0, q=q)
    torch.testing.assert_close(adapter_energy, explicit_energy)


def test_unsupported_d4_model_subclass_is_rejected_explicitly() -> None:
    """Do not silently erase custom model behavior while copying setup data."""
    numbers, _ = _geometry()

    class CustomD4Model(d4.model.D4Model):
        pass

    dispersion = DispersionD4(
        numbers,
        {"a1": torch.tensor(0.5, **DD), "a2": torch.tensor(5.0, **DD)},
        charge=torch.tensor(0.0, **DD),
        **DD,
    )
    model = CustomD4Model(numbers, **DD)
    with pytest.raises(TypeError, match="subclasses cannot be copied safely"):
        dispersion.get_cache(numbers, model=model)


def test_classical_d4_modes_keep_distinct_configuration() -> None:
    """Ordinary D4 and GFN2 ATM retain their existing parameter modes."""
    numbers, _ = _geometry()
    ordinary = _system(_parameters(self_consistent=False)).setup(numbers)
    gfn2 = Model(
        par=_parameters(self_consistent=True),
        config=Config.create(exclude="scf"),
    ).setup(numbers)

    ordinary_setup = ordinary.d4_setup
    atm_setup = gfn2.d4_setup
    assert ordinary_setup is not None and atm_setup is not None
    ordinary_param = dict(ordinary_setup.parameters)
    atm_param = dict(atm_setup.parameters)
    assert ordinary_setup.ref_charges == "eeq"
    assert atm_setup.ref_charges == "gfn2"
    assert ordinary_param["s6"] != 0.0
    assert ordinary_param["s8"] != 0.0
    assert atm_param["s6"] == 0.0
    assert atm_param["s8"] == 0.0
    assert atm_param["s9"] != 0.0
    assert any(
        interaction.label == "DispersionD4SC"
        for interaction in gfn2.interactions.components
    )


def test_core_charge_is_explicit_and_history_independent() -> None:
    """Core classical D4 uses each call's total charge and local EEQ q."""
    numbers, positions = _geometry()
    system = _system(_parameters(self_consistent=False)).setup(numbers)

    neutral = _classical_energy(system.singlepoint(positions, chrg=0.0))
    moved = positions.clone()
    moved[1, 0] += 0.08
    _classical_energy(system.singlepoint(moved, chrg=0.0))
    charged = _classical_energy(system.singlepoint(positions, chrg=1.0))
    neutral_again = _classical_energy(
        system.singlepoint(positions.clone(), chrg=0.0)
    )

    assert not torch.isclose(neutral, charged)
    torch.testing.assert_close(neutral, neutral_again)


def test_d4_model_tables_are_not_mutated_by_geometry_or_charge_calls() -> None:
    """External D4 model tables stay unchanged across A/B/A evaluations."""
    numbers, positions = _geometry()
    system = _system(_parameters(self_consistent=False)).setup(numbers)
    setup = system.d4_setup
    assert setup is not None
    before_numbers = setup.model.numbers.clone()
    before_wf = setup.model.wf.clone()
    before_rc6 = setup.model.rc6.clone()
    before_cutoff = tuple(
        getattr(setup.cutoff, name).clone()
        for name in ("disp2", "disp3", "cn", "cn_eeq")
    )

    a1 = dispersion_d4_energy(setup, positions, 0.0)
    moved = positions.clone()
    moved[1, 0] += 0.08
    derivative_position = positions.clone().requires_grad_(True)
    differentiated = dispersion_d4_energy(setup, derivative_position, 0.0)
    torch.autograd.grad(differentiated.sum(), derivative_position)
    dispersion_d4_energy(setup, moved, 1.0)
    a2 = dispersion_d4_energy(setup, positions.clone(), 0.0)

    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(setup.model.numbers, before_numbers)
    torch.testing.assert_close(setup.model.wf, before_wf)
    torch.testing.assert_close(setup.model.rc6, before_rc6)
    for name, value in zip(("disp2", "disp3", "cn", "cn_eeq"), before_cutoff):
        torch.testing.assert_close(getattr(setup.cutoff, name), value)


def test_actual_parameter_leaf_survives_setup_and_stale_object_mutation() -> (
    None
):
    """D4 setup isolates its values but preserves ParamModule gradients."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=False)
    a1 = par.get("dispersion.d4.a1")
    a1.requires_grad_(True)
    model = _system(par)
    system = model.setup(numbers)
    setup = system.d4_setup
    assert setup is not None and system.d4_classical is not None

    before = _classical_energy(system.singlepoint(positions))
    (gradient,) = torch.autograd.grad(before, a1)
    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) > 0

    # Replacing the compatibility object's mapping entry cannot alter setup.
    system.d4_classical.param["a1"] = torch.tensor(9.0, **DD)
    after = _classical_energy(system.singlepoint(positions.clone()))
    torch.testing.assert_close(before, after)

    with torch.no_grad():
        a1.fill_(0.25)
    fresh = model.setup(numbers)
    changed = _classical_energy(fresh.singlepoint(positions))
    assert not torch.allclose(changed, after)


def test_parameter_gradient_repeats_with_fresh_setup() -> None:
    """A consumed D4 setup graph is rebuilt from the same parameter leaf."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=False)
    leaf = par.get("dispersion.d4.a1")
    leaf.requires_grad_(True)
    model = _system(par)

    first = model.setup(numbers)
    energy_first = _classical_energy(first.singlepoint(positions))
    (gradient_first,) = torch.autograd.grad(energy_first, leaf)

    second = model.setup(numbers)
    energy_second = _classical_energy(second.singlepoint(positions.clone()))
    (gradient_second,) = torch.autograd.grad(energy_second, leaf)

    torch.testing.assert_close(energy_first, energy_second)
    torch.testing.assert_close(gradient_first, gradient_second)


@pytest.mark.parametrize(
    "parameter",
    ["a1", "a2", "s6", "s8", "s9", "s10", "alp"],
)
def test_ordinary_d4_gradients_reach_parameter_module(parameter: str) -> None:
    """Ordinary D4 parameters remain connected through Model.setup."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=False)
    leaf = par.get(f"dispersion.d4.{parameter}")
    leaf.requires_grad_(True)
    system = _system(par).setup(numbers)
    energy = _classical_energy(system.singlepoint(positions, chrg=0.0))
    (gradient,) = torch.autograd.grad(energy, leaf)
    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("parameter", ["s9", "alp"])
def test_gfn2_atm_gradients_reach_parameter_module(parameter: str) -> None:
    """The standard GFN2 classical ATM path retains active parameter graphs."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=True)
    leaf = par.get(f"dispersion.d4.{parameter}")
    leaf.requires_grad_(True)
    system = _system(par).setup(numbers)
    energy = _classical_energy(system.singlepoint(positions, chrg=0.0))
    (gradient,) = torch.autograd.grad(energy, leaf)
    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) > 0


def test_d4_mixed_geometry_parameter_derivative() -> None:
    """Position derivatives through setup remain differentiable by D4 params."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=False)
    a1 = par.get("dispersion.d4.a1")
    a1.requires_grad_(True)
    system = _system(par).setup(numbers)
    pos = positions.clone().requires_grad_(True)
    energy = _classical_energy(system.singlepoint(pos, chrg=0.0))
    (gradient,) = torch.autograd.grad(energy, pos, create_graph=True)
    direction = torch.arange(1, pos.numel() + 1, dtype=pos.dtype).reshape_as(
        pos
    )
    projected = (gradient * direction).sum()
    (mixed,) = torch.autograd.grad(projected, a1)
    assert torch.isfinite(mixed)
    assert torch.count_nonzero(mixed) > 0


def test_d4_jacfwd_and_jvp_match_reverse_derivative() -> None:
    """Forward transforms agree with the reverse derivative of D4 energy."""
    from torch.func import jacfwd, jvp

    numbers, positions = _geometry()
    setup = _system(_parameters(self_consistent=False)).setup(numbers).d4_setup
    assert setup is not None

    def energy(pos: Tensor) -> Tensor:
        return dispersion_d4_energy(setup, pos, 0.0).sum()

    direction = torch.linspace(-0.3, 0.4, positions.numel(), **DD).reshape_as(
        positions
    )
    _, tangent = jvp(energy, (positions,), (direction,))
    leaf = positions.clone().requires_grad_(True)
    reverse = torch.autograd.grad(energy(leaf), leaf)[0]

    torch.testing.assert_close(tangent, torch.sum(reverse * direction))
    torch.testing.assert_close(jacfwd(energy)(positions), reverse)


def test_d4_history_after_backward_matches_fresh_setup() -> None:
    """A consumed geometry graph does not affect later D4 calls."""
    numbers, positions = _geometry()
    model = _system(_parameters(self_consistent=False))
    system = model.setup(numbers)
    pos_a = positions.clone().requires_grad_(True)
    energy_a = _classical_energy(system.singlepoint(pos_a, chrg=0.0))
    (gradient_a,) = torch.autograd.grad(energy_a, pos_a)

    pos_b = positions.clone()
    pos_b[1, 0] += 0.08
    _classical_energy(system.singlepoint(pos_b, chrg=1.0))

    pos_a_again = positions.clone().requires_grad_(True)
    energy_a_again = _classical_energy(
        system.singlepoint(pos_a_again, chrg=0.0)
    )
    (gradient_a_again,) = torch.autograd.grad(energy_a_again, pos_a_again)

    fresh = model.setup(numbers)
    fresh_energy = _classical_energy(
        fresh.singlepoint(positions.clone(), chrg=0.0)
    )
    torch.testing.assert_close(energy_a, energy_a_again)
    torch.testing.assert_close(energy_a, fresh_energy)
    torch.testing.assert_close(gradient_a, gradient_a_again)


def test_exact_d4_update_reset_and_calculator_reset_preserve_setup() -> None:
    """Exact D4 mutation is rejected and Calculator.reset leaves setup alone."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=False)
    system = _system(par).setup(numbers)
    d4_component = system.d4_classical
    setup = system.d4_setup
    assert d4_component is not None and setup is not None

    with pytest.raises(RuntimeError, match="setup-derived"):
        d4_component.update(charge=torch.tensor(1.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        d4_component.reset()
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.update("DispersionD4", charge=torch.tensor(1.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.reset("DispersionD4")

    before = _classical_energy(system.singlepoint(positions))
    system.classicals.reset_all()
    after = _classical_energy(system.singlepoint(positions.clone()))
    assert system.d4_setup is setup
    torch.testing.assert_close(before, after)

    calc_parameters = _parameters(self_consistent=False)
    calc_parameters.get("dispersion.d4.a1").requires_grad_(True)
    calc = Calculator(
        numbers,
        calc_parameters,
        dtype=torch.double,
        opts={"verbosity": 0, "exclude": ["scf", "d4sc"]},
    )
    calc_setup = calc.system.d4_setup
    calc_before = _classical_energy(calc.singlepoint(positions))
    calc.reset()
    calc_after = _classical_energy(calc.singlepoint(positions.clone()))
    assert calc.system.d4_setup is calc_setup
    torch.testing.assert_close(calc_before, calc_after)
    leaf = calc.model.par.get("dispersion.d4.a1")
    (gradient,) = torch.autograd.grad(calc_before, leaf)
    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) > 0


def test_exact_user_d4_replaces_excluded_builtin() -> None:
    """The setup is constructed from the active user-provided D4 term."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=False)
    dispersion = DispersionD4(
        numbers,
        {
            "a1": torch.tensor(0.2, **DD),
            "a2": torch.tensor(5.0, **DD),
            "s6": torch.tensor(1.0, **DD),
            "s8": torch.tensor(2.7, **DD),
            "s9": torch.tensor(5.0, **DD),
            "s10": torch.tensor(0.0, **DD),
        },
        charge=torch.tensor(0.0, **DD),
        ref_charges="eeq",
        **DD,
    )
    system = Model(
        par=par,
        config=Config.create(exclude=("disp", "scf")),
        classical=(dispersion,),
    ).setup(numbers)

    assert system.d4_classical is dispersion
    assert system.d4_setup is not None
    assert dict(system.d4_setup.parameters)["a1"].item() == 0.2
    result = system.singlepoint(positions, chrg=0.0)
    assert torch.isfinite(_classical_energy(result))


def test_duplicate_classical_d4_labels_are_rejected() -> None:
    """Active D4 labels cannot overwrite one another in classical call data."""
    numbers, _ = _geometry()
    par = _parameters(self_consistent=False)
    user = DispersionD4(
        numbers,
        {"a1": torch.tensor(0.5, **DD), "a2": torch.tensor(5.0, **DD)},
        charge=torch.tensor(0.0, **DD),
        **DD,
    )
    with pytest.raises(ValueError, match="Duplicate DispersionD4 labels"):
        Model(par=par, classical=(user,)).setup(numbers)

    class D4Extension(DispersionD4):
        pass

    subclass = D4Extension(
        numbers,
        user.param,
        charge=user.charge,
        ref_charges=user.ref_charges,
        **DD,
    )
    subclass.label = "DispersionD4"
    with pytest.raises(ValueError, match="Duplicate DispersionD4 labels"):
        Model(par=par, classical=(subclass,)).setup(numbers)
    with pytest.raises(ValueError, match="Duplicate DispersionD4 labels"):
        Model(
            par=par,
            config=Config.create(exclude=("disp", "scf")),
            classical=(
                user,
                DispersionD4(
                    numbers,
                    {"a1": torch.tensor(0.6, **DD)},
                    charge=torch.tensor(0.0, **DD),
                    **DD,
                ),
            ),
        ).setup(numbers)


def test_d4_subclass_remains_on_legacy_extension_path() -> None:
    """A D4 subclass keeps its cache, energy, update, and reset hooks."""
    numbers, positions = _geometry()
    par = _parameters(self_consistent=False)
    base = DispersionD4(
        numbers,
        {"a1": torch.tensor(0.5, **DD)},
        charge=torch.tensor(0.0, **DD),
        **DD,
    )

    class D4Extension(DispersionD4):
        __slots__ = ("scale", "cache_calls", "energy_calls", "reset_calls")

        def __init__(self) -> None:
            super().__init__(
                base.numbers,
                base.param,
                charge=base.charge,
                ref_charges=base.ref_charges,
                **DD,
            )
            self.label = "DispersionD4"
            self.scale = torch.tensor(2.0, **DD)
            self.cache_calls = 0
            self.energy_calls = 0
            self.reset_calls = 0

        def get_cache(self, numbers, ihelp=None, **kwargs):
            self.cache_calls += 1
            return torch.tensor(1.0, **DD)

        def get_energy(self, positions, cache, **kwargs):
            self.energy_calls += 1
            return positions.new_ones(positions.shape[-2]) * self.scale

        def reset(self) -> None:
            self.reset_calls += 1
            super().reset()

    extension = D4Extension()
    system = Model(
        par=par,
        config=Config.create(exclude=("disp", "scf")),
        classical=(extension,),
    ).setup(numbers)

    assert system.d4_setup is None
    assert system.d4_classical is None
    assert extension.cache_calls == 1
    energy = _classical_energy(system.singlepoint(positions))
    assert extension.energy_calls == 1
    torch.testing.assert_close(
        energy, torch.tensor(2.0 * positions.shape[-2], **DD)
    )

    system.classicals.update("DispersionD4", scale=torch.tensor(3.0, **DD))
    updated = _classical_energy(system.singlepoint(positions.clone()))
    torch.testing.assert_close(
        updated, torch.tensor(3.0 * positions.shape[-2], **DD)
    )
    system.classicals.reset("DispersionD4")
    system.classicals.reset_all()
    assert extension.reset_calls == 2
