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
"""Tests for explicit D4SC setup and call-local data."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import tad_dftd4 as d4
import torch
from tad_mctc.batch import pack

from dxtb import GFN2_XTB, Calculator, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.interactions import InteractionList
from dxtb._src.components.interactions.base import InteractionCache
from dxtb._src.components.interactions.dispersion.d4sc import (
    D4SCSetup,
    DispersionD4SC,
    DispersionD4SCCache,
    build_d4sc_data,
    new_d4sc,
    setup_d4sc,
)
from dxtb._src.param import Param
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples

DD = {"device": DEVICE, "dtype": torch.double}


def _sample(name: str = "LiH") -> tuple[Tensor, Tensor]:
    sample = samples[name]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _parameters() -> ParamModule:
    return ParamModule(GFN2_XTB, **DD)


def _model(par: ParamModule, *, exclude: tuple[str, ...] = ()) -> Model:
    return Model(
        par=par,
        config=Config.create(exclude=exclude),
        auto_int_level=False,
    )


def test_old_culling_alias_is_removed() -> None:
    """Culling local D4SC data never edits the component model."""
    numbers1, positions1 = _sample("LiH")
    numbers2, positions2 = _sample("SiH4")
    numbers = pack((numbers1, numbers2))
    positions = pack((positions1, positions2))
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None

    setup = setup_d4sc(interaction, numbers)
    persistent_numbers = interaction.model.numbers.clone()
    persistent_wf = setup.model_wf.clone()
    persistent_rc6 = setup.model_rc6.clone()
    persistent_parameters = tuple(
        (name, value.clone() if isinstance(value, Tensor) else value)
        for name, value in setup.parameters
    )
    local_positions = positions.clone().requires_grad_(True)
    data = build_d4sc_data(setup, local_positions)
    local_numbers = data.model.numbers.clone()

    assert data.model is not interaction.model
    assert torch.equal(setup.numbers, persistent_numbers)
    q = torch.zeros(numbers.shape, **DD)
    energy = interaction.get_monopole_atom_energy(data, q).sum()
    potential = interaction.get_monopole_atom_potential(data, q)
    (position_grad,) = torch.autograd.grad(energy, local_positions)
    assert torch.isfinite(potential).all()
    assert torch.isfinite(position_grad).all()
    data.cull(
        torch.tensor([True, False], device=DEVICE),
        {"atom": [slice(0, numbers.shape[-1])]},
    )
    assert torch.equal(interaction.model.numbers, persistent_numbers)
    assert torch.equal(setup.numbers, persistent_numbers)
    assert torch.equal(setup.model_wf, persistent_wf)
    assert torch.equal(setup.model_rc6, persistent_rc6)
    for (name, value), (snapshot_name, snapshot) in zip(
        setup.parameters, persistent_parameters
    ):
        assert name == snapshot_name
        if isinstance(value, Tensor):
            assert isinstance(snapshot, Tensor)
            assert torch.equal(value, snapshot)
        else:
            assert value == snapshot

    data.restore()
    assert torch.equal(data.model.numbers, local_numbers)
    assert torch.equal(interaction.model.numbers, persistent_numbers)
    assert build_d4sc_data(setup, positions) is not data


def test_direct_get_cache_is_fresh_and_nonpersistent() -> None:
    """The compatibility spelling returns fresh data without cache state."""
    numbers, positions = _sample()
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None

    first = interaction.get_cache(numbers=numbers, positions=positions)
    second = interaction.get_cache(numbers=numbers, positions=positions)
    assert isinstance(first, DispersionD4SCCache)
    assert first is not second
    assert first.model is not interaction.model
    assert second.model is not interaction.model
    assert interaction.cache is None
    assert interaction._cachevars is None
    assert interaction._cachegrad is None


def test_setup_is_frozen_and_contains_only_static_data() -> None:
    """D4SC setup excludes models, geometry and call outputs."""
    numbers, _ = _sample()
    par = _parameters()
    system = _model(par, exclude=("es2", "es3", "aes2")).setup(numbers)
    setup = system.d4sc_setup

    assert isinstance(setup, D4SCSetup)
    assert system.d4sc_interaction is not None
    assert setup is not None
    assert {field.name for field in fields(setup)} == {
        "numbers",
        "parameters",
        "rcov",
        "r4r2",
        "cn_cutoff",
        "pair_weight",
        "counting_function",
        "model_type",
        "model_ga",
        "model_gc",
        "model_wf",
        "model_ref_charges",
        "model_rc6",
        "model_extra",
    }
    assert isinstance(setup.parameters, tuple)
    assert all(isinstance(entry, tuple) for entry in setup.parameters)
    assert not any(
        isinstance(
            getattr(setup, field.name),
            (
                Param,
                ParamModule,
                Model,
                DispersionD4SC,
                DispersionD4SCCache,
                d4.model.D4Model,
            ),
        )
        for field in fields(setup)
    )
    assert not {"positions", "cn", "dispmat", "charge"} & {
        field.name for field in fields(setup)
    }
    with pytest.raises(FrozenInstanceError):
        setup.cn_cutoff = torch.tensor(1.0, **DD)  # type: ignore[misc]


def test_core_singlepoint_bypasses_exact_get_cache(monkeypatch) -> None:
    """Single-system core builds D4SC data from System setup directly."""
    numbers, positions = _sample()
    system = _model(_parameters(), exclude=("es2", "es3", "aes2")).setup(
        numbers
    )
    interaction = system.d4sc_interaction
    assert interaction is not None

    def fail_get_cache(*args, **kwargs):
        raise AssertionError("core called exact D4SC.get_cache")

    monkeypatch.setattr(DispersionD4SC, "get_cache", fail_get_cache)
    data = _interaction_data(system, positions)
    assert isinstance(data[interaction.label], DispersionD4SCCache)
    assert data[interaction.label].model is not interaction.model
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()


def test_custom_model_subclass_is_reconstructed_for_direct_compatibility() -> (
    None
):
    """Direct D4SC compatibility preserves D4Model subclass behavior."""
    numbers, positions = _sample()
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None

    class ScaledD4Model(d4.model.D4Model):
        def weight_references(self, *args, **kwargs):
            result = super().weight_references(*args, **kwargs)
            if isinstance(result, tuple):
                return tuple(1.05 * value for value in result)
            return 1.05 * result

    interaction.model = ScaledD4Model(
        numbers,
        ga=interaction.model.ga,
        gc=interaction.model.gc,
        wf=interaction.model.wf,
        ref_charges=interaction.model.ref_charges,
        rc6=interaction.model.rc6,
        **DD,
    )
    data = interaction.get_cache(numbers=numbers, positions=positions)
    assert type(data.model) is ScaledD4Model
    q = torch.tensor([0.0, 0.0], **DD)
    actual = interaction.get_monopole_atom_energy(data, q)

    standard = new_d4sc(numbers, _parameters(), **DD)
    assert standard is not None
    standard_data = standard.get_cache(numbers=numbers, positions=positions)
    expected = standard.get_monopole_atom_energy(standard_data, q)
    assert not torch.allclose(actual, expected)

    system = Model(
        par=_parameters(),
        interaction=(interaction,),
        config=Config.create(exclude=("d4sc", "es2", "es3", "aes2")),
        auto_int_level=False,
    ).setup(numbers)
    assert system.d4sc_setup is not None
    call_data = build_d4sc_data(system.d4sc_setup, positions)
    assert type(call_data.model) is ScaledD4Model


def test_call_local_potential_is_energy_charge_derivative() -> None:
    """D4SC potential remains the derivative of the atom energy sum."""
    numbers, positions = _sample("SiH4")
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None
    data = build_d4sc_data(setup_d4sc(interaction, numbers), positions)
    charges = torch.tensor(
        [0.05, -0.02, 0.01, -0.02, -0.02], **DD, requires_grad=True
    )
    energy = interaction.get_monopole_atom_energy(data, charges).sum()
    (gradient,) = torch.autograd.grad(energy, charges)
    potential = interaction.get_monopole_atom_potential(data, charges)
    torch.testing.assert_close(gradient, potential)

    direction = torch.linspace(-0.2, 0.3, charges.numel(), **DD)
    _, tangent = torch.func.jvp(
        lambda q: interaction.get_monopole_atom_potential(data, q),
        (charges.detach(),),
        (direction,),
    )
    jac = torch.func.jacfwd(
        lambda q: interaction.get_monopole_atom_potential(data, q)
    )(charges.detach())
    torch.testing.assert_close(tangent, jac @ direction)
    batch = torch.stack((charges.detach(), charges.detach() + 0.01))
    mapped = torch.func.vmap(
        lambda q: interaction.get_monopole_atom_potential(data, q)
    )(batch)
    loop = torch.stack(
        [interaction.get_monopole_atom_potential(data, q) for q in batch]
    )
    torch.testing.assert_close(mapped, loop)


def test_setup_parameters_retain_parammodule_gradient() -> None:
    """D4SC parameter tensors remain connected to ParamModule leaves."""
    numbers, positions = _sample("SiH4")
    par = _parameters()
    active = ("a1", "a2", "s6", "s8", "s10")
    leaves = [par.get(f"dispersion.d4.{name}") for name in active]
    for leaf in leaves:
        leaf.requires_grad_(True)
    interaction = new_d4sc(numbers, par, **DD)
    assert interaction is not None
    setup = setup_d4sc(interaction, numbers)
    data = build_d4sc_data(setup, positions)
    charges = torch.tensor([0.05, -0.02, 0.01, -0.02, -0.02], **DD)
    energy = interaction.get_monopole_atom_energy(data, charges).sum()
    grads = torch.autograd.grad(energy, leaves, allow_unused=True)

    for grad in grads:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0

    par2 = _parameters()
    leaves2 = [par2.get(f"dispersion.d4.{name}") for name in active]
    for leaf in leaves2:
        leaf.requires_grad_(True)
    interaction2 = new_d4sc(numbers, par2, **DD)
    assert interaction2 is not None
    data2 = build_d4sc_data(setup_d4sc(interaction2, numbers), positions)
    energy2 = interaction2.get_monopole_atom_energy(data2, charges).sum()
    grads2 = torch.autograd.grad(energy2, leaves2)
    for grad, grad2 in zip(grads, grads2):
        torch.testing.assert_close(grad, grad2)

    par3 = _parameters()
    s9 = par3.get("dispersion.d4.s9")
    s9.requires_grad_(True)
    interaction3 = new_d4sc(numbers, par3, **DD)
    assert interaction3 is not None
    data3 = build_d4sc_data(setup_d4sc(interaction3, numbers), positions)
    energy3 = interaction3.get_monopole_atom_energy(data3, charges).sum()
    assert not energy3.requires_grad


def test_mixed_position_parameter_derivative_is_finite() -> None:
    """D4SC position derivatives compose with a real parameter leaf."""
    numbers, positions = _sample("SiH4")
    par = _parameters()
    a1 = par.get("dispersion.d4.a1")
    a1.requires_grad_(True)
    interaction = new_d4sc(numbers, par, **DD)
    assert interaction is not None
    setup = setup_d4sc(interaction, numbers)
    pos = positions.clone().requires_grad_(True)
    q = torch.tensor([0.03, -0.02, -0.01, 0.01, -0.01], **DD)

    def energy(x: Tensor) -> Tensor:
        return interaction.get_monopole_atom_energy(
            build_d4sc_data(setup, x), q
        ).sum()

    (grad_pos,) = torch.autograd.grad(energy(pos), pos, create_graph=True)
    projection = (
        grad_pos * torch.linspace(-0.2, 0.4, pos.numel(), **DD).reshape_as(pos)
    ).sum()
    (mixed,) = torch.autograd.grad(projection, a1)
    assert torch.isfinite(mixed).all()
    assert torch.count_nonzero(mixed) > 0


def test_stale_component_mutation_does_not_change_existing_setup() -> None:
    """Only changed ParamModule values affect a newly built System setup."""
    numbers, positions = _sample("SiH4")
    par = _parameters()
    system = _model(par, exclude=("es2", "es3", "aes2")).setup(numbers)
    interaction = system.d4sc_interaction
    setup = system.d4sc_setup
    assert interaction is not None and setup is not None
    q = torch.tensor([0.03, -0.02, -0.01, 0.01, -0.01], **DD)

    def energy(current_setup: D4SCSetup) -> Tensor:
        data = build_d4sc_data(current_setup, positions)
        return interaction.get_monopole_atom_energy(data, q).sum()

    before = energy(setup)
    with torch.no_grad():
        interaction.param["a1"].add_(0.2)
        interaction.model.wf.add_(0.5)
        interaction.model.rc6.mul_(1.1)
        interaction.model.numbers = interaction.model.numbers.clone()
        interaction.model.numbers[0] = 8
    torch.testing.assert_close(energy(setup), before)

    leaf = par.get("dispersion.d4.a1")
    with torch.no_grad():
        leaf.add_(0.2)
    fresh = _model(par, exclude=("es2", "es3", "aes2")).setup(numbers)
    assert fresh.d4sc_setup is not None
    assert not torch.allclose(energy(fresh.d4sc_setup), before)


def test_d4sc_charge_and_geometry_history_is_independent() -> None:
    """A/B/A geometry and charge sequences never reuse prior call data."""
    numbers, pos_a = _sample("SiH4")
    pos_b = pos_a.clone()
    pos_b[0, 0] += 0.08
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None
    setup = setup_d4sc(interaction, numbers)
    q0 = torch.tensor([0.02, -0.01, -0.005, -0.002, -0.003], **DD)
    q1 = q0.clone()
    q1[0] += 0.1

    def evaluate(pos: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        data = build_d4sc_data(setup, pos)
        return (
            interaction.get_monopole_atom_energy(data, q).sum(),
            interaction.get_monopole_atom_potential(data, q),
        )

    a0 = evaluate(pos_a, q0)
    _ = evaluate(pos_b, q0)
    _ = evaluate(pos_a, q1)
    a1 = evaluate(pos_a.clone(), q0)
    torch.testing.assert_close(a0[0], a1[0])
    torch.testing.assert_close(a0[1], a1[1])

    pos_leaf = pos_a.clone().requires_grad_(True)
    first = evaluate(pos_leaf, q0)
    (prior_grad,) = torch.autograd.grad(first[0], pos_leaf)
    assert torch.isfinite(prior_grad).all()
    after_backward = evaluate(pos_a.clone(), q0)
    torch.testing.assert_close(a0[0], after_backward[0])
    torch.testing.assert_close(a0[1], after_backward[1])


def test_d4sc_position_jacfwd_and_potential_transforms() -> None:
    """Position energy and potential use the same differentiable builder."""
    numbers, positions = _sample("SiH4")
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None
    setup = setup_d4sc(interaction, numbers)
    charges = torch.tensor([0.05, -0.02, 0.01, -0.02, -0.02], **DD)

    def energy(pos: Tensor) -> Tensor:
        data = build_d4sc_data(setup, pos)
        return interaction.get_monopole_atom_energy(data, charges).sum()

    pos = positions.clone().requires_grad_(True)
    (reverse,) = torch.autograd.grad(energy(pos), pos)
    forward = torch.func.jacfwd(energy)(positions)
    torch.testing.assert_close(forward, reverse)

    charge_leaf = charges.clone().requires_grad_(True)
    charge_energy = lambda q: interaction.get_monopole_atom_energy(
        build_d4sc_data(setup, positions), q
    ).sum()
    (charge_reverse,) = torch.autograd.grad(
        charge_energy(charge_leaf), charge_leaf
    )
    charge_forward = torch.func.jacfwd(charge_energy)(charges)
    torch.testing.assert_close(charge_forward, charge_reverse)

    def potential(pos: Tensor) -> Tensor:
        data = build_d4sc_data(setup, pos)
        return interaction.get_monopole_atom_potential(data, charges)

    direction = torch.linspace(-0.2, 0.4, positions.numel(), **DD).reshape_as(
        positions
    )
    _, tangent = torch.func.jvp(potential, (positions,), (direction,))
    jac = torch.func.jacfwd(potential)(positions)
    torch.testing.assert_close(tangent, torch.tensordot(jac, direction, dims=2))

    changed = positions.clone()
    changed[0, 0] += 0.07
    conformers = torch.stack((positions, changed))
    mapped = torch.func.vmap(potential)(conformers)
    loop = torch.stack([potential(pos) for pos in conformers])
    torch.testing.assert_close(mapped, loop)


def test_exact_update_and_reset_are_rejected() -> None:
    """Exact D4SC terms cannot diverge from setup after construction."""
    numbers, _ = _sample()
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None
    interactions = InteractionList(interaction)

    with pytest.raises(RuntimeError, match="setup-derived"):
        interaction.update(param={})
    with pytest.raises(RuntimeError, match="setup-derived"):
        interaction.reset()
    with pytest.raises(RuntimeError, match="setup-derived"):
        interactions.update(interaction.label, param={})
    with pytest.raises(RuntimeError, match="setup-derived"):
        interactions.reset(interaction.label)
    interactions.reset_all()


def test_duplicate_d4sc_labels_are_rejected() -> None:
    """Label-keyed D4SC call data reject ambiguous component entries."""
    numbers, _ = _sample()
    first = new_d4sc(numbers, _parameters(), **DD)
    second = new_d4sc(numbers, _parameters(), **DD)
    assert first is not None and second is not None

    with pytest.raises(ValueError, match="Duplicate"):
        InteractionList(first, second)

    class D4SCExtension(DispersionD4SC):
        __slots__ = ()

    extension = D4SCExtension(
        first.param,
        first.model,
        first.rcov,
        first.r4r2,
        first.cutoff,
        first.counting_function,
        first.damping_function,
        **DD,
    )
    extension.label = first.label
    with pytest.raises(ValueError, match="Duplicate"):
        InteractionList(first, extension)

    with pytest.raises(ValueError, match="Duplicate"):
        Model(
            par=_parameters(),
            interaction=(first,),
            config=Config.create(exclude=("es2", "es3", "aes2")),
            auto_int_level=False,
        ).setup(numbers)


def test_subclass_stays_on_fresh_legacy_cache_path() -> None:
    """D4SC subclasses keep their update/reset and get_cache extension hooks."""
    numbers, positions = _sample()
    base = new_d4sc(numbers, _parameters(), **DD)
    assert base is not None

    class ScaleCache(InteractionCache):
        __slots__ = ("scale",)

        def __init__(self, scale: Tensor) -> None:
            super().__init__(**DD)
            self.scale = scale

    class D4SCExtension(DispersionD4SC):
        __slots__ = ("scale",)

        def get_cache(self, *, numbers=None, positions=None, ihelp=None, **_):
            assert numbers is not None and positions is not None
            return ScaleCache(self.scale.clone())

        def get_monopole_atom_energy(self, cache, qat, **_) -> Tensor:
            assert isinstance(cache, ScaleCache)
            return cache.scale * qat

    extension = D4SCExtension(
        base.param,
        base.model,
        base.rcov,
        base.r4r2,
        base.cutoff,
        base.counting_function,
        base.damping_function,
        **DD,
    )
    extension.label = base.label
    extension.scale = torch.tensor(1.0, **DD)
    system = Model(
        par=_parameters(),
        interaction=(extension,),
        config=Config.create(exclude=("d4sc", "es2", "es3", "aes2")),
        auto_int_level=False,
    ).setup(numbers)
    assert system.d4sc_setup is None
    assert system.d4sc_interaction is None
    q = torch.tensor([0.2, -0.1], **DD)

    first_data = _interaction_data(system, positions)[extension.label]
    first_energy = extension.get_monopole_atom_energy(first_data, q).sum()
    extension.update(scale=torch.tensor(2.0, **DD))
    second_data = _interaction_data(system, positions)[extension.label]
    second_energy = extension.get_monopole_atom_energy(second_data, q).sum()
    assert first_data is not second_data
    assert not torch.equal(first_energy, second_energy)

    extension.reset()
    system.interactions.reset_all()
    after_reset = _interaction_data(system, positions)[extension.label]
    torch.testing.assert_close(after_reset.scale, extension.scale)


def test_calculator_reset_preserves_exact_d4sc_setup() -> None:
    """Calculator reset skips exact D4SC setup authority and its graph."""
    numbers, positions = _sample()
    par = _parameters()
    a1 = par.get("dispersion.d4.a1")
    a1.requires_grad_(True)
    calc = Calculator(
        numbers,
        par,
        opts={"verbosity": 0, "exclude": ["es2", "es3", "aes2"]},
        auto_int_level=False,
        **DD,
    )
    setup = calc.system.d4sc_setup
    interaction = calc.system.d4sc_interaction
    assert setup is not None and interaction is not None
    calc.reset()
    assert calc.system.d4sc_setup is setup
    data = build_d4sc_data(setup, positions)
    q = torch.tensor([0.1, -0.1], **DD)
    energy = interaction.get_monopole_atom_energy(data, q).sum()
    (gradient,) = torch.autograd.grad(energy, a1)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_d4sc_setup_energy_matches_compatibility_formula() -> None:
    """Setup-based call data retain the established D4SC energy value."""
    numbers, positions = _sample("SiH4")
    interaction = new_d4sc(numbers, _parameters(), **DD)
    assert interaction is not None
    charges = torch.tensor([0.03, -0.02, -0.01, 0.01, -0.01], **DD)

    setup = setup_d4sc(interaction, numbers)
    actual = interaction.get_monopole_atom_energy(
        build_d4sc_data(setup, positions), charges
    )
    compatibility = interaction.get_monopole_atom_energy(
        interaction.get_cache(numbers=numbers, positions=positions), charges
    )
    torch.testing.assert_close(actual, compatibility)


def test_system_setup_carries_exact_replacement_parameters() -> None:
    """An exact custom D4SC replaces the excluded built-in contribution."""
    numbers, positions = _sample("SiH4")
    params = _parameters()
    custom = new_d4sc(numbers, params, **DD)
    assert custom is not None
    custom.param["a1"] = custom.param["a1"] + 0.2

    # Model stores additional interactions separately from the method terms.
    replaced = Model(
        par=params,
        interaction=(custom,),
        config=Config.create(exclude=("d4sc", "es2", "es3", "aes2")),
        auto_int_level=False,
    ).setup(numbers)
    assert replaced.d4sc_interaction is custom
    assert replaced.d4sc_setup is not None
    assert dict(replaced.d4sc_setup.parameters)["a1"] == custom.param["a1"]

    default = Model(
        par=_parameters(),
        config=Config.create(exclude=("es2", "es3", "aes2")),
        auto_int_level=False,
    ).setup(numbers)
    assert default.d4sc_setup is not None
    q = torch.tensor([0.03, -0.02, -0.01, 0.01, -0.01], **DD)
    energy_custom = custom.get_monopole_atom_energy(
        build_d4sc_data(replaced.d4sc_setup, positions), q
    ).sum()
    assert default.d4sc_interaction is not None
    energy_default = default.d4sc_interaction.get_monopole_atom_energy(
        build_d4sc_data(default.d4sc_setup, positions), q
    ).sum()
    assert not torch.allclose(energy_custom, energy_default)
