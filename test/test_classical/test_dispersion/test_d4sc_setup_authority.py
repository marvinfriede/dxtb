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
"""Tests that functional D4SC preparation uses the System setup snapshot."""

from __future__ import annotations

from typing import Any

import pytest
import tad_dftd4 as d4
import torch

import dxtb._src.calculators.singlepoint as singlepoint_module
from dxtb import GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.interactions.dispersion.d4sc import (
    D4SCCoreSetup,
    DispersionD4SC,
    _StaticList,
    _StaticMapping,
    new_d4sc,
)
from dxtb._src.components.interactions.list import InteractionList
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional.core import (
    CoreIndexData,
    replace_tensor_leaves,
    tensor_leaves,
)
from dxtb._src.scf.functional.data import SCFCharges, setup_scf_inputs
from dxtb._src.scf.functional.interactions import (
    _entry_energy,
    _potential_parts,
    prepare_functional_interactions,
)
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.typing import Tensor
from dxtb.config import Config

from .samples import samples

DEVICE = torch.device("cpu")
DD = {"device": DEVICE, "dtype": torch.double}


def _sample() -> tuple[Tensor, Tensor]:
    sample = samples["LiH"]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _system_with_custom_model(
    numbers: Tensor,
    *,
    scf_mode: int = labels.SCF_MODE_FULL,
    int_level: int | None = None,
) -> tuple[Model, DispersionD4SC, Tensor]:
    par = ParamModule(GFN2_XTB, **DD)
    interaction = new_d4sc(numbers, par, **DD)
    assert interaction is not None
    base = interaction.model

    class ScaledD4Model(d4.model.D4Model):
        def weight_references(self, *args, **kwargs):
            values = super().weight_references(*args, **kwargs)
            scale = self.scale
            if isinstance(values, tuple):
                return tuple(value * scale for value in values)
            return values * scale

    model = ScaledD4Model(
        numbers,
        ga=base.ga,
        gc=base.gc,
        wf=base.wf,
        ref_charges=base.ref_charges,
        rc6=base.rc6,
        **DD,
    )
    scale = torch.tensor(1.07, **DD, requires_grad=True)
    model.scale = scale
    model.extra_state = {"weights": [scale, (scale * 0.5, "custom")]}
    interaction.model = model

    config_options: dict[str, Any] = {
        "exclude": ("es2", "es3", "aes2", "d4sc"),
        "scf_mode": scf_mode,
        "mixer": labels.MIXER_ANDERSON,
    }
    if int_level is not None:
        config_options["int_level"] = int_level
    config = Config.create(**config_options)
    calculator = Model(
        par=par,
        config=config,
        interaction=(interaction,),
        auto_int_level=False,
    )
    return calculator, interaction, scale


def _functional_entry(
    system, positions: Tensor, interactions: InteractionList | None = None
):
    selected = system.interactions if interactions is None else interactions
    cache = _interaction_data(system, positions, interactions=selected)
    charge = positions.new_zeros((1,))
    spin = positions.new_zeros((1,))
    integrals, refocc, _ = build_integral_matrices(
        system.integral_setup,
        system.h0_setup,
        positions,
        charge,
    )
    n0, occupation, nel = get_refocc(refocc, charge, spin, system.ihelp)
    inputs = setup_scf_inputs(
        system.numbers,
        integrals,
        n0,
        occupation,
        nel,
        selected,
        cache,
        system.ihelp,
        system.config.scf,
    )
    return cache, inputs.interactions.entries[0]


def _assert_static_equal(left, right) -> None:
    if isinstance(left, Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, _StaticMapping):
        assert isinstance(right, _StaticMapping)
        assert tuple(key for key, _ in left.items) == tuple(
            key for key, _ in right.items
        )
        for (_, lhs), (_, rhs) in zip(left.items, right.items):
            _assert_static_equal(lhs, rhs)
    elif isinstance(left, _StaticList):
        assert isinstance(right, _StaticList)
        assert len(left.items) == len(right.items)
        for lhs, rhs in zip(left.items, right.items):
            _assert_static_equal(lhs, rhs)
    elif isinstance(left, tuple):
        assert isinstance(right, tuple)
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_static_equal(lhs, rhs)
    else:
        assert type(left) is type(right)
        assert left == right


def _assert_model_extra_equal(left, right) -> None:
    assert tuple(name for name, _ in left) == tuple(name for name, _ in right)
    for (_, lhs), (_, rhs) in zip(left, right):
        _assert_static_equal(lhs, rhs)


def _functional_energy_potential(
    entry, atomic_charges: Tensor
) -> tuple[Tensor, Tensor]:
    atom_map = torch.arange(
        atomic_charges.numel(), device=atomic_charges.device
    )
    index = CoreIndexData(
        shells_to_atom=atom_map,
        orbitals_to_shell=atom_map,
    )
    charges = SCFCharges(mono=atomic_charges)
    reduced_shells = index.reduce_orbital_to_shell(charges.mono)
    reduced_atoms = index.reduce_shell_to_atom(reduced_shells)
    energy = _entry_energy(entry, charges, reduced_shells, reduced_atoms).sum()
    _, potential, _, _ = _potential_parts(entry, charges, index)
    return energy, potential


def test_functional_d4sc_uses_system_owned_model_snapshot() -> None:
    """Component mutation after setup cannot alter an existing System."""
    numbers, positions = _sample()
    calculator, component, source_scale = _system_with_custom_model(numbers)
    system = calculator.setup(numbers)
    setup = system.d4sc_setup
    assert type(setup) is D4SCCoreSetup
    assert setup is not None
    assert system.config.scf.scf_mode == labels.SCF_MODE_FULL
    full_before = system.singlepoint(positions)
    assert full_before.status is not None

    first_cache, first_entry = _functional_entry(system, positions)
    first_setup = first_entry.option("setup")
    assert torch.equal(first_setup.model_wf, setup.model_wf)
    assert torch.equal(first_setup.model_rc6, setup.model_rc6)
    assert first_setup.model_type is setup.model_type
    assert first_setup.model_ref_charges == setup.model_ref_charges
    assert first_setup.counting_function is setup.counting_function
    assert first_setup.model_ga == setup.model_ga
    assert first_setup.model_gc == setup.model_gc
    _assert_model_extra_equal(first_setup.parameters, setup.parameters)
    _assert_model_extra_equal(first_setup.model_extra, setup.model_extra)
    state_leaves = tensor_leaves(first_entry)
    state_paths = [path for path, _ in state_leaves if "model_extra" in path]
    assert state_paths
    original_state_tensor = dict(state_leaves)[state_paths[0]]
    replaced_tensor = torch.tensor(9.0, **DD)
    replaced_entry = replace_tensor_leaves(
        first_entry, {state_paths[0]: replaced_tensor}
    )
    assert (
        dict(tensor_leaves(replaced_entry))[state_paths[0]] is replaced_tensor
    )
    assert (
        dict(tensor_leaves(first_entry))[state_paths[0]]
        is original_state_tensor
    )

    atomic_charges = torch.tensor([0.17, -0.17], **DD, requires_grad=True)
    first_energy, first_potential = _functional_energy_potential(
        first_entry, atomic_charges
    )
    first_scale_gradient = torch.autograd.grad(
        first_energy, source_scale, retain_graph=True
    )[0]
    assert torch.isfinite(first_scale_gradient)
    assert first_scale_gradient.abs() > 0
    (charge_gradient,) = torch.autograd.grad(
        first_energy, atomic_charges, retain_graph=True
    )
    torch.testing.assert_close(charge_gradient, first_potential)

    # Change model inputs that are read by the functional reference-weight
    # reconstruction, including a custom tensor-valued extension state.
    component.model.wf = component.model.wf + 0.4
    component.model.rc6 = component.model.rc6 + 0.3
    component.model.scale = component.model.scale + 0.2
    component.model.extra_state["weights"][0] = torch.tensor(4.0, **DD)

    second_cache, second_entry = _functional_entry(system, positions)
    second_setup = second_entry.option("setup")

    # Functional preparation must retain the original snapshot even though
    # the component's mutable model has changed.
    torch.testing.assert_close(second_setup.model_wf, setup.model_wf)
    torch.testing.assert_close(second_setup.model_rc6, setup.model_rc6)
    assert second_setup.model_type is setup.model_type
    assert second_setup.model_ref_charges == setup.model_ref_charges
    assert second_setup.counting_function is setup.counting_function
    _assert_model_extra_equal(second_setup.parameters, setup.parameters)
    _assert_model_extra_equal(second_setup.model_extra, setup.model_extra)

    second_energy, second_potential = _functional_energy_potential(
        second_entry, atomic_charges
    )
    torch.testing.assert_close(second_energy, first_energy)
    torch.testing.assert_close(second_potential, first_potential)
    assert system.d4sc_setup is setup
    assert (
        second_cache[component.label].model
        is not first_cache[component.label].model
    )
    geometry_b = positions.clone()
    geometry_b[0, 0] += 0.05
    system.singlepoint(geometry_b)
    full_after = system.singlepoint(positions)
    torch.testing.assert_close(full_after.energy, full_before.energy)
    torch.testing.assert_close(full_after.fenergy, full_before.fenergy)
    torch.testing.assert_close(
        full_after.charges.mono, full_before.charges.mono
    )
    torch.testing.assert_close(full_after.density, full_before.density)
    torch.testing.assert_close(
        full_after.potential.mono, full_before.potential.mono
    )

    # A later System captures the newly selected configuration, while the
    # original System continues to use its own snapshot.
    second_system = calculator.setup(numbers)
    assert second_system.d4sc_setup is not setup
    _, second_system_entry = _functional_entry(second_system, positions)
    second_system_setup = second_system_entry.option("setup")
    torch.testing.assert_close(second_system_setup.model_wf, component.model.wf)
    assert not torch.equal(second_system_setup.model_wf, setup.model_wf)
    second_model_extra = dict(second_system_setup.model_extra)
    torch.testing.assert_close(
        second_model_extra["scale"], component.model.scale
    )
    assert not torch.equal(
        second_model_extra["scale"], dict(setup.model_extra)["scale"]
    )

    # Direct compatibility callers without System authority keep the
    # historical behavior of snapshotting the supplied component.
    compatibility_cache = _interaction_data(system, positions)
    compatibility = prepare_functional_interactions(
        system.interactions, compatibility_cache, system.numbers
    )
    compatibility_setup = compatibility.entries[0].option("setup")
    torch.testing.assert_close(compatibility_setup.model_wf, component.model.wf)


def test_implicit_d4sc_uses_system_setup_and_preserves_gradients() -> None:
    """Implicit legacy evaluation keeps its System setup and graph edges."""
    numbers, positions = _sample()
    calculator, component, source_scale = _system_with_custom_model(
        numbers, scf_mode=labels.SCF_MODE_IMPLICIT
    )
    system = calculator.setup(numbers)
    positions = positions.clone().requires_grad_(True)
    before = system.singlepoint(positions)
    # System.singlepoint's implicit compatibility engine returns the legacy
    # mapping, whose optional convergence fields remain absent.
    assert before.status is None
    assert before.converged is None
    assert before.residual is None
    position_gradient, scale_gradient = torch.autograd.grad(
        before.energy.sum(), (positions, source_scale), retain_graph=True
    )
    assert torch.isfinite(position_gradient).all()
    assert torch.isfinite(scale_gradient)
    assert scale_gradient.abs() > 0

    component.model.wf = component.model.wf + 0.2
    component.model.scale = component.model.scale + 0.3
    after = system.singlepoint(positions.detach())
    torch.testing.assert_close(after.energy, before.energy.detach())
    torch.testing.assert_close(after.charges.mono, before.charges.mono.detach())


def test_alternate_d4sc_uses_its_own_setup_authority() -> None:
    """An alternate selected D4SC cannot inherit the System's setup."""
    numbers, positions = _sample()
    calculator, component_a, _ = _system_with_custom_model(numbers)
    system = calculator.setup(numbers)
    assert system.d4sc_interaction is component_a
    assert type(system.d4sc_setup) is D4SCCoreSetup

    component_b = new_d4sc(numbers, ParamModule(GFN2_XTB, **DD), **DD)
    assert component_b is not None
    component_b.model.wf = component_b.model.wf + 0.4
    selected = InteractionList(component_b)

    data = _interaction_data(system, positions, interactions=selected)
    cache = data[component_b.label]
    torch.testing.assert_close(cache.model.wf, component_b.model.wf)
    assert data.d4sc_setup is None

    selected_data, selected_entry = _functional_entry(
        system, positions, interactions=selected
    )
    assert selected_data.d4sc_setup is None
    selected_cache = selected_data[component_b.label]
    selected_model_setup = selected_entry.option("setup")
    torch.testing.assert_close(
        selected_model_setup.model_wf, component_b.model.wf
    )

    atomic_charges = torch.tensor([0.13, -0.13], **DD)
    functional_energy, functional_potential = _functional_energy_potential(
        selected_entry, atomic_charges
    )
    legacy_energy = component_b.get_monopole_atom_energy(
        selected_cache, atomic_charges
    ).sum()
    legacy_potential = component_b.get_monopole_atom_potential(
        selected_cache, atomic_charges
    )
    torch.testing.assert_close(functional_energy, legacy_energy)
    torch.testing.assert_close(functional_potential, legacy_potential)

    no_d4_data = _interaction_data(
        system, positions, interactions=InteractionList()
    )
    assert no_d4_data.d4sc_setup is None

    no_d4_calculator = Model(
        par=ParamModule(GFN2_XTB, **DD),
        config=Config.create(exclude=("es2", "es3", "aes2", "d4sc")),
        auto_int_level=False,
    )
    no_d4_system = no_d4_calculator.setup(numbers)
    assert no_d4_system.d4sc_interaction is None
    external_d4_data = _interaction_data(
        no_d4_system, positions, interactions=selected
    )
    assert external_d4_data.d4sc_setup is None


def test_per_call_field_preserves_d4sc_setup_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """singlepoint field additions keep the selected System D4SC setup."""
    numbers, positions = _sample()
    calculator, _, _ = _system_with_custom_model(
        numbers, int_level=labels.INTLEVEL_DIPOLE
    )
    system = calculator.setup(numbers)
    observed: list[object] = []
    original = singlepoint_module._interaction_data

    def capture_authority(system, positions, interactions=None):
        data = original(system, positions, interactions=interactions)
        observed.append(data.d4sc_setup)
        return data

    monkeypatch.setattr(
        singlepoint_module, "_interaction_data", capture_authority
    )
    system.singlepoint(positions, field=torch.tensor([0.01, 0.0, 0.0], **DD))
    assert len(observed) == 1
    assert observed[0] is system.d4sc_setup


def test_independent_systems_keep_d4sc_setup_authority() -> None:
    """Same-label D4SC components retain per-System setup provenance."""
    numbers, positions = _sample()
    calculator_a, component_a, _ = _system_with_custom_model(numbers)
    system_a = calculator_a.setup(numbers)

    component_b = new_d4sc(numbers, ParamModule(GFN2_XTB, **DD), **DD)
    assert component_b is not None
    component_b.model.wf = component_b.model.wf + 0.4
    calculator_b = Model(
        par=ParamModule(GFN2_XTB, **DD),
        config=Config.create(exclude=("es2", "es3", "aes2", "d4sc")),
        interaction=(component_b,),
        auto_int_level=False,
    )
    system_b = calculator_b.setup(numbers)

    assert system_a.d4sc_interaction is component_a
    assert system_b.d4sc_interaction is component_b
    assert system_a.d4sc_setup is not system_b.d4sc_setup
    _, entry_b = _functional_entry(system_b, positions)
    _, entry_a = _functional_entry(system_a, positions)
    torch.testing.assert_close(
        entry_b.option("setup").model_wf, component_b.model.wf
    )
    torch.testing.assert_close(
        entry_a.option("setup").model_wf, system_a.d4sc_setup.model_wf
    )
    _, entry_b_again = _functional_entry(system_b, positions)
    torch.testing.assert_close(
        entry_b_again.option("setup").model_wf,
        entry_b.option("setup").model_wf,
    )


def test_custom_d4sc_subclass_does_not_inherit_system_authority() -> None:
    """A custom D4SC subclass keeps its legacy cache route."""
    numbers, positions = _sample()
    calculator, _, _ = _system_with_custom_model(numbers)
    system = calculator.setup(numbers)
    exact = new_d4sc(numbers, ParamModule(GFN2_XTB, **DD), **DD)
    assert exact is not None

    class CustomD4SC(DispersionD4SC):
        pass

    custom = CustomD4SC(
        param=exact.param,
        model=exact.model,
        rcov=exact.rcov,
        r4r2=exact.r4r2,
        cutoff=exact.cutoff,
        counting_function=exact.counting_function,
        damping_function=exact.damping_function,
        device=DEVICE,
        dtype=torch.double,
    )
    selected = InteractionList(custom)
    data = _interaction_data(system, positions, interactions=selected)
    assert data.d4sc_setup is None
    assert data[custom.label].model is not custom.model
