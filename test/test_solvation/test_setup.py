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
"""Tests for explicit GeneralizedBorn setup and call-local data."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.utils import _pytree

from dxtb import Calculator, GFN1_XTB, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.interactions import InteractionList
from dxtb._src.components.interactions.solvation.alpb import (
    GeneralizedBorn,
    GeneralizedBornCache,
    GeneralizedBornCoreSetup,
    GeneralizedBornSetup,
    build_generalized_born_data,
    core_generalized_born_setup,
    new_solvation,
    setup_generalized_born,
)
from dxtb._src.components.interactions.container import Charges
from dxtb._src.param import Param
from dxtb._src.typing import Tensor

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


def _geometry() -> tuple[Tensor, Tensor]:
    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **DD)
    return numbers, positions


def _component(
    numbers: Tensor,
    *,
    dielectric: Tensor | float = 78.9,
    kernel: str = "p16",
    alpb: bool = True,
    **kwargs,
) -> GeneralizedBorn:
    return GeneralizedBorn(
        numbers, dielectric, kernel=kernel, alpb=alpb, **DD, **kwargs
    )


def _energy(
    setup: GeneralizedBornSetup, positions: Tensor, charges: Tensor
) -> Tensor:
    data = build_generalized_born_data(setup, positions)
    return 0.5 * (charges * (data.mat @ charges)).sum()


@pytest.mark.parametrize(("kernel", "alpb"), [("p16", True), ("still", False)])
def test_setup_builder_preserves_direct_energy(kernel: str, alpb: bool) -> None:
    """Both kernels match the direct component energy convention."""
    numbers, positions = _geometry()
    component = _component(numbers, kernel=kernel, alpb=alpb)
    setup = setup_generalized_born(component, numbers)
    charges = torch.tensor([0.2, -0.2], **DD)
    direct = component.get_monopole_atom_energy(
        component.get_cache(numbers=numbers, positions=positions), charges
    )
    built = component.get_monopole_atom_energy(
        build_generalized_born_data(setup, positions), charges
    )
    torch.testing.assert_close(built, direct)


def test_setup_is_frozen_and_call_data_is_fresh() -> None:
    """Setup contains only static values and data is not retained."""
    numbers, positions = _geometry()
    component = _component(numbers)
    setup = setup_generalized_born(component, numbers)
    assert isinstance(setup, GeneralizedBornSetup)
    assert {field.name for field in fields(setup)} == {
        "numbers",
        "kernel",
        "alpbet",
        "keps",
        "rvdw",
        "born_scale",
        "born_offset",
        "cutoff",
        "descreening",
        "obc",
        "apply_alpb_shape_correction",
    }
    assert not {"positions", "born", "radii", "mat", "charges"} & {
        field.name for field in fields(setup)
    }
    assert not any(
        isinstance(
            getattr(setup, field.name),
            (
                dict,
                Param,
                ParamModule,
                Model,
                GeneralizedBorn,
                GeneralizedBornCache,
                InteractionList,
            ),
        )
        for field in fields(setup)
    )
    with pytest.raises(FrozenInstanceError):
        setup.kernel = "still"  # type: ignore[misc]

    first = component.get_cache(numbers=numbers, positions=positions)
    second = component.get_cache(numbers=numbers, positions=positions)
    assert isinstance(first, GeneralizedBornCache)
    assert first is not second
    assert component.cache is None
    assert component._cachevars is None
    assert component._cachegrad is None

    core = core_generalized_born_setup(setup)
    assert isinstance(core, Node)
    assert type(core) is GeneralizedBornCoreSetup
    leaves, spec = _pytree.tree_flatten(core)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is GeneralizedBornCoreSetup
    assert all(a is b for a, b in zip(leaves, _pytree.tree_leaves(rebuilt)))
    converted = core.to(dtype=torch.float32)
    assert converted.keps.dtype == torch.float32
    assert converted.numbers.dtype == core.numbers.dtype
    meta = core.to(device="meta")
    assert meta.keps.device.type == "meta"
    assert meta.numbers.dtype == core.numbers.dtype
    stacked = stack((core, core))
    assert type(stacked) is GeneralizedBornCoreSetup
    assert stacked.keps.shape[0] == 2
    torch.testing.assert_close(
        build_generalized_born_data(core, positions).mat,
        build_generalized_born_data(setup, positions).mat,
    )


def test_position_transforms_rebuild_call_data() -> None:
    """Position derivatives and vmap use the same setup-based builder."""
    numbers, positions = _geometry()
    setup = setup_generalized_born(_component(numbers), numbers)
    charges = torch.tensor([0.2, -0.2], **DD)
    direction = torch.tensor([[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0]], **DD)

    def energy(pos: Tensor) -> Tensor:
        return _energy(setup, pos, charges)

    position = positions.clone().requires_grad_(True)
    value = energy(position)
    (first,) = torch.autograd.grad(value, position, create_graph=True)
    first_projection = (first * direction).sum()
    (second,) = torch.autograd.grad(
        first_projection, position, create_graph=True
    )
    second_projection = (second * direction).sum()
    (third,) = torch.autograd.grad(second_projection, position)
    assert all(torch.isfinite(value).all() for value in (first, second, third))

    _, tangent = torch.func.jvp(energy, (positions,), (direction,))
    assert torch.isfinite(tangent)
    jac = torch.func.jacfwd(energy)(positions)
    leaf = positions.clone().requires_grad_(True)
    (reverse_grad,) = torch.autograd.grad(energy(leaf), leaf)
    torch.testing.assert_close(jac, reverse_grad)
    torch.testing.assert_close(tangent, (reverse_grad * direction).sum())

    positions_b = positions.clone()
    positions_b[1, 0] = 0.12
    conformers = torch.stack((positions, positions_b))
    mapped = torch.func.vmap(energy)(conformers)
    loop = torch.stack([energy(pos) for pos in conformers])
    torch.testing.assert_close(mapped, loop)


def test_charge_potential_and_dielectric_gradients() -> None:
    """Charge potential and setup preserve dielectric autograd."""
    numbers, positions = _geometry()
    dielectric = torch.tensor(78.9, **DD, requires_grad=True)
    component = _component(numbers, dielectric=dielectric)
    setup = setup_generalized_born(component, numbers)
    charges = torch.tensor([0.2, -0.2], **DD, requires_grad=True)
    data = build_generalized_born_data(setup, positions)
    energy = component.get_monopole_atom_energy(data, charges).sum()
    charge_gradient, dielectric_gradient = torch.autograd.grad(
        energy, (charges, dielectric)
    )
    potential = component.get_monopole_atom_potential(data, charges)
    torch.testing.assert_close(charge_gradient, potential)
    assert torch.isfinite(dielectric_gradient)
    assert torch.count_nonzero(dielectric_gradient) > 0

    def charge_energy(q: Tensor) -> Tensor:
        return _energy(setup, positions, q)

    charge_direction = torch.tensor([0.1, -0.2], **DD)
    _, charge_tangent = torch.func.jvp(
        charge_energy, (charges.detach(),), (charge_direction,)
    )
    charge_jacobian = torch.func.jacfwd(charge_energy)(charges.detach())
    torch.testing.assert_close(
        charge_tangent, charge_jacobian @ charge_direction
    )
    charge_batch = torch.stack((charges.detach(), charges.detach() + 0.01))
    charge_mapped = torch.func.vmap(charge_energy)(charge_batch)
    charge_loop = torch.stack([charge_energy(q) for q in charge_batch])
    torch.testing.assert_close(charge_mapped, charge_loop)


def test_mixed_position_dielectric_derivative() -> None:
    """Position gradients remain connected to dielectric configuration."""
    numbers, positions = _geometry()
    positions = positions.clone().requires_grad_(True)
    dielectric = torch.tensor(78.9, **DD, requires_grad=True)
    component = _component(numbers, dielectric=dielectric)
    setup = setup_generalized_born(component, numbers)
    charges = torch.tensor([0.2, -0.2], **DD)
    direction = torch.tensor([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]], **DD)

    energy = _energy(setup, positions, charges)
    position_gradient = torch.autograd.grad(
        energy, positions, create_graph=True
    )[0]
    projected_gradient = (position_gradient * direction).sum()
    mixed = torch.autograd.grad(projected_gradient, dielectric)[0]

    assert torch.isfinite(mixed)
    assert torch.count_nonzero(mixed) > 0


def test_potential_transforms_and_born_radii_transforms() -> None:
    """Potential and Born-radii paths support forward transforms and vmap."""
    from dxtb._src.components.interactions.solvation.born import (
        get_born_radii,
    )

    numbers, positions = _geometry()
    setup = setup_generalized_born(_component(numbers), numbers)
    charges = torch.tensor([0.2, -0.2], **DD)

    def potential(pos: Tensor, q: Tensor = charges) -> Tensor:
        data = build_generalized_born_data(setup, pos)
        return data.mat @ q

    direction = torch.tensor([[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0]], **DD)
    _, tangent = torch.func.jvp(
        lambda pos: potential(pos), (positions,), (direction,)
    )
    jac = torch.func.jacfwd(lambda pos: potential(pos))(positions)
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(jac).all()
    moved = positions.clone()
    moved[1, 0] = 0.1
    conformers = torch.stack((positions, moved))
    mapped = torch.func.vmap(lambda pos: potential(pos))(conformers)
    loop = torch.stack([potential(pos) for pos in conformers])
    torch.testing.assert_close(mapped, loop)

    radii = lambda pos: get_born_radii(numbers, pos)
    _, radii_tangent = torch.func.jvp(radii, (positions,), (direction,))
    radii_jacobian = torch.func.jacfwd(radii)(positions)
    assert torch.isfinite(radii_tangent).all()
    assert torch.isfinite(radii_jacobian).all()
    radii_mapped = torch.func.vmap(radii)(conformers)
    radii_loop = torch.stack([radii(pos) for pos in conformers])
    torch.testing.assert_close(radii_mapped, radii_loop)

    def charge_potential(q: Tensor) -> Tensor:
        return build_generalized_born_data(setup, positions).mat @ q

    q_direction = torch.tensor([0.1, -0.2], **DD)
    _, q_tangent = torch.func.jvp(charge_potential, (charges,), (q_direction,))
    q_jacobian = torch.func.jacfwd(charge_potential)(charges)
    torch.testing.assert_close(q_tangent, q_jacobian @ q_direction)
    q_batch = torch.stack((charges, charges + 0.01))
    torch.testing.assert_close(
        torch.func.vmap(charge_potential)(q_batch),
        torch.stack([charge_potential(q) for q in q_batch]),
    )


def test_system_setup_bypasses_cache_and_freezes_configuration(
    monkeypatch,
) -> None:
    """Core interaction data comes from setup, not the component cache."""
    numbers, positions = _geometry()
    component = _component(numbers)
    model = Model(
        par=ParamModule(GFN1_XTB, **DD),
        interaction=(component,),
        auto_int_level=False,
    )
    system = model.setup(numbers)
    setup = system.generalized_born_setup
    assert setup is not None

    def fail_get_cache(*args, **kwargs):
        raise AssertionError("exact GeneralizedBorn.get_cache was called")

    monkeypatch.setattr(GeneralizedBorn, "get_cache", fail_get_cache)
    data = _interaction_data(system, positions)
    assert isinstance(data[component.label], GeneralizedBornCache)
    assert torch.isfinite(data[component.label].mat).all()
    baseline_energy = system.singlepoint(positions).energy.clone()

    component.kernel = "still"
    component.keps = component.keps + 0.5
    same_system_data = _interaction_data(system, positions)
    torch.testing.assert_close(
        data[component.label].mat, same_system_data[component.label].mat
    )
    assert system.generalized_born_setup is setup
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()
    torch.testing.assert_close(result.energy, baseline_energy)

    changed = Model(
        par=model.par,
        interaction=(component,),
        auto_int_level=model.auto_int_level,
    ).setup(numbers)
    assert changed.generalized_born_setup is not None
    assert changed.generalized_born_setup.kernel == "still"
    assert not torch.allclose(
        build_generalized_born_data(
            changed.generalized_born_setup, positions
        ).mat,
        data[component.label].mat,
    )


def test_setup_preserves_tensor_born_configuration_gradients() -> None:
    """Optional tensor radii settings remain connected through setup."""
    numbers, positions = _geometry()
    rvdw = torch.tensor([1.4, 1.5], **DD, requires_grad=True)
    scale = torch.tensor(1.0, **DD, requires_grad=True)
    offset = torch.tensor(0.1, **DD, requires_grad=True)
    component = _component(
        numbers, rvdw=rvdw, born_scale=scale, born_offset=offset
    )
    setup = setup_generalized_born(component, numbers)
    charges = torch.tensor([0.2, -0.2], **DD)
    energy = _energy(setup, positions, charges)
    gradients = torch.autograd.grad(energy, (rvdw, scale, offset))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_fresh_setup_dielectric_transforms() -> None:
    """Fresh component and setup construction preserve dielectric transforms."""
    numbers, positions = _geometry()
    charges = torch.tensor([0.2, -0.2], **DD)

    def energy_from_dielectric(eps: Tensor) -> Tensor:
        component = _component(numbers, dielectric=eps)
        setup = setup_generalized_born(component, numbers)
        return _energy(setup, positions, charges)

    dielectric = torch.tensor(78.9, **DD)
    direction = torch.tensor(0.37, **DD)
    reverse = torch.func.grad(energy_from_dielectric)(dielectric)
    _, tangent = torch.func.jvp(
        energy_from_dielectric, (dielectric,), (direction,)
    )
    jacobian = torch.func.jacfwd(energy_from_dielectric)(dielectric)
    step = 1.0e-4
    finite_difference = (
        energy_from_dielectric(dielectric + step)
        - energy_from_dielectric(dielectric - step)
    ) / (2 * step)

    assert torch.isfinite(reverse)
    assert torch.count_nonzero(reverse) > 0
    torch.testing.assert_close(tangent, reverse * direction)
    torch.testing.assert_close(jacobian, reverse)
    torch.testing.assert_close(
        finite_difference, reverse, atol=1.0e-9, rtol=1.0e-5
    )


def test_fresh_setup_tensor_born_configuration_jvp() -> None:
    """Fresh setup construction preserves tensor Born-scale differentiation."""
    numbers, positions = _geometry()
    charges = torch.tensor([0.2, -0.2], **DD)

    def energy_from_scale(scale: Tensor) -> Tensor:
        component = _component(numbers, born_scale=scale)
        setup = setup_generalized_born(component, numbers)
        return _energy(setup, positions, charges)

    scale = torch.tensor(1.0, **DD)
    direction = torch.tensor(0.25, **DD)
    reverse = torch.func.grad(energy_from_scale)(scale)
    _, tangent = torch.func.jvp(energy_from_scale, (scale,), (direction,))
    torch.testing.assert_close(tangent, reverse * direction)
    assert torch.isfinite(reverse)
    assert torch.count_nonzero(reverse) > 0


def test_same_position_leaf_can_be_differentiated_repeatedly() -> None:
    """Fresh per-call matrices allow repeated force evaluation on one leaf."""
    numbers, positions = _geometry()
    setup = setup_generalized_born(_component(numbers), numbers)
    charges = torch.tensor([0.2, -0.2], **DD)
    leaf = positions.clone().requires_grad_(True)
    gradients = []
    for _ in range(2):
        (gradient,) = torch.autograd.grad(_energy(setup, leaf, charges), leaf)
        gradients.append(gradient)
    fresh_leaf = positions.clone().requires_grad_(True)
    (fresh_gradient,) = torch.autograd.grad(
        _energy(
            setup_generalized_born(_component(numbers), numbers),
            fresh_leaf,
            charges,
        ),
        fresh_leaf,
    )
    torch.testing.assert_close(gradients[0], fresh_gradient)
    torch.testing.assert_close(gradients[1], fresh_gradient)


def test_core_repeated_force_matches_fresh_system() -> None:
    """Repeated core force evaluations reuse only static System setup."""
    numbers, positions = _geometry()
    component = _component(numbers)
    model = Model(
        par=ParamModule(GFN1_XTB, **DD),
        interaction=(component,),
    )
    system = model.setup(numbers)
    leaf = positions.clone().requires_grad_(True)
    gradients = []
    for _ in range(2):
        result = system.singlepoint(leaf)
        (gradient,) = torch.autograd.grad(result.energy.sum(), leaf)
        gradients.append(gradient)

    fresh_system = model.setup(numbers)
    fresh_leaf = positions.clone().requires_grad_(True)
    fresh_result = fresh_system.singlepoint(fresh_leaf)
    (fresh_gradient,) = torch.autograd.grad(
        fresh_result.energy.sum(), fresh_leaf
    )
    torch.testing.assert_close(gradients[0], fresh_gradient)
    torch.testing.assert_close(gradients[1], fresh_gradient)


def test_padded_ghost_has_finite_zero_contribution_and_gradient() -> None:
    """Padded atoms stay finite and contribute no solvation energy."""
    numbers = torch.tensor([1, 1, 0], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, 0.0]], **DD
    ).requires_grad_(True)
    setup = setup_generalized_born(_component(numbers), numbers)
    data = build_generalized_born_data(setup, positions)
    charges = torch.tensor([0.2, -0.2, 0.0], **DD)
    energies = 0.5 * charges * (data.mat @ charges)
    (gradient,) = torch.autograd.grad(energies.sum(), positions)
    assert torch.isfinite(data.mat).all()
    assert torch.isfinite(energies).all()
    assert torch.isfinite(gradient).all()
    assert energies[-1] == 0
    torch.testing.assert_close(gradient[-1], torch.zeros_like(gradient[-1]))


def test_exact_mutation_rejected_and_duplicates_rejected() -> None:
    """Exact setup-derived settings are immutable and labels stay unique."""
    numbers, _ = _geometry()
    component = _component(numbers)
    with pytest.raises(RuntimeError, match="setup-derived"):
        component.update(kernel="still")
    with pytest.raises(RuntimeError, match="setup-derived"):
        component.reset()
    with pytest.raises(RuntimeError, match="setup-derived"):
        InteractionList(component).update(component.label, kernel="still")
    with pytest.raises(RuntimeError, match="setup-derived"):
        InteractionList(component).reset(component.label)
    with pytest.raises(ValueError, match="GeneralizedBorn"):
        InteractionList(component, _component(numbers))

    calculator = Calculator(
        numbers, GFN1_XTB, interaction=[component], dtype=torch.double
    )
    setup = calculator.system.generalized_born_setup
    calculator.reset()
    assert calculator.system.generalized_born_setup is setup


def test_python_dielectric_uses_requested_dtype() -> None:
    """Python numeric dielectric values follow the requested tensor dtype."""
    numbers, _ = _geometry()
    component = GeneralizedBorn(numbers, 78.9, dtype=torch.double)
    assert component.alpbet.dtype == torch.double
    assert component.keps.dtype == torch.double

    raw = GFN1_XTB.model_dump()
    raw["solvation"] = {
        "alpb": {
            "alpb": True,
            "kernel": "p16",
            "born_scale": 1.0,
            "born_offset": 0.0,
        }
    }
    parameters = Param.model_validate(raw)
    module = ParamModule(parameters, dtype=torch.double)
    from_module = new_solvation(numbers, module, 78.9, dtype=torch.double)
    assert from_module is not None
    assert from_module.alpbet.dtype == torch.double


def test_geometry_charge_and_backward_history_is_independent() -> None:
    """A/B/A, charge changes and backward leave no call data behind."""
    numbers, pos_a = _geometry()
    pos_b = pos_a.clone()
    pos_b[1, 0] = 0.2
    setup = setup_generalized_born(_component(numbers), numbers)
    q0 = torch.tensor([0.2, -0.2], **DD)
    q1 = torch.tensor([0.1, -0.1], **DD)
    first = _energy(setup, pos_a, q0)
    second = _energy(setup, pos_b, q0)
    charged = _energy(setup, pos_a, q1)
    repeated = _energy(setup, pos_a.clone(), q0)
    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, second)
    assert not torch.allclose(first, charged)

    leaf = pos_b.clone().requires_grad_(True)
    previous = _energy(setup, leaf, q0)
    torch.autograd.grad(previous, leaf)
    torch.testing.assert_close(_energy(setup, pos_a, q0), first)


def test_subclass_uses_legacy_cache_update_and_reset() -> None:
    """Unmigrated subclasses keep cache-driven extension behavior."""
    numbers, positions = _geometry()

    class ScaledGeneralizedBorn(GeneralizedBorn):
        __slots__ = ["scale"]

        def __init__(self) -> None:
            super().__init__(numbers, 78.9, **DD)
            self.scale = torch.tensor(1.0, **DD)

        def get_cache(self, *, numbers=None, positions=None, ihelp=None):
            return (
                super().get_cache(
                    numbers=numbers, positions=positions, ihelp=ihelp
                ),
                self.scale.clone(),
            )

        def get_monopole_atom_potential(self, cache, qat, *args, **kwargs):
            base, scale = cache
            return scale * (base.mat @ qat)

        def get_monopole_atom_energy(self, cache, qat, **kwargs):
            return 0.5 * qat * self.get_monopole_atom_potential(cache, qat)

    component = ScaledGeneralizedBorn()
    system = Model(
        par=ParamModule(GFN1_XTB, **DD),
        interaction=(component,),
        auto_int_level=False,
    ).setup(numbers)
    first_data = _interaction_data(system, positions)
    charges = torch.tensor([0.2, -0.1, -0.05, -0.05], **DD)
    first = component.get_energy(
        first_data[component.label], Charges(mono=charges), system.ihelp
    )
    component.update(scale=torch.tensor(2.0, **DD))
    second_data = _interaction_data(system, positions)
    second = component.get_energy(
        second_data[component.label], Charges(mono=charges), system.ihelp
    )
    assert not torch.allclose(first, second)
    component.reset()
    second_after_reset_data = _interaction_data(system, positions)
    second_after_reset = component.get_energy(
        second_after_reset_data[component.label],
        Charges(mono=charges),
        system.ihelp,
    )
    torch.testing.assert_close(second, second_after_reset)

    component.update(scale=torch.tensor(3.0, **DD))
    scale_before_reset_all = component.scale
    InteractionList(component).reset_all()
    assert component.scale is not scale_before_reset_all
    third_data = _interaction_data(system, positions)
    third = component.get_energy(
        third_data[component.label], Charges(mono=charges), system.ihelp
    )
    torch.testing.assert_close(first * 3, third)

    calculator_component = ScaledGeneralizedBorn()
    calculator = Calculator(
        numbers,
        ParamModule(GFN1_XTB, **DD),
        interaction=(calculator_component,),
        opts={"verbosity": 0},
        dtype=torch.double,
        auto_int_level=False,
    )
    calculator_component.update(scale=torch.tensor(3.0, **DD))
    calculator_data = _interaction_data(calculator.system, positions)
    calculator_energy_before_reset = calculator_component.get_energy(
        calculator_data[calculator_component.label],
        Charges(mono=charges),
        calculator.system.ihelp,
    )
    scale_before_calculator_reset = calculator_component.scale
    calculator.reset()
    assert calculator_component.scale is not scale_before_calculator_reset
    reset_data = _interaction_data(calculator.system, positions)
    reset_energy = calculator_component.get_energy(
        reset_data[calculator_component.label],
        Charges(mono=charges),
        calculator.system.ihelp,
    )
    torch.testing.assert_close(calculator_energy_before_reset, reset_energy)


def test_core_node_mixed_dtype_device_and_context_edge_cases() -> None:
    """Core GeneralizedBorn preserves mixed dtypes, scalar device normalization, and static context."""
    numbers, positions = _geometry()
    scale = torch.tensor(0.95, device=DEVICE, dtype=torch.float32, requires_grad=True)
    offset = torch.tensor(0.08, device=DEVICE, dtype=torch.float32, requires_grad=True)
    descreening = torch.tensor(
        [0.72, 0.74], device=DEVICE, dtype=torch.float32, requires_grad=True
    )
    component = _component(
        numbers,
        born_scale=scale,
        born_offset=offset,
    )
    legacy = replace(
        setup_generalized_born(component, numbers),
        descreening=descreening,
    )
    core = core_generalized_born_setup(legacy)
    assert core.keps.dtype == torch.float64
    assert core.born_scale.dtype == torch.float32
    assert core.born_offset.dtype == torch.float32
    assert core.descreening.dtype == torch.float32
    converted = core.to(dtype=torch.float64)
    assert converted.born_scale.dtype == torch.float64
    assert converted.born_offset.dtype == torch.float64
    assert converted.descreening.dtype == torch.float64
    charges = torch.tensor([0.2, -0.2], **DD)
    energy = _energy(core, positions, charges)
    grads = torch.autograd.grad(energy, (scale, offset, descreening))
    assert all(torch.isfinite(g).all() and (g.abs() > 0).any() for g in grads)

    base_setup = setup_generalized_born(_component(numbers), numbers)
    meta_fields = {
        name: getattr(base_setup, name).to(device="meta")
        for name in ("numbers", "alpbet", "keps", "rvdw", "cutoff", "obc")
    }
    meta_setup = replace(
        base_setup,
        **meta_fields,
        born_scale=torch.tensor(0.95, device="cpu", dtype=torch.double),
        born_offset=torch.tensor(0.08, device="cpu", dtype=torch.double),
    )
    meta_core = core_generalized_born_setup(meta_setup)
    assert meta_core.born_scale.device.type == "meta"
    assert meta_core.born_offset.device.type == "meta"
    with pytest.raises(RuntimeError, match="different devices"):
        core_generalized_born_setup(
            replace(
                meta_setup,
                rvdw=torch.tensor([1.2, 1.2], device="cpu", dtype=torch.double),
            )
        )

    still_core = core_generalized_born_setup(
        setup_generalized_born(_component(numbers, kernel="still"), numbers)
    )
    no_alpb_core = core_generalized_born_setup(
        setup_generalized_born(_component(numbers, alpb=False), numbers)
    )
    core_spec = _pytree.tree_flatten(core)[1]
    assert core_spec != _pytree.tree_flatten(still_core)[1]
    assert core_spec != _pytree.tree_flatten(no_alpb_core)[1]
    assert (
        Model(par=ParamModule(GFN1_XTB, **DD)).setup(numbers).generalized_born_setup
        is None
    )
