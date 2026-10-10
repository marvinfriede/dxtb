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
"""Explicit setup, local geometry data, and transforms for AES2."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN2_XTB, Calculator, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.interactions.container import Charges
from dxtb._src.components.interactions.coulomb import (
    AES2,
    AES2Setup,
    build_aes2_data,
    new_aes2,
    setup_aes2,
)
from dxtb._src.components.interactions.coulomb.multipole import AES2Cache
from dxtb._src.components.interactions.list import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.param import Param
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ..conftest import DEVICE
from ..molecules import mols

DD = {"device": DEVICE, "dtype": torch.float64}


def _inputs() -> tuple[Tensor, Tensor, ParamModule, IndexHelper, AES2]:
    numbers = mols["H2O"]["numbers"].clone().to(DEVICE)
    positions = mols["H2O"]["positions"].clone().to(**DD)
    parameters = ParamModule(GFN2_XTB, **DD)
    ihelp = IndexHelper.from_numbers(numbers, parameters)
    interaction = new_aes2(torch.unique(numbers), parameters, **DD)
    assert interaction is not None
    return numbers, positions, parameters, ihelp, interaction


def _charges(ihelp: IndexHelper, numbers: Tensor) -> Charges:
    """Create deterministic monopole and multipole inputs."""
    nat = numbers.numel()
    return Charges(
        mono=torch.linspace(0.1, 0.6, ihelp.nao, **DD),
        dipole=torch.tensor(
            [[0.2, -0.1, 0.3], [-0.1, 0.4, 0.2], [0.3, 0.2, -0.2]],
            **DD,
        )[:nat],
        quad=torch.arange(1, nat * 6 + 1, **DD).reshape(nat, 6) * 0.03,
    )


def _energy(
    setup: AES2Setup,
    positions: Tensor,
    interaction: AES2,
    charges: Charges,
    ihelp: IndexHelper,
) -> Tensor:
    return interaction.get_energy(
        build_aes2_data(setup, positions), charges, ihelp
    ).sum()


def test_setup_is_frozen_and_call_data_is_fresh() -> None:
    """Setup is structural while geometry matrices are freshly built."""
    numbers, positions, _, ihelp, interaction = _inputs()
    setup = setup_aes2(interaction, numbers, ihelp)
    assert isinstance(setup, AES2Setup)
    assert {field.name for field in fields(setup)} == {
        "numbers",
        "dmp3",
        "dmp5",
        "shift",
        "kexp",
        "rmax",
        "dkernel",
        "qkernel",
        "rad",
        "vcn",
        "pair_mask",
    }
    forbidden = (Param, ParamModule, Model, AES2, AES2Cache, InteractionList)
    assert not any(
        isinstance(getattr(setup, field.name), forbidden)
        for field in fields(setup)
    )
    assert setup.dkernel.shape == numbers.shape
    assert setup.qkernel.shape == numbers.shape
    assert setup.rad.shape == numbers.shape
    assert setup.vcn.shape == numbers.shape
    assert setup.pair_mask.shape == (numbers.numel(), numbers.numel())
    assert not {"positions", "cn", "mrad", "amat_sd", "amat_dd", "amat_sq"} & {
        field.name for field in fields(setup)
    }
    with pytest.raises(FrozenInstanceError):
        setup.dmp3 = torch.tensor(2.0, **DD)  # type: ignore[misc]

    first = build_aes2_data(setup, positions)
    second = build_aes2_data(setup, positions)
    assert first is not second
    assert first.amat_sd.data_ptr() != second.amat_sd.data_ptr()
    assert interaction.cache is None
    assert interaction._cachevars is None
    assert interaction._cachegrad is None
    torch.testing.assert_close(first.amat_sd, second.amat_sd)


def test_aes2_positions_supports_reverse_and_forward_transforms() -> None:
    """Fresh geometry data supports reverse, JVP, jacfwd, and vmap."""
    numbers, positions, _, ihelp, interaction = _inputs()
    setup = setup_aes2(interaction, numbers, ihelp)
    charges = _charges(ihelp, numbers)

    def energy(pos: Tensor) -> Tensor:
        return _energy(setup, pos, interaction, charges, ihelp)

    leaf = positions.clone().requires_grad_(True)
    first = energy(leaf)
    (gradient,) = torch.autograd.grad(first, leaf, create_graph=True)
    assert torch.isfinite(gradient).all()
    (second,) = torch.autograd.grad(gradient.sum(), leaf, create_graph=True)
    assert torch.isfinite(second).all()
    (third,) = torch.autograd.grad(second.sum(), leaf)
    assert torch.isfinite(third).all()

    direction = torch.linspace(-0.2, 0.3, positions.numel(), **DD).reshape_as(
        positions
    )
    _, tangent = jvp(energy, (positions,), (direction,))
    torch.testing.assert_close(tangent, (gradient * direction).sum())
    torch.testing.assert_close(jacfwd(energy)(positions), gradient.detach())

    moved = positions.clone()
    moved[1, 2] += 0.08
    conformers = torch.stack((positions, moved))
    mapped = vmap(energy)(conformers)
    looped = torch.stack([energy(pos) for pos in conformers])
    torch.testing.assert_close(mapped, looped)
    assert not torch.isclose(mapped[0], mapped[1])


def test_aes2_potential_transforms_include_interaction_list() -> None:
    """AES2 potential is transform-safe through its list container."""
    numbers, positions, _, ihelp, interaction = _inputs()
    setup = setup_aes2(interaction, numbers, ihelp)
    charges = _charges(ihelp, numbers)
    interactions = InteractionList(interaction)

    def potential(pos: Tensor) -> Tensor:
        call_data = InteractionListCache()
        call_data[interaction.label] = build_aes2_data(setup, pos)
        value = interactions.get_potential(call_data, charges, ihelp)
        assert value.mono is not None
        assert value.dipole is not None
        assert value.quad is not None
        return torch.cat(
            (
                value.mono.reshape(-1),
                value.dipole.reshape(-1),
                value.quad.reshape(-1),
            )
        ).sum()

    leaf = positions.clone().requires_grad_(True)
    (reverse,) = torch.autograd.grad(potential(leaf), leaf)
    _, tangent = jvp(potential, (positions,), (torch.ones_like(positions),))
    torch.testing.assert_close(tangent, reverse.sum())
    torch.testing.assert_close(jacfwd(potential)(positions), reverse)
    moved = positions.clone()
    moved[1, 0] += 0.05
    conformers = torch.stack((positions, moved))
    torch.testing.assert_close(
        vmap(potential)(conformers),
        torch.stack([potential(pos) for pos in conformers]),
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "multipole.damped.dmp3",
        "multipole.damped.dmp5",
        "multipole.damped.shift",
        "multipole.damped.kexp",
        "multipole.damped.rmax",
        "element.O.dkernel",
        "element.O.qkernel",
        "element.O.mprad",
        "element.O.mpvcn",
    ],
)
def test_setup_energy_keeps_parameter_leaf_gradient(parameter: str) -> None:
    """AES2 setup preserves gradients to scalar and element parameter leaves."""
    numbers, positions, parameters, ihelp, _ = _inputs()
    leaf = parameters.get(parameter)
    leaf.requires_grad_(True)
    interaction = new_aes2(torch.unique(numbers), parameters, **DD)
    assert interaction is not None
    setup = setup_aes2(interaction, numbers, ihelp)
    energy = _energy(
        setup, positions, interaction, _charges(ihelp, numbers), ihelp
    )
    (gradient,) = torch.autograd.grad(energy, leaf)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_fresh_setup_parameter_gradient_and_mixed_derivative() -> None:
    """Fresh parameter setups repeat gradients and compose position AD."""
    numbers, positions, parameters, ihelp, _ = _inputs()
    leaf = parameters.get("multipole.damped.dmp3")
    leaf.requires_grad_(True)
    charges = _charges(ihelp, numbers)
    gradients = []
    for _ in range(2):
        interaction = new_aes2(torch.unique(numbers), parameters, **DD)
        assert interaction is not None
        setup = setup_aes2(interaction, numbers, ihelp)
        energy = _energy(setup, positions, interaction, charges, ihelp)
        (gradient,) = torch.autograd.grad(energy, leaf)
        gradients.append(gradient)
    torch.testing.assert_close(gradients[0], gradients[1])

    interaction = new_aes2(torch.unique(numbers), parameters, **DD)
    assert interaction is not None
    setup = setup_aes2(interaction, numbers, ihelp)
    pos = positions.clone().requires_grad_(True)
    energy = _energy(setup, pos, interaction, charges, ihelp)
    (position_gradient,) = torch.autograd.grad(energy, pos, create_graph=True)
    direction = torch.arange(positions.numel(), **DD).reshape_as(positions)
    (mixed,) = torch.autograd.grad((position_gradient * direction).sum(), leaf)
    assert torch.isfinite(mixed).all()
    assert torch.count_nonzero(mixed) > 0


def test_exact_aes2_update_reset_and_setup_authority() -> None:
    """Exact AES2 is setup-owned and its compatibility cache is nonpersistent."""
    numbers, positions, parameters, ihelp, interaction = _inputs()
    setup = setup_aes2(interaction, numbers, ihelp)
    reference_energy = _energy(
        setup, positions, interaction, _charges(ihelp, numbers), ihelp
    )
    first = interaction.get_cache(
        numbers=numbers, positions=positions, ihelp=ihelp
    )
    second = interaction.get_cache(
        numbers=numbers, positions=positions, ihelp=ihelp
    )
    assert first is not second
    assert interaction.cache is None
    assert interaction._cachevars is None
    assert interaction._cachegrad is None

    with torch.no_grad():
        interaction.dmp3.fill_(20.0)
    energy_after_mutation = _energy(
        setup, positions, interaction, _charges(ihelp, numbers), ihelp
    )
    torch.testing.assert_close(energy_after_mutation, reference_energy)
    with pytest.raises(RuntimeError, match="setup-derived"):
        interaction.update(dmp3=torch.tensor(2.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        interaction.reset()

    model = Model(
        par=parameters,
        config=Config.create(exclude=("aes2",)),
        interaction=(interaction,),
    )
    system = model.setup(numbers)
    assert system.aes2_interaction is interaction
    assert system.aes2_setup is not None
    assert torch.allclose(system.aes2_setup.dmp3, interaction.dmp3)
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.interactions.update("AES2", dmp3=torch.tensor(2.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.interactions.reset("AES2")
    system.interactions.reset_all()
    assert system.aes2_setup is not None

    calc = Calculator(
        numbers,
        parameters,
        dtype=torch.float64,
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "int_level": 4,
            "exclude": ("d4sc", "disp"),
        },
    )
    calculator_setup = calc.system.aes2_setup
    assert calculator_setup is not None
    calc.reset()
    assert calc.system.aes2_setup is calculator_setup


def test_aes2_exact_replacement_and_duplicate_label_rejection() -> None:
    """One excluded-built-in AES2 replacement is explicit and unambiguous."""
    numbers, positions, parameters, _, custom_base = _inputs()
    custom = AES2(
        dmp3=custom_base.dmp3 * 1.7,
        dmp5=custom_base.dmp5,
        dkernel=custom_base.dkernel,
        qkernel=custom_base.qkernel,
        shift=custom_base.shift,
        kexp=custom_base.kexp,
        rmax=custom_base.rmax,
        rad=custom_base.rad,
        vcn=custom_base.vcn,
        **DD,
    )
    system = Model(
        par=parameters,
        config=Config.create(exclude=("aes2", "d4sc", "disp")),
        interaction=(custom,),
    ).setup(numbers)
    assert system.aes2_interaction is custom
    assert system.aes2_setup is not None
    torch.testing.assert_close(system.aes2_setup.dmp3, custom.dmp3)
    assert not torch.isclose(system.aes2_setup.dmp3, custom_base.dmp3)

    default_system = Model(
        par=parameters,
        config=Config.create(exclude=("d4sc", "disp")),
    ).setup(numbers)
    custom_result = system.singlepoint(positions)
    default_result = default_system.singlepoint(positions)
    assert not torch.allclose(custom_result.energy, default_result.energy)

    duplicate = AES2(
        custom_base.dmp3,
        custom_base.dmp5,
        custom_base.dkernel,
        custom_base.qkernel,
        custom_base.shift,
        custom_base.kexp,
        custom_base.rmax,
        custom_base.rad,
        custom_base.vcn,
        **DD,
    )
    with pytest.raises(ValueError, match="AES2.*ambiguous"):
        InteractionList(custom, duplicate)

    class Extension(AES2):
        pass

    subclass = Extension(
        custom_base.dmp3,
        custom_base.dmp5,
        custom_base.dkernel,
        custom_base.qkernel,
        custom_base.shift,
        custom_base.kexp,
        custom_base.rmax,
        custom_base.rad,
        custom_base.vcn,
        **DD,
    )
    subclass.label = "AES2"
    with pytest.raises(ValueError, match="AES2.*ambiguous"):
        InteractionList(custom, subclass)


def test_singlepoint_core_bypasses_exact_aes2_get_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Singlepoint uses System AES2 setup instead of persistent get_cache."""
    numbers, positions, parameters, _, _ = _inputs()
    config = Config.create(
        int_driver="pytorch",
        int_level=4,
        exclude=("d4sc", "disp"),
    )
    system = Model(par=parameters, config=config, auto_int_level=False).setup(
        numbers
    )
    assert system.aes2_setup is not None

    def fail_cache(*_: object, **__: object) -> AES2Cache:
        raise AssertionError("single-system core called AES2.get_cache")

    monkeypatch.setattr(AES2, "get_cache", fail_cache)
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()


def test_core_interaction_data_is_call_local_after_geometry_change() -> None:
    """System keeps setup only while call-local matrices track geometry."""
    numbers, positions, parameters, _, _ = _inputs()
    system = Model(par=parameters, auto_int_level=False).setup(numbers)
    assert system.aes2_setup is not None
    first = _interaction_data(system, positions)["AES2"]
    moved = positions.clone()
    moved[1, 2] += 0.1
    second = _interaction_data(system, moved)["AES2"]
    third = _interaction_data(system, positions.clone())["AES2"]
    assert first is not second and second is not third
    assert not torch.allclose(first.amat_sd, second.amat_sd)
    torch.testing.assert_close(first.amat_sd, third.amat_sd)


def test_aes2_subclass_remains_a_cache_driven_extension() -> None:
    """Subclasses keep per-call cache refresh and legacy reset semantics."""
    numbers, positions, parameters, ihelp, base = _inputs()

    class AES2Extension(AES2):
        __slots__ = ("scale", "initial_scale", "cache_calls", "reset_calls")

        def __init__(self) -> None:
            super().__init__(
                base.dmp3,
                base.dmp5,
                base.dkernel,
                base.qkernel,
                base.shift,
                base.kexp,
                base.rmax,
                base.rad,
                base.vcn,
                **DD,
            )
            self.scale = torch.tensor(1.0, **DD)
            self.initial_scale = self.scale.clone()
            self.cache_calls = 0
            self.reset_calls = 0

        def get_cache(self, **_: object) -> Tensor:
            self.cache_calls += 1
            return self.scale.clone()

        def get_monopole_atom_energy(
            self, cache: Tensor, qat: Tensor, **_: object
        ) -> Tensor:
            return torch.ones_like(qat) * cache

        def reset(self) -> None:
            self.reset_calls += 1
            self.scale = self.initial_scale.clone()

    extension = AES2Extension()
    system = Model(
        par=parameters,
        config=Config.create(exclude=("aes2",)),
        interaction=(extension,),
    ).setup(numbers)
    assert system.aes2_interaction is None
    charges = Charges(mono=torch.ones(ihelp.nao, **DD))

    def energy() -> Tensor:
        cache = _interaction_data(system, positions)
        return system.interactions.get_energy_as_dict(charges, cache, ihelp)[
            extension.label
        ].sum()

    one = energy()
    extension.update(scale=torch.tensor(2.0, **DD))
    two = energy()
    assert two > one
    assert extension.cache_calls == 2
    system.interactions.reset(extension.label)
    assert extension.reset_calls == 1
    torch.testing.assert_close(energy(), one)

    calc = Calculator(
        numbers,
        parameters,
        interaction=extension,
        dtype=torch.float64,
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "int_level": 4,
            "exclude": ("aes2", "d4sc", "disp"),
        },
    )
    calc.reset()
    assert extension.reset_calls == 2
    calc_data = _interaction_data(calc.system, positions)
    calc_energy = calc.interactions.get_energy_as_dict(
        charges, calc_data, calc.ihelp
    )[extension.label].sum()
    torch.testing.assert_close(calc_energy, one)


def test_aes2_setup_builder_is_the_only_matrix_formula() -> None:
    """Exact compatibility hooks contain no persistent or indexed writes."""
    import inspect

    from dxtb._src.components.interactions.coulomb.multipole import (
        _build_atom_coulomb_matrix,
    )

    cache_source = inspect.getsource(AES2.get_cache)
    matrix_source = inspect.getsource(_build_atom_coulomb_matrix)
    assert "cache_is_latest" not in cache_source
    assert "self.cache" not in cache_source
    assert "_cachevars" not in cache_source
    assert "_cachegrad" not in cache_source
    assert "torch.empty" not in matrix_source
    assert "sq[" not in matrix_source
    assert "self." not in matrix_source


def test_cull_restore_only_changes_call_local_aes2_data() -> None:
    """Culling cannot mutate persistent AES2 setup or leak into next call."""
    numbers, positions, parameters, _, interaction = _inputs()
    batch_numbers = torch.stack((numbers, numbers))
    batch_positions = torch.stack((positions, positions + 0.04))
    ihelp = IndexHelper.from_numbers(batch_numbers, parameters)
    setup = setup_aes2(interaction, batch_numbers, ihelp)
    setup_values = {
        field.name: getattr(setup, field.name).clone()
        for field in fields(setup)
        if isinstance(getattr(setup, field.name), Tensor)
    }
    first = build_aes2_data(setup, batch_positions)
    first.cull(
        torch.tensor([False, True], device=DEVICE),
        {"atom": (...,)},
    )
    first.restore()
    for name, value in setup_values.items():
        torch.testing.assert_close(getattr(setup, name), value)
    second = build_aes2_data(setup, batch_positions)
    assert first is not second
    torch.testing.assert_close(first.amat_sd, second.amat_sd)
