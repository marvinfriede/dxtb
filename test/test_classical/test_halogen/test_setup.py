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
"""Tests for fixed-shape halogen setup and evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, Calculator, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    Halogen,
    HalogenSetup,
    halogen_energy,
    new_halogen,
)
from dxtb._src.components.classicals.base import Classical
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ...conftest import DEVICE
from ...utils import get_param_module
from .samples import samples

DD = {"device": DEVICE, "dtype": torch.double}
_EXCLUDE = (
    "scf",
    "es2",
    "aes2",
    "es3",
    "d4sc",
    "disp",
    "ies",
    "rep",
    "srb",
)


def _model(par: ParamModule, *, halogen: bool = True) -> Model:
    exclude = list(_EXCLUDE)
    if not halogen:
        exclude.append("hal")
    return Model(par=par, config=Config.create(exclude=tuple(exclude)))


def _fresh_parameters() -> ParamModule:
    """Create private parameter leaves for gradient tests."""
    return ParamModule(GFN1_XTB.model_copy(deep=True), **DD)


def _geometry(name: str = "br2nh3") -> tuple[Tensor, Tensor]:
    sample = samples[name]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _classical_energy(system: object, positions: Tensor) -> Tensor:
    result = system.singlepoint(positions)  # type: ignore[attr-defined]
    return dict(result.classical)["Halogen"].sum()


def test_setup_is_frozen_and_contains_only_structural_values() -> None:
    """Halogen setup is a frozen bundle of gathered numbers-only data."""
    numbers, _ = _geometry()
    system = _model(get_param_module("gfn1", **DD)).setup(numbers)
    setup = system.halogen_setup

    assert isinstance(setup, HalogenSetup)
    assert system.halogen_classical is not None
    assert {field.name for field in fields(setup)} == {
        "numbers",
        "xbond",
        "atomic_radii",
        "damp",
        "cutoff",
        "halogen_mask",
        "base_mask",
        "valid_atom_mask",
        "pair_type_mask",
        "neighbor_type_mask",
    }
    assert setup.pair_type_mask.shape == (numbers.numel(), numbers.numel())
    assert setup.neighbor_type_mask.shape == (numbers.numel(), numbers.numel())
    assert setup.halogen_mask.dtype == torch.bool
    assert setup.base_mask.dtype == torch.bool
    assert setup.valid_atom_mask.dtype == torch.bool
    assert not any(
        any(
            term in field.name
            for term in ("position", "adj", "nearest", "energy")
        )
        for field in fields(setup)
    )
    assert not any(
        isinstance(getattr(setup, field.name), (Classical, Model))
        for field in fields(setup)
    )
    assert setup.numbers is system.numbers
    assert setup.damp is not system.halogen_classical.damp
    with pytest.raises(FrozenInstanceError):
        setup.cutoff = torch.tensor(2.0, **DD)  # type: ignore[misc]

    first = system.halogen_classical.get_cache(numbers, system.ihelp)
    second = system.halogen_classical.get_cache(numbers, system.ihelp)
    assert first is not second
    assert system.halogen_classical.cache is None
    assert system.halogen_classical._cachevars is None


def test_no_halogen_no_base_and_cutoff_masks_return_finite_zero() -> None:
    """Fixed masks return zero for absent species and distant pairs."""
    par = _fresh_parameters()
    for numbers, positions in (
        (
            torch.tensor([7, 1, 1], device=DEVICE),
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], **DD
            ),
        ),
        (
            torch.tensor([35, 35], device=DEVICE),
            torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], **DD),
        ),
        (
            torch.tensor([35, 7, 1], device=DEVICE),
            torch.tensor(
                [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 1.0, 0.0]], **DD
            ),
        ),
    ):
        system = _model(par).setup(numbers)
        assert system.halogen_setup is not None
        value = halogen_energy(system.halogen_setup, positions)
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, torch.zeros_like(value))


def test_setup_honors_component_species_lists_and_freezes_them() -> None:
    """Species lists configure setup and cannot mutate an existing System."""
    from dxtb import IndexHelper

    numbers, positions = _geometry()
    par = get_param_module("gfn1", **DD)
    default = new_halogen(torch.unique(numbers), par, **DD)
    assert default is not None
    custom = Halogen(
        default.damp,
        default.rscale,
        default.bond_strength,
        cutoff=default.cutoff,
        **DD,
    )
    custom_system = Model(
        par=par,
        config=Config.create(exclude=(*_EXCLUDE, "hal")),
        classical=(custom,),
    ).setup(numbers)
    assert custom_system.halogen_setup is not None

    default_energy = _classical_energy(custom_system, positions)
    assert not torch.isclose(default_energy, torch.zeros_like(default_energy))

    # Setup captures the lists as masks; later object mutation affects only a
    # newly constructed System.
    custom.halogens = []
    unchanged = _classical_energy(custom_system, positions.clone())
    torch.testing.assert_close(unchanged, default_energy)
    changed_system = Model(
        par=par,
        config=Config.create(exclude=(*_EXCLUDE, "hal")),
        classical=(custom,),
    ).setup(numbers)
    torch.testing.assert_close(
        _classical_energy(changed_system, positions),
        torch.zeros_like(default_energy),
    )

    custom.halogens = [17, 35, 53, 85]
    custom.bases = []
    no_base_system = Model(
        par=par,
        config=Config.create(exclude=(*_EXCLUDE, "hal")),
        classical=(custom,),
    ).setup(numbers)
    torch.testing.assert_close(
        _classical_energy(no_base_system, positions),
        torch.zeros_like(default_energy),
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    direct = custom.get_cache(numbers, ihelp)
    assert direct.halogen_mask.any()
    assert not direct.base_mask.any()


def test_halogen_subclass_inherited_get_cache_honors_species_lists() -> None:
    """Inherited compatibility setup reads a subclass's species lists."""
    from dxtb import IndexHelper

    numbers, _ = _geometry()
    par = get_param_module("gfn1", **DD)
    ordinary = new_halogen(torch.unique(numbers), par, **DD)
    assert ordinary is not None

    class CustomSpeciesHalogen(Halogen):
        pass

    custom = CustomSpeciesHalogen(
        ordinary.damp,
        ordinary.rscale,
        ordinary.bond_strength,
        cutoff=ordinary.cutoff,
        **DD,
    )
    custom.halogens = []
    setup = custom.get_cache(numbers, IndexHelper.from_numbers(numbers, par))
    assert not setup.halogen_mask.any()


def test_vmap_handles_distinct_cutoff_masks() -> None:
    """Cutoff changes alter mask values but retain fixed output shape."""
    from torch.func import vmap

    numbers, positions = _geometry()
    setup = _model(get_param_module("gfn1", **DD)).setup(numbers).halogen_setup
    assert setup is not None
    outside = positions.clone()
    outside[2, 0] += 100.0
    conformers = torch.stack((positions, outside))

    def energy(pos: Tensor) -> Tensor:
        return halogen_energy(setup, pos).sum()

    mapped = vmap(energy)(conformers)
    looped = torch.stack([energy(pos) for pos in conformers])
    torch.testing.assert_close(mapped, looped)
    assert not torch.isclose(mapped[0], mapped[1])


def test_vmap_matches_loop_with_stable_neighbor_topology() -> None:
    """Fixed-topology nonrigid conformers agree under vmap and a loop."""
    from torch.func import vmap

    numbers, positions = _geometry()
    setup = _model(get_param_module("gfn1", **DD)).setup(numbers).halogen_setup
    assert setup is not None
    moved = positions.clone()
    moved[2, 1] += 0.01
    conformers = torch.stack((positions, moved))

    def energy(pos: Tensor) -> Tensor:
        return halogen_energy(setup, pos).sum()

    torch.testing.assert_close(
        vmap(energy)(conformers),
        torch.stack([energy(pos) for pos in conformers]),
    )


def test_legacy_padded_batch_has_zero_ghost_energy_and_gradient() -> None:
    """The compatibility batch wrapper preserves zero ghost contributions."""
    sample_a, sample_b = samples["br2nh3"], samples["br2och2"]
    numbers = pack(
        (sample_a["numbers"].to(DEVICE), sample_b["numbers"].to(DEVICE))
    )
    positions = pack(
        (sample_a["positions"].to(**DD), sample_b["positions"].to(**DD))
    ).requires_grad_(True)
    par = get_param_module("gfn1", **DD)
    component = new_halogen(torch.unique(numbers), par, **DD)
    assert component is not None
    from dxtb import IndexHelper

    setup = component.get_cache(numbers, IndexHelper.from_numbers(numbers, par))
    energy = component.get_energy(positions, setup)
    (gradient,) = torch.autograd.grad(energy.sum(), positions)
    ghost = numbers == 0
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(energy[ghost]) == 0
    assert torch.count_nonzero(gradient[ghost]) == 0


def test_nearest_neighbor_rule_excludes_zero_distance_candidates() -> None:
    """Nearest-neighbor selection excludes padding and all coincident atoms."""
    numbers, positions = _geometry()
    setup = _model(get_param_module("gfn1", **DD)).setup(numbers).halogen_setup
    assert setup is not None
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)
    r2 = (delta * delta).sum(-1)
    candidates = setup.neighbor_type_mask & (r2 > 0.0)
    indices = torch.argmin(torch.where(candidates, r2, torch.inf), dim=-1)
    assert indices[:2].tolist() == [1, 0]

    # Move the second Br far away: Br 0 now chooses the nearest N/H atom.
    switched = positions.clone()
    switched[1, 0] += 50.0
    moved_delta = switched.unsqueeze(0) - switched.unsqueeze(1)
    moved_r2 = (moved_delta * moved_delta).sum(-1)
    moved_candidates = setup.neighbor_type_mask & (moved_r2 > 0.0)
    moved_indices = torch.argmin(
        torch.where(moved_candidates, moved_r2, torch.inf), dim=-1
    )
    assert moved_indices[0] != indices[0]
    ordinary_energy = halogen_energy(setup, positions).sum()
    switched_energy = halogen_energy(setup, switched).sum()
    torch.testing.assert_close(
        ordinary_energy,
        samples["br2nh3"]["energy"].to(**DD),
    )
    torch.testing.assert_close(switched_energy, torch.tensor(0.0, **DD))

    coincident_numbers = torch.tensor([35, 7, 1], device=DEVICE)
    coincident_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]],
        **DD,
    )
    coincident_setup = (
        _model(get_param_module("gfn1", **DD))
        .setup(coincident_numbers)
        .halogen_setup
    )
    assert coincident_setup is not None
    coincident_r2 = (
        (coincident_positions.unsqueeze(0) - coincident_positions.unsqueeze(1))
        .square()
        .sum(-1)
    )
    valid = coincident_setup.neighbor_type_mask & (coincident_r2 > 0.0)
    nearest = torch.argmin(torch.where(valid, coincident_r2, torch.inf), dim=-1)
    assert nearest[0] == 1


def test_nearest_neighbor_switch_matches_task_base_values() -> None:
    """A live pair follows nearest-neighbor switches as in the old kernel."""
    numbers = torch.tensor([35, 7, 1, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.5, 0.0],
            [0.0, -2.0, 0.0],
        ],
        **DD,
    )
    switched = positions.clone()
    switched[2] = torch.tensor([2.0, 0.0, 0.0], **DD)
    setup = _model(get_param_module("gfn1", **DD)).setup(numbers).halogen_setup
    assert setup is not None

    def nearest_index(pos: Tensor) -> int:
        delta = pos.unsqueeze(0) - pos.unsqueeze(1)
        r2 = delta.square().sum(dim=-1)
        candidate = setup.neighbor_type_mask & (r2 > 0.0)
        return int(
            torch.argmin(torch.where(candidate, r2, torch.inf), dim=-1)[0]
        )

    assert nearest_index(positions) == 2
    assert nearest_index(switched) == 3

    # These values were characterized from the pre-refactor implementation at
    # task base c71e4a0; both geometries retain the same active Br--N pair.
    torch.testing.assert_close(
        halogen_energy(setup, positions).sum(),
        torch.tensor(2.6027936682145247e-7, **DD),
        rtol=1e-10,
        atol=1e-14,
    )
    torch.testing.assert_close(
        halogen_energy(setup, switched).sum(),
        torch.tensor(4.122720004389e-4, **DD),
        rtol=1e-10,
        atol=1e-14,
    )


def test_jvp_jacfwd_and_mixed_parameter_derivative() -> None:
    """Forward transforms and mixed derivatives use the plain tensor path."""
    from torch.func import jacfwd, jvp

    numbers, positions = _geometry()
    par = _fresh_parameters()
    damp = par.get("halogen.classical.damping")
    damp.requires_grad_(True)
    setup = _model(par).setup(numbers).halogen_setup
    assert setup is not None

    def energy(pos: Tensor) -> Tensor:
        return halogen_energy(setup, pos).sum()

    direction = torch.linspace(-0.2, 0.3, positions.numel(), **DD).reshape_as(
        positions
    )
    _, tangent = jvp(energy, (positions,), (direction,))
    pos = positions.clone().requires_grad_(True)
    reverse = torch.autograd.grad(energy(pos), pos, create_graph=True)[0]
    torch.testing.assert_close(tangent, (reverse * direction).sum())
    torch.testing.assert_close(jacfwd(energy)(positions), reverse.detach())

    projection = (reverse * direction).sum()
    (mixed,) = torch.autograd.grad(projection, damp)
    assert torch.isfinite(mixed)
    assert torch.count_nonzero(mixed) > 0


@pytest.mark.parametrize(
    "parameter", ["halogen.classical.damping", "halogen.classical.rscale"]
)
def test_setup_parameters_retain_param_module_gradients(parameter: str) -> None:
    """Damping and radii scale remain connected to real model leaves."""
    numbers, positions = _geometry()
    par = _fresh_parameters()
    leaf = par.get(parameter)
    leaf.requires_grad_(True)
    system = _model(par).setup(numbers)
    energy = _classical_energy(system, positions)
    (gradient,) = torch.autograd.grad(energy, leaf)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_parameter_gradient_repeats_with_fresh_halogen_setup() -> None:
    """Fresh setup graphs produce repeatable halogen parameter gradients."""
    numbers, positions = _geometry()
    par = _fresh_parameters()
    leaf = par.get("halogen.classical.damping")
    leaf.requires_grad_(True)
    model = _model(par)

    first = model.setup(numbers)
    first_energy = _classical_energy(first, positions)
    (first_gradient,) = torch.autograd.grad(first_energy, leaf)

    second = model.setup(numbers)
    second_energy = _classical_energy(second, positions.clone())
    (second_gradient,) = torch.autograd.grad(second_energy, leaf)

    torch.testing.assert_close(first_energy, second_energy)
    torch.testing.assert_close(first_gradient, second_gradient)


def test_xbond_parameter_gradient_reaches_element_leaf() -> None:
    """Gathered bromine xbond values retain their ParamModule graph."""
    numbers, positions = _geometry()
    par = _fresh_parameters()
    leaf = par.get("element.Br.xbond")
    leaf.requires_grad_(True)
    system = _model(par).setup(numbers)
    energy = _classical_energy(system, positions)
    (gradient,) = torch.autograd.grad(energy, leaf)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_halogen_history_and_prior_backward_are_independent() -> None:
    """A/B/A calls and consumed gradients do not retain topology state."""
    numbers, positions = _geometry()
    model = _model(get_param_module("gfn1", **DD))
    system = model.setup(numbers)
    a = positions.clone()
    b = positions.clone()
    b[2, 0] += 0.2
    pos = a.clone().requires_grad_(True)
    energy_a = _classical_energy(system, pos)
    torch.autograd.grad(energy_a, pos)
    energy_b = _classical_energy(system, b)
    energy_a_again = _classical_energy(system, a.clone())
    fresh = _classical_energy(model.setup(numbers), a.clone())
    assert not torch.isclose(energy_a, energy_b)
    torch.testing.assert_close(energy_a, energy_a_again)
    torch.testing.assert_close(energy_a, fresh)


def test_stale_component_mutation_does_not_change_system_setup() -> None:
    """Direct compatibility-object mutation cannot change frozen setup data."""
    numbers, positions = _geometry()
    par = _fresh_parameters()
    rscale = par.get("halogen.classical.rscale")
    rscale.requires_grad_(True)
    model = _model(par)
    system = model.setup(numbers)
    assert system.halogen_classical is not None
    before = _classical_energy(system, positions)

    with torch.no_grad():
        system.halogen_classical.damp.fill_(10.0)
        system.halogen_classical.rscale.fill_(10.0)
    after = _classical_energy(system, positions.clone())
    torch.testing.assert_close(before, after)

    with torch.no_grad():
        rscale.mul_(1.1)
    fresh = model.setup(numbers)
    changed = _classical_energy(fresh, positions.clone())
    assert not torch.isclose(changed, after)


def test_exact_user_halogen_replaces_excluded_builtin() -> None:
    """An exact replacement supplies the setup parameters for its System."""
    numbers, positions = _geometry()
    par = get_param_module("gfn1", **DD)
    ordinary = new_halogen(torch.unique(numbers), par, **DD)
    assert ordinary is not None
    custom = Halogen(
        ordinary.damp,
        ordinary.rscale,
        ordinary.bond_strength * 2.0,
        cutoff=ordinary.cutoff,
        **DD,
    )
    custom_system = Model(
        par=par,
        config=Config.create(exclude=(*_EXCLUDE, "hal")),
        classical=(custom,),
    ).setup(numbers)
    default_system = _model(par).setup(numbers)
    assert custom_system.halogen_classical is custom
    assert custom_system.halogen_setup is not None
    assert not torch.isclose(
        _classical_energy(custom_system, positions),
        _classical_energy(default_system, positions),
    )


def test_halogen_subclass_refreshes_call_data_and_stays_mutable() -> None:
    """A subclass uses refreshed cache hooks and legacy update/reset calls."""

    class HalogenExtension(Halogen):
        __slots__ = ("scale", "cache_calls", "reset_calls")

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.scale = torch.tensor(1.0, **DD)
            self.initial_scale = self.scale.clone()
            self.cache_calls = 0
            self.reset_calls = 0

        def get_cache(self, numbers, ihelp=None, **kwargs):
            self.cache_calls += 1
            return self.scale.clone()

        def get_energy(self, positions, cache, **kwargs):
            return positions.new_ones(positions.shape[-2]) * cache

        def reset(self) -> None:
            self.reset_calls += 1
            self.scale = self.initial_scale.clone()
            super().reset()

    numbers, positions = _geometry()
    par = get_param_module("gfn1", **DD)
    original = new_halogen(torch.unique(numbers), par, **DD)
    assert original is not None
    extension = HalogenExtension(
        original.damp,
        original.rscale,
        original.bond_strength,
        cutoff=original.cutoff,
        **DD,
    )
    extension.label = "Halogen"
    system = Model(
        par=par,
        config=Config.create(exclude=(*_EXCLUDE, "hal")),
        classical=(extension,),
    ).setup(numbers)

    first = _classical_energy(system, positions)
    system.classicals.update("Halogen", scale=torch.tensor(2.0, **DD))
    updated = _classical_energy(system, positions)
    assert updated == 2.0 * first
    assert extension.cache_calls == 3  # setup plus two evaluations
    system.classicals.reset("Halogen")
    restored = _classical_energy(system, positions)
    torch.testing.assert_close(restored, first)
    system.classicals.update("Halogen", scale=torch.tensor(2.0, **DD))
    updated = _classical_energy(system, positions)
    system.classicals.reset_all()
    assert extension.reset_calls == 2
    torch.testing.assert_close(_classical_energy(system, positions), first)

    calc = Calculator(
        numbers,
        par,
        classical=(extension,),
        opts={"exclude": (*_EXCLUDE, "hal"), "verbosity": 0},
        dtype=torch.double,
        auto_int_level=False,
    )
    calc.reset()
    assert extension.reset_calls == 3
    torch.testing.assert_close(_classical_energy(calc.system, positions), first)


@pytest.mark.parametrize("with_subclass", [False, True])
def test_duplicate_halogen_labels_are_rejected(with_subclass: bool) -> None:
    """Built-in and custom Halogen labels cannot overwrite call data."""
    numbers, _ = _geometry()
    par = get_param_module("gfn1", **DD)
    base = new_halogen(torch.unique(numbers), par, **DD)
    assert base is not None
    custom: Classical
    if with_subclass:

        class HalogenExtension(Halogen):
            pass

        custom = HalogenExtension(
            base.damp, base.rscale, base.bond_strength, **DD
        )
        custom.label = "Halogen"
    else:
        custom = Halogen(base.damp, base.rscale, base.bond_strength, **DD)
    with pytest.raises(ValueError, match="Halogen.*ambiguous"):
        Model(
            par=par,
            config=Config.create(exclude=_EXCLUDE),
            classical=(custom,),
        ).setup(numbers)


def test_two_user_halogen_labels_are_rejected() -> None:
    """Two user terms with the same label cannot overwrite call data."""
    numbers, _ = _geometry()
    par = get_param_module("gfn1", **DD)
    base = new_halogen(torch.unique(numbers), par, **DD)
    assert base is not None
    first = Halogen(base.damp, base.rscale, base.bond_strength, **DD)
    second = Halogen(base.damp, base.rscale, base.bond_strength, **DD)
    with pytest.raises(ValueError, match="Halogen.*ambiguous"):
        Model(
            par=par,
            config=Config.create(exclude=(*_EXCLUDE, "hal")),
            classical=(first, second),
        ).setup(numbers)


def test_exact_halogen_update_and_reset_are_rejected() -> None:
    """Exact Halogen parameters cannot diverge from the System setup."""
    numbers, positions = _geometry()
    system = _model(get_param_module("gfn1", **DD)).setup(numbers)
    component = system.halogen_classical
    setup = system.halogen_setup
    assert component is not None and setup is not None
    with pytest.raises(RuntimeError, match="setup-derived"):
        component.update(damp=torch.tensor(2.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        component.reset()
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.update("Halogen", damp=torch.tensor(2.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        system.classicals.reset("Halogen")

    before = _classical_energy(system, positions)
    system.classicals.reset_all()
    assert system.halogen_setup is setup
    torch.testing.assert_close(_classical_energy(system, positions), before)

    par = _fresh_parameters()
    damping = par.get("halogen.classical.damping")
    damping.requires_grad_(True)
    calc = Calculator(
        numbers,
        par,
        dtype=torch.double,
        opts={"verbosity": 0, "exclude": _EXCLUDE},
    )
    calc_setup = calc.system.halogen_setup
    calc_before = _classical_energy(calc.system, positions)
    calc.reset()
    calc_after = _classical_energy(calc.system, positions.clone())
    assert calc.system.halogen_setup is calc_setup
    torch.testing.assert_close(calc_before, calc_after)
    (gradient,) = torch.autograd.grad(calc_before, damping)
    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) > 0
