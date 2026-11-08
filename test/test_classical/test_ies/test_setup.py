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
from functools import partial

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.ncoord import CNModel, erf_count
from tad_mctc.tree import Node, stack
from tad_multicharge.model.eeq import EEQModel
from torch.utils import _pytree

from dxtb import GFN0_XTB, Calculator, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    IES,
    IESCoreSetup,
    IESSetup,
    new_ies,
)
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

    assert isinstance(setup, Node)
    assert type(setup) is IESCoreSetup
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
    assert isinstance(setup.cutoff, float)
    for name in ("numbers", "rcov", "cn_max", "cn_kcn"):
        assert isinstance(getattr(setup, name), torch.Tensor)
    assert not any(
        isinstance(getattr(setup, field.name), (ParamModule, Model, IES))
        for field in fields(setup)
    )
    leaves, spec = _pytree.tree_flatten(setup)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is IESCoreSetup
    assert rebuilt.cutoff == setup.cutoff
    assert all(a is b for a, b in zip(leaves, _pytree.tree_leaves(rebuilt)))
    converted = setup.to(dtype=torch.float32)
    assert converted.eeq.chi.dtype == torch.float32
    assert converted.numbers.dtype == numbers.dtype
    assert converted.cutoff == setup.cutoff
    meta = setup.to(device=torch.device("meta"))
    assert meta.numbers.device.type == "meta"
    assert meta.eeq.chi.device.type == "meta"
    stacked = stack((setup, setup))
    assert type(stacked) is IESCoreSetup
    assert stacked.eeq.chi.shape[0] == 2
    assert stacked.cutoff == setup.cutoff
    with pytest.raises(ValueError, match="different structure"):
        stack((setup, setup.replace(cutoff=setup.cutoff + 1.0)))

    legacy1 = system.ies_classical.get_cache(numbers)
    legacy2 = system.ies_classical.get_cache(numbers)
    assert type(legacy1) is IESSetup
    assert type(legacy2) is IESSetup
    assert legacy1 is not legacy2
    assert setup.eeq is not legacy1.eeq
    assert not hasattr(system.ies_classical, "cache")
    assert not hasattr(system.ies_classical, "_cachevars")
    with pytest.raises(FrozenInstanceError):
        setup.cutoff = 1.0  # type: ignore[misc]


def test_upstream_eeq_optional_solve_modes_match_schur() -> None:
    """Characterize upstream CG modes without changing IES defaults."""
    numbers, positions = _geometry()
    positions = positions.clone().requires_grad_(True)
    system = _model(_module()).setup(numbers)
    setup = system.ies_setup
    assert setup is not None
    charge = torch.tensor(0.23, **DD)
    structure = Structure(numbers=numbers, positions=positions, charge=charge)
    cn = CNModel(
        count=partial(erf_count, kcn=setup.cn_kcn),
        cutoff=setup.cutoff,
        cn_max=setup.cn_max,
        rcov=setup.rcov,
    )(structure)

    references = {}
    for mode in ("schur", "cg", "matrix-free"):
        charges, energy = setup.eeq.solve(
            structure, cn, return_energy=True, solve_mode=mode
        )
        grad = torch.autograd.grad(energy.sum(), positions, retain_graph=True)[0]
        references[mode] = (charges, energy, grad)

    expected_charges, expected_energy, expected_grad = references["schur"]
    for mode in ("cg", "matrix-free"):
        charges, energy, grad = references[mode]
        torch.testing.assert_close(
            charges, expected_charges, atol=1.0e-10, rtol=1.0e-10
        )
        torch.testing.assert_close(
            energy, expected_energy, atol=1.0e-10, rtol=1.0e-10
        )
        torch.testing.assert_close(
            grad, expected_grad, atol=1.0e-9, rtol=1.0e-9
        )


def test_dxtb_and_tad_dftd_wrappers_thread_solve_mode() -> None:
    """IES and tad_dftd charge wrappers forward solve_mode to EEQModel."""
    from tad_dftd.charge import get_charges, get_eeq_charges

    numbers, positions = _geometry()
    system = _model(_module()).setup(numbers)
    setup = system.ies_setup
    ies_comp = system.ies_classical
    assert setup is not None and ies_comp is not None
    charge = torch.tensor(0.23, **DD)

    e_schur = ies_energy(setup, positions, charge, solve_mode="schur")
    q_dftd_schur = get_charges(numbers, positions, charge, solve_mode="schur")
    for mode in ("cg", "matrix-free"):
        e_mode = ies_energy(setup, positions, charge, solve_mode=mode)
        e_comp = ies_comp.get_energy(
            positions, setup, charge=charge, solve_mode=mode
        )
        q_dftd = get_charges(numbers, positions, charge, solve_mode=mode)
        q_eeq = get_eeq_charges(numbers, positions, charge, solve_mode=mode)
        torch.testing.assert_close(e_mode, e_schur, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(e_comp, e_schur, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(
            q_dftd, q_dftd_schur, atol=1.0e-10, rtol=1.0e-10
        )
        torch.testing.assert_close(
            q_eeq, q_dftd_schur, atol=1.0e-10, rtol=1.0e-10
        )


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
    disabled = _energy(system, positions.clone())
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
    assert type(system.ies_setup) is IESCoreSetup
    assert system.classical_cache["IES"] is system.ies_setup
    assert isinstance(system.ies_setup.cutoff, float)
    assert system.ies_setup.cutoff == pytest.approx(float(custom.cutoff))
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


def test_ies_subclass_is_rejected() -> None:
    """IES subclasses are custom Classicals and fail at setup."""
    numbers, _ = _geometry()
    par = _module()
    base = new_ies(numbers, par, **DD)
    assert base is not None

    class IESExtension(IES):
        pass

    extension = IESExtension(
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
    with pytest.raises(TypeError, match="Only the built-in"):
        Model(
            par=par,
            config=Config.create(exclude=("ies", *EXCLUDE)),
            classical=(extension,),
        ).setup(numbers)


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
