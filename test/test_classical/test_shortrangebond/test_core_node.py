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
"""Node structure and parity tests for System-owned SRB setup values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.func import vmap
from torch.utils import _pytree

from dxtb import GFN0_XTB, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    ShortRangeBondCoreSetup,
    ShortRangeBondSetup,
    core_srb_setup,
    new_srb,
    short_range_bond_energy,
)
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples

_FIELDS = (
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
)
_TENSOR_FIELDS = tuple(name for name in _FIELDS if name != "counting_function")


def _setup(
    *, differentiable: bool = False
) -> tuple[Tensor, Tensor, ParamModule, object, ShortRangeBondSetup]:
    numbers = samples["NO2"]["numbers"].to(DEVICE)
    positions = samples["NO2"]["positions"].to(
        device=DEVICE, dtype=torch.double
    )
    par = ParamModule(GFN0_XTB, device=DEVICE, dtype=torch.double)
    if differentiable:
        par.set_differentiable("short_range.srb.prefactor")
    component = new_srb(
        torch.unique(numbers), par, device=DEVICE, dtype=torch.double
    )
    assert component is not None
    legacy = component.get_cache(
        numbers, IndexHelper.from_numbers(numbers, par)
    )
    assert type(legacy) is ShortRangeBondSetup
    return numbers, positions, par, component, legacy


def test_core_schema_identity_and_pytree_roundtrip() -> None:
    """Represent all fifteen tensors dynamically and preserve identities."""
    _, _, _, _, legacy = _setup()
    core = core_srb_setup(legacy)

    assert isinstance(core, Node)
    assert type(core) is ShortRangeBondCoreSetup
    assert tuple(field.name for field in fields(core)) == _FIELDS
    leaves, spec = _pytree.tree_flatten(core)
    assert len(leaves) == len(_TENSOR_FIELDS)
    assert all(isinstance(leaf, Tensor) for leaf in leaves)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is ShortRangeBondCoreSetup
    for name in _TENSOR_FIELDS:
        assert getattr(core, name) is getattr(legacy, name)
        assert getattr(rebuilt, name) is getattr(core, name)
    assert core.counting_function is legacy.counting_function
    assert rebuilt.counting_function is core.counting_function
    assert core.numbers.dtype == legacy.numbers.dtype
    assert not hasattr(core, "component")
    assert not hasattr(core, "param")
    with pytest.raises(FrozenInstanceError):
        core.r0 = torch.ones_like(core.r0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        core.counting_function = None  # type: ignore[assignment]


def test_core_dtype_device_and_noop_conversion() -> None:
    """Convert floating children while retaining integer numbers and callable."""
    _, _, _, _, legacy = _setup()
    core = core_srb_setup(legacy)
    assert core.to() is core
    assert core.to(dtype=core.dtype) is core

    for dtype in (torch.float32, torch.float64):
        converted = core.to(dtype=dtype)
        assert converted.numbers.dtype == legacy.numbers.dtype
        for name in _TENSOR_FIELDS[1:]:
            assert getattr(converted, name).dtype == dtype
        assert converted.counting_function is core.counting_function
    assert core.r0.dtype == torch.float64

    moved = core.to(device="meta")
    for name in _TENSOR_FIELDS:
        assert getattr(moved, name).device.type == "meta"
    assert moved.numbers.dtype == legacy.numbers.dtype
    assert moved.counting_function is core.counting_function


def test_core_stack_and_vmap_probe() -> None:
    """Stack matching Nodes and map a pure expression over their leaves."""
    _, _, _, _, legacy = _setup()
    core = core_srb_setup(legacy)
    stacked = stack((core, core))
    assert type(stacked) is ShortRangeBondCoreSetup
    for name in _TENSOR_FIELDS:
        assert getattr(stacked, name).shape[0] == 2
    assert stacked.counting_function is core.counting_function

    def probe(setup: ShortRangeBondCoreSetup) -> Tensor:
        return setup.r0.sum() + setup.prefactor.sum() + setup.enpoly.sum()

    actual = vmap(probe)(stacked)
    expected = torch.stack((probe(core), probe(core)))
    torch.testing.assert_close(actual, expected)


def test_get_energy_accepts_core_and_rejects_unrelated_object() -> None:
    """The compatibility component accepts only either declared setup."""
    _, positions, _, component, legacy = _setup()
    core = core_srb_setup(legacy)
    assert not isinstance(core, ShortRangeBondSetup)

    # Before the consumer update this call raised the established cache error:
    # the new Node was not a ShortRangeBondSetup instance.
    legacy_energy = component.get_energy(positions, legacy)
    core_energy = component.get_energy(positions, core)
    torch.testing.assert_close(core_energy, legacy_energy)
    torch.testing.assert_close(
        short_range_bond_energy(core, positions), legacy_energy
    )
    with pytest.raises(TypeError, match="ShortRangeBondSetup"):
        component.get_energy(positions, object())


def test_model_setup_owns_core_cache_alias_and_legacy_batch() -> None:
    """Single systems store one Node while packed systems retain dataclasses."""
    numbers, _, par, _, _ = _setup()
    system = Model(par=par, config=Config.create(exclude="scf")).setup(numbers)
    assert type(system.srb_setup) is ShortRangeBondCoreSetup
    assert system.classical_cache["ShortRangeBond"] is system.srb_setup
    direct_cache = system.classicals.get_cache(numbers, system.ihelp)
    assert type(direct_cache["ShortRangeBond"]) is ShortRangeBondSetup
    result = system.singlepoint(
        samples["NO2"]["positions"].to(device=DEVICE, dtype=torch.double)
    )
    assert "ShortRangeBond" in dict(result.classical)
    assert torch.count_nonzero(result.scf) == 0

    packed = Model(par=par, config=Config.create(exclude="scf")).setup(
        numbers.unsqueeze(0), batch_mode=1
    )
    assert type(packed.srb_setup) is ShortRangeBondSetup
    assert packed.classical_cache["ShortRangeBond"] is packed.srb_setup


def test_exact_replacement_and_setup_parameter_gradient() -> None:
    """An exact replacement feeds a differentiable System-owned Node."""
    numbers, positions, par, replacement, _ = _setup(differentiable=True)
    parameter = par.get("short_range.srb.prefactor")
    model = Model(
        par=par,
        config=Config.create(exclude=("scf", "srb")),
        classical=(replacement,),
    )
    system = model.setup(numbers)
    assert system.srb_classical is replacement
    assert type(system.srb_setup) is ShortRangeBondCoreSetup
    assert system.classical_cache["ShortRangeBond"] is system.srb_setup

    energy = system.singlepoint(positions).classical
    # The established Result API stores ordered (label, tensor) pairs.
    total = dict(energy)["ShortRangeBond"].sum()
    (gradient,) = torch.autograd.grad(total, parameter)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_core_position_gradient_matches_legacy() -> None:
    """The Node conversion preserves position and mixed derivatives."""
    _, positions, par, component, legacy = _setup(differentiable=True)
    core = core_srb_setup(legacy)
    parameter = par.get("short_range.srb.prefactor")

    def derivatives(setup):
        pos = positions.clone().requires_grad_(True)
        energy = short_range_bond_energy(setup, pos).sum()
        (position_gradient,) = torch.autograd.grad(
            energy, pos, create_graph=True
        )
        (parameter_gradient,) = torch.autograd.grad(
            energy, parameter, create_graph=True, retain_graph=True
        )
        direction = torch.arange(
            pos.numel(), device=pos.device, dtype=pos.dtype
        )
        mixed = torch.autograd.grad(
            (position_gradient * direction.reshape_as(pos)).sum(), parameter
        )[0]
        return energy, position_gradient, parameter_gradient, mixed

    expected = derivatives(legacy)
    actual = derivatives(core)
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs, rhs)
        assert torch.isfinite(lhs).all()
