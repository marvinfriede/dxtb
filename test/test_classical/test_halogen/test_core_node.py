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
"""Halogen core setup Node and parity tests."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.utils import _pytree

from dxtb import GFN1_XTB, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    Halogen,
    HalogenCoreSetup,
    HalogenSetup,
    core_halogen_setup,
    halogen_energy,
    new_halogen,
)
from dxtb._src.components.classicals.halogen.hal import setup_halogen
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ...conftest import DEVICE
from ...utils import get_param_module
from .samples import samples

DD = {"device": DEVICE, "dtype": torch.double}
EXCLUDE = (
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
FLOAT_FIELDS = ("xbond", "atomic_radii", "damp", "cutoff")
MASK_FIELDS = (
    "halogen_mask",
    "base_mask",
    "valid_atom_mask",
    "pair_type_mask",
    "neighbor_type_mask",
)
SETUP_FIELDS = (
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
)


def _geometry() -> tuple[Tensor, Tensor]:
    sample = samples["br2nh3"]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _model(par: ParamModule | None = None) -> Model:
    if par is None:
        par = ParamModule(GFN1_XTB.model_copy(deep=True), **DD)
    return Model(par=par, config=Config.create(exclude=EXCLUDE))


def _tensor_leaves(setup: HalogenCoreSetup) -> list[Tensor]:
    leaves, _ = _pytree.tree_flatten(setup)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def test_halogen_core_node_schema_identity_and_roundtrip() -> None:
    """The core Node preserves all setup tensors as ordered dynamic leaves."""
    numbers, _ = _geometry()
    par = ParamModule(GFN1_XTB.model_copy(deep=True), **DD)
    component = new_halogen(torch.unique(numbers), par, **DD)
    assert component is not None
    legacy = setup_halogen(
        component, numbers, IndexHelper.from_numbers(numbers, par)
    )
    core = core_halogen_setup(legacy)

    assert isinstance(core, Node)
    assert type(core) is HalogenCoreSetup
    assert tuple(field.name for field in fields(core)) == SETUP_FIELDS
    assert all(
        getattr(core, name) is getattr(legacy, name) for name in SETUP_FIELDS
    )

    leaves = _tensor_leaves(core)
    assert len(leaves) == 10
    assert all(
        actual is getattr(legacy, name)
        for actual, name in zip(leaves, SETUP_FIELDS)
    )
    assert core.numbers.dtype == numbers.dtype
    assert all(getattr(core, name).dtype == torch.bool for name in MASK_FIELDS)

    rebuilt_leaves, spec = _pytree.tree_flatten(core)
    rebuilt = _pytree.tree_unflatten(rebuilt_leaves, spec)
    assert type(rebuilt) is HalogenCoreSetup
    assert all(
        getattr(rebuilt, name) is getattr(legacy, name) for name in SETUP_FIELDS
    )


def test_halogen_core_node_dtype_and_device_conversion() -> None:
    """Node conversions preserve integer and Boolean fields."""
    numbers, _ = _geometry()
    system = _model().setup(numbers)
    assert isinstance(system.halogen_setup, HalogenCoreSetup)
    core = system.halogen_setup

    for dtype in (torch.float32, torch.float64):
        converted = core.to(dtype=dtype)
        assert type(converted) is HalogenCoreSetup
        assert converted.numbers.dtype == numbers.dtype
        assert all(
            getattr(converted, name).dtype == dtype for name in FLOAT_FIELDS
        )
        assert all(
            getattr(converted, name).dtype == torch.bool for name in MASK_FIELDS
        )

    meta = core.to(device=torch.device("meta"))
    assert type(meta) is HalogenCoreSetup
    assert meta.numbers.device.type == "meta"
    assert meta.numbers.dtype == numbers.dtype
    assert all(
        getattr(meta, name).device.type == "meta" for name in SETUP_FIELDS
    )
    assert all(getattr(meta, name).dtype == torch.bool for name in MASK_FIELDS)


def test_halogen_core_node_stacking_and_vmap_probe() -> None:
    """Compatible setup Nodes stack and expose transformable tensor leaves."""
    numbers, _ = _geometry()
    system = _model().setup(numbers)
    assert isinstance(system.halogen_setup, HalogenCoreSetup)
    core = system.halogen_setup
    stacked = stack((core, core))
    assert type(stacked) is HalogenCoreSetup
    for source, target in zip(_tensor_leaves(core), _tensor_leaves(stacked)):
        assert target.shape == (2, *source.shape)
    assert all(
        getattr(stacked, name).dtype == torch.bool for name in MASK_FIELDS
    )

    def probe(setup: HalogenCoreSetup) -> Tensor:
        return (
            setup.xbond.sum()
            + setup.atomic_radii.sum()
            + setup.damp.sum()
            + setup.cutoff.sum()
            + setup.pair_type_mask.sum()
            + setup.neighbor_type_mask.sum()
        )

    mapped = torch.func.vmap(probe)(stacked)
    explicit = torch.stack((probe(core), probe(core)))
    torch.testing.assert_close(mapped, explicit)


def test_halogen_core_energy_and_derivative_parity() -> None:
    """Core and legacy setup share Halogen energy and derivative behavior."""
    numbers, positions = _geometry()
    par = ParamModule(GFN1_XTB.model_copy(deep=True), **DD)
    damp = par.get("halogen.classical.damping")
    damp.requires_grad_(True)
    component = new_halogen(torch.unique(numbers), par, **DD)
    assert component is not None
    legacy = setup_halogen(
        component, numbers, IndexHelper.from_numbers(numbers, par)
    )
    core = core_halogen_setup(legacy)
    assert not torch.isclose(
        halogen_energy(core, positions).sum(), torch.zeros((), **DD)
    )
    torch.testing.assert_close(
        halogen_energy(core, positions), halogen_energy(legacy, positions)
    )
    torch.testing.assert_close(
        component.get_energy(positions, core),
        component.get_energy(positions, legacy),
    )

    direction = torch.linspace(-0.2, 0.3, positions.numel(), **DD).reshape_as(
        positions
    )

    def energy(pos: Tensor, setup: HalogenSetup | HalogenCoreSetup) -> Tensor:
        return halogen_energy(setup, pos).sum()

    legacy_positions = positions.clone().requires_grad_(True)
    legacy_energy = energy(legacy_positions, legacy)
    (legacy_grad,) = torch.autograd.grad(
        legacy_energy, legacy_positions, create_graph=True
    )
    (legacy_mixed,) = torch.autograd.grad(
        (legacy_grad * direction).sum(), damp, retain_graph=True
    )
    core_positions = positions.clone().requires_grad_(True)
    core_energy = energy(core_positions, core)
    (core_grad,) = torch.autograd.grad(
        core_energy, core_positions, create_graph=True
    )
    (core_mixed,) = torch.autograd.grad((core_grad * direction).sum(), damp)
    torch.testing.assert_close(core_grad, legacy_grad)
    torch.testing.assert_close(core_mixed, legacy_mixed)
    assert torch.isfinite(core_mixed)

    from torch.func import jacfwd, jvp

    _, tangent = jvp(lambda pos: energy(pos, core), (positions,), (direction,))
    torch.testing.assert_close(
        tangent, (legacy_grad.detach() * direction).sum()
    )
    torch.testing.assert_close(
        jacfwd(lambda pos: energy(pos, core))(positions), legacy_grad.detach()
    )


def test_halogen_system_ownership_legacy_batch_and_type_validation() -> None:
    """System Nodes alias the cache while legacy caches and batches remain."""
    numbers, positions = _geometry()
    system = _model().setup(numbers)
    assert type(system.halogen_setup) is HalogenCoreSetup
    assert system.classical_cache["Halogen"] is system.halogen_setup
    assert system.halogen_classical is not None
    legacy = system.halogen_classical.get_cache(numbers, system.ihelp)
    assert type(legacy) is HalogenSetup
    torch.testing.assert_close(
        system.halogen_classical.get_energy(positions, system.halogen_setup),
        system.halogen_classical.get_energy(positions, legacy),
    )
    with pytest.raises(TypeError, match="HalogenSetup.*HalogenCoreSetup"):
        system.halogen_classical.get_energy(positions, object())  # type: ignore[arg-type]

    stacked = stack((system.halogen_setup, system.halogen_setup))
    with pytest.raises(ValueError, match="one System at a time"):
        system.halogen_classical.get_energy(positions, stacked)
