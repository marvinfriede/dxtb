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
"""System-owned AES2 setup Node representation and parity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.func import jacfwd, jvp, vmap
from torch.utils import _pytree

from dxtb.config import Config
from dxtb._src.calculators.model import Model
from dxtb._src.components.interactions.coulomb.multipole import AES2Cache
from dxtb._src.components.interactions.coulomb import (
    AES2,
    AES2CoreSetup,
    build_aes2_data,
    core_aes2_setup,
    new_aes2,
    setup_aes2,
)
from dxtb._src.typing import Tensor

from .test_aes2_setup import _charges, _energy, _inputs

_FIELDS = (
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
)
_FLOAT_FIELDS = (
    "dmp3",
    "dmp5",
    "shift",
    "kexp",
    "rmax",
    "dkernel",
    "qkernel",
    "rad",
    "vcn",
)


def test_aes2_core_setup_roundtrip_and_tensor_identity() -> None:
    """Keep the full AES2 setup as eleven dynamic tensor leaves."""
    numbers, _, _, ihelp, interaction = _inputs()
    legacy = setup_aes2(interaction, numbers, ihelp)
    core = core_aes2_setup(legacy)

    assert isinstance(core, Node)
    assert type(core) is AES2CoreSetup
    assert tuple(field.name for field in fields(core)) == _FIELDS
    assert all(isinstance(getattr(core, name), Tensor) for name in _FIELDS)
    leaves, spec = _pytree.tree_flatten(core)
    assert len(leaves) == len(_FIELDS) == 11
    assert all(isinstance(leaf, Tensor) for leaf in leaves)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is AES2CoreSetup
    for name in _FIELDS:
        assert getattr(core, name) is getattr(legacy, name)
        assert getattr(rebuilt, name) is getattr(core, name)

    assert core.numbers.dtype == legacy.numbers.dtype
    assert core.pair_mask.dtype == legacy.pair_mask.dtype == torch.bool
    for name in _FLOAT_FIELDS:
        assert getattr(core, name).dtype == getattr(legacy, name).dtype

    with pytest.raises(FrozenInstanceError):
        core.dmp3 = torch.ones_like(core.dmp3)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        core.numbers = torch.ones_like(core.numbers)  # type: ignore[misc]


def test_aes2_core_setup_device_and_dtype_conversion() -> None:
    """Move all leaves while preserving integer and Boolean dtypes."""
    numbers, _, _, ihelp, interaction = _inputs()
    legacy = setup_aes2(interaction, numbers, ihelp)
    core = core_aes2_setup(legacy)

    assert core.to() is core
    assert core.to(dtype=core.dtype) is core
    converted = core.to(dtype=torch.float32)
    assert converted.numbers.dtype == core.numbers.dtype
    assert converted.pair_mask.dtype == torch.bool
    for name in _FLOAT_FIELDS:
        assert getattr(converted, name).dtype == torch.float32

    on_meta = core.to(device="meta")
    assert on_meta.numbers.device.type == "meta"
    assert on_meta.pair_mask.device.type == "meta"
    assert on_meta.numbers.dtype == core.numbers.dtype
    assert on_meta.pair_mask.dtype == torch.bool
    for name in _FLOAT_FIELDS:
        assert getattr(on_meta, name).device.type == "meta"
        assert getattr(on_meta, name).dtype == core.dmp3.dtype

    # setup_aes2's established scalar clones isolate component storage; the
    # Node conversion itself keeps exactly those setup tensor objects.
    for name in ("dmp3", "dmp5", "shift", "kexp", "rmax"):
        assert getattr(core, name) is getattr(legacy, name)
        assert (
            getattr(core, name).data_ptr()
            != getattr(interaction, name).data_ptr()
        )


def test_aes2_core_data_builder_matches_legacy_and_keeps_validation() -> None:
    """Build identical call-local geometry data from either setup value."""
    numbers, positions, _, ihelp, interaction = _inputs()
    legacy = setup_aes2(interaction, numbers, ihelp)
    core = core_aes2_setup(legacy)
    assert isinstance(build_aes2_data(legacy, positions), AES2Cache)
    expected = build_aes2_data(legacy, positions)
    actual = build_aes2_data(core, positions)
    for name in ("mrad", "dkernel", "qkernel", "amat_sd", "amat_dd", "amat_sq"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name)
        )

    bad_positions = positions[:-1]
    with pytest.raises(ValueError, match="positions must have shape"):
        build_aes2_data(legacy, bad_positions)
    with pytest.raises(ValueError, match="positions must have shape"):
        build_aes2_data(core, bad_positions)

    with pytest.raises(RuntimeError, match="share a dtype"):
        build_aes2_data(core, positions.to(dtype=torch.float32))


def test_aes2_core_energy_position_derivatives_match_legacy() -> None:
    """Preserve reverse, higher reverse, JVP, and jacfwd responses."""
    numbers, positions, _, ihelp, interaction = _inputs()
    legacy = setup_aes2(interaction, numbers, ihelp)
    core = core_aes2_setup(legacy)
    charges = _charges(ihelp, numbers)

    legacy_fn = lambda pos: _energy(legacy, pos, interaction, charges, ihelp)
    core_fn = lambda pos: _energy(core, pos, interaction, charges, ihelp)
    torch.testing.assert_close(core_fn(positions), legacy_fn(positions))

    pos_legacy = positions.clone().requires_grad_(True)
    pos_core = positions.clone().requires_grad_(True)
    (grad_legacy,) = torch.autograd.grad(
        legacy_fn(pos_legacy), pos_legacy, create_graph=True
    )
    (grad_core,) = torch.autograd.grad(
        core_fn(pos_core), pos_core, create_graph=True
    )
    torch.testing.assert_close(grad_core, grad_legacy)
    (second_legacy,) = torch.autograd.grad(
        grad_legacy.sum(), pos_legacy, create_graph=True
    )
    (second_core,) = torch.autograd.grad(
        grad_core.sum(), pos_core, create_graph=True
    )
    torch.testing.assert_close(second_core, second_legacy)
    (third_legacy,) = torch.autograd.grad(second_legacy.sum(), pos_legacy)
    (third_core,) = torch.autograd.grad(second_core.sum(), pos_core)
    torch.testing.assert_close(third_core, third_legacy)

    direction = torch.linspace(-0.2, 0.3, positions.numel()).reshape_as(
        positions
    )
    _, tangent_legacy = jvp(legacy_fn, (positions,), (direction,))
    _, tangent_core = jvp(core_fn, (positions,), (direction,))
    torch.testing.assert_close(tangent_core, tangent_legacy)
    torch.testing.assert_close(jacfwd(core_fn)(positions), grad_core.detach())


@pytest.mark.parametrize(
    "parameter",
    (
        "multipole.damped.dmp3",
        "multipole.damped.dmp5",
        "multipole.damped.shift",
        "multipole.damped.kexp",
        "multipole.damped.rmax",
        "element.O.dkernel",
        "element.O.qkernel",
        "element.O.mprad",
        "element.O.mpvcn",
    ),
)
def test_aes2_core_setup_parameter_gradients_match_legacy(
    parameter: str,
) -> None:
    """Keep gradients from every setup parameter family connected."""
    numbers, positions, parameters, ihelp, interaction = _inputs()
    leaf = parameters.get(parameter)
    leaf.requires_grad_(True)
    interaction = new_aes2(
        torch.unique(numbers),
        parameters,
        device=numbers.device,
        dtype=positions.dtype,
    )
    assert interaction is not None
    legacy = setup_aes2(interaction, numbers, ihelp)
    core = core_aes2_setup(legacy)
    charges = _charges(ihelp, numbers)

    (legacy_gradient,) = torch.autograd.grad(
        _energy(legacy, positions, interaction, charges, ihelp),
        leaf,
        retain_graph=True,
    )
    (core_gradient,) = torch.autograd.grad(
        _energy(core, positions, interaction, charges, ihelp), leaf
    )
    torch.testing.assert_close(core_gradient, legacy_gradient)
    assert torch.isfinite(core_gradient).all()
    assert torch.count_nonzero(core_gradient) > 0


def test_aes2_core_setup_mixed_parameter_geometry_derivative() -> None:
    """Preserve mixed position and dmp3 response through the core Node."""
    numbers, positions, parameters, ihelp, interaction = _inputs()
    leaf = parameters.get("multipole.damped.dmp3")
    leaf.requires_grad_(True)
    interaction = new_aes2(
        torch.unique(numbers),
        parameters,
        device=numbers.device,
        dtype=positions.dtype,
    )
    assert interaction is not None
    legacy = setup_aes2(interaction, numbers, ihelp)
    core = core_aes2_setup(legacy)
    charges = _charges(ihelp, numbers)
    pos = positions.clone().requires_grad_(True)
    energy = _energy(core, pos, interaction, charges, ihelp)
    (position_gradient,) = torch.autograd.grad(energy, pos, create_graph=True)
    direction = torch.arange(
        positions.numel(), dtype=positions.dtype
    ).reshape_as(positions)
    (mixed_core,) = torch.autograd.grad(
        (position_gradient * direction).sum(), leaf
    )

    pos_legacy = positions.clone().requires_grad_(True)
    legacy_energy = _energy(legacy, pos_legacy, interaction, charges, ihelp)
    (legacy_position_gradient,) = torch.autograd.grad(
        legacy_energy, pos_legacy, create_graph=True
    )
    (mixed_legacy,) = torch.autograd.grad(
        (legacy_position_gradient * direction).sum(), leaf
    )
    torch.testing.assert_close(mixed_core, mixed_legacy)
    assert torch.isfinite(mixed_core).all()


def test_aes2_core_setup_system_ownership_and_legacy_cache() -> None:
    """Store one core Node while retaining the legacy AES2 cache route."""
    numbers, positions, parameters, ihelp, interaction = _inputs()
    system = Model(par=parameters).setup(numbers)
    assert type(system.aes2_setup) is AES2CoreSetup
    assert system.aes2_interaction is not None
    assert any(
        component is system.aes2_interaction
        for component in system.interactions.components
    )
    assert not hasattr(system, "aes2_legacy_setup")

    selected = system.aes2_interaction
    assert {field.name for field in fields(system.aes2_setup)} == set(_FIELDS)
    assert system.aes2_setup is not None
    legacy_setup = setup_aes2(selected, numbers, ihelp)
    cache = selected.get_cache(
        numbers=numbers, positions=positions, ihelp=ihelp
    )
    assert type(cache) is AES2Cache
    expected = build_aes2_data(legacy_setup, positions)
    for name in ("mrad", "dkernel", "qkernel", "amat_sd", "amat_dd", "amat_sq"):
        torch.testing.assert_close(
            getattr(cache, name), getattr(expected, name)
        )

    excluded = Model(
        par=parameters,
        config=Config.create(exclude=("aes2", "d4sc", "disp")),
    ).setup(numbers)
    assert excluded.aes2_setup is None
    assert excluded.aes2_interaction is None


def test_aes2_core_setup_uses_selected_replacement_component() -> None:
    """Convert the selected exact AES2 replacement rather than defaults."""
    numbers, _, parameters, _, base = _inputs()
    custom = AES2(
        dmp3=base.dmp3 * 1.7,
        dmp5=base.dmp5,
        dkernel=base.dkernel,
        qkernel=base.qkernel,
        shift=base.shift,
        kexp=base.kexp,
        rmax=base.rmax,
        rad=base.rad,
        vcn=base.vcn,
        device=numbers.device,
        dtype=base.dmp3.dtype,
    )
    system = Model(
        par=parameters,
        config=Config.create(exclude=("aes2", "d4sc", "disp")),
        interaction=(custom,),
    ).setup(numbers)

    assert type(system.aes2_setup) is AES2CoreSetup
    assert system.aes2_interaction is custom
    assert system.aes2_setup is not None
    torch.testing.assert_close(system.aes2_setup.dmp3, custom.dmp3)
    assert not torch.isclose(system.aes2_setup.dmp3, base.dmp3)


def test_aes2_core_setup_stack_and_vmap() -> None:
    """Stack matching setup Nodes and map a pure tensor probe over them."""
    numbers, _, _, ihelp, interaction = _inputs()
    core = core_aes2_setup(setup_aes2(interaction, numbers, ihelp))
    batched = stack((core, core))
    assert type(batched) is AES2CoreSetup
    for name in _FIELDS:
        assert getattr(batched, name).shape[0] == 2
    assert batched.numbers.dtype == core.numbers.dtype
    assert batched.pair_mask.dtype == torch.bool

    def probe(setup: AES2CoreSetup) -> Tensor:
        return (
            setup.dkernel.sum()
            + setup.qkernel.sum()
            + setup.rad.sum()
            + setup.numbers.sum().to(setup.dkernel.dtype)
            + setup.pair_mask.sum().to(setup.dkernel.dtype)
        )

    mapped = vmap(probe)(batched)
    expected = torch.stack((probe(core), probe(core)))
    torch.testing.assert_close(mapped, expected)
