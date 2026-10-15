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
"""Tests for the single-System GeneralizedBorn core setup Node."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.utils import _pytree as pytree

from dxtb import GFN1_XTB, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.interactions.solvation.alpb import (
    GeneralizedBorn,
    GeneralizedBornCache,
    GeneralizedBornCoreSetup,
    GeneralizedBornSetup,
    build_generalized_born_data,
    core_generalized_born_setup,
    setup_generalized_born,
)
from dxtb._src.typing import Tensor

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}
_FIELD_NAMES = (
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
)
_TENSOR_FIELDS = ("numbers", "alpbet", "keps", "rvdw", "cutoff", "obc")
_FLOAT_FIELDS = ("alpbet", "keps", "rvdw", "cutoff", "obc")


def _geometry() -> tuple[Tensor, Tensor]:
    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **DD)
    return numbers, positions


def _setup(
    *,
    kernel: str = "p16",
    alpb: bool = True,
    dielectric: Tensor | float = 78.9,
    born_scale: Tensor | float = 1.0,
    born_offset: Tensor | float = 0.0,
) -> tuple[GeneralizedBornSetup, GeneralizedBornCoreSetup, GeneralizedBorn]:
    numbers, _ = _geometry()
    component = GeneralizedBorn(
        numbers,
        dielectric,
        kernel=kernel,
        alpb=alpb,
        born_scale=born_scale,
        born_offset=born_offset,
        **DD,
    )
    legacy = setup_generalized_born(component, numbers)
    return legacy, core_generalized_born_setup(legacy), component


def _energy(
    setup: GeneralizedBornSetup | GeneralizedBornCoreSetup,
    positions: Tensor,
    charges: Tensor,
) -> Tensor:
    matrix = build_generalized_born_data(setup, positions).mat
    return 0.5 * (charges * (matrix @ charges)).sum()


def test_core_node_schema_round_trip_and_identity() -> None:
    """All setup fields survive a tensor-identity-preserving Node roundtrip."""
    legacy, core, _ = _setup()
    assert isinstance(core, Node)
    assert type(core) is GeneralizedBornCoreSetup
    assert tuple(field.name for field in fields(core)) == _FIELD_NAMES
    assert core._child_names == (
        "numbers",
        "alpbet",
        "keps",
        "rvdw",
        "born_scale",
        "born_offset",
        "cutoff",
        "descreening",
        "obc",
    )
    assert core._context_names == ("kernel", "apply_alpb_shape_correction")
    assert len(pytree.tree_leaves(core)) == len(_TENSOR_FIELDS)
    for name in _FIELD_NAMES:
        value = getattr(legacy, name)
        if isinstance(value, Tensor):
            assert getattr(core, name) is value
        else:
            assert getattr(core, name) == value

    leaves, spec = pytree.tree_flatten(core)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is GeneralizedBornCoreSetup
    for name in _FIELD_NAMES:
        value = getattr(core, name)
        if isinstance(value, Tensor):
            assert getattr(rebuilt, name) is value
        else:
            assert getattr(rebuilt, name) == value
    assert all(isinstance(leaf, Tensor) for leaf in leaves)
    assert all(
        leaf is getattr(core, name)
        for leaf, name in zip(leaves, _TENSOR_FIELDS)
    )
    with pytest.raises(FrozenInstanceError):
        core.keps = core.alpbet  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        core.kernel = "still"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["kernel", "apply_alpb_shape_correction"])
def test_static_setup_values_are_tree_context(field: str) -> None:
    """Kernel and correction choice participate in static Node structure."""
    _, p16, _ = _setup(kernel="p16", alpb=True)
    _, still, _ = _setup(kernel="still", alpb=True)
    _, no_correction, _ = _setup(kernel="p16", alpb=False)
    values = {
        "kernel": (p16, still),
        "apply_alpb_shape_correction": (p16, no_correction),
    }[field]
    left = pytree.tree_flatten(values[0])[1]
    right = pytree.tree_flatten(values[1])[1]
    assert left != right
    assert getattr(values[0], field) != getattr(values[1], field)


def test_scalar_and_tensor_born_settings_keep_their_layout() -> None:
    """Python scalars stay scalar while tensor parameters remain leaves."""
    legacy, core, _ = _setup()
    for name in ("born_scale", "born_offset", "descreening"):
        assert type(getattr(core, name)) is type(getattr(legacy, name))
        assert getattr(core, name) == getattr(legacy, name)
    integer_setup = core_generalized_born_setup(
        replace(legacy, born_scale=1, born_offset=0, descreening=1)
    )
    for name, value in (
        ("born_scale", 1),
        ("born_offset", 0),
        ("descreening", 1),
    ):
        assert getattr(integer_setup, name) == value
        assert type(getattr(integer_setup, name)) is int

    scale = torch.tensor(1.1, **DD, requires_grad=True)
    offset = torch.tensor(0.1, **DD, requires_grad=True)
    descreening = torch.tensor([0.8, 0.8], **DD, requires_grad=True)
    legacy, _, _ = _setup(born_scale=scale, born_offset=offset)
    legacy = replace(legacy, descreening=descreening)
    core = core_generalized_born_setup(legacy)
    leaves, spec = pytree.tree_flatten(core)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    for name in ("born_scale", "born_offset", "descreening"):
        assert getattr(core, name) is getattr(legacy, name)
        assert getattr(rebuilt, name) is getattr(legacy, name)
        assert any(leaf is getattr(core, name) for leaf in leaves)

    numbers, positions = _geometry()
    charges = torch.tensor([0.2, -0.2], **DD)
    energy = _energy(core, positions, charges)
    gradients = torch.autograd.grad(energy, (scale, offset, descreening))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) for gradient in gradients)


def test_mixed_dtype_tensor_born_settings_remain_supported() -> None:
    """Caller tensors may keep their dtype until an explicit Node conversion."""
    numbers, positions = _geometry()
    scale = torch.tensor(1.1, dtype=torch.float32, requires_grad=True)
    offset = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    descreening = torch.full((2,), 0.8, dtype=torch.float32, requires_grad=True)
    component = GeneralizedBorn(
        numbers,
        78.9,
        born_scale=scale,
        born_offset=offset,
        **DD,
    )
    legacy = replace(
        setup_generalized_born(component, numbers),
        descreening=descreening,
    )
    core = core_generalized_born_setup(legacy)
    for name in ("born_scale", "born_offset", "descreening"):
        assert getattr(core, name) is getattr(legacy, name)
        assert getattr(core, name).dtype is torch.float32

    charges = torch.tensor([0.2, -0.2], **DD)
    legacy_energy = _energy(legacy, positions, charges)
    core_energy = _energy(core, positions, charges)
    torch.testing.assert_close(core_energy, legacy_energy)
    gradients = torch.autograd.grad(core_energy, (scale, offset, descreening))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) for gradient in gradients)

    converted = core.to(dtype=torch.float64)
    for name in ("born_scale", "born_offset", "descreening"):
        assert getattr(converted, name).dtype is torch.float64


def test_dtype_device_and_noop_conversion() -> None:
    """Node conversion affects tensors only and preserves structural dtypes."""
    legacy, _, _ = _setup(
        born_scale=torch.tensor(1.0, **DD),
        born_offset=torch.tensor(0.1, **DD),
    )
    legacy = replace(
        legacy,
        descreening=torch.tensor([0.8, 0.8], **DD),
    )
    core = core_generalized_born_setup(legacy)
    assert core.to() is core
    assert core.to(dtype=core.dtype) is core
    converted = core.to(dtype=torch.float32)
    assert converted.numbers.dtype == core.numbers.dtype
    for name in (*_FLOAT_FIELDS, "born_scale", "born_offset", "descreening"):
        assert getattr(converted, name).dtype is torch.float32
    assert converted.kernel == core.kernel
    assert (
        converted.apply_alpb_shape_correction
        is core.apply_alpb_shape_correction
    )

    scalar_legacy, scalar_core, _ = _setup()
    scalar_converted = scalar_core.to(dtype=torch.float32)
    scalar_moved = scalar_core.to(device="meta")
    for name in ("born_scale", "born_offset", "descreening"):
        assert type(getattr(scalar_converted, name)) is type(
            getattr(scalar_legacy, name)
        )
        assert getattr(scalar_converted, name) == getattr(scalar_legacy, name)
        assert type(getattr(scalar_moved, name)) is type(
            getattr(scalar_legacy, name)
        )

    moved = core.to(device="meta")
    for name in (*_TENSOR_FIELDS, "born_scale", "born_offset", "descreening"):
        assert getattr(moved, name).device.type == "meta"
    assert moved.numbers.dtype == core.numbers.dtype
    assert moved.kernel == core.kernel
    assert moved.apply_alpb_shape_correction is core.apply_alpb_shape_correction


@pytest.mark.parametrize(
    ("kernel", "alpb"),
    [("p16", True), ("p16", False), ("still", True), ("still", False)],
)
def test_builder_matrix_energy_and_potential_parity(
    kernel: str, alpb: bool
) -> None:
    """Core and compatibility setups share matrix and derived results."""
    legacy, core, component = _setup(kernel=kernel, alpb=alpb)
    _, positions = _geometry()
    charges = torch.tensor([0.2, -0.2], **DD)
    legacy_data = build_generalized_born_data(legacy, positions)
    core_data = build_generalized_born_data(core, positions)
    assert isinstance(
        component.get_cache(numbers=legacy.numbers, positions=positions),
        GeneralizedBornCache,
    )
    torch.testing.assert_close(core_data.mat, legacy_data.mat)
    torch.testing.assert_close(
        0.5 * charges @ core_data.mat @ charges,
        0.5 * charges @ legacy_data.mat @ charges,
    )
    torch.testing.assert_close(
        component.get_monopole_atom_potential(core_data, charges),
        component.get_monopole_atom_potential(legacy_data, charges),
    )


def test_core_setup_preserves_geometry_and_dielectric_derivatives() -> None:
    """The Node conversion keeps position, dielectric and mixed gradients."""
    numbers, positions = _geometry()
    dielectric = torch.tensor(78.9, **DD, requires_grad=True)
    component = GeneralizedBorn(numbers, dielectric, alpb=True, **DD)
    legacy = setup_generalized_born(component, numbers)
    core = core_generalized_born_setup(legacy)
    charges = torch.tensor([0.2, -0.2], **DD)
    direction = torch.tensor([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]], **DD)

    pos = positions.clone().requires_grad_(True)
    value = _energy(core, pos, charges)
    position_grad, dielectric_grad = torch.autograd.grad(
        value, (pos, dielectric), create_graph=True
    )
    legacy_pos = positions.clone().requires_grad_(True)
    legacy_value = _energy(legacy, legacy_pos, charges)
    legacy_position_grad, legacy_dielectric_grad = torch.autograd.grad(
        legacy_value, (legacy_pos, dielectric), create_graph=True
    )
    torch.testing.assert_close(value, legacy_value)
    torch.testing.assert_close(position_grad, legacy_position_grad)
    torch.testing.assert_close(dielectric_grad, legacy_dielectric_grad)
    assert torch.isfinite(position_grad).all()
    assert torch.isfinite(dielectric_grad)
    mixed = torch.autograd.grad((position_grad * direction).sum(), dielectric)[
        0
    ]
    assert torch.isfinite(mixed)
    assert torch.count_nonzero(mixed) > 0

    def energy(pos_: Tensor) -> Tensor:
        return _energy(core, pos_, charges)

    def legacy_energy(pos_: Tensor) -> Tensor:
        return _energy(legacy, pos_, charges)

    _, tangent = torch.func.jvp(energy, (positions,), (direction,))
    _, legacy_tangent = torch.func.jvp(
        legacy_energy, (positions,), (direction,)
    )
    jacobian = torch.func.jacfwd(energy)(positions)
    legacy_jacobian = torch.func.jacfwd(legacy_energy)(positions)
    torch.testing.assert_close(tangent, legacy_tangent)
    torch.testing.assert_close(jacobian, legacy_jacobian)
    torch.testing.assert_close(tangent, (position_grad * direction).sum())
    torch.testing.assert_close(jacobian, position_grad)


def test_matching_core_setups_stack_and_vmap() -> None:
    """Matching setup Nodes stack and serve as transform inputs."""
    _, first, _ = _setup()
    _, second, _ = _setup()
    stacked = stack((first, second))
    assert type(stacked) is GeneralizedBornCoreSetup
    assert stacked.numbers.shape[0] == 2
    assert stacked.alpbet.shape[0] == 2
    assert stacked.keps.shape[0] == 2
    assert stacked.kernel == first.kernel

    def probe(setup: GeneralizedBornCoreSetup) -> Tensor:
        return setup.keps * setup.alpbet + setup.rvdw.sum()

    mapped = torch.func.vmap(probe)(stacked)
    loop = torch.stack([probe(first), probe(second)])
    torch.testing.assert_close(mapped, loop)


def test_model_setup_owns_node_and_absent_interaction_is_none() -> None:
    """The exact selected component produces one System-owned Node."""
    numbers, _ = _geometry()
    component = GeneralizedBorn(numbers, 78.9, **DD)
    system = Model(
        par=ParamModule(GFN1_XTB, **DD), interaction=(component,)
    ).setup(numbers)
    assert type(system.generalized_born_setup) is GeneralizedBornCoreSetup
    assert system.generalized_born_interaction is component
    assert not hasattr(system, "generalized_born_legacy_setup")

    absent = Model(par=ParamModule(GFN1_XTB, **DD)).setup(numbers)
    assert absent.generalized_born_setup is None
    assert absent.generalized_born_interaction is None


def test_system_setup_is_stable_across_geometry_history() -> None:
    """Geometry-dependent matrices stay local while System retains its Node."""
    numbers, positions_a = _geometry()
    positions_b = positions_a.clone()
    positions_b[1, 0] = 0.2
    component = GeneralizedBorn(numbers, 78.9, **DD)
    system = Model(
        par=ParamModule(GFN1_XTB, **DD), interaction=(component,)
    ).setup(numbers)
    setup = system.generalized_born_setup
    assert type(setup) is GeneralizedBornCoreSetup

    first = _interaction_data(system, positions_a)[component.label].mat
    second = _interaction_data(system, positions_b)[component.label].mat
    repeated = _interaction_data(system, positions_a)[component.label].mat
    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, second)
    assert system.generalized_born_setup is setup
    for name in _FIELD_NAMES:
        assert getattr(system.generalized_born_setup, name) is getattr(
            setup, name
        )
