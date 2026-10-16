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
"""System-owned Repulsion setup Node representation and parity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.func import jacfwd, jvp, vmap
from torch.utils import _pytree

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    RepulsionCoreSetup,
    RepulsionSetup,
    core_repulsion_setup,
    new_repulsion,
)
from dxtb._src.components.classicals.repulsion.base import repulsion_energy
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ...conftest import DEVICE
from .samples import samples

_FIELDS = ("arep", "zeff", "kexp", "mask", "cutoff")


def _setup(
    method=GFN1_XTB,
    name: str = "H2O",
    *,
    differentiable: bool = False,
    cutoff: float = 50.0,
) -> tuple[Tensor, Tensor, ParamModule, object, RepulsionSetup]:
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(device=DEVICE, dtype=torch.double)
    par = ParamModule(method, device=DEVICE, dtype=torch.double)
    if differentiable:
        par.set_differentiable("element.H.arep")
    repulsion = new_repulsion(torch.unique(numbers), par, cutoff=cutoff)
    assert repulsion is not None
    ihelp = IndexHelper.from_numbers(numbers, par)
    legacy = repulsion.get_cache(numbers, ihelp)
    assert type(legacy) is RepulsionSetup
    return numbers, positions, par, repulsion, legacy


def test_core_setup_schema_identity_and_roundtrip() -> None:
    """Keep the five setup tensors as dynamic identity-preserving leaves."""
    _, _, _, _, legacy = _setup()
    core = core_repulsion_setup(legacy)

    assert isinstance(core, Node)
    assert type(core) is RepulsionCoreSetup
    assert tuple(field.name for field in fields(core)) == _FIELDS
    leaves, spec = _pytree.tree_flatten(core)
    assert len(leaves) == len(_FIELDS)
    assert all(isinstance(leaf, Tensor) for leaf in leaves)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is RepulsionCoreSetup
    for name in _FIELDS:
        assert getattr(core, name) is getattr(legacy, name)
        assert getattr(rebuilt, name) is getattr(core, name)
    assert core.mask.dtype == torch.bool
    assert core.mask.shape == core.arep.shape
    assert not hasattr(core, "component")
    assert not hasattr(core, "param")
    with pytest.raises(FrozenInstanceError):
        core.arep = torch.ones_like(core.arep)  # type: ignore[misc]


def test_core_setup_dtype_and_device_conversion() -> None:
    """Convert floating children while preserving the Boolean pair mask."""
    _, _, _, _, legacy = _setup()
    core = core_repulsion_setup(legacy)

    assert core.to() is core
    assert core.to(dtype=core.dtype) is core
    for dtype in (torch.float32, torch.float64):
        converted = core.to(dtype=dtype)
        for name in ("arep", "zeff", "kexp", "cutoff"):
            assert getattr(converted, name).dtype == dtype
        assert converted.mask.dtype == torch.bool
    converted = core.to(device="meta")
    for name in _FIELDS:
        assert getattr(converted, name).device.type == "meta"
    assert converted.mask.dtype == torch.bool


def test_core_setup_stack_vmap_and_formula() -> None:
    """Stack matching setups and vmap the existing repulsion equation."""
    _, positions, _, _, legacy = _setup()
    core = core_repulsion_setup(legacy)
    batched = stack((core, core))
    assert type(batched) is RepulsionCoreSetup
    for name in _FIELDS:
        assert getattr(batched, name).shape[0] == 2
    assert batched.mask.dtype == torch.bool

    position_batch = torch.stack((positions, positions + 0.01))

    def evaluate(
        pos: Tensor,
        arep: Tensor,
        zeff: Tensor,
        kexp: Tensor,
        mask: Tensor,
        cutoff: Tensor,
    ) -> Tensor:
        return repulsion_energy(pos, mask, arep, kexp, zeff, cutoff)

    actual = vmap(evaluate)(
        position_batch,
        batched.arep,
        batched.zeff,
        batched.kexp,
        batched.mask,
        batched.cutoff,
    )
    expected = torch.stack(
        [
            repulsion_energy(
                pos,
                core.mask,
                core.arep,
                core.kexp,
                core.zeff,
                core.cutoff,
            )
            for pos in position_batch
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_core_setup_preserves_cutoff_selection() -> None:
    """The setup-time cutoff still controls the original Boolean selection."""
    _, positions, _, repulsion, legacy = _setup(cutoff=1.0)
    core = core_repulsion_setup(legacy)
    legacy_energy = repulsion.get_energy(positions, legacy)
    core_energy = repulsion.get_energy(positions, core)
    torch.testing.assert_close(core_energy, legacy_energy)
    assert torch.count_nonzero(core_energy) == 0


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_legacy_and_core_energy_and_gradient_parity(method) -> None:
    """The Node representation preserves repulsion values and derivatives."""
    _, positions, par, repulsion, legacy = _setup(method, differentiable=True)
    core = core_repulsion_setup(legacy)
    parameter = par.get("element.H.arep")

    pos_legacy = positions.clone().requires_grad_(True)
    legacy_energy = repulsion.get_energy(pos_legacy, legacy).sum()
    legacy_pos_grad, legacy_par_grad = torch.autograd.grad(
        legacy_energy,
        (pos_legacy, parameter),
        create_graph=True,
    )
    legacy_mixed = torch.autograd.grad(
        legacy_pos_grad.sum(), parameter, retain_graph=True
    )[0]

    pos_core = positions.clone().requires_grad_(True)
    core_energy = repulsion.get_energy(pos_core, core).sum()
    core_pos_grad, core_par_grad = torch.autograd.grad(
        core_energy,
        (pos_core, parameter),
        create_graph=True,
    )
    core_mixed = torch.autograd.grad(
        core_pos_grad.sum(), parameter, retain_graph=True
    )[0]

    torch.testing.assert_close(core_energy, legacy_energy)
    torch.testing.assert_close(core_pos_grad, legacy_pos_grad)
    torch.testing.assert_close(core_par_grad, legacy_par_grad)
    torch.testing.assert_close(core_mixed, legacy_mixed)
    assert torch.isfinite(core_pos_grad).all()
    assert torch.count_nonzero(core_par_grad) > 0


def test_core_position_jvp_and_jacfwd_match_legacy() -> None:
    """Forward-mode geometry derivatives use the same physical builder."""
    _, positions, _, repulsion, legacy = _setup()
    core = core_repulsion_setup(legacy)
    direction = torch.arange(
        positions.numel(), dtype=positions.dtype, device=positions.device
    ).reshape_as(positions)

    def energy(
        pos: Tensor, setup: RepulsionSetup | RepulsionCoreSetup
    ) -> Tensor:
        return repulsion.get_energy(pos, setup).sum()

    for setup in (legacy, core):
        _, tangent = jvp(
            lambda pos: energy(pos, setup), (positions,), (direction,)
        )
        jacobian = jacfwd(lambda pos: energy(pos, setup))(positions)
        assert torch.isfinite(tangent)
        assert torch.isfinite(jacobian).all()
        if setup is legacy:
            legacy_tangent = tangent
            legacy_jacobian = jacobian
        else:
            torch.testing.assert_close(tangent, legacy_tangent)
            torch.testing.assert_close(jacobian, legacy_jacobian)


def test_model_setup_owns_one_core_cache_for_single_system() -> None:
    """Single-System setup replaces the legacy cache entry exactly once."""
    numbers, _, par, _, _ = _setup()
    system = Model(par=par, config=Config.create(exclude="scf")).setup(numbers)

    assert type(system.repulsion_setup) is RepulsionCoreSetup
    assert system.classical_cache["Repulsion"] is system.repulsion_setup
    assert system.repulsion_classical is not None


def test_batch_setup_and_direct_cache_remain_legacy() -> None:
    """Direct and packed legacy cache APIs continue returning RepulsionSetup."""
    numbers = samples["H2"]["numbers"].to(DEVICE)
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.double)
    repulsion = new_repulsion(torch.unique(numbers), par)
    assert repulsion is not None
    ihelp = IndexHelper.from_numbers(numbers, par)
    assert type(repulsion.get_cache(numbers, ihelp)) is RepulsionSetup
    assert type(repulsion.get_cache(numbers, ihelp)) is RepulsionSetup

    packed_numbers = numbers.unsqueeze(0)
    system = Model(par=par, config=Config.create(exclude="scf")).setup(
        packed_numbers, batch_mode=1
    )
    assert type(system.repulsion_setup) is RepulsionSetup
    assert system.classical_cache["Repulsion"] is system.repulsion_setup
