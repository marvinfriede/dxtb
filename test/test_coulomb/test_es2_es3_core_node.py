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
"""Node migration checks for System-owned ES2 and ES3 setup values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.func import vmap
from torch.utils import _pytree

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb.config import Config
from dxtb._src.components.interactions.coulomb import (
    ES2CoreSetup,
    ES2Setup,
    ES3CoreSetup,
    ES3Setup,
    build_es2_coulomb,
    core_es2_setup,
    core_es3_setup,
    new_es2,
    setup_es2,
    setup_es3,
)
from dxtb._src.components.interactions.coulomb.secondorder import ES2Cache
from dxtb._src.components.interactions.coulomb.thirdorder import ES3Cache

from ..molecules import mols


def _system_inputs(dtype: torch.dtype = torch.float64):
    numbers = mols["H2O"]["numbers"].clone()
    positions = mols["H2O"]["positions"].to(dtype=dtype)
    return numbers, positions


def _es2_setup(shell_resolved: bool):
    numbers, positions = _system_inputs()
    par = ParamModule(GFN1_XTB, dtype=positions.dtype)
    ihelp = IndexHelper.from_numbers(numbers, par)
    interaction = new_es2(
        torch.unique(numbers),
        par,
        shell_resolved=shell_resolved,
        dtype=positions.dtype,
    )
    assert interaction is not None
    legacy = setup_es2(
        numbers,
        interaction.hubbard,
        ihelp,
        lhubbard=interaction.lhubbard,
        gexp=interaction.gexp,
        average=interaction.average,
        shell_resolved=interaction.shell_resolved,
    )
    return (
        numbers,
        positions,
        ihelp,
        interaction,
        legacy,
        core_es2_setup(legacy),
    )


def _es3_setups(shell_resolved: bool):
    numbers, _ = _system_inputs()
    par = ParamModule(GFN2_XTB if shell_resolved else GFN1_XTB)
    ihelp = IndexHelper.from_numbers(numbers, par)
    model = Model(par=par)
    system = model.setup(numbers)
    interaction = system.es3_interaction
    assert interaction is not None
    legacy = setup_es3(
        interaction.hubbard_derivs,
        ihelp,
        shell_scale=interaction.shell_scale if shell_resolved else None,
    )
    return numbers, ihelp, interaction, legacy, core_es3_setup(legacy)


@pytest.mark.parametrize("shell_resolved", [False, True])
def test_es2_core_node_layout_roundtrip_and_conversion(
    shell_resolved: bool,
) -> None:
    """Keep ES2 tensor leaves, static configuration and identities intact."""
    _, _, _, interaction, legacy, core = _es2_setup(shell_resolved)
    assert isinstance(core, Node)
    assert isinstance(core, ES2CoreSetup)
    assert core.average is legacy.average
    assert core.shell_resolved is legacy.shell_resolved

    leaves, spec = _pytree.tree_flatten(core)
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is ES2CoreSetup
    assert rebuilt.average is core.average
    assert rebuilt.shell_resolved is core.shell_resolved
    for name in (
        "atom_pair_mask",
        "atom_mask",
        "atom_hubbard",
        "shell_pair_mask",
        "shell_mask",
        "shell_hubbard",
        "shells_to_atom",
        "gexp",
    ):
        original = getattr(core, name)
        assert getattr(rebuilt, name) is original

    assert core.atom_pair_mask.dtype == torch.bool
    assert core.atom_mask.dtype == torch.bool
    assert core.gexp.dtype == torch.float64
    if shell_resolved:
        assert core.atom_hubbard is None
        assert core.shell_pair_mask is not None
        assert core.shell_mask is not None
        assert core.shell_hubbard is not None
        assert core.shells_to_atom is not None
        assert core.shell_pair_mask.dtype == torch.bool
        assert core.shell_mask.dtype == torch.bool
        assert core.shells_to_atom.dtype in (torch.int32, torch.int64)
    else:
        assert core.atom_hubbard is not None
        assert core.shell_pair_mask is None
        assert core.shell_mask is None
        assert core.shell_hubbard is None
        assert core.shells_to_atom is None

    assert core.gexp is legacy.gexp
    assert core.gexp.data_ptr() != interaction.gexp.data_ptr()


@pytest.mark.parametrize("shell_resolved", [False, True])
def test_es2_core_matrix_and_gradients_match_legacy(
    shell_resolved: bool,
) -> None:
    """Use the same Coulomb builder and preserve position/parameter gradients."""
    numbers, positions, ihelp, interaction, _, _ = _es2_setup(shell_resolved)
    gexp = interaction.gexp.detach().clone().requires_grad_(True)
    hubbard = interaction.hubbard.detach().clone().requires_grad_(True)
    legacy = setup_es2(
        numbers,
        hubbard,
        ihelp,
        lhubbard=interaction.lhubbard,
        gexp=gexp,
        average=interaction.average,
        shell_resolved=shell_resolved,
    )
    core = core_es2_setup(legacy)
    torch.testing.assert_close(
        build_es2_coulomb(core, positions), build_es2_coulomb(legacy, positions)
    )

    pos_legacy = positions.clone().requires_grad_(True)
    pos_core = positions.clone().requires_grad_(True)
    mat_legacy = build_es2_coulomb(legacy, pos_legacy)
    mat_core = build_es2_coulomb(core, pos_core)
    (grad_legacy,) = torch.autograd.grad(mat_legacy.square().sum(), pos_legacy)
    (grad_core,) = torch.autograd.grad(mat_core.square().sum(), pos_core)
    torch.testing.assert_close(grad_core, grad_legacy)

    grad_legacy = torch.autograd.grad(
        build_es2_coulomb(legacy, positions).square().sum(),
        (gexp, hubbard),
        retain_graph=True,
    )
    grad_core = torch.autograd.grad(
        build_es2_coulomb(core, positions).square().sum(),
        (gexp, hubbard),
    )
    for actual, expected in zip(grad_core, grad_legacy):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shell_resolved", [False, True])
def test_es3_core_node_roundtrip_and_formula_parity(
    shell_resolved: bool,
) -> None:
    """Preserve ES3 resolution, setup leaves and established formulas."""
    _, _, interaction, legacy, core = _es3_setups(shell_resolved)
    assert isinstance(core, Node)
    assert isinstance(core, ES3CoreSetup)
    assert core.shell_resolved is shell_resolved
    assert legacy.shell_resolved is shell_resolved
    assert core.hubbard_derivs is legacy.hubbard_derivs

    leaves, spec = _pytree.tree_flatten(core)
    assert len(leaves) == 1
    assert leaves[0] is legacy.hubbard_derivs
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is ES3CoreSetup
    assert rebuilt.hubbard_derivs is core.hubbard_derivs
    assert rebuilt.shell_resolved is shell_resolved

    charges = torch.linspace(
        -0.2,
        0.3,
        core.hubbard_derivs.numel(),
        dtype=core.hubbard_derivs.dtype,
    )
    legacy_cache = ES3Cache(
        legacy.hubbard_derivs, shell_resolved=legacy.shell_resolved
    )
    core_cache = ES3Cache(
        core.hubbard_derivs, shell_resolved=core.shell_resolved
    )
    energy_fn = (
        interaction.get_monopole_shell_energy
        if shell_resolved
        else interaction.get_monopole_atom_energy
    )
    potential_fn = (
        interaction.get_monopole_shell_potential
        if shell_resolved
        else interaction.get_monopole_atom_potential
    )
    torch.testing.assert_close(
        energy_fn(core_cache, charges), energy_fn(legacy_cache, charges)
    )
    torch.testing.assert_close(
        potential_fn(core_cache, charges), potential_fn(legacy_cache, charges)
    )

    source = interaction.hubbard_derivs.detach().clone().requires_grad_(True)
    system_index = IndexHelper.from_numbers(
        _system_inputs()[0],
        ParamModule(GFN2_XTB if shell_resolved else GFN1_XTB),
    )
    legacy_grad_setup = setup_es3(
        source,
        system_index,
        shell_scale=interaction.shell_scale if shell_resolved else None,
    )
    core_grad_setup = core_es3_setup(legacy_grad_setup)
    (gradient,) = torch.autograd.grad(
        core_grad_setup.hubbard_derivs.sum(), source
    )
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_system_setup_owns_core_nodes_and_legacy_setup_contracts_remain() -> (
    None
):
    """Convert once for System while preserving the legacy setup/cache API."""
    numbers, positions = _system_inputs()
    par = ParamModule(GFN1_XTB, dtype=positions.dtype)
    system = Model(par=par).setup(numbers)
    assert isinstance(system.es2_setup, ES2CoreSetup)
    assert isinstance(system.es3_setup, ES3CoreSetup)
    assert not hasattr(system, "es2_legacy_setup")
    assert not hasattr(system, "es3_legacy_setup")

    es2_interaction = system.es2_interaction
    es3_interaction = system.es3_interaction
    assert es2_interaction is not None and es3_interaction is not None
    es2_cache = es2_interaction.get_cache(
        numbers=numbers, positions=positions, ihelp=system.ihelp
    )
    es3_cache = es3_interaction.get_cache(numbers=numbers, ihelp=system.ihelp)
    assert isinstance(es2_cache, ES2Cache)
    assert isinstance(es3_cache, ES3Cache)

    setup_es3_positional = ES3Setup(
        system.es3_setup.hubbard_derivs, shell_resolved=False
    )
    assert isinstance(setup_es3_positional, ES3Setup)
    legacy_es3 = setup_es3(es3_interaction.hubbard_derivs, system.ihelp)
    assert isinstance(legacy_es3, ES3Setup)
    assert not legacy_es3.shell_resolved


def test_setup_nodes_device_dtype_and_frozen_semantics() -> None:
    """Delegate conversion and immutability semantics to tad-mctc Node."""
    _, _, _, _, _, es2 = _es2_setup(shell_resolved=True)
    _, _, _, _, es3 = _es3_setups(shell_resolved=True)
    for setup in (es2, es3):
        assert setup.to() is setup
        assert setup.to(device="cpu") is setup
        assert setup.to(dtype=setup.dtype) is setup
        converted = setup.to(device="meta")
        converted_leaves = _pytree.tree_leaves(converted)
        assert converted_leaves
        assert all(leaf.device.type == "meta" for leaf in converted_leaves)

        dtype_converted = setup.to(dtype=torch.float32)
        assert all(
            leaf.dtype == torch.float32
            for leaf in _pytree.tree_leaves(dtype_converted)
            if leaf.is_floating_point()
        )
        for leaf in _pytree.tree_leaves(dtype_converted):
            if leaf.dtype == torch.bool:
                assert leaf.dtype is torch.bool
        if isinstance(setup, ES2CoreSetup):
            assert (
                dtype_converted.shells_to_atom.dtype
                == setup.shells_to_atom.dtype
            )
            assert dtype_converted.average is setup.average
            assert dtype_converted.shell_resolved is setup.shell_resolved
            with pytest.raises(FrozenInstanceError):
                setup.gexp = torch.ones_like(setup.gexp)  # type: ignore[misc]
        else:
            assert dtype_converted.shell_resolved is setup.shell_resolved
            with pytest.raises(FrozenInstanceError):
                setup.hubbard_derivs = torch.ones_like(  # type: ignore[misc]
                    setup.hubbard_derivs
                )


def test_unhashable_average_keeps_identity_and_legacy_cache_support() -> None:
    """Keep custom callable support while making its Node context hashable."""

    class CustomAverage:
        __hash__ = None

        def __call__(self, values: torch.Tensor) -> torch.Tensor:
            return 0.5 * (values.unsqueeze(-1) + values.unsqueeze(-2))

    numbers, positions, ihelp, interaction, _, _ = _es2_setup(False)
    average = CustomAverage()
    interaction.average = average
    legacy = setup_es2(
        numbers,
        interaction.hubbard,
        ihelp,
        gexp=interaction.gexp,
        average=average,
        shell_resolved=False,
    )
    core = core_es2_setup(legacy)
    assert core.average is average
    assert _pytree.tree_leaves(core)
    torch.testing.assert_close(
        build_es2_coulomb(core, positions), build_es2_coulomb(legacy, positions)
    )

    cache = interaction.get_cache(
        numbers=numbers, positions=positions, ihelp=ihelp
    )
    assert isinstance(cache, ES2Cache)
    torch.testing.assert_close(cache.mat, build_es2_coulomb(legacy, positions))

    par = ParamModule(GFN1_XTB, dtype=positions.dtype)
    system = Model(
        par=par,
        config=Config.create(exclude="es2"),
        interaction=(interaction,),
    ).setup(numbers)
    assert isinstance(system.es2_setup, ES2CoreSetup)
    assert system.es2_setup.average is average


def test_core_setup_stack_and_vmap_preserve_static_context() -> None:
    """Map tensor children while keeping ES2/ES3 static context fixed."""
    _, _, _, _, _, es2 = _es2_setup(shell_resolved=False)
    _, _, _, _, es3 = _es3_setups(shell_resolved=False)
    stacked_es2 = stack((es2, es2))
    stacked_es3 = stack((es3, es3))
    assert isinstance(stacked_es2, ES2CoreSetup)
    assert isinstance(stacked_es3, ES3CoreSetup)
    assert stacked_es2.average is es2.average
    assert not stacked_es2.shell_resolved
    assert not stacked_es3.shell_resolved
    assert stacked_es2.atom_hubbard.shape[0] == 2
    assert stacked_es3.hubbard_derivs.shape[0] == 2

    mapped_es2 = vmap(lambda setup: setup.atom_hubbard.sum())(stacked_es2)
    mapped_es3 = vmap(lambda setup: setup.hubbard_derivs.sum())(stacked_es3)
    torch.testing.assert_close(
        mapped_es2,
        torch.stack((es2.atom_hubbard.sum(), es2.atom_hubbard.sum())),
    )
    torch.testing.assert_close(
        mapped_es3,
        torch.stack((es3.hubbard_derivs.sum(), es3.hubbard_derivs.sum())),
    )
