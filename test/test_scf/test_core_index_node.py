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
"""Node contract and legacy parity for single-System core index mappings."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import stack as tree_stack
from torch.func import vmap

from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.basis.indexhelper import IndexHelper
from dxtb._src.scf.functional.core import CoreIndexData

from ..molecules import mols
from ..utils import get_param_module

_SYSTEMS = (
    pytest.param(GFN1_XTB, "H2O", id="gfn1-h2o"),
    pytest.param(GFN2_XTB, "NO2", id="gfn2-no2"),
    pytest.param(GFN1_XTB, "CH4", id="gfn1-ch4"),
)


def _indices(method, molecule: str) -> tuple[IndexHelper, CoreIndexData]:
    numbers = mols[molecule]["numbers"].to(device="cpu")
    par = get_param_module(
        "gfn1" if method == GFN1_XTB else "gfn2",
        device="cpu",
        dtype=torch.float64,
    )
    legacy = IndexHelper.from_numbers(numbers, par, batch_mode=0)
    core = CoreIndexData(
        shells_to_atom=legacy.shells_to_atom,
        orbitals_to_shell=legacy.orbitals_to_shell,
    )
    return legacy, core


@pytest.mark.parametrize("method,molecule", _SYSTEMS)
def test_core_index_node_layout_conversion_and_stack(method, molecule) -> None:
    legacy, index = _indices(method, molecule)

    assert index._child_names == ("shells_to_atom", "orbitals_to_shell")
    assert index._context_names == ()
    assert index._keep_dtype_names == {
        "shells_to_atom",
        "orbitals_to_shell",
    }
    assert not any(
        hasattr(index, name)
        for name in (
            "batch_mode",
            "cull",
            "restore",
            "update",
            "reset",
            "invalidate",
        )
    )

    leaves, spec = torch.utils._pytree.tree_flatten(index)
    assert len(leaves) == 2
    assert leaves[0] is index.shells_to_atom
    assert leaves[1] is index.orbitals_to_shell
    assert spec.type is CoreIndexData
    assert spec.context == (
        ("shells_to_atom", "orbitals_to_shell"),
        (),
        (),
    )
    rebuilt = torch.utils._pytree.tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, CoreIndexData)
    for name in ("shells_to_atom", "orbitals_to_shell"):
        actual = getattr(rebuilt, name)
        expected = getattr(index, name)
        assert actual.dtype == expected.dtype == torch.long
        assert actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    converted = index.to(dtype=torch.float32)
    assert converted is index
    for name in ("shells_to_atom", "orbitals_to_shell"):
        assert getattr(converted, name).dtype == torch.long
    cpu = index.to(device="cpu")
    assert isinstance(cpu, CoreIndexData)
    assert cpu.shells_to_atom.device.type == "cpu"
    assert cpu.orbitals_to_shell.device.type == "cpu"

    stacked = tree_stack((index, rebuilt))
    assert isinstance(stacked, CoreIndexData)
    assert stacked.shells_to_atom.shape[0] == 2
    assert stacked.orbitals_to_shell.shape[0] == 2
    assert stacked.shells_to_atom.dtype == torch.long
    assert stacked.orbitals_to_shell.dtype == torch.long


@pytest.mark.parametrize("method,molecule", _SYSTEMS)
def test_core_index_methods_match_indexhelper_and_gradients(
    method, molecule, capsys
):
    legacy, index = _indices(method, molecule)
    dtype = torch.float64
    nat = mols[molecule]["numbers"].numel()
    nshell = legacy.orbitals_per_shell.numel()
    norb = legacy.orbitals_to_shell.numel()

    orbital = torch.linspace(-0.4, 0.7, norb, dtype=dtype)
    shell = torch.linspace(-0.2, 0.5, nshell, dtype=dtype)
    atom = torch.linspace(0.1, 0.8, nat, dtype=dtype)

    ordinary = (
        (
            index.reduce_orbital_to_shell(orbital),
            legacy.reduce_orbital_to_shell(orbital),
        ),
        (index.reduce_shell_to_atom(shell), legacy.reduce_shell_to_atom(shell)),
        (
            index.reduce_orbital_to_atom(orbital),
            legacy.reduce_orbital_to_atom(orbital),
        ),
        (index.spread_atom_to_shell(atom), legacy.spread_atom_to_shell(atom)),
        (
            index.spread_shell_to_orbital(shell),
            legacy.spread_shell_to_orbital(shell),
        ),
        (
            index.spread_atom_to_orbital(atom),
            legacy.spread_atom_to_orbital(atom),
        ),
    )
    for actual, expected in ordinary:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Match the non-default dim / extra convention used by multipole reduction
    # in functional SCF charge construction.
    orbital_multipoles = torch.arange(2 * norb * 6, dtype=dtype).reshape(
        2, norb, 6
    )
    shell_multipoles = torch.arange(2 * nshell * 6, dtype=dtype).reshape(
        2, nshell, 6
    )
    atom_multipoles = torch.arange(2 * nat * 6, dtype=dtype).reshape(2, nat, 6)
    extra = (
        (
            index.reduce_orbital_to_shell(
                orbital_multipoles, dim=-2, extra=True
            ),
            legacy.reduce_orbital_to_shell(
                orbital_multipoles, dim=-2, extra=True
            ),
        ),
        (
            index.reduce_shell_to_atom(shell_multipoles, dim=-2, extra=True),
            legacy.reduce_shell_to_atom(shell_multipoles, dim=-2, extra=True),
        ),
        (
            index.reduce_orbital_to_atom(
                orbital_multipoles, dim=-2, extra=True
            ),
            legacy.reduce_orbital_to_atom(
                orbital_multipoles, dim=-2, extra=True
            ),
        ),
        (
            index.spread_atom_to_shell(atom_multipoles, dim=-2, extra=True),
            legacy.spread_atom_to_shell(atom_multipoles, dim=-2, extra=True),
        ),
        (
            index.spread_shell_to_orbital(shell_multipoles, dim=-2, extra=True),
            legacy.spread_shell_to_orbital(
                shell_multipoles, dim=-2, extra=True
            ),
        ),
        (
            index.spread_atom_to_orbital(atom_multipoles, dim=-2, extra=True),
            legacy.spread_atom_to_orbital(atom_multipoles, dim=-2, extra=True),
        ),
    )
    for actual, expected in extra:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    core_values = torch.randn((2, norb, 6), dtype=dtype, requires_grad=True)
    legacy_values = core_values.detach().clone().requires_grad_(True)
    core_output = index.reduce_orbital_to_atom(core_values, dim=-2, extra=True)
    legacy_output = legacy.reduce_orbital_to_atom(
        legacy_values, dim=-2, extra=True
    )
    weights = torch.arange(core_output.numel(), dtype=dtype).reshape_as(
        core_output
    )
    core_gradient = torch.autograd.grad(
        (core_output * weights).sum(), core_values
    )[0]
    legacy_gradient = torch.autograd.grad(
        (legacy_output * weights).sum(), legacy_values
    )[0]
    torch.testing.assert_close(core_gradient, legacy_gradient, rtol=0, atol=0)

    batched = torch.stack((orbital, orbital * 1.7, orbital - 0.3))
    torch._C._debug_only_display_vmap_fallback_warnings(True)
    try:
        mapped = vmap(lambda values: index.reduce_orbital_to_atom(values))(
            batched
        )
    finally:
        torch._C._debug_only_display_vmap_fallback_warnings(False)
    assert not capsys.readouterr().err
    loop = torch.stack(
        [index.reduce_orbital_to_atom(values) for values in batched]
    )
    torch.testing.assert_close(mapped, loop, rtol=0, atol=0)
