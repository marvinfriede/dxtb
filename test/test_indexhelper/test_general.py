# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""
General tests for IndexHelper covering instantiation, changing dtypes, and
moving to devices.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.param import Param
from dxtb._src.param.element import get_elem_angular

from ..conftest import DEVICE


def test_valid_init_integer_dtypes() -> None:
    numbers = torch.tensor([6, 1, 1], device=DEVICE)
    base = IndexHelper.from_numbers_angular(numbers, {1: [0], 6: [0, 1]})

    for int_dtype in (torch.int16, torch.int32, torch.int64):
        ihelp = IndexHelper(
            unique_angular=base.unique_angular.to(dtype=int_dtype),
            angular=base.angular.to(dtype=int_dtype),
            atom_to_unique=base.atom_to_unique.to(dtype=int_dtype),
            ushells_to_unique=base.ushells_to_unique.to(dtype=int_dtype),
            ushells_per_unique=base.ushells_per_unique.to(dtype=int_dtype),
            shells_to_ushell=base.shells_to_ushell.to(dtype=int_dtype),
            shells_per_atom=base.shells_per_atom.to(dtype=int_dtype),
            shell_index=base.shell_index.to(dtype=int_dtype),
            shells_to_atom=base.shells_to_atom.to(dtype=int_dtype),
            orbitals_per_shell=base.orbitals_per_shell.to(dtype=int_dtype),
            orbital_index=base.orbital_index.to(dtype=int_dtype),
            orbitals_to_shell=base.orbitals_to_shell.to(dtype=int_dtype),
            batch_mode=0,
        )
        assert ihelp.unique_angular.dtype == int_dtype
        assert ihelp.shells_to_atom.dtype == int_dtype
        assert ihelp.orbitals_to_shell.dtype == int_dtype


def test_fail_init_dtype() -> None:
    numbers = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    # Floating-point tensor is rejected.
    with pytest.raises(ValueError):
        IndexHelper(
            unique_angular=ihelp.unique_angular.type(torch.float),
            angular=ihelp.angular,
            atom_to_unique=ihelp.atom_to_unique,
            ushells_to_unique=ihelp.ushells_to_unique,
            ushells_per_unique=ihelp.ushells_per_unique,
            shells_to_ushell=ihelp.shells_to_ushell,
            shells_per_atom=ihelp.shells_per_atom,
            shell_index=ihelp.shell_index,
            shells_to_atom=ihelp.shells_to_atom,
            orbitals_per_shell=ihelp.orbitals_per_shell,
            orbital_index=ihelp.orbital_index,
            orbitals_to_shell=ihelp.orbitals_to_shell,
            batch_mode=0,
        )

    # Mixed integer dtypes are rejected.
    with pytest.raises(ValueError, match="All tensors must have same dtype"):
        IndexHelper(
            unique_angular=ihelp.unique_angular.to(dtype=torch.int32),
            angular=ihelp.angular,
            atom_to_unique=ihelp.atom_to_unique,
            ushells_to_unique=ihelp.ushells_to_unique,
            ushells_per_unique=ihelp.ushells_per_unique,
            shells_to_ushell=ihelp.shells_to_ushell,
            shells_per_atom=ihelp.shells_per_atom,
            shell_index=ihelp.shell_index,
            shells_to_atom=ihelp.shells_to_atom,
            orbitals_per_shell=ihelp.orbitals_per_shell,
            orbital_index=ihelp.orbital_index,
            orbitals_to_shell=ihelp.orbitals_to_shell,
            batch_mode=0,
        )


def test_fail_init_device_meta() -> None:
    numbers = torch.tensor([1], device=torch.device("cpu"))
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    with pytest.raises(
        ValueError, match="All tensors must be on the same device"
    ):
        IndexHelper(
            unique_angular=ihelp.unique_angular.to(torch.device("meta")),
            angular=ihelp.angular,
            atom_to_unique=ihelp.atom_to_unique,
            ushells_to_unique=ihelp.ushells_to_unique,
            ushells_per_unique=ihelp.ushells_per_unique,
            shells_to_ushell=ihelp.shells_to_ushell,
            shells_per_atom=ihelp.shells_per_atom,
            shell_index=ihelp.shell_index,
            shells_to_atom=ihelp.shells_to_atom,
            orbitals_per_shell=ihelp.orbitals_per_shell,
            orbital_index=ihelp.orbital_index,
            orbitals_to_shell=ihelp.orbitals_to_shell,
            batch_mode=0,
        )


@pytest.mark.cuda
def test_fail_init_device() -> None:
    numbers = torch.tensor([1], device=torch.device("cpu"))
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    with pytest.raises(ValueError):
        IndexHelper(
            unique_angular=ihelp.unique_angular.to(str_to_device("cuda")),
            angular=ihelp.angular,
            atom_to_unique=ihelp.atom_to_unique,
            ushells_to_unique=ihelp.ushells_to_unique,
            ushells_per_unique=ihelp.ushells_per_unique,
            shells_to_ushell=ihelp.shells_to_ushell,
            shells_per_atom=ihelp.shells_per_atom,
            shell_index=ihelp.shell_index,
            shells_to_atom=ihelp.shells_to_atom,
            orbitals_per_shell=ihelp.orbitals_per_shell,
            orbital_index=ihelp.orbital_index,
            orbitals_to_shell=ihelp.orbitals_to_shell,
            batch_mode=0,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """
    Under the ``tad_mctc.tree.Node`` contract, ``IndexHelper`` children are
    registered with ``keep_dtype=True``, so ``.type(dtype)`` and
    ``.to(dtype=dtype)`` preserve their integer tensor dtypes unchanged.
    """
    numbers = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})
    orig_int_dtype = ihelp.unique_angular.dtype

    converted = ihelp.type(dtype)
    assert converted.unique_angular.dtype == orig_int_dtype
    assert converted.shells_to_atom.dtype == orig_int_dtype
    assert converted.orbitals_to_shell.dtype == orig_int_dtype


def test_change_type_fail() -> None:
    """
    Under the ``tad_mctc.tree.Node`` contract, ``.dtype`` raises
    ``AttributeError`` when a ``Node`` has no floating-point tensor leaves,
    and setting ``.dtype`` fails on the frozen dataclass.
    """
    numbers = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    with pytest.raises(AttributeError):
        _ = ihelp.dtype

    with pytest.raises(AttributeError):
        ihelp.dtype = torch.float64  # type: ignore[misc]


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)

    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]}).to(device)
    assert ihelp.device == device


def test_change_device_fail() -> None:
    ihelp = IndexHelper.from_numbers_angular(torch.tensor([1]), {1: [0]})

    # trying to use setter
    with pytest.raises(AttributeError):
        ihelp.device = "cpu"


@pytest.mark.parametrize("par", [GFN0_XTB, GFN1_XTB, GFN2_XTB])
def test_from_numbers_param_matches_module(par: Param) -> None:
    """Raw `Param` and `ParamModule` yield the same index helper."""
    numbers = torch.tensor([1, 6, 8, 14, 17, 26, 79], device=DEVICE)

    parmod = ParamModule(par)
    assert get_elem_angular(par.element) == parmod.get_elem_angular()

    ihelp_par = IndexHelper.from_numbers(numbers, par)
    ihelp_mod = IndexHelper.from_numbers(numbers, parmod)

    assert (ihelp_par.angular == ihelp_mod.angular).all()
    assert (ihelp_par.shells_to_atom == ihelp_mod.shells_to_atom).all()
    assert (ihelp_par.orbitals_to_shell == ihelp_mod.orbitals_to_shell).all()


@pytest.mark.parametrize("module", [False, True])
def test_from_numbers_unknown_shell(module: bool) -> None:
    """Raw `Param` and `ParamModule` reject unknown shell types alike."""
    par = GFN1_XTB.model_copy(deep=True)
    par.element["H"].shells = ["1s", "5h"]

    numbers = torch.tensor([1], device=DEVICE)
    with pytest.raises(ValueError, match="Unknown shell type 'h'"):
        IndexHelper.from_numbers(numbers, ParamModule(par) if module else par)


def test_fail_init_shape() -> None:
    numbers = torch.tensor([6, 1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0], 6: [0, 1]})

    with pytest.raises(ValueError):
        IndexHelper(
            unique_angular=ihelp.unique_angular,
            angular=ihelp.angular,
            atom_to_unique=ihelp.atom_to_unique,
            ushells_to_unique=ihelp.ushells_to_unique,
            ushells_per_unique=ihelp.ushells_per_unique,
            shells_to_ushell=ihelp.shells_to_ushell,
            shells_per_atom=ihelp.shells_per_atom[:1],
            shell_index=ihelp.shell_index,
            shells_to_atom=ihelp.shells_to_atom,
            orbitals_per_shell=ihelp.orbitals_per_shell,
            orbital_index=ihelp.orbital_index,
            orbitals_to_shell=ihelp.orbitals_to_shell,
            batch_mode=0,
        )
