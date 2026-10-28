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
Testing symmetry of (higher) multipole integrals.
"""

from __future__ import annotations

from itertools import permutations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor

if has_libcint is True:
    from dxtb._src.exlibs import libcint

from ..conftest import DEVICE
from ..utils import get_param_module
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "S2", "H2O", "SiH4"]
mp_ints = ["jj"]


def check_multipole_symmetry(multipole_tensor: Tensor) -> bool:
    """
    Check the symmetry of a multipole moment tensor.

    Args:
    multipole_tensor: A PyTorch tensor that represents a multipole moment.

    Returns:
    A Boolean value indicating whether the tensor is symmetric.
    """

    # Get the number of dimensions in the tensor
    ndim = len(multipole_tensor.shape)

    # Get all permutations of the first (ndim-2) dimensions
    perms = permutations(range(ndim - 2))

    # Check symmetry for all permutations
    is_symmetric = all(
        torch.allclose(
            multipole_tensor,
            multipole_tensor.permute(*perm, ndim - 2, ndim - 1),
        )
        for perm in perms
    )

    return is_symmetric


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("intstr", mp_ints)
def test_single(dtype: torch.dtype, intstr: str, name: str) -> None:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    par = get_param_module("gfn1", **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)

    libcint_basis, structure = bas.create_libcint(positions)
    i = libcint.int1e(intstr, libcint_basis, structure, packed=False)

    nao = ihelp.nao
    mpdims = (3, 3)
    i = i.reshape((*mpdims, nao, nao))

    assert i.shape == torch.Size((*mpdims, ihelp.nao, ihelp.nao))
    assert check_multipole_symmetry(i) is True


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("intstr", mp_ints)
def test_batch(dtype: torch.dtype, name1: str, name2: str, intstr: str) -> None:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = samples[name1], samples[name2]
    par = get_param_module("gfn1", **dd)
    int_list = []
    ao_counts = []
    for sample in (sample1, sample2):
        numbers = sample["numbers"].to(DEVICE)
        positions = sample["positions"].to(**dd)
        ihelp = IndexHelper.from_numbers(numbers, par)
        bas = Basis(numbers, par, ihelp, **dd)
        libcint_basis, structure = bas.create_libcint(positions)
        i = libcint.int1e(intstr, libcint_basis, structure, packed=False)
        int_list.append(i.reshape((3, 3, ihelp.nao, ihelp.nao)))
        ao_counts.append(ihelp.nao)

    i = pack(int_list)

    assert i.shape == torch.Size((2, 3, 3, max(ao_counts), max(ao_counts)))
