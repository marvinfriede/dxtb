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
Test force calculation.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck
from tad_mctc.batch import pack
from tad_mctc.convert import tensor_to_numpy

from dxtb import GFN1_XTB, Calculator
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import scf_strategy
from .samples import samples

slist = [
    "H",
    "LiH",
    "H2O",
    pytest.param("CH4", marks=pytest.mark.slow),
    pytest.param("SiH4", marks=pytest.mark.slow),
]
slist_large = ["HHe", "PbH4-BiH3", "MB16_43_01"]  # "LYS_xao"

opts = {
    "fermi_diff_order": 0,
    "maxiter": 100,
    "strategy": scf_strategy("anderson"),
    "differentiation": "unrolled",
    "verbosity": 0,
    "x_atol": 1e-10,
}


# FIXME: Fails with "Numerical gradient for function expected to be zero"
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def skip_test_autograd(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # required for autodiff of energy w.r.t. positions
    pos = positions.clone().requires_grad_(True)

    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)

    def f(p: Tensor) -> Tensor:
        return calc.forces(p, charge)

    assert dgradcheck(f, pos)


def single(
    name: str,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    execute(numbers, positions, charge, dd, atol, rtol)


def batched(
    name1: str,
    name2: str,
    dd: DD,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ],
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ],
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    execute(numbers, positions, charge, dd, atol, rtol)


def execute(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    dd: DD,
    atol: float,
    rtol: float,
) -> None:
    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)

    num_forces = calc.forces_numerical(positions, charge)
    assert num_forces.grad_fn is None
    numforces = tensor_to_numpy(num_forces)

    pos = positions.clone().detach().requires_grad_(True)
    forces1 = tensor_to_numpy(calc.forces(pos, charge))
    assert pos.grad is None
    assert pytest.approx(numforces, abs=atol, rel=rtol) == forces1

    calc.reset()
    pos2 = positions.clone().detach().requires_grad_(True)
    forces2 = tensor_to_numpy(calc.get_forces(pos2, charge))
    assert pos2.grad is None
    assert pytest.approx(numforces, abs=atol, rel=rtol) == forces2
    assert pytest.approx(forces1, abs=atol, rel=rtol) == forces2

    if numbers.ndim == 1:
        pos3 = positions.clone().detach()
        forces3 = tensor_to_numpy(calc.system.forces(pos3, charge))
        assert pytest.approx(forces1, abs=atol, rel=rtol) == forces3


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    single(name, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", slist_large)
def test_single_large(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    single(name, dd=dd)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    batched(name1, name2, dd=dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", slist_large)
def test_batch_large(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    batched(name1, name2, dd=dd)
