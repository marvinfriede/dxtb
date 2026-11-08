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
Test for SCF.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import Calculator
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..utils import get_param_module, scf_strategy
from .samples import samples

opts = {
    "verbosity": 0,
    "maxiter": 300,
    "differentiation": "unrolled",
    "strategy": scf_strategy("anderson", "potential"),
}


def single(
    dtype: torch.dtype,
    name: str,
    converger: str,
    tol: float,
    variable: str,
    differentiation: str,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["egfn1"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if converger == "linear" else 0.4,
            "strategy": scf_strategy(converger, variable),
            "differentiation": differentiation,
            "x_atol": tol,
            "int_driver": "pytorch",
        },
    )
    calc = Calculator(
        numbers, get_param_module("gfn1", **dd), opts=options, **dd
    )

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH"])
@pytest.mark.parametrize("converger", ["anderson", "linear"])
@pytest.mark.parametrize("variable", ["charge", "potential", "fock"])
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_single(
    dtype: torch.dtype,
    name: str,
    converger: str,
    variable: str,
    differentiation: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    single(dtype, name, converger, tol, variable, differentiation)


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "SiH4", "MB16_43_01", "LYS_xao"])
@pytest.mark.parametrize("converger", ["anderson", "linear"])
@pytest.mark.parametrize("variable", ["charge", "potential", "fock"])
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_single_medium(
    dtype: torch.dtype,
    name: str,
    converger: str,
    variable: str,
    differentiation: str,
) -> None:
    """Test a few larger system."""
    tol = sqrt(torch.finfo(dtype).eps) * 50
    single(dtype, name, converger, tol, variable, differentiation)


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["S2", "LYS_xao_dist"])
@pytest.mark.parametrize("converger", ["anderson", "linear"])
@pytest.mark.parametrize("variable", ["charge", "potential", "fock"])
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_single_difficult(
    dtype: torch.dtype,
    name: str,
    converger: str,
    variable: str,
    differentiation: str,
) -> None:
    """These systems do not reproduce tblite energies to high accuracy."""
    tol = 5e-3
    single(dtype, name, converger, tol, variable, differentiation)


# Convergers are compared on C60; the (much slower) vancoh2 only runs with
# the Anderson converger. All SCF variables run on both systems.
large_cases = [
    (name, converger, variable)
    for name, convergers in (
        ("C60", ["anderson", "linear"]),
        ("vancoh2", ["anderson"]),
    )
    for converger in convergers
    for variable in ["charge", "potential", "fock"]
]


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name, converger, variable", large_cases)
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_single_large(
    dtype: torch.dtype,
    name: str,
    converger: str,
    variable: str,
    differentiation: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    single(dtype, name, converger, tol, variable, differentiation)


def batched(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    converger: str,
    variable: str,
    differentiation: str,
    tol: float,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name1], samples[name2]
    numbers = pack(
        (
            sample[0]["numbers"].to(DEVICE),
            sample[1]["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample[0]["positions"].to(**dd),
            sample[1]["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample[0]["egfn1"].to(**dd),
            sample[1]["egfn1"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.05 if converger == "linear" else 0.4,
            "strategy": scf_strategy(converger, variable),
            "differentiation": differentiation,
            "x_atol": tol,
        },
    )
    calc = Calculator(
        numbers, get_param_module("gfn1", **dd), opts=options, **dd
    )

    result = calc.singlepoint(positions, charges)
    res = result.scf.sum(-1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == res.cpu()


# Unrolled (full gradient tracking) runs are not combined with Broyden.
converger_differentiations = [
    pytest.param(
        converger,
        differentiation,
        marks=(pytest.mark.slow,) if converger == "linear" else (),
    )
    for differentiation in ["unrolled", "implicit"]
    for converger in ["anderson", "broyden", "linear"]
    if not (differentiation == "unrolled" and converger == "broyden")
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["SiH4"])  # padding
@pytest.mark.parametrize("variable", ["charge", "potential", "fock"])
@pytest.mark.parametrize(
    "converger, differentiation", converger_differentiations
)
def test_batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    converger: str,
    variable: str,
    differentiation: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    batched(dtype, name1, name2, converger, variable, differentiation, tol)


# A batch of identical molecules (no padding) only needs every converger and
# SCF variable once; the full grid runs with padding in `test_batch`.
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("variable", ["charge", "potential", "fock"])
@pytest.mark.parametrize("converger", ["anderson", "broyden", "linear"])
@pytest.mark.parametrize("differentiation", ["implicit"])
def test_batch_no_padding(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    converger: str,
    variable: str,
    differentiation: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    batched(dtype, name1, name2, converger, variable, differentiation, tol)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("name3", ["SiH4"])
@pytest.mark.parametrize(
    "converger", ["anderson", pytest.param("linear", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("variable", ["charge", "potential", "fock"])
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_batch_three(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    name3: str,
    converger: str,
    variable: str,
    differentiation: str,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name1], samples[name2], samples[name3]
    numbers = pack(
        (
            sample[0]["numbers"].to(DEVICE),
            sample[1]["numbers"].to(DEVICE),
            sample[2]["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample[0]["positions"].to(**dd),
            sample[1]["positions"].to(**dd),
            sample[2]["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample[0]["egfn1"].to(**dd),
            sample[1]["egfn1"].to(**dd),
            sample[2]["egfn1"].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0, 0.0], **dd)

    options = dict(
        opts,
        **{
            "damp": 0.1 if converger == "linear" else 0.4,
            "strategy": scf_strategy(converger, variable),
            "differentiation": differentiation,
            "x_atol": tol,
        },
    )
    calc = Calculator(
        numbers, get_param_module("gfn1", **dd), opts=options, **dd
    )

    result = calc.singlepoint(positions, charges)
    assert pytest.approx(ref.cpu(), abs=tol) == result.scf.sum(-1).cpu()
