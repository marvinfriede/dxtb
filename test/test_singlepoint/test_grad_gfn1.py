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
Run tests for singlepoint gradient calculation with read from coord file.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import torch
from tad_mctc import read, read_chrg

from dxtb import Calculator
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import get_param_module, load_from_npz, scf_strategy

HERE = Path(__file__).resolve().parent / "refs" / "gfn1"
ref_grad = np.load(HERE / "grad.npz")
"""['H2', 'H2O', 'CH4', 'SiH4', 'LYS_xao', 'AD7en+', 'C60', 'vancoh2']"""


opts = {
    "maxiter": 50,
    "differentiation": "implicit",
    "strategy": scf_strategy("anderson", "potential"),
    "verbosity": 0,
}


def _strategy(differentiation: str):
    """Anderson for unrolled, Broyden for implicit SCF differentiation."""
    converger = "anderson" if differentiation == "unrolled" else "broyden"
    return scf_strategy(converger, "potential")


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "CH4"])
@pytest.mark.parametrize("differentiation", ["implicit", "unrolled"])
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_analytical(
    dtype: torch.dtype, name: str, differentiation: str
) -> None:
    atol, rtol = 1e-5, 1e-4
    analytical(dtype, name, atol, rtol, differentiation)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["C60"])
@pytest.mark.parametrize("differentiation", ["implicit", "unrolled"])
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_analytical_large(
    dtype: torch.dtype, name: str, differentiation: str
) -> None:
    atol = rtol = sqrt(torch.finfo(dtype).eps)
    analytical(dtype, name, atol, rtol, differentiation)


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["AD7en+", "LYS_xao"])
@pytest.mark.parametrize("differentiation", ["implicit", "unrolled"])
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_analytical_large2(
    dtype: torch.dtype, name: str, differentiation: str
) -> None:
    if "cuda" in str(DEVICE) and dtype == torch.float:
        atol = 1e-4
    else:
        atol = 1e-5

    rtol = 1e-3
    analytical(dtype, name, atol, rtol, differentiation)


def analytical(
    dtype: torch.dtype,
    name: str,
    atol: float,
    rtol: float,
    differentiation: str,
    int_driver: str | None = None,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    _struct = read(Path(base, "coord"), **dd)
    numbers, positions = _struct.numbers, _struct.positions
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    positions = positions.clone().requires_grad_(True)

    options = dict(
        opts,
        **{
            "strategy": _strategy(differentiation),
            "x_atol": 1e-5 if dtype == torch.float else 1e-10,
            "differentiation": differentiation,
        },
    )
    if int_driver is not None:
        options["int_driver"] = int_driver

    calc = Calculator(
        numbers, get_param_module("gfn1", **dd), opts=options, **dd
    )
    result = -calc.forces(positions, charge)
    gradient = result.detach()

    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(ref.cpu(), abs=atol, rel=rtol) == gradient.cpu()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["H2", "H2O", "SiH4"])
@pytest.mark.parametrize("differentiation", ["implicit", "unrolled"])
def test_backward(dtype: torch.dtype, name: str, differentiation: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 50  # slightly larger for H2O!
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    _struct = read(Path(base, "coord"), **dd)
    numbers, positions = _struct.numbers, _struct.positions
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    positions = positions.clone().requires_grad_(True)

    # do calc
    options = dict(
        opts,
        **{
            "strategy": _strategy(differentiation),
            "x_atol": 1e-5 if dtype == torch.float else 1e-10,
            "differentiation": differentiation,
        },
    )
    calc = Calculator(
        numbers, get_param_module("gfn1", **dd), opts=options, **dd
    )
    result = calc.singlepoint(positions, charge)
    energy = result.energy.sum(-1)

    # autograd
    energy.backward()
    assert positions.grad is not None
    autograd = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    # tblite reference grad
    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(ref.cpu(), abs=tol, rel=1e-4) == autograd.cpu()


@pytest.mark.grad
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "name", ["H2", "H2O", pytest.param("CH4", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("differentiation", ["implicit", "unrolled"])
def test_num(name: str, differentiation: str) -> None:
    dtype = torch.double
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # read from file
    base = Path(Path(__file__).parent, "mols", name)
    _struct = read(Path(base, "coord"), **dd)
    numbers, positions = _struct.numbers, _struct.positions
    charge = read_chrg(Path(base, ".CHRG"), **dd)

    # do calc
    gradient = num_grad(numbers, positions, charge, differentiation, dd)

    ref = load_from_npz(ref_grad, name, dtype)
    assert pytest.approx(ref.cpu(), abs=1e-6, rel=1e-4) == gradient.cpu()


def num_grad(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    differentiation: str,
    dd: DD,
) -> Tensor:
    """Calculate gradient numerically for reference."""

    options = dict(
        opts,
        **{
            "differentiation": differentiation,
            "strategy": _strategy(differentiation),
        },
    )
    calc = Calculator(
        numbers, get_param_module("gfn1", **dd), opts=options, **dd
    )

    gradient = torch.zeros_like(positions)
    step = 1.0e-6

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            result = calc.singlepoint(positions, charge)
            er = result.energy.sum(-1)

            positions[i, j] -= 2 * step
            result = calc.singlepoint(positions, charge)
            el = result.energy.sum(-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient
