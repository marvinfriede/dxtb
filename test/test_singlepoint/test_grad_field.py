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
Testing automatic energy gradient w.r.t. electric field vector.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.units import VAA2AU

from dxtb import Calculator
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Callable, Tensor

from ..conftest import DEVICE
from ..utils import get_param_module, scf_strategy
from .samples import samples

opts = {
    "int_level": labels.INTLEVEL_DIPOLE,
    "x_atol": 1.0e-8,
    "maxiter": 100,
    "strategy": scf_strategy("anderson", "potential"),
    "verbosity": 0,
}

tol = 1e-1

# FIXME: There seem to be multiple issues with this gradient here.
# - SiH4 fails for 0.0 (0.01 check depends on eps)
# - non-negligible differences between --fast and --slow
sample_list = ["H2", "H2O"]
xfields = [0.0, 1.0, pytest.param(-2.0, marks=pytest.mark.slow)]

# The first-order single-sample check covers all fields. Double gradients use
# the field-free and one finite-field case; batched variants (padding only)
# use a single finite field.
xfields_reduced = [0.0, 1.0]
xfields_batch = [1.0]
batch_name2_list = ["H2", pytest.param("H2O", marks=pytest.mark.slow)]


def _strategy(differentiation: str):
    """Anderson for unrolled, Broyden for implicit SCF differentiation."""
    converger = "anderson" if differentiation == "unrolled" else "broyden"
    return scf_strategy(converger, "potential")


def gradchecker(
    dtype: torch.dtype,
    name: str,
    xfield: float,
    gfn: str,
    differentiation: str,
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    options = dict(
        opts,
        **{
            "differentiation": differentiation,
            "strategy": _strategy(differentiation),
        },
    )

    par = get_param_module(gfn, **dd)

    # variables to be differentiated
    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * VAA2AU
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        calc = Calculator(numbers, par, opts=options, **dd)
        return calc.energy(positions, charge, field=field_vector)

    return func, field_vector


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
@pytest.mark.parametrize("differentiation", ["implicit", "unrolled"])
def test_gradcheck(
    dtype: torch.dtype,
    name: str,
    xfield: float,
    gfn: str,
    differentiation: str,
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield, gfn, differentiation)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("xfield", xfields_reduced)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_gradgradcheck(
    dtype: torch.dtype,
    name: str,
    xfield: float,
    gfn: str,
    differentiation: str,
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, xfield, gfn, differentiation)
    assert dgradgradcheck(func, diffvars, atol=tol)


################################################################################


def gradchecker_batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    xfield: float,
    gfn: str,
    differentiation: str,
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    par = get_param_module(gfn, **dd)

    options = dict(
        opts,
        **{
            "differentiation": differentiation,
            "strategy": _strategy(differentiation),
        },
    )

    # variables to be differentiated
    field_vector = torch.tensor([xfield, 0.0, 0.0], **dd) * VAA2AU
    field_vector.requires_grad_(True)

    def func(field_vector: Tensor) -> Tensor:
        calc = Calculator(numbers, par, opts=options, **dd)
        return calc.energy(positions, charge, field=field_vector)

    return func, field_vector


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", batch_name2_list)
@pytest.mark.parametrize("xfield", xfields_batch)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_gradcheck_batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    xfield: float,
    gfn: str,
    differentiation: str,
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(
        dtype, name1, name2, xfield, gfn, differentiation
    )
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O"])
@pytest.mark.parametrize("name2", batch_name2_list)
@pytest.mark.parametrize("xfield", xfields_batch)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
@pytest.mark.parametrize("differentiation", ["unrolled"])
def test_gradgradcheck_batch(
    dtype: torch.dtype,
    name1: str,
    name2: str,
    xfield: float,
    gfn: str,
    differentiation: str,
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(
        dtype, name1, name2, xfield, gfn, differentiation
    )
    assert dgradgradcheck(func, diffvars, atol=tol, eps=1e-8)
