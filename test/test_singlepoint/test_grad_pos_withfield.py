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
Run tests for the nuclear gradient in an electric field.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.units import VAA2AU

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.components.interactions import new_efield
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Callable, Tensor

from ..conftest import DEVICE
from .samples import samples

opts = {
    "int_level": labels.INTLEVEL_DIPOLE,
    "x_atol": 1.0e-10,
    "maxiter": 100,
    "damp": 0.3,
    "damp_init": 0.01,
    "damp_dynamic": False,
    "mixer": labels.MIXER_ANDERSON,
    "scf_mode": labels.SCF_MODE_FULL,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}

tol = 1e-4

sample_list = ["H2", "H2O", pytest.param("SiH4", marks=pytest.mark.slow)]


def gradchecker(dtype: torch.dtype, name: str, gfn: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        raise ValueError(f"Unknown GFN: {gfn}")

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * VAA2AU

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, opts=opts, interaction=efield, **dd)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        result = calc.singlepoint(p, charge)
        energy = result.total.sum(-1)
        return energy

    return func, pos


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_gradcheck(dtype: torch.dtype, name: str, gfn: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, gfn)
    assert dgradcheck(
        func, diffvars, atol=tol, nondet_tol=1e-7, fast_mode=False
    )


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H2O"])
@pytest.mark.parametrize(
    "gfn", ["gfn1", pytest.param("gfn2", marks=pytest.mark.slow)]
)
def test_gradgradcheck(dtype: torch.dtype, name: str, gfn: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, gfn)
    assert dgradgradcheck(
        func, diffvars, atol=tol, eps=1e-5, nondet_tol=1e-7, fast_mode=False
    )


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str, gfn: str
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

    if gfn == "gfn1":
        par = GFN1_XTB
    elif gfn == "gfn2":
        par = GFN2_XTB
    else:
        raise ValueError(f"Unknown GFN: {gfn}")

    field_vector = torch.tensor([-2.0, 0.0, 0.0], **dd) * VAA2AU

    # create additional interaction and pass to Calculator
    efield = new_efield(field_vector)
    calc = Calculator(numbers, par, interaction=[efield], opts=opts, **dd)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        result = calc.singlepoint(p, charge)
        energy = result.total.sum(-1)
        return energy

    return func, pos


# Batching only needs one pair with padding and one identical (cheap) pair
# without padding; the single-sample tests above cover the per-molecule
# physics. The batched second-order check is marked `large` (not run in the
# default tox tier).
batch_pairs = [
    ("H2O", "H2"),
    pytest.param("H2", "H2", marks=pytest.mark.slow),
]


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1, name2", batch_pairs)
@pytest.mark.parametrize(
    "gfn", ["gfn1", pytest.param("gfn2", marks=pytest.mark.slow)]
)
def test_gradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, gfn: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, gfn)
    assert dgradcheck(
        func, diffvars, atol=tol, nondet_tol=1e-7, fast_mode=False
    )


@pytest.mark.grad
@pytest.mark.large
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1, name2", batch_pairs)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str, gfn: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2, gfn)
    assert dgradgradcheck(
        func, diffvars, atol=tol, eps=1e-4, nondet_tol=1e-7, fast_mode=False
    )


def full_vs_implicit_autograd(
    numbers: Tensor, positions: Tensor, charge: Tensor, dd: DD
) -> None:
    field_vector = torch.tensor([-2.0, 1.0, 0.5], **dd) * VAA2AU

    options_imp = dict(opts, **{"scf_mode": labels.SCF_MODE_IMPLICIT})
    options_full = dict(opts, **{"scf_mode": labels.SCF_MODE_FULL})

    calc_imp = Calculator(
        numbers,
        GFN1_XTB,
        interaction=new_efield(field_vector),
        opts=options_imp,
        **dd,
    )
    pos = positions.clone().requires_grad_(True)
    ref = calc_imp.forces(pos, charge).detach()

    calc_full = Calculator(
        numbers,
        GFN1_XTB,
        interaction=new_efield(field_vector),
        opts=options_full,
        **dd,
    )
    pos = positions.clone().requires_grad_(True)
    forces = calc_full.forces(pos, charge).detach()

    assert pytest.approx(ref.cpu(), abs=1e-7, rel=1e-5) == forces.cpu()


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
def test_full_vs_implicit_forces_with_field(dtype: torch.dtype) -> None:
    """
    Check autograd forces in an electric field across full and implicit SCF
    modes for an anion with non-zero atomic charges and dipole moments.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = samples["NO2"]["numbers"].to(DEVICE)
    positions = samples["NO2"]["positions"].to(**dd)
    charge = torch.tensor(-1.0, **dd)

    full_vs_implicit_autograd(numbers, positions, charge, dd)


@pytest.mark.grad
@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.double])
def test_full_vs_implicit_forces_with_field_batch(dtype: torch.dtype) -> None:
    """
    Check autograd forces in an electric field across full and implicit SCF
    modes for a batch of a neutral molecule and an anion.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples["SiH4"], samples["NO2"]
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
    charge = torch.tensor([0.0, -1.0], **dd)

    full_vs_implicit_autograd(numbers, positions, charge, dd)
