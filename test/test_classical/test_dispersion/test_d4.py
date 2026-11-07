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
Testing dispersion energy and autodiff.

These tests are taken from https://github.com/dftd4/tad-dftd4/tree/main/test
and are only included for the sake of completeness.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.ncoord import cn_d4

from dxtb._src.components.classicals.dispersion import (
    DispersionD4,
    _d4_adapter as d4,
    new_dispersion,
)
from dxtb._src.param.gfn2 import GFN2_XTB
from dxtb._src.scf.guess import get_eeq_guess
from dxtb._src.typing import DD, Tensor

from ...conftest import DEVICE
from .samples import samples

par = GFN2_XTB.model_copy(deep=True)
par.dispersion.d4.sc = False  # type: ignore


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    charge = positions.new_zeros(numbers.shape[0])

    param = d4.Param(
        a1=torch.tensor(0.49484001, **dd),
        a2=torch.tensor(5.73083694, **dd),
        s8=torch.tensor(0.78981345, **dd),
        s9=torch.tensor(1.00000000, **dd),
    )

    # tad-dftd4 0.8 needs the legacy coordination number and EEQ charges
    # (see `dxtb._src.mctc_shim`)
    q = get_eeq_guess(numbers, positions, charge)
    energy = d4.dftd4(numbers, positions, charge, param, q=q, cn_function=cn_d4)
    assert energy.dtype == dtype

    # create copy as `par` lives in global scope
    _par = par.model_copy(deep=True)
    if _par.dispersion is None or _par.dispersion.d4 is None:
        assert False

    # set parameters explicitly
    _par.dispersion.d4.a1 = param["a1"].item()
    _par.dispersion.d4.a2 = param["a2"].item()
    _par.dispersion.d4.s8 = param["s8"].item()
    _par.dispersion.d4.s9 = param["s9"].item()

    disp = new_dispersion(numbers, _par, charge=charge, **dd)
    if disp is None:
        assert False

    # Add kwargs explicitly for coverage
    model = d4.model.D4Model(numbers, **dd)
    rcov = radii.COV_D3(**dd)[numbers]
    r4r2 = d4.data.R4R2(**dd)[numbers]
    cutoff = d4.cutoff.Cutoff(**dd)

    cache = disp.get_cache(
        numbers, model=model, rcov=rcov, r4r2=r4r2, cutoff=cutoff
    )

    edisp = disp.get_energy(positions, cache)
    assert edisp.dtype == dtype
    assert pytest.approx(edisp.cpu()) == energy.cpu()


@pytest.mark.grad
def test_grad_pos() -> None:
    dtype = torch.double
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach()
    charge = positions.new_tensor(0.0)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    disp = new_dispersion(numbers, par, charge, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    def func(p: Tensor) -> Tensor:
        return disp.get_energy(p, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, pos)


@pytest.mark.grad
def test_grad_param() -> None:
    dtype = torch.double
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = positions.new_tensor(0.0)
    param = (
        torch.tensor(1.00000000, requires_grad=True, **dd),
        torch.tensor(0.78981345, requires_grad=True, **dd),
        torch.tensor(0.49484001, requires_grad=True, **dd),
        torch.tensor(5.73083694, requires_grad=True, **dd),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs):
        input_param = d4.Param(
            **{label[i]: input for i, input in enumerate(inputs)}
        )
        disp = DispersionD4(numbers, input_param, charge, **dd)
        cache = disp.get_cache(numbers)
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, param)


def test_three_way_d4_parity_values_and_gradients() -> None:
    """dxtb DispersionD4, _d4_adapter.dftd4, and native tad_dftd.dftd4 agree to roundoff."""
    import tad_dftd
    from tad_mctc import Structure

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    pos_dxtb = positions.clone().requires_grad_(True)
    chrg_dxtb = torch.tensor(0.2, **dd, requires_grad=True)
    p_dxtb = {
        "s6": torch.tensor(1.0, **dd, requires_grad=True),
        "s8": torch.tensor(0.78981345, **dd, requires_grad=True),
        "a1": torch.tensor(0.49484001, **dd, requires_grad=True),
        "a2": torch.tensor(5.73083694, **dd, requires_grad=True),
        "s9": torch.tensor(1.0, **dd, requires_grad=True),
        "s10": torch.tensor(0.25, **dd, requires_grad=True),
        "alp": torch.tensor(16.0, **dd, requires_grad=True),
    }
    disp = DispersionD4(numbers, p_dxtb, charge=chrg_dxtb, **dd)
    cache = disp.get_cache(numbers)
    e_dxtb = disp.get_energy(pos_dxtb, cache, charge=chrg_dxtb)

    pos_native = positions.clone().requires_grad_(True)
    chrg_native = torch.tensor(0.2, **dd, requires_grad=True)
    p_native = {k: v.detach().clone().requires_grad_(True) for k, v in p_dxtb.items()}
    mol = Structure(numbers=numbers, positions=pos_native, charge=chrg_native)
    e_native = tad_dftd.dftd4(mol, p_native)

    torch.testing.assert_close(e_dxtb, e_native, atol=1e-13, rtol=1e-13)
    # Also compare against the independent reference stored in samples.py (for standard GFN2 D4 params)
    gfn2_param = {
        "s6": torch.tensor(1.0, **dd),
        "s8": torch.tensor(par.dispersion.d4.s8, **dd),
        "a1": torch.tensor(par.dispersion.d4.a1, **dd),
        "a2": torch.tensor(par.dispersion.d4.a2, **dd),
        "s9": torch.tensor(par.dispersion.d4.s9, **dd),
        "alp": torch.tensor(par.dispersion.d4.alp, **dd),
    }
    q_eeq = get_eeq_guess(numbers, positions, torch.tensor(0.0, **dd))
    e_gfn2_native = tad_dftd.dftd4(
        Structure(numbers=numbers, positions=positions),
        gfn2_param,
        q=q_eeq,
        qmodel="gfn2",
    )
    disp_gfn2 = DispersionD4(
        numbers, gfn2_param, charge=torch.tensor(0.0, **dd), ref_charges="gfn2", **dd
    )
    e_gfn2_dxtb = disp_gfn2.get_energy(positions, disp_gfn2.get_cache(numbers))
    torch.testing.assert_close(e_gfn2_dxtb, e_gfn2_native, atol=1e-13, rtol=1e-13)

    grads_dxtb = torch.autograd.grad(
        e_dxtb.sum(),
        (pos_dxtb, chrg_dxtb, *p_dxtb.values()),
    )
    grads_native = torch.autograd.grad(
        e_native.sum(),
        (pos_native, chrg_native, *p_native.values()),
    )
    for g_d, g_n in zip(grads_dxtb, grads_native):
        torch.testing.assert_close(g_d, g_n, atol=1e-12, rtol=1e-12)


def test_d4_trainable_zero_s9_and_weight_derivatives() -> None:
    """Trainable zero s9 retains ATM gradient and weight_references analytical derivatives match autograd."""
    import tad_dftd
    from tad_mctc import Structure

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    s9_zero = torch.tensor(0.0, **dd, requires_grad=True)
    param_zero = {
        "s6": torch.tensor(1.0, **dd),
        "s8": torch.tensor(0.8, **dd),
        "a1": torch.tensor(0.45, **dd),
        "a2": torch.tensor(4.5, **dd),
        "s9": s9_zero,
    }
    disp = DispersionD4(numbers, param_zero, charge=torch.tensor(0.0, **dd), **dd)
    e_zero = disp.get_energy(positions, disp.get_cache(numbers)).sum()
    (grad_s9_dxtb,) = torch.autograd.grad(e_zero, s9_zero)

    s9_zero_nat = torch.tensor(0.0, **dd, requires_grad=True)
    e_zero_nat = tad_dftd.dftd4(
        Structure(numbers=numbers, positions=positions),
        {**param_zero, "s9": s9_zero_nat},
    ).sum()
    (grad_s9_nat,) = torch.autograd.grad(e_zero_nat, s9_zero_nat)
    assert torch.count_nonzero(grad_s9_dxtb) > 0
    torch.testing.assert_close(grad_s9_dxtb, grad_s9_nat, atol=1e-13, rtol=1e-13)

    # Check analytical dgwdcn and dgwdq against autograd
    model = d4.model.D4Model(numbers, **dd)
    cn = torch.linspace(0.5, 3.5, numbers.numel(), **dd, requires_grad=True)
    q = torch.linspace(-0.3, 0.3, numbers.numel(), **dd, requires_grad=True)
    gw, dgwdcn, dgwdq = model.weight_references(
        cn, q, with_dgwdcn=True, with_dgwdq=True
    )
    v = torch.randn_like(gw)
    grad_cn, grad_q = torch.autograd.grad((gw * v).sum(), (cn, q))
    torch.testing.assert_close((dgwdcn * v).sum(dim=-1), grad_cn, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((dgwdq * v).sum(dim=-1), grad_q, atol=1e-12, rtol=1e-12)


def test_d4_adapter_edge_cases_rs10_doi_and_zero_damping() -> None:
    """Verify rs10/doi parameter handling and ZeroDamping with rs8 or zero radii."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples["LiH"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    r4r2 = d4.data.R4R2(**dd)[numbers]
    c6 = torch.ones((numbers.shape[-1], numbers.shape[-1]), **dd)
    rvdw = torch.ones_like(c6) * 2.5

    # 1. dispersion2 with rs10 and doi in param does not raise TypeError
    e_rs10 = d4.dispersion2(
        numbers,
        positions,
        {"a1": 0.4, "a2": 4.5, "s6": 1.0, "s8": 0.5, "s10": 0.2, "rs10": 1.0, "doi": "10.1063/1.5090222"},  # type: ignore[typeddict-unknown-key]
        c6,
        r4r2,
    )
    assert torch.isfinite(e_rs10).all()

    # 2. ZeroDamping with rs8 provided and s8 omitted includes order-8 contribution
    zd = d4.ZeroDamping()
    e_zd_rs8 = d4.dispersion2(
        numbers,
        positions,
        {"rs6": 1.0, "rs8": 1.2, "alp": 14.0},
        c6,
        r4r2,
        damping_function=zd,
        rvdw=rvdw,
    )
    e_zd_no_rs8 = d4.dispersion2(
        numbers,
        positions,
        {"rs6": 1.0, "alp": 14.0},
        c6,
        r4r2,
        damping_function=zd,
        rvdw=rvdw,
    )
    assert not torch.allclose(e_zd_rs8, e_zd_no_rs8)

    # 3. Direct call to ZeroDamping with zero entries in radii is finite
    dist = torch.tensor([[0.0, 2.5], [2.5, 0.0]], **dd, requires_grad=True)
    rad_zero = torch.tensor([[0.0, 2.0], [2.0, 0.0]], **dd)
    out = zd(dist, rad_zero, 6, rs6=1.0)
    assert torch.isfinite(out).all()
    (g_dist,) = torch.autograd.grad(out.sum(), dist)
    assert torch.isfinite(g_dist).all()

