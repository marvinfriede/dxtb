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

These tests are taken from https://github.com/awvwgk/tad-dftd3/tree/main/tests
and are only included for the sake of completeness.
"""

from __future__ import annotations

import pytest
import tad_dftd as d3
import torch
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.io.structure import Structure

from dxtb import GFN1_XTB
from dxtb._src.components.classicals.dispersion import (
    core_dispersion_d3_setup,
    dispersion_d3_energy,
)
from dxtb._src.components.classicals.dispersion.d3 import (
    _DXTB_D3_CN_CUTOFF,
    _DXTB_D3_DISPERSION_CUTOFF,
)
from dxtb._src.typing import DD, Tensor
from dxtb.components.dispersion import DispersionD3, new_dispersion

from ...conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_batch(dtype: torch.dtype) -> None:
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
    c6 = pack(
        (
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["edisp"].to(**dd),
            sample2["edisp"].to(**dd),
        )
    )

    rvdw = radii.VDW_PAIRWISE(**dd)[
        numbers.unsqueeze(-1), numbers.unsqueeze(-2)
    ]
    r4r2 = d3.data.R4R2(**dd)[numbers]
    param = {
        "a1": torch.tensor(0.49484001, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }

    energy = d3.disp.dispersion(
        numbers,
        positions,
        param,
        c6,
        rvdw,
        r4r2,
        d3.damping.rational_damping,
        cutoff=d3.Cutoff(
            disp2=_DXTB_D3_DISPERSION_CUTOFF,
            disp3=_DXTB_D3_DISPERSION_CUTOFF,
        ),
    )
    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()

    # create copy as `par` lives in global scope
    _par = GFN1_XTB.model_copy(deep=True)
    if _par.dispersion is None or _par.dispersion.d3 is None:
        assert False

    # set parameters explicitly
    _par.dispersion.d3.a1 = param["a1"].item()
    _par.dispersion.d3.a2 = param["a2"].item()
    _par.dispersion.d3.s8 = param["s8"].item()

    disp = new_dispersion(numbers, _par, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)
    edisp = disp.get_energy(positions, cache)
    assert edisp.dtype == dtype
    assert pytest.approx(edisp.cpu()) == ref.cpu()
    assert pytest.approx(edisp.cpu()) == energy.cpu()


def test_three_way_d3_parity_values_and_gradients() -> None:
    """
    Three-way comparison required by H1.1:
    1. dxtb integrated D3 (legacy cache and core setup).
    2. Native tad_dftd D3 with equivalent physical configuration and cutoffs.
    3. Independent reference energies and analytical gradients from tblite/s-dftd3.
    """
    dtype = torch.float64
    dd: DD = {"dtype": dtype, "device": DEVICE}
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach().clone()
    ref_energy = sample["edisp"].to(**dd)
    ref_grad = sample["grad"].to(**dd)

    disp = new_dispersion(numbers, GFN1_XTB, **dd)
    assert isinstance(disp, DispersionD3)
    cache = disp.get_cache(numbers)
    assert _DXTB_D3_CN_CUTOFF == 25.0
    assert _DXTB_D3_DISPERSION_CUTOFF == 50.0
    assert cache.cutoff.item() == 25.0
    core = core_dispersion_d3_setup(cache, numbers, disp.numbers, disp.param)

    pos_dxtb = positions.clone().requires_grad_(True)
    e_dxtb = disp.get_energy(pos_dxtb, cache)
    g_dxtb = torch.autograd.grad(e_dxtb.sum(), pos_dxtb)[0]

    pos_core = positions.clone().requires_grad_(True)
    e_core = dispersion_d3_energy(core, pos_core)
    g_core = torch.autograd.grad(e_core.sum(), pos_core)[0]

    pos_native = positions.clone().requires_grad_(True)
    mol = Structure(numbers=numbers, positions=pos_native)
    dxtb_cutoff = d3.Cutoff(
        cn=_DXTB_D3_CN_CUTOFF,
        disp2=_DXTB_D3_DISPERSION_CUTOFF,
        disp3=_DXTB_D3_DISPERSION_CUTOFF,
    )
    e_native = d3.dftd3(mol, disp.param, cutoff=dxtb_cutoff)
    g_native = torch.autograd.grad(e_native.sum(), pos_native)[0]

    # Roundoff-level parity between dxtb integrated D3 and native tad_dftd D3
    torch.testing.assert_close(e_dxtb, e_native, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(e_core, e_native, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(g_dxtb, g_native, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(g_core, g_native, rtol=1e-13, atol=1e-14)

    # Parity against independent tblite reference gradient for GFN1-xTB
    torch.testing.assert_close(g_dxtb, ref_grad, rtol=1e-7, atol=1e-10)
    torch.testing.assert_close(g_native, ref_grad, rtol=1e-7, atol=1e-10)

    # Parity against independent s-dftd3 reference energy (r2SCAN-D3(BJ))
    r2scan_par = GFN1_XTB.model_copy(deep=True)
    assert r2scan_par.dispersion is not None and r2scan_par.dispersion.d3 is not None
    r2scan_par.dispersion.d3.a1 = 0.49484001
    r2scan_par.dispersion.d3.s8 = 0.78981345
    r2scan_par.dispersion.d3.a2 = 5.73083694
    disp_r2scan = new_dispersion(numbers, r2scan_par, **dd)
    assert isinstance(disp_r2scan, DispersionD3)
    cache_r2scan = disp_r2scan.get_cache(numbers)
    core_r2scan = core_dispersion_d3_setup(
        cache_r2scan, numbers, disp_r2scan.numbers, disp_r2scan.param
    )
    e_r2scan_dxtb = disp_r2scan.get_energy(positions, cache_r2scan)
    e_r2scan_core = dispersion_d3_energy(core_r2scan, positions)
    e_r2scan_native = d3.dftd3(
        Structure(numbers=numbers, positions=positions),
        disp_r2scan.param,
        cutoff=dxtb_cutoff,
    )
    torch.testing.assert_close(
        e_r2scan_dxtb, e_r2scan_native, rtol=1e-14, atol=1e-15
    )
    torch.testing.assert_close(
        e_r2scan_core, e_r2scan_native, rtol=1e-14, atol=1e-15
    )
    torch.testing.assert_close(e_r2scan_dxtb, ref_energy, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(
        e_r2scan_native, ref_energy, rtol=1e-7, atol=1e-7
    )

    # Three-body ATM parity (nonzero s9 and trainable zero s9)
    for s9_val in (1.0, 0.0):
        s9_dxtb = torch.tensor(s9_val, **dd, requires_grad=True)
        s9_native = torch.tensor(s9_val, **dd, requires_grad=True)
        param_dxtb = dict(disp.param, s9=s9_dxtb)
        param_native = dict(disp.param, s9=s9_native)

        comp_atm = DispersionD3(numbers, param_dxtb, **dd)
        cache_atm = comp_atm.get_cache(numbers)
        core_atm = core_dispersion_d3_setup(
            cache_atm, numbers, comp_atm.numbers, comp_atm.param
        )

        p1 = positions.clone().requires_grad_(True)
        e1 = dispersion_d3_energy(core_atm, p1)
        g_p1, g_s1 = torch.autograd.grad(e1.sum(), (p1, s9_dxtb))

        p2 = positions.clone().requires_grad_(True)
        e2 = d3.dftd3(
            Structure(numbers=numbers, positions=p2),
            param_native,
            cutoff=dxtb_cutoff,
        )
        g_p2, g_s2 = torch.autograd.grad(e2.sum(), (p2, s9_native))

        torch.testing.assert_close(e1, e2, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(g_p1, g_p2, rtol=1e-13, atol=1e-14)
        torch.testing.assert_close(g_s1, g_s2, rtol=1e-14, atol=1e-15)
        assert torch.count_nonzero(g_s1) > 0


@pytest.mark.grad
def test_grad_pos() -> None:
    dtype = torch.double
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach().clone()

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    disp = new_dispersion(numbers, GFN1_XTB, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    def func(p: Tensor) -> Tensor:
        return disp.get_energy(p, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, pos)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
def test_grad_pos_tblite(dtype: torch.dtype) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd).detach().clone()
    ref = sample["grad"].to(**dd)

    disp = new_dispersion(numbers, GFN1_XTB, **dd)
    if disp is None:
        assert False

    cache = disp.get_cache(numbers)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    # automatic gradient
    energy = torch.sum(disp.get_energy(pos, cache), dim=-1)
    energy.backward()

    if pos.grad is None:
        assert False
    grad_backward = pos.grad.clone()

    assert pytest.approx(grad_backward.cpu(), abs=1e-10) == ref.cpu()


@pytest.mark.grad
def test_grad_param() -> None:
    dtype = torch.double
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples["C4H5NCS"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    param = (
        torch.tensor(1.00000000, requires_grad=True, dtype=dtype),
        torch.tensor(0.78981345, requires_grad=True, dtype=dtype),
        torch.tensor(0.49484001, requires_grad=True, dtype=dtype),
        torch.tensor(5.73083694, requires_grad=True, dtype=dtype),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs):
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        disp = DispersionD3(numbers, input_param, **dd)
        cache = disp.get_cache(numbers)
        return disp.get_energy(positions, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, param)


def test_d3_h2o_gfn1_float64_reference_parity() -> None:
    """
    Verify that DispersionD3 in float64 constructs full-precision float64
    reference tables and matches the Fortran dftd H2O GFN1-xTB D3(BJ) energy
    (-0.00013758459371839634 Eh) to machine precision (H3.1).
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, -0.74288549752983],
            [-1.43472674945442, 0.0, 0.37144274876492],
            [1.43472674945442, 0.0, 0.37144274876492],
        ],
        **dd,
    )
    disp = new_dispersion(numbers, GFN1_XTB, **dd)
    assert isinstance(disp, DispersionD3)
    cache = disp.get_cache(numbers)
    core = core_dispersion_d3_setup(cache, numbers, disp.numbers, disp.param)

    e_cache = disp.get_energy(positions, cache).sum()
    e_core = dispersion_d3_energy(core, positions).sum()
    e_fortran = torch.tensor(-0.00013758459371839634, **dd)

    torch.testing.assert_close(e_cache, e_fortran, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(e_core, e_fortran, rtol=0.0, atol=1e-15)
