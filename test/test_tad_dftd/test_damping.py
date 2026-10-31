# This file is part of tad-dftd.
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
Tests for two-body and three-body damping functions and kernels in
``tad_dftd.damping``, ``tad_dftd.twobody``, and ``tad_dftd.threebody``.
"""
from __future__ import annotations

import pytest
import tad_mctc as mctc
import torch

from tad_dftd import Cutoff, data, pairs, threebody, twobody
from tad_dftd.damping import (
    CSOTwoBody,
    DampingParam,
    FermiTwoBody,
    KoideThreeBody,
    KoideTwoBody,
    ModifiedZeroTwoBody,
    OptimizedPowerTwoBody,
    PairData,
    RationalThreeBody,
    RationalTwoBody,
    ScreenedThreeBody,
    ScreenedTwoBody,
    TripleData,
    ZeroProductThreeBody,
    ZeroThreeBodyD3,
    ZeroThreeBodyD4,
    ZeroTwoBody,
    ZTwoBody,
    damping_from_name,
)


def test_damping_from_name() -> None:
    """Verify damping lookup by name for all two-body and three-body variants."""
    assert isinstance(damping_from_name("rational").two, RationalTwoBody)
    assert isinstance(damping_from_name("bj").two, RationalTwoBody)
    assert isinstance(damping_from_name("zero").two, ZeroTwoBody)
    assert isinstance(damping_from_name("mzero").two, ModifiedZeroTwoBody)
    assert isinstance(damping_from_name("op").two, OptimizedPowerTwoBody)
    assert isinstance(damping_from_name("cso").two, CSOTwoBody)
    assert isinstance(damping_from_name("fermi").two, FermiTwoBody)
    assert isinstance(damping_from_name("z").two, ZTwoBody)
    assert isinstance(damping_from_name("koide").two, KoideTwoBody)
    assert isinstance(damping_from_name("screened").two, ScreenedTwoBody)

    # Three-body variants
    assert isinstance(damping_from_name("rational", three_body=True).three, ZeroThreeBodyD3)
    assert isinstance(
        damping_from_name("rational", three_body="rational").three,
        RationalThreeBody,
    )
    assert isinstance(
        damping_from_name("screened", three_body="screened").three,
        ScreenedThreeBody,
    )

    with pytest.raises(ValueError, match="Unknown"):
        damping_from_name("nonexistent")


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        (RationalTwoBody(), DampingParam(s6=1.0, s8=0.5, a1=0.4, a2=4.0)),
        (ZeroTwoBody(), DampingParam(s6=1.0, s8=0.5, rs6=1.2, rs8=1.0, alp=14.0)),
        (ModifiedZeroTwoBody(), DampingParam(s6=1.0, s8=0.5, rs6=1.2, rs8=1.0, bet=0.1, alp=14.0)),
        (OptimizedPowerTwoBody(), DampingParam(s6=1.0, s8=0.5, a1=0.4, a2=4.0, bet=4.0)),
        (CSOTwoBody(), DampingParam(s6=1.0, s8=0.0, a1=0.8, a2=2.5, a3=0.5, a4=4.0)),
        (FermiTwoBody(), DampingParam(s6=1.0, s8=0.0, rs6=1.0, a1=1.0, a2=0.0, alp=23.0)),
        (ZTwoBody(), DampingParam(s6=1.0, s8=0.5, a1=1.2, a2=0.5, a3=0.8, a4=0.4, bet=1.2, rs9=0.5)),
        (KoideTwoBody(), DampingParam(s6=1.0, s8=0.5, rs6=1.0, rs8=1.0, a1=1.0, a2=0.0, a3=0.6, a4=1.4)),
        (ScreenedTwoBody(), DampingParam(s6=1.0, s8=0.5, rs6=1.0, rs8=1.0, a1=1.0, a2=0.0, a3=0.5, a4=2.0)),
    ],
)
def test_twobody_damping_gradcheck(cls, params: DampingParam) -> None:
    """Verify analytical autograd first and second derivatives for every 2B damping class."""
    r = torch.tensor([3.2, 4.5, 6.0], dtype=torch.float64, requires_grad=True)
    qq = torch.tensor([25.0, 30.25, 36.0], dtype=torch.float64)
    c6 = torch.tensor([15.0, 20.0, 25.0], dtype=torch.float64)
    rvdw = torch.tensor([3.5, 4.0, 4.5], dtype=torch.float64)
    znum = torch.tensor([12.0, 16.0, 28.0], dtype=torch.float64)
    rdamp = torch.sqrt(qq)

    def func(dist: torch.Tensor) -> torch.Tensor:
        pdata = PairData(
            distances=dist, qq=qq, c6=c6, rvdw=rvdw, znum=znum, rdamp=rdamp
        )
        return cls(pdata, params)

    assert torch.autograd.gradcheck(func, (r,), eps=1e-6, atol=1e-8, rtol=1e-6)
    assert torch.autograd.gradgradcheck(func, (r,), eps=1e-5, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        (RationalThreeBody(), DampingParam(s9=1.0, a1=0.4, a2=4.0)),
        (ZeroThreeBodyD3(), DampingParam(s9=1.0, rs9=4.0 / 3.0, alp3=16.0)),
        (ZeroThreeBodyD4(), DampingParam(s9=1.0, rs9=1.0, alp3=16.0)),
        (ZeroProductThreeBody(), DampingParam(s9=1.0, a1=1.0, a2=0.0, rs9=1.0, alp3=16.0)),
        (KoideThreeBody(), DampingParam(s9=1.0, a1=1.0, a2=0.0, a3=0.6, a4=1.4, rs9=1.0)),
        (ScreenedThreeBody(), DampingParam(s9=1.0, a1=1.0, a2=0.0, a3=0.5, a4=2.0, rs9=1.0)),
    ],
)
def test_threebody_damping_gradcheck(cls, params: DampingParam) -> None:
    """Verify analytical autograd first and second derivatives for every 3B damping class."""
    rij = torch.tensor([3.2, 4.0], dtype=torch.float64, requires_grad=True)
    rik = torch.tensor([3.5, 4.2], dtype=torch.float64, requires_grad=True)
    rjk = torch.tensor([3.8, 4.5], dtype=torch.float64, requires_grad=True)

    rvdw = torch.tensor([3.6, 4.0], dtype=torch.float64)
    rdamp = torch.tensor([4.5, 5.0], dtype=torch.float64)

    def func(d_ij: torch.Tensor, d_ik: torch.Tensor, d_jk: torch.Tensor) -> torch.Tensor:
        r2ij = d_ij * d_ij
        r2ik = d_ik * d_ik
        r2jk = d_jk * d_jk
        r1 = torch.sqrt(r2ij * r2ik * r2jk)
        tdata = TripleData(
            r=r1,
            r2ij=r2ij,
            r2ik=r2ik,
            r2jk=r2jk,
            rvdwij=rvdw,
            rvdwik=rvdw,
            rvdwjk=rvdw,
            rdampij=rdamp,
            rdampik=rdamp,
            rdampjk=rdamp,
        )
        return cls(tdata, params)

    assert torch.autograd.gradcheck(func, (rij, rik, rjk), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_twobody_and_threebody_dense_vs_sparse_consistency() -> None:
    """Verify dense and sparse two-body and three-body kernels agree to machine precision."""
    numbers = torch.tensor([6, 8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.00000000, 2.30000000],
            [1.77000000, 0.00000000, -1.10000000],
            [-1.77000000, 0.00000000, -1.10000000],
        ],
        dtype=torch.float64,
    )
    mol = mctc.Structure(numbers=numbers, positions=positions)
    cutoff = Cutoff(cn=30.0, disp2=40.0, disp3=25.0)
    dp = pairs.DensePairs.build(mol, cutoff)
    sp = pairs.SparsePairs.build(mol, cutoff)

    # Synthetic symmetric C6 matrix
    c6_elem = torch.tensor([0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 22.0, 0.0, 15.0], dtype=torch.float64)
    c6 = torch.sqrt(c6_elem[numbers].unsqueeze(-1) * c6_elem[numbers].unsqueeze(-2))
    r4r2 = data.R4R2(dtype=torch.float64)
    rvdw = data.VDW_PAIRWISE(dtype=torch.float64)

    p = DampingParam(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0)

    e2_dense = twobody._dispersion2_dense(
        mol,
        p,
        c6,
        dp.disp2,
        damping=RationalTwoBody(),
        rvdw_table=rvdw,
        r4r2_table=r4r2,
        cutoff=cutoff.disp2,
        width=cutoff.width2,
    )
    e2_sparse = twobody.dispersion2_sparse(
        mol,
        p,
        c6,
        sp.disp2,
        damping=RationalTwoBody(),
        rvdw_table=rvdw,
        r4r2_table=r4r2,
        cutoff=cutoff.disp2,
        width=cutoff.width2,
    )
    assert torch.allclose(e2_dense, e2_sparse, atol=1e-13, rtol=1e-13)

    e3_dense = threebody.dispersion_atm(
        mol,
        c6,
        p,
        damping=ZeroThreeBodyD3(),
        rvdw_table=rvdw,
        r4r2_table=r4r2,
        cutoff=cutoff.disp3,
        width=cutoff.width3,
    )
    assert sp.disp3 is not None
    e3_sparse = threebody.dispersion_atm_sparse(
        mol,
        c6,
        p,
        sp.disp3,
        damping=ZeroThreeBodyD3(),
        rvdw_table=rvdw,
        r4r2_table=r4r2,
        cutoff=cutoff.disp3,
        width=cutoff.width3,
    )
    assert torch.allclose(e3_dense, e3_sparse, atol=1e-13, rtol=1e-13)


def test_zero_threebody_d3_alp_vs_alp3() -> None:
    """Verify ZeroThreeBodyD3 uses alp3 directly when set, or alp + 2 when unset."""
    r2 = torch.tensor([9.0, 16.0], dtype=torch.float64)
    r = torch.sqrt(r2 * r2 * r2)
    rvdw = torch.tensor([3.5, 4.0], dtype=torch.float64)
    tdata = TripleData(
        r=r,
        r2ij=r2,
        r2ik=r2,
        r2jk=r2,
        rdampij=rvdw,
        rdampik=rvdw,
        rdampjk=rvdw,
        rvdwij=rvdw,
        rvdwik=rvdw,
        rvdwjk=rvdw,
    )
    damp = ZeroThreeBodyD3()
    out_alp = damp(tdata, DampingParam(s9=1.0, rs9=4.0 / 3.0, alp=14.0))
    out_alp3 = damp(tdata, DampingParam(s9=1.0, rs9=4.0 / 3.0, alp3=16.0))
    assert torch.allclose(out_alp, out_alp3, atol=1e-15, rtol=1e-15)


def test_zero_and_mzero_radius_scaling_and_damp() -> None:
    """Verify ZeroTwoBody and ModifiedZeroTwoBody apply a1*r0+a2 in __call__ and damp()."""
    r2 = torch.tensor([0.0, 12.25, 20.25], dtype=torch.float64, requires_grad=True)
    rdamp = torch.tensor([0.0, 3.6, 4.2], dtype=torch.float64)
    p_zero = DampingParam(s6=1.0, rs6=1.2, rs8=1.0, alp=14.0, a1=0.9, a2=0.3)
    d6, d8 = ZeroTwoBody().damp(p_zero, r2, rdamp)
    assert torch.all(d8 == 0.0)
    assert d6[0].item() == 0.0
    (grad,) = torch.autograd.grad(d6.sum(), r2)
    assert torch.isfinite(grad).all()

    # Check consistency between __call__ and damp() when s8 is set
    p_zero_s8 = p_zero.replace(s8=0.5)
    d6_s8, d8_s8 = ZeroTwoBody().damp(p_zero_s8, r2[1:], rdamp[1:])
    r_pos = torch.sqrt(r2[1:])
    pdata = PairData(
        distances=r_pos,
        qq=torch.ones_like(r_pos),
        c6=torch.ones_like(r_pos),
        rvdw=rdamp[1:],
        znum=torch.ones_like(r_pos),
        rdamp=rdamp[1:],
    )
    total = ZeroTwoBody()(pdata, p_zero_s8)
    assert torch.allclose(d6_s8 + d8_s8, total, atol=1e-15, rtol=1e-15)


def test_cso_both_conventions() -> None:
    """Verify CSOTwoBody supports both s-dftd3 (a1, a2, rs6, rs8) and dftd (a3, a4, a2) conventions."""
    r = torch.tensor([3.5, 5.0], dtype=torch.float64)
    rdamp = torch.tensor([4.0, 4.5], dtype=torch.float64)
    pdata = PairData(
        distances=r,
        qq=rdamp * rdamp,
        c6=torch.ones_like(r),
        rvdw=rdamp,
        znum=torch.ones_like(r),
        rdamp=rdamp,
    )
    cso = CSOTwoBody()
    p_sdftd3 = DampingParam(s6=1.0, a1=0.86, a2=2.5, rs6=0.0, rs8=6.25)
    p_dftd = DampingParam(s6=1.0, a3=0.86, a4=2.5, a2=6.25)
    assert torch.allclose(cso(pdata, p_sdftd3), cso(pdata, p_dftd), atol=1e-15, rtol=1e-15)
