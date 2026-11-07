# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Regression tests for upstream tad_mctc.ncoord CNModel migration."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.ncoord import CNModel, cn_d3, cn_gfn2, exp_count
from torch.func import vmap

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals.dispersion import (
    _d4_adapter as d4,
    new_dispersion,
)
from dxtb._src.components.classicals.dispersion.d4 import dispersion_d4_energy
from dxtb._src.components.classicals.ies import ies_energy
from dxtb._src.components.classicals.shortrangebond import (
    short_range_bond_energy,
)
from dxtb._src.components.interactions.coulomb.multipole import build_aes2_data
from dxtb._src.xtb.gfn0 import GFN0Hamiltonian
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian
from dxtb._src.xtb.gfn2 import GFN2Hamiltonian
from dxtb._src.xtb.h0 import build_hcore, core_h0_setup, setup_h0
from dxtb.config import Config

from .samples import samples

_METHODS = {
    "gfn0": GFN0_XTB,
    "gfn1": GFN1_XTB,
    "gfn2": GFN2_XTB,
}


def _eval_hcore(core, positions, overlap, *, charge, ihelp):
    if hasattr(core, "index"):
        return build_hcore(core, positions, overlap, charge=charge)
    return build_hcore(core, positions, overlap, charge=charge, ihelp=ihelp)


def test_cn_model_integration_and_batched_geometry_differentiation() -> None:
    """All migrated CN callers use upstream CNModel and support batched geometries."""
    numbers = samples["H2O"]["numbers"].to(device="cpu")
    positions = samples["H2O"]["positions"].to(
        device="cpu", dtype=torch.float64
    )

    for method, expected_cn, cls in (
        ("gfn0", None, GFN0Hamiltonian),
        ("gfn1", cn_d3, GFN1Hamiltonian),
        ("gfn2", cn_gfn2, GFN2Hamiltonian),
    ):
        par = ParamModule(
            _METHODS[method].model_copy(deep=True),
            device=positions.device,
            dtype=positions.dtype,
        )
        ihelp = IndexHelper.from_numbers(numbers, par)
        legacy = setup_h0(numbers, par, ihelp)
        core = core_h0_setup(legacy)

        assert legacy.cn is expected_cn
        assert core.cn is expected_cn
        ham = cls(numbers, par, ihelp, dtype=positions.dtype)
        assert isinstance(ham.cn, CNModel)
        if method == "gfn0":
            cn_val = ham.get_coordination_number(positions)
            cn_ref = ham.cn(Structure(numbers=numbers, positions=positions))
            torch.testing.assert_close(cn_val, cn_ref, rtol=1e-12, atol=1e-13)
        else:
            equivalent = replace(
                legacy, cn=expected_cn.replace(cutoff=expected_cn.cutoff)
            )
            assert core_h0_setup(equivalent).cn_kind == core.cn_kind
            modified_pw = replace(
                legacy,
                cn=expected_cn.replace(
                    pair_weight=lambda d, n, c: torch.ones_like(
                        d, dtype=positions.dtype
                    )
                ),
            )
            with pytest.raises(
                TypeError, match="requires the built-in.*CN strategy"
            ):
                core_h0_setup(modified_pw)

        batched_pos = torch.stack(
            (positions, positions + 0.03), dim=0
        ).requires_grad_(True)
        if method == "gfn0":
            cn_b = ham.get_coordination_number(batched_pos)
            q_b = ham.get_eeq_charges(
                batched_pos, torch.zeros(2, dtype=positions.dtype), cn_b
            )
            assert q_b.shape == (2, numbers.shape[-1])
            assert torch.isfinite(q_b).all()

        overlap = torch.eye(
            ihelp.nao, device=positions.device, dtype=positions.dtype
        )
        charge_scalar = 0.0 if method == "gfn0" else None
        h_batch = vmap(
            lambda p: _eval_hcore(
                core, p, overlap, charge=charge_scalar, ihelp=ihelp
            )[0]
        )(batched_pos)
        h0, _ = _eval_hcore(
            core, batched_pos[0], overlap, charge=charge_scalar, ihelp=ihelp
        )
        h1, _ = _eval_hcore(
            core, batched_pos[1], overlap, charge=charge_scalar, ihelp=ihelp
        )
        torch.testing.assert_close(
            h_batch, torch.stack((h0, h1), dim=0), rtol=1e-12, atol=1e-13
        )
        (grad_h,) = torch.autograd.grad(
            h_batch.square().sum(), batched_pos, create_graph=True
        )
        (hess_h,) = torch.autograd.grad(grad_h.sum(), batched_pos)
        assert torch.isfinite(grad_h).all()
        assert torch.isfinite(hess_h).all()

        packed_numbers = numbers.unsqueeze(0).expand(2, -1)
        packed_ihelp = IndexHelper.from_numbers(packed_numbers, par)
        packed_setup = setup_h0(packed_numbers, par, packed_ihelp)
        packed_overlap = overlap.unsqueeze(0).expand(2, -1, -1)
        packed_charge = (
            torch.zeros(2, dtype=positions.dtype) if method == "gfn0" else None
        )
        h_packed, _ = build_hcore(
            packed_setup, batched_pos, packed_overlap, charge=packed_charge
        )
        torch.testing.assert_close(
            h_packed, h_batch, rtol=1e-12, atol=1e-13
        )

    # Check IES and SRB (GFN0) on NO2 (heteronuclear Z in [5, 9] so SRB is active)
    no2_num = samples["NO2"]["numbers"].to(device="cpu")
    no2_pos = samples["NO2"]["positions"].to(
        device="cpu", dtype=torch.float64
    )
    batched_no2 = torch.stack((no2_pos, no2_pos + 0.02)).requires_grad_(True)

    gfn0_sys = Model(
        par=ParamModule(GFN0_XTB.model_copy(deep=True), dtype=torch.float64),
        config=Config.create(exclude="scf"),
    ).setup(no2_num)
    assert gfn0_sys.ies_setup is not None and gfn0_sys.srb_setup is not None
    e_ies = ies_energy(
        gfn0_sys.ies_setup,
        batched_no2,
        torch.zeros(2, dtype=torch.float64),
    )
    e_srb = short_range_bond_energy(gfn0_sys.srb_setup, batched_no2)
    assert (e_srb.abs() > 1e-6).all()
    (g_srb,) = torch.autograd.grad(
        e_srb.sum(), batched_no2, retain_graph=True
    )
    assert (g_srb.abs() > 1e-6).any()
    (g_gfn0,) = torch.autograd.grad((e_ies + e_srb).sum(), batched_no2)
    assert torch.isfinite(g_gfn0).all()

    # Check AES2 and D4 (GFN2) with batched positions and custom counting_function
    lih_num = samples["LiH"]["numbers"].to(device="cpu")
    lih_pos = samples["LiH"]["positions"].to(
        device="cpu", dtype=torch.float64
    )
    batched_lih = torch.stack((lih_pos, lih_pos + 0.02)).requires_grad_(True)

    gfn2_par = GFN2_XTB.model_copy(deep=True)
    gfn2_par.dispersion.d4.sc = False  # type: ignore[union-attr]
    gfn2_pm = ParamModule(gfn2_par, dtype=torch.float64)
    gfn2_sys = Model(
        par=gfn2_pm,
        config=Config.create(exclude=("scf", "d4sc")),
    ).setup(lih_num)
    assert gfn2_sys.d4_setup is not None and gfn2_sys.aes2_setup is not None
    e_d4_vmap = vmap(
        lambda p, q: dispersion_d4_energy(gfn2_sys.d4_setup, p, q)
    )(batched_lih, torch.zeros(2, dtype=torch.float64))
    batched_lih_num = lih_num.unsqueeze(0).expand(2, -1)
    disp = new_dispersion(
        batched_lih_num,
        gfn2_par,
        charge=torch.zeros(2, dtype=torch.float64),
        dtype=torch.float64,
    )
    assert disp is not None
    d4_batched_setup = disp.get_cache(batched_lih_num)
    e_d4_packed = dispersion_d4_energy(
        d4_batched_setup,
        batched_lih,
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        e_d4_packed, e_d4_vmap, rtol=1e-12, atol=1e-13
    )
    aes2_data = build_aes2_data(gfn2_sys.aes2_setup, batched_lih)
    (g_gfn2,) = torch.autograd.grad(
        e_d4_packed.sum() + aes2_data.mrad.sum(), batched_lih
    )
    assert torch.isfinite(g_gfn2).all()

    # Verify custom counting_function on dftd4 works without AttributeError
    e_d4_exp = d4.dftd4(
        lih_num,
        lih_pos,
        torch.tensor(0.0, dtype=torch.float64),
        d4_batched_setup.as_param(),
        q=torch.zeros_like(lih_num, dtype=torch.float64),
        counting_function=exp_count,
    )
    assert torch.isfinite(e_d4_exp).all()
