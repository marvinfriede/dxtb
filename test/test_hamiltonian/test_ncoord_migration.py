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
from tad_mctc.ncoord import CNModel, cn_d3, cn_d4, cn_gfn2, exp_count
from tad_mctc.tree import stack
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

# Non-uniform per-atom displacements so interatomic distances and coordination
# numbers genuinely differ across batch items.
_DISP3 = torch.tensor(
    [[0.05, -0.02, 0.01], [-0.04, 0.06, -0.03], [0.02, -0.05, 0.04]],
    dtype=torch.float64,
)
_DISP2 = torch.tensor(
    [[0.06, -0.03, 0.02], [-0.05, 0.04, -0.02]],
    dtype=torch.float64,
)

# Fixed pre-migration references computed on commit 715b5c82 (legacy ncoord).
_REF_H0_NORM2 = {
    "gfn0": torch.tensor(
        [1.3761281461438764, 1.3763487534358037], dtype=torch.float64
    ),
    "gfn1": torch.tensor(
        [2.3769598897297306, 2.376945034878008], dtype=torch.float64
    ),
    "gfn2": torch.tensor(
        [1.831066274381938, 1.83110921505906], dtype=torch.float64
    ),
}
_REF_H0_GRAD_NORM = {
    "gfn0": 0.032490761785646784,
    "gfn1": 0.000745590699428378,
    "gfn2": 0.0021592104585757242,
}
_REF_IES_ENERGY = torch.tensor(
    [-0.0005678785066277336, -0.0005732900077461897], dtype=torch.float64
)
_REF_IES_GRAD_NORM = 0.0003378784542060113
_REF_SRB_ENERGY = torch.tensor(
    [-0.025539163380886784, -0.025006652161334155], dtype=torch.float64
)
_REF_SRB_GRAD_NORM = 0.015951193124680425
_REF_D4_ENERGY = torch.tensor(
    [-0.00028652371795898085, -0.00028814442621400023], dtype=torch.float64
)
_REF_D4_GRAD_NORM = 8.747674738819562e-05
_REF_AES2_MRAD_SUM = torch.tensor(
    [6.422876577202871, 6.423684392175904], dtype=torch.float64
)
_REF_AES2_GRAD_NORM = 0.043517886912258344
_REF_D4_H2O_ERF = (-0.00013714966072616224, 9.747138173817167e-07)
_REF_D4_H2O_EXP = (-0.00013714937476920898, 1.0767700288052765e-06)


def test_cn_model_integration_and_batched_geometry_differentiation() -> None:
    """All migrated CN callers use upstream CNModel and match pre-migration references."""
    numbers = samples["H2O"]["numbers"].to(device="cpu")
    positions = samples["H2O"]["positions"].to(
        device="cpu", dtype=torch.float64
    )

    for method, expected_cn, expected_kind, cls in (
        ("gfn0", None, "erf_count", GFN0Hamiltonian),
        ("gfn1", cn_d3, "cn_d3", GFN1Hamiltonian),
        ("gfn2", cn_gfn2, "cn_gfn2", GFN2Hamiltonian),
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
        assert core.cn_kind == expected_kind
        ham = cls(numbers, par, ihelp, dtype=positions.dtype)
        assert isinstance(ham.cn, CNModel)
        assert isinstance(ham.cn.cutoff, float)
        if method == "gfn0":
            assert isinstance(legacy.cn_cutoff, float)
            assert isinstance(core.cn_cutoff, float)
            assert isinstance(ham.cn_cutoff, float)
            cn_val = ham.get_coordination_number(positions)
            cn_ref = ham.cn(Structure(numbers=numbers, positions=positions))
            torch.testing.assert_close(cn_val, cn_ref, rtol=1e-12, atol=1e-13)
            with pytest.raises(ValueError, match="different structure"):
                stack((core, core.replace(cn_cutoff=core.cn_cutoff + 1.0)))
        else:
            assert legacy.cn_cutoff is None
            assert core.cn_cutoff is None
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
            (positions, positions + _DISP3), dim=0
        ).requires_grad_(True)
        cn_batch = ham.cn(
            Structure(
                numbers=numbers.unsqueeze(0).expand(2, -1),
                positions=batched_pos,
            )
        )
        assert not torch.allclose(cn_batch[0], cn_batch[1])
        if method == "gfn0":
            cn_b = ham.get_coordination_number(batched_pos)
            torch.testing.assert_close(cn_b, cn_batch, rtol=1e-12, atol=1e-13)
            q_b = ham.get_eeq_charges(
                batched_pos, torch.zeros(2, dtype=positions.dtype), cn_b
            )
            assert q_b.shape == (2, numbers.shape[-1])
            assert not torch.allclose(q_b[0], q_b[1])

        overlap = torch.eye(
            ihelp.nao, device=positions.device, dtype=positions.dtype
        )
        charge_scalar = 0.0 if method == "gfn0" else None
        h_batch = vmap(
            lambda p: build_hcore(
                core, p, overlap, charge=charge_scalar, ihelp=ihelp
            )[0]
        )(batched_pos)
        h0, _ = build_hcore(
            core, batched_pos[0], overlap, charge=charge_scalar, ihelp=ihelp
        )
        h1, _ = build_hcore(
            core, batched_pos[1], overlap, charge=charge_scalar, ihelp=ihelp
        )
        assert not torch.allclose(h0, h1)
        torch.testing.assert_close(
            h_batch, torch.stack((h0, h1), dim=0), rtol=1e-12, atol=1e-13
        )
        h_norm2 = h_batch.square().sum(dim=(-2, -1))
        torch.testing.assert_close(
            h_norm2, _REF_H0_NORM2[method], rtol=1e-12, atol=1e-13
        )
        (grad_h,) = torch.autograd.grad(
            h_norm2.sum(), batched_pos, create_graph=True
        )
        torch.testing.assert_close(
            grad_h.norm(),
            torch.tensor(_REF_H0_GRAD_NORM[method], dtype=torch.float64),
            rtol=1e-11,
            atol=1e-12,
        )
        (hess_h,) = torch.autograd.grad(grad_h.sum(), batched_pos)
        assert torch.isfinite(hess_h).all()

        # Stacked setup vmap and packed batch parity
        stacked_core = stack((core, core))
        h_stacked = vmap(
            lambda s, p: build_hcore(
                s, p, overlap, charge=charge_scalar, ihelp=ihelp
            )[0]
        )(stacked_core, batched_pos)
        torch.testing.assert_close(h_stacked, h_batch, rtol=1e-12, atol=1e-13)

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
        torch.testing.assert_close(h_packed, h_batch, rtol=1e-12, atol=1e-13)

    # Check IES and SRB (GFN0) on NO2 with non-uniform batched displacement
    no2_num = samples["NO2"]["numbers"].to(device="cpu")
    no2_pos = samples["NO2"]["positions"].to(
        device="cpu", dtype=torch.float64
    )
    batched_no2 = torch.stack((no2_pos, no2_pos + _DISP3)).requires_grad_(True)

    gfn0_sys = Model(
        par=ParamModule(GFN0_XTB.model_copy(deep=True), dtype=torch.float64),
        config=Config.create(exclude="scf"),
    ).setup(no2_num)
    assert gfn0_sys.ies_setup is not None and gfn0_sys.srb_setup is not None
    e_ies = ies_energy(
        gfn0_sys.ies_setup,
        batched_no2,
        torch.zeros(2, dtype=torch.float64),
    ).sum(-1)
    e_srb = short_range_bond_energy(gfn0_sys.srb_setup, batched_no2).sum(-1)
    assert not torch.allclose(e_ies[0], e_ies[1])
    assert not torch.allclose(e_srb[0], e_srb[1])
    torch.testing.assert_close(e_ies, _REF_IES_ENERGY, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(e_srb, _REF_SRB_ENERGY, rtol=1e-12, atol=1e-13)
    (g_ies,) = torch.autograd.grad(e_ies.sum(), batched_no2, retain_graph=True)
    (g_srb,) = torch.autograd.grad(e_srb.sum(), batched_no2)
    torch.testing.assert_close(
        g_ies.norm(),
        torch.tensor(_REF_IES_GRAD_NORM, dtype=torch.float64),
        rtol=1e-11,
        atol=1e-12,
    )
    torch.testing.assert_close(
        g_srb.norm(),
        torch.tensor(_REF_SRB_GRAD_NORM, dtype=torch.float64),
        rtol=1e-11,
        atol=1e-12,
    )

    # Stacked setup vmap for IES and SRB
    assert isinstance(gfn0_sys.ies_setup.cutoff, float)
    assert isinstance(gfn0_sys.srb_setup.cn_cutoff, float)
    stacked_ies = stack((gfn0_sys.ies_setup, gfn0_sys.ies_setup))
    stacked_srb = stack((gfn0_sys.srb_setup, gfn0_sys.srb_setup))
    e_ies_vmap = vmap(ies_energy)(
        stacked_ies, batched_no2, torch.zeros(2, dtype=torch.float64)
    ).sum(-1)
    e_srb_vmap = vmap(short_range_bond_energy)(stacked_srb, batched_no2).sum(-1)
    torch.testing.assert_close(e_ies_vmap, e_ies, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(e_srb_vmap, e_srb, rtol=1e-12, atol=1e-13)

    with pytest.raises(ValueError, match="different structure"):
        stack(
            (
                gfn0_sys.ies_setup,
                gfn0_sys.ies_setup.replace(
                    cutoff=gfn0_sys.ies_setup.cutoff + 1.0
                ),
            )
        )
    with pytest.raises(ValueError, match="different structure"):
        stack(
            (
                gfn0_sys.srb_setup,
                gfn0_sys.srb_setup.replace(
                    cn_cutoff=gfn0_sys.srb_setup.cn_cutoff + 1.0
                ),
            )
        )

    # Check AES2 and D4 (GFN2) with non-uniform batched displacement
    lih_num = samples["LiH"]["numbers"].to(device="cpu")
    lih_pos = samples["LiH"]["positions"].to(
        device="cpu", dtype=torch.float64
    )
    batched_lih = torch.stack((lih_pos, lih_pos + _DISP2)).requires_grad_(True)

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
    )(batched_lih, torch.zeros(2, dtype=torch.float64)).sum(-1)
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
    ).sum(-1)
    assert not torch.allclose(e_d4_packed[0], e_d4_packed[1])
    torch.testing.assert_close(
        e_d4_packed, e_d4_vmap, rtol=1e-12, atol=1e-13
    )
    torch.testing.assert_close(
        e_d4_packed, _REF_D4_ENERGY, rtol=1e-12, atol=1e-13
    )
    aes2_data = build_aes2_data(gfn2_sys.aes2_setup, batched_lih)
    mrad_sum = aes2_data.mrad.sum(-1)
    assert not torch.allclose(mrad_sum[0], mrad_sum[1])
    torch.testing.assert_close(
        mrad_sum, _REF_AES2_MRAD_SUM, rtol=1e-12, atol=1e-13
    )
    (g_d4,) = torch.autograd.grad(
        e_d4_packed.sum(), batched_lih, retain_graph=True
    )
    (g_aes2,) = torch.autograd.grad(mrad_sum.sum(), batched_lih)
    torch.testing.assert_close(
        g_d4.norm(),
        torch.tensor(_REF_D4_GRAD_NORM, dtype=torch.float64),
        rtol=1e-11,
        atol=1e-12,
    )
    torch.testing.assert_close(
        g_aes2.norm(),
        torch.tensor(_REF_AES2_GRAD_NORM, dtype=torch.float64),
        rtol=1e-11,
        atol=1e-12,
    )


def test_d4_custom_counting_function_changes_cn_and_matches_reference() -> None:
    """Custom D4 counting_function changes CN/energy and matches independent reference."""
    h2o_num = samples["H2O"]["numbers"].to(device="cpu")
    h2o_pos = (
        samples["H2O"]["positions"]
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .requires_grad_(True)
    )
    gfn2_par = GFN2_XTB.model_copy(deep=True)
    gfn2_par.dispersion.d4.sc = False  # type: ignore[union-attr]
    disp = new_dispersion(
        h2o_num,
        gfn2_par,
        charge=torch.tensor(0.0, dtype=torch.float64),
        dtype=torch.float64,
    )
    assert disp is not None
    setup = disp.get_cache(h2o_num)
    q = torch.tensor([-0.6, 0.3, 0.3], dtype=torch.float64)

    structure = Structure(numbers=h2o_num, positions=h2o_pos)
    cn_erf = cn_d4(structure)
    cn_exp = cn_d4.replace(count=exp_count)(structure)
    assert not torch.allclose(cn_erf, cn_exp, atol=1e-4)

    e_erf = d4.dftd4(h2o_num, h2o_pos, 0.0, setup.as_param(), q=q).sum()
    (g_erf,) = torch.autograd.grad(e_erf, h2o_pos, retain_graph=True)

    e_exp = d4.dftd4(
        h2o_num,
        h2o_pos,
        0.0,
        setup.as_param(),
        q=q,
        counting_function=exp_count,
    ).sum()
    (g_exp,) = torch.autograd.grad(e_exp, h2o_pos, retain_graph=True)

    # Independent reference via explicit cn_d4.replace(count=exp_count) callable
    e_exp_independent = d4.dftd4(
        h2o_num,
        h2o_pos,
        0.0,
        setup.as_param(),
        q=q,
        cn_function=lambda num, pos, **_: cn_d4.replace(count=exp_count)(
            Structure(numbers=num, positions=pos)
        ),
    ).sum()
    (g_exp_independent,) = torch.autograd.grad(e_exp_independent, h2o_pos)

    assert not torch.allclose(e_erf, e_exp, rtol=0.0, atol=1e-10)
    assert not torch.allclose(g_erf, g_exp, rtol=0.0, atol=1e-8)
    torch.testing.assert_close(
        e_exp, e_exp_independent, rtol=1e-14, atol=1e-15
    )
    torch.testing.assert_close(
        g_exp, g_exp_independent, rtol=1e-14, atol=1e-15
    )
    torch.testing.assert_close(
        e_erf,
        torch.tensor(_REF_D4_H2O_ERF[0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-14,
    )
    torch.testing.assert_close(
        g_erf.norm(),
        torch.tensor(_REF_D4_H2O_ERF[1], dtype=torch.float64),
        rtol=1e-11,
        atol=1e-14,
    )
    torch.testing.assert_close(
        e_exp,
        torch.tensor(_REF_D4_H2O_EXP[0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-14,
    )
    torch.testing.assert_close(
        g_exp.norm(),
        torch.tensor(_REF_D4_H2O_EXP[1], dtype=torch.float64),
        rtol=1e-11,
        atol=1e-14,
    )


def test_cn_model_callers_compile_fullgraph() -> None:
    """GFN0/1/2 H0, IES, and SRB CNModel constructions compile and compose with vmap."""
    from torch.func import grad, jacfwd, jacrev, jvp

    torch._dynamo.reset()
    numbers = samples["NO2"]["numbers"].to(device="cpu")
    positions = samples["NO2"]["positions"].to(
        device="cpu", dtype=torch.float64
    )
    batched_pos = torch.stack((positions, positions + _DISP3), dim=0)

    for method in ("gfn0", "gfn1", "gfn2"):
        par = ParamModule(
            _METHODS[method].model_copy(deep=True),
            device=positions.device,
            dtype=positions.dtype,
        )
        ihelp = IndexHelper.from_numbers(numbers, par)
        core = core_h0_setup(setup_h0(numbers, par, ihelp))
        overlap = torch.eye(
            ihelp.nao, device=positions.device, dtype=positions.dtype
        )
        chrg = 0.0 if method == "gfn0" else None
        eager_h, eager_occ = build_hcore(
            core, positions, overlap, charge=chrg, ihelp=ihelp
        )
        compiled_h0 = torch.compile(
            lambda s, p, o: build_hcore(s, p, o, charge=chrg, ihelp=ihelp),
            fullgraph=True,
        )
        comp_h, comp_occ = compiled_h0(core, positions, overlap)
        torch.testing.assert_close(comp_h, eager_h, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(comp_occ, eager_occ, rtol=1e-12, atol=1e-13)

        stacked_core = stack((core, core))
        eager_vmap_h = vmap(
            lambda s, p: build_hcore(
                s, p, overlap, charge=chrg, ihelp=ihelp
            )[0]
        )(stacked_core, batched_pos)
        compiled_vmap_h0 = torch.compile(
            vmap(
                lambda s, p: build_hcore(
                    s, p, overlap, charge=chrg, ihelp=ihelp
                )[0]
            ),
            fullgraph=True,
        )
        comp_vmap_h = compiled_vmap_h0(stacked_core, batched_pos)
        torch.testing.assert_close(
            comp_vmap_h, eager_vmap_h, rtol=1e-12, atol=1e-13
        )

    gfn0_model = Model(
        par=ParamModule(GFN0_XTB.model_copy(deep=True), dtype=torch.float64),
        config=Config.create(exclude="scf"),
    )
    gfn0_sys = gfn0_model.setup(numbers)
    assert gfn0_sys.ies_setup is not None and gfn0_sys.srb_setup is not None

    charge_zero = torch.tensor(0.0, dtype=torch.float64)
    eager_ies = ies_energy(gfn0_sys.ies_setup, positions, charge_zero)
    comp_ies = torch.compile(ies_energy, fullgraph=True)(
        gfn0_sys.ies_setup, positions, charge_zero
    )
    torch.testing.assert_close(comp_ies, eager_ies, rtol=1e-12, atol=1e-13)

    stacked_ies = stack((gfn0_sys.ies_setup, gfn0_sys.ies_setup))
    charges_batch = torch.zeros(2, dtype=torch.float64)
    eager_vmap_ies = vmap(ies_energy)(stacked_ies, batched_pos, charges_batch)
    comp_vmap_ies = torch.compile(vmap(ies_energy), fullgraph=True)(
        stacked_ies, batched_pos, charges_batch
    )
    torch.testing.assert_close(
        comp_vmap_ies, eager_vmap_ies, rtol=1e-12, atol=1e-13
    )

    eager_srb = short_range_bond_energy(gfn0_sys.srb_setup, positions)
    comp_srb = torch.compile(short_range_bond_energy, fullgraph=True)(
        gfn0_sys.srb_setup, positions
    )
    torch.testing.assert_close(comp_srb, eager_srb, rtol=1e-12, atol=1e-13)

    stacked_srb = stack((gfn0_sys.srb_setup, gfn0_sys.srb_setup))
    eager_vmap_srb = vmap(short_range_bond_energy)(stacked_srb, batched_pos)
    comp_vmap_srb = torch.compile(
        vmap(short_range_bond_energy), fullgraph=True
    )(stacked_srb, batched_pos)
    torch.testing.assert_close(
        comp_vmap_srb, eager_vmap_srb, rtol=1e-12, atol=1e-13
    )

    # Setup creation under differentiation transforms (grad, jvp, jacfwd, jacrev)
    base_tree = gfn0_model.parameter_tree
    srb_key = "parameter_tree.short_range.srb.prefactor.param"
    base_prefactor = base_tree.parameters.params[srb_key]

    def functional_gfn0_classical(prefactor_val: torch.Tensor) -> torch.Tensor:
        updated = dict(base_tree.parameters.params)
        updated[srb_key] = prefactor_val
        tree = base_tree.replace(
            parameters=base_tree.parameters.replace(params=updated)
        )
        sys = gfn0_model.setup_functional(numbers, tree)
        assert sys.h0_setup is not None
        assert sys.ies_setup is not None and sys.srb_setup is not None
        assert isinstance(sys.h0_setup.cn_cutoff, float)
        assert isinstance(sys.ies_setup.cutoff, float)
        assert isinstance(sys.srb_setup.cn_cutoff, float)
        return short_range_bond_energy(sys.srb_setup, positions).sum()

    g_grad = grad(functional_gfn0_classical)(base_prefactor)
    _, g_jvp = jvp(
        functional_gfn0_classical,
        (base_prefactor,),
        (torch.ones_like(base_prefactor),),
    )
    g_jacfwd = jacfwd(functional_gfn0_classical)(base_prefactor)
    g_jacrev = jacrev(functional_gfn0_classical)(base_prefactor)
    assert torch.isfinite(g_grad) and g_grad.abs() > 0
    torch.testing.assert_close(g_grad, g_jvp, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(g_grad, g_jacfwd, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(g_grad, g_jacrev, rtol=1e-12, atol=1e-13)
