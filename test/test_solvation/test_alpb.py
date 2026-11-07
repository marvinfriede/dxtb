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
Test the Analytical linearized Poisson-Boltzmann model.
"""

from __future__ import annotations

from math import sqrt

import pytest
import tblite.interface as tblite
import torch
from tad_mctc.data import VDW_D3
from tad_mctc.units import AA2AU

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, Param, ParamModule
from dxtb._src.components.interactions.solvation import alpb, born
from dxtb._src.constants import labels
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples

opts = {
    "x_atol": 1e-10,
    "scf_mode": labels.SCF_MODE_IMPLICIT,
    "scp_mode": labels.SCP_MODE_POTENTIAL,
    "verbosity": 0,
}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
def test_gb_single(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    dielectric_constant = torch.tensor(dielectric_constant, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charges = sample["charges"].to(**dd)
    ref = sample["energies"].to(**dd)

    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)
    cache = gb.get_cache(numbers=numbers, positions=positions)
    energies = gb.get_monopole_atom_energy(cache, charges)

    assert pytest.approx(energies.cpu(), abs=tol) == ref.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01", "MB16_43_02"])
def test_gb_still_single(
    dtype: torch.dtype, name: str, dielectric_constant=78.9
):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    dc = torch.tensor(dielectric_constant, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charges = sample["charges"].to(**dd)
    rvdw = VDW_D3(**dd)[numbers]
    ref = sample["energies_still"].to(**dd)

    gb = alpb.GeneralizedBorn(
        numbers, dc, kernel="still", rvdw=rvdw, alpb=False, **dd
    )
    cache = gb.get_cache(numbers=numbers, positions=positions)
    energies = gb.get_monopole_atom_energy(cache, charges)

    assert pytest.approx(energies.cpu(), abs=tol) == ref.cpu()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gb_scf(dtype: torch.dtype, name: str, dielectric_constant=78.9):
    tol = 1e-3 if dtype == torch.float else 1e-5
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["gsolv"]
    charges = torch.tensor(0.0).type(dtype)

    dielectric_constant = torch.tensor(dielectric_constant, **dd)
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant, **dd)

    calc_vac = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    calc_sol = Calculator(numbers, GFN1_XTB, interaction=[gb], opts=opts, **dd)

    results_vac = calc_vac.singlepoint(positions, charges)
    results_sol = calc_sol.singlepoint(positions, charges)

    gsolv = results_sol.scf - results_vac.scf

    assert pytest.approx(ref.cpu(), abs=tol) == gsolv.cpu()


def test_gb_kernel_defaults_and_overrides() -> None:
    """ALPB defaults to p16, GBSA defaults to still, and explicit kernels override."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    sample = samples["MB16_43_01"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charges = sample["charges"].to(**dd)
    dc = torch.tensor(78.9, **dd)

    gb_alpb_default = alpb.GeneralizedBorn(numbers, dc, alpb=True, **dd)
    gb_gbsa_default = alpb.GeneralizedBorn(numbers, dc, alpb=False, **dd)
    gb_alpb_still = alpb.GeneralizedBorn(
        numbers, dc, alpb=True, kernel="still", **dd
    )
    gb_gbsa_p16 = alpb.GeneralizedBorn(
        numbers, dc, alpb=False, kernel="p16", **dd
    )

    assert gb_alpb_default.kernel == "p16"
    assert gb_gbsa_default.kernel == "still"
    assert gb_alpb_still.kernel == "still"
    assert gb_gbsa_p16.kernel == "p16"

    e_alpb = gb_alpb_default.get_monopole_atom_energy(
        gb_alpb_default.get_cache(numbers=numbers, positions=positions), charges
    )
    e_gbsa = gb_gbsa_default.get_monopole_atom_energy(
        gb_gbsa_default.get_cache(numbers=numbers, positions=positions), charges
    )
    torch.testing.assert_close(
        e_alpb, sample["energies"].to(**dd), atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        e_gbsa, sample["energies_still"].to(**dd), atol=1e-7, rtol=1e-7
    )

    # Parameter-based construction with omitted or explicit kernel.
    for alpb_flag, explicit_kernel, expected_kernel in (
        (True, None, "p16"),
        (False, None, "still"),
        (True, "still", "still"),
        (False, "p16", "p16"),
    ):
        raw = GFN1_XTB.model_dump()
        alpb_cfg: dict[str, object] = {
            "alpb": alpb_flag,
            "born_scale": 1.0,
            "born_offset": 0.0,
        }
        if explicit_kernel is not None:
            alpb_cfg["kernel"] = explicit_kernel
        raw["solvation"] = {"alpb": alpb_cfg}
        param = Param.model_validate(raw)
        module = ParamModule(param, **dd)
        solv = alpb.new_solvation(numbers, module, 78.9, **dd)
        assert solv is not None
        assert solv.kernel == expected_kernel


@pytest.mark.tblite
@pytest.mark.parametrize(
    "alpb_flag", [True, False], ids=["alpb_p16", "gb_still"]
)
def test_gb_live_tblite(alpb_flag: bool) -> None:
    """Compare ALPB (p16) and GB (still) default kernels against live tblite 0.8.0."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    sample = samples["SiH4"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    eps = 78.9

    # tblite's gbe-solvation and gb-solvation use COSMO van der Waals radii
    # (H = 1.30 A, Si = 2.20 A in Bohr) without empirical CDS/shift terms.
    cosmo_r_aa = {1: 1.30, 14: 2.20}
    rvdw = torch.tensor([cosmo_r_aa[int(z)] * AA2AU for z in numbers], **dd)

    tb_kind = "gbe-solvation" if alpb_flag else "gb-solvation"
    tb_kernel = "p16" if alpb_flag else "still"

    tb_vac = tblite.Calculator(
        "GFN2-xTB", numbers.cpu().numpy(), positions.cpu().numpy()
    )
    tb_vac.set("accuracy", 1.0e-8)
    tb_vac.set("verbosity", 0)
    res_tb_vac = tb_vac.singlepoint()

    tb_sol = tblite.Calculator(
        "GFN2-xTB", numbers.cpu().numpy(), positions.cpu().numpy()
    )
    tb_sol.set("accuracy", 1.0e-8)
    tb_sol.set("verbosity", 0)
    tb_sol.add(tb_kind, eps, tb_kernel)
    res_tb_sol = tb_sol.singlepoint()

    gfn2_opts = {
        "int_driver": labels.INTDRIVER_PYTORCH,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": labels.SCP_MODE_POTENTIAL,
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-9,
        "verbosity": 0,
    }
    pos_vac = positions.detach().clone().requires_grad_(True)
    calc_vac = Calculator(numbers, GFN2_XTB, opts=gfn2_opts, **dd)
    res_dx_vac = calc_vac.singlepoint(pos_vac)
    e_dx_vac = res_dx_vac.total.sum(-1)

    # Omit kernel= so GeneralizedBorn resolves the default ("p16" or "still").
    gb = alpb.GeneralizedBorn(numbers, eps, alpb=alpb_flag, rvdw=rvdw, **dd)
    assert gb.kernel == tb_kernel
    pos_sol = positions.detach().clone().requires_grad_(True)
    calc_sol = Calculator(
        numbers, GFN2_XTB, interaction=[gb], opts=gfn2_opts, **dd
    )
    res_dx_sol = calc_sol.singlepoint(pos_sol)
    e_dx_sol = res_dx_sol.total.sum(-1)
    (grad_dx_sol,) = torch.autograd.grad(e_dx_sol, pos_sol)
    q_dx_sol = calc_sol.ihelp.reduce_orbital_to_atom(res_dx_sol.charges.mono)

    gsolv_tb = res_tb_sol.get("energy") - res_tb_vac.get("energy")
    gsolv_dx = float((e_dx_sol - e_dx_vac).item())

    assert abs(res_tb_sol.get("energy") - float(e_dx_sol.item())) < 1.0e-6
    assert abs(gsolv_tb - gsolv_dx) < 1.0e-9
    torch.testing.assert_close(
        q_dx_sol.detach().cpu(),
        torch.tensor(res_tb_sol.get("charges"), dtype=torch.double),
        atol=3.0e-8,
        rtol=0.0,
    )
    torch.testing.assert_close(
        grad_dx_sol.detach().cpu(),
        torch.tensor(res_tb_sol.get("gradient"), dtype=torch.double),
        atol=1.0e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("alpb_flag", [True, False], ids=["alpb", "gbsa"])
def test_h_o_cs_atom_order_permutation_invariance(alpb_flag: bool) -> None:
    """Verify Born radii and solvation energy are invariant under H/O/Cs atom permutation (tblite #431)."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers_1 = torch.tensor([1, 8, 55], device=DEVICE)
    positions_1 = torch.tensor(
        [
            [0.0, 0.0, -1.8],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.2],
        ],
        **dd,
    )
    charges_1 = torch.tensor([0.35, -0.85, 0.50], **dd)

    perm = torch.tensor([2, 1, 0], device=DEVICE)
    numbers_2 = numbers_1[perm]
    positions_2 = positions_1[perm]
    charges_2 = charges_1[perm]

    rads_1 = born.get_born_radii(numbers_1, positions_1)
    rads_2 = born.get_born_radii(numbers_2, positions_2)
    torch.testing.assert_close(rads_1, rads_2[perm], atol=1.0e-14, rtol=1.0e-14)

    gb_1 = alpb.GeneralizedBorn(numbers_1, 78.9, alpb=alpb_flag, **dd)
    gb_2 = alpb.GeneralizedBorn(numbers_2, 78.9, alpb=alpb_flag, **dd)

    e_atom_1 = gb_1.get_monopole_atom_energy(
        gb_1.get_cache(numbers=numbers_1, positions=positions_1), charges_1
    )
    e_atom_2 = gb_2.get_monopole_atom_energy(
        gb_2.get_cache(numbers=numbers_2, positions=positions_2), charges_2
    )
    torch.testing.assert_close(
        e_atom_1, e_atom_2[perm], atol=1.0e-14, rtol=1.0e-14
    )
    torch.testing.assert_close(
        e_atom_1.sum(), e_atom_2.sum(), atol=1.0e-14, rtol=1.0e-14
    )

    calc_vac_1 = Calculator(numbers_1, GFN1_XTB, opts=opts, **dd)
    calc_sol_1 = Calculator(
        numbers_1, GFN1_XTB, interaction=[gb_1], opts=opts, **dd
    )
    gsolv_1 = (
        calc_sol_1.singlepoint(positions_1).scf
        - calc_vac_1.singlepoint(positions_1).scf
    )

    calc_vac_2 = Calculator(numbers_2, GFN1_XTB, opts=opts, **dd)
    calc_sol_2 = Calculator(
        numbers_2, GFN1_XTB, interaction=[gb_2], opts=opts, **dd
    )
    gsolv_2 = (
        calc_sol_2.singlepoint(positions_2).scf
        - calc_vac_2.singlepoint(positions_2).scf
    )

    torch.testing.assert_close(
        gsolv_1, gsolv_2[perm], atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        gsolv_1.sum(), gsolv_2.sum(), atol=1.0e-13, rtol=1.0e-13
    )
