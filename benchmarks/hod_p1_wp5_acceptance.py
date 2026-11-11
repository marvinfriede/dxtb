#!/usr/bin/env python3
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
"""WP5 higher-order numerical acceptance suite for HOD-P1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.properties.functional import (
    system_dipole,
    system_energy,
    system_parameter_force_jacobian,
    system_polarizability,
)
from dxtb.config import Config
from test.test_baseline.molecules import SYSTEMS as BASELINE_SYSTEMS
from test.utils import scf_strategy

DD: dict[str, Any] = {"device": torch.device("cpu"), "dtype": torch.float64}

SYSTEMS: dict[str, dict[str, Any]] = {
    "H2": {
        "numbers": [1, 1],
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.40]],
        "chrg": 0.0,
        "spin": None,
        "maxiter": 8,
    },
    "LiH": {
        "numbers": [3, 1],
        "positions": [[0.0, 0.0, -1.50796689], [0.0, 0.0, 1.50796689]],
        "chrg": 0.0,
        "spin": None,
        "maxiter": 15,
    },
    "H2O": {
        "numbers": [8, 1, 1],
        "positions": [
            [0.0, 0.0, -0.742885],
            [-1.434727, 0.0, 0.371443],
            [1.434727, 0.0, 0.371443],
        ],
        "chrg": 0.0,
        "spin": None,
        "maxiter": 18,
    },
    "NO2": {
        "numbers": [7, 8, 8],
        "positions": BASELINE_SYSTEMS["NO2"].positions.tolist(),
        "chrg": 0.0,
        "spin": 1.0,
        "maxiter": 25,
    },
}


def _make_system_and_calc(
    mol_name: str,
    method_name: str,
    driver_name: str,
    diff_mode: str,
    maxiter: int = 40,
) -> tuple[Model, torch.Tensor, torch.Tensor, float, float | None, Any, Calculator]:
    mol = SYSTEMS[mol_name]
    is_gfn2 = method_name == "gfn2"
    method_obj = GFN2_XTB if is_gfn2 else GFN1_XTB
    numbers = torch.tensor(mol["numbers"], dtype=torch.int64, device=DD["device"])
    positions = torch.tensor(mol["positions"], **DD)
    chrg = float(mol["chrg"])
    spin = mol["spin"]

    opts: dict[str, Any] = {
        "method": labels.GFN2_XTB if is_gfn2 else labels.GFN1_XTB,
        "int_driver": (
            labels.INTDRIVER_LIBCINT
            if driver_name == "libcint"
            else labels.INTDRIVER_PYTORCH
        ),
        "int_level": (
            labels.INTLEVEL_QUADRUPOLE if is_gfn2 else labels.INTLEVEL_DIPOLE
        ),
        "differentiation": diff_mode,
        "strategy": scf_strategy(
            "anderson",
            "potential"
            if mol["spin"] is not None
            else ("fock" if is_gfn2 else "charge"),
        ),
        "maxiter": maxiter,
        "x_atol": 1.0e-12,
        "x_atol_max": 1.0e-12,
        "fermi_diff_order": 0 if mol["spin"] is None else 3,
    }
    if mol["spin"] is not None:
        opts["fermi_etemp"] = 300.0
        opts["fermi_thresh"] = 1.0e-10

    cfg = Config.create(**opts)
    par = ParamModule(method_obj, **DD)
    model = Model(par=par, config=cfg)
    system = model.setup(numbers)
    calc = Calculator(numbers, par, opts=opts | {"verbosity": 0}, **DD)
    return model, numbers, positions, chrg, spin, system, calc


def _fd_hyperpol_from_pol(
    pol_fn,
    f0: torch.Tensor,
    step: float,
    *,
    four_point: bool = True,
) -> torch.Tensor:
    """2-point or 4-point central finite difference of analytical polarizability."""
    cols = []
    for k in range(3):
        ek = torch.zeros_like(f0)
        ek[k] = 1.0
        p_m1 = pol_fn(f0 - 1.0 * step * ek)
        p_p1 = pol_fn(f0 + 1.0 * step * ek)
        if four_point:
            p_m2 = pol_fn(f0 - 2.0 * step * ek)
            p_p2 = pol_fn(f0 + 2.0 * step * ek)
            dpol_dk = (p_m2 - 8.0 * p_m1 + 8.0 * p_p1 - p_p2) / (12.0 * step)
        else:
            dpol_dk = (p_p1 - p_m1) / (2.0 * step)
        cols.append(dpol_dk)
    return torch.stack(cols, dim=-1)


def _permutation_symmetry_defect(beta: torch.Tensor) -> float:
    perms = [
        beta.permute(0, 1, 2),
        beta.permute(0, 2, 1),
        beta.permute(1, 0, 2),
        beta.permute(1, 2, 0),
        beta.permute(2, 0, 1),
        beta.permute(2, 1, 0),
    ]
    return float(max((p - beta).abs().max().item() for p in perms[1:]))


def _rotation_matrix() -> torch.Tensor:
    angle1 = torch.tensor(0.37, **DD)
    angle2 = torch.tensor(-0.61, **DD)
    c1, s1 = torch.cos(angle1), torch.sin(angle1)
    c2, s2 = torch.cos(angle2), torch.sin(angle2)
    rz = torch.tensor([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], **DD)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]], **DD)
    return rz @ rx


def main() -> None:
    report: dict[str, Any] = {
        "single_system_matrix": [],
        "fd_step_sweep_lih": {},
        "rotation_covariance": {},
        "batch_and_vmap": {},
        "other_higher_order_properties": {},
    }

    # 1. Single-system matrix across H2, LiH, H2O, NO2; GFN1/GFN2; pytorch/libcint
    for mol_name in ("H2", "LiH", "H2O", "NO2"):
        for method_name in ("gfn1", "gfn2"):
            for driver_name in ("pytorch", "libcint"):
                mol_maxiter = int(SYSTEMS[mol_name]["maxiter"])
                _, _, pos, chrg, spin, system, calc = _make_system_and_calc(
                    mol_name, method_name, driver_name, "unrolled", maxiter=mol_maxiter
                )
                f0 = torch.zeros((3,), **DD)
                f_req = f0.clone().requires_grad_(True)
                pos_req = pos.clone().requires_grad_(True)

                mu_sys = system.dipole(pos, chrg=chrg, spin=spin, field=f0)
                mu_calc = calc.dipole(
                    pos_req, chrg=chrg, spin=spin, field=f_req, use_functorch=False
                )
                alpha_sys = system.polarizability(
                    pos, chrg=chrg, spin=spin, field=f0
                )
                alpha_calc = calc.polarizability(
                    pos_req, chrg=chrg, spin=spin, field=f_req, use_functorch=False
                )
                beta_sys = system.hyperpolarizability(
                    pos, chrg=chrg, spin=spin, field=f0
                )
                beta_calc_ag = calc.hyperpolarizability(
                    pos_req, chrg=chrg, spin=spin, field=f_req, use_functorch=False
                )
                beta_calc_ft_pol = calc.hyperpolarizability(
                    pos_req,
                    chrg=chrg,
                    spin=spin,
                    field=f_req,
                    use_functorch=True,
                    derived_quantity="pol",
                )
                beta_calc_ft_dip = calc.hyperpolarizability(
                    pos_req,
                    chrg=chrg,
                    spin=spin,
                    field=f_req,
                    use_functorch=True,
                    derived_quantity="dipole",
                )
                beta_calc_ft_ene = calc.hyperpolarizability(
                    pos_req,
                    chrg=chrg,
                    spin=spin,
                    field=f_req,
                    use_functorch=True,
                    derived_quantity="energy",
                )

                def _pol_for_fd(f: torch.Tensor) -> torch.Tensor:
                    f_in = f.detach().clone().requires_grad_(True)
                    return calc.polarizability(
                        pos.detach().clone(),
                        chrg=chrg,
                        spin=spin,
                        field=f_in,
                        use_functorch=False,
                    ).detach()

                beta_fd = _fd_hyperpol_from_pol(
                    _pol_for_fd, f0, step=1.0e-4, four_point=False
                )

                entry = {
                    "molecule": mol_name,
                    "method": method_name,
                    "driver": driver_name,
                    "shape": list(beta_sys.shape),
                    "all_finite": bool(torch.isfinite(beta_sys).all().item()),
                    "dipole_sys_vs_calc_max_abs": float(
                        (mu_sys - mu_calc).abs().max().item()
                    ),
                    "pol_sys_vs_calc_max_abs": float(
                        (alpha_sys - alpha_calc).abs().max().item()
                    ),
                    "beta_sys_vs_calc_autograd_max_abs": float(
                        (beta_sys - beta_calc_ag).abs().max().item()
                    ),
                    "beta_calc_ag_vs_ft_pol_max_abs": float(
                        (beta_calc_ag - beta_calc_ft_pol).abs().max().item()
                    ),
                    "beta_ft_pol_vs_ft_dip_max_abs": float(
                        (beta_calc_ft_pol - beta_calc_ft_dip).abs().max().item()
                    ),
                    "beta_ft_dip_vs_ft_ene_max_abs": float(
                        (beta_calc_ft_dip - beta_calc_ft_ene).abs().max().item()
                    ),
                    "beta_sys_vs_fd_max_abs": float(
                        (beta_sys - beta_fd).abs().max().item()
                    ),
                    "beta_calc_ag_vs_fd_max_abs": float(
                        (beta_calc_ag - beta_fd).abs().max().item()
                    ),
                    "beta_perm_symmetry_defect_sys": _permutation_symmetry_defect(
                        beta_sys
                    ),
                    "beta_perm_symmetry_defect_calc_ag": _permutation_symmetry_defect(
                        beta_calc_ag
                    ),
                }
                print(
                    f"[CHECK] {mol_name:4s} {method_name:4s} {driver_name:7s} "
                    f"sys_vs_fd={entry['beta_sys_vs_fd_max_abs']:.2e} "
                    f"ag_vs_fd={entry['beta_calc_ag_vs_fd_max_abs']:.2e} "
                    f"ag_vs_ft={entry['beta_calc_ag_vs_ft_pol_max_abs']:.2e} "
                    f"sym={entry['beta_perm_symmetry_defect_sys']:.2e}",
                    flush=True,
                )
                assert entry["shape"] == [3, 3, 3]
                assert entry["all_finite"]
                assert entry["beta_calc_ag_vs_ft_pol_max_abs"] < 1.0e-5
                assert entry["beta_ft_pol_vs_ft_dip_max_abs"] < 1.0e-5
                assert entry["beta_ft_dip_vs_ft_ene_max_abs"] < 1.0e-5
                assert entry["beta_sys_vs_fd_max_abs"] < 1.0e-2
                assert entry["beta_perm_symmetry_defect_sys"] < 1.0e-5
                assert entry["beta_perm_symmetry_defect_calc_ag"] < 1.0e-5
                report["single_system_matrix"].append(entry)

    # 2. FD step-size sweep on LiH (h = 1e-3, 5e-4, 1e-4, 5e-5) + implicit vs unrolled
    _, _, pos_lih, chrg_lih, spin_lih, sys_lih_unr, calc_lih_unr = (
        _make_system_and_calc("LiH", "gfn1", "pytorch", "unrolled", maxiter=15)
    )
    _, _, _, _, _, sys_lih_imp, calc_lih_imp = _make_system_and_calc(
        "LiH", "gfn1", "pytorch", "implicit", maxiter=40
    )
    f0 = torch.zeros((3,), **DD)
    f_req = f0.clone().requires_grad_(True)
    beta_sys_unr = sys_lih_unr.hyperpolarizability(pos_lih, field=f0)
    beta_calc_unr = calc_lih_unr.hyperpolarizability(
        pos_lih.clone().requires_grad_(True), field=f_req, use_functorch=False
    )
    pol_calc_imp = calc_lih_imp.polarizability(
        pos_lih.clone().requires_grad_(True), field=f_req, use_functorch=False
    )
    pol_sys_unr = sys_lih_unr.polarizability(pos_lih, field=f0)

    def _lih_pol_unr(f: torch.Tensor) -> torch.Tensor:
        f_in = f.detach().clone().requires_grad_(True)
        return calc_lih_unr.polarizability(
            pos_lih.detach().clone(), field=f_in, use_functorch=False
        ).detach()

    def _lih_pol_imp(f: torch.Tensor) -> torch.Tensor:
        f_in = f.detach().clone().requires_grad_(True)
        return calc_lih_imp.polarizability(
            pos_lih.detach().clone(), field=f_in, use_functorch=False
        ).detach()

    step_results = {}
    for h in (1.0e-3, 5.0e-4, 1.0e-4, 5.0e-5):
        fd_unr = _fd_hyperpol_from_pol(_lih_pol_unr, f0, step=h)
        fd_imp = _fd_hyperpol_from_pol(_lih_pol_imp, f0, step=h)
        step_results[str(h)] = {
            "beta_sys_unr_vs_fd_unr_max_abs": float(
                (beta_sys_unr - fd_unr).abs().max().item()
            ),
            "beta_calc_unr_vs_fd_unr_max_abs": float(
                (beta_calc_unr - fd_unr).abs().max().item()
            ),
            "fd_unr_vs_fd_imp_max_abs": float(
                (fd_unr[:, :, 2] - fd_imp[:, :, 2]).abs().max().item()
            ),
        }
    report["fd_step_sweep_lih"] = {
        "pol_unrolled_vs_implicit_max_abs": float(
            (pol_sys_unr - pol_calc_imp).abs().max().item()
        ),
        "steps": step_results,
    }
    print("[OK] Section 2 (FD step sweep + implicit vs unrolled)", flush=True)

    # 3. Rotation covariance on H2O with non-zero electric field
    _, _, pos_h2o, _, _, sys_h2o, calc_h2o = _make_system_and_calc(
        "H2O", "gfn1", "pytorch", "unrolled", maxiter=18
    )
    rot = _rotation_matrix()
    field_vec = torch.tensor([0.002, -0.001, 0.0015], **DD)
    pos_rot = pos_h2o @ rot.T
    field_rot = field_vec @ rot.T

    beta_orig = sys_h2o.hyperpolarizability(pos_h2o, field=field_vec)
    beta_rot = sys_h2o.hyperpolarizability(pos_rot, field=field_rot)
    beta_cov = torch.einsum("ia,jb,kc,abc->ijk", rot, rot, rot, beta_orig)
    rot_err_sys = float((beta_rot - beta_cov).abs().max().item())

    f_orig_req = field_vec.clone().requires_grad_(True)
    f_rot_req = field_rot.clone().requires_grad_(True)
    beta_calc_orig = calc_h2o.hyperpolarizability(
        pos_h2o.clone().requires_grad_(True),
        field=f_orig_req,
        use_functorch=False,
    )
    beta_calc_rot = calc_h2o.hyperpolarizability(
        pos_rot.clone().requires_grad_(True),
        field=f_rot_req,
        use_functorch=False,
    )
    beta_calc_cov = torch.einsum(
        "ia,jb,kc,abc->ijk", rot, rot, rot, beta_calc_orig
    )
    rot_err_calc = float((beta_calc_rot - beta_calc_cov).abs().max().item())
    assert rot_err_sys < 1.0e-8
    assert rot_err_calc < 1.0e-8
    report["rotation_covariance"] = {
        "h2o_system_rot_covariance_max_abs": rot_err_sys,
        "h2o_calc_autograd_rot_covariance_max_abs": rot_err_calc,
    }

    # 4. Batched Calculator (shared & per-lane field) and System vmap
    opts_lin = {
        "method": labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": labels.INTLEVEL_DIPOLE,
        "differentiation": "unrolled",
        "strategy": scf_strategy("linear", "charge"),
        "maxiter": 20,
        "x_atol": 1.0e-12,
        "x_atol_max": 1.0e-12,
        "fermi_diff_order": 0,
    }
    par_gfn1 = ParamModule(GFN1_XTB, **DD)
    num_lih_t = torch.tensor([3, 1], dtype=torch.int64, device=DD["device"])
    sys_lih_lin = Model(par=par_gfn1, config=Config.create(**opts_lin)).setup(
        num_lih_t
    )
    calc_lih_lin = Calculator(
        num_lih_t, par_gfn1, opts=opts_lin | {"verbosity": 0}, **DD
    )
    pos_batch_same = torch.stack([pos_lih, pos_lih * 1.03], dim=0)

    def _vmap_hyp_shared(p: torch.Tensor) -> torch.Tensor:
        return sys_lih_lin.hyperpolarizability(p, field=f0)

    def _vmap_hyp_per_lane(p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return sys_lih_lin.hyperpolarizability(p, field=f)

    f_lanes = torch.stack(
        [
            torch.tensor([0.0, 0.0, 0.0], **DD),
            torch.tensor([0.001, -0.001, 0.002], **DD),
        ],
        dim=0,
    )
    vmap_shared = torch.func.vmap(_vmap_hyp_shared)(pos_batch_same)
    vmap_per_lane = torch.func.vmap(_vmap_hyp_per_lane)(pos_batch_same, f_lanes)
    single_lane0 = sys_lih_lin.hyperpolarizability(pos_lih, field=f0)
    single_lane1 = sys_lih_lin.hyperpolarizability(
        pos_batch_same[1], field=f_lanes[1]
    )
    beta_calc_lih_lin = calc_lih_lin.hyperpolarizability(
        pos_lih.clone().requires_grad_(True),
        field=f0.clone().requires_grad_(True),
        use_functorch=False,
    )

    num_pack = pack(
        [
            torch.tensor([3, 1], dtype=torch.int64),
            torch.tensor([8, 1, 1], dtype=torch.int64),
        ]
    )
    pos_pack = pack([pos_lih, pos_h2o])
    chrg_pack = torch.tensor([0.0, 0.0], **DD)
    opts_pack = opts_lin | {"verbosity": 0}
    calc_pack = Calculator(num_pack, par_gfn1, opts=opts_pack, **DD)
    f_shared_req = f0.clone().requires_grad_(True)
    f_lanes_req = f_lanes.clone().requires_grad_(True)
    beta_pack_shared_ag = calc_pack.hyperpolarizability(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_shared_req,
        use_functorch=False,
    )
    beta_pack_shared_ft = calc_pack.hyperpolarizability(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_shared_req,
        use_functorch=True,
        derived_quantity="pol",
    )
    beta_pack_lanes_ag = calc_pack.hyperpolarizability(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_lanes_req,
        use_functorch=False,
    )
    beta_pack_lanes_ft = calc_pack.hyperpolarizability(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_lanes_req,
        use_functorch=True,
        derived_quantity="pol",
    )

    report["batch_and_vmap"] = {
        "vmap_shared_lane0_vs_single_max_abs": float(
            (vmap_shared[0] - single_lane0).abs().max().item()
        ),
        "vmap_per_lane_lane1_vs_single_max_abs": float(
            (vmap_per_lane[1] - single_lane1).abs().max().item()
        ),
        "calc_pack_shared_ag_vs_ft_max_abs": float(
            (beta_pack_shared_ag - beta_pack_shared_ft).abs().max().item()
        ),
        "calc_pack_per_lane_ag_vs_ft_max_abs": float(
            (beta_pack_lanes_ag - beta_pack_lanes_ft).abs().max().item()
        ),
        "calc_pack_lane0_vs_single_lih_max_abs": float(
            (beta_pack_shared_ag[0] - beta_calc_lih_lin).abs().max().item()
        ),
    }
    for k, val in report["batch_and_vmap"].items():
        assert val < 1.0e-7, f"{k} = {val}"

    # 5. Other higher-order properties: pol_deriv, third_order_directional,
    #    hessian, dipole_deriv, parameter_force_jacobian
    model_h2o, num_h2o, pos_h2o, _, _, sys_h2o, calc_h2o = (
        _make_system_and_calc("H2O", "gfn1", "pytorch", "unrolled", maxiter=18)
    )
    f0_req = f0.clone().requires_grad_(True)
    pos_h2o_req = pos_h2o.clone().requires_grad_(True)

    hess_sys = sys_h2o.hessian(pos_h2o)
    hess_calc_ag = calc_h2o.hessian(pos_h2o_req, use_functorch=False)
    hess_calc_ft = calc_h2o.hessian(pos_h2o_req, use_functorch=True)

    dip_deriv_sys = sys_h2o.dipole_deriv(pos_h2o, field=f0)
    dip_deriv_calc_ag = calc_h2o.dipole_deriv(
        pos_h2o_req, field=f0_req, use_functorch=False
    )
    dip_deriv_calc_ft = calc_h2o.dipole_deriv(
        pos_h2o_req, field=f0_req, use_functorch=True
    )

    pol_deriv_sys = sys_h2o.pol_deriv(pos_h2o, field=f0)
    pol_deriv_calc_ag = calc_h2o.pol_deriv(
        pos_h2o_req, field=f0_req, use_functorch=False
    )
    pol_deriv_calc_ft = calc_h2o.pol_deriv(
        pos_h2o_req, field=f0_req, use_functorch=True
    )

    # Batched pol_deriv check (shared & per-lane field)
    pold_pack_shared_ag = calc_pack.pol_deriv(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_shared_req,
        use_functorch=False,
    )
    pold_pack_shared_ft = calc_pack.pol_deriv(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_shared_req,
        use_functorch=True,
    )
    pold_pack_lanes_ag = calc_pack.pol_deriv(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_lanes_req,
        use_functorch=False,
    )
    pold_pack_lanes_ft = calc_pack.pol_deriv(
        pos_pack.clone().requires_grad_(True),
        chrg_pack,
        field=f_lanes_req,
        use_functorch=True,
    )

    direction = torch.tensor(
        [[0.0, 0.0, -1.0], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]], **DD
    )
    direction = direction / torch.linalg.vector_norm(direction)
    third_dir = sys_h2o.third_order_directional(pos_h2o, direction)
    step = 1.0e-3
    h_vals = [
        (
            sys_h2o.hessian(pos_h2o + mult * step * direction, matrix=True)
            @ direction.reshape(-1)
            * direction.reshape(-1)
        ).sum()
        for mult in (-2.0, -1.0, 1.0, 2.0)
    ]
    fd_third_dir = (
        h_vals[0] - 8.0 * h_vals[1] + 8.0 * h_vals[2] - h_vals[3]
    ) / (12.0 * step)

    param_key = "parameter_tree.hamiltonian.xtb.shell.ss.param"
    pf_jac = system_parameter_force_jacobian(
        model_h2o, num_h2o, pos_h2o, param_key
    )

    report["other_higher_order_properties"] = {
        "hessian_sys_vs_calc_ag_max_abs": float(
            (hess_sys - hess_calc_ag).abs().max().item()
        ),
        "hessian_calc_ag_vs_ft_max_abs": float(
            (hess_calc_ag - hess_calc_ft).abs().max().item()
        ),
        "dipole_deriv_sys_vs_calc_ag_max_abs": float(
            (dip_deriv_sys - dip_deriv_calc_ag).abs().max().item()
        ),
        "dipole_deriv_calc_ag_vs_ft_max_abs": float(
            (dip_deriv_calc_ag - dip_deriv_calc_ft).abs().max().item()
        ),
        "pol_deriv_sys_vs_calc_ag_max_abs": float(
            (pol_deriv_sys - pol_deriv_calc_ag).abs().max().item()
        ),
        "pol_deriv_calc_ag_vs_ft_max_abs": float(
            (pol_deriv_calc_ag - pol_deriv_calc_ft).abs().max().item()
        ),
        "pol_deriv_pack_shared_ag_vs_ft_max_abs": float(
            (pold_pack_shared_ag - pold_pack_shared_ft).abs().max().item()
        ),
        "pol_deriv_pack_per_lane_ag_vs_ft_max_abs": float(
            (pold_pack_lanes_ag - pold_pack_lanes_ft).abs().max().item()
        ),
        "third_order_directional_vs_fd_abs": float(
            (third_dir - fd_third_dir).abs().item()
        ),
        "parameter_force_jacobian_shape": list(pf_jac.shape),
        "parameter_force_jacobian_max_abs": float(pf_jac.abs().max().item()),
    }
    assert report["other_higher_order_properties"]["pol_deriv_calc_ag_vs_ft_max_abs"] < 1.0e-8
    assert report["other_higher_order_properties"]["pol_deriv_pack_shared_ag_vs_ft_max_abs"] < 1.0e-8
    assert report["other_higher_order_properties"]["pol_deriv_pack_per_lane_ag_vs_ft_max_abs"] < 1.0e-8
    assert report["other_higher_order_properties"]["third_order_directional_vs_fd_abs"] < 1.0e-7

    out_path = Path("benchmarks/results/hod_p1_wp5_numerical_acceptance.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved WP5 numerical acceptance report to {out_path}")


if __name__ == "__main__":
    main()
