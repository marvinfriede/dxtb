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
"""Tests for B8a, B8b, and E8 functional property and 3rd-order evaluators."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.properties.functional import (
    system_parameter_force_jacobian,
)
from dxtb.config import Config

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}


def _setup_model_and_calculator(method: object, numbers: tuple[int, ...]):
    is_gfn2 = method is GFN2_XTB
    opts = {
        "method": labels.GFN2_XTB if is_gfn2 else labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": (
            labels.SCP_MODE_FOCK if is_gfn2 else labels.SCP_MODE_CHARGE
        ),
        "mixer": (labels.MIXER_ANDERSON if is_gfn2 else labels.MIXER_LINEAR),
        "maxiter": 3,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    config = Config.create(**opts)
    par = ParamModule(method, **_DD)
    model = Model(par=par, config=config)
    num_tensor = torch.tensor(numbers, dtype=torch.int64)
    system = model.setup(num_tensor)
    calc = Calculator(num_tensor, par, opts=opts | {"verbosity": 0}, **_DD)
    return model, num_tensor, system, calc


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB])
def test_b8a_first_and_second_order_functional_properties(
    method: object,
) -> None:
    """System 1st/2nd-order properties match Calculator and compose under vmap."""
    _, _, system, calc = _setup_model_and_calculator(method, (8, 1, 1))
    pos = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]], **_DD
    )
    field0 = torch.zeros((3,), **_DD, requires_grad=True)
    fgrad0 = torch.zeros((3, 3), **_DD, requires_grad=True)
    pos_req = pos.clone().requires_grad_(True)

    # Energy, forces, Hessian, bond orders
    torch.testing.assert_close(
        system.energy(pos), calc.energy(pos_req), atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        system.forces(pos), calc.forces(pos_req), atol=1.0e-11, rtol=1.0e-11
    )
    torch.testing.assert_close(
        system.hessian(pos),
        calc.hessian(pos_req),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        system.hessian(pos, matrix=True),
        calc.hessian(pos_req, matrix=True),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        system.bond_orders(pos),
        calc.bond_orders(pos),
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    # Dipole, quadrupole, polarizability, dipole_deriv
    torch.testing.assert_close(
        system.dipole(pos),
        calc.dipole(pos_req, field=field0),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    torch.testing.assert_close(
        system.quadrupole(pos),
        calc.quadrupole(pos_req, field_grad=fgrad0),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    torch.testing.assert_close(
        system.polarizability(pos),
        calc.polarizability(pos_req, use_analytical=False, field=field0),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        system.dipole_deriv(pos),
        calc.dipole_deriv(pos_req, use_analytical_dipmom=False, field=field0),
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    # Vibrational analysis and IR spectrum without reset_all
    vib_sys = system.vibration(pos)
    vib_calc = calc.vibration(pos_req)
    torch.testing.assert_close(
        vib_sys.freqs, vib_calc.freqs, atol=1.0e-8, rtol=1.0e-8
    )
    ir_sys = system.ir(pos)
    ir_calc = calc.ir(pos_req, use_analytical_dipmom=False, field=field0)
    torch.testing.assert_close(
        ir_sys.freqs, ir_calc.freqs, atol=1.0e-8, rtol=1.0e-8
    )
    torch.testing.assert_close(
        ir_sys.ints, ir_calc.ints, atol=1.0e-8, rtol=1.0e-8
    )

    # vmap over conformer geometries
    batch_pos = torch.stack([pos, pos * 1.02], dim=0)
    vmap_forces = torch.func.vmap(system.forces)(batch_pos)
    vmap_dipole = torch.func.vmap(system.dipole)(batch_pos)
    vmap_pol = torch.func.vmap(system.polarizability)(batch_pos)
    torch.testing.assert_close(
        vmap_forces[0], system.forces(pos), atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        vmap_dipole[0], system.dipole(pos), atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        vmap_pol[0], system.polarizability(pos), atol=1.0e-11, rtol=1.0e-11
    )


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB])
def test_b8b_e8_third_order_hyperpolarizability_raman_and_param_forces(
    method: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Third-order properties (hyperpolarizability, pol_deriv, Raman, 3rd dir) pass E8."""
    model, numbers, system, calc = _setup_model_and_calculator(method, (1, 1))
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.44]], **_DD)
    pos_req = pos.clone().requires_grad_(True)
    field0 = torch.tensor([0.002, -0.001, 0.003], **_DD, requires_grad=True)

    # Verify reset_all is NOT called during Calculator.ir or Calculator.raman on single-System exact core
    reset_calls = 0

    def forbid_reset():
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(calc.classicals, "reset_all", forbid_reset)
    monkeypatch.setattr(calc.interactions, "reset_all", forbid_reset)
    monkeypatch.setattr(calc.integrals, "reset_all", forbid_reset)

    # 1. Hyperpolarizability beta = -d^3E/dF^3
    beta_sys = system.hyperpolarizability(pos, field=field0.detach())
    beta_calc = calc.hyperpolarizability(pos_req, field=field0)
    assert beta_sys.shape == (3, 3, 3)
    assert bool(torch.isfinite(beta_sys).all())
    torch.testing.assert_close(beta_sys, beta_calc, atol=1.0e-9, rtol=1.0e-9)

    # 2. Polarizability derivative d(alpha)/dR and Raman spectrum
    chi_sys = system.pol_deriv(pos, field=field0.detach())
    chi_calc = calc.pol_deriv(pos_req, field=field0)
    assert chi_sys.shape == (3, 3, 2, 3)
    torch.testing.assert_close(chi_sys, chi_calc, atol=1.0e-9, rtol=1.0e-9)

    raman_sys = system.raman(pos, field=field0.detach())
    raman_calc = calc.raman(pos_req, field=field0)
    torch.testing.assert_close(
        raman_sys.freqs, raman_calc.freqs, atol=1.0e-8, rtol=1.0e-8
    )
    torch.testing.assert_close(
        raman_sys.ints, raman_calc.ints, atol=1.0e-8, rtol=1.0e-8
    )
    torch.testing.assert_close(
        raman_sys.depol, raman_calc.depol, atol=1.0e-8, rtol=1.0e-8
    )
    assert reset_calls == 0

    # 3. Third directional derivative vs 4-point central finite difference of 2nd derivative
    direction = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], **_DD)
    direction = direction / torch.linalg.vector_norm(direction)
    third_sys = system.third_order_directional(pos, direction)
    step = 1.0e-3
    h_vals = [
        (
            system.hessian(pos + mult * step * direction, matrix=True)
            @ direction.reshape(-1)
            * direction.reshape(-1)
        ).sum()
        for mult in (-2.0, -1.0, 1.0, 2.0)
    ]
    fd_third = (h_vals[0] - 8.0 * h_vals[1] + 8.0 * h_vals[2] - h_vals[3]) / (
        12.0 * step
    )
    torch.testing.assert_close(third_sys, fd_third, atol=1.0e-8, rtol=1.0e-8)

    # 4. Mixed parameter-force Jacobian dF(R, theta)/d(theta)
    param_key = "parameter_tree.hamiltonian.xtb.shell.ss.param"
    pf_jac = system_parameter_force_jacobian(model, numbers, pos, param_key)
    assert pf_jac.shape == (2, 3)
    assert bool(torch.isfinite(pf_jac).all()) and pf_jac.abs().max() > 0.0

    # 5. Finite-temperature open-shell free-energy parity (no fenergy double-counting)
    is_gfn2 = method is GFN2_XTB
    ft_opts = {
        "method": labels.GFN2_XTB if is_gfn2 else labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": (
            labels.SCP_MODE_FOCK if is_gfn2 else labels.SCP_MODE_CHARGE
        ),
        "mixer": labels.MIXER_ANDERSON,
        "maxiter": 40,
        "fermi_etemp": 5000.0,
        "x_atol": 1.0e-12,
        "x_atol_max": 1.0e-12,
    }
    ft_cfg = Config.create(**ft_opts)
    ft_par = ParamModule(method, **_DD)
    ft_model = Model(par=ft_par, config=ft_cfg)
    oh_numbers = torch.tensor([8, 1], dtype=torch.int64)
    oh_sys = ft_model.setup(oh_numbers)
    oh_calc = Calculator(
        oh_numbers, ft_par, opts=ft_opts | {"verbosity": 0}, **_DD
    )
    oh_pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], **_DD)
    oh_res = oh_sys.singlepoint(oh_pos)
    assert float(oh_res.fenergy.sum().abs()) > 1.0e-3
    torch.testing.assert_close(
        oh_sys.energy(oh_pos, include_free_energy=True),
        oh_calc.energy(oh_pos),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        oh_sys.energy(oh_pos, include_free_energy=False),
        (oh_res.energy - oh_res.fenergy).sum(-1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        oh_sys.forces(oh_pos),
        oh_calc.forces(oh_pos.clone().requires_grad_(True)),
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def _setup_converged_model_and_calculator(
    method: object, numbers: tuple[int, ...]
):
    """Build Contract B genuinely converged GFN1/GFN2 System and Calculator."""
    is_gfn2 = method is GFN2_XTB
    maxiter = 10 if numbers == (1, 1) else (15 if is_gfn2 else 12)
    opts = {
        "method": labels.GFN2_XTB if is_gfn2 else labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": (
            labels.SCP_MODE_FOCK if is_gfn2 else labels.SCP_MODE_CHARGE
        ),
        "mixer": labels.MIXER_ANDERSON,
        "maxiter": maxiter,
        "x_atol": 1.0e-12,
        "x_atol_max": 1.0e-12,
    }
    config = Config.create(**opts)
    par = ParamModule(method, **_DD)
    model = Model(par=par, config=config)
    num_tensor = torch.tensor(numbers, dtype=torch.int64)
    system = model.setup(num_tensor)
    calc = Calculator(num_tensor, par, opts=opts | {"verbosity": 0}, **_DD)
    return model, num_tensor, system, calc


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB])
def test_contract_b_converged_water_properties_symmetries_and_mixed_derivatives(
    method: object,
) -> None:
    """Contract B: Converged GFN1/GFN2 H2O properties, symmetries, FD, and mixed-derivative identities."""
    from dxtb._src.calculators.properties.functional import (
        system_dipole,
        system_dipole_deriv,
        system_energy,
        system_forces,
        system_polarizability,
    )

    _, _, system, calc = _setup_converged_model_and_calculator(
        method, (8, 1, 1)
    )
    pos = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]], **_DD
    )
    res = system.singlepoint(pos)
    assert res.converged is not None and bool(res.converged)
    assert (
        res.residual is not None and float(res.residual.abs().max()) < 1.0e-12
    )

    field0 = torch.tensor([0.0015, -0.001, 0.002], **_DD, requires_grad=True)
    fgrad0 = torch.zeros((3, 3), **_DD, requires_grad=True)
    pos_req = pos.clone().requires_grad_(True)

    # 1. Converged parity against Calculator across 1st, 2nd, and 3rd order properties
    e_sys = system.energy(pos, field=field0.detach())
    f_sys = system.forces(pos, field=field0.detach())
    h_sys = system.hessian(pos, field=field0.detach())
    h_mat = system.hessian(pos, field=field0.detach(), matrix=True)
    mu_sys = system.dipole(pos, field=field0.detach())
    q_sys = system.quadrupole(pos, field=field0.detach())
    alpha_sys = system.polarizability(pos, field=field0.detach())
    dmu_dr_sys = system.dipole_deriv(pos, field=field0.detach())
    chi_sys = system.pol_deriv(pos, field=field0.detach())
    beta_sys = system.hyperpolarizability(pos, field=field0.detach())

    torch.testing.assert_close(
        e_sys, calc.energy(pos_req, field=field0), atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(
        f_sys, calc.forces(pos_req, field=field0), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        h_sys, calc.hessian(pos_req, field=field0), atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        mu_sys, calc.dipole(pos_req, field=field0), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        q_sys,
        calc.quadrupole(pos_req, field=field0, field_grad=fgrad0),
        atol=1e-9,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        alpha_sys,
        calc.polarizability(pos_req, use_analytical=False, field=field0),
        atol=1e-8,
        rtol=1e-8,
    )
    torch.testing.assert_close(
        dmu_dr_sys,
        calc.dipole_deriv(pos_req, use_analytical_dipmom=False, field=field0),
        atol=1e-8,
        rtol=1e-8,
    )
    if method is GFN1_XTB:
        torch.testing.assert_close(
            chi_sys, calc.pol_deriv(pos_req, field=field0), atol=1e-7, rtol=1e-7
        )
    torch.testing.assert_close(
        beta_sys,
        calc.hyperpolarizability(pos_req, field=field0),
        atol=1e-7,
        rtol=1e-7,
    )

    # 2. Physical symmetries: Hessian, polarizability, and hyperpolarizability Kleinman permutation symmetry
    torch.testing.assert_close(h_mat, h_mat.T, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(alpha_sys, alpha_sys.T, atol=1e-10, rtol=1e-10)
    for perm in (
        (1, 0, 2),
        (0, 2, 1),
        (2, 1, 0),
        (1, 2, 0),
        (2, 0, 1),
    ):
        torch.testing.assert_close(
            beta_sys, beta_sys.permute(*perm), atol=1e-9, rtol=1e-9
        )
    # Polarizability derivative symmetry in its two field indices: chi_{ab,ic} == chi_{ba,ic}
    torch.testing.assert_close(
        chi_sys, chi_sys.transpose(0, 1), atol=1e-9, rtol=1e-9
    )

    # 3. Translational invariance (neutral system in uniform external field)
    torch.testing.assert_close(
        f_sys.sum(dim=0), torch.zeros(3, **_DD), atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(
        h_sys.sum(dim=2), torch.zeros((3, 3, 3), **_DD), atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        dmu_dr_sys.sum(dim=1), torch.zeros((3, 3), **_DD), atol=1e-9, rtol=1e-9
    )

    # 4. Equivalent mixed-derivative constructions
    # (a) Hessian via jacrev(grad(E)) vs jacfwd(grad(E))
    hess_rev = torch.func.jacrev(
        torch.func.grad(
            lambda p: system_energy(system, p, field=field0.detach())
        )
    )(pos)
    torch.testing.assert_close(h_sys, hess_rev, atol=1e-10, rtol=1e-10)

    # (b) Dipole derivative d(mu)/dR == d(forces)/dF
    df_dfield = torch.func.jacfwd(
        lambda f: system_forces(system, pos, field=f)
    )(field0.detach()).permute(2, 0, 1)
    torch.testing.assert_close(dmu_dr_sys, df_dfield, atol=1e-9, rtol=1e-9)

    # (c) Polarizability derivative d(alpha)/dR == d(dipole_deriv)/dF
    ddmudr_df = torch.func.jacfwd(
        lambda f: system_dipole_deriv(system, pos, field=f)
    )(field0.detach()).permute(0, 3, 1, 2)
    torch.testing.assert_close(chi_sys, ddmudr_df, atol=1e-8, rtol=1e-8)

    # 5. Independent central finite-difference checks on converged root
    eps_f = 1.0e-4
    fd_alpha = torch.empty((3, 3), **_DD)
    fd_beta = torch.empty((3, 3, 3), **_DD)
    for k in range(3):
        df = torch.zeros(3, **_DD)
        df[k] = eps_f
        f_p = field0.detach() + df
        f_m = field0.detach() - df
        fd_alpha[:, k] = (
            system_dipole(system, pos, field=f_p)
            - system_dipole(system, pos, field=f_m)
        ) / (2.0 * eps_f)
        fd_beta[:, :, k] = (
            system_polarizability(system, pos, field=f_p)
            - system_polarizability(system, pos, field=f_m)
        ) / (2.0 * eps_f)
    torch.testing.assert_close(alpha_sys, fd_alpha, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(beta_sys, fd_beta, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB])
def test_contract_b_equilibrium_vibrations_ir_raman_and_third_order(
    method: object,
) -> None:
    """Contract B: Converged equilibrium H2 and H2O vibrational zero modes, IR, Raman, and 3rd-order FD."""
    is_gfn2 = method is GFN2_XTB
    # 1. Converged H2 at exact stationary equilibrium geometry (dE/dR == 0)
    _, _, sys_h2, calc_h2 = _setup_converged_model_and_calculator(
        method, (1, 1)
    )
    z_h2 = 1.4677462454976220 if is_gfn2 else 1.4467531706758971
    pos_h2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, z_h2]], **_DD)
    res_h2 = sys_h2.singlepoint(pos_h2)
    assert res_h2.converged is not None and bool(res_h2.converged)
    assert (
        res_h2.residual is not None
        and float(res_h2.residual.abs().max()) < 1.0e-12
    )
    assert float(sys_h2.forces(pos_h2).abs().max()) < 1.0e-11

    # At equilibrium: unprojected vibration has 5 zero modes (3 translational + 2 rotational) and 1 positive mode
    vib_unproj = sys_h2.vibration(
        pos_h2, project_translational=False, project_rotational=False
    )
    assert vib_unproj.freqs.shape == (6,)
    assert float(vib_unproj.freqs[:5].abs().max()) < 1.0e-6
    assert float(vib_unproj.freqs[5]) > 1.0e-2

    # Projected vibration has exactly 1 stretching mode matching unprojected mode 5
    vib_proj = sys_h2.vibration(
        pos_h2, project_translational=True, project_rotational=True
    )
    assert vib_proj.freqs.shape == (1,)
    torch.testing.assert_close(
        vib_proj.freqs[0], vib_unproj.freqs[5], atol=1e-8, rtol=1e-8
    )

    # IR and Raman at equilibrium H2: homonuclear diatomic has zero IR intensity and positive Raman intensity
    ir_h2 = sys_h2.ir(pos_h2)
    raman_h2 = sys_h2.raman(pos_h2)
    assert float(ir_h2.ints.abs().max()) < 1.0e-12
    assert float(raman_h2.ints[0]) > 1.0e-2

    pos_h2_req = pos_h2.clone().requires_grad_(True)
    field0 = torch.zeros(3, **_DD, requires_grad=True)
    raman_calc = calc_h2.raman(pos_h2_req, field=field0)
    torch.testing.assert_close(
        raman_h2.freqs, raman_calc.freqs, atol=1e-8, rtol=1e-8
    )
    torch.testing.assert_close(
        raman_h2.ints, raman_calc.ints, atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        raman_h2.depol, raman_calc.depol, atol=1e-7, rtol=1e-7
    )

    # Third directional derivative on converged H2 vs 4-point central FD of Hessian
    direction = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], **_DD)
    direction = direction / torch.linalg.vector_norm(direction)
    third_h2 = sys_h2.third_order_directional(pos_h2, direction)
    step = 1.0e-3
    h_vals = [
        (
            sys_h2.hessian(pos_h2 + mult * step * direction, matrix=True)
            @ direction.reshape(-1)
            * direction.reshape(-1)
        ).sum()
        for mult in (-2.0, -1.0, 1.0, 2.0)
    ]
    fd_third = (h_vals[0] - 8.0 * h_vals[1] + 8.0 * h_vals[2] - h_vals[3]) / (
        12.0 * step
    )
    torch.testing.assert_close(third_h2, fd_third, atol=1e-8, rtol=1e-8)

    # 2. Converged H2O at stationary equilibrium geometry (3N-6 = 3 internal modes)
    _, _, sys_h2o, calc_h2o = _setup_converged_model_and_calculator(
        method, (8, 1, 1)
    )
    x_h2o, z_h2o = (
        (1.4592259594119610, 1.0753387019858363)
        if is_gfn2
        else (1.4567035315471497, 1.0752902294683269)
    )
    pos_h2o = torch.tensor(
        [[0.0, 0.0, 0.0], [-x_h2o, 0.0, z_h2o], [x_h2o, 0.0, z_h2o]], **_DD
    )
    assert float(sys_h2o.forces(pos_h2o).abs().max()) < 1.0e-11
    res_h2o = sys_h2o.singlepoint(pos_h2o)
    assert res_h2o.converged is not None and bool(res_h2o.converged)
    assert (
        res_h2o.residual is not None
        and float(res_h2o.residual.abs().max()) < 1.0e-12
    )

    vib_h2o_unproj = sys_h2o.vibration(
        pos_h2o, project_translational=False, project_rotational=False
    )
    vib_h2o_proj = sys_h2o.vibration(
        pos_h2o, project_translational=True, project_rotational=True
    )
    assert vib_h2o_unproj.freqs.shape == (9,)
    assert vib_h2o_proj.freqs.shape == (3,)
    # 6 translational + rotational zero modes at equilibrium
    assert float(vib_h2o_unproj.freqs[:6].abs().max()) < 1.0e-5
    torch.testing.assert_close(
        vib_h2o_proj.freqs, vib_h2o_unproj.freqs[6:], atol=1e-6, rtol=1e-6
    )

    ir_h2o = sys_h2o.ir(pos_h2o)
    raman_h2o = sys_h2o.raman(pos_h2o)
    pos_h2o_req = pos_h2o.clone().requires_grad_(True)
    ir_h2o_calc = calc_h2o.ir(
        pos_h2o_req, use_analytical_dipmom=False, field=field0
    )
    torch.testing.assert_close(
        ir_h2o.ints, ir_h2o_calc.ints, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        raman_h2o.freqs, vib_h2o_proj.freqs, atol=1e-8, rtol=1e-8
    )
    if not is_gfn2:
        raman_h2o_calc = calc_h2o.raman(pos_h2o_req, field=field0)
        torch.testing.assert_close(
            raman_h2o.ints, raman_h2o_calc.ints, atol=1e-6, rtol=1e-6
        )
    assert (ir_h2o.ints > 0.0).all()
    assert (raman_h2o.ints > 0.0).all()
    assert (raman_h2o.depol >= 0.0).all()
