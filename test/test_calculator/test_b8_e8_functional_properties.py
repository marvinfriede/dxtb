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
        "mixer": (
            labels.MIXER_ANDERSON if is_gfn2 else labels.MIXER_LINEAR
        ),
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
        calc.polarizability(
            pos_req, use_analytical=False, field=field0
        ),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        system.dipole_deriv(pos),
        calc.dipole_deriv(
            pos_req, use_analytical_dipmom=False, field=field0
        ),
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
    torch.testing.assert_close(
        beta_sys, beta_calc, atol=1.0e-9, rtol=1.0e-9
    )

    # 2. Polarizability derivative d(alpha)/dR and Raman spectrum
    chi_sys = system.pol_deriv(pos, field=field0.detach())
    chi_calc = calc.pol_deriv(pos_req, field=field0)
    assert chi_sys.shape == (3, 3, 2, 3)
    torch.testing.assert_close(
        chi_sys, chi_calc, atol=1.0e-9, rtol=1.0e-9
    )

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
    fd_third = (
        h_vals[0] - 8.0 * h_vals[1] + 8.0 * h_vals[2] - h_vals[3]
    ) / (12.0 * step)
    torch.testing.assert_close(
        third_sys, fd_third, atol=1.0e-8, rtol=1.0e-8
    )

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
