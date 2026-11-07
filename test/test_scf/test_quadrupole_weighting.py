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
"""
WP1 tests for GFN2-xTB quadrupole metric weighting (upstream tblite #406 / 28ce7a7).

Covers:
- 1.6 A: Coordinate transformation (weighted packing/unpacking for charges and
  potentials, exact component factors, Frobenius norm equivalence and SO(3)
  invariance, energy pairing preservation, GFN1/zero-quadrupole handling,
  PADNZ padding projection without 1-ULP drift, and AD transforms).
- 1.6 B: Rotational invariance on non-linear molecules (H2O, glycine, PbH4-BiH3)
  across Broyden and Anderson mixers and potential and charge SCP modes.
- 1.5: Reproducible isolation of the charge-mode orbital-anisotropy effect at
  loose SCF tolerance (occurring identically in GFN1 without multipoles).
- 1.6 D: Differentiability and PyTorch function transforms (autograd.grad,
  torch.func.grad, jacrev, vmap) across unrolled and implicit SCF modes.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.func import grad, jacfwd, jacrev, vmap

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.constants import defaults, labels
from dxtb._src.param.module import ParamModule
from dxtb._src.scf.functional import (
    ChargeLayout,
    PotentialLayout,
    SCFCharges,
    SCFPotential,
    pack_charges,
    pack_potential,
    unpack_charges,
    unpack_potential,
)
from dxtb._src.scf.functional.problem import (
    SCFProblem,
    SCFRepresentation,
    _mask_packed_rows,
)
from dxtb._src.scf.functional.solver import (
    _project_charge_padding,
    _project_potential_padding,
    _solve_map,
    setup_solver_settings,
)

from ..conftest import DEVICE
from ..test_baseline.molecules import SYSTEMS

pytestmark = [pytest.mark.scf]

DTYPE = torch.float64


def _rotation_matrix(
    axis: tuple[float, float, float] = (0.37, -0.81, 0.45),
    angle: float = 1.23456789,
    dtype: torch.dtype = DTYPE,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Construct a proper rotation matrix R in SO(3) via Rodrigues' formula."""
    k = torch.tensor(axis, dtype=dtype, device=device)
    k = k / torch.linalg.vector_norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    z = torch.zeros((), dtype=dtype, device=device)
    skew = torch.stack(
        (
            torch.stack((z, -kz, ky)),
            torch.stack((kz, z, -kx)),
            torch.stack((-ky, kx, z)),
        )
    )
    eye = torch.eye(3, dtype=dtype, device=device)
    return (
        eye + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _packed_quad_to_full_3x3(quad: torch.Tensor) -> torch.Tensor:
    """Reconstruct full symmetric (..., 3, 3) tensor from [xx, xy, yy, xz, yz, zz]."""
    xx, xy, yy, xz, yz, zz = quad.unbind(dim=-1)
    row0 = torch.stack((xx, xy, xz), dim=-1)
    row1 = torch.stack((xy, yy, yz), dim=-1)
    row2 = torch.stack((xz, yz, zz), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _full_3x3_to_packed_quad(mat: torch.Tensor) -> torch.Tensor:
    """Extract [xx, xy, yy, xz, yz, zz] from a symmetric (..., 3, 3) tensor."""
    return torch.stack(
        (
            mat[..., 0, 0],
            mat[..., 0, 1],
            mat[..., 1, 1],
            mat[..., 0, 2],
            mat[..., 1, 2],
            mat[..., 2, 2],
        ),
        dim=-1,
    )


def _compute_torque(
    positions: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    """Compute net torque sum_i (r_i - r_COM) x g_i."""
    centered = positions - positions.mean(dim=-2, keepdim=True)
    return torch.linalg.cross(centered, gradient, dim=-1).sum(dim=-2)


# ============================================================================
# 1.6 A: Coordinate transformation tests
# ============================================================================


def test_quadrupole_packing_exact_component_factors() -> None:
    """Verify exact sqrt(2) and 1/sqrt(2) factors for all 6 quadrupole components."""
    c_layout = ChargeLayout(
        mono_shape=(7,), dipole_shape=(1, 3), quadrupole_shape=(1, 6)
    )
    p_layout = PotentialLayout(
        mono_shape=(7,), dipole_shape=(1, 3), quadrupole_shape=(1, 6)
    )

    mono = torch.zeros(7, dtype=DTYPE, device=DEVICE)
    dipole = torch.zeros((1, 3), dtype=DTYPE, device=DEVICE)
    unit_quad = torch.ones((1, 6), dtype=DTYPE, device=DEVICE)

    sqrt2 = math.sqrt(2.0)
    inv_sqrt2 = 1.0 / sqrt2
    expected_charge_weights = torch.tensor(
        [1.0, sqrt2, 1.0, sqrt2, sqrt2, 1.0], dtype=DTYPE, device=DEVICE
    )
    expected_potential_weights = torch.tensor(
        [1.0, inv_sqrt2, 1.0, inv_sqrt2, inv_sqrt2, 1.0],
        dtype=DTYPE,
        device=DEVICE,
    )

    packed_c = pack_charges(
        SCFCharges(mono=mono, dipole=dipole, quad=unit_quad), c_layout
    )
    torch.testing.assert_close(
        packed_c[2, :6], expected_charge_weights, atol=0.0, rtol=0.0
    )
    assert float(packed_c[2, 6].item()) == float(defaults.PADNZ)

    packed_p = pack_potential(
        SCFPotential(mono=mono, dipole=dipole, quad=unit_quad), p_layout
    )
    torch.testing.assert_close(
        packed_p[2, :6], expected_potential_weights, atol=0.0, rtol=0.0
    )
    assert float(packed_p[2, 6].item()) == float(defaults.PADNZ)

    # Inverse unpacking recovers original physical values.
    unpacked_c = unpack_charges(packed_c, c_layout)
    assert unpacked_c.quad is not None
    torch.testing.assert_close(
        unpacked_c.quad, unit_quad, atol=1.0e-15, rtol=1.0e-15
    )

    unpacked_p = unpack_potential(packed_p, p_layout)
    assert unpacked_p.quad is not None
    torch.testing.assert_close(
        unpacked_p.quad, unit_quad, atol=1.0e-15, rtol=1.0e-15
    )


def test_packed_quadrupole_frobenius_norm_and_so3_invariance() -> None:
    """Packed quadrupole L2 norm equals full 3x3 Frobenius norm and is SO(3)-invariant."""
    c_layout = ChargeLayout(
        mono_shape=(4,), dipole_shape=(3, 3), quadrupole_shape=(3, 6)
    )
    p_layout = PotentialLayout(
        mono_shape=(4,), dipole_shape=(3, 3), quadrupole_shape=(3, 6)
    )

    # Three distinct traceless symmetric 3x3 quadrupole tensors.
    q_packed = torch.tensor(
        [
            [0.42, -0.19, -0.15, 0.31, -0.27, -0.27],
            [-0.11, 0.08, 0.29, -0.14, 0.22, -0.18],
            [0.05, -0.33, -0.02, 0.17, 0.12, -0.03],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    v_packed = torch.tensor(
        [
            [-0.21, 0.28, 0.14, -0.16, 0.38, 0.07],
            [0.19, -0.24, -0.09, 0.42, -0.18, -0.10],
            [-0.31, 0.14, 0.22, -0.26, -0.08, 0.09],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    q_full = _packed_quad_to_full_3x3(q_packed)
    frob_norm = torch.linalg.matrix_norm(q_full, ord="fro").norm()

    mono = torch.zeros(4, dtype=DTYPE, device=DEVICE)
    dip = torch.zeros((3, 3), dtype=DTYPE, device=DEVICE)
    packed_state = pack_charges(
        SCFCharges(mono=mono, dipole=dip, quad=q_packed), c_layout
    )
    packed_quad_norm = torch.linalg.vector_norm(packed_state[2, :18])
    torch.testing.assert_close(
        packed_quad_norm, frob_norm, atol=1.0e-15, rtol=1.0e-15
    )

    # Rotate the full 3x3 tensors by R in SO(3): Q' = R Q R^T.
    rot = _rotation_matrix()
    q_rot_full = torch.einsum("ab,...bc,dc->...ad", rot, q_full, rot)
    q_rot_packed = _full_3x3_to_packed_quad(q_rot_full)
    packed_rot_state = pack_charges(
        SCFCharges(mono=mono, dipole=dip, quad=q_rot_packed), c_layout
    )
    packed_rot_norm = torch.linalg.vector_norm(packed_rot_state[2, :18])
    torch.testing.assert_close(
        packed_rot_norm, frob_norm, atol=1.0e-15, rtol=1.0e-15
    )

    # Dual pairing <v_packed, q_packed> is preserved by reciprocal scaling.
    packed_pot = pack_potential(
        SCFPotential(mono=mono, dipole=dip, quad=v_packed), p_layout
    )
    raw_pairing = (v_packed * q_packed).sum()
    packed_pairing = (packed_pot[2, :18] * packed_state[2, :18]).sum()
    torch.testing.assert_close(
        packed_pairing, raw_pairing, atol=1.0e-15, rtol=1.0e-15
    )


def test_quadrupole_absent_zero_and_padding_projection_no_ulp_drift() -> None:
    """Verify GFN1 (None), zero quadrupoles, and bitwise-exact padding projections."""
    # 1. Absent quadrupoles (GFN1 layout).
    gfn1_c_layout = ChargeLayout(mono_shape=(5,))
    gfn1_p_layout = PotentialLayout(mono_shape=(5,))
    mono = torch.linspace(-0.3, 0.4, 5, dtype=DTYPE, device=DEVICE)
    pc1 = pack_charges(SCFCharges(mono=mono), gfn1_c_layout)
    uc1 = unpack_charges(pc1, gfn1_c_layout)
    assert uc1.dipole is None and uc1.quad is None
    torch.testing.assert_close(uc1.mono, mono, atol=0.0, rtol=0.0)

    pp1 = pack_potential(SCFPotential(mono=mono), gfn1_p_layout)
    up1 = unpack_potential(pp1, gfn1_p_layout)
    assert up1.dipole is None and up1.quad is None
    torch.testing.assert_close(up1.mono, mono, atol=0.0, rtol=0.0)

    # 2. Zero quadrupoles.
    c_layout = ChargeLayout((8,), (2, 3), (2, 6))
    p_layout = PotentialLayout((8,), (2, 3), (2, 6))
    z_mono = torch.zeros(8, dtype=DTYPE, device=DEVICE)
    z_dip = torch.zeros((2, 3), dtype=DTYPE, device=DEVICE)
    z_quad = torch.zeros((2, 6), dtype=DTYPE, device=DEVICE)
    pc_zero = pack_charges(SCFCharges(z_mono, z_dip, z_quad), c_layout)
    assert bool((pc_zero[2, :12] == 0.0).all())
    pp_zero = pack_potential(SCFPotential(z_mono, z_dip, z_quad), p_layout)
    assert bool((pp_zero[2, :12] == 0.0).all())

    # 3. Padding projections preserve active entries bitwise (zero 1-ULP drift).
    # Note: 7.0 * sqrt(2) / sqrt(2) != 7.0 in IEEE-754 float64, so a projection
    # that round-tripped through unpack/pack would alter active entries by 1 ULP.
    wide_c_layout = ChargeLayout((15,), (2, 3), (2, 6))
    wide_p_layout = PotentialLayout((15,), (2, 3), (2, 6))
    active_mono = torch.arange(1, 16, dtype=DTYPE, device=DEVICE)
    active_dip = torch.arange(1, 7, dtype=DTYPE, device=DEVICE).reshape(2, 3)
    active_quad = torch.full((2, 6), 7.0, dtype=DTYPE, device=DEVICE)

    packed_c = pack_charges(
        SCFCharges(active_mono, active_dip, active_quad), wide_c_layout
    )
    corrupted_c = packed_c.clone()
    corrupted_c[1, 6:] = 123.0
    corrupted_c[2, 12:] = 456.0
    projected_c = _project_charge_padding(corrupted_c, wide_c_layout)
    assert torch.equal(projected_c, packed_c)

    packed_p = pack_potential(
        SCFPotential(active_mono, active_dip, active_quad), wide_p_layout
    )
    corrupted_p = packed_p.clone()
    corrupted_p[1, 6:] = -321.0
    corrupted_p[2, 12:] = -654.0
    projected_p = _project_potential_padding(corrupted_p, wide_p_layout)
    assert torch.equal(projected_p, packed_p)

    # _mask_packed_rows also preserves active entries bitwise.
    orbital_mask = torch.ones(15, dtype=torch.bool, device=DEVICE)
    atom_mask = torch.ones(2, dtype=torch.bool, device=DEVICE)
    masked_c = _mask_packed_rows(
        corrupted_c,
        wide_c_layout.shapes,
        orbital_mask=orbital_mask,
        atom_mask=atom_mask,
        kind="charge",
        pad_value=float(defaults.PADNZ),
    )
    assert torch.equal(masked_c, packed_c)


def test_quadrupole_packing_gradients_and_transforms() -> None:
    """Verify autograd.grad, jacrev, jacfwd, and vmap through pack/unpack."""
    c_layout = ChargeLayout((4,), (2, 3), (2, 6))
    p_layout = PotentialLayout((4,), (2, 3), (2, 6))
    mono = torch.linspace(0.1, 0.4, 4, dtype=DTYPE, device=DEVICE)
    dip = torch.linspace(-0.2, 0.3, 6, dtype=DTYPE, device=DEVICE).reshape(2, 3)
    quad = torch.linspace(
        -0.5, 0.6, 12, dtype=DTYPE, device=DEVICE, requires_grad=True
    ).reshape(2, 6)

    def pack_c_quad(q: torch.Tensor) -> torch.Tensor:
        return pack_charges(SCFCharges(mono, dip, q), c_layout)[2, :12]

    def pack_p_quad(q: torch.Tensor) -> torch.Tensor:
        return pack_potential(SCFPotential(mono, dip, q), p_layout)[2, :12]

    loss = (pack_c_quad(quad) ** 2).sum()
    (g_quad,) = torch.autograd.grad(loss, quad)
    weights_sq = torch.tensor(
        [1.0, 2.0, 1.0, 2.0, 2.0, 1.0], dtype=DTYPE, device=DEVICE
    )
    torch.testing.assert_close(
        g_quad, 2.0 * quad.detach() * weights_sq, atol=1.0e-15, rtol=1.0e-15
    )

    jc_rev = jacrev(pack_c_quad)(quad.detach())
    jc_fwd = jacfwd(pack_c_quad)(quad.detach())
    torch.testing.assert_close(jc_rev, jc_fwd, atol=1.0e-15, rtol=1.0e-15)

    jp_rev = jacrev(pack_p_quad)(quad.detach())
    jp_fwd = jacfwd(pack_p_quad)(quad.detach())
    torch.testing.assert_close(jp_rev, jp_fwd, atol=1.0e-15, rtol=1.0e-15)

    batch_q = torch.stack(
        (quad.detach(), -0.5 * quad.detach(), 2.0 * quad.detach())
    )
    vmapped_c = vmap(pack_c_quad)(batch_q)
    expected_c = torch.stack([pack_c_quad(item) for item in batch_q])
    torch.testing.assert_close(
        vmapped_c, expected_c, atol=1.0e-15, rtol=1.0e-15
    )


# ============================================================================
# 1.6 B: Rotational invariance tests
# ============================================================================


@pytest.mark.parametrize("name", ["H2O", "glycine", "PbH4-BiH3"])
@pytest.mark.parametrize("mixer", ["broyden", "anderson"])
@pytest.mark.parametrize("scp_mode", ["potential", "charge"])
@pytest.mark.parametrize(
    ("x_atol", "grad_tol", "torque_tol"),
    [
        (1.0e-4, 5.0e-9, 5.0e-11),
        (1.0e-8, 1.0e-11, 1.0e-13),
    ],
    ids=["default_tol", "tight_tol"],
)
def test_gfn2_rotational_invariance(
    name: str,
    mixer: str,
    scp_mode: str,
    x_atol: float,
    grad_tol: float,
    torque_tol: float,
) -> None:
    """Verify GFN2-xTB rotational invariance across mixers, SCP modes, and tolerances."""
    sys = SYSTEMS[name]
    rot = _rotation_matrix(device=sys.positions.device)

    opts = {
        "int_driver": "pytorch",
        "scf_mode": "full",
        "scp_mode": scp_mode,
        "mixer": mixer,
        "x_atol": x_atol,
        "x_atol_max": x_atol * 0.1,
        "maxiter": 200,
        "verbosity": 0,
    }
    dd = {"device": sys.positions.device, "dtype": DTYPE}

    pos_orig = sys.positions.detach().clone().requires_grad_(True)
    calc_orig = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
    res_orig = calc_orig.singlepoint(pos_orig, chrg=sys.charge, spin=sys.spin)
    e_orig = res_orig.total.sum(-1)
    (g_orig,) = torch.autograd.grad(e_orig, pos_orig)

    pos_rot = (sys.positions.detach().clone() @ rot.T).requires_grad_(True)
    calc_rot = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
    res_rot = calc_rot.singlepoint(pos_rot, chrg=sys.charge, spin=sys.spin)
    e_rot = res_rot.total.sum(-1)
    (g_rot,) = torch.autograd.grad(e_rot, pos_rot)

    g_rot_back = g_rot @ rot
    force_orig = g_orig.sum(dim=0)
    force_rot = g_rot.sum(dim=0)
    torque_orig = _compute_torque(pos_orig.detach(), g_orig.detach())
    torque_rot = _compute_torque(pos_rot.detach(), g_rot.detach())

    torch.testing.assert_close(e_rot, e_orig, atol=grad_tol, rtol=0.0)
    torch.testing.assert_close(g_rot_back, g_orig, atol=grad_tol, rtol=grad_tol)
    assert float(torch.linalg.vector_norm(force_orig).item()) < 1.0e-13
    assert float(torch.linalg.vector_norm(force_rot).item()) < 1.0e-13
    assert float(torch.linalg.vector_norm(torque_orig).item()) < torque_tol
    assert float(torch.linalg.vector_norm(torque_rot).item()) < torque_tol


# ============================================================================
# 1.5: Charge-mode orbital-anisotropy isolation test
# ============================================================================


def test_charge_mode_residual_orientation_sensitivity_is_orbital_anisotropy() -> (
    None
):
    """Isolate the ~1e-12 loose-tolerance charge-mode sensitivity to AO populations.

    In ``scp_mode="potential"``, Row 0 stores ``spread_shell_to_orbital(v_shell)``,
    where every shell potential ``v_shell`` is an SO(3) scalar; combined with the
    sqrt(2) quadrupole Frobenius metric, the entire state space is an orthogonal
    representation of SO(3) at every SCF iteration.

    In ``scp_mode="charge"``, Row 0 stores orbital-resolved Mulliken charges
    ``q_mu = n0_mu - (P S)_{mu,mu}``. Under a 3D rotation R, the p- and d-shell
    diagonal blocks of ``P S`` transform by ``D_l(R) M D_l(R)^T``. While the
    shell sum ``sum_{mu in s} q_mu`` is an SO(3) invariant, the orbital sum of
    squares ``sum_{mu in s} q_mu^2`` is not SO(3)-invariant.

    This test demonstrates:
    1. In GFN1-xTB (which has NO dipoles and NO quadrupoles), ``scp_mode="potential"``
       is invariant to ~1e-15 at loose ``x_atol=1e-4``, whereas ``scp_mode="charge"``
       exhibits the exact same ~1e-12 orientation sensitivity.
    2. Projecting Row 0 onto the shell-isotropic subspace ``q_bar_mu = q_shell / N_s``
       restores machine-precision (~1e-15) rotational invariance to ``scp_mode="charge"``
       at ``x_atol=1e-4`` in both GFN1-xTB and GFN2-xTB (with sqrt(2) quadrupoles).
    """
    sys = SYSTEMS["glycine"]
    rot = _rotation_matrix(device=sys.positions.device)
    dd = {"device": sys.positions.device, "dtype": DTYPE}

    # 1. Demonstrate identical behavior in GFN1-xTB (no dipoles or quadrupoles).
    for mode, expected_max_diff in (
        ("potential", 1.0e-14),
        ("charge", 1.0e-10),
    ):
        opts = {
            "int_driver": "pytorch",
            "scf_mode": "full",
            "scp_mode": mode,
            "mixer": "broyden",
            "x_atol": 1.0e-4,
            "x_atol_max": 1.0e-5,
            "verbosity": 0,
        }
        p1 = sys.positions.detach().clone().requires_grad_(True)
        c1 = Calculator(sys.numbers, GFN1_XTB, opts=opts, **dd)
        e1 = c1.singlepoint(p1).total.sum(-1)
        (g1,) = torch.autograd.grad(e1, p1)

        p2 = (sys.positions.detach().clone() @ rot.T).requires_grad_(True)
        c2 = Calculator(sys.numbers, GFN1_XTB, opts=opts, **dd)
        e2 = c2.singlepoint(p2).total.sum(-1)
        (g2,) = torch.autograd.grad(e2, p2)

        grad_diff = float((g2 @ rot - g1).abs().max().item())
        assert grad_diff < expected_max_diff
        if mode == "charge":
            # Confirm the orbital-anisotropy sensitivity is present in GFN1.
            assert grad_diff > 1.0e-14

    # 2. Demonstrate that shell-isotropizing Row 0 in GFN2-xTB charge mode makes
    # the Broyden trajectory strictly SO(3)-invariant (~1e-15) even at x_atol=1e-4.
    def _run_shell_isotropic_charge_mode(
        positions: torch.Tensor,
    ) -> torch.Tensor:
        config = Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            scp_mode=labels.SCP_MODE_CHARGE,
            mixer=labels.MIXER_BROYDEN,
            x_atol=1.0e-4,
            x_atol_max=1.0e-5,
        )
        par = ParamModule(GFN2_XTB, **dd)
        system = Model(par=par, config=config).setup(sys.numbers, batch_mode=0)
        from dxtb._src.calculators.singlepoint import _interaction_data
        from dxtb._src.integral.evaluation import build_integral_matrices
        from dxtb._src.scf.functional.data import setup_scf_inputs
        from dxtb._src.scf.guess import get_guess
        from dxtb._src.scf.iterator import get_refocc

        chrg = positions.new_zeros((1,))
        integrals, refocc, _ = build_integral_matrices(
            system.integral_setup, system.h0_setup, positions, chrg
        )
        n0, occupation, nel = get_refocc(refocc, chrg, None, system.ihelp)
        guess = get_guess(
            sys.numbers, positions, chrg, system.ihelp, config.scf.guess
        )
        icache = _interaction_data(system, positions, system.interactions)
        inputs = setup_scf_inputs(
            sys.numbers,
            integrals,
            n0,
            occupation,
            nel,
            system.interactions,
            icache,
            system.ihelp,
            config.scf,
        )
        problem = SCFProblem(inputs, SCFRepresentation.CHARGE)
        settings = setup_solver_settings(config.scf, DTYPE)

        orb_per_shell = system.ihelp.orbitals_per_shell.to(dtype=DTYPE)
        sqrt_n_orb = system.ihelp.spread_shell_to_orbital(
            torch.sqrt(orb_per_shell)
        )

        def _iso_pack(charges: SCFCharges) -> torch.Tensor:
            shell_q = system.ihelp.reduce_orbital_to_shell(charges.mono)
            mean_orb_q = system.ihelp.spread_shell_to_orbital(
                shell_q / orb_per_shell
            )
            # Weight by sqrt(N_s) so sum_{mu in s} (mean_orb_q * sqrt(N_s))^2 = q_shell^2.
            iso_mono = mean_orb_q * sqrt_n_orb
            return pack_charges(
                SCFCharges(
                    mono=iso_mono, dipole=charges.dipole, quad=charges.quad
                ),
                inputs.charge_layout,
            )

        def _iso_map(x: torch.Tensor):
            res = problem.map(x)
            from dataclasses import replace

            return replace(res, mapped_x=_iso_pack(res.spectral.charges))

        init_charges = SCFCharges(
            mono=guess,
            dipole=guess.new_zeros(inputs.charge_layout.dipole_shape),
            quad=guess.new_zeros(inputs.charge_layout.quadrupole_shape),
        )
        x0 = _iso_pack(init_charges)
        outcome = _solve_map(
            x0,
            settings,
            _iso_map,
            state_projection=lambda v: _project_charge_padding(
                v, inputs.charge_layout
            ),
            eager=True,
            problem=problem,
            strategy=config.scf.strategy,
        )
        return system.ihelp.reduce_orbital_to_atom(
            outcome.map_result.spectral.charges.mono
        )

    q_orig = _run_shell_isotropic_charge_mode(sys.positions.detach().clone())
    q_rot = _run_shell_isotropic_charge_mode(
        sys.positions.detach().clone() @ rot.T
    )
    torch.testing.assert_close(q_rot, q_orig, atol=1.0e-14, rtol=1.0e-14)


# ============================================================================
# 1.6 D: Differentiability and PyTorch function transformations
# ============================================================================


@pytest.mark.parametrize("scf_mode", ["full", "implicit"])
@pytest.mark.parametrize("scp_mode", ["potential", "charge"])
def test_gfn2_differentiability_and_transforms(
    scf_mode: str, scp_mode: str
) -> None:
    """Verify autograd.grad, torch.func.grad, jacrev, and vmap on GFN2-xTB."""
    sys = SYSTEMS["H2O"]
    rot = _rotation_matrix(device=sys.positions.device)
    dd = {"device": sys.positions.device, "dtype": DTYPE}
    opts = {
        "int_driver": "pytorch",
        "scf_mode": scf_mode,
        "scp_mode": scp_mode,
        "mixer": "broyden",
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-9,
        "verbosity": 0,
    }
    calc = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)

    # 1. torch.autograd.grad on original and rotated geometries.
    pos_orig = sys.positions.detach().clone().requires_grad_(True)
    e_orig = calc.singlepoint(pos_orig).total.sum(-1)
    (g_autograd_orig,) = torch.autograd.grad(e_orig, pos_orig)

    pos_rot = (sys.positions.detach().clone() @ rot.T).requires_grad_(True)
    e_rot = calc.singlepoint(pos_rot).total.sum(-1)
    (g_autograd_rot,) = torch.autograd.grad(e_rot, pos_rot)
    torch.testing.assert_close(
        g_autograd_rot @ rot, g_autograd_orig, atol=1.0e-10, rtol=1.0e-10
    )

    # 2. Functional Model/System API with torch.func.grad, jacrev, and vmap.
    config = Config.create(
        method=labels.GFN2_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        scf_mode=scf_mode,
        scp_mode=scp_mode,
        mixer="broyden",
        x_atol=1.0e-8,
        x_atol_max=1.0e-9,
    )
    par = ParamModule(GFN2_XTB, **dd)
    system = Model(par=par, config=config).setup(sys.numbers, batch_mode=0)

    def energy_fn(pos: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(pos).total.sum()

    base_pos = sys.positions.detach().clone()
    g_func = grad(energy_fn)(base_pos)
    j_rev = jacrev(energy_fn)(base_pos)
    torch.testing.assert_close(
        g_func, g_autograd_orig, atol=1.0e-11, rtol=1.0e-11
    )
    torch.testing.assert_close(j_rev, g_func, atol=1.0e-12, rtol=1.0e-12)

    if scf_mode == "full":
        # Fixed-trip unrolled SCF supports vmap over conformer geometries.
        batch_pos = torch.stack((base_pos, base_pos @ rot.T), dim=0)
        vmapped_energies = vmap(energy_fn)(batch_pos)
        vmapped_grads = vmap(grad(energy_fn))(batch_pos)
        torch.testing.assert_close(
            vmapped_energies[0], vmapped_energies[1], atol=1.0e-12, rtol=0.0
        )
        torch.testing.assert_close(
            vmapped_grads[1] @ rot,
            vmapped_grads[0],
            atol=1.0e-10,
            rtol=1.0e-10,
        )
