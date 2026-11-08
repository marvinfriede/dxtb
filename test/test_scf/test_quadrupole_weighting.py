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
  across Broyden and Anderson convergers and potential and charge SCF variables.
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
    SCFConvergence,
    SCFPotential,
    pack_charges,
    pack_potential,
    unpack_charges,
    unpack_potential,
)
from dxtb._src.scf.functional.problem import (
    SCFProblem,
    SCFVariable,
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
from ..utils import scf_strategy

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


def test_quadrupole_dual_pairing_and_derivatives() -> None:
    """Verify dual pairing <V_packed, q_packed> = <V_phys, q_phys> and its derivatives."""
    c_layout = ChargeLayout(
        mono_shape=(5,), dipole_shape=(3, 3), quadrupole_shape=(3, 6)
    )
    p_layout = PotentialLayout(
        mono_shape=(5,), dipole_shape=(3, 3), quadrupole_shape=(3, 6)
    )

    q_mono = torch.tensor(
        [0.18, -0.31, 0.09, -0.14, 0.22],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    q_dip = torch.tensor(
        [
            [0.24, -0.17, 0.11],
            [-0.19, 0.33, -0.28],
            [0.07, -0.12, 0.41],
        ],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    q_quad = torch.tensor(
        [
            [0.42, -0.19, -0.15, 0.31, -0.27, -0.27],
            [-0.11, 0.08, 0.29, -0.14, 0.22, -0.18],
            [0.05, -0.33, -0.02, 0.17, 0.12, -0.03],
        ],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    v_mono = torch.tensor(
        [-0.27, 0.44, -0.15, 0.36, -0.08],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    v_dip = torch.tensor(
        [
            [-0.35, 0.21, -0.16],
            [0.14, -0.29, 0.38],
            [-0.22, 0.18, -0.11],
        ],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    v_quad = torch.tensor(
        [
            [-0.21, 0.28, 0.14, -0.16, 0.38, 0.07],
            [0.19, -0.24, -0.09, 0.42, -0.18, -0.10],
            [-0.31, 0.14, 0.22, -0.26, -0.08, 0.09],
        ],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    def packed_pairing(
        qm: torch.Tensor,
        qd: torch.Tensor,
        qq: torch.Tensor,
        vm: torch.Tensor,
        vd: torch.Tensor,
        vq: torch.Tensor,
    ) -> torch.Tensor:
        pc = pack_charges(SCFCharges(mono=qm, dipole=qd, quad=qq), c_layout)
        pv = pack_potential(SCFPotential(mono=vm, dipole=vd, quad=vq), p_layout)
        # Sum over active columns of each row (excluding PADNZ tails).
        return (
            (pc[0, :5] * pv[0, :5]).sum()
            + (pc[1, :9] * pv[1, :9]).sum()
            + (pc[2, :18] * pv[2, :18]).sum()
        )

    def physical_pairing(
        qm: torch.Tensor,
        qd: torch.Tensor,
        qq: torch.Tensor,
        vm: torch.Tensor,
        vd: torch.Tensor,
        vq: torch.Tensor,
    ) -> torch.Tensor:
        return (qm * vm).sum() + (qd * vd).sum() + (qq * vq).sum()

    e_packed = packed_pairing(q_mono, q_dip, q_quad, v_mono, v_dip, v_quad)
    e_phys = physical_pairing(q_mono, q_dip, q_quad, v_mono, v_dip, v_quad)
    torch.testing.assert_close(e_packed, e_phys, atol=1.0e-15, rtol=1.0e-15)

    grads_packed = torch.autograd.grad(
        e_packed,
        (q_mono, q_dip, q_quad, v_mono, v_dip, v_quad),
        create_graph=True,
    )
    grads_phys = torch.autograd.grad(
        e_phys,
        (q_mono, q_dip, q_quad, v_mono, v_dip, v_quad),
        create_graph=True,
    )
    for gp, gr in zip(grads_packed, grads_phys):
        torch.testing.assert_close(gp, gr, atol=1.0e-15, rtol=1.0e-15)

    # Mixed second derivative d^2 <V, q> / (d q_quad d v_quad) must be identity.
    cross_hess = jacrev(
        grad(
            lambda qq, vq: packed_pairing(
                q_mono.detach(),
                q_dip.detach(),
                qq,
                v_mono.detach(),
                v_dip.detach(),
                vq,
            ),
            argnums=0,
        ),
        argnums=1,
    )(q_quad.detach(), v_quad.detach())
    torch.testing.assert_close(
        cross_hess.reshape(18, 18),
        torch.eye(18, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )


def test_quadrupole_45deg_z_rotation_l2_vs_max_abs() -> None:
    """Demonstrate that 45-deg z-rotation preserves Frobenius L2 but changes max_abs from 1 to sqrt(2)."""
    c_layout = ChargeLayout(
        mono_shape=(1,), dipole_shape=(1, 3), quadrupole_shape=(1, 6)
    )
    p_layout = PotentialLayout(
        mono_shape=(1,), dipole_shape=(1, 3), quadrupole_shape=(1, 6)
    )
    mono = torch.zeros(1, dtype=DTYPE, device=DEVICE)
    zero_dip = torch.zeros((1, 3), dtype=DTYPE, device=DEVICE)

    # Q = diag(1, -1, 0) -> [xx, xy, yy, xz, yz, zz] = [1, 0, -1, 0, 0, 0]
    q_0 = torch.tensor(
        [[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE
    )
    q_0_full = _packed_quad_to_full_3x3(q_0)

    # Rotate by 45 degrees (pi / 4) around the z-axis: Q' = R_z(pi/4) Q R_z(pi/4)^T
    angle = math.pi / 4.0
    c, s = math.cos(angle), math.sin(angle)
    rot_z45 = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    q_45_full = rot_z45 @ q_0_full[0] @ rot_z45.T
    q_45 = _full_3x3_to_packed_quad(q_45_full.unsqueeze(0))

    # In physical components, Q_45 has xy = 1 and all diagonal entries 0.
    torch.testing.assert_close(
        q_45,
        torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE
        ),
        atol=1.0e-15,
        rtol=1.0e-15,
    )

    packed_c_0 = pack_charges(
        SCFCharges(mono=mono, dipole=zero_dip, quad=q_0), c_layout
    )[2, :6]
    packed_c_45 = pack_charges(
        SCFCharges(mono=mono, dipole=zero_dip, quad=q_45), c_layout
    )[2, :6]

    sqrt2 = math.sqrt(2.0)
    # Euclidean (Frobenius) norm is sqrt(2) in both orientations.
    torch.testing.assert_close(
        torch.linalg.vector_norm(packed_c_0),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(packed_c_45),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    # Componentwise max_abs changes from 1.0 at 0 deg to sqrt(2) at 45 deg.
    torch.testing.assert_close(
        packed_c_0.abs().amax(),
        torch.tensor(1.0, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        packed_c_45.abs().amax(),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )

    # Potential quadrupoles (where off-diagonal physical components carry factor 2,
    # so V_45 has V_xy = 2.0) exhibit the exact same packed behavior.
    v_0 = torch.tensor(
        [[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE
    )
    v_45 = torch.tensor(
        [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE
    )
    packed_p_0 = pack_potential(
        SCFPotential(mono=mono, dipole=zero_dip, quad=v_0), p_layout
    )[2, :6]
    packed_p_45 = pack_potential(
        SCFPotential(mono=mono, dipole=zero_dip, quad=v_45), p_layout
    )[2, :6]
    torch.testing.assert_close(
        torch.linalg.vector_norm(packed_p_0),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(packed_p_45),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        packed_p_0.abs().amax(),
        torch.tensor(1.0, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        packed_p_45.abs().amax(),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )


def test_dipole_rotation_l2_vs_max_abs() -> None:
    """Demonstrate that dipole rotation preserves L2 norm but changes max_abs by up to sqrt(2) (2D) or sqrt(3) (3D)."""
    c_layout = ChargeLayout(mono_shape=(1,), dipole_shape=(1, 3))
    mono = torch.zeros(1, dtype=DTYPE, device=DEVICE)

    # 2D rotation by -45 deg around z: mu = (1, 1, 0) -> (sqrt(2), 0, 0)
    dip_0 = torch.tensor([[1.0, 1.0, 0.0]], dtype=DTYPE, device=DEVICE)
    angle = -math.pi / 4.0
    c, s = math.cos(angle), math.sin(angle)
    rot_z = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    dip_45 = dip_0 @ rot_z.T

    packed_0 = pack_charges(SCFCharges(mono=mono, dipole=dip_0), c_layout)[
        1, :3
    ]
    packed_45 = pack_charges(SCFCharges(mono=mono, dipole=dip_45), c_layout)[
        1, :3
    ]
    sqrt2 = math.sqrt(2.0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(packed_0),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(packed_45),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        packed_0.abs().amax(),
        torch.tensor(1.0, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    torch.testing.assert_close(
        packed_45.abs().amax(),
        torch.tensor(sqrt2, dtype=DTYPE, device=DEVICE),
        atol=1.0e-15,
        rtol=1.0e-15,
    )


def test_scf_convergence_stopping_predicate_near_max_abs_boundary() -> None:
    """Exercise SCFConvergence stopping predicate and H2O SCF trajectory near the x_atol_max boundary."""
    # 1. Direct stopping-predicate boundary test for quadrupole and dipole residuals.
    c_layout = ChargeLayout(
        mono_shape=(1,), dipole_shape=(1, 3), quadrupole_shape=(1, 6)
    )
    mono = torch.zeros(1, dtype=DTYPE, device=DEVICE)
    zero_dip = torch.zeros((1, 3), dtype=DTYPE, device=DEVICE)

    q_0 = torch.tensor(
        [[1.0e-5, 0.0, -1.0e-5, 0.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE
    )
    q_45 = torch.tensor(
        [[0.0, 1.0e-5, 0.0, 0.0, 0.0, 0.0]], dtype=DTYPE, device=DEVICE
    )
    delta_0 = pack_charges(
        SCFCharges(mono=mono, dipole=zero_dip, quad=q_0), c_layout
    )
    delta_45 = pack_charges(
        SCFCharges(mono=mono, dipole=zero_dip, quad=q_45), c_layout
    )
    # Zero out PADNZ tails as project_residual does.
    delta_0[:, 1:] = torch.where(
        delta_0[:, 1:] == defaults.PADNZ, 0.0, delta_0[:, 1:]
    )
    delta_45[:, 1:] = torch.where(
        delta_45[:, 1:] == defaults.PADNZ, 0.0, delta_45[:, 1:]
    )

    res_0 = SCFConvergence.residual(delta_0)
    res_45 = SCFConvergence.residual(delta_45)
    # Both orientations have identical L2 norm = sqrt(2) * 1e-5 < x_atol = 1e-4,
    # but max_abs is 1.0e-5 at 0 deg vs sqrt(2) * 1e-5 at 45 deg.
    criterion = SCFConvergence(x_atol=1.0e-4, x_atol_max=1.2e-5)
    assert bool(res_0.l2 < criterion.x_atol) and bool(
        res_45.l2 < criterion.x_atol
    )
    assert bool(criterion.converged(res_0)) is True
    assert bool(criterion.converged(res_45)) is False

    # 2. Real SCF trajectory boundary on H2O (potential mode, Broyden, x_atol=1e-4, x_atol_max=1e-5):
    # At step 13 (iterations=14), L2 is 1.7436e-5 (< 1e-4) in both orientations,
    # while the oxygen dipole potential residual has Euclidean norm 1.3003e-5:
    # - In the axis-aligned original frame, max_abs = 1.3003e-5 > 1e-5 -> continues to iteration 15.
    # - In the rotated frame, max_abs = 8.5076e-6 < 1e-5 -> stops at iteration 14.
    # Meanwhile, a multipole-block invariant maximum norm (scalar Row 0, Euclidean Row 1,
    # Frobenius Row 2) is identical to ~1e-16 across both orientations at every step.
    sys = SYSTEMS["H2O"]
    rot = _rotation_matrix(device=sys.positions.device)
    opts = {
        "int_driver": "pytorch",
        "differentiation": "unrolled",
        "strategy": scf_strategy("broyden", "potential"),
        "x_atol": 1.0e-4,
        "x_atol_max": 1.0e-5,
        "maxiter": 50,
        "verbosity": 0,
    }
    dd = {"device": sys.positions.device, "dtype": DTYPE}
    c_orig = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
    r_orig = c_orig.singlepoint(sys.positions, chrg=sys.charge, spin=sys.spin)
    c_rot = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
    r_rot = c_rot.singlepoint(
        sys.positions @ rot.T, chrg=sys.charge, spin=sys.spin
    )
    assert int(r_orig.iterations.item()) == 15
    assert int(r_rot.iterations.item()) == 14

    # When evaluated at a matched iteration budget (e.g. 14 iterations),
    # the unrotated and rotated energies agree to machine precision (~1e-15).
    opts_fixed14 = {
        **opts,
        "maxiter": 14,
        "x_atol": 1.0e-16,
        "x_atol_max": 1.0e-16,
    }
    cf_orig = Calculator(sys.numbers, GFN2_XTB, opts=opts_fixed14, **dd)
    rf_orig = cf_orig.singlepoint(sys.positions, chrg=sys.charge, spin=sys.spin)
    cf_rot = Calculator(sys.numbers, GFN2_XTB, opts=opts_fixed14, **dd)
    rf_rot = cf_rot.singlepoint(
        sys.positions @ rot.T, chrg=sys.charge, spin=sys.spin
    )
    torch.testing.assert_close(
        rf_rot.energy, rf_orig.energy, atol=1.0e-15, rtol=1.0e-15
    )


# ============================================================================
# 1.6 B: Rotational invariance tests
# ============================================================================


@pytest.mark.parametrize("name", ["H2O", "glycine", "PbH4-BiH3"])
@pytest.mark.parametrize("converger", ["broyden", "anderson"])
@pytest.mark.parametrize("variable", ["potential", "charge"])
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
    converger: str,
    variable: str,
    x_atol: float,
    grad_tol: float,
    torque_tol: float,
) -> None:
    """Verify GFN2-xTB rotational invariance across convergers, SCF variables, and tolerances."""
    sys = SYSTEMS[name]
    rot = _rotation_matrix(device=sys.positions.device)

    opts = {
        "int_driver": "pytorch",
        "differentiation": "unrolled",
        "strategy": scf_strategy(converger, variable),
        "x_atol": x_atol,
        "x_atol_max": x_atol * 0.1,
        "maxiter": 200,
        "verbosity": 0,
    }
    dd = {"device": sys.positions.device, "dtype": DTYPE}

    pos_orig = sys.positions.detach().clone().requires_grad_(True)
    calc_orig = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
    res_orig = calc_orig.singlepoint(pos_orig, chrg=sys.charge, spin=sys.spin)
    e_orig = res_orig.energy.sum(-1)
    (g_orig,) = torch.autograd.grad(e_orig, pos_orig)

    pos_rot = (sys.positions.detach().clone() @ rot.T).requires_grad_(True)
    calc_rot = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
    res_rot = calc_rot.singlepoint(pos_rot, chrg=sys.charge, spin=sys.spin)
    e_rot = res_rot.energy.sum(-1)
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

    With the potential SCF variable, Row 0 stores ``spread_shell_to_orbital(v_shell)``,
    where every shell potential ``v_shell`` is an SO(3) scalar; combined with the
    sqrt(2) quadrupole Frobenius metric, the entire state space is an orthogonal
    variable of SO(3) at every SCF iteration.

    With the charge SCF variable, Row 0 stores orbital-resolved Mulliken charges
    ``q_mu = n0_mu - (P S)_{mu,mu}``. Under a 3D rotation R, the p- and d-shell
    diagonal blocks of ``P S`` transform by ``D_l(R) M D_l(R)^T``. While the
    shell sum ``sum_{mu in s} q_mu`` is an SO(3) invariant, the orbital sum of
    squares ``sum_{mu in s} q_mu^2`` is not SO(3)-invariant.

    This test demonstrates:
    1. In GFN1-xTB (which has NO dipoles and NO quadrupoles), the potential
       variable is invariant to ~1e-15 at loose ``x_atol=1e-4``, whereas the
       charge variable exhibits the exact same ~1e-12 orientation sensitivity.
    2. Projecting Row 0 onto the shell-isotropic subspace ``q_bar_mu = q_shell / N_s``
       restores machine-precision (~1e-15) rotational invariance to the charge variable
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
            "differentiation": "unrolled",
            "strategy": scf_strategy("broyden", mode),
            "x_atol": 1.0e-4,
            "x_atol_max": 1.0e-5,
            "verbosity": 0,
        }
        p1 = sys.positions.detach().clone().requires_grad_(True)
        c1 = Calculator(sys.numbers, GFN1_XTB, opts=opts, **dd)
        e1 = c1.singlepoint(p1).energy.sum(-1)
        (g1,) = torch.autograd.grad(e1, p1)

        p2 = (sys.positions.detach().clone() @ rot.T).requires_grad_(True)
        c2 = Calculator(sys.numbers, GFN1_XTB, opts=opts, **dd)
        e2 = c2.singlepoint(p2).energy.sum(-1)
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
            strategy=scf_strategy("broyden", "charge"),
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
        problem = SCFProblem(inputs, SCFVariable.CHARGE)
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


def test_charge_mode_quadrupole_weighting_strictly_improves_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm sqrt(2) quadrupole weighting improves charge-mode invariance and never worsens the orbital-anisotropy baseline."""
    import dxtb._src.scf.functional.data as scf_data

    def _measure_rot_and_torque(name: str) -> tuple[float, float, float]:
        sys = SYSTEMS[name]
        rot = _rotation_matrix(device=sys.positions.device)
        dd = {"device": sys.positions.device, "dtype": DTYPE}
        opts = {
            "int_driver": "pytorch",
            "differentiation": "unrolled",
            "strategy": scf_strategy("broyden", "charge"),
            "x_atol": 1.0e-4,
            "x_atol_max": 1.0e-5,
            "verbosity": 0,
        }
        r1 = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd).singlepoint(
            sys.positions, chrg=sys.charge, spin=sys.spin
        )
        r2 = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd).singlepoint(
            sys.positions @ rot.T, chrg=sys.charge, spin=sys.spin
        )
        energy_diff = float((r2.energy - r1.energy).abs().max().item())

        p1 = sys.positions.detach().clone().requires_grad_(True)
        c1 = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
        e1 = c1.singlepoint(p1, chrg=sys.charge, spin=sys.spin).energy.sum(-1)
        (g1,) = torch.autograd.grad(e1, p1)

        p2 = (sys.positions.detach().clone() @ rot.T).requires_grad_(True)
        c2 = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)
        e2 = c2.singlepoint(p2, chrg=sys.charge, spin=sys.spin).energy.sum(-1)
        (g2,) = torch.autograd.grad(e2, p2)

        grad_diff = float((g2 @ rot - g1).abs().max().item())
        torque_norm = max(
            float(
                torch.linalg.vector_norm(
                    _compute_torque(p1.detach(), g1.detach())
                ).item()
            ),
            float(
                torch.linalg.vector_norm(
                    _compute_torque(p2.detach(), g2.detach())
                ).item()
            ),
        )
        return energy_diff, grad_diff, torque_norm

    w_gly_de, w_gly_g, w_gly_tau = _measure_rot_and_torque("glycine")
    w_pb_de, w_pb_g, w_pb_tau = _measure_rot_and_torque("PbH4-BiH3")

    # Temporarily disable off-diagonal sqrt(2) weighting to measure pre-WP1 behavior.
    monkeypatch.setattr(
        scf_data, "_QUAD_WEIGHT_FACTORS", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    )
    _, _, u_gly_tau = _measure_rot_and_torque("glycine")
    u_pb_de, u_pb_g, u_pb_tau = _measure_rot_and_torque("PbH4-BiH3")

    # On PbH4-BiH3, sqrt(2) quadrupole weighting eliminates the 1-iteration
    # stopping discrepancy and reduces energy/gradient/torque errors by >20x-1000x.
    assert w_pb_de < u_pb_de * 0.01
    assert w_pb_g < u_pb_g * 0.05
    assert w_pb_tau < u_pb_tau * 0.05

    # On glycine, sqrt(2) quadrupole weighting strictly reduces net torque while
    # keeping gradients within the ~1e-11 AO-population anisotropy regime.
    assert w_gly_tau < u_gly_tau
    assert w_gly_g < 5.0e-11
    assert w_gly_de < 1.0e-5


# ============================================================================
# 1.6 D: Differentiability and PyTorch function transformations
# ============================================================================


@pytest.mark.parametrize("differentiation", ["unrolled", "implicit"])
@pytest.mark.parametrize("variable", ["potential", "charge"])
def test_gfn2_differentiability_and_transforms(
    differentiation: str, variable: str
) -> None:
    """Verify autograd.grad, torch.func.grad, jacrev, and vmap on GFN2-xTB."""
    sys = SYSTEMS["H2O"]
    rot = _rotation_matrix(device=sys.positions.device)
    dd = {"device": sys.positions.device, "dtype": DTYPE}
    opts = {
        "int_driver": "pytorch",
        "differentiation": differentiation,
        "strategy": scf_strategy("broyden", variable),
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-9,
        "verbosity": 0,
    }
    calc = Calculator(sys.numbers, GFN2_XTB, opts=opts, **dd)

    # 1. torch.autograd.grad on original and rotated geometries.
    pos_orig = sys.positions.detach().clone().requires_grad_(True)
    e_orig = calc.singlepoint(pos_orig).energy.sum(-1)
    (g_autograd_orig,) = torch.autograd.grad(e_orig, pos_orig)

    pos_rot = (sys.positions.detach().clone() @ rot.T).requires_grad_(True)
    e_rot = calc.singlepoint(pos_rot).energy.sum(-1)
    (g_autograd_rot,) = torch.autograd.grad(e_rot, pos_rot)
    torch.testing.assert_close(
        g_autograd_rot @ rot, g_autograd_orig, atol=1.0e-10, rtol=1.0e-10
    )

    # 2. Functional Model/System API with torch.func.grad, jacrev, and vmap.
    config = Config.create(
        method=labels.GFN2_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        differentiation=differentiation,
        strategy=scf_strategy("broyden", variable),
        x_atol=1.0e-8,
        x_atol_max=1.0e-9,
    )
    par = ParamModule(GFN2_XTB, **dd)
    system = Model(par=par, config=config).setup(sys.numbers, batch_mode=0)

    def energy_fn(pos: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(pos).energy.sum()

    base_pos = sys.positions.detach().clone()
    g_func = grad(energy_fn)(base_pos)
    j_rev = jacrev(energy_fn)(base_pos)
    torch.testing.assert_close(
        g_func, g_autograd_orig, atol=1.0e-11, rtol=1.0e-11
    )
    torch.testing.assert_close(j_rev, g_func, atol=1.0e-12, rtol=1.0e-12)

    if differentiation == "unrolled":
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
