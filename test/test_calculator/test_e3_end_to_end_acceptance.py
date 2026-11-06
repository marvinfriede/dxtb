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
"""End-to-end E3 numerical and derivative acceptance suite (Milestone B)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.model import Model, ModelParameterNode
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
from dxtb._src.scf.functional.implicit import implicit_root
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb.config import Config
from dxtb.scf import (
    SCFDifferentiation,
    SCFRepresentation,
    SCFStage,
    SCFStrategy,
    strategy_preset,
)

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}


def _build_system(
    method: object,
    numbers: tuple[int, ...],
    *,
    strategy: str | None = None,
    representation: SCFRepresentation | None = None,
    maxiter: int = 3,
    temperature: float = 0.0,
    int_level: int = labels.INTLEVEL_QUADRUPOLE,
    tolerance: float = 0.0,
):
    """Build a pure functional Model and System for transform testing."""
    is_gfn2 = method is GFN2_XTB
    selected_strategy = strategy_preset(
        strategy or ("anderson" if is_gfn2 else "linear")
    )
    selected_representation = representation or (
        SCFRepresentation.FOCK if is_gfn2 else SCFRepresentation.CHARGE
    )
    selected_strategy = SCFStrategy(
        (
            SCFStage(
                selected_strategy.stages[0].method,
                representation=selected_representation,
            ),
        )
    )
    config = Config.create(
        method=labels.GFN2_XTB if is_gfn2 else labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=int_level,
        strategy=selected_strategy,
        differentiation="unrolled",
        maxiter=maxiter,
        fermi_etemp=temperature,
        x_atol=tolerance,
        x_atol_max=tolerance,
    )
    parameters = ParamModule(method, **_DD)
    model = Model(par=parameters, config=config)
    atomic_numbers = torch.tensor(numbers, dtype=torch.int64)
    return model, atomic_numbers, model.setup(atomic_numbers)


def _central_fd_4pt(fn, step: float = 1.0e-3) -> torch.Tensor:
    """Four-point central finite difference derivative at scalar 0.0."""
    zero = torch.zeros((), **_DD)
    vals = [fn(zero + mult * step) for mult in (-2.0, -1.0, 1.0, 2.0)]
    return (vals[0] - 8.0 * vals[1] + 8.0 * vals[2] - vals[3]) / (12.0 * step)


def test_scalar_and_generalized_overlap_matrix_fixed_point_references() -> None:
    """Scalar and generalized-overlap 2x2 matrix fixed points satisfy E3 transforms."""

    # 1. Scalar contraction x = tanh(0.4 * x + theta)
    def scalar_solve_primal(theta: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = torch.zeros_like(theta)
            for _ in range(150):
                x = torch.tanh(0.4 * x + theta)
        return x

    def scalar_objective(theta: torch.Tensor) -> torch.Tensor:
        root = scalar_solve_primal(theta)
        attached = implicit_root(
            root,
            theta,
            raw_map=lambda x, th: torch.tanh(0.4 * x + th),
        )
        return (attached**2 + 0.5 * theta * attached).sum()

    theta0 = torch.tensor(0.25, **_DD, requires_grad=True)
    val = scalar_objective(theta0)
    (grad_auto,) = torch.autograd.grad(val, theta0)
    grad_func = torch.func.grad(scalar_objective)(theta0.detach())
    jac_rev = torch.func.jacrev(scalar_objective)(theta0.detach())
    jac_fwd = torch.func.jacfwd(scalar_objective)(theta0.detach())
    _, jvp_1 = torch.func.jvp(
        scalar_objective, (theta0.detach(),), (torch.ones_like(theta0),)
    )
    hess_fwd_rev = torch.func.jacfwd(torch.func.grad(scalar_objective))(
        theta0.detach()
    )
    hess_rev_rev = torch.func.jacrev(torch.func.grad(scalar_objective))(
        theta0.detach()
    )
    _, jvp_2 = torch.func.jvp(
        torch.func.grad(scalar_objective),
        (theta0.detach(),),
        (torch.ones_like(theta0),),
    )
    _, jvp_3 = torch.func.jvp(
        lambda th: torch.func.jvp(
            torch.func.grad(scalar_objective), (th,), (torch.ones_like(th),)
        )[1],
        (theta0.detach(),),
        (torch.ones_like(theta0),),
    )

    # Analytical fixed point derivatives at root x*:
    # x* = tanh(0.4 x* + theta), sech^2(u) = 1 - x*^2
    x_star = scalar_solve_primal(theta0.detach())
    s2 = 1.0 - x_star**2
    denom = 1.0 - 0.4 * s2
    dx = s2 / denom
    # d/du(sech^2 u) = -2 x* s2, du = 0.4 dx + 1 = 1 / denom
    d2x = (-2.0 * x_star * s2) / (denom**3)
    # Objective f = x^2 + 0.5 * theta * x
    exact_1 = 2.0 * x_star * dx + 0.5 * x_star + 0.5 * theta0.detach() * dx
    exact_2 = 2.0 * (dx**2 + x_star * d2x) + dx + 0.5 * theta0.detach() * d2x
    fd_3 = _central_fd_4pt(
        lambda dt: torch.func.jacfwd(torch.func.grad(scalar_objective))(
            theta0.detach() + dt
        ),
        step=1.0e-3,
    )

    for candidate in (grad_auto, grad_func, jac_rev, jac_fwd, jvp_1):
        torch.testing.assert_close(
            candidate, exact_1, atol=1.0e-11, rtol=1.0e-11
        )
    for candidate in (hess_fwd_rev, hess_rev_rev, jvp_2):
        torch.testing.assert_close(
            candidate, exact_2, atol=1.0e-10, rtol=1.0e-10
        )
    torch.testing.assert_close(jvp_3, fd_3, atol=1.0e-8, rtol=1.0e-8)

    batch_thetas = torch.tensor([0.1, 0.25, -0.2], **_DD)
    vmapped_grad = torch.func.vmap(torch.func.grad(scalar_objective))(
        batch_thetas
    )
    vmapped_hess = torch.func.vmap(
        torch.func.jacfwd(torch.func.grad(scalar_objective))
    )(batch_thetas)
    for idx, th in enumerate(batch_thetas):
        torch.testing.assert_close(
            vmapped_grad[idx],
            torch.func.grad(scalar_objective)(th),
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        torch.testing.assert_close(
            vmapped_hess[idx],
            torch.func.jacfwd(torch.func.grad(scalar_objective))(th),
            atol=1.0e-10,
            rtol=1.0e-10,
        )

    # 2. 2x2 generalized-overlap (S != I) matrix fixed-point problem
    def matrix_step(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        s_off = 0.25 + 0.05 * torch.sin(theta[0])
        overlap = torch.stack(
            [
                torch.stack([theta.new_tensor(1.0), s_off]),
                torch.stack([s_off, theta.new_tensor(1.0)]),
            ]
        )
        h0 = torch.stack(
            [
                torch.stack([-0.6 + 0.2 * theta[0], -0.35 + 0.1 * theta[1]]),
                torch.stack([-0.35 + 0.1 * theta[1], 0.4 - 0.15 * theta[0]]),
            ]
        )
        fock = (
            h0
            + 0.3 * torch.diag(state) @ overlap
            + 0.3 * overlap @ torch.diag(state)
        )
        # Orthonormalize via S^{-1/2}
        evals_s, evecs_s = torch.linalg.eigh(overlap)
        sinv_sqrt = (evecs_s * evals_s.rsqrt().unsqueeze(-2)) @ evecs_s.T
        f_orth = sinv_sqrt @ fock @ sinv_sqrt
        # Smooth Fermi projector at finite temperature around mu = 0.0
        evals, evecs = torch.linalg.eigh(f_orth)
        occ = 2.0 * torch.sigmoid(-evals / 0.2)
        p_orth = (evecs * occ.unsqueeze(-2)) @ evecs.T
        p_ao = sinv_sqrt @ p_orth @ sinv_sqrt
        mulliken = torch.diagonal(p_ao @ overlap)
        return 0.5 * state + 0.5 * mulliken

    def matrix_solve_primal(theta: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = theta.new_tensor([1.0, 1.0])
            for _ in range(200):
                x = matrix_step(x, theta)
        return x

    def matrix_energy(theta: torch.Tensor) -> torch.Tensor:
        root = matrix_solve_primal(theta)
        attached = implicit_root(root, theta, raw_map=matrix_step)
        return (
            attached[0] ** 2
            - 0.7 * attached[1] ** 2
            + 0.2 * (attached * theta).sum()
        )

    th_vec = torch.tensor([0.15, -0.2], **_DD)
    direction = torch.tensor([0.6, -0.8], **_DD)
    g_rev = torch.func.jacrev(matrix_energy)(th_vec)
    g_fwd = torch.func.jacfwd(matrix_energy)(th_vec)
    _, d1 = torch.func.jvp(matrix_energy, (th_vec,), (direction,))
    h_mat = torch.func.jacfwd(torch.func.grad(matrix_energy))(th_vec)
    _, d2 = torch.func.jvp(
        torch.func.grad(matrix_energy), (th_vec,), (direction,)
    )
    _, d3 = torch.func.jvp(
        lambda v: torch.func.jvp(
            torch.func.grad(matrix_energy), (v,), (direction,)
        )[1],
        (th_vec,),
        (direction,),
    )
    fd_d3 = _central_fd_4pt(
        lambda t: torch.func.jvp(
            torch.func.grad(matrix_energy),
            (th_vec + t * direction,),
            (direction,),
        )[1],
        step=1.0e-3,
    )
    torch.testing.assert_close(g_rev, g_fwd, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(
        d1, (g_rev * direction).sum(), atol=1.0e-11, rtol=1.0e-11
    )
    torch.testing.assert_close(
        d2, h_mat @ direction, atol=1.0e-10, rtol=1.0e-10
    )
    torch.testing.assert_close(d3, fd_d3, atol=1.0e-8, rtol=1.0e-8)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(GFN1_XTB, id="gfn1"),
        pytest.param(GFN2_XTB, id="gfn2", marks=pytest.mark.slow),
    ],
)
def test_charged_and_openshell_systems_transform_matrix(method: object) -> None:
    """Charged (OH-) and open-shell (OH radical) systems pass 1st-3rd order and vmap."""
    # 1. Charged closed-shell OH- (chrg = -1.0)
    _, _, sys_charged = _build_system(
        method, (8, 1), maxiter=2, temperature=0.0
    )
    pos_batch = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.82]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.88]],
        ],
        **_DD,
    )
    pos0 = pos_batch[0]
    tangent = torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], **_DD)

    res0 = sys_charged.singlepoint(pos0, chrg=-1.0)
    # Generalized overlap S != I check
    eye = torch.eye(res0.overlap.shape[-1], **_DD)
    assert not torch.allclose(res0.overlap, eye)
    torch.testing.assert_close(
        res0.charges.mono.sum(),
        torch.tensor(-1.0, **_DD),
        atol=1.0e-11,
        rtol=1.0e-11,
    )

    def charged_energy(p: torch.Tensor) -> torch.Tensor:
        return sys_charged.singlepoint(p, chrg=-1.0).energy.sum()

    p_req = pos0.clone().requires_grad_(True)
    (g_auto,) = torch.autograd.grad(charged_energy(p_req), p_req)
    g_func = torch.func.grad(charged_energy)(pos0)
    j_rev = torch.func.jacrev(charged_energy)(pos0)
    j_fwd = torch.func.jacfwd(charged_energy)(pos0)
    _, d1 = torch.func.jvp(charged_energy, (pos0,), (tangent,))
    _, d2 = torch.func.jvp(torch.func.grad(charged_energy), (pos0,), (tangent,))
    _, d3 = torch.func.jvp(
        lambda p: torch.func.jvp(
            torch.func.grad(charged_energy), (p,), (tangent,)
        )[1],
        (pos0,),
        (tangent,),
    )
    fd_d3 = _central_fd_4pt(
        lambda t: torch.func.jvp(
            torch.func.grad(charged_energy), (pos0 + t * tangent,), (tangent,)
        )[1],
        step=1.0e-3,
    )
    torch.testing.assert_close(g_auto, g_func, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(j_rev, j_fwd, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(
        d1, (g_func * tangent).sum(), atol=1.0e-11, rtol=1.0e-11
    )
    torch.testing.assert_close(d3, fd_d3, atol=1.0e-8, rtol=1.0e-8)

    vmap_g = torch.func.vmap(torch.func.grad(charged_energy))(pos_batch)
    vmap_h_dir = torch.func.vmap(
        lambda p: torch.func.jvp(
            torch.func.grad(charged_energy), (p,), (tangent,)
        )[1]
    )(pos_batch)
    torch.testing.assert_close(vmap_g[0], g_func, atol=1.0e-12, rtol=1.0e-12)
    torch.testing.assert_close(vmap_h_dir[0], d2, atol=1.0e-11, rtol=1.0e-11)

    # 2. Open-shell OH radical (spin = 1.0, finite temperature = 300 K)
    _, _, sys_open = _build_system(method, (8, 1), maxiter=2, temperature=300.0)
    res_open = sys_open.singlepoint(pos0, chrg=0.0, spin=1.0)
    # 7 valence electrons -> fractional / open-shell occupation
    torch.testing.assert_close(
        res_open.occupation.sum(),
        torch.tensor(7.0, **_DD),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert bool(
        ((res_open.occupation > 0.1) & (res_open.occupation < 1.9)).any()
    )

    def openshell_energy(p: torch.Tensor) -> torch.Tensor:
        out = sys_open.singlepoint(p, chrg=0.0, spin=1.0)
        return out.energy.sum()

    g_open_rev = torch.func.jacrev(openshell_energy)(pos0)
    g_open_fwd = torch.func.jacfwd(openshell_energy)(pos0)
    _, d2_open = torch.func.jvp(
        torch.func.grad(openshell_energy), (pos0,), (tangent,)
    )
    _, d3_open = torch.func.jvp(
        lambda p: torch.func.jvp(
            torch.func.grad(openshell_energy), (p,), (tangent,)
        )[1],
        (pos0,),
        (tangent,),
    )
    fd_d3_open = _central_fd_4pt(
        lambda t: torch.func.jvp(
            torch.func.grad(openshell_energy),
            (pos0 + t * tangent,),
            (tangent,),
        )[1],
        step=1.0e-3,
    )
    torch.testing.assert_close(
        g_open_rev, g_open_fwd, atol=1.0e-11, rtol=1.0e-11
    )
    assert bool(torch.isfinite(d2_open).all())
    torch.testing.assert_close(d3_open, fd_d3_open, atol=1.0e-8, rtol=1.0e-8)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(GFN1_XTB, id="gfn1"),
        pytest.param(GFN2_XTB, id="gfn2", marks=pytest.mark.slow),
    ],
)
def test_exact_and_near_degenerate_h3_plus_derivatives(method: object) -> None:
    """Equilateral D3h H3+ (exact and 1e-7 near-degenerate) has smooth 1st-3rd derivatives."""
    _, _, system = _build_system(
        method, (1, 1, 1), maxiter=2, temperature=300.0
    )
    bond = 1.65
    h_tri = bond * (3.0**0.5) / 2.0
    exact_d3h = torch.tensor(
        [
            [-0.5 * bond, -h_tri / 3.0, 0.0],
            [0.5 * bond, -h_tri / 3.0, 0.0],
            [0.0, 2.0 * h_tri / 3.0, 0.0],
        ],
        **_DD,
    )
    near_d3h = exact_d3h + torch.tensor(
        [[0.0, 0.0, 0.0], [1.0e-7, 0.0, 0.0], [0.0, -1.0e-7, 0.0]], **_DD
    )
    tangent = torch.tensor(
        [[-0.3, 0.2, 0.0], [0.4, -0.1, 0.0], [-0.1, -0.1, 0.0]], **_DD
    )
    tangent = tangent / torch.linalg.vector_norm(tangent)

    res_exact = system.singlepoint(exact_d3h, chrg=1.0)
    res_near = system.singlepoint(near_d3h, chrg=1.0)
    # E' symmetry orbitals are degenerate in exact D3h and split in perturbed D3h
    exact_min_gap = float(torch.diff(res_exact.emo).abs().min())
    near_min_gap = float(torch.diff(res_near.emo).abs().min())
    assert exact_min_gap < 1.0e-12
    assert 0.0 < near_min_gap < 1.0e-5

    def free_energy(p: torch.Tensor) -> torch.Tensor:
        out = system.singlepoint(p, chrg=1.0)
        return out.energy.sum()

    for geom in (exact_d3h, near_d3h):
        g_rev = torch.func.jacrev(free_energy)(geom)
        g_fwd = torch.func.jacfwd(free_energy)(geom)
        _, d1 = torch.func.jvp(free_energy, (geom,), (tangent,))
        _, d2 = torch.func.jvp(
            torch.func.grad(free_energy), (geom,), (tangent,)
        )
        _, d3 = torch.func.jvp(
            lambda p: torch.func.jvp(
                torch.func.grad(free_energy), (p,), (tangent,)
            )[1],
            (geom,),
            (tangent,),
        )
        fd_d3 = _central_fd_4pt(
            lambda t: torch.func.jvp(
                torch.func.grad(free_energy),
                (geom + t * tangent,),
                (tangent,),
            )[1],
            step=1.0e-3,
        )
        assert bool(torch.isfinite(g_rev).all())
        torch.testing.assert_close(g_rev, g_fwd, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(
            d1, (g_rev * tangent).sum(), atol=1.0e-10, rtol=1.0e-10
        )
        assert bool(torch.isfinite(d2).all())
        torch.testing.assert_close(d3, fd_d3, atol=1.0e-7, rtol=1.0e-7)

    # Continuity across exact vs 1e-7 near-degeneracy
    torch.testing.assert_close(
        torch.func.grad(free_energy)(exact_d3h),
        torch.func.grad(free_energy)(near_d3h),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_field_fieldgrad_parameter_and_mixed_derivatives() -> None:
    """Position, electric field, field gradient, and Model parameters compose through 3rd order."""
    model, numbers, sys_base = _build_system(
        GFN1_XTB,
        (8, 1, 1),
        maxiter=2,
        int_level=labels.INTLEVEL_QUADRUPOLE,
    )
    pos0 = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]], **_DD
    )
    field0 = torch.tensor([0.002, -0.001, 0.003], **_DD)
    # Traceless symmetric electric field gradient (3, 3)
    field_grad0 = torch.tensor(
        [
            [0.001, 0.0004, -0.0002],
            [0.0004, -0.0006, 0.0003],
            [-0.0002, 0.0003, -0.0004],
        ],
        **_DD,
    )
    base_tree = model.parameter_tree
    param_key = "parameter_tree.hamiltonian.xtb.shell.ss.param"
    base_k_sh = base_tree.parameters.params[param_key]

    def fast_energy_fn(
        pos: torch.Tensor,
        efield: torch.Tensor,
        efgrad: torch.Tensor,
    ) -> torch.Tensor:
        return sys_base.singlepoint(
            pos, field=efield, field_grad=efgrad
        ).energy.sum()

    def param_energy_fn(
        pos: torch.Tensor,
        efield: torch.Tensor,
        k_sh_param: torch.Tensor,
    ) -> torch.Tensor:
        new_params = dict(base_tree.parameters.params)
        new_params[param_key] = k_sh_param
        tree = ModelParameterNode(
            parameters=replace(base_tree.parameters, params=new_params),
            config=base_tree.config,
            auto_int_level=base_tree.auto_int_level,
        )
        sys = model.setup_functional(numbers, tree)
        res = sys.singlepoint(pos, field=efield, field_grad=field_grad0)
        return res.energy.sum()

    # 1st-order gradients w.r.t. (positions, field, field_grad) and model parameter
    g_rev = torch.func.jacrev(fast_energy_fn, argnums=(0, 1, 2))(
        pos0, field0, field_grad0
    )
    g_fwd = torch.func.jacfwd(fast_energy_fn, argnums=(0, 1, 2))(
        pos0, field0, field_grad0
    )
    for gr, gf in zip(g_rev, g_fwd):
        assert gr.abs().max() > 0.0
        torch.testing.assert_close(gr, gf, atol=1.0e-10, rtol=1.0e-10)

    g_param_rev = torch.func.jacrev(param_energy_fn, argnums=2)(
        pos0, field0, base_k_sh
    )
    g_param_fwd = torch.func.jacfwd(param_energy_fn, argnums=2)(
        pos0, field0, base_k_sh
    )
    assert g_param_rev.abs().max() > 0.0
    torch.testing.assert_close(
        g_param_rev, g_param_fwd, atol=1.0e-10, rtol=1.0e-10
    )

    # 2nd-order mixed derivatives: d^2E / dR dF, d^2E / dR dG, d^2E / dR dtheta
    d2_r_f = torch.func.jacfwd(
        torch.func.grad(fast_energy_fn, argnums=0), argnums=1
    )(pos0, field0, field_grad0)
    d2_f_r = torch.func.jacfwd(
        torch.func.grad(fast_energy_fn, argnums=1), argnums=0
    )(pos0, field0, field_grad0)
    torch.testing.assert_close(
        d2_r_f.reshape(9, 3), d2_f_r.reshape(3, 9).T, atol=1.0e-10, rtol=1.0e-10
    )

    d2_r_g = torch.func.jacfwd(
        torch.func.grad(fast_energy_fn, argnums=0), argnums=2
    )(pos0, field0, field_grad0)
    assert d2_r_g.shape == (3, 3, 3, 3)
    assert bool(torch.isfinite(d2_r_g).all()) and d2_r_g.abs().max() > 0.0

    d2_r_theta = torch.func.jacfwd(
        torch.func.grad(param_energy_fn, argnums=0), argnums=2
    )(pos0, field0, base_k_sh)
    assert d2_r_theta.shape == (*pos0.shape, *base_k_sh.shape)
    assert (
        bool(torch.isfinite(d2_r_theta).all()) and d2_r_theta.abs().max() > 0.0
    )

    # 3rd-order mixed directional derivative: d^3E / dR dF dtheta
    v_pos = torch.linspace(-0.2, 0.2, pos0.numel(), **_DD).reshape_as(pos0)
    v_field = torch.tensor([0.5, -0.3, 0.4], **_DD)
    v_theta = torch.tensor(0.25, **_DD)

    def grad_pos_dot(
        pos: torch.Tensor, efield: torch.Tensor, k_sh_param: torch.Tensor
    ) -> torch.Tensor:
        g_pos = torch.func.grad(param_energy_fn, argnums=0)(
            pos, efield, k_sh_param
        )
        return (g_pos * v_pos).sum()

    def mixed_2nd(
        efield: torch.Tensor, k_sh_param: torch.Tensor
    ) -> torch.Tensor:
        return torch.func.jvp(
            lambda f: grad_pos_dot(pos0, f, k_sh_param), (efield,), (v_field,)
        )[1]

    _, mixed_3rd = torch.func.jvp(
        lambda th: mixed_2nd(field0, th), (base_k_sh,), (v_theta,)
    )
    fd_mixed_3rd = _central_fd_4pt(
        lambda dt: mixed_2nd(field0, base_k_sh + dt * v_theta), step=1.0e-3
    )
    torch.testing.assert_close(
        mixed_3rd, fd_mixed_3rd, atol=1.0e-8, rtol=1.0e-8
    )


def test_unrolled_trajectory_vs_converged_implicit_root_derivatives() -> None:
    """Unrolled trajectory derivatives differ at low maxiter and match implicit root at convergence."""
    _, numbers, sys_low = _build_system(
        GFN1_XTB,
        (8, 1, 1),
        strategy="linear",
        representation=SCFRepresentation.CHARGE,
        maxiter=1,
        tolerance=0.0,
    )
    _, _, sys_conv = _build_system(
        GFN1_XTB,
        (8, 1, 1),
        strategy="anderson",
        representation=SCFRepresentation.CHARGE,
        maxiter=16,
        tolerance=1.0e-12,
    )
    pos0 = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]], **_DD
    )
    tangent = torch.tensor(
        [[0.0, 0.0, 0.4], [-0.3, 0.0, -0.2], [0.3, 0.0, -0.2]], **_DD
    )
    tangent = tangent / torch.linalg.vector_norm(tangent)

    def unrolled_low_energy(p: torch.Tensor) -> torch.Tensor:
        return sys_low.singlepoint(p).scf.sum()

    def unrolled_conv_energy(p: torch.Tensor) -> torch.Tensor:
        out = sys_conv.singlepoint(p)
        assert bool(out.converged)
        return out.scf.sum()

    def implicit_conv_energy(p: torch.Tensor) -> torch.Tensor:
        chrg = p.new_zeros((1,))
        spin = p.new_zeros((1,))
        integrals, refocc, _ = build_integral_matrices(
            sys_conv.integral_setup, sys_conv.h0_setup, p, chrg
        )
        n0, occupation, nel = get_refocc(refocc, chrg, spin, sys_conv.ihelp)
        interaction_data = _interaction_data(sys_conv, p, sys_conv.interactions)
        initial_charges = get_guess(
            numbers, p, chrg, sys_conv.ihelp, sys_conv.config.scf.guess
        )
        implicit_cfg = replace(
            sys_conv.config.scf,
            differentiation=SCFDifferentiation.IMPLICIT,
            maxiter=50,
            x_atol=1.0e-12,
            x_atol_max=1.0e-12,
        )
        solved = run_prepared_functional_implicit(
            numbers,
            initial_charges,
            n0,
            occupation,
            nel,
            sys_conv.interactions,
            interaction_data,
            sys_conv.ihelp,
            implicit_cfg,
            integrals,
        )
        assert bool(solved.converged)
        return solved.energy.sum()

    g_low = torch.func.grad(unrolled_low_energy)(pos0)
    g_conv = torch.func.grad(unrolled_conv_energy)(pos0)
    p_impl = pos0.clone().requires_grad_(True)
    e_impl = implicit_conv_energy(p_impl)
    (g_impl,) = torch.autograd.grad(e_impl, p_impl, create_graph=True)

    # At maxiter=1, the finite-trajectory derivative is genuinely different from the converged root
    assert float(torch.linalg.vector_norm(g_low - g_impl.detach())) > 1.0e-4
    assert float(
        torch.linalg.vector_norm(g_low - g_impl.detach())
    ) > 1.0e5 * float(torch.linalg.vector_norm(g_conv - g_impl.detach()))

    # At convergence, unrolled trajectory and implicit root agree closely with each other and FD
    torch.testing.assert_close(
        g_conv, g_impl.detach(), atol=1.0e-9, rtol=1.0e-9
    )

    _, h_conv_vec = torch.func.jvp(
        torch.func.grad(unrolled_conv_energy), (pos0,), (tangent,)
    )
    h_conv = (h_conv_vec * tangent).sum()
    (h_impl_vec,) = torch.autograd.grad((g_impl * tangent).sum(), p_impl)
    h_impl = (h_impl_vec.detach() * tangent).sum()
    fd_h = _central_fd_4pt(
        lambda dt: (
            torch.func.grad(unrolled_conv_energy)(pos0 + dt * tangent) * tangent
        ).sum(),
        step=1.0e-4,
    )
    torch.testing.assert_close(h_conv, h_impl, atol=1.0e-7, rtol=1.0e-7)
    torch.testing.assert_close(h_impl, fd_h, atol=1.0e-7, rtol=1.0e-7)
