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
"""Acceptance tests for batching coexistence, force AD, and Calculator lifecycle."""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, jacrev, vmap

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.model import Model, System
from dxtb._src.calculators.packed_result import (
    build_calculator_lane_systems,
    evaluate_packed_systems,
)
from dxtb._src.components.classicals import new_dispersion, new_repulsion
from dxtb._src.components.interactions.coulomb import new_es2
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ..conftest import DEVICE

DTYPE = torch.float64
DD = {"device": DEVICE, "dtype": DTYPE}


def _water_geometry(offset: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return H2O atomic numbers and geometry with an optional O-H stretch."""
    numbers = torch.tensor([8, 1, 1], device=DEVICE, dtype=torch.int64)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.42 + offset, 1.04],
            [0.0, -1.42, 1.04 - 0.5 * offset],
        ],
        **DD,
    )
    return numbers, positions


def _lih_geometry(offset: float = 0.0) -> tuple[Tensor, Tensor]:
    """Return LiH atomic numbers and geometry."""
    numbers = torch.tensor([3, 1], device=DEVICE, dtype=torch.int64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0 + offset]],
        **DD,
    )
    return numbers, positions


@pytest.mark.parametrize(
    ("method_name", "param"),
    [
        ("gfn0-xtb", GFN0_XTB),
        ("gfn1-xtb", GFN1_XTB),
        ("gfn2-xtb", GFN2_XTB),
    ],
)
def test_three_batching_interfaces_coexist_and_agree(
    method_name: str, param: object
) -> None:
    """Single System, batched Calculator (modes 1 & 2), and vmap agree on energy and forces."""
    opts = {
        "method": method_name,
        "verbosity": 0,
        "maxiter": 100,
        "x_atol": 1e-10,
        "fermi_etemp": 0.0,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": labels.INTLEVEL_QUADRUPOLE,
    }
    cfg_opts = {k: v for k, v in opts.items() if k != "verbosity"}
    par = ParamModule(param, **DD)
    model = Model(par=par, config=Config.create(**cfg_opts))

    num_w, pos_w1 = _water_geometry(0.0)
    _, pos_w2 = _water_geometry(0.08)
    num_lih, pos_lih = _lih_geometry(0.0)

    # (A) Single System evaluations
    sys_w = model.setup(num_w, dd=DD)
    sys_lih = model.setup(num_lih, dd=DD)
    res_w1 = sys_w.singlepoint(pos_w1)
    res_w2 = sys_w.singlepoint(pos_w2)
    res_lih = sys_lih.singlepoint(pos_lih)
    f_w1 = sys_w.forces(pos_w1)
    f_w2 = sys_w.forces(pos_w2)
    f_lih = sys_lih.forces(pos_lih)

    # (C) Functional torch.vmap over conformer geometries
    conformer_pos = torch.stack([pos_w1, pos_w2], dim=0)

    def singlepoint_of_pos(p: Tensor):
        return sys_w.singlepoint(p)

    def energy_of_pos(p: Tensor) -> Tensor:
        return sys_w.energy(p)

    def forces_of_pos(p: Tensor) -> Tensor:
        return sys_w.forces(p)

    vmap_res = vmap(singlepoint_of_pos)(conformer_pos)
    vmap_e = vmap(energy_of_pos)(conformer_pos)
    vmap_f = vmap(forces_of_pos)(conformer_pos)

    torch.testing.assert_close(
        vmap_res.energy,
        torch.stack([res_w1.energy, res_w2.energy]),
        atol=1e-9,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        vmap_e,
        torch.stack([res_w1.energy.sum(-1), res_w2.energy.sum(-1)]),
        atol=1e-9,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        vmap_f,
        torch.stack([f_w1, f_w2]),
        atol=1e-7,
        rtol=1e-7,
    )

    # (B.1) Explicit conformer batch (batch_mode=2)
    calc_mode2 = Calculator(
        torch.stack([num_w, num_w], dim=0),
        par,
        opts={**opts, "batch_mode": 2},
        **DD,
    )
    pos_m2 = conformer_pos.clone().requires_grad_(True)
    e_m2 = calc_mode2.energy(pos_m2)
    f_m2 = calc_mode2.forces(pos_m2)
    assert pos_m2.grad is None
    torch.testing.assert_close(e_m2, vmap_e, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(f_m2, vmap_f, atol=1e-7, rtol=1e-7)

    # (B.2) Explicit padded heterogeneous batch (batch_mode=1) including noncontiguous padding
    padded_nums = torch.tensor(
        [
            [8, 1, 1, 0],
            [3, 0, 1, 0],
            [0, 8, 1, 1],
        ],
        device=DEVICE,
        dtype=torch.int64,
    )
    z = torch.zeros((1, 3), **DD)
    padded_pos = torch.stack(
        [
            torch.cat([pos_w1, z], dim=0),
            torch.cat([pos_lih[:1], z, pos_lih[1:], z], dim=0),
            torch.cat([z, pos_w2], dim=0),
        ],
        dim=0,
    )
    calc_mode1 = Calculator(
        padded_nums,
        par,
        opts={**opts, "batch_mode": 1},
        **DD,
    )
    pos_m1 = padded_pos.clone().requires_grad_(True)
    sp_m1 = calc_mode1.singlepoint(pos_m1)
    f_m1 = calc_mode1.forces(pos_m1)
    assert pos_m1.grad is None

    expected_e = torch.stack(
        [res_w1.energy.sum(-1), res_lih.energy.sum(-1), res_w2.energy.sum(-1)]
    )
    torch.testing.assert_close(
        sp_m1.total.sum(-1), expected_e, atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(f_m1[0, :3], f_w1, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        f_m1[1, [0, 2]], f_lih, atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(f_m1[2, 1:], f_w2, atol=1e-7, rtol=1e-7)
    # Inactive padded atom rows are strictly zero
    assert torch.all(f_m1[0, 3] == 0.0)
    assert torch.all(f_m1[1, 1] == 0.0)
    assert torch.all(f_m1[1, 3] == 0.0)
    assert torch.all(f_m1[2, 0] == 0.0)


def test_batched_shared_and_per_lane_field_and_field_grad_semantics() -> None:
    """Batched Calculator and evaluate_packed_systems support shared and per-lane fields."""
    opts = {
        "verbosity": 0,
        "maxiter": 100,
        "x_atol": 1e-10,
        "int_level": labels.INTLEVEL_QUADRUPOLE,
    }
    num_w, pos_w = _water_geometry(0.0)
    num_lih, pos_lih = _lih_geometry(0.0)
    padded_nums = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    padded_pos = torch.stack(
        [pos_w, torch.cat([pos_lih, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )
    calc = Calculator(
        padded_nums, GFN2_XTB, opts={**opts, "batch_mode": 1}, **DD
    )
    sys_w = Calculator(num_w, GFN2_XTB, opts=opts, **DD).system
    sys_lih = Calculator(num_lih, GFN2_XTB, opts=opts, **DD).system

    # 1. Shared field (3,) and field_grad (3, 3)
    shared_f = torch.tensor([0.01, -0.02, 0.015], **DD, requires_grad=True)
    shared_fg = (torch.eye(3, **DD) * 0.005).requires_grad_(True)
    e_shared = calc.energy(padded_pos, field=shared_f, field_grad=shared_fg)
    e_w_sh = sys_w.energy(pos_w, field=shared_f, field_grad=shared_fg)
    e_lih_sh = sys_lih.energy(pos_lih, field=shared_f, field_grad=shared_fg)
    torch.testing.assert_close(
        e_shared, torch.stack([e_w_sh, e_lih_sh]), atol=1e-9, rtol=1e-9
    )

    # 2. Per-lane field (B, 3) and field_grad (B, 3, 3)
    f0 = torch.tensor([0.01, -0.02, 0.015], **DD)
    f1 = torch.tensor([-0.015, 0.01, -0.005], **DD)
    fg0 = torch.tensor(
        [[0.01, 0.0, 0.005], [0.0, -0.005, 0.0], [0.005, 0.0, -0.005]], **DD
    )
    fg1 = -0.5 * fg0
    batch_f = torch.stack([f0, f1], dim=0).requires_grad_(True)
    batch_fg = torch.stack([fg0, fg1], dim=0).requires_grad_(True)

    e_perlane = calc.energy(padded_pos, field=batch_f, field_grad=batch_fg)
    e0_ref = sys_w.energy(pos_w, field=batch_f[0], field_grad=batch_fg[0])
    e1_ref = sys_lih.energy(pos_lih, field=batch_f[1], field_grad=batch_fg[1])
    torch.testing.assert_close(
        e_perlane, torch.stack([e0_ref, e1_ref]), atol=1e-9, rtol=1e-9
    )

    grad_f, grad_fg = torch.autograd.grad(
        e_perlane.sum(), (batch_f, batch_fg), retain_graph=True
    )
    grad_f_ref, grad_fg_ref = torch.autograd.grad(
        e0_ref + e1_ref, (batch_f, batch_fg)
    )
    torch.testing.assert_close(grad_f, grad_f_ref, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(grad_fg, grad_fg_ref, atol=1e-7, rtol=1e-7)

    # Batched dipole and quadrupole with per-lane perturbations (both tensor and sequence)
    dip_perlane = calc.dipole(padded_pos, field=batch_f, field_grad=batch_fg)
    quad_perlane = calc.quadrupole(
        padded_pos, field=batch_f, field_grad=batch_fg
    )
    torch.testing.assert_close(dip_perlane, -grad_f_ref, atol=1e-7, rtol=1e-7)
    assert quad_perlane.shape == (2, 6)

    dip_seq = calc.dipole(
        padded_pos,
        field=[f0.clone().requires_grad_(True), f1.clone().requires_grad_(True)],
        field_grad=[fg0, None],
    )
    dip_w_seq = sys_w.dipole(pos_w, field=f0, field_grad=fg0)
    dip_lih_seq = sys_lih.dipole(pos_lih, field=f1, field_grad=None)
    torch.testing.assert_close(
        dip_seq, torch.stack([dip_w_seq, dip_lih_seq]), atol=1e-7, rtol=1e-7
    )
    quad_seq = calc.quadrupole(
        padded_pos,
        field=[f0, None],
        field_grad=[
            fg0.clone().requires_grad_(True),
            fg1.clone().requires_grad_(True),
        ],
    )
    assert quad_seq.shape == (2, 6)

    # 3. Per-lane sequence with None on one lane
    e_seq = calc.energy(
        padded_pos, field=[f0, None], field_grad=[None, fg1]
    )
    e0_seq_ref = sys_w.energy(pos_w, field=f0, field_grad=None)
    e1_seq_ref = sys_lih.energy(pos_lih, field=None, field_grad=fg1)
    torch.testing.assert_close(
        e_seq, torch.stack([e0_seq_ref, e1_seq_ref]), atol=1e-9, rtol=1e-9
    )

    # 4. Ambiguous shapes are clearly rejected
    with pytest.raises(ValueError, match="field must have shape"):
        calc.energy(padded_pos, field=torch.zeros((3, 3), **DD))
    with pytest.raises(ValueError, match="field_grad must have shape"):
        calc.energy(padded_pos, field_grad=torch.zeros((3, 3, 3), **DD))
    with pytest.raises(ValueError, match="sequence length"):
        evaluate_packed_systems(
            (sys_w, sys_lih), (pos_w, pos_lih), field=[f0]
        )


def test_grad_mode_removed_and_authoritative_force_ad() -> None:
    """Public grad_mode is rejected and forces use one authoritative AD definition."""
    num_w, pos_w = _water_geometry(0.0)
    calc = Calculator(
        num_w,
        GFN1_XTB,
        opts={"verbosity": 0, "int_level": labels.INTLEVEL_DIPOLE},
        **DD,
    )
    pos = pos_w.clone().requires_grad_(True)

    # Reject keyword grad_mode across all entry points
    for mode in ("autograd", "backward", "functorch", "row", "invalid"):
        with pytest.raises(TypeError, match="grad_mode"):
            calc.forces(pos, grad_mode=mode)
        with pytest.raises(TypeError, match="grad_mode"):
            calc.get_forces(pos, grad_mode=mode)
        with pytest.raises(TypeError, match="grad_mode"):
            calc.hessian(pos, grad_mode=mode)
        with pytest.raises(TypeError, match="grad_mode"):
            calc.calculate(["forces"], pos, grad_mode=mode)
        with pytest.raises(TypeError, match="grad_mode"):
            calc.get_property("forces", pos, grad_mode=mode)

    # Reject positional 4th argument on forces and get_forces
    with pytest.raises(TypeError):
        calc.forces(pos, 0, None, "autograd")  # type: ignore[misc]
    with pytest.raises(TypeError):
        calc.get_forces(pos, 0, None, "autograd")  # type: ignore[misc]

    # Higher-order derivative through calc.forces(create_graph=True) matches calc.hessian
    forces = calc.forces(pos, create_graph=True)
    assert pos.grad is None
    assert forces.grad_fn is not None
    (hess_row0,) = torch.autograd.grad(forces[0, 0], pos, retain_graph=True)
    hess_full = calc.hessian(pos)
    torch.testing.assert_close(-hess_row0, hess_full[0, 0], atol=1e-7, rtol=1e-7)


def test_all_13_setup_lifecycle_and_grad_mode_transitions() -> None:
    """Calculator supports all 13 grad/no_grad/inference_mode and parameter transitions."""
    num_w, pos_w1 = _water_geometry(0.0)
    _, pos_w2 = _water_geometry(0.06)
    num_lih, pos_lih = _lih_geometry(0.0)

    padded_nums = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    padded_pos1 = torch.stack(
        [pos_w1, torch.cat([pos_lih, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )
    padded_pos2 = torch.stack(
        [pos_w2, torch.cat([pos_lih, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )

    opts = {"verbosity": 0, "maxiter": 100, "x_atol": 1e-10}
    par = ParamModule(GFN1_XTB, **DD)
    par.set_differentiable("element", "O", "gam")
    gam_leaf = par.get("element.O.gam")

    cfg_opts = {k: v for k, v in opts.items() if k != "verbosity"}

    def fresh_ref(
        pos_1d: Tensor, current_par: ParamModule
    ) -> tuple[Tensor, Tensor, Tensor]:
        sys = Model(
            par=current_par,
            config=Config.create(method="gfn1-xtb", **cfg_opts),
        ).setup(num_w, dd=DD)
        p = pos_1d.detach().clone().requires_grad_(True)
        e = sys.energy(p)
        (f_neg, g_par) = torch.autograd.grad(e, (p, current_par.get("element.O.gam")))
        return e.detach(), -f_neg.detach(), g_par.detach()

    # Transition 1: Constructed under torch.no_grad(), then used for differentiation
    with torch.no_grad():
        calc_1d = Calculator(num_w, par, opts=opts, **DD)
        calc_2d = Calculator(
            padded_nums, par, opts={**opts, "batch_mode": 1}, **DD
        )
    assert calc_1d._integrals is None
    assert calc_2d._integrals is None

    e_ref, f_ref, g_ref = fresh_ref(pos_w1, par)
    p1 = pos_w1.clone().requires_grad_(True)
    f1 = calc_1d.forces(p1)
    e1 = calc_1d.energy(p1)
    (g1,) = torch.autograd.grad(e1, gam_leaf)
    torch.testing.assert_close(e1, e_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(f1, f_ref, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(g1, g_ref, atol=1e-7, rtol=1e-7)

    # Transition 2: Evaluated under torch.no_grad(), then grad-enabled
    with torch.no_grad():
        _ = calc_1d.energy(pos_w1)
        _ = calc_2d.energy(padded_pos1)
    p2 = pos_w1.clone().requires_grad_(True)
    pb2 = padded_pos1.clone().requires_grad_(True)
    torch.testing.assert_close(calc_1d.forces(p2), f_ref, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        calc_2d.forces(pb2)[0], f_ref, atol=1e-7, rtol=1e-7
    )

    # Transition 3: Constructed and evaluated under torch.inference_mode(), then grad-enabled
    with torch.inference_mode():
        par_inf = ParamModule(GFN1_XTB, **DD)
        calc_inf_1d = Calculator(num_w, par_inf, opts=opts, **DD)
        calc_inf_2d = Calculator(
            padded_nums, par, opts={**opts, "batch_mode": 1}, **DD
        )
        e_inf = calc_inf_1d.energy(pos_w1.clone())
        eb_inf = calc_inf_2d.energy(padded_pos1.clone())
    torch.testing.assert_close(e_inf.clone(), e_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(eb_inf[0].clone(), e_ref, atol=1e-9, rtol=1e-9)

    p3 = pos_w1.clone().requires_grad_(True)
    pb3 = padded_pos1.clone().requires_grad_(True)
    torch.testing.assert_close(
        calc_inf_1d.forces(p3), f_ref, atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        calc_inf_2d.forces(pb3)[0], f_ref, atol=1e-7, rtol=1e-7
    )
    par_inf.set_differentiable("element", "O", "gam")
    (g_inf,) = torch.autograd.grad(
        calc_inf_1d.energy(pos_w1), par_inf.get("element.O.gam")
    )
    torch.testing.assert_close(g_inf, g_ref, atol=1e-7, rtol=1e-7)

    # Transitions 4 & 5: Repeated first derivatives and repeated parameter gradients
    for _ in range(2):
        px = pos_w1.clone().requires_grad_(True)
        torch.testing.assert_close(
            calc_1d.forces(px), f_ref, atol=1e-7, rtol=1e-7
        )
        (gx,) = torch.autograd.grad(calc_1d.energy(pos_w1), gam_leaf)
        torch.testing.assert_close(gx, g_ref, atol=1e-7, rtol=1e-7)

    # Transition 6: backward() followed by another differentiation
    gam_leaf.grad = None
    p6 = pos_w1.clone().requires_grad_(True)
    calc_1d.energy(p6).backward()
    assert p6.grad is not None and gam_leaf.grad is not None
    torch.testing.assert_close(-p6.grad, f_ref, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(gam_leaf.grad, g_ref, atol=1e-7, rtol=1e-7)
    gam_leaf.grad = None
    p6_next = pos_w1.clone().requires_grad_(True)
    torch.testing.assert_close(
        calc_1d.forces(p6_next), f_ref, atol=1e-7, rtol=1e-7
    )

    # Transitions 7 & 8: torch.func.grad and nested jacrev(jacrev(...)) transforms
    def energy_fn(p: Tensor) -> Tensor:
        return calc_1d.energy(p)

    grad_pos = grad(energy_fn)(pos_w1)
    torch.testing.assert_close(-grad_pos, f_ref, atol=1e-7, rtol=1e-7)
    hess_ft = jacrev(jacrev(energy_fn))(pos_w1)
    hess_ad = calc_1d.hessian(pos_w1.clone().requires_grad_(True))
    torch.testing.assert_close(hess_ft, hess_ad, atol=1e-6, rtol=1e-6)

    # Transition 9: In-place Model parameter updates
    with torch.no_grad():
        gam_leaf.add_(0.04)
    e_ref_9, f_ref_9, g_ref_9 = fresh_ref(pos_w1, par)
    p9 = pos_w1.clone().requires_grad_(True)
    e9 = calc_1d.energy(p9)
    f9 = calc_1d.forces(p9)
    (g9,) = torch.autograd.grad(calc_1d.energy(pos_w1), gam_leaf)
    assert not torch.isclose(e9, e_ref)
    torch.testing.assert_close(e9, e_ref_9, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(f9, f_ref_9, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(g9, g_ref_9, atol=1e-7, rtol=1e-7)

    # Transition 10: Replacement of parameter tensors on Model.par
    par_replaced = ParamModule(GFN1_XTB, **DD)
    par_replaced.set_differentiable("element", "O", "gam")
    with torch.no_grad():
        par_replaced.get("element.O.gam").sub_(0.03)
    calc_1d.model = Model(
        par=par_replaced,
        config=calc_1d.opts,
        classical=calc_1d.model.classical,
        interaction=calc_1d.model.interaction,
        auto_int_level=False,
    )
    e_ref_10, f_ref_10, g_ref_10 = fresh_ref(pos_w1, par_replaced)
    p10 = pos_w1.clone().requires_grad_(True)
    torch.testing.assert_close(
        calc_1d.energy(p10), e_ref_10, atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        calc_1d.forces(p10), f_ref_10, atol=1e-7, rtol=1e-7
    )
    (g10,) = torch.autograd.grad(
        calc_1d.energy(pos_w1), par_replaced.get("element.O.gam")
    )
    torch.testing.assert_close(g10, g_ref_10, atol=1e-7, rtol=1e-7)

    # Transition 11: Dtype conversion (float64 -> float32 -> float64)
    calc_1d.type(torch.float32)
    calc_2d.type(torch.float32)
    pos32 = pos_w1.to(torch.float32).requires_grad_(True)
    ppos32 = padded_pos1.to(torch.float32).requires_grad_(True)
    e32 = calc_1d.energy(pos32)
    f32 = calc_1d.forces(pos32)
    eb32 = calc_2d.energy(ppos32)
    fb32 = calc_2d.forces(ppos32)
    assert e32.dtype == torch.float32 and f32.dtype == torch.float32
    assert eb32.dtype == torch.float32 and fb32.dtype == torch.float32
    torch.testing.assert_close(
        e32.to(torch.float64), e_ref_10, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        f32.to(torch.float64), f_ref_10, atol=1e-3, rtol=1e-3
    )
    calc_1d.type(torch.float64)
    calc_2d.type(torch.float64)
    # Also verify round-trip dtype conversion when Model.par has requires_grad=False
    calc_nograd_par = Calculator(num_w, GFN1_XTB, opts=opts, **DD)
    calc_nograd_par.type(torch.float32)
    assert calc_nograd_par.energy(pos_w1.to(torch.float32)).dtype == torch.float32
    calc_nograd_par.type(torch.float64)
    torch.testing.assert_close(
        calc_nograd_par.energy(pos_w1), e_ref, atol=1e-9, rtol=1e-9
    )

    # Transitions 12 & 13: Repeated evaluation with distinct positions and reset()
    e_ref_w2, f_ref_w2, _ = fresh_ref(pos_w2, par_replaced)
    p12 = pos_w2.clone().requires_grad_(True)
    torch.testing.assert_close(
        calc_1d.energy(p12), e_ref_w2, atol=1e-9, rtol=1e-9
    )
    torch.testing.assert_close(
        calc_1d.forces(p12), f_ref_w2, atol=1e-7, rtol=1e-7
    )
    calc_1d.reset()
    calc_2d.reset()
    assert calc_1d._integrals is None and calc_2d._integrals is None
    p13 = pos_w1.clone().requires_grad_(True)
    pb13 = padded_pos2.clone().requires_grad_(True)
    torch.testing.assert_close(
        calc_1d.forces(p13), f_ref_10, atol=1e-7, rtol=1e-7
    )
    assert torch.isfinite(calc_2d.forces(pb13)).all()


def test_caller_owned_components_are_not_mutated_and_exact_core_stays_active() -> None:
    """User-supplied classical and interaction objects remain unmodified across lane setups."""
    padded_nums = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    _, pos_w1 = _water_geometry(0.0)
    _, pos_w2 = _water_geometry(0.05)
    _, pos_lih = _lih_geometry(0.0)
    padded_pos = torch.stack(
        [pos_w1, torch.cat([pos_lih, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )
    par = ParamModule(GFN1_XTB, **DD)

    # 1. Heterogeneous padded batch with caller-owned DispersionD3 (2D numbers)
    user_d3 = new_dispersion(padded_nums, par, **DD)
    assert user_d3 is not None
    orig_d3_nums = user_d3.numbers
    orig_d3_label = user_d3.label

    calc_het = Calculator(
        padded_nums,
        par,
        classical=(user_d3,),
        opts={"verbosity": 0, "exclude": ["disp"], "batch_mode": 1},
        **DD,
    )
    lanes_het = build_calculator_lane_systems(calc_het)
    for lane in lanes_het:
        assert lane._active_exact_core() is not None
    assert user_d3.numbers is orig_d3_nums
    assert user_d3.numbers.shape == (2, 3)
    assert user_d3.label == orig_d3_label
    res_het = calc_het.singlepoint(padded_pos)
    assert torch.isfinite(res_het.total).all()

    # 2. Conformer batch with caller-owned DispersionD3, Repulsion, and ES2
    conf_nums = torch.tensor([[8, 1, 1], [8, 1, 1]], device=DEVICE)
    conf_pos = torch.stack([pos_w1, pos_w2], dim=0)
    unique_nums = torch.unique(conf_nums)
    user_d3_conf = new_dispersion(conf_nums, par, **DD)
    user_rep = new_repulsion(unique_nums, par, **DD)
    user_es2 = new_es2(unique_nums, par, **DD)
    assert (
        user_d3_conf is not None
        and user_rep is not None
        and user_es2 is not None
    )

    orig_conf_nums = user_d3_conf.numbers
    orig_rep_label = user_rep.label
    orig_es2_label = user_es2.label

    calc_conf = Calculator(
        conf_nums,
        par,
        classical=(user_d3_conf, user_rep),
        interaction=(user_es2,),
        opts={
            "verbosity": 0,
            "exclude": ["disp", "rep", "es2"],
            "batch_mode": 2,
        },
        **DD,
    )
    lanes_conf = build_calculator_lane_systems(calc_conf)
    for lane in lanes_conf:
        assert lane._active_exact_core() is not None
    assert user_d3_conf.numbers is orig_conf_nums
    assert user_d3_conf.numbers.shape == (2, 3)
    assert user_rep.label == orig_rep_label
    assert user_es2.label == orig_es2_label

    res_conf = calc_conf.singlepoint(conf_pos)
    assert torch.isfinite(res_conf.total).all()
