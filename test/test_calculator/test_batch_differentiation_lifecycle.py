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
from ..utils import scf_strategy

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
        "maxiter": 14,
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
        sp_m1.energy.sum(-1), expected_e, atol=1e-9, rtol=1e-9
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
        "maxiter": 14,
        "x_atol": 1e-10,
        "fermi_etemp": 0.0,
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
        opts={
            "verbosity": 0,
            "maxiter": 14,
            "fermi_etemp": 0.0,
            "int_level": labels.INTLEVEL_DIPOLE,
        },
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

    opts = {"verbosity": 0, "maxiter": 14, "x_atol": 1e-10, "fermi_etemp": 0.0}
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
    assert not hasattr(calc_1d, "integrals")
    assert not hasattr(calc_2d, "integrals")

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
    assert not hasattr(calc_1d, "integrals") and not hasattr(calc_2d, "integrals")
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
    assert torch.isfinite(res_het.energy).all()

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
    assert torch.isfinite(res_conf.energy).all()


@pytest.mark.parametrize("differentiation", ["unrolled", "implicit"])
def test_batched_higher_order_electric_responses_match_serial_and_preserve_grad(
    differentiation: str,
) -> None:
    """Batched dipole, quadrupole, polarizability, dipole_deriv, and bond_orders match serial Systems."""
    num_w, pos_w = _water_geometry(0.0)
    num_l, pos_l = _lih_geometry(0.0)
    # Non-contiguous padding in lane 1: [3, 0, 1]
    padded_nums = torch.tensor([[8, 1, 1], [3, 0, 1]], device=DEVICE)
    padded_pos = torch.stack(
        [
            pos_w,
            torch.stack([pos_l[0], torch.zeros(3, **DD), pos_l[1]], dim=0),
        ],
        dim=0,
    )

    opts = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "differentiation": differentiation,
        "strategy": scf_strategy("anderson"),
        "maxiter": 12,
        "x_atol": 1e-10,
        "fermi_etemp": 0.0,
    }
    par = ParamModule(GFN1_XTB, **DD)

    calc_batch = Calculator(padded_nums, par, opts=opts, **DD)
    calc_w = Calculator(num_w, par, opts=opts, **DD)
    calc_l = Calculator(num_l, par, opts=opts, **DD)

    shared_f = torch.tensor([0.01, -0.015, 0.02], **DD, requires_grad=True)
    lane_f = torch.tensor(
        [[0.01, -0.015, 0.02], [-0.02, 0.01, -0.01]],
        **DD,
        requires_grad=True,
    )
    shared_fg = torch.tensor(
        [[0.01, 0.005, 0.0], [0.005, -0.01, 0.002], [0.0, 0.002, 0.0]],
        **DD,
        requires_grad=True,
    )
    lane_fg = torch.stack([shared_fg.detach(), -0.5 * shared_fg.detach()]).requires_grad_(True)

    # 1. Bond orders (including non-contiguous padding in lane 1)
    bo_batch = calc_batch.bond_orders(padded_pos)
    bo_w = calc_w.bond_orders(pos_w)
    bo_l = calc_l.bond_orders(pos_l)
    assert bo_batch.shape == (2, 3, 3)
    torch.testing.assert_close(bo_batch[0], bo_w, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(
        bo_batch[1, [0, 2]][:, [0, 2]], bo_l, atol=1e-9, rtol=1e-9
    )
    assert torch.all(bo_batch[1, 1, :] == 0.0)
    assert torch.all(bo_batch[1, :, 1] == 0.0)

    # 2. Shared and per-lane dipole & quadrupole (supported for both "unrolled" and "implicit")
    for f_in, f_lanes in (
        (shared_f, (shared_f, shared_f)),
        (lane_f, (lane_f[0], lane_f[1])),
    ):
        dip_0 = calc_w.dipole(
            pos_w,
            use_functorch=False,
            field=f_lanes[0].detach().requires_grad_(True),
        )
        dip_1 = calc_l.dipole(
            pos_l,
            use_functorch=False,
            field=f_lanes[1].detach().requires_grad_(True),
        )
        if differentiation == "unrolled":
            pol_0 = calc_w.polarizability(
                pos_w,
                use_functorch=False,
                field=f_lanes[0].detach().requires_grad_(True),
            )
            pol_1 = calc_l.polarizability(
                pos_l,
                use_functorch=False,
                field=f_lanes[1].detach().requires_grad_(True),
            )
            dmu_0 = calc_w.dipole_deriv(
                pos_w.clone().requires_grad_(True),
                use_functorch=False,
                field=f_lanes[0].detach().requires_grad_(True),
            )
            dmu_1 = calc_l.dipole_deriv(
                pos_l.clone().requires_grad_(True),
                use_functorch=False,
                field=f_lanes[1].detach().requires_grad_(True),
            )
        for use_ft in (False, True):
            f_arg = f_in.detach() if use_ft else f_in
            dip_b = calc_batch.dipole(
                padded_pos, use_functorch=use_ft, field=f_arg
            )
            assert dip_b.shape == (2, 3)
            torch.testing.assert_close(dip_b[0], dip_0, atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(dip_b[1], dip_1, atol=1e-7, rtol=1e-7)

            if differentiation != "unrolled":
                continue

            # Polarizability (both derived_quantity="dipole" and "energy")
            dq_list = ("dipole",) if not use_ft else ("dipole", "energy")
            for dq in dq_list:
                pol_b = calc_batch.polarizability(
                    padded_pos,
                    use_functorch=use_ft,
                    derived_quantity=dq,
                    field=f_arg,
                )
                assert pol_b.shape == (2, 3, 3)
                torch.testing.assert_close(
                    pol_b[0], pol_0, atol=1e-6, rtol=1e-6
                )
                torch.testing.assert_close(
                    pol_b[1], pol_1, atol=1e-6, rtol=1e-6
                )

            # Dipole derivatives (including non-contiguous padding mask)
            pos_b_grad = padded_pos.clone().requires_grad_(True)
            dmu_b = calc_batch.dipole_deriv(
                pos_b_grad, use_functorch=use_ft, field=f_arg
            )
            assert dmu_b.shape == (2, 3, 3, 3)
            torch.testing.assert_close(dmu_b[0], dmu_0, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(
                dmu_b[1, :, [0, 2], :], dmu_1, atol=1e-6, rtol=1e-6
            )
            assert torch.all(dmu_b[1, :, 1, :] == 0.0)

    for fg_in, fg_lanes in (
        (shared_fg, (shared_fg, shared_fg)),
        (lane_fg, (lane_fg[0], lane_fg[1])),
    ):
        quad_0 = calc_w.quadrupole(
            pos_w,
            use_functorch=False,
            field_grad=fg_lanes[0].detach().requires_grad_(True),
        )
        quad_1 = calc_l.quadrupole(
            pos_l,
            use_functorch=False,
            field_grad=fg_lanes[1].detach().requires_grad_(True),
        )
        for use_ft in (False, True):
            fg_arg = fg_in.detach() if use_ft else fg_in
            quad_b = calc_batch.quadrupole(
                padded_pos, use_functorch=use_ft, field_grad=fg_arg
            )
            assert quad_b.shape == (2, 6)
            torch.testing.assert_close(quad_b[0], quad_0, atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(quad_b[1], quad_1, atol=1e-7, rtol=1e-7)

    if differentiation != "unrolled":
        return

    # 3. Differentiability of batched polarizability w.r.t. Model.par, positions, and field
    par.set_differentiable("element", "O", "gam")
    gam_leaf = par.get("element.O.gam")
    p_grad = padded_pos.clone().requires_grad_(True)
    f_grad = lane_f.detach().clone().requires_grad_(True)
    pol_diff = calc_batch.polarizability(
        p_grad, use_functorch=True, field=f_grad
    )
    loss_b = (pol_diff**2).sum()
    g_par_b, g_pos_b, g_f_b = torch.autograd.grad(
        loss_b, (gam_leaf, p_grad, f_grad)
    )

    p0_g = pos_w.clone().requires_grad_(True)
    p1_g = pos_l.clone().requires_grad_(True)
    f0_g = lane_f[0].detach().clone().requires_grad_(True)
    f1_g = lane_f[1].detach().clone().requires_grad_(True)
    pol_s0 = calc_w.polarizability(p0_g, use_functorch=True, field=f0_g)
    pol_s1 = calc_l.polarizability(p1_g, use_functorch=True, field=f1_g)
    loss_s = (pol_s0**2).sum() + (pol_s1**2).sum()
    g_par_s, g_pos0_s, g_pos1_s, g_f0_s, g_f1_s = torch.autograd.grad(
        loss_s, (gam_leaf, p0_g, p1_g, f0_g, f1_g)
    )
    torch.testing.assert_close(g_par_b, g_par_s, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(g_pos_b[0], g_pos0_s, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        g_pos_b[1, [0, 2]], g_pos1_s, atol=1e-6, rtol=1e-6
    )
    assert torch.all(g_pos_b[1, 1] == 0.0)
    torch.testing.assert_close(
        g_f_b, torch.stack([g_f0_s, g_f1_s], dim=0), atol=1e-6, rtol=1e-6
    )

    # 4. Numerical per-lane responses on a compact 2-lane s-shell batch
    small_nums = torch.tensor([[1, 1], [1, 0]], device=DEVICE)
    small_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        **DD,
    )
    small_chrg = torch.tensor([0.0, 1.0], **DD)
    calc_small = Calculator(small_nums, GFN1_XTB, opts=opts, **DD)
    calc_s0 = Calculator(
        torch.tensor([1, 1], device=DEVICE), GFN1_XTB, opts=opts, **DD
    )
    dip_num_b = calc_small.dipole_numerical(
        small_pos, small_chrg, field=lane_f.detach()
    )
    dip_num_0 = calc_s0.dipole_numerical(
        small_pos[0], 0.0, field=lane_f[0].detach()
    )
    torch.testing.assert_close(
        dip_num_b[0], dip_num_0, atol=1e-7, rtol=1e-7
    )
    pol_num_b = calc_small.polarizability_numerical(
        small_pos, small_chrg, field=lane_f.detach()
    )
    pol_num_0 = calc_s0.polarizability_numerical(
        small_pos[0], 0.0, field=lane_f[0].detach()
    )
    torch.testing.assert_close(
        pol_num_b[0], pol_num_0, atol=1e-6, rtol=1e-6
    )
    quad_num_b = calc_small.quadrupole_numerical(
        small_pos, small_chrg, field_grad=lane_fg.detach()
    )
    quad_num_0 = calc_s0.quadrupole_numerical(
        small_pos[0], 0.0, field_grad=lane_fg[0].detach()
    )
    torch.testing.assert_close(
        quad_num_b[0], quad_num_0, atol=1e-7, rtol=1e-7
    )


@pytest.mark.slow
def test_batched_third_order_electric_responses_match_serial() -> None:
    """Batched pol_deriv and hyperpolarizability match serial Systems for shared and per-lane fields."""
    num_l, pos_l = _lih_geometry(0.0)
    num_h2 = torch.tensor([1, 1], device=DEVICE)
    pos_h2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **DD)
    # Non-contiguous padding in lane 1: [1, 0, 1] and trailing padding in lane 0: [3, 1, 0]
    padded_nums = torch.tensor([[3, 1, 0], [1, 0, 1]], device=DEVICE)
    padded_pos = torch.stack(
        [
            torch.cat([pos_l, torch.zeros((1, 3), **DD)], dim=0),
            torch.stack([pos_h2[0], torch.zeros(3, **DD), pos_h2[1]], dim=0),
        ],
        dim=0,
    )
    opts = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "int_level": labels.INTLEVEL_DIPOLE,
        "differentiation": "unrolled",
        "strategy": scf_strategy("anderson"),
        "maxiter": 2,
        "x_atol": 1e-10,
        "fermi_etemp": 0.0,
    }
    calc_batch = Calculator(padded_nums, GFN1_XTB, opts=opts, **DD)
    calc_0 = Calculator(num_l, GFN1_XTB, opts=opts, **DD)
    calc_1 = Calculator(num_h2, GFN1_XTB, opts=opts, **DD)

    shared_f = torch.tensor([0.01, -0.015, 0.02], **DD)
    lane_f = torch.tensor(
        [[0.01, -0.015, 0.02], [-0.02, 0.01, -0.01]], **DD
    )

    chi_0 = calc_0.pol_deriv(
        pos_l.clone().requires_grad_(True),
        use_functorch=False,
        field=shared_f.clone().requires_grad_(True),
    )
    beta_0 = calc_0.hyperpolarizability(
        pos_l,
        use_functorch=False,
        field=shared_f.clone().requires_grad_(True),
    )

    chi_1_shared = calc_1.pol_deriv(
        pos_h2.clone().requires_grad_(True),
        use_functorch=False,
        field=shared_f.clone().requires_grad_(True),
    )
    beta_1_shared = calc_1.hyperpolarizability(
        pos_h2,
        use_functorch=False,
        field=shared_f.clone().requires_grad_(True),
    )
    chi_1_lane = calc_1.pol_deriv(
        pos_h2.clone().requires_grad_(True),
        use_functorch=False,
        field=lane_f[1].clone().requires_grad_(True),
    )
    beta_1_lane = calc_1.hyperpolarizability(
        pos_h2,
        use_functorch=False,
        field=lane_f[1].clone().requires_grad_(True),
    )

    for f_in, chi_1, beta_1, use_ft in (
        (shared_f, chi_1_shared, beta_1_shared, False),
        (lane_f, chi_1_lane, beta_1_lane, True),
    ):
        f_arg = f_in.clone().requires_grad_(not use_ft)
        chi_b = calc_batch.pol_deriv(
            padded_pos.clone().requires_grad_(True),
            use_functorch=use_ft,
            derived_quantity="dipole",
            field=f_arg,
        )
        assert chi_b.shape == (2, 3, 3, 3, 3)
        torch.testing.assert_close(
            chi_b[0, :, :, :2, :], chi_0, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            chi_b[1, :, :, [0, 2], :], chi_1, atol=1e-5, rtol=1e-5
        )
        assert torch.all(chi_b[0, :, :, 2, :] == 0.0)
        assert torch.all(chi_b[1, :, :, 1, :] == 0.0)

        beta_b = calc_batch.hyperpolarizability(
            padded_pos,
            use_functorch=use_ft,
            derived_quantity="pol",
            field=f_in.clone().requires_grad_(not use_ft),
        )
        assert beta_b.shape == (2, 3, 3, 3)
        torch.testing.assert_close(beta_b[0], beta_0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(beta_b[1], beta_1, atol=1e-5, rtol=1e-5)


def test_cross_lane_derivative_isolation_for_fields_and_positions() -> None:
    """Perturbing or differentiating lane 0 inputs never contaminates lane 1."""
    _, pos_w1 = _water_geometry(0.0)
    _, pos_l = _lih_geometry(0.0)
    padded_nums = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    padded_pos = torch.stack(
        [pos_w1, torch.cat([pos_l, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )
    opts = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "differentiation": "unrolled",
        "strategy": scf_strategy("anderson"),
        "maxiter": 12,
        "x_atol": 1e-10,
        "fermi_etemp": 0.0,
    }
    calc = Calculator(padded_nums, GFN1_XTB, opts=opts, **DD)

    f_base = torch.tensor(
        [[0.01, -0.02, 0.015], [-0.015, 0.025, -0.01]], **DD
    )
    f_pert0 = f_base.clone()
    f_pert0[0] = torch.tensor([0.05, 0.04, -0.03], **DD)

    # Perturbing lane 0 field leaves lane 1 outputs unchanged
    e_base = calc.energy(padded_pos, field=f_base)
    e_pert0 = calc.energy(padded_pos, field=f_pert0)
    assert not torch.isclose(e_base[0], e_pert0[0])
    torch.testing.assert_close(e_base[1], e_pert0[1], atol=1e-12, rtol=1e-12)

    for prop_name in ("dipole", "polarizability", "hyperpolarizability"):
        fn = getattr(calc, prop_name)
        v_base = fn(padded_pos, use_functorch=True, field=f_base)
        v_pert0 = fn(padded_pos, use_functorch=True, field=f_pert0)
        assert not torch.allclose(v_base[0], v_pert0[0])
        torch.testing.assert_close(
            v_base[1], v_pert0[1], atol=1e-11, rtol=1e-11
        )

    # Cross-lane field derivative blocks d(lane 1)/d(field[0]) == 0
    f_leaf = f_base.clone().requires_grad_(True)
    e_vec = calc.energy(padded_pos, field=f_leaf)
    (de1_df,) = torch.autograd.grad(e_vec[1], f_leaf, retain_graph=True)
    assert torch.all(de1_df[0] == 0.0)
    assert torch.count_nonzero(de1_df[1]) > 0

    mu_vec = calc.dipole(padded_pos, use_functorch=True, field=f_leaf)
    (dmu1_df,) = torch.autograd.grad(mu_vec[1].sum(), f_leaf, retain_graph=True)
    assert torch.all(dmu1_df[0] == 0.0)
    assert torch.count_nonzero(dmu1_df[1]) > 0

    alpha_vec = calc.polarizability(
        padded_pos, use_functorch=True, field=f_leaf
    )
    (da1_df,) = torch.autograd.grad(alpha_vec[1].sum(), f_leaf)
    assert torch.all(da1_df[0] == 0.0)
    assert torch.count_nonzero(da1_df[1]) > 0

    # Cross-lane field_grad derivative blocks d(lane 1)/d(field_grad[0]) == 0
    fg_base = torch.stack(
        [
            torch.eye(3, **DD) * 0.01,
            torch.tensor(
                [[0.01, 0.005, 0.0], [0.005, -0.01, 0.0], [0.0, 0.0, 0.0]],
                **DD,
            ),
        ],
        dim=0,
    ).requires_grad_(True)
    e_fg = calc.energy(padded_pos, field_grad=fg_base)
    (de1_dfg,) = torch.autograd.grad(e_fg[1], fg_base, retain_graph=True)
    assert torch.all(de1_dfg[0] == 0.0)
    assert torch.count_nonzero(de1_dfg[1]) > 0

    quad_fg = calc.quadrupole(padded_pos, use_functorch=True, field_grad=fg_base)
    (dq1_dfg,) = torch.autograd.grad(quad_fg[1].sum(), fg_base)
    assert torch.all(dq1_dfg[0] == 0.0)
    assert torch.count_nonzero(dq1_dfg[1]) > 0

    # Cross-lane position derivative blocks d(lane 1)/d(positions[0]) == 0
    pos_leaf = padded_pos.clone().requires_grad_(True)
    e_pos = calc.energy(pos_leaf, field=f_base)
    (de1_dr,) = torch.autograd.grad(e_pos[1], pos_leaf, retain_graph=True)
    assert torch.all(de1_dr[0] == 0.0)
    assert torch.all(de1_dr[1, 2] == 0.0)
    assert torch.count_nonzero(de1_dr[1, :2]) > 0

    mu_pos = calc.dipole(pos_leaf, use_functorch=True, field=f_base)
    (dmu1_dr,) = torch.autograd.grad(
        mu_pos[1].sum(), pos_leaf, retain_graph=True
    )
    assert torch.all(dmu1_dr[0] == 0.0)
    assert torch.all(dmu1_dr[1, 2] == 0.0)

    alpha_pos = calc.polarizability(pos_leaf, use_functorch=True, field=f_base)
    (da1_dr,) = torch.autograd.grad(alpha_pos[1].sum(), pos_leaf)
    assert torch.all(da1_dr[0] == 0.0)
    assert torch.all(da1_dr[1, 2] == 0.0)


def test_three_lane_batch_field_and_field_grad_disambiguation() -> None:
    """Three-lane (B == 3) batches disambiguate shared vs per-lane fields by tensor rank."""
    num_w1, pos_w1 = _water_geometry(0.0)
    num_w2, pos_w2 = _water_geometry(0.04)
    num_l, pos_l = _lih_geometry(0.0)

    nums_b3 = torch.tensor([[8, 1, 1], [8, 1, 1], [3, 1, 0]], device=DEVICE)
    pos_b3 = torch.stack(
        [pos_w1, pos_w2, torch.cat([pos_l, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )
    opts = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "differentiation": "unrolled",
        "strategy": scf_strategy("anderson"),
        "maxiter": 12,
        "x_atol": 1e-10,
        "fermi_etemp": 0.0,
    }
    calc_b3 = Calculator(nums_b3, GFN1_XTB, opts=opts, **DD)
    serial_calcs = (
        Calculator(num_w1, GFN1_XTB, opts=opts, **DD),
        Calculator(num_w2, GFN1_XTB, opts=opts, **DD),
        Calculator(num_l, GFN1_XTB, opts=opts, **DD),
    )
    serial_pos = (pos_w1, pos_w2, pos_l)

    # 1. Shared field (3,) vs per-lane field (3, 3) when B == 3
    f_shared = torch.tensor([0.01, -0.02, 0.03], **DD, requires_grad=True)
    f_batched = torch.tensor(
        [
            [0.01, -0.02, 0.03],
            [-0.015, 0.025, -0.01],
            [0.02, 0.01, -0.025],
        ],
        **DD,
        requires_grad=True,
    )
    f_seq = [f_batched[i].detach().clone().requires_grad_(True) for i in range(3)]

    e_shared = calc_b3.energy(pos_b3, field=f_shared)
    e_shared_ref = torch.stack(
        [c.energy(p, field=f_shared.detach()) for c, p in zip(serial_calcs, serial_pos)]
    )
    torch.testing.assert_close(e_shared, e_shared_ref, atol=1e-9, rtol=1e-9)

    e_batched = calc_b3.energy(pos_b3, field=f_batched)
    e_seq = calc_b3.energy(pos_b3, field=f_seq)
    e_batched_ref = torch.stack(
        [
            c.energy(p, field=f_batched[i].detach())
            for i, (c, p) in enumerate(zip(serial_calcs, serial_pos))
        ]
    )
    torch.testing.assert_close(e_batched, e_batched_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(e_seq, e_batched_ref, atol=1e-9, rtol=1e-9)

    dip_sh_ref = torch.stack(
        [
            c.dipole(
                p,
                use_functorch=False,
                field=f_shared.detach().requires_grad_(True),
            )
            for c, p in zip(serial_calcs, serial_pos)
        ]
    )
    dip_ba_ref = torch.stack(
        [
            c.dipole(
                p,
                use_functorch=False,
                field=f_batched[i].detach().requires_grad_(True),
            )
            for i, (c, p) in enumerate(zip(serial_calcs, serial_pos))
        ]
    )
    pol_sh_ref = torch.stack(
        [
            c.polarizability(
                p,
                use_functorch=False,
                field=f_shared.detach().requires_grad_(True),
            )
            for c, p in zip(serial_calcs, serial_pos)
        ]
    )
    pol_ba_ref = torch.stack(
        [
            c.polarizability(
                p,
                use_functorch=False,
                field=f_batched[i].detach().requires_grad_(True),
            )
            for i, (c, p) in enumerate(zip(serial_calcs, serial_pos))
        ]
    )
    for use_ft in (False, True):
        dip_sh = calc_b3.dipole(pos_b3, use_functorch=use_ft, field=f_shared)
        dip_ba = calc_b3.dipole(pos_b3, use_functorch=use_ft, field=f_batched)
        dip_sq = calc_b3.dipole(pos_b3, use_functorch=use_ft, field=f_seq)
        torch.testing.assert_close(dip_sh, dip_sh_ref, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(dip_ba, dip_ba_ref, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(dip_sq, dip_ba_ref, atol=1e-7, rtol=1e-7)

        pol_sh = calc_b3.polarizability(
            pos_b3, use_functorch=use_ft, field=f_shared
        )
        pol_ba = calc_b3.polarizability(
            pos_b3, use_functorch=use_ft, field=f_batched
        )
        torch.testing.assert_close(pol_sh, pol_sh_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(pol_ba, pol_ba_ref, atol=1e-6, rtol=1e-6)

    # 2. Shared field_grad (3, 3) vs per-lane field_grad (3, 3, 3) when B == 3
    fg_shared = torch.tensor(
        [[0.01, 0.005, 0.0], [0.005, -0.01, 0.002], [0.0, 0.002, 0.0]],
        **DD,
        requires_grad=True,
    )
    fg_batched = torch.stack(
        [
            fg_shared.detach(),
            -0.5 * fg_shared.detach(),
            0.8 * fg_shared.detach(),
        ],
        dim=0,
    ).requires_grad_(True)
    fg_seq = [
        fg_batched[i].detach().clone().requires_grad_(True) for i in range(3)
    ]

    q_sh_ref = torch.stack(
        [
            c.quadrupole(
                p,
                use_functorch=False,
                field_grad=fg_shared.detach().requires_grad_(True),
            )
            for c, p in zip(serial_calcs, serial_pos)
        ]
    )
    q_ba_ref = torch.stack(
        [
            c.quadrupole(
                p,
                use_functorch=False,
                field_grad=fg_batched[i].detach().requires_grad_(True),
            )
            for i, (c, p) in enumerate(zip(serial_calcs, serial_pos))
        ]
    )
    for use_ft in (False, True):
        q_sh = calc_b3.quadrupole(
            pos_b3, use_functorch=use_ft, field_grad=fg_shared
        )
        q_ba = calc_b3.quadrupole(
            pos_b3, use_functorch=use_ft, field_grad=fg_batched
        )
        q_sq = calc_b3.quadrupole(
            pos_b3, use_functorch=use_ft, field_grad=fg_seq
        )
        torch.testing.assert_close(q_sh, q_sh_ref, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(q_ba, q_ba_ref, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(q_sq, q_ba_ref, atol=1e-7, rtol=1e-7)

    # 3. Malformed field and field_grad shapes raise ValueError
    for bad_field in (
        torch.zeros((2, 3), **DD),
        torch.zeros((4, 3), **DD),
        torch.zeros((3, 3, 3), **DD),
    ):
        with pytest.raises(ValueError, match="field"):
            calc_b3.energy(pos_b3, field=bad_field)
        with pytest.raises(ValueError, match="field"):
            calc_b3.dipole(pos_b3, use_functorch=True, field=bad_field)

    for bad_fg in (
        torch.zeros((3,), **DD),
        torch.zeros((2, 3, 3), **DD),
        torch.zeros((3, 3, 3, 3), **DD),
    ):
        with pytest.raises(ValueError, match="field_grad"):
            calc_b3.energy(pos_b3, field_grad=bad_fg)
        with pytest.raises(ValueError, match="field_grad"):
            calc_b3.quadrupole(pos_b3, use_functorch=True, field_grad=bad_fg)


def test_inference_mode_param_module_ownership_and_identity() -> None:
    """Caller-owned ParamModule references are never silently replaced across inference-mode transitions."""
    num_w, pos_w = _water_geometry(0.0)
    _, pos_l = _lih_geometry(0.0)
    padded_nums = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    padded_pos = torch.stack(
        [pos_w, torch.cat([pos_l, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )
    opts = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "differentiation": "unrolled",
        "maxiter": 14,
        "fermi_etemp": 0.0,
    }

    # 1. Normal preexisting ParamModule: all parameter and buffer identities preserved
    par_norm = ParamModule(GFN1_XTB, **DD)
    par_norm.set_differentiable("element", "O", "gam")
    norm_params_before = {
        k: v for k, v in par_norm.named_parameters()
    }
    norm_buffers_before = {
        k: v for k, v in par_norm.named_buffers()
    }
    calc_norm = Calculator(num_w, par_norm, opts=opts, **DD)
    _ = calc_norm.energy(pos_w)
    for k, v in par_norm.named_parameters():
        assert v is norm_params_before[k]
    for k, v in par_norm.named_buffers():
        assert v is norm_buffers_before[k]

    # 2. ParamModule constructed under torch.inference_mode() with retained external references
    with torch.inference_mode():
        par_inf = ParamModule(GFN1_XTB, **DD)
        o_gam_ref = par_inf.get("element.O.gam")
        h_zeff_ref = par_inf.get("element.H.zeff")
        dummy_buf_ref = par_inf.dummy
        calc_inf_1d = Calculator(num_w, par_inf, opts=opts, **DD)
        calc_inf_2d = Calculator(padded_nums, par_inf, opts=opts, **DD)
        _ = calc_inf_1d.energy(pos_w)
        _ = calc_inf_2d.energy(padded_pos)

    assert not o_gam_ref.is_inference()
    assert not h_zeff_ref.is_inference()
    assert not dummy_buf_ref.is_inference()
    assert par_inf.get("element.O.gam") is o_gam_ref
    assert par_inf.get("element.H.zeff") is h_zeff_ref
    assert par_inf.dummy is dummy_buf_ref

    par_inf.set_differentiable("element", "O", "gam")
    assert o_gam_ref.requires_grad is True
    assert h_zeff_ref.requires_grad is False
    assert par_inf.get("element.O.gam") is o_gam_ref

    # Optimizer holds the original reference captured inside inference_mode
    optimizer = torch.optim.SGD([o_gam_ref], lr=0.1)
    optimizer.zero_grad()
    loss = calc_inf_1d.energy(pos_w) + calc_inf_2d.energy(padded_pos).sum()
    loss.backward()
    assert o_gam_ref.grad is not None and torch.isfinite(o_gam_ref.grad)
    val_before_step = o_gam_ref.detach().clone()
    optimizer.step()
    assert not torch.equal(o_gam_ref, val_before_step)
    assert par_inf.get("element.O.gam") is o_gam_ref

    # 3. ParamModule containing both normal and manually-injected inference tensors
    par_mixed = ParamModule(GFN1_XTB, **DD)
    par_mixed.set_differentiable("element", "O", "gam")
    o_gam_mixed = par_mixed.get("element.O.gam")
    orig_h_gam_val = par_mixed.get("element.H.gam").detach().clone()
    with torch.inference_mode():
        inf_h_gam = torch.nn.Parameter(
            orig_h_gam_val.clone(), requires_grad=False
        )
        inf_buf = torch.zeros(1, **DD)
    assert inf_h_gam.is_inference() and inf_buf.is_inference()
    par_mixed.get("element.H.gam", unwrapped=False)._parameters["param"] = (
        inf_h_gam
    )
    par_mixed.register_buffer("custom_inf_buf", inf_buf)

    calc_mixed_1d = Calculator(num_w, par_mixed, opts=opts, **DD)
    calc_mixed_2d = Calculator(padded_nums, par_mixed, opts=opts, **DD)
    e_m1 = calc_mixed_1d.energy(pos_w)
    e_m2 = calc_mixed_2d.energy(padded_pos)
    (g_m1,) = torch.autograd.grad(e_m1, o_gam_mixed)
    (g_m2,) = torch.autograd.grad(e_m2.sum(), o_gam_mixed)
    assert torch.isfinite(g_m1) and torch.isfinite(g_m2)

    # Caller's inference-tensor parameter and buffer references are NOT replaced in-place
    assert par_mixed.get("element.H.gam") is inf_h_gam
    assert par_mixed.get("element.H.gam").is_inference()
    assert par_mixed.get("element.O.gam") is o_gam_mixed
    assert par_mixed._buffers["custom_inf_buf"] is inf_buf

    # Requesting gradient tracking on the inference tensor raises a clear RuntimeError
    with pytest.raises(RuntimeError, match="inference-mode tensor"):
        par_mixed.set_differentiable("element", "H", "gam")

    # 4. Module parameter replaced between calculations is immediately observed
    new_o_gam = torch.nn.Parameter(
        o_gam_mixed.detach().clone() + 0.05, requires_grad=True
    )
    par_mixed.get("element.O.gam", unwrapped=False)._parameters["param"] = (
        new_o_gam
    )
    e_after_replace = calc_mixed_1d.energy(pos_w)
    assert not torch.isclose(e_after_replace, e_m1)
    (g_new,) = torch.autograd.grad(e_after_replace, new_o_gam)
    assert torch.isfinite(g_new) and g_new.abs().item() > 0.0
    assert par_mixed.get("element.O.gam") is new_o_gam


def test_call_local_setup_preserves_component_labels_and_state() -> None:
    """Evaluation never mutates caller-owned components or their labels."""
    num_w, pos_w = _water_geometry(0.0)
    _, pos_l = _lih_geometry(0.0)
    padded_nums = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    padded_pos = torch.stack(
        [pos_w, torch.cat([pos_l, torch.zeros((1, 3), **DD)], dim=0)],
        dim=0,
    )

    par = ParamModule(GFN1_XTB, **DD)
    s8 = torch.tensor(1.5, **DD, requires_grad=True)

    def extra_d3(label: str):
        component = new_dispersion(padded_nums, par, **DD)
        assert component is not None
        component.param = {**component.param, "s8": s8}
        component.label = label
        return component

    d3_a = new_dispersion(padded_nums, par, **DD)
    extra_cls = extra_d3("ExtraD3")
    unique_nums = torch.unique(padded_nums[padded_nums > 0])
    es2_a = new_es2(unique_nums, par, **DD)
    assert d3_a is not None and es2_a is not None

    calc_b = Calculator(
        padded_nums,
        par,
        classical=(d3_a, extra_cls),
        interaction=(es2_a,),
        opts={"verbosity": 0, "exclude": ["disp", "es2"], "batch_mode": 1},
        **DD,
    )
    extra_cls.label = "CallerOwnedD3"
    es2_a.label = "CallerOwnedES2"

    e_b = calc_b.energy(padded_pos)
    (g_s8,) = torch.autograd.grad(e_b.sum(), s8)
    assert torch.isfinite(g_s8) and g_s8.abs().item() > 0.0
    assert extra_cls.label == "CallerOwnedD3"
    assert es2_a.label == "CallerOwnedES2"
    assert d3_a.label == "DispersionD3"


def test_all_implemented_properties_dispatch_and_unsupported_batch_errors() -> None:
    """Every entry in implemented_properties dispatches cleanly and rejects unsupported batch modes."""
    from dxtb._src.calculators.types.autograd import AutogradCalculator
    from dxtb._src.calculators.types.numerical import NumericalCalculator

    assert set(Calculator.implemented_properties) == set(
        AutogradCalculator.implemented_properties
    )
    assert set(Calculator.implemented_properties) == set(
        NumericalCalculator.implemented_properties
    )

    # Use H2 (1D) and [H2, H+] (2D) for fast full-property verification
    num_1d = torch.tensor([1, 1], device=DEVICE)
    pos_1d = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **DD)
    num_2d = torch.tensor([[1, 1], [1, 0]], device=DEVICE)
    pos_2d = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        **DD,
    )
    chrg_2d = torch.tensor([0.0, 1.0], **DD)

    f_1d = torch.tensor([0.01, -0.01, 0.02], **DD, requires_grad=True)
    fg_1d = (torch.eye(3, **DD) * 0.01).requires_grad_(True)
    f_2d = torch.tensor(
        [[0.01, -0.01, 0.02], [-0.01, 0.02, -0.01]], **DD, requires_grad=True
    )
    fg_2d = torch.stack(
        [torch.eye(3, **DD) * 0.01, -torch.eye(3, **DD) * 0.01], dim=0
    ).requires_grad_(True)

    opts = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "differentiation": "unrolled",
        "strategy": scf_strategy("anderson"),
        "maxiter": 0,
        "fermi_etemp": 0.0,
    }
    calc_1d = Calculator(num_1d, GFN1_XTB, opts=opts, **DD)
    calc_2d = Calculator(num_2d, GFN1_XTB, opts=opts, **DD)

    # 1D: Every implemented property works through calculate, and representative
    # properties work through get_property and get_<prop>.
    all_1d = calc_1d.calculate(
        list(calc_1d.implemented_properties),
        pos_1d.clone().requires_grad_(True),
        use_functorch=False,
        field=f_1d,
        field_grad=fg_1d,
    )
    for prop in calc_1d.implemented_properties:
        assert prop in all_1d
        assert hasattr(calc_1d, f"get_{prop}")

    for prop in ("energy", "forces", "dipole", "polarizability", "hyperpolarizability"):
        # energy and forces do not take ``use_functorch``
        extra = {} if prop in ("energy", "forces") else {"use_functorch": False}
        val_gp = calc_1d.get_property(
            prop,
            pos_1d.clone().requires_grad_(True),
            field=f_1d,
            field_grad=fg_1d,
            **extra,
        )
        assert type(val_gp) is type(all_1d[prop])
        getter = getattr(calc_1d, f"get_{prop}")
        val_getter = getter(
            pos_1d.clone().requires_grad_(True),
            field=f_1d,
            field_grad=fg_1d,
            **extra,
        )
        assert val_getter is not None

    # 2D: Supported batched properties succeed; single-System-only properties raise NotImplementedError
    single_only_props = {
        "hessian",
        "vibration",
        "normal_modes",
        "frequencies",
        "ir",
        "ir_intensity",
        "ir_intensities",
        "raman",
        "raman_intensity",
        "raman_intensities",
        "rama_depol",
        "raman_depol",
    }
    supported_batch_props = [
        p for p in calc_2d.implemented_properties if p not in single_only_props
    ]
    all_2d = calc_2d.calculate(
        supported_batch_props,
        pos_2d.clone().requires_grad_(True),
        chrg=chrg_2d,
        use_functorch=False,
        field=f_2d,
        field_grad=fg_2d,
    )
    for prop in supported_batch_props:
        assert prop in all_2d

    for prop in single_only_props:
        with pytest.raises(NotImplementedError):
            calc_2d.get_property(
                prop,
                pos_2d.clone().requires_grad_(True),
                chrg=chrg_2d,
                use_functorch=True,
                field=f_2d,
                field_grad=fg_2d,
            )
        with pytest.raises(NotImplementedError):
            calc_2d.calculate(
                [prop],
                pos_2d.clone().requires_grad_(True),
                chrg=chrg_2d,
                use_functorch=True,
                field=f_2d,
                field_grad=fg_2d,
            )

    for num_method in (
        calc_2d.hessian_numerical,
        calc_2d.vibration_numerical,
        calc_2d.ir_numerical,
        calc_2d.raman_numerical,
    ):
        with pytest.raises(NotImplementedError):
            num_method(pos_2d, chrg=chrg_2d, field=f_2d, field_grad=fg_2d)


def test_derived_quantity_contract_explicit_validation() -> None:
    """derived_quantity must never be silently ignored when use_functorch=False."""
    numbers, positions = _lih_geometry()
    calc = Calculator(
        numbers,
        GFN1_XTB,
        opts={
            "int_driver": labels.INTDRIVER_PYTORCH,
            "int_level": labels.INTLEVEL_DIPOLE,
            "maxiter": 2,
            "fermi_etemp": 0.0,
            "verbosity": 0,
        },
        **DD,
    )
    field = torch.tensor([0.001, -0.002, 0.0015], **DD, requires_grad=True)

    # Supported use_functorch=False formulations succeed
    alpha_dip = calc.polarizability(
        positions, use_functorch=False, derived_quantity="dipole", field=field
    )
    chi_dip = calc.pol_deriv(
        positions.clone().requires_grad_(True),
        use_functorch=False,
        derived_quantity="dipole",
        field=field,
    )
    beta_pol = calc.hyperpolarizability(
        positions, use_functorch=False, derived_quantity="pol", field=field
    )
    beta_polarizability = calc.hyperpolarizability(
        positions,
        use_functorch=False,
        derived_quantity="polarizability",
        field=field,
    )
    assert alpha_dip.shape == (3, 3)
    assert chi_dip.shape == (3, 3, 2, 3)
    assert torch.allclose(beta_pol, beta_polarizability, atol=1e-12, rtol=1e-12)

    # Unsupported use_functorch=False formulations raise ValueError instead of silently ignoring derived_quantity
    with pytest.raises(ValueError, match="use_functorch=False"):
        calc.polarizability(
            positions,
            use_functorch=False,
            derived_quantity="energy",
            field=field,
        )

    with pytest.raises(ValueError, match="use_functorch=False"):
        calc.pol_deriv(
            positions.clone().requires_grad_(True),
            use_functorch=False,
            derived_quantity="energy",
            field=field,
        )

    for unsupported_dq in ("dipole", "energy"):
        with pytest.raises(ValueError, match="use_functorch=False"):
            calc.hyperpolarizability(
                positions,
                use_functorch=False,
                derived_quantity=unsupported_dq,  # type: ignore[arg-type]
                field=field,
            )

    # Unknown derived_quantity raises ValueError for both use_functorch=False and True
    for uf in (False, True):
        with pytest.raises(ValueError, match="Unknown `derived_quantity`"):
            calc.pol_deriv(
                positions.clone().requires_grad_(True),
                use_functorch=uf,
                derived_quantity="invalid",  # type: ignore[arg-type]
                field=field,
            )

    # Supported use_functorch=True formulations agree across derived_quantity options
    alpha_ft_dip = calc.polarizability(
        positions, use_functorch=True, derived_quantity="dipole", field=field
    )
    alpha_ft_ene = calc.polarizability(
        positions, use_functorch=True, derived_quantity="energy", field=field
    )
    assert torch.allclose(alpha_ft_dip, alpha_ft_ene, atol=1e-9, rtol=1e-8)

    chi_ft_dip = calc.pol_deriv(
        positions.clone().requires_grad_(True),
        use_functorch=True,
        derived_quantity="dipole",
        field=field,
    )
    chi_ft_ene = calc.pol_deriv(
        positions.clone().requires_grad_(True),
        use_functorch=True,
        derived_quantity="energy",
        field=field,
    )
    assert torch.allclose(chi_ft_dip, chi_ft_ene, atol=1e-8, rtol=1e-7)

