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
WP-INDEX-HARDEN regression tests for transform-safe structural validation,
meta schema separation, linear-time token snapshotting, and Dynamo contracts.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tad_mctc.tree import stack
from torch.func import jacrev, vmap
from torch.utils._pytree import tree_flatten, tree_unflatten

import dxtb._src.calculators.model as model_mod
from dxtb import GFN0_XTB, GFN2_XTB, IndexHelper, labels
from dxtb._src.calculators.lane_evaluation import (
    can_vmap_system,
    evaluate_lane_group,
    group_lanes,
)
from dxtb._src.calculators.model import Model, ModelParameterNode, System
from dxtb._src.calculators.singlepoint import build_integral_matrices
from dxtb._src.calculators.types.base import _convert_eval_system_dtype
from dxtb._src.integral.driver.pytorch import build_overlap, evaluate_moments
from dxtb._src.param import ParamModule
from dxtb._src.typing import Tensor
from dxtb.config import Config


def _build_water_systems() -> (
    tuple[Model, ParamModule, System, System, Tensor, Tensor]
):
    """Build same-shaped incompatible OH2 ([8, 1, 1]) and HOH ([1, 8, 1]) Systems."""
    nums_oh2 = torch.tensor([8, 1, 1], dtype=torch.int64)
    nums_hoh = torch.tensor([1, 8, 1], dtype=torch.int64)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
    )
    pos_batch = torch.stack([pos, pos * 1.02], dim=0)
    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    cfg = Config.create(
        method=labels.GFN2_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        maxiter=12,
    )
    model = Model(par=par, config=cfg)
    return (
        model,
        par,
        model.setup(nums_oh2),
        model.setup(nums_hoh),
        pos,
        pos_batch,
    )


def test_same_shape_index_helpers_eager_and_vmap_rejection() -> None:
    """Same-shaped incompatible IndexHelpers are rejected eagerly and under vmap."""
    _, par, sys_oh2, sys_hoh, _, pos_batch = _build_water_systems()

    for name in sys_oh2.ihelp._child_names:
        ta = getattr(sys_oh2.ihelp, name)
        tb = getattr(sys_hoh.ihelp, name)
        assert ta.shape == tb.shape
        assert ta.dtype is tb.dtype
        assert ta.device == tb.device
    assert sys_oh2.ihelp.batch_mode == sys_hoh.ihelp.batch_mode == 0
    assert not torch.equal(
        sys_oh2.ihelp.atom_to_unique, sys_hoh.ihelp.atom_to_unique
    )

    assert model_mod._index_helpers_equal(sys_oh2.ihelp, sys_hoh.ihelp) is False
    equiv_oh2 = IndexHelper.from_numbers(sys_oh2.numbers.clone(), par)
    assert model_mod._index_helpers_equal(sys_oh2.ihelp, equiv_oh2) is True

    ih_stack_bad = stack((sys_hoh.ihelp, sys_hoh.ihelp))
    ih_stack_same = stack((sys_oh2.ihelp, sys_oh2.ihelp))

    def check_wrapped_equality(
        ih_bad: IndexHelper, ih_same: IndexHelper, p: Tensor
    ) -> Tensor:
        assert model_mod._index_helpers_equal(ih_bad, sys_oh2.ihelp) is False
        assert model_mod._index_helpers_equal(sys_oh2.ihelp, ih_bad) is False
        assert model_mod._index_helpers_equal(ih_bad, ih_same) is False
        assert model_mod._index_helpers_equal(ih_same, ih_same) is False
        return p.sum()

    torch.func.vmap(check_wrapped_equality)(
        ih_stack_bad, ih_stack_same, pos_batch
    )


def test_vmap_system_core_and_tampered_system_paths() -> None:
    """Transformed tampered cores decline reuse; tampered Systems raise ValueError."""
    _, _, sys_oh2, sys_hoh, pos, pos_batch = _build_water_systems()
    ref_e = sys_oh2.energy(pos)
    ih_stack_bad = stack((sys_hoh.ihelp, sys_hoh.ihelp))
    nums_stack_oh2 = torch.stack([sys_oh2.numbers, sys_oh2.numbers], dim=0)

    def eval_with_tampered_core(ih_bad: IndexHelper, p: Tensor) -> Tensor:
        assert sys_oh2.exact_core is not None
        tampered = replace(
            sys_oh2, exact_core=sys_oh2.exact_core.replace(index=ih_bad)
        )
        assert tampered._active_exact_core() is None
        return tampered.energy(p)

    energies = torch.func.vmap(eval_with_tampered_core)(
        ih_stack_bad, torch.stack([pos, pos], dim=0)
    )
    torch.testing.assert_close(energies[0], ref_e, atol=1e-12, rtol=1e-12)

    pre_tampered = replace(
        sys_oh2, ihelp=sys_hoh.ihelp, exact_core=None, _setup_index_token=None
    )
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        torch.func.vmap(pre_tampered.energy)(pos_batch)

    def check_transformed_tampered_system(
        ih_bad: IndexHelper, n_wrapped: Tensor, p: Tensor
    ) -> Tensor:
        s_wrapped_ih = replace(
            sys_oh2, ihelp=ih_bad, exact_core=None, _setup_index_token=None
        )
        assert s_wrapped_ih._ihelp_compatible() is False
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            s_wrapped_ih.energy(p)

        s_wrapped_nums = replace(
            sys_oh2,
            numbers=n_wrapped,
            ihelp=sys_hoh.ihelp,
            exact_core=None,
            _setup_index_token=None,
        )
        assert s_wrapped_nums._ihelp_compatible() is False
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            s_wrapped_nums.energy(p)
        return p.sum()

    torch.func.vmap(check_transformed_tampered_system)(
        ih_stack_bad, nums_stack_oh2, pos_batch
    )

    # Grouped lane evaluation contract
    assert can_vmap_system(sys_oh2) is True
    groups = group_lanes(
        (sys_oh2, sys_oh2), (None, None), (None, None), (None, None)
    )
    assert len(groups) == 1 and groups[0].use_vmap is True
    chrg0 = torch.tensor(0.0, dtype=pos.dtype)
    lane_res = evaluate_lane_group(
        groups[0],
        (pos, pos),
        (chrg0, chrg0),
        (None, None),
        (None, None),
        (None, None),
    )
    torch.testing.assert_close(
        lane_res[0].energy.sum(), ref_e, atol=1e-12, rtol=1e-12
    )

    assert sys_oh2.exact_core is not None
    core_tampered = replace(
        sys_oh2, exact_core=sys_oh2.exact_core.replace(index=sys_hoh.ihelp)
    )
    assert can_vmap_system(core_tampered) is False
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        group_lanes(
            (pre_tampered, pre_tampered),
            (None, None),
            (None, None),
            (None, None),
        )


def test_ad_transforms_nested_wrappers_and_setup_functional() -> None:
    """Valid AD transforms succeed while nested/AD-wrapped mismatches fail."""
    _, _, sys_oh2, sys_hoh, pos, pos_batch = _build_water_systems()
    ref_e = sys_oh2.energy(pos)

    vmap_e = torch.func.vmap(sys_oh2.energy)(pos_batch)
    torch.testing.assert_close(vmap_e[0], ref_e, atol=1e-12, rtol=1e-12)
    _, jvp_val = torch.func.jvp(sys_oh2.energy, (pos,), (torch.ones_like(pos),))
    grad_rev = torch.func.jacrev(sys_oh2.energy)(pos)
    grad_fwd = torch.func.jacfwd(sys_oh2.energy)(pos)
    torch.testing.assert_close(grad_rev, grad_fwd, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(grad_rev.sum(), jvp_val, atol=1e-9, rtol=1e-9)
    batch_grad = torch.func.vmap(torch.func.jacrev(sys_oh2.energy))(pos_batch)
    torch.testing.assert_close(batch_grad[0], grad_rev, atol=1e-11, rtol=1e-11)

    # Nested vmap -> jacrev: outer layer is GradTrackingTensor, inner is BatchedTensor
    def outer_vmap_nested(lane_map: Tensor, p: Tensor) -> Tensor:
        def inner_grad(x: Tensor) -> Tensor:
            nested_map = lane_map + 0
            nested_ih = sys_oh2.ihelp.replace(atom_to_unique=nested_map)
            assert (
                model_mod._index_helpers_equal(nested_ih, sys_oh2.ihelp)
                is False
            )
            bad = replace(
                sys_oh2,
                ihelp=nested_ih,
                exact_core=None,
                _setup_index_token=None,
            )
            assert bad._ihelp_compatible() is False
            return (x * 2.0).sum()

        return torch.func.jacrev(inner_grad)(p)

    torch.func.vmap(outer_vmap_nested)(
        torch.stack(
            [sys_hoh.ihelp.atom_to_unique, sys_hoh.ihelp.atom_to_unique]
        ),
        pos_batch,
    )

    # Model.setup_functional under jacrev, jacfwd, jvp, and nested vmap(energy)
    p_mod = ParamModule(GFN2_XTB, dtype=torch.float64)
    p_mod.set_differentiable("element.O.slater")
    cfg = sys_oh2.config
    m_local = Model(par=p_mod, config=cfg)
    base_ptree = m_local.parameter_tree
    target_orig = p_mod.get("element.O.slater")
    base_slater = target_orig.detach().clone()
    leaves_orig, spec_orig = tree_flatten(base_ptree)

    def energy_from_slater(o_slater: Tensor) -> Tensor:
        new_leaves = [
            o_slater if leaf is target_orig else leaf for leaf in leaves_orig
        ]
        ptree: ModelParameterNode = tree_unflatten(new_leaves, spec_orig)
        s_valid = m_local.setup_functional(sys_oh2.numbers, ptree)
        s_mismatch = m_local.setup_functional(sys_hoh.numbers, ptree)

        assert (
            model_mod._index_helpers_equal(s_valid.ihelp, sys_oh2.ihelp) is True
        )
        assert (
            model_mod._index_helpers_equal(s_mismatch.ihelp, sys_oh2.ihelp)
            is False
        )
        bad_ad = replace(
            s_valid,
            ihelp=s_mismatch.ihelp,
            exact_core=None,
            _setup_index_token=None,
        )
        assert bad_ad._ihelp_compatible() is False
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            bad_ad.energy(pos)
        return torch.func.vmap(s_valid.energy)(pos_batch).sum()

    g_rev = torch.func.jacrev(energy_from_slater)(base_slater)
    g_fwd = torch.func.jacfwd(energy_from_slater)(base_slater)
    _, jvp_slater = torch.func.jvp(
        energy_from_slater, (base_slater,), (torch.ones_like(base_slater),)
    )
    torch.testing.assert_close(g_rev, g_fwd, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(g_rev.sum(), jvp_slater, atol=1e-9, rtol=1e-9)


def test_meta_schema_vs_concrete_equivalence_contract() -> None:
    """Meta tensors support only explicit schema checks, never value equivalence."""
    _, _, sys_oh2, sys_hoh, _, _ = _build_water_systems()
    meta_a = sys_oh2.ihelp.to(device="meta")
    meta_b = sys_hoh.ihelp.to(device="meta")

    assert model_mod._index_helpers_equal(meta_a, meta_a) is False
    assert model_mod._index_helpers_equal(meta_a, meta_b) is False

    assert sys_oh2.exact_core is not None
    meta_core = sys_oh2.exact_core.to(device="meta")
    meta_sys = replace(
        sys_oh2,
        numbers=meta_core.numbers,
        ihelp=meta_core.index,
        integral_setup=meta_core.integral,
        h0_setup=meta_core.h0_setup,
        es2_setup=(
            sys_oh2.es2_setup.to(device="meta")
            if sys_oh2.es2_setup is not None
            else None
        ),
        es3_setup=(
            sys_oh2.es3_setup.to(device="meta")
            if sys_oh2.es3_setup is not None
            else None
        ),
        exact_core=meta_core,
        _setup_index_token=None,
    )
    assert meta_sys._ihelp_schema_compatible() is True
    assert meta_sys._ihelp_compatible() is False
    assert meta_sys._active_exact_core() is None

    bad_meta_shape = replace(
        meta_sys,
        numbers=torch.empty((4,), dtype=torch.int64, device="meta"),
    )
    assert bad_meta_shape._ihelp_schema_compatible() is False


def test_inference_mode_and_snapshot_tensor_states_invariants() -> None:
    """Verify linear-time _snapshot_tensor_states invariants and inference_mode."""
    model, par, sys_oh2, _, pos, _ = _build_water_systems()
    ref_e = sys_oh2.energy(pos)

    t0 = torch.tensor([1, 2, 3], dtype=torch.int64)
    t1 = torch.tensor([4, 5], dtype=torch.int64)
    t0_view = t0[:]  # distinct Tensor sharing storage with t0

    states = model_mod._snapshot_tensor_states([t0, t1, t0, t0_view, t1])
    assert len(states) == 3
    assert states[0][0] is t0
    assert states[1][0] is t1
    assert states[2][0] is t0_view
    assert all(s[1] is not None for s in states)

    # Functorch-wrapped tensor records ver=None
    def check_wrapped_snapshot(x: Tensor) -> Tensor:
        st = model_mod._snapshot_tensor_states([x])
        assert len(st) == 1 and st[0][0] is x and st[0][1] is None
        return x.sum()

    torch.func.jvp(
        check_wrapped_snapshot,
        (torch.ones(2, dtype=torch.float64),),
        (torch.ones(2, dtype=torch.float64),),
    )

    # Inference-mode setup and evaluation
    with torch.inference_mode():
        inf_t = torch.tensor([8, 1, 1], dtype=torch.int64)
        inf_st = model_mod._snapshot_tensor_states([inf_t])
        assert len(inf_st) == 1 and inf_st[0][1] is None

        sys_inf = model.setup(inf_t)
        assert sys_inf._has_intact_setup_token() is False
        assert sys_inf._ihelp_compatible() is True
        e_inf = sys_inf.singlepoint(pos.clone()).energy.sum()
        torch.testing.assert_close(e_inf, ref_e, atol=1e-12, rtol=1e-12)

        bad_inf = replace(
            sys_inf,
            ihelp=IndexHelper.from_numbers(
                torch.tensor([1, 8, 1], dtype=torch.int64), par
            ),
        )
        assert bad_inf._ihelp_compatible() is False

    # Dtype conversion restamps intact token once and preserves zero autograd ownership
    conv_sys = _convert_eval_system_dtype(sys_oh2, torch.float32)
    assert conv_sys._has_intact_setup_token() is True
    tok = conv_sys._setup_index_token
    assert tok is not None
    for snap in (
        tok.numbers_snapshot,
        tok.unique_angular_snapshot,
        tok.ushells_per_unique_snapshot,
    ):
        assert snap is not None
        assert snap.requires_grad is False and snap.grad_fn is None
    for leaf, _, _, _, _ in tok.tensor_states:
        assert leaf.requires_grad is False and leaf.grad_fn is None

    # In-place mutation detection invalidates token
    mut_sys = model.setup(sys_oh2.numbers.clone())
    assert mut_sys._has_intact_setup_token() is True
    assert mut_sys.integral_setup is not None
    assert mut_sys.integral_setup.pytorch is not None
    mut_sys.integral_setup.pytorch.pair_plan.orbitals_to_atom[0] = 2
    assert mut_sys._has_intact_setup_token() is False
    assert mut_sys._ihelp_compatible() is False


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is not supported for this Python/torch combination",
)
def test_dynamo_fullgraph_kernels() -> None:
    """
    Pure integral/multipole kernels compile fullgraph.

    The compiled ``System.energy`` and its post-compilation mutation check
    are covered by ``test_scf_index_closure.py::
    test_scf_dynamo_capture_and_post_compile_mutation_guard``.
    """
    _, _, sys_oh2, _, pos, _ = _build_water_systems()
    assert sys_oh2.integral_setup is not None
    assert sys_oh2.integral_setup.pytorch is not None
    assert sys_oh2.h0_setup is not None
    pt_setup = sys_oh2.integral_setup.pytorch
    chrg0 = torch.tensor(0.0, dtype=pos.dtype)

    def fn_ovlp(p: Tensor) -> Tensor:
        return build_overlap(pt_setup, p)

    def fn_mom(p: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return evaluate_moments(pt_setup, p, labels.INTLEVEL_QUADRUPOLE)

    def fn_int_mats(p: Tensor) -> tuple[Tensor, Tensor]:
        assert sys_oh2.integral_setup is not None
        assert sys_oh2.h0_setup is not None
        mats, _ = build_integral_matrices(
            sys_oh2.integral_setup, sys_oh2.h0_setup, sys_oh2.ihelp, p, chrg0
        )
        return mats.overlap, mats.hcore

    torch._dynamo.reset()
    torch.testing.assert_close(
        torch.compile(fn_ovlp, backend="aot_eager", fullgraph=True)(pos),
        fn_ovlp(pos),
    )
    comp_mom = torch.compile(fn_mom, backend="aot_eager", fullgraph=True)(pos)
    ref_mom = fn_mom(pos)
    for actual, expected in zip(comp_mom, ref_mom):
        torch.testing.assert_close(actual, expected)

    comp_mats = torch.compile(fn_int_mats, backend="aot_eager", fullgraph=True)(
        pos
    )
    ref_mats = fn_int_mats(pos)
    for actual, expected in zip(comp_mats, ref_mats):
        torch.testing.assert_close(actual, expected)


def test_converged_root_batch_and_streaming_eval() -> None:
    """Implicit batch preparation/evaluation and streaming batch pass validated core."""
    from dxtb._src.calculators.batch_implicit import (
        evaluate_converged_root_batch,
        prepare_converged_root_batch,
    )
    from dxtb._src.calculators.batch_streaming import (
        stream_batch_value_and_grad,
    )
    from dxtb.config import ConfigSCF

    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    scf_cfg = ConfigSCF.create(
        differentiation="implicit",
        maxiter=50,
        x_atol=1.0e-10,
        x_atol_max=1.0e-10,
    )
    cfg = replace(
        Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            fermi_etemp=0.0,
        ),
        scf=scf_cfg,
    )
    model = Model(par=par, config=cfg)
    numbers = torch.tensor([8, 1, 1])
    sys_oh2 = model.setup(numbers)
    sys_hoh = model.setup(torch.tensor([1, 8, 1]))

    _, _, _, _, pos0, _ = _build_water_systems()
    batch_pos = torch.stack((pos0, pos0 * 1.02)).requires_grad_(True)

    prepared = prepare_converged_root_batch(sys_oh2, batch_pos.detach())
    evaluated = evaluate_converged_root_batch(prepared, batch_pos)
    (grad_pos,) = torch.autograd.grad(evaluated.energy.sum(), batch_pos)
    assert torch.isfinite(evaluated.energy).all()
    assert torch.isfinite(grad_pos).all()

    streamed = stream_batch_value_and_grad(
        prepared, model, batch_pos.detach(), chunk_size=1
    )
    torch.testing.assert_close(
        streamed.energy.sum(),
        evaluated.energy.detach().sum(),
        atol=1e-9,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        streamed.position_gradient, grad_pos, atol=1e-8, rtol=1e-8
    )

    tampered = replace(sys_oh2, ihelp=sys_hoh.ihelp, exact_core=None)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        prepare_converged_root_batch(tampered, batch_pos.detach())


def test_batched_modes_concrete_wrapped_and_mutated_tampering() -> None:
    """Padded (batch_mode=1) and conformer (batch_mode=2) Systems validate concrete indices."""
    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    model = Model(
        par=par,
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            maxiter=8,
        ),
    )

    # 1. Conformer batch (batch_mode=2): concrete, AD-wrapped (GradTrackingTensor), and vmap.
    nums_oh2_b = torch.tensor([[8, 1, 1], [8, 1, 1]])
    nums_hoh_b = torch.tensor([[1, 8, 1], [1, 8, 1]])
    sys_b2 = model.setup(nums_oh2_b.clone(), batch_mode=2)
    sys_hoh_b2 = model.setup(nums_hoh_b.clone(), batch_mode=2)

    assert sys_b2._has_intact_setup_token() is True
    assert sys_b2._ihelp_schema_compatible() is True
    assert sys_b2._ihelp_compatible() is True

    equiv_b2_ih = IndexHelper.from_numbers(nums_oh2_b, par, batch_mode=2)
    assert replace(sys_b2, ihelp=equiv_b2_ih)._ihelp_compatible() is True
    assert (
        replace(
            sys_b2, ihelp=equiv_b2_ih, _setup_index_token=None
        )._ihelp_compatible()
        is True
    )

    # Concrete same-shaped incompatible conformer IndexHelper -> False
    for clear_tok in (False, True):
        bad_b2 = replace(
            sys_b2,
            ihelp=sys_hoh_b2.ihelp,
            _setup_index_token=None if clear_tok else sys_b2._setup_index_token,
        )
        assert bad_b2._ihelp_schema_compatible() is True
        assert bad_b2._ihelp_compatible() is False

    # AD-wrapped (GradTrackingTensor under jacrev) mismatched conformer IndexHelper -> False
    captured_ad: list[bool] = []

    def probe_ad(x: Tensor) -> Tensor:
        z_zero = (x.sum() * 0).to(torch.int64)
        wrapped_nums = nums_oh2_b + z_zero
        wrapped_bad_ih = IndexHelper(
            **{
                name: getattr(sys_hoh_b2.ihelp, name) + z_zero
                for name in sys_hoh_b2.ihelp._child_names
            },
            batch_mode=2,
        )
        wrapped_good_ih = IndexHelper(
            **{
                name: getattr(sys_b2.ihelp, name) + z_zero
                for name in sys_b2.ihelp._child_names
            },
            batch_mode=2,
        )
        bad_wrapped_sys = replace(
            sys_b2,
            numbers=wrapped_nums,
            ihelp=wrapped_bad_ih,
            _setup_index_token=None,
        )
        good_wrapped_sys = replace(
            sys_b2,
            numbers=wrapped_nums,
            ihelp=wrapped_good_ih,
            _setup_index_token=None,
        )
        captured_ad.append(bad_wrapped_sys._ihelp_compatible())
        captured_ad.append(good_wrapped_sys._ihelp_compatible())
        return x * 2.0

    _ = jacrev(probe_ad)(torch.tensor([1.0], dtype=torch.float64))
    assert captured_ad == [False, True]

    # vmap-wrapped BatchedTensor conformer IndexHelper -> False
    stacked_b2_ih = stack((sys_hoh_b2.ihelp, sys_b2.ihelp))
    captured_vmap: list[bool] = []

    def probe_vmap_b2(ih: IndexHelper, dummy: Tensor) -> Tensor:
        cand = replace(sys_b2, ihelp=ih, _setup_index_token=None)
        captured_vmap.append(cand._ihelp_compatible())
        return dummy * 2.0

    _ = vmap(probe_vmap_b2)(stacked_b2_ih, torch.ones(2))
    assert captured_vmap == [False]

    # 2. Padded batch (batch_mode=1): valid Z=0 padding, in-place mutation, and recovery.
    nums_pad = torch.tensor([[8, 1, 1], [6, 1, 0]])
    sys_b1 = model.setup(nums_pad.clone(), batch_mode=1)
    assert sys_b1._has_intact_setup_token() is True
    assert sys_b1._ihelp_compatible() is True
    assert replace(sys_b1, _setup_index_token=None)._ihelp_compatible() is True

    sys_b1.ihelp.shells_to_atom[0, 0] = 2
    try:
        assert sys_b1._has_intact_setup_token() is False
        assert sys_b1._ihelp_compatible() is False
    finally:
        sys_b1.ihelp.shells_to_atom[0, 0] = 0
    assert sys_b1._ihelp_compatible() is True

    # 3. Cross-field shape-boundary concatenation collision (S=2, O=8 vs S'=3, O'=3, len=30).
    sys_h3 = model.setup(torch.tensor([[1, 1, 1]]), batch_mode=2)
    assert sys_h3.h0_setup is not None
    corrupted_ihelp = IndexHelper(
        unique_angular=torch.tensor([0]),
        angular=torch.tensor([[0, 0]]),
        atom_to_unique=torch.tensor([[0, 0, 0]]),
        ushells_to_unique=torch.tensor([0]),
        ushells_per_unique=torch.tensor([0]),
        shells_to_ushell=torch.tensor([[1, 0]]),
        shells_per_atom=torch.tensor([[0, 0, 1]]),
        shell_index=torch.tensor([[1, 1, 0]]),
        shells_to_atom=torch.tensor([[1, 2]]),
        orbitals_per_shell=torch.tensor([[0, 1]]),
        orbital_index=torch.tensor([[2, 1]]),
        orbitals_to_shell=torch.tensor([[1, 1, 0, 1, 2, 0, 1, 2]]),
        batch_mode=2,
    )
    assert (
        model_mod._eager_index_helpers_equal(corrupted_ihelp, sys_h3.ihelp)
        is False
    )
    assert (
        model_mod._eager_index_helpers_equal(
            sys_h3.ihelp, sys_h3.ihelp.to(device="meta")
        )
        is False
    )
    tampered_h3 = replace(
        sys_h3,
        ihelp=corrupted_ihelp,
        h0_setup=replace(sys_h3.h0_setup, ihelp=corrupted_ihelp),
        _setup_index_token=None,
    )
    assert tampered_h3._ihelp_compatible() is False

    # 4. Empty numbers (0,), (0, 3), (2, 0) return False cleanly without IndexError.
    for empty_shape, bmode in (((0,), 0), ((0, 3), 1), ((2, 0), 1)):
        e_nums = torch.empty(empty_shape, dtype=torch.int64)
        s_shape = (0,) if bmode == 0 else (empty_shape[0], 0)
        empty_ih = IndexHelper(
            unique_angular=torch.empty((0,), dtype=torch.int64),
            angular=torch.empty(s_shape, dtype=torch.int64),
            atom_to_unique=torch.empty(empty_shape, dtype=torch.int64),
            ushells_to_unique=torch.empty((0,), dtype=torch.int64),
            ushells_per_unique=torch.empty((0,), dtype=torch.int64),
            shells_to_ushell=torch.empty(s_shape, dtype=torch.int64),
            shells_per_atom=torch.empty(empty_shape, dtype=torch.int64),
            shell_index=torch.empty(empty_shape, dtype=torch.int64),
            shells_to_atom=torch.empty(s_shape, dtype=torch.int64),
            orbitals_per_shell=torch.empty(s_shape, dtype=torch.int64),
            orbital_index=torch.empty(s_shape, dtype=torch.int64),
            orbitals_to_shell=torch.empty(s_shape, dtype=torch.int64),
            batch_mode=bmode,
        )
        empty_sys = replace(
            sys_b1,
            numbers=e_nums,
            ihelp=empty_ih,
            batch_mode=bmode,
            h0_setup=None,
            integral_setup=None,
            es2_setup=None,
            es3_setup=None,
            exact_core=None,
            _setup_index_token=None,
        )
        assert empty_sys._ihelp_schema_compatible() is False
        assert empty_sys._ihelp_compatible() is False


def test_es3_partial_meta_and_dispatch_mode_guards() -> None:
    """Reject malformed ES3 shapes, partial-meta setup leaves, and slow-path dispatch modes."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from dxtb._src.calculators.model import _unwrap_concrete_structural_tensor

    _, _, sys_oh2, _, pos, _ = _build_water_systems()
    assert sys_oh2.exact_core is not None
    assert sys_oh2.es2_setup is not None
    assert sys_oh2.es3_setup is not None
    assert sys_oh2.h0_setup is not None
    assert sys_oh2.integral_setup is not None
    assert sys_oh2.integral_setup.pytorch is not None

    # 0. ES2Setup float shells_to_atom via unvalidated tree_unflatten.
    es2 = sys_oh2.es2_setup
    assert es2.shells_to_atom is not None
    es2_leaves, es2_spec = tree_flatten(es2)
    float_shells = es2.shells_to_atom.to(torch.float64) + 0.4
    bad_es2 = tree_unflatten(
        [
            float_shells if leaf is es2.shells_to_atom else leaf
            for leaf in es2_leaves
        ],
        es2_spec,
    )
    for clear_tok in (False, True):
        cand_es2 = replace(
            sys_oh2,
            es2_setup=bad_es2,
            exact_core=None,
            _setup_index_token=(
                None if clear_tok else sys_oh2._setup_index_token
            ),
        )
        assert cand_es2._ihelp_schema_compatible() is False
        assert cand_es2._ihelp_compatible() is False
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            cand_es2.singlepoint(pos)

    # 0b. batch_mode=0 active-species negative unique_angular (angular={}) & GFN0 kq shape.
    classical_model = Model(
        par=ParamModule(GFN2_XTB, dtype=torch.float64),
        config=Config.create(
            method=labels.GFN2_XTB,
            int_level=labels.INTLEVEL_NONE,
            exclude=["all"],
        ),
        auto_int_level=False,
    )
    sys_no_int = classical_model.setup(sys_oh2.numbers.clone())
    empty_ang_ih = IndexHelper.from_numbers_angular(
        sys_oh2.numbers.clone(), angular={}
    )
    assert (
        replace(
            sys_no_int,
            ihelp=empty_ang_ih,
            exact_core=None,
            _setup_index_token=None,
        )._ihelp_compatible()
        is False
    )

    gfn0_model = Model(
        par=ParamModule(GFN0_XTB, dtype=torch.float64),
        config=Config.create(
            method=labels.GFN0_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    )
    sys_gfn0 = gfn0_model.setup(sys_oh2.numbers.clone())
    assert sys_gfn0.h0_setup is not None and sys_gfn0.h0_setup.kq is not None
    kq_leaf = sys_gfn0.h0_setup.kq
    g0_leaves, g0_spec = tree_flatten(sys_gfn0.h0_setup)
    bad_g0_h0 = tree_unflatten(
        [
            torch.ones((1,), dtype=torch.float64) if leaf is kq_leaf else leaf
            for leaf in g0_leaves
        ],
        g0_spec,
    )
    cand_g0 = replace(sys_gfn0, h0_setup=bad_g0_h0, exact_core=None)
    assert cand_g0._ihelp_schema_compatible() is False
    assert cand_g0._ihelp_compatible() is False

    # 1. ES3Setup: 0D scalar (no IndexError!), 2D tensor, and unvalidated partial-meta leaf.
    es3 = sys_oh2.es3_setup
    n_expected = es3.hubbard_derivs.shape[0]
    _, es3_spec = tree_flatten(es3)
    core_leaves, core_spec = tree_flatten(sys_oh2.exact_core)
    for bad_hd in (
        torch.tensor(1.0, dtype=torch.float64),
        torch.ones((1, n_expected), dtype=torch.float64),
        es3.hubbard_derivs.to(device="meta"),
    ):
        bad_es3 = tree_unflatten([bad_hd], es3_spec)
        bad_core = tree_unflatten(
            [
                bad_hd if leaf is es3.hubbard_derivs else leaf
                for leaf in core_leaves
            ],
            core_spec,
        )
        cand = replace(
            sys_oh2,
            es3_setup=bad_es3,
            exact_core=bad_core,
        )
        assert cand._ihelp_schema_compatible() is False
        assert cand._ihelp_compatible() is False
        assert cand._active_exact_core() is None

    # 2. Partial-meta PytorchIntegralSetup.alphas via unvalidated tree_unflatten.
    pt_setup = sys_oh2.integral_setup.pytorch
    alpha0 = pt_setup.alphas[0]
    is_leaves, is_spec = tree_flatten(sys_oh2.integral_setup)
    bad_is = tree_unflatten(
        [
            leaf.to(device="meta") if leaf is alpha0 else leaf
            for leaf in is_leaves
        ],
        is_spec,
    )
    bad_pt_core = tree_unflatten(
        [
            leaf.to(device="meta") if leaf is alpha0 else leaf
            for leaf in core_leaves
        ],
        core_spec,
    )
    for clear_tok in (False, True):
        cand_pt = replace(
            sys_oh2,
            integral_setup=bad_is,
            exact_core=bad_pt_core,
            _setup_index_token=(
                None if clear_tok else sys_oh2._setup_index_token
            ),
        )
        assert cand_pt._ihelp_schema_compatible() is False
        assert cand_pt._ihelp_compatible() is False
        assert cand_pt._active_exact_core() is None

    # 3. Partial-meta H0SetupNode.selfenergy via unvalidated tree_unflatten.
    h0_leaves, h0_spec = tree_flatten(sys_oh2.h0_setup)
    selfenergy_leaf = sys_oh2.h0_setup.selfenergy
    bad_h0 = tree_unflatten(
        [
            leaf.to(device="meta") if leaf is selfenergy_leaf else leaf
            for leaf in h0_leaves
        ],
        h0_spec,
    )
    bad_h0_core = tree_unflatten(
        [
            leaf.to(device="meta") if leaf is selfenergy_leaf else leaf
            for leaf in core_leaves
        ],
        core_spec,
    )
    for clear_tok in (False, True):
        cand_h0 = replace(
            sys_oh2,
            h0_setup=bad_h0,
            exact_core=bad_h0_core,
            _setup_index_token=(
                None if clear_tok else sys_oh2._setup_index_token
            ),
        )
        assert cand_h0._ihelp_schema_compatible() is False
        assert cand_h0._ihelp_compatible() is False
        assert cand_h0._active_exact_core() is None

    # 4. Active TorchDispatchMode: intact fast token still succeeds with zero ops,
    # while _unwrap_concrete_structural_tensor declines concrete slow-path ops.
    class _NoOpDispatchMode(TorchDispatchMode):
        def __init__(self) -> None:
            self.ops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops += 1
            return func(*args, **(kwargs or {}))

    no_tok_sys = replace(sys_oh2, _setup_index_token=None)
    mode = _NoOpDispatchMode()
    with mode:
        assert _unwrap_concrete_structural_tensor(sys_oh2.numbers) is None
        assert sys_oh2._ihelp_compatible() is True
        assert mode.ops == 0
        assert no_tok_sys._ihelp_compatible() is False
