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
WP-SCF-INDEX-CLOSURE tests for single canonical SCF IndexHelper ownership,
retired dual-index plumbing, and standalone vs System-level validation.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, replace

import pytest
import torch
from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.tree import stack
from torch.func import jacfwd, jacrev, jvp, vmap

import dxtb._src.scf as scf_pkg
import dxtb._src.scf.data as scf_data
import dxtb._src.scf.eager as scf_eager
import dxtb._src.scf.iterator as scf_iterator
from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, Calculator, IndexHelper, labels
from dxtb._src.calculators.batch_implicit import (
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.calculators.model import Model, System
from dxtb._src.calculators.packed_result import evaluate_packed_systems
from dxtb._src.calculators.singlepoint import (
    _interaction_data,
    _PreparedSinglepoint,
    system_singlepoint,
)
from dxtb._src.calculators.types.base import _convert_eval_system_dtype
from dxtb._src.components.interactions import InteractionList
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.param import ParamModule
from dxtb._src.scf.data import SCFInputs, setup_scf_inputs
from dxtb._src.scf.iterator import get_refocc, solve
from dxtb._src.scf.map import scf_map
from dxtb._src.typing import Tensor
from dxtb.config import Config, ConfigSCF


def _water_fixtures() -> tuple[Model, ParamModule, System, System, Tensor]:
    """Return GFN2 Model, ParamModule, OH2/HOH Systems, and geometry."""
    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    cfg = Config.create(
        method=labels.GFN2_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        maxiter=30,
    )
    model = Model(par=par, config=cfg)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
    )
    sys_oh2 = model.setup(torch.tensor([8, 1, 1], dtype=torch.int64))
    sys_hoh = model.setup(torch.tensor([1, 8, 1], dtype=torch.int64))
    return model, par, sys_oh2, sys_hoh, pos


def test_dual_index_plumbing_removed_and_single_index_signatures() -> None:
    """Verify core_index/core_index_owner/solve_with_core_index are removed."""
    assert not hasattr(scf_pkg, "solve_with_core_index")
    assert not hasattr(scf_iterator, "solve_with_core_index")
    assert not hasattr(scf_iterator, "_solve_impl")

    prepared_field_names = {f.name for f in fields(_PreparedSinglepoint)}
    assert "ihelp" in prepared_field_names
    assert "core_index" not in prepared_field_names
    assert "core_index_owner" not in prepared_field_names

    for fn in (
        setup_scf_inputs,
        solve,
        scf_eager._prepared_scf_inputs,
        scf_eager.run_prepared_scf,
        scf_eager.prepare_system_fixed_trip,
        scf_eager.run_system_fixed_trip,
        scf_eager.run_prepared_implicit_scf,
    ):
        param_names = set(inspect.signature(fn).parameters)
        assert "ihelp" in param_names
        assert "core_index" not in param_names
        assert "core_index_owner" not in param_names


def test_single_index_identity_and_single_check_per_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify scf_inputs.index is system.ihelp with one check per call."""
    _, par, sys_oh2, sys_hoh, pos = _water_fixtures()
    captured_inputs: list[SCFInputs] = []
    orig_setup = scf_data.setup_scf_inputs

    def spy_setup(*args, **kwargs) -> SCFInputs:
        out = orig_setup(*args, **kwargs)
        captured_inputs.append(out)
        return out

    monkeypatch.setattr(scf_data, "setup_scf_inputs", spy_setup)
    monkeypatch.setattr(scf_eager, "setup_scf_inputs", spy_setup)

    compat_calls = 0
    orig_compat = System._ihelp_compatible

    def count_compat(self: System) -> bool:
        nonlocal compat_calls
        compat_calls += 1
        return orig_compat(self)

    monkeypatch.setattr(System, "_ihelp_compatible", count_compat)

    # 1. Normal System: eager singlepoint & fixed-trip energy each check once
    compat_calls = 0
    captured_inputs.clear()
    res_sp = sys_oh2.singlepoint(pos)
    assert compat_calls == 1
    assert len(captured_inputs) == 1
    assert captured_inputs[0].index is sys_oh2.ihelp

    compat_calls = 0
    captured_inputs.clear()
    e_ref = sys_oh2.energy(pos)
    assert compat_calls == 1
    assert len(captured_inputs) == 1
    assert captured_inputs[0].index is sys_oh2.ihelp
    torch.testing.assert_close(
        res_sp.energy.sum(), e_ref, atol=1e-12, rtol=1e-12
    )

    # 2. Equivalent replacement IndexHelper -> scf_inputs.index is equiv_ihelp
    equiv_ihelp = IndexHelper.from_numbers(sys_oh2.numbers.clone(), par)
    assert equiv_ihelp is not sys_oh2.ihelp
    equiv_sys = replace(sys_oh2, ihelp=equiv_ihelp)
    assert equiv_sys._active_exact_core() is sys_oh2.exact_core
    captured_inputs.clear()
    e_equiv = equiv_sys.energy(pos)
    assert len(captured_inputs) == 1
    assert captured_inputs[0].index is equiv_ihelp
    torch.testing.assert_close(e_equiv, e_ref, atol=1e-12, rtol=1e-12)

    # 5. Tampered ExactSystemCore.index -> falls back to canonical system.ihelp
    assert sys_oh2.exact_core is not None
    tampered_core_sys = replace(
        sys_oh2, exact_core=sys_oh2.exact_core.replace(index=sys_hoh.ihelp)
    )
    assert tampered_core_sys._active_exact_core() is None
    captured_inputs.clear()
    e_fallback = tampered_core_sys.singlepoint(pos).energy.sum()
    assert len(captured_inputs) == 1
    assert captured_inputs[0].index is sys_oh2.ihelp
    torch.testing.assert_close(e_fallback, e_ref, atol=1e-12, rtol=1e-12)

    # 8. Dtype conversion -> single canonical index & 1 compatibility check
    conv_sys = _convert_eval_system_dtype(sys_oh2, torch.float32)
    compat_calls = 0
    captured_inputs.clear()
    res_f32 = conv_sys.singlepoint(pos.to(torch.float32))
    assert compat_calls == 1
    assert len(captured_inputs) == 1
    assert captured_inputs[0].index is conv_sys.ihelp
    assert res_f32.energy.dtype is torch.float32

    # Grouped-vmap & packed/conformer System + standalone scf.solve() identity
    captured_inputs.clear()
    grouped_res = evaluate_packed_systems((sys_oh2, sys_oh2), (pos, pos * 1.01))
    assert captured_inputs and all(
        inp.index is sys_oh2.ihelp for inp in captured_inputs
    )
    torch.testing.assert_close(
        grouped_res.energy[0].sum(), e_ref, atol=1e-11, rtol=1e-11
    )

    sys_h2 = Model(par=par, config=sys_oh2.config).setup(
        torch.tensor([1, 1], dtype=torch.int64)
    )
    captured_inputs.clear()
    packed_res = evaluate_packed_systems((sys_oh2, sys_h2), (pos, pos[:2]))
    assert [inp.index for inp in captured_inputs] == [
        sys_oh2.ihelp,
        sys_h2.ihelp,
    ]
    torch.testing.assert_close(
        packed_res.energy[0].sum(), e_ref, atol=1e-11, rtol=1e-11
    )

    assert sys_oh2.integral_setup is not None
    assert sys_oh2.h0_setup is not None
    standalone_ih = IndexHelper.from_numbers(sys_oh2.numbers.clone(), par)
    chrg0 = torch.tensor([0.0], dtype=torch.float64)
    st_mats, st_refocc = build_integral_matrices(
        sys_oh2.integral_setup, sys_oh2.h0_setup, standalone_ih, pos, chrg0
    )
    st_cache = sys_oh2.interactions.get_cache(
        numbers=sys_oh2.numbers, positions=pos, ihelp=standalone_ih
    )
    captured_inputs.clear()
    solve(
        sys_oh2.numbers,
        pos,
        chrg0,
        None,
        sys_oh2.interactions,
        st_cache,
        standalone_ih,
        sys_oh2.config.scf,
        st_mats,
        st_refocc,
    )
    assert len(captured_inputs) == 1
    assert captured_inputs[0].index is standalone_ih

    # Implicit batch prep & evaluation -> scf_inputs.index is system.ihelp
    monkeypatch.undo()
    imp_cfg = Config.create(
        method=labels.GFN2_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        differentiation="implicit",
        maxiter=50,
        x_atol=1.0e-10,
        x_atol_max=1.0e-10,
        fermi_etemp=0.0,
    )
    imp_sys = Model(par=par, config=imp_cfg).setup(sys_oh2.numbers.clone())
    batch_pos = torch.stack([pos, pos * 1.02]).requires_grad_(True)
    prepared = prepare_converged_root_batch(imp_sys, batch_pos.detach())
    for lane in prepared.lanes:
        assert lane.inputs.index is imp_sys.ihelp
    evaluated = evaluate_converged_root_batch(prepared, batch_pos)
    (grad_b,) = torch.autograd.grad(evaluated.energy.sum(), batch_pos)
    assert torch.isfinite(grad_b).all()


def test_adversarial_system_mutations_transforms_inference_and_compile() -> (
    None
):
    """Cover B7 items 3, 4, 6, 7, 9, 10, and 11 without silent bad physics."""
    model, par, sys_oh2, sys_hoh, pos = _water_fixtures()
    pos_batch = torch.stack([pos, pos * 1.02], dim=0)
    ref_e = sys_oh2.energy(pos)
    ref_g = jacrev(sys_oh2.energy)(pos)

    # 3. Same-shaped incompatible replacement (eager, vmap, jacrev, jacfwd, jvp)
    bad_sys = replace(sys_oh2, ihelp=sys_hoh.ihelp, exact_core=None)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        bad_sys.singlepoint(pos)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        vmap(bad_sys.energy)(pos_batch)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        jacrev(bad_sys.energy)(pos)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        jacfwd(bad_sys.energy)(pos)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        jvp(bad_sys.energy, (pos,), (torch.ones_like(pos),))

    # 4, 6, 7. In-place ihelp, PairPlan, and H0/ES2 setup tampering
    mut_ih = model.setup(sys_oh2.numbers.clone())
    mut_ih.ihelp.shell_index[0] = 77
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        mut_ih.singlepoint(pos)

    mut_plan = model.setup(sys_oh2.numbers.clone())
    assert mut_plan.integral_setup is not None
    assert mut_plan.integral_setup.pytorch is not None
    mut_plan.integral_setup.pytorch.pair_plan.orbitals_to_atom[0] = 2
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        mut_plan.singlepoint(pos)

    assert sys_hoh.h0_setup is not None
    bad_h0 = replace(sys_oh2, h0_setup=sys_hoh.h0_setup, exact_core=None)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        bad_h0.singlepoint(pos)

    assert sys_hoh.es2_setup is not None
    bad_es2 = replace(sys_oh2, es2_setup=sys_hoh.es2_setup, exact_core=None)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        bad_es2.singlepoint(pos)

    # 9. Batched/conformer Calculator & System compatibility
    calc_b2 = Calculator(
        torch.stack([sys_oh2.numbers, sys_oh2.numbers]),
        par,
        opts={"batch_mode": 2, "verbosity": 0, "maxiter": 30},
        dtype=torch.float64,
    )
    e_b2 = calc_b2.energy(pos_batch)
    torch.testing.assert_close(e_b2[0], ref_e, atol=1e-9, rtol=1e-9)

    sys_b2 = model.setup(
        torch.stack([sys_oh2.numbers, sys_oh2.numbers]), batch_mode=2
    )
    sys_b2_bad = model.setup(
        torch.stack([sys_hoh.numbers, sys_hoh.numbers]), batch_mode=2
    )
    assert replace(sys_b2, ihelp=sys_b2_bad.ihelp)._ihelp_compatible() is False

    # 10. Nested vmap + jacrev (valid parity & tampered rejection)
    nested_g = vmap(jacrev(sys_oh2.energy))(pos_batch)
    torch.testing.assert_close(nested_g[0], ref_g, atol=1e-10, rtol=1e-10)

    ih_stack_bad = stack((sys_hoh.ihelp, sys_hoh.ihelp))

    def outer_vmap(ih_bad: IndexHelper, p: Tensor) -> Tensor:
        def inner_fn(x: Tensor) -> Tensor:
            tampered = replace(
                sys_oh2, ihelp=ih_bad, exact_core=None, _setup_index_token=None
            )
            with pytest.raises(ValueError, match="IndexHelper is incompatible"):
                system_singlepoint(tampered, x)
            return x.sum()

        return jacrev(inner_fn)(p)

    vmap(outer_vmap)(ih_stack_bad, pos_batch)

    # 11. Inference-mode setup, evaluation, and tampering rejection
    with torch.inference_mode():
        inf_sys = model.setup(sys_oh2.numbers.clone())
        e_inf = inf_sys.singlepoint(pos.clone()).energy.sum()
        torch.testing.assert_close(e_inf, ref_e, atol=1e-12, rtol=1e-12)
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            replace(inf_sys, ihelp=sys_hoh.ihelp).singlepoint(pos.clone())


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is not supported for this Python/torch combination",
)
def test_scf_dynamo_capture_and_post_compile_mutation_guard() -> None:
    """Cover B7 item 12: fullgraph pure SCF step and compiled System guard."""
    par1_comp = ParamModule(GFN1_XTB, dtype=torch.float64)
    nums_h2 = torch.tensor([1, 1], dtype=torch.int64)
    pos_h2 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    comp_cfg = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        exclude=["disp", "rep", "hal"],
        guess=labels.GUESS_SAD,
        fermi_etemp=0.0,
        strategy="linear",
        mix_guess=False,
        maxiter=1,
    )
    comp_sys = Model(par=par1_comp, config=comp_cfg).setup(nums_h2)
    assert comp_sys.integral_setup is not None
    assert comp_sys.h0_setup is not None
    chrg0 = torch.tensor([0.0], dtype=torch.float64)
    mats, refocc = build_integral_matrices(
        comp_sys.integral_setup,
        comp_sys.h0_setup,
        comp_sys.ihelp,
        pos_h2,
        chrg0,
    )
    n0, occ, nel = get_refocc(refocc, chrg0, None, comp_sys.ihelp)
    idata = _interaction_data(comp_sys, pos_h2, comp_sys.interactions)
    base_inputs = setup_scf_inputs(
        comp_sys.numbers,
        mats,
        n0,
        occ,
        nel,
        comp_sys.interactions,
        idata,
        comp_sys.ihelp,
        comp_sys.config.scf,
    )
    x0 = torch.zeros(
        (1, *base_inputs.charge_layout.mono_shape), dtype=torch.float64
    )

    def pure_map_only(x_in: Tensor) -> Tensor:
        assert base_inputs.index is comp_sys.ihelp
        return scf_map(x_in, base_inputs).mapped_x

    def setup_and_step(hcore_in: Tensor, x_in: Tensor) -> Tensor:
        local_mats = replace(mats, hcore=hcore_in)
        inp = setup_scf_inputs(
            comp_sys.numbers,
            local_mats,
            n0,
            occ,
            nel,
            comp_sys.interactions,
            idata,
            comp_sys.ihelp,
            comp_sys.config.scf,
        )
        assert inp.index is comp_sys.ihelp
        return scf_map(x_in, inp).mapped_x

    torch._dynamo.reset()
    ref_mapped = scf_map(x0, base_inputs).mapped_x
    comp_map = torch.compile(pure_map_only, backend="eager", fullgraph=True)
    torch.testing.assert_close(comp_map(x0), ref_mapped, atol=1e-12, rtol=1e-12)
    comp_step = torch.compile(setup_and_step, backend="eager")
    torch.testing.assert_close(
        comp_step(mats.hcore, x0), ref_mapped, atol=1e-12, rtol=1e-12
    )

    ref_comp_e = comp_sys.energy(pos_h2)
    comp_energy = torch.compile(comp_sys.energy, backend="eager")
    torch.testing.assert_close(
        comp_energy(pos_h2), ref_comp_e, atol=1e-12, rtol=1e-12
    )
    comp_sys.ihelp.shell_index[0] = 88
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        comp_energy(pos_h2)


def test_standalone_scf_without_system_object_and_metadata_guards() -> None:
    """Cover B7 item 13: standalone solve() and setup_scf_inputs() checks."""
    from dxtb._src.basis.bas import Basis
    from dxtb._src.components.interactions.coulomb.secondorder import new_es2
    from dxtb._src.integral.driver.pytorch import (
        build_overlap,
        evaluate_moments,
        setup_integrals,
    )
    from dxtb._src.xtb.h0 import build_hcore, setup_h0

    nums = torch.tensor([8, 1, 1], dtype=torch.int64)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
    )
    chrg = torch.tensor([0.0], dtype=torch.float64)

    # Pure standalone construction for GFN1 (no Model, System, or Calculator!)
    par1 = ParamModule(GFN1_XTB, dtype=torch.float64)
    ihelp1 = IndexHelper.from_numbers(nums, par1)
    bas1 = Basis(nums, par1, ihelp1, dtype=torch.float64)
    pt_setup1 = setup_integrals(bas1)
    ovlp1 = build_overlap(pt_setup1, pos)
    h0_1 = setup_h0(nums, par1, ihelp1, dtype=torch.float64)
    hcore1, _ = build_hcore(h0_1, pos, ovlp1, charge=chrg)
    mats1 = IntegralMatrices(hcore=hcore1, overlap=ovlp1)
    es2_1 = new_es2(nums, par1, shell_resolved=True, dtype=torch.float64)
    assert es2_1 is not None
    ilist1 = InteractionList(es2_1, device=pos.device, dtype=pos.dtype)
    icache1 = ilist1.get_cache(numbers=nums, positions=pos, ihelp=ihelp1)
    scf_cfg1 = ConfigSCF.create(method=labels.GFN1_XTB, maxiter=40)

    n0_1, occ_1, nel_1 = get_refocc(h0_1.refocc, chrg, None, ihelp1)
    standalone_inputs = setup_scf_inputs(
        nums,
        mats1,
        n0_1,
        occ_1,
        nel_1,
        ilist1,
        icache1,
        ihelp1,
        scf_cfg1,
    )
    assert standalone_inputs.index is ihelp1

    solved1 = solve(
        nums,
        pos,
        chrg,
        None,
        ilist1,
        icache1,
        ihelp1,
        scf_cfg1,
        mats1,
        h0_1.refocc,
    )
    assert bool(solved1.converged)

    # Standalone GFN0 non-SCF solve
    par0 = ParamModule(GFN0_XTB, dtype=torch.float64)
    ihelp0 = IndexHelper.from_numbers(nums, par0)
    bas0 = Basis(nums, par0, ihelp0, dtype=torch.float64)
    ovlp0 = build_overlap(setup_integrals(bas0), pos)
    h0_0 = setup_h0(nums, par0, ihelp0, dtype=torch.float64)
    hcore0, _ = build_hcore(h0_0, pos, ovlp0, charge=chrg)
    empty_ilist = InteractionList()
    res0 = solve(
        nums,
        pos,
        chrg,
        None,
        empty_ilist,
        empty_ilist.get_cache(numbers=nums, positions=pos, ihelp=ihelp0),
        ihelp0,
        ConfigSCF(method=labels.GFN0_XTB),
        IntegralMatrices(hcore=hcore0, overlap=ovlp0),
        h0_0.refocc,
    )
    assert res0["iterations"] == 0

    # Standalone GFN2 multipole inputs metadata validation
    par2 = ParamModule(GFN2_XTB, dtype=torch.float64)
    ihelp2 = IndexHelper.from_numbers(nums, par2)
    bas2 = Basis(nums, par2, ihelp2, dtype=torch.float64)
    ovlp2, dip2, quad2 = evaluate_moments(
        setup_integrals(bas2), pos, labels.INTLEVEL_QUADRUPOLE
    )
    h0_2 = setup_h0(nums, par2, ihelp2, dtype=torch.float64)
    hcore2, _ = build_hcore(h0_2, pos, ovlp2, charge=chrg)
    mats2 = IntegralMatrices(
        hcore=hcore2, overlap=ovlp2, dipole=dip2, quadrupole=quad2
    )
    n0_2, occ_2, nel_2 = get_refocc(h0_2.refocc, chrg, None, ihelp2)
    scf_cfg2 = ConfigSCF.create(method=labels.GFN2_XTB, maxiter=40)
    inputs2 = setup_scf_inputs(
        nums,
        mats2,
        n0_2,
        occ_2,
        nel_2,
        empty_ilist,
        empty_ilist.get_cache(numbers=nums, positions=pos, ihelp=ihelp2),
        ihelp2,
        scf_cfg2,
    )
    assert inputs2.index is ihelp2

    # Standalone rejection of invalid types, devices, dtypes, and shapes
    with pytest.raises(TypeError, match="IndexHelper"):
        solve(
            nums,
            pos,
            chrg,
            None,
            ilist1,
            icache1,
            object(),  # type: ignore[arg-type]
            scf_cfg1,
            mats1,
            h0_1.refocc,
        )

    with pytest.raises(TypeError, match="integer dtype"):
        setup_scf_inputs(
            nums.to(torch.float64),
            mats1,
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache1,
            ihelp1,
            scf_cfg1,
        )

    b_ihelp = IndexHelper.from_numbers(
        torch.stack([nums, nums]), par1, batch_mode=2
    )
    with pytest.raises(ValueError, match="rank"):
        setup_scf_inputs(
            nums,
            mats1,
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache1,
            b_ihelp,
            scf_cfg1,
        )

    with pytest.raises(DeviceError, match="Device mismatch"):
        setup_scf_inputs(
            nums,
            mats1,
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache1,
            ihelp1.to(device="meta"),
            scf_cfg1,
        )

    with pytest.raises(DeviceError, match="Device mismatch"):
        setup_scf_inputs(
            nums,
            replace(mats1, overlap=mats1.overlap.to(device="meta")),
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache1,
            ihelp1,
            scf_cfg1,
        )

    with pytest.raises(DtypeError, match="Dtype mismatch"):
        setup_scf_inputs(
            nums,
            replace(mats1, overlap=mats1.overlap.to(torch.float32)),
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache1,
            ihelp1,
            scf_cfg1,
        )

    # Mismatched IndexHelper orbital count (H2 ihelp with H2O matrices)
    ihelp_h2 = IndexHelper.from_numbers(
        torch.tensor([1, 1], dtype=torch.int64), par1
    )
    with pytest.raises(ValueError, match="atom shape"):
        setup_scf_inputs(
            nums,
            mats1,
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache1,
            ihelp_h2,
            scf_cfg1,
        )

    # Same nat=3, different nao (H3 ihelp has nao=6, whereas GFN1 H2O has nao=8)
    ihelp_h3 = IndexHelper.from_numbers(
        torch.tensor([1, 1, 1], dtype=torch.int64), par1
    )
    with pytest.raises(ValueError, match="hcore"):
        setup_scf_inputs(
            torch.tensor([1, 1, 1], dtype=torch.int64),
            mats1,
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache1,
            ihelp_h3,
            scf_cfg1,
        )

    # Wrong dipole shape & wrong ES2 cache matrix shape
    with pytest.raises(ValueError, match="dipole"):
        setup_scf_inputs(
            nums,
            replace(mats2, dipole=dip2[:2]),
            n0_2,
            occ_2,
            nel_2,
            empty_ilist,
            empty_ilist.get_cache(numbers=nums, positions=pos, ihelp=ihelp2),
            ihelp2,
            scf_cfg2,
        )

    icache_h2 = ilist1.get_cache(
        numbers=torch.tensor([1, 1], dtype=torch.int64),
        positions=pos[:2],
        ihelp=ihelp_h2,
    )
    with pytest.raises(ValueError, match=r"es2\.mat"):
        setup_scf_inputs(
            nums,
            mats1,
            n0_1,
            occ_1,
            nel_1,
            ilist1,
            icache_h2,
            ihelp1,
            scf_cfg1,
        )


def test_batched_implicit_solve_and_all_interaction_metadata_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify batched implicit solve, GFN0/solve guards, and interactions."""
    from tad_mctc.batch import pack

    from dxtb._src.components.interactions.dispersion.d4sc import (
        DispersionD4SCCache,
        new_d4sc,
    )
    from dxtb._src.components.interactions.field.efield import ElectricField
    from dxtb._src.components.interactions.field.efieldgrad import (
        ElectricFieldGrad,
    )
    from dxtb._src.components.interactions.solvation.alpb import (
        GeneralizedBorn,
    )
    from dxtb._src.xtb.h0 import setup_h0

    par1 = ParamModule(GFN1_XTB, dtype=torch.float64)
    par2 = ParamModule(GFN2_XTB, dtype=torch.float64)

    # 1. Public tensor-batched implicit solve (batch_mode=1 & batch_mode=2)
    nums_pad = pack([torch.tensor([3, 1]), torch.tensor([8, 1, 1])])
    pos_pad = pack(
        [
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64
            ),
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
                dtype=torch.float64,
            ),
        ]
    ).requires_grad_(True)
    chrg_pad = torch.zeros((2, 1), dtype=torch.float64)

    nums_conf = torch.tensor([[8, 1, 1], [8, 1, 1]], dtype=torch.int64)
    pos_conf = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.85], [0.0, 1.75, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    batch_cases = ((1, nums_pad, pos_pad), (2, nums_conf, pos_conf))
    for bmode, bnums, bpos in batch_cases:
        for m_lbl, m_par in ((labels.GFN1_XTB, par1), (labels.GFN2_XTB, par2)):
            calc_b = Calculator(
                bnums,
                m_par,
                opts={"batch_mode": bmode, "verbosity": 0, "maxiter": 50},
                dtype=torch.float64,
            )
            ref_res = calc_b.singlepoint(bpos.detach())
            assert ref_res.hcore is not None and ref_res.overlap is not None
            bmats = IntegralMatrices(
                hcore=ref_res.hcore,
                overlap=ref_res.overlap,
                dipole=ref_res.dipole_integrals,
                quadrupole=ref_res.quadrupole_integrals,
            )
            bh0 = setup_h0(bnums, m_par, calc_b.ihelp, dtype=torch.float64)
            bicache = calc_b.interactions.get_cache(
                numbers=bnums, positions=bpos, ihelp=calc_b.ihelp
            )
            bcfg = ConfigSCF.create(
                method=m_lbl,
                differentiation="implicit",
                maxiter=60,
                x_atol=1.0e-10,
                x_atol_max=1.0e-10,
            )
            seen_inputs: list[SCFInputs] = []
            orig_setup = scf_eager.setup_scf_inputs
            monkeypatch.setattr(
                scf_eager,
                "setup_scf_inputs",
                lambda *a, **kw: (
                    seen_inputs.append(out := orig_setup(*a, **kw)) or out
                ),
            )
            bsolved = solve(
                bnums,
                bpos,
                chrg_pad,
                None,
                calc_b.interactions,
                bicache,
                calc_b.ihelp,
                bcfg,
                bmats,
                bh0.refocc,
            )
            monkeypatch.undo()
            assert len(seen_inputs) == 1
            assert seen_inputs[0].index is calc_b.ihelp
            assert bool(bsolved.converged)
            assert ref_res.scf is not None
            torch.testing.assert_close(
                bsolved.energy.sum(-1),
                ref_res.scf.sum(-1),
                atol=1e-8,
                rtol=1e-8,
            )
            (bgrad,) = torch.autograd.grad(bsolved.energy.sum(), bpos)
            assert torch.isfinite(bgrad).all()
            assert float(bgrad.abs().max()) > 0.0

    # 2. Upfront guards in standalone solve() and GFN0 solve_nonselfconsistent()
    nums_h2o = torch.tensor([8, 1, 1], dtype=torch.int64)
    pos_h2o = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
    )
    chrg0 = torch.tensor([0.0], dtype=torch.float64)
    calc2 = Calculator(
        nums_h2o, par2, opts={"verbosity": 0}, dtype=torch.float64
    )
    sp2 = calc2.singlepoint(pos_h2o)
    assert sp2.hcore is not None and sp2.overlap is not None
    mats2 = IntegralMatrices(
        hcore=sp2.hcore,
        overlap=sp2.overlap,
        dipole=sp2.dipole_integrals,
        quadrupole=sp2.quadrupole_integrals,
    )
    h0_2 = setup_h0(nums_h2o, par2, calc2.ihelp, dtype=torch.float64)
    ic2 = calc2.interactions.get_cache(
        numbers=nums_h2o, positions=pos_h2o, ihelp=calc2.ihelp
    )
    cfg2 = ConfigSCF.create(method=labels.GFN2_XTB, maxiter=20)

    # Complex numbers, meta/mixed-device/int32 IndexHelper, and atom mismatch
    with pytest.raises(TypeError, match="integer dtype"):
        solve(
            nums_h2o.to(torch.complex64),
            pos_h2o,
            chrg0,
            None,
            calc2.interactions,
            ic2,
            calc2.ihelp,
            cfg2,
            mats2,
            h0_2.refocc,
        )
    with pytest.raises(DeviceError, match="Device mismatch"):
        solve(
            nums_h2o,
            pos_h2o,
            chrg0,
            None,
            calc2.interactions,
            ic2,
            calc2.ihelp.to(device="meta"),
            cfg2,
            mats2,
            h0_2.refocc,
        )
    mixed_dev_ih = IndexHelper.from_numbers(nums_h2o, par2)
    object.__setattr__(
        mixed_dev_ih,
        "shells_to_atom",
        mixed_dev_ih.shells_to_atom.to(device="meta"),
    )
    with pytest.raises(DeviceError, match="Device mismatch"):
        solve(
            nums_h2o,
            pos_h2o,
            chrg0,
            None,
            calc2.interactions,
            ic2,
            mixed_dev_ih,
            cfg2,
            mats2,
            h0_2.refocc,
        )
    int32_ih = IndexHelper(
        **{
            k: getattr(calc2.ihelp, k).to(torch.int32)
            for k in calc2.ihelp._child_names
        },
        batch_mode=0,
    )
    with pytest.raises(TypeError, match="int64"):
        solve(
            nums_h2o,
            pos_h2o,
            chrg0,
            None,
            calc2.interactions,
            ic2,
            int32_ih,
            cfg2,
            mats2,
            h0_2.refocc,
        )
    ih_h2 = IndexHelper.from_numbers(torch.tensor([1, 1]), par2)
    with pytest.raises(ValueError, match="atom shape"):
        solve(
            nums_h2o,
            pos_h2o,
            chrg0,
            None,
            calc2.interactions,
            ic2,
            ih_h2,
            cfg2,
            mats2,
            h0_2.refocc,
        )
    with pytest.raises(ValueError, match="positions"):
        solve(
            nums_h2o,
            pos_h2o[:2],
            chrg0,
            None,
            calc2.interactions,
            ic2,
            calc2.ihelp,
            cfg2,
            mats2,
            h0_2.refocc,
        )

    # GFN0 non-SCF rejects mismatched hcore/overlap shapes upfront
    par0 = ParamModule(GFN0_XTB, dtype=torch.float64)
    ih0_h2o = IndexHelper.from_numbers(nums_h2o, par0)
    h0_0 = setup_h0(nums_h2o, par0, ih0_h2o, dtype=torch.float64)
    empty_ilist = InteractionList()
    bad_mats0 = IntegralMatrices(
        hcore=torch.eye(2, dtype=torch.float64),
        overlap=torch.eye(2, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="hcore"):
        solve(
            nums_h2o,
            pos_h2o,
            chrg0,
            None,
            empty_ilist,
            empty_ilist.get_cache(
                numbers=nums_h2o, positions=pos_h2o, ihelp=ih0_h2o
            ),
            ih0_h2o,
            ConfigSCF(method=labels.GFN0_XTB),
            bad_mats0,
            h0_0.refocc,
        )

    # 3. Standalone interaction checks (es3, aes2, d4sc, gb, efield, efield_grad)
    n0_2, occ_2, nel_2 = get_refocc(h0_2.refocc, chrg0, None, calc2.ihelp)
    nums_h2 = torch.tensor([1, 1], dtype=torch.int64)
    calc2_h2 = Calculator(
        nums_h2, par2, opts={"verbosity": 0}, dtype=torch.float64
    )
    ic2_h2 = calc2_h2.interactions.get_cache(
        numbers=nums_h2, positions=pos_h2o[:2], ihelp=ih_h2
    )
    for comp in calc2.interactions.components:
        single_list = InteractionList(
            comp, device=pos_h2o.device, dtype=pos_h2o.dtype
        )
        bad_cache = empty_ilist.get_cache(
            numbers=nums_h2o, positions=pos_h2o, ihelp=calc2.ihelp
        )
        bad_cache[comp.label] = ic2_h2[comp.label]
        if comp.label == "DispersionD4SC":
            # Also verify wrong dispmat alone (valid cn, wrong dispmat)
            valid_d4_cache = single_list.get_cache(
                numbers=nums_h2o, positions=pos_h2o, ihelp=calc2.ihelp
            )
            valid_entry = valid_d4_cache[comp.label]
            bad_disp = ic2_h2[comp.label].dispmat
            valid_d4_cache[comp.label] = DispersionD4SCCache(
                valid_entry.cn, bad_disp, valid_entry.model
            )
            with pytest.raises(ValueError, match=r"d4sc\.dispmat"):
                setup_scf_inputs(
                    nums_h2o,
                    mats2,
                    n0_2,
                    occ_2,
                    nel_2,
                    single_list,
                    valid_d4_cache,
                    calc2.ihelp,
                    cfg2,
                )
        else:
            with pytest.raises(ValueError):
                setup_scf_inputs(
                    nums_h2o,
                    mats2,
                    n0_2,
                    occ_2,
                    nel_2,
                    single_list,
                    bad_cache,
                    calc2.ihelp,
                    cfg2,
                )

    # D4SCSetup standalone dtype / device / atom-count mismatch
    d4_f32 = new_d4sc(nums_h2o, par2, dtype=torch.float32)
    assert d4_f32 is not None
    d4_f64 = new_d4sc(nums_h2o, par2, dtype=torch.float64)
    assert d4_f64 is not None
    valid_d4_ic = InteractionList(
        d4_f64, device=pos_h2o.device, dtype=torch.float64
    ).get_cache(numbers=nums_h2o, positions=pos_h2o, ihelp=calc2.ihelp)
    with pytest.raises(DtypeError, match="d4sc.setup"):
        setup_scf_inputs(
            nums_h2o,
            mats2,
            n0_2,
            occ_2,
            nel_2,
            InteractionList(d4_f32, device=pos_h2o.device, dtype=torch.float32),
            valid_d4_ic,
            calc2.ihelp,
            cfg2,
        )
    import copy

    d4_meta = copy.copy(d4_f64)
    d4_meta.rcov = d4_f64.rcov.to(device="meta")
    d4_meta_list = InteractionList(
        d4_meta, device=pos_h2o.device, dtype=torch.float64
    )
    with pytest.raises(DeviceError, match="d4sc.setup"):
        setup_scf_inputs(
            nums_h2o,
            mats2,
            n0_2,
            occ_2,
            nel_2,
            d4_meta_list,
            valid_d4_ic,
            calc2.ihelp,
            cfg2,
        )
    d4_h2 = new_d4sc(nums_h2, par2, dtype=torch.float64)
    assert d4_h2 is not None
    with pytest.raises(ValueError, match="d4sc.setup"):
        setup_scf_inputs(
            nums_h2o,
            mats2,
            n0_2,
            occ_2,
            nel_2,
            InteractionList(d4_h2, device=pos_h2o.device, dtype=torch.float64),
            valid_d4_ic,
            calc2.ihelp,
            cfg2,
        )

    # ElectricField, ElectricFieldGrad, and GeneralizedBorn shape checks
    ef = ElectricField(torch.zeros((3,), dtype=torch.float64))
    ef_list = InteractionList(ef, device=pos_h2o.device, dtype=torch.float64)
    ef_bad = ef_list.get_cache(
        numbers=nums_h2, positions=pos_h2o[:2], ihelp=ih_h2
    )
    with pytest.raises(ValueError, match="electric_field"):
        setup_scf_inputs(
            nums_h2o,
            mats2,
            n0_2,
            occ_2,
            nel_2,
            ef_list,
            ef_bad,
            calc2.ihelp,
            cfg2,
        )

    efg = ElectricFieldGrad(torch.zeros((3, 3), dtype=torch.float64))
    efg_list = InteractionList(efg, device=pos_h2o.device, dtype=torch.float64)
    efg_bad = efg_list.get_cache(
        numbers=nums_h2, positions=pos_h2o[:2], ihelp=ih_h2
    )
    with pytest.raises(ValueError, match="electric_field_grad"):
        setup_scf_inputs(
            nums_h2o,
            mats2,
            n0_2,
            occ_2,
            nel_2,
            efg_list,
            efg_bad,
            calc2.ihelp,
            cfg2,
        )

    gb_h2 = GeneralizedBorn(
        nums_h2, torch.tensor(78.9, dtype=torch.float64), dtype=torch.float64
    )
    gb_list = InteractionList(gb_h2, device=pos_h2o.device, dtype=torch.float64)
    gb_bad_cache = gb_list.get_cache(
        numbers=nums_h2, positions=pos_h2o[:2], ihelp=ih_h2
    )
    with pytest.raises(ValueError, match=r"generalized_born\.mat"):
        setup_scf_inputs(
            nums_h2o,
            mats2,
            n0_2,
            occ_2,
            nel_2,
            gb_list,
            gb_bad_cache,
            calc2.ihelp,
            cfg2,
        )

    # Zero-orbital synthetic IndexHelper rejected before 0/0 normalization
    zero_orb_ih = copy.copy(calc2.ihelp)
    object.__setattr__(
        zero_orb_ih, "orbitals_to_shell", torch.empty((0,), dtype=torch.int64)
    )
    with pytest.raises(ValueError):
        setup_scf_inputs(
            nums_h2o,
            mats2,
            n0_2,
            occ_2,
            nel_2,
            empty_ilist,
            empty_ilist.get_cache(
                numbers=nums_h2o, positions=pos_h2o, ihelp=calc2.ihelp
            ),
            zero_orb_ih,
            cfg2,
        )


def _standalone_h2o_bundle(method_label: int, dtype: torch.dtype) -> tuple[
    Tensor,
    Tensor,
    InteractionList,
    object,
    IndexHelper,
    ConfigSCF,
    IntegralMatrices,
    Tensor,
]:
    """Build standalone H2O numbers, geometry, integrals, cache, and config."""
    from dxtb._src.basis.bas import Basis
    from dxtb._src.integral.driver.pytorch import (
        build_overlap,
        evaluate_moments,
        setup_integrals,
    )
    from dxtb._src.xtb.h0 import build_hcore, setup_h0

    raw_par = {
        labels.GFN0_XTB: GFN0_XTB,
        labels.GFN1_XTB: GFN1_XTB,
        labels.GFN2_XTB: GFN2_XTB,
    }[method_label]
    par = ParamModule(raw_par, dtype=dtype)
    nums = torch.tensor([8, 1, 1], dtype=torch.int64)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]], dtype=dtype
    )
    ihelp = IndexHelper.from_numbers(nums, par)
    pt_setup = setup_integrals(Basis(nums, par, ihelp, dtype=dtype))
    h0 = setup_h0(nums, par, ihelp, dtype=dtype)
    chrg0 = torch.tensor([0.0], dtype=dtype)
    if method_label == labels.GFN2_XTB:
        ovlp, dip, quad = evaluate_moments(
            pt_setup, pos, labels.INTLEVEL_QUADRUPOLE
        )
        hcore, _ = build_hcore(h0, pos, ovlp, charge=chrg0)
        mats = IntegralMatrices(
            hcore=hcore, overlap=ovlp, dipole=dip, quadrupole=quad
        )
    else:
        ovlp = build_overlap(pt_setup, pos)
        hcore, _ = build_hcore(h0, pos, ovlp, charge=chrg0)
        mats = IntegralMatrices(hcore=hcore, overlap=ovlp)
    if method_label == labels.GFN0_XTB:
        ilist = InteractionList()
        icache = ilist.get_cache(numbers=nums, positions=pos, ihelp=ihelp)
        cfg = ConfigSCF(method=labels.GFN0_XTB)
    else:
        calc = Calculator(
            nums, par, opts={"verbosity": 0, "maxiter": 40}, dtype=dtype
        )
        ilist = calc.interactions
        icache = ilist.get_cache(numbers=nums, positions=pos, ihelp=ihelp)
        cfg = ConfigSCF.create(method=method_label, maxiter=40)
    cfg = replace(cfg, fermi=replace(cfg.fermi, etemp=300.0))
    return nums, pos, ilist, icache, ihelp, cfg, mats, h0.refocc


def test_standalone_solve_charge_spin_dtypes_and_autograd() -> None:
    """Verify WP1.1/WP1.2 charge/spin dtypes, parity, and autograd on solve."""
    combos = (
        (torch.float64, torch.float64, torch.float64),
        (torch.float64, torch.float64, torch.int64),
        (torch.float64, torch.float64, torch.float32),
        (torch.float64, torch.int64, torch.float64),
        (torch.float64, torch.int64, torch.int64),
        (torch.float64, torch.float32, torch.int64),
        (torch.float32, torch.float32, torch.int64),
        (torch.float32, torch.float64, torch.int64),
    )
    bundles = {
        (m, dt): _standalone_h2o_bundle(m, dt)
        for m in (labels.GFN0_XTB, labels.GFN1_XTB, labels.GFN2_XTB)
        for dt in (torch.float64, torch.float32)
    }

    for m_lbl in (labels.GFN0_XTB, labels.GFN1_XTB, labels.GFN2_XTB):
        guesses = (
            (labels.GUESS_EEQ,)
            if m_lbl == labels.GFN0_XTB
            else (labels.GUESS_EEQ, labels.GUESS_SAD)
        )
        for guess in guesses:
            ref_energies: dict[torch.dtype, Tensor] = {}
            ref_chrg_grads: dict[torch.dtype, Tensor] = {}
            for dt in (torch.float64, torch.float32):
                nums, pos, ilist, icache, ihelp, cfg, mats, refocc = bundles[
                    (m_lbl, dt)
                ]
                cfg_g = replace(cfg, guess=guess)
                c_ref = torch.tensor([0.0], dtype=dt, requires_grad=True)
                s_ref = torch.tensor([0.0], dtype=dt)
                out_ref = solve(
                    nums,
                    pos,
                    c_ref,
                    s_ref,
                    ilist,
                    icache,
                    ihelp,
                    cfg_g,
                    mats,
                    refocc,
                )
                e_ref = (
                    out_ref["energy"]
                    if isinstance(out_ref, dict)
                    else out_ref.energy
                )
                ref_energies[dt] = e_ref.detach()
                ref_chrg_grads[dt] = torch.autograd.grad(e_ref.sum(), c_ref)[0]

            for calc_dt, chrg_dt, spin_dt in combos:
                nums, pos, ilist, icache, ihelp, cfg, mats, refocc = bundles[
                    (m_lbl, calc_dt)
                ]
                cfg_g = replace(cfg, guess=guess)
                pos_g = pos.clone().requires_grad_(True)
                chrg = torch.tensor([0], dtype=chrg_dt)
                if chrg.is_floating_point():
                    chrg = chrg.clone().requires_grad_(True)
                spin = torch.tensor([0], dtype=spin_dt)
                out = solve(
                    nums,
                    pos_g,
                    chrg,
                    spin,
                    ilist,
                    icache,
                    ihelp,
                    cfg_g,
                    mats,
                    refocc,
                )
                energy = out["energy"] if isinstance(out, dict) else out.energy
                assert energy.dtype is calc_dt
                tol = 0.0 if chrg_dt == calc_dt else 1e-6
                torch.testing.assert_close(
                    energy.detach(), ref_energies[calc_dt], atol=tol, rtol=tol
                )
                if m_lbl != labels.GFN0_XTB and guess == labels.GUESS_EEQ:
                    g_pos = torch.autograd.grad(
                        energy.sum(), pos_g, retain_graph=chrg.requires_grad
                    )[0]
                    assert torch.isfinite(g_pos).all()
                if chrg.requires_grad:
                    g_c = torch.autograd.grad(energy.sum(), chrg)[0]
                    assert g_c.dtype is chrg_dt
                    gtol = (
                        1e-5 if torch.float32 in (calc_dt, chrg_dt) else 1e-12
                    )
                    torch.testing.assert_close(
                        g_c.to(calc_dt),
                        ref_chrg_grads[calc_dt],
                        atol=gtol,
                        rtol=gtol,
                    )

    # Implicit SCF route also accepts integer/mixed spin and charge dtypes
    nums, pos, ilist, icache, ihelp, cfg, mats, refocc = bundles[
        (labels.GFN1_XTB, torch.float64)
    ]
    cfg_imp = replace(
        cfg, differentiation=cfg.differentiation.__class__.IMPLICIT
    )
    c_imp = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    s_imp = torch.tensor([0], dtype=torch.int64)
    res_imp = solve(
        nums, pos, c_imp, s_imp, ilist, icache, ihelp, cfg_imp, mats, refocc
    )
    assert not isinstance(res_imp, dict)
    assert res_imp.energy.dtype is torch.float64
    (g_imp,) = torch.autograd.grad(res_imp.energy.sum(), c_imp)
    assert g_imp.dtype is torch.float32 and torch.isfinite(g_imp).all()


def test_standalone_charge_spin_shapes_and_transforms() -> None:
    """Verify WP1.3/WP5 charge/spin shape contract, broadcasting guard, vmap."""
    from tad_mctc.batch import pack

    from dxtb._src.xtb.h0 import setup_h0

    nums, pos, ilist, icache, ihelp, cfg, mats, refocc = _standalone_h2o_bundle(
        labels.GFN1_XTB, torch.float64
    )
    e_ref = solve(
        nums,
        pos,
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0], dtype=torch.int64),
        ilist,
        icache,
        ihelp,
        cfg,
        mats,
        refocc,
    )
    assert not isinstance(e_ref, dict)
    ref_sum = e_ref.energy.detach().sum()

    # Accepted single-System scalar/1D/list forms
    for c_ok, s_ok in (
        (torch.tensor(0.0, dtype=torch.float64), None),
        (
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor([0], dtype=torch.int64),
        ),
        (torch.tensor([0.0], dtype=torch.float64), [0]),
    ):
        r_ok = solve(
            nums, pos, c_ok, s_ok, ilist, icache, ihelp, cfg, mats, refocc  # type: ignore[arg-type]
        )
        assert not isinstance(r_ok, dict)
        torch.testing.assert_close(
            r_ok.energy.sum(), ref_sum, atol=1e-12, rtol=1e-12
        )

    # Rejected single-System malformed shapes, devices, and dtypes
    for exc_cls, c_bad, s_bad in (
        (
            ValueError,
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0, dtype=torch.int64),
        ),
        (ValueError, torch.zeros((2,), dtype=torch.float64), None),
        (
            ValueError,
            torch.zeros((1,), dtype=torch.float64),
            torch.zeros((2,), dtype=torch.int64),
        ),
        (ValueError, torch.zeros((1, 1), dtype=torch.float64), None),
        (TypeError, 0.0, None),
        (TypeError, torch.zeros((1,), dtype=torch.float64), 0),
        (TypeError, torch.tensor([False]), None),
        (
            TypeError,
            torch.zeros((1,), dtype=torch.float64),
            torch.tensor([False]),
        ),
        (TypeError, torch.zeros((1,), dtype=torch.float64), [False]),
        (TypeError, torch.zeros((1,), dtype=torch.float64), ["bad"]),
        (TypeError, torch.tensor([0.0 + 0.0j]), None),
        (
            DeviceError,
            torch.tensor([0.0], dtype=torch.float64, device="meta"),
            None,
        ),
        (
            DeviceError,
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0], dtype=torch.int64, device="meta"),
        ),
    ):
        with pytest.raises(exc_cls):
            solve(
                nums,
                pos,
                c_bad,  # type: ignore[arg-type]
                s_bad,  # type: ignore[arg-type]
                ilist,
                icache,
                ihelp,
                cfg,
                mats,
                refocc,
            )

    # Batched System: (nbatch, 1) accepted; (nbatch,) 1D broadcasting rejected
    par0 = ParamModule(GFN0_XTB, dtype=torch.float64)
    bnums = pack([torch.tensor([1, 1]), torch.tensor([8, 1, 1])])
    bpos = pack([pos[:2], pos])
    calc_b0 = Calculator(
        bnums,
        par0,
        opts={"batch_mode": 1, "verbosity": 0},
        dtype=torch.float64,
    )
    sp_b0 = calc_b0.singlepoint(bpos)
    bmats0 = IntegralMatrices(hcore=sp_b0.hcore, overlap=sp_b0.overlap)
    bh0_0 = setup_h0(bnums, par0, calc_b0.ihelp, dtype=torch.float64)
    bcfg0 = ConfigSCF(method=labels.GFN0_XTB)
    b_ilist = InteractionList()
    b_icache = b_ilist.get_cache(
        numbers=bnums, positions=bpos, ihelp=calc_b0.ihelp
    )

    out_b0 = solve(
        bnums,
        bpos,
        torch.zeros((2, 1), dtype=torch.int64),
        torch.zeros((2, 1), dtype=torch.int64),
        b_ilist,
        b_icache,
        calc_b0.ihelp,
        bcfg0,
        bmats0,
        bh0_0.refocc,
    )
    assert isinstance(out_b0, dict)
    torch.testing.assert_close(
        out_b0["energy"].sum(), sp_b0.scf.sum(), atol=1e-12, rtol=1e-12
    )
    with pytest.raises(ValueError, match="shape"):
        solve(
            bnums,
            bpos,
            torch.zeros((2,), dtype=torch.float64),
            None,
            b_ilist,
            b_icache,
            calc_b0.ihelp,
            bcfg0,
            bmats0,
            bh0_0.refocc,
        )
    with pytest.raises(ValueError, match="incompatible"):
        get_refocc(
            bh0_0.refocc,
            torch.zeros((2,), dtype=torch.float64),
            None,
            calc_b0.ihelp,
        )
    with pytest.raises(TypeError):
        solve(
            bnums,
            bpos,
            torch.zeros((2, 1), dtype=torch.float64),
            [[False], [False]],  # type: ignore[arg-type]
            b_ilist,
            b_icache,
            calc_b0.ihelp,
            bcfg0,
            bmats0,
            bh0_0.refocc,
        )

    # Transform compatibility (vmap over 0D scalar chrg + int64 spin, jacrev/jvp)
    nums0, pos0, ilist0, icache0, ihelp0, cfg0, mats0, refocc0 = (
        _standalone_h2o_bundle(labels.GFN0_XTB, torch.float64)
    )

    def eval_gfn0_chrg(c_in: Tensor) -> Tensor:
        cfg0_t0 = replace(cfg0, fermi=replace(cfg0.fermi, etemp=0.0))
        res = solve(
            nums0,
            pos0,
            c_in,
            None,
            ilist0,
            icache0,
            ihelp0,
            cfg0_t0,
            mats0,
            refocc0,
        )
        assert isinstance(res, dict)
        return res["energy"].sum()

    def eval_gfn1_fixed_trip(c_in: Tensor, s_in: Tensor) -> Tensor:
        cfg_sad = replace(cfg, guess=labels.GUESS_SAD, maxiter=15)
        res_ft = scf_eager.run_system_fixed_trip(
            nums, pos, c_in, s_in, ilist, icache, ihelp, cfg_sad, mats, refocc
        )
        assert not isinstance(res_ft, dict)
        return res_ft.energy.sum()

    chrg_vec = torch.tensor([0.0, 0.1], dtype=torch.float64)
    spin_mat = torch.tensor([[0], [0]], dtype=torch.int64)
    vmapped_e0 = vmap(eval_gfn0_chrg)(chrg_vec)
    serial_e0 = torch.stack([eval_gfn0_chrg(chrg_vec[i]) for i in range(2)])
    torch.testing.assert_close(vmapped_e0, serial_e0, atol=1e-12, rtol=1e-12)

    vmapped_e1 = vmap(eval_gfn1_fixed_trip)(chrg_vec, spin_mat)
    serial_e1 = torch.stack(
        [eval_gfn1_fixed_trip(chrg_vec[i], spin_mat[i]) for i in range(2)]
    )
    torch.testing.assert_close(vmapped_e1, serial_e1, atol=1e-12, rtol=1e-12)

    c0 = torch.tensor([0.0], dtype=torch.float64)
    s0 = torch.tensor([0], dtype=torch.int64)

    def eval_c(c_val: Tensor) -> Tensor:
        res = solve(
            nums0,
            pos0,
            c_val,
            s0,
            ilist0,
            icache0,
            ihelp0,
            cfg0,
            mats0,
            refocc0,
        )
        assert isinstance(res, dict)
        return res["energy"].sum()

    g_rev = jacrev(eval_c)(c0)
    g_fwd = jacfwd(eval_c)(c0)
    _, g_jvp = jvp(eval_c, (c0,), (torch.ones_like(c0),))
    torch.testing.assert_close(g_rev, g_fwd, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        g_rev, g_jvp.reshape_as(g_rev), atol=1e-12, rtol=1e-12
    )
