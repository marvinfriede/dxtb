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
"""Acceptance suite for E6c1-W2c9 converged-root conformer batch implicit SCF."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.func import grad, jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.model import Model
from dxtb.config import Config
from dxtb._src.calculators.batch_implicit import (
    BatchRootFailureCode,
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
from dxtb._src.scf.functional.implicit import implicit_root
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)
from dxtb._src.wavefunction.filling import FermiStatus

DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _build_system(
    numbers: torch.Tensor,
    *,
    method: int = labels.GFN1_XTB,
    scp_mode: int = labels.SCP_MODE_CHARGE,
    mixer: int = labels.MIXER_ANDERSON,
    etemp: float = 300.0,
    partition: int = labels.FERMI_PARTITION_EQUAL,
    maxiter: int = 200,
    x_atol: float = 1.0e-10,
    x_atol_max: float = 1.0e-10,
    f_atol: float = 1.0e-10,
    force_convergence: bool = True,
    int_level: int | None = None,
    trainable_model: bool = False,
) -> tuple[Model, object]:
    par_source = GFN1_XTB if method == labels.GFN1_XTB else GFN2_XTB
    par = ParamModule(par_source, device=DEVICE, dtype=DTYPE)
    cfg_kwargs = dict(
        method=method,
        int_driver=labels.INTDRIVER_PYTORCH,
        scf_mode=labels.SCF_MODE_FULL,
        scp_mode=scp_mode,
        mixer=mixer,
        maxiter=maxiter,
        x_atol=x_atol,
        x_atol_max=x_atol_max,
        f_atol=f_atol,
        fermi_etemp=etemp,
        fermi_partition=partition,
        force_convergence=force_convergence,
    )
    if int_level is not None:
        cfg_kwargs["int_level"] = int_level
    config = Config.create(**cfg_kwargs)
    model = Model(
        par=par,
        config=config,
        auto_int_level=(int_level is None),
    )
    for p in model.par.parameters():
        if p.is_floating_point():
            p.requires_grad_(trainable_model)
    system = model.setup(numbers, batch_mode=0)
    return model, system


def _h2o_conformers() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    base = torch.tensor(
        [
            [0.00000000, 0.00000000, 0.22170000],
            [0.00000000, 1.43090000, -0.88670000],
            [0.00000000, -1.43090000, -0.88670000],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    c1 = base.clone()
    c1[1, 1] += 0.08
    c1[2, 2] -= 0.05
    c2 = base.clone()
    c2[0, 2] += 0.06
    c2[1, 0] += 0.04
    return numbers, torch.stack((base, c1, c2), dim=0)


def _h2_conformers() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = torch.tensor([1, 1], device=DEVICE)
    c0 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.45]], dtype=DTYPE, device=DEVICE
    )
    c1 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.65]], dtype=DTYPE, device=DEVICE
    )
    return numbers, torch.stack((c0, c1), dim=0)


def _no2_conformers() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = torch.tensor([7, 8, 8], device=DEVICE)
    c0 = torch.tensor(
        [
            [0.00000000, 0.00000000, 0.35000000],
            [0.00000000, 2.05000000, -0.82000000],
            [0.00000000, -2.05000000, -0.82000000],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    c1 = c0.clone()
    c1[1, 1] += 0.09
    c1[2, 2] -= 0.06
    return numbers, torch.stack((c0, c1), dim=0)


@pytest.mark.parametrize(
    ("method", "scp_mode", "mixer"),
    [
        (labels.GFN1_XTB, labels.SCP_MODE_CHARGE, labels.MIXER_ANDERSON),
        (labels.GFN1_XTB, labels.SCP_MODE_CHARGE, labels.MIXER_LINEAR),
        (labels.GFN1_XTB, labels.SCP_MODE_POTENTIAL, labels.MIXER_ANDERSON),
        (labels.GFN1_XTB, labels.SCP_MODE_FOCK, labels.MIXER_ANDERSON),
        (labels.GFN2_XTB, labels.SCP_MODE_CHARGE, labels.MIXER_ANDERSON),
        (labels.GFN2_XTB, labels.SCP_MODE_POTENTIAL, labels.MIXER_ANDERSON),
    ],
)
def test_conformer_batch_matches_serial_and_permutes_cleanly(
    method: int, scp_mode: int, mixer: int
) -> None:
    """Batch evaluation matches independent single-conformer implicit results."""
    numbers, base_positions = _h2o_conformers()
    _, system = _build_system(
        numbers, method=method, scp_mode=scp_mode, mixer=mixer
    )
    positions = base_positions.clone().requires_grad_(True)

    prepared = prepare_converged_root_batch(system, positions)
    assert prepared.system is system
    assert prepared.all_roots_valid
    assert len(prepared.lanes) == 3

    result = evaluate_converged_root_batch(prepared)
    grad_batch = torch.autograd.grad(
        result.energy.sum(), positions, retain_graph=True
    )[0]

    # Check zero cross-lane position-gradient blocks.
    for lane_idx in range(3):
        lane_grad = torch.autograd.grad(
            result.energy[lane_idx].sum(),
            positions,
            retain_graph=(lane_idx < 2),
        )[0]
        torch.testing.assert_close(
            lane_grad[lane_idx], grad_batch[lane_idx], atol=1e-12, rtol=1e-12
        )
        for other_idx in range(3):
            if other_idx != lane_idx:
                torch.testing.assert_close(
                    lane_grad[other_idx],
                    torch.zeros_like(lane_grad[other_idx]),
                    atol=1e-13,
                    rtol=0.0,
                )

    # Compare against independent single-conformer evaluations.
    for lane_idx in range(3):
        single_pos = base_positions[lane_idx : lane_idx + 1].clone().requires_grad_(True)
        single_res = evaluate_converged_root_batch(system, single_pos)
        single_grad = torch.autograd.grad(single_res.energy.sum(), single_pos)[0]
        torch.testing.assert_close(
            result.energy[lane_idx],
            single_res.energy[0],
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            result.fenergy[lane_idx],
            single_res.fenergy[0],
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            result.charges.mono[lane_idx],
            single_res.charges.mono[0],
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            result.density[lane_idx],
            single_res.density[0],
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            grad_batch[lane_idx],
            single_grad[0],
            atol=1e-10,
            rtol=1e-10,
        )

    # Check lane permutation invariance.
    perm = [2, 0, 1]
    pos_perm = base_positions[perm].clone().requires_grad_(True)
    res_perm = evaluate_converged_root_batch(system, pos_perm)
    grad_perm = torch.autograd.grad(res_perm.energy.sum(), pos_perm)[0]
    torch.testing.assert_close(
        res_perm.energy, result.energy[perm], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        grad_perm, grad_batch[perm], atol=1e-11, rtol=1e-11
    )


@pytest.mark.parametrize(
    "partition",
    [labels.FERMI_PARTITION_EQUAL, labels.FERMI_PARTITION_ATOMIC],
)
def test_open_shell_finite_temperature_partially_occupied_no2(
    partition: int,
) -> None:
    """Open-shell doublet NO2 at 1000 K matches finite-difference gradients."""
    numbers, base_positions = _no2_conformers()
    _, system = _build_system(
        numbers,
        method=labels.GFN1_XTB,
        scp_mode=labels.SCP_MODE_CHARGE,
        etemp=1000.0,
        partition=partition,
    )
    positions = base_positions.clone().requires_grad_(True)
    result = evaluate_converged_root_batch(system, positions, spin=1.0)
    assert bool(result.converged.all())
    assert bool((result.fenergy.abs().sum(dim=-1) > 1e-7).all())

    analytic_grad = torch.autograd.grad(result.energy.sum(), positions)[0]

    step = 1.0e-6
    fd_grad = torch.zeros_like(base_positions)
    for b in range(base_positions.shape[0]):
        for a in range(base_positions.shape[1]):
            for c in range(3):
                pos_p = base_positions.clone()
                pos_m = base_positions.clone()
                pos_p[b, a, c] += step
                pos_m[b, a, c] -= step
                ep = evaluate_converged_root_batch(
                    system, pos_p, spin=1.0
                ).energy[b].sum()
                em = evaluate_converged_root_batch(
                    system, pos_m, spin=1.0
                ).energy[b].sum()
                fd_grad[b, a, c] = (ep - em) / (2.0 * step)

    torch.testing.assert_close(analytic_grad, fd_grad, atol=5e-7, rtol=5e-6)


def test_exact_and_near_spectral_degeneracy_h3_plus() -> None:
    """Equilateral and slightly perturbed H3+ support batch implicit derivatives."""
    numbers = torch.tensor([1, 1, 1], device=DEVICE)
    r = 1.65
    s3 = float(torch.sqrt(torch.tensor(3.0, dtype=DTYPE)))
    equilateral = torch.tensor(
        [
            [0.0, r / s3, 0.0],
            [-0.5 * r, -0.5 * r / s3, 0.0],
            [0.5 * r, -0.5 * r / s3, 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    near_deg = equilateral.clone()
    near_deg[0, 1] += 1.0e-7

    positions = torch.stack((equilateral, near_deg), dim=0).requires_grad_(True)
    _, system = _build_system(numbers, etemp=300.0)
    result = evaluate_converged_root_batch(system, positions, chrg=1.0)
    assert bool(torch.isfinite(result.energy).all())
    g = torch.autograd.grad(result.energy.sum(), positions)[0]
    assert bool(torch.isfinite(g).all())
    # Lane 0 (exact degeneracy) and lane 1 (1e-7 perturbation) must agree closely.
    torch.testing.assert_close(g[0], g[1], atol=1e-5, rtol=1e-5)


def test_noncontiguous_padding_preserves_values_and_gradients() -> None:
    """Noncontiguous canonical padding matches unpadded conformer batch."""
    numbers_lih = torch.tensor([3, 1], device=DEVICE)
    numbers_h2o, _ = _h2o_conformers()
    _, sys_lih = _build_system(
        numbers_lih, int_level=labels.INTLEVEL_HCORE
    )
    _, sys_h2o = _build_system(
        numbers_h2o, int_level=labels.INTLEVEL_HCORE
    )
    assert sys_lih.exact_core is not None
    assert sys_h2o.exact_core is not None

    bucket = bucket_canonical_exact_cores(
        (sys_lih.exact_core, sys_h2o.exact_core)
    )
    padded_lih = pad_canonical_exact_core(sys_lih.exact_core, bucket, 0)
    mapping = dict(bucket.source_maps)[0]
    active_atoms = torch.tensor(mapping.atoms, dtype=torch.int64, device=DEVICE)
    active_orbs = torch.tensor(
        mapping.orbitals, dtype=torch.int64, device=DEVICE
    )
    # Verify that the active atom and orbital indices are genuinely noncontiguous.
    assert not torch.equal(
        active_atoms, torch.arange(active_atoms.numel(), device=DEVICE)
    )
    assert not torch.equal(
        active_orbs, torch.arange(active_orbs.numel(), device=DEVICE)
    )

    c0 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]], dtype=DTYPE, device=DEVICE
    )
    c1 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.12, 3.180]], dtype=DTYPE, device=DEVICE
    )
    c2 = torch.tensor(
        [[0.0, -0.05, 0.0], [0.08, 0.0, 2.910]], dtype=DTYPE, device=DEVICE
    )
    pos_unpadded = torch.stack((c0, c1, c2), dim=0)

    padded_list = []
    inactive_mask = ~padded_lih.validity.atoms
    for idx in range(3):
        canon = canonicalize_positions(pos_unpadded[idx], bucket, 0).clone()
        if idx == 0:
            # Coincident with active atom 0
            canon[inactive_mask] = canon[active_atoms[0]].clone()
        else:
            # Displaced arbitrary coordinates on inactive padded slots
            canon[inactive_mask] = torch.tensor(
                [13.0 * idx, -9.0 * idx, 7.0 * idx],
                dtype=DTYPE,
                device=DEVICE,
            )
        padded_list.append(canon)
    pos_padded = torch.stack(padded_list, dim=0)

    p_unpad = pos_unpadded.clone().requires_grad_(True)
    p_pad = pos_padded.clone().requires_grad_(True)

    res_unpad = evaluate_converged_root_batch(sys_lih, p_unpad)
    res_pad = evaluate_converged_root_batch(padded_lih, p_pad)

    torch.testing.assert_close(
        res_pad.energy.sum(dim=-1),
        res_unpad.energy.sum(dim=-1),
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(
        res_pad.fenergy.sum(dim=-1),
        res_unpad.fenergy.sum(dim=-1),
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(
        res_pad.energy[:, active_atoms],
        res_unpad.energy,
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(
        res_pad.energy[:, inactive_mask],
        torch.zeros_like(res_pad.energy[:, inactive_mask]),
        atol=1e-14,
        rtol=0.0,
    )

    grad_unpad = torch.autograd.grad(res_unpad.energy.sum(), p_unpad)[0]
    grad_pad = torch.autograd.grad(res_pad.energy.sum(), p_pad)[0]

    torch.testing.assert_close(
        grad_pad[:, active_atoms, :], grad_unpad, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        grad_pad[:, inactive_mask, :],
        torch.zeros_like(grad_pad[:, inactive_mask, :]),
        atol=1e-14,
        rtol=0.0,
    )


def test_shared_model_parameter_gradients_and_leaf_classification() -> None:
    """Shared Model parameter gradients match sum of serial conformer gradients."""
    numbers, base_positions = _h2o_conformers()
    model, system = _build_system(
        numbers, method=labels.GFN1_XTB, trainable_model=True
    )
    param_items = tuple(
        (name, p)
        for name, p in model.par.named_parameters()
        if p.is_floating_point() and p.requires_grad
    )
    param_tensors = tuple(p for _, p in param_items)

    positions = base_positions[:2].clone().requires_grad_(True)
    result = evaluate_converged_root_batch(system, positions)
    batch_grads = torch.autograd.grad(
        result.energy.sum(),
        (positions, *param_tensors),
        allow_unused=True,
    )
    batch_pos_grad = batch_grads[0]
    batch_param_grads = batch_grads[1:]

    # Classify leaves into connected non-zero, connected zero, and disconnected.
    connected_nonzero = sum(
        g is not None and bool((g.abs() > 0.0).any()) for g in batch_param_grads
    )
    connected_zero = sum(
        g is not None and bool((g == 0.0).all()) for g in batch_param_grads
    )
    disconnected = sum(g is None for g in batch_param_grads)
    assert connected_nonzero > 0
    assert connected_zero + disconnected > 0
    assert connected_nonzero + connected_zero + disconnected == len(param_tensors)

    # Compare against sum of serial conformer gradients on the SAME Model owner.
    serial_param_accum = [None] * len(param_tensors)
    for idx in range(2):
        serial_system = model.setup(numbers, batch_mode=0)
        single_pos = base_positions[idx : idx + 1].clone().requires_grad_(True)
        single_res = evaluate_converged_root_batch(serial_system, single_pos)
        single_grads = torch.autograd.grad(
            single_res.energy.sum(),
            (single_pos, *param_tensors),
            allow_unused=True,
        )
        torch.testing.assert_close(
            batch_pos_grad[idx], single_grads[0][0], atol=1e-10, rtol=1e-10
        )
        for p_idx, g in enumerate(single_grads[1:]):
            if g is not None:
                serial_param_accum[p_idx] = (
                    g.detach().clone()
                    if serial_param_accum[p_idx] is None
                    else serial_param_accum[p_idx] + g.detach()
                )

    for (name, _), b_grad, s_grad in zip(
        param_items, batch_param_grads, serial_param_accum
    ):
        assert (b_grad is None) == (s_grad is None), f"Mismatch on {name}"
        if b_grad is not None and s_grad is not None:
            torch.testing.assert_close(
                b_grad, s_grad, atol=1e-10, rtol=1e-10, msg=f"Leaf {name}"
            )


def test_failure_policy_strict_and_non_strict_mixed_lanes() -> None:
    """Strict mode raises with lane index; non-strict isolates valid lanes."""
    numbers, base_positions = _h2o_conformers()

    # 1. maxiter=0 rejected.
    _, sys_zero = _build_system(numbers, maxiter=0, force_convergence=False)
    with pytest.raises(SCFConvergenceError, match="maxiter > 0"):
        prepare_converged_root_batch(sys_zero, base_positions[:2])

    # 2. maxiter exhaustion in strict vs non-strict mode.
    _, sys_one = _build_system(numbers, maxiter=1, force_convergence=False)
    with pytest.raises(SCFConvergenceError, match=r"\[Lane 0\]"):
        prepare_converged_root_batch(sys_one, base_positions[:2], strict=True)

    with pytest.warns(SCFConvergenceWarning):
        prep_non_strict = prepare_converged_root_batch(
            sys_one, base_positions[:2], strict=False
        )
    assert not prep_non_strict.all_roots_valid
    assert all(
        lane.failure_code == int(BatchRootFailureCode.MAXITER_EXCEEDED)
        for lane in prep_non_strict.lanes
    )

    # 3. Mixed valid (Lane 0) and invalid (Lane 1) lanes in non-strict mode.
    _, system = _build_system(numbers, force_convergence=False)
    pos_two = base_positions[:2].clone().requires_grad_(True)
    prepared = prepare_converged_root_batch(system, pos_two, strict=False)
    assert prepared.all_roots_valid

    # Corrupt Lane 1 into an invalid root record while keeping Lane 0 valid.
    invalid_lane_1 = replace(
        prepared.lanes[1],
        root_valid=False,
        solver_converged=torch.tensor(False, device=DEVICE),
        failure_code=int(BatchRootFailureCode.RESIDUAL_TOLERANCE),
        failure_reason="Injected residual failure on lane 1.",
        status=replace(
            prepared.lanes[1].status,
            final_failure=torch.tensor(
                int(BatchRootFailureCode.RESIDUAL_TOLERANCE),
                dtype=torch.int64,
                device=DEVICE,
            ),
        ),
    )
    mixed_batch = replace(
        prepared,
        lanes=(prepared.lanes[0], invalid_lane_1),
        valid_lane_mask=torch.tensor([True, False], device=DEVICE),
        all_roots_valid=False,
    )

    with pytest.raises(SCFConvergenceError, match=r"\[Lane 1\]"):
        evaluate_converged_root_batch(mixed_batch, strict=True)

    mixed_result = evaluate_converged_root_batch(mixed_batch, strict=False)
    assert bool(mixed_result.converged[0]) is True
    assert bool(mixed_result.converged[1]) is False
    assert int(mixed_result.status.final_failure[0].item()) == 0
    assert int(mixed_result.status.final_failure[1].item()) == int(
        BatchRootFailureCode.RESIDUAL_TOLERANCE
    )
    assert bool(torch.isfinite(mixed_result.energy[0]).all())
    assert bool(torch.isnan(mixed_result.energy[1]).all())
    assert bool(torch.isnan(mixed_result.density[1]).all())

    # Differentiating valid lane 0 must be completely uncorrupted by lane 1.
    grad_lane_0 = torch.autograd.grad(
        mixed_result.energy[0].sum(), pos_two
    )[0]
    assert bool(torch.isfinite(grad_lane_0[0]).all())
    torch.testing.assert_close(
        grad_lane_0[1], torch.zeros_like(grad_lane_0[1]), atol=0.0, rtol=0.0
    )

    pos_single = base_positions[:1].clone().requires_grad_(True)
    res_single = evaluate_converged_root_batch(system, pos_single)
    grad_single = torch.autograd.grad(res_single.energy.sum(), pos_single)[0]
    torch.testing.assert_close(
        mixed_result.energy[0], res_single.energy[0], atol=1e-14, rtol=1e-14
    )
    torch.testing.assert_close(
        grad_lane_0[0], grad_single[0], atol=1e-14, rtol=1e-14
    )

    # 4. Stale prepared root rejected when evaluated at shifted geometry.
    shifted_pos = base_positions[:2].clone()
    shifted_pos[1, 0, 2] += 0.25
    with pytest.raises(SCFConvergenceError, match=r"\[Lane 1\]"):
        evaluate_converged_root_batch(
            prepared, positions=shifted_pos, strict=True
        )


def test_fermi_and_linear_response_failure_isolation() -> None:
    """Too-many-electrons and singular linear-response lanes are isolated."""
    numbers, base_positions = _h2_conformers()
    _, system = _build_system(numbers, force_convergence=False)

    # Too many electrons on lane 1 (charge = -10.0 on H2 -> 12 electrons in 2 orbitals).
    charges = torch.tensor([0.0, -10.0], dtype=DTYPE, device=DEVICE)
    with pytest.raises(ValueError, match=r"\[Lane 1\]"):
        prepare_converged_root_batch(
            system, base_positions, chrg=charges, strict=True
        )

    with pytest.warns(SCFConvergenceWarning):
        prep = prepare_converged_root_batch(
            system, base_positions, chrg=charges, strict=False
        )
    assert prep.lanes[0].root_valid is True
    assert prep.lanes[1].root_valid is False
    assert prep.lanes[1].failure_code == int(
        BatchRootFailureCode.TOO_MANY_ELECTRONS
    )
    res = evaluate_converged_root_batch(prep, strict=False)
    assert bool(torch.isfinite(res.energy[0]).all())
    assert bool(torch.isnan(res.energy[1]).all())

    # Singular linear response in implicit_root under vmap:
    # Lane 0 has G(x, p) = 0.5 * x + p (nonsingular I - G_x = 0.5 I).
    # Lane 1 has G(x, p) = x + p (singular I - G_x = 0).
    roots = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=DTYPE, device=DEVICE)
    params = torch.tensor(
        [[1.0, 1.0], [0.0, 0.0]], dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    scales = torch.tensor([0.5, 1.0], dtype=DTYPE, device=DEVICE)

    def raw_map(x: torch.Tensor, p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return s * x + p

    attached = vmap(
        lambda r, p, s: implicit_root(r, p, s, raw_map=raw_map)
    )(roots, params, scales)
    grad_lane0 = torch.autograd.grad(
        attached[0].sum(), params, retain_graph=True
    )[0]
    grad_lane1 = torch.autograd.grad(attached[1].sum(), params)[0]
    torch.testing.assert_close(
        grad_lane0[0], torch.full((2,), 2.0, dtype=DTYPE), atol=1e-10, rtol=1e-10
    )
    assert bool(torch.isnan(grad_lane1[1]).all())


def test_transforms_grad_jvp_jacfwd_jacrev_hvp_vmap_and_higher_order() -> None:
    """Supports grad, jvp, jacfwd, jacrev, HVP, nested vmap, 2nd/3rd order, and compile."""
    numbers, base_positions = _h2_conformers()
    _, system = _build_system(
        numbers,
        etemp=300.0,
        int_level=labels.INTLEVEL_DIPOLE,
    )
    prepared = prepare_converged_root_batch(system, base_positions)

    def batch_energy_vec(pos: torch.Tensor) -> torch.Tensor:
        return evaluate_converged_root_batch(prepared, positions=pos).energy.sum(
            dim=-1
        )

    def batch_energy_sum(pos: torch.Tensor) -> torch.Tensor:
        return batch_energy_vec(pos).sum()

    # 1. grad, jacrev, jacfwd, jvp parity.
    g_func = grad(batch_energy_sum)(base_positions)
    j_rev = jacrev(batch_energy_vec)(base_positions)
    j_fwd = jacfwd(batch_energy_vec)(base_positions)
    tangent = torch.randn_like(base_positions)
    _, jvp_val = jvp(batch_energy_vec, (base_positions,), (tangent,))

    torch.testing.assert_close(j_rev, j_fwd, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        j_rev.sum(dim=0), g_func, atol=1e-10, rtol=1e-10
    )
    expected_jvp = (j_rev * tangent.unsqueeze(0)).sum(dim=(-3, -2, -1))
    torch.testing.assert_close(jvp_val, expected_jvp, atol=1e-10, rtol=1e-10)

    # 2. Nested vmap over tangent directions.
    directions = torch.stack((tangent, -0.5 * tangent, 2.0 * tangent), dim=0)
    vmapped_jvp = vmap(
        lambda d: jvp(batch_energy_vec, (base_positions,), (d,))[1]
    )(directions)
    for idx, scale in enumerate((1.0, -0.5, 2.0)):
        torch.testing.assert_close(
            vmapped_jvp[idx], scale * jvp_val, atol=1e-10, rtol=1e-10
        )

    # 3. 2nd-order (Hessian / HVP) and 3rd-order derivatives wrt bond stretch s.
    z_dir = torch.zeros_like(base_positions)
    z_dir[:, 1, 2] = 1.0

    def energy_along_stretch(s: torch.Tensor) -> torch.Tensor:
        pos = base_positions + s * z_dir
        return evaluate_converged_root_batch(
            prepared, positions=pos
        ).energy.sum()

    s0 = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
    d1 = grad(energy_along_stretch)(s0)
    d2 = grad(grad(energy_along_stretch))(s0)
    d3 = grad(grad(grad(energy_along_stretch)))(s0)

    # Finite-difference check on 1st, 2nd, and 3rd derivatives by re-solving roots at +/- h.
    h = 1.0e-4
    e_m2 = evaluate_converged_root_batch(
        system, base_positions - 2.0 * h * z_dir
    ).energy.sum()
    e_m1 = evaluate_converged_root_batch(
        system, base_positions - h * z_dir
    ).energy.sum()
    e_0 = evaluate_converged_root_batch(system, base_positions).energy.sum()
    e_p1 = evaluate_converged_root_batch(
        system, base_positions + h * z_dir
    ).energy.sum()
    e_p2 = evaluate_converged_root_batch(
        system, base_positions + 2.0 * h * z_dir
    ).energy.sum()

    fd_d1 = (e_p1 - e_m1) / (2.0 * h)
    fd_d2 = (e_p1 - 2.0 * e_0 + e_m1) / (h**2)
    fd_d3 = (e_p2 - 2.0 * e_p1 + 2.0 * e_m1 - e_m2) / (2.0 * h**3)

    torch.testing.assert_close(d1, fd_d1, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(d2, fd_d2, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(d3, fd_d3, atol=5e-4, rtol=5e-4)

    # 4. torch.compile first-order evaluation and gradient.
    compiled_fn = torch.compile(batch_energy_sum, backend="eager")
    pos_compile = base_positions.clone().requires_grad_(True)
    e_comp = compiled_fn(pos_compile)
    g_comp = torch.autograd.grad(e_comp, pos_compile)[0]
    torch.testing.assert_close(e_comp, e_0, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(g_comp, g_func, atol=1e-10, rtol=1e-10)


def test_open_shell_and_charged_conformer_batches() -> None:
    """Open-shell (Li, NH2) and charged (OH-, H3O+) batches preserve prepared chrg/spin."""
    # 1. Li (Z=[3], spin=1.0, 1 conformer)
    numbers_li = torch.tensor([3], device=DEVICE)
    pos_li = torch.zeros((1, 1, 3), dtype=DTYPE, device=DEVICE, requires_grad=True)
    _, sys_li = _build_system(numbers_li, etemp=300.0)
    prep_li = prepare_converged_root_batch(sys_li, pos_li, spin=1.0)
    assert prep_li.all_roots_valid
    res_li = evaluate_converged_root_batch(prep_li, positions=pos_li)
    assert res_li.occupation.shape == (1, 2, 4)
    torch.testing.assert_close(
        res_li.occupation[0, 0].sum(),
        torch.tensor(1.0, dtype=DTYPE, device=DEVICE),
        atol=1e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(
        res_li.occupation[0, 1].sum(),
        torch.tensor(0.0, dtype=DTYPE, device=DEVICE),
        atol=1e-10,
        rtol=0.0,
    )

    # 2. NH2 radical (Z=[7, 1, 1], spin=1.0, 2 conformers)
    numbers_nh2 = torch.tensor([7, 1, 1], device=DEVICE)
    nh2_0 = torch.tensor(
        [
            [0.00000000, 0.00000000, 0.27000000],
            [0.00000000, 1.51000000, -0.98000000],
            [0.00000000, -1.51000000, -0.98000000],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    nh2_1 = nh2_0.clone()
    nh2_1[1, 1] += 0.07
    nh2_1[2, 2] -= 0.05
    pos_nh2 = torch.stack((nh2_0, nh2_1), dim=0)
    _, sys_nh2 = _build_system(numbers_nh2, etemp=300.0)
    prep_nh2 = prepare_converged_root_batch(sys_nh2, pos_nh2, spin=1.0)

    # Evaluating prepared batch with positions= must preserve prepared spin=1.0!
    p_nh2_a = pos_nh2.clone().requires_grad_(True)
    p_nh2_b = pos_nh2.clone().requires_grad_(True)
    res_nh2_a = evaluate_converged_root_batch(prep_nh2, positions=p_nh2_a)
    res_nh2_b = evaluate_converged_root_batch(sys_nh2, p_nh2_b, spin=1.0)
    torch.testing.assert_close(
        res_nh2_a.energy, res_nh2_b.energy, atol=1e-13, rtol=1e-13
    )
    g_nh2_a = torch.autograd.grad(res_nh2_a.energy.sum(), p_nh2_a)[0]
    g_nh2_b = torch.autograd.grad(res_nh2_b.energy.sum(), p_nh2_b)[0]
    torch.testing.assert_close(g_nh2_a, g_nh2_b, atol=1e-12, rtol=1e-12)

    # Overriding spin with a mismatched value must fail validation.
    with pytest.raises(RuntimeError, match="electron counts"):
        evaluate_converged_root_batch(
            prep_nh2, positions=pos_nh2, spin=3.0, strict=True
        )

    # 3. Charged OH- (chrg=-1.0) and H3O+ (chrg=+1.0)
    numbers_oh = torch.tensor([8, 1], device=DEVICE)
    pos_oh = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.91]],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    _, sys_oh = _build_system(numbers_oh, etemp=300.0)
    prep_oh = prepare_converged_root_batch(sys_oh, pos_oh, chrg=-1.0)
    p_oh = pos_oh.clone().requires_grad_(True)
    res_oh = evaluate_converged_root_batch(prep_oh, positions=p_oh)
    res_oh_direct = evaluate_converged_root_batch(
        sys_oh, pos_oh.clone().requires_grad_(True), chrg=-1.0
    )
    torch.testing.assert_close(
        res_oh.energy, res_oh_direct.energy, atol=1e-13, rtol=1e-13
    )
    with pytest.raises(RuntimeError, match="electron counts"):
        evaluate_converged_root_batch(
            prep_oh, positions=pos_oh, chrg=0.0, strict=True
        )

    numbers_h3o = torch.tensor([8, 1, 1, 1], device=DEVICE)
    h3o_0 = torch.tensor(
        [
            [0.00000000, 0.00000000, 0.12000000],
            [0.00000000, 1.78000000, -0.45000000],
            [1.54150000, -0.89000000, -0.45000000],
            [-1.54150000, -0.89000000, -0.45000000],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    h3o_1 = h3o_0.clone()
    h3o_1[1, 1] += 0.06
    pos_h3o = torch.stack((h3o_0, h3o_1), dim=0)
    _, sys_h3o = _build_system(numbers_h3o, etemp=300.0)
    prep_h3o = prepare_converged_root_batch(sys_h3o, pos_h3o, chrg=1.0)
    res_h3o = evaluate_converged_root_batch(prep_h3o, positions=pos_h3o)
    assert bool(res_h3o.converged.all())
    torch.testing.assert_close(
        res_h3o.charges.mono.sum(dim=-1),
        torch.ones(2, dtype=DTYPE, device=DEVICE),
        atol=1e-9,
        rtol=0.0,
    )


def test_electric_field_and_field_grad_perturbations() -> None:
    """Batch evaluation preserves prepared field/field_grad and differentiates both."""
    numbers, base_positions = _h2o_conformers()
    _, system = _build_system(
        numbers, int_level=labels.INTLEVEL_QUADRUPOLE, etemp=300.0
    )
    pos = base_positions[:2].clone()
    efield = torch.tensor([0.002, -0.001, 0.003], dtype=DTYPE, device=DEVICE)
    efgrad = torch.zeros((3, 3), dtype=DTYPE, device=DEVICE)
    efgrad[0, 0] = 0.001
    efgrad[1, 1] = -0.0004
    efgrad[2, 2] = -0.0006

    prepared = prepare_converged_root_batch(
        system, pos, field=efield, field_grad=efgrad
    )
    assert prepared.all_roots_valid

    # Preserves prepared field and field_grad when only positions= is passed.
    p_ref = pos.clone().requires_grad_(True)
    res_preserved = evaluate_converged_root_batch(prepared, positions=p_ref)

    f_live = efield.clone().requires_grad_(True)
    fg_live = efgrad.clone().requires_grad_(True)
    p_live = pos.clone().requires_grad_(True)
    res_explicit = evaluate_converged_root_batch(
        prepared, positions=p_live, field=f_live, field_grad=fg_live
    )
    torch.testing.assert_close(
        res_preserved.energy, res_explicit.energy, atol=1e-13, rtol=1e-13
    )

    g_pos, g_field, g_fgrad = torch.autograd.grad(
        res_explicit.energy.sum(), (p_live, f_live, fg_live)
    )
    assert bool(torch.isfinite(g_pos).all())
    assert bool(torch.isfinite(g_field).all())
    assert bool(torch.isfinite(g_fgrad).all())

    # Finite-difference check on electric field gradient.
    eps = 1.0e-6
    fd_field = torch.zeros_like(efield)
    for k in range(3):
        fp = efield.clone()
        fm = efield.clone()
        fp[k] += eps
        fm[k] -= eps
        ep = evaluate_converged_root_batch(
            system, pos, field=fp, field_grad=efgrad
        ).energy.sum()
        em = evaluate_converged_root_batch(
            system, pos, field=fm, field_grad=efgrad
        ).energy.sum()
        fd_field[k] = (ep - em) / (2.0 * eps)
    torch.testing.assert_close(g_field, fd_field, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("etemp", [300.0, 5000.0])
def test_multi_orbital_noncontiguous_padding_and_potential_field(
    etemp: float,
) -> None:
    """Multi-orbital HF in (HF, H2O) bucket validates electron counts and potential+field."""
    numbers_hf = torch.tensor([9, 1], device=DEVICE)
    numbers_h2o, _ = _h2o_conformers()

    for scp_mode, field_val in (
        (labels.SCP_MODE_CHARGE, None),
        (
            labels.SCP_MODE_POTENTIAL,
            torch.tensor([0.002, -0.001, 0.0015], dtype=DTYPE, device=DEVICE),
        ),
    ):
        _, sys_hf = _build_system(
            numbers_hf,
            scp_mode=scp_mode,
            mixer=labels.MIXER_ANDERSON,
            etemp=etemp,
            int_level=labels.INTLEVEL_DIPOLE,
        )
        _, sys_h2o = _build_system(
            numbers_h2o,
            scp_mode=scp_mode,
            mixer=labels.MIXER_ANDERSON,
            etemp=etemp,
            int_level=labels.INTLEVEL_DIPOLE,
        )
        assert sys_hf.exact_core is not None
        assert sys_h2o.exact_core is not None

        bucket = bucket_canonical_exact_cores(
            (sys_hf.exact_core, sys_h2o.exact_core)
        )
        padded_hf = pad_canonical_exact_core(sys_hf.exact_core, bucket, 0)
        mapping = dict(bucket.source_maps)[0]
        active_atoms = torch.tensor(
            mapping.atoms, dtype=torch.int64, device=DEVICE
        )
        active_orbs = torch.tensor(
            mapping.orbitals, dtype=torch.int64, device=DEVICE
        )
        # HF in (HF, H2O) has active_orbs = [0, 1, 2, 3, 5] (noncontiguous multi-orbital).
        assert not torch.equal(
            active_orbs, torch.arange(active_orbs.numel(), device=DEVICE)
        )

        pos_unpadded = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.73]],
                [[0.0, 0.0, 0.0], [0.0, 0.08, 1.82]],
            ],
            dtype=DTYPE,
            device=DEVICE,
        )
        inactive_mask = ~padded_hf.validity.atoms
        padded_list = []
        for idx in range(2):
            canon = canonicalize_positions(pos_unpadded[idx], bucket, 0).clone()
            canon[inactive_mask] = torch.tensor(
                [5.0 * (idx + 1), -3.0 * (idx + 1), 2.0],
                dtype=DTYPE,
                device=DEVICE,
            )
            padded_list.append(canon)
        pos_padded = torch.stack(padded_list, dim=0)

        p_unpad = pos_unpadded.clone().requires_grad_(True)
        p_pad = pos_padded.clone().requires_grad_(True)
        f_unpad = (
            None if field_val is None else field_val.clone().requires_grad_(True)
        )
        f_pad = (
            None if field_val is None else field_val.clone().requires_grad_(True)
        )

        res_unpad = evaluate_converged_root_batch(
            sys_hf, p_unpad, field=f_unpad
        )
        res_pad = evaluate_converged_root_batch(
            padded_hf, p_pad, field=f_pad
        )
        assert bool(res_pad.converged.all())

        torch.testing.assert_close(
            res_pad.energy.sum(dim=-1),
            res_unpad.energy.sum(dim=-1),
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            res_pad.fenergy.sum(dim=-1),
            res_unpad.fenergy.sum(dim=-1),
            atol=1e-10,
            rtol=1e-10,
        )
        grad_unpad = torch.autograd.grad(
            res_unpad.energy.sum(),
            (p_unpad,) if f_unpad is None else (p_unpad, f_unpad),
        )
        grad_pad = torch.autograd.grad(
            res_pad.energy.sum(),
            (p_pad,) if f_pad is None else (p_pad, f_pad),
        )
        torch.testing.assert_close(
            grad_pad[0][:, active_atoms, :],
            grad_unpad[0],
            atol=1e-9,
            rtol=1e-9,
        )
        torch.testing.assert_close(
            grad_pad[0][:, inactive_mask, :],
            torch.zeros_like(grad_pad[0][:, inactive_mask, :]),
            atol=1e-13,
            rtol=0.0,
        )
        if f_unpad is not None and f_pad is not None:
            torch.testing.assert_close(
                grad_pad[1], grad_unpad[1], atol=1e-9, rtol=1e-9
            )


def test_nested_jvp_and_outer_nel_tangent() -> None:
    """Nested jvp(jvp(...)) unwraps primal tensors cleanly and propagates outer nel tangents."""
    from dxtb._src.scf.functional.spectral import (
        fermi_density_matrix,
        fermi_free_energy_matrix,
        fermi_free_energy_trace,
    )
    from dxtb._src.wavefunction.filling import _get_fermi_occupation_core_state

    # 1. Nested jvp(jvp(...)) through evaluate_converged_root_batch(system, pos)
    numbers, base_positions = _h2_conformers()
    _, system = _build_system(numbers, etemp=300.0)
    z_dir = torch.zeros_like(base_positions)
    z_dir[:, 1, 2] = 1.0

    def energy_from_scratch(s: torch.Tensor) -> torch.Tensor:
        return evaluate_converged_root_batch(
            system, base_positions + s * z_dir
        ).energy.sum()

    s0 = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
    one = torch.tensor(1.0, dtype=DTYPE, device=DEVICE)
    _, d2_nested_jvp = jvp(
        lambda s: jvp(energy_from_scratch, (s,), (one,))[1], (s0,), (one,)
    )
    d2_grad = grad(grad(energy_from_scratch))(s0)
    torch.testing.assert_close(
        d2_nested_jvp, d2_grad, atol=1e-9, rtol=1e-9
    )

    # 2. Outer JVP on nel inside inner JVP on hamiltonian in spectral.py
    h0 = torch.tensor(
        [[-0.45, -0.12], [-0.12, 0.25]], dtype=DTYPE, device=DEVICE
    )
    dh = torch.tensor(
        [[0.03, -0.01], [-0.01, -0.02]], dtype=DTYPE, device=DEVICE
    )
    kt = torch.tensor(0.02, dtype=DTYPE, device=DEVICE)
    nel0 = torch.tensor([1.0], dtype=DTYPE, device=DEVICE)
    dnel = torch.tensor([1.0], dtype=DTYPE, device=DEVICE)

    def _eval_spectral(h: torch.Tensor, nel: torch.Tensor) -> torch.Tensor:
        evals = torch.linalg.eigvalsh(h).unsqueeze(0)
        state = _get_fermi_occupation_core_state(nel, evals, kt)
        dens = fermi_density_matrix(h, nel, kt, state)
        fmat = fermi_free_energy_matrix(h, nel, kt, state)
        ftr = fermi_free_energy_trace(h, nel, kt, state, l_inv=None)
        return (dens * h).sum() + fmat.diagonal().sum() + ftr.sum()

    def _inner_h_jvp(nel: torch.Tensor) -> torch.Tensor:
        return jvp(lambda h: _eval_spectral(h, nel), (h0,), (dh,))[1]

    _, cross_deriv = jvp(_inner_h_jvp, (nel0,), (dnel,))
    eps = 1.0e-5
    fd_cross = (
        _inner_h_jvp(nel0 + eps * dnel) - _inner_h_jvp(nel0 - eps * dnel)
    ) / (2.0 * eps)
    torch.testing.assert_close(cross_deriv, fd_cross, atol=1e-7, rtol=1e-6)


def test_conformer_batch_size_sweep_1_2_4() -> None:
    """Batch sizes 1, 2, and 4 produce identical per-lane energies and gradients."""
    numbers, base_positions = _h2o_conformers()
    c3 = base_positions[0].clone()
    c3[1, 2] -= 0.06
    c3[2, 1] += 0.05
    pos4 = torch.cat((base_positions, c3.unsqueeze(0)), dim=0)
    _, system = _build_system(numbers, etemp=300.0)

    p4 = pos4.clone().requires_grad_(True)
    res4 = evaluate_converged_root_batch(system, p4)
    g4 = torch.autograd.grad(res4.energy.sum(), p4)[0]

    for n in (1, 2):
        pn = pos4[:n].clone().requires_grad_(True)
        resn = evaluate_converged_root_batch(system, pn)
        gn = torch.autograd.grad(resn.energy.sum(), pn)[0]
        torch.testing.assert_close(
            resn.energy, res4.energy[:n], atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(gn, g4[:n], atol=1e-11, rtol=1e-11)


def test_public_singlepoint_contract_preserved() -> None:
    """Public System.singlepoint() remains fixed-trip and exposes no batch API."""
    numbers, base_positions = _h2_conformers()
    _, system = _build_system(numbers, maxiter=2, force_convergence=False)
    assert not hasattr(system, "singlepoint_batch")
    assert not hasattr(system, "evaluate_converged_root_batch")
    pos = base_positions[0].clone().requires_grad_(True)
    res = system.singlepoint(pos)
    g = torch.autograd.grad(res.energy.sum(), pos)[0]
    assert bool(torch.isfinite(g).all())
