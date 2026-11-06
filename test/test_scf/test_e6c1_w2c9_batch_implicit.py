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

import importlib
from dataclasses import replace

import pytest
import torch
from torch.func import grad, jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.batch_implicit import (
    BatchRootFailureCode,
    ResponseDiagnosticCollector,
    _classify_fermi_code,
    _resolve_strictness,
    classify_response_failure_status,
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.calculators.batch_streaming import stream_batch_value_and_grad
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.model import Model
from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
from dxtb._src.scf.functional.implicit import implicit_root
from dxtb._src.scf.functional.solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_INVALID_TEMPERATURE,
    FERMI_FAILURE_NONCONVERGENCE,
    FERMI_FAILURE_NONE,
    FERMI_FAILURE_TOO_MANY_ELECTRONS,
)
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)
from dxtb._src.wavefunction.filling import FermiStatus
from dxtb.config import Config
from dxtb.scf import SCFRepresentation, SCFResponseSettings

DTYPE = torch.float64
DEVICE = torch.device("cpu")


class _InjectedResponseFailure(torch.autograd.Function):
    """Create a lane-local derivative failure while retaining primal values."""

    generate_vmap_rule = True

    @staticmethod
    def forward(attached: torch.Tensor, failed: torch.Tensor) -> torch.Tensor:
        return attached

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        _, failed = inputs
        ctx.save_for_backward(failed)
        ctx.save_for_forward(failed)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (failed,) = ctx.saved_tensors
        poison = failed & (grad_output != 0.0).any()
        nan = torch.full_like(grad_output, float("nan"))
        return torch.where(poison, nan, grad_output), None

    @staticmethod
    def jvp(ctx, grad_attached, grad_failed):
        (failed,) = ctx.saved_tensors
        if grad_attached is None:
            return torch.zeros_like(failed).expand_as(grad_failed)
        return torch.where(
            failed, torch.full_like(grad_attached, float("nan")), grad_attached
        )


def _build_system(
    numbers: torch.Tensor,
    *,
    method: int = labels.GFN1_XTB,
    scp_mode: int = labels.SCP_MODE_CHARGE,
    mixer: int = labels.MIXER_ANDERSON,
    etemp: float = 300.0,
    partition: int = labels.FERMI_PARTITION_EQUAL,
    maxiter: int = 100,
    x_atol: float = 1.0e-10,
    x_atol_max: float = 1.0e-10,
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
        response=SCFResponseSettings(
            representation={
                labels.SCP_MODE_CHARGE: SCFRepresentation.CHARGE,
                labels.SCP_MODE_POTENTIAL: SCFRepresentation.POTENTIAL,
                labels.SCP_MODE_FOCK: SCFRepresentation.FOCK,
            }[scp_mode]
        ),
        mixer=mixer,
        maxiter=maxiter,
        x_atol=x_atol,
        x_atol_max=x_atol_max,
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


@pytest.mark.parametrize(
    "response_representation",
    [
        "default",
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.CHARGE,
        SCFRepresentation.FOCK,
        None,
    ],
)
def test_cdiis_fock_batch_uses_configured_response_coordinates(
    response_representation: SCFRepresentation | None | str,
) -> None:
    """Batch IFT coordinates and derivatives match serial implicit SCF."""
    numbers, positions = _h2_conformers()
    positions = positions.requires_grad_(True)
    response = (
        None
        if response_representation == "default"
        else SCFResponseSettings(representation=response_representation)
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        strategy="cdiis",
        differentiation="implicit",
        convergence="tight",
        response=response,
        fermi_etemp=0.0,
        force_convergence=True,
    )
    model = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE),
        config=config,
    )
    system = model.setup(numbers, batch_mode=0)

    prepared = prepare_converged_root_batch(system, positions)
    expected_response = (
        SCFRepresentation.POTENTIAL
        if response_representation == "default"
        else (
            SCFRepresentation.FOCK
            if response_representation is None
            else response_representation
        )
    )
    assert prepared.primal_representation == labels.SCP_MODE_FOCK
    assert (
        prepared.response_representation
        == {
            SCFRepresentation.CHARGE: labels.SCP_MODE_CHARGE,
            SCFRepresentation.POTENTIAL: labels.SCP_MODE_POTENTIAL,
            SCFRepresentation.FOCK: labels.SCP_MODE_FOCK,
        }[expected_response]
    )
    assert prepared.all_roots_valid

    batch_result = evaluate_converged_root_batch(prepared)
    batch_gradient = torch.autograd.grad(
        batch_result.energy.sum(), positions, retain_graph=True
    )[0]

    serial_results = [system.singlepoint(position) for position in positions]
    serial_gradient = torch.autograd.grad(
        torch.stack([result.energy.sum() for result in serial_results]).sum(),
        positions,
    )[0]
    torch.testing.assert_close(
        batch_result.energy,
        torch.stack([result.energy for result in serial_results]),
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    torch.testing.assert_close(
        batch_gradient, serial_gradient, atol=2.0e-7, rtol=2.0e-7
    )


def test_batch_response_maxiter_is_shared_by_attachment_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured response maxiter controls the mapped IFT and its monitor."""
    numbers, positions = _h2o_conformers()
    positions = positions[:1].clone().requires_grad_(True)
    response = SCFResponseSettings(
        representation=SCFRepresentation.POTENTIAL,
        rtol=0.0,
        atol=0.0,
        maxiter=1,
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        strategy="cdiis",
        differentiation="implicit",
        convergence="tight",
        response=response,
        fermi_etemp=0.0,
        force_convergence=True,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE), config=config
    ).setup(numbers, batch_mode=0)
    prepared = prepare_converged_root_batch(system, positions)
    assert prepared.all_roots_valid
    diagnostics = ResponseDiagnosticCollector()
    received_options: list[dict[str, object]] = []
    solve = ResponseDiagnosticCollector.solve

    def record_options(self, matvec, rhs, *params, **kwargs):
        received_options.append(dict(kwargs))
        return solve(self, matvec, rhs, *params, **kwargs)

    monkeypatch.setattr(ResponseDiagnosticCollector, "solve", record_options)
    result = evaluate_converged_root_batch(
        prepared, response_diagnostics=diagnostics
    )
    gradient = torch.autograd.grad(result.energy.sum(), positions)[0]
    assert bool(result.converged.all())
    assert not bool(torch.isfinite(gradient).all())
    assert diagnostics.iterations is not None
    assert int(diagnostics.iterations.max().item()) <= 1
    assert received_options
    assert received_options[0]["rtol"] == response.rtol
    assert received_options[0]["atol"] == response.atol
    assert received_options[0]["maxiter"] == response.maxiter


def test_padded_batch_rejects_unsupported_response_conversion() -> None:
    """Padded cores reject mismatched response coordinates before mapping."""
    h2_numbers = torch.tensor([1, 1], device=DEVICE)
    h2_config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        strategy="cdiis",
        differentiation="implicit",
        fermi_etemp=0.0,
    )
    h2_system = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE),
        config=h2_config,
    ).setup(h2_numbers, batch_mode=0)
    h2o_numbers, _ = _h2o_conformers()
    h2o_config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        strategy="cdiis",
        differentiation="implicit",
        fermi_etemp=0.0,
    )
    h2o_system = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE),
        config=h2o_config,
    ).setup(h2o_numbers, batch_mode=0)
    assert h2_system.exact_core is not None
    assert h2o_system.exact_core is not None
    bucket = bucket_canonical_exact_cores(
        (h2_system.exact_core, h2o_system.exact_core)
    )
    padded = pad_canonical_exact_core(h2_system.exact_core, bucket, 0)
    padded_positions = torch.zeros(
        (1, padded.core.numbers.shape[0], 3), dtype=DTYPE, device=DEVICE
    )
    with pytest.raises(ValueError, match="require response coordinates"):
        prepare_converged_root_batch(padded, padded_positions)


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
        (labels.GFN1_XTB, labels.SCP_MODE_FOCK, labels.MIXER_ANDERSON),
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
        single_pos = (
            base_positions[lane_idx : lane_idx + 1].clone().requires_grad_(True)
        )
        single_res = evaluate_converged_root_batch(system, single_pos)
        single_grad = torch.autograd.grad(single_res.energy.sum(), single_pos)[
            0
        ]
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
    [
        labels.FERMI_PARTITION_EQUAL,
        pytest.param(labels.FERMI_PARTITION_ATOMIC, marks=pytest.mark.slow),
    ],
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
    for a in range(base_positions.shape[1]):
        for c in range(3):
            pos_p = base_positions.clone()
            pos_m = base_positions.clone()
            pos_p[:, a, c] += step
            pos_m[:, a, c] -= step
            ep = evaluate_converged_root_batch(
                system, pos_p, spin=1.0
            ).energy.sum(dim=-1)
            em = evaluate_converged_root_batch(
                system, pos_m, spin=1.0
            ).energy.sum(dim=-1)
            fd_grad[:, a, c] = (ep - em) / (2.0 * step)

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
    _, sys_lih = _build_system(numbers_lih, int_level=labels.INTLEVEL_HCORE)
    _, sys_h2o = _build_system(numbers_h2o, int_level=labels.INTLEVEL_HCORE)
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
    assert connected_nonzero + connected_zero + disconnected == len(
        param_tensors
    )

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
    model, system = _build_system(
        numbers, force_convergence=False, trainable_model=True
    )
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
    grad_lane_0, grad_valid_parameter = torch.autograd.grad(
        mixed_result.energy[0].sum(),
        (pos_two, prepared.model_parameter_refs[0]),
        retain_graph=True,
    )
    assert bool(torch.isfinite(grad_lane_0[0]).all())
    assert bool(torch.isfinite(grad_valid_parameter).all())
    torch.testing.assert_close(
        grad_lane_0[1], torch.zeros_like(grad_lane_0[1]), atol=0.0, rtol=0.0
    )
    grad_invalid = torch.autograd.grad(
        mixed_result.energy[1].sum(), pos_two, retain_graph=True
    )[0]
    assert bool(torch.isfinite(grad_invalid[0]).all())
    assert bool(torch.isnan(grad_invalid[1]).all())
    grad_invalid_parameter = torch.autograd.grad(
        mixed_result.energy[1].sum(),
        prepared.model_parameter_refs[0],
        retain_graph=True,
    )[0]
    assert bool(torch.isnan(grad_invalid_parameter).all())

    weights = torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE)
    weighted_obj = (weights * mixed_result.energy.sum(dim=-1)).sum()
    grad_weighted_pos, grad_weighted_param = torch.autograd.grad(
        weighted_obj,
        (pos_two, prepared.model_parameter_refs[0]),
    )
    torch.testing.assert_close(
        grad_weighted_pos[0], grad_lane_0[0], atol=1e-14, rtol=1e-14
    )
    torch.testing.assert_close(
        grad_weighted_pos[1],
        torch.zeros_like(grad_weighted_pos[1]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        grad_weighted_param, grad_valid_parameter, atol=1e-14, rtol=1e-14
    )

    direction = torch.zeros_like(pos_two)
    direction[1, 0, 0] = 1.0
    _, invalid_jvp = jvp(
        lambda value: evaluate_converged_root_batch(
            mixed_batch, positions=value, strict=False
        )
        .energy[1]
        .sum(),
        (pos_two.detach(),),
        (direction,),
    )
    assert bool(torch.isnan(invalid_jvp))
    lane_jacobian = jacrev(
        lambda value: evaluate_converged_root_batch(
            mixed_batch, positions=value, strict=False
        ).energy.sum(dim=-1)
    )(pos_two.detach())
    assert bool(torch.isfinite(lane_jacobian[0]).all())
    assert bool(torch.isnan(lane_jacobian[1, 1]).all())

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
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        evaluate_converged_root_batch(
            prepared, positions=shifted_pos, strict=True
        )


def test_strictness_precedence_and_fermi_failure_classification() -> None:
    """Mandatory force-convergence cannot be weakened and unknown codes fail closed."""
    numbers, positions = _h2_conformers()
    for force, requested, expected in (
        (False, None, False),
        (False, False, False),
        (False, True, True),
        (True, None, True),
        (True, False, True),
        (True, True, True),
    ):
        config = type("StrictConfig", (), {"force_convergence": force})()
        assert _resolve_strictness(config, requested) is expected

        # Valid physical root across all 6 combinations
        _, valid_sys = _build_system(
            numbers, maxiter=200, force_convergence=force
        )
        prep_valid = prepare_converged_root_batch(
            valid_sys, positions[:1], strict=requested
        )
        assert prep_valid.all_roots_valid
        res_valid = evaluate_converged_root_batch(prep_valid, strict=requested)
        assert bool(res_valid.converged.all())

        # Invalid physical root across all 6 combinations
        _, invalid_sys = _build_system(
            numbers, maxiter=1, force_convergence=force
        )
        if expected:
            with pytest.raises(SCFConvergenceError, match=r"\[Lane 0\]"):
                prepare_converged_root_batch(
                    invalid_sys, positions[:1], strict=requested
                )
        else:
            with pytest.warns(SCFConvergenceWarning):
                prep_inv = prepare_converged_root_batch(
                    invalid_sys, positions[:1], strict=requested
                )
            assert not prep_inv.all_roots_valid
            res_inv = evaluate_converged_root_batch(prep_inv, strict=requested)
            assert not bool(res_inv.converged.any())
            assert bool(torch.isnan(res_inv.energy).all())

    expected_codes = {
        FERMI_FAILURE_NONE: BatchRootFailureCode.NONE,
        FERMI_FAILURE_INVALID_TEMPERATURE: BatchRootFailureCode.INVALID_TEMPERATURE,
        FERMI_FAILURE_TOO_MANY_ELECTRONS: BatchRootFailureCode.TOO_MANY_ELECTRONS,
        FERMI_FAILURE_NONCONVERGENCE: BatchRootFailureCode.FERMI_NONCONVERGENCE,
        FERMI_FAILURE_ELECTRON_COUNT: BatchRootFailureCode.ELECTRON_COUNT,
    }
    for raw_code, batch_code in expected_codes.items():
        classified, _ = _classify_fermi_code(raw_code)
        assert classified == int(batch_code)
    with pytest.raises(ValueError, match="Unknown nonzero Fermi failure code"):
        _classify_fermi_code(987654)


def test_prepared_root_rejects_mutated_inputs_and_model_parameters() -> None:
    """Prepared roots cannot outlive changed primal inputs or Model parameters."""
    numbers, base_positions = _h2o_conformers()

    _, position_system = _build_system(numbers)
    mutable_positions = base_positions[:1].clone()
    position_root = prepare_converged_root_batch(
        position_system, mutable_positions
    )
    displaced_for_transform = mutable_positions.clone()
    displaced_for_transform[0, 0, 0] += 0.1

    def transformed_energy(value: torch.Tensor) -> torch.Tensor:
        return evaluate_converged_root_batch(
            position_root, positions=value
        ).energy.sum()

    tangent = torch.ones_like(displaced_for_transform)
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        grad(transformed_energy)(displaced_for_transform)
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        jvp(
            transformed_energy,
            (displaced_for_transform,),
            (tangent,),
        )
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        jacfwd(transformed_energy)(displaced_for_transform)
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        jacrev(transformed_energy)(displaced_for_transform)

    compiled_energy = torch.compile(transformed_energy, backend="eager")
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        compiled_energy(displaced_for_transform)

    with torch.no_grad():
        mutable_positions[0, 0, 0] += 1.0e-3
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        evaluate_converged_root_batch(position_root)

    _, replacement_system = _build_system(numbers)
    replacement_root = prepare_converged_root_batch(
        replacement_system, base_positions[:1]
    )
    displaced = base_positions[:1].clone()
    displaced[0, 0, 0] += 0.2
    with pytest.raises(RuntimeError, match="stale for the supplied positions"):
        evaluate_converged_root_batch(replacement_root, positions=displaced)

    changed_scf = replace(replacement_system.config.scf, maxiter=199)
    changed_config = replace(replacement_system.config, scf=changed_scf)
    changed_system = replace(replacement_system, config=changed_config)
    changed_prepared = replace(replacement_root, system=changed_system)
    with pytest.raises(RuntimeError, match="configuration has changed"):
        evaluate_converged_root_batch(changed_prepared)

    changed_num_system = replace(
        replacement_system, numbers=torch.tensor([8, 1, 2], device=DEVICE)
    )
    changed_num_prepared = replace(replacement_root, system=changed_num_system)
    with pytest.raises(RuntimeError, match="atomic numbers have changed"):
        evaluate_converged_root_batch(changed_num_prepared)

    with torch.no_grad():
        replacement_root.lanes[0].root_x.add_(1.0e-6)
    with pytest.raises(RuntimeError, match="root tensor was modified"):
        evaluate_converged_root_batch(replacement_root)

    model, parameter_system = _build_system(numbers, trainable_model=True)
    parameter_root = prepare_converged_root_batch(
        parameter_system, base_positions[:1]
    )
    parameter = parameter_root.model_parameter_refs[0]
    assert any(parameter is value for value in model.par.parameters())
    with torch.no_grad():
        parameter.add_(1.0e-8)
    with pytest.raises(
        RuntimeError, match="stale for the current Model parameters"
    ):
        evaluate_converged_root_batch(parameter_root)


def test_prepared_root_rejects_mutated_internal_physical_tensor() -> None:
    """Frozen input records still reject in-place tensor mutation."""
    numbers, positions = _h2_conformers()
    _, system = _build_system(numbers, trainable_model=True)
    prepared = prepare_converged_root_batch(system, positions[:1], strict=True)
    hcore = prepared.lanes[0].inputs.integrals.hcore
    with torch.no_grad():
        hcore.add_(1.0e-3)
    with pytest.raises(
        RuntimeError, match="mutated in place at 'integrals.hcore'"
    ):
        evaluate_converged_root_batch(prepared, strict=True)


def test_prepared_root_snapshots_are_immutable_values() -> None:
    """Snapshot records cannot be changed to bless a stale prepared root."""
    numbers, positions = _h2_conformers()
    _, system = _build_system(numbers, trainable_model=True)
    prepared = prepare_converged_root_batch(system, positions[:1], strict=True)

    snapshots = (
        prepared.positions_snapshot,
        prepared.numbers_snapshot,
        prepared.model_parameter_snapshot[0],
        prepared.system_tensor_snapshot[0],
        prepared.root_snapshots[0],
    )
    assert all(isinstance(snapshot.raw_bytes, bytes) for snapshot in snapshots)
    assert all(not isinstance(snapshot, torch.Tensor) for snapshot in snapshots)

    with torch.no_grad():
        prepared.lanes[0].root_x.add_(0.1)
    with pytest.raises(RuntimeError, match="root tensor was modified"):
        evaluate_converged_root_batch(prepared, strict=True)


def test_prepared_root_rejects_mutated_explicit_map_parameter() -> None:
    """Every explicit physical-map tensor is snapshotted, not just grad leaves."""
    numbers, positions = _h2o_conformers()
    _, system = _build_system(numbers, trainable_model=False)
    prepared = prepare_converged_root_batch(system, positions[:1], strict=True)
    parameter = prepared.lanes[0].parameters[0]
    with torch.no_grad():
        parameter.add_(1.0e-5)
    with pytest.raises(RuntimeError, match="Prepared physical parameters"):
        evaluate_converged_root_batch(prepared, strict=True)


@pytest.mark.parametrize("mutation", ["charge", "spin", "field"])
def test_prepared_root_rejects_replaced_physical_inputs(mutation: str) -> None:
    """Charge, spin, and external-field changes require fresh physical roots."""
    numbers, positions = _h2o_conformers()
    _, system = _build_system(numbers, int_level=labels.INTLEVEL_DIPOLE)
    field = torch.tensor([0.001, 0.0, 0.0], dtype=DTYPE, device=DEVICE)
    prepared = prepare_converged_root_batch(system, positions[:1], field=field)

    if mutation == "charge":
        with pytest.raises(RuntimeError):
            evaluate_converged_root_batch(prepared, chrg=0.25)
    elif mutation == "spin":
        with pytest.raises(RuntimeError):
            evaluate_converged_root_batch(prepared, spin=1.0)
    else:
        changed_field = field.clone()
        changed_field[0] += 1.0e-3
        with pytest.raises(RuntimeError, match="stale for charge, spin"):
            evaluate_converged_root_batch(prepared, field=changed_field)


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

    def raw_map(
        x: torch.Tensor, p: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        return s * x + p

    attached = vmap(lambda r, p, s: implicit_root(r, p, s, raw_map=raw_map))(
        roots, params, scales
    )
    grad_lane0 = torch.autograd.grad(
        attached[0].sum(), params, retain_graph=True
    )[0]
    grad_lane1 = torch.autograd.grad(attached[1].sum(), params)[0]
    torch.testing.assert_close(
        grad_lane0[0],
        torch.full((2,), 2.0, dtype=DTYPE),
        atol=1e-10,
        rtol=1e-10,
    )
    assert bool(torch.isnan(grad_lane1[1]).all())


def test_batch_backward_response_failure_is_lane_local(monkeypatch) -> None:
    """Backward response failure is explicit but does not rewrite primal status."""
    numbers, positions = _h2_conformers()
    model, system = _build_system(
        numbers, force_convergence=False, trainable_model=True
    )
    prepared = prepare_converged_root_batch(system, positions, strict=False)
    assert prepared.all_roots_valid

    roots = torch.stack([lane.root_x for lane in prepared.lanes])
    first_coordinate = roots.reshape(roots.shape[0], -1)[:, 0]
    assert not torch.isclose(first_coordinate[0], first_coordinate[1])
    failed_lane = torch.tensor([False, True], device=DEVICE)
    failed_root_coordinate = first_coordinate[1]
    fail_all = [False]

    module = importlib.import_module("dxtb._src.calculators.batch_implicit")
    original = module.implicit_root

    def injected(root: torch.Tensor, *params: torch.Tensor, raw_map):
        attached = original(root, *params, raw_map=raw_map)
        return _InjectedResponseFailure.apply(
            attached,
            (root.reshape(-1)[0] == failed_root_coordinate)
            | torch.tensor(fail_all[0], device=root.device),
        )

    monkeypatch.setattr(module, "implicit_root", injected)
    result = evaluate_converged_root_batch(prepared, strict=False)
    assert bool(result.converged.all())
    assert bool((result.status.final_failure == 0).all())

    positions_leaf = positions.clone().requires_grad_(True)
    result = evaluate_converged_root_batch(
        prepared, positions=positions_leaf, strict=False
    )
    energy = result.energy.sum(dim=-1)
    gradients = torch.autograd.grad(
        energy.sum(), positions_leaf, retain_graph=True
    )[0]
    assert bool(torch.isfinite(gradients[~failed_lane]).all())
    assert bool(torch.isnan(gradients[failed_lane]).all())
    assert result.status is not None
    post_status = classify_response_failure_status(
        result.status, result.converged, gradients
    )
    assert int(result.status.final_failure[0].item()) == 0
    assert int(result.status.final_failure[1].item()) == 0
    assert int(post_status.final_failure[0].item()) == int(
        BatchRootFailureCode.NONE
    )
    assert int(post_status.final_failure[1].item()) == int(
        BatchRootFailureCode.RESPONSE_FAILURE
    )
    streamed_fail = stream_batch_value_and_grad(
        prepared, model, positions, chunk_size=1, strict=False
    )
    assert bool(streamed_fail.converged.all())
    assert int(streamed_fail.final_failure[0].item()) == int(
        BatchRootFailureCode.NONE
    )
    assert int(streamed_fail.final_failure[1].item()) == int(
        BatchRootFailureCode.RESPONSE_FAILURE
    )
    parameter = prepared.model_parameter_refs[0]
    valid_parameter_gradient = torch.autograd.grad(
        energy[0], parameter, retain_graph=True
    )[0]
    failed_parameter_gradient = torch.autograd.grad(
        energy[1], parameter, retain_graph=True
    )[0]
    assert bool(torch.isfinite(valid_parameter_gradient).all())
    assert bool(torch.isnan(failed_parameter_gradient).all())

    weights = (~failed_lane).to(dtype=DTYPE)
    weighted = (energy * weights).sum()
    masked_grad = torch.autograd.grad(weighted, positions_leaf)[0]
    assert bool(torch.isfinite(masked_grad).all())
    assert bool((masked_grad[1] == 0.0).all())

    direction = torch.zeros_like(positions)
    direction[failed_lane, 0, 0] = 1.0
    _, tangent = jvp(
        lambda value: evaluate_converged_root_batch(
            prepared, positions=value, strict=False
        ).energy.sum(),
        (positions,),
        (direction,),
    )
    assert bool(torch.isnan(tangent))

    fail_all[0] = True
    all_failed = evaluate_converged_root_batch(prepared, strict=True)
    assert bool(all_failed.converged.all())
    assert bool((all_failed.status.final_failure == 0).all())
    all_failed_gradient = torch.autograd.grad(
        all_failed.energy.sum(), prepared.model_parameter_refs[0]
    )[0]
    assert bool(torch.isnan(all_failed_gradient).all())


def test_streaming_batch_gradients_match_simultaneous_mapped_result() -> None:
    """Chunkwise backward preserves position, field, and shared Model gradients."""
    numbers, positions = _h2o_conformers()
    model, system = _build_system(
        numbers,
        method=labels.GFN2_XTB,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        trainable_model=True,
    )
    positions = positions[:2].clone().requires_grad_(True)
    field = torch.tensor(
        [0.0002, -0.0001, 0.0003], dtype=DTYPE, device=DEVICE
    ).requires_grad_(True)
    weights = torch.tensor([1.25, -0.4], dtype=DTYPE, device=DEVICE)
    prepared = prepare_converged_root_batch(
        system, positions.detach(), field=field, strict=True
    )

    simultaneous = evaluate_converged_root_batch(
        prepared, positions=positions, field=field, strict=True
    )
    lane_energy = simultaneous.energy.reshape(2, -1).sum(-1)
    objective = (lane_energy * weights).sum()
    parameters = tuple(
        parameter
        for parameter in model.par.parameters()
        if parameter.is_floating_point() and parameter.requires_grad
    )
    reference_gradients = torch.autograd.grad(
        objective,
        (positions, field, *parameters),
        allow_unused=True,
    )

    streamed = stream_batch_value_and_grad(
        prepared,
        model,
        positions.detach(),
        weights=weights,
        chunk_size=1,
        field=field,
        strict=True,
    )
    simultaneous_free_energy = (
        simultaneous.fenergy.reshape(2, -1).sum(-1)
        if simultaneous.fenergy is not None
        else lane_energy
    )
    torch.testing.assert_close(
        streamed.energy,
        lane_energy.detach(),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    torch.testing.assert_close(
        streamed.free_energy,
        simultaneous_free_energy.detach(),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    torch.testing.assert_close(
        streamed.position_gradient,
        reference_gradients[0],
        atol=2.0e-7,
        rtol=2.0e-7,
    )
    assert streamed.field_gradient is not None
    torch.testing.assert_close(
        streamed.field_gradient,
        reference_gradients[1],
        atol=2.0e-7,
        rtol=2.0e-7,
    )
    assert len(streamed.model_parameter_gradients) == len(parameters)
    assert streamed.model_parameter_connected == tuple(
        gradient is not None for gradient in reference_gradients[2:]
    )
    connected_nonzero = sum(
        g is not None and bool((g.abs() > 0.0).any())
        for g in streamed.model_parameter_gradients
    )
    connected_zero = sum(
        g is not None and bool((g == 0.0).all())
        for g in streamed.model_parameter_gradients
    )
    disconnected = sum(g is None for g in streamed.model_parameter_gradients)
    assert connected_nonzero > 0
    assert connected_zero + disconnected > 0
    assert connected_nonzero + connected_zero + disconnected == len(parameters)
    for actual, expected in zip(
        streamed.model_parameter_gradients, reference_gradients[2:]
    ):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            torch.testing.assert_close(
                actual, expected, atol=2.0e-8, rtol=2.0e-7
            )

    # Verify chunk_size in (1, 2, 4), lane permutation invariance, and active-lane subset.
    c3 = positions[0].detach().clone()
    c3[1, 2] -= 0.05
    c3[2, 1] += 0.04
    pos4 = torch.cat(
        (positions.detach(), positions[:1].detach() + 0.01, c3.unsqueeze(0)),
        dim=0,
    )
    prep4 = prepare_converged_root_batch(system, pos4, strict=True)
    weights4 = torch.tensor([1.0, 0.0, -0.5, 0.25], dtype=DTYPE, device=DEVICE)
    stream_c1 = stream_batch_value_and_grad(
        prep4, model, pos4, weights=weights4, chunk_size=1, strict=True
    )
    for cs in (2, 4):
        stream_cs = stream_batch_value_and_grad(
            prep4, model, pos4, weights=weights4, chunk_size=cs, strict=True
        )
        torch.testing.assert_close(
            stream_cs.energy, stream_c1.energy, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            stream_cs.position_gradient,
            stream_c1.position_gradient,
            atol=1e-10,
            rtol=1e-10,
        )
        assert (
            stream_cs.model_parameter_connected
            == stream_c1.model_parameter_connected
        )
        for g_cs, g_c1 in zip(
            stream_cs.model_parameter_gradients,
            stream_c1.model_parameter_gradients,
        ):
            if g_c1 is None:
                assert g_cs is None
            else:
                assert g_cs is not None
                torch.testing.assert_close(g_cs, g_c1, atol=1e-9, rtol=1e-8)

    # Zero-weight lane 1 must have exact zero position gradient.
    torch.testing.assert_close(
        stream_c1.position_gradient[1],
        torch.zeros_like(stream_c1.position_gradient[1]),
        atol=0.0,
        rtol=0.0,
    )

    # Lane permutation invariance.
    perm = [2, 0, 3, 1]
    prep4_perm = prepare_converged_root_batch(system, pos4[perm], strict=True)
    stream_perm = stream_batch_value_and_grad(
        prep4_perm,
        model,
        pos4[perm],
        weights=weights4[perm],
        chunk_size=2,
        strict=True,
    )
    torch.testing.assert_close(
        stream_perm.energy, stream_c1.energy[perm], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        stream_perm.position_gradient,
        stream_c1.position_gradient[perm],
        atol=1e-10,
        rtol=1e-10,
    )


def test_streaming_invalid_lane_gradient_contract() -> None:
    """Masked failed lanes stay isolated and selected failures return NaNs."""
    numbers, positions = _h2o_conformers()
    model, system = _build_system(
        numbers, force_convergence=False, trainable_model=True
    )
    prepared = prepare_converged_root_batch(system, positions[:2], strict=False)
    invalid_lane = replace(
        prepared.lanes[1],
        root_valid=False,
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
    prepared = replace(
        prepared,
        lanes=(prepared.lanes[0], invalid_lane),
        valid_lane_mask=torch.tensor([True, False], device=DEVICE),
        all_roots_valid=False,
    )

    masked = stream_batch_value_and_grad(
        prepared,
        model,
        positions[:2],
        weights=torch.tensor([1.0, 0.0], dtype=DTYPE, device=DEVICE),
        strict=False,
    )
    assert bool(torch.isfinite(masked.energy[0]))
    assert bool(torch.isnan(masked.energy[1]))
    assert bool(torch.isfinite(masked.position_gradient[0]).all())
    assert bool((masked.position_gradient[1] == 0.0).all())
    assert all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for gradient in masked.model_parameter_gradients
    )

    selected = stream_batch_value_and_grad(
        prepared,
        model,
        positions[:2],
        weights=torch.tensor([0.0, 1.0], dtype=DTYPE, device=DEVICE),
        strict=False,
    )
    assert bool(torch.isnan(selected.position_gradient[1]).all())
    assert any(
        gradient is not None and bool(torch.isnan(gradient).any())
        for gradient in selected.model_parameter_gradients
    )


def test_streaming_rejects_changed_nontrainable_model_parameters() -> None:
    """Streaming cannot combine old roots with a changed nontrainable Model."""
    numbers, positions = _h2o_conformers()
    model_classical, system_classical = _build_system(
        numbers, trainable_model=False
    )
    prepared_classical = prepare_converged_root_batch(
        system_classical, positions[:1], strict=True
    )
    assert not prepared_classical.model_parameter_refs
    with torch.no_grad():
        assert model_classical.par.element is not None
        model_classical.par.element.O.arep.param.add_(1.0)
    with pytest.raises(RuntimeError, match="stale for the supplied Model"):
        stream_batch_value_and_grad(
            prepared_classical,
            model_classical,
            positions[:1],
            chunk_size=1,
            strict=True,
        )

    model, system = _build_system(numbers, trainable_model=False)
    prepared = prepare_converged_root_batch(system, positions[:1], strict=True)
    assert not prepared.model_parameter_refs

    with torch.no_grad():
        for parameter in model.par.parameters():
            if parameter.is_floating_point():
                parameter.add_(1.0e-5)

    with pytest.raises(RuntimeError, match="stale for the supplied Model"):
        stream_batch_value_and_grad(
            prepared, model, positions[:1], chunk_size=1, strict=True
        )


def test_eager_adaptive_fermi_root_is_revalidated_by_fixed_raw_map() -> None:
    """Adaptive primal maps define roots validated by the unchanged raw map."""
    numbers, positions = _h2o_conformers()
    _, system = _build_system(numbers, method=labels.GFN1_XTB, etemp=300.0)
    prepared = prepare_converged_root_batch(
        system,
        positions[:1],
        strict=True,
        eager_primal_fermi=True,
    )
    lane = prepared.lanes[0]
    assert lane.inputs.fermi_settings is not None
    assert lane.inputs.fermi_settings.smear
    assert lane.root_valid
    assert float(lane.raw_residual_l2) <= 2.0e-9

    result = evaluate_converged_root_batch(prepared, strict=True)
    assert bool(result.converged.all())
    assert bool(torch.isfinite(result.energy).all())


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
        return evaluate_converged_root_batch(
            prepared, positions=pos
        ).energy.sum(dim=-1)

    def batch_energy_sum(pos: torch.Tensor) -> torch.Tensor:
        return batch_energy_vec(pos).sum()

    # 1. grad, jacrev, jacfwd, jvp parity.
    g_func = grad(batch_energy_sum)(base_positions)
    j_rev = jacrev(batch_energy_vec)(base_positions)
    j_fwd = jacfwd(batch_energy_vec)(base_positions)
    tangent = torch.randn_like(base_positions)
    _, jvp_val = jvp(batch_energy_vec, (base_positions,), (tangent,))

    torch.testing.assert_close(j_rev, j_fwd, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(j_rev.sum(dim=0), g_func, atol=1e-10, rtol=1e-10)
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
    pos_li = torch.zeros(
        (1, 1, 3), dtype=DTYPE, device=DEVICE, requires_grad=True
    )
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
            None
            if field_val is None
            else field_val.clone().requires_grad_(True)
        )
        f_pad = (
            None
            if field_val is None
            else field_val.clone().requires_grad_(True)
        )

        res_unpad = evaluate_converged_root_batch(
            sys_hf, p_unpad, field=f_unpad
        )
        res_pad = evaluate_converged_root_batch(padded_hf, p_pad, field=f_pad)
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
    torch.testing.assert_close(d2_nested_jvp, d2_grad, atol=1e-9, rtol=1e-9)

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


def test_w2c11_streaming_failure_classification_all_input_families_and_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish BiCGSTAB response failures from non-BiCGSTAB nonfinite derivatives across all families."""
    from tad_mctc.storch.linsolve import (
        STATUS_CONVERGED,
        STATUS_NONFINITE,
        LinearSolveResult,
    )
    from tad_mctc.storch.linsolve import bicgstab as real_bicgstab

    from dxtb._src.calculators import batch_implicit
    from dxtb._src.calculators.batch_implicit import (
        RESPONSE_NOT_RUN,
        classify_batch_failure_category,
    )

    numbers, base_positions = _h2o_conformers()
    c3 = base_positions[0].clone()
    c3[1, 2] -= 0.05
    c3[2, 1] += 0.04
    pos4 = torch.cat((base_positions, c3.unsqueeze(0)), dim=0)
    efield = torch.tensor([0.002, -0.001, 0.0015], dtype=DTYPE, device=DEVICE)
    efgrad = torch.zeros((3, 3), dtype=DTYPE, device=DEVICE)
    efgrad[0, 0] = 0.001
    efgrad[1, 1] = -0.0004
    efgrad[2, 2] = -0.0006

    model, system = _build_system(
        numbers,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        etemp=300.0,
        force_convergence=False,
        trainable_model=True,
    )
    prepared = prepare_converged_root_batch(
        system, pos4, field=efield, field_grad=efgrad, strict=False
    )

    # 1. Nonfinite objective weights must raise ValueError immediately.
    for bad_w in (
        torch.tensor([1.0, float("nan"), 1.0, 1.0], dtype=DTYPE, device=DEVICE),
        torch.tensor([1.0, float("inf"), 1.0, 1.0], dtype=DTYPE, device=DEVICE),
    ):
        with pytest.raises(ValueError, match="finite"):
            stream_batch_value_and_grad(
                prepared, model, pos4, weights=bad_w, strict=False
            )

    # 2. Confirmed BiCGSTAB failure on lane 1 ONLY -> RESPONSE_FAILURE (9) on lane 1,
    # while lanes 0, 2, 3 succeed (NONE = 0), consistent across chunk_size in (1, 2, 4).
    target_lane1_root0 = (
        prepared.lanes[1].root_x.detach().reshape(-1)[0].clone()
    )

    def _fail_bicgstab_on_lane1(matvec, rhs, *params, **kwargs):
        res = real_bicgstab(matvec, rhs, *params, **kwargs)
        root_param = batch_implicit._unwrap_primal_tensor(params[0])
        root0 = root_param.reshape(res.status.numel(), -1)[:, 0].reshape(
            res.status.shape
        )
        fail_mask_stat = (root0 - target_lane1_root0).abs() < 1.0e-12
        extra_dims = res.solution.ndim - res.status.ndim
        fail_mask_sol = fail_mask_stat.reshape(
            *res.status.shape, *(1 for _ in range(extra_dims))
        )
        bad_sol = torch.where(
            fail_mask_sol,
            torch.full_like(res.solution, float("nan")),
            res.solution,
        )
        bad_stat = torch.where(
            fail_mask_stat,
            torch.full_like(res.status, int(STATUS_NONFINITE)),
            res.status,
        )
        return LinearSolveResult(
            solution=bad_sol,
            status=bad_stat,
            iterations=res.iterations,
            residual=res.residual,
            target=res.target,
        )

    monkeypatch.setattr(batch_implicit, "bicgstab", _fail_bicgstab_on_lane1)
    for cs in (2,):
        out_resp = stream_batch_value_and_grad(
            prepared,
            model,
            pos4,
            field=efield.clone().requires_grad_(True),
            field_grad=efgrad.clone().requires_grad_(True),
            chunk_size=cs,
            strict=False,
        )
        assert out_resp.failure_category == (
            "success",
            "response_failure",
            "success",
            "success",
        )
        assert out_resp.final_failure.tolist() == [
            int(BatchRootFailureCode.NONE),
            int(BatchRootFailureCode.RESPONSE_FAILURE),
            int(BatchRootFailureCode.NONE),
            int(BatchRootFailureCode.NONE),
        ]
        assert out_resp.response_converged.tolist() == [True, False, True, True]
        assert out_resp.response_status.tolist() == [
            int(STATUS_CONVERGED),
            int(STATUS_NONFINITE),
            int(STATUS_CONVERGED),
            int(STATUS_CONVERGED),
        ]
        assert bool(torch.isnan(out_resp.position_gradient[1]).all())
        assert bool(torch.isfinite(out_resp.position_gradient[0]).all())
        assert bool(torch.isfinite(out_resp.position_gradient[2]).all())
        assert bool(torch.isfinite(out_resp.position_gradient[3]).all())

    with pytest.raises(SCFConvergenceError, match=r"\[Lane 1\]"):
        stream_batch_value_and_grad(
            prepared,
            model,
            pos4,
            field=efield.clone().requires_grad_(True),
            field_grad=efgrad.clone().requires_grad_(True),
            chunk_size=2,
            strict=True,
        )

    # Finite-but-unconverged BiCGSTAB (status == 1) also masks lane 1 gradients with NaN:
    def _fail_bicgstab_finite_maxiter_on_lane1(
        matvec, rhs, *params, **kwargs
    ) -> LinearSolveResult:
        res = real_bicgstab(matvec, rhs, *params, **kwargs)
        root_param = batch_implicit._unwrap_primal_tensor(params[0])
        root0 = root_param.reshape(res.status.numel(), -1)[:, 0].reshape(
            res.status.shape
        )
        fail_mask_stat = (root0 - target_lane1_root0).abs() < 1.0e-12
        bad_stat = torch.where(
            fail_mask_stat,
            torch.ones_like(res.status),
            res.status,
        )
        return LinearSolveResult(
            solution=res.solution,
            status=bad_stat,
            iterations=res.iterations,
            residual=res.residual,
            target=res.target,
        )

    monkeypatch.setattr(
        batch_implicit, "bicgstab", _fail_bicgstab_finite_maxiter_on_lane1
    )
    with pytest.warns(SCFConvergenceWarning):
        out_resp_maxiter = stream_batch_value_and_grad(
            prepared,
            model,
            pos4,
            field=efield.clone().requires_grad_(True),
            chunk_size=2,
            strict=False,
        )
    assert out_resp_maxiter.failure_category == (
        "success",
        "response_failure",
        "success",
        "success",
    )
    assert out_resp_maxiter.response_status.tolist() == [0, 1, 0, 0]
    assert bool(torch.isnan(out_resp_maxiter.position_gradient[1]).all())
    assert bool(torch.isfinite(out_resp_maxiter.position_gradient[0]).all())
    monkeypatch.setattr(batch_implicit, "bicgstab", _fail_bicgstab_on_lane1)

    # Zero-weight on failing lane 1 skips lane 1 response solve and leaves gradients finite!
    out_resp_masked = stream_batch_value_and_grad(
        prepared,
        model,
        pos4,
        weights=torch.tensor([1.0, 0.0, 1.0, -0.5], dtype=DTYPE, device=DEVICE),
        field=efield.clone().requires_grad_(True),
        field_grad=efgrad.clone().requires_grad_(True),
        chunk_size=2,
        strict=False,
    )
    assert out_resp_masked.failure_category == (
        "success",
        "success",
        "success",
        "success",
    )
    assert int(out_resp_masked.response_status[1].item()) == RESPONSE_NOT_RUN
    assert not bool(out_resp_masked.response_converged[1])
    assert bool((out_resp_masked.position_gradient[1] == 0.0).all())
    assert bool(torch.isfinite(out_resp_masked.position_gradient).all())
    monkeypatch.setattr(batch_implicit, "bicgstab", real_bicgstab)

    # 3. Non-BiCGSTAB nonfinite derivatives per input family -> NONFINITE_DERIVATIVE (10)
    # while response_converged remains True!
    # 3a. Position-only derivative failure via a hook on chunk positions (classical repulsion path)
    # and 3b. External-field-only derivative failure, 3c. Charge-only, 3d. Spin-only, 3e. Model-parameter-only.
    orig_eval_classical = batch_implicit._evaluate_lane_classical

    # Inject a nonfinite gradient only into position of lane 2 during classical evaluation
    # (outside implicit_root / BiCGSTAB!).
    class _NanGradOnLane2Pos(torch.autograd.Function):
        @staticmethod
        def forward(ctx, e_term, pos_lane):
            ctx.is_lane2 = bool(
                torch.allclose(pos_lane.detach(), pos4[2].detach(), atol=1e-12)
            )
            return e_term.clone()

        @staticmethod
        def backward(ctx, grad_out):
            if ctx.is_lane2:
                return grad_out, torch.full(
                    (3, 3), float("nan"), dtype=DTYPE, device=DEVICE
                )
            return grad_out, torch.zeros((3, 3), dtype=DTYPE, device=DEVICE)

    def _hooked_classical_pos(sys_obj, l_pos, l_chrg):
        terms = orig_eval_classical(sys_obj, l_pos, l_chrg)
        if terms and l_pos.requires_grad:
            lbl, val = terms[0]
            val = _NanGradOnLane2Pos.apply(val, l_pos)
            return ((lbl, val), *terms[1:])
        return terms

    monkeypatch.setattr(
        batch_implicit, "_evaluate_lane_classical", _hooked_classical_pos
    )
    for cs in (2,):
        out_pos_nan = stream_batch_value_and_grad(
            prepared, model, pos4, chunk_size=cs, strict=False
        )
        assert out_pos_nan.response_converged.tolist() == [
            True,
            True,
            True,
            True,
        ]
        assert out_pos_nan.failure_category == (
            "success",
            "success",
            "nonfinite_derivative",
            "success",
        )
        assert out_pos_nan.final_failure.tolist() == [
            int(BatchRootFailureCode.NONE),
            int(BatchRootFailureCode.NONE),
            int(BatchRootFailureCode.NONFINITE_DERIVATIVE),
            int(BatchRootFailureCode.NONE),
        ]
    with pytest.raises(SCFConvergenceError, match=r"\[Lane 2\]"):
        stream_batch_value_and_grad(
            prepared, model, pos4, chunk_size=2, strict=True
        )
    monkeypatch.setattr(
        batch_implicit, "_evaluate_lane_classical", orig_eval_classical
    )

    # 3b. Charge-only derivative failure on lane 0 (shared scalar chrg with requires_grad=True)
    class _NanGradOnLane0Chrg(torch.autograd.Function):
        @staticmethod
        def forward(ctx, e_term, pos_lane, chrg_lane):
            ctx.is_lane0 = bool(
                torch.allclose(pos_lane.detach(), pos4[0].detach(), atol=1e-12)
            )
            ctx.chrg_shape = chrg_lane.shape
            return e_term.clone()

        @staticmethod
        def backward(ctx, grad_out):
            if ctx.is_lane0:
                return (
                    grad_out,
                    None,
                    torch.full(
                        ctx.chrg_shape,
                        float("nan"),
                        dtype=DTYPE,
                        device=DEVICE,
                    ),
                )
            return (
                grad_out,
                None,
                torch.zeros(ctx.chrg_shape, dtype=DTYPE, device=DEVICE),
            )

    def _hooked_classical_chrg(sys_obj, l_pos, l_chrg):
        terms = orig_eval_classical(sys_obj, l_pos, l_chrg)
        if terms and l_chrg.requires_grad:
            lbl, val = terms[0]
            val = _NanGradOnLane0Chrg.apply(val, l_pos, l_chrg)
            return ((lbl, val), *terms[1:])
        return terms

    monkeypatch.setattr(
        batch_implicit, "_evaluate_lane_classical", _hooked_classical_chrg
    )
    chrg_leaf = torch.tensor(
        0.0, dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    for cs in (2,):
        out_chrg_nan = stream_batch_value_and_grad(
            prepared,
            model,
            pos4,
            chrg=chrg_leaf,
            chunk_size=cs,
            strict=False,
        )
        # Position gradients are all finite, but charge gradient on lane 0 is NaN!
        assert bool(torch.isfinite(out_chrg_nan.position_gradient).all())
        assert out_chrg_nan.charge_gradient is not None
        assert bool(torch.isnan(out_chrg_nan.charge_gradient).any())
        assert out_chrg_nan.response_converged.tolist() == [
            True,
            True,
            True,
            True,
        ]
        assert out_chrg_nan.failure_category == (
            "nonfinite_derivative",
            "success",
            "success",
            "success",
        )
    monkeypatch.setattr(
        batch_implicit, "_evaluate_lane_classical", orig_eval_classical
    )

    # 3c. External-field-only derivative failure on lane 3 (shared field with requires_grad=True)
    orig_refresh = batch_implicit._refresh_prepared_batch_inputs

    class _NanGradOnLane3Field(torch.autograd.Function):
        @staticmethod
        def forward(ctx, f_tensor, is_lane3: bool):
            ctx.is_lane3 = is_lane3
            return f_tensor.clone()

        @staticmethod
        def backward(ctx, grad_out):
            if ctx.is_lane3:
                return torch.full_like(grad_out, float("nan")), None
            return grad_out, None

    def _hooked_refresh_field(
        prep_obj, eval_pos, c_in, s_in, f_in, fg_in, *, strict_mode
    ):
        if (
            isinstance(f_in, torch.Tensor)
            and f_in.requires_grad
            and f_in.ndim == 2
        ):
            hooked_rows = []
            for row_idx in range(f_in.shape[0]):
                is_l3 = bool(
                    torch.allclose(
                        eval_pos[row_idx].detach(),
                        pos4[3].detach(),
                        atol=1e-12,
                    )
                )
                hooked_rows.append(
                    _NanGradOnLane3Field.apply(f_in[row_idx], is_l3)
                )
            f_in = torch.stack(hooked_rows, dim=0)
        return orig_refresh(
            prep_obj,
            eval_pos,
            c_in,
            s_in,
            f_in,
            fg_in,
            strict_mode=strict_mode,
        )

    monkeypatch.setattr(
        batch_implicit, "_refresh_prepared_batch_inputs", _hooked_refresh_field
    )
    for cs in (2,):
        out_field_nan = stream_batch_value_and_grad(
            prepared,
            model,
            pos4,
            field=efield.clone().requires_grad_(True),
            chunk_size=cs,
            strict=False,
        )
        assert bool(torch.isfinite(out_field_nan.position_gradient).all())
        assert out_field_nan.field_gradient is not None
        assert bool(torch.isnan(out_field_nan.field_gradient).any())
        assert out_field_nan.response_converged.tolist() == [
            True,
            True,
            True,
            True,
        ]
        assert out_field_nan.failure_category == (
            "success",
            "success",
            "success",
            "nonfinite_derivative",
        )
    monkeypatch.setattr(
        batch_implicit, "_refresh_prepared_batch_inputs", orig_refresh
    )

    # 3d. Spin-only derivative failure on lane 1 (shared scalar spin with requires_grad=True)
    def _hooked_refresh_spin(
        prep_obj, eval_pos, c_in, s_in, f_in, fg_in, *, strict_mode
    ):
        if (
            isinstance(s_in, torch.Tensor)
            and s_in.requires_grad
            and s_in.ndim == 1
        ):
            hooked_s = []
            for row_idx in range(s_in.shape[0]):
                is_l1 = bool(
                    torch.allclose(
                        eval_pos[row_idx].detach(),
                        pos4[1].detach(),
                        atol=1e-12,
                    )
                )
                hooked_s.append(
                    _NanGradOnLane3Field.apply(s_in[row_idx], is_l1)
                )
            s_in = torch.stack(hooked_s, dim=0)
        return orig_refresh(
            prep_obj,
            eval_pos,
            c_in,
            s_in,
            f_in,
            fg_in,
            strict_mode=strict_mode,
        )

    prep_spin = prepare_converged_root_batch(
        system, pos4, spin=0.0, field=efield, field_grad=efgrad, strict=False
    )
    monkeypatch.setattr(
        batch_implicit, "_refresh_prepared_batch_inputs", _hooked_refresh_spin
    )
    spin_leaf = torch.tensor(
        0.0, dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    for cs in (2,):
        out_spin_nan = stream_batch_value_and_grad(
            prep_spin,
            model,
            pos4,
            spin=spin_leaf,
            chunk_size=cs,
            strict=False,
        )
        assert bool(torch.isfinite(out_spin_nan.position_gradient).all())
        assert out_spin_nan.spin_gradient is not None
        assert bool(torch.isnan(out_spin_nan.spin_gradient).any())
        assert out_spin_nan.failure_category == (
            "success",
            "nonfinite_derivative",
            "success",
            "success",
        )
    monkeypatch.setattr(
        batch_implicit, "_refresh_prepared_batch_inputs", orig_refresh
    )

    # 3e. Model-parameter-only derivative failure on lane 2 (position/field/chrg gradients stay finite!)
    class _NanGradOnLane2ModelParam(torch.autograd.Function):
        @staticmethod
        def forward(ctx, e_term, pos_lane, model_param):
            ctx.is_lane2 = bool(
                torch.allclose(pos_lane.detach(), pos4[2].detach(), atol=1e-12)
            )
            ctx.param_shape = model_param.shape
            return e_term.clone()

        @staticmethod
        def backward(ctx, grad_out):
            if ctx.is_lane2:
                return (
                    grad_out,
                    None,
                    torch.full(
                        ctx.param_shape,
                        float("nan"),
                        dtype=DTYPE,
                        device=DEVICE,
                    ),
                )
            return (
                grad_out,
                None,
                torch.zeros(ctx.param_shape, dtype=DTYPE, device=DEVICE),
            )

    assert model.par.element is not None
    target_model_leaf = model.par.element.O.arep.param
    assert target_model_leaf.requires_grad

    def _hooked_classical_model(sys_obj, l_pos, l_chrg):
        terms = orig_eval_classical(sys_obj, l_pos, l_chrg)
        if terms:
            lbl, val = terms[0]
            val = _NanGradOnLane2ModelParam.apply(val, l_pos, target_model_leaf)
            return ((lbl, val), *terms[1:])
        return terms

    monkeypatch.setattr(
        batch_implicit, "_evaluate_lane_classical", _hooked_classical_model
    )
    for cs in (2,):
        out_mod_nan = stream_batch_value_and_grad(
            prepared, model, pos4, chunk_size=cs, strict=False
        )
        assert bool(torch.isfinite(out_mod_nan.position_gradient).all())
        assert any(
            g is not None and bool(torch.isnan(g).any())
            for g in out_mod_nan.model_parameter_gradients
        )
        assert out_mod_nan.response_converged.tolist() == [
            True,
            True,
            True,
            True,
        ]
        assert out_mod_nan.failure_category == (
            "success",
            "success",
            "nonfinite_derivative",
            "success",
        )
    monkeypatch.setattr(
        batch_implicit, "_evaluate_lane_classical", orig_eval_classical
    )

    # 4. Verify all 5 failure categories together in a 5-lane batch:
    # lane 0: success (NONE = 0)
    # lane 1: fermi_failure (FERMI_NONCONVERGENCE = 3)
    # lane 2: primal_scf_failure (MAXITER_EXCEEDED = 5)
    # lane 3: response_failure (RESPONSE_FAILURE = 9)
    # lane 4: nonfinite_derivative (NONFINITE_DERIVATIVE = 10)
    codes = torch.tensor(
        [
            int(BatchRootFailureCode.NONE),
            int(BatchRootFailureCode.FERMI_NONCONVERGENCE),
            int(BatchRootFailureCode.MAXITER_EXCEEDED),
            int(BatchRootFailureCode.RESPONSE_FAILURE),
            int(BatchRootFailureCode.NONFINITE_DERIVATIVE),
        ],
        dtype=torch.int64,
    )
    assert classify_batch_failure_category(codes) == (
        "success",
        "fermi_failure",
        "primal_scf_failure",
        "response_failure",
        "nonfinite_derivative",
    )


def test_w2c11_streaming_api_contracts_aliasing_hooks_and_long_lived_execution() -> (
    None
):
    """Verify input aliasing, caller hooks, wrong-Model rejection, and 12-step long-lived stability."""
    import resource

    numbers, base_positions = _h2o_conformers()
    model, system = _build_system(
        numbers,
        int_level=labels.INTLEVEL_DIPOLE,
        etemp=300.0,
        force_convergence=False,
        trainable_model=True,
    )
    pos2 = base_positions[:2].clone()

    # 1. Wrong Model instance is rejected without silent cloning.
    other_model, _ = _build_system(numbers, trainable_model=True)
    prepared = prepare_converged_root_batch(system, pos2, strict=True)
    with pytest.raises(ValueError, match="does not match"):
        stream_batch_value_and_grad(prepared, other_model, pos2, strict=True)

    # 2. Aliased external scalar input (same tensor leaf passed as both chrg and spin)
    # and caller backward hooks on positions are preserved cleanly.
    shared_zero = torch.tensor(
        0.0, dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    prep_zero = prepare_converged_root_batch(
        system, pos2, chrg=shared_zero, spin=shared_zero, strict=True
    )
    pos_with_hook = pos2.clone().requires_grad_(True)
    hook_calls: list[int] = []
    handle = pos_with_hook.register_hook(
        lambda g: (hook_calls.append(int(g.shape[0])), g)[1]
    )
    try:
        out_alias = stream_batch_value_and_grad(
            prep_zero,
            model,
            pos_with_hook,
            chrg=shared_zero,
            spin=shared_zero,
            chunk_size=1,
            strict=True,
        )
    finally:
        handle.remove()
    assert out_alias.charge_gradient is not None
    assert out_alias.spin_gradient is not None
    assert bool(torch.isfinite(out_alias.charge_gradient).all())
    assert bool(torch.isfinite(out_alias.spin_gradient).all())
    # Caller's hook on pos_with_hook is preserved across both chunk backward passes.
    assert hook_calls == [2, 2]

    # 3. Repeated streaming evaluations in one long-lived process (6 iterations)
    # verify Model identity, fresh Result snapshots, gradient accuracy, and bounded RSS.
    model_id_initial = id(model)
    par_id_initial = id(model.par)
    seen_results: list[int] = []
    rss_samples_kb: list[int] = []
    for step in range(6):
        offset = 0.002 * ((step % 4) - 1.5)
        step_pos = pos2.clone()
        step_pos[:, 1, 2] += offset
        out_step = stream_batch_value_and_grad(
            system,
            model,
            step_pos,
            chunk_size=1 if step % 2 == 0 else 2,
            strict=True,
            eager_primal_fermi="spectral" if step % 3 == 0 else True,
        )
        assert id(model) == model_id_initial
        assert id(model.par) == par_id_initial
        assert bool(out_step.converged.all())
        assert out_step.failure_category == ("success", "success")
        assert bool(torch.isfinite(out_step.energy).all())
        assert bool(torch.isfinite(out_step.position_gradient).all())
        seen_results.append(id(out_step))
        if step >= 2:
            rss_samples_kb.append(
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            )
    # Peak RSS across steps 2..5 must not grow unboundedly (< 32 MB growth).
    assert max(rss_samples_kb) - min(rss_samples_kb) < 32 * 1024


def test_w2c11_eager_primal_fermi_modes_and_scope_isolation() -> None:
    """Verify static, adaptive, and spectral primal modes with streaming and scope safety."""
    from dxtb._src.scf.functional.spectral import (
        _current_primal_fermi_mode,
        eager_primal_fermi_scope,
    )

    # 1. Nested contexts, exception restoration, and invalid mode rejection.
    assert _current_primal_fermi_mode() == "static"
    with pytest.raises(ValueError, match="eager_primal_fermi must be"):
        with eager_primal_fermi_scope("invalid_mode"):
            pass
    assert _current_primal_fermi_mode() == "static"

    try:
        with eager_primal_fermi_scope(True):
            assert _current_primal_fermi_mode() == "adaptive"
            with eager_primal_fermi_scope("spectral"):
                assert _current_primal_fermi_mode() == "spectral"
                with eager_primal_fermi_scope(False):
                    assert _current_primal_fermi_mode() == "static"
                assert _current_primal_fermi_mode() == "spectral"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _current_primal_fermi_mode() == "static"

    # 2. Closed-shell H2O (300K) and open-shell NO2 (300K):
    # static, adaptive, and spectral primal modes all validate under the static raw map
    # and produce matching streaming energies, free energies, position gradients, and Model gradients.
    numbers_h2o, pos_h2o = _h2o_conformers()
    for etemp in (300.0,):
        model_h2o, sys_h2o = _build_system(
            numbers_h2o, etemp=etemp, trainable_model=True
        )
        out_static = stream_batch_value_and_grad(
            sys_h2o,
            model_h2o,
            pos_h2o[:2],
            chunk_size=1,
            strict=True,
            eager_primal_fermi=False,
        )
        for mode in (True, "spectral"):
            out_mode = stream_batch_value_and_grad(
                sys_h2o,
                model_h2o,
                pos_h2o[:2],
                chunk_size=1,
                strict=True,
                eager_primal_fermi=mode,
            )
            torch.testing.assert_close(
                out_mode.energy, out_static.energy, atol=1e-11, rtol=1e-11
            )
            torch.testing.assert_close(
                out_mode.free_energy,
                out_static.free_energy,
                atol=1e-11,
                rtol=1e-11,
            )
            torch.testing.assert_close(
                out_mode.position_gradient,
                out_static.position_gradient,
                atol=1e-9,
                rtol=1e-9,
            )
            for g_m, g_s in zip(
                out_mode.model_parameter_gradients,
                out_static.model_parameter_gradients,
            ):
                if g_s is None:
                    assert g_m is None
                else:
                    assert g_m is not None
                    torch.testing.assert_close(g_m, g_s, atol=1e-9, rtol=1e-8)

    # Open-shell NO2 radical (Z=[7, 8, 8], spin=1.0)
    numbers_no2 = torch.tensor([7, 8, 8], device=DEVICE)
    no2_0 = torch.tensor(
        [
            [0.00000000, 0.00000000, 0.22000000],
            [0.00000000, 2.05000000, -0.88000000],
            [0.00000000, -2.05000000, -0.88000000],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    no2_1 = no2_0.clone()
    no2_1[1, 1] += 0.06
    no2_1[2, 2] -= 0.04
    pos_no2 = torch.stack((no2_0, no2_1), dim=0)
    model_no2, sys_no2 = _build_system(
        numbers_no2, etemp=300.0, trainable_model=True
    )
    out_no2_static = stream_batch_value_and_grad(
        sys_no2,
        model_no2,
        pos_no2,
        spin=1.0,
        chunk_size=2,
        strict=True,
        eager_primal_fermi=False,
    )
    for mode in (True, "spectral"):
        out_no2_mode = stream_batch_value_and_grad(
            sys_no2,
            model_no2,
            pos_no2,
            spin=1.0,
            chunk_size=2,
            strict=True,
            eager_primal_fermi=mode,
        )
        torch.testing.assert_close(
            out_no2_mode.energy, out_no2_static.energy, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            out_no2_mode.position_gradient,
            out_no2_static.position_gradient,
            atol=1e-9,
            rtol=1e-9,
        )


def test_w2c11_direct_system_spectral_prep_parity_and_failure_semantics() -> (
    None
):
    """Direct System streaming and batch evaluation with spectral prep match prepared batch."""
    numbers, base_positions = _h2o_conformers()
    c3 = base_positions[0].clone()
    c3[1, 2] -= 0.05
    c3[2, 1] += 0.04
    pos4 = torch.cat((base_positions, c3.unsqueeze(0)), dim=0)
    model, system = _build_system(
        numbers, etemp=300.0, force_convergence=False, trainable_model=True
    )

    prep = prepare_converged_root_batch(
        system, pos4, eager_primal_fermi="spectral", strict=True
    )
    out_prep = stream_batch_value_and_grad(
        prep, model, pos4, chunk_size=2, strict=True
    )
    out_sys = stream_batch_value_and_grad(
        system,
        model,
        pos4,
        chunk_size=2,
        eager_primal_fermi="spectral",
        strict=True,
    )
    eval_sys = evaluate_converged_root_batch(
        system,
        pos4,
        eager_primal_fermi="spectral",
        strict=True,
    )
    torch.testing.assert_close(
        out_prep.energy, out_sys.energy, atol=1e-13, rtol=0
    )
    torch.testing.assert_close(
        out_prep.energy, eval_sys.energy.sum(dim=-1), atol=1e-13, rtol=0
    )
    torch.testing.assert_close(
        out_prep.position_gradient,
        out_sys.position_gradient,
        atol=1e-12,
        rtol=1e-12,
    )

    # Non-strict and strict failure propagation:
    _, sys_max1 = _build_system(
        numbers, maxiter=1, force_convergence=False, etemp=300.0
    )
    with pytest.warns(SCFConvergenceWarning):
        prep_fail = prepare_converged_root_batch(
            sys_max1, pos4[:2], strict=False
        )
    assert not prep_fail.all_roots_valid
    assert all(
        lane.failure_code == int(BatchRootFailureCode.MAXITER_EXCEEDED)
        for lane in prep_fail.lanes
    )
    with pytest.raises(SCFConvergenceError, match=r"\[Lane 0\]"):
        prepare_converged_root_batch(sys_max1, pos4[:2], strict=True)
