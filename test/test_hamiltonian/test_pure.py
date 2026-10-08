# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch.multipole import build_dipole
from dxtb._src.integral.driver.pytorch.overlap import build_overlap
from dxtb._src.integral.driver.pytorch.setup import setup_integrals
from dxtb._src.xtb.gfn0 import GFN0Hamiltonian
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian
from dxtb._src.xtb.gfn2 import GFN2Hamiltonian
from dxtb._src.xtb.h0 import build_hcore, setup_h0

from .samples import samples


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_pure_h0_is_call_history_independent(method: str) -> None:
    sample = samples["H2"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    params = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
    par = ParamModule(
        params[method].model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    setup = setup_h0(numbers, par, ihelp)
    overlap = torch.eye(
        ihelp.nao, dtype=positions.dtype, device=positions.device
    )
    charge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    charge_arg = charge if method == "gfn0" else None

    first, refocc = build_hcore(setup, positions, overlap, charge=charge_arg)
    moved = positions + torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=positions.dtype
    )
    second, _ = build_hcore(setup, moved, overlap, charge=charge_arg)
    repeated, _ = build_hcore(
        setup, positions.clone(), overlap, charge=charge_arg
    )

    torch.testing.assert_close(repeated, first)
    torch.testing.assert_close(refocc, setup.refocc)
    assert not torch.equal(second, first)

    leaf1 = positions.clone().requires_grad_()
    leaf2 = positions.clone().requires_grad_()
    value1, _ = build_hcore(setup, leaf1, overlap, charge=charge_arg)
    value2, _ = build_hcore(setup, leaf2, overlap, charge=charge_arg)
    grad1 = torch.autograd.grad(value1.sum(), leaf1, create_graph=True)[0]
    grad2 = torch.autograd.grad(value2.sum(), leaf2, create_graph=True)[0]
    torch.testing.assert_close(grad1, grad2)
    second_derivative = torch.autograd.grad(
        grad1.sum(), leaf1, create_graph=True
    )[0]
    assert torch.isfinite(second_derivative).all()
    third_derivative = torch.autograd.grad(second_derivative.sum(), leaf1)[0]
    assert torch.isfinite(third_derivative).all()

    jacobian = torch.func.jacfwd(
        lambda p: build_hcore(setup, p, overlap, charge=charge_arg)[0]
    )(positions)
    assert torch.isfinite(jacobian).all()

    _, tangent = torch.func.jvp(
        lambda p: build_hcore(setup, p, overlap, charge=charge_arg)[0],
        (positions,),
        (torch.ones_like(positions),),
    )
    assert torch.isfinite(tangent).all()

    batched_positions = torch.stack((positions, moved))
    batched = torch.func.vmap(
        lambda p: build_hcore(setup, p, overlap, charge=charge_arg)[0]
    )(batched_positions)
    torch.testing.assert_close(batched[0], first)
    torch.testing.assert_close(batched[1], second)


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_pure_h0_matches_legacy_and_explicit_refocc(method: str) -> None:
    sample = samples["H2"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    params = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
    hamiltonians = {
        "gfn0": GFN0Hamiltonian,
        "gfn1": GFN1Hamiltonian,
        "gfn2": GFN2Hamiltonian,
    }
    par = ParamModule(
        params[method].model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    setup = setup_h0(numbers, par, ihelp)
    overlap = torch.eye(ihelp.nao, dtype=positions.dtype, device=positions.device)
    # Include nonzero off-diagonal overlap elements so the explicit overlap is
    # observable in the returned H0 matrix.
    overlap = overlap + 0.05 * (torch.ones_like(overlap) - overlap)
    charge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    charge_arg = charge if method == "gfn0" else None

    hcore, refocc = build_hcore(setup, positions, overlap, charge=charge_arg)
    legacy = hamiltonians[method](
        numbers,
        par,
        ihelp,
        device=positions.device,
        dtype=positions.dtype,
        setup=setup,
    )
    legacy_hcore = legacy.build(positions, overlap, charge=charge_arg)

    torch.testing.assert_close(hcore, legacy_hcore)
    torch.testing.assert_close(refocc, legacy.refocc)
    assert refocc.dtype == positions.dtype
    assert refocc.device == positions.device

    before = refocc.clone()
    build_hcore(setup, positions + 0.01, overlap, charge=charge_arg)
    torch.testing.assert_close(refocc, before)
    torch.testing.assert_close(setup.refocc, before)

    # Equal-valued, independent leaves must produce matching matrix and
    # overlap gradients.
    overlap1 = overlap.clone().requires_grad_()
    overlap2 = overlap.clone().requires_grad_()
    h1, _ = build_hcore(setup, positions.clone(), overlap1, charge=charge_arg)
    h2, _ = build_hcore(setup, positions.clone(), overlap2, charge=charge_arg)
    grad1 = torch.autograd.grad(h1.sum(), overlap1)[0]
    grad2 = torch.autograd.grad(h2.sum(), overlap2)[0]
    torch.testing.assert_close(h1, h2)
    torch.testing.assert_close(grad1, grad2)


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_pure_h0_survives_prior_backward_and_integral_call_order(
    method: str,
) -> None:
    sample = samples["H2"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    params = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
    par = ParamModule(
        params[method].model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    setup = setup_h0(numbers, par, ihelp)
    integral_setup = setup_integrals(Basis(numbers, par, ihelp))
    charge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    charge_arg = charge if method == "gfn0" else None

    first_positions = positions.clone().requires_grad_()
    first_overlap = build_overlap(integral_setup, first_positions)
    first_hcore, first_refocc = build_hcore(
        setup, first_positions, first_overlap, charge=charge_arg
    )
    first_hcore.sum().backward()
    assert first_positions.grad is not None
    assert torch.isfinite(first_positions.grad).all()

    moved = positions + torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=positions.dtype
    )
    # First ordering: overlap -> H0 -> dipole -> H0.
    dipole = build_dipole(integral_setup, positions)
    moved_leaf = moved.clone().requires_grad_()
    second_overlap = build_overlap(integral_setup, moved_leaf)
    second_hcore, second_refocc = build_hcore(
        setup, moved_leaf, second_overlap, charge=charge_arg
    )
    second_gradient = torch.autograd.grad(second_hcore.sum(), moved_leaf)[0]
    assert torch.isfinite(second_gradient).all()
    assert torch.isfinite(dipole).all()

    # Second ordering: dipole -> overlap -> H0. Repeat the first geometry
    # after the preceding backward and moved-geometry evaluation.
    moved_dipole = build_dipole(integral_setup, moved)
    assert torch.isfinite(moved_dipole).all()
    repeated_leaf = positions.clone().requires_grad_()
    repeated_overlap = build_overlap(integral_setup, repeated_leaf)
    third_hcore, third_refocc = build_hcore(
        setup, repeated_leaf, repeated_overlap, charge=charge_arg
    )
    third_gradient = torch.autograd.grad(third_hcore.sum(), repeated_leaf)[0]

    torch.testing.assert_close(third_hcore, first_hcore.detach())
    torch.testing.assert_close(third_gradient, first_positions.grad)
    torch.testing.assert_close(third_refocc, first_refocc)
    torch.testing.assert_close(second_refocc, first_refocc)
    assert not torch.equal(second_hcore, third_hcore)


def test_pure_gfn0_requires_charge() -> None:
    sample = samples["H2"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    par = ParamModule(
        GFN0_XTB.model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    setup = setup_h0(numbers, par, IndexHelper.from_numbers(numbers, par))
    overlap = torch.eye(
        setup.ihelp.nao, dtype=positions.dtype, device=positions.device
    )

    with pytest.raises(TypeError, match="overlap"):
        build_hcore(setup, positions)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="charge is required"):
        build_hcore(setup, positions, overlap)


@pytest.mark.parametrize("method", ["gfn1", "gfn2"])
def test_pure_h0_multispecies_multishell(method: str) -> None:
    """Build H0 for a molecule with multiple elements and p shells."""
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    params = {"gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
    par = ParamModule(
        params[method].model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    setup = setup_h0(numbers, par, ihelp)
    overlap = torch.eye(ihelp.nao, dtype=positions.dtype, device=positions.device)

    hcore, refocc = build_hcore(setup, positions, overlap)

    assert hcore.shape == (ihelp.nao, ihelp.nao)
    assert refocc.shape == setup.refocc.shape
    assert torch.isfinite(hcore).all()
    assert torch.isfinite(refocc).all()
    assert torch.unique(numbers).numel() > 1
    assert torch.any(ihelp.unique_angular == 1)


def test_pure_gfn0_nonzero_charge_and_charge_gradient() -> None:
    """Charge-dependent GFN0 H0 remains differentiable."""
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    par = ParamModule(
        GFN0_XTB.model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    setup = setup_h0(numbers, par, ihelp)
    overlap = torch.eye(ihelp.nao, dtype=positions.dtype, device=positions.device)
    neutral = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    charged = torch.tensor(1.0, dtype=positions.dtype, device=positions.device)

    h_neutral, _ = build_hcore(setup, positions, overlap, charge=neutral)
    h_charged, _ = build_hcore(setup, positions, overlap, charge=charged)
    assert not torch.equal(h_neutral, h_charged)

    differentiable_charge = charged.clone().requires_grad_()
    h_differentiable, _ = build_hcore(
        setup, positions, overlap, charge=differentiable_charge
    )
    charge_gradient = torch.autograd.grad(
        h_differentiable.sum(), differentiable_charge
    )[0]
    assert torch.isfinite(charge_gradient).all()


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_h0_parameter_training_rebuilds_setup_per_loss(method: str) -> None:
    """Fresh composition setup preserves parameter gradients per loss call."""
    sample = samples["H2"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    params = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
    par = ParamModule(
        params[method].model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    refocc_parameter = dict(par.named_parameters())[
        "parameter_tree.element.H.refocc.param"
    ]
    refocc_parameter.requires_grad_(True)

    def loss_gradient() -> torch.Tensor:
        # Training rebuilds setup inside each differentiated loss evaluation.
        setup = setup_h0(numbers, par, IndexHelper.from_numbers(numbers, par))
        overlap = torch.eye(
            setup.ihelp.nao, dtype=positions.dtype, device=positions.device
        )
        charge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
        hcore, refocc = build_hcore(
            setup,
            positions,
            overlap,
            charge=charge if method == "gfn0" else None,
        )
        return torch.autograd.grad(
            hcore.sum() + refocc.sum(), refocc_parameter
        )[0]

    first = loss_gradient()
    second = loss_gradient()
    assert torch.isfinite(first).all()
    assert torch.any(first != 0)
    torch.testing.assert_close(second, first)


def test_gfn0_h0_setup_uses_zero_for_padded_element_parameters() -> None:
    numbers = pack((samples["H2"]["numbers"], samples["C"]["numbers"]))
    par = ParamModule(GFN0_XTB.model_copy(deep=True))
    ihelp = IndexHelper.from_numbers(numbers, par)
    setup = setup_h0(numbers, par, ihelp)
    padding = ihelp.unique_angular == -1

    assert padding.any()
    torch.testing.assert_close(
        setup.kcn[padding], torch.zeros_like(setup.kcn[padding])
    )
    torch.testing.assert_close(
        setup.selfenergy[padding], torch.zeros_like(setup.selfenergy[padding])
    )


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_h0_parameter_gradients(method: str) -> None:
    sample = samples["H2"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    params = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
    par = ParamModule(
        params[method].model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    for parameter in par.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)

    setup = setup_h0(numbers, par, IndexHelper.from_numbers(numbers, par))
    overlap = torch.eye(
        IndexHelper.from_numbers(numbers, par).nao,
        dtype=positions.dtype,
        device=positions.device,
    )
    charge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    hcore, refocc = build_hcore(
        setup, positions, overlap, charge=charge if method == "gfn0" else None
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in par.named_parameters()
        if parameter.requires_grad
    ]
    grads = torch.autograd.grad(
        hcore.sum() + refocc.sum(),
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
    )
    assert any(
        grad is not None and torch.isfinite(grad).all() for grad in grads
    )
    gradients = {
        name: grad for (name, _), grad in zip(named_parameters, grads)
    }
    refocc_gradient = gradients[
        "parameter_tree.element.H.refocc.param"
    ]
    assert refocc_gradient is not None
    assert torch.isfinite(refocc_gradient).all()
    assert torch.any(refocc_gradient != 0)
    if method == "gfn0":
        for parameter in ("chi", "eta", "kcn", "rad"):
            grad = gradients[f"parameter_tree.element.H.eeq_{parameter}.param"]
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert torch.any(grad != 0)
