# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
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

    with pytest.raises(ValueError, match="charge is required"):
        build_hcore(setup, positions)


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
    if method == "gfn0":
        gradients = {
            name: grad for (name, _), grad in zip(named_parameters, grads)
        }
        for parameter in ("chi", "eta", "kcn", "rad"):
            grad = gradients[f"parameter_tree.element.H.eeq_{parameter}.param"]
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert torch.any(grad != 0)
