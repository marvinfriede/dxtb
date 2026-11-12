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
"""Production ownership and numerical tests for PyTorch integral Nodes."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.func import jacfwd, jacrev, jvp, vmap
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, IndexHelper, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch import setup as pytorch_setup_module
from dxtb._src.integral.driver.pytorch.impls.kernels.os import compute_1d_os
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    PairClass,
    PairPlan,
    assemble_matrix,
    select_pairs,
)
from dxtb._src.integral.driver.pytorch.overlap import build_overlap
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetup,
    setup_integrals,
)
from dxtb._src.param import ParamModule
from dxtb._src.typing import Any, Tensor

from .samples import samples


def _calculator(method: Any, sample: str = "H2O", level: int | None = None):
    geometry = samples[sample]
    numbers = geometry["numbers"].to(device="cpu")
    positions = geometry["positions"].to(device="cpu", dtype=torch.float64)
    par = ParamModule(method, dtype=positions.dtype)
    par.set_differentiable("element.H.slater")
    opts = {"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH}
    if level is not None:
        opts["int_level"] = level
    calc = Calculator(numbers, par, dtype=positions.dtype, opts=opts)
    return calc, par, numbers, positions


def _assert_matrix_parity(actual: Tensor, expected: Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


def test_system_owns_registered_pair_nodes_and_conversion_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core structural data is isolated and registered as PyTree leaves."""
    calc, par, numbers, _ = _calculator(GFN1_XTB)
    original_prepare = pytorch_setup_module.prepare
    prepare_calls = []

    def count_prepare(ihelp, device, *args, **kwargs):
        prepare_calls.append(ihelp)
        return original_prepare(ihelp, device, *args, **kwargs)

    monkeypatch.setattr(pytorch_setup_module, "prepare", count_prepare)
    system = calc.model.setup(numbers, dd=calc.dd)
    assert len(prepare_calls) == 1
    assert system.integral_setup is not None
    setup = system.integral_setup.pytorch
    assert type(setup) is PytorchIntegralSetup
    assert isinstance(setup, Node)
    assert isinstance(setup.index, IndexHelper)
    assert setup.index is system.ihelp
    assert type(setup.pair_plan) is PairPlan
    assert type(setup.pair_plan.classes) is tuple
    assert all(type(item) is PairClass for item in setup.pair_plan.classes)
    assert setup.pair_plan.atom is setup.index.shells_to_atom
    assert not hasattr(setup, "ihelp")
    assert not hasattr(setup, "basis")
    assert not hasattr(setup, "param")
    assert setup.algorithm == setup.algorithm.casefold()

    leaves, spec = tree_flatten(setup)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is PytorchIntegralSetup
    assert type(rebuilt.pair_plan) is PairPlan
    assert all(
        any(leaf is tensor for leaf in leaves) for tensor in setup.alphas
    )
    assert all(
        any(leaf is tensor for leaf in leaves) for tensor in setup.coeffs
    )
    assert all(
        any(leaf is tensor for leaf in leaves)
        for tensor in (
            setup.pair_plan.atom,
            setup.index.orbitals_to_shell,
            *(item.ib for item in setup.pair_plan.classes),
            *(item.jk for item in setup.pair_plan.classes),
            *(item.idx for item in setup.pair_plan.classes),
            *(item.weight for item in setup.pair_plan.classes),
        )
    )
    assert setup.algorithm not in leaves

    other_ihelp = IndexHelper.from_numbers(numbers, par)
    other = setup_integrals(Basis(numbers, par, other_ihelp))
    system_originals = tuple(
        tensor.clone()
        for tensor in (
            setup.pair_plan.atom,
            setup.index.shells_to_atom,
            setup.index.orbitals_to_shell,
            setup.pair_plan.classes[0].ib,
            setup.pair_plan.classes[0].jk,
            setup.pair_plan.classes[0].idx,
            setup.pair_plan.classes[0].weight,
        )
    )
    other_cls = other.pair_plan.classes[0]
    with torch.no_grad():
        other_cls.ib[0] = other_cls.ib[0] + 1
        other_cls.jk[0] = other_cls.jk[0] + 1
        other_cls.idx[0] = other_cls.idx[0] + 1
        other_cls.weight[0] = other_cls.weight[0] + 7.0
        other.index.shells_to_atom[0] = other.index.shells_to_atom[0] + 1
    assert torch.equal(setup.pair_plan.atom, system_originals[0])
    assert torch.equal(setup.index.shells_to_atom, system_originals[1])
    assert torch.equal(setup.index.orbitals_to_shell, system_originals[2])
    assert torch.equal(setup.pair_plan.classes[0].ib, system_originals[3])
    assert torch.equal(setup.pair_plan.classes[0].jk, system_originals[4])
    assert torch.equal(setup.pair_plan.classes[0].idx, system_originals[5])
    assert torch.equal(setup.pair_plan.classes[0].weight, system_originals[6])

    def fail_prepare(*args, **kwargs):
        raise AssertionError("Node conversion must not rebuild the pair plan")

    monkeypatch.setattr(pytorch_setup_module, "prepare", fail_prepare)

    converted_dtype = setup.to(dtype=torch.float32)
    assert converted_dtype is not setup
    assert all(value.dtype == torch.float32 for value in converted_dtype.alphas)
    assert all(value.dtype == torch.float32 for value in converted_dtype.coeffs)
    assert converted_dtype.pair_plan.atom.dtype == torch.int64
    assert converted_dtype.index.orbitals_to_shell.dtype == torch.int64
    assert converted_dtype.pair_plan.classes[0].ib.dtype == torch.int64
    assert converted_dtype.pair_plan.classes[0].idx.dtype == torch.int64
    assert converted_dtype.pair_plan.classes[0].weight.dtype == torch.float32
    assert (
        converted_dtype.pair_plan.unique_angular
        == setup.pair_plan.unique_angular
    )
    assert converted_dtype.pair_plan.nao == setup.pair_plan.nao
    assert setup.alphas[0].dtype == torch.float64
    converted_back = converted_dtype.to(dtype=torch.float64)
    assert all(value.dtype == torch.float64 for value in converted_back.alphas)
    assert all(value.dtype == torch.float64 for value in converted_back.coeffs)
    assert converted_back.pair_plan.classes[0].weight.dtype == torch.float64
    assert converted_back.pair_plan.classes[0].idx.dtype == torch.int64
    assert converted_back.pair_plan.atom.dtype == torch.int64

    meta = setup.to(device="meta")
    assert all(value.device.type == "meta" for value in meta.alphas)
    assert all(value.device.type == "meta" for value in meta.coeffs)
    assert meta.pair_plan.atom.device.type == "meta"
    assert meta.pair_plan.classes[0].idx.device.type == "meta"
    assert meta.pair_plan.classes[0].ib.dtype == torch.int64

    stacked = stack((setup, setup))
    assert type(stacked) is PytorchIntegralSetup
    assert stacked.pair_plan.atom.shape[0] == 2
    assert stacked.pair_plan.classes[0].weight.shape[0] == 2


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_position_and_parameter_derivatives_and_transforms(
    method: Any,
) -> None:
    """Reverse, mixed, forward, and vmap transforms remain consistent."""
    calc, par, numbers, positions = _calculator(method)
    assert calc.system.integral_setup is not None
    core = calc.system.integral_setup.pytorch
    assert core is not None
    parameter = par.get("element.H.slater")

    def objective(setup: PytorchIntegralSetup, geometry: Tensor) -> Tensor:
        return build_overlap(setup, geometry).square().sum()

    def position_gradient(
        setup: PytorchIntegralSetup, geometry: Tensor
    ) -> Tensor:
        return torch.autograd.grad(
            objective(setup, geometry), geometry, create_graph=True
        )[0]

    core_positions = positions.clone().requires_grad_(True)
    core_value = objective(core, core_positions)
    (core_parameter_gradient,) = torch.autograd.grad(
        core_value, parameter, retain_graph=True
    )
    assert torch.isfinite(core_parameter_gradient).all()
    assert core_parameter_gradient.abs().sum() > 0

    core_position_gradient = position_gradient(core, core_positions)
    (core_mixed,) = torch.autograd.grad(
        core_position_gradient.sum(),
        parameter,
    )
    assert torch.isfinite(core_mixed).all()

    # Cross-check parameter and mixed position-parameter derivatives via FD.
    eps = 1.0e-6
    base_param = parameter.detach().clone()
    fd_param_grads = []
    fd_mixed_grads = []
    for idx in range(base_param.numel()):
        fd_values = []
        fd_pos_grad_sums = []
        for sign in (1.0, -1.0):
            par_shift = ParamModule(method, dtype=positions.dtype)
            shift_param = par_shift.get("element.H.slater")
            perturbed = base_param.clone()
            perturbed.view(-1)[idx] += sign * eps
            with torch.no_grad():
                shift_param.copy_(perturbed)
            ihelp_shift = IndexHelper.from_numbers(numbers, par_shift)
            setup_shift = setup_integrals(
                Basis(numbers, par_shift, ihelp_shift)
            )
            pos_leaf = positions.clone().requires_grad_(True)
            val_shift = objective(setup_shift, pos_leaf)
            grad_shift = torch.autograd.grad(val_shift, pos_leaf)[0]
            fd_values.append(val_shift.detach())
            fd_pos_grad_sums.append(grad_shift.detach().sum())
        fd_param_grads.append((fd_values[0] - fd_values[1]) / (2.0 * eps))
        fd_mixed_grads.append(
            (fd_pos_grad_sums[0] - fd_pos_grad_sums[1]) / (2.0 * eps)
        )
    fd_param_grad = torch.stack(fd_param_grads).reshape_as(
        core_parameter_gradient
    )
    fd_mixed_grad = torch.stack(fd_mixed_grads).reshape_as(core_mixed)
    torch.testing.assert_close(
        core_parameter_gradient, fd_param_grad, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        core_mixed, fd_mixed_grad, rtol=1e-6, atol=1e-7
    )

    pos_higher = positions.clone().requires_grad_(True)
    first = position_gradient(core, pos_higher)
    direction_ones = torch.ones_like(first)
    second = torch.autograd.grad(
        (first * direction_ones).sum(), pos_higher, create_graph=True
    )[0]
    third = torch.autograd.grad(
        (second * direction_ones).sum(), pos_higher
    )[0]
    assert torch.isfinite(second).all()
    assert torch.isfinite(third).all()

    def eval_overlap(geometry: Tensor) -> Tensor:
        return build_overlap(core, geometry)

    def flat_overlap(geometry: Tensor) -> Tensor:
        return build_overlap(core, geometry).reshape(-1)

    direction = torch.linspace(
        -0.25, 0.35, positions.numel(), dtype=positions.dtype
    ).reshape_as(positions)
    _, core_tangent = jvp(eval_overlap, (positions,), (direction,))
    fwd_jac = jacfwd(flat_overlap)(positions)
    rev_jac = jacrev(flat_overlap)(positions)
    _assert_matrix_parity(fwd_jac, rev_jac)
    expected_tangent = (
        fwd_jac.reshape(-1, positions.numel()) @ direction.reshape(-1)
    ).reshape_as(core_tangent)
    _assert_matrix_parity(core_tangent, expected_tangent)

    geometries = torch.stack((positions, positions + 0.03))
    mapped = vmap(eval_overlap)(geometries)
    explicit = torch.stack([eval_overlap(x) for x in geometries])
    _assert_matrix_parity(mapped, explicit)

    def eval_for_setup(setup_item: PytorchIntegralSetup) -> Tensor:
        return build_overlap(setup_item, positions)

    stacked = stack((core, core))
    setup_mapped = vmap(eval_for_setup)(stacked)
    setup_explicit = torch.stack(
        [build_overlap(core, positions), build_overlap(core, positions)]
    )
    _assert_matrix_parity(setup_mapped, setup_explicit)


def test_node_screening_chunking_checkpoint_and_missing_plan() -> None:
    """Node plans preserve selection and all established assembly options."""
    calc, _, _, positions = _calculator(GFN1_XTB)
    assert calc.system.integral_setup is not None
    core = calc.system.integral_setup.pytorch
    assert core is not None
    alphas, coeffs = list(core.alphas), list(core.coeffs)
    reference = assemble_matrix(
        compute_1d_os,
        None,
        alphas,
        coeffs,
        positions,
        plan=core.pair_plan,
    )
    _assert_matrix_parity(reference.squeeze(0), build_overlap(core, positions))

    core_pairs = select_pairs(
        compute_1d_os,
        None,
        alphas,
        coeffs,
        positions,
        1e-8,
        plan=core.pair_plan,
    )
    unplanned_pairs = select_pairs(
        compute_1d_os,
        core.index,
        alphas,
        coeffs,
        positions,
        1e-8,
    )
    assert len(core_pairs) == len(unplanned_pairs)
    for core_pair, unplanned_pair in zip(core_pairs, unplanned_pairs):
        assert torch.equal(core_pair, unplanned_pair)

    unplanned = assemble_matrix(
        compute_1d_os,
        core.index,
        alphas,
        coeffs,
        positions,
    )
    _assert_matrix_parity(unplanned, reference)

    for kwargs in (
        {"chunk_size": 1},
        {"chunk_size": 1, "checkpoint": True},
    ):
        actual = assemble_matrix(
            compute_1d_os,
            None,
            alphas,
            coeffs,
            positions,
            plan=core.pair_plan,
            **kwargs,
        )
        _assert_matrix_parity(actual, reference)

    from_pairs = assemble_matrix(
        compute_1d_os,
        None,
        alphas,
        coeffs,
        positions,
        plan=core.pair_plan,
        pairs=core_pairs,
    )
    screened = assemble_matrix(
        compute_1d_os,
        None,
        alphas,
        coeffs,
        positions,
        plan=core.pair_plan,
        screening_threshold=1e-8,
    )
    _assert_matrix_parity(from_pairs, screened)
    torch.testing.assert_close(screened, reference, rtol=1e-7, atol=1e-7)

    with pytest.raises(
        ValueError, match="prepared pair plan or an IndexHelper"
    ):
        assemble_matrix(
            compute_1d_os,
            None,
            alphas,
            coeffs,
            positions,
        )

    batched_numbers = torch.tensor([[1, 1], [1, 1]])
    batched_ihelp = IndexHelper.from_numbers(batched_numbers, GFN1_XTB)
    with pytest.raises(ValueError, match="requires one single System"):
        pytorch_setup_module.prepare(batched_ihelp, positions.device)
    with pytest.raises(ValueError, match="requires one single System"):
        setup_integrals(Basis(batched_numbers, GFN1_XTB, batched_ihelp))


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is not supported for this Python/torch combination",
)
def test_node_plan_compiles_fullgraph_without_host_rebuild() -> None:
    """Fullgraph compilation consumes the captured plan directly."""
    calc, _, _, positions = _calculator(GFN1_XTB, "H2")
    assert calc.system.integral_setup is not None
    setup = calc.system.integral_setup.pytorch
    assert setup is not None

    def eval_overlap(geometry: Tensor) -> Tensor:
        return build_overlap(setup, geometry)

    torch._dynamo.reset()
    actual = torch.compile(
        eval_overlap, fullgraph=True, backend="aot_eager"
    )(positions)
    torch.testing.assert_close(
        actual, eval_overlap(positions), rtol=1e-12, atol=1e-13
    )
