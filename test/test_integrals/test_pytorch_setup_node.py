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

from dataclasses import replace

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.func import jacfwd, jacrev, jvp, vmap
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch import setup as pytorch_setup_module
from dxtb._src.integral.driver.pytorch.impls.kernels.os import compute_1d_os
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    PairClassNode,
    PairPlanNode,
    assemble_matrix,
    select_pairs,
)
from dxtb._src.integral.driver.pytorch.multipole import (
    build_dipole,
    build_quadrupole,
)
from dxtb._src.integral.driver.pytorch.overlap import build_overlap
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetup,
    PytorchIntegralSetupNode,
    core_pytorch_integral_setup,
    setup_integrals,
)
from dxtb._src.integral.evaluation import IntegralSetup, build_integral_matrices
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional.core import CoreIndexData
from dxtb._src.typing import Tensor

from .samples import samples


def _calculator(method, sample: str = "H2O", level: int | None = None):
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


def _legacy_setup(calc: Calculator, par: ParamModule, numbers: Tensor):
    return setup_integrals(Basis(numbers, par, calc.system.ihelp))


def _assert_matrix_parity(actual: Tensor, expected: Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


def test_system_owns_registered_pair_nodes_and_conversion_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core structural data is copied; basis parameter tensors keep identity."""
    calc, par, numbers, _ = _calculator(GFN1_XTB)
    original_prepare = pytorch_setup_module.prepare
    prepare_calls = []

    def count_prepare(ihelp, device):
        prepare_calls.append(ihelp)
        return original_prepare(ihelp, device)

    monkeypatch.setattr(pytorch_setup_module, "prepare", count_prepare)
    system = calc.model.setup(numbers, dd=calc.dd)
    assert len(prepare_calls) == 1
    assert system.integral_setup is not None
    setup = system.integral_setup.pytorch
    legacy = setup_integrals(Basis(numbers, par, system.ihelp))
    assert type(setup) is PytorchIntegralSetupNode
    assert isinstance(setup, Node)
    assert type(setup.index) is CoreIndexData
    assert type(setup.pair_plan) is PairPlanNode
    assert type(setup.pair_plan.classes) is tuple
    assert not hasattr(setup, "ihelp")
    assert not hasattr(setup, "basis")
    assert not hasattr(setup, "param")
    assert setup.algorithm == setup.algorithm.casefold()

    assert type(legacy) is PytorchIntegralSetup
    converted = core_pytorch_integral_setup(legacy)
    assert all(a is b for a, b in zip(converted.alphas, legacy.alphas))
    assert all(a is b for a, b in zip(converted.coeffs, legacy.coeffs))
    assert all(type(item) is PairClassNode for item in setup.pair_plan.classes)
    assert setup.pair_plan.atom is setup.index.shells_to_atom

    leaves, spec = tree_flatten(setup)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is PytorchIntegralSetupNode
    assert type(rebuilt.pair_plan) is PairPlanNode
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

    originals = tuple(
        tensor.clone()
        for tensor in (
            converted.pair_plan.atom,
            converted.index.shells_to_atom,
            converted.index.orbitals_to_shell,
            converted.pair_plan.classes[0].ib,
            converted.pair_plan.classes[0].jk,
            converted.pair_plan.classes[0].idx,
            converted.pair_plan.classes[0].weight,
        )
    )
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
    legacy_cls = legacy.pair_plan.classes[0]
    with torch.no_grad():
        legacy_cls.ib[0] = legacy_cls.ib[0] + 1
        legacy_cls.jk[0] = legacy_cls.jk[0] + 1
        legacy_cls.idx[0] = legacy_cls.idx[0] + 1
        legacy_cls.weight[0] = legacy_cls.weight[0] + 7.0
        legacy.ihelp.shells_to_atom[0] = legacy.ihelp.shells_to_atom[0] + 1
    legacy_cls.idx = torch.zeros_like(legacy_cls.idx)
    assert torch.equal(converted.pair_plan.atom, originals[0])
    assert torch.equal(converted.index.shells_to_atom, originals[1])
    assert torch.equal(converted.index.orbitals_to_shell, originals[2])
    assert torch.equal(converted.pair_plan.classes[0].ib, originals[3])
    assert torch.equal(converted.pair_plan.classes[0].jk, originals[4])
    assert torch.equal(converted.pair_plan.classes[0].idx, originals[5])
    assert torch.equal(converted.pair_plan.classes[0].weight, originals[6])
    assert torch.equal(setup.pair_plan.atom, system_originals[0])
    assert torch.equal(setup.index.shells_to_atom, system_originals[1])
    assert torch.equal(setup.index.orbitals_to_shell, system_originals[2])
    assert torch.equal(setup.pair_plan.classes[0].ib, system_originals[3])
    assert torch.equal(setup.pair_plan.classes[0].jk, system_originals[4])
    assert torch.equal(setup.pair_plan.classes[0].idx, system_originals[5])
    assert torch.equal(setup.pair_plan.classes[0].weight, system_originals[6])

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
    assert type(stacked) is PytorchIntegralSetupNode
    assert stacked.pair_plan.atom.shape[0] == 2
    assert stacked.pair_plan.classes[0].weight.shape[0] == 2

    def fail_prepare(*args, **kwargs):
        raise AssertionError("Node conversion must not rebuild the pair plan")

    monkeypatch.setattr(pytorch_setup_module, "prepare", fail_prepare)
    assert setup.to(dtype=torch.float32).pair_plan.nao == setup.pair_plan.nao


def test_node_and_legacy_raw_and_normalized_integral_parity() -> None:
    """GFN1/2 basis fixtures retain all raw and shifted matrix values."""
    for method in (GFN1_XTB, GFN2_XTB):
        for sample in ("H2", "LiH", "H2O"):
            calc, par, numbers, positions = _calculator(
                method, sample, labels.INTLEVEL_QUADRUPOLE
            )
            system = calc.system
            core = system.integral_setup.pytorch
            legacy = _legacy_setup(calc, par, numbers)
            assert type(core) is PytorchIntegralSetupNode

            _assert_matrix_parity(
                build_overlap(core, positions), build_overlap(legacy, positions)
            )
            _assert_matrix_parity(
                build_dipole(core, positions), build_dipole(legacy, positions)
            )
            _assert_matrix_parity(
                build_quadrupole(core, positions),
                build_quadrupole(legacy, positions),
            )

            legacy_outer = replace(system.integral_setup, pytorch=legacy)
            legacy_matrices, legacy_refocc, legacy_norm = (
                build_integral_matrices(
                    legacy_outer, system.h0_setup, positions, charge=0.0
                )
            )
            core_matrices, core_refocc, core_norm = build_integral_matrices(
                system.integral_setup, system.h0_setup, positions, charge=0.0
            )
            for field in ("overlap", "dipole", "quadrupole", "hcore"):
                _assert_matrix_parity(
                    getattr(core_matrices, field),
                    getattr(legacy_matrices, field),
                )
            _assert_matrix_parity(core_refocc, legacy_refocc)
            _assert_matrix_parity(core_norm, legacy_norm)


def test_legacy_pytorch_setup_conversion_remains_compatible() -> None:
    """Legacy setup conversion still rebuilds its device-local plan."""
    calc, par, numbers, positions = _calculator(GFN2_XTB)
    legacy = _legacy_setup(calc, par, numbers)
    converted = legacy.to(device="cpu", dtype=torch.float32)
    assert type(converted) is PytorchIntegralSetup
    assert converted.pair_plan is not legacy.pair_plan
    assert converted.pair_plan.atom.device == torch.device(
        converted.ihelp.device
    )
    assert torch.equal(converted.pair_plan.atom, converted.ihelp.shells_to_atom)
    assert all(value.dtype == torch.float32 for value in converted.alphas)
    assert all(value.dtype == torch.float32 for value in converted.coeffs)
    overlap = build_overlap(converted, positions.to(dtype=torch.float32))
    assert torch.isfinite(overlap).all()


@pytest.mark.parametrize(
    "level",
    [
        labels.INTLEVEL_OVERLAP,
        labels.INTLEVEL_DIPOLE,
        labels.INTLEVEL_QUADRUPOLE,
    ],
)
def test_singlepoint_total_energy_matches_legacy_integral_setup(
    level: int,
) -> None:
    """Total energy and reference occupations are unchanged at every level."""
    calc, par, numbers, positions = _calculator(GFN1_XTB, "H2", level)
    system = calc.system
    legacy = _legacy_setup(calc, par, numbers)
    legacy_system = replace(
        system,
        integral_setup=replace(system.integral_setup, pytorch=legacy),
    )
    core_positions = positions.clone().requires_grad_(True)
    legacy_positions = positions.clone().requires_grad_(True)
    core_result = system.singlepoint(core_positions)
    legacy_result = legacy_system.singlepoint(legacy_positions)
    _assert_matrix_parity(core_result.energy, legacy_result.energy)
    (core_gradient,) = torch.autograd.grad(
        core_result.energy.sum(), core_positions
    )
    (legacy_gradient,) = torch.autograd.grad(
        legacy_result.energy.sum(), legacy_positions
    )
    _assert_matrix_parity(core_gradient, legacy_gradient)
    for core_term, legacy_term in zip(
        core_result.classical, legacy_result.classical
    ):
        assert core_term[0] == legacy_term[0]
        _assert_matrix_parity(core_term[1], legacy_term[1])


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_position_and_parameter_derivative_parity_including_mixed(
    method: str,
) -> None:
    """Reverse, mixed, and forward derivatives agree with legacy setup."""
    calc, par, numbers, positions = _calculator(method)
    core = calc.system.integral_setup.pytorch
    legacy = _legacy_setup(calc, par, numbers)
    parameter = par.get("element.H.slater")

    def objective(setup, geometry):
        return build_overlap(setup, geometry).square().sum()

    def position_gradient(setup, geometry):
        return torch.autograd.grad(
            objective(setup, geometry), geometry, create_graph=True
        )[0]

    core_positions = positions.clone().requires_grad_(True)
    legacy_positions = positions.clone().requires_grad_(True)
    core_value = objective(core, core_positions)
    legacy_value = objective(legacy, legacy_positions)
    (core_parameter_gradient,) = torch.autograd.grad(
        core_value, parameter, retain_graph=True
    )
    (legacy_parameter_gradient,) = torch.autograd.grad(
        legacy_value, parameter, retain_graph=True
    )
    _assert_matrix_parity(core_parameter_gradient, legacy_parameter_gradient)
    core_position_gradient = position_gradient(core, core_positions)
    legacy_position_gradient = position_gradient(legacy, legacy_positions)
    _assert_matrix_parity(core_position_gradient, legacy_position_gradient)
    (core_mixed,) = torch.autograd.grad(
        (
            core_position_gradient * torch.ones_like(core_position_gradient)
        ).sum(),
        parameter,
    )
    (legacy_mixed,) = torch.autograd.grad(
        (
            legacy_position_gradient * torch.ones_like(legacy_position_gradient)
        ).sum(),
        parameter,
    )
    _assert_matrix_parity(core_mixed, legacy_mixed)

    def higher_position_derivatives(setup, geometry):
        first = torch.autograd.grad(
            objective(setup, geometry), geometry, create_graph=True
        )[0]
        direction = torch.ones_like(first)
        second = torch.autograd.grad(
            (first * direction).sum(), geometry, create_graph=True
        )[0]
        third = torch.autograd.grad((second * direction).sum(), geometry)[0]
        return second, third

    core_second, core_third = higher_position_derivatives(
        core, positions.clone().requires_grad_(True)
    )
    legacy_second, legacy_third = higher_position_derivatives(
        legacy, positions.clone().requires_grad_(True)
    )
    _assert_matrix_parity(core_second, legacy_second)
    _assert_matrix_parity(core_third, legacy_third)

    direction = torch.randn_like(positions)
    _, core_tangent = jvp(
        lambda x: build_overlap(core, x), (positions,), (direction,)
    )
    _, legacy_tangent = jvp(
        lambda x: build_overlap(legacy, x), (positions,), (direction,)
    )
    _assert_matrix_parity(core_tangent, legacy_tangent)
    core_function = lambda x: build_overlap(core, x).reshape(-1)
    legacy_function = lambda x: build_overlap(legacy, x).reshape(-1)
    _assert_matrix_parity(
        jacfwd(core_function)(positions), jacfwd(legacy_function)(positions)
    )
    _assert_matrix_parity(
        jacrev(core_function)(positions), jacrev(legacy_function)(positions)
    )

    geometries = torch.stack((positions, positions + 0.03))
    mapped = vmap(lambda x: build_overlap(core, x))(geometries)
    explicit = torch.stack([build_overlap(core, x) for x in geometries])
    _assert_matrix_parity(mapped, explicit)

    stacked = stack((core, core))
    setup_mapped = vmap(lambda s: build_overlap(s, positions))(stacked)
    setup_explicit = torch.stack(
        [build_overlap(core, positions), build_overlap(core, positions)]
    )
    _assert_matrix_parity(setup_mapped, setup_explicit)


def test_node_screening_chunking_checkpoint_and_missing_plan() -> None:
    """Node plans preserve selection and all established assembly options."""
    calc, par, numbers, positions = _calculator(GFN1_XTB)
    core = calc.system.integral_setup.pytorch
    legacy = _legacy_setup(calc, par, numbers)
    alphas, coeffs = list(core.alphas), list(core.coeffs)
    core_pairs = select_pairs(
        compute_1d_os,
        None,
        alphas,
        coeffs,
        positions,
        1e-8,
        plan=core.pair_plan,
    )
    legacy_pairs = select_pairs(
        compute_1d_os,
        legacy.ihelp,
        list(legacy.alphas),
        list(legacy.coeffs),
        positions,
        1e-8,
        plan=legacy.pair_plan,
    )
    assert len(core_pairs) == len(legacy_pairs)
    for core_pair, legacy_pair in zip(core_pairs, legacy_pairs):
        assert torch.equal(core_pair, legacy_pair)

    for kwargs in (
        {"chunk_size": 1},
        {"chunk_size": 1, "checkpoint": True},
        {"pairs": core_pairs},
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
        expected = assemble_matrix(
            compute_1d_os,
            legacy.ihelp,
            list(legacy.alphas),
            list(legacy.coeffs),
            positions,
            plan=legacy.pair_plan,
            **kwargs,
        )
        _assert_matrix_parity(actual, expected)

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


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is not supported for this Python/torch combination",
)
def test_node_plan_compiles_fullgraph_without_host_rebuild() -> None:
    """Fullgraph compilation consumes the captured plan directly."""
    calc, _, _, positions = _calculator(GFN1_XTB, "H2")
    setup = calc.system.integral_setup.pytorch
    fn = lambda geometry: build_overlap(setup, geometry)
    torch._dynamo.reset()
    actual = torch.compile(fn, fullgraph=True, backend="aot_eager")(positions)
    torch.testing.assert_close(actual, fn(positions), rtol=1e-12, atol=1e-13)
