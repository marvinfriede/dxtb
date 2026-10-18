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
"""Executable schema proofs for the planned C7 integral/H0 Nodes."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import Node, child, context, stack
from torch.func import vmap
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, Calculator, IndexHelper, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.libcint.driver import (
    LibcintIntegralSetupNode,
)
from dxtb._src.integral.driver.pytorch.impls.kernels import get_kernel
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    PairClassNode,
    PairPlanNode,
    assemble_matrix,
)
from dxtb._src.integral.driver.pytorch.overlap import build_overlap
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetupNode,
    core_pytorch_integral_setup,
    setup_integrals,
)
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional.core import CoreIndexData
from dxtb._src.typing import Tensor
from dxtb._src.xtb.h0 import H0SetupNode

from .samples import samples


class _IntegralSetupNode(Node):
    """Backend selection and one optional backend subtree."""

    driver_type: int = context()
    intlevel: int = context()
    pytorch: PytorchIntegralSetupNode | None = child()
    libcint: LibcintIntegralSetupNode | None = child()


def _pytorch_node(setup) -> PytorchIntegralSetupNode:
    return core_pytorch_integral_setup(setup)


def _tensor_leaves(value: object) -> tuple[Tensor, ...]:
    leaves, _ = tree_flatten(value)
    return tuple(item for item in leaves if isinstance(item, Tensor))


def test_pair_plan_node_roundtrip_conversion_stack_and_overlap() -> None:
    """The proposed ordered pair schema drives the existing overlap kernel."""
    numbers = samples["H2O"]["numbers"]
    positions = samples["H2O"]["positions"].to(dtype=torch.float64)
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    par.set_differentiable("element.H.slater")
    calc = Calculator(
        numbers,
        par,
        dtype=torch.float64,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    system = calc.system
    assert system.integral_setup is not None
    assert system.integral_setup.pytorch is not None
    assert type(system.integral_setup.pytorch) is PytorchIntegralSetupNode
    legacy = setup_integrals(Basis(numbers, par, system.ihelp))
    proposed = _pytorch_node(legacy)

    leaves, spec = tree_flatten(proposed)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is PytorchIntegralSetupNode
    assert type(rebuilt.pair_plan) is PairPlanNode
    assert type(rebuilt.index) is CoreIndexData
    assert rebuilt.algorithm == legacy.algorithm
    assert all(
        any(leaf is tensor for leaf in leaves)
        for tensor in (*legacy.alphas, *legacy.coeffs)
    )

    converted = proposed.to(dtype=torch.float32)
    assert all(value.dtype == torch.float32 for value in converted.alphas)
    assert all(value.dtype == torch.float32 for value in converted.coeffs)
    assert converted.pair_plan.atom.dtype == legacy.pair_plan.atom.dtype
    for old, new in zip(legacy.pair_plan.classes, converted.pair_plan.classes):
        assert new.ib.dtype == old.ib.dtype
        assert new.jk.dtype == old.jk.dtype
        assert new.idx.dtype == old.idx.dtype
        assert new.weight.dtype == torch.float32

    structural_meta = proposed.to(device="meta")
    assert structural_meta.index.shells_to_atom.device.type == "meta"
    assert structural_meta.pair_plan.atom.device.type == "meta"
    assert all(value.device.type == "meta" for value in structural_meta.alphas)
    assert all(
        item.idx.device.type == "meta"
        for item in structural_meta.pair_plan.classes
    )
    stacked = stack((proposed, proposed))
    assert type(stacked) is PytorchIntegralSetupNode
    assert proposed.algorithm not in tree_flatten(proposed)[0]
    assert stacked.pair_plan.atom.shape[0] == 2
    assert stacked.pair_plan.classes[0].idx.shape[0] == 2

    def plan_probe(value: PytorchIntegralSetupNode) -> Tensor:
        return value.pair_plan.atom.sum() + sum(
            entry.weight.sum() for entry in value.pair_plan.classes
        )

    mapped = vmap(plan_probe)(stacked)
    expected = torch.stack((plan_probe(proposed), plan_probe(proposed)))
    torch.testing.assert_close(mapped, expected)

    baseline = build_overlap(legacy, positions)
    from_node = assemble_matrix(
        get_kernel(proposed.algorithm),
        None,
        list(proposed.alphas),
        list(proposed.coeffs),
        positions,
        plan=proposed.pair_plan,
    )[0]
    torch.testing.assert_close(from_node, baseline)
    reference_positions = positions.clone().requires_grad_(True)
    candidate_positions = positions.clone().requires_grad_(True)
    reference_overlap = build_overlap(legacy, reference_positions)
    candidate_overlap = assemble_matrix(
        get_kernel(proposed.algorithm),
        None,
        list(proposed.alphas),
        list(proposed.coeffs),
        candidate_positions,
        plan=proposed.pair_plan,
    )[0]
    (reference_position_gradient,) = torch.autograd.grad(
        reference_overlap.square().sum(), reference_positions
    )
    (candidate_position_gradient,) = torch.autograd.grad(
        candidate_overlap.square().sum(), candidate_positions
    )
    torch.testing.assert_close(
        candidate_position_gradient, reference_position_gradient
    )
    (gradient,) = torch.autograd.grad(
        from_node.sum(), par.get("element.H.slater")
    )
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    direction = torch.ones_like(positions)
    _, tangent = torch.func.jvp(
        lambda geometry: assemble_matrix(
            get_kernel(proposed.algorithm),
            None,
            list(proposed.alphas),
            list(proposed.coeffs),
            geometry,
            plan=proposed.pair_plan,
        )[0],
        (positions,),
        (direction,),
    )
    assert torch.isfinite(tangent).all()

    for old, new in zip(legacy.pair_plan.classes, proposed.pair_plan.classes):
        assert (old.ub, old.uk, old.lb, old.lk) == (
            new.ub,
            new.uk,
            new.lb,
            new.lk,
        )
        for name in ("ib", "jk", "idx", "weight"):
            old_value, new_value = getattr(old, name), getattr(new, name)
            assert new_value is not old_value
            assert old_value.shape == new_value.shape
            assert old_value.dtype == new_value.dtype
            assert old_value.device == new_value.device
            torch.testing.assert_close(old_value, new_value, rtol=0.0, atol=0.0)
            assert any(leaf is new_value for leaf in leaves)
    assert proposed.pair_plan.atom is proposed.index.shells_to_atom
    assert proposed.pair_plan.atom is not legacy.pair_plan.atom
    assert any(leaf is proposed.pair_plan.atom for leaf in leaves)
    assert (
        proposed.index.orbitals_to_shell is not legacy.ihelp.orbitals_to_shell
    )
    assert any(leaf is proposed.index.orbitals_to_shell for leaf in leaves)


def test_integral_and_h0_optional_backend_schemas() -> None:
    """Backend choice, H0 CN descriptors, and nested EEQ remain explicit."""
    numbers = samples["H2"]["numbers"]
    pytorch_calc = Calculator(
        numbers,
        GFN1_XTB,
        dtype=torch.float64,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    assert pytorch_calc.system.integral_setup is not None
    source = pytorch_calc.system.integral_setup
    assert source.pytorch is not None and source.libcint is None
    assert type(source.pytorch) is PytorchIntegralSetupNode
    pytorch_node = _IntegralSetupNode(
        driver_type=source.driver_type,
        intlevel=source.intlevel,
        pytorch=source.pytorch,
        libcint=None,
    )
    leaves, spec = tree_flatten(pytorch_node)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is _IntegralSetupNode
    assert rebuilt.driver_type == labels.INTDRIVER_PYTORCH
    assert rebuilt.pytorch is not None and rebuilt.libcint is None

    for method, parametrization, expected_cn in (
        ("gfn0", GFN0_XTB, "erf_count"),
        ("gfn1", GFN1_XTB, "cn_d3(exp_count)"),
        ("gfn2", GFN2_XTB, "cn_d3(gfn2_count)"),
    ):
        calc = Calculator(
            numbers,
            parametrization,
            dtype=torch.float64,
            opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
        )
        assert calc.system.h0_setup is not None
        source_h0 = calc.system.h0_setup
        assert type(source_h0) is H0SetupNode
        assert not hasattr(source_h0, "param")
        assert not isinstance(source_h0.index, IndexHelper)
        leaves, spec = tree_flatten(source_h0)
        rebuilt_h0 = tree_unflatten(leaves, spec)
        assert type(rebuilt_h0) is H0SetupNode
        assert rebuilt_h0.method == method
        assert rebuilt_h0.cn_kind == expected_cn
        assert all(
            any(leaf is tensor for leaf in leaves)
            for tensor in (
                source_h0.numbers,
                source_h0.unique,
                source_h0.hscale,
                source_h0.kcn,
                source_h0.kpair,
                source_h0.refocc,
                source_h0.selfenergy,
                source_h0.shpoly,
                source_h0.valence,
                source_h0.en,
                source_h0.enscale,
                source_h0.rad,
                source_h0.index.atom_to_unique,
                source_h0.index.shells_to_ushell,
                source_h0.index.unique_angular,
                source_h0.index.shells_to_atom,
                source_h0.index.orbitals_to_shell,
            )
        )
        for name in (
            "kq",
            "kqat",
            "h0rad",
            "kdiff",
            "enshell",
            "enscale4",
            "cn_radii",
            "cn_cutoff",
            "cn_max",
            "cn_kcn",
        ):
            tensor = getattr(source_h0, name)
            if tensor is not None:
                assert any(leaf is tensor for leaf in leaves)
        if source_h0.eeq_model is not None:
            assert all(
                any(leaf is tensor for leaf in leaves)
                for tensor in (
                    source_h0.eeq_model.chi,
                    source_h0.eeq_model.kcn,
                    source_h0.eeq_model.eta,
                    source_h0.eeq_model.rad,
                )
            )
        converted_h0 = source_h0.to(dtype=torch.float32)
        assert converted_h0.numbers.dtype == source_h0.numbers.dtype
        assert converted_h0.unique.dtype == source_h0.unique.dtype
        assert converted_h0.valence.dtype == source_h0.valence.dtype
        assert converted_h0.hscale.dtype == torch.float32
        if source_h0.eeq_model is not None:
            assert converted_h0.eeq_model is not source_h0.eeq_model
            assert converted_h0.eeq_model.chi.dtype == torch.float32

        assert converted_h0.index.atom_to_unique.dtype == torch.int64
        assert converted_h0.index.unique_angular.dtype == torch.int64
        assert converted_h0.cn_kind == source_h0.cn_kind


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_libcint_node_schema_keeps_wrappers_call_local() -> None:
    """Libcint structural values are separate from geometry wrappers."""
    numbers = samples["H2"]["numbers"]
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=torch.float64,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_LIBCINT},
    )
    assert calc.system.integral_setup is not None
    source = calc.system.integral_setup.libcint
    assert source is not None
    assert not hasattr(source, "drivers")
    assert type(source) is LibcintIntegralSetupNode
    proposed = source
    integral_node = _IntegralSetupNode(
        driver_type=calc.system.integral_setup.driver_type,
        intlevel=calc.system.integral_setup.intlevel,
        pytorch=None,
        libcint=proposed,
    )
    integral_leaves, integral_spec = tree_flatten(integral_node)
    integral_rebuilt = tree_unflatten(integral_leaves, integral_spec)
    assert integral_rebuilt.pytorch is None
    assert integral_rebuilt.libcint is not None
    leaves, spec = tree_flatten(proposed)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is LibcintIntegralSetupNode
    assert rebuilt.force_cpu is source.force_cpu
    assert all(not hasattr(basis, "par") for basis in rebuilt.basis_setups)
    assert all(not hasattr(basis, "ihelp") for basis in rebuilt.basis_setups)
    assert not hasattr(rebuilt, "ihelp")
    assert all(leaf.device.type == "cpu" for leaf in _tensor_leaves(rebuilt))
    converted = proposed.to(device="meta", dtype=torch.float32)
    assert converted.numbers.device.type == "cpu"
    assert all(leaf.device.type == "cpu" for leaf in _tensor_leaves(converted))
