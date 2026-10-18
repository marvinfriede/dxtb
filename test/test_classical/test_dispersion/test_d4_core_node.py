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
"""Tests for immutable classical D4 setup Nodes."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
import tad_dftd4 as d4
import torch
from tad_mctc.tree import Node, stack
from torch.utils import _pytree

from dxtb import GFN2_XTB, ParamModule
from dxtb._src.components.classicals import (
    DispersionD4,
    DispersionD4CoreSetup,
    DispersionD4Setup,
    core_dispersion_d4_setup,
    dispersion_d4_energy,
)
from dxtb._src.components.classicals.dispersion.d4 import _d4_setup_view
from dxtb._src.typing import Tensor

from ...conftest import DEVICE
from .samples import samples

DD = {"device": DEVICE, "dtype": torch.double}
FLOAT_SETUP_FIELDS = (
    "model_wf",
    "model_rc6",
    "rcov",
    "r4r2",
    "cutoff",
)
CUTOFF_FIELDS = ("disp2", "disp3", "cn", "cn_eeq")


def _geometry() -> tuple[Tensor, Tensor]:
    sample = samples["C4H5NCS"]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _component(numbers: Tensor) -> DispersionD4:
    return DispersionD4(
        numbers,
        {
            "s8": torch.tensor(1.2, **DD),
            "a1": torch.tensor(0.48, **DD),
            "a2": 5.1,
        },
        charge=torch.tensor(0.0, **DD),
        **DD,
    )


def _tensor_leaves(setup: DispersionD4CoreSetup) -> list[Tensor]:
    leaves, _ = _pytree.tree_flatten(setup)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def test_d4_core_schema_tensor_leaves_and_roundtrip() -> None:
    """D4 tensors remain dynamic while labels and scalar values stay static."""
    numbers, _ = _geometry()
    legacy = _component(numbers).get_cache(numbers)
    assert type(legacy) is DispersionD4Setup
    core = core_dispersion_d4_setup(legacy)

    assert isinstance(core, Node)
    assert type(core) is DispersionD4CoreSetup
    assert tuple(field.name for field in fields(core)) == (
        "numbers",
        "parameter_entries",
        "model_numbers",
        "model_ga",
        "model_gc",
        "model_wf",
        "model_ref_charges",
        "model_rc6",
        "rcov",
        "r4r2",
        "cutoff",
        "counting_function",
        "damping_type",
        "ref_charges",
        "compatibility_q",
    )
    assert core.numbers is legacy.numbers
    assert core.model_numbers is legacy.model.numbers
    assert core.model_wf is legacy.model.wf
    assert core.model_rc6 is legacy.model.rc6
    assert core.rcov is legacy.rcov
    assert core.r4r2 is legacy.r4r2
    assert all(
        getattr(core.cutoff, name) is getattr(legacy.cutoff, name)
        for name in CUTOFF_FIELDS
    )
    assert core.counting_function is legacy.counting_function
    assert core.ref_charges == legacy.ref_charges
    assert core.model_ref_charges == legacy.model.ref_charges

    expected_parameters = legacy.parameters
    assert tuple(name for name, _ in core.parameters) == tuple(
        name for name, _ in expected_parameters
    )
    assert type(core.parameters[0][1]) is torch.Tensor
    assert type(core.parameters[1][1]) is torch.Tensor
    assert type(core.parameters[2][1]) is float
    assert core.parameters[0][1] is expected_parameters[0][1]
    assert core.parameters[1][1] is expected_parameters[1][1]
    assert core.as_param() == dict(expected_parameters)

    leaves = _tensor_leaves(core)
    assert len(leaves) == 12
    assert core.numbers.dtype == numbers.dtype
    assert all(leaf.device == numbers.device for leaf in leaves)
    rebuilt_leaves, spec = _pytree.tree_flatten(core)
    rebuilt = _pytree.tree_unflatten(rebuilt_leaves, spec)
    assert type(rebuilt) is DispersionD4CoreSetup
    assert all(
        a is b for a, b in zip(_tensor_leaves(core), _tensor_leaves(rebuilt))
    )
    assert tuple(name for name, _ in rebuilt.parameters) == tuple(
        name for name, _ in expected_parameters
    )


def test_d4_core_dtype_device_stack_and_vmap() -> None:
    """D4 tensor children convert, stack, and remain transform-visible."""
    numbers, _ = _geometry()
    core = core_dispersion_d4_setup(_component(numbers).get_cache(numbers))

    for dtype in (torch.float32, torch.float64):
        converted = core.to(dtype=dtype)
        assert type(converted) is DispersionD4CoreSetup
        assert converted.numbers.dtype == numbers.dtype
        assert converted.model_numbers.dtype == numbers.dtype
        assert all(
            getattr(converted, name).dtype == dtype
            for name in FLOAT_SETUP_FIELDS
        )
        assert all(
            getattr(converted.cutoff, name).dtype == dtype
            for name in CUTOFF_FIELDS
        )
        assert converted.parameters[0][1].dtype == dtype
        assert type(converted.parameters[2][1]) is float

    meta = core.to(device=torch.device("meta"))
    assert type(meta) is DispersionD4CoreSetup
    assert all(leaf.device.type == "meta" for leaf in _tensor_leaves(meta))
    assert meta.numbers.dtype == numbers.dtype

    stacked = stack((core, core))
    assert type(stacked) is DispersionD4CoreSetup
    for original, batched in zip(_tensor_leaves(core), _tensor_leaves(stacked)):
        assert batched.shape == (2, *original.shape)

    def probe(setup: DispersionD4CoreSetup) -> Tensor:
        return (
            setup.model_wf.sum()
            + setup.model_rc6.sum()
            + setup.cutoff.disp2.sum()
            + setup.parameters[0][1].sum()
        )

    torch.testing.assert_close(
        torch.func.vmap(probe)(stacked), torch.stack((probe(core), probe(core)))
    )


def test_d4_core_call_local_model_cutoff_and_damping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation reconstructs captured tables and fresh mutable externals."""
    numbers, positions = _geometry()
    legacy = _component(numbers).get_cache(numbers)
    core = core_dispersion_d4_setup(legacy)
    first = _d4_setup_view(core)
    second = _d4_setup_view(core)

    assert type(first.model) is d4.model.D4Model
    assert first.model is not legacy.model
    assert first.model.numbers is not core.model_numbers
    assert first.model.wf is not core.model_wf
    assert first.model.rc6 is not core.model_rc6
    torch.testing.assert_close(first.model.numbers, legacy.model.numbers)
    torch.testing.assert_close(first.model.wf, legacy.model.wf)
    torch.testing.assert_close(first.model.rc6, legacy.model.rc6)
    assert type(first.cutoff) is d4.Cutoff
    assert first.cutoff is not second.cutoff
    assert all(
        getattr(first.cutoff, name) is not getattr(core.cutoff, name)
        for name in CUTOFF_FIELDS
    )
    assert first.damping_function is not second.damping_function
    assert type(first.damping_function) is type(second.damping_function)
    first.damping_function.doi = "mutated call-local DOI"
    assert second.damping_function.doi is None
    first.damping_function(
        positions.new_ones((1,)),
        positions.new_ones((1,)),
        order=6,
        a1=0.5,
        a2=5.0,
        doi="evaluation DOI",
    )
    assert first.damping_function.doi == "evaluation DOI"
    assert second.damping_function.doi is None

    original_dftd4 = d4.dftd4
    evaluated_damping: list[d4.damping.RationalDamping] = []

    def capture_damping(*args: Any, **kwargs: Any) -> Tensor:
        damping = kwargs["damping_function"]
        assert isinstance(damping, d4.damping.RationalDamping)
        evaluated_damping.append(damping)
        return original_dftd4(*args, **kwargs)

    monkeypatch.setattr(d4, "dftd4", capture_damping)
    expected = dispersion_d4_energy(legacy, positions, 0.0)
    actual = dispersion_d4_energy(core, positions, 0.0)
    torch.testing.assert_close(actual, expected)
    assert len(evaluated_damping) == 2
    assert evaluated_damping[0] is not evaluated_damping[1]
    evaluated_damping[1](
        positions.new_ones((1,)),
        positions.new_ones((1,)),
        order=6,
        a1=0.5,
        a2=5.0,
        doi="captured evaluation metadata",
    )
    assert evaluated_damping[1].doi == "captured evaluation metadata"
    dispersion_d4_energy(core, positions, 0.0)
    assert len(evaluated_damping) == 3
    assert evaluated_damping[2] is not evaluated_damping[1]
    assert evaluated_damping[2].doi is None
    assert core.damping_type is d4.damping.RationalDamping
    assert first.model.wf is not core.model_wf


def test_d4_core_sparse_parameters_and_compatibility_q() -> None:
    """Sparse ordered values and optional compatibility charges survive."""
    numbers, positions = _geometry()
    component = _component(numbers)
    q = torch.linspace(-0.2, 0.2, numbers.numel(), **DD)
    legacy = component.get_cache(numbers, q=q)
    core = core_dispersion_d4_setup(legacy)

    assert tuple(name for name, _ in core.parameters) == ("s8", "a1", "a2")
    assert "s9" not in core.as_param()
    assert "s10" not in core.as_param()
    assert core.parameters[0][1] is legacy.parameters[0][1]
    assert core.compatibility_q is legacy.compatibility_q
    assert any(leaf is core.compatibility_q for leaf in _tensor_leaves(core))
    torch.testing.assert_close(
        component.get_energy(positions, core),
        component.get_energy(positions, legacy),
    )


def test_d4_core_captures_cutoff_snapshot_and_keeps_custom_legacy_damping() -> (
    None
):
    """Core conversion snapshots cutoffs while direct overrides stay legacy."""
    numbers, positions = _geometry()
    component = _component(numbers)
    cutoff = d4.Cutoff(**DD)
    cutoff.disp2.fill_(2.0)
    legacy = component.get_cache(numbers, cutoff=cutoff)
    core = core_dispersion_d4_setup(legacy)
    before = dispersion_d4_energy(core, positions, 0.0)
    cutoff.disp2.fill_(0.0)
    torch.testing.assert_close(
        dispersion_d4_energy(core, positions, 0.0), before
    )

    class CustomDamping(d4.damping.RationalDamping):
        def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
            return 2.0 * super().__call__(*args, **kwargs)

    damping = CustomDamping()
    custom_legacy = component.get_cache(numbers, damping_function=damping)
    assert type(custom_legacy) is DispersionD4Setup
    assert custom_legacy.damping_function is damping
    custom_energy = component.get_energy(positions, custom_legacy)
    assert torch.isfinite(custom_energy).all()


def test_d4_core_keeps_model_and_setup_charge_modes_distinct() -> None:
    """Model and classical setup reference-charge modes stay independent."""
    numbers, positions = _geometry()
    component = _component(numbers)
    model = d4.model.D4Model(numbers, ref_charges="gfn2", **DD)
    legacy = component.get_cache(numbers, model=model)
    core = core_dispersion_d4_setup(legacy)

    assert core.ref_charges == "eeq"
    assert core.model_ref_charges == "gfn2"
    reconstructed = _d4_setup_view(core)
    assert reconstructed.ref_charges == "eeq"
    assert reconstructed.model.ref_charges == "gfn2"
    torch.testing.assert_close(
        dispersion_d4_energy(core, positions, 0.0),
        dispersion_d4_energy(legacy, positions, 0.0),
    )


def test_d4_core_preserves_model_tensor_gradients() -> None:
    """Captured model weight and reference tensors retain source gradients."""
    numbers, positions = _geometry()
    source_model = d4.model.D4Model(numbers, ref_charges="eeq", **DD)
    wf = source_model.wf.clone().requires_grad_()
    rc6 = source_model.rc6.clone().requires_grad_()
    model = d4.model.D4Model(
        numbers,
        ga=source_model.ga,
        gc=source_model.gc,
        wf=wf,
        ref_charges=source_model.ref_charges,
        rc6=rc6,
        **DD,
    )
    legacy = _component(numbers).get_cache(numbers, model=model)
    core = core_dispersion_d4_setup(legacy)
    assert core.model_wf is legacy.model.wf
    assert core.model_rc6 is legacy.model.rc6

    energy = dispersion_d4_energy(core, positions, 0.0).sum()
    grad_wf, grad_rc6 = torch.autograd.grad(energy, (wf, rc6))
    assert torch.isfinite(grad_wf).all()
    assert torch.isfinite(grad_rc6).all()
    assert torch.count_nonzero(grad_wf) > 0
    assert torch.count_nonzero(grad_rc6) > 0


def test_d4_core_energy_position_and_parameter_gradients() -> None:
    """Core setup retains the legacy energy and differentiable graph paths."""
    numbers, positions = _geometry()
    component = _component(numbers)
    parameter = component.param["a1"]
    parameter.requires_grad_(True)
    legacy = component.get_cache(numbers)
    core = core_dispersion_d4_setup(legacy)

    legacy_positions = positions.clone().requires_grad_(True)
    legacy_energy = dispersion_d4_energy(legacy, legacy_positions, 0.0).sum()
    legacy_position_grad, legacy_parameter_grad = torch.autograd.grad(
        legacy_energy,
        (legacy_positions, parameter),
        create_graph=True,
    )
    core_positions = positions.clone().requires_grad_(True)
    core_energy = dispersion_d4_energy(core, core_positions, 0.0).sum()
    core_position_grad, core_parameter_grad = torch.autograd.grad(
        core_energy,
        (core_positions, parameter),
        create_graph=True,
    )
    torch.testing.assert_close(core_energy, legacy_energy)
    torch.testing.assert_close(core_position_grad, legacy_position_grad)
    torch.testing.assert_close(core_parameter_grad, legacy_parameter_grad)

    direction = torch.linspace(-0.3, 0.4, positions.numel(), **DD).reshape_as(
        positions
    )
    legacy_mixed = torch.autograd.grad(
        (legacy_position_grad * direction).sum(), parameter, retain_graph=True
    )[0]
    core_mixed = torch.autograd.grad(
        (core_position_grad * direction).sum(), parameter
    )[0]
    torch.testing.assert_close(core_mixed, legacy_mixed)
    assert torch.isfinite(core_mixed)


def test_core_conversion_rejects_unsupported_model_or_damping() -> None:
    """Custom external extensions remain explicit legacy-only setup values."""
    numbers, _ = _geometry()
    component = _component(numbers)

    class CustomD4Model(d4.model.D4Model):
        pass

    with pytest.raises(TypeError, match="built-in D4Model"):
        core_dispersion_d4_setup(
            component.get_cache(
                numbers,
                model=CustomD4Model(numbers, ref_charges="eeq", **DD),
            )
        )

    with pytest.raises(TypeError, match="RationalDamping"):
        core_dispersion_d4_setup(
            component.get_cache(
                numbers, damping_function=d4.damping.ZeroDamping()
            )
        )

    with pytest.raises(
        TypeError,
        match="Setup in DispersionD4 is not of type 'DispersionD4Setup'",
    ):
        component.get_energy(
            positions=torch.zeros((1, 3), **DD), cache=object()
        )
