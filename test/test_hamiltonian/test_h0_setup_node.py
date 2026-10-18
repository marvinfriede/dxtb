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
"""Production Node schema and parity tests for pure H0 setup."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.func import jacfwd, jacrev, jvp, vmap
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import (
    GFN0_XTB,
    GFN1_XTB,
    GFN2_XTB,
    Calculator,
    IndexHelper,
    ParamModule,
    labels,
)
from dxtb._src.ncoord import cn_d3, exp_count, gfn2_count
from dxtb._src.xtb import h0 as h0_module
from dxtb._src.xtb.h0 import (
    H0IndexNode,
    H0Setup,
    H0SetupNode,
    build_hcore,
    build_hcore_raw,
    core_h0_setup,
    setup_h0,
)

from .samples import samples

_METHODS = {
    "gfn0": GFN0_XTB,
    "gfn1": GFN1_XTB,
    "gfn2": GFN2_XTB,
}


def _setups(method: str, sample: str = "H2O"):
    numbers = samples[sample]["numbers"].to(device="cpu")
    positions = samples[sample]["positions"].to(
        device="cpu", dtype=torch.float64
    )
    par = ParamModule(
        _METHODS[method].model_copy(deep=True),
        device=positions.device,
        dtype=positions.dtype,
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    legacy = setup_h0(numbers, par, ihelp)
    return numbers, positions, par, ihelp, legacy, core_h0_setup(legacy)


def _eye(size: int, positions: torch.Tensor) -> torch.Tensor:
    return torch.eye(size, device=positions.device, dtype=positions.dtype)


def _assert_close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_h0_core_schema_roundtrip_dtype_device_and_stack(method: str) -> None:
    """All setup tensors are visible leaves and structural dtype is retained."""
    _, _, _, _, legacy, core = _setups(method)
    assert type(core) is H0SetupNode
    assert isinstance(core, Node)
    assert core.method == method
    assert core.index is not None
    assert not hasattr(core, "param")
    assert not isinstance(core.index, IndexHelper)
    assert core.ihelp is core.index
    assert (
        core.cn_kind
        == {
            "gfn0": "erf_count",
            "gfn1": "cn_d3(exp_count)",
            "gfn2": "cn_d3(gfn2_count)",
        }[method]
    )

    leaves, spec = tree_flatten(core)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is H0SetupNode
    assert type(rebuilt.index) is H0IndexNode
    assert rebuilt.method == core.method
    assert rebuilt.cn_kind == core.cn_kind

    outer_tensor_fields = (
        "numbers",
        "unique",
        "hscale",
        "kcn",
        "kpair",
        "refocc",
        "selfenergy",
        "shpoly",
        "valence",
        "en",
        "enscale",
        "rad",
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
    )
    for name in outer_tensor_fields:
        tensor = getattr(core, name)
        if tensor is not None:
            assert any(leaf is tensor for leaf in leaves), name
            assert torch.equal(tensor, getattr(legacy, name))
    for name in (
        "atom_to_unique",
        "shells_to_ushell",
        "unique_angular",
        "shells_to_atom",
        "orbitals_to_shell",
    ):
        tensor = getattr(core.index, name)
        assert any(leaf is tensor for leaf in leaves), name
        assert tensor.dtype in (torch.int32, torch.int64)
        assert torch.equal(tensor, getattr(legacy.ihelp, name))

    assert core.numbers.dtype == legacy.numbers.dtype
    assert core.unique.dtype == legacy.unique.dtype
    assert core.valence.dtype == legacy.valence.dtype
    converted = core.to(dtype=torch.float32)
    assert converted.numbers.dtype == core.numbers.dtype
    assert converted.unique.dtype == core.unique.dtype
    assert converted.valence.dtype == core.valence.dtype
    assert (
        converted.index.unique_angular.dtype == core.index.unique_angular.dtype
    )
    assert converted.hscale.dtype == torch.float32
    assert converted.refocc.dtype == torch.float32
    assert core.hscale.dtype == torch.float64
    assert converted.cn_kind == core.cn_kind
    if core.eeq_model is not None:
        assert converted.eeq_model is not core.eeq_model
        assert converted.eeq_model.chi.dtype == torch.float32
        assert converted.eeq_model.kcn.dtype == torch.float32
        assert converted.eeq_model.eta.dtype == torch.float32
        assert converted.eeq_model.rad.dtype == torch.float32
        eeq_leaves = tree_flatten(core.eeq_model)[0]
        for tensor in (
            core.eeq_model.chi,
            core.eeq_model.kcn,
            core.eeq_model.eta,
            core.eeq_model.rad,
        ):
            assert any(leaf is tensor for leaf in leaves)
            assert any(leaf is tensor for leaf in eeq_leaves)

    meta = core.to(device="meta")
    assert meta.numbers.device.type == "meta"
    assert meta.index.atom_to_unique.device.type == "meta"
    assert meta.index.orbitals_to_shell.device.type == "meta"
    assert (
        meta.index.orbitals_to_shell.dtype == core.index.orbitals_to_shell.dtype
    )
    assert meta.hscale.device.type == "meta"
    if core.eeq_model is not None:
        assert meta.eeq_model.chi.device.type == "meta"
    stacked = stack((core, core))
    assert type(stacked) is H0SetupNode
    assert type(stacked.index) is H0IndexNode
    assert stacked.numbers.shape[0] == 2
    assert stacked.index.atom_to_unique.shape[0] == 2
    assert stacked.hscale.shape[0] == 2


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_h0_index_spreads_match_index_helper(method: str) -> None:
    """Narrow H0 gather maps match IndexHelper for vectors and pairs."""
    numbers, _, _, ihelp, _, core = _setups(method)
    index = core.index
    pairs = (
        ("spread_uspecies_to_atom", len(torch.unique(numbers))),
        ("spread_uspecies_to_shell", len(torch.unique(numbers))),
        (
            "spread_ushell_to_shell",
            int(ihelp.shells_to_ushell.max().item()) + 1,
        ),
        ("spread_atom_to_shell", numbers.numel()),
        ("spread_shell_to_orbital", ihelp.shells_to_atom.numel()),
    )
    for name, size in pairs:
        source = torch.arange(size, dtype=torch.float64, requires_grad=True)
        expected = getattr(ihelp, name)(source, dim=-1)
        actual = getattr(index, name)(source, dim=-1)
        _assert_close(actual, expected)
        expected_gradient = torch.autograd.grad(
            expected.square().sum(), source
        )[0]
        actual_gradient = torch.autograd.grad(actual.square().sum(), source)[0]
        _assert_close(actual_gradient, expected_gradient)

    for name, size in pairs:
        source = (
            torch.arange(size * size, dtype=torch.float64)
            .reshape(size, size)
            .requires_grad_(True)
        )
        expected = getattr(ihelp, name)(source, dim=(-2, -1))
        actual = getattr(index, name)(source, dim=(-2, -1))
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        _assert_close(actual, expected)
        expected_gradient = torch.autograd.grad(
            expected.square().sum(), source
        )[0]
        actual_gradient = torch.autograd.grad(actual.square().sum(), source)[0]
        _assert_close(actual_gradient, expected_gradient)

    source = torch.arange(numbers.numel() * 3, dtype=torch.float64).reshape(
        numbers.numel(), 3
    )
    expected = ihelp.spread_atom_to_shell(source, dim=-2, extra=True)
    actual = index.spread_atom_to_shell(source, dim=-2, extra=True)
    assert actual.shape == expected.shape
    _assert_close(actual, expected)

    stacked = stack((index, index))
    assert type(stacked) is H0IndexNode
    assert stacked.unique_angular.shape[0] == 2
    converted = index.to(dtype=torch.float32, device="meta")
    assert converted.atom_to_unique.dtype == index.atom_to_unique.dtype
    assert converted.atom_to_unique.device.type == "meta"


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
@pytest.mark.parametrize("sample", ["H2", "LiH", "H2O"])
def test_h0_raw_weighted_and_reference_occupation_parity(
    method: str, sample: str
) -> None:
    """Raw and overlap-weighted H0 reuse setup values exactly."""
    _, positions, _, ihelp, legacy, core = _setups(method, sample)
    overlap = _eye(ihelp.nao, positions) + 0.04 * (
        torch.ones((ihelp.nao, ihelp.nao), dtype=positions.dtype)
        - _eye(ihelp.nao, positions)
    )
    charge = torch.tensor(0.3, dtype=positions.dtype)
    charge_arg = charge if method == "gfn0" else None
    expected_raw, expected_refocc = build_hcore_raw(
        legacy, positions, charge=charge_arg
    )
    actual_raw, actual_refocc = build_hcore_raw(
        core, positions, charge=charge_arg
    )
    _assert_close(actual_raw, expected_raw)
    _assert_close(actual_refocc, expected_refocc)
    expected, expected_refocc = build_hcore(
        legacy, positions, overlap, charge=charge_arg
    )
    actual, actual_refocc_node = build_hcore(
        core, positions, overlap, charge=charge_arg
    )
    _assert_close(actual, expected)
    _assert_close(actual_refocc_node, expected_refocc)


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_h0_geometry_parameter_and_transform_parity(method: str) -> None:
    """Captured H0 retains parameter graphs and supported geometry transforms."""
    _, positions, par, ihelp, legacy, core = _setups(method, "H2")
    for parameter in par.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)
    # Rebuild after enabling differentiation so every setup gather is connected.
    legacy = setup_h0(legacy.numbers, par, ihelp)
    core = core_h0_setup(legacy)
    overlap = _eye(ihelp.nao, positions) + 0.03 * (
        torch.ones((ihelp.nao, ihelp.nao), dtype=positions.dtype)
        - _eye(ihelp.nao, positions)
    )
    charge = torch.tensor(0.2, dtype=positions.dtype, requires_grad=True)
    charge_arg = charge if method == "gfn0" else None

    def objective(setup, geometry):
        hcore, refocc = build_hcore(setup, geometry, overlap, charge=charge_arg)
        return hcore.square().sum() + refocc.square().sum()

    def gradient(setup, geometry):
        return torch.autograd.grad(
            objective(setup, geometry), geometry, create_graph=True
        )[0]

    x0 = positions.clone().requires_grad_(True)
    x1 = positions.clone().requires_grad_(True)
    value0, value1 = objective(core, x0), objective(legacy, x1)
    _assert_close(value0, value1)
    parameters = tuple(
        parameter for parameter in par.parameters() if parameter.requires_grad
    )
    grad0 = torch.autograd.grad(
        value0, parameters, retain_graph=True, allow_unused=True
    )
    grad1 = torch.autograd.grad(
        value1, parameters, retain_graph=True, allow_unused=True
    )
    for a, b in zip(grad0, grad1):
        if a is not None or b is not None:
            assert a is not None and b is not None
            _assert_close(a, b)
    assert any(
        value is not None and torch.isfinite(value).all() for value in grad0
    )

    g0, g1 = gradient(core, x0), gradient(legacy, x1)
    _assert_close(g0, g1)
    levels = dict(par.named_parameters())[
        "parameter_tree.element.H.levels.param"
    ]
    mixed0 = torch.autograd.grad(g0.sum(), levels, retain_graph=True)[0]
    mixed1 = torch.autograd.grad(g1.sum(), levels, retain_graph=True)[0]
    assert torch.isfinite(mixed0).all()
    _assert_close(mixed0, mixed1)
    second0 = torch.autograd.grad(g0.sum(), x0, create_graph=True)[0]
    second1 = torch.autograd.grad(g1.sum(), x1, create_graph=True)[0]
    _assert_close(second0, second1)
    third0 = torch.autograd.grad(second0.sum(), x0)[0]
    third1 = torch.autograd.grad(second1.sum(), x1)[0]
    _assert_close(third0, third1)

    direction = torch.ones_like(positions)
    _, tangent0 = jvp(
        lambda x: build_hcore(core, x, overlap, charge_arg)[0],
        (positions,),
        (direction,),
    )
    _, tangent1 = jvp(
        lambda x: build_hcore(legacy, x, overlap, charge_arg)[0],
        (positions,),
        (direction,),
    )
    _assert_close(tangent0, tangent1)
    f0 = lambda x: build_hcore(core, x, overlap, charge_arg)[0].reshape(-1)
    f1 = lambda x: build_hcore(legacy, x, overlap, charge_arg)[0].reshape(-1)
    _assert_close(jacfwd(f0)(positions), jacfwd(f1)(positions))
    _assert_close(jacrev(f0)(positions), jacrev(f1)(positions))
    geometries = torch.stack((positions, positions + 0.02))
    mapped = vmap(lambda x: build_hcore(core, x, overlap, charge_arg)[0])(
        geometries
    )
    explicit = torch.stack(
        [build_hcore(core, x, overlap, charge_arg)[0] for x in geometries]
    )
    _assert_close(mapped, explicit)
    stacked = stack((core, core))
    setup_probe = vmap(lambda setup: setup.hscale.sum() + setup.refocc.sum())(
        stacked
    )
    probe_value = core.hscale.sum() + core.refocc.sum()
    _assert_close(setup_probe, torch.stack((probe_value, probe_value)))
    if method == "gfn0":
        charge_grad = torch.autograd.grad(value0, charge, allow_unused=True)[0]
        assert charge_grad is not None and torch.isfinite(charge_grad).all()


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_h0_model_setup_owns_node_and_legacy_calculator_still_works(
    method: str,
) -> None:
    """Single-System setup owns Node; legacy Hamiltonian receives a view."""
    numbers = samples["H2"]["numbers"]
    calc = Calculator(
        numbers,
        _METHODS[method],
        dtype=torch.float64,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    assert type(calc.system.h0_setup) is H0SetupNode
    assert calc.integrals.hcore is not None
    assert type(calc.integrals.hcore.setup) is H0Setup
    assert calc.integrals.hcore.setup.ihelp is calc.system.ihelp
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    charge = 0.0 if method == "gfn0" else None
    overlap = calc.integrals.build_overlap(positions)
    legacy_h0 = calc.integrals.build_hcore(
        positions, with_overlap=True, charge=charge
    )
    pure_h0, _ = build_hcore(
        calc.system.h0_setup, positions, overlap, charge=charge
    )
    _assert_close(legacy_h0, pure_h0)
    result = calc.system.singlepoint(positions, chrg=0.0)
    assert torch.isfinite(result.energy).all()
    legacy_system = replace(calc.system, h0_setup=calc.integrals.hcore.setup)
    legacy_result = legacy_system.singlepoint(positions, chrg=0.0)
    _assert_close(result.energy, legacy_result.energy)


@pytest.mark.parametrize("method", ["gfn1", "gfn2"])
def test_legacy_h0_adapter_isolated_from_system_node(method: str) -> None:
    """The retained legacy H0 view cannot mutate the System-owned Node."""
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        _METHODS[method],
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    core = calc.system.h0_setup
    adapter = calc.integrals.hcore.setup
    assert type(core) is H0SetupNode
    assert type(adapter) is H0Setup

    for name in (
        "numbers",
        "unique",
        "hscale",
        "kcn",
        "kpair",
        "refocc",
        "selfenergy",
        "shpoly",
        "valence",
        "en",
        "enscale",
        "rad",
    ):
        core_value = getattr(core, name)
        adapter_value = getattr(adapter, name)
        assert adapter_value is not core_value, name
        assert torch.equal(adapter_value, core_value), name

    overlap = _eye(calc.ihelp.nao, positions)
    initial_h0, initial_refocc = build_hcore(core, positions, overlap)
    for name in ("hscale", "selfenergy", "refocc"):
        core_value = getattr(core, name)
        adapter_value = getattr(adapter, name)
        before = core_value.clone()
        adapter_before = adapter_value.clone()
        with torch.no_grad():
            adapter_value.add_(0.25)
        assert torch.equal(core_value, before), name
        # The Calculator-owned legacy Hamiltonian uses its own captured view.
        assert not torch.equal(adapter_value, adapter_before), name
        with torch.no_grad():
            adapter_value.copy_(adapter_before)
        assert torch.equal(core_value, before), name

    # Selfenergy changes diagonal and off-diagonal raw H0 deterministically.
    initial_raw, _ = build_hcore_raw(core, positions)
    adapter_selfenergy = adapter.selfenergy
    adapter_selfenergy_before = adapter_selfenergy.clone()
    with torch.no_grad():
        adapter_selfenergy.add_(0.5)
    core_raw, _ = build_hcore_raw(core, positions)
    adapter_raw, _ = build_hcore_raw(adapter, positions)
    _assert_close(core_raw, initial_raw)
    assert not torch.allclose(adapter_raw, initial_raw)
    with torch.no_grad():
        adapter_selfenergy.copy_(adapter_selfenergy_before)

    final_h0, final_refocc = build_hcore(core, positions, overlap)
    _assert_close(final_h0, initial_h0)
    _assert_close(final_refocc, initial_refocc)

    # The opposite mutation direction is isolated too.
    core_before = core.hscale.clone()
    adapter_before = adapter.hscale.clone()
    with torch.no_grad():
        core.hscale.add_(0.1)
    assert torch.equal(adapter.hscale, adapter_before)
    with torch.no_grad():
        core.hscale.copy_(core_before)


def test_gfn0_legacy_h0_adapter_owns_separate_eeq_model() -> None:
    """Calculator-local GFN0 EEQ tensors do not alias the core Node."""
    sample = samples["H2O"]
    positions = sample["positions"].to(dtype=torch.float64)
    calc = Calculator(
        sample["numbers"],
        GFN0_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    core = calc.system.h0_setup
    adapter = calc.integrals.hcore.setup
    assert type(core) is H0SetupNode
    assert type(adapter) is H0Setup
    assert core.eeq_model is not None and adapter.eeq_model is not None
    assert adapter.eeq_model is not core.eeq_model
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
        node_tensor = getattr(core, name)
        adapter_tensor = getattr(adapter, name)
        assert node_tensor is not None and adapter_tensor is not None
        assert adapter_tensor is not node_tensor
        assert torch.equal(adapter_tensor, node_tensor)
    for name in ("chi", "kcn", "eta", "rad"):
        node_tensor = getattr(core.eeq_model, name)
        adapter_tensor = getattr(adapter.eeq_model, name)
        assert adapter_tensor is not node_tensor
        assert torch.equal(adapter_tensor, node_tensor)

    node_chi = core.eeq_model.chi.clone()
    adapter_chi = adapter.eeq_model.chi.clone()
    with torch.no_grad():
        adapter.eeq_model.chi.add_(0.5)
    assert torch.equal(core.eeq_model.chi, node_chi)
    assert not torch.equal(adapter.eeq_model.chi, adapter_chi)


@pytest.mark.parametrize(
    "method, parameter_name",
    [
        ("gfn0", "parameter_tree.element.H.eeq_chi.param"),
        ("gfn1", "parameter_tree.element.H.levels.param"),
        ("gfn2", "parameter_tree.element.H.levels.param"),
    ],
)
def test_h0_adapter_clones_preserve_parameter_and_mixed_gradients(
    method: str, parameter_name: str
) -> None:
    """Node and isolated Calculator adapter retain the original graph edges."""
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    par = ParamModule(
        _METHODS[method].model_copy(deep=True), dtype=positions.dtype
    )
    parameter = dict(par.named_parameters())[parameter_name]
    parameter.requires_grad_(True)
    calc = Calculator(
        numbers,
        par,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    core = calc.system.h0_setup
    adapter = calc.integrals.hcore.setup
    assert isinstance(core, H0SetupNode)
    assert isinstance(adapter, H0Setup)
    overlap = _eye(calc.ihelp.nao, positions)
    charge = torch.tensor(0.2, dtype=positions.dtype)
    charge_arg = charge if method == "gfn0" else None
    x_core = positions.clone().requires_grad_(True)
    x_adapter = positions.clone().requires_grad_(True)

    def energy(setup, geometry):
        hcore, refocc = build_hcore(setup, geometry, overlap, charge=charge_arg)
        return hcore.square().sum() + refocc.square().sum()

    e_core = energy(core, x_core)
    e_adapter = energy(adapter, x_adapter)
    grad_core = torch.autograd.grad(
        e_core, parameter, create_graph=True, retain_graph=True
    )[0]
    grad_adapter = torch.autograd.grad(
        e_adapter, parameter, create_graph=True, retain_graph=True
    )[0]
    assert torch.isfinite(grad_core).all()
    _assert_close(grad_core, grad_adapter)

    position_grad_core = torch.autograd.grad(
        e_core, x_core, create_graph=True, retain_graph=True
    )[0]
    position_grad_adapter = torch.autograd.grad(
        e_adapter, x_adapter, create_graph=True, retain_graph=True
    )[0]
    _assert_close(position_grad_core, position_grad_adapter)
    mixed_core = torch.autograd.grad(
        position_grad_core.sum(), parameter, retain_graph=True
    )[0]
    mixed_adapter = torch.autograd.grad(position_grad_adapter.sum(), parameter)[
        0
    ]
    assert torch.isfinite(mixed_core).all()
    _assert_close(mixed_core, mixed_adapter)


def test_calculator_type_keeps_h0_adapter_ownership_and_dtype() -> None:
    """Calculator.type converts both independently owned H0 representations."""
    calc = Calculator(
        samples["H2"]["numbers"],
        GFN1_XTB,
        dtype=torch.float64,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    calc.type(torch.float32)
    core = calc.system.h0_setup
    adapter = calc.integrals.hcore.setup
    assert isinstance(core, H0SetupNode)
    assert isinstance(adapter, H0Setup)
    assert core.hscale.dtype == torch.float32
    assert adapter.hscale.dtype == torch.float32
    assert adapter.hscale is not core.hscale
    assert torch.equal(adapter.hscale, core.hscale)
    assert adapter.numbers.dtype == core.numbers.dtype
    assert adapter.valence.dtype == core.valence.dtype


def test_model_setup_captures_h0_once_and_packed_route_stays_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single setup gathers once; packed setup retains the dataclass route."""
    from tad_mctc.batch import pack

    numbers = samples["H2"]["numbers"]
    calc = Calculator(
        numbers,
        GFN1_XTB,
        dtype=torch.float64,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
    )
    original = h0_module.setup_h0
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(h0_module, "setup_h0", counted)
    single = calc.model.setup(numbers, dd=calc.dd)
    assert len(calls) == 1
    assert type(single.h0_setup) is H0SetupNode
    single.singlepoint(samples["H2"]["positions"].to(dtype=torch.float64))
    assert len(calls) == 1

    packed = pack((samples["H2"]["numbers"], samples["LiH"]["numbers"]))
    batch = calc.model.setup(packed, batch_mode=1, dd=calc.dd)
    assert type(batch.h0_setup) is H0Setup


def test_h0_legacy_custom_callables_remain_supported_and_conversion_rejects() -> (
    None
):
    """Legacy custom CN works; unsupported callables cannot enter Node context."""
    _, positions, _, ihelp, legacy, _ = _setups("gfn1", "H2")
    overlap = _eye(ihelp.nao, positions)
    captured = torch.tensor(0.25, dtype=positions.dtype, requires_grad=True)

    def closure(numbers, geometry):
        return torch.zeros_like(numbers, dtype=geometry.dtype) + captured

    custom = replace(legacy, cn=closure)
    value, _ = build_hcore(custom, positions, overlap)
    assert torch.isfinite(value).all()
    (closure_gradient,) = torch.autograd.grad(value.sum(), captured)
    assert torch.isfinite(closure_gradient).all()
    with pytest.raises(TypeError, match="custom GFN1 CN"):
        core_h0_setup(custom)

    wrong_count = replace(
        legacy,
        cn=partial(cn_d3, counting_function=gfn2_count),
    )
    with pytest.raises(TypeError, match="built-in GFN1 CN strategy"):
        core_h0_setup(wrong_count)

    extra_argument = replace(
        legacy,
        cn=partial(cn_d3, positions, counting_function=exp_count),
    )
    with pytest.raises(TypeError, match="built-in GFN1 CN strategy"):
        core_h0_setup(extra_argument)


def test_h0_setup_snapshot_isolated_and_history_independent() -> None:
    """System H0 tensors and index maps resist legacy alias mutation."""
    numbers = samples["H2O"]["numbers"]
    positions = samples["H2O"]["positions"].to(dtype=torch.float64)
    par = ParamModule(GFN1_XTB.model_copy(deep=True), dtype=positions.dtype)
    ihelp = IndexHelper.from_numbers(numbers, par)
    legacy = setup_h0(numbers, par, ihelp)
    core = core_h0_setup(legacy)
    overlap = _eye(ihelp.nao, positions)
    before, refocc = build_hcore(core, positions, overlap)
    a = positions.clone().requires_grad_(True)
    first, _ = build_hcore(core, a, overlap)
    grad_first = torch.autograd.grad(first.sum(), a)[0]
    moved = positions + 0.05
    build_hcore(core, moved, overlap)
    repeated, repeated_refocc = build_hcore(core, positions.clone(), overlap)
    _assert_close(repeated, before)
    _assert_close(repeated_refocc, refocc)
    b = positions.clone().requires_grad_(True)
    grad_repeat = torch.autograd.grad(
        build_hcore(core, b, overlap)[0].sum(), b
    )[0]
    _assert_close(grad_first, grad_repeat)

    snapshots = {
        "hscale": core.hscale.clone(),
        "refocc": core.refocc.clone(),
        "atom_to_unique": core.index.atom_to_unique.clone(),
        "eeq_chi": (
            None if core.eeq_model is None else core.eeq_model.chi.clone()
        ),
    }
    with torch.no_grad():
        legacy.hscale.add_(1.0)
        legacy.refocc.add_(1.0)
        legacy.ihelp.atom_to_unique.add_(1)
        if legacy.eeq_model is not None:
            legacy.eeq_model.chi.add_(1.0)
    _assert_close(core.hscale, snapshots["hscale"])
    _assert_close(core.refocc, snapshots["refocc"])
    assert torch.equal(core.index.atom_to_unique, snapshots["atom_to_unique"])
    if snapshots["eeq_chi"] is not None:
        _assert_close(core.eeq_model.chi, snapshots["eeq_chi"])


def test_gfn0_missing_charge_keeps_legacy_error() -> None:
    """Node H0 retains the established GFN0 charge requirement."""
    _, positions, _, ihelp, legacy, core = _setups("gfn0", "H2")
    overlap = _eye(ihelp.nao, positions)
    with pytest.raises(ValueError, match="charge is required"):
        build_hcore(legacy, positions, overlap)
    with pytest.raises(ValueError, match="charge is required"):
        build_hcore(core, positions, overlap)
