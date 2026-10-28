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
"""IES core setup Node and nested EEQ model tests."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.tree import Node, stack
from tad_multicharge.model.eeq import EEQModel
from torch.utils import _pytree

from dxtb import GFN0_XTB, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    IESCoreSetup,
    IESSetup,
    core_ies_setup,
    ies_energy,
    setup_ies,
)
from dxtb._src.ncoord import coordination_number, erf_count
from dxtb.config import Config

from .samples import samples

DD = {"device": torch.device("cpu"), "dtype": torch.float64}
EXCLUDE = ("scf", "rep", "srb", "hal", "disp")
SETUP_FIELDS = ("numbers", "eeq", "rcov", "cutoff", "cn_max", "cn_kcn")


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    sample = samples["SiH4"]
    return sample["numbers"], sample["positions"].to(**DD)


def _model(par: ParamModule | None = None) -> Model:
    if par is None:
        par = ParamModule(GFN0_XTB.model_copy(deep=True), **DD)
    return Model(par=par, config=Config.create(exclude=EXCLUDE))


def _tensor_leaves(setup: IESCoreSetup) -> list[torch.Tensor]:
    leaves, _ = _pytree.tree_flatten(setup)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def test_ies_core_node_schema_identity_and_roundtrip() -> None:
    """The IES Node preserves all setup references and nested EEQ leaves."""
    numbers, _ = _inputs()
    model_system = _model().setup(numbers)
    assert model_system.ies_classical is not None
    legacy = setup_ies(model_system.ies_classical, numbers)
    core = core_ies_setup(legacy)

    assert isinstance(core, Node)
    assert type(core) is IESCoreSetup
    assert tuple(field.name for field in fields(core)) == SETUP_FIELDS
    assert core.numbers is legacy.numbers
    assert core.eeq is legacy.eeq
    for name in ("rcov", "cutoff", "cn_max", "cn_kcn"):
        assert getattr(core, name) is getattr(legacy, name)

    leaves = _tensor_leaves(core)
    expected = [
        legacy.numbers,
        legacy.eeq.chi,
        legacy.eeq.kcn,
        legacy.eeq.eta,
        legacy.eeq.rad,
        legacy.rcov,
        legacy.cutoff,
        legacy.cn_max,
        legacy.cn_kcn,
    ]
    assert len(leaves) == len(expected) == 9
    assert all(actual is source for actual, source in zip(leaves, expected))
    assert type(core.eeq) is EEQModel

    rebuilt_leaves, spec = _pytree.tree_flatten(core)
    rebuilt = _pytree.tree_unflatten(rebuilt_leaves, spec)
    assert type(rebuilt) is IESCoreSetup
    assert type(rebuilt.eeq) is EEQModel
    assert rebuilt.eeq.cn.count is legacy.eeq.cn.count
    assert rebuilt.eeq.cn.cutoff == legacy.eeq.cn.cutoff
    assert rebuilt.eeq.cn.cn_max == legacy.eeq.cn.cn_max
    assert rebuilt.eeq.cn.rcov is legacy.eeq.cn.rcov
    assert rebuilt.eeq.cn.en is legacy.eeq.cn.en
    assert rebuilt.eeq.cn.pair_weight is legacy.eeq.cn.pair_weight
    for actual, source in zip(_tensor_leaves(rebuilt), expected):
        assert actual is source


def test_ies_core_node_dtype_and_device_conversion() -> None:
    """Node conversions reach nested EEQ tensors and preserve numbers."""
    numbers, _ = _inputs()
    model_system = _model().setup(numbers)
    assert model_system.ies_classical is not None
    legacy = setup_ies(model_system.ies_classical, numbers)
    core = core_ies_setup(legacy)

    for dtype in (torch.float32, torch.float64):
        converted = core.to(dtype=dtype)
        assert type(converted) is IESCoreSetup
        assert converted.numbers.dtype == numbers.dtype
        assert converted.eeq.chi.dtype == dtype
        assert converted.eeq.kcn.dtype == dtype
        assert converted.eeq.eta.dtype == dtype
        assert converted.eeq.rad.dtype == dtype
        for name in ("rcov", "cutoff", "cn_max", "cn_kcn"):
            assert getattr(converted, name).dtype == dtype
    assert core.eeq.chi.dtype == torch.float64
    float32 = core.to(dtype=torch.float32)
    roundtrip_dtype = float32.to(dtype=torch.float64)
    assert roundtrip_dtype.eeq.chi.dtype == torch.float64
    assert roundtrip_dtype.rcov.dtype == torch.float64
    assert roundtrip_dtype.numbers.dtype == numbers.dtype

    meta = core.to(device=torch.device("meta"))
    assert type(meta) is IESCoreSetup
    assert meta.numbers.device.type == "meta"
    assert meta.numbers.dtype == numbers.dtype
    assert type(meta.eeq) is EEQModel
    for tensor in _tensor_leaves(meta):
        assert tensor.device.type == "meta"


def test_ies_core_node_stacks_and_vmapped_probe() -> None:
    """Nested EEQ tensor leaves stack and transform as one IES Node."""
    numbers, _ = _inputs()
    model_system = _model().setup(numbers)
    assert isinstance(model_system.ies_setup, IESCoreSetup)
    setup = model_system.ies_setup
    stacked = stack((setup, setup))
    assert type(stacked) is IESCoreSetup
    assert stacked.numbers.shape[0] == 2
    assert stacked.eeq.chi.shape[0] == 2
    assert stacked.eeq.kcn.shape[0] == 2
    assert stacked.eeq.eta.shape[0] == 2
    assert stacked.eeq.rad.shape[0] == 2
    for original, batched in zip(
        _tensor_leaves(setup), _tensor_leaves(stacked)
    ):
        assert batched.shape == (2, *original.shape)

    def probe(item: IESCoreSetup) -> torch.Tensor:
        return (
            item.rcov.sum()
            + item.cutoff.sum()
            + item.cn_max.sum()
            + item.cn_kcn.sum()
            + item.eeq.chi.sum()
            + item.eeq.kcn.sum()
            + item.eeq.eta.sum()
            + item.eeq.rad.sum()
        )

    transformed = torch.func.vmap(probe)(stacked)
    explicit = torch.stack((probe(setup), probe(setup)))
    torch.testing.assert_close(transformed, explicit)


def test_ies_core_energy_charge_and_parameter_gradient_parity() -> None:
    """Legacy and core setup share IES energy and derivative behavior."""
    numbers, positions = _inputs()
    par = ParamModule(GFN0_XTB.model_copy(deep=True), **DD)
    leaf = dict(par.named_parameters())[
        "parameter_tree.element.Si.eeq_chi.param"
    ]
    leaf.requires_grad_(True)
    model_system = _model(par).setup(numbers)
    assert model_system.ies_classical is not None
    legacy = setup_ies(model_system.ies_classical, numbers)
    core = core_ies_setup(legacy)
    charge = torch.tensor(0.23, **DD, requires_grad=True)

    direction = torch.arange(1, positions.numel() + 1, **DD).reshape_as(
        positions
    )
    legacy_positions = positions.clone().requires_grad_(True)
    legacy_position_energy = ies_energy(
        legacy, legacy_positions, charge.detach()
    ).sum()
    (legacy_position_grad,) = torch.autograd.grad(
        legacy_position_energy, legacy_positions, create_graph=True
    )
    (legacy_mixed_grad,) = torch.autograd.grad(
        (legacy_position_grad * direction).sum(), leaf, retain_graph=True
    )
    core_positions = positions.clone().requires_grad_(True)
    core_position_energy = ies_energy(
        core, core_positions, charge.detach()
    ).sum()
    (core_position_grad,) = torch.autograd.grad(
        core_position_energy, core_positions, create_graph=True
    )
    (core_mixed_grad,) = torch.autograd.grad(
        (core_position_grad * direction).sum(), leaf, retain_graph=True
    )
    torch.testing.assert_close(core_position_grad, legacy_position_grad)
    torch.testing.assert_close(core_mixed_grad, legacy_mixed_grad)
    assert torch.isfinite(core_mixed_grad)

    neutral_legacy = ies_energy(legacy, positions, 0.0).sum()
    neutral_core = ies_energy(core, positions, 0.0).sum()
    torch.testing.assert_close(neutral_core, neutral_legacy)
    legacy_energy = ies_energy(legacy, positions, charge).sum()
    core_energy = ies_energy(core, positions, charge).sum()
    torch.testing.assert_close(core_energy, legacy_energy)
    legacy_charge_grad, legacy_param_grad = torch.autograd.grad(
        legacy_energy, (charge, leaf), retain_graph=True
    )
    core_charge_grad, core_param_grad = torch.autograd.grad(
        core_energy, (charge, leaf)
    )
    torch.testing.assert_close(core_charge_grad, legacy_charge_grad)
    torch.testing.assert_close(core_param_grad, legacy_param_grad)
    assert torch.isfinite(core_charge_grad)
    assert torch.isfinite(core_param_grad).all()
    assert torch.count_nonzero(core_param_grad) > 0


def test_upstream_eeq_optional_solve_modes_match_schur() -> None:
    """Characterize new upstream CG modes without changing IES defaults."""
    numbers, positions = _inputs()
    positions = positions.clone().requires_grad_(True)
    model_system = _model().setup(numbers)
    setup = model_system.ies_setup
    charge = torch.tensor(0.23, **DD)
    cn = coordination_number(
        numbers,
        positions,
        counting_function=erf_count,
        rcov=setup.rcov,
        cutoff=setup.cutoff,
        cn_max=setup.cn_max,
        kcn=setup.cn_kcn,
    )
    structure = Structure(
        numbers=numbers,
        positions=positions,
        charge=charge,
    )

    references = {}
    for mode in ("schur", "cg", "matrix-free"):
        charges, energy = setup.eeq.solve(
            structure,
            cn,
            return_energy=True,
            solve_mode=mode,
        )
        position_gradient = torch.autograd.grad(
            energy.sum(), positions, retain_graph=True
        )[0]
        references[mode] = (charges, energy, position_gradient)

    expected_charges, expected_energy, expected_gradient = references["schur"]
    torch.testing.assert_close(
        expected_charges.sum(), charge, atol=2.0e-12, rtol=2.0e-12
    )
    assert torch.count_nonzero(expected_charges) > 0
    for mode in ("cg", "matrix-free"):
        charges, energy, gradient = references[mode]
        torch.testing.assert_close(
            charges, expected_charges, atol=1.0e-10, rtol=1.0e-10
        )
        torch.testing.assert_close(
            energy, expected_energy, atol=1.0e-10, rtol=1.0e-10
        )
        torch.testing.assert_close(
            gradient, expected_gradient, atol=1.0e-9, rtol=1.0e-9
        )


def test_ies_get_energy_accepts_core_and_rejects_unrelated_cache() -> None:
    """IES.get_energy accepts both supported setup types only."""
    numbers, positions = _inputs()
    model_system = _model().setup(numbers)
    component = model_system.ies_classical
    assert component is not None
    legacy = component.get_cache(numbers)
    core = core_ies_setup(legacy)
    torch.testing.assert_close(
        component.get_energy(positions, core, charge=0.0),
        component.get_energy(positions, legacy, charge=0.0),
    )
    with pytest.raises(TypeError, match="IESSetup.*IESCoreSetup"):
        component.get_energy(positions, object(), charge=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Total molecular charge"):
        component.get_energy(positions, core)


def test_ies_core_singlepoint_coexists_with_scf() -> None:
    """System-owned IES remains a classical term in a normal SCF call."""
    numbers, positions = _inputs()
    par = ParamModule(GFN0_XTB.model_copy(deep=True), **DD)
    model = Model(
        par=par,
        config=Config.create(exclude=("rep", "srb", "hal", "disp")),
    )
    system = model.setup(numbers)
    assert isinstance(system.ies_setup, IESCoreSetup)

    result = system.singlepoint(positions)
    assert torch.isfinite(result.total).all()
    assert torch.isfinite(dict(result.classical)["IES"]).all()


def test_packed_ies_system_keeps_legacy_setup() -> None:
    """Packed legacy systems continue to retain IESSetup values."""
    from tad_mctc.batch import pack

    numbers, positions = _inputs()
    second = samples["LiH"]
    packed_numbers = pack((numbers, second["numbers"]))
    packed_positions = pack((positions, second["positions"].to(**DD)))
    system = _model().setup(packed_numbers)
    assert system.batch_mode == 1
    assert type(system.ies_setup) is IESSetup
    assert system.classical_cache["IES"] is system.ies_setup
    assert system.ies_classical is not None
    energy = system.ies_classical.get_energy(
        packed_positions,
        system.ies_setup,
        charge=torch.tensor([0.0, 0.0], **DD),
    )
    assert energy.shape == (2, packed_numbers.shape[-1])
