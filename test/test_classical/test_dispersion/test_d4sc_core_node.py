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
"""Tests for the System-owned D4SC Node variable."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from tad_mctc.tree import child, Node, stack
from torch.utils import _pytree

from dxtb import GFN2_XTB, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.classicals.dispersion import _d4_adapter as d4
from dxtb._src.components.interactions import InteractionList
from dxtb._src.components.interactions.dispersion.d4sc import (
    D4SCSetup,
    DispersionD4SC,
    build_d4sc_cache,
    new_d4sc,
    setup_d4sc,
)
from dxtb._src.typing import Tensor
from dxtb.config import Config

from ...utils import scf_strategy
from .samples import samples

DEVICE = torch.device("cpu")
DD = {"device": DEVICE, "dtype": torch.double}


def _sample(name: str = "LiH") -> tuple[Tensor, Tensor]:
    sample = samples[name]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**DD)


def _interaction(numbers: Tensor) -> DispersionD4SC:
    par = ParamModule(GFN2_XTB, **DD)
    interaction = new_d4sc(numbers, par, **DD)
    assert interaction is not None
    return interaction


def _model() -> Model:
    return Model(
        par=ParamModule(GFN2_XTB, **DD),
        config=Config.create(exclude=("es2", "es3", "aes2")),
        auto_int_level=False,
    )


def _setup_tensors(setup: D4SCSetup) -> list[Tensor]:
    return [
        setup.numbers,
        *setup.param.values(),
        setup.rcov,
        setup.r4r2,
        setup.cn_cutoff,
        setup.pair_weight,
        setup.model.numbers,
        setup.model.wf,
        setup.model.rc6,
    ]


def test_setup_roundtrip_has_complete_dynamic_tensor_leaves() -> None:
    """The D4SC setup round-trips every tensor as one dynamic leaf."""
    numbers, _ = _sample()
    setup = setup_d4sc(_interaction(numbers), numbers)

    assert isinstance(setup, Node)
    assert [field.name for field in fields(setup)] == [
        "numbers",
        "param",
        "rcov",
        "r4r2",
        "cn_cutoff",
        "pair_weight",
        "model",
        "counting_function",
    ]

    leaves, spec = _pytree.tree_flatten(setup)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is D4SCSetup
    assert all(isinstance(leaf, Tensor) for leaf in leaves)
    assert len(leaves) == len(_setup_tensors(setup))
    assert all(new is old for new, old in zip(leaves, _setup_tensors(setup)))
    rendered_paths = [
        str(path) for path, _ in _pytree.tree_flatten_with_path(setup)[0]
    ]
    assert any("param" in path for path in rendered_paths)
    assert any("model" in path for path in rendered_paths)
    assert rebuilt.counting_function is setup.counting_function
    assert type(rebuilt.model) is type(setup.model)
    assert rebuilt.model.ga == setup.model.ga
    assert setup.to() is setup


def test_sparse_parameters_are_not_filled() -> None:
    """Sparse D4 parameter mappings retain their exact ordered keys."""
    numbers, _ = _sample()
    setup = setup_d4sc(_interaction(numbers), numbers)
    for omitted in ("s9", "s10"):
        param = {k: v for k, v in setup.param.items() if k != omitted}
        sparse = setup.replace(param=param)
        assert tuple(sparse.param) == tuple(param)


def test_setup_conversion_and_compatible_stack_vmap() -> None:
    """Setup values convert dtype/device and map over matching structures."""
    numbers_a, _ = _sample("LiH")
    first = setup_d4sc(_interaction(numbers_a), numbers_a)
    second = first.replace(numbers=torch.tensor([11, 1]))

    converted = first.to(dtype=torch.float32)
    assert converted.numbers.dtype == first.numbers.dtype
    assert converted.rcov.dtype == torch.float32
    assert converted.model.rc6.dtype == torch.float32
    assert all(v.dtype == torch.float32 for v in converted.param.values())
    assert converted.counting_function is first.counting_function
    meta = first.to(device="meta")
    assert meta.numbers.device.type == "meta"
    assert meta.model.rc6.device.type == "meta"
    assert all(v.device.type == "meta" for v in meta.param.values())
    with pytest.raises(FrozenInstanceError):
        first.rcov = torch.ones_like(first.rcov)  # type: ignore[misc]

    stacked = stack((first, second))
    assert type(stacked) is D4SCSetup
    assert stacked.numbers.shape[0] == 2
    assert stacked.rcov.shape[0] == 2
    torch.testing.assert_close(
        torch.func.vmap(lambda node: node.rcov.sum())(stacked),
        torch.stack((first.rcov.sum(), second.rcov.sum())),
    )


def test_system_owns_core_node_and_core_builder_matches_legacy() -> None:
    """Model setup stores one Node and both setup forms share D4 physics."""
    numbers, positions = _sample()
    system = _model().setup(numbers)
    assert type(system.d4sc_setup) is D4SCSetup
    assert system.d4sc_interaction is not None
    setup = system.d4sc_setup
    assert setup is not None
    legacy = setup_d4sc(system.d4sc_interaction, numbers)
    assert setup.param.keys() == legacy.param.keys()
    assert type(setup.model) is type(legacy.model)
    legacy_data = build_d4sc_cache(legacy, positions)
    core_data = build_d4sc_cache(setup, positions)
    torch.testing.assert_close(core_data.cn, legacy_data.cn)
    torch.testing.assert_close(core_data.dispmat, legacy_data.dispmat)
    torch.testing.assert_close(core_data.model.rc6, legacy_data.model.rc6)
    q = torch.tensor([0.1, -0.1], **DD)
    torch.testing.assert_close(
        system.d4sc_interaction.get_monopole_atom_energy(core_data, q),
        system.d4sc_interaction.get_monopole_atom_energy(legacy_data, q),
    )

    before = system.singlepoint(positions)
    after = system.singlepoint(positions)
    torch.testing.assert_close(after.energy, before.energy)
    torch.testing.assert_close(after.charges.mono, before.charges.mono)


def test_implicit_d4sc_uses_system_setup_and_preserves_gradients() -> None:
    """Implicit SCF uses System D4SC setup and differentiates custom D4Model state."""
    numbers, base_positions = _sample()

    class ScaledD4Model(d4.model.D4Model):
        scale: Tensor = child(default=None)

        def weight_references(self, *args, **kwargs):
            weights = super().weight_references(*args, **kwargs)
            if isinstance(weights, tuple):
                return tuple(self.scale * value for value in weights)
            return self.scale * weights

    par = ParamModule(GFN2_XTB, **DD)
    interaction = new_d4sc(numbers, par, **DD)
    assert interaction is not None
    original_model = interaction.model
    custom_model = ScaledD4Model.create(
        numbers,
        ga=original_model.ga,
        gc=original_model.gc,
        wf=original_model.wf,
        ref_charges=original_model.ref_charges,
        rc6=original_model.rc6,
        **DD,
    )
    source_scale = torch.tensor(1.07, **DD, requires_grad=True)
    custom_model = custom_model.replace(scale=source_scale)
    interaction = interaction.replace(model=custom_model)

    model = Model(
        par=par,
        config=Config.create(
            exclude=("es2", "es3", "aes2", "d4sc"),
            differentiation="implicit",
            strategy=scf_strategy("linear", "charge"),
            maxiter=80,
            x_atol=1.0e-11,
            x_atol_max=1.0e-11,
        ),
        interaction=(interaction,),
        auto_int_level=False,
    )
    system = model.setup(numbers)

    positions = base_positions.clone().requires_grad_(True)
    result = system.singlepoint(positions)
    pos_grad, scale_grad = torch.autograd.grad(
        result.energy.sum(), (positions, source_scale)
    )
    assert torch.isfinite(pos_grad).all()
    assert torch.isfinite(scale_grad)
    assert scale_grad.abs() > 0


def test_independent_systems_and_alternate_d4sc_setup_authority() -> None:
    """Independent systems keep distinct setups, an alternate D4SC uses its own route, and subclasses are rejected."""
    numbers, positions = _sample()
    model_a = _model()
    par_b = GFN2_XTB.model_copy(deep=True)
    par_b.dispersion.d4.s8 = 2.5  # type: ignore[union-attr]
    model_b = Model(
        par=ParamModule(par_b, **DD),
        config=Config.create(exclude=("es2", "es3", "aes2")),
        auto_int_level=False,
    )
    system_a = model_a.setup(numbers)
    system_b = model_b.setup(numbers)
    assert system_a.d4sc_setup is not system_b.d4sc_setup
    assert not torch.allclose(
        system_a.singlepoint(positions).energy,
        system_b.singlepoint(positions).energy,
    )

    alternate = _interaction(numbers)
    alt_list = InteractionList(alternate)
    alt_data = _interaction_data(system_a, positions, alt_list)
    assert alt_data.d4sc_setup is None
    assert torch.allclose(
        alt_data[alternate.label].model.wf, alternate.model.wf
    )

    class CustomD4SC(DispersionD4SC):
        pass

    subclass = CustomD4SC(
        param=alternate.param,
        model=alternate.model,
        rcov=alternate.rcov,
        r4r2=alternate.r4r2,
        cutoff=alternate.cutoff,
    ).to(**DD)
    sub_system = Model(
        par=model_a.par,
        config=Config.create(exclude=("es2", "es3", "aes2", "d4sc")),
        interaction=(subclass,),
        auto_int_level=False,
    ).setup(numbers)
    assert sub_system.d4sc_setup is None
    assert sub_system.d4sc_interaction is None
    sub_data = _interaction_data(sub_system, positions)
    assert sub_data.d4sc_setup is None
    assert sub_data.scf_kinds == {}
