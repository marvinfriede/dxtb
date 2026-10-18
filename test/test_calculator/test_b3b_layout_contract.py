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
"""B3b final-layout ownership and transform boundary probes."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.exceptions import DtypeError
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, labels
from dxtb._src.basis.indexhelper import IndexHelper
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data, singlepoint
from dxtb._src.integral.driver.pytorch.overlap import build_overlap
from dxtb._src.param import ParamModule
from dxtb._src.xtb.h0 import H0IndexNode
from dxtb.config import Config


def _model(dtype: torch.dtype = torch.float64) -> Model:
    config = Config.create(int_driver=labels.INTDRIVER_PYTORCH)
    return Model(par=ParamModule(GFN1_XTB, dtype=dtype), config=config)


def test_single_and_packed_modes_keep_separate_ownership() -> None:
    """Only exact single Systems own core setup Nodes and a derived view."""
    numbers = torch.tensor([1, 1])
    model = _model()

    single = model.setup(numbers, batch_mode=0)
    assert single.batch_mode == 0
    assert single._packed_classical_cache is None
    assert single.repulsion_setup is not None
    assert type(single.repulsion_setup).__name__ == "RepulsionCoreSetup"
    assert single.classical_cache is not single.classical_cache
    assert single.classical_cache["Repulsion"] is single.repulsion_setup

    packed_numbers = torch.tensor([[1, 1], [1, 1]])
    for mode in (1, 2):
        packed = model.setup(packed_numbers, batch_mode=mode)
        assert packed.batch_mode == mode
        assert packed._packed_classical_cache is not None
        assert packed.classical_cache is packed._packed_classical_cache
        assert type(packed.repulsion_setup).__name__ == "RepulsionSetup"
        assert packed.integral_setup is None


def test_effective_dtype_fallback_preserves_explicit_setup_override() -> None:
    """No-integral setups still validate against the effective setup dtype."""
    config = Config.create(
        method=labels.GFN1_XTB,
        exclude=("all",),
        int_level=0,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=config,
        auto_int_level=False,
    ).setup(
        torch.tensor([1, 1]),
        dd={"device": torch.device("cpu"), "dtype": torch.float32},
    )
    assert system.h0_setup is None
    assert system.integral_setup is None
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float32
    )
    result = system.singlepoint(positions)
    assert result.energy.dtype == torch.float32
    with pytest.raises(DtypeError, match="Dtype mismatch"):
        system.singlepoint(positions.to(torch.float64))


def test_captured_integral_maps_are_not_rebuilt_during_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pure single-System path consumes its captured integral setup."""
    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = _model().setup(numbers)
    setup = system.integral_setup
    assert setup is not None and setup.pytorch is not None
    assert type(setup.pytorch).__name__ == "PytorchIntegralSetupNode"
    assert not hasattr(setup.pytorch, "ihelp")
    assert isinstance(system.h0_setup.ihelp, H0IndexNode)
    assert not isinstance(system.h0_setup.ihelp, IndexHelper)
    captured_shell_map = setup.pytorch.index.shells_to_atom
    legacy_shell_map = system.ihelp.shells_to_atom
    assert captured_shell_map is not legacy_shell_map
    captured_snapshot = captured_shell_map.clone()
    legacy_snapshot = legacy_shell_map.clone()
    try:
        with torch.no_grad():
            legacy_shell_map[0] = legacy_shell_map[0] + 1
        assert torch.equal(captured_shell_map, captured_snapshot)
    finally:
        with torch.no_grad():
            legacy_shell_map.copy_(legacy_snapshot)

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("IndexHelper construction ran during evaluation")

    monkeypatch.setattr(IndexHelper, "from_numbers", fail_rebuild)
    result = singlepoint(system, positions)
    assert torch.isfinite(result.energy).all()


def test_overlap_history_transforms_and_parameter_graph() -> None:
    """Captured setup supports A/B/A reuse and the established pure transforms."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    par.set_differentiable("element.H.slater")
    numbers = torch.tensor([1, 1])
    system = Model(
        par=par,
        config=Config.create(int_driver=labels.INTDRIVER_PYTORCH),
    ).setup(numbers)
    assert system.integral_setup is not None
    setup = system.integral_setup.pytorch
    assert setup is not None
    parameter = par.get("element.H.slater")

    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=torch.float64)
    evaluate = lambda geometry: build_overlap(setup, geometry)
    overlap_a = evaluate(a)
    evaluate(b)
    overlap_a_after = evaluate(a)
    torch.testing.assert_close(overlap_a_after, overlap_a)

    direction = torch.ones_like(a) * 0.1
    tangent = jvp(evaluate, (a,), (direction,))[1]
    forward = jacfwd(evaluate)(a)
    reverse = jacrev(evaluate)(a)
    torch.testing.assert_close(forward, reverse)
    torch.testing.assert_close(
        tangent, torch.einsum("ijab,ab->ij", forward, direction)
    )
    batched = vmap(evaluate)(torch.stack((a, b)))
    torch.testing.assert_close(batched, torch.stack((evaluate(a), evaluate(b))))

    value = evaluate(a).square().sum()
    (parameter_gradient,) = torch.autograd.grad(value, parameter)
    assert torch.isfinite(parameter_gradient).all()
    assert torch.count_nonzero(parameter_gradient) > 0


def test_singlepoint_geometry_history_preserves_core_outputs_and_gradients() -> (
    None
):
    """System evaluation remains independent of an intervening geometry."""
    numbers = torch.tensor([1, 1])
    system = _model().setup(numbers)
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=torch.float64)

    first = system.singlepoint(a)
    system.singlepoint(b)
    last = system.singlepoint(a)
    torch.testing.assert_close(first.energy, last.energy)
    torch.testing.assert_close(first.overlap, last.overlap)
    torch.testing.assert_close(first.hcore, last.hcore)
    assert tuple(name for name, _ in first.classical) == tuple(
        name for name, _ in last.classical
    )
    for (_, left), (_, right) in zip(first.classical, last.classical):
        torch.testing.assert_close(left, right)
    assert first.charges is not None and last.charges is not None
    assert first.potential is not None and last.potential is not None
    for name in ("mono", "dipole", "quadrupole"):
        left = getattr(first.charges, name)
        right = getattr(last.charges, name)
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right)
        left = getattr(first.potential, name)
        right = getattr(last.potential, name)
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right)

    positions_first = a.clone().requires_grad_(True)
    positions_last = a.clone().requires_grad_(True)
    energy_first = system.singlepoint(positions_first).energy
    energy_last = system.singlepoint(positions_last).energy
    (gradient_first,) = torch.autograd.grad(energy_first.sum(), positions_first)
    (gradient_last,) = torch.autograd.grad(energy_last.sum(), positions_last)
    torch.testing.assert_close(gradient_first, gradient_last)


def test_call_local_interaction_data_is_geometry_history_independent() -> None:
    """The ES2 matrix cache is rebuilt from setup and current geometry."""
    config = Config.create(
        int_driver=labels.INTDRIVER_PYTORCH,
        exclude=("d4sc",),
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64), config=config
    ).setup(torch.tensor([1, 1]))
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=torch.float64)
    first = _interaction_data(system, a)
    _interaction_data(system, b)
    last = _interaction_data(system, a)
    assert tuple(first) == tuple(last)
    for label in first:
        fields = tuple(
            name for name in ("mat", "hd") if hasattr(first[label], name)
        )
        assert fields
        for name in fields:
            torch.testing.assert_close(
                getattr(first[label], name), getattr(last[label], name)
            )
