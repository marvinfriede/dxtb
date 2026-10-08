# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""Core singlepoint and immutable Result tests."""

from __future__ import annotations

import inspect
import sys
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.config import Config
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.result import Result
from dxtb._src.calculators.singlepoint import singlepoint


def _assert_result_values_equal(left: Result, right: Result) -> None:
    """Compare stable tensor outputs without relying on object identity."""
    for name in (
        "energy",
        "scf",
        "fenergy",
        "density",
        "coefficients",
        "emo",
        "occupation",
        "hamiltonian",
        "overlap",
        "hcore",
        "dipole_integrals",
        "quadrupole_integrals",
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            torch.testing.assert_close(left_value, right_value)

    assert left.charges is not None and right.charges is not None
    assert left.potential is not None and right.potential is not None
    for name in ("mono", "dipole", "quad"):
        left_value = getattr(left.charges, name)
        right_value = getattr(right.charges, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            torch.testing.assert_close(left_value, right_value)
        left_value = getattr(left.potential, name)
        right_value = getattr(right.potential, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            torch.testing.assert_close(left_value, right_value)

    assert tuple(name for name, _ in left.classical) == tuple(
        name for name, _ in right.classical
    )
    for (_, left_value), (_, right_value) in zip(
        left.classical, right.classical
    ):
        torch.testing.assert_close(left_value, right_value)
    assert torch.equal(left.iterations, right.iterations)


def test_core_singlepoint_without_calculator_is_history_independent() -> None:
    """Direct core calls are independent of call order and tensor identity."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    numbers = torch.tensor([1, 1])
    model = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype))
    system = model.setup(numbers)

    first = singlepoint(system, positions)
    same_tensor = singlepoint(system, positions)
    _assert_result_values_equal(first, same_tensor)
    snapshot = {
        name: None if getattr(first, name) is None else getattr(first, name).clone()
        for name in (
            "energy",
            "scf",
            "fenergy",
            "density",
            "coefficients",
            "emo",
            "occupation",
            "hamiltonian",
            "overlap",
            "hcore",
            "dipole_integrals",
            "quadrupole_integrals",
        )
    }
    assert first.charges is not None and first.potential is not None
    nested_snapshot = {
        "charges": {
            name: None
            if getattr(first.charges, name) is None
            else getattr(first.charges, name).clone()
            for name in ("mono", "dipole", "quad")
        },
        "potential": {
            name: None
            if getattr(first.potential, name) is None
            else getattr(first.potential, name).clone()
            for name in ("mono", "dipole", "quad")
        },
        "classical": {
            name: value.clone() for name, value in first.classical
        },
    }
    moved = positions.clone()
    moved[1, 0] += 0.1
    second = singlepoint(system, moved)
    repeated = singlepoint(system, positions.clone())

    assert not torch.allclose(first.energy, second.energy)
    _assert_result_values_equal(first, repeated)
    for name, expected in snapshot.items():
        value = getattr(first, name)
        if expected is not None:
            assert value is not None
            torch.testing.assert_close(value, expected)
    for container_name in ("charges", "potential"):
        container = getattr(first, container_name)
        assert container is not None
        for name, expected in nested_snapshot[container_name].items():
            value = getattr(container, name)
            if expected is None:
                assert value is None
            else:
                assert value is not None
                torch.testing.assert_close(value, expected)
    for name, expected in nested_snapshot["classical"].items():
        torch.testing.assert_close(first.cenergies[name], expected)
    assert first.charges is not second.charges
    assert first.potential is not second.potential
    with pytest.raises(FrozenInstanceError):
        first.energy = second.energy  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first.charges.mono = second.charges.mono  # type: ignore[union-attr,misc]

    assert first.energy is first.total
    assert first.integrals is not None
    assert first.integrals.hcore is not None
    assert {field.name for field in fields(first)} >= {
        "energy",
        "scf",
        "classical",
        "fenergy",
        "charges",
        "multipoles",
        "density",
        "coefficients",
        "emo",
        "occupation",
        "potential",
        "hamiltonian",
        "overlap",
        "hcore",
        "dipole_integrals",
        "quadrupole_integrals",
        "iterations",
        "converged",
        "residual",
    }


def test_core_singlepoint_supports_independent_position_gradients() -> None:
    """One setup supports repeated position derivatives without graph reuse."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )

    gradients = []
    for geometry in (positions, positions + positions.new_tensor([0.01, 0, 0])):
        leaf = geometry.clone().requires_grad_(True)
        result = singlepoint(system, leaf)
        (gradient,) = torch.autograd.grad(result.energy.sum(), leaf)
        assert torch.isfinite(gradient).all()
        gradients.append(gradient)

    repeated = positions.clone().requires_grad_(True)
    result = singlepoint(system, repeated)
    (repeated_gradient,) = torch.autograd.grad(result.energy.sum(), repeated)
    torch.testing.assert_close(repeated_gradient, gradients[0])


def test_system_singlepoint_delegates_to_core() -> None:
    """System exposes a thin direct delegate to the singlepoint core."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )
    result = system.singlepoint(positions)
    direct = singlepoint(system, positions)
    _assert_result_values_equal(result, direct)
    source = inspect.getsource(sys.modules[singlepoint.__module__])
    assert "tensor_id" not in source
    assert "data_ptr" not in source
    assert "id(" not in source


def test_core_parameter_training_rebuilds_system_per_loss() -> None:
    """Parameter gradients use a fresh setup for each differentiated loss."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    numbers = torch.tensor([1, 1])
    model = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype))
    parameter = dict(model.par.named_parameters())[
        "parameter_tree.element.H.slater.param"
    ]
    parameter.requires_grad_(True)

    gradients = []
    for _ in range(2):
        system = model.setup(numbers)
        result = singlepoint(system, positions)
        (gradient,) = torch.autograd.grad(result.energy.sum(), parameter)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        gradients.append(gradient)
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.parametrize("exclude", [("scf",), ("all",)])
def test_core_singlepoint_early_exclusion_returns_complete_result(
    exclude: tuple[str, ...],
) -> None:
    """SCF exclusion paths return a complete immutable Result value."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(method=labels.GFN1_XTB, exclude=exclude)
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    )
    result = singlepoint(model.setup(torch.tensor([1, 1])), positions)
    assert result.scf is not None
    assert torch.equal(result.scf, torch.zeros_like(result.scf))
    assert result.integrals is None
    assert result.charges is None
    assert result.potential is None
    assert torch.isfinite(result.energy).all()


def test_core_singlepoint_rejects_missing_hcore_level() -> None:
    """The core preserves the established minimum integral-level error."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN1_XTB, int_level=labels.INTLEVEL_OVERLAP
    )
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype),
        config=config,
        auto_int_level=False,
    )
    system = model.setup(torch.tensor([1, 1]))
    with pytest.raises(NotImplementedError, match="Core Hamiltonian missing"):
        singlepoint(system, positions)


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 3), (3, 3)],
    ids=["batched-positions", "wrong-atom-count"],
)
def test_core_singlepoint_rejects_non_single_system_positions(
    shape: tuple[int, ...],
) -> None:
    """The core does not infer batching from a positions tensor."""
    positions = torch.zeros(shape, dtype=torch.float64)
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )
    with pytest.raises(ValueError, match=r"shape \(nat, 3\)"):
        singlepoint(system, positions)
