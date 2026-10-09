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
"""Regression tests for Calculator call-history and stale-value behavior.

These cases originate in the T0.5 cache findings. They verify that current
single-system evaluation does not reuse stale integral or result data across
calls. Legacy component-cache cleanup remains scheduled for B6.
"""

from __future__ import annotations

import random

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb.components.field import new_efield

from ...conftest import DEVICE

pytestmark = pytest.mark.cache

# water (bohr)
NUMBERS = [8, 1, 1]
POSITIONS = [
    [-0.00000000000000, -0.00000000000000, -0.74288549752983],
    [-1.43472674945442, -0.00000000000000, +0.37144274876492],
    [+1.43472674945442, +0.00000000000000, +0.37144274876492],
]

drivers = ["pytorch"] + (["libcint"] if has_libcint else [])


def _setup(dd: DD) -> tuple[Tensor, Tensor]:
    numbers = torch.tensor(NUMBERS, device=DEVICE)
    positions = torch.tensor(POSITIONS, **dd)
    return numbers, positions


def _fresh_forces(numbers: Tensor, positions: Tensor, par, opts) -> Tensor:
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    calc = Calculator(numbers, par, opts=opts, **dd)
    pos = positions.detach().clone().requires_grad_(True)
    return calc.get_forces(pos)


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
@pytest.mark.parametrize("driver", drivers)
def test_no_grad_call_then_forces(par, driver: str) -> None:
    """
    Test 1: ``get_energy`` without gradient tracking, then ``get_forces``
    with ``requires_grad`` switched on in place. Previously wrong by up to
    0.1 Eh/bohr; fixed by the stopgap.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)
    opts = {"verbosity": 0, "int_driver": driver}

    calc = Calculator(numbers, par, opts=opts, **dd)
    pos = positions.clone()
    calc.get_energy(pos)
    pos.requires_grad_(True)
    forces = calc.get_forces(pos)

    ref = _fresh_forces(numbers, positions, par, opts)
    assert pytest.approx(ref.cpu(), abs=1e-10) == forces.detach().cpu()


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
@pytest.mark.parametrize("driver", drivers)
def test_new_leaf_same_values(par, driver: str) -> None:
    """
    Variant of test 1: two different leaf tensors with the same values, both
    requiring gradients. The forces with respect to the second must not use
    the graph of the first. Fixed by the stopgap.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)
    opts = {"verbosity": 0, "int_driver": driver}

    calc = Calculator(numbers, par, opts=opts, **dd)
    calc.get_energy(positions.clone().requires_grad_(True))
    forces = calc.get_forces(positions.clone().requires_grad_(True))

    ref = _fresh_forces(numbers, positions, par, opts)
    assert pytest.approx(ref.cpu(), abs=1e-10) == forces.detach().cpu()


@pytest.mark.parametrize("driver", drivers)
def test_forces_twice_same_leaf(driver: str) -> None:
    """Repeated force calls match a fresh Calculator on the same leaf."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)

    opts = {"verbosity": 0, "int_driver": driver}
    calc = Calculator(numbers, GFN1_XTB, opts=opts, **dd)
    pos = positions.clone().requires_grad_(True)
    first = calc.get_forces(pos)
    second = calc.get_forces(pos)
    fresh = _fresh_forces(numbers, positions, GFN1_XTB, opts)

    torch.testing.assert_close(first, fresh, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(second, fresh, atol=1e-10, rtol=0.0)


def test_es2_es3_reused_system_random_call_history() -> None:
    """ES2/ES3 outputs and forces do not depend on evaluation history."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = torch.tensor([1, 1], device=DEVICE)
    positions_a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)
    positions_b = positions_a.clone()
    positions_b[1, 2] += 0.12
    options = {"verbosity": 0}
    operations = (
        "energy_a",
        "singlepoint_a",
        "force_new_a",
        "singlepoint_b",
        "force_same_leaf_a",
        "energy_clone_a",
        "singlepoint_clone_a",
        "backward_then_singlepoint_a",
    )

    def evaluate(calc: Calculator, operation: str, shared_leaf) -> object:
        if operation == "energy_a":
            return calc.get_energy(positions_a)
        if operation == "singlepoint_a":
            return calc.singlepoint(positions_a)
        if operation == "force_new_a":
            return calc.get_forces(shared_leaf)
        if operation == "singlepoint_b":
            return calc.singlepoint(positions_b)
        if operation == "force_same_leaf_a":
            return calc.get_forces(shared_leaf)
        if operation == "energy_clone_a":
            return calc.get_energy(positions_a.clone())
        if operation == "singlepoint_clone_a":
            return calc.singlepoint(positions_a.clone())
        if operation == "backward_then_singlepoint_a":
            leaf = positions_a.clone().requires_grad_(True)
            energy = calc.get_energy(leaf)
            torch.autograd.grad(energy.sum(), leaf)
            return calc.singlepoint(positions_a)
        raise AssertionError(f"Unknown operation: {operation}")

    def assert_same(actual: object, expected: object) -> None:
        if isinstance(actual, torch.Tensor):
            assert isinstance(expected, torch.Tensor)
            torch.testing.assert_close(actual, expected)
            return
        assert hasattr(actual, "energy") and hasattr(expected, "energy")
        for field in (
            "energy",
            "charges",
            "potential",
            "density",
            "hcore",
            "overlap",
        ):
            first_value = getattr(actual, field)
            second_value = getattr(expected, field)
            if field in ("charges", "potential"):
                first_value = first_value.mono
                second_value = second_value.mono
            if first_value is not None:
                torch.testing.assert_close(first_value, second_value)

    expected: dict[str, object] = {}
    for operation in operations:
        reference_calc = Calculator(numbers, GFN1_XTB, opts=options, **dd)
        reference_leaf = positions_a.clone().requires_grad_(True)
        expected[operation] = evaluate(
            reference_calc, operation, reference_leaf
        )

    for seed in (20261010, 42017):
        order = list(operations)
        random.Random(seed).shuffle(order)
        calc = Calculator(numbers, GFN1_XTB, opts=options, **dd)
        shared_leaf = positions_a.clone().requires_grad_(True)
        for operation in order:
            actual = evaluate(calc, operation, shared_leaf)
            assert_same(actual, expected[operation])


def test_numerical_dipole_without_result_cache() -> None:
    """Numerical dipole evaluations use current displaced-field energies."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)
    field = torch.zeros(3, **dd)
    calc = Calculator(
        numbers,
        GFN1_XTB,
        interaction=[new_efield(field)],
        opts={"verbosity": 0},
        **dd,
    )
    value = calc.dipole_numerical(positions)
    assert torch.isfinite(value).all()
    assert value.abs().max() > 0.5


def test_view_of_batch_is_validated_without_result_cache() -> None:
    """A batch view is checked as a single-system input, never reused."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)
    numbers = pack([numbers, torch.tensor([1, 1], device=DEVICE)])
    positions = pack(
        [positions, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)]
    )

    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, **dd)
    calc.get_energy(positions)
    with pytest.raises(ValueError):
        calc.get_energy(positions[0])
