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
"""Result flow through Calculator properties without persistent caching."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, Calculator
from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.typing import Tensor
from dxtb.calculators import AnalyticalCalculator

from ...conftest import DEVICE


def _system(dtype: torch.dtype = torch.float64):
    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=DEVICE,
        dtype=dtype,
    )
    return numbers, positions


def test_same_input_is_recomputed_and_user_result_is_retained() -> None:
    """Repeated calls recompute while a caller-owned Result remains stable."""
    numbers, positions = _system()
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype)
    before = calc._ncalcs

    retained = calc.singlepoint(positions)
    original_energy = retained.energy.clone()
    original_density = retained.density.clone()
    assert calc._ncalcs == before + 1

    repeated = calc.singlepoint(positions)
    assert calc._ncalcs == before + 2
    torch.testing.assert_close(retained.energy, repeated.energy)

    moved = positions.clone()
    moved[1, 2] += 0.1
    changed = calc.singlepoint(moved)
    assert calc._ncalcs == before + 3
    assert not torch.allclose(retained.energy, changed.energy)
    torch.testing.assert_close(retained.energy, original_energy)
    assert retained.density is not None and original_density is not None
    torch.testing.assert_close(retained.density, original_density)


def test_energy_recomputes_for_equal_leaves_and_inplace_geometry() -> None:
    """Energy has no identity or storage-version result key."""
    numbers, positions = _system()
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype)
    before = calc._ncalcs

    energy1 = calc.energy(positions)
    energy2 = calc.energy(positions.clone())
    assert calc._ncalcs == before + 2
    torch.testing.assert_close(energy1, energy2)

    positions[1, 2] += 0.1
    energy3 = calc.energy(positions)
    assert calc._ncalcs == before + 3
    assert not torch.allclose(energy1, energy3)


def test_get_property_returns_local_values_without_history() -> None:
    """get_property selects a value from this calculation's local Result."""
    numbers, positions = _system()
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype)
    before = calc._ncalcs

    density = calc.get_density(positions)
    assert isinstance(density, Tensor)
    assert calc._ncalcs == before + 1
    assert calc.get_property(
        "density", positions, allow_calculation=False
    ) is None

    clone = calc.get_property("density", positions, return_clone=True)
    assert isinstance(clone, Tensor)
    assert calc._ncalcs == before + 2
    assert clone.data_ptr() != density.data_ptr()


def test_calculate_returns_requested_values() -> None:
    """calculate returns requested properties instead of writing cache state."""
    numbers, positions = _system()
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype)
    before = calc._ncalcs

    values = calc.calculate(["energy", "charges"], positions)
    assert set(values) == {"energy", "charges"}
    assert isinstance(values["energy"], Tensor)
    assert hasattr(values["charges"], "mono")
    assert calc._ncalcs == before + 1


def test_bond_orders_use_result_matrices_directly() -> None:
    """Bond orders do not require store_overlap/store_density options."""
    numbers, positions = _system()
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype)
    before = calc._ncalcs
    bond_orders = calc.get_bond_orders(positions)
    assert isinstance(bond_orders, Tensor)
    assert torch.isfinite(bond_orders).all()
    assert calc._ncalcs == before + 1


@pytest.mark.parametrize(
    "option",
    [
        "store_charges",
        "store_coefficients",
        "store_density",
        "store_iterations",
        "store_mo_energies",
        "store_occupation",
        "store_potential",
        "store_fock",
        "store_hcore",
        "store_overlap",
        "store_dipole",
        "store_quadrupole",
    ],
)
def test_removed_store_options_raise(option: str) -> None:
    """Removed result-retention kwargs cannot be silently ignored."""
    numbers, positions = _system()
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype)
    with pytest.raises(TypeError, match="result caching has been removed"):
        calc.energy(positions, **{option: True})


def test_calculator_has_no_result_cache_state() -> None:
    """Neither Calculator nor Config retains result cache state."""
    numbers, positions = _system()
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype)
    assert not hasattr(calc, "cache")
    assert not hasattr(calc.opts, "cache")
    with pytest.raises(TypeError, match="result caching has been removed"):
        Calculator(
            numbers,
            GFN1_XTB,
            opts={"verbosity": 0},
            dtype=positions.dtype,
            cache=True,
        )


@pytest.mark.parametrize("batch_mode", [1, 2])
def test_legacy_batch_modes_recompute_without_result_cache(batch_mode: int) -> None:
    """Both supported legacy batch modes return current values without reuse."""
    numbers, positions = _system()
    batch_numbers = numbers.unsqueeze(0).expand(2, -1).clone()
    batch_positions = positions.unsqueeze(0).expand(2, -1, -1).clone()
    batch_calc = Calculator(
        batch_numbers,
        GFN1_XTB,
        opts={"verbosity": 0, "batch_mode": batch_mode},
        dtype=positions.dtype,
    )
    charges = torch.tensor(
        [0.0, 1.0], device=positions.device, dtype=positions.dtype
    )

    before = batch_calc._ncalcs
    first = batch_calc.singlepoint(batch_positions, chrg=charges)
    second = batch_calc.singlepoint(batch_positions.clone(), chrg=charges)
    assert batch_calc._ncalcs == before + 2
    torch.testing.assert_close(first.energy, second.energy)

    looped = []
    for charge in charges:
        calc = Calculator(
            numbers,
            GFN1_XTB,
            opts={"verbosity": 0},
            dtype=positions.dtype,
        )
        looped.append(calc.singlepoint(positions, chrg=charge).energy)
    torch.testing.assert_close(first.energy, torch.stack(looped))
    assert not torch.isclose(first.energy[0].sum(), first.energy[1].sum())

    assert first.charges is not None
    combined_charges = first.charges + Charges(
        mono=torch.zeros_like(first.charges.mono), batch_mode=batch_mode
    )
    assert combined_charges.batch_mode == batch_mode
    assert combined_charges.axis == first.charges.axis == 1
    assert combined_charges.as_tensor().shape[0] == 2

    assert first.potential is not None
    combined_potential = first.potential + Potential(
        mono=torch.zeros_like(first.potential.mono), batch_mode=batch_mode
    )
    assert combined_potential.batch_mode == batch_mode
    assert combined_potential.axis == first.potential.axis == 1
    assert combined_potential.as_tensor().shape[0] == 2


def test_analytical_forces_use_local_result_data() -> None:
    """Analytical forces work without result cache transport."""
    numbers, positions = _system()
    positions.requires_grad_(True)
    calc = AnalyticalCalculator(
        numbers, GFN1_XTB, opts={"verbosity": 0}, dtype=positions.dtype
    )
    forces = calc.forces_analytical(positions)
    assert torch.isfinite(forces).all()
