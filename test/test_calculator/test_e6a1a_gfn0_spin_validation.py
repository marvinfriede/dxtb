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
"""Regression tests for GFN0 non-self-consistent spin validation."""

from __future__ import annotations

import re

import pytest
import torch

from dxtb import GFN0_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.wavefunction.filling import get_alpha_beta_occupation
from dxtb.config import Config


@pytest.fixture(name="gfn0_h2")
def fixture_gfn0_h2():
    """Build an independent production PyTorch GFN0/H2 System."""
    dtype = torch.float64
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=dtype)
    config = Config.create(
        method=labels.GFN0_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
    )
    system = Model(par=ParamModule(GFN0_XTB, dtype=dtype), config=config).setup(
        numbers
    )
    return numbers, positions, system


def test_occupation_helper_validation_is_explicit() -> None:
    """Only the transform caller opts out of eager spin-value validation."""
    electrons = torch.tensor([2.0])
    spin = torch.tensor([1.0])

    with pytest.raises(ValueError, match="unpaired electrons"):
        get_alpha_beta_occupation(electrons, spin)

    relaxed = get_alpha_beta_occupation(electrons, spin, validate=False)
    torch.testing.assert_close(relaxed, torch.tensor([1.0, 1.0]))


def test_gfn0_nonselfconsistent_accepts_consistent_explicit_spin(
    gfn0_h2,
) -> None:
    """The physically consistent singlet retains its one-shot result."""
    _, positions, system = gfn0_h2

    result = system.singlepoint(positions, chrg=0.0, spin=0.0)

    assert torch.isfinite(result.energy).all()
    assert torch.isfinite(result.occupation).all()
    assert result.iterations == 0


@pytest.mark.parametrize(
    ("charge", "spin", "message"),
    [
        (0.0, 1.0, "Odd (even) number of unpaired electrons"),
        (1.0, 2.0, "Number of unpaired electrons"),
    ],
    ids=["spin-electron-parity", "too-many-unpaired-electrons"],
)
def test_gfn0_nonselfconsistent_rejects_inconsistent_explicit_spin(
    gfn0_h2, charge: float, spin: float, message: str
) -> None:
    """GFN0 keeps the established parity and electron-count checks."""
    _, positions, system = gfn0_h2

    with pytest.raises(ValueError, match=re.escape(message)):
        system.singlepoint(positions, chrg=charge, spin=spin)


def test_gfn0_calculator_keeps_eager_spin_validation(gfn0_h2) -> None:
    """The eager Calculator continues to reject parity-inconsistent spin."""
    numbers, positions, _ = gfn0_h2
    calculator = Calculator(
        numbers,
        GFN0_XTB,
        dtype=positions.dtype,
        opts={
            "verbosity": 0,
            "int_driver": labels.INTDRIVER_PYTORCH,
            "int_level": labels.INTLEVEL_HCORE,
        },
    )

    with pytest.raises(
        ValueError, match=r"Odd \(even\) number of unpaired electrons"
    ):
        calculator.singlepoint(positions, chrg=0.0, spin=1.0)
