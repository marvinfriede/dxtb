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
"""A batched Calculator sets up no padded System."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals.dispersion import new_dispersion

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.float64}

NUMBERS = torch.tensor([[8, 1, 1, 0], [6, 1, 1, 1]], device=DEVICE)


def _assert_index_helpers_equal(actual, expected) -> None:
    for field in dataclasses.fields(expected):
        a = getattr(actual, field.name)
        b = getattr(expected, field.name)
        if isinstance(b, torch.Tensor):
            assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
@pytest.mark.parametrize("batch_mode", [None, 1])
def test_batched_calculator_matches_padded_setup(par, batch_mode) -> None:
    """Options and IndexHelper equal those of a padded Model.setup."""
    opts = {"verbosity": 0, "batch_mode": batch_mode}
    calc = Calculator(NUMBERS, par, opts=opts, **DD)

    assert calc.system is None
    assert calc.classicals is None and calc.interactions is None

    padded = Model(par=ParamModule(par, **DD), config=calc.model.config).setup(
        NUMBERS, batch_mode=batch_mode
    )
    assert calc.opts == padded.config
    _assert_index_helpers_equal(calc.ihelp, padded.ihelp)


@pytest.mark.parametrize("batched", [False, True])
def test_duplicate_labels_raise_at_construction(batched: bool) -> None:
    """A second D3 with the default label is rejected, batched or not."""
    numbers = NUMBERS if batched else NUMBERS[0, :3]
    par = ParamModule(GFN1_XTB, **DD)
    d3 = new_dispersion(numbers, par, **DD)
    with pytest.raises(ValueError, match="label 'DispersionD3'"):
        Calculator(numbers, par, classical=[d3], opts={"verbosity": 0}, **DD)
