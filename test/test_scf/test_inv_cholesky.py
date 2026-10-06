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
"""
The unrolled SCF computes the inverse Cholesky factor of the overlap matrix
once and passes it to the eigensolver, instead of solving with the overlap in
every iteration.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import storch
from tad_mctc.convert import symbol_to_number

from dxtb import GFN1_XTB, Calculator
from dxtb._src.typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_factor_computed_once(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = symbol_to_number("O H H".split())
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.7, 0.0, -0.5]], **dd
    )

    calls = []
    original = storch.inv_cholesky_factor

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(storch, "inv_cholesky_factor", counting)

    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, **dd)
    calc.get_energy(positions.clone().requires_grad_(True))

    assert len(calls) == 1
