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
Device given as a string (baseline finding): ``Calculator(..., device="cpu")``
raised a ``DeviceError`` because ``"cpu" != torch.device("cpu")``.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data.molecules import mols

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb.components.coulomb import new_es2, new_es3
from dxtb.components.field import new_efield


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_calculator_device_string(par) -> None:
    numbers = mols["H2O"]["numbers"]
    positions = mols["H2O"]["positions"].double()
    opts = {"verbosity": 0, "int_driver": "pytorch"}

    ref = Calculator(
        numbers, par, opts=opts, device=torch.device("cpu"), dtype=torch.double
    ).get_energy(positions)
    calc = Calculator(numbers, par, opts=opts, device="cpu", dtype=torch.double)
    assert calc.device == positions.device
    assert pytest.approx(ref.item(), abs=1e-12) == calc.get_energy(positions)


def test_components_device_string() -> None:
    unique = torch.tensor([1, 8])
    assert new_es2(unique, GFN1_XTB, device="cpu") is not None
    assert new_es3(unique, GFN1_XTB, device="cpu") is not None
    assert new_efield(torch.zeros(3), device="cpu") is not None
