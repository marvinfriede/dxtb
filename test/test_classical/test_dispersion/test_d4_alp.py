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
The three-body exponent ``dispersion.d4.alp`` must reach the D4 energy.

Before, the D4 factory did not pass it to tad-dftd4, which then used its
built-in default (equal to the GFN2-xTB value): the energy was right, but the
parameter had no effect and no gradient (parameter-gradient coverage, T0.9).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data.molecules import mols

from dxtb import GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.components.classicals import new_dispersion


def _energy(par: ParamModule, numbers, positions) -> torch.Tensor:
    disp = new_dispersion(
        numbers,
        par,
        charge=torch.tensor(0.0, dtype=torch.double),
        dtype=torch.double,
    )
    assert disp is not None
    cache = disp.get_cache(numbers, IndexHelper.from_numbers(numbers, par))
    return disp.get_energy(positions, cache).sum()


def test_alp_reaches_energy() -> None:
    numbers = mols["CH4"]["numbers"]
    positions = mols["CH4"]["positions"].double()

    par = ParamModule(GFN2_XTB, dtype=torch.double)
    alp = par.get("dispersion.d4.alp")
    alp.requires_grad_(True)
    (grad,) = torch.autograd.grad(_energy(par, numbers, positions), alp)

    def energy_at(value: float) -> float:
        p = ParamModule(GFN2_XTB, dtype=torch.double)
        with torch.no_grad():
            p.get("dispersion.d4.alp").fill_(value)
            return _energy(p, numbers, positions).item()

    h = 1e-3
    fd = (energy_at(16.0 + h) - energy_at(16.0 - h)) / (2 * h)
    assert fd != 0.0
    assert pytest.approx(fd, rel=1e-5) == grad.item()
