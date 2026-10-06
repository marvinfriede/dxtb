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
Padding orbitals in batched SCF calculations must stay empty.

The occupation is masked by orbital position. In the implicit SCF modes, the
eigenvalues of the zero padding (0 Hartree) were sorted between the physical
ones, so an anion (HOMO close to zero) in a padded batch put electrons into
padding orbitals (OH- next to water: 8e-6 Hartree too low with GFN2-xTB).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..molecules import mols

# OH- (bohr)
OH_NUMBERS = [8, 1]
OH_POSITIONS = [[0.0, 0.0, -0.9248098731], [0.0, 0.0, 0.9248098731]]

opts = {
    "verbosity": 0,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
    "int_driver": "pytorch",
}


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
@pytest.mark.parametrize("scf_mode", ["full", "implicit"])
def test_anion_in_padded_batch(par, scf_mode: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = [
        mols["H2O"]["numbers"].to(DEVICE),
        torch.tensor(OH_NUMBERS, device=DEVICE),
    ]
    positions = [
        mols["H2O"]["positions"].to(**dd),
        torch.tensor(OH_POSITIONS, **dd),
    ]
    charges = [torch.tensor(0.0, **dd), torch.tensor(-1.0, **dd)]

    o = dict(opts, scf_mode=scf_mode)
    ref = torch.stack(
        [
            Calculator(n, par, opts=o, **dd).get_energy(p, chrg=c)
            for n, p, c in zip(numbers, positions, charges)
        ]
    )

    calc = Calculator(pack(numbers), par, opts=dict(o, batch_mode=1), **dd)
    energy = calc.get_energy(pack(positions), chrg=torch.stack(charges))

    assert pytest.approx(ref.cpu(), abs=1e-10) == energy.cpu()
