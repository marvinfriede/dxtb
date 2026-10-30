#!/usr/bin/env python3
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
https://github.com/grimme-lab/dxtb/issues/187
"""

import torch

import dxtb
from dxtb.typing import DD

############################################
# Setup
############################################

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

numbers = torch.tensor([8, 1, 1], device=dd["device"])
positions = torch.tensor(
    [
        [-2.95915993, 1.40005084, 0.24966306],
        [-2.1362031, 1.4795743, -1.38758999],
        [-2.40235213, 2.84218589, 1.24419946],
    ],
    requires_grad=True,
    **dd,
)

opts = {"verbosity": 0}


############################################
# Minimization
############################################


def main() -> int:
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=opts, **dd)
    assert calc.integrals.hcore is not None

    minpos = positions.detach().clone().requires_grad_(True)
    for iteration in range(200):
        energy = calc.get_energy(minpos)
        (gradient,) = torch.autograd.grad(energy, minpos)
        with torch.no_grad():
            minpos -= 1e-2 * gradient
        minpos.requires_grad_(True)
        if iteration % 20 == 0:
            print(f"step {iteration:3d}: E = {energy.item():.8f} Eh")

    print("\nInitial geometry:")
    print(positions.detach().numpy())

    print("Optimized geometry:")
    print(minpos.detach().numpy())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
