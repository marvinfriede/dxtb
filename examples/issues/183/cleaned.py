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
https://github.com/grimme-lab/dxtb/issues/183
"""

import torch

import dxtb
from dxtb.typing import DD

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

opts = {
    "differentiation": "unrolled",
}


def main() -> int:
    par = dxtb.ParamModule(dxtb.GFN1_XTB, **dd)
    par.set_differentiable("charge", "effective", "gexp")
    par.set_differentiable("element", "O", "levels")
    gexp = par.get("charge", "effective", "gexp")
    levels_o = par.get("element", "O", "levels")

    calc = dxtb.Calculator(numbers, par, opts=opts, **dd)

    def get_energy_force(calc: dxtb.Calculator):
        energy = calc.get_energy(positions)
        forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        return energy, forces

    # AD gradient w.r.t. params
    energy, force = get_energy_force(calc)
    de_dparam = torch.autograd.grad(energy, (gexp, levels_o), retain_graph=True)
    dfnorm_dparam = torch.autograd.grad(torch.norm(force), (gexp, levels_o))

    # Numerical gradient w.r.t. params
    dparam = 2e-6
    with torch.no_grad():
        orig_gexp = gexp.item()
        gexp.fill_(orig_gexp + dparam / 2)
    energy1, force1 = get_energy_force(calc)

    with torch.no_grad():
        gexp.fill_(orig_gexp - dparam / 2)
    energy2, force2 = get_energy_force(calc)
    with torch.no_grad():
        gexp.fill_(orig_gexp)

    de_dgexp = (energy1 - energy2) / dparam

    print(f"dE / dgexp (AD)  = {de_dparam[0]: .8f}")
    print(f"dE / dgexp (Num) = {de_dgexp: .8f}")

    dF_dgexp = (torch.norm(force1) - torch.norm(force2)) / dparam
    print(f"d|F| / dgexp (AD)  = {dfnorm_dparam[0]: .8f}")
    print(f"d|F| / dgexp (Num) = {dF_dgexp: .8f}")

    with torch.no_grad():
        orig_level = levels_o[0].item()
        levels_o[0] = orig_level + dparam / 2
    energy1, force1 = get_energy_force(calc)

    with torch.no_grad():
        levels_o[0] = orig_level - dparam / 2
    energy2, force2 = get_energy_force(calc)
    with torch.no_grad():
        levels_o[0] = orig_level

    de_dp = (energy1 - energy2) / dparam
    print(f"dE / dlevels_O[0] (AD)   = {de_dparam[1][0]: .8f}")
    print(f"dE / dlevels_O[0] (Num)  = {de_dp: .8f}")

    df_dp = (torch.norm(force1) - torch.norm(force2)) / dparam
    print(f"d|F| / dlevels_O[0] (AD)  = {dfnorm_dparam[1][0]: .8f}")
    print(f"d|F| / dlevels_O[0] (Num) = {df_dp: .8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
