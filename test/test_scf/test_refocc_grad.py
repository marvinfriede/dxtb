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
Derivative of the energy with respect to the reference occupations.

The reference occupation (``element.<X>.refocc``) enters twice: through the
reference populations of the charges and through the number of electrons,
which carries its graph since the electron count is not rounded anymore. For
a molecule with a large HOMO-LUMO gap (water: about 300 kT at 300 K), the
derivative with respect to the number of electrons used to be dropped by the
Fermi occupation, and the derivative with respect to ``refocc`` missed the
second part (water, GFN1: -0.0036 instead of -0.2259 for hydrogen).
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..molecules import mols

PARAMS = {"gfn1": GFN1_XTB, "gfn2": GFN2_XTB}

# central differences of the leaves, error of the step (measured) is 3e-5
# relative at most
STEP = 1e-5
RTOL = 1e-4
ATOL = 1e-7


def _energy(
    method: str, scf_mode: str, symbol: str, value: torch.Tensor, dd: DD
) -> torch.Tensor:
    mol = mols["H2O"]
    numbers = mol["numbers"].to(DEVICE)
    positions = mol["positions"].to(**dd)

    par = ParamModule(PARAMS[method], **dd)
    leaf = par.get_submodule(f"parameter_tree.element.{symbol}.refocc")
    del leaf._parameters["param"]  # pylint: disable=protected-access
    leaf.param = value

    opts = {"scf_mode": scf_mode, "int_driver": "pytorch"}
    calc = Calculator(numbers, par, opts=opts, **dd)
    return calc.energy(positions)


@pytest.mark.parametrize("method", ["gfn1", "gfn2"])
@pytest.mark.parametrize("scf_mode", ["full", "implicit"])
@pytest.mark.parametrize("symbol", ["H", "O"])
def test_refocc_gradient(method: str, scf_mode: str, symbol: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    base = ParamModule(PARAMS[method], **dd)
    ref = dict(base.named_parameters())[
        f"parameter_tree.element.{symbol}.refocc.param"
    ].detach()
    direction = torch.linspace(0.3, 1.0, ref.numel(), **dd).reshape(ref.shape)

    x = ref.clone().requires_grad_(True)
    energy = _energy(method, scf_mode, symbol, x, dd)
    (grad,) = torch.autograd.grad(energy, x)
    autograd = (grad * direction).sum()

    with torch.no_grad():
        plus = _energy(method, scf_mode, symbol, ref + STEP * direction, dd)
        minus = _energy(method, scf_mode, symbol, ref - STEP * direction, dd)
    numerical = (plus - minus) / (2 * STEP)

    assert pytest.approx(numerical.item(), rel=RTOL, abs=ATOL) == autograd.item()
