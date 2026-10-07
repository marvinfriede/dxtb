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
Model and system (B3): `Model.setup` builds everything that depends on the
atomic numbers, and gradients flow through it to the parameters.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, OutputHandler, ParamModule
from dxtb._src.calculators.model import Model, System
from dxtb._src.constants import labels

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


def test_setup() -> None:
    OutputHandler.verbosity = 0
    numbers = torch.tensor([3, 1], device=DEVICE)
    model = Model(par=ParamModule(GFN2_XTB, **DD))
    system = model.setup(numbers)

    assert isinstance(system, System)
    assert system.model is model
    assert system.batch_mode == 0
    # the integral level follows the parametrization
    assert system.config.ints.level == labels.INTLEVEL_QUADRUPOLE
    assert model.config.ints.level != labels.INTLEVEL_QUADRUPOLE
    # the data of the classical terms is part of the system
    assert set(system.classical_cache) == set(
        c.label for c in system.classicals.components
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        system.batch_mode = 1  # type: ignore[misc]

    batch = model.setup(torch.tensor([[3, 1], [1, 1]], device=DEVICE))
    assert batch.batch_mode == 1


def test_setup_is_differentiable() -> None:
    """The element parameters are gathered in `setup`."""
    OutputHandler.verbosity = 0
    numbers = torch.tensor([3, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)
    arep = dict(par.named_parameters())["parameter_tree.element.H.arep.param"]
    arep.requires_grad_(True)

    system = Model(par=par).setup(numbers)
    (grad,) = torch.autograd.grad(
        system.classical_cache["Repulsion"].arep.sum(), arep
    )
    assert grad.abs().sum() > 0


def test_calculator_holds_the_system() -> None:
    calc = Calculator(
        torch.tensor([3, 1], device=DEVICE), GFN1_XTB, opts={"verbosity": 0}
    )
    assert calc.ihelp is calc.system.ihelp
    assert calc.classicals is calc.system.classicals
    assert calc.opts is calc.system.config
