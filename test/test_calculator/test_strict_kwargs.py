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
"""Calculator methods reject keyword arguments that nothing takes."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, Calculator

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.float64}
OPTS = {"verbosity": 0, "int_driver": "pytorch", "fermi_etemp": 0.0}

H2 = torch.tensor([1, 1], device=DEVICE)
H2_POS = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **DD)


def _calc(batched: bool = False) -> tuple[Calculator, torch.Tensor]:
    if batched:
        numbers = torch.stack([H2, H2])
        positions = torch.stack([H2_POS, H2_POS + 0.1])
    else:
        numbers, positions = H2, H2_POS
    return Calculator(numbers, GFN1_XTB, opts=OPTS, **DD), positions


def test_constructor_rejects_unknown_keyword_arguments() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'typo'"):
        Calculator(H2, GFN1_XTB, opts=OPTS, typo=True, **DD)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("method", ["singlepoint", "energy", "forces"])
def test_methods_reject_unknown_keyword_arguments(
    batched: bool, method: str
) -> None:
    calc, positions = _calc(batched)
    positions = positions.clone().requires_grad_(method == "forces")
    with pytest.raises(TypeError, match="unexpected keyword argument 'typo'"):
        getattr(calc, method)(positions, typo=True)


@pytest.mark.parametrize("batched", [False, True])
def test_ignore_unknown_kwargs_is_unknown(batched: bool) -> None:
    calc, positions = _calc(batched)
    with pytest.raises(
        TypeError, match="unexpected keyword argument 'ignore_unknown_kwargs'"
    ):
        calc.singlepoint(positions, ignore_unknown_kwargs=True)


def test_energy_keepdim_and_singlepoint_options() -> None:
    calc, positions = _calc()
    energy = calc.energy(positions, keepdim=True, cuda_sync_in_scf=False)
    assert energy.shape == (1,)


def test_calculate_gives_each_property_its_keyword_arguments() -> None:
    """``use_functorch`` reaches the Hessian, not the single point."""
    calc, positions = _calc()
    values = calc.calculate(
        ["energy", "charges", "hessian"],
        positions.clone().requires_grad_(True),
        use_functorch=False,
        keepdim=True,
    )
    assert values["energy"].shape == (1,)
    assert values["hessian"].shape == (2, 3, 2, 3)


@pytest.mark.parametrize(
    "properties,name",
    [
        (["energy"], "typo"),
        (["energy", "charges"], "use_functorch"),
        (["dipole"], "keepdim"),
    ],
)
def test_calculate_rejects_keyword_arguments_no_property_takes(
    properties: list[str], name: str
) -> None:
    calc, positions = _calc()
    with pytest.raises(
        TypeError, match=f"unexpected keyword argument '{name}'"
    ):
        calc.calculate(properties, positions, **{name: True})
