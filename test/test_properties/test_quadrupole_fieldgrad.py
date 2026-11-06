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
Quadrupole moment from the derivative of the energy with respect to the
electric field gradient (autograd and finite differences).

The derivative of the (rotation-covariant) energy agrees with the analytical
quadrupole moment for the diagonal elements. For the off-diagonal elements, the
analytical result doubles the contribution of the point charges and dipoles
(following `tblite`); the difference is exactly ``1.5 * Q_ij`` of that part.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN2_XTB, Calculator
from dxtb._src.components.interactions import new_efield_grad
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

opts = {
    "maxiter": 100,
    "mixer": "anderson",
    "scf_mode": "full",
    "verbosity": 0,
    "x_atol": 1.0e-10,
}

DIAG = [0, 2, 5]  # xx, yy, zz of (xx, yx, yy, zx, zy, zz)

sample_list = [
    "H2O",
    pytest.param("SiH4", marks=pytest.mark.slow),
    "MB16_43_01",
]


def _calc(
    numbers: Tensor, dd: DD, grad: bool, **kwargs
) -> tuple[Calculator, Tensor]:
    field_grad = torch.zeros((3, 3), **dd).requires_grad_(grad)
    efg = new_efield_grad(field_grad)
    calc = Calculator(
        numbers, GFN2_XTB, interaction=[efg], opts=opts, **kwargs, **dd
    )
    return calc, field_grad


@pytest.mark.parametrize("name", sample_list[:2])
def test_autograd_vs_numerical(name: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    calc, _ = _calc(numbers, dd, grad=True)
    auto = calc.quadrupole(positions).detach()
    assert auto.shape == (6,)

    via_get = calc.get_quadrupole(positions).detach()
    via_sys = calc.system.quadrupole(positions).detach()
    assert pytest.approx(auto.cpu(), abs=1e-8) == via_get.cpu()
    assert pytest.approx(auto.cpu(), abs=1e-8) == via_sys.cpu()

    calc_ft, _ = _calc(numbers, dd, grad=True)
    quad_ft = calc_ft.quadrupole(positions, use_functorch=True).detach()
    assert pytest.approx(auto.cpu(), abs=1e-8) == quad_ft.cpu()

    calc, _ = _calc(numbers, dd, grad=False)
    num = calc.quadrupole_numerical(positions, step_size=1e-4)

    assert pytest.approx(auto.cpu(), abs=2e-4) == num.cpu()

    # traceless
    assert pytest.approx(auto[DIAG].sum().item(), abs=1e-8) == 0.0


def test_batch() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    names = ["H2O", "SiH4"]
    numbers = pack([samples[n]["numbers"].to(DEVICE) for n in names])
    positions = pack([samples[n]["positions"].to(**dd) for n in names])

    calc, _ = _calc(numbers, dd, grad=True)
    auto = calc.quadrupole(positions).detach()
    assert auto.shape == (2, 6)

    calc, _ = _calc(numbers, dd, grad=False)
    num = calc.quadrupole_numerical(positions, step_size=1e-4)
    assert pytest.approx(auto.cpu(), abs=2e-4) == num.cpu()

    for i, name in enumerate(names):
        calc, _ = _calc(samples[name]["numbers"].to(DEVICE), dd, grad=True)
        single = calc.quadrupole(samples[name]["positions"].to(**dd)).detach()
        assert pytest.approx(single.cpu(), abs=1e-6) == auto[i].cpu()


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(
            "libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
        "pytorch",
    ],
)
def test_drivers(driver: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    name = "H2O"
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    calc, _ = _calc(numbers, dd, grad=True)
    ref = calc.quadrupole(positions).detach()

    field_grad = torch.zeros((3, 3), **dd).requires_grad_(True)
    efg = new_efield_grad(field_grad)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        interaction=[efg],
        opts={**opts, "int_driver": driver},
        **dd,
    )
    quad = calc.quadrupole(positions).detach()

    assert pytest.approx(ref.cpu(), abs=1e-6) == quad.cpu()


def test_requires_field_gradient() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(**dd)

    calc = Calculator(numbers, GFN2_XTB, opts=opts, **dd)
    with pytest.raises(RuntimeError, match="electric field gradient"):
        calc.quadrupole(positions)
    with pytest.raises(RuntimeError, match="electric field gradient"):
        calc.get_quadrupole(positions)
    with pytest.raises(RuntimeError, match="electric field gradient"):
        calc.quadrupole_numerical(positions)


def test_requires_grad() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(**dd)

    calc, _ = _calc(numbers, dd, grad=False)
    with pytest.raises(RuntimeError, match="requires_grad"):
        calc.quadrupole(positions)
