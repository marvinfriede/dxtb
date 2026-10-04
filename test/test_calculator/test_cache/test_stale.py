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
Cache findings of the baseline (T0.5 in ``docs/plan/01-T0-baseline.md``).

The component caches and the integral driver are validated by comparing
tensor *values*. A cache built from positions without gradient tracking was
therefore reused for positions with gradient tracking, which silently dropped
the position dependence of these terms from the forces. The stopgap also
checks the gradient-tracking state (``dxtb._src.utils.tensors.grad_key``).

The remaining findings are known issues (``xfail(strict=True)``) that are
fixed by removing the caches (B5, B6). These tests are not maintained beyond
that point.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb.components.field import new_efield

from ...conftest import DEVICE

pytestmark = pytest.mark.cache

# water (bohr)
NUMBERS = [8, 1, 1]
POSITIONS = [
    [-0.00000000000000, -0.00000000000000, -0.74288549752983],
    [-1.43472674945442, -0.00000000000000, +0.37144274876492],
    [+1.43472674945442, +0.00000000000000, +0.37144274876492],
]

drivers = ["pytorch"] + (["libcint"] if has_libcint else [])


def _setup(dd: DD) -> tuple[Tensor, Tensor]:
    numbers = torch.tensor(NUMBERS, device=DEVICE)
    positions = torch.tensor(POSITIONS, **dd)
    return numbers, positions


def _fresh_forces(numbers: Tensor, positions: Tensor, par, opts) -> Tensor:
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    calc = Calculator(numbers, par, opts=opts, **dd)
    pos = positions.detach().clone().requires_grad_(True)
    return calc.get_forces(pos)


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
@pytest.mark.parametrize("driver", drivers)
def test_no_grad_call_then_forces(par, driver: str) -> None:
    """
    Test 1: ``get_energy`` without gradient tracking, then ``get_forces``
    with ``requires_grad`` switched on in place. Previously wrong by up to
    0.1 Eh/bohr; fixed by the stopgap.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)
    opts = {"verbosity": 0, "int_driver": driver}

    calc = Calculator(numbers, par, opts=opts, **dd)
    pos = positions.clone()
    calc.get_energy(pos)
    pos.requires_grad_(True)
    forces = calc.get_forces(pos)

    ref = _fresh_forces(numbers, positions, par, opts)
    assert pytest.approx(ref.cpu(), abs=1e-10) == forces.detach().cpu()


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
@pytest.mark.parametrize("driver", drivers)
def test_new_leaf_same_values(par, driver: str) -> None:
    """
    Variant of test 1: two different leaf tensors with the same values, both
    requiring gradients. The forces with respect to the second must not use
    the graph of the first. Fixed by the stopgap.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)
    opts = {"verbosity": 0, "int_driver": driver}

    calc = Calculator(numbers, par, opts=opts, **dd)
    calc.get_energy(positions.clone().requires_grad_(True))
    forces = calc.get_forces(positions.clone().requires_grad_(True))

    ref = _fresh_forces(numbers, positions, par, opts)
    assert pytest.approx(ref.cpu(), abs=1e-10) == forces.detach().cpu()


@pytest.mark.xfail(
    raises=RuntimeError,
    strict=True,
    reason=(
        "Known issue: the second call reuses a cached graph that the first "
        "backward pass has freed ('backward through the graph a second "
        "time'). Removed with the caches (B6)."
    ),
)
def test_forces_twice_same_leaf() -> None:
    """Repeated ``get_forces`` on the same leaf tensor raises."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)

    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, **dd)
    pos = positions.clone().requires_grad_(True)
    calc.get_forces(pos)
    calc.get_forces(pos)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known issue: with `cache_enabled=True`, `update_efield` does not "
        "invalidate the result cache and the energy key covers only the "
        "method arguments, so every displaced-field energy is the cached "
        "zero-field energy. Removed with the result cache (B5)."
    ),
)
def test_numerical_dipole_with_result_cache() -> None:
    """Test 2: numerical dipole with the result cache enabled."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)

    def dipole(cache: bool) -> Tensor:
        field = torch.zeros(3, **dd)
        calc = Calculator(
            numbers,
            GFN1_XTB,
            interaction=[new_efield(field)],
            opts={"verbosity": 0, "cache_enabled": cache},
            **dd,
        )
        return calc.dipole_numerical(positions)

    ref = dipole(cache=False)
    assert ref.abs().max() > 0.5  # water has a dipole moment
    assert pytest.approx(ref.cpu(), abs=1e-6) == dipole(cache=True).cpu()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known issue: the result cache keys tensors on `data_ptr()` and "
        "version only (`utils.tensors.tensor_id`), so a view sharing the "
        "storage of a cached batch returns the batched result instead of "
        "being validated. Removed with the result cache (B5)."
    ),
)
def test_view_of_batch_with_result_cache() -> None:
    """
    Test 3: batched energy, then energy of ``positions[0]``. Without the
    cache, the call is rejected (shape mismatch); with the cache, the cached
    batched energies are returned.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions = _setup(dd)
    numbers = pack([numbers, torch.tensor([1, 1], device=DEVICE)])
    positions = pack(
        [positions, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)]
    )

    calc = Calculator(
        numbers, GFN1_XTB, opts={"verbosity": 0, "cache_enabled": True}, **dd
    )
    calc.get_energy(positions)

    with pytest.raises(ValueError):
        calc.get_energy(positions[0])
