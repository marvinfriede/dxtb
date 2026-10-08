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
"""Tests for explicit, reusable PyTorch multipole setup data."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.func import jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch.dipole import DipolePytorch
from dxtb._src.integral.driver.pytorch.driver import IntDriverPytorch
from dxtb._src.integral.driver.pytorch.impls.kernels import (
    ALGORITHMS,
    get_kernel,
)
from dxtb._src.integral.driver.pytorch.impls.pairs import assemble_matrix
from dxtb._src.integral.driver.pytorch.impls.pipeline import (
    DIPOLE_COMPONENTS,
    QUADRUPOLE_COMPONENTS,
)
from dxtb._src.integral.driver.pytorch.multipole import (
    build_dipole,
    build_quadrupole,
)
from dxtb._src.integral.driver.pytorch.overlap import PytorchOverlapSetup
from dxtb._src.integral.driver.pytorch.quadrupole import QuadrupolePytorch
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetup,
    setup_integrals,
)
from dxtb._src.param import Param, ParamModule

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    ("build", "legacy"),
    [
        (build_dipole, DipolePytorch),
        (build_quadrupole, QuadrupolePytorch),
    ],
)
def test_pure_multipole_reuses_setup_without_call_state(
    par: Param | ParamModule,
    algorithm: str,
    build: Callable[[PytorchIntegralSetup, torch.Tensor], torch.Tensor],
    legacy: type[DipolePytorch] | type[QuadrupolePytorch],
) -> None:
    """One setup gives raw multipoles independent of geometry call history."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    basis = Basis(torch.unique(numbers), par, ihelp, **dd)
    setup = setup_integrals(basis, algorithm=algorithm)
    assert isinstance(setup, PytorchOverlapSetup)

    first = build(setup, positions)
    components = (
        DIPOLE_COMPONENTS if build is build_dipole else QUADRUPOLE_COMPONENTS
    )
    direct = assemble_matrix(
        get_kernel(algorithm),
        setup.ihelp,
        list(setup.alphas),
        list(setup.coeffs),
        positions,
        components,
        plan=setup.pair_plan,
    )
    changed = positions.clone()
    changed[1, 0] += 0.17
    second = build(setup, changed)
    repeated = build(setup, positions.clone())

    driver = IntDriverPytorch(numbers, par, ihelp, **dd)
    driver.algorithm = algorithm
    driver.setup(positions)
    old = legacy(**dd).build(driver)

    assert first.shape[0] == (3 if build is build_dipole else 9)
    assert torch.allclose(first, direct, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(first, old, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(first, repeated, atol=1.0e-12, rtol=1.0e-12)
    assert not torch.allclose(first, second, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("build", [build_dipole, build_quadrupole])
def test_pure_multipole_equal_leaf_gradients(
    par: Param | ParamModule,
    algorithm: str,
    build: Callable[[PytorchIntegralSetup, torch.Tensor], torch.Tensor],
) -> None:
    """Equal-valued position leaves produce equal multipoles and gradients."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    basis = Basis(torch.unique(numbers), par, ihelp, **dd)
    setup = setup_integrals(basis, algorithm=algorithm)

    left = positions.clone().requires_grad_(True)
    right = positions.clone().requires_grad_(True)
    left_value = build(setup, left)
    right_value = build(setup, right)
    left_grad = torch.autograd.grad(left_value.square().sum(), left)[0]
    right_grad = torch.autograd.grad(right_value.square().sum(), right)[0]

    assert torch.allclose(left_value, right_value, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(left_grad, right_grad, atol=1.0e-11, rtol=1.0e-11)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("build", [build_dipole, build_quadrupole])
def test_pure_multipole_supports_jvp_and_vmap(
    algorithm: str,
    build: Callable[[PytorchIntegralSetup, torch.Tensor], torch.Tensor],
) -> None:
    """The explicit multipole builders remain compatible with torch.func."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    basis = Basis(torch.unique(numbers), GFN1_XTB, ihelp, **dd)
    setup = setup_integrals(basis, algorithm=algorithm)
    tangent = torch.ones_like(positions)

    value, derivative = jvp(
        lambda pos: build(setup, pos), (positions,), (tangent,)
    )
    batched = vmap(lambda pos: build(setup, pos))(
        torch.stack((positions, positions.clone()))
    )

    assert torch.isfinite(value).all()
    assert torch.isfinite(derivative).all()
    assert torch.allclose(batched[0], value, atol=1.0e-12, rtol=1.0e-12)


def test_legacy_driver_algorithm_change_refreshes_setup() -> None:
    """Changing the legacy algorithm updates the explicit setup value."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    driver = IntDriverPytorch(numbers, GFN1_XTB, ihelp, **dd)
    driver.algorithm = "os"
    driver.setup(positions)
    driver.algorithm = "md"

    assert driver._integral_setup is not None
    assert driver._integral_setup.algorithm == "md"
    expected = build_dipole(driver._integral_setup, positions)
    actual = driver.eval_matrix(DIPOLE_COMPONENTS)

    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
