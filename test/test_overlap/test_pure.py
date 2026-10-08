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
"""Tests for explicit, reusable PyTorch overlap setup data."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch.overlap import (
    build_overlap,
    setup_overlap,
)
from dxtb._src.param import Param, ParamModule

from ..conftest import DEVICE
from .samples import samples
from .utils import calc_overlap


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
def test_pure_overlap_reuses_setup_without_call_state(
    par: Param | ParamModule,
) -> None:
    """One setup gives geometry- and call-history-independent matrices."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    basis = Basis(torch.unique(numbers), par, ihelp, **dd)
    setup = setup_overlap(basis)

    first = build_overlap(setup, positions)
    changed = positions.clone()
    changed[1, 0] += 0.17
    second = build_overlap(setup, changed)
    repeated = build_overlap(setup, positions.clone())
    legacy = calc_overlap(numbers, positions, par, dd, uplo="n")

    assert torch.allclose(first, legacy, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(first, repeated, atol=1.0e-12, rtol=1.0e-12)
    assert not torch.allclose(first, second, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
def test_pure_overlap_equal_leaf_gradients(
    par: Param | ParamModule,
) -> None:
    """Equal position tensors produce equal values and position gradients."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    basis = Basis(torch.unique(numbers), par, ihelp, **dd)
    setup = setup_overlap(basis)

    left = positions.clone().requires_grad_(True)
    right = positions.clone().requires_grad_(True)
    left_value = build_overlap(setup, left)
    right_value = build_overlap(setup, right)
    left_grad = torch.autograd.grad(left_value.square().sum(), left)[0]
    right_grad = torch.autograd.grad(right_value.square().sum(), right)[0]

    assert torch.allclose(left_value, right_value, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(left_grad, right_grad, atol=1.0e-11, rtol=1.0e-11)
