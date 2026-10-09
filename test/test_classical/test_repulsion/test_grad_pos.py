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
Run tests for nuclear repulsion gradient.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.components.classicals import Repulsion, new_repulsion
from dxtb._src.components.classicals.repulsion.base import BaseRepulsionCache
from dxtb._src.typing import DD, Callable, Tensor

from ...conftest import DEVICE
from ...utils import get_param_module
from .samples import samples

sample_list = ["H2O", "SiH4", "MB16_43_01", "MB16_43_02", "LYS_xao"]

tol = 1e-7


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_backward_vs_tblite(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["gfn1_grad"].to(**dd)

    par = get_param_module("gfn1", **dd)
    rep = new_repulsion(
        torch.unique(numbers),
        par,
        **dd,
    )
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    # automatic gradient
    pos = positions.clone().requires_grad_(True)
    energy = torch.sum(rep.get_energy(pos, cache), dim=-1)
    energy.backward()

    assert pos.grad is not None
    grad_backward = pos.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    pos.detach_()
    pos.grad.data.zero_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["H2O", "SiH4"])
@pytest.mark.parametrize("name2", ["H2O", "SiH4"])
def test_backward_batch_vs_tblite(
    dtype: torch.dtype, name1: str, name2: str
) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    ref = pack(
        [
            sample1["gfn1_grad"].to(**dd),
            sample2["gfn1_grad"].to(**dd),
        ]
    )

    par = get_param_module("gfn1", **dd)
    rep = new_repulsion(
        torch.unique(numbers),
        par,
        **dd,
    )
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    # automatic gradient
    pos = positions.clone().requires_grad_(True)
    energy = torch.sum(rep.get_energy(pos, cache))
    energy.backward()

    assert pos.grad is not None
    grad_backward = pos.grad.clone()

    # also zero out gradients when using `.backward()`
    grad_backward.detach_()
    pos.detach_()
    pos.grad.data.zero_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()


def calc_numerical_gradient(
    positions: Tensor, rep: Repulsion, cache: BaseRepulsionCache
) -> Tensor:
    """Calculate gradient numerically for reference."""

    n_atoms = positions.shape[0]

    # setup numerical gradient
    gradient = torch.zeros((n_atoms, 3), dtype=positions.dtype)
    step = 1.0e-6

    for i in range(n_atoms):
        for j in range(3):
            positions[i, j] += step
            er = rep.get_energy(positions, cache)
            er = torch.sum(er, dim=-1)

            positions[i, j] -= 2 * step
            el = rep.get_energy(positions, cache)
            el = torch.sum(el, dim=-1)

            positions[i, j] += step
            gradient[i, j] = 0.5 * (er - el) / step

    return gradient


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad_pos_autograd_vs_numerical(dtype: torch.dtype, name: str) -> None:
    """Compare the PyTorch position derivative with finite differences."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    atol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    par = get_param_module("gfn1", **dd)
    rep = new_repulsion(torch.unique(numbers), par, **dd)
    assert rep is not None

    ihelp = IndexHelper.from_numbers(numbers, par)
    cache = rep.get_cache(numbers, ihelp)

    pos = positions.clone().requires_grad_(True)
    energy = rep.get_energy(pos, cache).sum()
    (grad_autograd,) = torch.autograd.grad(energy, pos)

    grad_num = calc_numerical_gradient(positions.clone(), rep, cache)
    assert pytest.approx(grad_num.cpu(), abs=atol) == grad_autograd.cpu()


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    assert GFN1_XTB.repulsion is not None

    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    par = get_param_module("gfn1", **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    rep = new_repulsion(torch.unique(numbers), par, **dd)
    assert rep is not None
    cache = rep.get_cache(numbers, ihelp)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        return rep.get_energy(p, cache)

    return func, pos


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_grad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list + ["MB16_43_03"])
def test_gradgrad(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[Callable[[Tensor], Tensor], Tensor]:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    par = get_param_module("gfn1", **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    rep = new_repulsion(torch.unique(numbers), par, **dd)
    assert rep is not None
    cache = rep.get_cache(numbers, ihelp)

    # variables to be differentiated
    pos = positions.clone().requires_grad_(True)

    def func(p: Tensor) -> Tensor:
        return rep.get_energy(p, cache)

    return func, pos


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list + ["MB16_43_03"])
def test_grad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list + ["MB16_43_03"])
def test_gradgrad_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of positions against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)


def test_forward_transforms_match_reverse_and_loop() -> None:
    """The production repulsion path composes with PyTorch transforms."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = samples["H2"]["numbers"].to(DEVICE)
    positions = samples["H2"]["positions"].to(**dd)
    repulsion = new_repulsion(torch.unique(numbers), GFN1_XTB, **dd)
    assert repulsion is not None
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    cache = repulsion.get_cache(numbers, ihelp)

    def scalar(pos: Tensor) -> Tensor:
        return repulsion.get_energy(pos, cache).sum()

    reverse = jacrev(scalar)(positions)
    forward = jacfwd(scalar)(positions)
    torch.testing.assert_close(forward, reverse)

    direction = torch.tensor([[0.2, -0.1, 0.3], [-0.2, 0.1, -0.3]], **dd)
    _, tangent = jvp(scalar, (positions,), (direction,))
    torch.testing.assert_close(tangent, (reverse * direction).sum())

    moved = positions.clone()
    moved[1, 2] += 0.05
    conformers = torch.stack((positions, moved))
    mapped = vmap(scalar)(conformers)
    looped = torch.stack(tuple(scalar(pos) for pos in conformers))
    torch.testing.assert_close(mapped, looped)
    assert not torch.isclose(mapped[0], mapped[1])


def test_padding_has_finite_energy_and_position_gradient() -> None:
    """Padded pair masks keep the plain derivative finite."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers = pack(
        (
            samples["H2"]["numbers"].to(DEVICE),
            samples["H2O"]["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            samples["H2"]["positions"].to(**dd),
            samples["H2O"]["positions"].to(**dd),
        )
    ).requires_grad_(True)
    repulsion = new_repulsion(torch.unique(numbers), GFN1_XTB, **dd)
    assert repulsion is not None
    cache = repulsion.get_cache(
        numbers, IndexHelper.from_numbers(numbers, GFN1_XTB)
    )
    energy = repulsion.get_energy(positions, cache)
    (gradient,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(gradient).all()
