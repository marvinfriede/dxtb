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
Test the integral driver manager.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, IndexHelper, ParamModule
from dxtb._src.constants.labels import (
    INTDRIVER_LIBCINT,
    INTDRIVER_PYTORCH,
)
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.libcint import (
    IntDriverLibcint,
    LibcintCallData,
    OverlapLibcint,
    build_overlap_libcint,
)
from dxtb._src.integral.driver.manager import DriverManager
from dxtb._src.integral.driver.pytorch import (
    DipolePytorch,
    IntDriverPytorch,
    OverlapPytorch,
    QuadrupolePytorch,
)
from dxtb._src.typing import DD

from ...conftest import DEVICE


def test_fail() -> None:
    mgr = DriverManager(-99)

    with pytest.raises(RuntimeError):
        _ = mgr.driver

    with pytest.raises(ValueError):
        numbers = torch.tensor([1, 2], device=DEVICE)
        mgr.create_driver(
            numbers, GFN1_XTB, IndexHelper.from_numbers(numbers, GFN1_XTB)
        )


def single(name: int, dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.zeros((2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    mgr = DriverManager(name, force_cpu_for_libcint=force_cpu_for_libcint, **dd)
    mgr.create_driver(numbers, GFN1_XTB, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    call_data = mgr.setup_driver(positions)
    if name == INTDRIVER_PYTORCH:
        assert isinstance(mgr.driver, IntDriverPytorch)
        assert mgr.driver.is_latest(positions) is True
    elif name == INTDRIVER_LIBCINT:
        assert isinstance(mgr.driver, IntDriverLibcint)
        assert isinstance(call_data, LibcintCallData)
        assert mgr.driver.is_setup() is False

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    if name == INTDRIVER_PYTORCH:
        assert mgr.driver.is_latest(positions) is False
    else:
        moved_data = mgr.setup_driver(positions)
        assert isinstance(moved_data, LibcintCallData)
        assert moved_data is not call_data


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_libcint_single(
    dtype: torch.dtype, force_cpu_for_libcint: bool
) -> None:
    single(INTDRIVER_LIBCINT, dtype, force_cpu_for_libcint)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_pytorch_single(
    dtype: torch.dtype, force_cpu_for_libcint: bool
) -> None:
    single(INTDRIVER_PYTORCH, dtype, force_cpu_for_libcint)


def batch(name: int, dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    numbers = torch.tensor([[3, 1], [1, 0]], device=DEVICE)
    positions = torch.zeros((2, 2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    mgr = DriverManager(name, force_cpu_for_libcint=force_cpu_for_libcint, **dd)
    mgr.create_driver(numbers, GFN1_XTB, ihelp)

    if force_cpu_for_libcint is True:
        positions = positions.cpu()

    call_data = mgr.setup_driver(positions)
    if name == INTDRIVER_PYTORCH:
        assert isinstance(mgr.driver, IntDriverPytorch)
        assert mgr.driver.is_latest(positions) is True
    elif name == INTDRIVER_LIBCINT:
        assert isinstance(mgr.driver, IntDriverLibcint)
        assert isinstance(call_data, LibcintCallData)
        assert mgr.driver.is_setup() is False

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    if name == INTDRIVER_PYTORCH:
        assert mgr.driver.is_latest(positions) is False
    else:
        moved_data = mgr.setup_driver(positions)
        assert isinstance(moved_data, LibcintCallData)
        assert moved_data is not call_data


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_libcint_batch(dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    batch(INTDRIVER_LIBCINT, dtype, force_cpu_for_libcint)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("force_cpu_for_libcint", [True, False])
def test_pytorch_batch(dtype: torch.dtype, force_cpu_for_libcint: bool) -> None:
    batch(INTDRIVER_PYTORCH, dtype, force_cpu_for_libcint)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_libcint_setup_is_call_local() -> None:
    """Each libcint setup is fresh and independent of earlier geometries."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([1, 1], device=DEVICE)
    pos_a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)
    pos_b = pos_a.clone()
    pos_b[1, 2] += 0.2

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    manager = DriverManager(INTDRIVER_LIBCINT, **dd)
    manager.create_driver(numbers, GFN1_XTB, ihelp)
    overlap = OverlapLibcint(**dd)

    first_call = manager.setup_driver(pos_a)
    assert isinstance(first_call, LibcintCallData)
    pure_first = build_overlap_libcint(first_call)
    assert overlap._matrix is None
    assert torch.allclose(pure_first, OverlapLibcint(**dd).build(first_call))
    first = overlap.build(first_call).clone()

    second_call = manager.setup_driver(pos_b)
    assert isinstance(second_call, LibcintCallData)
    pure_second = build_overlap_libcint(second_call)
    second = overlap.build(second_call).clone()
    assert torch.allclose(pure_second, second)
    assert not torch.allclose(first, second)

    pos_a_leaf = pos_a.clone().requires_grad_()
    third_call = manager.setup_driver(pos_a_leaf)
    assert isinstance(third_call, LibcintCallData)
    pure_third = build_overlap_libcint(third_call)
    assert torch.allclose(pure_first, pure_third, atol=1e-13, rtol=0.0)
    third = overlap.build(third_call)
    assert torch.allclose(first, third, atol=1e-13, rtol=0.0)
    (gradient,) = torch.autograd.grad(third.sum(), pos_a_leaf)
    assert torch.isfinite(gradient).all()

    manager.invalidate_driver()
    invalidated_call = manager.setup_driver(pos_a)
    assert isinstance(invalidated_call, LibcintCallData)
    assert manager.driver.is_setup() is False
    assert not hasattr(manager.driver, "drv")


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_libcint_batched_basis_preserves_repeated_atoms() -> None:
    """Batched wrappers retain the full per-system atom ordering."""
    from dxtb._src.exlibs import libcint

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([[1, 1, 8], [3, 1, 0]], device=DEVICE)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6], [0.0, 0.0, 0.0]],
        ],
        **dd,
    )
    batch_ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    batch_manager = DriverManager(INTDRIVER_LIBCINT, **dd)
    batch_manager.create_driver(numbers, GFN1_XTB, batch_ihelp)
    batch_call = batch_manager.setup_driver(positions)
    assert isinstance(batch_call, LibcintCallData)

    for index, count in enumerate((3, 2)):
        system_numbers = numbers[index, :count]
        system_positions = positions[index, :count]
        system_ihelp = IndexHelper.from_numbers(system_numbers, GFN1_XTB)
        system_manager = DriverManager(INTDRIVER_LIBCINT, **dd)
        system_manager.create_driver(system_numbers, GFN1_XTB, system_ihelp)
        system_call = system_manager.setup_driver(system_positions)
        assert isinstance(system_call, LibcintCallData)
        assert torch.allclose(
            libcint.overlap(batch_call.drivers[index]),
            libcint.overlap(system_call.drivers[0]),
            atol=1e-13,
            rtol=0.0,
        )


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_libcint_setup_rebuilds_parameter_graph_per_call() -> None:
    """Repeated evaluations do not reuse an already-consumed parameter graph."""
    from torch.nn import Parameter

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([1, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **dd)
    par = ParamModule(GFN1_XTB, **dd)
    slater = par.get("element", "H", "slater")
    assert isinstance(slater, Parameter)
    slater.requires_grad_(True)

    ihelp = IndexHelper.from_numbers(numbers, par)
    manager = DriverManager(INTDRIVER_LIBCINT, **dd)
    manager.create_driver(numbers, par, ihelp)
    overlap = OverlapLibcint(**dd)

    gradients = []
    for _ in range(2):
        call_data = manager.setup_driver(positions)
        value = overlap.build(call_data).sum()
        (gradient,) = torch.autograd.grad(value, slater)
        gradients.append(gradient)

    assert torch.isfinite(gradients[0]).all()
    assert torch.isfinite(gradients[1]).all()
    assert torch.allclose(gradients[0], gradients[1])


@pytest.mark.parametrize("kind", ["overlap", "quadrupole"])
def test_pytorch_driver_rebuilds_integrals_for_new_positions(kind: str) -> None:
    """After the positions change (or the driver is invalidated), the next
    ``setup_driver`` + ``build`` must give the integrals of the new geometry,
    identical to a freshly created driver."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([3, 1, 8], device=DEVICE)
    pos_a = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [1.5, 0.0, -1.0]], **dd
    )
    shift = torch.tensor(
        [[0.0, 0.1, 0.0], [0.2, 0.0, 0.1], [0.0, 0.0, -0.3]], **dd
    )
    pos_b = pos_a + shift

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    def manager() -> DriverManager:
        mgr = DriverManager(INTDRIVER_PYTORCH, algorithm="os", **dd)
        mgr.create_driver(numbers, GFN1_XTB, ihelp)
        return mgr

    def build(mgr: DriverManager) -> torch.Tensor:
        cls = {
            "overlap": OverlapPytorch,
            "dipole": DipolePytorch,
            "quadrupole": QuadrupolePytorch,
        }
        return cls[kind](**dd).build(mgr.driver)

    fresh = manager()
    fresh.setup_driver(pos_b)
    expected = build(fresh)

    mgr = manager()
    mgr.setup_driver(pos_a)
    first = build(mgr)
    assert mgr.driver.is_latest(pos_b) is False

    # moved positions: setup again and rebuild
    mgr.setup_driver(pos_b)
    moved = build(mgr)
    assert not torch.allclose(first, moved, atol=1e-6)
    assert torch.allclose(moved, expected, atol=1e-13, rtol=0.0)

    # explicit invalidation, then back to the first geometry
    mgr.invalidate_driver()
    mgr.setup_driver(pos_a)
    assert torch.allclose(build(mgr), first, atol=1e-13, rtol=0.0)
