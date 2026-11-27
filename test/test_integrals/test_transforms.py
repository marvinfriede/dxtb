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
"""Tests for pure integral matrix transforms."""

from __future__ import annotations

import pytest
import torch
from torch.func import jvp, vmap

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.constants import labels
from dxtb._src.integral.driver.pytorch import build_overlap
from dxtb._src.integral.driver.pytorch.multipole import (
    _shift_dipole_to_ket,
    _shift_quadrupole_to_ket,
    build_dipole,
    build_quadrupole,
    evaluate_moments,
)
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetup,
    setup_integrals,
)
from dxtb._src.integral.evaluation import _make_quadrupole_traceless

from ..conftest import DEVICE
from ..utils import make_basis, pytorch_setup
from .samples import samples

_LOWER_TRIANGLE = [0, 3, 4, 6, 7, 8]  # xx, yx, yy, zx, zy, zz of the 9 raw


def test_pure_dipole_shift_without_mutation() -> None:
    """Dipole origin shifts use explicit matrices and positions."""
    dd = {"device": DEVICE, "dtype": torch.double}
    overlap = torch.eye(3, **dd)
    dipole = torch.randn(3, 3, 3, **dd)
    positions = torch.randn(3, 3, **dd)
    originals = (dipole.clone(), overlap.clone(), positions.clone())

    _shift_dipole_to_ket(dipole, overlap, positions)

    assert torch.equal(dipole, originals[0])
    assert torch.equal(overlap, originals[1])
    assert torch.equal(positions, originals[2])


def test_pure_quadrupole_transforms_without_mutation() -> None:
    """Quadrupole shift and traceless conversion are pure."""
    dd = {"device": DEVICE, "dtype": torch.double}
    raw6 = torch.randn(6, 3, 3, **dd)
    overlap = torch.eye(3, **dd)
    dipole = torch.randn(3, 3, 3, **dd)
    positions = torch.randn(3, 3, **dd)
    originals = tuple(x.clone() for x in (raw6, overlap, dipole, positions))

    shifted = _shift_quadrupole_to_ket(raw6, dipole, overlap, positions)
    shifted_before = shifted.clone()
    _make_quadrupole_traceless(shifted)

    assert torch.equal(shifted, shifted_before)
    for tensor, original in zip((raw6, overlap, dipole, positions), originals):
        assert torch.equal(tensor, original)


def _setup_and_inputs() -> (
    tuple[PytorchIntegralSetup, torch.Tensor, torch.Tensor, IndexHelper]
):
    """Create a reusable small PyTorch integral setup and geometry."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    basis = make_basis(torch.unique(numbers), GFN1_XTB, ihelp, **dd)
    return pytorch_setup(basis), numbers, positions, ihelp


def _evaluate_multipoles(
    setup: PytorchIntegralSetup,
    positions: torch.Tensor,
    ihelp: IndexHelper,
    *,
    reverse_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build and transform overlap and multipoles for one geometry."""
    overlap = build_overlap(setup, positions)

    if reverse_order:
        quadrupole_centered = build_quadrupole(setup, positions)
        dipole_centered = build_dipole(setup, positions)
    else:
        dipole_centered = build_dipole(setup, positions)
        quadrupole_centered = build_quadrupole(setup, positions)

    orbital_positions = ihelp.spread_atom_to_orbital(
        positions, dim=-2, extra=True
    )
    quadrupole_shifted = _shift_quadrupole_to_ket(
        quadrupole_centered[_LOWER_TRIANGLE],
        dipole_centered,
        overlap,
        orbital_positions,
    )
    dipole = _shift_dipole_to_ket(dipole_centered, overlap, orbital_positions)
    quadrupole = _make_quadrupole_traceless(quadrupole_shifted)
    return overlap, dipole, quadrupole


def test_pure_transform_pipeline_is_call_local_and_differentiable() -> None:
    """Geometry order and prior backward calls do not affect transforms."""
    setup, _, positions, ihelp = _setup_and_inputs()
    first_positions = positions.clone().requires_grad_(True)
    first = _evaluate_multipoles(setup, first_positions, ihelp)
    first_grad = torch.autograd.grad(
        sum(value.square().sum() for value in first), first_positions
    )[0]

    changed = positions.clone()
    changed[1, 0] += 0.17
    second = _evaluate_multipoles(setup, changed, ihelp, reverse_order=True)
    repeated = _evaluate_multipoles(
        setup, positions.clone(), ihelp, reverse_order=True
    )
    cloned_leaf = positions.clone().requires_grad_(True)
    cloned = _evaluate_multipoles(setup, cloned_leaf, ihelp)
    cloned_grad = torch.autograd.grad(
        sum(value.square().sum() for value in cloned), cloned_leaf
    )[0]

    assert torch.isfinite(first_grad).all()
    assert torch.allclose(first_grad, cloned_grad, atol=1e-11, rtol=1e-11)

    for original, again, clone_value in zip(first, repeated, cloned):
        assert torch.allclose(original, again, atol=1e-12, rtol=1e-12)
        assert torch.allclose(original, clone_value, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(first[1], second[1], atol=1e-10, rtol=1e-10)
    assert not torch.allclose(first[2], second[2], atol=1e-10, rtol=1e-10)

    _, tangent_result = jvp(
        lambda pos: _evaluate_multipoles(setup, pos, ihelp)[2],
        (positions,),
        (torch.ones_like(positions),),
    )
    batched = vmap(lambda pos: _evaluate_multipoles(setup, pos, ihelp)[2])(
        torch.stack((positions, positions.clone()))
    )
    assert torch.isfinite(tangent_result).all()
    assert torch.allclose(batched[0], first[2], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ket_centered_moments_are_translation_invariant(
    dtype: torch.dtype,
) -> None:
    """Moving the molecule far from the origin leaves the moments unchanged."""
    dd = {"device": DEVICE, "dtype": dtype}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    setup = pytorch_setup(
        make_basis(torch.unique(numbers), GFN1_XTB, ihelp, **dd)
    )

    reference = evaluate_moments(setup, positions, labels.INTLEVEL_QUADRUPOLE)
    translated = evaluate_moments(
        setup, positions + 100.0, labels.INTLEVEL_QUADRUPOLE
    )

    # translating rounds the positions themselves (eps * 100 Bohr)
    tol = torch.finfo(dtype).eps * 1e3
    for ref, value in zip(reference, translated):
        torch.testing.assert_close(value, ref, atol=tol, rtol=0.0)
