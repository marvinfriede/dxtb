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
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch.multipole import (
    build_dipole,
    build_quadrupole,
)
from dxtb._src.integral.driver.pytorch.overlap import build_overlap
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetup,
    setup_integrals,
)
from dxtb._src.integral.types.dipole import shift_dipole_origin
from dxtb._src.integral.types.quadrupole import (
    _reduce_9_to_6,
    make_quadrupole_traceless,
    reduce_quadrupole_9_to_6,
    shift_quadrupole_origin,
)
from dxtb._src.integral.utils import normalize_integral_matrix, snorm

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("batched", [False, True])
def test_pure_normalization_without_mutating_inputs(batched: bool) -> None:
    """Matrix normalization returns new tensor values."""
    dd = {"device": DEVICE, "dtype": torch.double}
    seed = torch.randn((2, 4, 4) if batched else (4, 4), **dd)
    raw = seed @ seed.mT + torch.eye(4, **dd)
    norm = torch.rand((2, 4) if batched else (4,), **dd) + 0.5
    raw_before = raw.clone()
    norm_before = norm.clone()

    normalize_integral_matrix(raw, norm)

    assert torch.equal(raw, raw_before)
    assert torch.equal(norm, norm_before)

    if not batched:
        torch.testing.assert_close(
            normalize_integral_matrix(raw),
            normalize_integral_matrix(raw, snorm(raw)),
            atol=1e-14,
            rtol=1e-14,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_snorm_zero_diagonal_has_finite_gradients(dtype: torch.dtype) -> None:
    """Zero diagonal elements in padded overlap do not produce NaN gradients."""
    dd = {"device": DEVICE, "dtype": dtype}
    ovlp = torch.diag(torch.tensor([0.0, 1.0, 4.0, 0.0], **dd)).requires_grad_(
        True
    )

    norm = snorm(ovlp)
    assert norm.dtype == dtype
    expected_val = torch.tensor([0.0, 1.0, 0.5, 0.0], **dd)
    torch.testing.assert_close(norm, expected_val, atol=1e-6, rtol=1e-6)

    grad_1 = torch.autograd.grad(norm.sum(), ovlp, create_graph=True)[0]
    grad_2 = torch.autograd.grad(grad_1.sum(), ovlp)[0]
    assert torch.isfinite(grad_1).all()
    assert torch.isfinite(grad_2).all()

    expected_grad_diag = torch.tensor([0.0, -0.5, -0.5 * (4.0**-1.5), 0.0], **dd)
    torch.testing.assert_close(
        grad_1.diagonal(dim1=-1, dim2=-2),
        expected_grad_diag,
        atol=1e-6,
        rtol=1e-6,
    )

    def _norm_sum(mat: torch.Tensor) -> torch.Tensor:
        return normalize_integral_matrix(mat).sum()

    primal_out, tangent_out = jvp(
        _norm_sum, (ovlp.detach(),), (torch.eye(4, **dd),)
    )
    assert torch.isfinite(primal_out).all()
    assert torch.isfinite(tangent_out).all()

    neg_ovlp = torch.diag(torch.tensor([-1.0, 0.0, 1.0], **dd))
    neg_norm = snorm(neg_ovlp)
    assert torch.isnan(neg_norm[0])
    assert neg_norm[1].item() == 0.0
    assert neg_norm[2].item() == 1.0


def test_pure_dipole_shift_without_mutation() -> None:
    """Dipole origin shifts use explicit matrices and positions."""
    dd = {"device": DEVICE, "dtype": torch.double}
    overlap = torch.eye(3, **dd)
    dipole = torch.randn(3, 3, 3, **dd)
    positions = torch.randn(3, 3, **dd)
    originals = (dipole.clone(), overlap.clone(), positions.clone())

    shift_dipole_origin(dipole, overlap, positions)

    assert torch.equal(dipole, originals[0])
    assert torch.equal(overlap, originals[1])
    assert torch.equal(positions, originals[2])


@pytest.mark.parametrize(
    "uplo, indices",
    [("l", [0, 3, 4, 6, 7, 8]), ("u", [0, 1, 2, 4, 5, 8])],
)
def test_pure_quadrupole_reduction_preserves_component_order(
    uplo: str, indices: list[int]
) -> None:
    """Nine-to-six reduction preserves the selected triangular ordering."""
    raw = torch.arange(9 * 2 * 2, dtype=torch.double, device=DEVICE).reshape(
        9, 2, 2
    )
    original = raw.clone()

    result = reduce_quadrupole_9_to_6(raw, uplo=uplo)  # type: ignore[arg-type]
    assert torch.equal(result, raw[indices])
    assert torch.equal(raw, original)
    assert torch.equal(result, _reduce_9_to_6(raw, uplo=uplo))  # type: ignore[arg-type]


def test_pure_quadrupole_transforms_without_mutation() -> None:
    """Quadrupole reduction, shift and traceless conversion are pure."""
    dd = {"device": DEVICE, "dtype": torch.double}
    raw9 = torch.randn(9, 3, 3, **dd)
    raw6 = reduce_quadrupole_9_to_6(raw9)
    overlap = torch.eye(3, **dd)
    dipole = torch.randn(3, 3, 3, **dd)
    positions = torch.randn(3, 3, **dd)
    originals = tuple(
        x.clone() for x in (raw9, raw6, overlap, dipole, positions)
    )

    reduced = reduce_quadrupole_9_to_6(raw9)
    shifted = shift_quadrupole_origin(raw6, dipole, overlap, positions)
    shifted_from_9 = shift_quadrupole_origin(
        reduce_quadrupole_9_to_6(raw9), dipole, overlap, positions
    )
    shifted_before = shifted.clone()
    make_quadrupole_traceless(shifted)

    assert torch.allclose(shifted_from_9, shifted, atol=1e-14, rtol=1e-14)
    assert torch.equal(shifted, shifted_before)
    assert torch.equal(reduced, raw6)
    for tensor, original in zip(
        (raw9, raw6, overlap, dipole, positions), originals
    ):
        assert torch.equal(tensor, original)


def test_pure_transform_shape_errors_match_legacy_contract() -> None:
    """Pure transforms retain existing input shape validation."""
    dd = {"device": DEVICE, "dtype": torch.double}
    matrix = torch.eye(2, **dd)

    with pytest.raises(ValueError, match="Invalid norm shape"):
        normalize_integral_matrix(matrix, torch.ones(1, 2, 2, **dd))

    with pytest.raises(RuntimeError, match="orbital-resolution"):
        shift_dipole_origin(
            torch.zeros(3, 2, 2, **dd), matrix, torch.zeros(1, 3, **dd)
        )

    with pytest.raises(RuntimeError, match="shape"):
        make_quadrupole_traceless(torch.zeros(3, 2, 2, **dd))

    with pytest.raises(RuntimeError, match="shape"):
        make_quadrupole_traceless(torch.zeros(6, 2, **dd))

    with pytest.raises(RuntimeError, match="shape"):
        reduce_quadrupole_9_to_6(torch.zeros(6, 2, 2, **dd))

    for bad_rank in (torch.zeros(9, 2, **dd), torch.zeros(1, 1, 9, 2, 2, **dd)):
        with pytest.raises(RuntimeError, match="shape"):
            reduce_quadrupole_9_to_6(bad_rank)
        with pytest.raises(RuntimeError, match="shape"):
            _reduce_9_to_6(bad_rank)

    with pytest.raises(ValueError, match="uplo"):
        reduce_quadrupole_9_to_6(torch.zeros(9, 2, 2, **dd), uplo="x")  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="tensor of shape"):
        shift_quadrupole_origin(
            torch.zeros(5, 2, 2, **dd),
            torch.zeros(3, 2, 2, **dd),
            torch.eye(2, **dd),
            torch.zeros(2, 3, **dd),
        )

    with pytest.raises(RuntimeError, match="tensor of shape"):
        shift_quadrupole_origin(
            torch.zeros(1, 1, 6, 2, 2, **dd),
            torch.zeros(3, 2, 2, **dd),
            torch.eye(2, **dd),
            torch.zeros(2, 3, **dd),
        )


def _setup_and_inputs() -> (
    tuple[PytorchIntegralSetup, torch.Tensor, torch.Tensor, IndexHelper]
):
    """Create a reusable small PyTorch integral setup and geometry."""
    dd = {"device": DEVICE, "dtype": torch.double}
    sample = samples["H2O"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    basis = Basis(torch.unique(numbers), GFN1_XTB, ihelp, **dd)
    return setup_integrals(basis), numbers, positions, ihelp


def _evaluate_multipoles(
    setup: PytorchIntegralSetup,
    positions: torch.Tensor,
    ihelp: IndexHelper,
    *,
    reverse_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build and transform overlap and multipoles for one geometry."""
    overlap_raw = build_overlap(setup, positions)
    overlap_norm = snorm(overlap_raw)
    overlap = normalize_integral_matrix(overlap_raw, overlap_norm)

    if reverse_order:
        quadrupole_raw = build_quadrupole(setup, positions)
        dipole_raw = build_dipole(setup, positions)
    else:
        dipole_raw = build_dipole(setup, positions)
        quadrupole_raw = build_quadrupole(setup, positions)

    dipole_centered = normalize_integral_matrix(dipole_raw, overlap_norm)
    quadrupole_centered = normalize_integral_matrix(
        quadrupole_raw, overlap_norm
    )
    orbital_positions = ihelp.spread_atom_to_orbital(
        positions, dim=-2, extra=True
    )
    quadrupole_shifted = shift_quadrupole_origin(
        reduce_quadrupole_9_to_6(quadrupole_centered),
        dipole_centered,
        overlap,
        orbital_positions,
    )
    dipole = shift_dipole_origin(dipole_centered, overlap, orbital_positions)
    quadrupole = make_quadrupole_traceless(quadrupole_shifted)
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
