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
Test wrappers for integrals.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Param
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.utils import normalize_integral_matrix, snorm
from dxtb.integrals import wrappers

numbers = torch.tensor([14, 1, 1, 1, 1])
positions = torch.tensor(
    [
        [+0.00000000000000, +0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ],
    dtype=torch.float64,
)

h2o_numbers = torch.tensor([8, 1, 1])
h2o_positions = torch.tensor(
    [
        [0.00000000000000, 0.00000000000000, 0.22170000000000],
        [0.00000000000000, 1.43090000000000, -0.88680000000000],
        [0.00000000000000, -1.43090000000000, -0.88680000000000],
    ],
    dtype=torch.float64,
)


@pytest.mark.parametrize("par", [GFN1_XTB, GFN2_XTB])
def test_fail(par: Param) -> None:
    """Invalid parametrization metadata and unknown drivers raise ValueError."""
    with pytest.raises(ValueError):
        _par = par.model_copy(deep=True)
        _par.meta = None
        wrappers.hcore(numbers, positions, _par)

    with pytest.raises(ValueError):
        _par = par.model_copy(deep=True)
        assert _par.meta is not None

        _par.meta.name = None
        wrappers.hcore(numbers, positions, _par)

    with pytest.raises(ValueError):
        _par = par.model_copy(deep=True)
        assert _par.meta is not None

        _par.meta.name = "fail"
        wrappers.hcore(numbers, positions, _par)

    with pytest.raises(ValueError):
        wrappers.overlap(numbers, positions, par, driver=-1)


@pytest.mark.parametrize(
    "fn", [wrappers.overlap, wrappers.dipint, wrappers.quadint]
)
def test_batch_input_validation_errors(fn) -> None:
    """Incompatible batch/atom shapes and invalid atomic numbers fail explicitly."""
    # Mismatched batch dimension (must not silently truncate via zip)
    with pytest.raises(ValueError, match="Expected positions of shape"):
        fn(
            torch.stack([h2o_numbers, h2o_numbers]),
            torch.stack([h2o_positions, h2o_positions, h2o_positions]),
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )

    # Mismatched atom dimension (must not silently slice positions)
    with pytest.raises(ValueError, match="Expected positions of shape"):
        fn(
            torch.stack([h2o_numbers, h2o_numbers]),
            torch.zeros((2, 4, 3), dtype=torch.float64),
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )

    # Mismatched single-structure atom count
    with pytest.raises(ValueError, match="Expected positions of shape"):
        fn(
            h2o_numbers,
            positions,
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )

    # Wrong coordinate dimension
    with pytest.raises(ValueError, match="Expected positions of shape"):
        fn(
            h2o_numbers,
            h2o_positions[:, :2],
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )

    # Unsupported 3D atomic numbers tensor
    with pytest.raises(ValueError, match="Expected 1D or 2D"):
        fn(
            h2o_numbers.unsqueeze(0).unsqueeze(0),
            h2o_positions.unsqueeze(0).unsqueeze(0),
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )

    # Negative atomic number
    with pytest.raises(ValueError, match="non-negative"):
        fn(
            torch.tensor([8, -1, 1]),
            h2o_positions,
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )

    # All-zero padding lane in batch
    with pytest.raises(ValueError, match="at least one positive"):
        fn(
            torch.tensor([[8, 1, 1], [0, 0, 0]]),
            torch.stack([h2o_positions, torch.zeros_like(h2o_positions)]),
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )


def test_hcore_rejects_batched_or_mismatched_inputs() -> None:
    """hcore validates single-structure layout and matching coordinates."""
    with pytest.raises(ValueError, match="Expected 1D"):
        wrappers.hcore(
            torch.stack([h2o_numbers, h2o_numbers]),
            torch.stack([h2o_positions, h2o_positions]),
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )

    with pytest.raises(ValueError, match="Expected positions of shape"):
        wrappers.hcore(
            h2o_numbers,
            positions,
            GFN1_XTB,
            driver=labels.INTDRIVER_PYTORCH,
        )


def test_h0_gfn1() -> None:
    """Single-structure GFN1-xTB core Hamiltonian shape."""
    h0 = wrappers.hcore(numbers, positions, GFN1_XTB)
    assert h0.shape == (17, 17)


def test_h0_gfn2() -> None:
    """Single-structure GFN2-xTB core Hamiltonian shape."""
    h0 = wrappers.hcore(numbers, positions, GFN2_XTB)
    assert h0.shape == (13, 13)


def test_overlap() -> None:
    """Single-structure overlap matrix shapes for GFN1 and GFN2."""
    s = wrappers.overlap(numbers, positions, GFN1_XTB)
    assert s.shape == (17, 17)

    s = wrappers.overlap(numbers, positions, GFN2_XTB)
    assert s.shape == (13, 13)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_dipole() -> None:
    """Single-structure dipole integral shapes for GFN1 and GFN2."""
    s = wrappers.dipint(numbers, positions, GFN1_XTB)
    assert s.shape == (3, 17, 17)

    s = wrappers.dipint(numbers, positions, GFN2_XTB)
    assert s.shape == (3, 13, 13)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_quad() -> None:
    """Single-structure quadrupole integral shapes for GFN1 and GFN2."""
    s = wrappers.quadint(numbers, positions, GFN1_XTB)
    assert s.shape == (9, 17, 17)

    s = wrappers.quadint(numbers, positions, GFN2_XTB)
    assert s.shape == (9, 13, 13)


@pytest.mark.parametrize(
    "driver",
    [
        labels.INTDRIVER_PYTORCH,
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "fn", [wrappers.overlap, wrappers.dipint, wrappers.quadint]
)
def test_trailing_and_noncontiguous_padding_batch_agreement_and_grad(
    driver: int, fn
) -> None:
    """Batched wrappers handle trailing and non-contiguous padding with exact gradients."""
    # Lane 0: SiH4 (5 atoms, 9 orbitals in GFN2)
    # Lane 1: H2O trailing padding [8, 1, 1, 0, 0] (3 atoms, 6 orbitals in GFN2)
    # Lane 2: H2O non-contiguous padding [8, 0, 1, 0, 1] (3 atoms at 0, 2, 4)
    batch_numbers = torch.tensor(
        [
            [14, 1, 1, 1, 1],
            [8, 1, 1, 0, 0],
            [8, 0, 1, 0, 1],
        ]
    )
    bogus_a = torch.tensor([91.0, -37.0, 44.0], dtype=torch.float64)
    bogus_b = torch.tensor([-53.0, 82.0, -19.0], dtype=torch.float64)
    lane1_pos = torch.stack(
        [
            h2o_positions[0],
            h2o_positions[1],
            h2o_positions[2],
            bogus_a,
            bogus_b,
        ]
    )
    lane2_pos = torch.stack(
        [
            h2o_positions[0],
            bogus_a,
            h2o_positions[1],
            bogus_b,
            h2o_positions[2],
        ]
    )
    batch_pos = (
        torch.stack([positions, lane1_pos, lane2_pos])
        .clone()
        .requires_grad_(True)
    )

    sih4_pos = positions.clone().requires_grad_(True)
    h2o_pos = h2o_positions.clone().requires_grad_(True)

    ref_sih4 = fn(numbers, sih4_pos, GFN2_XTB, driver=driver)
    ref_h2o = fn(h2o_numbers, h2o_pos, GFN2_XTB, driver=driver)
    out_batch = fn(batch_numbers, batch_pos, GFN2_XTB, driver=driver)

    nao_sih4 = ref_sih4.shape[-1]
    nao_h2o = ref_h2o.shape[-1]
    assert nao_sih4 == 13
    assert nao_h2o == 6

    torch.testing.assert_close(
        out_batch[0], ref_sih4, atol=1.0e-13, rtol=1.0e-13
    )
    torch.testing.assert_close(
        out_batch[1, ..., :nao_h2o, :nao_h2o],
        ref_h2o,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    torch.testing.assert_close(
        out_batch[2, ..., :nao_h2o, :nao_h2o],
        ref_h2o,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    assert torch.all(out_batch[1, ..., nao_h2o:, :] == 0.0)
    assert torch.all(out_batch[1, ..., :, nao_h2o:] == 0.0)
    assert torch.all(out_batch[2, ..., nao_h2o:, :] == 0.0)
    assert torch.all(out_batch[2, ..., :, nao_h2o:] == 0.0)

    # First derivative w.r.t. positions must match single-system gradients on
    # active atoms and vanish on padded atoms.
    weights = torch.arange(
        1, out_batch.shape[-2] * out_batch.shape[-1] + 1, dtype=torch.float64
    ).reshape(out_batch.shape[-2], out_batch.shape[-1])
    w_h2o = weights[:nao_h2o, :nao_h2o]

    (grad_batch,) = torch.autograd.grad((out_batch * weights).sum(), batch_pos)
    (grad_sih4,) = torch.autograd.grad((ref_sih4 * weights).sum(), sih4_pos)
    (grad_h2o,) = torch.autograd.grad((ref_h2o * w_h2o).sum(), h2o_pos)

    torch.testing.assert_close(
        grad_batch[0], grad_sih4, atol=1.0e-12, rtol=1.0e-12
    )
    torch.testing.assert_close(
        grad_batch[1, :3], grad_h2o, atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.all(grad_batch[1, 3:] == 0.0)
    torch.testing.assert_close(
        grad_batch[2, [0, 2, 4]], grad_h2o, atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.all(grad_batch[2, [1, 3]] == 0.0)


def test_hcore_noncontiguous_padding_and_grad() -> None:
    """hcore strips non-contiguous padding and preserves position derivatives."""
    padded_numbers = torch.tensor([8, 0, 1, 0, 1])
    padded_pos = torch.stack(
        [
            h2o_positions[0],
            torch.tensor([12.0, -34.0, 56.0], dtype=torch.float64),
            h2o_positions[1],
            torch.tensor([-78.0, 90.0, -11.0], dtype=torch.float64),
            h2o_positions[2],
        ]
    ).requires_grad_(True)
    clean_pos = h2o_positions.clone().requires_grad_(True)

    h_pad = wrappers.hcore(
        padded_numbers, padded_pos, GFN1_XTB, driver=labels.INTDRIVER_PYTORCH
    )
    h_ref = wrappers.hcore(
        h2o_numbers, clean_pos, GFN1_XTB, driver=labels.INTDRIVER_PYTORCH
    )
    torch.testing.assert_close(h_pad, h_ref, atol=1.0e-13, rtol=1.0e-13)

    (g_pad,) = torch.autograd.grad(h_pad.sum(), padded_pos)
    (g_ref,) = torch.autograd.grad(h_ref.sum(), clean_pos)
    torch.testing.assert_close(
        g_pad[[0, 2, 4]], g_ref, atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.all(g_pad[[1, 3]] == 0.0)


def test_normalization_origin_shift_and_quadrupole_ordering() -> None:
    """Verify normalization flag, origin-centered multipoles, and 9-component ordering."""
    s_norm = wrappers.overlap(
        h2o_numbers,
        h2o_positions,
        GFN2_XTB,
        driver=labels.INTDRIVER_PYTORCH,
        normalize=True,
    )
    s_raw = wrappers.overlap(
        h2o_numbers,
        h2o_positions,
        GFN2_XTB,
        driver=labels.INTDRIVER_PYTORCH,
        normalize=False,
    )
    torch.testing.assert_close(
        torch.diagonal(s_norm),
        torch.ones(s_norm.shape[-1], dtype=torch.float64),
        atol=1.0e-13,
        rtol=0.0,
    )
    torch.testing.assert_close(
        s_norm,
        normalize_integral_matrix(s_raw, snorm(s_raw)),
        atol=1.0e-13,
        rtol=0.0,
    )

    d_norm = wrappers.dipint(
        h2o_numbers,
        h2o_positions,
        GFN2_XTB,
        driver=labels.INTDRIVER_PYTORCH,
        normalize=True,
    )
    d_raw = wrappers.dipint(
        h2o_numbers,
        h2o_positions,
        GFN2_XTB,
        driver=labels.INTDRIVER_PYTORCH,
        normalize=False,
    )
    torch.testing.assert_close(
        d_norm,
        normalize_integral_matrix(d_raw, snorm(s_raw)),
        atol=1.0e-13,
        rtol=0.0,
    )

    q_norm = wrappers.quadint(
        h2o_numbers,
        h2o_positions,
        GFN2_XTB,
        driver=labels.INTDRIVER_PYTORCH,
        normalize=True,
    )
    # 9-component Cartesian ordering (xx, xy, xz, yx, yy, yz, zx, zy, zz)
    assert q_norm.shape == (9, 6, 6)
    for a in range(3):
        for b in range(3):
            torch.testing.assert_close(
                q_norm[3 * a + b], q_norm[3 * b + a], atol=1.0e-13, rtol=0.0
            )

    # Origin-centered translation relation: r -> r + t
    shift = torch.tensor([0.35, -0.8, 1.15], dtype=torch.float64)
    shifted_pos = h2o_positions + shift
    d_shifted = wrappers.dipint(
        h2o_numbers,
        shifted_pos,
        GFN2_XTB,
        driver=labels.INTDRIVER_PYTORCH,
    )
    q_shifted = wrappers.quadint(
        h2o_numbers,
        shifted_pos,
        GFN2_XTB,
        driver=labels.INTDRIVER_PYTORCH,
    )
    torch.testing.assert_close(
        d_shifted,
        d_norm + shift[:, None, None] * s_norm,
        atol=1.0e-12,
        rtol=0.0,
    )
    for a in range(3):
        for b in range(3):
            expected_q = (
                q_norm[3 * a + b]
                + shift[a] * d_norm[b]
                + shift[b] * d_norm[a]
                + shift[a] * shift[b] * s_norm
            )
            torch.testing.assert_close(
                q_shifted[3 * a + b], expected_q, atol=1.0e-12, rtol=0.0
            )


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize(
    "fn", [wrappers.overlap, wrappers.dipint, wrappers.quadint, wrappers.hcore]
)
def test_pytorch_libcint_parity_and_first_derivative(fn) -> None:
    """PyTorch and libcint wrapper values and position gradients agree."""
    pos_pt = h2o_positions.clone().requires_grad_(True)
    pos_lc = h2o_positions.clone().requires_grad_(True)

    out_pt = fn(h2o_numbers, pos_pt, GFN2_XTB, driver=labels.INTDRIVER_PYTORCH)
    out_lc = fn(h2o_numbers, pos_lc, GFN2_XTB, driver=labels.INTDRIVER_LIBCINT)
    torch.testing.assert_close(out_pt, out_lc, atol=1.0e-11, rtol=1.0e-11)

    weights = torch.linspace(
        0.1, 1.0, out_pt.numel(), dtype=torch.float64
    ).reshape(out_pt.shape)
    (grad_pt,) = torch.autograd.grad((out_pt * weights).sum(), pos_pt)
    (grad_lc,) = torch.autograd.grad((out_lc * weights).sum(), pos_lc)
    torch.testing.assert_close(grad_pt, grad_lc, atol=1.0e-10, rtol=1.0e-10)
