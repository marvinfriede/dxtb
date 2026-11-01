# This file is part of tad-dftd.
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
Tests for Casimir-Polder frequency integrators in ``tad_dftd.integrator``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tad_dftd.integrator import (
    DEFAULT_TRAPZ_FREQUENCIES,
    GaussLegendreIntegrator,
    TrapezoidIntegrator,
    gauss_legendre_1d,
)


def test_trapezoid_weights_and_c6() -> None:
    """Verify 23-point trapezoidal frequency grid and Casimir-Polder weights."""
    integ = TrapezoidIntegrator(dtype=torch.float64)
    assert integ.ngrid == 23
    assert integ.freq.shape == (23,)
    assert integ.weights.shape == (23,)
    assert integ.cp_weights.shape == (23,)

    # Sum of trapezoidal weights on [f_0, f_22] must equal f_22 - f_0
    span = DEFAULT_TRAPZ_FREQUENCIES[-1] - DEFAULT_TRAPZ_FREQUENCIES[0]
    assert pytest.approx(integ.weights.sum().item(), rel=1e-12) == span

    # Constant polarizability alpha(iw) = 1 -> integral = span, C6 = (3/pi) * span
    ones = torch.ones((4, 23), dtype=torch.float64)
    val = integ.integrate(ones)
    assert torch.allclose(val, torch.full((4,), span, dtype=torch.float64))

    c6 = integ.integrate_c6(ones, ones)
    assert torch.allclose(
        c6, torch.full((4,), (3.0 / math.pi) * span, dtype=torch.float64)
    )


def test_gauss_legendre_1d_and_casimir_polder() -> None:
    """
    Verify Gauss-Legendre nodes/weights on [-1, 1] and rational transformation
    on [0, inf) against exact Casimir-Polder Lorentz oscillator integrals.
    """
    x10, w10 = gauss_legendre_1d(10, dtype=torch.float64)
    assert x10.shape == (10,)
    assert w10.shape == (10,)
    # Integral of 1 on [-1, 1] is 2
    assert pytest.approx(w10.sum().item(), rel=1e-14) == 2.0
    # Nodes are symmetric around 0
    assert torch.allclose(x10, -x10.flip(0), atol=1e-15)
    assert torch.allclose(w10, w10.flip(0), atol=1e-15)

    # Also test dynamic root finder for ngrid=12
    x12, w12 = gauss_legendre_1d(12, dtype=torch.float64)
    assert pytest.approx(w12.sum().item(), rel=1e-14) == 2.0
    # Integrate x^4 on [-1, 1] -> 2/5 = 0.4
    assert pytest.approx(torch.sum(w12 * (x12**4)).item(), rel=1e-14) == 0.4

    # Analytical Casimir-Polder integral for single-pole Lorentz polarizabilities:
    # alpha_i(iw) = a_i / (1 + (w / w_i)^2)
    # Exact: (3/pi) \int_0^\infty alpha_i(iw) alpha_j(iw) dw
    #      = (3/2) * a_i * a_j * (w_i * w_j) / (w_i + w_j)
    integ = GaussLegendreIntegrator(ngrid=10, scale=0.151295, dtype=torch.float64)
    w = integ.freq  # (10,)
    a_i, w_i = 4.5, 0.6
    a_j, w_j = 3.2, 0.8
    alpha_i = a_i / (1.0 + (w / w_i) ** 2)
    alpha_j = a_j / (1.0 + (w / w_j) ** 2)

    c6_num = integ.integrate_c6(alpha_i, alpha_j)
    c6_exact = 1.5 * a_i * a_j * (w_i * w_j) / (w_i + w_j)
    assert pytest.approx(c6_num.item(), rel=1e-4) == c6_exact

    # 20-point Gauss-Legendre converges to < 1e-9 on the same integral
    integ20 = GaussLegendreIntegrator(ngrid=20, scale=0.6, dtype=torch.float64)
    w20 = integ20.freq
    c6_num20 = integ20.integrate_c6(
        a_i / (1.0 + (w20 / w_i) ** 2),
        a_j / (1.0 + (w20 / w_j) ** 2),
    )
    assert pytest.approx(c6_num20.item(), rel=1e-9) == c6_exact

    # Test C8 and C9 helpers
    c8_num = integ.integrate_c8(alpha_i, 2.0 * alpha_i, alpha_j, 3.0 * alpha_j)
    c8_expected = (15.0 / (2.0 * math.pi)) * (math.pi / 3.0) * (3.0 + 2.0) * c6_num
    assert pytest.approx(c8_num.item(), rel=1e-14) == c8_expected.item()

    c9_num = integ.integrate_c9(alpha_i, alpha_j, alpha_i)
    assert c9_num.item() > 0.0
