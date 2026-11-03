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
Trapezoidal Casimir-Polder frequency integrator (from ``dftd/integrator/trapezoid.f90``).
"""
from __future__ import annotations

import math

import torch
from tad_mctc.typing import Self, Tensor, get_default_dtype

__all__ = ["DEFAULT_TRAPZ_FREQUENCIES", "TrapezoidIntegrator"]

DEFAULT_TRAPZ_FREQUENCIES: list[float] = [
    0.000001,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
]


class TrapezoidIntegrator:
    """
    Trapezoidal rule integrator for Casimir-Polder dispersion coefficients.

    Matches ``dftd/integrator/trapezoid.f90``.
    """

    freq: Tensor
    """Imaginary frequency grid ``(ngrid,)``."""

    weights: Tensor
    """Trapezoidal quadrature weights ``(ngrid,)``."""

    cp_weights: Tensor
    """Casimir-Polder :math:`C_6` weights ``(3 / pi) * weights`` of shape ``(ngrid,)``."""

    def __init__(
        self,
        frequencies: list[float] | Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        dt = get_default_dtype() if dtype is None else dtype
        if isinstance(frequencies, Tensor):
            if device is None:
                device = frequencies.device
            if dtype is None:
                dt = frequencies.dtype
            freq_64 = frequencies.to(device=device, dtype=torch.float64)
        else:
            raw = DEFAULT_TRAPZ_FREQUENCIES if frequencies is None else frequencies
            freq_64 = torch.tensor(raw, device=device, dtype=torch.float64)

        if freq_64.ndim != 1 or freq_64.shape[0] < 2:
            raise ValueError("Frequency grid must be 1D with at least 2 points.")

        w_64 = torch.empty_like(freq_64)
        w_64[0] = 0.5 * (freq_64[1] - freq_64[0])
        if freq_64.shape[0] > 2:
            w_64[1:-1] = 0.5 * (freq_64[2:] - freq_64[:-2])
        w_64[-1] = 0.5 * (freq_64[-1] - freq_64[-2])

        cp_w_64 = (3.0 / math.pi) * w_64

        self.freq = freq_64.to(dtype=dt)
        self.weights = w_64.to(dtype=dt)
        self.cp_weights = cp_w_64.to(dtype=dt)

    @property
    def device(self) -> torch.device:
        """Device of the integrator tensors."""
        return self.freq.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype of the integrator tensors."""
        return self.freq.dtype

    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return a copy moved to ``device`` and/or ``dtype``."""
        new = object.__new__(type(self))
        new.freq = self.freq.to(device=device, dtype=dtype)
        new.weights = self.weights.to(device=device, dtype=dtype)
        new.cp_weights = self.cp_weights.to(device=device, dtype=dtype)
        return new

    def type(self, dtype: torch.dtype) -> Self:
        """Return a copy converted to ``dtype``."""
        return self.to(dtype=dtype)

    @property
    def ngrid(self) -> int:
        """Number of frequency points."""
        return int(self.freq.shape[0])

    def integrate(self, pol: Tensor, dim: int = -1) -> Tensor:
        """
        Integrate values ``pol`` over the frequency grid along ``dim`` using
        raw trapezoidal weights ``sum(pol * weights)``.
        """
        w = self.weights.to(device=pol.device, dtype=pol.dtype)
        if dim != -1 and dim != pol.ndim - 1:
            pol = pol.movedim(dim, -1)
        return torch.sum(pol * w, dim=-1)

    def integrate_c6(self, alpha_i: Tensor, alpha_j: Tensor) -> Tensor:
        """
        Compute Casimir-Polder :math:`C_6 = \\frac{3}{\\pi} \\int_0^\\infty \\alpha_i(i\\omega)\\alpha_j(i\\omega)\\,d\\omega`.
        """
        w = self.weights.to(device=alpha_i.device, dtype=alpha_i.dtype)
        return (3.0 / math.pi) * torch.sum(alpha_i * alpha_j * w, dim=-1)
