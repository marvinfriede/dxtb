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
Gauss-Legendre Casimir-Polder frequency integrator (from ``dftd/integrator/gauss_legendre.f90``).
"""
from __future__ import annotations

import math

import torch
from tad_mctc.typing import Self, Tensor, get_default_dtype

__all__ = [
    "GAUSS_LEGENDRE_NODES_10",
    "GAUSS_LEGENDRE_WEIGHTS_10",
    "GaussLegendreIntegrator",
    "gauss_legendre_1d",
]

GAUSS_LEGENDRE_NODES_10: list[float] = [
    -0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
    0.1488743389816312,
    0.4333953941292472,
    0.6794095682990244,
    0.8650633666889845,
    0.9739065285171717,
]

GAUSS_LEGENDRE_WEIGHTS_10: list[float] = [
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159821,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159821,
    0.1494513491505806,
    0.0666713443086881,
]


def _legendre_eval(n: int, z: float) -> tuple[float, float]:
    """Evaluate Legendre polynomial P_n(z) and its derivative P'_n(z)."""
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return z, 1.0

    p0 = 1.0
    p1 = z
    for j in range(2, n + 1):
        jf = float(j)
        p2 = ((2.0 * jf - 1.0) * z * p1 - (jf - 1.0) * p0) / jf
        p0 = p1
        p1 = p2

    p = p1
    pm1 = p0
    dp = float(n) * (z * p - pm1) / (z * z - 1.0)
    return p, dp


def gauss_legendre_1d(
    ngrid: int = 10,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Compute standard Gauss-Legendre nodes and weights on ``[-1, 1]`` matching
    ``dftd/integrator/gauss_legendre.f90``.
    """
    if ngrid <= 0:
        raise ValueError(f"ngrid must be positive, got {ngrid}.")

    dt = get_default_dtype() if dtype is None else dtype
    if ngrid == 10:
        nodes = torch.tensor(GAUSS_LEGENDRE_NODES_10, device=device, dtype=dt)
        weights = torch.tensor(GAUSS_LEGENDRE_WEIGHTS_10, device=device, dtype=dt)
        return nodes, weights

    tol = 10.0 * 2.220446049250313e-16
    max_iter = 100
    x = [0.0] * ngrid
    w = [0.0] * ngrid
    ngrid_pos = (ngrid + 1) // 2

    for igrid in range(1, ngrid_pos + 1):
        z = math.cos(math.pi * (float(igrid) - 0.25) / (float(ngrid) + 0.5))
        converged = False
        for _ in range(max_iter):
            pol, dpol = _legendre_eval(ngrid, z)
            zold = z
            z = zold - pol / dpol
            if abs(z - zold) <= tol:
                converged = True
                break
        if not converged:
            raise RuntimeError("Gauss-Legendre root iteration failed.")

        _, dpol = _legendre_eval(ngrid, z)
        x[igrid - 1] = -z
        x[ngrid - igrid] = z
        weight = 2.0 / ((1.0 - z * z) * dpol * dpol)
        w[igrid - 1] = weight
        w[ngrid - igrid] = weight

    return (
        torch.tensor(x, device=device, dtype=dt),
        torch.tensor(w, device=device, dtype=dt),
    )


class GaussLegendreIntegrator:
    """
    Gauss-Legendre quadrature on ``[0, inf)`` via the rational transformation

    .. math::

        \\omega(t) = \\omega_0 \\frac{1 + t}{1 - t}, \\quad
        W(t) = w(t) \\frac{2 \\omega_0}{(1 - t)^2}

    Matches ``dftd/integrator/gauss_legendre.f90``.
    """

    nodes: Tensor
    """Standard Gauss-Legendre nodes on ``[-1, 1]``."""

    raw_weights: Tensor
    """Standard Gauss-Legendre weights on ``[-1, 1]``."""

    freq: Tensor
    """Transformed imaginary frequencies on ``[0, inf)``."""

    weights: Tensor
    """Transformed quadrature weights on ``[0, inf)``."""

    scale: Tensor
    """Frequency scaling factor ``omega0``."""

    def __init__(
        self,
        ngrid: int = 10,
        scale: float | Tensor = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        dt = get_default_dtype() if dtype is None else dtype
        if isinstance(scale, Tensor):
            if device is None:
                device = scale.device
            if dtype is None:
                dt = scale.dtype
            omega0_64 = scale.to(device=device, dtype=torch.float64)
        else:
            omega0_64 = torch.tensor(scale, device=device, dtype=torch.float64)

        t_64, w_64 = gauss_legendre_1d(ngrid=ngrid, device=device, dtype=torch.float64)
        freq_64 = omega0_64 * (1.0 + t_64) / (1.0 - t_64)
        weights_64 = w_64 * 2.0 * omega0_64 / ((1.0 - t_64) ** 2)

        self.nodes = t_64.to(dtype=dt)
        self.raw_weights = w_64.to(dtype=dt)
        self.freq = freq_64.to(dtype=dt)
        self.weights = weights_64.to(dtype=dt)
        self.scale = omega0_64.to(dtype=dt)

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
        new.nodes = self.nodes.to(device=device, dtype=dtype)
        new.raw_weights = self.raw_weights.to(device=device, dtype=dtype)
        new.freq = self.freq.to(device=device, dtype=dtype)
        new.weights = self.weights.to(device=device, dtype=dtype)
        new.scale = self.scale.to(device=device, dtype=dtype)
        return new

    def type(self, dtype: torch.dtype) -> Self:
        """Return a copy converted to ``dtype``."""
        return self.to(dtype=dtype)

    @property
    def ngrid(self) -> int:
        """Number of quadrature points."""
        return int(self.freq.shape[0])

    def integrate(self, pol: Tensor, dim: int = -1) -> Tensor:
        """
        Integrate ``pol`` over ``[0, inf)`` using transformed weights ``sum(pol * weights)``.
        """
        w = self.weights.to(device=pol.device, dtype=pol.dtype)
        if dim != -1 and dim != pol.ndim - 1:
            pol = pol.movedim(dim, -1)
        return torch.sum(pol * w, dim=-1)

    def integrate_c6(self, alpha_dd_i: Tensor, alpha_dd_j: Tensor) -> Tensor:
        """
        Casimir-Polder :math:`C_6 = \\frac{3}{\\pi} \\int_0^\\infty \\alpha_{dd,i}(i\\omega)\\alpha_{dd,j}(i\\omega)\\,d\\omega`.
        """
        w = self.weights.to(device=alpha_dd_i.device, dtype=alpha_dd_i.dtype)
        return (3.0 / math.pi) * torch.sum(alpha_dd_i * alpha_dd_j * w, dim=-1)

    def integrate_c8(
        self,
        alpha_dd_i: Tensor,
        alpha_qq_i: Tensor,
        alpha_dd_j: Tensor,
        alpha_qq_j: Tensor,
    ) -> Tensor:
        """
        Casimir-Polder :math:`C_8 = \\frac{15}{2\\pi} \\int_0^\\infty (\\alpha_{dd,i}\\alpha_{qq,j} + \\alpha_{qq,i}\\alpha_{dd,j})\\,d\\omega`.
        """
        w = self.weights.to(device=alpha_dd_i.device, dtype=alpha_dd_i.dtype)
        integrand = alpha_dd_i * alpha_qq_j + alpha_qq_i * alpha_dd_j
        return (15.0 / (2.0 * math.pi)) * torch.sum(integrand * w, dim=-1)

    def integrate_c9(
        self,
        alpha_dd_i: Tensor,
        alpha_dd_j: Tensor,
        alpha_dd_k: Tensor,
    ) -> Tensor:
        """
        Casimir-Polder :math:`C_9 = \\frac{3}{\\pi} \\int_0^\\infty \\alpha_{dd,i}\\alpha_{dd,j}\\alpha_{dd,k}\\,d\\omega`.
        """
        w = self.weights.to(device=alpha_dd_i.device, dtype=alpha_dd_i.dtype)
        return (3.0 / math.pi) * torch.sum(
            alpha_dd_i * alpha_dd_j * alpha_dd_k * w, dim=-1
        )
