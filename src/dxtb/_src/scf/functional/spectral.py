# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Gauge-invariant spectral matrices for the functional SCF core."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dxtb._src.typing import Tensor

if TYPE_CHECKING:
    from dxtb._src.wavefunction.filling import _FermiCoreState

__all__ = ["aufbau_density_matrix", "fermi_density_matrix"]


def _active_mask(matrix: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones(
            matrix.shape[:-1], dtype=torch.bool, device=matrix.device
        )
    return torch.broadcast_to(
        mask.to(device=matrix.device, dtype=torch.bool), matrix.shape[:-1]
    )


def _prepare_matrix(
    matrix: Tensor, l_inv: Tensor | None, mask: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    """Orthogonalize a generalized problem and add a detached padding block."""
    matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    active = _active_mask(matrix, mask)
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    matrix = torch.where(pair, matrix, torch.zeros_like(matrix))
    if l_inv is not None:
        l_inv = torch.where(pair, l_inv, torch.zeros_like(l_inv))
        matrix = l_inv @ matrix @ l_inv.transpose(-2, -1)

    count = active.sum(-1).clamp_min(1).to(matrix.dtype)
    center = matrix.diagonal(dim1=-2, dim2=-1).sum(-1) / count
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    centered = matrix - center[..., None, None] * eye
    row_sum = centered.abs().sum(-1)
    bound = torch.where(active, row_sum, 0.0).amax(-1)
    sentinel = center.detach() + bound.detach() + 1.0
    diagonal = torch.where(active, 0.0, sentinel.unsqueeze(-1))
    spectral = matrix + torch.diag_embed(diagonal)
    return matrix, spectral, active


def _matrix_fermi(
    matrix: Tensor,
    shift: Tensor,
    kt: Tensor,
    active: Tensor,
    *,
    wide_range: bool,
) -> Tensor:
    """Evaluate the symmetric matrix Fermi function with fixed scaling."""
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    eye = eye.expand(matrix.shape)
    exponent = (matrix - shift[..., None, None] * eye) / kt
    # The existing ten-step path is accurate for float64 xTB ranges. In
    # float32, sub-microhartree temperatures can make its initial exponential
    # overflow, so use the smaller argument and twenty fixed tanh doublings.
    # The dtype choice is static configuration; both branches evaluate the
    # same matrix function without tensor-dependent host control.
    scale = 2_097_152.0 if wide_range else 2048.0
    steps = 20 if wide_range else 10
    scaled = exponent / scale
    exp_scaled = torch.matrix_exp(2.0 * scaled)
    tangent = torch.linalg.solve(exp_scaled + eye, exp_scaled - eye)
    for _ in range(steps):
        tangent = (
            2.0 * tangent @ torch.linalg.solve(eye + tangent @ tangent, eye)
        )
    density = 0.5 * (eye - tangent)
    density = 0.5 * (density + density.transpose(-2, -1))
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    return torch.where(pair, density, torch.zeros_like(density))


def _gap_electron_response(
    matrix: Tensor,
    spectral: Tensor,
    active: Tensor,
    below: Tensor,
    gap: Tensor,
    kt: Tensor,
) -> Tensor:
    """Build the stable fixed-count electron response inside a wide gap."""
    n = matrix.shape[-1]
    active_count = active.sum(-1)
    rank = below.sum(-1).to(torch.long)
    rank = torch.where(gap, rank, torch.zeros_like(rank))
    occupied = _projector(matrix, spectral, active, rank)
    occupied = torch.where(gap[..., None, None], occupied, 0.0)
    active_projector = torch.diag_embed(active.to(matrix.dtype))
    virtual = active_projector - occupied
    padded = (
        torch.eye(n, dtype=matrix.dtype, device=matrix.device)
        - active_projector
    )

    energies = torch.linalg.eigvalsh(spectral.detach())
    lower_idx = (rank - 1).clamp(0, n - 1)
    upper_idx = rank.clamp(0, n - 1)
    homo = energies.gather(-1, lower_idx.unsqueeze(-1)).squeeze(-1)
    lumo = energies.gather(-1, upper_idx.unsqueeze(-1)).squeeze(-1)
    scale = energies.abs().amax(-1) + homo.abs() + lumo.abs() + 1.0
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device).expand(
        matrix.shape
    )

    virtual_argument = (
        virtual @ (spectral - lumo[..., None, None] * eye) @ virtual
    )
    virtual_argument = virtual_argument + scale[..., None, None] * (
        occupied + padded
    )
    occupied_argument = (
        occupied @ (homo[..., None, None] * eye - spectral) @ occupied
    )
    occupied_argument = occupied_argument + scale[..., None, None] * (
        virtual + padded
    )

    virtual_tail = torch.matrix_exp(-virtual_argument / kt) @ virtual
    occupied_tail = torch.matrix_exp(-occupied_argument / kt) @ occupied
    virtual_sum = virtual_tail.diagonal(dim1=-2, dim2=-1).sum(-1)
    occupied_sum = occupied_tail.diagonal(dim1=-2, dim2=-1).sum(-1)
    tiny = torch.finfo(matrix.dtype).tiny
    virtual_part = virtual_tail / virtual_sum.clamp_min(tiny)[..., None, None]
    occupied_part = (
        occupied_tail / occupied_sum.clamp_min(tiny)[..., None, None]
    )
    has_both = (rank > 0) & (rank < active_count)
    response = torch.where(
        has_both[..., None, None],
        0.5 * (virtual_part + occupied_part),
        virtual_part + occupied_part,
    )
    return torch.where(gap[..., None, None], response, 0.0)


def fermi_density_matrix(
    hamiltonian: Tensor,
    nel: Tensor,
    kt: Tensor,
    state: _FermiCoreState,
    *,
    l_inv: Tensor | None = None,
    mask: Tensor | None = None,
    diff_steps: int = 2,
) -> Tensor:
    """Build an invariant Fermi density from dxtb's shared primal state."""
    work_dtype = (
        torch.float64
        if hamiltonian.dtype != torch.float64
        else hamiltonian.dtype
    )
    h = hamiltonian.to(dtype=work_dtype)
    li = None if l_inv is None else l_inv.to(dtype=work_dtype)
    structural_mask = state.orbital_mask
    if structural_mask is not None and structural_mask.ndim == h.ndim:
        structural_mask = structural_mask.select(-2, 0)
    if structural_mask is None:
        structural_mask = mask
    h, spectral, active = _prepare_matrix(h, li, structural_mask)

    h = h.unsqueeze(-3).expand(*nel.shape, h.shape[-2], h.shape[-1])
    spectral = spectral.unsqueeze(-3).expand_as(h)
    active = active.unsqueeze(-2).expand(*nel.shape, active.shape[-1])
    if li is not None:
        li = li.unsqueeze(-3).expand_as(h)
        pair = active.unsqueeze(-1) & active.unsqueeze(-2)
        li = torch.where(pair, li, torch.zeros_like(li))
    kt = kt.to(device=h.device, dtype=h.dtype)
    kt_pos = state.kt_positive.to(device=h.device)
    safe_kt = torch.where(kt_pos, kt, torch.ones_like(kt))
    ref = state.chemical_potential_ref.to(device=h.device, dtype=h.dtype)
    offset = state.chemical_potential_offset.to(device=h.device, dtype=h.dtype)
    ref = ref.squeeze(-1)
    offset = offset.squeeze(-1)
    channel_shape = (*nel.shape, 1, 1)
    eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
    relative = spectral - ref.reshape(channel_shape) * eye
    relative = relative - offset.reshape(channel_shape) * eye
    target = nel.to(device=h.device, dtype=h.dtype)
    empty = state.empty.to(device=h.device)
    full = state.full.to(device=h.device)
    too_many = state.too_many.to(device=h.device)
    active_channel = state.active.to(device=h.device)
    gap_mask = state.gap.to(device=h.device, dtype=torch.bool)
    below = state.below.to(device=h.device, dtype=torch.bool)
    diff_mask = active_channel & ~gap_mask & ~empty & ~full & ~too_many
    shift = torch.zeros_like(target)
    for _ in range(diff_steps):
        density = _matrix_fermi(
            relative,
            shift,
            safe_kt,
            active,
            wide_range=hamiltonian.dtype != torch.float64,
        )
        count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
        slope = (density - density @ density).diagonal(dim1=-2, dim2=-1).sum(
            -1
        ) / safe_kt
        safe_slope = torch.where(
            slope.abs() > torch.finfo(h.dtype).tiny, slope, 1.0
        )
        correction = torch.where(diff_mask, (target - count) / safe_slope, 0.0)
        shift = shift + correction

    density = _matrix_fermi(
        relative,
        shift,
        safe_kt,
        active,
        wide_range=hamiltonian.dtype != torch.float64,
    )
    eye_channel = eye * active.unsqueeze(-1).to(h.dtype)
    density = torch.where(empty[..., None, None], 0.0, density)
    density = torch.where(too_many[..., None, None], 0.0, density)
    density = torch.where(full[..., None, None], eye_channel, density)
    response = _gap_electron_response(
        h,
        spectral,
        active,
        state.below.to(device=h.device, dtype=torch.bool),
        gap_mask,
        safe_kt,
    )
    density = density + (target - target.detach())[..., None, None] * response
    if li is not None:
        density = li.transpose(-2, -1) @ density @ li
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    density = torch.where(pair, density, torch.zeros_like(density))
    return density.sum(-3).to(dtype=hamiltonian.dtype)


def _projector(
    matrix: Tensor,
    spectral: Tensor,
    active: Tensor,
    rank: Tensor,
) -> Tensor:
    """Evaluate a fixed-rank projector using a matrix-sign iteration."""
    n = matrix.shape[-1]
    active_count = active.sum(-1)
    evals = torch.linalg.eigvalsh(spectral.detach())
    lo_idx = (rank - 1).clamp(0, n - 1)
    hi_idx = rank.clamp(0, n - 1)
    e_lo = evals.gather(-1, lo_idx.unsqueeze(-1)).squeeze(-1)
    e_hi = evals.gather(-1, hi_idx.unsqueeze(-1)).squeeze(-1)
    e_first = evals[..., 0]
    e_last = evals.gather(
        -1, (active_count - 1).clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    span = (e_last - e_first).abs().clamp_min(torch.finfo(matrix.dtype).tiny)
    e_lo = torch.where(rank == 0, e_first - span, e_lo)
    e_hi = torch.where(rank == 0, e_first, e_hi)
    e_lo = torch.where(rank == active_count, e_last, e_lo)
    e_hi = torch.where(rank == active_count, e_last + span, e_hi)
    mu = (0.5 * (e_lo + e_hi)).detach()

    scale = torch.maximum(
        (evals - mu.unsqueeze(-1)).abs().masked_fill(~active, 0.0).amax(-1),
        span * 0.5,
    ).clamp_min(torch.finfo(matrix.dtype).tiny)
    pad = torch.where(active, 0.0, (mu + 2.0 * scale).unsqueeze(-1))
    h = matrix + torch.diag_embed(pad)
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device).expand(h.shape)
    x = (h - mu[..., None, None] * eye) / scale[..., None, None]
    # Newton-Schulz sign iteration. The fixed iteration count avoids
    # data-dependent control flow, and the differentiable operations are
    # matrix functions.
    for _ in range(12):
        inverse = torch.linalg.solve(x, eye)
        x_norm = torch.linalg.matrix_norm(x, ord="fro", dim=(-2, -1))
        inverse_norm = torch.linalg.matrix_norm(
            inverse, ord="fro", dim=(-2, -1)
        )
        gamma = torch.sqrt(
            inverse_norm / x_norm.clamp_min(torch.finfo(x.dtype).tiny)
        )[..., None, None]
        x = 0.5 * (gamma * x + inverse / gamma)
    result = 0.5 * (eye - x)
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    return torch.where(pair, result, torch.zeros_like(result))


def aufbau_density_matrix(
    hamiltonian: Tensor,
    nel: Tensor,
    *,
    l_inv: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """Build a spin-channel Aufbau density without eigenvector derivatives.

    The detached integer rank selects the structural occupied subspace. The
    fractional remainder remains differentiable and weights the isolated
    frontier projector, preserving dxtb's fractional-electron convention.
    An exact degeneracy across differently occupied states is outside the
    differentiable contract: the matrix-sign projector becomes singular and
    its ``torch.linalg.solve`` raises rather than manufacturing a response.
    """
    work_dtype = (
        torch.float64
        if hamiltonian.dtype != torch.float64
        else hamiltonian.dtype
    )
    h = hamiltonian.to(dtype=work_dtype)
    li = None if l_inv is None else l_inv.to(dtype=work_dtype)
    h, spectral, active = _prepare_matrix(h, li, mask)

    h = h.unsqueeze(-3).expand(*nel.shape, h.shape[-2], h.shape[-1])
    spectral = spectral.unsqueeze(-3).expand_as(h)
    active = active.unsqueeze(-2).expand(*nel.shape, active.shape[-1])
    if li is not None:
        li = li.unsqueeze(-3).expand_as(h)
        pair = active.unsqueeze(-1) & active.unsqueeze(-2)
        li = torch.where(pair, li, torch.zeros_like(li))
    active_count = active.sum(-1)
    count = active_count.to(dtype=h.dtype)
    electron_count = nel.to(dtype=h.dtype, device=h.device)
    effective_nel = torch.where(
        electron_count < 0.0,
        0.0,
        torch.where(electron_count >= count, count, electron_count),
    )
    rank = effective_nel.detach().floor().to(torch.long)
    rank = torch.minimum(rank, active_count)
    fraction = effective_nel - rank.to(h.dtype)
    next_rank = torch.minimum(rank + 1, active_count)
    occupied = _projector(h, spectral, active, rank)
    frontier = _projector(h, spectral, active, next_rank)
    density = occupied + fraction[..., None, None] * (frontier - occupied)
    if li is not None:
        density = li.transpose(-2, -1) @ density @ li
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    density = torch.where(pair, density, torch.zeros_like(density))
    return density.sum(-3).to(dtype=hamiltonian.dtype)
