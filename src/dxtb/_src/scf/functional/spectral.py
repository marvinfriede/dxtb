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

import numpy as np
import torch

from dxtb._src.typing import Tensor

if TYPE_CHECKING:
    from dxtb._src.wavefunction.filling import _FermiCoreState

__all__ = [
    "aufbau_density_matrix",
    "aufbau_free_energy_matrix",
    "fermi_density_matrix",
    "fermi_free_energy_matrix",
]

_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(512)
_GL_NODES_NP = 0.5 * (_GL_NODES_NP + 1.0)
_GL_WEIGHTS_NP = 0.5 * _GL_WEIGHTS_NP


def _active_mask(matrix: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones(
            matrix.shape[:-1], dtype=torch.bool, device=matrix.device
        )
    return torch.broadcast_to(
        mask.to(device=matrix.device, dtype=torch.bool), matrix.shape[:-1]
    )


def _solve_positive_definite(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Use Cholesky solves for transform-consistent higher derivatives."""
    matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    factor = torch.linalg.cholesky(matrix)
    return torch.cholesky_solve(rhs, factor)


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
    low_temperature: Tensor,
) -> Tensor:
    """Evaluate the symmetric matrix Fermi function with fixed scaling."""
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    eye = eye.expand(matrix.shape)
    exponent = (matrix - shift[..., None, None] * eye) / kt
    # For relative spectral distances up to 200 Eh, the ordinary-temperature
    # branch (kt >= 5e-4 Eh) keeps the initial matrix_exp argument below 391.
    # At lower smearing, use scale 2**22 and 21 doublings; with the supported
    # kt floor of 3e-7 Eh this also keeps the argument below 318. Both branches
    # evaluate the same matrix function, selected by a tensor mask and fixed
    # trip arithmetic without host control.
    scale = torch.where(low_temperature, 4_194_304.0, 2048.0)
    scaled = exponent / scale[..., None, None]
    exp_scaled = torch.matrix_exp(2.0 * scaled)
    tangent = _solve_positive_definite(exp_scaled + eye, exp_scaled - eye)
    for index in range(21):
        doubled = (
            2.0
            * tangent
            @ _solve_positive_definite(eye + tangent @ tangent, eye)
        )
        update = low_temperature | (index < 10)
        tangent = torch.where(update[..., None, None], doubled, tangent)
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
    # The Fermi core's mask indexes sorted orbital energies. ``mask`` indexes
    # the AO basis and may be noncontiguous in a canonical padded layout.
    # Matrix operations must use the latter; the former remains in ``state``
    # for filling and status calculations.
    structural_mask = mask
    if structural_mask is None:
        structural_mask = state.orbital_mask
        if structural_mask is not None and structural_mask.ndim == h.ndim:
            structural_mask = structural_mask.select(-2, 0)
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
    low_temperature = kt < 5.0e-4
    diff_mask = active_channel & ~gap_mask & ~empty & ~full & ~too_many
    shift = torch.zeros_like(target)
    for _ in range(diff_steps):
        density = _matrix_fermi(
            relative,
            shift,
            safe_kt,
            active,
            low_temperature,
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
        low_temperature,
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


def _binary_entropy_matrix(
    x: Tensor, occupation: Tensor, active: Tensor
) -> Tensor:
    """Evaluate binary entropy as a fixed quadrature of matrix tanh."""
    identity = torch.diag_embed(active.to(x.dtype))
    tanh_endpoint = identity - 2.0 * occupation
    nodes = torch.as_tensor(_GL_NODES_NP, dtype=x.dtype, device=x.device)
    weights = torch.as_tensor(_GL_WEIGHTS_NP, dtype=x.dtype, device=x.device)
    scaled = nodes.reshape(-1, 1, 1, 1) * x.unsqueeze(0)
    active_q = active.unsqueeze(0).expand_as(scaled[..., 0, :])
    kt_one = x.new_ones(())
    low = torch.ones((), dtype=torch.bool, device=x.device)
    thermal = _matrix_fermi(
        scaled, torch.zeros_like(nodes).reshape(-1, 1), kt_one, active_q, low
    )
    tanh_nodes = identity.unsqueeze(0) - 2.0 * thermal
    integral = (weights.reshape(-1, 1, 1, 1) * tanh_nodes).sum(0)
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    entropy = 0.5 * (x @ (tanh_endpoint - integral)) - np.log(2.0) * identity
    entropy = 0.5 * (entropy + entropy.transpose(-2, -1))
    return torch.where(pair, entropy, torch.zeros_like(entropy))


def fermi_free_energy_matrix(
    hamiltonian: Tensor,
    nel: Tensor,
    kt: Tensor,
    state: _FermiCoreState,
    *,
    l_inv: Tensor | None = None,
    mask: Tensor | None = None,
    diff_steps: int = 2,
) -> Tensor:
    """Build the invariant finite-temperature free-energy spectral matrix."""
    work_dtype = torch.float64
    h = hamiltonian.to(dtype=work_dtype)
    li = None if l_inv is None else l_inv.to(dtype=work_dtype)
    # Keep the Fermi-energy mask (state.orbital_mask) separate from the AO
    # basis mask supplied by the SCF input. Canonical AO validity can be
    # noncontiguous even though padded eigenvalues sort to a valid prefix.
    structural_mask = mask
    if structural_mask is None:
        structural_mask = state.orbital_mask
        if structural_mask is not None and structural_mask.ndim == h.ndim:
            structural_mask = structural_mask.select(-2, 0)
    h, spectral, active = _prepare_matrix(h, li, structural_mask)

    h = h.unsqueeze(-3).expand(*nel.shape, h.shape[-2], h.shape[-1])
    spectral = spectral.unsqueeze(-3).expand_as(h)
    active = active.unsqueeze(-2).expand(*nel.shape, active.shape[-1])
    if li is not None:
        li = li.unsqueeze(-3).expand_as(h)
        pair = active.unsqueeze(-1) & active.unsqueeze(-2)
        li = torch.where(pair, li, torch.zeros_like(li))

    kt = kt.to(device=h.device, dtype=work_dtype)
    kt_positive = state.kt_positive.to(device=h.device)
    safe_kt = torch.where(kt_positive, kt, torch.ones_like(kt))
    ref = state.chemical_potential_ref.to(device=h.device, dtype=work_dtype)
    offset = state.chemical_potential_offset.to(
        device=h.device, dtype=work_dtype
    )
    ref, offset = ref.squeeze(-1), offset.squeeze(-1)
    channel_shape = (*nel.shape, 1, 1)
    eye = torch.eye(h.shape[-1], dtype=work_dtype, device=h.device)
    relative = spectral - ref.reshape(channel_shape) * eye
    relative = relative - offset.reshape(channel_shape) * eye
    target = nel.to(device=h.device, dtype=work_dtype)
    empty = state.empty.to(device=h.device)
    full = state.full.to(device=h.device)
    too_many = state.too_many.to(device=h.device)
    active_channel = state.active.to(device=h.device)
    gap_mask = state.gap.to(device=h.device, dtype=torch.bool)
    below = state.below.to(device=h.device, dtype=torch.bool)
    low_temperature = kt < 5.0e-4

    diff_mask = active_channel & ~gap_mask & ~empty & ~full & ~too_many
    shift = torch.zeros_like(target)
    for _ in range(diff_steps):
        density = _matrix_fermi(
            relative, shift, safe_kt, active, low_temperature
        )
        count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
        slope = (density - density @ density).diagonal(dim1=-2, dim2=-1).sum(
            -1
        ) / safe_kt
        safe_slope = torch.where(
            slope.abs() > torch.finfo(work_dtype).tiny, slope, 1.0
        )
        correction = torch.where(diff_mask, (target - count) / safe_slope, 0.0)
        shift = shift + correction

    x = (relative - shift[..., None, None] * eye) / safe_kt
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    x = torch.where(pair, x, torch.zeros_like(x))
    occupation = _matrix_fermi(
        relative, shift, safe_kt, active, low_temperature
    )
    entropy = _binary_entropy_matrix(x, occupation, active)
    q = safe_kt * entropy
    identity = torch.diag_embed(active.to(work_dtype))
    spectral_bound = x.abs().sum(-1).masked_fill(~active, 0.0).amax(-1)
    resolved_gap_tail = gap_mask & (spectral_bound > 1.0e4)
    q = torch.where(resolved_gap_tail[..., None, None], 0.0, q)
    # The Fermi occupation helper retains a linear electron-count response in
    # wide gaps where the thermal tails are below the derivative floor. Apply
    # the corresponding (mu - epsilon) weighted response to this matrix too.
    gap_response = _gap_electron_response(
        h, spectral, active, below, gap_mask, safe_kt
    )
    mu_gap = ref + offset + shift
    relative_mu = (mu_gap[..., None, None] * identity - spectral) / safe_kt
    count_delta = target - target.detach()
    q = q + safe_kt * count_delta[..., None, None] * (
        relative_mu @ gap_response
    )
    status = state.result.status
    valid_channel = (
        status.converged.to(device=h.device)
        & status.electron_count_ok.to(device=h.device)
        & ~status.invalid_temperature.to(device=h.device)
        & ~status.too_many_electrons.to(device=h.device)
    )
    entropy_floor = torch.finfo(hamiltonian.dtype).eps
    floor_weight = (
        safe_kt
        * entropy_floor
        * torch.log(
            torch.tensor(entropy_floor, dtype=work_dtype, device=h.device)
        )
    )
    q = torch.where((~valid_channel | too_many)[..., None, None], 0.0, q)
    saturated = valid_channel & (empty | full)
    q = torch.where(saturated[..., None, None], floor_weight * identity, q)
    gap_value = q - q.detach() + floor_weight * identity
    valid_gap = gap_mask & valid_channel & ~empty & ~full & ~too_many
    q = torch.where(valid_gap[..., None, None], gap_value, q)
    if li is not None:
        q = li.transpose(-2, -1) @ q @ li
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    q = torch.where(pair, q, torch.zeros_like(q))
    return q.sum(-3).to(dtype=hamiltonian.dtype)


def aufbau_free_energy_matrix(
    hamiltonian: Tensor,
    nel: Tensor,
    kt: Tensor,
    *,
    l_inv: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """Build the low-temperature clamped Aufbau free-energy matrix."""
    work_dtype = torch.float64
    h = hamiltonian.to(dtype=work_dtype)
    li = None if l_inv is None else l_inv.to(dtype=work_dtype)
    h, spectral, active = _prepare_matrix(h, li, mask)
    n = h.shape[-1]
    h = h.unsqueeze(-3).expand(*nel.shape, n, n)
    spectral = spectral.unsqueeze(-3).expand_as(h)
    active = active.unsqueeze(-2).expand(*nel.shape, n)
    if li is not None:
        li = li.unsqueeze(-3).expand_as(h)
    eye = torch.eye(n, dtype=work_dtype, device=h.device)
    active_projector = torch.diag_embed(active.to(work_dtype))
    count = active.sum(-1).to(work_dtype)
    effective = torch.minimum(nel.to(work_dtype).clamp_min(0.0), count)
    rank = effective.detach().floor().to(torch.long)
    fraction = effective - rank

    lower = _projector(h, spectral, active, rank)
    upper_rank = torch.where(
        fraction > 0.0,
        torch.minimum(rank + 1, count.to(torch.long)),
        rank,
    )
    upper = _projector(h, spectral, active, upper_rank)
    frontier = torch.where(
        (rank < count)[..., None, None], upper - lower, torch.zeros_like(h)
    )
    eps = torch.finfo(hamiltonian.dtype).eps
    base_weight = (
        kt.to(device=h.device, dtype=work_dtype)
        * eps
        * torch.log(torch.tensor(eps, dtype=work_dtype, device=h.device))
    )
    occ = fraction.clamp(min=eps)
    hole = (1.0 - fraction).clamp(min=eps)
    frontier_weight = kt.to(device=h.device, dtype=work_dtype) * (
        occ * torch.log(occ) + hole * torch.log(hole)
    )
    q = (
        base_weight * active_projector
        + (frontier_weight - base_weight)[..., None, None] * frontier
    )
    if li is not None:
        q = li.transpose(-2, -1) @ q @ li
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    q = torch.where(pair, q, torch.zeros_like(q))
    return q.sum(-3).to(dtype=hamiltonian.dtype)


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

    # ``active`` indexes basis functions, while ``evals`` is sorted by energy.
    # A canonical padded layout can place physical orbitals at noncontiguous
    # AO indices, so the basis mask cannot be applied directly to eigenvalues.
    eigenvalue_active = torch.arange(
        n, device=matrix.device
    ) < active_count.unsqueeze(-1)
    scale = torch.maximum(
        (evals - mu.unsqueeze(-1))
        .abs()
        .masked_fill(~eigenvalue_active, 0.0)
        .amax(-1),
        span * 0.5,
    ).clamp_min(torch.finfo(matrix.dtype).tiny)
    pad = torch.where(active, 0.0, (mu + 2.0 * scale).unsqueeze(-1))
    h = matrix + torch.diag_embed(pad)
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device).expand(h.shape)
    x = (h - mu[..., None, None] * eye) / scale[..., None, None]
    zero_rank = rank == 0
    full_rank = rank == active_count
    boundary_rank = zero_rank | full_rank
    gap = (e_hi - e_lo).abs()
    safe_gap = torch.where(boundary_rank, torch.ones_like(gap), gap)
    preconditioner = torch.sqrt(2.0 * scale / safe_gap)
    x = x * preconditioner[..., None, None]
    crossing = (~boundary_rank) & (gap == 0.0)
    x = torch.where(crossing[..., None, None], torch.zeros_like(x), x)
    x = torch.where(zero_rank[..., None, None], eye, x)
    x = torch.where(full_rank[..., None, None], -eye, x)
    # Newton-Schulz sign iteration. The fixed iteration count avoids
    # data-dependent control flow. The detached spectral-gap preconditioner
    # balances the smallest and largest magnitudes before the iteration,
    # keeping the inverse systems nonsingular. Since x is symmetric and
    # invertible, solve (x @ x) y = x for y = x**-1; the coefficient matrix is
    # positive definite and can use the transform-safe Cholesky solve above.
    for _ in range(14):
        inverse = _solve_positive_definite(x @ x, x)
        x = 0.5 * (x + inverse)
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
    its positive-definite factorization raises rather than manufacturing a
    response.
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
