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

from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import torch
from tad_mctc.autograd import attach_gradient
from tad_mctc.storch.linsolve import dense_solve
from tad_mctc.tools.compile import is_compiling

from dxtb._src.typing import Tensor

if TYPE_CHECKING:
    from dxtb._src.wavefunction.filling import _FermiCoreState

__all__ = [
    "aufbau_density_matrix",
    "aufbau_free_energy_matrix",
    "fermi_density_matrix",
    "fermi_free_energy_matrix",
    "fermi_free_energy_trace",
]

_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(512)
_GL_NODES_NP = 0.5 * (_GL_NODES_NP + 1.0)
_GL_WEIGHTS_NP = 0.5 * _GL_WEIGHTS_NP

_FERMI_MAX_SCALE_POWER = 30
_FERMI_MAX_SCALED_NORM = 4.0


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
) -> Tensor:
    """Evaluate the symmetric matrix Fermi function with bounded scaling.

    Active spectra are supported when their induced infinity norm after
    scaling is at most four and the required outer scale power is at most 30.
    Inputs outside this range produce NaNs lane-locally. The detached scale
    selects an evaluation schedule for the same matrix function.
    """
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    eye = eye.expand(matrix.shape)
    active_pair = active.unsqueeze(-1) & active.unsqueeze(-2)

    # Ignore invalid structural entries before checking finiteness. Replace
    # invalid lanes before subtraction, division, logarithms, and matrix
    # solves so they cannot create invalid intermediates.
    active_matrix = torch.where(active_pair, matrix, torch.zeros_like(matrix))
    active_any = active.any(-1)
    finite_matrix = torch.where(
        active_pair, torch.isfinite(matrix), torch.ones_like(active_pair)
    ).all(dim=(-1, -2))
    finite_shift_kt = torch.isfinite(shift) & torch.isfinite(kt) & (kt > 0.0)
    finite_inputs = finite_matrix & torch.where(
        active_any, finite_shift_kt, torch.ones_like(active_any)
    )
    safe_matrix = torch.where(
        finite_inputs[..., None, None], active_matrix, torch.zeros_like(matrix)
    )
    safe_shift = torch.where(
        finite_inputs & active_any, shift, torch.zeros_like(shift)
    )
    safe_kt = torch.where(finite_inputs & active_any, kt, torch.ones_like(kt))

    # Detect only a genuinely overflowing diagonal subtraction. Large
    # absolute energies that cancel against the chemical-potential shift can
    # still have a small, supported exponent and must remain usable.
    diagonal = safe_matrix.diagonal(dim1=-2, dim2=-1)
    shift_diagonal = safe_shift.unsqueeze(-1)
    opposite_signs = (diagonal < 0.0) & (shift_diagonal > 0.0)
    opposite_signs = opposite_signs | (
        (diagonal > 0.0) & (shift_diagonal < 0.0)
    )
    maximum_value = torch.finfo(matrix.dtype).max
    subtraction_overflow = (
        active
        & opposite_signs
        & (diagonal.abs() > maximum_value - safe_shift.abs().unsqueeze(-1))
    ).any(-1)
    safe_inputs = finite_inputs & ~subtraction_overflow
    safe_matrix = torch.where(
        safe_inputs[..., None, None], safe_matrix, torch.zeros_like(matrix)
    )
    safe_shift = torch.where(
        safe_inputs, safe_shift, torch.zeros_like(safe_shift)
    )
    safe_kt = torch.where(safe_inputs, safe_kt, torch.ones_like(safe_kt))

    difference = safe_matrix - safe_shift[..., None, None] * eye
    maximum_exponent = _FERMI_MAX_SCALED_NORM * float(2**_FERMI_MAX_SCALE_POWER)
    entry_in_range = (
        difference.abs().amax(dim=(-1, -2)) / maximum_exponent <= safe_kt
    )
    safe_inputs = safe_inputs & (entry_in_range | ~active_any)
    difference = torch.where(
        safe_inputs[..., None, None], difference, torch.zeros_like(difference)
    )
    safe_kt = torch.where(safe_inputs, safe_kt, torch.ones_like(safe_kt))
    exponent = difference / safe_kt[..., None, None]
    low_temperature = safe_kt < 5.0e-4

    # Select a scaling schedule from a detached bound on the active spectrum.
    # This prevents wide finite-temperature spectra from overflowing during
    # the matrix exponential while leaving the physical derivative unchanged.
    exponent = torch.where(active_pair, exponent, torch.zeros_like(exponent))
    exponent_bound = exponent.detach().abs().sum(-1).amax(-1)
    required_power = torch.ceil(
        torch.log2(exponent_bound.clamp_min(1.0) / _FERMI_MAX_SCALED_NORM)
    ).to(torch.long)

    base_power = torch.where(low_temperature, 22, 11)
    in_range = safe_inputs & (required_power <= _FERMI_MAX_SCALE_POWER)
    exponent = torch.where(
        in_range[..., None, None], exponent, torch.zeros_like(exponent)
    )
    scale_power = torch.maximum(base_power, required_power).clamp(
        min=0, max=_FERMI_MAX_SCALE_POWER
    )
    scale = torch.pow(exponent.new_tensor(2.0), scale_power)
    exp_argument = 2.0 * exponent / scale[..., None, None]
    expm1_scaled = _matrix_expm1(exp_argument)
    tangent = dense_solve(expm1_scaled + 2.0 * eye, expm1_scaled)
    for index in range(_FERMI_MAX_SCALE_POWER - 1):
        denominator = eye + tangent @ tangent
        doubled = 2.0 * dense_solve(denominator, tangent)
        update = index < (scale_power - 1)
        tangent = torch.where(update[..., None, None], doubled, tangent)
    density = 0.5 * (eye - tangent)
    density = 0.5 * (density + density.transpose(-2, -1))
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    density = torch.where(pair, density, torch.zeros_like(density))
    nan = torch.full_like(density, float("nan"))
    return torch.where(in_range[..., None, None], density, nan)


def _matrix_expm1(matrix: Tensor) -> Tensor:
    """Evaluate a symmetric matrix ``expm1`` without subtractive cancellation.

    Supported Fermi spectra are scaled lane-wise to have an induced infinity
    norm no greater than 1/4. A 12-term Taylor polynomial is then accurate
    beyond float64 precision on that interval. Repeatedly applying
    ``expm1(2 X) = expm1(X) @ (expm1(X) + 2 I)`` restores the original scale
    without subtracting nearly equal matrices.
    """
    identity = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    )
    identity = identity.expand(matrix.shape)
    finite = torch.isfinite(matrix).all(dim=(-1, -2))
    maximum = matrix.detach().abs().amax(dim=(-1, -2))
    prevalid = finite & (maximum <= 64.0)
    safe_matrix = torch.where(
        prevalid[..., None, None], matrix, torch.zeros_like(matrix)
    )
    norm_bound = safe_matrix.detach().abs().sum(-1).amax(-1)
    requested_scale_power = torch.ceil(
        torch.log2(norm_bound.clamp_min(0.25) / 0.25)
    ).to(torch.long)
    # The upstream Fermi scaling bounds the exponential argument by 8 over
    # the supported spectral range, so at most five halvings are normally
    # needed. The larger fixed loop also covers broader spectra without
    # data-dependent Python control flow under vmap.
    in_supported_range = prevalid & (requested_scale_power <= 8)
    safe_matrix = torch.where(
        in_supported_range[..., None, None],
        safe_matrix,
        torch.zeros_like(matrix),
    )
    scale_power = requested_scale_power.clamp(0, 8)
    scale = torch.pow(matrix.new_tensor(2.0), scale_power)
    scaled = safe_matrix / scale[..., None, None]

    square = scaled @ scaled
    cube = square @ scaled
    fourth = square @ square

    block_zero = (
        identity + 0.5 * scaled + (1.0 / 6.0) * square + (1.0 / 24.0) * cube
    )
    block_one = (
        (1.0 / 120.0) * identity
        + (1.0 / 720.0) * scaled
        + (1.0 / 5040.0) * square
        + (1.0 / 40320.0) * cube
    )
    block_two = (
        (1.0 / 362880.0) * identity
        + (1.0 / 3628800.0) * scaled
        + (1.0 / 39916800.0) * square
        + (1.0 / 479001600.0) * cube
    )
    result = block_two @ fourth + block_one
    result = result @ fourth + block_zero
    result = scaled @ result
    result = 0.5 * (result + result.transpose(-2, -1))

    for index in range(8):
        doubled = result @ (result + 2.0 * identity)
        doubled = 0.5 * (doubled + doubled.transpose(-2, -1))
        update = index < scale_power
        result = torch.where(update[..., None, None], doubled, result)

    result = torch.where(
        in_supported_range[..., None, None],
        result,
        torch.full_like(result, float("nan")),
    )
    return result


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


def _tracks_derivatives(x: Tensor) -> bool:
    """Check if a tensor tracks reverse- or forward-mode derivatives."""
    if is_compiling():
        return True
    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    raw_stack = ft.get_interpreter_stack()
    stack = (
        {}
        if raw_stack is None
        else {interp.level(): interp.key() for interp in raw_stack}
    )
    jvp_levels = [
        lvl for lvl, key in stack.items() if key == ft.TransformType.Jvp
    ]
    top_jvp_level = max(jvp_levels) if jvp_levels else None
    current = x
    while True:
        if ft.is_batchedtensor(current):
            current = ft.get_unwrapped(current)
            continue
        if torch.is_grad_enabled() and current.requires_grad:
            return True
        level = ft.maybe_get_level(current)
        if level == -1:
            if (
                torch.autograd.forward_ad.unpack_dual(current).tangent
                is not None
            ):
                return True
        elif stack.get(level) == ft.TransformType.Jvp:
            if level != top_jvp_level:
                return True
            if (
                torch.autograd.forward_ad.unpack_dual(current).tangent
                is not None
            ):
                return True
        if not ft.is_functorch_wrapped_tensor(current):
            break
        current = ft.get_unwrapped(current)
    return False


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
    setup = _fermi_free_energy_setup(
        hamiltonian, nel, kt, state, l_inv, mask, diff_steps
    )
    h = setup.h
    active = setup.active
    eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
    eye_channel = eye * active.unsqueeze(-1).to(h.dtype)
    density = torch.where(setup.empty[..., None, None], 0.0, setup.occupation)
    density = torch.where(setup.too_many[..., None, None], 0.0, density)
    density = torch.where(setup.full[..., None, None], eye_channel, density)
    count_delta = setup.target - setup.target.detach()
    if _tracks_derivatives(nel) or _tracks_derivatives(count_delta):
        response = _gap_electron_response(
            h,
            setup.spectral,
            active,
            setup.below,
            setup.gap_mask,
            setup.safe_kt,
        )
        density = density + count_delta[..., None, None] * response
    if setup.li is not None:
        density = setup.li.transpose(-2, -1) @ density @ setup.li
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
    thermal = _matrix_fermi(
        scaled, torch.zeros_like(nodes).reshape(-1, 1), kt_one, active_q
    )
    tanh_nodes = identity.unsqueeze(0) - 2.0 * thermal
    integral = (weights.reshape(-1, 1, 1, 1) * tanh_nodes).sum(0)
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    entropy = 0.5 * (x @ (tanh_endpoint - integral)) - np.log(2.0) * identity
    entropy = 0.5 * (entropy + entropy.transpose(-2, -1))
    return torch.where(pair, entropy, torch.zeros_like(entropy))


def _trace_logcosh(matrix: Tensor) -> Tensor:
    """``Tr ln cosh(X/2)`` from the rotation-invariant eigenvalues of ``X``."""
    magnitude = torch.linalg.eigvalsh(matrix).abs()
    return (
        0.5 * magnitude
        + torch.log1p(torch.exp(-magnitude))
        - float(np.log(2.0))
    ).sum(-1)


class _FermiFreeEnergySetup(NamedTuple):
    """Per-channel quantities shared by the free-energy matrix and trace."""

    h: Tensor
    spectral: Tensor
    active: Tensor
    li: Tensor | None
    safe_kt: Tensor
    ref: Tensor
    offset: Tensor
    shift: Tensor
    target: Tensor
    empty: Tensor
    full: Tensor
    too_many: Tensor
    gap_mask: Tensor
    below: Tensor
    x: Tensor
    occupation: Tensor


def _fermi_free_energy_setup(
    hamiltonian: Tensor,
    nel: Tensor,
    kt: Tensor,
    state: _FermiCoreState,
    l_inv: Tensor | None,
    mask: Tensor | None,
    diff_steps: int,
) -> _FermiFreeEnergySetup:
    work_dtype = torch.float64
    h = hamiltonian.to(dtype=work_dtype)
    li = None if l_inv is None else l_inv.to(dtype=work_dtype)
    # The explicit mask indexes the AO basis; state.orbital_mask indexes the
    # energy-sorted spectrum. Canonical AO validity may be noncontiguous.
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

    diff_mask = active_channel & ~gap_mask & ~empty & ~full & ~too_many
    shift = torch.zeros_like(target)
    for _ in range(diff_steps):
        density = _matrix_fermi(relative, shift, safe_kt, active)
        count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
        slope = (count - (density * density).sum(dim=(-2, -1))) / safe_kt
        safe_slope = torch.where(
            slope.abs() > torch.finfo(work_dtype).tiny, slope, 1.0
        )
        correction = torch.where(diff_mask, (target - count) / safe_slope, 0.0)
        shift = shift + correction

    x = (relative - shift[..., None, None] * eye) / safe_kt
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    x = torch.where(pair, x, torch.zeros_like(x))
    occupation = _matrix_fermi(relative, shift, safe_kt, active)
    return _FermiFreeEnergySetup(
        h=h,
        spectral=spectral,
        active=active,
        li=li,
        safe_kt=safe_kt,
        ref=ref,
        offset=offset,
        shift=shift,
        target=target,
        empty=empty,
        full=full,
        too_many=too_many,
        gap_mask=gap_mask,
        below=below,
        x=x,
        occupation=occupation,
    )


def _fermi_free_energy_closure(
    q: Tensor,
    gap_term: Tensor,
    identity: Tensor,
    setup: _FermiFreeEnergySetup,
    state: _FermiCoreState,
    dtype: torch.dtype,
    channel: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    Apply the gap, saturation and validity rules to an entropy term ``q``.

    ``q``, ``gap_term`` (the ``(mu - epsilon)``-weighted gap response) and
    ``identity`` are either channel matrices or their traces; ``channel``
    broadcasts a per-channel mask or scalar onto them.
    """
    work_dtype = torch.float64
    device = setup.x.device
    spectral_bound = (
        setup.x.abs().sum(-1).masked_fill(~setup.active, 0.0).amax(-1)
    )
    resolved_gap_tail = setup.gap_mask & (spectral_bound > 1.0e4)
    q = torch.where(channel(resolved_gap_tail), 0.0, q)
    # The Fermi occupation helper retains a linear electron-count response in
    # wide gaps where the thermal tails are below the derivative floor. Apply
    # the corresponding (mu - epsilon) weighted response to this matrix too.
    count_delta = setup.target - setup.target.detach()
    q = q + channel(setup.safe_kt * count_delta) * gap_term
    status = state.result.status
    valid_channel = (
        status.converged.to(device=device)
        & status.electron_count_ok.to(device=device)
        & ~status.invalid_temperature.to(device=device)
        & ~status.too_many_electrons.to(device=device)
    )
    entropy_floor = torch.finfo(dtype).eps
    floor_weight = (
        setup.safe_kt
        * entropy_floor
        * torch.log(
            torch.tensor(entropy_floor, dtype=work_dtype, device=device)
        )
    )
    q = torch.where(channel(~valid_channel | setup.too_many), 0.0, q)
    saturated = valid_channel & (setup.empty | setup.full)
    q = torch.where(channel(saturated), channel(floor_weight) * identity, q)
    gap_value = q - q.detach() + channel(floor_weight) * identity
    valid_gap = (
        setup.gap_mask
        & valid_channel
        & ~setup.empty
        & ~setup.full
        & ~setup.too_many
    )
    q = torch.where(channel(valid_gap), gap_value, q)
    finite_occupation = torch.isfinite(setup.occupation).all(dim=(-1, -2))
    return torch.where(
        channel(~finite_occupation), torch.full_like(q, float("nan")), q
    )


def _gap_relative_mu(setup: _FermiFreeEnergySetup, identity: Tensor) -> Tensor:
    mu_gap = setup.ref + setup.offset + setup.shift
    return (mu_gap[..., None, None] * identity - setup.spectral) / setup.safe_kt


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
    setup = _fermi_free_energy_setup(
        hamiltonian, nel, kt, state, l_inv, mask, diff_steps
    )
    active = setup.active
    entropy = _binary_entropy_matrix(setup.x, setup.occupation, active)
    identity = torch.diag_embed(active.to(torch.float64))
    count_delta = setup.target - setup.target.detach()
    if _tracks_derivatives(nel) or _tracks_derivatives(count_delta):
        gap_response = _gap_electron_response(
            setup.h,
            setup.spectral,
            active,
            setup.below,
            setup.gap_mask,
            setup.safe_kt,
        )
        gap_term = _gap_relative_mu(setup, identity) @ gap_response
    else:
        gap_term = torch.zeros_like(setup.x)
    q = _fermi_free_energy_closure(
        setup.safe_kt * entropy,
        gap_term,
        identity,
        setup,
        state,
        hamiltonian.dtype,
        lambda value: value[..., None, None],
    )
    if setup.li is not None:
        q = setup.li.transpose(-2, -1) @ q @ setup.li
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    q = torch.where(pair, q, torch.zeros_like(q))
    return q.sum(-3).to(dtype=hamiltonian.dtype)


def fermi_free_energy_trace(
    hamiltonian: Tensor,
    nel: Tensor,
    kt: Tensor,
    state: _FermiCoreState,
    *,
    l_inv: Tensor,
    mask: Tensor | None = None,
    diff_steps: int = 2,
) -> Tensor:
    """
    Overlap-weighted trace of :func:`fermi_free_energy_matrix`.

    With ``l_inv`` the inverse Cholesky factor of the overlap ``S``, the
    trace ``Tr(S M)`` of the AO matrix equals the sum of the orthogonal-basis
    channel traces. This evaluates those traces without the matrix
    quadrature: ``Tr ln cosh(X/2)`` comes from the eigenvalues, and its
    derivatives from the differentiable Fermi occupation
    (``tad_mctc.autograd.attach_gradient``). Neither eigenvectors nor the
    quadrature are differentiated. Values agree with the quadrature matrix
    to its quadrature error (about 1e-10 relative).
    """
    setup = _fermi_free_energy_setup(
        hamiltonian, nel, kt, state, l_inv, mask, diff_steps
    )
    active = setup.active
    identity = torch.diag_embed(active.to(torch.float64))
    x, occupation = setup.x, setup.occupation

    # Binary entropy: Tr[X (1 - 2f)] / 2 - Tr ln cosh(X/2) - ln(2) n. The
    # gradient of Tr ln cosh(X/2) is tanh(X/2)/2 = 1/2 - f(X), so all its
    # derivatives follow from those of the differentiable occupation.
    half_tanh = 0.5 * identity - occupation
    half_tanh = 0.5 * (half_tanh + half_tanh.transpose(-2, -1))
    entropy = (
        (x * half_tanh.transpose(-2, -1)).sum((-2, -1))
        - attach_gradient(_trace_logcosh, x, half_tanh)
        - float(np.log(2.0)) * active.sum(-1).to(torch.float64)
    )
    count_delta = setup.target - setup.target.detach()
    if _tracks_derivatives(nel) or _tracks_derivatives(count_delta):
        gap_response = _gap_electron_response(
            setup.h,
            setup.spectral,
            active,
            setup.below,
            setup.gap_mask,
            setup.safe_kt,
        )
        gap_term = (
            _gap_relative_mu(setup, identity) * gap_response.transpose(-2, -1)
        ).sum((-2, -1))
    else:
        gap_term = torch.zeros_like(setup.shift)
    q = _fermi_free_energy_closure(
        setup.safe_kt * entropy,
        gap_term,
        active.sum(-1).to(torch.float64),
        setup,
        state,
        hamiltonian.dtype,
        lambda value: value,
    )
    return q.sum(-1).to(dtype=hamiltonian.dtype)


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
    # invertible, solve (x @ x) y = x for y = x**-1 with Cholesky.
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
    its positive-definite Cholesky factorization raises rather than
    manufacturing a response.
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
