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
"""Gauge-invariant spectral matrices for the SCF core."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, NamedTuple

import numpy as np
import torch
from tad_mctc.autograd import attach_gradient
from tad_mctc.storch.linsolve import dense_solve
from tad_mctc.tools.compile import is_compiling
from torch.overrides import has_torch_function_unary
from torch.utils._python_dispatch import _get_current_dispatch_mode

from dxtb._src.typing import Tensor

if TYPE_CHECKING:
    from dxtb._src.wavefunction.filling import _FermiCoreState

__all__ = [
    "aufbau_density_matrix",
    "aufbau_free_energy_matrix",
    "eager_primal_fermi_scope",
    "fermi_density_matrix",
    "fermi_free_energy_matrix",
    "fermi_free_energy_trace",
]

_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(512)
_GL_NODES_NP = 0.5 * (_GL_NODES_NP + 1.0)
_GL_WEIGHTS_NP = 0.5 * _GL_WEIGHTS_NP

_FERMI_MAX_SCALE_POWER = 30
_FERMI_MAX_SCALED_NORM = 4.0
_MAX_DIFF_STEP_KT = 3.0
_DALECKII_KREIN_TAYLOR_CUTOFF = 1.0e-4

_EAGER_PRIMAL_STATE = threading.local()


def _normalize_primal_fermi_mode(mode: bool | str) -> str:
    """Normalize a primal Fermi execution strategy identifier."""
    if isinstance(mode, bool):
        return "adaptive" if mode else "static"
    if isinstance(mode, str) and mode in ("static", "adaptive", "spectral"):
        return mode
    raise ValueError(
        "eager_primal_fermi must be a bool or one of 'static', 'adaptive', "
        f"or 'spectral'; got {mode!r}."
    )


def _current_primal_fermi_mode(explicit: bool | str = False) -> str:
    """Resolve the active primal Fermi mode from explicit argument or scope."""
    explicit_mode = _normalize_primal_fermi_mode(explicit)
    if explicit_mode != "static":
        return explicit_mode
    scope_mode = getattr(_EAGER_PRIMAL_STATE, "mode", None)
    if scope_mode is not None:
        return _normalize_primal_fermi_mode(scope_mode)
    if bool(getattr(_EAGER_PRIMAL_STATE, "enabled", False)):
        return "adaptive"
    return "static"


@contextmanager
def eager_primal_fermi_scope(enabled: bool | str = True) -> Iterator[None]:
    """Configure the thread-local primal Fermi evaluation strategy."""
    mode = _normalize_primal_fermi_mode(enabled)
    prev_enabled = getattr(_EAGER_PRIMAL_STATE, "enabled", False)
    prev_mode = getattr(
        _EAGER_PRIMAL_STATE,
        "mode",
        "adaptive" if prev_enabled else "static",
    )
    _EAGER_PRIMAL_STATE.mode = mode
    _EAGER_PRIMAL_STATE.enabled = mode != "static"
    try:
        yield
    finally:
        _EAGER_PRIMAL_STATE.mode = prev_mode
        _EAGER_PRIMAL_STATE.enabled = prev_enabled


def _require_detached_eager_primal(name: str, *tensors: Tensor) -> None:
    """Ensure a detached-primal helper never runs with gradients or transforms."""
    if torch.is_grad_enabled() or is_compiling():
        raise RuntimeError(
            f"{name} is restricted to detached eager primal evaluation."
        )
    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    if ft.get_interpreter_stack():
        raise RuntimeError(f"{name} cannot run under torch.func.")
    for tensor in tensors:
        if ft.is_functorch_wrapped_tensor(tensor):
            raise RuntimeError(
                f"{name} cannot run on functorch-wrapped tensors."
            )
        if (
            tensor.is_floating_point() or tensor.is_complex()
        ) and torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None:
            raise RuntimeError(f"{name} cannot run on forward-AD dual tensors.")


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
    if is_compiling():
        factor = torch.linalg.cholesky(matrix)
        return torch.cholesky_solve(rhs, factor)
    return dense_solve(matrix, rhs, method="cholesky")


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
    return _matrix_fermi_impl(matrix, shift, kt, active, eager=False)


def _matrix_fermi_eager(
    matrix: Tensor,
    shift: Tensor,
    kt: Tensor,
    active: Tensor,
) -> Tensor:
    """Eager-only equivalent that stops after the largest required doubling."""
    _require_detached_eager_primal(
        "Adaptive matrix-Fermi doubling", matrix, shift, kt
    )
    return _matrix_fermi_impl(matrix, shift, kt, active, eager=True)


def _matrix_fermi_spectral(
    matrix: Tensor,
    shift: Tensor,
    kt: Tensor,
    active: Tensor,
    *,
    precomputed_eigh: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Detached-primal spectral Fermi evaluation ``U diag(f(e_i)) U^T``.

    Strictly restricted to detached eager primal execution. Preserves the
    exact S0.1b supported-range and nonfinite-input semantics of
    :func:`_matrix_fermi` and projects onto the active AO block so that
    degenerate subspaces and noncontiguous padding are rotation- and
    ordering-invariant.
    """
    _require_detached_eager_primal(
        "Detached-primal spectral matrix-Fermi evaluation", matrix, shift, kt
    )
    return _matrix_fermi_impl(
        matrix,
        shift,
        kt,
        active,
        eager=True,
        spectral=True,
        precomputed_eigh=precomputed_eigh,
    )


def _fermi_daleckii_krein_jvp(
    matrix: Tensor,
    tangent_matrix: Tensor,
    shift: Tensor,
    tangent_shift: Tensor,
    kt: Tensor,
    tangent_kt: Tensor,
    active: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate the spectral Fermi matrix and its Daleckii-Krein JVP.

    Uses the removable-singularity divided-difference kernel on the active
    orthogonalized spectrum so exact and near-degenerate subspaces remain
    smooth and rotation-invariant.
    """
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    eye = eye.expand(matrix.shape)
    active_pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    sym_matrix = 0.5 * (matrix + matrix.transpose(-2, -1))
    sym_matrix = torch.where(active_pair, sym_matrix, torch.zeros_like(matrix))
    sym_tangent = 0.5 * (tangent_matrix + tangent_matrix.transpose(-2, -1))
    sym_tangent = torch.where(
        active_pair, sym_tangent, torch.zeros_like(tangent_matrix)
    )

    count = active.sum(-1).clamp_min(1).to(matrix.dtype)
    center = sym_matrix.diagonal(dim1=-2, dim2=-1).sum(-1) / count
    centered = sym_matrix - center[..., None, None] * eye
    bound = torch.where(active, centered.abs().sum(-1), 0.0).amax(-1)
    sentinel = (center + bound + 1.0).unsqueeze(-1)
    eig_input = sym_matrix + torch.diag_embed(
        torch.where(active, torch.zeros_like(sentinel), sentinel)
    )
    evals, evecs = torch.linalg.eigh(eig_input)
    active_evecs = torch.where(
        active.unsqueeze(-1), evecs, torch.zeros_like(evecs)
    )

    x_eig = (evals - shift.unsqueeze(-1)) / kt.unsqueeze(-1)
    occ = 0.5 * (1.0 - torch.tanh(0.5 * x_eig))
    density = (active_evecs * occ.unsqueeze(-2)) @ active_evecs.transpose(
        -2, -1
    )
    density = 0.5 * (density + density.transpose(-2, -1))
    density = torch.where(active_pair, density, torch.zeros_like(density))

    dx = x_eig.unsqueeze(-1) - x_eig.unsqueeze(-2)
    x_mid = 0.5 * (x_eig.unsqueeze(-1) + x_eig.unsqueeze(-2))
    f_mid = 0.5 * (1.0 - torch.tanh(0.5 * x_mid))
    p_mid = f_mid * (1.0 - f_mid)
    d1_mid = -p_mid
    d3_mid = -p_mid * (1.0 - 6.0 * p_mid)
    d5_mid = -p_mid * (1.0 - 30.0 * p_mid + 120.0 * p_mid * p_mid)
    dx2 = dx * dx
    taylor_kernel = d1_mid + (dx2 / 24.0) * d3_mid + (dx2 * dx2 / 1920.0) * d5_mid
    far = dx.abs() > _DALECKII_KREIN_TAYLOR_CUTOFF
    safe_dx = torch.where(far, dx, torch.ones_like(dx))
    diff_kernel = (occ.unsqueeze(-1) - occ.unsqueeze(-2)) / safe_dx
    divided_diff = torch.where(far, diff_kernel, taylor_kernel)

    dh_eigen = active_evecs.transpose(-2, -1) @ sym_tangent @ active_evecs
    diag_idx = torch.arange(matrix.shape[-1], device=matrix.device)
    dh_eigen = dh_eigen.clone()
    dh_eigen[..., diag_idx, diag_idx] = (
        dh_eigen[..., diag_idx, diag_idx]
        - tangent_shift.unsqueeze(-1)
        - x_eig * tangent_kt.unsqueeze(-1)
    )
    dx_eigen = dh_eigen / kt[..., None, None]
    tangent_density = (
        active_evecs @ (divided_diff * dx_eigen) @ active_evecs.transpose(-2, -1)
    )
    tangent_density = 0.5 * (
        tangent_density + tangent_density.transpose(-2, -1)
    )
    tangent_density = torch.where(
        active_pair, tangent_density, torch.zeros_like(tangent_density)
    )
    return density, tangent_density


def _matrix_fermi_impl(
    matrix: Tensor,
    shift: Tensor,
    kt: Tensor,
    active: Tensor,
    *,
    eager: bool,
    spectral: bool = False,
    precomputed_eigh: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Shared matrix-Fermi algebra with a selected static/eager/spectral schedule."""
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
    if spectral:
        if precomputed_eigh is not None:
            evals, active_evecs = precomputed_eigh
        else:
            sym_matrix = 0.5 * (safe_matrix + safe_matrix.transpose(-2, -1))
            count = active.sum(-1).clamp_min(1).to(matrix.dtype)
            center = sym_matrix.diagonal(dim1=-2, dim2=-1).sum(-1) / count
            centered = sym_matrix - center[..., None, None] * eye
            bound = torch.where(active, centered.abs().sum(-1), 0.0).amax(-1)
            sentinel = (center + bound + 1.0).unsqueeze(-1)
            eig_input = sym_matrix + torch.diag_embed(
                torch.where(active, torch.zeros_like(sentinel), sentinel)
            )
            evals, evecs = torch.linalg.eigh(eig_input)
            active_evecs = torch.where(
                active.unsqueeze(-1), evecs, torch.zeros_like(evecs)
            )
        scaled_arg = (evals - safe_shift.unsqueeze(-1)) / (
            2.0 * safe_kt.unsqueeze(-1)
        )
        occ = 0.5 * (1.0 - torch.tanh(scaled_arg))
        density = (active_evecs * occ.unsqueeze(-2)) @ active_evecs.transpose(
            -2, -1
        )
        density = 0.5 * (density + density.transpose(-2, -1))
        density = torch.where(active_pair, density, torch.zeros_like(density))
        nan = torch.full_like(density, float("nan"))
        return torch.where(in_range[..., None, None], density, nan)

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
    if eager:
        doubling_steps = max(0, int(scale_power.max()) - 1)
    else:
        max_power = _max_scale_power_host(scale_power)
        doubling_steps = (
            _FERMI_MAX_SCALE_POWER - 1
            if max_power is None
            else max(0, min(_FERMI_MAX_SCALE_POWER, max_power) - 1)
        )

    for index in range(doubling_steps):
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


def _max_scale_power_host(scale_power: Tensor) -> int | None:
    """Return the maximum integer scaling power across all batch/vmap lanes.

    ``scale_power`` is a non-differentiable ``torch.long`` tensor computed from
    detached spectral bounds. Unwrapping ``torch.func`` wrappers on CPU bounds
    the loop count to the active maximum across all lanes without executing
    no-op iterations where ``index < scale_power`` is identically ``False``.
    Returns ``None`` when compiling, tracing, or on non-CPU devices so those
    paths keep their static loop bounds.
    """
    if (
        scale_power.device.type != "cpu"
        or is_compiling()
        or torch.jit.is_tracing()
        or _get_current_dispatch_mode() is not None
        or has_torch_function_unary(scale_power)
    ):
        return None
    with torch._C._DisableFuncTorch():
        raw = scale_power
        for _ in range(16):
            if torch._C._functorch.is_batchedtensor(raw):
                raw = torch._C._functorch.get_unwrapped(raw)
                continue
            if torch._C._functorch.is_gradtrackingtensor(raw):
                raw = torch._C._functorch.get_unwrapped(raw)
                continue
            if torch._C._functorch.is_functionaltensor(raw):
                raw = torch._C._functorch.get_unwrapped(raw)
                continue
            break
        if type(raw) is not Tensor or raw.numel() == 0:
            return None
        return int(raw.amax().item())


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

    max_power = _max_scale_power_host(scale_power)
    steps = 8 if max_power is None else min(8, max(0, max_power))
    for index in range(steps):
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
    return _gap_electron_response_and_mu(
        matrix, spectral, active, below, gap, kt
    )[0]


def _gap_electron_response_and_mu(
    matrix: Tensor,
    spectral: Tensor,
    active: Tensor,
    below: Tensor,
    gap: Tensor,
    kt: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build the fixed-count electron response and chemical potential in a gap."""
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
    occupied_clamped = occupied_sum.clamp_min(tiny)
    virtual_clamped = virtual_sum.clamp_min(tiny)
    virtual_part = virtual_tail / virtual_clamped[..., None, None]
    occupied_part = occupied_tail / occupied_clamped[..., None, None]
    has_both = (rank > 0) & (rank < active_count)
    response = torch.where(
        has_both[..., None, None],
        0.5 * (virtual_part + occupied_part),
        virtual_part + occupied_part,
    )
    response = torch.where(gap[..., None, None], response, 0.0)
    mu_occ = homo + kt * torch.log(occupied_clamped)
    mu_virt = lumo - kt * torch.log(virtual_clamped)
    mu_gap = torch.where(
        has_both,
        0.5 * (mu_occ + mu_virt),
        torch.where(rank > 0, mu_occ, mu_virt),
    )
    mu_gap = torch.where(gap, mu_gap, 0.0)
    return response, mu_gap


def _tracks_derivatives(x: Tensor) -> bool:
    """Check if a tensor tracks reverse- or forward-mode derivatives."""
    if is_compiling():
        return True
    try:
        return _tracks_derivatives_functorch(x)
    except (AttributeError, RuntimeError):
        # PyTorch has no public API for asking whether an arbitrary tensor is
        # active in a torch.func transform. If its private transform inspector
        # changes, conservatively run the response instead of risking a wrong
        # derivative.
        return True


def _has_jvp_tangent_at_level(x: Tensor, target_level: int) -> bool:
    """Check whether a functorch Jvp wrapper at target_level carries a non-None tangent."""
    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    popped = []
    try:
        while True:
            raw_stack = ft.get_interpreter_stack()
            if not raw_stack:
                break
            top_level = raw_stack[-1].level()
            if top_level <= target_level:
                break
            popped.append(ft.pop_dynamic_layer_stack())
        return torch.autograd.forward_ad.unpack_dual(x).tangent is not None
    finally:
        while popped:
            ft.push_dynamic_layer_stack(popped.pop())


def _tracks_derivatives_functorch(x: Tensor) -> bool:
    """Inspect currently supported PyTorch functorch wrapper layers."""
    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    raw_stack = ft.get_interpreter_stack()
    stack = (
        {}
        if raw_stack is None
        else {interp.level(): interp.key() for interp in raw_stack}
    )
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
            if _has_jvp_tangent_at_level(current, level):
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
    eager_primal: bool | str = False,
) -> Tensor:
    """Build an invariant Fermi density from dxtb's shared primal state.

    ``eager_primal=True`` selects adaptive doubling only for detached eager
    root preparation. Differentiable and transformed evaluation must use the
    default fixed schedule.
    """
    setup = _fermi_free_energy_setup(
        hamiltonian,
        nel,
        kt,
        state,
        l_inv,
        mask,
        diff_steps,
        eager_primal=eager_primal,
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
    *,
    eager_primal: bool | str = False,
) -> _FermiFreeEnergySetup:
    work_dtype = torch.float64
    h = hamiltonian.to(dtype=work_dtype)
    li = None if l_inv is None else l_inv.to(dtype=work_dtype)
    # The explicit mask indexes the AO basis; state.orbital_mask indexes the
    # energy-sorted spectrum. AO validity may be noncontiguous.
    structural_mask = mask
    if structural_mask is None:
        structural_mask = state.orbital_mask
        if structural_mask is not None and structural_mask.ndim == h.ndim:
            structural_mask = structural_mask.select(-2, 0)
    h_base, spectral_base, active_base = _prepare_matrix(h, li, structural_mask)

    h = h_base.unsqueeze(-3).expand(*nel.shape, h_base.shape[-2], h_base.shape[-1])
    spectral = spectral_base.unsqueeze(-3).expand_as(h)
    active = active_base.unsqueeze(-2).expand(*nel.shape, active_base.shape[-1])
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
    active_count = active.sum(-1).clamp_min(1).to(work_dtype)
    slope_floor_numer = 4.0 * active_count * torch.finfo(work_dtype).eps
    primal_mode = _current_primal_fermi_mode(eager_primal)
    if primal_mode == "spectral":
        _require_detached_eager_primal(
            "Detached-primal spectral matrix-Fermi evaluation",
            hamiltonian,
            nel,
            kt,
            *((l_inv,) if l_inv is not None else ()),
        )
        base_pair = active_base.unsqueeze(-1) & active_base.unsqueeze(-2)
        finite_base = torch.where(
            base_pair, torch.isfinite(h_base), torch.ones_like(base_pair)
        ).all(dim=(-1, -2))
        safe_spectral_base = torch.where(
            finite_base[..., None, None],
            spectral_base,
            torch.zeros_like(spectral_base),
        )
        evals_base, evecs_base = torch.linalg.eigh(safe_spectral_base)
        active_evecs_base = torch.where(
            active_base.unsqueeze(-1), evecs_base, torch.zeros_like(evecs_base)
        )
        active_evecs = active_evecs_base.unsqueeze(-3).expand_as(h)
        col_norm_sq = (
            (active_evecs_base * active_evecs_base)
            .sum(dim=-2)
            .unsqueeze(-2)
            .expand(*nel.shape, evals_base.shape[-1])
        )
        ref_offset = (ref + offset).unsqueeze(-1)
        safe_ref_offset = torch.where(
            torch.isfinite(ref_offset), ref_offset, torch.zeros_like(ref_offset)
        )
        evals_rel = (
            evals_base.unsqueeze(-2).expand(*nel.shape, evals_base.shape[-1])
            - safe_ref_offset
        )
        precomputed_eigh = (evals_rel, active_evecs)
        active_any = active.any(-1)
        active_pair = active.unsqueeze(-1) & active_any.unsqueeze(-1).unsqueeze(
            -1
        )
        active_rel = torch.where(
            active.unsqueeze(-1) & active.unsqueeze(-2),
            relative,
            torch.zeros_like(relative),
        )
        finite_rel = torch.where(
            active.unsqueeze(-1) & active.unsqueeze(-2),
            torch.isfinite(relative),
            torch.ones_like(active.unsqueeze(-1) & active.unsqueeze(-2)),
        ).all(dim=(-1, -2))
        safe_rel_diag = torch.where(
            finite_rel.unsqueeze(-1) & active,
            active_rel.diagonal(dim1=-2, dim2=-1),
            torch.zeros_like(evals_rel),
        )
        off_diag_row_sum = (
            torch.where(
                finite_rel[..., None, None],
                active_rel.abs(),
                torch.zeros_like(active_rel),
            ).sum(dim=-1)
            - safe_rel_diag.abs()
        )
        max_off_diag = (
            torch.where(
                finite_rel[..., None, None],
                active_rel.abs(),
                torch.zeros_like(active_rel),
            )
            .masked_fill(
                torch.eye(
                    h.shape[-1], dtype=torch.bool, device=h.device
                ).expand_as(active_rel),
                0.0,
            )
            .amax(dim=(-1, -2))
        )
        del active_pair
        maximum_value = torch.finfo(work_dtype).max
        maximum_exponent = _FERMI_MAX_SCALED_NORM * float(
            2**_FERMI_MAX_SCALE_POWER
        )

        for step in range(diff_steps):
            finite_sh_kt = (
                torch.isfinite(shift) & torch.isfinite(safe_kt) & (safe_kt > 0.0)
            )
            fin_in = finite_rel & torch.where(
                active_any, finite_sh_kt, torch.ones_like(active_any)
            )
            s_shift = torch.where(
                fin_in & active_any, shift, torch.zeros_like(shift)
            )
            s_kt = torch.where(
                fin_in & active_any, safe_kt, torch.ones_like(safe_kt)
            )
            sh_diag = s_shift.unsqueeze(-1)
            opp_signs = ((safe_rel_diag < 0.0) & (sh_diag > 0.0)) | (
                (safe_rel_diag > 0.0) & (sh_diag < 0.0)
            )
            sub_ovf = (
                active
                & opp_signs
                & (
                    safe_rel_diag.abs()
                    > maximum_value - s_shift.abs().unsqueeze(-1)
                )
            ).any(-1)
            s_in = fin_in & ~sub_ovf
            s_shift = torch.where(s_in, s_shift, torch.zeros_like(s_shift))
            s_kt = torch.where(s_in, s_kt, torch.ones_like(s_kt))
            diag_diff = torch.where(
                active,
                (safe_rel_diag - s_shift.unsqueeze(-1)).abs(),
                torch.zeros_like(safe_rel_diag),
            )
            max_entry = torch.maximum(diag_diff.amax(dim=-1), max_off_diag)
            entry_ok = (max_entry / maximum_exponent <= s_kt) | ~active_any
            s_in = s_in & entry_ok
            s_kt = torch.where(s_in, s_kt, torch.ones_like(s_kt))
            row_abs_sum = torch.where(
                active,
                diag_diff + off_diag_row_sum,
                torch.zeros_like(diag_diff),
            )
            exp_bound = torch.where(
                s_in, row_abs_sum.amax(dim=-1) / s_kt, torch.zeros_like(s_kt)
            )
            req_pow = torch.ceil(
                torch.log2(exp_bound.clamp_min(1.0) / _FERMI_MAX_SCALED_NORM)
            ).to(torch.long)
            step_in_range = s_in & (req_pow <= _FERMI_MAX_SCALE_POWER)
            scaled_arg = (evals_rel - s_shift.unsqueeze(-1)) / (
                2.0 * s_kt.unsqueeze(-1)
            )
            occ_1d = 0.5 * (1.0 - torch.tanh(scaled_arg))
            count = (col_norm_sq * occ_1d).sum(dim=-1)
            slope = (col_norm_sq * occ_1d * (1.0 - occ_1d)).sum(
                dim=-1
            ) / s_kt
            count = torch.where(
                step_in_range, count, torch.full_like(count, float("nan"))
            )
            slope = torch.where(
                step_in_range, slope, torch.full_like(slope, float("nan"))
            )
            nonfinite_step = ~torch.isfinite(count) | ~torch.isfinite(slope)
            slope_ok = slope > (slope_floor_numer / s_kt)
            if step == 0:
                gap_mask = gap_mask | (diff_mask & ~nonfinite_step & ~slope_ok)
                diff_mask = diff_mask & ~gap_mask
            safe_slope = torch.where(slope_ok, slope, 1.0)
            raw_correction = (target - count) / safe_slope
            step_ok = slope_ok & (
                raw_correction.abs() <= _MAX_DIFF_STEP_KT * s_kt
            )
            correction = torch.where(
                nonfinite_step,
                torch.full_like(raw_correction, float("nan")),
                torch.where(step_ok, raw_correction, 0.0),
            )
            shift = shift + torch.where(diff_mask, correction, 0.0)

        occupation = _matrix_fermi_spectral(
            relative, shift, safe_kt, active, precomputed_eigh=precomputed_eigh
        )
    else:
        matrix_fermi: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = (
            _matrix_fermi_eager if primal_mode == "adaptive" else _matrix_fermi
        )
        slope_floor = slope_floor_numer / safe_kt
        step_bound = _MAX_DIFF_STEP_KT * safe_kt
        for step in range(diff_steps):
            density = matrix_fermi(relative, shift, safe_kt, active)
            diag_density = density.diagonal(dim1=-2, dim2=-1)
            count = diag_density.sum(-1)
            slope = (diag_density - (density * density).sum(dim=-1)).sum(
                -1
            ) / safe_kt
            nonfinite_step = ~torch.isfinite(count) | ~torch.isfinite(slope)
            slope_ok = slope > slope_floor
            if step == 0:
                gap_mask = gap_mask | (diff_mask & ~nonfinite_step & ~slope_ok)
                diff_mask = diff_mask & ~gap_mask
            safe_slope = torch.where(slope_ok, slope, 1.0)
            raw_correction = (target - count) / safe_slope
            step_ok = slope_ok & (raw_correction.abs() <= step_bound)
            correction = torch.where(
                nonfinite_step,
                torch.full_like(raw_correction, float("nan")),
                torch.where(step_ok, raw_correction, 0.0),
            )
            shift = shift + torch.where(diff_mask, correction, 0.0)
        occupation = matrix_fermi(relative, shift, safe_kt, active)

    x = (relative - shift[..., None, None] * eye) / safe_kt
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    x = torch.where(pair, x, torch.zeros_like(x))
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


def _gap_relative_mu(
    setup: _FermiFreeEnergySetup,
    identity: Tensor,
    mu_gap: Tensor | None = None,
) -> Tensor:
    mu_base = setup.ref + setup.offset + setup.shift
    if mu_gap is None:
        mu = mu_base
    else:
        mu = mu_base + torch.where(
            setup.gap_mask, mu_gap - mu_gap.detach(), 0.0
        )
    return (mu[..., None, None] * identity - setup.spectral) / setup.safe_kt


def fermi_free_energy_matrix(
    hamiltonian: Tensor,
    nel: Tensor,
    kt: Tensor,
    state: _FermiCoreState,
    *,
    l_inv: Tensor | None = None,
    mask: Tensor | None = None,
    diff_steps: int = 2,
    eager_primal: bool | str = False,
) -> Tensor:
    """Build the invariant finite-temperature free-energy spectral matrix."""
    setup = _fermi_free_energy_setup(
        hamiltonian,
        nel,
        kt,
        state,
        l_inv,
        mask,
        diff_steps,
        eager_primal=eager_primal,
    )
    active = setup.active
    entropy = _binary_entropy_matrix(setup.x, setup.occupation, active)
    identity = torch.diag_embed(active.to(torch.float64))
    count_delta = setup.target - setup.target.detach()
    if _tracks_derivatives(nel) or _tracks_derivatives(count_delta):
        gap_response, mu_gap = _gap_electron_response_and_mu(
            setup.h,
            setup.spectral,
            active,
            setup.below,
            setup.gap_mask,
            setup.safe_kt,
        )
        gap_term = _gap_relative_mu(setup, identity, mu_gap) @ gap_response
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
    eager_primal: bool | str = False,
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
        hamiltonian,
        nel,
        kt,
        state,
        l_inv,
        mask,
        diff_steps,
        eager_primal=eager_primal,
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
        gap_response, mu_gap = _gap_electron_response_and_mu(
            setup.h,
            setup.spectral,
            active,
            setup.below,
            setup.gap_mask,
            setup.safe_kt,
        )
        gap_term = (
            _gap_relative_mu(setup, identity, mu_gap)
            * gap_response.transpose(-2, -1)
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
    # keeping the inverse systems nonsingular without squaring the condition
    # number of x.
    for _ in range(14):
        inverse = torch.linalg.inv(x)
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
    its matrix inversion raises rather than manufacturing a response.
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
