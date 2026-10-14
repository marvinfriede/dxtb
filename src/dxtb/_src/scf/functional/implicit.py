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
"""Transform-compatible implicit attachment for an already prepared root.

This module deliberately knows nothing about SCF preparation or finalization.
The caller supplies the accepted root value, explicit tensor parameters, and
the raw physical map. Primal convergence checks belong before this boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, replace

import torch

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import SCFConvergenceError

__all__ = ["LinearSolveResult", "implicit_root", "solve_linear"]

RawMap = Callable[..., Tensor]

_LINEAR_SOLVE_EPS_MULTIPLIER = 10.0


@dataclass(frozen=True, eq=False)
class LinearSolveResult:
    """Private diagnostics from a matrix-free linear solve."""

    solution: Tensor
    residual: Tensor
    iterations: Tensor
    converged: Tensor


def _dot(left: Tensor, right: Tensor) -> Tensor:
    return torch.sum(left.reshape(-1) * right.reshape(-1))


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    tiny = torch.finfo(denominator.dtype).tiny
    safe = torch.where(
        denominator.abs() > tiny, denominator, torch.ones_like(denominator)
    )
    return numerator / safe


def solve_linear(
    operator: Callable[[Tensor], Tensor],
    rhs: Tensor,
    *,
    maxiter: int | None = None,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> LinearSolveResult:
    """Solve a nonsymmetric system with a fixed-trip matrix-free BiCGSTAB.

    The iteration count depends only on the static tensor shape. Convergence
    and breakdown handling use tensor masks, so this routine composes with
    ``torch.func`` transforms and can be vmapped over prepared roots.
    """
    size = rhs.numel()
    steps = 2 * size if maxiter is None else maxiter
    if steps < 1:
        raise ValueError("A matrix-free linear solve requires maxiter >= 1.")

    x = torch.zeros_like(rhs)
    residual = rhs - operator(x)
    shadow = residual
    direction = torch.zeros_like(rhs)
    image = torch.zeros_like(rhs)
    rho_previous = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    alpha_previous = torch.ones_like(rho_previous)
    omega_previous = torch.ones_like(rho_previous)
    rhs_norm = torch.linalg.vector_norm(rhs)
    # A requested tolerance below roundoff can reject a useful low-precision
    # solution. Keep the float64 threshold unchanged while imposing a modest
    # dtype-relative floor for lower-precision solves.
    effective_rtol = max(
        rtol,
        _LINEAR_SOLVE_EPS_MULTIPLIER * torch.finfo(rhs.dtype).eps,
    )
    target = atol + effective_rtol * rhs_norm
    active = torch.linalg.vector_norm(residual) > target
    iterations = torch.zeros((), dtype=torch.int64, device=rhs.device)

    for _ in range(steps):
        rho = _dot(shadow, residual)
        beta = _safe_divide(rho * alpha_previous, rho_previous * omega_previous)
        trial_direction = residual + beta * (direction - omega_previous * image)
        direction = torch.where(active, trial_direction, direction)
        image_new = operator(direction)
        alpha = _safe_divide(rho, _dot(shadow, image_new))
        intermediate = residual - alpha * image_new
        correction = operator(intermediate)
        omega = _safe_divide(
            _dot(correction, intermediate), _dot(correction, correction)
        )
        trial_x = x + alpha * direction + omega * intermediate
        trial_residual = intermediate - omega * correction
        x = torch.where(active, trial_x, x)
        residual = torch.where(active, trial_residual, residual)
        image = torch.where(active, image_new, image)
        rho_previous = torch.where(active, rho, rho_previous)
        alpha_previous = torch.where(active, alpha, alpha_previous)
        omega_previous = torch.where(active, omega, omega_previous)
        next_active = torch.linalg.vector_norm(residual) > target
        iterations = iterations + active.to(torch.int64)
        active = next_active

    residual_norm = torch.linalg.vector_norm(rhs - operator(x))
    return LinearSolveResult(
        solution=x,
        residual=residual_norm,
        iterations=iterations,
        converged=residual_norm <= target,
    )


def _require_linear_convergence(result: LinearSolveResult) -> Tensor:
    """Make a failed transformed solve observable without host inspection.

    Primal SCF convergence is checked eagerly before entering this module.
    The matrix-free tangent/adjoint solve can only report a tensor status
    inside ``torch.func`` transforms. Returning NaNs on a failed linear solve
    prevents its finite but unconverged iterate from masquerading as a valid
    physical-root derivative, while remaining compatible with ``vmap``.
    """
    nan = torch.full_like(result.solution, torch.nan)
    return torch.where(result.converged, result.solution, nan)


class _ImplicitRoot(torch.autograd.Function):
    """Attach derivatives defined by the raw fixed-point equation."""

    generate_vmap_rule = True

    @staticmethod
    def forward(root_x: Tensor, *args: object) -> Tensor:
        return root_x.view_as(root_x)

    @staticmethod
    def setup_context(ctx: object, inputs: tuple, output: Tensor) -> None:
        _, *args = inputs
        *parameters, raw_map = args
        ctx.raw_map = raw_map
        tensors = (*parameters, output)
        ctx.save_for_backward(*tensors)
        ctx.save_for_forward(*tensors)

    @staticmethod
    def backward(ctx: object, cotangent: Tensor) -> tuple:
        saved = ctx.saved_tensors
        parameters, output = saved[:-1], saved[-1]
        raw_map = ctx.raw_map

        _, pullback_x = torch.func.vjp(
            lambda x: raw_map(x, *parameters), output
        )

        def adjoint_operator(value: Tensor) -> Tensor:
            return value - pullback_x(value)[0]

        adjoint = _require_linear_convergence(
            solve_linear(adjoint_operator, cotangent)
        )
        _, pullback_parameters = torch.func.vjp(
            lambda *values: raw_map(output, *values), *parameters
        )
        parameter_cotangents = pullback_parameters(adjoint)
        return (torch.zeros_like(output), *parameter_cotangents, None)

    @staticmethod
    def jvp(ctx: object, root_tangent: Tensor | None, *args: object) -> Tensor:
        del root_tangent
        *parameter_tangents, raw_map_tangent = args
        del raw_map_tangent
        saved = ctx.saved_tensors
        parameters, output = saved[:-1], saved[-1]
        raw_map = ctx.raw_map

        _, parameter_response = torch.func.jvp(
            lambda *values: raw_map(output, *values),
            tuple(parameters),
            tuple(parameter_tangents),
        )

        def tangent_operator(value: Tensor) -> Tensor:
            _, mapped_tangent = torch.func.jvp(
                lambda x: raw_map(x, *parameters), (output,), (value,)
            )
            return value - mapped_tangent

        return _require_linear_convergence(
            solve_linear(tangent_operator, parameter_response)
        )


def implicit_root(
    root_x: Tensor, *parameters: Tensor, raw_map: RawMap
) -> Tensor:
    """Return ``root_x`` with derivatives from ``G(x, p) - x = 0``.

    ``root_x`` must already be validated by an eager preparation boundary.
    All differentiable map inputs are explicit tensor arguments in
    ``parameters``; ``raw_map`` is static context and may contain only static
    SCF structure. The returned value is not finalized into an observable.
    """
    # The solver trajectory selects a primal value only. Its derivatives are
    # replaced by the explicit IFT response from ``parameters`` below.
    return _ImplicitRoot.apply(root_x.detach(), *parameters, raw_map)


@dataclass(frozen=True, eq=False)
class _PreparedSCFRoot:
    """Validated primal state retained across the transform boundary."""

    root_x: Tensor
    map_result: object
    iterations: Tensor
    converged: Tensor
    residual: Tensor
    trajectory_status: object | None
    damping_state: object
    raw_map: RawMap
    parameters: tuple[Tensor, ...]

    def attach(self) -> Tensor:
        """Attach only the implicit root derivative; leave finalization out."""
        return implicit_root(
            self.root_x, *self.parameters, raw_map=self.raw_map
        )


def _prepare_scf_root(
    outcome: object,
    *,
    maxiter: int,
    final_valid: Tensor,
    raw_map: RawMap,
    parameters: tuple[Tensor, ...],
) -> _PreparedSCFRoot:
    """Validate primal/Fermi status before any differentiation transform."""
    if maxiter == 0:
        raise SCFConvergenceError(
            "Implicit physical-root differentiation requires maxiter > 0."
        )
    valid = outcome.converged.all() & final_valid
    status = outcome.trajectory_status
    if status is not None:
        valid = (
            valid
            & status.fermi_converged.all()
            & status.electron_count_ok.all()
            & ~status.invalid_temperature.any()
            & ~status.too_many_electrons.any()
        )
    if not bool(valid):
        raise SCFConvergenceError(
            "Cannot differentiate an implicit physical root because the "
            "primal SCF/Fermi solve did not converge."
        )
    return _PreparedSCFRoot(
        root_x=outcome.root_x,
        map_result=outcome.map_result,
        iterations=outcome.iterations,
        converged=outcome.converged,
        residual=outcome.residual,
        trajectory_status=outcome.trajectory_status,
        damping_state=outcome.damping_state,
        raw_map=raw_map,
        parameters=parameters,
    )


@dataclass(frozen=True, eq=False)
class _SCFMapTensorTree:
    """Explicit floating tensor adapter for the existing E4core raw map.

    Structural objects and integer/bool masks remain static. Every floating
    tensor stored directly on ``SCFInputs``, its integral/Fermi records, or a
    functional interaction cache becomes an explicit custom-function input.
    Cache class/field schemas are resolved once when the adapter is prepared.
    """

    template: object
    representation: int
    potential_layout: object | None
    input_fields: tuple[str, ...]
    cache_fields: tuple[tuple[object, tuple[str, ...]], ...]
    parameter_paths: tuple[str, ...]

    @classmethod
    def prepare(
        cls,
        inputs: object,
        representation: int,
        potential_layout: object | None = None,
    ) -> tuple[_SCFMapTensorTree, tuple[Tensor, ...]]:
        from .data import SCFInputs

        if not isinstance(inputs, SCFInputs):
            raise TypeError("The implicit SCF adapter requires SCFInputs.")

        input_fields = tuple(
            name
            for name in (
                "numbers",
                "n0",
                "nel",
                "occupation",
                "inverse_cholesky",
                "orbital_mask",
            )
            if isinstance(getattr(inputs, name), torch.Tensor)
            and getattr(inputs, name).is_floating_point()
        )
        names = list(input_fields)
        integral_fields = tuple(
            name
            for name in ("hcore", "overlap", "dipole", "quadrupole")
            if isinstance(getattr(inputs.integrals, name), torch.Tensor)
            and getattr(inputs.integrals, name).is_floating_point()
        )
        names.extend(f"integrals.{name}" for name in integral_fields)
        fermi_fields: tuple[str, ...] = ()
        if inputs.fermi_settings is not None:
            fermi_fields = tuple(
                name
                for name in ("kt", "threshold")
                if isinstance(
                    getattr(inputs.fermi_settings, name), torch.Tensor
                )
                and getattr(inputs.fermi_settings, name).is_floating_point()
            )
            names.extend(f"fermi_settings.{name}" for name in fermi_fields)

        cache_fields: list[tuple[object, tuple[str, ...]]] = []
        for index, cache in enumerate(inputs.interaction_data.values()):
            fields = tuple(
                name
                for name in _public_slot_names(type(cache))
                if isinstance(getattr(cache, name, None), torch.Tensor)
                and getattr(cache, name).is_floating_point()
            )
            cache_fields.append((type(cache), fields))
            names.extend(f"cache.{index}.{name}" for name in fields)

        adapter = cls(
            template=inputs,
            representation=representation,
            potential_layout=potential_layout,
            input_fields=input_fields,
            cache_fields=tuple(cache_fields),
            parameter_paths=tuple(names),
        )
        return adapter, adapter.flatten()

    def flatten(self) -> tuple[Tensor, ...]:
        inputs = self.template
        values = [getattr(inputs, name) for name in self.input_fields]
        for name in ("hcore", "overlap", "dipole", "quadrupole"):
            value = getattr(inputs.integrals, name)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                values.append(value)
        if inputs.fermi_settings is not None:
            for name in ("kt", "threshold"):
                value = getattr(inputs.fermi_settings, name)
                if (
                    isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                ):
                    values.append(value)
        for cache, (_, fields) in zip(
            inputs.interaction_data.values(), self.cache_fields
        ):
            values.extend(getattr(cache, name) for name in fields)
        # Functional interaction caches may contain expanded views (for
        # example a field broadcast over atoms). Explicit transform leaves
        # must have non-overlapping storage for forward-AD dual attachment.
        return tuple(value.contiguous().clone() for value in values)

    def _rebuild(self, parameters: tuple[Tensor, ...]):
        from dxtb._src.constants import labels
        from dxtb._src.integral.container import IntegralMatrices

        from .data import SCFInputs
        from .map import fock_map, potential_map, scf_map

        if len(parameters) != len(self.parameter_paths):
            raise ValueError(
                "The SCF input tensor tree shape changed after preparation."
            )
        cursor = 0
        changes: dict[str, Tensor] = {}
        for name in self.input_fields:
            changes[name] = parameters[cursor]
            cursor += 1

        integral_changes: dict[str, Tensor | None] = {}
        for name in ("hcore", "overlap", "dipole", "quadrupole"):
            value = getattr(self.template.integrals, name)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                integral_changes[name] = parameters[cursor]
                cursor += 1
            else:
                integral_changes[name] = value
        integrals = IntegralMatrices(**integral_changes)

        fermi_settings = self.template.fermi_settings
        if fermi_settings is not None:
            fermi_changes = {}
            for name in ("kt", "threshold"):
                value = getattr(fermi_settings, name)
                if (
                    isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                ):
                    fermi_changes[name] = parameters[cursor]
                    cursor += 1
            fermi_settings = replace(fermi_settings, **fermi_changes)

        interaction_data = copy(self.template.interaction_data)
        for index, (key, cache) in enumerate(
            self.template.interaction_data.items()
        ):
            cache_copy = copy(cache)
            # Cache keys and field schemas are static for this prepared root.
            _, fields = self.cache_fields[index]
            for name in fields:
                setattr(cache_copy, name, parameters[cursor])
                cursor += 1
            interaction_data[key] = cache_copy

        inputs = replace(
            self.template,
            integrals=integrals,
            interaction_data=interaction_data,
            fermi_settings=fermi_settings,
            **changes,
        )
        if self.representation == labels.SCP_MODE_CHARGE:
            return inputs, scf_map
        if self.representation == labels.SCP_MODE_POTENTIAL:
            return inputs, lambda x, current: potential_map(
                x, current, self.potential_layout
            )
        if self.representation == labels.SCP_MODE_FOCK:
            return inputs, fock_map
        raise ValueError(
            f"Unsupported implicit representation {self.representation}."
        )

    def evaluate(self, x: Tensor, *parameters: Tensor):
        inputs, map_fn = self._rebuild(parameters)
        return map_fn(x, inputs)

    def __call__(self, x: Tensor, *parameters: Tensor) -> Tensor:
        return self.evaluate(x, *parameters).mapped_x


def _public_slot_names(cls: type) -> tuple[str, ...]:
    """Return declared public cache slots in stable base-to-derived order."""
    names: list[str] = []
    for parent in reversed(cls.__mro__):
        slots = vars(parent).get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(name for name in slots if not name.startswith("_"))
    return tuple(dict.fromkeys(names))
