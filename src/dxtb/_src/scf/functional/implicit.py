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
from dataclasses import dataclass, replace

import torch
from tad_mctc.storch.implicit import implicit_fixed_point
from tad_mctc.storch.linsolve import STATUS_CONVERGED
from tad_mctc.storch.linsolve import LinearSolveResult as SolveResult
from tad_mctc.storch.linsolve import bicgstab

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import SCFConvergenceError

from .core import replace_tensor_leaves, tensor_leaves

__all__ = ["LinearSolveResult", "implicit_root", "solve_linear"]

RawMap = Callable[..., Tensor]

_LINEAR_SOLVE_RTOL = 1.0e-10
_LINEAR_SOLVE_ATOL = 1.0e-12


@dataclass(frozen=True, eq=False)
class LinearSolveResult:
    """Private diagnostics from a matrix-free linear solve."""

    solution: Tensor
    residual: Tensor
    iterations: Tensor
    converged: Tensor


def _vector(value: Tensor) -> Tensor:
    """The whole state is one vector; a scalar becomes a 1-vector."""
    return value.reshape(1) if value.ndim == 0 else value


def _on_vector(function: Callable[..., Tensor], shape: torch.Size):
    """Operator on the state shape, applied to `_vector` values."""

    def apply(value: Tensor, *args: Tensor) -> Tensor:
        return function(value.reshape(shape), *args).reshape(value.shape)

    return apply


def solve_linear(
    operator: Callable[[Tensor], Tensor],
    rhs: Tensor,
    *,
    maxiter: int | None = None,
    rtol: float = _LINEAR_SOLVE_RTOL,
    atol: float = _LINEAR_SOLVE_ATOL,
) -> LinearSolveResult:
    """Solve a nonsymmetric system with a matrix-free BiCGSTAB.

    The whole tensor is one vector. Convergence and breakdown handling use
    tensor masks (``tad_mctc.storch.linsolve.bicgstab``), so this routine
    composes with ``torch.func`` transforms and can be vmapped over
    prepared roots. Its relative threshold is ``max(rtol, 10 eps)``.
    """
    size = rhs.numel()
    steps = 2 * size if maxiter is None else maxiter
    if steps < 1:
        raise ValueError("A matrix-free linear solve requires maxiter >= 1.")

    vector = _vector(rhs)
    result = bicgstab(
        _on_vector(operator, rhs.shape),
        vector,
        rtol=rtol,
        atol=atol,
        maxiter=steps,
        vdims=vector.ndim,
    )
    return LinearSolveResult(
        solution=result.solution.reshape(rhs.shape),
        residual=result.residual,
        iterations=result.iterations,
        converged=result.status == STATUS_CONVERGED,
    )


def _require_linear_convergence(
    result: LinearSolveResult | SolveResult,
) -> Tensor:
    """Make a failed transformed solve observable without host inspection.

    Primal SCF convergence is checked eagerly before entering this module.
    The matrix-free tangent/adjoint solve can only report a tensor status
    inside ``torch.func`` transforms. Returning NaNs on a failed linear solve
    prevents its finite but unconverged iterate from masquerading as a valid
    physical-root derivative, while remaining compatible with ``vmap``.
    """
    nan = torch.full_like(result.solution, torch.nan)
    return torch.where(result.converged, result.solution, nan)


def implicit_root(
    root_x: Tensor, *parameters: Tensor, raw_map: RawMap
) -> Tensor:
    """Return ``root_x`` with derivatives from ``G(x, p) - x = 0``.

    ``root_x`` must already be validated by an eager preparation boundary.
    All differentiable map inputs are explicit tensor arguments in
    ``parameters``; ``raw_map`` is static context and may contain only static
    SCF structure. The returned value is not finalized into an observable.

    The response equations ``(I - G_x) x_dot = G_p p_dot`` and
    ``(I - G_x)^T lambda = x_bar`` are solved matrix-free with BiCGSTAB by
    ``tad_mctc.storch.implicit.implicit_fixed_point``, whose nested
    derivatives are those of the linear systems, not of the iterations. A
    response solve that does not converge gives NaN.
    """
    # The solver trajectory selects a primal value only. Its derivatives are
    # replaced by the explicit IFT response from ``parameters``.
    return implicit_fixed_point(
        raw_map,
        root_x,
        *parameters,
        method="bicgstab",
        rtol=_LINEAR_SOLVE_RTOL,
        atol=_LINEAR_SOLVE_ATOL,
    )


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
    certification: object | None
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
    x_atol: float,
    x_atol_max: float,
    final_valid: Tensor,
    raw_map: RawMap,
    parameters: tuple[Tensor, ...],
) -> _PreparedSCFRoot:
    """Validate the root against the exact inputs before differentiation."""
    if maxiter == 0:
        raise SCFConvergenceError(
            "Implicit physical-root differentiation requires maxiter > 0."
        )
    certification = getattr(outcome, "certification", None)
    if certification is None:
        raise SCFConvergenceError(
            "Implicit physical-root differentiation requires an "
            "authoritatively certified SCF root."
        )
    converged = outcome.converged.all()
    certified = certification.certified.all()
    if bool(converged) != bool(certified):
        raise RuntimeError(
            "SCF outcome convergence and physical certification disagree."
        )
    valid = converged & certified & final_valid

    # Re-evaluate the raw physical map with the exact tensors that will be
    # passed to the implicit attachment. This catches a root prepared for a
    # different finite parameter value before it can enter the IFT boundary.
    with torch.no_grad():
        independent_residual = raw_map(outcome.root_x, *parameters)
        independent_residual = independent_residual - outcome.root_x
        residual_norms = torch.stack(
            (
                torch.linalg.vector_norm(independent_residual),
                independent_residual.abs().amax(),
            )
        )
        residual_valid = torch.isfinite(independent_residual).all()
        residual_valid = (
            residual_valid
            & (residual_norms[0] <= x_atol)
            & (residual_norms[1] <= x_atol_max)
        )
    valid = valid & residual_valid
    if not bool(valid):
        raise SCFConvergenceError(
            "Cannot differentiate an implicit physical root because the "
            "primal SCF/Fermi solve did not converge for the supplied "
            "physical inputs."
        )
    return _PreparedSCFRoot(
        root_x=outcome.root_x,
        map_result=outcome.map_result,
        iterations=outcome.iterations,
        converged=outcome.converged,
        residual=outcome.residual,
        trajectory_status=outcome.trajectory_status,
        damping_state=outcome.damping_state,
        certification=getattr(outcome, "certification", None),
        raw_map=raw_map,
        parameters=parameters,
    )


@dataclass(frozen=True, eq=False)
class _SCFMapTensorTree:
    """Explicit floating tensor adapter for the existing E4core raw map.

    Structural objects and integer/bool masks remain static. Every floating
    tensor stored directly on ``SCFInputs``, its integral/Fermi records, or the
    immutable functional interaction tree becomes an explicit custom-function
    input. The interaction tree paths are resolved once when prepared.
    """

    template: object
    representation: int
    potential_layout: object | None
    input_fields: tuple[str, ...]
    interaction_tensor_paths: tuple[str, ...]
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

        interaction_tensor_paths = tuple(
            path
            for path, value in tensor_leaves(inputs.interactions)
            if value.is_floating_point()
        )
        names.extend(
            f"interactions.{path}" for path in interaction_tensor_paths
        )

        adapter = cls(
            template=inputs,
            representation=representation,
            potential_layout=potential_layout,
            input_fields=input_fields,
            interaction_tensor_paths=interaction_tensor_paths,
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
        interaction_leaves = dict(tensor_leaves(inputs.interactions))
        values.extend(
            interaction_leaves[path] for path in self.interaction_tensor_paths
        )
        # Functional interaction data may contain expanded views (for
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

        interaction_changes = {
            path: parameters[cursor + index]
            for index, path in enumerate(self.interaction_tensor_paths)
        }
        cursor += len(self.interaction_tensor_paths)
        interactions = replace_tensor_leaves(
            self.template.interactions, interaction_changes
        )

        inputs = replace(
            self.template,
            integrals=integrals,
            interactions=interactions,
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
