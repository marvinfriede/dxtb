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
"""Internal response-coordinate preparation for certified SCF roots."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import SCFConvergenceError

from .certification import (
    certify_scf_core_state,
    resolve_certification_thresholds,
)
from .data import (
    PotentialLayout,
    pack_charges,
    pack_potential,
    potential_layout,
)
from .implicit import _SCFMapTensorTree
from .map import potential_from_charges
from .problem import SCFPhysicalState, SCFProblem, SCFRepresentation
from .solver import SCFSolverSettings

__all__ = [
    "SCFResponseSettings",
    "SCFResponseRoot",
    "prepare_response_root",
]


@dataclass(frozen=True)
class SCFResponseSettings:
    """Response coordinates and matrix-free linear-solve controls."""

    representation: SCFRepresentation | None = None
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    maxiter: int | None = None

    def __post_init__(self) -> None:
        if self.representation is not None and not isinstance(
            self.representation, SCFRepresentation
        ):
            raise TypeError(
                "Response representation must be an SCFRepresentation."
            )
        if (
            not isfinite(self.rtol)
            or not isfinite(self.atol)
            or self.rtol < 0.0
            or self.atol < 0.0
        ):
            raise ValueError(
                "Response solve tolerances must be finite and non-negative."
            )
        if self.maxiter is not None and self.maxiter < 1:
            raise ValueError("Response solve maxiter must be positive.")


@dataclass(frozen=True, eq=False)
class SCFResponseOutcome:
    """Minimal method-independent outcome used after root preparation."""

    root_x: Tensor
    map_result: object
    iterations: Tensor
    converged: Tensor
    residual: object
    trajectory_status: object | None
    damping_state: object
    certification: object


@dataclass(frozen=True, eq=False)
class SCFResponseRoot:
    """Certified physical root expressed in its IFT response coordinates."""

    problem: SCFProblem
    root_x: Tensor
    map_result: object
    certification: object
    adapter: _SCFMapTensorTree
    parameters: tuple[Tensor, ...]
    independent_map_result: object | None
    outcome: SCFResponseOutcome
    settings: SCFResponseSettings
    source_representation: SCFRepresentation


def _response_candidate(
    outcome: object,
    representation: SCFRepresentation,
) -> tuple[SCFProblem, Tensor]:
    source_problem = outcome.final_problem
    inputs = source_problem.inputs
    spectral = outcome.map_result.spectral
    problem: SCFProblem

    if representation is SCFRepresentation.CHARGE:
        problem = SCFProblem(inputs, SCFRepresentation.CHARGE)
        root_x = pack_charges(spectral.charges, inputs.charge_layout)
        return problem, root_x

    if representation is SCFRepresentation.POTENTIAL:
        potential = potential_from_charges(spectral.charges, inputs)
        layout = potential_layout(potential)
        problem = SCFProblem(
            inputs, SCFRepresentation.POTENTIAL, potential_layout=layout
        )
        return problem, pack_potential(potential, layout)

    if representation is SCFRepresentation.FOCK:
        problem = SCFProblem(inputs, SCFRepresentation.FOCK)
        return problem, problem.rebuild_hamiltonian(spectral.charges)

    raise ValueError(
        f"Unsupported SCF response representation {representation!r}."
    )


def prepare_response_root(
    outcome: object,
    solver_settings: SCFSolverSettings,
    response_settings: SCFResponseSettings | None = None,
) -> SCFResponseRoot:
    """Prepare and validate one response-coordinate root without iteration."""
    if outcome.final_problem is None or outcome.certification is None:
        raise SCFConvergenceError(
            "Response preparation requires a certified physical SCF root."
        )
    if not bool(outcome.converged.all()) or not bool(
        outcome.certification.certified.all()
    ):
        raise SCFConvergenceError(
            "Response preparation cannot rescue an uncertified primal root."
        )

    settings = (
        SCFResponseSettings()
        if response_settings is None
        else response_settings
    )
    source_problem = outcome.final_problem
    target = (
        source_problem.representation
        if settings.representation is None
        else settings.representation
    )

    independent_map_result = None
    if target is source_problem.representation:
        problem = source_problem
        root_x = outcome.root_x
        map_result = outcome.map_result
        certification = outcome.certification
    else:
        problem, root_x = _response_candidate(outcome, target)

    adapter, parameters = _SCFMapTensorTree.prepare(
        problem.inputs,
        target,
        problem.potential_layout,
    )

    if target is not source_problem.representation:
        # Alternate response coordinates receive exactly one physical map.
        # There is intentionally no response-root refinement loop.
        with torch.no_grad():
            map_result = adapter.evaluate(root_x, *parameters)
        independent_map_result = map_result
        state = SCFPhysicalState(root_x, map_result)
        thresholds = resolve_certification_thresholds(
            problem,
            x_atol=solver_settings.x_atol,
            x_atol_max=solver_settings.x_atol_max,
            dtype=root_x.dtype,
            requested_x_atol=solver_settings.requested_x_atol,
            requested_x_atol_max=solver_settings.requested_x_atol_max,
        )
        certification = certify_scf_core_state(problem, state, thresholds)
        if not bool(certification.certified.all()):
            source = source_problem.representation.value
            target_name = target.value
            raise SCFConvergenceError(
                "Converted implicit response root failed independent S2 "
                f"certification ({source} -> {target_name}) at the requested "
                "primal tolerances."
            )

    response_outcome = SCFResponseOutcome(
        root_x=root_x,
        map_result=map_result,
        iterations=outcome.iterations,
        converged=(
            outcome.converged
            if target is source_problem.representation
            else certification.certified
        ),
        residual=outcome.residual,
        trajectory_status=outcome.trajectory_status,
        damping_state=outcome.damping_state,
        certification=certification,
    )

    return SCFResponseRoot(
        problem=problem,
        root_x=root_x,
        map_result=map_result,
        certification=certification,
        adapter=adapter,
        parameters=parameters,
        independent_map_result=independent_map_result,
        outcome=response_outcome,
        settings=settings,
        source_representation=source_problem.representation,
    )
