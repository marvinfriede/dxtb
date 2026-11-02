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
"""Eager research methods for exact-energy and reference energy DIIS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch
from scipy.optimize import minimize

from dxtb._src.scf.functional.energy import electronic_energy_from_density
from dxtb._src.scf.functional.map import density_to_charges_and_core_energy
from dxtb._src.scf.functional.method import (
    SCFMethodContext,
    SCFMethodProposal,
    simple_mix,
)
from dxtb._src.scf.functional.problem import (
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
)


@dataclass(frozen=True, eq=False)
class ResearchDIISState:
    densities: tuple[torch.Tensor, ...]
    energies: tuple[torch.Tensor, ...]
    focks: tuple[torch.Tensor, ...]


@dataclass(eq=False)
class ResearchDIISMethod:
    """Prototype simplex energy interpolation for benchmark execution."""

    mode: str = "exact"
    capacity: int = 6
    records: list[dict[str, object]] = field(default_factory=list)
    supports_fixed_trip: ClassVar[bool] = False
    required_representation: ClassVar[SCFRepresentation] = (
        SCFRepresentation.FOCK
    )

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state: ResearchDIISState | None,
        context: SCFMethodContext,
    ) -> SCFMethodProposal:
        if (
            problem.inputs.fermi_settings is not None
            and problem.inputs.fermi_settings.smear
        ):
            raise NotImplementedError(
                "The fixed-occupation EDIIS research objective does not support active smearing."
            )
        density = physical_state.map_result.spectral.density.detach()
        energy = electronic_energy_from_density(density, problem.inputs).sum()
        fock = context.proposal_x.detach()
        if method_state is None:
            histories = (density,), (energy,), (fock,)
        else:
            histories = (
                (density, *method_state.densities)[: self.capacity],
                (energy, *method_state.energies)[: self.capacity],
                (fock, *method_state.focks)[: self.capacity],
            )
        densities, energies, focks = histories

        if len(densities) == 1:
            target = fock
        else:
            coefficients, optimizer_record = _minimize_coefficients(
                self.mode, densities, energies, focks, problem
            )
            trial_density = _interpolate_density(coefficients, densities)
            trial_energy = electronic_energy_from_density(
                trial_density, problem.inputs
            ).sum()
            self.records.append(
                {
                    "history_size": len(densities),
                    "coefficients": coefficients.tolist(),
                    "selected_energy": float(trial_energy),
                    "best_vertex_energy": float(
                        torch.min(torch.stack(energies))
                    ),
                    **optimizer_record,
                }
            )
            if self.mode == "exact":
                charges, _ = density_to_charges_and_core_energy(
                    trial_density, problem.inputs
                )
                target = problem.rebuild_hamiltonian(charges)
            else:
                target = torch.einsum(
                    "i,ijk->jk", coefficients, torch.stack(focks)
                )

        proposal = simple_mix(physical_state.x, target, context.damp)
        state = ResearchDIISState(*histories)
        return SCFMethodProposal(proposal, state)


def _interpolate_density(
    coefficients: torch.Tensor, densities: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return torch.einsum("i,ijk->jk", coefficients, torch.stack(densities))


def exact_energy_value_gradient(
    coefficients: torch.Tensor,
    densities: tuple[torch.Tensor, ...],
    problem: SCFProblem,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact xTB interpolation energy and its Fock-derived gradient."""
    trial = _interpolate_density(coefficients, densities)
    energy = electronic_energy_from_density(trial, problem.inputs).sum()
    charges, _ = density_to_charges_and_core_energy(trial, problem.inputs)
    fock = problem.rebuild_hamiltonian(charges)
    gradient = torch.einsum("ij,kij->k", fock, torch.stack(densities))
    return energy, gradient


def adiis_model_matrices(
    densities: tuple[torch.Tensor, ...],
    focks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the PySCF ADIIS reference linear and quadratic coefficients."""
    density_stack = torch.stack(densities)
    fock_stack = torch.stack(focks)
    overlaps = torch.einsum("iab,jab->ij", density_stack, fock_stack)
    d_fn = overlaps[:, 0]
    dn_f = overlaps[0]
    dn_fn = overlaps[0, 0]
    linear = 2.0 * (d_fn - dn_fn)
    quadratic = overlaps - d_fn[:, None] - dn_f[None, :] + dn_fn
    return linear, quadratic


def _minimize_coefficients(
    mode: str,
    densities: tuple[torch.Tensor, ...],
    energies: tuple[torch.Tensor, ...],
    focks: tuple[torch.Tensor, ...],
    problem: SCFProblem,
) -> tuple[torch.Tensor, dict[str, object]]:
    count = len(densities)
    density_stack = torch.stack(densities)
    energy_stack = torch.stack(energies)
    fock_stack = torch.stack(focks)
    pair = torch.einsum(
        "ijab,ijab->ij",
        density_stack[:, None] - density_stack[None, :],
        fock_stack[:, None] - fock_stack[None, :],
    )
    if mode in ("classic", "quadratic"):
        model_matrix = pair
        correction = 1.0 if mode == "classic" else 0.25

        def objective_and_gradient(
            coefficients: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            values = torch.as_tensor(coefficients, dtype=density_stack.dtype)
            value = values @ energy_stack - correction * (
                values @ model_matrix @ values
            )
            gradient = energy_stack - 2.0 * correction * model_matrix @ values
            return float(value), gradient.numpy()

    elif mode == "adiis":
        linear, curvature = adiis_model_matrices(densities, focks)

        def objective_and_gradient(
            coefficients: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            values = torch.as_tensor(coefficients, dtype=density_stack.dtype)
            value = values @ linear + values @ curvature @ values
            gradient = linear + 2.0 * curvature @ values
            return float(value), gradient.numpy()

    elif mode == "exact":

        def objective_and_gradient(
            coefficients: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            values = torch.as_tensor(coefficients, dtype=density_stack.dtype)
            energy, gradient = exact_energy_value_gradient(
                values, densities, problem
            )
            return float(energy), gradient.numpy()

    else:
        raise ValueError(f"Unknown research DIIS mode {mode!r}.")

    constraint = {"type": "eq", "fun": lambda values: values.sum() - 1.0}
    bounds = [(0.0, 1.0)] * count
    starts = [
        np.full(count, 1.0 / count),
        np.eye(count)[0],
        np.eye(count)[int(torch.argmin(energy_stack))],
    ]
    candidates = []
    evaluation_count = 0

    def counted_objective(
        coefficients: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        nonlocal evaluation_count
        evaluation_count += 1
        return objective_and_gradient(coefficients)

    failures = 0
    optimizer_started = time.perf_counter()
    for initial in starts:
        result = minimize(
            counted_objective,
            initial,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=(constraint,),
            options={"maxiter": 80, "ftol": 1.0e-12},
        )
        failures += int(not result.success)
        if not result.success or not np.isfinite(result.x).all():
            continue
        values = np.clip(result.x, 0.0, 1.0)
        if not np.isfinite(values.sum()) or values.sum() <= 0.0:
            failures += 1
            continue
        values /= values.sum()
        objective_value = counted_objective(values)[0]
        if np.isfinite(objective_value):
            candidates.append((objective_value, values))
        else:
            failures += 1
    used_vertex = False
    vertex_energy = float(torch.min(energy_stack))
    if not candidates or (
        mode == "exact"
        and min(item[0] for item in candidates) > vertex_energy + 1.0e-10
    ):
        best = np.eye(count)[int(torch.argmin(energy_stack))]
        used_vertex = True
    else:
        _, best = min(candidates, key=lambda item: item[0])
    optimizer_seconds = time.perf_counter() - optimizer_started
    return torch.as_tensor(best, dtype=density_stack.dtype), {
        "optimizer_evaluations": evaluation_count,
        "optimizer_failures": failures,
        "used_best_vertex": used_vertex,
        "optimizer_seconds": optimizer_seconds,
    }
