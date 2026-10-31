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
"""Audit reference EDIIS objectives on actual functional SCF histories."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from test.test_scf.test_functional_map import _prepare_case

import numpy as np
import torch
from scipy.optimize import minimize

from benchmarks.prototypes.scf_energy_diis import exact_energy_value_gradient
from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.scf.functional import unpack_charges
from dxtb._src.scf.functional.cdiis import CDIISMethod
from dxtb._src.scf.functional.core import FunctionalInteractionSet
from dxtb._src.scf.functional.data import SCFInputs
from dxtb._src.scf.functional.energy import electronic_energy_from_density
from dxtb._src.scf.functional.map import (
    density_to_charges_and_core_energy,
    fock_initial_guess,
)
from dxtb._src.scf.functional.problem import SCFProblem, SCFRepresentation
from dxtb._src.scf.functional.solver import SCFSolverSettings, _solve_map
from dxtb._src.scf.functional.strategy import SCFStage, SCFStrategy


def _classic_objective(
    coefficients: np.ndarray,
    energies: torch.Tensor,
    pair: torch.Tensor,
    correction: float = 1.0,
) -> float:
    values = torch.as_tensor(coefficients, dtype=energies.dtype)
    return float(values @ energies - correction * (values @ pair @ values))


def _minimize_classic(
    energies: torch.Tensor, pair: torch.Tensor, correction: float = 1.0
) -> tuple[np.ndarray, int]:
    count = energies.numel()
    starts = (np.full(count, 1.0 / count), np.eye(count)[0])
    results = []
    evaluations = 0

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        values = torch.as_tensor(coefficients, dtype=energies.dtype)
        value = values @ energies - correction * (values @ pair @ values)
        gradient = energies - 2.0 * correction * pair @ values
        return float(value), gradient.numpy()

    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            jac=True,
            bounds=[(0.0, 1.0)] * count,
            constraints=({"type": "eq", "fun": lambda c: c.sum() - 1.0},),
            options={"maxiter": 80, "ftol": 1.0e-12},
        )
        coefficients = np.clip(result.x, 0.0, 1.0)
        coefficients /= coefficients.sum()
        results.append(
            (_classic_objective(coefficients, energies, pair), coefficients)
        )
    return min(results, key=lambda item: item[0])[1], evaluations


def _density_fock(problem: SCFProblem, density: torch.Tensor) -> torch.Tensor:
    charges, _ = density_to_charges_and_core_energy(density, problem.inputs)
    return problem.rebuild_hamiltonian(charges)


def _audit_history(
    problem: SCFProblem,
    densities: tuple[torch.Tensor, ...],
    seed: int,
) -> dict[str, object]:
    energies = torch.stack(
        [
            electronic_energy_from_density(p, problem.inputs).sum()
            for p in densities
        ]
    )
    focks = torch.stack([_density_fock(problem, p) for p in densities])
    density_stack = torch.stack(densities)
    pair = torch.einsum(
        "ijab,ijab->ij",
        density_stack[:, None] - density_stack[None, :],
        focks[:, None] - focks[None, :],
    )
    rng = np.random.default_rng(seed)
    random_coefficients = rng.dirichlet(np.ones(len(densities)), size=500)
    exact_values = []
    classic_values = []
    quadratic_values = []
    electron_count_error = 0.0
    padding_density_max = 0.0
    padding_fock_max = 0.0
    pair_mask = (
        problem.inputs.orbital_mask[..., :, None]
        & problem.inputs.orbital_mask[..., None, :]
    )
    target_electrons = problem.inputs.nel.sum()
    overlap = problem.inputs.integrals.overlap
    for coefficients in random_coefficients:
        coeff = torch.as_tensor(coefficients, dtype=energies.dtype)
        density = torch.einsum("i,ijk->jk", coeff, density_stack)
        exact_values.append(
            float(electronic_energy_from_density(density, problem.inputs).sum())
        )
        classic_values.append(_classic_objective(coefficients, energies, pair))
        quadratic_values.append(
            _classic_objective(coefficients, energies, pair, correction=0.25)
        )
        electron_count_error = max(
            electron_count_error,
            abs(float(torch.sum(density * overlap.mT) - target_electrons)),
        )
        padding_density_max = max(
            padding_density_max,
            float(torch.where(pair_mask, 0.0, density).abs().max()),
        )
        trial_fock = _density_fock(problem, density)
        padding_fock_max = max(
            padding_fock_max,
            float(torch.where(pair_mask, 0.0, trial_fock).abs().max()),
        )

    exact_min_candidates = []
    exact_evaluations = 0
    count = len(densities)
    starts = (
        np.full(count, 1.0 / count),
        np.eye(count)[0],
        np.eye(count)[int(torch.argmin(energies))],
    )

    def exact_objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal exact_evaluations
        exact_evaluations += 1
        coeff = torch.as_tensor(coefficients, dtype=energies.dtype)
        value, gradient = exact_energy_value_gradient(coeff, densities, problem)
        return float(value), gradient.numpy()

    for start in starts:
        result = minimize(
            exact_objective,
            start,
            method="SLSQP",
            jac=True,
            bounds=[(0.0, 1.0)] * count,
            constraints=({"type": "eq", "fun": lambda c: c.sum() - 1.0},),
            options={"maxiter": 80, "ftol": 1.0e-12},
        )
        coeff = np.clip(result.x, 0.0, 1.0)
        coeff /= coeff.sum()
        exact_min_candidates.append((exact_objective(coeff)[0], coeff))
    exact_min, exact_coeff = min(exact_min_candidates, key=lambda item: item[0])
    classic_coeff, classic_evaluations = _minimize_classic(energies, pair)
    quadratic_coeff, quadratic_evaluations = _minimize_classic(
        energies, pair, correction=0.25
    )
    exact_at_classic = exact_objective(classic_coeff)[0]
    exact_at_quadratic = exact_objective(quadratic_coeff)[0]
    classic_at_exact = _classic_objective(exact_coeff, energies, pair)
    scale = max(float(energies.max() - energies.min()), 1.0e-14)

    two = densities[:2]
    coefficient = 0.37
    exact_pair_fock = _density_fock(
        problem, coefficient * two[0] + (1.0 - coefficient) * two[1]
    )
    linear_pair_fock = coefficient * focks[0] + (1.0 - coefficient) * focks[1]
    nonlinear_error = float(
        torch.linalg.vector_norm(exact_pair_fock - linear_pair_fock)
    )
    reduced_entries = tuple(
        entry
        for entry in problem.inputs.interactions.entries
        if entry.kind not in ("es3", "d4sc")
    )
    reduced_inputs = replace(
        problem.inputs,
        interactions=FunctionalInteractionSet(reduced_entries),
    )
    reduced_problem = replace(problem, inputs=reduced_inputs)
    reduced_focks = torch.stack(
        [_density_fock(reduced_problem, density) for density in densities]
    )
    reduced_exact_fock = _density_fock(
        reduced_problem,
        coefficient * two[0] + (1.0 - coefficient) * two[1],
    )
    reduced_linear_fock = (
        coefficient * reduced_focks[0] + (1.0 - coefficient) * reduced_focks[1]
    )

    if count > 1:
        deltas = density_stack[1:] - density_stack[0]
        zero = density_stack.new_zeros(count - 1)

        def exact_local(values: torch.Tensor) -> torch.Tensor:
            trial = density_stack[0] + torch.einsum("i,ijk->jk", values, deltas)
            return electronic_energy_from_density(trial, problem.inputs).sum()

        raw = torch.einsum("iab,jab->ij", density_stack, focks)
        d_fn = raw[:, 0]
        dn_f = raw[0]
        dn_fn = raw[0, 0]
        adiis_linear = 2.0 * (d_fn - dn_fn)
        adiis_curvature = raw - d_fn[:, None] - dn_f[None, :] + dn_fn

        def adiis_local(values: torch.Tensor) -> torch.Tensor:
            coeff = torch.cat(((1.0 - values.sum()).reshape(1), values))
            return coeff @ adiis_linear + coeff @ adiis_curvature @ coeff

        exact_gradient = torch.func.grad(exact_local)(zero)
        exact_hessian = torch.func.hessian(exact_local)(zero)
        adiis_gradient = torch.func.grad(adiis_local)(zero)
        adiis_hessian = torch.func.hessian(adiis_local)(zero)
        adiis_gradient_error = float(
            torch.linalg.vector_norm(adiis_gradient - exact_gradient)
        )
        adiis_hessian_error = float(
            torch.linalg.matrix_norm(adiis_hessian - exact_hessian)
        )
    else:
        adiis_gradient_error = 0.0
        adiis_hessian_error = 0.0

    grid_steps = {2: 0.001, 3: 0.02, 4: 0.05}[count]
    grid_coefficients = []
    divisions = round(1.0 / grid_steps)
    if count == 2:
        grid_coefficients = [
            (index / divisions, 1.0 - index / divisions)
            for index in range(divisions + 1)
        ]
    elif count == 3:
        for first in range(divisions + 1):
            for second in range(divisions + 1 - first):
                grid_coefficients.append(
                    (
                        first / divisions,
                        second / divisions,
                        1.0 - (first + second) / divisions,
                    )
                )
    else:
        for first in range(divisions + 1):
            for second in range(divisions + 1 - first):
                for third in range(divisions + 1 - first - second):
                    grid_coefficients.append(
                        (
                            first / divisions,
                            second / divisions,
                            third / divisions,
                            1.0 - (first + second + third) / divisions,
                        )
                    )
    grid_minimum = float("inf")
    for coefficients in grid_coefficients:
        coeff = torch.tensor(coefficients, dtype=energies.dtype)
        trial = torch.einsum("i,ijk->jk", coeff, density_stack)
        grid_minimum = min(
            grid_minimum,
            float(electronic_energy_from_density(trial, problem.inputs).sum()),
        )
    return {
        "history_size": count,
        "energies": [float(value) for value in energies],
        "random_simplex_samples": len(random_coefficients),
        "classic_objective_abs_error_max": float(
            np.max(
                np.abs(np.asarray(classic_values) - np.asarray(exact_values))
            )
        ),
        "classic_objective_abs_error_mean": float(
            np.mean(
                np.abs(np.asarray(classic_values) - np.asarray(exact_values))
            )
        ),
        "classic_objective_relative_to_history_range_max": float(
            np.max(
                np.abs(np.asarray(classic_values) - np.asarray(exact_values))
            )
            / scale
        ),
        "quadratic_objective_abs_error_max": float(
            np.max(
                np.abs(np.asarray(quadratic_values) - np.asarray(exact_values))
            )
        ),
        "quadratic_objective_relative_to_history_range_max": float(
            np.max(
                np.abs(np.asarray(quadratic_values) - np.asarray(exact_values))
            )
            / scale
        ),
        "exact_minimum_energy": exact_min,
        "exact_minimum_coefficients": exact_coeff.tolist(),
        "classic_minimum_coefficients": classic_coeff.tolist(),
        "classic_minimum_exact_energy": exact_at_classic,
        "classic_energy_gap_to_exact_minimum": exact_at_classic - exact_min,
        "quadratic_minimum_coefficients": quadratic_coeff.tolist(),
        "quadratic_minimum_exact_energy": exact_at_quadratic,
        "quadratic_energy_gap_to_exact_minimum": exact_at_quadratic - exact_min,
        "classic_model_value_at_exact_minimum": classic_at_exact,
        "coefficient_l2_displacement": float(
            np.linalg.norm(classic_coeff - exact_coeff)
        ),
        "exact_optimizer_evaluations": exact_evaluations,
        "classic_optimizer_evaluations": classic_evaluations,
        "quadratic_optimizer_evaluations": quadratic_evaluations,
        "fock_interpolation_error_at_0.37": nonlinear_error,
        "max_electron_count_error": electron_count_error,
        "max_padding_density": padding_density_max,
        "max_padding_fock": padding_fock_max,
        "fock_interpolation_error_without_es3_d4sc": float(
            torch.linalg.vector_norm(reduced_exact_fock - reduced_linear_fock)
        ),
        "adiis_local_gradient_error": adiis_gradient_error,
        "adiis_local_hessian_error": adiis_hessian_error,
        "simplex_grid_points": len(grid_coefficients),
        "simplex_grid_minimum_energy": grid_minimum,
        "optimizer_minus_grid_energy": exact_min - grid_minimum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="H2O")
    parser.add_argument("--model", choices=("gfn1", "gfn2"), default="gfn2")
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument("--history", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    method = GFN1_XTB if args.model == "gfn1" else GFN2_XTB
    case = _prepare_case(
        method,
        sample_name=args.case,
        charge=args.charge,
        spin=args.spin,
        dtype=torch.float64,
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    guess = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    trajectory = _solve_map(
        guess,
        SCFSolverSettings(args.history - 1, False, 0.5, 0.0, 0.0),
        problem.map,
        problem=problem,
        strategy=SCFStrategy((SCFStage(CDIISMethod()),)),
        record_trace=True,
    )
    history = tuple(trajectory.trace.densities[: args.history])
    results = [
        _audit_history(problem, history[:size], seed=810 + size)
        for size in (2, 3, 4)
        if size <= len(history)
    ]
    output = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_tree_dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "src/dxtb",
                    "benchmarks/prototypes",
                    "benchmarks/scf_s10_objective_audit.py",
                    "test/test_scf/test_functional_ediis.py",
                ],
                text=True,
            ).strip()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dtype": "float64",
        "threads": torch.get_num_threads(),
        "case": args.case,
        "model": args.model,
        "charge": args.charge,
        "spin": args.spin,
        "source_history": "first CDIIS macro-observations from functional controller",
        "results": results,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
