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
"""Compare bounded unrolled and prepared-root GFN1 SCF diagnostics.

This private proof-of-concept harness uses the test fixture builder so the
physical setup matches the accepted implicit-root reference tests. It does not
change Calculator routing or provide a production execution path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.func import jacrev, vmap


def _rss_mb() -> tuple[float | None, float]:
    current = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = float(line.split()[1]) / 1024.0
                break
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak /= 1024.0 * 1024.0
    else:
        peak /= 1024.0
    return current, peak


def _event(stage: str, **values: Any) -> None:
    current, peak = _rss_mb()
    record = {
        "stage": stage,
        "elapsed_s": time.perf_counter() - _START,
        "rss_current_mb": current,
        "rss_peak_mb": peak,
        **values,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


_START = time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        choices=("implicit", "unrolled", "mapped-roots"),
        required=True,
    )
    parser.add_argument("--molecule", choices=("H2", "H2O"), default="H2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--higher-order", action="store_true")
    args = parser.parse_args()

    from test.test_scf.test_functional_root_derivatives import _case, _run_root

    from dxtb import GFN1_XTB, labels
    from dxtb._src.calculators.singlepoint import _interaction_data
    from dxtb._src.integral.evaluation import build_integral_matrices
    from dxtb._src.scf.functional import setup_scf_inputs
    from dxtb._src.scf.functional.eager import (
        run_prepared_functional_implicit,
    )
    from dxtb._src.scf.guess import get_guess
    from dxtb._src.scf.iterator import get_refocc

    _event(
        "imports",
        python=__import__("sys").version.split()[0],
        torch=torch.__version__,
        route=args.route,
        molecule=args.molecule,
        temperature_kelvin=args.temperature,
        maxiter=args.maxiter,
    )

    setup_start = time.perf_counter()
    case = _case(
        args.molecule,
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        mixer=labels.MIXER_ANDERSON,
        tolerance=args.tolerance,
        maxiter=args.maxiter,
        temperature=args.temperature,
    )
    model_parameter = case.system.h0_setup.selfenergy
    model_parameter.requires_grad_(True)
    positions = case.positions.detach().clone().requires_grad_(True)
    integrals, reference_occupation, _ = build_integral_matrices(
        case.system.integral_setup,
        case.system.h0_setup,
        positions,
        case.charge,
    )
    n0, occupation, nel = get_refocc(
        reference_occupation,
        case.charge,
        case.spin,
        case.system.ihelp,
    )
    interaction_cache = _interaction_data(
        case.system, positions, case.interactions
    )
    inputs = setup_scf_inputs(
        case.numbers,
        integrals,
        n0,
        occupation,
        nel,
        case.interactions,
        interaction_cache,
        case.system.ihelp,
        case.system.config.scf,
    )
    initial_charges = get_guess(
        case.numbers,
        positions,
        case.charge,
        case.system.ihelp,
        case.system.config.scf.guess,
    )
    case.positions = positions
    case.integrals = integrals
    case.n0 = n0
    case.occupation = occupation
    case.nel = nel
    case.interaction_data = interaction_cache
    case.inputs = inputs
    case.initial_charges = initial_charges
    parameter_direction = torch.zeros_like(n0)
    parameter_direction[-1] = 0.2
    parameter = torch.zeros(
        (), dtype=n0.dtype, device=n0.device, requires_grad=True
    )
    differentiable_n0 = n0 + parameter * parameter_direction
    inputs = replace(inputs, n0=differentiable_n0)
    _event("prepared_inputs", duration_s=time.perf_counter() - setup_start)

    if args.route == "mapped-roots":
        from dxtb._src.scf.functional.implicit import (
            _SCFMapTensorTree,
            implicit_root,
        )
        from dxtb._src.scf.functional.solver import finalize_scf

        alternate = SimpleNamespace(**vars(case))
        alternate.initial_charges = case.initial_charges + torch.linspace(
            -0.2,
            0.2,
            case.initial_charges.numel(),
            dtype=n0.dtype,
            device=n0.device,
        )
        first_root = _run_root(
            case,
            inputs,
            tolerance=args.tolerance,
            maxiter=args.maxiter,
            eager=True,
        )
        second_root = _run_root(
            alternate,
            inputs,
            tolerance=args.tolerance,
            maxiter=args.maxiter,
            eager=True,
        )
        if not first_root.converged or not second_root.converged:
            raise RuntimeError("Both prepared physical roots must converge.")

        adapter, flat = _SCFMapTensorTree.prepare(
            inputs, labels.SCP_MODE_CHARGE
        )
        roots = torch.stack((first_root.root, second_root.root))
        n0_index = adapter.parameter_paths.index("n0")
        stacked_parameters = tuple(
            torch.stack((value, value)) if index == n0_index else value
            for index, value in enumerate(flat)
        )
        parameter_dims = tuple(
            0 if index == n0_index else None
            for index in range(len(stacked_parameters))
        )
        mapped_dims = (0, 0, *parameter_dims)
        deltas = torch.tensor([0.0, 1.0e-3], dtype=n0.dtype, device=n0.device)
        direction = torch.zeros_like(n0)
        direction[-1] = 0.2

        def attach_one(root, delta, *values):
            lane_values = list(values)
            lane_values[n0_index] = lane_values[n0_index] + delta * direction
            return implicit_root(root, *lane_values, raw_map=adapter)

        def lane_energy(root, delta, *values):
            lane_values = list(values)
            lane_values[n0_index] = lane_values[n0_index] + delta * direction
            current_inputs, _ = adapter._rebuild(tuple(lane_values))
            attached = implicit_root(root, *lane_values, raw_map=adapter)
            mapped_result = adapter.evaluate(attached, *lane_values)
            finalized = finalize_scf(
                mapped_result,
                current_inputs,
                torch.zeros((), dtype=torch.long, device=root.device),
                torch.ones((), dtype=torch.bool, device=root.device),
                torch.zeros_like(root),
                None,
            )
            return finalized.energy.sum() + finalized.fenergy.sum()

        mapped = vmap(attach_one, in_dims=mapped_dims)(
            roots, deltas, *stacked_parameters
        )

        def parameters_for_lane(index):
            return tuple(
                value[index] if item == n0_index else value
                for item, value in enumerate(stacked_parameters)
            )

        serial = torch.stack(
            tuple(
                attach_one(
                    roots[index],
                    deltas[index],
                    *parameters_for_lane(index),
                )
                for index in range(2)
            )
        )
        jacobian = jacrev(
            lambda increments: vmap(attach_one, in_dims=mapped_dims)(
                roots, increments, *stacked_parameters
            )
        )(deltas)
        mapped_energies = vmap(lane_energy, in_dims=mapped_dims)(
            roots, deltas, *stacked_parameters
        )
        serial_energies = torch.stack(
            tuple(
                lane_energy(
                    roots[index],
                    deltas[index],
                    *parameters_for_lane(index),
                )
                for index in range(2)
            )
        )
        mapped_model_gradient = torch.autograd.grad(
            mapped_energies.sum(),
            model_parameter,
            retain_graph=True,
        )[0]
        serial_model_gradient = torch.autograd.grad(
            serial_energies.sum(),
            model_parameter,
            retain_graph=True,
        )[0]
        energy_jacobian = jacrev(
            lambda increments: vmap(lane_energy, in_dims=mapped_dims)(
                roots, increments, *stacked_parameters
            )
        )(deltas)
        serial_energy_jacobian = []
        for index in range(2):
            lane_parameters = parameters_for_lane(index)
            serial_energy_jacobian.append(
                jacrev(
                    lambda increment: lane_energy(
                        roots[index], increment, *lane_parameters
                    )
                )(deltas[index])
            )
        serial_energy_jacobian = torch.stack(serial_energy_jacobian)
        serial_derivatives = []
        for index in range(2):
            lane_parameters = parameters_for_lane(index)
            serial_derivatives.append(
                jacrev(
                    lambda increment: attach_one(
                        roots[index], increment, *lane_parameters
                    )
                )(deltas[index])
            )
        serial_derivatives = torch.stack(serial_derivatives)
        off_diagonal = torch.cat(
            (jacobian[0, ..., 1].reshape(-1), jacobian[1, ..., 0].reshape(-1))
        )
        diagonal = torch.stack((jacobian[0, ..., 0], jacobian[1, ..., 1]))
        _event(
            "mapped_physical_roots",
            convergence_iterations=[
                first_root.map_count,
                second_root.map_count,
            ],
            convergence_residuals=[
                float(torch.linalg.vector_norm(first_root.residual).detach()),
                float(torch.linalg.vector_norm(second_root.residual).detach()),
            ],
            mapped_serial_max_abs=float((mapped - serial).abs().max().detach()),
            mapped_serial_energy_value_max_abs=float(
                (mapped_energies - serial_energies).abs().max().detach()
            ),
            shared_model_parameter_gradient_max_abs=float(
                (mapped_model_gradient - serial_model_gradient)
                .abs()
                .max()
                .detach()
            ),
            jacobian_shape=list(jacobian.shape),
            cross_lane_max_abs=float(off_diagonal.abs().max().detach()),
            mapped_finite=bool(torch.isfinite(mapped).all()),
            diagonal_response_finite=bool(torch.isfinite(diagonal).all()),
            serial_response_finite=bool(
                torch.isfinite(serial_derivatives).all()
            ),
            mapped_serial_response_max_abs=float(
                (diagonal - serial_derivatives).abs().max().detach()
            ),
            diagonal_response_norms=torch.linalg.vector_norm(
                diagonal.reshape(2, -1), dim=-1
            )
            .detach()
            .tolist(),
            mapped_energy_jacobian=energy_jacobian.detach().tolist(),
            serial_energy_jacobian=serial_energy_jacobian.detach().tolist(),
            mapped_serial_energy_max_abs=float(
                (energy_jacobian.diagonal() - serial_energy_jacobian)
                .abs()
                .max()
                .detach()
            ),
            energy_cross_lane_max_abs=float(
                torch.stack((energy_jacobian[0, 1], energy_jacobian[1, 0]))
                .abs()
                .max()
                .detach()
            ),
        )
        return

    forward_start = time.perf_counter()
    if args.route == "implicit":
        root_module = importlib.import_module("tad_mctc.storch.implicit.root")
        original_implicit_solve = root_module.implicit_solve
        response_records = []

        def record_implicit_solve(*solve_args, **solve_kwargs):
            result = original_implicit_solve(*solve_args, **solve_kwargs)
            response_records.append(
                (
                    result.status.detach().clone(),
                    result.iterations.detach().clone(),
                    result.residual.detach().clone(),
                )
            )
            return result

        root_module.implicit_solve = record_implicit_solve
        config = replace(
            case.system.config.scf,
            scf_mode=labels.SCF_MODE_IMPLICIT,
            maxiter=args.maxiter,
            x_atol=args.tolerance,
            x_atol_max=args.tolerance,
        )
        result = run_prepared_functional_implicit(
            case.numbers,
            initial_charges,
            differentiable_n0,
            occupation,
            nel,
            case.interactions,
            interaction_cache,
            case.system.ihelp,
            config,
            integrals,
        )
        observable = result.energy.sum() + result.fenergy.sum()
        electron_count_error = (result.occupation.sum(-1) - nel).abs().amax()
        fermi_status = result.map_result.fermi_status
        trajectory_status = result.trajectory_status
        _event(
            "root_preparation_and_implicit_forward",
            duration_s=time.perf_counter() - forward_start,
            energy=float(result.energy.sum().detach()),
            free_energy=float(result.fenergy.sum().detach()),
            iterations=int(result.iterations.detach().max()),
            converged=bool(result.converged.detach().all()),
            residual_norm=float(
                torch.linalg.vector_norm(result.residual).detach()
            ),
            map_evaluations=int(result.iterations.detach().max()),
            electron_count_error=float(electron_count_error.detach()),
            fermi_converged=(
                None
                if fermi_status is None
                else bool(fermi_status.converged.detach().all())
            ),
            fermi_residual=(
                None
                if fermi_status is None
                else float(fermi_status.residual.detach().abs().max())
            ),
            trajectory_fermi_converged=(
                None
                if trajectory_status is None
                else bool(trajectory_status.fermi_converged.detach().all())
            ),
            trajectory_electron_count_ok=(
                None
                if trajectory_status is None
                else bool(trajectory_status.electron_count_ok.detach().all())
            ),
            observable_requires_grad=observable.requires_grad,
        )
    else:
        from test.test_scf.test_functional_root_derivatives import _run_root

        result = _run_root(
            case,
            inputs,
            tolerance=args.tolerance,
            maxiter=args.maxiter,
            eager=False,
        )
        observable = result.finalized_energy.sum()
        _event(
            "unrolled_root_forward",
            duration_s=time.perf_counter() - forward_start,
            energy=float(result.electronic_energy.detach()),
            free_energy=float(
                (result.finalized_energy - result.electronic_energy).detach()
            ),
            iterations=result.map_count,
            converged=result.converged,
            residual_norm=float(
                torch.linalg.vector_norm(result.residual).detach()
            ),
            map_evaluations=result.solver_map_evaluations,
            observable_requires_grad=observable.requires_grad,
        )

    backward_start = time.perf_counter()
    (
        position_gradient,
        parameter_gradient,
        model_parameter_gradient,
    ) = torch.autograd.grad(
        observable,
        (positions, parameter, model_parameter),
        create_graph=args.higher_order,
    )
    higher = {}
    if args.higher_order:
        second = torch.autograd.grad(
            parameter_gradient,
            parameter,
            create_graph=True,
            retain_graph=True,
        )[0]
        third = torch.autograd.grad(second, parameter)[0]
        higher = {
            "n0_second_derivative": float(second.detach()),
            "n0_third_derivative": float(third.detach()),
            "higher_order_finite": bool(
                torch.isfinite(second).all() & torch.isfinite(third).all()
            ),
        }
    _event(
        "position_and_n0_parameter_backward",
        duration_s=time.perf_counter() - backward_start,
        position_gradient_norm=float(
            torch.linalg.vector_norm(position_gradient).detach()
        ),
        position_gradient=position_gradient.detach().tolist(),
        parameter_gradient=float(parameter_gradient.detach()),
        model_parameter_gradient=model_parameter_gradient.detach().tolist(),
        all_finite=bool(
            torch.isfinite(position_gradient).all()
            & torch.isfinite(parameter_gradient).all()
            & torch.isfinite(model_parameter_gradient).all()
        ),
        **higher,
    )
    if args.route == "implicit":
        root_module.implicit_solve = original_implicit_solve
        _event(
            "implicit_response_solve_diagnostics",
            solves=[
                {
                    "status": status.cpu().tolist(),
                    "iterations": iterations.cpu().tolist(),
                    "residual_norm": float(
                        torch.linalg.vector_norm(residual).cpu()
                    ),
                }
                for status, iterations, residual in response_records
            ],
        )


if __name__ == "__main__":
    main()
