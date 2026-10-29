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
from torch.func import vmap


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
    parser.add_argument(
        "--mapped-roots-stage",
        choices=(
            "all",
            "roots",
            "attachment",
            "energy",
            "state-response",
            "serial-state-response",
        ),
        default="all",
        help="Stop after a completed stage in the mapped-roots diagnostic.",
    )
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
        from dxtb._src.scf.functional.solver import (
            SCFTrajectoryStatus,
            finalize_scf,
        )

        alternate = SimpleNamespace(**vars(case))
        alternate.initial_charges = case.initial_charges + torch.linspace(
            -0.2,
            0.2,
            case.initial_charges.numel(),
            dtype=n0.dtype,
            device=n0.device,
        )
        lane_deltas = torch.tensor(
            [0.0, 1.0e-3], dtype=n0.dtype, device=n0.device
        )
        adapter, flat = _SCFMapTensorTree.prepare(
            inputs, labels.SCP_MODE_CHARGE
        )
        n0_index = adapter.parameter_paths.index("n0")
        lane_records = []
        lane_parameter_values = []
        for lane, delta in enumerate(lane_deltas):
            lane_start = time.perf_counter()
            lane_case = case if lane == 0 else alternate
            lane_n0 = n0 + delta * parameter_direction
            lane_inputs = replace(inputs, n0=lane_n0)
            lane_adapter, lane_values = _SCFMapTensorTree.prepare(
                lane_inputs, labels.SCP_MODE_CHARGE
            )
            if lane_adapter.parameter_paths != adapter.parameter_paths:
                raise RuntimeError("Mapped lane SCF input trees differ.")

            with torch.no_grad():
                lane_root = _run_root(
                    lane_case,
                    lane_inputs,
                    tolerance=args.tolerance,
                    maxiter=args.maxiter,
                    eager=True,
                )
            raw_residual = (
                lane_adapter.evaluate(lane_root.root, *lane_values).mapped_x
                - lane_root.root
            )
            if not lane_root.converged or not bool(
                torch.isfinite(raw_residual).all()
            ):
                raise RuntimeError(
                    "Both prepared physical roots must converge."
                )
            if torch.linalg.vector_norm(raw_residual) >= args.tolerance:
                raise RuntimeError(
                    "Independently evaluated physical-root residual exceeds "
                    "the requested tolerance."
                )
            lane_records.append(lane_root)
            lane_parameter_values.append(lane_values)
            _event(
                "mapped_lane_root_prepared",
                lane=lane,
                parameter_perturbation=float(delta),
                iterations=lane_root.map_count,
                converged=lane_root.converged,
                solver_residual=lane_root.reported_residual.detach().tolist(),
                independent_residual_norm=float(
                    torch.linalg.vector_norm(raw_residual).detach()
                ),
                fermi_converged=bool(
                    lane_root.outcome.trajectory_status.fermi_converged.all()
                ),
                electron_count_ok=bool(
                    lane_root.outcome.trajectory_status.electron_count_ok.all()
                ),
                duration_s=time.perf_counter() - lane_start,
            )

        if args.mapped_roots_stage == "roots":
            return

        first_root, second_root = lane_records
        if not first_root.converged or not second_root.converged:
            raise RuntimeError("Both prepared physical roots must converge.")

        roots = torch.stack((first_root.root, second_root.root))
        stacked_parameters = tuple(
            torch.stack(
                (
                    lane_parameter_values[0][index],
                    lane_parameter_values[1][index],
                )
            )
            for index, value in enumerate(flat)
        )
        parameter_dims = (0,) * len(stacked_parameters)
        mapped_dims = (0, 0, *parameter_dims)
        iterations = torch.tensor(
            [record.map_count for record in lane_records],
            dtype=torch.long,
            device=n0.device,
        )
        converged = torch.tensor(
            [record.converged for record in lane_records],
            dtype=torch.bool,
            device=n0.device,
        )
        residuals = torch.stack(
            [record.reported_residual for record in lane_records]
        )
        statuses = tuple(
            torch.stack(
                [
                    getattr(record.outcome.trajectory_status, name)
                    for record in lane_records
                ]
            )
            for name in (
                "fermi_converged",
                "electron_count_ok",
                "invalid_temperature",
                "too_many_electrons",
                "first_failure",
            )
        )
        mapped_energy_dims = (
            0,
            0,
            parameter_dims,
            0,
            0,
            0,
            (0, 0, 0, 0, 0),
        )
        # Differentiate around each validated lane-specific base point. The
        # finite parameter perturbation is already included in lane 1's n0
        # and was used to prepare lane 1's physical root above.
        deltas = torch.zeros(2, dtype=n0.dtype, device=n0.device)
        direction = torch.zeros_like(n0)
        direction[-1] = 0.2

        def attach_one(root, delta, *values):
            lane_values = list(values)
            lane_values[n0_index] = lane_values[n0_index] + delta * direction
            return implicit_root(root, *lane_values, raw_map=adapter)

        def lane_energy(
            root, delta, values, iteration, is_converged, residual, status
        ):
            lane_values = list(values)
            lane_values[n0_index] = lane_values[n0_index] + delta * direction
            current_inputs, _ = adapter._rebuild(tuple(lane_values))
            attached = implicit_root(root, *lane_values, raw_map=adapter)
            mapped_result = adapter.evaluate(attached, *lane_values)
            finalized = finalize_scf(
                mapped_result,
                current_inputs,
                iteration,
                is_converged,
                residual,
                SCFTrajectoryStatus(*status),
            )
            return finalized.energy.sum() + finalized.fenergy.sum()

        def parameters_for_lane(index):
            return tuple(value[index] for value in stacked_parameters)

        mapped = vmap(attach_one, in_dims=mapped_dims)(
            roots, deltas, *stacked_parameters
        )
        serial_attached = torch.stack(
            tuple(
                attach_one(
                    roots[index], deltas[index], *parameters_for_lane(index)
                )
                for index in range(2)
            )
        )
        torch.testing.assert_close(
            mapped,
            serial_attached,
            atol=2.0e-10,
            rtol=2.0e-10,
            msg="Mapped attached roots differ from serial attachments.",
        )

        _event(
            "mapped_implicit_attachment_values",
            finite=bool(torch.isfinite(mapped).all()),
            max_abs=float(mapped.abs().max().detach()),
            mapped_serial_max_abs=float(
                (mapped - serial_attached).abs().max().detach()
            ),
        )
        if args.mapped_roots_stage == "attachment":
            return

        if args.mapped_roots_stage == "serial-state-response":
            root_module = importlib.import_module(
                "tad_mctc.storch.implicit.root"
            )
            original_implicit_solve = root_module.implicit_solve
            for index in range(2):
                response_records = []

                def record_implicit_solve(*solve_args, **solve_kwargs):
                    result = original_implicit_solve(
                        *solve_args, **solve_kwargs
                    )
                    response_records.append(
                        (
                            result.status.detach().clone(),
                            result.iterations.detach().clone(),
                            result.residual.detach().clone(),
                            solve_args[1].detach().clone(),
                        )
                    )
                    return result

                root_module.implicit_solve = record_implicit_solve
                try:
                    state_derivative = torch.func.jvp(
                        lambda increment: attach_one(
                            roots[index],
                            increment,
                            *parameters_for_lane(index),
                        ),
                        (deltas[index],),
                        (torch.ones_like(deltas[index]),),
                    )[1]
                finally:
                    root_module.implicit_solve = original_implicit_solve
                _event(
                    "serial_root_directional_response",
                    lane=index,
                    finite=bool(torch.isfinite(state_derivative).all()),
                    state_shape=list(state_derivative.shape),
                    solver_records=[
                        {
                            "status": status.cpu().tolist(),
                            "iterations": iterations.cpu().tolist(),
                            "residual_norm": float(
                                torch.linalg.vector_norm(residual).cpu()
                            ),
                            "rhs_norm": float(
                                torch.linalg.vector_norm(rhs).cpu()
                            ),
                            "relative_residual": float(
                                (
                                    torch.linalg.vector_norm(residual)
                                    / torch.linalg.vector_norm(rhs).clamp_min(
                                        torch.finfo(rhs.dtype).tiny
                                    )
                                ).cpu()
                            ),
                        }
                        for status, iterations, residual, rhs in response_records
                    ],
                )
            return

        if args.mapped_roots_stage == "energy":
            _event("mapped_energy_evaluation_started")
            mapped_energies = vmap(lane_energy, in_dims=mapped_energy_dims)(
                roots,
                deltas,
                stacked_parameters,
                iterations,
                converged,
                residuals,
                statuses,
            )
            _event(
                "mapped_scalar_observables",
                energy=mapped_energies.detach().tolist(),
                finite=bool(torch.isfinite(mapped_energies).all()),
            )
            serial_energies = torch.stack(
                tuple(
                    lane_energy(
                        roots[index],
                        deltas[index],
                        parameters_for_lane(index),
                        iterations[index],
                        converged[index],
                        residuals[index],
                        tuple(value[index] for value in statuses),
                    )
                    for index in range(2)
                )
            )
            torch.testing.assert_close(
                mapped_energies,
                serial_energies,
                atol=3.0e-9,
                rtol=3.0e-9,
            )
            _event(
                "mapped_serial_scalar_energy_parity",
                mapped=mapped_energies.detach().tolist(),
                serial=serial_energies.detach().tolist(),
                max_abs=float(
                    (mapped_energies - serial_energies).abs().max().detach()
                ),
            )
            return

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
        root_module = importlib.import_module("tad_mctc.storch.implicit.root")
        original_implicit_solve = root_module.implicit_solve
        from tad_mctc.storch.linsolve.krylov import (
            bicgstab as upstream_bicgstab,
        )

        response_records = []

        def traced_bicgstab(matvec, rhs, *solver_args, **solver_kwargs):
            product_count = 0
            solver_start = time.perf_counter()

            def traced_matvec(value, *operator_args):
                nonlocal product_count
                product_start = time.perf_counter()
                output = matvec(value, *operator_args)
                product_count += 1
                if product_count == 1 or product_count % 4 == 0:
                    _event(
                        "implicit_response_matvec_progress",
                        matvec_count=product_count,
                        product_duration_s=time.perf_counter() - product_start,
                        rhs_norm=float(torch.linalg.vector_norm(rhs).detach()),
                    )
                return output

            result = upstream_bicgstab(
                traced_matvec, rhs, *solver_args, **solver_kwargs
            )
            _event(
                "implicit_response_solver_completed",
                matvec_count=product_count,
                duration_s=time.perf_counter() - solver_start,
                status=result.status.detach().cpu().tolist(),
                iterations=result.iterations.detach().cpu().tolist(),
                residual_norm=float(
                    torch.linalg.vector_norm(result.residual).detach()
                ),
                rhs_norm=float(torch.linalg.vector_norm(rhs).detach()),
            )
            return result

        def record_implicit_solve(*solve_args, **solve_kwargs):
            _event(
                "implicit_response_solver_entered",
                rhs_norm=float(
                    torch.linalg.vector_norm(solve_args[1]).detach()
                ),
            )
            solve_kwargs["method"] = traced_bicgstab
            result = original_implicit_solve(*solve_args, **solve_kwargs)
            response_records.append(
                (
                    result.status.detach().clone(),
                    result.iterations.detach().clone(),
                    result.residual.detach().clone(),
                    solve_args[1].detach().clone(),
                )
            )
            return result

        root_module.implicit_solve = record_implicit_solve
        try:
            lane_responses = []
            for active_lane in range(2):
                _event(
                    "mapped_root_directional_response_started",
                    lane=active_lane,
                )
                lane_direction = torch.zeros_like(deltas)
                lane_direction[active_lane] = 1.0
                response_record_count = len(response_records)

                def mapped_state_response(lane_deltas):
                    return vmap(attach_one, in_dims=mapped_dims)(
                        roots, lane_deltas, *stacked_parameters
                    )

                response = torch.func.jvp(
                    mapped_state_response,
                    (deltas,),
                    (lane_direction,),
                )[1]
                lane_responses.append(response)
                lane_response_records = response_records[response_record_count:]
                torch.testing.assert_close(
                    response[1 - active_lane],
                    torch.zeros_like(response[1 - active_lane]),
                    atol=2.0e-10,
                    rtol=0.0,
                    msg="A lane's n0 response coupled into the other System.",
                )
                _event(
                    "mapped_root_directional_response_completed",
                    lane=active_lane,
                    finite=bool(torch.isfinite(response).all()),
                    response_norm=float(
                        torch.linalg.vector_norm(response).detach()
                    ),
                    cross_lane_norm=float(
                        torch.linalg.vector_norm(
                            response[1 - active_lane]
                        ).detach()
                    ),
                    solver_records=[
                        {
                            "status": record[0].cpu().tolist(),
                            "iterations": record[1].cpu().tolist(),
                            "residual_norm": float(
                                torch.linalg.vector_norm(record[2]).cpu()
                            ),
                            "rhs_norm": float(
                                torch.linalg.vector_norm(record[3]).cpu()
                            ),
                            "relative_residual": float(
                                (
                                    torch.linalg.vector_norm(record[2])
                                    / torch.linalg.vector_norm(
                                        record[3]
                                    ).clamp_min(
                                        torch.finfo(record[3].dtype).tiny
                                    )
                                ).cpu()
                            ),
                        }
                        for record in lane_response_records
                    ],
                )
            mapped_responses = torch.stack(lane_responses, dim=-1)
            if args.mapped_roots_stage == "state-response":
                return

            def mapped_energy_values(increments):
                return vmap(lane_energy, in_dims=mapped_energy_dims)(
                    roots,
                    increments,
                    stacked_parameters,
                    iterations,
                    converged,
                    residuals,
                    statuses,
                )

            energy_jacobian_columns = []
            for active_lane in range(2):
                lane_direction = torch.zeros_like(deltas)
                lane_direction[active_lane] = 1.0
                energy_jacobian_columns.append(
                    torch.func.jvp(
                        mapped_energy_values,
                        (deltas,),
                        (lane_direction,),
                    )[1]
                )
            energy_jacobian = torch.stack(energy_jacobian_columns, dim=-1)
            _event("mapped_energy_parameter_jacobian_completed")
        finally:
            root_module.implicit_solve = original_implicit_solve
        _event("mapped_energy_evaluation_started")
        mapped_energies = vmap(lane_energy, in_dims=mapped_energy_dims)(
            roots,
            deltas,
            stacked_parameters,
            iterations,
            converged,
            residuals,
            statuses,
        )
        _event(
            "mapped_scalar_observables",
            energy=mapped_energies.detach().tolist(),
            finite=bool(torch.isfinite(mapped_energies).all()),
        )
        serial_energies = torch.stack(
            tuple(
                lane_energy(
                    roots[index],
                    deltas[index],
                    parameters_for_lane(index),
                    iterations[index],
                    converged[index],
                    residuals[index],
                    tuple(value[index] for value in statuses),
                )
                for index in range(2)
            )
        )
        torch.testing.assert_close(
            mapped_energies,
            serial_energies,
            atol=3.0e-9,
            rtol=3.0e-9,
        )
        _event(
            "mapped_serial_scalar_energy_parity",
            mapped=mapped_energies.detach().tolist(),
            serial=serial_energies.detach().tolist(),
            max_abs=float(
                (mapped_energies - serial_energies).abs().max().detach()
            ),
        )
        if args.mapped_roots_stage == "energy":
            return
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
        serial_energy_jacobian = []
        serial_response_records = []
        for index in range(2):
            lane_parameters = parameters_for_lane(index)
            lane_solve_records = []

            def record_serial_implicit_solve(*solve_args, **solve_kwargs):
                result = original_implicit_solve(*solve_args, **solve_kwargs)
                lane_solve_records.append(
                    (
                        result.status.detach().clone(),
                        result.iterations.detach().clone(),
                        result.residual.detach().clone(),
                        solve_args[1].detach().clone(),
                    )
                )
                return result

            root_module.implicit_solve = record_serial_implicit_solve
            try:
                serial_energy_jacobian.append(
                    torch.func.jvp(
                        lambda increment: lane_energy(
                            roots[index],
                            increment,
                            lane_parameters,
                            iterations[index],
                            converged[index],
                            residuals[index],
                            tuple(value[index] for value in statuses),
                        ),
                        (deltas[index],),
                        (torch.ones_like(deltas[index]),),
                    )[1]
                )
            finally:
                root_module.implicit_solve = original_implicit_solve
            serial_response_records.append(lane_solve_records)
        serial_energy_jacobian = torch.stack(serial_energy_jacobian)
        serial_derivatives = []
        serial_state_response_records = []
        for index in range(2):
            lane_parameters = parameters_for_lane(index)
            lane_solve_records = []

            def record_state_implicit_solve(*solve_args, **solve_kwargs):
                result = original_implicit_solve(*solve_args, **solve_kwargs)
                lane_solve_records.append(
                    (
                        result.status.detach().clone(),
                        result.iterations.detach().clone(),
                        result.residual.detach().clone(),
                        solve_args[1].detach().clone(),
                    )
                )
                return result

            root_module.implicit_solve = record_state_implicit_solve
            try:
                serial_derivatives.append(
                    torch.func.jvp(
                        lambda increment: attach_one(
                            roots[index], increment, *lane_parameters
                        ),
                        (deltas[index],),
                        (torch.ones_like(deltas[index]),),
                    )[1]
                )
            finally:
                root_module.implicit_solve = original_implicit_solve
            serial_state_response_records.append(lane_solve_records)
        serial_derivatives = torch.stack(serial_derivatives)
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
            lane_parameter_perturbations=lane_deltas.detach().tolist(),
            independently_validated_residual_norms=[
                float(
                    torch.linalg.vector_norm(
                        adapter.evaluate(
                            roots[index], *parameters_for_lane(index)
                        ).mapped_x
                        - roots[index]
                    ).detach()
                )
                for index in range(2)
            ],
            root_status=[
                {
                    "iterations": record.map_count,
                    "converged": record.converged,
                    "solver_residual": record.reported_residual.detach().tolist(),
                    "fermi_converged": bool(
                        record.outcome.trajectory_status.fermi_converged.all()
                    ),
                    "electron_count_ok": bool(
                        record.outcome.trajectory_status.electron_count_ok.all()
                    ),
                }
                for record in lane_records
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
            energy_jacobian_shape=list(energy_jacobian.shape),
            mapped_finite=bool(torch.isfinite(mapped).all()),
            mapped_state_response_finite=bool(
                torch.isfinite(mapped_responses).all()
            ),
            serial_state_response_finite=bool(
                torch.isfinite(serial_derivatives).all()
            ),
            mapped_serial_state_response_max_abs=float(
                (
                    torch.stack(
                        [
                            mapped_responses[index, :, index]
                            for index in range(2)
                        ]
                    )
                    - serial_derivatives
                )
                .abs()
                .max()
                .detach()
            ),
            mapped_state_response_norms=torch.linalg.vector_norm(
                mapped_responses.reshape(2, -1), dim=-1
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
            energy_diagonal_jacobian=energy_jacobian.diagonal()
            .detach()
            .tolist(),
            response_solves=[
                {
                    "status": status.cpu().tolist(),
                    "iterations": iterations.cpu().tolist(),
                    "residual_norm": float(
                        torch.linalg.vector_norm(residual).cpu()
                    ),
                    "rhs_norm": float(torch.linalg.vector_norm(rhs).cpu()),
                    "relative_residual": float(
                        (
                            torch.linalg.vector_norm(residual)
                            / torch.linalg.vector_norm(rhs).clamp_min(
                                torch.finfo(rhs.dtype).tiny
                            )
                        ).cpu()
                    ),
                }
                for status, iterations, residual, rhs in response_records
            ],
            serial_response_solves=[
                [
                    {
                        "status": status.cpu().tolist(),
                        "iterations": iterations.cpu().tolist(),
                        "residual_norm": float(
                            torch.linalg.vector_norm(residual).cpu()
                        ),
                        "rhs_norm": float(torch.linalg.vector_norm(rhs).cpu()),
                        "relative_residual": float(
                            (
                                torch.linalg.vector_norm(residual)
                                / torch.linalg.vector_norm(rhs).clamp_min(
                                    torch.finfo(rhs.dtype).tiny
                                )
                            ).cpu()
                        ),
                    }
                    for status, iterations, residual, rhs in lane_records
                ]
                for lane_records in serial_response_records
            ],
            serial_state_response_solves=[
                [
                    {
                        "status": status.cpu().tolist(),
                        "iterations": iterations.cpu().tolist(),
                        "residual_norm": float(
                            torch.linalg.vector_norm(residual).cpu()
                        ),
                        "rhs_norm": float(torch.linalg.vector_norm(rhs).cpu()),
                        "relative_residual": float(
                            (
                                torch.linalg.vector_norm(residual)
                                / torch.linalg.vector_norm(rhs).clamp_min(
                                    torch.finfo(rhs.dtype).tiny
                                )
                            ).cpu()
                        ),
                    }
                    for status, iterations, residual, rhs in lane_records
                ]
                for lane_records in serial_state_response_records
            ],
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
