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
"""Bounded tmpda primal-root and implicit-response feasibility diagnostic.

Run one conformer in a fresh process, retaining JSON Lines progress if a
resource limit or external timeout stops the calculation::

    python -u benchmarks/e6c1_w2c2_implicit.py --stage root

The frozen W2 configuration is used by default. No derivative is attached
unless the independently evaluated physical root passes its convergence gate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from molecules import mols  # noqa: E402

import dxtb as dxtb_package  # noqa: E402
from dxtb import GFN1_XTB, ParamModule, labels  # noqa: E402
from dxtb._src.calculators.model import Model  # noqa: E402
from dxtb._src.calculators.result import Result  # noqa: E402
from dxtb._src.calculators.singlepoint import (  # noqa: E402
    _evaluate_classical_descriptor,
    _interaction_data,
)
from dxtb._src.integral.evaluation import build_integral_matrices  # noqa: E402
from dxtb._src.scf.functional import (  # noqa: E402
    SCFCharges,
    pack_charges,
    setup_scf_inputs,
)
from dxtb._src.scf.functional.anderson import _solve_map_anderson  # noqa: E402
from dxtb._src.scf.functional.eager import _solver_settings  # noqa: E402
from dxtb._src.scf.functional.implicit import (  # noqa: E402
    _prepare_scf_root,
    _SCFMapTensorTree,
    implicit_root,
)
from dxtb._src.scf.functional.map import scf_map  # noqa: E402
from dxtb._src.scf.functional.solver import (  # noqa: E402
    SCFTrajectoryStatus,
    fermi_failure_code,
    fermi_status_ok,
    finalize_scf,
)
from dxtb._src.scf.guess import get_guess  # noqa: E402
from dxtb._src.scf.iterator import get_refocc  # noqa: E402
from dxtb._src.typing.exceptions import SCFConvergenceError  # noqa: E402
from dxtb.config import Config  # noqa: E402

DTYPE = torch.float64
STARTED = time.perf_counter()
_MEMORY_LIMIT_MB = 16_000.0
_MEMORY_STOP = threading.Event()


@dataclass(frozen=True)
class PhysicalCase:
    """Frozen W2 geometry and production System setup."""

    model: Model
    system: Any
    numbers: torch.Tensor
    positions: torch.Tensor
    charge: torch.Tensor
    spin: torch.Tensor | None


@dataclass(frozen=True)
class PreparedInputs:
    """Call-local integral, interaction, and SCF inputs for one geometry."""

    case: PhysicalCase
    integrals: Any
    refocc: torch.Tensor
    overlap_norm: torch.Tensor
    n0: torch.Tensor
    occupation: torch.Tensor
    nel: torch.Tensor
    interaction_data: Any
    inputs: Any
    initial_state: torch.Tensor


def current_rss_mb() -> float | None:
    """Return current Linux resident memory when procfs is available."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def peak_rss_mb() -> float:
    """Return the process lifetime RSS high-water mark in MiB."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def emit(stage: str, **values: Any) -> None:
    """Write one flushed, machine-readable stage event."""
    print(
        json.dumps(
            {
                "sequence": emit.sequence,
                "stage": stage,
                "elapsed_s": time.perf_counter() - STARTED,
                "rss_current_mb": current_rss_mb(),
                "rss_peak_mb": peak_rss_mb(),
                **values,
            },
            default=str,
            sort_keys=True,
        ),
        flush=True,
    )
    emit.sequence += 1


emit.sequence = 0


def monitor_memory() -> None:
    """Stop the diagnostic if resident memory exceeds its configured cap."""
    while not _MEMORY_STOP.wait(0.5):
        rss = current_rss_mb()
        if rss is not None and rss > _MEMORY_LIMIT_MB:
            emit(
                "memory_limit_exceeded",
                limit_mb=_MEMORY_LIMIT_MB,
                observed_rss_mb=rss,
            )
            os._exit(86)


def construct_workload(
    conformer_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one conformer from the frozen tmpda geometry generator."""
    source = mols["tmpda"]
    base_positions = source["positions"].double()
    generator = torch.Generator().manual_seed(0)
    displacement = (
        0.1
        * torch.randn((8, *base_positions.shape), generator=generator).double()
    )
    displacement[0] = 0.0
    if not 0 <= conformer_index < displacement.shape[0]:
        raise ValueError("The requested conformer is outside the W2 workload.")
    positions = (base_positions.unsqueeze(0) + displacement)[conformer_index]
    return source["numbers"].to(dtype=torch.int64), positions


def setup_model_system(
    numbers: torch.Tensor,
    *,
    maxiter: int,
    x_atol: float,
    x_atol_max: float,
) -> tuple[Model, Any]:
    """Build the GFN1 Model/System with the frozen W2 options."""
    parameters = ParamModule(GFN1_XTB, dtype=DTYPE)
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        mixer=labels.MIXER_ANDERSON,
        scf_mode=labels.SCF_MODE_FULL,
        maxiter=maxiter,
        f_atol=1.0e-8,
        x_atol=x_atol,
        x_atol_max=x_atol_max,
        fermi_etemp=300.0,
    )
    model = Model(par=parameters, config=config, auto_int_level=True)

    # Enable the real Model-owned parameter leaves before System setup so
    # setup-derived H0 and interaction tensors keep their parameter graph.
    for parameter in model.par.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)

    system = model.setup(numbers, batch_mode=0)
    return model, system


def prepare_call_inputs(case: PhysicalCase) -> PreparedInputs:
    """Prepare integrals, default interactions, occupations, and SCF state."""
    integrals, refocc, overlap_norm = build_integral_matrices(
        case.system.integral_setup,
        case.system.h0_setup,
        case.positions,
        case.charge,
    )
    n0, occupation, nel = get_refocc(
        refocc, case.charge, case.spin, case.system.ihelp
    )
    interaction_data = _interaction_data(case.system, case.positions)
    inputs = setup_scf_inputs(
        case.numbers,
        integrals,
        n0,
        occupation,
        nel,
        case.system.interactions,
        interaction_data,
        case.system.ihelp,
        case.system.config.scf,
    )
    guess = get_guess(
        case.numbers,
        case.positions,
        case.charge,
        case.system.ihelp,
        case.system.config.scf.guess,
    )
    charges = SCFCharges(
        mono=guess,
        dipole=(
            None
            if inputs.charge_layout.dipole_shape is None
            else guess.new_zeros(inputs.charge_layout.dipole_shape)
        ),
        quad=(
            None
            if inputs.charge_layout.quadrupole_shape is None
            else guess.new_zeros(inputs.charge_layout.quadrupole_shape)
        ),
    )
    return PreparedInputs(
        case=case,
        integrals=integrals,
        refocc=refocc,
        overlap_norm=overlap_norm,
        n0=n0,
        occupation=occupation,
        nel=nel,
        interaction_data=interaction_data,
        inputs=inputs,
        initial_state=pack_charges(charges, inputs.charge_layout),
    )


def prepare_primal_root(prepared: PreparedInputs) -> tuple[Any, int, float]:
    """Run the accepted eager Anderson controller without a trajectory graph."""
    config = prepared.case.system.config.scf
    settings = _solver_settings(config, DTYPE)
    map_evaluations = 0

    def observed_map(state: torch.Tensor):
        nonlocal map_evaluations
        map_evaluations += 1
        if map_evaluations == 1 or map_evaluations % 10 == 0:
            emit("primal_map_progress", map_evaluations=map_evaluations)
        return scf_map(state, prepared.inputs)

    map_start = time.perf_counter()
    with torch.no_grad():
        outcome = _solve_map_anderson(
            prepared.initial_state,
            settings,
            observed_map,
            eager=True,
        )
    return outcome, map_evaluations, time.perf_counter() - map_start


def validate_physical_root(
    prepared: PreparedInputs, outcome: Any
) -> tuple[Any, tuple[torch.Tensor, ...], torch.Tensor, Any]:
    """Check solver status and raw residual at the exact attachment inputs."""
    adapter, parameters = _SCFMapTensorTree.prepare(
        prepared.inputs, labels.SCP_MODE_CHARGE
    )
    with torch.no_grad():
        independent_map = adapter.evaluate(outcome.root_x, *parameters)
        raw_residual = independent_map.mapped_x - outcome.root_x
    residual_l2 = torch.linalg.vector_norm(raw_residual)
    residual_max = raw_residual.abs().amax()
    fermi_status = independent_map.fermi_status
    solver_valid = bool(outcome.converged)
    residual_valid = bool(
        torch.isfinite(raw_residual).all()
        & (residual_l2 <= 1.0e-8)
        & (residual_max <= 1.0e-8)
    )
    fermi_valid = bool(fermi_status_ok(fermi_status, like=outcome.converged))
    emit(
        "independent_root_check",
        solver_iterations=int(outcome.iterations.detach()),
        solver_converged=solver_valid,
        residual_valid=residual_valid,
        fermi_valid=fermi_valid,
        solver_residual=outcome.residual.detach().cpu().tolist(),
        independent_residual_l2=float(residual_l2.detach()),
        independent_residual_max=float(residual_max.detach()),
        electron_count_error=(
            independent_map.occupation.sum(dim=-1) - prepared.inputs.nel
        )
        .detach()
        .cpu()
        .tolist(),
        fermi_converged=(
            None
            if fermi_status is None
            else fermi_status.converged.detach().cpu().tolist()
        ),
        fermi_residual=(
            None
            if fermi_status is None
            else fermi_status.residual.detach().cpu().tolist()
        ),
        electron_count_ok=(
            None
            if fermi_status is None
            else fermi_status.electron_count_ok.detach().cpu().tolist()
        ),
        trajectory_failure=(
            None
            if outcome.trajectory_status is None
            else outcome.trajectory_status.first_failure.detach().cpu().tolist()
        ),
        final_failure=fermi_failure_code(fermi_status, like=outcome.converged)
        .detach()
        .cpu()
        .tolist(),
        density_finite=bool(torch.isfinite(independent_map.density).all()),
    )
    if not (solver_valid and residual_valid and fermi_valid):
        raise SCFConvergenceError(
            "The tmpda primal root did not pass independent physical-root "
            "validation; implicit differentiation is not attached."
        )
    validated = _prepare_scf_root(
        outcome,
        maxiter=prepared.case.system.config.scf.maxiter,
        x_atol=prepared.case.system.config.scf.x_atol,
        x_atol_max=prepared.case.system.config.scf.x_atol_max,
        final_valid=fermi_status_ok(
            independent_map.fermi_status, like=outcome.converged
        ),
        raw_map=adapter,
        parameters=parameters,
    )
    return adapter, parameters, raw_residual, validated


def attach_implicit_root(
    validated: Any,
) -> torch.Tensor:
    """Attach IFT derivatives using the same validated tensor inputs."""
    root = validated.attach()
    emit("implicit_root_attached", root_shape=list(root.shape))
    return root


def finalize_physical_result(
    prepared: PreparedInputs,
    outcome: Any,
    map_result: Any,
) -> tuple[Result, Any]:
    """Finalize SCF and default classical terms with production functions."""
    solve_result = finalize_scf(
        map_result,
        prepared.inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
    )
    core = prepared.case.system._active_exact_core()
    if core is None:
        raise RuntimeError("The W2 System has no active exact core.")
    classical = tuple(
        (
            term.label,
            _evaluate_classical_descriptor(
                term,
                prepared.case.positions,
                prepared.case.charge,
            ),
        )
        for term in core.classical_terms
    )
    if len(classical) != len(prepared.case.system.classicals.components):
        raise RuntimeError(
            "The W2 physical classical composition is not fully represented "
            "by the captured exact descriptors."
        )
    classical_energy = (
        torch.stack(tuple(value for _, value in classical)).sum(0)
        if classical
        else torch.zeros_like(solve_result.energy)
    )
    energy = classical_energy + solve_result.energy + solve_result.fenergy
    physical_result = Result.snapshot(
        energy=energy,
        scf=solve_result.energy,
        classical=classical,
        fenergy=solve_result.fenergy,
        charges=solve_result.final_charges,
        density=solve_result.density,
        coefficients=solve_result.coefficients,
        emo=solve_result.emo,
        occupation=solve_result.occupation,
        potential=solve_result.final_potential,
        hamiltonian=solve_result.hamiltonian,
        overlap=prepared.integrals.overlap,
        hcore=prepared.integrals.hcore,
        dipole_integrals=prepared.integrals.dipole,
        quadrupole_integrals=prepared.integrals.quadrupole,
        overlap_norm=prepared.overlap_norm,
        iterations=solve_result.iterations,
        converged=solve_result.converged,
    )
    return physical_result, solve_result


def evaluate_energy(result: Result) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen W2 energy and the separately requested sum."""
    energy_only = result.energy.sum()
    energy_plus_fenergy = energy_only + result.fenergy.sum()
    return energy_only, energy_plus_fenergy


def _instrument_linear_response():
    """Return scoped upstream response instrumentation handles."""
    import tad_mctc.storch.implicit.root as implicit_root_module

    original = implicit_root_module.implicit_solve
    records: list[dict[str, Any]] = []

    def instrument_operator(operator, name: str):
        if operator is None:
            return None

        def measured(*args, **kwargs):
            started = time.perf_counter()
            output = operator(*args, **kwargs)
            record = {
                "operator": name,
                "seconds": time.perf_counter() - started,
            }
            records.append(record)
            count = sum(item["operator"] == name for item in records)
            if count == 1 or count % 25 == 0:
                emit("response_operator_progress", name=name, count=count)
            return output

        return measured

    def instrument_factory(factory, name: str):
        if factory is None:
            return None

        def make(*args, **kwargs):
            return instrument_operator(factory(*args, **kwargs), name)

        return make

    def traced_implicit_solve(matvec, rhs, *args, **kwargs):
        started = time.perf_counter()
        entry = {
            "rhs_norm": float(torch.linalg.vector_norm(rhs).detach()),
            "pre_solver_seconds": started - _RESPONSE_STARTED[0],
        }
        emit("implicit_response_solver_entered", **entry)
        kwargs["linearize"] = instrument_factory(
            kwargs.get("linearize"), "linearize"
        )
        kwargs["rlinearize"] = instrument_factory(
            kwargs.get("rlinearize"), "rlinearize"
        )
        if kwargs.get("linearize") is None:
            kwargs["linearize"] = None
        if kwargs.get("rlinearize") is None:
            kwargs["rlinearize"] = None
        result = original(
            instrument_operator(matvec, "matvec"),
            rhs,
            *args,
            **kwargs,
        )
        entry.update(
            duration_s=time.perf_counter() - started,
            status=result.status.detach().cpu().tolist(),
            iterations=result.iterations.detach().cpu().tolist(),
            residual_norm=float(
                torch.linalg.vector_norm(result.residual).detach()
            ),
        )
        emit("implicit_response_solver_completed", **entry)
        return result

    implicit_root_module.implicit_solve = traced_implicit_solve
    return implicit_root_module, original, records


_RESPONSE_STARTED = [0.0]


def differentiate_positions(
    result: Result,
    positions: torch.Tensor,
    model_parameters: tuple[torch.Tensor, ...],
    *,
    observable: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Differentiate the selected physical Result observable once."""
    energy_only, energy_plus_fenergy = evaluate_energy(result)
    objective = energy_only if observable == "energy" else energy_plus_fenergy
    parameters = (positions, *model_parameters)
    response_module, original, records = _instrument_linear_response()
    _RESPONSE_STARTED[0] = time.perf_counter()
    emit("backward_started", observable=observable)
    started = time.perf_counter()
    try:
        gradients = torch.autograd.grad(
            objective,
            parameters,
            allow_unused=True,
        )
    finally:
        response_module.implicit_solve = original
    emit(
        "backward_completed",
        observable=observable,
        duration_s=time.perf_counter() - started,
        gradient_finite=(
            None
            if gradients[0] is None
            else bool(torch.isfinite(gradients[0]).all())
        ),
        gradient_max_abs=(
            None if gradients[0] is None else float(gradients[0].abs().max())
        ),
        gradient_norm=(
            None
            if gradients[0] is None
            else float(torch.linalg.vector_norm(gradients[0]))
        ),
        model_gradient_count=sum(value is not None for value in gradients[1:]),
        model_gradients_finite=all(
            value is None or bool(torch.isfinite(value).all())
            for value in gradients[1:]
        ),
        response_operator_counts={
            name: sum(item["operator"] == name for item in records)
            for name in ("matvec", "linearize", "rlinearize")
        },
        response_operator_seconds={
            name: sum(
                item["seconds"] for item in records if item["operator"] == name
            )
            for name in ("matvec", "linearize", "rlinearize")
        },
    )
    return objective.detach(), gradients


def differentiate_model_parameters(
    result: Result,
    model_parameters: tuple[torch.Tensor, ...],
    *,
    observable: str,
) -> tuple[torch.Tensor | None, ...]:
    """Expose Model-owned gradients separately when the memory budget allows."""
    energy_only, energy_plus_fenergy = evaluate_energy(result)
    objective = energy_only if observable == "energy" else energy_plus_fenergy
    return torch.autograd.grad(
        objective,
        model_parameters,
        allow_unused=True,
    )


def run_mapped_diagnostic(
    model: Model,
    system: Any,
    numbers: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Prepare independent tmpda roots and map their implicit responses."""
    from torch.func import vmap

    conformer_count = args.nconformers
    if not 2 <= conformer_count <= 8:
        raise ValueError("Mapped diagnostics require 2 to 8 conformers.")

    prepared_lanes: list[PreparedInputs] = []
    outcomes: list[Any] = []
    adapters: list[Any] = []
    lane_parameters: list[tuple[torch.Tensor, ...]] = []
    validated_roots: list[Any] = []
    lane_geometries = [
        construct_workload(index)[1] for index in range(conformer_count)
    ]
    positions = torch.stack(lane_geometries).requires_grad_(True)

    for lane_index in range(conformer_count):
        lane_numbers, _ = construct_workload(lane_index)
        if not torch.equal(lane_numbers, numbers):
            raise RuntimeError("The frozen conformers have different atoms.")
        lane_positions = positions[lane_index]
        case = PhysicalCase(
            model=model,
            system=system,
            numbers=numbers,
            positions=lane_positions,
            charge=torch.zeros(1, dtype=DTYPE),
            spin=None,
        )
        prepared = prepare_call_inputs(case)
        emit("mapped_lane_root_preparation_started", lane=lane_index)
        outcome, map_count, root_seconds = prepare_primal_root(prepared)
        adapter, parameters, raw_residual, validated = validate_physical_root(
            prepared, outcome
        )
        emit(
            "mapped_lane_root_validated",
            lane=lane_index,
            map_evaluations=map_count,
            root_seconds=root_seconds,
            iterations=int(outcome.iterations.detach()),
            converged=bool(outcome.converged),
            residual_l2=float(torch.linalg.vector_norm(raw_residual).detach()),
            residual_max=float(raw_residual.abs().amax().detach()),
            rss_peak_mb=peak_rss_mb(),
        )
        prepared_lanes.append(prepared)
        outcomes.append(outcome)
        adapters.append(adapter)
        lane_parameters.append(parameters)
        validated_roots.append(validated)

    adapter = adapters[0]
    if any(
        item.parameter_paths != adapter.parameter_paths for item in adapters
    ):
        raise RuntimeError("Mapped lane SCF tensor structures differ.")
    for lane_index, validated in enumerate(validated_roots):
        if validated.raw_map is not adapters[lane_index]:
            raise RuntimeError("A validated root lost its lane adapter.")

    roots = torch.stack([root.root_x for root in validated_roots])
    stacked_parameters = tuple(
        torch.stack([lane[index] for lane in lane_parameters])
        for index in range(len(adapter.parameter_paths))
    )
    iterations = torch.stack([outcome.iterations for outcome in outcomes])
    converged = torch.stack([outcome.converged for outcome in outcomes])
    residuals = torch.stack([outcome.residual for outcome in outcomes])
    statuses = [outcome.trajectory_status for outcome in outcomes]
    if any(status is None for status in statuses):
        raise RuntimeError("A mapped primal root lacks trajectory status.")
    status_values = tuple(
        torch.stack([getattr(status, name) for status in statuses])
        for name in (
            "fermi_converged",
            "electron_count_ok",
            "invalid_temperature",
            "too_many_electrons",
            "first_failure",
        )
    )

    core = system._active_exact_core()
    if core is None:
        raise RuntimeError("The mapped W2 System has no active exact core.")
    classical_terms = core.classical_terms
    charges = torch.stack([prepared.case.charge for prepared in prepared_lanes])

    def evaluate_classical(
        lane_positions: torch.Tensor, lane_charge: torch.Tensor
    ) -> torch.Tensor:
        values = tuple(
            _evaluate_classical_descriptor(
                term,
                lane_positions,
                lane_charge,
            )
            for term in classical_terms
        )
        return torch.stack(values).sum(0)

    classical = vmap(evaluate_classical)(positions, charges)

    def evaluate_lane(
        root: torch.Tensor,
        iteration: torch.Tensor,
        is_converged: torch.Tensor,
        residual: torch.Tensor,
        classical_energy: torch.Tensor,
        *values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameter_count = len(adapter.parameter_paths)
        parameters = values[:parameter_count]
        trajectory = SCFTrajectoryStatus(*values[parameter_count:])
        inputs, _ = adapter._rebuild(parameters)
        attached = implicit_root(root, *parameters, raw_map=adapter)
        mapped_result = adapter.evaluate(attached, *parameters)
        finalized = finalize_scf(
            mapped_result,
            inputs,
            iteration,
            is_converged,
            residual,
            trajectory,
        )
        energy = (
            classical_energy.sum()
            + finalized.energy.sum()
            + finalized.fenergy.sum()
        )
        return energy, finalized.fenergy.sum()

    dimensions = (
        0,
        0,
        0,
        0,
        0,
        *((0,) * len(stacked_parameters)),
        *((0,) * len(status_values)),
    )
    emit(
        "mapped_implicit_attachment_started",
        conformers=conformer_count,
        lane_iterations=iterations.detach().cpu().tolist(),
        lane_residuals=residuals.detach().cpu().tolist(),
        simultaneous_mapped_lanes=conformer_count,
    )
    started = time.perf_counter()
    mapped_energy, mapped_fenergy = vmap(evaluate_lane, in_dims=dimensions)(
        roots,
        iterations,
        converged,
        residuals,
        classical,
        *stacked_parameters,
        *status_values,
    )
    emit(
        "mapped_physical_values_completed",
        duration_s=time.perf_counter() - started,
        energy_only=mapped_energy.detach().cpu().tolist(),
        free_energy=mapped_fenergy.detach().cpu().tolist(),
        finite=bool(
            torch.isfinite(mapped_energy).all()
            & torch.isfinite(mapped_fenergy).all()
        ),
        rss_peak_mb=peak_rss_mb(),
    )
    if not (
        torch.isfinite(mapped_energy).all()
        and torch.isfinite(mapped_fenergy).all()
    ):
        raise RuntimeError("Mapped tmpda energies are nonfinite.")

    serial_energy = []
    serial_fenergy = []
    for lane_index, (prepared, outcome, lane_adapter, parameters) in enumerate(
        zip(prepared_lanes, outcomes, adapters, lane_parameters)
    ):
        map_result = lane_adapter.evaluate(outcome.root_x, *parameters)
        physical_result, _ = finalize_physical_result(
            prepared, outcome, map_result
        )
        serial_energy.append(physical_result.energy.sum())
        serial_fenergy.append(physical_result.fenergy.sum())
        emit(
            "mapped_lane_serial_reference",
            lane=lane_index,
            energy_only=float(physical_result.energy.sum().detach()),
            free_energy=float(physical_result.fenergy.sum().detach()),
        )
    serial_energy_tensor = torch.stack(serial_energy)
    serial_fenergy_tensor = torch.stack(serial_fenergy)
    torch.testing.assert_close(
        mapped_energy,
        serial_energy_tensor,
        atol=3.0e-8,
        rtol=3.0e-8,
        msg="Mapped tmpda energy differs from independently finalized lanes.",
    )
    torch.testing.assert_close(
        mapped_fenergy,
        serial_fenergy_tensor,
        atol=3.0e-9,
        rtol=3.0e-8,
        msg="Mapped tmpda free energy differs from independent lanes.",
    )

    emit("mapped_backward_started", observable="energy-only")
    backward_start = time.perf_counter()
    position_gradients = torch.autograd.grad(
        mapped_energy.sum(),
        positions,
        retain_graph=True,
    )[0]
    if not torch.isfinite(position_gradients).all():
        raise RuntimeError("Mapped tmpda position gradients are nonfinite.")
    emit(
        "mapped_backward_completed",
        duration_s=time.perf_counter() - backward_start,
        gradient_norm=float(
            torch.linalg.vector_norm(position_gradients).detach()
        ),
        gradient_max_abs=float(position_gradients.abs().max().detach()),
        rss_peak_mb=peak_rss_mb(),
    )
    emit(
        "mapped_diagnostic_complete",
        conformers=conformer_count,
        iterations=iterations.detach().cpu().tolist(),
        mapped_energy=mapped_energy.detach().cpu().tolist(),
        mapped_free_energy=mapped_fenergy.detach().cpu().tolist(),
        gradient_norm=float(
            torch.linalg.vector_norm(position_gradients).detach()
        ),
        rss_peak_mb=peak_rss_mb(),
    )


def run_diagnostic(args: argparse.Namespace) -> None:
    """Execute setup, root validation, and the selected observable boundary."""
    global _MEMORY_LIMIT_MB
    _MEMORY_LIMIT_MB = float(args.memory_limit_mb)
    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()

    emit(
        "environment",
        python=platform.python_version(),
        torch=torch.__version__,
        dxtb_source=dxtb_package.__file__,
        hostname=socket.gethostname(),
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        memory_limit_mb=_MEMORY_LIMIT_MB,
        workload="tmpda conformer 0, 69 atoms, float64, GFN1, PyTorch",
        displacement={
            "atom": args.displace_atom,
            "axis": args.displace_axis,
            "bohr": args.displace_bohr,
        },
        scf={
            "temperature_kelvin": 300.0,
            "mixer": "anderson",
            "maxiter": args.maxiter,
            "x_atol": args.x_atol,
            "x_atol_max": args.x_atol_max,
            "f_atol": 1.0e-8,
        },
    )

    started = time.perf_counter()
    numbers, base_positions = construct_workload()
    if args.displace_atom is not None:
        if not 0 <= args.displace_atom < numbers.numel():
            raise ValueError("--displace-atom is outside the tmpda geometry.")
        if args.displace_axis not in (0, 1, 2):
            raise ValueError("--displace-axis must be 0, 1, or 2.")
        base_positions = base_positions.clone()
        base_positions[
            args.displace_atom, args.displace_axis
        ] += args.displace_bohr
    elif args.displace_bohr != 0.0:
        raise ValueError("A nonzero displacement requires --displace-atom.")

    positions = base_positions.detach().clone().requires_grad_(True)
    charge = torch.zeros(1, dtype=DTYPE, device="cpu")
    case_start = time.perf_counter()
    model, system = setup_model_system(
        numbers,
        maxiter=args.maxiter,
        x_atol=args.x_atol,
        x_atol_max=args.x_atol_max,
    )
    case = PhysicalCase(
        model=model,
        system=system,
        numbers=numbers,
        positions=positions,
        charge=charge,
        spin=None,
    )
    emit(
        "model_system_setup_completed",
        duration_s=time.perf_counter() - case_start,
        atoms=int(numbers.numel()),
        classical_labels=[x.label for x in system.classicals.components],
        interaction_labels=[x.label for x in system.interactions.components],
    )

    if args.stage == "mapped":
        run_mapped_diagnostic(model, system, numbers, args)
        _MEMORY_STOP.set()
        return

    input_start = time.perf_counter()
    prepared = prepare_call_inputs(case)
    emit(
        "integral_interaction_scf_input_setup_completed",
        duration_s=time.perf_counter() - input_start,
        hcore_shape=list(prepared.integrals.hcore.shape),
        overlap_shape=list(prepared.integrals.overlap.shape),
        initial_state_shape=list(prepared.initial_state.shape),
        graph_on_position_inputs=bool(prepared.integrals.hcore.requires_grad),
    )

    emit("primal_root_preparation_started")
    try:
        outcome, map_evaluations, root_seconds = prepare_primal_root(prepared)
    except BaseException as error:
        emit(
            "primal_root_preparation_failed",
            error_type=type(error).__name__,
            error=str(error),
            duration_s=time.perf_counter() - started,
        )
        raise
    emit(
        "primal_root_preparation_completed",
        duration_s=root_seconds,
        map_evaluations=map_evaluations,
        iterations=int(outcome.iterations.detach()),
        converged=bool(outcome.converged),
        solver_residual=outcome.residual.detach().cpu().tolist(),
        trajectory_failure=(
            None
            if outcome.trajectory_status is None
            else outcome.trajectory_status.first_failure.detach().cpu().tolist()
        ),
        density_finite=bool(torch.isfinite(outcome.map_result.density).all()),
    )

    try:
        (
            adapter,
            tensor_parameters,
            raw_residual,
            validated_root,
        ) = validate_physical_root(prepared, outcome)
    except SCFConvergenceError as error:
        emit(
            "root_gate_failed_no_implicit_attachment",
            error=str(error),
        )
        raise
    emit(
        "root_gate_passed",
        map_evaluations=map_evaluations,
        independent_residual_norm=float(
            torch.linalg.vector_norm(raw_residual).detach()
        ),
    )
    with torch.no_grad():
        primal_result, primal_solve_result = finalize_physical_result(
            prepared,
            outcome,
            adapter.evaluate(outcome.root_x, *tensor_parameters),
        )
    emit(
        "primal_result_finalized",
        energy_atomwise=primal_result.energy.detach().cpu().tolist(),
        energy_only_hartree=float(primal_result.energy.detach().sum()),
        free_energy_atomwise=primal_result.fenergy.detach().cpu().tolist(),
        energy_plus_free_energy_hartree=float(
            (primal_result.energy + primal_result.fenergy).detach().sum()
        ),
        classical=[
            [name, value.detach().cpu().tolist()]
            for name, value in primal_result.classical
        ],
        energy_finite=bool(torch.isfinite(primal_result.energy).all()),
        free_energy_finite=bool(torch.isfinite(primal_result.fenergy).all()),
        density_finite=bool(torch.isfinite(primal_result.density).all()),
        final_converged=primal_solve_result.converged.detach().cpu().tolist(),
        final_iterations=primal_solve_result.iterations.detach().cpu().tolist(),
        final_residual=primal_solve_result.residual.detach().cpu().tolist(),
    )
    if not torch.isfinite(primal_result.energy).all():
        raise RuntimeError("The primal physical Result energy is nonfinite.")
    if args.stage == "root":
        emit("diagnostic_complete", completed_stage="root")
        _MEMORY_STOP.set()
        return

    model_parameters = tuple(
        value
        for _, value in model.par.named_parameters()
        if value.requires_grad
    )
    emit(
        "implicit_attachment_started",
        model_parameter_count=len(model_parameters),
        position_inputs_require_grad=positions.requires_grad,
        parameter_input_count=len(tensor_parameters),
    )
    root = attach_implicit_root(validated_root)
    physical_result, solve_result = finalize_physical_result(
        prepared, outcome, adapter.evaluate(root, *tensor_parameters)
    )
    energy_only, energy_plus_fenergy = evaluate_energy(physical_result)
    emit(
        "physical_result_finalized",
        energy_atomwise=physical_result.energy.detach().cpu().tolist(),
        energy_only_hartree=float(energy_only.detach()),
        free_energy_atomwise=physical_result.fenergy.detach().cpu().tolist(),
        energy_plus_free_energy_hartree=float(energy_plus_fenergy.detach()),
        scf_energy_atomwise=physical_result.scf.detach().cpu().tolist(),
        classical=[
            [name, value.detach().cpu().tolist()]
            for name, value in physical_result.classical
        ],
        density_finite=bool(torch.isfinite(physical_result.density).all()),
        charges_finite=bool(torch.isfinite(physical_result.charges.mono).all()),
        occupations_finite=bool(
            torch.isfinite(physical_result.occupation).all()
        ),
        orbital_energies_finite=bool(torch.isfinite(physical_result.emo).all()),
        iterations=physical_result.iterations.detach().cpu().tolist(),
        convergence=(
            None
            if physical_result.converged is None
            else physical_result.converged.detach().cpu().tolist()
        ),
        residual=solve_result.residual.detach().cpu().tolist(),
    )

    _, gradients = differentiate_positions(
        physical_result,
        positions,
        model_parameters,
        observable=args.observable,
    )
    if gradients[0] is None or not torch.isfinite(gradients[0]).all():
        raise RuntimeError("The implicit position gradient is not finite.")
    emit(
        "physical_forces",
        observable=args.observable,
        forces=(-gradients[0]).detach().cpu().tolist(),
    )
    emit("diagnostic_complete", completed_stage=args.stage)
    _MEMORY_STOP.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("root", "energy", "mapped"), default="root"
    )
    parser.add_argument(
        "--observable",
        choices=("energy", "energy-plus-free-energy"),
        default="energy",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--x-atol", type=float, default=1.0e-8)
    parser.add_argument("--x-atol-max", type=float, default=1.0e-8)
    parser.add_argument("--memory-limit-mb", type=float, default=16_000.0)
    parser.add_argument("--nconformers", type=int, default=2)
    parser.add_argument("--displace-atom", type=int)
    parser.add_argument("--displace-axis", type=int, default=0)
    parser.add_argument("--displace-bohr", type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    run_diagnostic(args)


if __name__ == "__main__":
    main()
