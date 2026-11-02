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
"""Acceptance checks for mapped, independently prepared implicit SCF roots."""

from __future__ import annotations

from dataclasses import replace
from test.test_scf.test_functional_root_derivatives import _case, _run_root
from types import SimpleNamespace

import pytest
import torch
from torch.func import jacrev, jvp, vmap

from dxtb import labels
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional import setup_scf_inputs
from dxtb._src.scf.functional.implicit import (
    _prepare_scf_root,
    _SCFMapTensorTree,
    implicit_root,
)
from dxtb._src.scf.functional.solver import (
    SCFTrajectoryStatus,
    finalize_scf,
)
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.typing.exceptions import SCFConvergenceError

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}
_TOL = 1.0e-12


def _inputs_at_positions(case, positions: torch.Tensor):
    """Build position-dependent SCF tensors from the shared Model owner."""
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
    interaction_data = _interaction_data(
        case.system, positions, case.interactions
    )
    inputs = setup_scf_inputs(
        case.numbers,
        integrals,
        n0,
        occupation,
        nel,
        case.interactions,
        interaction_data,
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
    lane = SimpleNamespace(**vars(case))
    lane.positions = positions
    lane.integrals = integrals
    lane.n0 = n0
    lane.occupation = occupation
    lane.nel = nel
    lane.interaction_data = interaction_data
    lane.inputs = inputs
    lane.initial_charges = initial_charges
    return lane, inputs


def _prepared_lane(case, positions: torch.Tensor):
    from dxtb._src.scf.functional.implicit import _SCFMapTensorTree

    lane, inputs = _inputs_at_positions(case, positions)
    with torch.no_grad():
        outcome = _run_root(
            lane,
            inputs,
            tolerance=_TOL,
            maxiter=300,
            eager=True,
        )
    adapter, parameters = _SCFMapTensorTree.prepare(
        inputs, labels.SCP_MODE_CHARGE
    )
    raw_residual = (
        adapter.evaluate(outcome.root, *parameters).mapped_x - outcome.root
    )
    residual_norm = torch.linalg.vector_norm(raw_residual)
    assert outcome.converged
    assert torch.isfinite(raw_residual).all()
    assert residual_norm <= _TOL
    assert outcome.outcome.trajectory_status is not None
    assert bool(outcome.outcome.trajectory_status.fermi_converged.all())
    assert bool(outcome.outcome.trajectory_status.electron_count_ok.all())
    return lane, inputs, outcome, adapter, parameters, residual_norm


def _metadata(outcomes):
    statuses = [outcome.outcome.trajectory_status for outcome in outcomes]
    assert all(status is not None for status in statuses)
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
    return (
        torch.stack(
            [
                torch.as_tensor(o.map_count, device=o.root.device)
                for o in outcomes
            ]
        ),
        torch.stack(
            [
                torch.as_tensor(o.converged, device=o.root.device)
                for o in outcomes
            ]
        ),
        torch.stack([o.reported_residual for o in outcomes]),
        status_values,
    )


def _evaluate_lane(
    adapter, root, parameters, iterations, converged, residual, status
):
    inputs, _ = adapter._rebuild(parameters)
    attached = implicit_root(root, *parameters, raw_map=adapter)
    map_result = adapter.evaluate(attached, *parameters)
    trajectory_status = SCFTrajectoryStatus(*status)
    result = finalize_scf(
        map_result,
        inputs,
        iterations,
        converged,
        residual,
        trajectory_status,
        root_x=attached,
    )
    return result.energy.sum(), result.fenergy.sum()


def _mapped_observables(
    positions: torch.Tensor,
    roots: torch.Tensor,
    case,
    outcomes,
    adapter,
):
    from dxtb._src.scf.functional.implicit import _SCFMapTensorTree

    lane_parameters = []
    for lane_index in range(positions.shape[0]):
        _, inputs = _inputs_at_positions(case, positions[lane_index])
        lane_adapter, values = _SCFMapTensorTree.prepare(
            inputs, labels.SCP_MODE_CHARGE
        )
        assert lane_adapter.parameter_paths == adapter.parameter_paths
        lane_parameters.append(values)
    stacked = tuple(
        torch.stack([values[index] for values in lane_parameters])
        for index in range(len(adapter.parameter_paths))
    )
    iterations, converged, residual, status = _metadata(outcomes)

    def evaluate(
        root,
        parameters,
        iteration,
        is_converged,
        raw_residual,
        *status_values,
    ):
        return _evaluate_lane(
            adapter,
            root,
            parameters,
            iteration,
            is_converged,
            raw_residual,
            status_values,
        )

    return vmap(evaluate, in_dims=(0, 0, 0, 0, 0, *((0,) * 5)))(
        roots,
        stacked,
        iterations,
        converged,
        residual,
        *status,
    )


def _mapped_attached_roots(roots, parameters, adapter):
    """Attach each prepared root to its matching lane parameters."""

    def attach(root, *lane_parameters):
        return implicit_root(root, *lane_parameters, raw_map=adapter)

    parameter_dims = (0,) * len(parameters)
    return vmap(attach, in_dims=(0, *parameter_dims))(roots, *parameters)


def test_two_geometry_roots_match_independent_serial_energy_and_derivatives() -> (
    None
):
    """Map separately validated H2O roots and preserve lane independence."""
    from dxtb import GFN1_XTB
    from dxtb._src.scf.functional.implicit import _SCFMapTensorTree

    case = _case(
        "H2O",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=_TOL,
        maxiter=300,
        temperature=300.0,
        int_driver=labels.INTDRIVER_PYTORCH,
    )
    model_parameter = case.system.h0_setup.selfenergy
    model_parameter.requires_grad_(True)
    base = case.positions.detach().clone()
    displaced = base.clone()
    displaced[1, 0] += 0.2
    positions = torch.stack((base, displaced))

    prepared = [_prepared_lane(case, positions[index]) for index in range(2)]
    lanes, inputs, outcomes, adapters, parameters, residuals = zip(*prepared)
    assert adapters[0].parameter_paths == adapters[1].parameter_paths
    torch.testing.assert_close(
        inputs[0].index.shells_to_atom,
        inputs[1].index.shells_to_atom,
    )
    torch.testing.assert_close(
        inputs[0].index.orbitals_to_shell,
        inputs[1].index.orbitals_to_shell,
    )
    assert lanes[0].system is lanes[1].system
    assert outcomes[0].map_count != outcomes[1].map_count
    torch.testing.assert_close(
        residuals[0],
        torch.linalg.vector_norm(
            adapters[0].evaluate(outcomes[0].root, *parameters[0]).mapped_x
            - outcomes[0].root
        ),
        atol=0,
        rtol=0,
    )

    # The normal preparation boundary also rejects a converged root if the
    # explicit inputs no longer describe that root's physical problem.
    with pytest.raises(SCFConvergenceError, match="supplied physical inputs"):
        _prepare_scf_root(
            outcomes[0].outcome,
            maxiter=300,
            x_atol=_TOL,
            x_atol_max=_TOL,
            final_valid=torch.ones((), dtype=torch.bool),
            raw_map=adapters[0],
            parameters=parameters[1],
        )

    roots = torch.stack([outcome.root for outcome in outcomes])
    adapter = adapters[0]
    lane_parameters = []
    for lane_inputs in inputs:
        _, values = _SCFMapTensorTree.prepare(
            lane_inputs, labels.SCP_MODE_CHARGE
        )
        lane_parameters.append(values)
    stacked_parameters = tuple(
        torch.stack([values[index] for values in lane_parameters])
        for index in range(len(adapter.parameter_paths))
    )
    mapped_attached = _mapped_attached_roots(roots, stacked_parameters, adapter)
    serial_attached = torch.stack(
        [
            implicit_root(
                roots[index], *lane_parameters[index], raw_map=adapters[index]
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(
        mapped_attached,
        serial_attached,
        atol=2.0e-10,
        rtol=2.0e-10,
        msg="Mapped attached roots differ from independent serial attachments.",
    )
    mapped_energy, mapped_fenergy = _mapped_observables(
        positions, roots, case, outcomes, adapter
    )
    serial_energy = torch.stack(
        [outcome.electronic_energy for outcome in outcomes]
    )
    serial_total = torch.stack(
        [outcome.finalized_energy for outcome in outcomes]
    )
    serial_fenergy = serial_total - serial_energy
    torch.testing.assert_close(
        mapped_energy, serial_energy, atol=3e-9, rtol=3e-9
    )
    torch.testing.assert_close(
        mapped_fenergy, serial_fenergy, atol=3e-9, rtol=3e-9
    )

    n0_index = adapter.parameter_paths.index("n0")
    n0_direction = torch.zeros_like(lane_parameters[0][n0_index])
    n0_direction[-1] = 0.2
    iterations, converged, reported_residual, trajectory_status = _metadata(
        outcomes
    )

    def lane_energy_at_n0_delta(
        root,
        delta,
        values,
        iteration,
        is_converged,
        residual,
        *status,
    ):
        current_values = list(values)
        current_values[n0_index] = (
            current_values[n0_index] + delta * n0_direction
        )
        energy, fenergy = _evaluate_lane(
            adapter,
            root,
            tuple(current_values),
            iteration,
            is_converged,
            residual,
            status,
        )
        return energy + fenergy

    def mapped_energy_at_n0_deltas(deltas):
        return vmap(
            lane_energy_at_n0_delta,
            in_dims=(0, 0, 0, 0, 0, 0, *((0,) * 5)),
        )(
            roots,
            deltas,
            stacked_parameters,
            iterations,
            converged,
            reported_residual,
            *trajectory_status,
        )

    mapped_n0_jacobian = jacrev(mapped_energy_at_n0_deltas)(
        torch.zeros(2, **_DD)
    )
    serial_n0_jacobian = []
    for lane_index in range(2):
        lane_status = tuple(value[lane_index] for value in trajectory_status)

        def serial_energy_at_n0_delta(delta):
            return lane_energy_at_n0_delta(
                roots[lane_index],
                delta,
                lane_parameters[lane_index],
                iterations[lane_index],
                converged[lane_index],
                reported_residual[lane_index],
                *lane_status,
            )

        serial_n0_jacobian.append(
            jacrev(serial_energy_at_n0_delta)(torch.zeros((), **_DD))
        )
    serial_n0_jacobian = torch.stack(serial_n0_jacobian)
    torch.testing.assert_close(
        mapped_n0_jacobian.diagonal(),
        serial_n0_jacobian,
        atol=2.0e-7,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        mapped_n0_jacobian[0, 1],
        torch.zeros_like(mapped_n0_jacobian[0, 1]),
        atol=2.0e-9,
        rtol=0.0,
    )
    torch.testing.assert_close(
        mapped_n0_jacobian[1, 0],
        torch.zeros_like(mapped_n0_jacobian[1, 0]),
        atol=2.0e-9,
        rtol=0.0,
    )

    serial_attached_total = []
    for index in range(2):
        iterations, converged, residual, status = _metadata([outcomes[index]])
        attached_energy, attached_fenergy = _evaluate_lane(
            adapters[index],
            roots[index],
            lane_parameters[index],
            iterations[0],
            converged[0],
            residual[0],
            tuple(value[0] for value in status),
        )
        serial_attached_total.append(attached_energy + attached_fenergy)
    serial_attached_total = torch.stack(serial_attached_total)

    model_gradient = torch.autograd.grad(
        (mapped_energy + mapped_fenergy).sum(),
        model_parameter,
        retain_graph=True,
    )[0]
    serial_gradient = torch.autograd.grad(
        serial_attached_total.sum(), model_parameter, retain_graph=True
    )[0]
    torch.testing.assert_close(
        model_gradient, serial_gradient, atol=2e-8, rtol=2e-7
    )

    def mapped_total(current_positions):
        energy, fenergy = _mapped_observables(
            current_positions, roots, case, outcomes, adapter
        )
        return energy + fenergy

    mapped_position_jacobian = jacrev(mapped_total)(positions)
    serial_position_jacobian = []
    for lane_index in range(2):
        lane_adapter = adapters[lane_index]
        lane_outcome = outcomes[lane_index]

        def serial_total_at(current_position):
            lane, current_inputs = _inputs_at_positions(case, current_position)
            current_adapter, current_parameters = _SCFMapTensorTree.prepare(
                current_inputs, labels.SCP_MODE_CHARGE
            )
            assert (
                current_adapter.parameter_paths == lane_adapter.parameter_paths
            )
            attached = implicit_root(
                lane_outcome.root,
                *current_parameters,
                raw_map=lane_adapter,
            )
            current_map = lane_adapter.evaluate(attached, *current_parameters)
            status = lane_outcome.outcome.trajectory_status
            finalized = finalize_scf(
                current_map,
                current_inputs,
                torch.as_tensor(lane_outcome.map_count),
                torch.as_tensor(lane_outcome.converged),
                lane_outcome.reported_residual,
                status,
                root_x=attached,
            )
            return finalized.energy.sum() + finalized.fenergy.sum()

        serial_position_jacobian.append(
            jacrev(serial_total_at)(positions[lane_index])
        )
    serial_position_jacobian = torch.stack(serial_position_jacobian)
    for lane_index in range(2):
        torch.testing.assert_close(
            mapped_position_jacobian[lane_index, lane_index],
            serial_position_jacobian[lane_index],
            atol=2e-7,
            rtol=2e-6,
        )
    torch.testing.assert_close(
        mapped_position_jacobian[0, 1],
        torch.zeros_like(mapped_position_jacobian[0, 1]),
        atol=2e-9,
        rtol=0,
    )
    torch.testing.assert_close(
        mapped_position_jacobian[1, 0],
        torch.zeros_like(mapped_position_jacobian[1, 0]),
        atol=2e-9,
        rtol=0,
    )

    reverse_positions = positions.flip(0)
    reverse_roots = roots.flip(0)
    reverse_outcomes = tuple(reversed(outcomes))
    reverse_values = _mapped_observables(
        reverse_positions, reverse_roots, case, reverse_outcomes, adapter
    )
    torch.testing.assert_close(reverse_values[0], mapped_energy.flip(0))
    torch.testing.assert_close(reverse_values[1], mapped_fenergy.flip(0))


def test_mismatched_physical_root_is_rejected_before_implicit_attachment() -> (
    None
):
    """Reject a root prepared for n0 when explicit inputs use perturbed n0."""
    from dxtb import GFN1_XTB
    from dxtb._src.scf.functional.implicit import _SCFMapTensorTree

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=_TOL,
        maxiter=300,
        int_driver=labels.INTDRIVER_PYTORCH,
    )
    with torch.no_grad():
        outcome = _run_root(
            case,
            case.inputs,
            tolerance=_TOL,
            maxiter=300,
            eager=True,
        )
    adapter, parameters = _SCFMapTensorTree.prepare(
        case.inputs, labels.SCP_MODE_CHARGE
    )
    n0_index = adapter.parameter_paths.index("n0")
    mismatched = list(parameters)
    direction = torch.zeros_like(parameters[n0_index])
    direction[-1] = 0.2
    mismatched[n0_index] = mismatched[n0_index] + 1.0e-3 * direction

    with pytest.raises(SCFConvergenceError, match="supplied physical inputs"):
        _prepare_scf_root(
            outcome.outcome,
            maxiter=300,
            x_atol=_TOL,
            x_atol_max=_TOL,
            final_valid=torch.ones((), dtype=torch.bool),
            raw_map=adapter,
            parameters=tuple(mismatched),
        )


def test_implicit_root_retains_finite_temperature_free_energy_responses() -> (
    None
):
    """Check nonzero entropy and temperature derivatives at a physical root."""
    from dxtb import GFN1_XTB

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=_TOL,
        maxiter=300,
        temperature=10000.0,
        int_driver=labels.INTDRIVER_PYTORCH,
    )
    model_parameter = case.system.h0_setup.selfenergy
    model_parameter.requires_grad_(True)
    positions = case.positions.detach().clone().requires_grad_(True)
    lane, inputs = _inputs_at_positions(case, positions)
    with torch.no_grad():
        outcome = _run_root(
            lane,
            inputs,
            tolerance=_TOL,
            maxiter=300,
            eager=True,
        )
    assert outcome.converged
    adapter, base_parameters = _SCFMapTensorTree.prepare(
        inputs, labels.SCP_MODE_CHARGE
    )
    independent_residual = (
        adapter.evaluate(outcome.root, *base_parameters).mapped_x - outcome.root
    )
    assert torch.linalg.vector_norm(independent_residual) <= _TOL

    kt_index = adapter.parameter_paths.index("fermi_settings.kt")
    temperature_parameter = torch.zeros((), **_DD, requires_grad=True)
    prepared = _prepare_scf_root(
        outcome.outcome,
        maxiter=300,
        x_atol=_TOL,
        x_atol_max=_TOL,
        final_valid=torch.ones((), dtype=torch.bool),
        raw_map=adapter,
        parameters=base_parameters,
    )

    def parameters_at_temperature_delta(delta):
        values = list(base_parameters)
        values[kt_index] = values[kt_index] + delta
        return tuple(values)

    def observables_at_temperature_delta(delta):
        values = parameters_at_temperature_delta(delta)
        attached = implicit_root(prepared.root_x, *values, raw_map=adapter)
        map_result = adapter.evaluate(attached, *values)
        current_inputs, _ = adapter._rebuild(values)
        finalized = finalize_scf(
            map_result,
            current_inputs,
            outcome.outcome.iterations,
            outcome.outcome.converged,
            outcome.outcome.residual,
            outcome.outcome.trajectory_status,
            outcome.outcome.damping_state,
            root_x=attached,
        )
        return finalized.energy.sum(), finalized.fenergy.sum()

    electronic_energy, free_energy = observables_at_temperature_delta(
        temperature_parameter
    )
    finalized_total = electronic_energy + free_energy

    torch.testing.assert_close(
        electronic_energy,
        outcome.electronic_energy,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    torch.testing.assert_close(
        finalized_total,
        outcome.finalized_energy,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    assert torch.isfinite(free_energy)
    assert abs(float(free_energy.detach())) > 1.0e-4
    assert torch.isfinite(outcome.last_map_result.spectral.occupation).all()
    assert torch.any(
        (outcome.last_map_result.spectral.occupation > 1.0e-6)
        & (outcome.last_map_result.spectral.occupation < 1.0 - 1.0e-6)
    )
    torch.testing.assert_close(
        outcome.last_map_result.spectral.occupation.sum(),
        inputs.nel.sum(),
        atol=2.0e-10,
        rtol=0.0,
    )

    direction = torch.ones_like(temperature_parameter)

    def free_energy(delta):
        return observables_at_temperature_delta(delta)[1]

    def first_response(delta):
        return jvp(free_energy, (delta,), (direction,))[1]

    def second_response(delta):
        return jvp(first_response, (delta,), (direction,))[1]

    first = first_response(temperature_parameter.detach())
    second = second_response(temperature_parameter.detach())
    third = jvp(
        second_response, (temperature_parameter.detach(),), (direction,)
    )[1]
    assert torch.isfinite(first)
    assert torch.isfinite(second)
    assert torch.isfinite(third)
    assert abs(float(first.detach())) > 1.0e-8

    def independently_converged_free_energy(kt: torch.Tensor) -> torch.Tensor:
        physical_inputs = replace(
            inputs,
            fermi_settings=replace(inputs.fermi_settings, kt=kt),
        )
        with torch.no_grad():
            reference = _run_root(
                lane,
                physical_inputs,
                tolerance=_TOL,
                maxiter=300,
                eager=True,
            )
            reference_adapter, reference_parameters = _SCFMapTensorTree.prepare(
                physical_inputs, labels.SCP_MODE_CHARGE
            )
            reference_residual = (
                reference_adapter.evaluate(
                    reference.root, *reference_parameters
                ).mapped_x
                - reference.root
            )
        assert reference.converged
        assert torch.linalg.vector_norm(reference_residual) <= _TOL
        return reference.finalized_energy - reference.electronic_energy

    kt_base = base_parameters[kt_index].detach()
    fd_step = kt_base * 1.0e-2
    f_minus_two = independently_converged_free_energy(kt_base - 2.0 * fd_step)
    f_minus_one = independently_converged_free_energy(kt_base - fd_step)
    f_plus_one = independently_converged_free_energy(kt_base + fd_step)
    f_plus_two = independently_converged_free_energy(kt_base + 2.0 * fd_step)
    f_center = free_energy(temperature_parameter.detach())
    fd_first = (
        f_minus_two - 8.0 * f_minus_one + 8.0 * f_plus_one - f_plus_two
    ) / (12.0 * fd_step)
    fd_second = (
        -f_plus_two
        + 16.0 * f_plus_one
        - 30.0 * f_center
        + 16.0 * f_minus_one
        - f_minus_two
    ) / (12.0 * fd_step.square())
    fd_third = (
        f_plus_two - 2.0 * f_plus_one + 2.0 * f_minus_one - f_minus_two
    ) / (2.0 * fd_step.pow(3))
    torch.testing.assert_close(first, fd_first, atol=2.0e-7, rtol=4.0e-3)
    torch.testing.assert_close(second, fd_second, atol=2.0e-4, rtol=2.0e-2)
    torch.testing.assert_close(third, fd_third, atol=2.0e-3, rtol=3.0e-2)

    position_gradient, model_gradient = torch.autograd.grad(
        finalized_total, (positions, model_parameter)
    )
    assert torch.isfinite(position_gradient).all()
    assert torch.isfinite(model_gradient).all()

    serial_positions = case.positions.detach().clone().requires_grad_(True)
    serial_lane, serial_inputs = _inputs_at_positions(case, serial_positions)
    with torch.no_grad():
        serial_outcome = _run_root(
            serial_lane,
            serial_inputs,
            tolerance=_TOL,
            maxiter=300,
            eager=True,
        )
    serial_adapter, serial_parameters = _SCFMapTensorTree.prepare(
        serial_inputs, labels.SCP_MODE_CHARGE
    )
    serial_residual = (
        serial_adapter.evaluate(
            serial_outcome.root, *serial_parameters
        ).mapped_x
        - serial_outcome.root
    )
    assert serial_outcome.converged
    assert torch.linalg.vector_norm(serial_residual) <= _TOL
    serial_prepared = _prepare_scf_root(
        serial_outcome.outcome,
        maxiter=300,
        x_atol=_TOL,
        x_atol_max=_TOL,
        final_valid=torch.ones((), dtype=torch.bool),
        raw_map=serial_adapter,
        parameters=serial_parameters,
    )
    serial_attached = serial_prepared.attach()
    serial_map_result = serial_adapter.evaluate(
        serial_attached, *serial_parameters
    )
    serial_inputs_rebuilt, _ = serial_adapter._rebuild(serial_parameters)
    serial_finalized = finalize_scf(
        serial_map_result,
        serial_inputs_rebuilt,
        serial_outcome.outcome.iterations,
        serial_outcome.outcome.converged,
        serial_outcome.outcome.residual,
        serial_outcome.outcome.trajectory_status,
        serial_outcome.outcome.damping_state,
        root_x=serial_attached,
    )
    serial_physical_total = (
        serial_finalized.energy.sum() + serial_finalized.fenergy.sum()
    )
    serial_position_gradient, serial_model_gradient = torch.autograd.grad(
        serial_physical_total, (serial_positions, model_parameter)
    )
    torch.testing.assert_close(
        serial_finalized.energy.sum(),
        electronic_energy,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    torch.testing.assert_close(
        serial_finalized.fenergy.sum(),
        free_energy(temperature_parameter.detach()),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    torch.testing.assert_close(
        position_gradient,
        serial_position_gradient,
        atol=2.0e-7,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        model_gradient,
        serial_model_gradient,
        atol=2.0e-8,
        rtol=2.0e-7,
    )
    assert outcome.outcome.trajectory_status is not None
    assert bool(outcome.outcome.trajectory_status.fermi_converged.all())
    assert bool(outcome.outcome.trajectory_status.electron_count_ok.all())
