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
"""S1 problem, physical-state, method, strategy, and controller contracts."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB
from dxtb._src.scf.functional import (
    DynamicDampingSettings,
    LinearMethod,
    SCFConvergence,
    SCFMethodContext,
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
    SCFSolverSettings,
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
    fock_initial_guess,
    fock_map,
    functional_subsolves_ok,
    potential_initial_guess,
    potential_map,
    scf_map,
    simple_mix,
    solve_scf_simple,
    unpack_charges,
)
from dxtb._src.scf.functional.solver import _solve_map

from .test_functional_map import _prepare_case


def _map_for_representation(case, representation: SCFRepresentation):
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    if representation is SCFRepresentation.CHARGE:
        return case.x, None, scf_map
    if representation is SCFRepresentation.POTENTIAL:
        value, layout = potential_initial_guess(charges, case.inputs)
        return value, layout, potential_map
    value = fock_initial_guess(charges, case.inputs)
    return value, None, fock_map


def _assert_map_result_equal(actual, expected) -> None:
    torch.testing.assert_close(actual.mapped_x, expected.mapped_x)
    for name in (
        "hamiltonian",
        "density",
        "emo",
        "coefficients",
        "occupation",
        "orbital_energy",
    ):
        torch.testing.assert_close(
            getattr(actual.spectral, name), getattr(expected.spectral, name)
        )
    torch.testing.assert_close(actual.charges.mono, expected.charges.mono)
    for name in ("dipole", "quad"):
        actual_value = getattr(actual.charges, name)
        expected_value = getattr(expected.charges, name)
        assert (actual_value is None) == (expected_value is None)
        if actual_value is not None:
            torch.testing.assert_close(actual_value, expected_value)
    if actual.driving_potential is not None:
        assert expected.driving_potential is not None
        for name in ("mono", "dipole", "quad"):
            actual_value = getattr(actual.driving_potential, name)
            expected_value = getattr(expected.driving_potential, name)
            assert (actual_value is None) == (expected_value is None)
            if actual_value is not None:
                torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize(
    "representation",
    [
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 300.0], ids=["aufbau", "fermi"])
def test_problem_evaluation_delegates_to_authoritative_maps(
    representation: SCFRepresentation, temperature: float
) -> None:
    """All S1 representations reuse the existing physical map functions."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={"fermi_etemp": temperature},
    )
    x, layout, map_function = _map_for_representation(case, representation)
    problem = SCFProblem(case.inputs, representation, layout)

    actual = problem.evaluate(x)
    expected = (
        map_function(x, case.inputs)
        if layout is None
        else map_function(x, case.inputs, layout)
    )

    assert actual.x is x
    _assert_map_result_equal(actual.map_result, expected)
    torch.testing.assert_close(
        actual.delta, expected.mapped_x - x, atol=0.0, rtol=0.0
    )
    if temperature > 0.0:
        assert actual.map_result.fermi_status is not None


def test_physical_state_binds_tested_input_to_raw_map_result() -> None:
    """The physical state stores x and its exact raw map observation."""
    case = _prepare_case(GFN1_XTB)
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    state = problem.evaluate(case.x)

    assert state.x is case.x
    _assert_map_result_equal(state.map_result, scf_map(case.x, case.inputs))
    torch.testing.assert_close(
        state.delta,
        state.map_result.mapped_x - state.x,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "representation",
    [
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ],
)
def test_linear_strategy_controller_supports_each_representation(
    representation: SCFRepresentation,
) -> None:
    """One linear stage controls all three existing physical maps."""
    case = _prepare_case(GFN1_XTB)
    x, layout, _ = _map_for_representation(case, representation)
    problem = SCFProblem(case.inputs, representation, layout)
    settings = SCFSolverSettings(2, True, 0.4, 0.0, 0.0)
    strategy = SCFStrategy((SCFStage(LinearMethod()),))

    before = _solve_map(x, settings, problem.map, record_trace=True)
    after = _solve_map(
        x,
        settings,
        problem.map,
        record_trace=True,
        problem=problem,
        strategy=strategy,
    )

    torch.testing.assert_close(after.root_x, before.root_x)
    _assert_map_result_equal(after.map_result, before.map_result)
    torch.testing.assert_close(after.iterations, before.iterations)
    torch.testing.assert_close(after.converged, before.converged)
    torch.testing.assert_close(after.residual, before.residual)
    assert after.trace is not None and before.trace is not None
    torch.testing.assert_close(after.trace.map_inputs, before.trace.map_inputs)
    torch.testing.assert_close(after.trace.raw_mapped, before.trace.raw_mapped)
    torch.testing.assert_close(after.trace.iterates, before.trace.iterates)


def test_eager_and_fixed_trip_linear_controllers_agree() -> None:
    """Both lifecycle routes stop on the same tested physical root."""
    case = _prepare_case(GFN1_XTB, sample_name="H2")
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    settings = SCFSolverSettings(40, False, 0.4, 1.0e-8, 1.0e-8)
    strategy = SCFStrategy((SCFStage(LinearMethod()),))
    common = {
        "problem": problem,
        "strategy": strategy,
    }

    fixed = _solve_map(case.x, settings, problem.map, **common)
    eager = _solve_map(case.x, settings, problem.map, eager=True, **common)

    torch.testing.assert_close(eager.root_x, fixed.root_x)
    _assert_map_result_equal(eager.map_result, fixed.map_result)
    torch.testing.assert_close(eager.iterations, fixed.iterations)
    torch.testing.assert_close(eager.converged, fixed.converged)
    torch.testing.assert_close(eager.residual, fixed.residual)
    assert bool(eager.converged)


def test_zero_budget_fixed_trip_and_eager_keep_distinct_status_ownership() -> (
    None
):
    """Fixed-trip zero budget omits trajectory status; eager keeps it."""
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        scf_options={"fermi_etemp": 300.0, "maxiter": 0},
    )
    settings = SCFSolverSettings(0, False, 0.4, 1.0e-8, 1.0e-8)

    fixed = solve_scf_simple(case.x, case.inputs, settings)
    eager = solve_scf_simple(case.x, case.inputs, settings, eager=True)

    assert int(fixed.iterations) == int(eager.iterations) == 1
    assert not bool(fixed.converged)
    assert not bool(eager.converged)
    assert fixed.trajectory_status is None
    assert eager.trajectory_status is not None
    assert fixed.map_result.fermi_status is not None
    assert fixed.final_evaluation.fermi_status is not None
    assert bool(functional_subsolves_ok(fixed))

    _assert_map_result_equal(fixed.map_result, eager.map_result)
    torch.testing.assert_close(fixed.energy, eager.energy)
    torch.testing.assert_close(fixed.fenergy, eager.fenergy)
    torch.testing.assert_close(fixed.density, eager.density)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("damp", [0.0, 0.4, 0.999])
@pytest.mark.parametrize("residual", [0.0, 0.25])
def test_linear_method_matches_simple_mix(
    dtype: torch.dtype, damp: float, residual: float
) -> None:
    """LinearMethod preserves simple_mix arithmetic for both supported dtypes."""
    case = _prepare_case(GFN1_XTB, dtype=dtype)
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    state = problem.evaluate(case.x)
    proposal_x = state.x + residual
    context = SCFMethodContext(
        step=3,
        transition_active=torch.ones((), dtype=torch.bool),
        damp=case.x.new_tensor(damp),
        proposal_x=proposal_x,
    )
    proposal = LinearMethod().propose(problem, state, None, context)
    expected = simple_mix(state.x, proposal_x, context.damp)

    torch.testing.assert_close(proposal.x, expected, atol=0.0, rtol=0.0)
    assert proposal.state is None


def test_single_stage_strategy_helper_rejects_dynamic_execution() -> None:
    """The fixed-trip convenience rejects eager-only stage composition."""
    method = LinearMethod()
    stage = SCFStage(method)
    assert SCFStrategy((stage,)).single_stage() is stage
    first = SCFStage(
        method, switch_policy=SCFSwitchPolicy(max_stage_iterations=2)
    )
    with pytest.raises(NotImplementedError, match="eager-only"):
        SCFStrategy((first, stage)).single_stage()


def test_fixed_trip_multi_stage_strategy_is_rejected_by_solver_entry() -> None:
    """Fixed-trip execution rejects dynamic stages rather than ignoring them."""
    case = _prepare_case(GFN1_XTB)
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    settings = SCFSolverSettings(1, False, 0.4, 1.0e-8, 1.0e-8)
    stage = SCFStage(
        LinearMethod(), switch_policy=SCFSwitchPolicy(max_stage_iterations=1)
    )
    final = SCFStage(LinearMethod())

    with pytest.raises(NotImplementedError, match="eager-only"):
        _solve_map(
            case.x,
            settings,
            problem.map,
            problem=problem,
            strategy=SCFStrategy((stage, final)),
        )


def test_convergence_boundary_preserves_strict_two_norm_predicate() -> None:
    """The S1 boundary uses the unchanged strict L2 and max norms."""
    residual = SCFConvergence.residual(torch.tensor([0.5, 0.0]))
    assert not bool(SCFConvergence(0.5, 0.6).converged(residual))
    assert bool(SCFConvergence(0.5001, 0.6).converged(residual))


@pytest.mark.parametrize("mix_guess", [False, True])
@pytest.mark.parametrize("maxiter", [0, 1, 6])
@pytest.mark.parametrize("eager", [False, True])
def test_linear_controller_matches_transitional_simple_adapter(
    mix_guess: bool, maxiter: int, eager: bool
) -> None:
    """The new method adapter preserves shared controller lifecycle behavior."""
    case = _prepare_case(GFN1_XTB)
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    settings = SCFSolverSettings(
        maxiter=maxiter,
        mix_guess=mix_guess,
        damp=0.4,
        x_atol=0.0,
        x_atol_max=0.0,
        dynamic_damping=DynamicDampingSettings(enabled=True, factor=0.99),
    )
    strategy = SCFStrategy((SCFStage(LinearMethod()),))
    baseline = _solve_map(
        case.x,
        settings,
        problem.map,
        eager=eager,
        record_trace=not eager,
    )
    routed = _solve_map(
        case.x,
        settings,
        problem.map,
        eager=eager,
        record_trace=not eager,
        problem=problem,
        strategy=strategy,
    )

    torch.testing.assert_close(routed.root_x, baseline.root_x)
    _assert_map_result_equal(routed.map_result, baseline.map_result)
    torch.testing.assert_close(routed.iterations, baseline.iterations)
    torch.testing.assert_close(routed.converged, baseline.converged)
    torch.testing.assert_close(routed.residual, baseline.residual)
    torch.testing.assert_close(
        routed.damping_state.value, baseline.damping_state.value
    )
    torch.testing.assert_close(
        routed.damping_state.switched, baseline.damping_state.switched
    )
    assert (routed.trace is None) == eager
    if not eager:
        assert baseline.trace is not None and routed.trace is not None
        for field in fields(routed.trace):
            actual = getattr(routed.trace, field.name)
            expected = getattr(baseline.trace, field.name)
            if isinstance(actual, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            elif isinstance(actual, tuple):
                assert len(actual) == len(expected)
                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(actual_tensor, expected_tensor)
            else:
                assert actual == expected


@pytest.mark.parametrize(
    "representation",
    [SCFRepresentation.CHARGE, SCFRepresentation.POTENTIAL],
)
def test_converged_outcome_retains_the_tested_map_input(
    representation: SCFRepresentation,
) -> None:
    """A finite-tolerance root remains the input paired with its map result."""
    case = _prepare_case(GFN1_XTB)
    x, layout, _ = _map_for_representation(case, representation)
    problem = SCFProblem(case.inputs, representation, layout)
    settings = SCFSolverSettings(100, False, 0.4, 1.0e-8, 1.0e-8)
    outcome = _solve_map(
        x,
        settings,
        problem.map,
        problem=problem,
        strategy=SCFStrategy((SCFStage(LinearMethod()),)),
    )
    expected = problem.evaluate(outcome.root_x)

    assert bool(outcome.converged)
    assert torch.count_nonzero(outcome.map_result.mapped_x - outcome.root_x) > 0
    torch.testing.assert_close(outcome.root_x, expected.x, atol=0.0, rtol=0.0)
    _assert_map_result_equal(outcome.map_result, expected.map_result)


def test_linear_route_supports_reverse_and_forward_transforms() -> None:
    """The fixed-trip linear route remains usable by torch.func transforms."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={"maxiter": 2, "mix_guess": True, "x_atol": 0.0},
    )
    settings = SCFSolverSettings(2, True, 0.4, 0.0, 0.0)

    def energy(guess: torch.Tensor) -> torch.Tensor:
        return solve_scf_simple(guess, case.inputs, settings).energy.sum()

    direction = torch.ones_like(case.x) / case.x.numel() ** 0.5
    value, tangent = jvp(energy, (case.x,), (direction,))
    forward = jacfwd(energy)(case.x)
    reverse = jacrev(energy)(case.x)
    mapped = vmap(energy)(torch.stack((case.x, case.x + 1.0e-7)))

    assert torch.isfinite(value)
    assert torch.isfinite(tangent)
    assert torch.isfinite(forward).all()
    assert torch.isfinite(reverse).all()
    assert torch.isfinite(mapped).all()
    torch.testing.assert_close(forward, reverse, atol=2.0e-10, rtol=2.0e-10)
