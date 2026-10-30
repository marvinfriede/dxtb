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
"""S6 stage policy, paired representation conversion, and budget tests."""

from __future__ import annotations

from dataclasses import replace
from inspect import getsource

import pytest
import torch
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.scf.functional.anderson import (
    AndersonMethod,
    AndersonMethodSettings,
)
from dxtb._src.scf.functional.broyden import BroydenMethod
from dxtb._src.scf.functional.cdiis import CDIISMethod
from dxtb._src.scf.functional.data import unpack_charges
from dxtb._src.scf.functional.map import (
    fock_initial_guess,
    fock_map,
    potential_initial_guess,
)
from dxtb._src.scf.functional.method import LinearMethod, SCFMethod
from dxtb._src.scf.functional.problem import (
    SCFProblem,
    SCFRepresentation,
    convert_physical_state_to_fock,
)
from dxtb._src.scf.functional.solver import (
    DampingState,
    DynamicDampingSettings,
    SCFSolverSettings,
    SCFSwitchReason,
    _method_state_tensors,
    _solve_map,
)
from dxtb._src.scf.functional.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)

from .test_functional_map import _prepare_case


@pytest.mark.parametrize(
    "representation",
    [SCFRepresentation.CHARGE, SCFRepresentation.POTENTIAL],
)
def test_compact_observation_converts_to_exact_fock_pair(
    representation: SCFRepresentation,
) -> None:
    """Conversion pairs the generating Hamiltonian with its fresh Fock map."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    if representation is SCFRepresentation.CHARGE:
        guess, layout = case.x, None
    else:
        guess, layout = potential_initial_guess(charges, case.inputs)

    problem = SCFProblem(case.inputs, representation, layout)
    physical = problem.evaluate(guess)
    fock_problem, converted = convert_physical_state_to_fock(problem, physical)

    assert fock_problem.representation is SCFRepresentation.FOCK
    torch.testing.assert_close(
        converted.x, physical.map_result.spectral.hamiltonian
    )
    direct = fock_map(converted.x, case.inputs)
    torch.testing.assert_close(converted.map_result.mapped_x, direct.mapped_x)
    torch.testing.assert_close(
        converted.map_result.spectral.hamiltonian,
        direct.spectral.hamiltonian,
    )
    torch.testing.assert_close(
        converted.map_result.spectral.density, direct.spectral.density
    )
    torch.testing.assert_close(
        converted.map_result.spectral.emo, direct.spectral.emo
    )
    torch.testing.assert_close(
        converted.map_result.spectral.coefficients,
        direct.spectral.coefficients,
    )
    torch.testing.assert_close(
        converted.map_result.spectral.charges.mono,
        direct.spectral.charges.mono,
    )
    torch.testing.assert_close(
        converted.map_result.spectral.occupation, direct.spectral.occupation
    )
    torch.testing.assert_close(
        converted.map_result.spectral.orbital_energy,
        direct.spectral.orbital_energy,
    )
    assert (converted.map_result.fermi_status is None) == (
        direct.fermi_status is None
    )
    if converted.map_result.fermi_status is not None:
        assert direct.fermi_status is not None
        for name in (
            "converged",
            "residual",
            "invalid_temperature",
            "too_many_electrons",
            "electron_count_ok",
        ):
            torch.testing.assert_close(
                getattr(converted.map_result.fermi_status, name),
                getattr(direct.fermi_status, name),
            )
    assert converted.map_result.driving_potential is None


def test_fock_observation_conversion_is_identity() -> None:
    """A Fock stage keeps the exact input/result pair without a remap."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    guess = fock_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    physical = problem.evaluate(guess)
    converted_problem, converted = convert_physical_state_to_fock(
        problem, physical
    )

    assert converted_problem is problem
    assert converted is physical


@pytest.mark.parametrize(
    "method",
    [
        BroydenMethod(),
        AndersonMethod(AndersonMethodSettings(0.1, 5, True, 0.01)),
    ],
    ids=["broyden", "anderson"],
)
def test_charge_strategy_switches_to_fock_without_an_extra_map(
    method: SCFMethod,
) -> None:
    """Charge stages seed CDIIS from the same map observation at the switch."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    strategy = SCFStrategy(
        (
            SCFStage(
                method,
                representation=SCFRepresentation.CHARGE,
                switch_policy=SCFSwitchPolicy(max_stage_iterations=1),
            ),
            SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),
        )
    )
    settings = SCFSolverSettings(1, False, 0.5, 0.0, 0.0)

    outcome = _solve_map(
        case.x,
        settings,
        problem.map,
        problem=problem,
        strategy=strategy,
        eager=True,
    )

    assert int(outcome.strategy_diagnostics.switch_count) == 1
    assert int(outcome.iterations) == 2
    assert outcome.final_problem.representation is SCFRepresentation.FOCK
    torch.testing.assert_close(
        outcome.root_x, outcome.map_result.spectral.hamiltonian
    )
    torch.testing.assert_close(
        outcome.map_result.mapped_x,
        outcome.final_problem.rebuild_hamiltonian(
            outcome.map_result.spectral.charges
        ),
    )


@pytest.mark.parametrize(
    "first_method",
    [
        BroydenMethod(),
        AndersonMethod(AndersonMethodSettings(0.1, 5, True, 0.01)),
    ],
)
@pytest.mark.parametrize(
    "maxiter,stage_counts",
    [(1, (2, 0)), (2, (2, 1))],
    ids=["switch-at-budget-boundary", "one-map-after-switch"],
)
def test_eager_compact_method_to_cdiis_switch_shares_global_budget(
    first_method: SCFMethod,
    maxiter: int,
    stage_counts: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact methods convert to CDIIS without spending another map."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    guess, layout = potential_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.POTENTIAL, layout)
    strategy = SCFStrategy(
        (
            SCFStage(
                first_method,
                representation=SCFRepresentation.POTENTIAL,
                switch_policy=SCFSwitchPolicy(max_stage_iterations=1),
                initial_damp=0.5,
            ),
            SCFStage(
                CDIISMethod(),
                representation=SCFRepresentation.FOCK,
                initial_damp=1.0,
            ),
        )
    )
    settings = SCFSolverSettings(
        maxiter=maxiter,
        mix_guess=True,
        damp=0.5,
        x_atol=0.0,
        x_atol_max=0.0,
        dynamic_damping=DynamicDampingSettings(enabled=True),
    )

    from dxtb._src.scf.functional import problem as problem_module
    from dxtb._src.scf.functional import solver as solver_module

    original_transition = solver_module._method_transition
    original_update_damping = solver_module._update_damping
    original_potential_map = problem_module.potential_map
    original_fock_map = problem_module.fock_map
    cdiis_entries = []
    damping_inputs = []
    damping_updates = 0
    map_calls = 0

    def record_transition(*args, **kwargs):
        method = args[1]
        if isinstance(method, CDIISMethod):
            cdiis_entries.append(
                (args[2].clone(), args[3], args[5], args[6], args[8].clone())
            )
        return original_transition(*args, **kwargs)

    def mark_dynamic_switch(*args, **kwargs):
        nonlocal damping_updates
        damping_inputs.append(args[0])
        result = original_update_damping(*args, **kwargs)
        damping_updates += 1
        if damping_updates == 1:
            return DampingState(result.value, torch.ones_like(result.switched))
        return result

    def record_potential_map(x, inputs, layout):
        nonlocal map_calls
        map_calls += 1
        return original_potential_map(x, inputs, layout)

    def record_fock_map(x, inputs):
        nonlocal map_calls
        map_calls += 1
        return original_fock_map(x, inputs)

    monkeypatch.setattr(solver_module, "_method_transition", record_transition)
    monkeypatch.setattr(solver_module, "_update_damping", mark_dynamic_switch)
    monkeypatch.setattr(problem_module, "potential_map", record_potential_map)
    monkeypatch.setattr(problem_module, "fock_map", record_fock_map)

    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        problem=problem,
        strategy=strategy,
        eager=True,
    )

    assert int(outcome.strategy_diagnostics.switch_count) == 1
    assert int(outcome.strategy_diagnostics.final_stage_index) == 1
    assert int(outcome.strategy_diagnostics.switch_reasons[0]) == int(
        SCFSwitchReason.MAX_STAGE_ITERATIONS
    )
    assert (
        tuple(
            int(value)
            for value in outcome.strategy_diagnostics.stage_iterations
        )
        == stage_counts
    )
    # The initial map and the global candidate budget include no switch map.
    assert int(outcome.iterations) == maxiter + 1
    assert map_calls == maxiter + 1
    assert outcome.final_problem.representation is SCFRepresentation.FOCK
    fock_input, fock_observation, state, step, damp = cdiis_entries[0]
    assert state is None
    assert step == 1
    torch.testing.assert_close(damp, guess.new_tensor(1.0))
    if maxiter == 1:
        torch.testing.assert_close(
            outcome.damping_state.value, guess.new_tensor(1.0)
        )
        assert not bool(outcome.damping_state.switched)
    else:
        # The next candidate transition observes the reset stage damping.
        torch.testing.assert_close(
            damping_inputs[1].value, guess.new_tensor(1.0)
        )
        assert not bool(damping_inputs[1].switched)
    torch.testing.assert_close(
        fock_input, fock_observation.spectral.hamiltonian
    )
    converted_problem = outcome.final_problem
    torch.testing.assert_close(
        fock_observation.mapped_x,
        converted_problem.rebuild_hamiltonian(
            fock_observation.spectral.charges
        ),
    )


def test_broyden_to_cdiis_metric_switch_records_reason() -> None:
    """A configured physical progress threshold drives an eager switch."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    guess, layout = potential_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.POTENTIAL, layout)
    strategy = SCFStrategy(
        (
            SCFStage(
                BroydenMethod(),
                representation=SCFRepresentation.POTENTIAL,
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=1, fixed_point_l2=1.0e20
                ),
            ),
            SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),
        )
    )
    settings = SCFSolverSettings(2, False, 0.5, 0.0, 0.0)

    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        problem=problem,
        strategy=strategy,
        eager=True,
    )

    assert int(outcome.strategy_diagnostics.switch_count) == 1
    assert int(outcome.strategy_diagnostics.switch_reasons[0]) == int(
        SCFSwitchReason.PHYSICAL_THRESHOLD
    )
    switch_metrics = outcome.strategy_diagnostics.switch_metrics[0]
    assert torch.isfinite(switch_metrics.fixed_point_l2)
    assert torch.isfinite(switch_metrics.fixed_point_max)
    assert torch.isfinite(switch_metrics.stationarity)


def test_broyden_to_cdiis_same_fock_representation_switches_method_only() -> (
    None
):
    """A same-representation switch resets method state without conversion."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    guess = fock_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    strategy = SCFStrategy(
        (
            SCFStage(
                BroydenMethod(),
                representation=SCFRepresentation.FOCK,
                switch_policy=SCFSwitchPolicy(max_stage_iterations=1),
            ),
            SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),
        )
    )
    settings = SCFSolverSettings(2, False, 0.5, 0.0, 0.0)

    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        problem=problem,
        strategy=strategy,
        eager=True,
    )

    assert int(outcome.strategy_diagnostics.switch_count) == 1
    assert outcome.final_problem is problem
    assert int(outcome.iterations) == 3


@pytest.mark.parametrize(
    "sample_name,spin,temperature",
    [("H2O", 0.0, 0.0), ("NO2", 1.0, 300.0)],
    ids=["h2o", "no2-finite-temperature"],
)
def test_switched_certified_root_enters_history_free_implicit_boundary(
    sample_name: str, spin: float, temperature: float
) -> None:
    """S2 implicit preparation discards switched method history."""
    from dxtb import labels
    from dxtb._src.scf.functional.implicit import (
        _prepare_scf_root,
        _SCFMapTensorTree,
    )

    case = _prepare_case(
        GFN2_XTB,
        sample_name=sample_name,
        spin=spin,
        scf_options={"fermi_etemp": temperature},
    )
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    potential, layout = potential_initial_guess(charges, case.inputs)
    fock = fock_initial_guess(charges, case.inputs)
    compact_problem = SCFProblem(
        case.inputs, SCFRepresentation.POTENTIAL, layout
    )
    fock_problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    root_tolerance = 1.0e-12 if sample_name == "NO2" else 1.0e-7
    settings = SCFSolverSettings(
        100, False, 0.5, root_tolerance, root_tolerance
    )
    switched_strategy = SCFStrategy(
        (
            SCFStage(
                BroydenMethod(),
                representation=SCFRepresentation.POTENTIAL,
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=3,
                    fixed_point_l2=1.0e-2,
                    stationarity=1.0e-2,
                    max_stage_iterations=15,
                ),
            ),
            SCFStage(
                CDIISMethod(),
                representation=SCFRepresentation.FOCK,
                initial_damp=1.0,
            ),
        )
    )
    switched = _solve_map(
        potential,
        settings,
        compact_problem.map,
        problem=compact_problem,
        strategy=switched_strategy,
        eager=True,
    )
    standalone = _solve_map(
        fock,
        settings,
        fock_problem.map,
        problem=fock_problem,
        strategy=SCFStrategy(
            (SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),)
        ),
        eager=True,
    )

    assert bool(switched.converged) and bool(standalone.converged)
    torch.testing.assert_close(
        switched.root_x, standalone.root_x, rtol=0.0, atol=2.0e-6
    )
    assert switched.final_problem is not None

    adapter, flat_parameters = _SCFMapTensorTree.prepare(
        case.inputs, labels.SCP_MODE_FOCK
    )
    hcore_index = adapter.parameter_paths.index("integrals.hcore")
    mutable_parameters = list(flat_parameters)
    mutable_parameters[hcore_index].requires_grad_(True)
    parameters = tuple(mutable_parameters)

    def prepare(outcome):
        return _prepare_scf_root(
            outcome,
            maxiter=settings.maxiter,
            x_atol=settings.x_atol,
            x_atol_max=settings.x_atol_max,
            final_valid=torch.ones((), dtype=torch.bool),
            raw_map=adapter,
            parameters=parameters,
        )

    prepared_switched = prepare(switched)
    prepared_standalone = prepare(standalone)
    assert not hasattr(prepared_switched, "method_state")
    switched_attached = prepared_switched.attach()
    standalone_attached = prepared_standalone.attach()
    torch.testing.assert_close(switched_attached, switched.root_x)
    torch.testing.assert_close(standalone_attached, standalone.root_x)

    hcore = parameters[hcore_index]
    switched_response = torch.autograd.grad(
        switched_attached.square().sum(),
        hcore,
        create_graph=True,
        retain_graph=True,
    )[0]
    standalone_response = torch.autograd.grad(
        standalone_attached.square().sum(), hcore, create_graph=True
    )[0]
    assert torch.isfinite(switched_response).all()
    torch.testing.assert_close(
        switched_response,
        standalone_response,
        rtol=2.0e-4,
        atol=2.0e-6,
    )
    switched_second = torch.autograd.grad(
        switched_response.square().sum(), hcore, retain_graph=True
    )[0]
    standalone_second = torch.autograd.grad(
        standalone_response.square().sum(), hcore
    )[0]
    assert torch.isfinite(switched_second).all()
    torch.testing.assert_close(
        switched_second,
        standalone_second,
        rtol=3.0e-3,
        atol=2.0e-5,
    )


def test_dynamic_multistage_is_explicitly_fixed_trip_unsupported() -> None:
    """Static transforms cannot silently execute only the first stage."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    strategy = SCFStrategy(
        (
            SCFStage(
                BroydenMethod(),
                representation=SCFRepresentation.CHARGE,
                switch_policy=SCFSwitchPolicy(max_stage_iterations=1),
            ),
            SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),
        )
    )
    settings = SCFSolverSettings(2, False, 0.5, 0.0, 0.0)

    with pytest.raises(NotImplementedError, match="eager-only"):
        _solve_map(
            case.x,
            settings,
            problem.map,
            problem=problem,
            strategy=strategy,
            eager=False,
        )


def test_zero_budget_does_not_enter_a_later_stage() -> None:
    """The compatibility one-shot map cannot trigger a strategy switch."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    strategy = SCFStrategy(
        (
            SCFStage(
                BroydenMethod(),
                representation=SCFRepresentation.CHARGE,
                switch_policy=SCFSwitchPolicy(max_stage_iterations=1),
            ),
            SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),
        )
    )
    settings = SCFSolverSettings(0, False, 0.5, 0.0, 0.0)

    outcome = _solve_map(
        case.x,
        settings,
        problem.map,
        problem=problem,
        strategy=strategy,
        eager=True,
    )

    assert int(outcome.iterations) == 1
    assert not bool(outcome.converged)
    assert int(outcome.strategy_diagnostics.switch_count) == 0
    assert int(outcome.strategy_diagnostics.final_stage_index) == 0
    assert tuple(
        int(value) for value in outcome.strategy_diagnostics.stage_iterations
    ) == (1, 0)


def test_strategy_rejects_missing_exit_policy_and_invalid_final_policy() -> (
    None
):
    """Executable strategies validate every stage transition in advance."""
    method = BroydenMethod()
    with pytest.raises(ValueError, match="exit switch policy"):
        SCFStrategy((SCFStage(method), SCFStage(CDIISMethod())))

    with pytest.raises(ValueError, match="final SCF stage"):
        SCFStrategy(
            (
                SCFStage(
                    method,
                    switch_policy=SCFSwitchPolicy(max_stage_iterations=2),
                ),
                SCFStage(
                    CDIISMethod(),
                    switch_policy=SCFSwitchPolicy(max_stage_iterations=2),
                ),
            )
        )


def test_strategy_rejects_unsupported_fock_to_compact_transition() -> None:
    """The S6 conversion boundary rejects reverse representation changes."""
    strategy = SCFStrategy(
        (
            SCFStage(
                BroydenMethod(),
                representation=SCFRepresentation.FOCK,
                switch_policy=SCFSwitchPolicy(max_stage_iterations=2),
            ),
            SCFStage(
                BroydenMethod(), representation=SCFRepresentation.POTENTIAL
            ),
        )
    )

    with pytest.raises(NotImplementedError, match="compact-to-Fock"):
        strategy.resolve_representations(SCFRepresentation.FOCK)


def test_methods_do_not_own_stage_selection_or_controller_lifecycle() -> None:
    """The strategy controller depends only on the SCFMethod interface."""
    from dxtb._src.scf.functional import anderson, broyden, cdiis, solver

    controller_source = getsource(solver)
    for concrete_name in ("AndersonMethod", "BroydenMethod", "CDIISMethod"):
        assert concrete_name not in controller_source

    for module in (anderson, broyden, cdiis):
        method_source = getsource(module)
        assert "_solve_map_control" not in method_source
        assert "_solve_map_anderson" not in method_source
        assert "SCFSwitchPolicy" not in method_source


@pytest.mark.parametrize(
    "method,representation",
    [
        (LinearMethod(), SCFRepresentation.CHARGE),
        (BroydenMethod(), SCFRepresentation.CHARGE),
        (
            AndersonMethod(AndersonMethodSettings(0.1, 4, True, 0.01)),
            SCFRepresentation.CHARGE,
        ),
        (CDIISMethod(), SCFRepresentation.FOCK),
    ],
    ids=["linear", "broyden", "anderson", "cdiis"],
)
def test_terminal_dummy_trips_keep_method_state_finite_under_derivatives(
    method: SCFMethod, representation: SCFRepresentation
) -> None:
    """Post-terminal dummy maps do not feed NaNs into method algebra."""
    case = _prepare_case(GFN1_XTB, sample_name="H2")
    if representation is SCFRepresentation.CHARGE:
        guess = case.x
    else:
        charges = unpack_charges(case.x, case.inputs.charge_layout)
        guess = fock_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, representation)
    base = guess.detach().clone()
    nan = torch.full_like(guess, torch.nan)

    def map_fn(x: torch.Tensor):
        result = problem.map(x)
        poisoned = torch.any(x != base)
        spectral = replace(
            result.spectral,
            density=torch.where(
                poisoned,
                torch.full_like(result.spectral.density, torch.nan),
                result.spectral.density,
            ),
        )
        mapped_x = torch.where(poisoned, nan, result.mapped_x)
        return replace(result, mapped_x=mapped_x, spectral=spectral)

    settings = SCFSolverSettings(3, False, 0.5, 0.0, 0.0)
    strategy = SCFStrategy((SCFStage(method),))

    outcome = _solve_map(
        guess,
        settings,
        map_fn,
        problem=problem,
        strategy=strategy,
        record_trace=True,
    )
    assert bool(outcome.terminal_failed)
    assert not bool(outcome.converged)
    assert torch.isfinite(outcome.root_x).all()
    assert not torch.isfinite(outcome.map_result.spectral.density).all()
    assert outcome.trace is not None
    for state_tensor in _method_state_tensors(outcome.trace.method_state):
        assert torch.isfinite(state_tensor).all()

    def solve_root(value: torch.Tensor) -> torch.Tensor:
        return _solve_map(
            value,
            settings,
            map_fn,
            problem=problem,
            strategy=strategy,
        ).root_x

    _, tangent = jvp(solve_root, (guess,), (torch.ones_like(guess),))
    jacobian = jacfwd(solve_root)(guess)
    reverse_jacobian = jacrev(solve_root)(guess)
    mapped_roots = vmap(solve_root)(
        torch.stack((guess, guess + guess.new_tensor(1.0e-3)))
    )
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(reverse_jacobian).all()
    assert torch.isfinite(mapped_roots).all()
