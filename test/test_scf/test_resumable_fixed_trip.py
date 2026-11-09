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
"""
Resumable fixed-trip control.

Advancing the fixed-trip controller through consecutive trip ranges must be
the same computation as one call over ``range(maxiter)``: identical solver
state (including method history, damping, masks and certification) and an
identical finalized result, bit for bit.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields, replace

import pytest
import torch

from dxtb import labels
from dxtb._src.calculators.singlepoint import system_singlepoint
from dxtb._src.scf import eager as eager_mod
from dxtb._src.scf import solver as solver_mod
from dxtb._src.scf.core import tensor_leaves
from dxtb._src.scf.problem import SCFVariable
from dxtb._src.scf.solver import (
    SCFSolverTrace,
    _finalize_problem_outcome,
    _problem_state_projection,
    finalize_fixed_trip_problem,
    fixed_trip_problem_control,
    solve_scf_problem,
)
from test.test_calculator.test_grouped_vmap_batching import (
    H2,
    H2O,
    _geometry,
    _model,
    _setup,
)

from ..utils import scf_strategy

# (method, settings): production GFN1 charge/potential and GFN2 Fock, every
# fixed-trip capable method family.
_CONFIGS = {
    "gfn1-potential-default": (labels.GFN1_XTB, {}),
    "gfn1-charge-anderson": (
        labels.GFN1_XTB,
        {"strategy": scf_strategy("anderson", "charge")},
    ),
    "gfn1-potential-broyden": (labels.GFN1_XTB, {"strategy": "broyden"}),
    "gfn1-potential-linear": (labels.GFN1_XTB, {"strategy": "linear"}),
    "gfn2-fock-cdiis": (labels.GFN2_XTB, {"strategy": "cdiis"}),
    "gfn2-potential-default": (labels.GFN2_XTB, {}),
}

# (numbers, charge, spin, settings)
_PHYSICS = {
    "closed-shell-300K": (H2O, 0.0, None, {}),
    "closed-shell-0K": (H2O, 0.0, None, {"fermi_etemp": 0.0}),
    "fractional-charge": (H2O, 0.3, None, {}),
    "explicit-spin": (H2O, 1.0, 1.0, {}),
    "fermi-failure": (H2, -6.5, None, {}),
}


def _partitions(maxiter: int) -> Iterator[tuple[int, ...]]:
    yield (maxiter,)
    yield (0, maxiter)
    if maxiter > 1:
        yield (1,) * maxiter
    if maxiter > 4:
        yield (4, maxiter - 4)
    if maxiter > 6:
        yield (3, 3, maxiter - 6)


def _capture_problem(monkeypatch, method, numbers, charge, spin, settings):
    """Run the System fixed-trip route once and capture its SCF problem."""
    calls = []

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return solve_scf_problem(*args, **kwargs)

    monkeypatch.setattr(eager_mod, "solve_scf_problem", recording)
    system = _setup(_model(method, **settings), numbers)
    system_singlepoint(
        system,
        _geometry(numbers, 3),
        torch.tensor(charge, dtype=torch.float64),
        None if spin is None else torch.tensor(spin, dtype=torch.float64),
    )
    monkeypatch.undo()
    ((args, kwargs),) = calls
    assert kwargs["eager"] is False
    guess, problem, solver_settings, strategy = args
    return guess, problem, solver_settings, strategy, kwargs


def _assert_identical(left, right) -> None:
    left_leaves = tensor_leaves(left)
    right_leaves = tensor_leaves(right)
    assert [path for path, _ in left_leaves] == [
        path for path, _ in right_leaves
    ]
    assert left_leaves, "nothing compared"
    for (path, a), (_, b) in zip(left_leaves, right_leaves):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        torch.testing.assert_close(
            a, b, rtol=0.0, atol=0.0, equal_nan=True, msg=path
        )


def _chunked(control, guess, chunks):
    state = control.start(guess)
    trip = 0
    for chunk in chunks:
        state, _ = control.advance(state, trip, trip + chunk)
        trip += chunk
    assert trip == control.maxiter
    return state


def _chunked_cases() -> list:
    """All (config, physics, maxiter) cases; the long ones run nightly.

    The boundary budgets 0, 1 and 2 run for every case. Of the longer
    budgets, the routine suite keeps ``maxiter=12`` for every config on the
    closed-shell case and for every physics case on the default config;
    ``maxiter=25`` and the remaining ``maxiter=12`` cases are marked `slow`.
    """
    cases = []
    for maxiter in (0, 1, 2, 12, 25):
        for config in sorted(_CONFIGS):
            for physics in sorted(_PHYSICS):
                routine = maxiter <= 2 or (
                    maxiter == 12
                    and (
                        physics == "closed-shell-300K"
                        or config == "gfn1-potential-default"
                    )
                )
                cases.append(
                    pytest.param(
                        config,
                        physics,
                        maxiter,
                        marks=() if routine else pytest.mark.slow,
                        id=f"{maxiter}-{config}-{physics}",
                    )
                )
    return cases


@pytest.mark.parametrize("config,physics,maxiter", _chunked_cases())
def test_chunked_advance_matches_single_range(
    monkeypatch, config: str, physics: str, maxiter: int
) -> None:
    method, method_settings = _CONFIGS[config]
    numbers, charge, spin, physics_settings = _PHYSICS[physics]
    guess, problem, settings, strategy, kwargs = _capture_problem(
        monkeypatch,
        method,
        numbers,
        charge,
        spin,
        {**method_settings, **physics_settings, "maxiter": maxiter},
    )
    profile = kwargs["certification_profile"]
    reference = solve_scf_problem(
        guess,
        problem,
        settings,
        strategy,
        eager=False,
        certification_profile=profile,
    )

    control = fixed_trip_problem_control(
        guess, problem, settings, strategy, certification_profile=profile
    )
    single = _chunked(control, guess, (maxiter,))
    _assert_identical(
        finalize_fixed_trip_problem(control, single, guess), reference
    )

    for chunks in _partitions(maxiter):
        state = _chunked(control, guess, chunks)
        _assert_identical(state, single)
        _assert_identical(
            finalize_fixed_trip_problem(control, state, guess), reference
        )


def test_fixture_covers_terminal_failure_and_exhaustion(monkeypatch) -> None:
    """The physics fixtures reach the intended controller outcomes."""
    outcomes = {}
    for physics, maxiter in (("fermi-failure", 12), ("closed-shell-300K", 2)):
        numbers, charge, spin, physics_settings = _PHYSICS[physics]
        guess, problem, settings, strategy, kwargs = _capture_problem(
            monkeypatch,
            labels.GFN1_XTB,
            numbers,
            charge,
            spin,
            {**physics_settings, "maxiter": maxiter},
        )
        control = fixed_trip_problem_control(
            guess,
            problem,
            settings,
            strategy,
            certification_profile=kwargs["certification_profile"],
        )
        outcomes[physics] = _chunked(control, guess, (maxiter,))

    failed = outcomes["fermi-failure"]
    assert bool(failed.terminal_failed) and not bool(failed.certified)
    exhausted = outcomes["closed-shell-300K"]
    assert bool(exhausted.active) and not bool(exhausted.certified)
    assert int(exhausted.iterations) == 3


def test_advance_rejects_ranges_outside_the_schedule(monkeypatch) -> None:
    guess, problem, settings, strategy, _ = _capture_problem(
        monkeypatch, labels.GFN1_XTB, H2O, 0.0, None, {"maxiter": 4}
    )
    control = fixed_trip_problem_control(guess, problem, settings, strategy)
    state = control.start(guess)
    for start, stop in ((0, 5), (3, 2), (-1, 1)):
        with pytest.raises(ValueError, match="Invalid fixed-trip range"):
            control.advance(state, start, stop)


# Solver trace


def _initial_observations(state) -> tuple[tuple[object, ...], ...]:
    """Initial trace row, as recorded by ``_solve_map_control``."""
    return (
        (
            state.map_input_x,
            state.map_result,
            state.current_x,
            state.residual,
            state.active,
            state.active,
            state.iterations,
            state.damping_state.value,
            state.converger_state,
        ),
    )


def _traced(control, guess, chunks):
    """Chunked solve that carries trace observations between chunks."""
    state = control.start(guess)
    observations = _initial_observations(state)
    trip = 0
    for chunk in chunks:
        state, observations = control.advance(
            state, trip, trip + chunk, observations
        )
        trip += chunk
    assert trip == control.maxiter
    return _finalize_problem_outcome(
        control.finish(state, guess, observations), control.problem
    )


def _assert_same_trace(left: SCFSolverTrace, right: SCFSolverTrace) -> None:
    for item in fields(SCFSolverTrace):
        a, b = getattr(left, item.name), getattr(right, item.name)
        assert (a is None) == (b is None), item.name
        if isinstance(a, tuple):
            assert len(a) == len(b), item.name
            pairs = list(zip(a, b))
        else:
            pairs = [] if a is None else [(a, b)]
        for x, y in pairs:
            assert x.dtype == y.dtype and x.shape == y.shape, item.name
            torch.testing.assert_close(
                x, y, rtol=0.0, atol=0.0, equal_nan=True, msg=item.name
            )


_TRACE_CONFIGS = (
    "gfn1-potential-default",
    "gfn1-charge-anderson",
    "gfn1-potential-broyden",
    "gfn2-fock-cdiis",
)
_TRACE_PARTITIONS = ((12,), (4, 8), (3, 3, 6), (1,) * 12)


@pytest.mark.parametrize("physics", ["closed-shell-300K", "fermi-failure"])
@pytest.mark.parametrize("config", _TRACE_CONFIGS)
def test_chunked_trace_matches_monolithic_trace(
    monkeypatch, config: str, physics: str
) -> None:
    """``record_trace=True``: observations carried between chunks."""
    method, method_settings = _CONFIGS[config]
    numbers, charge, spin, physics_settings = _PHYSICS[physics]
    guess, problem, settings, strategy, kwargs = _capture_problem(
        monkeypatch,
        method,
        numbers,
        charge,
        spin,
        {**method_settings, **physics_settings, "maxiter": 12},
    )
    profile = kwargs["certification_profile"]
    reference = solve_scf_problem(
        guess,
        problem,
        settings,
        strategy,
        eager=False,
        record_trace=True,
        certification_profile=profile,
    )
    trace = reference.trace
    assert trace is not None
    assert trace.map_inputs.shape[0] == 1 + 12
    if config != "gfn1-potential-default":
        assert trace.converger_state, "stateful method records no history"

    control = fixed_trip_problem_control(
        guess, problem, settings, strategy, certification_profile=profile
    )
    for chunks in _TRACE_PARTITIONS:
        chunked = _traced(control, guess, chunks)
        assert chunked.trace is not None
        _assert_same_trace(chunked.trace, trace)
        _assert_identical(chunked, reference)


# Initial status and the maxiter = 0 / 1 boundaries

# (physics, maxiter, record_initial_status) -> pinned outcome:
# (trajectory status recorded, iterations, converged, terminal failure)
_BOUNDARY_OUTCOMES = {
    ("closed-shell-300K", 0, None): (False, 1, False, False),
    ("closed-shell-300K", 0, True): (True, 1, False, False),
    ("closed-shell-300K", 0, False): (False, 1, False, False),
    ("closed-shell-300K", 1, None): (True, 2, False, False),
    ("closed-shell-300K", 1, True): (True, 2, False, False),
    ("closed-shell-300K", 1, False): (True, 2, False, False),
    ("fermi-failure", 0, None): (False, 1, False, True),
    ("fermi-failure", 0, True): (True, 1, False, True),
    ("fermi-failure", 0, False): (False, 1, False, True),
    ("fermi-failure", 1, None): (True, 1, False, True),
    ("fermi-failure", 1, True): (True, 1, False, True),
    ("fermi-failure", 1, False): (True, 1, False, True),
}


@pytest.mark.parametrize(
    "physics, maxiter, record", sorted(_BOUNDARY_OUTCOMES, key=str)
)
def test_initial_status_and_short_schedules(
    monkeypatch, physics: str, maxiter: int, record: bool | None
) -> None:
    numbers, charge, spin, physics_settings = _PHYSICS[physics]
    guess, problem, settings, strategy, kwargs = _capture_problem(
        monkeypatch,
        labels.GFN1_XTB,
        numbers,
        charge,
        spin,
        {**physics_settings, "maxiter": maxiter},
    )
    profile = kwargs["certification_profile"]
    reference = solve_scf_problem(
        guess,
        problem,
        settings,
        strategy,
        eager=False,
        record_initial_status=record,
        certification_profile=profile,
    )

    control = fixed_trip_problem_control(
        guess, problem, settings, strategy, certification_profile=profile
    )
    # ``None`` resolves to recording the initial status iff trips follow.
    assert control.record_initial_status is (maxiter > 0)
    if record is not None:
        control = replace(control, record_initial_status=record)
    for chunks in ((maxiter,), (0, maxiter), (maxiter, 0)):
        state = _chunked(control, guess, chunks)
        _assert_identical(
            finalize_fixed_trip_problem(control, state, guess), reference
        )

    recorded, iterations, converged, terminal = _BOUNDARY_OUTCOMES[
        (physics, maxiter, record)
    ]
    assert (reference.trajectory_status is not None) is recorded
    assert int(reference.iterations) == iterations
    assert bool(reference.converged) is converged
    assert bool(state.terminal_failed) is terminal
    assert bool(state.active) is (not terminal and not converged)


# Variable-owned state projection (rebase contract)

_REPRESENTATION_CONFIGS = {
    SCFVariable.CHARGE: "gfn1-charge-anderson",
    SCFVariable.POTENTIAL: "gfn1-potential-default",
    SCFVariable.FOCK: "gfn2-fock-cdiis",
}


def _spy_projections(monkeypatch, events: list[str]) -> None:
    for name, label in (
        ("_project_charge_padding", "charge"),
        ("_project_potential_padding", "potential"),
    ):
        original = getattr(solver_mod, name)

        def spy(value, layout, original=original, label=label):
            events.append(label)
            return original(value, layout)

        monkeypatch.setattr(solver_mod, name, spy)


@pytest.mark.parametrize(
    "variable", list(_REPRESENTATION_CONFIGS), ids=lambda r: r.name
)
def test_problem_state_projection_contract(
    monkeypatch, variable: SCFVariable
) -> None:
    method, settings = _CONFIGS[_REPRESENTATION_CONFIGS[variable]]
    guess, problem, solver_settings, strategy, kwargs = _capture_problem(
        monkeypatch, method, H2O, 0.0, None, {**settings, "maxiter": 3}
    )
    assert problem.variable is variable
    expected = {
        SCFVariable.CHARGE: "charge",
        SCFVariable.POTENTIAL: "potential",
        SCFVariable.FOCK: None,
    }[variable]

    events: list[str] = []
    _spy_projections(monkeypatch, events)

    def caller(value):
        events.append("caller")
        return value

    if expected is None:
        assert _problem_state_projection(problem, None) is None
        assert _problem_state_projection(problem, caller) is caller
    else:
        projection = _problem_state_projection(problem, None)
        assert projection is not None
        projection(guess)
        assert events == [expected]
        events.clear()
        combined = _problem_state_projection(problem, caller)
        assert combined is not None
        combined(guess)
        assert events == [expected, "caller"]
        events.clear()

    # Through the monolithic and the resumable fixed-trip controllers.
    profile = kwargs["certification_profile"]
    reference = solve_scf_problem(
        guess,
        problem,
        solver_settings,
        strategy,
        eager=False,
        certification_profile=profile,
    )
    monolithic = list(events)
    events.clear()
    control = fixed_trip_problem_control(
        guess, problem, solver_settings, strategy, certification_profile=profile
    )
    state = _chunked(control, guess, (1, 2))
    _assert_identical(
        finalize_fixed_trip_problem(control, state, guess), reference
    )
    assert events == monolithic
    if expected is None:
        assert monolithic == []
    else:
        assert monolithic and set(monolithic) == {expected}
