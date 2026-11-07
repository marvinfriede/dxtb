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

import pytest
import torch

from dxtb import labels
from dxtb._src.calculators.singlepoint import system_singlepoint
from dxtb._src.scf.functional import eager as eager_mod
from dxtb._src.scf.functional.core import tensor_leaves
from dxtb._src.scf.functional.solver import (
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

# (method, settings): production GFN1 charge/potential and GFN2 Fock, every
# fixed-trip capable method family.
_CONFIGS = {
    "gfn1-potential-default": (labels.GFN1_XTB, {}),
    "gfn1-charge-anderson": (
        labels.GFN1_XTB,
        {"mixer": "anderson", "scp_mode": "charge"},
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


@pytest.mark.parametrize("physics", sorted(_PHYSICS))
@pytest.mark.parametrize("config", sorted(_CONFIGS))
@pytest.mark.parametrize("maxiter", [0, 1, 2, 12, 25])
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
