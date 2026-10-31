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
"""Characterize Dynamo/eager compilation of SCF maps and fixed-trip control."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from test.test_scf.test_functional_root_derivatives import _case, _initial_state
from test.test_scf.test_functional_solver import _map_result

import torch

from dxtb import GFN1_XTB, labels
from dxtb._src.scf.functional.anderson import (
    AndersonMethod,
    AndersonMethodSettings,
)
from dxtb._src.scf.functional.broyden import BroydenMethod
from dxtb._src.scf.functional.cdiis import CDIISMethod
from dxtb._src.scf.functional.method import LinearMethod, SCFMethodContext
from dxtb._src.scf.functional.problem import SCFProblem, SCFRepresentation
from dxtb._src.scf.functional.solver import SCFSolverSettings, _solve_map

_REPRESENTATIONS = {
    labels.SCP_MODE_CHARGE: SCFRepresentation.CHARGE,
    labels.SCP_MODE_POTENTIAL: SCFRepresentation.POTENTIAL,
    labels.SCP_MODE_FOCK: SCFRepresentation.FOCK,
}


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def _compile_map(case_name: str, mode: int) -> dict[str, object]:
    """Compile one physical map with the Dynamo/eager backend."""
    case = _case(
        case_name,
        method=GFN1_XTB,
        representation=mode,
        mixer=labels.MIXER_LINEAR,
        maxiter=8,
        tolerance=1.0e-10,
    )
    guess, layout = _initial_state(case, case.inputs)
    problem = SCFProblem(
        case.inputs,
        _REPRESENTATIONS[mode],
        potential_layout=layout,
    )

    def raw_map(value: torch.Tensor) -> torch.Tensor:
        return problem.map(value).mapped_x

    expected = raw_map(guess)
    started = time.perf_counter()
    explanation = torch._dynamo.explain(raw_map)(guess)
    explain_seconds = time.perf_counter() - started
    compiled = torch.compile(raw_map, backend="eager")
    started = time.perf_counter()
    actual = compiled(guess)
    first_call_seconds = time.perf_counter() - started
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-12)
    started = time.perf_counter()
    shifted = guess + 1.0e-8
    repeated = compiled(shifted)
    steady_call_seconds = time.perf_counter() - started
    expected_shifted = raw_map(shifted)
    torch.testing.assert_close(
        repeated,
        expected_shifted,
        rtol=0.0,
        atol=1.0e-12,
    )
    leaf = guess.detach().clone().requires_grad_(True)
    compiled_reverse = torch.autograd.grad(compiled(leaf).sum(), leaf)[0]
    expected_reverse = torch.func.grad(lambda value: raw_map(value).sum())(
        guess
    )
    torch.testing.assert_close(
        compiled_reverse, expected_reverse, rtol=2.0e-10, atol=2.0e-12
    )
    direction = torch.ones_like(guess)
    _, compiled_jvp = torch.func.jvp(compiled, (guess,), (direction,))
    _, expected_jvp = torch.func.jvp(raw_map, (guess,), (direction,))
    torch.testing.assert_close(
        compiled_jvp, expected_jvp, rtol=2.0e-10, atol=2.0e-12
    )
    return {
        "case": case_name,
        "method": "GFN1-xTB",
        "representation": _REPRESENTATIONS[mode].value,
        "shape": list(guess.shape),
        "backend": "eager",
        "fullgraph": False,
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "explain_seconds": explain_seconds,
        "first_call_seconds": first_call_seconds,
        "steady_call_seconds": steady_call_seconds,
        "max_value_error": float((actual - expected).abs().max()),
        "repeat_value_error": float((repeated - expected_shifted).abs().max()),
        "reverse_supported": True,
        "reverse_max_error": float(
            (compiled_reverse - expected_reverse).abs().max()
        ),
        "jvp_supported": True,
        "jvp_max_error": float((compiled_jvp - expected_jvp).abs().max()),
    }


def _compile_fixed_trip(budget: int) -> dict[str, object]:
    """Compile a tensor-only affine fixed-trip control with static budget."""
    settings = SCFSolverSettings(
        maxiter=budget,
        mix_guess=True,
        damp=0.4,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    slope = torch.tensor(0.3, dtype=torch.float64)
    offset = torch.tensor(0.7, dtype=torch.float64)

    def fixed_trip(initial: torch.Tensor) -> tuple[torch.Tensor, ...]:
        def affine(value: torch.Tensor):
            return _map_result(slope * value + offset, value)

        outcome = _solve_map(initial, settings, affine)
        return (
            outcome.root_x,
            outcome.iterations,
            outcome.residual,
            outcome.converged,
        )

    initial = torch.tensor([0.15], dtype=torch.float64)
    expected = fixed_trip(initial)
    started = time.perf_counter()
    explanation = torch._dynamo.explain(fixed_trip)(initial)
    explain_seconds = time.perf_counter() - started
    compiled = torch.compile(fixed_trip, backend="eager")
    started = time.perf_counter()
    actual = compiled(initial)
    first_call_seconds = time.perf_counter() - started
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
    assert torch.equal(actual[3], expected[3])
    started = time.perf_counter()
    compiled(initial + 1.0e-4)
    steady_call_seconds = time.perf_counter() - started
    return {
        "budget": budget,
        "backend": "eager",
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "explain_seconds": explain_seconds,
        "first_call_seconds": first_call_seconds,
        "steady_call_seconds": steady_call_seconds,
        "value_parity": True,
    }


def _compile_physical_solver(name: str, budget: int) -> dict[str, object]:
    """Attempt Dynamo/eager compilation of a physical fixed-trip solve."""
    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_FOCK,
        mixer=labels.MIXER_LINEAR,
        maxiter=budget,
        tolerance=0.0,
    )
    initial, _ = _initial_state(case, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    settings = SCFSolverSettings(
        maxiter=budget,
        mix_guess=True,
        damp=0.4,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    method_factories = {
        "Linear": LinearMethod,
        "Anderson": lambda: AndersonMethod(
            AndersonMethodSettings(
                damp_init=0.3,
                generations=2,
                soft_start=False,
                diagonal_offset=None,
            )
        ),
        "Broyden": BroydenMethod,
        "CDIIS": CDIISMethod,
    }
    method = method_factories[name]()

    def physical_fixed_trip(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outcome = _solve_map(
            value,
            settings,
            problem.map,
            problem=problem,
            method=method,
        )
        return (
            outcome.root_x,
            outcome.iterations,
            outcome.residual,
            outcome.converged,
        )

    result: dict[str, object] = {
        "method": name,
        "budget": budget,
        "backend": "eager",
    }
    try:
        expected = physical_fixed_trip(initial)
        started = time.perf_counter()
        explanation = torch._dynamo.explain(physical_fixed_trip)(initial)
        result["explain_seconds"] = time.perf_counter() - started
        result["graph_count"] = explanation.graph_count
        result["graph_break_count"] = explanation.graph_break_count
        compiled = torch.compile(physical_fixed_trip, backend="eager")
        started = time.perf_counter()
        actual = compiled(initial)
        result["first_call_seconds"] = time.perf_counter() - started
        torch.testing.assert_close(
            actual[0], expected[0], rtol=0.0, atol=1.0e-12
        )
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actual[2], expected[2], rtol=0.0, atol=1.0e-12
        )
        assert torch.equal(actual[3], expected[3])
        started = time.perf_counter()
        compiled(initial + 1.0e-8)
        result["steady_call_seconds"] = time.perf_counter() - started
        result["value_parity"] = True
    except Exception as exc:  # benchmark records unsupported routes
        result["supported"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["supported"] = True
    return result


def _compile_transition(name: str) -> dict[str, object]:
    """Compile a fixed-shape Fock method transition with method history."""
    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_FOCK,
        mixer=labels.MIXER_LINEAR,
        maxiter=4,
        tolerance=1.0e-10,
    )
    guess, _ = _initial_state(case, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    physical = problem.evaluate(guess)
    methods = {
        "Linear": LinearMethod,
        "Anderson": lambda: AndersonMethod(
            AndersonMethodSettings(
                damp_init=0.3,
                generations=2,
                soft_start=False,
                diagonal_offset=None,
            )
        ),
        "Broyden": BroydenMethod,
        "CDIIS": CDIISMethod,
    }
    method = methods[name]()
    active = torch.ones((), dtype=torch.bool)
    damp = guess.new_tensor(0.5)
    first_context = SCFMethodContext(
        step=1,
        transition_active=active,
        damp=damp,
        proposal_x=physical.map_result.mapped_x,
    )
    state = method.propose(problem, physical, None, first_context).state

    def transition(proposal_x: torch.Tensor):
        context = SCFMethodContext(
            step=2,
            transition_active=active,
            damp=damp,
            proposal_x=proposal_x,
        )
        proposal = method.propose(problem, physical, state, context)
        return proposal.x, proposal.state

    raw_proposal = physical.map_result.mapped_x + 1.0e-5
    expected = transition(raw_proposal)
    started = time.perf_counter()
    explanation = torch._dynamo.explain(transition)(raw_proposal)
    explain_seconds = time.perf_counter() - started
    compiled = torch.compile(transition, backend="eager")
    started = time.perf_counter()
    actual = compiled(raw_proposal)
    first_call_seconds = time.perf_counter() - started
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=1.0e-12)
    if state is not None:
        actual_leaves, actual_tree = torch.utils._pytree.tree_flatten(actual[1])
        expected_leaves, expected_tree = torch.utils._pytree.tree_flatten(
            expected[1]
        )
        assert actual_tree == expected_tree
        assert len(actual_leaves) == len(expected_leaves)
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
            torch.testing.assert_close(
                actual_leaf, expected_leaf, rtol=0.0, atol=1.0e-12
            )
    return {
        "method": name,
        "representation": "fock",
        "backend": "eager",
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "explain_seconds": explain_seconds,
        "first_call_seconds": first_call_seconds,
        "value_parity": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="H2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    maps = [
        _compile_map(args.case, mode)
        for mode in (
            labels.SCP_MODE_CHARGE,
            labels.SCP_MODE_POTENTIAL,
            labels.SCP_MODE_FOCK,
        )
    ]
    fixed_trip = [_compile_fixed_trip(budget) for budget in (4, 8, 16)]
    physical_fixed_trip = [
        _compile_physical_solver("Linear", budget) for budget in (4, 8, 16)
    ]
    physical_method_smokes = [
        _compile_physical_solver(name, 4)
        for name in ("Anderson", "Broyden", "CDIIS")
    ]
    transitions = [
        _compile_transition(name)
        for name in ("Linear", "Anderson", "Broyden", "CDIIS")
    ]
    result = {
        "source_sha": _git_value("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git_value("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "intraop_threads": torch.get_num_threads(),
        "interoperability_threads": torch.get_num_interop_threads(),
        "compile_backend": "eager",
        "compile_scope": "Dynamo/eager graph-break and value characterization; not optimized compilation",
        "raw_maps": maps,
        "transitions": transitions,
        "fixed_trip_affine": fixed_trip,
        "physical_fixed_trip": physical_fixed_trip,
        "physical_method_smokes": physical_method_smokes,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
