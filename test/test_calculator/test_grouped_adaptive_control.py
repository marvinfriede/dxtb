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
Transform-compatible early stopping of single-pass ``vmap`` lane groups.

On CPU, a mapped lane group runs a single ``torch.func.vmap`` call over
``system_singlepoint`` and exits ``FixedTripControl.advance`` once no lane
remains active across untransformed, autograd, and ``torch.func`` transformed
evaluations. Lanes that finish earlier stay frozen via ``torch.where``,
reproducing the full static ``maxiter`` schedule across every Result leaf,
derivative, iteration count, status, warning, and error.
"""

from __future__ import annotations

from collections.abc import Callable
from test.test_calculator.test_grouped_vmap_batching import (
    DEVICE,
    DTYPE,
    H2,
    H2O,
    OH,
    TOL,
    _geometry,
    _model,
    _setup,
)
from test.utils import scf_strategy
from typing import Any

import pytest
import torch
from torch.utils._pytree import keystr, tree_leaves_with_path

from dxtb import GFN1_XTB, GFN2_XTB, OutputHandler, ParamModule, labels
from dxtb._src.calculators.packed_result import evaluate_packed_systems
from dxtb._src.scf import solver as solver_mod
from dxtb._src.scf.core import tensor_leaves
from dxtb._src.scf.problem import SCFProblem
from dxtb._src.scf.solver import FixedTripControl
from dxtb._src.utils import tensors as tensors_mod
from dxtb._src.utils.tensors import disable_functorch, unwrap_functorch_tensor

EXACT: dict[str, Any] = {"rtol": 0.0, "atol": 0.0, "equal_nan": True}
_ORIGINAL_CAN_STOP = solver_mod._can_stop_fixed_trip_early


def _disable_early_stop(_active: torch.Tensor) -> bool:
    """Disable CPU fixed-trip early stopping so static ``maxiter`` executes."""
    return False


def _route(monkeypatch: pytest.MonkeyPatch, static: bool) -> None:
    """Select static ``maxiter`` or default CPU early-stopped fixed-trip mode."""
    monkeypatch.setattr(
        solver_mod,
        "_can_stop_fixed_trip_early",
        _disable_early_stop if static else _ORIGINAL_CAN_STOP,
    )


def _leaves(value: object) -> list[tuple[str, torch.Tensor]]:
    """Extract detached ``(path, tensor)`` leaves from a PyTree."""
    return [
        (keystr(path), leaf.detach())
        for path, leaf in tree_leaves_with_path(value)
        if isinstance(leaf, torch.Tensor)
    ]


def _assert_same_leaves(left: object, right: object) -> None:
    """Assert identical PyTree tensor leaf paths and bit-exact values."""
    left_leaves, right_leaves = _leaves(left), _leaves(right)
    assert [p for p, _ in left_leaves] == [p for p, _ in right_leaves]
    assert left_leaves
    for (path, a), (_, b) in zip(left_leaves, right_leaves):
        torch.testing.assert_close(a, b, msg=path, **EXACT)


def _spy_advance_trips(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record ``out_state.executed_trips`` on each ``FixedTripControl.advance`` call."""
    recorded: list[int] = []
    original = FixedTripControl.advance

    def counting(
        self: FixedTripControl,
        state: Any,
        start: int,
        stop: int,
        observations: Any = None,
    ) -> Any:
        """Forward ``FixedTripControl.advance`` and record ``executed_trips``."""
        out_state, out_obs = original(
            self, state, start, stop, observations=observations
        )
        recorded.append(int(out_state.executed_trips))
        return out_state, out_obs

    monkeypatch.setattr(FixedTripControl, "advance", counting)
    return recorded


def _spy_problem_maps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``SCFProblem.map`` evaluations."""
    calls: list[int] = []
    original = SCFProblem.map

    def counting(self: SCFProblem, x: torch.Tensor) -> Any:
        """Increment call counter and delegate to ``SCFProblem.map``."""
        calls.append(1)
        return original(self, x)

    monkeypatch.setattr(SCFProblem, "map", counting)
    return calls


# Values and derivatives

_CONFIGS: dict[str, tuple[int, dict[str, Any]]] = {
    "gfn1-potential": (labels.GFN1_XTB, {}),
    "gfn1-charge-anderson": (
        labels.GFN1_XTB,
        {"strategy": scf_strategy("anderson", "charge")},
    ),
    "gfn1-0K": (labels.GFN1_XTB, {"fermi_etemp": 0.0}),
    "gfn2-potential": (labels.GFN2_XTB, {}),
    "gfn2-fock-cdiis": (labels.GFN2_XTB, {"strategy": "cdiis"}),
}


def _differentiated_call(
    system: Any,
    numbers: tuple[int, ...],
    spin: tuple[float, ...] | None,
    params: tuple[torch.Tensor, ...] = (),
) -> tuple[Any, tuple[torch.Tensor | None, ...]]:
    """Run a packed mapped call differentiating every input family."""
    positions = tuple(
        _geometry(numbers, seed).requires_grad_(True) for seed in range(3)
    )
    charges = torch.tensor([0.0, 0.3, -0.25], dtype=DTYPE, requires_grad=True)
    field = torch.tensor(
        [1.0e-3, -2.0e-3, 5.0e-4], dtype=DTYPE, requires_grad=True
    )
    field_grads = tuple(
        (2.0e-4 * (lane + 1) * torch.eye(3, dtype=DTYPE)).requires_grad_(True)
        for lane in range(3)
    )
    result = evaluate_packed_systems(
        (system,) * 3,
        positions,
        chrg=charges if spin is None else torch.zeros(3, dtype=DTYPE),
        spin=spin,
        field=field,
        field_grad=field_grads,
    )
    inputs: tuple[torch.Tensor, ...] = (*positions, field, *field_grads)
    if spin is None:
        inputs = (*inputs, charges)
    grads = torch.autograd.grad(
        result.energy.sum(),
        (*inputs, *params),
        allow_unused=True,
        retain_graph=bool(params),
    )
    return result, grads


@pytest.mark.parametrize("config", sorted(_CONFIGS))
def test_adaptive_matches_static_values_and_derivatives(
    monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    """Early-stopped mapped groups match static ``maxiter`` values and gradients."""
    method, settings = _CONFIGS[config]
    par = ParamModule(
        GFN1_XTB if method == labels.GFN1_XTB else GFN2_XTB,
        device=DEVICE,
        dtype=DTYPE,
    )
    for element in ("O", "H"):
        par.set_differentiable("element", element, "gam")
    system = _setup(_model(method, par=par, **settings), H2O)
    trips = _spy_advance_trips(monkeypatch)

    outputs = {}
    for static in (True, False):
        _route(monkeypatch, static)
        outputs[static] = _differentiated_call(
            system,
            H2O,
            None,
            tuple(p for p in par.parameters() if p.requires_grad),
        )

    assert len(trips) == 2
    assert trips[0] == system.config.scf.maxiter
    assert 0 < trips[1] < system.config.scf.maxiter

    static_result, static_grads = outputs[True]
    adaptive_result, adaptive_grads = outputs[False]
    _assert_same_leaves(static_result, adaptive_result)
    nonzero = 0
    for a, b in zip(static_grads, adaptive_grads):
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            torch.testing.assert_close(a, b, **EXACT)
            nonzero += int(bool(a.abs().max() > 0))
    assert nonzero >= 8  # 3 positions, field, 3 field_grads, charges, params


def test_adaptive_open_shell_explicit_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open-shell mapped groups with explicit spin stop early and match static."""
    system = _setup(_model(), OH)
    trips = _spy_advance_trips(monkeypatch)
    outputs = {}
    for static in (True, False):
        _route(monkeypatch, static)
        outputs[static] = _differentiated_call(system, OH, (1.0, 1.0, 1.0))
    assert trips[0] == system.config.scf.maxiter
    assert 0 < trips[1] < system.config.scf.maxiter
    _assert_same_leaves(outputs[True][0], outputs[False][0])
    for a, b in zip(outputs[True][1], outputs[False][1]):
        if a is not None and b is not None:
            torch.testing.assert_close(a, b, **EXACT)


def test_adaptive_derivatives_stay_lane_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-lane energy gradients inside an early-stopped group do not leak across lanes."""
    _route(monkeypatch, False)
    system = _setup(_model(), H2O)
    positions = tuple(
        _geometry(H2O, seed).requires_grad_(True) for seed in range(3)
    )
    result = evaluate_packed_systems((system,) * 3, positions)
    grads = torch.autograd.grad(result.energy[1].sum(), positions)
    assert float(grads[1].abs().max()) > 0.0
    assert torch.count_nonzero(grads[0]) == 0
    assert torch.count_nonzero(grads[2]) == 0


# Stopping and trip counts


def test_group_stops_after_last_active_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapped fixed-trip execution exits immediately after the slowest lane certifies."""
    trips = _spy_advance_trips(monkeypatch)
    _route(monkeypatch, False)
    system = _setup(_model(maxiter=60), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(4))
    result = evaluate_packed_systems((system,) * 4, positions)

    iterations = result.iterations.tolist()
    assert result.converged is not None and bool(result.converged.all())
    assert trips == [max(iterations) - 1]
    assert trips[0] < 60


def test_all_terminal_group_runs_no_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group whose lanes all fail at the initial SCF map executes zero trips."""
    system = _setup(_model(), H2)
    positions = tuple(_geometry(H2, seed) for seed in range(2))
    _route(monkeypatch, True)
    reference = evaluate_packed_systems(
        (system,) * 2, positions, chrg=(-6.5, -6.5)
    )
    _route(monkeypatch, False)
    trips = _spy_advance_trips(monkeypatch)
    adaptive = evaluate_packed_systems(
        (system,) * 2, positions, chrg=(-6.5, -6.5)
    )
    assert trips == [0]
    assert adaptive.converged is not None and not bool(adaptive.converged.any())
    _assert_same_leaves(reference, adaptive)


def test_maxiter_exhaustion_runs_full_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconverged groups execute all ``maxiter`` trips and match static leaves."""
    system = _setup(_model(maxiter=5), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(2))
    _route(monkeypatch, True)
    reference = evaluate_packed_systems((system,) * 2, positions)
    _route(monkeypatch, False)
    trips = _spy_advance_trips(monkeypatch)
    adaptive = evaluate_packed_systems((system,) * 2, positions)
    assert trips == [5]
    assert adaptive.iterations.tolist() == [6, 6]
    assert adaptive.converged is not None and not bool(adaptive.converged.any())
    _assert_same_leaves(reference, adaptive)


# Failure and warning semantics


def _outcome(
    call: Callable[[], object],
) -> tuple[tuple[Any, ...], list[tuple[str, Any]]]:
    """Capture outcome or exception tuple and emitted warnings from ``call``."""
    OutputHandler.clear_warnings()
    outcome: tuple[Any, ...]
    try:
        result = call()
    except Exception as error:  # noqa: BLE001 - compared between routes
        outcome = (
            "raised",
            type(error),
            str(error),
            getattr(error, "source_id", None),
        )
    else:
        outcome = ("ok", result)
    warnings = [(str(w[0]), w[1]) for w in OutputHandler.warnings]
    OutputHandler.clear_warnings()
    return outcome, warnings


_FAILURES: dict[str, tuple[tuple[int, ...], dict[str, Any], Any, Any, bool]] = {
    # name: (numbers, model settings, charges, spins, eager_status_policy)
    "unconverged-warn": (H2O, {"maxiter": 3}, None, None, True),
    "unconverged-force": (
        H2O,
        {"maxiter": 3, "force_convergence": True},
        None,
        None,
        True,
    ),
    "one-fermi-failure": (H2, {}, (0.0, -6.5, 0.0), None, True),
    "two-fermi-failures": (H2, {}, (0.0, -6.5, -6.5), None, True),
    "fermi-failure-status": (H2, {}, (0.0, -6.5, 0.0), None, False),
    "invalid-spin": (H2O, {}, None, (0.0, 1.0, 0.0), True),
    "too-many-electrons-spin": (H2O, {}, None, (0.0, 10.0, 0.0), False),
    "mixed-exhaustion": (H2O, {"maxiter": 8}, None, None, True),
}


@pytest.mark.parametrize("case", sorted(_FAILURES))
def test_failure_semantics_match_static_schedule(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Warnings, exceptions, ``source_id``, and status leaves match static ``maxiter``."""
    numbers, settings, charges, spins, eager = _FAILURES[case]
    system = _setup(_model(**settings), numbers)
    positions = tuple(_geometry(numbers, seed) for seed in range(3))

    def call() -> Any:
        """Invoke ``evaluate_packed_systems`` for the failure fixture."""
        return evaluate_packed_systems(
            (system,) * 3,
            positions,
            chrg=0.0 if charges is None else charges,
            spin=spins,
            eager_status_policy=eager,
        )

    _route(monkeypatch, True)
    static, static_warnings = _outcome(call)
    _route(monkeypatch, False)
    adaptive, adaptive_warnings = _outcome(call)

    assert static[0] == adaptive[0]
    assert static_warnings == adaptive_warnings
    if static[0] == "raised":
        assert static[1:] == adaptive[1:]
    else:
        _assert_same_leaves(static[1], adaptive[1])


def test_mixed_exhaustion_fixture_has_converged_and_unconverged_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``mixed-exhaustion`` fixture spans both certified and unconverged lanes."""
    _route(monkeypatch, False)
    system = _setup(_model(maxiter=8), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))
    OutputHandler.clear_warnings()
    result = evaluate_packed_systems((system,) * 3, positions)
    OutputHandler.clear_warnings()
    assert result.converged is not None
    converged = result.converged.tolist()
    assert any(converged) and not all(converged)


# Routing and function transforms


def test_eager_calls_use_the_adaptive_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both autograd-enabled and ``no_grad`` mapped calls stop early on CPU."""
    trips = _spy_advance_trips(monkeypatch)
    system = _setup(_model(maxiter=60), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(2))
    result = evaluate_packed_systems((system,) * 2, positions)
    with torch.no_grad():
        evaluate_packed_systems((system,) * 2, positions)
    expected_trips = int(result.iterations.max()) - 1
    assert 0 < expected_trips < 60
    assert trips == [expected_trips, expected_trips]


@pytest.mark.parametrize("transform", ["jacrev", "jacfwd", "jvp", "vmap"])
def test_transforms_use_the_static_schedule(
    monkeypatch: pytest.MonkeyPatch, transform: str
) -> None:
    """CPU ``torch.func`` transforms over mapped groups stop early and match static."""
    maxiter = 60
    system = _setup(_model(maxiter=maxiter), H2O)
    base = torch.stack([_geometry(H2O, seed) for seed in range(2)])
    expected = evaluate_packed_systems((system,) * 2, tuple(base.unbind(0)))
    expected_maps = int(expected.iterations.max())

    def energy(stacked: torch.Tensor) -> torch.Tensor:
        """Sum total energy across a 2-lane mapped conformer group."""
        return evaluate_packed_systems(
            (system,) * 2, tuple(stacked.unbind(0))
        ).energy.sum()

    def apply_transform() -> torch.Tensor:
        """Evaluate the selected ``torch.func`` transform on ``energy``."""
        if transform == "jacrev":
            return torch.func.jacrev(energy)(base)
        if transform == "jacfwd":
            return torch.func.jacfwd(energy)(base)
        if transform == "jvp":
            return torch.func.jvp(energy, (base,), (torch.ones_like(base),))[1]
        return torch.func.vmap(energy)(torch.stack([base, base + 0.01]))

    trips = _spy_advance_trips(monkeypatch)
    maps = _spy_problem_maps(monkeypatch)
    _route(monkeypatch, False)
    value = apply_transform()
    assert len(trips) == 1
    assert 0 < trips[0] < maxiter
    assert len(maps) == trips[0] + 1 < maxiter + 1
    if transform != "vmap":
        assert len(maps) == expected_maps

    trips.clear()
    maps.clear()
    _route(monkeypatch, True)
    static_value = apply_transform()
    assert trips == [maxiter]
    assert len(maps) == maxiter + 1
    torch.testing.assert_close(value, static_value, **EXACT)

    _route(monkeypatch, False)
    if transform == "vmap":
        reference = torch.stack(
            [energy(base).detach(), energy(base + 0.01).detach()]
        )
    else:
        stacked = base.clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(energy(stacked), stacked)
        reference = (
            gradient.sum() if transform == "jvp" else gradient
        ).detach()
    torch.testing.assert_close(
        value, reference, rtol=TOL["rtol"], atol=TOL["atol"]
    )


def test_compiling_requires_the_static_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active ``torch.compile`` tracing falls back to all ``maxiter`` trips."""
    maxiter = 45
    system = _setup(_model(maxiter=maxiter), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(2))
    trips = _spy_advance_trips(monkeypatch)

    early = evaluate_packed_systems((system,) * 2, positions)
    assert len(trips) == 1 and 0 < trips[0] < maxiter

    def always_compiling() -> bool:
        """Simulate ``torch.compiler.is_compiling()`` returning ``True``."""
        return True

    trips.clear()
    with monkeypatch.context() as m:
        m.setattr(tensors_mod, "is_compiling", always_compiling)
        compiled = evaluate_packed_systems((system,) * 2, positions)
    assert trips == [maxiter]
    _assert_same_leaves(compiled, early)


def test_forward_mode_dual_tensors_match_static_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager forward-mode dual tensors stop early on CPU and match static ``maxiter``."""
    import torch.autograd.forward_ad as fwAD

    system = _setup(_model(maxiter=45), H2O)
    positions = [_geometry(H2O, seed) for seed in range(2)]
    generator = torch.Generator().manual_seed(3)
    tangents = [
        torch.randn(p.shape, generator=generator, dtype=DTYPE)
        for p in positions
    ]
    trips = _spy_advance_trips(monkeypatch)
    tangent_out = {}
    for static in (True, False):
        _route(monkeypatch, static)
        with fwAD.dual_level():
            dual = tuple(
                fwAD.make_dual(p, t) for p, t in zip(positions, tangents)
            )
            energy = evaluate_packed_systems((system,) * 2, dual).energy
            tangent = fwAD.unpack_dual(energy).tangent
            assert tangent is not None
            tangent_out[static] = tangent.clone()
    assert trips[0] == 45
    assert 0 < trips[1] < 45
    assert float(tangent_out[False].abs().max()) > 0.0
    torch.testing.assert_close(tangent_out[True], tangent_out[False], **EXACT)


# Higher-order ordinary autograd


def test_second_order_autograd_matches_static_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Double backward through early-stopped ``vmap`` matches static ``maxiter``."""
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    par.set_differentiable("element", "O", "gam")
    (gam,) = (p for p in par.parameters() if p.requires_grad)
    system = _setup(_model(labels.GFN1_XTB, par=par, maxiter=30), H2O)
    generator = torch.Generator().manual_seed(11)
    directions = tuple(
        torch.randn((3, 3), generator=generator, dtype=DTYPE) for _ in range(2)
    )
    trips = _spy_advance_trips(monkeypatch)

    outputs = {}
    for static in (True, False):
        _route(monkeypatch, static)
        positions = tuple(
            _geometry(H2O, seed).requires_grad_(True) for seed in range(2)
        )
        result = evaluate_packed_systems((system,) * 2, positions)
        assert result.converged is not None and bool(result.converged.all())
        gradient = torch.autograd.grad(
            result.energy.sum(), positions, create_graph=True
        )
        directional = sum((g * v).sum() for g, v in zip(gradient, directions))
        assert isinstance(directional, torch.Tensor)
        second = torch.autograd.grad(
            directional, (*positions, gam), retain_graph=True
        )
        outputs[static] = tuple(g.detach() for g in (*gradient, *second))

    assert trips[0] == 30
    assert 0 < trips[1] < 30
    for a, b in zip(outputs[True], outputs[False]):
        assert float(a.abs().max()) > 0.0
        torch.testing.assert_close(a, b, **EXACT)


# Lanes finishing at different iterations


def _snapshot_batched_state(state: object) -> dict[str, torch.Tensor]:
    """Unwrap and clone lane-batched tensor leaves of ``_SolverControlState``."""
    ft = torch._C._functorch
    captured: dict[str, torch.Tensor] = {}
    for path, leaf in tensor_leaves(state):
        if not ft.is_batchedtensor(leaf):
            continue
        raw = unwrap_functorch_tensor(leaf, allow_vmap=True)
        assert raw is not None
        with disable_functorch():
            captured[path] = raw.detach().clone()
    return captured


def test_unequal_lane_iterations_match_and_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Earlier-certified lanes stay frozen while remaining lanes continue."""
    system = _setup(_model(maxiter=40), H2O)
    charges = (0.0, 0.4, -0.3)

    def call() -> tuple[Any, tuple[torch.Tensor, ...]]:
        """Evaluate energies and position gradients across heterogeneous lanes."""
        positions = tuple(
            _geometry(H2O, seed).requires_grad_(True) for seed in range(3)
        )
        result = evaluate_packed_systems((system,) * 3, positions, chrg=charges)
        gradient = torch.autograd.grad(result.energy.sum(), positions)
        return result, tuple(g.detach() for g in gradient)

    _route(monkeypatch, True)
    static, static_gradient = call()

    _route(monkeypatch, False)
    snapshots: list[dict[str, torch.Tensor]] = []
    original_start = FixedTripControl.start
    original_step = solver_mod._solver_control_step

    def recording_start(self: FixedTripControl, guess: torch.Tensor) -> Any:
        """Record initial post-``start`` lane state (0 executed trips)."""
        state = original_start(self, guess)
        snapshots.append(_snapshot_batched_state(state))
        return state

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        """Record lane state after each executed fixed-trip pass."""
        state = original_step(*args, **kwargs)
        snapshots.append(_snapshot_batched_state(state))
        return state

    monkeypatch.setattr(FixedTripControl, "start", recording_start)
    monkeypatch.setattr(solver_mod, "_solver_control_step", recording_step)
    adaptive, adaptive_gradient = call()

    iterations = adaptive.iterations.tolist()
    assert bool(adaptive.converged.all())
    assert len(set(iterations)) > 1, iterations
    _assert_same_leaves(static, adaptive)
    for a, b in zip(static_gradient, adaptive_gradient):
        torch.testing.assert_close(a, b, **EXACT)
    assert static.iterations.tolist() == iterations
    assert static.converged.tolist() == adaptive.converged.tolist()

    assert len(snapshots) == max(iterations)
    for lane, count in enumerate(iterations):
        done = count - 1
        frozen_state = snapshots[done]
        assert frozen_state
        for later_state in snapshots[done + 1 :]:
            assert set(later_state) == set(frozen_state)
            for path, frozen_leaf in frozen_state.items():
                torch.testing.assert_close(
                    later_state[path][lane],
                    frozen_leaf[lane],
                    msg=path,
                    **EXACT,
                )
    first, last = min(iterations) - 1, max(iterations) - 1
    late = iterations.index(max(iterations))
    assert any(
        not torch.equal(
            snapshots[first][path][late], snapshots[last][path][late]
        )
        for path in snapshots[first]
    )
