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
Adaptive (chunked) execution of mapped lane groups.

Outside of ``torch.func`` transforms and ``torch.compile``, a mapped group
runs the fixed-trip controller in chunks of trips and stops once no lane is
active. Lanes that finish earlier are frozen by the shared transition, so the
adaptive route must reproduce the full static schedule exactly: every Result
leaf, derivatives, iteration counts, status, warnings and errors.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.utils._pytree import keystr, tree_leaves_with_path

from dxtb import GFN1_XTB, GFN2_XTB, OutputHandler, ParamModule, labels
from dxtb._src.calculators import lane_evaluation
from dxtb._src.calculators.packed_result import evaluate_packed_systems
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

EXACT = {"rtol": 0.0, "atol": 0.0, "equal_nan": True}


def _route(monkeypatch: pytest.MonkeyPatch, static: bool) -> None:
    monkeypatch.setattr(
        lane_evaluation, "_requires_static_schedule", lambda: static
    )


def _leaves(value) -> list[tuple[str, torch.Tensor]]:
    return [
        (keystr(path), leaf.detach())
        for path, leaf in tree_leaves_with_path(value)
        if isinstance(leaf, torch.Tensor)
    ]


def _assert_same_leaves(left, right) -> None:
    left_leaves, right_leaves = _leaves(left), _leaves(right)
    assert [p for p, _ in left_leaves] == [p for p, _ in right_leaves]
    assert left_leaves
    for (path, a), (_, b) in zip(left_leaves, right_leaves):
        torch.testing.assert_close(a, b, msg=path, **EXACT)


def _count_advances(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    original = lane_evaluation._group_advance

    def counting(capture, start, stop, fixed, state):
        ranges.append((start, stop))
        return original(capture, start, stop, fixed, state)

    monkeypatch.setattr(lane_evaluation, "_group_advance", counting)
    return ranges


# Values and derivatives

_CONFIGS = {
    "gfn1-potential": (labels.GFN1_XTB, {}),
    "gfn1-charge-anderson": (
        labels.GFN1_XTB,
        {"mixer": "anderson", "scp_mode": "charge"},
    ),
    "gfn1-0K": (labels.GFN1_XTB, {"fermi_etemp": 0.0}),
    "gfn2-potential": (labels.GFN2_XTB, {}),
    "gfn2-fock-cdiis": (labels.GFN2_XTB, {"strategy": "cdiis"}),
}


def _differentiated_call(system, numbers, spin, params=()):
    """Packed call with every differentiable input family."""
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
    inputs = (*positions, field, *field_grads)
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
    method, settings = _CONFIGS[config]
    par = ParamModule(
        GFN1_XTB if method == labels.GFN1_XTB else GFN2_XTB,
        device=DEVICE,
        dtype=DTYPE,
    )
    for element in ("O", "H"):
        par.set_differentiable("element", element, "gam")
    system = _setup(_model(method, par=par, **settings), H2O)

    outputs = {}
    for static in (True, False):
        _route(monkeypatch, static)
        outputs[static] = _differentiated_call(
            system,
            H2O,
            None,
            tuple(p for p in par.parameters() if p.requires_grad),
        )

    static_result, static_grads = outputs[True]
    adaptive_result, adaptive_grads = outputs[False]
    _assert_same_leaves(static_result, adaptive_result)
    nonzero = 0
    for a, b in zip(static_grads, adaptive_grads):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, **EXACT)
            nonzero += int(bool(a.abs().max() > 0))
    assert nonzero >= 8  # 3 positions, field, 3 field_grads, charges, params


def test_adaptive_open_shell_explicit_spin(monkeypatch) -> None:
    system = _setup(_model(), OH)
    outputs = {}
    for static in (True, False):
        _route(monkeypatch, static)
        outputs[static] = _differentiated_call(system, OH, (1.0, 1.0, 1.0))
    _assert_same_leaves(outputs[True][0], outputs[False][0])
    for a, b in zip(outputs[True][1], outputs[False][1]):
        if a is not None:
            torch.testing.assert_close(a, b, **EXACT)


def test_adaptive_derivatives_stay_lane_local(monkeypatch) -> None:
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


# Stopping, chunking and map counts


def test_group_stops_after_last_active_lane(monkeypatch) -> None:
    """Map trips end with the chunk in which the slowest lane finished."""
    ranges = _count_advances(monkeypatch)
    _route(monkeypatch, False)
    system = _setup(_model(maxiter=60), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(4))
    result = evaluate_packed_systems((system,) * 4, positions)

    iterations = result.iterations.tolist()
    assert bool(result.converged.all())
    chunk = lane_evaluation._ADAPTIVE_CHUNK_TRIPS
    trips = ranges[-1][1]
    # iterations = 1 (initial map) + active trips; the last lane certifies
    # inside the final chunk.
    assert trips - chunk < max(iterations) - 1 <= trips < 60
    assert ranges == [(i, i + chunk) for i in range(0, trips, chunk)]


@pytest.mark.parametrize("chunk", [1, 3, 16, 1000])
def test_chunk_length_does_not_change_results(monkeypatch, chunk) -> None:
    system = _setup(_model(maxiter=25), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))
    charges = (0.0, 0.4, -0.3)

    _route(monkeypatch, True)
    reference = evaluate_packed_systems((system,) * 3, positions, chrg=charges)
    _route(monkeypatch, False)
    monkeypatch.setattr(lane_evaluation, "_ADAPTIVE_CHUNK_TRIPS", chunk)
    ranges = _count_advances(monkeypatch)
    adaptive = evaluate_packed_systems((system,) * 3, positions, chrg=charges)
    _assert_same_leaves(reference, adaptive)
    assert ranges[-1][1] <= 25
    assert all(stop - start <= chunk for start, stop in ranges)


def test_all_terminal_group_runs_no_trips(monkeypatch) -> None:
    """A group whose lanes all fail at the initial map stops immediately."""
    system = _setup(_model(), H2)
    positions = tuple(_geometry(H2, seed) for seed in range(2))
    _route(monkeypatch, True)
    reference = evaluate_packed_systems(
        (system,) * 2, positions, chrg=(-6.5, -6.5)
    )
    _route(monkeypatch, False)
    ranges = _count_advances(monkeypatch)
    adaptive = evaluate_packed_systems(
        (system,) * 2, positions, chrg=(-6.5, -6.5)
    )
    assert ranges == []
    assert not bool(adaptive.converged.any())
    _assert_same_leaves(reference, adaptive)


def test_maxiter_exhaustion_runs_full_schedule(monkeypatch) -> None:
    system = _setup(_model(maxiter=5), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(2))
    _route(monkeypatch, True)
    reference = evaluate_packed_systems((system,) * 2, positions)
    _route(monkeypatch, False)
    ranges = _count_advances(monkeypatch)
    adaptive = evaluate_packed_systems((system,) * 2, positions)
    assert ranges[-1][1] == 5
    assert adaptive.iterations.tolist() == [6, 6]
    assert not bool(adaptive.converged.any())
    _assert_same_leaves(reference, adaptive)


# Failure and warning semantics


def _outcome(call: Callable[[], object]):
    OutputHandler.clear_warnings()
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


_FAILURES = {
    # name: (numbers, model settings, charges, spins, legacy_compat)
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
def test_failure_semantics_match_static_schedule(monkeypatch, case) -> None:
    numbers, settings, charges, spins, legacy = _FAILURES[case]
    system = _setup(_model(**settings), numbers)
    positions = tuple(_geometry(numbers, seed) for seed in range(3))

    def call():
        return evaluate_packed_systems(
            (system,) * 3,
            positions,
            chrg=0.0 if charges is None else charges,
            spin=spins,
            legacy_compat=legacy,
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
    monkeypatch,
) -> None:
    _route(monkeypatch, False)
    system = _setup(_model(maxiter=8), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))
    OutputHandler.clear_warnings()
    result = evaluate_packed_systems((system,) * 3, positions)
    OutputHandler.clear_warnings()
    converged = result.converged.tolist()
    assert any(converged) and not all(converged)


# Routing


def _spy_adaptive(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = lane_evaluation._evaluate_group_adaptive

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lane_evaluation, "_evaluate_group_adaptive", spy)
    return calls


def test_eager_calls_use_the_adaptive_route(monkeypatch) -> None:
    calls = _spy_adaptive(monkeypatch)
    system = _setup(_model(), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(2))
    evaluate_packed_systems((system,) * 2, positions)
    with torch.no_grad():
        evaluate_packed_systems((system,) * 2, positions)
    assert calls == [1, 1]


@pytest.mark.parametrize("transform", ["jacrev", "jacfwd", "jvp", "vmap"])
def test_transforms_use_the_static_schedule(monkeypatch, transform) -> None:
    """Inside torch.func transforms the full static schedule is used."""
    system = _setup(_model(), H2O)
    base = torch.stack([_geometry(H2O, seed) for seed in range(2)])

    def energy(stacked: torch.Tensor) -> torch.Tensor:
        return evaluate_packed_systems(
            (system,) * 2, tuple(stacked.unbind(0))
        ).energy.sum()

    calls = _spy_adaptive(monkeypatch)
    if transform == "jacrev":
        value = torch.func.jacrev(energy)(base)
    elif transform == "jacfwd":
        value = torch.func.jacfwd(energy)(base)
    elif transform == "jvp":
        value = torch.func.jvp(energy, (base,), (torch.ones_like(base),))[1]
    else:
        value = torch.func.vmap(energy)(torch.stack([base, base + 0.01]))
    assert calls == []

    # Eager reference through the adaptive route.
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
    assert len(calls) >= 1
    torch.testing.assert_close(value, reference, **TOL)


def test_compiling_requires_the_static_schedule(monkeypatch) -> None:
    assert not lane_evaluation._requires_static_schedule()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert lane_evaluation._requires_static_schedule()


def test_carry_rejects_lane_tensors_outside_the_dataclass_tree() -> None:
    """A batched tensor held by a plain object cannot escape a vmap call."""

    class Holder:
        def __init__(self, value):
            self.value = value

    def stage(x):
        lane_evaluation._carry({}, "state", (x * 2.0, Holder(x + 1.0)))
        return x

    with pytest.raises(RuntimeError, match="outside the carried"):
        torch.func.vmap(stage)(torch.ones(2, 3))


def test_forward_mode_dual_tensors_match_static_schedule(monkeypatch) -> None:
    """Eager forward-mode AD (not a functorch level) runs adaptively."""
    import torch.autograd.forward_ad as fwAD

    system = _setup(_model(), H2O)
    positions = [_geometry(H2O, seed) for seed in range(2)]
    generator = torch.Generator().manual_seed(3)
    tangents = [
        torch.randn(p.shape, generator=generator, dtype=DTYPE)
        for p in positions
    ]
    calls = _spy_adaptive(monkeypatch)
    tangent_out = {}
    for static in (True, False):
        _route(monkeypatch, static)
        with fwAD.dual_level():
            dual = tuple(
                fwAD.make_dual(p, t) for p, t in zip(positions, tangents)
            )
            energy = evaluate_packed_systems((system,) * 2, dual).energy
            tangent_out[static] = fwAD.unpack_dual(energy).tangent.clone()
    assert calls == [1]
    assert float(tangent_out[False].abs().max()) > 0.0
    torch.testing.assert_close(tangent_out[True], tangent_out[False], **EXACT)
