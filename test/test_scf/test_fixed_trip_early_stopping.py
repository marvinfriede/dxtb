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
Transform-compatible fixed-trip SCF early stopping and wrapper unwrapping.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from dxtb import labels
from dxtb._src.calculators.model import Model, System
from dxtb._src.scf import (
    LinearConverger,
    SCFCharges,
    SCFMapResult,
    SCFPotential,
    SCFSpectralResult,
)
from dxtb._src.scf import eager as eager_mod
from dxtb._src.scf import solve_scf_problem
from dxtb._src.scf import solver as solver_mod
from dxtb._src.scf.solver import (
    DynamicDampingSettings,
    _can_stop_fixed_trip_early,
    _solve_map_control,
)
from dxtb._src.typing import Tensor
from dxtb._src.utils import tensors as tensors_mod
from dxtb._src.utils.tensors import disable_functorch, unwrap_functorch_tensor
from dxtb.config import Config

from ..utils import get_param_module, scf_strategy
from .samples import samples


class _PassthroughDispatchMode(TorchDispatchMode):
    """Minimal Python dispatch mode that forwards every operation unchanged."""

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Forward ``func`` without modification."""
        return func(*args, **(kwargs or {}))


class _WrappedBoolTensor:
    """Custom ``__torch_function__`` carrier rejected by ``has_torch_function_unary``."""

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Delegate PyTorch operations to default behavior."""
        return NotImplemented


def test_unwrap_plain_tensor_returns_identity() -> None:
    """Plain ``torch.Tensor`` unwraps to itself regardless of ``allow_vmap``."""
    flag = torch.tensor([True, False])
    assert unwrap_functorch_tensor(flag, allow_vmap=False) is flag
    assert unwrap_functorch_tensor(flag, allow_vmap=True) is flag


@pytest.mark.parametrize("allow_vmap", [False, True])
def test_unwrap_gradtracking_tensor_peels_nested_ad(allow_vmap: bool) -> None:
    """Nested ``jvp``/``jacrev``/``jacfwd`` wrappers unwrap to a plain tensor."""
    observed: list[Tensor | None] = []
    ft = torch._C._functorch

    def cubic_probe(x: Tensor) -> Tensor:
        """Probe ``unwrap_functorch_tensor`` inside a 3rd-order transform stack."""
        active = x > 0.0
        raw = unwrap_functorch_tensor(active, allow_vmap=allow_vmap)
        observed.append(raw)
        return (x**3).sum()

    x0 = torch.tensor([1.0, -2.0], dtype=torch.float64)
    third = torch.func.jacfwd(torch.func.jacrev(torch.func.grad(cubic_probe)))(
        x0
    )
    assert third.shape == (2, 2, 2)
    assert observed
    for raw in observed:
        assert raw is not None
        assert type(raw) is Tensor
        assert not ft.is_functorch_wrapped_tensor(raw)
        with disable_functorch():
            assert torch.equal(raw, torch.tensor([True, False]))


def test_unwrap_batched_tensor_respects_allow_vmap() -> None:
    """``BatchedTensor`` returns ``None`` unless ``allow_vmap=True``."""
    ft = torch._C._functorch
    disallowed: list[Tensor | None] = []
    allowed: list[Tensor | None] = []

    def mapped_probe(x: Tensor) -> Tensor:
        """Record unwrapped ``active`` under nested ``vmap`` and ``jacrev``."""
        active = x > 0.0
        disallowed.append(unwrap_functorch_tensor(active, allow_vmap=False))
        allowed.append(unwrap_functorch_tensor(active, allow_vmap=True))
        return (x**2).sum()

    batch = torch.tensor([[1.0, 2.0], [-1.0, -3.0]], dtype=torch.float64)
    grads = torch.func.vmap(torch.func.jacrev(mapped_probe))(batch)
    assert grads.shape == (2, 2)
    assert disallowed == [None]
    assert len(allowed) == 1
    raw = allowed[0]
    assert raw is not None
    assert type(raw) is Tensor
    assert not ft.is_functorch_wrapped_tensor(raw)
    with disable_functorch():
        assert raw.shape == (2, 2)
        assert bool(raw.any()) is True


@pytest.mark.parametrize("allow_vmap", [False, True])
def test_unwrap_rejects_functional_tensor_unconditionally(
    allow_vmap: bool,
) -> None:
    """``FunctionalTensor`` returns ``None`` even when ``allow_vmap=True``."""
    observed: list[Tensor | None] = []

    def inner_fn(x: Tensor) -> Tensor:
        """Inspect ``active`` inside ``functionalize``."""
        active = x > 0.0
        observed.append(unwrap_functorch_tensor(active, allow_vmap=allow_vmap))
        return x + 1.0

    functional_fn = torch.func.functionalize(inner_fn)
    out = functional_fn(torch.tensor([1.0, -1.0], dtype=torch.float64))
    assert out.shape == (2,)
    assert observed == [None]

    observed.clear()
    mapped_functional = torch.func.vmap(functional_fn)
    batch_out = mapped_functional(
        torch.tensor([[1.0, -1.0], [2.0, 3.0]], dtype=torch.float64)
    )
    assert batch_out.shape == (2, 2)
    assert observed == [None]


def test_unwrap_rejects_compile_jit_dispatch_and_torch_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracing, compilation, dispatch modes, and ``__torch_function__`` return ``None``."""
    flag = torch.tensor([False, False])

    with _PassthroughDispatchMode():
        assert unwrap_functorch_tensor(flag, allow_vmap=True) is None

    def always_compiling() -> bool:
        """Simulate an active ``torch.compile`` Dynamo frame."""
        return True

    monkeypatch.setattr(tensors_mod, "is_compiling", always_compiling)
    assert unwrap_functorch_tensor(flag, allow_vmap=True) is None
    monkeypatch.undo()

    trace_observations: list[tuple[bool, Tensor | None]] = []

    def traced_fn(x: Tensor) -> Tensor:
        """Probe ``unwrap_functorch_tensor`` during ``torch.jit.trace``."""
        trace_observations.append(
            (
                bool(torch.jit.is_tracing()),
                unwrap_functorch_tensor(x > 0.0, allow_vmap=True),
            )
        )
        return x * 2.0

    traced = torch.jit.trace(
        traced_fn,
        (torch.tensor([1.0, -1.0], dtype=torch.float64),),
        check_trace=False,
    )
    assert traced is not None
    assert trace_observations == [(True, None)]

    class _TensorSubclass(Tensor):
        """Tensor subclass triggering ``has_torch_function_unary``."""

        @classmethod
        def __torch_function__(
            cls,
            func: Any,
            types: tuple[type, ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            """Delegate to ``Tensor.__torch_function__``."""
            return super().__torch_function__(func, types, args, kwargs or {})

    sub = flag.as_subclass(_TensorSubclass)
    assert unwrap_functorch_tensor(sub, allow_vmap=True) is None
    assert (
        unwrap_functorch_tensor(
            _WrappedBoolTensor(),  # type: ignore[arg-type]
            allow_vmap=True,
        )
        is None
    )


def test_unwrap_rejects_wrapper_depth_exceeding_sixteen() -> None:
    """Wrapper stacks deeper than 16 layers return ``None`` instead of a wrapper."""
    observed: list[Tensor | None] = []

    def leaf_probe(x: Tensor) -> Tensor:
        """Check unwrapping at the bottom of a 17-layer ``jvp`` stack."""
        observed.append(unwrap_functorch_tensor(x > 0.0, allow_vmap=True))
        return x

    def make_jvp_layer(fn: Any) -> Any:
        """Wrap ``fn`` in one additional forward-mode ``jvp`` layer."""

        def wrapped(x: Tensor) -> Tensor:
            """Evaluate ``jvp(fn, (x,), (ones,))`` and return primal."""
            y, _ = torch.func.jvp(fn, (x,), (torch.ones_like(x),))
            return y

        return wrapped

    fn: Any = leaf_probe
    for _ in range(17):
        fn = make_jvp_layer(fn)

    out = fn(torch.tensor([1.0], dtype=torch.float64))
    assert out.shape == (1,)
    assert observed == [None]


def _build_system(
    sample_name: str,
    *,
    method: int = labels.GFN1_XTB,
    maxiter: int = 35,
    strategy_spec: object | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[System, Tensor]:
    """Construct a cached ``System`` and positions tensor on CPU."""
    device = torch.device("cpu")
    mol = samples[sample_name]
    numbers = mol["numbers"].to(device=device)
    positions = mol["positions"].to(device=device, dtype=dtype)
    param_name: Literal["gfn1", "gfn2"] = (
        "gfn1" if method == labels.GFN1_XTB else "gfn2"
    )
    par = get_param_module(param_name, device=device, dtype=dtype)
    opts: dict[str, Any] = {
        "method": method,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "maxiter": maxiter,
        "fermi_etemp": 300.0,
        "differentiation": "unrolled",
    }
    if strategy_spec is not None and strategy_spec != "default":
        opts["strategy"] = strategy_spec
    config = Config.create(**opts)
    model = Model(par=par, config=config, auto_int_level=False)
    return (
        model.setup(numbers, dd={"device": device, "dtype": dtype}),
        positions,
    )


def _spy_executed_trips(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record ``state.executed_trips`` at every ``_solver_outcome`` call."""
    recorded: list[int] = []
    original = solver_mod._solver_outcome

    def spy(state: Any, *args: Any, **kwargs: Any) -> Any:
        """Record ``state.executed_trips`` and delegate to ``_solver_outcome``."""
        recorded.append(int(state.executed_trips))
        return original(state, *args, **kwargs)

    monkeypatch.setattr(solver_mod, "_solver_outcome", spy)
    return recorded


def _disable_early_stop(_active: Tensor) -> bool:
    """Disable CPU early stopping so ``FixedTripControl`` runs ``maxiter`` trips."""
    return False


def test_meta_device_disables_early_stopping_without_host_sync() -> None:
    """Non-CPU ``meta`` tensors skip host boolean reads and run ``maxiter`` trips."""
    meta_active = torch.zeros((2,), dtype=torch.bool, device="meta")
    assert _can_stop_fixed_trip_early(meta_active) is False

    guess = torch.zeros((3,), dtype=torch.float64, device="meta")

    def meta_map(x: Tensor) -> SCFMapResult:
        """Synthetic ``meta``-device SCF map with zero residual."""
        return SCFMapResult(
            mapped_x=x,
            driving_potential=SCFPotential(mono=x),
            spectral=SCFSpectralResult(
                charges=SCFCharges(mono=x),
                hamiltonian=x,
                density=x,
                emo=x,
                coefficients=x,
                occupation=x,
                orbital_energy=x,
            ),
        )

    outcome = _solve_map_control(
        guess,
        maxiter=4,
        mix_guess=False,
        x_atol=1.0,
        x_atol_max=1.0,
        map_fn=meta_map,
        method=LinearConverger(),
        damping_settings=DynamicDampingSettings(),
    )
    assert outcome.executed_trips == 4
    assert outcome.root_x.device.type == "meta"


@pytest.mark.parametrize(
    "sample_name,method,strategy_id,strategy_spec",
    [
        ("LiH", labels.GFN1_XTB, "charge-anderson", "charge-anderson"),
        ("LiH", labels.GFN1_XTB, "broyden", "broyden"),
        ("LiH", labels.GFN1_XTB, "linear", "linear"),
        ("H2O", labels.GFN2_XTB, "cdiis", "cdiis"),
        ("H2O", labels.GFN2_XTB, "default", "default"),
    ],
    ids=[
        "lih-gfn1-anderson",
        "lih-gfn1-broyden",
        "lih-gfn1-linear",
        "h2o-gfn2-cdiis",
        "h2o-gfn2-default",
    ],
)
def test_single_system_transform_derivatives_match_static_schedule(
    monkeypatch: pytest.MonkeyPatch,
    sample_name: str,
    method: int,
    strategy_id: str,
    strategy_spec: str,
) -> None:
    """1st--3rd order AD transforms stop early on CPU and match static ``maxiter``."""
    maxiter = 18
    resolved_strategy: object = (
        scf_strategy("anderson", "charge")
        if strategy_id == "charge-anderson"
        else strategy_spec
    )
    system, positions = _build_system(
        sample_name,
        method=method,
        maxiter=maxiter,
        strategy_spec=resolved_strategy,
    )
    field0 = torch.tensor([0.002, -0.001, 0.0015], dtype=torch.float64)
    direction = torch.tensor([0.6, -0.3, 0.7], dtype=torch.float64)
    cotangent = torch.tensor(1.0, dtype=torch.float64)
    alpha0 = torch.zeros((), dtype=torch.float64)

    def field_energy(field: Tensor) -> Tensor:
        """Scalar total energy as a function of external electric field."""
        return system.energy(positions, field=field)

    def directional_field_energy(alpha: Tensor) -> Tensor:
        """Scalar total energy along ``field0 + alpha * direction``."""
        return system.energy(positions, field=field0 + alpha * direction)

    def evaluate_suite() -> tuple[Tensor, ...]:
        """Compute primal and 1st--3rd order transform outputs."""
        field_leaf = field0.clone().requires_grad_(True)
        primal_e = field_energy(field_leaf)
        grad_auto = torch.autograd.grad(primal_e, field_leaf)[0]
        jvp_val, jvp_tan = torch.func.jvp(field_energy, (field0,), (direction,))
        vjp_val, vjp_fn = torch.func.vjp(field_energy, field0)
        vjp_cot = vjp_fn(cotangent)[0]
        jac_rev = torch.func.jacrev(field_energy)(field0)
        jac_fwd = torch.func.jacfwd(field_energy)(field0)
        d2 = torch.func.jacfwd(torch.func.jacrev(directional_field_energy))(
            alpha0
        )
        d3 = torch.func.jacfwd(
            torch.func.jacfwd(torch.func.jacrev(directional_field_energy))
        )(alpha0)
        return (
            primal_e.detach(),
            grad_auto,
            jvp_val,
            jvp_tan,
            vjp_val,
            vjp_cot,
            jac_rev,
            jac_fwd,
            d2,
            d3,
        )

    trips = _spy_executed_trips(monkeypatch)
    early_outputs = evaluate_suite()
    assert trips
    assert all(0 < count < maxiter for count in trips)

    trips.clear()
    with monkeypatch.context() as m:
        m.setattr(solver_mod, "_can_stop_fixed_trip_early", _disable_early_stop)
        static_outputs = evaluate_suite()
    assert trips
    assert all(count == maxiter for count in trips)

    for actual, expected in zip(early_outputs, static_outputs):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_float32_single_system_early_stopping_matches_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Float32 single-system ``jacrev`` stops early and matches static execution."""
    maxiter = 18
    system, positions = _build_system(
        "LiH",
        method=labels.GFN1_XTB,
        maxiter=maxiter,
        dtype=torch.float32,
    )

    def pos_energy(pos: Tensor) -> Tensor:
        """Scalar float32 total energy as a function of positions."""
        return system.energy(pos)

    trips = _spy_executed_trips(monkeypatch)
    early_grad = torch.func.jacrev(pos_energy)(positions)
    assert trips and all(0 < count < maxiter for count in trips)

    trips.clear()
    with monkeypatch.context() as m:
        m.setattr(solver_mod, "_can_stop_fixed_trip_early", _disable_early_stop)
        static_grad = torch.func.jacrev(pos_energy)(positions)
    assert trips and all(count == maxiter for count in trips)
    torch.testing.assert_close(early_grad, static_grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "sample_name,method",
    [("LiH", labels.GFN1_XTB), ("H2O", labels.GFN2_XTB)],
    ids=["lih-gfn1", "h2o-gfn2"],
)
def test_system_public_property_methods_stop_early_and_match_static(
    monkeypatch: pytest.MonkeyPatch,
    sample_name: str,
    method: int,
) -> None:
    """``System`` forces, Hessian, ``pol_deriv``, and hyperpolarizability match static."""
    maxiter = 14
    system, positions = _build_system(
        sample_name, method=method, maxiter=maxiter
    )

    def evaluate_properties() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate 1st, 2nd, and 3rd order ``System`` response properties."""
        return (
            system.forces(positions),
            system.hessian(positions),
            system.pol_deriv(positions),
            system.hyperpolarizability(positions),
        )

    trips = _spy_executed_trips(monkeypatch)
    early_props = evaluate_properties()
    assert len(trips) == 4
    assert all(0 < count < maxiter for count in trips)

    trips.clear()
    with monkeypatch.context() as m:
        m.setattr(solver_mod, "_can_stop_fixed_trip_early", _disable_early_stop)
        static_props = evaluate_properties()
    assert trips == [maxiter] * 4

    for actual, expected in zip(early_props, static_props):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_vmap_and_nested_ad_stop_after_slowest_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapped ``vmap`` and nested ``jacrev``/``jacfwd`` exit after the slowest lane."""
    maxiter = 18
    system, base_positions = _build_system(
        "H2O",
        method=labels.GFN1_XTB,
        maxiter=maxiter,
        strategy_spec="cdiis",
    )
    gen = torch.Generator().manual_seed(7)
    lane0 = base_positions + 0.02 * torch.randn(
        base_positions.shape, generator=gen, dtype=torch.float64
    )
    lane1 = base_positions + 0.02 * torch.randn(
        base_positions.shape, generator=gen, dtype=torch.float64
    )
    lane1[1, 0] += 0.9
    lane2 = base_positions + 0.02 * torch.randn(
        base_positions.shape, generator=gen, dtype=torch.float64
    )
    lane2[1, 0] += 2.1
    lane2[2, 2] -= 0.7
    batch_positions = torch.stack((lane0, lane1, lane2))

    def single_energy_and_iterations(pos: Tensor) -> tuple[Tensor, Tensor]:
        """Return total energy and iteration count for one ``System`` lane."""
        res = system.singlepoint(pos)
        assert res.iterations is not None
        return res.energy.sum(), res.iterations

    def single_energy(pos: Tensor) -> Tensor:
        """Return scalar total energy for one ``System`` lane."""
        return system.energy(pos)

    mapped_singlepoint = torch.func.vmap(single_energy_and_iterations)

    def summed_mapped_energy(pos_batch: Tensor) -> Tensor:
        """Sum mapped energies across the heterogeneous conformer batch."""
        return torch.func.vmap(single_energy)(pos_batch).sum()

    mapped_jacfwd = torch.func.vmap(torch.func.jacfwd(single_energy))

    trips = _spy_executed_trips(monkeypatch)
    early_energies, early_iters = mapped_singlepoint(batch_positions)
    assert len(set(early_iters.tolist())) >= 2
    slowest_trips = int(early_iters.max()) - 1
    assert 0 < slowest_trips < maxiter
    assert trips == [slowest_trips]

    trips.clear()
    early_jac_outer = torch.func.jacrev(summed_mapped_energy)(batch_positions)
    assert trips == [slowest_trips]

    trips.clear()
    early_jac_inner = mapped_jacfwd(batch_positions)
    assert trips == [slowest_trips]
    assert bool(torch.isfinite(early_jac_outer).all())
    assert bool(torch.isfinite(early_jac_inner).all())

    trips.clear()
    with monkeypatch.context() as m:
        m.setattr(solver_mod, "_can_stop_fixed_trip_early", _disable_early_stop)
        static_energies, static_iters = mapped_singlepoint(batch_positions)
        static_jac_outer = torch.func.jacrev(summed_mapped_energy)(
            batch_positions
        )
        static_jac_inner = mapped_jacfwd(batch_positions)
    assert trips == [maxiter, maxiter, maxiter]

    assert torch.equal(early_iters, static_iters)
    torch.testing.assert_close(
        early_energies, static_energies, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        early_jac_outer, static_jac_outer, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        early_jac_inner, static_jac_inner, atol=0.0, rtol=0.0
    )


def test_record_trace_runs_full_maxiter_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_trace=True`` executes all ``maxiter`` trips and keeps ``1 + maxiter`` rows."""
    maxiter = 25
    captured: list[tuple[Any, Any]] = []

    def capture_solve(*args: Any, **kwargs: Any) -> Any:
        """Capture ``solve_scf_problem`` arguments from ``System.singlepoint``."""
        captured.append((args, kwargs))
        return solve_scf_problem(*args, **kwargs)

    system, positions = _build_system(
        "LiH", method=labels.GFN1_XTB, maxiter=maxiter
    )
    with monkeypatch.context() as m:
        m.setattr(eager_mod, "solve_scf_problem", capture_solve)
        system.singlepoint(positions)
    ((args, kwargs),) = captured
    guess, problem, settings, strategy = args
    profile = kwargs["certification_profile"]

    untraced = solve_scf_problem(
        guess,
        problem,
        settings,
        strategy,
        eager=False,
        record_trace=False,
        certification_profile=profile,
    )
    traced = solve_scf_problem(
        guess,
        problem,
        settings,
        strategy,
        eager=False,
        record_trace=True,
        certification_profile=profile,
    )
    assert bool(untraced.converged) and bool(traced.converged)
    assert untraced.diagnostics is not None and traced.diagnostics is not None
    assert (
        untraced.diagnostics.executed_trips
        == int(untraced.iterations) - 1
        < maxiter
    )
    assert traced.diagnostics.executed_trips == maxiter
    assert traced.trace is not None
    assert traced.trace.map_inputs.shape[0] == 1 + maxiter
    torch.testing.assert_close(
        untraced.energy, traced.energy, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        untraced.final_charges.mono,
        traced.final_charges.mono,
        atol=0.0,
        rtol=0.0,
    )


def test_functionalize_and_compile_flag_run_static_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``functionalize``, dispatch modes, and ``is_compiling`` run ``maxiter`` trips."""
    contract_maxiter = 12
    guess = torch.tensor([1.0, -0.5], dtype=torch.float64)

    def contracting_map(x: Tensor) -> SCFMapResult:
        """Pure-tensor contracting map compatible with ``functionalize``."""
        next_x = 0.1 * x
        return SCFMapResult(
            mapped_x=next_x,
            driving_potential=SCFPotential(mono=next_x),
            spectral=SCFSpectralResult(
                charges=SCFCharges(mono=next_x),
                hamiltonian=next_x,
                density=next_x,
                emo=next_x,
                coefficients=next_x,
                occupation=next_x,
                orbital_energy=next_x,
            ),
        )

    def solve_root(x0: Tensor) -> Tensor:
        """Run ``_solve_map_control`` and return ``root_x``."""
        return _solve_map_control(
            x0,
            maxiter=contract_maxiter,
            mix_guess=False,
            x_atol=1.0e-2,
            x_atol_max=1.0e-2,
            map_fn=contracting_map,
            method=LinearConverger(),
            damping_settings=DynamicDampingSettings(),
            initial_damp=1.0,
        ).root_x

    trips = _spy_executed_trips(monkeypatch)
    early_root = solve_root(guess)
    assert len(trips) == 1 and 0 < trips[0] < contract_maxiter

    trips.clear()
    func_root = torch.func.functionalize(solve_root)(guess)
    assert trips == [contract_maxiter]
    torch.testing.assert_close(early_root, func_root, atol=0.0, rtol=0.0)

    maxiter = 16
    system, positions = _build_system(
        "LiH", method=labels.GFN1_XTB, maxiter=maxiter
    )

    def energy_fn(pos: Tensor) -> Tensor:
        """Return scalar total energy of ``LiH``."""
        return system.energy(pos)

    trips.clear()
    early_energy = energy_fn(positions)
    assert len(trips) == 1 and 0 < trips[0] < maxiter

    def always_compiling() -> bool:
        """Simulate ``torch.compile`` tracing inside ``unwrap_functorch_tensor``."""
        return True

    trips.clear()
    with monkeypatch.context() as m:
        m.setattr(tensors_mod, "is_compiling", always_compiling)
        compiled_energy = energy_fn(positions)
    assert trips == [maxiter]
    torch.testing.assert_close(
        early_energy, compiled_energy, atol=0.0, rtol=0.0
    )

    trips.clear()
    with _PassthroughDispatchMode():
        dispatched_energy = energy_fn(positions)
    assert trips == [maxiter]
    torch.testing.assert_close(
        early_energy, dispatched_energy, atol=0.0, rtol=0.0
    )
