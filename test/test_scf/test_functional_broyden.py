"""Tests for native functional Broyden method updates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.scf.functional import (
    SCFCharges,
    SCFMapResult,
    SCFPotential,
    SCFSolverSettings,
    SCFSpectralResult,
    SCFStage,
    SCFStrategy,
)
from dxtb._src.scf.functional.broyden import (
    BroydenMethod,
    BroydenMethodSettings,
    BroydenState,
    _apply_inverse,
    _update_inverse,
)
from dxtb._src.scf.functional.method import SCFMethodContext
from dxtb._src.scf.functional.problem import (
    SCFMapResult,
    SCFPhysicalState,
    SCFProblem,
)
from dxtb._src.scf.functional.solver import _solve_map


def _propose(
    method: BroydenMethod,
    x: torch.Tensor,
    mapped: torch.Tensor,
    state: BroydenState | None,
    *,
    step: int,
    damp: float = 0.5,
    active: bool = True,
    raw_mapped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, BroydenState]:
    active_tensor = torch.tensor(active, device=x.device)
    context = SCFMethodContext(
        step=step,
        transition_active=active_tensor,
        damp=x.new_tensor(damp),
        proposal_x=mapped,
    )
    physical_state = SCFPhysicalState(
        x=x,
        map_result=cast(
            SCFMapResult,
            SimpleNamespace(
                mapped_x=mapped if raw_mapped is None else raw_mapped
            ),
        ),
    )
    proposal = method.propose(
        cast(SCFProblem, None), physical_state, state, context
    )
    assert isinstance(proposal.state, BroydenState)
    return proposal.x, proposal.state


def test_broyden_first_proposal_has_expected_sign_and_scale() -> None:
    """The initial step is damp times one half of G(x)-x."""
    method = BroydenMethod()
    x = torch.tensor([1.25, -0.75], dtype=torch.float64)
    delta = torch.tensor([-0.4, 0.2], dtype=x.dtype)
    proposal, state = _propose(method, x, x + delta, None, step=1)

    torch.testing.assert_close(proposal, x + 0.25 * delta, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state.previous_x, x, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state.previous_f, -delta, rtol=0.0, atol=1e-16)
    assert int(state.history_count) == 0
    assert torch.equal(state.c_history, torch.zeros_like(state.c_history))


@pytest.mark.parametrize("mix_guess", [False, True])
@pytest.mark.parametrize("maxiter", [0, 1, 2])
def test_broyden_preserves_common_budget_and_initial_guess_semantics(
    maxiter: int, mix_guess: bool
) -> None:
    """The shared controller keeps its zero/short-budget contract."""
    initial = torch.tensor([[0.2]], dtype=torch.float64)
    slope = torch.tensor(0.3, dtype=initial.dtype)
    outcome = _affine_outcome(
        initial, slope, maxiter=maxiter, mix_guess=mix_guess
    )
    assert int(outcome.iterations) == maxiter + 1
    assert bool(outcome.converged) is False


def test_broyden_uses_controller_projected_proposal() -> None:
    """The method uses the projected proposal supplied by the controller."""
    method = BroydenMethod()
    x = torch.tensor([0.5, -0.3], dtype=torch.float64)
    projected_delta = torch.tensor([0.2, -0.1], dtype=x.dtype)
    unprojected_map = torch.tensor([20.0, -30.0], dtype=x.dtype)
    proposal, _ = _propose(
        method,
        x,
        x + projected_delta,
        None,
        step=1,
        raw_mapped=unprojected_map,
    )
    torch.testing.assert_close(proposal, x + 0.25 * projected_delta)


def test_broyden_low_rank_update_matches_dense_first_method_reference() -> None:
    """Native low-rank updates match a test-local dense Broyden-first formula."""
    method = BroydenMethod(BroydenMethodSettings(max_rank=8))
    x = torch.tensor([0.4, -0.2, 0.6], dtype=torch.float64)
    inverse = 0.5 * torch.eye(3, dtype=x.dtype)
    matrix = torch.tensor(
        [[0.35, 0.08, -0.02], [0.03, 0.27, 0.04], [0.01, -0.06, 0.31]],
        dtype=x.dtype,
    )

    def observe(old_x: torch.Tensor, state: BroydenState | None, step: int):
        f = (
            matrix @ old_x
            + 0.03 * old_x.square()
            - torch.tensor([0.1, -0.05, 0.08], dtype=x.dtype)
        )
        mapped = old_x - f
        new_x, next_state = _propose(
            method, old_x, mapped, state, step=step, damp=0.37
        )
        return f, new_x, next_state

    f, x_next, state = observe(x, None, 1)
    previous_x = x
    previous_f = f
    expected_step = -inverse @ f
    torch.testing.assert_close(
        x_next, x + 0.37 * expected_step, rtol=0.0, atol=1e-15
    )

    for step in range(2, 6):
        f_next, proposed, next_state = observe(x_next, state, step)
        s = x_next - previous_x
        y = f_next - previous_f
        v = inverse.mT @ s
        c = s - inverse @ y
        d = v / torch.dot(y, v)
        inverse = inverse + torch.outer(c, d)
        vector = torch.tensor([0.7, -0.4, 0.2], dtype=x.dtype)
        expected_inverse_action = inverse @ vector
        actual_inverse_action = _apply_inverse(next_state, vector, 0.5)
        torch.testing.assert_close(
            actual_inverse_action,
            expected_inverse_action,
            rtol=1e-13,
            atol=1e-14,
        )
        correction_count = int(next_state.history_count)
        torch.testing.assert_close(
            next_state.c_history[correction_count - 1],
            c,
            rtol=1e-13,
            atol=1e-14,
        )
        torch.testing.assert_close(
            next_state.d_history[correction_count - 1],
            d,
            rtol=1e-13,
            atol=1e-14,
        )
        expected_step = -inverse @ f_next
        torch.testing.assert_close(
            proposed, x_next + 0.37 * expected_step, rtol=1e-13, atol=1e-14
        )
        assert int(next_state.history_count) == step - 1
        previous_x, previous_f = x_next, f_next
        x_next, state = proposed, next_state


def test_broyden_capacity_restarts_and_inserts_current_secant() -> None:
    """A full correction buffer restarts at H0 before adding the latest pair."""
    method = BroydenMethod(BroydenMethodSettings(max_rank=2))
    x = torch.tensor([0.2, -0.3], dtype=torch.float64)
    state = None
    for step in range(1, 5):
        mapped = torch.stack((0.7 * x[0] + 0.1, 0.8 * x[1] - 0.1))
        x, state = _propose(method, x, mapped, state, step=step)

    assert int(state.history_count) == 1
    assert int(state.restart_count) == 1
    assert bool(state.last_update_accepted)
    assert state.c_history.shape == (2, 2)
    assert state.d_history.shape == (2, 2)
    assert int(state.valid_history.sum()) == 1


def test_broyden_rejects_degenerate_secant_without_nan_history() -> None:
    """A zero secant denominator restarts instead of storing NaN corrections."""
    method = BroydenMethod(BroydenMethodSettings(secant_quality_tolerance=0.0))
    x0 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    mapped0 = torch.tensor([0.4, 0.7], dtype=x0.dtype)
    x1, state = _propose(method, x0, mapped0, None, step=1)

    # Keep F unchanged while changing x, giving y=0 and a degenerate update.
    mapped1 = x1 - (x0 - mapped0)
    _, state = _propose(method, x1, mapped1, state, step=2)

    assert int(state.history_count) == 0
    assert int(state.restart_count) == 1
    assert not bool(state.last_update_accepted)
    assert torch.isfinite(state.c_history).all()
    assert torch.isfinite(state.d_history).all()
    assert torch.equal(state.c_history, torch.zeros_like(state.c_history))
    assert torch.equal(state.d_history, torch.zeros_like(state.d_history))


def test_broyden_rejects_nonfinite_secant_without_nan_corrections() -> None:
    """Non-finite secants clear corrections and never enter the low-rank state."""
    x0 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    _, state = _propose(
        BroydenMethod(),
        x0,
        torch.tensor([0.3, 0.6], dtype=x0.dtype),
        None,
        step=1,
    )
    state = BroydenState(
        previous_x=state.previous_x,
        previous_f=torch.full_like(state.previous_f, torch.nan),
        c_history=state.c_history,
        d_history=state.d_history,
        valid_history=state.valid_history,
        history_count=state.history_count,
        restart_count=state.restart_count,
        last_update_accepted=state.last_update_accepted,
    )
    state = _update_inverse(
        state,
        x0 + torch.tensor([0.1, -0.2], dtype=x0.dtype),
        torch.tensor([-0.5, 0.3], dtype=x0.dtype),
        BroydenMethodSettings(),
    )
    assert int(state.history_count) == 0
    assert int(state.restart_count) == 1
    assert not bool(state.last_update_accepted)
    assert torch.isfinite(state.c_history).all()
    assert torch.isfinite(state.d_history).all()


def test_broyden_inactive_transition_freezes_all_history() -> None:
    """Inactive lanes keep every Broyden state tensor unchanged."""
    method = BroydenMethod()
    x0 = torch.tensor([0.5, -0.1], dtype=torch.float64)
    _, state = _propose(method, x0, 0.8 * x0, None, step=1)
    x1 = state.previous_x + 0.1
    _, frozen = _propose(method, x1, 0.7 * x1, state, step=2, active=False)

    for name in BroydenState.__dataclass_fields__:
        assert torch.equal(getattr(frozen, name), getattr(state, name))


@pytest.mark.parametrize("rank", [1, 4, 8])
def test_broyden_history_has_fixed_bounded_shapes(rank: int) -> None:
    """History allocation scales as rank times vector size, never n squared."""
    method = BroydenMethod(BroydenMethodSettings(max_rank=rank))
    x = torch.linspace(-0.2, 0.2, 11, dtype=torch.float64)
    _, state = _propose(method, x, 0.9 * x + 0.01, None, step=1)
    assert state.c_history.shape == (rank, x.numel())
    assert state.d_history.shape == (rank, x.numel())
    assert state.c_history.numel() == rank * x.numel()
    assert state.d_history.numel() == rank * x.numel()


def _synthetic_map(mapped: torch.Tensor) -> SCFMapResult:
    return SCFMapResult(
        mapped_x=mapped,
        driving_potential=SCFPotential(mono=mapped),
        spectral=SCFSpectralResult(
            charges=SCFCharges(mono=mapped),
            hamiltonian=mapped,
            density=mapped,
            emo=mapped,
            coefficients=mapped,
            occupation=mapped,
            orbital_energy=mapped,
        ),
    )


def _affine_outcome(
    initial: torch.Tensor,
    slope: torch.Tensor,
    *,
    tolerance: float = 0.0,
    maxiter: int = 5,
    mix_guess: bool = True,
    record_trace: bool = False,
):
    settings = SCFSolverSettings(
        maxiter=10 if tolerance > 0.0 else maxiter,
        mix_guess=mix_guess,
        damp=0.4,
        x_atol=tolerance,
        x_atol_max=tolerance,
    )
    method = BroydenMethod(BroydenMethodSettings(max_rank=3))

    def map_fn(x: torch.Tensor) -> SCFMapResult:
        return _synthetic_map(slope * x + 0.7)

    outcome = _solve_map(
        initial,
        settings,
        map_fn,
        problem=None,
        strategy=SCFStrategy((SCFStage(method),)),
        record_trace=record_trace,
    )
    return outcome


def _solve_affine(initial: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    return _affine_outcome(initial, slope).map_result.mapped_x.sum()


def test_broyden_shared_controller_supports_transforms_and_trace() -> None:
    """Low-rank state remains transformable in the fixed-trip controller."""
    initial = torch.tensor([[0.25, -0.4]], dtype=torch.float64)
    slope = torch.tensor(0.3, dtype=torch.float64)
    direction = torch.ones_like(initial)

    value, tangent = jvp(
        lambda x: _solve_affine(x, slope), (initial,), (direction,)
    )
    leaf = initial.detach().requires_grad_(True)
    reverse = torch.autograd.grad(
        _solve_affine(leaf, slope), leaf, create_graph=True
    )[0]
    forward = jacfwd(lambda x: _solve_affine(x, slope))(initial)
    assert torch.isfinite(value)
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(forward).all()
    assert torch.isfinite(reverse).all()
    second = torch.autograd.grad(reverse.sum(), leaf)[0]
    assert torch.isfinite(second).all()

    guesses = torch.tensor(
        [[[0.1, -0.2]], [[0.4, 0.3]], [[-0.7, 0.6]]], dtype=torch.float64
    )
    mapped = vmap(lambda x: _solve_affine(x, slope))(guesses)
    looped = torch.stack([_solve_affine(item, slope) for item in guesses])
    torch.testing.assert_close(mapped, looped)

    mixed_guesses = torch.tensor(
        [[[1.0]], [[0.0]], [[0.4]]], dtype=torch.float64
    )

    def mixed_solve(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outcome = _affine_outcome(x, slope, tolerance=0.05)
        return (
            outcome.map_result.mapped_x.sum(),
            outcome.iterations,
            outcome.converged,
        )

    mixed = vmap(mixed_solve)(mixed_guesses)
    assert mixed[1].unique().numel() > 1
    assert torch.equal(mixed[2], torch.ones_like(mixed[2], dtype=torch.bool))

    traced = _affine_outcome(initial, slope, mix_guess=False, record_trace=True)
    assert traced.trace is not None
    assert len(traced.trace.method_state) == 8
    assert traced.trace.method_state[0].shape[0] == 6
    assert torch.equal(
        traced.trace.method_state[0][0],
        torch.zeros_like(traced.trace.method_state[0][0]),
    )
    assert traced.trace.method_state[0][0].shape == (2,)


def test_broyden_production_module_has_no_xitorch_dependency() -> None:
    """The production method is independent of the legacy root solver."""
    source = (
        Path(__file__).parents[2] / "src/dxtb/_src/scf/functional/broyden.py"
    )
    text = source.read_text(encoding="utf-8")
    assert "dxtb._src.exlibs.xitorch" not in text


def test_full_broyden_keeps_position_parameter_and_field_gradients() -> None:
    """The FULL Broyden route keeps all three physical input graphs connected."""
    from test.test_scf.test_functional_map import _prepare_case

    from dxtb import GFN1_XTB, labels
    from dxtb._src.param import ParamModule
    from dxtb._src.scf.functional.eager import run_functional_full_eager

    dtype = torch.float64
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [1.0, 0.0, 1.8]],
        dtype=dtype,
        requires_grad=True,
    )
    field = torch.tensor(
        [0.001, -0.0007, 0.0004], dtype=dtype, requires_grad=True
    )
    parameter_module = ParamModule(GFN1_XTB, device="cpu", dtype=dtype)
    parameter = parameter_module.get("element.O.levels")
    parameter.requires_grad_(True)
    case = _prepare_case(
        GFN1_XTB,
        positions=positions,
        field=field,
        par=parameter_module,
        scf_options={
            "mixer": labels.MIXER_BROYDEN,
            "maxiter": 3,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    result = run_functional_full_eager(
        case.system.numbers,
        positions,
        case.charge,
        case.spin,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        case.system.config.scf,
        case.integrals,
        case.refocc,
    )
    assert result.functional is not None
    energy = result.functional.energy.sum() + result.functional.fenergy.sum()
    gradients = torch.autograd.grad(energy, (positions, parameter, field))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    "sample_name, method, charge, spin, temperature, dtype",
    [
        pytest.param(
            "NO2",
            GFN2_XTB,
            0.0,
            1.0,
            300.0,
            torch.float64,
            id="gfn2-no2-doublet",
        ),
        pytest.param(
            "H2O",
            GFN2_XTB,
            1.0,
            1.0,
            1000.0,
            torch.float64,
            id="gfn2-charged-doublet",
        ),
        pytest.param(
            "C4H5NCS",
            GFN2_XTB,
            0.0,
            0.0,
            300.0,
            torch.float64,
            id="gfn2-difficult",
        ),
        pytest.param(
            "stretched-H2",
            GFN1_XTB,
            0.0,
            0.0,
            0.0,
            torch.float64,
            id="gfn1-stretched",
        ),
        pytest.param(
            "H2O",
            GFN2_XTB,
            0.0,
            0.0,
            300.0,
            torch.float32,
            id="gfn2-float32-300K",
        ),
    ],
)
def test_broyden_full_physical_case_matrix(
    sample_name: str,
    method: int,
    charge: float,
    spin: float,
    temperature: float,
    dtype: torch.dtype,
) -> None:
    """Open-shell, finite-T, stretched, and difficult roots certify natively."""
    from test.test_scf.samples import samples
    from test.test_scf.test_functional_map import _prepare_case

    positions = None
    molecule = sample_name
    if sample_name == "stretched-H2":
        molecule = "H2"
        positions = torch.tensor(
            [[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]], dtype=dtype
        )
        assert samples[molecule]["numbers"].numel() == 2
    case = _prepare_case(
        method,
        sample_name=molecule,
        positions=positions,
        charge=charge,
        spin=spin,
        dtype=dtype,
        scf_options={
            "mixer": labels.MIXER_BROYDEN,
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "fermi_etemp": temperature,
            "maxiter": 150,
            "x_atol": 1.0e-8,
            "x_atol_max": 1.0e-8,
        },
    )
    from dxtb._src.scf.functional.eager import run_functional_full_eager

    result = run_functional_full_eager(
        case.system.numbers,
        case.positions,
        case.charge,
        case.spin,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        case.system.config.scf,
        case.integrals,
        case.refocc,
    )
    assert result.route.effective_mixer == labels.MIXER_BROYDEN
    assert result.functional is not None
    assert result.functional.converged.item() is True
    assert torch.isfinite(result.functional.energy).all()
    assert torch.isfinite(result.functional.fenergy).all()
    diagnostics = result.functional.solve_result.diagnostics
    assert diagnostics is not None and diagnostics.certification is not None
    assert diagnostics.certification.result.certified.item() is True
