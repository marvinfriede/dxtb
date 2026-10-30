"""First-class method and shared-controller architecture tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from dxtb import GFN1_XTB, labels
from dxtb._src.scf.functional import anderson as anderson_module
from dxtb._src.scf.functional.anderson import (
    AndersonMethod,
    AndersonMethodSettings,
    AndersonState,
    anderson_mix,
)
from dxtb._src.scf.functional.data import SCFMapResult
from dxtb._src.scf.functional.method import SCFMethodContext
from dxtb._src.scf.functional.problem import SCFProblem, SCFRepresentation
from dxtb._src.scf.functional.strategy import SCFStrategy

from ._anderson_helpers import AndersonTestSettings, solve_common
from .test_functional_map import _prepare_case


def _method() -> AndersonMethod:
    return AndersonMethod(
        AndersonMethodSettings(
            damp_init=0.15,
            generations=2,
            soft_start=True,
            diagonal_offset=0.03,
        )
    )


def test_anderson_proposal_uses_controller_projected_value() -> None:
    """Anderson consumes context.proposal_x after controller projection."""
    case = _prepare_case(GFN1_XTB)
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    physical_state = problem.evaluate(case.x)
    projected = physical_state.map_result.mapped_x + 0.125
    assert not torch.equal(projected, physical_state.map_result.mapped_x)
    active = torch.ones((), dtype=torch.bool, device=case.x.device)
    damping = case.x.new_tensor(0.4)
    context = SCFMethodContext(1, active, damping, projected)

    proposal = _method().propose(problem, physical_state, None, context)
    expected, expected_state = anderson_mix(
        physical_state.x,
        projected,
        None,
        _method().settings,
        1,
        transition_active=active,
        damp=damping,
    )
    torch.testing.assert_close(proposal.x, expected, rtol=0.0, atol=0.0)
    assert isinstance(proposal.state, AndersonState)
    torch.testing.assert_close(
        proposal.state.x_hist, expected_state.x_hist, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        proposal.state.f_hist, expected_state.f_hist, rtol=0.0, atol=0.0
    )


def test_anderson_inactive_transition_freezes_fixed_size_history() -> None:
    """Inactive method proposals leave every history tensor unchanged."""
    case = _prepare_case(GFN1_XTB)
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    state = problem.evaluate(case.x)
    method = _method()
    active = torch.ones((), dtype=torch.bool, device=case.x.device)
    context = SCFMethodContext(
        1, active, case.x.new_tensor(0.4), state.map_result.mapped_x
    )
    initialized = method.propose(problem, state, None, context).state
    assert isinstance(initialized, AndersonState)
    expected_shape = (method.settings.generations + 1, case.x.numel())
    assert initialized.x_hist.shape == expected_shape
    assert initialized.f_hist.shape == expected_shape

    inactive = torch.zeros_like(active)
    frozen_context = SCFMethodContext(
        2, inactive, case.x.new_tensor(0.4), state.map_result.mapped_x + 0.25
    )
    frozen = method.propose(problem, state, initialized, frozen_context).state
    assert isinstance(frozen, AndersonState)
    torch.testing.assert_close(frozen.x_hist, initialized.x_hist, rtol=0.0, atol=0.0)
    torch.testing.assert_close(frozen.f_hist, initialized.f_hist, rtol=0.0, atol=0.0)


def test_effective_configuration_builds_a_single_method_strategy() -> None:
    """Configuration chooses a method before the shared solver starts."""
    from dxtb._src.scf.functional.eager import _prepare_solver

    case = _prepare_case(GFN1_XTB)
    config = case.system.config.scf
    _, linear_strategy = _prepare_solver(
        replace(config, mixer=labels.MIXER_LINEAR), case.positions.dtype
    )
    _, anderson_strategy = _prepare_solver(
        replace(config, mixer=labels.MIXER_ANDERSON),
        case.positions.dtype,
    )
    assert isinstance(linear_strategy, SCFStrategy)
    assert isinstance(anderson_strategy, SCFStrategy)
    assert len(linear_strategy.stages) == len(anderson_strategy.stages) == 1
    assert isinstance(anderson_strategy.single_stage().method, AndersonMethod)


def test_anderson_module_has_no_solver_lifecycle() -> None:
    """The method module cannot select or run a separate controller."""
    source = Path(anderson_module.__file__).read_text()
    for forbidden in (
        "_solve_map_anderson",
        "_solve_map_control",
        "_solve_map_control_eager",
        "solve_scf_",
    ):
        assert forbidden not in source

    solver_source = Path("src/dxtb/_src/scf/functional/solver.py").read_text()
    assert "AndersonMethod" not in solver_source
    assert "mix_fn" not in solver_source


def test_terminal_map_does_not_advance_anderson_history(
    monkeypatch,
) -> None:
    """A terminal non-finite observation cannot contaminate method history."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_ANDERSON,
            "maxiter": 2,
            "mix_guess": False,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    calls = 0
    observed: list[tuple[torch.Tensor, AndersonState]] = []
    original = anderson_module.anderson_mix

    def map_fn(value: torch.Tensor):
        nonlocal calls
        calls += 1
        mapped = problem.map(value)
        if calls == 3:
            return replace(
                mapped,
                mapped_x=torch.full_like(mapped.mapped_x, torch.nan),
                spectral=replace(
                    mapped.spectral,
                    density=torch.full_like(mapped.spectral.density, torch.nan),
                ),
            )
        return mapped

    def record(old, mapped, state, settings, step, transition_active=None, damp=None):
        mixed, next_state = original(
            old,
            mapped,
            state,
            settings,
            step,
            transition_active=transition_active,
            damp=damp,
        )
        observed.append((transition_active.clone(), next_state))
        return mixed, next_state

    monkeypatch.setattr(anderson_module, "anderson_mix", record)
    outcome = solve_common(
        case.x,
        AndersonTestSettings(
            maxiter=3,
            mix_guess=False,
            damp=0.4,
            x_atol=0.0,
            x_atol_max=0.0,
        ),
        map_fn,
        problem=problem,
    )
    assert outcome.terminal_failed.item()
    assert len(observed) >= 2
    assert not observed[1][0].item()
    torch.testing.assert_close(observed[1][1].x_hist, observed[0][1].x_hist)
    torch.testing.assert_close(observed[1][1].f_hist, observed[0][1].f_hist)
    assert torch.isfinite(observed[1][1].x_hist).all()
    assert torch.isfinite(observed[1][1].f_hist).all()
