"""Tests for native Fock-space commutator DIIS."""

from __future__ import annotations

from dataclasses import fields
from test.test_scf.test_functional_map import _prepare_case

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.scf.functional.cdiis import (
    CDIISMethod,
    CDIISMethodSettings,
    CDIISState,
    _history_condition,
    _insert_observation,
    _prune_history,
    _solve_pulay_coefficients,
)
from dxtb._src.scf.functional.method import SCFMethodContext
from dxtb._src.scf.functional.problem import (
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
)
from dxtb._src.scf.functional.solver import (
    SCFSolverSettings,
    _solve_map,
    solve_scf_problem,
)
from dxtb._src.scf.functional.stationarity import (
    ao_stationarity_matrix,
    orthogonalized_stationarity_matrix,
)
from dxtb._src.scf.functional.strategy import SCFStage, SCFStrategy


def _physical_fock_case(
    *, dtype: torch.dtype = torch.float64
) -> tuple[SCFProblem, SCFPhysicalState]:
    """Construct an independent H2O/GFN2 Fock observation."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O", dtype=dtype)
    from dxtb._src.scf.functional import fock_initial_guess, unpack_charges

    charges = unpack_charges(case.x, case.inputs.charge_layout)
    initial = fock_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    return problem, problem.evaluate(initial)


def _context(
    physical: SCFPhysicalState,
    proposal_x: torch.Tensor,
    *,
    active: bool = True,
    damp: float = 0.5,
) -> SCFMethodContext:
    """Create a single transition context for method-level tests."""
    return SCFMethodContext(
        step=1,
        transition_active=torch.tensor(active, device=physical.x.device),
        damp=physical.x.new_tensor(damp),
        proposal_x=proposal_x,
    )


def _pulay_state(
    errors: torch.Tensor,
    *,
    count: int | None = None,
) -> CDIISState:
    """Build a fixed-size state fixture from newest-first error matrices."""
    max_history = errors.shape[0]
    focks = torch.zeros_like(errors)
    active_count = max_history if count is None else count
    return CDIISState(
        fock_history=focks,
        error_history=errors,
        history_count=torch.tensor(active_count, dtype=torch.int64),
        restart_count=torch.tensor(0, dtype=torch.int64),
        prune_count=torch.tensor(0, dtype=torch.int64),
        last_condition=torch.tensor(0.0, dtype=errors.dtype),
        last_regularization=torch.tensor(0.0, dtype=errors.dtype),
        last_coeff_max=torch.tensor(0.0, dtype=errors.dtype),
        last_solve_ok=torch.tensor(False),
    )


def test_cdiis_stores_fresh_mapped_fock_commutator() -> None:
    """CDIIS stores the projected fresh Fock and orthogonalized error."""
    problem, physical = _physical_fock_case()
    fresh = physical.map_result.mapped_x
    rebuilt = problem.rebuild_hamiltonian(physical.map_result.spectral.charges)
    torch.testing.assert_close(fresh, rebuilt, rtol=0.0, atol=2.0e-14)

    proposal = CDIISMethod().propose(
        problem, physical, None, _context(physical, fresh)
    )
    assert isinstance(proposal.state, CDIISState)
    state = proposal.state
    expected_ao = ao_stationarity_matrix(
        fresh,
        physical.map_result.spectral.density,
        problem.inputs.integrals.overlap,
        problem.inputs.orbital_mask,
    )
    expected_error = orthogonalized_stationarity_matrix(
        expected_ao,
        problem.inputs.inverse_cholesky,
        problem.inputs.orbital_mask,
    )
    torch.testing.assert_close(state.fock_history[0], fresh)
    torch.testing.assert_close(state.error_history[0], expected_error)
    torch.testing.assert_close(
        proposal.x, physical.x + 0.5 * (fresh - physical.x)
    )
    assert int(state.history_count) == 1
    assert not bool(state.last_solve_ok)


def test_cdiis_reaches_an_s2_certified_fock_root() -> None:
    """Standalone Fock CDIIS uses the shared physical convergence gate."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    from dxtb._src.scf.functional import fock_initial_guess, unpack_charges

    charges = unpack_charges(case.x, case.inputs.charge_layout)
    initial = fock_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    outcome = solve_scf_problem(
        initial,
        problem,
        SCFSolverSettings(
            maxiter=100,
            mix_guess=True,
            damp=1.0,
            x_atol=1.0e-8,
            x_atol_max=1.0e-8,
        ),
        SCFStrategy((SCFStage(CDIISMethod()),)),
    )
    assert bool(outcome.converged)
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.certification is not None
    assert bool(outcome.diagnostics.certification.result.certified)


def test_cdiis_certified_root_enters_history_free_implicit_preparation() -> (
    None
):
    """The implicit boundary receives only the certified physical root."""
    from dxtb._src.scf.functional.implicit import _prepare_scf_root

    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    from dxtb._src.scf.functional import fock_initial_guess, unpack_charges

    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    outcome = _solve_map(
        initial,
        SCFSolverSettings(100, True, 1.0, 1.0e-8, 1.0e-8),
        problem.map,
        eager=True,
        problem=problem,
        strategy=SCFStrategy((SCFStage(CDIISMethod()),)),
    )
    assert bool(outcome.converged)
    assert outcome.certification is not None
    assert bool(outcome.certification.certified)

    def raw_map(root: torch.Tensor) -> torch.Tensor:
        return problem.map(root).mapped_x

    prepared = _prepare_scf_root(
        outcome,
        maxiter=100,
        x_atol=1.0e-8,
        x_atol_max=1.0e-8,
        final_valid=torch.tensor(True),
        raw_map=raw_map,
        parameters=(),
    )
    attached = prepared.attach()
    torch.testing.assert_close(attached, outcome.root_x)
    assert "method_state" not in prepared.__dataclass_fields__


def test_cdiis_rejects_non_fock_representations() -> None:
    """CDIIS refuses charge and potential working variables."""
    problem, physical = _physical_fock_case()
    charge_problem = SCFProblem(problem.inputs, SCFRepresentation.CHARGE)
    from dxtb._src.scf.functional import potential_initial_guess

    charges = physical.map_result.spectral.charges
    _, layout = potential_initial_guess(charges, problem.inputs)
    potential_problem = SCFProblem(
        problem.inputs, SCFRepresentation.POTENTIAL, layout
    )
    method = CDIISMethod()
    for invalid_problem in (charge_problem, potential_problem):
        with pytest.raises(ValueError, match="Fock working representation"):
            method.propose(
                invalid_problem,
                physical,
                None,
                _context(physical, physical.map_result.mapped_x),
            )


def test_cdiis_pulay_solution_matches_regularized_kkt_system() -> None:
    """The affine Pulay solve matches a direct augmented KKT solution."""
    errors = torch.tensor(
        [
            [[1.0, 0.1], [0.2, -0.3]],
            [[0.1, -0.4], [0.6, 0.2]],
            [[-0.2, 0.3], [0.1, 0.8]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    settings = CDIISMethodSettings(
        max_history=4,
        condition_limit=1.0e8,
        relative_regularization=1.0e-10,
    )
    coefficients, regularization, solve_ok = _solve_pulay_coefficients(
        errors, torch.tensor(3), settings
    )
    assert bool(solve_ok)

    flat = errors[:3].reshape(3, -1)
    gram = flat @ flat.T
    scale = torch.diagonal(gram).mean()
    regularized = gram + torch.eye(3, dtype=gram.dtype) * regularization
    ones = torch.ones((3, 1), dtype=gram.dtype)
    kkt = torch.cat(
        (
            torch.cat((regularized, ones), dim=1),
            torch.cat((ones.T, torch.zeros((1, 1), dtype=gram.dtype)), dim=1),
        ),
        dim=0,
    )
    rhs = torch.zeros((4, 1), dtype=gram.dtype)
    rhs[-1] = 1.0
    expected = torch.linalg.solve(kkt, rhs)[:3, 0]
    torch.testing.assert_close(
        coefficients[:3], expected, rtol=1e-11, atol=1e-12
    )
    assert abs(coefficients.sum().item() - 1.0) < 1.0e-12
    assert coefficients[3].item() == 0.0
    assert torch.isfinite(coefficients).all()
    assert regularization.item() > 0.0
    assert torch.isfinite(scale)


def test_cdiis_orthogonal_error_gram_is_invariant_under_basis_rescaling() -> (
    None
):
    """Orthogonal-error Gram, coefficients, and extrapolation transform cleanly."""
    dtype = torch.float64
    generator = torch.Generator().manual_seed(41)
    raw = torch.randn((3, 3), generator=generator, dtype=dtype)
    overlap = raw.T @ raw + 2.0 * torch.eye(3, dtype=dtype)
    values, vectors = torch.linalg.eigh(overlap)
    inverse_sqrt = (vectors * values.rsqrt()) @ vectors.T
    basis = torch.tensor(
        [[1.7, 0.3, -0.1], [0.0, 0.6, 0.2], [0.0, 0.0, 1.3]],
        dtype=dtype,
    )
    inverse_basis = torch.linalg.inv(basis)
    transformed_overlap = basis.T @ overlap @ basis
    transformed_values, transformed_vectors = torch.linalg.eigh(
        transformed_overlap
    )
    transformed_inverse_sqrt = (
        transformed_vectors * transformed_values.rsqrt()
    ) @ transformed_vectors.T

    focks = []
    densities = []
    for index in range(3):
        matrix = torch.randn((3, 3), generator=generator, dtype=dtype)
        focks.append(0.5 * (matrix + matrix.T))
        density = torch.randn((3, 3), generator=generator, dtype=dtype)
        densities.append(0.5 * (density + density.T))

    errors = []
    transformed_errors = []
    transformed_focks = []
    raw_errors = []
    for fock, density in zip(focks, densities, strict=True):
        ao_error = fock @ density @ overlap - overlap @ density @ fock
        transformed_fock = basis.T @ fock @ basis
        transformed_density = inverse_basis @ density @ inverse_basis.T
        transformed_ao = (
            transformed_fock @ transformed_density @ transformed_overlap
            - transformed_overlap @ transformed_density @ transformed_fock
        )
        errors.append(
            orthogonalized_stationarity_matrix(
                ao_error, inverse_sqrt, torch.ones(3, dtype=torch.bool)
            )
        )
        transformed_errors.append(
            orthogonalized_stationarity_matrix(
                transformed_ao,
                transformed_inverse_sqrt,
                torch.ones(3, dtype=torch.bool),
            )
        )
        transformed_focks.append(transformed_fock)
        raw_errors.append(ao_error)

    errors_t = torch.stack(errors)
    transformed_errors_t = torch.stack(transformed_errors)
    gram = errors_t.reshape(3, -1) @ errors_t.reshape(3, -1).T
    transformed_gram = (
        transformed_errors_t.reshape(3, -1)
        @ transformed_errors_t.reshape(3, -1).T
    )
    raw_gram = torch.stack(raw_errors).reshape(3, -1)
    raw_gram = raw_gram @ raw_gram.T
    torch.testing.assert_close(
        gram, transformed_gram, rtol=2.0e-12, atol=2.0e-12
    )
    assert not torch.allclose(gram, raw_gram, rtol=1.0e-4, atol=1.0e-4)

    settings = CDIISMethodSettings(max_history=3)
    coeff, _, ok = _solve_pulay_coefficients(
        errors_t, torch.tensor(3), settings
    )
    transformed_coeff, _, transformed_ok = _solve_pulay_coefficients(
        transformed_errors_t, torch.tensor(3), settings
    )
    assert bool(ok) and bool(transformed_ok)
    torch.testing.assert_close(
        coeff, transformed_coeff, rtol=2.0e-11, atol=2.0e-12
    )
    fock_mix = torch.einsum("i,ijk->jk", coeff, torch.stack(focks))
    transformed_mix = torch.einsum(
        "i,ijk->jk", transformed_coeff, torch.stack(transformed_focks)
    )
    torch.testing.assert_close(
        transformed_mix, basis.T @ fock_mix @ basis, rtol=2.0e-11, atol=2.0e-12
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cdiis_exactly_dependent_history_restarts_safely(
    dtype: torch.dtype,
) -> None:
    """Dependent errors are pruned/restarted without invalid coefficients."""
    first = torch.tensor([[1.0, 0.2], [-0.1, 0.4]], dtype=dtype)
    errors = torch.stack((first, first.clone(), torch.zeros_like(first)))
    state = _pulay_state(errors, count=2)
    settings = CDIISMethodSettings(max_history=3)
    condition, valid = _history_condition(
        state.error_history, state.history_count
    )
    assert bool(valid)
    assert condition > settings.resolved_condition_limit(dtype)

    pruned, count, _, prune_count, restart = _prune_history(
        state, state.history_count, settings
    )
    assert bool(restart)
    assert int(count) == 2
    assert int(prune_count) == 0
    coefficients, regularization, solve_ok = _solve_pulay_coefficients(
        pruned.error_history, count, settings
    )
    assert not bool(solve_ok)
    assert torch.isfinite(coefficients).all()
    assert torch.equal(coefficients, torch.zeros_like(coefficients))
    assert torch.isfinite(regularization)


def test_cdiis_near_dependence_threshold_tracks_dtype() -> None:
    """Float32 rejects weaker secant separation than float64 can resolve."""
    settings = CDIISMethodSettings(max_history=2)
    conditions: dict[torch.dtype, torch.Tensor] = {}
    for dtype in (torch.float64, torch.float32):
        errors = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0e-3]]], dtype=dtype)
        condition, valid = _history_condition(
            errors, torch.tensor(2, dtype=torch.int64)
        )
        assert bool(valid)
        conditions[dtype] = condition
    assert conditions[torch.float64] < settings.resolved_condition_limit(
        torch.float64
    )
    assert conditions[torch.float32] > settings.resolved_condition_limit(
        torch.float32
    )


def test_cdiis_prunes_oldest_and_rolls_at_fixed_capacity() -> None:
    """Condition pruning drops the oldest slot and history has fixed shape."""
    basis = torch.eye(3, dtype=torch.float64)
    errors = torch.stack((basis[0], basis[1], basis[2], basis[0] + basis[1]))
    state = _pulay_state(errors)
    settings = CDIISMethodSettings(max_history=4, condition_limit=1.1)
    pruned, count, _, prune_count, restart = _prune_history(
        state, state.history_count, settings
    )
    assert int(count) == 3
    assert int(prune_count) == 1
    assert pruned.error_history.shape == (4, 3)
    assert not bool(restart)


def test_cdiis_insertion_drops_oldest_at_capacity() -> None:
    """A full fixed-size history shifts newest first and drops its tail."""
    errors = torch.stack(
        [
            torch.full((2, 2), float(index + 1), dtype=torch.float64)
            for index in range(3)
        ]
    )
    state = _pulay_state(errors)
    focks = errors.clone()
    state = CDIISState(
        fock_history=focks,
        error_history=errors,
        history_count=torch.tensor(3),
        restart_count=state.restart_count,
        prune_count=state.prune_count,
        last_condition=state.last_condition,
        last_regularization=state.last_regularization,
        last_coeff_max=state.last_coeff_max,
        last_solve_ok=state.last_solve_ok,
    )
    fresh = torch.eye(2, dtype=torch.float64) * 5.0
    next_state, count, observation_ok, restart = _insert_observation(
        state, fresh, fresh, 3
    )
    assert bool(observation_ok)
    assert not bool(restart)
    assert int(count) == 3
    torch.testing.assert_close(next_state.fock_history[0], fresh)
    torch.testing.assert_close(next_state.fock_history[1], focks[0])
    torch.testing.assert_close(next_state.fock_history[2], focks[1])


def test_cdiis_rejects_zero_and_nonfinite_observations() -> None:
    """Unresolvable stationarity data never enter bounded history."""
    zeros = torch.zeros((3, 2, 2), dtype=torch.float64)
    state = _pulay_state(zeros, count=2)
    finite_fock = torch.eye(2, dtype=torch.float64)
    rejected, count, observation_ok, restart = _insert_observation(
        state, finite_fock, torch.zeros_like(finite_fock), 3
    )
    assert not bool(observation_ok)
    assert bool(restart)
    assert int(count) == 0
    assert torch.equal(rejected.fock_history, zeros)
    assert torch.equal(rejected.error_history, zeros)

    nonfinite = torch.full_like(finite_fock, float("nan"))
    rejected, count, observation_ok, restart = _insert_observation(
        state, nonfinite, finite_fock, 3
    )
    assert not bool(observation_ok)
    assert bool(restart)
    assert int(count) == 0
    assert torch.isfinite(rejected.fock_history).all()
    assert torch.isfinite(rejected.error_history).all()


def test_cdiis_inactive_transition_freezes_every_state_leaf() -> None:
    """An inactive CDIIS lane preserves history and diagnostics exactly."""
    problem, physical = _physical_fock_case()
    method = CDIISMethod()
    fresh = physical.map_result.mapped_x
    first = method.propose(problem, physical, None, _context(physical, fresh))
    assert isinstance(first.state, CDIISState)
    second = method.propose(
        problem,
        physical,
        first.state,
        _context(
            physical, fresh + 0.1 * torch.eye(fresh.shape[-1]), active=False
        ),
    )
    assert isinstance(second.state, CDIISState)
    for field in fields(CDIISState):
        assert torch.equal(
            getattr(first.state, field.name), getattr(second.state, field.name)
        )
    torch.testing.assert_close(second.x, physical.x)


def test_cdiis_pulay_solver_supports_vmap_and_forward_derivatives() -> None:
    """Fixed-shape Pulay algebra works with vmap, JVP, and jacfwd."""
    dtype = torch.float64
    settings = CDIISMethodSettings(max_history=4)

    def solve(scale: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        errors = torch.stack(
            (
                torch.stack((scale, scale * 0.1)),
                torch.stack((scale * 0.2, -scale * 0.7)),
                torch.stack((-scale * 0.4, scale * 0.3)),
                torch.zeros(2, dtype=dtype),
            )
        ).reshape(4, 1, 2)
        coeffs, _, ok = _solve_pulay_coefficients(errors, count, settings)
        return torch.where(ok, coeffs, torch.zeros_like(coeffs))

    scales = torch.tensor([0.8, 1.0, 1.2], dtype=dtype)
    counts = torch.tensor([2, 3, 2], dtype=torch.int64)
    mapped = vmap(solve)(scales, counts)
    serial = torch.stack(
        [
            solve(scale, count)
            for scale, count in zip(scales, counts, strict=True)
        ]
    )
    torch.testing.assert_close(mapped, serial)
    fixed_count = torch.tensor(3, dtype=torch.int64)

    def scalar_solve(scale: torch.Tensor) -> torch.Tensor:
        return solve(scale, fixed_count)

    value, tangent = jvp(
        scalar_solve,
        (torch.tensor(1.0, dtype=dtype),),
        (torch.ones((), dtype=dtype),),
    )
    jacobian = jacfwd(scalar_solve)(torch.tensor(1.0, dtype=dtype))
    scalar = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    first = solve(scalar, fixed_count)[0]
    reverse = torch.autograd.grad(first, scalar, create_graph=True)[0]
    nested = torch.autograd.grad(reverse, scalar)[0]
    assert torch.isfinite(value).all()
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(reverse)
    assert torch.isfinite(nested)


def test_cdiis_state_history_is_fixed_size_and_tensor_only() -> None:
    """The method state stores bounded tensors rather than Python histories."""
    problem, physical = _physical_fock_case()
    proposal = CDIISMethod(CDIISMethodSettings(max_history=6)).propose(
        problem,
        physical,
        None,
        _context(physical, physical.map_result.mapped_x),
    )
    assert isinstance(proposal.state, CDIISState)
    assert proposal.state.fock_history.shape == (6, *physical.x.shape)
    assert proposal.state.error_history.shape == (6, *physical.x.shape)
    assert all(
        isinstance(getattr(proposal.state, field.name), torch.Tensor)
        for field in fields(CDIISState)
    )


def test_cdiis_fixed_trip_fock_controller_supports_vmap_and_jvp() -> None:
    """A short physical CDIIS trajectory is transform-compatible."""
    case = _prepare_case(GFN1_XTB, sample_name="H2")
    from dxtb._src.scf.functional import fock_initial_guess, unpack_charges

    charges = unpack_charges(case.x, case.inputs.charge_layout)
    initial = fock_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    settings = SCFSolverSettings(
        maxiter=4,
        mix_guess=True,
        damp=0.5,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    strategy = SCFStrategy((SCFStage(CDIISMethod()),))

    def solve(fock: torch.Tensor) -> torch.Tensor:
        return _solve_map(
            fock,
            settings,
            problem.map,
            problem=problem,
            strategy=strategy,
        ).root_x

    guesses = torch.stack(
        (
            initial,
            initial
            + 1.0e-3 * torch.eye(initial.shape[-1], dtype=initial.dtype),
        )
    )
    mapped = vmap(solve)(guesses)
    serial = torch.stack([solve(item) for item in guesses])
    torch.testing.assert_close(mapped, serial)
    value, tangent = jvp(solve, (initial,), (torch.ones_like(initial),))
    derivative = jacfwd(solve)(initial)
    assert torch.isfinite(value).all()
    assert torch.isfinite(tangent).all()
    assert torch.isfinite(derivative).all()


def test_cdiis_trace_and_zero_budget_keep_shared_controller_contract() -> None:
    """Trace flattens CDIIS history and zero budget skips method transitions."""
    case = _prepare_case(GFN1_XTB, sample_name="H2")
    from dxtb._src.scf.functional import fock_initial_guess, unpack_charges

    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    strategy = SCFStrategy((SCFStage(CDIISMethod()),))
    traced = _solve_map(
        initial,
        SCFSolverSettings(3, True, 0.5, 0.0, 0.0),
        problem.map,
        problem=problem,
        strategy=strategy,
        record_trace=True,
    )
    assert traced.trace is not None
    assert len(traced.trace.method_state) == len(fields(CDIISState))
    assert traced.trace.method_state[0].shape == (4, 8, *initial.shape)

    zero_budget = _solve_map(
        initial,
        SCFSolverSettings(0, True, 0.5, 0.0, 0.0),
        problem.map,
        problem=problem,
        eager=False,
        strategy=strategy,
    )
    assert int(zero_budget.iterations) == 1
    assert not bool(zero_budget.converged)
    assert zero_budget.trajectory_status is None
    assert zero_budget.certification is None
