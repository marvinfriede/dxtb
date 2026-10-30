"""Transform-gate tests for the private physical-root attachment."""

from __future__ import annotations

import pytest
import torch

from dxtb import labels
from dxtb._src.scf.functional.implicit import (
    _require_linear_convergence,
    implicit_root,
    solve_linear,
)

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}


def _scalar_map(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return torch.tanh(a * x + b)


def _scalar_primal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = torch.zeros_like(b)
        for _ in range(200):
            x = _scalar_map(x, a, b)
    return x


def _scalar_attached(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    root = _scalar_primal(a, b)
    return implicit_root(root, a, b, raw_map=_scalar_map)


def test_scalar_implicit_root_reverse_orders_and_mixed_derivative() -> None:
    a = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)

    attached = _scalar_attached(a, b)
    unrolled = _scalar_primal_unrolled(a, b)
    torch.testing.assert_close(attached, unrolled, atol=1.0e-13, rtol=1.0e-13)

    attached_derivatives = _reverse_derivatives(_scalar_attached, a, b)
    unrolled_derivatives = _reverse_derivatives(_scalar_primal_unrolled, a, b)
    for actual, expected in zip(attached_derivatives, unrolled_derivatives):
        torch.testing.assert_close(actual, expected, atol=2.0e-9, rtol=2.0e-8)

    mixed = torch.autograd.grad(attached_derivatives[1], a, retain_graph=True)[
        0
    ]
    reference_mixed = torch.autograd.grad(
        unrolled_derivatives[1], a, retain_graph=True
    )[0]
    torch.testing.assert_close(mixed, reference_mixed, atol=2.0e-8, rtol=2.0e-7)


def test_implicit_attachment_detaches_the_primal_transition_graph() -> None:
    a = torch.tensor(0.31, dtype=torch.float64)
    b = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)
    unrolled_root = _scalar_primal_unrolled(a, b)
    assert unrolled_root.grad_fn is not None

    attached = implicit_root(unrolled_root, a, b, raw_map=_scalar_map)
    derivative = torch.autograd.grad(attached, b)[0]
    slope = 1.0 - attached.square()
    expected = slope / (1.0 - a * slope)
    torch.testing.assert_close(derivative, expected, atol=1.0e-10, rtol=1.0e-10)


def _scalar_primal_unrolled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(b)
    for _ in range(200):
        x = _scalar_map(x, a, b)
    return x


def _reverse_derivatives(function, a: torch.Tensor, b: torch.Tensor):
    value = function(a, b)
    first = torch.autograd.grad(value, b, create_graph=True, retain_graph=True)[
        0
    ]
    second = torch.autograd.grad(
        first, b, create_graph=True, retain_graph=True
    )[0]
    third = torch.autograd.grad(second, b, retain_graph=True)[0]
    return first, second, third


def test_scalar_implicit_root_jvp_jacfwd_jacrev() -> None:
    a = torch.tensor(0.31, dtype=torch.float64)
    b = torch.tensor(0.27, dtype=torch.float64)
    tangent = (
        torch.tensor(0.4, dtype=torch.float64),
        torch.tensor(-0.2, dtype=torch.float64),
    )

    _, jvp = torch.func.jvp(_scalar_attached, (a, b), tangent)
    jacfwd = torch.func.jacfwd(_scalar_attached, argnums=(0, 1))(a, b)
    jacrev = torch.func.jacrev(_scalar_attached, argnums=(0, 1))(a, b)
    expected = jacfwd[0] * tangent[0] + jacfwd[1] * tangent[1]
    torch.testing.assert_close(jvp, expected, atol=1.0e-10, rtol=1.0e-9)
    torch.testing.assert_close(jacfwd[0], jacrev[0], atol=1.0e-10, rtol=1.0e-9)
    torch.testing.assert_close(jacfwd[1], jacrev[1], atol=1.0e-10, rtol=1.0e-9)


def test_scalar_implicit_root_nested_forward_and_reverse_modes() -> None:
    """Every second-order nesting, including forward-over-forward."""
    a = torch.tensor(0.31, dtype=torch.float64)
    b = torch.tensor(0.27, dtype=torch.float64)

    def attached(value: torch.Tensor) -> torch.Tensor:
        return _scalar_attached(a, value)

    def unrolled(value: torch.Tensor) -> torch.Tensor:
        return _scalar_primal_unrolled(a, value)

    jacrev, jacfwd = torch.func.jacrev, torch.func.jacfwd
    expected = jacrev(jacrev(unrolled))(b)
    for nested in (
        jacrev(jacrev(attached)),
        jacfwd(jacrev(attached)),
        jacrev(jacfwd(attached)),
        jacfwd(jacfwd(attached)),
    ):
        torch.testing.assert_close(
            nested(b), expected, atol=1.0e-10, rtol=1.0e-9
        )

    # All eight third-order nestings, including jacfwd(jacfwd(jacfwd)).
    expected = jacrev(jacrev(jacrev(unrolled)))(b)
    for outer in (jacrev, jacfwd):
        for middle in (jacrev, jacfwd):
            for inner in (jacrev, jacfwd):
                third = outer(middle(inner(attached)))(b)
                torch.testing.assert_close(
                    third, expected, atol=1.0e-9, rtol=1.0e-8
                )


def test_nonsymmetric_vector_tangent_and_adjoint_match_dense_reference() -> (
    None
):
    matrix = torch.tensor([[0.18, 0.31], [-0.07, 0.22]], dtype=torch.float64)
    parameter_map = torch.tensor([[0.7, -0.2], [0.1, 0.5]], dtype=torch.float64)

    def raw_map(x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        return matrix @ x + parameter_map @ parameter

    parameter = torch.tensor(
        [0.3, -0.4], dtype=torch.float64, requires_grad=True
    )
    root = torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) - matrix,
        parameter_map @ parameter,
    )
    attached = implicit_root(root, parameter, raw_map=raw_map)
    direction = torch.tensor([-0.2, 0.6], dtype=torch.float64)
    _, tangent = torch.func.jvp(
        lambda p: implicit_root(root, p, raw_map=raw_map),
        (parameter,),
        (direction,),
    )
    expected_tangent = torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) - matrix,
        parameter_map @ direction,
    )
    torch.testing.assert_close(
        tangent, expected_tangent, atol=1.0e-10, rtol=1.0e-10
    )

    cotangent = torch.tensor([0.4, -0.9], dtype=torch.float64)
    parameter_gradient = torch.autograd.grad(
        torch.dot(attached, cotangent), parameter
    )[0]
    expected_adjoint = torch.linalg.solve(
        (torch.eye(2, dtype=torch.float64) - matrix).mT, cotangent
    )
    expected_gradient = parameter_map.mT @ expected_adjoint
    torch.testing.assert_close(
        parameter_gradient, expected_gradient, atol=1.0e-10, rtol=1.0e-10
    )


def test_matrix_free_linear_solve_matches_dense_nonsymmetric_system() -> None:
    matrix = torch.tensor(
        [[2.0, 0.7, -0.1], [-0.2, 1.4, 0.3], [0.4, 0.1, 1.8]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([1.0, -0.3, 0.8], dtype=torch.float64)
    result = solve_linear(lambda value: matrix @ value, rhs, maxiter=12)
    expected = torch.linalg.solve(matrix, rhs)
    torch.testing.assert_close(
        result.solution, expected, atol=1.0e-10, rtol=1.0e-10
    )
    assert bool(result.converged)
    assert result.residual < 1.0e-10

    transpose_result = solve_linear(
        lambda value: matrix.mT @ value, rhs, maxiter=12
    )
    transpose_expected = torch.linalg.solve(matrix.mT, rhs)
    torch.testing.assert_close(
        transpose_result.solution,
        transpose_expected,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert bool(transpose_result.converged)
    assert transpose_result.residual < 1.0e-10


def test_matrix_free_linear_solve_matches_float32_dense_reference() -> None:
    """Check dtype-aware convergence against a nonsymmetric dense solve."""
    matrix = torch.tensor(
        [
            [2.0, 0.7, -0.1, 0.2],
            [-0.2, 1.4, 0.3, -0.1],
            [0.4, 0.1, 1.8, 0.2],
            [0.05, -0.3, 0.15, 1.6],
        ],
        dtype=torch.float32,
    )
    rhs = torch.tensor([1.0, -0.3, 0.8, 0.2], dtype=torch.float32)
    result = solve_linear(lambda value: matrix @ value, rhs, maxiter=16)
    expected = torch.linalg.solve(matrix, rhs)

    assert bool(result.converged)
    assert torch.isfinite(result.solution).all()
    assert torch.isfinite(result.residual)
    torch.testing.assert_close(
        result.solution, expected, atol=2.0e-5, rtol=2.0e-5
    )
    assert result.residual <= 10.0 * torch.finfo(
        torch.float32
    ).eps * torch.linalg.vector_norm(rhs)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_unconverged_linear_solve_is_explicit_and_never_used_as_a_derivative(
    dtype: torch.dtype,
) -> None:
    """Reject singular systems and preserve their NaN derivative signal."""
    rhs = torch.tensor([1.0], dtype=dtype)
    failed = solve_linear(lambda value: torch.zeros_like(value), rhs, maxiter=2)
    assert not bool(failed.converged)
    assert failed.residual > 0.9

    def singular_map(x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        return x + parameter

    root = torch.zeros((), dtype=dtype)
    parameter = torch.zeros((), dtype=dtype, requires_grad=True)
    attached = implicit_root(root, parameter, raw_map=singular_map)

    reverse = torch.autograd.grad(attached, parameter)[0]
    assert not bool(torch.isfinite(reverse))
    forward_jacobian = torch.func.jacfwd(
        lambda value: implicit_root(root, value, raw_map=singular_map)
    )(parameter.detach())
    assert not bool(torch.isfinite(forward_jacobian))

    _, tangent = torch.func.jvp(
        lambda value: implicit_root(root, value, raw_map=singular_map),
        (parameter.detach(),),
        (torch.ones_like(parameter),),
    )
    assert not bool(torch.isfinite(tangent))


def test_float32_scalar_implicit_root_reverse_jvp_and_jacfwd() -> None:
    """Check float32 implicit transforms against the analytic tangent."""
    a = torch.tensor(0.31, dtype=torch.float32)
    b = torch.tensor(0.27, dtype=torch.float32, requires_grad=True)
    root = _scalar_primal(a, b)
    attached = implicit_root(root, a, b, raw_map=_scalar_map)
    derivative = torch.autograd.grad(attached, b)[0]
    slope = 1.0 - attached.square()
    expected = slope / (1.0 - a * slope)
    _, tangent = torch.func.jvp(
        lambda value: _scalar_attached_float32(a, value),
        (b.detach(),),
        (torch.ones_like(b),),
    )
    jacobian = torch.func.jacfwd(
        lambda value: _scalar_attached_float32(a, value)
    )(b.detach())

    for actual in (derivative, tangent, jacobian):
        assert torch.isfinite(actual)
        torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def _scalar_attached_float32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    root = _scalar_primal(a, b)
    return implicit_root(root, a, b, raw_map=_scalar_map)


def test_linear_solve_failure_signal_is_vmap_compatible() -> None:
    matrices = torch.tensor([[[2.0]], [[0.0]]], dtype=torch.float64)
    rhs = torch.ones((2, 1), dtype=torch.float64)

    def response(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        result = solve_linear(lambda value: matrix @ value, vector, maxiter=2)
        return _require_linear_convergence(result)

    values = torch.func.vmap(response)(matrices, rhs)
    torch.testing.assert_close(
        values[0], torch.tensor([0.5], dtype=torch.float64)
    )
    assert not bool(torch.isfinite(values[1]).all())


def test_implicit_primitive_has_no_graph_or_ad_mode_discovery() -> None:
    from pathlib import Path

    source_path = (
        Path(__file__).parents[2] / "src/dxtb/_src/scf/functional/implicit.py"
    )
    source = source_path.read_text()
    for forbidden in (
        "gc.get_referents",
        "_grad_leaves",
        "_reaches_grad",
        "requires_grad",
        "torch.is_grad_enabled",
        "_AdjointSolve",
    ):
        assert forbidden not in source


def test_prepared_roots_vmap_matches_loop_and_nonleaf_input_gradient() -> None:
    a = torch.tensor(0.24, dtype=torch.float64)
    parameters = torch.tensor([0.1, -0.2, 0.4], dtype=torch.float64)
    roots = torch.stack([_scalar_primal(a, value) for value in parameters])

    def response(root: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return implicit_root(root, a, value, raw_map=_scalar_map)

    vmapped = torch.func.vmap(response)(roots, parameters)
    looped = torch.stack(
        [response(root, value) for root, value in zip(roots, parameters)]
    )
    torch.testing.assert_close(vmapped, looped)

    directions = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)

    def tangent_response(
        root: torch.Tensor, value: torch.Tensor, tangent: torch.Tensor
    ):
        return torch.func.jvp(
            lambda current: response(root, current),
            (value,),
            (tangent,),
        )[1]

    vmapped_tangent = torch.func.vmap(tangent_response)(
        roots, parameters, directions
    )
    looped_tangent = torch.stack(
        [
            tangent_response(root, value, tangent)
            for root, value, tangent in zip(roots, parameters, directions)
        ]
    )
    torch.testing.assert_close(vmapped_tangent, looped_tangent)

    base = torch.tensor(0.13, dtype=torch.float64)
    delta = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    nonleaf = base + delta
    root = _scalar_primal(a, nonleaf)
    output = implicit_root(root, a, nonleaf, raw_map=_scalar_map)
    derivative = torch.autograd.grad(output, delta)[0]
    occupancy_slope = 1.0 - output.square()
    expected = occupancy_slope / (1.0 - a * occupancy_slope)
    torch.testing.assert_close(derivative, expected, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize(
    ("representation", "temperature"),
    [
        (0, 0.0),
        (1, 0.0),
        (2, 0.0),
        (0, 300.0),
    ],
    ids=("charge", "potential", "fock", "charge-300K"),
)
def test_scf_input_tree_uses_explicit_integral_tensors_for_raw_map(
    representation: int, temperature: float
) -> None:
    from test.test_scf.test_functional_root_derivatives import (
        _case,
        _initial_state,
        _map_fn,
        _run_root,
    )

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional.implicit import _SCFMapTensorTree

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=representation,
        tolerance=1.0e-12,
        temperature=temperature,
    )
    root_record = _run_root(
        case,
        case.inputs,
        tolerance=1.0e-12,
        maxiter=100,
        eager=True,
    )
    assert root_record.converged
    _, layout = _initial_state(case, case.inputs)
    adapter, parameters = _SCFMapTensorTree.prepare(
        case.inputs, representation, layout
    )
    expected = _map_fn(case, case.inputs, layout)(root_record.root)
    actual = adapter.evaluate(root_record.root, *parameters)
    torch.testing.assert_close(actual.mapped_x, expected.mapped_x)
    torch.testing.assert_close(actual.density, expected.density)

    hcore_index = adapter.parameter_paths.index("integrals.hcore")
    base_hcore = parameters[hcore_index]
    direction = torch.zeros_like(base_hcore)
    direction[0, 0] = 1.0
    changed = tuple(parameters)
    origin = base_hcore.new_zeros(())
    unit = base_hcore.new_ones(())

    def raw(state: torch.Tensor, increment: torch.Tensor) -> torch.Tensor:
        updated = (
            changed[:hcore_index]
            + (base_hcore + increment * direction,)
            + changed[hcore_index + 1 :]
        )
        return adapter(state, *updated)

    _, attached_tangent = torch.func.jvp(
        lambda increment: implicit_root(
            root_record.root,
            *(
                changed[:hcore_index]
                + (base_hcore + increment * direction,)
                + changed[hcore_index + 1 :]
            ),
            raw_map=adapter,
        ),
        (origin,),
        (unit,),
    )
    state_jacobian = torch.func.jacrev(lambda state: raw(state, origin))(
        root_record.root
    )
    parameter_response = torch.func.jvp(
        lambda increment: raw(root_record.root, increment), (origin,), (unit,)
    )[1]
    identity = torch.eye(root_record.root.numel(), **_DD)
    tangent_matrix = identity - state_jacobian.reshape(identity.shape)
    tangent_rhs = parameter_response.reshape(-1)
    expected_tangent = torch.linalg.solve(
        tangent_matrix,
        tangent_rhs,
    ).reshape_as(root_record.root)
    tangent_solve = solve_linear(
        lambda value: tangent_matrix @ value, tangent_rhs
    )
    assert bool(tangent_solve.converged)
    assert float(tangent_solve.residual) < 1.0e-9
    torch.testing.assert_close(
        tangent_solve.solution,
        expected_tangent.reshape(-1),
        atol=1.0e-9,
        rtol=1.0e-8,
    )

    adjoint_rhs = torch.arange(
        1, root_record.root.numel() + 1, dtype=torch.float64
    )
    adjoint_reference = torch.linalg.solve(tangent_matrix.mT, adjoint_rhs)
    adjoint_solve = solve_linear(
        lambda value: tangent_matrix.mT @ value,
        adjoint_rhs,
    )
    assert bool(adjoint_solve.converged)
    assert float(adjoint_solve.residual) < 1.0e-9
    torch.testing.assert_close(
        adjoint_solve.solution,
        adjoint_reference,
        atol=1.0e-9,
        rtol=1.0e-8,
    )
    torch.testing.assert_close(
        attached_tangent, expected_tangent, atol=1.0e-9, rtol=1.0e-8
    )


@pytest.mark.parametrize(
    "representation",
    [0, 1, 2],
    ids=("charge", "potential", "fock"),
)
@pytest.mark.parametrize(
    "mixer",
    [labels.MIXER_ANDERSON, labels.MIXER_BROYDEN],
    ids=("anderson", "broyden"),
)
def test_prepared_h2_implicit_route_matches_direct_energy_fd(
    representation: int,
    mixer: int,
) -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    field = torch.zeros(3, **_DD, requires_grad=True)
    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=representation,
        mixer=mixer,
        tolerance=1.0e-12,
        temperature=0.0,
        field=field,
    )
    config = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        mixer=mixer,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    result = run_prepared_functional_implicit(
        case.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
    )
    assert bool(result.converged)
    from test.test_scf.test_functional_root_derivatives import (
        _initial_state,
        _map_fn,
    )

    _, layout = _initial_state(case, case.inputs)
    paired_map = _map_fn(case, case.inputs, layout)(result.root_x)
    torch.testing.assert_close(paired_map.mapped_x, result.map_result.mapped_x)
    root_gradient = torch.autograd.grad(
        result.root_x.square().sum(), field, retain_graph=True
    )[0]
    assert torch.isfinite(root_gradient).all()
    assert torch.linalg.vector_norm(root_gradient) > 0.0
    observable = result.energy.sum() + result.fenergy.sum()
    gradient = torch.autograd.grad(observable, field)[0][1]

    step = torch.tensor(2.0e-4, **_DD)
    energies = []
    for sign in (-1.0, 1.0):
        perturbed_field = torch.zeros(3, **_DD)
        perturbed_field[1] = sign * step
        perturbed = _case(
            "H2",
            method=GFN1_XTB,
            representation=representation,
            mixer=mixer,
            tolerance=1.0e-12,
            temperature=0.0,
            field=perturbed_field,
        )
        perturbed_config = replace(
            perturbed.system.config.scf,
            scf_mode=labels.SCF_MODE_IMPLICIT,
            mixer=mixer,
            maxiter=100,
            x_atol=1.0e-12,
            x_atol_max=1.0e-12,
        )
        perturbed_result = run_prepared_functional_implicit(
            perturbed.numbers,
            perturbed.initial_charges,
            perturbed.n0,
            perturbed.occupation,
            perturbed.nel,
            perturbed.interactions,
            perturbed.interaction_data,
            perturbed.system.ihelp,
            perturbed_config,
            perturbed.integrals,
        )
        energies.append(
            perturbed_result.energy.sum() + perturbed_result.fenergy.sum()
        )
    finite_difference = (energies[1] - energies[0]) / (2.0 * step)
    torch.testing.assert_close(
        gradient, finite_difference, atol=2.0e-5, rtol=2.0e-4
    )


def test_final_map_is_recomputed_from_the_detached_accepted_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf import functional as functional_package
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        mixer=labels.MIXER_LINEAR,
        tolerance=1.0e-12,
        temperature=0.0,
    )
    assert case.system.h0_setup is not None
    config = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    solver_module = functional_package.eager
    original_solve_map = solver_module._solve_map
    outcomes = []

    def capture_outcome(*args, **kwargs):
        outcome = original_solve_map(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(solver_module, "_solve_map", capture_outcome)
    result = run_prepared_functional_implicit(
        case.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
    )
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert result.map_result is not outcome.map_result
    torch.testing.assert_close(
        result.map_result.mapped_x,
        outcome.map_result.mapped_x,
        atol=1.0e-13,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        result.map_result.mapped_x - outcome.map_result.mapped_x,
        torch.zeros_like(result.map_result.mapped_x),
        atol=1.0e-13,
        rtol=0.0,
    )


def test_implicit_h2_response_is_independent_of_primal_mixer() -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    responses = []
    for mixer in (labels.MIXER_LINEAR, labels.MIXER_ANDERSON):
        field = torch.zeros(3, **_DD, requires_grad=True)
        case = _case(
            "H2",
            method=GFN1_XTB,
            representation=labels.SCP_MODE_CHARGE,
            mixer=mixer,
            tolerance=1.0e-12,
            temperature=0.0,
            field=field,
        )
        config = replace(
            case.system.config.scf,
            scf_mode=labels.SCF_MODE_IMPLICIT,
            maxiter=300,
            x_atol=1.0e-12,
            x_atol_max=1.0e-12,
        )
        result = run_prepared_functional_implicit(
            case.numbers,
            case.initial_charges,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
        )
        assert bool(result.converged)
        observable = result.energy.sum() + result.fenergy.sum()
        responses.append(torch.autograd.grad(observable, field)[0][1])

    torch.testing.assert_close(
        responses[0], responses[1], atol=2.0e-8, rtol=2.0e-7
    )


def test_implicit_reference_occupation_parameter_matches_tight_unroll_fd() -> (
    None
):
    """The explicit model-derived n0 input keeps its IFT response."""
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case, _run_root

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-12,
        temperature=0.0,
    )
    config = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    direction = torch.zeros_like(case.n0)
    direction[0] = 0.2
    delta = torch.zeros((), **_DD, requires_grad=True)
    n0 = case.n0 + delta * direction
    result = run_prepared_functional_implicit(
        case.numbers,
        case.initial_charges,
        n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
    )
    response = torch.autograd.grad(result.energy.sum(), delta)[0]

    step = torch.tensor(2.0e-4, **_DD)
    energies = []
    for sign in (-1.0, 1.0):
        perturbed_inputs = replace(
            case.inputs,
            n0=case.n0 + sign * step * direction,
        )
        record = _run_root(
            case,
            perturbed_inputs,
            tolerance=1.0e-12,
            maxiter=100,
            eager=True,
        )
        energies.append(record.finalized_energy.sum())
    finite_difference = (energies[1] - energies[0]) / (2.0 * step)
    torch.testing.assert_close(
        response, finite_difference, atol=2.0e-5, rtol=3.0e-4
    )


def test_implicit_preparation_rejects_nonconverged_and_one_shot_roots() -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    import pytest

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
    from dxtb._src.typing.exceptions import SCFConvergenceError

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-14,
        maxiter=10,
    )
    for maxiter in (0, 1):
        for force_convergence in (False, True):
            config = replace(
                case.system.config.scf,
                scf_mode=labels.SCF_MODE_IMPLICIT,
                maxiter=maxiter,
                x_atol=1.0e-14,
                x_atol_max=1.0e-14,
                force_convergence=force_convergence,
                strict=force_convergence,
            )
            with pytest.raises(SCFConvergenceError):
                run_prepared_functional_implicit(
                    case.numbers,
                    case.initial_charges,
                    case.n0,
                    case.occupation,
                    case.nel,
                    case.interactions,
                    case.interaction_data,
                    case.system.ihelp,
                    config,
                    case.integrals,
                )

    from test.test_scf.test_functional_map import _prepare_case

    invalid = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        charge=-10.0,
        scf_options={
            "scp_mode": labels.SCP_MODE_CHARGE,
            "mixer": labels.MIXER_ANDERSON,
            "fermi_etemp": 300.0,
            "maxiter": 100,
            "x_atol": 1.0e-12,
            "x_atol_max": 1.0e-12,
        },
    )
    invalid_config = replace(
        invalid.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    with pytest.raises(ValueError, match="Number of electrons exceeds"):
        run_prepared_functional_implicit(
            invalid.system.numbers,
            invalid.initial_charges,
            invalid.n0,
            invalid.occupation,
            invalid.nel,
            invalid.interactions,
            invalid.interaction_data,
            invalid.system.ihelp,
            invalid_config,
            invalid.integrals,
        )


def test_public_singlepoint_keeps_implicit_mode_on_legacy_route() -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    from dxtb import GFN1_XTB, labels

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-12,
        temperature=0.0,
    )
    scf = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    system = replace(
        case.system,
        config=replace(case.system.config, scf=scf),
    )
    positions = case.positions.clone().requires_grad_(True)
    field = torch.zeros(3, **_DD, requires_grad=True)
    result = system.singlepoint(positions, field=field)
    # System.singlepoint uses the fixed-trip boundary only for FULL mode.
    # The implicit compatibility engine returns the legacy mapping, whose
    # optional convergence metadata remains None under the B5.6 contract.
    assert result.converged is None
    assert result.residual is None
    assert result.status is None
    field_gradient = torch.autograd.grad(
        result.energy.sum(), field, create_graph=True, retain_graph=True
    )[0]
    geometry_gradient = torch.autograd.grad(
        result.energy.sum(), positions, retain_graph=True
    )[0]
    mixed = torch.autograd.grad(field_gradient[1], positions)[0][0, 0]
    assert torch.isfinite(field_gradient).all()
    assert torch.isfinite(geometry_gradient).all()
    assert torch.isfinite(mixed)

    step = torch.tensor(1.0e-4, **_DD)
    energy_values = []
    field_responses = []
    for sign in (-1.0, 1.0):
        perturbed_positions = case.positions.clone()
        perturbed_positions[0, 0] += sign * step
        perturbed_field = torch.zeros(3, **_DD, requires_grad=True)
        perturbed = system.singlepoint(
            perturbed_positions, field=perturbed_field
        )
        energy_values.append(perturbed.energy.sum())
        field_responses.append(
            torch.autograd.grad(perturbed.energy.sum(), perturbed_field)[0][1]
        )
    geometry_fd = (energy_values[1] - energy_values[0]) / (2.0 * step)
    mixed_fd = (field_responses[1] - field_responses[0]) / (2.0 * step)
    torch.testing.assert_close(
        geometry_gradient[0, 0], geometry_fd, atol=2.0e-5, rtol=3.0e-4
    )
    torch.testing.assert_close(mixed, mixed_fd, atol=5.0e-4, rtol=2.0e-3)


def test_public_functional_implicit_h2_float32_response() -> None:
    """Check a public float32 implicit response against functional unrolling."""
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    from dxtb import GFN1_XTB, labels

    tolerance = 1.0e-5
    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=tolerance,
        temperature=0.0,
        field=torch.tensor([0.0, 1.0e-2, 0.0], dtype=torch.float32),
        dtype=torch.float32,
    )

    def energy_and_response(mode: int) -> tuple[torch.Tensor, torch.Tensor]:
        scf = replace(
            case.system.config.scf,
            scf_mode=mode,
            mixer=labels.MIXER_ANDERSON,
            maxiter=100,
            x_atol=tolerance,
            x_atol_max=tolerance,
        )
        system = replace(
            case.system,
            config=replace(case.system.config, scf=scf),
        )
        field = torch.tensor(
            [0.0, 1.0e-2, 0.0], dtype=torch.float32, requires_grad=True
        )
        result = system.singlepoint(case.positions, field=field)
        if mode == labels.SCF_MODE_IMPLICIT:
            assert result.converged is None
            assert result.residual is None
            assert result.status is None
        else:
            assert result.converged is not None and bool(result.converged)
        response = torch.autograd.grad(result.energy.sum(), field)[0][1]
        assert torch.isfinite(result.energy).all()
        assert torch.isfinite(response)
        return result.energy, response

    implicit_energy, implicit_response = energy_and_response(
        labels.SCF_MODE_IMPLICIT
    )
    unrolled_energy, unrolled_response = energy_and_response(
        labels.SCF_MODE_FULL
    )
    torch.testing.assert_close(
        implicit_energy, unrolled_energy, atol=2.0e-5, rtol=2.0e-5
    )
    torch.testing.assert_close(
        implicit_response, unrolled_response, atol=2.0e-4, rtol=2.0e-3
    )


def test_implicit_mixer_policy_selects_native_broyden_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case
    from types import SimpleNamespace

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional import data as functional_data
    from dxtb._src.scf.iterator import _use_functional_implicit_route
    from dxtb.config import ConfigSCF

    case = _case("H2", method=GFN1_XTB, temperature=0.0)
    base = replace(case.system.config.scf, scf_mode=labels.SCF_MODE_IMPLICIT)
    default = ConfigSCF()
    assert default.mixer == labels.MIXER_BROYDEN

    # Supported charge/potential implicit calls honor the configured mixer.
    for strict in (False, True):
        broyden = replace(base, mixer=labels.MIXER_BROYDEN, strict=strict)
        assert _use_functional_implicit_route(
            replace(broyden, mixer=default.mixer),
            case.interactions,
            case.system.ihelp,
            case.numbers,
            case.integrals,
            legacy_engine_overrides=False,
        )
        assert _use_functional_implicit_route(
            broyden,
            case.interactions,
            case.system.ihelp,
            case.numbers,
            case.integrals,
            legacy_engine_overrides=False,
        )
        for mixer in (labels.MIXER_LINEAR, labels.MIXER_ANDERSON):
            assert _use_functional_implicit_route(
                replace(base, mixer=mixer, strict=strict),
                case.interactions,
                case.system.ihelp,
                case.numbers,
                case.integrals,
                legacy_engine_overrides=False,
            )

    fock_broyden = replace(
        base, mixer=labels.MIXER_BROYDEN, scp_mode=labels.SCP_MODE_FOCK
    )
    assert _use_functional_implicit_route(
        fock_broyden,
        case.interactions,
        case.system.ihelp,
        case.numbers,
        case.integrals,
        legacy_engine_overrides=False,
    )

    assert not _use_functional_implicit_route(
        replace(base, mixer=labels.MIXER_ANDERSON),
        case.interactions,
        case.system.ihelp,
        case.numbers,
        case.integrals,
        legacy_engine_overrides=True,
    )
    assert not _use_functional_implicit_route(
        replace(base, mixer=labels.MIXER_ANDERSON),
        case.interactions,
        SimpleNamespace(batch_mode=1),
        case.numbers,
        case.integrals,
        legacy_engine_overrides=False,
    )
    monkeypatch.setattr(
        functional_data,
        "functional_scf_interactions_eligible",
        lambda _: (False, "test extension"),
    )
    assert not _use_functional_implicit_route(
        replace(base, mixer=labels.MIXER_ANDERSON),
        case.interactions,
        case.system.ihelp,
        case.numbers,
        case.integrals,
        legacy_engine_overrides=False,
    )
    monkeypatch.undo()
    assert not _use_functional_implicit_route(
        replace(base, mixer=labels.MIXER_ANDERSON),
        case.interactions,
        case.system.ihelp,
        case.numbers.unsqueeze(0),
        case.integrals,
        legacy_engine_overrides=False,
    )
    assert not _use_functional_implicit_route(
        replace(base, mixer=labels.MIXER_ANDERSON),
        case.interactions,
        case.system.ihelp,
        case.numbers,
        SimpleNamespace(hcore=None, overlap=None),
        legacy_engine_overrides=False,
    )
    assert not _use_functional_implicit_route(
        replace(base, mixer=labels.MIXER_ANDERSON, maxiter=-1),
        case.interactions,
        case.system.ihelp,
        case.numbers,
        case.integrals,
        legacy_engine_overrides=False,
    )
    assert not _use_functional_implicit_route(
        replace(
            base,
            mixer=labels.MIXER_ANDERSON,
            fermi=replace(base.fermi, etemp=-1.0),
        ),
        case.interactions,
        case.system.ihelp,
        case.numbers,
        case.integrals,
        legacy_engine_overrides=False,
    )
    assert not _use_functional_implicit_route(
        replace(
            base,
            mixer=labels.MIXER_ANDERSON,
            fermi=replace(base.fermi, maxiter=-1),
        ),
        case.interactions,
        case.system.ihelp,
        case.numbers,
        case.integrals,
        legacy_engine_overrides=False,
    )


def test_public_implicit_broyden_default_matches_functional_forward_values() -> (
    None
):
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    from dxtb import GFN1_XTB, labels

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-12,
        temperature=0.0,
    )
    legacy_config = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        mixer=labels.MIXER_BROYDEN,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    functional_config = replace(legacy_config, mixer=labels.MIXER_ANDERSON)
    functional_system = replace(
        case.system, config=replace(case.system.config, scf=functional_config)
    )
    functional = functional_system.singlepoint(
        case.positions, field=torch.zeros(3, **_DD)
    )

    broyden_results = []
    for strict in (False, True):
        broyden_system = replace(
            case.system,
            config=replace(
                case.system.config,
                scf=replace(legacy_config, strict=strict),
            ),
        )
        broyden_results.append(
            broyden_system.singlepoint(
                case.positions, field=torch.zeros(3, **_DD)
            )
        )

    def assert_tree_close(actual, expected) -> None:
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(
                actual, expected, atol=2.0e-8, rtol=2.0e-7
            )
        elif hasattr(actual, "__dataclass_fields__"):
            for field in actual.__dataclass_fields__:
                assert_tree_close(
                    getattr(actual, field), getattr(expected, field)
                )
        elif actual is None:
            assert expected is None
        else:
            assert actual == expected

    for name in (
        "energy",
        "fenergy",
        "charges",
        "density",
        "potential",
        "emo",
        "occupation",
    ):
        for broyden in broyden_results:
            assert_tree_close(getattr(functional, name), getattr(broyden, name))


def test_iterator_routes_implicit_broyden_away_from_legacy_rootfinder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eligible iterator route prepares a native Broyden physical root."""
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    import dxtb._src.scf.implicit as legacy_implicit
    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.iterator import solve

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        mixer=labels.MIXER_BROYDEN,
        tolerance=1.0e-10,
        maxiter=100,
        temperature=0.0,
    )
    assert case.system.h0_setup is not None

    def unexpected_legacy_root(*args, **kwargs):
        raise AssertionError(
            "eligible implicit Broyden reached the legacy rootfinder"
        )

    monkeypatch.setattr(
        legacy_implicit, "SelfConsistentFieldImplicit", unexpected_legacy_root
    )
    config = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        mixer=labels.MIXER_BROYDEN,
        maxiter=100,
        x_atol=1.0e-10,
        x_atol_max=1.0e-10,
    )
    result = solve(
        case.numbers,
        case.positions,
        case.charge,
        case.spin,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
        case.system.h0_setup.refocc,
    )
    assert bool(result.converged)


def test_h2_implicit_higher_responses_match_tight_unrolled_reference() -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import (
        _case,
        _run_root,
    )

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    def derivatives(energy: torch.Tensor, field: torch.Tensor):
        first = torch.autograd.grad(
            energy, field, create_graph=True, retain_graph=True
        )[0][1]
        second = torch.autograd.grad(
            first, field, create_graph=True, retain_graph=True
        )[0][1]
        third = torch.autograd.grad(second, field)[0][1]
        return first, second, third

    implicit_field = torch.zeros(3, **_DD, requires_grad=True)
    implicit_case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-12,
        temperature=0.0,
        field=implicit_field,
    )
    implicit_config = replace(
        implicit_case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    implicit_result = run_prepared_functional_implicit(
        implicit_case.numbers,
        implicit_case.initial_charges,
        implicit_case.n0,
        implicit_case.occupation,
        implicit_case.nel,
        implicit_case.interactions,
        implicit_case.interaction_data,
        implicit_case.system.ihelp,
        implicit_config,
        implicit_case.integrals,
    )
    implicit_derivatives = derivatives(
        implicit_result.energy.sum(), implicit_field
    )

    unrolled_field = torch.zeros(3, **_DD, requires_grad=True)
    unrolled_case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-12,
        temperature=0.0,
        field=unrolled_field,
    )
    unrolled = _run_root(
        unrolled_case,
        unrolled_case.inputs,
        tolerance=1.0e-12,
        maxiter=100,
        eager=True,
    )
    unrolled_derivatives = derivatives(
        unrolled.finalized_energy.sum(), unrolled_field
    )
    for actual, expected in zip(implicit_derivatives, unrolled_derivatives):
        torch.testing.assert_close(actual, expected, atol=3.0e-6, rtol=3.0e-5)


def test_finite_temperature_free_energy_runs_only_during_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from test.test_scf.test_functional_root_derivatives import _case

    from dxtb import GFN1_XTB, labels
    from dxtb._src.scf.functional import energy
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-12,
        temperature=300.0,
        field=torch.zeros(3, **_DD),
    )
    config = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    calls = 0

    def counting(original):
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        return counted

    # the default equal partition evaluates only the trace
    for name in ("fermi_free_energy_matrix", "fermi_free_energy_trace"):
        monkeypatch.setattr(energy, name, counting(getattr(energy, name)))
    electron_count = case.nel.clone().requires_grad_(True)
    result = run_prepared_functional_implicit(
        case.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        electron_count,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
    )
    assert calls == 1
    assert torch.isfinite(result.fenergy).all()
    torch.autograd.grad(
        result.energy.sum() + result.fenergy.sum(), electron_count
    )
    assert calls == 1


@pytest.mark.slow
@pytest.mark.parametrize(
    "mixer",
    [labels.MIXER_ANDERSON, labels.MIXER_BROYDEN],
    ids=("anderson", "broyden"),
)
def test_no2_potential_implicit_response_matches_dense_ift_and_physical_fd(
    mixer: int,
) -> None:
    from test.test_scf.test_functional_root_derivatives import (
        _case,
        _initial_state,
        _inputs_at_field,
        _run_root,
    )

    from dxtb import GFN2_XTB, labels
    from dxtb._src.scf.functional.implicit import _SCFMapTensorTree
    from dxtb._src.scf.functional.solver import finalize_scf_potential

    tolerance = 1.0e-12
    origin = torch.zeros((), **_DD, requires_grad=True)
    unit = torch.ones((), **_DD)
    field_leaf = torch.zeros(3, **_DD, requires_grad=True)
    case = _case(
        "NO2",
        method=GFN2_XTB,
        representation=labels.SCP_MODE_POTENTIAL,
        mixer=mixer,
        tolerance=tolerance,
        maxiter=300,
        temperature=300.0,
        field=field_leaf,
    )
    assert case.system.h0_setup is not None
    torch.testing.assert_close(
        case.system.h0_setup.refocc,
        torch.tensor([1.5, 3.5, 2.0, 4.0], **_DD),
    )
    root_record = _run_root(
        case,
        case.inputs,
        tolerance=tolerance,
        maxiter=300,
        eager=True,
    )
    assert root_record.converged
    unrolled_field_response = torch.autograd.grad(
        root_record.finalized_energy, field_leaf, retain_graph=True
    )[0][1]
    root = root_record.root.detach()
    outcome = root_record.outcome
    assert outcome.map_result.fermi_status is not None
    torch.testing.assert_close(
        outcome.map_result.occupation.sum(-1),
        case.nel,
        atol=1.0e-8,
        rtol=1.0e-8,
    )
    _, layout = _initial_state(case, case.inputs)
    adapter, parameters = _SCFMapTensorTree.prepare(
        case.inputs, labels.SCP_MODE_POTENTIAL, layout
    )
    field_entry_index = next(
        index
        for index, entry in enumerate(case.inputs.interactions.entries)
        if entry.kind == "electric_field"
    )
    field_prefix = f"interactions.entries[{field_entry_index}].data"
    vat_index = adapter.parameter_paths.index(f"{field_prefix}[0][1]")
    vdp_index = adapter.parameter_paths.index(f"{field_prefix}[1][1]")
    vat_direction = case.positions[:, 1]
    vdp_direction = torch.zeros_like(parameters[vdp_index])
    vdp_direction[:, 1] = 1.0

    def field_parameters(delta: Tensor, values: tuple[Tensor, ...]):
        updated = list(values)
        updated[vat_index] = values[vat_index] + delta * vat_direction
        updated[vdp_index] = values[vdp_index] + delta * vdp_direction
        return tuple(updated)

    def raw_map(state: Tensor, delta: Tensor, *values: Tensor) -> Tensor:
        return adapter(state, *field_parameters(delta, values))

    _, implicit_tangent = torch.func.jvp(
        lambda delta: implicit_root(root, delta, *parameters, raw_map=raw_map),
        (origin,),
        (unit,),
    )
    state_jacobian = torch.func.jacrev(
        lambda state: raw_map(state, origin, *parameters)
    )(root)
    parameter_response = torch.func.jvp(
        lambda delta: raw_map(root, delta, *parameters),
        (origin,),
        (unit,),
    )[1]
    identity = torch.eye(root.numel(), **_DD)
    dense_tangent = torch.linalg.solve(
        identity - state_jacobian.reshape(identity.shape),
        parameter_response.reshape(-1),
    ).reshape_as(root)
    torch.testing.assert_close(
        implicit_tangent, dense_tangent, atol=2.0e-8, rtol=2.0e-7
    )

    def output(state: Tensor, delta: Tensor) -> Tensor:
        values = field_parameters(delta, parameters)
        current_inputs, _ = adapter._rebuild(values)
        map_result = adapter.evaluate(state, *values)
        finalized = finalize_scf_potential(
            map_result,
            current_inputs,
            layout,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
            outcome.trajectory_status,
            outcome.damping_state,
            root_x=state,
        )
        return (finalized.energy + finalized.fenergy).sum()

    def implicit_output(delta: Tensor) -> Tensor:
        attached = implicit_root(root, delta, *parameters, raw_map=raw_map)
        return output(attached, delta)

    implicit_energy_response = torch.autograd.grad(
        implicit_output(origin), origin, allow_unused=False
    )[0]
    energy_state_gradient = torch.func.jacrev(
        lambda state: output(state, origin)
    )(root)
    energy_parameter_response = torch.func.jvp(
        lambda delta: output(root, delta), (origin,), (unit,)
    )[1]
    dense_energy_response = energy_parameter_response + torch.sum(
        energy_state_gradient * dense_tangent
    )
    torch.testing.assert_close(
        implicit_energy_response,
        dense_energy_response,
        atol=2.0e-7,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        implicit_energy_response,
        unrolled_field_response,
        atol=2.0e-4,
        rtol=2.0e-3,
    )

    def third_response(energy: Tensor, parameter: Tensor) -> Tensor:
        first = torch.autograd.grad(
            energy, parameter, create_graph=True, retain_graph=True
        )[0]
        second = torch.autograd.grad(
            first, parameter, create_graph=True, retain_graph=True
        )[0]
        return torch.autograd.grad(second, parameter)[0]

    implicit_third = third_response(implicit_output(origin), origin)

    def unrolled_output(delta: Tensor) -> Tensor:
        perturbed_inputs = _inputs_at_field(case, delta)
        record = _run_root(
            case,
            perturbed_inputs,
            tolerance=tolerance,
            maxiter=300,
            eager=True,
        )
        return record.finalized_energy

    unrolled_third = third_response(unrolled_output(origin), origin)
    torch.testing.assert_close(
        implicit_third, unrolled_third, atol=2.0e-3, rtol=2.0e-3
    )

    # Match the independently established E3.1a physical-forward window for
    # NO2; smaller steps are below the stable direct-energy FD region.
    step = torch.tensor(5.0e-4, **_DD)
    energies = []
    for sign in (-1.0, 1.0):
        field = torch.zeros(3, **_DD)
        field[1] = sign * step
        perturbed = _case(
            "NO2",
            method=GFN2_XTB,
            representation=labels.SCP_MODE_POTENTIAL,
            mixer=mixer,
            tolerance=tolerance,
            maxiter=300,
            temperature=300.0,
            field=field,
        )
        perturbed_root = _run_root(
            perturbed,
            perturbed.inputs,
            tolerance=tolerance,
            maxiter=300,
            eager=True,
        )
        energies.append(perturbed_root.finalized_energy)
    physical_fd = (energies[1] - energies[0]) / (2.0 * step)
    torch.testing.assert_close(
        implicit_energy_response, physical_fd, atol=2.0e-4, rtol=2.0e-3
    )
