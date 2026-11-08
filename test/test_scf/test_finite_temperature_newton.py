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
"""Thermal response and matrix-free physical Newton operator checks."""

from __future__ import annotations

from inspect import getsource

import pytest
import torch
from torch.func import jacfwd, jvp

from benchmarks.prototypes.scf_newton_krylov import (
    NewtonKrylovConverger,
    NewtonKrylovState,
    fixed_point_jvp,
)
from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.scf import (
    SCFPhysicalState,
    SCFProblem,
    SCFVariable,
)
from dxtb._src.scf.broyden import BroydenConverger
from dxtb._src.scf.data import unpack_charges
from dxtb._src.scf.map import (
    fock_initial_guess,
    potential_initial_guess,
)
from dxtb._src.scf.converger import SCFConvergerContext
from dxtb._src.scf.solver import (
    SCFSolverSettings,
    _solve_map,
)
from dxtb._src.scf.spectral import (
    eager_primal_fermi_scope,
    fermi_density_matrix,
)
from dxtb._src.scf.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)
from dxtb._src.wavefunction.filling import _get_fermi_occupation_core_state

from .test_map import _prepare_case


def _small_active_case(
    variable: SCFVariable,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[SCFProblem, torch.Tensor]:
    """Build a fractional-charge active-smearing H2 map in one variable."""
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        charge=0.35,
        dtype=dtype,
        scf_options={"fermi_etemp": 5000.0},
    )
    assert case.inputs.fermi_settings is not None
    assert case.inputs.fermi_settings.smear
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    if variable is SCFVariable.CHARGE:
        return SCFProblem(case.inputs, variable), case.x
    if variable is SCFVariable.POTENTIAL:
        guess, layout = potential_initial_guess(charges, case.inputs)
        return SCFProblem(case.inputs, variable, layout), guess
    guess = fock_initial_guess(charges, case.inputs)
    return SCFProblem(case.inputs, variable), guess


def test_orbital_commutator_misses_diagonal_thermal_response() -> None:
    """Diagonal thermal occupations respond while the orbital commutator vanishes."""
    dtype = torch.float64
    hamiltonian = torch.diag(torch.tensor([-0.2, 0.2, 0.5], dtype=dtype))
    nel = torch.tensor([[1.0, 1.0]], dtype=dtype)
    kt = torch.tensor(0.5, dtype=dtype)
    energies = torch.linalg.eigvalsh(hamiltonian).reshape(1, 1, 3)
    energies = energies.expand(1, 2, 3).clone()
    state = _get_fermi_occupation_core_state(nel, energies, kt, maxiter=100)
    density_fn = lambda value: fermi_density_matrix(value, nel, kt, state)
    direction = torch.diag(torch.tensor([1.0, -0.5, -0.5], dtype=dtype))

    density, density_tangent = jvp(density_fn, (hamiltonian,), (direction,))
    orbital_gradient = hamiltonian @ density - density @ hamiltonian

    assert torch.linalg.matrix_norm(orbital_gradient) == 0.0
    assert torch.linalg.matrix_norm(density_tangent) > 1.0e-2
    torch.testing.assert_close(
        torch.diagonal(density_tangent, dim1=-2, dim2=-1).sum(),
        torch.zeros((), dtype=dtype),
        atol=2.0e-13,
        rtol=0.0,
    )
    step = torch.tensor(1.0e-5, dtype=dtype)
    finite_difference = (
        density_fn(hamiltonian + step * direction)
        - density_fn(hamiltonian - step * direction)
    ) / (2.0 * step)
    torch.testing.assert_close(
        density_tangent, finite_difference, atol=2.0e-8, rtol=2.0e-7
    )
    assert bool(
        (
            (state.result.occupation > 0.0) & (state.result.occupation < 1.0)
        ).any()
    )


def test_active_smearing_map_jvp_preserves_electron_count() -> None:
    """The full physical map includes thermal response at fixed electron number."""
    problem, x = _small_active_case(SCFVariable.FOCK)
    overlap = problem.inputs.integrals.overlap
    direction = torch.diag(
        torch.linspace(-0.5, 0.75, x.shape[-1], dtype=x.dtype)
    )

    def density_from_fock(value: torch.Tensor) -> torch.Tensor:
        return problem.map(value).spectral.density

    density, density_tangent = jvp(density_from_fock, (x,), (direction,))
    electron_count = torch.einsum("ij,ji->", density, overlap)
    electron_count_tangent = torch.einsum("ij,ji->", density_tangent, overlap)

    torch.testing.assert_close(
        electron_count,
        problem.inputs.nel.sum(),
        atol=2.0e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(
        electron_count_tangent,
        torch.zeros_like(electron_count_tangent),
        atol=3.0e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "variable",
    [
        SCFVariable.CHARGE,
        SCFVariable.POTENTIAL,
        SCFVariable.FOCK,
    ],
)
def test_fixed_point_jvp_matches_reduced_dense_jacobian(
    variable: SCFVariable,
) -> None:
    """The projected matrix-free operator matches a tiny dense reference."""
    problem, x = _small_active_case(variable)
    dimension = 3
    basis = torch.zeros((dimension, *x.shape), dtype=x.dtype, device=x.device)
    if variable is SCFVariable.FOCK:
        basis[0, 0, 0] = 1.0
        basis[1, 1, 1] = 1.0
        basis[2, 0, 1] = 2.0**-0.5
        basis[2, 1, 0] = 2.0**-0.5
    else:
        basis.reshape(dimension, -1)[
            torch.arange(dimension), torch.arange(dimension)
        ] = 1.0

    def reduced_map(coordinates: torch.Tensor) -> torch.Tensor:
        direction = torch.einsum("i,ijk->jk", coordinates, basis)
        mapped = problem.map(x + direction).mapped_x
        return torch.einsum("ijk,jk->i", basis, mapped)

    origin = x.new_zeros((dimension,))
    jacobian = jacfwd(reduced_map)(origin)
    dense_operator = torch.eye(dimension, dtype=x.dtype) - jacobian
    vector = torch.tensor([0.25, -0.5, 0.75], dtype=x.dtype)
    physical_direction = torch.einsum("i,ijk->jk", vector, basis)
    matrix_free = fixed_point_jvp(problem, x, physical_direction)
    reduced_matrix_free = torch.einsum("ijk,jk->i", basis, matrix_free)

    if variable is SCFVariable.FOCK:
        torch.testing.assert_close(
            matrix_free, matrix_free.transpose(-2, -1), atol=1.0e-12, rtol=0.0
        )
    torch.testing.assert_close(
        reduced_matrix_free,
        dense_operator @ vector,
        atol=2.0e-9,
        rtol=2.0e-7,
    )
    step = torch.tensor(1.0e-5, dtype=x.dtype)
    finite_difference = problem.project_residual(
        (
            problem.map(x + step * physical_direction).mapped_x
            - problem.map(x - step * physical_direction).mapped_x
        )
        / (2.0 * step)
    )
    finite_difference = physical_direction - finite_difference
    finite_reduced = torch.einsum("ijk,jk->i", basis, finite_difference)
    torch.testing.assert_close(
        reduced_matrix_free,
        finite_reduced,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


@pytest.mark.parametrize(
    "variable",
    [
        SCFVariable.CHARGE,
        SCFVariable.POTENTIAL,
        SCFVariable.FOCK,
    ],
)
def test_fixed_point_jvp_matches_full_small_dense_jacobian(
    variable: SCFVariable,
) -> None:
    """The complete tiny-map operator agrees with a dense test reference."""
    problem, x = _small_active_case(variable)
    shape = x.shape

    def flat_map(flat: torch.Tensor) -> torch.Tensor:
        state = flat.reshape(shape)
        return problem.map(state).mapped_x.reshape(-1)

    def project(value: torch.Tensor) -> torch.Tensor:
        projected = problem.project_residual(value)
        if variable is SCFVariable.FOCK:
            projected = 0.5 * (projected + projected.transpose(-2, -1))
            projected = problem.project_residual(projected)
        return projected

    jacobian = jacfwd(flat_map)(x.reshape(-1))
    basis = torch.eye(x.numel(), dtype=x.dtype)
    projection = torch.stack(
        [project(vector.reshape(shape)).reshape(-1) for vector in basis],
        dim=1,
    )
    operator = projection - projection @ jacobian @ projection
    direction = torch.randn_like(x)
    matrix_free = fixed_point_jvp(problem, x, direction)
    torch.testing.assert_close(
        matrix_free.reshape(-1),
        operator @ direction.reshape(-1),
        atol=3.0e-9,
        rtol=3.0e-7,
    )


def test_fixed_point_jvp_uses_static_fermi_schedule_inside_eager_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Newton JVP bypasses W2c10's detached eager adaptive schedule."""
    import dxtb._src.scf.spectral as spectral

    problem, x = _small_active_case(SCFVariable.CHARGE)
    calls = 0
    matrix_fermi = spectral._matrix_fermi

    def count_static(*args, **kwargs):
        nonlocal calls
        calls += 1
        return matrix_fermi(*args, **kwargs)

    def reject_adaptive(*args, **kwargs):
        raise AssertionError("adaptive Fermi schedule entered a Newton JVP")

    monkeypatch.setattr(spectral, "_matrix_fermi", count_static)
    monkeypatch.setattr(spectral, "_matrix_fermi_eager", reject_adaptive)
    direction = torch.randn_like(x)
    with eager_primal_fermi_scope(True):
        result = fixed_point_jvp(problem, x, direction)

    assert calls > 0
    assert torch.isfinite(result).all()


def test_detached_primal_may_use_adaptive_fermi_schedule() -> None:
    """The W2c10 adaptive schedule remains available to detached primal maps."""
    problem, x = _small_active_case(SCFVariable.CHARGE)
    with torch.no_grad(), eager_primal_fermi_scope(True):
        result = problem.map(x)

    assert torch.isfinite(result.spectral.density).all()
    assert torch.isfinite(result.mapped_x).all()


def test_newton_krylov_requires_active_smearing() -> None:
    """The finite-temperature root method refuses fixed-Aufbau problems."""
    case = _prepare_case(
        GFN2_XTB,
        sample_name="H2O",
        dtype=torch.float64,
        scf_options={"fermi_etemp": 0.0},
    )
    problem = SCFProblem(case.inputs, SCFVariable.CHARGE)
    physical = problem.evaluate(case.x)
    method = NewtonKrylovConverger()
    context = SCFConvergerContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=case.x.new_tensor(1.0),
        proposal_x=physical.map_result.mapped_x,
    )

    with pytest.raises(NotImplementedError, match="Fermi smearing at nonzero electronic temperature"):
        method.propose(problem, physical, None, context)


def test_newton_krylov_rejected_before_fixed_trip_map() -> None:
    """The eager-only method is rejected before a fixed-trip physical map."""
    from benchmarks.prototypes.scf_newton_krylov import NewtonKrylovConverger
    from dxtb._src.scf.solver import _solve_map
    from dxtb._src.scf.strategy import SCFStage, SCFStrategy

    problem, guess = _small_active_case(SCFVariable.CHARGE)
    calls = 0

    def counted_map(value: torch.Tensor):
        nonlocal calls
        calls += 1
        return problem.map(value)

    with pytest.raises(NotImplementedError, match="eager execution only"):
        _solve_map(
            guess,
            SCFSolverSettings(10, False, 0.5, 1.0e-6, 1.0e-6),
            counted_map,
            problem=problem,
            strategy=SCFStrategy((SCFStage(NewtonKrylovConverger()),)),
            eager=False,
        )

    assert calls == 0


def test_newton_krylov_state_freezes_on_inactive_transition() -> None:
    """Inactive method transitions preserve their input and immutable state."""
    problem, x = _small_active_case(SCFVariable.CHARGE)
    physical = problem.evaluate(x)
    method = NewtonKrylovConverger()
    context = SCFConvergerContext(
        step=1,
        transition_active=torch.tensor(False),
        damp=x.new_tensor(1.0),
        proposal_x=physical.map_result.mapped_x,
    )
    result = method.propose(problem, physical, None, context)

    assert result.x is physical.x
    assert result.state is None


def test_failed_krylov_uses_explicit_fixed_point_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A singular linearized operator takes a diagnosed finite fallback."""
    import benchmarks.prototypes.scf_newton_krylov as newton

    problem, x = _small_active_case(SCFVariable.CHARGE)
    physical = problem.evaluate(x)
    mapped = physical.map_result.mapped_x
    method = NewtonKrylovConverger()
    context = SCFConvergerContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=x.new_tensor(0.5),
        proposal_x=mapped,
    )
    monkeypatch.setattr(
        newton,
        "fixed_point_jvp",
        lambda _problem, _x, direction: torch.zeros_like(direction),
    )
    result = method.propose(problem, physical, None, context)

    assert isinstance(result.state, newton.NewtonKrylovState)
    assert bool(result.state.used_fixed_point_fallback)
    assert int(result.state.fallback_count) == 1
    assert not bool(result.state.last_linear_converged)
    assert torch.isfinite(result.x).all()


def test_newton_krylov_float32_step_and_operator_are_finite() -> None:
    """The float32 active-smearing path keeps finite JVP and step diagnostics."""
    problem, x = _small_active_case(
        SCFVariable.POTENTIAL, dtype=torch.float32
    )
    physical = problem.evaluate(x)
    context = SCFConvergerContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=x.new_tensor(0.5),
        proposal_x=physical.map_result.mapped_x,
    )
    proposal = NewtonKrylovConverger().propose(problem, physical, None, context)

    assert torch.isfinite(fixed_point_jvp(problem, x, torch.ones_like(x))).all()
    assert torch.isfinite(proposal.x).all()
    assert isinstance(proposal.state, NewtonKrylovState)


def test_bicgstab_solves_a_nonsymmetric_dense_reference() -> None:
    """The matrix-free Krylov kernel matches a small nonsymmetric solve."""
    from benchmarks.prototypes.scf_newton_krylov import _bicgstab

    matrix = torch.tensor([[2.0, -0.4], [0.3, 1.2]], dtype=torch.float64)
    rhs = torch.tensor([0.7, -1.3], dtype=torch.float64)
    result = _bicgstab(
        lambda value: matrix @ value,
        rhs,
        max_iterations=10,
        relative_tolerance=1.0e-12,
        absolute_tolerance=rhs.new_tensor(1.0e-14),
    )

    assert bool(result.converged)
    assert int(result.jvp_count) > 0
    torch.testing.assert_close(
        result.solution,
        torch.linalg.solve(matrix, rhs),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_newton_residual_sign_gives_the_root_step() -> None:
    """Solving ``(I-G_x) delta = G(x)-x`` has the accepted Newton sign."""
    slope = 0.2
    offset = 1.0
    x = 2.0
    residual = slope * x + offset - x
    delta = residual / (1.0 - slope)
    root_step = x + delta
    opposite_step = x - delta

    assert abs(slope * root_step + offset - root_step) < 1.0e-14
    assert abs(slope * opposite_step + offset - opposite_step) > 1.0e-1


def test_newton_prototype_operator_is_matrix_free() -> None:
    """The experimental Newton operator never materializes a dense Jacobian."""
    import benchmarks.prototypes.scf_newton_krylov as newton

    source = getsource(newton)
    assert "jacfwd" not in source
    assert "jacrev" not in source
    assert "torch.func.hessian" not in source


def test_no2_active_smearing_broyden_to_newton_is_certified() -> None:
    """A materially thermal open-shell case reaches the S2 root after switching."""
    case = _prepare_case(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        dtype=torch.float64,
        scf_options={"fermi_etemp": 2000.0},
    )
    assert case.inputs.fermi_settings is not None
    assert case.inputs.fermi_settings.smear
    # Use the open-shell case's own physical inputs for this strategy.
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    guess, layout = potential_initial_guess(charges, case.inputs)
    problem = SCFProblem(case.inputs, SCFVariable.POTENTIAL, layout)
    strategy = SCFStrategy(
        (
            SCFStage(
                BroydenConverger(),
                variable=SCFVariable.POTENTIAL,
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=2, fixed_point_l2=1.0e-2
                ),
            ),
            SCFStage(
                NewtonKrylovConverger(),
                variable=SCFVariable.POTENTIAL,
            ),
        )
    )
    settings = SCFSolverSettings(100, False, 0.5, 1.0e-7, 1.0e-7)
    outcome = _solve_map(
        guess,
        settings,
        problem.map,
        problem=problem,
        strategy=strategy,
        eager=True,
    )

    assert bool(outcome.converged)
    assert outcome.certification is not None
    assert bool(outcome.certification.certified)
    assert outcome.strategy_diagnostics is not None
    assert int(outcome.strategy_diagnostics.switch_count) == 1
    assert int(outcome.strategy_diagnostics.final_stage_index) == 1
    assert int(outcome.strategy_diagnostics.stage_iterations[1]) > 0
