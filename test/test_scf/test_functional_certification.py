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
"""Physical SCF certification metrics and transform behavior."""

from __future__ import annotations

from dataclasses import replace
from inspect import Parameter, signature

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.scf.functional import (
    AndersonMethod,
    AndersonMethodSettings,
    BroydenMethod,
    CDIISMethod,
    LinearMethod,
    SCFCertificationThresholds,
    SCFCoreFailure,
    SCFInternalCertificationProfile,
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
    SCFSolverSettings,
)
from dxtb._src.scf.functional import certification as certification_module
from dxtb._src.scf.functional import (
    certification_core_metrics,
    certification_metrics,
    certify_scf_core_state,
    certify_scf_state,
    electronic_energy_from_spectral,
    fock_initial_guess,
    free_energy_from_spectral,
    interaction_energy_from_charges,
    potential_initial_guess,
    resolve_certification_thresholds,
    scf_map,
    scf_progress_metrics,
    stationarity_matrix,
)
from dxtb._src.scf.functional.certification import (
    SCF_CERTIFICATION_HIGHER_ORDER_REFERENCE,
    SCF_CERTIFICATION_STANDARD,
    SCF_CERTIFICATION_TIGHT,
    _stationarity_norms,
)
from dxtb._src.scf.functional.method import SCFMethod
from dxtb._src.scf.functional.solver import (
    DynamicDampingSettings,
    SCFTrajectoryStatus,
    _resolve_tolerance,
    _solve_map,
    _solve_map_control,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
    setup_simple_solver_settings,
    solve_scf_simple,
)
from dxtb._src.typing.exceptions import SCFConvergenceError

from .scf_certification_characterization import characterize_case
from .test_functional_map import _prepare_case


def _problem_and_state(
    method: int = GFN1_XTB,
    representation: SCFRepresentation = SCFRepresentation.CHARGE,
    *,
    sample_name: str = "H2O",
    temperature: float = 0.0,
    charge: float = 0.0,
    spin: float = 0.0,
    dtype: torch.dtype = torch.float64,
):
    case = _prepare_case(
        method,
        sample_name=sample_name,
        charge=charge,
        spin=spin,
        dtype=dtype,
        scf_options={"fermi_etemp": temperature},
    )
    if representation is SCFRepresentation.CHARGE:
        x = case.x
        layout = None
    else:
        from dxtb._src.scf.functional import unpack_charges

        charges = unpack_charges(case.x, case.inputs.charge_layout)
        if representation is SCFRepresentation.POTENTIAL:
            x, layout = potential_initial_guess(charges, case.inputs)
        else:
            x = fock_initial_guess(charges, case.inputs)
            layout = None
    problem = SCFProblem(case.inputs, representation, layout)
    return case, problem, problem.evaluate(x)


def _loose_thresholds(
    representation: SCFRepresentation,
) -> SCFCertificationThresholds:
    return SCFCertificationThresholds(
        representation=representation,
        fixed_point_l2=1.0e6,
        fixed_point_max=1.0e6,
        stationarity_orthogonalized_per_sqrt_orbital=1.0e6,
        electron_count=1.0e6,
    )


@pytest.mark.parametrize(
    "method,config_method,h0_method,parameter_name,reference",
    [
        (GFN1_XTB, labels.GFN1_XTB, "gfn1", "GFN1-xTB", -5.8052489623704),
        (GFN2_XTB, labels.GFN2_XTB, "gfn2", "GFN2-xTB", -5.1041590251073),
    ],
    ids=["gfn1-h2o", "gfn2-h2o"],
)
def test_characterization_h2o_helper_proves_resolved_method(
    method: int,
    config_method: int,
    h0_method: str,
    parameter_name: str,
    reference: float,
) -> None:
    """Guard characterization rows against a requested/resolved model mixup."""
    case = _prepare_case(method, sample_name="H2O")
    assert case.calc.model.config.method == config_method
    assert case.system.config.method == config_method
    assert case.system.h0_setup.method == h0_method
    assert case.par.forward().meta.name.value == parameter_name

    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)
    outcome = _solve_map(
        case.x,
        SCFSolverSettings(120, False, 0.2, 1.0e-6, 1.0e-6),
        problem.map,
    )
    assert bool(outcome.converged)
    spectral = outcome.map_result.spectral
    energy = (
        electronic_energy_from_spectral(spectral, case.inputs).sum()
        + free_energy_from_spectral(spectral, case.inputs).sum()
    ).item()
    assert abs(energy - reference) < 1.0e-3


def test_characterization_row_contains_verified_model_and_initial_state() -> (
    None
):
    """Structured rows carry checked model and physical input provenance."""
    row = characterize_case(
        GFN2_XTB,
        sample_name="H2",
        representation=SCFRepresentation.CHARGE,
        dtype=torch.float64,
        tolerance=1.0e-6,
    )
    assert row["model"] == "GFN2-xTB"
    assert row["model_config"] == labels.GFN2_XTB
    assert row["h0_method"] == "gfn2"
    assert row["parameter_model"] == "GFN2-xTB"
    assert row["initial_x_sha256"]
    assert row["occupation"]
    assert row["nel"]
    assert row["hcore_norm"] > 0.0
    assert row["overlap_norm"] > 0.0
    assert row["converged"]


def test_orthogonalized_stationarity_scaling_for_direct_sums() -> None:
    """The per-sqrt-orbital norm is constant for repeated residual blocks."""
    block = torch.tensor(
        [[0.0, 0.4, -0.1], [-0.4, 0.0, 0.2], [0.1, -0.2, 0.0]],
        dtype=torch.float64,
    )

    def metrics(scale: torch.Tensor, copies: int) -> tuple[torch.Tensor, ...]:
        residual = torch.block_diag(*(scale * block for _ in range(copies)))
        mask = torch.ones(residual.shape[-1], dtype=torch.bool)
        inverse_sqrt_overlap = torch.eye(
            residual.shape[-1], dtype=residual.dtype
        )
        return _stationarity_norms(residual, inverse_sqrt_overlap, mask)

    base = metrics(torch.tensor(1.0, dtype=torch.float64), 1)
    for copies in (1, 2, 4, 8):
        values = metrics(torch.tensor(1.0, dtype=torch.float64), copies)
        assert torch.allclose(values[2], base[2] * copies**0.5)
        assert torch.allclose(values[3], base[3])
        assert torch.allclose(values[4], base[4])
        assert torch.allclose(values[5], base[5] / copies**0.5)
        assert torch.allclose(
            values[2] / (3 * copies), base[2] / 3 / copies**0.5
        )

    direction = torch.tensor(0.3, dtype=torch.float64)

    def normalized(scale: torch.Tensor) -> torch.Tensor:
        return metrics(scale, 4)[4]

    value, tangent = torch.func.jvp(
        normalized, (torch.tensor(1.0, dtype=torch.float64),), (direction,)
    )
    jacobian = torch.func.jacfwd(normalized)(
        torch.tensor(1.0, dtype=torch.float64)
    )
    assert torch.allclose(value, base[4])
    assert torch.allclose(tangent, jacobian * direction)


def test_float32_electron_trace_error_scales_with_repeated_blocks() -> None:
    """Independent Tr(P S) roundoff grows linearly under block replication."""
    density64 = torch.diag(
        torch.tensor([0.123456789123, 0.234567891234], dtype=torch.float64)
    )
    overlap64 = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64)
    block_target = torch.diagonal(density64 @ overlap64).sum()
    errors = []
    for copies in (1, 2, 4, 8):
        density = torch.block_diag(
            *(density64.to(torch.float32) for _ in range(copies))
        )
        overlap = torch.block_diag(
            *(overlap64.to(torch.float32) for _ in range(copies))
        )
        measured = torch.diagonal(density @ overlap).sum()
        target = copies * block_target
        errors.append((measured.double() - target).abs())
    for copies, error in zip((1, 2, 4, 8), errors):
        assert error.item() == pytest.approx(
            errors[0].item() * copies, rel=1e-6
        )


def test_internal_profiles_have_explicit_float32_stationarity_floor() -> None:
    """Reference profiles never imply float64 tolerances in float32."""
    assert SCF_CERTIFICATION_STANDARD.fixed_point_l2 == 1.0e-4
    assert SCF_CERTIFICATION_STANDARD.fixed_point_max == 1.0e-5
    for profile in (
        SCF_CERTIFICATION_STANDARD,
        SCF_CERTIFICATION_TIGHT,
        SCF_CERTIFICATION_HIGHER_ORDER_REFERENCE,
    ):
        assert profile.stationarity_tolerance(torch.float32) == 1.0e-5
    assert (
        SCF_CERTIFICATION_HIGHER_ORDER_REFERENCE.stationarity_tolerance(
            torch.float64
        )
        == 1.0e-12
    )


@pytest.mark.parametrize("tolerance", [1.0e-2, 1.0e-8])
def test_standard_thresholds_inherit_effective_caller_tolerances(
    tolerance: float,
) -> None:
    """Standard certification neither tightens nor loosens caller settings."""
    _, problem, state = _problem_and_state()
    thresholds = resolve_certification_thresholds(
        problem,
        x_atol=tolerance,
        x_atol_max=tolerance * 2.0,
        dtype=state.x.dtype,
    )
    assert thresholds.fixed_point_l2 == tolerance
    assert thresholds.fixed_point_max == tolerance * 2.0
    assert thresholds.electron_count == tolerance


def test_controller_uses_core_certification_without_energy_observables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The authoritative solve gate does not call energy or entropy helpers."""

    def forbidden(*args, **kwargs):
        raise AssertionError("full energy diagnostics entered the SCF gate")

    monkeypatch.setattr(
        certification_module, "electronic_energy_from_spectral", forbidden
    )
    monkeypatch.setattr(
        certification_module, "free_energy_from_spectral", forbidden
    )
    case, problem, _ = _problem_and_state(sample_name="H2O")
    settings = SCFSolverSettings(100, False, 0.2, 1.0e-6, 1.0e-6)
    result = solve_scf_simple(case.x, case.inputs, settings)
    assert result.diagnostics is not None
    assert result.diagnostics.certification is not None


@pytest.mark.parametrize("eager", [False, True], ids=["fixed-trip", "eager"])
def test_raw_candidate_with_failed_stationarity_keeps_iterating(
    eager: bool,
) -> None:
    """A loose raw root does not stop while physical stationarity fails."""
    case, problem, _ = _problem_and_state(sample_name="H2O")
    profile = SCFInternalCertificationProfile(
        name="stationarity-reject",
        fixed_point_l2=1.0e6,
        fixed_point_max=1.0e6,
        stationarity_orthogonalized_per_sqrt_orbital=0.0,
    )
    settings = SCFSolverSettings(2, False, 0.2, 1.0e6, 1.0e6)
    result = solve_scf_simple(
        case.x,
        case.inputs,
        settings,
        eager=eager,
        certification_profile=profile,
    )
    assert not bool(result.converged)
    assert int(result.iterations) == 3
    assert result.diagnostics is not None
    assert not bool(result.diagnostics.terminal_failed)
    assert result.diagnostics.certification is not None
    assert int(result.diagnostics.certification.result.failure_code) == 2


@pytest.mark.parametrize("eager", [False, True], ids=["fixed-trip", "eager"])
def test_independent_count_failure_blocks_controller_certification(
    eager: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The independent trace gate blocks an otherwise valid candidate."""
    original = certification_module.certification_core_metrics

    def count_inconsistent(problem, state, *, fixed_point_residual=None):
        metrics = original(
            problem, state, fixed_point_residual=fixed_point_residual
        )
        return replace(
            metrics,
            stationarity_orthogonalized_l2_per_sqrt_orbital=state.x.new_zeros(
                ()
            ),
            electron_count_error=state.x.new_tensor(1.0e9),
        )

    monkeypatch.setattr(
        certification_module, "certification_core_metrics", count_inconsistent
    )
    case, _, _ = _problem_and_state(sample_name="H2O")
    settings = SCFSolverSettings(2, False, 0.2, 1.0e6, 1.0e6)
    result = solve_scf_simple(case.x, case.inputs, settings, eager=eager)
    assert not bool(result.converged)
    assert result.diagnostics is not None
    assert result.diagnostics.certification is not None
    assert bool(result.diagnostics.certification.result.fixed_point_ok)
    assert bool(result.diagnostics.certification.result.stationarity_ok)
    assert not bool(result.diagnostics.certification.result.electron_count_ok)
    assert int(result.diagnostics.certification.result.failure_code) == int(
        SCFCoreFailure.ELECTRON_COUNT
    )


def test_implicit_preparation_rejects_raw_only_root() -> None:
    """Implicit attachment cannot accept a raw-converged uncertified root."""
    from dxtb._src.scf.functional.implicit import _prepare_scf_root

    case, _, _ = _problem_and_state(sample_name="H2O")
    profile = SCFInternalCertificationProfile(
        name="stationarity-reject",
        fixed_point_l2=1.0e6,
        fixed_point_max=1.0e6,
        stationarity_orthogonalized_per_sqrt_orbital=0.0,
    )
    outcome = _solve_map(
        case.x,
        SCFSolverSettings(2, False, 0.2, 1.0e6, 1.0e6),
        SCFProblem(case.inputs, SCFRepresentation.CHARGE).map,
        problem=SCFProblem(case.inputs, SCFRepresentation.CHARGE),
        certification_profile=profile,
    )
    assert not bool(outcome.converged)

    def raw_map(x: torch.Tensor) -> torch.Tensor:
        return SCFProblem(case.inputs, SCFRepresentation.CHARGE).map(x).mapped_x

    with pytest.raises(SCFConvergenceError):
        _prepare_scf_root(
            outcome,
            maxiter=2,
            x_atol=1.0e6,
            x_atol_max=1.0e6,
            final_valid=torch.ones((), dtype=torch.bool),
            raw_map=raw_map,
            parameters=(),
        )


def test_certification_does_not_accept_trajectory_history_as_input() -> None:
    """One paired physical state has one verdict independent of history."""
    _, problem, state = _problem_and_state(sample_name="H2O")
    thresholds = _loose_thresholds(problem.representation)
    first = certify_scf_core_state(problem, state, thresholds)
    second = certify_scf_core_state(problem, state, thresholds)
    assert torch.equal(first.certified, second.certified)
    assert torch.equal(first.failure_code, second.failure_code)
    assert first.metrics is not None
    # Trajectory data is a solver-result diagnostic and is not accepted by the
    # physical certification function.
    status = SCFTrajectoryStatus(
        fermi_converged=torch.tensor(False),
        electron_count_ok=torch.tensor(False),
        invalid_temperature=torch.tensor(False),
        too_many_electrons=torch.tensor(False),
        first_failure=torch.tensor(3),
    )
    assert status.first_failure.item() == 3


def test_solver_diagnostics_pair_exact_root_and_map_result() -> None:
    """Stored diagnostics describe the tested x paired with G(x)."""
    case, problem, _ = _problem_and_state(sample_name="H2O")
    result = solve_scf_simple(
        case.x,
        case.inputs,
        SCFSolverSettings(140, False, 0.2, 1.0e-8, 1.0e-8),
    )
    assert bool(result.converged)
    assert not torch.equal(result.root_x, result.map_result.mapped_x)
    state = SCFPhysicalState(result.root_x, result.map_result)
    expected = certification_core_metrics(problem, state)
    assert result.diagnostics is not None
    assert result.diagnostics.certification is not None
    actual = result.diagnostics.certification.result.metrics
    torch.testing.assert_close(actual.fixed_point.l2, expected.fixed_point.l2)
    torch.testing.assert_close(
        actual.stationarity_orthogonalized_l2_per_sqrt_orbital,
        expected.stationarity_orthogonalized_l2_per_sqrt_orbital,
    )

    historical_failure = SCFTrajectoryStatus(
        fermi_converged=torch.tensor(False),
        electron_count_ok=torch.tensor(False),
        invalid_temperature=torch.tensor(False),
        too_many_electrons=torch.tensor(False),
        first_failure=torch.tensor(3),
    )
    altered_history = replace(result, trajectory_status=historical_failure)
    assert torch.equal(result.converged, altered_history.converged)


@pytest.mark.parametrize(
    "representation",
    [
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ],
    ids=("charge", "potential", "fock"),
)
def test_fixed_trip_certification_matches_selected_root_diagnostics(
    representation: SCFRepresentation,
) -> None:
    """Final diagnostics agree with controller bits for the selected pair."""
    case, problem, state = _problem_and_state(
        representation=representation, sample_name="H2"
    )
    assert (
        signature(finalize_scf).parameters["root_x"].default is Parameter.empty
    )
    assert (
        signature(finalize_scf_potential).parameters["root_x"].default
        is Parameter.empty
    )
    assert (
        signature(finalize_scf_fock).parameters["root_x"].default
        is Parameter.empty
    )
    outcome = _solve_map(
        state.x,
        SCFSolverSettings(100, False, 0.2, 1.0e-8, 1.0e-8),
        problem.map,
        problem=problem,
    )
    assert outcome.certification is not None
    assert torch.equal(outcome.converged, outcome.certification.certified)
    assert not bool(outcome.terminal_failed)

    if representation is SCFRepresentation.CHARGE:
        result = finalize_scf(
            outcome.map_result,
            case.inputs,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
            outcome.trajectory_status,
            root_x=outcome.root_x,
            certification=outcome.certification,
        )
    elif representation is SCFRepresentation.POTENTIAL:
        assert problem.potential_layout is not None
        result = finalize_scf_potential(
            outcome.map_result,
            case.inputs,
            problem.potential_layout,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
            outcome.trajectory_status,
            root_x=outcome.root_x,
            certification=outcome.certification,
        )
    else:
        result = finalize_scf_fock(
            outcome.map_result,
            case.inputs,
            outcome.iterations,
            outcome.converged,
            outcome.residual,
            outcome.trajectory_status,
            root_x=outcome.root_x,
            certification=outcome.certification,
        )
    assert result.diagnostics is not None
    assert result.diagnostics.certification is not None
    assert torch.equal(
        result.converged,
        result.diagnostics.certification.result.certified,
    )
    paired = problem.evaluate(result.root_x)
    torch.testing.assert_close(
        paired.map_result.mapped_x, result.map_result.mapped_x
    )


def test_fixed_trip_budget_exhaustion_retains_uncertified_final_diagnostic() -> (
    None
):
    """Budget exhaustion cannot become convergence through final recompute."""
    case, problem, state = _problem_and_state(sample_name="H2O")
    outcome = _solve_map(
        state.x,
        SCFSolverSettings(1, False, 0.2, 1.0e-12, 1.0e-12),
        problem.map,
        problem=problem,
    )
    assert outcome.certification is not None
    assert not bool(outcome.converged)
    assert not bool(outcome.certification.certified)
    assert not bool(outcome.terminal_failed)


def test_implicit_preparation_ignores_trajectory_history_for_same_root() -> (
    None
):
    """The same certified root prepares identically with different history."""
    from dxtb._src.scf.functional.implicit import _prepare_scf_root

    case, problem, _ = _problem_and_state(sample_name="H2O")
    settings = SCFSolverSettings(140, False, 0.2, 1.0e-8, 1.0e-8)
    outcome = _solve_map(case.x, settings, problem.map, problem=problem)
    assert bool(outcome.converged)
    altered = replace(
        outcome,
        trajectory_status=SCFTrajectoryStatus(
            fermi_converged=torch.tensor(False),
            electron_count_ok=torch.tensor(False),
            invalid_temperature=torch.tensor(False),
            too_many_electrons=torch.tensor(False),
            first_failure=torch.tensor(3),
        ),
    )

    def raw_map(x: torch.Tensor) -> torch.Tensor:
        return problem.map(x).mapped_x

    kwargs = dict(
        maxiter=140,
        x_atol=1.0e-8,
        x_atol_max=1.0e-8,
        final_valid=torch.ones((), dtype=torch.bool),
        raw_map=raw_map,
        parameters=(),
    )
    prepared = _prepare_scf_root(outcome, **kwargs)
    prepared_with_history = _prepare_scf_root(altered, **kwargs)
    torch.testing.assert_close(prepared.root_x, prepared_with_history.root_x)

    with pytest.raises(SCFConvergenceError, match="authoritatively certified"):
        _prepare_scf_root(replace(outcome, certification=None), **kwargs)
    assert outcome.certification is not None
    false_certification = replace(
        outcome.certification,
        certified=torch.zeros_like(outcome.certification.certified),
    )
    with pytest.raises(RuntimeError, match="disagree"):
        _prepare_scf_root(
            replace(outcome, certification=false_certification), **kwargs
        )


@pytest.mark.parametrize(
    "representation",
    [
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ],
)
def test_fixed_point_metric_uses_paired_projected_delta(
    representation: SCFRepresentation,
) -> None:
    """The physical fixed-point metric is projected G(x)-x."""
    _, problem, state = _problem_and_state(representation=representation)
    metrics = certification_metrics(problem, state)
    delta = problem.project_residual(state.map_result.mapped_x - state.x)
    torch.testing.assert_close(
        metrics.fixed_point.l2, torch.linalg.vector_norm(delta)
    )
    torch.testing.assert_close(metrics.fixed_point.max_abs, delta.abs().amax())


def test_stationarity_uses_fresh_hamiltonian_not_diagonalized_hamiltonian() -> (
    None
):
    """A fresh charge-built Fock detects a deliberately unconverged state."""
    _, problem, state = _problem_and_state(sample_name="H2O")
    spectral = state.map_result.spectral
    overlap = problem.inputs.integrals.overlap
    density = spectral.density
    diagonalized_residual = (
        spectral.hamiltonian @ density @ overlap
        - overlap @ density @ spectral.hamiltonian
    )
    diagonalized_residual = torch.where(
        problem.inputs.orbital_mask[:, None]
        & problem.inputs.orbital_mask[None, :],
        diagonalized_residual,
        0.0,
    )
    fresh = stationarity_matrix(problem, state)
    old_norm = torch.linalg.vector_norm(diagonalized_residual)
    fresh_norm = torch.linalg.vector_norm(fresh)

    assert old_norm < 1.0e-9
    assert fresh_norm > old_norm * 100.0
    assert fresh_norm > 1.0e-7


def test_stationarity_is_nonorthogonal_and_permutation_covariant() -> None:
    """Raw and orthogonalized norms are invariant under AO permutation."""
    _, problem, state = _problem_and_state(sample_name="H2O")
    spectral = state.map_result.spectral
    fock = problem.rebuild_hamiltonian(spectral.charges)
    density = spectral.density
    overlap = problem.inputs.integrals.overlap
    mask = problem.inputs.orbital_mask
    residual = stationarity_matrix(problem, state)
    inverse_sqrt = problem.inputs.inverse_cholesky
    orthogonalized = inverse_sqrt.T @ residual @ inverse_sqrt
    identity = torch.eye(
        overlap.shape[-1], dtype=overlap.dtype, device=overlap.device
    )
    assert not torch.allclose(overlap, identity)

    permutation = torch.arange(overlap.shape[-1] - 1, -1, -1)
    permuted_fock = fock[permutation][:, permutation]
    permuted_density = density[..., permutation, :][..., :, permutation]
    permuted_overlap = overlap[permutation][:, permutation]
    permuted_mask = mask[permutation]
    pair_mask = permuted_mask[:, None] & permuted_mask[None, :]
    permuted_residual = (
        permuted_fock @ permuted_density @ permuted_overlap
        - permuted_overlap @ permuted_density @ permuted_fock
    )
    permuted_residual = torch.where(pair_mask, permuted_residual, 0.0)
    permuted_inverse_sqrt = inverse_sqrt[permutation][:, permutation]
    permuted_orthogonalized = (
        permuted_inverse_sqrt.T @ permuted_residual @ permuted_inverse_sqrt
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(permuted_residual),
        torch.linalg.vector_norm(residual),
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    torch.testing.assert_close(
        permuted_residual.abs().amax(),
        residual.abs().amax(),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(permuted_orthogonalized),
        torch.linalg.vector_norm(orthogonalized),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        permuted_orthogonalized.abs().amax(),
        orthogonalized.abs().amax(),
        atol=2e-12,
        rtol=2e-12,
    )


def test_orthogonalized_stationarity_matches_orthogonal_basis_algebra() -> None:
    """The transformed commutator equals X.T R_AO X for SPD overlap."""
    dtype = torch.float64
    overlap_seed = torch.tensor(
        [[1.4, 0.2, -0.1], [0.2, 1.1, 0.15], [-0.1, 0.15, 0.9]],
        dtype=dtype,
    )
    overlap = overlap_seed @ overlap_seed.T
    fock = torch.tensor(
        [[-0.8, 0.1, 0.3], [0.1, 0.4, -0.2], [0.3, -0.2, 0.9]],
        dtype=dtype,
    )
    density_seed = torch.tensor(
        [[0.7, 0.1, 0.05], [0.1, 0.3, -0.02], [0.05, -0.02, 0.2]],
        dtype=dtype,
    )
    density = density_seed @ density_seed.T
    eigenvalues, eigenvectors = torch.linalg.eigh(overlap)
    inverse_sqrt = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(-2)
    ) @ eigenvectors.T

    residual_ao = fock @ density @ overlap - overlap @ density @ fock
    residual_orth_from_ao = inverse_sqrt.T @ residual_ao @ inverse_sqrt
    fock_orth = inverse_sqrt.T @ fock @ inverse_sqrt
    inverse = torch.linalg.inv(inverse_sqrt)
    density_orth = inverse @ density @ inverse.T
    residual_orth_direct = fock_orth @ density_orth - density_orth @ fock_orth

    torch.testing.assert_close(
        residual_orth_from_ao, residual_orth_direct, atol=2e-13, rtol=2e-13
    )


def test_stationarity_metrics_respond_to_basis_rescaling() -> None:
    """Orthogonalized Frobenius norm is basis-covariant under rescaling."""
    dtype = torch.float64
    seed = torch.tensor(
        [[1.4, 0.2, -0.1], [0.2, 1.1, 0.15], [-0.1, 0.15, 0.9]],
        dtype=dtype,
    )
    overlap = seed @ seed.T
    fock = torch.tensor(
        [[-0.8, 0.1, 0.3], [0.1, 0.4, -0.2], [0.3, -0.2, 0.9]],
        dtype=dtype,
    )
    density_seed = torch.tensor(
        [[0.7, 0.1, 0.05], [0.1, 0.3, -0.02], [0.05, -0.02, 0.2]],
        dtype=dtype,
    )
    density = density_seed @ density_seed.T

    def metrics(s: torch.Tensor, f: torch.Tensor, p: torch.Tensor):
        eigenvalues, eigenvectors = torch.linalg.eigh(s)
        inverse_sqrt = (
            eigenvectors * eigenvalues.rsqrt().unsqueeze(-2)
        ) @ eigenvectors.T
        residual = f @ p @ s - s @ p @ f
        residual_orth = inverse_sqrt.T @ residual @ inverse_sqrt
        return (
            torch.linalg.vector_norm(residual),
            residual.abs().amax(),
            torch.linalg.vector_norm(residual_orth),
            residual_orth.abs().amax(),
        )

    original = metrics(overlap, fock, density)
    basis_change = torch.diag(torch.tensor([0.35, 1.7, 3.2], dtype=dtype))
    inverse_change = torch.linalg.inv(basis_change)
    rescaled = metrics(
        basis_change.T @ overlap @ basis_change,
        basis_change.T @ fock @ basis_change,
        inverse_change @ density @ inverse_change.T,
    )
    assert not torch.isclose(original[0], rescaled[0], rtol=1e-3, atol=1e-8)
    assert not torch.isclose(original[1], rescaled[1], rtol=1e-3, atol=1e-8)
    torch.testing.assert_close(original[2], rescaled[2], atol=2e-12, rtol=2e-12)
    assert not torch.isclose(original[3], rescaled[3], rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize(
    "representation",
    [
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ],
)
def test_fresh_hamiltonian_matches_authoritative_representation_map(
    representation: SCFRepresentation,
) -> None:
    """Charge-derived fresh Fock agrees with the authoritative raw map."""
    _, problem, state = _problem_and_state(representation=representation)
    fresh = problem.rebuild_hamiltonian(state.map_result.spectral.charges)
    if representation is SCFRepresentation.FOCK:
        expected = state.map_result.mapped_x
    else:
        from dxtb._src.scf.functional import potential_from_charges
        from dxtb._src.scf.functional.map import potential_to_hamiltonian

        potential = potential_from_charges(
            state.map_result.spectral.charges, problem.inputs
        )
        integrals = problem.inputs.integrals
        expected = potential_to_hamiltonian(
            integrals.hcore,
            integrals.overlap,
            integrals.dipole,
            integrals.quadrupole,
            problem.inputs.index,
            potential,
        )
    torch.testing.assert_close(fresh, expected, atol=0.0, rtol=0.0)


def test_electron_count_is_measured_independently() -> None:
    """Certification reports Tr(P S) separately from the Fermi status."""
    _, problem, state = _problem_and_state()
    metrics = certification_metrics(problem, state)
    density = state.map_result.spectral.density
    overlap = problem.inputs.integrals.overlap
    mask = problem.inputs.orbital_mask
    density = torch.where(mask[:, None] & mask[None, :], density, 0.0)
    expected = torch.diagonal(density @ overlap, dim1=-2, dim2=-1).sum()
    torch.testing.assert_close(metrics.electron_count, expected)
    torch.testing.assert_close(
        metrics.electron_count_error,
        expected - problem.inputs.nel.sum(),
    )
    assert metrics.fermi_ok
    assert metrics.finite_ok


def test_open_shell_electron_count_keeps_both_spin_targets() -> None:
    """Independent Tr(P S) uses the target population of both spin channels."""
    case, problem, state = _problem_and_state(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        temperature=300.0,
    )
    metrics = certification_metrics(problem, state)
    assert problem.inputs.nel.shape[-1] == 2
    assert bool((problem.inputs.nel > 0).all())
    torch.testing.assert_close(
        metrics.electron_count, problem.inputs.nel.sum(), atol=2e-12, rtol=0.0
    )
    torch.testing.assert_close(
        metrics.electron_count_error,
        metrics.electron_count - problem.inputs.nel.sum(),
    )
    assert metrics.fermi_electron_count_ok


@pytest.mark.parametrize("temperature", [300.0, 1.0])
def test_float32_finite_temperature_certification_floor(
    temperature: float,
) -> None:
    """Fermi and physical metrics remain valid at the float32 tolerance floor."""
    case, problem, state = _problem_and_state(
        GFN2_XTB,
        sample_name="H2O",
        temperature=temperature,
        dtype=torch.float32,
    )
    metrics = certification_metrics(problem, state)
    effective = _resolve_tolerance(1.0e-12, state.x.dtype)
    assert effective == pytest.approx(5.0e-5)
    assert metrics.fermi_ok
    assert metrics.finite_ok
    assert torch.isfinite(metrics.fixed_point.l2)
    assert torch.isfinite(metrics.stationarity_ao_l2)
    assert torch.isfinite(metrics.electron_count_error)
    assert torch.isfinite(metrics.electronic_free_energy)
    assert case.inputs.nel.sum().item() == pytest.approx(
        metrics.electron_count.item(), abs=2e-6
    )


def test_fractional_float32_count_error_stays_below_effective_root_floor() -> (
    None
):
    """Material Fermi occupations retain the count check within f32 floor."""
    row = characterize_case(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        temperature=2000.0,
        dtype=torch.float32,
        tolerance=1.0e-12,
    )
    assert row["model"] == "GFN2-xTB"
    assert row["effective_root_tolerance"] == pytest.approx(5.0e-5)
    assert row["fractional_occupation_count"] > 0
    assert 1.0e-3 < row["fractional_occupation_min"] < 1.0
    assert abs(row["electron_count_error"]) < row["effective_root_tolerance"]
    assert row["fermi_ok"] and row["finite_ok"]


def test_padded_residual_entries_do_not_contribute() -> None:
    """Representation residual projection excludes inactive AO entries."""
    case, problem, _ = _problem_and_state()
    delta = torch.ones_like(case.x)
    projected = problem.project_residual(delta)
    for row, shape in enumerate(problem.inputs.charge_layout.shapes):
        length = 1
        for dimension in shape:
            length *= dimension
        assert torch.count_nonzero(projected[row, length:]) == 0


@pytest.mark.parametrize(
    "representation",
    [
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ],
)
def test_certification_thresholds_are_explicit_and_representation_bound(
    representation: SCFRepresentation,
) -> None:
    """Explicit thresholds yield a verdict and reject representation mismatch."""
    _, problem, state = _problem_and_state(representation=representation)
    result = certify_scf_state(
        problem, state, _loose_thresholds(representation)
    )
    assert result.certified, result.metrics
    strict = replace(
        _loose_thresholds(representation),
        stationarity_orthogonalized_per_sqrt_orbital=0.0,
    )
    assert not certify_scf_state(problem, state, strict).certified
    wrong = _loose_thresholds(SCFRepresentation.CHARGE)
    if representation is not SCFRepresentation.CHARGE:
        with pytest.raises(ValueError, match="do not match"):
            certify_scf_state(problem, state, wrong)


@pytest.mark.parametrize("temperature", [0.0, 300.0], ids=["aufbau", "fermi"])
def test_energy_observables_match_existing_solver_result(
    temperature: float,
) -> None:
    """Extracted energy helpers retain current physical partition values."""
    if temperature > 0.0:
        case, problem, state = _problem_and_state(
            GFN2_XTB,
            sample_name="NO2",
            spin=1.0,
            temperature=temperature,
        )
    else:
        case, problem, state = _problem_and_state(temperature=temperature)
    spectral = state.map_result.spectral
    expected_energy = case.inputs.index.reduce_orbital_to_atom(
        spectral.orbital_energy
    ) + interaction_energy_from_charges(spectral.charges, case.inputs)
    torch.testing.assert_close(
        electronic_energy_from_spectral(spectral, case.inputs), expected_energy
    )
    assert (
        free_energy_from_spectral(spectral, case.inputs).shape
        == case.inputs.numbers.shape
    )
    metrics = certification_metrics(problem, state)
    torch.testing.assert_close(metrics.electronic_energy, expected_energy.sum())


def test_core_certification_matches_full_metrics() -> None:
    """Optional observables do not change the core certification values."""
    _, problem, state = _problem_and_state(
        GFN2_XTB, sample_name="NO2", spin=1.0, temperature=300.0
    )
    core = certification_core_metrics(problem, state)
    full = certification_metrics(problem, state)
    torch.testing.assert_close(core.fixed_point.l2, full.fixed_point.l2)
    torch.testing.assert_close(core.stationarity_ao_l2, full.stationarity_ao_l2)
    torch.testing.assert_close(
        core.stationarity_orthogonalized_l2_per_sqrt_orbital,
        full.stationarity_orthogonalized_l2_per_sqrt_orbital,
    )
    torch.testing.assert_close(
        core.electron_count_error, full.electron_count_error
    )
    assert bool(core.finite_ok)
    assert bool(full.finite_ok)
    assert core.fermi_ok == full.fermi_ok


@pytest.mark.parametrize(
    "observable",
    ["electronic_energy_from_spectral", "free_energy_from_spectral"],
)
def test_full_diagnostics_require_finite_optional_observables(
    observable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional NaN observables fail full but not core finite validity."""
    _, problem, state = _problem_and_state(sample_name="H2O")
    core = certification_core_metrics(problem, state)
    assert bool(core.finite_ok)

    def return_nan(spectral, inputs):
        return spectral.orbital_energy.new_full(inputs.numbers.shape, torch.nan)

    monkeypatch.setattr(certification_module, observable, return_nan)
    full = certification_metrics(problem, state, core_metrics=core)
    assert not bool(full.finite_ok)
    result = certify_scf_state(
        problem, state, _loose_thresholds(problem.representation)
    )
    assert not bool(result.finite_ok)
    assert not bool(result.certified)


@pytest.mark.parametrize("copy_standard", [True, False])
def test_standard_profile_tolerance_inheritance_is_explicit(
    copy_standard: bool,
) -> None:
    """Copied and modified standard profiles retain solver inheritance."""
    from dataclasses import replace

    _, problem, state = _problem_and_state()
    profile = replace(SCF_CERTIFICATION_STANDARD)
    if not copy_standard:
        profile = replace(
            profile,
            stationarity_orthogonalized_per_sqrt_orbital=5.0e-6,
        )
    thresholds = resolve_certification_thresholds(
        problem,
        x_atol=1.0e-8,
        x_atol_max=2.0e-8,
        dtype=state.x.dtype,
        profile=profile,
    )
    assert thresholds.fixed_point_l2 == 1.0e-8
    assert thresholds.fixed_point_max == 2.0e-8
    if not copy_standard:
        assert thresholds.stationarity_orthogonalized_per_sqrt_orbital == 5.0e-6


def test_fermi_validity_and_finite_validity_are_independent() -> None:
    """Unsupported matrix-Fermi output fails despite an OK Fermi status."""
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        scf_options={"fermi_etemp": 300.0},
    )
    inputs = replace(
        case.inputs,
        integrals=replace(
            case.inputs.integrals,
            hcore=case.inputs.integrals.hcore * 1.0e20,
        ),
    )
    problem = SCFProblem(inputs, SCFRepresentation.CHARGE)
    broken_state = problem.evaluate(case.x)
    metrics = certification_metrics(problem, broken_state)
    assert broken_state.map_result.fermi_status is not None
    assert bool(broken_state.map_result.fermi_status.converged.all())
    assert metrics.fermi_ok
    assert not metrics.finite_ok
    assert not certify_scf_state(
        problem, broken_state, _loose_thresholds(problem.representation)
    ).certified


def test_independent_electron_count_threshold_rejects_perturbed_density() -> (
    None
):
    """A finite state with valid Fermi status can fail the independent count."""
    _, problem, state = _problem_and_state(
        GFN2_XTB, sample_name="NO2", spin=1.0, temperature=300.0
    )
    spectral = state.map_result.spectral
    perturbation = (
        torch.eye(spectral.density.shape[-1], dtype=spectral.density.dtype)
        * 1.0e-3
    )
    bad_spectral = replace(spectral, density=spectral.density + perturbation)
    bad_map = replace(state.map_result, spectral=bad_spectral)
    bad_state = replace(state, map_result=bad_map)
    result = certify_scf_state(
        problem,
        bad_state,
        replace(
            _loose_thresholds(problem.representation),
            electron_count=1.0e-5,
        ),
    )
    assert result.metrics.finite_ok
    assert result.metrics.fermi_ok
    assert result.metrics.electron_count_error.abs() > 1.0e-5
    assert not result.electron_count_ok
    assert not result.certified


def test_loose_root_has_larger_fresh_fock_stationarity_than_tight_root() -> (
    None
):
    """Fresh-Fock stationarity resolves beyond a loose raw-root threshold."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    problem = SCFProblem(case.inputs, SCFRepresentation.CHARGE)

    def solve(tolerance: float) -> SCFPhysicalState:
        settings = SCFSolverSettings(
            maxiter=100,
            mix_guess=False,
            damp=0.2,
            x_atol=tolerance,
            x_atol_max=tolerance,
        )
        outcome = _solve_map(
            case.x,
            settings,
            problem.map,
        )
        assert bool(outcome.converged)
        return SCFPhysicalState(outcome.root_x, outcome.map_result)

    loose = solve(1.0e-6)
    tight = solve(1.0e-10)
    loose_metrics = certification_metrics(problem, loose)
    tight_metrics = certification_metrics(problem, tight)
    assert loose_metrics.fixed_point.l2 < 1.0e-6
    assert tight_metrics.fixed_point.l2 < 1.0e-10
    assert (
        loose_metrics.stationarity_ao_l2
        > tight_metrics.stationarity_ao_l2 * 10.0
    )


def test_fermi_failure_is_reported_without_changing_physical_metrics() -> None:
    """Fermi failure remains distinct from fixed-point and stationarity data."""
    _, problem, state = _problem_and_state(temperature=300.0)
    status = state.map_result.fermi_status
    assert status is not None
    failed_status = replace(
        status, converged=torch.zeros_like(status.converged)
    )
    failed_spectral = replace(
        state.map_result.spectral, fermi_status=failed_status
    )
    failed_state = SCFPhysicalState(
        state.x,
        replace(state.map_result, spectral=failed_spectral),
    )
    metrics = certification_metrics(problem, failed_state)
    assert not metrics.fermi_ok
    assert metrics.finite_ok
    assert torch.isfinite(metrics.fixed_point.l2)
    assert torch.isfinite(metrics.stationarity_ao_l2)
    assert not certify_scf_state(
        problem, failed_state, _loose_thresholds(problem.representation)
    ).certified


def test_loose_fixed_point_threshold_does_not_imply_stationarity() -> None:
    """Raw root acceptance and physical stationarity remain separate metrics."""
    _, problem, state = _problem_and_state(sample_name="H2O")
    thresholds = SCFCertificationThresholds(
        representation=problem.representation,
        fixed_point_l2=1.0e6,
        fixed_point_max=1.0e6,
        stationarity_orthogonalized_per_sqrt_orbital=1.0e-12,
        electron_count=1.0,
    )
    result = certify_scf_state(problem, state, thresholds)
    assert result.fixed_point_ok
    assert not result.stationarity_ok
    assert not result.certified


def test_finite_temperature_status_and_free_energy_are_recorded() -> None:
    """Finite-temperature certification reports Fermi status and total F."""
    _, problem, state = _problem_and_state(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        temperature=1000.0,
    )
    metrics = certification_metrics(problem, state)
    assert state.map_result.fermi_status is not None
    assert metrics.fermi_ok
    assert torch.isfinite(metrics.electronic_energy)
    assert torch.isfinite(metrics.electronic_free_energy)
    assert not torch.equal(
        metrics.electronic_energy, metrics.electronic_free_energy
    )


def test_free_energy_partition_changes_distribution_not_total() -> None:
    """Equal and atomic free-energy partitions retain the same total."""
    case, _, state = _problem_and_state(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        temperature=1000.0,
    )
    assert case.inputs.fermi_settings is not None
    settings = case.inputs.fermi_settings
    equal = free_energy_from_spectral(state.map_result.spectral, case.inputs)
    atomic_inputs = replace(
        case.inputs,
        fermi_settings=replace(
            settings, partition=labels.FERMI_PARTITION_ATOMIC
        ),
    )
    atomic = free_energy_from_spectral(state.map_result.spectral, atomic_inputs)
    assert not torch.allclose(equal, atomic)
    torch.testing.assert_close(
        equal.sum(), atomic.sum(), atol=5e-10, rtol=5e-10
    )


def test_near_smearing_boundary_certification_is_finite() -> None:
    """Physical metrics remain finite just above the accepted smear cutoff."""
    _, problem, state = _problem_and_state(temperature=0.1)
    metrics = certification_metrics(problem, state)
    assert metrics.fermi_ok
    assert metrics.finite_ok


def test_progress_compares_existing_states_without_extra_map() -> None:
    """Progress uses two existing observations and reports energy changes."""
    _, problem, first = _problem_and_state()
    second = problem.evaluate(first.x + 1.0e-4)
    progress = scf_progress_metrics(problem, first, second)
    torch.testing.assert_close(
        progress.energy_change,
        progress.current_energy - progress.previous_energy,
    )
    torch.testing.assert_close(
        progress.free_energy_change,
        progress.current_free_energy - progress.previous_free_energy,
    )
    torch.testing.assert_close(
        progress.absolute_energy_change, progress.energy_change.abs()
    )


def test_certification_does_not_evaluate_another_scf_map(monkeypatch) -> None:
    """Fresh-Fock certification uses algebra, not an additional spectrum."""
    _, problem, state = _problem_and_state(sample_name="H2")

    def unexpected_map(*args, **kwargs):
        raise AssertionError("certification must not evaluate an SCF map")

    monkeypatch.setattr(SCFProblem, "map", unexpected_map)
    metrics = certification_metrics(problem, state)
    assert torch.isfinite(metrics.stationarity_ao_l2)


def test_certification_metrics_support_vmap_jvp_jacfwd_and_nested_derivative() -> (
    None
):
    """Continuous fresh-Fock metrics compose with torch.func transforms."""
    _, problem, state = _problem_and_state(sample_name="H2")
    direction = torch.ones_like(state.x)

    def metric(x: torch.Tensor) -> torch.Tensor:
        observation = problem.evaluate(x)
        residual = stationarity_matrix(problem, observation)
        return residual.square().sum()

    value, tangent = jvp(metric, (state.x,), (direction,))
    assert torch.isfinite(value)
    assert torch.isfinite(tangent)
    jacobian = jacfwd(metric)(state.x)
    assert torch.isfinite(jacobian).all()
    second = jacfwd(jacfwd(metric))(state.x)
    assert torch.isfinite(second).all()

    candidates = torch.stack((state.x, state.x + 1.0e-5 * direction))
    vmapped = vmap(metric)(candidates)
    explicit = torch.stack(tuple(metric(x) for x in candidates))
    torch.testing.assert_close(vmapped, explicit, atol=2e-12, rtol=2e-12)


def test_fresh_fock_stationarity_requires_no_spectral_evaluation(
    monkeypatch,
) -> None:
    """Stationarity construction does not diagonalize the rebuilt Hamiltonian."""
    _, problem, state = _problem_and_state(sample_name="H2")
    import dxtb._src.scf.functional.map as physical_map

    def unexpected_diagonalization(*args, **kwargs):
        raise AssertionError("fresh-Fock certification must not diagonalize")

    monkeypatch.setattr(
        physical_map, "evaluate_hamiltonian", unexpected_diagonalization
    )
    metrics = certification_metrics(problem, state)
    assert torch.isfinite(metrics.stationarity_ao_l2)


def test_float32_effective_root_tolerance_is_reported() -> None:
    """Diagnostics retain requested and effective float32 root tolerances."""
    case, problem, state = _problem_and_state(
        dtype=torch.float32, sample_name="H2"
    )
    metrics = certification_metrics(problem, state)
    requested = 1.0e-12
    effective = _resolve_tolerance(requested, state.x.dtype)
    assert effective > requested
    assert metrics.fixed_point.l2.dtype is torch.float32
    assert metrics.stationarity_ao_l2.dtype is torch.float32
    assert metrics.electron_count_error.dtype is torch.float32
    config = replace(
        case.system.config.scf,
        maxiter=1,
        mixer=labels.MIXER_LINEAR,
        x_atol=requested,
        x_atol_max=requested,
    )
    settings = setup_simple_solver_settings(config, state.x.dtype)
    outcome = _solve_map(state.x, settings, problem.map, problem=problem)
    assert outcome.certification is not None
    diagnostic = outcome.certification
    assert (
        diagnostic.requested_fixed_point_l2_tolerance.item()
        == pytest.approx(requested)
    )
    assert (
        diagnostic.requested_fixed_point_max_tolerance.item()
        == pytest.approx(requested)
    )
    assert (
        diagnostic.effective_fixed_point_l2_tolerance.item()
        == pytest.approx(effective)
    )
    assert (
        diagnostic.effective_fixed_point_max_tolerance.item()
        == pytest.approx(effective)
    )
    assert diagnostic.effective_stationarity_tolerance.item() == pytest.approx(
        1.0e-5
    )
    assert (
        diagnostic.effective_electron_count_tolerance.item()
        == pytest.approx(effective)
    )


@pytest.mark.parametrize(
    "method,representation",
    [
        (LinearMethod(), SCFRepresentation.CHARGE),
        (
            AndersonMethod(AndersonMethodSettings(0.1, 4, True, 0.01)),
            SCFRepresentation.CHARGE,
        ),
        (BroydenMethod(), SCFRepresentation.CHARGE),
        (CDIISMethod(), SCFRepresentation.FOCK),
    ],
    ids=["linear", "anderson", "broyden", "cdiis"],
)
def test_terminal_nonfinite_candidate_does_not_reach_method(
    monkeypatch: pytest.MonkeyPatch,
    method: SCFMethod,
    representation: SCFRepresentation,
) -> None:
    """A bad map is frozen and the method sees only finite transition data."""
    import dxtb._src.scf.functional.solver as solver_module

    _, problem, physical = _problem_and_state(
        sample_name="H2O", representation=representation
    )
    calls = 0
    transitions = 0
    inactive_observations = []
    original_transition = solver_module._method_transition

    def map_fn(x: torch.Tensor):
        nonlocal calls
        calls += 1
        mapped = problem.map(x)
        if calls == 2:
            bad_spectral = replace(
                mapped.spectral,
                density=torch.full_like(mapped.spectral.density, torch.nan),
            )
            return replace(
                mapped,
                mapped_x=torch.full_like(mapped.mapped_x, torch.nan),
                spectral=bad_spectral,
            )
        return mapped

    def check_transition(*args, **kwargs):
        nonlocal transitions
        map_result = args[3]
        assert torch.isfinite(map_result.mapped_x).all()
        assert torch.isfinite(map_result.spectral.density).all()
        if not bool(args[7]):
            inactive_observations.append(
                (args[2].clone(), map_result.mapped_x.clone(), args[4].clone())
            )
        transitions += 1
        proposal = original_transition(*args, **kwargs)
        if not bool(args[7]):
            before = solver_module._method_state_tensors(args[5])
            after = solver_module._method_state_tensors(proposal.state)
            assert len(before) == len(after)
            for old_value, new_value in zip(before, after):
                torch.testing.assert_close(
                    new_value, old_value, rtol=0.0, atol=0.0
                )
        return proposal

    monkeypatch.setattr(solver_module, "_method_transition", check_transition)
    outcome = _solve_map_control(
        physical.x,
        maxiter=2,
        mix_guess=True,
        x_atol=1.0e-8,
        x_atol_max=1.0e-8,
        map_fn=map_fn,
        problem=problem,
        method=method,
        damping_settings=DynamicDampingSettings(),
    )
    assert calls == 3
    assert transitions == 3
    assert not bool(outcome.converged)
    assert bool(outcome.terminal_failed)
    assert outcome.certification is not None
    assert torch.equal(
        outcome.terminal_failed, outcome.certification.terminal_failed
    )
    assert int(outcome.certification.failure_code) == int(
        SCFCoreFailure.NONFINITE
    )
    assert torch.isfinite(outcome.root_x).all()
    assert not torch.isfinite(outcome.map_result.spectral.density).all()
    assert len(inactive_observations) == 2
    for observation in inactive_observations[1:]:
        for actual, expected in zip(observation, inactive_observations[0]):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
