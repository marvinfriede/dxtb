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

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.scf.functional import (
    SCFCertificationThresholds,
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
    SCFSolverSettings,
    certification_metrics,
    certify_scf_state,
    electronic_energy_from_spectral,
    fock_initial_guess,
    free_energy_from_spectral,
    interaction_energy_from_charges,
    potential_initial_guess,
    scf_map,
    scf_progress_metrics,
    stationarity_matrix,
)
from dxtb._src.scf.functional.solver import _resolve_tolerance, _solve_map

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
        stationarity_l2=1.0e6,
        stationarity_max=1.0e6,
        electron_count=1.0e6,
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
    """The AO commutator norm is invariant under simultaneous AO permutation."""
    _, problem, state = _problem_and_state(sample_name="H2O")
    spectral = state.map_result.spectral
    fock = problem.rebuild_hamiltonian(spectral.charges)
    density = spectral.density
    overlap = problem.inputs.integrals.overlap
    mask = problem.inputs.orbital_mask
    residual = stationarity_matrix(problem, state)
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
    strict = replace(_loose_thresholds(representation), stationarity_l2=0.0)
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
    assert loose_metrics.stationarity_l2 > tight_metrics.stationarity_l2 * 10.0


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
    assert torch.isfinite(metrics.stationarity_l2)
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
        stationarity_l2=1.0e-12,
        stationarity_max=1.0e-12,
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
    assert torch.isfinite(metrics.stationarity_l2)


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
    assert torch.isfinite(metrics.stationarity_l2)


def test_float32_effective_root_tolerance_is_reported() -> None:
    """Certification can report metrics at the existing float32 floor."""
    case, problem, state = _problem_and_state(
        dtype=torch.float32, sample_name="H2"
    )
    metrics = certification_metrics(problem, state)
    requested = 1.0e-12
    effective = _resolve_tolerance(requested, state.x.dtype)
    assert effective > requested
    assert metrics.fixed_point.l2.dtype is torch.float32
    assert metrics.stationarity_l2.dtype is torch.float32
    assert metrics.electron_count_error.dtype is torch.float32
    del case
