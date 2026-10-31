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
"""Mathematical checks for fixed-occupation orbital response."""

from __future__ import annotations

import inspect
from dataclasses import replace
from test.test_scf.test_functional_map import _prepare_case

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.scf.functional import unpack_charges
from dxtb._src.scf.functional.broyden import BroydenMethod
from dxtb._src.scf.functional.cdiis import CDIISMethod
from dxtb._src.scf.functional.data import FermiSettings
from dxtb._src.scf.functional.energy import (
    electronic_energy_from_density,
    electronic_energy_from_spectral,
)
from dxtb._src.scf.functional.implicit import (
    _prepare_scf_root,
    _SCFMapTensorTree,
    implicit_root,
)
from dxtb._src.scf.functional.map import (
    density_to_charges_and_core_energy,
    fock_initial_guess,
    potential_initial_guess,
)
from dxtb._src.scf.functional.method import SCFMethodContext
from dxtb._src.scf.functional.orbital import build_orbital_model
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
from dxtb._src.scf.functional.soscf import (
    SOSCFMethod,
    SOSCFState,
    _build_fock_seed,
    _pcg_newton_step,
)
from dxtb._src.scf.functional.stationarity import (
    ao_stationarity_matrix,
    orthogonalized_stationarity_matrix,
)
from dxtb._src.scf.functional.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)


def _orbital_model(
    method: int,
    name: str,
    *,
    charge: float = 0.0,
    spin: float = 0.0,
):
    """Build one fixed-occupation Fock-space orbital model."""
    case = _prepare_case(
        method,
        sample_name=name,
        charge=charge,
        spin=spin,
        dtype=torch.float64,
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    physical = problem.evaluate(initial)
    return (
        case,
        problem,
        physical,
        build_orbital_model(problem, physical.map_result.spectral),
    )


@pytest.mark.parametrize(
    ("method", "name", "charge", "spin"),
    [
        (GFN1_XTB, "H2", 0.0, 0.0),
        (GFN1_XTB, "H2O", 0.0, 0.0),
        (GFN2_XTB, "H2O", 0.0, 0.0),
        (GFN2_XTB, "NO2", 0.0, 1.0),
        (GFN2_XTB, "H2O", 1.0, 1.0),
    ],
    ids=["h2-gfn1", "h2o-gfn1", "h2o-gfn2", "no2-gfn2", "h2o-plus-gfn2"],
)
def test_soscf_orbital_gradient_matches_fixed_occupation_energy_ad(
    method: int, name: str, charge: float, spin: float
) -> None:
    """The analytic xTB orbital gradient is the derivative of its energy."""
    _, _, physical, model = _orbital_model(
        method, name, charge=charge, spin=spin
    )
    rotation = torch.zeros(
        model.nrot, dtype=physical.x.dtype, device=physical.x.device
    )
    analytic = model.physical_gradient_at(rotation)
    automatic = torch.func.grad(model.energy_at)(rotation)
    torch.testing.assert_close(analytic, automatic, rtol=2.0e-9, atol=2.0e-10)


def test_soscf_one_pair_gradient_sign_matches_finite_difference() -> None:
    """Lock the occupied-to-virtual rotation sign and gradient factor."""
    _, _, physical, model = _orbital_model(GFN1_XTB, "H2")
    zero = physical.x.new_zeros(model.nrot)
    step = physical.x.new_tensor(1.0e-5)
    direction = torch.zeros_like(zero)
    direction[0] = 1.0
    finite_difference = (
        model.energy_at(step * direction) - model.energy_at(-step * direction)
    ) / (2.0 * step)
    analytic = model.physical_gradient_at(zero)[0]
    torch.testing.assert_close(
        analytic, finite_difference, rtol=1.0e-8, atol=2.0e-10
    )


def test_soscf_stretched_h2_gradient_matches_energy_ad() -> None:
    """Small-gap fixed-occupation H2 retains the same gradient identity."""
    positions = torch.tensor(
        [[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]], dtype=torch.float64
    )
    case = _prepare_case(
        GFN1_XTB, sample_name="H2", positions=positions, dtype=torch.float64
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    initial = fock_initial_guess(charges, case.inputs)
    physical = problem.evaluate(initial)
    model = build_orbital_model(problem, physical.map_result.spectral)
    zero = initial.new_zeros(model.nrot)
    torch.testing.assert_close(
        model.physical_gradient_at(zero),
        torch.func.grad(model.energy_at)(zero),
        rtol=2.0e-9,
        atol=2.0e-10,
    )


def test_soscf_gradient_uses_fresh_physical_fock() -> None:
    """The gradient Fock is the map output rebuilt from the current density."""
    _, problem, physical, model = _orbital_model(GFN2_XTB, "H2O")
    fresh = physical.map_result.mapped_x
    rebuilt = problem.rebuild_hamiltonian(physical.map_result.spectral.charges)
    torch.testing.assert_close(fresh, rebuilt, rtol=2.0e-12, atol=2.0e-12)
    zero = physical.x.new_zeros(model.nrot)
    torch.testing.assert_close(
        model.physical_gradient_at(zero, fresh),
        model.physical_gradient_at(zero, rebuilt),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("method", "name", "spin"),
    [
        (GFN1_XTB, "H2", 0.0),
        (GFN1_XTB, "H2O", 0.0),
        (GFN2_XTB, "H2O", 0.0),
        (GFN2_XTB, "NO2", 1.0),
    ],
    ids=["h2-gfn1", "h2o-gfn1", "h2o-gfn2", "no2-gfn2"],
)
def test_soscf_hessian_vector_matches_energy_hessian(
    method: int, name: str, spin: float
) -> None:
    """Local HVPs match the dense test Hessian and remain symmetric."""
    _, _, physical, model = _orbital_model(method, name, charge=0.0, spin=spin)
    zero = physical.x.new_zeros(model.nrot)
    hessian = torch.func.hessian(model.energy_at)(zero)
    direction = torch.linspace(-0.5, 0.5, model.nrot, dtype=zero.dtype)
    hvp = torch.func.jvp(
        torch.func.grad(model.energy_at), (zero,), (direction,)
    )[1]
    torch.testing.assert_close(hvp, hessian @ direction, rtol=2.0e-9, atol=2e-9)
    torch.testing.assert_close(hessian, hessian.mT, rtol=0.0, atol=2e-9)


def test_soscf_hvp_matches_centered_gradient_difference() -> None:
    """The matrix-free HVP matches an independent centered gradient sweep."""
    _, _, physical, model = _orbital_model(GFN2_XTB, "H2O")
    zero = physical.x.new_zeros(model.nrot)
    direction = torch.linspace(-0.4, 0.6, model.nrot, dtype=zero.dtype)
    direction = direction / torch.linalg.vector_norm(direction)
    hvp = torch.func.jvp(
        torch.func.grad(model.energy_at), (zero,), (direction,)
    )[1]
    step = zero.new_tensor(2.0e-5)
    finite_difference = (
        torch.func.grad(model.energy_at)(step * direction)
        - torch.func.grad(model.energy_at)(-step * direction)
    ) / (2.0 * step)
    torch.testing.assert_close(hvp, finite_difference, rtol=2.0e-6, atol=2.0e-7)


def test_soscf_gradient_norm_matches_s2_stationarity_scaling() -> None:
    """The orbital-gradient norm has the expected S2 commutator scaling."""
    _, problem, physical, model = _orbital_model(GFN2_XTB, "H2O")
    fresh = physical.map_result.mapped_x
    density = physical.map_result.spectral.density
    ao = ao_stationarity_matrix(
        fresh,
        density,
        problem.inputs.integrals.overlap,
        problem.inputs.orbital_mask,
    )
    orthogonalized = orthogonalized_stationarity_matrix(
        ao,
        problem.inputs.inverse_cholesky,
        problem.inputs.orbital_mask,
    )
    gradient = model.physical_gradient_at(
        physical.x.new_zeros(model.nrot), fresh
    )
    expected = 2**0.5 * orthogonalized.norm()
    torch.testing.assert_close(
        gradient.norm(), expected, rtol=2.0e-9, atol=2.0e-10
    )


def test_soscf_density_energy_matches_spectral_energy() -> None:
    """The arbitrary-density energy reuses the accepted spectral partition."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    density_energy = electronic_energy_from_density(
        physical.map_result.spectral.density, problem.inputs
    )
    spectral_energy = electronic_energy_from_spectral(
        physical.map_result.spectral, problem.inputs
    )
    torch.testing.assert_close(
        density_energy.sum(), spectral_energy.sum(), rtol=0.0, atol=2.0e-13
    )


def test_soscf_open_shell_pairs_cover_all_unequal_occupation_sectors() -> None:
    """Open-shell rotations include 2-to-1, 2-to-0, and 1-to-0 sectors."""
    _, _, physical, model = _orbital_model(GFN2_XTB, "NO2", spin=1.0)
    total = physical.map_result.spectral.occupation.sum(dim=-2)
    sectors = {
        (float(total[i]), float(total[a]))
        for i, a in zip(model.pair_i.tolist(), model.pair_a.tolist())
    }
    assert {(2.0, 1.0), (2.0, 0.0), (1.0, 0.0)} <= sectors


def test_soscf_rotation_preserves_orthogonality_and_electron_count() -> None:
    """Finite matrix-exponential rotations preserve metric and particle count."""
    _, problem, physical, model = _orbital_model(GFN2_XTB, "H2O")
    rotation = torch.linspace(-0.02, 0.02, model.nrot, dtype=physical.x.dtype)
    coefficients = model.coefficients_at(rotation)
    overlap = problem.inputs.integrals.overlap
    torch.testing.assert_close(
        coefficients.mT @ overlap @ coefficients,
        torch.eye(coefficients.shape[-1], dtype=coefficients.dtype),
        rtol=0.0,
        atol=2.0e-12,
    )
    density = model.density_at(rotation)
    count = torch.trace(density @ overlap)
    torch.testing.assert_close(
        count, problem.inputs.nel.sum(), rtol=0.0, atol=6.0e-12
    )


def test_soscf_fock_seed_reproduces_accepted_orbital_observation() -> None:
    """Fresh-Fock gauge blocks seed the same rotated density and map."""
    _, problem, physical, model = _orbital_model(GFN2_XTB, "H2O")
    rotation = torch.linspace(-0.01, 0.01, model.nrot, dtype=physical.x.dtype)
    fresh_fock = model.fock_at(rotation)
    seed = _build_fock_seed(model, rotation, fresh_fock)
    seeded = problem.evaluate(seed)
    expected_density = model.density_at(rotation)
    torch.testing.assert_close(
        seeded.map_result.spectral.density,
        expected_density,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    expected_charges, _ = density_to_charges_and_core_energy(
        expected_density, problem.inputs
    )
    torch.testing.assert_close(
        seeded.map_result.spectral.charges.mono,
        expected_charges.mono,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        seeded.map_result.mapped_x, fresh_fock, rtol=2.0e-10, atol=2.0e-10
    )


def test_soscf_stationary_seed_matches_certified_fock_root() -> None:
    """At an S2 root the gauge-block seed equals its fresh physical Fock."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    outcome = solve_scf_problem(
        physical.x,
        problem,
        SCFSolverSettings(100, True, 1.0, 1.0e-9, 1.0e-9),
        SCFStrategy((SCFStage(CDIISMethod()),)),
        eager=True,
    )
    assert bool(outcome.converged)
    root = SCFPhysicalState(outcome.root_x, outcome.map_result)
    fresh = outcome.map_result.mapped_x
    model = build_orbital_model(problem, outcome.map_result.spectral)
    seed = _build_fock_seed(model, outcome.root_x.new_zeros(model.nrot), fresh)
    torch.testing.assert_close(seed, fresh, rtol=0.0, atol=2.0e-8)
    metrics = outcome.diagnostics.certification.result.metrics
    assert metrics.stationarity_orthogonalized_l2_per_sqrt_orbital < 1.0e-8
    assert torch.linalg.vector_norm(root.delta) < 2.0e-8


def test_soscf_stationary_orbitals_reseed_stale_input_fock() -> None:
    """A zero orbital gradient still repairs stale Fock eigenvalues."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    outcome = solve_scf_problem(
        physical.x,
        problem,
        SCFSolverSettings(120, True, 1.0, 1.0e-10, 1.0e-10),
        SCFStrategy((SCFStage(CDIISMethod()),)),
        eager=True,
    )
    assert bool(outcome.converged)

    spectral = outcome.map_result.spectral
    overlap = problem.inputs.integrals.overlap
    coefficients = spectral.coefficients
    offsets = torch.linspace(
        -0.01, 0.01, coefficients.shape[-1], dtype=coefficients.dtype
    )
    stale_energies = spectral.emo + offsets
    stale_input = (
        overlap
        @ coefficients
        @ torch.diag(stale_energies)
        @ coefficients.mT
        @ overlap
    )
    physical_stale = problem.evaluate(stale_input)
    assert torch.linalg.vector_norm(physical_stale.delta) > 1.0e-3

    context = SCFMethodContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=stale_input.new_tensor(1.0),
        proposal_x=physical_stale.map_result.mapped_x,
    )
    proposal = SOSCFMethod().propose(problem, physical_stale, None, context)
    assert isinstance(proposal.state, SOSCFState)
    torch.testing.assert_close(
        proposal.x,
        physical_stale.map_result.mapped_x,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    assert proposal.state.last_gradient_norm < 1.0e-7
    assert proposal.state.last_hvp_count == 0


def test_soscf_equal_occupation_gauge_rotation_is_physically_invariant() -> (
    None
):
    """Rotations inside a doubly occupied subspace leave observables intact."""
    _, problem, physical, model = _orbital_model(GFN2_XTB, "H2O")
    total = physical.map_result.spectral.occupation.sum(dim=-2)
    occupied = torch.nonzero(total == 2.0).flatten()
    assert occupied.numel() >= 2
    coefficients = model.coefficients.clone()
    generator = torch.zeros_like(coefficients)
    angle = coefficients.new_tensor(0.31)
    i, j = int(occupied[0]), int(occupied[1])
    generator = generator.index_put(
        (torch.tensor([j, i]), torch.tensor([i, j])),
        torch.stack((angle, -angle)),
    )
    rotated = coefficients @ torch.matrix_exp(generator)
    occupation = total
    density0 = (coefficients * occupation.unsqueeze(-2)) @ coefficients.mT
    density1 = (rotated * occupation.unsqueeze(-2)) @ rotated.mT
    torch.testing.assert_close(density0, density1, rtol=0.0, atol=2.0e-12)
    charges0, _ = density_to_charges_and_core_energy(density0, problem.inputs)
    charges1, _ = density_to_charges_and_core_energy(density1, problem.inputs)
    torch.testing.assert_close(
        charges0.mono, charges1.mono, rtol=0.0, atol=2e-12
    )
    torch.testing.assert_close(
        electronic_energy_from_density(density0, problem.inputs).sum(),
        electronic_energy_from_density(density1, problem.inputs).sum(),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_soscf_method_makes_finite_energy_decreasing_proposal() -> None:
    """The eager SOSCF method returns a finite Fock seed and diagnostics."""
    _, problem, physical, model = _orbital_model(GFN2_XTB, "H2O")
    context = SCFMethodContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=physical.x.new_tensor(1.0),
        proposal_x=physical.map_result.mapped_x,
    )
    proposal = SOSCFMethod().propose(problem, physical, None, context)
    assert isinstance(proposal.state, SOSCFState)
    assert torch.isfinite(proposal.x).all()
    assert bool(proposal.state.last_step_accepted)
    assert proposal.state.last_actual_reduction > 0.0
    assert proposal.state.last_hvp_count > 0
    assert proposal.state.last_step_norm <= 0.2


def test_soscf_float32_proposal_is_finite() -> None:
    """Dtype-scaled floors keep a single-precision Newton step stable."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O", dtype=torch.float32)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    physical = problem.evaluate(initial)
    context = SCFMethodContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=initial.new_tensor(1.0),
        proposal_x=physical.map_result.mapped_x,
    )
    proposal = SOSCFMethod().propose(problem, physical, None, context)
    assert torch.isfinite(proposal.x).all()
    assert isinstance(proposal.state, SOSCFState)
    assert torch.isfinite(proposal.state.last_inner_residual)


def test_soscf_rejects_active_fermi_smearing() -> None:
    """SOSCF never silently freezes active thermal occupations."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    old_settings = problem.inputs.fermi_settings
    if old_settings is None:
        old_settings = FermiSettings(
            kt=physical.x.new_tensor(0.01),
            smear=False,
            maxiter=100,
            threshold=physical.x.new_tensor(1.0e-10),
            diff_order=1,
            partition=0,
        )
    new_inputs = replace(
        problem.inputs,
        fermi_settings=replace(
            old_settings, kt=physical.x.new_tensor(0.01), smear=True
        ),
    )
    smeared_problem = replace(problem, inputs=new_inputs)
    context = SCFMethodContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=physical.x.new_tensor(1.0),
        proposal_x=physical.map_result.mapped_x,
    )
    with pytest.raises(NotImplementedError, match="active Fermi smearing"):
        SOSCFMethod().propose(smeared_problem, physical, None, context)


def test_soscf_accepts_nonzero_temperature_without_smearing() -> None:
    """Low temperature with a fixed occupation branch remains eligible."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    fixed = FermiSettings(
        kt=physical.x.new_tensor(1.0e-8),
        smear=False,
        maxiter=100,
        threshold=physical.x.new_tensor(1.0e-10),
        diff_order=1,
        partition=0,
    )
    low_temperature_problem = replace(
        problem, inputs=replace(problem.inputs, fermi_settings=fixed)
    )
    model = build_orbital_model(
        low_temperature_problem, physical.map_result.spectral
    )
    assert model.nrot > 0


def test_soscf_proposal_does_not_evaluate_an_extra_physical_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HVPs, line search, and seed construction do not diagonalize/map."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")

    def fail_map(self, value):
        raise AssertionError("SOSCF must not evaluate the SCF map internally.")

    monkeypatch.setattr(SCFProblem, "map", fail_map)
    context = SCFMethodContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=physical.x.new_tensor(1.0),
        proposal_x=physical.map_result.mapped_x,
    )
    proposal = SOSCFMethod().propose(problem, physical, None, context)
    assert torch.isfinite(proposal.x).all()


def test_soscf_inactive_transition_freezes_method_state() -> None:
    """An inactive eager transition preserves every SOSCF diagnostic."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    method = SOSCFMethod()
    active = SCFMethodContext(
        step=1,
        transition_active=torch.tensor(True),
        damp=physical.x.new_tensor(1.0),
        proposal_x=physical.map_result.mapped_x,
    )
    state = method.propose(problem, physical, None, active).state
    assert isinstance(state, SOSCFState)

    inactive = replace(
        active,
        transition_active=torch.tensor(False),
        proposal_x=torch.full_like(physical.x, torch.nan),
    )
    proposal = method.propose(problem, physical, state, inactive)
    assert proposal.state is state
    assert proposal.x is physical.x


def test_soscf_production_is_matrix_free_and_eager_only() -> None:
    """Production uses HVPs and does not materialize orbital Hessians."""
    from dxtb._src.scf.functional import solver as solver_module
    from dxtb._src.scf.functional import soscf

    source = inspect.getsource(soscf)
    controller_source = inspect.getsource(solver_module)
    assert SOSCFMethod.supports_fixed_trip is False
    assert "torch.func.hessian" not in source
    assert "torch.func.jacfwd" not in source
    assert "torch.linalg.solve(" not in source
    assert "SOSCFMethod" not in controller_source


def test_soscf_fixed_trip_is_rejected_before_solver_map() -> None:
    """Fixed-trip controller rejects SOSCF before performing physical work."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    strategy = SCFStrategy((SCFStage(SOSCFMethod()),))
    settings = SCFSolverSettings(10, True, 0.5, 1.0e-7, 1.0e-7)
    with pytest.raises(NotImplementedError, match="fixed-trip"):
        solve_scf_problem(
            physical.x,
            problem,
            settings,
            strategy,
            eager=False,
        )


def test_soscf_cdiis_strategy_reaches_certified_root() -> None:
    """The eager shared controller can switch CDIIS into Fock SOSCF."""
    _, problem, physical, _ = _orbital_model(GFN2_XTB, "H2O")
    strategy = SCFStrategy(
        (
            SCFStage(
                CDIISMethod(),
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=2, stationarity=2.0e-2
                ),
                initial_damp=1.0,
            ),
            SCFStage(SOSCFMethod(), initial_damp=1.0),
        )
    )
    settings = SCFSolverSettings(140, True, 1.0, 1.0e-7, 1.0e-7)
    outcome = _solve_map(
        physical.x,
        settings,
        problem.map,
        eager=True,
        problem=problem,
        strategy=strategy,
    )
    assert bool(outcome.converged)
    assert outcome.certification is not None
    assert bool(outcome.certification.certified)
    assert outcome.strategy_diagnostics is not None
    assert int(outcome.strategy_diagnostics.switch_count) == 1

    cdiis_outcome = _solve_map(
        physical.x,
        settings,
        problem.map,
        eager=True,
        problem=problem,
        strategy=SCFStrategy((SCFStage(CDIISMethod()),)),
    )
    assert bool(cdiis_outcome.converged)

    adapter, parameters = _SCFMapTensorTree.prepare(
        problem.inputs, labels.SCP_MODE_FOCK
    )
    prepared = _prepare_scf_root(
        outcome,
        maxiter=settings.maxiter,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        final_valid=torch.tensor(True),
        raw_map=adapter,
        parameters=parameters,
    )
    assert "method_state" not in prepared.__dataclass_fields__
    hcore_index = adapter.parameter_paths.index("integrals.hcore")
    hcore = parameters[hcore_index]
    direction = torch.zeros_like(hcore)
    direction[0, 0] = 1.0
    origin = hcore.new_zeros(())

    def attached(prepared, delta: torch.Tensor) -> torch.Tensor:
        updated = list(parameters)
        updated[hcore_index] = hcore + delta * direction
        return implicit_root(prepared.root_x, *updated, raw_map=adapter)

    prepared_soscf = _prepare_scf_root(
        outcome,
        maxiter=settings.maxiter,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        final_valid=torch.tensor(True),
        raw_map=adapter,
        parameters=parameters,
    )
    prepared_cdiis = _prepare_scf_root(
        cdiis_outcome,
        maxiter=settings.maxiter,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        final_valid=torch.tensor(True),
        raw_map=adapter,
        parameters=parameters,
    )
    _, tangent_soscf = torch.func.jvp(
        lambda delta: attached(prepared_soscf, delta),
        (origin,),
        (origin.new_ones(()),),
    )
    _, tangent_cdiis = torch.func.jvp(
        lambda delta: attached(prepared_cdiis, delta),
        (origin,),
        (origin.new_ones(()),),
    )
    assert torch.isfinite(tangent_soscf).all()
    assert torch.isfinite(tangent_cdiis).all()
    second_soscf = torch.func.jacfwd(
        torch.func.jacfwd(lambda delta: attached(prepared_soscf, delta))
    )(origin)
    second_cdiis = torch.func.jacfwd(
        torch.func.jacfwd(lambda delta: attached(prepared_cdiis, delta))
    )(origin)
    assert torch.isfinite(second_soscf).all()
    assert torch.isfinite(second_cdiis).all()
    torch.testing.assert_close(
        tangent_soscf, tangent_cdiis, rtol=2.0e-5, atol=2.0e-7
    )
    torch.testing.assert_close(
        second_soscf, second_cdiis, rtol=2.0e-4, atol=2.0e-6
    )


def test_soscf_implicit_geometry_response_matches_cdiis() -> None:
    """Implicit root geometry response is independent of the primal method."""
    from test.test_scf.test_functional_root_derivatives import (
        _case,
        _initial_state,
    )

    from dxtb._src.calculators.singlepoint import _interaction_data
    from dxtb._src.components.interactions import InteractionList
    from dxtb._src.components.interactions.field.efield import ElectricField
    from dxtb._src.integral.evaluation import build_integral_matrices
    from dxtb._src.scf.functional import setup_scf_inputs

    case = _case(
        "H2O",
        method=GFN2_XTB,
        representation=labels.SCP_MODE_FOCK,
        tolerance=1.0e-9,
        maxiter=140,
        dtype=torch.float64,
    )
    initial, _ = _initial_state(case, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    settings = SCFSolverSettings(140, True, 1.0, 1.0e-9, 1.0e-9)
    cdiis = solve_scf_problem(
        initial,
        problem,
        settings,
        SCFStrategy((SCFStage(CDIISMethod()),)),
        eager=True,
    )
    assert bool(cdiis.converged)

    root_state = SCFPhysicalState(cdiis.root_x, cdiis.map_result)
    model = build_orbital_model(problem, cdiis.map_result.spectral)
    rotation = cdiis.root_x.new_zeros(model.nrot)
    rotation[0] = 0.02
    displaced_seed = _build_fock_seed(model, rotation, model.fock_at(rotation))
    soscf = _solve_map(
        displaced_seed,
        SCFSolverSettings(80, False, 1.0, 1.0e-7, 1.0e-7),
        problem.map,
        eager=True,
        problem=problem,
        strategy=SCFStrategy((SCFStage(SOSCFMethod()),)),
    )
    assert bool(soscf.converged)
    assert torch.linalg.vector_norm(soscf.root_x - root_state.x) < 3.0e-7

    def inputs_at_positions(positions: torch.Tensor):
        integrals, _, _ = build_integral_matrices(
            case.system.integral_setup,
            case.system.h0_setup,
            positions,
            case.charge,
        )
        interactions = InteractionList(
            *case.system.interactions.components,
            ElectricField(positions.new_zeros(3)),
            device=positions.device,
            dtype=positions.dtype,
        )
        cache = _interaction_data(case.system, positions, interactions)
        return setup_scf_inputs(
            case.numbers,
            integrals,
            case.n0,
            case.occupation,
            case.nel,
            interactions,
            cache,
            case.system.ihelp,
            case.system.config.scf,
        )

    def attached_geometry(root_x: torch.Tensor, delta: torch.Tensor):
        positions = case.positions.clone()
        positions[0, 0] = positions[0, 0] + delta
        inputs = inputs_at_positions(positions)
        adapter, parameters = _SCFMapTensorTree.prepare(
            inputs, labels.SCP_MODE_FOCK
        )
        return implicit_root(root_x, *parameters, raw_map=adapter)

    origin = case.positions.new_zeros((), requires_grad=True)
    cdiis_attached = attached_geometry(cdiis.root_x, origin)
    cdiis_tangent = torch.autograd.grad(cdiis_attached.sum(), origin)[0]
    soscf_attached = attached_geometry(soscf.root_x, origin)
    soscf_tangent = torch.autograd.grad(soscf_attached.sum(), origin)[0]
    torch.testing.assert_close(
        cdiis_tangent, soscf_tangent, rtol=2.0e-4, atol=2.0e-6
    )


def test_soscf_potential_broyden_strategy_reaches_certified_root() -> None:
    """Potential Broyden switches to Fock SOSCF without a fresh-map stage."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O", dtype=torch.float64)
    charges = unpack_charges(case.x, case.inputs.charge_layout)
    initial, layout = potential_initial_guess(charges, case.inputs)
    problem = SCFProblem(
        case.inputs, SCFRepresentation.POTENTIAL, potential_layout=layout
    )
    strategy = SCFStrategy(
        (
            SCFStage(
                BroydenMethod(),
                representation=SCFRepresentation.POTENTIAL,
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=2, stationarity=2.0e-2
                ),
                initial_damp=0.5,
            ),
            SCFStage(
                SOSCFMethod(),
                representation=SCFRepresentation.FOCK,
                initial_damp=1.0,
            ),
        )
    )
    outcome = solve_scf_problem(
        initial,
        problem,
        SCFSolverSettings(60, True, 0.5, 1.0e-7, 1.0e-7),
        strategy,
        eager=True,
    )
    assert bool(outcome.converged)
    assert outcome.root_x.ndim == 2
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.strategy is not None
    assert int(outcome.diagnostics.strategy.final_stage_index) == 1
    assert int(outcome.diagnostics.strategy.switch_count) == 1
    stage_maps = sum(
        int(value) for value in outcome.diagnostics.strategy.stage_iterations
    )
    assert stage_maps == int(outcome.iterations)


@pytest.mark.parametrize(
    ("method", "name", "charge", "spin", "stretched"),
    [
        (GFN1_XTB, "H2", 0.0, 0.0, False),
        (GFN1_XTB, "H2O", 0.0, 0.0, False),
        (GFN2_XTB, "H2O", 1.0, 1.0, False),
        (GFN1_XTB, "H2", 0.0, 0.0, True),
    ],
    ids=["h2-gfn1", "h2o-gfn1", "h2o-plus-gfn2", "stretched-h2-gfn1"],
)
def test_soscf_reference_cases_reach_certified_fixed_occupation_roots(
    method: int,
    name: str,
    charge: float,
    spin: float,
    stretched: bool,
) -> None:
    """Representative roots certify; easy H2 cases may stop before SOSCF."""
    positions = None
    if stretched:
        positions = torch.tensor(
            [[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]], dtype=torch.float64
        )
    case = _prepare_case(
        method,
        sample_name=name,
        charge=charge,
        spin=spin,
        positions=positions,
        dtype=torch.float64,
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    strategy = SCFStrategy(
        (
            SCFStage(
                CDIISMethod(),
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=2, stationarity=2.0e-2
                ),
                initial_damp=1.0,
            ),
            SCFStage(SOSCFMethod(), initial_damp=1.0),
        )
    )
    outcome = _solve_map(
        initial,
        SCFSolverSettings(140, True, 1.0, 1.0e-7, 1.0e-7),
        problem.map,
        eager=True,
        problem=problem,
        strategy=strategy,
    )
    assert bool(outcome.converged)
    assert outcome.strategy_diagnostics is not None
    expected_stage = 0 if name == "H2" else 1
    assert int(outcome.strategy_diagnostics.final_stage_index) == expected_stage


@pytest.mark.parametrize(
    "stretched", [False, True], ids=["equilibrium", "stretched"]
)
def test_soscf_near_root_h2_gfn1_certifies(stretched: bool) -> None:
    """SOSCF converges from a controlled small orbital displacement in H2."""
    positions = None
    if stretched:
        positions = torch.tensor(
            [[0.0, 0.0, -1.5], [0.0, 0.0, 1.5]], dtype=torch.float64
        )
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        positions=positions,
        dtype=torch.float64,
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    root = solve_scf_problem(
        initial,
        problem,
        SCFSolverSettings(100, True, 1.0, 1.0e-9, 1.0e-9),
        SCFStrategy((SCFStage(CDIISMethod()),)),
        eager=True,
    )
    assert bool(root.converged)
    root_state = SCFPhysicalState(root.root_x, root.map_result)
    model = build_orbital_model(problem, root.map_result.spectral)
    rotation = root.root_x.new_zeros(model.nrot)
    rotation[0] = 0.02
    fresh = model.fock_at(rotation)
    seed = _build_fock_seed(model, rotation, fresh)
    outcome = _solve_map(
        seed,
        SCFSolverSettings(60, False, 1.0, 1.0e-8, 1.0e-8),
        problem.map,
        eager=True,
        problem=problem,
        strategy=SCFStrategy((SCFStage(SOSCFMethod()),)),
    )
    assert bool(outcome.converged)
    assert outcome.certification is not None
    assert bool(outcome.certification.certified)
    torch.testing.assert_close(
        outcome.map_result.spectral.density,
        root_state.map_result.spectral.density,
        rtol=0.0,
        atol=3.0e-7,
    )


def test_soscf_open_shell_no2_strategy_reaches_certified_root() -> None:
    """SOSCF includes NO2 singly occupied rotations and certifies the root."""
    case = _prepare_case(
        GFN2_XTB, sample_name="NO2", spin=1.0, dtype=torch.float64
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    physical = problem.evaluate(initial)
    orbital_model = build_orbital_model(problem, physical.map_result.spectral)
    total_occupation = physical.map_result.spectral.occupation.sum(dim=-2)
    sectors = {
        (float(total_occupation[i]), float(total_occupation[a]))
        for i, a in zip(
            orbital_model.pair_i.tolist(), orbital_model.pair_a.tolist()
        )
    }
    assert {(2.0, 1.0), (2.0, 0.0), (1.0, 0.0)} <= sectors

    strategy = SCFStrategy(
        (
            SCFStage(
                CDIISMethod(),
                switch_policy=SCFSwitchPolicy(
                    min_stage_iterations=2, stationarity=1.0e-3
                ),
                initial_damp=1.0,
            ),
            SCFStage(SOSCFMethod(), initial_damp=1.0),
        )
    )
    outcome = solve_scf_problem(
        initial,
        problem,
        SCFSolverSettings(120, True, 1.0, 1.0e-7, 1.0e-7),
        strategy,
        eager=True,
    )
    assert bool(outcome.converged)
    assert outcome.diagnostics is not None
    assert outcome.diagnostics.certification is not None
    metrics = outcome.diagnostics.certification.result.metrics
    assert metrics.stationarity_orthogonalized_l2_per_sqrt_orbital < 1.0e-7
    assert metrics.fixed_point.l2 < 1.0e-7


def test_soscf_no2_implicit_response_matches_cdiis() -> None:
    """Open-shell NO2 implicit response ignores the primal method history."""
    case = _prepare_case(
        GFN2_XTB, sample_name="NO2", spin=1.0, dtype=torch.float64
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    settings = SCFSolverSettings(120, True, 1.0, 1.0e-7, 1.0e-7)
    cdiis = _solve_map(
        initial,
        settings,
        problem.map,
        eager=True,
        problem=problem,
        strategy=SCFStrategy((SCFStage(CDIISMethod()),)),
    )
    switched = _solve_map(
        initial,
        settings,
        problem.map,
        eager=True,
        problem=problem,
        strategy=SCFStrategy(
            (
                SCFStage(
                    CDIISMethod(),
                    switch_policy=SCFSwitchPolicy(
                        min_stage_iterations=2, stationarity=1.0e-3
                    ),
                    initial_damp=1.0,
                ),
                SCFStage(SOSCFMethod(), initial_damp=1.0),
            )
        ),
    )
    assert bool(cdiis.converged)
    assert bool(switched.converged)
    assert int(switched.strategy_diagnostics.final_stage_index) == 1

    adapter, parameters = _SCFMapTensorTree.prepare(
        case.inputs, labels.SCP_MODE_FOCK
    )
    hcore_index = adapter.parameter_paths.index("integrals.hcore")
    hcore = parameters[hcore_index]
    direction = torch.zeros_like(hcore)
    direction[0, 0] = 1.0
    origin = hcore.new_zeros(())
    unit = origin.new_ones(())

    def attach(root_x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        updated = list(parameters)
        updated[hcore_index] = hcore + delta * direction
        return implicit_root(root_x, *updated, raw_map=adapter)

    prepared_cdiis = _prepare_scf_root(
        cdiis,
        maxiter=settings.maxiter,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        final_valid=torch.tensor(True),
        raw_map=adapter,
        parameters=parameters,
    )
    prepared_soscf = _prepare_scf_root(
        switched,
        maxiter=settings.maxiter,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        final_valid=torch.tensor(True),
        raw_map=adapter,
        parameters=parameters,
    )
    _, tangent_cdiis = torch.func.jvp(
        lambda delta: attach(prepared_cdiis.root_x, delta),
        (origin,),
        (unit,),
    )
    _, tangent_soscf = torch.func.jvp(
        lambda delta: attach(prepared_soscf.root_x, delta),
        (origin,),
        (unit,),
    )
    torch.testing.assert_close(
        tangent_cdiis, tangent_soscf, rtol=3.0e-4, atol=3.0e-6
    )


def test_soscf_inner_pcg_matches_spd_reference() -> None:
    """Preconditioned CG solves a small synthetic SPD Newton system."""
    matrix = torch.tensor([[4.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    gradient = torch.tensor([0.3, -0.7], dtype=torch.float64)
    shift = gradient.new_zeros(())

    def operator(vector: torch.Tensor) -> torch.Tensor:
        return matrix @ vector

    settings = replace(SOSCFMethod().settings, micro_rtol=1.0e-12)
    step, _, _, residual, _, _, fallback = _pcg_newton_step(
        operator,  # type: ignore[arg-type]
        gradient,
        torch.diagonal(matrix),
        shift,
        settings,
    )
    expected = torch.linalg.solve(matrix, -gradient)
    torch.testing.assert_close(step, expected, rtol=1e-10, atol=1e-10)
    assert residual < 1.0e-10
    assert not fallback


def test_soscf_curvature_threshold_scales_with_tiny_newton_direction() -> None:
    """Small converged-state residuals are not mistaken for curvature loss."""
    matrix = torch.tensor([[4.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    gradient = torch.tensor([0.3, -0.7], dtype=torch.float64) * 1.0e-8
    settings = replace(SOSCFMethod().settings, micro_rtol=1.0e-12)

    def operator(vector: torch.Tensor) -> torch.Tensor:
        return matrix @ vector

    step, _, _, residual, _, negative, fallback = _pcg_newton_step(
        operator,  # type: ignore[arg-type]
        gradient,
        torch.diagonal(matrix),
        gradient.new_zeros(()),
        settings,
    )
    expected = torch.linalg.solve(matrix, -gradient)
    torch.testing.assert_close(step, expected, rtol=1.0e-9, atol=1.0e-16)
    assert not bool(negative)
    assert not fallback
    assert residual < 1.0e-16


def test_soscf_inner_pcg_detects_negative_curvature() -> None:
    """Indefinite curvature triggers level shifts and finite descent fallback."""
    matrix = torch.tensor([[-1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    gradient = torch.tensor([0.4, -0.2], dtype=torch.float64)
    settings = SOSCFMethod().settings

    def operator(vector: torch.Tensor) -> torch.Tensor:
        return matrix @ vector

    step, shift, _, _, _, negative, fallback = _pcg_newton_step(
        operator,  # type: ignore[arg-type]
        gradient,
        torch.tensor([1.0, 2.0], dtype=gradient.dtype),
        gradient.new_tensor(0.01),
        settings,
    )
    assert bool(negative)
    assert not fallback or bool(torch.isfinite(step).all())
    assert shift > 0.01
    assert torch.isfinite(step).all()
    assert torch.dot(gradient, step) < 0.0
