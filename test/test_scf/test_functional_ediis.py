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
"""Variational energy/Fock checks for energy-DIIS research."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from benchmarks.prototypes.scf_energy_diis import (
    ResearchDIISMethod,
    _minimize_coefficients,
    adiis_model_matrices,
    exact_energy_value_gradient,
)
from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.constants import labels
from dxtb._src.scf.functional import unpack_charges
from dxtb._src.scf.functional.broyden import BroydenMethod
from dxtb._src.scf.functional.cdiis import CDIISMethod
from dxtb._src.scf.functional.core import FunctionalInteractionSet
from dxtb._src.scf.functional.energy import (
    electronic_energy_from_density,
    free_energy_from_spectral,
)
from dxtb._src.scf.functional.map import (
    density_to_charges_and_core_energy,
    fock_initial_guess,
)
from dxtb._src.scf.functional.method import SCFMethodContext
from dxtb._src.scf.functional.problem import SCFProblem, SCFRepresentation
from dxtb._src.scf.functional.solver import SCFSolverSettings, _solve_map
from dxtb._src.scf.functional.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)

from .test_functional_map import _prepare_case


@pytest.mark.parametrize(
    ("method", "field"),
    [
        (GFN1_XTB, None),
        (GFN2_XTB, None),
        (GFN2_XTB, torch.tensor([0.0, 0.0, 0.001], dtype=torch.float64)),
    ],
    ids=["gfn1", "gfn2", "gfn2-field"],
)
def test_density_energy_jvp_matches_rebuilt_fock(
    method: int, field: torch.Tensor | None
) -> None:
    """The physical Fock is the gradient of xTB density energy."""
    case = _prepare_case(
        method,
        sample_name="H2O",
        field=field,
        dtype=torch.float64,
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    spectral = problem.evaluate(initial).map_result.spectral

    kinds = {entry.kind for entry in case.inputs.interactions.entries}
    if method is GFN1_XTB:
        assert {"es2", "es3"}.issubset(kinds)
    else:
        assert {"es2", "aes2", "es3", "d4sc"}.issubset(kinds)

    torch.manual_seed(401)
    density = spectral.density + 0.03 * torch.randn_like(spectral.density)
    density = 0.5 * (density + density.mT)
    direction = torch.randn_like(density)
    direction = 0.5 * (direction + direction.mT)
    pair_mask = (
        case.inputs.orbital_mask[..., :, None]
        & case.inputs.orbital_mask[..., None, :]
    )
    density = torch.where(pair_mask, density, 0.0)
    direction = torch.where(pair_mask, direction, 0.0)

    def energy(value: torch.Tensor) -> torch.Tensor:
        return electronic_energy_from_density(value, case.inputs).sum()

    derivative = torch.func.jvp(energy, (density,), (direction,))[1]
    charges, _ = density_to_charges_and_core_energy(density, case.inputs)
    fock = problem.rebuild_hamiltonian(charges)
    contraction = torch.sum(fock * direction)
    torch.testing.assert_close(
        derivative, contraction, rtol=2.0e-12, atol=2.0e-12
    )

    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            energy(density + step * direction)
            - energy(density - step * direction)
        ) / (2.0 * step)
        torch.testing.assert_close(
            finite_difference, contraction, rtol=2.0e-7, atol=2.0e-8
        )


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_rebuilt_fock_is_nonlinear_in_density_interpolation(
    method: int,
) -> None:
    """Nonlinear xTB interactions forbid interpolating stored Focks."""
    case = _prepare_case(method, sample_name="H2O", dtype=torch.float64)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    torch.manual_seed(402)
    perturbation = torch.randn_like(initial)
    perturbation = 0.5 * (perturbation + perturbation.mT)
    perturbation = 0.025 * perturbation
    pair_mask = (
        case.inputs.orbital_mask[..., :, None]
        & case.inputs.orbital_mask[..., None, :]
    )
    perturbation = torch.where(pair_mask, perturbation, 0.0)
    density_1, fock_1 = _density_and_fock(problem, initial)
    density_2, fock_2 = _density_and_fock(problem, initial + perturbation)
    coefficient = 0.37
    density = coefficient * density_1 + (1.0 - coefficient) * density_2
    exact_fock = _fock_from_density(problem, density)
    interpolated_fock = coefficient * fock_1 + (1.0 - coefficient) * fock_2
    difference = torch.linalg.vector_norm(exact_fock - interpolated_fock)
    assert float(difference) > 1.0e-8
    if method is GFN2_XTB:
        reduced_entries = tuple(
            entry
            for entry in case.inputs.interactions.entries
            if entry.kind not in ("es3", "d4sc")
        )
        reduced_inputs = replace(
            case.inputs,
            interactions=FunctionalInteractionSet(reduced_entries),
        )
        reduced_problem = replace(problem, inputs=reduced_inputs)
        reduced_focks = tuple(
            _fock_from_density(reduced_problem, density)
            for density in (density_1, density_2)
        )
        reduced_exact = _fock_from_density(reduced_problem, density)
        reduced_interpolated = (
            coefficient * reduced_focks[0]
            + (1.0 - coefficient) * reduced_focks[1]
        )
        reduced_difference = torch.linalg.vector_norm(
            reduced_exact - reduced_interpolated
        )
        assert float(reduced_difference) < 1.0e-10


def test_quadratic_density_energy_interpolation_is_exact() -> None:
    """The exact pairwise correction reproduces a quadratic toy energy."""
    torch.manual_seed(403)
    dtype = torch.float64
    densities = torch.randn((3, 4, 4), dtype=dtype)
    densities = 0.5 * (densities + densities.mT)
    matrix = torch.randn((16, 16), dtype=dtype)
    hessian = 0.5 * (matrix + matrix.mT)
    linear = torch.randn((16,), dtype=dtype)

    def energy(density: torch.Tensor) -> torch.Tensor:
        vector = density.reshape(-1)
        return 0.5 * vector @ hessian @ vector + linear @ vector

    def gradient(density: torch.Tensor) -> torch.Tensor:
        vector = density.reshape(-1)
        return (hessian @ vector + linear).reshape_as(density)

    energies = torch.stack([energy(value) for value in densities])
    focks = torch.stack([gradient(value) for value in densities])
    pair = torch.einsum(
        "ijab,ijab->ij",
        densities[:, None] - densities[None, :],
        focks[:, None] - focks[None, :],
    )
    coefficients = torch.tensor([0.17, 0.29, 0.54], dtype=dtype)
    interpolated = torch.einsum("i,ijk->jk", coefficients, densities)
    quadratic_reference = coefficients @ energies - 0.25 * (
        coefficients @ pair @ coefficients
    )
    torch.testing.assert_close(
        quadratic_reference, energy(interpolated), rtol=1.0e-13, atol=1.0e-13
    )

    # PySCF's historical EDIIS heuristic uses coefficient 1 for this
    # pairwise term. Under dxtb's tested dE = <F, dP> convention, that
    # heuristic is not the exact energy of even this quadratic toy model.
    pyscf_ediis_reference = coefficients @ energies - (
        coefficients @ pair @ coefficients
    )
    assert not torch.isclose(
        pyscf_ediis_reference,
        energy(interpolated),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_adiis_reference_coefficients_match_pyscf_arh_formula() -> None:
    """The ADIIS prototype uses the published augmented RH coefficients."""
    torch.manual_seed(406)
    densities = torch.randn((4, 5, 5), dtype=torch.float64)
    focks = torch.randn_like(densities)
    densities = 0.5 * (densities + densities.mT)
    focks = 0.5 * (focks + focks.mT)
    overlaps = torch.einsum("iab,jab->ij", densities, focks)
    d_fn = overlaps[:, 0]
    dn_f = overlaps[0]
    dn_fn = overlaps[0, 0]
    reference_linear = 2.0 * (d_fn - dn_fn)
    reference_quadratic = overlaps - d_fn[:, None] - dn_f[None, :] + dn_fn
    actual_linear, actual_quadratic = adiis_model_matrices(
        tuple(densities), tuple(focks)
    )
    torch.testing.assert_close(actual_linear, reference_linear)
    torch.testing.assert_close(actual_quadratic, reference_quadratic)


def test_exact_ediis_coefficient_gradient_matches_autograd_and_fd() -> None:
    """The coefficient gradient is the contraction with the exact Fock."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O", dtype=torch.float64)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    torch.manual_seed(404)
    states = [problem.evaluate(initial + 0.01 * torch.randn_like(initial))]
    states.append(problem.evaluate(initial + 0.02 * torch.randn_like(initial)))
    states.append(problem.evaluate(initial + 0.03 * torch.randn_like(initial)))
    densities = tuple(state.map_result.spectral.density for state in states)
    coefficients = torch.tensor([0.21, 0.33, 0.46], dtype=torch.float64)

    def objective(values: torch.Tensor) -> torch.Tensor:
        density = torch.einsum("i,ijk->jk", values, torch.stack(densities))
        return electronic_energy_from_density(density, case.inputs).sum()

    value, analytic = exact_energy_value_gradient(
        coefficients, densities, problem
    )
    automatic = torch.func.grad(objective)(coefficients)
    torch.testing.assert_close(analytic, automatic, rtol=2.0e-12, atol=2.0e-12)
    step = 1.0e-4
    finite = torch.stack(
        [
            (
                objective(coefficients + step * torch.eye(3)[i])
                - objective(coefficients - step * torch.eye(3)[i])
            )
            / (2.0 * step)
            for i in range(3)
        ]
    )
    torch.testing.assert_close(analytic, finite, rtol=1.0e-7, atol=2.0e-7)
    torch.testing.assert_close(value, objective(coefficients))


def test_exact_ediis_simplex_optimizer_matches_dense_two_state_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference optimizer is feasible and no worse than a dense grid."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O", dtype=torch.float64)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    torch.manual_seed(405)
    states = (
        problem.evaluate(initial + 0.01 * torch.randn_like(initial)),
        problem.evaluate(initial + 0.025 * torch.randn_like(initial)),
    )
    densities = tuple(state.map_result.spectral.density for state in states)
    energies = tuple(
        electronic_energy_from_density(density, case.inputs).sum()
        for density in densities
    )
    focks = tuple(_fock_from_density(problem, density) for density in densities)

    def unexpected_map(self: SCFProblem, value: torch.Tensor) -> object:
        del self, value
        raise AssertionError(
            "Coefficient optimization must not run an SCF map."
        )

    monkeypatch.setattr(SCFProblem, "map", unexpected_map)
    coefficients, _ = _minimize_coefficients(
        "exact", densities, energies, focks, problem
    )
    assert torch.all(coefficients >= 0.0)
    torch.testing.assert_close(
        coefficients.sum(), coefficients.new_ones(()), rtol=0.0, atol=1.0e-14
    )

    def energy(value: torch.Tensor) -> torch.Tensor:
        density = torch.einsum("i,ijk->jk", value, torch.stack(densities))
        return electronic_energy_from_density(density, case.inputs).sum()

    selected = energy(coefficients)
    best_vertex = torch.stack(energies).min()
    assert selected <= best_vertex + 1.0e-10
    overlap = case.inputs.integrals.overlap
    counts = torch.stack(
        [torch.sum(density * overlap.mT) for density in densities]
    )
    interpolated = torch.einsum(
        "i,ijk->jk", coefficients, torch.stack(densities)
    )
    torch.testing.assert_close(
        torch.sum(interpolated * overlap.mT),
        coefficients @ counts,
        rtol=0.0,
        atol=2.0e-12,
    )
    grid = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    grid_minimum = torch.stack(
        [energy(torch.stack((value, 1.0 - value))) for value in grid]
    ).min()
    assert selected <= grid_minimum + 1.0e-8


def test_energy_diis_prototype_rejects_active_smearing() -> None:
    """Energy-only density interpolation never enters active thermal SCF."""
    case = _prepare_case(
        GFN2_XTB,
        sample_name="H2O",
        dtype=torch.float64,
        scf_options={"fermi_etemp": 300.0},
    )
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    method = ResearchDIISMethod()
    assert method.supports_fixed_trip is False
    with pytest.raises(NotImplementedError, match="active smearing"):
        method.propose(problem, None, None, None)


def test_energy_diis_prototype_is_rejected_before_fixed_trip_map() -> None:
    """Eager-only energy DIIS cannot enter the fixed-trip physical map."""
    case = _prepare_case(GFN2_XTB, sample_name="H2O", dtype=torch.float64)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    method = ResearchDIISMethod()
    strategy = SCFStrategy((SCFStage(method, SCFRepresentation.FOCK),))

    def unexpected_map(value: torch.Tensor) -> object:
        del value
        raise AssertionError(
            "Fixed-trip rejection must precede map evaluation."
        )

    with pytest.raises(NotImplementedError, match="eager execution only"):
        _solve_map(
            case.x,
            SCFSolverSettings(5, False, 0.5, 0.0, 0.0),
            unexpected_map,
            problem=problem,
            strategy=strategy,
        )


@pytest.mark.parametrize("charge", [0.0, 0.3], ids=["integer", "fractional"])
def test_subcutoff_fixed_occupation_entropy_is_constant_in_density_history(
    charge: float,
) -> None:
    """Fixed-occupation low-temperature entropy does not alter EDIIS ranking."""
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2O",
        charge=charge,
        dtype=torch.float64,
        scf_options={
            "fermi_etemp": 0.05,
            "mixer": labels.MIXER_LINEAR,
        },
    )
    settings = case.inputs.fermi_settings
    assert settings is not None and not settings.smear
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, case.inputs.charge_layout), case.inputs
    )
    first = problem.evaluate(initial).map_result.spectral
    perturbation = torch.randn_like(initial)
    perturbation = 0.5 * (perturbation + perturbation.mT)
    second = problem.evaluate(initial + 0.01 * perturbation).map_result.spectral
    first_energy = electronic_energy_from_density(
        first.density, case.inputs
    ).sum()
    second_energy = electronic_energy_from_density(
        second.density, case.inputs
    ).sum()
    first_free = free_energy_from_spectral(first, case.inputs).sum()
    second_free = free_energy_from_spectral(second, case.inputs).sum()
    torch.testing.assert_close(first_free, second_free, rtol=0.0, atol=2.0e-13)
    torch.testing.assert_close(
        first_energy - second_energy,
        (first_energy + first_free) - (second_energy + second_free),
        rtol=0.0,
        atol=2.0e-13,
    )


@pytest.mark.parametrize(
    ("method", "sample", "spin"),
    [
        (GFN2_XTB, "H2O", 0.0),
        (GFN2_XTB, "NO2", 1.0),
    ],
    ids=["h2o", "no2-open-shell"],
)
def test_exact_ediis_hybrid_reaches_first_order_reference_root(
    method: int, sample: str, spin: float
) -> None:
    """Exact-energy startup agrees with independently run physical roots."""
    case = _prepare_case(
        method,
        sample_name=sample,
        spin=spin,
        dtype=torch.float64,
    )
    inputs = case.inputs
    problem = SCFProblem(inputs, SCFRepresentation.FOCK)
    initial = fock_initial_guess(
        unpack_charges(case.x, inputs.charge_layout), inputs
    )
    settings = SCFSolverSettings(160, False, 0.5, 1.0e-7, 1.0e-7)

    def solve(strategy: SCFStrategy):
        return _solve_map(
            initial,
            settings,
            problem.map,
            problem=problem,
            strategy=strategy,
            eager=True,
        )

    switch = SCFSwitchPolicy(
        min_stage_iterations=3,
        fixed_point_l2=1.0e-2,
        max_stage_iterations=30,
    )
    candidate = solve(
        SCFStrategy(
            (
                SCFStage(
                    ResearchDIISMethod(mode="exact", capacity=6),
                    SCFRepresentation.FOCK,
                    switch,
                    initial_damp=1.0,
                ),
                SCFStage(
                    CDIISMethod(),
                    SCFRepresentation.FOCK,
                    initial_damp=1.0,
                ),
            )
        )
    )
    broyden = solve(
        SCFStrategy((SCFStage(BroydenMethod(), SCFRepresentation.FOCK),))
    )
    cdiis = solve(
        SCFStrategy((SCFStage(CDIISMethod(), SCFRepresentation.FOCK),))
    )

    assert candidate.converged and broyden.converged and cdiis.converged
    actual = candidate.map_result.spectral
    for reference in (broyden.map_result.spectral, cdiis.map_result.spectral):
        torch.testing.assert_close(
            actual.density, reference.density, rtol=2.0e-7, atol=2.0e-7
        )
        for name in ("mono", "dipole", "quad"):
            torch.testing.assert_close(
                getattr(actual.charges, name),
                getattr(reference.charges, name),
                rtol=2.0e-7,
                atol=2.0e-7,
            )
        actual_fock = _fock_from_density(problem, actual.density)
        reference_fock = _fock_from_density(problem, reference.density)
        torch.testing.assert_close(
            actual_fock, reference_fock, rtol=2.0e-7, atol=2.0e-7
        )


def _density_and_fock(
    problem: SCFProblem, fock_input: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state = problem.evaluate(fock_input)
    return state.map_result.spectral.density, state.map_result.mapped_x


def _fock_from_density(
    problem: SCFProblem, density: torch.Tensor
) -> torch.Tensor:
    charges, _ = density_to_charges_and_core_energy(density, problem.inputs)
    return problem.rebuild_hamiltonian(charges)
