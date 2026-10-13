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
"""Diagnosis-only finite-difference probes for E3.0 (T0.2 NO2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from dxtb import Calculator, GFN1_XTB, GFN2_XTB
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.classicals.dispersion.d3 import DispersionD3
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.list import InteractionList
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional.data import (
    SCFCharges,
    SCFPotential,
    pack_potential,
    potential_layout,
    setup_scf_inputs,
    unpack_potential,
)
from dxtb._src.scf.functional.anderson import (
    _solve_map_anderson,
    setup_anderson_solver_settings,
)
from dxtb._src.scf.functional.eager import resolve_full_scf_route
from dxtb._src.scf.functional.map import (
    _transitional_diagonalize,
    evaluate_hamiltonian,
    potential_from_charges,
    potential_initial_guess,
    potential_map,
    potential_to_hamiltonian,
)
from dxtb._src.scf.functional.solver import (
    _project_potential_padding,
    finalize_scf_potential,
    free_energy_from_spectral,
    interaction_energy_from_charges,
)
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.scf.utils import get_density
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction.filling import get_fermi_occupation_core

from .compute import DRIVERS, _calculator
from .compute import SCF_OPTIONS
from .molecules import SYSTEMS
from .refdata import REFERENCE_DIR, load_reference
from .tolerances import TBLITE

pytestmark = [pytest.mark.baseline, pytest.mark.slow]

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}
_STEPS = (5.0e-3, 2.5e-3, 1.25e-3)
_DIAGNOSTIC_STEPS = (
    1.0e-2,
    5.0e-3,
    2.0e-3,
    1.0e-3,
    5.0e-4,
    2.0e-4,
    1.0e-4,
    5.0e-5,
    2.0e-5,
    1.0e-5,
)
_HYPER_INDEX = (1, 1, 1)
_POL_INDEX = (1, 1, 0, 1)


def _energy(calculator, positions: Tensor, field: Tensor) -> float:
    """Evaluate exact T0.2 total energy at one field/geometry point."""
    system = SYSTEMS["NO2"]
    value = calculator.energy(
        positions.clone(),
        system.charge,
        system.spin,
        field=field,
    )
    return float(value.sum().detach())


def _hyper_fd(calculator, step: float) -> float:
    """Third central energy difference along field component y."""
    system = SYSTEMS["NO2"]
    direction = torch.zeros(3, **_DD)
    direction[1] = 1.0
    positions = system.positions
    e = lambda n: _energy(calculator, positions, n * step * direction)
    # The dipole is -dE/dF, so beta is -d^3E/dF^3.
    return -(e(2.0) - 2.0 * e(1.0) + 2.0 * e(-1.0) - e(-2.0)) / (2.0 * step**3)


def _pol_fd(calculator, step: float) -> float:
    """Mixed third central energy difference at the NO2 diagnostic component."""
    system = SYSTEMS["NO2"]
    direction = torch.zeros(3, **_DD)
    direction[1] = 1.0
    field0 = torch.zeros(3, **_DD)
    positions_plus = system.positions.clone()
    positions_minus = system.positions.clone()
    positions_plus[0, 1] += step
    positions_minus[0, 1] -= step

    def field_second(positions: Tensor) -> float:
        plus = _energy(calculator, positions, step * direction)
        zero = _energy(calculator, positions, field0)
        minus = _energy(calculator, positions, -step * direction)
        return (plus - 2.0 * zero + minus) / step**2

    # alpha = -d^2E/dF^2.
    return -(field_second(positions_plus) - field_second(positions_minus)) / (
        2.0 * step
    )


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize(
    ("quantity", "index"),
    (
        pytest.param(
            "hyperpolarizability",
            _HYPER_INDEX,
            id="hyperpolarizability",
        ),
        pytest.param("pol_deriv", _POL_INDEX, id="pol-deriv"),
    ),
)
def test_no2_gfn2_third_order_fd_diagnostic(
    driver: str,
    quantity: str,
    index: tuple[int, ...],
) -> None:
    """Record nested, stored, dxtb-FD, and tblite endpoint comparisons."""
    system = SYSTEMS["NO2"]
    calculator = _calculator(system, "gfn2", driver)
    reference, _, _ = load_reference(
        REFERENCE_DIR / "tblite" / "gfn2" / "NO2.npz"
    )
    stored, _, _ = load_reference(REFERENCE_DIR / f"gfn2-{driver}" / "NO2.npz")
    base_result = calculator.singlepoint(
        system.positions,
        system.charge,
        system.spin,
        field=torch.zeros(3, **_DD),
    )
    field = torch.zeros(3, **_DD, requires_grad=True)
    if quantity == "hyperpolarizability":
        nested = calculator.hyperpolarizability(
            system.positions, system.charge, system.spin, field=field
        )[index]
    else:
        nested = calculator.pol_deriv(
            system.positions.clone().requires_grad_(),
            system.charge,
            system.spin,
            field=field,
        )[index]
    system_field = torch.zeros(3, **_DD)
    field_axis = torch.zeros(3, **_DD)
    field_axis[1] = 1.0

    def evaluate(positions: Tensor, field_value: Tensor) -> tuple[float, int]:
        result = calculator.singlepoint(
            positions, system.charge, system.spin, field=field_value
        )
        return float(result.energy.sum().detach()), int(result.iterations)

    sweep_records = []
    samples = []
    for step in _DIAGNOSTIC_STEPS:
        if quantity == "hyperpolarizability":
            points = [
                evaluate(
                    system.positions, system_field + offset * step * field_axis
                )
                for offset in (-2.0, -1.0, 1.0, 2.0)
            ]
            fd = -(
                points[3][0]
                - 2.0 * points[2][0]
                + 2.0 * points[1][0]
                - points[0][0]
            ) / (2.0 * step**3)
            iterations = [item[1] for item in points]
        else:
            points = []
            for atom_offset in (-1.0, 1.0):
                displaced = system.positions.clone()
                displaced[0, 1] += atom_offset * step
                points.append(
                    [
                        evaluate(
                            displaced,
                            system_field + field_offset * step * field_axis,
                        )
                        for field_offset in (1.0, 0.0, -1.0)
                    ]
                )
            alpha_minus = (
                -(points[0][0][0] - 2.0 * points[0][1][0] + points[0][2][0])
                / step**2
            )
            alpha_plus = (
                -(points[1][0][0] - 2.0 * points[1][1][0] + points[1][2][0])
                / step**2
            )
            fd = (alpha_plus - alpha_minus) / (2.0 * step)
            iterations = [item[1] for row in points for item in row]
        samples.append(float(fd))
        sweep_records.append(
            {"step": step, "fd": float(fd), "iterations": iterations}
        )
    target = float(np.asarray(reference[quantity])[index])
    dxtb_target = float(np.asarray(stored[quantity])[index])
    tolerance = TBLITE[quantity]
    stable_samples = samples[2:6]
    assert all(np.isfinite(sample) for sample in samples)
    assert abs(float(np.median(stable_samples)) - target) <= (
        tolerance.atol + tolerance.rtol * abs(target)
    )
    assert abs(float(nested.detach()) - target) > (
        tolerance.atol + tolerance.rtol * abs(target)
    )
    evidence = {
        "quantity": quantity,
        "driver": driver,
        "composition": (
            "-d^3 E/dF_y^3 through Calculator.hyperpolarizability"
            if quantity == "hyperpolarizability"
            else "-d/dR_(atom0,y) d^2E/dF_y^2 through Calculator.pol_deriv"
        ),
        "nested_autograd": float(nested.detach()),
        "base_iterations": int(base_result.iterations),
        "base_converged": bool(base_result.converged),
        "stored_dxtb": dxtb_target,
        "step_sweep": sweep_records,
        "stable_fd_median": float(np.median(stable_samples)),
        "tblite": target,
        "nested_vs_stored_abs": abs(float(nested.detach()) - dxtb_target),
        "nested_vs_stored_rel": abs(float(nested.detach()) - dxtb_target)
        / max(abs(dxtb_target), np.finfo(float).tiny),
        "nested_vs_fd_abs": abs(
            float(nested.detach()) - float(np.median(stable_samples))
        ),
        "nested_vs_fd_rel": abs(
            float(nested.detach()) - float(np.median(stable_samples))
        )
        / max(abs(float(np.median(stable_samples))), np.finfo(float).tiny),
        "fd_vs_tblite_abs": abs(float(np.median(stable_samples)) - target),
        "fd_vs_tblite_rel": abs(float(np.median(stable_samples)) - target)
        / max(abs(target), np.finfo(float).tiny),
    }
    print(evidence)


def test_no2_gfn1_without_dispersion_tracks_third_order_fd() -> None:
    """GFN1 term-side comparison excludes DFT-D3 and follows direct FDs."""
    system = SYSTEMS["NO2"]
    options = dict(
        SCF_OPTIONS,
        int_driver="pytorch",
        batch_mode=system.batch_mode,
        int_level=3,
        exclude=("disp",),
    )
    calculator = Calculator(
        system.numbers,
        GFN1_XTB,
        opts=options,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert not any(
        type(component) is DispersionD3
        for component in calculator.system.classicals.components
    )
    route = resolve_full_scf_route(
        calculator.opts.scf,
        calculator.interactions,
        calculator.ihelp,
        calculator.numbers,
        calculator.integrals,
    )
    assert route.kind == "functional"

    positions = system.positions.clone()
    field = torch.zeros(3, **_DD, requires_grad=True)
    nested_hyper = calculator.hyperpolarizability(
        positions, system.charge, system.spin, field=field
    )[_HYPER_INDEX]
    nested_pol = calculator.pol_deriv(
        positions.clone().requires_grad_(),
        system.charge,
        system.spin,
        field=field,
    )[_POL_INDEX]
    hyper_samples = [_hyper_fd(calculator, step) for step in _STEPS]
    pol_samples = [_pol_fd(calculator, step) for step in _STEPS]

    assert max(hyper_samples) - min(hyper_samples) < 2.0e-3
    assert max(pol_samples) - min(pol_samples) < 5.0e-4
    assert abs(float(nested_hyper.detach()) - hyper_samples[-1]) < 1.0e-2
    assert abs(float(nested_pol.detach()) - pol_samples[-1]) < 2.0e-4


def _gfn1_no2_system():
    """Set up the GFN1 interaction terms without invoking DFT-D3."""
    system = SYSTEMS["NO2"]
    options = dict(
        SCF_OPTIONS,
        int_driver="pytorch",
        batch_mode=system.batch_mode,
        int_level=3,
        exclude=("disp",),
    )
    calculator = Calculator(
        system.numbers,
        GFN1_XTB,
        opts=options,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert not any(
        type(component) is DispersionD3
        for component in calculator.system.classicals.components
    )
    return calculator.system


def _no2_functional_input(driver: str = "pytorch"):
    """Build raw solver inputs and retain the selected pre-finalization state."""
    system = SYSTEMS["NO2"]
    calculator = _calculator(system, "gfn2", driver)
    model_system = calculator.system
    route = resolve_full_scf_route(
        calculator.opts.scf,
        calculator.interactions,
        calculator.ihelp,
        calculator.numbers,
        calculator.integrals,
    )
    assert route.kind == "functional"
    positions = system.positions.clone()
    charge = torch.atleast_1d(system.charge)
    spin = torch.atleast_1d(system.spin)
    field = torch.zeros(3, **_DD)

    matrices, refocc, _ = build_integral_matrices(
        model_system.integral_setup,
        model_system.h0_setup,
        positions,
        charge,
    )
    n0, occupation, nel = get_refocc(refocc, charge, spin, model_system.ihelp)
    interactions = InteractionList(
        *model_system.interactions.components,
        ElectricField(field),
        device=positions.device,
        dtype=positions.dtype,
    )
    interaction_data = _interaction_data(model_system, positions, interactions)
    route = resolve_full_scf_route(
        calculator.opts.scf,
        interactions,
        model_system.ihelp,
        model_system.numbers,
        matrices,
    )
    assert route.kind == "functional"
    effective_config = replace(calculator.opts.scf, mixer=route.effective_mixer)
    inputs = setup_scf_inputs(
        model_system.numbers,
        matrices,
        n0,
        occupation,
        nel,
        interactions,
        interaction_data,
        model_system.ihelp,
        effective_config,
    )
    guess = get_guess(
        model_system.numbers,
        positions,
        charge,
        model_system.ihelp,
        effective_config.guess,
    )
    guess_charges = inputs.charge_layout
    initial_charges = SCFCharges(
        mono=guess,
        dipole=(
            None
            if guess_charges.dipole_shape is None
            else guess.new_zeros(guess_charges.dipole_shape)
        ),
        quad=(
            None
            if guess_charges.quadrupole_shape is None
            else guess.new_zeros(guess_charges.quadrupole_shape)
        ),
    )
    initial_state, layout = potential_initial_guess(initial_charges, inputs)
    settings = setup_anderson_solver_settings(effective_config, positions.dtype)
    outcome = _solve_map_anderson(
        initial_state,
        settings,
        lambda value: potential_map(value, inputs, layout),
        state_projection=lambda value: _project_potential_padding(
            value, layout
        ),
        eager=True,
    )
    finalized = finalize_scf_potential(
        outcome.map_result,
        inputs,
        layout,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
    )
    # The map's selected state is the solver state. Public Result.potential is
    # deliberately not used here: it is evaluated after this state is mapped.
    state = outcome.map_result.mapped_x.detach()
    return (
        model_system,
        positions,
        field,
        matrices,
        inputs,
        layout,
        state,
        outcome,
        finalized,
    )


def _functional_solve_at(
    calculator: Calculator,
    positions: Tensor,
    field: Tensor,
    *,
    eager: bool,
    maxiter: int | None = None,
):
    """Return trajectory outcome and finalization from one prepared solve."""
    model_system = calculator.system
    charge = torch.atleast_1d(SYSTEMS["NO2"].charge)
    spin = torch.atleast_1d(SYSTEMS["NO2"].spin)
    matrices, refocc, _ = build_integral_matrices(
        model_system.integral_setup,
        model_system.h0_setup,
        positions,
        charge,
    )
    n0, occupation, nel = get_refocc(refocc, charge, spin, model_system.ihelp)
    interactions = InteractionList(
        *model_system.interactions.components,
        ElectricField(field),
        device=positions.device,
        dtype=positions.dtype,
    )
    data = _interaction_data(model_system, positions, interactions)
    route = resolve_full_scf_route(
        calculator.opts.scf,
        interactions,
        model_system.ihelp,
        model_system.numbers,
        matrices,
    )
    assert route.kind == "functional"
    config = replace(calculator.opts.scf, mixer=route.effective_mixer)
    if maxiter is not None:
        config = replace(config, maxiter=maxiter)
    inputs = setup_scf_inputs(
        model_system.numbers,
        matrices,
        n0,
        occupation,
        nel,
        interactions,
        data,
        model_system.ihelp,
        config,
    )
    guess = get_guess(
        model_system.numbers,
        positions,
        charge,
        model_system.ihelp,
        config.guess,
    )
    initial = SCFCharges(
        mono=guess,
        dipole=(
            None
            if inputs.charge_layout.dipole_shape is None
            else guess.new_zeros(inputs.charge_layout.dipole_shape)
        ),
        quad=(
            None
            if inputs.charge_layout.quadrupole_shape is None
            else guess.new_zeros(inputs.charge_layout.quadrupole_shape)
        ),
    )
    packed, layout = potential_initial_guess(initial, inputs)
    settings = setup_anderson_solver_settings(config, positions.dtype)
    outcome = _solve_map_anderson(
        packed,
        settings,
        lambda value: potential_map(value, inputs, layout),
        state_projection=lambda value: _project_potential_padding(
            value, layout
        ),
        eager=eager,
    )
    finalized = finalize_scf_potential(
        outcome.map_result,
        inputs,
        layout,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
    )
    return outcome, inputs, layout, finalized


def _selected_map_energy(outcome, inputs) -> Tensor:
    """Gauge-invariant energy evaluated from the selected raw-map spectrum."""
    return _spectral_energy(outcome.map_result.spectral, inputs)


def _spectral_energy(spectral, inputs) -> Tensor:
    """Evaluate the physical energy of one gauge-invariant spectrum."""
    h0 = inputs.ihelp.reduce_orbital_to_atom(spectral.orbital_energy)
    return (
        h0
        + interaction_energy_from_charges(spectral.charges, inputs)
        + free_energy_from_spectral(spectral, inputs)
    ).sum()


def _third_derivative(function) -> Tensor:
    """Differentiate a scalar probe three times without transform wrappers."""
    value = torch.zeros((), **_DD, requires_grad=True)
    output = function(value)
    for _ in range(3):
        (output,) = torch.autograd.grad(
            output, value, create_graph=True, retain_graph=True
        )
    return output


@pytest.mark.parametrize("driver", DRIVERS)
def test_no2_gfn2_frozen_spectral_density_third_order_matches_fd(
    driver: str,
) -> None:
    """A fixed-input H/S spectral evaluation has a stable third derivative."""
    (
        model_system,
        positions,
        field0,
        matrices,
        inputs,
        layout,
        state,
        outcome,
        _,
    ) = _no2_functional_input(driver)
    selected = outcome.map_result.spectral
    axis = torch.zeros_like(field0)
    axis[1] = 1.0
    probe = torch.linspace(
        -0.7, 1.1, selected.density.numel(), **_DD
    ).reshape_as(selected.density)
    fixed_occupation = selected.occupation.sum(-2).detach()

    def density_probe(parameter: Tensor) -> Tensor:
        field = field0 + parameter * axis
        interactions = InteractionList(
            *model_system.interactions.components,
            ElectricField(field),
            device=positions.device,
            dtype=positions.dtype,
        )
        data = _interaction_data(model_system, positions, interactions)
        call_inputs = replace(
            inputs, interactions=interactions, interaction_data=data
        )
        driving = potential_from_charges(
            unpack_potential(state, layout), call_inputs
        )
        hamiltonian = potential_to_hamiltonian(
            matrices.hcore,
            matrices.overlap,
            matrices.dipole,
            matrices.quadrupole,
            model_system.ihelp,
            driving,
        )
        _, coefficients = _transitional_diagonalize(hamiltonian, call_inputs)
        density = get_density(coefficients, fixed_occupation)
        return (density * probe).sum()

    automatic = float(_third_derivative(density_probe).detach())

    def finite_difference(step: float) -> float:
        def evaluate(point: float) -> float:
            return float(density_probe(torch.tensor(point, **_DD)).detach())

        return (
            evaluate(2.0 * step)
            - 2.0 * evaluate(step)
            + 2.0 * evaluate(-step)
            - evaluate(-2.0 * step)
        ) / (2.0 * step**3)

    samples = [finite_difference(step) for step in _STEPS]
    assert abs(samples[-2] - samples[-1]) < 2.0
    assert abs(automatic - samples[-1]) < 1.0
    print(
        {
            "driver": driver,
            "density_nested_third": automatic,
            "density_fd": samples,
        }
    )


def test_no2_spectral_density_at_exact_and_near_degeneracy() -> None:
    """Probe gauge-invariant density/energy through exact and near gaps."""
    _, _, _, _, inputs, _, _, _, _ = _no2_functional_input("pytorch")
    inverse = inputs.inverse_cholesky
    cholesky = torch.linalg.inv(inverse)
    n_orbitals = inputs.integrals.hcore.shape[-1]
    energies = torch.linspace(-2.0, 2.0, n_orbitals, **_DD)
    channel_electrons = inputs.nel.reshape(-1)
    frontier = max(0, min(n_orbitals - 2, int(channel_electrons.max()) - 1))
    energies[frontier] = -0.1
    density_probe = torch.linspace(-0.4, 0.8, n_orbitals**2, **_DD).reshape(
        n_orbitals, n_orbitals
    )

    records = {}
    for label, gap in (("exact", 0.0), ("near", 1.0e-6)):
        center = energies.clone()
        center[frontier + 1] = center[frontier] + gap
        split_matrix = torch.zeros((n_orbitals, n_orbitals), **_DD)
        split_matrix[frontier, frontier + 1] = 0.5
        split_matrix[frontier + 1, frontier] = 0.5

        def gauge_invariant_probe(parameter: Tensor) -> Tensor:
            hamiltonian = (
                cholesky
                @ (torch.diag(center) + parameter * split_matrix)
                @ cholesky.mT
            )
            spectral = evaluate_hamiltonian(hamiltonian, inputs)
            energy = (spectral.emo * spectral.occupation.sum(-2)).sum()
            density = (spectral.density * density_probe).sum()
            return energy + density

        automatic = float(_third_derivative(gauge_invariant_probe).detach())
        samples = []
        for step in _DIAGNOSTIC_STEPS:

            def at(point: float) -> float:
                return float(
                    gauge_invariant_probe(torch.tensor(point, **_DD)).detach()
                )

            fd = (
                at(2.0 * step)
                - 2.0 * at(step)
                + 2.0 * at(-step)
                - at(-2.0 * step)
            ) / (2.0 * step**3)
            samples.append(float(fd))
        assert np.isfinite(automatic)
        assert np.isfinite(samples).all()
        records[label] = {
            "gap": gap,
            "nested_third": automatic,
            "fd_sweep": list(zip(_DIAGNOSTIC_STEPS, samples)),
        }
    print(records)


@pytest.mark.parametrize("driver", DRIVERS)
def test_no2_gfn2_frozen_spectrum_fermi_response_matches_fd(
    driver: str,
) -> None:
    """Probe Fermi response to a fixed-spectrum directional perturbation."""
    _, _, _, _, inputs, _, _, outcome, _ = _no2_functional_input(driver)
    settings = inputs.fermi_settings
    assert settings is not None and settings.smear

    selected = outcome.map_result.spectral
    spectrum = selected.emo.detach()
    direction = torch.linspace(-1.0, 1.0, spectrum.numel(), **_DD).reshape_as(
        spectrum
    )
    probe = torch.linspace(
        -0.7, 1.1, selected.occupation.numel(), **_DD
    ).reshape_as(selected.occupation)

    def fermi_probe(parameter: Tensor) -> Tensor:
        energies = (
            (spectrum + parameter * direction)
            .unsqueeze(-2)
            .expand(*inputs.nel.shape, spectrum.shape[-1])
        )
        occupations = get_fermi_occupation_core(
            inputs.nel,
            energies,
            settings.kt,
            mask=inputs.orbital_mask.unsqueeze(-2).expand_as(energies),
            thr=settings.threshold,
            maxiter=settings.maxiter,
            diff_order=settings.diff_order,
        ).occupation
        return (occupations * probe).sum()

    automatic = float(_third_derivative(fermi_probe).detach())

    def finite_difference(step: float) -> float:
        def evaluate(point: float) -> float:
            return float(fermi_probe(torch.tensor(point, **_DD)).detach())

        return (
            evaluate(2.0 * step)
            - 2.0 * evaluate(step)
            + 2.0 * evaluate(-step)
            - evaluate(-2.0 * step)
        ) / (2.0 * step**3)

    samples = [finite_difference(step) for step in _STEPS]
    assert float(
        (selected.occupation - selected.occupation.round()).abs().max()
    ) < (1.0e-10)
    assert max(samples) - min(samples) < 5.0e-8
    assert abs(automatic - samples[-1]) < 5.0e-8
    print(
        {
            "driver": driver,
            "fractional_occupation_max": float(
                (selected.occupation - selected.occupation.round()).abs().max()
            ),
            "fermi_nested_third": automatic,
            "fermi_fd": samples,
        }
    )


def test_no2_gfn1_gfn2_term_activation_at_same_gfn2_state() -> None:
    """Compare GFN1/GFN2 SCF terms at one fixed GFN2 converged state."""
    model_system, _, _, _, inputs, layout, state, _, _ = _no2_functional_input()
    gfn1_system = _gfn1_no2_system()
    gfn1_calculator = Calculator(
        SYSTEMS["NO2"].numbers,
        GFN1_XTB,
        opts={
            **SCF_OPTIONS,
            "int_driver": "pytorch",
            "batch_mode": SYSTEMS["NO2"].batch_mode,
            "int_level": 3,
            "exclude": ("disp",),
        },
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    _, gfn1_scf_inputs, gfn1_layout, _ = _functional_solve_at(
        gfn1_calculator,
        SYSTEMS["NO2"].positions,
        torch.zeros(3, **_DD),
        eager=True,
    )
    assert torch.equal(model_system.numbers, gfn1_system.numbers)
    assert torch.equal(
        model_system.ihelp.shell_index, gfn1_system.ihelp.shell_index
    )
    assert torch.equal(
        model_system.ihelp.orbital_index, gfn1_system.ihelp.orbital_index
    )
    for name in (
        "shells_to_atom",
        "orbitals_to_shell",
        "orbitals_to_atom",
    ):
        assert torch.equal(
            getattr(model_system.ihelp, name), getattr(gfn1_system.ihelp, name)
        )
    assert inputs.integrals.hcore.shape == gfn1_scf_inputs.integrals.hcore.shape
    assert (
        inputs.integrals.overlap.shape
        == gfn1_scf_inputs.integrals.overlap.shape
    )
    assert (
        inputs.charge_layout.mono_shape
        == gfn1_scf_inputs.charge_layout.mono_shape
    )
    assert (
        inputs.charge_layout.dipole_shape
        == gfn1_scf_inputs.charge_layout.dipole_shape
    )
    assert inputs.charge_layout.quadrupole_shape is not None
    assert gfn1_scf_inputs.charge_layout.quadrupole_shape is None
    assert layout.mono_shape == gfn1_layout.mono_shape
    assert layout.dipole_shape == gfn1_layout.dipole_shape
    gfn1_interactions = InteractionList(
        *gfn1_system.interactions.components,
        ElectricField(torch.zeros(3, **_DD)),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    gfn1_inputs = replace(
        inputs,
        interactions=gfn1_interactions,
        interaction_data=_interaction_data(
            gfn1_system, SYSTEMS["NO2"].positions, gfn1_interactions
        ),
    )

    # Both term sets use the same GFN2 hcore, integrals, and converged SCF
    # potential. GFN1 has no AES2 quadrupole channel, so compare its map on
    # the shared monopole/dipole coordinates of that same GFN2 state.
    assert {type(item) for item in gfn1_interactions.components} == {
        type(model_system.es2_interaction),
        type(model_system.es3_interaction),
        ElectricField,
    }
    selected_potential = unpack_potential(state, layout)
    assert selected_potential.mono is not None
    assert selected_potential.dipole is not None
    gfn1_potential = SCFPotential(
        selected_potential.mono, selected_potential.dipole
    )
    gfn1_layout = potential_layout(gfn1_potential)
    gfn1_state = pack_potential(gfn1_potential, gfn1_layout)
    mono_size = selected_potential.mono.numel()
    dipole_size = selected_potential.dipole.numel()
    common_size = mono_size + dipole_size
    assert torch.equal(state[0, :mono_size], gfn1_state[0, :mono_size])
    assert torch.equal(state[1, :dipole_size], gfn1_state[1, :dipole_size])
    direction_common = torch.linspace(-0.2, 0.2, common_size, **_DD)
    projection = torch.linspace(-0.9, 1.2, common_size, **_DD)
    direction_gfn1 = torch.zeros_like(gfn1_state)
    direction_gfn1[0, :mono_size] = direction_common[:mono_size]
    direction_gfn1[1, :dipole_size] = direction_common[mono_size:]
    direction_gfn2 = torch.zeros_like(state)
    direction_gfn2[0, :mono_size] = direction_common[:mono_size]
    direction_gfn2[1, :dipole_size] = direction_common[mono_size:]
    cases = (
        (
            inputs,
            layout,
            state,
            direction_gfn2,
            (1.0e-2, 7.5e-3, 5.0e-3),
        ),
        (
            gfn1_inputs,
            gfn1_layout,
            gfn1_state,
            direction_gfn1,
            (2.0e-2, 1.5e-2, 1.0e-2),
        ),
    )

    for call_inputs, call_layout, call_state, direction, steps in cases:

        def map_probe(parameter: Tensor) -> Tensor:
            mapped = potential_map(
                call_state + parameter * direction,
                call_inputs,
                call_layout,
            )
            mapped_common = torch.cat(
                (
                    mapped.mapped_x[0, :mono_size],
                    mapped.mapped_x[1, :dipole_size],
                )
            )
            return (mapped_common * projection).sum()

        automatic = float(_third_derivative(map_probe).detach())

        def finite_difference(step: float) -> float:
            def evaluate(point: float) -> float:
                return float(map_probe(torch.tensor(point, **_DD)).detach())

            return (
                evaluate(2.0 * step)
                - 2.0 * evaluate(step)
                + 2.0 * evaluate(-step)
                - evaluate(-2.0 * step)
            ) / (2.0 * step**3)

        samples = [finite_difference(step) for step in steps]
        assert abs(samples[-2] - samples[-1]) < 1.0e-3
        assert abs(automatic - samples[-1]) < 5.0e-3


@pytest.mark.parametrize("driver", DRIVERS)
def test_no2_gfn2_raw_potential_map_third_order_matches_fd(
    driver: str,
) -> None:
    """The one-map derivative is checked at a fixed converged input state."""
    _, _, _, _, inputs, layout, state, _, _ = _no2_functional_input(driver)
    direction = torch.linspace(-0.2, 0.2, state.numel(), **_DD).reshape_as(
        state
    )

    def map_probe(parameter: Tensor) -> Tensor:
        mapped = potential_map(state + parameter * direction, inputs, layout)
        return _spectral_energy(mapped.spectral, inputs).sum()

    automatic = float(_third_derivative(map_probe).detach())

    def finite_difference(step: float) -> float:
        def evaluate(point: float) -> float:
            return float(map_probe(torch.tensor(point, **_DD)).detach())

        return (
            evaluate(2.0 * step)
            - 2.0 * evaluate(step)
            + 2.0 * evaluate(-step)
            - evaluate(-2.0 * step)
        ) / (2.0 * step**3)

    samples = [finite_difference(step) for step in _DIAGNOSTIC_STEPS]
    plateau = samples[2:6]
    assert max(plateau) - min(plateau) < 2.0e-2
    assert abs(automatic - float(np.median(plateau))) < 5.0e-2
    print(
        {
            "driver": driver,
            "selected_solver_state_raw_map_energy_nested_third": automatic,
            "steps": _DIAGNOSTIC_STEPS,
            "raw_map_energy_fd": samples,
        }
    )


def test_no2_gfn2_solver_finalization_and_property_boundaries() -> None:
    """Separate solver, finalization and mixed property-response boundaries."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    positions = SYSTEMS["NO2"].positions
    field_axis = torch.zeros(3, **_DD)
    field_axis[1] = 1.0

    def solve_pair(
        parameter: Tensor,
        displacement: Tensor,
        eager: bool,
        maxiter: int | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        displaced_positions = positions.clone()
        displaced_positions[0, 1] = displaced_positions[0, 1] + displacement
        outcome, inputs, _, finalized = _functional_solve_at(
            calculator,
            displaced_positions,
            parameter * field_axis,
            eager=eager,
            maxiter=maxiter,
        )
        return (
            _selected_map_energy(outcome, inputs),
            (finalized.energy + finalized.fenergy).sum(),
            outcome.iterations,
        )

    evidence: dict[str, object] = {}
    for eager in (True, False):
        control = "eager" if eager else "fixed-trip"
        # The fixed-trip comparison caps the static loop at 40 only after
        # checking every stencil point converges earlier. The exact 300-trip
        # baseline is compared with this selected outcome at the center below.
        maxiter = 40 if not eager else None
        stage_evidence: dict[str, dict[str, object]] = {}
        for stage in ("solver", "finalized"):
            field_index = 0 if stage == "solver" else 1

            def energy_fn(parameter: Tensor, displacement: Tensor) -> Tensor:
                return solve_pair(parameter, displacement, eager, maxiter)[
                    field_index
                ]

            automatic = -float(
                _third_derivative(
                    lambda parameter: energy_fn(
                        parameter, torch.zeros((), **_DD)
                    )
                ).detach()
            )
            scalar = torch.zeros((), **_DD, requires_grad=True)
            value = energy_fn(scalar, torch.zeros((), **_DD))
            first = torch.autograd.grad(value, scalar, create_graph=True)[0]
            second = torch.autograd.grad(first, scalar, create_graph=True)[0]
            reverse_first = float(first.detach())
            reverse_second = float(second.detach())
            jvp = float(
                torch.func.jvp(
                    lambda parameter: energy_fn(
                        parameter, torch.zeros((), **_DD)
                    ),
                    (torch.zeros((), **_DD),),
                    (torch.ones((), **_DD),),
                )[1]
            )
            pol_geometry = torch.zeros((), **_DD, requires_grad=True)

            def field_second(displacement: Tensor) -> Tensor:
                field_value = torch.zeros((), **_DD, requires_grad=True)
                energy = energy_fn(field_value, displacement)
                first_field = torch.autograd.grad(
                    energy, field_value, create_graph=True
                )[0]
                second_field = torch.autograd.grad(
                    first_field, field_value, create_graph=True
                )[0]
                return -second_field

            alpha = field_second(pol_geometry)
            mixed_reverse = float(
                torch.autograd.grad(alpha, pol_geometry)[0].detach()
            )
            mixed_jvp = float(
                torch.func.jvp(
                    field_second,
                    (torch.zeros((), **_DD),),
                    (torch.ones((), **_DD),),
                )[1]
            )
            stage_evidence[stage] = {
                "hyper_autograd_third": automatic,
                "reverse_first": reverse_first,
                "jvp_first": jvp,
                "reverse_second": reverse_second,
                "pol_autograd_mixed_third": mixed_reverse,
                "pol_jvp_mixed_third": mixed_jvp,
            }
        # Finite differences reuse each solver outcome for both the selected
        # raw-map energy and the potential-mode finalized energy.
        for step in _DIAGNOSTIC_STEPS:
            pairs = [
                solve_pair(
                    torch.tensor(offset * step, **_DD),
                    torch.zeros((), **_DD),
                    eager,
                    maxiter,
                )
                for offset in (-2.0, -1.0, 0.0, 1.0, 2.0)
            ]
            for stage, index in (("solver", 0), ("finalized", 1)):
                points = [(pair[index], pair[2]) for pair in pairs]
                fd = -(
                    points[4][0]
                    - 2.0 * points[3][0]
                    + 2.0 * points[1][0]
                    - points[0][0]
                ) / (2.0 * step**3)
                row = {
                    "step": step,
                    "fd": float(fd),
                    "fd_first": float(
                        (points[3][0] - points[1][0]) / (2.0 * step)
                    ),
                    "fd_second": float(
                        (points[3][0] - 2.0 * points[2][0] + points[1][0])
                        / step**2
                    ),
                    "iterations_at_-2h_-h_0_h_2h": [
                        int(item[1]) for item in points
                    ],
                }
                stage_evidence[stage].setdefault("hyper_sweep", []).append(row)
            if not eager:
                assert all(int(pair[2]) < maxiter for pair in pairs)

        for step in _DIAGNOSTIC_STEPS:
            geometry_field_pairs = []
            for geometry_offset in (-1.0, 1.0):
                geometry_field_pairs.append(
                    [
                        solve_pair(
                            torch.tensor(field_offset * step, **_DD),
                            torch.tensor(geometry_offset * step, **_DD),
                            eager,
                            maxiter,
                        )
                        for field_offset in (1.0, 0.0, -1.0)
                    ]
                )
            for stage, index in (("solver", 0), ("finalized", 1)):
                alpha_values = [
                    -(row[0][index] - 2.0 * row[1][index] + row[2][index])
                    / step**2
                    for row in geometry_field_pairs
                ]
                mixed_fd = (alpha_values[1] - alpha_values[0]) / (2.0 * step)
                stage_evidence[stage].setdefault("pol_sweep", []).append(
                    {
                        "step": step,
                        "fd": float(mixed_fd),
                        "iterations_at_Rminus_(Fplus,F0,Fminus)": [
                            int(item[2]) for item in geometry_field_pairs[0]
                        ],
                        "iterations_at_Rplus_(Fplus,F0,Fminus)": [
                            int(item[2]) for item in geometry_field_pairs[1]
                        ],
                    }
                )
            if not eager:
                assert all(
                    int(item[2]) < maxiter
                    for row in geometry_field_pairs
                    for item in row
                )
        evidence.update(
            {
                f"{control}-{stage}": record
                for stage, record in stage_evidence.items()
            }
        )
        for record in stage_evidence.values():
            hyper_sweep = record.get("hyper_sweep", [])
            pol_sweep = record.get("pol_sweep", [])
            hyper_plateau = hyper_sweep[2:6]
            pol_plateau = pol_sweep[2:6]
            hyper_fd = float(np.median([row["fd"] for row in hyper_plateau]))
            pol_fd = float(np.median([row["fd"] for row in pol_plateau]))
            record["hyper_stable_fd"] = hyper_fd
            record["hyper_nested_vs_fd_abs"] = abs(
                record["hyper_autograd_third"] - hyper_fd
            )
            record["hyper_jvp_vs_reverse_abs"] = abs(
                record["jvp_first"] - record["reverse_first"]
            )
            record["hyper_reverse_first_vs_fd_abs"] = abs(
                record["reverse_first"]
                - float(np.median([row["fd_first"] for row in hyper_plateau]))
            )
            record["hyper_reverse_second_vs_fd_abs"] = abs(
                record["reverse_second"]
                - float(np.median([row["fd_second"] for row in hyper_plateau]))
            )
            record["pol_stable_fd"] = pol_fd
            record["pol_nested_vs_fd_abs"] = abs(
                record["pol_autograd_mixed_third"] - pol_fd
            )
            record["pol_jvp_vs_reverse_abs"] = abs(
                record["pol_jvp_mixed_third"]
                - record["pol_autograd_mixed_third"]
            )
        if not eager:
            full_result, _, _, full_finalized = _functional_solve_at(
                calculator,
                positions,
                torch.zeros(3, **_DD),
                eager=False,
                maxiter=None,
            )
            capped_result, _, _, capped_finalized = _functional_solve_at(
                calculator,
                positions,
                torch.zeros(3, **_DD),
                eager=False,
                maxiter=40,
            )
            assert int(full_result.iterations) == int(capped_result.iterations)
            assert torch.equal(full_result.converged, capped_result.converged)
            assert torch.allclose(
                full_result.map_result.mapped_x,
                capped_result.map_result.mapped_x,
                atol=1.0e-12,
                rtol=1.0e-12,
            )
            full_map = full_result.map_result
            capped_map = capped_result.map_result
            for name in (
                "hamiltonian",
                "density",
                "emo",
                "occupation",
                "orbital_energy",
            ):
                assert torch.allclose(
                    getattr(full_map, name),
                    getattr(capped_map, name),
                    atol=1.0e-12,
                    rtol=1.0e-12,
                ), name
            for name in ("mono", "dipole", "quad"):
                full_charge = getattr(full_map.charges, name)
                capped_charge = getattr(capped_map.charges, name)
                if full_charge is None or capped_charge is None:
                    assert full_charge is capped_charge, name
                else:
                    assert torch.allclose(
                        full_charge,
                        capped_charge,
                        atol=1.0e-12,
                        rtol=1.0e-12,
                    ), f"charges.{name}"
            assert full_map.driving_potential is not None
            assert capped_map.driving_potential is not None
            for name in ("mono", "dipole", "quad"):
                full_potential = getattr(full_map.driving_potential, name)
                capped_potential = getattr(capped_map.driving_potential, name)
                if full_potential is None or capped_potential is None:
                    assert full_potential is capped_potential, name
                else:
                    assert torch.allclose(
                        full_potential,
                        capped_potential,
                        atol=1.0e-12,
                        rtol=1.0e-12,
                    ), f"driving_potential.{name}"
            assert torch.allclose(
                full_finalized.energy + full_finalized.fenergy,
                capped_finalized.energy + capped_finalized.fenergy,
                atol=1.0e-12,
                rtol=1.0e-12,
            )

    # Save full machine-readable evidence in pytest output for the diagnosis
    # record; this test intentionally does not assert an assumed failing layer.
    import json

    print(json.dumps(evidence, indent=2))
    assert len(evidence) == 4


def test_no2_gfn2_case_a_and_production_route_agree() -> None:
    """Reproduce explicit-Anderson and configured-Broyden route parity."""
    reference_system = SYSTEMS["NO2"]
    options = {
        **SCF_OPTIONS,
        "int_driver": "pytorch",
        "batch_mode": reference_system.batch_mode,
        "int_level": 3,
    }
    case_a = Calculator(
        reference_system.numbers,
        GFN2_XTB,
        opts=options,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    case_b = Calculator(
        reference_system.numbers,
        GFN2_XTB,
        opts={**options, "mixer": "broyden", "strict": False},
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    route_b = resolve_full_scf_route(
        case_b.opts.scf,
        case_b.interactions,
        case_b.ihelp,
        case_b.numbers,
        case_b.integrals,
    )
    assert route_b.kind == "functional"
    assert route_b.effective_mixer == case_a.opts.scf.mixer
    positions = reference_system.positions
    field = torch.zeros(3, **_DD)
    result_a = case_a.system.singlepoint(
        positions, reference_system.charge, reference_system.spin, field
    )
    result_b = case_b.system.singlepoint(
        positions, reference_system.charge, reference_system.spin, field
    )
    for name in (
        "energy",
        "emo",
        "occupation",
        "density",
        "hamiltonian",
        "iterations",
        "converged",
    ):
        assert torch.allclose(
            getattr(result_a, name),
            getattr(result_b, name),
            atol=1.0e-12,
            rtol=1.0e-12,
        ), name
    for field_name in ("mono", "dipole", "quadrupole"):
        for owner_name in ("charges", "potential"):
            left = getattr(getattr(result_a, owner_name), field_name)
            right = getattr(getattr(result_b, owner_name), field_name)
            if left is None or right is None:
                assert left is right, f"{owner_name}.{field_name}"
            else:
                assert torch.allclose(
                    left, right, atol=1.0e-12, rtol=1.0e-12
                ), f"{owner_name}.{field_name}"

    # Compare the endpoint derivatives without relying on eigenvector gauge.
    field_a = torch.zeros(3, **_DD, requires_grad=True)
    field_b = torch.zeros(3, **_DD, requires_grad=True)
    hyper_a = case_a.hyperpolarizability(
        positions, reference_system.charge, reference_system.spin, field=field_a
    )[_HYPER_INDEX]
    hyper_b = case_b.hyperpolarizability(
        positions, reference_system.charge, reference_system.spin, field=field_b
    )[_HYPER_INDEX]
    pos_a = positions.clone().requires_grad_()
    pos_b = positions.clone().requires_grad_()
    pol_a = case_a.pol_deriv(
        pos_a, reference_system.charge, reference_system.spin, field=field_a
    )[_POL_INDEX]
    pol_b = case_b.pol_deriv(
        pos_b, reference_system.charge, reference_system.spin, field=field_b
    )[_POL_INDEX]
    assert torch.isfinite(hyper_a) and torch.isfinite(hyper_b)
    assert torch.isfinite(pol_a) and torch.isfinite(pol_b)
    hyper_delta = float((hyper_a - hyper_b).detach())
    pol_delta = float((pol_a - pol_b).detach())
    assert abs(hyper_delta) < 5.0e-4 * max(
        abs(float(hyper_a.detach())), abs(float(hyper_b.detach()))
    )
    assert abs(pol_delta) < 5.0e-4 * max(
        abs(float(pol_a.detach())), abs(float(pol_b.detach()))
    )
    print(
        {
            "case_a_iterations": int(result_a.iterations),
            "case_b_iterations": int(result_b.iterations),
            "case_a_energy": float(result_a.energy.sum()),
            "case_b_energy": float(result_b.energy.sum()),
            "case_a_hyperpolarizability": float(hyper_a.detach()),
            "case_b_hyperpolarizability": float(hyper_b.detach()),
            "hyperpolarizability_delta": hyper_delta,
            "case_a_pol_deriv": float(pol_a.detach()),
            "case_b_pol_deriv": float(pol_b.detach()),
            "pol_deriv_delta": pol_delta,
        }
    )
