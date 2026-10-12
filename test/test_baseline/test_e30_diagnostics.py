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

from dxtb import Calculator, GFN1_XTB
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.classicals.dispersion.d3 import DispersionD3
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.list import InteractionList
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional.data import (
    SCFPotential,
    pack_potential,
    potential_layout,
    setup_scf_inputs,
    unpack_potential,
)
from dxtb._src.scf.functional.map import (
    _transitional_diagonalize,
    potential_from_charges,
    potential_map,
    potential_to_hamiltonian,
)
from dxtb._src.scf.functional.eager import resolve_full_scf_route
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.scf.utils import get_density
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction.filling import get_fermi_occupation_core

from .compute import DRIVERS, _calculator
from .molecules import SYSTEMS
from .refdata import REFERENCE_DIR, load_reference
from .compute import SCF_OPTIONS
from .tolerances import TBLITE

pytestmark = [pytest.mark.baseline, pytest.mark.slow]

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}
_STEPS = (5.0e-3, 2.5e-3, 1.25e-3)
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
    ("quantity", "index", "finite_difference", "spread_limit"),
    (
        pytest.param(
            "hyperpolarizability",
            _HYPER_INDEX,
            _hyper_fd,
            2.0e-3,
            id="hyperpolarizability",
        ),
        pytest.param("pol_deriv", _POL_INDEX, _pol_fd, 5.0e-4, id="pol-deriv"),
    ),
)
def test_no2_gfn2_third_order_fd_diagnostic(
    driver: str,
    quantity: str,
    index: tuple[int, ...],
    finite_difference,
    spread_limit: float,
) -> None:
    """Direct energy finite differences reproduce the independent T0.2 values."""
    system = SYSTEMS["NO2"]
    calculator = _calculator(system, "gfn2", driver)
    reference, _, _ = load_reference(
        REFERENCE_DIR / "tblite" / "gfn2" / "NO2.npz"
    )
    samples = [finite_difference(calculator, step) for step in _STEPS]
    target = float(np.asarray(reference[quantity])[index])
    tolerance = TBLITE[quantity]

    assert max(samples) - min(samples) < spread_limit
    assert all(
        abs(value - target) <= tolerance.atol + tolerance.rtol * abs(target)
        for value in samples
    )


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
    """Build fixed NO2 potential-map inputs from the converged T0.2 state."""
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
    result = model_system.singlepoint(positions, charge, spin, field)

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
    inputs = setup_scf_inputs(
        model_system.numbers,
        matrices,
        n0,
        occupation,
        nel,
        interactions,
        interaction_data,
        model_system.ihelp,
        model_system.config.scf,
    )
    potential = SCFPotential(
        result.potential.mono,
        result.potential.dipole,
        result.potential.quadrupole,
    )
    layout = potential_layout(potential)
    state = pack_potential(potential, layout).detach()
    return (
        model_system,
        positions,
        field,
        matrices,
        inputs,
        layout,
        state,
        result,
    )


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
    model_system, positions, field0, matrices, inputs, layout, state, result = (
        _no2_functional_input(driver)
    )
    axis = torch.zeros_like(field0)
    axis[1] = 1.0
    probe = torch.linspace(-0.7, 1.1, result.density.numel(), **_DD).reshape_as(
        result.density
    )
    fixed_occupation = result.occupation.sum(-2).detach()

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


@pytest.mark.parametrize("driver", DRIVERS)
def test_no2_gfn2_frozen_spectrum_fermi_response_is_inactive(
    driver: str,
) -> None:
    """Probe Fermi response to a fixed-spectrum directional perturbation."""
    _, _, _, _, inputs, _, _, result = _no2_functional_input(driver)
    settings = inputs.fermi_settings
    assert settings is not None and settings.smear

    spectrum = result.emo.detach()
    direction = torch.linspace(-1.0, 1.0, spectrum.numel(), **_DD).reshape_as(
        spectrum
    )
    probe = torch.linspace(
        -0.7, 1.1, result.occupation.numel(), **_DD
    ).reshape_as(result.occupation)

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
        (result.occupation - result.occupation.round()).abs().max()
    ) < (1.0e-10)
    assert max(samples) - min(samples) < 5.0e-8
    assert abs(automatic - samples[-1]) < 5.0e-8


def test_no2_gfn1_gfn2_term_activation_at_same_gfn2_state() -> None:
    """Compare GFN1/GFN2 SCF terms at one fixed GFN2 converged state."""
    model_system, _, _, _, inputs, layout, state, result = (
        _no2_functional_input()
    )
    gfn1_system = _gfn1_no2_system()
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
    gfn1_potential = SCFPotential(
        result.potential.mono, result.potential.dipole
    )
    gfn1_layout = potential_layout(gfn1_potential)
    gfn1_state = pack_potential(gfn1_potential, gfn1_layout)
    mono_size = result.potential.mono.numel()
    dipole_size = result.potential.dipole.numel()
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
    _, _, _, _, inputs, layout, state, _ = _no2_functional_input(driver)
    direction = torch.linspace(-0.2, 0.2, state.numel(), **_DD).reshape_as(
        state
    )
    projection = torch.linspace(-0.9, 1.2, state.numel(), **_DD).reshape_as(
        state
    )

    def map_probe(parameter: Tensor) -> Tensor:
        mapped = potential_map(state + parameter * direction, inputs, layout)
        return (mapped.mapped_x * projection).sum()

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

    steps = (5.0e-3, 2.5e-3) if driver == "pytorch" else (1.25e-2, 1.0e-2)
    samples = [finite_difference(step) for step in steps]
    assert abs(samples[0] - samples[1]) < 2.0e-3
    assert abs(automatic - samples[-1]) < 1.0e-2
