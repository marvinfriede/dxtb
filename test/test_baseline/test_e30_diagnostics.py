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
from tad_mctc import storch

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
    AndersonState,
    anderson_mix,
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
    simple_mix,
)
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.scf.utils import get_density
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction.filling import get_fermi_occupation_core

from .compute import (
    DRIVERS,
    _calculator,
    _hyperpolarizability,
    _pol_deriv,
    compute,
)
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
    reference, _, _ = load_reference(
        REFERENCE_DIR / "tblite" / "gfn2" / "NO2.npz"
    )
    stored, _, _ = load_reference(REFERENCE_DIR / f"gfn2-{driver}" / "NO2.npz")
    if quantity == "hyperpolarizability":
        nested = _hyperpolarizability(system, "gfn2", driver)[quantity][index]
    else:
        nested = _pol_deriv(system, "gfn2", driver)[quantity][index]
    base_calculator = _calculator(system, "gfn2", driver)
    base_result = base_calculator.singlepoint(
        system.positions.clone(),
        system.charge,
        system.spin,
        field=torch.zeros(3, **_DD),
    )
    system_field = torch.zeros(3, **_DD)
    field_axis = torch.zeros(3, **_DD)
    field_axis[1] = 1.0

    def evaluate(positions: Tensor, field_value: Tensor) -> tuple[float, int]:
        point_calculator = _calculator(system, "gfn2", driver)
        result = point_calculator.singlepoint(
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
    tolerance: float | None = None,
    map_trace: list[dict[str, object]] | None = None,
):
    """Return trajectory outcome and finalization from one prepared solve."""
    model_system, inputs, layout, packed, settings = _prepare_functional_at(
        calculator, positions, field, maxiter=maxiter, tolerance=tolerance
    )

    def map_fn(value: Tensor):
        result = potential_map(value, inputs, layout)
        if map_trace is not None:
            map_trace.append(_map_trace_record(value, result, len(map_trace)))
        return result

    outcome = _solve_map_anderson(
        packed,
        settings,
        map_fn,
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


def _prepare_functional_at(
    calculator: Calculator,
    positions: Tensor,
    field: Tensor,
    *,
    maxiter: int | None = None,
    tolerance: float | None = None,
):
    """Prepare the exact functional potential map without running SCF."""
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
    if tolerance is not None:
        config = replace(config, x_atol=tolerance, x_atol_max=tolerance)
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
    return model_system, inputs, layout, packed, settings


def _map_trace_record(value: Tensor, result, index: int) -> dict[str, object]:
    """Capture gauge-invariant spectrum and raw residual for one map call."""
    energies = result.spectral.emo.detach()
    occupations = result.spectral.occupation.detach()
    pair_gaps = (energies[..., 1:] - energies[..., :-1]).abs()
    occupied_pair_gaps = []
    occupied_pair_records = []
    frontier_gaps = []
    for channel in range(occupations.shape[-2]):
        channel_gaps = (energies[1:] - energies[:-1]).abs()
        occupied_pairs = (occupations[channel, 1:] > 0.5) & (
            occupations[channel, :-1] > 0.5
        )
        if occupied_pairs.any():
            masked_gaps = channel_gaps.masked_fill(~occupied_pairs, torch.inf)
            pair_index = int(masked_gaps.argmin())
            gap = float(masked_gaps.min())
            occupied_pair_gaps.append(gap)
            occupied_pair_records.append((channel, pair_index, gap))
        occupied_energies = energies[occupations[channel] > 1.0e-8]
        virtual_energies = energies[occupations[channel] < 1.0 - 1.0e-8]
        if occupied_energies.numel() and virtual_energies.numel():
            frontier_gaps.append(
                float(
                    (occupied_energies[:, None] - virtual_energies[None, :])
                    .abs()
                    .min()
                )
            )
    fractional = (occupations - occupations.round()).abs().max()
    return {
        "index": index,
        "input_state_tensor": value.detach(),
        "residual_l2": float(
            torch.linalg.vector_norm(result.mapped_x.detach() - value.detach())
        ),
        "minimum_adjacent_gap": float(pair_gaps.min()),
        "minimum_occupied_pair": min(
            occupied_pair_records, key=lambda row: row[2]
        ),
        "minimum_occupied_pair_gap": min(occupied_pair_gaps),
        "minimum_frontier_gap": min(frontier_gaps) if frontier_gaps else None,
        "orbital_energies": energies.cpu().tolist(),
        "occupations": occupations.cpu().tolist(),
        "max_fractional_occupation": float(fractional),
        "fermi_changes_materially": bool(fractional > 1.0e-8),
    }


def _fixed_n_anderson_at(
    calculator: Calculator,
    positions: Tensor,
    field: Tensor,
    n_maps: int,
    *,
    mixing: bool = True,
    map_trace: list[dict[str, object]] | None = None,
    boundary_record: dict[str, object] | None = None,
    anderson_trace: list[dict[str, object]] | None = None,
    trajectory_trace: list[dict[str, object]] | None = None,
    simple_damping: float | None = None,
    boundary_map: int = 8,
):
    """Compose fixed-N maps, optionally retaining a chosen solver boundary."""
    _, inputs, layout, guess, settings = _prepare_functional_at(
        calculator, positions, field
    )
    if boundary_record is not None:
        boundary_record["inputs"] = inputs
        boundary_record["settings"] = settings
    if settings.dynamic_damping.enabled:
        raise AssertionError(
            "NO2 fixed-N probe expects static Anderson damping"
        )

    def map_fn(value: Tensor):
        result = potential_map(value, inputs, layout)
        if map_trace is not None:
            map_trace.append(_map_trace_record(value, result, len(map_trace)))
        return result

    def trace_map(index: int, value: Tensor, result) -> None:
        if trajectory_trace is None:
            return
        record = _map_trace_record(value, result, index - 1)
        trajectory_trace.append(
            {
                "map": index,
                "input": value.detach().clone(),
                "mapped_x": result.mapped_x.detach().clone(),
                "raw": (result.mapped_x - value).detach().clone(),
                "hamiltonian": result.hamiltonian.detach().clone(),
                "density": result.density.detach().clone(),
                "emo": result.emo.detach().clone(),
                "coefficients": result.spectral.coefficients.detach().clone(),
                "occupation": result.spectral.occupation.detach().clone(),
                "overlap": inputs.integrals.overlap.detach().clone(),
                "inverse_cholesky": inputs.inverse_cholesky.detach().clone(),
                "minimum_occupied_pair": record["minimum_occupied_pair"],
            }
        )

    selected = map_fn(guess)
    trace_map(1, guess, selected)
    current = guess
    mixer_state = None
    if simple_damping is not None:
        current = simple_mix(current, selected.mapped_x, simple_damping)
    elif mixing and settings.mix_guess:
        current, mixer_state = anderson_mix(
            current,
            selected.mapped_x,
            mixer_state,
            settings,
            step=1,
            damp=settings.damp,
        )
    else:
        current = selected.mapped_x

    for map_index in range(1, n_maps):
        input_value = current
        selected = map_fn(current)
        trace_map(map_index + 1, input_value, selected)
        if simple_damping is not None:
            before = current
            current = simple_mix(current, selected.mapped_x, simple_damping)
            if boundary_record is not None and map_index == boundary_map - 1:
                boundary_record.update(
                    transition_map_index=boundary_map,
                    transition_map_input=before,
                    transition_map_result=selected,
                    transition_state_before=before,
                    transition_state_after=current,
                    transition_stages=None,
                    following_map_result=None,
                )
            if boundary_record is not None and map_index == boundary_map:
                boundary_record.update(following_map_result=selected)
        elif mixing:
            step = map_index + (1 if settings.mix_guess else 0)
            if anderson_trace is not None and step > settings.generations:
                n = min(step - 1, settings.generations)
                raw = selected.mapped_x.reshape(-1) - current.reshape(-1)
                if mixer_state is None:
                    hist = raw.new_zeros(
                        (settings.generations + 1, raw.numel())
                    )
                    x_hist = torch.cat(
                        (current.reshape(-1).unsqueeze(0), hist[1:]), dim=0
                    )
                    f_hist = hist
                else:
                    x_hist, f_hist = mixer_state.x_hist, mixer_state.f_hist
                f_current = torch.cat((raw.unsqueeze(0), f_hist[1:]), dim=0)
                df = f_current[0] - f_current[1 : n + 1]
                a = torch.einsum("iv,jv->ij", df, df)
                b = torch.einsum("hv,v->h", df, f_current[0])
                if settings.diagonal_offset is not None:
                    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
                    scale = torch.where(
                        eye != 0,
                        eye + settings.diagonal_offset**2,
                        torch.ones_like(eye),
                    )
                    a = a * scale
                singular_values = torch.linalg.svdvals(a.detach())
                theta = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
                anderson_trace.append(
                    {
                        "map_index": map_index,
                        "step": step,
                        "history_length": n,
                        "singular_values": singular_values.cpu().tolist(),
                        "condition_number": float(
                            singular_values.max()
                            / singular_values.min().clamp_min(
                                torch.finfo(singular_values.dtype).tiny
                            )
                        ),
                        "df_row_norms": torch.linalg.vector_norm(
                            df.detach(), dim=-1
                        )
                        .cpu()
                        .tolist(),
                        "b_norm": float(torch.linalg.vector_norm(b.detach())),
                        "theta_norm": float(
                            torch.linalg.vector_norm(theta.detach())
                        ),
                        "damping": settings.damp,
                    }
                )
            before = current
            state_before = mixer_state
            current, mixer_state = anderson_mix(
                current,
                selected.mapped_x,
                mixer_state,
                settings,
                step=step,
                transition_active=torch.ones(
                    (), dtype=torch.bool, device=current.device
                ),
                damp=settings.damp,
            )
            if trajectory_trace is not None:
                stages = _anderson_transition_stages(
                    before,
                    selected.mapped_x,
                    state_before,
                    settings,
                    step,
                )
                trajectory_trace[-1]["anderson"] = {
                    name: value.detach().clone()
                    for name, value in stages.items()
                    if isinstance(value, Tensor)
                }
                trajectory_trace[-1]["next_current"] = current.detach().clone()
                trajectory_trace[-1][
                    "next_x_hist"
                ] = mixer_state.x_hist.detach().clone()
                trajectory_trace[-1][
                    "next_f_hist"
                ] = mixer_state.f_hist.detach().clone()
            if boundary_record is not None and map_index == boundary_map - 1:
                stages = _anderson_transition_stages(
                    before,
                    selected.mapped_x,
                    state_before,
                    settings,
                    step,
                )
                assert torch.allclose(
                    stages["mixed"].reshape_as(current),
                    current,
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )
                assert torch.allclose(
                    stages["next_x_hist"],
                    mixer_state.x_hist,
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )
                assert torch.allclose(
                    stages["next_f_hist"],
                    mixer_state.f_hist,
                    atol=1.0e-12,
                    rtol=1.0e-12,
                )
                boundary_record.update(
                    transition_map_index=boundary_map,
                    transition_map_input=before,
                    transition_map_result=selected,
                    transition_mixer_state_before=state_before,
                    transition_mixer_state_after=mixer_state,
                    transition_state_before=before,
                    transition_state_after=current,
                    transition_stages=stages,
                )
                if boundary_map == 8:
                    boundary_record.update(
                        map8_input=before,
                        map8_result=selected,
                        map8_mixer_state_before=state_before,
                        map8_mixer_state_after=mixer_state,
                        map8_state_before_transition=before,
                        map8_state_after_transition=current,
                        map8_transition_stages=stages,
                    )
            if boundary_record is not None and map_index == boundary_map:
                boundary_record.update(following_map_result=selected)
                if boundary_map == 8:
                    boundary_record.update(map9_result=selected)
        else:
            current = selected.mapped_x
    return selected, inputs


def _anderson_transition_stages(
    old: Tensor,
    mapped: Tensor,
    state: AndersonState | None,
    settings,
    step: int,
) -> dict[str, Tensor]:
    """Expose differentiable intermediates of the existing Anderson formula."""
    flat_old = old.reshape(-1)
    flat_mapped = mapped.reshape(-1)
    if state is None:
        x_hist = torch.cat(
            (
                flat_old.unsqueeze(0),
                flat_old.new_zeros((settings.generations, flat_old.numel())),
            ),
            dim=0,
        )
        f_hist = flat_old.new_zeros(
            (settings.generations + 1, flat_old.numel())
        )
        previous = AndersonState(x_hist=x_hist, f_hist=f_hist)
    else:
        previous = state
    raw = flat_mapped - flat_old
    f_current = torch.cat((raw.unsqueeze(0), previous.f_hist[1:]), dim=0)
    x_current = torch.cat((flat_old.unsqueeze(0), previous.x_hist[1:]), dim=0)
    if not (
        step > settings.generations or (step > 1 and not settings.soft_start)
    ):
        mixed = x_current[0] + f_current[0] * settings.damp_init
        rolled_x = torch.roll(x_current, 1, 0)
        return {
            "raw": raw,
            "f_current": f_current,
            "x_current": x_current,
            "mixed": mixed,
            "next_x_hist": torch.cat((mixed.unsqueeze(0), rolled_x[1:]), dim=0),
            "next_f_hist": torch.roll(f_current, 1, 0),
        }
    n = min(step - 1, settings.generations)
    df = f_current[0] - f_current[1 : n + 1]
    a = torch.einsum("iv,jv->ij", df, df)
    b = torch.einsum("hv,v->h", df, f_current[0])
    if settings.diagonal_offset is not None:
        eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        one = a.new_tensor(1.0)
        scale = torch.where(eye != 0, eye + settings.diagonal_offset**2, one)
        a = a * scale
    theta = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
    x_bar_raw = torch.einsum(
        "h,hv->v", theta, x_current[1 : n + 1] - x_current[0]
    )
    f_bar_raw = torch.einsum("h,hv->v", theta, -df)
    x_bar = x_bar_raw + x_current[0]
    f_bar = f_bar_raw + f_current[0]
    mixed = x_bar + settings.damp * f_bar
    rolled_x = torch.roll(x_current, 1, 0)
    return {
        "raw": raw,
        "f_current": f_current,
        "x_current": x_current,
        "df": df,
        "A": a,
        "b": b,
        "theta": theta,
        "x_bar_raw": x_bar_raw,
        "f_bar_raw": f_bar_raw,
        "x_bar": x_bar,
        "f_bar": f_bar,
        "mixed": mixed,
        "next_x_hist": torch.cat((mixed.unsqueeze(0), rolled_x[1:]), dim=0),
        "next_f_hist": torch.roll(f_current, 1, 0),
    }


def _energy_from_map_result(map_result, inputs) -> Tensor:
    """Scalar physical energy from one raw map evaluation."""
    return _spectral_energy(map_result.spectral, inputs)


def _third_response(
    energy_fn,
    step: float | tuple[float, ...],
    *,
    check_jvp: bool = False,
    convention: float = -1.0,
) -> dict[str, object]:
    """Compare nested response with a central FD of the second derivative."""
    center = torch.zeros((), **_DD, requires_grad=True)
    energy, iterations = energy_fn(center, torch.zeros((), **_DD))
    first = (
        torch.autograd.grad(energy, center, create_graph=True)[0]
        if energy.requires_grad
        else torch.zeros_like(center)
    )
    second = (
        torch.autograd.grad(first, center, create_graph=True)[0]
        if first.requires_grad
        else torch.zeros_like(first)
    )
    third = (
        torch.autograd.grad(second, center)[0]
        if second.requires_grad
        else torch.zeros_like(second)
    )
    jvp = None
    if check_jvp:
        jvp = torch.func.jvp(
            lambda x: energy_fn(x, torch.zeros((), **_DD))[0],
            (torch.zeros((), **_DD),),
            (torch.ones((), **_DD),),
        )[1]

    def derivatives_at(field_value: float) -> tuple[float, float, float, int]:
        field = torch.tensor(field_value, **_DD, requires_grad=True)
        value, count = energy_fn(field, torch.zeros((), **_DD))
        first_local = (
            torch.autograd.grad(value, field, create_graph=True)[0]
            if value.requires_grad
            else torch.zeros_like(field)
        )
        second_local = (
            torch.autograd.grad(first_local, field)[0]
            if first_local.requires_grad
            else torch.zeros_like(first_local)
        )
        return (
            float(value.detach()),
            float(first_local.detach()),
            float(second_local.detach()),
            int(count),
        )

    steps = (step,) if isinstance(step, float) else step
    fd_sweep = []
    for step_value in steps:
        plus = derivatives_at(step_value)
        minus = derivatives_at(-step_value)
        fd_sweep.append(
            {
                "step": step_value,
                "fd_first": (plus[0] - minus[0]) / (2.0 * step_value),
                "fd_second": (plus[1] - minus[1]) / (2.0 * step_value),
                "fd_of_second": convention
                * (plus[2] - minus[2])
                / (2.0 * step_value),
                "iterations_plus": plus[3],
                "iterations_minus": minus[3],
            }
        )
    result: dict[str, object] = {
        "iterations_center": int(iterations),
        "reverse_first": float(first.detach()),
        "reverse_second": float(second.detach()),
        "reverse_third": convention * float(third.detach()),
        "fd_of_second_sweep": fd_sweep,
    }
    if jvp is not None:
        result["jvp_first"] = float(jvp)
        result["jvp_vs_reverse_first_abs"] = abs(
            float(jvp) - float(first.detach())
        )
    return result


def _mixed_response(
    energy_fn, step: float | tuple[float, ...]
) -> dict[str, object]:
    """Compare mixed nested response with an FD of field polarizability."""
    geometry = torch.zeros((), **_DD, requires_grad=True)
    field = torch.zeros((), **_DD, requires_grad=True)
    energy, center_iterations = energy_fn(field, geometry)
    first_field_result = torch.autograd.grad(
        energy, field, create_graph=True, allow_unused=True
    )[0]
    first_field = (
        first_field_result
        if first_field_result is not None
        else torch.zeros_like(field)
    )
    second_field = (
        torch.autograd.grad(first_field, field, create_graph=True)[0]
        if first_field.requires_grad
        else torch.zeros_like(first_field)
    )
    alpha = -second_field
    mixed_result = (
        torch.autograd.grad(alpha, geometry, allow_unused=True)[0]
        if alpha.requires_grad
        else None
    )
    mixed = (
        mixed_result if mixed_result is not None else torch.zeros_like(alpha)
    )

    def alpha_at(displacement: float) -> tuple[float, int]:
        field_value = torch.zeros((), **_DD, requires_grad=True)
        geom_value = torch.tensor(displacement, **_DD)
        value, iterations = energy_fn(field_value, geom_value)
        first_result = (
            torch.autograd.grad(
                value, field_value, create_graph=True, allow_unused=True
            )[0]
            if value.requires_grad
            else None
        )
        first = (
            first_result
            if first_result is not None
            else torch.zeros_like(field_value)
        )
        second = (
            torch.autograd.grad(first, field_value)[0]
            if first.requires_grad
            else torch.zeros_like(first)
        )
        return -float(second.detach()), int(iterations)

    steps = (step,) if isinstance(step, float) else step
    fd_sweep = []
    for step_value in steps:
        plus, plus_iterations = alpha_at(step_value)
        minus, minus_iterations = alpha_at(-step_value)
        fd_sweep.append(
            {
                "step": step_value,
                "fd_of_field_second": (plus - minus) / (2.0 * step_value),
                "iterations_plus": plus_iterations,
                "iterations_minus": minus_iterations,
            }
        )
    return {
        "iterations_center": int(center_iterations),
        "iterations_plus": fd_sweep[0]["iterations_plus"],
        "iterations_minus": fd_sweep[0]["iterations_minus"],
        "nested_mixed_third": float(mixed.detach()),
        "fd_of_field_second": fd_sweep[0]["fd_of_field_second"],
        "fd_sweep": fd_sweep,
    }


def _trajectory_energy_at(
    calculator: Calculator,
    field_value: Tensor,
    displacement: Tensor,
    *,
    eager: bool,
    maxiter: int | None,
    stage: str,
    tolerance: float | None = None,
    solver_record: dict[str, object] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return one solver/finalization energy for physical field/geometry."""
    positions = SYSTEMS["NO2"].positions.clone()
    positions[0, 1] = positions[0, 1] + displacement
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0
    outcome, inputs, _, finalized = _functional_solve_at(
        calculator,
        positions,
        field_value * axis,
        eager=eager,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    if solver_record is not None:
        solver_record.update(
            {
                "iterations": int(outcome.iterations),
                "converged": bool(outcome.converged),
                "residual_max_abs": float(
                    outcome.residual.detach().abs().max()
                ),
            }
        )
    if stage == "solver":
        energy = _selected_map_energy(outcome, inputs)
    elif stage == "finalized":
        energy = (finalized.energy + finalized.fenergy).sum()
    else:
        raise ValueError(f"Unknown diagnostic stage: {stage}")
    return energy, outcome.iterations


def _raw_map_energy_at_state(
    calculator: Calculator,
    state: Tensor,
    field_value: Tensor,
    displacement: Tensor,
) -> tuple[Tensor, int]:
    """Evaluate one physical raw map with the SCF input state held fixed."""
    positions = SYSTEMS["NO2"].positions.clone()
    positions[0, 1] = positions[0, 1] + displacement
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0
    _, inputs, layout, _, _ = _prepare_functional_at(
        calculator, positions, field_value * axis
    )
    mapped = potential_map(state, inputs, layout)
    return _energy_from_map_result(mapped, inputs), 1


def _raw_map_density_probe_at_state(
    calculator: Calculator,
    state: Tensor,
    field_value: Tensor,
    displacement: Tensor,
    probe: Tensor,
) -> tuple[Tensor, int]:
    """Project gauge-invariant density from one fixed-state physical map."""
    positions = SYSTEMS["NO2"].positions.clone()
    positions[0, 1] = positions[0, 1] + displacement
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0
    _, inputs, layout, _, _ = _prepare_functional_at(
        calculator, positions, field_value * axis
    )
    mapped = potential_map(state, inputs, layout)
    return (mapped.spectral.density * probe).sum(), 1


def _two_map_density_probe_at_state(
    calculator: Calculator,
    state: Tensor,
    field_value: Tensor,
    displacement: Tensor,
    probe: Tensor,
) -> tuple[Tensor, int]:
    """Project density after two raw maps from one trajectory state."""
    positions = SYSTEMS["NO2"].positions.clone()
    positions[0, 1] = positions[0, 1] + displacement
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0
    _, inputs, layout, _, _ = _prepare_functional_at(
        calculator, positions, field_value * axis
    )
    first = potential_map(state, inputs, layout)
    second = potential_map(first.mapped_x, inputs, layout)
    return (second.spectral.density * probe).sum(), 2


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


@pytest.mark.parametrize("n_maps", (18, 20, 22, 24))
def test_no2_fixed_n_anderson_derivative_matches_fixed_n_fd(
    n_maps: int,
) -> None:
    """Compare AD and derivative-level FD for an exact fixed-N map function."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")

    def energy_fn(field_value: Tensor, displacement: Tensor):
        positions = SYSTEMS["NO2"].positions.clone()
        positions[0, 1] = positions[0, 1] + displacement
        axis = torch.zeros(3, **_DD)
        axis[1] = 1.0
        map_result, inputs = _fixed_n_anderson_at(
            calculator,
            positions,
            field_value * axis,
            n_maps,
        )
        return _energy_from_map_result(map_result, inputs), n_maps

    result = _third_response(
        energy_fn, (1.0e-4, 5.0e-5, 2.0e-5), check_jvp=True
    )
    assert result["iterations_center"] == n_maps
    assert all(
        row["iterations_plus"] == row["iterations_minus"] == n_maps
        for row in result["fd_of_second_sweep"]
    )
    fd_values = [
        float(row["fd_of_second"]) for row in result["fd_of_second_sweep"]
    ]
    assert max(fd_values) - min(fd_values) < 5.0e-4
    derivative_gap = abs(
        float(result["reverse_third"]) - float(np.median(fd_values))
    )
    expected_gap_floor = {18: 100.0, 20: 1.0, 22: 0.01, 24: 0.0}[n_maps]
    assert derivative_gap > expected_gap_floor
    if n_maps == 24:
        assert derivative_gap < 1.0e-3
    print({"fixed_n_maps": n_maps, **result})


def test_no2_fixed_n_simple_mixing_composition_vs_fd() -> None:
    """Check fixed-damping map composition without an Anderson solve."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0

    def energy_fn(field_value: Tensor, displacement: Tensor):
        positions = SYSTEMS["NO2"].positions.clone()
        positions[0, 1] += displacement
        map_result, inputs = _fixed_n_anderson_at(
            calculator,
            positions,
            field_value * axis,
            20,
            simple_damping=0.2,
        )
        return _energy_from_map_result(map_result, inputs), 20

    result = _third_response(
        energy_fn, (1.0e-4, 5.0e-5, 2.0e-5), check_jvp=True
    )
    fd_values = [
        float(row["fd_of_second"]) for row in result["fd_of_second_sweep"]
    ]
    assert max(fd_values) - min(fd_values) < 5.0e-4
    assert (
        abs(float(result["reverse_third"]) - float(np.median(fd_values)))
        < 5.0e-4
    )
    print({"fixed_n_maps": 20, "simple_damping": 0.2, **result})


def test_no2_fixed_n_prefix_derivative_localization() -> None:
    """Check small-step consistency at the map-eight/nine boundary."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0
    records = []
    for n_maps in (8, 9):

        def energy_fn(field_value: Tensor, displacement: Tensor):
            positions = SYSTEMS["NO2"].positions.clone()
            positions[0, 1] += displacement
            map_result, inputs = _fixed_n_anderson_at(
                calculator,
                positions,
                field_value * axis,
                n_maps,
            )
            return _energy_from_map_result(map_result, inputs), n_maps

        response = _third_response(
            energy_fn, (1.0e-5, 5.0e-6, 2.0e-6, 1.0e-6, 5.0e-7)
        )
        fd_values = [
            float(row["fd_of_second"])
            for row in response["fd_of_second_sweep"][-3:]
        ]
        nested = float(response["reverse_third"])
        plateau_spread = max(fd_values) - min(fd_values)
        distance = max(min(fd_values) - nested, nested - max(fd_values), 0.0)
        records.append(
            {
                "map_prefix": n_maps,
                "nested_third": nested,
                "fd_plateau": fd_values,
                "plateau_spread": plateau_spread,
                "distance_outside_plateau": distance,
            }
        )
    assert all(np.isfinite(row["distance_outside_plateau"]) for row in records)
    assert all(
        row["distance_outside_plateau"] <= row["plateau_spread"]
        for row in records
    )
    print({"fixed_n_prefixes": records})


@pytest.mark.parametrize("n_maps", range(9, 21))
def test_no2_later_fixed_n_prefix_derivative_consistency(n_maps: int) -> None:
    """Find where longer Anderson prefixes lose derivative-level consistency."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0

    def energy_fn(field_value: Tensor, displacement: Tensor):
        positions = SYSTEMS["NO2"].positions.clone()
        positions[0, 1] += displacement
        result, inputs = _fixed_n_anderson_at(
            calculator, positions, field_value * axis, n_maps
        )
        return _energy_from_map_result(result, inputs), n_maps

    response = _third_response(
        energy_fn,
        (1.0e-5, 5.0e-6, 2.0e-6, 1.0e-6, 5.0e-7),
        check_jvp=True,
    )
    fd_values = [
        float(row["fd_of_second"])
        for row in response["fd_of_second_sweep"][-3:]
    ]
    nested = float(response["reverse_third"])
    spread = max(fd_values) - min(fd_values)
    distance = max(min(fd_values) - nested, nested - max(fd_values), 0.0)
    assert all(
        row["iterations_plus"] == row["iterations_minus"] == n_maps
        for row in response["fd_of_second_sweep"]
    )
    assert response["jvp_vs_reverse_first_abs"] < 1.0e-5 * max(
        1.0, abs(float(response["reverse_first"]))
    )
    print(
        {
            "fixed_n": n_maps,
            "hyper_derivative_level_fd_plateau": fd_values,
            "plateau_spread": spread,
            "nested_third": nested,
            "distance_outside_plateau": distance,
            "consistent_with_plateau_bound": distance <= spread,
            **response,
        }
    )


def test_no2_fixed_n20_pol_deriv_level_consistency() -> None:
    """Check mixed-response consistency for the historical map count."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0

    def energy_fn(field_value: Tensor, displacement: Tensor):
        positions = SYSTEMS["NO2"].positions.clone()
        positions[0, 1] += displacement
        result, inputs = _fixed_n_anderson_at(
            calculator, positions, field_value * axis, 20
        )
        return _energy_from_map_result(result, inputs), 20

    response = _mixed_response(
        energy_fn, (1.0e-5, 5.0e-6, 2.0e-6, 1.0e-6, 5.0e-7)
    )
    fd_values = [
        float(row["fd_of_field_second"]) for row in response["fd_sweep"][-3:]
    ]
    nested = float(response["nested_mixed_third"])
    spread = max(fd_values) - min(fd_values)
    distance = max(min(fd_values) - nested, nested - max(fd_values), 0.0)
    assert all(
        row["iterations_plus"] == row["iterations_minus"] == 20
        for row in response["fd_sweep"]
    )
    assert distance <= spread
    print(
        {
            "fixed_n": 20,
            "pol_deriv_level_fd_plateau": fd_values,
            "plateau_spread": spread,
            "nested_mixed_third": nested,
            "distance_outside_plateau": distance,
            **response,
        }
    )


def _prefix_boundary_energy(
    calculator: Calculator,
    field_value: Tensor,
    displacement: Tensor,
    boundary: str,
    transition_map: int = 8,
) -> tuple[Tensor, int]:
    """Evaluate a transition boundary or the physical map immediately after."""
    positions = SYSTEMS["NO2"].positions.clone()
    positions[0, 1] += displacement
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0
    record: dict[str, object] = {}
    _fixed_n_anderson_at(
        calculator,
        positions,
        field_value * axis,
        transition_map + 1,
        boundary_record=record,
        boundary_map=transition_map,
    )
    if boundary == f"map{transition_map}":
        result = record["transition_map_result"]
        inputs = record["inputs"]
        return _energy_from_map_result(result, inputs), transition_map + 1
    transition = record["transition_stages"]
    transition_outputs = {
        "current": record["transition_state_after"],
        "x_hist": record["transition_mixer_state_after"].x_hist,
        "f_hist": record["transition_mixer_state_after"].f_hist,
    }
    if boundary in transition_outputs:
        value = transition_outputs[boundary]
        weights = torch.linspace(-1.0, 1.0, value.numel(), **_DD).reshape_as(
            value
        )
        return (
            weights * value
        ).sum() / value.numel() ** 0.5, transition_map + 1
    if boundary in transition:
        value = transition[boundary]
        weights = torch.linspace(-1.0, 1.0, value.numel(), **_DD).reshape_as(
            value
        )
        return (
            weights * value
        ).sum() / value.numel() ** 0.5, transition_map + 1
    if boundary == f"map{transition_map + 1}":
        result = record["following_map_result"]
        inputs = record["inputs"]
        return _energy_from_map_result(result, inputs), transition_map + 1
    raise ValueError(f"Unknown prefix boundary: {boundary}")


@pytest.mark.parametrize(
    "boundary",
    (
        "map8",
        "current",
        "x_hist",
        "f_hist",
        "raw",
        "f_current",
        "x_current",
        "df",
        "A",
        "b",
        "theta",
        "x_bar_raw",
        "f_bar_raw",
        "x_bar",
        "f_bar",
        "mixed",
        "next_x_hist",
        "next_f_hist",
        "map9",
    ),
)
def test_no2_prefix8_transition_map9_localization(boundary: str) -> None:
    """Probe map 8, each transition output and physical map 9 separately."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")

    def energy_fn(field_value: Tensor, displacement: Tensor):
        return _prefix_boundary_energy(
            calculator, field_value, displacement, boundary
        )

    if boundary == "map8":
        steps = (1.0e-4, 5.0e-5, 2.0e-5, 1.0e-5, 5.0e-6, 2.0e-6)
    elif boundary == "current":
        steps = (
            1.0e-4,
            5.0e-5,
            2.0e-5,
            1.0e-5,
            5.0e-6,
            2.0e-6,
            1.0e-6,
            5.0e-7,
            2.0e-7,
        )
    else:
        steps = (1.0e-5, 5.0e-6, 2.0e-6, 1.0e-6, 5.0e-7, 2.0e-7)
    response = _third_response(energy_fn, steps, check_jvp=True)
    assert response["jvp_vs_reverse_first_abs"] < 1.0e-5 * max(
        1.0, abs(float(response["reverse_first"]))
    )
    fd_values = [
        float(row["fd_of_second"]) for row in response["fd_of_second_sweep"]
    ]
    plateau = fd_values[-3:]
    plateau_center = float(np.median(plateau))
    plateau_spread = max(plateau) - min(plateau)
    nested = float(response["reverse_third"])
    gap = abs(nested - plateau_center)
    distance_outside_plateau = max(
        min(plateau) - nested, nested - max(plateau), 0.0
    )
    assert np.isfinite(plateau_center)
    assert np.isfinite(plateau_spread)
    assert np.isfinite(gap)
    assert distance_outside_plateau <= plateau_spread
    print(
        {
            "prefix_boundary": boundary,
            "fd_plateau": plateau,
            "plateau_spread": plateau_spread,
            "nested_vs_fd_plateau": gap,
            "distance_outside_fd_plateau": distance_outside_plateau,
            **response,
        }
    )


@pytest.mark.parametrize(
    "boundary",
    (
        "map10",
        "raw",
        "f_current",
        "x_current",
        "df",
        "A",
        "b",
        "theta",
        "x_bar_raw",
        "f_bar_raw",
        "x_bar",
        "f_bar",
        "current",
        "mixed",
        "next_x_hist",
        "next_f_hist",
        "map11",
    ),
)
def test_no2_prefix10_transition_map11_characterization(boundary: str) -> None:
    """Inspect the accelerated transition before the first unstable prefix."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")

    def energy_fn(field_value: Tensor, displacement: Tensor):
        return _prefix_boundary_energy(
            calculator,
            field_value,
            displacement,
            boundary,
            transition_map=10,
        )

    response = _third_response(
        energy_fn,
        (1.0e-6, 5.0e-7, 2.0e-7, 1.0e-7, 5.0e-8, 2.0e-8),
        check_jvp=True,
    )
    fd_values = [
        float(row["fd_of_second"]) for row in response["fd_of_second_sweep"]
    ]
    plateau = fd_values[-3:]
    plateau_spread = max(plateau) - min(plateau)
    nested = float(response["reverse_third"])
    distance_outside_plateau = max(
        min(plateau) - nested, nested - max(plateau), 0.0
    )
    assert np.isfinite(float(response["reverse_third"]))
    assert all(np.isfinite(value) for value in fd_values)
    assert distance_outside_plateau <= plateau_spread
    relative_jvp_gap = float(response["jvp_vs_reverse_first_abs"]) / max(
        1.0, abs(float(response["reverse_first"]))
    )
    preceding_operations = {"raw", "f_current", "x_current", "df", "A", "b"}
    if boundary in preceding_operations:
        assert relative_jvp_gap < 1.0e-6
    if boundary == "theta":
        assert 1.0e-6 < relative_jvp_gap < 1.0e-4
    print(
        {
            "transition_after_map": 10,
            "prefix11_boundary": boundary,
            "nested_third": response["reverse_third"],
            "fd_sweep": fd_values,
            "fd_spread": plateau_spread,
            "distance_outside_fd_plateau": distance_outside_plateau,
            "jvp_reverse_first_abs": response["jvp_vs_reverse_first_abs"],
            "jvp_reverse_first_relative_gap": relative_jvp_gap,
            **response,
        }
    )


@pytest.mark.parametrize(
    "boundary", ("map8", "current", "x_hist", "f_hist", "map9")
)
def test_no2_pol_deriv_prefix8_transition_map9(boundary: str) -> None:
    """Apply the same prefix boundary split to the mixed response."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")

    def energy_fn(field_value: Tensor, displacement: Tensor):
        return _prefix_boundary_energy(
            calculator, field_value, displacement, boundary
        )

    response = _mixed_response(
        energy_fn,
        (1.0e-5, 5.0e-6, 2.0e-6, 1.0e-6, 5.0e-7, 2.0e-7),
    )
    sweep = response["fd_sweep"]
    plateau = [float(row["fd_of_field_second"]) for row in sweep[-3:]]
    plateau_center = float(np.median(plateau))
    plateau_spread = max(plateau) - min(plateau)
    nested = float(response["nested_mixed_third"])
    gap = abs(nested - plateau_center)
    distance_outside_plateau = max(
        min(plateau) - nested, nested - max(plateau), 0.0
    )
    assert np.isfinite(gap)
    assert distance_outside_plateau <= plateau_spread
    print(
        {
            "pol_deriv_prefix_boundary": boundary,
            "fd_plateau": plateau,
            "plateau_spread": plateau_spread,
            "nested_vs_fd_plateau": gap,
            "distance_outside_fd_plateau": distance_outside_plateau,
            **response,
        }
    )


def test_no2_map9_actual_state_tangent_and_spectral_derivative() -> None:
    """Probe map 9 along the propagated state tangent and its physical FD."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    center_field = torch.zeros((), **_DD)
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0

    def mixed_state(field_value: Tensor, displacement: Tensor) -> Tensor:
        positions = SYSTEMS["NO2"].positions.clone()
        positions[0, 1] += displacement
        record: dict[str, object] = {}
        _fixed_n_anderson_at(
            calculator,
            positions,
            field_value * axis,
            8,
            boundary_record=record,
        )
        return record["map8_state_after_transition"]

    state, field_tangent = torch.func.jvp(
        lambda field_value: mixed_state(field_value, torch.zeros((), **_DD)),
        (center_field,),
        (torch.ones_like(center_field),),
    )
    _, geometry_tangent = torch.func.jvp(
        lambda displacement: mixed_state(center_field, displacement),
        (torch.zeros((), **_DD),),
        (torch.ones((), **_DD),),
    )
    positions = SYSTEMS["NO2"].positions.clone()
    _, inputs, layout, _, _ = _prepare_functional_at(
        calculator, positions, axis * center_field
    )
    probe = torch.linspace(
        -0.9,
        1.2,
        potential_map(state, inputs, layout).spectral.density.numel(),
        **_DD,
    ).reshape_as(potential_map(state, inputs, layout).spectral.density)

    def check_tangent(tangent: Tensor) -> dict[str, object]:
        def physical_map(parameter: Tensor) -> Tensor:
            result = potential_map(state + parameter * tangent, inputs, layout)
            return (result.spectral.density * probe).sum()

        parameter = torch.zeros((), **_DD, requires_grad=True)
        value = physical_map(parameter)
        first = torch.autograd.grad(value, parameter, create_graph=True)[0]
        second = torch.autograd.grad(first, parameter, create_graph=True)[0]
        third = torch.autograd.grad(second, parameter)[0]
        reverse = first.detach()
        jvp = torch.func.jvp(
            physical_map,
            (torch.zeros((), **_DD),),
            (torch.ones((), **_DD),),
        )[1]
        steps = (1.0e-4, 5.0e-5, 2.0e-5)
        fd_values = [
            float(
                (
                    physical_map(torch.tensor(step, **_DD))
                    - physical_map(torch.tensor(-step, **_DD))
                )
                / (2.0 * step)
            )
            for step in steps
        ]

        def second_at(point: float) -> Tensor:
            local = torch.tensor(point, **_DD, requires_grad=True)
            local_value = physical_map(local)
            local_first = torch.autograd.grad(
                local_value, local, create_graph=True
            )[0]
            return torch.autograd.grad(local_first, local)[0].detach()

        third_fd = [
            float((second_at(step) - second_at(-step)) / (2.0 * step))
            for step in steps
        ]
        return {
            "state_tangent_norm": float(torch.linalg.vector_norm(tangent)),
            "reverse": float(reverse),
            "jvp": float(jvp),
            "forward_fd": fd_values,
            "nested_third": float(third.detach()),
            "third_derivative_level_fd": third_fd,
        }

    field_probe = check_tangent(field_tangent)
    geometry_probe = check_tangent(geometry_tangent)
    for probe_result in (field_probe, geometry_probe):
        fd_values = probe_result["forward_fd"]
        third_fd_values = probe_result["third_derivative_level_fd"]
        assert max(fd_values) - min(fd_values) < 1.0e-5
        assert abs(probe_result["reverse"] - probe_result["jvp"]) < 1.0e-5
        assert (
            abs(probe_result["reverse"] - float(np.median(fd_values))) < 1.0e-5
        )
        assert max(third_fd_values) - min(third_fd_values) < 1.0e-2
        assert (
            abs(
                probe_result["nested_third"] - float(np.median(third_fd_values))
            )
            < 1.0e-2
        )
    print(
        {
            "map9_state_tangent_field": field_probe,
            "map9_state_tangent_geometry": geometry_probe,
        }
    )


@pytest.mark.parametrize("n_maps", (20, 24))
def test_no2_pol_deriv_fixed_n_anderson_response(n_maps: int) -> None:
    """Localize mixed-response behavior at fixed Anderson map counts."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0

    def energy_fn(field_value: Tensor, displacement: Tensor):
        positions = SYSTEMS["NO2"].positions.clone()
        positions[0, 1] += displacement
        result, inputs = _fixed_n_anderson_at(
            calculator, positions, field_value * axis, n_maps
        )
        return _energy_from_map_result(result, inputs), n_maps

    response = _mixed_response(energy_fn, 5.0e-5)
    assert response["iterations_center"] == n_maps
    assert response["iterations_plus"] == response["iterations_minus"] == n_maps
    gap = abs(
        float(response["nested_mixed_third"])
        - float(response["fd_of_field_second"])
    )
    if n_maps == 20:
        assert gap > 0.1
    else:
        assert gap < 1.0e-3
    response["absolute_gap"] = gap
    print({"pol_deriv_fixed_n_maps": n_maps, **response})


def test_no2_pol_deriv_tighter_tolerance_response() -> None:
    """Compare mixed response at historical and tighter SCF tolerances."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    records = []
    for tolerance in (1.0e-10, 1.0e-12):

        def energy_fn(field_value: Tensor, displacement: Tensor):
            return _trajectory_energy_at(
                calculator,
                field_value,
                displacement,
                eager=False,
                maxiter=40,
                stage="solver",
                tolerance=tolerance,
            )

        response = _mixed_response(energy_fn, 5.0e-5)
        records.append({"tolerance": tolerance, **response})
    assert all(np.isfinite(record["nested_mixed_third"]) for record in records)
    assert (
        records[0]["iterations_center"],
        records[0]["iterations_plus"],
        records[0]["iterations_minus"],
    ) == (20, 20, 20)
    assert (
        records[1]["iterations_center"],
        records[1]["iterations_plus"],
        records[1]["iterations_minus"],
    ) == (24, 23, 23)
    gaps = [
        abs(float(row["nested_mixed_third"]) - float(row["fd_of_field_second"]))
        for row in records
    ]
    assert gaps[0] > 0.1
    assert gaps[1] < 1.0e-3
    for row, gap in zip(records, gaps):
        row["absolute_gap"] = gap
    print({"pol_deriv_tolerance_sweep": records})


def test_no2_anderson_transition_conditioning() -> None:
    """Record Anderson linear-system conditioning near the first bad prefix."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0
    records = {}
    for label, field_value in (
        ("minus", -5.0e-5),
        ("center", 0.0),
        ("plus", 5.0e-5),
    ):
        trace: list[dict[str, object]] = []
        _fixed_n_anderson_at(
            calculator,
            SYSTEMS["NO2"].positions,
            torch.tensor(field_value, **_DD) * axis,
            24,
            anderson_trace=trace,
        )
        records[label] = [
            row for row in trace if row["map_index"] in (5, 6, 7, 8, 9, 23)
        ]
    assert all(
        np.isfinite(row["condition_number"])
        and row["condition_number"] > 1.0e4
        and row["history_length"] <= 5
        for values in records.values()
        for row in values
    )
    print({"anderson_transition_conditioning": records})


@pytest.mark.parametrize(
    "observable", ("hyperpolarizability", "pol_deriv"), ids=("hyper", "pol")
)
def test_no2_fresh_calculator_reproducibility(observable: str) -> None:
    """Characterize repeat variation across fresh same-process calculators."""
    system = SYSTEMS["NO2"]

    def result_record(target) -> dict[str, object]:
        calculator = _calculator(
            target, "gfn2" if target is system else "gfn1", "pytorch"
        )
        result = calculator.singlepoint(
            target.positions.clone(), target.charge, target.spin
        )
        charges = result.charges
        potential = result.potential
        return {
            "energy": float(result.total.sum().detach()),
            "iterations": int(result.iterations),
            "charges": (
                None
                if charges is None
                else charges.mono.detach().cpu().tolist()
            ),
            "density": (
                None
                if result.density is None
                else result.density.detach().cpu().tolist()
            ),
            "emo": (
                None
                if result.emo is None
                else result.emo.detach().cpu().tolist()
            ),
            "occupation": (
                None
                if result.occupation is None
                else result.occupation.detach().cpu().tolist()
            ),
            "hamiltonian": (
                None
                if result.hamiltonian is None
                else result.hamiltonian.detach().cpu().tolist()
            ),
            "potential_mono": (
                None
                if potential is None
                else potential.mono.detach().cpu().tolist()
            ),
        }

    def result_summary(record: dict[str, object]) -> dict[str, object]:
        summary = {}
        for key, value in record.items():
            if key in ("energy", "iterations") or value is None:
                summary[key] = value
                continue
            array = np.asarray(value)
            summary[key] = {
                "shape": array.shape,
                "norm": float(np.linalg.norm(array)),
                "max_abs": float(np.max(np.abs(array))),
            }
        return summary

    def endpoint() -> float:
        if observable == "hyperpolarizability":
            result = _hyperpolarizability(system, "gfn2", "pytorch")
            return float(result[observable][_HYPER_INDEX].detach())
        result = _pol_deriv(system, "gfn2", "pytorch")
        return float(result[observable][_POL_INDEX].detach())

    result_before = result_record(system)
    first = endpoint()
    second = endpoint()
    third = endpoint()
    unrelated = result_record(SYSTEMS["H2O"])
    after_unrelated = endpoint()
    result_after = result_record(system)
    values = (first, second, third, after_unrelated)
    spread = max(values) - min(values)
    state_deltas = {}
    for key, before in result_before.items():
        after = result_after[key]
        if before is None or after is None:
            state_deltas[key] = None
        else:
            state_deltas[key] = float(
                np.max(np.abs(np.asarray(before) - np.asarray(after)))
            )
    assert np.isfinite(values).all()
    assert (
        max(value for value in state_deltas.values() if value is not None)
        < 1.0e-10
    )
    assert spread > 1.0e-5
    print(
        {
            "observable": observable,
            "same_process_AAA_then_unrelated_A": values,
            "spread": spread,
            "result_before": result_summary(result_before),
            "unrelated_result": result_summary(unrelated),
            "result_after": result_summary(result_after),
            "result_before_after_max_abs_delta": state_deltas,
        }
    )


def test_no2_fresh_calculator_forward_trajectory_and_gauge() -> None:
    """Compare hidden center trajectories, solve tensors and spectral gauge."""
    system = SYSTEMS["NO2"]
    traces: list[list[dict[str, object]]] = []
    endpoint_values = []
    for repeat in range(3):
        endpoint_values.append(
            float(
                _hyperpolarizability(system, "gfn2", "pytorch")[
                    "hyperpolarizability"
                ][_HYPER_INDEX].detach()
            )
        )
        calculator = _calculator(system, "gfn2", "pytorch")
        trace: list[dict[str, object]] = []
        _fixed_n_anderson_at(
            calculator,
            system.positions,
            torch.zeros(3, **_DD),
            20,
            trajectory_trace=trace,
        )
        traces.append(trace)
        if repeat == 1:
            unrelated = _calculator(SYSTEMS["H2O"], "gfn1", "pytorch")
            unrelated.singlepoint(
                SYSTEMS["H2O"].positions.clone(),
                SYSTEMS["H2O"].charge,
                SYSTEMS["H2O"].spin,
            )

    def leaves(row: dict[str, object]):
        for key, value in row.items():
            if isinstance(value, Tensor):
                yield key, value
            elif isinstance(value, dict):
                for child_key, child in value.items():
                    if isinstance(child, Tensor):
                        yield f"{key}.{child_key}", child

    first_forward_difference = None
    first_bitwise_difference = None
    first_physical_difference = None
    largest_physical_difference = None
    max_forward_delta = 0.0
    max_physical_delta = 0.0
    for map_index, rows in enumerate(zip(*traces), start=1):
        first = dict(leaves(rows[0]))
        for repeat, row in enumerate(rows[1:], start=1):
            for name, value in leaves(row):
                delta = float((first[name] - value).abs().max())
                if (
                    not torch.equal(first[name], value)
                    and first_bitwise_difference is None
                ):
                    first_bitwise_difference = {
                        "map": map_index,
                        "operation": name,
                        "repeat": repeat,
                        "max_abs": delta,
                    }
                max_forward_delta = max(max_forward_delta, delta)
                if delta > 1.0e-12 and first_forward_difference is None:
                    first_forward_difference = {
                        "map": map_index,
                        "operation": name,
                        "repeat": repeat,
                        "max_abs": delta,
                    }
                if "coefficients" not in name:
                    max_physical_delta = max(max_physical_delta, delta)
                    if (
                        largest_physical_difference is None
                        or delta > largest_physical_difference["max_abs"]
                    ):
                        largest_physical_difference = {
                            "map": map_index,
                            "operation": name,
                            "repeat": repeat,
                            "max_abs": delta,
                        }
                    if delta > 1.0e-12 and first_physical_difference is None:
                        first_physical_difference = {
                            "map": map_index,
                            "operation": name,
                            "repeat": repeat,
                            "max_abs": delta,
                        }

    gauge_records = []
    for map_index, (left, right) in enumerate(zip(traces[0], traces[1]), 1):
        channel, orbital, _ = left["minimum_occupied_pair"]
        coeff0 = left["coefficients"]
        coeff1 = right["coefficients"]
        if coeff0.ndim == 2:
            c0 = coeff0[:, orbital : orbital + 2]
            c1 = coeff1[:, orbital : orbital + 2]
        else:
            c0 = coeff0[channel, :, orbital : orbital + 2]
            c1 = coeff1[channel, :, orbital : orbital + 2]
        overlap = left["overlap"]
        principal_cosines = torch.linalg.svdvals(c0.mT @ overlap @ c1)
        principal_angles = torch.acos(principal_cosines.clamp(-1.0, 1.0))
        gauge_records.append(
            {
                "map": map_index,
                "pair": (channel, orbital),
                "coefficient_max_abs": float(
                    (left["coefficients"] - right["coefficients"]).abs().max()
                ),
                "occupied_pair_principal_angles": principal_angles.tolist(),
                "density_max_abs": float(
                    (left["density"] - right["density"]).abs().max()
                ),
                "hamiltonian_max_abs": float(
                    (left["hamiltonian"] - right["hamiltonian"]).abs().max()
                ),
                "energy_max_abs": float(
                    (left["emo"] - right["emo"]).abs().max()
                ),
            }
        )

    spectral_thirds = []
    for row in (trace[1] for trace in traces):
        h0 = row["hamiltonian"]
        l_inv = row["inverse_cholesky"]
        occupation = row["occupation"]
        density_probe = torch.linspace(
            -0.8, 1.1, row["density"].numel(), **_DD
        ).reshape_as(row["density"])
        direction = torch.linspace(-1.0, 1.0, h0.numel(), **_DD).reshape_as(h0)
        direction = 0.5 * (direction + direction.mT)

        def spectral_scalar(parameter: Tensor) -> Tensor:
            hamiltonian = h0 + parameter * direction
            _, coefficients = storch.eighb(
                a=hamiltonian,
                b=None,
                l_inv=l_inv,
                is_posdef=True,
                factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
                broadening_method=(
                    "lorn" if hamiltonian.requires_grad else None
                ),
            )
            density = get_density(coefficients, occupation)
            return (density * density_probe).sum()

        parameter = torch.zeros((), **_DD, requires_grad=True)
        first_derivative = torch.autograd.grad(
            spectral_scalar(parameter), parameter, create_graph=True
        )[0]
        second_derivative = torch.autograd.grad(
            first_derivative, parameter, create_graph=True
        )[0]
        third_derivative = torch.autograd.grad(second_derivative, parameter)[0]
        spectral_thirds.append(float(third_derivative.detach()))

    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_record: dict[str, object]
    try:
        torch.use_deterministic_algorithms(True)
        det_calc = _calculator(system, "gfn2", "pytorch")
        det_trace: list[dict[str, object]] = []
        _fixed_n_anderson_at(
            det_calc,
            system.positions,
            torch.zeros(3, **_DD),
            20,
            trajectory_trace=det_trace,
        )
        deterministic_record = {
            "supported": True,
            "max_map20_state_delta": float(
                (traces[0][-1]["next_current"] - det_trace[-1]["next_current"])
                .abs()
                .max()
            ),
        }
    except RuntimeError as exc:
        deterministic_record = {"supported": False, "error": str(exc)}
    finally:
        torch.use_deterministic_algorithms(deterministic)

    assert len({len(trace) for trace in traces}) == 1
    assert max_forward_delta >= 0.0
    assert all(np.isfinite(row["coefficient_max_abs"]) for row in gauge_records)
    assert all(
        max(row["occupied_pair_principal_angles"]) < 1.0e-6
        and row["density_max_abs"] < 1.0e-10
        and row["hamiltonian_max_abs"] < 1.0e-10
        and row["energy_max_abs"] < 1.0e-10
        for row in gauge_records
    )
    if deterministic_record["supported"]:
        assert deterministic_record["max_map20_state_delta"] < 1.0e-10
    print(
        {
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "fresh_endpoint_AAA": endpoint_values,
            "fresh_endpoint_spread": max(endpoint_values)
            - min(endpoint_values),
            "same_input_forward_first_divergence": first_forward_difference,
            "same_input_forward_first_bitwise_difference": first_bitwise_difference,
            "same_input_forward_max_abs_delta": max_forward_delta,
            "first_gauge_invariant_forward_difference": first_physical_difference,
            "largest_gauge_invariant_forward_difference": largest_physical_difference,
            "gauge_invariant_forward_max_abs_delta": max_physical_delta,
            "first_vs_second_spectral_gauge_by_map": gauge_records,
            "map2_frozen_spectral_thirds_by_trajectory": spectral_thirds,
            "map2_frozen_spectral_third_spread": max(spectral_thirds)
            - min(spectral_thirds),
            "deterministic_algorithms_probe": deterministic_record,
        }
    )


def test_no2_frozen_spectral_derivative_reproducibility() -> None:
    """Repeat one frozen suspicious H/S spectral derivative in one process."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    boundary_record: dict[str, object] = {}
    trajectory: list[dict[str, object]] = []
    _fixed_n_anderson_at(
        calculator,
        SYSTEMS["NO2"].positions,
        torch.zeros(3, **_DD),
        7,
        trajectory_trace=trajectory,
        boundary_record=boundary_record,
    )
    # Map 7 is where repeated center trajectories first show an occupied-pair
    # coefficient gauge flip while their physical spectral outputs agree.
    result = trajectory[6]
    inputs = boundary_record["inputs"]
    h0 = result["hamiltonian"].detach().clone()
    s0 = inputs.integrals.overlap.detach().clone()
    l_inv = inputs.inverse_cholesky.detach().clone()
    occupation = result["occupation"].detach()
    direction = torch.linspace(-1.0, 1.0, h0.numel(), **_DD).reshape_as(h0)
    direction = 0.5 * (direction + direction.mT)
    density_probe = torch.linspace(
        -0.8, 1.1, result["density"].numel(), **_DD
    ).reshape_as(result["density"])

    def spectral_scalar(parameter: Tensor) -> Tensor:
        hamiltonian = h0 + parameter * direction
        broadening = "lorn" if hamiltonian.requires_grad else None
        emo, coefficients = storch.eighb(
            a=hamiltonian,
            b=None,
            l_inv=l_inv,
            is_posdef=True,
            factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
            broadening_method=broadening,
        )
        density = get_density(coefficients, occupation)
        return (density * density_probe).sum() + (density * hamiltonian).sum()

    def derivatives() -> dict[str, float]:
        parameter = torch.zeros((), **_DD, requires_grad=True)
        value = spectral_scalar(parameter)
        first = torch.autograd.grad(value, parameter, create_graph=True)[0]
        second = torch.autograd.grad(first, parameter, create_graph=True)[0]
        third = torch.autograd.grad(second, parameter)[0]
        jvp = torch.func.jvp(
            spectral_scalar,
            (torch.zeros((), **_DD),),
            (torch.ones((), **_DD),),
        )[1]
        return {
            "value": float(value.detach()),
            "reverse_first": float(first.detach()),
            "jvp_first": float(jvp.detach()),
            "reverse_second": float(second.detach()),
            "reverse_third": float(third.detach()),
        }

    before = [derivatives() for _ in range(3)]
    unrelated_h = h0 + 0.05 * torch.eye(
        h0.shape[-1], dtype=h0.dtype, device=h0.device
    )
    storch.eighb(
        a=unrelated_h,
        b=None,
        l_inv=l_inv,
        is_posdef=True,
        factor=torch.finfo(h0.dtype).eps ** 0.5,
        broadening_method=None,
    )
    after = derivatives()
    deltas = {
        key: max(abs(row[key] - before[0][key]) for row in (*before[1:], after))
        for key in before[0]
    }
    assert torch.equal(h0, result["hamiltonian"].detach())
    assert torch.equal(s0, inputs.integrals.overlap.detach())
    assert max(deltas.values()) < 1.0e-10
    print(
        {
            "frozen_hs_exact": True,
            "frozen_map": result["map"],
            "h_norm": float(torch.linalg.vector_norm(h0)),
            "s_norm": float(torch.linalg.vector_norm(s0)),
            "same_process_spectral_repeats": before,
            "after_unrelated_eigensolve": after,
            "max_abs_derivative_deltas": deltas,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }
    )


def test_no2_frozen_anderson_transition_derivative_reproducibility() -> None:
    """Repeat the map-8 mixer solve on fixed incoming values/tangents."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    axis = torch.zeros(3, **_DD)
    axis[1] = 1.0

    def incoming(field: Tensor):
        record: dict[str, object] = {}
        _fixed_n_anderson_at(
            calculator,
            SYSTEMS["NO2"].positions,
            field * axis,
            8,
            boundary_record=record,
        )
        state = record["map8_mixer_state_before"]
        return (
            record["map8_state_before_transition"],
            record["map8_result"].mapped_x,
            state.x_hist,
            state.f_hist,
        )

    zero = torch.zeros((), **_DD)
    values, tangents = torch.func.jvp(
        incoming, (zero,), (torch.ones_like(zero),)
    )
    center_record: dict[str, object] = {}
    _fixed_n_anderson_at(
        calculator,
        SYSTEMS["NO2"].positions,
        torch.zeros(3, **_DD),
        8,
        boundary_record=center_record,
    )
    settings = center_record["settings"]

    def transition_projection(name: str):
        def evaluate(field: Tensor, _geometry: Tensor):
            perturbed = [
                value + field * tangent
                for value, tangent in zip(values, tangents)
            ]
            old, mapped, x_hist, f_hist = perturbed
            state = AndersonState(x_hist=x_hist, f_hist=f_hist)
            stages = _anderson_transition_stages(
                old, mapped, state, settings, step=8
            )
            value = stages[name]
            weights = torch.linspace(
                -1.0, 1.0, value.numel(), **_DD
            ).reshape_as(value)
            return (weights * value).sum() / value.numel() ** 0.5, 8

        return evaluate

    records = {}
    for stage in ("theta", "mixed"):
        fn = transition_projection(stage)
        first = _third_response(fn, (1.0e-5, 2.0e-6, 5.0e-7), check_jvp=True)
        torch.linalg.solve(
            torch.tensor([[2.0, 0.1], [0.1, 1.0]], **_DD),
            torch.ones(2, **_DD),
        )
        second = _third_response(fn, (1.0e-5, 2.0e-6, 5.0e-7), check_jvp=True)
        assert abs(first["reverse_third"] - second["reverse_third"]) < 1.0e-10
        records[stage] = {"first": first, "after_unrelated_solve": second}
    print(
        {
            "frozen_incoming_old_mapped_xhist_fhist": True,
            "field_tangent_norms": [
                float(torch.linalg.vector_norm(value)) for value in tangents
            ],
            "frozen_transition_derivatives": records,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }
    )


def test_no2_intermediate_trajectory_spectra_and_local_maps() -> None:
    """Inspect every center map and probe its smallest-gap neighborhood."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    trace: list[dict[str, object]] = []
    outcome, inputs, layout, _ = _functional_solve_at(
        calculator,
        SYSTEMS["NO2"].positions,
        torch.zeros(3, **_DD),
        eager=True,
        map_trace=trace,
    )
    assert len(trace) == int(outcome.iterations)
    gap_indices = [
        index
        for index, record in enumerate(trace)
        if record["minimum_frontier_gap"] is not None
    ]
    smallest = min(
        gap_indices,
        key=lambda index: trace[index]["minimum_occupied_pair_gap"],
    )
    selected_indices = sorted(
        {
            max(0, smallest - 1),
            max(0, smallest - 2),
            smallest,
            min(len(trace) - 1, smallest + 1),
            len(trace) - 1,
        }
    )

    local_results = []
    states = []
    density_probe = torch.linspace(
        -0.9,
        1.2,
        outcome.map_result.spectral.density.numel(),
        **_DD,
    ).reshape_as(outcome.map_result.spectral.density)
    for index in selected_indices:
        record = trace[index]
        state = record["input_state_tensor"]
        states.append(state)

        def field_energy(field_value: Tensor, displacement: Tensor):
            return _raw_map_energy_at_state(
                calculator, state, field_value, displacement
            )

        response = _third_response(field_energy, (1.0e-4, 5.0e-5, 2.0e-5))

        def density_energy(field_value: Tensor, displacement: Tensor):
            return _two_map_density_probe_at_state(
                calculator, state, field_value, displacement, density_probe
            )

        density_response = _third_response(
            density_energy,
            (1.0e-4, 5.0e-5, 2.0e-5),
            convention=1.0,
        )

        def geometry_density(displacement: Tensor, _field_value: Tensor):
            return _raw_map_density_probe_at_state(
                calculator,
                state,
                torch.zeros((), **_DD),
                displacement,
                density_probe,
            )

        geometry_response = _third_response(
            geometry_density,
            (1.0e-4, 5.0e-5, 2.0e-5),
            convention=1.0,
        )
        geometry_density_response = _third_response(
            lambda displacement, _field: _two_map_density_probe_at_state(
                calculator,
                state,
                torch.zeros((), **_DD),
                displacement,
                density_probe,
            ),
            (1.0e-4, 5.0e-5, 2.0e-5),
            convention=1.0,
        )
        local_results.append(
            {
                "iteration": index,
                "residual_l2": record["residual_l2"],
                "minimum_adjacent_gap": record["minimum_adjacent_gap"],
                "minimum_occupied_pair": record["minimum_occupied_pair"],
                "minimum_occupied_pair_gap": record[
                    "minimum_occupied_pair_gap"
                ],
                "minimum_frontier_gap": record["minimum_frontier_gap"],
                "max_fractional_occupation": record[
                    "max_fractional_occupation"
                ],
                "fermi_changes_materially": record["fermi_changes_materially"],
                "orbital_energies": record["orbital_energies"],
                "occupations": record["occupations"],
                "field_raw_map_response": response,
                "geometry_raw_map_density_response": geometry_response,
                "field_two_map_density_response": density_response,
                "geometry_two_map_density_response": geometry_density_response,
                "mixed_raw_map_density_response": (
                    _mixed_response(density_energy, 5.0e-5)
                    if index == smallest
                    else None
                ),
            }
        )

    for local in local_results:
        for name, tolerance in (
            ("field_raw_map_response", 1.0e-3),
            ("geometry_raw_map_density_response", 1.0e-3),
            ("field_two_map_density_response", 1.0e-1),
            ("geometry_two_map_density_response", 1.0e-2),
        ):
            probe = local[name]
            fd_values = [
                float(row["fd_of_second"])
                for row in probe["fd_of_second_sweep"]
            ]
            assert max(fd_values) - min(fd_values) < tolerance
            assert (
                abs(float(probe["reverse_third"]) - float(np.median(fd_values)))
                < tolerance
            )

    # Compare the ordinary forward and derivative-enabled (Lorentzian)
    # spectral branches at the smallest-gap map, using gauge-invariant values.
    index = smallest
    map_result = potential_map(
        states[selected_indices.index(index)], inputs, layout
    )
    hamiltonian = map_result.spectral.hamiltonian.detach()
    ordinary = evaluate_hamiltonian(hamiltonian, inputs)
    lorentzian = evaluate_hamiltonian(
        hamiltonian.clone().requires_grad_(True), inputs
    )
    branch_difference = {
        name: float(
            (getattr(ordinary, name) - getattr(lorentzian, name).detach())
            .abs()
            .max()
        )
        for name in ("density", "emo", "occupation")
    }
    branch_difference["ordinary_path"] = "broadening_method=None"
    branch_difference["derivative_path"] = "broadening_method=lorn"
    assert (
        max(
            branch_difference[name] for name in ("density", "emo", "occupation")
        )
        < 1.0e-12
    )
    print(
        {
            "map_count": len(trace),
            "selected_indices": selected_indices,
            "smallest_gap_index": smallest,
            "trajectory": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "input_state_tensor"
                }
                for item in trace
            ],
            "local_raw_maps": local_results,
            "forward_vs_lorentzian_values": branch_difference,
        }
    )


@pytest.mark.parametrize(
    "tolerance", (1.0e-8, 1.0e-10, 1.0e-12, 1.0e-13, 1.0e-14)
)
def test_no2_solver_tolerance_sweep(tolerance: float) -> None:
    """Record response sensitivity to the paired SCF residual tolerances."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    center_record: dict[str, object] = {}

    def energy_fn(field_value: Tensor, displacement: Tensor):
        return _trajectory_energy_at(
            calculator,
            field_value,
            displacement,
            eager=False,
            maxiter=None,
            stage="solver",
            tolerance=tolerance,
            solver_record=(
                center_record
                if float(field_value.detach()) == 0.0
                and float(displacement.detach()) == 0.0
                else None
            ),
        )

    response = _third_response(energy_fn, (1.0e-4, 5.0e-5, 2.0e-5))
    assert all(
        np.isfinite(value)
        for value in (
            response["reverse_first"],
            response["reverse_second"],
            response["reverse_third"],
        )
    )
    nested_third = float(response["reverse_third"])
    fd_third_values = [
        float(row["fd_of_second"]) for row in response["fd_of_second_sweep"]
    ]
    fd_third = float(np.median(fd_third_values))
    assert max(fd_third_values) - min(fd_third_values) < 5.0e-4
    expected_iterations = {
        1.0e-8: 17,
        1.0e-10: 20,
        1.0e-12: 24,
        1.0e-13: 25,
    }
    if tolerance == 1.0e-14:
        assert 28 <= center_record["iterations"] <= 40
    else:
        assert center_record["iterations"] == expected_iterations[tolerance]
    if tolerance == 1.0e-8:
        assert abs(nested_third - fd_third) > 1.0e3
    elif tolerance == 1.0e-10:
        assert abs(nested_third - fd_third) > 1.0
    else:
        assert abs(nested_third - fd_third) < 1.0e-3
    if tolerance in (1.0e-10, 1.0e-12):
        middle_step = response["fd_of_second_sweep"][1]
        assert (
            middle_step["iterations_plus"]
            == middle_step["iterations_minus"]
            == response["iterations_center"]
        )
    print(
        {
            "x_atol_and_x_atol_max": tolerance,
            "center_iterations": center_record["iterations"],
            "center_converged": center_record["converged"],
            "center_residual_max_abs": center_record["residual_max_abs"],
            **response,
        }
    )


@pytest.mark.parametrize("eager", (True, False), ids=("eager", "fixed-trip"))
@pytest.mark.parametrize("stage", ("solver", "finalized"))
def test_no2_gfn2_solver_hyper_boundary(eager: bool, stage: str) -> None:
    """Compare one solver stage's hyper-response on a fixed local branch."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    maxiter = None if eager else 40

    def energy_fn(field_value: Tensor, displacement: Tensor):
        return _trajectory_energy_at(
            calculator,
            field_value,
            displacement,
            eager=eager,
            maxiter=maxiter,
            stage=stage,
        )

    result = _third_response(energy_fn, 5.0e-5, check_jvp=not eager)
    assert result["iterations_center"] == 20
    assert all(
        row["iterations_plus"] == row["iterations_minus"] == 20
        for row in result["fd_of_second_sweep"]
    )
    fd = result["fd_of_second_sweep"][0]
    assert abs(float(result["reverse_first"]) - float(fd["fd_first"])) < 1.0e-4
    assert (
        abs(float(result["reverse_second"]) - float(fd["fd_second"])) < 2.0e-3
    )
    assert abs(float(result["reverse_third"]) - float(fd["fd_of_second"])) > 1.0
    print(
        {
            "control": "eager" if eager else "fixed-trip",
            "stage": stage,
            **result,
        }
    )


@pytest.mark.parametrize("eager", (True, False), ids=("eager", "fixed-trip"))
@pytest.mark.parametrize("stage", ("solver", "finalized"))
def test_no2_gfn2_solver_mixed_boundary(eager: bool, stage: str) -> None:
    """Compare mixed response by differencing field polarizability locally."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    maxiter = None if eager else 40

    def energy_fn(field_value: Tensor, displacement: Tensor):
        return _trajectory_energy_at(
            calculator,
            field_value,
            displacement,
            eager=eager,
            maxiter=maxiter,
            stage=stage,
        )

    result = _mixed_response(energy_fn, 5.0e-5)
    assert (
        result["iterations_center"]
        == result["iterations_plus"]
        == result["iterations_minus"]
        == 20
    )
    assert (
        abs(
            float(result["nested_mixed_third"])
            - float(result["fd_of_field_second"])
        )
        > 0.1
    )
    print(
        {
            "control": "eager" if eager else "fixed-trip",
            "stage": stage,
            **result,
        }
    )


def test_no2_eager_fixed_trip_center_parity() -> None:
    """Assert eager and transform-fixed controllers agree at the center."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    records = {}
    for eager, maxiter in ((True, None), (False, 40)):
        result, inputs, _, finalized = _functional_solve_at(
            calculator,
            SYSTEMS["NO2"].positions,
            torch.zeros(3, **_DD),
            eager=eager,
            maxiter=maxiter,
        )
        records["eager" if eager else "fixed-trip"] = {
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "solver_energy": float(_selected_map_energy(result, inputs)),
            "finalized_energy": float(
                (finalized.energy + finalized.fenergy).sum()
            ),
            "map_result": result.map_result,
        }
    eager = records["eager"]
    fixed = records["fixed-trip"]
    assert eager["iterations"] == fixed["iterations"] == 20
    assert eager["converged"] and fixed["converged"]
    for name in ("solver_energy", "finalized_energy"):
        assert abs(eager[name] - fixed[name]) < 1.0e-12
    left = eager["map_result"]
    right = fixed["map_result"]
    for name in ("mapped_x", "hamiltonian", "density", "emo", "occupation"):
        assert torch.allclose(
            getattr(left, name),
            getattr(right, name),
            atol=1.0e-12,
            rtol=1.0e-12,
        ), name
    print(
        {
            key: {k: v for k, v in value.items() if k != "map_result"}
            for key, value in records.items()
        }
    )


def test_no2_fixed_trip_cap_matches_300_center() -> None:
    """Keep the bounded diagnostic cap equivalent to the baseline at center."""
    calculator = _calculator(SYSTEMS["NO2"], "gfn2", "pytorch")
    results = []
    for maxiter in (None, 40):
        outcome, inputs, _, finalized = _functional_solve_at(
            calculator,
            SYSTEMS["NO2"].positions,
            torch.zeros(3, **_DD),
            eager=False,
            maxiter=maxiter,
        )
        results.append((outcome, inputs, finalized))
    full, capped = results
    assert int(full[0].iterations) == int(capped[0].iterations) == 20
    assert torch.equal(full[0].converged, capped[0].converged)
    assert torch.allclose(
        full[0].map_result.mapped_x,
        capped[0].map_result.mapped_x,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.allclose(
        (full[2].energy + full[2].fenergy).sum(),
        (capped[2].energy + capped[2].fenergy).sum(),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_no2_gfn2_case_a_and_production_route_agree() -> None:
    """Reproduce explicit-Anderson and configured-Broyden route parity."""
    reference_system = SYSTEMS["NO2"]
    options = {
        **SCF_OPTIONS,
        "int_driver": "pytorch",
        "batch_mode": reference_system.batch_mode,
    }
    positions = reference_system.positions.clone()
    field = torch.zeros(3, **_DD)
    # Case A is the ordinary T0.2 path, including one fresh Calculator per
    # quantity. Keep its returned endpoints as the comparison source rather
    # than evaluating the derivatives again on a reused object.
    case_a_t02 = compute(reference_system, "gfn2", "pytorch", ("third",))
    hyper_a = float(case_a_t02["hyperpolarizability"][_HYPER_INDEX])
    pol_a = float(case_a_t02["pol_deriv"][_POL_INDEX])
    case_a = _calculator(reference_system, "gfn2", "pytorch")
    result_a = case_a.singlepoint(
        positions.clone(), reference_system.charge, reference_system.spin, field
    )
    # Case B changes only mixer/strict routing after historical Case A is done.
    field_b = torch.zeros(3, **_DD, requires_grad=True)
    case_b = Calculator(
        reference_system.numbers,
        GFN2_XTB,
        opts={**options, "mixer": "broyden", "strict": False},
        interaction=[ElectricField(field_b)],
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
    result_b = case_b.system.singlepoint(
        positions.clone(), reference_system.charge, reference_system.spin, field
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

    # Compare route B endpoints with fresh calculators, without gauge-sensitive
    # coefficient comparisons.
    hyper_b_field = torch.zeros(3, **_DD, requires_grad=True)
    hyper_b_calculator = Calculator(
        reference_system.numbers,
        GFN2_XTB,
        opts={**options, "mixer": "broyden", "strict": False},
        interaction=[ElectricField(hyper_b_field)],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    pol_b_field = torch.zeros(3, **_DD, requires_grad=True)
    pol_b_calculator = Calculator(
        reference_system.numbers,
        GFN2_XTB,
        opts={**options, "mixer": "broyden", "strict": False},
        interaction=[ElectricField(pol_b_field)],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    hyper_b = hyper_b_calculator.hyperpolarizability(
        positions.clone(), reference_system.charge, reference_system.spin
    )[_HYPER_INDEX]
    pos_b = reference_system.positions.clone().requires_grad_()
    pol_b = pol_b_calculator.pol_deriv(
        pos_b, reference_system.charge, reference_system.spin
    )[_POL_INDEX]
    assert np.isfinite(hyper_a) and torch.isfinite(hyper_b)
    assert np.isfinite(pol_a) and torch.isfinite(pol_b)
    hyper_delta = float(hyper_a - hyper_b.detach())
    pol_delta = float(pol_a - pol_b.detach())
    assert abs(hyper_delta) < 5.0e-4 * max(
        abs(hyper_a), abs(float(hyper_b.detach()))
    )
    assert abs(pol_delta) < 5.0e-4 * max(abs(pol_a), abs(float(pol_b.detach())))
    print(
        {
            "case_a_iterations": int(result_a.iterations),
            "case_b_iterations": int(result_b.iterations),
            "case_a_energy": float(result_a.energy.sum()),
            "case_b_energy": float(result_b.energy.sum()),
            "case_a_hyperpolarizability": hyper_a,
            "case_b_hyperpolarizability": float(hyper_b.detach()),
            "hyperpolarizability_delta": hyper_delta,
            "case_a_pol_deriv": pol_a,
            "case_b_pol_deriv": float(pol_b.detach()),
            "pol_deriv_delta": pol_delta,
        }
    )
