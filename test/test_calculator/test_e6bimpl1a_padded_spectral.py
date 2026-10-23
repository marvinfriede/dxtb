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
"""Padded GFN1 spectral input and evaluation acceptance."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tad_mctc import storch
from tad_mctc.tree import stack
from tad_mctc.units.energy import KELVIN2AU
from torch.utils._pytree import tree_flatten

from dxtb import GFN1_XTB, ParamModule, SCFFailureCode, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.padded_integrals import evaluate_padded_integrals
from dxtb._src.calculators.result import Result
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional.core import (
    CoreIndexData,
    FunctionalInteractionSet,
)
from dxtb._src.scf.functional.data import (
    FermiSettings,
    SCFInputs,
    _single_system_layout,
)
from dxtb._src.scf.functional.map import evaluate_hamiltonian
from dxtb._src.scf.functional.padded import (
    PADDED_ORBITAL_ENERGY,
    PaddedSCFSpectralResult,
    evaluate_padded_fixed_trip,
    evaluate_padded_scf_spectrum,
    evaluate_padded_systems,
    validate_padded_charge_spin,
)
from dxtb._src.scf.iterator import get_refocc

DTYPE = torch.float64


def _model(
    *,
    differentiable_h0: bool = False,
    differentiable_es2: bool = False,
    mixer: int | None = None,
    maxiter: int | None = None,
    scp_mode: int | None = None,
    int_level: int = labels.INTLEVEL_HCORE,
    differentiable_d3: tuple[str, ...] = (),
    fermi_etemp: float = 0.0,
    fermi_maxiter: int = 200,
    fermi_thresh: float | None = None,
) -> Model:
    parameters = ParamModule(GFN1_XTB, dtype=DTYPE)
    if differentiable_h0:
        parameters.set_differentiable("hamiltonian", "xtb", "kpol")
    if differentiable_es2:
        parameters.set_differentiable("element.H.gam")
    for name in differentiable_d3:
        parameters.set_differentiable(f"dispersion.d3.{name}")
    config_options = {
        "method": labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": int_level,
        "fermi_etemp": fermi_etemp,
        "fermi_maxiter": fermi_maxiter,
        "fermi_thresh": fermi_thresh,
    }
    if mixer is not None:
        config_options["mixer"] = mixer
    if maxiter is not None:
        config_options["maxiter"] = maxiter
    if scp_mode is not None:
        config_options["scp_mode"] = scp_mode
    return Model(
        par=parameters,
        config=Config.create(**config_options),
        auto_int_level=False,
    )


def _positions(numbers: tuple[int, ...]) -> torch.Tensor:
    coordinates = {
        (1, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        (1, 1, 1): [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [-0.3, 0.8, 0.7],
        ],
        (8, 1, 1): [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.6],
            [-0.8, 0.0, 0.6],
        ],
        (6, 1, 1, 1, 1): [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
    }
    return torch.tensor(coordinates[numbers], dtype=DTYPE)


def _systems(
    numbers: tuple[tuple[int, ...], ...],
    *,
    differentiable=False,
    differentiable_es2=False,
    mixer=None,
    maxiter=None,
    scp_mode=None,
    int_level: int = labels.INTLEVEL_HCORE,
    differentiable_d3=(),
    fermi_etemp: float = 0.0,
    fermi_maxiter: int = 200,
    fermi_thresh: float | None = None,
):
    model = _model(
        differentiable_h0=differentiable,
        differentiable_es2=differentiable_es2,
        mixer=mixer,
        maxiter=maxiter,
        scp_mode=scp_mode,
        int_level=int_level,
        differentiable_d3=differentiable_d3,
        fermi_etemp=fermi_etemp,
        fermi_maxiter=fermi_maxiter,
        fermi_thresh=fermi_thresh,
    )
    systems = tuple(
        model.setup(torch.tensor(entry, dtype=torch.int64)) for entry in numbers
    )
    assert all(system.exact_core is not None for system in systems)
    return model, systems


def _prepare(systems, source_positions, charges, spins):
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, index)
        for index, core in enumerate(cores)
    )
    for core, charge, spin in zip(padded, charges, spins):
        validate_padded_charge_spin(core, charge, spin)
    canonical_positions = tuple(
        canonicalize_positions(position, bucket, index)
        for index, position in enumerate(source_positions)
    )
    return bucket, padded, canonical_positions, charges, spins


def _padded_lane(core, position, charge, spin):
    return evaluate_padded_scf_spectrum(core, position, charge, spin)


def _serial_h0_spectrum(system, position, charge, spin):
    assert system.integral_setup is not None and system.h0_setup is not None
    matrices, _, _ = build_integral_matrices(
        system.integral_setup, system.h0_setup, position, charge
    )
    n0, occupation, nel = get_refocc(
        system.h0_setup.refocc,
        charge,
        None if spin is None else spin.reshape(1),
        system.ihelp,
    )
    inverse = storch.inv_cholesky_factor(matrices.overlap, is_posdef=True)
    inputs = SCFInputs(
        numbers=system.numbers,
        index=CoreIndexData(
            shells_to_atom=system.ihelp.shells_to_atom.clone(),
            orbitals_to_shell=system.ihelp.orbitals_to_shell.clone(),
        ),
        integrals=matrices,
        n0=n0,
        nel=nel,
        occupation=occupation,
        interactions=FunctionalInteractionSet(entries=()),
        charge_layout=_single_system_layout(system.numbers, matrices),
        inverse_cholesky=inverse,
        orbital_mask=torch.ones(
            matrices.hcore.shape[-1],
            dtype=torch.bool,
            device=matrices.hcore.device,
        ),
        fermi_settings=_serial_fermi_settings(system, matrices.hcore),
    )
    return inputs, evaluate_hamiltonian(matrices.hcore, inputs)


def _serial_fermi_settings(system, hcore):
    config = system.config.scf.fermi
    kt = config.etemp * KELVIN2AU
    if kt == 0.0:
        return None
    threshold = (
        hcore.new_tensor(
            min(
                torch.finfo(hcore.dtype).eps ** 0.5,
                1.0e5 * torch.finfo(hcore.dtype).eps,
                1.0e-4,
            )
        )
        if config.thresh is None
        else hcore.new_tensor(config.thresh)
    )
    return FermiSettings(
        kt=hcore.new_tensor(kt),
        smear=kt >= 3.0e-7,
        maxiter=config.maxiter,
        threshold=threshold,
        diff_order=config.diff_order,
        partition=config.partition,
    )


@pytest.mark.parametrize(
    ("numbers", "charges", "spins"),
    [
        (((1, 1), (1, 1, 1)), (0.0, 1.0), (0.0, 0.0)),
        (((1, 1), (8, 1, 1)), (1.0, 0.0), (1.0, 0.0)),
        (((1, 1), (8, 1, 1)), (0.0, 0.0), (0.0, 0.0)),
        (((8, 1, 1), (1, 1)), (0.0, 0.0), (0.0, 0.0)),
        (((8, 1, 1), (6, 1, 1, 1, 1)), (0.0, 0.0), (0.0, 0.0)),
        (((6, 1, 1, 1, 1), (8, 1, 1)), (0.0, 0.0), (0.0, 0.0)),
    ],
    ids=[
        "h2-h3-plus",
        "h2-plus-water",
        "h2-water",
        "water-h2",
        "water-ch4",
        "ch4-water",
    ],
)
def test_padded_mapped_spectrum_matches_independent_serial_systems(
    numbers, charges, spins
):
    """Mapped padded spectra retain independent physical electronic blocks."""
    _, systems = _systems(numbers)
    positions = tuple(_positions(entry) for entry in numbers)
    charge_values = torch.tensor(charges, dtype=DTYPE)
    spin_values = torch.tensor(spins, dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charge_values, spin_values
    )

    mapped = torch.func.vmap(_padded_lane)(
        stack(padded),
        torch.stack(canonical_positions),
        charge_values,
        spin_values,
    )
    assert isinstance(mapped, PaddedSCFSpectralResult)
    assert mapped.density.shape == (
        len(systems),
        bucket.orbitals,
        bucket.orbitals,
    )

    for index, (system, position, charge, spin) in enumerate(
        zip(systems, positions, charge_values, spin_values)
    ):
        serial_inputs, serial = _serial_h0_spectrum(
            system, position, charge, spin
        )
        mapping = dict(bucket.source_maps)[index]
        ao = torch.tensor(mapping.orbitals, dtype=torch.int64)
        torch.testing.assert_close(
            mapped.hamiltonian[index].index_select(-2, ao).index_select(-1, ao),
            serial.hamiltonian,
            atol=2e-10,
            rtol=2e-10,
        )
        torch.testing.assert_close(
            mapped.density[index].index_select(-2, ao).index_select(-1, ao),
            serial.density,
            atol=2e-8,
            rtol=2e-8,
        )
        torch.testing.assert_close(
            mapped.charges[index].index_select(-1, ao),
            serial.charges.mono,
            atol=2e-8,
            rtol=2e-8,
        )
        torch.testing.assert_close(
            mapped.occupation[index].sum(-1), serial.occupation.sum(-1)
        )
        torch.testing.assert_close(
            mapped.emo[index, : system.ihelp.nao],
            serial.emo,
            atol=2e-8,
            rtol=2e-8,
        )
        if tuple(numbers[index]) == (1, 1):
            assert torch.any(serial.emo > 0.0)
            assert torch.all(
                mapped.emo[index, system.ihelp.nao :] > serial.emo.max() + 100.0
            )
        assert (
            torch.count_nonzero(
                mapped.density[index, ~padded[index].validity.orbitals]
            )
            == 0
        )
        assert torch.all(
            mapped.emo[index, system.ihelp.nao :] > 0.5 * PADDED_ORBITAL_ENERGY
        )
        torch.testing.assert_close(
            mapped.nel[index], serial_inputs.nel, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(
            mapped.n0[index].index_select(-1, ao),
            serial_inputs.n0,
            atol=1e-12,
            rtol=1e-12,
        )
        assert mapped.orbital_mask[index].dtype == torch.bool
        assert torch.equal(
            mapped.orbital_mask[index], padded[index].validity.orbitals
        )
        expected_mask = torch.zeros(bucket.orbitals, dtype=torch.bool)
        expected_mask[ao] = True
        assert torch.equal(mapped.orbital_mask[index], expected_mask)
        if numbers[index] == (8, 1, 1) and any(
            6 in source for source in numbers
        ):
            assert torch.any(
                mapped.orbital_mask[index, 1:]
                & ~mapped.orbital_mask[index, :-1]
            )
        assert (
            torch.count_nonzero(
                mapped.occupation[index, :, ~padded[index].validity.orbitals]
            )
            == 0
        )
        torch.testing.assert_close(
            mapped.charges[index].sum(), charge, atol=2e-8, rtol=2e-8
        )


def test_padded_overlap_is_positive_definite_and_physical_factor_is_unchanged():
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers)
    positions = tuple(_positions(entry) for entry in numbers)
    charges = torch.zeros(2, dtype=DTYPE)
    spins = torch.zeros(2, dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charges, spins
    )
    mapped_integrals = torch.func.vmap(evaluate_padded_integrals)(
        stack(padded), torch.stack(canonical_positions), charges
    )

    for index, (system, position) in enumerate(zip(systems, positions)):
        mapping = dict(bucket.source_maps)[index]
        ao = torch.tensor(mapping.orbitals, dtype=torch.int64)
        safe_overlap = mapped_integrals.overlap[index] + torch.diag_embed(
            (~padded[index].validity.orbitals).to(DTYPE)
        )
        torch.linalg.cholesky(safe_overlap)
        padded_inverse = storch.inv_cholesky_factor(
            safe_overlap, is_posdef=True
        )
        serial_integrals, _, _ = build_integral_matrices(
            system.integral_setup,
            system.h0_setup,
            position,
            torch.zeros((), dtype=DTYPE),
        )
        serial_inverse = storch.inv_cholesky_factor(
            serial_integrals.overlap.index_select(
                -2,
                torch.tensor(
                    sorted(
                        range(len(mapping.orbitals)),
                        key=lambda source: mapping.orbitals[source],
                    ),
                    dtype=torch.int64,
                ),
            ).index_select(
                -1,
                torch.tensor(
                    sorted(
                        range(len(mapping.orbitals)),
                        key=lambda source: mapping.orbitals[source],
                    ),
                    dtype=torch.int64,
                ),
            ),
            is_posdef=True,
        )
        canonical_ao = torch.tensor(sorted(mapping.orbitals), dtype=torch.int64)
        torch.testing.assert_close(
            padded_inverse.index_select(-2, canonical_ao).index_select(
                -1, canonical_ao
            ),
            serial_inverse,
            atol=2e-10,
            rtol=2e-10,
        )


def test_padded_coordinate_perturbations_do_not_change_spectrum():
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers)
    positions = tuple(_positions(entry) for entry in numbers)
    charges = torch.zeros(2, dtype=DTYPE)
    spins = torch.zeros(2, dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charges, spins
    )
    stacked_core = stack(padded)
    original = torch.stack(canonical_positions)
    perturbed = original.clone()
    for index, padded_system in enumerate(padded):
        inactive_atoms = ~padded_system.validity.atoms
        perturbed[index, inactive_atoms] = torch.tensor(
            [1000.0, -1000.0, 1000.0], dtype=DTYPE
        )

    reference = torch.func.vmap(_padded_lane)(
        stacked_core, original, charges, spins
    )
    changed = torch.func.vmap(_padded_lane)(
        stacked_core, perturbed, charges, spins
    )
    torch.testing.assert_close(changed.density, reference.density)
    torch.testing.assert_close(changed.emo, reference.emo)

    coincident = original.clone()
    for index, padded_system in enumerate(padded):
        inactive_atoms = ~padded_system.validity.atoms
        coincident[index, inactive_atoms] = original[index, 0]
    coincident_result = torch.func.vmap(_padded_lane)(
        stacked_core, coincident, charges, spins
    )
    torch.testing.assert_close(coincident_result.density, reference.density)
    torch.testing.assert_close(coincident_result.emo, reference.emo)


def test_finite_temperature_padded_fermi_status_matches_serial():
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers, fermi_etemp=300.0)
    positions = tuple(_positions(entry) for entry in numbers)
    charges = torch.zeros(2, dtype=DTYPE)
    spins = torch.zeros(2, dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charges, spins
    )
    mapped = torch.func.vmap(_padded_lane)(
        stack(padded), torch.stack(canonical_positions), charges, spins
    )
    assert mapped.fermi_converged is not None
    assert mapped.fermi_electron_count_ok is not None
    assert mapped.fermi_residual is not None
    assert mapped.fermi_converged.dtype == torch.bool
    assert mapped.fermi_electron_count_ok.dtype == torch.bool
    assert torch.all(mapped.fermi_converged)
    assert torch.all(mapped.fermi_electron_count_ok)

    for index, (system, position) in enumerate(zip(systems, positions)):
        _, serial = _serial_h0_spectrum(
            system,
            position,
            torch.zeros((), dtype=DTYPE),
            torch.zeros((), dtype=DTYPE),
        )
        mapping = dict(bucket.source_maps)[index]
        ao = torch.tensor(mapping.orbitals, dtype=torch.int64)
        torch.testing.assert_close(
            mapped.density[index].index_select(-2, ao).index_select(-1, ao),
            serial.density,
            atol=2e-8,
            rtol=2e-8,
        )
        torch.testing.assert_close(
            mapped.fermi_converged[index], serial.fermi_status.converged
        )
        torch.testing.assert_close(
            mapped.fermi_electron_count_ok[index],
            serial.fermi_status.electron_count_ok,
        )
        torch.testing.assert_close(
            mapped.fermi_residual[index], serial.fermi_status.residual
        )


def test_too_many_electrons_status_is_lane_local():
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers, fermi_etemp=300.0)
    positions = tuple(_positions(entry) for entry in numbers)
    charges = torch.tensor([-7.0, 0.0], dtype=DTYPE)
    spins = torch.tensor([1.0, 0.0], dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charges, spins
    )
    mapped = torch.func.vmap(_padded_lane)(
        stack(padded), torch.stack(canonical_positions), charges, spins
    )
    assert mapped.fermi_too_many_electrons is not None
    assert mapped.fermi_electron_count_ok is not None
    assert mapped.fermi_too_many_electrons.shape == (2, 2)
    torch.testing.assert_close(
        mapped.fermi_too_many_electrons[0], torch.tensor([True, False])
    )
    torch.testing.assert_close(
        mapped.fermi_too_many_electrons[1], torch.tensor([False, False])
    )
    torch.testing.assert_close(
        mapped.fermi_electron_count_ok[0], torch.tensor([False, True])
    )
    torch.testing.assert_close(
        mapped.fermi_electron_count_ok[1], torch.tensor([True, True])
    )

    for index, (system, position, charge, spin) in enumerate(
        zip(systems, positions, charges, spins)
    ):
        _, serial = _serial_h0_spectrum(system, position, charge, spin)
        torch.testing.assert_close(
            mapped.fermi_too_many_electrons[index],
            serial.fermi_status.too_many_electrons,
        )


@pytest.mark.parametrize(
    ("numbers", "charges", "spins"),
    [
        (((1, 1), (1, 1, 1)), (0.0, 1.0), (0.0, 0.0)),
        (((1, 1), (1, 1, 1)), (0.0, 0.0), (0.0, 1.0)),
        (((1, 1), (8, 1, 1)), (0.0, 0.0), (0.0, 0.0)),
        (((8, 1, 1), (1, 1)), (0.0, 0.0), (0.0, 0.0)),
        (((8, 1, 1), (6, 1, 1, 1, 1)), (0.0, 0.0), (0.0, 0.0)),
    ],
    ids=(
        "h2-h3-plus",
        "h2-h3-open-shell",
        "h2-water",
        "water-h2",
        "water-ch4",
    ),
)
def test_padded_full_fixed_trip_result_matches_serial_systems(
    numbers: tuple[tuple[int, ...], ...],
    charges: tuple[float, ...],
    spins: tuple[float, ...],
) -> None:
    """The mapped public Result tree retains full fixed-trip GFN1 values."""
    _, systems = _systems(numbers, mixer=labels.MIXER_LINEAR, maxiter=4)
    source_positions = tuple(_positions(item) for item in numbers)
    charge_values = tuple(torch.tensor(item, dtype=DTYPE) for item in charges)
    spin_values = tuple(torch.tensor(item, dtype=DTYPE) for item in spins)
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charge_values),
        torch.stack(spin_values),
    )
    if numbers == ((8, 1, 1), (1, 1)):
        source_ids = (19, 4)
        source_maps = dict(bucket.source_maps)
        bucket = replace(
            bucket,
            indices=source_ids,
            source_maps=tuple(
                (source_id, source_maps[lane_index])
                for lane_index, source_id in enumerate(source_ids)
            ),
        )
    elif numbers == ((1, 1), (8, 1, 1)):
        source_ids = (4, 19)
        source_maps = dict(bucket.source_maps)
        bucket = replace(
            bucket,
            indices=source_ids,
            source_maps=tuple(
                (source_id, source_maps[lane_index])
                for lane_index, source_id in enumerate(source_ids)
            ),
        )
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, charge_values, spin_values
    )
    assert isinstance(mapped, Result)
    assert len(decoded) == len(systems)
    for leaf in tree_flatten(mapped)[0]:
        if isinstance(leaf, torch.Tensor):
            assert leaf.ndim >= 1
            assert leaf.shape[0] == len(systems)

    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charge_values, spin_values
        )
    )
    for mapped_result, serial_result in zip(decoded, serial):
        assert tree_flatten(mapped_result)[1] == tree_flatten(serial_result)[1]
        for name in (
            "energy",
            "scf",
            "fenergy",
            "density",
            "emo",
            "occupation",
            "hamiltonian",
            "overlap",
            "hcore",
            "overlap_norm",
            "converged",
            "residual",
            "iterations",
        ):
            actual = getattr(mapped_result, name)
            expected = getattr(serial_result, name)
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=3e-8,
                    rtol=3e-8,
                    msg=f"Result field {name} differs",
                )
        assert mapped_result.coefficients is not None
        assert serial_result.coefficients is not None
        assert (
            mapped_result.coefficients.shape == serial_result.coefficients.shape
        )
        assert (
            mapped_result.coefficients.dtype == serial_result.coefficients.dtype
        )
        assert mapped_result.status is not None
        assert serial_result.status is not None
        torch.testing.assert_close(
            mapped_result.status.trajectory_failure,
            serial_result.status.trajectory_failure,
        )
        torch.testing.assert_close(
            mapped_result.status.final_failure,
            serial_result.status.final_failure,
        )
        assert tuple(label for label, _ in mapped_result.classical) == tuple(
            label for label, _ in serial_result.classical
        )
        for (_, actual), (_, expected) in zip(
            mapped_result.classical, serial_result.classical
        ):
            torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
        assert mapped_result.charges is not None
        assert serial_result.charges is not None
        torch.testing.assert_close(
            mapped_result.charges.mono,
            serial_result.charges.mono,
            atol=3e-8,
            rtol=3e-8,
        )
        assert mapped_result.potential is not None
        assert serial_result.potential is not None
        if mapped_result.potential is not None:
            assert serial_result.potential is not None
            torch.testing.assert_close(
                mapped_result.potential.mono,
                serial_result.potential.mono,
                atol=3e-8,
                rtol=3e-8,
            )


@pytest.mark.parametrize(
    ("numbers", "charges", "spins", "mixer", "fermi_etemp"),
    [
        (
            ((1, 1), (1, 1, 1)),
            (0.0, 1.0),
            (0.0, 0.0),
            labels.MIXER_LINEAR,
            0.0,
        ),
        (
            ((1, 1), (8, 1, 1)),
            (0.0, 0.0),
            (0.0, 0.0),
            labels.MIXER_LINEAR,
            0.0,
        ),
        (
            ((1, 1), (8, 1, 1)),
            (0.0, 0.0),
            (0.0, 0.0),
            labels.MIXER_ANDERSON,
            0.0,
        ),
        (
            ((1, 1), (8, 1, 1)),
            (0.0, 0.0),
            (0.0, 0.0),
            labels.MIXER_LINEAR,
            300.0,
        ),
    ],
    ids=(
        "h2-h3-plus-linear",
        "h2-water-linear",
        "h2-water-anderson",
        "h2-water-finite-temperature",
    ),
)
def test_padded_solver_trace_matches_independent_serial_systems(
    numbers: tuple[tuple[int, ...], ...],
    charges: tuple[float, ...],
    spins: tuple[float, ...],
    mixer: int,
    fermi_etemp: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same fixed-trip solver transitions match independent serial calls."""
    from dxtb._src.scf.functional import eager as eager_functional

    _, systems = _systems(
        numbers, mixer=mixer, maxiter=5, fermi_etemp=fermi_etemp
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charge_values = tuple(torch.tensor(item, dtype=DTYPE) for item in charges)
    spin_values = tuple(torch.tensor(item, dtype=DTYPE) for item in spins)
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charge_values),
        torch.stack(spin_values),
    )

    original_solve = eager_functional._solve_representation
    serial_traces = []

    def traced_serial_solve(*args, **kwargs):
        solved = original_solve(*args, record_trace=True, **kwargs)
        serial_traces.append(solved.trace)
        return solved

    monkeypatch.setattr(
        eager_functional, "_solve_representation", traced_serial_solve
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charge_values, spin_values
        )
    )

    from tad_mctc.tree import stack as tree_stack

    stacked_core = tree_stack(padded)
    mapped_results, mapped_traces = torch.func.vmap(
        lambda core, position, charge, spin: evaluate_padded_fixed_trip(
            core, position, charge, spin, record_trace=True
        )
    )(
        stacked_core,
        torch.stack(positions),
        torch.stack(charge_values),
        torch.stack(spin_values),
    )
    if numbers == ((1, 1), (8, 1, 1)):
        assert torch.any(
            mapped_traces.active_after[0] != mapped_traces.active_after[1]
        )
    assert len(serial_traces) == len(systems)
    assert all(trace is not None for trace in serial_traces)
    for lane, (system, serial_result, serial_trace) in enumerate(
        zip(systems, serial, serial_traces)
    ):
        assert serial_trace is not None
        source_map = dict(bucket.source_maps)[lane]
        atom_indices = torch.tensor(source_map.atoms, dtype=torch.int64)
        orbital_indices = torch.tensor(source_map.orbitals, dtype=torch.int64)
        mapped_trace = mapped_traces
        torch.testing.assert_close(
            mapped_results.energy[lane].index_select(0, atom_indices),
            serial_result.energy,
        )

        def physical_vector(name: str) -> torch.Tensor:
            mapped_value = getattr(mapped_trace, name)[lane]
            serial_value = getattr(serial_trace, name)
            indices = (
                atom_indices
                if serial_value.shape[-1] == len(source_map.atoms)
                else orbital_indices
            )
            return mapped_value.index_select(-1, indices)

        for name in ("initial_guess", "map_inputs", "raw_mapped", "iterates"):
            torch.testing.assert_close(
                physical_vector(name),
                getattr(serial_trace, name),
                atol=2.0e-9,
                rtol=2.0e-9,
            )
        for name in ("charges",):
            mapped_value = physical_vector(name)
            torch.testing.assert_close(
                mapped_value,
                getattr(serial_trace, name),
                atol=2.0e-9,
                rtol=2.0e-9,
            )
        for name in ("hamiltonians", "densities"):
            mapped_value = getattr(mapped_trace, name)[lane]
            mapped_value = mapped_value.index_select(
                -2, orbital_indices
            ).index_select(-1, orbital_indices)
            torch.testing.assert_close(
                mapped_value,
                getattr(serial_trace, name),
                atol=2.0e-8,
                rtol=2.0e-8,
            )
        for name in (
            "residuals",
            "active_before",
            "active_after",
            "iterations",
            "damping",
        ):
            torch.testing.assert_close(
                getattr(mapped_trace, name)[lane],
                getattr(serial_trace, name),
                atol=2.0e-10,
                rtol=2.0e-10,
            )
        for name in (
            "fermi_converged",
            "fermi_electron_count_ok",
            "fermi_invalid_temperature",
            "fermi_too_many_electrons",
            "fermi_residual",
        ):
            mapped_status = getattr(mapped_trace, name)
            serial_status = getattr(serial_trace, name)
            assert (mapped_status is None) == (serial_status is None)
            if mapped_status is not None and serial_status is not None:
                torch.testing.assert_close(
                    mapped_status[lane],
                    serial_status,
                    atol=2.0e-10,
                    rtol=2.0e-10,
                )
        assert len(mapped_trace.mixer_state) == len(serial_trace.mixer_state)
        for mapped_history, serial_history in zip(
            mapped_trace.mixer_state, serial_trace.mixer_state
        ):
            torch.testing.assert_close(
                mapped_history[lane].index_select(-1, orbital_indices),
                serial_history,
                atol=2.0e-9,
                rtol=2.0e-9,
            )


@pytest.mark.parametrize(
    ("int_level", "has_dipole", "has_quadrupole"),
    [
        (labels.INTLEVEL_DIPOLE, True, False),
        (labels.INTLEVEL_QUADRUPOLE, True, True),
    ],
    ids=("dipole", "quadrupole"),
)
def test_padded_full_fixed_trip_optional_integrals_match_serial(
    int_level: int, has_dipole: bool, has_quadrupole: bool
) -> None:
    """Mapped Result decoding preserves optional integral levels."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(
        numbers,
        mixer=labels.MIXER_LINEAR,
        maxiter=2,
        int_level=int_level,
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charges = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    assert all(result.dipole_integrals is not None for result in decoded)
    assert all(result.dipole_integrals is not None for result in serial)
    assert all(
        result.quadrupole_integrals is not None for result in decoded
    ) == (has_quadrupole)
    assert (
        all(result.quadrupole_integrals is not None for result in serial)
        == has_quadrupole
    )
    assert (mapped.dipole_integrals is not None) == has_dipole
    assert (mapped.quadrupole_integrals is not None) == has_quadrupole
    for actual, expected in zip(decoded, serial):
        torch.testing.assert_close(
            actual.dipole_integrals, expected.dipole_integrals
        )
        if has_quadrupole:
            torch.testing.assert_close(
                actual.quadrupole_integrals, expected.quadrupole_integrals
            )
        else:
            assert actual.quadrupole_integrals is None
            assert expected.quadrupole_integrals is None


@pytest.mark.parametrize(
    "numbers",
    [((1, 1), (8, 1, 1)), ((8, 1, 1), (1, 1))],
    ids=("h2-water", "water-h2"),
)
def test_padded_full_shared_field_matches_serial_and_derivatives(
    numbers: tuple[tuple[int, ...], ...],
) -> None:
    """A shared electric field retains energy and response in both lane orders."""
    model, systems = _systems(
        numbers,
        mixer=labels.MIXER_LINEAR,
        maxiter=3,
        int_level=labels.INTLEVEL_DIPOLE,
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    spins = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    field = torch.tensor([0.015, -0.01, 0.025], dtype=DTYPE, requires_grad=True)
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins, field=field
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin, field=field)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    for actual, expected in zip(decoded, serial):
        torch.testing.assert_close(
            actual.energy, expected.energy, atol=4e-7, rtol=4e-7
        )
        torch.testing.assert_close(
            actual.density, expected.density, atol=4e-7, rtol=4e-7
        )
        torch.testing.assert_close(
            actual.charges.mono, expected.charges.mono, atol=4e-7, rtol=4e-7
        )
        assert actual.dipole_integrals is not None
        assert expected.dipole_integrals is not None
    mapped_field_gradient = torch.autograd.grad(mapped.energy.sum(), field)[0]
    serial_field_gradient = torch.autograd.grad(
        sum(result.energy.sum() for result in serial), field
    )[0]
    torch.testing.assert_close(
        mapped_field_gradient, serial_field_gradient, atol=4e-6, rtol=4e-6
    )
    assert torch.isfinite(mapped_field_gradient).all()
    assert torch.count_nonzero(mapped_field_gradient) > 0
    mapped_field_jacobian = torch.func.jacfwd(
        lambda value: evaluate_padded_systems(
            padded, bucket, positions, charges, spins, field=value
        )[0].energy.sum(-1)
    )(field)
    serial_field_jacobian = torch.func.jacfwd(
        lambda value: torch.stack(
            tuple(
                system.singlepoint(
                    position, chrg=charge, spin=spin, field=value
                ).energy.sum()
                for system, position, charge, spin in zip(
                    systems, source_positions, charges, spins
                )
            )
        )
    )(field)
    torch.testing.assert_close(
        mapped_field_jacobian,
        serial_field_jacobian,
        atol=5e-6,
        rtol=5e-6,
    )

    zero_field = torch.zeros(3, dtype=DTYPE)
    zero_mapped, _ = evaluate_padded_systems(
        padded, bucket, positions, charges, spins, field=zero_field
    )
    no_field, _ = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    torch.testing.assert_close(zero_mapped.energy, no_field.energy)


@pytest.mark.parametrize(
    "numbers",
    [((1, 1), (8, 1, 1)), ((8, 1, 1), (1, 1))],
    ids=("h2-water", "water-h2"),
)
def test_padded_full_shared_field_gradient_matches_serial_and_derivatives(
    numbers: tuple[tuple[int, ...], ...],
) -> None:
    """A shared field gradient uses quadrupoles and matches serial response."""
    _, systems = _systems(
        numbers,
        mixer=labels.MIXER_LINEAR,
        maxiter=3,
        int_level=labels.INTLEVEL_QUADRUPOLE,
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    spins = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    field_grad = torch.tensor(
        [
            [0.012, 0.003, -0.002],
            [0.003, -0.006, 0.001],
            [-0.002, 0.001, -0.006],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins, field_grad=field_grad
    )
    serial = tuple(
        system.singlepoint(
            position, chrg=charge, spin=spin, field_grad=field_grad
        )
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    for actual, expected in zip(decoded, serial):
        torch.testing.assert_close(
            actual.energy, expected.energy, atol=4e-7, rtol=4e-7
        )
        torch.testing.assert_close(
            actual.density, expected.density, atol=4e-7, rtol=4e-7
        )
        assert actual.quadrupole_integrals is not None
        assert expected.quadrupole_integrals is not None
        assert actual.charges.quadrupole is not None
        assert expected.charges.quadrupole is not None
        torch.testing.assert_close(
            actual.charges.quadrupole,
            expected.charges.quadrupole,
            atol=4e-7,
            rtol=4e-7,
        )
    mapped_response = torch.autograd.grad(mapped.energy.sum(), field_grad)[0]
    serial_response = torch.autograd.grad(
        sum(result.energy.sum() for result in serial), field_grad
    )[0]
    torch.testing.assert_close(
        mapped_response, serial_response, atol=5e-6, rtol=5e-6
    )
    assert torch.isfinite(mapped_response).all()
    assert torch.count_nonzero(mapped_response) > 0
    mapped_response_jacobian = torch.func.jacfwd(
        lambda value: evaluate_padded_systems(
            padded, bucket, positions, charges, spins, field_grad=value
        )[0].energy.sum(-1)
    )(field_grad)
    serial_response_jacobian = torch.func.jacfwd(
        lambda value: torch.stack(
            tuple(
                system.singlepoint(
                    position, chrg=charge, spin=spin, field_grad=value
                ).energy.sum()
                for system, position, charge, spin in zip(
                    systems, source_positions, charges, spins
                )
            )
        )
    )(field_grad)
    torch.testing.assert_close(
        mapped_response_jacobian,
        serial_response_jacobian,
        atol=7e-6,
        rtol=7e-6,
    )


def test_padded_field_inputs_reject_bad_shapes_and_integral_levels() -> None:
    """Padded field validation matches single-System integral requirements."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers, mixer=labels.MIXER_LINEAR, maxiter=1)
    source_positions = tuple(_positions(item) for item in numbers)
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    spins = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    with pytest.raises(ValueError, match="field must have shape"):
        evaluate_padded_systems(
            padded,
            bucket,
            positions,
            charges,
            spins,
            field=torch.zeros(2, dtype=DTYPE),
        )
    with pytest.raises(RuntimeError, match="requires dipole integrals"):
        evaluate_padded_systems(
            padded,
            bucket,
            positions,
            charges,
            spins,
            field=torch.zeros(3, dtype=DTYPE),
        )
    with pytest.raises(RuntimeError, match="requires quadrupole integrals"):
        evaluate_padded_systems(
            padded,
            bucket,
            positions,
            charges,
            spins,
            field_grad=torch.zeros((3, 3), dtype=DTYPE),
        )


def test_padded_shared_field_and_gradient_combine_without_double_counting() -> (
    None
):
    """Dipole and quadrupole field terms compose as in serial Systems."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(
        numbers,
        mixer=labels.MIXER_LINEAR,
        maxiter=2,
        int_level=labels.INTLEVEL_QUADRUPOLE,
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    spins = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    field = torch.tensor([0.01, -0.02, 0.005], dtype=DTYPE)
    field_grad = torch.tensor(
        [[0.01, 0.0, 0.0], [0.0, -0.005, 0.0], [0.0, 0.0, -0.005]],
        dtype=DTYPE,
    )
    _, decoded = evaluate_padded_systems(
        padded,
        bucket,
        positions,
        charges,
        spins,
        field=field,
        field_grad=field_grad,
    )
    serial = tuple(
        system.singlepoint(
            position,
            chrg=charge,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    for actual, expected in zip(decoded, serial):
        torch.testing.assert_close(actual.energy, expected.energy)
        torch.testing.assert_close(actual.density, expected.density)


@pytest.mark.parametrize("defect", ("duplicate", "missing", "association"))
def test_padded_evaluator_rejects_invalid_source_maps(defect: str) -> None:
    """Malformed bucket identities fail before mapped numerical execution."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers, mixer=labels.MIXER_LINEAR, maxiter=1)
    source_positions = tuple(_positions(item) for item in numbers)
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    spins = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    bucket, padded, positions, _, _ = _prepare(
        systems, source_positions, torch.stack(charges), torch.stack(spins)
    )
    maps = tuple(bucket.source_maps)
    if defect == "duplicate":
        bucket = replace(bucket, indices=(4, 4))
        expected = "source identifiers must be unique"
    elif defect == "missing":
        bucket = replace(bucket, indices=(4, 99))
        expected = "source maps must match"
    else:
        bucket = replace(
            bucket,
            indices=(4, 19),
            source_maps=((4, maps[1][1]), (19, maps[0][1])),
        )
        expected = "does not match its padded"
    with pytest.raises(ValueError, match=expected):
        evaluate_padded_systems(padded, bucket, positions, charges, spins)


@pytest.mark.parametrize(
    "scp_mode",
    (labels.SCP_MODE_CHARGE, labels.SCP_MODE_POTENTIAL, labels.SCP_MODE_FOCK),
    ids=("charge", "potential", "fock"),
)
def test_padded_solver_projects_adversarial_inactive_state(
    scp_mode: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large inactive charge/potential/Fock entries cannot change SCF control."""
    from dxtb._src.scf.functional import eager as eager_functional
    from dxtb._src.scf.functional import padded as padded_module
    from dxtb._src.scf.functional.data import pack_potential, unpack_potential

    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(
        numbers,
        mixer=labels.MIXER_ANDERSON,
        maxiter=4,
        scp_mode=scp_mode,
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    spins = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in numbers)
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    stacked_core = stack(padded)
    position_batch = torch.stack(positions)
    charge_batch = torch.stack(charges)
    spin_batch = torch.stack(spins)

    def run():
        return torch.func.vmap(
            lambda core, position, charge, spin: evaluate_padded_fixed_trip(
                core, position, charge, spin, record_trace=True
            )
        )(stacked_core, position_batch, charge_batch, spin_batch)

    expected_result, expected_trace = run()
    monkeypatch_context = monkeypatch.context()
    with monkeypatch_context as scoped:
        if scp_mode == labels.SCP_MODE_CHARGE:
            original = padded_module._padded_initial_guess

            def corrupt_guess(core, position, charge, inputs):
                value = original(core, position, charge, inputs)
                return torch.where(core.validity.orbitals, value, value + 999.0)

            scoped.setattr(
                padded_module, "_padded_initial_guess", corrupt_guess
            )
        elif scp_mode == labels.SCP_MODE_POTENTIAL:
            original = eager_functional.potential_initial_guess

            def corrupt_potential(charges_value, inputs):
                packed, layout = original(charges_value, inputs)
                potential = unpack_potential(packed, layout)
                atoms = inputs.numbers != 0
                potential = type(potential)(
                    mono=torch.where(
                        inputs.orbital_mask,
                        potential.mono,
                        potential.mono + 999.0,
                    ),
                    dipole=(
                        None
                        if potential.dipole is None
                        else torch.where(
                            atoms.unsqueeze(-1),
                            potential.dipole,
                            potential.dipole + 999.0,
                        )
                    ),
                    quad=(
                        None
                        if potential.quad is None
                        else torch.where(
                            atoms.unsqueeze(-1),
                            potential.quad,
                            potential.quad + 999.0,
                        )
                    ),
                )
                return pack_potential(potential, layout), layout

            scoped.setattr(
                eager_functional, "potential_initial_guess", corrupt_potential
            )
        else:
            original = eager_functional.fock_initial_guess

            def corrupt_fock(charges_value, inputs):
                value = original(charges_value, inputs)
                mask = inputs.orbital_mask.unsqueeze(
                    -1
                ) & inputs.orbital_mask.unsqueeze(-2)
                return torch.where(mask, value, value + 999.0)

            scoped.setattr(eager_functional, "fock_initial_guess", corrupt_fock)
        actual_result, actual_trace = run()

    torch.testing.assert_close(actual_result.energy, expected_result.energy)
    torch.testing.assert_close(actual_result.residual, expected_result.residual)
    torch.testing.assert_close(
        actual_result.iterations, expected_result.iterations
    )
    torch.testing.assert_close(
        actual_result.converged, expected_result.converged
    )
    torch.testing.assert_close(actual_result.density, expected_result.density)
    for name in (
        "map_inputs",
        "raw_mapped",
        "iterates",
        "residuals",
        "active_before",
        "active_after",
        "iterations",
        "damping",
    ):
        torch.testing.assert_close(
            getattr(actual_trace, name), getattr(expected_trace, name)
        )
    for actual, expected in zip(
        actual_trace.mixer_state, expected_trace.mixer_state
    ):
        torch.testing.assert_close(actual, expected)


def test_padded_full_fixed_trip_position_and_parameter_derivatives() -> None:
    """Mapped SCF energy retains position and Model-owned kpol derivatives."""
    numbers = ((1, 1), (1, 1, 1))
    model, systems = _systems(
        numbers,
        differentiable=True,
        mixer=labels.MIXER_LINEAR,
        maxiter=4,
    )
    source_positions = tuple(
        _positions(item).requires_grad_() for item in numbers
    )
    charges = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    stacked_core = stack(padded)
    stacked_positions = torch.stack(canonical_positions)
    stacked_charges = torch.stack(charges)
    stacked_spins = torch.stack(spins)

    def mapped_results(pos):
        return torch.func.vmap(evaluate_padded_fixed_trip)(
            stacked_core, pos, stacked_charges, stacked_spins
        )

    mapped = mapped_results(stacked_positions)
    assert isinstance(mapped, Result)
    mapped_energy = mapped.energy.sum()
    mapped_position_gradient = torch.autograd.grad(
        mapped_energy, stacked_positions, retain_graph=True
    )[0]
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    serial_position_gradients = torch.autograd.grad(
        sum(result.energy.sum() for result in serial),
        source_positions,
        retain_graph=True,
    )
    for index, (system, serial_gradient) in enumerate(
        zip(systems, serial_position_gradients)
    ):
        mapping = dict(bucket.source_maps)[index]
        atom = torch.tensor(mapping.atoms, dtype=torch.int64)
        torch.testing.assert_close(
            mapped_position_gradient[index].index_select(0, atom),
            serial_gradient,
            atol=2e-6,
            rtol=2e-6,
        )
        inactive = ~padded[index].validity.atoms
        assert (
            torch.count_nonzero(mapped_position_gradient[index, inactive]) == 0
        )
    assert (
        torch.count_nonzero(
            mapped_position_gradient[0, padded[0].validity.atoms]
        )
        > 0
    )

    def energy_vector(pos):
        return mapped_results(pos).energy.sum(-1)

    position_jacobian = torch.func.jacrev(energy_vector)(stacked_positions)
    assert position_jacobian.shape == (
        len(systems),
        len(systems),
        stacked_positions.shape[1],
        3,
    )
    for output_lane in range(len(systems)):
        for input_lane in range(len(systems)):
            if output_lane != input_lane:
                assert (
                    torch.count_nonzero(
                        position_jacobian[output_lane, input_lane]
                    )
                    == 0
                )
    forward_jacobian = torch.func.jacfwd(energy_vector)(stacked_positions)
    torch.testing.assert_close(
        forward_jacobian,
        position_jacobian,
        atol=3e-6,
        rtol=3e-6,
    )

    functional_gradient = torch.func.grad(
        lambda pos: mapped_results(pos).energy.sum()
    )(stacked_positions)
    torch.testing.assert_close(
        functional_gradient,
        mapped_position_gradient,
        atol=3e-6,
        rtol=3e-6,
    )

    lane_gradient = torch.func.vmap(
        torch.func.grad(
            lambda pos, core, charge, spin: evaluate_padded_fixed_trip(
                core, pos, charge, spin
            ).energy.sum(),
            argnums=0,
        )
    )(stacked_positions, stacked_core, stacked_charges, stacked_spins)
    torch.testing.assert_close(
        lane_gradient,
        mapped_position_gradient,
        atol=3e-6,
        rtol=3e-6,
    )

    parameter = model.par.get("hamiltonian.xtb.kpol")
    mapped_parameter_gradient = torch.autograd.grad(
        mapped_energy, parameter, retain_graph=True
    )[0]
    serial_parameter_gradient = torch.autograd.grad(
        sum(result.energy.sum() for result in serial), parameter
    )[0]
    torch.testing.assert_close(
        mapped_parameter_gradient,
        serial_parameter_gradient,
        atol=3e-7,
        rtol=3e-7,
    )
    assert torch.isfinite(mapped_parameter_gradient).all()
    assert torch.count_nonzero(mapped_parameter_gradient) > 0

    direction = torch.ones_like(stacked_positions) * 0.01
    _, forward = torch.func.jvp(
        lambda pos: mapped_results(pos).energy.sum(),
        (stacked_positions.detach(),),
        (direction,),
    )
    reverse = (mapped_position_gradient.detach() * direction).sum()
    torch.testing.assert_close(forward, reverse, atol=3e-6, rtol=3e-6)


def test_padded_full_scf_ignores_inactive_coordinate_perturbations() -> None:
    """Padded EEQ, interactions, classical energy, and SCF ignore dummy atoms."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers, mixer=labels.MIXER_LINEAR, maxiter=2)
    positions = tuple(_positions(item) for item in numbers)
    charges = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, torch.stack(charges), torch.stack(spins)
    )
    stacked_core = stack(padded)
    source = torch.stack(canonical_positions).detach().requires_grad_()
    perturbed = source.detach().clone()
    for lane, padded_core in enumerate(padded):
        inactive = ~padded_core.validity.atoms
        perturbed[lane, inactive] = source[lane, 0].detach()
    perturbed.requires_grad_()
    charge_batch = torch.stack(charges)
    spin_batch = torch.stack(spins)

    def run(pos):
        return torch.func.vmap(evaluate_padded_fixed_trip)(
            stacked_core, pos, charge_batch, spin_batch
        )

    reference = run(source)
    actual = run(perturbed)
    torch.testing.assert_close(actual.energy, reference.energy)
    torch.testing.assert_close(actual.density, reference.density)
    assert actual.charges is not None and reference.charges is not None
    torch.testing.assert_close(actual.charges.mono, reference.charges.mono)
    gradient = torch.autograd.grad(actual.energy.sum(), perturbed)[0]
    for lane, padded_core in enumerate(padded):
        assert (
            torch.count_nonzero(gradient[lane, ~padded_core.validity.atoms])
            == 0
        )


def test_padded_full_d3_parameter_gradient_matches_serial() -> None:
    """The mapped classical D3 contribution retains its Model-owned s6 leaf."""
    numbers = ((1, 1), (1, 1, 1))
    model, systems = _systems(
        numbers,
        mixer=labels.MIXER_LINEAR,
        maxiter=2,
        differentiable_d3=("s6",),
    )
    parameter = model.par.get("dispersion.d3.s6", unwrapped=False).param
    snapshot = parameter.detach().clone()
    source_positions = tuple(_positions(item) for item in numbers)
    charges = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    mapped, _ = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    mapped_gradient = torch.autograd.grad(
        mapped.energy.sum(), parameter, retain_graph=True
    )[0]
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    serial_gradient = torch.autograd.grad(
        sum(result.energy.sum() for result in serial), parameter
    )[0]
    torch.testing.assert_close(
        mapped_gradient, serial_gradient, atol=4e-7, rtol=4e-7
    )
    assert torch.isfinite(mapped_gradient).all()
    assert torch.count_nonzero(mapped_gradient) > 0
    torch.testing.assert_close(parameter, snapshot)


def test_padded_full_es2_parameter_gradient_matches_serial() -> None:
    """The mapped ES2 Hubbard gradient reaches the single Model parameter."""
    numbers = ((1, 1), (1, 1, 1))
    model, systems = _systems(
        numbers,
        differentiable_es2=True,
        mixer=labels.MIXER_LINEAR,
        maxiter=2,
    )
    parameter = model.par.get("element.H.gam")
    snapshot = parameter.detach().clone()
    source_positions = tuple(_positions(item) for item in numbers)
    charges = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    mapped, _ = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    mapped_gradient = torch.autograd.grad(
        mapped.energy.sum(), parameter, retain_graph=True
    )[0]
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    serial_gradient = torch.autograd.grad(
        sum(result.energy.sum() for result in serial), parameter
    )[0]
    torch.testing.assert_close(
        mapped_gradient, serial_gradient, atol=4e-7, rtol=4e-7
    )
    assert torch.isfinite(mapped_gradient).all()
    assert torch.count_nonzero(mapped_gradient) > 0
    torch.testing.assert_close(parameter, snapshot)


@pytest.mark.parametrize(
    ("scp_mode", "mixer"),
    [
        (labels.SCP_MODE_CHARGE, labels.MIXER_ANDERSON),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_LINEAR),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_ANDERSON),
        (labels.SCP_MODE_FOCK, labels.MIXER_LINEAR),
        (labels.SCP_MODE_FOCK, labels.MIXER_ANDERSON),
    ],
    ids=(
        "charge-anderson",
        "potential-linear",
        "potential-anderson",
        "fock-linear",
        "fock-anderson",
    ),
)
def test_padded_full_fixed_trip_representation_matches_serial(
    scp_mode: int, mixer: int
) -> None:
    """Existing functional representations/mixers run on a padded pair."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(
        numbers,
        mixer=mixer,
        maxiter=2,
        scp_mode=scp_mode,
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charges = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    _, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    for actual, expected in zip(decoded, serial):
        torch.testing.assert_close(
            actual.energy, expected.energy, atol=3e-7, rtol=3e-7
        )
        torch.testing.assert_close(
            actual.density, expected.density, atol=3e-7, rtol=3e-7
        )
        torch.testing.assert_close(actual.iterations, expected.iterations)
        assert actual.status is not None and expected.status is not None
        torch.testing.assert_close(
            actual.status.trajectory_failure,
            expected.status.trajectory_failure,
        )
        torch.testing.assert_close(
            actual.status.final_failure, expected.status.final_failure
        )


@pytest.mark.parametrize(
    ("maxiter", "fermi_etemp", "fermi_maxiter", "fermi_thresh"),
    [
        (0, 0.0, 200, None),
        (2, 300.0, 200, None),
        (2, 300.0, 0, 1.0e-20),
    ],
    ids=("zero-trip", "finite-temperature", "fermi-status"),
)
def test_padded_full_fixed_trip_status_matches_serial(
    maxiter: int,
    fermi_etemp: float,
    fermi_maxiter: int,
    fermi_thresh: float | None,
) -> None:
    """Outer trip count and lane-local spectral status preserve serial meaning."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(
        numbers,
        mixer=labels.MIXER_LINEAR,
        maxiter=maxiter,
        fermi_etemp=fermi_etemp,
        fermi_maxiter=fermi_maxiter,
        fermi_thresh=fermi_thresh,
    )
    source_positions = tuple(_positions(item) for item in numbers)
    charges = (
        torch.tensor(
            0.2 if fermi_maxiter == 0 and fermi_etemp > 0.0 else 0.0,
            dtype=DTYPE,
        ),
        torch.tensor(0.0, dtype=DTYPE),
    )
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
    bucket, padded, positions, _, _ = _prepare(
        systems,
        source_positions,
        torch.stack(charges),
        torch.stack(spins),
    )
    _, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    if maxiter == 0:
        assert all(not bool(result.converged) for result in decoded)
        assert all(
            int(result.status.final_failure) == int(SCFFailureCode.NONE)
            for result in decoded
        )
    elif fermi_maxiter == 0 and fermi_etemp > 0.0:
        assert int(decoded[0].status.trajectory_failure) == int(
            SCFFailureCode.FERMI_NONCONVERGENCE
        )
        assert int(decoded[0].status.final_failure) == int(
            SCFFailureCode.FERMI_NONCONVERGENCE
        )
        assert int(decoded[1].status.trajectory_failure) == int(
            SCFFailureCode.NONE
        )
        assert int(decoded[1].status.final_failure) == int(SCFFailureCode.NONE)
    elif fermi_etemp > 0.0:
        assert bool(decoded[0].converged)
        assert not bool(decoded[1].converged)
        assert all(
            int(result.status.final_failure) == int(SCFFailureCode.NONE)
            for result in decoded
        )
    for actual, expected in zip(decoded, serial):
        torch.testing.assert_close(actual.energy, expected.energy)
        torch.testing.assert_close(actual.iterations, expected.iterations)
        assert torch.equal(actual.converged, expected.converged)
        assert actual.status is not None and expected.status is not None
        torch.testing.assert_close(
            actual.status.trajectory_failure,
            expected.status.trajectory_failure,
        )
        torch.testing.assert_close(
            actual.status.final_failure, expected.status.final_failure
        )


def test_fermi_nonconvergence_status_maps_without_host_inspection():
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(
        numbers,
        fermi_etemp=300.0,
        fermi_maxiter=0,
        fermi_thresh=1.0e-20,
    )
    positions = tuple(_positions(entry) for entry in numbers)
    charges = torch.tensor([0.2, 0.0], dtype=DTYPE)
    spins = torch.zeros(2, dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charges, spins
    )
    mapped = torch.func.vmap(_padded_lane)(
        stack(padded), torch.stack(canonical_positions), charges, spins
    )
    assert mapped.fermi_converged is not None
    assert mapped.fermi_residual is not None
    torch.testing.assert_close(
        mapped.fermi_converged,
        torch.tensor([[False, False], [True, True]]),
    )
    for index, (system, position) in enumerate(zip(systems, positions)):
        _, serial = _serial_h0_spectrum(
            system,
            position,
            charges[index],
            spins[index],
        )
        torch.testing.assert_close(
            mapped.fermi_converged[index], serial.fermi_status.converged
        )
        torch.testing.assert_close(
            mapped.fermi_residual[index], serial.fermi_status.residual
        )


def test_eager_padded_spin_validation_rejects_inconsistent_source():
    numbers = ((1, 1),)
    _, systems = _systems(numbers)
    core = systems[0].exact_core
    assert core is not None
    bucket = bucket_canonical_exact_cores((core,))
    padded = pad_canonical_exact_core(core, bucket, 0)

    valid = validate_padded_charge_spin(
        padded,
        torch.tensor(0.0, dtype=DTYPE),
        torch.tensor(0.0, dtype=DTYPE),
    )
    torch.testing.assert_close(valid, torch.tensor([1.0, 1.0], dtype=DTYPE))

    with pytest.raises(ValueError, match=r"Odd \(even\) number of unpaired"):
        validate_padded_charge_spin(
            padded,
            torch.tensor(0.0, dtype=DTYPE),
            torch.tensor(1.0, dtype=DTYPE),
        )


def test_eager_padded_spin_validation_rejects_too_many_unpaired_electrons():
    numbers = ((1, 1),)
    _, systems = _systems(numbers)
    core = systems[0].exact_core
    assert core is not None
    bucket = bucket_canonical_exact_cores((core,))
    padded = pad_canonical_exact_core(core, bucket, 0)

    with pytest.raises(ValueError, match="larger than number of electrons"):
        validate_padded_charge_spin(
            padded,
            torch.tensor(1.0, dtype=DTYPE),
            torch.tensor(2.0, dtype=DTYPE),
        )


def test_padded_spectral_jacobian_is_lane_local_and_position_differentiable():
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _systems(numbers)
    positions = tuple(_positions(entry) for entry in numbers)
    charges = torch.zeros(2, dtype=DTYPE)
    spins = torch.zeros(2, dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charges, spins
    )
    stacked_core = stack(padded)
    stacked_positions = torch.stack(canonical_positions)
    weights = torch.arange(bucket.orbitals**2, dtype=DTYPE).reshape(
        bucket.orbitals, bucket.orbitals
    )

    def energy_vector(position_batch):
        spectral = torch.func.vmap(_padded_lane)(
            stacked_core, position_batch, charges, spins
        )
        return (spectral.density * weights).sum((-2, -1))

    jacobian = torch.func.jacrev(energy_vector)(stacked_positions)
    forward_jacobian = torch.func.jacfwd(energy_vector)(stacked_positions)
    assert jacobian.shape == (2, 2, *stacked_positions.shape[1:])
    torch.testing.assert_close(forward_jacobian, jacobian, atol=3e-6, rtol=3e-6)
    assert torch.count_nonzero(jacobian[0, 1]) == 0
    assert torch.count_nonzero(jacobian[1, 0]) == 0
    for index, padded_system in enumerate(padded):
        inactive = ~padded_system.validity.atoms
        assert torch.count_nonzero(jacobian[index, index, inactive]) == 0

    for index, (system, position) in enumerate(zip(systems, positions)):
        mapping = dict(bucket.source_maps)[index]
        ao = torch.tensor(mapping.orbitals, dtype=torch.int64)
        atom = torch.tensor(mapping.atoms, dtype=torch.int64)
        physical_weights = weights.index_select(0, ao).index_select(1, ao)
        reference = torch.func.grad(
            lambda value: (
                _serial_h0_spectrum(
                    system,
                    value,
                    torch.zeros((), dtype=DTYPE),
                    torch.zeros((), dtype=DTYPE),
                )[1].density
                * physical_weights
            ).sum()
        )(position)
        torch.testing.assert_close(
            jacobian[index, index].index_select(0, atom),
            reference,
            atol=2e-6,
            rtol=2e-6,
        )

    direction = torch.ones_like(stacked_positions)
    _, tangent = torch.func.jvp(
        energy_vector, (stacked_positions,), (direction,)
    )
    diagonal = torch.stack([jacobian[index, index].sum() for index in range(2)])
    torch.testing.assert_close(tangent, diagonal)


def test_two_axis_density_reduction_is_vmap_safe():
    from dxtb._src.scf.functional.padded import PaddedCoreIndexData

    index = PaddedCoreIndexData(
        shells_to_atom=torch.tensor([0, -1, 1]),
        orbitals_to_shell=torch.tensor([0, -1, 1]),
        atom_count=2,
        shell_count=2,
        valid_shells=torch.tensor([True, False, True]),
        valid_orbitals=torch.tensor([True, False, True]),
    )
    matrices = torch.tensor(
        [
            [[3.0, 999.0, 5.0], [999.0, 999.0, 999.0], [7.0, 999.0, 11.0]],
            [[13.0, 999.0, 17.0], [999.0, 999.0, 999.0], [19.0, 999.0, 23.0]],
        ]
    )
    mapped = torch.func.vmap(
        lambda value: index.reduce_orbital_to_shell(value, dim=(0, 1))
    )(matrices)
    expected = torch.tensor(
        [
            [[3.0, 5.0], [7.0, 11.0]],
            [[13.0, 17.0], [19.0, 23.0]],
        ]
    )
    torch.testing.assert_close(mapped, expected)


def test_padded_h0_parameter_gradient_reaches_original_parameter():
    numbers = ((1, 1), (8, 1, 1))
    model, systems = _systems(numbers, differentiable=True)
    parameter = model.par.get("hamiltonian.xtb.kpol")
    parameter_snapshot = parameter.detach().clone()
    positions = tuple(_positions(entry) for entry in numbers)
    charges = torch.zeros(2, dtype=DTYPE)
    spins = torch.zeros(2, dtype=DTYPE)
    bucket, padded, canonical_positions, _, _ = _prepare(
        systems, positions, charges, spins
    )
    mapped = torch.func.vmap(_padded_lane)(
        stack(padded), torch.stack(canonical_positions), charges, spins
    )
    weights = torch.arange(bucket.orbitals**2, dtype=DTYPE).reshape(
        bucket.orbitals, bucket.orbitals
    )
    mapped_value = (mapped.density * weights).sum()
    mapped_gradient = torch.autograd.grad(
        mapped_value, parameter, retain_graph=True
    )[0]

    serial_values = []
    for system, position in zip(systems, positions):
        _, reference = _serial_h0_spectrum(
            system,
            position,
            torch.zeros((), dtype=DTYPE),
            torch.zeros((), dtype=DTYPE),
        )
        mapping = dict(bucket.source_maps)[len(serial_values)]
        ao = torch.tensor(mapping.orbitals, dtype=torch.int64)
        physical_weights = weights.index_select(0, ao).index_select(1, ao)
        serial_values.append((reference.density * physical_weights).sum())
    serial_gradient = torch.autograd.grad(sum(serial_values), parameter)[0]
    torch.testing.assert_close(
        mapped_gradient, serial_gradient, atol=2e-7, rtol=2e-7
    )
    assert id(model.par.get("hamiltonian.xtb.kpol")) == id(parameter)
    torch.testing.assert_close(parameter, parameter_snapshot)
