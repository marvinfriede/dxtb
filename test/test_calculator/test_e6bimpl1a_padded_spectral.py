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

import pytest
import torch
from tad_mctc import storch
from tad_mctc.tree import stack
from tad_mctc.units.energy import KELVIN2AU

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.padded_integrals import evaluate_padded_integrals
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
    evaluate_padded_scf_spectrum,
    validate_padded_charge_spin,
)
from dxtb._src.scf.iterator import get_refocc

DTYPE = torch.float64


def _model(
    *,
    differentiable_h0: bool = False,
    fermi_etemp: float = 0.0,
    fermi_maxiter: int = 200,
    fermi_thresh: float | None = None,
) -> Model:
    parameters = ParamModule(GFN1_XTB, dtype=DTYPE)
    if differentiable_h0:
        parameters.set_differentiable("hamiltonian", "xtb", "kpol")
    return Model(
        par=parameters,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_HCORE,
            fermi_etemp=fermi_etemp,
            fermi_maxiter=fermi_maxiter,
            fermi_thresh=fermi_thresh,
        ),
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
    fermi_etemp: float = 0.0,
    fermi_maxiter: int = 200,
    fermi_thresh: float | None = None,
):
    model = _model(
        differentiable_h0=differentiable,
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
