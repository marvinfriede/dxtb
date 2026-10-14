# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""Pure raw physical maps and spectral evaluation for one System."""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.math import einsum

from dxtb import IndexHelper, labels
from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.typing import Tensor

from ...wavefunction import filling
from .data import (
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFMapResult,
    SCFPotential,
    SCFSpectralResult,
    pack_charges,
    pack_potential,
    potential_layout,
    unpack_charges,
    unpack_potential,
)
from .spectral import aufbau_density_matrix, fermi_density_matrix

__all__ = [
    "evaluate_hamiltonian",
    "evaluate_potential",
    "fock_initial_guess",
    "fock_map",
    "potential_from_charges",
    "potential_initial_guess",
    "potential_to_hamiltonian",
    "potential_map",
    "scf_map",
    "validate_fock_state",
]


def potential_to_hamiltonian(
    hcore: Tensor,
    overlap: Tensor,
    dipole_integrals: Tensor | None,
    quadrupole_integrals: Tensor | None,
    ihelp: IndexHelper,
    potential: SCFPotential,
) -> Tensor:
    """Return the Hamiltonian assembled from explicit integral values."""
    hamiltonian = hcore
    if potential.mono is not None:
        vmono = potential.mono.unsqueeze(-1) + potential.mono.unsqueeze(-2)
        hamiltonian = hamiltonian - 0.5 * overlap * vmono

    def add_multipole(
        current: Tensor, integrals: Tensor, value: Tensor
    ) -> Tensor:
        spread = ihelp.spread_atom_to_orbital(value, dim=-2, extra=True)
        contribution = 0.5 * einsum("...kij,...jk->...ij", integrals, spread)
        return current - (contribution + contribution.mT)

    if potential.dipole is not None and dipole_integrals is not None:
        hamiltonian = add_multipole(
            hamiltonian, dipole_integrals, potential.dipole
        )
    if potential.quad is not None and quadrupole_integrals is not None:
        hamiltonian = add_multipole(
            hamiltonian, quadrupole_integrals, potential.quad
        )
    return hamiltonian


def _legacy_charges(charges: SCFCharges) -> Charges:
    """Make the short-lived mutable adapter required by interactions."""
    return Charges(mono=charges.mono, dipole=charges.dipole, quad=charges.quad)


def _legacy_potential(potential: Potential) -> SCFPotential:
    """Capture a local interaction potential in an immutable value."""
    return SCFPotential(
        mono=potential.mono,
        dipole=potential.dipole,
        quad=potential.quad,
    )


def potential_from_charges(
    charges: SCFCharges, inputs: SCFInputs
) -> SCFPotential:
    """Evaluate a fresh potential value from arbitrary explicit charges."""
    local = inputs.interactions.get_potential(
        inputs.interaction_data, _legacy_charges(charges), inputs.ihelp
    )
    return _legacy_potential(local)


def _transitional_diagonalize(
    hamiltonian: Tensor, inputs: SCFInputs
) -> tuple[Tensor, Tensor]:
    """Return ordinary eigensystem values for reporting and filling only."""
    return storch.eighb(
        a=hamiltonian,
        b=None,
        l_inv=inputs.inverse_cholesky,
        is_posdef=True,
        factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
        broadening_method=None,
    )


def _density_to_charges(
    density: Tensor, inputs: SCFInputs
) -> tuple[SCFCharges, Tensor]:
    """Return Mulliken multipoles and orbital-resolved core energy."""
    integrals = inputs.integrals
    orbital_energy = einsum("...ik,...ki->...i", density, integrals.hcore)
    populations = einsum("...ik,...ki->...i", density, integrals.overlap)
    charges = SCFCharges(mono=inputs.n0 - populations)

    dipole = None
    if integrals.dipole is not None:
        dipole = inputs.ihelp.reduce_orbital_to_atom(
            -einsum("...ik,...mki->...im", density, integrals.dipole),
            extra=True,
            dim=-2,
        )

    quad = None
    if integrals.quadrupole is not None:
        quad = inputs.ihelp.reduce_orbital_to_atom(
            -einsum("...ik,...mki->...im", density, integrals.quadrupole),
            extra=True,
            dim=-2,
        )

    return SCFCharges(charges.mono, dipole=dipole, quad=quad), orbital_energy


def evaluate_hamiltonian(
    hamiltonian: Tensor, inputs: SCFInputs
) -> SCFSpectralResult:
    """Evaluate spectrum, occupation, density and charges for a Hamiltonian."""
    emo, coefficients = _transitional_diagonalize(hamiltonian, inputs)
    fermi_status = None
    fermi_state = None
    if inputs.fermi_settings is None or not inputs.fermi_settings.smear:
        occupation = inputs.occupation
        density = aufbau_density_matrix(
            hamiltonian,
            inputs.nel,
            l_inv=inputs.inverse_cholesky,
            mask=inputs.orbital_mask,
        )
    else:
        settings = inputs.fermi_settings
        channel_energies = emo.unsqueeze(-2).expand(
            *inputs.nel.shape, emo.shape[-1]
        )
        channel_mask = inputs.orbital_mask.unsqueeze(-2).expand_as(
            channel_energies
        )
        fermi_state = filling._get_fermi_occupation_core_state(
            inputs.nel,
            channel_energies,
            settings.kt,
            mask=channel_mask,
            thr=settings.threshold,
            maxiter=settings.maxiter,
            diff_order=settings.diff_order,
        )
        fermi_result = fermi_state.result
        occupation = fermi_result.occupation
        fermi_status = fermi_result.status
        density = fermi_density_matrix(
            hamiltonian,
            inputs.nel,
            settings.kt,
            fermi_state,
            l_inv=inputs.inverse_cholesky,
            mask=inputs.orbital_mask,
            diff_steps=filling._diff_steps(settings.diff_order),
        )
    charges, orbital_energy = _density_to_charges(density, inputs)
    return SCFSpectralResult(
        hamiltonian=hamiltonian,
        density=density,
        emo=emo,
        coefficients=coefficients,
        occupation=occupation,
        charges=charges,
        orbital_energy=orbital_energy,
        fermi_status=fermi_status,
        _fermi_state=fermi_state,
    )


def evaluate_potential(
    potential: SCFPotential, inputs: SCFInputs
) -> SCFSpectralResult:
    """Evaluate one potential-driven Hamiltonian spectral state."""
    integrals = inputs.integrals
    hamiltonian = potential_to_hamiltonian(
        integrals.hcore,
        integrals.overlap,
        integrals.dipole,
        integrals.quadrupole,
        inputs.ihelp,
        potential,
    )
    return evaluate_hamiltonian(hamiltonian, inputs)


def _map_result(
    mapped_x: Tensor,
    spectral: SCFSpectralResult,
    driving_potential: SCFPotential | None = None,
) -> SCFMapResult:
    """Bind representation-specific map values to a shared spectral result."""
    return SCFMapResult(
        mapped_x=mapped_x,
        spectral=spectral,
        driving_potential=driving_potential,
    )


def scf_map(x: Tensor, inputs: SCFInputs) -> SCFMapResult:
    """Evaluate one raw charge-mode map ``G(x)`` without solver state.

    The map is single-System and contains no mixing, convergence check, or
    persistent state. Charge-mode raw maps support both Aufbau occupation and
    fixed-trip finite-temperature Fermi occupation.
    """
    charges_in = unpack_charges(x, inputs.charge_layout)
    driving_potential = potential_from_charges(charges_in, inputs)
    spectral = evaluate_potential(driving_potential, inputs)
    return _map_result(
        pack_charges(spectral.charges, inputs.charge_layout),
        spectral,
        driving_potential,
    )


def potential_map(
    x: Tensor, inputs: SCFInputs, layout: PotentialLayout
) -> SCFMapResult:
    """Evaluate one raw potential-representation map without solver state."""
    driving_potential = unpack_potential(x, layout)
    spectral = evaluate_potential(driving_potential, inputs)
    mapped_potential = potential_from_charges(spectral.charges, inputs)
    return _map_result(
        pack_potential(mapped_potential, layout), spectral, driving_potential
    )


def validate_fock_state(hamiltonian: Tensor, inputs: SCFInputs) -> None:
    """Validate fixed single-System Fock matrix structure."""
    expected = inputs.integrals.hcore.shape
    if hamiltonian.ndim != 2 or hamiltonian.shape != expected:
        raise ValueError(
            "Fock state must be a single-System matrix with shape "
            f"{expected}; received shape {tuple(hamiltonian.shape)}."
        )


def fock_initial_guess(charges: SCFCharges, inputs: SCFInputs) -> Tensor:
    """Build the initial Fock matrix from the physical charge guess."""
    potential = potential_from_charges(charges, inputs)
    integrals = inputs.integrals
    return potential_to_hamiltonian(
        integrals.hcore,
        integrals.overlap,
        integrals.dipole,
        integrals.quadrupole,
        inputs.ihelp,
        potential,
    )


def fock_map(hamiltonian: Tensor, inputs: SCFInputs) -> SCFMapResult:
    """Evaluate one raw Fock map ``H -> spectrum -> charges -> potential -> H``."""
    validate_fock_state(hamiltonian, inputs)
    spectral = evaluate_hamiltonian(hamiltonian, inputs)
    mapped_potential = potential_from_charges(spectral.charges, inputs)
    integrals = inputs.integrals
    mapped_hamiltonian = potential_to_hamiltonian(
        integrals.hcore,
        integrals.overlap,
        integrals.dipole,
        integrals.quadrupole,
        inputs.ihelp,
        mapped_potential,
    )
    return _map_result(mapped_hamiltonian, spectral)


def potential_initial_guess(
    charges: SCFCharges, inputs: SCFInputs
) -> tuple[Tensor, PotentialLayout]:
    """Build the packed potential guess from the initial physical charges."""
    potential = potential_from_charges(charges, inputs)
    layout = potential_layout(potential)
    return pack_potential(potential, layout), layout
