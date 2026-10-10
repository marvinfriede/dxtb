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
"""Pure raw physical map for single-System charge-mode SCF."""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.math import einsum
from tad_mctc.units.energy import KELVIN2AU

from dxtb import IndexHelper
from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.typing import Tensor

from ..utils import get_density
from .data import (
    SCFCharges,
    SCFInputs,
    SCFMapResult,
    SCFPotential,
    pack_charges,
    unpack_charges,
)

__all__ = ["potential_to_hamiltonian", "scf_map"]


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


def _potential_from_charges(
    charges: SCFCharges, inputs: SCFInputs
) -> SCFPotential:
    local = inputs.interactions.get_potential(
        inputs.interaction_data, _legacy_charges(charges), inputs.ihelp
    )
    return _legacy_potential(local)


def _transitional_diagonalize(
    hamiltonian: Tensor, inputs: SCFInputs
) -> tuple[Tensor, Tensor]:
    """Use current full-mode eigensolver semantics pending E3.

    The ``requires_grad`` selection intentionally mirrors the current full
    solver only for raw-map forward equivalence. It is transitional and must
    be replaced by the common derivative contract owned by E3.
    """
    if hamiltonian.requires_grad is False and (
        inputs.inverse_cholesky.requires_grad is False
    ):
        broadening_method = None
    else:
        broadening_method = "lorn"
    return storch.eighb(
        a=hamiltonian,
        b=None,
        l_inv=inputs.inverse_cholesky,
        is_posdef=True,
        factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
        broadening_method=broadening_method,
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


def scf_map(x: Tensor, inputs: SCFInputs) -> SCFMapResult:
    """Evaluate one raw charge-mode map ``G(x)`` without solver state.

    The map is single-System and contains no mixing, convergence check, or
    persistent state. Finite-temperature occupation remains deferred to the
    transform-compatible Fermi work in a later E4 package.
    """
    from dxtb._src.constants import labels

    if inputs.config.scp_mode != labels.SCP_MODE_CHARGE:
        raise NotImplementedError(
            "E4.0 currently implements the raw charge SCF map only; "
            "potential and Fock modes are deferred."
        )

    if inputs.config.fermi.etemp * KELVIN2AU >= 3.0e-7:
        raise NotImplementedError(
            "E4.0 raw-map evaluation currently requires effectively zero "
            "electronic temperature; finite-temperature Fermi handling is "
            "deferred."
        )

    charges_in = unpack_charges(x, inputs.charge_layout)
    potential = _potential_from_charges(charges_in, inputs)
    integrals = inputs.integrals
    hamiltonian = potential_to_hamiltonian(
        integrals.hcore,
        integrals.overlap,
        integrals.dipole,
        integrals.quadrupole,
        inputs.ihelp,
        potential,
    )

    emo, coefficients = _transitional_diagonalize(hamiltonian, inputs)
    occupation = inputs.occupation
    density = get_density(coefficients, occupation.sum(-2))
    charges_out, orbital_energy = _density_to_charges(density, inputs)

    return SCFMapResult(
        mapped_x=pack_charges(charges_out, inputs.charge_layout),
        charges=charges_out,
        potential=potential,
        hamiltonian=hamiltonian,
        density=density,
        emo=emo,
        coefficients=coefficients,
        occupation=occupation,
        orbital_energy=orbital_energy,
    )
