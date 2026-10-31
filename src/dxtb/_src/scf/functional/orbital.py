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
"""Fixed-occupation orbital response primitives for functional SOSCF."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dxtb._src.typing import Tensor

from .data import SCFSpectralResult
from .energy import electronic_energy_from_density
from .map import density_to_charges_and_core_energy
from .problem import SCFProblem, SCFRepresentation

__all__ = ["SOSCFOrbitalModel", "build_orbital_model"]


def _build_rotation_pairs(
    occupation: Tensor, orbital_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Build occupied-to-less-occupied orbital rotation pairs."""
    total = occupation.sum(dim=-2)
    values = total.detach().cpu().tolist()
    valid = orbital_mask.detach().cpu().tolist()
    occupied: list[int] = []
    virtual: list[int] = []
    differences: list[float] = []
    for i, n_i in enumerate(values):
        if not valid[i]:
            continue
        for a, n_a in enumerate(values):
            if valid[a] and n_i > n_a:
                occupied.append(i)
                virtual.append(a)
                differences.append(n_i - n_a)
    device = occupation.device
    return (
        torch.tensor(occupied, dtype=torch.int64, device=device),
        torch.tensor(virtual, dtype=torch.int64, device=device),
        occupation.new_tensor(differences),
    )


def _rotate_coefficients(
    coefficients: Tensor, pair_i: Tensor, pair_a: Tensor, rotation: Tensor
) -> Tensor:
    """Apply an exact orthogonal orbital rotation to coefficient columns."""
    generator = torch.zeros_like(coefficients)
    generator = generator.index_put((pair_a, pair_i), rotation)
    generator = generator.index_put((pair_i, pair_a), -rotation)
    return coefficients @ torch.matrix_exp(generator)


@dataclass(frozen=True, eq=False)
class SOSCFOrbitalModel:
    """Fixed-occupation xTB energy and orbital gradient at one observation."""

    problem: SCFProblem
    coefficients: Tensor
    occupation: Tensor
    pair_i: Tensor
    pair_a: Tensor
    occupation_difference: Tensor

    @property
    def nrot(self) -> int:
        """Number of independent unequal-occupation orbital rotations."""
        return self.pair_i.numel()

    def coefficients_at(self, rotation: Tensor) -> Tensor:
        """Return coefficients after applying a packed rotation vector."""
        return _rotate_coefficients(
            self.coefficients, self.pair_i, self.pair_a, rotation
        )

    def density_at(self, rotation: Tensor) -> Tensor:
        """Return the fixed-occupation density for a rotation vector."""
        coefficients = self.coefficients_at(rotation)
        total_occupation = self.occupation.sum(dim=-2)
        return (coefficients * total_occupation.unsqueeze(-2)) @ coefficients.mT

    def energy_at(self, rotation: Tensor) -> Tensor:
        """Return electronic xTB energy for a rotated fixed occupation."""
        density = self.density_at(rotation)
        return electronic_energy_from_density(
            density, self.problem.inputs
        ).sum()

    def fock_at(self, rotation: Tensor) -> Tensor:
        """Rebuild the physical Fock from a rotated density, without a map."""
        density = self.density_at(rotation)
        charges, _ = density_to_charges_and_core_energy(
            density, self.problem.inputs
        )
        return self.problem.rebuild_hamiltonian(charges)

    def physical_gradient_at(
        self, rotation: Tensor, fock: Tensor | None = None
    ) -> Tensor:
        """Evaluate the physical commutator gradient in the rotated basis."""
        coefficients = self.coefficients_at(rotation)
        physical_fock = self.fock_at(rotation) if fock is None else fock
        fock_mo = coefficients.mT @ physical_fock @ coefficients
        return (
            2.0 * self.occupation_difference * fock_mo[self.pair_a, self.pair_i]
        )

    def gradient_at(self, rotation: Tensor) -> Tensor:
        """Return the exact coordinate gradient in the exponential chart."""
        return torch.func.grad(self.energy_at)(rotation)


def build_orbital_model(
    problem: SCFProblem,
    spectral: SCFSpectralResult,
    *,
    occupation_tolerance: float = 1.0e-10,
) -> SOSCFOrbitalModel:
    """Validate S8 occupation scope and build the local orbital model."""
    if problem.representation is not SCFRepresentation.FOCK:
        raise ValueError("SOSCF requires a Fock working representation.")
    settings = problem.inputs.fermi_settings
    if settings is not None and settings.smear:
        raise NotImplementedError(
            "SOSCF requires fixed/Aufbau occupations; active Fermi smearing "
            "is not supported."
        )
    occupation = spectral.occupation
    rounded = occupation.round()
    if not bool(
        torch.allclose(occupation, rounded, atol=occupation_tolerance, rtol=0.0)
    ):
        raise NotImplementedError(
            "SOSCF requires fixed integer spin-channel occupations."
        )
    if bool(torch.any((rounded < 0.0) | (rounded > 1.0))):
        raise NotImplementedError(
            "SOSCF supports only fixed 0/1 spin-channel occupations."
        )
    pair_i, pair_a, difference = _build_rotation_pairs(
        rounded, problem.inputs.orbital_mask
    )
    if pair_i.numel() == 0:
        raise ValueError("SOSCF has no unequal-occupation orbital rotations.")
    return SOSCFOrbitalModel(
        problem,
        spectral.coefficients,
        rounded,
        pair_i,
        pair_a,
        difference,
    )
