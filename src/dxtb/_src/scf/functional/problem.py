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
"""Physical problem and input-bound physical state for functional SCF."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from dxtb._src.typing import Tensor

from .data import (
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFMapResult,
    pack_charges,
    pack_potential,
    unpack_charges,
    unpack_potential,
)
from .map import (
    fock_map,
    potential_from_charges,
    potential_map,
    potential_to_hamiltonian,
    scf_map,
)

__all__ = [
    "SCFPhysicalState",
    "SCFProblem",
    "SCFRepresentation",
]


class SCFRepresentation(Enum):
    """Working representation used by an existing physical SCF map."""

    CHARGE = "charge"
    POTENTIAL = "potential"
    FOCK = "fock"


@dataclass(frozen=True, eq=False)
class SCFPhysicalState:
    """Pair one raw-map input with its physical evaluation."""

    x: Tensor
    map_result: SCFMapResult

    @property
    def delta(self) -> Tensor:
        """Return the raw fixed-point residual ``G(x) - x``."""
        return self.map_result.mapped_x - self.x


@dataclass(frozen=True, eq=False)
class SCFProblem:
    """Fixed physical inputs and representation-specific map structure."""

    inputs: SCFInputs
    representation: SCFRepresentation
    potential_layout: PotentialLayout | None = None

    def __post_init__(self) -> None:
        if self.representation is SCFRepresentation.POTENTIAL:
            if self.potential_layout is None:
                raise ValueError("Potential SCF requires a PotentialLayout.")
        elif self.potential_layout is not None:
            raise ValueError(
                "PotentialLayout is only valid for potential representation."
            )

    def map(self, x: Tensor) -> SCFMapResult:
        """Evaluate the authoritative raw map for this representation."""
        if self.representation is SCFRepresentation.CHARGE:
            return scf_map(x, self.inputs)
        if self.representation is SCFRepresentation.POTENTIAL:
            assert self.potential_layout is not None
            return potential_map(x, self.inputs, self.potential_layout)
        if self.representation is SCFRepresentation.FOCK:
            return fock_map(x, self.inputs)
        raise ValueError(
            f"Unsupported SCF representation {self.representation!r}."
        )

    def evaluate(self, x: Tensor) -> SCFPhysicalState:
        """Evaluate and bind ``x`` to its corresponding physical map result."""
        return SCFPhysicalState(x=x, map_result=self.map(x))

    def rebuild_hamiltonian(self, charges: SCFCharges) -> Tensor:
        """Rebuild the physical Hamiltonian corresponding to explicit charges."""
        potential = potential_from_charges(charges, self.inputs)
        integrals = self.inputs.integrals
        return potential_to_hamiltonian(
            integrals.hcore,
            integrals.overlap,
            integrals.dipole,
            integrals.quadrupole,
            self.inputs.index,
            potential,
        )

    def project_residual(self, delta: Tensor) -> Tensor:
        """Remove structural padding from a representation residual."""
        inputs = self.inputs
        orbital_mask = inputs.orbital_mask
        atom_mask = inputs.numbers != 0

        if self.representation is SCFRepresentation.FOCK:
            pair_mask = orbital_mask[..., :, None] & orbital_mask[..., None, :]
            return torch.where(pair_mask, delta, 0.0)

        def mask_atoms(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            shape = (*atom_mask.shape, *((1,) * (value.ndim - atom_mask.ndim)))
            return torch.where(atom_mask.reshape(shape), value, 0.0)

        if self.representation is SCFRepresentation.CHARGE:
            charges = unpack_charges(delta, inputs.charge_layout)
            packed = pack_charges(
                SCFCharges(
                    mono=torch.where(orbital_mask, charges.mono, 0.0),
                    dipole=mask_atoms(charges.dipole),
                    quad=mask_atoms(charges.quad),
                ),
                inputs.charge_layout,
            )
            return _zero_row_padding(packed, inputs.charge_layout.shapes)

        assert self.potential_layout is not None
        potential = unpack_potential(delta, self.potential_layout)
        packed = pack_potential(
            type(potential)(
                mono=(
                    torch.where(orbital_mask, potential.mono, 0.0)
                    if potential.mono is not None
                    else None
                ),
                dipole=mask_atoms(potential.dipole),
                quad=mask_atoms(potential.quad),
            ),
            self.potential_layout,
        )
        return _zero_row_padding(packed, self.potential_layout.shapes)


def _zero_row_padding(
    value: Tensor, shapes: tuple[tuple[int, ...], ...]
) -> Tensor:
    lengths = tuple(_shape_numel(shape) for shape in shapes)
    row_lengths = torch.as_tensor(lengths, device=value.device)
    columns = torch.arange(value.shape[-1], device=value.device)
    valid = columns.unsqueeze(0) < row_lengths.unsqueeze(-1)
    return torch.where(valid, value, 0.0)


def _shape_numel(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size
