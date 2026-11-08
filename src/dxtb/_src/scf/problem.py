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

from .data import PotentialLayout, SCFCharges, SCFInputs, SCFMapResult
from .map import (
    fock_map,
    potential_from_charges,
    potential_map,
    potential_to_hamiltonian,
    scf_map,
)

__all__ = [
    "convert_physical_state_to_fock",
    "SCFPhysicalState",
    "SCFProblem",
    "SCFVariable",
]


class SCFVariable(Enum):
    """Working variable used by an existing physical SCF map."""

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
    """Fixed physical inputs and variable-specific map structure."""

    inputs: SCFInputs
    variable: SCFVariable
    potential_layout: PotentialLayout | None = None

    def __post_init__(self) -> None:
        if self.variable is SCFVariable.POTENTIAL:
            if self.potential_layout is None:
                raise ValueError("Potential SCF requires a PotentialLayout.")
        elif self.potential_layout is not None:
            raise ValueError(
                "PotentialLayout is only valid for potential variable."
            )

    def map(self, x: Tensor) -> SCFMapResult:
        """Evaluate the authoritative raw map for this variable."""
        if self.variable is SCFVariable.CHARGE:
            return scf_map(x, self.inputs)
        if self.variable is SCFVariable.POTENTIAL:
            assert self.potential_layout is not None
            return potential_map(x, self.inputs, self.potential_layout)
        if self.variable is SCFVariable.FOCK:
            return fock_map(x, self.inputs)
        raise ValueError(
            f"Unsupported SCF variable {self.variable!r}."
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
        """Remove structural padding from a variable residual."""
        inputs = self.inputs
        orbital_mask = inputs.orbital_mask
        atom_mask = inputs.numbers != 0

        if self.variable is SCFVariable.FOCK:
            pair_mask = orbital_mask[..., :, None] & orbital_mask[..., None, :]
            return torch.where(pair_mask, delta, 0.0)

        if self.variable is SCFVariable.CHARGE:
            return _mask_packed_rows(
                delta,
                inputs.charge_layout.shapes,
                orbital_mask=orbital_mask,
                atom_mask=atom_mask,
                kind="charge",
                pad_value=0.0,
            )

        assert self.potential_layout is not None
        return _mask_packed_rows(
            delta,
            self.potential_layout.shapes,
            orbital_mask=orbital_mask,
            atom_mask=atom_mask,
            kind="potential",
            pad_value=0.0,
        )


def convert_physical_state_to_fock(
    problem: SCFProblem, state: SCFPhysicalState
) -> tuple[SCFProblem, SCFPhysicalState]:
    """Convert one compact observation to its exactly paired Fock state.

    The Fock input is the Hamiltonian that generated the observation's
    density. Its mapped output is rebuilt from that density's charges. The
    conversion performs no spectrum evaluation or physical map.
    """
    if problem.variable is SCFVariable.FOCK:
        return problem, state
    if problem.variable not in (
        SCFVariable.CHARGE,
        SCFVariable.POTENTIAL,
    ):
        raise NotImplementedError(
            f"Cannot convert {problem.variable.value} to Fock."
        )

    fock_problem = SCFProblem(problem.inputs, SCFVariable.FOCK)
    spectral = state.map_result.spectral
    paired_input = spectral.hamiltonian
    fresh_fock = fock_problem.rebuild_hamiltonian(spectral.charges)
    fock_result = SCFMapResult(
        mapped_x=fresh_fock,
        spectral=spectral,
        driving_potential=None,
    )
    return fock_problem, SCFPhysicalState(paired_input, fock_result)


def _mask_packed_rows(
    value: Tensor,
    shapes: tuple[tuple[int, ...], ...],
    *,
    orbital_mask: Tensor,
    atom_mask: Tensor,
    kind: str,
    pad_value: float,
) -> Tensor:
    """Mask Padding orbitals/atoms and row tails without rescaling multipoles."""
    lengths = tuple(_shape_numel(shape) for shape in shapes)
    expected_width = max(lengths)
    if value.ndim != 2 or value.shape[0] != len(shapes):
        if kind == "charge":
            raise ValueError(
                "Packed charge iterate does not match the configured charge layout."
            )
        raise ValueError(
            "Packed potential iterate does not match the configured layout."
        )
    if value.shape[-1] != expected_width:
        raise ValueError(
            f"Packed {kind} width {value.shape[-1]} does not match "
            f"{expected_width}."
        )
    rows: list[Tensor] = []
    for index, (length, shape) in enumerate(zip(lengths, shapes)):
        active = value[index, :length].reshape(shape)
        if index == 0:
            masked = torch.where(orbital_mask, active, 0.0)
        else:
            mask_shape = (
                *atom_mask.shape,
                *((1,) * (active.ndim - atom_mask.ndim)),
            )
            masked = torch.where(atom_mask.reshape(mask_shape), active, 0.0)
        rows.append(
            torch.cat(
                (
                    masked.reshape(-1),
                    value.new_full((expected_width - length,), pad_value),
                )
            )
        )
    return torch.stack(rows)


def _shape_numel(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size
