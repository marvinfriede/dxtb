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

from dxtb._src.typing import Tensor

from .data import PotentialLayout, SCFInputs, SCFMapResult
from .map import fock_map, potential_map, scf_map

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
