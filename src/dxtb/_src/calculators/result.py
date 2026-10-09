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
"""
Calculators: Result
===================

Immutable values produced by a single-point calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

from dxtb import OutputHandler
from dxtb._src.components.interactions import Charges, Potential
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Tensor

__all__ = ["ChargeResult", "Multipoles", "PotentialResult", "Result"]


@dataclass(frozen=True, eq=False)
class ChargeResult:
    """Immutable charge tensors returned by a calculation."""

    mono: Tensor
    dipole: Tensor | None = None
    quadrupole: Tensor | None = None


@dataclass(frozen=True, eq=False)
class PotentialResult:
    """Immutable potential tensors returned by a calculation."""

    mono: Tensor | None = None
    dipole: Tensor | None = None
    quadrupole: Tensor | None = None


@dataclass(frozen=True, eq=False)
class Multipoles:
    """Immutable dipole and quadrupole charge outputs."""

    dipole: Tensor | None = None
    quadrupole: Tensor | None = None


@dataclass(frozen=True, eq=False)
class Result:
    """Immutable structure for single-system or legacy-batch outputs.

    SCF currently exposes iteration count but no reliable convergence flag or
    residual value, so those two planned fields remain ``None``. Fields are
    never rebound by evaluation, and later evaluations do not mutate a prior
    Result. Tensor values remain ordinary PyTorch tensors, so callers can still
    mutate them in place. Charge and potential outputs use plain immutable
    values whose tensors remain graph-connected.
    """

    energy: Tensor
    scf: Tensor | None
    classical: tuple[tuple[str, Tensor], ...]
    fenergy: Tensor | None
    iterations: Tensor
    charges: ChargeResult | None = None
    density: Tensor | None = None
    coefficients: Tensor | None = None
    emo: Tensor | None = None
    occupation: Tensor | None = None
    potential: PotentialResult | None = None
    hamiltonian: Tensor | None = None
    overlap: Tensor | None = None
    hcore: Tensor | None = None
    dipole_integrals: Tensor | None = None
    quadrupole_integrals: Tensor | None = None
    overlap_norm: Tensor | None = None
    converged: Tensor | None = None
    residual: Tensor | None = None

    @classmethod
    def snapshot(
        cls,
        *,
        energy: Tensor,
        scf: Tensor | None,
        classical: tuple[tuple[str, Tensor], ...],
        fenergy: Tensor | None,
        iterations: Tensor,
        charges: Charges | None = None,
        density: Tensor | None = None,
        coefficients: Tensor | None = None,
        emo: Tensor | None = None,
        occupation: Tensor | None = None,
        potential: Potential | None = None,
        hamiltonian: Tensor | None = None,
        overlap: Tensor | None = None,
        hcore: Tensor | None = None,
        dipole_integrals: Tensor | None = None,
        quadrupole_integrals: Tensor | None = None,
        overlap_norm: Tensor | None = None,
        converged: Tensor | None = None,
        residual: Tensor | None = None,
    ) -> Result:
        """Construct the core Result from call-local evaluation outputs.

        Core evaluation must use this factory. Numeric tensors are retained
        directly because their producers create per-call outputs and do not
        mutate them later; this preserves autograd without copying O(nao²)
        values. Classical energies are stored as an immutable tuple.
        """
        result_charges = (
            None
            if charges is None
            else ChargeResult(
                mono=charges.mono,
                dipole=charges.dipole,
                quadrupole=charges.quad,
            )
        )
        result_potential = (
            None
            if potential is None
            else PotentialResult(
                mono=potential.mono,
                dipole=potential.dipole,
                quadrupole=potential.quad,
            )
        )
        return cls(
            energy=energy,
            scf=scf,
            classical=tuple(classical),
            fenergy=fenergy,
            iterations=iterations,
            charges=result_charges,
            density=density,
            coefficients=coefficients,
            emo=emo,
            occupation=occupation,
            potential=result_potential,
            hamiltonian=hamiltonian,
            overlap=overlap,
            hcore=hcore,
            dipole_integrals=dipole_integrals,
            quadrupole_integrals=quadrupole_integrals,
            overlap_norm=overlap_norm,
            converged=converged,
            residual=residual,
        )

    @property
    def total(self) -> Tensor:
        """Compatibility alias for the canonical total-energy field."""
        return self.energy

    @property
    def iter(self) -> Tensor:
        """Compatibility alias for the per-system iteration tensor."""
        return self.iterations

    @property
    def multipoles(self) -> Multipoles | None:
        """Expose charge multipoles without duplicating their tensors."""
        if self.charges is None:
            return None
        return Multipoles(
            dipole=self.charges.dipole,
            quadrupole=self.charges.quadrupole,
        )

    @property
    def cenergies(self) -> dict[str, Tensor]:
        """Materialize the immutable classical-energy pairs as a new dict."""
        return dict(self.classical)

    @property
    def integrals(self) -> IntegralMatrices | None:
        """Return the immutable integral matrices when they were evaluated."""
        if self.hcore is None or self.overlap is None:
            return None
        return IntegralMatrices(
            hcore=self.hcore,
            overlap=self.overlap,
            dipole=self.dipole_integrals,
            quadrupole=self.quadrupole_integrals,
        )

    def get_energies(self) -> dict[str, dict[str, object]]:
        """Return energy contributions formatted for user-facing output."""
        key = "value"
        classical = {
            name: {key: value.sum().item()} for name, value in self.classical
        }
        ctotal = sum(value[key] for value in classical.values())
        scf_energy = 0.0 if self.scf is None else self.scf.sum().item()
        free_energy = 0.0 if self.fenergy is None else self.fenergy.sum().item()
        return {
            "total": {key: self.energy.sum().item()},
            "Classical": {key: ctotal, "sub": classical},
            "Electronic": {
                key: scf_energy + free_energy,
                "sub": {
                    "SCF": {key: scf_energy},
                    "Free Energy (Fermi)": {key: free_energy},
                },
            },
        }

    def print_energies(self, v: int = 4, precision: int = 14) -> None:
        """Print energy contributions in a table."""
        OutputHandler.write_table(
            self.get_energies(),
            title="Energies",
            columns=["Contribution", "Energy (Eh)"],
            v=v,
            precision=precision,
        )
