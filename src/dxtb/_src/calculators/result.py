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

import torch

from dxtb import OutputHandler
from dxtb._src.components.interactions import Charges, Potential
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Tensor

__all__ = ["Multipoles", "Result"]


def _copy_tensor(value: Tensor | None) -> Tensor | None:
    return None if value is None else value.clone()


class _FrozenCharges(Charges):
    """Read-only snapshot of the mutable SCF charge container."""

    __slots__ = ("_locked",)

    def __init__(self, value: Charges):
        self._locked = False
        super().__init__(
            mono=_copy_tensor(value.mono),
            dipole=_copy_tensor(value.dipole),
            quad=_copy_tensor(value.quad),
            label=list(value.label),
            batch_mode=value.batch_mode,
        )
        self.label = tuple(self.label)
        self._locked = True

    def __setattr__(self, name: str, value: object) -> None:
        if not getattr(self, "_locked", False):
            super().__setattr__(name, value)
            return
        raise AttributeError("Result charge values are immutable.")


class _FrozenPotential(Potential):
    """Read-only snapshot of the mutable SCF potential container."""

    __slots__ = ("_locked",)

    def __init__(self, value: Potential):
        self._locked = False
        super().__init__(
            mono=_copy_tensor(value.mono),
            dipole=_copy_tensor(value.dipole),
            quad=_copy_tensor(value.quad),
            label=list(value.label),
            batch_mode=value.batch_mode,
        )
        self.label = tuple(self.label)
        self._locked = True

    def __setattr__(self, name: str, value: object) -> None:
        if not getattr(self, "_locked", False):
            super().__setattr__(name, value)
            return
        raise AttributeError("Result potential values are immutable.")


@dataclass(frozen=True, eq=False)
class Multipoles:
    """Immutable dipole and quadrupole charge outputs."""

    dipole: Tensor | None = None
    quadrupole: Tensor | None = None


@dataclass(frozen=True, eq=False)
class Result:
    """Immutable single-system or legacy-batch calculation output.

    SCF currently exposes iteration count but no reliable convergence flag or
    residual value, so those two planned fields remain ``None``.
    """

    energy: Tensor
    scf: Tensor | None
    classical: tuple[tuple[str, Tensor], ...]
    fenergy: Tensor | None
    iterations: Tensor
    charges: Charges | None = None
    multipoles: Multipoles | None = None
    density: Tensor | None = None
    coefficients: Tensor | None = None
    emo: Tensor | None = None
    occupation: Tensor | None = None
    potential: Potential | None = None
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
        multipoles: Multipoles | None = None,
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
        """Create a Result with independent tensor snapshots.

        ``clone`` preserves autograd connectivity while keeping the result
        independent of mutable SCF and component output containers.
        """
        frozen_charges = None if charges is None else _FrozenCharges(charges)
        frozen_potential = (
            None if potential is None else _FrozenPotential(potential)
        )
        if frozen_charges is not None:
            frozen_multipoles = Multipoles(
                dipole=frozen_charges.dipole,
                quadrupole=frozen_charges.quad,
            )
        elif multipoles is not None:
            frozen_multipoles = Multipoles(
                dipole=_copy_tensor(multipoles.dipole),
                quadrupole=_copy_tensor(multipoles.quadrupole),
            )
        else:
            frozen_multipoles = None
        return cls(
            energy=energy.clone(),
            scf=_copy_tensor(scf),
            classical=tuple((name, value.clone()) for name, value in classical),
            fenergy=_copy_tensor(fenergy),
            iterations=iterations.clone(),
            charges=frozen_charges,
            multipoles=frozen_multipoles,
            density=_copy_tensor(density),
            coefficients=_copy_tensor(coefficients),
            emo=_copy_tensor(emo),
            occupation=_copy_tensor(occupation),
            potential=frozen_potential,
            hamiltonian=_copy_tensor(hamiltonian),
            overlap=_copy_tensor(overlap),
            hcore=_copy_tensor(hcore),
            dipole_integrals=_copy_tensor(dipole_integrals),
            quadrupole_integrals=_copy_tensor(quadrupole_integrals),
            overlap_norm=_copy_tensor(overlap_norm),
            converged=_copy_tensor(converged),
            residual=_copy_tensor(residual),
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
            name: {key: value.sum().item()}
            for name, value in self.classical
        }
        ctotal = sum(value[key] for value in classical.values())
        scf_energy = 0.0 if self.scf is None else self.scf.sum().item()
        free_energy = (
            0.0 if self.fenergy is None else self.fenergy.sum().item()
        )
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
