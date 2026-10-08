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
from dxtb._src.components.interactions.container import Container
from dxtb._src.constants import defaults
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import ContainerData, Tensor

__all__ = ["Multipoles", "Result"]


class _FrozenCharges(Charges):
    """Read-only structural wrapper for call-local SCF charges.

    The tensors are already call-local SCF outputs, so retaining them avoids
    duplicating large values while preserving their autograd graph. Mutation
    methods inherited from :class:`Charges` are intentionally unavailable.
    """

    __slots__ = ("_locked",)

    def __init__(self, value: Charges):
        self._locked = False
        super().__init__(
            mono=value.mono,
            dipole=value.dipole,
            quad=value.quad,
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

    def nullify_padding(self, pad: int = defaults.PADNZ) -> None:
        del pad
        raise AttributeError("Result charge values cannot be mutated.")

    def __iadd__(self, other: Container) -> _FrozenCharges:
        del other
        raise AttributeError("Result charge values cannot be mutated.")

    def __add__(self, other: Container) -> Container:
        if not isinstance(other, Container):
            raise TypeError("Only containers can be added together.")
        if other.batch_mode != self.batch_mode:
            raise ValueError("Cannot add containers with different batch modes.")
        return Container(
            mono=self.add_tensors(self.mono, other.mono),
            dipole=self.add_tensors(self.dipole, other.dipole),
            quad=self.add_tensors(self.quad, other.quad),
            label=[*self.label, *other.label],
            batch_mode=self.batch_mode,
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        data: ContainerData,
        batch_mode: int = 0,
        pad: int = defaults.PADNZ,
    ) -> _FrozenCharges:
        del tensor, data, batch_mode, pad
        raise TypeError(
            "Result charge values cannot be constructed by mutation."
        )


class _FrozenPotential(Potential):
    """Read-only structural wrapper for call-local SCF potential values."""

    __slots__ = ("_locked",)

    def __init__(self, value: Potential):
        self._locked = False
        super().__init__(
            mono=value.mono,
            dipole=value.dipole,
            quad=value.quad,
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

    def nullify_padding(self, pad: int = defaults.PADNZ) -> None:
        del pad
        raise AttributeError("Result potential values cannot be mutated.")

    def reset(self) -> None:
        raise AttributeError("Result potential values cannot be mutated.")

    def __iadd__(self, other: Container) -> _FrozenPotential:
        del other
        raise AttributeError("Result potential values cannot be mutated.")

    def __add__(self, other: Container) -> Container:
        if not isinstance(other, Container):
            raise TypeError("Only containers can be added together.")
        if other.batch_mode != self.batch_mode:
            raise ValueError("Cannot add containers with different batch modes.")
        return Container(
            mono=self.add_tensors(self.mono, other.mono),
            dipole=self.add_tensors(self.dipole, other.dipole),
            quad=self.add_tensors(self.quad, other.quad),
            label=[*self.label, *other.label],
            batch_mode=self.batch_mode,
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        data: ContainerData,
        batch_mode: int = 0,
        pad: int = defaults.PADNZ,
    ) -> _FrozenPotential:
        del tensor, data, batch_mode, pad
        raise TypeError(
            "Result potential values cannot be constructed by mutation."
        )


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
    mutate them in place. Charges and Potential are temporary B5.5 compatibility
    wrappers around call-local SCF outputs.
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
        """Construct the core Result from call-local evaluation outputs.

        Core evaluation must use this factory. Numeric tensors are retained
        directly because their producers create per-call outputs and do not
        mutate them later; this preserves autograd without copying O(nao²)
        values. Classical energies are stored as an immutable tuple.
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
                dipole=multipoles.dipole,
                quadrupole=multipoles.quadrupole,
            )
        else:
            frozen_multipoles = None
        return cls(
            energy=energy,
            scf=scf,
            classical=tuple(classical),
            fenergy=fenergy,
            iterations=iterations,
            charges=frozen_charges,
            multipoles=frozen_multipoles,
            density=density,
            coefficients=coefficients,
            emo=emo,
            occupation=occupation,
            potential=frozen_potential,
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
