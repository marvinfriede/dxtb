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

from dataclasses import dataclass, fields
from typing import Protocol

from torch.utils import _pytree

from dxtb import OutputHandler
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf.status import SCFStatus
from dxtb._src.typing import Tensor

__all__ = ["ChargeResult", "Multipoles", "PotentialResult", "Result"]


class ChargeLike(Protocol):
    """Structural charge values accepted by Result.snapshot."""

    @property
    def mono(self) -> Tensor: ...

    @property
    def dipole(self) -> Tensor | None: ...

    @property
    def quad(self) -> Tensor | None: ...


class PotentialLike(Protocol):
    """Structural potential values accepted by Result.snapshot."""

    @property
    def mono(self) -> Tensor | None: ...

    @property
    def dipole(self) -> Tensor | None: ...

    @property
    def quad(self) -> Tensor | None: ...


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

    Functional full-SCF routes provide convergence, residual and solver-status
    tensors; legacy fallback routes may leave those fields ``None``. Fields are
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
    status: SCFStatus | None = None

    @classmethod
    def snapshot(
        cls,
        *,
        energy: Tensor,
        scf: Tensor | None,
        classical: tuple[tuple[str, Tensor], ...],
        fenergy: Tensor | None,
        iterations: Tensor,
        charges: ChargeLike | None = None,
        density: Tensor | None = None,
        coefficients: Tensor | None = None,
        emo: Tensor | None = None,
        occupation: Tensor | None = None,
        potential: PotentialLike | None = None,
        hamiltonian: Tensor | None = None,
        overlap: Tensor | None = None,
        hcore: Tensor | None = None,
        dipole_integrals: Tensor | None = None,
        quadrupole_integrals: Tensor | None = None,
        overlap_norm: Tensor | None = None,
        converged: Tensor | None = None,
        residual: Tensor | None = None,
        status: SCFStatus | None = None,
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
            status=status,
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


def _flatten_optional_dataclass(
    value: object,
) -> tuple[list[object], tuple[str, ...]]:
    """Flatten declared, present fields of one immutable result value."""
    active_names = tuple(
        field.name
        for field in fields(value)
        if getattr(value, field.name) is not None
    )
    return [getattr(value, name) for name in active_names], active_names


def _flatten_optional_dataclass_with_keys(
    value: object,
) -> tuple[list[tuple[_pytree.KeyEntry, object]], tuple[str, ...]]:
    """Flatten present fields with stable dataclass field paths."""
    children, active_names = _flatten_optional_dataclass(value)
    return [
        (_pytree.GetAttrKey(name), child)
        for name, child in zip(active_names, children)
    ], active_names


def _unflatten_optional_dataclass(
    cls: type,
    children: object,
    active_names: tuple[str, ...],
) -> object:
    """Rebuild an immutable result value without inspecting its leaves."""
    values = {field.name: None for field in fields(cls)}
    values.update(zip(active_names, children))  # type: ignore[arg-type]
    return cls(**values)


for _result_value_type in (ChargeResult, PotentialResult, Multipoles):
    _pytree.register_pytree_node(
        _result_value_type,
        _flatten_optional_dataclass,
        lambda children, context, cls=_result_value_type: (
            _unflatten_optional_dataclass(cls, children, context)
        ),
        flatten_with_keys_fn=_flatten_optional_dataclass_with_keys,
    )


def _flatten_result(
    result: Result,
) -> tuple[list[object], tuple[tuple[str, ...], tuple[str, ...]]]:
    """Flatten Result tensors, keeping classical labels in static context."""
    active_names: list[str] = []
    children: list[object] = []
    labels = tuple(label for label, _ in result.classical)
    for field in fields(Result):
        value = getattr(result, field.name)
        if value is None:
            continue
        active_names.append(field.name)
        children.append(
            tuple(energy for _, energy in result.classical)
            if field.name == "classical"
            else value
        )
    return children, (tuple(active_names), labels)


def _flatten_result_with_keys(
    result: Result,
) -> tuple[
    list[tuple[_pytree.KeyEntry, object]],
    tuple[tuple[str, ...], tuple[str, ...]],
]:
    """Flatten Result fields with deterministic attribute paths."""
    children, context = _flatten_result(result)
    active_names, _ = context
    return [
        (_pytree.GetAttrKey(name), child)
        for name, child in zip(active_names, children)
    ], context


def _unflatten_result(
    children: object,
    context: tuple[tuple[str, ...], tuple[str, ...]],
) -> Result:
    """Rebuild Result fields and ordered classical label/energy pairs."""
    active_names, labels = context
    values: dict[str, object] = {field.name: None for field in fields(Result)}
    for name, child in zip(active_names, children):  # type: ignore[arg-type]
        values[name] = (
            tuple(zip(labels, child)) if name == "classical" else child
        )
    return Result(**values)  # type: ignore[arg-type]


_pytree.register_pytree_node(
    Result,
    _flatten_result,
    _unflatten_result,
    flatten_with_keys_fn=_flatten_result_with_keys,
)
