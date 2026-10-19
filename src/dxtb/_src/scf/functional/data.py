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
"""Immutable values for the functional SCF map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tad_mctc.units.energy import KELVIN2AU

from dxtb import IndexHelper
from dxtb._src.components.interactions import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.components.interactions.list import (
    _FUNCTIONAL_SCF_INTERACTION_TYPES,
)
from dxtb._src.constants import defaults
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction.filling import FermiStatus, _FermiCoreState
from dxtb.config import ConfigSCF

from .core import CoreIndexData, FunctionalInteractionSet

__all__ = [
    "ChargeLayout",
    "FermiSettings",
    "PotentialLayout",
    "SCFCharges",
    "SCFInputs",
    "SCFMapResult",
    "SCFSpectralResult",
    "SCFPotential",
    "pack_potential",
    "potential_layout",
    "setup_scf_inputs",
    "unpack_potential",
    "validate_functional_scf_interactions",
    "functional_scf_interactions_eligible",
]


@dataclass(frozen=True, eq=False)
class ChargeLayout:
    """Structural shapes of a single-System charge vector."""

    mono_shape: tuple[int, ...]
    dipole_shape: tuple[int, ...] | None = None
    quadrupole_shape: tuple[int, ...] | None = None

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Present component shapes in packed charge order."""
        return tuple(
            shape
            for shape in (
                self.mono_shape,
                self.dipole_shape,
                self.quadrupole_shape,
            )
            if shape is not None
        )


@dataclass(frozen=True, eq=False)
class SCFCharges:
    """Immutable internal charge value."""

    mono: Tensor
    dipole: Tensor | None = None
    quad: Tensor | None = None


@dataclass(frozen=True, eq=False)
class PotentialLayout:
    """Structural shapes of a packed single-System potential."""

    mono_shape: tuple[int, ...]
    dipole_shape: tuple[int, ...] | None = None
    quadrupole_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.quadrupole_shape is not None and self.dipole_shape is None:
            raise ValueError("Quadrupole potential requires a dipole row.")

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Present component shapes in legacy packed order."""
        return tuple(
            shape
            for shape in (
                self.mono_shape,
                self.dipole_shape,
                self.quadrupole_shape,
            )
            if shape is not None
        )


@dataclass(frozen=True, eq=False)
class FermiSettings:
    """Resolved finite-temperature occupation settings for one SCF call."""

    kt: Tensor
    smear: bool
    maxiter: int
    threshold: Tensor
    diff_order: int
    partition: int


@dataclass(frozen=True, eq=False)
class SCFPotential:
    """Immutable internal potential value."""

    mono: Tensor | None
    dipole: Tensor | None = None
    quad: Tensor | None = None


@dataclass(frozen=True, eq=False)
class SCFInputs:
    """Immutable tensor and structural inputs to the single-System core.

    Legacy setup/component containers are converted once at the setup
    boundary. No IndexHelper, mutable interaction list, or cache is retained.
    Solver and convergence configuration is validated during setup and is not
    retained here.
    """

    numbers: Tensor
    index: CoreIndexData
    integrals: IntegralMatrices
    n0: Tensor
    nel: Tensor
    occupation: Tensor
    interactions: FunctionalInteractionSet
    charge_layout: ChargeLayout
    inverse_cholesky: Tensor
    orbital_mask: Tensor
    fermi_settings: FermiSettings | None


@dataclass(frozen=True, eq=False)
class SCFMapResult:
    """Physical outputs from one raw representation-map evaluation.

    ``mapped_x`` is the raw output in the selected solver representation.
    ``driving_potential`` is the potential supplied to a charge- or
    potential-driven spectral evaluation. It is structurally absent for a
    Fock-driven evaluation. ``spectral`` owns the Hamiltonian, density,
    charges, orbital energies, coefficients, occupation and status produced
    by that evaluation.
    """

    mapped_x: Tensor
    spectral: SCFSpectralResult
    driving_potential: SCFPotential | None = None

    @property
    def charges(self) -> SCFCharges:
        """Charges produced by this map's spectral evaluation."""
        return self.spectral.charges

    @property
    def hamiltonian(self) -> Tensor:
        """Hamiltonian from the map's spectral evaluation."""
        return self.spectral.hamiltonian

    @property
    def density(self) -> Tensor:
        """Density from the map's spectral evaluation."""
        return self.spectral.density

    @property
    def emo(self) -> Tensor:
        """Orbital energies from the map's spectral evaluation."""
        return self.spectral.emo

    @property
    def coefficients(self) -> Tensor:
        """Coefficients from the map's spectral evaluation."""
        return self.spectral.coefficients

    @property
    def occupation(self) -> Tensor:
        """Occupation from the map's spectral evaluation."""
        return self.spectral.occupation

    @property
    def orbital_energy(self) -> Tensor:
        """Orbital energy from the map's spectral evaluation."""
        return self.spectral.orbital_energy

    @property
    def fermi_status(self) -> FermiStatus | None:
        """Fermi search status for the map's spectral evaluation."""
        return self.spectral.fermi_status


@dataclass(frozen=True, eq=False)
class SCFSpectralResult:
    """One Hamiltonian-driven physical spectral evaluation."""

    hamiltonian: Tensor
    density: Tensor
    emo: Tensor
    coefficients: Tensor
    occupation: Tensor
    charges: SCFCharges
    orbital_energy: Tensor
    fermi_status: FermiStatus | None = None
    _fermi_state: _FermiCoreState | None = None


def _single_system_layout(
    numbers: Tensor, integrals: IntegralMatrices
) -> ChargeLayout:
    """Resolve charge shapes from fixed composition and available integrals."""
    nao = integrals.hcore.shape[-1]
    mono_shape = (nao,)
    dipole_shape = None
    quadrupole_shape = None
    if integrals.dipole is not None:
        dipole_shape = (numbers.shape[-1], integrals.dipole.shape[-3])
    if integrals.quadrupole is not None:
        quadrupole_shape = (
            numbers.shape[-1],
            integrals.quadrupole.shape[-3],
        )
    return ChargeLayout(mono_shape, dipole_shape, quadrupole_shape)


def setup_scf_inputs(
    numbers: Tensor,
    integrals: IntegralMatrices,
    n0: Tensor,
    occupation: Tensor,
    nel: Tensor,
    interactions: InteractionList,
    interaction_data: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    *,
    core_index: CoreIndexData | None = None,
    core_index_owner: object | None = None,
) -> SCFInputs:
    """Build immutable core values from one legacy single-System setup.

    ``core_index`` and ``core_index_owner`` are reserved for a trusted index
    captured by the active System integral setup. The owner must be the
    compatible System façade whose active exact core owns the supplied map.
    Its map values are established at setup; this function validates owner,
    IndexHelper provenance, shape, dtype, and device without synchronizing
    tensor values during SCF preparation. Direct and legacy callers should
    omit both and retain the isolated IndexHelper snapshot.
    """
    if config.fermi.etemp < 0.0:
        raise ValueError("Electronic temperature must be non-negative.")
    if config.fermi.maxiter < 0:
        raise ValueError("Functional Fermi maxiter must be non-negative.")
    validate_functional_scf_interactions(interactions)
    if numbers.ndim != 1 or ihelp.batch_mode != 0:
        raise ValueError("E4.0 SCFInputs require one unbatched System.")
    if integrals.hcore is None or integrals.overlap is None:
        raise ValueError("SCFInputs require hcore and overlap integrals.")
    if core_index is not None:
        if type(core_index) is not CoreIndexData:
            raise TypeError("Captured SCF index must be CoreIndexData.")
        active_core = getattr(
            core_index_owner, "_active_exact_core", lambda: None
        )()
        compatibility = getattr(core_index_owner, "_exact_compatibility", None)
        if (
            active_core is None
            or active_core.index is not core_index
            or active_core.numbers is not numbers
            or compatibility is None
            or compatibility.index_helper_identity != id(ihelp)
        ):
            raise ValueError(
                "Captured SCF index does not belong to this System setup."
            )
        if (
            core_index.shells_to_atom.shape != ihelp.shells_to_atom.shape
            or core_index.orbitals_to_shell.shape
            != ihelp.orbitals_to_shell.shape
            or core_index.shells_to_atom.dtype != ihelp.shells_to_atom.dtype
            or core_index.orbitals_to_shell.dtype
            != ihelp.orbitals_to_shell.dtype
            or core_index.shells_to_atom.device != ihelp.shells_to_atom.device
            or core_index.orbitals_to_shell.device
            != ihelp.orbitals_to_shell.device
        ):
            raise ValueError(
                "Captured SCF index maps must match the IndexHelper shape, "
                "dtype, and device."
            )
    elif core_index_owner is not None:
        raise ValueError("A captured SCF index owner requires its index map.")

    orbital_mask = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)
    orbital_mask = orbital_mask.to(dtype=torch.bool)
    padding_diagonal = (~orbital_mask).to(dtype=integrals.overlap.dtype)
    overlap = integrals.overlap + torch.diag_embed(padding_diagonal)

    from tad_mctc import storch

    inverse_cholesky = storch.inv_cholesky_factor(overlap, is_posdef=True)
    from .interactions import prepare_functional_interactions

    core_interactions = prepare_functional_interactions(
        interactions,
        interaction_data,
        numbers,
        d4sc_setup=getattr(interaction_data, "d4sc_setup", None),
        kind_by_label=getattr(interaction_data, "functional_kinds", None),
    )
    kt_value = config.fermi.etemp * KELVIN2AU
    if kt_value == 0.0:
        fermi_settings = None
    else:
        if config.fermi.thresh is None:
            eps = torch.finfo(integrals.hcore.dtype).eps
            threshold = integrals.hcore.new_tensor(
                min(eps**0.5, 1e5 * eps, 1e-4)
            )
        else:
            threshold = torch.as_tensor(
                config.fermi.thresh,
                device=integrals.hcore.device,
                dtype=integrals.hcore.dtype,
            )
        fermi_settings = FermiSettings(
            kt=integrals.hcore.new_tensor(kt_value),
            smear=kt_value >= 3.0e-7,
            maxiter=config.fermi.maxiter,
            threshold=threshold,
            diff_order=config.fermi.diff_order,
            partition=config.fermi.partition,
        )

    return SCFInputs(
        numbers=numbers,
        index=(
            core_index
            if core_index is not None
            else CoreIndexData(
                # Direct and legacy callers own an isolated structural
                # snapshot so later IndexHelper mutation cannot change it.
                shells_to_atom=ihelp.shells_to_atom.clone(),
                orbitals_to_shell=ihelp.orbitals_to_shell.clone(),
            )
        ),
        integrals=integrals,
        n0=n0,
        nel=nel,
        occupation=occupation,
        interactions=core_interactions,
        charge_layout=_single_system_layout(numbers, integrals),
        inverse_cholesky=inverse_cholesky,
        orbital_mask=orbital_mask,
        fermi_settings=fermi_settings,
    )


def validate_functional_scf_interactions(
    interactions: InteractionList,
) -> None:
    """Reject interaction extensions without a functional-purity contract.

    The E4.0 raw map is guaranteed pure for exact migrated built-in
    interactions. Arbitrary subclasses remain supported by the legacy SCF
    route and must not be admitted to future functional SCF execution.
    """
    eligible, names = functional_scf_interactions_eligible(interactions)
    if not eligible:
        raise TypeError(
            "Functional SCF currently supports exact migrated built-in "
            f"interactions only; unsupported extension(s): {names}."
        )


def functional_scf_interactions_eligible(
    interactions: InteractionList,
) -> tuple[bool, str]:
    """Check exact interaction types before functional setup begins."""
    unsupported = [
        component
        for component in interactions.components
        if type(component) not in _FUNCTIONAL_SCF_INTERACTION_TYPES
    ]
    names = ", ".join(
        f"{type(component).__module__}.{type(component).__qualname__}"
        for component in unsupported
    )
    return not unsupported, names


def unpack_charges(x: Tensor, layout: ChargeLayout) -> SCFCharges:
    """Unpack fixed-shape single-System charge rows without deflation."""
    shapes = layout.shapes
    if x.ndim != 2 or x.shape[0] != len(shapes):
        raise ValueError(
            "Packed charge iterate does not match the configured charge layout."
        )
    lengths = tuple(_numel(shape) for shape in shapes)
    expected_width = max(lengths)
    if x.shape[-1] != expected_width:
        raise ValueError(
            f"Packed charge width {x.shape[-1]} does not match {expected_width}."
        )

    values = [
        x[index, :length].reshape(shape)
        for index, (length, shape) in enumerate(zip(lengths, shapes))
    ]
    return SCFCharges(
        mono=values[0],
        dipole=(values[1] if layout.dipole_shape is not None else None),
        quad=(values[-1] if layout.quadrupole_shape is not None else None),
    )


def pack_charges(charges: SCFCharges, layout: ChargeLayout) -> Tensor:
    """Pack fixed-shape charges using the legacy PADNZ row convention."""
    values: list[Tensor] = [charges.mono.reshape(-1)]
    if layout.dipole_shape is not None:
        if charges.dipole is None:
            raise ValueError("Dipole charge is required by this layout.")
        values.append(charges.dipole.reshape(-1))
    if layout.quadrupole_shape is not None:
        if charges.quad is None:
            raise ValueError("Quadrupole charge is required by this layout.")
        values.append(charges.quad.reshape(-1))

    width = max(value.shape[-1] for value in values)
    rows = [
        torch.cat(
            (
                value,
                value.new_full((width - value.shape[-1],), defaults.PADNZ),
            )
        )
        for value in values
    ]
    return torch.stack(rows)


def pack_potential(potential: SCFPotential, layout: PotentialLayout) -> Tensor:
    """Pack fixed-shape potential rows using the legacy PADNZ convention."""
    values = _validated_components(potential, layout)
    width = max(value.shape[-1] for value in values)
    rows = [
        torch.cat(
            (
                value.reshape(-1),
                value.new_full((width - value.numel(),), defaults.PADNZ),
            )
        )
        for value in values
    ]
    return torch.stack(rows)


def unpack_potential(x: Tensor, layout: PotentialLayout) -> SCFPotential:
    """Unpack fixed structural potential rows without value-based sizing."""
    shapes = layout.shapes
    lengths = tuple(_numel(shape) for shape in shapes)
    expected_width = max(lengths)
    if x.ndim != 2 or x.shape[0] != len(shapes):
        raise ValueError(
            "Packed potential iterate does not match the configured layout."
        )
    if x.shape[-1] != expected_width:
        raise ValueError(
            f"Packed potential width {x.shape[-1]} does not match "
            f"{expected_width}."
        )
    values = [
        x[index, :length].reshape(shape)
        for index, (length, shape) in enumerate(zip(lengths, shapes))
    ]
    return SCFPotential(
        mono=values[0],
        dipole=(values[1] if layout.dipole_shape is not None else None),
        quad=(values[-1] if layout.quadrupole_shape is not None else None),
    )


def potential_layout(potential: SCFPotential) -> PotentialLayout:
    """Freeze component shapes from one call-local physical potential."""
    if potential.mono is None:
        raise ValueError("A functional potential requires a monopole row.")
    return PotentialLayout(
        tuple(potential.mono.shape),
        None if potential.dipole is None else tuple(potential.dipole.shape),
        None if potential.quad is None else tuple(potential.quad.shape),
    )


def _validated_components(
    potential: SCFPotential, layout: PotentialLayout
) -> list[Tensor]:
    components = (potential.mono, potential.dipole, potential.quad)
    shapes = (layout.mono_shape, layout.dipole_shape, layout.quadrupole_shape)
    values: list[Tensor] = []
    for name, value, shape in zip(
        ("mono", "dipole", "quad"), components, shapes
    ):
        if shape is None:
            if value is not None:
                raise ValueError(f"Unexpected {name} potential component.")
            continue
        if value is None:
            raise ValueError(f"{name.capitalize()} potential is required.")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{name.capitalize()} potential shape {tuple(value.shape)} "
                f"does not match {shape}."
            )
        values.append(value.reshape(-1))
    return values


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result
