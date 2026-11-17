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
"""Immutable values for the SCF map."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tad_mctc.units.energy import KELVIN2AU

from dxtb import IndexHelper
from dxtb._src.components.interactions import (
    Interaction,
    InteractionList,
    InteractionListCache,
)
from dxtb._src.components.interactions.base import (
    validate_unique_interaction_labels,
)
from dxtb._src.constants import defaults
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import DeviceError, DtypeError
from dxtb._src.wavefunction.filling import FermiStatus, _FermiCoreState
from dxtb.config import ConfigSCF

from .core import SCFInteractionSet, tensor_leaves

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
    "validate_scf_interactions",
    "scf_interactions_eligible",
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

    Component setup containers are converted once at the setup
    boundary. No mutable interaction list or cache is retained.
    Solver and convergence configuration is validated during setup and is not
    retained here.
    """

    numbers: Tensor
    index: IndexHelper
    integrals: IntegralMatrices
    n0: Tensor
    nel: Tensor
    occupation: Tensor
    interactions: SCFInteractionSet
    charge_layout: ChargeLayout
    inverse_cholesky: Tensor
    orbital_mask: Tensor
    fermi_settings: FermiSettings | None


@dataclass(frozen=True, eq=False)
class SCFMapResult:
    """Physical outputs from one raw variable-map evaluation.

    ``mapped_x`` is the raw output in the selected solver variable.
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


def _is_valid_spin_list(items: list[object]) -> bool:
    """Return whether ``items`` is a list of real non-boolean numbers."""
    return all(
        _is_valid_spin_list(x)
        if isinstance(x, list)
        else (isinstance(x, (int, float)) and not isinstance(x, bool))
        for x in items
    )


def _check_scf_tensor(
    name: str,
    tensor: object,
    dev: torch.device,
    dtype: torch.dtype | None,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    """Validate Tensor type, device, dtype (or integer kind), and shape."""
    if not isinstance(tensor, Tensor):
        raise TypeError(f"SCFInputs '{name}' must be a Tensor.")
    if tensor.device != dev:
        raise DeviceError(
            f"Device mismatch: numbers is on '{dev}', but '{name}' is on "
            f"'{tensor.device}'."
        )
    if dtype is None:
        if (
            tensor.is_floating_point()
            or tensor.is_complex()
            or tensor.dtype is torch.bool
        ):
            raise TypeError(f"SCFInputs '{name}' must have an integer dtype.")
    elif tensor.dtype != dtype:
        raise DtypeError(
            f"Dtype mismatch: expected '{dtype}', but '{name}' has "
            f"dtype '{tensor.dtype}'."
        )
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"SCFInputs '{name}' shape {tuple(tensor.shape)} does not match "
            f"{expected_shape}."
        )


def _validate_scf_preamble(
    numbers: Tensor,
    ihelp: IndexHelper,
    *,
    integrals: IntegralMatrices | None = None,
    positions: Tensor | None = None,
    refocc: Tensor | None = None,
    chrg: object = None,
    spin: object = None,
) -> tuple[torch.device, torch.dtype, tuple[int, ...], int, int, int]:
    """Validate SCF numbers, IndexHelper, and explicit input tensors."""
    if (
        not isinstance(numbers, Tensor)
        or numbers.ndim < 1
        or numbers.numel() == 0
    ):
        raise ValueError("SCFInputs require a non-empty atomic-number tensor.")
    dev = numbers.device
    _check_scf_tensor("numbers", numbers, dev, None)
    if not isinstance(ihelp, IndexHelper):
        raise TypeError("SCFInputs requires an IndexHelper.")
    if any(
        not isinstance(getattr(ihelp, name), Tensor)
        for name in ihelp._child_names
    ):
        raise TypeError("IndexHelper fields must be torch.Tensor instances.")
    if not ihelp.has_uniform_device(dev):
        raise DeviceError(
            f"Device mismatch: numbers is on '{dev}', but IndexHelper is not."
        )
    ihelp._validate()
    if ihelp.unique_angular.dtype is not torch.int64:
        raise TypeError("SCFInputs IndexHelper must have dtype torch.int64.")

    if numbers.ndim != (1 if ihelp.batch_mode == 0 else 2):
        raise ValueError(
            "SCF numbers rank does not match IndexHelper.batch_mode."
        )
    if ihelp.atom_to_unique.shape != numbers.shape:
        raise ValueError(
            "SCFInputs IndexHelper atom shape "
            f"{tuple(ihelp.atom_to_unique.shape)} does not match numbers "
            f"shape {tuple(numbers.shape)}."
        )
    b, nat = tuple(numbers.shape[:-1]), numbers.shape[-1]
    nsh, nao = ihelp.shells_to_atom.shape[-1], ihelp.orbitals_to_shell.shape[-1]
    if nsh == 0 or nao == 0:
        raise ValueError("SCF requires at least one shell and orbital.")

    if integrals is not None and (
        integrals.hcore is None or integrals.overlap is None
    ):
        raise ValueError("SCFInputs require hcore and overlap integrals.")
    anchor = (
        integrals.hcore
        if integrals is not None
        else (positions if positions is not None else refocc)
    )
    if not isinstance(anchor, Tensor) or not anchor.is_floating_point():
        raise TypeError("SCFInputs floating inputs must have a floating dtype.")
    fdtype = anchor.dtype

    if refocc is not None:
        _check_scf_tensor("refocc", refocc, dev, fdtype)
        nushell = ihelp.unique_angular.shape[-1]
        if tuple(refocc.shape) not in ((nushell,), (*b, nushell)):
            raise ValueError("SCFInputs 'refocc' shape is incompatible.")
    for name, val, shapes in (
        ("chrg", chrg, ((), (1,)) if not b else ((*b, 1),)),
        ("spin", spin, ((*b, 1),)),
    ):
        if val is None or (
            name == "spin" and isinstance(val, list) and _is_valid_spin_list(val)
        ):
            continue
        if (
            not isinstance(val, Tensor)
            or val.is_complex()
            or val.dtype is torch.bool
        ):
            raise TypeError(
                f"SCFInputs '{name}' must be a real numeric Tensor."
            )
        if val.device != dev:
            raise DeviceError(
                f"Device mismatch: numbers is on '{dev}', but '{name}' is on "
                f"'{val.device}'."
            )
        if tuple(val.shape) not in shapes:
            raise ValueError(
                f"SCFInputs '{name}' shape {tuple(val.shape)} is incompatible."
            )
    tensors: list[tuple[str, Tensor | None, tuple[int, ...] | None]] = [
        ("positions", positions, (*b, nat, 3)),
    ]
    if integrals is not None:
        tensors.extend(
            (
                ("hcore", integrals.hcore, (*b, nao, nao)),
                ("overlap", integrals.overlap, (*b, nao, nao)),
                ("dipole", integrals.dipole, (*b, 3, nao, nao)),
                ("quadrupole", integrals.quadrupole, (*b, 6, nao, nao)),
            )
        )
    for name, t, shp in tensors:
        if t is not None:
            _check_scf_tensor(name, t, dev, fdtype, shp)
    return dev, fdtype, b, nat, nsh, nao


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
) -> SCFInputs:
    """Build immutable core values from explicit SCF inputs."""
    if config.fermi.etemp < 0.0:
        raise ValueError("Electronic temperature must be non-negative.")
    if config.fermi.maxiter < 0:
        raise ValueError("Fermi maxiter must be non-negative.")
    validate_scf_interactions(interactions)
    dev, fdtype, b, nat, nsh, nao = _validate_scf_preamble(
        numbers, ihelp, integrals=integrals
    )
    for name, t, shp in (
        ("n0", n0, (*b, nao)),
        ("occupation", occupation, (*b, 2, nao)),
        ("nel", nel, (*b, 2)),
    ):
        _check_scf_tensor(name, t, dev, fdtype, shp)

    orbital_mask = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)
    orbital_mask = orbital_mask.to(dtype=torch.bool)
    padding_diagonal = (~orbital_mask).to(dtype=integrals.overlap.dtype)
    overlap = integrals.overlap + torch.diag_embed(padding_diagonal)

    inverse_cholesky = _symmetric_inverse_sqrt(overlap)
    from .interactions import prepare_scf_interactions

    core_interactions = prepare_scf_interactions(
        interactions,
        interaction_data,
        numbers,
        d4sc_setup=getattr(interaction_data, "d4sc_setup", None),
        kind_by_label=getattr(interaction_data, "scf_kinds", None),
        target_dtype=fdtype,
    )
    for entry in core_interactions.entries:
        if entry.kind == "custom":
            for t_name, t_val in entry.data:
                if isinstance(t_val, Tensor) and t_val.is_complex():
                    raise TypeError(
                        f"SCFInputs 'custom.{t_name}' must not be complex."
                    )
                expected_dt = (
                    fdtype
                    if t_name.startswith("float_")
                    else (
                        t_val.dtype if isinstance(t_val, Tensor) else fdtype
                    )
                )
                _check_scf_tensor(
                    f"custom.{t_name}", t_val, dev, expected_dt
                )
            continue
        if entry.kind == "d4sc":
            setup = entry.option("setup")
            rc6 = getattr(setup, "model_rc6", None)
            _check_scf_tensor("d4sc.setup.model_rc6", rc6, dev, fdtype)
            assert isinstance(rc6, Tensor)
            if rc6.ndim != len(b) + 4 or rc6.shape[-2] != rc6.shape[-1]:
                raise ValueError("SCFInputs 'd4sc.setup.model_rc6' invalid.")
            nref = int(rc6.shape[-1])
            d4_shapes = {
                "numbers": (*b, nat),
                "rcov": (*b, nat),
                "r4r2": (*b, nat),
                "pair_weight": (*b, nat, nat),
                "model_rc6": (*b, nat, nat, nref, nref),
            }
            for path, leaf in tensor_leaves(setup):
                dt = (
                    None
                    if path == "numbers"
                    else (fdtype if leaf.is_floating_point() else leaf.dtype)
                )
                _check_scf_tensor(
                    f"d4sc.setup.{path}", leaf, dev, dt, d4_shapes.get(path)
                )
            req = {"cn": (*b, nat), "dispmat": (*b, nat, nat, nref, nref)}
        else:
            dim = (
                nsh
                if entry.kind in ("es2", "es3")
                and entry.option("shell_resolved")
                else nat
            )
            req = {
                "es2": {"mat": (*b, dim, dim)},
                "es3": {"hd": (*b, dim)},
                "aes2": {
                    "dkernel": (*b, nat, 1),
                    "qkernel": (*b, nat, 1),
                    "amat_sd": (*b, nat, nat, 3),
                    "amat_dd": (*b, nat, nat, 3, 3),
                    "amat_sq": (*b, nat, nat, 6),
                },
                "generalized_born": {"mat": (*b, nat, nat)},
                "electric_field": {"vat": (*b, nat), "vdp": (*b, nat, 3)},
                "electric_field_grad": {
                    "vat": (*b, nat),
                    "vdp": (*b, nat, 3),
                    "vqp": (*b, nat, 6),
                },
            }.get(entry.kind, {})
        for t_name, t_val in entry.data:
            _check_scf_tensor(
                f"{entry.kind}.{t_name}", t_val, dev, fdtype, req.get(t_name)
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
        index=ihelp,
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


def _symmetric_inverse_sqrt(overlap: Tensor) -> Tensor:
    """Return a smooth, permutation-covariant overlap orthogonalizer.

    A triangular Cholesky orthogonalizer depends on AO ordering. Its third
    derivatives can therefore differ after a physically equivalent AO
    permutation, even when the generalized spectrum and first two derivatives
    agree. The fixed Newton--Schulz matrix iteration computes the symmetric
    inverse square root without eigenvector derivatives. Frobenius scaling
    puts every positive overlap spectrum in the convergence interval.
    """
    input_dtype = overlap.dtype
    work_dtype = torch.float64 if input_dtype != torch.float64 else input_dtype
    overlap = overlap.to(dtype=work_dtype)
    overlap = 0.5 * (overlap + overlap.transpose(-2, -1))
    size = overlap.shape[-1]
    identity = torch.eye(
        size, dtype=overlap.dtype, device=overlap.device
    ).expand_as(overlap)
    scale = torch.linalg.matrix_norm(overlap, ord="fro", dim=(-2, -1))
    y = overlap / scale[..., None, None]
    z = identity
    for _ in range(24):
        correction = 0.5 * (3.0 * identity - z @ y)
        y = y @ correction
        z = correction @ z
    inverse_root = z / scale.sqrt()[..., None, None]
    result = 0.5 * (inverse_root + inverse_root.transpose(-2, -1))
    return result.to(dtype=input_dtype)


def validate_scf_interactions(
    interactions: InteractionList,
) -> None:
    """Validate that all SCF interaction entries inherit from :class:`Interaction` and have unique labels."""
    eligible, names = scf_interactions_eligible(interactions)
    if not eligible:
        raise TypeError(
            "SCF requires Interaction instances; got unsupported "
            f"component(s): {names}."
        )
    validate_unique_interaction_labels(interactions.components)


def scf_interactions_eligible(
    interactions: InteractionList,
) -> tuple[bool, str]:
    """Check that all entries in ``interactions`` inherit from :class:`Interaction`."""
    unsupported = [
        component
        for component in interactions.components
        if not isinstance(component, Interaction)
    ]
    names = ", ".join(
        f"{type(component).__module__}.{type(component).__qualname__}"
        for component in unsupported
    )
    return not unsupported, names


_SQRT2 = math.sqrt(2.0)
_QUAD_WEIGHT_FACTORS = (1.0, _SQRT2, 1.0, _SQRT2, _SQRT2, 1.0)


def _quadrupole_weights(quad: Tensor) -> Tensor:
    """Return off-diagonal ``sqrt(2)`` weights for 6-component quadrupoles.

    In dxtb's ``[xx, xy, yy, xz, yz, zz]`` ordering, off-diagonal indices are
    ``[1, 3, 4]``. Scaling charge quadrupoles by ``sqrt(2)`` and potential
    quadrupoles by ``1 / sqrt(2)`` makes the packed Euclidean norm match the
    rotationally invariant Frobenius norm of the full symmetric 3x3 tensor.
    """
    return quad.new_tensor(_QUAD_WEIGHT_FACTORS)


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
    quad = None
    if layout.quadrupole_shape is not None:
        raw_quad = values[-1]
        quad = raw_quad / _quadrupole_weights(raw_quad)
    return SCFCharges(
        mono=values[0],
        dipole=(values[1] if layout.dipole_shape is not None else None),
        quad=quad,
    )


def pack_charges(charges: SCFCharges, layout: ChargeLayout) -> Tensor:
    """Pack fixed-shape charges using the PADNZ row convention.

    Off-diagonal quadrupole components ``[1, 3, 4]`` (``xy``, ``xz``, ``yz``)
    are weighted by ``sqrt(2)`` so the packed quadrupole row carries the
    :math:`\\mathrm{SO}(3)`-invariant Frobenius metric. Row 0 stores the
    orbital-resolved Mulliken populations ``charges.mono`` without shell
    averaging so the physical charge-mode fixed-point problem is unchanged
    (note that individual :math:`p`/:math:`d` orbital populations are not
    rotational scalars, unlike the shell-isotropic monopole potential in
    :func:`pack_potential`).
    """
    values: list[Tensor] = [charges.mono.reshape(-1)]
    if layout.dipole_shape is not None:
        if charges.dipole is None:
            raise ValueError("Dipole charge is required by this layout.")
        values.append(charges.dipole.reshape(-1))
    if layout.quadrupole_shape is not None:
        if charges.quad is None:
            raise ValueError("Quadrupole charge is required by this layout.")
        weighted_quad = charges.quad * _quadrupole_weights(charges.quad)
        values.append(weighted_quad.reshape(-1))

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
    """Pack fixed-shape potential rows using the PADNZ convention."""
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
    quad = None
    if layout.quadrupole_shape is not None:
        raw_quad = values[-1]
        quad = raw_quad * _quadrupole_weights(raw_quad)
    return SCFPotential(
        mono=values[0],
        dipole=(values[1] if layout.dipole_shape is not None else None),
        quad=quad,
    )


def potential_layout(potential: SCFPotential) -> PotentialLayout:
    """Freeze component shapes from one call-local physical potential."""
    if potential.mono is None:
        raise ValueError("A potential requires a monopole row.")
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
        if name == "quad":
            value = value / _quadrupole_weights(value)
        values.append(value.reshape(-1))
    return values


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result
