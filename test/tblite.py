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
Testsuite Interface for ``tblite``
==================================

Focused test helper around ``tblite.interface.Calculator``. Converts ``dxtb``
test structures and options to ``tblite`` inputs, executes a single
``tblite`` calculation, and returns double-precision reference properties.

Option names and defaults follow ``tblite.interface.Calculator``:

=================== ========================================= ================
 Option              Description                               Default
=================== ========================================= ================
 ``method``          Parametrization name or dictionary        ``"GFN2-xTB"``
 ``accuracy``        SCC numerical threshold scaling           ``1.0``
 ``guess``           Initial guess (``"sad"``/0, ``"eeq"``/1)  ``0`` (SAD)
 ``max-iter``        Maximum number of SCC iterations          ``250``
 ``mixer-damping``   Broyden mixer damping parameter           ``0.4``
 ``save-integrals``  Retain overlap and Hamiltonian matrices   ``False``
 ``temperature``     Electronic temperature in Hartree         ``9.500e-4``
 ``verbosity``       Output verbosity level                    ``0``
 ``electric-field``  Uniform electric field vector ``(3,)``    ``None``
 ``alpb-solvation``  ALPB solvation ``(solvent, ...)``         ``None``
 ``gbsa-solvation``  GBSA solvation ``(solvent, ...)``         ``None``
 ``gbe-solvation``   Non-empirical ALPB ``(eps, kernel)``      ``None``
 ``gb-solvation``    Non-empirical GB ``(eps, kernel)``        ``None``
=================== ========================================= ================

Conventions
-----------
- **Units**: Coordinates and lattice vectors must be in Bohr ($a_0$).
  ``temperature`` is in Hartree ($E_h$); use ``temperature_kelvin`` to pass a
  temperature in Kelvin (converted via ``KELVIN2AU``).
- **Spin**: ``uhf`` (and its alias ``spin``) is the integer number of unpaired
  electrons ($N_\\alpha - N_\\beta$), **not** the spin multiplicity ($2S + 1$)
  or total spin quantum number ($S$).
- **Precision**: ``tblite`` evaluates in double precision (``float64``). Tensor
  outputs default to ``torch.float64`` regardless of the input coordinate
  ``dtype`` so reference precision is never silently discarded; pass ``dtype=``
  explicitly if a downcast tensor is required.
- **Quadrupole**: :func:`get_tblite_quadrupole` and ``TbliteResult.quadrupole``
  return ``tblite``'s 6-component traceless molecular quadrupole moment in
  lower-triangular component order ``(xx, yx, yy, zx, zy, zz)`` in $e\\,a_0^2$
  (matching :meth:`dxtb.Calculator.quadrupole`). Raw AO quadrupole integrals in
  :func:`dxtb.integrals.wrappers.quadint` instead use all 9 Cartesian
  components ``(xx, xy, xz, yx, yy, yz, zx, zy, zz)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import torch
from tad_mctc.io.structure import Structure as MctcStructure
from tad_mctc.units.energy import KELVIN2AU

from dxtb._src.typing import Tensor

__all__ = [
    "Options",
    "Structure",
    "TbliteResult",
    "get_tblite_charges",
    "get_tblite_dipole",
    "get_tblite_energy",
    "get_tblite_forces",
    "get_tblite_gradient",
    "get_tblite_hamiltonian_matrix",
    "get_tblite_overlap_matrix",
    "get_tblite_quadrupole",
    "get_tblite_version",
    "tblite_singlepoint",
]

_METHOD_MAP: dict[str, str] = {
    "gfn1": "GFN1-xTB",
    "gfn1-xtb": "GFN1-xTB",
    "GFN1-xTB": "GFN1-xTB",
    "gfn2": "GFN2-xTB",
    "gfn2-xtb": "GFN2-xTB",
    "GFN2-xTB": "GFN2-xTB",
    "ipea1": "IPEA1-xTB",
    "ipea1-xtb": "IPEA1-xTB",
    "IPEA1-xTB": "IPEA1-xTB",
}

_GUESS_MAP: dict[str, int] = {
    "sad": 0,
    "eeq": 1,
}

_OPTION_KEY_ALIASES: dict[str, str] = {
    "max-iter": "max_iter",
    "mixer-damping": "mixer_damping",
    "save-integrals": "save_integrals",
    "electric-field": "electric_field",
    "spin-polarization": "spin_polarization",
    "alpb-solvation": "alpb_solvation",
    "gbsa-solvation": "gbsa_solvation",
    "gbe-solvation": "gbe_solvation",
    "gb-solvation": "gb_solvation",
}


class Structure:
    """
    Molecular structure data for ``tblite`` reference evaluations.

    Mirrors ``tblite.interface.Structure`` while accepting both PyTorch
    :class:`~torch.Tensor` and NumPy :class:`~numpy.ndarray` inputs.

    Parameters
    ----------
    numbers : Tensor | np.ndarray | Sequence[int]
        Atomic numbers of shape ``(nat,)``.
    positions : Tensor | np.ndarray | Sequence[Sequence[float]]
        Cartesian coordinates in Bohr of shape ``(nat, 3)``.
    charge : Tensor | float | int | None, optional
        Total molecular charge. Defaults to ``None`` (neutral).
    uhf : Tensor | int | float | None, optional
        Integer number of unpaired electrons ($N_\\alpha - N_\\beta$), not spin
        multiplicity. Defaults to ``None`` (closed-shell).
    lattice : Tensor | np.ndarray | None, optional
        Lattice vectors in Bohr of shape ``(3, 3)``. Defaults to ``None``.
    periodic : Tensor | np.ndarray | Sequence[bool] | None, optional
        Periodic boundary flags of shape ``(3,)``. Defaults to ``None``.
    spin : Tensor | int | float | None, optional
        Alias for ``uhf`` (unpaired electron count). Defaults to ``None``.
    """

    def __init__(
        self,
        numbers: Tensor | np.ndarray | Sequence[int],
        positions: Tensor | np.ndarray | Sequence[Sequence[float]],
        charge: Tensor | float | int | None = None,
        uhf: Tensor | int | float | None = None,
        lattice: Tensor | np.ndarray | None = None,
        periodic: Tensor | np.ndarray | Sequence[bool] | None = None,
        *,
        spin: Tensor | int | float | None = None,
    ) -> None:
        resolved_uhf = _resolve_uhf(uhf, spin)
        self.numbers = numbers
        self.positions = positions
        self.charge = charge
        self.uhf = resolved_uhf
        self.lattice = lattice
        self.periodic = periodic

    @property
    def spin(self) -> Tensor | int | float | None:
        """Alias for :attr:`uhf` (number of unpaired electrons)."""
        return self.uhf

    def __len__(self) -> int:
        return int(len(self.numbers))

    def replace(self, **changes: Any) -> Structure:
        """Return a new :class:`Structure` with updated fields."""
        kwargs: dict[str, Any] = {
            "numbers": self.numbers,
            "positions": self.positions,
            "charge": self.charge,
            "uhf": self.uhf,
            "lattice": self.lattice,
            "periodic": self.periodic,
        }
        if "uhf" in changes or "spin" in changes:
            changes = dict(changes)
            changes["uhf"] = _resolve_uhf(
                changes.pop("uhf", None),
                changes.pop("spin", None),
            )
        kwargs.update(changes)
        return Structure(**kwargs)


@dataclass(frozen=True)
class Options:
    """
    Calculation options for ``tblite`` single-point evaluations.

    Option names and defaults follow ``tblite.interface.Calculator``. Both
    underscored attribute names and hyphenated ``tblite`` option keys (when
    constructed via :meth:`from_mapping`) are supported.

    Parameters
    ----------
    method : str | dict[str, Any], optional
        Method name (``"GFN2-xTB"``, ``"GFN1-xTB"``, ``"IPEA1-xTB"``, or
        lowercase aliases) or a ``tblite`` parametrization dictionary.
        Defaults to ``"GFN2-xTB"``.
    accuracy : float, optional
        Numerical threshold scaling for SCC. Defaults to ``1.0``.
    guess : Literal["sad", "eeq"] | int | None, optional
        Initial wavefunction guess (``"sad"``/``0`` or ``"eeq"``/``1``).
        Defaults to ``None`` (uses ``tblite`` default ``0``).
    max_iter : int | None, optional
        Maximum number of SCC iterations (``"max-iter"``). Defaults to
        ``None`` (uses ``tblite`` default ``250``).
    mixer_damping : float | None, optional
        Broyden mixer damping parameter (``"mixer-damping"``). Defaults to
        ``None`` (uses ``tblite`` default ``0.4``).
    save_integrals : bool, optional
        Whether to keep overlap and Hamiltonian matrices in the result
        (``"save-integrals"``). Defaults to ``False``.
    temperature : float | None, optional
        Electronic temperature in Hartree (``"temperature"``). Defaults to
        ``None`` (uses ``tblite`` default ``9.500e-4`` Eh).
    temperature_kelvin : float | None, optional
        Electronic temperature in Kelvin, converted to Hartree via
        ``KELVIN2AU``. Mutually exclusive with ``temperature``. Defaults to
        ``None``.
    verbosity : int, optional
        Printout verbosity level (``"verbosity"``). Defaults to ``0``.
    electric_field : Tensor | np.ndarray | Sequence[float] | None, optional
        Uniform electric field vector of shape ``(3,)`` in atomic units
        (``"electric-field"``). Defaults to ``None``.
    spin_polarization : float | None, optional
        Spin-polarization scaling factor (``"spin-polarization"``). Defaults
        to ``None``.
    alpb_solvation : str | float | Sequence[Any] | None, optional
        Arguments for ``"alpb-solvation"``. Defaults to ``None``.
    gbsa_solvation : str | float | Sequence[Any] | None, optional
        Arguments for ``"gbsa-solvation"``. Defaults to ``None``.
    gbe_solvation : float | Sequence[Any] | None, optional
        Arguments for ``"gbe-solvation"``, e.g. ``(eps, "p16")``. Defaults to
        ``None``.
    gb_solvation : float | Sequence[Any] | None, optional
        Arguments for ``"gb-solvation"``, e.g. ``(eps, "still")``. Defaults to
        ``None``.
    solvation : tuple[Any, ...] | None, optional
        Interaction tuple ``(name, *args)`` passed to ``calc.add(name, *args)``.
        Defaults to ``None``.
    """

    method: str | dict[str, Any] = "GFN2-xTB"
    accuracy: float = 1.0
    guess: Literal["sad", "eeq"] | int | None = None
    max_iter: int | None = None
    mixer_damping: float | None = None
    save_integrals: bool = False
    temperature: float | None = None
    temperature_kelvin: float | None = None
    verbosity: int = 0
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None
    spin_polarization: float | None = None
    alpb_solvation: str | float | Sequence[Any] | None = None
    gbsa_solvation: str | float | Sequence[Any] | None = None
    gbe_solvation: float | Sequence[Any] | None = None
    gb_solvation: float | Sequence[Any] | None = None
    solvation: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and self.temperature_kelvin is not None:
            raise ValueError(
                "Specify at most one of 'temperature' (Hartree) or "
                "'temperature_kelvin' (Kelvin)."
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Options:
        """Construct :class:`Options` from a mapping with tblite or Python keys."""
        normalized: dict[str, Any] = {}
        valid_fields = set(cls.__dataclass_fields__)
        for raw_key, value in mapping.items():
            key = _OPTION_KEY_ALIASES.get(raw_key, raw_key)
            if key not in valid_fields:
                raise ValueError(
                    f"Unsupported tblite option '{raw_key}'. "
                    f"Supported options are: {sorted(valid_fields)}."
                )
            if key in normalized:
                raise ValueError(
                    f"Duplicate specification for tblite option '{key}'."
                )
            normalized[key] = value
        return cls(**normalized)


@dataclass(frozen=True)
class TbliteResult:
    """
    Container for results of a ``tblite`` single-point evaluation.

    Notes
    -----
    - ``gradient`` is $\\nabla_{\\mathbf{R}} E$ of shape ``(nat, 3)`` in
      $E_h / a_0$; ``forces`` is $-\\nabla_{\\mathbf{R}} E$ (``-gradient``).
    - ``quadrupole`` is the 6-component traceless molecular quadrupole moment
      in ``tblite`` component order ``(xx, yx, yy, zx, zy, zz)`` in $e\\,a_0^2$.
    - ``overlap_matrix`` and ``hamiltonian_matrix`` are populated only when
      ``save_integrals=True``.
    """

    method: str
    version: tuple[int, int, int]
    energy: Tensor | float
    energies: Tensor | np.ndarray
    gradient: Tensor | np.ndarray
    forces: Tensor | np.ndarray
    charges: Tensor | np.ndarray
    dipole: Tensor | np.ndarray
    quadrupole: Tensor | np.ndarray
    orbital_energies: Tensor | np.ndarray
    orbital_occupations: Tensor | np.ndarray
    density_matrix: Tensor | np.ndarray
    overlap_matrix: Tensor | np.ndarray | None = None
    hamiltonian_matrix: Tensor | np.ndarray | None = None


def get_tblite_version() -> tuple[int, int, int]:
    """Return the active native ``tblite`` library version tuple."""
    import tblite.library

    ver = tblite.library.get_version()
    return int(ver[0]), int(ver[1]), int(ver[2])


def _to_numpy_numbers(
    numbers: Tensor | np.ndarray | Sequence[int],
) -> np.ndarray:
    if isinstance(numbers, torch.Tensor):
        if (
            numbers.is_floating_point()
            or numbers.is_complex()
            or numbers.dtype == torch.bool
        ):
            raise ValueError(
                f"Atomic numbers must have an integer dtype, got {numbers.dtype}."
            )
        arr = numbers.detach().cpu().numpy().astype(np.int32, copy=True)
    else:
        raw = np.asarray(numbers)
        if raw.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"Atomic numbers must have an integer dtype, got {raw.dtype}."
            )
        arr = raw.astype(np.int32, copy=True)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"Expected 1D non-empty atomic numbers array, got shape {arr.shape}."
        )
    if np.any(arr <= 0):
        raise ValueError(
            "All atomic numbers passed to tblite must be positive (Z > 0)."
        )
    return np.ascontiguousarray(arr, dtype=np.int32)


def _to_numpy_positions(
    positions: Tensor | np.ndarray | Sequence[Sequence[float]],
    *,
    nat: int,
) -> np.ndarray:
    if isinstance(positions, torch.Tensor):
        if not positions.is_floating_point():
            raise ValueError(
                f"Positions must have a floating-point dtype, got {positions.dtype}."
            )
        arr = positions.detach().cpu().numpy().astype(np.float64, copy=True)
    else:
        arr = np.array(positions, dtype=np.float64, copy=True)
    if arr.shape != (nat, 3):
        raise ValueError(
            f"Expected positions of shape ({nat}, 3), got {arr.shape}."
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _to_numpy_lattice(
    lattice: Tensor | np.ndarray | Sequence[Sequence[float]] | None,
) -> np.ndarray | None:
    if lattice is None:
        return None
    if isinstance(lattice, torch.Tensor):
        arr = lattice.detach().cpu().numpy().astype(np.float64, copy=True)
    else:
        arr = np.array(lattice, dtype=np.float64, copy=True)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected lattice of shape (3, 3), got {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _to_numpy_periodic(
    periodic: Tensor | np.ndarray | Sequence[bool] | None,
) -> np.ndarray | None:
    if periodic is None:
        return None
    if isinstance(periodic, torch.Tensor):
        arr = periodic.detach().cpu().numpy().astype(bool, copy=True)
    else:
        arr = np.array(periodic, dtype=bool, copy=True)
    if arr.shape != (3,):
        raise ValueError(f"Expected periodic flags of shape (3,), got {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=bool)


def _to_scalar_float(value: Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Expected scalar tensor, got shape {tuple(value.shape)}."
            )
        return float(value.detach().cpu().item())
    return float(value)


def _to_scalar_int(
    value: Tensor | float | int | None, *, name: str = "uhf"
) -> int | None:
    if value is None:
        return None
    scalar = _to_scalar_float(value)
    assert scalar is not None
    rounded = round(scalar)
    if abs(scalar - rounded) > 1.0e-12:
        raise ValueError(
            f"Expected an integer number of unpaired electrons for '{name}', "
            f"got {scalar!r}."
        )
    if rounded < 0:
        raise ValueError(
            f"Number of unpaired electrons '{name}' must be non-negative, "
            f"got {rounded}."
        )
    return int(rounded)


def _resolve_uhf(
    uhf: Tensor | int | float | None,
    spin: Tensor | int | float | None,
) -> Tensor | int | float | None:
    uhf_int = _to_scalar_int(uhf, name="uhf")
    spin_int = _to_scalar_int(spin, name="spin")
    if uhf_int is not None and spin_int is not None and uhf_int != spin_int:
        raise ValueError(
            f"Conflicting 'uhf' ({uhf_int}) and 'spin' ({spin_int}) values."
        )
    return uhf if uhf is not None else spin


def _coerce_structure(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    *,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
) -> Structure:
    override_uhf = _resolve_uhf(uhf, spin)

    if isinstance(structure, Structure):
        out = structure
    elif isinstance(structure, Mapping):
        out = Structure(
            numbers=structure["numbers"],
            positions=structure["positions"],
            charge=structure.get("charge"),
            uhf=_resolve_uhf(structure.get("uhf"), structure.get("spin")),
            lattice=structure.get("lattice"),
            periodic=structure.get("periodic"),
        )
    elif hasattr(structure, "numbers") and hasattr(structure, "positions"):
        struct_uhf = getattr(structure, "uhf", None)
        struct_spin = getattr(structure, "spin", None)
        resolved = _resolve_uhf(struct_uhf, struct_spin)
        out = Structure(
            numbers=structure.numbers,
            positions=structure.positions,
            charge=getattr(structure, "charge", None),
            uhf=resolved,
            lattice=getattr(structure, "lattice", None),
            periodic=getattr(structure, "periodic", None),
        )
    else:
        raise TypeError(
            "Expected a 'Structure' (or object/mapping with 'numbers' and "
            f"'positions'), got '{type(structure).__name__}'."
        )

    if charge is not None or override_uhf is not None:
        out = out.replace(
            charge=charge if charge is not None else out.charge,
            uhf=override_uhf if override_uhf is not None else out.uhf,
        )
    return out


def _resolve_options(
    method: str | dict[str, Any] | Options | None = None,
    options: Options | Mapping[str, Any] | None = None,
    *,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    save_integrals: bool | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    verbosity: int | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    spin_polarization: float | None = None,
    alpb_solvation: str | float | Sequence[Any] | None = None,
    gbsa_solvation: str | float | Sequence[Any] | None = None,
    gbe_solvation: float | Sequence[Any] | None = None,
    gb_solvation: float | Sequence[Any] | None = None,
    solvation: tuple[Any, ...] | None = None,
) -> Options:
    if temperature is not None and temperature_kelvin is not None:
        raise ValueError(
            "Specify at most one of 'temperature' (Hartree) or "
            "'temperature_kelvin' (Kelvin)."
        )
    if isinstance(method, Options):
        if options is not None:
            raise ValueError(
                "Pass 'Options' either as 'method' or 'options', not both."
            )
        base = method
        explicit_method = None
    else:
        explicit_method = method
        if options is None:
            base = Options()
        elif isinstance(options, Options):
            base = options
        elif isinstance(options, Mapping):
            base = Options.from_mapping(options)
        else:
            raise TypeError(
                "Expected 'options' to be an 'Options' instance or mapping, "
                f"got '{type(options).__name__}'."
            )

    updates: dict[str, Any] = {}
    if explicit_method is not None:
        updates["method"] = explicit_method
    if accuracy is not None:
        updates["accuracy"] = accuracy
    if guess is not None:
        updates["guess"] = guess
    if max_iter is not None:
        updates["max_iter"] = max_iter
    if mixer_damping is not None:
        updates["mixer_damping"] = mixer_damping
    if save_integrals is not None:
        updates["save_integrals"] = save_integrals
    if temperature is not None:
        if base.temperature_kelvin is not None:
            raise ValueError(
                "Conflicting 'temperature' override on Options with 'temperature_kelvin' set."
            )
        updates["temperature"] = temperature
    if temperature_kelvin is not None:
        if base.temperature is not None:
            raise ValueError(
                "Conflicting 'temperature_kelvin' override on Options with 'temperature' set."
            )
        updates["temperature_kelvin"] = temperature_kelvin
    if verbosity is not None:
        updates["verbosity"] = verbosity
    if electric_field is not None:
        updates["electric_field"] = electric_field
    if spin_polarization is not None:
        updates["spin_polarization"] = spin_polarization
    if alpb_solvation is not None:
        updates["alpb_solvation"] = alpb_solvation
    if gbsa_solvation is not None:
        updates["gbsa_solvation"] = gbsa_solvation
    if gbe_solvation is not None:
        updates["gbe_solvation"] = gbe_solvation
    if gb_solvation is not None:
        updates["gb_solvation"] = gb_solvation
    if solvation is not None:
        updates["solvation"] = solvation

    if updates:
        return replace(base, **updates)
    return base


def _add_solvation_arg(calc: Any, name: str, spec: Any) -> None:
    if isinstance(spec, (tuple, list)):
        calc.add(name, *spec)
    else:
        calc.add(name, spec)


def tblite_singlepoint(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    save_integrals: bool | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    verbosity: int | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    spin_polarization: float | None = None,
    alpb_solvation: str | float | Sequence[Any] | None = None,
    gbsa_solvation: str | float | Sequence[Any] | None = None,
    gbe_solvation: float | Sequence[Any] | None = None,
    gb_solvation: float | Sequence[Any] | None = None,
    solvation: tuple[Any, ...] | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> TbliteResult:
    """
    Execute a single-point calculation through ``tblite.interface.Calculator``.

    Parameters
    ----------
    structure : Structure | MctcStructure | Mapping[str, Any]
        Molecular structure containing ``numbers`` and ``positions`` (in Bohr),
        plus optional ``charge``, ``uhf``, ``lattice``, and ``periodic``.
    method : str | dict[str, Any] | Options | None, optional
        Method name (e.g. ``"GFN2-xTB"``, ``"GFN1-xTB"``, ``"gfn2"``,
        ``"gfn1"``), parametrization dict, or :class:`Options`. Defaults to
        ``"GFN2-xTB"`` (or ``options.method`` if ``options`` is provided).
    options : Options | Mapping[str, Any] | None, optional
        Calculator options. Explicit keyword arguments below override fields on
        ``options``.
    charge : Tensor | float | int | None, optional
        Optional override for ``structure.charge``.
    uhf : Tensor | int | float | None, optional
        Optional override for ``structure.uhf`` (unpaired electrons).
    spin : Tensor | int | float | None, optional
        Alias for ``uhf``.
    accuracy : float | None, optional
        SCC numerical threshold scaling.
    guess : Literal["sad", "eeq"] | int | None, optional
        Initial wavefunction guess.
    max_iter : int | None, optional
        Maximum number of SCC iterations.
    mixer_damping : float | None, optional
        Broyden mixer damping parameter.
    save_integrals : bool | None, optional
        Retain overlap and Hamiltonian matrices in the result.
    temperature : float | None, optional
        Electronic temperature in Hartree. Mutually exclusive with
        ``temperature_kelvin``.
    temperature_kelvin : float | None, optional
        Electronic temperature in Kelvin (converted via ``KELVIN2AU``).
    verbosity : int | None, optional
        Printout verbosity level.
    electric_field : Tensor | np.ndarray | Sequence[float] | None, optional
        Uniform electric field vector of shape ``(3,)`` in atomic units.
    spin_polarization : float | None, optional
        Spin-polarization scaling factor.
    alpb_solvation, gbsa_solvation, gbe_solvation, gb_solvation : Any, optional
        Implicit solvation specification for the corresponding ``tblite`` model.
    solvation : tuple[Any, ...] | None, optional
        Interaction tuple ``(name, *args)`` passed to ``calc.add(name, *args)``.
    as_tensor : bool | None, optional
        If ``True``, return array quantities as :class:`~torch.Tensor` (and
        ``energy`` as a 0-D :class:`~torch.Tensor`). If ``False``, return
        NumPy arrays (and ``energy`` as ``float``). When ``None`` (default),
        inferred from ``isinstance(structure.positions, torch.Tensor)``.
    dtype : torch.dtype | None, optional
        Target floating-point dtype when returning tensors. Defaults to
        ``torch.float64`` so ``tblite``'s double-precision reference values are
        preserved unless a different dtype is explicitly requested.
    device : torch.device | None, optional
        Target device when returning tensors. Defaults to
        ``structure.positions.device`` if ``positions`` is a tensor.

    Returns
    -------
    TbliteResult
        Computed ``tblite`` single-point quantities.
    """
    mol = _coerce_structure(structure, charge=charge, uhf=uhf, spin=spin)
    opts = _resolve_options(
        method=method,
        options=options,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        save_integrals=save_integrals,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        verbosity=verbosity,
        electric_field=electric_field,
        spin_polarization=spin_polarization,
        alpb_solvation=alpb_solvation,
        gbsa_solvation=gbsa_solvation,
        gbe_solvation=gbe_solvation,
        gb_solvation=gb_solvation,
        solvation=solvation,
    )

    resolved_method: str | dict[str, Any]
    method_label: str
    if isinstance(opts.method, str):
        resolved_method = _METHOD_MAP.get(
            opts.method, _METHOD_MAP.get(opts.method.lower(), opts.method)
        )
        method_label = str(resolved_method)
    else:
        resolved_method = opts.method
        method_label = "custom-xTB"

    nums_np = _to_numpy_numbers(mol.numbers)
    pos_np = _to_numpy_positions(mol.positions, nat=nums_np.shape[0])
    chrg_val = _to_scalar_float(mol.charge)
    uhf_val = _to_scalar_int(mol.uhf, name="uhf")
    lat_np = _to_numpy_lattice(mol.lattice)
    per_np = _to_numpy_periodic(mol.periodic)

    from tblite.interface import Calculator as TbliteCalculator

    calc = TbliteCalculator(
        resolved_method,
        nums_np,
        pos_np,
        charge=chrg_val,
        uhf=uhf_val,
        lattice=lat_np,
        periodic=per_np,
    )
    calc.set("verbosity", int(opts.verbosity))
    calc.set("accuracy", float(opts.accuracy))

    if opts.max_iter is not None:
        calc.set("max-iter", int(opts.max_iter))
    if opts.temperature is not None:
        calc.set("temperature", float(opts.temperature))
    elif opts.temperature_kelvin is not None:
        calc.set("temperature", float(opts.temperature_kelvin) * KELVIN2AU)
    if opts.mixer_damping is not None:
        calc.set("mixer-damping", float(opts.mixer_damping))
    if opts.guess is not None:
        guess_code = (
            _GUESS_MAP[opts.guess.lower()]
            if isinstance(opts.guess, str)
            else int(opts.guess)
        )
        calc.set("guess", guess_code)
    if opts.save_integrals:
        calc.set("save-integrals", 1)

    if opts.electric_field is not None:
        from tblite.library import ffi

        field_np = (
            opts.electric_field.detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
            if isinstance(opts.electric_field, torch.Tensor)
            else np.ascontiguousarray(opts.electric_field, dtype=np.float64)
        )
        if field_np.shape != (3,):
            raise ValueError(
                f"Expected electric_field of shape (3,), got {field_np.shape}."
            )
        _field_vec = ffi.new("double[]", [float(x) for x in field_np])
        calc.add("electric-field", _field_vec)
    if opts.spin_polarization is not None:
        calc.add("spin-polarization", float(opts.spin_polarization))
    if opts.alpb_solvation is not None:
        _add_solvation_arg(calc, "alpb-solvation", opts.alpb_solvation)
    if opts.gbsa_solvation is not None:
        _add_solvation_arg(calc, "gbsa-solvation", opts.gbsa_solvation)
    if opts.gbe_solvation is not None:
        _add_solvation_arg(calc, "gbe-solvation", opts.gbe_solvation)
    if opts.gb_solvation is not None:
        _add_solvation_arg(calc, "gb-solvation", opts.gb_solvation)
    if opts.solvation is not None:
        calc.add(opts.solvation[0], *opts.solvation[1:])

    res = calc.singlepoint()

    return_tensors = (
        isinstance(mol.positions, torch.Tensor)
        if as_tensor is None
        else bool(as_tensor)
    )
    target_device = (
        device
        if device is not None
        else (
            mol.positions.device
            if isinstance(mol.positions, torch.Tensor)
            else None
        )
    )
    target_dtype = dtype if dtype is not None else torch.float64

    def _conv_float(val: Any) -> Tensor | np.ndarray:
        arr = np.asarray(val, dtype=np.float64)
        if return_tensors:
            return torch.as_tensor(
                arr, dtype=target_dtype, device=target_device
            )
        return arr

    energy_scalar = float(res.get("energy"))
    energy_out: Tensor | float = (
        torch.tensor(energy_scalar, dtype=target_dtype, device=target_device)
        if return_tensors
        else energy_scalar
    )
    grad_np = np.asarray(res.get("gradient"), dtype=np.float64)
    overlap_out = (
        _conv_float(res.get("overlap-matrix")) if opts.save_integrals else None
    )
    hamiltonian_out = (
        _conv_float(res.get("hamiltonian-matrix"))
        if opts.save_integrals
        else None
    )

    return TbliteResult(
        method=method_label,
        version=get_tblite_version(),
        energy=energy_out,
        energies=_conv_float(res.get("energies")),
        gradient=_conv_float(grad_np),
        forces=_conv_float(-grad_np),
        charges=_conv_float(res.get("charges")),
        dipole=_conv_float(res.get("dipole")),
        quadrupole=_conv_float(res.get("quadrupole")),
        orbital_energies=_conv_float(res.get("orbital-energies")),
        orbital_occupations=_conv_float(res.get("orbital-occupations")),
        density_matrix=_conv_float(res.get("density-matrix")),
        overlap_matrix=overlap_out,
        hamiltonian_matrix=hamiltonian_out,
    )


def get_tblite_energy(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    solvation: tuple[Any, ...] | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | float:
    """Return total ``tblite`` energy in Hartree for ``structure``."""
    return tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        electric_field=electric_field,
        solvation=solvation,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    ).energy


def get_tblite_gradient(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    solvation: tuple[Any, ...] | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | np.ndarray:
    """Return Cartesian ``tblite`` gradient ``(nat, 3)`` in Hartree/Bohr."""
    return tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        electric_field=electric_field,
        solvation=solvation,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    ).gradient


def get_tblite_forces(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    solvation: tuple[Any, ...] | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | np.ndarray:
    """Return Cartesian ``tblite`` forces ``(nat, 3)`` (``-gradient``) in Hartree/Bohr."""
    return tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        electric_field=electric_field,
        solvation=solvation,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    ).forces


def get_tblite_charges(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    solvation: tuple[Any, ...] | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | np.ndarray:
    """Return atomic partial charges ``(nat,)`` in elementary charge units."""
    return tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        electric_field=electric_field,
        solvation=solvation,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    ).charges


def get_tblite_dipole(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    solvation: tuple[Any, ...] | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | np.ndarray:
    """Return molecular dipole moment ``(3,)`` in e*Bohr."""
    return tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        electric_field=electric_field,
        solvation=solvation,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    ).dipole


def get_tblite_quadrupole(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
    solvation: tuple[Any, ...] | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | np.ndarray:
    """
    Return molecular traceless quadrupole moment ``(6,)`` in e*Bohr^2.

    Component order follows ``tblite``'s lower-triangular packing
    ``(xx, yx, yy, zx, zy, zz)`` (matching :meth:`dxtb.Calculator.quadrupole`),
    distinct from the 9-component raw integral matrix returned by
    :func:`dxtb.integrals.wrappers.quadint`.
    """
    return tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        electric_field=electric_field,
        solvation=solvation,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    ).quadrupole


def get_tblite_overlap_matrix(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | np.ndarray:
    """Return AO overlap matrix ``(norb, norb)`` (enables ``save_integrals=True``)."""
    res = tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        save_integrals=True,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    )
    assert res.overlap_matrix is not None
    return res.overlap_matrix


def get_tblite_hamiltonian_matrix(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
    accuracy: float | None = None,
    guess: Literal["sad", "eeq"] | int | None = None,
    max_iter: int | None = None,
    mixer_damping: float | None = None,
    temperature: float | None = None,
    temperature_kelvin: float | None = None,
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor | np.ndarray:
    """Return core AO Hamiltonian matrix ``(norb, norb)`` in Hartree (enables ``save_integrals=True``)."""
    res = tblite_singlepoint(
        structure,
        method,
        options=options,
        charge=charge,
        uhf=uhf,
        spin=spin,
        accuracy=accuracy,
        guess=guess,
        max_iter=max_iter,
        mixer_damping=mixer_damping,
        save_integrals=True,
        temperature=temperature,
        temperature_kelvin=temperature_kelvin,
        as_tensor=as_tensor,
        dtype=dtype,
        device=device,
    )
    assert res.hamiltonian_matrix is not None
    return res.hamiltonian_matrix
