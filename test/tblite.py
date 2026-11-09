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

Provides a high-level wrapper around ``tblite.interface`` for the test suite.
Tests pass a :class:`Structure` (or any structure-like object providing
``numbers`` and ``positions``) to property getters such as
:func:`get_tblite_energy`, :func:`get_tblite_gradient`,
:func:`get_tblite_dipole`, or :func:`tblite_singlepoint` without interacting
with ``tblite.interface.Calculator`` directly.

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
    "TbliteOptions",
    "TbliteResult",
    "get_tblite_bond_orders",
    "get_tblite_charges",
    "get_tblite_density_matrix",
    "get_tblite_dipole",
    "get_tblite_energies",
    "get_tblite_energy",
    "get_tblite_forces",
    "get_tblite_gradient",
    "get_tblite_hamiltonian_matrix",
    "get_tblite_orbital_coefficients",
    "get_tblite_orbital_energies",
    "get_tblite_orbital_occupations",
    "get_tblite_overlap_matrix",
    "get_tblite_property",
    "get_tblite_quadrupole",
    "get_tblite_version",
    "get_tblite_virial",
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
    "maxiter": "max_iter",
    "mixer-damping": "mixer_damping",
    "save-integrals": "save_integrals",
    "electric-field": "electric_field",
    "spin-polarization": "spin_polarization",
    "alpb-solvation": "alpb_solvation",
    "gbsa-solvation": "gbsa_solvation",
    "gbe-solvation": "gbe_solvation",
    "gb-solvation": "gb_solvation",
    "ddX-solvation": "ddx_solvation",
    "ddx-solvation": "ddx_solvation",
    "post-processing": "post_processing",
    "etemp": "temperature_kelvin",
    "fermi_etemp": "temperature_kelvin",
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
        Number of unpaired electrons. Defaults to ``None`` (closed-shell).
    lattice : Tensor | np.ndarray | None, optional
        Lattice vectors in Bohr of shape ``(3, 3)``. Defaults to ``None``.
    periodic : Tensor | np.ndarray | Sequence[bool] | None, optional
        Periodic boundary flags of shape ``(3,)``. Defaults to ``None``.
    spin : Tensor | int | float | None, optional
        Alias for ``uhf``. Defaults to ``None``.
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
        if uhf is not None and spin is not None:
            if _to_scalar_int(uhf) != _to_scalar_int(spin):
                raise ValueError(
                    "Conflicting 'uhf' and 'spin' values passed to Structure."
                )
        resolved_uhf = uhf if uhf is not None else spin

        self.numbers = numbers
        self.positions = positions
        self.charge = charge
        self.uhf = resolved_uhf
        self.lattice = lattice
        self.periodic = periodic

    @property
    def spin(self) -> Tensor | int | float | None:
        """Alias for :attr:`uhf`."""
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
        if "spin" in changes and "uhf" not in changes:
            changes = dict(changes)
            changes["uhf"] = changes.pop("spin")
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
    ddx_solvation : str | float | Sequence[Any] | None, optional
        Arguments for ``"ddX-solvation"``. Defaults to ``None``.
    solvation : tuple[str, *tuple[Any, ...]] | None, optional
        Generic interaction tuple ``(name, *args)`` passed to
        ``calc.add(name, *args)``. Defaults to ``None``.
    post_processing : Sequence[str] | str | None, optional
        Post-processing specification(s) passed to ``calc.add(...)``.
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
    ddx_solvation: str | float | Sequence[Any] | None = None
    solvation: tuple[str, *tuple[Any, ...]] | None = None
    post_processing: Sequence[str] | str | None = None

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
            normalized[key] = value
        return cls(**normalized)


TbliteOptions = Options


@dataclass(frozen=True)
class TbliteResult:
    """
    Container for results of a ``tblite`` single-point evaluation.

    Supports both attribute access (e.g. ``res.energy``, ``res.gradient``,
    ``res.orbital_energies``) and dictionary-style / ``.get(...)`` lookup
    using either ``tblite`` hyphenated keys (``"orbital-energies"``) or
    underscored Python names (``"orbital_energies"``).
    """

    method: str
    version: tuple[int, int, int]
    energy: Tensor | float
    energies: Tensor | np.ndarray
    gradient: Tensor | np.ndarray
    forces: Tensor | np.ndarray
    virial: Tensor | np.ndarray
    charges: Tensor | np.ndarray
    bond_orders: Tensor | np.ndarray
    dipole: Tensor | np.ndarray
    quadrupole: Tensor | np.ndarray
    orbital_energies: Tensor | np.ndarray
    orbital_occupations: Tensor | np.ndarray
    orbital_coefficients: Tensor | np.ndarray
    density_matrix: Tensor | np.ndarray
    overlap_matrix: Tensor | np.ndarray | None = None
    hamiltonian_matrix: Tensor | np.ndarray | None = None
    post_processing: dict[str, Any] | None = None
    angular_momenta: Tensor | np.ndarray | None = None
    orbital_map: Tensor | np.ndarray | None = None
    shell_map: Tensor | np.ndarray | None = None

    def get(self, key: str) -> Any:
        """Retrieve a result quantity by its ``tblite`` or Python name."""
        attr = _PROPERTY_KEY_MAP.get(key, key.replace("-", "_"))
        if not hasattr(self, attr):
            raise KeyError(f"Unknown tblite result property '{key}'.")
        val = getattr(self, attr)
        if val is None and attr in ("overlap_matrix", "hamiltonian_matrix"):
            raise KeyError(
                f"Property '{key}' requires 'save_integrals=True' in Options."
            )
        return val

    def __getitem__(self, key: str) -> Any:
        return self.get(key)


_PROPERTY_KEY_MAP: dict[str, str] = {
    "energy": "energy",
    "energies": "energies",
    "gradient": "gradient",
    "forces": "forces",
    "virial": "virial",
    "charges": "charges",
    "bond-orders": "bond_orders",
    "bond_orders": "bond_orders",
    "dipole": "dipole",
    "quadrupole": "quadrupole",
    "orbital-energies": "orbital_energies",
    "orbital_energies": "orbital_energies",
    "orbital-occupations": "orbital_occupations",
    "orbital_occupations": "orbital_occupations",
    "orbital-coefficients": "orbital_coefficients",
    "orbital_coefficients": "orbital_coefficients",
    "density-matrix": "density_matrix",
    "density_matrix": "density_matrix",
    "overlap-matrix": "overlap_matrix",
    "overlap_matrix": "overlap_matrix",
    "hamiltonian-matrix": "hamiltonian_matrix",
    "hamiltonian_matrix": "hamiltonian_matrix",
    "post-processing-dict": "post_processing",
    "post_processing": "post_processing",
    "angular-momenta": "angular_momenta",
    "angular_momenta": "angular_momenta",
    "orbital-map": "orbital_map",
    "orbital_map": "orbital_map",
    "shell-map": "shell_map",
    "shell_map": "shell_map",
}


def get_tblite_version() -> tuple[int, int, int]:
    """Return the active native ``tblite`` library version tuple."""
    import tblite.library

    ver = tblite.library.get_version()
    return int(ver[0]), int(ver[1]), int(ver[2])


def _to_numpy_numbers(
    numbers: Tensor | np.ndarray | Sequence[int],
) -> np.ndarray:
    if isinstance(numbers, torch.Tensor):
        return numbers.detach().cpu().numpy().astype(np.int32, copy=False)
    return np.asarray(numbers, dtype=np.int32)


def _to_numpy_positions(
    positions: Tensor | np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    if isinstance(positions, torch.Tensor):
        return positions.detach().cpu().numpy().astype(np.float64, copy=True)
    return np.asarray(positions, dtype=np.float64)


def _to_scalar_float(value: Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _to_scalar_int(value: Tensor | float | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(round(float(value.detach().cpu().item())))
    return int(round(float(value)))


def _coerce_structure(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    *,
    charge: Tensor | float | int | None = None,
    uhf: Tensor | int | float | None = None,
    spin: Tensor | int | float | None = None,
) -> Structure:
    if uhf is not None and spin is not None:
        if _to_scalar_int(uhf) != _to_scalar_int(spin):
            raise ValueError("Conflicting 'uhf' and 'spin' keyword arguments.")
    override_uhf = uhf if uhf is not None else spin

    if isinstance(structure, Structure):
        out = structure
    elif isinstance(structure, Mapping):
        out = Structure(
            numbers=structure["numbers"],
            positions=structure["positions"],
            charge=structure.get("charge"),
            uhf=structure.get("uhf", structure.get("spin")),
            lattice=structure.get("lattice"),
            periodic=structure.get("periodic"),
        )
    elif hasattr(structure, "numbers") and hasattr(structure, "positions"):
        struct_uhf = getattr(structure, "uhf", None)
        if struct_uhf is None:
            struct_uhf = getattr(structure, "spin", None)
        out = Structure(
            numbers=structure.numbers,
            positions=structure.positions,
            charge=getattr(structure, "charge", None),
            uhf=struct_uhf,
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
    **kwargs: Any,
) -> Options:
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
    if kwargs:
        override_opts = Options.from_mapping(kwargs)
        for k in kwargs:
            field_name = _OPTION_KEY_ALIASES.get(k, k)
            updates[field_name] = getattr(override_opts, field_name)

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
    as_tensor: bool | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    **option_kwargs: Any,
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
        Calculator options. Can also be passed directly as ``**option_kwargs``.
    charge : Tensor | float | int | None, optional
        Optional override for ``structure.charge``.
    uhf : Tensor | int | float | None, optional
        Optional override for ``structure.uhf``.
    spin : Tensor | int | float | None, optional
        Alias for ``uhf``.
    as_tensor : bool | None, optional
        If ``True``, return array quantities as :class:`~torch.Tensor` (and
        ``energy`` as a 0-D :class:`~torch.Tensor`). If ``False``, return
        NumPy arrays (and ``energy`` as ``float``). When ``None`` (default),
        inferred from ``isinstance(structure.positions, torch.Tensor)``.
    dtype : torch.dtype | None, optional
        Target floating-point dtype when returning tensors. Defaults to
        ``structure.positions.dtype`` if ``positions`` is a floating tensor,
        else ``torch.float64``.
    device : torch.device | None, optional
        Target device when returning tensors. Defaults to
        ``structure.positions.device`` if ``positions`` is a tensor.
    **option_kwargs : Any
        Individual option overrides (e.g. ``accuracy=1e-6``, ``max_iter=300``,
        ``electric_field=...``, ``solvation=...``).

    Returns
    -------
    TbliteResult
        Computed ``tblite`` single-point quantities.
    """
    mol = _coerce_structure(structure, charge=charge, uhf=uhf, spin=spin)
    opts = _resolve_options(method=method, options=options, **option_kwargs)

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
    pos_np = _to_numpy_positions(mol.positions)
    chrg_val = _to_scalar_float(mol.charge)
    uhf_val = _to_scalar_int(mol.uhf)
    lat_np = (
        _to_numpy_positions(mol.lattice) if mol.lattice is not None else None
    )
    per_np = (
        (
            mol.periodic.detach().cpu().numpy().astype(bool, copy=False)
            if isinstance(mol.periodic, torch.Tensor)
            else np.asarray(mol.periodic, dtype=bool)
        )
        if mol.periodic is not None
        else None
    )

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
        if field_np.size != 3:
            raise ValueError(
                f"Expected a 3-component electric field vector, got shape {field_np.shape}."
            )
        field_vec = ffi.new("double[]", [float(x) for x in field_np.reshape(3)])
        calc.add("electric-field", field_vec)
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
    if opts.ddx_solvation is not None:
        _add_solvation_arg(calc, "ddX-solvation", opts.ddx_solvation)
    if opts.solvation is not None:
        calc.add(opts.solvation[0], *opts.solvation[1:])
    if opts.post_processing is not None:
        if isinstance(opts.post_processing, str):
            calc.add(opts.post_processing)
        else:
            for item in opts.post_processing:
                calc.add(item)

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
    target_dtype = (
        dtype
        if dtype is not None
        else (
            mol.positions.dtype
            if isinstance(mol.positions, torch.Tensor)
            and mol.positions.is_floating_point()
            else torch.float64
        )
    )

    def _conv_float(val: Any) -> Tensor | np.ndarray:
        arr = np.asarray(val, dtype=np.float64)
        if return_tensors:
            return torch.as_tensor(
                arr, dtype=target_dtype, device=target_device
            )
        return arr

    def _conv_int(val: Any) -> Tensor | np.ndarray:
        arr = np.asarray(val, dtype=np.int64)
        if return_tensors:
            return torch.as_tensor(arr, dtype=torch.long, device=target_device)
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
        virial=_conv_float(res.get("virial")),
        charges=_conv_float(res.get("charges")),
        bond_orders=_conv_float(res.get("bond-orders")),
        dipole=_conv_float(res.get("dipole")),
        quadrupole=_conv_float(res.get("quadrupole")),
        orbital_energies=_conv_float(res.get("orbital-energies")),
        orbital_occupations=_conv_float(res.get("orbital-occupations")),
        orbital_coefficients=_conv_float(res.get("orbital-coefficients")),
        density_matrix=_conv_float(res.get("density-matrix")),
        overlap_matrix=overlap_out,
        hamiltonian_matrix=hamiltonian_out,
        post_processing=res.get("post-processing-dict"),
        angular_momenta=_conv_int(calc.get("angular-momenta")),
        orbital_map=_conv_int(calc.get("orbital-map")),
        shell_map=_conv_int(calc.get("shell-map")),
    )


def get_tblite_property(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    property_name: str,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Any:
    """
    Evaluate and return a single ``tblite`` property for ``structure``.

    Automatically enables ``save_integrals=True`` when ``"overlap-matrix"`` or
    ``"hamiltonian-matrix"`` is requested.
    """
    attr = _PROPERTY_KEY_MAP.get(property_name, property_name.replace("-", "_"))
    if attr in ("overlap_matrix", "hamiltonian_matrix"):
        kwargs = dict(kwargs)
        kwargs.setdefault("save_integrals", True)
    res = tblite_singlepoint(
        structure,
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )
    return res.get(property_name)


def get_tblite_energy(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | float:
    """Return total ``tblite`` energy in Hartree for ``structure``."""
    return get_tblite_property(
        structure,
        "energy",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_energies(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return atom-resolved ``tblite`` energies ``(nat,)`` in Hartree."""
    return get_tblite_property(
        structure,
        "energies",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_gradient(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return Cartesian ``tblite`` gradient ``(nat, 3)`` in Hartree/Bohr."""
    return get_tblite_property(
        structure,
        "gradient",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_forces(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return Cartesian ``tblite`` forces ``(nat, 3)`` (``-gradient``) in Hartree/Bohr."""
    return get_tblite_property(
        structure,
        "forces",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_virial(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return ``tblite`` virial tensor ``(3, 3)`` in Hartree."""
    return get_tblite_property(
        structure,
        "virial",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_charges(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return atomic partial charges ``(nat,)`` in elementary charge units."""
    return get_tblite_property(
        structure,
        "charges",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_bond_orders(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return Wiberg bond orders ``(nat, nat)``."""
    return get_tblite_property(
        structure,
        "bond-orders",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_dipole(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return molecular dipole moment ``(3,)`` in e*Bohr."""
    return get_tblite_property(
        structure,
        "dipole",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_quadrupole(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return molecular traceless quadrupole moment ``(6,)`` (xx, yx, yy, zx, zy, zz) in e*Bohr^2."""
    return get_tblite_property(
        structure,
        "quadrupole",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_orbital_energies(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return orbital energies ``(norb,)`` or ``(2, norb)`` in Hartree."""
    return get_tblite_property(
        structure,
        "orbital-energies",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_orbital_occupations(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return orbital occupations ``(norb,)`` or ``(2, norb)`` in electrons."""
    return get_tblite_property(
        structure,
        "orbital-occupations",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_orbital_coefficients(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return orbital coefficients ``(norb, norb)`` or ``(2, norb, norb)``."""
    return get_tblite_property(
        structure,
        "orbital-coefficients",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_density_matrix(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return AO density matrix ``(norb, norb)`` or ``(2, norb, norb)``."""
    return get_tblite_property(
        structure,
        "density-matrix",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_overlap_matrix(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return AO overlap matrix ``(norb, norb)`` (enables ``save_integrals=True``)."""
    return get_tblite_property(
        structure,
        "overlap-matrix",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )


def get_tblite_hamiltonian_matrix(
    structure: Structure | MctcStructure | Mapping[str, Any] | Any,
    method: str | dict[str, Any] | Options | None = None,
    *,
    options: Options | Mapping[str, Any] | None = None,
    as_tensor: bool | None = None,
    **kwargs: Any,
) -> Tensor | np.ndarray:
    """Return core AO Hamiltonian matrix ``(norb, norb)`` in Hartree (enables ``save_integrals=True``)."""
    return get_tblite_property(
        structure,
        "hamiltonian-matrix",
        method,
        options=options,
        as_tensor=as_tensor,
        **kwargs,
    )
