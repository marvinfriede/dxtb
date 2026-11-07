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
Live reference helper executing ``tblite.interface.Calculator`` dynamically.

Provides a deterministic wrapper around tblite 0.8.0 so pytest suites can
validate dxtb against live upstream tblite calculations with matched physical
settings (coordinates in Bohr, charge, spin/uhf, electronic temperature in
Hartree, SCF accuracy, max iterations, and optional implicit solvation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import tblite.library
import torch
from tad_mctc.units.energy import KELVIN2AU
from tblite.interface import Calculator as TbliteCalculator

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.typing import Tensor
from dxtb.components.field import new_efield

from .refdata import Deviation, Tolerance, compare

__all__ = [
    "DEFAULT_LIVE_TOLERANCES",
    "EXPECTED_TBLITE_VERSION",
    "TbliteLiveResult",
    "assert_tblite_agrees",
    "evaluate_dxtb_reference",
    "get_tblite_version",
    "run_tblite_live",
]

EXPECTED_TBLITE_VERSION: tuple[int, int, int] = (0, 8, 0)
"""Expected upstream tblite library version for the live reference suite."""

DEFAULT_LIVE_TOLERANCES: dict[str, Tolerance] = {
    "energy": Tolerance(atol=1.0e-7, rtol=1.0e-7),
    "gradient": Tolerance(atol=1.0e-6, rtol=1.0e-5),
    "forces": Tolerance(atol=1.0e-6, rtol=1.0e-5),
    "charges": Tolerance(atol=1.0e-6, rtol=1.0e-5),
    "dipole": Tolerance(atol=1.0e-6, rtol=1.0e-5),
}
"""Default comparison tolerances between dxtb and live tblite 0.8.0."""

_DXTB_METHODS = {
    "gfn1": GFN1_XTB,
    "gfn1-xtb": GFN1_XTB,
    "GFN1-xTB": GFN1_XTB,
    "gfn2": GFN2_XTB,
    "gfn2-xtb": GFN2_XTB,
    "GFN2-xTB": GFN2_XTB,
}

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


@dataclass(frozen=True)
class TbliteLiveResult:
    """Outputs retrieved from a live ``tblite.interface.Calculator`` run."""

    method: str
    version: tuple[int, int, int]
    energy: float
    energies: np.ndarray
    gradient: np.ndarray
    forces: np.ndarray
    charges: np.ndarray
    dipole: np.ndarray
    quadrupole: np.ndarray
    orbital_energies: np.ndarray
    orbital_occupations: np.ndarray
    density_matrix: np.ndarray
    overlap_matrix: np.ndarray | None = None
    post_processing: dict[str, Any] | None = None

    def as_reference_dict(self) -> dict[str, np.ndarray]:
        """Format primary quantities for ``refdata.compare``."""
        return {
            "energy": np.asarray(self.energy, dtype=np.float64),
            "gradient": self.gradient,
            "forces": self.forces,
            "charges": self.charges,
            "dipole": self.dipole,
        }


def get_tblite_version() -> tuple[int, int, int]:
    """Return the active native tblite library version tuple."""
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


def _to_scalar_float(value: Tensor | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _to_scalar_int(value: Tensor | float | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(round(float(value.detach().cpu().item())))
    return int(round(float(value)))


def run_tblite_live(
    method: str | dict[str, Any],
    numbers: Tensor | np.ndarray | Sequence[int],
    positions: Tensor | np.ndarray | Sequence[Sequence[float]],
    *,
    charge: Tensor | float | int | None = 0.0,
    uhf: Tensor | float | int | None = 0,
    accuracy: float = 1.0e-6,
    max_iter: int = 300,
    temperature_kelvin: float = 300.0,
    mixer_damping: float | None = None,
    guess: str | int | None = None,
    save_integrals: bool = False,
    solvation: tuple[str, *tuple[Any, ...]] | None = None,
    electric_field: Tensor | np.ndarray | Sequence[float] | None = None,
) -> TbliteLiveResult:
    """
    Execute a live single-point calculation through ``tblite.interface.Calculator``.

    Parameters
    ----------
    method : str | dict[str, Any]
        Method name (``"GFN1-xTB"``, ``"GFN2-xTB"``, ``"gfn1"``, ``"gfn2"``) or
        a tblite parametrization dictionary.
    numbers : Tensor | np.ndarray | Sequence[int]
        Atomic numbers of shape ``(nat,)``.
    positions : Tensor | np.ndarray | Sequence[Sequence[float]]
        Cartesian coordinates in Bohr of shape ``(nat, 3)``.
    charge : Tensor | float | int | None, optional
        Total molecular charge. Defaults to ``0.0``.
    uhf : Tensor | float | int | None, optional
        Number of unpaired electrons. Defaults to ``0``.
    accuracy : float, optional
        tblite SCC numerical accuracy parameter. Defaults to ``1.0e-6`` for
        tightly converged reference comparisons.
    max_iter : int, optional
        Maximum number of SCC iterations. Defaults to ``300``.
    temperature_kelvin : float, optional
        Electronic temperature in Kelvin (converted using ``KELVIN2AU`` to match
        dxtb's exact Fermi temperature in Hartree). Defaults to ``300.0``.
    mixer_damping : float | None, optional
        Optional Broyden mixer damping factor.
    guess : str | int | None, optional
        Optional initial guess (``"sad"``/``0`` or ``"eeq"``/``1``).
    save_integrals : bool, optional
        Whether to retain integral matrices in the result. Defaults to ``False``.
    solvation : tuple[str, ...] | None, optional
        Interaction tuple passed to ``calc.add(*solvation)``, e.g.
        ``("alpb-solvation", "water")`` or ``("gbsa-solvation", "water")``.
    electric_field : Tensor | np.ndarray | Sequence[float] | None, optional
        Optional uniform electric field vector of shape ``(3,)`` in atomic units.

    Returns
    -------
    TbliteLiveResult
        Computed energy, Cartesian gradient, forces, atomic charges, dipole,
        quadrupole, and orbital quantities.
    """
    resolved_method: str | dict[str, Any]
    method_label: str
    if isinstance(method, str):
        resolved_method = _METHOD_MAP.get(method, method)
        method_label = str(resolved_method)
    else:
        resolved_method = method
        method_label = "custom-xTB"

    nums_np = _to_numpy_numbers(numbers)
    pos_np = _to_numpy_positions(positions)
    chrg_val = _to_scalar_float(charge)
    uhf_val = _to_scalar_int(uhf)

    calc = TbliteCalculator(
        resolved_method,
        nums_np,
        pos_np,
        charge=chrg_val,
        uhf=uhf_val,
    )
    calc.set("verbosity", 0)
    calc.set("accuracy", float(accuracy))
    calc.set("max-iter", int(max_iter))
    calc.set("temperature", float(temperature_kelvin) * KELVIN2AU)

    if mixer_damping is not None:
        calc.set("mixer-damping", float(mixer_damping))
    if guess is not None:
        guess_code = (
            {"sad": 0, "eeq": 1}[guess.lower()]
            if isinstance(guess, str)
            else int(guess)
        )
        calc.set("guess", guess_code)
    if save_integrals:
        calc.set("save-integrals", 1)

    if solvation is not None:
        calc.add(solvation[0], *solvation[1:])
    if electric_field is not None:
        field_np = (
            electric_field.detach().cpu().numpy().astype(np.float64)
            if isinstance(electric_field, torch.Tensor)
            else np.asarray(electric_field, dtype=np.float64)
        )
        calc.add("electric-field", field_np)

    res = calc.singlepoint()
    grad = np.asarray(res.get("gradient"), dtype=np.float64)
    overlap = (
        np.asarray(res.get("overlap-matrix"), dtype=np.float64)
        if save_integrals
        else None
    )
    return TbliteLiveResult(
        method=method_label,
        version=get_tblite_version(),
        energy=float(res.get("energy")),
        energies=np.asarray(res.get("energies"), dtype=np.float64),
        gradient=grad,
        forces=-grad,
        charges=np.asarray(res.get("charges"), dtype=np.float64),
        dipole=np.asarray(res.get("dipole"), dtype=np.float64),
        quadrupole=np.asarray(res.get("quadrupole"), dtype=np.float64),
        orbital_energies=np.asarray(
            res.get("orbital-energies"), dtype=np.float64
        ),
        orbital_occupations=np.asarray(
            res.get("orbital-occupations"), dtype=np.float64
        ),
        density_matrix=np.asarray(res.get("density-matrix"), dtype=np.float64),
        overlap_matrix=overlap,
        post_processing=res.get("post-processing-dict"),
    )


def evaluate_dxtb_reference(
    method: str,
    numbers: Tensor,
    positions: Tensor,
    *,
    charge: Tensor | float | int | None = None,
    spin: Tensor | float | int | None = None,
    int_driver: str = "pytorch",
    scp_mode: str = "potential",
    mixer: str = "broyden",
    x_atol: float = 1.0e-10,
    maxiter: int = 300,
    fermi_etemp: float = 300.0,
    include_dipole: bool = True,
    extra_opts: Mapping[str, Any] | None = None,
    extra_interactions: Sequence[Any] | None = None,
) -> dict[str, Tensor]:
    """
    Evaluate dxtb energy, Cartesian gradient, atomic charges, and dipole.

    Parameters
    ----------
    method : str
        Method identifier (``"gfn1"`` or ``"gfn2"``).
    numbers : Tensor
        Atomic numbers of shape ``(nat,)``.
    positions : Tensor
        Cartesian coordinates in Bohr of shape ``(nat, 3)``.
    charge : Tensor | float | int | None, optional
        Total molecular charge. Defaults to ``None`` (neutral).
    spin : Tensor | float | int | None, optional
        Number of unpaired electrons. Defaults to ``None`` (singlet).
    int_driver : str, optional
        Integral driver (``"pytorch"`` or ``"libcint"``). Defaults to ``"pytorch"``.
    scp_mode : str, optional
        SCF representation mode (``"potential"`` or ``"charge"``). Defaults to
        ``"potential"``.
    mixer : str, optional
        SCF mixer (``"broyden"`` or ``"anderson"``). Defaults to ``"broyden"``.
    x_atol : float, optional
        SCF root tolerance. Defaults to ``1.0e-10``.
    maxiter : int, optional
        Maximum SCF iterations. Defaults to ``300``.
    fermi_etemp : float, optional
        Electronic temperature in Kelvin. Defaults to ``300.0``.
    include_dipole : bool, optional
        Whether to evaluate the molecular dipole moment via electric-field
        differentiation. Defaults to ``True``.
    extra_opts : Mapping[str, Any] | None, optional
        Additional calculator options.
    extra_interactions : Sequence[Any] | None, optional
        Additional interactions passed to ``Calculator(..., interaction=...)``.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing ``energy``, ``gradient``, ``forces``, ``charges``,
        and optionally ``dipole``, ``atomic_dipoles``, and ``atomic_quadrupoles``.
    """
    dd: dict[str, Any] = {
        "device": positions.device,
        "dtype": positions.dtype,
    }
    param = _DXTB_METHODS[method]
    opts: dict[str, Any] = {
        "int_driver": int_driver,
        "scp_mode": scp_mode,
        "mixer": mixer,
        "scf_mode": "full",
        "x_atol": x_atol,
        "x_atol_max": x_atol * 0.1,
        "maxiter": maxiter,
        "fermi_etemp": fermi_etemp,
        "verbosity": 0,
    }
    if extra_opts is not None:
        opts.update(extra_opts)

    interactions_list = (
        list(extra_interactions) if extra_interactions is not None else None
    )
    calc = Calculator(
        numbers,
        param,
        opts=opts,
        interaction=interactions_list,
        **dd,
    )
    pos = positions.detach().clone().requires_grad_(True)
    res = calc.singlepoint(pos, chrg=charge, spin=spin)
    energy = res.total.sum(-1)
    (gradient,) = torch.autograd.grad(energy, pos)
    mono = res.charges.mono
    atomic_charges = calc.ihelp.reduce_orbital_to_atom(mono)

    out: dict[str, Tensor] = {
        "energy": energy.detach(),
        "gradient": gradient.detach(),
        "forces": (-gradient).detach(),
        "charges": atomic_charges.detach(),
    }
    if res.charges.dipole is not None:
        out["atomic_dipoles"] = res.charges.dipole.detach()
    if res.charges.quadrupole is not None:
        out["atomic_quadrupoles"] = res.charges.quadrupole.detach()

    if include_dipole:
        field = torch.zeros(3, **dd, requires_grad=True)
        dipole_interactions = [new_efield(field)]
        if extra_interactions is not None:
            dipole_interactions.extend(extra_interactions)
        calc_dip = Calculator(
            numbers,
            param,
            opts=opts,
            interaction=dipole_interactions,
            **dd,
        )
        dipole = calc_dip.dipole(
            positions.detach().clone(), chrg=charge, spin=spin
        )
        out["dipole"] = dipole.detach()

    return out


def assert_tblite_agrees(
    *,
    method: str,
    structure: str,
    live: TbliteLiveResult,
    computed: Mapping[str, Tensor | np.ndarray | float],
    tolerances: Mapping[str, Tolerance] | None = None,
    keys: Sequence[str] = ("energy", "gradient", "charges", "dipole"),
) -> list[Deviation]:
    """
    Compare dxtb quantities against a live tblite result with detailed reporting.

    Errors report the method, structure, tblite version, quantity, both ``dxtb``
    and ``tblite`` values, maximum absolute and relative deviation, and
    configured tolerance.
    """
    active_tolerances = (
        DEFAULT_LIVE_TOLERANCES if tolerances is None else tolerances
    )
    ref_dict = live.as_reference_dict()
    new_dict: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in computed:
            continue
        val = computed[key]
        if isinstance(val, torch.Tensor):
            new_dict[key] = val.detach().cpu().numpy().astype(np.float64)
        else:
            new_dict[key] = np.asarray(val, dtype=np.float64)

    ver_str = ".".join(str(v) for v in live.version)
    header = (
        f"Live tblite {ver_str} comparison failed for "
        f"method={method}, structure={structure}"
    )
    deviations = compare(ref_dict, new_dict, active_tolerances, keys)
    if not all(d.ok for d in deviations):
        lines = [header]
        for dev in deviations:
            lines.append(str(dev))
            if not dev.ok:
                dxtb_val = new_dict.get(dev.key)
                tblite_val = ref_dict.get(dev.key)
                lines.append(
                    f"    dxtb[{dev.key}]   = "
                    f"{np.array2string(dxtb_val, precision=12, suppress_small=False) if dxtb_val is not None else '<missing>'}"
                )
                lines.append(
                    f"    tblite[{dev.key}] = "
                    f"{np.array2string(tblite_val, precision=12, suppress_small=False) if tblite_val is not None else '<missing>'}"
                )
        raise AssertionError("\n".join(lines))
    return deviations
