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
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, labels
from dxtb._src.typing import Tensor

from ..tblite import (
    Structure,
    get_tblite_version,
    tblite_singlepoint,
)
from ..utils import scf_strategy
from .refdata import Deviation, Tolerance, compare

__all__ = [
    "DEFAULT_CONVERGENCE_LIVE_TOLERANCES",
    "DEFAULT_LIVE_TOLERANCES",
    "EXPECTED_TBLITE_VERSION",
    "TIGHT_LIVE_TOLERANCES",
    "TbliteLiveResult",
    "assert_tblite_agrees",
    "evaluate_dxtb_reference",
    "get_tblite_version",
    "run_tblite_live",
]

EXPECTED_TBLITE_VERSION: tuple[int, int, int] = (0, 8, 0)
"""Expected upstream tblite library version for the live reference suite."""

TIGHT_LIVE_TOLERANCES: dict[str, Tolerance] = {
    "energy": Tolerance(atol=1.0e-6, rtol=1.0e-6),
    "gradient": Tolerance(atol=1.0e-8, rtol=1.0e-7),
    "forces": Tolerance(atol=1.0e-8, rtol=1.0e-7),
    "charges": Tolerance(atol=3.0e-8, rtol=1.0e-6),
    "dipole": Tolerance(atol=3.0e-8, rtol=1.0e-6),
}
"""Tight-convergence comparison tolerances between dxtb (x_atol<=1e-9) and live tblite 0.8.0 (accuracy=1e-6).

Note: Total energy carries an ~2e-7 to 8e-7 Eh constant-conversion offset
between dxtb's compiled parameter tables and tblite's Fortran parameter tables,
while analytical gradients, atomic charges, and dipoles agree to ~1e-8.
"""

DEFAULT_CONVERGENCE_LIVE_TOLERANCES: dict[str, Tolerance] = {
    "energy": Tolerance(atol=1.0e-6, rtol=1.0e-6),
    "gradient": Tolerance(atol=2.0e-6, rtol=1.0e-5),
    "forces": Tolerance(atol=2.0e-6, rtol=1.0e-5),
    "charges": Tolerance(atol=1.0e-4, rtol=1.0e-4),
    "dipole": Tolerance(atol=5.0e-5, rtol=1.0e-4),
}
"""Default-convergence comparison tolerances between dxtb (x_atol=1e-4) and live tblite 0.8.0 (accuracy=1.0)."""

DEFAULT_LIVE_TOLERANCES: dict[str, Tolerance] = TIGHT_LIVE_TOLERANCES
"""Alias for tight-convergence live comparison tolerances."""

_DXTB_METHODS = {
    "gfn1": GFN1_XTB,
    "gfn1-xtb": GFN1_XTB,
    "GFN1-xTB": GFN1_XTB,
    "gfn2": GFN2_XTB,
    "gfn2-xtb": GFN2_XTB,
    "GFN2-xTB": GFN2_XTB,
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

    def as_reference_dict(self) -> dict[str, np.ndarray]:
        """Format primary quantities for ``refdata.compare``."""
        return {
            "energy": np.asarray(self.energy, dtype=np.float64),
            "gradient": self.gradient,
            "forces": self.forces,
            "charges": self.charges,
            "dipole": self.dipole,
        }


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
    mol = Structure(
        numbers=numbers,
        positions=positions,
        charge=0.0 if charge is None else charge,
        uhf=0 if uhf is None else uhf,
    )
    res = tblite_singlepoint(
        mol,
        method,
        accuracy=accuracy,
        max_iter=max_iter,
        temperature_kelvin=temperature_kelvin,
        mixer_damping=mixer_damping,
        guess=guess,
        save_integrals=save_integrals,
        solvation=solvation,
        electric_field=electric_field,
        as_tensor=False,
    )
    return TbliteLiveResult(
        method=res.method,
        version=res.version,
        energy=float(res.energy),
        energies=np.asarray(res.energies, dtype=np.float64),
        gradient=np.asarray(res.gradient, dtype=np.float64),
        forces=np.asarray(res.forces, dtype=np.float64),
        charges=np.asarray(res.charges, dtype=np.float64),
        dipole=np.asarray(res.dipole, dtype=np.float64),
        quadrupole=np.asarray(res.quadrupole, dtype=np.float64),
        orbital_energies=np.asarray(res.orbital_energies, dtype=np.float64),
        orbital_occupations=np.asarray(
            res.orbital_occupations, dtype=np.float64
        ),
        density_matrix=np.asarray(res.density_matrix, dtype=np.float64),
        overlap_matrix=(
            np.asarray(res.overlap_matrix, dtype=np.float64)
            if res.overlap_matrix is not None
            else None
        ),
    )


def evaluate_dxtb_reference(
    method: str,
    numbers: Tensor,
    positions: Tensor,
    *,
    charge: Tensor | float | int | None = None,
    spin: Tensor | float | int | None = None,
    int_driver: str = "pytorch",
    variable: str = "potential",
    converger: str = "broyden",
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
    variable : str, optional
        SCF variable (``"potential"`` or ``"charge"``). Defaults to
        ``"potential"``.
    converger : str, optional
        SCF converger (``"broyden"`` or ``"anderson"``). Defaults to
        ``"broyden"``.
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
        "strategy": scf_strategy(converger, variable),
        "differentiation": "unrolled",
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
    energy = res.energy.sum(-1)
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
        dipole_opts = dict(
            opts,
            int_level=max(
                opts.get("int_level", labels.INTLEVEL_DIPOLE),
                labels.INTLEVEL_DIPOLE,
            ),
        )
        calc_dip = Calculator(
            numbers,
            param,
            opts=dipole_opts,
            interaction=interactions_list,
            **dd,
        )
        dipole = calc_dip.dipole(
            positions.detach().clone(), chrg=charge, spin=spin, field=field
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
