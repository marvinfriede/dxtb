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
Computation of the reference quantities with the current dxtb API (T0.2).

This is the only module of the reference data that depends on the dxtb API;
it is rewritten when the API changes (TB), while the reference files and
``refdata.py`` stay untouched.

Generation rules: a fresh calculator for every quantity, result cache
disabled, float64, tight SCF convergence. All derivatives are taken by
autograd of the energy through the unrolled SCF (``scf_mode="full"``):

- forces: first derivative of the energy (positions);
- Hessian: derivative of the forces (``derived_quantity="forces"``);
- dipole: first derivative of the energy (electric field);
- polarizability: derivative of the dipole (``use_analytical=False``);
- hyperpolarizability: derivative of the polarizability;
- dipole derivative: positions derivative of the dipole
  (``use_analytical_dipmom=False``);
- polarizability derivative: positions derivative of the polarizability.
"""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.typing import Tensor
from dxtb.components.field import new_efield

from .molecules import System

__all__ = [
    "METHODS",
    "DRIVERS",
    "QUANTITY_GROUPS",
    "SCF_OPTIONS",
    "compute",
    "inputs_of",
    "metadata",
]

METHODS = {"gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
DRIVERS = ("libcint", "pytorch")

SCF_OPTIONS: dict[str, Any] = {
    "scf_mode": "full",
    "mixer": "anderson",
    "maxiter": 300,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
    "cache_enabled": False,
    "verbosity": 0,
}
"""Options of every reference calculation (plus driver and batch mode)."""

QUANTITY_GROUPS: dict[str, tuple[str, ...]] = {
    "energy": ("energy", "charges"),
    "forces": ("forces",),
    "dipole": ("dipole",),
    "second": ("hessian", "polarizability", "dipole_deriv"),
    "third": ("hyperpolarizability", "pol_deriv"),
}
"""Quantities by derivative order; ``energy`` also yields per-term energies,
shell charges and (GFN2) atomic multipoles."""

BATCH_GROUPS = ("energy", "forces")
"""Groups computed for batched systems."""

DD = {"device": torch.device("cpu"), "dtype": torch.float64}


def _calculator(
    system: System, method: str, driver: str, field: Tensor | None = None
) -> Calculator:
    opts = dict(SCF_OPTIONS, int_driver=driver, batch_mode=system.batch_mode)
    interaction = [new_efield(field)] if field is not None else None
    return Calculator(
        system.numbers,
        METHODS[method],
        opts=opts,
        interaction=interaction,
        **DD,
    )


def _args(system: System) -> tuple[Tensor, Tensor | None]:
    return system.charge, system.spin


def _energy(system: System, method: str, driver: str) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver)
    pos = system.positions.clone()
    chrg, spin = _args(system)

    res = calc.singlepoint(pos, chrg, spin)
    out: dict[str, Tensor] = {"energy": res.total.sum(-1)}

    for label, e in res.cenergies.items():
        out[f"energy.{label}"] = e.sum(-1)

    icache = calc.interactions.get_cache(
        numbers=calc.numbers, positions=pos, ihelp=calc.ihelp
    )
    eint = calc.interactions.get_energy_as_dict(res.charges, icache, calc.ihelp)
    for label, e in eint.items():
        out[f"energy.{label}"] = e.sum(-1)

    out["energy.scf"] = res.scf.sum(-1)
    out["energy.fermi"] = res.fenergy.sum(-1)
    out["energy.h0"] = out["energy.scf"] - sum(e.sum(-1) for e in eint.values())

    mono = res.charges.mono
    out["shell_charges"] = calc.ihelp.reduce_orbital_to_shell(mono)
    out["charges"] = calc.ihelp.reduce_orbital_to_atom(mono)
    if res.charges.dipole is not None:
        out["atomic_dipoles"] = res.charges.dipole
    if res.charges.quad is not None:
        out["atomic_quadrupoles"] = res.charges.quad

    out["iterations"] = res.iterations
    return out


def _forces(system: System, method: str, driver: str) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver)
    pos = system.positions.clone().requires_grad_(True)
    if not system.batched:
        return {"forces": calc.forces(pos, *_args(system))}

    # `Calculator.forces` differentiates the per-system energies with
    # `torch.autograd.grad`, which needs a scalar and raises for batches
    # (baseline finding). The systems are independent, so the gradient of
    # the summed energy gives the forces of every entry.
    energy = calc.energy(pos, *_args(system))
    (grad,) = torch.autograd.grad(energy.sum(), pos)
    return {"forces": -grad}


def _field() -> Tensor:
    return torch.zeros(3, **DD, requires_grad=True)


def _dipole(system: System, method: str, driver: str) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver, field=_field())
    return {"dipole": calc.dipole(system.positions.clone(), *_args(system))}


def _hessian(system: System, method: str, driver: str) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver)
    pos = system.positions.clone().requires_grad_(True)
    return {"hessian": calc.hessian(pos, *_args(system))}


def _polarizability(
    system: System, method: str, driver: str
) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver, field=_field())
    pos = system.positions.clone()
    alpha = calc.polarizability(pos, *_args(system), use_analytical=False)
    return {"polarizability": alpha}


def _dipole_deriv(
    system: System, method: str, driver: str
) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver, field=_field())
    pos = system.positions.clone().requires_grad_(True)
    dmu = calc.dipole_deriv(pos, *_args(system), use_analytical_dipmom=False)
    return {"dipole_deriv": dmu}


def _hyperpolarizability(
    system: System, method: str, driver: str
) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver, field=_field())
    pos = system.positions.clone()
    return {
        "hyperpolarizability": calc.hyperpolarizability(pos, *_args(system))
    }


def _pol_deriv(system: System, method: str, driver: str) -> dict[str, Tensor]:
    calc = _calculator(system, method, driver, field=_field())
    pos = system.positions.clone().requires_grad_(True)
    return {"pol_deriv": calc.pol_deriv(pos, *_args(system))}


_FUNCS: dict[str, Callable[[System, str, str], dict[str, Tensor]]] = {
    "energy": _energy,
    "charges": lambda *_: {},  # part of `_energy`
    "forces": _forces,
    "dipole": _dipole,
    "hessian": _hessian,
    "polarizability": _polarizability,
    "dipole_deriv": _dipole_deriv,
    "hyperpolarizability": _hyperpolarizability,
    "pol_deriv": _pol_deriv,
}


def groups_for(system: System) -> tuple[str, ...]:
    """Quantity groups computed for a system."""
    return BATCH_GROUPS if system.batched else tuple(QUANTITY_GROUPS)


def compute(
    system: System, method: str, driver: str, groups: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """
    Compute the quantities of the given groups.

    Every quantity uses a fresh calculator.

    Returns
    -------
    dict[str, np.ndarray]
        Quantities as float64 arrays (``iterations`` as int).
    """
    out: dict[str, np.ndarray] = {}
    for group in groups:
        for quantity in QUANTITY_GROUPS[group]:
            for key, value in _FUNCS[quantity](system, method, driver).items():
                out[key] = value.detach().cpu().numpy()
    return out


def inputs_of(system: System) -> dict[str, np.ndarray]:
    """Inputs of a system as plain arrays."""
    inputs = {
        "numbers": system.numbers.numpy(),
        "positions": system.positions.numpy(),
        "charge": system.charge.numpy(),
    }
    if system.spin is not None:
        inputs["spin"] = system.spin.numpy()
    return inputs


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_commit() -> str:
    """Current commit, with ``-dirty`` if ``src/`` has local changes."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--", ":/src")
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def metadata(system: System, method: str, driver: str) -> dict[str, Any]:
    """Metadata stored with a reference file."""
    pkgs = ["dxtb", "torch", "numpy", "tad-mctc", "tad-dftd3", "tad-dftd4"]
    pkgs += ["tad-multicharge", "tad-libcint"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = version(p)
        except Exception:  # pragma: no cover
            versions[p] = "not installed"

    return {
        "system": system.name,
        "group": system.group,
        "note": system.note,
        "method": method,
        "driver": driver,
        "batch_mode": system.batch_mode,
        "scf": dict(SCF_OPTIONS),
        "dtype": "float64",
        "device": "cpu",
        "units": "atomic units (bohr, hartree, e)",
        "dxtb_commit": _git_commit(),
        "versions": versions,
        "torch_build": torch.__version__,
        "python": platform.python_version(),
        "groups": list(groups_for(system)),
        "derivatives": "autograd through the unrolled SCF, see compute.py",
    }
