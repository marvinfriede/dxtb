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
Independent reference data from tblite (T0.2).

The standalone xtb implementation of the Grimme group, through the tblite
Python API (``pip install tblite``; not a dependency of dxtb or of the test
suite). Run from the repository root::

    python -m test.test_baseline.generate_tblite
    python -m test.test_baseline.generate_tblite --systems H2O --methods gfn2

Writes ``reference/tblite/<method>/<system>.npz`` with the same keys and
shapes as the dxtb reference files, so both can be compared with
``refdata.compare``:

- analytical from tblite: ``energy``, ``forces`` (negative gradient),
  ``charges``, ``dipole`` (plus ``tblite.quadrupole``, ``tblite.energies``,
  which have no dxtb counterpart in the reference files);
- central finite differences (steps in the metadata): ``hessian`` (of the
  gradient), ``polarizability`` (of the dipole in the field),
  ``hyperpolarizability`` (of the polarizability in the field),
  ``dipole_deriv`` (of the dipole in the positions), ``pol_deriv`` (of the
  polarizability in the positions).

Conventions as in dxtb: ``mu = -dE/dF``, ``alpha_ij = dmu_i/dF_j``,
``beta_ijk = dalpha_ij/dF_k``, ``dipole_deriv[i, a, x] = dmu_i/dR_ax``,
``pol_deriv[i, j, a, x] = dalpha_ij/dR_ax``. The electronic temperature is
300 K in both programs.
"""

from __future__ import annotations

import argparse
import time
from importlib.metadata import version
from typing import Any

import numpy as np

from .molecules import SYSTEMS, System
from .refdata import REFERENCE_DIR, save_reference

TBLITE_METHODS = {"gfn1": "GFN1-xTB", "gfn2": "GFN2-xTB"}

ACCURACY = 1e-4
"""tblite accuracy (scales the SCF convergence thresholds)."""

STEPS = {
    "positions": 1e-4,  # bohr; Hessian, dipole derivative
    "field": 1e-4,  # a.u.; polarizability
    "field_hyper": 5e-4,  # a.u.; outer step of the hyperpolarizability
    "positions_pol": 1e-3,  # bohr; outer step of the pol. derivative
}
"""Finite difference steps."""


class Molecule:
    """Single-point evaluations of one molecule with tblite."""

    def __init__(
        self,
        method: str,
        numbers: np.ndarray,
        positions: np.ndarray,
        charge: float,
        uhf: int | None,
    ) -> None:
        self.method = TBLITE_METHODS[method]
        self.numbers = numbers
        self.positions = positions
        self.charge = charge
        self.uhf = uhf

    def run(
        self, positions: np.ndarray | None = None, field: Any = None
    ) -> Any:
        # pylint: disable=import-outside-toplevel
        from tblite.interface import Calculator
        from tblite.library import ffi

        pos = self.positions if positions is None else positions
        calc = Calculator(
            self.method, self.numbers, pos, charge=self.charge, uhf=self.uhf
        )
        calc.set("verbosity", 0)
        calc.set("accuracy", ACCURACY)
        if field is not None:
            # tblite 0.7.0 passes the argument on as `double *` unconverted
            vec = ffi.new("double[]", [float(f) for f in field])
            calc.add("electric-field", vec)
        return calc.singlepoint()

    def dipole(
        self, positions: np.ndarray | None = None, field: Any = None
    ) -> np.ndarray:
        return np.asarray(self.run(positions, field).get("dipole"))

    def gradient(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(self.run(positions).get("gradient"))

    def polarizability(
        self,
        positions: np.ndarray | None = None,
        field: np.ndarray | None = None,
    ) -> np.ndarray:
        """``alpha[i, j] = dmu_i/dF_j`` by central differences."""
        f0 = np.zeros(3) if field is None else field
        h = STEPS["field"]
        alpha = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            mu_p = self.dipole(positions, f0 + e)
            mu_m = self.dipole(positions, f0 - e)
            alpha[:, j] = (mu_p - mu_m) / (2 * h)
        return alpha

    def hessian(self) -> np.ndarray:
        nat = len(self.numbers)
        h = STEPS["positions"]
        hess = np.zeros((nat, 3, nat, 3))
        for a in range(nat):
            for x in range(3):
                p = self.positions.copy()
                p[a, x] += h
                gp = self.gradient(p)
                p[a, x] -= 2 * h
                gm = self.gradient(p)
                hess[a, x] = (gp - gm) / (2 * h)
        # symmetrize: the finite difference error is not symmetric
        flat = hess.reshape(3 * nat, 3 * nat)
        return (0.5 * (flat + flat.T)).reshape(nat, 3, nat, 3)

    def hyperpolarizability(self) -> np.ndarray:
        h = STEPS["field_hyper"]
        beta = np.zeros((3, 3, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            a_p = self.polarizability(field=e)
            a_m = self.polarizability(field=-e)
            beta[:, :, k] = (a_p - a_m) / (2 * h)
        return beta

    def dipole_deriv(self) -> np.ndarray:
        nat = len(self.numbers)
        h = STEPS["positions"]
        out = np.zeros((3, nat, 3))
        for a in range(nat):
            for x in range(3):
                p = self.positions.copy()
                p[a, x] += h
                mu_p = self.dipole(p)
                p[a, x] -= 2 * h
                mu_m = self.dipole(p)
                out[:, a, x] = (mu_p - mu_m) / (2 * h)
        return out

    def pol_deriv(self) -> np.ndarray:
        nat = len(self.numbers)
        h = STEPS["positions_pol"]
        out = np.zeros((3, 3, nat, 3))
        for a in range(nat):
            for x in range(3):
                p = self.positions.copy()
                p[a, x] += h
                a_p = self.polarizability(p)
                p[a, x] -= 2 * h
                a_m = self.polarizability(p)
                out[:, :, a, x] = (a_p - a_m) / (2 * h)
        return out


def _molecules(system: System, method: str) -> list[Molecule]:
    """One molecule per batch entry (padding removed)."""
    numbers = system.numbers.numpy()
    positions = system.positions.numpy()
    charge = system.charge.numpy()
    uhf = None if system.spin is None else int(system.spin.item())

    if not system.batched:
        return [Molecule(method, numbers, positions, float(charge), uhf)]

    out = []
    for nums, pos, chrg in zip(numbers, positions, charge):
        nat = int(np.count_nonzero(nums))
        out.append(Molecule(method, nums[:nat], pos[:nat], float(chrg), uhf))
    return out


def compute(system: System, method: str) -> dict[str, np.ndarray]:
    """tblite quantities for a system (same keys and shapes as dxtb)."""
    mols = _molecules(system, method)
    nat = system.numbers.shape[-1]

    singles = [m.run() for m in mols]
    values: dict[str, list[np.ndarray]] = {
        "energy": [],
        "forces": [],
        "charges": [],
        "dipole": [],
        "tblite.energies": [],
        "tblite.quadrupole": [],
    }
    for m, res in zip(mols, singles):
        n = len(m.numbers)
        pad = ((0, nat - n),)
        values["energy"].append(np.asarray(res.get("energy")))
        values["forces"].append(
            np.pad(-np.asarray(res.get("gradient")), pad + ((0, 0),))
        )
        values["charges"].append(np.pad(np.asarray(res.get("charges")), pad))
        values["dipole"].append(np.asarray(res.get("dipole")))
        values["tblite.energies"].append(
            np.pad(np.asarray(res.get("energies")), pad)
        )
        values["tblite.quadrupole"].append(np.asarray(res.get("quadrupole")))

    out = {
        k: (np.stack(v) if system.batched else v[0]) for k, v in values.items()
    }
    if system.batched:
        return out

    m = mols[0]
    out["hessian"] = m.hessian()
    out["polarizability"] = m.polarizability()
    out["dipole_deriv"] = m.dipole_deriv()
    out["hyperpolarizability"] = m.hyperpolarizability()
    out["pol_deriv"] = m.pol_deriv()
    return out


def generate(system: System, method: str) -> None:
    start = time.perf_counter()
    values = compute(system, method)
    inputs = {
        "numbers": system.numbers.numpy(),
        "positions": system.positions.numpy(),
        "charge": system.charge.numpy(),
    }
    if system.spin is not None:
        inputs["spin"] = system.spin.numpy()

    meta = {
        "system": system.name,
        "method": method,
        "program": f"tblite {version('tblite')} (Python API)",
        "accuracy": ACCURACY,
        "electronic_temperature": "300 K (tblite default 9.5e-4 Eh)",
        "finite_difference_steps": STEPS,
        "units": "atomic units (bohr, hartree, e)",
        "generation_seconds": round(time.perf_counter() - start, 1),
    }
    path = REFERENCE_DIR / "tblite" / method / f"{system.name}.npz"
    save_reference(path, values, inputs, meta)
    print(f"{path} ({meta['generation_seconds']} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="tblite reference data")
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    parser.add_argument("--methods", nargs="*", default=list(TBLITE_METHODS))
    args = parser.parse_args()

    for name in args.systems:
        for method in args.methods:
            generate(SYSTEMS[name], method)


if __name__ == "__main__":
    main()
