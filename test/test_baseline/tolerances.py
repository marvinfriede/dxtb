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
Tolerances of the reference data comparisons (T0.2, fixed in T0.3).

``REGRESSION`` compares a recomputation with the stored dxtb reference
(same code path, float64). ``TBLITE`` compares dxtb with the independent
tblite reference, whose higher derivatives are finite differences. See
``docs/plan/T0-baseline-report.md`` for the observed spread the values are
based on.
"""

from __future__ import annotations

from .refdata import Tolerance

__all__ = ["REGRESSION", "TBLITE", "NOT_COMPARED", "KNOWN_WRONG", "known_wrong"]

NOT_COMPARED = frozenset({"iterations"})
"""Informational quantities."""

REGRESSION: dict[str, Tolerance] = {
    "energy": Tolerance(1e-9),
    "shell_charges": Tolerance(1e-8),
    "charges": Tolerance(1e-8),
    "atomic_dipoles": Tolerance(1e-8),
    "atomic_quadrupoles": Tolerance(1e-8),
    "forces": Tolerance(1e-8),
    "dipole": Tolerance(1e-8),
    "hessian": Tolerance(1e-7),
    "polarizability": Tolerance(1e-6, 1e-8),
    "dipole_deriv": Tolerance(1e-7),
    # third order of systems with degenerate orbitals (e.g., NH4+) differs
    # by up to 4e-5 between the integral drivers
    "hyperpolarizability": Tolerance(1e-3, 1e-6),
    "pol_deriv": Tolerance(1e-4, 1e-6),
}

TBLITE: dict[str, Tolerance] = {
    # systematic difference of ~5e-8 Eh per atom
    "energy": Tolerance(1e-6, 1e-7),
    "charges": Tolerance(1e-6),
    "forces": Tolerance(1e-6),
    "dipole": Tolerance(1e-6),
    # tblite side by finite differences (steps in its metadata)
    "hessian": Tolerance(1e-5),
    "polarizability": Tolerance(1e-4, 1e-5),
    "dipole_deriv": Tolerance(1e-4),
    "hyperpolarizability": Tolerance(1e-1, 5e-3),
    "pol_deriv": Tolerance(5e-2, 5e-3),
}


_DEGENERATE = (
    "wrong autograd derivative at exactly degenerate orbitals (eigensolver "
    "backward); finite differences of dxtb agree with tblite"
)

KNOWN_WRONG: dict[tuple[str, str | None], dict[str, str]] = {
    # (system, method or None for both): {quantity: reason}
    ("C6H6", None): {
        q: _DEGENERATE
        for q in (
            "hessian",
            "polarizability",
            "dipole_deriv",
            "hyperpolarizability",
            "pol_deriv",
        )
    },
    ("Fe(CO)5", None): {
        q: _DEGENERATE for q in ("hyperpolarizability", "pol_deriv")
    },
    ("NO2", "gfn2"): {
        q: (
            "wrong third derivative for the open-shell system (autograd); "
            "finite differences of dxtb agree with tblite"
        )
        for q in ("hyperpolarizability", "pol_deriv")
    },
}
"""
Quantities whose stored dxtb value is known to be wrong (see the baseline
report). They are not enforced as regression targets; the test checks that
they still disagree with tblite and fails once they agree (fixed: regenerate
the reference and remove the entry).
"""


def known_wrong(system: str, method: str) -> dict[str, str]:
    """Known wrong quantities of a system and method."""
    out = dict(KNOWN_WRONG.get((system, None), {}))
    out.update(KNOWN_WRONG.get((system, method), {}))
    return out
