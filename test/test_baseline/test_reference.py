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
Reproduce the reference data (T0.2).

Recomputes the quantities of every reference file and compares them with
the stored dxtb values (``REGRESSION``) and with the independent tblite
values (``TBLITE``). Energies, charges, forces and dipoles run on every
CI run; second- and third-order quantities and the conformer batch are
marked ``slow`` (nightly).
"""

from __future__ import annotations

import pytest

from dxtb._src.exlibs.available import has_libcint

from .compute import DRIVERS, METHODS, QUANTITY_GROUPS, compute, groups_for
from .molecules import SYSTEMS
from .refdata import (
    REFERENCE_DIR,
    assert_close,
    compare,
    load_reference,
    reference_path,
)
from .tolerances import NOT_COMPARED, REGRESSION, TBLITE, known_wrong

pytestmark = pytest.mark.physics_values

SLOW_GROUPS = {"second", "third"}
SLOW_SYSTEMS = {"LYS_xao-conformers"}


def _cases() -> list:
    cases = []
    for name, system in SYSTEMS.items():
        for method in METHODS:
            for driver in DRIVERS:
                for group in groups_for(system):
                    marks = []
                    if group in SLOW_GROUPS or name in SLOW_SYSTEMS:
                        marks.append(pytest.mark.slow)
                    if driver == "libcint":
                        marks.append(
                            pytest.mark.skipif(
                                not has_libcint, reason="libcint not available"
                            )
                        )
                    cases.append(
                        pytest.param(
                            name,
                            method,
                            driver,
                            group,
                            marks=marks,
                            id=f"{method}-{driver}-{name}-{group}",
                        )
                    )
    return cases


@pytest.mark.parametrize("name,method,driver,group", _cases())
def test_reference(name: str, method: str, driver: str, group: str) -> None:
    system = SYSTEMS[name]
    ref, inputs, _ = load_reference(reference_path(method, driver, name))

    # the stored inputs are the ones the reference was computed for
    assert (inputs["numbers"] == system.numbers.numpy()).all()
    assert (inputs["positions"] == system.positions.numpy()).all()

    new = compute(system, method, driver, (group,))
    wrong = known_wrong(name, method)
    keys = [k for k in new if k not in NOT_COMPARED]
    good = [k for k in keys if k not in wrong]

    header = f"{method}-{driver} {name} [{group}] vs. stored dxtb reference"
    assert_close(compare(ref, new, REGRESSION, good), header)

    path = REFERENCE_DIR / "tblite" / method / f"{name}.npz"
    tblite, _, _ = load_reference(path)
    header = f"{method}-{driver} {name} [{group}] vs. tblite"
    assert_close(
        compare(tblite, new, TBLITE, [k for k in good if k in tblite]), header
    )

    bad = [k for k in keys if k in wrong]
    if not bad:
        return

    # known issues: fail as soon as they are fixed (strict expectation)
    fixed = [d for d in compare(tblite, new, TBLITE, bad) if d.ok]
    assert not fixed, (
        "Known issue fixed, regenerate the reference and update "
        "`tolerances.KNOWN_WRONG`: " + ", ".join(d.key for d in fixed)
    )
    pytest.xfail("; ".join(f"{k}: {wrong[k]}" for k in bad))


def test_groups_cover_quantities() -> None:
    """Every quantity group of the reference has a regression tolerance."""
    for quantities in QUANTITY_GROUPS.values():
        for q in quantities:
            assert q in REGRESSION
