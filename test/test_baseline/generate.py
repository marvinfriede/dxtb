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
Generate the reference data (T0.2).

Run from the repository root::

    python -m test.test_baseline.generate                  # everything
    python -m test.test_baseline.generate --systems H2O --methods gfn1

Existing files are overwritten. Each file holds the quantities, the inputs
and the metadata (dxtb commit, versions, driver, SCF settings). Where the
test suite already compares with the standalone xtb program (SCF energies in
``test/test_scf/samples.py``), the comparison is recorded in the metadata
(``xtb_comparison``).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .compute import DRIVERS, METHODS, compute, groups_for, inputs_of, metadata
from .molecules import SYSTEMS, System
from .refdata import reference_path, save_reference

XTB_SCF_ENERGIES: dict[str, dict[str, float]] = {
    # test/test_scf/samples.py (same geometries, from tad_mctc.data.molecules)
    "H2O": {"gfn1": -5.8052489623704e00, "gfn2": -5.1041590251073e00},
    "CH4": {"gfn1": -4.3393059719255e00, "gfn2": -4.2422934998888e00},
    "PbH4-BiH3": {"gfn1": -7.6074262079844e00, "gfn2": -8.1872417734706e00},
    # first conformer is the undisplaced LYS_xao structure
    "LYS_xao-conformers": {
        "gfn1": -4.8850798066902e01,
        "gfn2": -4.6438843703081e01,
    },
}
"""SCF energies (``energy.scf``) of the standalone xtb program."""


def _xtb_comparison(system: System, method: str, values: dict) -> dict | None:
    ref = XTB_SCF_ENERGIES.get(system.name, {}).get(method)
    if ref is None or "energy.scf" not in values:
        return None
    e = float(np.ravel(values["energy.scf"])[0])
    return {
        "quantity": "energy.scf" + ("[0]" if system.batched else ""),
        "xtb": ref,
        "dxtb": e,
        "deviation": e - ref,
        "source": "test/test_scf/samples.py",
    }


def generate(system: System, method: str, driver: str) -> None:
    """Compute and write one reference file."""
    start = time.perf_counter()
    values = compute(system, method, driver, groups_for(system))
    meta = metadata(system, method, driver)
    meta["generation_seconds"] = round(time.perf_counter() - start, 1)
    xtb = _xtb_comparison(system, method, values)
    if xtb is not None:
        meta["xtb_comparison"] = xtb

    path = reference_path(method, driver, system.name)
    save_reference(path, values, inputs_of(system), meta)
    print(f"{path} ({meta['generation_seconds']} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--drivers", nargs="*", default=list(DRIVERS))
    args = parser.parse_args()

    for name in args.systems:
        for method in args.methods:
            for driver in args.drivers:
                generate(SYSTEMS[name], method, driver)


if __name__ == "__main__":
    main()
