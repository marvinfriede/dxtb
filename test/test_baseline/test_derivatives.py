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
Derivative status matrix (T0.3).

One test per cell (quantity, path, method, driver, SCF mode, input). Cells
that do not pass today are ``xfail(strict=True)`` with the recorded outcome
(``status/matrix.json``), so every change of status shows up in CI. Order 3,
batched inputs and the non-default SCF modes at order 2 are ``slow``.
Regenerate the status with ``python -m test.test_baseline.status matrix``.
"""

from __future__ import annotations

import json

import pytest

from dxtb._src.exlibs.available import has_libcint

from .matrix import CELLS, Cell, run_cell
from .status import STATUS_DIR, run_isolated

_STATUS_FILE = STATUS_DIR / "matrix.json"
STATUS: dict[str, dict] = (
    json.loads(_STATUS_FILE.read_text()) if _STATUS_FILE.exists() else {}
)


def _is_slow(cell: Cell) -> bool:
    if cell.order >= 3 or cell.input != "single":
        return True
    if cell.order == 2 and cell.scf != "full":
        return True
    return STATUS.get(cell.id, {}).get("seconds", 0.0) > 30.0


def _param(cell: Cell):
    marks = [pytest.mark.scf]
    if _is_slow(cell):
        marks.append(pytest.mark.slow)
    if cell.input != "single":
        marks.append(pytest.mark.batch_mode)
    if cell.quantity in ("dipole", "polarizability", "dipole_deriv"):
        marks.append(pytest.mark.efield)
    if cell.quantity in ("hyperpolarizability", "pol_deriv"):
        marks.append(pytest.mark.efield)
    if cell.driver == "libcint":
        marks.append(
            pytest.mark.skipif(not has_libcint, reason="libcint not available")
        )

    recorded = STATUS.get(cell.id)
    if recorded is None:
        marks.append(pytest.mark.skip(reason="no recorded status"))
    elif recorded["status"] != "pass":
        reason = f"{recorded['status']}: {recorded['message'] or ''}".strip()
        if recorded.get("max_abs") is not None:
            reason += f" (max_abs={recorded['max_abs']:.2e})"
        marks.append(pytest.mark.xfail(strict=True, reason=reason))
    return pytest.param(cell, marks=marks, id=cell.id)


@pytest.mark.parametrize("cell", [_param(c) for c in CELLS])
def test_cell(cell: Cell) -> None:
    if STATUS.get(cell.id, {}).get("status", "pass") != "pass":
        # may crash or exhaust memory: run with limits in a fresh process
        res = run_isolated("matrix", cell.id)
        assert res["status"] == "pass", res["message"]
        return

    result = run_cell(cell)
    assert result.status == "pass", (
        f"{cell.id}: {result.status} {result.message} "
        f"(max_abs={result.max_abs}, max_rel={result.max_rel})"
    )
