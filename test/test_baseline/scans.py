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
Additional checks of the derivative status matrix (T0.3).

Run from the repository root::

    python -m test.test_baseline.scans

- Step-size scan (water, GFN1 and GFN2, PyTorch driver): finite
  differences of the energy (forces), the forces (Hessian) and the dipole
  (polarizability) for steps 1e-2 ... 1e-6 against autograd, to separate the
  truncation error from real disagreement.
- SCF convergence scan (water, unrolled SCF): the third-order cells
  (``third_order``, ``hyperpolarizability``, ``pol_deriv``, autograd) with
  convergence thresholds 1e-6 ... 1e-12 against the finite difference
  references, to see whether the unrolled derivatives are limited by the
  convergence.

Results: ``status/scans.json``.
"""

from __future__ import annotations

import json

import torch

from . import matrix as mx
from .matrix_refs import _central
from .status import STATUS_DIR

STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
THRESHOLDS = (1e-6, 1e-8, 1e-10, 1e-12)


def step_scan(method: str) -> dict[str, dict[str, float]]:
    numbers, positions, charge = mx.single_system("H2O")
    args = (method, "pytorch", "full", numbers)
    f0 = mx.zero_field(False)

    forces = mx.forces_ad(*args, positions, charge)
    pos = positions.detach().clone().requires_grad_(True)
    calc = mx.calculator(*args)
    hess = calc.hessian(pos, charge).detach()
    alpha = mx.polarizability_ad(*args, positions, charge).detach()

    out: dict[str, dict[str, float]] = {
        "forces": {},
        "hessian": {},
        "polarizability": {},
    }
    for h in STEPS:
        fd = -_central(lambda p: mx.energy(*args, p, charge), positions, h)
        out["forces"][f"{h:g}"] = float((fd - forces).abs().max())
        fd = -_central(lambda p: mx.forces_ad(*args, p, charge), positions, h)
        out["hessian"][f"{h:g}"] = float((fd - hess).abs().max())
        fd = _central(
            lambda f: mx.dipole_ad(*args, positions, charge, field=f), f0, h
        )
        out["polarizability"][f"{h:g}"] = float((fd - alpha).abs().max())
    return out


def convergence_scan(method: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    original = dict(mx.SCF_OPTIONS)
    try:
        for thr in THRESHOLDS:
            mx.SCF_OPTIONS.update(f_atol=thr, x_atol=thr)
            for q in ("third_order", "hyperpolarizability", "pol_deriv"):
                cell = mx.Cell(
                    q, "autograd", method, "pytorch", "full", "single"
                )
                res = mx.run_cell(cell)
                out.setdefault(q, {})[f"{thr:g}"] = (
                    res.max_abs if res.max_abs is not None else float("nan")
                )
    finally:
        mx.SCF_OPTIONS.clear()
        mx.SCF_OPTIONS.update(original)
    return out


def main() -> None:
    torch.set_num_threads(1)
    data = {}
    for method in mx.METHODS:
        data[f"{method}.step_scan"] = step_scan(method)
        data[f"{method}.convergence_scan"] = convergence_scan(method)
        print(method, json.dumps(data[f"{method}.step_scan"]), flush=True)
        print(
            method, json.dumps(data[f"{method}.convergence_scan"]), flush=True
        )
    STATUS_DIR.mkdir(exist_ok=True)
    (STATUS_DIR / "scans.json").write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
