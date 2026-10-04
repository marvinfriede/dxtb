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
Finite difference references of the derivative status matrix (T0.3).

Run from the repository root (slow; ~1 h)::

    python -m test.test_baseline.matrix_refs [--methods gfn1] [--drivers pytorch]

Writes ``reference/matrix/<method>-<driver>.npz`` with keys
``<system>.<quantity>`` (and ``<system>.<quantity>.<leaf>`` for parameter
derivatives). Every finite difference is a central difference with the
steps in ``matrix.STEPS``, taken of the quantity one order lower computed
by autograd through the unrolled SCF (tight convergence, ``scf="full"``).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import matrix as mx

SCF = "full"


def _central(fn, x: torch.Tensor, h: float) -> torch.Tensor:
    """Derivative of ``fn`` w.r.t. every element of ``x``: shape out + x."""
    cols = []
    flat = x.detach().clone().reshape(-1)
    for i in range(flat.numel()):
        xp, xm = flat.clone(), flat.clone()
        xp[i] += h
        xm[i] -= h
        fp = fn(xp.reshape(x.shape)).detach()
        fm = fn(xm.reshape(x.shape)).detach()
        cols.append((fp - fm) / (2 * h))
    out = torch.stack(cols, dim=-1)
    return out.reshape(*out.shape[:-1], *x.shape)


def references(method: str, driver: str, name: str) -> dict[str, np.ndarray]:
    numbers, positions, charge = mx.single_system(name)
    args = (method, driver, SCF, numbers)
    hp, hf, hq = mx.STEPS["positions"], mx.STEPS["field"], mx.STEPS["param"]
    f0 = mx.zero_field(False)
    out: dict[str, torch.Tensor] = {}

    # order 1
    out["forces"] = -_central(
        lambda p: mx.energy(*args, p, charge), positions, hp
    )
    out["dipole"] = -_central(
        lambda f: mx.energy_in_field(*args, positions, charge, f), f0, hf
    )

    # order 2
    out["hessian"] = -_central(
        lambda p: mx.forces_ad(*args, p, charge), positions, hp
    )
    out["polarizability"] = _central(
        lambda f: mx.dipole_ad(*args, positions, charge, field=f), f0, hf
    )
    out["dipole_deriv"] = _central(
        lambda p: mx.dipole_ad(*args, p, charge), positions, hp
    )

    # order 3
    out["hyperpolarizability"] = _central(
        lambda f: mx.polarizability_ad(*args, positions, charge, field=f),
        f0,
        hf,
    )
    out["pol_deriv"] = _central(
        lambda p: mx.polarizability_ad(*args, p, charge), positions, hp
    )

    if name == "H2O":
        v, w = mx.third_order_directions(positions)
        out["third_order"] = _central(
            lambda p: mx.hvp_scalar(*args, p, charge, v, w), positions, hp
        )

        # parameters: directional derivatives per leaf
        u = mx.force_direction(positions)
        params = dict(mx.param_module(method).named_parameters())
        for leaf in mx.param_leaves(method, numbers):
            p0 = params[leaf].detach().clone()
            d = mx.param_direction(method, leaf)
            h = hq * max(1.0, float(p0.abs().max()))

            def energy(p: torch.Tensor) -> torch.Tensor:
                return mx.param_energy(*args, positions, charge, leaf, p)

            def forces(p: torch.Tensor) -> torch.Tensor:
                fr = mx.param_forces(*args, positions, charge, leaf, p)
                return (fr * u).sum()

            for key, fn in (("dE_dparam", energy), ("dforces_dparam", forces)):
                diff = (fn(p0 + h * d) - fn(p0 - h * d)) / (2 * h)
                out[f"{key}.{leaf}"] = diff.detach()

    return {f"{name}.{k}": v.cpu().numpy() for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="matrix references")
    parser.add_argument("--methods", nargs="*", default=list(mx.METHODS))
    parser.add_argument("--drivers", nargs="*", default=list(mx.DRIVERS))
    args = parser.parse_args()

    names = sorted({n for v in mx.INPUT_SYSTEMS.values() for n in v})
    for method in args.methods:
        for driver in args.drivers:
            start = time.perf_counter()
            arrays: dict[str, np.ndarray] = {}
            for name in names:
                arrays.update(references(method, driver, name))
            meta = {
                "method": method,
                "driver": driver,
                "scf": SCF,
                "scf_options": mx.SCF_OPTIONS,
                "steps": mx.STEPS,
                "systems": names,
                "seconds": round(time.perf_counter() - start, 1),
            }
            arrays["__meta__"] = np.array(json.dumps(meta, sort_keys=True))
            path = mx.reference_file(method, driver)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            print(f"{path} ({meta['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
