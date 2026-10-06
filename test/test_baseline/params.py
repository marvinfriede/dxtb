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
Parameter-gradient coverage (T0.9).

Run from the repository root::

    python -m test.test_baseline.params [--methods gfn1 gfn2]

For GFN1 and GFN2: build a :class:`~dxtb.ParamModule`, switch on gradients
of every floating point leaf, compute energies and forces (projected on a
fixed random direction) for a set of molecules covering s, p and d
elements, a halogen bond, a charged and an open-shell system, and record
for every leaf

- the autograd gradient (``None``, exactly zero or non-zero), and
- for leaves of the elements in the set and global leaves: a central finite
  difference along a random direction, which separates "unused" (both zero)
  from "gradient path cut" (autograd zero, finite difference not) and
  "wrong" (both non-zero, but different).

Results: ``status/param_coverage.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from tad_mctc.data import pse

from . import matrix as mx
from .molecules import get_system
from .status import STATUS_DIR

SYSTEMS = (
    "H2O",
    "CH4",
    "NH4+",
    "NO2",
    "Fe(CO)5",
    "PbH4-BiH3",
    "br2nh3",
)
"""Molecule set (``br2nh3``: halogen bond, from tad-mctc)."""


def _system(name: str):
    if name == "br2nh3":
        # pylint: disable=import-outside-toplevel
        from ..molecules import mols

        m = mols[name]
        return (
            m["numbers"].clone(),
            m["positions"].to(**mx.DD).clone(),
            torch.tensor(0.0, **mx.DD),
            None,
        )
    s = get_system(name)
    return s.numbers, s.positions, s.charge, s.spin


def _leaf_label(name: str) -> str:
    return name.replace("parameter_tree.", "").replace(".param", "")


def _relevant(name: str, symbols: set[str]) -> bool:
    if ".element." in name:
        return name.split(".element.")[1].split(".")[0] in symbols
    if ".kpair." in name:
        pair = name.split(".kpair.")[1].split(".")[0]
        return set(pair.split("-")) <= symbols
    return ".meta." not in name


def coverage(method: str) -> dict[str, dict]:
    par = mx.param_module(method)
    leaves = {n: p for n, p in par.named_parameters() if p.is_floating_point()}
    for p in leaves.values():
        p.requires_grad_(True)

    names = list(leaves)
    grad_e = {n: None for n in names}
    grad_f = {n: None for n in names}
    fd: dict[str, dict[str, float]] = {}
    symbols_all: set[str] = set()

    for sysname in SYSTEMS:
        numbers, positions, charge, spin = _system(sysname)
        symbols = {pse.Z2S[int(z)] for z in torch.unique(numbers) if z > 0}
        symbols_all |= symbols

        calc = mx.calculator(method, "pytorch", "full", numbers, par=par)
        pos = positions.detach().clone().requires_grad_(True)
        energy = calc.energy(pos, charge, spin)
        u = mx.force_direction(pos)
        (g,) = torch.autograd.grad(energy, pos, create_graph=True)
        fproj = (-g * u).sum()

        ge = torch.autograd.grad(
            energy, list(leaves.values()), retain_graph=True, allow_unused=True
        )
        gf = torch.autograd.grad(
            fproj, list(leaves.values()), allow_unused=True
        )
        for n, a, b in zip(names, ge, gf):
            for store, val in ((grad_e, a), (grad_f, b)):
                if val is None:
                    continue
                prev = store[n]
                store[n] = val.detach() if prev is None else prev + val.detach()

        # finite differences of the energy for the relevant leaves
        for n in names:
            if not _relevant(n, symbols):
                continue
            v = mx.param_direction(method, n)
            p0 = leaves[n].detach().clone()
            h = 1e-5 * max(1.0, float(p0.abs().max()))

            def e_of(p: torch.Tensor) -> float:
                with torch.no_grad():
                    return float(
                        mx.param_energy(
                            method,
                            "pytorch",
                            "full",
                            numbers,
                            positions,
                            charge,
                            n,
                            p,
                        )
                    )

            diff = (e_of(p0 + h * v) - e_of(p0 - h * v)) / (2 * h)
            ad = (
                0.0
                if ge[names.index(n)] is None
                else float((ge[names.index(n)] * v).sum())
            )
            fd.setdefault(n, {})[sysname] = {"fd": diff, "ad": ad}

    out: dict[str, dict] = {"__not_checked__": {"count": 0}}
    for n in names:
        ge_n, gf_n = grad_e[n], grad_f[n]
        rec = {
            "energy_grad": (
                "none"
                if ge_n is None
                else ("zero" if bool((ge_n == 0).all()) else "nonzero")
            ),
            "forces_grad": (
                "none"
                if gf_n is None
                else ("zero" if bool((gf_n == 0).all()) else "nonzero")
            ),
            "relevant": _relevant(n, symbols_all),
        }
        checks = fd.get(n, {})
        verdict = "not checked (element not in set)"
        if checks:
            cut = wrong = used = False
            for c in checks.values():
                scale = max(abs(c["fd"]), abs(c["ad"]), 1e-12)
                if abs(c["fd"]) > 1e-7 or abs(c["ad"]) > 1e-7:
                    used = True
                    if abs(c["ad"]) <= 1e-12:
                        cut = True
                    elif abs(c["fd"] - c["ad"]) > 1e-4 * scale + 1e-8:
                        wrong = True
            verdict = (
                "gradient path cut"
                if cut
                else (
                    "wrong gradient"
                    if wrong
                    else "ok" if used else "no effect on this set"
                )
            )
            rec["fd_checks"] = checks
        rec["verdict"] = verdict
        if not checks:
            # elements outside the set: only counted, to keep the file small
            out["__not_checked__"]["count"] += 1
            continue
        out[_leaf_label(n)] = rec
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="parameter coverage")
    parser.add_argument("--methods", nargs="*", default=["gfn1", "gfn2"])
    args = parser.parse_args()

    path = STATUS_DIR / "param_coverage.json"
    for method in args.methods:
        start = time.perf_counter()
        result = coverage(method)
        print(f"{method}: {time.perf_counter() - start:.0f} s", flush=True)
        counts: dict[str, int] = {
            "not checked (element not in set)": result["__not_checked__"][
                "count"
            ]
        }
        for key, rec in result.items():
            if key != "__not_checked__":
                counts[rec["verdict"]] = counts.get(rec["verdict"], 0) + 1
        print(counts)

        # read just before writing: methods may run in parallel processes
        data = json.loads(path.read_text()) if path.exists() else {}
        data[method] = result
        STATUS_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
