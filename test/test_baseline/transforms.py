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
Transform status (T0.7). Informational only; nothing is asserted.

Run from the repository root::

    python -m test.test_baseline.transforms

Checks how far ``torch.func`` and ``torch.compile`` get on the current
Calculator API (GFN1 and GFN2, PyTorch and libcint drivers, water):

- ``vmap`` of the energy and of the forces over positions: a single system
  (vmap over two geometries) and the conformer batch (``batch_mode=2``,
  vmap over the batch of geometries is not applicable, so ``vmap`` over a
  stack of two conformer batches);
- ``jacrev``, ``jacfwd`` and ``hessian`` of the energy;
- ``torch.compile`` of an energy call (``fullgraph=False``): number of graph
  breaks and their causes (``torch._dynamo.explain``), once including the
  calculator setup and once for the per-call path only.

All runs turn vmap fallback warnings into errors, so silent per-sample loops
show up. Results: ``status/transforms.json``.
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from collections import Counter
from typing import Any, Callable

import torch

from .matrix import DD, calculator, single_system
from .status import STATUS_DIR

METHODS = ("gfn1", "gfn2")
DRIVERS = ("pytorch", "libcint")


def _guarded(fn: Callable[[], Any]) -> dict[str, Any]:
    # pylint: disable=protected-access
    torch._C._functorch._set_vmap_fallback_warning_enabled(True)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*fallback.*")
            fn()
        status, msg = "works", ""
    except Exception as e:  # pylint: disable=broad-except
        tb = traceback.extract_tb(e.__traceback__)
        where = ""
        for frame in reversed(tb):
            if "/dxtb/_src/" in frame.filename or "/tad_" in frame.filename:
                where = frame.filename.split("site-packages/")[-1]
                where = where.split("/src/")[-1] + f":{frame.lineno}"
                break
        first = str(e).strip().splitlines()[0][:250] if str(e).strip() else ""
        fallback = "fallback" in str(e).lower()
        status = "falls back" if fallback else "error"
        msg = f"{type(e).__name__}: {first}" + (f" [{where}]" if where else "")
    finally:
        torch._C._functorch._set_vmap_fallback_warning_enabled(False)
    return {
        "status": status,
        "seconds": time.perf_counter() - start,
        "message": msg,
    }


def _checks(method: str, driver: str) -> dict[str, dict[str, Any]]:
    # pylint: disable=import-outside-toplevel
    from torch.func import hessian, jacfwd, jacrev, vmap

    numbers, positions, charge = single_system("H2O")
    _, pos2, _ = single_system("H2O*")
    stack = torch.stack([positions, pos2])

    def e(pos: torch.Tensor) -> torch.Tensor:
        return calculator(method, driver, "full", numbers).energy(pos, charge)

    def f(pos: torch.Tensor) -> torch.Tensor:
        return -jacrev(e)(pos)

    bnumbers = torch.stack([numbers, numbers])
    bcharge = torch.zeros(2, **DD)

    def e_batch(pos: torch.Tensor) -> torch.Tensor:
        calc = calculator(method, driver, "full", bnumbers, batch_mode=2)
        return calc.energy(pos, bcharge)

    out = {
        "vmap(energy) single": _guarded(lambda: vmap(e)(stack)),
        "vmap(forces) single": _guarded(lambda: vmap(f)(stack)),
        "vmap(energy) conformer batch": _guarded(
            lambda: vmap(e_batch)(torch.stack([stack, stack]))
        ),
        "jacrev(energy)": _guarded(lambda: jacrev(e)(positions)),
        "jacfwd(energy)": _guarded(lambda: jacfwd(e)(positions)),
        "hessian(energy)": _guarded(lambda: hessian(e)(positions)),
    }
    out["torch.compile(energy)"] = _compile(e, positions)

    # per-call path only: calculator (setup) built outside the compiled code
    calc = calculator(method, driver, "full", numbers)
    out["torch.compile(calc.energy)"] = _compile(
        lambda pos: calc.energy(pos, charge), positions
    )
    return out


def _compile(fn: Callable, x: torch.Tensor) -> dict[str, Any]:
    # pylint: disable=import-outside-toplevel
    import torch._dynamo as dynamo

    dynamo.reset()
    start = time.perf_counter()
    try:
        explanation = dynamo.explain(fn)(x)
    except Exception as e:  # pylint: disable=broad-except
        return {
            "status": "error",
            "seconds": time.perf_counter() - start,
            "message": f"{type(e).__name__}: {str(e).splitlines()[0][:250]}",
        }

    reasons = Counter()
    for br in explanation.break_reasons:
        reason = str(getattr(br, "reason", br)).splitlines()[0][:120]
        frame = ""
        if getattr(br, "user_stack", None):
            top = br.user_stack[-1]
            frame = f"{top.filename.split('/src/')[-1]}:{top.lineno}"
        reasons[f"{reason} @ {frame}"] += 1

    return {
        "status": "works" if explanation.graph_break_count == 0 else "breaks",
        "seconds": time.perf_counter() - start,
        "graphs": explanation.graph_count,
        "graph_breaks": explanation.graph_break_count,
        "ops": explanation.op_count,
        "top_break_reasons": reasons.most_common(15),
        "message": f"{explanation.graph_break_count} graph breaks",
    }


def main() -> None:
    results = {}
    for method in METHODS:
        for driver in DRIVERS:
            key = f"{method}-{driver}"
            results[key] = _checks(method, driver)
            for name, r in results[key].items():
                print(
                    f"{key:14s} {name:30s} {r['status']:10s} {r['message'][:150]}"
                )
    STATUS_DIR.mkdir(exist_ok=True)
    (STATUS_DIR / "transforms.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
