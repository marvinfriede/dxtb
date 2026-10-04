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
Record the derivative status matrix (T0.3) and the component checks (T0.4).

Run from the repository root after ``matrix_refs``::

    python -m test.test_baseline.status matrix [--jobs 4] [--filter hessian]
    python -m test.test_baseline.status components [--jobs 4]

Every cell runs in a fresh process with a memory limit and a timeout (see
``run_isolated``), which also gives the peak memory per cell. The results go
to ``status/matrix.json`` and ``status/components.json``; the tests in
``test_derivatives.py`` and ``test_components.py`` read them to mark the
cells that do not pass today as ``xfail(strict=True)``. A filtered run
updates only the selected cells.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STATUS_DIR = Path(__file__).parent / "status"
ROOT = Path(__file__).parents[2]

TIMEOUT = 1800
"""Seconds before a cell is stopped and recorded as an error."""

MEMORY_GB = float(os.environ.get("DXTB_BASELINE_MEMORY_GB", "12"))
"""
Address space limit of a cell process (GB). Note that the CUDA builds of
torch reserve several GB of address space without using it.
"""


COMPONENT_MEMORY_GB = 4.0
"""
Address space limit for the component checks, which include an unbounded
recursion (``repulsion_ag.arep``) that should fail quickly.
"""


def _memory_limit(gb: float):
    def limit() -> None:  # pragma: no cover (runs in the child)
        nbytes = int(gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))

    return limit


def run_isolated(kind: str, key: str, timeout: float = TIMEOUT) -> dict:
    """
    Run one cell (``kind`` is ``matrix`` or ``components``) in a fresh
    process with a memory limit and a timeout. A crash (e.g., unbounded
    recursion running out of memory) or a timeout is an ``error``.
    """
    env = dict(os.environ, OMP_NUM_THREADS="1")
    gb = MEMORY_GB if kind == "matrix" else COMPONENT_MEMORY_GB
    cmd = [sys.executable, "-m", "test.test_baseline.status", "one", kind, key]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_memory_limit(gb),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "seconds": time.perf_counter() - start,
            "message": f"timeout after {timeout:.0f} s",
        }

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)

    err = proc.stderr.strip().splitlines()
    tail = " | ".join(err[-2:])[:300] if err else ""
    return {
        "status": "error",
        "seconds": time.perf_counter() - start,
        "message": (
            f"process died (exit code {proc.returncode}, memory limit "
            f"{gb:g} GB): {tail}"
        ),
    }


def _one(kind: str, key: str) -> None:  # pragma: no cover (child)
    # pylint: disable=import-outside-toplevel
    import torch

    torch.set_num_threads(1)
    if kind == "matrix":
        from .matrix import CELLS, run_cell

        cell = next(c for c in CELLS if c.id == key)
        result = run_cell(cell).as_dict()
    else:
        from .components import run_check

        target, check = key.split(":")
        result = run_check(target, check).as_dict()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result["peak_rss_mb"] = round(peak, 1)
    print(json.dumps(result))


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(data.items())), indent=1) + "\n")


def _run(kind: str, keys: list[str], jobs: int, path: Path) -> None:
    data = _load(path)
    start = time.perf_counter()
    with ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(run_isolated, kind, k): k for k in keys}
        for i, fut in enumerate(as_completed(futures), start=1):
            key, result = futures[fut], fut.result()
            data[key] = result
            print(
                f"[{i}/{len(keys)}] {key}: {result['status']} "
                f"({result['seconds']:.1f} s) {result['message'][:120]}",
                flush=True,
            )
            if i % 10 == 0:
                _save(path, data)
    _save(path, data)
    print(f"done in {time.perf_counter() - start:.0f} s -> {path}")


def keys_of(kind: str) -> list[str]:
    """All cell keys of a kind."""
    # pylint: disable=import-outside-toplevel
    if kind == "matrix":
        from .matrix import CELLS

        return [c.id for c in CELLS]

    from .components import CHECKS, TARGETS

    return [f"{t}:{c}" for t in TARGETS for c in CHECKS]


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        _one(sys.argv[2], sys.argv[3])
        return

    parser = argparse.ArgumentParser(description="record status")
    parser.add_argument("what", choices=["matrix", "components"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--filter", nargs="*", default=[])
    args = parser.parse_args()

    keys = keys_of(args.what)
    if args.filter:
        keys = [k for k in keys if all(f in k for f in args.filter)]
    _run(args.what, keys, args.jobs, STATUS_DIR / f"{args.what}.json")


if __name__ == "__main__":
    main()
