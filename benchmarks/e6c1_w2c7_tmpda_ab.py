# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare the pre-W2c6 and production Fermi recurrences on tmpda.

This private benchmark forwards the W2c2 physical-root diagnostic arguments.
The ``identity-rhs`` mode scopes the original pre-W2c6 recurrence to this
process and restores production state even if the diagnostic fails.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch

import dxtb
from benchmarks import e6c1_w2c2_implicit as w2c2
from benchmarks.e6c1_w2c6_tanh_solve import _matrix_fermi_identity_rhs
from dxtb._src.scf.functional import spectral


def main() -> None:
    """Run one isolated W2c2 diagnostic with the selected recurrence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel", choices=("production", "identity-rhs"), required=True
    )
    args, forwarded = parser.parse_known_args()

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be explicitly set to 1")

    threads = 1
    for index, value in enumerate(forwarded):
        if value == "--threads" and index + 1 < len(forwarded):
            threads = int(forwarded[index + 1])
    if threads != 1:
        raise RuntimeError("W2c7 A/B measurements require one thread")
    expected_source = (ROOT / "src/dxtb/__init__.py").resolve()
    if Path(dxtb.__file__).resolve() != expected_source:
        raise RuntimeError(
            "dxtb import does not resolve to the W2c7 source worktree"
        )

    print(
        json.dumps(
            {
                "event": "w2c7_kernel_run",
                "kernel": args.kernel,
                "source_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "source_dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"],
                        cwd=ROOT,
                        text=True,
                    ).strip()
                ),
                "dxtb_source": str(Path(dxtb.__file__).resolve()),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_intraop_threads": threads,
                "torch_interop_threads": 1,
                "thread_environment": {
                    name: os.environ[name]
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                    )
                },
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "tad-mctc",
                        "tad-multicharge",
                        "tad-libcint",
                        "tad-dftd3",
                        "tad-dftd4",
                    )
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    sys.argv = [sys.argv[0], *forwarded]
    if args.kernel == "production":
        w2c2.main()
        return

    original = spectral._matrix_fermi
    spectral._matrix_fermi = _matrix_fermi_identity_rhs
    try:
        w2c2.main()
    finally:
        spectral._matrix_fermi = original


if __name__ == "__main__":
    main()
