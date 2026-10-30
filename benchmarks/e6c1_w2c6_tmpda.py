# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Run W2c2's bounded tmpda diagnostic with the W2c6 experimental kernel.

The module-level substitution is scoped to this private diagnostic process and
restored in ``finally``. No production source or persistent state is changed.
Forward all W2c2 arguments after the script name, for example::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmarks/e6c1_w2c6_tmpda.py \
        --stage energy --observable energy --conformer-index 0
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

import dxtb
from benchmarks import e6c1_w2c2_implicit as w2c2
from benchmarks.e6c1_w2c6_tanh_solve import _matrix_fermi_direct_rhs
from dxtb._src.scf.functional import spectral as spectral_module

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the established W2c2 benchmark under the direct-RHS candidate."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be explicitly set to 1")

    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    print(
        json.dumps(
            {
                "event": "w2c6_candidate_environment",
                "source_revision": source_revision,
                "source_dirty": source_dirty,
                "dxtb_source": str(Path(dxtb.__file__).resolve()),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cpu": platform.processor() or platform.machine(),
                "threads": {
                    key: os.environ[key]
                    for key in (
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
                "kernel": "direct-rhs-tanh-doubling",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    original_matrix_fermi = spectral_module._matrix_fermi
    spectral_module._matrix_fermi = _matrix_fermi_direct_rhs
    try:
        w2c2.main()
    finally:
        spectral_module._matrix_fermi = original_matrix_fermi


if __name__ == "__main__":
    main()
