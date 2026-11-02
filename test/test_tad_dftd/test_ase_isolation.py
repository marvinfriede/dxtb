# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
"""
Deterministic optional-dependency isolation tests for ``tad_dftd`` without ASE.

This module intentionally does not import ``ase`` during test collection so it
always executes even in environments where ASE is not installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

_ISOLATION_SCRIPT = """
import importlib.abc
import sys

class _BlockASEFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "ase" or fullname.startswith("ase."):
            raise ModuleNotFoundError("No module named 'ase'", name="ase")
        return None

for _mod in list(sys.modules):
    if _mod == "ase" or _mod.startswith("ase."):
        del sys.modules[_mod]
sys.meta_path.insert(0, _BlockASEFinder())

# 1. Confirm ASE and its submodules are deterministically inaccessible
for _blocked in ("ase", "ase.atoms", "ase.calculators.calculator", "ase.units"):
    try:
        __import__(_blocked)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"Expected {_blocked!r} to be blocked by _BlockASEFinder")

# 2. Verify tad_dftd imports cleanly without ASE
import torch
import tad_dftd
from tad_dftd import dftd

assert "ase" not in sys.modules
assert "tad_dftd.ase" not in sys.modules

# 3. Verify a native numerical dispersion energy + force calculation succeeds
from tad_mctc import Structure

pos = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
    dtype=torch.float64,
    requires_grad=True,
)
mol = Structure(numbers=torch.tensor([8, 1, 1], dtype=torch.long), positions=pos)
params = tad_dftd.param.get_params("PBE", model="d4")
atom_e = dftd(mol, params, model="d4")
total_e = atom_e.sum()
(grad,) = torch.autograd.grad(total_e, pos)

assert atom_e.shape == (3,)
assert torch.isfinite(total_e) and float(total_e.item()) < 0.0
assert grad.shape == (3, 3) and torch.isfinite(grad).all()
assert "ase" not in sys.modules
assert "tad_dftd.ase" not in sys.modules

# 4. Verify importing tad_dftd.ase raises a helpful ModuleNotFoundError
try:
    import tad_dftd.ase
except ModuleNotFoundError as exc:
    assert "requires ASE" in str(exc), f"Unexpected message: {exc}"
else:
    raise AssertionError("Expected ModuleNotFoundError when importing tad_dftd.ase without ASE")
"""


def run_ase_isolation_check() -> None:
    """Run the deterministic ASE import-blocked subprocess check."""
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_DIR)
    )
    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATION_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        proc.returncode == 0
    ), f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_tad_dftd_without_ase_installed() -> None:
    """Verify tad_dftd imports and evaluates without ASE while tad_dftd.ase raises ModuleNotFoundError."""
    run_ase_isolation_check()
