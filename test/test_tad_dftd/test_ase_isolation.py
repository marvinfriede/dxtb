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

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

_ISOLATION_SCRIPT = """
import importlib.abc
import json
import os
import sys
from pathlib import Path

expected_root = Path(os.environ["TAD_DFTD_EXPECTED_IMPORT_ROOT"]).resolve()
forbid_repo = os.environ.get("TAD_DFTD_FORBID_REPO_ROOT")
if forbid_repo:
    repo_root = Path(forbid_repo).resolve()
    sys.path = [
        p for p in sys.path
        if p and not Path(p).resolve().is_relative_to(repo_root)
    ]

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

# 2. Verify tad_dftd imports cleanly without ASE from the expected target root
import torch
import tad_dftd
from tad_dftd import dftd

mod_path = Path(tad_dftd.__file__).resolve()
assert mod_path.is_relative_to(expected_root), (
    f"Expected tad_dftd.__file__ ({mod_path}) to be under {expected_root}"
)
if forbid_repo:
    assert not mod_path.is_relative_to(Path(forbid_repo).resolve()), (
        f"tad_dftd.__file__ ({mod_path}) leaked from repository root {forbid_repo}"
    )

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

print(json.dumps({"tad_dftd_file": str(mod_path), "energy_eh": float(total_e.item())}))
"""

_INSTALLED_WITH_ASE_SCRIPT = """
import json
import os
import sys
from pathlib import Path

expected_root = Path(os.environ["TAD_DFTD_EXPECTED_IMPORT_ROOT"]).resolve()
repo_root = Path(os.environ["TAD_DFTD_FORBID_REPO_ROOT"]).resolve()
sys.path = [
    p for p in sys.path
    if p and not Path(p).resolve().is_relative_to(repo_root)
]

import ase
from ase.build import molecule
import tad_dftd
import tad_dftd.ase
from tad_dftd.ase import DFTD

pkg_path = Path(tad_dftd.__file__).resolve()
ase_mod_path = Path(tad_dftd.ase.__file__).resolve()
assert pkg_path.is_relative_to(expected_root), (
    f"Expected tad_dftd.__file__ ({pkg_path}) under {expected_root}"
)
assert ase_mod_path.is_relative_to(expected_root), (
    f"Expected tad_dftd.ase.__file__ ({ase_mod_path}) under {expected_root}"
)
assert not pkg_path.is_relative_to(repo_root)
assert not ase_mod_path.is_relative_to(repo_root)

atoms = molecule("H2O")
atoms.calc = DFTD(method="PBE", model="d4")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
assert isinstance(energy, float) and energy < 0.0
assert forces.shape == (3, 3)

print(
    json.dumps(
        {
            "tad_dftd_file": str(pkg_path),
            "tad_dftd_ase_file": str(ase_mod_path),
            "ase_file": str(Path(ase.__file__).resolve()),
            "energy_ev": energy,
        }
    )
)
"""


def run_ase_isolation_check(
    *,
    import_root: Path = SRC_DIR,
    forbid_repo_root: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Run the deterministic ASE import-blocked subprocess check."""
    env = dict(os.environ)
    raw_pp = env.get("PYTHONPATH", "")
    pp_entries = [p for p in raw_pp.split(os.pathsep) if p]
    if forbid_repo_root is not None:
        resolved_forbid = forbid_repo_root.resolve()
        pp_entries = [
            p
            for p in pp_entries
            if not Path(p).resolve().is_relative_to(resolved_forbid)
        ]
        env["TAD_DFTD_FORBID_REPO_ROOT"] = str(resolved_forbid)
    else:
        env.pop("TAD_DFTD_FORBID_REPO_ROOT", None)

    env["TAD_DFTD_EXPECTED_IMPORT_ROOT"] = str(import_root.resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        [str(import_root.resolve()), *pp_entries]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _ISOLATION_SCRIPT],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        proc.returncode == 0
    ), f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    last_line = [line for line in proc.stdout.strip().splitlines() if line][-1]
    return json.loads(last_line)


def run_installed_wheel_ase_verification(
    work_dir: Path | None = None,
) -> dict[str, str]:
    """
    Build a wheel, install it into an isolated target directory without the
    repository ``src`` directory on ``PYTHONPATH`` or ``sys.path``, and verify
    both ASE-blocked native dispersion execution and ASE-enabled calculator
    execution from the installed wheel.
    """
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp_str:
        tmp_root = Path(tmp_str).resolve()
        wheelhouse = tmp_root / "wheelhouse"
        install_target = tmp_root / "installed_site"
        run_cwd = tmp_root / "run_cwd"
        wheelhouse.mkdir(parents=True, exist_ok=True)
        install_target.mkdir(parents=True, exist_ok=True)
        run_cwd.mkdir(parents=True, exist_ok=True)

        # 1. Build the wheel
        build_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                str(REPO_ROOT),
                "-w",
                str(wheelhouse),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            build_proc.returncode == 0
        ), f"Wheel build failed:\n{build_proc.stdout}\n{build_proc.stderr}"
        wheels = sorted(wheelhouse.glob("*.whl"))
        assert (
            len(wheels) == 1
        ), f"Expected 1 wheel in {wheelhouse}, got {wheels}"
        wheel_path = wheels[0].resolve()

        # 2. Install into isolated target directory
        install_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(install_target),
                str(wheel_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            install_proc.returncode == 0
        ), f"Wheel install failed:\n{install_proc.stdout}\n{install_proc.stderr}"

        # 3-7. Verify installed wheel with ASE blocked and no source-tree contamination
        blocked_res = run_ase_isolation_check(
            import_root=install_target,
            forbid_repo_root=REPO_ROOT,
            cwd=run_cwd,
        )

        # 8. Verify installed wheel with ASE available
        env = dict(os.environ)
        raw_pp = env.get("PYTHONPATH", "")
        resolved_repo = REPO_ROOT.resolve()
        pp_entries = [
            p
            for p in raw_pp.split(os.pathsep)
            if p and not Path(p).resolve().is_relative_to(resolved_repo)
        ]
        env["PYTHONPATH"] = os.pathsep.join([str(install_target), *pp_entries])
        env["TAD_DFTD_EXPECTED_IMPORT_ROOT"] = str(install_target)
        env["TAD_DFTD_FORBID_REPO_ROOT"] = str(resolved_repo)

        ase_proc = subprocess.run(
            [sys.executable, "-c", _INSTALLED_WITH_ASE_SCRIPT],
            cwd=str(run_cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            ase_proc.returncode == 0
        ), f"STDOUT:\n{ase_proc.stdout}\nSTDERR:\n{ase_proc.stderr}"
        ase_res = json.loads(
            [line for line in ase_proc.stdout.strip().splitlines() if line][-1]
        )

        return {
            "wheel_path": str(wheel_path),
            "install_target": str(install_target),
            "tad_dftd_file_ase_blocked": str(blocked_res["tad_dftd_file"]),
            "tad_dftd_file_with_ase": str(ase_res["tad_dftd_file"]),
            "tad_dftd_ase_file": str(ase_res["tad_dftd_ase_file"]),
            "ase_file": str(ase_res["ase_file"]),
        }


def test_tad_dftd_without_ase_installed() -> None:
    """Verify tad_dftd imports and evaluates without ASE while tad_dftd.ase raises ModuleNotFoundError."""
    run_ase_isolation_check()


@pytest.mark.skipif(
    os.environ.get("TAD_DFTD_VERIFY_WHEEL") != "1",
    reason="Release wheel verification test; set TAD_DFTD_VERIFY_WHEEL=1 or run test_ase_isolation.py as a script",
)
def test_installed_wheel_ase_isolation() -> None:
    """Verify installed-wheel isolation without source-tree contamination."""
    report = run_installed_wheel_ase_verification()
    assert report["tad_dftd_file_ase_blocked"].startswith(
        report["install_target"]
    )
    assert report["tad_dftd_ase_file"].startswith(report["install_target"])


if __name__ == "__main__":
    print(json.dumps(run_installed_wheel_ase_verification(), indent=2))
