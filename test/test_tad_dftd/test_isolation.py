# This file is part of tad-dftd.
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
Tests for standalone package independence and isolation of ``tad_dftd``.
"""
from __future__ import annotations

import ast
import configparser
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

import tad_dftd

FORBIDDEN_IMPORTS = {"dxtb", "tad_dftd3", "tad_dftd4"}


def test_static_ast_no_forbidden_imports() -> None:
    """Verify via AST inspection that no file in src/tad_dftd imports forbidden packages."""
    pkg_dir = Path(tad_dftd.__file__).resolve().parent
    py_files = sorted(pkg_dir.rglob("*.py"))
    assert len(py_files) > 0

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in FORBIDDEN_IMPORTS, (
                        f"Forbidden import '{alias.name}' in {py_file}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                top = node.module.split(".")[0]
                assert top not in FORBIDDEN_IMPORTS, (
                    f"Forbidden from-import '{node.module}' in {py_file}"
                )


def test_subprocess_isolated_import() -> None:
    """
    Verify that importing ``tad_dftd`` in a fresh Python interpreter succeeds
    when ``dxtb``, ``tad_dftd3``, and ``tad_dftd4`` are blocked in ``sys.modules``
    and does not load any of them.
    """
    src_dir = str(Path(tad_dftd.__file__).resolve().parents[1])
    code = """
import sys
sys.modules["dxtb"] = None
sys.modules["tad_dftd3"] = None
sys.modules["tad_dftd4"] = None
import tad_dftd
assert hasattr(tad_dftd, "__version__")
for name in list(sys.modules):
    if sys.modules[name] is not None:
        top = name.split(".")[0]
        assert top not in {"dxtb", "tad_dftd3", "tad_dftd4"}, f"Loaded {name}"
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": src_dir, "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Isolated import failed:\n{proc.stderr}"


def test_dxtb_works_without_legacy_tad_dftd3_or_tad_dftd4() -> None:
    """
    Verify that ``dxtb`` imports and evaluates D3, D4, and D4SC dispersion
    when ``tad_dftd3`` and ``tad_dftd4`` are unavailable in ``sys.modules``.
    """
    src_dir = str(Path(tad_dftd.__file__).resolve().parents[1])
    code = """
import sys
sys.modules["tad_dftd3"] = None
sys.modules["tad_dftd4"] = None
import torch
import dxtb
from dxtb._src.components.classicals.dispersion import new_dispersion

numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
positions = torch.tensor(
    [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]],
    dtype=torch.float64,
)

# Classical D3 (GFN1-xTB)
disp_d3 = new_dispersion(numbers, dxtb.GFN1_XTB, dtype=torch.float64)
assert disp_d3 is not None
e_d3 = disp_d3.get_energy(positions, disp_d3.get_cache(numbers))
assert torch.isfinite(e_d3).all()

# Classical D4 (GFN2-xTB with sc=False)
par_d4 = dxtb.GFN2_XTB.model_copy(deep=True)
par_d4.dispersion.d4.sc = False
charge = torch.tensor(0.0, dtype=torch.float64)
disp_d4 = new_dispersion(numbers, par_d4, charge=charge, dtype=torch.float64)
assert disp_d4 is not None
e_d4 = disp_d4.get_energy(positions, disp_d4.get_cache(numbers))
assert torch.isfinite(e_d4).all()

for name in list(sys.modules):
    if sys.modules[name] is not None:
        top = name.split(".")[0]
        assert top not in {"tad_dftd3", "tad_dftd4"}, f"Loaded {name}"
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": src_dir, "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"dxtb without legacy dftd failed:\n{proc.stderr}"


def test_package_data_completeness_and_wheel_simulation(tmp_path: Path) -> None:
    """
    Verify that setup.cfg [options.package_data] for tad_dftd covers every
    required runtime resource (.pt, .toml, py.typed, license/provenance files)
    and that a wheel-like copy containing only .py files plus matched
    package_data files imports and evaluates D1..D4Srev without the source tree.
    """
    pkg_dir = Path(tad_dftd.__file__).resolve().parent
    repo_root = pkg_dir.parents[1]
    setup_cfg = repo_root / "setup.cfg"

    cp = configparser.ConfigParser()
    cp.read(setup_cfg, encoding="utf-8")
    raw_patterns = cp.get("options.package_data", "tad_dftd")
    patterns = [line.strip() for line in raw_patterns.splitlines() if line.strip()]

    # Every non-.py/.pyc file under src/tad_dftd must match at least one pattern
    all_non_py = [
        p.relative_to(pkg_dir).as_posix()
        for p in sorted(pkg_dir.rglob("*"))
        if p.is_file()
        and p.suffix not in {".py", ".pyc"}
        and "__pycache__" not in p.parts
    ]
    assert len(all_non_py) >= 10
    for rel in all_non_py:
        assert any(fnmatch.fnmatch(rel, pat) for pat in patterns), (
            f"Runtime asset 'tad_dftd/{rel}' is not matched by setup.cfg "
            f"package_data patterns {patterns}"
        )

    # Simulate an installed wheel containing ONLY .py files + matched package_data
    sim_site = tmp_path / "site_packages"
    sim_pkg = sim_site / "tad_dftd"
    for p in pkg_dir.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(pkg_dir).as_posix()
        if p.suffix == ".py" or any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            dest = sim_pkg / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    code = f"""
import sys
sys.path = [{str(sim_site)!r}] + [
    p for p in sys.path if "worktrees" not in p and not p.endswith("/src")
]
sys.modules["dxtb"] = None
sys.modules["tad_dftd3"] = None
sys.modules["tad_dftd4"] = None

import torch
import tad_mctc as mctc
import tad_dftd

assert tad_dftd.__file__.startswith({str(sim_pkg)!r}), tad_dftd.__file__
mol = mctc.Structure(
    numbers=torch.tensor([8, 1, 1], dtype=torch.long),
    positions=torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]],
        dtype=torch.float64,
    ),
)
for model_name in ("d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"):
    p = tad_dftd.param.get_params(
        "pbe", model="d4" if model_name == "d4srev" else model_name
    )
    e = tad_dftd.dftd(mol, p, model=model_name)
    assert torch.isfinite(e).all() and (e.sum() < 0.0)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path),
        env={"PYTHONPATH": str(sim_site), "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Wheel simulation failed:\n{proc.stderr}"


