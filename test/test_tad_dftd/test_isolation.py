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
