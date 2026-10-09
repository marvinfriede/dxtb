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
Repository rules of the restructuring (``docs/plan/00-overview.md``,
section 10).
"""

from __future__ import annotations

import re

from pathlib import Path

SRC = Path(__file__).parents[2] / "src" / "dxtb"


def test_no_object_setattr() -> None:
    """
    Working agreement 5: no ``object.__setattr__`` outside the ``Node`` base
    (which lives in tad-mctc). Frozen objects change through ``replace()``.
    """
    offenders = [
        f"{path.relative_to(SRC.parent)}:{lineno}"
        for path in SRC.rglob("*.py")
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if "object.__setattr__" in line
    ]
    assert not offenders, "object.__setattr__ found in " + ", ".join(offenders)


def test_no_calculator_result_cache_or_identity_keying() -> None:
    """Calculator evaluation cannot retain or key prior result outputs."""
    calculator_source = SRC / "_src" / "calculators"
    text = "\n".join(
        path.read_text() for path in calculator_source.rglob("*.py")
    )
    forbidden = (
        "class CalculatorCache",
        "class ConfigCache",
        "class ConfigCacheStore",
        "@cdec.cache",
        "set_cache_key",
        "get_cache_key",
        "tensor_id(",
        "data_ptr()",
        "hashed_key",
        "cache_key",
        "_last_result",
        "last_result",
        "result_cache",
        "cached_result",
        "self.cache",
    )
    found = [token for token in forbidden if token in text]
    assert not found, "Calculator result retention returned: " + ", ".join(
        found
    )
    identity_patterns = (
        r"\btensor_id\s*\(",
        r"\.data_ptr\s*\(",
        r"(?:untyped_)?storage\s*\(\)\.data_ptr",
        r"\bid\s*\(\s*(?:positions|tensor)\b",
    )
    identity_found = [
        pattern for pattern in identity_patterns if re.search(pattern, text)
    ]
    assert (
        not identity_found
    ), "Calculator result identity keying returned: " + ", ".join(
        identity_found
    )


def test_repulsion_uses_plain_torch_derivatives() -> None:
    """The repulsion package has one ordinary-PyTorch implementation."""
    repulsion_source = SRC / "_src" / "components" / "classicals" / "repulsion"
    text = "\n".join(
        path.read_text() for path in repulsion_source.rglob("*.py")
    )
    forbidden = (
        "RepulsionAG",
        "RepulsionAnalytical",
        "torch.autograd.Function",
        "with_analytical_gradient",
        "repulsion_gradient",
    )
    found = [token for token in forbidden if token in text]
    assert not found, "Removed repulsion path returned: " + ", ".join(found)


def test_es2_uses_plain_torch_coulomb_construction() -> None:
    """ES2 Coulomb matrices do not use a custom autograd shortcut."""
    source = (
        SRC
        / "_src"
        / "components"
        / "interactions"
        / "coulomb"
        / "secondorder.py"
    ).read_text()
    forbidden = (
        "CoulombMatrixAG",
        "torch.autograd.Function",
        "coulomb_matrix_atom_gradient",
        "coulomb_matrix_shell_gradient",
    )
    found = [token for token in forbidden if token in source]
    assert not found, "Removed ES2 custom path returned: " + ", ".join(found)


def test_es2_es3_do_not_retain_persistent_call_data() -> None:
    """ES2/ES3 cache methods construct call data without component reuse."""
    coulomb = SRC / "_src" / "components" / "interactions" / "coulomb"
    es2 = (coulomb / "secondorder.py").read_text()
    es3 = (coulomb / "thirdorder.py").read_text()
    migrated_sources = es2 + es3
    forbidden = (
        "cache_is_latest",
        "_cachevars",
        "_cachegrad",
        "self.cache =",
    )
    found = [token for token in forbidden if token in migrated_sources]
    assert not found, "Persistent ES2/ES3 cache path returned: " + ", ".join(
        found
    )

    interaction_list = (
        SRC / "_src" / "components" / "interactions" / "list.py"
    ).read_text()
    removed_apis = (
        "def update_es2(",
        "def reset_es2(",
        "def update_es3(",
        "def reset_es3(",
    )
    found = [token for token in removed_apis if token in interaction_list]
    assert not found, "ES2/ES3 mutation API returned: " + ", ".join(found)
