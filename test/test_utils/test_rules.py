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
