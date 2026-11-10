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
Dependency-free unit tests for the ``test.tblite`` test-reference adapter.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ..tblite import Structure, _coerce_structure, _resolve_uhf, tblite_singlepoint
from .molecules import SYSTEMS


def test_tblite_spin_uhf_alias_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify spin/uhf alias resolution across all structure forms without requiring tblite."""
    no2 = SYSTEMS["NO2"]
    nums = no2.numbers
    pos = no2.positions

    # 1. Direct _resolve_uhf and Structure construction.
    assert _resolve_uhf(None, 1) == 1
    assert _resolve_uhf(1, None) == 1
    assert _resolve_uhf(1, 1) == 1
    assert _resolve_uhf(None, None) is None
    with pytest.raises(ValueError, match="Conflicting"):
        _resolve_uhf(1, 2)
    with pytest.raises(
        ValueError, match="integer number of unpaired electrons"
    ):
        _resolve_uhf(None, 0.5)
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_uhf(None, -1)

    assert Structure(nums, pos, uhf=None, spin=1).uhf == 1
    assert Structure(nums, pos, uhf=1, spin=None).uhf == 1
    assert Structure(nums, pos, uhf=1, spin=1).uhf == 1
    assert Structure(nums, pos).uhf is None
    assert Structure(nums, pos, uhf=None, spin=None).uhf is None
    with pytest.raises(ValueError, match="Conflicting"):
        Structure(nums, pos, uhf=1, spin=2)
    with pytest.raises(
        ValueError, match="integer number of unpaired electrons"
    ):
        Structure(nums, pos, spin=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        Structure(nums, pos, spin=-2)

    # 2. Structure.replace.
    base = Structure(nums, pos, charge=0.0, uhf=2)
    assert base.replace(uhf=None, spin=1).uhf == 1
    assert base.replace(uhf=1, spin=None).uhf == 1
    assert base.replace(uhf=1, spin=1).uhf == 1
    assert base.replace(charge=-1.0).uhf == 2
    assert Structure(nums, pos).replace(charge=-1.0).uhf is None
    assert base.replace(uhf=None, spin=None).uhf is None
    assert base.replace(uhf=None).uhf is None
    assert base.replace(spin=None).uhf is None
    with pytest.raises(ValueError, match="Conflicting"):
        base.replace(uhf=1, spin=2)
    with pytest.raises(
        ValueError, match="integer number of unpaired electrons"
    ):
        base.replace(uhf=None, spin=0.25)
    with pytest.raises(ValueError, match="non-negative"):
        base.replace(uhf=-1, spin=None)

    # 3. Generic supported structure objects.
    obj_none_spin1 = SimpleNamespace(
        numbers=nums, positions=pos, uhf=None, spin=1
    )
    obj_uhf1_none = SimpleNamespace(
        numbers=nums, positions=pos, uhf=1, spin=None
    )
    obj_both1 = SimpleNamespace(numbers=nums, positions=pos, uhf=1, spin=1)
    obj_default = SimpleNamespace(numbers=nums, positions=pos)
    obj_conflict = SimpleNamespace(numbers=nums, positions=pos, uhf=1, spin=2)
    obj_frac = SimpleNamespace(numbers=nums, positions=pos, uhf=None, spin=0.5)
    obj_neg = SimpleNamespace(numbers=nums, positions=pos, uhf=-1, spin=None)

    assert _coerce_structure(obj_none_spin1).uhf == 1
    assert _coerce_structure(obj_uhf1_none).uhf == 1
    assert _coerce_structure(obj_both1).uhf == 1
    assert _coerce_structure(obj_default).uhf is None
    assert _coerce_structure(obj_none_spin1, uhf=None, spin=None).uhf == 1
    with pytest.raises(ValueError, match="Conflicting"):
        _coerce_structure(obj_conflict)
    with pytest.raises(
        ValueError, match="integer number of unpaired electrons"
    ):
        _coerce_structure(obj_frac)
    with pytest.raises(ValueError, match="non-negative"):
        _coerce_structure(obj_neg)

    # 4. Mapping input to tblite_singlepoint (stubbed so native tblite is not required).
    recorded_uhf: list[int | None] = []

    class _StubResult:
        def get(self, key: str):
            nat = int(nums.shape[0])
            if key == "energy":
                return -10.0
            if key == "energies":
                return np.zeros(nat, dtype=np.float64)
            if key == "gradient":
                return np.zeros((nat, 3), dtype=np.float64)
            if key == "charges":
                return np.zeros(nat, dtype=np.float64)
            if key == "dipole":
                return np.zeros(3, dtype=np.float64)
            if key == "quadrupole":
                return np.zeros(6, dtype=np.float64)
            if key in ("orbital-energies", "orbital-occupations"):
                return np.zeros(4, dtype=np.float64)
            if key == "density-matrix":
                return np.zeros((4, 4), dtype=np.float64)
            raise KeyError(key)

    class _StubCalculator:
        def __init__(
            self,
            method,
            numbers,
            positions,
            charge=None,
            uhf=None,
            lattice=None,
            periodic=None,
        ):
            recorded_uhf.append(uhf)

        def set(self, key: str, value) -> None:
            pass

        def add(self, key: str, *args) -> None:
            pass

        def singlepoint(self) -> _StubResult:
            return _StubResult()

    stub_tblite = ModuleType("tblite")
    stub_interface = ModuleType("tblite.interface")
    stub_library = ModuleType("tblite.library")
    stub_interface.Calculator = _StubCalculator  # type: ignore[attr-defined]
    stub_library.get_version = lambda: (0, 8, 0)  # type: ignore[attr-defined]
    stub_tblite.interface = stub_interface  # type: ignore[attr-defined]
    stub_tblite.library = stub_library  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tblite", stub_tblite)
    monkeypatch.setitem(sys.modules, "tblite.interface", stub_interface)
    monkeypatch.setitem(sys.modules, "tblite.library", stub_library)

    tblite_singlepoint(
        {"numbers": nums, "positions": pos, "uhf": None, "spin": 1}
    )
    tblite_singlepoint(
        {"numbers": nums, "positions": pos, "uhf": 1, "spin": None}
    )
    tblite_singlepoint({"numbers": nums, "positions": pos, "uhf": 1, "spin": 1})
    tblite_singlepoint({"numbers": nums, "positions": pos})
    tblite_singlepoint(
        {"numbers": nums, "positions": pos, "uhf": None, "spin": None}
    )
    tblite_singlepoint({"numbers": nums, "positions": pos}, uhf=None, spin=1)
    tblite_singlepoint(
        {"numbers": nums, "positions": pos, "spin": 1}, uhf=None, spin=None
    )
    assert recorded_uhf == [1, 1, 1, None, None, 1, 1]

    with pytest.raises(ValueError, match="Conflicting"):
        tblite_singlepoint(
            {"numbers": nums, "positions": pos, "uhf": 1, "spin": 2}
        )
    with pytest.raises(ValueError, match="Conflicting"):
        tblite_singlepoint({"numbers": nums, "positions": pos}, uhf=1, spin=2)
    with pytest.raises(
        ValueError, match="integer number of unpaired electrons"
    ):
        tblite_singlepoint(
            {"numbers": nums, "positions": pos, "uhf": None, "spin": 0.5}
        )
    with pytest.raises(ValueError, match="non-negative"):
        tblite_singlepoint(
            {"numbers": nums, "positions": pos, "uhf": -1, "spin": None}
        )
