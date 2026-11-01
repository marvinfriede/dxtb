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
Tests for ``tad_dftd.cutoff`` and ``tad_dftd.pairs``.
"""
from __future__ import annotations

import pytest
import tad_mctc as mctc
import torch

from tad_dftd import defaults
from tad_dftd.cutoff import Cutoff, smooth_cutoff
from tad_dftd.pairs import DensePairs, SparsePairs


def test_cutoff_validation_and_from_model() -> None:
    """Verify Cutoff construction, validation, and model-specific defaults."""
    c_default = Cutoff()
    assert c_default.cn == defaults.D3_CN_CUTOFF
    assert c_default.disp2 == defaults.D3_DISP2_CUTOFF
    assert c_default.disp3 == defaults.D3_DISP3_CUTOFF
    assert c_default.cn_eeq == defaults.D4_CN_EEQ_CUTOFF
    assert c_default.width2 == 0.0
    assert c_default.width3 == 0.0

    c_d1 = Cutoff.from_model("d1")
    assert c_d1.disp2 == defaults.D1_DISP2_CUTOFF

    c_d4 = Cutoff.from_model("d4")
    assert c_d4.cn == defaults.D4_CN_CUTOFF
    assert c_d4.disp2 == defaults.D4_DISP2_CUTOFF
    assert c_d4.disp3 == defaults.D4_DISP3_CUTOFF

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        Cutoff.from_model("d5")

    with pytest.raises(TypeError, match="must be a plain number"):
        Cutoff(cn=torch.tensor(40.0))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must not be negative"):
        Cutoff(width2=-1.0)


def test_smooth_cutoff_behavior_and_grad() -> None:
    """Verify quintic C2 smooth_cutoff switching function and its derivatives."""
    r = torch.tensor([20.0, 40.0, 45.0, 50.0, 55.0], dtype=torch.float64, requires_grad=True)
    s = smooth_cutoff(r, cutoff=50.0, width=10.0)

    # r <= 40 -> 1.0; r = 45 (midpoint) -> 0.5; r >= 50 -> 0.0
    assert pytest.approx(s[0].item(), abs=1e-14) == 1.0
    assert pytest.approx(s[1].item(), abs=1e-14) == 1.0
    assert pytest.approx(s[2].item(), abs=1e-14) == 0.5
    assert pytest.approx(s[3].item(), abs=1e-14) == 0.0
    assert pytest.approx(s[4].item(), abs=1e-14) == 0.0

    # First derivative vanishes at endpoints r=40 and r=50
    grad = torch.autograd.grad(s.sum(), r)[0]
    assert pytest.approx(grad[1].item(), abs=1e-14) == 0.0
    assert pytest.approx(grad[3].item(), abs=1e-14) == 0.0
    assert grad[2].item() < 0.0  # monotonically decreasing on (40, 50)


def test_dense_and_sparse_pairs_molecular_and_periodic() -> None:
    """Verify DensePairs and SparsePairs construction and staleness tracking."""
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, -0.73578586],
            [1.44183153, 0.0, 0.36789293],
            [-1.44183153, 0.0, 0.36789293],
        ],
        dtype=torch.float64,
    )
    mol = mctc.Structure(numbers=numbers, positions=positions)
    cutoff = Cutoff(cn=20.0, disp2=30.0, disp3=15.0)

    dp = DensePairs.build(mol, cutoff)
    assert dp.cn is None
    assert dp.disp2 is None
    assert not dp.stale(mol).item()

    sp = SparsePairs.build(mol, cutoff, skin=2.0)
    assert isinstance(sp, SparsePairs)
    assert not sp.stale(mol).item()

    # Displace an atom by more than half skin -> stale becomes True
    displaced = mctc.Structure(
        numbers=numbers,
        positions=positions + torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert sp.stale(displaced).item()

    # Periodic structure
    lattice = torch.eye(3, dtype=torch.float64) * 10.0
    periodic = torch.tensor([True, True, True])
    cell = mctc.Structure(
        numbers=torch.tensor([6, 6], dtype=torch.long),
        positions=torch.tensor([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]], dtype=torch.float64),
        lattice=lattice,
        periodic=periodic,
    )
    dp_pbc = DensePairs.build(cell, cutoff)
    assert dp_pbc.disp2 is not None
    assert not dp_pbc.stale(cell).item()
