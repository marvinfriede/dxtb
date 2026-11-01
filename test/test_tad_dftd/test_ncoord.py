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
Tests for coordination number models in ``tad_dftd.ncoord``.
"""
from __future__ import annotations

import pytest
import tad_mctc as mctc
import torch

from tad_dftd import ncoord


@pytest.mark.parametrize(
    "fn",
    [
        ncoord.cn_d3,
        ncoord.cn_d4,
        ncoord.cn_d4srev,
        ncoord.cn_eeq,
    ],
)
def test_coordination_numbers_and_autograd(fn) -> None:
    """Verify all CN models evaluate finite positive values on H2O, mask padding, and differentiate."""
    numbers = torch.tensor([[8, 1, 1, 0]], dtype=torch.long)
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, -0.73578586],
                [1.44183153, 0.0, 0.36789293],
                [-1.44183153, 0.0, 0.36789293],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    mol = mctc.Structure(numbers=numbers, positions=positions)
    cn = fn(mol)
    assert cn.shape == (1, 4)
    # Real atoms have positive CN
    assert torch.all(cn[0, :3] > 0.0)
    # Both H atoms have equal CN by symmetry
    assert pytest.approx(cn[0, 1].item(), abs=1e-12) == cn[0, 2].item()
    # Padding atom has zero CN
    assert cn[0, 3].item() == 0.0

    # Check gradient w.r.t. positions
    loss = cn.sum()
    (grad,) = torch.autograd.grad(loss, positions)
    assert torch.all(torch.isfinite(grad))
    # Translation invariance: sum of gradients over atoms is zero
    assert torch.allclose(grad.sum(dim=-2), torch.zeros(1, 3, dtype=torch.float64), atol=1e-12)
    # Padding atom gradient is zero
    assert torch.all(grad[0, 3] == 0.0)
