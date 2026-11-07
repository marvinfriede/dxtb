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
"""
Repeated grouped-vmap batch evaluation must not retain tensors.

Each call constructs a batched Calculator, evaluates it through the packed
route (mapped H2O conformer group, or a mapped H2O group plus an eager CH4
singleton), differentiates, and drops everything. After warm-up, tensor count
and storage must reach a plateau.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, Calculator, labels

from ..conftest import DEVICE
from ..molecules import mols as samples
from .util import garbage_collect, has_tensor_memory_growth

DTYPE = torch.double
OPTS = {
    "verbosity": 0,
    "int_driver": labels.INTDRIVER_PYTORCH,
    "maxiter": 30,
}
BATCHES = {
    "conformers": ("H2O", "H2O", "H2O"),
    "mixed": ("H2O", "CH4", "H2O"),
}


def _evaluate(batch: str, mode: str, create_graph: bool) -> None:
    names = BATCHES[batch]
    numbers = pack([samples[name]["numbers"] for name in names]).to(DEVICE)
    positions = pack(
        [
            samples[name]["positions"].to(DEVICE, DTYPE) + 0.01 * lane
            for lane, name in enumerate(names)
        ]
    )
    calc = Calculator(numbers, GFN1_XTB, opts=OPTS, dtype=DTYPE, device=DEVICE)

    if mode == "energy":
        with torch.no_grad():
            calc.singlepoint(positions)
        return

    pos = positions.clone().requires_grad_(True)
    energy = calc.singlepoint(pos).energy.sum()
    (grad,) = torch.autograd.grad(energy, pos, create_graph=create_graph)
    if create_graph:
        # second-order use of the graph; known reference cycle as in
        # test_scf.py, collected by the garbage collector
        grad.square().sum().backward()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("batch", sorted(BATCHES))
@pytest.mark.parametrize(
    "mode, create_graph",
    [("energy", False), ("forces", False), ("forces", True)],
)
def test_repeated_grouped_evaluation_plateaus(
    batch: str, mode: str, create_graph: bool
) -> None:
    garbage_collect()
    leak = has_tensor_memory_growth(
        lambda: _evaluate(batch, mode, create_graph),
        repeats=4,
        gccollect=True,
    )
    garbage_collect()
    assert not leak, "Memory leak detected"
