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
"""Test helpers for converting PySCF AO references to dxtb order."""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.typing import Tensor


def pyscf_ao_permutation(ihelp: IndexHelper) -> Tensor:
    """Return source indices selecting PySCF AO data in dxtb shell order."""
    angular = ihelp.angular.detach().cpu().tolist()
    orbitals_per_shell = ihelp.orbitals_per_shell.detach().cpu().tolist()
    indices: list[int] = []
    offset = 0

    for momentum, count in zip(angular, orbitals_per_shell):
        if momentum == 1:
            indices.extend((offset + 1, offset + 2, offset))
        else:
            indices.extend(range(offset, offset + count))
        offset += count

    return torch.tensor(indices, dtype=torch.long, device=ihelp.device)


def reorder_pyscf_ao(tensor: Tensor, ihelp: IndexHelper) -> Tensor:
    """Reorder the final two AO axes, preserving any leading components."""
    permutation = pyscf_ao_permutation(ihelp)
    return tensor.index_select(-2, permutation).index_select(-1, permutation)
