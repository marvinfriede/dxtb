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
"""Pure AO stationarity matrices shared by certification and SCF methods."""

from __future__ import annotations

import torch

from dxtb._src.typing import Tensor

__all__ = [
    "ao_stationarity_matrix",
    "orthogonalized_stationarity_matrix",
]


def ao_stationarity_matrix(
    fock: Tensor,
    density: Tensor,
    overlap: Tensor,
    orbital_mask: Tensor,
) -> Tensor:
    """Return the padded AO commutator ``F P S - S P F``."""
    residual = fock @ density @ overlap - overlap @ density @ fock
    pair_mask = orbital_mask[..., :, None] & orbital_mask[..., None, :]
    return torch.where(pair_mask, residual, 0.0)


def orthogonalized_stationarity_matrix(
    ao_residual: Tensor,
    inverse_sqrt_overlap: Tensor,
    orbital_mask: Tensor,
) -> Tensor:
    """Transform and mask the AO commutator in the orthonormal basis."""
    transformed = (
        inverse_sqrt_overlap.transpose(-2, -1)
        @ ao_residual
        @ inverse_sqrt_overlap
    )
    pair_mask = orbital_mask[..., :, None] & orbital_mask[..., None, :]
    return torch.where(pair_mask, transformed, 0.0)
