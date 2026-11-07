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
Classical EEQ (2019) charge model wrapper around :mod:`tad_multicharge.eeq`.
"""
from __future__ import annotations

import torch
from tad_mctc.io.structure import Structure
from tad_mctc.typing import Tensor
from tad_multicharge.model.base import SolveMode
from tad_multicharge.model.eeq import EEQModel

__all__ = ["EEQModel", "get_eeq_charges"]


def get_eeq_charges(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor | float | int | None = None,
    lattice: Tensor | None = None,
    periodic: Tensor | None = None,
    cutoff: Tensor | float | None = None,
    *,
    solve_mode: SolveMode = "schur",
) -> Tensor:
    """
    Calculate atomic partial charges using the EEQ (2019) charge model.
    """
    if isinstance(charge, (float, int)):
        chrg = torch.tensor(charge, device=positions.device, dtype=positions.dtype)
    elif charge is None:
        chrg = torch.zeros(
            positions.shape[:-2], device=positions.device, dtype=positions.dtype
        )
    else:
        chrg = charge.to(device=positions.device, dtype=positions.dtype)

    struct = Structure(
        numbers=numbers,
        positions=positions,
        charge=chrg,
        lattice=lattice,
        periodic=periodic,
    )
    model = EEQModel.param2019(device=positions.device, dtype=positions.dtype)
    if cutoff is not None:
        model = model.replace(cn=model.cn.replace(cutoff=cutoff))
    return model(struct, return_energy=False, solve_mode=solve_mode)
