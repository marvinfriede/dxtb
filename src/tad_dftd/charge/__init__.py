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
Charge model dispatch (`eeq`, `eeqbc`) for charge-dependent DFT-D models.
"""
from __future__ import annotations

import torch
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.tools import is_compiling
from tad_mctc.typing import Tensor

from .eeq import EEQModel, get_eeq_charges
from .eeqbc import EEQBCModel, get_eeqbc_charges
from .eeqbc_param import MAX_ELEM_EEQBC

__all__ = [
    "EEQBCModel",
    "EEQModel",
    "MAX_ELEM_EEQBC",
    "get_charges",
    "get_charges_eeq",
    "get_charges_eeqbc",
    "get_eeq_charges",
    "get_eeqbc_charges",
]

get_charges_eeq = get_eeq_charges
get_charges_eeqbc = get_eeqbc_charges


def get_charges(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor | float | int | None = None,
    qmod: str = "eeq",
    lattice: Tensor | None = None,
    periodic: Tensor | None = None,
    cutoff: Tensor | float | None = None,
) -> Tensor:
    """
    Evaluate atomic partial charges for a given charge model ``qmod``.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates in Bohr of shape ``(..., nat, 3)``.
    charge : Tensor | float | int | None, optional
        Total molecular charge. Defaults to ``0.0``.
    qmod : str, optional
        Charge model identifier (``"eeq"`` or ``"eeqbc"``). Defaults to ``"eeq"``.
    lattice : Tensor | None, optional
        Lattice vectors of shape ``(..., 3, 3)``.
    periodic : Tensor | None, optional
        Periodic boundary condition flags of shape ``(..., 3)`` or ``(3,)``.
    cutoff : Tensor | float | None, optional
        Real-space coordination-number cutoff in Bohr for the charge model.

    Returns
    -------
    Tensor
        Atomic partial charges of shape ``(..., nat)``.
    """
    qkey = qmod.lower().strip()
    if qkey == "eeq2019":
        qkey = "eeq"
    elif qkey == "eeqbc2025":
        qkey = "eeqbc"
    if qkey in ("gfn2", "gxtb"):
        raise ValueError("This dispersion model requires external charges")
    if qkey not in ("eeq", "eeqbc"):
        raise ValueError(f"Unknown charge model '{qmod}'.")

    if not is_compiling() and not is_functorch_tensor(numbers):
        if bool((numbers < 0).any()) or bool((numbers > MAX_ELEM_EEQBC).any()):
            raise ValueError(
                f"Structure contains unsupported element for '{qkey}' charge model "
                f"(valid range: 1..{MAX_ELEM_EEQBC})."
            )

    if qkey == "eeq":
        return get_eeq_charges(
            numbers=numbers,
            positions=positions,
            charge=charge,
            lattice=lattice,
            periodic=periodic,
            cutoff=cutoff,
        )
    return get_eeqbc_charges(
        numbers=numbers,
        positions=positions,
        charge=charge,
        lattice=lattice,
        periodic=periodic,
        cutoff=cutoff,
    )
