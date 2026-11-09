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
Charge model dispatch (`eeq`) for charge-dependent DFT-D models.
"""
from __future__ import annotations

from tad_mctc.typing import Tensor
from tad_multicharge.model.base import SolveMode

from ..data.table import guard_numbers
from .eeq import EEQModel, get_eeq_charges

__all__ = [
    "EEQModel",
    "MAX_ELEM_EEQ",
    "get_charges",
    "get_charges_eeq",
    "get_eeq_charges",
    "normalize_qmodel",
]

MAX_ELEM_EEQ: int = 103

get_charges_eeq = get_eeq_charges


def normalize_qmodel(qmod: str) -> str:
    """
    Canonical key of a charge model name: lower case and stripped, with
    ``"eeq2019"`` mapped to ``"eeq"``.

    Raises
    ------
    NotImplementedError
        For ``"eeqbc"`` and ``"eeqbc2025"``, which are not supported.
    """
    qkey = qmod.lower().strip()
    if qkey == "eeq2019":
        return "eeq"
    if qkey in ("eeqbc", "eeqbc2025"):
        raise NotImplementedError(
            f"Charge model '{qmod}' is not supported in this release "
            "because the upstream model is not finalized."
        )
    return qkey


def get_charges(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor | float | int | None = None,
    qmod: str = "eeq",
    lattice: Tensor | None = None,
    periodic: Tensor | None = None,
    cutoff: Tensor | float | None = None,
    *,
    solve_mode: SolveMode = "schur",
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
        Charge model identifier (``"eeq"`` or ``"eeq2019"``). Defaults to ``"eeq"``.
    lattice : Tensor | None, optional
        Lattice vectors of shape ``(..., 3, 3)``.
    periodic : Tensor | None, optional
        Periodic boundary condition flags of shape ``(..., 3)`` or ``(3,)``.
    cutoff : Tensor | float | None, optional
        Real-space coordination-number cutoff in Bohr for the charge model.
    solve_mode : SolveMode, optional
        Linear solver backend for the EEQ solve. Defaults to ``"schur"``.

    Returns
    -------
    Tensor
        Atomic partial charges of shape ``(..., nat)``.
    """
    qkey = normalize_qmodel(qmod)
    if qkey in ("gfn2", "gxtb"):
        raise ValueError("This dispersion model requires external charges")
    if qkey != "eeq":
        raise ValueError(f"Unknown charge model '{qmod}'.")

    invalid = (numbers < 0) | (numbers > MAX_ELEM_EEQ)
    numbers = guard_numbers(
        numbers,
        invalid,
        f"Structure contains unsupported element for '{qkey}' charge model "
        f"(valid range: 1..{MAX_ELEM_EEQ}).",
    )

    return get_eeq_charges(
        numbers=numbers,
        positions=positions,
        charge=charge,
        lattice=lattice,
        periodic=periodic,
        cutoff=cutoff,
        solve_mode=solve_mode,
    )
