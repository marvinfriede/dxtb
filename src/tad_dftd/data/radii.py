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
Covalent and van-der-Waals radii and damping radius constructors for ``tad_dftd``.
"""
from __future__ import annotations

from enum import IntEnum

import torch
from tad_mctc.data import radii as mctc_radii
from tad_mctc.typing import Tensor, get_default_dtype

from .d1 import D1_RADII
from .d2 import D2_RADII
from .r4r2 import R4R2

__all__ = [
    "COV_D3",
    "DampingRadii",
    "EEQBC_COV_RADII",
    "VDW_D3",
    "VDW_PAIRWISE",
    "get_damping_radii",
    "get_damping_radius",
]

COV_D3 = mctc_radii.COV_D3
VDW_D3 = mctc_radii.VDW_D3
VDW_PAIRWISE = mctc_radii.VDW_PAIRWISE

# From multicharge/param/eeqbc2025.f90 (scaled by 0.5_wp in Fortran, Z=0..103, padded to 118)
_EEQBC_COV_RADII_UNSCALED: list[float] = [
    0.0,
    0.5930850061, 0.6828783524, 2.5060627321, 2.1530174721,
    2.0808370973, 2.1547612587, 2.3825166136, 2.3845548778,
    2.2785689025, 1.5895245890, 3.2262348700, 3.0085723165,
    2.4255063979, 2.8239473425, 3.1186237627, 3.3331763174,
    3.1216907679, 1.5732371114, 3.2900208529, 3.4202206471,
    2.9200724722, 3.0066744258, 2.8630113712, 2.8148129713,
    2.4967937635, 3.0257979292, 3.1144574229, 2.8415686668,
    2.3076713228, 2.7929299468, 2.9384555000, 3.1211972610,
    3.3019239452, 3.5482037468, 3.5268705061, 2.3017605635,
    3.6453427844, 3.6627725418, 3.1732780680, 3.3336121647,
    3.1865126417, 3.1583027293, 2.9482110819, 3.1853598435,
    3.2430942807, 3.1951785079, 2.9793621882, 3.1543875559,
    3.1850367978, 3.5529181164, 3.4500895536, 3.8418786490,
    3.8953057563, 2.8735372659, 3.7155011702, 3.9719089931,
    3.1569703924, 3.4936913809, 3.3335208117, 3.9243845028,
    3.8732096780, 3.1689369113, 3.4987436252, 3.5190884141,
    3.6663276643, 3.7941855096, 3.6947440641, 3.7383619093,
    3.7065668579, 3.5989802683, 3.5113445936, 3.3622392888,
    3.0067203462, 3.0264024510, 2.8716706964, 3.2964048554,
    3.3131562414, 3.4380505276, 2.9801688586, 3.6006573663,
    3.4878494275, 4.0080167544, 3.4867375894, 3.8412537998,
    4.0986699510, 3.0264945260, 4.1292282410, 3.7772409489,
    1.8907313528, 1.0395376590, 1.9323562902, 3.0573080720,
    2.7107637944, 3.3756041761, 3.2094752255, 3.2207515613,
    2.7188494129, 2.7941335957, 3.2020860461, 1.5623494196,
    2.8208057454, 2.8130190589, 2.7482546063,
] + [0.0] * 15


def EEQBC_COV_RADII(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Covalent radii for EEQ_BC coordination numbers in Bohr for Z=0..118."""
    dt = get_default_dtype() if dtype is None else dtype
    rad_64 = 0.5 * torch.tensor(
        _EEQBC_COV_RADII_UNSCALED, device=device, dtype=torch.float64
    )
    return rad_64.to(dtype=dt)


class DampingRadii(IntEnum):
    """
    Enumeration of pairwise damping radius definitions matching ``dftd/damping/radii.f90``.
    """

    RV_D1 = 1
    """Additive D1 atomic van-der-Waals radii: ``R_0(A) + R_0(B)``."""

    RV_D2 = 2
    """Additive D2 atomic van-der-Waals radii: ``R_0(A) + R_0(B)``."""

    RV_DW = 3
    """Pairwise D3 van-der-Waals radii ``r0ab(Z_A, Z_B)``."""

    R_COV = 4
    """Scaled additive D3 covalent radii ``0.75 * (R_cov(A) + R_cov(B))``."""

    R4_R2 = 5
    """Expectation-value radii ``sqrt(3 * r4r2(A) * r4r2(B))``."""


def get_damping_radius(
    kind: DampingRadii | str | int,
    numbers_i: Tensor,
    numbers_j: Tensor,
    c8_over_c6: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Compute pairwise damping radii for index pairs ``(numbers_i, numbers_j)``.

    Parameters
    ----------
    kind : DampingRadii | str | int
        Type of damping radius (``"rvd1"``, ``"rvd2"``, ``"rvdw"``, ``"rcov"``,
        ``"r4r2"``, or :class:`DampingRadii`).
    numbers_i : Tensor
        Atomic numbers of first atom in pair.
    numbers_j : Tensor
        Atomic numbers of second atom in pair.
    c8_over_c6 : Tensor | None, optional
        Optional explicit ``sqrt(C8 / C6)`` tensor for ``"r4r2"`` mode.
    device : torch.device | None, optional
        Target device.
    dtype : torch.dtype | None, optional
        Target floating-point dtype.

    Returns
    -------
    Tensor
        Pairwise damping radii in Bohr with broadcasted shape of ``(numbers_i, numbers_j)``.
    """
    dev = numbers_i.device if device is None else device
    dt = get_default_dtype() if dtype is None else dtype

    if isinstance(kind, str):
        key = kind.lower().strip().replace("_", "")
        mapping = {
            "rvd1": DampingRadii.RV_D1,
            "d1": DampingRadii.RV_D1,
            "rvd2": DampingRadii.RV_D2,
            "d2": DampingRadii.RV_D2,
            "rvdw": DampingRadii.RV_DW,
            "vdw": DampingRadii.RV_DW,
            "rcov": DampingRadii.R_COV,
            "cov": DampingRadii.R_COV,
            "r4r2": DampingRadii.R4_R2,
        }
        if key not in mapping:
            raise ValueError(f"Unknown damping radius kind '{kind}'.")
        kind_enum = mapping[key]
    else:
        kind_enum = DampingRadii(kind)

    if kind_enum == DampingRadii.RV_D1:
        rad = D1_RADII(device=dev, dtype=dt)
        ri = rad[numbers_i]
        rj = rad[numbers_j]
        valid = (numbers_i > 0) & (numbers_j > 0) & (ri > 0.0) & (rj > 0.0)
        return torch.where(valid, ri + rj, torch.zeros_like(ri + rj))

    if kind_enum == DampingRadii.RV_D2:
        rad = D2_RADII(device=dev, dtype=dt)
        ri = rad[numbers_i]
        rj = rad[numbers_j]
        valid = (numbers_i > 0) & (numbers_j > 0) & (ri > 0.0) & (rj > 0.0)
        return torch.where(valid, ri + rj, torch.zeros_like(ri + rj))

    if kind_enum == DampingRadii.RV_DW:
        table = VDW_D3(device=dev, dtype=dt)
        zi = torch.clamp(numbers_i, min=0, max=table.shape[0] - 1)
        zj = torch.clamp(numbers_j, min=0, max=table.shape[1] - 1)
        out = table[zi, zj]
        valid = (
            (numbers_i > 0)
            & (numbers_i < table.shape[0])
            & (numbers_j > 0)
            & (numbers_j < table.shape[1])
        )
        return torch.where(valid, out, torch.zeros_like(out))

    if kind_enum == DampingRadii.R_COV:
        # In mctc-lib/dftd, get_covalent_rad returns COV_D3 * 0.75 (unscaled Bohr)
        rad = COV_D3(device=dev, dtype=dt) * 0.75
        ri = rad[numbers_i]
        rj = rad[numbers_j]
        valid = (numbers_i > 0) & (numbers_j > 0)
        return torch.where(valid, ri + rj, torch.zeros_like(ri + rj))

    if kind_enum == DampingRadii.R4_R2:
        if c8_over_c6 is not None:
            pos = c8_over_c6 > 0
            return torch.where(
                pos,
                torch.sqrt(torch.where(pos, c8_over_c6, torch.ones_like(c8_over_c6))),
                torch.zeros_like(c8_over_c6),
            )
        r4r2 = R4R2(device=dev, dtype=dt)
        ri = r4r2[numbers_i]
        rj = r4r2[numbers_j]
        prod = 3.0 * ri * rj
        pos = prod > 0
        return torch.where(
            pos,
            torch.sqrt(torch.where(pos, prod, torch.ones_like(prod))),
            torch.zeros_like(prod),
        )

    raise ValueError(f"Unsupported DampingRadii: {kind_enum}")


def get_damping_radii(
    kind: DampingRadii | str | int,
    numbers: Tensor,
    c8_over_c6: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Compute dense pairwise damping radii ``(..., nat, nat)`` for ``numbers`` of shape ``(..., nat)``.
    """
    return get_damping_radius(
        kind=kind,
        numbers_i=numbers.unsqueeze(-1),
        numbers_j=numbers.unsqueeze(-2),
        c8_over_c6=c8_over_c6,
        device=device,
        dtype=dtype,
    )
