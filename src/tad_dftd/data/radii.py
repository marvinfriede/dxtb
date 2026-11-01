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
    "VDW_D3",
    "VDW_PAIRWISE",
    "VDW_PAIRWISE_119",
    "get_damping_radii",
    "get_damping_radius",
]

COV_D3 = mctc_radii.COV_D3
VDW_D3 = mctc_radii.VDW_D3
VDW_PAIRWISE = mctc_radii.VDW_PAIRWISE


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
        table = VDW_PAIRWISE(device=dev, dtype=dt)
        max_vdw = table.shape[0] - 1  # 103
        zi = torch.clamp(numbers_i, min=0, max=max_vdw)
        zj = torch.clamp(numbers_j, min=0, max=max_vdw)
        out = table[zi, zj]
        # For Z > 103, mctc_data_vdwrad uses cov2vdw * get_covalent_rad(Z)
        cov_rad = COV_D3(device=dev, dtype=dt) * 0.75
        cov2vdw = 1.45
        diag_vdw = 0.5 * torch.diagonal(table)
        rad_i = torch.where(
            numbers_i <= max_vdw, diag_vdw[zi], cov2vdw * cov_rad[numbers_i]
        )
        rad_j = torch.where(
            numbers_j <= max_vdw, diag_vdw[zj], cov2vdw * cov_rad[numbers_j]
        )
        in_table = (
            (numbers_i > 0)
            & (numbers_i <= max_vdw)
            & (numbers_j > 0)
            & (numbers_j <= max_vdw)
        )
        out = torch.where(in_table, out, rad_i + rad_j)
        valid = (numbers_i > 0) & (numbers_j > 0)
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


_VDW_PAIRWISE_119_CPU: Tensor = get_damping_radii(
    DampingRadii.RV_DW,
    torch.arange(119, dtype=torch.long),
    dtype=torch.float64,
)


def VDW_PAIRWISE_119(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Pairwise van-der-Waals radii in Bohr extended up to Z=118 (shape ``(119, 119)``),
    matching ``mctc_data_vdwrad::get_vdw_rad_num``.
    """
    dt = get_default_dtype() if dtype is None else dtype
    return _VDW_PAIRWISE_119_CPU.to(device=device, dtype=dt)
