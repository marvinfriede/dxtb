# This file is part of dxtb.
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
Integral container
==================

Immutable container for the integral matrices of one evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dxtb._src.constants import defaults
from dxtb._src.typing import Tensor

__all__ = ["IntegralMatrices"]


@dataclass(frozen=True, eq=False)
class IntegralMatrices:
    """Immutable tensor values produced by one integral evaluation."""

    hcore: Tensor
    overlap: Tensor
    dipole: Tensor | None = None
    quadrupole: Tensor | None = None

    def __post_init__(self) -> None:
        reference = self.hcore
        if (
            reference.ndim not in (2, 3)
            or reference.shape[-2] != reference.shape[-1]
        ):
            raise ValueError("Tensor 'hcore' must have shape (..., nao, nao).")
        if self.overlap.shape != reference.shape:
            raise ValueError("Tensor 'overlap' must match the hcore shape.")
        if (
            self.overlap.device != reference.device
            or self.overlap.dtype != reference.dtype
        ):
            raise ValueError(
                "All integral matrices must share device and dtype."
            )

        for name, tensor, components in (
            ("dipole", self.dipole, (defaults.DP_SHAPE,)),
            ("quadrupole", self.quadrupole, (6, 9)),
        ):
            if tensor is None:
                continue
            if tensor.ndim != reference.ndim + 1:
                raise ValueError(f"Tensor '{name}' has an incompatible rank.")
            expected = (
                *reference.shape[:-2],
                tensor.shape[-3],
                *reference.shape[-2:],
            )
            if tensor.shape != expected or tensor.shape[-3] not in components:
                raise ValueError(f"Tensor '{name}' has an incompatible shape.")
            if (
                tensor.device != reference.device
                or tensor.dtype != reference.dtype
            ):
                raise ValueError(
                    "All integral matrices must share device and dtype."
                )

    @property
    def device(self) -> torch.device:
        return self.hcore.device

    @property
    def dtype(self) -> torch.dtype:
        return self.hcore.dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> IntegralMatrices:
        """Return a copy with all matrix tensors converted."""
        return IntegralMatrices(
            hcore=self.hcore.to(device=device, dtype=dtype),
            overlap=self.overlap.to(device=device, dtype=dtype),
            dipole=(
                None
                if self.dipole is None
                else self.dipole.to(device=device, dtype=dtype)
            ),
            quadrupole=(
                None
                if self.quadrupole is None
                else self.quadrupole.to(device=device, dtype=dtype)
            ),
        )

    def type(self, dtype: torch.dtype) -> IntegralMatrices:
        """Return a copy with a different floating point type."""
        return self.to(dtype=dtype)

    def slice(
        self, matrix_indices: tuple, multipole_indices: tuple
    ) -> IntegralMatrices:
        """Return a value with its orbital dimensions sliced."""
        return IntegralMatrices(
            hcore=self.hcore[matrix_indices],
            overlap=self.overlap[matrix_indices],
            dipole=(
                None if self.dipole is None else self.dipole[multipole_indices]
            ),
            quadrupole=(
                None
                if self.quadrupole is None
                else self.quadrupole[multipole_indices]
            ),
        )

    def __str__(self) -> str:  # pragma: no cover
        fields = ("hcore", "overlap", "dipole", "quadrupole")
        details = [
            f"\n  {name}={getattr(self, name).shape if getattr(self, name) is not None else 'None'}"
            for name in fields
        ]
        return f"IntegralMatrices({', '.join(details)}\n)"
