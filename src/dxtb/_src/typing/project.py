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
Typing: Project
===============

Project-specific type annotations.
"""

from __future__ import annotations

import torch
from tad_mctc.typing import CountingFunction, Tensor

from .builtin import Any, Protocol, TypedDict
from .compat import Slicer

__all__ = [
    "CNFunc",
    "CNFunction",
    "CNGradFunction",
    "ContainerData",
    "Slicers",
]


# tad-mctc 0.8 removed these protocols together with the functional
# coordination number API (see ``dxtb._src.ncoord.legacy``).


class CNFunc(Protocol):
    """Type annotation for a specific coordination number function."""

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        counting_function: CountingFunction | None = None,
    ) -> Tensor: ...


class CNFunction(Protocol):
    """Type annotation for a general coordination number function."""

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        *,
        counting_function: CountingFunction | None = None,
        rcov: Tensor | None = None,
        en: Tensor | None = None,
        cutoff: Tensor | None = None,
        kcn: float = 7.5,
        **kwargs: Any,
    ) -> Tensor: ...


class CNGradFunction(Protocol):
    """Type annotation for a coordination number gradient function."""

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        *,
        dcounting_function: CountingFunction | None = None,
        rcov: Tensor | None = None,
        en: Tensor | None = None,
        cutoff: Tensor | None = None,
        kcn: float = 7.5,
        **kwargs: Any,
    ) -> Tensor: ...


class Slicers(TypedDict):
    """Collection of slicers of different resolutions for culling in SCF."""

    orbital: Slicer
    """Slicer for orbital-resolved variables."""
    shell: Slicer
    """Slicer for shell-resolved variables."""
    atom: Slicer
    """Slicer for atom-resolved variables."""


class ContainerData(TypedDict):
    """Shape and label information of Potentials."""

    mono: torch.Size | None
    """Shape of the monopolar potential."""

    dipole: torch.Size | None
    """Shape of the dipolar potential."""

    quad: torch.Size | None
    """Shape of the quadrupolar potential."""

    label: list[str] | str | None
    """Labels for the interactions contributing to the potential."""
