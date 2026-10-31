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
Type annotations for ``tad_dftd``.
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Literal,
    NoReturn,
    Protocol,
    TypedDict,
    overload,
    runtime_checkable,
)

from tad_mctc.typing import (
    DD,
    CountingFunction,
    Self,
    Sliceable,
    TableFunction,
    Tensor,
    TensorLike,
    get_default_device,
    get_default_dtype,
    override,
)

__all__ = [
    "Any",
    "Callable",
    "CountingFunction",
    "DD",
    "DampingFunction",
    "DispersionMethod",
    "Literal",
    "ModelName",
    "NoReturn",
    "Protocol",
    "Self",
    "Sliceable",
    "TableFunction",
    "Tensor",
    "TensorLike",
    "TypedDict",
    "WeightingFunction",
    "get_default_device",
    "get_default_dtype",
    "overload",
    "override",
    "runtime_checkable",
]

ModelName = Literal["d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev", "d5"]

WeightingFunction = Callable[[Tensor, Tensor], Tensor]


@runtime_checkable
class DampingFunction(Protocol):
    """Protocol for pairwise and three-body damping functions."""

    def __call__(
        self,
        order: int,
        r: Tensor,
        radii: Tensor,
        param: dict[str, Tensor | float],
        *,
        qq: Tensor | None = None,
        c6: Tensor | None = None,
        numbers: Tensor | None = None,
    ) -> Tensor: ...


@runtime_checkable
class DispersionMethod(Protocol):
    """Protocol for dispersion evaluation functions."""

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        param: dict[str, Tensor | float],
        **kwargs: Any,
    ) -> Tensor: ...
