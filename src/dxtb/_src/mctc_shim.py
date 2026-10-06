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
Temporary shim for the unreleased dispersion packages
=====================================================

``tad-dftd3`` 0.7.0 and ``tad-dftd4`` 0.8.0 are built for tad-mctc 0.8 and
0.7 and use names that tad-mctc 0.9 no longer has. Until compatible releases
exist, the missing names are added to ``tad_mctc`` here, before the
dispersion packages are imported.

Remove this module together with :mod:`dxtb._src.ncoord.legacy` once
tad-dftd3 and tad-dftd4 are released for tad-mctc 0.9.
"""

from __future__ import annotations

import typing as _t

import tad_mctc.ncoord as _ncoord
import tad_mctc.storch as _storch
import tad_mctc.tools.memory as _memory
import tad_mctc.typing as _typing
import torch

__all__: list[str] = []


def _memory_device(device: torch.device) -> tuple[float, float]:
    """Available and total memory of the device in MB (removed in 0.8)."""
    if not isinstance(device, torch.device):
        raise TypeError(
            f"Device should be a `torch.device` object, but is a {type(device)}."
        )

    if device.type == "cpu":
        from psutil import virtual_memory  # pylint: disable=C0415

        mem = virtual_memory()
        free, total = mem.available, mem.total
    elif device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
    else:
        raise ValueError(f"Unsupported device: {device}")

    return free / (1024**2), total / (1024**2)


def _apply() -> None:
    if not hasattr(_memory, "memory_device"):
        _memory.memory_device = _memory_device  # type: ignore[attr-defined]

    # names of the standard library typing module that tad-mctc 0.9 dropped
    for name in ("Any", "Literal", "TypedDict", "overload"):
        if not hasattr(_typing, name):
            setattr(_typing, name, getattr(_t, name))

    # tad-mctc 0.8 prefixed the safe operations (``divide`` -> ``safe_divide``)
    for name in ("divide", "pow", "reciprocal", "sqrt"):
        if not hasattr(_storch, name):
            setattr(_storch, name, getattr(_storch, f"safe_{name}"))

    if not hasattr(_typing, "CNFunc"):
        from dxtb._src.typing import CNFunc

        _typing.CNFunc = CNFunc  # type: ignore[attr-defined]

    if not hasattr(_ncoord, "coordination_number"):
        from dxtb._src.ncoord import legacy

        _ncoord.coordination_number = legacy.coordination_number  # type: ignore[attr-defined]


_apply()
