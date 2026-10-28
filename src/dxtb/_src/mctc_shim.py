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
Compatibility shim for published dispersion packages
=====================================================

The supported tad-dftd3 0.7.0 release requires older tad-mctc APIs, while
dxtb requires tad-mctc 0.11.3. The separately installed tad-dftd4 0.8.0 also
uses names removed from tad-mctc 0.9. These published packages are installed
with --no-deps in development environments; the minimal names they need are
restored here before either dispersion package is imported.

This does not make the package metadata conflict resolver-compatible. Remove
the shim only after supported dispersion releases accept tad-mctc 0.11.3.
"""

from __future__ import annotations

import typing as _t

import tad_mctc.autograd as _autograd
import tad_mctc.ncoord as _ncoord
import tad_mctc.storch as _storch
import tad_mctc.typing as _typing

__all__: list[str] = []


def _apply() -> None:
    # tad-dftd3 0.7.0 imports this legacy name. The current mctc equivalent
    # asks whether a vmap layer is active, including nested transforms.
    if not hasattr(_autograd, "is_batched"):
        _autograd.is_batched = _autograd.is_vmapped  # type: ignore[attr-defined]

    # names of the standard library typing module that tad-mctc 0.9 dropped
    for name in ("Any", "Literal", "TypedDict", "overload"):
        if not hasattr(_typing, name):
            setattr(_typing, name, getattr(_t, name))

    # tad-mctc 0.8 prefixed the safe operations (``divide`` -> ``safe_divide``)
    for name in ("divide", "sqrt"):
        if not hasattr(_storch, name):
            setattr(_storch, name, getattr(_storch, f"safe_{name}"))

    if not hasattr(_typing, "CNFunc"):
        from dxtb._src.typing import CNFunc

        _typing.CNFunc = CNFunc  # type: ignore[attr-defined]

    if not hasattr(_ncoord, "coordination_number"):
        from dxtb._src.ncoord import legacy

        _ncoord.coordination_number = legacy.coordination_number  # type: ignore[attr-defined]


_apply()
