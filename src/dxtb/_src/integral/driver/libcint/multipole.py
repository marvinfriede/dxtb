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
Implementation: Multipole Base
==============================

Template for calculation and modification of multipole integrals.
"""

from __future__ import annotations

from dxtb._src.typing import Tensor

from .driver import LibcintCallData

__all__ = ["build_multipole_libcint"]


def build_multipole_libcint(
    call_data: LibcintCallData, intstring: str
) -> Tensor:
    """Build a raw libcint multipole matrix for the requested center."""
    from dxtb._src.exlibs import libcint

    allowed_mps = ("r0", "j", "r0r0", "jj", "jjj")
    if intstring not in allowed_mps:
        raise ValueError(
            f"Unknown integral string '{intstring}' provided. "
            f"Only '{', '.join(allowed_mps)}' are allowed."
        )

    return libcint.int1e(intstring, call_data.basis, call_data.structure)
