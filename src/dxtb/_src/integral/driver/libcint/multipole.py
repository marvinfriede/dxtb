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

from typing import TYPE_CHECKING

from tad_mctc.batch import pack

from dxtb._src.typing import Tensor

from .base import IntegralLibcint
from .driver import LibcintCallData

if TYPE_CHECKING:
    from .driver import IntDriverLibcint

__all__ = ["MultipoleLibcint", "build_multipole_libcint"]


def build_multipole_libcint(
    call_data: LibcintCallData, intstring: str
) -> Tensor:
    """Build a raw origin-centered libcint multipole matrix."""
    from dxtb._src.exlibs import libcint

    allowed_mps = ("r0", "r0r0", "r0r0r0")
    if intstring not in allowed_mps:
        raise ValueError(
            f"Unknown integral string '{intstring}' provided. "
            f"Only '{', '.join(allowed_mps)}' are allowed."
        )

    matrices = [
        libcint.int1e(intstring, driver) for driver in call_data.drivers
    ]
    if call_data.batch_mode > 0:
        return pack(matrices)
    if len(matrices) != 1:
        raise RuntimeError("Single-system libcint setup needs one wrapper.")
    return matrices[0]


class MultipoleLibcint(IntegralLibcint):
    """
    Base class for multipole integrals calculated with `libcint`.
    """

    def multipole(self, call_data: LibcintCallData, intstring: str) -> Tensor:
        """
        Calculation of multipole integral. The integral is normalized, using
        the diagonal of the overlap integral.

        Parameters
        ----------
        call_data : LibcintCallData
            Call-local wrappers for the current geometry.
        intstring : str
            String for `libcint` integral engine.

        Returns
        -------
        Tensor
            Normalized multipole integral.
        """
        super().checks(call_data)

        self.matrix = build_multipole_libcint(call_data, intstring)
        return self.matrix
