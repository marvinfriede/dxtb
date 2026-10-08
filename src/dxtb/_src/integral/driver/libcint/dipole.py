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
Implementation: Dipole
======================

Dipole integral implementation based on `libcint`.
"""

from __future__ import annotations

from dxtb._src.typing import Tensor

from ...types import DipoleIntegral
from .driver import LibcintCallData
from .multipole import MultipoleLibcint, build_multipole_libcint

__all__ = ["DipoleLibcint", "build_dipole_libcint"]


def build_dipole_libcint(call_data: LibcintCallData) -> Tensor:
    """Build a raw origin-centered dipole matrix from call-local wrappers."""
    return build_multipole_libcint(call_data, "r0")


class DipoleLibcint(DipoleIntegral, MultipoleLibcint):
    """
    Dipole integral from atomic orbitals.
    """

    def build(self, call_data: LibcintCallData) -> Tensor:
        """
        Calculation of dipole integral using libcint.

        Parameters
        ----------
        call_data : LibcintCallData
            Call-local wrappers for the current geometry.

        Returns
        -------
        Tensor
            Dipole integral.
        """
        return self.multipole(call_data, "r0")

    def get_gradient(self, call_data: LibcintCallData) -> Tensor:
        """
        Calculation of dipole gradient using libcint.

        Parameters
        ----------
        call_data : LibcintCallData
            Call-local wrappers for the current geometry.

        Returns
        -------
        Tensor
            Dipole gradient.
        """
        raise NotImplementedError("Gradient calculation not implemented.")
