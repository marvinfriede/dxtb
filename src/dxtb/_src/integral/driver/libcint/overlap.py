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
Implementation: Overlap
=======================

Overlap implementation based on `libcint`.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import pack
from tad_mctc.math import einsum

from dxtb._src.typing import Any, Tensor

from ...types import OverlapIntegral
from ...utils import snorm
from .base import IntegralLibcint
from .driver import LibcintCallData

__all__ = ["OverlapLibcint", "build_overlap_libcint"]


def build_overlap_libcint(call_data: LibcintCallData) -> Tensor:
    """Build the overlap matrix from call-local libcint data."""
    if not isinstance(call_data, LibcintCallData):
        raise TypeError("Expected call-local LibcintCallData.")

    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    matrices = [
        libcint.overlap(item.basis, item.structure)
        for item in call_data.inputs
    ]
    if call_data.batch_mode > 0:
        return pack(matrices)

    if len(matrices) != 1:
        raise RuntimeError("Single-system libcint setup needs one input.")
    return matrices[0]


class OverlapLibcint(OverlapIntegral, IntegralLibcint):
    """
    Overlap integral from atomic orbitals.

    Use the :meth:`build` method to calculate the overlap integral. The
    returned matrix uses a custom autograd function to calculate the
    backward pass with the analytical gradient.
    For the full gradient, i.e., a matrix of shape ``(..., norb, norb, 3)``,
    the :meth:`get_gradient` method should be used.
    """

    def build(self, call_data: LibcintCallData, **_: Any) -> Tensor:
        """
        Calculation of overlap integral using libcint.

        Returns
        -------
        call_data : LibcintCallData
            Call-local wrappers for the current geometry.

        Returns
        -------
        Tensor
            Overlap integral matrix of shape ``(..., norb, norb)``.
        """
        super().checks(call_data)

        self.matrix = build_overlap_libcint(call_data)
        if call_data.batch_mode > 0:
            self.norm = pack(
                [
                    snorm(
                        self.matrix[
                            i,
                            : item.basis.layout(item.structure.numbers).nao,
                            : item.basis.layout(item.structure.numbers).nao,
                        ]
                    )
                    for i, item in enumerate(call_data.inputs)
                ]
            )
        else:
            self.norm = snorm(self.matrix)
        return self.matrix

    def get_gradient(self, call_data: LibcintCallData, **_: Any) -> Tensor:
        """
        Overlap gradient calculation using libcint.

        Parameters
        ----------
        call_data : LibcintCallData
            Call-local wrappers for the current geometry.

        Returns
        -------
        Tensor
            Overlap gradient of shape ``(..., norb, norb, 3)``.
        """
        super().checks(call_data)

        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import libcint

        # build norm if not already available
        if self.norm is None:
            self.build(call_data)

        def fcn(item: Any) -> Tensor:
            # (3, norb, norb)
            grad = libcint.int1e("ipovlp", item.basis, item.structure)

            # Move xyz dimension to last, which is required for the
            # reduction (only works with extra dimension in last)
            return -einsum("...xij->...ijx", grad)

        # batched mode
        if call_data.batch_mode > 0:
            if call_data.batch_mode == 1:
                self.gradient = pack([fcn(item) for item in call_data.inputs])
                return self.gradient

            if call_data.batch_mode == 2:
                self.gradient = torch.stack(
                    [fcn(item) for item in call_data.inputs]
                )
                return self.gradient

            raise ValueError(f"Unknown batch mode '{call_data.batch_mode}'.")

        # single mode
        if len(call_data.inputs) != 1:
            raise RuntimeError("Single-system libcint setup needs one input.")

        self.gradient = fcn(call_data.inputs[0])
        return self.gradient
