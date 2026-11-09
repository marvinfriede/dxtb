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

from tad_mctc.batch import pack

from dxtb._src.typing import Tensor

from .driver import LibcintCallData

__all__ = ["build_overlap_libcint"]


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
