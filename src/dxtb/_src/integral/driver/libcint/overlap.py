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
Implementation: Overlap (Libcint)
=================================

Overlap implementation based on `libcint`.
"""

from __future__ import annotations

from dxtb._src.typing import Tensor

from .driver import LibcintIntegralSetup

__all__ = ["build_overlap", "build_overlap_libcint"]


def build_overlap(
    setup: LibcintIntegralSetup,
    positions: Tensor,
) -> Tensor:
    """Build the normalized overlap matrix from setup data and positions."""
    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    basis, structure = setup.create_libcint(positions)
    return libcint.overlap(basis, structure)


build_overlap_libcint = build_overlap
