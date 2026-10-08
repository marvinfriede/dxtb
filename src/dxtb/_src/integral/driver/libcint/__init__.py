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
Integral Driver: Libcint
========================

`libcint`-based integral implementations.
"""

from .dipole import DipoleLibcint
from .driver import (
    IntDriverLibcint,
    LibcintCallData,
    LibcintIntegralSetup,
    setup_libcint,
)
from .overlap import OverlapLibcint, build_overlap_libcint
from .quadrupole import QuadrupoleLibcint

__all__ = [
    "DipoleLibcint",
    "IntDriverLibcint",
    "LibcintCallData",
    "LibcintIntegralSetup",
    "OverlapLibcint",
    "QuadrupoleLibcint",
    "build_overlap_libcint",
    "setup_libcint",
]
