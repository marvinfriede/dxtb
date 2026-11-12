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

from .driver import LibcintIntegralSetup, setup_libcint
from .multipole import (
    build_dipole,
    build_dipole_libcint,
    build_multipole_libcint,
    build_quadrupole,
    build_quadrupole_libcint,
    evaluate_moments,
)
from .overlap import build_overlap, build_overlap_libcint

__all__ = [
    "LibcintIntegralSetup",
    "setup_libcint",
    "build_overlap",
    "build_overlap_libcint",
    "build_dipole",
    "build_dipole_libcint",
    "build_multipole_libcint",
    "build_quadrupole",
    "build_quadrupole_libcint",
    "evaluate_moments",
]
