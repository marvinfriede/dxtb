# This file is part of tad-dftd.
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
Frequency integrators for Casimir-Polder dispersion coefficients.
"""
from __future__ import annotations

from .gauss_legendre import (
    GAUSS_LEGENDRE_NODES_10,
    GAUSS_LEGENDRE_WEIGHTS_10,
    GaussLegendreIntegrator,
    gauss_legendre_1d,
)
from .trapezoid import DEFAULT_TRAPZ_FREQUENCIES, TrapezoidIntegrator

__all__ = [
    "DEFAULT_TRAPZ_FREQUENCIES",
    "GAUSS_LEGENDRE_NODES_10",
    "GAUSS_LEGENDRE_WEIGHTS_10",
    "GaussLegendreIntegrator",
    "TrapezoidIntegrator",
    "gauss_legendre_1d",
]
