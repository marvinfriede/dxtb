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
Torch Autodiff DFT-D (tad-dftd)
===============================

Unified, differentiable PyTorch implementation of Grimme-lab DFT-D dispersion
correction models (D1, D2, D3, D3S, D4, D4S, D4Srev, D5).
"""
from __future__ import annotations

from . import (
    cutoff,
    damping,
    data,
    defaults,
    disp,
    exception,
    integrator,
    model,
    ncoord,
    pairs,
    pairvalues,
    param,
    reference,
    threebody,
    twobody,
    typing,
)
from .__version__ import __version__
from .cutoff import Cutoff, smooth_cutoff
from .damping import (
    CSOTwoBody,
    Damping,
    DampingParam,
    FermiTwoBody,
    KoideThreeBody,
    KoideTwoBody,
    ModifiedZeroTwoBody,
    OptimizedPowerTwoBody,
    RationalThreeBody,
    RationalTwoBody,
    ScreenedThreeBody,
    ScreenedTwoBody,
    ZeroProductThreeBody,
    ZeroThreeBodyD3,
    ZeroThreeBodyD4,
    ZeroTwoBody,
    ZTwoBody,
    damping_from_name,
)
from .disp import (
    D3Model,
    D3SModel,
    Environment,
    default_damping,
    dftd,
    dftd3,
    dftd3s,
    dispersion,
    dispersion2,
    dispersion3,
)
from .pairs import DensePairs, Pairs, SparsePairs
from .reference import Reference

__all__ = [
    "CSOTwoBody",
    "Cutoff",
    "D3Model",
    "D3SModel",
    "Damping",
    "DampingParam",
    "DensePairs",
    "Environment",
    "FermiTwoBody",
    "KoideThreeBody",
    "KoideTwoBody",
    "ModifiedZeroTwoBody",
    "OptimizedPowerTwoBody",
    "Pairs",
    "RationalThreeBody",
    "RationalTwoBody",
    "Reference",
    "ScreenedThreeBody",
    "ScreenedTwoBody",
    "SparsePairs",
    "ZTwoBody",
    "ZeroProductThreeBody",
    "ZeroThreeBodyD3",
    "ZeroThreeBodyD4",
    "ZeroTwoBody",
    "__version__",
    "cutoff",
    "damping",
    "damping_from_name",
    "data",
    "default_damping",
    "defaults",
    "dftd",
    "dftd3",
    "dftd3s",
    "disp",
    "dispersion",
    "dispersion2",
    "dispersion3",
    "exception",
    "integrator",
    "model",
    "ncoord",
    "pairs",
    "pairvalues",
    "param",
    "reference",
    "smooth_cutoff",
    "threebody",
    "twobody",
    "typing",
]
