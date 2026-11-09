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
Element-wise agreement of the overlap of all pytorch drivers with libcint.

Both the pytorch drivers and tad-libcint 0.4 follow the CCA ordering of tblite
(spherical components in ascending m, with y, z, x for p-orbitals).

Guards the double precision of the cartesian-to-spherical transformation
matrices and the exact normalization of the contracted shells; either would
show up as deviations of up to ~5e-9 for d shells (SiH4).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dxtb import GFN1_XTB as par
from dxtb._src.constants.labels import (
    INTDRIVER_LIBCINT,
    INTDRIVER_PYTORCH,
)
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.pytorch.impls.trafo import TRAFO
from dxtb._src.typing import DD
from dxtb.integrals.wrappers import overlap

from ..conftest import DEVICE
from .samples import samples


def test_trafo_is_double() -> None:
    """Irrational coefficients (sqrt(3), ...) must not be truncated to the
    default dtype at import."""
    for trafo in TRAFO:
        assert trafo.dtype == np.float64

    assert TRAFO[2][1, 4].item() == 3.0**0.5


@pytest.mark.skipif(
    has_libcint is False, reason="libcint interface not installed"
)
@pytest.mark.parametrize(
    "name", ["H2", "LiH", "CH4", "NH3", "SiH4", "LYS_xao_dist"]
)
def test_overlap_matches_libcint(name: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    s_lib = overlap(numbers, positions, par, driver=INTDRIVER_LIBCINT).to(
        DEVICE
    )
    s_pt = overlap(numbers, positions, par, driver=INTDRIVER_PYTORCH)

    assert torch.allclose(s_pt, s_lib, atol=1e-12, rtol=0.0)
