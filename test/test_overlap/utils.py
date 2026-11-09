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
Utility functions for overlap calculation.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import deflate, pack
from tad_mctc.convert import symmetrize

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch import (
    PytorchIntegralSetup,
    build_overlap,
    setup_integrals,
)
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Literal, Tensor


def overlap_setup(
    numbers: Tensor, par: Param | ParamModule, dd: DD
) -> PytorchIntegralSetup:
    """PyTorch integral setup of a single Structure."""
    pm = par if isinstance(par, ParamModule) else ParamModule(par, **dd)
    ihelp = IndexHelper.from_numbers(numbers, pm)
    return setup_integrals(Basis(numbers, pm, ihelp, **dd))


def calc_overlap(
    numbers: Tensor,
    positions: Tensor,
    par: Param | ParamModule,
    dd: DD,
    uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
    mask: Tensor | None = None,
) -> Tensor:
    """
    Raw (not normalized) PyTorch overlap of a Structure or a padded batch.
    With ``uplo="n"``, the matrix is symmetrized.
    """
    if numbers.ndim == 1:
        s = build_overlap(overlap_setup(numbers, par, dd), positions)
    else:
        matrices = []
        for i, nums in enumerate(numbers):
            nums = deflate(nums)
            if mask is not None:
                pos = torch.masked_select(positions[i], mask[i]).reshape(
                    (-1, 3)
                )
            else:
                pos = positions[i, : nums.shape[-1]]
            matrices.append(build_overlap(overlap_setup(nums, par, dd), pos))
        s = pack(matrices)

    if uplo.casefold() == "n":
        return symmetrize(s, force=False)
    return s
