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
xTB Hamiltonians: GFN2-xTB
==========================

The GFN2-xTB Hamiltonian.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch

from dxtb import IndexHelper
from dxtb._src.components.interactions import Potential
from dxtb._src.param.base import Param
from dxtb._src.param.module import ParamModule
from dxtb._src.typing import Any, Tensor

from .base import BaseHamiltonian
from .h0 import gather_hscale, setup_h0

__all__ = ["GFN2Hamiltonian"]


class GFN2Hamiltonian(BaseHamiltonian):
    """
    The GFN2-xTB Hamiltonian.
    """

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        setup = kwargs.pop("setup", None)
        has_custom_cn = "cn" in kwargs
        cn = kwargs.pop("cn", None)
        if setup is None:
            setup = setup_h0(
                numbers, par, ihelp, device=device, dtype=dtype, cn=cn
            )
            if has_custom_cn and cn is None:
                setup = replace(setup, cn=None)
        elif has_custom_cn:
            setup = replace(setup, cn=cn)
        super().__init__(numbers, par, ihelp, device, dtype, setup=setup)

        # coordination number function
        if self.cn is None and not (has_custom_cn and cn is None):
            # pylint: disable=import-outside-toplevel
            from dxtb._src.ncoord import cn_d3, gfn2_count

            self.cn = partial(cn_d3, counting_function=gfn2_count)

    def _get_hscale(self, par: ParamModule) -> Tensor:
        if par.is_none("hamiltonian"):
            raise RuntimeError("No Hamiltonian specified.")
        return gather_hscale(
            "gfn2", self.unique, self.ihelp, self.valence, par, self.dd
        )

    def get_gradient(
        self,
        positions: Tensor,
        overlap: Tensor,
        doverlap: Tensor,
        pmat: Tensor,
        wmat: Tensor,
        pot: Potential,
        cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("GFN2 not implemented yet.")
