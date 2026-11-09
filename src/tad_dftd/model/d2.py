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
DFT-D2 Dispersion Model
=======================

Implementation of the DFT-D2 dispersion model (`D2Model`) with fixed atomic
:math:`C_6` coefficients combined by the geometric mean, following
``dftd/src/dftd/model/d2/orig.f90`` and ``dftd/src/dftd/model/fixed.f90``.
"""
from __future__ import annotations

import torch
from tad_mctc import Structure, storch
from tad_mctc.tree import context
from tad_mctc.typing import Tensor

from .. import defaults
from ..cutoff import Cutoff
from ..data.d2 import D2_C6, D2_RADII_PAIRWISE
from .catalogue import DispersionModelId
from .d1 import _FixedC6Model, _validate_fixed_c6_numbers
from .d3 import Environment

__all__ = ["D2Model"]


def _default_d2_cutoff() -> Cutoff:
    return Cutoff.from_model("d2")


def _validate_d2_numbers(numbers: Tensor, c6_table: Tensor) -> Tensor:
    return _validate_fixed_c6_numbers(
        numbers,
        c6_table,
        defaults.MAX_ELEM_D2,
        "Structure contains unsupported element for D2 dispersion model "
        f"(only H-Rn, Z <= {defaults.MAX_ELEM_D2}, are supported).",
    )


class D2Model(_FixedC6Model):
    """
    DFT-D2 dispersion model with fixed atomic :math:`C_6` coefficients
    combined via the geometric mean:

    .. math::

        C_6^{AB} = \\sqrt{C_6^A C_6^B}.

    Parameters
    ----------
    c6_table : Tensor | TableFunction | None, optional
        Atomic :math:`C_6` coefficients per element, of shape ``(119,)``.
        ``None`` is :func:`tad_dftd.data.D2_C6`.
    rdamp_table : Tensor | TableFunction | None, optional
        Pairwise damping radii per element pair, of shape ``(119, 119)``.
        ``None`` is :func:`tad_dftd.data.D2_RADII_PAIRWISE`.
    rvdw_table : Tensor | TableFunction | None, optional
        Van-der-Waals radii per element pair.
    r4r2_table : Tensor | TableFunction | None, optional
        Expectation values :math:`\\langle r^4 \\rangle / \\langle r^2 \\rangle`.
    cutoff : Cutoff, optional
        Real-space cutoffs.
    max_triples : int, optional
        Upper bound on triples evaluated at once in the optional 3-body term.
    checkpoint : bool, optional
        Recompute chunks in backward pass.
    """

    model_id = DispersionModelId.D2

    _name = "d2"
    _default_c6 = staticmethod(D2_C6)
    _default_rdamp = staticmethod(D2_RADII_PAIRWISE)
    _validate_numbers = staticmethod(_validate_d2_numbers)

    cutoff: Cutoff = context(default_factory=_default_d2_cutoff)

    def coefficients(
        self,
        structure: Structure,
        environment: Environment | None = None,
    ) -> Tensor:
        """
        Pairwise :math:`C_6` matrix of shape ``(..., nat, nat)`` combined by
        the geometric mean :math:`\\sqrt{C_6^A C_6^B}`.
        """
        _ = environment
        ci_i, ci_j, numbers = self._pair_c6(structure)

        valid = (numbers.unsqueeze(-1) > 0) & (numbers.unsqueeze(-2) > 0)
        prod = ci_i * ci_j
        prod = torch.where(valid, prod, torch.zeros_like(prod))
        return storch.safe_sqrt(prod)
