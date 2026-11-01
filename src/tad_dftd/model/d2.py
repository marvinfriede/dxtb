# This file is part of tad-dftd.
#
# SPDX-Identifier: LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
"""
DFT-D2 Dispersion Model
=======================

Implementation of the DFT-D2 dispersion model (`D2Model`) with fixed atomic
:math:`C_6` coefficients combined by the geometric mean, following
``dftd/src/dftd/model/d2/orig.f90`` and ``dftd/src/dftd/model/fixed.f90``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
from tad_mctc import Structure, storch
from tad_mctc.data import resolve_table
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.tree import child, context
from tad_mctc.typing import TableFunction, Tensor

from .. import defaults
from ..cutoff import Cutoff
from ..damping import (
    Damping,
    DampingParam,
    ThreeBodyDamping,
    TwoBodyDamping,
    ZeroThreeBodyD4,
    as_damping_param,
)
from ..data.d2 import D2_C6, D2_RADII_PAIRWISE
from ..pairs import DensePairs, Pairs, TermPairs
from .base import DispersionModel, DispersionProperties
from .c6 import AtomicC6
from .catalogue import DispersionModelId, validate_model_damping
from .d3 import (
    Environment,
    _eval_dispersion2,
    _eval_dispersion3,
    default_damping,
)

__all__ = ["D2Model"]

Param: TypeAlias = DampingParam | Mapping[str, Any]


def _default_d2_cutoff() -> Cutoff:
    return Cutoff(
        disp2=defaults.D2_DISP2_CUTOFF,
        disp3=defaults.D3_DISP3_CUTOFF,
        cn=defaults.D3_CN_CUTOFF,
        width2=defaults.D3_DISP2_WIDTH,
        width3=defaults.D3_DISP3_WIDTH,
    )


def _unwrap_functorch(x: Tensor) -> Tensor:
    while torch._C._functorch.is_functorch_wrapped_tensor(x):
        x = torch._C._functorch.get_unwrapped(x)
    return x


def _validate_d2_numbers(numbers: Tensor, c6_table: Tensor) -> None:
    if torch.compiler.is_compiling():
        return
    raw_num = _unwrap_functorch(numbers)
    raw_c6 = _unwrap_functorch(c6_table)
    max_z = min(defaults.MAX_ELEM_D2, int(raw_c6.shape[0]) - 1)
    if bool((raw_num < 0).any()) or bool((raw_num > max_z).any()):
        raise ValueError(
            "Structure contains unsupported element for D2 dispersion model "
            f"(only H-Rn, Z <= {defaults.MAX_ELEM_D2}, are supported)."
        )
    invalid = (raw_num > 0) & (raw_c6[raw_num] <= 0.0)
    if bool(invalid.any()):
        raise ValueError(
            "Structure contains unsupported element for D2 dispersion model "
            f"(only H-Rn, Z <= {defaults.MAX_ELEM_D2}, are supported)."
        )


class D2Model(DispersionModel):
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

    c6_table: Tensor | TableFunction | None = child(default=None)
    rdamp_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    cutoff: Cutoff = context(default_factory=_default_d2_cutoff)
    max_triples: int = context(default=2_000_000)
    checkpoint: bool = context(default=False)

    def _c6_table(self, like: Tensor) -> Tensor:
        if self.c6_table is None:
            return resolve_table(D2_C6, like)
        out = resolve_table(self.c6_table, like)
        if out.shape != (119,):
            raise ValueError(
                f"'c6_table' must have shape (119,), got {tuple(out.shape)}."
            )
        return out

    def _rdamp_table(self) -> Tensor | TableFunction:
        return D2_RADII_PAIRWISE if self.rdamp_table is None else self.rdamp_table

    def environment(
        self, structure: Structure, pairs: TermPairs = None
    ) -> Environment:
        """Return a zero coordination-number environment (D2 is environment-independent)."""
        _ = pairs
        _validate_d2_numbers(
            structure.numbers, self._c6_table(structure.positions)
        )
        return Environment(
            cn=torch.zeros_like(structure.positions[..., 0]),
            q=None,
        )

    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment | None = None,
    ) -> Tensor:
        """Return the pairwise :math:`C_6` matrix for `structure`."""
        return self.coefficients(structure, environment)

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
        numbers, positions = structure.numbers, structure.positions
        c6_tab = self._c6_table(positions)
        _validate_d2_numbers(numbers, c6_tab)

        ci = c6_tab[numbers]
        ci_i = ci.unsqueeze(-1)
        ci_j = ci.unsqueeze(-2)
        valid = (numbers.unsqueeze(-1) > 0) & (numbers.unsqueeze(-2) > 0)
        prod = torch.where(valid, ci_i * ci_j, torch.zeros_like(ci_i * ci_j))
        return storch.safe_sqrt(prod)

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """Compute dispersion properties (`cn`, `c6`, `c8`) for `structure`."""
        _ = pairs, q
        c6 = self.coefficients(structure)
        cn_val = (
            torch.zeros_like(structure.positions[..., 0]) if cn is None else cn
        )
        c8 = torch.zeros_like(c6)
        return DispersionProperties(
            cn=cn_val,
            c6=c6,
            c8=c8,
            q=None,
            alphadd=None,
            alphaqq=None,
        )

    def dispersion2(
        self,
        structure: Structure,
        param: Param,
        c6: Tensor | AtomicC6,
        pairs: PeriodicShifts | NeighborList | None = None,
        *,
        damping: TwoBodyDamping | None = None,
    ) -> Tensor:
        """Two-body dispersion energy per atom of `structure`."""
        param = as_damping_param(param)
        if damping is None:
            damping = default_damping(param, model="d2").two
        return _eval_dispersion2(
            structure,
            param,
            c6,
            pairs,
            damping,
            rvdw_table=self.rvdw_table,
            r4r2_table=self.r4r2_table,
            rdamp_table=self._rdamp_table(),
            cutoff=self.cutoff,
            checkpoint=self.checkpoint,
        )

    def dispersion3(
        self,
        structure: Structure,
        param: Param,
        c6: Tensor | AtomicC6,
        pairs: TermPairs = None,
        *,
        damping: ThreeBodyDamping | None = None,
    ) -> Tensor:
        """Three-body Axilrod-Teller-Muto dispersion energy per atom (expert option)."""
        param = as_damping_param(param)
        if damping is None:
            damping = ZeroThreeBodyD4()
        return _eval_dispersion3(
            structure,
            param,
            c6,
            pairs,
            damping,
            rvdw_table=self.rvdw_table,
            r4r2_table=self.r4r2_table,
            rdamp_table=self._rdamp_table(),
            cutoff=self.cutoff,
            max_triples=self.max_triples,
            checkpoint=self.checkpoint,
        )

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        """Total DFT-D2 dispersion energy per atom of `structure`."""
        param = as_damping_param(param)
        if damping is None:
            if param.damping is not None:
                validate_model_damping("d2", param.damping)
            damping = default_damping(param, model="d2")
        else:
            validate_model_damping("d2", damping.two)

        if pairs is None:
            pairs = DensePairs.build(
                structure,
                self.cutoff,
                three_body=damping.three is not None,
            )

        c6 = self.coefficients(structure)
        energy = self.dispersion2(
            structure, param, c6, pairs.disp2, damping=damping.two
        )

        if damping.three is not None:
            if pairs.disp3 is None and structure.lattice is not None:
                raise ValueError(
                    "'pairs' was built with 'three_body=False', but 'damping' "
                    "has a three-body term."
                )
            energy = energy + self.dispersion3(
                structure, param, c6, pairs.disp3, damping=damping.three
            )

        return energy
