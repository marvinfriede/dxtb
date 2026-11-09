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
DFT-D1 Dispersion Model
=======================

Implementation of the original DFT-D1 dispersion model (`D1Model`) with fixed
atomic :math:`C_6` coefficients combined by the harmonic mean, following
``dftd/src/dftd/model/d1/orig.f90`` and ``dftd/src/dftd/model/fixed.f90``.

The machinery shared with the D2 model (`_FixedC6Model`) lives here, too.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

import torch
from tad_mctc import Structure
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
from ..data.d1 import D1_C6, D1_RADII_PAIRWISE
from ..data.table import guard_numbers
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

__all__ = ["D1Model"]

Param: TypeAlias = DampingParam | Mapping[str, Any]


def _default_d1_cutoff() -> Cutoff:
    return Cutoff.from_model("d1")


def _validate_fixed_c6_numbers(
    numbers: Tensor, c6_table: Tensor, max_elem: int, message: str
) -> Tensor:
    """Atomic numbers of a fixed-C6 model: in ``0..max_elem`` and, if not
    padding, with a positive entry in `c6_table`."""
    max_z = min(max_elem, int(c6_table.shape[0]) - 1)
    oob = (numbers < 0) | (numbers > max_z)
    safe_num = torch.where(oob, torch.zeros_like(numbers), numbers)
    invalid = oob | ((safe_num > 0) & (c6_table[safe_num] <= 0.0))
    return guard_numbers(numbers, invalid, message)


def _validate_d1_numbers(numbers: Tensor, c6_table: Tensor) -> Tensor:
    return _validate_fixed_c6_numbers(
        numbers,
        c6_table,
        defaults.MAX_ELEM_D1,
        "Structure contains unsupported element for D1 dispersion model "
        "(only H and C-Ne, Z in {1, 6..10}, are supported).",
    )


class _FixedC6Model(DispersionModel):
    """
    Base of the D1 and D2 models: fixed atomic :math:`C_6` coefficients,
    without any dependence on the environment.

    A subclass sets the class variables ``_name``, ``_default_c6``,
    ``_default_rdamp`` and ``_validate_numbers``, redeclares the `cutoff`
    field with its default, and implements :meth:`coefficients`.
    """

    _name: ClassVar[str]
    """Name of the model, e.g. ``"d1"``."""

    _default_c6: ClassVar[TableFunction]
    """Default atomic :math:`C_6` table."""

    _default_rdamp: ClassVar[TableFunction]
    """Default pairwise damping radii."""

    _validate_numbers: ClassVar[Any]
    """``(numbers, c6_table) -> numbers``, validating the atomic numbers."""

    c6_table: Tensor | TableFunction | None = child(default=None)
    rdamp_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    cutoff: Cutoff = context()
    max_triples: int = context(default=2_000_000)
    checkpoint: bool = context(default=False)

    def _c6_table(self, like: Tensor) -> Tensor:
        if self.c6_table is None:
            return resolve_table(self._default_c6, like)
        out = resolve_table(self.c6_table, like)
        if out.shape != (119,):
            raise ValueError(
                f"'c6_table' must have shape (119,), got {tuple(out.shape)}."
            )
        return out

    def _rdamp_table(self) -> Tensor | TableFunction:
        return self._default_rdamp if self.rdamp_table is None else self.rdamp_table

    def environment(
        self, structure: Structure, pairs: TermPairs = None
    ) -> Environment:
        """Return a zero coordination-number environment (the model is environment-independent)."""
        _ = pairs
        numbers = self._validate_numbers(
            structure.numbers, self._c6_table(structure.positions)
        )
        return Environment(
            cn=torch.zeros_like(structure.positions[..., 0])
            + (numbers[..., :1] * 0).to(structure.positions.dtype),
            q=None,
        )

    def coefficients(
        self,
        structure: Structure,
        environment: Environment | None = None,
    ) -> Tensor:
        """Pairwise :math:`C_6` matrix of shape ``(..., nat, nat)``."""
        raise NotImplementedError

    def _pair_c6(self, structure: Structure) -> tuple[Tensor, Tensor, Tensor]:
        """Atomic :math:`C_6` of the atoms as a column and a row, ``(..., nat,
        1)`` and ``(..., 1, nat)``, and the validated atomic numbers."""
        c6_tab = self._c6_table(structure.positions)
        numbers = self._validate_numbers(structure.numbers, c6_tab)
        ci = c6_tab[numbers]
        return ci.unsqueeze(-1), ci.unsqueeze(-2), numbers

    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment | None = None,
    ) -> Tensor:
        """Return the pairwise :math:`C_6` matrix for `structure`."""
        return self.coefficients(structure, environment)

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
            damping = default_damping(param, model=self._name).two
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
        """Total dispersion energy per atom of `structure`."""
        param = as_damping_param(param)
        if damping is None:
            if param.damping is not None:
                validate_model_damping(self._name, param.damping)
            damping = default_damping(param, model=self._name)
        else:
            validate_model_damping(self._name, damping.two)

        structure = structure.replace(
            numbers=self._validate_numbers(
                structure.numbers, self._c6_table(structure.positions)
            )
        )
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


class D1Model(_FixedC6Model):
    """
    DFT-D1 dispersion model with fixed atomic :math:`C_6` coefficients
    combined via the harmonic mean:

    .. math::

        C_6^{AB} = \\frac{2 C_6^A C_6^B}{C_6^A + C_6^B}.

    Parameters
    ----------
    c6_table : Tensor | TableFunction | None, optional
        Atomic :math:`C_6` coefficients per element, of shape ``(119,)``.
        ``None`` is :func:`tad_dftd.data.D1_C6`.
    rdamp_table : Tensor | TableFunction | None, optional
        Pairwise damping radii per element pair, of shape ``(119, 119)``.
        ``None`` is :func:`tad_dftd.data.D1_RADII_PAIRWISE`.
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

    model_id = DispersionModelId.D1

    _name = "d1"
    _default_c6 = staticmethod(D1_C6)
    _default_rdamp = staticmethod(D1_RADII_PAIRWISE)
    _validate_numbers = staticmethod(_validate_d1_numbers)

    cutoff: Cutoff = context(default_factory=_default_d1_cutoff)

    def coefficients(
        self,
        structure: Structure,
        environment: Environment | None = None,
    ) -> Tensor:
        """
        Pairwise :math:`C_6` matrix of shape ``(..., nat, nat)`` combined by
        the harmonic mean.
        """
        _ = environment
        ci_i, ci_j, numbers = self._pair_c6(structure)

        denom = ci_i + ci_j
        valid = (
            (numbers.unsqueeze(-1) > 0)
            & (numbers.unsqueeze(-2) > 0)
            & (denom > 0.0)
        )
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        return torch.where(
            valid,
            2.0 * ci_i * ci_j / safe_denom,
            torch.zeros_like(denom),
        )
