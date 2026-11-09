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
Dispersion Model Base Classes
=============================

Abstract base class and property containers for all DFT-D dispersion models.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import torch
from tad_mctc import Structure
from tad_mctc.autograd import is_functorch_tensor, is_vmapped
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.tree import Node
from tad_mctc.typing import TableFunction, Tensor

from ..cutoff import Cutoff
from ..damping import (
    Damping,
    DampingParam,
    ThreeBodyDamping,
    TwoBodyDamping,
    as_damping_param,
    damping_from_name,
)
from ..pairs import DensePairs, Pairs, SparsePairs, TermPairs, TripleList
from ..threebody import (
    dispersion_atm,
    dispersion_atm_periodic,
    dispersion_atm_sparse,
)
from ..twobody import _dispersion2_dense, dispersion2_sparse
from .c6 import AtomicC6
from .catalogue import (
    DispersionModelId,
    get_model_metadata,
    validate_model_damping,
)

__all__ = [
    "DispersionModel",
    "DispersionProperties",
    "ModelProperties",
    "default_damping",
]

Param: TypeAlias = DampingParam | Mapping[str, Any]


@dataclass(frozen=True)
class DispersionProperties:
    """
    Physical properties evaluated by a dispersion model (matching
    ``get_properties`` in Grimme-lab ``dftd/disp.f90``).

    Attributes
    ----------
    cn : Tensor
        Atomic coordination numbers of shape ``(..., nat)`` (zeros for models
        without coordination numbers).
    q : Tensor | None
        Atomic partial charges of shape ``(..., nat)``, or ``None`` for
        charge-independent models.
    c6 : Tensor
        Pairwise :math:`C_6` dispersion coefficients of shape ``(..., nat, nat)``.
    c8 : Tensor | None
        Pairwise :math:`C_8` dispersion coefficients of shape ``(..., nat, nat)``.
    alphadd : Tensor | None
        Static dipole-dipole polarizabilities :math:`\alpha_{dd}(0)` of shape
        ``(..., nat)``, or ``None`` if not available for the model.
    alphaqq : Tensor | None
        Static quadrupole-quadrupole polarizabilities :math:`\alpha_{qq}(0)` of
        shape ``(..., nat)``, or ``None`` if not available for the model.
    """

    cn: Tensor
    c6: Tensor
    c8: Tensor | None = None
    q: Tensor | None = None
    alphadd: Tensor | None = None
    alphaqq: Tensor | None = None


ModelProperties = DispersionProperties


def default_damping(param: DampingParam, *, model: str = "d3") -> Damping:
    """
    The damping named by ``param.damping`` (the default two-body damping of
    `model` if unset), with the default three-body damping of `model` if `s9`
    is set and not a non-trainable zero.

    A differentiable, functorch-traced, or compiled tensor `s9` always gets the
    three-body term (even at zero) so that the choice is static under
    ``torch.compile`` and ``vmap`` and the derivative with respect to `s9`
    equals the three-body energy.
    """
    s9 = param.s9
    if isinstance(s9, Tensor):
        three_body = (
            s9.requires_grad
            or is_functorch_tensor(s9)
            or torch.compiler.is_compiling()
            or bool(s9 != 0.0)
        )
    else:
        three_body = s9 is not None and s9 != 0.0
    m = model.lower().strip()
    return damping_from_name(
        param.damping or get_model_metadata(m).default_damping_2b,
        three_body,
        model=m,
    )


def _matrix(c6: Tensor | AtomicC6) -> Tensor:
    """The ``(..., nat, nat)`` C6 matrix required by dense terms."""
    return c6.dense() if isinstance(c6, AtomicC6) else c6


def _eval_dispersion2(
    structure: Structure,
    param: DampingParam,
    c6: Tensor | AtomicC6,
    pairs: PeriodicShifts | NeighborList | None,
    damping: TwoBodyDamping,
    *,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    cutoff: Cutoff | Tensor | float = Cutoff(),
    checkpoint: bool = False,
    as_matrix: bool = False,
) -> Tensor:
    if isinstance(cutoff, Cutoff):
        cut: float | Tensor = cutoff.disp2
        width = cutoff.width2
    else:
        cut = cutoff
        width = 0.0

    if isinstance(pairs, NeighborList):
        return dispersion2_sparse(
            structure,
            param,
            c6,
            pairs,
            damping=damping,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            rdamp_table=rdamp_table,
            cutoff=float(cut),
            width=width,
            checkpoint=checkpoint,
        )

    return _dispersion2_dense(
        structure,
        param,
        _matrix(c6),
        pairs,
        damping=damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        rdamp_table=rdamp_table,
        rvdw=rvdw,
        r4r2=r4r2,
        cutoff=cut,
        width=width,
        as_matrix=as_matrix,
    )


def _eval_dispersion3(
    structure: Structure,
    param: DampingParam,
    c6: Tensor | AtomicC6,
    pairs: TermPairs,
    damping: ThreeBodyDamping,
    *,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    rdamp_table: Tensor | TableFunction | None = None,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    cutoff: Cutoff | Tensor | float = Cutoff(),
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    if isinstance(cutoff, Cutoff):
        cut: float | Tensor = cutoff.disp3
        width = cutoff.width3
    else:
        cut = cutoff
        width = 0.0
    if isinstance(pairs, (NeighborList, TripleList)):
        return dispersion_atm_sparse(
            structure,
            c6,
            param,
            pairs,
            damping=damping,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            rdamp_table=rdamp_table,
            cutoff=float(cut),
            width=width,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )

    if isinstance(pairs, PeriodicShifts):
        return dispersion_atm_periodic(
            structure,
            _matrix(c6),
            param,
            pairs,
            damping=damping,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            rdamp_table=rdamp_table,
            cutoff=float(cut),
            width=width,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )

    return dispersion_atm(
        structure,
        _matrix(c6),
        param,
        damping=damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        rdamp_table=rdamp_table,
        rvdw=rvdw,
        r4r2=r4r2,
        cutoff=cut,
        width=width,
    )


class DispersionModel(Node):
    """
    Base class for DFT-D dispersion models.

    The models D3, D3S, D4, D4S and D4Srev share the evaluation of the
    two-body and three-body terms (`dispersion2`, `dispersion3`) and the setup
    of damping and pair lists (`_prepare`) defined here. They provide the
    fields read below and their `_model_id`.
    """

    if TYPE_CHECKING:
        rvdw_table: Tensor | TableFunction | None
        r4r2_table: Tensor | TableFunction | None
        cutoff: Cutoff
        max_triples: int
        checkpoint: bool

    _model_id: ClassVar[DispersionModelId]

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """
        Evaluate coordination numbers, partial charges, pairwise :math:`C_6` and
        :math:`C_8` coefficients, and static polarizabilities for ``structure``.
        """
        raise NotImplementedError

    def dispersion2(
        self,
        structure: Structure,
        param: DampingParam,
        c6: Tensor | AtomicC6,
        pairs: PeriodicShifts | NeighborList | None,
        damping: TwoBodyDamping,
    ) -> Tensor:
        """Two-body dispersion energy from the atomic `c6` coefficients."""
        return _eval_dispersion2(
            structure,
            param,
            c6,
            pairs,
            damping,
            rvdw_table=self.rvdw_table,
            r4r2_table=self.r4r2_table,
            cutoff=self.cutoff,
            checkpoint=self.checkpoint,
        )

    def dispersion3(
        self,
        structure: Structure,
        param: DampingParam,
        c6: Tensor | AtomicC6,
        pairs: TermPairs,
        damping: ThreeBodyDamping,
    ) -> Tensor:
        """Three-body (Axilrod-Teller-Muto) energy from the atomic `c6`."""
        return _eval_dispersion3(
            structure,
            param,
            c6,
            pairs,
            damping,
            rvdw_table=self.rvdw_table,
            r4r2_table=self.r4r2_table,
            cutoff=self.cutoff,
            max_triples=self.max_triples,
            checkpoint=self.checkpoint,
        )

    def _cn_pairs(
        self,
        structure: Structure,
        pairs: Pairs | None,
        cn: Tensor | None = None,
    ) -> PeriodicShifts | NeighborList | None:
        """
        The pairs for the coordination number, if it is not given as `cn`: from
        `pairs` if they hold it, else built for a periodic structure.
        """
        if cn is not None:
            return None
        if isinstance(pairs, DensePairs):
            pairs = pairs.fill(structure, self.cutoff, three_body=False)
        cn_pairs = pairs.cn if pairs is not None else None
        if cn_pairs is None and structure.lattice is not None:
            cn_pairs = SparsePairs.build(
                structure, self.cutoff, three_body=False
            ).cn
        return cn_pairs

    def _prepare(
        self,
        structure: Structure,
        param: Param,
        damping: Damping | None,
        pairs: Pairs | None,
    ) -> tuple[DampingParam, Damping, Pairs]:
        param = as_damping_param(param)
        if damping is None:
            if param.damping is not None:
                validate_model_damping(self._model_id, param.damping)
            damping = default_damping(
                param, model=self._model_id.name.lower()
            )
        three_body = damping.three is not None

        if pairs is None:
            if (
                structure.lattice is None
                or is_vmapped(structure.positions)
                or is_vmapped(structure.numbers)
                or is_vmapped(structure.lattice)
            ):
                pairs = DensePairs()
            else:
                pairs = SparsePairs.build(
                    structure,
                    self.cutoff,
                    three_body=three_body,
                    chunk_size=self.max_triples,
                )
        if isinstance(pairs, DensePairs):
            pairs = pairs.fill(structure, self.cutoff, three_body=three_body)

        if (
            isinstance(pairs, SparsePairs)
            and three_body
            and pairs.disp3 is None
        ):
            raise ValueError(
                "The damping has a three-body term, but 'pairs' were "
                "built without it ('three_body=False')."
            )
        return param, damping, pairs
