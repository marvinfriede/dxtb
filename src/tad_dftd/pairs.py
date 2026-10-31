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
Pairs
=====

How the pairs of each term of a dispersion energy are enumerated within its
cutoff (see :class:`~tad_dftd.cutoff.Cutoff`): of the coordination number
(``cn``), the two-body term (``disp2``), the three-body term (``disp3``), and
optionally the EEQ charge-model coordination number (``cn_eeq``).
"""
from __future__ import annotations

import contextlib
from typing import TypeAlias

import torch
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor.images import PeriodicShifts, build_periodic_shifts
from tad_mctc.neighbor.list import NeighborList, StageHook, build_neighborlists
from tad_mctc.neighbor.triples import TripleList
from tad_mctc.tree import Node, child
from tad_mctc.typing import Tensor

from .cutoff import Cutoff

__all__ = [
    "DensePairs",
    "Pairs",
    "SparsePairs",
    "TermPairs",
    "TripleList",
]


TermPairs: TypeAlias = PeriodicShifts | NeighborList | TripleList | None
"""The pairs of one term: all pairs of a molecule (``None``), the shift table
of a cell, a neighbour list or, for the three-body term, the triangles of a
list."""


def _no_stage(label: str) -> contextlib.AbstractContextManager[object]:
    """The default stage hook: runs each step of a build as it is."""
    del label
    return contextlib.nullcontext()


class DensePairs(Node):
    """
    All pairs of atoms of each term: for a periodic cell, every periodic
    image within the term's cutoff, from its shift table; for a molecule,
    nothing (all fields ``None``).
    """

    cn: PeriodicShifts | None = child(default=None)
    disp2: PeriodicShifts | None = child(default=None)
    disp3: PeriodicShifts | None = child(default=None)
    cn_eeq: PeriodicShifts | None = child(default=None)

    @classmethod
    def build(
        cls,
        structure: Structure,
        cutoff: Cutoff,
        *,
        three_body: bool = True,
        with_cn_eeq: bool = False,
        stage: StageHook = _no_stage,
    ) -> DensePairs:
        """
        The shift table of each term of `structure`, at its own cutoff.
        """
        return cls().fill(
            structure,
            cutoff,
            three_body=three_body,
            with_cn_eeq=with_cn_eeq,
            stage=stage,
        )

    def fill(
        self,
        structure: Structure,
        cutoff: Cutoff,
        *,
        three_body: bool = True,
        with_cn_eeq: bool = False,
        stage: StageHook = _no_stage,
    ) -> DensePairs:
        """
        These pairs, with the shift table of every term of a cell that has
        none built at its cutoff, see :meth:`build`.
        """
        if structure.lattice is None:
            return self

        assert structure.periodic is not None

        def shifts(term: str, reach: float) -> PeriodicShifts:
            assert structure.lattice is not None
            assert structure.periodic is not None
            with stage(f"periodic shifts ({term})"):
                return build_periodic_shifts(
                    structure.lattice, structure.periodic, reach
                )

        missing: dict[str, PeriodicShifts] = {}
        if self.cn is None:
            missing["cn"] = shifts("cn", cutoff.cn)
        if self.disp2 is None:
            missing["disp2"] = shifts("disp2", cutoff.disp2)
        if three_body and self.disp3 is None:
            missing["disp3"] = shifts("disp3", cutoff.disp3)
        if with_cn_eeq and self.cn_eeq is None:
            missing["cn_eeq"] = shifts("cn_eeq", cutoff.cn_eeq)
        return self.replace(**missing) if missing else self

    def stale(self, structure: Structure) -> Tensor:
        """
        Never: whether the shifts still cover a cutoff is checked by each
        term, which raises if not.
        """
        return structure.positions.new_zeros((), dtype=torch.bool)


class SparsePairs(Node):
    """
    A neighbour list per term, and for the three-body term the triangles of
    its list (see :class:`~tad_mctc.neighbor.triples.TripleList`), which
    take memory linear in the number of atoms.
    """

    cn: NeighborList = child()
    disp2: NeighborList = child()
    disp3: TripleList | None = child(default=None)
    cn_eeq: NeighborList | None = child(default=None)

    @classmethod
    def build(
        cls,
        structure: Structure,
        cutoff: Cutoff,
        *,
        three_body: bool = True,
        with_cn_eeq: bool = False,
        skin: float = 0.0,
        chunk_size: int = 2_000_000,
        stage: StageHook = _no_stage,
    ) -> SparsePairs:
        """
        The lists of the terms of `structure`, each at its own cutoff, from
        one search at the largest of them, and the triangles of the list of
        the three-body term.
        """
        lists = cls.search(
            structure,
            cutoff,
            three_body=three_body,
            with_cn_eeq=with_cn_eeq,
            skin=skin,
            stage=stage,
        )

        disp3 = None
        if three_body:
            with stage("triples"):
                disp3 = TripleList.from_neighborlist(
                    lists["disp3"], chunk_size=chunk_size
                )

        return cls(
            cn=lists["cn"],
            disp2=lists["disp2"],
            disp3=disp3,
            cn_eeq=lists.get("cn_eeq"),
        )

    @staticmethod
    def search(
        structure: Structure,
        cutoff: Cutoff,
        *,
        three_body: bool = True,
        with_cn_eeq: bool = False,
        skin: float = 0.0,
        stage: StageHook = _no_stage,
    ) -> dict[str, NeighborList]:
        """
        The neighbour lists of :meth:`build`, without the triangles of the
        three-body term.
        """

        def search_stage(
            label: str,
        ) -> contextlib.AbstractContextManager[object]:
            return stage(f"nlist: {label}")

        names: tuple[str, ...] = ("cn", "disp2")
        cutoffs: tuple[float, ...] = (cutoff.cn, cutoff.disp2)
        if three_body:
            names += ("disp3",)
            cutoffs += (cutoff.disp3,)
        if with_cn_eeq:
            names += ("cn_eeq",)
            cutoffs += (cutoff.cn_eeq,)
        lists = build_neighborlists(
            structure, cutoffs, skin=skin, stage=search_stage
        )
        return dict(zip(names, lists))

    def lists(self) -> dict[str, NeighborList]:
        """
        The neighbour list of each term, keyed by its name.
        """
        lists = {"cn": self.cn, "disp2": self.disp2}
        if self.disp3 is not None:
            lists["disp3"] = self.disp3.pairs
        if self.cn_eeq is not None:
            lists["cn_eeq"] = self.cn_eeq
        return lists

    def stale(self, structure: Structure) -> Tensor:
        """
        Whether any of the lists should be rebuilt.
        """
        stale = self.cn.stale(structure) | self.disp2.stale(structure)
        if self.disp3 is not None:
            stale = stale | self.disp3.stale(structure)
        if self.cn_eeq is not None:
            stale = stale | self.cn_eeq.stale(structure)
        return stale


Pairs: TypeAlias = DensePairs | SparsePairs
"""How the pairs of every term are enumerated, dense or sparse."""
