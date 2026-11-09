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
DFT-D3 and DFT-D3S Dispersion Models
====================================

Implementation of the environment-dependent DFT-D3 (`D3Model`) and smooth
element-pair-weighted DFT-D3S (`D3SModel`) dispersion models, following
``dftd/src/dftd/model/d3/orig.f90`` and ``dftd/src/dftd/model/d3/smooth.f90``.
"""
from __future__ import annotations

from typing import ClassVar, NamedTuple

import torch
from tad_mctc import Structure
from tad_mctc.data import resolve_table
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.tree import child, context
from tad_mctc.typing import CountingFunction, TableFunction, Tensor

from .. import defaults, ncoord
from ..cutoff import Cutoff
from ..damping import Damping
from ..data.table import element_table, validate_max_elem
from ..data.wfpair import WFPAIR_D3S
from ..pairvalues import pair_radii
from ..pairs import DensePairs, Pairs
from ..reference.d3 import (
    Reference,
    _default_reference,
    _default_reference_d3s,
)
from .base import (  # noqa: F401 (re-exported for the other models)
    DispersionModel,
    DispersionProperties,
    Param,
    _eval_dispersion2,
    _eval_dispersion3,
    _matrix,
    default_damping,
)
from .c6 import AtomicC6, factored_c6, factored_c6_d3s
from .catalogue import DispersionModelId
from .weights import (
    WeightingFunction,
    gaussian_log_weight,
    weight_references,
    weight_references_d3s,
)

__all__ = [
    "D3Model",
    "D3SModel",
    "Environment",
    "default_damping",
]


class Environment(NamedTuple):
    """
    Per-atom descriptors that the dispersion coefficients depend on.
    """

    cn: Tensor
    """Coordination number of each atom, of the shape of ``numbers``."""

    q: Tensor | None = None
    """Optional atomic partial charges (used by charge-dependent models)."""


def _validate_d3_numbers(
    numbers: Tensor, max_elem: int = defaults.MAX_ELEM_D3
) -> Tensor:
    """
    Validate that atomic numbers are within the supported D3 range
    (``Z == 0`` for padding, ``1 <= Z <= max_elem``, default ``103``).
    """
    return validate_max_elem(numbers, "D3", max_elem)


def _validate_d3s_numbers(
    numbers: Tensor, max_elem: int = defaults.MAX_ELEM_D3S
) -> Tensor:
    """
    Validate that atomic numbers are within the supported D3S range
    (``Z == 0`` for padding, ``1 <= Z <= max_elem``, default ``94``).
    """
    return validate_max_elem(numbers, "D3S", max_elem)


def _cn_d3(
    model: D3Model | D3SModel,
    structure: Structure,
    pairs: PeriodicShifts | NeighborList | None,
) -> Tensor:
    """The D3 coordination number of `structure` with the settings of `model`."""
    cn_model = ncoord.cn_d3.replace(
        count=model.counting_function,
        cutoff=model.cutoff.cn,
        rcov=element_table(model.rcov_table, "rcov_table", structure.positions),
    )
    recompute = model.checkpoint and isinstance(pairs, NeighborList)
    return cn_model(
        structure, pairs, mode="recompute" if recompute else "graph"
    )


class D3Model(DispersionModel):
    """
    A DFT-D3 model as an immutable :class:`~tad_mctc.tree.Node` configuration.

    Parameters
    ----------
    rcov_table : Tensor | TableFunction | None, optional
        Covalent radii per element, of shape ``(119,)``. ``None`` is
        :func:`tad_mctc.data.radii.COV_D3`.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``.
        ``None`` is :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``.
        ``None`` is :func:`tad_dftd.data.R4R2`.
    ref : Reference | None, optional
        Reference C6 coefficients. ``None`` is the default D3 reference on the
        device and dtype of the structure.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model.
    counting_function : CountingFunction, optional
        Counting function of the coordination number.
    weighting_function : WeightingFunction, optional
        Logarithm of the weight of a reference system.
    max_triples : int, optional
        Upper bound on the triples evaluated at once in the three-body term.
    checkpoint : bool, optional
        Recompute each chunk of pairs and triples in the backward pass.
    """

    rcov_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    ref: Reference | None = child(default=None)
    cutoff: Cutoff = context(default_factory=Cutoff)
    counting_function: CountingFunction = context(default=ncoord.exp_count)
    weighting_function: WeightingFunction = context(
        default=gaussian_log_weight
    )
    max_triples: int = context(default=2_000_000)
    checkpoint: bool = context(default=False)

    _model_id: ClassVar[DispersionModelId] = DispersionModelId.D3

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        ref = self._reference(structure)
        structure = self._checked(structure, ref)
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self._environment(structure, pairs.cn)
        w = self._weights(structure, env, ref)
        c6: Tensor | AtomicC6 = self._coefficients(structure, w, ref)
        if isinstance(pairs, DensePairs):
            c6 = _matrix(c6)

        energy = self.dispersion2(
            structure, param, c6, pairs.disp2, damping.two
        )
        if damping.three is not None:
            energy = energy + self.dispersion3(
                structure, param, c6, pairs.disp3, damping.three
            )
        return energy

    def environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None = None,
    ) -> Environment:
        """Compute the D3 coordination number of `structure`."""
        return self._environment(self._checked(structure), pairs)

    def weights(self, structure: Structure, environment: Environment) -> Tensor:
        """Normalized reference weights of shape ``(..., nat, 7)``."""
        ref = self._reference(structure)
        return self._weights(self._checked(structure, ref), environment, ref)

    def coefficients(self, structure: Structure, weights: Tensor) -> AtomicC6:
        """Factored atomic C6 coefficients from reference `weights`."""
        ref = self._reference(structure)
        return self._coefficients(self._checked(structure, ref), weights, ref)

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """Compute D3 model properties (`cn`, dense `c6`, and dense `c8`)."""
        _ = q
        ref = self._reference(structure)
        structure = self._checked(structure, ref)
        if cn is None:
            env = self._environment(
                structure, self._cn_pairs(structure, pairs)
            )
        else:
            env = Environment(cn=cn)
        w = self._weights(structure, env, ref)
        c6 = self._coefficients(structure, w, ref).dense()
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        return DispersionProperties(cn=env.cn, c6=c6, c8=c8)

    def _checked(
        self, structure: Structure, ref: Reference | None = None
    ) -> Structure:
        """`structure` with validated atomic numbers."""
        if ref is None:
            ref = self._reference(structure)
        return structure.replace(
            numbers=_validate_d3_numbers(
                structure.numbers, max_elem=int(ref.cn.shape[-2]) - 1
            )
        )

    def _environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None,
    ) -> Environment:
        return Environment(cn=_cn_d3(self, structure, pairs))

    def _weights(
        self, structure: Structure, environment: Environment, ref: Reference
    ) -> Tensor:
        return weight_references(
            structure.numbers,
            environment.cn,
            ref,
            self.weighting_function,
        )

    def _coefficients(
        self, structure: Structure, weights: Tensor, ref: Reference
    ) -> AtomicC6:
        return factored_c6(structure.numbers, weights, ref)

    def _reference(self, structure: Structure) -> Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference(structure.positions)


class D3SModel(DispersionModel):
    """
    A smooth DFT-D3S dispersion model as an immutable :class:`~tad_mctc.tree.Node`
    configuration, following ``dftd/src/dftd/model/d3/smooth.f90``
    (N. V. Tkachenko and M. Head-Gordon, J. Chem. Theory Comput. 2024, 20,
    9741-9753).

    In D3S, the Gaussian reference weighting factor :math:`k_3` (`wf`) depends
    on the element pair :math:`(Z_A, Z_B)` via :func:`~tad_dftd.data.WFPAIR_D3S`
    instead of being fixed to 4.0, and the elements Fr-Pu (Z=87..94) use the
    original D3 reference systems with which :math:`k_3` was parameterized.

    Parameters
    ----------
    rcov_table : Tensor | TableFunction | None, optional
        Covalent radii per element, of shape ``(119,)``.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``.
    wfpair_table : Tensor | TableFunction | None, optional
        Pairwise weighting factor table of shape ``(95, 95)``. ``None`` is
        :func:`tad_dftd.data.WFPAIR_D3S`.
    ref : Reference | None, optional
        Reference C6 coefficients. ``None`` is the default D3S reference on the
        device and dtype of the structure.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model.
    counting_function : CountingFunction, optional
        Counting function of the coordination number.
    max_triples : int, optional
        Upper bound on the triples evaluated at once in the three-body term.
    checkpoint : bool, optional
        Recompute each chunk of pairs and triples in the backward pass.
    """

    rcov_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    wfpair_table: Tensor | TableFunction | None = child(default=None)
    ref: Reference | None = child(default=None)
    cutoff: Cutoff = context(default_factory=Cutoff)
    counting_function: CountingFunction = context(default=ncoord.exp_count)
    max_triples: int = context(default=2_000_000)
    checkpoint: bool = context(default=False)

    _model_id: ClassVar[DispersionModelId] = DispersionModelId.D3S

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        structure = self._checked(structure)
        ref = self._reference(structure)
        wfpair = self._wfpair(structure)
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self._environment(structure, pairs.cn)
        c6: Tensor | AtomicC6 = self._coefficients(structure, env, ref, wfpair)
        if isinstance(pairs, DensePairs):
            c6 = _matrix(c6)

        energy = self.dispersion2(
            structure, param, c6, pairs.disp2, damping.two
        )
        if damping.three is not None:
            energy = energy + self.dispersion3(
                structure, param, c6, pairs.disp3, damping.three
            )
        return energy

    def environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None = None,
    ) -> Environment:
        """Compute the D3S coordination number of `structure`."""
        return self._environment(self._checked(structure), pairs)

    def weights(
        self,
        structure: Structure,
        environment: Environment,
        elements: Tensor | None = None,
    ) -> Tensor:
        """
        Partner-element-dependent reference weights of shape
        ``(..., nat, nel, 7)``.
        """
        structure = self._checked(structure)
        return weight_references_d3s(
            structure.numbers,
            environment.cn,
            self._reference(structure),
            self._wfpair(structure),
            elements=elements,
        )

    def coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
    ) -> AtomicC6:
        """Factored atomic C6 coefficients for D3S."""
        structure = self._checked(structure)
        return self._coefficients(
            structure,
            environment_or_weights,
            self._reference(structure),
            self._wfpair(structure),
        )

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """Compute D3S model properties (`cn`, dense `c6`, and dense `c8`)."""
        _ = q
        structure = self._checked(structure)
        if cn is None:
            env = self._environment(
                structure, self._cn_pairs(structure, pairs)
            )
        else:
            env = Environment(cn=cn)
        c6 = self._coefficients(
            structure,
            env,
            self._reference(structure),
            self._wfpair(structure),
        ).dense()
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        return DispersionProperties(cn=env.cn, c6=c6, c8=c8)

    def _checked(self, structure: Structure) -> Structure:
        """`structure` with validated atomic numbers."""
        return structure.replace(
            numbers=_validate_d3s_numbers(structure.numbers)
        )

    def _environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None,
    ) -> Environment:
        return Environment(cn=_cn_d3(self, structure, pairs))

    def _coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
        ref: Reference,
        wfpair: Tensor,
    ) -> AtomicC6:
        if isinstance(environment_or_weights, Environment):
            return factored_c6_d3s(
                structure.numbers,
                environment_or_weights.cn,
                ref,
                wfpair,
            )
        return factored_c6_d3s(
            structure.numbers,
            torch.zeros_like(structure.positions[..., 0]),
            ref,
            wfpair,
            weights=environment_or_weights,
        )

    def _reference(self, structure: Structure) -> Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference_d3s(structure.positions)

    def _wfpair(self, structure: Structure) -> Tensor:
        if self.wfpair_table is None:
            return resolve_table(WFPAIR_D3S, structure.positions)
        out = resolve_table(self.wfpair_table, structure.positions)
        if out.shape != (95, 95):
            raise ValueError(
                "'wfpair_table' for D3S must have shape (95, 95), "
                f"got {tuple(out.shape)}."
            )
        return out
