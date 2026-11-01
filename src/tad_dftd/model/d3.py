# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
"""
DFT-D3 and DFT-D3S Dispersion Models
====================================

Implementation of the environment-dependent DFT-D3 (`D3Model`) and smooth
element-pair-weighted DFT-D3S (`D3SModel`) dispersion models, following
``dftd/src/dftd/model/d3/orig.f90`` and ``dftd/src/dftd/model/d3/smooth.f90``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, TypeAlias

import torch
from tad_mctc import Structure
from tad_mctc.autograd import is_functorch_tensor, is_vmapped
from tad_mctc.data import resolve_table
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import CountingFunction, TableFunction, Tensor

from .. import defaults, ncoord
from ..cutoff import Cutoff
from ..damping import (
    Damping,
    DampingParam,
    ThreeBodyDamping,
    TwoBodyDamping,
    as_damping_param,
    damping_from_name,
)
from ..data.table import element_table, guard_numbers
from ..data.wfpair import WFPAIR_D3S
from ..pairvalues import pair_radii
from ..pairs import DensePairs, Pairs, SparsePairs, TermPairs
from ..reference.d3 import (
    Reference,
    _default_reference,
    _default_reference_d3s,
)
from ..threebody import (
    dispersion_atm,
    dispersion_atm_periodic,
    dispersion_atm_sparse,
)
from ..twobody import _dispersion2_dense, dispersion2_sparse
from .base import DispersionModel, DispersionProperties
from .c6 import AtomicC6, factored_c6, factored_c6_d3s
from .catalogue import DispersionModelId, validate_model_damping
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

Param: TypeAlias = DampingParam | Mapping[str, Any]


class Environment(NamedTuple):
    """
    Per-atom descriptors that the dispersion coefficients depend on.
    """

    cn: Tensor
    """Coordination number of each atom, of the shape of ``numbers``."""

    q: Tensor | None = None
    """Optional atomic partial charges (used by charge-dependent models)."""


def default_damping(param: DampingParam, *, model: str = "d3") -> Damping:
    """
    The damping named by ``param.damping`` (rational damping if unset, or Fermi
    damping for D1/D2), with the default three-body damping of `model` if `s9`
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
    def_2b = "fermi" if m in ("d1", "d2") else "rational"
    return damping_from_name(param.damping or def_2b, three_body, model=m)


def _validate_d3_numbers(
    numbers: Tensor, max_elem: int = defaults.MAX_ELEM_D3
) -> Tensor:
    """
    Validate that atomic numbers are within the supported D3 range
    (``Z == 0`` for padding, ``1 <= Z <= max_elem``, default ``103``).
    """
    invalid = (numbers < 0) | (numbers > max_elem)
    return guard_numbers(
        numbers,
        invalid,
        f"Unsupported atomic number {{bad}} for D3 dispersion model "
        f"(supported: Z in 1..{max_elem}).",
        format_bad=True,
    )


def _validate_d3s_numbers(
    numbers: Tensor, max_elem: int = defaults.MAX_ELEM_D3S
) -> Tensor:
    """
    Validate that atomic numbers are within the supported D3S range
    (``Z == 0`` for padding, ``1 <= Z <= max_elem``, default ``94``).
    """
    invalid = (numbers < 0) | (numbers > max_elem)
    return guard_numbers(
        numbers,
        invalid,
        f"Unsupported atomic number {{bad}} for D3S dispersion model "
        f"(supported: Z in 1..{max_elem}).",
        format_bad=True,
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
    from tad_mctc.neighbor.triples import TripleList

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

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        structure = structure.replace(
            numbers=_validate_d3_numbers(
                structure.numbers,
                max_elem=int(self._reference(structure).cn.shape[-2]) - 1,
            )
        )
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self.environment(structure, pairs.cn)
        w = self.weights(structure, env)
        c6: Tensor | AtomicC6 = self.coefficients(structure, w)
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
        structure = structure.replace(
            numbers=_validate_d3_numbers(
                structure.numbers,
                max_elem=int(self._reference(structure).cn.shape[-2]) - 1,
            )
        )
        cn_model = ncoord.cn_d3.replace(
            count=self.counting_function,
            cutoff=self.cutoff.cn,
            rcov=element_table(
                self.rcov_table, "rcov_table", structure.positions
            ),
        )
        recompute = self.checkpoint and isinstance(pairs, NeighborList)
        cn = cn_model(
            structure, pairs, mode="recompute" if recompute else "graph"
        )
        return Environment(cn=cn)

    def weights(self, structure: Structure, environment: Environment) -> Tensor:
        """Normalized reference weights of shape ``(..., nat, 7)``."""
        ref = self._reference(structure)
        numbers = _validate_d3_numbers(
            structure.numbers, max_elem=int(ref.cn.shape[-2]) - 1
        )
        return weight_references(
            numbers,
            environment.cn,
            ref,
            self.weighting_function,
        )

    def coefficients(self, structure: Structure, weights: Tensor) -> AtomicC6:
        """Factored atomic C6 coefficients from reference `weights`."""
        ref = self._reference(structure)
        numbers = _validate_d3_numbers(
            structure.numbers, max_elem=int(ref.cn.shape[-2]) - 1
        )
        return factored_c6(numbers, weights, ref)

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
    ) -> DispersionProperties:
        """Compute D3 model properties (`cn`, dense `c6`, and dense `c8`)."""
        structure = structure.replace(
            numbers=_validate_d3_numbers(
                structure.numbers,
                max_elem=int(self._reference(structure).cn.shape[-2]) - 1,
            )
        )
        if cn is None:
            if isinstance(pairs, DensePairs):
                pairs = pairs.fill(structure, self.cutoff, three_body=False)
            cn_pairs = pairs.cn if pairs is not None else None
            if cn_pairs is None and structure.lattice is not None:
                cn_pairs = SparsePairs.build(
                    structure, self.cutoff, three_body=False
                ).cn
            env = self.environment(structure, cn_pairs)
        else:
            env = Environment(cn=cn)
        w = self.weights(structure, env)
        c6 = self.coefficients(structure, w).dense()
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        return DispersionProperties(cn=env.cn, c6=c6, c8=c8)

    def dispersion2(
        self,
        structure: Structure,
        param: DampingParam,
        c6: Tensor | AtomicC6,
        pairs: PeriodicShifts | NeighborList | None,
        damping: TwoBodyDamping,
    ) -> Tensor:
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

    def _reference(self, structure: Structure) -> Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference(structure.positions)

    def _prepare(
        self,
        structure: Structure,
        param: Param,
        damping: Damping | None,
        pairs: Pairs | None,
        *,
        model_id: DispersionModelId = DispersionModelId.D3,
    ) -> tuple[DampingParam, Damping, Pairs]:
        param = as_damping_param(param)
        if damping is None:
            if param.damping is not None:
                validate_model_damping(model_id, param.damping)
            damping = default_damping(param, model=model_id.name.lower())
        three_body = damping.three is not None

        if pairs is None:
            if (
                structure.lattice is None
                or is_vmapped(structure.positions)
                or is_vmapped(structure.numbers)
                or (
                    structure.lattice is not None
                    and is_vmapped(structure.lattice)
                )
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

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        structure = structure.replace(
            numbers=_validate_d3s_numbers(structure.numbers)
        )
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self.environment(structure, pairs.cn)
        c6: Tensor | AtomicC6 = self.coefficients(structure, env)
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
        structure = structure.replace(
            numbers=_validate_d3s_numbers(structure.numbers)
        )
        cn_model = ncoord.cn_d3.replace(
            count=self.counting_function,
            cutoff=self.cutoff.cn,
            rcov=element_table(
                self.rcov_table, "rcov_table", structure.positions
            ),
        )
        recompute = self.checkpoint and isinstance(pairs, NeighborList)
        cn = cn_model(
            structure, pairs, mode="recompute" if recompute else "graph"
        )
        return Environment(cn=cn)

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
        numbers = _validate_d3s_numbers(structure.numbers)
        return weight_references_d3s(
            numbers,
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
        numbers = _validate_d3s_numbers(structure.numbers)
        ref = self._reference(structure)
        wfpair = self._wfpair(structure)
        if isinstance(environment_or_weights, Environment):
            return factored_c6_d3s(
                numbers,
                environment_or_weights.cn,
                ref,
                wfpair,
            )
        return factored_c6_d3s(
            numbers,
            torch.zeros_like(structure.positions[..., 0]),
            ref,
            wfpair,
            weights=environment_or_weights,
        )

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
    ) -> DispersionProperties:
        """Compute D3S model properties (`cn`, dense `c6`, and dense `c8`)."""
        structure = structure.replace(
            numbers=_validate_d3s_numbers(structure.numbers)
        )
        if cn is None:
            if isinstance(pairs, DensePairs):
                pairs = pairs.fill(structure, self.cutoff, three_body=False)
            cn_pairs = pairs.cn if pairs is not None else None
            if cn_pairs is None and structure.lattice is not None:
                cn_pairs = SparsePairs.build(
                    structure, self.cutoff, three_body=False
                ).cn
            env = self.environment(structure, cn_pairs)
        else:
            env = Environment(cn=cn)
        c6 = self.coefficients(structure, env).dense()
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        return DispersionProperties(cn=env.cn, c6=c6, c8=c8)

    def dispersion2(
        self,
        structure: Structure,
        param: DampingParam,
        c6: Tensor | AtomicC6,
        pairs: PeriodicShifts | NeighborList | None,
        damping: TwoBodyDamping,
    ) -> Tensor:
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

    def _prepare(
        self,
        structure: Structure,
        param: Param,
        damping: Damping | None,
        pairs: Pairs | None,
    ) -> tuple[DampingParam, Damping, Pairs]:
        return D3Model(
            cutoff=self.cutoff,
            max_triples=self.max_triples,
            checkpoint=self.checkpoint,
        )._prepare(
            structure,
            param,
            damping,
            pairs,
            model_id=DispersionModelId.D3S,
        )
