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
DFT-D4, DFT-D4S, and DFT-D4Srev Dispersion Models
=================================================

Implementation of the charge-dependent DFT-D4 (`D4Model`), smooth element-pair-
weighted DFT-D4S (`D4SModel`), and revised smooth DFT-D4Srev (`D4SrevModel`)
dispersion models, following ``dftd/src/dftd/model/d4/{orig,smooth,srev}.f90``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, overload

import torch
from tad_mctc import Structure
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.data import resolve_table
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.tools import is_compiling
from tad_mctc.tree import child, context
from tad_mctc.typing import CountingFunction, TableFunction, Tensor

from .. import defaults, ncoord
from ..charge import get_charges
from ..cutoff import Cutoff
from ..damping import Damping, DampingParam, ThreeBodyDamping, TwoBodyDamping
from ..data import GAM, TANH_PARAMS, ZEFF
from ..data.r4r2 import R4R2
from ..data.table import element_table
from ..data.wfpair import WFPAIR_D4S
from ..pairvalues import pair_radii
from ..pairs import DensePairs, Pairs, SparsePairs, TermPairs
from ..reference.d4 import (
    D4Reference,
    _default_reference_d4,
    _default_reference_d4srev,
)
from .base import DispersionModel, DispersionProperties
from .c6 import AtomicC6, factored_c6, factored_c6_d4s
from .catalogue import DispersionModelId
from .d3 import (
    D3Model,
    Environment,
    _eval_dispersion2,
    _eval_dispersion3,
    _matrix,
)
from .weights import weight_references_d4, weight_references_d4s

__all__ = [
    "D4Model",
    "D4SModel",
    "D4SrevModel",
]

Param: TypeAlias = DampingParam | Mapping[str, Any]


def _default_d4_cutoff() -> Cutoff:
    """Default real-space cutoffs for D4, D4S, and D4Srev."""
    return Cutoff.from_model("d4")


def _validate_d4_numbers(numbers: Tensor) -> None:
    """
    Validate that atomic numbers are within the supported D4 range
    (``Z == 0`` for padding, ``1 <= Z <= 103``, or ``112 <= Z <= 118``).
    """
    if is_compiling() or is_functorch_tensor(numbers):
        return
    invalid = (
        (numbers < 0)
        | ((numbers > 103) & (numbers < 112))
        | (numbers > defaults.MAX_ELEM_D4)
    )
    if bool(invalid.any()):
        bad = int(numbers[invalid][0].item())
        raise ValueError(
            f"Unsupported atomic number {bad} for D4 dispersion model "
            "(supported: Z in 1..103 and 112..118)."
        )


def _validate_d4srev_numbers(numbers: Tensor) -> None:
    """
    Validate that atomic numbers are within the supported D4Srev range
    (``Z == 0`` for padding, ``1 <= Z <= 103``; ``dftd/data/tanh.f90`` defines
    tanh charge-scaling parameters only up to ``Z = 103``).
    """
    if is_compiling() or is_functorch_tensor(numbers):
        return
    invalid = (numbers < 0) | (numbers > 103)
    if bool(invalid.any()):
        bad = int(numbers[invalid][0].item())
        raise ValueError(
            f"Unsupported atomic number {bad} for D4Srev dispersion model "
            "(supported: Z in 1..103)."
        )


class D4Model(DispersionModel):
    """
    A DFT-D4 model as an immutable :class:`~tad_mctc.tree.Node` configuration,
    following ``dftd/src/dftd/model/d4/orig.f90``.

    Parameters
    ----------
    ga : Tensor | float, optional
        Maximum charge scaling height :math:`g_a`. Defaults to ``3.0``.
    gc : Tensor | float, optional
        Charge scaling steepness :math:`g_c`. Defaults to ``2.0``.
    wf : Tensor | float, optional
        Gaussian reference coordination number weighting factor :math:`w_f`.
        Defaults to ``6.0``.
    qmodel : str, optional
        Reference charge model (``"eeq"``, ``"eeqbc"``, ``"gfn2"``, or
        ``"gxtb"``). Defaults to ``"eeq"``.
    rcov_table : Tensor | TableFunction | None, optional
        Covalent radii per element, of shape ``(119,)``.
    en_table : Tensor | TableFunction | None, optional
        Pauling electronegativity per element, of shape ``(119,)``.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``.
    zeff_table : Tensor | TableFunction | None, optional
        Effective nuclear charges per element, of shape ``(119,)``.
    gam_table : Tensor | TableFunction | None, optional
        Chemical hardnesses per element, of shape ``(119,)``.
    ref : D4Reference | None, optional
        Reference data table. ``None`` loads the default D4 reference for
        `qmodel`.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model.
    counting_function : CountingFunction, optional
        Counting function of the coordination number.
    max_triples : int, optional
        Upper bound on the triples evaluated at once in the three-body term.
    checkpoint : bool, optional
        Recompute each chunk of pairs and triples in the backward pass.
    """

    ga: Tensor | float = child(default=defaults.D4_GA)
    gc: Tensor | float = child(default=defaults.D4_GC)
    wf: Tensor | float = child(default=defaults.D4_WF)
    qmodel: str = context(default="eeq")
    rcov_table: Tensor | TableFunction | None = child(default=None)
    en_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    zeff_table: Tensor | TableFunction | None = child(default=None)
    gam_table: Tensor | TableFunction | None = child(default=None)
    ref: D4Reference | None = child(default=None)
    cutoff: Cutoff = context(default_factory=_default_d4_cutoff)
    counting_function: CountingFunction = context(default=ncoord.erf_count)
    max_triples: int = context(default=2_000_000)
    checkpoint: bool = context(default=False)

    @property
    def ref_charges(self) -> str:
        """Alias for :attr:`qmodel`."""
        return self.qmodel

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        q: Tensor | None = None,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        _validate_d4_numbers(structure.numbers)
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self.environment(structure, pairs.cn, q=q)
        w = self.weights(structure, env)
        c6: Tensor | AtomicC6 = self.coefficients(structure, w)
        if isinstance(pairs, DensePairs):
            c6 = _matrix(c6)

        energy = self.dispersion2(
            structure, param, c6, pairs.disp2, damping.two
        )
        if damping.three is not None:
            # D4 three-body ATM term is evaluated at neutral atomic charges
            # (charge_dep_3b = False in dftd/disp.f90).
            env3 = Environment(cn=env.cn, q=torch.zeros_like(env.cn))
            w3 = self.weights(structure, env3)
            c6_3: Tensor | AtomicC6 = self.coefficients(structure, w3)
            if isinstance(pairs, DensePairs):
                c6_3 = _matrix(c6_3)
            energy = energy + self.dispersion3(
                structure, param, c6_3, pairs.disp3, damping.three
            )
        return energy

    def environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None = None,
        *,
        q: Tensor | None = None,
        cn: Tensor | None = None,
    ) -> Environment:
        """Compute the D4 coordination number and atomic partial charges."""
        _validate_d4_numbers(structure.numbers)
        if cn is None:
            cn_model = ncoord.cn_d4.replace(
                count=self.counting_function,
                cutoff=self.cutoff.cn,
                rcov=element_table(
                    self.rcov_table, "rcov_table", structure.positions
                ),
                en=element_table(
                    self.en_table, "en_table", structure.positions
                ),
            )
            recompute = self.checkpoint and isinstance(pairs, NeighborList)
            cn = cn_model(
                structure, pairs, mode="recompute" if recompute else "graph"
            )
        if q is None:
            charge = (
                structure.charge
                if structure.charge is not None
                else structure.positions.new_zeros(
                    structure.numbers.shape[:-1]
                )
            )
            q = get_charges(
                structure.numbers,
                structure.positions,
                charge,
                qmod=self.qmodel,
                lattice=structure.lattice,
                periodic=structure.periodic,
                cutoff=self.cutoff.cn_eeq,
            )
        return Environment(cn=cn, q=q)

    @overload
    def weights(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dgwdq: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def weights(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dgwdq: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def weights(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dgwdq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Normalized charge-scaled reference weights of shape ``(..., nat, 7)``."""
        _validate_d4_numbers(structure.numbers)
        q = (
            environment.q
            if environment.q is not None
            else torch.zeros_like(environment.cn)
        )
        zeff_tab = self._zeff(structure)
        gam_tab = self._gam(structure)
        if with_dgwdq:
            return weight_references_d4(
                structure.numbers,
                environment.cn,
                q,
                self._reference(structure),
                wf=self.wf,
                ga=self.ga,
                gc=self.gc,
                is_srev=False,
                zeff_table=zeff_tab,
                gam_table=gam_tab,
                with_dgwdq=True,
            )
        return weight_references_d4(
            structure.numbers,
            environment.cn,
            q,
            self._reference(structure),
            wf=self.wf,
            ga=self.ga,
            gc=self.gc,
            is_srev=False,
            zeff_table=zeff_tab,
            gam_table=gam_tab,
            with_dgwdq=False,
        )

    def coefficients(
        self,
        structure: Structure,
        weights_or_env: Tensor | Environment,
    ) -> AtomicC6:
        """Factored atomic C6 coefficients from reference `weights` or `environment`."""
        _validate_d4_numbers(structure.numbers)
        w = (
            self.weights(structure, weights_or_env)
            if isinstance(weights_or_env, Environment)
            else weights_or_env
        )
        return factored_c6(
            structure.numbers, w, self._reference(structure)
        )

    @overload
    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute dense pairwise :math:`C_6` coefficients ``(..., nat, nat)`` and
        optional partial-charge derivatives ``dc6dq[..., i, j] = d(C6_ij)/d(q_i)``.
        """
        _validate_d4_numbers(structure.numbers)
        ref = self._reference(structure)
        if not with_dc6dq:
            w = self.weights(structure, environment)
            return factored_c6(structure.numbers, w, ref).dense()
        gw, dgwdq = self.weights(structure, environment, with_dgwdq=True)
        c6_f = factored_c6(structure.numbers, gw, ref)
        dc6_f = factored_c6(structure.numbers, dgwdq, ref)
        dc6dq = AtomicC6(
            partial=dc6_f.partial,
            weights=c6_f.weights,
            element=c6_f.element,
        ).dense()
        return c6_f.dense(), dc6dq

    def get_polarizabilities(
        self,
        structure: Structure,
        environment: Environment | None = None,
        *,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute static dipole-dipole (`alphadd`) and quadrupole-quadrupole
        (`alphaqq`) atomic polarizabilities.
        """
        _validate_d4_numbers(structure.numbers)
        if weights is None:
            if environment is None:
                environment = self.environment(structure)
            weights = self.weights(structure, environment)
        ref = self._reference(structure)
        aiw = ref.aiw[structure.numbers]
        adiw = torch.einsum("...ir,...irw->...iw", weights, aiw)
        alphadd = adiw[..., 0]
        r4r2_tab = resolve_table(
            R4R2 if self.r4r2_table is None else self.r4r2_table,
            structure.positions,
        )
        alphaqq = alphadd * r4r2_tab[structure.numbers]
        return alphadd, alphaqq

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """
        Compute D4 model properties (`cn`, `q`, dense `c6`, dense `c8`,
        `alphadd`, and `alphaqq`).
        """
        _validate_d4_numbers(structure.numbers)
        cn_pairs: PeriodicShifts | NeighborList | None = None
        if cn is None:
            if isinstance(pairs, DensePairs):
                pairs = pairs.fill(structure, self.cutoff, three_body=False)
            cn_pairs = pairs.cn if pairs is not None else None
            if cn_pairs is None and structure.lattice is not None:
                cn_pairs = SparsePairs.build(
                    structure, self.cutoff, three_body=False
                ).cn
        env = self.environment(structure, cn_pairs, q=q, cn=cn)
        w = self.weights(structure, env)
        c6 = self.coefficients(structure, w).dense()
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        alphadd, alphaqq = self.get_polarizabilities(structure, weights=w)
        return DispersionProperties(
            cn=env.cn,
            c6=c6,
            c8=c8,
            q=env.q,
            alphadd=alphadd,
            alphaqq=alphaqq,
        )

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

    def _reference(self, structure: Structure) -> D4Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference_d4(
            structure.positions, qmod=self.qmodel, ga=self.ga, gc=self.gc
        )

    def _zeff(self, structure: Structure) -> Tensor:
        if self.zeff_table is None:
            return resolve_table(ZEFF, structure.positions)
        return resolve_table(self.zeff_table, structure.positions)

    def _gam(self, structure: Structure) -> Tensor:
        if self.gam_table is None:
            return resolve_table(GAM, structure.positions)
        return resolve_table(self.gam_table, structure.positions)

    def _prepare(
        self,
        structure: Structure,
        param: Param,
        damping: Damping | None,
        pairs: Pairs | None,
        *,
        model_id: DispersionModelId = DispersionModelId.D4,
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
            model_id=model_id,
        )


class D4SModel(DispersionModel):
    """
    A smooth DFT-D4S dispersion model as an immutable :class:`~tad_mctc.tree.Node`
    configuration, following ``dftd/src/dftd/model/d4/smooth.f90``
    (N. V. Tkachenko and M. Head-Gordon, J. Chem. Theory Comput. 2024, 20,
    9741-9753).
    """

    ga: Tensor | float = child(default=defaults.D4_GA)
    gc: Tensor | float = child(default=defaults.D4_GC)
    qmodel: str = context(default="eeq")
    rcov_table: Tensor | TableFunction | None = child(default=None)
    en_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    wfpair_table: Tensor | TableFunction | None = child(default=None)
    zeff_table: Tensor | TableFunction | None = child(default=None)
    gam_table: Tensor | TableFunction | None = child(default=None)
    ref: D4Reference | None = child(default=None)
    cutoff: Cutoff = context(default_factory=_default_d4_cutoff)
    counting_function: CountingFunction = context(default=ncoord.erf_count)
    max_triples: int = context(default=2_000_000)
    checkpoint: bool = context(default=False)

    @property
    def ref_charges(self) -> str:
        """Alias for :attr:`qmodel`."""
        return self.qmodel

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        q: Tensor | None = None,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        _validate_d4_numbers(structure.numbers)
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self.environment(structure, pairs.cn, q=q)
        c6: Tensor | AtomicC6 = self.coefficients(structure, env)
        if isinstance(pairs, DensePairs):
            c6 = _matrix(c6)

        energy = self.dispersion2(
            structure, param, c6, pairs.disp2, damping.two
        )
        if damping.three is not None:
            env3 = Environment(cn=env.cn, q=torch.zeros_like(env.cn))
            c6_3: Tensor | AtomicC6 = self.coefficients(structure, env3)
            if isinstance(pairs, DensePairs):
                c6_3 = _matrix(c6_3)
            energy = energy + self.dispersion3(
                structure, param, c6_3, pairs.disp3, damping.three
            )
        return energy

    def environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None = None,
        *,
        q: Tensor | None = None,
        cn: Tensor | None = None,
    ) -> Environment:
        """Compute the D4S coordination number and atomic partial charges."""
        return D4Model(
            ga=self.ga,
            gc=self.gc,
            qmodel=self.qmodel,
            rcov_table=self.rcov_table,
            en_table=self.en_table,
            cutoff=self.cutoff,
            counting_function=self.counting_function,
            checkpoint=self.checkpoint,
        ).environment(structure, pairs, q=q, cn=cn)

    @overload
    def weights(
        self,
        structure: Structure,
        environment: Environment,
        elements: Tensor | None = None,
        *,
        with_dgwdq: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def weights(
        self,
        structure: Structure,
        environment: Environment,
        elements: Tensor | None = None,
        *,
        with_dgwdq: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def weights(
        self,
        structure: Structure,
        environment: Environment,
        elements: Tensor | None = None,
        *,
        with_dgwdq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Partner-element-dependent charge-scaled reference weights of shape
        ``(..., nat, nel, 7)``.
        """
        _validate_d4_numbers(structure.numbers)
        q = (
            environment.q
            if environment.q is not None
            else torch.zeros_like(environment.cn)
        )
        zeff_tab = self._zeff(structure)
        gam_tab = self._gam(structure)
        if with_dgwdq:
            return weight_references_d4s(
                structure.numbers,
                environment.cn,
                q,
                self._reference(structure),
                self._wfpair(structure),
                elements=elements,
                ga=self.ga,
                gc=self.gc,
                is_srev=False,
                zeff_table=zeff_tab,
                gam_table=gam_tab,
                with_dgwdq=True,
            )
        return weight_references_d4s(
            structure.numbers,
            environment.cn,
            q,
            self._reference(structure),
            self._wfpair(structure),
            elements=elements,
            ga=self.ga,
            gc=self.gc,
            is_srev=False,
            zeff_table=zeff_tab,
            gam_table=gam_tab,
            with_dgwdq=False,
        )

    def coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
    ) -> AtomicC6:
        """Factored atomic C6 coefficients for D4S."""
        _validate_d4_numbers(structure.numbers)
        ref = self._reference(structure)
        wfpair = self._wfpair(structure)
        if isinstance(environment_or_weights, Environment):
            q = (
                environment_or_weights.q
                if environment_or_weights.q is not None
                else torch.zeros_like(environment_or_weights.cn)
            )
            return factored_c6_d4s(
                structure.numbers,
                environment_or_weights.cn,
                q,
                ref,
                wfpair,
                ga=self.ga,
                gc=self.gc,
                is_srev=False,
                zeff_table=self._zeff(structure),
                gam_table=self._gam(structure),
            )
        z = torch.zeros_like(structure.positions[..., 0])
        return factored_c6_d4s(
            structure.numbers,
            z,
            z,
            ref,
            wfpair,
            weights=environment_or_weights,
        )

    @overload
    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute dense pairwise :math:`C_6` coefficients ``(..., nat, nat)`` and
        optional partial-charge derivatives ``dc6dq[..., i, j] = d(C6_ij)/d(q_i)``.
        """
        _validate_d4_numbers(structure.numbers)
        if not with_dc6dq:
            return self.coefficients(structure, environment).dense()
        gw, dgwdq = self.weights(structure, environment, with_dgwdq=True)
        c6_f = self.coefficients(structure, gw)
        dc6_f = self.coefficients(structure, dgwdq)
        dc6dq = AtomicC6(
            partial=dc6_f.partial,
            weights=c6_f.weights,
            element=c6_f.element,
        ).dense()
        return c6_f.dense(), dc6dq

    def get_polarizabilities(
        self,
        structure: Structure,
        environment: Environment | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute static dipole-dipole (`alphadd`) and quadrupole-quadrupole
        (`alphaqq`) atomic polarizabilities using diagonal element-pair
        weights ``wf = wfpair[Z_i, Z_i]`` (``dftd/model/d4/smooth.f90``).
        """
        _validate_d4_numbers(structure.numbers)
        if environment is None:
            environment = self.environment(structure)
        q = (
            environment.q
            if environment.q is not None
            else torch.zeros_like(environment.cn)
        )
        wfpair = self._wfpair(structure)
        wf_diag = wfpair[structure.numbers, structure.numbers]
        ref = self._reference(structure)
        gw_diag = weight_references_d4(
            structure.numbers,
            environment.cn,
            q,
            ref,
            wf=wf_diag,
            ga=self.ga,
            gc=self.gc,
            is_srev=False,
            zeff_table=self._zeff(structure),
            gam_table=self._gam(structure),
        )
        aiw = ref.aiw[structure.numbers]
        adiw = torch.einsum("...ir,...irw->...iw", gw_diag, aiw)
        alphadd = adiw[..., 0]
        r4r2_tab = resolve_table(
            R4R2 if self.r4r2_table is None else self.r4r2_table,
            structure.positions,
        )
        alphaqq = alphadd * r4r2_tab[structure.numbers]
        return alphadd, alphaqq

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """
        Compute D4S model properties (`cn`, `q`, dense `c6`, dense `c8`,
        `alphadd`, and `alphaqq`).
        """
        _validate_d4_numbers(structure.numbers)
        cn_pairs: PeriodicShifts | NeighborList | None = None
        if cn is None:
            if isinstance(pairs, DensePairs):
                pairs = pairs.fill(structure, self.cutoff, three_body=False)
            cn_pairs = pairs.cn if pairs is not None else None
            if cn_pairs is None and structure.lattice is not None:
                cn_pairs = SparsePairs.build(
                    structure, self.cutoff, three_body=False
                ).cn
        env = self.environment(structure, cn_pairs, q=q, cn=cn)
        c6 = self.coefficients(structure, env).dense()
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        alphadd, alphaqq = self.get_polarizabilities(structure, env)
        return DispersionProperties(
            cn=env.cn,
            c6=c6,
            c8=c8,
            q=env.q,
            alphadd=alphadd,
            alphaqq=alphaqq,
        )

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

    def _reference(self, structure: Structure) -> D4Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference_d4(
            structure.positions, qmod=self.qmodel, ga=self.ga, gc=self.gc
        )

    def _wfpair(self, structure: Structure) -> Tensor:
        if self.wfpair_table is None:
            return resolve_table(WFPAIR_D4S, structure.positions)
        out = resolve_table(self.wfpair_table, structure.positions)
        if out.shape != (119, 119):
            raise ValueError(
                "'wfpair_table' for D4S must have shape (119, 119), "
                f"got {tuple(out.shape)}."
            )
        return out

    def _zeff(self, structure: Structure) -> Tensor:
        if self.zeff_table is None:
            return resolve_table(ZEFF, structure.positions)
        return resolve_table(self.zeff_table, structure.positions)

    def _gam(self, structure: Structure) -> Tensor:
        if self.gam_table is None:
            return resolve_table(GAM, structure.positions)
        return resolve_table(self.gam_table, structure.positions)

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
            model_id=DispersionModelId.D4S,
        )


class D4SrevModel(DispersionModel):
    """
    Revised smooth DFT-D4Srev dispersion model as an immutable
    :class:`~tad_mctc.tree.Node` configuration, following
    ``dftd/src/dftd/model/d4/srev.f90``.

    Compared to `D4SModel`, `D4SrevModel` uses:

    1. Error-function coordination number without electronegativity weighting
       (:data:`~tad_dftd.ncoord.cn_d4srev`, ``kcn=7.5``, ``cutoff=30.0``).
    2. Hyperbolic-tangent charge scaling :func:`~tad_dftd.model.zeta_tanh`
       with element-specific parameters :func:`~tad_dftd.data.TANH_PARAMS`.
    3. Revised reference systems (:meth:`~tad_dftd.reference.D4Reference.load_d4srev`).
    """

    qmodel: str = context(default="eeq")
    rcov_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    wfpair_table: Tensor | TableFunction | None = child(default=None)
    tanh_table: Tensor | TableFunction | None = child(default=None)
    ref: D4Reference | None = child(default=None)
    cutoff: Cutoff = context(default_factory=_default_d4_cutoff)
    counting_function: CountingFunction = context(default=ncoord.erf_count)
    max_triples: int = context(default=2_000_000)
    checkpoint: bool = context(default=False)

    @property
    def ref_charges(self) -> str:
        """Alias for :attr:`qmodel`."""
        return self.qmodel

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        q: Tensor | None = None,
        damping: Damping | None = None,
        pairs: Pairs | None = None,
    ) -> Tensor:
        _validate_d4srev_numbers(structure.numbers)
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self.environment(structure, pairs.cn, q=q)
        c6: Tensor | AtomicC6 = self.coefficients(structure, env)
        if isinstance(pairs, DensePairs):
            c6 = _matrix(c6)

        energy = self.dispersion2(
            structure, param, c6, pairs.disp2, damping.two
        )
        if damping.three is not None:
            env3 = Environment(cn=env.cn, q=torch.zeros_like(env.cn))
            c6_3: Tensor | AtomicC6 = self.coefficients(structure, env3)
            if isinstance(pairs, DensePairs):
                c6_3 = _matrix(c6_3)
            energy = energy + self.dispersion3(
                structure, param, c6_3, pairs.disp3, damping.three
            )
        return energy

    def environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None = None,
        *,
        q: Tensor | None = None,
        cn: Tensor | None = None,
    ) -> Environment:
        """Compute the D4Srev coordination number and atomic partial charges."""
        _validate_d4srev_numbers(structure.numbers)
        if cn is None:
            cn_model = ncoord.cn_d4srev.replace(
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
        if q is None:
            charge = (
                structure.charge
                if structure.charge is not None
                else structure.positions.new_zeros(
                    structure.numbers.shape[:-1]
                )
            )
            q = get_charges(
                structure.numbers,
                structure.positions,
                charge,
                qmod=self.qmodel,
                lattice=structure.lattice,
                periodic=structure.periodic,
                cutoff=self.cutoff.cn_eeq,
            )
        return Environment(cn=cn, q=q)

    @overload
    def weights(
        self,
        structure: Structure,
        environment: Environment,
        elements: Tensor | None = None,
        *,
        with_dgwdq: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def weights(
        self,
        structure: Structure,
        environment: Environment,
        elements: Tensor | None = None,
        *,
        with_dgwdq: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def weights(
        self,
        structure: Structure,
        environment: Environment,
        elements: Tensor | None = None,
        *,
        with_dgwdq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Partner-element-dependent charge-scaled reference weights of shape
        ``(..., nat, nel, 7)`` for D4Srev.
        """
        _validate_d4srev_numbers(structure.numbers)
        q = (
            environment.q
            if environment.q is not None
            else torch.zeros_like(environment.cn)
        )
        tanh_tab = self._tanh(structure)
        if with_dgwdq:
            return weight_references_d4s(
                structure.numbers,
                environment.cn,
                q,
                self._reference(structure),
                self._wfpair(structure),
                elements=elements,
                is_srev=True,
                tanh_table=tanh_tab,
                with_dgwdq=True,
            )
        return weight_references_d4s(
            structure.numbers,
            environment.cn,
            q,
            self._reference(structure),
            self._wfpair(structure),
            elements=elements,
            is_srev=True,
            tanh_table=tanh_tab,
            with_dgwdq=False,
        )

    def coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
    ) -> AtomicC6:
        """Factored atomic C6 coefficients for D4Srev."""
        _validate_d4srev_numbers(structure.numbers)
        ref = self._reference(structure)
        wfpair = self._wfpair(structure)
        if isinstance(environment_or_weights, Environment):
            q = (
                environment_or_weights.q
                if environment_or_weights.q is not None
                else torch.zeros_like(environment_or_weights.cn)
            )
            return factored_c6_d4s(
                structure.numbers,
                environment_or_weights.cn,
                q,
                ref,
                wfpair,
                is_srev=True,
                tanh_table=self._tanh(structure),
            )
        z = torch.zeros_like(structure.positions[..., 0])
        return factored_c6_d4s(
            structure.numbers,
            z,
            z,
            ref,
            wfpair,
            weights=environment_or_weights,
        )

    @overload
    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def get_atomic_c6(
        self,
        structure: Structure,
        environment: Environment,
        *,
        with_dc6dq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute dense pairwise :math:`C_6` coefficients ``(..., nat, nat)`` and
        optional partial-charge derivatives ``dc6dq[..., i, j] = d(C6_ij)/d(q_i)``.
        """
        _validate_d4srev_numbers(structure.numbers)
        if not with_dc6dq:
            return self.coefficients(structure, environment).dense()
        gw, dgwdq = self.weights(structure, environment, with_dgwdq=True)
        c6_f = self.coefficients(structure, gw)
        dc6_f = self.coefficients(structure, dgwdq)
        dc6dq = AtomicC6(
            partial=dc6_f.partial,
            weights=c6_f.weights,
            element=c6_f.element,
        ).dense()
        return c6_f.dense(), dc6dq

    def get_polarizabilities(
        self,
        structure: Structure,
        environment: Environment | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute static dipole-dipole (`alphadd`) and quadrupole-quadrupole
        (`alphaqq`) atomic polarizabilities for D4Srev.
        """
        _validate_d4srev_numbers(structure.numbers)
        if environment is None:
            environment = self.environment(structure)
        q = (
            environment.q
            if environment.q is not None
            else torch.zeros_like(environment.cn)
        )
        wfpair = self._wfpair(structure)
        wf_diag = wfpair[structure.numbers, structure.numbers]
        ref = self._reference(structure)
        gw_diag = weight_references_d4(
            structure.numbers,
            environment.cn,
            q,
            ref,
            wf=wf_diag,
            is_srev=True,
            tanh_table=self._tanh(structure),
        )
        aiw = ref.aiw[structure.numbers]
        adiw = torch.einsum("...ir,...irw->...iw", gw_diag, aiw)
        alphadd = adiw[..., 0]
        r4r2_tab = resolve_table(
            R4R2 if self.r4r2_table is None else self.r4r2_table,
            structure.positions,
        )
        alphaqq = alphadd * r4r2_tab[structure.numbers]
        return alphadd, alphaqq

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """
        Compute D4Srev model properties (`cn`, `q`, dense `c6`, dense `c8`,
        `alphadd`, and `alphaqq`).
        """
        _validate_d4srev_numbers(structure.numbers)
        cn_pairs: PeriodicShifts | NeighborList | None = None
        if cn is None:
            if isinstance(pairs, DensePairs):
                pairs = pairs.fill(structure, self.cutoff, three_body=False)
            cn_pairs = pairs.cn if pairs is not None else None
            if cn_pairs is None and structure.lattice is not None:
                cn_pairs = SparsePairs.build(
                    structure, self.cutoff, three_body=False
                ).cn
        env = self.environment(structure, cn_pairs, q=q, cn=cn)
        c6 = self.coefficients(structure, env).dense()
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        alphadd, alphaqq = self.get_polarizabilities(structure, env)
        return DispersionProperties(
            cn=env.cn,
            c6=c6,
            c8=c8,
            q=env.q,
            alphadd=alphadd,
            alphaqq=alphaqq,
        )

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

    def _reference(self, structure: Structure) -> D4Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference_d4srev(structure.positions, qmod=self.qmodel)

    def _wfpair(self, structure: Structure) -> Tensor:
        if self.wfpair_table is None:
            return resolve_table(WFPAIR_D4S, structure.positions)
        out = resolve_table(self.wfpair_table, structure.positions)
        if out.shape != (119, 119):
            raise ValueError(
                "'wfpair_table' for D4Srev must have shape (119, 119), "
                f"got {tuple(out.shape)}."
            )
        return out

    def _tanh(self, structure: Structure) -> Tensor:
        if self.tanh_table is None:
            return resolve_table(TANH_PARAMS, structure.positions)
        return resolve_table(self.tanh_table, structure.positions)

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
            model_id=DispersionModelId.D4SREV,
        )
