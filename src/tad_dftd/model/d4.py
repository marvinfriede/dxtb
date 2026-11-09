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

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    overload,
)

import torch
from tad_mctc import Structure
from tad_mctc.data import resolve_table
from tad_mctc.math import einsum
from tad_mctc.neighbor.images import PeriodicShifts
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.tree import child, context
from tad_mctc.typing import CountingFunction, TableFunction, Tensor

from .. import defaults, ncoord
from ..charge import get_charges
from ..cutoff import Cutoff
from ..damping import (
    Damping,
    DampingParam,
    RationalTwoBody,
    TwoBodyDamping,
)
from ..data import GAM, TANH_PARAMS, ZEFF
from ..data.r4r2 import R4R2
from ..data.table import element_table, guard_numbers, validate_max_elem
from ..data.wfpair import WFPAIR_D4S
from ..pairvalues import pair_radii
from ..pairs import DensePairs, Pairs
from ..reference.d4 import (
    D4Reference,
    _default_reference_d4,
    _default_reference_d4srev,
)
from .base import (
    DispersionModel,
    DispersionProperties,
    Param,
    _eval_dispersion2,
    _matrix,
)
from .c6 import AtomicC6, factored_c6, factored_c6_d4s
from .catalogue import DispersionModelId
from .d3 import Environment
from .weights import weight_references_d4, weight_references_d4s

__all__ = [
    "D4Model",
    "D4SModel",
    "D4SrevModel",
    "d4sc_atom_energy",
    "d4sc_atom_potential",
    "d4sc_dispersion_matrix",
]


def _default_d4_cutoff() -> Cutoff:
    """Default real-space cutoffs for D4, D4S, and D4Srev."""
    return Cutoff.from_model("d4")


def _validate_d4_numbers(numbers: Tensor) -> Tensor:
    """
    Validate that atomic numbers are within the supported D4 range
    (``Z == 0`` for padding, ``1 <= Z <= 103``, or ``112 <= Z <= 118``).
    """
    invalid = (
        (numbers < 0)
        | ((numbers > 103) & (numbers < 112))
        | (numbers > defaults.MAX_ELEM_D4)
    )
    return guard_numbers(
        numbers,
        invalid,
        "Unsupported atomic number {bad} for D4 dispersion model "
        "(supported: Z in 1..103 and 112..118).",
        format_bad=True,
    )


def _validate_d4srev_numbers(numbers: Tensor) -> Tensor:
    """
    Validate that atomic numbers are within the supported D4Srev range
    (``Z == 0`` for padding, ``1 <= Z <= 103``; ``dftd/data/tanh.f90`` defines
    tanh charge-scaling parameters only up to ``Z = 103``).
    """
    return validate_max_elem(numbers, "D4Srev", defaults.MAX_ELEM_D4SREV)


def _eeq_charges(
    structure: Structure, qmodel: str, cutoff: Cutoff
) -> Tensor:
    """EEQ atomic partial charges of `structure` (neutral if it has no charge)."""
    charge = (
        structure.charge
        if structure.charge is not None
        else structure.positions.new_zeros(structure.numbers.shape[:-1])
    )
    return get_charges(
        structure.numbers,
        structure.positions,
        charge,
        qmod=qmodel,
        lattice=structure.lattice,
        periodic=structure.periodic,
        cutoff=cutoff.cn_eeq,
    )


def _charges(environment: Environment) -> Tensor:
    """The charges of `environment`, zero if it has none."""
    if environment.q is not None:
        return environment.q
    return torch.zeros_like(environment.cn)


def _zeff_gam(model: D4Model | D4SModel, structure: Structure) -> tuple[Tensor, Tensor]:
    """Effective nuclear charges and chemical hardnesses of `model`."""
    pos = structure.positions
    return (
        resolve_table(ZEFF if model.zeff_table is None else model.zeff_table, pos),
        resolve_table(GAM if model.gam_table is None else model.gam_table, pos),
    )


class _D4Base(DispersionModel):
    """
    What D4, D4S and D4Srev share: the evaluation, the environment and the
    structure of the public methods. The subclasses provide the weights
    (`_weights`), the coefficients (`_coefficients`) and the polarizability
    weights (`_alpha_weights`); the private methods expect `structure` with
    validated atomic numbers and the reference `ref` resolved by the caller,
    so that a call builds the reference once and validates the numbers once.
    """

    if TYPE_CHECKING:
        qmodel: str
        rcov_table: Tensor | TableFunction | None
        en_table: Tensor | TableFunction | None
        counting_function: CountingFunction
        ref: D4Reference | None
        ga: Tensor | float
        gc: Tensor | float

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
        ref = self._reference(structure)
        structure = self._checked(structure)
        param, damping, pairs = self._prepare(structure, param, damping, pairs)

        env = self._environment(structure, pairs.cn, q=q)
        c6: Tensor | AtomicC6 = self._coefficients(structure, env, ref)
        if isinstance(pairs, DensePairs):
            c6 = _matrix(c6)

        energy = self.dispersion2(
            structure, param, c6, pairs.disp2, damping.two
        )
        if damping.three is not None:
            # The three-body ATM term is evaluated at neutral atomic charges
            # (charge_dep_3b = False in dftd/disp.f90).
            env3 = Environment(cn=env.cn, q=torch.zeros_like(env.cn))
            c6_3: Tensor | AtomicC6 = self._coefficients(structure, env3, ref)
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
        """Compute the coordination number and atomic partial charges."""
        return self._environment(
            self._checked(structure), pairs, q=q, cn=cn
        )

    def coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
    ) -> AtomicC6:
        """
        Factored atomic C6 coefficients from an `environment` or from reference
        weights.
        """
        ref = self._reference(structure)
        return self._coefficients(
            self._checked(structure), environment_or_weights, ref
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
        ref = self._reference(structure)
        structure = self._checked(structure)
        if not with_dc6dq:
            return self._coefficients(structure, environment, ref).dense()
        gw, dgwdq = self._weights(structure, environment, ref, with_dgwdq=True)
        c6_f = self._coefficients(structure, gw, ref)
        dc6_f = self._coefficients(structure, dgwdq, ref)
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
        (`alphaqq`) atomic polarizabilities, from the reference `weights` of
        the atoms or of the `environment`.
        """
        ref = self._reference(structure)
        structure = self._checked(structure)
        if weights is None:
            if environment is None:
                environment = self._environment(structure, None)
            weights = self._alpha_weights(structure, environment, ref)
        return self._alpha(structure, weights, ref)

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """
        Compute the model properties (`cn`, `q`, dense `c6`, dense `c8`,
        `alphadd`, and `alphaqq`).
        """
        ref = self._reference(structure)
        structure = self._checked(structure)
        env = self._environment(
            structure, self._cn_pairs(structure, pairs, cn), q=q, cn=cn
        )
        c6, alphadd, alphaqq = self._c6_alpha(structure, env, ref)
        radii = pair_radii(structure, self.rvdw_table, self.r4r2_table)
        c8 = c6 * radii.qq
        return DispersionProperties(
            cn=env.cn,
            c6=c6,
            c8=c8,
            q=env.q,
            alphadd=alphadd,
            alphaqq=alphaqq,
        )

    def _validate_numbers(self, numbers: Tensor) -> Tensor:
        return _validate_d4_numbers(numbers)

    def _checked(self, structure: Structure) -> Structure:
        """`structure` with validated atomic numbers."""
        return structure.replace(
            numbers=self._validate_numbers(structure.numbers)
        )

    def _coordination_number(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None,
    ) -> Tensor:
        cn_model = ncoord.cn_d4.replace(
            count=self.counting_function,
            cutoff=self.cutoff.cn,
            rcov=element_table(
                self.rcov_table, "rcov_table", structure.positions
            ),
            en=element_table(self.en_table, "en_table", structure.positions),
        )
        recompute = self.checkpoint and isinstance(pairs, NeighborList)
        return cn_model(
            structure, pairs, mode="recompute" if recompute else "graph"
        )

    def _environment(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None,
        *,
        q: Tensor | None = None,
        cn: Tensor | None = None,
    ) -> Environment:
        if cn is None:
            cn = self._coordination_number(structure, pairs)
        if q is None:
            q = _eeq_charges(structure, self.qmodel, self.cutoff)
        return Environment(cn=cn, q=q)

    def _weights(
        self,
        structure: Structure,
        environment: Environment,
        ref: D4Reference,
        *,
        with_dgwdq: bool = False,
    ) -> Any:
        raise NotImplementedError

    def _coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
        ref: D4Reference,
    ) -> AtomicC6:
        raise NotImplementedError

    def _alpha_weights(
        self, structure: Structure, environment: Environment, ref: D4Reference
    ) -> Tensor:
        """The reference weights that give the polarizabilities."""
        raise NotImplementedError

    def _c6_alpha(
        self, structure: Structure, environment: Environment, ref: D4Reference
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Dense C6 of the pairs and the polarizabilities `alphadd`, `alphaqq`."""
        c6 = self._coefficients(structure, environment, ref).dense()
        weights = self._alpha_weights(structure, environment, ref)
        return (c6, *self._alpha(structure, weights, ref))

    def _alpha(
        self, structure: Structure, weights: Tensor, ref: D4Reference
    ) -> tuple[Tensor, Tensor]:
        aiw = ref.aiw[structure.numbers]
        adiw = torch.einsum("...ir,...irw->...iw", weights, aiw)
        alphadd = adiw[..., 0]
        r4r2_tab = resolve_table(
            R4R2 if self.r4r2_table is None else self.r4r2_table,
            structure.positions,
        )
        alphaqq = alphadd * r4r2_tab[structure.numbers]
        return alphadd, alphaqq

    def _reference(self, structure: Structure) -> D4Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference_d4(
            structure.positions, qmod=self.qmodel, ga=self.ga, gc=self.gc
        )


class D4Model(_D4Base):
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
        Reference charge model (``"eeq"``, ``"eeq2019"``, or ``"gfn2"``).
        Defaults to ``"eeq"``.
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

    _model_id: ClassVar[DispersionModelId] = DispersionModelId.D4

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
        ref = self._reference(structure)
        return self._weights(
            self._checked(structure), environment, ref, with_dgwdq=with_dgwdq
        )

    def _weights(
        self,
        structure: Structure,
        environment: Environment,
        ref: D4Reference,
        *,
        with_dgwdq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        zeff_tab, gam_tab = _zeff_gam(self, structure)
        return weight_references_d4(
            structure.numbers,
            environment.cn,
            _charges(environment),
            ref,
            wf=self.wf,
            ga=self.ga,
            gc=self.gc,
            is_srev=False,
            zeff_table=zeff_tab,
            gam_table=gam_tab,
            with_dgwdq=with_dgwdq,  # type: ignore[call-overload]
        )

    def _coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
        ref: D4Reference,
    ) -> AtomicC6:
        if isinstance(environment_or_weights, Environment):
            weights = self._weights(structure, environment_or_weights, ref)
        else:
            weights = environment_or_weights
        return factored_c6(structure.numbers, weights, ref)

    def _alpha_weights(
        self, structure: Structure, environment: Environment, ref: D4Reference
    ) -> Tensor:
        return self._weights(structure, environment, ref)

    def _c6_alpha(
        self, structure: Structure, environment: Environment, ref: D4Reference
    ) -> tuple[Tensor, Tensor, Tensor]:
        # The weights give both the C6 and the polarizabilities.
        w = self._weights(structure, environment, ref)
        c6 = self._coefficients(structure, w, ref).dense()
        return (c6, *self._alpha(structure, w, ref))


class _D4SmoothBase(_D4Base):
    """
    What D4S and D4Srev share: the partner-element-dependent weights with the
    element-pair table `wfpair`. The subclasses name their model (`_label`,
    `_is_srev`) and provide the charge scaling (`_scaling`).
    """

    _is_srev: ClassVar[bool]
    _label: ClassVar[str]

    if TYPE_CHECKING:
        wfpair_table: Tensor | TableFunction | None

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
        ref = self._reference(structure)
        return self._weights(
            self._checked(structure),
            environment,
            ref,
            elements,
            with_dgwdq=with_dgwdq,
        )

    def _scaling(self, structure: Structure) -> dict[str, Any]:
        """The keyword arguments of the charge scaling of the weights."""
        raise NotImplementedError

    def _weights(
        self,
        structure: Structure,
        environment: Environment,
        ref: D4Reference,
        elements: Tensor | None = None,
        *,
        with_dgwdq: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return weight_references_d4s(
            structure.numbers,
            environment.cn,
            _charges(environment),
            ref,
            self._wfpair(structure),
            elements=elements,
            is_srev=self._is_srev,
            with_dgwdq=with_dgwdq,  # type: ignore[call-overload]
            **self._scaling(structure),
        )

    def _coefficients(
        self,
        structure: Structure,
        environment_or_weights: Environment | Tensor,
        ref: D4Reference,
    ) -> AtomicC6:
        wfpair = self._wfpair(structure)
        if isinstance(environment_or_weights, Environment):
            return factored_c6_d4s(
                structure.numbers,
                environment_or_weights.cn,
                _charges(environment_or_weights),
                ref,
                wfpair,
                is_srev=self._is_srev,
                **self._scaling(structure),
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

    def _alpha_weights(
        self, structure: Structure, environment: Environment, ref: D4Reference
    ) -> Tensor:
        # Diagonal element-pair weights wf = wfpair[Z_i, Z_i], as in
        # ``dftd/model/d4/{smooth,srev}.f90``.
        wfpair = self._wfpair(structure)
        return weight_references_d4(
            structure.numbers,
            environment.cn,
            _charges(environment),
            ref,
            wf=wfpair[structure.numbers, structure.numbers],
            is_srev=self._is_srev,
            **self._scaling(structure),
        )

    def _wfpair(self, structure: Structure) -> Tensor:
        if self.wfpair_table is None:
            return resolve_table(WFPAIR_D4S, structure.positions)
        out = resolve_table(self.wfpair_table, structure.positions)
        if out.shape != (119, 119):
            raise ValueError(
                f"'wfpair_table' for {self._label} must have shape "
                f"(119, 119), got {tuple(out.shape)}."
            )
        return out


class D4SModel(_D4SmoothBase):
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

    _model_id: ClassVar[DispersionModelId] = DispersionModelId.D4S
    _is_srev: ClassVar[bool] = False
    _label: ClassVar[str] = "D4S"

    def _scaling(self, structure: Structure) -> dict[str, Any]:
        zeff_tab, gam_tab = _zeff_gam(self, structure)
        return {
            "ga": self.ga,
            "gc": self.gc,
            "zeff_table": zeff_tab,
            "gam_table": gam_tab,
        }


class D4SrevModel(_D4SmoothBase):
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

    _model_id: ClassVar[DispersionModelId] = DispersionModelId.D4SREV
    _is_srev: ClassVar[bool] = True
    _label: ClassVar[str] = "D4Srev"

    def _validate_numbers(self, numbers: Tensor) -> Tensor:
        return _validate_d4srev_numbers(numbers)

    def _coordination_number(
        self,
        structure: Structure,
        pairs: PeriodicShifts | NeighborList | None,
    ) -> Tensor:
        cn_model = ncoord.cn_d4srev.replace(
            count=self.counting_function,
            cutoff=self.cutoff.cn,
            rcov=element_table(
                self.rcov_table, "rcov_table", structure.positions
            ),
        )
        recompute = self.checkpoint and isinstance(pairs, NeighborList)
        return cn_model(
            structure, pairs, mode="recompute" if recompute else "graph"
        )

    def _scaling(self, structure: Structure) -> dict[str, Any]:
        return {"tanh_table": self._tanh(structure)}

    def _reference(self, structure: Structure) -> D4Reference:
        if self.ref is not None:
            return self.ref
        return _default_reference_d4srev(structure.positions, qmod=self.qmodel)

    def _tanh(self, structure: Structure) -> Tensor:
        if self.tanh_table is None:
            return resolve_table(TANH_PARAMS, structure.positions)
        return resolve_table(self.tanh_table, structure.positions)


def d4sc_dispersion_matrix(
    numbers: Tensor,
    positions: Tensor,
    param: Param,
    r4r2: Tensor,
    rc6: Tensor,
    *,
    cutoff: Cutoff | Tensor | float = defaults.D4_DISP2_CUTOFF,
    damping: TwoBodyDamping | None = None,
    rvdw: Tensor | None = None,
) -> Tensor:
    """
    Evaluate the self-consistent D4 pairwise reference dispersion matrix
    ``dispmat`` of shape ``(..., nat, nat, nref, nref)``.
    """
    numbers = _validate_d4_numbers(numbers)
    if isinstance(param, DampingParam):
        dparam = param.replace(
            s6=defaults.D4_S6 if param.s6 is None else param.s6,
            s8=0.0 if param.s8 is None else param.s8,
            a1=defaults.A1 if param.a1 is None else param.a1,
            a2=defaults.A2 if param.a2 is None else param.a2,
        )
    else:
        s6_val = (
            0.0
            if (
                "s6" in param
                and param["s6"] is None
                and param.get("rs6") is None
            )
            else (
                defaults.D4_S6
                if param.get("s6") is None
                else param["s6"]
            )
        )
        s8_val = (
            0.0
            if (param.get("s8") is None and param.get("rs8") is None)
            else (
                defaults.D4_S8
                if param.get("s8") is None
                else param["s8"]
            )
        )
        dparam = DampingParam.from_dict(
            {
                "a1": defaults.A1,
                "a2": defaults.A2,
                **param,
                "s6": s6_val,
                "s8": s8_val,
            }
        )
    damp = RationalTwoBody() if damping is None else damping
    unit_c6 = torch.ones(
        (*numbers.shape, numbers.shape[-1]),
        device=positions.device,
        dtype=positions.dtype,
    )
    edisp = _eval_dispersion2(
        Structure(numbers=numbers, positions=positions),
        dparam,
        unit_c6,
        None,
        damp,
        rvdw=rvdw,
        r4r2=r4r2,
        cutoff=cutoff,
        as_matrix=True,
    )
    return edisp.unsqueeze(-1).unsqueeze(-1) * rc6


def d4sc_atom_energy(dispmat: Tensor, weights: Tensor) -> Tensor:
    """
    Contract a D4SC dispersion matrix ``(..., nat, nat, nref, nref)`` with
    charge-dependent reference weights ``(..., nat, nref)`` into atom-resolved
    dispersion energies ``(..., nat)``.
    """
    return 0.5 * einsum(
        "...ijab,...ia,...jb->...j",
        *(dispmat, weights, weights),
        optimize=[(0, 1), (0, 1)],
    )


def d4sc_atom_potential(
    dispmat: Tensor, weights: Tensor, dgwdq: Tensor
) -> Tensor:
    """
    Contract a D4SC dispersion matrix ``(..., nat, nat, nref, nref)`` with
    reference weights ``weights`` and their charge derivatives ``dgwdq`` into
    the atomwise monopole potential ``dE_disp / dq_i`` of shape ``(..., nat)``.
    """
    return einsum("...ijab,...jb,...ia->...i", dispmat, weights, dgwdq)
