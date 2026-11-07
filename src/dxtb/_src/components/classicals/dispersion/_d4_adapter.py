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
dxtb D4 Setup and Evaluation Adapter
====================================

Adapts ``tad_dftd``'s D4 reference tables, atomic data, Casimir-Polder
integrator, and EEQ charge solver to ``dxtb``'s classical D4 and self-consistent
D4SC setup and core-node contracts.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import wraps
from types import SimpleNamespace
from typing import Any, Literal, TypedDict, overload

import torch
import tad_dftd as _tad_dftd
from tad_dftd import data, defaults
from tad_dftd.damping import (
    CallableTwoBodyDamping,
    DampingParam,
    RationalTwoBody,
    ThreeBodyDamping,
    TripleData,
    ZeroThreeBodyD4,
    rational_damping,
    zero_damping,
)
from tad_dftd.model.c6 import contract_rc6
from tad_dftd.model.d4 import _validate_d4_numbers
from tad_dftd.model.weights import weight_references_d4
from tad_dftd.reference.d4 import D4Reference, load_d4
from tad_dftd.threebody import dispersion_atm
from tad_mctc import Structure, storch
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.convert import any_to_tensor
from tad_mctc.data import VDW_PAIRWISE
from tad_mctc.math import einsum
from tad_mctc.ncoord import CNModel, cn_d4, erf_count
from tad_mctc.typing import (
    DD,
    CountingFunction,
    Tensor,
    TensorLike,
)
from tad_multicharge import get_eeq_charges


class Param(TypedDict, total=False):
    """Dispersion damping parameter dictionary for D4/D4SC."""

    a1: Tensor
    a2: Tensor
    s6: Tensor
    s8: Tensor
    s9: Tensor
    s10: Tensor
    rs6: Tensor
    rs8: Tensor
    rs10: Tensor
    alp: Tensor

__all__ = [
    "Cutoff",
    "D4Model",
    "Damping",
    "Param",
    "RationalDamping",
    "ZeroDamping",
    "damping",
    "data",
    "defaults",
    "dftd4",
    "dispersion",
    "dispersion2",
    "get_atm_dispersion",
    "model",
]

_MISSING = object()


class Cutoff(TensorLike):
    """Collection of real-space cutoffs for D4 in dxtb."""

    disp2: Tensor
    disp3: Tensor
    cn: Tensor
    cn_eeq: Tensor

    __slots__ = ("disp2", "disp3", "cn", "cn_eeq")

    def __init__(
        self,
        disp2: int | float | Tensor = defaults.D4_DISP2_CUTOFF,
        disp3: int | float | Tensor = defaults.D4_DISP3_CUTOFF,
        cn: int | float | Tensor = defaults.D4_CN_CUTOFF,
        cn_eeq: int | float | Tensor = defaults.D4_CN_EEQ_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.disp2 = any_to_tensor(disp2, device=device, dtype=dtype)
        self.disp3 = any_to_tensor(disp3, device=device, dtype=dtype)
        self.cn = any_to_tensor(cn, device=device, dtype=dtype)
        self.cn_eeq = any_to_tensor(cn_eeq, device=device, dtype=dtype)


class Damping(ABC):
    """Base interface for D4 damping functions in dxtb."""

    radius_type: Literal["r4r2", "rvdw"]
    doi: str | None = None

    def __init__(self, doi: str | None = None) -> None:
        self.doi = doi

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        req_map: dict[int, tuple[str, ...]] = getattr(
            cls, "REQUIRED_BY_ORDER", {}
        )
        if not req_map:
            return

        sig = inspect.signature(cls._f)
        unknown = {
            name
            for seq in req_map.values()
            for name in seq
            if name not in sig.parameters
        }
        if unknown:
            raise TypeError(
                f"{cls.__name__}.REQUIRED_BY_ORDER refers to "
                f"unknown parameter(s): {', '.join(unknown)}"
            )

        original_f = cls._f

        @wraps(original_f)
        def _checked_f(self: Any, *args: Any, **kwargs: Any) -> Tensor:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            order_val = bound.arguments["order"]
            required = req_map.get(order_val, ())
            missing = [
                name
                for name in required
                if bound.arguments.get(name, _MISSING) in (None, _MISSING)
            ]
            if missing:
                raise TypeError(
                    f"{cls.__name__} (order {order_val}) "
                    f"requires keyword(s): {', '.join(missing)}"
                )
            return original_f(self, *args, **kwargs)

        cls._f = _checked_f  # type: ignore[method-assign]

    def __call__(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        only_damping: bool = False,
        *,
        a1: Tensor | float | int | None = None,
        a2: Tensor | float | int | None = None,
        s6: Tensor | float | int | None = defaults.D4_S6,
        rs6: Tensor | float | int | None = None,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = None,
        s9: Tensor | float | int | None = defaults.D4_S9,
        rs9: Tensor | float | int | None = defaults.D4_RS9,
        s10: Tensor | float | int | None = None,
        rs10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = defaults.D4_ALP,
        bet: Tensor | float | int | None = None,
        doi: str | None = None,
    ) -> Tensor:
        del doi, rs10

        if order == 6 and (s6 is None and rs6 is None):
            return torch.zeros_like(distances)
        if order == 8 and (s8 is None and rs8 is None):
            return torch.zeros_like(distances)
        if order == 9 and (s9 is None and rs9 is None):
            return torch.zeros_like(distances)
        if order == 10 and s10 is None:
            return torch.zeros_like(distances)

        return self._f(
            distances,
            radii,
            order,
            a1=a1,
            a2=a2,
            s6=s6,
            rs6=rs6,
            s8=s8,
            rs8=rs8,
            s9=s9,
            rs9=rs9,
            s10=s10,
            alp=alp,
            bet=bet,
            only_damping=only_damping,
        )

    @abstractmethod
    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int | None = None,
        a2: Tensor | float | int | None = None,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = None,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = None,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = None,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = None,
        bet: Tensor | float | int | None = None,
        only_damping: bool = False,
    ) -> Tensor: ...

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Damping):
            return False
        return self.__class__ is other.__class__


class RationalDamping(Damping):
    """Becke-Johnson rational damping for D4."""

    radius_type = "r4r2"
    REQUIRED_BY_ORDER = {
        6: ("a1", "a2"),
        8: ("a1", "a2"),
        10: ("a1", "a2"),
    }

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int | None = defaults.A1,
        a2: Tensor | float | int | None = defaults.A2,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = None,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = None,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = None,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = None,
        bet: Tensor | float | int | None = None,
        only_damping: bool = False,
    ) -> Tensor:
        assert a1 is not None and a2 is not None
        t_n = rational_damping(order, distances, radii, a1=a1, a2=a2)
        if only_damping:
            return distances.pow(order) * t_n
        return t_n


class ZeroDamping(Damping):
    """Chai-Head-Gordon zero damping for D4."""

    radius_type = "rvdw"
    REQUIRED_BY_ORDER = {
        6: ("rs6", "alp"),
        8: ("rs8", "alp"),
        9: ("rs9", "alp"),
    }

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int | None = None,
        a2: Tensor | float | int | None = None,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = defaults.D4_RS6,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = defaults.D4_RS8,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = defaults.D4_RS9,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = defaults.D4_ALP,
        bet: Tensor | float | int | None = None,
        only_damping: bool = False,
    ) -> Tensor:
        assert alp is not None
        if order == 6:
            assert rs6 is not None
            rs = rs6
            alp_arg = alp
        elif order == 8:
            assert rs8 is not None
            rs = rs8
            alp_arg = alp
        elif order == 9:
            assert rs9 is not None
            rs = rs9
            alp_arg = alp - 3.0
        else:
            raise ValueError(f"Unsupported order {order} for ZeroDamping.")

        f_n = zero_damping(
            order,
            radii,
            distances,
            rsn=rs,
            alp=alp_arg,
        )
        if only_damping is True:
            return f_n
        return storch.safe_divide(f_n, distances**order)


class _CallableThreeBodyD4(ThreeBodyDamping):
    """Wrap a custom ``Damping`` instance into ``tad_dftd``'s ``ThreeBodyDamping``."""

    defaults = (("s9", defaults.D4_S9), ("alp", defaults.D4_ALP))
    label = "Custom D4 three-body damping"

    def __init__(self, fn: Damping) -> None:
        self._fn = fn

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        r0 = triples.rdampij * triples.rdampik * triples.rdampjk
        fdamp = self._fn(
            r0,
            triples.r,
            order=9,
            alp=self.value(param, "alp") / 3.0,
            only_damping=True,
        )
        return self.value(param, "s9") * fdamp


class _BaseD4Model(TensorLike):
    """Base slot container for per-system D4 models."""

    __slots__ = ("numbers", "ga", "gc", "wf", "ref_charges", "rc6")


class D4Model(_BaseD4Model):
    """Per-system D4 model adapter backed by ``tad_dftd`` reference tables."""

    numbers: Tensor
    ga: float
    gc: float
    wf: Tensor
    ref_charges: Literal["eeq", "gfn2"]
    rc6: Tensor

    def __init__(
        self,
        numbers: Tensor,
        ga: float = defaults.D4_GA,
        gc: float = defaults.D4_GC,
        wf: Tensor | float | None = None,
        ref_charges: Literal["eeq", "gfn2"] = "eeq",
        rc6: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        _validate_d4_numbers(numbers)
        self.numbers = numbers
        self.ga = ga
        self.gc = gc
        self.ref_charges = ref_charges
        self.wf = self._get_wf() if wf is None else any_to_tensor(wf, **self.dd)
        self.rc6 = self._get_refc6() if rc6 is None else rc6

    def _get_wf(self) -> Tensor:
        return torch.tensor(defaults.D4_WF, **self.dd)

    def _reference(self) -> D4Reference:
        if self.ref_charges not in ("eeq", "gfn2"):
            raise ValueError(f"Unknown reference charges: {self.ref_charges}")
        return load_d4(
            qmod=self.ref_charges,
            ga=self.ga,
            gc=self.gc,
            device=self.device,
            dtype=self.dtype,
        )

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[False] = ...,
        with_dgwdcn: Literal[False] = ...,
    ) -> Tensor: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[True],
        with_dgwdcn: Literal[False] = ...,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[False] = ...,
        with_dgwdcn: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: Literal[True],
        with_dgwdcn: Literal[True],
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        *,
        with_dgwdq: bool = False,
        with_dgwdcn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        if cn is None:
            cn = torch.zeros(self.numbers.shape, **self.dd)
        if q is None:
            q = torch.zeros(self.numbers.shape, **self.dd)

        return weight_references_d4(  # type: ignore[call-overload,no-any-return]
            self.numbers,
            cn,
            q,
            self._reference(),
            wf=self.wf,
            ga=self.ga,
            gc=self.gc,
            is_srev=False,
            with_dgwdq=with_dgwdq,
            with_dgwdcn=with_dgwdcn,
        )

    def get_atomic_c6(self, gw: Tensor) -> Tensor:
        return contract_rc6(self.rc6, gw)

    def get_weighted_pols(self, gw: Tensor) -> Tensor:
        a = self._get_alpha()
        return einsum("...nr,...nrw->...nw", gw, a)

    def get_polarizabilities(self, weights: Tensor) -> Tensor:
        return einsum("...nr,...nr->...n", weights, self._get_alpha()[..., 0])

    def _get_alpha(self) -> Tensor:
        return self._reference().aiw[self.numbers]

    def _get_refc6(self) -> Tensor:
        return self._reference().c6[
            self.numbers.unsqueeze(-1), self.numbers.unsqueeze(-2)
        ]

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}(\n"
            f"  ga={self.ga},\n"
            f"  gc={self.gc},\n"
            f"  wf={self.wf},\n"
            f"  ref_charges={self.ref_charges},\n"
            f"  rc6={self.rc6.shape},\n"
            f")"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


def dispersion2(
    numbers: Tensor,
    positions: Tensor,
    param: Param,
    c6: Tensor,
    r4r2: Tensor,
    rvdw: Tensor | None = None,
    damping_function: Damping = RationalDamping(),
    cutoff: Tensor | None = None,
    as_matrix: bool = False,
) -> Tensor:
    """Calculate two-body D4 dispersion energy between pairs of atoms."""
    if damping_function.radius_type == "rvdw":
        if rvdw is None:
            raise ValueError("`rvdw` must be provided for `rvdw` radius type.")
    elif damping_function.radius_type != "r4r2":
        raise ValueError(f"Unknown radius type: {damping_function.radius_type}")

    cut: Tensor | float = (
        defaults.D4_DISP2_CUTOFF if cutoff is None else cutoff
    )
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
    param_2b = {**param, "s6": s6_val, "s8": s8_val}
    if type(damping_function) is RationalDamping:
        damp_2b = RationalTwoBody()
        param_arg: DampingParam | dict[str, Any] = DampingParam.from_dict(
            {
                "a1": defaults.A1,
                "a2": defaults.A2,
                **param_2b,
            }
        )
    else:
        damp_2b = CallableTwoBodyDamping(
            damping_function,
            param,
            radius_type=damping_function.radius_type,
        )
        param_arg = param_2b
    return _tad_dftd.disp.dispersion2(
        numbers,
        positions,
        param_arg,
        c6,
        r4r2=r4r2,
        rvdw=rvdw,
        damping=damp_2b,
        cutoff=cut,
        as_matrix=as_matrix,
    )


def get_atm_dispersion(
    numbers: Tensor,
    positions: Tensor,
    c9: Tensor,
    radii: Tensor,
    cutoff: Tensor,
    damping_function: Damping = ZeroDamping(),
    s9: Tensor | float | int = defaults.D4_S9,
    alp: Tensor | float | int = defaults.D4_ALP,
) -> Tensor:
    """Axilrod-Teller-Muto three-body dispersion term for D4."""
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    s9_t = any_to_tensor(s9, **dd)
    alp_t = any_to_tensor(alp, **dd)
    c6_dummy = torch.zeros(
        (*numbers.shape, numbers.shape[-1]),
        device=positions.device,
        dtype=positions.dtype,
    )
    dparam = DampingParam(
        s9=s9_t,
        alp=alp_t,
        a1=1.0,
        a2=0.0,
        rs9=1.0,
    )
    three_damp: ThreeBodyDamping = (
        ZeroThreeBodyD4()
        if type(damping_function) is ZeroDamping
        else _CallableThreeBodyD4(damping_function)
    )
    return dispersion_atm(
        Structure(numbers=numbers, positions=positions),
        c6_dummy,
        dparam,
        damping=three_damp,
        rdamp=radii,
        c9=c9,
        cutoff=cutoff,
    )


def dftd4(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor | float | int,
    param: Param,
    *,
    model: str | D4Model = "d4",
    rcov: Tensor | None = None,
    r4r2: Tensor | None = None,
    rvdw: Tensor | None = None,
    q: Tensor | None = None,
    cutoff: Cutoff | None = None,
    cn_function: Any = cn_d4,
    counting_function: CountingFunction = erf_count,
    damping_function: Damping = RationalDamping(),
) -> Tensor:
    """Evaluate DFT-D4 dispersion energy for a (batch of) molecule(s)."""
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    charge_t = any_to_tensor(charge, **dd)

    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    if cutoff is None:
        cutoff = Cutoff(**dd)

    if isinstance(model, str):
        if model.casefold() != "d4":
            raise ValueError(f"Unsupported model key '{model}'.")
        model_inst = D4Model(numbers=numbers, **dd)
    else:
        model_inst = model

    if rcov is not None and numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii ({rcov.shape}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    if r4r2 is None:
        r4r2 = data.R4R2(**dd)[numbers]
    elif numbers.shape != r4r2.shape:
        raise ValueError(
            f"Shape of expectation values r4r2 ({r4r2.shape}) is not "
            f"consistent with atomic numbers ({numbers.shape})."
        )

    if rvdw is None:
        rvdw = VDW_PAIRWISE(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    elif numbers.shape != rvdw.shape[:-1]:
        raise ValueError(
            f"Shape of van der Waals radii ({rvdw.shape}) is not "
            f"consistent with atomic numbers ({numbers.shape})."
        )

    if isinstance(cn_function, CNModel):
        cn_model = cn_function
        if counting_function is not erf_count:
            cn_model = cn_model.replace(count=counting_function)
        cn = cn_model(
            Structure(
                numbers=numbers.expand(positions.shape[:-1]),
                positions=positions,
            )
        )
    else:
        cn = cn_function(
            numbers,
            positions,
            counting_function=counting_function,
        )

    if q is None:
        q = get_eeq_charges(numbers, positions, charge_t, cutoff=cutoff.cn_eeq)
    elif numbers.shape != q.shape:
        raise ValueError(
            f"Shape of atomic charges ({q.shape}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    weights = model_inst.weight_references(cn, q)
    c6 = model_inst.get_atomic_c6(weights)
    energy = dispersion2(
        numbers,
        positions,
        param,
        c6,
        r4r2,
        rvdw,
        damping_function,
        cutoff.disp2,
    )

    s9 = param.get("s9", torch.tensor(defaults.D4_S9, **dd))
    if isinstance(s9, Tensor):
        use_atm = s9.requires_grad or is_functorch_tensor(s9)
        if not use_atm:
            use_atm = bool(s9 != 0.0)
    else:
        use_atm = s9 is not None and s9 != 0.0

    if use_atm:
        weights_3b = model_inst.weight_references(cn, None)
        c6_3b = model_inst.get_atomic_c6(weights_3b)
        param_3b = {
            "a1": param.get("a1", torch.tensor(defaults.A1, **dd)),
            "a2": param.get("a2", torch.tensor(defaults.A2, **dd)),
            "s9": s9,
            "alp": param.get("alp", torch.tensor(defaults.D4_ALP, **dd)),
        }
        energy = energy + _tad_dftd.disp.dispersion3(
            numbers,
            positions,
            param_3b,
            c6_3b,
            rvdw=rvdw,
            r4r2=r4r2,
            cutoff=cutoff.disp3,
            damping=ZeroThreeBodyD4(),
        )

    return energy


cutoff = SimpleNamespace(Cutoff=Cutoff)
model = SimpleNamespace(D4Model=D4Model, BaseModel=_BaseD4Model)
damping = SimpleNamespace(
    Damping=Damping,
    RationalDamping=RationalDamping,
    ZeroDamping=ZeroDamping,
)
dispersion = SimpleNamespace(
    dispersion2=dispersion2,
    get_atm_dispersion=get_atm_dispersion,
)
