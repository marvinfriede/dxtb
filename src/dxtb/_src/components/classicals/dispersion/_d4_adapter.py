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
from tad_dftd import data, defaults
from tad_dftd.reference.d4 import _D4_RAW, _DEFAULT_INTEGRATOR_64
from tad_mctc import storch
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.batch import real_pairs, real_triples
from tad_mctc.convert import any_to_tensor
from tad_mctc.data import VDW_PAIRWISE
from tad_mctc.math import einsum
from tad_mctc.typing import (
    DD,
    CountingFunction,
    Tensor,
    TensorLike,
)
from tad_multicharge import get_eeq_charges

from dxtb._src.ncoord import cn_d4, erf_count


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


def _is_exceptional(val: Tensor, dtype: torch.dtype) -> Tensor:
    """Detect non-finite or overflow values for safe fallback."""
    return (
        torch.isnan(val)
        | torch.isinf(val)
        | (torch.abs(val) > torch.finfo(dtype).max)
    )


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
        alp: Tensor | float | int | None = defaults.D4_ALP,
        bet: Tensor | float | int | None = None,
        doi: str | None = None,
    ) -> Tensor:
        self.doi = doi

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
        radius = a1 * torch.sqrt(radii) + a2
        if only_damping:
            r_order = distances.pow(order)
            return r_order / (r_order + radius.pow(order))
        return 1.0 / (distances.pow(order) + radius.pow(order))


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
            alp_n = alp
            rs = rs6
        elif order == 8:
            assert rs8 is not None
            alp_n = alp + 2.0
            rs = rs8
        elif order == 9:
            assert rs9 is not None
            alp_n = alp
            rs = rs9
        else:
            raise ValueError(f"Unsupported order {order} for ZeroDamping.")

        t_n = rs * storch.safe_divide(distances, radii)
        f_n = 1.0 / (1.0 + 6.0 * t_n**alp_n)
        if only_damping is True:
            return f_n
        return storch.safe_divide(f_n, distances**order)


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
        self.numbers = numbers
        self.ga = ga
        self.gc = gc
        self.ref_charges = ref_charges
        self.wf = self._get_wf() if wf is None else any_to_tensor(wf, **self.dd)
        self.rc6 = self._get_refc6() if rc6 is None else rc6

    def _get_wf(self) -> Tensor:
        return torch.tensor(defaults.D4_WF, **self.dd)

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

        if self.ref_charges == "eeq":
            refq_tab = _D4_RAW["clsq"]
        elif self.ref_charges == "gfn2":
            refq_tab = _D4_RAW["refq"]
        else:
            raise ValueError(f"Unknown reference charges: {self.ref_charges}")

        refq = refq_tab.to(**self.dd)[self.numbers]
        zero = torch.tensor(0.0, **self.dd)
        zero_double = torch.tensor(0.0, device=self.device, dtype=torch.double)

        refc = _D4_RAW["ngw"].to(self.device)[self.numbers]
        mask = refc > 0

        refcn = _D4_RAW["refcovcn"].to(device=self.device, dtype=torch.double)[
            self.numbers
        ]
        dcn = cn.unsqueeze(-1).type(torch.double) - refcn
        tmp = torch.exp(-dcn * dcn)

        def refc_pow(n: int) -> Tensor:
            return sum(
                (torch.pow(tmp, i * self.wf) for i in range(1, n + 1)),
                torch.tensor(0.0, device=tmp.device, dtype=tmp.dtype),
            )

        refc_pow_1 = torch.where(refc == 1, refc_pow(1), tmp)
        refc_pow_final = torch.where(refc == 3, refc_pow(3), refc_pow_1)
        expw = torch.where(mask, refc_pow_final, zero_double)

        norm = torch.where(
            mask,
            torch.sum(expw, dim=-1, keepdim=True),
            torch.tensor(1e-300, device=self.device, dtype=torch.double),
        )
        gw_temp = (expw / norm).type(self.dtype)
        maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]

        gw = torch.where(
            _is_exceptional(gw_temp, self.dtype),
            torch.where(refcn == maxcn, torch.tensor(1.0, **self.dd), zero),
            gw_temp,
        )

        zeff = data.ZEFF(device=self.device, dtype=self.dtype)[
            self.numbers
        ].unsqueeze(-1)
        gam = data.GAM(**self.dd)[self.numbers].unsqueeze(-1) * self.gc
        q_exp = q.unsqueeze(-1)

        zeta_val = torch.where(
            mask, self._zeta(gam, refq + zeff, q_exp + zeff), zero
        )

        if with_dgwdq is False and with_dgwdcn is False:
            return zeta_val * gw

        outputs = [zeta_val * gw]

        if with_dgwdcn is True:

            def _dpow(n: int) -> Tensor:
                return sum(
                    (
                        2 * i * self.wf * dcn * torch.pow(tmp, i * self.wf)
                        for i in range(1, n + 1)
                    ),
                    zero_double,
                )

            wf_1 = torch.where(refc == 1, _dpow(1), zero_double)
            wf_3 = torch.where(refc == 3, _dpow(3), zero_double)
            dexpw = wf_1 + wf_3
            dnorm = torch.sum(dexpw, dim=-1, keepdim=True)
            _dgw = (dexpw - expw * dnorm / norm) / norm
            dgw = torch.where(
                _is_exceptional(_dgw, self.dtype), zero, _dgw.type(self.dtype)
            )
            outputs.append(zeta_val * dgw)

        if with_dgwdq is True:
            dzeta_val = torch.where(
                mask, self._dzeta(gam, refq + zeff, q_exp + zeff), zero
            )
            outputs.append(dzeta_val * gw)

        return tuple(outputs)  # type: ignore[return-value]

    def get_atomic_c6(self, gw: Tensor) -> Tensor:
        return einsum(
            "...ijab,...ia,...jb->...ij",
            *(self.rc6, gw, gw),
            optimize=[(0, 1), (0, 1)],
        )

    def get_weighted_pols(self, gw: Tensor) -> Tensor:
        a = self._get_alpha()
        return einsum("...nr,...nrw->...nw", gw, a)

    def get_polarizabilities(self, weights: Tensor) -> Tensor:
        return einsum("...nr,...nr->...n", weights, self._get_alpha()[..., 0])

    def _zeta(self, gam: Tensor, qref: Tensor, qmod: Tensor) -> Tensor:
        eps = torch.tensor(torch.finfo(self.dtype).eps, **self.dd)
        ga = torch.tensor(self.ga, **self.dd)
        scale = torch.exp(gam * (1.0 - qref / (qmod - eps)))
        return torch.where(
            qmod > 0.0,
            torch.exp(ga * (1.0 - scale)),
            torch.exp(ga),
        )

    def _dzeta(self, gam: Tensor, qref: Tensor, qmod: Tensor) -> Tensor:
        eps = torch.tensor(torch.finfo(self.dtype).eps, **self.dd)
        ga = torch.tensor(self.ga, **self.dd)
        scale = torch.exp(gam * (1.0 - qref / (qmod - eps)))
        zeta_val = torch.exp(ga * (1.0 - scale))
        return torch.where(
            qmod > 0.0,
            -ga * gam * scale * zeta_val * storch.safe_divide(qref, qmod**2),
            torch.tensor(0.0, **self.dd),
        )

    def _get_alpha(self) -> Tensor:
        zero = torch.tensor(0.0, **self.dd)
        refsys = _D4_RAW["refsys"].to(self.device)[self.numbers]
        refascale = _D4_RAW["ascale"].to(**self.dd)[self.numbers]
        refalpha = _D4_RAW["alphaiw"].to(**self.dd)[self.numbers]
        refscount = _D4_RAW["hcount"].to(**self.dd)[self.numbers]
        secscale = _D4_RAW["sscale"].to(**self.dd)
        secalpha = _D4_RAW["secaiw"].to(**self.dd)

        if self.ref_charges == "eeq":
            refsq = _D4_RAW["clsh"].to(**self.dd)[self.numbers]
        elif self.ref_charges == "gfn2":
            refsq = _D4_RAW["refh"].to(**self.dd)[self.numbers]
        else:
            raise ValueError(f"Unknown reference charges: {self.ref_charges}")

        mask = refsys > 0
        zeff = data.ZEFF(device=self.device, dtype=self.dtype)[refsys]
        gam = data.GAM(**self.dd)[refsys] * self.gc

        zeta_val = torch.where(
            mask,
            self._zeta(gam, zeff, refsq + zeff),
            zero,
        )

        aiw = (
            secscale[refsys].unsqueeze(-1)
            * secalpha[refsys]
            * zeta_val.unsqueeze(-1)
        )
        h = refalpha - refscount.unsqueeze(-1) * aiw
        alpha = refascale.unsqueeze(-1) * h
        return torch.where(alpha > 0.0, alpha, zero)

    def _get_refc6(self) -> Tensor:
        alpha = self._get_alpha()
        cp_weights = _DEFAULT_INTEGRATOR_64.cp_weights.to(**self.dd)
        return einsum("w,...irw,...jsw->...ijrs", cp_weights, alpha, alpha)

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
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    zero = torch.tensor(0.0, **dd)

    if cutoff is None:
        cutoff = torch.tensor(defaults.D4_DISP2_CUTOFF, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    qq = 3.0 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    if damping_function.radius_type == "rvdw":
        if rvdw is None:
            raise ValueError("`rvdw` must be provided for `rvdw` radius type.")
        radii = rvdw
    elif damping_function.radius_type == "r4r2":
        radii = qq
    else:
        raise ValueError(f"Unknown radius type: {damping_function.radius_type}")

    active = mask * (distances <= cutoff)
    t6 = torch.where(
        active,
        damping_function(distances, radii, 6, **param),
        zero,
    )
    t8 = torch.where(
        active,
        damping_function(distances, radii, 8, **param),
        zero,
    )

    if as_matrix is True:
        e6 = c6 * t6
        e8 = c8 * t8
    else:
        e6 = torch.sum(c6 * t6, dim=-1)
        e8 = torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", torch.tensor(defaults.D4_S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.D4_S8, **dd))
    edisp = s6 * e6 + s8 * e8

    if "s10" in param:
        c10 = c6 * torch.pow(qq, 2) * 49.0 / 40.0
        t10 = torch.where(
            active,
            damping_function(distances, radii, 10, **param),
            zero,
        )
        e10 = c10 * t10 if as_matrix is True else torch.sum(c10 * t10, dim=-1)
        edisp = edisp + param["s10"] * e10

    if as_matrix is True:
        return -edisp
    return -0.5 * edisp


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

    cutoff2 = cutoff * cutoff
    mask_pairs = real_pairs(numbers, mask_diagonal=True)
    mask_triples = real_triples(numbers, mask_diagonal=True, mask_self=True)

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    one = torch.tensor(1.0, **dd)

    r0ij = radii.unsqueeze(-1)
    r0ik = radii.unsqueeze(-2)
    r0jk = radii.unsqueeze(-3)
    r0 = r0ij * r0ik * r0jk

    distances = torch.pow(
        torch.where(
            mask_pairs,
            storch.cdist(positions, positions, p=2),
            eps,
        ),
        2.0,
    )

    r2ij = distances.unsqueeze(-1)
    r2ik = distances.unsqueeze(-2)
    r2jk = distances.unsqueeze(-3)
    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    r3 = torch.where(mask_triples, r1 * r2, eps)
    r5 = torch.where(mask_triples, r2 * r3, eps)

    fdamp = torch.where(
        mask_triples,
        damping_function(
            r0,
            torch.where(mask_triples, r1, one),
            order=9,
            alp=alp_t / 3.0,
            only_damping=True,
        ),
        zero,
    )

    s = torch.where(
        mask_triples,
        (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik),
        zero,
    )

    ang = torch.where(
        mask_triples
        * (r2ij <= cutoff2)
        * (r2ik <= cutoff2)
        * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        zero,
    )

    energy = ang * fdamp * s9_t * c9
    return torch.sum(energy, dim=(-2, -1)) / 6.0


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
        c9 = storch.safe_sqrt(
            torch.abs(
                c6_3b.unsqueeze(-1) * c6_3b.unsqueeze(-2) * c6_3b.unsqueeze(-3)
            )
        )
        a1 = param.get("a1", torch.tensor(defaults.A1, **dd))
        a2 = param.get("a2", torch.tensor(defaults.A2, **dd))
        radii_3b = (
            a1 * storch.safe_sqrt(3.0 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)) + a2
        )
        alp = param.get("alp", torch.tensor(defaults.D4_ALP, **dd))
        energy = energy + get_atm_dispersion(
            numbers,
            positions,
            c9,
            radii_3b,
            cutoff=cutoff.disp3,
            damping_function=ZeroDamping(),
            s9=s9,
            alp=alp,
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
