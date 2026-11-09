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
Reference systems for DFT-D4, DFT-D4S, and DFT-D4Srev.

Matches ``dftd/src/dftd/model/d4/reference/{orig,srev}.f90`` and
``dftd/src/dftd/model/d4/{orig,smooth,srev}.f90``.
"""
from __future__ import annotations

import math

import torch
from tad_mctc.tools import is_compiling
from tad_mctc.tree import Node, child
from tad_mctc.typing import Self, Tensor, get_default_device, get_default_dtype

from .. import defaults
from ..charge import normalize_qmodel
from ..data.hardness import GAM
from ..data.table import load_pkg_tensor
from ..data.tanh import TANH_PARAMS
from ..data.zeff import ZEFF
from ..integrator.trapezoid import TrapezoidIntegrator
from ..model.weights import zeta, zeta_tanh

__all__ = [
    "D4Reference",
    "_default_reference_d4",
    "_default_reference_d4srev",
    "load_d4",
    "load_d4srev",
]


_D4_RAW: dict[str, Tensor] = load_pkg_tensor(__package__, "d4-ref.pt")
_D4SREV_RAW: dict[str, Tensor] = load_pkg_tensor(__package__, "d4srev-ref.pt")

_DEFAULT_INTEGRATOR_64 = TrapezoidIntegrator(
    device=torch.device("cpu"), dtype=torch.float64
)


def _compute_d4_aiw_and_c6(
    raw: dict[str, Tensor],
    qmod: str,
    ga: Tensor | float,
    gc: Tensor | float,
    *,
    is_srev: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute ``(refn, ngw, cn, q, aiw, c6)`` for D4/D4S or D4Srev.
    """
    qkey = normalize_qmodel(qmod)
    if is_srev:
        q_map = {
            "eeq": ("clsq", "clsh"),
            "gfn2": ("refq", "refh"),
            "gxtb": ("gxtbq", "gxtbh"),
        }
        if qkey not in q_map:
            raise ValueError(
                f"Invalid option '{qmod}' for charge model in D4Srev."
            )
    else:
        if qkey == "gxtb":
            raise ValueError("g-xTB charges are not supported for D4")
        q_map = {
            "eeq": ("clsq", "clsh"),
            "gfn2": ("refq", "refh"),
        }
        if qkey not in q_map:
            raise ValueError(
                f"Invalid option '{qmod}' for charge model in D4."
            )

    q_name, h_name = q_map[qkey]

    refn = raw["refn"].to(device=device)
    ngw = raw["ngw"].to(device=device)
    refsys = raw["refsys"].to(device=device)
    sscale = raw["sscale"].to(device=device, dtype=dtype)
    secaiw = raw["secaiw"].to(device=device, dtype=dtype)
    alphaiw = raw["alphaiw"].to(device=device, dtype=dtype)
    ascale = raw["ascale"].to(device=device, dtype=dtype)
    hcount = raw["hcount"].to(device=device, dtype=dtype)
    refcovcn = raw["refcovcn"].to(device=device, dtype=dtype)
    ref_q = raw[q_name].to(device=device, dtype=dtype)
    ref_h = raw[h_name].to(device=device, dtype=dtype)

    # Mask of valid reference entries: ir < refn[Z]
    ir_idx = torch.arange(7, device=device)[None, :]  # (1, 7)
    valid_ref = (ir_idx < refn[:, None]) & (refsys > 0)

    if is_srev:
        tanh_tab = TANH_PARAMS(device=device, dtype=dtype)  # (119, 4)
        a_s = tanh_tab[refsys, 0]
        b_s = tanh_tab[refsys, 1]
        c_s = tanh_tab[refsys, 2]
        d_s = tanh_tab[refsys, 3]
        z_sec = zeta_tanh(
            a_s,
            b_s,
            c_s,
            d_s,
            torch.zeros_like(ref_h),
            ref_h,
            mask=valid_ref,
        )
    else:
        zeff_tab = ZEFF(device=device, dtype=dtype)
        gam_tab = GAM(device=device, dtype=dtype)
        iz = zeff_tab[refsys]
        gi = gam_tab[refsys] * gc
        z_sec = zeta(ga, gi, iz, ref_h + iz, mask=valid_ref)

    aiw_sec = sscale[refsys][..., None] * secaiw[refsys] * z_sec[..., None]
    raw_aiw = ascale[..., None] * (alphaiw - hcount[..., None] * aiw_sec)
    aiw = torch.where(
        valid_ref[..., None],
        torch.clamp_min(raw_aiw, 0.0),
        torch.zeros_like(raw_aiw),
    )

    weights = _DEFAULT_INTEGRATOR_64.weights.to(
        device=device, dtype=dtype
    )
    c6 = (3.0 / math.pi) * torch.einsum("w,irw,jsw->ijrs", weights, aiw, aiw)

    return refn, ngw, refcovcn, ref_q, aiw, c6


_D4_CACHE: dict[tuple[str, bool], D4Reference] = {}


class D4Reference(Node):
    """
    Reference data container for D4, D4S, and D4Srev.

    Parameters
    ----------
    refn : Tensor
        Number of references per element, shape ``(119,)``.
    ngw : Tensor
        Number of Gaussian weights per reference, shape ``(119, 7)``.
    cn : Tensor
        Reference covalent coordination numbers, shape ``(119, 7)``.
    q : Tensor
        Reference partial charges, shape ``(119, 7)``.
    aiw : Tensor
        Reference dynamic polarizabilities, shape ``(119, 7, 23)``.
    c6 : Tensor
        Reference :math:`C_6` coefficients, shape ``(119, 119, 7, 7)``.
    """

    refn: Tensor = child()
    ngw: Tensor = child()
    cn: Tensor = child()
    q: Tensor = child()
    aiw: Tensor = child()
    c6: Tensor = child()

    @property
    def n_elem(self) -> int:
        """Number of elements in the reference table (including padding 0)."""
        return int(self.cn.shape[0])

    @property
    def n_ref(self) -> int:
        """Maximum number of reference systems per element (7)."""
        return int(self.cn.shape[1])

    @classmethod
    def _from_computed(
        cls,
        computed: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        dtype: torch.dtype,
    ) -> Self:
        """The reference of freshly computed tensors, floats cast to `dtype`."""
        refn, ngw, cn, q, aiw, c6 = computed
        return cls(
            refn=refn,
            ngw=ngw,
            cn=cn.to(dtype=dtype),
            q=q.to(dtype=dtype),
            aiw=aiw.to(dtype=dtype),
            c6=c6.to(dtype=dtype),
        )

    @classmethod
    def _load_default(
        cls,
        qkey: str,
        *,
        is_srev: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Self:
        """
        The reference with the default charge scaling of the model, from the
        module cache `_D4_CACHE` (filled once, on the CPU in double precision).
        """
        cache_key = (qkey, is_srev)
        if is_srev:
            raw, ga, gc = _D4SREV_RAW, 0.0, 0.0
        else:
            raw, ga, gc = _D4_RAW, defaults.D4_GA, defaults.D4_GC

        if cache_key not in _D4_CACHE:
            if is_compiling():
                return cls._from_computed(
                    _compute_d4_aiw_and_c6(
                        raw,
                        qkey,
                        ga,
                        gc,
                        is_srev=is_srev,
                        device=device,
                        dtype=torch.float64,
                    ),
                    dtype,
                )
            with torch._C._DisableFuncTorch(), torch.no_grad():  # pyright: ignore[reportAttributeAccessIssue]
                refn, ngw, cn, q, aiw, c6 = _compute_d4_aiw_and_c6(
                    raw,
                    qkey,
                    ga,
                    gc,
                    is_srev=is_srev,
                    device=torch.device("cpu"),
                    dtype=torch.float64,
                )
                _D4_CACHE[cache_key] = D4Reference(
                    refn=refn, ngw=ngw, cn=cn, q=q, aiw=aiw, c6=c6
                )
        cached = _D4_CACHE[cache_key]
        return cls(
            refn=cached.refn.to(device=device),
            ngw=cached.ngw.to(device=device),
            cn=cached.cn.to(device=device, dtype=dtype),
            q=cached.q.to(device=device, dtype=dtype),
            aiw=cached.aiw.to(device=device, dtype=dtype),
            c6=cached.c6.to(device=device, dtype=dtype),
        )

    @classmethod
    def load_d4(
        cls,
        qmod: str = "eeq",
        ga: Tensor | float = defaults.D4_GA,
        gc: Tensor | float = defaults.D4_GC,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """
        Load D4 / D4S reference data for a given charge model ``qmod`` and
        charge-scaling parameters ``(ga, gc)``.
        """
        dev = get_default_device() if device is None else device
        dt = get_default_dtype() if dtype is None else dtype
        qkey = normalize_qmodel(qmod)

        is_default_scalars = (
            not isinstance(ga, Tensor)
            and not isinstance(gc, Tensor)
            and float(ga) == defaults.D4_GA
            and float(gc) == defaults.D4_GC
        )
        if is_default_scalars:
            return cls._load_default(
                qkey, is_srev=False, device=dev, dtype=dt
            )

        # Compute in float64 if inputs are Python floats, or in `dt` if Tensor
        compute_dt = dt if (isinstance(ga, Tensor) or isinstance(gc, Tensor)) else torch.float64
        return cls._from_computed(
            _compute_d4_aiw_and_c6(
                _D4_RAW,
                qkey,
                ga,
                gc,
                is_srev=False,
                device=dev,
                dtype=compute_dt,
            ),
            dt,
        )

    @classmethod
    def load_d4srev(
        cls,
        qmod: str = "eeq",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """
        Load D4Srev reference data for a given charge model ``qmod``.
        """
        dev = get_default_device() if device is None else device
        dt = get_default_dtype() if dtype is None else dtype
        return cls._load_default(
            normalize_qmodel(qmod), is_srev=True, device=dev, dtype=dt
        )


def load_d4(
    qmod: str = "eeq",
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> D4Reference:
    """Load D4 / D4S reference data."""
    return D4Reference.load_d4(
        qmod=qmod, ga=ga, gc=gc, device=device, dtype=dtype
    )


def load_d4srev(
    qmod: str = "eeq",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> D4Reference:
    """Load D4Srev reference data."""
    return D4Reference.load_d4srev(qmod=qmod, device=device, dtype=dtype)


def _default_reference_d4(
    like: Tensor,
    *,
    qmod: str = "eeq",
    ga: Tensor | float = defaults.D4_GA,
    gc: Tensor | float = defaults.D4_GC,
) -> D4Reference:
    """Load D4 / D4S reference data matching the device and dtype of ``like``."""
    return D4Reference.load_d4(
        qmod=qmod, ga=ga, gc=gc, device=like.device, dtype=like.dtype
    )


def _default_reference_d4srev(
    like: Tensor,
    *,
    qmod: str = "eeq",
) -> D4Reference:
    """Load D4Srev reference data matching the device and dtype of ``like``."""
    return D4Reference.load_d4srev(
        qmod=qmod, device=like.device, dtype=like.dtype
    )
