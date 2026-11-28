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
Dispersion: D4
==============

DFT-D4 dispersion model.
"""

from __future__ import annotations

from typing import Any

from tad_mctc.convert import any_to_tensor
from tad_mctc.data import radii
from tad_mctc.ncoord import cn_d4, erf_count
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import CountingFunction, Tensor, override

from dxtb import IndexHelper
from dxtb._src.scf.guess import get_eeq_guess

from . import _d4_adapter as d4
from ._d4_adapter import Damping
from .base import Dispersion

__all__ = [
    "LABEL_DISPERSIOND4",
    "DispersionD4",
    "DispersionD4Setup",
    "dispersion_d4_energy",
]

LABEL_DISPERSIOND4 = "DispersionD4"


class DispersionD4Setup(Node):
    """Numbers-only values required by one classical D4 evaluation."""

    numbers: Tensor = child(keep_dtype=True)
    param: d4.Param = child()
    model: d4.model.D4Model = child()
    rcov: Tensor = child()
    r4r2: Tensor = child()
    cutoff: d4.Cutoff = child()
    counting_function: CountingFunction = context()
    damping_function: Damping = context()
    ref_charges: str = context()


def _copy_model(
    model: d4.model.D4Model, numbers: Tensor, dd: dict[str, Any]
) -> d4.model.D4Model:
    """Copy an external D4 model without changing caller-owned state."""
    return d4.model.D4Model.create(
        numbers,
        ga=model.ga,
        gc=model.gc,
        wf=model.wf.to(**dd).clone(),
        ref_charges=model.ref_charges,
        rc6=model.rc6.to(**dd).clone(),
        **dd,
    )


def setup_dispersion_d4(
    dispersion: DispersionD4,
    numbers: Tensor,
    **kwargs: Any,
) -> DispersionD4Setup:
    """Resolve element-only D4 data from an exact or extension component."""
    dd = {"device": dispersion.device, "dtype": dispersion.dtype}

    model = kwargs.pop("model", None)
    if model is not None and not isinstance(model, d4.model.D4Model):
        raise TypeError("D4: Model is not of type 'd4.model.D4Model'.")
    if model is None:
        model = d4.model.D4Model.create(
            numbers, ref_charges=dispersion.ref_charges, **dd
        )
    elif type(model) is not d4.model.D4Model:
        # Keep a caller's D4Model subclass for direct component calls.
        # Model.setup never accepts or stores a custom D4Model.
        model = model.type(dd["dtype"]).to(dd["device"])
    else:
        model = _copy_model(model, numbers, dd)

    rcov = kwargs.pop("rcov", None)
    if rcov is not None and not isinstance(rcov, Tensor):
        raise TypeError("D4: 'rcov' is not of type 'Tensor'.")
    if rcov is None:
        rcov = radii.COV_D3(**dd)[numbers]
    else:
        rcov = rcov.to(**dd).clone()

    r4r2 = kwargs.pop("r4r2", None)
    if r4r2 is not None and not isinstance(r4r2, Tensor):
        raise TypeError("D4: 'r4r2' is not of type 'Tensor'.")
    if r4r2 is None:
        r4r2 = d4.data.R4R2(**dd)[numbers]
    else:
        r4r2 = r4r2.to(**dd).clone()

    cutoff = kwargs.pop("cutoff", None)
    if cutoff is not None and not isinstance(cutoff, d4.Cutoff):
        raise TypeError("D4: 'cutoff' is not of type 'd4.Cutoff'.")
    if cutoff is None:
        cutoff = d4.Cutoff.create(**dd)
    else:
        cutoff = d4.Cutoff.create(
            cutoff.disp2.to(**dd).clone(),
            cutoff.disp3.to(**dd).clone(),
            cutoff.cn.to(**dd).clone(),
            cutoff.cn_eeq.to(**dd).clone(),
            **dd,
        )

    counting_function = kwargs.pop("counting_function", erf_count)
    damping_function = kwargs.pop(
        "damping_function", d4.damping.RationalDamping()
    )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected D4 setup argument(s): {unknown}.")

    return DispersionD4Setup(
        numbers=numbers,
        param={key: value.clone() for key, value in dispersion.param.items()},
        model=model,
        rcov=rcov,
        r4r2=r4r2,
        cutoff=cutoff,
        counting_function=counting_function,
        damping_function=damping_function,
        ref_charges=dispersion.ref_charges,
    )


def dispersion_d4_energy(
    setup: DispersionD4Setup,
    positions: Tensor,
    charge: Tensor | float | int,
    *,
    q: Tensor | None = None,
) -> Tensor:
    """Evaluate classical D4 using explicit setup and call inputs."""
    if positions.device != setup.numbers.device:
        raise RuntimeError("D4 setup and positions must be on the same device.")
    if positions.dtype != setup.rcov.dtype:
        raise RuntimeError("D4 setup and positions must have the same dtype.")

    charge_tensor = any_to_tensor(
        charge, device=positions.device, dtype=positions.dtype
    )
    if q is None:
        q = get_eeq_guess(
            setup.numbers,
            positions,
            charge_tensor,
            cutoff=setup.cutoff.cn_eeq,
        )

    return d4.dftd4(
        setup.numbers,
        positions,
        charge_tensor,
        setup.param,
        model=setup.model,
        rcov=setup.rcov,
        r4r2=setup.r4r2,
        q=q,
        cn_function=cn_d4,
        cutoff=setup.cutoff,
        counting_function=setup.counting_function,
        damping_function=setup.damping_function,
    )


class DispersionD4(Dispersion):
    """Classical for the plain-PyTorch D4 dispersion term."""

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **kwargs: Any
    ) -> DispersionD4Setup:
        """Build fresh element-derived D4 setup; no component state is stored."""
        return setup_dispersion_d4(self, numbers, **kwargs)

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: DispersionD4Setup,
        q: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate D4 through the setup-based numerical implementation."""
        if not isinstance(cache, DispersionD4Setup):
            raise TypeError(
                f"Setup in {self.label} is not of type 'DispersionD4Setup'."
            )
        charge = kwargs.pop("charge", self.charge)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected D4 energy argument(s): {unknown}.")
        if charge is None:
            charge = positions.new_tensor(0.0)
        return dispersion_d4_energy(cache, positions, charge, q=q)
