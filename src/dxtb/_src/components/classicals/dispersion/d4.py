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

from dataclasses import dataclass
from typing import Any

import tad_dftd4 as d4
import torch
from tad_dftd4.damping import Damping
from tad_mctc.convert import any_to_tensor
from tad_mctc.data import radii
from tad_mctc.ncoord import erf_count
from tad_mctc.typing import CountingFunction, Tensor, override

from dxtb import IndexHelper
from dxtb._src.ncoord import cn_d4
from dxtb._src.scf.guess import get_eeq_guess

from ..base import ClassicalCache
from .base import Dispersion

__all__ = [
    "LABEL_DISPERSIOND4",
    "DispersionD4",
    "DispersionD4Setup",
    "dispersion_d4_energy",
]

LABEL_DISPERSIOND4 = "DispersionD4"


@dataclass(frozen=True, eq=False)
class DispersionD4Setup:
    """Numbers-only values required by one classical D4 evaluation."""

    numbers: Tensor
    parameters: tuple[tuple[str, Tensor | float | int | str], ...]
    model: d4.model.D4Model
    rcov: Tensor
    r4r2: Tensor
    cutoff: d4.Cutoff
    counting_function: CountingFunction
    damping_function: Damping
    ref_charges: str
    compatibility_q: Tensor | None = None

    def as_param(self) -> d4.Param:
        """Materialize the external library mapping for this call."""
        return dict(self.parameters)  # type: ignore[return-value]


def _copy_model(
    model: d4.model.D4Model, numbers: Tensor, dd: dict[str, Any]
) -> d4.model.D4Model:
    """Copy an external D4 model without changing caller-owned state."""
    return d4.model.D4Model(
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
    if model is not None and type(model) is not d4.model.D4Model:
        raise TypeError(
            "D4: D4Model subclasses cannot be copied safely into setup data."
        )
    if model is None:
        model = d4.model.D4Model(
            numbers, ref_charges=dispersion.ref_charges, **dd
        )
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
        cutoff = d4.Cutoff(**dd)
    else:
        cutoff = d4.Cutoff(
            cutoff.disp2.to(**dd).clone(),
            cutoff.disp3.to(**dd).clone(),
            cutoff.cn.to(**dd).clone(),
            cutoff.cn_eeq.to(**dd).clone(),
            **dd,
        )

    compatibility_q = kwargs.pop("q", None)
    if compatibility_q is not None and not isinstance(compatibility_q, Tensor):
        raise TypeError("D4: 'q' is not of type 'Tensor'.")

    counting_function = kwargs.pop("counting_function", erf_count)
    damping_function = kwargs.pop(
        "damping_function", d4.damping.RationalDamping()
    )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected D4 setup argument(s): {unknown}.")

    # Keep the parameter structure immutable while retaining tensor leaves.
    parameters = tuple(
        (key, value.clone() if isinstance(value, Tensor) else value)
        for key, value in dispersion.param.items()
    )
    if compatibility_q is not None:
        compatibility_q = compatibility_q.clone()
    return DispersionD4Setup(
        numbers=numbers,
        parameters=parameters,
        model=model,
        rcov=rcov,
        r4r2=r4r2,
        cutoff=cutoff,
        counting_function=counting_function,
        damping_function=damping_function,
        ref_charges=dispersion.ref_charges,
        compatibility_q=compatibility_q,
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
        setup.as_param(),
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
    """Compatibility component for the plain-PyTorch classical D4 term."""

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
        cache: ClassicalCache,
        q: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate D4 through the setup-based numerical implementation."""
        if not isinstance(cache, DispersionD4Setup):
            raise TypeError(
                f"Setup in {self.label} is not of type 'DispersionD4Setup'."
            )
        if q is None:
            q = cache.compatibility_q
        charge = kwargs.pop("charge", self.charge)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected D4 energy argument(s): {unknown}.")
        if charge is None:
            charge = positions.new_tensor(0.0)
        return dispersion_d4_energy(cache, positions, charge, q=q)

    def update(self, **kwargs: Any) -> None:
        """Reject mutation of setup-derived exact D4 parameters."""
        if type(self) is DispersionD4:
            raise RuntimeError(
                "Exact classical D4 is setup-derived and immutable. Create a "
                "new Model/System after changing its parameters."
            )
        super().update(**kwargs)

    def reset(self) -> None:
        """Reject mutation of setup-derived exact D4 parameters."""
        if type(self) is DispersionD4:
            raise RuntimeError(
                "Exact classical D4 is setup-derived and immutable. Create a "
                "new Model/System after changing its parameters."
            )
        super().reset()
