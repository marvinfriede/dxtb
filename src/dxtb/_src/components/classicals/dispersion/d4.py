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
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import CountingFunction, Tensor, override

from dxtb import IndexHelper
from dxtb._src.ncoord import cn_d4
from dxtb._src.scf.guess import get_eeq_guess

from ..base import ClassicalCache
from .base import Dispersion

__all__ = [
    "LABEL_DISPERSIOND4",
    "DispersionD4",
    "DispersionD4CoreSetup",
    "DispersionD4Setup",
    "core_dispersion_d4_setup",
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


class _D4ParameterValue(Node):
    """Typed representation for one tensor or static D4 parameter value."""

    kind: str = context()
    tensor_value: Tensor | None = child(keep_dtype=True)
    static_value: bool | int | float | str | None = context()

    def _convert_child(
        self,
        name: str,
        value: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> Any:
        if name == "tensor_value" and isinstance(value, Tensor):
            target_dtype = dtype if value.is_floating_point() else None
            return value.to(device=device, dtype=target_dtype)
        return super()._convert_child(name, value, device, dtype)

    @classmethod
    def encode(cls, value: Tensor | float | int | str) -> _D4ParameterValue:
        """Separate dynamic tensors from static scalar parameter values."""
        if isinstance(value, Tensor):
            return cls(kind="tensor", tensor_value=value, static_value=None)
        if isinstance(value, bool):
            return cls(kind="bool", tensor_value=None, static_value=value)
        if isinstance(value, int):
            return cls(kind="int", tensor_value=None, static_value=value)
        if isinstance(value, float):
            return cls(kind="float", tensor_value=None, static_value=value)
        if isinstance(value, str):
            return cls(kind="str", tensor_value=None, static_value=value)
        raise TypeError(
            "D4 parameter values must be tensors or immutable scalar values; "
            f"received {type(value).__name__}."
        )

    def decode(self) -> Tensor | bool | int | float | str:
        """Rebuild the original supported parameter value type."""
        if self.kind == "tensor":
            if self.tensor_value is None:
                raise TypeError("Tensor D4 parameter has no tensor value.")
            return self.tensor_value
        if self.kind in ("bool", "int", "float", "str"):
            return self.static_value  # type: ignore[return-value]
        raise TypeError(f"Unsupported D4 parameter value kind: {self.kind!r}.")


class _D4NamedParameter(Node):
    """One ordered D4 parameter, with its name kept as static metadata."""

    name: str = context()
    value: _D4ParameterValue = child()


class _D4CutoffSetup(Node):
    """Dynamic tensor representation of the four D4 cutoff values."""

    disp2: Tensor = child()
    disp3: Tensor = child()
    cn: Tensor = child()
    cn_eeq: Tensor = child()


class DispersionD4CoreSetup(Node):
    """Immutable dynamic setup values for single-System classical D4."""

    numbers: Tensor = child(keep_dtype=True)
    parameter_entries: tuple[_D4NamedParameter, ...] = child()
    model_numbers: Tensor = child(keep_dtype=True)
    model_ga: float = context()
    model_gc: float = context()
    model_wf: Tensor = child()
    model_ref_charges: str = context()
    model_rc6: Tensor = child()
    rcov: Tensor = child()
    r4r2: Tensor = child()
    cutoff: _D4CutoffSetup = child()
    counting_function: CountingFunction = context()
    damping_type: type[Damping] = context()
    ref_charges: str = context()
    compatibility_q: Tensor | None = child()

    @property
    def parameters(self) -> tuple[tuple[str, Any], ...]:
        """Return the ordered legacy-compatible parameter view."""
        return tuple(
            (entry.name, entry.value.decode())
            for entry in self.parameter_entries
        )

    def as_param(self) -> d4.Param:
        """Materialize the parameter mapping expected by tad-dftd4."""
        return dict(self.parameters)  # type: ignore[return-value]


def core_dispersion_d4_setup(
    setup: DispersionD4Setup,
) -> DispersionD4CoreSetup:
    """Convert a supported built-in legacy D4 setup without cloning tensors.

    Custom models and damping strategies remain available through the direct
    legacy setup API. The System-owned Node path accepts only the canonical
    built-in model and rational damping strategy.
    """
    if type(setup.model) is not d4.model.D4Model:
        raise TypeError(
            "System-owned classical D4 setup requires the built-in D4Model; "
            "custom model subclasses remain on the legacy path."
        )
    if type(setup.damping_function) is not d4.damping.RationalDamping:
        raise TypeError(
            "System-owned classical D4 setup requires RationalDamping; "
            "custom damping strategies remain on the legacy path."
        )
    if setup.counting_function is not erf_count:
        raise TypeError(
            "System-owned classical D4 setup requires the built-in "
            "stateless erf_count function."
        )
    if not isinstance(setup.model.ga, float) or not isinstance(
        setup.model.gc, float
    ):
        raise TypeError("D4 model ga and gc must be Python floats.")
    if not isinstance(setup.model.ref_charges, str) or not isinstance(
        setup.ref_charges, str
    ):
        raise TypeError(
            "D4 model and setup reference-charge modes must be strings."
        )

    return DispersionD4CoreSetup(
        numbers=setup.numbers,
        parameter_entries=tuple(
            _D4NamedParameter(name=name, value=_D4ParameterValue.encode(value))
            for name, value in setup.parameters
        ),
        model_numbers=setup.model.numbers,
        model_ga=setup.model.ga,
        model_gc=setup.model.gc,
        model_wf=setup.model.wf,
        model_ref_charges=setup.model.ref_charges,
        model_rc6=setup.model.rc6,
        rcov=setup.rcov,
        r4r2=setup.r4r2,
        cutoff=_D4CutoffSetup(
            disp2=setup.cutoff.disp2,
            disp3=setup.cutoff.disp3,
            cn=setup.cutoff.cn,
            cn_eeq=setup.cutoff.cn_eeq,
        ),
        counting_function=setup.counting_function,
        damping_type=type(setup.damping_function),
        ref_charges=setup.ref_charges,
        compatibility_q=setup.compatibility_q,
    )


def _d4_setup_view(setup: DispersionD4CoreSetup) -> DispersionD4Setup:
    """Construct fresh external D4 objects from one captured core setup."""
    model = d4.model.D4Model(
        setup.model_numbers.clone(),
        ga=setup.model_ga,
        gc=setup.model_gc,
        wf=setup.model_wf.clone(),
        ref_charges=setup.model_ref_charges,
        rc6=setup.model_rc6.clone(),
        device=setup.numbers.device,
        dtype=setup.model_wf.dtype,
    )
    cutoff = d4.Cutoff(
        setup.cutoff.disp2.clone(),
        setup.cutoff.disp3.clone(),
        setup.cutoff.cn.clone(),
        setup.cutoff.cn_eeq.clone(),
        device=setup.numbers.device,
        dtype=setup.rcov.dtype,
    )
    damping_function = setup.damping_type()
    return DispersionD4Setup(
        numbers=setup.numbers,
        parameters=setup.parameters,
        model=model,
        rcov=setup.rcov,
        r4r2=setup.r4r2,
        cutoff=cutoff,
        counting_function=setup.counting_function,
        damping_function=damping_function,
        ref_charges=setup.ref_charges,
        compatibility_q=setup.compatibility_q,
    )


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
    if model is None:
        model = d4.model.D4Model(
            numbers, ref_charges=dispersion.ref_charges, **dd
        )
    elif type(model) is not d4.model.D4Model:
        # Preserve caller extensions at this legacy compatibility boundary.
        # Canonical Model.setup never accepts or stores a custom D4Model.
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
    setup: DispersionD4Setup | DispersionD4CoreSetup,
    positions: Tensor,
    charge: Tensor | float | int,
    *,
    q: Tensor | None = None,
) -> Tensor:
    """Evaluate classical D4 using explicit setup and call inputs."""
    if isinstance(setup, DispersionD4CoreSetup):
        setup = _d4_setup_view(setup)

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
        cache: DispersionD4Setup | DispersionD4CoreSetup,
        q: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate D4 through the setup-based numerical implementation."""
        if not isinstance(cache, (DispersionD4Setup, DispersionD4CoreSetup)):
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
