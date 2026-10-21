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
Dispersion: D3
==============

The DFT-D3 dispersion model.
"""

from __future__ import annotations

import inspect
import warnings

import tad_dftd3 as d3
import torch
from tad_mctc.data import radii
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper
from dxtb._src.ncoord import coordination_number, exp_count
from dxtb._src.typing import Any, CountingFunction, Tensor, override

from ..base import ClassicalCache, ComponentCache
from .base import Dispersion

__all__ = [
    "LABEL_DISPERSIOND3",
    "DispersionD3",
    "DispersionD3Cache",
    "DispersionD3CoreSetup",
    "core_dispersion_d3_setup",
    "dispersion_d3_energy",
]

_ATOMIC_C6_SUPPORTS_CHUNK_SIZE = (
    "chunk_size" in inspect.signature(d3.model.atomic_c6).parameters
)


def _atomic_c6(
    numbers: Tensor,
    weights: Tensor,
    reference: d3.reference.Reference,
    *,
    chunk_size: int | None,
) -> Tensor:
    """Call the installed tad-dftd3 atomic C6 API compatibly.

    tad-dftd3 0.6 exposes explicit chunking for its pair/reference
    intermediate. The 0.7 composite implementation replaces that large
    intermediate with an element-grouped contraction and removes the option.
    Keep accepting the legacy keyword during migration, but make its changed
    semantics visible when a caller supplies a value on the new API.
    """
    atomic_c6 = d3.model.atomic_c6
    if _ATOMIC_C6_SUPPORTS_CHUNK_SIZE:
        return atomic_c6(numbers, weights, reference, chunk_size=chunk_size)

    if chunk_size is not None:
        warnings.warn(
            "tad-dftd3 no longer accepts chunk_size; its grouped atomic-C6 "
            "algorithm bounds the former pair/reference intermediate, so "
            "the requested chunk_size is not applied",
            DeprecationWarning,
            stacklevel=2,
        )
    return atomic_c6(numbers, weights, reference)


LABEL_DISPERSIOND3 = "DispersionD3"


class DispersionD3Cache(ClassicalCache):
    """
    Cache for the dispersion settings.

    Note
    ----
    The dispersion parameters (a1, a2, ...) are given in the constructor.
    """

    __slots__ = [
        "ref",
        "rcov",
        "rvdw",
        "r4r2",
        "cutoff",
        "counting_function",
        "weighting_function",
        "damping_function",
    ]

    def __init__(
        self,
        ref: d3.reference.Reference,
        rcov: Tensor,
        rvdw: Tensor,
        r4r2: Tensor,
        cutoff: Tensor,
        counting_function: CountingFunction,
        weighting_function: d3.typing.WeightingFunction,
        damping_function: d3.typing.DampingFunction,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else rcov.device,
            dtype=dtype if dtype is None else rcov.dtype,
        )
        self.ref = ref
        self.rcov = rcov
        self.rvdw = rvdw
        self.r4r2 = r4r2
        self.cutoff = cutoff
        self.counting_function = counting_function
        self.weighting_function = weighting_function
        self.damping_function = damping_function


class _D3ParameterValue(Node):
    """Typed representation of one tensor or scalar D3 parameter."""

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
    def encode(cls, value: Tensor | float | int | str) -> _D3ParameterValue:
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
            "D3 parameter values must be tensors or immutable scalar values; "
            f"received {type(value).__name__}."
        )

    def decode(self) -> Tensor | bool | int | float | str:
        if self.kind == "tensor":
            if self.tensor_value is None:
                raise TypeError("Tensor D3 parameter has no tensor value.")
            return self.tensor_value
        if self.kind in ("bool", "int", "float", "str"):
            return self.static_value  # type: ignore[return-value]
        raise TypeError(f"Unsupported D3 parameter value kind: {self.kind!r}.")


class _D3NamedParameter(Node):
    """An ordered D3 parameter with its key kept as static metadata."""

    name: str = context()
    value: _D3ParameterValue = child()


class DispersionD3CoreSetup(Node):
    """Explicit, transform-visible setup for exact single-System D3."""

    setup_numbers: Tensor = child(keep_dtype=True)
    component_numbers: Tensor = child(keep_dtype=True)
    parameter_entries: tuple[_D3NamedParameter, ...] = child()
    reference_cn: Tensor = child()
    reference_c6: Tensor = child()
    rcov: Tensor = child()
    rvdw: Tensor = child()
    r4r2: Tensor = child()
    cutoff: Tensor = child()
    counting_function: CountingFunction = context()
    weighting_function: Any = context()
    damping_function: Any = context()

    @property
    def parameters(self) -> tuple[tuple[str, Any], ...]:
        """Return ordered parameters with their original value types."""
        return tuple(
            (entry.name, entry.value.decode())
            for entry in self.parameter_entries
        )

    def as_param(self) -> dict[str, Any]:
        """Materialize the mapping accepted by tad-dftd3."""
        return dict(self.parameters)


def core_dispersion_d3_setup(
    setup: DispersionD3Cache,
    setup_numbers: Tensor,
    component_numbers: Tensor,
    parameters: dict[str, Any],
    *,
    snapshot: bool = False,
) -> DispersionD3CoreSetup:
    """Represent one legacy D3 cache explicitly, preserving its two number sources.

    The legacy API gathers ``rcov``, ``rvdw`` and ``r4r2`` from the numbers
    passed to ``get_cache`` but evaluates the D3 kernels using the component's
    own numbers. The two tensors are therefore retained separately.

    ``snapshot`` is used only at the System ownership boundary: it clones the
    number and parameter inputs while preserving their autograd edges. Other
    cache tensors transfer to the Node when the legacy cache entry is replaced.
    """
    if setup.counting_function is not exp_count:
        raise TypeError(
            "System-owned D3 setup requires the built-in stateless exp_count "
            "counting function."
        )
    if setup.weighting_function is not d3.model.gaussian_weight:
        raise TypeError(
            "System-owned D3 setup requires the built-in stateless "
            "gaussian_weight function."
        )
    if setup.damping_function is not d3.damping.rational_damping:
        raise TypeError(
            "System-owned D3 setup requires the built-in stateless "
            "rational_damping function."
        )

    def capture(value: Any) -> Any:
        return (
            value.clone() if snapshot and isinstance(value, Tensor) else value
        )

    return DispersionD3CoreSetup(
        setup_numbers=capture(setup_numbers),
        component_numbers=capture(component_numbers),
        parameter_entries=tuple(
            _D3NamedParameter(
                name=name,
                value=_D3ParameterValue.encode(capture(value)),
            )
            for name, value in parameters.items()
        ),
        reference_cn=setup.ref.cn,
        reference_c6=setup.ref.c6,
        rcov=setup.rcov,
        rvdw=setup.rvdw,
        r4r2=setup.r4r2,
        cutoff=setup.cutoff,
        counting_function=setup.counting_function,
        weighting_function=setup.weighting_function,
        damping_function=setup.damping_function,
    )


def dispersion_d3_energy(
    setup: DispersionD3CoreSetup,
    positions: Tensor,
    *,
    chunk_size: int | None = None,
) -> Tensor:
    """Evaluate D3 using only captured setup values and call-local geometry."""
    # Keep the fixed atomic numbers aligned with geometry batch dimensions.
    # The upstream AtomicC6 vmap rule expects its numbers input to carry the
    # same mapped axis as its reference weights; without this zero-valued
    # structural broadcast, conformer vmap can mistake the atom axis for the
    # conformer axis (or reject the call when the dimensions differ).
    numbers = setup.component_numbers + torch.zeros_like(
        positions[..., 0], dtype=setup.component_numbers.dtype
    )
    # Reference is a lightweight view over immutable captured tensor tables;
    # the upstream weighting/C6 routines only read these values.
    reference = d3.reference.Reference(
        cn=setup.reference_cn,
        c6=setup.reference_c6,
    )
    cn = coordination_number(
        numbers,
        positions,
        counting_function=setup.counting_function,
        rcov=setup.rcov,
        cutoff=setup.cutoff,
    )
    weights = d3.model.weight_references(
        numbers, cn, reference, setup.weighting_function
    )
    c6 = _atomic_c6(numbers, weights, reference, chunk_size=chunk_size)
    return d3.disp.dispersion(
        numbers,
        positions,
        setup.as_param(),
        c6,
        setup.rvdw,
        setup.r4r2,
        setup.damping_function,
    )


class DispersionD3(Dispersion):
    """
    Representation of the DFT-D3(BJ) dispersion correction
    (:class:`.DispersionD3`).
    """

    @override
    def get_cache(
        self, numbers: Tensor, ihelp: IndexHelper | None = None, **kwargs: Any
    ) -> DispersionD3Cache:
        """
        Obtain cache for storage of settings.

        Settings can be passed as ``kwargs``. The available optional parameters
        are the same as in :func:`tad_dftd3.dftd3`, i.e., "ref", "rcov",
        "rvdw", and "r4r2".

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).

        Returns
        -------
        DispersionD3Cache
            Cache for the D3 dispersion.
        """
        ref = kwargs.pop(
            "ref",
            d3.reference.Reference(),
        ).to(**self.dd)

        rcov = kwargs.pop(
            "rcov",
            radii.COV_D3(**self.dd)[numbers],
        ).to(**self.dd)

        rvdw = kwargs.pop(
            "rvdw",
            radii.VDW_PAIRWISE(**self.dd)[
                numbers.unsqueeze(-1), numbers.unsqueeze(-2)
            ],
        ).to(**self.dd)

        r4r2 = kwargs.pop(
            "r4r2",
            d3.data.R4R2(**self.dd)[numbers],
        ).to(**self.dd)

        cutoff = kwargs.pop(
            "cutoff",
            torch.tensor(d3.defaults.D3_CN_CUTOFF, **self.dd),
        ).to(**self.dd)

        cf = kwargs.pop("counting_function", exp_count)
        wf = kwargs.pop("weighting_function", d3.model.gaussian_weight)
        df = kwargs.pop("damping_function", d3.damping.rational_damping)

        return DispersionD3Cache(ref, rcov, rvdw, r4r2, cutoff, cf, wf, df)

    @override
    def get_energy(
        self, positions: Tensor, cache: ComponentCache, **kwargs: Any
    ) -> Tensor:
        """
        Get D3 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        cache : ComponentCache
            Dispersion cache containing settings.

        Returns
        -------
        Tensor
            Atom-resolved D3 dispersion energy.
        """
        if isinstance(cache, DispersionD3CoreSetup):
            return dispersion_d3_energy(
                cache,
                positions,
                chunk_size=kwargs.pop("chunk_size", None),
            )

        if not isinstance(cache, DispersionD3Cache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'DispersionD3Cache'."
            )

        cn = coordination_number(
            self.numbers,
            positions,
            counting_function=cache.counting_function,
            rcov=cache.rcov,
            cutoff=cache.cutoff,
        )
        weights = d3.model.weight_references(
            self.numbers, cn, cache.ref, cache.weighting_function
        )

        chunk_size = kwargs.pop("chunk_size", None)
        c6 = _atomic_c6(self.numbers, weights, cache.ref, chunk_size=chunk_size)

        return d3.disp.dispersion(
            self.numbers,
            positions,
            self.param,
            c6,
            cache.rvdw,
            cache.r4r2,
            cache.damping_function,
        )
