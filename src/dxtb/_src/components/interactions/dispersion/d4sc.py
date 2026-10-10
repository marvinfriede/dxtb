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
Interactions: Self-consistent D4 Dispersion
===========================================

Self-consistent D4 dispersion correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tad_dftd4 as d4
import tad_dftd4.defaults as d4_defaults
import torch
from tad_mctc.data import PAULING
from tad_mctc.exceptions import DeviceError
from tad_mctc.math import einsum
from tad_mctc.typing import (
    DD,
    CountingFunction,
    Tensor,
    TensorLike,
    get_default_dtype,
    override,
)

from dxtb import IndexHelper
from dxtb._src.ncoord import coordination_number, erf_count
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import Slicers
from dxtb._src.utils.tensors import normalize_device

from ..base import Interaction, InteractionCache

__all__ = [
    "D4SCSetup",
    "DispersionD4SC",
    "DispersionD4SCCache",
    "LABEL_DISPERSIOND4SC",
    "build_d4sc_data",
    "new_d4sc",
    "setup_d4sc",
]


LABEL_DISPERSIOND4SC = "DispersionD4SC"
"""Label for the :class:`.DispersionD4SC` interaction, coinciding with the class name."""


@dataclass(frozen=True, eq=False)
class D4SCSetup:
    """Static numbers and parameter data for self-consistent D4."""

    numbers: Tensor
    parameters: tuple[tuple[str, Tensor | float | int | str], ...]
    rcov: Tensor
    r4r2: Tensor
    cn_cutoff: Tensor
    pair_weight: Tensor
    counting_function: CountingFunction
    model_type: type[d4.model.D4Model]
    model_ga: float
    model_gc: float
    model_wf: Tensor
    model_ref_charges: str
    model_rc6: Tensor
    model_extra: tuple[tuple[str, Any], ...] = ()

    def as_param(self) -> d4.Param:
        """Materialize the parameter mapping expected by tad-dftd4."""
        return d4.Param(**dict(self.parameters))


@dataclass(frozen=True, eq=False)
class _StaticMapping:
    items: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True, eq=False)
class _StaticList:
    items: tuple[Any, ...]


def _clone_static(value: Any) -> Any:
    """Clone tensor-valued static model data without cutting its graph."""
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_static(item) for item in value)
    if isinstance(value, list):
        return _StaticList(tuple(_clone_static(item) for item in value))
    if isinstance(value, dict):
        return _StaticMapping(
            tuple((key, _clone_static(item)) for key, item in value.items())
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "Custom D4Model state must contain only tensors and immutable scalar "
        f"values; received {type(value).__name__}."
    )


def _extra_model_state(model: d4.model.D4Model) -> tuple[tuple[str, Any], ...]:
    """Capture immutable extension state added by a D4Model subclass."""
    base_fields = {"numbers", "ga", "gc", "wf", "ref_charges", "rc6"}
    state: dict[str, Any] = {}
    if hasattr(model, "__dict__"):
        state.update(
            {
                key: value
                for key, value in vars(model).items()
                if key not in base_fields
            }
        )

    for cls in type(model).__mro__:
        if cls is d4.model.D4Model or cls is d4.model.BaseModel:
            break
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in base_fields and hasattr(model, name):
                state[name] = getattr(model, name)

    return tuple((key, _clone_static(value)) for key, value in state.items())


def setup_d4sc(component: DispersionD4SC, numbers: Tensor) -> D4SCSetup:
    """Gather D4SC parameter and model tables for one System."""
    en = PAULING(device=numbers.device, dtype=component.rcov.dtype)[numbers]
    endiff = torch.abs(en.unsqueeze(-2) - en.unsqueeze(-1))
    pair_weight = d4_defaults.D4_K4 * torch.exp(
        -((endiff + d4_defaults.D4_K5) ** 2.0) / d4_defaults.D4_K6
    )

    model = component.model
    param = component.param
    parameters = tuple(
        (name, _clone_static(param[name]))
        for name in ("a1", "a2", "s6", "s8", "s9", "s10")
    )

    return D4SCSetup(
        numbers=numbers.clone(),
        parameters=parameters,
        rcov=component.rcov.clone(),
        r4r2=component.r4r2.clone(),
        cn_cutoff=component.cutoff.cn.clone(),
        pair_weight=pair_weight,
        counting_function=component.counting_function,
        model_type=type(model),
        model_ga=model.ga,
        model_gc=model.gc,
        model_wf=model.wf.clone(),
        model_ref_charges=model.ref_charges,
        model_rc6=model.rc6.clone(),
        model_extra=_extra_model_state(model),
    )


def _build_d4sc_model(setup: D4SCSetup) -> d4.model.D4Model:
    """Construct a model object owned by one call-data value."""
    try:
        model = setup.model_type(
            setup.numbers.clone(),
            ga=setup.model_ga,
            gc=setup.model_gc,
            wf=setup.model_wf.clone(),
            ref_charges=setup.model_ref_charges,
            rc6=setup.model_rc6.clone(),
            device=setup.numbers.device,
            dtype=setup.model_wf.dtype,
        )
    except TypeError as exc:
        raise TypeError(
            "A custom D4Model used by D4SC must accept the standard D4Model "
            "constructor arguments to support call-local evaluation."
        ) from exc

    for name, value in setup.model_extra:
        setattr(model, name, _restore_static(value))
    return model


def _restore_static(value: Any) -> Any:
    """Rebuild mutable values owned by the call-local external model."""
    if isinstance(value, _StaticMapping):
        return {key: _restore_static(item) for key, item in value.items}
    if isinstance(value, _StaticList):
        return [_restore_static(item) for item in value.items]
    if isinstance(value, tuple):
        return tuple(_restore_static(item) for item in value)
    if isinstance(value, Tensor):
        return value.clone()
    return value


def build_d4sc_data(setup: D4SCSetup, positions: Tensor) -> DispersionD4SCCache:
    """Build fresh geometry-dependent D4SC data for one evaluation."""
    numbers = setup.numbers
    cn = coordination_number(
        numbers,
        positions,
        counting_function=setup.counting_function,
        rcov=setup.rcov,
        cutoff=setup.cn_cutoff,
        pair_weight=setup.pair_weight,
    )
    edisp = d4.dispersion.dispersion2(
        numbers,
        positions,
        setup.as_param(),
        torch.ones(
            (*numbers.shape, numbers.shape[-1]),
            device=positions.device,
            dtype=positions.dtype,
        ),
        setup.r4r2,
        as_matrix=True,
    )
    model = _build_d4sc_model(setup)
    dispmat = edisp.unsqueeze(-1).unsqueeze(-1) * model.rc6
    return DispersionD4SCCache(cn, dispmat, model)


class DispersionD4SCCache(InteractionCache, TensorLike):
    """
    Restart data for the :class:`.DispersionD4SC` interaction.

    Note
    ----
    The dispersion parameters (a1, a2, ...) are given in the dispersion
    class constructor.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    cn: Tensor
    """Coordination number of every atom."""

    dispmat: Tensor
    """
    Dispersion matrix. This quantity is almost equal to the dispersion energy,
    except for multiplication with C6 and C8.
    """

    model: d4.model.D4Model
    """Fresh model instance owned by this one call-data object."""

    __slots__ = ["__store", "cn", "dispmat", "model"]

    def __init__(
        self,
        cn: Tensor,
        dispmat: Tensor,
        model: d4.model.D4Model,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else cn.device,
            dtype=dtype if dtype is None else cn.dtype,
        )

        self.cn = cn
        self.dispmat = dispmat

        # The model is constructed for this call-data object only.
        self.model = model

        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        cn: Tensor
        """Coordination number of every atom."""

        dispmat: Tensor
        """
        Dispersion matrix. This quantity is almost equal to the dispersion
        energy, except for multiplication with C6 and C8.
        """

        def __init__(
            self, cn: Tensor, dispmat: Tensor, model: d4.model.D4Model
        ) -> None:
            self.cn = cn
            self.dispmat = dispmat

            # Store the call-local numbers view before culling.
            self.numbers = model.numbers

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(self.cn, self.dispmat, self.model)

        slicer = slicers["atom"]
        self.cn = self.cn[tuple([~conv, *slicer])]
        self.dispmat = self.dispmat[tuple([~conv, *slicer, *slicer])]

        self.model.numbers = self.model.numbers[tuple([~conv, *slicer])]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.cn = self.__store.cn
        self.dispmat = self.__store.dispmat
        self.model.numbers = self.__store.numbers


class DispersionD4SC(Interaction):
    """
    Self-consistent D4 dispersion correction (:class:`.DispersionD4SC`).
    """

    param: d4.Param
    """Dispersion parameters."""

    model: d4.model.D4Model
    """Model for the D4 dispersion correction."""

    rcov: Tensor
    """Covalent radii of all atoms."""

    r4r2: Tensor
    """R4/R2 ratio of all atoms."""

    cutoff: d4.cutoff.Cutoff
    """Real-space cutoff for the D4 dispersion correction."""

    counting_function: CountingFunction
    """
    Counting function for the coordination number.

    :default: :func:`tad_mctc.ncoord.erf_count`
    """

    damping_function: d4.damping.Damping
    """
    Damping function for the dispersion correction.

    :default: :func:`d4.damping.RationalDamping`
    """

    __slots__ = [
        "param",
        "model",
        "rcov",
        "r4r2",
        "cutoff",
        "counting_function",
        "damping_function",
    ]

    def __init__(
        self,
        param: d4.Param,
        model: d4.model.D4Model,
        rcov: Tensor,
        r4r2: Tensor,
        cutoff: d4.cutoff.Cutoff,
        counting_function: CountingFunction = erf_count,
        damping_function: d4.damping.Damping = d4.damping.RationalDamping(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.param = param
        self.model = model
        self.rcov = rcov
        self.r4r2 = r4r2
        self.cutoff = cutoff
        self.counting_function = counting_function
        self.damping_function = damping_function

    def update(self, **kwargs: Any) -> None:
        """Reject mutation of setup-derived exact D4SC terms."""
        if type(self) is DispersionD4SC:
            raise RuntimeError(
                "Exact DispersionD4SC parameters are setup-derived. "
                "Create a new Model/System with changed parameters."
            )
        super().update(**kwargs)

    def reset(self) -> None:
        """Reject resetting setup-derived exact D4SC terms."""
        if type(self) is DispersionD4SC:
            raise RuntimeError(
                "Exact DispersionD4SC parameters are setup-derived. "
                "Create a new Model/System with changed parameters."
            )
        super().reset()

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
        **_,
    ) -> DispersionD4SCCache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        DispersionD4SCCache
            Restart data for the interaction.

        Note
        ----
        If the :class:`.DispersionD4SC` interaction is evaluated within the
        :class:`dxtb.components.InteractionList`, ``positions`` will be passed
        as an argument, too. Hence, it is necessary to absorb the ``positions``
        in the signature of the function (also see
        :meth:`dxtb.components.Interaction.get_cache`).
        """
        if numbers is None:
            raise ValueError(
                "Atomic numbers are required for DispersionD4SC data."
            )
        if positions is None:
            raise ValueError("Positions are required for D4SC data.")
        return build_d4sc_data(setup_d4sc(self, numbers), positions)

    @override
    def get_monopole_atom_energy(
        self, cache: InteractionCache, qat: Tensor, **_: Any
    ) -> Tensor:
        """
        Calculate the D4 dispersion correction energy.

        Parameters
        ----------
        cache : DispersionD4SCCache
            Restart data for the interaction.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise D4 dispersion correction energies.
        """
        if not isinstance(cache, DispersionD4SCCache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'DispersionD4SCCache'."
            )

        # `numbers` in model are updated in cache (for culling)
        weights = cache.model.weight_references(cache.cn, qat)

        return 0.5 * einsum(
            "...ijab,...ia,...jb->...j",
            *(cache.dispmat, weights, weights),
            optimize=[(0, 1), (0, 1)],
        )

    @override
    def get_monopole_atom_potential(
        self, cache: InteractionCache, qat: Tensor, *_: Any, **__: Any
    ) -> Tensor:
        """
        Calculate the D4 dispersion correction potential.

        Parameters
        ----------
        cache : DispersionD4SCCache
            Restart data for the interaction.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise dispersion correction potential.
        """
        if not isinstance(cache, DispersionD4SCCache):
            raise TypeError(
                f"Cache in {self.label} is not of type 'DispersionD4SCCache'."
            )

        weights, dgwdq = cache.model.weight_references(
            cache.cn, qat, with_dgwdq=True
        )

        return einsum(
            "...ijab,...jb,...ia->...i", cache.dispmat, weights, dgwdq
        )


def new_d4sc(
    numbers: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DispersionD4SC | None:
    """
    Create new instance of :class:`.DispersionD4SC`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.

    Returns
    -------
    DispersionD4SC | None
        Instance of the :class:`.DispersionD4SC` class or ``None`` if
        no :class:`.DispersionD4SC` is used.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    # compatibility with previous version based on `Param`
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

    if "dispersion" not in par or par.is_none("dispersion"):
        return None

    if par.is_none("dispersion.d4"):
        return None

    if par.is_false("dispersion.d4.sc"):
        return None

    if device is not None:
        if normalize_device(device) != numbers.device:
            raise DeviceError(
                f"Passed device ({device}) and device of `numbers` tensor "
                f"({numbers.device}) do not match."
            )

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    param = d4.Param(
        **{
            "a1": par.get("dispersion.d4.a1"),
            "a2": par.get("dispersion.d4.a2"),
            "s6": par.get("dispersion.d4.s6"),
            "s8": par.get("dispersion.d4.s8"),
            "s9": par.get("dispersion.d4.s9"),
            "s10": par.get("dispersion.d4.s10"),
        }
    )

    rcov = d4.data.COV_D3(**dd)[numbers]
    r4r2 = d4.data.R4R2(**dd)[numbers]
    model = d4.model.D4Model(numbers, ref_charges="gfn2", **dd)
    cutoff = d4.cutoff.Cutoff(disp2=50.0, disp3=25.0, **dd)

    return DispersionD4SC(
        param, model=model, rcov=rcov, r4r2=r4r2, cutoff=cutoff, **dd
    )
