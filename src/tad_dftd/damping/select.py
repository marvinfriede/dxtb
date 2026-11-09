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
Selecting a damping by name
===========================

The name of a damping variant, as in the data base of functionals, and the
two-body and three-body damping classes it stands for.
"""

from __future__ import annotations

from .base import Damping, ThreeBodyDamping, TwoBodyDamping
from .cso import CSOTwoBody
from .fermi import FermiTwoBody
from .koide import KoideThreeBody, KoideTwoBody
from .mzero import ModifiedZeroTwoBody
from .optimizedpower import OptimizedPowerTwoBody
from .rational import RationalThreeBody, RationalTwoBody
from .screened import ScreenedThreeBody, ScreenedTwoBody
from .z import ZTwoBody
from .zero import (
    ZeroProductThreeBody,
    ZeroThreeBodyD3,
    ZeroThreeBodyD4,
    ZeroTwoBody,
)

__all__ = [
    "THREE_BODY_DAMPINGS",
    "TWO_BODY_DAMPINGS",
    "damping_from_name",
    "get_damping_id",
    "identify_damping",
    "new_damping",
]

TWO_BODY_DAMPINGS: tuple[tuple[str, type[TwoBodyDamping]], ...] = (
    ("rational", RationalTwoBody),
    ("zero", ZeroTwoBody),
    ("mzero", ModifiedZeroTwoBody),
    ("optimizedpower", OptimizedPowerTwoBody),
    ("cso", CSOTwoBody),
    ("z", ZTwoBody),
    ("screened", ScreenedTwoBody),
    ("koide", KoideTwoBody),
    ("fermi", FermiTwoBody),
)
"""Names of the damping variants and their two-body damping."""

THREE_BODY_DAMPINGS: tuple[tuple[str, type[ThreeBodyDamping]], ...] = (
    ("zero_d3", ZeroThreeBodyD3),
    ("rational", RationalThreeBody),
    ("screened", ScreenedThreeBody),
    ("zero_d4", ZeroThreeBodyD4),
    ("zero_product", ZeroProductThreeBody),
    ("koide", KoideThreeBody),
)
"""Names of the three-body dampings."""

_TWO_BODY_ALIASES: dict[str, str] = {
    "bj": "rational",
    "bjm": "rational",
    "chg": "zero",
    "d1": "fermi",
    "d2": "fermi",
    "zerom": "mzero",
    "m-zero": "mzero",
    "op": "optimizedpower",
    "optimisedpower": "optimizedpower",
    "optpower": "optimizedpower",
    "opt-power": "optimizedpower",
    "optimized-power": "optimizedpower",
    "optimised-power": "optimizedpower",
    "tang_toennies": "screened",
    "z-damping": "z",
    "koide-bjm": "koide",
}

_THREE_BODY_ALIASES: dict[str, str] = {
    "bj": "rational",
    "zero": "zero_product",
    "zero_avg": "zero_d4",
    "zero-avg": "zero_d4",
    "d3_zero_avg": "zero_d3",
    "d3-zero-avg": "zero_d3",
}


def damping_from_name(
    name: str,
    three_body: bool | str = True,
    *,
    model: str = "d3",
) -> Damping:
    """
    The damping of a variant.

    Parameters
    ----------
    name : str
        The two-body damping, one of the names of
        :data:`TWO_BODY_DAMPINGS` (or an alias such as ``"bj"`` or ``"zerom"``).
    three_body : bool | str, optional
        The three-body damping: one of the names of
        :data:`THREE_BODY_DAMPINGS`, ``True`` for the default ATM damping of
        `model` (``"zero_d3"`` for D3/D3S, ``"zero_d4"`` for D4/D4S/D4Srev)
        or ``False`` for none. Defaults to ``True``.
    model : str, optional
        Model name used when ``three_body is True``. Defaults to ``"d3"``.

    Raises
    ------
    ValueError
        If a name is unknown.
    """
    clean_2b = name.lower().strip()
    norm_2b = _TWO_BODY_ALIASES.get(clean_2b, clean_2b)

    if three_body is True:
        from ..model.catalogue import get_model_metadata

        three_body = get_model_metadata(model).atm_damping

    three = None
    if three_body is not False and three_body is not None:
        clean_3b = three_body.lower().strip()
        norm_3b = _THREE_BODY_ALIASES.get(clean_3b, clean_3b)
        three = _lookup(THREE_BODY_DAMPINGS, norm_3b, "three-body damping")()

    return Damping(_lookup(TWO_BODY_DAMPINGS, norm_2b, "damping")(), three)


def new_damping(
    damping_2b: str,
    damping_3b: str | bool | None = False,
    *,
    model: str = "d3",
) -> Damping:
    """Construct a :class:`Damping` pair from 2-body and 3-body damping names."""
    tb: bool | str = False if damping_3b is None else damping_3b
    return damping_from_name(damping_2b, tb, model=model)


def get_damping_id(name: str, *, three_body: bool = False) -> str:
    """Return the canonical damping name."""
    clean = name.lower().strip()
    norm = (
        _THREE_BODY_ALIASES.get(clean, clean)
        if three_body
        else _TWO_BODY_ALIASES.get(clean, clean)
    )
    table = THREE_BODY_DAMPINGS if three_body else TWO_BODY_DAMPINGS
    _lookup(table, norm, "three-body damping" if three_body else "damping")
    return norm


def identify_damping(
    damping: TwoBodyDamping | ThreeBodyDamping,
) -> str | None:
    """
    Return the canonical catalogue name for a :class:`TwoBodyDamping` or
    :class:`ThreeBodyDamping` instance, or ``None`` for a custom subclass not
    inheriting from a built-in catalogue damping class.
    """
    if isinstance(damping, ThreeBodyDamping):
        for name_3b, cls_3b in THREE_BODY_DAMPINGS:
            if isinstance(damping, cls_3b):
                return name_3b
        return None
    for name_2b, cls_2b in TWO_BODY_DAMPINGS:
        if isinstance(damping, cls_2b):
            return name_2b
    return None


def _lookup(table: tuple[tuple[str, type], ...], name: str, what: str) -> type:
    """The class of `name` in `table`."""
    for key, cls in table:
        if key == name:
            return cls

    names = [key for key, _ in table]
    raise ValueError(f"Unknown {what} '{name}'; expected one of {names}.")
