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
Functional Parameter Lookup
===========================

Look up damping parameters by density functional name for D1, D2, D3/D3S, and
D4/D4S dispersion models, resolving functional aliases from ``s-dftd3`` and
Grimme-lab ``dftd/param/functionals.f90``.
"""
from __future__ import annotations

import os.path as op
from typing import Any

import torch
from tad_mctc.typing import DD, Tensor, get_default_device, get_default_dtype

try:
    import tomllib  # type: ignore[import-not-found,unused-ignore]  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef,unused-ignore]

__all__ = ["get_functional_params", "get_params"]

_META_KEYS = ("mbd", "damping", "doi")

_d3_parameters: dict[str, Any] | None = None
_d4_parameters: dict[str, Any] | None = None
_aliases: dict[str, str] | None = None

# D1 functional parameters from dftd/param/d1.f90 (S. Grimme, JCC 25, 1463 (2004))
_D1_PARAMETERS: dict[str, dict[str, Any]] = {
    "default": {"s6": 1.0, "rs6": 1.0, "alp": 23.0, "damping": "fermi"},
    "blyp": {"s6": 1.4, "rs6": 1.0, "alp": 23.0, "damping": "fermi"},
    "bp": {"s6": 1.3, "rs6": 1.0, "alp": 23.0, "damping": "fermi"},
    "pbe": {"s6": 0.7, "rs6": 1.0, "alp": 23.0, "damping": "fermi"},
}

# D2 functional parameters from dftd/param/d2.f90 (S. Grimme, JCC 27, 1787 (2006))
_D2_PARAMETERS: dict[str, dict[str, Any]] = {
    "default": {"s6": 1.0, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "blyp": {"s6": 1.2, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "bp": {"s6": 1.05, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "b97d": {"s6": 1.25, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "revpbe": {"s6": 1.25, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "pbe": {"s6": 0.75, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "tpss": {"s6": 1.0, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "b3lyp": {"s6": 1.05, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "pbe0": {"s6": 0.6, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "pw6b95": {"s6": 0.5, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "tpss0": {"s6": 0.85, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "b2plyp": {"s6": 0.55, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "b2gpplyp": {"s6": 0.4, "rs6": 1.1, "alp": 20.0, "damping": "fermi"},
    "dsdblyp": {"s6": 0.41, "rs6": 1.1, "alp": 60.0, "damping": "fermi"},
}

# Additional D4 parameters from dftd/param/d4.f90 not in assets/parameters.toml
_D4_EXTRA_PARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    "dftb_3ob": {
        "bj-eeq-two": {"s6": 1.0, "s8": 0.4727337, "a1": 0.5467502, "a2": 4.4955068},
        "bj-eeq-atm": {"s6": 1.0, "s8": 0.6635015, "a1": 0.5523240, "a2": 4.3537076},
    },
    "dftb_matsci": {
        "bj-eeq-two": {"s6": 1.0, "s8": 2.7711819, "a1": 0.4681712, "a2": 5.2918629},
        "bj-eeq-atm": {"s6": 1.0, "s8": 3.3157614, "a1": 0.4826330, "a2": 5.3811976},
    },
    "dftb_mio": {
        "bj-eeq-two": {"s6": 1.0, "s8": 1.1948145, "a1": 0.6074567, "a2": 4.9336133},
        "bj-eeq-atm": {"s6": 1.0, "s8": 1.2916225, "a1": 0.5965326, "a2": 4.8778602},
    },
    "dftb_ob2": {
        "bj-eeq-two": {"s6": 1.0, "s8": 2.7611320, "a1": 0.6037249, "a2": 5.3900004},
        "bj-eeq-atm": {"s6": 1.0, "s8": 2.9692689, "a1": 0.6068916, "a2": 5.4476789},
    },
    "dftb_pbc": {
        "bj-eeq-two": {"s6": 1.0, "s8": 1.7303734, "a1": 0.5546548, "a2": 4.7973454},
        "bj-eeq-atm": {"s6": 1.0, "s8": 2.1667394, "a1": 0.5646391, "a2": 4.9576353},
    },
}


def _load_d3_parameters() -> dict[str, Any]:
    global _d3_parameters
    if _d3_parameters is None:
        path = op.join(op.dirname(__file__), "d3_parameters.toml")
        with open(path, "rb") as fh:
            _d3_parameters = tomllib.load(fh)
    return _d3_parameters


def _load_d4_parameters() -> dict[str, Any]:
    global _d4_parameters
    if _d4_parameters is None:
        path = op.join(op.dirname(__file__), "d4_parameters.toml")
        with open(path, "rb") as fh:
            _d4_parameters = tomllib.load(fh)
    return _d4_parameters


def _load_aliases() -> dict[str, str]:
    global _aliases
    if _aliases is None:
        path = op.join(op.dirname(__file__), "aliases.toml")
        with open(path, "rb") as fh:
            _aliases = dict(tomllib.load(fh)["alias"])
    assert _aliases is not None
    return _aliases


def _normalize(functional: str) -> str:
    """Normalize a functional name: lowercase and strip hyphens."""
    return functional.lower().strip().replace("-", "")


def _merge(
    entry: dict[str, Any],
    base: dict[str, Any],
    preference: list[str],
    keep_meta: bool,
) -> dict[str, Any]:
    for variant in preference:
        if variant not in entry:
            continue
        merged = {**base.get(variant, {}), **entry[variant]}
        if not keep_meta:
            for key in _META_KEYS:
                merged.pop(key, None)
        return merged

    raise ValueError(
        f"none of the requested damping variants {preference!r} are available "
        "for this functional"
    )


def _to_tensor_dict(
    merged: dict[str, Any],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    dd: DD = {
        "dtype": dtype if dtype is not None else get_default_dtype(),
        "device": device if device is not None else get_default_device(),
    }
    params: dict[str, Any] = {}
    for key, value in merged.items():
        if isinstance(value, (int, float)):
            params[key] = torch.tensor(float(value), **dd)
        else:
            params[key] = value
    return params


def get_params(
    functional: str,
    *,
    model: str = "d3",
    damping: str | list[str] | None = None,
    keep_meta: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """
    Look up damping parameters for a density functional and dispersion model.

    Parameters
    ----------
    functional : str
        Name of the density functional (case- and hyphen-insensitive).
    model : str, optional
        Dispersion model name (``"d1"``, ``"d2"``, ``"d3"``, ``"d3s"``, ``"d4"``,
        ``"d4s"``). Defaults to ``"d3"``.
    damping : str | list[str] | None, optional
        Damping variant(s) to look up.
    keep_meta : bool, optional
        Keep ``mbd``, ``damping``, and ``doi`` metadata keys. Defaults to
        ``False``.
    dtype : torch.dtype | None, optional
        Floating-point precision for returned tensors.
    device : torch.device | None, optional
        Device for returned tensors.

    Returns
    -------
    dict[str, Tensor]
        Damping parameters.
    """
    aliases = _load_aliases()
    normalized = _normalize(functional)
    canonical = aliases.get(normalized, normalized)
    m = model.lower().strip()

    if m == "d1":
        if canonical not in _D1_PARAMETERS:
            raise KeyError(
                f"unknown D1 functional {functional!r} (normalized: {canonical!r})"
            )
        if damping is not None:
            variants = [damping] if isinstance(damping, str) else damping
            if not any(v.lower() in ("fermi", "d1") for v in variants):
                raise ValueError(
                    f"none of the requested damping variants {variants!r} are "
                    "available for D1"
                )
        merged = dict(_D1_PARAMETERS[canonical])
        if not keep_meta:
            for key in _META_KEYS:
                merged.pop(key, None)
        return _to_tensor_dict(merged, dtype=dtype, device=device)

    if m == "d2":
        if canonical not in _D2_PARAMETERS:
            raise KeyError(
                f"unknown D2 functional {functional!r} (normalized: {canonical!r})"
            )
        if damping is not None:
            variants = [damping] if isinstance(damping, str) else damping
            if not any(v.lower() in ("fermi", "d2") for v in variants):
                raise ValueError(
                    f"none of the requested damping variants {variants!r} are "
                    "available for D2"
                )
        merged = dict(_D2_PARAMETERS[canonical])
        if not keep_meta:
            for key in _META_KEYS:
                merged.pop(key, None)
        return _to_tensor_dict(merged, dtype=dtype, device=device)

    if m in ("d3", "d3s"):
        data = _load_d3_parameters()
        try:
            entry = data["parameter"][canonical]["d3"]
        except KeyError as e:
            raise KeyError(
                f"unknown functional {functional!r} (normalized: {canonical!r})"
            ) from e

        if damping is None:
            preference = list(data["default"]["d3"])
        elif isinstance(damping, str):
            d_low = damping.lower().strip()
            if d_low == "rational":
                preference = ["bj"]
            elif d_low == "mzero":
                preference = ["zerom"]
            elif d_low in ("optimizedpower", "optimisedpower"):
                preference = ["op"]
            else:
                preference = [d_low]
        else:
            preference = list(damping)

        base = data["default"]["parameter"]["d3"]
        merged = _merge(entry, base, preference, keep_meta)
        return _to_tensor_dict(merged, dtype=dtype, device=device)

    if m in ("d4", "d4s"):
        data = _load_d4_parameters()
        if canonical in data["parameter"] and "d4" in data["parameter"][canonical]:
            entry = dict(data["parameter"][canonical]["d4"])
        elif canonical in _D4_EXTRA_PARAMETERS:
            entry = dict(_D4_EXTRA_PARAMETERS[canonical])
        else:
            raise KeyError(
                f"unknown D4 functional {functional!r} (normalized: {canonical!r})"
            )

        if damping is None:
            preference = list(data["default"]["d4"])
        elif isinstance(damping, str):
            d_low = damping.lower().strip()
            if d_low in ("bj", "rational"):
                preference = ["bj-eeq-atm", "bj-eeq-two"]
            else:
                preference = [d_low]
        else:
            preference = list(damping)

        base = data["default"]["parameter"]["d4"]
        merged = _merge(entry, base, preference, keep_meta)
        return _to_tensor_dict(merged, dtype=dtype, device=device)

    raise ValueError(
        f"No functional damping parameter tables are available for model '{model}'."
    )


def get_functional_params(
    functional: str,
    damping: str | list[str] | None = None,
    keep_meta: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    *,
    model: str = "d3",
) -> dict[str, Tensor]:
    """
    Look up damping parameters for a functional by name (compatible with
    :func:`tad_dftd3.param.get_functional_params`).
    """
    return get_params(
        functional,
        model=model,
        damping=damping,
        keep_meta=keep_meta,
        dtype=dtype,
        device=device,
    )
