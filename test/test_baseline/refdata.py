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
Storage and comparison of reference data (T0.2).

Deliberately independent of the dxtb API: a reference file is a plain
``.npz`` archive of NumPy arrays plus a JSON metadata record, and the
comparison works on plain arrays. Only ``compute.py`` talks to dxtb, so this
module survives every API change.

File layout: ``reference/<method>-<driver>/<system>.npz``. Array keys are
the quantity names (``energy``, ``forces``, ``energy.ES2``, ...); the
inputs are stored under ``input.*``; the metadata is the JSON string under
``__meta__``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "REFERENCE_DIR",
    "Tolerance",
    "Deviation",
    "reference_path",
    "save_reference",
    "load_reference",
    "compare",
    "assert_close",
]

REFERENCE_DIR = Path(__file__).parent / "reference"
"""Directory of the reference files."""

META_KEY = "__meta__"
INPUT_PREFIX = "input."


@dataclass(frozen=True)
class Tolerance:
    """Absolute and relative tolerance: ``|a - b| <= atol + rtol * |b|``."""

    atol: float
    rtol: float = 0.0


@dataclass(frozen=True)
class Deviation:
    """Result of comparing one quantity."""

    key: str
    max_abs: float
    max_rel: float
    tol: Tolerance
    ok: bool
    message: str = ""

    def __str__(self) -> str:
        status = "ok  " if self.ok else "FAIL"
        msg = f" ({self.message})" if self.message else ""
        return (
            f"{status} {self.key:<28s} max_abs={self.max_abs:.3e} "
            f"max_rel={self.max_rel:.3e} atol={self.tol.atol:.1e} "
            f"rtol={self.tol.rtol:.1e}{msg}"
        )


def reference_path(method: str, driver: str, system: str) -> Path:
    """Path of the reference file for a method, driver and system."""
    return REFERENCE_DIR / f"{method}-{driver}" / f"{system}.npz"


def save_reference(
    path: Path,
    values: Mapping[str, np.ndarray],
    inputs: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> None:
    """
    Write a reference file.

    Parameters
    ----------
    path : Path
        Target file (``.npz``).
    values : Mapping[str, np.ndarray]
        Computed quantities.
    inputs : Mapping[str, np.ndarray]
        Inputs of the calculation (numbers, positions, charge, spin).
    meta : Mapping[str, Any]
        JSON-serializable metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in values.items()}
    arrays.update({INPUT_PREFIX + k: np.asarray(v) for k, v in inputs.items()})
    arrays[META_KEY] = np.array(json.dumps(meta, indent=1, sort_keys=True))
    np.savez_compressed(path, **arrays)


def load_reference(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """
    Read a reference file.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]
        Quantities, inputs and metadata.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data[META_KEY]))
        values, inputs = {}, {}
        for key in data.files:
            if key == META_KEY:
                continue
            if key.startswith(INPUT_PREFIX):
                inputs[key[len(INPUT_PREFIX) :]] = data[key]
            else:
                values[key] = data[key]
    return values, inputs, meta


def _deviation(key: str, ref: np.ndarray, new: np.ndarray, tol: Tolerance):
    ref = np.asarray(ref, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    if ref.shape != new.shape:
        return Deviation(
            key, np.inf, np.inf, tol, False, f"shape {new.shape} != {ref.shape}"
        )
    if not np.all(np.isfinite(new)):
        return Deviation(key, np.inf, np.inf, tol, False, "not finite")
    if ref.size == 0:
        return Deviation(key, 0.0, 0.0, tol, True)

    diff = np.abs(new - ref)
    max_abs = float(diff.max())
    scale = np.abs(ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, diff / scale, np.where(diff > 0, np.inf, 0))
    max_rel = float(rel.max())
    ok = bool(np.all(diff <= tol.atol + tol.rtol * scale))
    return Deviation(key, max_abs, max_rel, tol, ok)


def compare(
    ref: Mapping[str, np.ndarray],
    new: Mapping[str, np.ndarray],
    tolerances: Mapping[str, Tolerance],
    keys: Iterable[str] | None = None,
) -> list[Deviation]:
    """
    Compare quantities against a reference.

    Parameters
    ----------
    ref : Mapping[str, np.ndarray]
        Reference quantities.
    new : Mapping[str, np.ndarray]
        Recomputed quantities.
    tolerances : Mapping[str, Tolerance]
        Tolerance per quantity. A key ``energy.X`` falls back to ``energy``;
        a missing tolerance is an error.
    keys : Iterable[str] | None, optional
        Quantities to compare. Defaults to all keys of ``new``.

    Returns
    -------
    list[Deviation]
        One entry per compared quantity.
    """
    out = []
    for key in new.keys() if keys is None else keys:
        tol = tolerances.get(key, tolerances.get(key.split(".")[0]))
        if tol is None:
            raise KeyError(f"No tolerance for '{key}'.")
        if key not in ref:
            out.append(Deviation(key, np.inf, np.inf, tol, False, "missing"))
            continue
        if key not in new:
            out.append(
                Deviation(key, np.inf, np.inf, tol, False, "not computed")
            )
            continue
        out.append(_deviation(key, ref[key], new[key], tol))
    return out


def assert_close(deviations: list[Deviation], header: str = "") -> None:
    """Raise an ``AssertionError`` listing all quantities if one fails."""
    if all(d.ok for d in deviations):
        return
    lines = [header] if header else []
    lines += [str(d) for d in deviations]
    raise AssertionError("\n".join(lines))
