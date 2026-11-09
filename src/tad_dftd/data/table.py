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
Data: Element tables
====================

Resolution of the per-element tables (indexed by atomic number) that the
element parameters ``rcov_table``, ``rvdw_table`` and ``r4r2_table`` are given as.
"""
from __future__ import annotations

from importlib.resources import as_file, files

import torch
from tad_mctc.data import radii, resolve_table
from tad_mctc.typing import TableFunction, Tensor

from .en import PAULING_EN
from .hardness import GAM
from .r4r2 import R4R2
from .radii import VDW_PAIRWISE_119
from .zeff import ZEFF

__all__ = [
    "TABLES",
    "element_table",
    "guard_numbers",
    "load_pkg_tensor",
    "validate_max_elem",
]


def _unwrap_functorch(x: Tensor) -> Tensor:
    while torch._C._functorch.is_functorch_wrapped_tensor(x):
        x = torch._C._functorch.get_unwrapped(x)
    return x


def guard_numbers(
    numbers: Tensor,
    invalid: Tensor,
    message: str,
    *,
    format_bad: bool = False,
) -> Tensor:
    """
    Validate atomic numbers eagerly (`ValueError`) and, when tracing under
    `torch.compile`, guard `numbers` with a graph-connected out-of-bounds
    table lookup so that invalid atomic numbers (including negative values and
    unsupported gap elements) deterministically fail at runtime instead of
    silently producing meaningless energies.
    """
    if torch.compiler.is_compiling():
        sentinel = torch.full_like(numbers, 1000)
        guard = torch.zeros(1, device=numbers.device, dtype=numbers.dtype)
        return (
            torch.where(invalid, sentinel, numbers)
            + guard[invalid.to(torch.long)]
        )

    raw_num = _unwrap_functorch(numbers)
    raw_inv = _unwrap_functorch(invalid)
    if bool(raw_inv.any()):
        if format_bad:
            bad = int(raw_num[raw_inv][0].item())
            raise ValueError(message.format(bad=bad))
        raise ValueError(message)
    return numbers


def validate_max_elem(numbers: Tensor, label: str, max_elem: int) -> Tensor:
    """
    Validate that all atomic `numbers` are in ``0..max_elem`` (zero is
    padding), see :func:`guard_numbers`. `label` names the model in the error.
    """
    invalid = (numbers < 0) | (numbers > max_elem)
    return guard_numbers(
        numbers,
        invalid,
        f"Unsupported atomic number {{bad}} for {label} dispersion model "
        f"(supported: Z in 1..{max_elem}).",
        format_bad=True,
    )


def load_pkg_tensor(package: str, filename: str) -> Tensor:
    """Load the tensor file `filename` that is shipped in `package` (the
    ``__package__`` of the caller), onto the CPU."""
    res = files(package).joinpath(filename)
    with as_file(res) as path:
        return torch.load(path, map_location="cpu", weights_only=True)


TABLES: dict[str, tuple[TableFunction, tuple[int, ...]]] = {
    "rcov_table": (radii.COV_D3, (119,)),
    "r4r2_table": (R4R2, (119,)),
    "rvdw_table": (VDW_PAIRWISE_119, (119, 119)),
    "en_table": (PAULING_EN, (119,)),
    "gam_table": (GAM, (119,)),
    "zeff_table": (ZEFF, (119,)),
}
"""Default and required shape of each per-element table, by argument name."""


def element_table(
    table: Tensor | TableFunction | None, name: str, like: Tensor
) -> Tensor:
    """
    Resolve a per-element table on the device and dtype of `like`.

    Tables built from a :class:`~tad_mctc.typing.TableFunction` (including
    the default) are cached per device and dtype by
    :func:`tad_mctc.data.resolve_table`, for as long as the function exists.

    A table that is passed must have the shape given in :data:`TABLES` for
    `name`, e.g. ``(119,)`` for ``r4r2_table``. Only the shape is checked,
    which does not break the graph under ``torch.compile``.

    Parameters
    ----------
    table : Tensor | TableFunction | None
        The table, or ``None`` for the default of `name`.
    name : str
        Name of the argument, a key of :data:`TABLES`.
    like : Tensor
        Tensor whose device and dtype the table is resolved to.

    Returns
    -------
    Tensor
        The table. Callers must treat it as read-only, since it may be
        shared through the cache.

    Raises
    ------
    ValueError
        If the table does not have the required shape.
    """
    if name == "rdamp_table":
        if table is None:
            return resolve_table(VDW_PAIRWISE_119, like)
        out = resolve_table(table, like)
        if out.ndim != 2 or out.shape[0] != out.shape[1] or out.shape[0] < 11:
            raise ValueError(
                f"'{name}' is a pairwise table indexed by atomic number, of "
                f"square shape (n, n), got shape {tuple(out.shape)}."
            )
        return out

    if name == "rvdw_table":
        if table is None:
            return resolve_table(VDW_PAIRWISE_119, like)
        out = resolve_table(table, like)
        if out.shape not in ((104, 104), (119, 119)):
            raise ValueError(
                f"'{name}' is a table indexed by atomic number, of shape "
                f"(104, 104) or (119, 119), not one value per atom or atom pair "
                f"(got shape {tuple(out.shape)})."
            )
        return out

    default, shape = TABLES[name]
    if table is None:
        return resolve_table(default, like)

    out = resolve_table(table, like)
    if out.shape != shape:
        raise ValueError(
            f"'{name}' is a table indexed by atomic number, of shape "
            f"{shape}, not one value per atom or atom pair (got shape "
            f"{tuple(out.shape)})."
        )

    return out
