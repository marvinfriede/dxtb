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
Utility: Tensor Ops
===================

Collection of utility functions for matrices/tensors.
"""

from __future__ import annotations

import weakref
from typing import Optional, Tuple

import torch
from tad_mctc.autograd.checks import is_gradtracking

from dxtb._src.typing import Tensor

__all__ = [
    "t2int",
    "tensor_id",
    "GradKey",
    "grad_key",
    "grad_key_matches",
    "normalize_device",
]


def t2int(x: Tensor) -> int:
    """
    Convert tensor to int.

    Parameters
    ----------
    x : Tensor
        Tensor to convert.

    Returns
    -------
    int
        Integer value of the tensor.
    """
    return int(x.item())


def tensor_id(x: Tensor) -> str:
    """
    Generate an identifier for a tensor based on its data pointer and version.
    """
    grad = int(x.requires_grad)
    v = x._version
    dtype = x.dtype

    # Data pointer easily accessible without functorch
    if not is_gradtracking(x):
        return f"tensor(ptr={x.data_ptr()},v={v},grad={grad},dtype={dtype})"

    # functorch dual tensors have no storage
    if is_batched(x):
        return f"batched_tensor(id={id(x)},v={v},grad={grad},dtype={dtype})"

    # Peel off gradtracking layers to get to the underlying tensor
    while is_gradtracking(x):
        x = torch._C._functorch.get_unwrapped(x)

        if is_batched(x):
            return f"batched_tensor(id={id(x)},v={v},grad={grad},dtype={dtype})"

    return f"tensor(ptr={x.data_ptr()},v={v},grad={grad},dtype={dtype})"


GradKey = Tuple[Tuple[bool, Optional["weakref.ReferenceType[Tensor]"]], ...]
"""Gradient-tracking state of the tensors a cache was built from."""


def grad_key(*tensors: Tensor) -> GradKey:
    """
    Record the gradient-tracking state of the tensors a cache is built from.

    Caches that are validated by comparing tensor *values* return results
    whose autograd graph belongs to the tensors of the call that built the
    cache. Such a result is only valid for a later call if that call's
    tensors have the same ``requires_grad`` flag and, if they require
    gradients, are the very same tensor objects.

    Stopgap for the stale-graph bug (T0.5 in ``docs/plan``); removed with the
    caches.

    Parameters
    ----------
    *tensors : Tensor
        Tensors the cache depends on.

    Returns
    -------
    GradKey
        Per tensor, the ``requires_grad`` flag and, if set, a weak reference
        to the tensor.
    """
    return tuple(
        (t.requires_grad, weakref.ref(t) if t.requires_grad else None)
        for t in tensors
    )


def grad_key_matches(key: GradKey | None, *tensors: Tensor) -> bool:
    """
    Check whether a cache built for ``key`` may be reused for ``tensors``.

    Parameters
    ----------
    key : GradKey | None
        Key recorded with :func:`grad_key` when the cache was built.
    *tensors : Tensor
        Tensors of the current call (same order as for :func:`grad_key`).

    Returns
    -------
    bool
        ``False`` if no key was recorded, if a ``requires_grad`` flag differs,
        or if a tensor that requires gradients is not the recorded object.
    """
    if key is None or len(key) != len(tensors):
        return False

    for (flag, ref), t in zip(key, tensors):
        if flag != t.requires_grad:
            return False
        if ref is not None and ref() is not t:
            return False

    return True


def normalize_device(
    device: torch.device | str | int | None,
) -> torch.device | None:
    """
    Convert a device specification to the :class:`torch.device` that tensors
    created on it report.

    Strings (``"cpu"``) and CUDA devices without index (``"cuda"``) do not
    compare equal to ``tensor.device`` (``torch.device("cpu")``,
    ``torch.device("cuda:0")``), which made device checks fail.

    Parameters
    ----------
    device : torch.device | str | int | None
        Device specification. ``None`` is returned unchanged.

    Returns
    -------
    torch.device | None
        Normalized device (an invalid specification is returned unchanged).
    """
    if device is None:
        return None

    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError):
        # invalid specification: leave it to the caller's device check
        return device  # type: ignore[return-value]

    if dev.type == "cuda" and dev.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        dev = torch.device("cuda", index)
    return dev
