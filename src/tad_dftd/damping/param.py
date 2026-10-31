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
Damping parameters
==================

The parameters of a functional, as one set shared by the two- and the
three-body damping, like ``param_type`` of ``dftd``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import Tensor

__all__ = ["DampingParam", "Param", "ParamInput", "as_damping_param"]


class DampingParam(Node):
    """
    Damping parameters, a set shared by the two- and three-body damping.

    A frozen :class:`~tad_mctc.tree.Node`: a tensor field is a pytree leaf,
    which can be differentiated or batched with ``torch.func``, and a Python
    number or ``None`` is static. A field that is ``None`` is not set: a
    damping that has a default for it uses that, one that needs it raises
    a :class:`ValueError` when it reads it.
    """

    s6: Tensor | float | None = child(default=None)
    s8: Tensor | float | None = child(default=None)
    s9: Tensor | float | None = child(default=None)
    s10: Tensor | float | None = child(default=None)
    a1: Tensor | float | None = child(default=None)
    a2: Tensor | float | None = child(default=None)
    a3: Tensor | float | None = child(default=None)
    a4: Tensor | float | None = child(default=None)
    a13: Tensor | float | None = child(default=None)
    a23: Tensor | float | None = child(default=None)
    rs6: Tensor | float | None = child(default=None)
    rs8: Tensor | float | None = child(default=None)
    rs9: Tensor | float | None = child(default=None)
    alp: Tensor | float | None = child(default=None)
    alp3: Tensor | float | None = child(default=None)
    bet: Tensor | float | None = child(default=None)
    sxc: Tensor | float | None = child(default=None)
    rsxc: Tensor | float | None = child(default=None)
    damping: str | None = context(default=None)

    def get(self, name: str, default: Any = None) -> Any:
        """Dictionary-compatible ``get`` accessor."""
        if hasattr(self, name):
            val = getattr(self, name)
            return default if val is None else val
        return default

    def __contains__(self, name: object) -> bool:
        """Dictionary-compatible ``in`` check for non-None parameters."""
        if not isinstance(name, str) or not hasattr(self, name):
            return False
        return getattr(self, name) is not None

    def __getitem__(self, name: str) -> Any:
        """Dictionary-compatible item lookup."""
        if hasattr(self, name):
            val = getattr(self, name)
            if val is not None:
                return val
        raise KeyError(name)

    @classmethod
    def from_functional(
        cls,
        functional: str,
        damping: str | list[str] | None = None,
        atm: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        *,
        model: str = "d3",
    ) -> DampingParam:
        """
        Load the parameters of a functional.
        """
        from ..param import get_functional_params

        values: dict[str, Any] = {
            **get_functional_params(
                functional,
                damping=damping,
                keep_meta=True,
                dtype=dtype,
                device=device,
                model=model,
            )
        }
        for key in ("mbd", "doi", "citation"):
            values.pop(key, None)
        if not atm:
            values.pop("s9", None)
        return cls(**values)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> DampingParam:
        """
        Build a :class:`DampingParam` from a mapping, ignoring metadata keys
        such as ``"doi"``, ``"citation"``, or ``"mbd"``.
        """
        fields = set(cls._child_names) | set(cls._context_names)
        kwargs: dict[str, Any] = {}
        for key, val in values.items():
            if key not in fields or val is None:
                continue
            if key in cls._context_names:
                kwargs[key] = val
            elif isinstance(val, Tensor):
                if device is not None or dtype is not None:
                    kwargs[key] = val.to(device=device, dtype=dtype)
                else:
                    kwargs[key] = val
            elif device is not None or dtype is not None:
                kwargs[key] = torch.tensor(val, device=device, dtype=dtype)
            else:
                kwargs[key] = val
        return cls(**kwargs)


Param = DampingParam
ParamInput: TypeAlias = DampingParam | Mapping[str, Any]


def as_damping_param(param: ParamInput) -> DampingParam:
    """
    The parameters as a :class:`DampingParam`: one is returned as it is, a
    dictionary is unpacked into one, so an unknown name is a
    :class:`TypeError`. A name it leaves out is not set; nothing is filled
    in.
    """
    if isinstance(param, DampingParam):
        return param
    return DampingParam(**param)
