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
Calculator: Decorators
======================

Decorators for the Calculator class. These decorators can mark:

- functions that require ``requires_grad=True`` for certain tensors
- functions that require specific interactions to be present
- functions that are computed numerically
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING, cast

import torch

from dxtb import OutputHandler
from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Callable, Tensor, TypeVar

if TYPE_CHECKING:
    from ..base import Calculator

__all__ = [
    "requires_positions_grad",
    "numerical",
]

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


def requires_positions_grad(
    func: Callable[..., Tensor],
) -> Callable[..., Tensor]:
    @wraps(func)
    def wrapper(
        self: Calculator,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if not positions.requires_grad:
            raise RuntimeError(
                f"Position tensor needs ``requires_grad=True`` in '{func.__name__}'."
            )

        return func(self, positions, chrg, spin, *args, **kwargs)

    return wrapper


def _numerical(nograd: bool = False, noprint: bool = False) -> Callable[[F], F]:
    """
    Decorator for numerical differentiation.
    Pass ``True`` to turns off gradient tracking for the function.
    """

    class NoOpContext:
        def __enter__(self):
            pass

        def __exit__(self, *_, **__):
            pass

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print_context = (
                OutputHandler.with_verbosity(0)
                if noprint is True
                else NoOpContext()
            )
            grad_context = torch.no_grad() if nograd is True else NoOpContext()

            with print_context, grad_context:
                result = func(*args, **kwargs)

            return result

        return cast(F, wrapper)

    return decorator


def numerical(func: F) -> F:
    """
    Decorator for numerical differentiation. Turns off gradient tracking.

    .. warning::

        This decorator turns off gradient tracking for the numerical
        differentiation. Explicit field arguments are evaluated as supplied
        and are never stored or modified.
    """
    return _numerical(nograd=True)(func)
