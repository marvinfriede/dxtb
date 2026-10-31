# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Internal numerical-method contracts for functional SCF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch

from dxtb._src.typing import Tensor

from .problem import SCFPhysicalState, SCFProblem

__all__ = [
    "LinearMethod",
    "SCFMethod",
    "SCFMethodContext",
    "SCFMethodProposal",
    "SCFMethodState",
    "simple_mix",
]


@dataclass(frozen=True, eq=False)
class SCFMethodContext:
    """Controller progress and projected proposal input for one transition."""

    step: int
    transition_active: Tensor
    damp: Tensor
    proposal_x: Tensor


class SCFMethodState(Protocol):
    """Marker protocol for immutable, method-specific state values."""


@dataclass(frozen=True, eq=False)
class SCFMethodProposal:
    """Proposed working state and explicit immutable method state."""

    x: Tensor
    state: SCFMethodState | None


class SCFMethod(Protocol):
    """Local numerical method acting on a paired physical observation."""

    supports_fixed_trip: ClassVar[bool]

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state: SCFMethodState | None,
        context: SCFMethodContext,
    ) -> SCFMethodProposal: ...


@dataclass(frozen=True, eq=False)
class LinearMethod:
    """Stateless linear mixing with the accepted simple-mixer semantics."""

    supports_fixed_trip: ClassVar[bool] = True

    def propose(
        self,
        problem: SCFProblem,
        physical_state: SCFPhysicalState,
        method_state: SCFMethodState | None,
        context: SCFMethodContext,
    ) -> SCFMethodProposal:
        """Mix the current input toward the controller-projected map value."""
        del problem
        if method_state is not None:
            raise TypeError("LinearMethod does not retain method history.")
        return SCFMethodProposal(
            x=simple_mix(physical_state.x, context.proposal_x, context.damp),
            state=None,
        )


def simple_mix(old: Tensor, mapped: Tensor, damp: Tensor | float) -> Tensor:
    """Apply one out-of-place linear mixing transition."""
    return old + (mapped - old) * damp
