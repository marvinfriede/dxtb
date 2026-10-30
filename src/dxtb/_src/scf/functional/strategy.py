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
"""Immutable SCF stages, switch policies, and strategy validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .method import SCFMethod
from .problem import SCFRepresentation

__all__ = ["SCFStage", "SCFSwitchPolicy", "SCFStrategy"]


@dataclass(frozen=True, eq=False)
class SCFSwitchPolicy:
    """Physical progress or explicit stage-budget condition for switching."""

    min_stage_iterations: int = 0
    fixed_point_l2: float | None = None
    stationarity: float | None = None
    max_stage_iterations: int | None = None

    def __post_init__(self) -> None:
        if self.min_stage_iterations < 0:
            raise ValueError("Minimum stage iterations cannot be negative.")
        if self.max_stage_iterations is not None:
            if self.max_stage_iterations < 1:
                raise ValueError("Maximum stage iterations must be positive.")
            if self.max_stage_iterations < self.min_stage_iterations:
                raise ValueError(
                    "Maximum stage iterations cannot be below the minimum."
                )
        for name, value in (
            ("fixed-point L2", self.fixed_point_l2),
            ("stationarity", self.stationarity),
        ):
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(
                    f"{name} threshold must be finite and nonnegative."
                )
        if (
            self.fixed_point_l2 is None
            and self.stationarity is None
            and self.max_stage_iterations is None
        ):
            raise ValueError("A switch policy needs a metric or stage budget.")


@dataclass(frozen=True, eq=False)
class SCFStage:
    """One method stage and its representation, damping, and exit policy."""

    method: SCFMethod
    representation: SCFRepresentation | None = None
    switch_policy: SCFSwitchPolicy | None = None
    initial_damp: float | None = None

    def __post_init__(self) -> None:
        if self.initial_damp is not None and (
            not isfinite(self.initial_damp)
            or not 0.0 < self.initial_damp <= 1.0
        ):
            raise ValueError("Stage damping must be finite and in (0, 1].")


@dataclass(frozen=True, eq=False)
class SCFStrategy:
    """Immutable ordered stages executed under one global SCF budget."""

    stages: tuple[SCFStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("An SCF strategy must contain at least one stage.")
        if not isinstance(self.stages, tuple):
            raise TypeError("SCF strategy stages must be an immutable tuple.")
        if any(stage.switch_policy is None for stage in self.stages[:-1]):
            raise ValueError(
                "Every non-final SCF stage requires an exit switch policy."
            )
        if self.stages[-1].switch_policy is not None:
            raise ValueError("The final SCF stage cannot have an exit policy.")

    def single_stage(self) -> SCFStage:
        """Return a single executable stage or reject dynamic composition."""
        if len(self.stages) != 1:
            raise NotImplementedError(
                "Dynamic multi-stage SCF is currently eager-only."
            )
        return self.stages[0]

    def resolve_representations(
        self, initial: SCFRepresentation
    ) -> tuple[SCFRepresentation, ...]:
        """Resolve inherited representations and validate each conversion."""
        resolved: list[SCFRepresentation] = []
        previous = initial
        for index, stage in enumerate(self.stages):
            target = (
                previous
                if stage.representation is None
                else stage.representation
            )
            if index == 0 and target is not initial:
                raise NotImplementedError(
                    "The first strategy stage must use the supplied problem "
                    "representation."
                )
            required = getattr(stage.method, "required_representation", None)
            if required is not None and target is not required:
                raise ValueError(
                    f"{type(stage.method).__name__} requires the "
                    f"{required.value} representation."
                )
            if index > 0 and target is not previous:
                if target is not SCFRepresentation.FOCK:
                    raise NotImplementedError(
                        "S6 supports compact-to-Fock conversion only; "
                        "Fock-to-compact switching is unsupported."
                    )
                if previous not in (
                    SCFRepresentation.CHARGE,
                    SCFRepresentation.POTENTIAL,
                ):
                    raise NotImplementedError(
                        "This SCF representation conversion is unsupported."
                    )
            resolved.append(target)
            previous = target
        return tuple(resolved)
