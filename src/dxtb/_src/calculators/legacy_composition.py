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
"""Private ordered composition for retained legacy single-System routes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Mapping

from torch import Tensor

from dxtb import labels
from dxtb._src.calculators.model import System
from dxtb._src.calculators.result import Result
from dxtb._src.calculators.singlepoint import _singlepoint_impl
from dxtb._src.scf.functional.data import functional_scf_interactions_eligible

__all__ = [
    "LegacyCompositionResult",
    "LegacySCFOptions",
    "LegacySystemRoute",
    "classify_legacy_system",
    "compose_legacy_systems",
]


class LegacySystemRoute(Enum):
    """Static execution ownership for one independently prepared System."""

    MAPPED = auto()
    LEGACY_COMPOSED = auto()
    SEPARATE_ROUTE = auto()
    REJECTED = auto()


@dataclass(frozen=True)
class LegacySCFOptions:
    """Engine constructor extras retained by the low-level SCF contract."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] | None = None

    def as_kwargs(self) -> dict[str, Any]:
        """Return a call-local mapping while retaining value identity."""
        return {} if self.kwargs is None else dict(self.kwargs)


@dataclass(frozen=True)
class LegacyCompositionResult:
    """Ordered physical Results from retained independent System calls."""

    source_ids: tuple[int, ...]
    results: tuple[Result, ...]


def classify_legacy_system(
    system: System,
    *,
    legacy_options: LegacySCFOptions | None = None,
) -> LegacySystemRoute:
    """Classify a retained call using setup and extension metadata only."""
    if system.batch_mode != 0 or system.numbers.ndim != 1:
        return LegacySystemRoute.REJECTED

    config = system.config
    if config.scf.requires_iterations is False or {"all", "scf"} & set(
        config.exclude
    ):
        return LegacySystemRoute.SEPARATE_ROUTE

    core = system.exact_core
    if core is None:
        return LegacySystemRoute.REJECTED

    if core.integral is None:
        return LegacySystemRoute.REJECTED
    if core.integral.driver_type == labels.INTDRIVER_LIBCINT:
        return LegacySystemRoute.LEGACY_COMPOSED
    if core.integral.driver_type != labels.INTDRIVER_PYTORCH:
        return LegacySystemRoute.REJECTED

    if legacy_options is not None and (
        legacy_options.args or legacy_options.kwargs
    ):
        return LegacySystemRoute.LEGACY_COMPOSED

    if config.scf.scf_mode != labels.SCF_MODE_FULL:
        return LegacySystemRoute.LEGACY_COMPOSED

    eligible, _ = functional_scf_interactions_eligible(system.interactions)
    if not eligible:
        return LegacySystemRoute.LEGACY_COMPOSED

    if len(core.classical_terms) != len(system.classicals.components):
        return LegacySystemRoute.LEGACY_COMPOSED

    scf_config = config.scf
    if scf_config.fermi.etemp < 0.0 or scf_config.fermi.maxiter < 0:
        return LegacySystemRoute.LEGACY_COMPOSED
    if scf_config.maxiter < 0:
        raise ValueError("The functional SCF solver requires maxiter >= 0.")

    return LegacySystemRoute.MAPPED


def compose_legacy_systems(
    systems: tuple[System, ...],
    positions: tuple[Tensor, ...],
    charges: tuple[Tensor | float | int, ...] | None = None,
    spins: tuple[Tensor | float | int | None, ...] | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    legacy_options: LegacySCFOptions | None = None,
) -> LegacyCompositionResult:
    """Evaluate retained Systems independently and preserve input order.

    This private boundary deliberately performs an outer Python composition
    of independent legacy calls. It is not a fallback for the mapped route.
    """
    count = len(systems)
    if count == 0:
        raise ValueError("Legacy composition requires at least one System.")
    if len(positions) != count:
        raise ValueError("Systems and positions must have the same length.")
    if charges is None:
        charges = tuple(0 for _ in systems)
    if len(charges) != count:
        raise ValueError("Systems and charges must have the same length.")
    if spins is None:
        spins = tuple(None for _ in systems)
    if len(spins) != count:
        raise ValueError("Systems and spins must have the same length.")
    if source_ids is None:
        source_ids = tuple(range(count))
    if len(source_ids) != count or len(set(source_ids)) != count:
        raise ValueError("Source identifiers must be unique and aligned.")
    if any(not isinstance(source_id, int) for source_id in source_ids):
        raise TypeError("Source identifiers must be integers.")

    routes = tuple(
        classify_legacy_system(system, legacy_options=legacy_options)
        for system in systems
    )
    rejected = [
        source_ids[i]
        for i, route in enumerate(routes)
        if route is LegacySystemRoute.REJECTED
    ]
    if rejected:
        raise ValueError(
            f"Systems {rejected} are not supported by retained composition."
        )
    mapped = [
        source_ids[i]
        for i, route in enumerate(routes)
        if route is LegacySystemRoute.MAPPED
    ]
    if mapped:
        raise ValueError(
            "Mapped-eligible Systems must use the private mapped evaluator; "
            f"legacy composition was not selected for source IDs {mapped}."
        )

    kwargs = None if legacy_options is None else legacy_options.as_kwargs()
    args = () if legacy_options is None else legacy_options.args
    results_list: list[Result] = []
    for source_id, system, position, charge, spin, route in zip(
        source_ids, systems, positions, charges, spins, routes
    ):
        try:
            result = _singlepoint_impl(
                system,
                position,
                charge,
                spin,
                field,
                field_grad,
                fixed_trip=False,
                force_legacy_scf=(
                    route is LegacySystemRoute.LEGACY_COMPOSED
                    or bool(args)
                    or bool(kwargs)
                ),
                legacy_scf_args=args,
                legacy_scf_kwargs=kwargs,
                refresh_extension_caches=True,
            )
        except Exception as error:
            # Keep the original exception object and category while making
            # the failing source entry identifiable to compatibility callers.
            try:
                setattr(error, "source_id", source_id)
            except (AttributeError, TypeError):
                pass
            add_note = getattr(error, "add_note", None)
            if add_note is not None:
                add_note(
                    f"Legacy composition failed for source ID {source_id}."
                )
            raise
        results_list.append(result)
    results = tuple(results_list)
    return LegacyCompositionResult(source_ids=source_ids, results=results)
