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
"""Ordered evaluation for independent single-System calculations."""

from __future__ import annotations

from torch import Tensor

from dxtb import labels
from dxtb._src.calculators.model import System
from dxtb._src.calculators.result import Result
from dxtb._src.calculators.singlepoint import _singlepoint_impl
from dxtb._src.scf.functional.data import functional_scf_interactions_eligible

__all__ = [
    "can_pad_system",
    "compose_legacy_systems",
]


def can_pad_system(system: System) -> bool:
    """Return True if ``system`` can use canonical padded vmap evaluation.

    Returns False when ``system`` must be evaluated independently via
    :func:`compose_legacy_systems`, and raises ``ValueError`` or
    ``NotImplementedError`` on invalid configurations.
    """
    if system.batch_mode != 0 or system.numbers.ndim != 1:
        raise ValueError(
            "Independent System evaluation requires single-System "
            "(batch_mode=0) instances."
        )
    if not system._ihelp_compatible():
        raise ValueError(
            "System IndexHelper is incompatible with the captured System setup."
        )

    config = system.config
    if {"all", "scf"} & set(config.exclude):
        return False

    core = system.exact_core
    if (
        core is None
        or core.integral is None
        or system.h0_setup is None
        or config.ints.level < labels.INTLEVEL_HCORE
    ):
        return False

    if config.scf.requires_iterations is False:
        return False

    eligible, _ = functional_scf_interactions_eligible(system.interactions)
    if not eligible:
        names = ", ".join(
            type(component).__name__
            for component in system.interactions.components
        )
        raise NotImplementedError(
            f"Functional SCF does not support interaction extension(s): {names}."
        )

    scf_config = config.scf
    if scf_config.fermi.etemp < 0.0:
        raise ValueError("Electronic Temperature must be non-negative.")
    if scf_config.fermi.maxiter < 0:
        raise ValueError("Fermi maxiter must be non-negative.")
    if scf_config.maxiter < 0:
        raise ValueError("The functional SCF solver requires maxiter >= 0.")

    if any(
        type(stage.method).__name__ == "SOSCFMethod"
        for stage in scf_config.strategy.stages
    ):
        kt = float(scf_config.fermi.etemp) * 3.166811563e-6
        if kt >= 3.0e-7:
            raise ValueError(
                "SOSCF cannot be used with active Fermi smearing. Use "
                "Anderson, Broyden, or CDIIS, or disable smearing explicitly."
            )

    eager_only = len(scf_config.strategy.stages) != 1 or any(
        not getattr(stage.method, "supports_fixed_trip", True)
        for stage in scf_config.strategy.stages
    )
    if scf_config.differentiation.value == "unrolled" and eager_only:
        raise ValueError(
            "This SCF strategy requires eager/implicit execution; "
            "unrolled fixed-trip execution does not support its "
            "dynamic or eager-only method stages."
        )

    if core.integral.driver_type == labels.INTDRIVER_LIBCINT:
        return False
    if core.integral.driver_type != labels.INTDRIVER_PYTORCH:
        raise ValueError(
            f"Unsupported integral driver {core.integral.driver_type}."
        )

    if (
        len(core.classical_terms) != len(system.classicals.components)
        or system._active_exact_core() is None
        or scf_config.differentiation.value == "implicit"
        or eager_only
    ):
        return False

    return True


def compose_legacy_systems(
    systems: tuple[System, ...],
    positions: tuple[Tensor, ...],
    charges: tuple[Tensor | float | int, ...] | None = None,
    spins: tuple[Tensor | float | int | None, ...] | None = None,
    *,
    source_ids: tuple[int, ...] | None = None,
    field: Tensor | tuple[Tensor | None, ...] | None = None,
    field_grad: Tensor | tuple[Tensor | None, ...] | None = None,
) -> tuple[Result, ...]:
    """Evaluate Systems independently and return Results in input order."""
    count = len(systems)
    if count == 0:
        raise ValueError(
            "Independent System evaluation requires at least one System."
        )
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

    padded_eligible = [
        source_ids[i]
        for i, system in enumerate(systems)
        if can_pad_system(system)
    ]
    if padded_eligible:
        raise ValueError(
            "Mapped-eligible Systems must use the private mapped evaluator; "
            f"independent evaluation was not selected for source IDs {padded_eligible}."
        )

    if isinstance(field, tuple) and len(field) != count:
        raise ValueError("Systems and per-lane fields must have equal length.")
    if isinstance(field_grad, tuple) and len(field_grad) != count:
        raise ValueError(
            "Systems and per-lane field gradients must have equal length."
        )

    results: list[Result] = []
    for idx, (source_id, system, position, charge, spin) in enumerate(
        zip(source_ids, systems, positions, charges, spins)
    ):
        lane_field = field[idx] if isinstance(field, tuple) else field
        lane_field_grad = (
            field_grad[idx] if isinstance(field_grad, tuple) else field_grad
        )
        try:
            result = _singlepoint_impl(
                system,
                position,
                charge,
                spin,
                lane_field,
                lane_field_grad,
                fixed_trip=False,
                refresh_extension_caches=True,
            )
        except Exception as error:
            try:
                setattr(error, "source_id", source_id)
            except (AttributeError, TypeError):
                pass
            raise
        results.append(result)
    return tuple(results)
