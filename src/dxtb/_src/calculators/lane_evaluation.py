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
"""Execution of per-lane single-System calculations for packed batches.

A packed batch is a sequence of ordinary single ``System`` lanes. This module
decides how those lanes execute; it never changes a System's structural
variable:

- Lanes that share one ``System`` object (and the same optional call inputs)
  form a group that runs through ``torch.func.vmap`` over the authoritative
  fixed-trip single-System kernel.
- Every other lane, including groups of size one and Systems that cannot run
  under the transform-compatible route, is evaluated independently with the
  Eager controller (Independent evaluation).

Different molecules are therefore never padded to a common layout for
physics. Padding across heterogeneous shapes only happens afterwards, when
per-lane Results are packed into the public batched Result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial

import torch
from torch import Tensor
from torch.utils._pytree import tree_map

from dxtb import labels
from dxtb._src.calculators.model import System
from dxtb._src.calculators.result import Result
from dxtb._src.calculators.singlepoint import (
    _prepare_singlepoint,
    _singlepoint_impl,
    _singlepoint_result,
    system_singlepoint,
)
from dxtb._src.scf.core import replace_tensor_leaves, tensor_leaves
from dxtb._src.scf.data import scf_interactions_eligible
from dxtb._src.scf.eager import (
    fixed_trip_control,
    scf_core_result,
    prepare_system_fixed_trip,
)
from dxtb._src.scf.solver import finalize_fixed_trip_problem
from dxtb._src.scf.iterator import get_refocc, reference_electron_count
from dxtb._src.wavefunction.filling import spin_electron_violations

__all__ = [
    "LaneGroup",
    "can_vmap_system",
    "evaluate_lane_group",
    "evaluate_systems_independently",
    "group_lanes",
]


def can_vmap_system(system: System) -> bool:
    """Return whether ``system`` may run in a Mapped evaluation.

    A Mapped evaluation maps the Fixed-trip controller single-System
    kernel. Systems that are valid but need the eager controller (libcint,
    implicit differentiation, multi-stage or eager-only SCF strategies,
    classical extensions without exact setup, SCF-free calculations) return
    ``False`` and are evaluated independently.

    Raises
    ------
    ValueError, NotImplementedError
        For configurations that are invalid for every evaluation. These errors are
        raised here so that a batch fails before any lane is evaluated.
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

    # Calculations without an iterative SCF have nothing to map over.
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

    # Invalid SCF configurations fail regardless of how a Lane is evaluated.
    eligible, _ = scf_interactions_eligible(system.interactions)
    if not eligible:
        names = ", ".join(
            type(component).__name__
            for component in system.interactions.components
        )
        raise NotImplementedError(
            f"SCF does not support interaction extension(s): {names}."
        )

    scf_config = config.scf
    if scf_config.fermi.etemp < 0.0:
        raise ValueError("Electronic Temperature must be non-negative.")
    if scf_config.fermi.maxiter < 0:
        raise ValueError("Fermi maxiter must be non-negative.")
    if scf_config.maxiter < 0:
        raise ValueError("The SCF solver requires maxiter >= 0.")

    if any(
        type(stage.converger).__name__ == "SOSCFConverger"
        for stage in scf_config.strategy.stages
    ):
        kt = float(scf_config.fermi.etemp) * 3.166811563e-6
        if kt >= 3.0e-7:
            raise ValueError(
                "SOSCF cannot be used with Fermi smearing at nonzero electronic temperature. Use "
                "Anderson, Broyden, or CDIIS, or disable smearing explicitly."
            )

    eager_only = len(scf_config.strategy.stages) != 1 or any(
        not getattr(stage.converger, "supports_fixed_trip", True)
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

    # Valid configurations that still require the eager controller.
    if (
        len(core.classical_terms) != len(system.classicals.components)
        or system._active_exact_core() is None
        or scf_config.differentiation.value == "implicit"
        or eager_only
    ):
        return False

    return True


@dataclass(frozen=True)
class LaneGroup:
    """Source lanes that execute together.

    ``use_vmap`` groups share one ``System`` object and the same presence of
    optional call inputs. All other groups contain exactly one lane.
    """

    system: System
    source_ids: tuple[int, ...]
    use_vmap: bool


def group_lanes(
    systems: Sequence[System],
    spins: Sequence[Tensor | None],
    fields: Sequence[Tensor | None],
    field_grads: Sequence[Tensor | None],
) -> tuple[LaneGroup, ...]:
    """Validate the Systems and partition source lanes into execution groups.

    Lanes are grouped only when they reference the very same ``System`` object
    and agree on whether ``spin``, ``field`` and ``field_grad`` are given; the
    tensor values themselves may differ per lane. Groups are returned in the
    order of their first source lane. Every lane belongs to exactly one group.

    Each distinct System is passed to :func:`can_vmap_system` once, so a
    configuration that is invalid for every evaluation raises here, before any lane
    is evaluated. The returned groups are therefore prevalidated.
    """
    mapped: dict[tuple[int, bool, bool, bool], list[int]] = {}
    groups: list[tuple[System, list[int], bool]] = []

    # Every distinct System is validated and classified exactly once. All
    # Systems are alive in ``systems`` for the whole call, so ``id`` is a
    # stable key here; the cache does not outlive this call.
    eligible: dict[int, bool] = {}
    for system in systems:
        if id(system) not in eligible:
            eligible[id(system)] = can_vmap_system(system)

    for source_id, system in enumerate(systems):
        if not eligible[id(system)]:
            groups.append((system, [source_id], False))
            continue

        key = (
            id(system),
            spins[source_id] is not None,
            fields[source_id] is not None,
            field_grads[source_id] is not None,
        )
        members = mapped.get(key)
        if members is None:
            members = mapped[key] = []
            groups.append((system, members, True))
        members.append(source_id)

    # A single mapped lane gains nothing from vmap; evaluate it directly.
    return tuple(
        LaneGroup(
            system=system,
            source_ids=tuple(members),
            use_vmap=use_vmap and len(members) > 1,
        )
        for system, members, use_vmap in groups
    )


def evaluate_systems_independently(
    systems: Sequence[System],
    positions: Sequence[Tensor],
    charges: Sequence[Tensor | float | int] | None = None,
    spins: Sequence[Tensor | float | int | None] | None = None,
    *,
    source_ids: Sequence[int] | None = None,
    field: Tensor | Sequence[Tensor | None] | None = None,
    field_grad: Tensor | Sequence[Tensor | None] | None = None,
) -> tuple[Result, ...]:
    """Evaluate Systems one by one (Independent evaluation, Eager controller).

    Errors raised by a lane carry its ``source_id`` attribute.
    """
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
    if isinstance(field, Sequence) and len(field) != count:
        raise ValueError("Systems and per-lane fields must have equal length.")
    if isinstance(field_grad, Sequence) and len(field_grad) != count:
        raise ValueError(
            "Systems and per-lane field gradients must have equal length."
        )

    # Configuration validation shared with Mapped evaluation (once per
    # distinct System); the eligibility verdict itself is not needed here.
    validated: set[int] = set()
    for system in systems:
        if id(system) not in validated:
            can_vmap_system(system)
            validated.add(id(system))

    return _evaluate_prevalidated(
        systems,
        positions,
        charges,
        spins,
        tuple(_lane_value(field, idx) for idx in range(count)),
        tuple(_lane_value(field_grad, idx) for idx in range(count)),
        source_ids,
    )


def _evaluate_prevalidated(
    systems: Sequence[System],
    positions: Sequence[Tensor],
    charges: Sequence[Tensor | float | int],
    spins: Sequence[Tensor | float | int | None],
    fields: Sequence[Tensor | None],
    field_grads: Sequence[Tensor | None],
    source_ids: Sequence[int],
) -> tuple[Result, ...]:
    """Run aligned Lanes through Independent evaluation.

    The Systems must already have passed :func:`can_vmap_system`.
    """
    results: list[Result] = []
    for idx, source_id in enumerate(source_ids):
        try:
            result = _singlepoint_impl(
                systems[idx],
                positions[idx],
                charges[idx],
                spins[idx],
                fields[idx],
                field_grads[idx],
                fixed_trip=False,
            )
        except Exception as error:
            _attach_source_id(error, source_id)
            raise
        results.append(result)
    return tuple(results)


def evaluate_lane_group(
    group: LaneGroup,
    positions: Sequence[Tensor],
    charges: Sequence[Tensor],
    spins: Sequence[Tensor | None],
    fields: Sequence[Tensor | None],
    field_grads: Sequence[Tensor | None],
) -> tuple[Result, ...]:
    """Evaluate one group and return one Result per member lane.

    The per-lane sequences are indexed by source lane. ``group`` must come
    from :func:`group_lanes`, which has validated its System. Groups of size
    one use Independent evaluation; larger groups stack their per-lane
    inputs and map the fixed-trip single-System kernel with
    ``torch.func.vmap``.
    """
    ids = group.source_ids
    if not group.use_vmap:
        return _evaluate_prevalidated(
            tuple(group.system for _ in ids),
            tuple(positions[i] for i in ids),
            tuple(charges[i] for i in ids),
            tuple(spins[i] for i in ids),
            tuple(fields[i] for i in ids),
            tuple(field_grads[i] for i in ids),
            ids,
        )

    # vmap cannot raise data-dependent errors, so electron-count and spin
    # consistency is checked eagerly, once for the whole group.
    _validate_group_electrons(group, charges, spins)

    # Shared tensors (e.g. one field for all lanes) are stacked as repeated
    # references; autograd accumulates their gradients across lanes.
    stacked_spins = _stack_optional(tuple(spins[i] for i in ids))
    stacked_fields = _stack_optional(tuple(fields[i] for i in ids))
    stacked_field_grads = _stack_optional(tuple(field_grads[i] for i in ids))
    in_dims = (
        0,
        0,
        None if stacked_spins is None else 0,
        None if stacked_fields is None else 0,
        None if stacked_field_grads is None else 0,
    )
    args = (
        torch.stack([positions[i] for i in ids]),
        torch.stack([charges[i] for i in ids]),
        stacked_spins,
        stacked_fields,
        stacked_field_grads,
    )
    if _requires_static_schedule():
        mapped = torch.func.vmap(
            partial(system_singlepoint, group.system), in_dims=in_dims
        )(*args)
    else:
        mapped = _evaluate_group_adaptive(group.system, args, in_dims)
    return tuple(_select_lane(mapped, lane) for lane in range(len(ids)))


# Adaptive execution of mapped groups
#
# The fixed-trip controller runs ``maxiter`` map trips and freezes finished
# lanes by masks. Outside of traced execution, a mapped group instead runs
# the same controller in static chunks of trips, each chunk one ``vmap``
# call, and stops after the chunk in which its last lane certified or failed
# terminally. Lanes that finish inside a chunk are frozen by the shared
# transition exactly as in the full schedule, so values, iteration counts,
# status and derivatives are those of the full schedule.
#
# Lane-dependent tensors cross the ``vmap`` calls as flat leaves of the
# immutable dataclass trees (prepared non-SCF terms, SCF problem, solver
# state); everything else in those trees is lane-independent and is reused
# as is.

_ADAPTIVE_CHUNK_TRIPS = 1
"""Map trips per ``vmap`` call of an adaptive group (performance only).

Measured on CPU (``benchmarks/results/adaptive-chunk-sweep-cpu.json``,
chunk lengths 1-16): one extra ``vmap`` call and host read costs a few
milliseconds, far less than one map trip, so the shortest chunk avoids all
overshoot trips and was fastest or tied in every case. Chunk length 1 is
the measured CPU optimum; the CUDA optimum is unknown (every chunk boundary
is a device-to-host read there) and must be measured separately.
"""


def _requires_static_schedule() -> bool:
    """Return whether a mapped group must run the full static schedule.

    Inside an active ``torch.func`` transform or ``torch.compile`` tracing,
    the group is part of a traced computation and cannot stop on tensor
    values.
    """
    return (
        torch.compiler.is_compiling()
        or torch._C._functorch.maybe_current_level() is not None
    )


def _evaluate_group_adaptive(
    system: System,
    args: tuple[Tensor | None, ...],
    in_dims: tuple[int | None, ...],
) -> Result:
    """Map one group through the fixed-trip controller in static chunks.

    Advancing stops once no lane is active (every lane certified or failed
    terminally) or after the configured ``maxiter`` trips. Each stop
    decision is one host read for the whole group.
    """
    capture: dict[str, tuple[object, tuple[str, ...]]] = {}
    fixed, state = torch.func.vmap(
        partial(_group_start, system, capture), in_dims=in_dims
    )(*args)

    maxiter = system.config.scf.maxiter
    trip = 0
    while trip < maxiter and bool(_group_active(capture, state).any()):
        stop = min(trip + _ADAPTIVE_CHUNK_TRIPS, maxiter)
        state = torch.func.vmap(
            partial(_group_advance, capture, trip, stop), in_dims=(0, 0)
        )(fixed, state)
        trip = stop

    return torch.func.vmap(partial(_group_finish, capture), in_dims=(0, 0, 0))(
        fixed, state, args[0]
    )


def _group_start(
    system: System,
    capture: dict[str, tuple[object, tuple[str, ...]]],
    positions: Tensor,
    charge: Tensor,
    spin: Tensor | None,
    field: Tensor | None,
    field_grad: Tensor | None,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Prepare one lane and start its controller (inside ``vmap``)."""
    prepared = _prepare_singlepoint(
        system, positions, charge, spin, field, field_grad
    )
    if isinstance(prepared, Result):
        raise RuntimeError("Mapped groups require an iterative SCF.")
    x, problem = prepare_system_fixed_trip(
        *prepared.scf_arguments(),
        core_index=prepared.core_index,
        core_index_owner=prepared.core_index_owner,
    )
    state = fixed_trip_control(x, problem, prepared.config.scf).start(x)
    # Call-time objects (interaction list and cache, positions, the System)
    # are not needed after this point and are not carried.
    carried = replace(
        prepared,
        positions=None,
        interactions=None,
        interaction_data=None,
        core_index_owner=None,
    )
    return (
        _carry(capture, "fixed", (carried, problem)),
        _carry(capture, "state", state),
    )


def _group_advance(
    capture: dict[str, tuple[object, tuple[str, ...]]],
    start: int,
    stop: int,
    fixed: tuple[Tensor, ...],
    state: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Advance one lane by trips ``start <= index < stop`` (inside ``vmap``)."""
    prepared, problem = _restore(capture, "fixed", fixed)
    solver_state = _restore(capture, "state", state)
    control = fixed_trip_control(
        solver_state.current_x, problem, prepared.config.scf
    )
    solver_state, _ = control.advance(solver_state, start, stop)
    return _carry(capture, "state", solver_state)


def _group_finish(
    capture: dict[str, tuple[object, tuple[str, ...]]],
    fixed: tuple[Tensor, ...],
    state: tuple[Tensor, ...],
    positions: Tensor,
) -> Result:
    """Finalize one lane and assemble its Result (inside ``vmap``)."""
    prepared, problem = _restore(capture, "fixed", fixed)
    solver_state = _restore(capture, "state", state)
    control = fixed_trip_control(
        solver_state.current_x, problem, prepared.config.scf
    )
    scf_result = finalize_fixed_trip_problem(
        control, solver_state, solver_state.current_x
    )
    return _singlepoint_result(
        replace(prepared, positions=positions),
        scf_core_result(scf_result),
    )


def _group_active(
    capture: dict[str, tuple[object, tuple[str, ...]]],
    state: tuple[Tensor, ...],
) -> Tensor:
    """Return the per-lane ``active`` flags of a carried solver state."""
    template, paths = capture["state"]
    if "active" in paths:
        return state[paths.index("active")]
    return template.active  # type: ignore[attr-defined]


def _carry(
    capture: dict[str, tuple[object, tuple[str, ...]]],
    key: str,
    tree: object,
) -> tuple[Tensor, ...]:
    """Record ``tree`` as template and return its lane-batched leaves."""
    leaves = tuple(
        (path, value)
        for path, value in tensor_leaves(tree)
        if torch._C._functorch.is_batchedtensor(value)
    )
    values = tuple(value for _, value in leaves)
    if _has_uncarried_lane_tensor(tree, {id(value) for value in values}):
        raise RuntimeError(
            "A lane-dependent tensor is held outside the carried dataclass "
            "tree of a mapped group."
        )
    capture[key] = (tree, tuple(path for path, _ in leaves))
    return values


def _restore(
    capture: dict[str, tuple[object, tuple[str, ...]]],
    key: str,
    values: tuple[Tensor, ...],
):
    """Rebuild a carried tree from its template and lane leaves."""
    template, paths = capture[key]
    return replace_tensor_leaves(template, dict(zip(paths, values)))


def _has_uncarried_lane_tensor(
    value: object, carried: set[int], seen: set[int] | None = None
) -> bool:
    """Return whether ``value`` reaches a transform tensor not in ``carried``.

    Such a tensor would escape its ``vmap`` call inside the static template.
    """
    if isinstance(value, Tensor):
        return id(
            value
        ) not in carried and torch._C._functorch.is_functorch_wrapped_tensor(
            value
        )
    if value is None or isinstance(
        value, (str, bytes, int, float, complex, torch.dtype, torch.device)
    ):
        return False
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if is_dataclass(value) and not isinstance(value, type):
        children = [getattr(value, item.name) for item in fields(value)]
    elif isinstance(value, (tuple, list, set, frozenset)):
        children = list(value)
    elif isinstance(value, dict):
        children = list(value.values())
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        children = list(vars(value).values())
    else:
        children = []
    return any(
        _has_uncarried_lane_tensor(child, carried, seen) for child in children
    )


def _validate_group_electrons(
    group: LaneGroup,
    charges: Sequence[Tensor],
    spins: Sequence[Tensor | None],
) -> None:
    """Apply the eager electron/spin checks to all lanes of a mapped group.

    The rules are those of a single-System evaluation (``get_refocc`` with
    ``validate_spin=True``), evaluated for the whole group at once so that a
    valid group needs a single host decision. The first invalid source lane
    is then re-checked with the single-System route, which raises the exact
    eager error; the error carries that lane's ``source_id``.
    """
    ids = group.source_ids
    # Without an explicit spin a single-System evaluation performs no value
    # checks (lanes of one group agree on spin presence).
    if spins[ids[0]] is None:
        return

    system = group.system
    h0_setup = _electron_setup(system)
    with torch.no_grad():
        nel = reference_electron_count(h0_setup.refocc, system.ihelp) - (
            torch.stack([charges[i].detach().reshape(()) for i in ids])
        )
        uhf = torch.stack(
            [spins[i].detach().reshape(()) for i in ids]  # type: ignore[union-attr]
        ).type(nel.dtype)
        too_many, parity_mismatch = spin_electron_violations(nel, uhf)
        invalid = (too_many | parity_mismatch).tolist()

    for source_id, lane_invalid in zip(ids, invalid):
        if not lane_invalid:
            continue
        try:
            _validate_electrons(system, charges[source_id], spins[source_id])
        except Exception as error:
            _attach_source_id(error, source_id)
            raise
        raise RuntimeError(
            "Grouped electron validation disagrees with the single-System "
            f"check for source lane {source_id}."
        )


def _electron_setup(system: System):
    """Return the H0 setup that owns the reference occupation."""
    core = system._active_exact_core()
    h0_setup = core.h0_setup if core is not None else system.h0_setup
    assert h0_setup is not None
    return h0_setup


def _validate_electrons(
    system: System, charge: Tensor, spin: Tensor | None
) -> None:
    """Run the eager electron/spin checks of a single-System evaluation."""
    h0_setup = _electron_setup(system)

    # Match the one-element scalar layout of a single-System evaluation.
    get_refocc(
        h0_setup.refocc,
        charge.reshape(1),
        None if spin is None else spin.reshape(1),
        system.ihelp,
        validate_spin=True,
    )


def _lane_value(
    value: Tensor | Sequence[Tensor | None] | None, idx: int
) -> Tensor | None:
    """Return a shared perturbation or the entry for one lane."""
    if isinstance(value, Sequence):
        return value[idx]
    return value


def _stack_optional(values: Sequence[Tensor | None]) -> Tensor | None:
    """Stack per-lane tensors of a group that agrees on their presence."""
    if values[0] is None:
        return None
    return torch.stack(values)  # type: ignore[arg-type]


def _select_lane(mapped: Result, lane: int) -> Result:
    """Index every tensor leaf of a mapped Result at one lane."""
    return tree_map(lambda leaf: leaf.select(0, lane), mapped)


def _attach_source_id(error: Exception, source_id: int) -> None:
    """Record the failing source lane on an exception when possible."""
    try:
        setattr(error, "source_id", source_id)
    except (AttributeError, TypeError):
        pass
