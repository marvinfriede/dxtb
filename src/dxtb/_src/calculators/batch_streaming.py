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
"""Experimental first-order streaming gradients for converged SCF batches.

This module intentionally does not return a differentiable :class:`Result`.
It evaluates an explicitly separable weighted energy objective and returns
detached values and gradients after releasing each chunk's autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from dxtb._src.calculators.batch_implicit import (
    _UNSET,
    BatchRootFailureCode,
    PreparedConvergedRootBatch,
    _collect_system_and_model_tensors,
    _matches_snapshot,
    _tensor_version,
    evaluate_converged_root_batch,
)
from dxtb._src.calculators.model import Model, System
from dxtb._src.typing import Tensor

__all__ = ["StreamingBatchValueAndGrad", "stream_batch_value_and_grad"]


@dataclass(frozen=True, eq=False)
class StreamingBatchValueAndGrad:
    """Detached objective values, primal status, and first-order gradients."""

    energy: Tensor
    free_energy: Tensor
    converged: Tensor
    final_failure: Tensor
    iterations: Tensor
    position_gradient: Tensor
    charge_gradient: Tensor | None
    spin_gradient: Tensor | None
    field_gradient: Tensor | None
    field_grad_gradient: Tensor | None
    model_parameter_gradients: tuple[Tensor | None, ...]
    model_parameter_connected: tuple[bool, ...]


def _slice_batch_value(
    value,
    start: int,
    stop: int,
    batch_size: int,
    tail_ndim: int,
):
    """Slice a leading batch dimension while retaining shared inputs."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim > tail_ndim and value.shape[0] == batch_size:
        return value[start:stop]
    return value


def _chunk_prepared(
    prepared: PreparedConvergedRootBatch,
    model: Model,
    start: int,
    stop: int,
) -> PreparedConvergedRootBatch:
    """Pair a root slice with a fresh Model setup graph for this chunk."""
    system = prepared.system
    if not isinstance(system, System):
        raise TypeError(
            "Streaming gradients currently require a single-System prepared "
            "batch; padded exact-core batches are unsupported."
        )

    if (
        model.config.method != system.config.method
        or model.config.scf != system.config.scf
    ):
        raise ValueError(
            "The supplied Model does not match the prepared SCF configuration."
        )

    if len(prepared.system_tensor_refs) != len(
        prepared.system_tensor_snapshot
    ) or any(
        (ver is not None and _tensor_version(value) != ver)
        or not _matches_snapshot(value, snapshot)
        for value, snapshot, ver in zip(
            prepared.system_tensor_refs,
            prepared.system_tensor_snapshot,
            prepared.system_tensor_versions,
        )
    ):
        raise RuntimeError(
            "Prepared physical roots are stale for the supplied Model or "
            "because System setup tensors were modified; prepare roots again."
        )

    fresh_system = model.setup(system.numbers, batch_mode=system.batch_mode)
    fresh_core = fresh_system._active_exact_core()
    fresh_excluded_ids = {
        id(fresh_system.numbers),
        *({id(fresh_core.numbers)} if fresh_core is not None else set()),
    }
    fresh_system_tensor_refs, _ = _collect_system_and_model_tensors(
        fresh_system, fresh_excluded_ids
    )
    lanes = prepared.lanes[start:stop]
    return replace(
        prepared,
        system=fresh_system,
        system_tensor_refs=fresh_system_tensor_refs,
        system_tensor_versions=tuple(None for _ in fresh_system_tensor_refs),
        positions=prepared.positions[start:stop],
        lanes=lanes,
        config_identity=id(fresh_system.config),
        positions_snapshot=prepared.positions_snapshot[start:stop],
        lane_input_snapshots=prepared.lane_input_snapshots[start:stop],
        parameter_snapshots=prepared.parameter_snapshots[start:stop],
        root_snapshots=prepared.root_snapshots[start:stop],
        input_tensor_versions=prepared.input_tensor_versions[start:stop],
        valid_lane_mask=prepared.valid_lane_mask[start:stop],
        all_roots_valid=bool(prepared.valid_lane_mask[start:stop].all()),
    )


def _input_target(value):
    """Make an explicit floating input an independent gradient target."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        return None
    return value.detach().clone().requires_grad_(True)


def stream_batch_value_and_grad(
    prepared: PreparedConvergedRootBatch,
    model: Model,
    positions: Tensor,
    *,
    weights: Tensor | None = None,
    chunk_size: int = 1,
    chrg=_UNSET,
    spin=_UNSET,
    field=_UNSET,
    field_grad=_UNSET,
    strict: bool | None = None,
) -> StreamingBatchValueAndGrad:
    """Evaluate a separable objective and accumulate first-order chunk gradients.

    The operation is eager-only, does not return a differentiable batched
    ``Result``, and intentionally does not support ``torch.func`` transforms,
    ``create_graph`` or gradients of the returned gradients. A caller supplies
    the ``Model`` that owns the prepared System. Roots must still match the
    exact saved primal point. ``weights[b]`` defines
    ``sum_b weights[b] * energy[b].sum()``. Zero-weight failed lanes are
    excluded; selecting a failed lane propagates its invalid gradient.

    Position and explicitly supplied floating ``chrg``, ``spin``, ``field``
    and ``field_grad`` tensors receive returned sensitivities. Shared inputs
    accumulate all chunk contributions. Trainable ``Model.par`` parameters
    accumulate by identity. Each chunk rebuilds the System setup graph, runs
    mapped implicit evaluation, computes gradients, and releases its local
    graph before continuing.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if positions.ndim < 3 or positions.shape[0] != len(prepared.lanes):
        raise ValueError(
            "positions must have shape (batch, nat, 3) matching prepared roots."
        )
    if torch._C._functorch.get_interpreter_stack():
        raise NotImplementedError(
            "Streaming value-and-gradient evaluation is eager-only and does "
            "not support torch.func transforms."
        )
    if torch.compiler.is_compiling():
        raise NotImplementedError(
            "Streaming value-and-gradient evaluation does not support "
            "torch.compile."
        )

    batch_size = positions.shape[0]
    device, dtype = positions.device, positions.dtype
    if weights is None:
        weights = torch.ones(batch_size, device=device, dtype=dtype)
    elif tuple(weights.shape) != (batch_size,):
        raise ValueError(f"weights must have shape ({batch_size},).")
    else:
        weights = weights.to(device=device, dtype=dtype)

    position_target = _input_target(positions)
    assert position_target is not None
    external_values = (chrg, spin, field, field_grad)
    tail_dimensions = (0, 0, 1, 2)
    external_names = ("chrg", "spin", "field", "field_grad")
    external_targets = tuple(_input_target(value) for value in external_values)
    external_eval_values = tuple(
        target if target is not None else value
        for value, target in zip(external_values, external_targets)
    )
    active_external_indices = tuple(
        index
        for index, target in enumerate(external_targets)
        if target is not None
    )
    model_parameters = tuple(
        parameter
        for parameter in model.par.parameters()
        if parameter.is_floating_point() and parameter.requires_grad
    )
    prepared_model_ids = {id(value) for value in prepared.model_parameter_refs}
    if prepared_model_ids and not prepared_model_ids.issubset(
        {id(value) for value in model_parameters}
    ):
        raise ValueError(
            "The supplied Model does not own the prepared differentiable "
            "parameter leaves."
        )

    energy_chunks: list[Tensor] = []
    free_energy_chunks: list[Tensor] = []
    convergence_chunks: list[Tensor] = []
    failure_chunks: list[Tensor] = []
    iteration_chunks: list[Tensor] = []
    position_gradient = torch.zeros_like(position_target)
    external_gradients: list[Tensor | None] = [
        None if target is None else torch.zeros_like(target)
        for target in external_targets
    ]
    parameter_gradients: list[Tensor | None] = [None for _ in model_parameters]
    parameter_connected = [False for _ in model_parameters]

    target_values = (
        position_target,
        *(external_targets[index] for index in active_external_indices),
        *model_parameters,
    )
    with torch.enable_grad():
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            chunk_positions = position_target[start:stop]
            chunk_prepared = _chunk_prepared(prepared, model, start, stop)
            overrides = tuple(
                (
                    _slice_batch_value(
                        value,
                        start,
                        stop,
                        batch_size,
                        tail_ndim,
                    )
                    if value is not _UNSET
                    else _UNSET
                )
                for value, tail_ndim in zip(
                    external_eval_values, tail_dimensions
                )
            )
            kwargs = {
                name: value
                for name, value in zip(external_names, overrides)
                if value is not _UNSET
            }
            result = evaluate_converged_root_batch(
                chunk_prepared,
                positions=chunk_positions,
                strict=strict,
                **kwargs,
            )
            lane_energy = result.energy.reshape(stop - start, -1).sum(-1)
            lane_free_energy = (
                result.fenergy.reshape(stop - start, -1).sum(-1)
                if result.fenergy is not None
                else lane_energy
            )
            lane_weights = weights[start:stop]
            weighted_terms = torch.where(
                lane_weights != 0.0,
                lane_energy * lane_weights,
                torch.zeros_like(lane_energy),
            )
            objective = weighted_terms.sum()

            gradients = torch.autograd.grad(
                objective,
                target_values,
                allow_unused=True,
                materialize_grads=False,
            )
            position_contribution = gradients[0]
            if position_contribution is not None:
                position_gradient[start:stop] += position_contribution[
                    start:stop
                ].detach()

            for index, gradient in zip(
                active_external_indices,
                gradients[1 : 1 + len(active_external_indices)],
            ):
                if gradient is not None:
                    assert external_gradients[index] is not None
                    external_gradients[index] += gradient.detach()

            parameter_offset = 1 + len(active_external_indices)
            for index, gradient in enumerate(gradients[parameter_offset:]):
                if gradient is None:
                    continue
                parameter_connected[index] = True
                if parameter_gradients[index] is None:
                    parameter_gradients[index] = gradient.detach().clone()
                else:
                    parameter_gradients[index] += gradient.detach()

            energy_chunks.append(lane_energy.detach())
            free_energy_chunks.append(lane_free_energy.detach())
            convergence_chunks.append(result.converged.detach())
            if result.status is None:
                chunk_failure = torch.zeros(
                    stop - start, dtype=torch.int64, device=device
                )
            else:
                chunk_failure = result.status.final_failure.detach().clone()
            chunk_pos_grad = position_gradient[start:stop].reshape(
                stop - start, -1
            )
            response_failed = (
                result.converged.detach()
                & (lane_weights != 0.0)
                & (chunk_failure == int(BatchRootFailureCode.NONE))
                & ~torch.isfinite(chunk_pos_grad).all(dim=-1)
            )
            chunk_failure = torch.where(
                response_failed,
                torch.full_like(
                    chunk_failure, int(BatchRootFailureCode.RESPONSE_FAILURE)
                ),
                chunk_failure,
            )
            failure_chunks.append(chunk_failure)
            iteration_chunks.append(result.iterations.detach())

            del gradients, result, objective, lane_energy, lane_free_energy
            del chunk_prepared, chunk_positions

    return StreamingBatchValueAndGrad(
        energy=torch.cat(energy_chunks),
        free_energy=torch.cat(free_energy_chunks),
        converged=torch.cat(convergence_chunks),
        final_failure=torch.cat(failure_chunks),
        iterations=torch.cat(iteration_chunks),
        position_gradient=position_gradient.detach(),
        charge_gradient=(
            None
            if external_gradients[0] is None
            else external_gradients[0].detach()
        ),
        spin_gradient=(
            None
            if external_gradients[1] is None
            else external_gradients[1].detach()
        ),
        field_gradient=(
            None
            if external_gradients[2] is None
            else external_gradients[2].detach()
        ),
        field_grad_gradient=(
            None
            if external_gradients[3] is None
            else external_gradients[3].detach()
        ),
        model_parameter_gradients=tuple(parameter_gradients),
        model_parameter_connected=tuple(parameter_connected),
    )
