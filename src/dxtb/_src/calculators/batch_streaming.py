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

import warnings
from dataclasses import dataclass, replace

import torch

from dxtb._src.calculators.batch_implicit import (
    _UNSET,
    RESPONSE_NOT_RUN,
    BatchRootFailureCode,
    PreparedConvergedRootBatch,
    ResponseDiagnosticCollector,
    _collect_system_and_model_tensors,
    _matches_snapshot,
    _resolve_strictness,
    _snapshot_required_tensor,
    _tensor_version,
    classify_batch_failure_category,
    classify_response_failure_status,
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.calculators.model import Model, System
from dxtb._src.constants import defaults
from dxtb._src.io import OutputHandler
from dxtb._src.scf.functional.spectral import eager_primal_fermi_scope
from dxtb._src.scf.status import SCFStatus
from dxtb._src.typing import Any, Tensor
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)

__all__ = ["StreamingBatchValueAndGrad", "stream_batch_value_and_grad"]


@dataclass(frozen=True, eq=False)
class StreamingBatchValueAndGrad:
    """Detached objective values, primal/response status, and first-order gradients."""

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
    response_converged: Tensor | None = None
    response_status: Tensor | None = None
    response_iterations: Tensor | None = None
    response_residual: Tensor | None = None
    failure_category: tuple[str, ...] = ()


def _is_batched_external(
    value: Any,
    batch_size: int,
    tail_ndim: int,
) -> bool:
    """Return True if an external input carries a leading conformer dimension."""
    if not isinstance(value, torch.Tensor):
        return False
    if tail_ndim == 0:
        if value.ndim == 2 and value.shape == (batch_size, 1):
            return True
        if value.ndim == 1 and value.shape[0] == batch_size:
            return batch_size > 1
        return False
    return value.ndim == tail_ndim + 1 and value.shape[0] == batch_size


def _prepare_chunk_external(
    value: Any,
    target: Tensor | None,
    indices: list[int],
    batch_size: int,
    tail_ndim: int,
    *,
    with_grad: bool,
) -> tuple[Any, Tensor | None]:
    """Prepare the chunk external input and optional per-lane intermediate."""
    if value is _UNSET or value is None:
        return value, None
    n_lanes = len(indices)
    if not isinstance(value, torch.Tensor):
        return value, None
    source = target if (with_grad and target is not None) else value.detach()
    if _is_batched_external(source, batch_size, tail_ndim):
        sliced = source[indices]
        if with_grad and target is not None:
            chunk_inter = sliced.clone()
            return chunk_inter, chunk_inter
        return sliced, None
    if not with_grad or target is None:
        return source, None
    if tail_ndim == 0:
        if source.ndim == 0:
            chunk_inter = source.unsqueeze(0).expand(n_lanes).clone()
        elif source.ndim == 1 and source.shape[0] == 1:
            chunk_inter = source.unsqueeze(0).expand(n_lanes, 1).clone()
        else:
            # Let downstream shape validation raise the canonical ValueError.
            return source, None
    else:
        if source.ndim == tail_ndim:
            chunk_inter = (
                source.unsqueeze(0).expand(n_lanes, *source.shape).clone()
            )
        else:
            return source, None
    return chunk_inter, chunk_inter


def _validate_model_and_prepared(
    prepared: PreparedConvergedRootBatch,
    model: Model,
) -> tuple[System, tuple[Tensor, ...]]:
    """Verify Model ownership and untouched System setup tensors."""
    system = prepared.system
    if not isinstance(system, System):
        raise TypeError(
            "Streaming gradients currently require a single-System prepared "
            "batch; padded exact-core batches are unsupported."
        )

    if (
        model.config.method != system.config.method
        or model.config.scf != system.config.scf
        or repr(model.config) != prepared.config_repr
    ):
        raise ValueError(
            "The supplied Model does not match the prepared SCF configuration."
        )

    all_float_params = tuple(
        parameter
        for parameter in model.par.parameters()
        if parameter.is_floating_point()
    )
    model_parameters = tuple(
        parameter for parameter in all_float_params if parameter.requires_grad
    )
    prepared_model_ids = {id(value) for value in prepared.model_parameter_refs}
    if prepared_model_ids and (
        not prepared_model_ids.issubset({id(p) for p in all_float_params})
        or not prepared_model_ids.issubset({id(p) for p in model_parameters})
    ):
        raise ValueError(
            "The supplied Model does not own the prepared differentiable "
            "parameter leaves."
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

    return system, model_parameters


def _subset_prepared(
    prepared: PreparedConvergedRootBatch,
    model: Model,
    indices: list[int],
) -> PreparedConvergedRootBatch:
    """Pair a subset of prepared roots with a fresh Model setup graph."""
    system = prepared.system
    assert isinstance(system, System)
    fresh_system = model.setup(system.numbers, batch_mode=system.batch_mode)
    fresh_core = fresh_system._active_exact_core()
    fresh_excluded_ids = {
        id(fresh_system.numbers),
        *({id(fresh_core.numbers)} if fresh_core is not None else set()),
    }
    fresh_system_tensor_refs, fresh_model_param_refs = (
        _collect_system_and_model_tensors(fresh_system, fresh_excluded_ids)
    )
    if len(fresh_system_tensor_refs) != len(
        prepared.system_tensor_snapshot
    ) or any(
        not _matches_snapshot(value, snapshot)
        for value, snapshot in zip(
            fresh_system_tensor_refs, prepared.system_tensor_snapshot
        )
    ):
        raise RuntimeError(
            "Prepared physical roots are stale for the supplied Model or "
            "because System setup tensors were modified; prepare roots again."
        )

    lanes = tuple(prepared.lanes[i] for i in indices)
    sub_positions = prepared.positions[indices]
    sub_mask = prepared.valid_lane_mask[indices]
    return replace(
        prepared,
        system=fresh_system,
        model_parameter_refs=fresh_model_param_refs,
        model_parameter_snapshot=tuple(
            _snapshot_required_tensor(value) for value in fresh_model_param_refs
        ),
        model_parameter_versions=tuple(
            _tensor_version(value) for value in fresh_model_param_refs
        ),
        system_tensor_refs=fresh_system_tensor_refs,
        system_tensor_versions=tuple(None for _ in fresh_system_tensor_refs),
        positions=sub_positions,
        lanes=lanes,
        config_identity=id(fresh_system.config),
        positions_snapshot=prepared.positions_snapshot[indices],
        lane_input_snapshots=tuple(
            prepared.lane_input_snapshots[i] for i in indices
        ),
        parameter_snapshots=tuple(
            prepared.parameter_snapshots[i] for i in indices
        ),
        root_snapshots=tuple(prepared.root_snapshots[i] for i in indices),
        response_root_snapshots=tuple(
            prepared.response_root_snapshots[i] for i in indices
        ),
        input_tensor_versions=tuple(
            prepared.input_tensor_versions[i] for i in indices
        ),
        valid_lane_mask=sub_mask,
        all_roots_valid=bool(sub_mask.all()),
    )


def _input_target(value: Any, seen_leaf_ids: set[int]) -> Tensor | None:
    """Create or reuse a differentiable leaf target for an explicit floating input."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        return None
    if value.is_leaf and value.requires_grad and id(value) not in seen_leaf_ids:
        seen_leaf_ids.add(id(value))
        return value
    return value.detach().clone().requires_grad_(True)


def stream_batch_value_and_grad(
    prepared: PreparedConvergedRootBatch | System,
    model: Model,
    positions: Tensor,
    *,
    weights: Tensor | None = None,
    chunk_size: int = 1,
    chrg: Any = _UNSET,
    spin: Any = _UNSET,
    field: Any = _UNSET,
    field_grad: Any = _UNSET,
    strict: bool | None = None,
    eager_primal_fermi: bool | str = False,
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

    if isinstance(prepared, System):
        prepared = prepare_converged_root_batch(
            prepared,
            positions,
            chrg=defaults.CHRG if chrg is _UNSET else chrg,
            spin=defaults.SPIN if spin is _UNSET else spin,
            field=None if field is _UNSET else field,
            field_grad=None if field_grad is _UNSET else field_grad,
            strict=strict,
            eager_primal_fermi=eager_primal_fermi,
        )
    elif not isinstance(prepared, PreparedConvergedRootBatch):
        raise TypeError(
            "Expected PreparedConvergedRootBatch or System as first argument."
        )

    if positions.ndim != 3 or positions.shape[0] != len(prepared.lanes):
        raise ValueError(
            "positions must have shape (batch, nat, 3) matching prepared roots."
        )

    batch_size = positions.shape[0]
    device, dtype = positions.device, positions.dtype
    if weights is None:
        weights = torch.ones(batch_size, device=device, dtype=dtype)
    elif not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor or None.")
    elif tuple(weights.shape) != (batch_size,):
        raise ValueError(f"weights must have shape ({batch_size},).")
    elif weights.is_complex() or weights.dtype == torch.bool:
        raise ValueError(
            "weights must be a real floating-point or numeric tensor."
        )
    else:
        weights = weights.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must contain only finite values.")

    _, model_parameters = _validate_model_and_prepared(prepared, model)

    seen_leaf_ids: set[int] = {id(p) for p in model_parameters}
    position_target = _input_target(positions, seen_leaf_ids)
    assert position_target is not None
    external_values = (chrg, spin, field, field_grad)
    tail_dimensions = (0, 0, 1, 2)
    external_names = ("chrg", "spin", "field", "field_grad")
    external_targets = tuple(
        _input_target(value, seen_leaf_ids) for value in external_values
    )
    active_external_indices = tuple(
        index
        for index, target in enumerate(external_targets)
        if target is not None
    )

    energy_out = torch.empty(batch_size, dtype=dtype, device=device)
    free_energy_out = torch.empty(batch_size, dtype=dtype, device=device)
    converged_out = torch.empty(batch_size, dtype=torch.bool, device=device)
    failure_out = torch.empty(batch_size, dtype=torch.int64, device=device)
    iterations_out = torch.empty(batch_size, dtype=torch.int64, device=device)
    resp_converged_out = torch.zeros(
        batch_size, dtype=torch.bool, device=device
    )
    resp_status_out = torch.full(
        (batch_size,), RESPONSE_NOT_RUN, dtype=torch.int64, device=device
    )
    resp_iterations_out = torch.zeros(
        batch_size, dtype=torch.int64, device=device
    )
    resp_residual_out = torch.full(
        (batch_size,), float("nan"), dtype=dtype, device=device
    )

    position_gradient = torch.zeros_like(position_target)
    external_gradients: list[Tensor | None] = [
        None if target is None else torch.zeros_like(target)
        for target in external_targets
    ]
    parameter_gradients: list[Tensor | None] = [None for _ in model_parameters]
    parameter_connected = [False for _ in model_parameters]

    with eager_primal_fermi_scope(False):
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            chunk_indices = list(range(start, stop))
            active_indices = [
                idx
                for idx in chunk_indices
                if bool((weights[idx] != 0.0).item())
            ]
            zero_indices = [
                idx
                for idx in chunk_indices
                if not bool((weights[idx] != 0.0).item())
            ]

            if zero_indices:
                with torch.no_grad():
                    zero_prepared = _subset_prepared(
                        prepared, model, zero_indices
                    )
                    zero_pos = positions.detach()[zero_indices]
                    zero_kwargs: dict[str, Any] = {}
                    for name, value, target, tail_ndim in zip(
                        external_names,
                        external_values,
                        external_targets,
                        tail_dimensions,
                    ):
                        if value is not _UNSET:
                            ov, _ = _prepare_chunk_external(
                                value,
                                target,
                                zero_indices,
                                batch_size,
                                tail_ndim,
                                with_grad=False,
                            )
                            zero_kwargs[name] = ov
                    zero_res = evaluate_converged_root_batch(
                        zero_prepared,
                        positions=zero_pos,
                        strict=strict,
                        **zero_kwargs,
                    )
                    z_idx = torch.tensor(
                        zero_indices, dtype=torch.long, device=device
                    )
                    n_zero = len(zero_indices)
                    z_energy = zero_res.energy.reshape(n_zero, -1).sum(-1)
                    z_fenergy = (
                        zero_res.fenergy.reshape(n_zero, -1).sum(-1)
                        if zero_res.fenergy is not None
                        else z_energy
                    )
                    energy_out[z_idx] = z_energy.detach()
                    free_energy_out[z_idx] = z_fenergy.detach()
                    converged_out[z_idx] = zero_res.converged.detach()
                    failure_out[z_idx] = (
                        torch.zeros(n_zero, dtype=torch.int64, device=device)
                        if zero_res.status is None
                        else zero_res.status.final_failure.detach().clone()
                    )
                    iterations_out[z_idx] = zero_res.iterations.detach()
                    del zero_res, zero_prepared, zero_pos, z_energy, z_fenergy

            if not active_indices:
                continue

            with torch.enable_grad():
                n_active = len(active_indices)
                a_idx = torch.tensor(
                    active_indices, dtype=torch.long, device=device
                )
                chunk_positions = position_target[active_indices].clone()
                chunk_prepared = _subset_prepared(
                    prepared, model, active_indices
                )
                active_kwargs: dict[str, Any] = {}
                ext_chunk_intermediates: list[Tensor] = []
                ext_chunk_slots: list[int] = []
                for ext_slot, (name, value, target, tail_ndim) in enumerate(
                    zip(
                        external_names,
                        external_values,
                        external_targets,
                        tail_dimensions,
                    )
                ):
                    if value is not _UNSET:
                        ov, chunk_inter = _prepare_chunk_external(
                            value,
                            target,
                            active_indices,
                            batch_size,
                            tail_ndim,
                            with_grad=True,
                        )
                        active_kwargs[name] = ov
                        if chunk_inter is not None:
                            ext_chunk_intermediates.append(chunk_inter)
                            ext_chunk_slots.append(ext_slot)

                collector = ResponseDiagnosticCollector()
                result = evaluate_converged_root_batch(
                    chunk_prepared,
                    positions=chunk_positions,
                    strict=strict,
                    response_diagnostics=collector,
                    **active_kwargs,
                )
                lane_energy = result.energy.reshape(n_active, -1).sum(-1)
                lane_free_energy = (
                    result.fenergy.reshape(n_active, -1).sum(-1)
                    if result.fenergy is not None
                    else lane_energy
                )
                lane_weights = weights[a_idx]
                weighted_terms = lane_energy * lane_weights
                objective = weighted_terms.sum()

                n_ext_leaves = len(active_external_indices)
                n_ext_inter = len(ext_chunk_intermediates)
                n_model = len(model_parameters)
                grad_targets = (
                    position_target,
                    chunk_positions,
                    *(
                        external_targets[index]
                        for index in active_external_indices
                    ),
                    *ext_chunk_intermediates,
                    *model_parameters,
                )
                need_retain = n_active > 1
                gradients = torch.autograd.grad(
                    objective,
                    grad_targets,
                    retain_graph=need_retain,
                    allow_unused=True,
                    materialize_grads=False,
                )

                grad_pos_leaf = gradients[0]
                grad_pos_chunk = gradients[1]
                pos_contrib = (
                    grad_pos_leaf[a_idx]
                    if grad_pos_leaf is not None
                    else grad_pos_chunk
                )
                if pos_contrib is not None:
                    position_gradient[a_idx] += pos_contrib.detach()
                    lane_pos_grad_for_status = pos_contrib.detach()
                else:
                    lane_pos_grad_for_status = torch.zeros_like(chunk_positions)

                ext_leaf_grads = gradients[2 : 2 + n_ext_leaves]
                ext_inter_grads = gradients[
                    2 + n_ext_leaves : 2 + n_ext_leaves + n_ext_inter
                ]
                model_grads = gradients[2 + n_ext_leaves + n_ext_inter :]

                for index, gradient in zip(
                    active_external_indices, ext_leaf_grads
                ):
                    if gradient is not None:
                        assert external_gradients[index] is not None
                        external_gradients[index] += gradient.detach()

                per_lane_ext_grads: list[Tensor] = []
                ext_inter_map = {
                    slot: grad
                    for slot, grad in zip(ext_chunk_slots, ext_inter_grads)
                }
                for index, leaf_grad in zip(
                    active_external_indices, ext_leaf_grads
                ):
                    inter_grad = ext_inter_map.get(index)
                    target = external_targets[index]
                    assert target is not None
                    tail_ndim = tail_dimensions[index]
                    if leaf_grad is not None and _is_batched_external(
                        target, batch_size, tail_ndim
                    ):
                        per_lane_ext_grads.append(leaf_grad.detach()[a_idx])
                    elif inter_grad is not None:
                        per_lane_dt = inter_grad.detach()
                        if (
                            leaf_grad is not None
                            and not bool(torch.isfinite(leaf_grad).all())
                            and bool(torch.isfinite(per_lane_dt).all())
                        ):
                            if n_active == 1:
                                per_lane_dt = torch.full_like(
                                    per_lane_dt, float("nan")
                                )
                            else:
                                collector.frozen = True
                                hook_finite = torch.ones(
                                    n_active,
                                    dtype=torch.bool,
                                    device=device,
                                )
                                for local_i in range(n_active):
                                    lg = torch.autograd.grad(
                                        weighted_terms[local_i],
                                        (target,),
                                        retain_graph=True,
                                        allow_unused=True,
                                    )[0]
                                    if lg is not None and not bool(
                                        torch.isfinite(lg).all()
                                    ):
                                        hook_finite[local_i] = False
                                per_lane_dt = torch.where(
                                    hook_finite.reshape(
                                        n_active,
                                        *(1 for _ in per_lane_dt.shape[1:]),
                                    ),
                                    per_lane_dt,
                                    torch.full_like(per_lane_dt, float("nan")),
                                )
                        per_lane_ext_grads.append(per_lane_dt)

                failed_model_params: list[Tensor] = []
                for index, gradient in enumerate(model_grads):
                    if gradient is None:
                        continue
                    parameter_connected[index] = True
                    grad_d = gradient.detach()
                    if parameter_gradients[index] is None:
                        parameter_gradients[index] = grad_d.clone()
                    else:
                        parameter_gradients[index] += grad_d
                    if not bool(torch.isfinite(grad_d).all()):
                        failed_model_params.append(model_parameters[index])

                valid_local_indices = [
                    local_i
                    for local_i, lane in enumerate(chunk_prepared.lanes)
                    if lane.root_valid
                ]
                diag = collector.finalize(
                    valid_local_indices, n_active, device, dtype
                )

                lane_model_grad_sentinel = torch.zeros(
                    (n_active, 1), dtype=dtype, device=device
                )
                if failed_model_params:
                    if n_active == 1:
                        lane_model_grad_sentinel.fill_(float("nan"))
                    else:
                        collector.frozen = True
                        for local_i, lane_idx in enumerate(active_indices):
                            if not chunk_prepared.lanes[
                                local_i
                            ].root_valid or bool(
                                diag.confirmed_failure[local_i]
                            ):
                                lane_model_grad_sentinel[local_i] = float("nan")
                                continue
                            single_pos = (
                                positions.detach()[lane_idx : lane_idx + 1]
                                .clone()
                                .requires_grad_(True)
                            )
                            single_prep = _subset_prepared(
                                prepared, model, [lane_idx]
                            )
                            single_kwargs: dict[str, Any] = {}
                            for (
                                name,
                                value,
                                target,
                                tail_ndim,
                            ) in zip(
                                external_names,
                                external_values,
                                external_targets,
                                tail_dimensions,
                            ):
                                if value is not _UNSET:
                                    single_kwargs[name] = (
                                        _prepare_chunk_external(
                                            value,
                                            target,
                                            [lane_idx],
                                            batch_size,
                                            tail_ndim,
                                            with_grad=False,
                                        )[0]
                                    )
                            single_res = evaluate_converged_root_batch(
                                single_prep,
                                positions=single_pos,
                                strict=strict,
                                **single_kwargs,
                            )
                            single_obj = (
                                single_res.energy.reshape(-1).sum()
                                * weights[lane_idx]
                            )
                            if single_obj.requires_grad:
                                lp_grads = torch.autograd.grad(
                                    single_obj,
                                    tuple(failed_model_params),
                                    allow_unused=True,
                                )
                                if any(
                                    g is not None
                                    and not bool(torch.isfinite(g).all())
                                    for g in lp_grads
                                ):
                                    lane_model_grad_sentinel[local_i] = float(
                                        "nan"
                                    )
                base_status = (
                    SCFStatus(
                        trajectory_failure=torch.zeros(
                            n_active, dtype=torch.int64, device=device
                        ),
                        final_failure=torch.zeros(
                            n_active, dtype=torch.int64, device=device
                        ),
                    )
                    if result.status is None
                    else result.status
                )
                post_status = classify_response_failure_status(
                    base_status,
                    result.converged.detach(),
                    lane_pos_grad_for_status,
                    *per_lane_ext_grads,
                    lane_model_grad_sentinel,
                    response_diagnostics=diag,
                )

                effective_strict = _resolve_strictness(model.config.scf, strict)
                for local_i, lane_idx in enumerate(active_indices):
                    code = int(post_status.final_failure[local_i].item())
                    if code == int(BatchRootFailureCode.RESPONSE_FAILURE):
                        position_gradient[lane_idx].fill_(float("nan"))
                        for index in active_external_indices:
                            ext_grad = external_gradients[index]
                            target = external_targets[index]
                            tail_ndim = tail_dimensions[index]
                            if ext_grad is not None and target is not None:
                                if _is_batched_external(
                                    target, batch_size, tail_ndim
                                ):
                                    ext_grad[lane_idx].fill_(float("nan"))
                                else:
                                    ext_grad.fill_(float("nan"))
                        for index, connected in enumerate(parameter_connected):
                            if (
                                connected
                                and parameter_gradients[index] is not None
                            ):
                                parameter_gradients[index].fill_(float("nan"))
                        reason = (
                            f"[Lane {lane_idx}] Implicit linear-response solve "
                            f"failed to converge (status={int(diag.status[local_i].item())})."
                        )
                        if effective_strict:
                            raise SCFConvergenceError(reason)
                        warnings.warn(
                            reason, SCFConvergenceWarning, stacklevel=2
                        )
                        OutputHandler.warn(reason, SCFConvergenceWarning)
                    elif code == int(BatchRootFailureCode.NONFINITE_DERIVATIVE):
                        reason = (
                            f"[Lane {lane_idx}] Nonfinite derivative encountered "
                            "outside linear-response solve."
                        )
                        if effective_strict:
                            raise SCFConvergenceError(reason)
                        warnings.warn(
                            reason, SCFConvergenceWarning, stacklevel=2
                        )
                        OutputHandler.warn(reason, SCFConvergenceWarning)

                energy_out[a_idx] = lane_energy.detach()
                free_energy_out[a_idx] = lane_free_energy.detach()
                converged_out[a_idx] = result.converged.detach()
                failure_out[a_idx] = post_status.final_failure.detach()
                iterations_out[a_idx] = result.iterations.detach()
                resp_converged_out[a_idx] = diag.converged.detach()
                resp_status_out[a_idx] = diag.status.detach()
                resp_iterations_out[a_idx] = diag.iterations.detach()
                resp_residual_out[a_idx] = diag.residual.detach()

                del (
                    gradients,
                    grad_pos_leaf,
                    grad_pos_chunk,
                    ext_leaf_grads,
                    ext_inter_grads,
                    model_grads,
                    result,
                    objective,
                    weighted_terms,
                    lane_energy,
                    lane_free_energy,
                    chunk_prepared,
                    chunk_positions,
                    ext_chunk_intermediates,
                    collector,
                    diag,
                )

    return StreamingBatchValueAndGrad(
        energy=energy_out,
        free_energy=free_energy_out,
        converged=converged_out,
        final_failure=failure_out,
        iterations=iterations_out,
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
        response_converged=resp_converged_out,
        response_status=resp_status_out,
        response_iterations=resp_iterations_out,
        response_residual=resp_residual_out,
        failure_category=classify_batch_failure_category(failure_out),
    )
