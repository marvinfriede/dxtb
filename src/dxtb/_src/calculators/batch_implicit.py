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
"""
Calculators: Converged-Root Batch Implicit SCF
==============================================

Internal orchestration for conformer batches of a single :class:`System`
using independent physical-root preparation, explicit root validation,
and mapped implicit differentiation.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import IntEnum
from types import SimpleNamespace

import torch
from tad_mctc.exceptions import DeviceError, DtypeError
from tad_mctc.storch.implicit import implicit_fixed_point
from tad_mctc.storch.linsolve import (
    STATUS_CONVERGED,
    STATUS_NONFINITE,
    LinearSolveResult,
    bicgstab,
)
from tad_mctc.tools.compile import is_compiling
from torch.func import vmap

from dxtb import OutputHandler, labels
from dxtb._src.calculators.model import System
from dxtb._src.calculators.padded_integrals import evaluate_padded_integrals
from dxtb._src.calculators.padding import PaddedExactSystemCore
from dxtb._src.calculators.result import Result
from dxtb._src.calculators.singlepoint import (
    _evaluate_classical_descriptor,
    _floating_setup_tensor,
    _interaction_data,
)
from dxtb._src.components.classicals.dispersion.d3 import (
    DispersionD3,
    DispersionD3CoreSetup,
    dispersion_d3_energy,
)
from dxtb._src.components.classicals.dispersion.d4 import (
    DispersionD4,
    DispersionD4CoreSetup,
    dispersion_d4_energy,
)
from dxtb._src.components.classicals.halogen import (
    Halogen,
    HalogenCoreSetup,
    halogen_energy,
)
from dxtb._src.components.classicals.ies import IES, IESCoreSetup, ies_energy
from dxtb._src.components.classicals.repulsion.rep import (
    Repulsion,
    RepulsionCoreSetup,
)
from dxtb._src.components.classicals.shortrangebond import (
    ShortRangeBond,
    ShortRangeBondCoreSetup,
)
from dxtb._src.components.interactions.field.efield import (
    LABEL_EFIELD,
    ElectricField,
)
from dxtb._src.components.interactions.field.efieldgrad import (
    LABEL_EFIELD_GRAD,
    ElectricFieldGrad,
)
from dxtb._src.components.interactions.list import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.constants import defaults
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional.certification import (
    certify_scf_core_state,
    resolve_certification_thresholds,
)
from dxtb._src.scf.functional.core import (
    CoreIndexData,
    replace_tensor_leaves,
    tensor_leaves,
)
from dxtb._src.scf.functional.data import (
    PotentialLayout,
    SCFCharges,
    SCFInputs,
    SCFPotential,
    pack_charges,
    setup_scf_inputs,
)
from dxtb._src.scf.functional.eager import (
    _padded_charge_projections,
    _padded_potential_projections,
    _prepare_solver,
    public_scf_status,
    resolve_full_scf_route,
)
from dxtb._src.scf.functional.implicit import (
    _LINEAR_SOLVE_ATOL,
    _LINEAR_SOLVE_RTOL,
    _SCFMapTensorTree,
)
from dxtb._src.scf.functional.implicit import (
    implicit_root as _functional_implicit_root,
)
from dxtb._src.scf.functional.map import (
    evaluate_hamiltonian,
    evaluate_potential,
    fock_initial_guess,
    fock_map,
    potential_initial_guess,
    potential_map,
    scf_map,
    unpack_potential,
)
from dxtb._src.scf.functional.padded import (
    _padded_initial_guess,
    setup_padded_scf_inputs,
)
from dxtb._src.scf.functional.problem import (
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
)
from dxtb._src.scf.functional.response import (
    SCFResponseSettings,
    prepare_response_root,
)
from dxtb._src.scf.functional.solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_INVALID_TEMPERATURE,
    FERMI_FAILURE_NONCONVERGENCE,
    FERMI_FAILURE_NONE,
    FERMI_FAILURE_TOO_MANY_ELECTRONS,
    DampingState,
    SCFTrajectoryStatus,
    _project_potential_padding,
    _solve_map,
    fermi_failure_code,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
)
from dxtb._src.scf.functional.spectral import eager_primal_fermi_scope
from dxtb._src.scf.functional.strategy import SCFStrategy
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.scf.status import SCFStatus
from dxtb._src.typing import Any, Tensor
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)
from dxtb._src.wavefunction.filling import FermiStatus

__all__ = [
    "RESPONSE_NOT_RUN",
    "BatchResponseDiagnostics",
    "BatchRootFailureCode",
    "ConvergedRootLaneRecord",
    "PreparedConvergedRootBatch",
    "ResponseDiagnosticCollector",
    "classify_batch_failure_category",
    "classify_response_failure_status",
    "evaluate_converged_root_batch",
    "prepare_converged_root_batch",
]

_ELECTRON_COUNT_TOL = 1.0e-6
_UNSET: Any = object()
RESPONSE_NOT_RUN: int = -1


class BatchRootFailureCode(IntEnum):
    """Failure categories for converged-root batch validation and response.

    Codes ``1`` through ``8`` represent primal root-preparation failures and
    are recorded in ``Result.status``. Code ``9`` (``RESPONSE_FAILURE``)
    identifies a confirmed implicit linear-response failure during
    differentiation, whereas code ``10`` (``NONFINITE_DERIVATIVE``) identifies
    a nonfinite derivative whose origin is outside or not attributable to a
    failed linear-response solve. Because ``Result`` is an immutable primal
    snapshot constructed prior to derivative execution, ``Result.converged``
    and ``Result.status`` remain primal-root diagnostics while failed response
    solves produce lane-local ``NaN`` derivatives when selected and ``0.0``
    when masked.
    """

    NONE = 0
    INVALID_TEMPERATURE = 1
    TOO_MANY_ELECTRONS = 2
    FERMI_NONCONVERGENCE = 3
    ELECTRON_COUNT = 4
    MAXITER_EXCEEDED = 5
    RESIDUAL_TOLERANCE = 6
    NONFINITE_STATE = 7
    INVALID_PADDING = 8
    RESPONSE_FAILURE = 9
    NONFINITE_DERIVATIVE = 10


def classify_batch_failure_category(final_failure: Tensor) -> tuple[str, ...]:
    """Map per-lane ``BatchRootFailureCode`` values to canonical category names."""
    categories: list[str] = []
    for raw_code in final_failure.detach().reshape(-1):
        code = int(raw_code)
        if code == int(BatchRootFailureCode.NONE):
            categories.append("success")
        elif code in (
            int(BatchRootFailureCode.INVALID_TEMPERATURE),
            int(BatchRootFailureCode.TOO_MANY_ELECTRONS),
            int(BatchRootFailureCode.FERMI_NONCONVERGENCE),
            int(BatchRootFailureCode.ELECTRON_COUNT),
        ):
            categories.append("fermi_failure")
        elif code in (
            int(BatchRootFailureCode.MAXITER_EXCEEDED),
            int(BatchRootFailureCode.RESIDUAL_TOLERANCE),
            int(BatchRootFailureCode.NONFINITE_STATE),
            int(BatchRootFailureCode.INVALID_PADDING),
        ):
            categories.append("primal_scf_failure")
        elif code == int(BatchRootFailureCode.RESPONSE_FAILURE):
            categories.append("response_failure")
        elif code == int(BatchRootFailureCode.NONFINITE_DERIVATIVE):
            categories.append("nonfinite_derivative")
        else:
            raise ValueError(f"Unknown BatchRootFailureCode: {code}.")
    return tuple(categories)


@dataclass(frozen=True, eq=False)
class BatchResponseDiagnostics:
    """Call-local linear-response diagnostics for a conformer batch."""

    executed: Tensor
    converged: Tensor
    status: Tensor
    iterations: Tensor
    residual: Tensor
    target: Tensor
    incoming_finite: Tensor
    outgoing_finite: Tensor

    @property
    def confirmed_failure(self) -> Tensor:
        """True for lanes where the implicit linear-response solve itself failed."""
        return (
            self.executed
            & self.incoming_finite
            & (~self.converged | ~self.outgoing_finite)
        )


def implicit_root(
    root_x: Tensor,
    *parameters: Tensor,
    raw_map: Callable[..., Tensor],
    method: Any = "bicgstab",
    rtol: float = _LINEAR_SOLVE_RTOL,
    atol: float = _LINEAR_SOLVE_ATOL,
    maxiter: int | None = None,
) -> Tensor:
    """Attach implicit fixed-point derivatives with optional custom linear solver."""
    if method == "bicgstab":
        if (
            rtol == _LINEAR_SOLVE_RTOL
            and atol == _LINEAR_SOLVE_ATOL
            and maxiter is None
        ):
            return _functional_implicit_root(
                root_x, *parameters, raw_map=raw_map
            )
        return _functional_implicit_root(
            root_x,
            *parameters,
            raw_map=raw_map,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
    return implicit_fixed_point(
        raw_map,
        root_x,
        *parameters,
        method=method,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
    )


_BATCH_DEFAULT_IMPLICIT_ROOT = implicit_root


@dataclass(frozen=True, eq=False)
class ConvergedRootLaneRecord:
    """Call-local prepared physical root and validation diagnostics for one lane."""

    lane_index: int
    positions: Tensor
    charge: Tensor
    spin: Tensor | None
    field: Tensor | None
    field_grad: Tensor | None
    integrals: IntegralMatrices
    refocc: Tensor
    overlap_norm: Tensor
    n0: Tensor
    occupation: Tensor
    nel: Tensor
    interactions: InteractionList | None
    interaction_data: InteractionListCache | None
    inputs: SCFInputs
    potential_layout: PotentialLayout | None
    adapter: _SCFMapTensorTree
    parameters: tuple[Tensor, ...]
    root_x: Tensor
    response_adapter: _SCFMapTensorTree | None
    response_parameters: tuple[Tensor, ...] | None
    response_root_x: Tensor | None
    response_potential_layout: PotentialLayout | None
    response_certification: Any | None
    iterations: Tensor
    solver_converged: Tensor
    solver_residual: Tensor
    raw_residual_l2: Tensor
    raw_residual_max: Tensor
    electron_count_error: Tensor
    density_finite: bool
    padding_valid: bool
    trajectory_status: SCFTrajectoryStatus | None
    fermi_status: FermiStatus | None
    damping_state: DampingState
    status: SCFStatus
    failure_code: int
    failure_reason: str | None
    root_valid: bool


@dataclass(frozen=True, eq=False)
class _TensorValueSnapshot:
    """Immutable exact-value record for a prepared physical tensor."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    raw_bytes: bytes

    def to_tensor(self, device: torch.device | None = None) -> Tensor:
        """Reconstruct an ordinary detached tensor from raw snapshot bytes."""
        target_device = self.device if device is None else device
        if not self.raw_bytes:
            return torch.empty(
                self.shape, dtype=self.dtype, device=target_device
            )
        return (
            torch.frombuffer(bytearray(self.raw_bytes), dtype=torch.uint8)
            .clone()
            .view(self.dtype)
            .reshape(self.shape)
            .to(device=target_device)
        )

    def __getitem__(self, index: Any) -> _TensorValueSnapshot:
        """Slice a snapshot tensor without retaining autograd state."""
        return _snapshot_required_tensor(self.to_tensor()[index])


@dataclass(frozen=True, eq=False)
class PreparedConvergedRootBatch:
    """Validated conformer batch sharing one :class:`System` or padded core owner."""

    system: System | PaddedExactSystemCore
    positions: Tensor
    primal_representation: int
    response_representation: int
    response_settings: SCFResponseSettings
    strategy_label: str
    lanes: tuple[ConvergedRootLaneRecord, ...]
    parameter_paths: tuple[str, ...]
    model_parameter_refs: tuple[Tensor, ...]
    model_parameter_snapshot: tuple[_TensorValueSnapshot, ...]
    model_parameter_versions: tuple[int | None, ...]
    system_tensor_refs: tuple[Tensor, ...]
    system_tensor_snapshot: tuple[_TensorValueSnapshot, ...]
    system_tensor_versions: tuple[int | None, ...]
    positions_snapshot: _TensorValueSnapshot
    positions_version: int | None
    numbers_snapshot: _TensorValueSnapshot
    numbers_version: int | None
    config_identity: int
    config_repr: str
    lane_input_snapshots: tuple[tuple[_TensorValueSnapshot | None, ...], ...]
    parameter_snapshots: tuple[tuple[_TensorValueSnapshot, ...], ...]
    root_snapshots: tuple[_TensorValueSnapshot, ...]
    response_root_snapshots: tuple[_TensorValueSnapshot | None, ...]
    input_tensor_versions: tuple[dict[str, int], ...]
    valid_lane_mask: Tensor
    all_roots_valid: bool


def _unwrap_primal_tensor(x: Tensor) -> Tensor:
    """Peel grad-tracking, batched, and dual layers for detached primal root finding."""
    if is_compiling():
        return x.detach()
    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    current = x
    while True:
        if ft.is_functorch_wrapped_tensor(current):
            current = ft.get_unwrapped(current)
            continue
        with torch._C._DisableFuncTorch():  # pyright: ignore[reportAttributeAccessIssue]
            if not (current.is_floating_point() or current.is_complex()):
                return current.detach()
            dual = torch.autograd.forward_ad.unpack_dual(current)
            if dual.tangent is not None:
                current = dual.primal
                continue
            return current.detach()


def _tensor_version(value: Tensor | None) -> int | None:
    """Return the storage version counter of a primal tensor."""
    if value is None:
        return None
    return int(_unwrap_primal_tensor(value)._version)


def _tensor_raw_bytes(value: Tensor) -> bytes:
    """Extract exact storage bytes from an ordinary detached tensor."""
    primal = _unwrap_primal_tensor(value)
    with torch._C._DisableFuncTorch():  # pyright: ignore[reportAttributeAccessIssue]
        return primal.detach().contiguous().cpu().numpy().tobytes()


def _snapshot_tensor(
    value: Tensor | None,
) -> _TensorValueSnapshot | None:
    """Store immutable exact tensor values without retaining autograd graphs."""
    if value is None:
        return None
    primal = _unwrap_primal_tensor(value)
    with torch._C._DisableFuncTorch():  # pyright: ignore[reportAttributeAccessIssue]
        detached = primal.detach().contiguous()
        raw = detached.cpu().numpy().tobytes()
    return _TensorValueSnapshot(
        shape=tuple(detached.shape),
        dtype=detached.dtype,
        device=detached.device,
        raw_bytes=raw,
    )


def _snapshot_required_tensor(value: Tensor) -> _TensorValueSnapshot:
    """Snapshot a tensor known to be present."""
    snapshot = _snapshot_tensor(value)
    assert snapshot is not None
    return snapshot


def _matches_snapshot(actual: Tensor, expected: _TensorValueSnapshot) -> bool:
    """Compare exact metadata and primal values to an immutable snapshot."""
    primal = _unwrap_primal_tensor(actual)
    if (
        tuple(primal.shape) != expected.shape
        or primal.dtype != expected.dtype
        or primal.device != expected.device
    ):
        return False
    if not is_compiling():
        return _tensor_raw_bytes(primal) == expected.raw_bytes
    reference = expected.to_tensor(device=primal.device)
    if primal.is_floating_point():
        return bool(
            torch.all(
                (primal == reference)
                | (torch.isnan(primal) & torch.isnan(reference))
            )
        )
    return bool(torch.equal(primal, reference))


def _resolve_strictness(config: object, strict: bool | None) -> bool:
    """Keep mandatory configured convergence strictness from being weakened."""
    force = bool(getattr(config, "force_convergence"))
    return force or (strict is True)


def _detach_scf_inputs(inputs: SCFInputs) -> SCFInputs:
    """Create a detached view of SCFInputs for eager no_grad primal solving."""
    changes: dict[str, Any] = {}
    for name in (
        "numbers",
        "n0",
        "nel",
        "occupation",
        "inverse_cholesky",
        "orbital_mask",
    ):
        value = getattr(inputs, name)
        if isinstance(value, Tensor):
            changes[name] = _unwrap_primal_tensor(value)
    integral_changes: dict[str, Tensor | None] = {}
    for name in ("hcore", "overlap", "dipole", "quadrupole"):
        value = getattr(inputs.integrals, name)
        integral_changes[name] = (
            _unwrap_primal_tensor(value) if isinstance(value, Tensor) else value
        )
    integrals = IntegralMatrices(**integral_changes)
    fermi_settings = inputs.fermi_settings
    if fermi_settings is not None:
        fermi_changes = {}
        for name in ("kt", "threshold"):
            value = getattr(fermi_settings, name)
            if isinstance(value, Tensor):
                fermi_changes[name] = _unwrap_primal_tensor(value)
        fermi_settings = replace(fermi_settings, **fermi_changes)
    interaction_changes = {
        path: _unwrap_primal_tensor(value)
        for path, value in tensor_leaves(inputs.interactions)
    }
    interactions = replace_tensor_leaves(
        inputs.interactions, interaction_changes
    )
    return replace(
        inputs,
        integrals=integrals,
        interactions=interactions,
        fermi_settings=fermi_settings,
        **changes,
    )


def _collect_object_tensors(
    obj: object,
    excluded_ids: set[int],
    seen_objs: dict[int, object] | None = None,
    found: dict[int, Tensor] | None = None,
) -> dict[int, Tensor]:
    """Recursively collect all Tensor references held by a System or core setup."""
    if seen_objs is None:
        seen_objs = {}
    if found is None:
        found = {}
    if obj is None or isinstance(obj, (bool, int, float, str, bytes, type)):
        return found
    if isinstance(obj, torch.Tensor):
        if id(obj) not in excluded_ids:
            found[id(obj)] = obj
        return found
    oid = id(obj)
    if oid in seen_objs:
        return found
    seen_objs[oid] = obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_object_tensors(k, excluded_ids, seen_objs, found)
            _collect_object_tensors(v, excluded_ids, seen_objs, found)
    elif isinstance(obj, (tuple, list, set, frozenset)):
        for item in obj:
            _collect_object_tensors(item, excluded_ids, seen_objs, found)
    elif dataclasses.is_dataclass(obj):
        for field_info in dataclasses.fields(obj):
            _collect_object_tensors(
                getattr(obj, field_info.name, None),
                excluded_ids,
                seen_objs,
                found,
            )
    elif hasattr(obj, "__dict__"):
        _collect_object_tensors(vars(obj), excluded_ids, seen_objs, found)
    return found


def _collect_system_and_model_tensors(
    system: System | PaddedExactSystemCore,
    excluded_ids: set[int],
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Collect System setup tensors and connected differentiable Model leaf tensors."""
    found_sys: dict[int, Tensor] = {}
    seen_objs: dict[int, object] = {}
    if isinstance(system, System):
        for attr in (
            "h0_setup",
            "integral_setup",
            "interactions",
            "_exact_core",
            "classicals",
        ):
            _collect_object_tensors(
                getattr(system, attr, None), excluded_ids, seen_objs, found_sys
            )
    _collect_object_tensors(system, excluded_ids, seen_objs, found_sys)
    sys_tensors = tuple(found_sys.values())
    stack = [t.grad_fn for t in reversed(sys_tensors) if t.grad_fn is not None]
    seen_nodes: set[Any] = set()
    leaves: dict[int, Tensor] = {}
    for t in sys_tensors:
        if t.is_leaf and t.requires_grad and id(t) not in excluded_ids:
            leaves[id(t)] = t
    while stack:
        node = stack.pop()
        if node is None or node in seen_nodes:
            continue
        seen_nodes.add(node)
        variable = getattr(node, "variable", None)
        if (
            isinstance(variable, torch.Tensor)
            and variable.is_leaf
            and variable.requires_grad
            and id(variable) not in excluded_ids
        ):
            leaves[id(variable)] = variable
        stack.extend(
            next_node
            for next_node, _ in reversed(node.next_functions)
            if next_node is not None
        )
    return sys_tensors, tuple(leaves.values())


def _normalize_batch_scalar(
    name: str,
    value: Tensor | float | int,
    positions: Tensor,
    batch_size: int,
) -> tuple[Tensor, ...]:
    """Normalize a shared or per-lane scalar input to per-lane (1,) tensors."""
    if isinstance(value, Tensor):
        if value.device != positions.device:
            raise DeviceError(
                f"Device mismatch: positions are on '{positions.device}', "
                f"but {name} is on '{value.device}'."
            )
        if value.dtype != positions.dtype:
            raise DtypeError(
                f"Dtype mismatch: positions are of type '{positions.dtype}', "
                f"but {name} is of type '{value.dtype}'."
            )
        if value.ndim == 0:
            item = value.unsqueeze(0)
            return tuple(item for _ in range(batch_size))
        if value.ndim == 1 and value.shape[0] == batch_size:
            return tuple(value[i].unsqueeze(0) for i in range(batch_size))
        if value.ndim == 1 and value.shape[0] == 1:
            return tuple(value for _ in range(batch_size))
        if value.ndim == 2 and value.shape == (batch_size, 1):
            return tuple(value[i] for i in range(batch_size))
        raise ValueError(
            f"Batch {name} must be a scalar, (1,), ({batch_size},), or "
            f"({batch_size}, 1); got shape {tuple(value.shape)}."
        )
    scalar = torch.tensor(
        [value], dtype=positions.dtype, device=positions.device
    )
    return tuple(scalar for _ in range(batch_size))


def _normalize_batch_perturbation(
    name: str,
    value: Tensor | None,
    positions: Tensor,
    batch_size: int,
    event_shape: tuple[int, ...],
) -> tuple[Tensor | None, ...]:
    """Normalize a shared or per-lane perturbation tensor without severing grad."""
    if value is None:
        return tuple(None for _ in range(batch_size))
    if value.device != positions.device:
        raise DeviceError(
            f"Device mismatch: positions are on '{positions.device}', "
            f"but {name} is on '{value.device}'."
        )
    if value.dtype != positions.dtype:
        raise DtypeError(
            f"Dtype mismatch: positions are of type '{positions.dtype}', "
            f"but {name} is of type '{value.dtype}'."
        )
    if value.shape == event_shape:
        return tuple(value for _ in range(batch_size))
    if value.shape == (batch_size, *event_shape):
        return tuple(value[i] for i in range(batch_size))
    raise ValueError(
        f"{name} must have shape {event_shape} or {(batch_size, *event_shape)}, "
        f"got {tuple(value.shape)}."
    )


def _validate_system_and_positions(
    system: System | PaddedExactSystemCore,
    positions: Tensor,
    field: Tensor | None,
    field_grad: Tensor | None,
):
    """Validate System or PaddedExactSystemCore eligibility and batch positions."""
    if isinstance(system, PaddedExactSystemCore):
        core = system.core
        numbers = core.numbers
        config = core.config
        integral_setup = core.integral
        h0_setup = core.h0_setup
        if numbers.ndim != 1:
            raise ValueError(
                "Converged-root conformer batching requires 1D atomic numbers."
            )
        if positions.ndim != 3 or positions.shape[0] < 1:
            raise ValueError(
                "Converged-root conformer batching expects positions with shape "
                f"(B, nat, 3) and B >= 1; got {tuple(positions.shape)}."
            )
        if positions.shape[1:] != (numbers.shape[0], 3):
            raise ValueError(
                "Converged-root conformer batching expects positions with shape "
                f"(B, {numbers.shape[0]}, 3); got {tuple(positions.shape)}."
            )
        system_device = numbers.device
        if positions.device != system_device:
            raise DeviceError(
                f"Device mismatch: System is on '{system_device}', but positions "
                f"are on '{positions.device}'."
            )
        if positions.dtype != core.effective_dtype:
            raise DtypeError(
                "Dtype mismatch: System floating setup is of type "
                f"'{core.effective_dtype}', but positions are of type "
                f"'{positions.dtype}'."
            )
        if field is not None and config.ints.level < labels.INTLEVEL_DIPOLE:
            raise RuntimeError(
                "An electric field requires dipole integrals. Construct the "
                "System with an integral level of at least "
                f"{labels.INTLEVEL_DIPOLE}."
            )
        if (
            field_grad is not None
            and config.ints.level < labels.INTLEVEL_QUADRUPOLE
        ):
            raise RuntimeError(
                "An electric field gradient requires quadrupole integrals. "
                "Construct the System with an integral level of at least "
                f"{labels.INTLEVEL_QUADRUPOLE}."
            )
        if h0_setup is None:
            raise NotImplementedError("Core Hamiltonian setup is missing.")
        if integral_setup is None:
            raise NotImplementedError(
                "Single-system integral setup is missing."
            )
        if config.ints.level < labels.INTLEVEL_HCORE:
            raise NotImplementedError(
                "Core Hamiltonian missing. Please increase the integral level to "
                "at least 2."
            )
        if config.scf.maxiter <= 0:
            raise SCFConvergenceError(
                "Implicit physical-root differentiation requires maxiter > 0."
            )
        strategy_label = " -> ".join(
            type(stage.method).__name__.removesuffix("Method").lower()
            for stage in config.scf.strategy.stages
        )
        route = SimpleNamespace(
            kind="functional",
            representation=config.scf._legacy_representation_code,
            strategy_label=strategy_label,
            strategy=config.scf.strategy,
            reason=None,
            warning=None,
        )
        return core, numbers, config, integral_setup, h0_setup, route, None

    core = system._active_exact_core()
    numbers = core.numbers if core is not None else system.numbers
    config = core.config if core is not None else system.config
    integral_setup = system.integral_setup
    h0_setup = core.h0_setup if core is not None else system.h0_setup

    if numbers.ndim != 1 or system.ihelp.batch_mode != 0:
        raise ValueError(
            "Converged-root conformer batching requires a single-System "
            "owner (numbers.ndim == 1 and batch_mode == 0)."
        )
    if positions.ndim != 3 or positions.shape[0] < 1:
        raise ValueError(
            "Converged-root conformer batching expects positions with shape "
            f"(B, nat, 3) and B >= 1; got {tuple(positions.shape)}."
        )
    if positions.shape[1:] != (numbers.shape[0], 3):
        raise ValueError(
            "Converged-root conformer batching expects positions with shape "
            f"(B, {numbers.shape[0]}, 3); got {tuple(positions.shape)}."
        )
    system_device = numbers.device
    if positions.device != system_device:
        raise DeviceError(
            f"Device mismatch: System is on '{system_device}', but positions "
            f"are on '{positions.device}'."
        )

    floating_setup = _floating_setup_tensor(system)
    expected_dtype = (
        core.effective_dtype
        if core is not None
        else (
            system.dd["dtype"]
            if floating_setup is None
            else floating_setup.dtype
        )
    )
    if floating_setup is not None:
        if floating_setup.device != system_device:
            raise DeviceError(
                "System floating setup is on "
                f"'{floating_setup.device}', but System numbers are on "
                f"'{system_device}'."
            )
        if core is not None and floating_setup.dtype != core.effective_dtype:
            raise DtypeError(
                "Dtype mismatch: System floating setup is of type "
                f"'{floating_setup.dtype}', but exact core declares "
                f"'{core.effective_dtype}'."
            )
    if positions.dtype != expected_dtype:
        raise DtypeError(
            "Dtype mismatch: System floating setup is of type "
            f"'{expected_dtype}', but positions are of type "
            f"'{positions.dtype}'."
        )

    if field is not None and config.ints.level < labels.INTLEVEL_DIPOLE:
        raise RuntimeError(
            "An electric field requires dipole integrals. Construct the "
            "System with an integral level of at least "
            f"{labels.INTLEVEL_DIPOLE}."
        )
    if (
        field_grad is not None
        and config.ints.level < labels.INTLEVEL_QUADRUPOLE
    ):
        raise RuntimeError(
            "An electric field gradient requires quadrupole integrals. "
            "Construct the System with an integral level of at least "
            f"{labels.INTLEVEL_QUADRUPOLE}."
        )
    if h0_setup is None:
        raise NotImplementedError("Core Hamiltonian setup is missing.")
    if integral_setup is None:
        raise NotImplementedError("Single-system integral setup is missing.")
    if config.ints.level < labels.INTLEVEL_HCORE:
        raise NotImplementedError(
            "Core Hamiltonian missing. Please increase the integral level to "
            "at least 2."
        )
    if config.scf.maxiter <= 0:
        raise SCFConvergenceError(
            "Implicit physical-root differentiation requires maxiter > 0."
        )

    route = resolve_full_scf_route(
        config.scf,
        system.interactions,
        system.ihelp,
        numbers,
        allow_implicit=True,
    )
    if route.kind != "functional":
        raise NotImplementedError(
            f"System is not eligible for functional full SCF: {route.reason}."
        )
    if route.warning is not None:
        OutputHandler.warn(route.warning)

    candidate_index = core.index if core is not None else None
    core_index = (
        candidate_index
        if candidate_index is not None
        and candidate_index.shells_to_atom.device
        == system.ihelp.shells_to_atom.device
        and candidate_index.orbitals_to_shell.device
        == system.ihelp.orbitals_to_shell.device
        else None
    )
    return core, numbers, config, integral_setup, h0_setup, route, core_index


def _build_lane_inputs(
    system: System | PaddedExactSystemCore,
    numbers: Tensor,
    config: Any,
    integral_setup: Any,
    h0_setup: Any,
    core_index: Any,
    effective_scf_config: Any,
    lane_positions: Tensor,
    lane_charge: Tensor,
    lane_spin: Tensor | None,
    lane_field: Tensor | None,
    lane_field_grad: Tensor | None,
):
    """Build call-local integrals, interactions, and SCFInputs for one lane."""
    if isinstance(system, PaddedExactSystemCore):
        masked_pos = torch.where(
            system.validity.atoms.unsqueeze(-1), lane_positions, 0.0
        )
        chrg_scalar = lane_charge.squeeze(0)
        spin_scalar = None if lane_spin is None else lane_spin.squeeze(0)
        padded_ints = evaluate_padded_integrals(system, masked_pos, chrg_scalar)
        inputs = setup_padded_scf_inputs(
            system,
            padded_ints,
            masked_pos,
            chrg_scalar,
            spin_scalar,
            lane_field,
            lane_field_grad,
        )
        return (
            inputs.integrals,
            inputs.n0,
            padded_ints.overlap_norm,
            inputs.n0,
            inputs.occupation,
            inputs.nel,
            None,
            None,
            inputs,
        )

    matrices, refocc, overlap_norm = build_integral_matrices(
        integral_setup,
        h0_setup,
        lane_positions,
        lane_charge,
    )
    call_components = list(system.interactions.components)
    if lane_field is not None:
        if any(
            component.label == LABEL_EFIELD for component in call_components
        ):
            raise ValueError(
                "Explicit field= cannot be combined with a custom interaction "
                f"using the '{LABEL_EFIELD}' label."
            )
        call_components.append(ElectricField(lane_field))
    if lane_field_grad is not None:
        if any(
            component.label == LABEL_EFIELD_GRAD
            for component in call_components
        ):
            raise ValueError(
                "Explicit field_grad= cannot be combined with a custom "
                f"interaction using the '{LABEL_EFIELD_GRAD}' label."
            )
        call_components.append(ElectricFieldGrad(lane_field_grad))
    call_interactions = InteractionList(
        *call_components,
        device=lane_positions.device,
        dtype=lane_positions.dtype,
    )
    interaction_data = _interaction_data(
        system, lane_positions, call_interactions
    )
    n0, occupation, nel = get_refocc(
        refocc, lane_charge, lane_spin, system.ihelp, validate_spin=False
    )
    inputs = setup_scf_inputs(
        numbers,
        matrices,
        n0,
        occupation,
        nel,
        call_interactions,
        interaction_data,
        system.ihelp,
        effective_scf_config,
        core_index=core_index,
        core_index_owner=(system if core_index is not None else None),
    )
    return (
        matrices,
        refocc,
        overlap_norm,
        n0,
        occupation,
        nel,
        call_interactions,
        interaction_data,
        inputs,
    )


def _raise_strict_lane_failure(
    lane_index: int,
    failure_code: int,
    failure_reason: str,
) -> None:
    """Raise the canonical eager exception identifying the failing lane."""
    prefix = f"[Lane {lane_index}] "
    if failure_code in (
        int(BatchRootFailureCode.INVALID_TEMPERATURE),
        int(BatchRootFailureCode.TOO_MANY_ELECTRONS),
    ):
        raise ValueError(prefix + failure_reason)
    if failure_code in (
        int(BatchRootFailureCode.FERMI_NONCONVERGENCE),
        int(BatchRootFailureCode.ELECTRON_COUNT),
    ):
        raise RuntimeError(prefix + failure_reason)
    raise SCFConvergenceError(prefix + failure_reason)


def _classify_fermi_code(code_value: int) -> tuple[int, str]:
    """Map an internal Fermi failure code to a BatchRootFailureCode and message."""
    if code_value == FERMI_FAILURE_INVALID_TEMPERATURE:
        return (
            int(BatchRootFailureCode.INVALID_TEMPERATURE),
            "Electronic Temperature must be non-negative.",
        )
    if code_value == FERMI_FAILURE_TOO_MANY_ELECTRONS:
        return (
            int(BatchRootFailureCode.TOO_MANY_ELECTRONS),
            "Number of electrons exceeds the number of orbitals.",
        )
    if code_value == FERMI_FAILURE_NONCONVERGENCE:
        return (
            int(BatchRootFailureCode.FERMI_NONCONVERGENCE),
            "Fermi energy failed to converge.",
        )
    if code_value == FERMI_FAILURE_ELECTRON_COUNT:
        return (
            int(BatchRootFailureCode.ELECTRON_COUNT),
            "Number of electrons changed during Fermi smearing.",
        )
    if code_value == FERMI_FAILURE_NONE:
        return int(BatchRootFailureCode.NONE), ""
    raise ValueError(f"Unknown nonzero Fermi failure code: {code_value}.")


class ResponseDiagnosticCollector:
    """Call-local collector for matrix-free BiCGSTAB response diagnostics."""

    def __init__(self) -> None:
        self.frozen: bool = False
        self.executed: Tensor | None = None
        self.status: Tensor | None = None
        self.converged: Tensor | None = None
        self.iterations: Tensor | None = None
        self.residual: Tensor | None = None
        self.target: Tensor | None = None
        self.incoming_finite: Tensor | None = None
        self.outgoing_finite: Tensor | None = None

    def solve(
        self,
        matvec: Callable[..., Tensor],
        rhs: Tensor,
        *params: Tensor,
        **kwargs: Any,
    ) -> LinearSolveResult:
        res = bicgstab(matvec, rhs, *params, **kwargs)
        if not self.frozen:
            rhs_d = _unwrap_primal_tensor(rhs)
            sol_d = _unwrap_primal_tensor(res.solution)
            stat_d = _unwrap_primal_tensor(res.status).to(dtype=torch.int64)
            iter_d = _unwrap_primal_tensor(res.iterations).to(dtype=torch.int64)
            res_d = _unwrap_primal_tensor(res.residual)
            tgt_d = _unwrap_primal_tensor(res.target)
            n_lanes = max(1, int(stat_d.numel()))
            self.executed = torch.ones_like(stat_d, dtype=torch.bool)
            self.status = stat_d
            self.converged = stat_d == int(STATUS_CONVERGED)
            self.iterations = iter_d
            self.residual = res_d
            self.target = tgt_d
            if self.incoming_finite is None:
                self.incoming_finite = (
                    torch.isfinite(rhs_d.reshape(n_lanes, -1))
                    .all(dim=-1)
                    .reshape(stat_d.shape)
                )
            if self.outgoing_finite is None:
                sol_fin = (
                    torch.isfinite(sol_d.reshape(n_lanes, -1))
                    .all(dim=-1)
                    .reshape(stat_d.shape)
                )
                self.outgoing_finite = self.converged & sol_fin
        conv_mask = res.status == int(STATUS_CONVERGED)
        while conv_mask.ndim < res.solution.ndim:
            conv_mask = conv_mask.unsqueeze(-1)
        masked_sol = torch.where(
            conv_mask,
            res.solution,
            torch.full_like(res.solution, float("nan")),
        )
        return LinearSolveResult(
            solution=masked_sol,
            status=res.status,
            iterations=res.iterations,
            residual=res.residual,
            target=res.target,
        )

    def record_incoming(self, grad_or_tangent: Tensor) -> None:
        if self.frozen:
            return
        detached = _unwrap_primal_tensor(grad_or_tangent)
        if detached.ndim >= 1:
            flat = detached.reshape(detached.shape[0], -1)
            finite = torch.isfinite(flat).all(dim=-1)
            active = (flat != 0.0).any(dim=-1)
        else:
            finite = torch.isfinite(detached).all()
            active = (detached != 0.0).any()
        self.incoming_finite = finite | ~active

    def record_outgoing(
        self, grads_or_tangents: tuple[Tensor | None, ...]
    ) -> None:
        if self.frozen:
            return
        first = next((g for g in grads_or_tangents if g is not None), None)
        if first is None:
            return
        first_d = _unwrap_primal_tensor(first)
        batch_shape = first_d.shape[:1] if first_d.ndim >= 1 else ()
        finite = torch.ones(
            batch_shape, dtype=torch.bool, device=first_d.device
        )
        for g in grads_or_tangents:
            if g is None:
                continue
            gd = _unwrap_primal_tensor(g)
            if gd.ndim >= 1 and batch_shape:
                finite = finite & torch.isfinite(
                    gd.reshape(gd.shape[0], -1)
                ).all(dim=-1)
            else:
                finite = finite & torch.isfinite(gd).all()
        self.outgoing_finite = finite

    def finalize(
        self,
        valid_indices: list[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchResponseDiagnostics:
        executed = torch.zeros(batch_size, dtype=torch.bool, device=device)
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        status = torch.full(
            (batch_size,), RESPONSE_NOT_RUN, dtype=torch.int64, device=device
        )
        iterations = torch.zeros(batch_size, dtype=torch.int64, device=device)
        residual = torch.full(
            (batch_size,), float("nan"), dtype=dtype, device=device
        )
        target = torch.full(
            (batch_size,), float("nan"), dtype=dtype, device=device
        )
        incoming_finite = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
        outgoing_finite = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
        if not valid_indices:
            return BatchResponseDiagnostics(
                executed=executed,
                converged=converged,
                status=status,
                iterations=iterations,
                residual=residual,
                target=target,
                incoming_finite=incoming_finite,
                outgoing_finite=outgoing_finite,
            )
        idx = torch.tensor(valid_indices, dtype=torch.long, device=device)
        n_valid = len(valid_indices)

        def _expand_bool(val: Tensor | None, default: bool) -> Tensor:
            if val is None:
                return torch.full(
                    (n_valid,), default, dtype=torch.bool, device=device
                )
            return (
                val.reshape(-1)
                .expand(n_valid)
                .to(device=device, dtype=torch.bool)
            )

        def _expand_int(val: Tensor | None, default: int) -> Tensor:
            if val is None:
                return torch.full(
                    (n_valid,), default, dtype=torch.int64, device=device
                )
            return (
                val.reshape(-1)
                .expand(n_valid)
                .to(device=device, dtype=torch.int64)
            )

        def _expand_float(val: Tensor | None) -> Tensor:
            if val is None:
                return torch.full(
                    (n_valid,), float("nan"), dtype=dtype, device=device
                )
            return (
                val.reshape(-1).expand(n_valid).to(device=device, dtype=dtype)
            )

        inc_v = _expand_bool(self.incoming_finite, True)
        out_v = _expand_bool(self.outgoing_finite, True)
        has_monitor = (
            self.incoming_finite is not None or self.outgoing_finite is not None
        )
        exec_v = _expand_bool(self.executed, has_monitor)
        conv_v = _expand_bool(self.converged, has_monitor) & inc_v & out_v
        default_stat = (
            int(STATUS_CONVERGED) if has_monitor else RESPONSE_NOT_RUN
        )
        stat_v = _expand_int(self.status, default_stat)
        if has_monitor and self.status is None:
            stat_v = torch.where(
                out_v & inc_v,
                stat_v,
                torch.full_like(stat_v, int(STATUS_NONFINITE)),
            )
        elif self.status is not None:
            stat_v = torch.where(
                (stat_v == int(STATUS_CONVERGED)) & (~inc_v | ~out_v),
                torch.full_like(stat_v, int(STATUS_NONFINITE)),
                stat_v,
            )
        iter_v = _expand_int(self.iterations, 0)
        res_v = _expand_float(self.residual)
        tgt_v = _expand_float(self.target)

        executed[idx] = exec_v
        converged[idx] = conv_v
        status[idx] = stat_v
        iterations[idx] = iter_v
        residual[idx] = res_v
        target[idx] = tgt_v
        incoming_finite[idx] = inc_v
        outgoing_finite[idx] = out_v
        return BatchResponseDiagnostics(
            executed=executed,
            converged=converged,
            status=status,
            iterations=iterations,
            residual=residual,
            target=target,
            incoming_finite=incoming_finite,
            outgoing_finite=outgoing_finite,
        )


class _ResponseParamsMonitor(torch.autograd.Function):
    """Record finiteness of parameter sensitivities leaving implicit_root."""

    @staticmethod
    def forward(
        ctx: Any,
        collector: ResponseDiagnosticCollector,
        *params: Tensor,
    ) -> tuple[Tensor, ...]:
        ctx.collector = collector
        return tuple(p.clone() for p in params)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple[None, ...]:
        ctx.collector.record_outgoing(grad_outputs)
        return (None, *grad_outputs)


class _ResponseOutputMonitor(torch.autograd.Function):
    """Record finiteness of state sensitivities entering implicit_root."""

    @staticmethod
    def forward(
        ctx: Any,
        collector: ResponseDiagnosticCollector,
        attached_roots: Tensor,
    ) -> Tensor:
        ctx.collector = collector
        return attached_roots.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[None, Tensor]:
        ctx.collector.record_incoming(grad_output)
        return None, grad_output


def classify_response_failure_status(
    status: SCFStatus,
    converged: Tensor,
    *lane_gradients: Tensor | None,
    response_diagnostics: (
        BatchResponseDiagnostics | ResponseDiagnosticCollector | None
    ) = None,
    active_mask: Tensor | None = None,
) -> SCFStatus:
    """Return a post-differentiation SCFStatus classifying nonfinite lane derivatives.

    Because :class:`Result` is an immutable primal snapshot constructed before
    backward or forward AD executes, ``Result.status`` is never mutated in
    place during differentiation. Callers that inspect per-lane derivative
    tensors ``(B, ...)`` can use this helper to obtain a new :class:`SCFStatus`.
    When ``response_diagnostics`` is supplied, confirmed linear-response solver
    failures are marked with :attr:`BatchRootFailureCode.RESPONSE_FAILURE` and
    other nonfinite derivatives on primal-converged lanes are marked with
    :attr:`BatchRootFailureCode.NONFINITE_DERIVATIVE`.
    """
    batch_size = int(converged.shape[0])
    device = converged.device
    grad_finite = torch.ones(batch_size, dtype=torch.bool, device=device)
    for grad_tensor in lane_gradients:
        if grad_tensor is None:
            continue
        if grad_tensor.ndim >= 1 and grad_tensor.shape[0] == batch_size:
            grad_finite = grad_finite & torch.isfinite(
                grad_tensor.reshape(batch_size, -1)
            ).all(dim=-1)
        elif batch_size == 1:
            grad_finite = grad_finite & torch.isfinite(
                grad_tensor
            ).all().reshape(1)
        else:
            raise ValueError(
                f"Per-lane gradient tensor must have leading dimension "
                f"{batch_size}; got shape {tuple(grad_tensor.shape)}."
            )

    primal_ok = converged.to(dtype=torch.bool) & (
        status.final_failure == int(BatchRootFailureCode.NONE)
    )
    if active_mask is not None:
        primal_ok = primal_ok & active_mask.to(
            device=device, dtype=torch.bool
        ).reshape(batch_size)

    if isinstance(response_diagnostics, ResponseDiagnosticCollector):
        valid_indices = [
            i
            for i in range(batch_size)
            if bool(converged.reshape(-1)[i].item())
            and int(status.final_failure.reshape(-1)[i].item())
            == int(BatchRootFailureCode.NONE)
        ]
        diag: BatchResponseDiagnostics | None = response_diagnostics.finalize(
            valid_indices, batch_size, device, torch.float64
        )
    else:
        diag = response_diagnostics

    if diag is not None:
        confirmed_resp_fail = primal_ok & diag.confirmed_failure.to(
            device=device
        )
        unclassified_nonfinite = primal_ok & ~confirmed_resp_fail & ~grad_finite
        updated_final = torch.where(
            confirmed_resp_fail,
            torch.full_like(
                status.final_failure, int(BatchRootFailureCode.RESPONSE_FAILURE)
            ),
            status.final_failure,
        )
        updated_final = torch.where(
            unclassified_nonfinite,
            torch.full_like(
                updated_final, int(BatchRootFailureCode.NONFINITE_DERIVATIVE)
            ),
            updated_final,
        )
    else:
        response_failed = primal_ok & ~grad_finite
        updated_final = torch.where(
            response_failed,
            torch.full_like(
                status.final_failure, int(BatchRootFailureCode.RESPONSE_FAILURE)
            ),
            status.final_failure,
        )
    return SCFStatus(
        trajectory_failure=status.trajectory_failure.clone(),
        final_failure=updated_final,
    )


def _make_representation_projections(
    primal_inputs: SCFInputs,
    representation: int,
    guess_charges: SCFCharges,
    lane_padded: bool,
) -> tuple[
    Tensor,
    PotentialLayout | None,
    Callable[[Tensor], Tensor] | None,
    Callable[[Tensor], Tensor] | None,
    Callable[[Tensor], Any],
]:
    """Construct initial state, layout, padding projections, and raw map."""
    state_projection: Callable[[Tensor], Tensor] | None = None
    residual_projection: Callable[[Tensor], Tensor] | None = None
    if representation == labels.SCP_MODE_CHARGE:
        initial_state = pack_charges(guess_charges, primal_inputs.charge_layout)
        layout = None
        if lane_padded:
            state_projection, residual_projection = _padded_charge_projections(
                primal_inputs
            )
        map_fn = lambda x, _inp=primal_inputs: scf_map(x, _inp)
    elif representation == labels.SCP_MODE_POTENTIAL:
        initial_state, layout = potential_initial_guess(
            guess_charges, primal_inputs
        )
        base_proj = lambda val, _lay=layout: _project_potential_padding(
            val, _lay
        )
        if lane_padded:
            pad_state, residual_projection = _padded_potential_projections(
                primal_inputs, layout
            )
            state_projection = lambda val, _ps=pad_state, _bp=base_proj: _ps(
                _bp(val)
            )

            def _potential_map_padded(
                x: Tensor,
                _inp: SCFInputs = primal_inputs,
                _lay: PotentialLayout = layout,
                _sp: Callable[[Tensor], Tensor] = state_projection,
            ):
                res = potential_map(x, _inp, _lay)
                return replace(res, mapped_x=_sp(res.mapped_x))

            map_fn = _potential_map_padded
        else:
            state_projection = base_proj
            map_fn = lambda x, _inp=primal_inputs, _lay=layout: potential_map(
                x, _inp, _lay
            )
    elif representation == labels.SCP_MODE_FOCK:
        initial_state = fock_initial_guess(guess_charges, primal_inputs)
        layout = None
        if lane_padded:
            matrix_mask = primal_inputs.orbital_mask.unsqueeze(
                -1
            ) & primal_inputs.orbital_mask.unsqueeze(-2)
            state_projection = lambda val, _mm=matrix_mask, _hc=primal_inputs.integrals.hcore: torch.where(
                _mm, val, _hc
            )
            residual_projection = lambda val, _mm=matrix_mask: torch.where(
                _mm, val, 0.0
            )
        map_fn = lambda x, _inp=primal_inputs: fock_map(x, _inp)
    else:
        raise ValueError(
            f"Unknown convergence target (SCP mode) '{representation}'."
        )

    if lane_padded and state_projection is not None:
        initial_state = state_projection(initial_state)

    return initial_state, layout, state_projection, residual_projection, map_fn


@torch._dynamo.disable
def _validate_prepared_primal(
    prepared: PreparedConvergedRootBatch,
    positions: Tensor,
    charges: tuple[Tensor, ...],
    spins: tuple[Tensor | None, ...],
    fields: tuple[Tensor | None, ...],
    field_grads: tuple[Tensor | None, ...],
    numbers: Tensor,
    config: object,
) -> None:
    """Require exact primal-input identity and untouched versions before reusing roots."""

    def same(
        actual: Tensor | None,
        expected: _TensorValueSnapshot | None,
        expected_version: int | None = None,
        ref: Tensor | None = None,
    ) -> bool:
        if expected is None:
            return actual is None
        if actual is None:
            return False
        if expected_version is not None and ref is not None:
            if _tensor_version(ref) != expected_version:
                return False
        if expected_version is not None and actual is ref:
            if _tensor_version(actual) != expected_version:
                return False
        return _matches_snapshot(actual, expected)

    system = prepared.system
    if isinstance(system, System):
        sys_cfg_mismatch = (
            id(system.config) != prepared.config_identity
            or repr(system.config) != prepared.config_repr
        )
        sys_num_mismatch = not same(
            system.numbers,
            prepared.numbers_snapshot,
            prepared.numbers_version,
            system.numbers,
        )
    else:
        sys_cfg_mismatch = False
        sys_num_mismatch = False

    if (
        sys_cfg_mismatch
        or id(config) != prepared.config_identity
        or repr(config) != prepared.config_repr
    ):
        raise RuntimeError(
            "Prepared physical roots cannot be reused after the System "
            "configuration has changed; prepare roots again."
        )
    if sys_num_mismatch or not same(
        numbers,
        prepared.numbers_snapshot,
        prepared.numbers_version,
        numbers,
    ):
        raise RuntimeError(
            "Prepared physical roots cannot be reused after atomic numbers "
            "have changed; prepare roots again."
        )
    if not same(
        positions,
        prepared.positions_snapshot,
        (
            prepared.positions_version
            if positions is prepared.positions
            else None
        ),
        positions if positions is prepared.positions else None,
    ):
        raise RuntimeError(
            "Prepared physical roots are stale for the supplied positions; "
            "prepare roots again."
        )
    if len(prepared.model_parameter_refs) != len(
        prepared.model_parameter_snapshot
    ) or any(
        not same(value, snapshot, ver, value)
        for value, snapshot, ver in zip(
            prepared.model_parameter_refs,
            prepared.model_parameter_snapshot,
            prepared.model_parameter_versions,
        )
    ):
        raise RuntimeError(
            "Prepared physical roots are stale for the current Model "
            "parameters; prepare roots again."
        )
    if len(prepared.system_tensor_refs) != len(
        prepared.system_tensor_snapshot
    ) or any(
        not same(value, snapshot, ver, value)
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
    if any(
        not same(lane.root_x, root_snapshot)
        for lane, root_snapshot in zip(prepared.lanes, prepared.root_snapshots)
    ):
        raise RuntimeError(
            "A prepared physical root tensor was modified; prepare roots again."
        )
    if any(
        lane.response_root_x is not None
        and snapshot is not None
        and not same(lane.response_root_x, snapshot)
        for lane, snapshot in zip(
            prepared.lanes, prepared.response_root_snapshots
        )
    ):
        raise RuntimeError(
            "A prepared response root tensor was modified; prepare roots again."
        )

    for lane, expected_versions in zip(
        prepared.lanes, prepared.input_tensor_versions
    ):
        actual_versions = {
            path: int(_unwrap_primal_tensor(value)._version)
            for path, value in tensor_leaves(lane.inputs)
        }
        if set(actual_versions) != set(expected_versions):
            raise RuntimeError(
                f"Prepared lane {lane.lane_index} SCF input tensor structure "
                "changed after root preparation."
            )
        for path, expected_version in expected_versions.items():
            if actual_versions[path] != expected_version:
                raise RuntimeError(
                    f"Prepared lane {lane.lane_index} SCF input tensor was "
                    f"mutated in place at '{path}'."
                )

    for lane, param_snapshots in zip(
        prepared.lanes, prepared.parameter_snapshots
    ):
        if len(lane.parameters) != len(param_snapshots) or any(
            not same(param, snapshot)
            for param, snapshot in zip(lane.parameters, param_snapshots)
        ):
            raise RuntimeError(
                f"Prepared physical parameters for lane {lane.lane_index} "
                "were modified; prepare roots again."
            )

    if prepared.response_representation != prepared.primal_representation:
        for lane, param_snapshots in zip(
            prepared.lanes, prepared.parameter_snapshots
        ):
            if not lane.root_valid:
                continue
            assert lane.response_adapter is not None
            assert lane.response_parameters is not None
            if len(lane.response_parameters) != len(param_snapshots) or any(
                not same(param, snapshot)
                for param, snapshot in zip(
                    lane.response_parameters, param_snapshots
                )
            ):
                raise RuntimeError(
                    f"Prepared response parameters for lane {lane.lane_index} "
                    "were modified; prepare roots again."
                )

    current_inputs = tuple(
        zip(positions.unbind(0), charges, spins, fields, field_grads)
    )
    for lane_index, (lane, values, snapshots) in enumerate(
        zip(
            prepared.lanes,
            current_inputs,
            prepared.lane_input_snapshots,
        )
    ):
        if any(
            not same(value, snapshot)
            for value, snapshot in zip(values, snapshots)
        ):
            raise RuntimeError(
                f"Prepared physical root for lane {lane_index} is stale for "
                "charge, spin, or external fields; prepare roots again."
            )


@dataclass(frozen=True, eq=False)
class _LanePrimalSolveOutcome:
    """Detached primal fixed-point solve and validation result for one lane."""

    root_x: Tensor
    map_result: Any
    iterations: Tensor
    lane_converged: Tensor
    lane_residual: Tensor
    raw_residual_l2: Tensor
    raw_residual_max: Tensor
    electron_count_error: Tensor
    density_finite: bool
    padding_valid: bool
    trajectory_status: SCFTrajectoryStatus | None
    fermi_status: FermiStatus | None
    damping_state: DampingState
    status: SCFStatus
    failure_code: int
    failure_reason: str | None
    root_valid: bool


def _solve_and_validate_lane_primal(
    primal_inputs: SCFInputs,
    guess_charges: SCFCharges,
    numbers: Tensor,
    representation: int,
    settings: Any,
    strategy: SCFStrategy,
    lane_padded: bool,
    strategy_label: str,
    maxiter: int,
    damp_default: float,
    damp_dynamic_factor: float,
    eager_primal_fermi: bool | str,
) -> tuple[
    PotentialLayout | None,
    _SCFMapTensorTree,
    tuple[Tensor, ...],
    _LanePrimalSolveOutcome,
]:
    """Solve one lane's detached fixed-point problem and validate with static Fermi."""
    with torch.no_grad():
        (
            initial_state,
            layout,
            state_projection,
            residual_projection,
            map_fn,
        ) = _make_representation_projections(
            primal_inputs, representation, guess_charges, lane_padded
        )
        solver_problem = None
        if (
            getattr(strategy.stages[0].method, "required_representation", None)
            is SCFRepresentation.FOCK
        ):
            solver_problem = SCFProblem(
                primal_inputs,
                SCFRepresentation.FOCK,
            )

        with eager_primal_fermi_scope(eager_primal_fermi):
            outcome = _solve_map(
                initial_state,
                settings,
                map_fn,
                state_projection=state_projection,
                problem=solver_problem,
                strategy=strategy,
                eager=True,
            )

        with eager_primal_fermi_scope(False):
            adapter, primal_params = _SCFMapTensorTree.prepare(
                primal_inputs, representation, layout
            )
            primal_params = tuple(p.clone() for p in primal_params)
            independent_map = adapter.evaluate(outcome.root_x, *primal_params)
            independent_mapped_x = (
                state_projection(independent_map.mapped_x)
                if (
                    representation == labels.SCP_MODE_POTENTIAL
                    and lane_padded
                    and state_projection is not None
                )
                else independent_map.mapped_x
            )
            certified_map_result = replace(
                independent_map, mapped_x=independent_mapped_x
            )

            if representation == labels.SCP_MODE_CHARGE:
                final_spectral = certified_map_result.spectral
            elif representation == labels.SCP_MODE_POTENTIAL:
                assert layout is not None
                final_state = unpack_potential(independent_mapped_x, layout)
                final_spectral = evaluate_potential(final_state, primal_inputs)
            else:
                final_spectral = evaluate_hamiltonian(
                    independent_mapped_x, primal_inputs
                )

        final_fermi_status = final_spectral.fermi_status
        raw_diff = independent_mapped_x - outcome.root_x
        if residual_projection is not None:
            raw_diff = residual_projection(raw_diff)
        raw_residual_l2 = torch.linalg.vector_norm(raw_diff)
        raw_residual_max = raw_diff.abs().amax()

        active_orb = primal_inputs.orbital_mask
        active_count = active_orb.sum(dim=-1, keepdim=True)
        spectral_mask = (
            torch.arange(
                final_spectral.occupation.shape[-1],
                device=active_orb.device,
            )
            < active_count
        )
        occ_masked = torch.where(
            spectral_mask.unsqueeze(-2), final_spectral.occupation, 0.0
        )
        mo_electron_error = (
            (occ_masked.sum(dim=-1) - primal_inputs.nel).abs().amax()
        )
        ao_populations = torch.where(
            active_orb,
            primal_inputs.n0 - final_spectral.charges.mono,
            0.0,
        )
        ao_electron_error = (
            (ao_populations.sum(dim=-1) - primal_inputs.nel.sum(dim=-1))
            .abs()
            .amax()
        )
        electron_count_error = torch.maximum(
            mo_electron_error, ao_electron_error
        )

        root_finite = bool(torch.isfinite(outcome.root_x).all())
        residual_finite = bool(torch.isfinite(raw_diff).all())
        density_finite = bool(torch.isfinite(final_spectral.density).all())

        has_real_atoms = bool((numbers != 0).any())
        has_active_orbitals = bool(primal_inputs.orbital_mask.any())
        padding_valid = has_real_atoms and has_active_orbitals
        if padding_valid and lane_padded and state_projection is not None:
            projected_root = state_projection(outcome.root_x)
            projected_mapped = state_projection(independent_mapped_x)
            ao_matrix_mask = active_orb.unsqueeze(-1) & active_orb.unsqueeze(-2)
            inactive_density_max = (
                final_spectral.density.masked_fill(ao_matrix_mask, 0.0)
                .abs()
                .amax()
            )
            padding_valid = bool(
                torch.isfinite(projected_root).all()
                and torch.isfinite(projected_mapped).all()
                and torch.allclose(
                    projected_root, outcome.root_x, atol=1.0e-12, rtol=0.0
                )
                and torch.allclose(
                    projected_mapped,
                    independent_mapped_x,
                    atol=1.0e-12,
                    rtol=0.0,
                )
                and bool(inactive_density_max <= 1.0e-12)
            )

        traj_code = (
            0
            if outcome.trajectory_status is None
            else int(outcome.trajectory_status.first_failure.item())
        )
        final_code = int(
            fermi_failure_code(
                final_fermi_status, like=outcome.converged
            ).item()
        )

        failure_code = int(BatchRootFailureCode.NONE)
        failure_reason: str | None = None

        if not padding_valid:
            failure_code = int(BatchRootFailureCode.INVALID_PADDING)
            failure_reason = (
                "Invalid padding state or empty active orbital/atom mask."
            )
        elif traj_code != FERMI_FAILURE_NONE:
            failure_code, failure_reason = _classify_fermi_code(traj_code)
        elif final_code != FERMI_FAILURE_NONE:
            failure_code, failure_reason = _classify_fermi_code(final_code)
        elif not (root_finite and residual_finite and density_finite):
            failure_code = int(BatchRootFailureCode.NONFINITE_STATE)
            failure_reason = (
                "Primal root state, residual, or density is nonfinite."
            )
        elif bool(electron_count_error > _ELECTRON_COUNT_TOL):
            failure_code = int(BatchRootFailureCode.ELECTRON_COUNT)
            failure_reason = (
                f"Electron count error {float(electron_count_error):.3e} "
                f"exceeds tolerance {_ELECTRON_COUNT_TOL:.1e}."
            )
        elif not bool(outcome.converged):
            failure_code = int(BatchRootFailureCode.MAXITER_EXCEEDED)
            damp = (
                damp_dynamic_factor
                if bool(outcome.damping_state.switched)
                else damp_default
            )
            failure_reason = (
                f"SCF did not certify after {maxiter} iterations using "
                f"strategy '{strategy_label}' (damping {damp}; "
                f"residual L2={float(raw_residual_l2):.3e}, "
                f"max={float(raw_residual_max):.3e})."
            )
        elif not bool(
            (raw_residual_l2 <= settings.x_atol)
            & (raw_residual_max <= settings.x_atol_max)
        ):
            failure_code = int(BatchRootFailureCode.RESIDUAL_TOLERANCE)
            failure_reason = (
                f"Independent raw map residual (L2={float(raw_residual_l2):.3e}, "
                f"max={float(raw_residual_max):.3e}) exceeds tolerance "
                f"(x_atol={settings.x_atol:.1e}, "
                f"x_atol_max={settings.x_atol_max:.1e})."
            )

        root_valid = failure_code == int(BatchRootFailureCode.NONE)
        base_status = public_scf_status(
            outcome.trajectory_status,
            final_fermi_status,
            like=outcome.converged,
        )
        if not root_valid and int(base_status.final_failure.item()) == 0:
            status = SCFStatus(
                trajectory_failure=base_status.trajectory_failure,
                final_failure=torch.tensor(
                    failure_code,
                    dtype=torch.int64,
                    device=numbers.device,
                ),
            )
        else:
            status = base_status

        lane_converged = (
            outcome.converged
            if root_valid
            else torch.zeros_like(outcome.converged)
        )
        lane_residual = torch.stack((raw_residual_l2, raw_residual_max))

        return (
            layout,
            adapter,
            primal_params,
            _LanePrimalSolveOutcome(
                root_x=outcome.root_x.detach(),
                map_result=certified_map_result,
                iterations=outcome.iterations.detach(),
                lane_converged=lane_converged.detach(),
                lane_residual=lane_residual.detach(),
                raw_residual_l2=raw_residual_l2.detach(),
                raw_residual_max=raw_residual_max.detach(),
                electron_count_error=electron_count_error.detach(),
                density_finite=density_finite,
                padding_valid=padding_valid,
                trajectory_status=outcome.trajectory_status,
                fermi_status=final_fermi_status,
                damping_state=outcome.damping_state,
                status=status,
                failure_code=failure_code,
                failure_reason=failure_reason,
                root_valid=root_valid,
            ),
        )


def _scf_representation_from_label(representation: int) -> SCFRepresentation:
    """Translate the controller representation label to its public enum."""
    return {
        labels.SCP_MODE_CHARGE: SCFRepresentation.CHARGE,
        labels.SCP_MODE_POTENTIAL: SCFRepresentation.POTENTIAL,
        labels.SCP_MODE_FOCK: SCFRepresentation.FOCK,
    }[representation]


def _prepare_lane_response_root(
    inputs: SCFInputs,
    primal_representation: int,
    primal_layout: PotentialLayout | None,
    solve: _LanePrimalSolveOutcome,
    settings: Any,
    response_settings: SCFResponseSettings,
    *,
    padded: bool,
):
    """Use shared S11 response conversion/certification for one valid lane."""
    source_representation = _scf_representation_from_label(
        primal_representation
    )
    problem = SCFProblem(
        inputs,
        source_representation,
        potential_layout=primal_layout,
    )
    thresholds = resolve_certification_thresholds(
        problem,
        x_atol=settings.x_atol,
        x_atol_max=settings.x_atol_max,
        dtype=solve.root_x.dtype,
        requested_x_atol=settings.requested_x_atol,
        requested_x_atol_max=settings.requested_x_atol_max,
    )
    certification = certify_scf_core_state(
        problem,
        SCFPhysicalState(solve.root_x, solve.map_result),
        thresholds,
    )
    primal_outcome = SimpleNamespace(
        final_problem=problem,
        root_x=solve.root_x,
        map_result=solve.map_result,
        converged=certification.certified,
        certification=certification,
        iterations=solve.iterations,
        residual=solve.lane_residual,
        trajectory_status=solve.trajectory_status,
        damping_state=solve.damping_state,
    )
    target = (
        source_representation
        if response_settings.representation is None
        else response_settings.representation
    )
    if padded and target is not source_representation:
        raise ValueError(
            "Padded conformer batches currently require response coordinates "
            "to match the primal representation."
        )
    with eager_primal_fermi_scope(False):
        return prepare_response_root(
            primal_outcome, settings, response_settings
        )


def prepare_converged_root_batch(
    system: System | PaddedExactSystemCore,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    strict: bool | None = None,
    eager_primal_fermi: bool | str = False,
) -> PreparedConvergedRootBatch:
    """Prepare and validate independent physical SCF roots for a conformer batch.

    Parameters
    ----------
    system : System | PaddedExactSystemCore
        Single-System owner providing shared atomic numbers, basis/index
        structure, and Model parameters.
    positions : Tensor
        Conformer geometries of shape ``(B, nat, 3)``.
    chrg : Tensor | float | int, default 0
        Shared scalar or per-lane total charge.
    spin : Tensor | float | int | None, default None
        Shared scalar or per-lane unpaired electron count.
    field : Tensor | None, default None
        Shared ``(3,)`` or per-lane ``(B, 3)`` electric field.
    field_grad : Tensor | None, default None
        Shared ``(3, 3,)`` or per-lane ``(B, 3, 3)`` electric field gradient.
    strict : bool | None, default None
        ``force_convergence=True`` always raises on a failed lane. Otherwise,
        ``True`` enables strict batch validation and ``False`` or ``None``
        records primal failure metadata so valid lanes can be evaluated.
    eager_primal_fermi : bool | str, default False
        Primal Fermi solver mode used exclusively during detached physical root
        solving (``False``/``"static"``, ``True``/``"adaptive"``, or
        ``"spectral"``) while keeping post-preparation residual validation and
        differentiable evaluation on the fixed-depth Fermi formulation.
    """
    (
        _core,
        numbers,
        config,
        integral_setup,
        h0_setup,
        route,
        core_index,
    ) = _validate_system_and_positions(system, positions, field, field_grad)
    batch_size = positions.shape[0]
    effective_scf_config = config.scf
    strict_mode = _resolve_strictness(effective_scf_config, strict)

    charges = _normalize_batch_scalar("charge", chrg, positions, batch_size)
    spins = (
        tuple(None for _ in range(batch_size))
        if spin is None
        else _normalize_batch_scalar("spin", spin, positions, batch_size)
    )
    fields = _normalize_batch_perturbation(
        "field", field, positions, batch_size, (3,)
    )
    field_grads = _normalize_batch_perturbation(
        "field_grad", field_grad, positions, batch_size, (3, 3)
    )

    representation = route.representation
    response_settings = effective_scf_config.response
    primal_response_representation = _scf_representation_from_label(
        representation
    )
    requested_response_representation = (
        primal_response_representation
        if response_settings.representation is None
        else response_settings.representation
    )
    if (
        isinstance(system, PaddedExactSystemCore)
        and requested_response_representation
        is not primal_response_representation
    ):
        raise ValueError(
            "Padded conformer batches currently require response coordinates "
            "to match the primal representation."
        )
    response_representation = {
        SCFRepresentation.CHARGE: labels.SCP_MODE_CHARGE,
        SCFRepresentation.POTENTIAL: labels.SCP_MODE_POTENTIAL,
        SCFRepresentation.FOCK: labels.SCP_MODE_FOCK,
    }[requested_response_representation]
    settings, strategy = _prepare_solver(effective_scf_config, positions.dtype)
    has_padding = bool((numbers == 0).any())

    external_input_ids = {
        id(value)
        for value in (
            positions,
            numbers,
            *((system.numbers,) if isinstance(system, System) else ()),
            *charges,
            *(value for value in spins if value is not None),
            *(value for value in fields if value is not None),
            *(value for value in field_grads if value is not None),
        )
    }
    system_tensor_refs, model_parameter_refs = (
        _collect_system_and_model_tensors(system, external_input_ids)
    )

    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    eager_no_grad_build = (
        not is_compiling() and ft.get_interpreter_stack() is None
    )

    lanes: list[ConvergedRootLaneRecord] = []
    for lane_index in range(batch_size):
        lane_pos = positions[lane_index]
        lane_chrg = charges[lane_index]
        lane_spin = spins[lane_index]
        lane_field = fields[lane_index]
        lane_fgrad = field_grads[lane_index]

        primal_pos = _unwrap_primal_tensor(lane_pos)
        primal_chrg = _unwrap_primal_tensor(lane_chrg)
        primal_spin = (
            None if lane_spin is None else _unwrap_primal_tensor(lane_spin)
        )
        primal_field = (
            None if lane_field is None else _unwrap_primal_tensor(lane_field)
        )
        primal_fgrad = (
            None if lane_fgrad is None else _unwrap_primal_tensor(lane_fgrad)
        )

        build_ctx = torch.no_grad() if eager_no_grad_build else nullcontext()
        with build_ctx:
            (
                matrices,
                refocc,
                overlap_norm,
                n0,
                occupation,
                nel,
                call_interactions,
                interaction_data,
                raw_inputs,
            ) = _build_lane_inputs(
                system,
                numbers,
                config,
                integral_setup,
                h0_setup,
                core_index,
                effective_scf_config,
                primal_pos if eager_no_grad_build else lane_pos,
                primal_chrg if eager_no_grad_build else lane_chrg,
                primal_spin if eager_no_grad_build else lane_spin,
                primal_field if eager_no_grad_build else lane_field,
                primal_fgrad if eager_no_grad_build else lane_fgrad,
            )

        primal_inputs = _detach_scf_inputs(raw_inputs)
        primal_refocc = _unwrap_primal_tensor(refocc)
        primal_overlap_norm = _unwrap_primal_tensor(overlap_norm)
        lane_padded = has_padding or bool((~primal_inputs.orbital_mask).any())

        with torch.no_grad():
            if isinstance(system, PaddedExactSystemCore):
                primal_masked_pos = torch.where(
                    system.validity.atoms.unsqueeze(-1), primal_pos, 0.0
                )
                guess = _padded_initial_guess(
                    system,
                    primal_masked_pos,
                    primal_chrg.squeeze(0),
                    primal_inputs,
                )
            else:
                guess = get_guess(
                    numbers,
                    primal_pos,
                    primal_chrg,
                    system.ihelp,
                    effective_scf_config.guess,
                    validate=False,
                )
            guess_charges = SCFCharges(
                mono=guess,
                dipole=(
                    None
                    if primal_inputs.charge_layout.dipole_shape is None
                    else guess.new_zeros(
                        primal_inputs.charge_layout.dipole_shape
                    )
                ),
                quad=(
                    None
                    if primal_inputs.charge_layout.quadrupole_shape is None
                    else guess.new_zeros(
                        primal_inputs.charge_layout.quadrupole_shape
                    )
                ),
            )

        layout, adapter, primal_params, solve_outcome = (
            _solve_and_validate_lane_primal(
                primal_inputs,
                guess_charges,
                numbers,
                representation,
                settings,
                strategy,
                lane_padded,
                route.strategy_label,
                effective_scf_config.maxiter,
                effective_scf_config.damp,
                effective_scf_config.damp_dynamic_factor,
                eager_primal_fermi,
            )
        )

        same_representation = (
            requested_response_representation is primal_response_representation
        )
        response_root = None
        if solve_outcome.root_valid and not same_representation:
            response_root = _prepare_lane_response_root(
                primal_inputs,
                representation,
                layout,
                solve_outcome,
                settings,
                response_settings,
                padded=isinstance(system, PaddedExactSystemCore),
            )

        if not solve_outcome.root_valid:
            assert solve_outcome.failure_reason is not None
            if strict_mode:
                _raise_strict_lane_failure(
                    lane_index,
                    solve_outcome.failure_code,
                    solve_outcome.failure_reason,
                )
            msg = f"[Lane {lane_index}] {solve_outcome.failure_reason}"
            warnings.warn(msg, SCFConvergenceWarning, stacklevel=2)
            OutputHandler.warn(msg, SCFConvergenceWarning)

        primal_root_x = solve_outcome.root_x.detach()
        response_root_x = (
            None
            if not solve_outcome.root_valid
            else (
                primal_root_x if same_representation else response_root.root_x
            )
        )

        lanes.append(
            ConvergedRootLaneRecord(
                lane_index=lane_index,
                positions=lane_pos,
                charge=lane_chrg,
                spin=lane_spin,
                field=lane_field,
                field_grad=lane_fgrad,
                integrals=primal_inputs.integrals,
                refocc=primal_refocc,
                overlap_norm=primal_overlap_norm,
                n0=primal_inputs.n0,
                occupation=primal_inputs.occupation,
                nel=primal_inputs.nel,
                interactions=None,
                interaction_data=None,
                inputs=primal_inputs,
                potential_layout=layout,
                adapter=adapter,
                parameters=primal_params,
                root_x=primal_root_x,
                response_adapter=(
                    None
                    if not solve_outcome.root_valid
                    else (
                        adapter
                        if same_representation
                        else response_root.adapter
                    )
                ),
                response_parameters=(
                    None
                    if not solve_outcome.root_valid
                    else (
                        primal_params
                        if same_representation
                        else response_root.parameters
                    )
                ),
                response_root_x=response_root_x,
                response_potential_layout=(
                    None
                    if not solve_outcome.root_valid
                    else (
                        layout
                        if same_representation
                        else response_root.problem.potential_layout
                    )
                ),
                response_certification=(
                    None
                    if not solve_outcome.root_valid or same_representation
                    else response_root.certification
                ),
                iterations=solve_outcome.iterations,
                solver_converged=solve_outcome.lane_converged,
                solver_residual=solve_outcome.lane_residual,
                raw_residual_l2=solve_outcome.raw_residual_l2,
                raw_residual_max=solve_outcome.raw_residual_max,
                electron_count_error=solve_outcome.electron_count_error,
                density_finite=solve_outcome.density_finite,
                padding_valid=solve_outcome.padding_valid,
                trajectory_status=solve_outcome.trajectory_status,
                fermi_status=solve_outcome.fermi_status,
                damping_state=solve_outcome.damping_state,
                status=solve_outcome.status,
                failure_code=solve_outcome.failure_code,
                failure_reason=solve_outcome.failure_reason,
                root_valid=solve_outcome.root_valid,
            )
        )

    parameter_paths = lanes[0].adapter.parameter_paths
    if any(
        lane.adapter.parameter_paths != parameter_paths for lane in lanes[1:]
    ):
        raise RuntimeError(
            "Conformer lanes produced incompatible SCF parameter trees."
        )
    valid_response_lanes = [lane for lane in lanes if lane.root_valid]
    response_parameter_paths = (
        valid_response_lanes[0].response_adapter.parameter_paths
        if valid_response_lanes
        else ()
    )
    if any(
        lane.response_adapter is None
        or lane.response_adapter.parameter_paths != response_parameter_paths
        for lane in valid_response_lanes
    ):
        raise RuntimeError(
            "Valid conformer lanes produced incompatible response parameter trees."
        )
    valid_mask = torch.tensor(
        [lane.root_valid for lane in lanes],
        dtype=torch.bool,
        device=positions.device,
    )
    return PreparedConvergedRootBatch(
        system=system,
        positions=positions,
        primal_representation=representation,
        response_representation=response_representation,
        response_settings=response_settings,
        strategy_label=route.strategy_label,
        lanes=tuple(lanes),
        parameter_paths=parameter_paths,
        model_parameter_refs=model_parameter_refs,
        model_parameter_snapshot=tuple(
            _snapshot_required_tensor(value) for value in model_parameter_refs
        ),
        model_parameter_versions=tuple(
            _tensor_version(value) for value in model_parameter_refs
        ),
        system_tensor_refs=system_tensor_refs,
        system_tensor_snapshot=tuple(
            _snapshot_required_tensor(value) for value in system_tensor_refs
        ),
        system_tensor_versions=tuple(
            _tensor_version(value) for value in system_tensor_refs
        ),
        positions_snapshot=_snapshot_required_tensor(positions),
        positions_version=_tensor_version(positions),
        numbers_snapshot=_snapshot_required_tensor(numbers),
        numbers_version=_tensor_version(numbers),
        config_identity=id(config),
        config_repr=repr(config),
        lane_input_snapshots=tuple(
            (
                _snapshot_tensor(lane.positions),
                _snapshot_tensor(lane.charge),
                _snapshot_tensor(lane.spin),
                _snapshot_tensor(lane.field),
                _snapshot_tensor(lane.field_grad),
            )
            for lane in lanes
        ),
        parameter_snapshots=tuple(
            tuple(_snapshot_required_tensor(param) for param in lane.parameters)
            for lane in lanes
        ),
        root_snapshots=tuple(
            _snapshot_required_tensor(lane.root_x) for lane in lanes
        ),
        response_root_snapshots=tuple(
            (
                None
                if lane.response_root_x is None
                or response_representation == representation
                else _snapshot_required_tensor(lane.response_root_x)
            )
            for lane in lanes
        ),
        input_tensor_versions=tuple(
            {
                path: int(_unwrap_primal_tensor(value)._version)
                for path, value in tensor_leaves(lane.inputs)
            }
            for lane in lanes
        ),
        valid_lane_mask=valid_mask,
        all_roots_valid=bool(valid_mask.all()),
    )


def _evaluate_lane_classical(
    system: System | PaddedExactSystemCore,
    lane_positions: Tensor,
    lane_charge: Tensor,
) -> tuple[tuple[str, Tensor], ...]:
    """Evaluate classical energy components for a single conformer lane."""
    if isinstance(system, PaddedExactSystemCore):
        if not system.core.classical_terms:
            return ()
        masked_pos = torch.where(
            system.validity.atoms.unsqueeze(-1), lane_positions, 0.0
        )
        chrg_scalar = lane_charge.squeeze(0)
        return tuple(
            (
                term.label,
                _evaluate_classical_descriptor(term, masked_pos, chrg_scalar),
            )
            for term in system.core.classical_terms
        )

    if not system.classicals.components:
        return ()
    core = system._active_exact_core()
    classical_terms = (
        {term.position: term for term in core.classical_terms}
        if core is not None
        else {}
    )
    classical: list[tuple[str, Tensor]] = []
    for position, component in enumerate(system.classicals.components):
        term = classical_terms.get(position)
        if term is not None:
            energy = _evaluate_classical_descriptor(
                term, lane_positions, lane_charge
            )
            classical.append((term.label, energy))
            continue

        captured_setup = (
            system.classical_setup(component) if core is not None else None
        )
        d3_setup = (
            captured_setup
            if isinstance(captured_setup, DispersionD3CoreSetup)
            else next(
                (
                    setup
                    for d3_component, setup in system.d3_setups
                    if component is d3_component
                ),
                None,
            )
        )
        if type(component) is DispersionD3:
            if d3_setup is None:
                raise RuntimeError("Single-system D3 setup is missing.")
            classical.append(
                (
                    component.label,
                    dispersion_d3_energy(d3_setup, lane_positions),
                )
            )
        elif type(component) is DispersionD4:
            d4_setup = (
                captured_setup
                if isinstance(captured_setup, DispersionD4CoreSetup)
                else (
                    system.d4_setup
                    if component is system.d4_classical
                    else None
                )
            )
            if d4_setup is None:
                raise RuntimeError("Single-system D4 setup is missing.")
            classical.append(
                (
                    component.label,
                    dispersion_d4_energy(d4_setup, lane_positions, lane_charge),
                )
            )
        elif type(component) is Halogen:
            halogen_setup = (
                captured_setup
                if isinstance(captured_setup, HalogenCoreSetup)
                else (
                    system.halogen_setup
                    if component is system.halogen_classical
                    else None
                )
            )
            if halogen_setup is None:
                raise RuntimeError("Single-system halogen setup is missing.")
            classical.append(
                (
                    component.label,
                    halogen_energy(halogen_setup, lane_positions),
                )
            )
        elif isinstance(captured_setup, IESCoreSetup) or (
            captured_setup is None and component is system.ies_classical
        ):
            ies_setup = (
                captured_setup
                if isinstance(captured_setup, IESCoreSetup)
                else system.ies_setup
            )
            if ies_setup is None:
                raise RuntimeError("Single-system IES setup is missing.")
            classical.append(
                (
                    component.label,
                    ies_energy(ies_setup, lane_positions, lane_charge),
                )
            )
        else:
            if isinstance(
                captured_setup, (RepulsionCoreSetup, ShortRangeBondCoreSetup)
            ):
                setup = captured_setup
            elif component is system.repulsion_classical:
                setup = system.repulsion_setup
            elif component is system.srb_classical:
                setup = system.srb_setup
            elif (
                (
                    isinstance(component, Repulsion)
                    and type(component) is not Repulsion
                )
                or (
                    isinstance(component, ShortRangeBond)
                    and type(component) is not ShortRangeBond
                )
                or (
                    isinstance(component, DispersionD4)
                    and type(component) is not DispersionD4
                )
                or (
                    isinstance(component, Halogen)
                    and type(component) is not Halogen
                )
                or (isinstance(component, IES) and type(component) is not IES)
            ):
                setup = component.get_cache(system.numbers, system.ihelp)
            else:
                setup = system._classical_extension_cache[component.label]
            classical.append(
                (
                    component.label,
                    component.get_energy(
                        lane_positions, setup, charge=lane_charge
                    ),
                )
            )
    return tuple(classical)


class _InvalidLaneOutput(torch.autograd.Function):
    """Return NaN values and NaN derivatives only when the failed lane is selected."""

    generate_vmap_rule = True

    @staticmethod
    def forward(
        like: Tensor,
        shape: tuple[int, ...],
        *dependencies: Tensor,
    ) -> Tensor:
        return torch.full(
            shape, float("nan"), dtype=like.dtype, device=like.device
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        like, shape, *dependencies = inputs
        ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
        ctx.dtype = like.dtype
        ctx.device = like.device
        ctx.shape = shape
        ctx.dep_metas = tuple(
            (
                d.shape,
                d.dtype,
                d.device,
                bool(d.requires_grad or ft.is_gradtrackingtensor(d)),
            )
            for d in dependencies
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        failed_lane_selected = (grad_output != 0.0).any()
        grad_deps = []
        for d_shape, d_dtype, d_device, d_req_grad in ctx.dep_metas:
            if not d_req_grad:
                grad_deps.append(None)
            else:
                nan = torch.full(
                    d_shape, float("nan"), dtype=d_dtype, device=d_device
                )
                zeros = torch.zeros(d_shape, dtype=d_dtype, device=d_device)
                grad_deps.append(torch.where(failed_lane_selected, nan, zeros))
        return (None, None, *grad_deps)

    @staticmethod
    def jvp(ctx, grad_like, grad_shape, *grad_dependencies):
        active_tangent = torch.zeros((), dtype=torch.bool, device=ctx.device)
        for tangent in grad_dependencies:
            if tangent is not None:
                active_tangent = active_tangent | (tangent != 0.0).any()
        nan = torch.full(
            ctx.shape, float("nan"), dtype=ctx.dtype, device=ctx.device
        )
        zeros = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        return torch.where(active_tangent, nan, zeros)


def _invalid_lane_dependencies(
    lane: ConvergedRootLaneRecord,
    lane_classical: tuple[tuple[str, Tensor], ...],
    model_parameter_refs: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Return differentiable inputs whose selected failed-lane gradient is invalid."""
    values = (
        lane.positions,
        lane.charge,
        lane.spin,
        lane.field,
        lane.field_grad,
        *lane.parameters,
        *(term_val for _, term_val in lane_classical),
        *model_parameter_refs,
    )
    return tuple(
        value
        for value in values
        if value is not None and value.is_floating_point()
    )


def _refresh_prepared_batch_inputs(
    prepared: PreparedConvergedRootBatch,
    positions: Tensor,
    chrg: Tensor | float | int | object,
    spin: Tensor | float | int | None | object,
    field: Tensor | None | object,
    field_grad: Tensor | None | object,
    *,
    strict_mode: bool,
) -> tuple[ConvergedRootLaneRecord, ...]:
    """Validate prepared primal state and rebuild call-local differentiable lane inputs."""
    system = prepared.system
    first_lane = prepared.lanes[0]
    val_field = first_lane.field if field is _UNSET else field
    val_fgrad = first_lane.field_grad if field_grad is _UNSET else field_grad
    (
        _core,
        numbers,
        config,
        integral_setup,
        h0_setup,
        route,
        core_index,
    ) = _validate_system_and_positions(
        system,
        positions,
        val_field,  # type: ignore[arg-type]
        val_fgrad,  # type: ignore[arg-type]
    )
    batch_size = positions.shape[0]
    if batch_size != len(prepared.lanes):
        raise ValueError(
            f"Prepared batch has {len(prepared.lanes)} lanes, but positions "
            f"has {batch_size} lanes."
        )
    effective_scf_config = config.scf
    if chrg is _UNSET:
        charges = tuple(
            lane.charge.to(device=positions.device, dtype=positions.dtype)
            for lane in prepared.lanes
        )
    else:
        charges = _normalize_batch_scalar(
            "charge",
            chrg,  # type: ignore[arg-type]
            positions,
            batch_size,
        )
    if spin is _UNSET:
        spins = tuple(
            (
                None
                if lane.spin is None
                else lane.spin.to(
                    device=positions.device, dtype=positions.dtype
                )
            )
            for lane in prepared.lanes
        )
    elif spin is None:
        spins = tuple(None for _ in range(batch_size))
    else:
        spins = _normalize_batch_scalar(
            "spin",
            spin,  # type: ignore[arg-type]
            positions,
            batch_size,
        )
    if field is _UNSET:
        fields = tuple(
            (
                None
                if lane.field is None
                else lane.field.to(
                    device=positions.device, dtype=positions.dtype
                )
            )
            for lane in prepared.lanes
        )
    else:
        fields = _normalize_batch_perturbation(
            "field",
            field,  # type: ignore[arg-type]
            positions,
            batch_size,
            (3,),
        )
    if field_grad is _UNSET:
        field_grads = tuple(
            (
                None
                if lane.field_grad is None
                else lane.field_grad.to(
                    device=positions.device, dtype=positions.dtype
                )
            )
            for lane in prepared.lanes
        )
    else:
        field_grads = _normalize_batch_perturbation(
            "field_grad",
            field_grad,  # type: ignore[arg-type]
            positions,
            batch_size,
            (3, 3),
        )

    updated_lanes: list[ConvergedRootLaneRecord] = []
    for lane_index, lane in enumerate(prepared.lanes):
        lane_pos = positions[lane_index]
        lane_chrg = charges[lane_index]
        lane_spin = spins[lane_index]
        lane_field = fields[lane_index]
        lane_fgrad = field_grads[lane_index]
        (
            matrices,
            refocc,
            overlap_norm,
            n0,
            occupation,
            nel,
            call_interactions,
            interaction_data,
            inputs,
        ) = _build_lane_inputs(
            system,
            numbers,
            config,
            integral_setup,
            h0_setup,
            core_index,
            effective_scf_config,
            lane_pos,
            lane_chrg,
            lane_spin,
            lane_field,
            lane_fgrad,
        )
        new_adapter, parameters = _SCFMapTensorTree.prepare(
            inputs, prepared.primal_representation, lane.potential_layout
        )
        if new_adapter.parameter_paths != lane.adapter.parameter_paths:
            raise RuntimeError(
                "Refreshed SCF input parameter paths do not match prepared batch."
            )
        response_adapter = None
        response_parameters = None
        if lane.root_valid:
            assert lane.response_adapter is not None
            if (
                prepared.response_representation
                == prepared.primal_representation
            ):
                response_adapter, response_parameters = new_adapter, parameters
            else:
                response_adapter, response_parameters = (
                    _SCFMapTensorTree.prepare(
                        inputs,
                        prepared.response_representation,
                        lane.response_potential_layout,
                    )
                )
            if (
                response_adapter.parameter_paths
                != lane.response_adapter.parameter_paths
            ):
                raise RuntimeError(
                    "Refreshed SCF response parameter paths do not match "
                    "the prepared response batch."
                )
        root_valid = lane.root_valid
        failure_code = lane.failure_code
        failure_reason = lane.failure_reason
        status = lane.status
        solver_converged = lane.solver_converged

        primal_nel = _unwrap_primal_tensor(nel)
        prepared_nel = _unwrap_primal_tensor(lane.nel)
        nel_matches = primal_nel.shape == prepared_nel.shape and bool(
            torch.allclose(
                primal_nel,
                prepared_nel,
                atol=_ELECTRON_COUNT_TOL,
                rtol=0.0,
            )
        )
        if (
            (chrg is not _UNSET or spin is not _UNSET)
            and root_valid
            and not nel_matches
        ):
            root_valid = False
            failure_code = int(BatchRootFailureCode.ELECTRON_COUNT)
            failure_reason = (
                "Refreshed charge/spin electron counts do not match the "
                "prepared physical root."
            )
            if strict_mode:
                _raise_strict_lane_failure(
                    lane_index, failure_code, failure_reason
                )
            msg = f"[Lane {lane_index}] {failure_reason}"
            warnings.warn(msg, SCFConvergenceWarning, stacklevel=2)
            OutputHandler.warn(msg, SCFConvergenceWarning)
            solver_converged = torch.zeros_like(lane.solver_converged)
            status = SCFStatus(
                trajectory_failure=lane.status.trajectory_failure,
                final_failure=torch.tensor(
                    failure_code,
                    dtype=torch.int64,
                    device=positions.device,
                ),
            )

        updated_lanes.append(
            replace(
                lane,
                positions=lane_pos,
                charge=lane_chrg,
                spin=lane_spin,
                field=lane_field,
                field_grad=lane_fgrad,
                integrals=matrices,
                refocc=refocc,
                overlap_norm=overlap_norm,
                n0=n0,
                occupation=occupation,
                nel=nel,
                interactions=call_interactions,
                interaction_data=interaction_data,
                inputs=inputs,
                parameters=parameters,
                response_adapter=response_adapter,
                response_parameters=response_parameters,
                solver_converged=solver_converged,
                status=status,
                failure_code=failure_code,
                failure_reason=failure_reason,
                root_valid=root_valid,
            )
        )

    _validate_prepared_primal(
        prepared,
        positions,
        charges,
        spins,
        fields,
        field_grads,
        numbers,
        config,
    )
    if not is_compiling():
        for lane, param_snapshots in zip(
            updated_lanes, prepared.parameter_snapshots
        ):
            if len(lane.parameters) != len(param_snapshots) or any(
                not _matches_snapshot(param, snapshot)
                for param, snapshot in zip(lane.parameters, param_snapshots)
            ):
                raise RuntimeError(
                    "Prepared physical roots are stale for the supplied Model "
                    "or System setup; prepare roots again."
                )
    if strict_mode:
        for lane in updated_lanes:
            if not lane.root_valid:
                _raise_strict_lane_failure(
                    lane.lane_index,
                    lane.failure_code,
                    lane.failure_reason or "Invalid physical root.",
                )
    return tuple(updated_lanes)


def _evaluate_valid_lanes_mapped(
    valid_lanes: list[ConvergedRootLaneRecord],
    valid_indices: list[int],
    representation: int,
    primal_representation: int,
    response_settings: SCFResponseSettings,
    device: torch.device,
    strict_mode: bool,
    response_diagnostics: ResponseDiagnosticCollector | None = None,
):
    """Evaluate valid conformer lanes simultaneously via vmap over implicit_root."""
    del valid_indices, strict_mode
    same_representation = representation == primal_representation
    adapter = (
        valid_lanes[0].adapter
        if same_representation
        else valid_lanes[0].response_adapter
    )
    assert adapter is not None
    layout = (
        valid_lanes[0].potential_layout
        if same_representation
        else valid_lanes[0].response_potential_layout
    )
    param_count = len(adapter.parameter_paths)
    pot_state_proj: Callable[[Tensor], Tensor] | None = None
    if representation == labels.SCP_MODE_POTENTIAL and (
        bool((adapter.template.numbers == 0).any())
        or bool((~adapter.template.orbital_mask).any())
    ):
        assert layout is not None
        pad_state, _ = _padded_potential_projections(adapter.template, layout)
        pot_state_proj = lambda val, _ps=pad_state, _lay=layout: _ps(
            _project_potential_padding(val, _lay)
        )
    raw_map: Callable[..., Tensor] = (
        (lambda x, *p, _sp=pot_state_proj: _sp(adapter(x, *p)))
        if pot_state_proj is not None
        else adapter
    )

    roots = torch.stack(
        [
            lane.root_x if same_representation else lane.response_root_x
            for lane in valid_lanes
        ],
        dim=0,
    )
    iterations_v = torch.stack([lane.iterations for lane in valid_lanes], dim=0)
    converged_v = torch.stack(
        [lane.solver_converged for lane in valid_lanes], dim=0
    )
    residuals_v = torch.stack(
        [lane.solver_residual for lane in valid_lanes], dim=0
    )
    damp_values_v = torch.stack(
        [lane.damping_state.value for lane in valid_lanes], dim=0
    )
    damp_switched_v = torch.stack(
        [lane.damping_state.switched for lane in valid_lanes], dim=0
    )
    stacked_params = tuple(
        torch.stack(
            [
                (
                    lane.parameters
                    if same_representation
                    else lane.response_parameters
                )[p_idx]
                for lane in valid_lanes
            ],
            dim=0,
        )
        for p_idx in range(param_count)
    )

    has_traj = all(lane.trajectory_status is not None for lane in valid_lanes)
    if has_traj:
        traj_fields = tuple(
            torch.stack(
                [getattr(lane.trajectory_status, name) for lane in valid_lanes],
                dim=0,
            )
            for name in (
                "fermi_converged",
                "electron_count_ok",
                "invalid_temperature",
                "too_many_electrons",
                "first_failure",
            )
        )
    else:
        bool_zeros = torch.zeros(
            len(valid_lanes), dtype=torch.bool, device=device
        )
        int_zeros = torch.zeros(
            len(valid_lanes), dtype=torch.int64, device=device
        )
        traj_fields = (
            bool_zeros,
            bool_zeros,
            bool_zeros,
            bool_zeros,
            int_zeros,
        )

    def _attach_lane_root(root: Tensor, *params: Tensor) -> Tensor:
        if implicit_root is not _BATCH_DEFAULT_IMPLICIT_ROOT:
            return implicit_root(root, *params, raw_map=raw_map)
        if (
            response_diagnostics is None
            and response_settings.rtol == _LINEAR_SOLVE_RTOL
            and response_settings.atol == _LINEAR_SOLVE_ATOL
            and response_settings.maxiter is None
        ):
            return _functional_implicit_root(root, *params, raw_map=raw_map)
        method = (
            "bicgstab"
            if response_diagnostics is None
            else response_diagnostics.solve
        )
        return implicit_root(
            root,
            *params,
            raw_map=raw_map,
            method=method,
            rtol=response_settings.rtol,
            atol=response_settings.atol,
            maxiter=response_settings.maxiter,
        )

    def _finalize_valid_lane(
        attached: Tensor,
        iteration: Tensor,
        is_converged: Tensor,
        residual: Tensor,
        damp_val: Tensor,
        damp_sw: Tensor,
        t_fconv: Tensor,
        t_ecount: Tensor,
        t_itemp: Tensor,
        t_tmany: Tensor,
        t_ffail: Tensor,
        *params: Tensor,
    ):
        lane_inputs, _ = adapter._rebuild(params)
        mapped_result = adapter.evaluate(attached, *params)
        if pot_state_proj is not None:
            mapped_result = replace(
                mapped_result, mapped_x=pot_state_proj(mapped_result.mapped_x)
            )
        traj_status = (
            SCFTrajectoryStatus(t_fconv, t_ecount, t_itemp, t_tmany, t_ffail)
            if has_traj
            else None
        )
        damp_state = DampingState(damp_val, damp_sw)
        if representation == labels.SCP_MODE_CHARGE:
            finalized = finalize_scf(
                mapped_result,
                lane_inputs,
                iteration,
                is_converged,
                residual,
                traj_status,
                damp_state,
                root_x=attached,
            )
        elif representation == labels.SCP_MODE_POTENTIAL:
            assert layout is not None
            finalized = finalize_scf_potential(
                mapped_result,
                lane_inputs,
                layout,
                iteration,
                is_converged,
                residual,
                traj_status,
                damp_state,
                root_x=attached,
            )
        else:
            finalized = finalize_scf_fock(
                mapped_result,
                lane_inputs,
                iteration,
                is_converged,
                residual,
                traj_status,
                damp_state,
                root_x=attached,
            )
        dummy = finalized.energy.new_zeros(())
        q_dip = (
            finalized.final_charges.dipole
            if finalized.final_charges.dipole is not None
            else dummy
        )
        q_quad = (
            finalized.final_charges.quad
            if finalized.final_charges.quad is not None
            else dummy
        )
        v_dip = (
            finalized.final_potential.dipole
            if finalized.final_potential.dipole is not None
            else dummy
        )
        v_quad = (
            finalized.final_potential.quad
            if finalized.final_potential.quad is not None
            else dummy
        )
        return (
            finalized.energy,
            finalized.fenergy,
            finalized.final_charges.mono,
            q_dip,
            q_quad,
            finalized.density,
            finalized.coefficients,
            finalized.emo,
            finalized.occupation,
            finalized.final_potential.mono,
            v_dip,
            v_quad,
            finalized.hamiltonian,
        )

    attach_in_dims = (0,) * (1 + param_count)
    finalize_in_dims = (0,) * (11 + param_count)

    @torch._dynamo.disable
    def _run_mapped_lanes(
        roots_in: Tensor,
        iter_in: Tensor,
        conv_in: Tensor,
        res_in: Tensor,
        dval_in: Tensor,
        dsw_in: Tensor,
        tfc_in: Tensor,
        tec_in: Tensor,
        tit_in: Tensor,
        ttm_in: Tensor,
        tff_in: Tensor,
        *params_in: Tensor,
    ):
        ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
        use_monitors = (
            response_diagnostics is not None
            and not is_compiling()
            and ft.get_interpreter_stack() is None
            and torch.is_grad_enabled()
        )
        if use_monitors and response_diagnostics is not None:
            params_for_attach: tuple[Tensor, ...] = (
                _ResponseParamsMonitor.apply(response_diagnostics, *params_in)
            )
        else:
            params_for_attach = params_in
        attached_roots = vmap(_attach_lane_root, in_dims=attach_in_dims)(
            roots_in, *params_for_attach
        )
        if use_monitors and response_diagnostics is not None:
            attached_roots = _ResponseOutputMonitor.apply(
                response_diagnostics, attached_roots
            )
        return vmap(_finalize_valid_lane, in_dims=finalize_in_dims)(
            attached_roots,
            iter_in,
            conv_in,
            res_in,
            dval_in,
            dsw_in,
            tfc_in,
            tec_in,
            tit_in,
            ttm_in,
            tff_in,
            *params_in,
        )

    return _run_mapped_lanes(
        roots,
        iterations_v,
        converged_v,
        residuals_v,
        damp_values_v,
        damp_switched_v,
        *traj_fields,
        *stacked_params,
    )


def _assemble_batch_result(
    system: System | PaddedExactSystemCore,
    lanes: tuple[ConvergedRootLaneRecord, ...],
    eval_positions: Tensor,
    valid_indices: list[int],
    valid_outputs: tuple[Tensor, ...] | None,
    model_parameter_refs: tuple[Tensor, ...],
) -> Result:
    """Assemble the immutable batched Result from valid and invalid lane outputs."""
    batch_size = len(lanes)
    lane_classicals = [
        _evaluate_lane_classical(system, lane.positions, lane.charge)
        for lane in lanes
    ]
    invalid_dependencies = {
        idx: _invalid_lane_dependencies(
            lane, lane_classicals[idx], model_parameter_refs
        )
        for idx, lane in enumerate(lanes)
        if not lane.root_valid
    }
    classical_labels = (
        tuple(label for label, _ in lane_classicals[0])
        if lane_classicals and lane_classicals[0]
        else ()
    )
    batched_classical: list[tuple[str, Tensor]] = []
    for term_idx, label in enumerate(classical_labels):
        term_values = [
            (
                lane_classicals[idx][term_idx][1]
                if lanes[idx].root_valid
                else _InvalidLaneOutput.apply(
                    lane_classicals[idx][term_idx][1],
                    tuple(lane_classicals[idx][term_idx][1].shape),
                    *invalid_dependencies[idx],
                )
            )
            for idx in range(batch_size)
        ]
        batched_classical.append((label, torch.stack(term_values, dim=0)))

    if batched_classical:
        classical_energy = torch.stack(
            tuple(value for _, value in batched_classical), dim=0
        ).sum(dim=0)
    else:
        classical_energy = eval_positions.new_zeros(eval_positions.shape[:-1])

    first_inputs = lanes[0].inputs
    has_dipole = first_inputs.charge_layout.dipole_shape is not None
    has_quad = first_inputs.charge_layout.quadrupole_shape is not None

    if len(valid_indices) == batch_size and valid_outputs is not None:
        (
            scf_energy,
            fenergy,
            q_mono,
            q_dip,
            q_quad,
            density,
            coefficients,
            emo,
            occupation,
            v_mono,
            v_dip,
            v_quad,
            hamiltonian,
        ) = valid_outputs
    else:
        nat = eval_positions.shape[1]
        norb = first_inputs.orbital_mask.shape[-1]
        nao = first_inputs.integrals.hcore.shape[-1]
        nspin = first_inputs.nel.shape[0]
        valid_pos_map = {
            lane_idx: out_idx for out_idx, lane_idx in enumerate(valid_indices)
        }

        def _assemble_field(
            out_tuple_idx: int, shape: tuple[int, ...]
        ) -> Tensor:
            items: list[Tensor] = []
            for idx in range(batch_size):
                if idx in valid_pos_map and valid_outputs is not None:
                    items.append(
                        valid_outputs[out_tuple_idx][valid_pos_map[idx]]
                    )
                else:
                    items.append(
                        _InvalidLaneOutput.apply(
                            lanes[idx].positions,
                            shape,
                            *invalid_dependencies[idx],
                        )
                    )
            return torch.stack(items, dim=0)

        scf_energy = _assemble_field(0, (nat,))
        fenergy = _assemble_field(1, (nat,))
        q_mono = _assemble_field(2, (norb,))
        q_dip = _assemble_field(3, (nat, 3)) if has_dipole else None
        q_quad = _assemble_field(4, (nat, 6)) if has_quad else None
        density = _assemble_field(5, (nao, nao))
        coefficients = _assemble_field(6, (nao, nao))
        emo = _assemble_field(7, (nao,))
        occupation = _assemble_field(8, (nspin, nao))
        v_mono = _assemble_field(9, (norb,))
        v_dip = _assemble_field(10, (nat, 3)) if has_dipole else None
        v_quad = _assemble_field(11, (nat, 6)) if has_quad else None
        hamiltonian = _assemble_field(12, (nao, nao))

    charges_out = SCFCharges(
        mono=q_mono,
        dipole=q_dip if has_dipole else None,
        quad=q_quad if has_quad else None,
    )
    potential_out = SCFPotential(
        mono=v_mono,
        dipole=v_dip if has_dipole else None,
        quad=v_quad if has_quad else None,
    )

    overlap = torch.stack([lane.integrals.overlap for lane in lanes], dim=0)
    hcore = torch.stack([lane.integrals.hcore for lane in lanes], dim=0)
    dipole_ints = (
        None
        if lanes[0].integrals.dipole is None
        else torch.stack(
            [
                lane.integrals.dipole
                for lane in lanes
                if lane.integrals.dipole is not None
            ],
            dim=0,
        )
    )
    quad_ints = (
        None
        if lanes[0].integrals.quadrupole is None
        else torch.stack(
            [
                lane.integrals.quadrupole
                for lane in lanes
                if lane.integrals.quadrupole is not None
            ],
            dim=0,
        )
    )
    overlap_norm = torch.stack([lane.overlap_norm for lane in lanes], dim=0)
    iterations = torch.stack([lane.iterations for lane in lanes], dim=0)
    converged = torch.stack([lane.solver_converged for lane in lanes], dim=0)
    residual = torch.stack([lane.solver_residual for lane in lanes], dim=0)
    status = SCFStatus(
        trajectory_failure=torch.stack(
            [lane.status.trajectory_failure for lane in lanes], dim=0
        ),
        final_failure=torch.stack(
            [lane.status.final_failure for lane in lanes], dim=0
        ),
    )

    total_energy = classical_energy + scf_energy + fenergy
    return Result.snapshot(
        energy=total_energy,
        scf=scf_energy,
        classical=tuple(batched_classical),
        fenergy=fenergy,
        charges=charges_out,
        density=density,
        coefficients=coefficients,
        emo=emo,
        occupation=occupation,
        potential=potential_out,
        hamiltonian=hamiltonian,
        overlap=overlap,
        hcore=hcore,
        dipole_integrals=dipole_ints,
        quadrupole_integrals=quad_ints,
        overlap_norm=overlap_norm,
        iterations=iterations,
        converged=converged,
        residual=residual,
        status=status,
    )


def evaluate_converged_root_batch(
    batch_or_system: (
        PreparedConvergedRootBatch | System | PaddedExactSystemCore
    ),
    positions: Tensor | None = None,
    chrg: Tensor | float | int = _UNSET,
    spin: Tensor | float | int | None = _UNSET,
    field: Tensor | None = _UNSET,
    field_grad: Tensor | None = _UNSET,
    *,
    strict: bool | None = None,
    eager_primal_fermi: bool | str = False,
    response_diagnostics: ResponseDiagnosticCollector | None = None,
) -> Result:
    """Evaluate a conformer batch using mapped implicit differentiation.

    Can be called either with a :class:`PreparedConvergedRootBatch` (optionally
    supplying transformed ``positions``, ``chrg``, ``field``, or ``field_grad``
    at the exact prepared primal point) or directly with ``(system, positions,
    ...)``. Reusing prepared roots at changed physical inputs is rejected;
    call :func:`prepare_converged_root_batch` again for a new primal point.
    Valid lanes are evaluated simultaneously via ``torch.func.vmap`` over
    :func:`implicit_root` and functional SCF finalization. Invalid lanes in
    non-strict mode are isolated from :func:`implicit_root` and receive
    ``NaN`` physical observables and ``NaN`` derivatives without corrupting
    valid lanes.
    """
    if isinstance(batch_or_system, PreparedConvergedRootBatch):
        prepared = batch_or_system
        system = prepared.system
        sys_cfg = (
            system.core.config
            if isinstance(system, PaddedExactSystemCore)
            else system.config
        )
        effective_strict = _resolve_strictness(sys_cfg.scf, strict)
        eval_positions = prepared.positions if positions is None else positions
        with eager_primal_fermi_scope(False):
            lanes = _refresh_prepared_batch_inputs(
                prepared,
                eval_positions,
                chrg,
                spin,
                field,
                field_grad,
                strict_mode=effective_strict,
            )
            representation = prepared.response_representation
            valid_indices = [
                idx for idx, lane in enumerate(lanes) if lane.root_valid
            ]
            if valid_indices:
                valid_lanes = [lanes[idx] for idx in valid_indices]
                valid_outputs = _evaluate_valid_lanes_mapped(
                    valid_lanes,
                    valid_indices,
                    representation,
                    prepared.primal_representation,
                    prepared.response_settings,
                    eval_positions.device,
                    effective_strict,
                    response_diagnostics=response_diagnostics,
                )
            else:
                valid_outputs = None

            return _assemble_batch_result(
                system,
                lanes,
                eval_positions,
                valid_indices,
                valid_outputs,
                prepared.model_parameter_refs,
            )
    elif isinstance(batch_or_system, (System, PaddedExactSystemCore)):
        if positions is None:
            raise ValueError(
                "positions must be provided when calling "
                "evaluate_converged_root_batch with a System."
            )
        prepared = prepare_converged_root_batch(
            batch_or_system,
            positions,
            chrg=defaults.CHRG if chrg is _UNSET else chrg,
            spin=defaults.SPIN if spin is _UNSET else spin,
            field=None if field is _UNSET else field,
            field_grad=None if field_grad is _UNSET else field_grad,
            strict=strict,
            eager_primal_fermi=eager_primal_fermi,
        )
        return evaluate_converged_root_batch(
            prepared,
            positions=positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
            strict=strict,
            response_diagnostics=response_diagnostics,
        )
    else:
        raise TypeError(
            "Expected PreparedConvergedRootBatch, System, or PaddedExactSystemCore as first argument."
        )
