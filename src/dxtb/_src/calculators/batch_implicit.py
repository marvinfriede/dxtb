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

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import IntEnum
import warnings

import torch
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.exceptions import DeviceError, DtypeError
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
from dxtb._src.scf.functional.anderson import _solve_map_anderson
from dxtb._src.scf.functional.core import replace_tensor_leaves, tensor_leaves
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
    _solver_settings,
    public_scf_status,
    resolve_full_scf_route,
)
from dxtb._src.scf.functional.implicit import _SCFMapTensorTree, implicit_root
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
from dxtb._src.scf.functional.solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_INVALID_TEMPERATURE,
    FERMI_FAILURE_NONCONVERGENCE,
    FERMI_FAILURE_NONE,
    FERMI_FAILURE_TOO_MANY_ELECTRONS,
    DampingState,
    SCFSolverSettings,
    SCFTrajectoryStatus,
    _project_potential_padding,
    _solve_map,
    fermi_failure_code,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
)
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
    "BatchRootFailureCode",
    "ConvergedRootLaneRecord",
    "PreparedConvergedRootBatch",
    "evaluate_converged_root_batch",
    "prepare_converged_root_batch",
]

_ELECTRON_COUNT_TOL = 1.0e-6
_UNSET: Any = object()


class BatchRootFailureCode(IntEnum):
    """Explicit failure categories for converged-root batch validation."""

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
class PreparedConvergedRootBatch:
    """Validated conformer batch sharing one :class:`System` or padded core owner."""

    system: System | PaddedExactSystemCore
    positions: Tensor
    representation: int
    effective_mixer: int
    lanes: tuple[ConvergedRootLaneRecord, ...]
    parameter_paths: tuple[str, ...]
    valid_lane_mask: Tensor
    all_roots_valid: bool


def _unwrap_primal_tensor(x: Tensor) -> Tensor:
    """Peel grad-tracking and dual layers for detached primal root finding."""
    if is_compiling():
        return x.detach()
    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    current = x
    while ft.is_functorch_wrapped_tensor(current):
        current = ft.get_unwrapped(current)
    with torch._C._DisableFuncTorch():  # pyright: ignore[reportAttributeAccessIssue]
        primal = torch.autograd.forward_ad.unpack_dual(current).primal
        return primal.detach()


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
        effective_mixer = (
            labels.MIXER_ANDERSON
            if config.scf.mixer == labels.MIXER_BROYDEN
            else config.scf.mixer
        )
        from types import SimpleNamespace

        route = SimpleNamespace(
            kind="functional",
            representation=config.scf.scp_mode,
            effective_mixer=effective_mixer,
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
            f"Converged-root conformer batching expects positions with shape "
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
    """Build differentiable call-local integrals, interactions, and SCFInputs."""
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
    return int(BatchRootFailureCode.NONE), ""


def prepare_converged_root_batch(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    strict: bool | None = None,
) -> PreparedConvergedRootBatch:
    """Prepare and validate independent physical SCF roots for a conformer batch.

    Parameters
    ----------
    system : System
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
        Shared ``(3, 3)`` or per-lane ``(B, 3, 3)`` electric field gradient.
    strict : bool | None, default None
        When ``True`` (or when ``None`` and ``config.scf.force_convergence``
        is enabled), raise on any lane root or Fermi validation failure.
        When ``False``, record per-lane failure metadata without raising on
        primal nonconvergence or residual/state failures so valid lanes can
        be evaluated independently.
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
    effective_scf_config = replace(config.scf, mixer=route.effective_mixer)
    strict_mode = (
        bool(effective_scf_config.force_convergence)
        if strict is None
        else bool(strict)
    )

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
    settings = _solver_settings(effective_scf_config, positions.dtype)
    has_padding = bool((numbers == 0).any())

    lanes: list[ConvergedRootLaneRecord] = []
    for lane_index in range(batch_size):
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

        primal_inputs = _detach_scf_inputs(inputs)
        primal_pos = _unwrap_primal_tensor(lane_pos)
        primal_chrg = _unwrap_primal_tensor(lane_chrg)
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

            state_projection: Callable[[Tensor], Tensor] | None = None
            residual_projection: Callable[[Tensor], Tensor] | None = None
            if representation == labels.SCP_MODE_CHARGE:
                initial_state = pack_charges(
                    guess_charges, primal_inputs.charge_layout
                )
                layout = None
                if lane_padded:
                    state_projection, residual_projection = (
                        _padded_charge_projections(primal_inputs)
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
                    pad_state, residual_projection = (
                        _padded_potential_projections(primal_inputs, layout)
                    )
                    state_projection = (
                        lambda val, _ps=pad_state, _bp=base_proj: _ps(_bp(val))
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
                    map_fn = (
                        lambda x, _inp=primal_inputs, _lay=layout: potential_map(
                            x, _inp, _lay
                        )
                    )
            elif representation == labels.SCP_MODE_FOCK:
                initial_state = fock_initial_guess(guess_charges, primal_inputs)
                layout = None
                if lane_padded:
                    matrix_mask = primal_inputs.orbital_mask.unsqueeze(
                        -1
                    ) & primal_inputs.orbital_mask.unsqueeze(-2)
                    state_projection = (
                        lambda val, _mm=matrix_mask, _hc=primal_inputs.integrals.hcore: torch.where(
                            _mm, val, _hc
                        )
                    )
                    residual_projection = (
                        lambda val, _mm=matrix_mask: torch.where(_mm, val, 0.0)
                    )
                map_fn = lambda x, _inp=primal_inputs: fock_map(x, _inp)
            else:
                raise ValueError(
                    f"Unknown convergence target (SCP mode) '{representation}'."
                )

            if lane_padded and state_projection is not None:
                initial_state = state_projection(initial_state)

            if isinstance(settings, SCFSolverSettings):
                outcome = _solve_map(
                    initial_state,
                    settings,
                    map_fn,
                    state_projection=state_projection,
                    eager=True,
                )
            else:
                outcome = _solve_map_anderson(
                    initial_state,
                    settings,
                    map_fn,
                    state_projection=state_projection,
                    eager=True,
                )

            adapter, primal_params = _SCFMapTensorTree.prepare(
                primal_inputs, representation, layout
            )
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

            if representation == labels.SCP_MODE_CHARGE:
                final_spectral = independent_map.spectral
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
                occ_masked.sum(dim=-1) - primal_inputs.nel
            ).abs().amax()
            ao_populations = torch.where(
                active_orb,
                primal_inputs.n0 - final_spectral.charges.mono,
                0.0,
            )
            ao_electron_error = (
                ao_populations.sum(dim=-1) - primal_inputs.nel.sum(dim=-1)
            ).abs().amax()
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
                ao_matrix_mask = active_orb.unsqueeze(-1) & active_orb.unsqueeze(
                    -2
                )
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
                mixer_label = (
                    "Simple"
                    if route.effective_mixer == labels.MIXER_LINEAR
                    else "Anderson"
                )
                damp = (
                    effective_scf_config.damp_dynamic_factor
                    if bool(outcome.damping_state.switched)
                    else effective_scf_config.damp
                )
                failure_reason = (
                    f"SCF does not converge after {effective_scf_config.maxiter} "
                    f"cycles using {mixer_label} mixing with a damping factor "
                    f"of {damp} (residual L2={float(raw_residual_l2):.3e}, "
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
                        device=positions.device,
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

        if not root_valid:
            assert failure_reason is not None
            if strict_mode:
                _raise_strict_lane_failure(
                    lane_index, failure_code, failure_reason
                )
            msg = f"[Lane {lane_index}] {failure_reason}"
            warnings.warn(msg, SCFConvergenceWarning, stacklevel=2)
            OutputHandler.warn(msg, SCFConvergenceWarning)

        _, parameters = _SCFMapTensorTree.prepare(
            inputs, representation, layout
        )
        lanes.append(
            ConvergedRootLaneRecord(
                lane_index=lane_index,
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
                potential_layout=layout,
                adapter=adapter,
                parameters=parameters,
                root_x=outcome.root_x.detach(),
                iterations=outcome.iterations,
                solver_converged=lane_converged,
                solver_residual=lane_residual,
                raw_residual_l2=raw_residual_l2,
                raw_residual_max=raw_residual_max,
                electron_count_error=electron_count_error,
                density_finite=density_finite,
                padding_valid=padding_valid,
                trajectory_status=outcome.trajectory_status,
                fermi_status=final_fermi_status,
                damping_state=outcome.damping_state,
                status=status,
                failure_code=failure_code,
                failure_reason=failure_reason,
                root_valid=root_valid,
            )
        )

    parameter_paths = lanes[0].adapter.parameter_paths
    if any(
        lane.adapter.parameter_paths != parameter_paths for lane in lanes[1:]
    ):
        raise RuntimeError(
            "Conformer lanes produced incompatible SCF parameter trees."
        )
    valid_mask = torch.tensor(
        [lane.root_valid for lane in lanes],
        dtype=torch.bool,
        device=positions.device,
    )
    return PreparedConvergedRootBatch(
        system=system,
        positions=positions,
        representation=representation,
        effective_mixer=route.effective_mixer,
        lanes=tuple(lanes),
        parameter_paths=parameter_paths,
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
                (component.label, dispersion_d3_energy(d3_setup, lane_positions))
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
    """Rebuild differentiable lane inputs at transformed tensors for an existing root batch."""
    system = prepared.system
    val_field = None if field is _UNSET else field
    val_fgrad = None if field_grad is _UNSET else field_grad
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
    effective_scf_config = replace(config.scf, mixer=route.effective_mixer)
    settings = _solver_settings(effective_scf_config, positions.dtype)
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
            None
            if lane.spin is None
            else lane.spin.to(device=positions.device, dtype=positions.dtype)
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
            None
            if lane.field is None
            else lane.field.to(device=positions.device, dtype=positions.dtype)
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
            None
            if lane.field_grad is None
            else lane.field_grad.to(
                device=positions.device, dtype=positions.dtype
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
            inputs, prepared.representation, lane.potential_layout
        )
        if new_adapter.parameter_paths != lane.adapter.parameter_paths:
            raise RuntimeError(
                "Refreshed SCF input parameter paths do not match prepared batch."
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
        if root_valid and not nel_matches:
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

        in_transform = is_compiling() or any(
            is_functorch_tensor(p)
            or torch.autograd.forward_ad.unpack_dual(p).tangent is not None
            for p in parameters
        )
        if root_valid and not in_transform:
            with torch.no_grad():
                primal_params = tuple(p.detach() for p in parameters)
                mapped_x = lane.adapter(lane.root_x, *primal_params)
                lane_is_padded = bool((numbers == 0).any()) or bool(
                    (~inputs.orbital_mask).any()
                )
                if (
                    lane_is_padded
                    and prepared.representation == labels.SCP_MODE_POTENTIAL
                ):
                    assert lane.potential_layout is not None
                    pad_state, res_proj = _padded_potential_projections(
                        lane.adapter.template, lane.potential_layout
                    )
                    mapped_x = pad_state(
                        _project_potential_padding(
                            mapped_x, lane.potential_layout
                        )
                    )
                    raw_diff = res_proj(mapped_x - lane.root_x)
                else:
                    raw_diff = mapped_x - lane.root_x
                    if lane_is_padded:
                        if prepared.representation == labels.SCP_MODE_CHARGE:
                            _, res_proj = _padded_charge_projections(
                                lane.adapter.template
                            )
                            raw_diff = res_proj(raw_diff)
                        elif prepared.representation == labels.SCP_MODE_FOCK:
                            om = lane.adapter.template.orbital_mask
                            mm = om.unsqueeze(-1) & om.unsqueeze(-2)
                            raw_diff = torch.where(mm, raw_diff, 0.0)
                res_l2 = torch.linalg.vector_norm(raw_diff)
                res_max = raw_diff.abs().amax()
                if not bool(
                    torch.isfinite(raw_diff).all()
                    & (res_l2 <= settings.x_atol)
                    & (res_max <= settings.x_atol_max)
                ):
                    root_valid = False
                    failure_code = int(BatchRootFailureCode.RESIDUAL_TOLERANCE)
                    failure_reason = (
                        "Prepared physical root is no longer valid for the "
                        f"supplied physical inputs (L2={float(res_l2):.3e}, "
                        f"max={float(res_max):.3e})."
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
                solver_converged=solver_converged,
                status=status,
                failure_code=failure_code,
                failure_reason=failure_reason,
                root_valid=root_valid,
            )
        )
    return tuple(updated_lanes)


def evaluate_converged_root_batch(
    batch_or_system: PreparedConvergedRootBatch | System | PaddedExactSystemCore,
    positions: Tensor | None = None,
    chrg: Tensor | float | int = _UNSET,
    spin: Tensor | float | int | None = _UNSET,
    field: Tensor | None = _UNSET,
    field_grad: Tensor | None = _UNSET,
    *,
    strict: bool | None = None,
) -> Result:
    """Evaluate a conformer batch using mapped implicit differentiation.

    Can be called either with a :class:`PreparedConvergedRootBatch` (optionally
    supplying transformed ``positions``, ``chrg``, ``field``, or ``field_grad``
    at the same primal point) or directly with ``(system, positions, ...)``.
    Valid lanes are evaluated simultaneously via ``torch.func.vmap`` over
    :func:`implicit_root` and functional SCF finalization. Invalid lanes in
    non-strict mode are isolated from :func:`implicit_root` and receive
    ``NaN`` physical observables without corrupting valid lanes.
    """
    if isinstance(batch_or_system, PreparedConvergedRootBatch):
        prepared = batch_or_system
        system = prepared.system
        sys_cfg = (
            system.core.config
            if isinstance(system, PaddedExactSystemCore)
            else system.config
        )
        effective_strict = (
            bool(sys_cfg.scf.force_convergence)
            if strict is None
            else bool(strict)
        )
        if (
            positions is not None
            or chrg is not _UNSET
            or spin is not _UNSET
            or field is not _UNSET
            or field_grad is not _UNSET
        ):
            eval_positions = (
                prepared.positions if positions is None else positions
            )
            lanes = _refresh_prepared_batch_inputs(
                prepared,
                eval_positions,
                chrg,
                spin,
                field,
                field_grad,
                strict_mode=effective_strict,
            )
        else:
            eval_positions = prepared.positions
            lanes = prepared.lanes
            if effective_strict and not prepared.all_roots_valid:
                for lane in lanes:
                    if not lane.root_valid:
                        _raise_strict_lane_failure(
                            lane.lane_index,
                            lane.failure_code,
                            lane.failure_reason or "Invalid physical root.",
                        )
        representation = prepared.representation
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
        )
        system = prepared.system
        eval_positions = prepared.positions
        lanes = prepared.lanes
        representation = prepared.representation
    else:
        raise TypeError(
            "Expected PreparedConvergedRootBatch, System, or PaddedExactSystemCore as first argument."
        )

    batch_size = len(lanes)
    valid_indices = [idx for idx, lane in enumerate(lanes) if lane.root_valid]

    lane_classicals = [
        _evaluate_lane_classical(system, lane.positions, lane.charge)
        for lane in lanes
    ]
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
                else torch.full_like(
                    lane_classicals[idx][term_idx][1], float("nan")
                )
                + 0.0 * lanes[idx].positions.sum()
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

    if valid_indices:
        valid_lanes = [lanes[idx] for idx in valid_indices]
        adapter = valid_lanes[0].adapter
        layout = valid_lanes[0].potential_layout
        param_count = len(adapter.parameter_paths)
        pot_state_proj: Callable[[Tensor], Tensor] | None = None
        if representation == labels.SCP_MODE_POTENTIAL and (
            bool((adapter.template.numbers == 0).any())
            or bool((~adapter.template.orbital_mask).any())
        ):
            assert layout is not None
            pad_state, _ = _padded_potential_projections(
                adapter.template, layout
            )
            pot_state_proj = lambda val, _ps=pad_state, _lay=layout: _ps(
                _project_potential_padding(val, _lay)
            )
        raw_map: Callable[..., Tensor] = (
            (lambda x, *p, _sp=pot_state_proj: _sp(adapter(x, *p)))
            if pot_state_proj is not None
            else adapter
        )

        roots = torch.stack([lane.root_x for lane in valid_lanes], dim=0)
        iterations_v = torch.stack(
            [lane.iterations for lane in valid_lanes], dim=0
        )
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
            torch.stack([lane.parameters[p_idx] for lane in valid_lanes], dim=0)
            for p_idx in range(param_count)
        )

        has_traj = all(
            lane.trajectory_status is not None for lane in valid_lanes
        )
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
                len(valid_lanes), dtype=torch.bool, device=eval_positions.device
            )
            int_zeros = torch.zeros(
                len(valid_lanes),
                dtype=torch.int64,
                device=eval_positions.device,
            )
            traj_fields = (
                bool_zeros,
                bool_zeros,
                bool_zeros,
                bool_zeros,
                int_zeros,
            )

        def _evaluate_valid_lane(
            root: Tensor,
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
            attached = implicit_root(root, *params, raw_map=raw_map)
            mapped_result = adapter.evaluate(attached, *params)
            if pot_state_proj is not None:
                mapped_result = replace(
                    mapped_result, mapped_x=pot_state_proj(mapped_result.mapped_x)
                )
            traj_status = (
                SCFTrajectoryStatus(
                    t_fconv, t_ecount, t_itemp, t_tmany, t_ffail
                )
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

        in_dims = (0,) * (11 + param_count)

        @torch._dynamo.disable
        def _run_mapped_lanes(*args: Tensor):
            return vmap(_evaluate_valid_lane, in_dims=in_dims)(*args)

        valid_outputs = _run_mapped_lanes(
            roots,
            iterations_v,
            converged_v,
            residuals_v,
            damp_values_v,
            damp_switched_v,
            *traj_fields,
            *stacked_params,
        )
    else:
        valid_outputs = None

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
                    nan_anchor = 0.0 * lanes[idx].positions.sum() + float("nan")
                    items.append(
                        nan_anchor.expand(shape).clone()
                        if shape
                        else nan_anchor
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
