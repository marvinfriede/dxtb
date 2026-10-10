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
"""Functional simple-mixing SCF solver, kept parallel to production SCF."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.components.interactions.container import Charges
from dxtb._src.constants import labels
from dxtb._src.typing import Tensor

from .data import (
    SCFCharges,
    SCFInputs,
    SCFMapResult,
    SCFPotential,
    unpack_charges,
)
from .map import potential_from_charges, scf_map

__all__ = [
    "SCFResidual",
    "SCFSolveResult",
    "SCFSolverSettings",
    "finalize_scf",
    "interaction_energy_from_charges",
    "setup_simple_solver_settings",
    "simple_mix",
    "solve_scf_simple",
]


@dataclass(frozen=True, eq=False)
class SCFSolverSettings:
    """Static root-finding settings for the E4.1 simple solver."""

    maxiter: int
    mix_guess: bool
    damp: float
    x_atol: float
    x_atol_max: float


@dataclass(frozen=True, eq=False)
class SCFResidual:
    """L2 and max-absolute norms of the raw physical residual."""

    l2: Tensor
    max_abs: Tensor

    def as_tensor(self) -> Tensor:
        """Return norms in the public residual order."""
        return torch.stack((self.l2, self.max_abs))


@dataclass(frozen=True, eq=False)
class SCFSolveResult:
    """Finalized physical result and tensor-valued solver status."""

    map_result: SCFMapResult
    final_charges: SCFCharges
    final_potential: SCFPotential
    energy: Tensor
    iterations: Tensor
    converged: Tensor
    residual: Tensor


@dataclass(frozen=True, eq=False)
class _SolverOutcome:
    """Transform-friendly outcome before interaction finalization."""

    map_result: SCFMapResult
    iterations: Tensor
    converged: Tensor
    residual: Tensor


def _resolve_tolerance(value: float, dtype: torch.dtype) -> float:
    """Match the legacy Simple mixer's first-call resolution downgrade."""
    resolution = torch.finfo(dtype).resolution
    return resolution * 50 if resolution > value else value


def setup_simple_solver_settings(
    config: ConfigSCF, dtype: torch.dtype
) -> SCFSolverSettings:
    """Resolve supported simple-solver settings without mutating config."""
    if config.maxiter < 0:
        raise ValueError("The functional SCF solver requires maxiter >= 0.")
    if config.mixer != labels.MIXER_LINEAR:
        raise NotImplementedError(
            "The E4.1 functional solver supports only simple linear mixing."
        )
    if config.damp_dynamic:
        raise NotImplementedError(
            "The E4.1 functional solver does not support dynamic damping."
        )

    return SCFSolverSettings(
        maxiter=config.maxiter,
        mix_guess=config.mix_guess,
        damp=config.damp,
        x_atol=_resolve_tolerance(config.x_atol, dtype),
        x_atol_max=_resolve_tolerance(config.x_atol_max, dtype),
    )


def simple_mix(old: Tensor, mapped: Tensor, damp: float) -> Tensor:
    """Apply one out-of-place linear mixing transition."""
    return old + (mapped - old) * damp


def _residual(delta: Tensor) -> SCFResidual:
    return SCFResidual(
        l2=torch.linalg.vector_norm(delta),
        max_abs=torch.amax(torch.abs(delta)),
    )


def _select_tensor(mask: Tensor, new: Tensor, old: Tensor) -> Tensor:
    return torch.where(mask, new, old)


def _select_map_result(
    mask: Tensor, new: SCFMapResult, old: SCFMapResult
) -> SCFMapResult:
    """Select map leaves without selecting Python structure dynamically."""

    def select_optional(
        new_value: Tensor | None, old_value: Tensor | None
    ) -> Tensor | None:
        if new_value is None and old_value is None:
            return None
        if new_value is None or old_value is None:
            raise ValueError("SCF map result structures must match.")
        return _select_tensor(mask, new_value, old_value)

    return SCFMapResult(
        mapped_x=_select_tensor(mask, new.mapped_x, old.mapped_x),
        charges=SCFCharges(
            mono=_select_tensor(mask, new.charges.mono, old.charges.mono),
            dipole=select_optional(new.charges.dipole, old.charges.dipole),
            quad=select_optional(new.charges.quad, old.charges.quad),
        ),
        driving_potential=SCFPotential(
            mono=select_optional(
                new.driving_potential.mono, old.driving_potential.mono
            ),
            dipole=select_optional(
                new.driving_potential.dipole, old.driving_potential.dipole
            ),
            quad=select_optional(
                new.driving_potential.quad, old.driving_potential.quad
            ),
        ),
        hamiltonian=_select_tensor(mask, new.hamiltonian, old.hamiltonian),
        density=_select_tensor(mask, new.density, old.density),
        emo=_select_tensor(mask, new.emo, old.emo),
        coefficients=_select_tensor(mask, new.coefficients, old.coefficients),
        occupation=_select_tensor(mask, new.occupation, old.occupation),
        orbital_energy=_select_tensor(
            mask, new.orbital_energy, old.orbital_energy
        ),
    )


def _solve_map(
    guess: Tensor,
    settings: SCFSolverSettings,
    map_fn: Callable[[Tensor], SCFMapResult],
) -> _SolverOutcome:
    """Run a fixed-trip simple-mixing solver around one raw map callback."""
    initial = map_fn(guess)
    initial_residual = _residual(initial.mapped_x - guess).as_tensor()
    iterations = torch.ones((), dtype=torch.int64, device=guess.device)
    active = torch.ones((), dtype=torch.bool, device=guess.device)

    if settings.maxiter == 0:
        return _SolverOutcome(
            map_result=initial,
            iterations=iterations,
            converged=torch.zeros_like(active),
            residual=initial_residual,
        )

    current_x = (
        simple_mix(guess, initial.mapped_x, settings.damp)
        if settings.mix_guess
        else initial.mapped_x
    )
    final_result = initial
    residual_value = initial_residual

    for _ in range(settings.maxiter):
        candidate = map_fn(current_x)
        raw_delta = candidate.mapped_x - current_x
        candidate_residual = _residual(raw_delta)
        is_converged = (candidate_residual.l2 < settings.x_atol) & (
            candidate_residual.max_abs < settings.x_atol_max
        )
        final_result = _select_map_result(active, candidate, final_result)
        residual_value = _select_tensor(
            active, candidate_residual.as_tensor(), residual_value
        )
        iterations = torch.where(active, iterations + 1, iterations)
        active_after_step = active & ~is_converged
        mixed_x = simple_mix(current_x, candidate.mapped_x, settings.damp)
        current_x = _select_tensor(active_after_step, mixed_x, current_x)
        active = active_after_step

    return _SolverOutcome(
        map_result=final_result,
        iterations=iterations,
        converged=~active,
        residual=residual_value,
    )


def interaction_energy_from_charges(
    charges: SCFCharges, inputs: SCFInputs
) -> Tensor:
    """Return atom-resolved interaction energy for explicit final charges."""
    local_charges = Charges(
        mono=charges.mono, dipole=charges.dipole, quad=charges.quad
    )
    return inputs.interactions.get_energy(
        local_charges, inputs.interaction_data, inputs.ihelp
    )


def finalize_scf(
    map_result: SCFMapResult,
    inputs: SCFInputs,
    iterations: Tensor,
    converged: Tensor,
    residual: Tensor,
) -> SCFSolveResult:
    """Finalize energy/potential from the selected raw map output charges."""
    final_charges = SCFCharges(
        mono=map_result.charges.mono,
        dipole=map_result.charges.dipole,
        quad=map_result.charges.quad,
    )
    final_potential = potential_from_charges(final_charges, inputs)
    h0_energy = inputs.ihelp.reduce_orbital_to_atom(map_result.orbital_energy)
    energy = h0_energy + interaction_energy_from_charges(final_charges, inputs)
    return SCFSolveResult(
        map_result=map_result,
        final_charges=final_charges,
        final_potential=final_potential,
        energy=energy,
        iterations=iterations,
        converged=converged,
        residual=residual,
    )


def solve_scf_simple(
    guess: Tensor, inputs: SCFInputs, settings: SCFSolverSettings
) -> SCFSolveResult:
    """Solve the single-System charge map with fixed-trip simple mixing."""
    outcome = _solve_map(guess, settings, lambda x: scf_map(x, inputs))
    return finalize_scf(
        outcome.map_result,
        inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
    )
