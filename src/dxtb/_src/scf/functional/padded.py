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
"""Padded fixed-System SCF input preparation and spectral evaluation."""

from __future__ import annotations

from dataclasses import replace

import torch
from tad_mctc import storch
from tad_mctc.tree import Node, child, context
from tad_mctc.units.energy import KELVIN2AU

from dxtb import labels
from dxtb._src.calculators.padded_integrals import (
    PaddedIntegralEvaluation,
    evaluate_padded_integrals,
)
from dxtb._src.calculators.padding import PaddedExactSystemCore
from dxtb._src.calculators.result import Result
from dxtb._src.components.interactions.coulomb.secondorder import (
    ES2CoreSetup,
    build_es2_coulomb,
)
from dxtb._src.components.interactions.coulomb.thirdorder import ES3CoreSetup
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf.functional.core import (
    CoreIndexData,
    FunctionalInteractionEntry,
    FunctionalInteractionSet,
)
from dxtb._src.scf.functional.data import (
    ChargeLayout,
    FermiSettings,
    SCFInputs,
    SCFSpectralResult,
    _single_system_layout,
)
from dxtb._src.scf.functional.eager import (
    _solve_representation,
    public_scf_status,
)
from dxtb._src.scf.functional.map import evaluate_hamiltonian
from dxtb._src.typing import Tensor
from dxtb._src.utils import wrap_gather
from dxtb._src.wavefunction import filling
from dxtb._src.wavefunction.filling import _aufbau_occupation

__all__ = [
    "PADDED_ORBITAL_ENERGY",
    "PaddedCoreIndexData",
    "PaddedSCFSpectralResult",
    "evaluate_padded_scf_spectrum",
    "evaluate_padded_fixed_trip",
    "evaluate_padded_spectrum",
    "evaluate_padded_systems",
    "decode_padded_result",
    "setup_padded_scf_inputs",
    "validate_padded_charge_spin",
]

# Accepted GFN1 core levels in the fixtures remain far below this fixed,
# parameter-independent dummy level. It is only assigned to inactive AO
# diagonal entries and cannot couple to physical orbitals.
PADDED_ORBITAL_ENERGY = 1.0e4


class PaddedCoreIndexData(CoreIndexData):
    """Registered index maps with static target extents for padded sums."""

    atom_count: int = context()
    shell_count: int = context()
    valid_shells: Tensor = child(keep_dtype=True)
    valid_orbitals: Tensor = child(keep_dtype=True)

    def _validate(self) -> None:
        if self.valid_shells.shape != self.shells_to_atom.shape:
            raise ValueError("Padded shell validity does not match its map.")
        if self.valid_orbitals.shape != self.orbitals_to_shell.shape:
            raise ValueError("Padded orbital validity does not match its map.")
        for name, mask in (
            ("shell", self.valid_shells),
            ("orbital", self.valid_orbitals),
        ):
            if mask.dtype != torch.bool:
                raise TypeError(f"Padded {name} validity must be Boolean.")
            if mask.device != self.shells_to_atom.device:
                raise ValueError(
                    f"Padded {name} validity is on the wrong device."
                )

    def spread_atom_to_shell(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        result = super().spread_atom_to_shell(values, dim=dim, extra=extra)
        return torch.where(self._shell_mask_for(result, dim), result, 0.0)

    def spread_shell_to_orbital(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        result = super().spread_shell_to_orbital(values, dim=dim, extra=extra)
        return torch.where(self._orbital_mask_for(result, dim), result, 0.0)

    def _shell_mask_for(self, values: Tensor, dim: int | tuple[int, int]):
        return self._broadcast_mask(self.valid_shells, values, dim)

    def _orbital_mask_for(self, values: Tensor, dim: int | tuple[int, int]):
        return self._broadcast_mask(self.valid_orbitals, values, dim)

    @staticmethod
    def _broadcast_mask(
        mask: Tensor, values: Tensor, dim: int | tuple[int, int]
    ) -> Tensor:
        dims = (dim,) if isinstance(dim, int) else dim
        result = torch.ones_like(values, dtype=torch.bool)
        for axis in dims:
            axis = axis % values.ndim
            shape = [1] * values.ndim
            shape[axis] = mask.shape[0]
            result = result & mask.reshape(shape)
        return result

    def reduce_orbital_to_shell(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        from dxtb._src.utils import wrap_scatter_reduce

        return wrap_scatter_reduce(
            values,
            dim,
            self.orbitals_to_shell,
            reduce,
            extra=extra,
            output_size=(
                (self.shell_count, self.shell_count)
                if isinstance(dim, tuple)
                else self.shell_count
            ),
        )

    def reduce_shell_to_atom(
        self,
        values: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        from dxtb._src.utils import wrap_scatter_reduce

        return wrap_scatter_reduce(
            values,
            dim,
            self.shells_to_atom,
            reduce,
            extra=extra,
            output_size=(
                (self.atom_count, self.atom_count)
                if isinstance(dim, tuple)
                else self.atom_count
            ),
        )


class PaddedSCFSpectralResult(Node):
    """Registered tensor result of one padded spectral evaluation."""

    hamiltonian: Tensor = child()
    density: Tensor = child()
    emo: Tensor = child()
    coefficients: Tensor = child()
    occupation: Tensor = child()
    charges: Tensor = child()
    orbital_energy: Tensor = child()
    n0: Tensor = child()
    nel: Tensor = child()
    orbital_mask: Tensor = child(keep_dtype=True)
    fermi_converged: Tensor | None = child()
    fermi_electron_count_ok: Tensor | None = child()
    fermi_too_many_electrons: Tensor | None = child()
    fermi_invalid_temperature: Tensor | None = child()
    fermi_residual: Tensor | None = child()


def setup_padded_scf_inputs(
    padded_core: PaddedExactSystemCore,
    integrals: PaddedIntegralEvaluation,
    positions: Tensor,
    charge: Tensor,
    spin: Tensor | None = None,
) -> SCFInputs:
    """Prepare mask-aware SCF and interaction inputs without a System."""
    core = padded_core.core
    if core.h0_setup is None or core.integral is None:
        raise ValueError(
            "Padded SCF input preparation requires H0/integral setup."
        )
    if core.h0_setup.method != "gfn1":
        raise TypeError(
            "Padded SCF input preparation currently supports GFN1 only."
        )
    if core.integral.driver_type != labels.INTDRIVER_PYTORCH:
        raise TypeError(
            "Padded SCF input preparation requires PyTorch integrals."
        )
    hcore = integrals.hcore
    overlap = integrals.overlap
    validity = padded_core.validity
    orbital_mask = validity.orbitals
    nao = orbital_mask.shape[-1]
    if hcore.shape != (nao, nao) or overlap.shape != hcore.shape:
        raise ValueError(
            "Padded integral matrices do not match orbital validity."
        )
    if orbital_mask.dtype != torch.bool or orbital_mask.device != hcore.device:
        raise ValueError("Padded orbital validity has an invalid dtype/device.")
    if (
        charge.ndim != 0
        or charge.dtype != hcore.dtype
        or charge.device != hcore.device
    ):
        raise ValueError(
            "Padded charge must be a scalar matching integral dtype/device."
        )
    if spin is not None and (
        spin.ndim != 0
        or spin.dtype != hcore.dtype
        or spin.device != hcore.device
    ):
        raise ValueError(
            "Padded spin must be a scalar matching integral dtype/device."
        )

    backend = core.integral.backend
    if backend is None:
        raise ValueError(
            "Padded PyTorch integral setup is missing its backend."
        )
    index = PaddedCoreIndexData(
        shells_to_atom=backend.index.shells_to_atom,
        orbitals_to_shell=backend.index.orbitals_to_shell,
        atom_count=validity.atoms.shape[-1],
        shell_count=validity.shells.shape[-1],
        valid_shells=validity.shells,
        valid_orbitals=validity.orbitals,
    )

    h0_index = core.h0_setup.index
    shell_refs = wrap_gather(
        core.h0_setup.refocc, -1, h0_index.shells_to_ushell
    )
    shell_refs = torch.where(validity.shells, shell_refs, 0.0)
    orbital_refs = wrap_gather(shell_refs, -1, backend.index.orbitals_to_shell)
    shell_angular = wrap_gather(
        h0_index.unique_angular, -1, h0_index.shells_to_ushell
    )
    shell_width = (2 * shell_angular + 1).to(dtype=hcore.dtype)
    orbital_width = wrap_gather(
        shell_width, -1, backend.index.orbitals_to_shell
    )
    n0 = torch.where(
        orbital_mask,
        torch.where(
            orbital_width > 0, orbital_refs / orbital_width.clamp_min(1), 0
        ),
        0.0,
    )

    total_electrons = shell_refs.sum() - charge
    total_electrons = total_electrons.reshape(1)
    spin_value = None if spin is None else spin.reshape(1)
    nel = filling.get_alpha_beta_occupation(
        total_electrons, spin_value, validate=False
    )
    occupation = _aufbau_occupation(
        nel.unsqueeze(-1),
        nao,
        orbital_mask.unsqueeze(0),
        hcore.dtype,
        hcore.device,
    )

    matrix_mask = orbital_mask.unsqueeze(-1) & orbital_mask.unsqueeze(-2)
    safe_overlap = torch.where(matrix_mask, overlap, torch.zeros_like(overlap))
    safe_overlap = safe_overlap + torch.diag_embed(
        (~orbital_mask).to(hcore.dtype)
    )
    inverse_cholesky = storch.inv_cholesky_factor(safe_overlap, is_posdef=True)
    padded_hcore = torch.where(matrix_mask, hcore, torch.zeros_like(hcore))
    padded_hcore = padded_hcore + torch.diag_embed(
        (~orbital_mask).to(hcore.dtype) * PADDED_ORBITAL_ENERGY
    )
    padded_integrals = IntegralMatrices(
        hcore=padded_hcore,
        overlap=safe_overlap,
        dipole=getattr(integrals, "dipole", None),
        quadrupole=getattr(integrals, "quadrupole", None),
    )

    fermi_config = core.config.scf.fermi
    kt_value = fermi_config.etemp * KELVIN2AU
    if kt_value == 0.0:
        fermi_settings = None
    else:
        if fermi_config.thresh is None:
            eps = torch.finfo(hcore.dtype).eps
            threshold = hcore.new_tensor(min(eps**0.5, 1.0e5 * eps, 1.0e-4))
        else:
            threshold = torch.as_tensor(
                fermi_config.thresh, device=hcore.device, dtype=hcore.dtype
            )
        fermi_settings = FermiSettings(
            kt=hcore.new_tensor(kt_value),
            smear=kt_value >= 3.0e-7,
            maxiter=fermi_config.maxiter,
            threshold=threshold,
            diff_order=fermi_config.diff_order,
            partition=fermi_config.partition,
        )

    charge_layout: ChargeLayout = _single_system_layout(
        core.numbers, padded_integrals
    )
    interactions = _prepare_padded_interactions(padded_core, positions)
    return SCFInputs(
        numbers=core.numbers,
        index=index,
        integrals=padded_integrals,
        n0=n0,
        nel=nel,
        occupation=occupation,
        interactions=interactions,
        charge_layout=charge_layout,
        inverse_cholesky=inverse_cholesky,
        orbital_mask=orbital_mask,
        fermi_settings=fermi_settings,
    )


def _prepare_padded_interactions(
    padded_core: PaddedExactSystemCore,
    positions: Tensor,
) -> FunctionalInteractionSet:
    """Prepare exact supported interaction descriptors from captured setup.

    ES2's geometry-dependent matrix is built at the numerical call boundary.
    """
    entries: list[FunctionalInteractionEntry] = []
    for descriptor in padded_core.core.interaction_terms:
        if descriptor.kind == "es2" and isinstance(
            descriptor.setup, ES2CoreSetup
        ):
            matrix = build_es2_coulomb(descriptor.setup, positions)
            entries.append(
                FunctionalInteractionEntry(
                    "es2",
                    (("mat", matrix),),
                    (("shell_resolved", descriptor.setup.shell_resolved),),
                )
            )
        elif descriptor.kind == "es3" and isinstance(
            descriptor.setup, ES3CoreSetup
        ):
            entries.append(
                FunctionalInteractionEntry(
                    "es3",
                    (("hd", descriptor.setup.hubbard_derivs),),
                    (("shell_resolved", descriptor.setup.shell_resolved),),
                )
            )
        else:
            raise TypeError(
                "Padded fixed-trip SCF does not support captured interaction "
                f"'{descriptor.kind}'."
            )
    return FunctionalInteractionSet(tuple(entries))


def validate_padded_charge_spin(
    padded_core: PaddedExactSystemCore,
    charge: Tensor,
    spin: Tensor | None = None,
) -> Tensor:
    """Eagerly validate one source's physical electron/spin configuration.

    Call this for each independent source before stacking and mapping. The
    transformed lane setup uses the same occupation arithmetic with eager
    value checks disabled because those checks cannot run inside ``vmap``.
    """
    core = padded_core.core
    if core.h0_setup is None:
        raise ValueError("Padded SCF input preparation requires H0 setup.")
    if (
        charge.ndim != 0
        or charge.device != core.numbers.device
        or charge.dtype != core.effective_dtype
    ):
        raise ValueError(
            "Padded charge must be a scalar matching core dtype/device."
        )
    if spin is not None and (
        spin.ndim != 0
        or spin.device != core.numbers.device
        or spin.dtype != core.effective_dtype
    ):
        raise ValueError(
            "Padded spin must be a scalar matching core dtype/device."
        )
    shell_refs = wrap_gather(
        core.h0_setup.refocc, -1, core.h0_setup.index.shells_to_ushell
    )
    shell_refs = torch.where(padded_core.validity.shells, shell_refs, 0.0)
    total_electrons = (shell_refs.sum() - charge).reshape(1)
    spin_value = None if spin is None else spin.reshape(1)
    return filling.get_alpha_beta_occupation(
        total_electrons, spin_value, validate=True
    )


def evaluate_padded_spectrum(inputs: SCFInputs) -> PaddedSCFSpectralResult:
    """Evaluate the existing spectral primitive and return a registered Node."""
    spectral: SCFSpectralResult = evaluate_hamiltonian(
        inputs.integrals.hcore, inputs
    )
    status = spectral.fermi_status
    return PaddedSCFSpectralResult(
        hamiltonian=spectral.hamiltonian,
        density=spectral.density,
        emo=spectral.emo,
        coefficients=spectral.coefficients,
        occupation=spectral.occupation,
        charges=spectral.charges.mono,
        orbital_energy=spectral.orbital_energy,
        fermi_converged=None if status is None else status.converged,
        fermi_electron_count_ok=(
            None if status is None else status.electron_count_ok
        ),
        fermi_too_many_electrons=(
            None if status is None else status.too_many_electrons
        ),
        fermi_invalid_temperature=(
            None if status is None else status.invalid_temperature
        ),
        fermi_residual=None if status is None else status.residual,
        n0=inputs.n0,
        nel=inputs.nel,
        orbital_mask=inputs.orbital_mask,
    )


def evaluate_padded_scf_spectrum(
    padded_core: PaddedExactSystemCore,
    positions: Tensor,
    charge: Tensor,
    spin: Tensor | None = None,
) -> PaddedSCFSpectralResult:
    """Evaluate mapped padded H0 integrals and the shared spectral primitive."""
    integrals = evaluate_padded_integrals(padded_core, positions, charge)
    inputs = setup_padded_scf_inputs(
        padded_core, integrals, positions, charge, spin
    )
    return evaluate_padded_spectrum(inputs)


def _padded_initial_guess(
    padded_core: PaddedExactSystemCore,
    positions: Tensor,
    charge: Tensor,
    inputs: SCFInputs,
) -> Tensor:
    """Construct the production EEQ/SAD guess using padded index maps."""
    from dxtb import labels
    from dxtb._src.scf.guess import get_eeq_guess

    kind = padded_core.core.config.scf.guess
    if kind == labels.GUESS_SAD:
        atomic = torch.zeros_like(
            padded_core.core.numbers, dtype=positions.dtype
        )
    elif kind == labels.GUESS_EEQ:
        atomic = get_eeq_guess(padded_core.core.numbers, positions, charge)
    else:
        raise ValueError(f"Unsupported padded SCF guess '{kind}'.")

    shell_count = inputs.index.reduce_shell_to_atom(
        padded_core.validity.shells.to(dtype=positions.dtype)
    )
    shell_denominator = inputs.index.spread_atom_to_shell(shell_count)
    shell_values = torch.where(
        padded_core.validity.shells,
        inputs.index.spread_atom_to_shell(atomic)
        / shell_denominator.clamp_min(1.0),
        0.0,
    )
    orbital_count = inputs.index.reduce_orbital_to_shell(
        padded_core.validity.orbitals.to(dtype=positions.dtype)
    )
    orbital_denominator = inputs.index.spread_shell_to_orbital(orbital_count)
    return torch.where(
        padded_core.validity.orbitals,
        inputs.index.spread_shell_to_orbital(shell_values)
        / orbital_denominator.clamp_min(1.0),
        0.0,
    )


def evaluate_padded_fixed_trip(
    padded_core: PaddedExactSystemCore,
    positions: Tensor,
    charge: Tensor,
    spin: Tensor,
) -> Result:
    """Evaluate one padded GFN1 System using the shared fixed-trip solver.

    The function consumes only registered exact-core setup, explicit call
    tensors and the existing functional solver. It does not construct a
    mutable System, IndexHelper, InteractionList or persistent cache.
    """
    from dxtb import labels
    from dxtb._src.calculators.singlepoint import _evaluate_classical_descriptor
    from dxtb._src.scf.functional.eager import public_scf_status

    core = padded_core.core
    if core.config.method != labels.GFN1_XTB:
        raise TypeError(
            "Padded fixed-trip evaluation currently supports GFN1 only."
        )
    if core.config.scf.scf_mode != labels.SCF_MODE_FULL:
        raise TypeError("Padded fixed-trip evaluation requires FULL SCF mode.")
    if not core.config.scf.requires_iterations:
        raise TypeError(
            "Padded fixed-trip evaluation requires an iterative method."
        )
    if core.config.scf.maxiter < 0:
        raise ValueError("Padded fixed-trip evaluation requires maxiter >= 0.")

    integrals = evaluate_padded_integrals(padded_core, positions, charge)
    inputs = setup_padded_scf_inputs(
        padded_core, integrals, positions, charge, spin
    )
    # The input-preparation spectrum carries the same orbital seed and filling
    # contract as the ordinary fixed-System path. The shared solver owns every
    # subsequent map, mixer transition, residual and finalization.
    guess = _padded_initial_guess(padded_core, positions, charge, inputs)
    solved = _solve_representation(guess, inputs, core.config.scf, eager=False)

    classical = tuple(
        (
            descriptor.label,
            _evaluate_classical_descriptor(descriptor, positions, charge),
        )
        for descriptor in core.classical_terms
    )
    classical_total = (
        torch.stack(tuple(value for _, value in classical)).sum(0)
        if classical
        else torch.zeros_like(core.numbers, dtype=positions.dtype)
    )
    status = public_scf_status(
        solved.trajectory_status,
        solved.final_evaluation.fermi_status,
        like=solved.converged,
    )
    charges = solved.final_charges
    potential = solved.final_potential
    fenergy = solved.fenergy
    return Result.snapshot(
        energy=classical_total + solved.energy + fenergy,
        scf=solved.energy,
        classical=classical,
        fenergy=fenergy,
        iterations=solved.iterations,
        charges=charges,
        density=solved.density,
        coefficients=solved.coefficients,
        emo=solved.emo,
        occupation=solved.occupation,
        potential=potential,
        hamiltonian=solved.hamiltonian,
        overlap=integrals.overlap,
        hcore=integrals.hcore,
        dipole_integrals=integrals.dipole,
        quadrupole_integrals=integrals.quadrupole,
        overlap_norm=integrals.overlap_norm,
        converged=solved.converged,
        residual=solved.residual,
        status=status,
    )


def evaluate_padded_systems(
    padded_cores: tuple[PaddedExactSystemCore, ...],
    bucket,
    positions: tuple[Tensor, ...],
    charges: tuple[Tensor, ...],
    spins: tuple[Tensor, ...],
) -> tuple[Result, tuple[Result, ...]]:
    """Map a bucket-ordered core sequence and decode physical Results.

    The core and explicit-input tuples must follow ``bucket.indices`` order.
    Source indices identify outputs but are not required to be contiguous.
    """
    from tad_mctc.tree import stack

    count = len(padded_cores)
    if (
        count == 0
        or len(positions) != count
        or len(charges) != count
        or len(spins) != count
    ):
        raise ValueError(
            "Padded Systems and explicit inputs must have equal nonzero lengths."
        )
    if len(bucket.indices) != count:
        raise ValueError(
            "Padded lane count does not match the canonical bucket."
        )
    for core, position, charge, spin in zip(
        padded_cores, positions, charges, spins
    ):
        validate_padded_charge_spin(core, charge, spin)
        if position.shape != (core.core.numbers.numel(), 3):
            raise ValueError("Padded positions do not match their exact core.")
    stacked_core = stack(padded_cores)
    mapped = torch.func.vmap(evaluate_padded_fixed_trip)(
        stacked_core,
        torch.stack(positions),
        torch.stack(charges),
        torch.stack(spins),
    )
    decoded = tuple(
        decode_padded_result(
            mapped, bucket, source_index, lane_index=lane_index
        )
        for lane_index, source_index in enumerate(bucket.indices)
    )
    return mapped, decoded


def decode_padded_result(
    result: Result,
    bucket,
    source_index: int,
    *,
    lane_index: int | None = None,
) -> Result:
    """Restore one source's physical tensor extents and canonical ordering."""
    from dataclasses import replace as dataclass_replace

    maps = dict(bucket.source_maps)
    if source_index not in maps:
        raise ValueError(f"No canonical map for source System {source_index}.")
    mapping = maps[source_index]
    device = result.energy.device
    lane_index = source_index if lane_index is None else lane_index

    def vector(tensor: Tensor | None, indices: tuple[int, ...]):
        if tensor is None:
            return None
        lane = tensor.select(0, lane_index)
        axis = lane.ndim - 1
        idx = torch.tensor(indices, dtype=torch.int64, device=device)
        return lane.index_select(axis, idx)

    def atom_tensor(tensor: Tensor | None):
        if tensor is None:
            return None
        lane = tensor.select(0, lane_index)
        if lane.ndim >= 2 and lane.shape[-2] == bucket.atoms:
            indices = torch.tensor(
                mapping.atoms, dtype=torch.int64, device=device
            )
            return lane.index_select(-2, indices)
        if lane.ndim >= 1 and lane.shape[-1] == bucket.atoms:
            indices = torch.tensor(
                mapping.atoms, dtype=torch.int64, device=device
            )
            return lane.index_select(-1, indices)
        return lane

    def matrix(tensor: Tensor | None, axes: tuple[int, int] = (-2, -1)):
        if tensor is None:
            return None
        lane = tensor.select(0, lane_index)
        idx = torch.tensor(mapping.orbitals, dtype=torch.int64, device=device)
        return lane.index_select(axes[0], idx).index_select(axes[1], idx)

    def coefficients(tensor: Tensor | None):
        if tensor is None:
            return None
        lane = tensor.select(0, lane_index)
        rows = torch.tensor(mapping.orbitals, dtype=torch.int64, device=device)
        columns = torch.arange(len(mapping.orbitals), device=device)
        return lane.index_select(-2, rows).index_select(-1, columns)

    atom = tuple(mapping.atoms)
    orbital = tuple(mapping.orbitals)
    molecular_orbitals = tuple(range(len(mapping.orbitals)))
    charges = result.charges
    if charges is not None:
        charges = dataclass_replace(
            charges,
            mono=vector(charges.mono, orbital),
            dipole=atom_tensor(charges.dipole),
            quadrupole=atom_tensor(charges.quadrupole),
        )
    potential = result.potential
    if potential is not None:
        potential = dataclass_replace(
            potential,
            mono=vector(potential.mono, orbital),
            dipole=atom_tensor(potential.dipole),
            quadrupole=atom_tensor(potential.quadrupole),
        )
    classical = tuple(
        (name, vector(value, atom)) for name, value in result.classical
    )
    status = result.status
    if status is not None:
        status = dataclass_replace(
            status,
            trajectory_failure=status.trajectory_failure.select(0, lane_index),
            final_failure=status.final_failure.select(0, lane_index),
        )
    return dataclass_replace(
        result,
        energy=vector(result.energy, atom),
        scf=vector(result.scf, atom),
        classical=classical,
        fenergy=vector(result.fenergy, atom),
        iterations=result.iterations.select(0, lane_index),
        charges=charges,
        density=matrix(result.density),
        coefficients=coefficients(result.coefficients),
        emo=vector(result.emo, molecular_orbitals),
        occupation=vector(result.occupation, molecular_orbitals),
        potential=potential,
        hamiltonian=matrix(result.hamiltonian),
        overlap=matrix(result.overlap),
        hcore=matrix(result.hcore),
        dipole_integrals=(
            None
            if result.dipole_integrals is None
            else matrix(result.dipole_integrals)
        ),
        quadrupole_integrals=(
            None
            if result.quadrupole_integrals is None
            else matrix(result.quadrupole_integrals)
        ),
        overlap_norm=vector(result.overlap_norm, orbital),
        converged=(
            None
            if result.converged is None
            else result.converged.select(0, lane_index)
        ),
        residual=(
            None
            if result.residual is None
            else result.residual.select(0, lane_index)
        ),
        status=status,
    )
