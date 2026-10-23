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
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf.functional.core import (
    CoreIndexData,
    FunctionalInteractionSet,
)
from dxtb._src.scf.functional.data import (
    ChargeLayout,
    FermiSettings,
    SCFInputs,
    SCFSpectralResult,
    _single_system_layout,
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
    "evaluate_padded_spectrum",
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
    charge: Tensor,
    spin: Tensor | None = None,
) -> SCFInputs:
    """Prepare mask-aware SCF values without a System or IndexHelper.

    This boundary prepares the core-Hamiltonian spectral input only. The next
    mixed-System SCF stage owns immutable interaction preparation and the
    fixed-trip controller.
    """
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
    return SCFInputs(
        numbers=core.numbers,
        index=index,
        integrals=padded_integrals,
        n0=n0,
        nel=nel,
        occupation=occupation,
        interactions=FunctionalInteractionSet(entries=()),
        charge_layout=charge_layout,
        inverse_cholesky=inverse_cholesky,
        orbital_mask=orbital_mask,
        fermi_settings=fermi_settings,
    )


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
    inputs = setup_padded_scf_inputs(padded_core, integrals, charge, spin)
    return evaluate_padded_spectrum(inputs)
