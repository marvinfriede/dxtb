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
"""Physical electronic energy and free-energy observables for functional SCF."""

from __future__ import annotations

import torch
from tad_mctc.batch import real_atoms

from dxtb import labels
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction import filling

from .data import SCFCharges, SCFInputs, SCFSpectralResult
from .interactions import interaction_energy_from_charges as _core_interaction
from .spectral import (
    aufbau_free_energy_matrix,
    fermi_free_energy_matrix,
    fermi_free_energy_trace,
)

__all__ = [
    "electronic_energy_from_spectral",
    "free_energy_from_spectral",
    "interaction_energy_from_charges",
]


def interaction_energy_from_charges(
    charges: SCFCharges, inputs: SCFInputs
) -> Tensor:
    """Return atom-resolved interaction energy for explicit charges."""
    return _core_interaction(charges, inputs.interactions, inputs.index)


def electronic_energy_from_spectral(
    spectral: SCFSpectralResult, inputs: SCFInputs
) -> Tensor:
    """Return atom-resolved electronic energy excluding entropy."""
    h0_energy = inputs.index.reduce_orbital_to_atom(spectral.orbital_energy)
    return h0_energy + interaction_energy_from_charges(spectral.charges, inputs)


def free_energy_from_spectral(
    spectral: SCFSpectralResult, inputs: SCFInputs
) -> Tensor:
    """Return the atom-resolved entropy contribution for one spectrum."""
    if inputs.fermi_settings is None:
        return inputs.integrals.hcore.new_zeros(inputs.numbers.shape)
    settings = inputs.fermi_settings
    if settings.smear and spectral._fermi_state is None:
        raise RuntimeError(
            "A smeared spectral result must retain its authoritative "
            "Fermi state for free-energy evaluation."
        )
    if settings.smear and settings.partition == labels.FERMI_PARTITION_EQUAL:
        total = fermi_free_energy_trace(
            spectral.hamiltonian,
            inputs.nel,
            settings.kt,
            spectral._fermi_state,
            l_inv=inputs.inverse_cholesky,
            mask=inputs.orbital_mask,
            diff_steps=filling._diff_steps(settings.diff_order),
        ).unsqueeze(-1)
        real = real_atoms(inputs.numbers)
        count = real.count_nonzero(dim=-1).unsqueeze(-1)
        return torch.where(real, (total / count).expand_as(real), 0.0)
    if settings.smear:
        entropy_matrix = fermi_free_energy_matrix(
            spectral.hamiltonian,
            inputs.nel,
            settings.kt,
            spectral._fermi_state,
            l_inv=inputs.inverse_cholesky,
            mask=inputs.orbital_mask,
            diff_steps=filling._diff_steps(settings.diff_order),
        )
    else:
        entropy_matrix = aufbau_free_energy_matrix(
            spectral.hamiltonian,
            inputs.nel,
            settings.kt,
            l_inv=inputs.inverse_cholesky,
            mask=inputs.orbital_mask,
        )
    if settings.partition == labels.FERMI_PARTITION_EQUAL:
        real = real_atoms(inputs.numbers)
        count = real.count_nonzero(dim=-1).unsqueeze(-1)
        total = torch.diagonal(
            inputs.integrals.overlap @ entropy_matrix,
            dim1=-2,
            dim2=-1,
        ).sum(-1, keepdim=True)
        equal = total / count
        return torch.where(real, equal.expand_as(real), 0.0)
    if settings.partition == labels.FERMI_PARTITION_ATOMIC:
        orbital_populations = torch.diagonal(
            entropy_matrix @ inputs.integrals.overlap, dim1=-2, dim2=-1
        )
        return inputs.index.reduce_orbital_to_atom(orbital_populations)
    raise ValueError(
        f"Unknown Fermi free-energy partition {settings.partition}."
    )
