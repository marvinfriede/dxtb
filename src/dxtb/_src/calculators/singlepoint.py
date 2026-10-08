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
"""
Calculators: singlepoint core
=============================

Single-system evaluation from explicit System and call inputs.
"""

from __future__ import annotations

import torch

from dxtb import labels
from dxtb._src import scf
from dxtb._src.calculators.model import System
from dxtb._src.calculators.result import Result
from dxtb._src.constants import defaults
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.typing import Tensor

__all__ = ["singlepoint"]


def singlepoint(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
) -> Result:
    """Evaluate one System without Calculator or legacy integral state."""
    if system.numbers.ndim != 1:
        raise ValueError("Core singlepoint accepts one system at a time.")
    if positions.shape != (system.numbers.shape[0], 3):
        raise ValueError(
            "Core singlepoint expects positions with shape (nat, 3)."
        )

    charge = torch.atleast_1d(
        torch.as_tensor(
            chrg, dtype=positions.dtype, device=positions.device
        )
    )
    spin_tensor = (
        None
        if spin is None
        else torch.atleast_1d(
            torch.as_tensor(
                spin, dtype=positions.dtype, device=positions.device
            )
        )
    )

    if system.classicals.components:
        classical = system.classicals.get_energy(
            positions, system.classical_cache, charge=charge
        )
        classical_energy = torch.stack(tuple(classical.values())).sum(0)
    else:
        classical = {}
        classical_energy = positions.new_zeros(positions.shape[:-1])
    zero_energy = torch.zeros_like(classical_energy)

    if {"all", "scf"} & set(system.config.exclude):
        return Result.snapshot(
            energy=classical_energy,
            scf=zero_energy,
            classical=tuple(classical.items()),
            fenergy=zero_energy,
            iterations=torch.zeros((), dtype=torch.int64, device=positions.device),
        )

    if system.h0_setup is None:
        raise NotImplementedError("Core Hamiltonian setup is missing.")
    if system.integral_setup is None:
        raise NotImplementedError(
            "Single-system integral setup is missing. Core evaluation does "
            "not support legacy batched Calculator setup."
        )
    if system.config.ints.level < labels.INTLEVEL_HCORE:
        raise NotImplementedError(
            "Core Hamiltonian missing. Skipping the Core Hamiltonian in "
            "the SCF is currently not supported. Please increase the "
            "integral level to at least '2'. Currently, the level is set "
            f"to '{system.config.ints.level}'."
        )

    matrices, refocc, overlap_norm = build_integral_matrices(
        system.integral_setup,
        system.h0_setup,
        positions,
        charge,
    )
    interaction_data = system.interactions.get_cache(
        numbers=system.numbers,
        positions=positions,
        ihelp=system.ihelp,
    )
    scf_results = scf.solve(
        system.numbers,
        positions,
        charge,
        spin_tensor,
        system.interactions,
        interaction_data,
        system.ihelp,
        system.config.scf,
        matrices,
        refocc,
    )

    total_energy = (
        classical_energy + scf_results["energy"] + scf_results["fenergy"]
    )
    return Result.snapshot(
        energy=total_energy,
        scf=scf_results["energy"],
        classical=tuple(classical.items()),
        fenergy=scf_results["fenergy"],
        charges=scf_results["charges"],
        density=scf_results["density"],
        coefficients=scf_results["coefficients"],
        emo=scf_results["emo"],
        occupation=scf_results["occupation"],
        potential=scf_results["potential"],
        hamiltonian=scf_results["hamiltonian"],
        overlap=matrices.overlap,
        hcore=matrices.hcore,
        dipole_integrals=matrices.dipole,
        quadrupole_integrals=matrices.quadrupole,
        overlap_norm=overlap_norm,
        iterations=torch.tensor(
            scf_results["iterations"], dtype=torch.int64, device=positions.device
        ),
    )
