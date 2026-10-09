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
from tad_mctc.exceptions import DeviceError, DtypeError

from dxtb import labels
from dxtb._src import scf
from dxtb._src.calculators.model import System
from dxtb._src.calculators.result import Result
from dxtb._src.constants import defaults
from dxtb._src.components.interactions.list import InteractionListCache
from dxtb._src.components.interactions.coulomb.secondorder import (
    ES2Cache,
    build_es2_coulomb,
)
from dxtb._src.components.interactions.coulomb.thirdorder import ES3Cache
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.typing import Tensor

__all__ = ["singlepoint"]


def _interaction_data(
    system: System, positions: Tensor
) -> InteractionListCache:
    """Build call-local interaction data for one SCF evaluation.

    Built-in ES2 and ES3 use explicit System setup values. Other interactions
    keep their current cache APIs until their B6a packages, but returned data
    is local to this call.
    """
    data = InteractionListCache()
    for interaction in system.interactions.components:
        if interaction is system.es2_interaction:
            if system.es2_setup is None:
                raise RuntimeError("Single-system ES2 setup is missing.")
            matrix = build_es2_coulomb(system.es2_setup, positions)
            data[interaction.label] = ES2Cache(
                matrix, shell_resolved=system.es2_setup.shell_resolved
            )
        elif interaction is system.es3_interaction:
            if system.es3_setup is None:
                raise RuntimeError("Single-system ES3 setup is missing.")
            data[interaction.label] = ES3Cache(
                system.es3_setup.hubbard_derivs,
                shell_resolved=system.es3_setup.shell_resolved,
            )
        else:
            data[interaction.label] = interaction.get_cache(
                numbers=system.numbers,
                positions=positions,
                ihelp=system.ihelp,
            )
    return data


def _floating_setup_tensor(system: System) -> Tensor | None:
    """Return a gathered floating tensor that defines the System dtype."""
    if system.h0_setup is not None:
        return system.h0_setup.hscale

    setup = system.integral_setup
    if setup is None:
        return None
    if setup.pytorch is not None and setup.pytorch.alphas:
        return setup.pytorch.alphas[0]
    if setup.libcint is not None and setup.libcint.basis_setups:
        return setup.libcint.basis_setups[0].slater
    return None


def _call_scalar(
    name: str,
    value: Tensor | float | int,
    positions: Tensor,
) -> Tensor:
    """Validate a scalar call input without copying tensor inputs."""
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
            return value.unsqueeze(0)
        if value.ndim == 1 and value.shape[0] == 1:
            return value
        raise ValueError(
            f"Core singlepoint expects {name} to be a scalar or a "
            "one-element tensor."
        )

    return torch.tensor([value], dtype=positions.dtype, device=positions.device)


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
    system_device = system.numbers.device
    if positions.device != system_device:
        raise DeviceError(
            f"Device mismatch: System is on '{system_device}', but positions "
            f"are on '{positions.device}'."
        )

    floating_setup = _floating_setup_tensor(system)
    expected_dtype = (
        system.dd["dtype"] if floating_setup is None else floating_setup.dtype
    )
    if floating_setup is not None:
        if floating_setup.device != system_device:
            raise DeviceError(
                "System floating setup is on "
                f"'{floating_setup.device}', but System numbers are on "
                f"'{system_device}'."
            )
    if positions.dtype != expected_dtype:
        raise DtypeError(
            "Dtype mismatch: System floating setup is of type "
            f"'{expected_dtype}', but positions are of type "
            f"'{positions.dtype}'."
        )

    charge = _call_scalar("charge", chrg, positions)
    spin_tensor = (
        None if spin is None else _call_scalar("spin", spin, positions)
    )

    # B5 provides the pure evaluation boundary. B6a removes persistent
    # component state behind this boundary.
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
            iterations=torch.zeros(
                (), dtype=torch.int64, device=positions.device
            ),
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
    interaction_data = _interaction_data(system, positions)
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
            scf_results["iterations"],
            dtype=torch.int64,
            device=positions.device,
        ),
    )
