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
Self-consistent field iteration
===============================

Provides implementation of self consistent field iterations for the xTB
Hamiltonian. The iterations are not like in ab initio calculations expressed in
the density matrix and the derivative of the energy w.r.t. to the density
matrix, i.e. the Hamiltonian, but the Mulliken populations (or partial charges)
of the respective orbitals as well as the derivative of the energy w.r.t. to
those populations, i.e. the potential vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from tad_mctc import storch

from dxtb import IndexHelper
from dxtb._src.components.interactions import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction import filling
from dxtb.config import ConfigSCF

from .guess import get_guess
from .result import SCFResult

if TYPE_CHECKING:
    from .eager import SCFCoreResult

__all__ = ["solve"]


def _implicit_differentiation_supported(
    config: ConfigSCF,
    interactions: InteractionList,
    ihelp: IndexHelper,
    numbers: Tensor,
    integrals: IntegralMatrices,
) -> bool:
    """Return whether Implicit differentiation supports this SCF.

    Public packed implicit calls retain their tensor-batched behavior through
    the map and response. This is separate from W2c9's internal
    converged-root conformer orchestration.
    """
    from .data import scf_interactions_eligible

    eligible, _ = scf_interactions_eligible(interactions)
    is_batch = ihelp.batch_mode != 0 or numbers.ndim != 1
    strategy = config.strategy
    if is_batch and (
        len(strategy.stages) != 1
        or not getattr(strategy.stages[0].converger, "supports_fixed_trip", True)
    ):
        return False
    return (
        eligible
        and config.fermi.etemp >= 0.0
        and config.fermi.maxiter >= 0
        and config.maxiter >= 0
        and integrals.hcore is not None
        and integrals.overlap is not None
    )


def solve(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    spin: Tensor | None,
    interactions: InteractionList,
    cache: InteractionListCache,
    ihelp: IndexHelper,
    config: ConfigSCF,
    integrals: IntegralMatrices,
    refocc: Tensor,
) -> SCFResult | SCFCoreResult:
    """
    Obtain self-consistent solution for a given Hamiltonian.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    chrg : Tensor
        Total charge.
    spin : Tensor | None
        Number of unpaired electrons.
    interactions : InteractionList
        Collection of ``Interaction`` objects.
    cache : InteractionListCache
        Call-local interaction data.
    ihelp : IndexHelper
        Canonical molecular index helper.
    config : ConfigSCF
        Configuration for the SCF calculation.
    integrals : IntegralMatrices
        Integral matrices (core Hamiltonian, overlap and multipoles).
    refocc : Tensor
        Reference shell occupations.

    Returns
    -------
    SCFResult | SCFCoreResult
        Non-SCF GFN0 result mapping or tensor-status iterative SCF result.
    """
    if not isinstance(ihelp, IndexHelper):
        raise TypeError("SCF solver requires an IndexHelper.")
    if config.fermi.etemp < 0.0:
        raise ValueError("Electronic Temperature must be non-negative.")
    if config.fermi.maxiter < 0:
        raise ValueError("Fermi maxiter must be non-negative.")
    if config.maxiter < 0:
        raise ValueError("The SCF solver requires maxiter >= 0.")

    if (
        config.requires_iterations is False
        or config.differentiation.value == "unrolled"
    ):
        from .eager import run_eager_scf

        eager_res = run_eager_scf(
            numbers,
            positions,
            chrg,
            spin,
            interactions,
            cache,
            ihelp,
            config,
            integrals,
            refocc,
        )
        if eager_res.scf is not None:
            return eager_res.scf
        assert eager_res.non_scf is not None
        return cast("SCFResult", eager_res.non_scf)

    if config.differentiation.value != "implicit":
        raise ValueError(
            f"Unknown differentiation policy '{config.differentiation}'."
        )

    from .data import _validate_scf_preamble, scf_interactions_eligible
    from .eager import run_prepared_implicit_scf, scf_core_result

    _validate_scf_preamble(
        numbers, ihelp, positions=positions, refocc=refocc, chrg=chrg, spin=spin
    )
    n0, occupation, nab = get_refocc(refocc, chrg, spin, ihelp)
    charges = get_guess(numbers, positions, chrg, ihelp, config.guess)
    is_public_batch = ihelp.batch_mode != 0 or numbers.ndim != 1
    route_name = (
        "batched implicit SCF"
        if is_public_batch
        else "single-System implicit SCF"
    )
    if not _implicit_differentiation_supported(
        config, interactions, ihelp, numbers, integrals
    ):
        eligible, names = scf_interactions_eligible(interactions)
        if not eligible:
            raise NotImplementedError(
                f"{route_name} does not support interaction "
                f"extension(s): {names}."
            )
        if integrals.hcore is None or integrals.overlap is None:
            raise ValueError(
                f"{route_name} requires hcore and overlap integrals."
            )
        raise ValueError(
            f"Unsupported {route_name} strategy "
            f"'{type(config.strategy.stages[0].converger).__name__}'."
        )

    result = run_prepared_implicit_scf(
        numbers,
        charges,
        n0,
        occupation,
        nab,
        interactions,
        cache,
        ihelp,
        config,
        integrals,
    )
    return scf_core_result(result)


def reference_electron_count(refs: Tensor, ihelp: IndexHelper) -> Tensor:
    """
    Number of electrons of the neutral reference occupation.

    Parameters
    ----------
    refs : Tensor
        Occupation from parametrization.
    ihelp : IndexHelper
        Helper for indexing.

    Returns
    -------
    Tensor
        Reference number of electrons (shape: [..., 1]).
    """
    # sum per shell (exact, as in tblite); padding shells have no orbitals
    shell_occ = ihelp.spread_ushell_to_shell(refs)
    shell_occ = torch.where(
        ihelp.orbitals_per_shell != 0, shell_occ, torch.zeros_like(shell_occ)
    )
    return torch.sum(shell_occ, -1, keepdim=True)


def get_refocc(
    refs: Tensor,
    chrg: Tensor,
    spin: Tensor | None,
    ihelp: IndexHelper,
    *,
    validate_spin: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Obtain reference occupation and total number of electrons.

    Parameters
    ----------
    refs : Tensor
        Occupation from parametrization.
    chrg : Tensor
        Total charge.
    spin : Tensor | None
        Number of unpaired electrons.
    ihelp : IndexHelper
        Helper for indexing.
    validate_spin : bool, optional
        Whether to perform eager value checks for spin/electron consistency.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Reference occupation, occupation, and number of alpha and beta
        electrons (shape: [..., 2]).
    """

    refocc = ihelp.spread_ushell_to_orbital(refs)
    orb_per_shell = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)

    n0 = torch.where(
        orb_per_shell != 0,
        storch.safe_divide(refocc, orb_per_shell.type(refocc.dtype)),
        torch.tensor(0, device=refs.device, dtype=refs.dtype),
    )

    ref_nel = reference_electron_count(refs, ihelp)
    chrg_f = (
        chrg.to(dtype=ref_nel.dtype)
        if isinstance(chrg, Tensor) and chrg.dtype != ref_nel.dtype
        else chrg
    )
    nel = ref_nel - chrg_f
    if nel.shape != ref_nel.shape:
        raise ValueError(
            f"SCF 'chrg' shape {tuple(getattr(chrg, 'shape', ()))} is "
            f"incompatible with electron count shape {tuple(ref_nel.shape)}."
        )

    # get alpha and beta electrons and occupation
    nab = filling.get_alpha_beta_occupation(nel, spin, validate=validate_spin)
    occupation = filling.get_aufbau_occupation(
        torch.tensor(ihelp.nao, device=refs.device, dtype=torch.int64),
        nab,
    )

    return n0, occupation, nab
