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

from typing import TYPE_CHECKING

import torch
from tad_mctc import storch

from dxtb import IndexHelper
from dxtb._src.components.interactions import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.constants import labels
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Any, Tensor
from dxtb._src.wavefunction import filling
from dxtb.config import ConfigSCF

from .base import SCFResult
from .guess import get_guess

if TYPE_CHECKING:
    from .functional.core import CoreIndexData

if TYPE_CHECKING:
    from .functional.eager import FunctionalCoreResult

__all__ = ["solve"]


def _use_functional_implicit_route(
    config: ConfigSCF,
    interactions: InteractionList,
    ihelp: IndexHelper,
    numbers: Tensor,
    integrals: IntegralMatrices,
    *,
    legacy_engine_overrides: bool,
) -> bool:
    """Resolve the opt-in implicit implementation without changing FULL.

    The legacy implicit engine historically owns Broyden/default-mixer calls
    and ignores ``ConfigSCF.mixer``. Keep those calls on that compatibility
    route. Only explicit Simple and Anderson settings select the new
    functional physical-root implementation.
    """
    from .functional.data import functional_scf_interactions_eligible

    eligible, _ = functional_scf_interactions_eligible(interactions)
    return (
        not legacy_engine_overrides
        and eligible
        and ihelp.batch_mode == 0
        and numbers.ndim == 1
        and config.fermi.etemp >= 0.0
        and config.fermi.maxiter >= 0
        and config.maxiter >= 0
        and integrals.hcore is not None
        and integrals.overlap is not None
        and config.mixer in (labels.MIXER_LINEAR, labels.MIXER_ANDERSON)
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
    *args: Any,
    core_index: CoreIndexData | None = None,
    **kwargs: Any,
) -> SCFResult | FunctionalCoreResult:
    """
    Obtain self-consistent solution for a given Hamiltonian.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    chrg : Tensor
        Total charge.
    interactions : InteractionList
        Collection of `Interation` objects.
    ihelp : IndexHelper
        Index helper object.
    config : ConfigSCF
        Configuration for the SCF calculation.
    integrals : Integrals
        Container for all integrals.
    args : Tuple
        Positional arguments to pass to the engine.
    kwargs : dict
        Keyword arguments to pass to the engine.

    Returns
    -------
    SCFResult | FunctionalCoreResult
        Legacy result mapping or tensor-status functional full-SCF result.
    """
    return _solve_impl(
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
        args,
        core_index,
        None,
        kwargs,
    )


def solve_with_core_index(
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
    *,
    core_index: CoreIndexData,
    core_index_owner: object,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> SCFResult | FunctionalCoreResult:
    """Run SCF with an internally trusted captured structural map."""
    return _solve_impl(
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
        args,
        core_index,
        core_index_owner,
        {} if kwargs is None else kwargs,
    )


def _solve_impl(
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
    args: tuple[Any, ...],
    core_index: CoreIndexData | None,
    core_index_owner: object | None,
    kwargs: dict[str, Any],
) -> SCFResult | FunctionalCoreResult:
    """Share numerical dispatch while preserving the public kwargs mapping."""
    n0, occupation, nab = get_refocc(refocc, chrg, spin, ihelp)

    if config.requires_iterations is False:
        # pylint: disable=import-outside-toplevel
        from .nonselfconsistent import solve_nonselfconsistent

        return solve_nonselfconsistent(
            numbers,
            interactions,
            cache,
            ihelp,
            config,
            integrals,
            n0,
            occupation,
            **kwargs,
        )

    charges = get_guess(numbers, positions, chrg, ihelp, config.guess)

    if not isinstance(config.scf_mode, int):
        raise ValueError(
            "SCF mode must be an integer within `solve`. This can only "
            "happen if you explicitly change the configuration object."
        )

    if config.scf_mode == labels.SCF_MODE_IMPLICIT:
        # The transform-compatible route has an explicit eligibility gate.
        # Unsupported legacy options, extensions and batches retain their
        # established xitorch compatibility implementation.
        functional_implicit = _use_functional_implicit_route(
            config,
            interactions,
            ihelp,
            numbers,
            integrals,
            legacy_engine_overrides=bool(args) or bool(kwargs),
        )
        if functional_implicit:
            from .functional.eager import (
                functional_core_result,
                run_prepared_functional_implicit,
            )

            result = run_prepared_functional_implicit(
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
                core_index=core_index,
                core_index_owner=core_index_owner,
            )
            return functional_core_result(result)

        # pylint: disable=import-outside-toplevel
        from .implicit import SelfConsistentFieldImplicit as SCF
    elif config.scf_mode == labels.SCF_MODE_FULL:
        # pylint: disable=import-outside-toplevel
        from dxtb import OutputHandler

        from .functional.eager import (
            apply_functional_full_policy,
            resolve_full_scf_route,
            run_prepared_functional_full,
        )

        legacy_engine_overrides = bool(args) or bool(kwargs)
        route = resolve_full_scf_route(
            config,
            interactions,
            ihelp,
            numbers,
            integrals,
            legacy_engine_overrides,
        )
        if route.kind == "functional":
            if route.warning is not None:
                OutputHandler.warn(route.warning)
            result = run_prepared_functional_full(
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
                route,
                core_index=core_index,
                core_index_owner=core_index_owner,
            )
            return apply_functional_full_policy(result, config, route)

        from .unrolling import SelfConsistentFieldFull as SCF
    elif config.scf_mode == labels.SCF_MODE_EXPERIMENTAL:
        # pylint: disable=import-outside-toplevel
        from .unrolling import SelfConsistentFieldSingleShot as SCF
    else:
        raise ValueError(f"Unknown SCF mode '{config.scf_mode}'.")

    return SCF(
        interactions,
        occupation,
        n0,
        *args,
        numbers=numbers,
        ihelp=ihelp,
        cache=cache,
        integrals=integrals,
        config=config,
        nel=nab,
        **kwargs,
    )(charges)


def get_refocc(
    refs: Tensor, chrg: Tensor, spin: Tensor | None, ihelp: IndexHelper
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

    # Obtain the reference occupation and total number of electrons
    # sum per shell (exact, as in tblite); padding shells have no orbitals
    shell_occ = ihelp.spread_ushell_to_shell(refs)
    shell_occ = torch.where(
        ihelp.orbitals_per_shell != 0, shell_occ, torch.zeros_like(shell_occ)
    )
    nel = torch.sum(shell_occ, -1, keepdim=True) - chrg

    # get alpha and beta electrons and occupation
    nab = filling.get_alpha_beta_occupation(nel, spin)
    occupation = filling.get_aufbau_occupation(
        torch.tensor(ihelp.nao, device=refs.device, dtype=torch.int64),
        nab,
    )

    return n0, occupation, nab
