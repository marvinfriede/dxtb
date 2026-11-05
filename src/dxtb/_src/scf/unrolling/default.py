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
SCF Unrolling: Standard Variant
===============================

Standard implementation of single-system SCF iterations with unrolling or full
gradient tracking for the backward pass.
"""

from __future__ import annotations

from dxtb import OutputHandler
from dxtb._src.components.interactions import Charges, Potential
from dxtb._src.typing import Literal, Tensor, exceptions, overload

from .base import BaseTSCF

__all__ = ["SelfConsistentFieldFull"]


class SelfConsistentFieldFull(BaseTSCF):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    This SCF class uses a straightforward implementation of simple or Anderson
    mixing (taken from TBMaLT). Therefore, the gradient tracking is enabled
    for all iterations.
    """

    @overload
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[True] = True,
    ) -> Charges: ...

    @overload
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[False] = False,
    ) -> Charges | Potential | Tensor: ...

    def scf(
        self, guess: Tensor, return_charges: bool = True
    ) -> Charges | Potential | Tensor:
        # "SCF function"
        fcn = self._fcn

        maxiter = self.config.maxiter
        batched = self._data.ihelp.batch_mode

        # Evaluate initial guess outside of SCF loop to make maxiter=0 possible.
        q_new = fcn(guess)
        if OutputHandler.verbosity >= 3:  # pragma: no cover
            charges = self.converged_to_charges(q_new)
            energy = self.get_energy(charges)
            self._print(charges, energy)

        if maxiter == 0:
            if return_charges is True:
                return self.converged_to_charges(q_new)
            return q_new

        if batched != 0:
            raise NotImplementedError(
                "Batched SCF execution on SelfConsistentFieldFull is retired; "
                "use functional packed SCF or compose_legacy_systems."
            )

        # Also mix the initial guess with the first SCF iteration
        if self.config.mix_guess is True:
            q = self.mixer.iter(q_new, guess)
        else:
            q = q_new

        for _ in range(maxiter):
            q_new = fcn(q)

            # Important: Calculate energy with charges before mixing!
            if OutputHandler.verbosity >= 3:  # pragma: no cover
                charges = self.converged_to_charges(q_new)
                energy = self.get_energy(charges)
                self._print(charges, energy)

            q = self.mixer.iter(q_new, q)

            if self.mixer.converged:
                # Do not return mixed charges here, but from last SCF call!
                q_converged = q_new
                break

            if self.config.damp_dynamic is True:
                # Switch off damping if the norm of the difference is small
                if self.mixer.delta_norm < 0.1:
                    self.mixer.options["damp"] = (
                        self.config.damp_dynamic_factor
                    )

        else:
            msg = (
                f"SCF does not converge after {maxiter} cycles using "
                f"{self.mixer.label} mixing with a damping factor of "
                f"{self.mixer.options['damp']}."
            )
            if self.config.force_convergence is True:
                raise exceptions.SCFConvergenceError(msg)

            # only issue warning, return anyway
            OutputHandler.warn(msg, exceptions.SCFConvergenceWarning)

            # Do not return mixed charges here, but from last SCF call!
            q_converged = q_new

        if return_charges is True:
            return self.converged_to_charges(q_converged)
        return q_converged

