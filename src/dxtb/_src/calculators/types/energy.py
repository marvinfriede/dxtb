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
Calculators: Energy
===================

Base calculator for the energy calculation of an extended tight-binding model.
"""

from __future__ import annotations

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.io.checks import content_checks, shape_checks

from dxtb import OutputHandler, labels
from dxtb._src import scf
from dxtb._src.constants import defaults
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor

from ..result import Result
from .base import BaseCalculator, reject_removed_store_kwargs

__all__ = ["EnergyCalculator"]

class EnergyCalculator(BaseCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides the basic functionality for the extended tight-binding
    model. It provides methods for single point calculations, nuclear
    gradients, Hessians, molecular properties, and spectra.
    """

    # all from an SCF
    implemented_properties: list[str] = [
        "bond_orders",
        "energy",
        "coefficients",
        "charges",
        "density",
        "iterations",
        "mo_energies",
        "occupation",
        "potential",
    ]
    """Names of implemented methods of the Calculator."""

    __slots__ = [
        "numbers",
        "opts",
        "classicals",
        "interactions",
        "integrals",
        "ihelp",
    ]

    def singlepoint(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ):
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0 (also for batched calculations). May
            be fractional. At finite electronic temperature, the energy is
            differentiable with respect to it.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to 0.
        """
        reject_removed_store_kwargs(kwargs)

        # shape checks
        assert shape_checks(
            self.numbers,
            positions,
            allow_batched=True,
        )
        assert content_checks(
            self.numbers,
            positions,
            self.opts.max_element,
            allow_batched=True,
        )

        OutputHandler.write_stdout("Singlepoint ", v=3)

        is_batched = self.numbers.ndim == 2

        if is_batched:
            if isinstance(chrg, (float, int)):
                if chrg != defaults.CHRG:
                    raise ValueError(
                        "Cannot set charge for batched calculations with a "
                        "single float or integer. Please provide a 1D tensor "
                        "of charges."
                    )

                chrg = torch.tensor([chrg, chrg], **self.dd)

        _chrg: Tensor = torch.atleast_1d(any_to_tensor(chrg, **self.dd))
        if spin is not None:
            _spin = torch.atleast_1d(any_to_tensor(spin, **self.dd))
        else:
            _spin = None

        # Attempt reshaping to proper batch shape: (n,) -> (n, 1)
        if is_batched is True:
            if _chrg.ndim == 1 and _chrg.numel() != 1:
                _chrg = _chrg.view(-1, 1)
            if _spin is not None and _spin.ndim == 1 and _spin.numel() != 1:
                _spin = _spin.view(-1, 1)

        if not is_batched:
            return self._singlepoint_system(positions, _chrg, _spin, kwargs)

        classical: dict[str, Tensor] = {}
        total_energy = torch.zeros(positions.shape[:-1], **self.dd)

        ###########################
        # CLASSICAL CONTRIBUTIONS #
        ###########################

        if len(self.classicals.components) > 0:
            OutputHandler.write_stdout_nf(" - Classicals        ... ", v=3)
            timer.start("Classicals")

            ccaches = self.system.classical_cache
            classical = self.classicals.get_energy(
                positions, ccaches, charge=_chrg
            )
            total_energy = total_energy + torch.stack(
                list(classical.values())
            ).sum(0)

            timer.stop("Classicals")
            OutputHandler.write_stdout("done", v=3)

        if {"all", "scf"} & set(self.opts.exclude):
            result = Result.snapshot(
                energy=total_energy,
                scf=torch.zeros_like(total_energy),
                classical=tuple(classical.items()),
                fenergy=torch.zeros_like(total_energy),
                iterations=torch.zeros((), dtype=torch.int64, device=self.device),
            )
            self._ncalcs += 1
            return result

        #############
        # INTEGRALS #
        #############

        timer.start("Integrals")
        OutputHandler.write_stdout_nf(" - Integral matrices ... ", v=3)
        if self.system.h0_setup is None:
            raise NotImplementedError("Core Hamiltonian setup is missing.")

        # Batched Calculator calls remain on legacy mutable integral builders
        # until E6 introduces stacked-System evaluation.
        overlap_matrix = self.integrals.build_overlap(positions)
        dipole_matrix = None
        quadrupole_matrix = None
        if self.opts.ints.level >= labels.INTLEVEL_DIPOLE:
            dipole_matrix = self.integrals.build_dipole(positions)
        if self.opts.ints.level >= labels.INTLEVEL_QUADRUPOLE:
            quadrupole_matrix = self.integrals.build_quadrupole(positions)
            if self.integrals.dipole is not None:
                dipole_matrix = self.integrals.dipole.matrix
        hcore_matrix = self.integrals.build_hcore(
            positions,
            with_overlap=True,
            charge=_chrg,
        )
        if self.integrals.hcore is None:
            raise RuntimeError("Legacy H0 adapter is not initialized.")
        refocc = self.integrals.hcore.refocc
        intmats = IntegralMatrices(
            hcore=hcore_matrix,
            overlap=overlap_matrix,
            dipole=dipole_matrix,
            quadrupole=quadrupole_matrix,
        ).to(self.device)
        timer.stop("Integrals")
        OutputHandler.write_stdout("done", v=3)

        for key, matrix, integral in (
            ("write_overlap", intmats.overlap, self.integrals.overlap),
            ("write_dipole", intmats.dipole, self.integrals.dipole),
            ("write_quadrupole", intmats.quadrupole, self.integrals.quadrupole),
            ("write_hcore", intmats.hcore, self.integrals.hcore),
        ):
            path = kwargs.get(key, False)
            if path is False or matrix is None:
                continue
            if path is None or path is True:
                if integral is None:
                    raise RuntimeError(f"No legacy label is available for {key}.")
                path = integral.label.casefold() + ".pt"
            torch.save(matrix, path)

        ###################################
        # SELF-CONSISTENT FIELD PROCEDURE #
        ###################################

        old_cuda_sync = timer.cuda_sync
        timer.cuda_sync = kwargs.get(
            "cuda_sync_in_scf", False if self.device.type == "cpu" else True
        )
        timer.start("SCF", "Self-Consistent Field")

        # get caches of all interactions
        timer.start("Interaction Cache", parent_uid="SCF")
        OutputHandler.write_stdout_nf(" - Interaction Cache ... ", v=3)
        icaches = self.interactions.get_cache(
            numbers=self.numbers, positions=positions, ihelp=self.ihelp
        )
        timer.stop("Interaction Cache")
        OutputHandler.write_stdout("done", v=3)

        # Electronic solve
        if self.opts.scf.requires_iterations:
            OutputHandler.write_stdout("\nStarting SCF Iterations...", v=3)
        else:
            OutputHandler.write_stdout(
                "\nStarting non-self-consistent electronic solve...", v=3
            )

        scf_results = scf.solve(
            self.numbers,
            positions,
            _chrg,
            _spin,
            self.interactions,
            icaches,
            self.ihelp,
            self.opts.scf,
            intmats,
            refocc,
        )

        timer.stop("SCF")
        timer.cuda_sync = old_cuda_sync
        if self.opts.scf.requires_iterations:
            OutputHandler.write_stdout(
                f"SCF finished in {scf_results['iterations']} iterations.", v=3
            )
        else:
            OutputHandler.write_stdout(
                "Non-self-consistent electronic solve finished.", v=3
            )

        scf_energy = scf_results["energy"] + scf_results["fenergy"]
        result = Result.snapshot(
            energy=total_energy + scf_energy,
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
            overlap=intmats.overlap,
            hcore=intmats.hcore,
            dipole_integrals=intmats.dipole,
            quadrupole_integrals=intmats.quadrupole,
            overlap_norm=(
                None if self.integrals.overlap is None else self.integrals.overlap.norm
            ),
            iterations=torch.tensor(
                scf_results["iterations"], dtype=torch.int64, device=self.device
            ),
        )

        if self.ihelp.batch_mode == 0:
            OutputHandler.write_stdout(
                "SCF Energy  : %.14f Hartree.",
                scf_results["energy"].sum(-1),
                v=2,
            )
            OutputHandler.write_stdout(
                "Total Energy: %.14f Hartree.",
                result.energy.sum(-1),
                v=1,
            )

        self._ncalcs += 1
        return result

    def _singlepoint_system(
        self,
        positions: Tensor,
        charge: Tensor,
        spin: Tensor | None,
        kwargs: dict[str, Any],
    ) -> Result:
        """Adapt a pure System result to legacy Calculator state and options."""
        old_cuda_sync = timer.cuda_sync
        timer.cuda_sync = kwargs.get(
            "cuda_sync_in_scf", False if self.device.type == "cpu" else True
        )
        try:
            result = self.system.singlepoint(positions, charge, spin)
        finally:
            timer.cuda_sync = old_cuda_sync

        # Legacy wrappers remain for analytical gradient operations and
        # write_* filenames. Per-call matrices stay on the returned Result.
        for key, matrix, integral in (
            ("write_overlap", result.overlap, self.integrals.overlap),
            ("write_dipole", result.dipole_integrals, self.integrals.dipole),
            (
                "write_quadrupole",
                result.quadrupole_integrals,
                self.integrals.quadrupole,
            ),
            ("write_hcore", result.hcore, self.integrals.hcore),
        ):
            path = kwargs.get(key, False)
            if path is False or matrix is None:
                continue
            if path is None or path is True:
                if integral is None:
                    raise RuntimeError(f"No legacy label is available for {key}.")
                path = integral.label.casefold() + ".pt"
            torch.save(matrix, path)

        self._ncalcs += 1
        return result

    def energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the total energy :math:`E` of the system.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        Tensor
            Total energy of the system (scalar value).
        """
        result = self.singlepoint(positions, chrg, spin, **kwargs)
        return result.energy.sum(-1, keepdim=kwargs.get("keepdim", False))

    def bond_orders(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the (Wiberg) bond order matrix.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        Tensor
            Bond order matrix.
        """
        result = self.singlepoint(positions, chrg, spin, **kwargs)
        if result.overlap is None or result.density is None:
            raise RuntimeError("Bond orders require overlap and density matrices.")

        # pylint: disable=import-outside-toplevel
        from dxtb._src.wavefunction.wiberg import get_bond_order

        return get_bond_order(result.overlap, result.density, self.ihelp)

    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ):
        """
        Calculate the requested properties. This is more of a dispatcher method
        that calls the appropriate methods of the Calculator.

        Parameters
        ----------
        properties : list[str]
            List of properties to calculate.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        dict
            Dictionary of calculated properties.
        """
        reject_removed_store_kwargs(kwargs)
        values: dict[str, Any] = {}
        result_fields = {
            "charges": "charges",
            "coefficients": "coefficients",
            "density": "density",
            "iterations": "iterations",
            "mo_energies": "emo",
            "occupation": "occupation",
            "potential": "potential",
        }
        needs_result = bool(
            set(properties) & (set(result_fields) | {"energy", "bond_orders"})
        )
        result = (
            self.singlepoint(positions, chrg, spin, **kwargs)
            if needs_result
            else None
        )
        if result is not None:
            for name in properties:
                if name in result_fields:
                    values[name] = getattr(result, result_fields[name])
                elif name == "energy":
                    values[name] = result.energy.sum(
                        -1, keepdim=kwargs.get("keepdim", False)
                    )
        if "bond_orders" in properties:
            assert result is not None
            if result.overlap is None or result.density is None:
                raise RuntimeError("Bond orders require overlap and density matrices.")
            from dxtb._src.wavefunction.wiberg import get_bond_order

            values["bond_orders"] = get_bond_order(
                result.overlap, result.density, self.ihelp
            )
        return values
