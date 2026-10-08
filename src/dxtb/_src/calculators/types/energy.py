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
from dxtb._src.utils.tensors import tensor_id
from dxtb._src.integral.evaluation import build_integral_matrices

from ..result import Result
from . import decorators as cdec
from .base import BaseCalculator

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
        "cache",
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

        # get the hashed key for the cache from all arguments
        hashed_key = ""
        all_args = (positions, chrg, spin) + tuple(kwargs.values())
        for i, arg in enumerate(all_args):
            sep = "_" if i > 0 else ""
            if isinstance(arg, Tensor):
                hashed_key += f"{sep}{tensor_id(arg)}"
            else:
                hashed_key += f"{sep}{arg}"

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

        result = Result(positions, **self.dd)

        ###########################
        # CLASSICAL CONTRIBUTIONS #
        ###########################

        if len(self.classicals.components) > 0:
            OutputHandler.write_stdout_nf(" - Classicals        ... ", v=3)
            timer.start("Classicals")

            ccaches = self.system.classical_cache
            cenergies = self.classicals.get_energy(
                positions, ccaches, charge=_chrg
            )
            result.cenergies = cenergies
            result.total += torch.stack(list(cenergies.values())).sum(0)

            timer.stop("Classicals")
            OutputHandler.write_stdout("done", v=3)

        if {"all", "scf"} & set(self.opts.exclude):
            self.cache["energy"] = result.total

            return result

        #############
        # INTEGRALS #
        #############

        timer.start("Integrals")
        OutputHandler.write_stdout_nf(" - Integral matrices ... ", v=3)
        if self.system.h0_setup is None:
            raise NotImplementedError("Core Hamiltonian setup is missing.")

        if self.system.integral_setup is not None:
            intmats, refocc, overlap_norm = build_integral_matrices(
                self.system.integral_setup,
                self.system.h0_setup,
                positions,
                _chrg,
            )
            if self.integrals.overlap is None:
                raise RuntimeError("Legacy overlap adapter is not initialized.")
            self.integrals.overlap.matrix = intmats.overlap
            self.integrals.overlap.norm = overlap_norm
            if self.integrals.hcore is None:
                raise RuntimeError("Legacy H0 adapter is not initialized.")
            self.integrals.hcore.matrix = intmats.hcore
            if intmats.dipole is not None:
                if self.integrals.dipole is None:
                    raise RuntimeError(
                        "Legacy dipole adapter is not initialized."
                    )
                self.integrals.dipole.matrix = intmats.dipole
            if intmats.quadrupole is not None:
                if self.integrals.quadrupole is None:
                    raise RuntimeError(
                        "Legacy quadrupole adapter is not initialized."
                    )
                self.integrals.quadrupole.matrix = intmats.quadrupole
        else:
            # Existing batched Calculator calls still use the legacy builder
            # until the later single-system/vmap batching package.
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

        result.integrals = intmats

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

        # store SCF results
        result.charges = scf_results["charges"]
        result.coefficients = scf_results["coefficients"]
        result.density = scf_results["density"]
        result.emo = scf_results["emo"]
        result.fenergy = scf_results["fenergy"]
        result.hamiltonian = scf_results["hamiltonian"]
        result.occupation = scf_results["occupation"]
        result.potential = scf_results["potential"]
        result.scf = scf_results["energy"]
        result.fenergy = scf_results["fenergy"]
        result.iter = scf_results["iterations"]

        scf_energy = scf_results["energy"] + scf_results["fenergy"]
        result.total += scf_energy

        if self.ihelp.batch_mode == 0:
            OutputHandler.write_stdout(
                "SCF Energy  : %.14f Hartree.",
                scf_results["energy"].sum(-1),
                v=2,
            )
            OutputHandler.write_stdout(
                "Total Energy: %.14f Hartree.",
                result.total.sum(-1),
                v=1,
            )

        # Store results. Energy always stored.
        self.cache["energy"] = result.total

        copts = self.opts.cache.store

        if kwargs.get("store_charges", copts.charges):
            self.cache["charges"] = scf_results["charges"]
            self.cache.set_cache_key("charges", "charges:" + hashed_key)
        if kwargs.get("store_coefficients", copts.coefficients):
            self.cache["coefficients"] = scf_results["coefficients"]
            self.cache.set_cache_key(
                "coefficients", "coefficients:" + hashed_key
            )
        if kwargs.get("store_density", copts.density):
            self.cache["density"] = scf_results["density"]
            self.cache.set_cache_key("density", "density:" + hashed_key)
        if kwargs.get("store_iterations", copts.iterations):
            self.cache["iterations"] = torch.tensor(
                scf_results["iterations"], device=self.device
            )
            self.cache.set_cache_key("iterations", "iterations:" + hashed_key)
        if kwargs.get("store_mo_energies", copts.mo_energies):
            self.cache["mo_energies"] = scf_results["emo"]
            self.cache.set_cache_key("mo_energies", "mo_energies:" + hashed_key)
        if kwargs.get("store_occupation", copts.occupation):
            self.cache["occupation"] = scf_results["occupation"]
            self.cache.set_cache_key("occupation", "occupation:" + hashed_key)
        if kwargs.get("store_potential", copts.potential):
            self.cache["potential"] = scf_results["potential"]
            self.cache.set_cache_key("potential", "potential:" + hashed_key)

        if kwargs.get("store_fock", copts.fock):
            self.cache["fock"] = scf_results["hamiltonian"]

        if kwargs.get("store_hcore", copts.hcore):
            self.cache["hcore"] = intmats.hcore

        if kwargs.get("store_overlap", copts.overlap):
            self.cache["overlap"] = self.integrals.overlap
        else:
            if self.integrals.overlap is not None:
                if self.integrals.overlap.requires_grad is False:
                    self.integrals.overlap.clear()

        if kwargs.get("store_dipole", copts.dipole):
            self.cache["dipint"] = self.integrals.dipole
        else:
            if self.integrals.dipole is not None:
                if self.integrals.dipole.requires_grad is False:
                    self.integrals.dipole.clear()

        if kwargs.get("store_quadrupole", copts.quadrupole):
            self.cache["quadint"] = self.integrals.quadrupole
        else:
            if self.integrals.quadrupole is not None:
                if self.integrals.quadrupole.requires_grad is False:
                    self.integrals.quadrupole.clear()

        self._ncalcs += 1
        return result

    @cdec.cache
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
        self.singlepoint(positions, chrg, spin, **kwargs)
        e = self.cache["energy"]

        if e is None:
            raise RuntimeError(
                "Energy not found in cache after singlepoint calculation. "
                "This should not happen; the `singlepoint` method should "
                "always write at least the energy to the cache (even "
                "without caching enabled). Please report this issue."
            )

        assert isinstance(e, Tensor)
        return e.sum(-1, keepdim=kwargs.get("keepdim", False))

    @cdec.cache
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
        self.singlepoint(positions, chrg, spin, **kwargs)

        ovlp_msg = (
            "Overlap matrix not found in cache. The overlap is not saved "
            "per default. Enable saving either via the calculator options "
            '(`opts={"cache_overlap": True}`) or by passing the '
            "`store_overlap=True` keyword argument to called method, e.g., "
            "`calc.energy(positions, store_overlap=True)`"
        )

        overlap = self.cache["overlap"]
        if overlap is None:
            raise RuntimeError(ovlp_msg)

        # pylint: disable=import-outside-toplevel
        from dxtb._src.integral.types import OverlapIntegral

        assert isinstance(overlap, OverlapIntegral)
        if overlap.matrix is None:
            raise RuntimeError(ovlp_msg)

        density = self.cache["density"]
        if density is None:
            raise RuntimeError(
                "Density matrix not found in cache. The density is not saved "
                "per default. Enable saving either via the calculator options "
                '(`opts={"cache_density": True}`) or by passing the '
                "`store_density=True` keyword argument to called method, e.g., "
                "`calc.energy(positions, store_density=True)`"
            )

        # pylint: disable=import-outside-toplevel
        from dxtb._src.wavefunction.wiberg import get_bond_order

        return get_bond_order(overlap.matrix, density, self.ihelp)

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
        if self.opts.cache.enabled is False:
            self.cache.reset_all()

        # treat bond orders separately for better error message
        if "bond_orders" in properties:
            self.bond_orders(positions, chrg, spin, **kwargs)

        props = self.get_implemented_properties()
        props.remove("bond_orders")
        if set(props) & set(properties):
            self.energy(positions, chrg, spin, **kwargs)
