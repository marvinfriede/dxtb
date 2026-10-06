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
Config: Cache
=============

Configuration for the cache.

The configuration is immutable: a changed setting is a new object, created
with :func:`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from dxtb._src.constants import defaults

__all__ = ["ConfigCache", "ConfigCacheStore"]


@dataclass(frozen=True, kw_only=True)
class ConfigCacheStore:
    """
    Configuration for the cache store.
    """

    hcore: bool = defaults.CACHE_STORE_HCORE
    """Whether to store the core Hamiltonian matrix."""

    overlap: bool = defaults.CACHE_STORE_OVERLAP
    """Whether to store the overlap matrix."""

    dipole: bool = defaults.CACHE_STORE_DIPOLE
    """Whether to store the dipole moment."""

    quadrupole: bool = defaults.CACHE_STORE_QUADRUPOLE
    """Whether to store the quadrupole moment."""

    #

    charges: bool = defaults.CACHE_STORE_CHARGES
    """Whether to store the atomic charges."""

    coefficients: bool = defaults.CACHE_STORE_COEFFICIENTS
    """Whether to store the MO coefficients."""

    density: bool = defaults.CACHE_STORE_DENSITY
    """Whether to store the density matrix."""

    fock: bool = defaults.CACHE_STORE_FOCK
    """Whether to store the Fock matrix."""

    iterations: bool = defaults.CACHE_STORE_ITERATIONS
    """Whether to store the number of SCF iterations."""

    mo_energies: bool = defaults.CACHE_STORE_MO_ENERGIES
    """Whether to store the MO energies."""

    occupation: bool = defaults.CACHE_STORE_OCCUPATIONS
    """Whether to store the occupation numbers."""

    potential: bool = defaults.CACHE_STORE_POTENTIAL
    """Whether to store the potential matrix."""


@dataclass(frozen=True, kw_only=True)
class ConfigCache:
    """
    Configuration for the cache of the calculator.

    This configuration only stores a flag whether caching is enabled and a
    storage class that flags all properties which should be cached.
    """

    enabled: bool = defaults.CACHE_ENABLED
    """
    Enable or disable the cache.

    .. warning::

        With the cache enabled, numerical derivatives with respect to the
        electric field return zero, and a view of a previously calculated
        batch (e.g., ``positions[0]``) returns the cached batch result. See
        :ref:`help_known_issues`.
    """

    store: ConfigCacheStore = field(default_factory=ConfigCacheStore)
    """Container for which quantities to store."""

    @classmethod
    def create(
        cls,
        *,
        enabled: bool = defaults.CACHE_ENABLED,
        #
        hcore: bool = defaults.CACHE_STORE_HCORE,
        overlap: bool = defaults.CACHE_STORE_OVERLAP,
        dipole: bool = defaults.CACHE_STORE_DIPOLE,
        quadrupole: bool = defaults.CACHE_STORE_QUADRUPOLE,
        #
        charges: bool = defaults.CACHE_STORE_CHARGES,
        coefficients: bool = defaults.CACHE_STORE_COEFFICIENTS,
        density: bool = defaults.CACHE_STORE_DENSITY,
        fock: bool = defaults.CACHE_STORE_FOCK,
        iterations: bool = defaults.CACHE_STORE_ITERATIONS,
        mo_energies: bool = defaults.CACHE_STORE_MO_ENERGIES,
        occupation: bool = defaults.CACHE_STORE_OCCUPATIONS,
        potential: bool = defaults.CACHE_STORE_POTENTIAL,
    ) -> ConfigCache:
        """Create the configuration from the flat list of options."""
        return cls(
            enabled=enabled,
            store=ConfigCacheStore(
                hcore=hcore,
                overlap=overlap,
                dipole=dipole,
                quadrupole=quadrupole,
                #
                charges=charges,
                coefficients=coefficients,
                density=density,
                fock=fock,
                iterations=iterations,
                mo_energies=mo_energies,
                occupation=occupation,
                potential=potential,
            ),
        )

    # Helpers for analytical gradient calculations

    def for_analytical_gradient(self) -> ConfigCache:
        """
        Configuration that also stores all quantities required for an
        analytical gradient calculation.
        """
        return replace(
            self,
            store=replace(
                self.store,
                charges=True,
                coefficients=True,
                density=True,
                mo_energies=True,
                occupation=True,
                overlap=True,
                potential=True,
            ),
        )

    def is_setup_for_analytical_gradient(self) -> bool:
        """
        Check if all quantities required for an analytical gradient calculation
        are stored.

        Returns
        -------
        bool
            ``True`` if all required quantities are enabled.
        """
        return all(
            [
                self.store.charges,
                self.store.coefficients,
                self.store.density,
                self.store.mo_energies,
                self.store.occupation,
                self.store.overlap,
                self.store.potential,
            ]
        )
