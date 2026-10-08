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
Integral container
==================

A class that acts as a container for integrals.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass

import torch

from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Any, Tensor, TensorLike
from dxtb._src.xtb.base import BaseHamiltonian

from .base import BaseIntegral
from .driver import DriverManager
from .types import DipoleIntegral, OverlapIntegral, QuadrupoleIntegral

__all__ = ["Integrals", "IntegralMatrices"]

logger = logging.getLogger(__name__)


class IntegralContainer(TensorLike):
    """
    Base class for integral container.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _run_checks: bool = True,
    ):
        super().__init__(device, dtype)
        self._run_checks = _run_checks

    @property
    def run_checks(self) -> bool:
        return self._run_checks

    @run_checks.setter
    def run_checks(self, run_checks: bool) -> None:
        current = self.run_checks
        self._run_checks = run_checks

        # switching from False to True should automatically run checks
        if current is False and run_checks is True:
            self.checks()

    @abstractmethod
    def checks(self) -> None:
        """Run checks for integrals."""


class Integrals(IntegralContainer):
    """
    Integral container.
    """

    __slots__ = [
        "mgr",
        "intlevel",
        "_hcore",
        "_overlap",
        "_dipole",
        "_quadrupole",
    ]

    def __init__(
        self,
        mgr: DriverManager,
        *,
        intlevel: int = defaults.INTLEVEL,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _hcore: BaseHamiltonian | None = None,
        _overlap: OverlapIntegral | None = None,
        _dipole: DipoleIntegral | None = None,
        _quadrupole: QuadrupoleIntegral | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.mgr = mgr
        self.intlevel = intlevel

        self._hcore = _hcore
        self._overlap = _overlap
        self._dipole = _dipole
        self._quadrupole = _quadrupole

    def setup_driver(self, positions: Tensor, **kwargs: Any) -> Any:
        """
        Setup the driver for the integrals.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        **kwargs : Any
            Additional keyword arguments for the driver.
        """
        return self.mgr.setup_driver(positions, **kwargs)

    # Core Hamiltonian

    @property
    def hcore(self) -> BaseHamiltonian | None:
        return self._hcore

    @hcore.setter
    def hcore(self, hcore: BaseHamiltonian) -> None:
        self._hcore = hcore
        self.checks()

    def build_hcore(
        self,
        positions: Tensor,
        with_overlap: bool = True,
        charge: Tensor | float | int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        logger.debug("Core Hamiltonian: Start building matrix.")

        if self.hcore is None:
            raise RuntimeError("Core Hamiltonian integral not initialized.")

        # if overlap integral required
        if with_overlap is True and self.overlap is None:
            self.build_overlap(positions, **kwargs)

        if with_overlap is True:
            assert self.overlap is not None
            overlap = self.overlap.matrix
        else:
            overlap = None

        hcore = self.hcore.build(positions, overlap=overlap, charge=charge)
        logger.debug("Core Hamiltonian: All finished.")
        return hcore

    # overlap

    @property
    def overlap(self) -> OverlapIntegral | None:
        """
        Overlap integral class. The integral matrix of shape
        ``(..., nao, nao)`` is stored in the :attr:`matrix` attribute.

        Returns
        -------
        Tensor | None
            Overlap integral if set, else ``None``.
        """
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: OverlapIntegral) -> None:
        self._overlap = overlap
        self.checks()

    def build_overlap(self, positions: Tensor, **kwargs: Any) -> Tensor:
        """
        Build the overlap integral (shape: ``(..., nao, nao)``).

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

        Returns
        -------
        Tensor
            Overlap integral of shape ``(..., nao, nao)``.
        """
        # in case CPU is forced for libcint, move positions to CPU
        if self.mgr.force_cpu_for_libcint is True:
            if positions.device != torch.device("cpu"):
                positions = positions.to(device=torch.device("cpu"))

        driver_data = self.mgr.setup_driver(positions, **kwargs)
        logger.debug("Overlap integral: Start building matrix.")

        if self.overlap is None:
            # pylint: disable=import-outside-toplevel
            from .factory import new_overlap

            self.overlap = new_overlap(
                self.mgr.driver_type, **self.dd, **kwargs
            )

        # DEVNOTE: If the overlap is only built if `self.overlap._matrix` is
        # `None`, the overlap will not be rebuilt if the positions change,
        # i.e., when the driver was invalidated. Hence, we would require a
        # full reset of the integrals via `reset_all`. However, the integral
        # reset cannot be triggered by the driver manager, so we cannot add this
        # check here. If we do, the hessian tests will fail as the overlap is
        # not recalculated for positions + delta.
        self.overlap.build(driver_data)
        self.overlap.normalize()
        assert self.overlap.matrix is not None

        # move integral to the correct device...
        if self.mgr.force_cpu_for_libcint is True:
            # ...but only if no other multipole integrals are required
            if self.intlevel <= labels.INTLEVEL_HCORE:
                self.overlap = self.overlap.to(device=self.device)

                # DEVNOTE: This is a sanity check to avoid the following
                # scenario: When the overlap is built on CPU (forced by
                # libcint), it will be moved to the correct device after
                # the last integral is built. Now, in case of a second
                # call with an invalid cache, the overlap class already
                # is on the correct device, but the matrix is not. Hence,
                # the `to` method must be called on the matrix as well,
                # which is handled in the custom `to` method of all
                # integrals.
                # Also make sure to pass the `force_cpu_for_libcint`
                # flag when instantiating the integral classes.
                assert self.overlap is not None
                if self.overlap.device != self.overlap.matrix.device:
                    raise RuntimeError(
                        f"Device of '{self.overlap.label}' integral class "
                        f"({self.overlap.device}) and its matrix "
                        f"({self.overlap.matrix.device}) do not match."
                    )

        logger.debug("Overlap integral: All finished.")

        return self.overlap.matrix

    def grad_overlap(self, positions: Tensor, **kwargs) -> Tensor:
        """
        Calculate the gradient of the overlap integral.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

        Returns
        -------
        Tensor
            Gradient of the overlap integral (shape: ``(..., nao, nao, 3)``).
        """
        # in case CPU is forced for libcint, move positions to CPU
        if self.mgr.force_cpu_for_libcint is True:
            if positions.device != torch.device("cpu"):
                positions = positions.to(device=torch.device("cpu"))

        driver_data = self.mgr.setup_driver(positions, **kwargs)

        if self.overlap is None:
            # pylint: disable=import-outside-toplevel
            from .factory import new_overlap

            self.overlap = new_overlap(
                self.mgr.driver_type,
                **self.dd,
                **kwargs,
            )

        logger.debug("Overlap gradient: Start.")
        self.overlap.get_gradient(driver_data, **kwargs)
        self.overlap.gradient = self.overlap.gradient.to(self.device)
        self.overlap.normalize_gradient()
        logger.debug("Overlap gradient: All finished.")

        return self.overlap.gradient.to(self.device)

    # dipole

    @property
    def dipole(self) -> DipoleIntegral | None:
        """
        Dipole integral class. The integral matrix of shape
        ``(..., 3, nao, nao)``is stored in the :attr:`matrix` attribute.

        Returns
        -------
        Tensor | None
            Dipole integral if set, else ``None``.
        """
        return self._dipole

    @dipole.setter
    def dipole(self, dipole: DipoleIntegral) -> None:
        self._dipole = dipole
        self.checks()

    def build_dipole(
        self, positions: Tensor, shift: bool = True, **kwargs: Any
    ):
        # in case CPU is forced for libcint, move positions to CPU
        if self.mgr.force_cpu_for_libcint is True:
            if positions.device != torch.device("cpu"):
                positions = positions.to(device=torch.device("cpu"))

        driver_data = self.mgr.setup_driver(positions, **kwargs)
        logger.debug("Dipole integral: Start building matrix.")

        if self.dipole is None:
            # pylint: disable=import-outside-toplevel
            from .factory import new_dipint

            self.dipole = new_dipint(self.mgr.driver_type, **self.dd, **kwargs)

        # The overlap normalization used below belongs to this geometry.
        self.build_overlap(positions, **kwargs)
        assert self.overlap is not None

        # build (with overlap norm)
        self.dipole.build(driver_data)
        self.dipole.normalize(self.overlap.norm)
        logger.debug("Dipole integral: Finished building matrix.")

        # DEVNOTE: Finally, we move the integral to the correct device, but
        # only if no higher multipole integrals are required. Otherwise, the
        # move is deferred to the highest moment integral.
        # We also have to wait with the shift, because the quadrupole integral
        # requires the r0-centered dipole integral.

        # If higher moment integrals are required, everything will be deferred.
        if self.intlevel > labels.INTLEVEL_DIPOLE:
            logger.debug("Dipole integral: All finished (r0).")
            return self.dipole.matrix

        # If not, we have to move all integrals calculated so far and
        # potentially shift the dipole integral. We shift first, because
        # the `positions` are on CPU if `force_cpu_for_libcint=True`.
        if shift is True:
            logger.debug("Dipole integral: Start shifting operator (r0->rj).")

            # shift to rj if required (requires overlap integral)
            self.dipole.shift_r0_rj(
                self.overlap.matrix,
                driver_data.ihelp.spread_atom_to_orbital(
                    positions,
                    dim=-2,
                    extra=True,
                ),
            )

            logger.debug("Dipole integral: Finished shifting operator.")

        # move integral to the correct device
        if self.mgr.force_cpu_for_libcint is True:
            self.overlap = self.overlap.to(device=self.device)
            self.dipole = self.dipole.to(device=self.device)

        logger.debug("Dipole integral: All finished (rj).")
        return self.dipole.matrix

    # quadrupole

    @property
    def quadrupole(self) -> QuadrupoleIntegral | None:
        """
        Quadrupole integral class. The integral matrix of shape
        ``(..., 6, nao, nao)`` or ``(..., 9, nao, nao)`` is stored
        in the :attr:`matrix` attribute.

        Returns
        -------
        Tensor | None
            Quadrupole integral if set, else ``None``.
        """
        return self._quadrupole

    @quadrupole.setter
    def quadrupole(self, quadrupole: QuadrupoleIntegral) -> None:
        self._quadrupole = quadrupole
        self.checks()

    def build_quadrupole(
        self,
        positions: Tensor,
        shift: bool = True,
        traceless: bool = True,
        **kwargs: Any,
    ):
        """
        Build the quadrupole integral (shape: ``(6, nao, nao)``).

        The quadrupole integral is returned, but also written to the
        :attr:`matrix` attribute of the quadrupole class.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        shift : bool, optional
            Shift the moment operator from the origin (``r0``) to the atoms
            (``rj``). Defaults to ``True``.
        traceless : bool, optional
            Make the quadrupole moment traceless. Defaults to ``True``.
        **kwargs : Any
            Additional keyword arguments for the integral factories.

        Returns
        -------
        Tensor
            Quadrupole integral of shape ``(6, nao, nao)``.
        """
        if self.intlevel > labels.INTLEVEL_MAX:
            if shift is True or traceless is True:
                raise RuntimeError(
                    f"An integral level of '{self.intlevel}' is requested, "
                    "but the quadrupole integral (level "
                    f"{labels.INTLEVEL_QUADRUPOLE}) is the highest moment "
                    "supported. I am cautiously raising an error here, "
                    "as this is likely a mistake and messes up parts of the "
                    "internal integral logic."
                    "\nDetailed explanation: With a higher integral level "
                    "present, the moment operator is not shifted (r0->rj) "
                    "and the quadrupole integral is not made traceless as "
                    "these operations are deferred to the highest moment "
                    "integral. If the provided integral level does not "
                    "correspond to an implemented multipole integral, these "
                    "integral changes are never executed. The construction of "
                    "the Hamiltonian, however, expects `rj`-centered integrals "
                    "and traceless quadrupole integrals."
                )

        # in case CPU is forced for libcint, move positions to CPU
        if self.mgr.force_cpu_for_libcint is True:
            if positions.device != torch.device("cpu"):
                positions = positions.to(device=torch.device("cpu"))

        # check all instantiations
        driver_data = self.mgr.setup_driver(positions, **kwargs)
        logger.debug("Quad integral: Start building matrix.")

        if self.quadrupole is None:
            # pylint: disable=import-outside-toplevel
            from .factory import new_quadint

            self.quadrupole = new_quadint(
                self.mgr.driver_type,
                **self.dd,
                **kwargs,
            )

        # Both normalization and the shift below depend on the current
        # geometry, even when this container has built integrals before.
        self.build_overlap(positions, **kwargs)
        assert self.overlap is not None

        # build
        self.quadrupole.build(driver_data)
        self.quadrupole.normalize(self.overlap.norm)
        logger.debug("Quad integral: Finished building matrix.")

        # DEVNOTE: Finally, we move the integral to the correct device, but
        # only if no higher multipole integrals are required (redundant for
        # quadrupole integrals, because this is the highest moment).

        # If higher moment integrals are required, everything will be deferred.
        # (Never happens, because we raise an error above.)
        if self.intlevel > labels.INTLEVEL_QUADRUPOLE:  # pragma: no cover
            logger.debug("Dipole integral: All finished (r0).")
            return self.quadrupole.matrix

        # If not, we have to move all integrals calculated so far and
        # potentially shift all integrals. We shift first, because
        # the `positions` are still on CPU if `force_cpu_for_libcint=True`.
        if shift is True:
            # Build dipole integral if not already done, but NO shift yet (r0)!
            # The r0-centered dipole used for the quadrupole shift must also
            # be from this geometry.
            self.build_dipole(positions, shift=False, **kwargs)
            assert self.dipole is not None

            logger.debug("Quad integral: Start shifting operator (r0r0->rjrj).")

            # shift to rjrj requires overlap and r0-centered dipole integral
            self.quadrupole.shift_r0r0_rjrj(
                self.dipole.matrix,
                self.overlap.matrix,
                driver_data.ihelp.spread_atom_to_orbital(
                    positions,
                    dim=-2,
                    extra=True,
                ),
            )

            logger.debug("Quad integral: Finished shifting (r0r0->rjrj).")
            logger.debug("Dipole integral: Start shifting operator (r0->rj).")

            self.dipole.shift_r0_rj(
                self.overlap.matrix,
                driver_data.ihelp.spread_atom_to_orbital(
                    positions,
                    dim=-2,
                    extra=True,
                ),
            )

            logger.debug("Dipole integral: Finished shifting (r0->rj).")

        # Make traceless only after shifting!
        if traceless is True:
            logger.debug("Quad integral: Start making traceless.")
            self.quadrupole.traceless()
            logger.debug("Quad integral: Finished making traceless.")

        if self.mgr.force_cpu_for_libcint is True:
            self.overlap = self.overlap.to(self.device)
            self.quadrupole = self.quadrupole.to(self.device)

            # Dipole integral can be missing if we only require the quadrupole
            # and do not shift.
            if self.dipole is not None:
                self.dipole = self.dipole.to(self.device)

        logger.debug("Quad integral: All finished.")
        return self.quadrupole.matrix

    # checks

    def checks(self) -> None:
        if self.run_checks is False:
            return

        for name in ["hcore", "overlap", "dipole", "quadrupole"]:
            cls: (
                BaseHamiltonian
                | OverlapIntegral
                | DipoleIntegral
                | QuadrupoleIntegral
                | None
            ) = getattr(self, f"_{name}")

            if cls is None:
                continue

            if cls.dtype != self.dtype:
                raise RuntimeError(
                    f"Data type of '{cls.label}' integral ({cls.dtype}) and "
                    f"integral container ({self.dtype}) do not match."
                )
            if self.mgr.force_cpu_for_libcint is False:
                if cls.device != self.device:
                    raise RuntimeError(
                        f"Device of '{cls.label}' integral ({cls.device}) and "
                        f"integral container ({self.device}) do not match."
                    )

            if name != "hcore":
                assert not isinstance(cls, BaseHamiltonian)

                family_integral = cls.family
                family_driver = self.mgr.driver.family
                driver_label = self.mgr.driver
                if family_integral != family_driver:
                    raise RuntimeError(
                        f"The '{cls.label}' integral implementation "
                        f"requests the '{family_integral}' family, but "
                        f"the integral driver '{driver_label}' is "
                        "configured.\n"
                        "If you want to request the 'pytorch' implementations, "
                        "specify the driver name in the constructors of both "
                        "the integral container and the actual integral class."
                    )

    def reset_all(self) -> None:
        self.mgr.invalidate_driver()

        for slot in self.__slots__:
            i = getattr(self, slot)

            if not slot.startswith("_") or i is None:
                continue

            if isinstance(i, BaseIntegral) or isinstance(i, BaseHamiltonian):
                i.clear()

    @property
    def labels(self) -> list[str]:
        """Return all initialized integrals by label."""
        return [
            slot[1:]
            for slot in self.__slots__
            if slot.startswith("_") and getattr(self, slot) is not None
        ]

    def __len__(self) -> int:
        """Print all initialized integrals."""
        return sum(
            1
            for slot in self.__slots__
            if slot.startswith("_") and getattr(self, slot) is not None
        )

    # pretty print

    def __str__(self) -> str:  # pragma: no cover
        attributes = ["hcore", "overlap", "dipole", "quadrupole"]
        details = []

        for attr in attributes:
            i = getattr(self, "_" + attr)
            info = str(i) if i is not None else "None"
            details.append(f"\n  {attr}={info}")

        return f"Integrals({', '.join(details)}\n)"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


@dataclass(frozen=True, eq=False)
class IntegralMatrices:
    """Immutable tensor values produced by one integral evaluation."""

    hcore: Tensor
    overlap: Tensor
    dipole: Tensor | None = None
    quadrupole: Tensor | None = None

    def __post_init__(self) -> None:
        reference = self.hcore
        if (
            reference.ndim not in (2, 3)
            or reference.shape[-2] != reference.shape[-1]
        ):
            raise ValueError("Tensor 'hcore' must have shape (..., nao, nao).")
        if self.overlap.shape != reference.shape:
            raise ValueError("Tensor 'overlap' must match the hcore shape.")
        if (
            self.overlap.device != reference.device
            or self.overlap.dtype != reference.dtype
        ):
            raise ValueError(
                "All integral matrices must share device and dtype."
            )

        for name, tensor, components in (
            ("dipole", self.dipole, (defaults.DP_SHAPE,)),
            ("quadrupole", self.quadrupole, (6, 9)),
        ):
            if tensor is None:
                continue
            if tensor.ndim != reference.ndim + 1:
                raise ValueError(f"Tensor '{name}' has an incompatible rank.")
            expected = (
                *reference.shape[:-2],
                tensor.shape[-3],
                *reference.shape[-2:],
            )
            if tensor.shape != expected or tensor.shape[-3] not in components:
                raise ValueError(f"Tensor '{name}' has an incompatible shape.")
            if (
                tensor.device != reference.device
                or tensor.dtype != reference.dtype
            ):
                raise ValueError(
                    "All integral matrices must share device and dtype."
                )

    @property
    def device(self) -> torch.device:
        return self.hcore.device

    @property
    def dtype(self) -> torch.dtype:
        return self.hcore.dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> IntegralMatrices:
        """Return a copy with all matrix tensors converted."""
        return IntegralMatrices(
            hcore=self.hcore.to(device=device, dtype=dtype),
            overlap=self.overlap.to(device=device, dtype=dtype),
            dipole=(
                None
                if self.dipole is None
                else self.dipole.to(device=device, dtype=dtype)
            ),
            quadrupole=(
                None
                if self.quadrupole is None
                else self.quadrupole.to(device=device, dtype=dtype)
            ),
        )

    def type(self, dtype: torch.dtype) -> IntegralMatrices:
        """Return a copy with a different floating point type."""
        return self.to(dtype=dtype)

    def slice(
        self, matrix_indices: tuple, multipole_indices: tuple
    ) -> IntegralMatrices:
        """Return a value with its orbital dimensions sliced."""
        return IntegralMatrices(
            hcore=self.hcore[matrix_indices],
            overlap=self.overlap[matrix_indices],
            dipole=(
                None if self.dipole is None else self.dipole[multipole_indices]
            ),
            quadrupole=(
                None
                if self.quadrupole is None
                else self.quadrupole[multipole_indices]
            ),
        )

    def __str__(self) -> str:  # pragma: no cover
        fields = ("hcore", "overlap", "dipole", "quadrupole")
        details = [
            f"\n  {name}={getattr(self, name).shape if getattr(self, name) is not None else 'None'}"
            for name in fields
        ]
        return f"IntegralMatrices({', '.join(details)}\n)"
