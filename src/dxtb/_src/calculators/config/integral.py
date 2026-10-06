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
Config: Integrals
=================

Configuration for the integrals.
"""

from __future__ import annotations

from dataclasses import dataclass

from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Literal

__all__ = ["ConfigIntegrals"]


@dataclass(frozen=True, kw_only=True)
class ConfigIntegrals:
    """
    Configuration for the integrals.

    All configuration options are represented as integers. String options are
    converted to integers by :meth:`create`, which is the entry point for
    user input. The constructor only accepts final values.
    """

    level: int = defaults.INTLEVEL
    """
    Indicator for integrals to compute.

    - 0: None
    - 1: overlap
    - 2: +core Hamiltonian
    - 3: +dipole
    - 4: +quadrupole
    """

    cutoff: float = defaults.INTCUTOFF
    """
    Real-space cutoff (in Bohr) for integral evaluation for PyTorch.
    The ``libint`` driver ignores this option.
    """

    driver: int = defaults.INTDRIVER
    """Type of integral driver."""

    uplo: Literal["n", "l", "u"] = defaults.INTUPLO  # type: ignore
    """Integral mode for PyTorch integral calculation."""

    algorithm: str | None = None
    """
    1D kernel of the PyTorch integral driver (``None``: the default,
    ``os``).
    """

    def __post_init__(self) -> None:
        if not isinstance(self.level, int):
            raise TypeError(
                f"The received integral level (`{self.level}`) is not an "
                f"integer, but {type(self.level)}."
            )

        if self.uplo not in ("n", "u", "l"):
            raise ValueError(
                f"Unknown option for `uplo` chosen: '{self.uplo}'."
            )

        if not isinstance(self.driver, int):
            raise TypeError(
                "The driver must be of type 'int' (use `ConfigIntegrals."
                f"create` for strings), but '{type(self.driver)}' was given."
            )
        if self.driver not in (labels.INTDRIVER_LIBCINT, labels.INTDRIVER_PYTORCH):
            raise ValueError(f"Unknown integral driver '{self.driver}'.")

        if self.algorithm is not None:
            if self.algorithm not in labels.INTALGORITHM_CHOICES:
                raise ValueError(
                    f"Unknown integral algorithm '{self.algorithm}'. Choose "
                    f"one of: {', '.join(labels.INTALGORITHM_CHOICES)}."
                )
            if self.driver == labels.INTDRIVER_LIBCINT:
                raise ValueError(
                    "The integral algorithm can only be chosen for the "
                    "PyTorch integral driver, not for `libcint`."
                )

    @classmethod
    def create(
        cls,
        *,
        level: int = defaults.INTLEVEL,
        cutoff: float = defaults.INTCUTOFF,
        driver: str | int = defaults.INTDRIVER,
        uplo: str = defaults.INTUPLO,
        algorithm: str | None = None,
    ) -> ConfigIntegrals:
        """
        Create the configuration from user input (strings are converted).

        Raises
        ------
        ValueError
            An option is unknown, or the libcint driver was requested
            explicitly, but is not installed.
        TypeError
            An option has the wrong type.
        """
        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        uplo = uplo.casefold()

        if isinstance(driver, str):
            if driver.casefold() in labels.INTDRIVER_LIBCINT_STRS:
                # pylint: disable=import-outside-toplevel
                from dxtb._src.exlibs.available import has_libcint

                # The default input is an integer. So, if we receive a string
                # here, we need to assume that the libcint driver was
                # explicitly requested and we need to check if the libcint
                # interface is available.
                if has_libcint is False:
                    raise ValueError(
                        "The integral driver seems to be have been set "
                        f"explicitly to '{driver}'. However, the libcint "
                        "interface is not installed."
                    )

                driver = labels.INTDRIVER_LIBCINT
            elif driver.casefold() in labels.INTDRIVER_PYTORCH_STRS:
                driver = labels.INTDRIVER_PYTORCH
            else:
                raise ValueError(f"Unknown integral driver '{driver}'.")
        elif isinstance(driver, int):
            if driver not in (
                labels.INTDRIVER_LIBCINT,
                labels.INTDRIVER_PYTORCH,
            ):
                raise ValueError(f"Unknown integral driver '{driver}'.")

            if driver == labels.INTDRIVER_LIBCINT:
                # pylint: disable=import-outside-toplevel
                from dxtb._src.exlibs.available import has_libcint

                # If we receive the default integer here, we issue a warning
                # and fall back to the PyTorch driver.
                if has_libcint is False:
                    from dxtb import OutputHandler

                    OutputHandler.warn(
                        "The libcint interface is not installed. "
                        "Falling back to the PyTorch driver."
                    )
                    driver = labels.INTDRIVER_PYTORCH
        else:
            raise TypeError(
                "The driver must be of type 'int' or 'str', but "
                f"'{type(driver)}' was given."
            )

        if algorithm is not None:
            if not isinstance(algorithm, str) or (
                algorithm.casefold() not in labels.INTALGORITHM_CHOICES
            ):
                raise ValueError(
                    f"Unknown integral algorithm '{algorithm}'. Choose one "
                    f"of: {', '.join(labels.INTALGORITHM_CHOICES)}."
                )
            algorithm = algorithm.casefold()

        return cls(
            level=level,
            cutoff=cutoff,
            driver=driver,
            uplo=uplo,  # type: ignore
            algorithm=algorithm,
        )
