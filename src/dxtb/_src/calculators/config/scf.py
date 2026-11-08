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
SCF configuration.

The configurations are immutable. A changed setting is a new object, created
with :func:`dataclasses.replace`. User input (strings, NumPy integers) is
converted by the ``create`` class methods; the constructors only accept the
final values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from operator import index

from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Any

__all__ = ["ConfigSCF", "ConfigFermi"]


def _label(
    value: str | int,
    name: str,
    table: dict[int, tuple[str, ...]],
    unknown: str,
) -> int:
    """
    Convert a string (or integer) option to its integer label.

    Parameters
    ----------
    value : str | int
        User input.
    name : str
        Name of the option for the error message of a wrong type.
    table : dict[int, tuple[str, ...]]
        Integer label and the strings that select it.
    unknown : str
        Beginning of the error message for an unknown option.

    Returns
    -------
    int
        Integer label.
    """
    choices = ", ".join(s for strs in table.values() for s in strs)

    if isinstance(value, str):
        for code, strs in table.items():
            if value.casefold() in strs:
                return code
    elif isinstance(value, int):
        if value in table:
            return value
    else:
        raise TypeError(
            f"The {name} must be of type 'int' or 'str', but "
            f"'{type(value)}' was given."
        )

    raise ValueError(f"{unknown} '{value}'. Use one of '{choices}'.")


_GUESS = {
    labels.GUESS_EEQ: labels.GUESS_EEQ_STRS,
    labels.GUESS_SAD: labels.GUESS_SAD_STRS,
}
_PARTITION = {
    labels.FERMI_PARTITION_EQUAL: labels.FERMI_PARTITION_EQUAL_STRS,
    labels.FERMI_PARTITION_ATOMIC: labels.FERMI_PARTITION_ATOMIC_STRS,
}


def _check_label(value: Any, name: str, table: dict, unknown: str) -> None:
    """Check that a final value is a known integer label."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"The {name} must be an integer label (use `create` for "
            f"strings), but '{type(value)}' was given."
        )
    if value not in table:
        raise ValueError(f"{unknown} '{value}'.")


@dataclass(frozen=True, kw_only=True)
class ConfigFermi:
    """
    Configuration for fermi smearing.
    """

    etemp: float | int = defaults.FERMI_ETEMP
    """Electronic temperature for Fermi smearing (in Kelvin)."""

    maxiter: int = defaults.FERMI_MAXITER
    """Maximum number of iterations for Fermi smearing."""

    thresh: float | int | None = defaults.FERMI_THRESH
    """Threshold for Fermi iterations."""

    diff_order: int = defaults.FERMI_DIFF_ORDER
    """
    Highest order of the derivatives of the Fermi occupations that is exact.
    It sets the number of differentiable Newton steps of the Fermi energy.
    """

    partition: int = defaults.FERMI_PARTITION
    """Partitioning scheme for electronic free energy."""

    def __post_init__(self) -> None:
        if isinstance(self.diff_order, bool) or not isinstance(
            self.diff_order, int
        ):
            raise TypeError(
                "The derivative order of the Fermi occupations must be of "
                f"type 'int', but '{type(self.diff_order)}' was given."
            )
        if self.diff_order < 0:
            raise ValueError(
                "The derivative order of the Fermi occupations must not be "
                f"negative ({self.diff_order})."
            )

        _check_label(
            self.partition,
            "partition",
            _PARTITION,
            "Unknown partitioning scheme for the free energy in Fermi "
            "smearing",
        )

    @classmethod
    def create(
        cls,
        *,
        etemp: float | int = defaults.FERMI_ETEMP,
        maxiter: int = defaults.FERMI_MAXITER,
        thresh: float | int | None = defaults.FERMI_THRESH,
        diff_order: int = defaults.FERMI_DIFF_ORDER,
        partition: str | int = defaults.FERMI_PARTITION,
    ) -> ConfigFermi:
        """Create the configuration from user input."""
        # integers of any kind (e.g., NumPy), but not bool
        if not isinstance(diff_order, bool):
            try:
                diff_order = index(diff_order)
            except TypeError:
                pass

        return cls(
            etemp=etemp,
            maxiter=maxiter,
            thresh=thresh,
            diff_order=diff_order,
            partition=_label(
                partition,
                "partition",
                _PARTITION,
                "Unknown partitioning scheme for the free energy in Fermi "
                "smearing",
            ),
        )

    def info(self) -> dict[str, dict[str, None | float | int | str]]:
        """
        Return a dictionary with the Fermi smearing configuration.

        Returns
        -------
        dict[str, dict[str, float | int | str]]
            Dictionary with the Fermi smearing configuration.
        """
        return {
            "Fermi Smearing": {
                "Temperature": self.etemp,
                "Maxiter": self.maxiter,
                "Threshold": self.thresh,
                "Derivative order": self.diff_order,
                "Partioning": labels.FERMI_PARTITION_MAP[self.partition],
            }
        }


def check_tols(value: float, dtype: Any) -> float:
    """
    Set tolerances to catch unreasonably small values.

    Parameters
    ----------
    value : float
        Selected tolerance that will be checked.
    dtype : torch.dtype
        Floating point precision to adjust tolerances to.

    Returns
    -------
    float
        Possibly corrected tolerance.
    """
    # pylint: disable=import-outside-toplevel
    import torch

    from dxtb import OutputHandler

    eps = torch.finfo(dtype).eps

    if value < eps:
        OutputHandler.warn(
            f"Selected tolerance ({value:.2E}) is smaller than the "
            f"smallest value for the selected dtype ({dtype}, "
            f"{eps:.2E}). Switching to {100*eps:.2E} instead."
        )
        return 100 * eps

    return value


# The public SCF configuration lives in public_scf; ConfigFermi and the
# tolerance checks above are shared with it.
from .public_scf import ConfigSCF as ConfigSCF  # noqa: E402
from .public_scf import SCFConvergenceProfile as SCFConvergenceProfile
from .public_scf import SCFDifferentiation as SCFDifferentiation

__all__ += ["SCFConvergenceProfile", "SCFDifferentiation"]
