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

from dataclasses import dataclass, field
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
_SCF_MODE = {
    labels.SCF_MODE_IMPLICIT: labels.SCF_MODE_IMPLICIT_STRS,
    labels.SCF_MODE_FULL: labels.SCF_MODE_FULL_STRS,
    labels.SCF_MODE_EXPERIMENTAL: labels.SCF_MODE_EXPERIMENTAL_STRS,
}
_SCP_MODE = {
    labels.SCP_MODE_CHARGE: labels.SCP_MODE_CHARGE_STRS,
    labels.SCP_MODE_POTENTIAL: labels.SCP_MODE_POTENTIAL_STRS,
    labels.SCP_MODE_FOCK: labels.SCP_MODE_FOCK_STRS,
}
_MIXER = {
    labels.MIXER_LINEAR: labels.MIXER_LINEAR_STRS,
    labels.MIXER_ANDERSON: labels.MIXER_ANDERSON_STRS,
    labels.MIXER_BROYDEN: labels.MIXER_BROYDEN_STRS,
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


@dataclass(frozen=True, kw_only=True)
class ConfigSCF:
    """
    Configuration for the SCF.

    All configuration options are represented as integers. String options are
    converted to integers by :meth:`create`.

    The settings for Fermi smearing are stored separately in the
    :class:`ConfigFermi` class, which can be accessed via the :attr:`fermi`
    attribute.

    The configuration does not know the data type of the calculation. The
    tolerances are checked against the precision of the tensors by the SCF
    (:func:`check_tols`).
    """

    strict: bool = False
    """Strict mode for SCF configuration. Always throws errors if ``True``."""

    method: int = defaults.METHOD
    """Integer code for tight-binding method."""

    guess: int = defaults.GUESS
    """Initial guess for the SCF."""

    maxiter: int = defaults.MAXITER
    """Maximum number of SCF iterations."""

    mixer: int = defaults.MIXER
    """Mixing scheme for SCF iterations."""

    mix_guess: bool = defaults.MIX_GUESS
    """Include the initial guess in the mixing scheme."""

    damp: float = defaults.DAMP
    """Damping factor for the SCF iterations."""

    damp_init: float = defaults.DAMP_INIT
    """Initial damping factor for the SCF iterations."""

    damp_dynamic: bool = defaults.DAMP_DYNAMIC
    """Whether to use dynamic damping in the SCF iterations."""

    damp_dynamic_factor: float = defaults.DAMP_DYNAMIC_FACTOR
    """
    Damping factor for dynamic damping in the SCF iterations, i.e., when
    the norm of the error falls below a threshold.
    """

    damp_soft_start: bool = defaults.DAMP_SOFT_START
    """
    If enabled, then simple mixing will be used for the first ``generations``
    number of steps, otherwise only for the first (in Anderson mixing only).
    """

    damp_generations: int = defaults.DAMP_GENERATIONS
    """
    Number of generations to use during mixing.
    Defaults to 5 as suggested by Eyert.
    """

    damp_diagonal_offset: float = defaults.DAMP_DIAGONAL_OFFSET
    """
    Offset added to the equation system's diagonal's to prevent a linear
    dependence during the mixing process. If set to ``None`` then rescaling
    will be disabled.
    """

    scf_mode: int = defaults.SCF_MODE
    """
    SCF convergence approach (denoted by backward strategy).

    .. warning::

        The implicit mode (``"implicit"``) gives correct derivatives with
        ``torch.autograd``, but does not work with ``torch.func`` transforms
        (``jacrev``, ``jacfwd``, ``vmap``); use the default (``"full"``) for
        those. See :ref:`help_known_issues`.
    """

    scp_mode: int = defaults.SCP_MODE
    """SCF convergence target (self-consistent property)."""

    x_atol: float = defaults.X_ATOL
    """Absolute tolerance for argument (x) in SCF solver."""

    x_atol_max: float = defaults.X_ATOL_MAX
    """Absolute tolerance for max norm (L∞) of the error in the SCF."""

    f_atol: float = defaults.F_ATOL
    """Absolute tolerance for function value (f(x)) the SCF solver."""

    force_convergence: bool = defaults.SCF_FORCE_CONVERGENCE
    """Force convergence of the SCF iterations."""

    fermi: ConfigFermi = field(default_factory=ConfigFermi)
    """Configuration of the Fermi smearing."""

    def __post_init__(self) -> None:
        _check_label(self.guess, "guess", _GUESS, "Unknown guess method")
        _check_label(self.scf_mode, "scf_mode", _SCF_MODE, "Unknown SCF mode")
        _check_label(
            self.scp_mode,
            "scp_mode",
            _SCP_MODE,
            "Unknown convergence target (SCP mode)",
        )
        _check_label(self.mixer, "mixer", _MIXER, "Unknown mixer")

    @classmethod
    def create(
        cls,
        *,
        strict: bool = False,
        method: int = defaults.METHOD,
        guess: str | int = defaults.GUESS,
        maxiter: int = defaults.MAXITER,
        mixer: str | int = defaults.MIXER,
        mix_guess: bool = defaults.MIX_GUESS,
        damp: float = defaults.DAMP,
        damp_init: float = defaults.DAMP_INIT,
        damp_dynamic: bool = defaults.DAMP_DYNAMIC,
        damp_dynamic_factor: float = defaults.DAMP_DYNAMIC_FACTOR,
        damp_soft_start: bool = defaults.DAMP_SOFT_START,
        damp_generations: int = defaults.DAMP_GENERATIONS,
        damp_diagonal_offset: float = defaults.DAMP_DIAGONAL_OFFSET,
        scf_mode: str | int = defaults.SCF_MODE,
        scp_mode: str | int = defaults.SCP_MODE,
        x_atol: float = defaults.X_ATOL,
        x_atol_max: float = defaults.X_ATOL_MAX,
        f_atol: float = defaults.F_ATOL,
        force_convergence: bool = defaults.SCF_FORCE_CONVERGENCE,
        # Fermi
        fermi_etemp: float | int = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: float | int | None = defaults.FERMI_THRESH,
        fermi_diff_order: int = defaults.FERMI_DIFF_ORDER,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
    ) -> ConfigSCF:
        """Create the configuration from user input."""
        if isinstance(scf_mode, str) and (
            scf_mode.casefold() in labels.SCF_MODE_REMOVED_STRS
        ):
            raise ValueError(
                f"The SCF mode '{scf_mode}' was removed. Use "
                f"'{labels.SCF_MODE_REMOVED_STRS[scf_mode.casefold()]}' "
                "instead."
            )
        if scf_mode == labels.SCF_MODE_REMOVED and not isinstance(
            scf_mode, bool
        ):
            raise ValueError(
                f"The SCF mode with integer code {scf_mode} (non-pure "
                f"implicit) was removed. Use '{labels.SCF_MODE_REPLACEMENT}' "
                f"(code {labels.SCF_MODE_IMPLICIT}) instead."
            )

        return cls(
            strict=strict,
            method=method,
            guess=_label(guess, "guess", _GUESS, "Unknown guess method"),
            maxiter=maxiter,
            mixer=_label(mixer, "mixer", _MIXER, "Unknown mixer"),
            mix_guess=mix_guess,
            damp=damp,
            damp_init=damp_init,
            damp_dynamic=damp_dynamic,
            damp_dynamic_factor=damp_dynamic_factor,
            damp_soft_start=damp_soft_start,
            damp_generations=damp_generations,
            damp_diagonal_offset=damp_diagonal_offset,
            scf_mode=_label(scf_mode, "scf_mode", _SCF_MODE, "Unknown SCF mode"),
            scp_mode=_label(
                scp_mode,
                "scp_mode",
                _SCP_MODE,
                "Unknown convergence target (SCP mode)",
            ),
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            f_atol=f_atol,
            force_convergence=force_convergence,
            fermi=ConfigFermi.create(
                etemp=fermi_etemp,
                maxiter=fermi_maxiter,
                thresh=fermi_thresh,
                diff_order=fermi_diff_order,
                partition=fermi_partition,
            ),
        )

    @property
    def requires_iterations(self) -> bool:
        """
        Whether the selected tight-binding method requires SCF iterations.

        GFN0-xTB is a non-self-consistent method. This property is derived
        solely from :attr:`method`.
        """
        return self.method != labels.GFN0_XTB

    def info(self) -> dict[str, Any]:
        """
        Return a dictionary with the SCF configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary with the SCF configuration.
        """
        return {
            "SCF Options": {
                "TB Method": labels.GFN_XTB_MAP[self.method],
                "Guess Method": labels.GUESS_MAP[self.guess],
                "SCF Mode": labels.SCF_MODE_MAP[self.scf_mode],
                "SCP Mode": labels.SCP_MODE_MAP[self.scp_mode],
                "Maxiter": self.maxiter,
                "Mixer": labels.MIXER_MAP[self.mixer],
                "Damping Factor": self.damp,
                "Force Convergence": self.force_convergence,
                "x tolerance": self.x_atol,
                "f(x) tolerance": self.f_atol,
                **self.fermi.info(),
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
