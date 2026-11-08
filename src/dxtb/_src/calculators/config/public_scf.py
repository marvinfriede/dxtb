# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Strategy-based public SCF configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from dxtb._src.calculators.config.scf import _GUESS, ConfigFermi, _label
from dxtb._src.constants import defaults, labels


class SCFConvergenceProfile(str, Enum):
    """Named physical-root accuracy policies."""

    STANDARD = "standard"
    TIGHT = "tight"
    HIGHER_ORDER = "higher_order"


class SCFDifferentiation(str, Enum):
    """Derivative policy for a converged SCF result."""

    UNROLLED = "unrolled"
    IMPLICIT = "implicit"


def _preset(name: str):
    from dxtb.scf import strategy_preset

    return strategy_preset(name)


def _default_response():
    from dxtb._src.scf.functional.problem import SCFRepresentation
    from dxtb._src.scf.functional.response import SCFResponseSettings

    return SCFResponseSettings(representation=SCFRepresentation.POTENTIAL)


def _default_strategy():
    return _preset("anderson")


def _first_representation(strategy: Any):
    from dxtb._src.scf.functional.problem import SCFRepresentation

    stage = strategy.stages[0]
    if stage.representation is not None:
        return stage.representation
    required = getattr(stage.method, "required_representation", None)
    return SCFRepresentation.POTENTIAL if required is None else required


@dataclass(frozen=True, kw_only=True, eq=False)
class ConfigSCF:
    """Public SCF configuration with a single authoritative strategy.

    The core stores immutable stages, derivative policy, response policy, and
    named accuracy profile.

    Example
    -------
    >>> config = ConfigSCF.create(
    ...     strategy="broyden", differentiation="implicit"
    ... )
    >>> config.strategy.stages[0].representation.value
    'potential'

    Raises
    ------
    ValueError
        if a preset, convergence profile, or execution-policy combination is
        invalid.
    """

    method: int = defaults.METHOD
    guess: int = defaults.GUESS
    maxiter: int = defaults.MAXITER
    strategy: Any = field(default_factory=_default_strategy)
    differentiation: SCFDifferentiation = SCFDifferentiation.UNROLLED
    convergence: SCFConvergenceProfile = SCFConvergenceProfile.STANDARD
    response: Any = field(default_factory=_default_response)
    mix_guess: bool = defaults.MIX_GUESS
    damp: float = defaults.DAMP
    damp_dynamic: bool = defaults.DAMP_DYNAMIC
    damp_dynamic_factor: float = defaults.DAMP_DYNAMIC_FACTOR
    x_atol: float | None = None
    x_atol_max: float | None = None
    force_convergence: bool = defaults.SCF_FORCE_CONVERGENCE
    fermi: ConfigFermi = field(default_factory=ConfigFermi)

    def __post_init__(self) -> None:
        from dxtb._src.scf.functional.response import SCFResponseSettings
        from dxtb._src.scf.functional.strategy import SCFStrategy

        if not isinstance(self.strategy, SCFStrategy):
            raise TypeError("strategy must be an SCFStrategy.")
        if not isinstance(self.response, SCFResponseSettings):
            raise TypeError("response must be SCFResponseSettings.")
        if not isinstance(self.differentiation, SCFDifferentiation):
            try:
                self.__dict__["differentiation"] = SCFDifferentiation(
                    self.differentiation
                )
            except ValueError as exc:
                raise ValueError(
                    "differentiation must be 'unrolled' or 'implicit'."
                ) from exc
        if not isinstance(self.convergence, SCFConvergenceProfile):
            try:
                self.__dict__["convergence"] = SCFConvergenceProfile(
                    self.convergence
                )
            except ValueError as exc:
                raise ValueError(
                    "convergence must be 'standard', 'tight', or 'higher_order'."
                ) from exc

        thresholds = {
            SCFConvergenceProfile.STANDARD: (
                defaults.X_ATOL,
                defaults.X_ATOL_MAX,
            ),
            SCFConvergenceProfile.TIGHT: (1.0e-10, 1.0e-10),
            SCFConvergenceProfile.HIGHER_ORDER: (1.0e-12, 1.0e-12),
        }[self.convergence]
        if self.x_atol is None:
            self.__dict__["x_atol"] = thresholds[0]
        if self.x_atol_max is None:
            self.__dict__["x_atol_max"] = thresholds[1]

        initial = _first_representation(self.strategy)
        self.strategy.resolve_representations(initial)
        if self.differentiation is SCFDifferentiation.UNROLLED:
            eager_only = any(
                not getattr(stage.method, "supports_fixed_trip", True)
                for stage in self.strategy.stages
            )
            if len(self.strategy.stages) > 1 or eager_only:
                raise ValueError(
                    "This SCF strategy requires eager/implicit execution; "
                    "unrolled fixed-trip execution does not support its "
                    "dynamic or eager-only method stages."
                )

        if any(
            type(stage.method).__name__ == "SOSCFMethod"
            for stage in self.strategy.stages
        ):
            kt = float(self.fermi.etemp) * 3.166811563e-6
            if kt >= 3.0e-7:
                raise ValueError(
                    "SOSCF cannot be used with Fermi smearing at nonzero electronic temperature. Use "
                    "Anderson, Broyden, or CDIIS, or disable smearing explicitly."
                )

    @classmethod
    def create(
        cls,
        *,
        method: int = defaults.METHOD,
        guess: str | int = defaults.GUESS,
        maxiter: int = defaults.MAXITER,
        strategy: str | Any | None = None,
        differentiation: str | SCFDifferentiation | None = None,
        convergence: str | SCFConvergenceProfile = "standard",
        response: Any = None,
        mix_guess: bool = defaults.MIX_GUESS,
        damp: float = defaults.DAMP,
        damp_dynamic: bool = defaults.DAMP_DYNAMIC,
        damp_dynamic_factor: float = defaults.DAMP_DYNAMIC_FACTOR,
        x_atol: float | None = None,
        x_atol_max: float | None = None,
        force_convergence: bool = defaults.SCF_FORCE_CONVERGENCE,
        fermi_etemp: float | int = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: float | int | None = defaults.FERMI_THRESH,
        fermi_diff_order: int = defaults.FERMI_DIFF_ORDER,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
    ) -> ConfigSCF:
        """Create a ConfigSCF from a named preset or immutable strategy."""
        from dxtb.scf import SCFStrategy, strategy_preset

        if strategy is None:
            strategy_value = strategy_preset("anderson")
        elif isinstance(strategy, str):
            strategy_value = strategy_preset(strategy)
        elif isinstance(strategy, SCFStrategy):
            strategy_value = strategy
        else:
            raise TypeError("strategy must be a preset name or SCFStrategy.")

        if differentiation is None:
            differentiation = "unrolled"

        try:
            differentiation_value = SCFDifferentiation(differentiation)
        except ValueError as exc:
            raise ValueError(
                "differentiation must be 'unrolled' or 'implicit'."
            ) from exc
        try:
            convergence_value = SCFConvergenceProfile(convergence)
        except ValueError as exc:
            raise ValueError(
                "convergence must be 'standard', 'tight', or 'higher_order'."
            ) from exc
        return cls(
            method=method,
            guess=_label(guess, "guess", _GUESS, "Unknown guess method"),
            maxiter=maxiter,
            strategy=strategy_value,
            differentiation=differentiation_value,
            convergence=convergence_value,
            response=_default_response() if response is None else response,
            mix_guess=mix_guess,
            damp=damp,
            damp_dynamic=damp_dynamic,
            damp_dynamic_factor=damp_dynamic_factor,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
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
        return self.method != labels.GFN0_XTB

    @property
    def initial_representation(self):
        return _first_representation(self.strategy)

    def info(self) -> dict[str, Any]:
        """Return stable, address-free strategy and convergence diagnostics."""
        stage_details = []
        for index, stage in enumerate(self.strategy.stages):
            representation = (
                stage.representation
                or self.strategy.resolve_representations(
                    self.initial_representation
                )[index]
            )
            switch = (
                None
                if stage.switch_policy is None
                else {
                    key: value
                    for key, value in asdict(stage.switch_policy).items()
                    if value is not None
                }
            )
            method_settings = getattr(stage.method, "settings", None)
            stage_details.append(
                {
                    "Method": type(stage.method).__name__.removesuffix(
                        "Method"
                    ),
                    "Representation": representation.value,
                    "Settings": (
                        None
                        if method_settings is None
                        else asdict(method_settings)
                    ),
                    "Initial damping": stage.initial_damp,
                    "Switch policy": switch,
                }
            )
        strategy_label = " -> ".join(
            f"{stage['Method']} / {stage['Representation']}"
            for stage in stage_details
        )
        if not self.requires_iterations:
            strategy_label = "SCF not required"
        response_representation = (
            "primal"
            if self.response.representation is None
            else self.response.representation.value
        )
        return {
            "SCF Options": {
                "TB Method": labels.GFN_XTB_MAP[self.method],
                "Guess Method": labels.GUESS_MAP[self.guess],
                "Strategy": strategy_label,
                "Stages": stage_details,
                "Differentiation": self.differentiation.value,
                "Convergence profile": self.convergence.value,
                "Max iterations": self.maxiter,
                "Mix guess": self.mix_guess,
                "Damping": self.damp,
                "Dynamic damping": self.damp_dynamic,
                "Dynamic damping factor": self.damp_dynamic_factor,
                "Fixed-point L2 tolerance": self.x_atol,
                "Fixed-point max tolerance": self.x_atol_max,
                "Implicit response representation": response_representation,
                "Implicit response rtol": self.response.rtol,
                "Implicit response atol": self.response.atol,
                "Implicit response maxiter": self.response.maxiter,
                "Force convergence": self.force_convergence,
                **self.fermi.info(),
            }
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfigSCF):
            return NotImplemented
        return self.info() == other.info()

    def __hash__(self) -> int:
        return hash(
            json.dumps(self.info(), sort_keys=True, separators=(",", ":"))
        )
