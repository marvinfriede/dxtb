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

"""Public SCF strategies and numerical method settings.

Strategies are immutable ordered method stages. Each stage selects its own
representation, and the controller applies common physical-root
certification independently of the method.

Example
-------
>>> from dxtb.config import Config
>>> config = Config.create(strategy="anderson", differentiation="implicit")
>>> config.scf.strategy.stages[0].representation.value
'potential'

Raises
------
ValueError
    if a strategy preset or its combination with the selected execution mode
    is invalid.
"""

from __future__ import annotations

from dxtb._src.calculators.config.public_scf import (
    SCFConvergenceProfile,
    SCFDifferentiation,
)
from dxtb._src.scf.functional.anderson import (
    AndersonMethod,
    AndersonMethodSettings,
)
from dxtb._src.scf.functional.broyden import (
    BroydenMethod,
    BroydenMethodSettings,
)
from dxtb._src.scf.functional.cdiis import CDIISMethod, CDIISMethodSettings
from dxtb._src.scf.functional.method import LinearMethod
from dxtb._src.scf.functional.problem import SCFRepresentation
from dxtb._src.scf.functional.response import SCFResponseSettings
from dxtb._src.scf.functional.soscf import SOSCFMethod, SOSCFMethodSettings
from dxtb._src.scf.functional.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)

__all__ = [
    "AndersonMethod",
    "AndersonMethodSettings",
    "BroydenMethod",
    "BroydenMethodSettings",
    "CDIISMethod",
    "CDIISMethodSettings",
    "LinearMethod",
    "SCFConvergenceProfile",
    "SCFDifferentiation",
    "SCFRepresentation",
    "SCFResponseSettings",
    "SCFStage",
    "SCFStrategy",
    "SCFSwitchPolicy",
    "SOSCFMethod",
    "SOSCFMethodSettings",
    "strategy_preset",
]


def strategy_preset(name: str) -> SCFStrategy:
    """Construct one of the stable named SCF strategy presets.

    Presets contain explicit methods and representations. They never choose a
    solver based on tensor values, temperature, or autodiff state.
    """
    key = name.casefold().replace("_", "-")
    anderson = AndersonMethod(
        AndersonMethodSettings(
            damp_init=0.1,
            generations=5,
            soft_start=True,
            diagonal_offset=0.01,
        )
    )
    broyden = BroydenMethod(BroydenMethodSettings(max_rank=8))
    cdiis = CDIISMethod(CDIISMethodSettings(max_history=8))
    soscf = SOSCFMethod(SOSCFMethodSettings())
    potential = SCFRepresentation.POTENTIAL
    fock = SCFRepresentation.FOCK

    presets = {
        "linear": SCFStrategy(
            (SCFStage(LinearMethod(), representation=potential),)
        ),
        "anderson": SCFStrategy(
            (SCFStage(anderson, representation=potential),)
        ),
        "broyden": SCFStrategy((SCFStage(broyden, representation=potential),)),
        "cdiis": SCFStrategy((SCFStage(cdiis, representation=fock),)),
        "broyden-cdiis": SCFStrategy(
            (
                SCFStage(
                    broyden,
                    representation=potential,
                    switch_policy=SCFSwitchPolicy(fixed_point_l2=1.0e-2),
                ),
                SCFStage(cdiis, representation=fock),
            )
        ),
        "anderson-cdiis": SCFStrategy(
            (
                SCFStage(
                    anderson,
                    representation=potential,
                    switch_policy=SCFSwitchPolicy(fixed_point_l2=1.0e-2),
                ),
                SCFStage(cdiis, representation=fock),
            )
        ),
        "cdiis-soscf": SCFStrategy(
            (
                SCFStage(
                    cdiis,
                    representation=fock,
                    switch_policy=SCFSwitchPolicy(stationarity=1.0e-3),
                ),
                SCFStage(soscf, representation=fock),
            )
        ),
        "broyden-cdiis-soscf": SCFStrategy(
            (
                SCFStage(
                    broyden,
                    representation=potential,
                    switch_policy=SCFSwitchPolicy(fixed_point_l2=1.0e-2),
                ),
                SCFStage(
                    cdiis,
                    representation=fock,
                    switch_policy=SCFSwitchPolicy(stationarity=1.0e-3),
                ),
                SCFStage(soscf, representation=fock),
            )
        ),
    }
    try:
        return presets[key]
    except KeyError as exc:
        choices = ", ".join(repr(item) for item in presets)
        raise ValueError(
            f"Unknown SCF strategy preset {name!r}. Choose from {choices}."
        ) from exc
