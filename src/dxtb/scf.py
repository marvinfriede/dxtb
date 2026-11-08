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
variable, and the controller applies common physical-root
certification independently of the method.

Example
-------
>>> from dxtb.config import Config
>>> config = Config.create(strategy="anderson", differentiation="implicit")
>>> config.scf.strategy.stages[0].variable.value
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
    AndersonConverger,
    AndersonConvergerSettings,
)
from dxtb._src.scf.functional.broyden import (
    BroydenConverger,
    BroydenConvergerSettings,
)
from dxtb._src.scf.functional.cdiis import CDIISConverger, CDIISConvergerSettings
from dxtb._src.scf.functional.method import LinearConverger
from dxtb._src.scf.functional.problem import SCFVariable
from dxtb._src.scf.functional.response import CPSCFSettings
from dxtb._src.scf.functional.soscf import SOSCFConverger
from dxtb._src.scf.functional.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)

__all__ = [
    "AndersonConverger",
    "AndersonConvergerSettings",
    "BroydenConverger",
    "BroydenConvergerSettings",
    "CDIISConverger",
    "CDIISConvergerSettings",
    "CPSCFSettings",
    "LinearConverger",
    "SCFConvergenceProfile",
    "SCFDifferentiation",
    "SCFStage",
    "SCFStrategy",
    "SCFSwitchPolicy",
    "SCFVariable",
    "SOSCFConverger",
    "strategy_preset",
]


def strategy_preset(name: str) -> SCFStrategy:
    """Construct one of the stable named SCF strategy presets.

    Presets contain explicit methods and variables. They never choose a
    solver based on tensor values, temperature, or autodiff state.
    """
    key = name.casefold().replace("_", "-")
    anderson = AndersonConverger(
        AndersonConvergerSettings(
            damp_init=0.1,
            generations=5,
            soft_start=True,
            diagonal_offset=0.01,
        )
    )
    broyden = BroydenConverger(BroydenConvergerSettings(max_rank=8))
    cdiis = CDIISConverger(CDIISConvergerSettings(max_history=8))
    soscf = SOSCFConverger()
    potential = SCFVariable.POTENTIAL
    fock = SCFVariable.FOCK

    presets = {
        "linear": SCFStrategy(
            (SCFStage(LinearConverger(), variable=potential),)
        ),
        "anderson": SCFStrategy(
            (SCFStage(anderson, variable=potential),)
        ),
        "broyden": SCFStrategy((SCFStage(broyden, variable=potential),)),
        "cdiis": SCFStrategy((SCFStage(cdiis, variable=fock),)),
        "broyden-cdiis": SCFStrategy(
            (
                SCFStage(
                    broyden,
                    variable=potential,
                    switch_policy=SCFSwitchPolicy(fixed_point_l2=1.0e-2),
                ),
                SCFStage(cdiis, variable=fock),
            )
        ),
        "anderson-cdiis": SCFStrategy(
            (
                SCFStage(
                    anderson,
                    variable=potential,
                    switch_policy=SCFSwitchPolicy(fixed_point_l2=1.0e-2),
                ),
                SCFStage(cdiis, variable=fock),
            )
        ),
        "cdiis-soscf": SCFStrategy(
            (
                SCFStage(
                    cdiis,
                    variable=fock,
                    switch_policy=SCFSwitchPolicy(stationarity=1.0e-3),
                ),
                SCFStage(soscf, variable=fock),
            )
        ),
        "broyden-cdiis-soscf": SCFStrategy(
            (
                SCFStage(
                    broyden,
                    variable=potential,
                    switch_policy=SCFSwitchPolicy(fixed_point_l2=1.0e-2),
                ),
                SCFStage(
                    cdiis,
                    variable=fock,
                    switch_policy=SCFSwitchPolicy(stationarity=1.0e-3),
                ),
                SCFStage(soscf, variable=fock),
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
