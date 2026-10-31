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
General tests for SCF setup.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dxtb import labels, scf
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf.unrolling import SelfConsistentFieldFull
from dxtb.components.base import InteractionList
from dxtb.config import ConfigSCF


def test_invalid_public_strategy_and_differentiation_are_rejected() -> None:
    """The public API validates named strategy and derivative policy values."""
    with pytest.raises(ValueError, match="Unknown SCF strategy"):
        ConfigSCF.create(strategy="fail")
    with pytest.raises(ValueError, match="unrolled.*implicit"):
        ConfigSCF.create(differentiation="single-shot")


def test_legacy_mixer_constructor_is_not_part_of_core_config() -> None:
    """Mixer labels do not remain fields of immutable ConfigSCF."""
    with pytest.raises(TypeError):
        ConfigSCF(mixer=object())  # type: ignore[call-arg]


def test_packed_full_does_not_silently_substitute_cdiis() -> None:
    """The compatibility engine rejects a strategy it cannot represent."""
    dummy = torch.tensor([1], dtype=torch.float)
    kwargs = {
        "numbers": dummy,
        "occupation": dummy,
        "n0": dummy,
        "ihelp": SimpleNamespace(batch_mode=0),
        "cache": dummy,
        "integrals": IntegralMatrices(
            hcore=torch.eye(1, dtype=dummy.dtype),
            overlap=torch.eye(1, dtype=dummy.dtype),
        ),
    }
    config = ConfigSCF.create(strategy="cdiis")
    assert isinstance(config.strategy.stages[0].method, scf.CDIISMethod)
    assert config.initial_representation is scf.SCFRepresentation.FOCK
    with pytest.raises(NotImplementedError, match="legacy packed-SCF"):
        _ = SelfConsistentFieldFull(InteractionList(), **kwargs, config=config)
