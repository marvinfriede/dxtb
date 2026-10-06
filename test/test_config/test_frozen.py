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
Configurations are immutable values (B2): changes create a new object, equal
settings compare and hash equal, and the batch mode, device and data type are
not settings.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dxtb import GFN1_XTB, Calculator
from dxtb._src.constants import labels
from dxtb.config import Config


def test_frozen() -> None:
    cfg = Config.create()

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.grad = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.scf.mixer = labels.MIXER_LINEAR  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.scf.fermi.etemp = 1.0  # type: ignore[misc]


def test_replace_and_equality() -> None:
    cfg = Config.create(scf_mode="full", exclude="disp")
    assert cfg == Config.create(scf_mode=labels.SCF_MODE_FULL, exclude=["disp"])
    assert hash(cfg) == hash(Config.create(scf_mode="full", exclude=("disp",)))

    new = dataclasses.replace(
        cfg, scf=dataclasses.replace(cfg.scf, maxiter=7)
    )
    assert new.scf.maxiter == 7
    assert cfg.scf.maxiter != 7
    assert new != cfg


def test_no_batch_mode_device_dtype() -> None:
    names = {f.name for f in dataclasses.fields(Config)}
    names |= {f.name for f in dataclasses.fields(Config.create().scf)}
    assert not names & {"batch_mode", "device", "dtype"}


def test_batch_mode_follows_numbers() -> None:
    single = Calculator(torch.tensor([1, 1]), GFN1_XTB)
    assert single.ihelp.batch_mode == 0

    batch = torch.tensor([[1, 1], [1, 1]])
    assert Calculator(batch, GFN1_XTB).ihelp.batch_mode == 1
    # conformers without padding are requested by the option
    calc = Calculator(batch, GFN1_XTB, opts={"batch_mode": 2})
    assert calc.ihelp.batch_mode == 2
