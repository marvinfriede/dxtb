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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration and geometry guards for the private W2c2 diagnostic."""

from __future__ import annotations

import torch

from benchmarks.baseline.workloads import conformers
from benchmarks.e6c1_w2c2_implicit import (
    construct_workload,
    floating_parameter_counts,
    setup_model_system,
)
from dxtb import labels
from dxtb._src.constants import defaults


def test_w2c2_configuration_and_gradient_modes_match_frozen_workload() -> None:
    """Verify frozen conformers, W2 SCF settings, and explicit grad modes."""
    expected_numbers, expected_positions = conformers(
        "tmpda", 8, sigma=0.1, seed=0
    )
    for lane in range(8):
        numbers, positions = construct_workload(lane)
        assert torch.equal(numbers, expected_numbers[lane])
        assert torch.equal(positions, expected_positions[lane])

    numbers, _ = construct_workload(0)
    model, system = setup_model_system(
        numbers,
        maxiter=300,
        x_atol=1.0e-8,
        x_atol_max=defaults.X_ATOL_MAX,
        gradient_mode="positions-only",
    )
    scf = system.config.scf
    assert scf.scp_mode == labels.SCP_MODE_CHARGE
    assert scf.scf_mode == labels.SCF_MODE_FULL
    assert scf.mixer == labels.MIXER_ANDERSON
    assert scf.maxiter == 300
    assert scf.x_atol == 1.0e-8
    assert scf.x_atol_max == defaults.X_ATOL_MAX
    assert scf.f_atol == 1.0e-8
    assert scf.fermi.etemp == 300.0
    assert floating_parameter_counts(model)[1] == 0

    model_with_grad, _ = setup_model_system(
        numbers,
        maxiter=300,
        x_atol=1.0e-8,
        x_atol_max=defaults.X_ATOL_MAX,
        gradient_mode="positions-and-model",
    )
    total, trainable = floating_parameter_counts(model_with_grad)
    assert total > 0
    assert trainable == total
