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

import pytest
import torch

from dxtb import GFN1_XTB, Calculator, scf
from dxtb._src.integral.evaluation import (
    IntegralMatrices,
    build_integral_matrices,
)
from dxtb._src.scf.iterator import solve
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


def test_scf_solve_rejects_batched_input_and_runs_single_system() -> None:
    """Direct single-System SCF solve rejects batched inputs and runs single Systems."""
    from tad_mctc.batch import pack

    numbers_batch = pack([torch.tensor([1, 1]), torch.tensor([1, 1])])
    positions_batch = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
        ],
        dtype=torch.float64,
    )
    calc_batch = Calculator(
        numbers_batch,
        GFN1_XTB,
        dtype=torch.float64,
        opts={"verbosity": 0, "mixer": "linear", "maxiter": 5},
        batch_mode=2,
    )
    charge_batch = torch.zeros((2, 1), dtype=torch.float64)
    spin_batch = torch.zeros_like(charge_batch)
    integrals_batch = IntegralMatrices(
        hcore=torch.zeros((2, 2, 2), dtype=torch.float64),
        overlap=torch.eye(2, dtype=torch.float64).expand(2, 2, 2),
    )
    refocc_batch = torch.ones((2, 2), dtype=torch.float64)
    cache_batch = calc_batch.interactions.get_cache(
        numbers=numbers_batch, positions=positions_batch, ihelp=calc_batch.ihelp
    )
    with pytest.raises(
        NotImplementedError,
        match="single-System SCF only",
    ):
        solve(
            numbers_batch,
            positions_batch,
            charge_batch,
            spin_batch,
            calc_batch.interactions,
            cache_batch,
            calc_batch.ihelp,
            calc_batch.opts.scf,
            integrals_batch,
            refocc_batch,
        )

    numbers_single = torch.tensor([8, 1, 1])
    positions_single = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.1], [0.0, -1.4, 1.1]],
        dtype=torch.float64,
    )
    calc_single = Calculator(
        numbers_single,
        GFN1_XTB,
        dtype=torch.float64,
        opts={
            "verbosity": 0,
            "mixer": "anderson",
            "scp_mode": "charge",
            "maxiter": 30,
        },
    )
    charge_single = torch.zeros((1,), dtype=torch.float64)
    spin_single = torch.zeros_like(charge_single)
    integrals_single, refocc_single, _ = build_integral_matrices(
        calc_single.system.integral_setup,
        calc_single.system.h0_setup,
        positions_single,
        charge_single,
    )
    cache_single = calc_single.interactions.get_cache(
        numbers=numbers_single,
        positions=positions_single,
        ihelp=calc_single.ihelp,
    )
    solved = solve(
        numbers_single,
        positions_single,
        charge_single,
        spin_single,
        calc_single.interactions,
        cache_single,
        calc_single.ihelp,
        calc_single.opts.scf,
        integrals_single,
        refocc_single,
    )
    ref_result = calc_single.singlepoint(positions_single)
    assert ref_result.charges is not None
    torch.testing.assert_close(
        solved.charges.mono, ref_result.charges.mono, atol=1e-10, rtol=1e-10
    )
