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


def test_legacy_scf_batch_guard_and_maxiter_zero_and_single_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batched SelfConsistentFieldFull rejects before map evaluation for maxiter > 0."""
    from tad_mctc.batch import pack

    from dxtb import GFN1_XTB, Calculator
    from dxtb._src.scf.guess import get_guess
    from dxtb._src.scf.iterator import get_refocc

    # 1 & 2: Batched maxiter > 0 raises NotImplementedError before _fcn(guess).
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
    overlap_batch = calc_batch.integrals.build_overlap(positions_batch)
    hcore_batch = calc_batch.integrals.build_hcore(
        positions_batch, with_overlap=True, charge=charge_batch
    )
    integrals_batch = IntegralMatrices(hcore=hcore_batch, overlap=overlap_batch)
    refocc_batch = calc_batch.integrals.hcore.refocc
    n0_b, occ_b, nel_b = get_refocc(
        refocc_batch, charge_batch, spin_batch, calc_batch.ihelp
    )
    cache_batch = calc_batch.interactions.get_cache(
        numbers=numbers_batch, positions=positions_batch, ihelp=calc_batch.ihelp
    )
    engine_batch = SelfConsistentFieldFull(
        calc_batch.interactions,
        occ_b,
        n0_b,
        numbers=numbers_batch,
        ihelp=calc_batch.ihelp,
        cache=cache_batch,
        integrals=integrals_batch,
        config=calc_batch.opts.scf,
        nel=nel_b,
    )
    guess_batch = engine_batch._guess(
        get_guess(
            numbers_batch,
            positions_batch,
            charge_batch,
            calc_batch.ihelp,
            calc_batch.opts.scf.guess,
        )
    )
    fcn_calls = 0
    orig_fcn = engine_batch.iterate_charges

    def counting_fcn(x: torch.Tensor) -> torch.Tensor:
        nonlocal fcn_calls
        fcn_calls += 1
        return orig_fcn(x)

    monkeypatch.setattr(engine_batch, "iterate_charges", counting_fcn)
    with pytest.raises(
        NotImplementedError,
        match="Batched SCF execution on SelfConsistentFieldFull is retired",
    ):
        engine_batch.scf(guess_batch)
    assert fcn_calls == 0

    # 3: Batched maxiter == 0 still evaluates the single initial map step.
    config_zero = ConfigSCF.create(
        mixer=labels.MIXER_LINEAR, maxiter=0, scp_mode=labels.SCP_MODE_CHARGE
    )
    engine_zero = SelfConsistentFieldFull(
        calc_batch.interactions,
        occ_b,
        n0_b,
        numbers=numbers_batch,
        ihelp=calc_batch.ihelp,
        cache=cache_batch,
        integrals=integrals_batch,
        config=config_zero,
        nel=nel_b,
    )
    zero_calls = 0
    orig_zero_fcn = engine_zero.iterate_charges

    def counting_zero_fcn(x: torch.Tensor) -> torch.Tensor:
        nonlocal zero_calls
        zero_calls += 1
        return orig_zero_fcn(x)

    monkeypatch.setattr(engine_zero, "iterate_charges", counting_zero_fcn)
    guess_zero = engine_zero._guess(
        get_guess(
            numbers_batch,
            positions_batch,
            charge_batch,
            calc_batch.ihelp,
            config_zero.guess,
        )
    )
    charges_zero = engine_zero.scf(guess_zero, return_charges=True)
    assert zero_calls == 1
    assert charges_zero.mono.shape == (2, 4)

    # 4: Supported single-System (batch_mode == 0, maxiter > 0) behavior is unchanged.
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
    overlap_single = calc_single.integrals.build_overlap(positions_single)
    hcore_single = calc_single.integrals.build_hcore(
        positions_single, with_overlap=True, charge=charge_single
    )
    integrals_single = IntegralMatrices(
        hcore=hcore_single, overlap=overlap_single
    )
    n0_s, occ_s, nel_s = get_refocc(
        calc_single.integrals.hcore.refocc,
        charge_single,
        spin_single,
        calc_single.ihelp,
    )
    cache_single = calc_single.interactions.get_cache(
        numbers=numbers_single,
        positions=positions_single,
        ihelp=calc_single.ihelp,
    )
    engine_single = SelfConsistentFieldFull(
        calc_single.interactions,
        occ_s,
        n0_s,
        numbers=numbers_single,
        ihelp=calc_single.ihelp,
        cache=cache_single,
        integrals=integrals_single,
        config=calc_single.opts.scf,
        nel=nel_s,
    )
    guess_single = engine_single._guess(
        get_guess(
            numbers_single,
            positions_single,
            charge_single,
            calc_single.ihelp,
            calc_single.opts.scf.guess,
        )
    )
    charges_single = engine_single.scf(guess_single, return_charges=True)
    ref_result = calc_single.singlepoint(positions_single)
    assert ref_result.charges is not None
    torch.testing.assert_close(
        charges_single.mono, ref_result.charges.mono, atol=1e-10, rtol=1e-10
    )
