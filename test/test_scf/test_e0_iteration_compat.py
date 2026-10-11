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
"""Compatibility semantics for full-mode SCF evaluation counts."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from dxtb import GFN1_XTB, Calculator, OutputHandler, Result
from dxtb._src.constants import labels
from dxtb._src.scf import base as scf_base
from dxtb._src.scf.mixer.simple import Simple

from ..utils import get_param_module
from .samples import samples

_CASES = [
    pytest.param(
        0,
        True,
        1,
        0,
        -5.7674748965126055,
        [
            0.2855700080025312,
            -0.6666666666666667,
            -0.3288230967414383,
            -0.02413374219667852,
            0.3979326924681822,
            -0.030905943667056898,
            0.39793269246818186,
            -0.030905943667056925,
        ],
        id="maxiter-0-mix-guess",
    ),
    pytest.param(
        0,
        False,
        1,
        0,
        -5.7674748965126055,
        [
            0.2855700080025312,
            -0.6666666666666667,
            -0.3288230967414383,
            -0.02413374219667852,
            0.3979326924681822,
            -0.030905943667056898,
            0.39793269246818186,
            -0.030905943667056925,
        ],
        id="maxiter-0-direct-guess",
    ),
    pytest.param(
        1,
        True,
        2,
        2,
        -5.768234535786348,
        [
            0.2939889013410317,
            -0.6666666666666681,
            -0.3220727381894495,
            -0.01158894430169033,
            0.384601936456171,
            -0.03143221254778239,
            0.3846019364561706,
            -0.031432212547782265,
        ],
        id="maxiter-1-mix-guess",
    ),
    pytest.param(
        1,
        False,
        2,
        1,
        -5.768064929714654,
        [
            0.3198050957629175,
            -0.6666666666666663,
            -0.3014181120840056,
            0.026781487968248863,
            0.34375343021101246,
            -0.03300433270125858,
            0.34375343021101257,
            -0.03300433270125874,
        ],
        id="maxiter-1-direct-guess",
    ),
    pytest.param(
        2,
        True,
        3,
        3,
        -5.768491792471992,
        [
            0.2989105466303572,
            -0.6666666666666672,
            -0.3181262384117811,
            -0.004256952521704216,
            0.37680647598313566,
            -0.03173682049823836,
            0.37680647598313544,
            -0.03173682049823819,
        ],
        id="maxiter-2-mix-guess",
    ),
    pytest.param(
        2,
        False,
        3,
        2,
        -5.768438426259488,
        [
            0.3137716574472518,
            -0.6666666666666672,
            -0.30622445307306334,
            0.01785019607036742,
            0.3532775921078922,
            -0.03264295899683744,
            0.3532775921078919,
            -0.032642958996837314,
        ],
        id="maxiter-2-direct-guess",
    ),
]


@pytest.mark.parametrize(
    "maxiter,mix_guess,map_evaluations,mixer_transitions,energy,charges",
    _CASES,
)
def test_full_mode_iteration_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    maxiter: int,
    mix_guess: bool,
    map_evaluations: int,
    mixer_transitions: int,
    energy: float,
    charges: list[float],
) -> None:
    """Lock map-evaluation counts and raw final-map outputs for E4."""
    counts = {"map": 0, "mixer": 0}
    original_map: Callable = scf_base.iterations.iterate_charges
    original_mixer: Callable = Simple.iter

    def count_map(*args, **kwargs):
        counts["map"] += 1
        return original_map(*args, **kwargs)

    def count_mixer(self, x_new, x_old=None):
        counts["mixer"] += 1
        return original_mixer(self, x_new, x_old)

    monkeypatch.setattr(scf_base.iterations, "iterate_charges", count_map)
    monkeypatch.setattr(Simple, "iter", count_mixer)

    sample = samples["H2O"]
    numbers = sample["numbers"].to(device="cpu")
    positions = sample["positions"].to(device="cpu", dtype=torch.double).clone()
    par = get_param_module("gfn1", device="cpu", dtype=torch.double)
    opts = {
        "verbosity": 0,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": labels.SCP_MODE_CHARGE,
        "mixer": "simple",
        "mix_guess": mix_guess,
        "damp": 0.25,
        "maxiter": maxiter,
        "f_atol": 1.0e-12,
        "x_atol": 1.0e-12,
        "x_atol_max": 1.0e-12,
        "force_convergence": False,
        "int_driver": labels.INTDRIVER_PYTORCH,
    }

    OutputHandler.clear_warnings()
    try:
        result = Calculator(
            numbers, par, opts=opts, device="cpu", dtype=torch.double
        ).singlepoint(positions)

        assert counts["map"] == map_evaluations
        assert result.iterations.item() == map_evaluations
        assert counts["mixer"] == mixer_transitions
        torch.testing.assert_close(
            result.energy.sum(),
            torch.tensor(energy, dtype=torch.double),
            rtol=0.0,
            atol=2.0e-12,
        )
        torch.testing.assert_close(
            result.charges.mono.reshape(-1),
            torch.tensor(charges, dtype=torch.double),
            rtol=0.0,
            atol=2.0e-12,
        )

        if maxiter == 0:
            assert not any(
                "does not converge" in msg for msg, _ in OutputHandler.warnings
            )
        else:
            assert any(
                f"does not converge after {maxiter} cycles" in msg
                for msg, _ in OutputHandler.warnings
            )
    finally:
        OutputHandler.clear_warnings()


def test_full_mode_maxiter_zero_ignores_force_convergence() -> None:
    """Legacy maxiter=0 returns one-shot output even when forcing convergence."""
    sample = samples["H2O"]
    numbers = sample["numbers"].to(device="cpu")
    positions = sample["positions"].to(device="cpu", dtype=torch.double).clone()
    par = get_param_module("gfn1", device="cpu", dtype=torch.double)
    OutputHandler.clear_warnings()
    try:
        result = Calculator(
            numbers,
            par,
            opts={
                "verbosity": 0,
                "scf_mode": labels.SCF_MODE_FULL,
                "scp_mode": labels.SCP_MODE_CHARGE,
                "mixer": "simple",
                "maxiter": 0,
                "force_convergence": True,
                "int_driver": labels.INTDRIVER_PYTORCH,
            },
            device="cpu",
            dtype=torch.double,
        ).singlepoint(positions)
        assert result.iterations.item() == 1
        assert not any(
            "does not converge" in msg for msg, _ in OutputHandler.warnings
        )
    finally:
        OutputHandler.clear_warnings()


def _full_result(
    *, f_atol: float, x_atol: float, x_atol_max: float
) -> tuple[Result, tuple[str, ...]]:
    sample = samples["H2O"]
    numbers = sample["numbers"].to(device="cpu")
    positions = sample["positions"].to(device="cpu", dtype=torch.double).clone()
    par = get_param_module("gfn1", device="cpu", dtype=torch.double)
    opts = {
        "verbosity": 0,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": labels.SCP_MODE_CHARGE,
        "mixer": "simple",
        "mix_guess": True,
        "damp": 0.25,
        "maxiter": 5,
        "f_atol": f_atol,
        "x_atol": x_atol,
        "x_atol_max": x_atol_max,
        "force_convergence": False,
        "int_driver": labels.INTDRIVER_PYTORCH,
    }
    OutputHandler.clear_warnings()
    try:
        result = Calculator(
            numbers, par, opts=opts, device="cpu", dtype=torch.double
        ).singlepoint(positions)
        warnings = tuple(msg for msg, _ in OutputHandler.warnings)
    finally:
        OutputHandler.clear_warnings()
    return result, warnings


def test_full_f_atol_does_not_change_convergence() -> None:
    """Full-mode stopping depends on x residual tolerances, not f_atol."""
    loose, loose_warnings = _full_result(
        f_atol=1.0e3, x_atol=1.0e-12, x_atol_max=1.0e-12
    )
    tight, tight_warnings = _full_result(
        f_atol=1.0e-12, x_atol=1.0e-12, x_atol_max=1.0e-12
    )

    assert loose.iterations.item() == tight.iterations.item() == 6
    torch.testing.assert_close(loose.energy, tight.energy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        loose.charges.mono, tight.charges.mono, rtol=0.0, atol=0.0
    )
    assert loose_warnings == tight_warnings
    assert any("does not converge" in warning for warning in tight_warnings)


def test_full_x_atol_controls_convergence() -> None:
    """The two raw-residual x tolerances determine full-mode stopping."""
    loose, loose_warnings = _full_result(
        f_atol=1.0e-8, x_atol=1.0e2, x_atol_max=1.0e2
    )
    tight, tight_warnings = _full_result(
        f_atol=1.0e-8, x_atol=1.0e-12, x_atol_max=1.0e-12
    )

    assert loose.iterations.item() == 2
    assert tight.iterations.item() == 6
    assert not any("does not converge" in warning for warning in loose_warnings)
    assert any("does not converge" in warning for warning in tight_warnings)
