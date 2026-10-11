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
"""Functional SCF temperature-cutoff compatibility tests."""

from __future__ import annotations

import torch
from tad_mctc.units.energy import KELVIN2AU

from dxtb import GFN1_XTB, labels
from dxtb._src.scf.functional import (
    functional_result_ok,
    setup_simple_solver_settings,
    solve_scf_simple,
)

from .test_functional_map import _prepare_case


def test_positive_subcutoff_temperature_preserves_legacy_free_energy() -> None:
    """Positive sub-cutoff kT keeps the legacy Aufbau entropy contribution."""
    cutoff_kelvin = 3.0e-7 / KELVIN2AU
    temperature = 0.5 * cutoff_kelvin
    options = {
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 0,
        "fermi_etemp": temperature,
    }
    legacy = _prepare_case(GFN1_XTB, charge=0.3, scf_options=options)
    functional = _prepare_case(GFN1_XTB, charge=0.3, scf_options=options)

    legacy.engine.scf(legacy.x, return_charges=False)
    expected = legacy.engine.get_electronic_free_energy()
    actual = solve_scf_simple(
        functional.x,
        functional.inputs,
        setup_simple_solver_settings(
            functional.system.config.scf, functional.positions.dtype
        ),
    )

    assert torch.count_nonzero(expected) > 0
    torch.testing.assert_close(actual.fenergy, expected, rtol=0.0, atol=0.0)


def test_subcutoff_fractional_charge_gradient_matches_legacy() -> None:
    """Positive sub-cutoff entropy keeps the fractional-charge derivative."""
    cutoff_kelvin = 3.0e-7 / KELVIN2AU
    options = {
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 0,
        "fermi_etemp": 0.5 * cutoff_kelvin,
    }
    charge = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    functional = _prepare_case(GFN1_XTB, charge=charge, scf_options=options)
    actual = solve_scf_simple(
        functional.x,
        functional.inputs,
        setup_simple_solver_settings(
            functional.system.config.scf, functional.positions.dtype
        ),
    )
    actual_gradient = torch.autograd.grad(
        (actual.energy + actual.fenergy).sum(), charge
    )[0]

    legacy = _prepare_case(GFN1_XTB, charge=charge, scf_options=options)
    raw = legacy.engine.scf(legacy.x, return_charges=False)
    charges = legacy.engine.converged_to_charges(raw)
    legacy_value = legacy.engine.get_energy(charges).sum()
    legacy_value = (
        legacy_value + legacy.engine.get_electronic_free_energy().sum()
    )
    expected_gradient = torch.autograd.grad(legacy_value, charge)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=0.0, atol=2.0e-13
    )


def test_temperature_cutoff_boundaries_preserve_legacy_split() -> None:
    """Only exact zero omits free-energy settings; cutoff equality smears."""
    cutoff = 3.0e-7
    cases = [
        (0.0, None),
        ((cutoff * (1.0 - 1.0e-8)) / KELVIN2AU, False),
        ((cutoff / KELVIN2AU), True),
        ((cutoff * (1.0 + 1.0e-8)) / KELVIN2AU, True),
    ]
    for temperature, expected_smear in cases:
        case = _prepare_case(
            GFN1_XTB,
            sample_name="H2",
            scf_options={
                "mixer": labels.MIXER_LINEAR,
                "fermi_etemp": temperature,
                "maxiter": 0,
            },
        )
        settings = case.inputs.fermi_settings
        if expected_smear is None:
            assert settings is None
            result = solve_scf_simple(
                case.x,
                case.inputs,
                setup_simple_solver_settings(
                    case.system.config.scf, case.positions.dtype
                ),
            )
            assert torch.count_nonzero(result.fenergy) == 0
        else:
            assert settings is not None
            assert settings.smear is expected_smear
            result = solve_scf_simple(
                case.x,
                case.inputs,
                setup_simple_solver_settings(
                    case.system.config.scf, case.positions.dtype
                ),
            )
            if expected_smear:
                assert result.map_result.fermi_status is not None
            else:
                assert result.map_result.fermi_status is None
            if not expected_smear:
                assert torch.count_nonzero(result.fenergy) > 0
                case.engine.scf(case.x, return_charges=False)
                torch.testing.assert_close(
                    result.fenergy,
                    case.engine.get_electronic_free_energy(),
                    rtol=0.0,
                    atol=0.0,
                )


def test_outer_convergence_and_fermi_status_remain_separate() -> None:
    """Outer SCF may converge while its selected inner Fermi search fails."""
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2O",
        charge=0.3,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "x_atol": 1.0e6,
            "x_atol_max": 1.0e6,
            "fermi_etemp": 300.0,
            "fermi_maxiter": 0,
            "fermi_thresh": 1.0e-30,
        },
    )
    result = solve_scf_simple(
        case.x,
        case.inputs,
        setup_simple_solver_settings(
            case.system.config.scf, case.positions.dtype
        ),
    )
    status = result.map_result.fermi_status
    assert result.converged.item() is True
    assert status is not None
    assert not bool(status.converged.all())
    assert functional_result_ok(result).item() is False
