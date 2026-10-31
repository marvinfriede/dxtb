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
    functional_converged_and_valid,
    functional_subsolves_ok,
    setup_simple_solver_settings,
    solve_scf_simple,
)
from dxtb._src.wavefunction import filling

from .test_functional_map import _prepare_case


def test_positive_subcutoff_temperature_matches_low_temperature_entropy() -> (
    None
):
    """Sub-cutoff free energy matches the independent Aufbau entropy."""
    cutoff_kelvin = 3.0e-7 / KELVIN2AU
    temperature = 0.5 * cutoff_kelvin
    options = {
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 0,
        "fermi_etemp": temperature,
        "fermi_partition": labels.FERMI_PARTITION_EQUAL,
    }
    case = _prepare_case(GFN1_XTB, charge=0.3, scf_options=options)
    actual = solve_scf_simple(
        case.x,
        case.inputs,
        setup_simple_solver_settings(
            case.system.config.scf, case.positions.dtype
        ),
    )
    settings = case.inputs.fermi_settings
    assert settings is not None
    assert settings.smear is False
    expected_orbital = filling.get_fermi_entropy_low_temperature(
        case.inputs.occupation, settings.kt
    )
    expected_total = expected_orbital.sum()

    assert torch.count_nonzero(expected_total) > 0
    torch.testing.assert_close(
        actual.fenergy.sum(), expected_total, rtol=0.0, atol=2.0e-13
    )
    torch.testing.assert_close(
        actual.fenergy,
        expected_total.expand_as(actual.fenergy) / case.system.numbers.numel(),
        rtol=0.0,
        atol=2.0e-13,
    )


def test_subcutoff_entropy_fractional_charge_gradient_matches_reference() -> (
    None
):
    """Sub-cutoff entropy has the reference fractional-charge derivative."""
    cutoff_kelvin = 3.0e-7 / KELVIN2AU
    options = {
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 0,
        "fermi_etemp": 0.5 * cutoff_kelvin,
        "fermi_partition": labels.FERMI_PARTITION_EQUAL,
    }
    charge = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    case = _prepare_case(GFN1_XTB, charge=charge, scf_options=options)
    actual = solve_scf_simple(
        case.x,
        case.inputs,
        setup_simple_solver_settings(
            case.system.config.scf, case.positions.dtype
        ),
    )
    settings = case.inputs.fermi_settings
    assert settings is not None
    assert settings.smear is False
    actual_gradient = torch.autograd.grad(
        actual.fenergy.sum(), charge, retain_graph=True
    )[0]
    expected = filling.get_fermi_entropy_low_temperature(
        case.inputs.occupation, settings.kt
    ).sum()
    expected_gradient = torch.autograd.grad(expected, charge)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=0.0, atol=2.0e-13
    )


def test_temperature_cutoff_boundaries_preserve_spectral_contract() -> None:
    """Exact zero, sub-cutoff, and smeared temperatures keep their contracts."""
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
            assert result.trajectory_status is None
            assert result.final_evaluation.fermi_status is None
            assert functional_subsolves_ok(result).item() is True
            assert functional_converged_and_valid(result).item() is False
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
                assert result.trajectory_status is None
                assert result.final_evaluation.fermi_status is None
                assert functional_subsolves_ok(result).item() is True
                assert settings is not None
                expected_entropy = filling.get_fermi_entropy_low_temperature(
                    case.inputs.occupation, settings.kt
                ).sum()
                torch.testing.assert_close(
                    result.fenergy.sum(),
                    expected_entropy,
                    rtol=0.0,
                    atol=2.0e-13,
                )


def test_raw_fixed_point_and_fermi_status_remain_separate() -> None:
    """Raw fixed-point progress is distinct from Fermi validity/certification."""
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
    certification = result.diagnostics.certification.result
    assert certification.fixed_point_ok.item() is True
    assert certification.fermi_ok.item() is False
    assert certification.certified.item() is False
    assert result.converged.item() is False
    assert status is not None
    assert not bool(status.converged.all())
    assert functional_subsolves_ok(result).item() is False
    assert functional_converged_and_valid(result).item() is False
