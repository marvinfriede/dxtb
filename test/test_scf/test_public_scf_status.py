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
"""Public functional SCF status schema tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest
import torch
from torch.func import vmap

from dxtb import SCFFailureCode
from dxtb._src.scf.functional.eager import (
    _check_subsolve_status,
    public_scf_status,
)
from dxtb._src.scf.functional.solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_INVALID_TEMPERATURE,
    FERMI_FAILURE_NONE,
    FERMI_FAILURE_NONCONVERGENCE,
    FERMI_FAILURE_TOO_MANY_ELECTRONS,
    SCFTrajectoryStatus,
    accumulate_trajectory_status,
    fermi_failure_code,
)
from dxtb._src.scf.status import SCFStatus
from dxtb._src.wavefunction.filling import FermiStatus
from dxtb._src.calculators.singlepoint import _normalize_scf_output
from dxtb._src.scf.functional.data import SCFCharges, SCFPotential


def _fermi_status(
    *,
    converged: bool = True,
    invalid_temperature: bool = False,
    too_many_electrons: bool = False,
    electron_count_ok: bool = True,
) -> FermiStatus:
    return FermiStatus(
        converged=torch.tensor(converged),
        residual=torch.tensor(0.0),
        invalid_temperature=torch.tensor(invalid_temperature),
        too_many_electrons=torch.tensor(too_many_electrons),
        electron_count_ok=torch.tensor(electron_count_ok),
    )


def test_public_failure_enum_and_status_value() -> None:
    """Expose the stable public values in a frozen two-field value."""
    assert {
        name: int(code) for name, code in SCFFailureCode.__members__.items()
    } == {
        "NONE": 0,
        "INVALID_TEMPERATURE": 1,
        "TOO_MANY_ELECTRONS": 2,
        "FERMI_NONCONVERGENCE": 3,
        "ELECTRON_COUNT": 4,
    }

    trajectory = torch.tensor(2, dtype=torch.int64)
    final = torch.tensor(3, dtype=torch.int64)
    status = SCFStatus(trajectory, final)
    assert [field.name for field in fields(status)] == [
        "trajectory_failure",
        "final_failure",
    ]
    assert status.trajectory_failure is trajectory
    assert status.final_failure is final
    assert status.trajectory_failure.dtype == torch.int64
    assert status.final_failure.dtype == torch.int64
    assert status.trajectory_failure.device == trajectory.device
    with pytest.raises(FrozenInstanceError):
        status.final_failure = trajectory  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        status.trajectory_failure = final  # type: ignore[misc]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (_fermi_status(), SCFFailureCode.NONE),
        (
            _fermi_status(invalid_temperature=True),
            SCFFailureCode.INVALID_TEMPERATURE,
        ),
        (
            _fermi_status(too_many_electrons=True),
            SCFFailureCode.TOO_MANY_ELECTRONS,
        ),
        (
            _fermi_status(converged=False),
            SCFFailureCode.FERMI_NONCONVERGENCE,
        ),
        (
            _fermi_status(electron_count_ok=False),
            SCFFailureCode.ELECTRON_COUNT,
        ),
    ],
)
def test_internal_failure_projection_matches_public_enum(
    status: FermiStatus, expected: SCFFailureCode
) -> None:
    """Map each actual internal Fermi status to its public category."""
    code = fermi_failure_code(status, like=torch.zeros(()))
    assert code.dtype == torch.int64
    assert code.device == status.converged.device
    assert code.item() == int(expected)


def test_failure_precedence_is_preserved() -> None:
    """Earlier eager categories retain priority when flags overlap."""
    cases = [
        (
            _fermi_status(
                converged=False,
                invalid_temperature=True,
                too_many_electrons=True,
                electron_count_ok=False,
            ),
            SCFFailureCode.INVALID_TEMPERATURE,
        ),
        (
            _fermi_status(
                converged=False,
                too_many_electrons=True,
                electron_count_ok=False,
            ),
            SCFFailureCode.TOO_MANY_ELECTRONS,
        ),
        (
            _fermi_status(converged=False, electron_count_ok=False),
            SCFFailureCode.FERMI_NONCONVERGENCE,
        ),
    ]
    for status, expected in cases:
        assert fermi_failure_code(status, like=torch.zeros(())).item() == int(
            expected
        )


def test_public_projection_keeps_trajectory_and_final_independent() -> None:
    """Keep chronological trajectory and final-spectrum codes separate."""
    like = torch.zeros((), dtype=torch.float64)
    trajectory = SCFTrajectoryStatus(
        fermi_converged=torch.tensor(False),
        electron_count_ok=torch.tensor(True),
        invalid_temperature=torch.tensor(False),
        too_many_electrons=torch.tensor(False),
        first_failure=torch.tensor(
            SCFFailureCode.FERMI_NONCONVERGENCE, dtype=torch.int64
        ),
    )
    absent = public_scf_status(None, None, like=like)
    assert absent.trajectory_failure.item() == int(SCFFailureCode.NONE)
    assert absent.final_failure.item() == int(SCFFailureCode.NONE)
    first_only = public_scf_status(trajectory, None, like=like)
    assert first_only.trajectory_failure.item() == int(
        SCFFailureCode.FERMI_NONCONVERGENCE
    )
    assert first_only.final_failure.item() == int(SCFFailureCode.NONE)

    final_only = public_scf_status(
        None, _fermi_status(electron_count_ok=False), like=like
    )
    assert final_only.trajectory_failure.item() == int(SCFFailureCode.NONE)
    assert final_only.final_failure.item() == int(SCFFailureCode.ELECTRON_COUNT)

    both = public_scf_status(
        SCFTrajectoryStatus(
            fermi_converged=torch.tensor(False),
            electron_count_ok=torch.tensor(True),
            invalid_temperature=torch.tensor(False),
            too_many_electrons=torch.tensor(True),
            first_failure=torch.tensor(
                SCFFailureCode.TOO_MANY_ELECTRONS, dtype=torch.int64
            ),
        ),
        _fermi_status(converged=False),
        like=like,
    )
    assert both.trajectory_failure.item() == int(
        SCFFailureCode.TOO_MANY_ELECTRONS
    )
    assert both.final_failure.item() == int(SCFFailureCode.FERMI_NONCONVERGENCE)
    assert both.trajectory_failure.dtype == torch.int64
    assert both.final_failure.device == like.device


def test_outer_nonconvergence_is_not_a_fermi_failure() -> None:
    """Valid subsolves retain zero codes when the outer solve is unfinished."""
    outer_converged = torch.tensor(False)
    status = public_scf_status(None, None, like=outer_converged)
    assert not outer_converged
    assert status.trajectory_failure.item() == int(SCFFailureCode.NONE)
    assert status.final_failure.item() == int(SCFFailureCode.NONE)


def test_public_projection_is_tensor_transform_compatible() -> None:
    """Status categories remain tensor values under a mapped projection."""

    def project(flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        status = FermiStatus(
            converged=~flags[0],
            residual=flags.new_zeros(()),
            invalid_temperature=flags[1],
            too_many_electrons=flags[2],
            electron_count_ok=~flags[3],
        )
        public = public_scf_status(None, status, like=flags[0])
        return public.trajectory_failure, public.final_failure

    flags = torch.tensor(
        [[False, False, False, False], [True, True, True, True]]
    )
    trajectory, final = vmap(project)(flags)
    assert trajectory.shape == torch.Size([2])
    assert final.shape == torch.Size([2])
    assert trajectory.dtype == torch.int64
    assert final.dtype == torch.int64
    assert torch.equal(trajectory, torch.zeros(2, dtype=torch.int64))
    assert torch.equal(
        final,
        torch.tensor(
            [SCFFailureCode.NONE, SCFFailureCode.INVALID_TEMPERATURE],
            dtype=torch.int64,
        ),
    )


def test_trajectory_projection_preserves_first_active_failure() -> None:
    """Later and inactive raw maps do not replace the first failure."""
    active = torch.tensor(True)
    inactive = torch.tensor(False)
    good = _fermi_status()
    invalid = _fermi_status(invalid_temperature=True)
    too_many = _fermi_status(too_many_electrons=True)

    first = accumulate_trajectory_status(None, good, active)
    first = accumulate_trajectory_status(first, invalid, active)
    assert first is not None
    assert first.first_failure.item() == int(SCFFailureCode.INVALID_TEMPERATURE)
    first = accumulate_trajectory_status(first, too_many, active)
    assert first is not None
    assert first.first_failure.item() == int(SCFFailureCode.INVALID_TEMPERATURE)
    first = accumulate_trajectory_status(first, good, active)
    assert first is not None
    assert first.first_failure.item() == int(SCFFailureCode.INVALID_TEMPERATURE)

    inactive_first = accumulate_trajectory_status(None, invalid, inactive)
    assert inactive_first is not None
    assert inactive_first.first_failure.item() == int(SCFFailureCode.NONE)
    inactive_first = accumulate_trajectory_status(
        inactive_first, too_many, inactive
    )
    assert inactive_first.first_failure.item() == int(SCFFailureCode.NONE)


def test_eager_policy_checks_trajectory_before_final_spectrum() -> None:
    """Trajectory failure is interpreted before a later final failure."""
    trajectory = SCFTrajectoryStatus(
        fermi_converged=torch.tensor(False),
        electron_count_ok=torch.tensor(True),
        invalid_temperature=torch.tensor(False),
        too_many_electrons=torch.tensor(False),
        first_failure=torch.tensor(FERMI_FAILURE_NONCONVERGENCE),
    )
    fake_result = SimpleNamespace(
        trajectory_status=trajectory,
        final_evaluation=SimpleNamespace(
            fermi_status=_fermi_status(invalid_temperature=True)
        ),
        converged=torch.tensor(False),
    )
    with pytest.raises(RuntimeError, match="Fermi energy failed"):
        _check_subsolve_status(fake_result)  # type: ignore[arg-type]


def test_legacy_failure_constants_are_public_enum_aliases() -> None:
    """Keep internal constant imports while deriving values from the enum."""
    assert FERMI_FAILURE_NONE == int(SCFFailureCode.NONE)
    assert FERMI_FAILURE_INVALID_TEMPERATURE == int(
        SCFFailureCode.INVALID_TEMPERATURE
    )
    assert FERMI_FAILURE_TOO_MANY_ELECTRONS == int(
        SCFFailureCode.TOO_MANY_ELECTRONS
    )
    assert FERMI_FAILURE_NONCONVERGENCE == int(
        SCFFailureCode.FERMI_NONCONVERGENCE
    )
    assert FERMI_FAILURE_ELECTRON_COUNT == int(SCFFailureCode.ELECTRON_COUNT)


def test_legacy_output_normalization_keeps_status_absent() -> None:
    """Legacy dictionaries retain their historical status-less contract."""
    value = torch.zeros(())
    legacy = {
        "energy": value,
        "fenergy": value,
        "charges": SCFCharges(mono=value),
        "density": value,
        "coefficients": value,
        "emo": value,
        "occupation": value,
        "potential": SCFPotential(mono=value),
        "hamiltonian": value,
        "iterations": 0,
    }
    output = _normalize_scf_output(legacy, value)
    assert output.status is None
