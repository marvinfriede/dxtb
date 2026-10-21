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
"""Public System status integration and conformer-lane tests for E6a.2."""

from __future__ import annotations

import pytest
import torch

from dxtb import (
    GFN1_XTB,
    GFN2_XTB,
    Calculator,
    ParamModule,
    SCFFailureCode,
    labels,
)
from dxtb._src.calculators.model import Model
from dxtb.config import Config


def _system(
    *,
    method: object = GFN1_XTB,
    numbers: tuple[int, ...] = (1, 1),
    maxiter: int = 3,
    temperature: float = 0.0,
    fermi_maxiter: int = 200,
    fermi_thresh: float | None = None,
    force_convergence: bool = False,
):
    """Build an independent production GFN1/GFN2 PyTorch System."""
    dtype = torch.float64
    method_label = labels.GFN2_XTB if method is GFN2_XTB else labels.GFN1_XTB
    mixer = labels.MIXER_ANDERSON if method is GFN2_XTB else labels.MIXER_LINEAR
    scp_mode = (
        labels.SCP_MODE_FOCK if method is GFN2_XTB else labels.SCP_MODE_CHARGE
    )
    config = Config.create(
        method=method_label,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        mixer=mixer,
        scp_mode=scp_mode,
        maxiter=maxiter,
        force_convergence=force_convergence,
        fermi_etemp=temperature,
        fermi_maxiter=fermi_maxiter,
        fermi_thresh=fermi_thresh,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    return Model(par=ParamModule(method, dtype=dtype), config=config).setup(
        torch.tensor(numbers, dtype=torch.int64)
    )


def _positions() -> torch.Tensor:
    """Return three distinct, nonsingular H2 conformer geometries."""
    base = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    return torch.stack(
        [
            base + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, delta]])
            for delta in (-0.12, 0.0, 0.18)
        ]
    )


def _h2o_positions() -> torch.Tensor:
    """Return three distinct production GFN1/H2O geometries."""
    base = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]],
        dtype=torch.float64,
    )
    return torch.stack(
        [
            base
            + base.new_tensor(
                [[0.0, 0.0, 0.0], [delta, 0.0, 0.0], [-delta, 0.0, 0.0]]
            )
            for delta in (-0.04, 0.0, 0.06)
        ]
    )


def _assert_status_batch_matches_serial(mapped, serial) -> None:
    """Compare public Result values and both status fields lane by lane."""
    assert type(mapped) is type(serial[0])
    assert mapped.status is not None
    assert all(result.status is not None for result in serial)

    for name in ("energy", "fenergy", "iterations", "converged", "residual"):
        actual = getattr(mapped, name)
        expected = torch.stack([getattr(result, name) for result in serial])
        torch.testing.assert_close(actual, expected)

    assert mapped.status.trajectory_failure.dtype == torch.int64
    assert mapped.status.final_failure.dtype == torch.int64
    assert (
        mapped.status.trajectory_failure.device
        == serial[0].status.trajectory_failure.device
    )
    assert (
        mapped.status.final_failure.device
        == serial[0].status.final_failure.device
    )
    torch.testing.assert_close(
        mapped.status.trajectory_failure,
        torch.stack([result.status.trajectory_failure for result in serial]),
    )
    torch.testing.assert_close(
        mapped.status.final_failure,
        torch.stack([result.status.final_failure for result in serial]),
    )


@pytest.mark.parametrize("temperature", [0.0, 300.0], ids=["zero-k", "300-k"])
def test_public_status_for_valid_zero_and_finite_temperature(
    temperature: float,
) -> None:
    """Public status reports valid spectra for both supported temperatures."""
    system = _system(temperature=temperature)
    positions = _positions()
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_status_batch_matches_serial(mapped, serial)
    assert mapped.status is not None
    assert torch.all(
        mapped.status.trajectory_failure == int(SCFFailureCode.NONE)
    )
    assert torch.all(mapped.status.final_failure == int(SCFFailureCode.NONE))
    assert mapped.status.trajectory_failure.shape == (3,)
    assert mapped.status.trajectory_failure.device == positions.device
    assert mapped.converged is not None


def test_public_gfn2_status_maps_valid_spectra() -> None:
    """The accepted GFN2/Fock/Anderson path exposes status under vmap."""
    system = _system(method=GFN2_XTB, maxiter=2)
    positions = _positions()
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_status_batch_matches_serial(mapped, serial)
    assert mapped.status is not None
    assert torch.all(
        mapped.status.trajectory_failure == int(SCFFailureCode.NONE)
    )
    assert torch.all(mapped.status.final_failure == int(SCFFailureCode.NONE))


def test_public_gfn1_h2o_status_maps_outer_nonconvergence() -> None:
    """Larger GFN1 Systems expose valid spectrum codes beside outer status."""
    system = _system(numbers=(8, 1, 1), maxiter=1, temperature=300.0)
    positions = _h2o_positions()
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_status_batch_matches_serial(mapped, serial)
    assert mapped.status is not None
    assert torch.equal(
        mapped.status.trajectory_failure,
        torch.zeros(3, dtype=torch.int64),
    )
    assert torch.equal(
        mapped.status.final_failure,
        torch.zeros(3, dtype=torch.int64),
    )
    assert mapped.converged is not None
    assert not mapped.converged.any()
    assert torch.equal(
        mapped.iterations, torch.full((3,), 2, dtype=torch.int64)
    )


def test_public_outer_nonconvergence_is_not_a_subsolve_failure() -> None:
    """Positive fixed trips report outer failure separately from valid spectra."""
    system = _system(maxiter=1, force_convergence=True)
    positions = _positions()
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_status_batch_matches_serial(mapped, serial)
    assert mapped.status is not None
    assert torch.equal(
        mapped.status.trajectory_failure,
        torch.zeros(3, dtype=torch.int64),
    )
    assert torch.equal(
        mapped.status.final_failure,
        torch.zeros(3, dtype=torch.int64),
    )
    assert mapped.converged is not None
    assert not mapped.converged.any()
    assert torch.equal(
        mapped.iterations, torch.full((3,), 2, dtype=torch.int64)
    )


@pytest.mark.parametrize(
    ("maxiter", "temperature", "fermi_maxiter", "fermi_thresh", "final"),
    [
        (0, 0.0, 200, None, SCFFailureCode.NONE),
        (0, 300.0, 0, 0.0, SCFFailureCode.FERMI_NONCONVERGENCE),
    ],
    ids=["zero-trip-aufbau", "zero-trip-final-fermi-failure"],
)
def test_public_zero_trip_has_final_only_status(
    maxiter: int,
    temperature: float,
    fermi_maxiter: int,
    fermi_thresh: float | None,
    final: SCFFailureCode,
) -> None:
    """No physical trajectory is reported for a one-shot final evaluation."""
    system = _system(
        maxiter=maxiter,
        temperature=temperature,
        fermi_maxiter=fermi_maxiter,
        fermi_thresh=fermi_thresh,
        force_convergence=True,
    )
    positions = _positions()
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_status_batch_matches_serial(mapped, serial)
    assert mapped.status is not None
    assert torch.equal(
        mapped.status.trajectory_failure,
        torch.full((3,), int(SCFFailureCode.NONE), dtype=torch.int64),
    )
    assert torch.equal(
        mapped.status.final_failure,
        torch.full((3,), int(final), dtype=torch.int64),
    )
    assert mapped.iterations.shape == (3,)
    assert torch.equal(mapped.iterations, torch.ones(3, dtype=torch.int64))
    assert not mapped.converged.any()


def test_public_fermi_nonconvergence_is_tensor_status() -> None:
    """Exhausting the configured Fermi search is reported without raising."""
    system = _system(
        maxiter=2,
        temperature=300.0,
        fermi_maxiter=0,
        fermi_thresh=0.0,
    )
    positions = _positions()
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_status_batch_matches_serial(mapped, serial)
    assert mapped.status is not None
    expected = torch.full(
        (3,), int(SCFFailureCode.FERMI_NONCONVERGENCE), dtype=torch.int64
    )
    torch.testing.assert_close(mapped.status.trajectory_failure, expected)
    torch.testing.assert_close(mapped.status.final_failure, expected)


def test_public_too_many_electrons_status_is_lane_local() -> None:
    """One over-capacity lane does not alter neighboring status values."""
    system = _system(maxiter=2, temperature=300.0)
    positions = _positions()
    charges = torch.tensor([0.0, 0.0, -7.0], dtype=positions.dtype)
    spins = torch.tensor([0.0, 0.0, 1.0], dtype=positions.dtype)

    mapped = torch.func.vmap(
        lambda position, charge, spin: system.singlepoint(
            position, chrg=charge, spin=spin
        ),
        in_dims=(0, 0, 0),
    )(positions, charges, spins)
    serial = [
        system.singlepoint(position, chrg=charge, spin=spin)
        for position, charge, spin in zip(positions, charges, spins)
    ]

    _assert_status_batch_matches_serial(mapped, serial)
    assert mapped.status is not None
    expected = torch.tensor(
        [
            SCFFailureCode.NONE,
            SCFFailureCode.NONE,
            SCFFailureCode.TOO_MANY_ELECTRONS,
        ],
        dtype=torch.int64,
    )
    torch.testing.assert_close(mapped.status.trajectory_failure, expected)
    torch.testing.assert_close(mapped.status.final_failure, expected)
    assert mapped.converged is not None
    assert mapped.converged.shape == (3,)
    assert not mapped.converged[2]
    assert torch.isfinite(mapped.energy).all()


@pytest.mark.parametrize("maxiter", [0, 2], ids=["zero-trip", "iterative"])
def test_calculator_keeps_eager_fermi_failure_policy(maxiter: int) -> None:
    """Calculator raises on Fermi failure while System returns tensor status."""
    positions = _positions()[0]
    system = _system(
        maxiter=maxiter,
        temperature=300.0,
        fermi_maxiter=0,
        fermi_thresh=0.0,
    )
    result = system.singlepoint(positions)
    assert result.status is not None
    assert result.status.final_failure.item() == int(
        SCFFailureCode.FERMI_NONCONVERGENCE
    )

    calculator = Calculator(
        torch.tensor([1, 1]),
        GFN1_XTB,
        dtype=positions.dtype,
        opts={
            "verbosity": 0,
            "int_driver": labels.INTDRIVER_PYTORCH,
            "int_level": labels.INTLEVEL_HCORE,
            "maxiter": maxiter,
            "fermi_etemp": 300.0,
            "fermi_maxiter": 0,
            "fermi_thresh": 0.0,
        },
    )
    with pytest.raises(RuntimeError, match="Fermi energy failed to converge"):
        calculator.singlepoint(positions)
