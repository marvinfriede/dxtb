# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public System.singlepoint conformer-transform tests for E6a.0."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

from dxtb import GFN1_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb.config import Config


def _system(
    *,
    maxiter: int = 3,
    temperature: float = 0.0,
    intlevel: int | None = None,
    force_convergence: bool = False,
):
    """Construct an independent PyTorch-backed GFN1/H2 System."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=(labels.INTLEVEL_HCORE if intlevel is None else intlevel),
        mixer=labels.MIXER_LINEAR,
        maxiter=maxiter,
        force_convergence=force_convergence,
        fermi_etemp=temperature,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    return Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    ).setup(torch.tensor([1, 1]))


def _positions() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.35]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.65]],
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize(
    ("maxiter", "temperature"),
    [(0, 0.0), (3, 0.0), (3, 300.0)],
    ids=["zero-budget", "zero-k", "finite-temperature"],
)
def test_public_system_singlepoint_vmap_matches_serial_results(
    maxiter: int, temperature: float
) -> None:
    """The exact public Result tree maps over fixed-composition geometries."""
    system = _system(maxiter=maxiter, temperature=temperature)
    positions = _positions()

    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(value) for value in positions]

    assert type(mapped) is type(serial[0])
    mapped_leaves, mapped_spec = _pytree.tree_flatten(mapped)
    serial_leaves, serial_spec = _pytree.tree_flatten(serial[0])
    assert mapped_spec == serial_spec
    for index, (mapped_value, first_value) in enumerate(
        zip(mapped_leaves, serial_leaves)
    ):
        assert isinstance(mapped_value, torch.Tensor)
        assert all(isinstance(value, torch.Tensor) for value in serial_leaves)
        expected = torch.stack(
            [_pytree.tree_flatten(result)[0][index] for result in serial]
        )
        assert mapped_value.shape == expected.shape
        assert mapped_value.dtype == first_value.dtype
        assert mapped_value.device == first_value.device
        torch.testing.assert_close(mapped_value, expected)

    assert mapped.iterations.shape == (2,)
    assert mapped.converged is not None and mapped.converged.shape == (2,)
    assert mapped.status is not None
    assert mapped.status.trajectory_failure.shape == (2,)
    assert mapped.status.final_failure.shape == (2,)
    assert tuple(label for label, _ in mapped.classical) == tuple(
        label for label, _ in serial[0].classical
    )


def test_public_system_singlepoint_vmap_position_gradients_match_serial() -> (
    None
):
    """Mapped position gradients have no cross-conformer derivative blocks."""
    system = _system(maxiter=3)
    positions = _positions()

    def energy(value: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(value).energy.sum()

    mapped = torch.func.vmap(torch.func.grad(energy))(positions)
    serial = torch.stack(
        [torch.func.grad(energy)(value) for value in positions]
    )
    torch.testing.assert_close(mapped, serial)
    assert mapped.shape == positions.shape
    assert torch.isfinite(mapped).all()


def test_public_system_singlepoint_vmap_per_conformer_fields() -> None:
    """Per-conformer fields map explicitly without changing the public API."""
    system = _system(intlevel=labels.INTLEVEL_DIPOLE)
    positions = _positions()
    fields = torch.tensor(
        [[0.01, 0.0, -0.02], [-0.02, 0.03, 0.01]], dtype=torch.float64
    )

    mapped = torch.func.vmap(
        lambda position, field: system.singlepoint(position, field=field),
        in_dims=(0, 0),
    )(positions, fields)
    serial = torch.stack(
        [
            system.singlepoint(position, field=field).energy.sum()
            for position, field in zip(positions, fields)
        ]
    )
    assert mapped.energy.shape == (2, 2)
    torch.testing.assert_close(mapped.energy.sum(-1), serial)


def test_public_system_singlepoint_vmap_field_gradients() -> None:
    """Quadrupole field gradients map with their matching integral level."""
    system = _system(intlevel=labels.INTLEVEL_QUADRUPOLE)
    positions = _positions()
    field_grads = torch.stack((0.01 * torch.eye(3), -0.02 * torch.eye(3))).to(
        torch.float64
    )

    mapped = torch.func.vmap(
        lambda position, field_grad: system.singlepoint(
            position, field_grad=field_grad
        ),
        in_dims=(0, 0),
    )(positions, field_grads)
    serial = torch.stack(
        [
            system.singlepoint(position, field_grad=field_grad).energy.sum()
            for position, field_grad in zip(positions, field_grads)
        ]
    )
    assert mapped.quadrupole_integrals is not None
    assert mapped.quadrupole_integrals.shape[0] == 2
    torch.testing.assert_close(mapped.energy.sum(-1), serial)


def test_system_fixed_trip_and_calculator_eager_force_policy() -> None:
    """System reports status while Calculator retains force-convergence errors."""
    one_shot = _system(maxiter=0, force_convergence=True)
    positions = _positions()[0]
    result = one_shot.singlepoint(positions)
    assert result.converged is not None
    assert not bool(result.converged)
    assert result.status is not None

    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, -0.74], [-1.43, 0.0, 0.37], [1.43, 0.0, 0.37]],
        dtype=torch.float64,
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        mixer=labels.MIXER_LINEAR,
        maxiter=1,
        force_convergence=True,
        fermi_etemp=0.0,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    ).setup(numbers)
    nonconverged = system.singlepoint(positions)
    assert nonconverged.converged is not None
    assert not bool(nonconverged.converged)

    calculator = Calculator(
        numbers,
        GFN1_XTB,
        opts={
            "verbosity": 0,
            "int_driver": labels.INTDRIVER_PYTORCH,
            "int_level": labels.INTLEVEL_HCORE,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "force_convergence": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 0.0,
        },
        dtype=positions.dtype,
    )
    from dxtb._src.typing.exceptions import SCFConvergenceError

    with pytest.raises(SCFConvergenceError):
        calculator.singlepoint(positions)


def test_public_system_jacfwd_reports_e1_3_d3_blocker() -> None:
    """The public transform reaches the known DFT-D3 custom-JVP blocker."""
    system = _system(maxiter=2)
    positions = _positions()[0]

    def energy(value: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(value).energy.sum()

    with pytest.raises(NotImplementedError, match="jvp function"):
        torch.func.jacfwd(energy)(positions)


def test_calculator_retains_eager_single_system_adapter() -> None:
    """Calculator and System select their documented separate controllers."""
    positions = _positions()[0]
    calculator = Calculator(
        torch.tensor([1, 1]),
        GFN1_XTB,
        opts={
            "verbosity": 0,
            "int_driver": labels.INTDRIVER_PYTORCH,
            "int_level": labels.INTLEVEL_HCORE,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 0.0,
        },
        dtype=positions.dtype,
    )

    eager = calculator.singlepoint(positions)
    from dxtb._src.calculators.singlepoint import _singlepoint_eager

    reference = _singlepoint_eager(calculator.system, positions)
    torch.testing.assert_close(eager.energy, reference.energy)
    assert torch.equal(eager.iterations, reference.iterations)
    assert eager.status is not None and reference.status is not None
    assert torch.equal(
        eager.status.trajectory_failure,
        reference.status.trajectory_failure,
    )
    assert torch.equal(
        eager.status.final_failure, reference.status.final_failure
    )

    fixed = calculator.system.singlepoint(positions)
    assert fixed.status is not None
    assert fixed.iterations.dtype == torch.int64
