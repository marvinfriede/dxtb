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
    scf_mode: int = labels.SCF_MODE_FULL,
    mixer: int = labels.MIXER_LINEAR,
    strict: bool = False,
    fermi_maxiter: int | None = None,
):
    """Construct an independent PyTorch-backed GFN1/H2 System."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=(labels.INTLEVEL_HCORE if intlevel is None else intlevel),
        mixer=mixer,
        strict=strict,
        scf_mode=scf_mode,
        maxiter=maxiter,
        force_convergence=force_convergence,
        fermi_etemp=temperature,
        x_atol=0.0,
        x_atol_max=0.0,
        **({} if fermi_maxiter is None else {"fermi_maxiter": fermi_maxiter}),
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


def test_system_legacy_fallback_constructs_only_eager_guess(
    monkeypatch,
) -> None:
    """Legacy-owned temperature routes do not build an unused fixed guess."""
    import dxtb._src.scf.functional.eager as eager

    system = _system(maxiter=1, temperature=-1.0)
    positions = _positions()[0]
    original_get_guess = eager.get_guess
    original_eager_runner = eager.run_functional_full_eager
    guess_validation: list[bool] = []
    eager_calls = 0

    def tracked_get_guess(*args, validate: bool = True, **kwargs):
        guess_validation.append(validate)
        return original_get_guess(*args, validate=validate, **kwargs)

    def tracked_eager_runner(*args, **kwargs):
        nonlocal eager_calls
        eager_calls += 1
        return original_eager_runner(*args, **kwargs)

    monkeypatch.setattr(eager, "get_guess", tracked_get_guess)
    monkeypatch.setattr(
        eager, "run_functional_full_eager", tracked_eager_runner
    )

    result = system.singlepoint(positions)

    assert eager_calls == 1
    assert guess_validation == [True]
    assert result.status is None


def test_system_functional_fixed_trip_constructs_one_transform_guess(
    monkeypatch,
) -> None:
    """Eligible System execution builds exactly one non-eagerly-validated guess."""
    import dxtb._src.scf.functional.eager as eager

    system = _system(maxiter=1)
    original_get_guess = eager.get_guess
    guess_validation: list[bool] = []

    def tracked_get_guess(*args, validate: bool = True, **kwargs):
        guess_validation.append(validate)
        return original_get_guess(*args, validate=validate, **kwargs)

    monkeypatch.setattr(eager, "get_guess", tracked_get_guess)
    result = system.singlepoint(_positions()[0])

    assert guess_validation == [False]
    assert result.status is not None


@pytest.mark.parametrize(
    "route_options",
    [
        {"fermi_maxiter": -1},
        {"scf_mode": labels.SCF_MODE_EXPERIMENTAL},
    ],
    ids=["negative-fermi-maxiter", "non-full-mode"],
)
def test_system_static_legacy_routes_skip_transform_guess(
    monkeypatch, route_options
) -> None:
    """Legacy Fermi and SCF-mode routes dispatch before fixed-trip guessing."""
    import dxtb._src.scf.functional.eager as eager

    class LegacyRouteReached(Exception):
        pass

    system = _system(maxiter=1, **route_options)
    eager_calls = 0

    def unexpected_guess(*args, **kwargs):
        pytest.fail("legacy route must not build the transform-only guess")

    def legacy_adapter(*args, **kwargs):
        nonlocal eager_calls
        eager_calls += 1
        raise LegacyRouteReached

    monkeypatch.setattr(eager, "get_guess", unexpected_guess)
    monkeypatch.setattr(eager, "run_functional_full_eager", legacy_adapter)

    with pytest.raises(LegacyRouteReached):
        system.singlepoint(_positions()[0])
    assert eager_calls == 1


def test_system_strict_mixer_error_precedes_initial_guess(monkeypatch) -> None:
    """Static strict configuration errors do not construct any SCF guess."""
    import dxtb._src.scf.functional.eager as eager

    system = _system(mixer=labels.MIXER_BROYDEN, strict=True)

    def unexpected_guess(*args, **kwargs):
        pytest.fail("invalid static configuration must fail before guessing")

    monkeypatch.setattr(eager, "get_guess", unexpected_guess)

    with pytest.raises(
        NotImplementedError, match="Broyden mixer is not implemented"
    ):
        system.singlepoint(_positions()[0])


def test_public_system_jacfwd_d3_depends_on_upstream_atomic_c6_api() -> None:
    """The old C6 custom Function blocks JVP; composite upstream passes."""
    system = _system(maxiter=2)
    positions = _positions()[0]

    def energy(value: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(value).energy.sum()

    from dxtb._src.components.classicals.dispersion import d3 as d3_component

    if d3_component._D3_COMPOSITE_C6:
        forward = torch.func.jacfwd(energy)(positions)
        reverse = torch.func.jacrev(energy)(positions)
        assert torch.isfinite(forward).all()
        torch.testing.assert_close(forward, reverse)
    else:
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
