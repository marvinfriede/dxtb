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
"""Core singlepoint and immutable Result tests."""

from __future__ import annotations

import inspect
import sys
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from dxtb import Calculator, GFN1_XTB, ParamModule, labels
from dxtb._src.components.interactions.container import Charges, Potential
from tad_mctc.exceptions import DeviceError, DtypeError
from dxtb._src.calculators.config import Config
from dxtb._src.calculators.model import Model, System
from dxtb._src.calculators.result import ChargeResult, PotentialResult, Result
from dxtb._src.calculators.singlepoint import singlepoint


def _assert_result_values_equal(left: Result, right: Result) -> None:
    """Compare stable tensor outputs without relying on object identity."""
    for name in (
        "energy",
        "scf",
        "fenergy",
        "density",
        "coefficients",
        "emo",
        "occupation",
        "hamiltonian",
        "overlap",
        "hcore",
        "dipole_integrals",
        "quadrupole_integrals",
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            torch.testing.assert_close(left_value, right_value)

    assert left.charges is not None and right.charges is not None
    assert left.potential is not None and right.potential is not None
    for name in ("mono", "dipole", "quadrupole"):
        left_value = getattr(left.charges, name)
        right_value = getattr(right.charges, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            torch.testing.assert_close(left_value, right_value)
        left_value = getattr(left.potential, name)
        right_value = getattr(right.potential, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            torch.testing.assert_close(left_value, right_value)

    assert tuple(name for name, _ in left.classical) == tuple(
        name for name, _ in right.classical
    )
    for (_, left_value), (_, right_value) in zip(
        left.classical, right.classical
    ):
        torch.testing.assert_close(left_value, right_value)
    assert torch.equal(left.iterations, right.iterations)


def test_core_singlepoint_without_calculator_is_history_independent() -> None:
    """Direct core calls are independent of call order and tensor identity."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    numbers = torch.tensor([1, 1])
    model = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype))
    system = model.setup(numbers)

    first = singlepoint(system, positions)
    same_tensor = singlepoint(system, positions)
    _assert_result_values_equal(first, same_tensor)
    snapshot = {
        name: (
            None
            if getattr(first, name) is None
            else getattr(first, name).clone()
        )
        for name in (
            "energy",
            "scf",
            "fenergy",
            "density",
            "coefficients",
            "emo",
            "occupation",
            "hamiltonian",
            "overlap",
            "hcore",
            "dipole_integrals",
            "quadrupole_integrals",
        )
    }
    assert first.charges is not None and first.potential is not None
    nested_snapshot = {
        "charges": {
            name: (
                None
                if getattr(first.charges, name) is None
                else getattr(first.charges, name).clone()
            )
            for name in ("mono", "dipole", "quadrupole")
        },
        "potential": {
            name: (
                None
                if getattr(first.potential, name) is None
                else getattr(first.potential, name).clone()
            )
            for name in ("mono", "dipole", "quadrupole")
        },
        "classical": {name: value.clone() for name, value in first.classical},
    }
    moved = positions.clone()
    moved[1, 0] += 0.1
    second = singlepoint(system, moved)
    repeated = singlepoint(system, positions.clone())

    assert not torch.allclose(first.energy, second.energy)
    _assert_result_values_equal(first, repeated)
    for name, expected in snapshot.items():
        value = getattr(first, name)
        if expected is not None:
            assert value is not None
            torch.testing.assert_close(value, expected)
    for value_name in ("charges", "potential"):
        value = getattr(first, value_name)
        assert value is not None
        for name, expected in nested_snapshot[value_name].items():
            tensor = getattr(value, name)
            if expected is None:
                assert tensor is None
            else:
                assert tensor is not None
                torch.testing.assert_close(tensor, expected)
    for name, expected in nested_snapshot["classical"].items():
        torch.testing.assert_close(first.cenergies[name], expected)
    assert first.charges is not second.charges
    assert first.potential is not second.potential
    for name in ("energy", "density", "hamiltonian", "overlap", "hcore"):
        left = getattr(first, name)
        right = getattr(second, name)
        assert left is not None and right is not None
        assert (
            left.untyped_storage().data_ptr()
            != right.untyped_storage().data_ptr()
        )
    with pytest.raises(FrozenInstanceError):
        first.energy = second.energy  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.charges.mono = second.charges.mono  # type: ignore[union-attr,misc]
    with pytest.raises(FrozenInstanceError):
        first.potential.mono = second.potential.mono  # type: ignore[union-attr,misc]

    assert first.energy is first.total
    assert first.integrals is not None
    assert first.integrals.hcore is not None
    assert first.multipoles is not None
    assert first.charges is not None
    assert first.multipoles.dipole is first.charges.dipole
    assert first.multipoles.quadrupole is first.charges.quadrupole
    assert {field.name for field in fields(first)} >= {
        "energy",
        "scf",
        "classical",
        "fenergy",
        "charges",
        "density",
        "coefficients",
        "emo",
        "occupation",
        "potential",
        "hamiltonian",
        "overlap",
        "hcore",
        "dipole_integrals",
        "quadrupole_integrals",
        "iterations",
        "converged",
        "residual",
    }
    assert not isinstance(first.charges, Charges)
    assert not isinstance(first.potential, Potential)
    for value in (first.charges, first.potential):
        assert value is not None
        assert not hasattr(value, "batch_mode")
        assert not hasattr(value, "label")
        assert not hasattr(value, "reset")
        assert not hasattr(value, "nullify_padding")


def test_core_singlepoint_supports_independent_position_gradients() -> None:
    """One setup supports repeated position derivatives without graph reuse."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )

    gradients = []
    for geometry in (positions, positions + positions.new_tensor([0.01, 0, 0])):
        leaf = geometry.clone().requires_grad_(True)
        result = singlepoint(system, leaf)
        (gradient,) = torch.autograd.grad(result.energy.sum(), leaf)
        assert torch.isfinite(gradient).all()
        gradients.append(gradient)

    repeated = positions.clone().requires_grad_(True)
    result = singlepoint(system, repeated)
    (repeated_gradient,) = torch.autograd.grad(result.energy.sum(), repeated)
    torch.testing.assert_close(repeated_gradient, gradients[0])


def test_core_singlepoint_geometry_after_prior_backward() -> None:
    """A prior backward does not affect a fresh geometry call on one System."""
    system, positions = _system()
    first_positions = positions.clone().requires_grad_(True)
    first = singlepoint(system, first_positions)
    assert first.charges is not None and first.potential is not None
    charge_before = first.charges.mono.clone()
    potential_before = first.potential.mono.clone()
    first.energy.sum().backward()
    assert first_positions.grad is not None
    assert torch.isfinite(first_positions.grad).all()

    second_positions = positions.clone()
    second_positions[1, 0] += 0.03
    second_positions.requires_grad_(True)
    second = singlepoint(system, second_positions)
    (second_gradient,) = torch.autograd.grad(
        second.energy.sum(), second_positions
    )
    assert torch.isfinite(second_gradient).all()
    torch.testing.assert_close(first.charges.mono, charge_before)
    assert first.potential.mono is not None
    torch.testing.assert_close(first.potential.mono, potential_before)


def test_result_charge_geometry_gradient_is_connected() -> None:
    """Result charge tensors preserve their position-autograd connection."""
    numbers = torch.tensor([3, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]], dtype=torch.float64
    ).requires_grad_(True)
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        numbers
    )
    result = singlepoint(system, positions)
    assert result.charges is not None
    (gradient,) = torch.autograd.grad(
        result.charges.mono.square().sum(), positions
    )
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_system_singlepoint_delegates_to_core() -> None:
    """System exposes a thin direct delegate to the singlepoint core."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )
    result = system.singlepoint(positions)
    direct = singlepoint(system, positions)
    _assert_result_values_equal(result, direct)
    source = inspect.getsource(sys.modules[singlepoint.__module__])
    assert "Result.snapshot(" in source
    assert "Result(" not in source
    assert "tensor_id" not in source
    assert "data_ptr" not in source
    assert "id(" not in source


def test_core_parameter_training_rebuilds_system_per_loss() -> None:
    """Parameter gradients use a fresh setup for each differentiated loss."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    numbers = torch.tensor([1, 1])
    model = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype))
    parameter = dict(model.par.named_parameters())[
        "parameter_tree.element.H.slater.param"
    ]
    parameter.requires_grad_(True)

    gradients = []
    for _ in range(2):
        system = model.setup(numbers)
        result = singlepoint(system, positions)
        assert result.charges is not None
        loss = result.charges.mono.square().sum()
        (gradient,) = torch.autograd.grad(loss, parameter)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        gradients.append(gradient)
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.parametrize("exclude", [("scf",), ("all",)])
def test_core_singlepoint_early_exclusion_returns_complete_result(
    exclude: tuple[str, ...],
) -> None:
    """SCF exclusion paths return a complete immutable Result value."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(method=labels.GFN1_XTB, exclude=exclude)
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    )
    result = singlepoint(model.setup(torch.tensor([1, 1])), positions)
    assert result.scf is not None
    assert torch.equal(result.scf, torch.zeros_like(result.scf))
    assert result.integrals is None
    assert result.charges is None
    assert result.potential is None
    assert torch.isfinite(result.energy).all()
    classical_sum = sum(
        (value for _, value in result.classical),
        torch.zeros_like(result.energy),
    )
    torch.testing.assert_close(result.energy, classical_sum)
    assert result.iterations.shape == torch.Size([])
    assert result.iterations.device == positions.device
    assert result.iterations.item() == 0


def test_core_singlepoint_validates_dtype_without_integral_setup() -> None:
    """SCF-excluded Systems still validate position dtype from System.dd."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN1_XTB, exclude=("all",), int_level=0
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype),
        config=config,
        auto_int_level=False,
    ).setup(torch.tensor([1, 1]))
    assert system.h0_setup is None
    assert system.integral_setup is None
    with pytest.raises(DtypeError, match="Dtype mismatch"):
        singlepoint(system, positions.to(torch.float32))


def test_core_singlepoint_rejects_missing_hcore_level() -> None:
    """The core preserves the established minimum integral-level error."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN1_XTB, int_level=labels.INTLEVEL_OVERLAP
    )
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype),
        config=config,
        auto_int_level=False,
    )
    system = model.setup(torch.tensor([1, 1]))
    with pytest.raises(NotImplementedError, match="Core Hamiltonian missing"):
        singlepoint(system, positions)


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 3), (3, 3)],
    ids=["batched-positions", "wrong-atom-count"],
)
def test_core_singlepoint_rejects_non_single_system_positions(
    shape: tuple[int, ...],
) -> None:
    """The core does not infer batching from a positions tensor."""
    positions = torch.zeros(shape, dtype=torch.float64)
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )
    with pytest.raises(ValueError, match=r"shape \(nat, 3\)"):
        singlepoint(system, positions)


def _system(dtype: torch.dtype = torch.float64) -> tuple[System, torch.Tensor]:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=dtype)
    system = Model(par=ParamModule(GFN1_XTB, dtype=dtype)).setup(
        torch.tensor([1, 1])
    )
    return system, positions


def test_core_singlepoint_validates_positions_dtype_and_device() -> None:
    """Core rejects mismatched geometry dtype/device before evaluation."""
    system, positions = _system()
    with pytest.raises(DtypeError, match="Dtype mismatch"):
        singlepoint(system, positions.to(torch.float32))
    with pytest.raises(DeviceError, match="Device mismatch"):
        singlepoint(system, torch.empty(positions.shape, device="meta"))


@pytest.mark.parametrize(
    "charge",
    [
        0.0,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    ],
    ids=["python-scalar", "zero-d", "one-element"],
)
def test_core_singlepoint_accepts_scalar_charge_forms(
    charge: torch.Tensor | float,
) -> None:
    system, positions = _system()
    result = singlepoint(system, positions, chrg=charge)
    assert torch.isfinite(result.energy).all()


def test_core_singlepoint_rejects_multivalued_charge_and_spin() -> None:
    system, positions = _system()
    with pytest.raises(ValueError, match="one-element"):
        singlepoint(
            system,
            positions,
            chrg=torch.tensor([0.0, 1.0], dtype=positions.dtype),
        )
    with pytest.raises(ValueError, match="one-element"):
        singlepoint(
            system,
            positions,
            spin=torch.tensor([0.0, 1.0], dtype=positions.dtype),
        )


@pytest.mark.parametrize(
    "spin",
    [
        None,
        0.0,
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    ],
    ids=["none", "python-scalar", "zero-d", "one-element"],
)
def test_core_singlepoint_accepts_scalar_spin_forms(
    spin: torch.Tensor | float | None,
) -> None:
    system, positions = _system()
    result = singlepoint(system, positions, spin=spin)
    assert torch.isfinite(result.energy).all()


@pytest.mark.parametrize("name", ["charge", "spin"])
def test_core_singlepoint_validates_tensor_call_input_dtype_and_device(
    name: str,
) -> None:
    system, positions = _system()
    invalid_dtype = torch.tensor(0.0, dtype=torch.float32)
    with pytest.raises(DtypeError, match="Dtype mismatch"):
        singlepoint(
            system,
            positions,
            **{"chrg" if name == "charge" else "spin": invalid_dtype},
        )
    invalid_device = torch.empty((), device="meta")
    with pytest.raises(DeviceError, match="Device mismatch"):
        singlepoint(
            system,
            positions,
            **{"chrg" if name == "charge" else "spin": invalid_device},
        )


@pytest.mark.parametrize("case", ["shape", "dtype", "device", "charge", "spin"])
def test_core_and_calculator_reject_same_invalid_single_system_inputs(
    case: str,
) -> None:
    """Calculator adapter and direct core enforce the same input contract."""
    system, positions = _system()
    calculator = Calculator(
        torch.tensor([1, 1]),
        GFN1_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0},
    )
    kwargs = {}
    if case == "shape":
        positions = torch.zeros((1, 2, 3), dtype=positions.dtype)
    elif case == "dtype":
        positions = positions.to(torch.float32)
    elif case == "device":
        positions = torch.empty(positions.shape, device="meta")
    elif case == "charge":
        kwargs["chrg"] = torch.tensor([0.0, 1.0], dtype=positions.dtype)
    else:
        kwargs["spin"] = torch.tensor([0.0, 1.0], dtype=positions.dtype)

    with pytest.raises((ValueError, DtypeError, DeviceError)):
        singlepoint(system, positions, **kwargs)
    with pytest.raises((ValueError, DtypeError, DeviceError)):
        calculator.singlepoint(positions, **kwargs)


def test_core_singlepoint_tensor_charge_gradient_is_preserved() -> None:
    """Tensor call-input normalization preserves charge autograd."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(method=labels.GFN1_XTB, fermi_etemp=300)
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    )
    system = model.setup(torch.tensor([1, 1]))
    charge = torch.tensor(0.1, dtype=positions.dtype, requires_grad=True)
    result = singlepoint(system, positions, chrg=charge)
    (gradient,) = torch.autograd.grad(result.energy.sum(), charge)
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0


def test_result_charge_and_potential_are_plain_values() -> None:
    """Result payloads use narrow immutable values without SCF container API."""
    system, positions = _system()
    result = singlepoint(system, positions)
    assert isinstance(result.charges, ChargeResult)
    assert isinstance(result.potential, PotentialResult)
    assert result.charges.mono.numel() > 0
    assert result.potential.mono is not None
    assert result.potential.mono.numel() > 0
    for value in (result.charges, result.potential):
        for forbidden in (
            "batch_mode",
            "axis",
            "label",
            "reset",
            "nullify_padding",
        ):
            assert not hasattr(value, forbidden)
    assert result.multipoles is not None
    assert result.multipoles.dipole is result.charges.dipole
    assert result.multipoles.quadrupole is result.charges.quadrupole
