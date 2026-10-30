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
"""E6bcompat policy characterization for packed and low-level batch calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from dxtb import GFN1_XTB, Calculator
from dxtb._src.calculators.singlepoint import _call_tensor
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.scf import iterator


def _packed_calculator() -> tuple[Calculator, torch.Tensor, torch.Tensor]:
    """Build an independent two-conformer legacy Calculator fixture."""
    numbers = torch.tensor([[1, 1], [1, 1]])
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]],
        ],
        dtype=torch.float64,
    )
    calculator = Calculator(
        numbers,
        GFN1_XTB,
        dtype=torch.float64,
        opts={"verbosity": 0, "mixer": "linear", "maxiter": 0},
        batch_mode=2,
    )
    assert calculator.ihelp.batch_mode == 2
    return calculator, numbers, positions


def _packed_low_level_inputs(
    calculator: Calculator, numbers: torch.Tensor, positions: torch.Tensor
) -> tuple[Any, ...]:
    """Prepare the same matrices and call-local cache as the packed facade."""
    charge = torch.zeros((2, 1), dtype=positions.dtype)
    spin = torch.zeros_like(charge)
    overlap = calculator.integrals.build_overlap(positions)
    hcore = calculator.integrals.build_hcore(
        positions, with_overlap=True, charge=charge
    )
    integrals = IntegralMatrices(hcore=hcore, overlap=overlap)
    refocc = calculator.integrals.hcore.refocc
    cache = calculator.interactions.get_cache(
        numbers=numbers, positions=positions, ihelp=calculator.ihelp
    )
    return (
        numbers,
        positions,
        charge,
        spin,
        calculator.interactions,
        cache,
        calculator.ihelp,
        calculator.opts.scf,
        integrals,
        refocc,
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("fwd_options", {"damp": 0.25}),
        ("bck_options", {"posdef": False}),
        ("eigen_options", {"method": "exacteig"}),
        ("unexpected_option", object()),
        ("solver_options", {"unused": True}),
        ("another_unknown_option", object()),
    ],
    ids=[
        "fwd-options",
        "bck-options",
        "eigen-options",
        "unexpected-option",
        "solver-options",
        "other-unknown",
    ],
)
def test_batched_low_level_engine_kwargs_are_explicitly_retired(
    keyword: str, value: object
) -> None:
    """Arbitrary old engine kwargs never select the retained batch engine."""
    calculator, numbers, positions = _packed_calculator()
    call = _packed_low_level_inputs(calculator, numbers, positions)

    with pytest.raises(NotImplementedError, match="overrides are no longer"):
        iterator.solve(*call, **{keyword: value})


def test_batched_low_level_positional_engine_extra_is_explicitly_retired() -> (
    None
):
    """Arbitrary old engine positional extras do not enter a solver."""
    calculator, numbers, positions = _packed_calculator()
    call = _packed_low_level_inputs(calculator, numbers, positions)

    with pytest.raises(NotImplementedError, match="overrides are no longer"):
        iterator.solve(*call, object())


def test_batched_malformed_integral_shapes_fail_at_their_owner() -> None:
    """Container shape mismatch and internally wrong AO dimensions stay errors."""
    calculator, numbers, positions = _packed_calculator()
    call = _packed_low_level_inputs(calculator, numbers, positions)
    original = call[8]
    assert isinstance(original, IntegralMatrices)

    with pytest.raises(ValueError, match="must match the hcore shape"):
        IntegralMatrices(
            hcore=original.hcore,
            overlap=original.overlap[..., :1, :1],
        )

    wrong_shape = IntegralMatrices(
        hcore=original.hcore[..., :1, :1],
        overlap=original.overlap[..., :1, :1],
    )
    malformed = list(call)
    malformed[8] = wrong_shape
    with pytest.raises(RuntimeError, match="Expected size for first two"):
        iterator.solve(*malformed)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("hcore", "requires hcore and overlap"),
        ("overlap", "requires hcore and overlap"),
    ],
)
def test_batched_missing_integrals_are_setup_errors(
    field: str, message: str
) -> None:
    """Missing integrals fail before the packed compatibility route."""
    calculator, numbers, positions = _packed_calculator()
    call = list(_packed_low_level_inputs(calculator, numbers, positions))
    integrals = call[8]
    assert isinstance(integrals, IntegralMatrices)
    values = {
        "hcore": integrals.hcore,
        "overlap": integrals.overlap,
        "dipole": integrals.dipole,
        "quadrupole": integrals.quadrupole,
    }
    values[field] = None
    call[8] = SimpleNamespace(**values)

    with pytest.raises(ValueError, match=message):
        iterator.solve(*call)


def test_batched_nonfinite_integrals_keep_eigensolver_error() -> None:
    """Nonfinite matrices are not converted into a successful SCF result."""
    calculator, numbers, positions = _packed_calculator()
    call = _packed_low_level_inputs(calculator, numbers, positions)
    original = call[8]
    assert isinstance(original, IntegralMatrices)
    nonfinite = IntegralMatrices(
        hcore=original.hcore * torch.tensor(float("nan")),
        overlap=original.overlap,
    )
    malformed = list(call)
    malformed[8] = nonfinite

    with pytest.raises(RuntimeError, match="linalg.eigh"):
        iterator.solve(*malformed)


def test_public_packed_calculator_does_not_forward_engine_options() -> None:
    """Public packed extras are not low-level SCF forwarding options."""
    calculator, _, positions = _packed_calculator()
    baseline = calculator.singlepoint(positions)

    calculator_with_ignored_extra, _, _ = _packed_calculator()
    with_extra = calculator_with_ignored_extra.singlepoint(
        positions, unexpected_option=object(), solver_options={"unused": True}
    )

    assert baseline.energy.shape == (2, 2)
    assert with_extra.energy.shape == (2, 2)
    torch.testing.assert_close(with_extra.energy, baseline.energy)
    torch.testing.assert_close(with_extra.scf, baseline.scf)
    assert baseline.iterations is not None
    assert with_extra.iterations is not None
    torch.testing.assert_close(with_extra.iterations, baseline.iterations)


def test_public_packed_charge_shape_contract() -> None:
    """Batched public calls reject scalar charge but accept a charge vector."""
    calculator, _, positions = _packed_calculator()
    with pytest.raises(ValueError, match="single float or integer"):
        calculator.singlepoint(positions, chrg=1.0)

    charges = torch.zeros(2, dtype=positions.dtype)
    spins = torch.zeros(2, dtype=positions.dtype)
    result = calculator.singlepoint(positions, chrg=charges, spin=spins)
    assert result.energy.shape == (2, 2)
    assert result.charges is not None
    assert result.charges.mono.shape[0] == 2


def test_public_packed_field_is_shared_not_per_system() -> None:
    """Calculator field validation accepts shared vectors only."""
    positions = torch.zeros((2, 2, 3), dtype=torch.float64)
    shared_field = torch.zeros(3, dtype=positions.dtype)
    assert _call_tensor("field", shared_field, positions, (3,)) is shared_field
    with pytest.raises(ValueError, match=r"field must have shape \(3,\)"):
        _call_tensor(
            "field", torch.zeros((2, 3), dtype=positions.dtype), positions, (3,)
        )

    shared_gradient = torch.zeros((3, 3), dtype=positions.dtype)
    assert (
        _call_tensor("field_grad", shared_gradient, positions, (3, 3))
        is shared_gradient
    )
    with pytest.raises(
        ValueError, match=r"field_grad must have shape \(3, 3\)"
    ):
        _call_tensor(
            "field_grad",
            torch.zeros((2, 3, 3), dtype=positions.dtype),
            positions,
            (3, 3),
        )
