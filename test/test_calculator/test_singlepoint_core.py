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

import importlib
import inspect
import sys
from dataclasses import FrozenInstanceError, fields

import pytest
import torch
from tad_mctc.exceptions import DeviceError, DtypeError

from dxtb import (
    GFN1_XTB,
    GFN2_XTB,
    Calculator,
    OutputHandler,
    ParamModule,
    labels,
)
from dxtb._src.calculators.config import Config
from dxtb._src.calculators.model import Model, System
from dxtb._src.calculators.result import ChargeResult, PotentialResult, Result
from dxtb._src.calculators.singlepoint import singlepoint
from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.scf.status import SCFStatus

from ..utils import scf_strategy


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
    if left.status is None or right.status is None:
        assert left.status is right.status
    else:
        torch.testing.assert_close(
            left.status.trajectory_failure, right.status.trajectory_failure
        )
        torch.testing.assert_close(
            left.status.final_failure, right.status.final_failure
        )


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
        first.status = None  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.charges.mono = second.charges.mono  # type: ignore[union-attr,misc]
    with pytest.raises(FrozenInstanceError):
        first.potential.mono = second.potential.mono  # type: ignore[union-attr,misc]

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
        "status",
    }
    assert not isinstance(first.charges, Charges)
    assert not isinstance(first.potential, Potential)
    for value in (first.charges, first.potential):
        assert value is not None
        assert not hasattr(value, "batch_mode")
        assert not hasattr(value, "label")
        assert not hasattr(value, "reset")
        assert not hasattr(value, "nullify_padding")


def test_default_full_singlepoint_uses_functional_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default FULL singlepoint uses the functional SCF engine."""
    from dxtb._src.scf.functional.eager import FunctionalCoreResult

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )
    from dxtb.scf import AndersonMethod, SCFDifferentiation, SCFRepresentation

    scf_config = system.config.scf
    assert scf_config.differentiation is SCFDifferentiation.UNROLLED
    assert scf_config.initial_representation is SCFRepresentation.POTENTIAL
    assert isinstance(scf_config.strategy.stages[0].method, AndersonMethod)
    assert system.config.scf.fermi.etemp == 300.0

    core_module = importlib.import_module("dxtb._src.calculators.singlepoint")
    original_solve = core_module.scf.solve_with_core_index
    captured: dict[str, object] = {}

    def capture_solve(*args: object, **kwargs: object) -> object:
        result = original_solve(*args, **kwargs)
        captured["result"] = result
        return result

    monkeypatch.setattr(core_module.scf, "solve_with_core_index", capture_solve)
    OutputHandler.clear_warnings()
    result = singlepoint(system, positions)
    assert OutputHandler.warnings == []
    functional = captured["result"]
    assert isinstance(functional, FunctionalCoreResult)
    assert result.charges is not None
    assert result.potential is not None
    assert result.charges.mono is functional.charges.mono
    assert result.charges.dipole is functional.charges.dipole
    assert result.charges.quadrupole is functional.charges.quad
    assert result.potential.mono is functional.potential.mono
    assert result.potential.dipole is functional.potential.dipole
    assert result.potential.quadrupole is functional.potential.quad
    assert result.iterations is functional.iterations
    assert result.converged is functional.converged
    assert result.residual is functional.residual
    assert result.status is not None
    assert result.status.trajectory_failure.dtype == torch.int64
    assert result.status.final_failure.dtype == torch.int64
    solve_result = functional.solve_result
    assert solve_result.trajectory_status is not None
    assert (
        result.status.trajectory_failure
        is solve_result.trajectory_status.first_failure
    )
    assert result.status.final_failure.item() == 0


def test_result_snapshot_preserves_optional_status_and_tensor_identity() -> (
    None
):
    """Snapshot forwards status without changing graph-connected tensors."""
    from dxtb import SCFFailureCode

    energy = torch.tensor(1.0, requires_grad=True)
    code = torch.tensor(int(SCFFailureCode.NONE), dtype=torch.int64)
    status = SCFStatus(code, code)
    result = Result.snapshot(
        energy=energy,
        scf=energy,
        classical=(),
        fenergy=energy,
        iterations=torch.tensor(0, dtype=torch.int64),
        status=status,
    )
    assert result.status is status
    assert result.energy is energy
    assert result.scf is energy
    assert result.fenergy is energy
    assert result.status is not None
    assert result.status.trajectory_failure is code
    assert (
        Result.snapshot(
            energy=energy,
            scf=energy,
            classical=(),
            fenergy=energy,
            iterations=torch.tensor(0, dtype=torch.int64),
        ).status
        is None
    )
    assert torch.autograd.grad(result.energy, energy)[0].item() == 1.0


def test_zero_temperature_functional_result_has_success_status() -> None:
    """Functional zero-temperature calls expose integer zero status codes."""
    from dxtb import SCFFailureCode

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        fermi_etemp=0.0,
        maxiter=0,
        force_convergence=True,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    ).setup(torch.tensor([1, 1]))
    result = system.singlepoint(positions)
    assert result.status is not None
    assert result.status.trajectory_failure.item() == int(SCFFailureCode.NONE)
    assert result.status.final_failure.item() == int(SCFFailureCode.NONE)
    for code in (
        result.status.trajectory_failure,
        result.status.final_failure,
    ):
        assert code.dtype == torch.int64
        assert code.device == positions.device
    assert result.iterations.dtype == torch.int64
    assert result.converged is not None
    assert result.residual is not None
    assert result.iterations.item() == 1
    assert not result.converged.item()


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_production_converged_route_executes_only_physical_maps(
    monkeypatch: pytest.MonkeyPatch, method: int
) -> None:
    """Eager production stops mapping after convergence, unlike fixed trips."""
    from test.test_scf.test_functional_map import _prepare_case

    from dxtb._src.scf.functional import map as functional_map
    from dxtb._src.scf.functional import problem as functional_problem

    options = {
        "strategy": scf_strategy("anderson", "potential"),
        "fermi_etemp": 300.0,
        "maxiter": 100,
        "verbosity": 0,
    }
    production_case = _prepare_case(
        method, scf_options=options, setup_functional=False
    )
    functional_counts = {"raw_maps": 0, "spectra": 0}
    original_map = functional_problem.potential_map
    original_evaluate = functional_map.evaluate_hamiltonian

    def count_functional_map(*args, **kwargs):
        functional_counts["raw_maps"] += 1
        return original_map(*args, **kwargs)

    def count_functional_spectrum(*args, **kwargs):
        functional_counts["spectra"] += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        functional_problem, "potential_map", count_functional_map
    )
    monkeypatch.setattr(
        functional_map, "evaluate_hamiltonian", count_functional_spectrum
    )
    production_result = singlepoint(
        production_case.system, production_case.positions
    )
    assert production_result.converged is not None
    assert production_result.converged.item()
    assert production_result.status is not None
    for code in (
        production_result.status.trajectory_failure,
        production_result.status.final_failure,
    ):
        assert code.dtype == torch.int64
        assert code.device == production_case.positions.device
        assert code.shape == torch.Size([])
        assert code.item() == 0
    assert functional_counts["raw_maps"] == production_result.iterations.item()
    assert functional_counts["spectra"] == functional_counts["raw_maps"] + 1


@pytest.mark.parametrize(
    ("variable", "converger"),
    [
        ("charge", "linear"),
        ("charge", "anderson"),
        ("potential", "linear"),
        ("potential", "anderson"),
        ("fock", "linear"),
        ("fock", "anderson"),
    ],
    ids=[
        "charge-simple",
        "charge-anderson",
        "potential-simple",
        "potential-anderson",
        "fock-simple",
        "fock-anderson",
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 300.0], ids=["zero-k", "300-k"])
def test_production_singlepoint_matches_legacy_full(
    variable: str, converger: str, temperature: float
) -> None:
    """Production FULL dispatch preserves functional eager values in each route."""
    from test.test_scf.test_functional_map import _prepare_case

    from dxtb._src.scf.functional import run_functional_full_eager

    options = {
        "strategy": scf_strategy(converger, variable),
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
        "fermi_etemp": temperature,
    }
    functional_case = _prepare_case(
        GFN1_XTB, scf_options=options, setup_functional=False
    )
    production_case = _prepare_case(
        GFN1_XTB, scf_options=options, setup_functional=False
    )
    expected = run_functional_full_eager(
        functional_case.system.numbers,
        functional_case.positions,
        functional_case.charge,
        functional_case.spin,
        functional_case.interactions,
        functional_case.interaction_data,
        functional_case.system.ihelp,
        functional_case.system.config.scf,
        functional_case.integrals,
        functional_case.refocc,
    ).functional
    assert expected is not None
    actual = singlepoint(production_case.system, production_case.positions)
    assert actual.status is not None
    assert actual.status.trajectory_failure.dtype == torch.int64
    assert actual.status.final_failure.dtype == torch.int64
    assert torch.count_nonzero(actual.status.trajectory_failure) == 0
    assert torch.count_nonzero(actual.status.final_failure) == 0
    for name in (
        "density",
        "emo",
        "occupation",
        "hamiltonian",
    ):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name)
        )
    torch.testing.assert_close(
        actual.coefficients @ actual.coefficients.mT,
        expected.coefficients @ expected.coefficients.mT,
    )
    torch.testing.assert_close(actual.scf, expected.energy)
    torch.testing.assert_close(actual.fenergy, expected.fenergy)
    assert actual.charges is not None and actual.potential is not None
    for legacy_name, result_name in (
        ("mono", "mono"),
        ("dipole", "dipole"),
        ("quad", "quadrupole"),
    ):
        expected_charge = getattr(expected.charges, legacy_name, None)
        expected_potential = getattr(expected.potential, legacy_name, None)
        actual_charge = getattr(actual.charges, result_name)
        actual_potential = getattr(actual.potential, result_name)
        if expected_charge is None:
            assert actual_charge is None
        else:
            assert actual_charge is not None
            torch.testing.assert_close(actual_charge, expected_charge)
        if expected_potential is None:
            assert actual_potential is None
        else:
            assert actual_potential is not None
            torch.testing.assert_close(actual_potential, expected_potential)
    assert actual.iterations.item() == expected.iterations.item()
    assert actual.converged is not None and not actual.converged
    assert actual.residual is not None


@pytest.mark.parametrize(
    ("variable", "converger"),
    [
        ("charge", "linear"),
        ("potential", "anderson"),
        ("fock", "linear"),
    ],
    ids=["charge-simple", "potential-anderson", "fock-simple"],
)
@pytest.mark.parametrize("force_convergence", [False, True])
def test_production_maxiter_zero_is_valid_one_shot(
    variable: str, converger: str, force_convergence: bool
) -> None:
    """The production boundary preserves maxiter=0 for every representation."""
    from test.test_scf.test_functional_map import _prepare_case

    from dxtb import SCFFailureCode
    from dxtb._src.scf.functional import run_functional_full_eager

    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy(converger, variable),
            "maxiter": 0,
            "force_convergence": force_convergence,
            "fermi_etemp": 300.0,
        },
        setup_functional=False,
    )
    expected = run_functional_full_eager(
        case.system.numbers,
        case.positions,
        case.charge,
        case.spin,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        case.system.config.scf,
        case.integrals,
        case.refocc,
    ).functional
    assert expected is not None
    OutputHandler.clear_warnings()
    actual = singlepoint(case.system, case.positions)
    assert actual.iterations.item() == 1
    assert actual.converged is not None and not actual.converged
    assert actual.residual is not None
    assert actual.status is not None
    assert actual.status.trajectory_failure.dtype == torch.int64
    assert actual.status.final_failure.dtype == torch.int64
    assert actual.status.trajectory_failure.device == actual.scf.device
    assert actual.status.final_failure.device == actual.scf.device
    assert actual.status.trajectory_failure.item() == int(SCFFailureCode.NONE)
    assert actual.status.final_failure.item() == int(SCFFailureCode.NONE)
    assert OutputHandler.warnings == []
    torch.testing.assert_close(actual.scf, expected.energy)
    torch.testing.assert_close(actual.fenergy, expected.fenergy)


@pytest.mark.parametrize("force_convergence", [False, True])
def test_production_positive_budget_nonconvergence_policy(
    force_convergence: bool,
) -> None:
    """Production FULL policy warns or raises after a positive failed budget."""
    from test.test_scf.test_functional_map import _prepare_case

    from dxtb._src.typing.exceptions import (
        SCFConvergenceError,
        SCFConvergenceWarning,
    )

    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy("linear", "potential"),
            "maxiter": 1,
            "force_convergence": force_convergence,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
        setup_functional=False,
    )
    OutputHandler.clear_warnings()
    if force_convergence:
        with pytest.raises(SCFConvergenceError):
            singlepoint(case.system, case.positions)
        assert OutputHandler.warnings == []
    else:
        result = singlepoint(case.system, case.positions)
        assert result.converged is not None and not result.converged
        assert len(OutputHandler.warnings) == 1
        assert OutputHandler.warnings[0][1] is SCFConvergenceWarning


@pytest.mark.parametrize(
    "converger",
    ["linear", "anderson"],
    ids=["linear", "anderson"],
)
def test_production_negative_temperature_is_rejected(
    converger: str,
) -> None:
    """Negative electronic temperature fails before SCF iteration."""
    from test.test_scf.test_functional_map import _prepare_case

    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy(converger, "charge"),
            "maxiter": 1,
            "fermi_etemp": -1.0,
        },
        setup_functional=False,
    )
    OutputHandler.clear_warnings()
    with pytest.raises(
        ValueError, match="Electronic Temperature must be non-negative"
    ):
        singlepoint(case.system, case.positions)
    assert OutputHandler.warnings == []


@pytest.mark.parametrize(
    ("variable", "converger"),
    [
        ("charge", "linear"),
        ("potential", "anderson"),
        ("fock", "anderson"),
    ],
    ids=["charge-simple", "potential-anderson", "fock-anderson"],
)
def test_production_dynamic_damping_matches_legacy(
    variable: str, converger: str
) -> None:
    """Production routing retains dynamic-damping trajectory and warnings."""
    from test.test_scf.test_functional_map import _prepare_case

    from dxtb._src.scf.functional import run_functional_full_eager

    maxiter = 12 if variable == "charge" else 8
    anderson = (
        {"damp_init": 0.05, "generations": 2, "soft_start": True}
        if converger == "anderson"
        else {}
    )
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": scf_strategy(converger, variable, **anderson),
            "maxiter": maxiter,
            "mix_guess": True,
            "damp": 0.2,
            "damp_dynamic": True,
            "damp_dynamic_factor": 0.75,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
        setup_functional=False,
    )
    OutputHandler.clear_warnings()
    expected = run_functional_full_eager(
        case.system.numbers,
        case.positions,
        case.charge,
        case.spin,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        case.system.config.scf,
        case.integrals,
        case.refocc,
    ).functional
    assert expected is not None
    OutputHandler.clear_warnings()
    actual = singlepoint(case.system, case.positions)
    assert len(OutputHandler.warnings) == 1
    assert OutputHandler.warnings[0][1].__name__ == "SCFConvergenceWarning"
    assert "strategy" in OutputHandler.warnings[0][0]
    assert actual.iterations.item() == expected.iterations.item()
    for name in (
        "density",
        "emo",
        "occupation",
        "hamiltonian",
    ):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            msg=f"production field: {name}",
        )
    torch.testing.assert_close(
        actual.coefficients @ actual.coefficients.mT,
        expected.coefficients @ expected.coefficients.mT,
    )
    torch.testing.assert_close(actual.scf, expected.energy)
    torch.testing.assert_close(actual.fenergy, expected.fenergy)


def test_production_gfn2_default_broyden_route_is_native() -> None:
    """GFN2 potential/Broyden runs natively without Anderson substitution."""
    from dxtb.config import Config

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN2_XTB,
        strategy="broyden",
        differentiation="unrolled",
        maxiter=2,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_DIPOLE,
    )
    system = Model(
        par=ParamModule(GFN2_XTB, dtype=positions.dtype), config=config
    ).setup(torch.tensor([1, 1]))

    OutputHandler.clear_warnings()
    actual = singlepoint(system, positions)
    assert not any(
        "Using Anderson mixer instead" in warning[0]
        for warning in OutputHandler.warnings
    )
    assert actual.charges is not None and actual.potential is not None
    assert actual.density is not None and torch.isfinite(actual.density).all()
    assert actual.scf is not None and torch.isfinite(actual.scf).all()


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
    assert result.status is None


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
    from dxtb import SCFFailureCode

    assert result.status is not None
    assert result.status.trajectory_failure.item() == int(SCFFailureCode.NONE)
    assert result.status.final_failure.item() == int(SCFFailureCode.NONE)
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
