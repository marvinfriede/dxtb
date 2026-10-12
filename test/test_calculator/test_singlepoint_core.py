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
import importlib
import sys
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from dxtb import (
    Calculator,
    GFN1_XTB,
    GFN2_XTB,
    OutputHandler,
    ParamModule,
    labels,
)
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


def test_default_full_singlepoint_uses_functional_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default FULL singlepoint bypasses the mutable legacy full engine."""
    from dxtb._src.scf.functional.eager import FunctionalCoreResult
    from dxtb._src.scf import unrolling

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        torch.tensor([1, 1])
    )
    assert system.config.scf.scf_mode == labels.SCF_MODE_FULL
    assert system.config.scf.scp_mode == labels.SCP_MODE_POTENTIAL
    assert system.config.scf.mixer == labels.MIXER_BROYDEN
    assert system.config.scf.fermi.etemp == 300.0

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("eligible default route must use functional SCF")

    monkeypatch.setattr(unrolling, "SelfConsistentFieldFull", unexpected)
    core_module = importlib.import_module("dxtb._src.calculators.singlepoint")
    original_solve = core_module.scf.solve
    captured: dict[str, object] = {}

    def capture_solve(*args: object, **kwargs: object) -> object:
        result = original_solve(*args, **kwargs)
        captured["result"] = result
        return result

    monkeypatch.setattr(core_module.scf, "solve", capture_solve)
    OutputHandler.clear_warnings()
    result = singlepoint(system, positions)
    assert len(OutputHandler.warnings) == 1
    assert "Using Anderson mixer instead" in OutputHandler.warnings[0][0]
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


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_production_converged_route_executes_only_physical_maps(
    monkeypatch: pytest.MonkeyPatch, method: int
) -> None:
    """Eager production stops mapping after convergence, unlike fixed trips."""
    from test.test_scf.test_functional_map import _prepare_case
    from dxtb._src.scf.functional import anderson, map as functional_map
    from dxtb._src.scf.unrolling.base import BaseTSCF

    options = {
        "scp_mode": labels.SCP_MODE_POTENTIAL,
        "mixer": labels.MIXER_BROYDEN,
        "fermi_etemp": 300.0,
        "maxiter": 100,
        "verbosity": 0,
    }
    legacy_case = _prepare_case(
        method, scf_options=options, setup_functional=False
    )
    legacy_spectra = 0
    original_diagonalize = BaseTSCF.diagonalize

    def count_legacy_spectrum(self, hamiltonian):
        nonlocal legacy_spectra
        legacy_spectra += 1
        return original_diagonalize(self, hamiltonian)

    monkeypatch.setattr(BaseTSCF, "diagonalize", count_legacy_spectrum)
    legacy_result = legacy_case.engine(legacy_case.initial_charges.clone())
    legacy_raw_maps = legacy_case.engine._data.iter
    assert legacy_result["iterations"] == legacy_raw_maps
    assert legacy_spectra == legacy_raw_maps + 1

    production_case = _prepare_case(
        method, scf_options=options, setup_functional=False
    )
    functional_counts = {"raw_maps": 0, "spectra": 0}
    original_map = anderson.potential_map
    original_evaluate = functional_map.evaluate_hamiltonian

    def count_functional_map(*args, **kwargs):
        functional_counts["raw_maps"] += 1
        return original_map(*args, **kwargs)

    def count_functional_spectrum(*args, **kwargs):
        functional_counts["spectra"] += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(anderson, "potential_map", count_functional_map)
    monkeypatch.setattr(
        functional_map, "evaluate_hamiltonian", count_functional_spectrum
    )
    production_result = singlepoint(
        production_case.system, production_case.positions
    )
    assert production_result.converged is not None
    assert production_result.converged.item()
    assert production_result.iterations.item() == legacy_raw_maps
    assert functional_counts["raw_maps"] == production_result.iterations.item()
    assert functional_counts["spectra"] == functional_counts["raw_maps"] + 1


@pytest.mark.parametrize(
    ("representation", "mixer"),
    [
        (labels.SCP_MODE_CHARGE, labels.MIXER_LINEAR),
        (labels.SCP_MODE_CHARGE, labels.MIXER_ANDERSON),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_LINEAR),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_ANDERSON),
        (labels.SCP_MODE_FOCK, labels.MIXER_LINEAR),
        (labels.SCP_MODE_FOCK, labels.MIXER_ANDERSON),
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
    representation: int, mixer: int, temperature: float
) -> None:
    """Production FULL dispatch preserves legacy values in each route."""
    from dxtb._src.scf.functional import FunctionalCoreResult
    from dxtb._src.scf.unrolling import SelfConsistentFieldFull
    from test.test_scf.test_functional_map import _prepare_case

    options = {
        "scp_mode": representation,
        "mixer": mixer,
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
        "fermi_etemp": temperature,
    }
    legacy_case = _prepare_case(
        GFN1_XTB, scf_options=options, setup_functional=False
    )
    production_case = _prepare_case(
        GFN1_XTB, scf_options=options, setup_functional=False
    )
    expected = legacy_case.engine(legacy_case.initial_charges.clone())
    actual = singlepoint(production_case.system, production_case.positions)
    assert not isinstance(expected, FunctionalCoreResult)
    for name in (
        "density",
        "coefficients",
        "emo",
        "occupation",
        "hamiltonian",
    ):
        torch.testing.assert_close(getattr(actual, name), expected[name])
    torch.testing.assert_close(actual.scf, expected["energy"])
    torch.testing.assert_close(actual.fenergy, expected["fenergy"])
    assert actual.charges is not None and actual.potential is not None
    for legacy_name, result_name in (
        ("mono", "mono"),
        ("dipole", "dipole"),
        ("quad", "quadrupole"),
    ):
        expected_charge = getattr(expected["charges"], legacy_name, None)
        expected_potential = getattr(expected["potential"], legacy_name, None)
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
    assert actual.iterations.item() == expected["iterations"]
    assert actual.converged is not None and not actual.converged
    assert actual.residual is not None


@pytest.mark.parametrize(
    ("representation", "mixer"),
    [
        (labels.SCP_MODE_CHARGE, labels.MIXER_LINEAR),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_ANDERSON),
        (labels.SCP_MODE_FOCK, labels.MIXER_LINEAR),
    ],
    ids=["charge-simple", "potential-anderson", "fock-simple"],
)
@pytest.mark.parametrize("force_convergence", [False, True])
def test_production_maxiter_zero_is_valid_one_shot(
    representation: int, mixer: int, force_convergence: bool
) -> None:
    """The production boundary preserves maxiter=0 for every representation."""
    from test.test_scf.test_functional_map import _prepare_case

    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": representation,
            "mixer": mixer,
            "maxiter": 0,
            "force_convergence": force_convergence,
            "fermi_etemp": 300.0,
        },
        setup_functional=False,
    )
    expected = case.engine(case.initial_charges.clone())
    OutputHandler.clear_warnings()
    actual = singlepoint(case.system, case.positions)
    assert actual.iterations.item() == 1
    assert actual.converged is not None and not actual.converged
    assert actual.residual is not None
    assert OutputHandler.warnings == []
    torch.testing.assert_close(actual.scf, expected["energy"])
    torch.testing.assert_close(actual.fenergy, expected["fenergy"])


@pytest.mark.parametrize("force_convergence", [False, True])
def test_production_positive_budget_nonconvergence_policy(
    force_convergence: bool,
) -> None:
    """Production FULL policy warns or raises after a positive failed budget."""
    from dxtb._src.typing.exceptions import (
        SCFConvergenceError,
        SCFConvergenceWarning,
    )
    from test.test_scf.test_functional_map import _prepare_case

    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "mixer": labels.MIXER_LINEAR,
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
    "mixer",
    [labels.MIXER_LINEAR, labels.MIXER_BROYDEN],
    ids=["linear", "broyden"],
)
def test_production_negative_temperature_preserves_legacy_results(
    mixer: int,
) -> None:
    """Negative temperature keeps legacy low-T results and warning ordering."""
    from dxtb._src.scf.unrolling import SelfConsistentFieldFull
    from test.test_scf.test_functional_map import _prepare_case

    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": mixer,
            "maxiter": 1,
            "fermi_etemp": -1.0,
        },
        setup_functional=False,
    )
    OutputHandler.clear_warnings()
    engine = SelfConsistentFieldFull(
        case.interactions,
        case.occupation,
        case.n0,
        numbers=case.system.numbers,
        ihelp=case.system.ihelp,
        cache=case.interaction_data,
        integrals=case.integrals,
        config=case.system.config.scf,
        nel=case.nel,
    )
    expected = engine(case.initial_charges.clone())
    legacy_warnings = list(OutputHandler.warnings)

    OutputHandler.clear_warnings()
    actual = singlepoint(case.system, case.positions)
    assert OutputHandler.warnings == legacy_warnings
    torch.testing.assert_close(actual.scf, expected["energy"])
    torch.testing.assert_close(actual.fenergy, expected["fenergy"])
    torch.testing.assert_close(actual.density, expected["density"])
    torch.testing.assert_close(actual.hamiltonian, expected["hamiltonian"])
    assert actual.converged is None
    assert actual.residual is None


@pytest.mark.parametrize(
    ("representation", "mixer"),
    [
        (labels.SCP_MODE_CHARGE, labels.MIXER_LINEAR),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_ANDERSON),
        (labels.SCP_MODE_FOCK, labels.MIXER_ANDERSON),
    ],
    ids=["charge-simple", "potential-anderson", "fock-anderson"],
)
def test_production_dynamic_damping_matches_legacy(
    representation: int, mixer: int
) -> None:
    """Production routing retains dynamic-damping trajectory and warnings."""
    from test.test_scf.test_functional_map import _prepare_case

    maxiter = 12 if representation == labels.SCP_MODE_CHARGE else 8
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": representation,
            "mixer": mixer,
            "maxiter": maxiter,
            "mix_guess": True,
            "damp": 0.2,
            "damp_init": 0.05,
            "damp_dynamic": True,
            "damp_dynamic_factor": 0.75,
            "damp_generations": 2,
            "damp_soft_start": True,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
        setup_functional=False,
    )
    OutputHandler.clear_warnings()
    expected = case.engine(case.initial_charges.clone())
    legacy_warnings = list(OutputHandler.warnings)
    OutputHandler.clear_warnings()
    actual = singlepoint(case.system, case.positions)
    assert OutputHandler.warnings == legacy_warnings
    assert actual.iterations.item() == expected["iterations"]
    for name in (
        "density",
        "emo",
        "occupation",
        "hamiltonian",
    ):
        torch.testing.assert_close(
            getattr(actual, name),
            expected[name],
            msg=f"production field: {name}",
        )
    torch.testing.assert_close(
        actual.coefficients @ actual.coefficients.mT,
        expected["coefficients"] @ expected["coefficients"].mT,
    )
    torch.testing.assert_close(actual.scf, expected["energy"])
    torch.testing.assert_close(actual.fenergy, expected["fenergy"])


def test_production_gfn2_default_broyden_route_matches_legacy() -> None:
    """GFN2 default potential/Broyden/300 K route matches legacy outputs."""
    from dxtb._src.scf.unrolling import SelfConsistentFieldFull
    from test.test_scf.test_functional_map import _prepare_case

    options = {
        "scp_mode": labels.SCP_MODE_POTENTIAL,
        "mixer": labels.MIXER_BROYDEN,
        "fermi_etemp": 300.0,
        "maxiter": 2,
    }
    legacy_case = _prepare_case(
        GFN2_XTB,
        scf_options=options,
        setup_functional=False,
    )
    production_case = _prepare_case(
        GFN2_XTB,
        scf_options=options,
        setup_functional=False,
    )
    OutputHandler.clear_warnings()
    legacy_engine = SelfConsistentFieldFull(
        legacy_case.interactions,
        legacy_case.occupation,
        legacy_case.n0,
        numbers=legacy_case.system.numbers,
        ihelp=legacy_case.system.ihelp,
        cache=legacy_case.interaction_data,
        integrals=legacy_case.integrals,
        config=legacy_case.system.config.scf,
        nel=legacy_case.nel,
    )
    expected = legacy_engine(legacy_case.initial_charges.clone())
    legacy_warnings = list(OutputHandler.warnings)
    OutputHandler.clear_warnings()
    actual = singlepoint(production_case.system, production_case.positions)
    assert OutputHandler.warnings == legacy_warnings
    assert (
        sum(
            "Using Anderson mixer instead" in warning[0]
            for warning in legacy_warnings
        )
        == 1
    )
    for name in ("density", "emo", "occupation", "hamiltonian"):
        torch.testing.assert_close(getattr(actual, name), expected[name])
    torch.testing.assert_close(actual.scf, expected["energy"])
    torch.testing.assert_close(actual.fenergy, expected["fenergy"])
    assert actual.charges is not None and actual.potential is not None
    for legacy_name, result_name in (
        ("mono", "mono"),
        ("dipole", "dipole"),
        ("quad", "quadrupole"),
    ):
        expected_charge = getattr(expected["charges"], legacy_name, None)
        expected_potential = getattr(expected["potential"], legacy_name, None)
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
