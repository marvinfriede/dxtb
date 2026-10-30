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
"""Functional Fock-representation SCF parity and transform tests."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tad_mctc.units.energy import KELVIN2AU
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional import (
    SCFCharges,
    SCFPotential,
)
from dxtb._src.scf.functional import anderson as functional_anderson
from dxtb._src.scf.functional import (
    fock_initial_guess,
    fock_map,
    functional_subsolves_ok,
)
from dxtb._src.scf.functional import map as functional_map
from dxtb._src.scf.functional import potential_to_hamiltonian
from dxtb._src.scf.functional import problem as functional_problem
from dxtb._src.scf.functional import (
    setup_simple_solver_settings,
    solve_scf_fock_simple,
)
from dxtb._src.scf.functional import solver as functional_solver
from dxtb._src.scf.functional import validate_fock_state

from ._anderson_helpers import from_config as setup_anderson_solver_settings
from ._anderson_helpers import solve_fock as solve_scf_fock_anderson
from .test_functional_map import (
    _assert_directional_fd,
    _assert_tensor_equal,
    _prepare_case,
)


def _fock_guess(case: SimpleNamespace) -> torch.Tensor:
    layout = case.inputs.charge_layout
    initial = SCFCharges(
        mono=case.initial_charges,
        dipole=(
            None
            if layout.dipole_shape is None
            else case.initial_charges.new_zeros(layout.dipole_shape)
        ),
        quad=(
            None
            if layout.quadrupole_shape is None
            else case.initial_charges.new_zeros(layout.quadrupole_shape)
        ),
    )
    return fock_initial_guess(initial, case.inputs)


def _assert_public_parity(actual: object, expected: dict[str, object]) -> None:
    for key in (
        "density",
        "hamiltonian",
        "emo",
        "occupation",
        "energy",
        "fenergy",
    ):
        _assert_tensor_equal(getattr(actual, key), expected[key])
    _assert_tensor_equal(actual.final_charges.mono, expected["charges"].mono)
    for key in ("dipole", "quad"):
        got = getattr(actual.final_charges, key)
        want = getattr(expected["charges"], key)
        if want is None:
            assert got is None
        else:
            assert got is not None
            _assert_tensor_equal(got, want)
    for key in ("mono", "dipole", "quad"):
        got = getattr(actual.final_potential, key)
        want = getattr(expected["potential"], key)
        if want is None:
            assert got is None
        else:
            assert got is not None
            _assert_tensor_equal(got, want)
    assert int(actual.iterations) == expected["iterations"]


def _run_legacy_and_functional(
    options: dict[str, object], *, method: int = GFN1_XTB
) -> tuple[SimpleNamespace, torch.Tensor, object, dict[str, object]]:
    options = {"scp_mode": labels.SCP_MODE_FOCK, **options}
    legacy = _prepare_case(method, scf_options=options, setup_functional=False)
    functional = _prepare_case(method, scf_options=options)
    guess = _fock_guess(functional)
    _assert_tensor_equal(guess, legacy.x)
    mapped_states: list[tuple[torch.Tensor, torch.Tensor]] = []
    legacy_map = legacy.engine.iterate_fockian

    def record_map(value: torch.Tensor) -> torch.Tensor:
        mapped = legacy_map(value)
        mapped_states.append((value.clone(), mapped.clone()))
        return mapped

    legacy.engine.iterate_fockian = record_map
    legacy_result = legacy.engine(legacy.initial_charges.clone())
    config = functional.system.config.scf
    if config.mixer == labels.MIXER_LINEAR:
        settings = setup_simple_solver_settings(config, guess.dtype)
        result = solve_scf_fock_simple(guess, functional.inputs, settings)
    else:
        settings = setup_anderson_solver_settings(config, guess.dtype)
        result = solve_scf_fock_anderson(guess, functional.inputs, settings)
    expected = dict(legacy_result)
    expected["mapped_states"] = mapped_states
    return functional, guess, result, expected


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_fock_initial_guess_matches_legacy_without_spectrum(
    method: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initial Fock guess is a charge-to-potential-to-H conversion only."""
    case = _prepare_case(
        method,
        scf_options={"scp_mode": labels.SCP_MODE_FOCK},
    )
    functional_guess = _fock_guess(case)
    _assert_tensor_equal(functional_guess, case.x)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Fock initial guess performed a spectrum")

    monkeypatch.setattr(functional_map, "_transitional_diagonalize", forbidden)
    _fock_guess(case)
    assert case.system.ihelp.batch_mode == 0
    assert functional_guess.shape == case.integrals.hcore.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"charge": 1.0, "spin": 1.0},
        {"field": torch.tensor([0.002, -0.001, 0.0015], dtype=torch.float64)},
    ],
    ids=["charged-spin", "explicit-field"],
)
def test_fock_initial_guess_call_local_parity(
    kwargs: dict[str, object],
) -> None:
    """Fock guess conversion preserves explicit count and field inputs."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={"scp_mode": labels.SCP_MODE_FOCK},
        **kwargs,
    )
    _assert_tensor_equal(_fock_guess(case), case.x)


def test_fock_state_requires_single_structural_matrix() -> None:
    """Fock validation checks rank and static shape without values."""
    case = _prepare_case(GFN1_XTB, scf_options={"scp_mode": labels.SCP_MODE_FOCK})
    guess = _fock_guess(case)
    validate_fock_state(guess, case.inputs)
    with pytest.raises(ValueError, match="single-System matrix"):
        validate_fock_state(guess.unsqueeze(0), case.inputs)


@pytest.mark.parametrize(
    "temperature,method,perturb",
    [
        (0.0, GFN1_XTB, False),
        (0.0, GFN1_XTB, True),
        (300.0, GFN1_XTB, False),
        (0.5 * 3.0e-7 / KELVIN2AU, GFN1_XTB, False),
        (0.0, GFN2_XTB, False),
    ],
    ids=[
        "gfn1-initial",
        "gfn1-perturbed",
        "finite-t",
        "subcutoff",
        "gfn2-multipole",
    ],
)
def test_fock_raw_map_matches_legacy(
    temperature: float, method: int, perturb: bool
) -> None:
    """One functional Fock map matches one independent legacy iteration."""
    legacy = _prepare_case(
        method,
        scf_options={
            "scp_mode": labels.SCP_MODE_FOCK,
            "fermi_etemp": temperature,
        },
        setup_functional=False,
    )
    functional = _prepare_case(
        method,
        scf_options={
            "scp_mode": labels.SCP_MODE_FOCK,
            "fermi_etemp": temperature,
        },
    )
    x = _fock_guess(functional)
    if perturb:
        delta = torch.arange(x.numel(), dtype=x.dtype).reshape_as(x) / 1000
        delta = delta + delta.mT
        x = x + delta
    x_legacy = x.clone()
    expected_h = legacy.engine.iterate_fockian(x_legacy)
    actual = fock_map(x, functional.inputs)
    _assert_tensor_equal(actual.mapped_x, expected_h)
    _assert_tensor_equal(actual.spectral.hamiltonian, x)
    _assert_tensor_equal(actual.density, legacy.engine._data.density)
    _assert_tensor_equal(actual.emo, legacy.engine._data.evals)
    _assert_tensor_equal(actual.occupation, legacy.engine._data.occupation)
    _assert_tensor_equal(actual.orbital_energy, legacy.engine._data.energy)
    actual_projector = (
        actual.coefficients
        @ torch.diag_embed(actual.occupation.sum(-2))
        @ actual.coefficients.mT
    )
    expected_projector = (
        legacy.engine._data.evecs
        @ torch.diag_embed(legacy.engine._data.occupation.sum(-2))
        @ legacy.engine._data.evecs.mT
    )
    _assert_tensor_equal(actual_projector, expected_projector)
    assert actual.driving_potential is None
    for component in ("mono", "dipole", "quad"):
        got = getattr(actual.charges, component)
        if got is None:
            continue
        assert torch.isfinite(got).all()
    assert torch.allclose(actual.mapped_x, actual.mapped_x.mT, atol=2e-13, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"charge": 1.0, "spin": 1.0},
        {"field": torch.tensor([0.002, -0.001, 0.0015], dtype=torch.float64)},
    ],
    ids=["charged-spin", "explicit-field"],
)
def test_fock_raw_map_charged_spin_and_field(kwargs: dict[str, object]) -> None:
    """Fock raw maps retain explicit count and call-local field inputs."""
    legacy = _prepare_case(
        GFN1_XTB,
        scf_options={"scp_mode": labels.SCP_MODE_FOCK},
        setup_functional=False,
        **kwargs,
    )
    functional = _prepare_case(
        GFN1_XTB,
        scf_options={"scp_mode": labels.SCP_MODE_FOCK},
        **kwargs,
    )
    x = _fock_guess(functional)
    mapped = legacy.engine.iterate_fockian(x.clone())
    result = fock_map(x, functional.inputs)
    _assert_tensor_equal(result.mapped_x, mapped)
    _assert_tensor_equal(result.density, legacy.engine._data.density)
    _assert_tensor_equal(result.occupation, legacy.engine._data.occupation)


@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_fock_simple_finite_budget_matches_legacy(
    maxiter: int, mix_guess: bool
) -> None:
    """Fock Simple preserves finite-budget result phases and counts."""
    _, _, result, legacy = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_LINEAR,
            "maxiter": maxiter,
            "mix_guess": mix_guess,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        }
    )
    _assert_public_parity(result, legacy)
    assert len(legacy["mapped_states"]) == maxiter + 1
    _assert_tensor_equal(result.map_result.mapped_x, legacy["mapped_states"][-1][1])
    last_input, last_output = legacy["mapped_states"][-1]
    delta = last_output - last_input
    expected_residual = torch.stack(
        (torch.linalg.vector_norm(delta), delta.abs().amax())
    )
    torch.testing.assert_close(result.residual, expected_residual, rtol=0.0, atol=2e-13)
    _assert_tensor_equal(result.hamiltonian, result.map_result.mapped_x)
    _assert_tensor_equal(
        result.final_evaluation.hamiltonian, result.map_result.mapped_x
    )
    assert result.final_evaluation is not result.map_result.spectral
    assert not bool(result.converged)
    if maxiter == 0:
        assert not torch.equal(
            result.final_charges.mono, result.map_result.charges.mono
        )


@pytest.mark.parametrize(
    "mixer",
    [labels.MIXER_LINEAR, labels.MIXER_ANDERSON],
    ids=["simple", "anderson"],
)
def test_fock_normal_convergence_matches_legacy(mixer: int) -> None:
    """Fock Simple and Anderson converge to matching public results."""
    options: dict[str, object] = {
        "mixer": mixer,
        "maxiter": 80,
        "mix_guess": True,
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-8,
    }
    if mixer == labels.MIXER_ANDERSON:
        options.update({"damp_generations": 2, "damp_soft_start": False})
    _, _, result, legacy = _run_legacy_and_functional(options)
    _assert_public_parity(result, legacy)
    assert bool(result.converged)


def test_fock_anderson_early_convergence_matches_selected_map() -> None:
    """Raw convergence alone cannot stop before Fock certification passes."""
    case, _, result, _ = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_ANDERSON,
            "maxiter": 8,
            "mix_guess": True,
            "damp_generations": 2,
            "damp_soft_start": False,
            "x_atol": 1.0e6,
            "x_atol_max": 1.0e6,
        }
    )
    assert bool(result.converged)
    assert int(result.iterations) > 2
    assert result.diagnostics.certification is not None
    assert bool(result.diagnostics.certification.result.certified)
    expected = functional_problem.fock_map(result.root_x, case.inputs)
    _assert_tensor_equal(result.map_result.mapped_x, expected.mapped_x)
    _assert_tensor_equal(result.map_result.spectral.density, expected.spectral.density)


@pytest.mark.parametrize(
    "partition,temperature,charge",
    [
        (labels.FERMI_PARTITION_EQUAL, 300.0, 0.0),
        (labels.FERMI_PARTITION_ATOMIC, 300.0, 0.0),
        (
            labels.FERMI_PARTITION_EQUAL,
            0.5 * 3.0e-7 / KELVIN2AU,
            0.25,
        ),
    ],
    ids=["300K-equal", "300K-atomic", "subcutoff-fractional"],
)
def test_fock_final_free_energy_matches_legacy(
    partition: int, temperature: float, charge: float
) -> None:
    """Fock free energy is sourced from the final public spectrum."""
    _, _, result, legacy = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 0,
            "fermi_etemp": temperature,
            "fermi_partition": partition,
        }
    )
    if charge != 0.0:
        # Reprepare both paths independently with the fractional target.
        legacy_case = _prepare_case(
            GFN1_XTB,
            charge=charge,
            scf_options={
                "scp_mode": labels.SCP_MODE_FOCK,
                "mixer": labels.MIXER_LINEAR,
                "maxiter": 0,
                "fermi_etemp": temperature,
                "fermi_partition": partition,
            },
            setup_functional=False,
        )
        functional_case = _prepare_case(
            GFN1_XTB,
            charge=charge,
            scf_options={
                "scp_mode": labels.SCP_MODE_FOCK,
                "mixer": labels.MIXER_LINEAR,
                "maxiter": 0,
                "fermi_etemp": temperature,
                "fermi_partition": partition,
            },
        )
        guess = _fock_guess(functional_case)
        settings = setup_simple_solver_settings(
            functional_case.system.config.scf, guess.dtype
        )
        result = solve_scf_fock_simple(guess, functional_case.inputs, settings)
        legacy = legacy_case.engine(legacy_case.initial_charges.clone())
    _assert_tensor_equal(result.fenergy, legacy["fenergy"])
    _assert_tensor_equal(result.energy, legacy["energy"])


def test_fock_anderson_intermediate_matrices_remain_symmetric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial, mapped, and accelerated solver Fock matrices stay symmetric."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_FOCK,
            "mixer": labels.MIXER_ANDERSON,
            "maxiter": 5,
            "mix_guess": True,
            "damp_generations": 1,
            "damp_soft_start": False,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )
    guess = _fock_guess(case)
    states: list[torch.Tensor] = []
    original = functional_problem.fock_map

    def record(value: torch.Tensor, inputs: object):
        states.append(value.clone())
        result = original(value, inputs)
        states.append(result.mapped_x.clone())
        return result

    monkeypatch.setattr(functional_problem, "fock_map", record)
    settings = setup_anderson_solver_settings(case.system.config.scf, guess.dtype)
    result = solve_scf_fock_anderson(guess, case.inputs, settings)
    assert len(states) > 2
    for value in (
        guess,
        result.map_result.mapped_x,
        result.hamiltonian,
        *states,
    ):
        torch.testing.assert_close(value, value.mT, rtol=0.0, atol=2e-13)


@pytest.mark.parametrize("mixer", [labels.MIXER_LINEAR, labels.MIXER_ANDERSON])
def test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum(
    mixer: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-shot Fock solve has one map spectrum and one final spectrum."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_FOCK,
            "mixer": mixer,
            "maxiter": 0,
            "fermi_etemp": 300.0,
        },
    )
    guess = _fock_guess(case)
    settings = (
        setup_simple_solver_settings(case.system.config.scf, guess.dtype)
        if mixer == labels.MIXER_LINEAR
        else setup_anderson_solver_settings(case.system.config.scf, guess.dtype)
    )
    calls = 0
    original = functional_map._transitional_diagonalize

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(functional_map, "_transitional_diagonalize", counted)
    if mixer == labels.MIXER_LINEAR:
        result = solve_scf_fock_simple(guess, case.inputs, settings)
    else:
        result = solve_scf_fock_anderson(guess, case.inputs, settings)
    assert calls == 2
    assert int(result.iterations) == 1
    assert not bool(result.converged)
    assert result.trajectory_status is None
    assert result.map_result.fermi_status is not None
    assert result.final_evaluation.fermi_status is not None
    assert bool(functional_subsolves_ok(result))


@pytest.mark.parametrize("temperature", [0.0, 300.0])
def test_fock_anderson_accelerates_and_preserves_symmetry(
    temperature: float,
) -> None:
    """Accelerated full-matrix Anderson updates remain symmetric and legacy-compatible."""
    case, guess, result, legacy = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_ANDERSON,
            "maxiter": 8,
            "mix_guess": True,
            "damp_generations": 2,
            "damp_soft_start": False,
            "fermi_etemp": temperature,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        }
    )
    _assert_public_parity(result, legacy)
    assert len(legacy["mapped_states"]) == 9
    assert torch.allclose(guess, guess.mT, atol=2e-13, rtol=0)
    assert torch.allclose(
        result.map_result.mapped_x,
        result.map_result.mapped_x.mT,
        atol=2e-13,
        rtol=0,
    )
    assert torch.allclose(result.hamiltonian, result.hamiltonian.mT, atol=2e-13, rtol=0)


def test_fock_gfn2_multipole_solver_matches_legacy() -> None:
    """GFN2 Fock solving retains monopole, dipole, and quadrupole physics."""
    _, _, result, legacy = _run_legacy_and_functional(
        {
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "mix_guess": False,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 300.0,
        },
        method=GFN2_XTB,
    )
    _assert_public_parity(result, legacy)
    assert result.final_charges.dipole is not None
    assert result.final_charges.quad is not None
    assert result.final_potential.dipole is not None
    assert result.final_potential.quad is not None


def test_fock_position_derivatives_match_physical_fd() -> None:
    """Fock density response matches physical finite differences."""
    options: dict[str, object] = {
        "scp_mode": labels.SCP_MODE_FOCK,
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 1,
        "mix_guess": False,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    initial = _prepare_case(GFN1_XTB, scf_options=options)
    positions = initial.positions.clone().requires_grad_(True)
    functional = _prepare_case(GFN1_XTB, positions=positions, scf_options=options)
    guess = _fock_guess(functional)
    settings = setup_simple_solver_settings(
        functional.system.config.scf, positions.dtype
    )
    energy = solve_scf_fock_simple(guess, functional.inputs, settings).energy.sum()
    grad = torch.autograd.grad(energy, positions, create_graph=True)[0]
    hess = torch.autograd.grad(grad.square().sum(), positions)[0]
    assert torch.isfinite(grad).all()
    assert torch.isfinite(hess).all()

    def energy_at_positions(value: torch.Tensor) -> torch.Tensor:
        case = _prepare_case(GFN1_XTB, positions=value, scf_options=options)
        initial = _fock_guess(case)
        local_settings = setup_simple_solver_settings(
            case.system.config.scf, value.dtype
        )
        return solve_scf_fock_simple(initial, case.inputs, local_settings).energy.sum()

    _assert_directional_fd(energy_at_positions, positions, grad)


def test_fock_parameter_and_field_gradients() -> None:
    """Fock finalization preserves parameter and explicit-field graphs."""
    options: dict[str, object] = {
        "scp_mode": labels.SCP_MODE_FOCK,
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 0,
        "fermi_etemp": 300.0,
    }
    parameter = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    leaf = parameter.get("element.O.levels")
    leaf.requires_grad_(True)
    case = _prepare_case(GFN1_XTB, par=parameter, scf_options=options)
    guess = _fock_guess(case)
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)
    result = solve_scf_fock_simple(guess, case.inputs, settings)
    total = (result.energy + result.fenergy).sum()
    grad = torch.autograd.grad(total, leaf)[0]
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0

    field = torch.tensor(
        [0.001, -0.0007, 0.0004], dtype=torch.float64, requires_grad=True
    )
    field_case = _prepare_case(GFN1_XTB, field=field, scf_options=options)
    field_guess = _fock_guess(field_case)
    field_settings = setup_simple_solver_settings(
        field_case.system.config.scf, field_guess.dtype
    )
    field_result = solve_scf_fock_simple(field_guess, field_case.inputs, field_settings)
    field_grad = torch.autograd.grad(
        (field_result.energy + field_result.fenergy).sum(), field
    )[0]
    assert torch.isfinite(field_grad).all()
    assert torch.count_nonzero(field_grad) > 0

    def total_at_field(value: torch.Tensor) -> torch.Tensor:
        case = _prepare_case(GFN1_XTB, field=value, scf_options=options)
        initial = _fock_guess(case)
        local_settings = setup_simple_solver_settings(
            case.system.config.scf, value.dtype
        )
        result = solve_scf_fock_simple(initial, case.inputs, local_settings)
        return (result.energy + result.fenergy).sum()

    _assert_directional_fd(total_at_field, field, field_grad)


def test_fock_fractional_charge_gradient_matches_physical_fd() -> None:
    """Finite-temperature Fock finalization preserves charge/electron AD."""
    charge = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    options: dict[str, object] = {
        "scp_mode": labels.SCP_MODE_FOCK,
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 0,
        "fermi_etemp": 300.0,
    }
    functional = _prepare_case(GFN1_XTB, charge=charge, scf_options=options)
    guess = _fock_guess(functional)
    settings = setup_simple_solver_settings(functional.system.config.scf, guess.dtype)
    result = solve_scf_fock_simple(guess, functional.inputs, settings)
    actual = torch.autograd.grad((result.energy + result.fenergy).sum(), charge)[0]

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) > 0

    def total_at_charge(value: torch.Tensor) -> torch.Tensor:
        case = _prepare_case(GFN1_XTB, charge=value, scf_options=options)
        initial = _fock_guess(case)
        local_settings = setup_simple_solver_settings(
            case.system.config.scf, initial.dtype
        )
        result = solve_scf_fock_simple(initial, case.inputs, local_settings)
        return (result.energy + result.fenergy).sum()

    _assert_directional_fd(total_at_charge, charge, actual)


def test_fock_map_reverse_jvp_jacfwd_are_finite() -> None:
    """Fock map transforms remain finite for symmetric matrix directions."""
    case = _prepare_case(GFN1_XTB, scf_options={"scp_mode": labels.SCP_MODE_FOCK})
    x = _fock_guess(case)
    direction = torch.arange(x.numel(), dtype=x.dtype).reshape_as(x) / 1000
    direction = direction + direction.mT
    scalar = lambda z: fock_map(z, case.inputs).mapped_x.sum()
    reverse = torch.autograd.grad(scalar(x.requires_grad_()), x)[0]
    tangent = jvp(scalar, (x.detach(),), (direction,))[1]
    forward = jacfwd(scalar)(x.detach())
    assert torch.isfinite(reverse).all()
    assert torch.isfinite(tangent)
    assert torch.isfinite(forward).all()
    assert torch.isfinite(jacrev(scalar)(x.detach())).all()


def test_potential_to_hamiltonian_transform_ladder_is_consistent() -> None:
    """The smooth potential assembly subpath agrees across AD modes."""
    case = _prepare_case(GFN2_XTB, scf_options={"scp_mode": labels.SCP_MODE_FOCK})
    point = case.integrals.hcore.new_zeros(
        case.inputs.charge_layout.mono_shape
    ).requires_grad_()

    def scalar(mono: torch.Tensor) -> torch.Tensor:
        hamiltonian = potential_to_hamiltonian(
            case.integrals.hcore,
            case.integrals.overlap,
            case.integrals.dipole,
            case.integrals.quadrupole,
            case.inputs.index,
            SCFPotential(mono=mono),
        )
        return hamiltonian.square().sum()

    gradient = torch.autograd.grad(scalar(point), point)[0]
    tangent = torch.ones_like(point) / point.numel() ** 0.5
    _, directional = jvp(scalar, (point.detach(),), (tangent,))
    reverse_jacobian = jacrev(scalar)(point.detach())
    forward_jacobian = jacfwd(scalar)(point.detach())
    torch.testing.assert_close(
        directional,
        (reverse_jacobian * tangent).sum(),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        forward_jacobian, reverse_jacobian, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(gradient, reverse_jacobian, rtol=1e-12, atol=1e-12)


def test_fock_solver_vmap_matches_loop() -> None:
    """Fixed-System Fock solves map over symmetric starting matrices."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_FOCK,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "mix_guess": False,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
            "fermi_etemp": 300.0,
        },
    )
    guess = _fock_guess(case)
    perturb = torch.arange(guess.numel(), dtype=guess.dtype).reshape_as(guess)
    perturb = (perturb + perturb.mT) * 1.0e-5
    guesses = torch.stack((guess, guess + perturb))
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)

    def run(x: torch.Tensor):
        result = solve_scf_fock_simple(x, case.inputs, settings)
        return (
            result.energy,
            result.fenergy,
            result.iterations,
            result.converged,
            result.trajectory_status.fermi_converged,
            result.trajectory_status.electron_count_ok,
            result.final_evaluation.fermi_status.converged,
        )

    mapped = vmap(run)(guesses)
    looped = tuple(torch.stack([run(x)[i] for x in guesses]) for i in range(7))
    for actual, expected in zip(mapped, looped):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-13)


def test_fock_raw_failure_survives_valid_later_and_final_spectra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fock trajectory status keeps an earlier failed physical spectrum."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_FOCK,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "mix_guess": False,
            "fermi_etemp": 300.0,
            "x_atol": 1.0e6,
            "x_atol_max": 1.0e6,
        },
    )
    guess = _fock_guess(case)
    original = functional_problem.fock_map
    calls = 0

    def fail_initial(value: torch.Tensor, inputs: object):
        nonlocal calls
        result = original(value, inputs)
        calls += 1
        if calls == 1:
            assert result.fermi_status is not None
            status = replace(
                result.fermi_status,
                converged=torch.zeros_like(result.fermi_status.converged),
            )
            result = replace(
                result,
                spectral=replace(result.spectral, fermi_status=status),
            )
        return result

    monkeypatch.setattr(functional_problem, "fock_map", fail_initial)
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)
    result = solve_scf_fock_simple(guess, case.inputs, settings)
    assert calls == 2
    assert not bool(result.converged)
    assert result.diagnostics.certification is not None
    assert not bool(result.diagnostics.certification.result.stationarity_ok)
    assert result.trajectory_status is not None
    assert not bool(result.trajectory_status.fermi_converged.all())
    assert result.map_result.fermi_status is not None
    assert bool(result.map_result.fermi_status.converged.all())
    assert result.final_evaluation.fermi_status is not None
    assert bool(result.final_evaluation.fermi_status.converged.all())
    assert not bool(functional_subsolves_ok(result))


def test_fock_zero_budget_uses_final_status_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-budget Fock validity comes from the separate final spectrum."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "scp_mode": labels.SCP_MODE_FOCK,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 0,
            "fermi_etemp": 300.0,
        },
    )
    guess = _fock_guess(case)
    settings = setup_simple_solver_settings(case.system.config.scf, guess.dtype)
    original = functional_solver.evaluate_hamiltonian

    def fail_final(hamiltonian: torch.Tensor, inputs: object):
        result = original(hamiltonian, inputs)
        assert result.fermi_status is not None
        status = replace(
            result.fermi_status,
            converged=torch.zeros_like(result.fermi_status.converged),
        )
        return replace(result, fermi_status=status)

    monkeypatch.setattr(functional_solver, "evaluate_hamiltonian", fail_final)
    result = solve_scf_fock_simple(guess, case.inputs, settings)
    assert result.trajectory_status is None
    assert result.map_result.fermi_status is not None
    assert bool(result.map_result.fermi_status.converged.all())
    assert result.final_evaluation.fermi_status is not None
    assert not bool(result.final_evaluation.fermi_status.converged.all())
    assert not bool(functional_subsolves_ok(result))


def test_fock_path_has_no_value_based_padding_cleanup() -> None:
    """Single-System Fock source does not interpret matrix values as padding."""
    source = inspect.getsource(functional_map.fock_map)
    assert "PADNZ" not in source
    assert "!= defaults.PADNZ" not in source
    assert "== defaults.PADNZ" not in source
