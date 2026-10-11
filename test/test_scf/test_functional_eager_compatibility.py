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
"""Static routing and eager-policy characterization for functional SCF."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from dxtb import GFN1_XTB, GFN2_XTB, OutputHandler, labels

from dxtb._src.calculators.config.scf import ConfigFermi
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.interactions import InteractionList
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.field.efieldgrad import ElectricFieldGrad
from dxtb._src.scf.functional import (
    resolve_full_scf_route,
    run_functional_full_eager,
)
from dxtb._src.scf.functional.solver import (
    SCFTrajectoryStatus,
)
from dxtb._src.scf.functional.eager import _check_subsolve_status
from dxtb._src.scf.functional import anderson as functional_anderson
from dxtb._src.scf.functional import solver as functional_solver
from dxtb._src.wavefunction.filling import FermiStatus

from .test_functional_map import _prepare_case, _assert_tensor_equal
from .test_functional_fock import _assert_public_parity


def test_default_full_route_resolves_potential_and_broyden_compatibility() -> (
    None
):
    """Default full config resolves the currently supported default route."""
    case = _prepare_case(GFN1_XTB)
    config = replace(
        case.system.config.scf,
        scp_mode=labels.SCP_MODE_POTENTIAL,
        fermi=ConfigFermi(),
    )
    route = resolve_full_scf_route(config, case.interactions, case.system.ihelp)
    assert route.kind == "functional"
    assert route.representation == labels.SCP_MODE_POTENTIAL
    assert route.effective_mixer == labels.MIXER_ANDERSON
    assert route.warning == (
        "Broyden mixer is not implemented for SCF with full gradient tracking. "
        "Using Anderson mixer instead."
    )
    assert config.fermi.etemp == 300.0


def test_broyden_strict_is_rejected_by_static_resolver() -> None:
    """Strict configured Broyden fails before any functional setup."""
    case = _prepare_case(GFN1_XTB)
    config = replace(case.system.config.scf, strict=True)
    with pytest.raises(NotImplementedError, match="Broyden mixer"):
        resolve_full_scf_route(config, case.interactions, case.system.ihelp)


@pytest.mark.parametrize(
    "mode",
    [labels.SCF_MODE_IMPLICIT, labels.SCF_MODE_EXPERIMENTAL],
    ids=["implicit", "experimental"],
)
def test_non_full_modes_remain_legacy_owned(mode: int) -> None:
    """The full functional route never absorbs other solver modes."""
    case = _prepare_case(GFN1_XTB)
    route = resolve_full_scf_route(
        replace(case.system.config.scf, scf_mode=mode),
        case.interactions,
        case.system.ihelp,
    )
    assert route.kind == "legacy"


def test_nonselfconsistent_method_remains_legacy_owned() -> None:
    """The functional SCF route is not used for GFN0-like methods."""
    case = _prepare_case(GFN1_XTB)
    config = replace(case.system.config.scf, method=labels.GFN0_XTB)
    assert not config.requires_iterations
    route = resolve_full_scf_route(config, case.interactions, case.system.ihelp)
    assert route.kind == "legacy"


def test_batched_numbers_remain_legacy_owned() -> None:
    """A batched System request cannot enter the single-System adapter."""
    case = _prepare_case(GFN1_XTB)
    route = resolve_full_scf_route(
        case.system.config.scf,
        case.interactions,
        case.system.ihelp,
        case.system.numbers.unsqueeze(0),
    )
    assert route.kind == "legacy"


@pytest.mark.parametrize("field_kind", ["field", "field_grad"])
def test_exact_call_local_field_interactions_remain_functional_eligible(
    field_kind: str,
) -> None:
    """Exact built-in call-local field terms keep the functional route."""
    case = _prepare_case(GFN1_XTB)
    call_local = (
        ElectricField(torch.zeros(3, dtype=torch.float64))
        if field_kind == "field"
        else ElectricFieldGrad(torch.zeros((3, 3), dtype=torch.float64))
    )
    interactions = InteractionList(
        *case.system.interactions.components, call_local
    )

    route = resolve_full_scf_route(
        case.system.config.scf, interactions, case.system.ihelp
    )

    assert route.kind == "functional"


@pytest.mark.parametrize(
    ("category", "exception", "message"),
    [
        ("invalid_temperature", ValueError, "Electronic Temperature"),
        ("too_many_electrons", ValueError, "exceeds the number of orbitals"),
        ("fermi_converged", RuntimeError, "Fermi energy failed to converge"),
        ("electron_count_ok", RuntimeError, "electrons changed"),
    ],
)
def test_eager_subsolve_error_priority(
    category: str, exception: type[Exception], message: str
) -> None:
    """The eager layer preserves the legacy Fermi error families."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "scp_mode": labels.SCP_MODE_CHARGE,
            "maxiter": 0,
        },
    )
    result = run_functional_full_eager(
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
    assert result is not None
    values = {
        "fermi_converged": True,
        "electron_count_ok": True,
        "invalid_temperature": False,
        "too_many_electrons": False,
    }
    values[category] = not values[category]

    def status_tensor(value: bool) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.bool)

    status = FermiStatus(
        converged=status_tensor(values["fermi_converged"]),
        residual=torch.zeros((), dtype=torch.float64),
        invalid_temperature=status_tensor(values["invalid_temperature"]),
        too_many_electrons=status_tensor(values["too_many_electrons"]),
        electron_count_ok=status_tensor(values["electron_count_ok"]),
    )
    solve_result = result.solve_result
    final_evaluation = replace(
        solve_result.final_evaluation, fermi_status=status
    )
    trajectory = SCFTrajectoryStatus(
        fermi_converged=status.converged,
        electron_count_ok=status.electron_count_ok,
        invalid_temperature=status.invalid_temperature,
        too_many_electrons=status.too_many_electrons,
    )
    synthetic = replace(
        solve_result,
        final_evaluation=final_evaluation,
        trajectory_status=trajectory,
    )
    with pytest.raises(exception, match=message):
        _check_subsolve_status(synthetic)


def test_custom_interaction_routes_to_legacy_and_keeps_extension_behavior() -> (
    None
):
    """A custom field subclass is rejected before setup and runs via legacy."""
    field = torch.tensor([0.002, -0.001, 0.0015], dtype=torch.float64)
    case = _prepare_case(
        GFN1_XTB,
        field=field,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "scp_mode": labels.SCP_MODE_CHARGE,
            "maxiter": 2,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
    )

    class ExtendedField(ElectricField):
        """A legacy extension that shifts the atom field potential."""

        def __init__(self, value: torch.Tensor) -> None:
            super().__init__(value)
            self.calls = 0

        def get_monopole_atom_potential(self, cache, *args, **kwargs):
            self.calls += 1
            return (
                super().get_monopole_atom_potential(cache, *args, **kwargs)
                + 0.01
            )

    custom_field = ExtendedField(field)
    components = [
        component
        for component in case.interactions.components
        if type(component) is not ElectricField
    ]
    interactions = InteractionList(
        *components,
        custom_field,
        device=case.positions.device,
        dtype=case.positions.dtype,
    )
    route = resolve_full_scf_route(
        case.system.config.scf, interactions, case.system.ihelp
    )
    assert route.kind == "legacy"
    cache = _interaction_data(case.system, case.positions, interactions)
    result = run_functional_full_eager(
        case.system.numbers,
        case.positions,
        case.charge,
        case.spin,
        interactions,
        cache,
        case.system.ihelp,
        case.system.config.scf,
        case.integrals,
        case.refocc,
    )
    assert result.functional is None
    assert result.legacy is not None
    assert custom_field.calls > 0


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
def test_default_broyden_adapter_matches_legacy(method: int) -> None:
    """The direct eager adapter preserves default Anderson fallback parity."""
    options = {"scp_mode": labels.SCP_MODE_POTENTIAL, "fermi_etemp": 300.0}
    OutputHandler.clear_warnings()
    legacy_case = _prepare_case(method, scf_options=options)
    legacy_warnings = list(OutputHandler.warnings)
    OutputHandler.clear_warnings()
    functional_case = _prepare_case(method, scf_options=options)

    OutputHandler.clear_warnings()
    legacy = legacy_case.engine(legacy_case.initial_charges.clone())

    actual = run_functional_full_eager(
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
    )
    assert actual.functional is not None
    assert actual.legacy is None
    assert actual.route.effective_mixer == labels.MIXER_ANDERSON
    assert OutputHandler.warnings == legacy_warnings
    for name in (
        "density",
        "hamiltonian",
        "emo",
        "occupation",
        "energy",
        "fenergy",
    ):
        _assert_tensor_equal(getattr(actual.functional, name), legacy[name])
    _assert_tensor_equal(actual.functional.charges.mono, legacy["charges"].mono)
    _assert_tensor_equal(
        actual.functional.potential.mono, legacy["potential"].mono
    )
    assert int(actual.functional.iterations) == legacy["iterations"]
    assert isinstance(actual.functional.iterations, torch.Tensor)
    assert isinstance(actual.functional.converged, torch.Tensor)
    assert isinstance(actual.functional.residual, torch.Tensor)


@pytest.mark.parametrize(
    "representation",
    [labels.SCP_MODE_CHARGE, labels.SCP_MODE_POTENTIAL, labels.SCP_MODE_FOCK],
    ids=["charge", "potential", "fock"],
)
@pytest.mark.parametrize(
    "mixer",
    [labels.MIXER_LINEAR, labels.MIXER_ANDERSON],
    ids=["simple", "anderson"],
)
@pytest.mark.parametrize("temperature", [0.0, 300.0], ids=["zero-k", "300-k"])
def test_eager_representation_mixer_budget_parity(
    representation: int, mixer: int, temperature: float
) -> None:
    """All representation/mixer pairs preserve short-budget eager results."""
    options = {
        "scp_mode": representation,
        "mixer": mixer,
        "fermi_etemp": temperature,
        "maxiter": 2,
        "mix_guess": True,
        "damp_generations": 1,
        "damp_soft_start": False,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    legacy_case = _prepare_case(GFN1_XTB, scf_options=options)
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)

    OutputHandler.clear_warnings()
    legacy = legacy_case.engine(legacy_case.initial_charges.clone())
    legacy_warnings = list(OutputHandler.warnings)
    OutputHandler.clear_warnings()
    actual = run_functional_full_eager(
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
    )
    assert actual.functional is not None
    _assert_public_parity(actual.functional.solve_result, legacy)
    assert OutputHandler.warnings == legacy_warnings


@pytest.mark.parametrize(
    "representation",
    [labels.SCP_MODE_CHARGE, labels.SCP_MODE_POTENTIAL, labels.SCP_MODE_FOCK],
    ids=["charge", "potential", "fock"],
)
@pytest.mark.parametrize(
    "mixer",
    [labels.MIXER_LINEAR, labels.MIXER_ANDERSON],
    ids=["simple", "anderson"],
)
def test_eager_converged_representation_mixer_parity(
    representation: int, mixer: int
) -> None:
    """Converged eager results agree across all representations and mixers."""
    options = {
        "scp_mode": representation,
        "mixer": mixer,
        "fermi_etemp": 0.0,
        "maxiter": 100,
        "mix_guess": True,
        "damp_generations": 2,
        "damp_soft_start": False,
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-8,
    }
    legacy_case = _prepare_case(GFN1_XTB, scf_options=options)
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)
    OutputHandler.clear_warnings()
    legacy = legacy_case.engine(legacy_case.initial_charges.clone())
    legacy_warnings = list(OutputHandler.warnings)
    OutputHandler.clear_warnings()
    actual = run_functional_full_eager(
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
    )
    assert actual.functional is not None
    assert bool(actual.functional.converged)
    _assert_public_parity(actual.functional.solve_result, legacy)
    assert OutputHandler.warnings == legacy_warnings


def test_core_result_boundary_keeps_solver_status_tensors() -> None:
    """Functional core conversion preserves all status tensors directly."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "scp_mode": labels.SCP_MODE_CHARGE,
            "maxiter": 0,
        },
    )
    result = run_functional_full_eager(
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
    assert result is not None
    assert isinstance(result.iterations, torch.Tensor)
    assert isinstance(result.converged, torch.Tensor)
    assert isinstance(result.residual, torch.Tensor)
    assert result.iterations.ndim == 0


@pytest.mark.parametrize(
    "representation",
    [labels.SCP_MODE_CHARGE, labels.SCP_MODE_POTENTIAL, labels.SCP_MODE_FOCK],
    ids=["charge", "potential", "fock"],
)
@pytest.mark.parametrize(
    "mixer",
    [labels.MIXER_LINEAR, labels.MIXER_ANDERSON],
    ids=["simple", "anderson"],
)
def test_dynamic_damping_adapter_matches_legacy(
    representation: int, mixer: int
) -> None:
    """Dynamic damping agrees with legacy after the raw norm crosses 0.1."""
    options = {
        "scp_mode": representation,
        "mixer": mixer,
        "maxiter": 12,
        "mix_guess": True,
        "damp": 0.2,
        "damp_init": 0.05,
        "damp_dynamic": True,
        "damp_dynamic_factor": 0.75,
        "damp_generations": 2,
        "damp_soft_start": True,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    legacy_case = _prepare_case(GFN1_XTB, scf_options=options)
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)
    OutputHandler.clear_warnings()
    legacy = legacy_case.engine(legacy_case.initial_charges.clone())
    legacy_warnings = list(OutputHandler.warnings)
    OutputHandler.clear_warnings()
    actual = run_functional_full_eager(
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
    )
    assert actual.functional is not None
    _assert_public_parity(actual.functional.solve_result, legacy)
    assert actual.functional.solve_result.diagnostics is not None
    torch.testing.assert_close(
        actual.functional.solve_result.diagnostics.final_damp,
        torch.tensor(0.75, dtype=torch.float64),
    )
    assert OutputHandler.warnings == legacy_warnings


@pytest.mark.parametrize(
    ("representation", "mixer"),
    [
        (labels.SCP_MODE_CHARGE, labels.MIXER_LINEAR),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_LINEAR),
        (labels.SCP_MODE_FOCK, labels.MIXER_LINEAR),
        (labels.SCP_MODE_POTENTIAL, labels.MIXER_ANDERSON),
        (labels.SCP_MODE_FOCK, labels.MIXER_ANDERSON),
    ],
    ids=[
        "charge-simple",
        "potential-simple",
        "fock-simple",
        "potential-anderson",
        "fock-anderson",
    ],
)
def test_dynamic_damping_iterate_and_switch_parity(
    representation: int,
    mixer: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every raw iterate and ordinary-damping switch matches legacy."""
    maxiter = (
        12
        if representation == labels.SCP_MODE_CHARGE
        and mixer == labels.MIXER_LINEAR
        else 8
    )
    options = {
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
    }
    legacy_case = _prepare_case(GFN1_XTB, scf_options=options)
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)

    legacy_transitions: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    engine = legacy_case.engine
    legacy_mixer = engine.mixer
    legacy_iter = legacy_mixer.iter

    def record_legacy_transition(
        mapped: torch.Tensor, old: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert old is not None
        legacy_transitions.append(
            (old.clone(), mapped.clone(), float(legacy_mixer.options["damp"]))
        )
        return legacy_iter(mapped, old)

    monkeypatch.setattr(legacy_mixer, "iter", record_legacy_transition)
    engine(legacy_case.initial_charges.clone())

    map_inputs: list[torch.Tensor] = []
    transition_damps: list[torch.Tensor] = []
    module = (
        functional_solver
        if mixer == labels.MIXER_LINEAR
        else functional_anderson
    )
    map_name = {
        labels.SCP_MODE_CHARGE: "scf_map",
        labels.SCP_MODE_POTENTIAL: "potential_map",
        labels.SCP_MODE_FOCK: "fock_map",
    }[representation]
    original_map = getattr(module, map_name)

    def record_functional_map(x: torch.Tensor, *args, **kwargs):
        map_inputs.append(x.clone())
        return original_map(x, *args, **kwargs)

    monkeypatch.setattr(module, map_name, record_functional_map)
    if mixer == labels.MIXER_LINEAR:
        original_mix = functional_solver.simple_mix

        def record_simple_mix(old, mapped, damp):
            transition_damps.append(torch.as_tensor(damp).detach().clone())
            return original_mix(old, mapped, damp)

        monkeypatch.setattr(functional_solver, "simple_mix", record_simple_mix)
    else:
        original_mix = functional_anderson.anderson_mix

        def record_anderson_mix(
            old,
            mapped,
            state,
            settings,
            step,
            transition_active=None,
            damp=None,
        ):
            transition_damps.append(torch.as_tensor(damp).detach().clone())
            return original_mix(
                old,
                mapped,
                state,
                settings,
                step,
                transition_active,
                damp,
            )

        monkeypatch.setattr(
            functional_anderson, "anderson_mix", record_anderson_mix
        )

    result = run_functional_full_eager(
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
    )
    assert result.functional is not None

    legacy_map_inputs = [old for old, _, _ in legacy_transitions]
    assert len(map_inputs) == len(legacy_map_inputs) == maxiter + 1
    for actual, expected in zip(map_inputs, legacy_map_inputs):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-13)
    assert len(transition_damps) == len(legacy_transitions) == maxiter + 1

    expected_damp = options["damp"]
    switch_after: int | None = None
    for index, ((old, mapped, legacy_damp), functional_damp) in enumerate(
        zip(legacy_transitions, transition_damps)
    ):
        assert legacy_damp == expected_damp
        torch.testing.assert_close(
            functional_damp,
            torch.tensor(expected_damp, dtype=torch.float64),
        )
        if (
            index > 0
            and switch_after is None
            and torch.linalg.vector_norm(mapped - old) < 0.1
        ):
            switch_after = index + 1
            expected_damp = options["damp_dynamic_factor"]
    assert switch_after is not None
    assert legacy_transitions[switch_after][2] == options["damp_dynamic_factor"]
    assert (
        float(transition_damps[switch_after]) == options["damp_dynamic_factor"]
    )


@pytest.mark.parametrize(
    ("maxiter", "force", "expected"),
    [
        (0, False, "accept"),
        (0, True, "accept"),
        (1, False, "warn"),
        (1, True, "raise"),
        (80, False, "accept"),
        (80, True, "accept"),
    ],
)
def test_eager_force_convergence_policy(
    maxiter: int, force: bool, expected: str
) -> None:
    """Eager outer policy preserves the one-shot and force-convergence rules."""
    options = {
        "mixer": labels.MIXER_LINEAR,
        "scp_mode": labels.SCP_MODE_CHARGE,
        "maxiter": maxiter,
        "force_convergence": force,
        "x_atol": 1.0e-8 if maxiter == 80 else 0.0,
        "x_atol_max": 1.0e-8 if maxiter == 80 else 0.0,
    }
    legacy_case = _prepare_case(GFN1_XTB, scf_options=options)
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)
    OutputHandler.clear_warnings()
    if expected == "raise":
        from dxtb._src.typing.exceptions import SCFConvergenceError

        with pytest.raises(SCFConvergenceError) as legacy_error:
            legacy_case.engine(legacy_case.initial_charges.clone())
        legacy_message = str(legacy_error.value)
        OutputHandler.clear_warnings()
        with pytest.raises(SCFConvergenceError) as functional_error:
            run_functional_full_eager(
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
            )
        assert str(functional_error.value) == legacy_message
        return

    legacy = legacy_case.engine(legacy_case.initial_charges.clone())
    legacy_warnings = list(OutputHandler.warnings)
    OutputHandler.clear_warnings()
    result = run_functional_full_eager(
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
    )
    assert result.functional is not None
    _assert_public_parity(result.functional.solve_result, legacy)
    assert OutputHandler.warnings == legacy_warnings
    if expected == "warn":
        assert len(legacy_warnings) == 1
        assert "SCF does not converge" in legacy_warnings[0][0]
        assert "Simple" in legacy_warnings[0][0]
        assert str(maxiter) in legacy_warnings[0][0]
