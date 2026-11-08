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
"""Static routing and eager-policy characterization for SCF."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, OutputHandler, labels
from dxtb._src.calculators.config.scf import ConfigFermi
from dxtb._src.calculators.result import Result
from dxtb._src.calculators.singlepoint import (
    _interaction_data,
    _iteration_tensor,
)
from dxtb._src.components.interactions import InteractionList
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.field.efieldgrad import ElectricFieldGrad
from dxtb._src.scf import iterator as scf_iterator
from dxtb._src.scf import anderson as functional_anderson
from dxtb._src.scf import eager as functional_eager
from dxtb._src.scf import converger as functional_method
from dxtb._src.scf import problem as functional_problem
from dxtb._src.scf import (
    resolve_scf_plan,
    run_eager_scf,
    run_prepared_scf,
)
from dxtb._src.scf import solver as functional_solver
from dxtb._src.scf import spectral as functional_spectral
from dxtb._src.scf.data import setup_scf_inputs
from dxtb._src.scf.eager import (
    _check_subsolve_status,
    _solve_variable,
)
from dxtb._src.scf.solver import (
    FERMI_FAILURE_ELECTRON_COUNT,
    FERMI_FAILURE_NONE,
    SCFTrajectoryStatus,
)
from dxtb._src.wavefunction.filling import FermiStatus

from ..utils import scf_strategy
from .test_map import _assert_tensor_equal, _prepare_case


def _strategy(converger: str, variable: str, **anderson: object):
    """One-Stage strategy; Anderson settings only apply to Anderson."""
    return scf_strategy(
        converger, variable, **(anderson if converger == "anderson" else {})
    )


def test_default_full_route_resolves_native_anderson_potential() -> None:
    """The configured default is Anderson in potential coordinates."""
    from dxtb.config import ConfigSCF

    case = _prepare_case(GFN2_XTB, sample_name="H2O")
    route = resolve_scf_plan(
        ConfigSCF(),
        case.interactions,
        case.system.ihelp,
        case.system.numbers,
        case.integrals,
    )
    assert route.kind == "scf"
    assert route.strategy_label == "anderson"
    assert route.warning is None


def test_single_system_full_executes_functional_route() -> None:
    """Supported single-System FULL execution stays on the functional path."""
    case = _prepare_case(GFN1_XTB, sample_name="H2")
    result = scf_iterator.solve(
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
    )
    assert bool(result.converged)


@pytest.mark.parametrize(
    "override",
    ["fwd_options", "bck_options", "eigen_options", "unexpected_option"],
)
def test_legacy_engine_overrides_are_retired(override: str) -> None:
    """Opaque legacy engine settings fail instead of selecting another SCF."""
    case = _prepare_case(GFN1_XTB)
    with pytest.raises(TypeError):
        scf_iterator.solve(
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
            **{override: {"method": "exacteig"}},
        )


def test_legacy_engine_positional_override_is_retired() -> None:
    case = _prepare_case(GFN1_XTB)
    with pytest.raises(TypeError):
        scf_iterator.solve(
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
            object(),
        )


def test_invalid_temperature_and_fermi_budget_are_explicit_errors() -> None:
    case = _prepare_case(GFN1_XTB)
    with pytest.raises(ValueError, match="Temperature must be non-negative"):
        resolve_scf_plan(
            replace(case.system.config.scf, fermi=ConfigFermi(etemp=-1.0)),
            case.interactions,
            case.system.ihelp,
            case.system.numbers,
            case.integrals,
        )
    with pytest.raises(ValueError, match="Fermi maxiter must be non-negative"):
        resolve_scf_plan(
            replace(case.system.config.scf, fermi=ConfigFermi(maxiter=-1)),
            case.interactions,
            case.system.ihelp,
            case.system.numbers,
            case.integrals,
        )


@pytest.mark.parametrize("integral", ["hcore", "overlap"])
def test_missing_integrals_are_explicit_setup_errors(integral: str) -> None:
    case = _prepare_case(GFN1_XTB)
    values = {
        "hcore": case.integrals.hcore,
        "overlap": case.integrals.overlap,
        "dipole": case.integrals.dipole,
        "quadrupole": case.integrals.quadrupole,
    }
    values[integral] = None
    with pytest.raises(ValueError, match="requires hcore and overlap"):
        resolve_scf_plan(
            case.system.config.scf,
            case.interactions,
            case.system.ihelp,
            case.system.numbers,
            SimpleNamespace(**values),
        )


def test_nonfull_route_resolution_rejects_wrong_mode() -> None:
    case = _prepare_case(GFN1_XTB)
    with pytest.raises(ValueError, match="requires unrolled differentiation"):
        resolve_scf_plan(
            replace(
                case.system.config.scf,
                differentiation="implicit",
            ),
            case.interactions,
            case.system.ihelp,
        )


def test_broyden_strategy_has_a_native_potential_full_route() -> None:
    """The Broyden preset resolves directly to its native method stage."""
    case = _prepare_case(GFN1_XTB)
    from dxtb.config import ConfigSCF

    config = ConfigSCF.create(strategy="broyden")
    route = resolve_scf_plan(config, case.interactions, case.system.ihelp)
    assert route.kind == "scf"
    assert route.strategy_label == "broyden"
    assert route.warning is None


def test_explicit_broyden_runs_after_common_preparation() -> None:
    """An explicit Broyden stage reaches native functional execution."""
    from dxtb.config import ConfigSCF

    case = _prepare_case(GFN1_XTB)
    config = ConfigSCF.create(strategy="broyden")
    result = run_eager_scf(
        case.system.numbers,
        case.positions,
        case.charge,
        case.spin,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        replace(config, maxiter=0),
        case.integrals,
        case.refocc,
    )
    assert result.plan.strategy_label == "broyden"
    assert result.scf is not None


def test_invalid_spin_precedes_broyden_warning_in_both_routes() -> None:
    """Shared reference preparation fails before mixer compatibility warnings."""
    from dxtb._src.scf.iterator import solve as legacy_solve

    case = _prepare_case(GFN1_XTB)
    invalid_spin = case.spin + 1.0
    OutputHandler.clear_warnings()
    with pytest.raises(ValueError, match="unpaired electrons") as legacy_error:
        legacy_solve(
            case.system.numbers,
            case.positions,
            case.charge,
            invalid_spin,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            case.system.config.scf,
            case.integrals,
            case.refocc,
        )
    assert OutputHandler.warnings == []


def test_invalid_spin_precedes_functional_route_error() -> None:
    """Common spin validation precedes numerical strategy execution."""
    from dxtb._src.scf.iterator import solve as legacy_solve

    case = _prepare_case(GFN1_XTB)
    config = case.system.config.scf
    invalid_spin = case.spin + 1.0
    OutputHandler.clear_warnings()
    with pytest.raises(ValueError, match="unpaired electrons") as legacy_error:
        legacy_solve(
            case.system.numbers,
            case.positions,
            case.charge,
            invalid_spin,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            case.refocc,
        )
    assert OutputHandler.warnings == []

    with pytest.raises(type(legacy_error.value), match="unpaired electrons"):
        run_eager_scf(
            case.system.numbers,
            case.positions,
            case.charge,
            invalid_spin,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            case.refocc,
        )
    assert OutputHandler.warnings == []

    OutputHandler.clear_warnings()
    with pytest.raises(type(legacy_error.value), match="unpaired electrons"):
        run_eager_scf(
            case.system.numbers,
            case.positions,
            case.charge,
            invalid_spin,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            case.system.config.scf,
            case.integrals,
            case.refocc,
        )
    assert OutputHandler.warnings == []


def test_negative_scf_maxiter_is_functional_configuration_error() -> None:
    """E0's negative outer iteration policy is rejected before execution."""
    case = _prepare_case(GFN1_XTB)
    config = replace(case.system.config.scf, maxiter=-1)
    with pytest.raises(ValueError, match="maxiter >= 0"):
        resolve_scf_plan(config, case.interactions, case.system.ihelp)


def test_negative_scf_maxiter_invalid_spin_is_rejected_explicitly() -> None:
    """Negative iteration budgets fail before legacy spin validation."""
    from dxtb._src.scf.iterator import solve as legacy_solve

    case = _prepare_case(GFN1_XTB)
    config = replace(case.system.config.scf, maxiter=-1)
    invalid_spin = case.spin + 1.0
    with pytest.raises(ValueError, match="maxiter >= 0"):
        legacy_solve(
            case.system.numbers,
            case.positions,
            case.charge,
            invalid_spin,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            case.refocc,
        )
    with pytest.raises(ValueError, match="unpaired electrons"):
        run_eager_scf(
            case.system.numbers,
            case.positions,
            case.charge,
            invalid_spin,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            case.refocc,
        )


def test_negative_scf_maxiter_valid_preparation_raises_e0_error() -> None:
    """Functional-only negative maxiter is rejected after common setup."""
    case = _prepare_case(GFN1_XTB)
    config = replace(case.system.config.scf, maxiter=-1)
    with pytest.raises(ValueError, match="maxiter >= 0"):
        run_eager_scf(
            case.system.numbers,
            case.positions,
            case.charge,
            case.spin,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            case.refocc,
        )


def test_float32_damping_warning_uses_configured_decimal() -> None:
    """Functional float32 warnings preserve configured decimals."""
    options = {
        "strategy": _strategy("linear", "charge"),
        "maxiter": 12,
        "damp": 0.2,
        "damp_dynamic": True,
        "damp_dynamic_factor": 0.99,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    functional_case = _prepare_case(
        GFN1_XTB, dtype=torch.float32, scf_options=options
    )
    OutputHandler.clear_warnings()
    functional = run_eager_scf(
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
    assert functional.scf is not None
    diagnostics = functional.scf.solve_result.diagnostics
    assert diagnostics is not None
    assert diagnostics.dynamic_damping_switched
    assert OutputHandler.warnings
    assert "damping 0.99)." in OutputHandler.warnings[-1][0]


def test_gfn0_remains_a_nonselfconsistent_special_case() -> None:
    """The retained GFN0 path is one eigensolve, outside iterative SCF."""
    case = _prepare_case(GFN1_XTB)
    config = replace(case.system.config.scf, method=labels.GFN0_XTB)
    assert not config.requires_iterations
    from dxtb._src.scf.nonselfconsistent import solve_nonselfconsistent

    assert solve_nonselfconsistent is not None


def test_packed_full_batch_is_an_explicit_compatibility_boundary() -> None:
    """Batched numbers are rejected by single-System SCF route resolution."""
    case = _prepare_case(GFN1_XTB)
    with pytest.raises(NotImplementedError, match="single-System SCF only"):
        resolve_scf_plan(
            case.system.config.scf,
            case.interactions,
            case.system.ihelp,
            case.system.numbers.unsqueeze(0),
        )


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

    route = resolve_scf_plan(
        case.system.config.scf, interactions, case.system.ihelp
    )

    assert route.kind == "scf"


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
            "strategy": _strategy("linear", "charge"),
            "maxiter": 0,
        },
    )
    result = run_eager_scf(
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
    ).scf
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
        first_failure=torch.tensor(
            (
                FERMI_FAILURE_ELECTRON_COUNT
                if category == "electron_count_ok"
                else 0
            ),
            dtype=torch.int64,
        ),
    )
    synthetic = replace(
        solve_result,
        final_evaluation=final_evaluation,
        trajectory_status=trajectory,
    )
    with pytest.raises(exception, match=message):
        _check_subsolve_status(synthetic)


@pytest.mark.parametrize(
    ("variable", "first_code", "final_code", "message"),
    [
        (
            "charge",
            FERMI_FAILURE_ELECTRON_COUNT,
            0,
            "Number of electrons changed during Fermi smearing",
        ),
        (
            "potential",
            3,
            4,
            "Fermi energy failed to converge",
        ),
        (
            "fock",
            0,
            3,
            "Fermi energy failed to converge",
        ),
    ],
    ids=[
        "charge-earlier-error",
        "potential-fermi-before-electron",
        "fock-final-only",
    ],
)
def test_eager_failure_order_is_chronological(
    variable: str, first_code: int, final_code: int, message: str
) -> None:
    """Trajectory failures precede later final-spectrum failures."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": _strategy("linear", variable),
            "fermi_etemp": 300.0,
            "maxiter": 0,
        },
    )
    result = run_eager_scf(
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
    ).scf
    assert result is not None
    solve_result = result.solve_result
    trajectory = solve_result.trajectory_status
    if first_code == 0:
        trajectory = None
    else:
        assert trajectory is not None
        trajectory = replace(
            trajectory,
            first_failure=torch.tensor(first_code, dtype=torch.int64),
        )
    final_status = solve_result.final_evaluation.fermi_status
    assert final_status is not None
    final_status = replace(
        final_status,
        converged=torch.tensor(final_code != 3),
        electron_count_ok=torch.tensor(final_code != 4),
        invalid_temperature=torch.tensor(final_code == 1),
        too_many_electrons=torch.tensor(final_code == 2),
    )
    synthetic = replace(
        solve_result,
        trajectory_status=trajectory,
        final_evaluation=replace(
            solve_result.final_evaluation, fermi_status=final_status
        ),
    )
    with pytest.raises(RuntimeError, match=message):
        _check_subsolve_status(synthetic)


def test_custom_interaction_extension_is_explicitly_unsupported() -> None:
    """Unsupported interaction subclasses never select a legacy engine."""
    case = _prepare_case(GFN1_XTB)

    class ExtendedField(ElectricField):
        """A custom field extension outside the interaction API."""

    field = ExtendedField(torch.zeros(3, dtype=torch.float64))
    interactions = InteractionList(
        *case.interactions.components,
        field,
        device=case.positions.device,
        dtype=case.positions.dtype,
    )
    with pytest.raises(NotImplementedError, match="interaction extension"):
        resolve_scf_plan(
            case.system.config.scf, interactions, case.system.ihelp
        )


@pytest.mark.parametrize("method", [GFN1_XTB, GFN2_XTB], ids=["gfn1", "gfn2"])
@pytest.mark.parametrize(
    "variable",
    ["charge", "potential"],
    ids=["charge", "potential"],
)
def test_default_anderson_adapter_matches_legacy(
    method: int, variable: str
) -> None:
    """The default Anderson route reaches the same physical solution across variables."""
    options = {
        "strategy": _strategy("anderson", variable),
        "fermi_etemp": 300.0,
        "maxiter": 100,
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-8,
    }
    OutputHandler.clear_warnings()
    functional_case = _prepare_case(method, scf_options=options)
    ref_case = _prepare_case(
        method,
        scf_options={
            **options,
            "strategy": _strategy("anderson", "potential"),
        },
    )

    OutputHandler.clear_warnings()
    ref = run_eager_scf(
        ref_case.system.numbers,
        ref_case.positions,
        ref_case.charge,
        ref_case.spin,
        ref_case.interactions,
        ref_case.interaction_data,
        ref_case.system.ihelp,
        ref_case.system.config.scf,
        ref_case.integrals,
        ref_case.refocc,
    )
    assert ref.scf is not None

    OutputHandler.clear_warnings()
    actual = run_eager_scf(
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
    assert actual.scf is not None
    assert actual.non_scf is None
    assert actual.plan.strategy_label == "anderson"
    assert OutputHandler.warnings == []
    for name in (
        "density",
        "hamiltonian",
        "emo",
        "occupation",
        "energy",
        "fenergy",
    ):
        torch.testing.assert_close(
            getattr(actual.scf, name),
            getattr(ref.scf, name),
            rtol=1.0e-7,
            atol=1.0e-8,
        )
    torch.testing.assert_close(
        actual.scf.charges.mono,
        ref.scf.charges.mono,
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        actual.scf.potential.mono,
        ref.scf.potential.mono,
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    assert actual.scf.converged.item() is True
    assert isinstance(actual.scf.iterations, torch.Tensor)
    assert isinstance(actual.scf.converged, torch.Tensor)
    assert isinstance(actual.scf.residual, torch.Tensor)


@pytest.mark.parametrize(
    "variable",
    ["charge", "potential", "fock"],
    ids=["charge", "potential", "fock"],
)
@pytest.mark.parametrize(
    "converger",
    ["linear", "anderson"],
    ids=["simple", "anderson"],
)
@pytest.mark.parametrize("temperature", [0.0, 300.0], ids=["zero-k", "300-k"])
def test_eager_variable_mixer_budget_parity(
    variable: str, converger: str, temperature: float
) -> None:
    """All variable/mixer pairs produce finite short-budget eager results with warning."""
    options = {
        "strategy": _strategy(
            converger, variable, generations=1, soft_start=False
        ),
        "fermi_etemp": temperature,
        "maxiter": 2,
        "mix_guess": True,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)

    OutputHandler.clear_warnings()
    actual = run_eager_scf(
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
    assert actual.scf is not None
    assert not bool(actual.scf.converged)
    assert int(actual.scf.iterations) == 3
    assert torch.isfinite(actual.scf.energy).all()
    assert len(OutputHandler.warnings) == 1
    assert OutputHandler.warnings[0][0].startswith("SCF did not certify after")


@pytest.mark.parametrize(
    "variable",
    ["charge", "potential", "fock"],
    ids=["charge", "potential", "fock"],
)
@pytest.mark.parametrize(
    "converger",
    ["linear", "anderson"],
    ids=["simple", "anderson"],
)
def test_eager_converged_variable_mixer_parity(
    variable: str, converger: str
) -> None:
    """Converged eager results agree across all variables and mixers."""
    options = {
        "strategy": _strategy(
            converger, variable, generations=2, soft_start=False
        ),
        "fermi_etemp": 0.0,
        "maxiter": 100,
        "mix_guess": True,
        "x_atol": 1.0e-8,
        "x_atol_max": 1.0e-8,
    }
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)
    ref_case = _prepare_case(
        GFN1_XTB,
        scf_options={
            **options,
            "strategy": _strategy(
                "anderson", "potential", generations=2, soft_start=False
            ),
        },
    )
    OutputHandler.clear_warnings()
    ref = run_eager_scf(
        ref_case.system.numbers,
        ref_case.positions,
        ref_case.charge,
        ref_case.spin,
        ref_case.interactions,
        ref_case.interaction_data,
        ref_case.system.ihelp,
        ref_case.system.config.scf,
        ref_case.integrals,
        ref_case.refocc,
    )
    assert ref.scf is not None
    OutputHandler.clear_warnings()
    actual = run_eager_scf(
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
    assert actual.scf is not None
    assert bool(actual.scf.converged)
    assert OutputHandler.warnings == []
    torch.testing.assert_close(
        actual.scf.energy,
        ref.scf.energy,
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        actual.scf.charges.mono,
        ref.scf.charges.mono,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_core_result_boundary_keeps_solver_status_tensors() -> None:
    """SCF core conversion preserves all status tensors directly."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": _strategy("linear", "charge"),
            "maxiter": 0,
        },
    )
    result = run_eager_scf(
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
    ).scf
    assert result is not None
    assert isinstance(result.iterations, torch.Tensor)
    assert isinstance(result.converged, torch.Tensor)
    assert isinstance(result.residual, torch.Tensor)
    assert result.iterations.ndim == 0
    assert result.iterations is result.solve_result.iterations
    assert result.converged is result.solve_result.converged
    assert result.residual is result.solve_result.residual


def test_iteration_tensor_boundary_preserves_existing_tensor() -> None:
    """The future Result boundary does not copy functional iteration status."""
    like = torch.ones((), dtype=torch.float64)
    functional_iterations = torch.tensor(3, dtype=torch.int64)
    assert (
        _iteration_tensor(functional_iterations, like) is functional_iterations
    )
    legacy_iterations = _iteration_tensor(3, like)
    assert legacy_iterations.dtype == torch.int64
    assert legacy_iterations.device == like.device
    assert legacy_iterations.item() == 3


def test_prepared_functional_runner_does_not_repeat_common_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared functional execution consumes prepared tensors directly."""
    case = _prepare_case(GFN1_XTB)
    route = resolve_scf_plan(
        case.system.config.scf, case.interactions, case.system.ihelp
    )
    assert route.kind == "scf"

    def unexpected(*args, **kwargs):
        raise AssertionError("common SCF preparation was repeated")

    monkeypatch.setattr(functional_eager, "get_refocc", unexpected)
    monkeypatch.setattr(functional_eager, "get_guess", unexpected)
    result = run_prepared_scf(
        case.system.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        case.system.config.scf,
        case.integrals,
        route,
    )
    assert isinstance(result.iterations, torch.Tensor)
    assert isinstance(result.converged, torch.Tensor)
    assert isinstance(result.residual, torch.Tensor)


@pytest.mark.parametrize(
    "converger",
    ["linear", "anderson"],
    ids=["linear", "anderson"],
)
def test_prepared_runner_reports_configured_method(converger: str) -> None:
    """The retained packed FULL runner reports its configured method stage."""
    options = {
        "strategy": _strategy(converger, "potential"),
        "fermi_etemp": 300.0,
        "maxiter": 0,
    }
    case = _prepare_case(GFN1_XTB, scf_options=options)
    config = case.system.config.scf
    route = resolve_scf_plan(config, case.interactions, case.system.ihelp)
    assert route.kind == "scf"
    assert route.strategy_label == converger

    direct = run_prepared_scf(
        case.system.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
        route,
    )
    eager = run_eager_scf(
        case.system.numbers,
        case.positions,
        case.charge,
        case.spin,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
        case.refocc,
    )
    assert eager.scf is not None
    _assert_tensor_equal(direct.hamiltonian, eager.scf.hamiltonian)


def test_prepared_runner_rejects_variable_mismatch() -> None:
    """A route cannot be reused with another convergence variable."""
    case = _prepare_case(GFN1_XTB)
    route = resolve_scf_plan(
        case.system.config.scf, case.interactions, case.system.ihelp
    )
    from dxtb.scf import SCFVariable, SCFStrategy

    stage = replace(
        case.system.config.scf.strategy.stages[0],
        variable=SCFVariable.FOCK,
    )
    mismatched = replace(
        case.system.config.scf,
        strategy=SCFStrategy((stage,)),
    )
    with pytest.raises(ValueError, match="SCF variable of the prepared SCF plan does not match"):
        run_prepared_scf(
            case.system.numbers,
            case.initial_charges,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            mismatched,
            case.integrals,
            route,
        )


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
def test_eager_and_fixed_trip_controllers_match(
    variable: str, converger: str
) -> None:
    """Shared transitions keep eager and transform-fixed physical results equal."""
    options = {
        "strategy": _strategy(
            converger, variable, generations=2, soft_start=False
        ),
        "fermi_etemp": 300.0,
        "maxiter": 24,
        "mix_guess": True,
        "damp_dynamic": True,
        "damp_dynamic_factor": 0.82,
        "x_atol": 1.0e-6,
        "x_atol_max": 1.0e-6,
    }
    case = _prepare_case(GFN1_XTB, scf_options=options)
    route = resolve_scf_plan(
        case.system.config.scf,
        case.interactions,
        case.system.ihelp,
        case.system.numbers,
        case.integrals,
    )
    assert route.kind == "scf"
    config = case.system.config.scf
    inputs = setup_scf_inputs(
        case.system.numbers,
        case.integrals,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
    )
    eager = _solve_variable(
        case.initial_charges, inputs, config, eager=True
    )
    fixed = _solve_variable(
        case.initial_charges, inputs, config, eager=False
    )
    assert eager.converged.item() and fixed.converged.item()
    for field in (
        "energy",
        "fenergy",
        "iterations",
        "converged",
        "residual",
    ):
        torch.testing.assert_close(
            getattr(eager, field), getattr(fixed, field), rtol=0.0, atol=0.0
        )
    for field in ("mapped_x",):
        torch.testing.assert_close(
            getattr(eager.map_result, field),
            getattr(fixed.map_result, field),
            rtol=0.0,
            atol=0.0,
        )
    for field in (
        "hamiltonian",
        "density",
        "emo",
        "coefficients",
        "occupation",
        "orbital_energy",
    ):
        torch.testing.assert_close(
            getattr(eager.final_evaluation, field),
            getattr(fixed.final_evaluation, field),
            rtol=0.0,
            atol=0.0,
        )
    for left, right in (
        (eager.final_charges.mono, fixed.final_charges.mono),
        (eager.final_potential.mono, fixed.final_potential.mono),
        (eager.diagnostics.final_damp, fixed.diagnostics.final_damp),
        (
            eager.diagnostics.dynamic_damping_switched,
            fixed.diagnostics.dynamic_damping_switched,
        ),
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert eager.trajectory_status is not None
    assert fixed.trajectory_status is not None
    for field in (
        "fermi_converged",
        "electron_count_ok",
        "invalid_temperature",
        "too_many_electrons",
        "first_failure",
    ):
        torch.testing.assert_close(
            getattr(eager.trajectory_status, field),
            getattr(fixed.trajectory_status, field),
            rtol=0.0,
            atol=0.0,
        )
    assert eager.final_evaluation.fermi_status is not None
    assert fixed.final_evaluation.fermi_status is not None
    torch.testing.assert_close(
        eager.final_evaluation.fermi_status.converged,
        fixed.final_evaluation.fermi_status.converged,
    )


def test_result_snapshot_retains_functional_tensor_identity() -> None:
    """Result.snapshot accepts immutable outputs and retains tensor leaves."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "strategy": _strategy("linear", "charge"),
            "maxiter": 0,
        },
    )
    functional = run_eager_scf(
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
    ).scf
    assert functional is not None
    snapshot = Result.snapshot(
        energy=functional.energy,
        scf=None,
        classical=(),
        fenergy=functional.fenergy,
        iterations=functional.iterations,
        charges=functional.charges,
        potential=functional.potential,
        converged=functional.converged,
        residual=functional.residual,
    )
    assert snapshot.charges is not None
    assert snapshot.potential is not None
    assert snapshot.charges.mono is functional.charges.mono
    assert snapshot.charges.dipole is functional.charges.dipole
    assert snapshot.charges.quadrupole is functional.charges.quad
    assert snapshot.potential.mono is functional.potential.mono
    assert snapshot.potential.dipole is functional.potential.dipole
    assert snapshot.potential.quadrupole is functional.potential.quad
    assert snapshot.iterations is functional.iterations
    assert snapshot.converged is functional.converged
    assert snapshot.residual is functional.residual


@pytest.mark.parametrize(
    "variable",
    ["charge", "potential", "fock"],
    ids=["charge", "potential", "fock"],
)
@pytest.mark.parametrize(
    "converger",
    ["linear", "anderson"],
    ids=["simple", "anderson"],
)
def test_dynamic_damping_adapter_matches_legacy(
    variable: str, converger: str
) -> None:
    """Dynamic damping switches to damp_dynamic_factor after the raw norm crosses 0.1."""
    options = {
        "strategy": _strategy(
            converger,
            variable,
            damp_init=0.05,
            generations=2,
            soft_start=True,
        ),
        "maxiter": 12,
        "mix_guess": True,
        "damp": 0.2,
        "damp_dynamic": True,
        "damp_dynamic_factor": 0.75,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)
    OutputHandler.clear_warnings()
    actual = run_eager_scf(
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
    assert actual.scf is not None
    assert actual.scf.solve_result.diagnostics is not None
    assert actual.scf.solve_result.diagnostics.dynamic_damping_switched
    torch.testing.assert_close(
        actual.scf.solve_result.diagnostics.final_damp,
        torch.tensor(0.75, dtype=torch.float64),
    )
    if not bool(actual.scf.converged):
        assert len(OutputHandler.warnings) == 1
        assert OutputHandler.warnings[0][0].startswith(
            "SCF did not certify after"
        )
    else:
        assert len(OutputHandler.warnings) == 0


@pytest.mark.parametrize(
    ("variable", "converger"),
    [
        ("charge", "linear"),
        ("potential", "anderson"),
        ("fock", "anderson"),
    ],
    ids=[
        "charge-simple",
        "potential-anderson",
        "fock-anderson",
    ],
)
def test_dynamic_damping_iterate_and_switch_parity(
    variable: str,
    converger: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every raw iterate switches ordinary damping when the raw residual crosses 0.1."""
    maxiter = 12 if variable == "charge" and converger == "linear" else 8
    options = {
        "strategy": _strategy(
            converger,
            variable,
            damp_init=0.05,
            generations=2,
            soft_start=True,
        ),
        "maxiter": maxiter,
        "mix_guess": True,
        "damp": 0.2,
        "damp_dynamic": True,
        "damp_dynamic_factor": 0.75,
        "x_atol": 0.0,
        "x_atol_max": 0.0,
    }
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)

    transitions: list[tuple[torch.Tensor, torch.Tensor]] = []
    transition_damps: list[torch.Tensor] = []
    # Both method strategies now use the same SCFProblem map.
    module = functional_problem
    map_name = {
        "charge": "scf_map",
        "potential": "potential_map",
        "fock": "fock_map",
    }[variable]
    original_map = getattr(module, map_name)

    def record_functional_map(x: torch.Tensor, *args, **kwargs):
        out = original_map(x, *args, **kwargs)
        transitions.append((x.clone(), out.mapped_x.clone()))
        return out

    monkeypatch.setattr(module, map_name, record_functional_map)
    if converger == "linear":
        original_mix = functional_method.simple_mix

        def record_simple_mix(old, mapped, damp):
            transition_damps.append(torch.as_tensor(damp).detach().clone())
            return original_mix(old, mapped, damp)

        monkeypatch.setattr(functional_method, "simple_mix", record_simple_mix)
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
                transition_active=transition_active,
                damp=damp,
            )

        monkeypatch.setattr(
            functional_anderson, "anderson_mix", record_anderson_mix
        )

    result = run_eager_scf(
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
    assert result.scf is not None
    assert len(transitions) == maxiter + 1
    assert len(transition_damps) == maxiter + 1

    expected_damp = options["damp"]
    switch_after: int | None = None
    for index, ((old, mapped), functional_damp) in enumerate(
        zip(transitions, transition_damps)
    ):
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
        "strategy": _strategy("linear", "charge"),
        "maxiter": maxiter,
        "force_convergence": force,
        "x_atol": 1.0e-8 if maxiter == 80 else 0.0,
        "x_atol_max": 1.0e-8 if maxiter == 80 else 0.0,
    }
    functional_case = _prepare_case(GFN1_XTB, scf_options=options)
    OutputHandler.clear_warnings()
    if expected == "raise":
        from dxtb._src.typing.exceptions import SCFConvergenceError

        with pytest.raises(SCFConvergenceError) as functional_error:
            run_eager_scf(
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
        assert str(functional_error.value).startswith(
            "SCF did not certify after"
        )
        assert "physical SCF root was not certified" in str(
            functional_error.value
        )
        return

    result = run_eager_scf(
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
    assert result.scf is not None
    if expected == "warn":
        assert len(OutputHandler.warnings) == 1
        assert OutputHandler.warnings[0][0].startswith(
            "SCF did not certify after"
        )
        assert str(maxiter) in OutputHandler.warnings[0][0]
    else:
        assert OutputHandler.warnings == []


def test_unsupported_spectral_lane_cannot_report_public_scf_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported active Fermi map stays invalid through eager policies."""
    original = functional_spectral._matrix_fermi

    def unsupported(matrix, shift, kt, active):
        offset = torch.zeros_like(matrix)
        offset[..., -1, -1] = 1.0e10
        return original(matrix + offset, shift, kt, active)

    monkeypatch.setattr(functional_spectral, "_matrix_fermi", unsupported)

    one_shot = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        scf_options={
            "strategy": _strategy("linear", "charge"),
            "fermi_etemp": 300.0,
            "maxiter": 0,
            "force_convergence": True,
        },
    )
    one_shot_result = run_eager_scf(
        one_shot.system.numbers,
        one_shot.positions,
        one_shot.charge,
        one_shot.spin,
        one_shot.interactions,
        one_shot.interaction_data,
        one_shot.system.ihelp,
        one_shot.system.config.scf,
        one_shot.integrals,
        one_shot.refocc,
    )
    assert one_shot_result.scf is not None
    assert not one_shot_result.scf.converged
    assert not torch.isfinite(one_shot_result.scf.density).all()
    assert not torch.isfinite(one_shot_result.scf.energy).all()

    forced = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        scf_options={
            "strategy": _strategy("linear", "charge"),
            "fermi_etemp": 300.0,
            "maxiter": 2,
            "force_convergence": True,
        },
    )
    # The next map can encounter a NaN state before eager policy evaluation;
    # this still fails visibly and cannot produce an accepted result.
    with pytest.raises(RuntimeError):
        run_eager_scf(
            forced.system.numbers,
            forced.positions,
            forced.charge,
            forced.spin,
            forced.interactions,
            forced.interaction_data,
            forced.system.ihelp,
            forced.system.config.scf,
            forced.integrals,
            forced.refocc,
        )

    implicit = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        scf_options={
            "strategy": _strategy("linear", "charge"),
            "fermi_etemp": 300.0,
            "maxiter": 2,
        },
    )

    def no_root_attachment(*args, **kwargs):
        raise AssertionError("invalid primal reached implicit root attachment")

    monkeypatch.setattr(
        functional_eager._SCFMapTensorTree, "prepare", no_root_attachment
    )
    with pytest.raises(RuntimeError):
        functional_eager.run_prepared_implicit_scf(
            implicit.system.numbers,
            implicit.initial_charges,
            implicit.n0,
            implicit.occupation,
            implicit.nel,
            implicit.interactions,
            implicit.interaction_data,
            implicit.system.ihelp,
            implicit.system.config.scf,
            implicit.integrals,
        )
