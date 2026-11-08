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
"""Reference checks for derivatives of the physical functional SCF root."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from tad_mctc import storch
from torch.func import jacrev, jvp

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.interactions import InteractionList
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.functional import (
    BroydenMethod,
    BroydenMethodSettings,
    SCFCharges,
    SCFMapResult,
    SCFProblem,
    SCFRepresentation,
    SCFStage,
    SCFStrategy,
    fock_initial_guess,
    fock_map,
)
from dxtb._src.scf.functional import map as functional_map
from dxtb._src.scf.functional import (
    pack_charges,
    potential_initial_guess,
    potential_map,
    scf_map,
    setup_scf_inputs,
    setup_solver_settings,
)
from dxtb._src.scf.functional.solver import (
    _project_potential_padding,
    _solve_map,
    finalize_scf,
    finalize_scf_fock,
    finalize_scf_potential,
)
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import SCFConvergenceError
from dxtb.scf import (
    AndersonMethod,
    AndersonMethodSettings,
    LinearMethod,
)

from ..test_baseline.compute import _calculator
from ..test_baseline.molecules import SYSTEMS
from ..test_baseline.tolerances import TBLITE
from ..utils import get_param_module
from ._anderson_helpers import from_config as setup_anderson_solver_settings
from ._anderson_helpers import solve_common as solve_common
from .samples import samples

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}
_FIELD_AXIS = torch.tensor([0.0, 1.0, 0.0], **_DD)


@dataclass(frozen=True)
class _RootRecord:
    root: Tensor
    residual: Tensor
    reported_residual: Tensor
    map_count: int
    solver_map_evaluations: int
    diagnostic_map_evaluations: int
    last_map_input: Tensor
    last_map_result: SCFMapResult
    converged: bool
    electronic_energy: Tensor
    finalized_energy: Tensor
    outcome: object


def _force_lorentzian(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one explicit spectral derivative branch in all transform probes."""

    def diagonalize(hamiltonian: Tensor, inputs):
        return storch.eighb(
            a=hamiltonian,
            b=None,
            l_inv=inputs.inverse_cholesky,
            is_posdef=True,
            factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
            broadening_method="lorn",
        )

    monkeypatch.setattr(
        functional_map, "_transitional_diagonalize", diagonalize
    )


_CONVERGERS = {
    "linear": LinearMethod,
    "anderson": AndersonMethod,
    "broyden": BroydenMethod,
}


def _method(converger: type | str) -> object:
    """Build a default-settings SCF method from its class or preset name."""
    if isinstance(converger, str):
        converger = _CONVERGERS[converger]
    return {
        LinearMethod: LinearMethod(),
        AndersonMethod: AndersonMethod(AndersonMethodSettings()),
        BroydenMethod: BroydenMethod(BroydenMethodSettings()),
    }[converger]


def _case(
    sample_name: str,
    *,
    method: int = GFN1_XTB,
    representation: SCFRepresentation | str = SCFRepresentation.POTENTIAL,
    converger: type | str = AndersonMethod,
    tolerance: float = 1.0e-12,
    maxiter: int = 300,
    temperature: float = 0.0,
    field: Tensor | None = None,
    positions_override: Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    int_driver: int | None = None,
    parameter_module: ParamModule | None = None,
):
    dd = {"device": torch.device("cpu"), "dtype": dtype}
    source = SYSTEMS.get(sample_name)
    if source is None:
        source = samples[sample_name]
        numbers = source["numbers"].to(device="cpu")
        positions = source["positions"].to(device="cpu", dtype=dtype)
        charge = torch.as_tensor(source.get("charge", 0.0), **dd).reshape(-1)
        spin = torch.as_tensor(source.get("spin", 0.0), **dd).reshape(-1)
    else:
        numbers = source.numbers.to(device="cpu")
        positions = source.positions.to(device="cpu", dtype=dtype)
        charge = source.charge.to(device="cpu", dtype=dtype).reshape(-1)
        spin = (
            torch.zeros_like(charge)
            if source.spin is None
            else source.spin.to(device="cpu", dtype=dtype).reshape(-1)
        )
    if positions_override is not None:
        positions = positions_override
    strategy = SCFStrategy(
        (
            SCFStage(
                _method(converger),
                representation=SCFRepresentation(representation),
            ),
        )
    )
    options = {
        "strategy": strategy,
        "differentiation": "unrolled",
        "maxiter": maxiter,
        "x_atol": tolerance,
        "x_atol_max": tolerance,
        "fermi_etemp": temperature,
        "verbosity": 0,
        "int_level": (
            labels.INTLEVEL_DIPOLE
            if field is not None
            else labels.INTLEVEL_QUADRUPOLE
        ),
    }
    if int_driver is not None:
        options["int_driver"] = int_driver
    if field is None:
        field = torch.zeros(3, **dd)
    # The public parameter constants are lazy loaders. Their ``__eq__``
    # compares loaded values, so two not-yet-loaded loaders compare equal.
    # Use identity to preserve the requested model before either is loaded.
    parameter = "gfn1" if method is GFN1_XTB else "gfn2"
    parameters = (
        get_param_module(parameter, device="cpu", dtype=dtype)
        if parameter_module is None
        else parameter_module
    )
    calculator = Calculator(
        numbers,
        parameters,
        device="cpu",
        dtype=dtype,
        opts=options,
    )
    system = calculator.model.setup(numbers, dd=calculator.dd)
    integrals, refocc, _ = build_integral_matrices(
        system.integral_setup, system.h0_setup, positions, charge
    )
    n0, occupation, nel = get_refocc(refocc, charge, spin, system.ihelp)
    interactions = InteractionList(
        *system.interactions.components,
        ElectricField(field),
        device=positions.device,
        dtype=positions.dtype,
    )
    interaction_data = _interaction_data(system, positions, interactions)
    inputs = setup_scf_inputs(
        numbers,
        integrals,
        n0,
        occupation,
        nel,
        interactions,
        interaction_data,
        system.ihelp,
        system.config.scf,
    )
    initial_charges = get_guess(
        numbers,
        positions,
        charge,
        system.ihelp,
        system.config.scf.guess,
    )
    return SimpleNamespace(
        system=system,
        calculator=calculator,
        numbers=numbers,
        positions=positions,
        charge=charge,
        spin=spin,
        integrals=integrals,
        refocc=refocc,
        n0=n0,
        occupation=occupation,
        nel=nel,
        interactions=interactions,
        interaction_data=interaction_data,
        inputs=inputs,
        initial_charges=initial_charges,
    )


def _inputs_at_field(case, field_parameter: Tensor):
    field = field_parameter * _FIELD_AXIS
    interactions = InteractionList(
        *case.system.interactions.components,
        ElectricField(field),
        device=case.positions.device,
        dtype=case.positions.dtype,
    )
    interaction_data = _interaction_data(
        case.system, case.positions, interactions
    )
    return setup_scf_inputs(
        case.numbers,
        case.integrals,
        case.n0,
        case.occupation,
        case.nel,
        interactions,
        interaction_data,
        case.system.ihelp,
        case.system.config.scf,
    )


def _initial_state(case, inputs) -> tuple[Tensor, object | None]:
    layout = inputs.charge_layout
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
    representation = case.system.config.scf.initial_representation
    if representation is SCFRepresentation.CHARGE:
        return pack_charges(initial, layout), None
    if representation is SCFRepresentation.POTENTIAL:
        return potential_initial_guess(initial, inputs)
    if representation is SCFRepresentation.FOCK:
        return fock_initial_guess(initial, inputs), None
    raise ValueError(f"Unknown SCF representation: {representation}")


def _map_fn(case, inputs, layout):
    representation = case.system.config.scf.initial_representation
    if representation is SCFRepresentation.CHARGE:
        return lambda x: scf_map(x, inputs)
    if representation is SCFRepresentation.POTENTIAL:
        assert layout is not None
        return lambda x: potential_map(x, inputs, layout)
    if representation is SCFRepresentation.FOCK:
        return lambda x: fock_map(x, inputs)
    raise ValueError(f"Unknown SCF representation: {representation}")


def _finalize_map_result(
    case,
    map_result,
    inputs,
    iterations,
    converged,
    residual,
    trajectory_status=None,
    damping_state=None,
    *,
    root_x: Tensor,
):
    """Apply the configured representation's ordinary finalization."""
    representation = case.system.config.scf.initial_representation
    if representation is SCFRepresentation.CHARGE:
        return finalize_scf(
            map_result,
            inputs,
            iterations,
            converged,
            residual,
            trajectory_status,
            damping_state,
            root_x=root_x,
        )
    if representation is SCFRepresentation.POTENTIAL:
        _, layout = _initial_state(case, inputs)
        assert layout is not None
        return finalize_scf_potential(
            map_result,
            inputs,
            layout,
            iterations,
            converged,
            residual,
            trajectory_status,
            damping_state,
            root_x=root_x,
        )
    if representation is SCFRepresentation.FOCK:
        return finalize_scf_fock(
            map_result,
            inputs,
            iterations,
            converged,
            residual,
            trajectory_status,
            damping_state,
            root_x=root_x,
        )
    raise ValueError(f"Unknown SCF representation: {representation}")


def _run_root(
    case,
    inputs,
    *,
    tolerance: float,
    maxiter: int,
    eager: bool,
    converger: type | str | None = None,
) -> _RootRecord:
    if converger is None:
        converger = type(case.system.config.scf.strategy.stages[0].method)
    config = replace(
        case.system.config.scf,
        strategy=SCFStrategy(
            (
                replace(
                    case.system.config.scf.strategy.stages[0],
                    method=_method(converger),
                ),
            )
        ),
        maxiter=maxiter,
        x_atol=tolerance,
        x_atol_max=tolerance,
    )
    initial, layout = _initial_state(case, inputs)
    map_fn = _map_fn(case, inputs, layout)
    map_evaluations: list[tuple[Tensor, SCFMapResult]] = []

    def observed_map(value: Tensor) -> SCFMapResult:
        result = map_fn(value)
        map_evaluations.append((value, result))
        return result

    projection = (
        (lambda value: _project_potential_padding(value, layout))
        if config.initial_representation is SCFRepresentation.POTENTIAL
        else None
    )
    representation = case.system.config.scf.initial_representation
    problem = SCFProblem(inputs, representation, layout)
    method_type = type(config.strategy.stages[0].method)
    if method_type is AndersonMethod:
        settings = setup_anderson_solver_settings(config, initial.dtype)
        outcome = solve_common(
            initial,
            settings,
            observed_map,
            state_projection=projection,
            eager=eager,
            problem=problem,
        )
    elif method_type is LinearMethod:
        settings = setup_solver_settings(config, initial.dtype)
        outcome = _solve_map(
            initial,
            settings,
            observed_map,
            state_projection=projection,
            eager=eager,
            problem=problem,
        )
    elif method_type is BroydenMethod:
        settings = setup_solver_settings(config, initial.dtype)
        strategy = SCFStrategy(
            (SCFStage(BroydenMethod(BroydenMethodSettings())),)
        )
        outcome = _solve_map(
            initial,
            settings,
            observed_map,
            state_projection=projection,
            eager=eager,
            problem=problem,
            strategy=strategy,
        )
    else:
        raise ValueError(f"Unsupported reference method: {method_type}")

    finalized = _finalize_map_result(
        case,
        outcome.map_result,
        inputs,
        outcome.iterations,
        outcome.converged,
        outcome.residual,
        outcome.trajectory_status,
        outcome.damping_state,
        root_x=outcome.root_x,
    )
    root = outcome.root_x
    raw_residual = outcome.map_result.mapped_x - root
    assert map_evaluations
    last_map_input, last_map_result = map_evaluations[-1]
    return _RootRecord(
        root=root,
        residual=raw_residual,
        reported_residual=outcome.residual,
        map_count=int(outcome.iterations.detach()),
        solver_map_evaluations=len(map_evaluations),
        diagnostic_map_evaluations=0,
        last_map_input=last_map_input,
        last_map_result=last_map_result,
        converged=bool(outcome.converged.detach()),
        electronic_energy=finalized.energy.sum(),
        finalized_energy=(finalized.energy + finalized.fenergy).sum(),
        outcome=outcome,
    )


def test_raw_fixed_point_residual_uses_only_each_representation_map() -> None:
    """Charge, potential and Fock residuals are their raw physical maps."""
    for representation in (
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ):
        case = _case("H2", representation=representation)
        initial, layout = _initial_state(case, case.inputs)
        mapped = _map_fn(case, case.inputs, layout)(initial)
        residual = mapped.mapped_x - initial
        assert residual.shape == initial.shape
        assert torch.isfinite(residual).all()
        assert set(mapped.__dataclass_fields__) == {
            "mapped_x",
            "spectral",
            "driving_potential",
        }
        assert not hasattr(mapped, "method_state")
        assert not hasattr(mapped, "iterations")


@pytest.mark.parametrize(
    "converger",
    (
        pytest.param(LinearMethod, id="linear"),
        pytest.param(AndersonMethod, id="anderson"),
    ),
)
def test_solver_outcome_retains_the_convergence_tested_input(
    converger: type,
) -> None:
    """The selected map result and convergence residual use the same x* input."""
    tolerance = 1.0e-12
    case = _case("H2", converger=converger, tolerance=tolerance, maxiter=300)
    record = _run_root(
        case,
        case.inputs,
        tolerance=tolerance,
        maxiter=300,
        eager=True,
        converger=converger,
    )
    assert record.converged
    torch.testing.assert_close(
        record.root, record.last_map_input, rtol=0, atol=0
    )
    torch.testing.assert_close(
        record.outcome.map_result.mapped_x,
        record.last_map_result.mapped_x,
        rtol=0,
        atol=0,
    )
    for name in ("hamiltonian", "density"):
        torch.testing.assert_close(
            getattr(record.outcome.map_result, name),
            getattr(record.last_map_result, name),
            rtol=0,
            atol=0,
        )
    for name in ("emo", "occupation"):
        torch.testing.assert_close(
            getattr(record.outcome.map_result.spectral, name),
            getattr(record.last_map_result.spectral, name),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        record.residual,
        record.outcome.map_result.mapped_x - record.root,
        rtol=0,
        atol=0,
    )
    residual_norms = torch.stack(
        (
            torch.linalg.vector_norm(record.residual),
            record.residual.abs().amax(),
        )
    )
    torch.testing.assert_close(
        record.reported_residual, residual_norms, rtol=0, atol=0
    )
    assert (
        bool((residual_norms[0] < tolerance) & (residual_norms[1] < tolerance))
        == record.converged
    )
    assert record.solver_map_evaluations == record.map_count
    assert record.diagnostic_map_evaluations == 0
    torch.testing.assert_close(
        record.last_map_input, record.root, rtol=0, atol=0
    )
    assert record.last_map_result.mapped_x.shape == record.root.shape


@pytest.mark.parametrize(
    "converger",
    (
        pytest.param(LinearMethod, id="linear"),
        pytest.param(AndersonMethod, id="anderson"),
    ),
)
@pytest.mark.parametrize(
    "representation",
    (
        pytest.param(SCFRepresentation.CHARGE, id="charge"),
        pytest.param(SCFRepresentation.POTENTIAL, id="potential"),
    ),
)
def test_fixed_trip_outcome_retains_accepted_map_input(
    converger: type, representation: SCFRepresentation
) -> None:
    """Masked fixed-trip maps cannot replace the accepted root/result pair."""
    tolerance = 1.0e-10
    case = _case(
        "H2",
        representation=representation,
        converger=converger,
        tolerance=tolerance,
        maxiter=45,
    )
    short = _run_root(
        case,
        case.inputs,
        tolerance=tolerance,
        maxiter=35,
        eager=False,
        converger=converger,
    )
    long = _run_root(
        case,
        case.inputs,
        tolerance=tolerance,
        maxiter=45,
        eager=False,
        converger=converger,
    )

    assert short.converged and long.converged
    assert short.map_count < short.solver_map_evaluations
    assert long.map_count < long.solver_map_evaluations
    # Solver initialization evaluates the raw map once before fixed trips.
    assert short.solver_map_evaluations == 36
    assert long.solver_map_evaluations == 46
    assert short.diagnostic_map_evaluations == 0
    assert long.diagnostic_map_evaluations == 0
    torch.testing.assert_close(short.root, long.root, rtol=0, atol=0)
    torch.testing.assert_close(
        short.outcome.map_result.mapped_x,
        long.outcome.map_result.mapped_x,
        rtol=0,
        atol=0,
    )
    for name in ("hamiltonian", "density", "orbital_energy"):
        torch.testing.assert_close(
            getattr(short.outcome.map_result, name),
            getattr(long.outcome.map_result, name),
            rtol=0,
            atol=0,
        )
    for name in ("emo", "occupation", "orbital_energy"):
        torch.testing.assert_close(
            getattr(short.outcome.map_result.spectral, name),
            getattr(long.outcome.map_result.spectral, name),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(short.residual, long.residual, rtol=0, atol=0)
    torch.testing.assert_close(
        short.reported_residual, long.reported_residual, rtol=0, atol=0
    )
    torch.testing.assert_close(
        short.outcome.iterations, long.outcome.iterations, rtol=0, atol=0
    )

    _, layout = _initial_state(case, case.inputs)
    expected = _map_fn(case, case.inputs, layout)(long.root)
    torch.testing.assert_close(
        long.outcome.map_result.mapped_x, expected.mapped_x, rtol=0, atol=0
    )
    for name in ("hamiltonian", "density", "orbital_energy"):
        torch.testing.assert_close(
            getattr(long.outcome.map_result, name),
            getattr(expected, name),
            rtol=0,
            atol=0,
        )
    for name in ("emo", "occupation", "orbital_energy"):
        torch.testing.assert_close(
            getattr(long.outcome.map_result.spectral, name),
            getattr(expected.spectral, name),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        long.residual,
        long.outcome.map_result.mapped_x - long.root,
        rtol=0,
        atol=0,
    )
    residual_norms = torch.stack(
        (
            torch.linalg.vector_norm(long.residual),
            long.residual.abs().amax(),
        )
    )
    torch.testing.assert_close(
        long.reported_residual, residual_norms, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "converger",
    (
        pytest.param(LinearMethod, id="linear"),
        pytest.param(AndersonMethod, id="anderson"),
    ),
)
@pytest.mark.parametrize(
    "representation",
    (
        pytest.param(SCFRepresentation.CHARGE, id="charge"),
        pytest.param(SCFRepresentation.POTENTIAL, id="potential"),
    ),
)
def test_fixed_trip_exhaustion_retains_last_selected_map_input(
    converger: type, representation: SCFRepresentation
) -> None:
    """An exhausted fixed-trip solve reports its last selected map pair."""
    tolerance = 1.0e-14
    case = _case(
        "H2O",
        representation=representation,
        converger=converger,
        tolerance=tolerance,
        maxiter=1,
    )
    record = _run_root(
        case,
        case.inputs,
        tolerance=tolerance,
        maxiter=1,
        eager=False,
        converger=converger,
    )

    assert not record.converged
    # Initialization plus the one requested fixed trip.
    assert record.map_count == 2
    assert record.solver_map_evaluations == 2
    assert record.diagnostic_map_evaluations == 0
    torch.testing.assert_close(
        record.root, record.last_map_input, rtol=0, atol=0
    )
    torch.testing.assert_close(
        record.outcome.map_result.mapped_x,
        record.last_map_result.mapped_x,
        rtol=0,
        atol=0,
    )
    _, layout = _initial_state(case, case.inputs)
    expected = _map_fn(case, case.inputs, layout)(record.root)
    torch.testing.assert_close(
        record.outcome.map_result.mapped_x, expected.mapped_x, rtol=0, atol=0
    )
    torch.testing.assert_close(
        record.residual,
        record.outcome.map_result.mapped_x - record.root,
        rtol=0,
        atol=0,
    )
    residual_norms = torch.stack(
        (
            torch.linalg.vector_norm(record.residual),
            record.residual.abs().amax(),
        )
    )
    torch.testing.assert_close(
        record.reported_residual, residual_norms, rtol=0, atol=0
    )


@pytest.mark.parametrize("maxiter", (1, 0), ids=("one-step", "zero-step"))
def test_nonconverged_root_reference_is_not_presented_as_a_root(
    maxiter: int,
) -> None:
    """Zero-step and one-step results are rejected as physical-root records."""
    case = _case("H2O", maxiter=maxiter, tolerance=1.0e-14)
    record = _run_root(
        case,
        case.inputs,
        tolerance=1.0e-14,
        maxiter=maxiter,
        eager=True,
    )
    assert not record.converged
    assert (
        record.map_count == 1 if maxiter == 0 else 1 <= record.map_count <= 2
    )
    with pytest.raises(SCFConvergenceError, match="physical-root derivative"):
        _require_converged(record)


def _require_converged(record: _RootRecord) -> None:
    if not record.converged:
        raise SCFConvergenceError("Cannot report a physical-root derivative")


def _request_root_derivative(converged: Tensor, derivative_request):
    """Apply the pre-transform status gate before a derivative request."""
    if not bool(converged.all()):
        raise SCFConvergenceError(
            "Physical-root derivatives require converged roots"
        )
    return derivative_request()


def test_host_preflight_rejects_nonconverged_root_status() -> None:
    """The reference host gate rejects invalid status before its callback."""
    entered_transform = False

    def derivative_request() -> str:
        nonlocal entered_transform
        entered_transform = True
        return "derivative"

    for converged in (
        torch.tensor(False),
        torch.tensor([True, False]),
    ):
        with pytest.raises(
            SCFConvergenceError, match="require converged roots"
        ):
            _request_root_derivative(converged, derivative_request)
        assert not entered_transform
    assert (
        _request_root_derivative(torch.tensor([True, True]), derivative_request)
        == "derivative"
    )
    assert entered_transform


def _no2_trajectory_energy(
    base_case: SimpleNamespace,
    field_value: Tensor,
    displacement: Tensor,
    *,
    tolerance: float,
    maxiter: int = 40,
) -> Tensor:
    """Evaluate fixed-trip finalized NO2 energy at one field and displacement."""
    model_system = base_case.system
    positions = base_case.positions.clone()
    positions[0, 1] = positions[0, 1] + displacement
    matrices, refocc, _ = build_integral_matrices(
        model_system.integral_setup,
        model_system.h0_setup,
        positions,
        base_case.charge,
    )
    n0, occupation, nel = get_refocc(
        refocc, base_case.charge, base_case.spin, model_system.ihelp
    )
    interactions = InteractionList(
        *model_system.interactions.components,
        ElectricField(field_value * _FIELD_AXIS),
        device=positions.device,
        dtype=positions.dtype,
    )
    data = _interaction_data(model_system, positions, interactions)
    inputs = setup_scf_inputs(
        model_system.numbers,
        matrices,
        n0,
        occupation,
        nel,
        interactions,
        data,
        model_system.ihelp,
        model_system.config.scf,
    )
    guess = get_guess(
        model_system.numbers,
        positions,
        base_case.charge,
        model_system.ihelp,
        model_system.config.scf.guess,
    )
    call_case = SimpleNamespace(
        system=model_system,
        initial_charges=guess,
    )
    return _run_root(
        call_case,
        inputs,
        tolerance=tolerance,
        maxiter=maxiter,
        eager=False,
    ).finalized_energy


def _fresh_physical_fd(quantity: str, step: float) -> float:
    """Evaluate an independent direct-energy FD with a fresh Calculator per point."""
    system = SYSTEMS["NO2"]

    def energy(positions: Tensor, field: Tensor) -> float:
        calculator = _calculator(system, "gfn2", "pytorch")
        value = calculator.energy(
            positions.clone(),
            system.charge,
            system.spin,
            field=field,
        )
        return float(value.sum().detach())

    if quantity == "hyperpolarizability":
        points = [
            energy(system.positions, offset * step * _FIELD_AXIS)
            for offset in (-2.0, -1.0, 1.0, 2.0)
        ]
        return -(points[3] - 2.0 * points[2] + 2.0 * points[1] - points[0]) / (
            2.0 * step**3
        )

    if quantity == "pol_deriv":
        alpha = []
        for sign in (-1.0, 1.0):
            positions = system.positions.clone()
            positions[0, 1] += sign * step
            values = [
                energy(positions, offset * step * _FIELD_AXIS)
                for offset in (1.0, 0.0, -1.0)
            ]
            alpha.append(-(values[0] - 2.0 * values[1] + values[2]) / step**2)
        return (alpha[1] - alpha[0]) / (2.0 * step)

    raise ValueError(f"Unknown physical finite-difference quantity: {quantity}")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("quantity", "tblite_ref", "loose_gap"),
    (
        ("hyperpolarizability", -3.35277, 1.0),
        ("pol_deriv", 0.7977, 0.1),
    ),
)
def test_no2_tight_root_third_order_reference(
    quantity: str,
    tblite_ref: float,
    loose_gap: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tight functional roots recover the independent physical 3rd-order FD."""
    _force_lorentzian(monkeypatch)
    base_case = _case(
        "NO2",
        method=GFN2_XTB,
        representation=SCFRepresentation.POTENTIAL,
        converger=AndersonMethod,
        tolerance=1.0e-12,
        maxiter=300,
        temperature=300.0,
        field=torch.zeros(3, **_DD),
    )
    thirds = []
    for tolerance in (1.0e-10, 1.0e-12):
        root = _run_root(
            base_case,
            base_case.inputs,
            tolerance=tolerance,
            maxiter=300,
            eager=True,
        )
        assert root.converged
        assert root.map_count < 300
        assert float(root.residual.abs().max()) < max(10.0 * tolerance, 5.0e-14)
        field = torch.zeros((), **_DD, requires_grad=True)
        geometry = torch.zeros(
            (), **_DD, requires_grad=(quantity == "pol_deriv")
        )
        energy = _no2_trajectory_energy(
            base_case, field, geometry, tolerance=tolerance
        )
        first = torch.autograd.grad(energy, field, create_graph=True)[0]
        second = torch.autograd.grad(first, field, create_graph=True)[0]
        target = field if quantity == "hyperpolarizability" else geometry
        third = torch.autograd.grad(-second, target)[0]
        thirds.append(float(third.detach()))
    physical_fd = [
        _fresh_physical_fd(quantity, step) for step in (1.0e-3, 5.0e-4)
    ]
    physical_spread = max(physical_fd) - min(physical_fd)
    physical_target = float(sum(physical_fd) / len(physical_fd))
    assert physical_spread < (
        5.0e-4 if quantity == "hyperpolarizability" else 1.0e-3
    )
    physical_bound = max(4.0 * physical_spread, 1.0e-3)
    assert abs(thirds[0] - physical_target) > loose_gap
    assert abs(thirds[1] - physical_target) < physical_bound
    assert abs(physical_target - tblite_ref) < TBLITE[quantity].atol


@pytest.mark.parametrize(
    ("representation", "expected_map"),
    (
        (SCFRepresentation.CHARGE, "charge"),
        (SCFRepresentation.POTENTIAL, "potential"),
        (SCFRepresentation.FOCK, "fock"),
    ),
)
def test_fixed_point_map_representation_contract(
    representation: SCFRepresentation, expected_map: str
) -> None:
    """The configured root variable chooses its existing raw map."""
    case = _case("H2", representation=representation)
    initial, layout = _initial_state(case, case.inputs)
    map_fn = _map_fn(case, case.inputs, layout)
    expected = {
        "charge": scf_map,
        "potential": potential_map,
        "fock": fock_map,
    }[expected_map]
    if representation is SCFRepresentation.POTENTIAL:
        expected_result = expected(initial, case.inputs, layout)
    else:
        expected_result = expected(initial, case.inputs)
    actual = map_fn(initial)
    torch.testing.assert_close(actual.mapped_x, expected_result.mapped_x)
    assert type(actual.driving_potential) is type(
        expected_result.driving_potential
    )


def test_simple_and_anderson_reach_the_same_h2_root_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tight physical-root energy derivatives do not depend on the converger."""
    _force_lorentzian(monkeypatch)
    tolerance = 1.0e-12
    records = {}
    for converger in (LinearMethod, AndersonMethod):
        case = _case(
            "H2", converger=converger, tolerance=tolerance, maxiter=300
        )
        inputs = _inputs_at_field(case, torch.zeros((), **_DD))
        record = _run_root(
            case,
            inputs,
            tolerance=tolerance,
            maxiter=300,
            eager=True,
            converger=converger,
        )
        assert record.converged
        assert float(record.residual.abs().max()) < 1.0e-10
        records[converger] = record
    torch.testing.assert_close(
        records[LinearMethod].root,
        records[AndersonMethod].root,
        rtol=0.0,
        atol=2.0e-9,
    )
    torch.testing.assert_close(
        records[LinearMethod].finalized_energy,
        records[AndersonMethod].finalized_energy,
        rtol=0.0,
        atol=2.0e-11,
    )
    for name in ("density", "occupation", "emo"):
        torch.testing.assert_close(
            getattr(
                records[LinearMethod].outcome.map_result.spectral, name
            ),
            getattr(
                records[AndersonMethod].outcome.map_result.spectral,
                name,
            ),
            rtol=0.0,
            atol=2.0e-9,
        )

    derivatives = []
    for converger in (LinearMethod, AndersonMethod):
        case = _case(
            "H2", converger=converger, tolerance=tolerance, maxiter=300
        )

        def energy(parameter: Tensor) -> Tensor:
            inputs = _inputs_at_field(case, parameter)
            record = _run_root(
                case,
                inputs,
                tolerance=tolerance,
                maxiter=300,
                eager=True,
                converger=converger,
            )
            assert record.converged
            return record.finalized_energy

        parameter = torch.zeros((), **_DD, requires_grad=True)
        value = energy(parameter)
        first = torch.autograd.grad(value, parameter, create_graph=True)[0]
        second = torch.autograd.grad(first, parameter)[0]
        derivatives.append(torch.stack((first, second)))
    torch.testing.assert_close(
        derivatives[0], derivatives[1], rtol=1.0e-5, atol=2.0e-7
    )


@pytest.mark.parametrize(
    ("sample_name", "method", "temperature", "representation"),
    (
        pytest.param(
            "H2",
            GFN1_XTB,
            0.0,
            SCFRepresentation.CHARGE,
            id="h2-charge",
        ),
        pytest.param(
            "H2",
            GFN1_XTB,
            0.0,
            SCFRepresentation.POTENTIAL,
            id="h2-potential",
        ),
        pytest.param(
            "H2",
            GFN1_XTB,
            0.0,
            SCFRepresentation.FOCK,
            id="h2-fock",
        ),
        pytest.param(
            "H2",
            GFN1_XTB,
            300.0,
            SCFRepresentation.CHARGE,
            id="h2-finite-temperature",
        ),
        pytest.param(
            "NO2",
            GFN2_XTB,
            300.0,
            SCFRepresentation.POTENTIAL,
            marks=pytest.mark.slow,
            id="no2-potential",
        ),
    ),
)
def test_jacobian_root_reference_matches_unrolled_and_finite_difference(
    sample_name: str,
    method: int,
    temperature: float,
    representation: SCFRepresentation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense IFT response matches unrolled and physical FD by representation."""
    _force_lorentzian(monkeypatch)
    tolerance = 1.0e-12
    case = _case(
        sample_name,
        method=method,
        representation=representation,
        tolerance=tolerance,
        temperature=temperature,
        field=torch.zeros(3, **_DD),
    )
    assert "dispersion_d3" not in {
        item.kind for item in case.inputs.interactions.entries
    }
    root_record = _run_root(
        case,
        case.inputs,
        tolerance=tolerance,
        maxiter=300,
        eager=True,
    )
    assert root_record.converged
    assert float(root_record.residual.abs().max()) < 2.0e-11
    root = root_record.root.detach()

    _, layout = _initial_state(case, case.inputs)

    def mapped(state: Tensor, parameter: Tensor) -> Tensor:
        inputs = _inputs_at_field(case, parameter)
        return _map_fn(case, inputs, layout)(state).mapped_x

    origin = torch.zeros((), **_DD)
    state_jacobian = jacrev(lambda state: mapped(state, origin))(root)
    parameter_tangent = jvp(
        lambda parameter: mapped(root, parameter),
        (origin,),
        (torch.ones_like(origin),),
    )[1]
    tangent_matrix = torch.eye(root.numel(), **_DD) - state_jacobian.reshape(
        root.numel(), root.numel()
    )
    tangent = torch.linalg.solve(
        tangent_matrix,
        parameter_tangent.reshape(-1),
    ).reshape_as(root)
    torch.testing.assert_close(
        tangent_matrix @ tangent.reshape(-1),
        parameter_tangent.reshape(-1),
        rtol=1.0e-10,
        atol=1.0e-11,
    )

    def energy_at_state(state: Tensor, parameter: Tensor) -> Tensor:
        inputs = _inputs_at_field(case, parameter)
        map_result = _map_fn(case, inputs, layout)(state)
        zero = state.new_zeros(())
        yes = torch.ones((), dtype=torch.bool, device=state.device)
        finalized = _finalize_map_result(
            case, map_result, inputs, zero, yes, zero, root_x=state
        )
        return (finalized.energy + finalized.fenergy).sum()

    state_for_grad = root.detach().requires_grad_(True)
    parameter_for_grad = origin.detach().requires_grad_(True)
    energy = energy_at_state(state_for_grad, parameter_for_grad)
    state_gradient, explicit_gradient = torch.autograd.grad(
        energy, (state_for_grad, parameter_for_grad)
    )
    jacobian_response = explicit_gradient + (state_gradient * tangent).sum()

    def unrolled_energy(parameter: Tensor) -> Tensor:
        record = _run_root(
            case,
            _inputs_at_field(case, parameter),
            tolerance=tolerance,
            maxiter=45,
            eager=False,
        )
        assert record.converged
        return record.finalized_energy

    unrolled_parameter = origin.detach().requires_grad_(True)
    unrolled_value = unrolled_energy(unrolled_parameter)
    unrolled_derivative = torch.autograd.grad(
        unrolled_value, unrolled_parameter
    )[0]
    torch.testing.assert_close(
        jacobian_response, unrolled_derivative, rtol=2.0e-4, atol=2.0e-6
    )

    step = 5.0e-4

    def root_energy_at(field_step: float) -> float:
        record = _run_root(
            case,
            _inputs_at_field(case, torch.tensor(field_step, **_DD)),
            tolerance=tolerance,
            maxiter=300,
            eager=True,
        )
        assert record.converged
        return float(record.finalized_energy.detach())

    finite_difference = (root_energy_at(step) - root_energy_at(-step)) / (
        2.0 * step
    )
    assert torch.isfinite(jacobian_response)
    torch.testing.assert_close(
        jacobian_response.detach(),
        torch.tensor(finite_difference, **_DD),
        rtol=5.0e-4,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        unrolled_derivative.detach(),
        torch.tensor(finite_difference, **_DD),
        rtol=5.0e-4,
        atol=2.0e-5,
    )
