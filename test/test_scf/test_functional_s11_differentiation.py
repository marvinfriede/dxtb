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
"""Strategy-independent implicit response boundary tests."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.scf.functional.implicit import _SCFMapTensorTree
from dxtb._src.scf.functional.problem import SCFProblem, SCFRepresentation
from dxtb._src.scf.functional.response import (
    SCFResponseSettings,
    prepare_response_root,
)
from dxtb._src.scf.functional.solver import setup_solver_settings
from dxtb.scf import (
    AndersonMethod,
    AndersonMethodSettings,
    BroydenMethod,
    BroydenMethodSettings,
    LinearMethod,
    SCFDifferentiation,
    SCFStage,
    SCFStrategy,
)

from .test_functional_root_derivatives import _case, _run_root

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}


def _implicit_config(
    config,
    *,
    mixer: int,
    representation: int | None = None,
    **changes,
):
    """Build an implicit test configuration through the public strategy API."""
    stage = config.strategy.stages[0]
    if mixer == labels.MIXER_LINEAR:
        method = LinearMethod()
    elif mixer == labels.MIXER_ANDERSON:
        settings = (
            stage.method.settings
            if isinstance(stage.method, AndersonMethod)
            else AndersonMethodSettings()
        )
        method = AndersonMethod(settings)
    else:
        settings = (
            stage.method.settings
            if isinstance(stage.method, BroydenMethod)
            else BroydenMethodSettings()
        )
        method = BroydenMethod(settings)
    if representation is not None:
        stage_representation = {
            labels.SCP_MODE_CHARGE: SCFRepresentation.CHARGE,
            labels.SCP_MODE_POTENTIAL: SCFRepresentation.POTENTIAL,
            labels.SCP_MODE_FOCK: SCFRepresentation.FOCK,
        }[representation]
    else:
        stage_representation = stage.representation
    return replace(
        config,
        strategy=SCFStrategy(
            (
                replace(
                    stage, method=method, representation=stage_representation
                ),
            )
        ),
        differentiation=SCFDifferentiation.IMPLICIT,
        **changes,
    )


def _root_record(representation: int = labels.SCP_MODE_POTENTIAL):
    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=representation,
        mixer=labels.MIXER_BROYDEN,
        tolerance=1.0e-12,
        temperature=0.0,
    )
    record = _run_root(
        case,
        case.inputs,
        tolerance=1.0e-12,
        maxiter=100,
        eager=True,
        mixer=labels.MIXER_BROYDEN,
    )
    assert record.converged
    settings = setup_solver_settings(case.system.config.scf, record.root.dtype)
    return case, record, settings


def test_response_root_conversion_is_single_certified_physical_root() -> None:
    """All response spaces encode roots from each source representation."""
    representations = (
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    )
    for source in representations:
        _, record, settings = _root_record(
            {
                SCFRepresentation.CHARGE: labels.SCP_MODE_CHARGE,
                SCFRepresentation.POTENTIAL: labels.SCP_MODE_POTENTIAL,
                SCFRepresentation.FOCK: labels.SCP_MODE_FOCK,
            }[source]
        )
        for target in representations:
            prepared = prepare_response_root(
                record.outcome,
                settings,
                SCFResponseSettings(representation=target),
            )
            assert prepared.problem.representation is target
            assert bool(prepared.certification.certified)
            assert prepared.root_x.shape == prepared.map_result.mapped_x.shape
            assert len(prepared.adapter.parameter_paths) == len(
                prepared.parameters
            )
            mapped = prepared.adapter.evaluate(
                prepared.root_x, *prepared.parameters
            )
            torch.testing.assert_close(
                mapped.mapped_x, prepared.map_result.mapped_x
            )
            torch.testing.assert_close(
                prepared.map_result.density, record.outcome.map_result.density
            )
            torch.testing.assert_close(
                prepared.map_result.charges.mono,
                record.outcome.map_result.charges.mono,
            )


@pytest.mark.parametrize(
    "representation",
    [
        SCFRepresentation.CHARGE,
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ],
    ids=("charge", "potential", "fock-full-matrix"),
)
def test_matrix_free_tangent_and_adjoint_match_dense_operator(
    representation: SCFRepresentation,
) -> None:
    """Actual response coordinates match dense I-Gx and its transpose."""
    _, record, settings = _root_record()
    from dxtb._src.scf.functional.implicit import solve_linear

    prepared = prepare_response_root(
        record.outcome,
        settings,
        SCFResponseSettings(representation=representation),
    )
    root_shape = prepared.root_x.shape

    def flat_map(value: torch.Tensor) -> torch.Tensor:
        mapped = prepared.adapter(
            value.reshape(root_shape), *prepared.parameters
        )
        return mapped.reshape(-1)

    root = prepared.root_x.reshape(-1)
    jacobian = torch.func.jacfwd(flat_map)(root)
    identity = torch.eye(root.numel(), **_DD)
    dense_operator = identity - jacobian
    direction = torch.arange(1, root.numel() + 1, **_DD) / root.numel()
    direction = direction / torch.linalg.vector_norm(direction)
    _, jvp = torch.func.jvp(flat_map, (root,), (direction,))
    torch.testing.assert_close(
        direction - jvp,
        dense_operator @ direction,
        atol=2.0e-10,
        rtol=2.0e-9,
    )

    _, pullback = torch.func.vjp(flat_map, root)
    transpose_product = direction - pullback(direction)[0]
    torch.testing.assert_close(
        transpose_product,
        dense_operator.mT @ direction,
        atol=2.0e-10,
        rtol=2.0e-9,
    )

    rhs = torch.linspace(-0.3, 0.7, root.numel(), **_DD)
    tangent = solve_linear(
        lambda value: value - torch.func.jvp(flat_map, (root,), (value,))[1],
        rhs,
    )
    adjoint = solve_linear(lambda value: value - pullback(value)[0], rhs)
    assert bool(tangent.converged)
    assert bool(adjoint.converged)
    torch.testing.assert_close(
        tangent.solution,
        torch.linalg.solve(dense_operator, rhs),
        atol=2.0e-9,
        rtol=2.0e-8,
    )
    torch.testing.assert_close(
        adjoint.solution,
        torch.linalg.solve(dense_operator.mT, rhs),
        atol=2.0e-9,
        rtol=2.0e-8,
    )


def test_default_response_root_preserves_primal_pair() -> None:
    """Omitted response settings retain the final primal representation."""
    _, record, settings = _root_record(labels.SCP_MODE_FOCK)
    prepared = prepare_response_root(record.outcome, settings)

    assert prepared.problem.representation is SCFRepresentation.FOCK
    assert prepared.root_x is record.outcome.root_x
    assert prepared.map_result is record.outcome.map_result
    assert prepared.outcome.root_x is record.outcome.root_x
    assert not hasattr(prepared.outcome, "trace")
    assert not hasattr(prepared.outcome, "strategy_diagnostics")
    assert prepared.settings == SCFResponseSettings()


def test_alternate_response_candidate_uses_one_raw_map_and_no_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alternate Fock root is independently mapped once, never iterated."""
    original = _SCFMapTensorTree.evaluate
    calls = 0

    def counted(self, x: torch.Tensor, *parameters: torch.Tensor):
        nonlocal calls
        calls += 1
        return original(self, x, *parameters)

    monkeypatch.setattr(_SCFMapTensorTree, "evaluate", counted)
    _, record, settings = _root_record(labels.SCP_MODE_POTENTIAL)
    calls = 0
    prepared = prepare_response_root(
        record.outcome,
        settings,
        SCFResponseSettings(representation=SCFRepresentation.FOCK),
    )

    assert calls == 1
    assert prepared.independent_map_result is prepared.map_result
    assert bool(prepared.certification.certified)


def test_first_and_second_responses_ignore_strategy_and_coordinates() -> None:
    """Broyden and Anderson agree across charge, potential and Fock IFT spaces."""
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    field = torch.zeros(3, **_DD, requires_grad=True)
    results = []
    for mixer in (labels.MIXER_BROYDEN, labels.MIXER_ANDERSON):
        case = _case(
            "H2",
            method=GFN1_XTB,
            representation=labels.SCP_MODE_POTENTIAL,
            mixer=mixer,
            tolerance=1.0e-12,
            temperature=0.0,
            field=field,
        )
        config = _implicit_config(
            case.system.config.scf,
            mixer=mixer,
            maxiter=120,
            x_atol=1.0e-12,
            x_atol_max=1.0e-12,
        )
        for representation in (
            SCFRepresentation.CHARGE,
            SCFRepresentation.POTENTIAL,
            SCFRepresentation.FOCK,
        ):
            result = run_prepared_functional_implicit(
                case.numbers,
                case.initial_charges,
                case.n0,
                case.occupation,
                case.nel,
                case.interactions,
                case.interaction_data,
                case.system.ihelp,
                config,
                case.integrals,
                response_settings=SCFResponseSettings(
                    representation=representation
                ),
            )
            assert bool(result.converged)
            observable = result.energy.sum() + result.fenergy.sum()
            first = torch.autograd.grad(
                observable, field, create_graph=True, retain_graph=True
            )[0][1]
            second = torch.autograd.grad(first, field, retain_graph=True)[0][1]
            assert torch.isfinite(first)
            assert torch.isfinite(second)
            results.append(
                (
                    result.energy.sum().detach(),
                    result.fenergy.sum().detach(),
                    result.map_result.density.detach(),
                    result.final_charges.mono.detach(),
                    first.detach(),
                    second.detach(),
                )
            )

    reference = results[0]
    for values in results[1:]:
        for actual, expected in zip(values, reference):
            torch.testing.assert_close(
                actual, expected, atol=2.0e-8, rtol=2.0e-7
            )


def test_geometry_response_is_independent_of_primal_strategy() -> None:
    """Geometry, field, and mixed responses agree for Broyden and CDIIS."""
    from dxtb._src.scf.functional.broyden import BroydenMethod
    from dxtb._src.scf.functional.cdiis import CDIISMethod
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
    from dxtb._src.scf.functional.strategy import SCFStage, SCFStrategy

    from .test_functional_root_derivatives import samples

    base_positions = samples["H2"]["positions"].to(**_DD)
    bond = base_positions[1] - base_positions[0]
    bond = bond / torch.linalg.vector_norm(bond)
    displacement_direction = torch.zeros_like(base_positions)
    displacement_direction[1] = bond
    derivatives = []
    field_direction = torch.tensor([0.0, 1.0, 0.0], **_DD)

    strategies = (
        (
            SCFRepresentation.POTENTIAL,
            SCFStrategy(
                (
                    SCFStage(
                        BroydenMethod(),
                        representation=SCFRepresentation.POTENTIAL,
                    ),
                )
            ),
        ),
        (
            SCFRepresentation.FOCK,
            SCFStrategy(
                (
                    SCFStage(
                        CDIISMethod(), representation=SCFRepresentation.FOCK
                    ),
                )
            ),
        ),
    )
    for primal_representation, strategy in strategies:
        displacement = torch.zeros((), **_DD, requires_grad=True)
        field_parameter = torch.zeros((), **_DD, requires_grad=True)
        positions = base_positions + displacement * displacement_direction
        field = field_parameter * field_direction
        case = _case(
            "H2",
            method=GFN1_XTB,
            representation=(
                labels.SCP_MODE_POTENTIAL
                if primal_representation is SCFRepresentation.POTENTIAL
                else labels.SCP_MODE_FOCK
            ),
            mixer=labels.MIXER_BROYDEN,
            tolerance=1.0e-12,
            temperature=0.0,
            field=field,
            positions_override=positions,
        )
        config = _implicit_config(
            case.system.config.scf,
            mixer=labels.MIXER_BROYDEN,
            maxiter=120,
            x_atol=1.0e-12,
            x_atol_max=1.0e-12,
        )
        result = run_prepared_functional_implicit(
            case.numbers,
            case.initial_charges,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            response_settings=SCFResponseSettings(
                representation=SCFRepresentation.POTENTIAL
            ),
            strategy_override=strategy,
        )
        assert bool(result.converged)
        energy = result.energy.sum() + result.fenergy.sum()
        geometry_gradient = torch.autograd.grad(
            energy, displacement, create_graph=True, retain_graph=True
        )[0]
        geometry_hessian = torch.autograd.grad(
            geometry_gradient, displacement, retain_graph=True
        )[0]
        field_gradient = torch.autograd.grad(
            energy, field_parameter, create_graph=True, retain_graph=True
        )[0]
        field_hessian = torch.autograd.grad(
            field_gradient, field_parameter, retain_graph=True
        )[0]
        mixed_response = torch.autograd.grad(field_gradient, displacement)[0]
        response = torch.stack(
            (geometry_gradient, geometry_hessian, field_hessian, mixed_response)
        )
        assert torch.isfinite(response).all()
        derivatives.append(response.detach())

    torch.testing.assert_close(
        derivatives[0], derivatives[1], atol=2.0e-9, rtol=2.0e-8
    )


def test_response_tolerance_sweep_preserves_higher_derivatives() -> None:
    """The default IFT tolerances stabilize H2 response through order three."""
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    derivatives = []
    for rtol in (1.0e-8, 1.0e-12):
        field = torch.zeros(3, **_DD, requires_grad=True)
        case = _case(
            "H2",
            method=GFN1_XTB,
            representation=labels.SCP_MODE_POTENTIAL,
            mixer=labels.MIXER_BROYDEN,
            tolerance=1.0e-12,
            temperature=0.0,
            field=field,
        )
        config = _implicit_config(
            case.system.config.scf,
            mixer=labels.MIXER_BROYDEN,
            maxiter=120,
            x_atol=1.0e-12,
            x_atol_max=1.0e-12,
        )
        result = run_prepared_functional_implicit(
            case.numbers,
            case.initial_charges,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            response_settings=SCFResponseSettings(
                representation=SCFRepresentation.POTENTIAL,
                rtol=rtol,
                atol=rtol * 1.0e-2,
            ),
        )
        observable = result.energy.sum() + result.fenergy.sum()
        first = torch.autograd.grad(
            observable, field, create_graph=True, retain_graph=True
        )[0][1]
        second = torch.autograd.grad(
            first, field, create_graph=True, retain_graph=True
        )[0][1]
        third = torch.autograd.grad(second, field)[0][1]
        assert torch.isfinite(first)
        assert torch.isfinite(second)
        assert torch.isfinite(third)
        derivatives.append(torch.stack((first, second, third)).detach())

    for value in derivatives[1:]:
        torch.testing.assert_close(
            value, derivatives[0], atol=2.0e-7, rtol=2.0e-6
        )


def test_broyden_cdiis_soscf_implicit_matches_cdiis_reference() -> None:
    """The full fixed-occupation hybrid history does not enter the IFT."""
    from dxtb._src.scf.functional.anderson import (
        AndersonMethod,
        AndersonMethodSettings,
    )
    from dxtb._src.scf.functional.broyden import BroydenMethod
    from dxtb._src.scf.functional.cdiis import CDIISMethod
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
    from dxtb._src.scf.functional.soscf import SOSCFMethod
    from dxtb._src.scf.functional.strategy import (
        SCFStage,
        SCFStrategy,
        SCFSwitchPolicy,
    )

    field_responses = []
    physical_values = []
    cases = (
        (
            "cdiis",
            labels.SCP_MODE_FOCK,
            SCFStrategy((SCFStage(CDIISMethod()),)),
        ),
        (
            "broyden_cdiis_soscf",
            labels.SCP_MODE_POTENTIAL,
            SCFStrategy(
                (
                    SCFStage(
                        BroydenMethod(),
                        representation=SCFRepresentation.POTENTIAL,
                        switch_policy=SCFSwitchPolicy(max_stage_iterations=8),
                    ),
                    SCFStage(
                        CDIISMethod(),
                        representation=SCFRepresentation.FOCK,
                        switch_policy=SCFSwitchPolicy(
                            min_stage_iterations=2,
                            stationarity=1.0e-3,
                            max_stage_iterations=30,
                        ),
                        initial_damp=1.0,
                    ),
                    SCFStage(
                        SOSCFMethod(),
                        representation=SCFRepresentation.FOCK,
                        initial_damp=1.0,
                    ),
                )
            ),
        ),
    )
    for name, representation, strategy in cases:
        field = torch.zeros(3, **_DD, requires_grad=True)
        case = _case(
            "H2O",
            method=GFN2_XTB,
            representation=representation,
            mixer=labels.MIXER_BROYDEN,
            tolerance=1.0e-10,
            maxiter=300,
            temperature=0.0,
            field=field,
        )
        config = _implicit_config(
            case.system.config.scf,
            mixer=labels.MIXER_BROYDEN,
            representation=representation,
            maxiter=300,
            x_atol=1.0e-10,
            x_atol_max=1.0e-10,
        )
        result = run_prepared_functional_implicit(
            case.numbers,
            case.initial_charges,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            response_settings=SCFResponseSettings(
                representation=SCFRepresentation.FOCK
            ),
            strategy_override=strategy,
        )
        assert bool(result.converged)
        if name == "broyden_cdiis_soscf":
            assert result.diagnostics is not None
            assert int(result.diagnostics.strategy.final_stage_index) == 2
        if name == "anderson_cdiis":
            assert result.diagnostics is not None
            assert int(result.diagnostics.strategy.final_stage_index) == 1
        observable = result.energy.sum() + result.fenergy.sum()
        first = torch.autograd.grad(
            observable, field, create_graph=True, retain_graph=True
        )[0][1]
        second = torch.autograd.grad(
            first, field, create_graph=True, retain_graph=True
        )[0][1]
        third = torch.autograd.grad(second, field)[0][1]
        assert torch.isfinite(first)
        assert torch.isfinite(second)
        assert torch.isfinite(third)
        physical_values.append(
            (
                result.energy.sum().detach(),
                result.map_result.density.detach(),
                result.final_charges.mono.detach(),
            )
        )
        field_responses.append(
            (first.detach(), second.detach(), third.detach())
        )

    for actual, expected in zip(physical_values[1], physical_values[0]):
        torch.testing.assert_close(actual, expected, atol=2.0e-7, rtol=2.0e-7)
    for actual, expected in zip(field_responses[1], field_responses[0]):
        torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-4)


def test_active_smearing_responses_agree_across_first_order_strategies() -> (
    None
):
    """Thermal implicit derivatives do not depend on Broyden/CDIIS history."""
    from dxtb._src.scf.functional.anderson import (
        AndersonMethod,
        AndersonMethodSettings,
    )
    from dxtb._src.scf.functional.broyden import BroydenMethod
    from dxtb._src.scf.functional.cdiis import CDIISMethod
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
    from dxtb._src.scf.functional.problem import SCFRepresentation
    from dxtb._src.scf.functional.strategy import (
        SCFStage,
        SCFStrategy,
        SCFSwitchPolicy,
    )

    strategies = (
        (
            "broyden",
            labels.MIXER_BROYDEN,
            SCFStrategy(
                (
                    SCFStage(
                        BroydenMethod(),
                        representation=SCFRepresentation.FOCK,
                    ),
                )
            ),
        ),
        (
            "anderson",
            labels.MIXER_ANDERSON,
            SCFStrategy(
                (
                    SCFStage(
                        AndersonMethod(
                            AndersonMethodSettings(0.1, 5, True, 0.01)
                        ),
                        representation=SCFRepresentation.FOCK,
                    ),
                )
            ),
        ),
        (
            "broyden_cdiis",
            labels.MIXER_BROYDEN,
            SCFStrategy(
                (
                    SCFStage(
                        BroydenMethod(),
                        representation=SCFRepresentation.FOCK,
                        switch_policy=SCFSwitchPolicy(max_stage_iterations=12),
                    ),
                    SCFStage(
                        CDIISMethod(),
                        representation=SCFRepresentation.FOCK,
                    ),
                )
            ),
        ),
    )
    results = []
    for name, mixer, strategy in strategies:
        field = torch.zeros(3, **_DD, requires_grad=True)
        case = _case(
            "NO2",
            method=GFN2_XTB,
            representation=labels.SCP_MODE_FOCK,
            mixer=mixer,
            tolerance=1.0e-10,
            maxiter=300,
            temperature=2000.0,
            field=field,
        )
        assert case.inputs.fermi_settings is not None
        assert case.inputs.fermi_settings.smear
        config = _implicit_config(
            case.system.config.scf,
            mixer=mixer,
            representation=labels.SCP_MODE_FOCK,
            maxiter=300,
            x_atol=1.0e-10,
            x_atol_max=1.0e-10,
        )
        result = run_prepared_functional_implicit(
            case.numbers,
            case.initial_charges,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            response_settings=SCFResponseSettings(
                representation=SCFRepresentation.FOCK
            ),
            strategy_override=strategy,
        )
        assert bool(result.converged)
        occupation = result.occupation
        assert bool(((occupation > 0.0) & (occupation < 1.0)).any())
        observable = result.energy.sum() + result.fenergy.sum()
        first = torch.autograd.grad(
            observable, field, create_graph=True, retain_graph=True
        )[0][1]
        second = torch.autograd.grad(first, field, retain_graph=True)[0][1]
        assert torch.isfinite(first)
        assert torch.isfinite(second)
        results.append(
            (
                result.energy.sum().detach(),
                result.fenergy.sum().detach(),
                result.map_result.density.detach(),
                result.final_charges.mono.detach(),
                first.detach(),
                second.detach(),
            )
        )

    reference = results[0]
    for values in results[1:]:
        for actual, expected in zip(values, reference):
            torch.testing.assert_close(
                actual, expected, atol=2.0e-7, rtol=2.0e-6
            )


def test_no2_third_order_response_agrees_in_potential_and_fock_coordinates() -> (
    None
):
    """Open-shell GFN2 hyperpolarizability is coordinate independent."""
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    derivatives = []
    for representation in (
        SCFRepresentation.POTENTIAL,
        SCFRepresentation.FOCK,
    ):
        field = torch.zeros(3, **_DD, requires_grad=True)
        case = _case(
            "NO2",
            method=GFN2_XTB,
            representation=labels.SCP_MODE_POTENTIAL,
            mixer=labels.MIXER_BROYDEN,
            tolerance=1.0e-12,
            maxiter=300,
            temperature=0.0,
            field=field,
        )
        config = _implicit_config(
            case.system.config.scf,
            mixer=labels.MIXER_BROYDEN,
            maxiter=300,
            x_atol=1.0e-12,
            x_atol_max=1.0e-12,
        )
        result = run_prepared_functional_implicit(
            case.numbers,
            case.initial_charges,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            config,
            case.integrals,
            response_settings=SCFResponseSettings(
                representation=representation
            ),
        )
        assert bool(result.converged)
        observable = result.energy.sum() + result.fenergy.sum()
        first = torch.autograd.grad(observable, field, create_graph=True)[0][1]
        second = torch.autograd.grad(first, field, create_graph=True)[0][1]
        third = torch.autograd.grad(second, field)[0][1]
        assert torch.isfinite(third)
        derivatives.append(third.detach())

    torch.testing.assert_close(
        derivatives[0], derivatives[1], atol=2.0e-5, rtol=2.0e-4
    )


def test_response_adapter_flatten_roundtrip_is_complete() -> None:
    """Flattened physical leaves reproduce the exact raw map."""
    _, record, _ = _root_record()
    problem = record.outcome.final_problem
    adapter, parameters = _SCFMapTensorTree.prepare(
        problem.inputs,
        labels.SCP_MODE_POTENTIAL,
        problem.potential_layout,
    )
    assert len(adapter.parameter_paths) == len(adapter.flatten())
    assert len(adapter.parameter_paths) == len(parameters)
    actual = adapter.evaluate(record.root, *parameters)
    expected = problem.map(record.root)
    torch.testing.assert_close(actual.mapped_x, expected.mapped_x)
    torch.testing.assert_close(actual.density, expected.density)


def test_response_adapter_parameter_tree_covers_thermal_gfn2_field_leaves() -> (
    None
):
    """GFN2, field, multipole and thermal tensors remain explicit IFT leaves."""
    from .test_functional_root_derivatives import _initial_state

    field = torch.zeros(3, **_DD, requires_grad=True)
    case = _case(
        "NO2",
        method=GFN2_XTB,
        representation=labels.SCP_MODE_CHARGE,
        tolerance=1.0e-10,
        temperature=2000.0,
        field=field,
    )
    initial, layout = _initial_state(case, case.inputs)
    adapter, parameters = _SCFMapTensorTree.prepare(
        case.inputs, labels.SCP_MODE_CHARGE, layout
    )
    assert len(adapter.parameter_paths) == len(parameters)
    assert "fermi_settings.kt" in adapter.parameter_paths
    assert any(
        "interactions.entries" in path for path in adapter.parameter_paths
    )
    assert "integrals.dipole" in adapter.parameter_paths
    actual = adapter.evaluate(initial, *parameters)
    expected = SCFProblem(case.inputs, SCFRepresentation.CHARGE).map(initial)
    torch.testing.assert_close(actual.mapped_x, expected.mapped_x)
    torch.testing.assert_close(actual.density, expected.density)


def test_implicit_response_reaches_a_genuine_gfn2_model_parameter() -> None:
    """IFT differentiates through a live GFN2 parameter, not only hcore."""
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    parameter_module = ParamModule(GFN2_XTB, device="cpu", dtype=torch.float64)
    parameter = parameter_module.parameter_tree.charge.effective.gexp.param
    parameter.requires_grad_(True)
    case = _case(
        "H2O",
        method=GFN2_XTB,
        representation=labels.SCP_MODE_POTENTIAL,
        mixer=labels.MIXER_BROYDEN,
        tolerance=1.0e-12,
        temperature=0.0,
        parameter_module=parameter_module,
    )
    config = _implicit_config(
        case.system.config.scf,
        mixer=labels.MIXER_BROYDEN,
        maxiter=300,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    result = run_prepared_functional_implicit(
        case.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
        response_settings=SCFResponseSettings(
            representation=SCFRepresentation.CHARGE
        ),
    )
    observable = result.energy.sum() + result.fenergy.sum()
    first = torch.autograd.grad(observable, parameter, create_graph=True)[0]
    second = torch.autograd.grad(first, parameter)[0]
    assert torch.isfinite(first)
    assert torch.isfinite(second)
    assert first.abs() > 1.0e-12


def test_response_linear_solve_controls_are_forwarded_and_fail_closed() -> None:
    """A one-step physical response limit remains visibly non-finite."""
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit

    field = torch.tensor([0.0, 0.0, 0.0], **_DD, requires_grad=True)
    case = _case(
        "H2O",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        mixer=labels.MIXER_BROYDEN,
        tolerance=1.0e-12,
        temperature=0.0,
        field=field,
    )
    config = _implicit_config(
        case.system.config.scf,
        mixer=labels.MIXER_BROYDEN,
        maxiter=200,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    result = run_prepared_functional_implicit(
        case.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
        response_settings=SCFResponseSettings(rtol=0.0, atol=0.0, maxiter=1),
    )
    assert torch.isfinite(result.energy).all()
    gradient = torch.autograd.grad(
        result.energy.sum(), field, create_graph=True, retain_graph=True
    )[0]
    assert not torch.isfinite(gradient).all()
    second = torch.autograd.grad(gradient.sum(), field, retain_graph=True)[0]
    assert not torch.isfinite(second).all()

    normal = run_prepared_functional_implicit(
        case.numbers,
        case.initial_charges,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        config,
        case.integrals,
    )
    normal_gradient = torch.autograd.grad(normal.energy.sum(), field)[0]
    assert torch.isfinite(normal_gradient).all()


def test_physical_tangent_response_failure_is_nonfinite() -> None:
    """An unconverged physical tangent Krylov iterate is never returned."""
    from dxtb._src.scf.functional.implicit import implicit_root

    from .test_functional_root_derivatives import _initial_state

    field = torch.zeros(3, **_DD, requires_grad=True)
    case = _case(
        "H2O",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_CHARGE,
        mixer=labels.MIXER_BROYDEN,
        tolerance=1.0e-12,
        temperature=0.0,
        field=field,
    )
    # Use a real H2O primal observation and its complete explicit parameter tree.
    from .test_functional_root_derivatives import _run_root

    record = _run_root(
        case,
        case.inputs,
        tolerance=1.0e-12,
        maxiter=200,
        eager=True,
        mixer=labels.MIXER_BROYDEN,
    )
    problem = record.outcome.final_problem
    adapter, parameters = _SCFMapTensorTree.prepare(
        problem.inputs, labels.SCP_MODE_CHARGE
    )
    hcore_index = adapter.parameter_paths.index("integrals.hcore")
    hcore = parameters[hcore_index]
    direction = torch.zeros_like(hcore)
    direction[0, 0] = 1.0
    zero = hcore.new_zeros(())
    one = hcore.new_ones(())

    def attached_sum(increment: Tensor, *, maxiter: int | None) -> Tensor:
        updated = (
            parameters[:hcore_index]
            + (hcore + increment * direction,)
            + parameters[hcore_index + 1 :]
        )
        return implicit_root(
            record.root,
            *updated,
            raw_map=adapter,
            rtol=0.0 if maxiter is not None else 1.0e-10,
            atol=0.0 if maxiter is not None else 1.0e-12,
            maxiter=maxiter,
        ).sum()

    _, failed_tangent = torch.func.jvp(
        lambda increment: attached_sum(increment, maxiter=1),
        (zero,),
        (one,),
    )
    assert not torch.isfinite(failed_tangent)
    _, valid_tangent = torch.func.jvp(
        lambda increment: attached_sum(increment, maxiter=None),
        (zero,),
        (one,),
    )
    assert torch.isfinite(valid_tangent)


def test_response_settings_reject_invalid_linear_controls() -> None:
    """Internal response tolerances and iteration limits are validated."""
    with pytest.raises(ValueError, match="non-negative"):
        SCFResponseSettings(rtol=-1.0)
    with pytest.raises(ValueError, match="finite"):
        SCFResponseSettings(atol=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        SCFResponseSettings(maxiter=0)


def test_response_settings_representation_type_is_explicit() -> None:
    """Only the three physical SCF representations can be requested."""
    settings = SCFResponseSettings(representation=SCFRepresentation.CHARGE)
    assert settings.representation is SCFRepresentation.CHARGE
