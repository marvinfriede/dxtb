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
"""S12 fixed-trip method-state and result-contract checks."""

from __future__ import annotations

import ast
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from torch.utils import _pytree

from dxtb import GFN1_XTB, labels
from dxtb._src.scf.functional.anderson import AndersonState
from dxtb._src.scf.functional.broyden import BroydenState
from dxtb._src.scf.functional.cdiis import CDIISState
from dxtb._src.scf.functional.method import LinearMethod
from dxtb._src.scf.functional.problem import SCFProblem, SCFRepresentation
from dxtb._src.scf.functional.soscf import SOSCFMethod

_DD = {"device": torch.device("cpu"), "dtype": torch.float64}


def _method_states() -> tuple[AndersonState, BroydenState, CDIISState]:
    """Build representative immutable states with mixed tensor dtypes."""
    vector = torch.arange(6, **_DD)
    matrix = torch.arange(9, **_DD).reshape(3, 3)
    return (
        AndersonState(
            x_hist=vector.reshape(2, 3), f_hist=-vector.reshape(2, 3)
        ),
        BroydenState(
            previous_x=vector,
            previous_f=-vector,
            c_history=vector.reshape(2, 3),
            d_history=-vector.reshape(2, 3),
            valid_history=torch.tensor([True, False]),
            history_count=torch.tensor(1, dtype=torch.int64),
            restart_count=torch.tensor(2, dtype=torch.int64),
            last_update_accepted=torch.tensor(True),
        ),
        CDIISState(
            fock_history=matrix.unsqueeze(0).expand(2, -1, -1).clone(),
            error_history=-matrix.unsqueeze(0).expand(2, -1, -1).clone(),
            history_count=torch.tensor(2, dtype=torch.int64),
            restart_count=torch.tensor(1, dtype=torch.int64),
            prune_count=torch.tensor(3, dtype=torch.int64),
            last_condition=torch.tensor(2.0, **_DD),
            last_regularization=torch.tensor(1.0e-8, **_DD),
            last_coeff_max=torch.tensor(0.7, **_DD),
            last_solve_ok=torch.tensor(True),
        ),
    )


@pytest.mark.parametrize(
    "state", _method_states(), ids=("anderson", "broyden", "cdiis")
)
def test_method_state_pytree_round_trip(state: object) -> None:
    """Method state tensor leaves survive the project PyTree round trip."""
    leaves, structure = _pytree.tree_flatten(state)
    reconstructed = _pytree.tree_unflatten(leaves, structure)

    assert type(reconstructed) is type(state)
    state_fields = fields(state)
    assert len(leaves) == len(state_fields)
    for item in state_fields:
        actual = getattr(reconstructed, item.name)
        expected = getattr(state, item.name)
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)


def test_fixed_trip_capability_matrix() -> None:
    """Only eager-only SOSCF is outside the established fixed-trip contract."""
    from dxtb._src.scf.functional.anderson import AndersonMethod
    from dxtb._src.scf.functional.broyden import BroydenMethod
    from dxtb._src.scf.functional.cdiis import CDIISMethod

    assert LinearMethod.supports_fixed_trip is True
    assert AndersonMethod.supports_fixed_trip is True
    assert BroydenMethod.supports_fixed_trip is True
    assert CDIISMethod.supports_fixed_trip is True
    assert CDIISMethod.required_representation is SCFRepresentation.FOCK
    assert SOSCFMethod.supports_fixed_trip is False


def test_transform_methods_have_no_tensor_to_host_extraction() -> None:
    """Transform-capable method modules do not inspect tensor values in Python."""
    package = Path(__file__).parents[2] / "src/dxtb/_src/scf/functional"
    for filename in ("anderson.py", "broyden.py", "cdiis.py"):
        tree = ast.parse((package / filename).read_text())
        host_extractions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"item", "tolist"}
        ]
        assert not host_extractions, filename


def test_compile_uses_static_fermi_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiled active-smearing maps never call detached adaptive Fermi code."""
    from dxtb import GFN2_XTB
    from dxtb._src.scf.functional import spectral
    from dxtb._src.scf.functional.spectral import eager_primal_fermi_scope

    from .test_functional_root_derivatives import _case, _initial_state

    case = _case(
        "NO2",
        method=GFN2_XTB,
        representation=labels.SCP_MODE_FOCK,
        mixer=labels.MIXER_LINEAR,
        temperature=2000.0,
        maxiter=4,
    )
    assert case.inputs.fermi_settings is not None
    assert case.inputs.fermi_settings.smear
    guess, _ = _initial_state(case, case.inputs)
    problem = SCFProblem(case.inputs, SCFRepresentation.FOCK)

    def reject_adaptive(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError(
            "compiled map entered detached adaptive Fermi code"
        )

    monkeypatch.setattr(spectral, "_matrix_fermi_eager", reject_adaptive)

    def raw_fock_map(value: torch.Tensor) -> torch.Tensor:
        return problem.map(value).mapped_x

    expected = raw_fock_map(guess)
    with eager_primal_fermi_scope(False):
        compiled = torch.compile(raw_fock_map, backend="eager")
        actual = compiled(guess)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-11)


def test_broyden_vmap_lane_permutation_duplicate_and_isolation() -> None:
    """Batched Broyden keeps geometry lanes independent and ordered."""
    from test.test_calculator.test_e6a0_system_vmap import (
        _positions,
        _system,
    )

    system = _system(maxiter=4, mixer=labels.MIXER_BROYDEN)
    positions = _positions()
    lanes = torch.stack((positions[0], positions[1], positions[0]))

    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(lanes)
    permuted_order = torch.tensor([2, 0, 1])
    permuted = torch.func.vmap(system.singlepoint, in_dims=(0,))(
        lanes[permuted_order]
    )
    perturbed_lanes = lanes.clone()
    perturbed_lanes[0, 1, 2] += 0.07
    perturbed = torch.func.vmap(system.singlepoint, in_dims=(0,))(
        perturbed_lanes
    )

    torch.testing.assert_close(mapped.energy[0], mapped.energy[2])
    torch.testing.assert_close(
        permuted.energy,
        mapped.energy[permuted_order],
    )
    torch.testing.assert_close(
        permuted.density,
        mapped.density[permuted_order],
    )
    torch.testing.assert_close(perturbed.energy[1:], mapped.energy[1:])
    torch.testing.assert_close(perturbed.density[1:], mapped.density[1:])
    torch.testing.assert_close(perturbed.residual[1:], mapped.residual[1:])


def test_alternate_response_keeps_primal_residual_semantics() -> None:
    """An alternate IFT coordinate does not replace primal solve diagnostics."""
    from dxtb._src.scf.functional.eager import run_prepared_functional_implicit
    from dxtb._src.scf.functional.response import SCFResponseSettings

    from .test_functional_root_derivatives import _case, _run_root

    case = _case(
        "H2",
        method=GFN1_XTB,
        representation=labels.SCP_MODE_POTENTIAL,
        mixer=labels.MIXER_BROYDEN,
        tolerance=1.0e-11,
    )
    primal = _run_root(
        case,
        case.inputs,
        tolerance=1.0e-11,
        maxiter=120,
        eager=True,
        mixer=labels.MIXER_BROYDEN,
    )
    config = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        mixer=labels.MIXER_BROYDEN,
        maxiter=120,
        x_atol=1.0e-11,
        x_atol_max=1.0e-11,
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
    )

    assert torch.equal(result.iterations, primal.outcome.iterations)
    assert torch.equal(result.residual, primal.reported_residual)
    assert result.diagnostics is not None
    assert result.diagnostics.strategy is not None
