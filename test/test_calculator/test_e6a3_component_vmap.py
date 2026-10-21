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
"""Public System component and derivative acceptance for E6a.3."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.components.interactions.solvation.alpb import GeneralizedBorn
from dxtb.config import Config


def _model(
    method: object = GFN1_XTB,
    *,
    numbers: tuple[int, ...] = (8, 1, 1),
    parameters: ParamModule | None = None,
    interactions: tuple[object, ...] = (),
    exclude: tuple[str, ...] = (),
) -> tuple[Model, torch.Tensor]:
    """Build an independent production Model and its atomic numbers."""
    method_label = labels.GFN2_XTB if method is GFN2_XTB else labels.GFN1_XTB
    gfn2 = method is GFN2_XTB
    config = Config.create(
        method=method_label,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        mixer=labels.MIXER_ANDERSON if gfn2 else labels.MIXER_LINEAR,
        scp_mode=labels.SCP_MODE_FOCK if gfn2 else labels.SCP_MODE_CHARGE,
        maxiter=2,
        fermi_etemp=0.0,
        x_atol=0.0,
        x_atol_max=0.0,
        exclude=exclude,
    )
    params = (
        ParamModule(method, dtype=torch.float64)
        if parameters is None
        else parameters
    )
    model = Model(
        par=params,
        config=config,
        interaction=interactions,  # type: ignore[arg-type]
    )
    return model, torch.tensor(numbers, dtype=torch.int64)


def _water_positions(count: int = 3) -> torch.Tensor:
    """Return distinct, nondegenerate H2O conformers."""
    base = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]],
        dtype=torch.float64,
    )
    return torch.stack(
        [
            base
            + base.new_tensor(
                [[0.0, 0.0, 0.0], [delta, 0.0, 0.025], [-delta, 0.0, -0.025]]
            )
            for delta in torch.linspace(-0.035, 0.045, count)
        ]
    )


def _assert_result_matches_serial(mapped, serial: list[object]) -> None:
    """Compare public Result structure and every tensor leaf."""
    assert type(mapped) is type(serial[0])
    mapped_leaves, mapped_spec = _pytree.tree_flatten(mapped)
    flattened = [_pytree.tree_flatten(result) for result in serial]
    assert all(spec == mapped_spec for _, spec in flattened)
    assert all(len(leaves) == len(mapped_leaves) for leaves, _ in flattened)
    for index, actual in enumerate(mapped_leaves):
        assert isinstance(actual, torch.Tensor)
        values = [leaves[index] for leaves, _ in flattened]
        assert all(isinstance(value, torch.Tensor) for value in values)
        expected = torch.stack(values)
        assert actual.shape == (len(serial), *values[0].shape)
        assert actual.dtype == values[0].dtype
        assert actual.device == values[0].device
        torch.testing.assert_close(actual, expected)
    assert tuple(label for label, _ in mapped.classical) == tuple(
        label for label, _ in serial[0].classical
    )


def test_gfn1_public_vmap_preserves_classical_and_coulomb_terms() -> None:
    """GFN1 maps D3, repulsion, ES2, and ES3 as one production calculation."""
    model, numbers = _model()
    system = model.setup(numbers)
    positions = _water_positions()
    before = [
        leaf.detach().clone() for leaf in _pytree.tree_leaves(system.exact_core)
    ]
    parameter_before = {
        name: value.detach().clone()
        for name, value in model.par.named_parameters()
    }

    first = system.singlepoint(positions[0])
    second = system.singlepoint(positions[1])
    repeated = system.singlepoint(positions[0])
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_result_matches_serial(mapped, serial)
    assert [term.kind for term in system.exact_core.classical_terms] == [
        "halogen",
        "d3",
        "repulsion",
    ]
    assert [term.kind for term in system.exact_core.interaction_terms] == [
        "es2",
        "es3",
    ]
    classical = dict(first.classical)
    assert classical["Halogen"].count_nonzero() == 0
    assert classical["DispersionD3"].abs().sum() > 0
    assert classical["Repulsion"].abs().sum() > 0
    assert not torch.allclose(
        dict(first.classical)["DispersionD3"],
        dict(second.classical)["DispersionD3"],
    )
    assert not torch.allclose(
        first.energy,
        _model(exclude=("es2", "es3"))[0]
        .setup(numbers)
        .singlepoint(positions[0])
        .energy,
    )
    torch.testing.assert_close(first.energy, repeated.energy)
    torch.testing.assert_close(first.scf, repeated.scf)
    torch.testing.assert_close(first.density, repeated.density)
    for current, original in zip(
        _pytree.tree_leaves(system.exact_core), before
    ):
        torch.testing.assert_close(current, original)
    for name, value in model.par.named_parameters():
        torch.testing.assert_close(value, parameter_before[name])

    def energy(geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(geometry).energy.sum()

    mapped_gradient = torch.func.vmap(torch.func.grad(energy))(positions)
    serial_gradient = torch.stack(
        [torch.func.grad(energy)(p) for p in positions]
    )
    torch.testing.assert_close(mapped_gradient, serial_gradient)
    assert torch.isfinite(mapped_gradient).all()


def test_gfn2_public_vmap_preserves_aggregate_builtin_families() -> None:
    """GFN2 maps D4, repulsion, ES2, AES2, ES3, and D4SC together."""
    model, numbers = _model(GFN2_XTB)
    system = model.setup(numbers)
    positions = _water_positions(count=2)
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_result_matches_serial(mapped, serial)
    assert [term.kind for term in system.exact_core.classical_terms] == [
        "d4",
        "repulsion",
    ]
    assert [term.kind for term in system.exact_core.interaction_terms] == [
        "es2",
        "aes2",
        "es3",
        "d4sc",
    ]
    for _, contribution in mapped.classical:
        assert contribution.abs().sum() > 0
    assert mapped.charges is not None
    assert mapped.potential is not None
    assert mapped.density is not None
    assert mapped.status is not None
    assert mapped.status.trajectory_failure.shape == (2,)
    assert mapped.status.final_failure.shape == (2,)
    without_coulomb, _ = _model(
        GFN2_XTB, exclude=("es2", "aes2", "es3", "d4sc")
    )
    electronic_without_multipoles = without_coulomb.setup(numbers).singlepoint(
        positions[0]
    )
    assert not torch.allclose(
        serial[0].energy, electronic_without_multipoles.energy
    )


def test_gfn1_public_vmap_activates_halogen_energy() -> None:
    """A production Br2-ammonia fixture retains its active halogen term."""
    numbers = torch.tensor([35, 35, 7, 1, 1, 1], dtype=torch.int64)
    base = torch.tensor(
        [
            [0.0, 0.0, 3.114952513],
            [0.0, 0.0, -1.256718806],
            [0.0, 0.0, -6.302011301],
            [0.0, 1.787127097, -6.9747084],
            [-1.547696925, -0.893562604, -6.9747084],
            [1.547696925, -0.893562604, -6.9747084],
        ],
        dtype=torch.float64,
    )
    positions = torch.stack(
        [
            base
            + base.new_tensor(
                [
                    [0.01, -0.02, 0.01],
                    [0.04, 0.015, -0.025],
                    [0.01, 0.01, 0.005],
                    [0.01, -0.015, 0.005],
                    [-0.02, 0.005, -0.01],
                    [0.015, 0.01, 0.015],
                ]
            ),
            base
            + base.new_tensor(
                [
                    [-0.015, 0.012, -0.005],
                    [-0.025, 0.04, 0.02],
                    [-0.01, -0.01, -0.005],
                    [-0.02, 0.01, 0.013],
                    [0.008, -0.01, 0.02],
                    [-0.012, 0.005, -0.015],
                ]
            ),
        ]
    )
    model, _ = _model(numbers=tuple(numbers.tolist()))
    system = model.setup(numbers)

    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_result_matches_serial(mapped, serial)
    halogen = dict(mapped.classical)["Halogen"]
    assert torch.all(halogen.abs().sum(dim=-1) > 0)

    def energy(geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(geometry).energy.sum()

    mapped_gradient = torch.func.vmap(torch.func.grad(energy))(positions)
    serial_gradient = torch.stack(
        [torch.func.grad(energy)(p) for p in positions]
    )
    torch.testing.assert_close(mapped_gradient, serial_gradient)


def test_gfn2_aggregate_public_position_jacobian_is_lane_independent() -> None:
    """The full GFN2 aggregate path has matching diagonal and zero cross blocks."""
    model, numbers = _model(GFN2_XTB)
    system = model.setup(numbers)
    positions = _water_positions(count=2)

    def mapped_energy(batch: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(system.singlepoint, in_dims=(0,))(
            batch
        ).energy.sum(dim=-1)

    jacobian = torch.func.jacrev(mapped_energy)(positions)
    assert jacobian.shape == (2, 2, 3, 3)
    serial_gradients = torch.stack(
        [
            torch.func.grad(
                lambda geometry: system.singlepoint(geometry).energy.sum()
            )(geometry)
            for geometry in positions
        ]
    )
    for out_index in range(2):
        for in_index in range(2):
            if out_index == in_index:
                torch.testing.assert_close(
                    jacobian[out_index, in_index], serial_gradients[in_index]
                )
            else:
                torch.testing.assert_close(
                    jacobian[out_index, in_index],
                    torch.zeros_like(jacobian[out_index, in_index]),
                    atol=1.0e-12,
                    rtol=0.0,
                )


def test_gfn1_public_mapped_reverse_over_reverse_hessians_are_finite() -> None:
    """GFN1 D3/ES2/ES3 public mapping supports a second reverse derivative."""
    model, numbers = _model()
    system = model.setup(numbers)
    positions = _water_positions(count=2)

    def energy(flat_geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(flat_geometry.reshape(3, 3)).energy.sum()

    hessians = torch.func.vmap(torch.func.jacrev(torch.func.grad(energy)))(
        positions.reshape(2, -1)
    )
    assert hessians.shape == (2, 9, 9)
    assert torch.isfinite(hessians).all()
    torch.testing.assert_close(
        hessians, hessians.transpose(-1, -2), atol=1.0e-10, rtol=1.0e-10
    )


def test_mapped_classical_and_es2_parameters_keep_model_gradients() -> None:
    """Mapped aggregate gradients reach repulsion and ES2 Model leaves."""
    parameters = ParamModule(GFN1_XTB, dtype=torch.float64)
    parameters.set_differentiable("element.H.arep")
    parameters.set_differentiable("charge.effective.gexp")
    model, numbers = _model(parameters=parameters)
    system = model.setup(numbers)
    positions = _water_positions(count=2)
    arep = parameters.get("element.H.arep", unwrapped=False).param
    gexp = parameters.get("charge.effective.gexp", unwrapped=False).param
    assert arep is not None and gexp is not None
    values_before = (arep.detach().clone(), gexp.detach().clone())

    mapped_energy = torch.func.vmap(system.singlepoint, in_dims=(0,))(
        positions
    ).energy.sum()
    mapped_gradients = torch.autograd.grad(
        mapped_energy, (arep, gexp), retain_graph=True
    )
    serial_energy = torch.stack(
        [system.singlepoint(position).energy.sum() for position in positions]
    ).sum()
    serial_gradients = torch.autograd.grad(serial_energy, (arep, gexp))

    for mapped_gradient, serial_gradient in zip(
        mapped_gradients, serial_gradients
    ):
        torch.testing.assert_close(mapped_gradient, serial_gradient)
        assert torch.isfinite(mapped_gradient).all()
        assert mapped_gradient.abs().sum() > 0
    torch.testing.assert_close(arep, values_before[0])
    torch.testing.assert_close(gexp, values_before[1])
    assert not hasattr(system, "model")


def test_gfn1_alpb_public_vmap_matches_serial_and_position_gradients() -> None:
    """Exact GeneralizedBorn setup data survives mapped public evaluation."""
    numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
    solvation = GeneralizedBorn(numbers, 78.9, dtype=torch.float64)
    model, _ = _model(interactions=(solvation,))
    system = model.setup(numbers)
    positions = _water_positions(count=2)
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_result_matches_serial(mapped, serial)
    assert [term.kind for term in system.exact_core.interaction_terms] == [
        "es2",
        "es3",
        "generalized_born",
    ]
    assert system.generalized_born_setup is not None
    without_solvation, _ = _model()
    no_solvation_result = without_solvation.setup(numbers).singlepoint(
        positions[0]
    )
    assert not torch.allclose(serial[0].energy, no_solvation_result.energy)

    def energy(geometry: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(geometry).energy.sum()

    mapped_gradient = torch.func.vmap(torch.func.grad(energy))(positions)
    serial_gradient = torch.stack(
        [torch.func.grad(energy)(p) for p in positions]
    )
    torch.testing.assert_close(mapped_gradient, serial_gradient)
    assert torch.isfinite(mapped_gradient).all()
