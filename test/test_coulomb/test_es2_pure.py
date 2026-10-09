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
"""Pure setup and transform coverage for ES2 Coulomb matrices."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.calculators.model import Model
from dxtb._src.components.interactions.coulomb import (
    ES2Setup,
    build_es2_coulomb,
    new_es2,
    setup_es2,
)
from dxtb._src.components.interactions.coulomb.secondorder import (
    ES2,
    ES2Cache,
)
from dxtb._src.typing import Tensor

from ..molecules import mols


def _system_inputs(dtype: torch.dtype = torch.float64) -> tuple[Tensor, Tensor]:
    numbers = mols["H2O"]["numbers"].clone()
    positions = mols["H2O"]["positions"].to(dtype=dtype)
    return numbers, positions


def _es2_setup(
    shell_resolved: bool,
    method: str = "gfn1",
) -> tuple[Tensor, Tensor, IndexHelper, ES2, ES2Setup]:
    numbers, positions = _system_inputs()
    parameters = ParamModule(
        GFN1_XTB if method == "gfn1" else GFN2_XTB,
        dtype=positions.dtype,
    )
    ihelp = IndexHelper.from_numbers(numbers, parameters)
    interaction = new_es2(
        torch.unique(numbers),
        parameters,
        shell_resolved=shell_resolved,
        dtype=positions.dtype,
    )
    assert interaction is not None
    setup = setup_es2(
        numbers,
        interaction.hubbard,
        ihelp,
        lhubbard=interaction.lhubbard,
        gexp=interaction.gexp,
        average=interaction.average,
        shell_resolved=interaction.shell_resolved,
    )
    return numbers, positions, ihelp, interaction, setup


@pytest.mark.parametrize("shell_resolved", [False, True])
def test_setup_builder_history_and_named_resolution(
    shell_resolved: bool,
) -> None:
    """The pure builder is history-independent and dispatches by setup."""
    numbers, positions, ihelp, interaction, setup = _es2_setup(shell_resolved)

    matrix1 = build_es2_coulomb(setup, positions)
    moved = positions.clone()
    moved[1, 2] += 0.07
    matrix2 = build_es2_coulomb(setup, moved)
    matrix3 = build_es2_coulomb(setup, positions.clone())
    torch.testing.assert_close(matrix1, matrix3)
    assert not torch.allclose(matrix1, matrix2)

    differentiated_positions = positions.clone().requires_grad_(True)
    differentiated = build_es2_coulomb(setup, differentiated_positions)
    torch.autograd.grad(differentiated.square().sum(), differentiated_positions)
    matrix4 = build_es2_coulomb(setup, positions.clone())
    torch.testing.assert_close(matrix1, matrix4)

    legacy_cache = interaction.get_cache(
        numbers=numbers, positions=positions, ihelp=ihelp
    )
    assert isinstance(legacy_cache, ES2Cache)
    torch.testing.assert_close(matrix1, legacy_cache.mat)

    # The method name fixes its resolution even when the interaction object's
    # configured resolution differs.
    atom = interaction.get_atom_coulomb_matrix(numbers, positions, ihelp)
    assert atom.shape == (numbers.numel(), numbers.numel())
    if interaction.lhubbard is not None:
        shell = interaction.get_shell_coulomb_matrix(numbers, positions, ihelp)
        assert shell.shape == (ihelp.shells_to_atom.shape[-1],) * 2

    charges = torch.linspace(
        -0.2, 0.3, matrix1.shape[-1], dtype=positions.dtype
    )
    if shell_resolved:
        potential = interaction.get_monopole_shell_potential(
            legacy_cache, charges
        )
        energy = interaction.get_monopole_shell_energy(legacy_cache, charges)
    else:
        potential = interaction.get_monopole_atom_potential(
            legacy_cache, charges
        )
        energy = interaction.get_monopole_atom_energy(legacy_cache, charges)
    torch.testing.assert_close(potential, matrix1 @ charges)
    torch.testing.assert_close(energy, 0.5 * charges * potential)


@pytest.mark.parametrize("shell_resolved", [False, True])
def test_setup_builder_reverse_and_forward_transforms(
    shell_resolved: bool,
) -> None:
    """Plain ES2 construction supports reverse and forward transforms."""
    _, positions, _, _, setup = _es2_setup(shell_resolved)
    weights = torch.linspace(
        0.2,
        1.1,
        (
            setup.atom_mask.shape[-1] ** 2
            if not shell_resolved
            else setup.shell_mask.shape[-1] ** 2
        ),
        dtype=positions.dtype,
    ).reshape(build_es2_coulomb(setup, positions).shape)

    def scalar(pos: Tensor) -> Tensor:
        return (build_es2_coulomb(setup, pos) * weights).sum()

    position_leaf = positions.clone().requires_grad_(True)
    grad1 = torch.autograd.grad(
        scalar(position_leaf), position_leaf, create_graph=True
    )[0]
    grad2 = torch.autograd.grad(grad1.sum(), position_leaf, create_graph=True)[
        0
    ]
    grad3 = torch.autograd.grad(grad2.sum(), position_leaf)[0]
    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()
    assert torch.isfinite(grad3).all()

    direction = torch.linspace(
        -0.2, 0.3, positions.numel(), dtype=positions.dtype
    ).reshape_as(positions)
    primal, tangent = jvp(scalar, (positions,), (direction,))
    reverse = jacrev(scalar)(positions)
    forward = jacfwd(scalar)(positions)
    torch.testing.assert_close(primal, scalar(positions))
    torch.testing.assert_close(tangent, (reverse * direction).sum())
    torch.testing.assert_close(forward, reverse)

    moved = positions.clone()
    moved[1, 2] += 0.05
    conformers = torch.stack((positions, moved))
    mapped = vmap(lambda pos: build_es2_coulomb(setup, pos))(conformers)
    looped = torch.stack(
        tuple(build_es2_coulomb(setup, pos) for pos in conformers)
    )
    torch.testing.assert_close(mapped, looped)
    assert not torch.allclose(mapped[0], mapped[1])


@pytest.mark.parametrize(
    ("method", "shell_resolved", "parameter_suffix"),
    [
        ("gfn1", False, "parameter_tree.element.O.gam.param"),
        ("gfn1", True, "parameter_tree.element.O.lgam.param"),
        ("gfn1", True, "parameter_tree.charge.effective.gexp.param"),
        ("gfn2", True, "parameter_tree.element.O.lgam.param"),
        ("gfn2", True, "parameter_tree.charge.effective.gexp.param"),
    ],
)
def test_setup_builder_parameter_gradients(
    method: str, shell_resolved: bool, parameter_suffix: str
) -> None:
    """Gathered ES2 parameters remain connected to ParamModule leaves."""
    numbers, positions = _system_inputs()
    par = ParamModule(
        GFN1_XTB if method == "gfn1" else GFN2_XTB, dtype=positions.dtype
    )
    parameters = dict(par.named_parameters())
    parameter = parameters[parameter_suffix]
    parameter.requires_grad_(True)
    ihelp = IndexHelper.from_numbers(numbers, par)

    def evaluate() -> tuple[Tensor, Tensor]:
        interaction = new_es2(
            torch.unique(numbers),
            par,
            shell_resolved=shell_resolved,
            dtype=positions.dtype,
        )
        assert interaction is not None
        setup = setup_es2(
            numbers,
            interaction.hubbard,
            ihelp,
            lhubbard=interaction.lhubbard,
            gexp=interaction.gexp,
            average=interaction.average,
            shell_resolved=interaction.shell_resolved,
        )
        position_leaf = positions.clone().requires_grad_(True)
        loss = build_es2_coulomb(setup, position_leaf).square().sum()
        return loss, position_leaf

    loss1, pos1 = evaluate()
    grad1, position_grad = torch.autograd.grad(loss1, (parameter, pos1))
    loss2, pos2 = evaluate()
    grad2, _ = torch.autograd.grad(loss2, (parameter, pos2))
    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()
    assert torch.isfinite(position_grad).all()
    assert torch.count_nonzero(grad1) > 0
    torch.testing.assert_close(grad1, grad2)


def test_setup_builder_mixed_position_parameter_derivative() -> None:
    """Position and ES2 parameter derivatives compose naturally."""
    numbers, positions = _system_inputs()
    par = ParamModule(GFN1_XTB, dtype=positions.dtype)
    parameter = dict(par.named_parameters())[
        "parameter_tree.element.O.gam.param"
    ]
    parameter.requires_grad_(True)
    ihelp = IndexHelper.from_numbers(numbers, par)
    interaction = new_es2(torch.unique(numbers), par, shell_resolved=False)
    assert interaction is not None

    def scalar(pos: Tensor) -> Tensor:
        setup = setup_es2(
            numbers,
            interaction.hubbard,
            ihelp,
            gexp=interaction.gexp,
            average=interaction.average,
            shell_resolved=False,
        )
        return build_es2_coulomb(setup, pos).square().sum()

    position_leaf = positions.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(
        scalar(position_leaf), position_leaf, create_graph=True
    )
    direction = torch.arange(
        positions.numel(), dtype=positions.dtype
    ).reshape_as(positions)
    (mixed,) = torch.autograd.grad((gradient * direction).sum(), parameter)
    assert torch.isfinite(mixed).all()
    assert torch.count_nonzero(mixed) > 0


def test_setup_has_only_gathered_values_and_core_bypasses_es2_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """System retains narrow ES2 data and core never asks ES2 for a cache."""
    numbers, positions, _, _, setup = _es2_setup(shell_resolved=True)
    forbidden = ("Param", "ParamModule", "Model", "ES2", "Cache", "positions")
    assert isinstance(setup, ES2Setup)
    assert all(
        not any(token in field.name for token in forbidden)
        for field in fields(setup)
    )
    assert setup.shells_to_atom is not None
    assert setup.shell_hubbard is not None
    assert torch.unique(setup.shell_hubbard).numel() > 1

    system = Model(par=ParamModule(GFN1_XTB, dtype=positions.dtype)).setup(
        numbers
    )
    assert system.es2_setup is not None

    def fail_cache(*_: object, **__: object) -> ES2Cache:
        raise AssertionError("single-system core called ES2.get_cache")

    monkeypatch.setattr(ES2, "get_cache", fail_cache)
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()
