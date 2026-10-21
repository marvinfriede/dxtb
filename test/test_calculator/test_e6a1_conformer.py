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
"""Public conformer acceptance tests for E6a.1."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.wavefunction.filling import get_alpha_beta_occupation
from dxtb.config import Config


def _system(
    method: object,
    numbers: tuple[int, ...],
    *,
    maxiter: int = 2,
    temperature: float = 0.0,
    intlevel: int = labels.INTLEVEL_HCORE,
    mixer: int = labels.MIXER_LINEAR,
    scp_mode: int = labels.SCP_MODE_CHARGE,
    par: ParamModule | None = None,
):
    """Build an independent production PyTorch System."""
    dtype = torch.float64
    config = Config.create(
        method=(labels.GFN1_XTB if method is GFN1_XTB else labels.GFN2_XTB),
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=intlevel,
        mixer=mixer,
        scp_mode=scp_mode,
        scf_mode=labels.SCF_MODE_FULL,
        maxiter=maxiter,
        force_convergence=False,
        fermi_etemp=temperature,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    parameters = ParamModule(method, dtype=dtype) if par is None else par
    return Model(par=parameters, config=config).setup(
        torch.tensor(numbers, dtype=torch.int64)
    )


def _geometries(numbers: tuple[int, ...], count: int = 3) -> torch.Tensor:
    """Return distinct valid geometries for the requested fixed composition."""
    if numbers == (1, 1):
        base = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]], dtype=torch.float64
        )
        return torch.stack(
            [
                base + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, shift]])
                for shift in torch.linspace(-0.08, 0.12, count)
            ]
        )

    assert numbers == (8, 1, 1)
    base = torch.tensor(
        [[0.0, 0.0, -0.72], [-1.43, 0.0, 0.36], [1.43, 0.0, 0.36]],
        dtype=torch.float64,
    )
    variants = []
    for index, angle in enumerate(torch.linspace(-0.06, 0.08, count)):
        moved = base.clone()
        moved[1, 0] += angle
        moved[2, 1] -= angle * 0.7
        moved[1, 2] += index * 0.025
        variants.append(moved)
    return torch.stack(variants)


def _assert_result_batch_matches_serial(
    mapped, serial: list[object], batch_size: int
) -> None:
    """Check the registered Result tree and every batched tensor leaf."""
    assert type(mapped) is type(serial[0])
    mapped_leaves, mapped_spec = _pytree.tree_flatten(mapped)
    serial_flattened = [_pytree.tree_flatten(result) for result in serial]
    assert all(spec == mapped_spec for _, spec in serial_flattened)
    assert all(
        len(leaves) == len(mapped_leaves) for leaves, _ in serial_flattened
    )
    for index, actual in enumerate(mapped_leaves):
        assert isinstance(actual, torch.Tensor)
        values = [leaves[index] for leaves, _ in serial_flattened]
        assert all(isinstance(value, torch.Tensor) for value in values)
        expected = torch.stack(values)
        assert actual.shape == (batch_size, *values[0].shape)
        assert actual.dtype == values[0].dtype
        assert actual.device == values[0].device
        torch.testing.assert_close(actual, expected)
    assert tuple(label for label, _ in mapped.classical) == tuple(
        label for label, _ in serial[0].classical
    )


@pytest.mark.parametrize(
    ("method", "numbers", "mixer", "scp_mode", "batch_size"),
    [
        (
            GFN1_XTB,
            (8, 1, 1),
            labels.MIXER_LINEAR,
            labels.SCP_MODE_CHARGE,
            2,
        ),
        (GFN2_XTB, (1, 1), labels.MIXER_ANDERSON, labels.SCP_MODE_FOCK, 3),
        (
            GFN2_XTB,
            (8, 1, 1),
            labels.MIXER_ANDERSON,
            labels.SCP_MODE_FOCK,
            2,
        ),
    ],
    ids=["gfn1-h2o", "gfn2-h2", "gfn2-h2o"],
)
def test_public_system_conformer_vmap_result_matches_serial(
    method: object,
    numbers: tuple[int, ...],
    mixer: int,
    scp_mode: int,
    batch_size: int,
) -> None:
    """GFN1/GFN2 public calls map complete Result trees over conformers."""
    system = _system(method, numbers, mixer=mixer, scp_mode=scp_mode)
    positions = _geometries(numbers, count=batch_size)
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_result_batch_matches_serial(mapped, serial, len(positions))
    assert mapped.iterations.shape == (len(positions),)
    assert mapped.converged is not None
    assert mapped.status is not None
    assert mapped.status.trajectory_failure.shape == (len(positions),)
    assert mapped.status.final_failure.shape == (len(positions),)
    assert mapped.overlap is not None
    assert mapped.hcore is not None


def test_public_system_vmap_maps_explicit_charge_and_spin() -> None:
    """Valid per-conformer charge and spin inputs remain transform-safe."""
    system = _system(GFN1_XTB, (1, 1))
    positions = _geometries((1, 1))
    charges = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
    spins = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    mapped = torch.func.vmap(
        lambda position, charge, spin: system.singlepoint(
            position, chrg=charge, spin=spin
        ),
        in_dims=(0, 0, 0),
    )(positions, charges, spins)
    serial = [
        system.singlepoint(position, chrg=charge, spin=spin)
        for position, charge, spin in zip(positions, charges, spins)
    ]
    _assert_result_batch_matches_serial(mapped, serial, len(positions))

    shared_charge = torch.tensor(0.0, dtype=torch.float64)
    shared_spin = torch.tensor(0.0, dtype=torch.float64)
    shared = torch.func.vmap(
        lambda position, charge, spin: system.singlepoint(
            position, chrg=charge, spin=spin
        ),
        in_dims=(0, None, None),
    )(positions, shared_charge, shared_spin)
    shared_serial = [
        system.singlepoint(position, chrg=shared_charge, spin=shared_spin)
        for position in positions
    ]
    _assert_result_batch_matches_serial(shared, shared_serial, len(positions))


def test_public_system_vmap_captures_shared_electric_fields() -> None:
    """Shared electric fields remain constants across conformer mapping."""
    system = _system(
        GFN1_XTB,
        (1, 1),
        intlevel=labels.INTLEVEL_QUADRUPOLE,
    )
    positions = _geometries((1, 1), count=2)
    field = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64)
    field_grad = 0.01 * torch.eye(3, dtype=torch.float64)

    def evaluate(position: torch.Tensor):
        return system.singlepoint(position, field=field, field_grad=field_grad)

    mapped = torch.func.vmap(evaluate, in_dims=(0,))(positions)
    serial = [evaluate(position) for position in positions]

    _assert_result_batch_matches_serial(mapped, serial, len(positions))
    assert mapped.dipole_integrals is not None
    assert mapped.quadrupole_integrals is not None


def test_public_system_vmap_cross_conformer_position_jacobian() -> None:
    """The public mapped energy has independent conformer derivative blocks."""
    system = _system(GFN1_XTB, (1, 1))
    positions = _geometries((1, 1))

    def mapped_energies(batch: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(system.singlepoint, in_dims=(0,))(
            batch
        ).energy.sum(dim=-1)

    jacobian = torch.func.jacrev(mapped_energies)(positions)
    assert jacobian.shape == (3, 3, 2, 3)
    serial_gradients = torch.stack(
        [
            torch.func.grad(
                lambda value: system.singlepoint(value).energy.sum()
            )(position)
            for position in positions
        ]
    )
    for output_index in range(3):
        for input_index in range(3):
            if output_index == input_index:
                torch.testing.assert_close(
                    jacobian[output_index, input_index],
                    serial_gradients[input_index],
                )
            else:
                torch.testing.assert_close(
                    jacobian[output_index, input_index],
                    torch.zeros_like(jacobian[output_index, input_index]),
                    atol=1.0e-12,
                    rtol=0.0,
                )


def test_public_system_vmap_reports_outer_nonconvergence_as_status() -> None:
    """An insufficient positive fixed-trip budget returns status, not an error."""
    system = _system(GFN1_XTB, (8, 1, 1), maxiter=1)
    positions = _geometries((8, 1, 1), count=2)
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    serial = [system.singlepoint(position) for position in positions]

    _assert_result_batch_matches_serial(mapped, serial, len(positions))
    assert mapped.converged is not None
    assert not mapped.converged.any()
    assert mapped.iterations.shape == (2,)
    assert mapped.residual is not None and mapped.residual.shape[0] == 2
    assert mapped.status is not None
    assert mapped.status.trajectory_failure.shape == (2,)
    assert mapped.status.final_failure.shape == (2,)


def test_gfn2_anderson_public_vmap_position_gradients_match_serial() -> None:
    """GFN2 Anderson mapped reverse gradients match independent evaluations."""
    system = _system(
        GFN2_XTB,
        (1, 1),
        mixer=labels.MIXER_ANDERSON,
        scp_mode=labels.SCP_MODE_FOCK,
    )
    positions = _geometries((1, 1), count=2)

    def energy(position: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(position).energy.sum()

    mapped = torch.func.vmap(torch.func.grad(energy))(positions)
    serial = torch.stack(
        [torch.func.grad(energy)(value) for value in positions]
    )
    torch.testing.assert_close(mapped, serial)
    assert torch.isfinite(mapped).all()


def test_mapped_total_energy_preserves_repulsion_parameter_gradient() -> None:
    """Mapped public energies retain gradients to Model-owned parameters."""
    parameters = ParamModule(GFN1_XTB, dtype=torch.float64)
    parameters.set_differentiable("element.H.arep")
    system = _system(GFN1_XTB, (1, 1), par=parameters)
    parameter = parameters.get("element.H.arep", unwrapped=False).param
    positions = _geometries((1, 1))

    mapped_energy = torch.func.vmap(system.singlepoint, in_dims=(0,))(
        positions
    ).energy.sum()
    (mapped_gradient,) = torch.autograd.grad(
        mapped_energy, parameter, retain_graph=True
    )
    serial_energy = torch.stack(
        [system.singlepoint(position).energy.sum() for position in positions]
    ).sum()
    (serial_gradient,) = torch.autograd.grad(serial_energy, parameter)

    torch.testing.assert_close(mapped_gradient, serial_gradient)
    assert parameters.get("element.H.arep", unwrapped=False).param is parameter


def test_eager_spin_validation_remains_available() -> None:
    """Ordinary occupation construction retains its eager consistency checks."""
    with pytest.raises(ValueError, match="unpaired electrons"):
        get_alpha_beta_occupation(torch.tensor([2.0]), torch.tensor([1.0]))
