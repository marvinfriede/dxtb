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
"""Architecture tests for grouped-vmap heterogeneous batching.

Packed batches are split into ordinary single-System lanes. Lanes that share
one System run under ``torch.func.vmap``; every other lane is evaluated
independently. Results must equal serial single-System evaluation, keep the
source order, and keep derivatives lane-local.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule, labels, scf
from dxtb._src.calculators import lane_evaluation
from dxtb._src.calculators.lane_evaluation import (
    can_vmap_system,
    evaluate_systems_independently,
    group_lanes,
)
from dxtb._src.calculators.model import Model, System
from dxtb._src.calculators.packed_result import (
    build_calculator_lane_systems,
    evaluate_packed_systems,
)
from dxtb._src.calculators.result import Result
from dxtb.config import Config

DTYPE = torch.float64
DEVICE = torch.device("cpu")
TOL = {"atol": 1.0e-9, "rtol": 1.0e-9}

_SETTINGS = {
    "int_driver": labels.INTDRIVER_PYTORCH,
    "int_level": labels.INTLEVEL_QUADRUPOLE,
    "maxiter": 60,
    "fermi_etemp": 300.0,
}

_COORDINATES = {
    (1, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]],
    (8, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]],
    (8, 1, 1): [[0.0, 0.0, 0.0], [0.82, 0.0, 0.61], [-0.82, 0.0, 0.61]],
    (7, 1, 1, 1): [
        [0.0, 0.0, 0.0],
        [0.0, 1.77, 0.72],
        [1.53, -0.88, 0.72],
        [-1.53, -0.88, 0.72],
    ],
    (6, 1, 1, 1, 1): [
        [0.0, 0.0, 0.0],
        [1.19, 1.19, 1.19],
        [1.19, -1.19, -1.19],
        [-1.19, 1.19, -1.19],
        [-1.19, -1.19, 1.19],
    ],
}

H2 = (1, 1)
OH = (8, 1)
H2O = (8, 1, 1)
NH3 = (7, 1, 1, 1)
CH4 = (6, 1, 1, 1, 1)


def _geometry(numbers: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """Return a slightly distorted fixture so conformers differ."""
    base = torch.tensor(_COORDINATES[numbers], dtype=DTYPE)
    generator = torch.Generator().manual_seed(seed)
    return base + 0.03 * torch.randn(
        base.shape, generator=generator, dtype=DTYPE
    )


def _model(
    method: int = labels.GFN1_XTB, par: ParamModule | None = None, **settings
) -> Model:
    if par is None:
        par = ParamModule(
            GFN1_XTB if method == labels.GFN1_XTB else GFN2_XTB,
            device=DEVICE,
            dtype=DTYPE,
        )
    config = Config.create(method=method, **(_SETTINGS | settings))
    return Model(par=par, config=config, auto_int_level=False)


def _setup(model: Model, numbers: tuple[int, ...]) -> System:
    return model.setup(torch.tensor(numbers, dtype=torch.int64))


def _calculator(
    compositions: Sequence[tuple[int, ...]],
    par: ParamModule | None = None,
    **settings,
) -> Calculator:
    """Build a public batched Calculator with trailing zero padding."""
    max_atoms = max(map(len, compositions))
    numbers = torch.zeros((len(compositions), max_atoms), dtype=torch.int64)
    for lane, composition in enumerate(compositions):
        numbers[lane, : len(composition)] = torch.tensor(composition)
    opts = dict(_SETTINGS | settings, method=labels.GFN1_XTB, verbosity=0)
    return Calculator(
        numbers,
        GFN1_XTB if par is None else par,
        opts=opts,
        dtype=DTYPE,
        auto_int_level=False,
    )


def _packed_positions(
    compositions: Sequence[tuple[int, ...]], seeds: Sequence[int]
) -> torch.Tensor:
    max_atoms = max(map(len, compositions))
    positions = torch.zeros((len(compositions), max_atoms, 3), dtype=DTYPE)
    for lane, (composition, seed) in enumerate(zip(compositions, seeds)):
        positions[lane, : len(composition)] = _geometry(composition, seed)
    return positions


def _group_layout(groups) -> list[tuple[tuple[int, ...], bool]]:
    return [(group.source_ids, group.use_vmap) for group in groups]


def _assert_lane_matches(
    packed: Result, lane: int, reference: Result, nat: int, nao: int
) -> None:
    """Compare every populated field of one packed lane with a reference."""
    torch.testing.assert_close(
        packed.energy[lane, :nat], reference.energy, **TOL
    )
    torch.testing.assert_close(packed.scf[lane, :nat], reference.scf, **TOL)
    torch.testing.assert_close(
        packed.fenergy[lane, :nat], reference.fenergy, **TOL
    )
    assert torch.count_nonzero(packed.energy[lane, nat:]) == 0
    packed_classical = dict(packed.classical)
    for label, value in reference.classical:
        torch.testing.assert_close(
            packed_classical[label][lane, :nat], value, **TOL
        )

    for name in ("density", "hamiltonian", "overlap", "hcore"):
        torch.testing.assert_close(
            getattr(packed, name)[lane, :nao, :nao],
            getattr(reference, name),
            **TOL,
        )
    for name in ("dipole_integrals", "quadrupole_integrals"):
        torch.testing.assert_close(
            getattr(packed, name)[lane, :, :nao, :nao],
            getattr(reference, name),
            **TOL,
        )
    for name in ("emo", "overlap_norm"):
        torch.testing.assert_close(
            getattr(packed, name)[lane, :nao], getattr(reference, name), **TOL
        )
    torch.testing.assert_close(
        packed.occupation[lane, ..., :nao], reference.occupation, **TOL
    )
    torch.testing.assert_close(
        packed.coefficients[lane, :nao, :nao].abs(),
        reference.coefficients.abs(),
        **TOL,
    )

    assert packed.charges is not None and reference.charges is not None
    assert packed.potential is not None and reference.potential is not None
    torch.testing.assert_close(
        packed.charges.mono[lane, :nao], reference.charges.mono, **TOL
    )
    torch.testing.assert_close(
        packed.potential.mono[lane, :nao], reference.potential.mono, **TOL
    )
    for name in ("dipole", "quadrupole"):
        for packed_value, value in (
            (getattr(packed.charges, name), getattr(reference.charges, name)),
            (
                getattr(packed.potential, name),
                getattr(reference.potential, name),
            ),
        ):
            if value is None:
                continue
            torch.testing.assert_close(packed_value[lane, :nat], value, **TOL)

    assert int(packed.iterations[lane]) == int(reference.iterations)
    assert bool(packed.converged[lane]) == bool(reference.converged)
    torch.testing.assert_close(
        packed.residual[lane].reshape(reference.residual.shape),
        reference.residual,
        **TOL,
    )
    assert packed.status is not None and reference.status is not None
    assert int(packed.status.trajectory_failure[lane]) == int(
        reference.status.trajectory_failure
    )
    assert int(packed.status.final_failure[lane]) == int(
        reference.status.final_failure
    )


# Grouping and lane-System deduplication


def test_repeated_compositions_share_one_setup_per_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five lanes with two compositions need exactly two Model.setup calls."""
    calls: list[tuple[int, ...]] = []
    original_setup = Model.setup

    def counting_setup(self, numbers, *args, **kwargs):
        calls.append(tuple(int(z) for z in numbers))
        return original_setup(self, numbers, *args, **kwargs)

    compositions = (H2O, CH4, H2O, H2O, CH4)
    calc = _calculator(compositions)
    monkeypatch.setattr(Model, "setup", counting_setup)
    systems = build_calculator_lane_systems(calc)

    assert sorted(calls) == sorted([H2O, CH4])
    assert systems[0] is systems[2] is systems[3]
    assert systems[1] is systems[4]
    assert systems[0] is not systems[1]


def test_group_lanes_keeps_heterogeneous_source_order() -> None:
    """Shared Systems form vmap groups; a unique lane stays a singleton."""
    compositions = (H2O, CH4, H2O, NH3, CH4, H2O)
    systems = build_calculator_lane_systems(_calculator(compositions))
    none = [None] * len(systems)

    groups = group_lanes(systems, none, none, none)

    assert _group_layout(groups) == [
        ((0, 2, 5), True),
        ((1, 4), True),
        ((3,), False),
    ]


def test_group_lanes_separates_optional_call_modes() -> None:
    """spin/field presence changes the mapped call and splits groups."""
    system = _setup(_model(), H2O)
    systems = (system,) * 4
    spin = torch.tensor(0.0, dtype=DTYPE)
    field = torch.zeros(3, dtype=DTYPE)

    groups = group_lanes(
        systems,
        [None, spin, None, spin],
        [None, None, field, None],
        [None, None, None, None],
    )

    assert _group_layout(groups) == [
        ((0,), False),
        ((1, 3), True),
        ((2,), False),
    ]


def test_distinct_but_equivalent_systems_are_not_merged() -> None:
    """Grouping uses System identity, never structural comparison."""
    model = _model()
    first, second = _setup(model, H2O), _setup(model, H2O)
    none = [None, None]

    groups = group_lanes((first, second), none, none, none)

    assert _group_layout(groups) == [((0,), False), ((1,), False)]


# Numerical parity with serial single-System evaluation


def test_conformer_batch_is_one_vmap_group_matching_serial() -> None:
    """batch_mode=2 conformers run as one mapped group with serial parity."""
    compositions = (H2O, H2O, H2O)
    calc = _calculator(compositions)
    systems = build_calculator_lane_systems(calc)
    none = [None] * 3
    assert _group_layout(group_lanes(systems, none, none, none)) == [
        ((0, 1, 2), True)
    ]

    positions = _packed_positions(compositions, (0, 1, 2))
    positions.requires_grad_(True)
    packed = calc.singlepoint(positions)
    (forces,) = torch.autograd.grad(packed.energy.sum(), positions)

    for lane in range(3):
        reference_pos = positions[lane].detach().clone().requires_grad_(True)
        reference = systems[0].singlepoint(reference_pos)
        (reference_grad,) = torch.autograd.grad(
            reference.energy.sum(), reference_pos
        )
        torch.testing.assert_close(packed.energy[lane], reference.energy, **TOL)
        torch.testing.assert_close(forces[lane], reference_grad, **TOL)


def test_repeated_compositions_full_result_parity_in_source_order() -> None:
    """Every Result field of grouped and singleton lanes equals serial."""
    compositions = (H2O, CH4, H2O, NH3, CH4, H2O)
    systems = build_calculator_lane_systems(_calculator(compositions))
    positions = tuple(
        _geometry(composition, seed)
        for seed, composition in enumerate(compositions)
    )

    packed = evaluate_packed_systems(systems, positions)

    for lane, (system, position) in enumerate(zip(systems, positions)):
        reference = system.singlepoint(position)
        _assert_lane_matches(
            packed,
            lane,
            reference,
            nat=position.shape[0],
            nao=system.ihelp.nao,
        )


def test_fully_heterogeneous_batch_equals_independent_evaluation() -> None:
    """Without repeated Systems the batch degenerates to independent lanes."""
    compositions = (H2, H2O, CH4, NH3)
    systems = build_calculator_lane_systems(_calculator(compositions))
    none = [None] * 4
    groups = group_lanes(systems, none, none, none)
    assert all(not group.use_vmap for group in groups)

    positions = tuple(_geometry(c) for c in compositions)
    packed = evaluate_packed_systems(systems, positions, legacy_compat=False)
    independent = evaluate_systems_independently(systems, positions)

    for lane, (system, reference) in enumerate(zip(systems, independent)):
        nat = positions[lane].shape[0]
        nao = system.ihelp.nao
        torch.testing.assert_close(
            packed.energy[lane, :nat], reference.energy, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            packed.density[lane, :nao, :nao],
            reference.density,
            atol=0.0,
            rtol=0.0,
        )


def test_mapped_group_matches_independent_eager_evaluation() -> None:
    """Fixed-trip vmap and eager independent evaluation agree per lane."""
    system = _setup(_model(), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in range(3))
    none = (None,) * 3
    group = group_lanes((system,) * 3, none, none, none)[0]
    assert group.use_vmap

    mapped = lane_evaluation.evaluate_lane_group(
        group, positions, charges, none, none, none
    )
    eager = evaluate_systems_independently((system,) * 3, positions, charges)

    for lane in range(3):
        nao = system.ihelp.nao
        _assert_lane_matches(
            _stack_single(mapped[lane]), 0, eager[lane], 3, nao
        )


def _stack_single(result: Result) -> Result:
    """Add a leading lane axis to a single Result for packed-style checks."""
    from torch.utils._pytree import tree_map

    return tree_map(lambda leaf: leaf.unsqueeze(0), result)


def test_public_padded_mode1_trailing_and_noncontiguous_padding() -> None:
    """Zero padding in public input resolves to real lanes before physics."""
    calc = _calculator((H2O, H2, H2O))
    numbers = calc.numbers.clone()
    # Move lane 1's padding between its two real atoms.
    numbers[1] = torch.tensor([1, 0, 1])
    noncontiguous = Calculator(
        numbers,
        GFN1_XTB,
        opts=dict(_SETTINGS, method=labels.GFN1_XTB, verbosity=0),
        dtype=DTYPE,
        auto_int_level=False,
    )
    positions = torch.zeros((3, 3, 3), dtype=DTYPE)
    positions[0] = _geometry(H2O, 0)
    positions[2] = _geometry(H2O, 1)
    h2 = _geometry(H2)
    positions[1, 0] = h2[0]
    positions[1, 2] = h2[1]
    positions[1, 1] = torch.tensor([9.0, 9.0, 9.0], dtype=DTYPE)

    packed = noncontiguous.singlepoint(positions)

    systems = build_calculator_lane_systems(noncontiguous)
    assert systems[0] is systems[2]
    assert systems[1].numbers.tolist() == [1, 1]
    reference_h2 = systems[1].singlepoint(h2)
    torch.testing.assert_close(
        packed.energy[1, [0, 2]], reference_h2.energy, **TOL
    )
    assert packed.energy[1, 1] == 0.0
    for lane in (0, 2):
        reference = systems[0].singlepoint(positions[lane])
        torch.testing.assert_close(packed.energy[lane], reference.energy, **TOL)


def test_mixed_gfn1_gfn2_groups_are_separate() -> None:
    """Different methods never share a group but pack into one Result."""
    gfn2_strategy = scf.SCFStrategy(
        (
            scf.SCFStage(
                scf.AndersonMethod(scf.AndersonMethodSettings()),
                representation=scf.SCFRepresentation.FOCK,
            ),
        )
    )
    gfn1 = _setup(_model(labels.GFN1_XTB), H2O)
    gfn2 = _setup(_model(labels.GFN2_XTB, strategy=gfn2_strategy), H2O)
    systems = (gfn1, gfn2, gfn1, gfn2)
    positions = tuple(_geometry(H2O, seed) for seed in range(4))
    none = [None] * 4

    assert _group_layout(group_lanes(systems, none, none, none)) == [
        ((0, 2), True),
        ((1, 3), True),
    ]
    packed = evaluate_packed_systems(systems, positions)
    for lane, (system, position) in enumerate(zip(systems, positions)):
        reference = system.singlepoint(position)
        torch.testing.assert_close(packed.energy[lane], reference.energy, **TOL)


# Charge and spin


def test_per_lane_and_fractional_charges_with_derivatives() -> None:
    """Mapped per-lane charges keep serial values and charge derivatives."""
    system = _setup(_model(), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))
    charges = torch.tensor([0.0, 0.3, -0.25], dtype=DTYPE, requires_grad=True)

    packed = evaluate_packed_systems((system,) * 3, positions, chrg=charges)
    (packed_grad,) = torch.autograd.grad(packed.energy.sum(), charges)

    serial_charges = charges.detach().clone().requires_grad_(True)
    serial = [
        system.singlepoint(position, chrg=serial_charges[lane])
        for lane, position in enumerate(positions)
    ]
    (serial_grad,) = torch.autograd.grad(
        sum(result.energy.sum() for result in serial), serial_charges
    )
    for lane, reference in enumerate(serial):
        torch.testing.assert_close(packed.energy[lane], reference.energy, **TOL)
    torch.testing.assert_close(packed_grad, serial_grad, **TOL)
    assert float(packed_grad.abs().min()) > 0.0


def test_spin_none_explicit_and_open_shell_lanes() -> None:
    """spin=None and explicit spin form separate groups with serial parity."""
    model = _model()
    water = _setup(model, H2O)
    hydroxyl = _setup(model, OH)
    systems = (water, hydroxyl, water, hydroxyl)
    positions = (
        _geometry(H2O, 0),
        _geometry(OH, 0),
        _geometry(H2O, 1),
        _geometry(OH, 1),
    )
    spins = (None, 1.0, None, 1.0)

    groups = group_lanes(
        systems,
        [None, torch.tensor(1.0), None, torch.tensor(1.0)],
        [None] * 4,
        [None] * 4,
    )
    assert _group_layout(groups) == [((0, 2), True), ((1, 3), True)]

    positions_grad = tuple(p.clone().requires_grad_(True) for p in positions)
    packed = evaluate_packed_systems(systems, positions_grad, spin=spins)
    packed_grads = torch.autograd.grad(packed.energy.sum(), positions_grad)
    for lane, (system, position, spin) in enumerate(
        zip(systems, positions, spins)
    ):
        reference_pos = position.clone().requires_grad_(True)
        reference = system.singlepoint(reference_pos, spin=spin)
        (reference_grad,) = torch.autograd.grad(
            reference.energy.sum(), reference_pos
        )
        nat = position.shape[0]
        torch.testing.assert_close(
            packed.energy[lane, :nat], reference.energy, **TOL
        )
        torch.testing.assert_close(packed_grads[lane], reference_grad, **TOL)


def test_invalid_electron_configuration_in_mapped_group_raises() -> None:
    """Eager electron/spin validation runs before a group is mapped."""
    system = _setup(_model(), H2O)
    positions = (_geometry(H2O, 0), _geometry(H2O, 1))

    with pytest.raises(ValueError) as excinfo:
        evaluate_packed_systems((system, system), positions, spin=(0.0, 1.0))
    assert getattr(excinfo.value, "source_id", None) == 1


# Fields


@pytest.mark.parametrize("kind", ["field", "field_grad"])
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per-lane"])
def test_field_inputs_match_serial_with_gradients(
    kind: str, shared: bool
) -> None:
    """Shared and per-lane perturbations keep serial values and gradients."""
    system = _setup(_model(), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))
    shape = (3,) if kind == "field" else (3, 3)
    scale = 1.0e-3 if kind == "field" else 2.0e-4
    if shared:
        base = scale * torch.arange(1, 1 + torch.Size(shape).numel())
        value = base.reshape(shape).to(DTYPE).requires_grad_(True)
        lane_values = (value, value, value)
        call_value = value
        leaves = (value,)
    else:
        lane_values = tuple(
            (
                scale * (lane + 1) * torch.ones(shape, dtype=DTYPE)
            ).requires_grad_(True)
            for lane in range(3)
        )
        call_value = lane_values
        leaves = lane_values

    packed = evaluate_packed_systems(
        (system,) * 3, positions, **{kind: call_value}
    )
    packed_grads = torch.autograd.grad(packed.energy.sum(), leaves)

    serial = [
        system.singlepoint(position, **{kind: lane_value})
        for position, lane_value in zip(positions, lane_values)
    ]
    serial_grads = torch.autograd.grad(
        sum(result.energy.sum() for result in serial), leaves
    )
    for lane, reference in enumerate(serial):
        torch.testing.assert_close(packed.energy[lane], reference.energy, **TOL)
    for packed_grad, serial_grad in zip(packed_grads, serial_grads):
        torch.testing.assert_close(packed_grad, serial_grad, **TOL)
        assert float(packed_grad.abs().max()) > 0.0


# Derivative isolation and model parameters


def test_position_gradients_are_lane_local() -> None:
    """A lane's energy has no derivative w.r.t. other lanes' positions."""
    compositions = (H2O, CH4, H2O)
    calc = _calculator(compositions)
    positions = _packed_positions(compositions, (0, 1, 2)).requires_grad_(True)

    packed = calc.singlepoint(positions)
    (grad,) = torch.autograd.grad(packed.energy[0].sum(), positions)

    assert float(grad[0, :3].abs().max()) > 0.0
    assert torch.count_nonzero(grad[1]) == 0
    assert torch.count_nonzero(grad[2]) == 0


def test_shared_model_parameter_gradients_match_serial_sum() -> None:
    """Grouped and singleton lanes accumulate the serial parameter gradient."""
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    par.set_differentiable("element", "O", "gam")
    par.set_differentiable("element", "H", "gam")
    model = _model(par=par)
    water = _setup(model, H2O)
    methane = _setup(model, CH4)
    systems = (water, methane, water)
    positions = (_geometry(H2O, 0), _geometry(CH4, 0), _geometry(H2O, 1))
    params = tuple(
        tensor
        for tensor in model.parameter_tree.parameters.params.values()
        if tensor.requires_grad
    )
    assert params

    packed = evaluate_packed_systems(systems, positions)
    packed_grads = torch.autograd.grad(
        packed.energy.sum(), params, allow_unused=True, retain_graph=True
    )
    serial = [
        system.singlepoint(position)
        for system, position in zip(systems, positions)
    ]
    serial_grads = torch.autograd.grad(
        sum(result.energy.sum() for result in serial),
        params,
        allow_unused=True,
    )

    nonzero = 0
    for packed_grad, serial_grad in zip(packed_grads, serial_grads):
        if serial_grad is None:
            assert packed_grad is None
            continue
        torch.testing.assert_close(packed_grad, serial_grad, **TOL)
        nonzero += int(float(packed_grad.abs().max()) > 0.0)
    assert nonzero > 0


# Fallback routes


def test_valid_non_vmap_systems_are_evaluated_independently() -> None:
    """libcint, implicit, eager-only and SCF-free lanes use the eager route."""
    pytest.importorskip("tad_libcint")
    multistage = scf.strategy_preset("broyden-cdiis")
    variants = {
        "libcint": _model(int_driver=labels.INTDRIVER_LIBCINT),
        "implicit": _model(differentiation="implicit"),
        "eager-only": _model(strategy=multistage, differentiation="implicit"),
        "scf-free": _model(exclude=["scf"]),
    }
    for name, model in variants.items():
        system = _setup(model, H2O)
        assert not can_vmap_system(system), name
        systems = (system, system)
        positions = (_geometry(H2O, 0), _geometry(H2O, 1))
        none = [None, None]
        assert _group_layout(group_lanes(systems, none, none, none)) == [
            ((0,), False),
            ((1,), False),
        ], name

        packed = evaluate_packed_systems(systems, positions)
        for lane, position in enumerate(positions):
            reference = system.singlepoint(position)
            torch.testing.assert_close(
                packed.energy[lane], reference.energy, **TOL
            )


def test_invalid_configuration_raises_instead_of_falling_back() -> None:
    """Configuration errors are not hidden by the independent route."""
    system = _setup(_model(), H2O)
    invalid = _setup(_model(fermi_maxiter=-1), H2O)

    with pytest.raises(ValueError, match="Fermi maxiter"):
        evaluate_packed_systems(
            (system, invalid), (_geometry(H2O, 0), _geometry(H2O, 1))
        )


def test_result_fields_are_all_split_from_a_mapped_group() -> None:
    """Splitting a mapped Result keeps every populated field and label."""
    system = _setup(_model(), H2O)
    positions = (_geometry(H2O, 0), _geometry(H2O, 1))
    group = group_lanes((system, system), [None] * 2, [None] * 2, [None] * 2)[0]
    charges = tuple(torch.tensor(0.0, dtype=DTYPE) for _ in range(2))

    lanes = lane_evaluation.evaluate_lane_group(
        group, positions, charges, [None] * 2, [None] * 2, [None] * 2
    )
    reference = system.singlepoint(positions[1])

    for field in fields(Result):
        expected = getattr(reference, field.name)
        actual = getattr(lanes[1], field.name)
        assert (actual is None) == (expected is None), field.name
    assert [label for label, _ in lanes[1].classical] == [
        label for label, _ in reference.classical
    ]
    assert lanes[1].charges is not None and reference.charges is not None
    assert (lanes[1].charges.dipole is None) == (
        reference.charges.dipole is None
    )
    assert lanes[1].status is not None
    assert lanes[1].status.final_failure.ndim == 0


@pytest.mark.parametrize(
    "module",
    [
        "dxtb._src.calculators.canonical_padding",
        "dxtb._src.calculators.padding",
        "dxtb._src.calculators.padded_integrals",
        "dxtb._src.scf.functional.padded",
    ],
)
def test_cross_composition_padding_modules_are_removed(module: str) -> None:
    """No canonical cross-composition padding implementation remains."""
    from importlib.util import find_spec

    assert find_spec(module) is None
