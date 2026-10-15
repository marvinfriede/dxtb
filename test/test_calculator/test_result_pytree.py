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
"""Pytree representation tests for immutable public calculation results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest
import torch
from torch.func import vmap
from torch.utils import _pytree as pytree

from dxtb._src.calculators.result import (
    ChargeResult,
    Multipoles,
    PotentialResult,
    Result,
)
from dxtb._src.scf.status import SCFStatus
from tad_mctc.tree import combine, leaf_paths, partition, stack


def _tensor(index: int, *, requires_grad: bool = False) -> torch.Tensor:
    """Create a distinct scalar tensor for result-layout assertions."""
    return torch.tensor(
        float(index), dtype=torch.float64, requires_grad=requires_grad
    )


def _matrix(index: int, leading: int | None = None) -> torch.Tensor:
    """Create a small integral-shaped tensor with a distinct value."""
    tensor = torch.eye(2, dtype=torch.float64) * index
    return tensor if leading is None else tensor.expand(leading, 2, 2).clone()


def _full_result(*, offset: torch.Tensor | None = None) -> Result:
    """Construct a Result with every declared field populated."""
    x = _tensor(1) if offset is None else offset
    return Result.snapshot(
        energy=x + 1,
        scf=x + 2,
        classical=(
            ("my custom term", x + 3),
            ("term_with_punctuation!?", x + 4),
        ),
        fenergy=x + 5,
        iterations=torch.ones((), dtype=torch.int64),
        charges=SimpleNamespace(mono=x + 6, dipole=x + 7, quad=x + 8),
        density=x + 9,
        coefficients=x + 10,
        emo=x + 11,
        occupation=x + 12,
        potential=SimpleNamespace(mono=x + 13, dipole=x + 14, quad=x + 15),
        hamiltonian=x + 16,
        overlap=_matrix(17),
        hcore=_matrix(18),
        dipole_integrals=_matrix(19, leading=3),
        quadrupole_integrals=_matrix(20, leading=6),
        overlap_norm=x + 21,
        converged=torch.tensor(True),
        residual=x + 22,
        status=SCFStatus(
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(3, dtype=torch.int64),
        ),
    )


def _assert_same_tensor_fields(left: object, right: object) -> None:
    """Check each declared value field after an ordinary tree round-trip."""
    for field in fields(type(left)):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, torch.Tensor):
            assert right_value is left_value
        elif field.name == "classical":
            assert tuple(label for label, _ in right_value) == tuple(
                label for label, _ in left_value
            )
            assert all(
                a is b for (_, a), (_, b) in zip(left_value, right_value)
            )
        elif left_value is None:
            assert right_value is None
        elif type(left_value) in (
            ChargeResult,
            PotentialResult,
            Multipoles,
            SCFStatus,
        ):
            assert type(right_value) is type(left_value)
            _assert_same_tensor_fields(left_value, right_value)
        else:
            assert right_value == left_value


def test_result_and_nested_values_round_trip() -> None:
    """Reconstruct every public result value while preserving leaf identity."""
    assert tuple(field.name for field in fields(Result)) == (
        "energy",
        "scf",
        "classical",
        "fenergy",
        "iterations",
        "charges",
        "density",
        "coefficients",
        "emo",
        "occupation",
        "potential",
        "hamiltonian",
        "overlap",
        "hcore",
        "dipole_integrals",
        "quadrupole_integrals",
        "overlap_norm",
        "converged",
        "residual",
        "status",
    )
    result = _full_result()
    values = (
        result,
        result.charges,
        result.potential,
        result.multipoles,
        result.status,
        ChargeResult(_tensor(1), None, None),
        ChargeResult(_tensor(1), _tensor(2), _tensor(3)),
        PotentialResult(_tensor(1), None, None),
        PotentialResult(None, None, None),
        Multipoles(_tensor(1), _tensor(2)),
        Multipoles(None, None),
    )
    for value in values:
        assert value is not None
        leaves, spec = pytree.tree_flatten(value)
        rebuilt = pytree.tree_unflatten(leaves, spec)
        assert type(rebuilt) is type(value)
        _assert_same_tensor_fields(value, rebuilt)

    assert result.classical[0][0] == "my custom term"
    assert result.classical[1][0] == "term_with_punctuation!?"
    assert result.total is result.energy
    assert result.iter is result.iterations
    assert result.cenergies["my custom term"] is result.classical[0][1]
    assert result.integrals is not None
    assert result.integrals.hcore is result.hcore
    assert result.multipoles is not None
    assert result.multipoles.dipole is result.charges.dipole


@pytest.mark.parametrize(
    "result",
    [
        Result(
            energy=_tensor(1),
            scf=None,
            classical=(),
            fenergy=None,
            iterations=torch.ones((), dtype=torch.int64),
            status=SCFStatus(
                torch.tensor(0, dtype=torch.int64),
                torch.tensor(0, dtype=torch.int64),
            ),
        ),
        Result(
            energy=_tensor(1),
            scf=None,
            classical=(("legacy", _tensor(2)),),
            fenergy=None,
            iterations=torch.ones((), dtype=torch.int64),
        ),
        Result(
            energy=_tensor(1),
            scf=None,
            classical=(),
            fenergy=None,
            iterations=torch.ones((), dtype=torch.int64),
        ),
    ],
)
def test_result_optional_layouts_round_trip(result: Result) -> None:
    """Preserve zero-temperature, legacy and SCF-excluded field layouts."""
    leaves, spec = pytree.tree_flatten(result)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is Result
    for field in fields(Result):
        left = getattr(result, field.name)
        right = getattr(rebuilt, field.name)
        if field.name == "classical":
            assert tuple(label for label, _ in right) == tuple(
                label for label, _ in left
            )
            assert all(a is b for (_, a), (_, b) in zip(left, right))
        elif isinstance(left, torch.Tensor):
            assert right is left
        elif isinstance(left, (ChargeResult, PotentialResult, SCFStatus)):
            assert type(right) is type(left)
        else:
            assert right == left


def test_classical_labels_are_static_and_energy_terms_are_dynamic() -> None:
    """Keep labels out of leaves while preserving ordered energy leaves."""
    energy_a = _tensor(1)
    energy_b = _tensor(2)
    result = Result(
        energy=_tensor(0),
        scf=None,
        classical=(("a", energy_a), ("b", energy_b)),
        fenergy=None,
        iterations=torch.ones((), dtype=torch.int64),
    )
    leaves, spec = pytree.tree_flatten(result)
    assert all(
        isinstance(leaf, torch.Tensor) for leaf in pytree.tree_leaves(result)
    )
    assert energy_a in leaves
    assert energy_b in leaves
    assert "a" not in leaves and "b" not in leaves
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert rebuilt.classical == (("a", energy_a), ("b", energy_b))
    assert rebuilt.classical[0][1] is energy_a
    assert rebuilt.classical[1][1] is energy_b

    other_labels = Result(
        energy=_tensor(0),
        scf=None,
        classical=(("different", energy_a), ("b", energy_b)),
        fenergy=None,
        iterations=torch.ones((), dtype=torch.int64),
    )
    assert pytree.tree_structure(other_labels) != pytree.tree_structure(result)
    duplicates = Result(
        energy=_tensor(0),
        scf=None,
        classical=(("same", energy_a), ("same", energy_b)),
        fenergy=None,
        iterations=torch.ones((), dtype=torch.int64),
    )
    duplicate_leaves, duplicate_spec = pytree.tree_flatten(duplicates)
    duplicate_rebuilt = pytree.tree_unflatten(duplicate_leaves, duplicate_spec)
    assert duplicate_rebuilt.classical == (
        ("same", energy_a),
        ("same", energy_b),
    )


def test_placeholder_unflatten_keeps_result_layout() -> None:
    """Keep pytree layout stable when transforms supply leaf placeholders."""
    result = _full_result()
    leaves, spec = pytree.tree_flatten(result)
    placeholders = [index for index, _ in enumerate(leaves)]
    rebuilt = pytree.tree_unflatten(placeholders, spec)
    rebuilt_leaves, rebuilt_spec = pytree.tree_flatten(rebuilt)
    assert rebuilt_spec == spec
    assert len(rebuilt_leaves) == len(leaves)
    assert tuple(label for label, _ in rebuilt.classical) == (
        "my custom term",
        "term_with_punctuation!?",
    )
    assert all(isinstance(value, int) for _, value in rebuilt.classical)


def test_result_paths_identify_nested_dynamic_values() -> None:
    """Expose stable paths for nested values and both classical terms."""
    result = _full_result()
    path_leaves, _ = pytree.tree_flatten_with_path(result)
    paths, leaves = zip(*path_leaves)
    rendered = {str(path): leaf for path, leaf in zip(paths, leaves)}
    assert any("energy" in path for path in rendered)
    assert any("charges" in path and "mono" in path for path in rendered)
    assert any("potential" in path and "mono" in path for path in rendered)
    assert any("trajectory_failure" in path for path in rendered)
    assert any("final_failure" in path for path in rendered)
    assert sum("classical" in path for path in rendered) == 2
    assert all(isinstance(value, torch.Tensor) for value in leaves)

    paths_from_tad = leaf_paths(result)
    assert len(paths_from_tad) == len(leaves)


def _probe(x: torch.Tensor) -> Result:
    """Build a complete transform-safe Result from one mapped tensor."""
    failure = x.to(torch.int64)
    return Result.snapshot(
        energy=x.square(),
        scf=x + 1,
        classical=(("term_a", x * 2), ("term_b", x * 3)),
        fenergy=-x,
        iterations=torch.ones((), dtype=torch.int64),
        charges=SimpleNamespace(mono=x, dipole=x + 4, quad=x + 5),
        density=x + 6,
        coefficients=x + 7,
        emo=x + 8,
        occupation=x + 9,
        potential=SimpleNamespace(mono=x + 10, dipole=x + 11, quad=x + 12),
        hamiltonian=x + 13,
        overlap=x + 14,
        hcore=x + 15,
        dipole_integrals=x + 16,
        quadrupole_integrals=x + 17,
        overlap_norm=x + 18,
        converged=torch.tensor(True),
        residual=x + 19,
        status=SCFStatus(failure, failure + 1),
    )


def test_default_vmap_returns_complete_result_and_maps_every_leaf() -> None:
    """Map a full Result directly with default out_dims and preserve leaves."""
    xs = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    mapped = vmap(_probe)(xs)
    assert type(mapped) is Result
    assert mapped.charges is not None and mapped.potential is not None
    assert mapped.status is not None
    assert mapped.classical[0][0] == "term_a"
    assert mapped.classical[1][0] == "term_b"
    assert mapped.status.trajectory_failure.dtype == torch.int64
    assert mapped.status.final_failure.dtype == torch.int64
    assert mapped.status.trajectory_failure.device == xs.device
    assert mapped.status.final_failure.device == xs.device
    torch.testing.assert_close(mapped.energy, xs.square())
    torch.testing.assert_close(mapped.classical[0][1], xs * 2)
    torch.testing.assert_close(mapped.classical[1][1], xs * 3)
    torch.testing.assert_close(
        mapped.status.trajectory_failure, xs.to(torch.int64)
    )
    torch.testing.assert_close(
        mapped.status.final_failure, xs.to(torch.int64) + 1
    )

    single = _probe(xs[0])
    single_leaves = pytree.tree_leaves(single)
    mapped_leaves = pytree.tree_leaves(mapped)
    assert len(single_leaves) == len(mapped_leaves)
    for single_leaf, mapped_leaf in zip(single_leaves, mapped_leaves):
        assert mapped_leaf.shape == (xs.shape[0],) + single_leaf.shape
    assert all(isinstance(leaf, torch.Tensor) for leaf in mapped_leaves)


def test_default_vmap_preserves_empty_and_optional_nested_layouts() -> None:
    """Map Results with empty classical data and static optional absences."""
    xs = torch.tensor([1.0, 2.0], dtype=torch.float64)

    def empty_classical(x: torch.Tensor) -> Result:
        return Result(
            energy=x.square(),
            scf=None,
            classical=(),
            fenergy=None,
            iterations=torch.ones((), dtype=torch.int64),
        )

    empty = vmap(empty_classical)(xs)
    assert type(empty) is Result
    assert empty.classical == ()
    torch.testing.assert_close(empty.energy, xs.square())

    def optional(x: torch.Tensor) -> Result:
        return Result(
            energy=x,
            scf=None,
            classical=(),
            fenergy=None,
            iterations=torch.ones((), dtype=torch.int64),
            charges=ChargeResult(x + 1, None, None),
            potential=PotentialResult(x + 2, None, None),
        )

    nested = vmap(optional)(xs)
    assert type(nested) is Result
    assert nested.charges is not None and nested.potential is not None
    assert nested.charges.dipole is None
    assert nested.charges.quadrupole is None
    assert nested.potential.dipole is None
    assert nested.potential.quadrupole is None
    assert nested.charges.mono.shape == (2,)
    assert nested.potential.mono.shape == (2,)

    def absent(x: torch.Tensor) -> Result:
        return Result(
            energy=x,
            scf=None,
            classical=(),
            fenergy=None,
            iterations=torch.ones((), dtype=torch.int64),
        )

    no_optional = vmap(absent)(xs)
    assert no_optional.charges is None
    assert no_optional.potential is None
    assert no_optional.status is None


def test_result_and_classical_gradients_survive_vmap() -> None:
    """Preserve reverse-mode gradients through result and classical leaves."""
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    result = _probe(x)
    (energy_grad,) = torch.autograd.grad(
        result.energy.sum(), x, retain_graph=True
    )
    (classical_grad,) = torch.autograd.grad(result.classical[0][1].sum(), x)
    assert energy_grad.item() == pytest.approx(4.0)
    assert classical_grad.item() == pytest.approx(2.0)

    xs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    mapped = vmap(_probe)(xs)
    loss = mapped.energy.sum() + mapped.classical[1][1].sum()
    (gradient,) = torch.autograd.grad(loss, xs)
    torch.testing.assert_close(gradient, 2 * xs + 3)

    loop = torch.stack(
        [
            torch.autograd.grad(_probe(value).energy + 3 * value, value)[0]
            for value in xs
        ]
    )
    torch.testing.assert_close(gradient, loop)


def test_nested_result_collections_and_tad_tree_utilities() -> None:
    """Compose Results inside built-in containers and partition tensor leaves."""
    first = _full_result()
    second = _full_result(offset=_tensor(2))
    nested = (first, {"first": first, "second": second})
    leaves, spec = pytree.tree_flatten(nested)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is tuple
    assert type(rebuilt[0]) is Result
    assert type(rebuilt[1]["second"]) is Result
    stacked = stack((first, second))
    assert type(stacked) is Result
    assert stacked.energy.shape == (2,)
    assert stacked.status is not None
    assert stacked.status.final_failure.shape == (2,)
    assert stacked.classical[0][0] == "my custom term"
    assert stacked.classical[0][1].shape == (2,)

    def select_energy(path: object, _: torch.Tensor) -> bool:
        return "energy" in str(path)

    parameters, static = partition(first, select_energy)
    combined = combine(parameters, static)
    assert type(combined) is Result
    assert combined.energy is first.energy


def test_direct_constructors_and_frozen_values_remain_compatible() -> None:
    """Keep positional constructors and frozen dataclass fields unchanged."""
    trajectory = torch.tensor(1, dtype=torch.int64)
    final = torch.tensor(2, dtype=torch.int64)
    status = SCFStatus(trajectory, final)
    assert status.trajectory_failure is trajectory
    charge = ChargeResult(_tensor(1), None, None)
    potential = PotentialResult(_tensor(2), None, None)
    multipoles = Multipoles(None, None)
    result = Result(
        _tensor(0), None, (), None, torch.ones((), dtype=torch.int64)
    )
    assert result.charges is None and result.status is None
    with pytest.raises(FrozenInstanceError):
        result.energy = _tensor(9)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        status.final_failure = trajectory  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        charge.mono = _tensor(3)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        potential.mono = _tensor(3)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        multipoles.dipole = _tensor(3)  # type: ignore[misc]
