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
"""Order and duplicate behavior for shared component lists."""

from __future__ import annotations

import pytest
import torch

from dxtb._src.components.base import Component
from dxtb._src.components.classicals.base import Classical, ClassicalCache
from dxtb._src.components.classicals.list import ClassicalList
from dxtb._src.components.interactions.field.efield import (
    ElectricField,
    build_electric_field_data,
)
from dxtb._src.components.interactions.field.efieldgrad import (
    ElectricFieldGrad,
    build_electric_field_grad_data,
)
from dxtb._src.components.interactions.list import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.components.list import ComponentList
from dxtb._src.scf.functional.interactions import (
    prepare_functional_interactions,
)


class _OrderedValue(Component):
    def __init__(self, label: str, hash_value: int, key: str | None = None):
        super().__init__()
        self.label = label
        self.hash_value = hash_value
        self.key = label if key is None else key

    def __hash__(self) -> int:
        return self.hash_value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OrderedValue) and self.key == other.key

    def type(self, dtype: torch.dtype) -> _OrderedValue:
        del dtype
        return self


class _List(ComponentList):
    def get_energy(self, *_: object, **__: object) -> torch.Tensor:
        return torch.zeros(())

    def get_gradient(self, *_: object, **__: object) -> torch.Tensor:
        return torch.zeros(())

    def get_cache(self, *_: object, **__: object) -> dict:
        return {}


class _Classical(Classical):
    def __init__(self, label: str, value: float = 0.0):
        super().__init__()
        self.label = label
        self.value = value

    def get_cache(self, *_: object, **__: object) -> ClassicalCache:
        return ClassicalCache()

    def get_energy(self, *_: object, **__: object) -> torch.Tensor:
        return torch.tensor(self.value)


def test_component_list_preserves_input_order() -> None:
    """Distinct components retain their supplied order despite controlled hashes."""
    a = _OrderedValue("Z", 3)
    b = _OrderedValue("A", 1)
    c = _OrderedValue("M", 2)

    components = _List(a, b, c).components

    assert components == [a, b, c]
    assert components[0] is a
    assert components[1] is b
    assert components[2] is c


def test_component_list_deduplicates_first_equal_value() -> None:
    """First equal/hash-equivalent object wins, matching set duplicate semantics."""
    first = _OrderedValue("first", 17, key="same")
    duplicate = _OrderedValue("duplicate", 17, key="same")
    other = _OrderedValue("other", 4)

    components = _List(first, duplicate, other).components

    assert components == [first, other]
    assert components[0] is first


def test_component_list_keeps_none_and_duplicate_instance_contract() -> None:
    """None is ignored and repeated component instances appear only once."""
    a = _OrderedValue("Z", 3)
    b = _OrderedValue("A", 1)

    assert _List(None, a, None, b, a).components == [a, b]
    assert _List(b, a, b).components == [b, a]


def test_component_list_type_preserves_order() -> None:
    """Applying a dtype keeps the existing component sequence."""
    a = _OrderedValue("Z", 3)
    b = _OrderedValue("A", 1)
    c = _OrderedValue("M", 2)
    components = _List(a, b, c)

    components.type(torch.float32)

    assert components.components == [a, b, c]


def test_component_list_keeps_hashability_requirement() -> None:
    """Unhashable custom values remain unsupported as in the former set path."""

    class Unhashable(_OrderedValue):
        __hash__ = None

    with pytest.raises(TypeError):
        _List(Unhashable("A", 1))


def test_classical_list_preserves_order_and_duplicate_label_guard() -> None:
    """Classical contributions retain input order and reject migrated label collisions."""
    a = _Classical("A", 1.0)
    b = _Classical("B", 2.0)
    classicals = ClassicalList(a, b)
    assert classicals.components == [a, b]
    assert ClassicalList(a, a, b).components == [a, b]

    cache = classicals.get_cache(torch.tensor([1]), None)
    energies = classicals.get_energy(torch.zeros((1, 3)), cache)
    assert list(energies) == ["A", "B"]
    assert energies["A"] == 1.0
    assert energies["B"] == 2.0

    first = _Classical("Repulsion")
    second = _Classical("Repulsion")
    with pytest.raises(ValueError, match="Duplicate repulsion labels"):
        ClassicalList(first, second)


def test_interaction_and_functional_order_match() -> None:
    """Functional descriptors inherit the caller's interaction order."""
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    field = ElectricField(torch.tensor([0.0, 0.0, 0.2]))
    field_grad = ElectricFieldGrad(torch.eye(3))
    interactions = InteractionList(field_grad, field)
    cache = InteractionListCache()
    cache[field_grad.label] = build_electric_field_grad_data(
        positions, field_grad.field_grad
    )
    cache[field.label] = build_electric_field_data(positions, field.field)

    descriptors = prepare_functional_interactions(interactions, cache, numbers)

    assert interactions.components == [field_grad, field]
    assert [entry.kind for entry in descriptors.entries] == [
        "electric_field_grad",
        "electric_field",
    ]
    assert InteractionList(field, field).components == [field]


def test_interaction_list_rejects_distinct_duplicate_label() -> None:
    """Distinct migrated interaction objects with the same label remain invalid."""
    first = ElectricField(torch.zeros(3))
    second = ElectricField(torch.ones(3))

    with pytest.raises(
        ValueError, match="Duplicate migrated interaction labels"
    ):
        InteractionList(first, second)


def test_component_list_process_order_is_input_order() -> None:
    """Controlled component hashes do not change order across process launches."""
    import os
    import subprocess
    import sys

    program = """
from dxtb._src.components.list import ComponentList
class C:
    def __init__(self, label, hash_value):
        self.label, self.hash_value = label, hash_value
    def __hash__(self):
        return self.hash_value
    def __eq__(self, other):
        return self is other
class L(ComponentList):
    def get_energy(self, *args, **kwargs): pass
    def get_gradient(self, *args, **kwargs): pass
    def get_cache(self, *args, **kwargs): pass
items = [C('Z', 3), C('A', 1), C('M', 2)]
print(','.join(x.label for x in L(*items).components))
"""
    for seed in ("0", "1", "37"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            check=True,
            env=env,
            text=True,
        )
        assert result.stdout.strip() == "Z,A,M"
