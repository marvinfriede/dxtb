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
"""Dispatcher mechanics of grouped-vmap batching.

Structural batch handling (lane deduplication, System classification,
electron/spin validation of mapped groups, legacy status inspection and
Result packing) must resolve host decisions once per batch or group, not once
per atom, Result field or lane. On CPU, Python-level tensor-to-host
conversions (``item``, ``tolist``, ``bool(tensor)``, ...) serve as the proxy
for CUDA host synchronizations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from dxtb import GFN1_XTB, Calculator, OutputHandler, ParamModule, labels
from dxtb._src.calculators import lane_evaluation
from dxtb._src.calculators import packed_result as pr_mod
from dxtb._src.calculators.lane_evaluation import evaluate_systems_independently
from dxtb._src.calculators.packed_result import (
    build_calculator_lane_systems,
    evaluate_packed_systems,
    pack_system_results,
)
from dxtb._src.components.classicals import new_dispersion
from dxtb._src.typing.exceptions import (
    SCFConvergenceError,
    SCFConvergenceWarning,
)
from test.test_calculator.test_grouped_vmap_batching import (
    CH4,
    DTYPE,
    H2,
    H2O,
    NH3,
    TOL,
    _SETTINGS,
    _calculator,
    _geometry,
    _model,
    _packed_positions,
    _setup,
)

_HOST_CONVERSIONS = ("item", "tolist", "__bool__", "__int__", "__float__")


@contextmanager
def _count_host_conversions() -> Iterator[Counter]:
    """Count Python-level tensor-to-host conversions inside the block."""
    counter: Counter = Counter()
    originals = {name: getattr(torch.Tensor, name) for name in _HOST_CONVERSIONS}

    def _wrap(name: str, original: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            counter[name] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for name, original in originals.items():
        setattr(torch.Tensor, name, _wrap(name, original))
    try:
        yield counter
    finally:
        for name, original in originals.items():
            setattr(torch.Tensor, name, original)


def _old_dedup_partition(calc) -> list[int]:
    """Lane partition of the scalar-by-scalar key used up to 17b2bfe4."""
    keys: list[tuple] = []
    for idx in range(calc.numbers.shape[0]):
        row = calc.numbers[idx]
        real = row > 0
        component_key = tuple(
            (id(c), tuple(int(x.item()) for x in c.numbers[idx][real]))
            for c in calc.model.classical
            if isinstance(getattr(c, "numbers", None), torch.Tensor)
            and c.numbers.ndim == 2
        )
        keys.append((tuple(int(x.item()) for x in row[real]), component_key))
    return [keys.index(key) for key in keys]


def _identity_partition(systems) -> list[int]:
    ids = [id(system) for system in systems]
    return [ids.index(i) for i in ids]


# Lane deduplication


def _calculator_with_component_numbers(numbers: torch.Tensor):
    """Batched Calculator whose caller-owned D3 component owns 2D numbers."""
    par = ParamModule(GFN1_XTB, dtype=DTYPE)
    d3 = new_dispersion(numbers, par, dtype=DTYPE)
    assert d3 is not None and d3.numbers.ndim == 2
    calc = Calculator(
        numbers,
        par,
        classical=(d3,),
        opts=dict(
            _SETTINGS,
            method=labels.GFN1_XTB,
            verbosity=0,
            exclude=["disp"],
        ),
        dtype=DTYPE,
        auto_int_level=False,
    )
    return calc, d3


def _padded_numbers(compositions) -> torch.Tensor:
    numbers = torch.zeros(
        (len(compositions), max(map(len, compositions))), dtype=torch.int64
    )
    for lane, composition in enumerate(compositions):
        numbers[lane, : len(composition)] = torch.tensor(composition)
    return numbers


def test_dedup_key_matches_previous_scalar_key_with_component_numbers() -> None:
    """Host-resolved keys give the same partition, incl. component numbers."""
    numbers = _padded_numbers((H2O, CH4, H2O, NH3, CH4, H2O))
    calc, d3 = _calculator_with_component_numbers(numbers)
    assert any(c is d3 for c in calc.model.classical)

    systems = build_calculator_lane_systems(calc)
    assert _identity_partition(systems) == _old_dedup_partition(calc)
    assert _identity_partition(systems) == [0, 1, 0, 3, 1, 0]

    # Lanes whose main numbers agree but whose component-owned numbers
    # differ must not share a System.
    d3.numbers = d3.numbers.clone()
    d3.numbers[5, 1] = 9
    systems = build_calculator_lane_systems(calc)
    assert _identity_partition(systems) == _old_dedup_partition(calc)
    assert _identity_partition(systems) == [0, 1, 0, 3, 1, 5]


def test_dedup_noncontiguous_padding_matches_previous_key() -> None:
    """Non-contiguous zero slots are skipped exactly as before."""
    numbers = torch.zeros((3, 6), dtype=torch.int64)
    numbers[0, :3] = torch.tensor(H2O)
    numbers[1, [0, 2, 5]] = torch.tensor(H2O)
    numbers[2, [0, 1, 3, 4, 5]] = torch.tensor(CH4)
    calc, _ = _calculator_with_component_numbers(numbers)

    systems = build_calculator_lane_systems(calc)
    assert _identity_partition(systems) == _old_dedup_partition(calc)
    assert _identity_partition(systems) == [0, 0, 2]
    assert systems[2].numbers.tolist() == list(CH4)


def test_dedup_host_conversions_do_not_scale_with_batch_or_atoms() -> None:
    """Lane deduplication transfers the numbers in bulk, not per atom."""
    counts = []
    for batch in (2, 12):
        calc, _ = _calculator_with_component_numbers(
            _padded_numbers((CH4,) * batch)
        )
        with _count_host_conversions() as counter:
            systems = build_calculator_lane_systems(calc)
        assert len({id(s) for s in systems}) == 1
        counts.append(dict(counter))
    assert counts[0] == counts[1]


# Classification and validation


def test_each_distinct_system_is_classified_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Six lanes over three Systems need three eligibility decisions."""
    calls: list[int] = []
    original = lane_evaluation.can_vmap_system

    def counting(system):
        calls.append(id(system))
        return original(system)

    monkeypatch.setattr(lane_evaluation, "can_vmap_system", counting)
    compositions = (H2O, CH4, H2O, NH3, H2O, CH4)
    calc = _calculator(compositions)
    calc.singlepoint(_packed_positions(compositions, range(6)))
    assert len(calls) == 3
    assert len(set(calls)) == 3


def test_direct_independent_evaluation_still_validates() -> None:
    """Direct callers keep the configuration validation."""
    system = _setup(_model(fermi_maxiter=-1), H2O)
    with pytest.raises(ValueError, match="Fermi maxiter"):
        evaluate_systems_independently((system,), (_geometry(H2O, 0),))


# Electron/spin validation of mapped groups

_ELECTRON_CASES = [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1.0, 0.0),
    (0.0, 2.0),
    (0.5, 0.0),
    (0.6, 0.0),
    (-0.4, 0.0),
    (0.0, 10.0),
    (-2.0, 2.0),
]


def test_group_electron_rules_match_single_system_rules() -> None:
    """The grouped decision equals the eager single-System check per lane."""
    system = _setup(_model(), H2O)
    expected_invalid = []
    for lane, (charge, spin) in enumerate(_ELECTRON_CASES):
        try:
            lane_evaluation._validate_electrons(
                system,
                torch.tensor(charge, dtype=DTYPE),
                torch.tensor(spin, dtype=DTYPE),
            )
        except ValueError:
            expected_invalid.append(lane)
    assert expected_invalid  # the case list covers invalid lanes

    # Validate every single-lane prefix: the first failing lane is reported.
    for lane in range(len(_ELECTRON_CASES)):
        group = lane_evaluation.LaneGroup(
            system=system, source_ids=(lane,), use_vmap=True
        )
        charges = [torch.tensor(c, dtype=DTYPE) for c, _ in _ELECTRON_CASES]
        spins = [torch.tensor(s, dtype=DTYPE) for _, s in _ELECTRON_CASES]
        try:
            lane_evaluation._validate_group_electrons(group, charges, spins)
            raised = False
        except ValueError:
            raised = True
        assert raised == (lane in expected_invalid), lane


def test_group_electron_validation_reports_first_invalid_source_lane() -> None:
    """Mixed valid/invalid lanes raise the eager error of the first one."""
    system = _setup(_model(), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(5))
    charges = (0.0, 0.5, 1.0, 0.0, 0.0)
    spins = (0.0, 0.0, 0.0, 1.0, 0.0)  # lanes 2 and 3 violate parity

    with pytest.raises(ValueError) as excinfo:
        evaluate_packed_systems(
            (system,) * 5, positions, chrg=charges, spin=spins
        )
    assert getattr(excinfo.value, "source_id", None) == 2

    with pytest.raises(ValueError) as reference:
        lane_evaluation._validate_electrons(
            system, torch.tensor(1.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE)
        )
    assert str(excinfo.value) == str(reference.value)

    with pytest.raises(ValueError, match="larger than") as excinfo:
        evaluate_packed_systems(
            (system,) * 3,
            positions[:3],
            chrg=(0.0, 0.0, 0.0),
            spin=(0.0, 2.0, 10.0),
        )
    assert getattr(excinfo.value, "source_id", None) == 2


def test_valid_groups_need_no_per_lane_electron_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid groups (spin None or explicit) never run per-lane checks."""
    calls: list[int] = []
    original = lane_evaluation._validate_electrons

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(lane_evaluation, "_validate_electrons", counting)
    system = _setup(_model(), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))
    for spin in (None, (0.0, 2.0, 0.0)):
        evaluate_packed_systems(
            (system,) * 3, positions, chrg=(0.0, 0.0, 0.5), spin=spin
        )
    assert calls == []


# Legacy status inspection of mapped groups


@pytest.mark.parametrize("force", [False, True])
def test_mapped_nonconvergence_policy_per_lane(force: bool) -> None:
    """Every unconverged mapped lane warns, or the first one raises."""
    system = _setup(_model(maxiter=1, force_convergence=force), H2O)
    positions = tuple(_geometry(H2O, seed) for seed in range(3))

    OutputHandler.clear_warnings()
    if force:
        with pytest.raises(SCFConvergenceError) as excinfo:
            evaluate_packed_systems(
                (system,) * 3, positions, legacy_compat=True
            )
        assert getattr(excinfo.value, "source_id", None) == 0
        assert OutputHandler.warnings == []
    else:
        evaluate_packed_systems((system,) * 3, positions, legacy_compat=True)
        assert [w[1] for w in OutputHandler.warnings] == [
            SCFConvergenceWarning
        ] * 3
    OutputHandler.clear_warnings()


def test_mapped_failure_status_raises_with_source_id() -> None:
    """Fermi failures of mapped lanes raise the eager error of the first."""
    system = _setup(_model(), H2)
    positions = tuple(_geometry(H2, seed) for seed in range(3))

    # Reference: the same lane evaluated by the eager singleton route.
    with pytest.raises(ValueError) as reference:
        evaluate_packed_systems(
            (system,), positions[:1], chrg=(-6.5,), legacy_compat=True
        )
    with pytest.raises(ValueError) as excinfo:
        evaluate_packed_systems(
            (system,) * 3,
            positions,
            chrg=(0.0, -6.5, -6.5),
            legacy_compat=True,
        )
    assert str(excinfo.value) == str(reference.value)
    assert getattr(excinfo.value, "source_id", None) == 1


def test_group_status_inspection_is_one_host_decision() -> None:
    """Status of a mapped group is inspected with one host transfer."""
    system = _setup(_model(maxiter=1), H2O)
    counts = []
    for batch in (2, 6):
        positions = tuple(_geometry(H2O, seed) for seed in range(batch))
        (group,) = lane_evaluation.group_lanes(
            (system,) * batch, [None] * batch, [None] * batch, [None] * batch
        )
        assert group.use_vmap
        lanes = lane_evaluation.evaluate_lane_group(
            group,
            positions,
            [torch.tensor(0.0, dtype=DTYPE)] * batch,
            [None] * batch,
            [None] * batch,
            [None] * batch,
        )
        OutputHandler.clear_warnings()
        with _count_host_conversions() as counter:
            pr_mod._enforce_eager_status_policy(
                lanes, system, tuple(range(batch))
            )
        assert len(OutputHandler.warnings) == batch
        OutputHandler.clear_warnings()
        counts.append(dict(counter))
    assert counts[0] == counts[1]
    assert counts[0].get("tolist", 0) == 1


def test_mapped_dispatch_host_conversions_do_not_scale_with_group_size() -> None:
    """Public conformer batches: structural host decisions are O(1) in B."""
    counts = []
    for batch in (2, 6):
        calc = _calculator((H2O,) * batch)
        positions = _packed_positions((H2O,) * batch, range(batch))
        spin = torch.zeros(batch, dtype=DTYPE)
        calc.singlepoint(positions, spin=spin)  # warm up lazy imports
        with _count_host_conversions() as counter:
            calc.singlepoint(positions, spin=spin)
        counts.append(dict(counter))
    assert counts[0] == counts[1]


# Result packing


def _lane_results():
    model = _model()
    systems = (_setup(model, H2O), _setup(model, CH4), _setup(model, H2O))
    positions = (_geometry(H2O, 0), _geometry(CH4, 0), _geometry(H2O, 1))
    return evaluate_systems_independently(systems, positions)


def _masks(target: int) -> tuple[torch.Tensor, ...]:
    prefix = torch.arange(target) < 3
    scattered = torch.zeros(target, dtype=torch.bool)
    scattered[[0, 1, 3, 4, 6]] = True
    other = torch.zeros(target, dtype=torch.bool)
    other[[1, 4, 6]] = True
    return prefix, scattered, other


def test_packing_places_prefix_and_noncontiguous_atoms_with_gradients() -> None:
    """One gather placement handles both mask kinds, incl. derivatives."""
    results = _lane_results()
    target = 7
    masks = _masks(target)
    leaves = [
        r.energy.detach().clone().requires_grad_(True) for r in results
    ]
    dipoles = [
        torch.randn((r.energy.shape[-1], 3), dtype=DTYPE, requires_grad=True)
        for r in results
    ]
    lanes = [
        replace(
            r,
            energy=e,
            charges=replace(r.charges, dipole=d),
        )
        for r, e, d in zip(results, leaves, dipoles)
    ]

    packed = pack_system_results(lanes, target_atoms=target, atom_masks=masks)

    for lane, (mask, e, d) in enumerate(zip(masks, leaves, dipoles)):
        expected = torch.zeros(target, dtype=DTYPE)
        expected[mask] = e.detach()
        torch.testing.assert_close(packed.energy[lane], expected, **TOL)
        expected_dipole = torch.zeros((target, 3), dtype=DTYPE)
        expected_dipole[mask] = d.detach()
        torch.testing.assert_close(
            packed.charges.dipole[lane], expected_dipole, **TOL
        )

    weights = torch.randn(packed.energy.shape, dtype=DTYPE)
    dipole_weights = torch.randn(packed.charges.dipole.shape, dtype=DTYPE)
    objective = (weights * packed.energy).sum() + (
        dipole_weights * packed.charges.dipole
    ).sum()
    grads = torch.autograd.grad(objective, [*leaves, *dipoles])
    for lane, mask in enumerate(masks):
        torch.testing.assert_close(grads[lane], weights[lane, mask], **TOL)
        torch.testing.assert_close(
            grads[3 + lane], dipole_weights[lane, mask], **TOL
        )


def test_packing_mask_resolution_is_one_host_decision() -> None:
    """Atom masks are inspected once, not per lane and per Result field."""
    results = _lane_results()
    masks = _masks(7)
    counts = []
    for repeat in (1, 3):
        lanes = list(results) * repeat
        lane_masks = masks * repeat
        with _count_host_conversions() as counter:
            pack_system_results(lanes, target_atoms=7, atom_masks=lane_masks)
        counts.append(dict(counter))
    assert counts[0] == counts[1]
