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
"""E4.0 raw charge-map characterization and purity checks."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
from tad_mctc import storch
from torch.func import jacfwd, jacrev, jvp, vmap

import dxtb._src.scf.functional.map as functional_map
from dxtb import GFN1_XTB, GFN2_XTB, Calculator, labels
from dxtb._src.basis.indexhelper import IndexHelper
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.base import Component
from dxtb._src.components.classicals.dispersion.d3 import DispersionD3
from dxtb._src.components.interactions import Interaction, InteractionList
from dxtb._src.components.interactions.base import InteractionCache
from dxtb._src.components.interactions.container import Charges
from dxtb._src.components.interactions.coulomb.multipole import AES2
from dxtb._src.components.interactions.coulomb.secondorder import ES2
from dxtb._src.components.interactions.coulomb.thirdorder import ES3
from dxtb._src.components.interactions.dispersion.d4sc import DispersionD4SC
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.field.efieldgrad import ElectricFieldGrad
from dxtb._src.components.interactions.list import (
    _FUNCTIONAL_SCF_INTERACTION_TYPES,
    InteractionListCache,
)
from dxtb._src.components.interactions.solvation.alpb import GeneralizedBorn
from dxtb._src.constants import defaults
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional import (
    ChargeLayout,
    PotentialLayout,
    SCFCharges,
    SCFPotential,
    fock_initial_guess,
    fock_map,
    functional_scf_interactions_eligible,
    interaction_energy_from_charges,
    pack_charges,
    pack_potential,
    potential_from_charges,
    potential_initial_guess,
    potential_layout,
    potential_map,
    scf_map,
    setup_scf_inputs,
    unpack_charges,
    unpack_potential,
    validate_functional_scf_interactions,
)
from dxtb._src.scf.functional.core import (
    CoreIndexData,
    FunctionalInteractionSet,
    tensor_leaves,
)
from dxtb._src.scf.functional.map import (
    _transitional_diagonalize,
    evaluate_hamiltonian,
)
from dxtb._src.scf.functional.solver import (
    SCFSolverSettings,
    free_energy_from_spectral,
    solve_scf_simple,
)
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.scf.unrolling import SelfConsistentFieldFull
from dxtb._src.scf.utils import get_density
from dxtb._src.wavefunction import filling

from ..molecules import mols
from ..utils import get_param_module
from .samples import samples


def _prepare_case(
    method: int,
    *,
    sample_name: str = "H2O",
    charge: float | torch.Tensor = 0.0,
    spin: float = 0.0,
    field: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    par: ParamModule | None = None,
    dtype: torch.dtype = torch.float64,
    int_level: int | None = None,
    scf_options: dict[str, object] | None = None,
    setup_functional: bool = True,
) -> SimpleNamespace:
    sample = (
        samples[sample_name] if sample_name in samples else mols[sample_name]
    )
    numbers = sample["numbers"].to(device="cpu")
    if positions is None:
        positions = sample["positions"].to(device="cpu", dtype=dtype).clone()
    if par is None:
        par = get_param_module(
            "gfn1" if method == GFN1_XTB else "gfn2",
            device="cpu",
            dtype=dtype,
        )
    if int_level is None:
        int_level = (
            labels.INTLEVEL_DIPOLE
            if field is not None
            else labels.INTLEVEL_QUADRUPOLE
        )
    opts: dict[str, object] = {
        "verbosity": 0,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": labels.SCP_MODE_CHARGE,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "int_level": int_level,
        "fermi_etemp": 0.0,
    }
    if scf_options is not None:
        opts.update(scf_options)
    calc = Calculator(
        numbers,
        par,
        device="cpu",
        dtype=dtype,
        opts=opts,
    )
    system = calc.model.setup(numbers, dd=calc.dd)
    if isinstance(charge, torch.Tensor):
        charge_tensor = charge.reshape(1).to(
            device=positions.device, dtype=positions.dtype
        )
    else:
        charge_tensor = positions.new_tensor([charge])
    spin_tensor = positions.new_tensor([spin])
    integrals, refocc, _ = build_integral_matrices(
        system.integral_setup,
        system.h0_setup,
        positions,
        charge_tensor,
    )
    n0, occupation, nel = get_refocc(
        refocc, charge_tensor, spin_tensor, system.ihelp
    )

    interactions = system.interactions
    if field is not None:
        interactions = InteractionList(
            *system.interactions.components,
            ElectricField(field),
            device=positions.device,
            dtype=positions.dtype,
        )
    interaction_data = _interaction_data(system, positions, interactions)
    engine = SelfConsistentFieldFull(
        interactions,
        occupation,
        n0,
        numbers=numbers,
        ihelp=system.ihelp,
        cache=interaction_data,
        integrals=integrals,
        config=system.config.scf,
        nel=nel,
    )
    guess = get_guess(
        numbers,
        positions,
        charge_tensor,
        system.ihelp,
        system.config.scf.guess,
    )
    x = engine._guess(guess)
    inputs = None
    if setup_functional:
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
    return SimpleNamespace(
        method=method,
        sample_name=sample_name,
        int_level=int_level,
        field=field,
        par=par,
        calc=calc,
        system=system,
        positions=positions,
        charge=charge_tensor,
        spin=spin_tensor,
        integrals=integrals,
        refocc=refocc,
        n0=n0,
        occupation=occupation,
        nel=nel,
        interactions=interactions,
        interaction_data=interaction_data,
        engine=engine,
        x=x,
        initial_charges=guess,
        inputs=inputs,
        scf_options=scf_options,
    )


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-13)


def _assert_directional_fd(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    gradient: torch.Tensor,
    direction: torch.Tensor | None = None,
    one_sided: int | None = None,
) -> None:
    """Check one response direction against a detected central-FD window."""
    if direction is None:
        direction = gradient.detach()
    direction = direction / torch.linalg.vector_norm(direction)
    automatic = torch.sum(gradient.detach() * direction)
    steps = (1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5)
    finite_differences = []
    for step in steps:
        delta = point.new_tensor(step)
        if one_sided is None:
            difference = function(
                point.detach() + delta * direction
            ) - function(point.detach() - delta * direction)
            denominator = 2.0 * delta
        else:
            assert one_sided in (-1, 1)
            difference = function(
                point.detach() + one_sided * delta * direction
            ) - function(point.detach())
            denominator = one_sided * delta
        finite_differences.append(difference / denominator)
    windows = [
        torch.stack(finite_differences[index : index + 3])
        for index in range(len(finite_differences) - 2)
    ]
    plateau = min(
        windows, key=lambda values: float(values.max() - values.min())
    )
    center = plateau.median()
    spread = plateau.max() - plateau.min()
    assert torch.isfinite(plateau).all()
    assert spread < 5.0e-5 * max(1.0, float(center.abs()))
    assert torch.isclose(
        automatic,
        center,
        rtol=5.0e-6,
        atol=max(8.0 * float(spread), 2.0e-7),
    ), (automatic, center, spread, finite_differences)


def _snapshot_tree(value: object, seen: dict[int, int] | None = None) -> object:
    """Copy tensor values through a small persistent-state object graph."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if value is None or isinstance(value, (str, int, float, bool, torch.dtype)):
        return value
    if seen is None:
        seen = {}
    identity = id(value)
    if identity in seen:
        return ("cycle", seen[identity])
    seen[identity] = len(seen)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (field.name, _snapshot_tree(getattr(value, field.name), seen))
                for field in fields(value)
            ),
        )
    if isinstance(value, dict):
        return tuple(
            (key, _snapshot_tree(item, seen))
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot_tree(item, seen) for item in value)
    slots: list[str] = []
    for cls in type(value).__mro__:
        declared = getattr(cls, "__slots__", ())
        slots.extend((declared,) if isinstance(declared, str) else declared)
    if slots:
        return (
            type(value).__qualname__,
            tuple(
                (name, _snapshot_tree(getattr(value, name), seen))
                for name in dict.fromkeys(slots)
                if name not in ("__dict__", "__weakref__")
                and hasattr(value, name)
            ),
        )
    if hasattr(value, "__dict__"):
        return (
            type(value).__qualname__,
            _snapshot_tree(vars(value), seen),
        )
    return value


def _assert_snapshot_equal(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for current, original in zip(actual, expected):
            _assert_snapshot_equal(current, original)
        return
    assert actual == expected


def _tensor_leaves(
    value: object, seen: set[int] | None = None
) -> tuple[torch.Tensor, ...]:
    """Collect tensor values from call-local interaction data."""
    if isinstance(value, torch.Tensor):
        return (value,)
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return ()
    seen.add(identity)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            leaf
            for field in fields(value)
            for leaf in _tensor_leaves(getattr(value, field.name), seen)
        )
    if isinstance(value, dict):
        return tuple(
            leaf
            for item in value.values()
            for leaf in _tensor_leaves(item, seen)
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            leaf for item in value for leaf in _tensor_leaves(item, seen)
        )
    slots: list[str] = []
    for cls in type(value).__mro__:
        declared = getattr(cls, "__slots__", ())
        slots.extend((declared,) if isinstance(declared, str) else declared)
    if slots:
        return tuple(
            leaf
            for name in dict.fromkeys(slots)
            if name not in ("__dict__", "__weakref__") and hasattr(value, name)
            for leaf in _tensor_leaves(getattr(value, name), seen)
        )
    if hasattr(value, "__dict__"):
        return _tensor_leaves(vars(value), seen)
    return ()


def _persistent_identity_snapshot(system) -> tuple[object, ...]:
    """Record persistent object/tensor identity and tensor state."""
    roots = (
        system.ihelp,
        system.interactions,
        *system.interactions.components,
        system.classicals,
        *system.classicals.components,
        system.es2_setup,
        system.es3_setup,
        system.aes2_setup,
        system.d4sc_setup,
        system.generalized_born_setup,
        system.integral_setup,
        system.h0_setup,
        system._classical_extension_cache,
        system._packed_classical_cache,
        system.d3_setups,
        system.repulsion_setup,
        system.repulsion_classical,
        system.srb_setup,
        system.srb_classical,
        system.d4_setup,
        system.d4_classical,
        system.ies_setup,
        system.ies_classical,
        system.halogen_setup,
        system.halogen_classical,
        system.es2_interaction,
        system.es3_interaction,
        system.aes2_interaction,
        system.d4sc_interaction,
        system.generalized_born_interaction,
    )
    tensor_state: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
    seen: set[int] = set()
    for root in roots:
        for tensor in _tensor_leaves(root, seen):
            tensor_state.append(
                (tensor, tensor.detach().clone(), tensor.requires_grad)
            )
    # Strong references make this an actual identity check and prevent id reuse.
    return roots, tuple(tensor_state)


def _assert_persistent_identity_unchanged(
    system, before: tuple[object, ...]
) -> None:
    after = _persistent_identity_snapshot(system)
    before_roots, before_tensors = before
    after_roots, after_tensors = after
    assert len(after_roots) == len(before_roots)
    for actual, expected in zip(after_roots, before_roots):
        assert actual is expected
    assert len(after_tensors) == len(before_tensors)
    for (actual, actual_value, actual_grad), (
        expected,
        expected_value,
        expected_grad,
    ) in zip(after_tensors, before_tensors):
        assert actual is expected
        assert actual.requires_grad == expected_grad
        assert actual_grad == expected_grad
        torch.testing.assert_close(
            actual_value, expected_value, rtol=0.0, atol=0.0
        )


def _assert_pairwise_distinct(values: tuple[object, ...]) -> None:
    assert all(
        left is not right
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )


def _assert_map_result_equal(actual, expected) -> None:
    """Compare every stable tensor output from two raw physical maps."""
    for name in (
        "mapped_x",
        "hamiltonian",
        "density",
        "emo",
        "coefficients",
        "occupation",
        "orbital_energy",
    ):
        _assert_tensor_equal(getattr(actual, name), getattr(expected, name))

    actual_fermi = actual.fermi_status
    expected_fermi = expected.fermi_status
    if actual_fermi is None:
        assert expected_fermi is None
    else:
        assert expected_fermi is not None
        for name in (
            "converged",
            "residual",
            "invalid_temperature",
            "too_many_electrons",
            "electron_count_ok",
        ):
            _assert_tensor_equal(
                getattr(actual_fermi, name), getattr(expected_fermi, name)
            )

    for name in ("mono", "dipole", "quad"):
        actual_value = getattr(actual.charges, name)
        expected_value = getattr(expected.charges, name)
        if actual_value is None:
            assert expected_value is None
        else:
            assert expected_value is not None
            _assert_tensor_equal(actual_value, expected_value)

    for name in ("mono", "dipole", "quad"):
        actual_value = (
            None
            if actual.driving_potential is None
            else getattr(actual.driving_potential, name)
        )
        expected_value = (
            None
            if expected.driving_potential is None
            else getattr(expected.driving_potential, name)
        )
        if actual_value is None:
            assert expected_value is None
        else:
            assert expected_value is not None
            _assert_tensor_equal(actual_value, expected_value)


@pytest.mark.parametrize(
    "layout,charges",
    [
        pytest.param(
            ChargeLayout((5,)),
            SCFCharges(torch.arange(5, dtype=torch.double)),
            id="monopole",
        ),
        pytest.param(
            ChargeLayout((5,), (2, 3)),
            SCFCharges(
                torch.arange(5, dtype=torch.double),
                torch.arange(6, dtype=torch.double).reshape(2, 3),
            ),
            id="dipole",
        ),
        pytest.param(
            ChargeLayout((5,), (2, 3), (2, 6)),
            SCFCharges(
                torch.arange(5, dtype=torch.double),
                torch.arange(6, dtype=torch.double).reshape(2, 3),
                torch.arange(12, dtype=torch.double).reshape(2, 6),
            ),
            id="quadrupole",
        ),
    ],
)
def test_charge_layout_round_trip(
    layout: ChargeLayout, charges: SCFCharges
) -> None:
    packed = pack_charges(charges, layout)
    unpacked = unpack_charges(packed, layout)
    _assert_tensor_equal(packed, pack_charges(unpacked, layout))
    _assert_tensor_equal(unpacked.mono, charges.mono)
    if charges.dipole is not None:
        assert unpacked.dipole is not None
        _assert_tensor_equal(unpacked.dipole, charges.dipole)
    if charges.quad is not None:
        assert unpacked.quad is not None
        _assert_tensor_equal(unpacked.quad, charges.quad)
    if packed.shape[0] > 1:
        for row, shape in enumerate(layout.shapes):
            length = shape[0]
            for size in shape[1:]:
                length *= size
            if length < packed.shape[-1]:
                assert torch.all(packed[row, length:] == defaults.PADNZ)


def _compare_raw_map(case: SimpleNamespace, x: torch.Tensor) -> None:
    """Compare raw maps built from independent mutable preparations."""
    kwargs: dict[str, object] = {
        "method": case.method,
        "sample_name": case.sample_name,
        "charge": case.charge,
        "spin": case.spin,
        "field": case.field,
        "positions": case.positions.clone(),
        "par": case.par,
        "int_level": case.int_level,
        "scf_options": case.scf_options,
    }
    legacy_case = _prepare_case(**kwargs)
    functional_case = _prepare_case(**kwargs)
    legacy_x = x.clone()
    functional_x = x.clone()
    engine = legacy_case.engine
    expected_x = engine._fcn(legacy_x)
    actual = scf_map(functional_x, functional_case.inputs)

    _assert_tensor_equal(actual.mapped_x, expected_x)
    _assert_tensor_equal(actual.hamiltonian, engine._data.hamiltonian)
    _assert_tensor_equal(actual.density, engine._data.density)
    _assert_tensor_equal(actual.emo, engine._data.evals)
    _assert_tensor_equal(actual.occupation, engine._data.occupation)
    _assert_tensor_equal(actual.orbital_energy, engine._data.energy)
    if actual.fermi_status is not None:
        assert torch.isfinite(actual.fermi_status.residual).all()
        assert bool(actual.fermi_status.converged.all())
        assert bool(actual.fermi_status.electron_count_ok.all())

    expected_charges = engine.converged_to_charges(expected_x)
    _assert_tensor_equal(actual.charges.mono, expected_charges.mono)
    if expected_charges.dipole is not None:
        assert actual.charges.dipole is not None
        _assert_tensor_equal(actual.charges.dipole, expected_charges.dipole)
    if expected_charges.quad is not None:
        assert actual.charges.quad is not None
        _assert_tensor_equal(actual.charges.quad, expected_charges.quad)

    input_charges = engine.converged_to_charges(x)
    expected_potential = engine.charges_to_potential(input_charges)
    _assert_tensor_equal(actual.driving_potential.mono, expected_potential.mono)
    if expected_potential.dipole is not None:
        assert actual.driving_potential.dipole is not None
        _assert_tensor_equal(
            actual.driving_potential.dipole, expected_potential.dipole
        )
    if expected_potential.quad is not None:
        assert actual.driving_potential.quad is not None
        _assert_tensor_equal(
            actual.driving_potential.quad, expected_potential.quad
        )

    # Coefficients are compared only through their occupation-weighted
    # projector, which is invariant to eigenvector column signs.
    projector = torch.einsum(
        "ik,k,jk->ij",
        actual.coefficients,
        actual.occupation.sum(-2),
        actual.coefficients,
    )
    _assert_tensor_equal(projector, actual.density)


def test_gfn1_h2o_raw_map_matches_full_engine_initial_and_perturbed() -> None:
    initial = _prepare_case(GFN1_XTB)
    _compare_raw_map(initial, initial.x)

    perturbed = _prepare_case(GFN1_XTB)
    x = perturbed.x.clone()
    x[0, 0] += 0.025
    _compare_raw_map(perturbed, x)


def test_monopole_only_raw_map_matches_full_engine() -> None:
    case = _prepare_case(GFN1_XTB, int_level=labels.INTLEVEL_HCORE)
    assert case.inputs.charge_layout.dipole_shape is None
    assert case.inputs.charge_layout.quadrupole_shape is None
    _compare_raw_map(case, case.x)


def test_gfn2_h2o_multipole_raw_map_matches_full_engine() -> None:
    case = _prepare_case(GFN2_XTB)
    assert case.inputs.charge_layout.dipole_shape is not None
    assert case.inputs.charge_layout.quadrupole_shape is not None
    _compare_raw_map(case, case.x)


@pytest.mark.parametrize(
    "sample_name,method,charge,spin,temperature",
    [
        pytest.param("H2O", GFN1_XTB, 0.0, 0.0, 300.0, id="gfn1-300K"),
        pytest.param("H2O", GFN1_XTB, 1.0, 1.0, 1000.0, id="open-shell"),
        pytest.param(
            "H2O", GFN1_XTB, 0.25, 0.0, 5000.0, id="fractional-charge"
        ),
        pytest.param(
            "LiH", GFN1_XTB, 0.5, 0.0, 5000.0, id="small-gap-fractional"
        ),
        pytest.param("H2O", GFN2_XTB, 0.0, 0.0, 300.0, id="gfn2-multipoles"),
        pytest.param("NO2", GFN2_XTB, 0.0, 1.0, 300.0, id="gfn2-no2-doublet"),
    ],
)
def test_finite_temperature_raw_map_matches_full_engine(
    sample_name: str,
    method: int,
    charge: float,
    spin: float,
    temperature: float,
) -> None:
    """Finite-T occupation, density and charge map match legacy physics."""
    case = _prepare_case(
        method,
        sample_name=sample_name,
        charge=charge,
        spin=spin,
        scf_options={"fermi_etemp": temperature},
    )
    assert case.inputs.fermi_settings is not None
    _compare_raw_map(case, case.x)


def test_integrated_fermi_evaluation_uses_one_authoritative_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One evaluation shares configured Fermi primal state and status."""
    case = _prepare_case(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        scf_options={
            "fermi_etemp": 300.0,
            "fermi_maxiter": 37,
            "fermi_diff_order": 3,
            "fermi_thresh": 2.0e-12,
        },
    )
    settings = case.inputs.fermi_settings
    assert settings is not None
    original = filling._get_fermi_occupation_core_state
    calls: list[tuple[tuple[object, ...], dict[str, object], object]] = []

    def record(*args: object, **kwargs: object):
        state = original(*args, **kwargs)
        calls.append((args, kwargs, state))
        return state

    monkeypatch.setattr(filling, "_get_fermi_occupation_core_state", record)
    spectral = evaluate_hamiltonian(case.inputs.integrals.hcore, case.inputs)

    assert len(calls) == 1
    args, kwargs, state = calls[0]
    assert args[0] is case.inputs.nel
    assert args[1].shape[-2:] == (2, case.inputs.orbital_mask.shape[-1])
    assert args[2] is settings.kt
    assert kwargs["mask"].shape == (
        *case.inputs.nel.shape,
        *case.inputs.orbital_mask.shape,
    )
    assert torch.equal(
        kwargs["mask"], case.inputs.orbital_mask.expand_as(kwargs["mask"])
    )
    assert kwargs["thr"] is settings.threshold
    assert kwargs["maxiter"] == settings.maxiter == 37
    assert kwargs["diff_order"] == settings.diff_order == 3
    assert spectral.fermi_status is state.result.status
    torch.testing.assert_close(
        spectral.occupation.sum(-1), case.inputs.nel, atol=1.0e-4, rtol=0.0
    )
    assert torch.isfinite(spectral.density).all()


def test_no2_integrated_fermi_channels_share_density_response() -> None:
    """NO2 alpha/beta filling and invariant density retain channel semantics."""
    case = _prepare_case(
        GFN2_XTB,
        sample_name="NO2",
        spin=1.0,
        scf_options={"fermi_etemp": 300.0, "fermi_diff_order": 3},
    )
    inputs = case.inputs
    assert inputs.fermi_settings is not None
    h0 = inputs.integrals.hcore
    direction = torch.zeros_like(h0)
    direction[0, 1] = direction[1, 0] = 0.2
    weight = torch.zeros_like(h0)
    weight[0, 1] = weight[1, 0] = 0.5

    def scalar(parameter: torch.Tensor) -> torch.Tensor:
        result = evaluate_hamiltonian(h0 + parameter * direction, inputs)
        assert result.fermi_status is not None
        torch.testing.assert_close(
            result.occupation.sum(-1), inputs.nel, atol=1.0e-4, rtol=0.0
        )
        return torch.sum(weight * result.density)

    parameter = torch.zeros((), dtype=h0.dtype, requires_grad=True)
    primal = evaluate_hamiltonian(h0, inputs)
    assert primal.fermi_status is not None
    assert primal.occupation.shape[-2] == 2
    assert torch.allclose(
        primal.occupation.sum(-1), inputs.nel, atol=1.0e-4, rtol=0.0
    )
    reverse = torch.autograd.grad(scalar(parameter), parameter)[0]
    _, tangent = jvp(
        scalar, (parameter.detach(),), (torch.ones_like(parameter),)
    )
    forward = jacfwd(scalar)(parameter.detach())
    torch.testing.assert_close(reverse, tangent, atol=1.0e-7, rtol=1.0e-7)
    torch.testing.assert_close(reverse, forward, atol=1.0e-7, rtol=1.0e-7)


def test_integrated_fermi_exact_degeneracy_is_rotation_invariant() -> None:
    """The full filling-to-density path is smooth at finite-T degeneracy."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={"fermi_etemp": 300.0, "fermi_diff_order": 3},
    )
    n = case.inputs.integrals.hcore.shape[-1]
    identity = torch.eye(n, dtype=case.positions.dtype)
    inputs = replace(
        case.inputs,
        inverse_cholesky=identity,
        integrals=replace(case.inputs.integrals, overlap=identity),
    )
    h0 = torch.zeros_like(inputs.integrals.hcore)
    direction = torch.zeros_like(h0)
    direction[0, 0] = 1.0
    direction[1, 1] = -1.0
    direction[0, 1] = direction[1, 0] = 0.2
    weight = torch.zeros_like(h0)
    weight[0, 0] = 0.3
    weight[1, 1] = -0.1
    weight[0, 1] = weight[1, 0] = 0.4

    def density(matrix: torch.Tensor) -> torch.Tensor:
        spectral = evaluate_hamiltonian(matrix, inputs)
        assert spectral.fermi_status is not None
        torch.testing.assert_close(
            spectral.occupation.sum(-1), inputs.nel, atol=1e-4, rtol=0.0
        )
        torch.testing.assert_close(
            torch.sum(spectral.density * inputs.integrals.overlap.mT),
            inputs.nel.sum(),
            atol=2e-10,
            rtol=0.0,
        )
        return spectral.density

    parameter = torch.zeros((), dtype=h0.dtype, requires_grad=True)
    response = lambda value: torch.sum(weight * density(h0 + value * direction))
    reverse = torch.autograd.grad(response(parameter), parameter)[0]
    _, tangent = jvp(
        response, (parameter.detach(),), (torch.ones_like(parameter),)
    )
    forward = jacfwd(response)(parameter.detach())
    torch.testing.assert_close(reverse, tangent, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(reverse, forward, atol=2e-8, rtol=2e-8)
    step = h0.new_tensor(1.0e-7)
    finite_difference = (response(step) - response(-step)) / (2.0 * step)
    torch.testing.assert_close(reverse, finite_difference, atol=2e-5, rtol=2e-5)

    angle = h0.new_tensor(0.71)
    rotation = torch.eye(n, dtype=h0.dtype)
    rotation[:2, :2] = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    rotated_direction = rotation @ direction @ rotation.mT
    rotated_weight = rotation @ weight @ rotation.mT

    def rotated_density_response(value: torch.Tensor) -> torch.Tensor:
        result = evaluate_hamiltonian(
            rotation @ (h0 + value * direction) @ rotation.mT, inputs
        )
        return torch.sum(rotated_weight * result.density)

    rotated_parameter = torch.zeros((), dtype=h0.dtype, requires_grad=True)
    rotated_first = torch.autograd.grad(
        rotated_density_response(rotated_parameter), rotated_parameter
    )[0]
    torch.testing.assert_close(rotated_first, reverse, atol=2e-8, rtol=2e-8)


def test_integrated_fractional_aufbau_response_matches_legacy_density() -> None:
    """Production evaluation preserves fractional-channel density response."""
    case = _prepare_case(GFN1_XTB, charge=-0.25)
    inputs = case.inputs
    h0 = inputs.integrals.hcore
    direction = torch.diag(
        torch.linspace(-0.1, 0.1, h0.shape[-1], dtype=h0.dtype)
    )
    parameter = torch.zeros((), dtype=h0.dtype, requires_grad=True)

    def density(value: torch.Tensor) -> torch.Tensor:
        return evaluate_hamiltonian(h0 + value * direction, inputs).density

    spectral = evaluate_hamiltonian(h0, inputs)
    legacy = get_density(spectral.coefficients, inputs.occupation.sum(-2))
    torch.testing.assert_close(
        spectral.density, legacy, atol=2.0e-10, rtol=2.0e-10
    )
    torch.testing.assert_close(
        torch.sum(spectral.density * inputs.integrals.overlap.mT),
        inputs.nel.sum(),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    reverse = torch.autograd.grad(
        torch.sum(density(parameter) ** 2), parameter
    )[0]
    _, tangent = jvp(
        lambda value: torch.sum(density(value) ** 2),
        (parameter.detach(),),
        (torch.ones_like(parameter),),
    )
    forward = jacfwd(lambda value: torch.sum(density(value) ** 2))(
        parameter.detach()
    )
    torch.testing.assert_close(reverse, tangent, atol=2.0e-8, rtol=2.0e-8)
    torch.testing.assert_close(reverse, forward, atol=2.0e-8, rtol=2.0e-8)


def test_physical_scf_outputs_do_not_depend_on_reporting_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detaching reporting coefficients leaves physical map responses intact."""
    case = _prepare_case(
        GFN1_XTB, scf_options={"fermi_etemp": 300.0, "fermi_diff_order": 3}
    )
    original = functional_map._transitional_diagonalize
    direction = torch.zeros_like(case.inputs.integrals.hcore)
    direction[0, 1] = direction[1, 0] = 0.15
    parameter = torch.zeros((), dtype=direction.dtype, requires_grad=True)

    def run(detach_coefficients: bool) -> tuple[torch.Tensor, torch.Tensor]:
        def diagonalize(hamiltonian: torch.Tensor, inputs: object):
            energies, coefficients = original(hamiltonian, inputs)
            if detach_coefficients:
                coefficients = coefficients.detach()
            return energies, coefficients

        monkeypatch.setattr(
            functional_map, "_transitional_diagonalize", diagonalize
        )
        integrals = replace(
            case.inputs.integrals,
            hcore=case.inputs.integrals.hcore + parameter * direction,
        )
        inputs = replace(case.inputs, integrals=integrals)
        result = scf_map(case.x, inputs)
        spectral = result.spectral
        value = (
            torch.sum(spectral.density**2)
            + torch.sum(spectral.charges.mono**2)
            + torch.sum(result.mapped_x**2)
            + spectral.orbital_energy.sum()
        )
        derivative = torch.autograd.grad(value, parameter)[0]
        return value.detach(), derivative.detach()

    connected = run(False)
    detached = run(True)
    torch.testing.assert_close(
        connected[0], detached[0], atol=2e-10, rtol=2e-10
    )
    torch.testing.assert_close(connected[1], detached[1], atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize(
    "partition",
    [labels.FERMI_PARTITION_EQUAL, labels.FERMI_PARTITION_ATOMIC],
)
def test_free_energy_does_not_depend_on_reporting_coefficients(
    partition: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free-energy value and derivatives are owned by the invariant matrix."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "fermi_etemp": 300.0,
            "fermi_partition": partition,
            "fermi_diff_order": 3,
        },
    )
    original = functional_map._transitional_diagonalize
    h0 = case.inputs.integrals.hcore
    direction = torch.zeros_like(h0)
    direction[0, 1] = direction[1, 0] = 0.15

    def evaluate(parameter: torch.Tensor, detach: bool) -> torch.Tensor:
        def diagonalize(hamiltonian: torch.Tensor, inputs: object):
            energies, coefficients = original(hamiltonian, inputs)
            return energies, coefficients.detach() if detach else coefficients

        monkeypatch.setattr(
            functional_map, "_transitional_diagonalize", diagonalize
        )
        integral = replace(
            case.inputs.integrals, hcore=h0 + parameter * direction
        )
        inputs = replace(case.inputs, integrals=integral)
        spectral = evaluate_hamiltonian(integral.hcore, inputs)
        return free_energy_from_spectral(spectral, inputs).sum()

    parameter = torch.zeros((), dtype=h0.dtype, requires_grad=True)
    value = evaluate(parameter, False)
    first = torch.autograd.grad(value, parameter, create_graph=True)[0]
    second = torch.autograd.grad(first, parameter, create_graph=True)[0]
    third = torch.autograd.grad(second, parameter)[0]
    reverse = torch.autograd.grad(evaluate(parameter, False), parameter)[0]
    detached_parameter = parameter.detach().requires_grad_()
    detached_value = evaluate(detached_parameter, True)
    detached_first = torch.autograd.grad(detached_value, detached_parameter)[0]
    _, tangent = jvp(
        lambda p: evaluate(p, True),
        (parameter.detach(),),
        (torch.ones_like(parameter),),
    )
    forward = jacfwd(lambda p: evaluate(p, True))(parameter.detach())
    assert torch.isfinite(torch.stack([first, second, third])).all()
    torch.testing.assert_close(value.detach(), detached_value.detach())
    torch.testing.assert_close(
        first.detach(), detached_first, atol=2e-8, rtol=2e-8
    )
    torch.testing.assert_close(reverse, tangent, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(reverse, forward, atol=2e-8, rtol=2e-8)


def test_map_uses_input_potential_and_output_charge_potential_is_distinct() -> (
    None
):
    case = _prepare_case(GFN1_XTB)
    x = case.x.clone()
    x[0, 0] += 0.4
    result = scf_map(x, case.inputs)
    output_potential = potential_from_charges(result.charges, case.inputs)

    assert result.driving_potential.mono is not None
    assert output_potential.mono is not None
    difference = torch.max(
        torch.abs(result.driving_potential.mono - output_potential.mono)
    )
    assert difference > 1.0e-5


def test_full_finalization_recomputes_potential_from_returned_charges() -> None:
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "maxiter": 1,
            "mix_guess": False,
            "x_atol": 1.0e5,
            "x_atol_max": 1.0e5,
        },
    )
    engine = case.engine
    guess = case.x.clone()
    guess[0, 0] += 0.4
    initial_charges = engine.converged_to_charges(guess)
    original_iterate = engine.iterate_charges
    raw_calls: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            SCFPotential,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []

    def record_raw_call(x: torch.Tensor) -> torch.Tensor:
        input_charges = engine.converged_to_charges(x)
        potential = engine.charges_to_potential(input_charges)
        driving = SCFPotential(
            potential.mono,
            dipole=potential.dipole,
            quad=potential.quad,
        )
        mapped = original_iterate(x)
        raw_calls.append(
            (
                x.clone(),
                mapped.clone(),
                driving,
                engine._data.hamiltonian.clone(),
                engine._data.density.clone(),
                engine._data.evals.clone(),
            )
        )
        return mapped

    engine.iterate_charges = record_raw_call
    final = engine(charges=initial_charges)
    assert len(raw_calls) == 2

    (
        input_x,
        last_mapped_x,
        last_driving_potential,
        last_hamiltonian,
        last_density,
        last_emo,
    ) = raw_calls[-1]
    final_x = final["charges"].as_tensor()
    _assert_tensor_equal(final_x, last_mapped_x)

    # The returned spectral outputs belong to the last raw map evaluation.
    _assert_tensor_equal(final["hamiltonian"], last_hamiltonian)
    _assert_tensor_equal(final["density"], last_density)
    _assert_tensor_equal(final["emo"], last_emo)

    # Finalization deliberately evaluates the potential and interaction
    # energy from G(x), rather than reusing the V(x) that drove the last map.
    final_potential = final["potential"]
    expected_potential = engine.charges_to_potential(final["charges"])
    _assert_tensor_equal(final_potential.mono, expected_potential.mono)
    assert final_potential.mono is not None
    assert last_driving_potential.mono is not None
    assert (
        torch.max(torch.abs(final_potential.mono - last_driving_potential.mono))
        > 1.0e-5
    )

    h0_energy = engine._data.ihelp.reduce_orbital_to_atom(engine._data.energy)
    interaction_energy = engine.interactions.get_energy(
        final["charges"], engine._data.cache, engine._data.ihelp
    )
    _assert_tensor_equal(final["energy"], h0_energy + interaction_energy)
    input_charges = engine.converged_to_charges(input_x)
    input_interaction_energy = engine.interactions.get_energy(
        input_charges, engine._data.cache, engine._data.ihelp
    )
    assert (
        torch.max(torch.abs(interaction_energy - input_interaction_energy))
        > 1.0e-8
    )
    assert input_x.shape == final_x.shape


def test_static_scf_configuration_is_validated_at_setup() -> None:
    case = _prepare_case(GFN1_XTB)
    assert not hasattr(case.inputs, "config")
    baseline = scf_map(case.x, case.inputs)

    potential_config = replace(
        case.system.config.scf, scp_mode=labels.SCP_MODE_POTENTIAL
    )
    potential_inputs = setup_scf_inputs(
        case.system.numbers,
        case.integrals,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        potential_config,
    )
    _assert_tensor_equal(
        scf_map(case.x, potential_inputs).mapped_x, baseline.mapped_x
    )

    warm_config = replace(
        case.system.config.scf,
        fermi=replace(case.system.config.scf.fermi, etemp=1000.0),
    )
    warm_inputs = setup_scf_inputs(
        case.system.numbers,
        case.integrals,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        warm_config,
    )
    assert warm_inputs.fermi_settings is not None
    with pytest.raises(ValueError, match="non-negative"):
        setup_scf_inputs(
            case.system.numbers,
            case.integrals,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            replace(
                case.system.config.scf,
                fermi=replace(case.system.config.scf.fermi, etemp=-1.0),
            ),
        )
    with pytest.raises(ValueError, match="maxiter"):
        setup_scf_inputs(
            case.system.numbers,
            case.integrals,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            replace(
                warm_config,
                fermi=replace(warm_config.fermi, maxiter=-1),
            ),
        )

    iterates = torch.stack((case.x, case.x + 0.01))
    before_config_change = vmap(lambda xx: scf_map(xx, case.inputs).mapped_x)(
        iterates
    )
    # ConfigSCF is frozen, so a changed configuration is a replacement. The
    # legacy engine's replacement is the mutable authority boundary here;
    # SCFInputs does not retain that config or the engine.
    case.engine.config = replace(
        case.engine.config,
        scp_mode=labels.SCP_MODE_POTENTIAL,
        fermi=replace(case.engine.config.fermi, etemp=250.0),
    )
    after_config_change = vmap(lambda xx: scf_map(xx, case.inputs).mapped_x)(
        iterates
    )
    _assert_tensor_equal(after_config_change, before_config_change)
    after_config_change = vmap(lambda xx: scf_map(xx, case.inputs).mapped_x)(
        iterates
    )
    _assert_tensor_equal(after_config_change, before_config_change)


def _initial_potential_guess(
    case: SimpleNamespace,
) -> tuple[torch.Tensor, PotentialLayout]:
    layout = case.inputs.charge_layout
    charges = SCFCharges(
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
    return potential_initial_guess(charges, case.inputs)


def _legacy_density_charges(case: SimpleNamespace) -> SCFCharges:
    density = case.engine._data.density
    mono = case.n0 - torch.einsum("ik,ki->i", density, case.integrals.overlap)
    dipole = None
    if case.integrals.dipole is not None:
        dipole = case.system.ihelp.reduce_orbital_to_atom(
            -torch.einsum("ik,mki->im", density, case.integrals.dipole),
            extra=True,
            dim=-2,
        )
    quad = None
    if case.integrals.quadrupole is not None:
        quad = case.system.ihelp.reduce_orbital_to_atom(
            -torch.einsum("ik,mki->im", density, case.integrals.quadrupole),
            extra=True,
            dim=-2,
        )
    return SCFCharges(mono=mono, dipole=dipole, quad=quad)


@pytest.mark.parametrize(
    "layout,potential",
    [
        (
            PotentialLayout((5,)),
            SCFPotential(mono=torch.arange(5, dtype=torch.double)),
        ),
        (
            PotentialLayout((5,), (2, 3)),
            SCFPotential(
                mono=torch.arange(5, dtype=torch.double),
                dipole=torch.arange(6, dtype=torch.double).reshape(2, 3),
            ),
        ),
        (
            PotentialLayout((5,), (2, 3), (2, 6)),
            SCFPotential(
                mono=torch.arange(5, dtype=torch.double),
                dipole=torch.arange(6, dtype=torch.double).reshape(2, 3),
                quad=torch.arange(12, dtype=torch.double).reshape(2, 6),
            ),
        ),
    ],
)
def test_potential_layout_round_trip_and_transforms(
    layout: PotentialLayout, potential: SCFPotential
) -> None:
    """Potential packing is fixed-shape and composes with AD transforms."""
    packed = pack_potential(potential, layout)
    unpacked = unpack_potential(packed, layout)
    _assert_tensor_equal(pack_potential(unpacked, layout), packed)
    _assert_tensor_equal(unpacked.mono, potential.mono)
    if potential.dipole is not None:
        assert unpacked.dipole is not None
        _assert_tensor_equal(unpacked.dipole, potential.dipole)
    if potential.quad is not None:
        assert unpacked.quad is not None
        _assert_tensor_equal(unpacked.quad, potential.quad)

    def roundtrip(value: torch.Tensor) -> torch.Tensor:
        return pack_potential(unpack_potential(value, layout), layout)

    physical = torch.zeros(packed.numel(), dtype=torch.bool)
    width = packed.shape[-1]
    for row, shape in enumerate(layout.shapes):
        physical[
            row * width : row * width + int(torch.tensor(shape).prod())
        ] = True
    expected_jacobian = torch.diag(physical.to(packed.dtype))
    jacobian = jacrev(roundtrip)(packed).reshape(packed.numel(), packed.numel())
    torch.testing.assert_close(jacobian, expected_jacobian)
    tangent = torch.randn_like(packed)
    primal, tangent_out = jvp(roundtrip, (packed,), (tangent,))
    _assert_tensor_equal(primal, packed)
    _assert_tensor_equal(
        tangent_out, (tangent.reshape(-1) * physical).reshape_as(tangent)
    )
    _assert_tensor_equal(jacfwd(roundtrip)(packed), jacrev(roundtrip)(packed))
    batch = torch.stack((packed, packed + 0.25, packed - 0.5))
    _assert_tensor_equal(
        vmap(roundtrip)(batch), torch.stack([roundtrip(item) for item in batch])
    )


def test_potential_layout_rejects_incompatible_components() -> None:
    """Potential packing rejects static row and component mismatches."""
    mono_layout = PotentialLayout((5,))
    mono = torch.zeros(5, dtype=torch.double)
    dipole = torch.zeros((2, 3), dtype=torch.double)
    with pytest.raises(ValueError, match="Unexpected dipole"):
        pack_potential(SCFPotential(mono=mono, dipole=dipole), mono_layout)
    with pytest.raises(ValueError, match="Dipole potential is required"):
        pack_potential(SCFPotential(mono=mono), PotentialLayout((5,), (2, 3)))
    with pytest.raises(ValueError, match="shape"):
        pack_potential(
            SCFPotential(mono=mono, dipole=torch.zeros((3, 2))),
            PotentialLayout((5,), (2, 3)),
        )
    with pytest.raises(ValueError, match="Quadrupole potential requires"):
        PotentialLayout((5,), quadrupole_shape=(2, 6))
    with pytest.raises(ValueError, match="does not match the configured"):
        unpack_potential(torch.zeros((2, 5)), mono_layout)
    with pytest.raises(ValueError, match="Packed potential width"):
        unpack_potential(torch.zeros((1, 4)), mono_layout)


def _assert_potential_raw_parity(
    method: int = GFN1_XTB,
    *,
    charge: float = 0.0,
    spin: float = 0.0,
    field: torch.Tensor | None = None,
    temperature: float = 0.0,
    row: int | None = None,
) -> None:
    options = {
        "scp_mode": labels.SCP_MODE_POTENTIAL,
        "mixer": labels.MIXER_LINEAR,
        "maxiter": 0,
        "fermi_etemp": temperature,
    }
    legacy = _prepare_case(
        method,
        charge=charge,
        spin=spin,
        field=field,
        scf_options=options,
        setup_functional=False,
    )
    functional = _prepare_case(
        method,
        charge=charge,
        spin=spin,
        field=field,
        scf_options=options,
    )
    guess, layout = _initial_potential_guess(functional)
    _assert_tensor_equal(guess, functional.x)
    _assert_tensor_equal(guess, legacy.x)
    if row is not None:
        perturbed = guess.clone()
        perturbed[row, 0] += 0.037
        guess = perturbed
    legacy_x = guess.clone()
    mapped_legacy = legacy.engine.iterate_potential(legacy_x)
    expected = potential_map(guess, functional.inputs, layout)
    _assert_tensor_equal(expected.mapped_x, mapped_legacy)
    for row, shape in enumerate(layout.shapes):
        length = 1
        for size in shape:
            length *= size
        if length < guess.shape[-1]:
            assert torch.all(guess[row, length:] == defaults.PADNZ)
            assert torch.all(mapped_legacy[row, length:] == defaults.PADNZ)
            assert torch.all(
                mapped_legacy[row, length:] - guess[row, length:] == 0.0
            )
    for attr, value in (
        ("hamiltonian", legacy.engine._data.hamiltonian),
        ("density", legacy.engine._data.density),
        ("emo", legacy.engine._data.evals),
        ("coefficients", legacy.engine._data.evecs),
        ("occupation", legacy.engine._data.occupation),
        ("orbital_energy", legacy.engine._data.energy),
    ):
        _assert_tensor_equal(getattr(expected, attr), value)
    expected_charges = _legacy_density_charges(legacy)
    _assert_tensor_equal(expected.charges.mono, expected_charges.mono)
    if expected.charges.dipole is not None:
        _assert_tensor_equal(expected.charges.dipole, expected_charges.dipole)
    if expected.charges.quad is not None:
        _assert_tensor_equal(expected.charges.quad, expected_charges.quad)
    if temperature >= 300.0:
        assert expected.fermi_status is not None
        assert bool(expected.fermi_status.converged.all())


@pytest.mark.parametrize(
    "method,charge,spin,field,temperature",
    [
        (GFN1_XTB, 0.0, 0.0, None, 0.0),
        (GFN1_XTB, 1.0, 1.0, None, 0.0),
        (
            GFN1_XTB,
            0.0,
            0.0,
            torch.tensor([0.01, 0.02, -0.01], dtype=torch.float64),
            0.0,
        ),
        (GFN1_XTB, 0.5, 0.0, None, 300.0),
        (GFN2_XTB, 0.0, 0.0, None, 0.0),
    ],
)
def test_potential_map_matches_legacy_physical_evaluation(
    method: int,
    charge: float,
    spin: float,
    field: torch.Tensor | None,
    temperature: float,
) -> None:
    """Potential raw map matches an independently prepared legacy map."""
    _assert_potential_raw_parity(
        method, charge=charge, spin=spin, field=field, temperature=temperature
    )


@pytest.mark.parametrize("row", [0, 1, 2], ids=["mono", "dipole", "quad"])
def test_gfn2_potential_map_uses_each_packed_row(row: int) -> None:
    """Each GFN2 potential row affects the next physical map."""
    _assert_potential_raw_parity(GFN2_XTB, row=row)


class _MutatingPotentialInteraction(Interaction):
    label = "MutatingPotentialInteraction"

    def __init__(self) -> None:
        super().__init__(device=torch.device("cpu"), dtype=torch.float64)
        self.calls = 0

    def get_monopole_atom_potential(
        self,
        cache: InteractionCache,
        qat: torch.Tensor,
        qdp: torch.Tensor | None = None,
        qqp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(qat)


def test_functional_eligibility_rejects_custom_mutable_interactions() -> None:
    case = _prepare_case(GFN1_XTB)
    custom = _MutatingPotentialInteraction()
    interactions = InteractionList(custom, device="cpu", dtype=torch.float64)

    # Ordinary interaction construction remains available to the legacy path.
    assert len(interactions.components) == 1
    assert interactions.components[0] is custom
    with pytest.raises(TypeError, match="exact migrated built-in"):
        validate_functional_scf_interactions(interactions)
    assert custom.calls == 0
    with pytest.raises(TypeError, match="exact migrated built-in"):
        setup_scf_inputs(
            case.system.numbers,
            case.integrals,
            case.n0,
            case.occupation,
            case.nel,
            interactions,
            case.interaction_data,
            case.system.ihelp,
            case.system.config.scf,
        )


def test_charged_spin_raw_map_matches_full_engine() -> None:
    case = _prepare_case(GFN1_XTB, charge=1.0, spin=1.0)
    _compare_raw_map(case, case.x)


def test_explicit_field_raw_map_matches_full_engine() -> None:
    sample_positions = samples["H2O"]["positions"].to(
        device="cpu", dtype=torch.float64
    )
    field = torch.tensor([0.002, -0.001, 0.0015], dtype=torch.float64)
    case = _prepare_case(GFN1_XTB, field=field, positions=sample_positions)
    _compare_raw_map(case, case.x)


def test_map_is_history_independent_and_repeated_backward_is_fresh() -> None:
    case = _prepare_case(GFN1_XTB)
    x_a = case.x.clone()
    x_b = case.x.clone()
    x_b[0, 0] += 0.02
    result_a = scf_map(x_a, case.inputs)
    _ = scf_map(x_b, case.inputs)
    result_a_again = scf_map(x_a.clone(), case.inputs)
    _assert_tensor_equal(result_a.mapped_x, result_a_again.mapped_x)
    _assert_tensor_equal(result_a.hamiltonian, result_a_again.hamiltonian)

    leaf = case.x.clone().requires_grad_(True)
    grad1 = torch.autograd.grad(
        scf_map(leaf, case.inputs).mapped_x.square().sum(), leaf
    )[0]
    grad2 = torch.autograd.grad(
        scf_map(leaf, case.inputs).mapped_x.square().sum(), leaf
    )[0]
    _assert_tensor_equal(grad1, grad2)


def test_map_position_derivatives_and_iterate_vmap() -> None:
    case = _prepare_case(GFN1_XTB)
    x = case.x.clone()

    def scalar(pos: torch.Tensor) -> torch.Tensor:
        local = _prepare_case(GFN1_XTB, positions=pos)
        return scf_map(x, local.inputs).mapped_x.square().sum()

    position = case.positions.clone().requires_grad_(True)
    grad1 = torch.autograd.grad(scalar(position), position, create_graph=True)[
        0
    ]
    grad2 = torch.autograd.grad(grad1.sum(), position)[0]
    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()

    # JVP/jacfwd characterize the unchanged transitional spectral primitive.
    # Its broadening selection is based on requires_grad, so forward- and
    # reverse-mode derivatives are not required to agree until E3.
    direction = torch.ones_like(x) / x.numel() ** 0.5
    scalar = lambda xx: scf_map(xx, case.inputs).mapped_x.sum()
    _, tangent = jvp(scalar, (x,), (direction,))
    reverse_leaf = x.clone().requires_grad_(True)
    (reverse_gradient,) = torch.autograd.grad(
        scalar(reverse_leaf), reverse_leaf
    )
    forward_jacobian = jacfwd(scalar)(x)
    assert torch.isfinite(tangent)
    assert torch.isfinite(reverse_gradient).all()
    assert torch.isfinite(forward_jacobian).all()

    iterates = torch.stack((x, x + 0.01, x - 0.015))
    mapped = vmap(lambda xx: scf_map(xx, case.inputs).mapped_x)(iterates)
    looped = torch.stack([scf_map(xx, case.inputs).mapped_x for xx in iterates])
    _assert_tensor_equal(mapped, looped)


def test_forward_eigensystem_uses_one_fixed_derivative_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward eigensystem reporting no longer selects semantics from AD mode."""
    case = _prepare_case(GFN1_XTB)
    hamiltonian = scf_map(case.x, case.inputs).hamiltonian.detach()
    direction = torch.ones_like(hamiltonian)
    original = storch.eighb
    observed: list[str | None] = []

    def record(*args, **kwargs):
        observed.append(kwargs.get("broadening_method"))
        return original(*args, **kwargs)

    monkeypatch.setattr(storch, "eighb", record)
    _transitional_diagonalize(hamiltonian, case.inputs)
    reverse_leaf = hamiltonian.clone().requires_grad_(True)
    energy, _ = _transitional_diagonalize(reverse_leaf, case.inputs)
    torch.autograd.grad(energy.sum(), reverse_leaf)
    jvp(
        lambda value: _transitional_diagonalize(value, case.inputs)[0].sum(),
        (hamiltonian,),
        (direction,),
    )
    jacfwd(lambda value: _transitional_diagonalize(value, case.inputs)[0])(
        hamiltonian
    )
    assert observed == [None, None, None, None]


def test_map_parameter_graph_reaches_param_module() -> None:
    par = ParamModule(GFN1_XTB, device="cpu", dtype=torch.float64)
    parameter = par.get("element.O.levels")
    parameter.requires_grad_(True)
    case = _prepare_case(GFN1_XTB, par=par)
    value = scf_map(case.x, case.inputs).mapped_x.square().sum()
    gradient = torch.autograd.grad(value, parameter)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_map_inputs_are_not_mutated_and_values_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gfn2_par = ParamModule(GFN2_XTB, device="cpu", dtype=torch.float64)
    case = _prepare_case(GFN2_XTB, par=gfn2_par)
    inputs = case.inputs
    tensor_values = [
        inputs.numbers.clone(),
        inputs.integrals.hcore.clone(),
        inputs.integrals.overlap.clone(),
        inputs.n0.clone(),
        inputs.nel.clone(),
        inputs.occupation.clone(),
        inputs.inverse_cholesky.clone(),
        inputs.orbital_mask.clone(),
    ]
    integrals = (
        inputs.integrals.hcore,
        inputs.integrals.overlap,
        inputs.integrals.dipole,
        inputs.integrals.quadrupole,
    )
    integral_values = tuple(
        None if value is None else value.clone() for value in integrals
    )
    legacy_interactions = case.interactions
    legacy_components = tuple(legacy_interactions.components)
    cache_entries = tuple(case.interaction_data.items())
    legacy_setup_snapshot = _snapshot_tree(
        (
            legacy_interactions,
            case.interaction_data,
            case.system.ihelp,
            inputs.integrals,
        )
    )
    inputs = setup_scf_inputs(
        case.system.numbers,
        case.integrals,
        case.n0,
        case.occupation,
        case.nel,
        case.interactions,
        case.interaction_data,
        case.system.ihelp,
        case.system.config.scf,
    )
    _assert_snapshot_equal(
        _snapshot_tree(
            (
                legacy_interactions,
                case.interaction_data,
                case.system.ihelp,
                case.integrals,
            )
        ),
        legacy_setup_snapshot,
    )
    cache_tensors = {
        key: {
            name: value.clone()
            for name in ("cn", "dispmat")
            if isinstance((value := getattr(cache, name, None)), torch.Tensor)
        }
        for key, cache in cache_entries
    }
    nested_model_tensors: dict[str, torch.Tensor] = {}
    nested_model_values: dict[str, object] = {}
    for key, cache in cache_entries:
        model = getattr(cache, "model", None)
        if model is None:
            continue
        for name in ("numbers", "wf", "rc6"):
            value = getattr(model, name)
            if isinstance(value, torch.Tensor):
                nested_model_tensors[name] = value.clone()
            else:
                nested_model_values[name] = value
        for name in ("ga", "gc", "ref_charges"):
            nested_model_values[name] = getattr(model, name)
    assert {"numbers", "wf", "rc6"} <= nested_model_tensors.keys()
    index_state = {
        name: getattr(case.system.ihelp, name)
        for name in case.system.ihelp.__slots__
    }
    index_tensors = {
        name: value.clone()
        for name, value in index_state.items()
        if isinstance(value, torch.Tensor)
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("The raw map invoked a legacy state operation.")

    monkeypatch.setattr(Component, "update", forbidden)
    monkeypatch.setattr(Component, "reset", forbidden)
    monkeypatch.setattr(Interaction, "update", forbidden)
    monkeypatch.setattr(Interaction, "reset", forbidden)
    monkeypatch.setattr(InteractionList, "update", forbidden)
    monkeypatch.setattr(InteractionList, "reset", forbidden)
    monkeypatch.setattr(InteractionList, "reset_all", forbidden)
    monkeypatch.setattr(InteractionListCache, "cull", forbidden)
    monkeypatch.setattr(InteractionListCache, "restore", forbidden)
    for value in (
        case.system.ihelp,
        legacy_interactions,
        case.interaction_data,
        *legacy_components,
        *(cache for _, cache in cache_entries),
    ):
        for name in ("update", "reset", "reset_all", "cull", "restore"):
            if hasattr(type(value), name):
                monkeypatch.setattr(type(value), name, forbidden)

    assert isinstance(inputs.index, CoreIndexData)
    assert isinstance(inputs.interactions, FunctionalInteractionSet)
    assert not hasattr(inputs, "ihelp")
    assert not hasattr(inputs, "interaction_data")
    assert not hasattr(inputs.interactions, "components")
    assert not hasattr(inputs.interactions, "update")
    assert not hasattr(inputs.interactions, "reset")
    assert "IndexHelper" not in repr(inputs.__dataclass_fields__["index"].type)
    assert "InteractionList" not in repr(
        inputs.__dataclass_fields__["interactions"].type
    )
    core_tensor_values = tuple(
        (path, value.clone()) for path, value in tensor_leaves(inputs)
    )

    first = scf_map(case.x, inputs)
    second = scf_map(case.x, inputs)
    _assert_tensor_equal(first.mapped_x, second.mapped_x)
    for actual, expected in zip(
        (
            inputs.numbers,
            inputs.integrals.hcore,
            inputs.integrals.overlap,
            inputs.n0,
            inputs.nel,
            inputs.occupation,
            inputs.inverse_cholesky,
            inputs.orbital_mask,
        ),
        tensor_values,
    ):
        _assert_tensor_equal(actual, expected)
    assert all(
        actual is expected
        for actual, expected in zip(
            integrals,
            (
                inputs.integrals.hcore,
                inputs.integrals.overlap,
                inputs.integrals.dipole,
                inputs.integrals.quadrupole,
            ),
        )
    )
    for actual, expected in zip(integrals, integral_values):
        if expected is not None:
            assert actual is not None
            _assert_tensor_equal(actual, expected)
    assert tuple(case.interactions.components) == legacy_components
    assert tuple(key for key, _ in case.interaction_data.items()) == tuple(
        key for key, _ in cache_entries
    )
    assert all(
        case.interaction_data[key] is value for key, value in cache_entries
    )
    for key, cache in cache_entries:
        for name, expected in cache_tensors[key].items():
            _assert_tensor_equal(getattr(cache, name), expected)
    d4_model = case.interaction_data["DispersionD4SC"].model
    for name, expected in nested_model_tensors.items():
        _assert_tensor_equal(getattr(d4_model, name), expected)
    for name, expected in nested_model_values.items():
        assert getattr(d4_model, name) == expected
    for name, before in index_state.items():
        after = getattr(case.system.ihelp, name)
        if name in index_tensors:
            _assert_tensor_equal(after, index_tensors[name])
        else:
            assert after is before or after == before

    for path, expected in core_tensor_values:
        actual = dict(tensor_leaves(inputs))[path]
        _assert_tensor_equal(actual, expected)

    assert all(
        actual is original
        for actual, original in zip(
            legacy_components, case.interactions.components
        )
    )
    _assert_snapshot_equal(
        _snapshot_tree(
            (
                legacy_interactions,
                case.interaction_data,
                case.system.ihelp,
                inputs.integrals,
            )
        ),
        legacy_setup_snapshot,
    )

    with pytest.raises(FrozenInstanceError):
        inputs.n0 = inputs.n0  # type: ignore[misc]


def test_functional_core_has_no_legacy_mutation_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All raw representations run without receiving legacy containers."""
    case = _prepare_case(
        GFN2_XTB,
        field=torch.tensor([0.001, -0.0005, 0.00025], dtype=torch.float64),
        scf_options={"fermi_etemp": 300.0},
    )
    inputs = case.inputs
    assert isinstance(inputs.index, CoreIndexData)
    assert isinstance(inputs.interactions, FunctionalInteractionSet)
    assert {field.name for field in fields(type(inputs))}.isdisjoint(
        {"ihelp", "interaction_data"}
    )
    assert all(
        type(value).__name__
        not in {"IndexHelper", "InteractionList", "InteractionListCache"}
        for value in (inputs.index, inputs.interactions)
    )

    def reject_legacy_owner(value: object) -> None:
        if isinstance(
            value,
            (
                IndexHelper,
                Component,
                Interaction,
                InteractionList,
                InteractionListCache,
            ),
        ):
            raise AssertionError(
                f"Legacy mutable owner retained in core input: {type(value)}"
            )
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                reject_legacy_owner(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                reject_legacy_owner(item)

    reject_legacy_owner(inputs)

    def forbidden(*args, **kwargs):
        raise AssertionError("A functional raw map reached a legacy mutator.")

    for cls, names in (
        (IndexHelper, ("cull", "restore")),
        (Component, ("update", "reset")),
        (Interaction, ("update", "reset")),
        (InteractionList, ("update", "reset", "reset_all")),
        (InteractionListCache, ("cull", "restore")),
    ):
        for name in names:
            monkeypatch.setattr(cls, name, forbidden)

    initial = evaluate_hamiltonian(inputs.integrals.hcore, inputs).charges
    charge_first = scf_map(case.x, inputs)
    charge_again = scf_map(case.x, inputs)
    _assert_tensor_equal(charge_first.mapped_x, charge_again.mapped_x)

    potential_x, layout = potential_initial_guess(initial, inputs)
    potential_first = potential_map(potential_x, inputs, layout)
    potential_again = potential_map(potential_x, inputs, layout)
    _assert_tensor_equal(potential_first.mapped_x, potential_again.mapped_x)

    fock_x = fock_initial_guess(initial, inputs)
    fock_first = fock_map(fock_x, inputs)
    fock_again = fock_map(fock_x, inputs)
    _assert_tensor_equal(fock_first.mapped_x, fock_again.mapped_x)

    assert charge_first.fermi_status is not None
    assert potential_first.fermi_status is not None
    assert fock_first.fermi_status is not None
    solved = solve_scf_simple(
        case.x,
        inputs,
        SCFSolverSettings(
            maxiter=1,
            mix_guess=True,
            damp=0.4,
            x_atol=1.0e-8,
            x_atol_max=1.0e-8,
        ),
    )
    assert torch.isfinite(solved.energy).all()


def test_b6a_functional_interaction_boundary_is_history_independent() -> None:
    """All seven interactions together exercise the functional-map boundary."""
    case = _prepare_case(GFN2_XTB)
    numbers = case.system.numbers
    d3_case = _prepare_case(GFN1_XTB)
    d3_component = next(
        item
        for item in d3_case.system.classicals.components
        if type(item) is DispersionD3
    )
    model = Model(
        par=case.par,
        config=case.system.config,
        classical=(d3_component,),
        interaction=(
            GeneralizedBorn(numbers, 78.9, device="cpu", dtype=torch.float64),
        ),
        auto_int_level=False,
    )

    def make_system():
        return model.setup(numbers)

    system = make_system()
    expected_types = {
        ES2,
        ES3,
        AES2,
        DispersionD4SC,
        GeneralizedBorn,
        ElectricField,
        ElectricFieldGrad,
    }
    call_fields = (
        ElectricField(torch.zeros(3, dtype=torch.float64)),
        ElectricFieldGrad(torch.zeros((3, 3), dtype=torch.float64)),
    )
    interactions = InteractionList(
        *system.interactions.components, *call_fields
    )
    assert {type(item) for item in interactions.components} == expected_types
    assert functional_scf_interactions_eligible(interactions) == (True, "")

    def persistent_snapshot(current_system):
        return _snapshot_tree(
            (
                current_system.ihelp,
                current_system.interactions,
                current_system.classicals,
                current_system.es2_setup,
                current_system.es3_setup,
                current_system.aes2_setup,
                current_system.d4sc_setup,
                current_system.generalized_born_setup,
                current_system.integral_setup,
                current_system.h0_setup,
                current_system._classical_extension_cache,
                current_system._packed_classical_cache,
                current_system.d3_setups,
            )
        )

    before = persistent_snapshot(system)

    def evaluate(current_system, positions, field, field_grad):
        positions = positions.clone().requires_grad_(True)
        field = field.clone().requires_grad_(True)
        field_grad = field_grad.clone().requires_grad_(True)
        local_interactions = InteractionList(
            *current_system.interactions.components,
            ElectricField(field),
            ElectricFieldGrad(field_grad),
        )
        integrals, _, _ = build_integral_matrices(
            current_system.integral_setup,
            current_system.h0_setup,
            positions,
            case.charge,
        )
        data = _interaction_data(current_system, positions, local_interactions)
        assert any(
            type(item) is DispersionD3
            for item in current_system.classicals.components
        )
        assert set(data) == {
            "ES2",
            "ES3",
            "AES2",
            "DispersionD4SC",
            "GeneralizedBorn",
            "ElectricField",
            "ElectricFieldGrad",
        }
        assert "Dispersion" not in data
        assert "DispersionD3" not in data
        inputs = setup_scf_inputs(
            numbers,
            integrals,
            case.n0,
            case.occupation,
            case.nel,
            local_interactions,
            data,
            current_system.ihelp,
            current_system.config.scf,
        )
        assert {item.kind for item in inputs.interactions.entries} == {
            "es2",
            "es3",
            "aes2",
            "d4sc",
            "generalized_born",
            "electric_field",
            "electric_field_grad",
        }
        assert not hasattr(inputs, "classicals")
        mapped = scf_map(case.x, inputs)
        value = (
            mapped.mapped_x.square().sum() + mapped.hamiltonian.square().sum()
        )
        gradients = torch.autograd.grad(value, (positions, field, field_grad))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        return value, gradients, data

    positions_a = case.positions
    positions_b = positions_a.clone()
    positions_b[1, 2] += 0.07
    field_a = torch.tensor([0.001, -0.0005, 0.00025], dtype=torch.float64)
    field_b = field_a + torch.tensor([0.0003, 0.0002, -0.0001])
    field_grad_a = torch.full((3, 3), 0.0001, dtype=torch.float64)
    field_grad_b = field_grad_a * 1.5

    value_a, grad_a, data_a = evaluate(
        system, positions_a, field_a, field_grad_a
    )
    value_b, _, data_b = evaluate(system, positions_b, field_b, field_grad_b)
    value_a_again, grad_a_again, data_a_again = evaluate(
        system, positions_a, field_a, field_grad_a
    )
    fresh_system = make_system()
    value_a_fresh, grad_a_fresh, data_a_fresh = evaluate(
        fresh_system, positions_a, field_a, field_grad_a
    )

    torch.testing.assert_close(value_a_again, value_a, rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(value_a_fresh, value_a, rtol=2e-10, atol=2e-12)
    for actual, expected, fresh in zip(grad_a_again, grad_a, grad_a_fresh):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
        torch.testing.assert_close(actual, fresh, rtol=2e-8, atol=2e-10)

    assert data_a is not data_a_again
    assert data_a is not data_a_fresh
    for label in data_a:
        assert data_a[label] is not data_a_again[label]
        leaves_a = _tensor_leaves(data_a[label])
        leaves_b = _tensor_leaves(data_b[label])
        leaves_a_again = _tensor_leaves(data_a_again[label])
        leaves_a_fresh = _tensor_leaves(data_a_fresh[label])
        assert leaves_a
        assert len(leaves_a) == len(leaves_b) == len(leaves_a_again)
        for first, repeated in zip(leaves_a, leaves_a_again):
            _assert_tensor_equal(first, repeated)
        assert len(leaves_a) == len(leaves_a_fresh)
        if label == "ES3":
            for first, changed in zip(leaves_a, leaves_b):
                _assert_tensor_equal(first, changed)
        else:
            assert any(
                not torch.equal(first, changed)
                for first, changed in zip(leaves_a, leaves_b)
            ), f"{label} call data did not respond to changed geometry/input"

    _assert_snapshot_equal(persistent_snapshot(system), before)


def test_d3_setup_tensors_are_in_persistent_snapshots() -> None:
    """Snapshots observe all captured D3 Nodes, including nested parameters."""
    from dxtb._src.components.classicals.dispersion.factory import (
        new_dispersion,
    )
    from dxtb.config import Config

    numbers = samples["LiH"]["numbers"]
    parameters = ParamModule(GFN1_XTB, dtype=torch.float64)
    first = new_dispersion(numbers, parameters, dtype=torch.float64)
    second = new_dispersion(numbers, parameters, dtype=torch.float64)
    assert first is not None and second is not None
    first.label = "D3:snapshot-first"
    second.label = "D3:snapshot-second"
    system = Model(
        par=parameters,
        config=Config.create(exclude="all"),
        classical=(first, second),
    ).setup(numbers)
    assert len(system.d3_setups) == 2

    for component, setup in system.d3_setups:
        assert component is first or component is second
        leaves = _tensor_leaves(setup)
        expected = (
            setup.setup_numbers,
            setup.component_numbers,
            setup.reference_cn,
            setup.reference_c6,
            setup.rcov,
            setup.rvdw,
            setup.r4r2,
            setup.cutoff,
            *(
                value
                for _, value in setup.parameters
                if isinstance(value, torch.Tensor)
            ),
        )
        assert all(
            any(leaf is tensor for leaf in leaves) for tensor in expected
        )

    before_value = _snapshot_tree(system.d3_setups)
    before_identity = _persistent_identity_snapshot(system)
    selected_component, selected_setup = system.d3_setups[1]
    original_c6 = selected_setup.reference_c6.detach().clone()
    try:
        with torch.no_grad():
            selected_setup.reference_c6.add_(1.0)
        with pytest.raises(AssertionError):
            _assert_snapshot_equal(
                _snapshot_tree(system.d3_setups), before_value
            )
    finally:
        with torch.no_grad():
            selected_setup.reference_c6.copy_(original_c6)
    _assert_snapshot_equal(_snapshot_tree(system.d3_setups), before_value)
    old_c6 = selected_setup.reference_c6
    replacement_c6 = (
        old_c6.detach().clone().requires_grad_(old_c6.requires_grad)
    )
    assert replacement_c6 is not old_c6
    assert replacement_c6.dtype == old_c6.dtype
    assert replacement_c6.device == old_c6.device
    torch.testing.assert_close(replacement_c6, old_c6, rtol=0.0, atol=0.0)
    try:
        # Replace only the captured leaf while keeping every snapshot root
        # object identical, so this specifically exercises leaf identity.
        object.__setattr__(selected_setup, "reference_c6", replacement_c6)
        with pytest.raises(AssertionError):
            _assert_persistent_identity_unchanged(system, before_identity)
    finally:
        object.__setattr__(selected_setup, "reference_c6", old_c6)
    _assert_persistent_identity_unchanged(system, before_identity)


def test_h0_node_tensors_are_in_persistent_snapshots() -> None:
    """The persistent-state snapshot sees and detects H0 Node mutations."""
    system = _prepare_case(GFN1_XTB).system
    assert system.h0_setup is not None
    before = _persistent_identity_snapshot(system)
    tensor = system.h0_setup.hscale
    original = tensor.clone()
    try:
        with torch.no_grad():
            tensor.view(-1)[0].add_(1.0)
        with pytest.raises(AssertionError):
            _assert_persistent_identity_unchanged(system, before)
    finally:
        with torch.no_grad():
            tensor.copy_(original)
    _assert_persistent_identity_unchanged(system, before)


def test_functional_interaction_aggregates_match_legacy() -> None:
    """Pure interaction dispatch preserves legacy potential and energy."""
    case = _prepare_case(GFN2_XTB)
    numbers = case.system.numbers
    d3_case = _prepare_case(GFN1_XTB)
    d3_component = next(
        item
        for item in d3_case.system.classicals.components
        if type(item) is DispersionD3
    )
    model = Model(
        par=case.par,
        config=case.system.config,
        classical=(d3_component,),
        interaction=(
            GeneralizedBorn(numbers, 78.9, device="cpu", dtype=torch.float64),
        ),
        auto_int_level=False,
    )
    system = model.setup(numbers)
    field = ElectricField(
        torch.tensor([0.001, -0.0005, 0.00025], dtype=torch.float64)
    )
    field_grad = ElectricFieldGrad(
        torch.full((3, 3), 0.0001, dtype=torch.float64)
    )
    all_interactions = InteractionList(
        *system.interactions.components, field, field_grad
    )
    integrals, _, _ = build_integral_matrices(
        system.integral_setup,
        system.h0_setup,
        case.positions,
        case.charge,
    )
    nat = numbers.numel()
    nmo = case.n0.numel()
    core_charges = SCFCharges(
        mono=torch.linspace(-0.17, 0.23, nmo, dtype=torch.float64),
        dipole=torch.linspace(
            -0.08, 0.11, nat * 3, dtype=torch.float64
        ).reshape(nat, 3),
        quad=torch.linspace(-0.05, 0.07, nat * 6, dtype=torch.float64).reshape(
            nat, 6
        ),
    )
    legacy_charges = Charges(
        mono=core_charges.mono,
        dipole=core_charges.dipole,
        quad=core_charges.quad,
    )

    def compare(interactions: InteractionList) -> None:
        data = _interaction_data(system, case.positions, interactions)
        inputs = setup_scf_inputs(
            numbers,
            integrals,
            case.n0,
            case.occupation,
            case.nel,
            interactions,
            data,
            system.ihelp,
            system.config.scf,
        )
        old_potential = interactions.get_potential(
            data, legacy_charges, system.ihelp
        )
        new_potential = potential_from_charges(core_charges, inputs)
        for name in ("mono", "dipole", "quad"):
            old = getattr(old_potential, name)
            new = getattr(new_potential, name)
            assert (old is None) == (new is None), name
            if old is not None:
                assert new is not None
                torch.testing.assert_close(new, old, rtol=2e-12, atol=2e-13)

        old_energy = interactions.get_energy(legacy_charges, data, system.ihelp)
        new_energy = interaction_energy_from_charges(core_charges, inputs)
        torch.testing.assert_close(
            new_energy, old_energy, rtol=2e-12, atol=2e-13
        )

    # The singleton rows make each migrated formula independently visible;
    # the aggregate row also protects composition and optional multipoles.
    for interaction in all_interactions.components:
        compare(InteractionList(interaction))
    compare(all_interactions)


_B6A_FAMILY_ROWS = (
    pytest.param("ES2", "geometry", id="es2"),
    pytest.param("ES3", "static-setup", id="es3"),
    pytest.param("AES2", "geometry", id="aes2"),
    pytest.param("DispersionD4SC", "geometry", id="d4sc"),
    pytest.param("GeneralizedBorn", "geometry", id="generalized-born"),
    pytest.param("ElectricField", "geometry+field", id="electric-field"),
    pytest.param(
        "ElectricFieldGrad",
        "geometry+field_grad",
        id="electric-field-grad",
    ),
)


@pytest.mark.parametrize(
    "family,expected_data_behavior",
    _B6A_FAMILY_ROWS,
)
def test_b6a_functional_interaction_family_matrix(
    family: str, expected_data_behavior: str
) -> None:
    """Isolate each supported interaction's data, map, gradient, and state."""
    assert len(_FUNCTIONAL_SCF_INTERACTION_TYPES) == 7
    assert set(_FUNCTIONAL_SCF_INTERACTION_TYPES) == {
        ES2,
        ES3,
        AES2,
        DispersionD4SC,
        GeneralizedBorn,
        ElectricField,
        ElectricFieldGrad,
    }

    case = _prepare_case(GFN2_XTB)
    d3_case = _prepare_case(GFN1_XTB)
    d3_component = next(
        item
        for item in d3_case.system.classicals.components
        if type(item) is DispersionD3
    )
    model = Model(
        par=case.par,
        config=case.system.config,
        classical=(d3_component,),
        interaction=(
            GeneralizedBorn(
                case.system.numbers, 78.9, device="cpu", dtype=torch.float64
            ),
        ),
        auto_int_level=False,
    )
    system = model.setup(case.system.numbers)

    class_by_name = {
        cls.__name__: cls for cls in _FUNCTIONAL_SCF_INTERACTION_TYPES
    }
    interaction_type = class_by_name[family]

    positions_a = case.positions
    positions_b = positions_a.clone()
    positions_b[1, 2] += 0.07
    field_a = torch.tensor([0.001, -0.0005, 0.00025], dtype=torch.float64)
    field_b = field_a + torch.tensor([0.0003, 0.0002, -0.0001])
    field_grad_a = torch.tensor(
        [
            [0.0001, 0.0003, -0.0002],
            [-0.0001, 0.0002, 0.0004],
            [0.0005, -0.0003, -0.0003],
        ],
        dtype=torch.float64,
    )
    field_grad_b = field_grad_a + torch.tensor(
        [
            [0.0002, -0.0001, 0.0003],
            [0.0004, 0.0001, -0.0002],
            [-0.0003, 0.0005, -0.0001],
        ],
        dtype=torch.float64,
    )

    def evaluate(current_system, positions, field=None, field_grad=None):
        positions = positions.clone().requires_grad_(True)
        if family == "ElectricField":
            assert field is not None
            interaction = ElectricField(field.clone().requires_grad_(True))
            grad_variables = (positions, interaction.field)
        elif family == "ElectricFieldGrad":
            assert field_grad is not None
            interaction = ElectricFieldGrad(
                field_grad.clone().requires_grad_(True)
            )
            grad_variables = (positions, interaction.field_grad)
        else:
            interaction = next(
                item
                for item in current_system.interactions.components
                if type(item) is interaction_type
            )
            grad_variables = (positions,)

        interactions = InteractionList(interaction)
        data = _interaction_data(current_system, positions, interactions)
        integrals, _, _ = build_integral_matrices(
            current_system.integral_setup,
            current_system.h0_setup,
            positions,
            case.charge,
        )
        inputs = setup_scf_inputs(
            case.system.numbers,
            integrals,
            case.n0,
            case.occupation,
            case.nel,
            interactions,
            data,
            current_system.ihelp,
            current_system.config.scf,
        )
        assert functional_scf_interactions_eligible(interactions) == (True, "")
        expected_kind = {
            ES2: "es2",
            ES3: "es3",
            AES2: "aes2",
            DispersionD4SC: "d4sc",
            GeneralizedBorn: "generalized_born",
            ElectricField: "electric_field",
            ElectricFieldGrad: "electric_field_grad",
        }[interaction_type]
        assert {item.kind for item in inputs.interactions.entries} == {
            expected_kind
        }
        assert any(
            type(item) is DispersionD3
            for item in current_system.classicals.components
        )
        assert len(data) == 1
        result = scf_map(case.x, inputs)

        x_probe = torch.linspace(
            0.17, 1.13, result.mapped_x.numel(), dtype=result.mapped_x.dtype
        ).reshape_as(result.mapped_x)
        h_probe = torch.linspace(
            1.07,
            0.13,
            result.hamiltonian.numel(),
            dtype=result.hamiltonian.dtype,
        ).reshape_as(result.hamiltonian)
        projection = (result.mapped_x * x_probe).sum() + (
            result.hamiltonian * h_probe
        ).sum()
        gradients = torch.autograd.grad(projection, grad_variables)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        cache = data[interaction.label]
        if family == "ES3":
            assert current_system.es3_setup is not None
            assert cache.hd is current_system.es3_setup.hubbard_derivs
        return result, projection, gradients, data, cache, current_system

    before = _persistent_identity_snapshot(system)

    if family in ("ElectricField", "ElectricFieldGrad"):
        assert (
            expected_data_behavior
            == {
                "ElectricField": "geometry+field",
                "ElectricFieldGrad": "geometry+field_grad",
            }[family]
        )
        if family == "ElectricField":
            a = evaluate(system, positions_a, field_a)
            field_only_b = evaluate(system, positions_a, field_b)
            geometry_only_b = evaluate(system, positions_b, field_a)
            a_again = evaluate(system, positions_a, field_a)
            _assert_persistent_identity_unchanged(system, before)
            fresh_system = model.setup(case.system.numbers)
            fresh_a = evaluate(fresh_system, positions_a, field_a)
            assert not torch.equal(a[4].vat, field_only_b[4].vat)
            assert not torch.equal(a[4].vdp, field_only_b[4].vdp)
            assert not torch.equal(a[4].vat, geometry_only_b[4].vat)
            _assert_tensor_equal(a[4].vdp, geometry_only_b[4].vdp)
            changed_runs = (field_only_b, geometry_only_b)
        else:
            a = evaluate(system, positions_a, field_grad=field_grad_a)
            field_only_b = evaluate(
                system, positions_a, field_grad=field_grad_b
            )
            geometry_only_b = evaluate(
                system, positions_b, field_grad=field_grad_a
            )
            a_again = evaluate(system, positions_a, field_grad=field_grad_a)
            _assert_persistent_identity_unchanged(system, before)
            fresh_system = model.setup(case.system.numbers)
            fresh_a = evaluate(
                fresh_system, positions_a, field_grad=field_grad_a
            )
            for name in ("vat", "vdp", "vqp"):
                assert not torch.equal(
                    getattr(a[4], name), getattr(field_only_b[4], name)
                )
            for name in ("vat", "vdp"):
                assert not torch.equal(
                    getattr(a[4], name), getattr(geometry_only_b[4], name)
                )
            _assert_tensor_equal(a[4].vqp, geometry_only_b[4].vqp)
            changed_runs = (field_only_b, geometry_only_b)
    else:
        a = evaluate(system, positions_a)
        geometry_b = evaluate(system, positions_b)
        a_again = evaluate(system, positions_a)
        _assert_persistent_identity_unchanged(system, before)
        fresh_system = model.setup(case.system.numbers)
        fresh_a = evaluate(fresh_system, positions_a)
        changed_runs = (geometry_b,)
        if expected_data_behavior == "geometry":
            data_a = _tensor_leaves(a[4])
            data_b = _tensor_leaves(geometry_b[4])
            assert len(data_a) == len(data_b)
            assert any(
                not torch.equal(first, changed)
                for first, changed in zip(data_a, data_b)
            ), f"{family} call data did not respond to geometry alone"
        else:
            assert expected_data_behavior == "static-setup"
            _assert_snapshot_equal(
                _snapshot_tree(a[4]), _snapshot_tree(geometry_b[4])
            )
            assert a[4].hd is system.es3_setup.hubbard_derivs
            assert geometry_b[4].hd is system.es3_setup.hubbard_derivs

    _assert_map_result_equal(a[0], a_again[0])
    _assert_map_result_equal(a[0], fresh_a[0])
    for actual, expected in zip(a_again[2], a[2]):
        _assert_tensor_equal(actual, expected)
    for actual, expected in zip(fresh_a[2], a[2]):
        _assert_tensor_equal(actual, expected)
    for changed in changed_runs:
        assert all(torch.isfinite(gradient).all() for gradient in changed[2])

    evaluations = (a, *changed_runs, a_again, fresh_a)
    _assert_pairwise_distinct(tuple(item[3] for item in evaluations))
    _assert_pairwise_distinct(tuple(item[4] for item in evaluations))

    for item in evaluations:
        cache_tensors = _tensor_leaves(item[4])
        current_system = item[5]
        persistent_tensors = _tensor_leaves(current_system)
        shared = tuple(
            tensor
            for tensor in cache_tensors
            if any(tensor is persistent for persistent in persistent_tensors)
        )
        if family == "ES3":
            assert current_system.es3_setup is not None
            assert len(shared) == 1
            assert shared[0] is current_system.es3_setup.hubbard_derivs
        else:
            assert shared == ()

    if family == "ES3":
        assert system.es3_setup is not None
        assert fresh_system.es3_setup is not None
        assert (
            fresh_system.es3_setup.hubbard_derivs
            is not system.es3_setup.hubbard_derivs
        )
    _assert_persistent_identity_unchanged(system, before)


def test_functional_map_has_no_legacy_stateful_hooks() -> None:
    root = Path(__file__).parents[2] / "src/dxtb/_src/scf/functional"
    # E3.1b's isolated implicit-root primitive is the sole custom transform
    # boundary. Keep this architectural guard focused on the raw-map and
    # solver implementation, which must remain ordinary functional code.
    source = "\n".join(
        path.read_text()
        for path in root.glob("*.py")
        if path.name != "implicit.py"
    )
    for forbidden in (
        "_Data",
        ".cull(",
        ".restore(",
        ".reset(",
        ".update(",
        "self._data",
        "object.__setattr__",
        "torch.autograd.Function",
        "cache_is_latest",
        ".item(",
        ".tolist(",
    ):
        assert forbidden not in source
