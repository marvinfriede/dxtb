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

import pytest
import torch
from tad_mctc import storch
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, labels
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.calculators.model import Model
from dxtb._src.components.base import Component
from dxtb._src.components.interactions import Interaction, InteractionList
from dxtb._src.components.interactions.base import InteractionCache
from dxtb._src.components.interactions.container import Charges
from dxtb._src.components.interactions.coulomb.multipole import AES2
from dxtb._src.components.interactions.coulomb.secondorder import ES2
from dxtb._src.components.interactions.coulomb.thirdorder import ES3
from dxtb._src.components.classicals.dispersion.d3 import DispersionD3
from dxtb._src.components.interactions.dispersion.d4sc import DispersionD4SC
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.field.efieldgrad import ElectricFieldGrad
from dxtb._src.components.interactions.solvation.alpb import GeneralizedBorn
from dxtb._src.components.interactions.list import (
    InteractionListCache,
    _FUNCTIONAL_SCF_INTERACTION_TYPES,
)
from dxtb._src.constants import defaults
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional import (
    ChargeLayout,
    SCFCharges,
    PotentialLayout,
    SCFPotential,
    pack_potential,
    potential_initial_guess,
    potential_map,
    potential_layout,
    pack_charges,
    potential_from_charges,
    scf_map,
    setup_scf_inputs,
    unpack_charges,
    unpack_potential,
    validate_functional_scf_interactions,
    functional_scf_interactions_eligible,
)
from dxtb._src.scf.functional.map import _transitional_diagonalize
from dxtb._src.scf.guess import get_guess
from dxtb._src.scf.iterator import get_refocc
from dxtb._src.scf.unrolling import SelfConsistentFieldFull
from dxtb._src.scf.utils import get_density

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
    sample = samples[sample_name]
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
        system.classical_cache,
        *system.classical_cache.values(),
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


def test_transitional_spectral_derivative_branch_attribution() -> None:
    case = _prepare_case(GFN1_XTB)
    hamiltonian = scf_map(case.x, case.inputs).hamiltonian.detach()
    probe = torch.arange(
        hamiltonian.numel(), dtype=hamiltonian.dtype
    ).reshape_as(hamiltonian)
    probe = probe / hamiltonian.numel()
    direction = torch.arange(
        hamiltonian.numel(), dtype=hamiltonian.dtype
    ).reshape_as(hamiltonian)
    direction = (direction + direction.mT) / hamiltonian.numel()
    occupation = case.inputs.occupation.sum(-2)

    def spectral_projection(
        h: torch.Tensor, broadening_method: str | None = "transitional"
    ) -> torch.Tensor:
        if broadening_method == "transitional":
            _, coefficients = _transitional_diagonalize(h, case.inputs)
        else:
            _, coefficients = storch.eighb(
                a=h,
                b=None,
                l_inv=case.inputs.inverse_cholesky,
                is_posdef=True,
                factor=torch.finfo(h.dtype).eps ** 0.5,
                broadening_method=broadening_method,
            )
        density = get_density(coefficients, occupation)
        return torch.sum(probe * density)

    _, tangent = jvp(spectral_projection, (hamiltonian,), (direction,))
    _, explicit_forward_tangent = jvp(
        lambda h: spectral_projection(h, None),
        (hamiltonian,),
        (direction,),
    )
    h_leaf = hamiltonian.clone().requires_grad_(True)
    reverse_value = spectral_projection(h_leaf)
    (reverse_gradient,) = torch.autograd.grad(reverse_value, h_leaf)
    reverse_tangent = torch.sum(reverse_gradient * direction)
    forward_jacobian = jacfwd(spectral_projection)(hamiltonian)
    jacfwd_tangent = torch.sum(forward_jacobian * direction)

    h_lorn = hamiltonian.clone().requires_grad_(True)
    explicit_reverse_value = spectral_projection(h_lorn, "lorn")
    (explicit_reverse_gradient,) = torch.autograd.grad(
        explicit_reverse_value, h_lorn
    )
    explicit_reverse_tangent = torch.sum(explicit_reverse_gradient * direction)
    explicit_jacobian = jacfwd(lambda h: spectral_projection(h, None))(
        hamiltonian
    )
    explicit_jacfwd_tangent = torch.sum(explicit_jacobian * direction)

    # Reverse-mode selection is explicitly Lorentzian; forward-mode selection
    # is explicitly unbroadened. This attributes the known discrepancy to the
    # transitional spectral mode selection without requiring it to persist.
    _assert_tensor_equal(reverse_tangent, explicit_reverse_tangent)
    _assert_tensor_equal(tangent, explicit_forward_tangent)
    _assert_tensor_equal(jacfwd_tangent, explicit_jacfwd_tangent)


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
    interactions = tuple(inputs.interactions.components)
    cache_entries = tuple(inputs.interaction_data.items())
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
        name: getattr(inputs.ihelp, name) for name in inputs.ihelp.__slots__
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
    monkeypatch.setattr(InteractionListCache, "cull", forbidden)
    monkeypatch.setattr(InteractionListCache, "restore", forbidden)

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
    assert tuple(inputs.interactions.components) == interactions
    assert tuple(key for key, _ in inputs.interaction_data.items()) == tuple(
        key for key, _ in cache_entries
    )
    assert all(
        inputs.interaction_data[key] is value for key, value in cache_entries
    )
    for key, cache in cache_entries:
        for name, expected in cache_tensors[key].items():
            _assert_tensor_equal(getattr(cache, name), expected)
    d4_model = inputs.interaction_data["DispersionD4SC"].model
    for name, expected in nested_model_tensors.items():
        _assert_tensor_equal(getattr(d4_model, name), expected)
    for name, expected in nested_model_values.items():
        assert getattr(d4_model, name) == expected
    for name, before in index_state.items():
        after = getattr(inputs.ihelp, name)
        if name in index_tensors:
            _assert_tensor_equal(after, index_tensors[name])
        else:
            assert after is before or after == before

    with pytest.raises(FrozenInstanceError):
        inputs.n0 = inputs.n0  # type: ignore[misc]


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
                current_system.classical_cache,
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
        assert {type(item) for item in inputs.interactions.components} == (
            expected_types
        )
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


_B6A_FAMILY_ROWS = (
    pytest.param("ES2", "positions", "geometry", id="es2"),
    pytest.param("ES3", "setup", "static-setup", id="es3"),
    pytest.param("AES2", "positions", "geometry", id="aes2"),
    pytest.param("DispersionD4SC", "positions", "geometry", id="d4sc"),
    pytest.param(
        "GeneralizedBorn", "positions", "geometry", id="generalized-born"
    ),
    pytest.param(
        "ElectricField",
        "positions+field",
        "geometry+field",
        id="electric-field",
    ),
    pytest.param(
        "ElectricFieldGrad",
        "positions+field_grad",
        "geometry+field_grad",
        id="electric-field-grad",
    ),
)


@pytest.mark.parametrize(
    "family,input_owners,expected_data_behavior", _B6A_FAMILY_ROWS
)
def test_b6a_functional_interaction_family_matrix(
    family: str, input_owners: str, expected_data_behavior: str
) -> None:
    """Isolate each supported interaction's data, map, gradient, and state."""
    expected_owners = {
        "ES2": "positions",
        "ES3": "setup",
        "AES2": "positions",
        "DispersionD4SC": "positions",
        "GeneralizedBorn": "positions",
        "ElectricField": "positions+field",
        "ElectricFieldGrad": "positions+field_grad",
    }
    assert input_owners == expected_owners[family]
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
        assert {type(item) for item in inputs.interactions.components} == {
            interaction_type
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
    source = "\n".join(path.read_text() for path in root.glob("*.py"))
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
