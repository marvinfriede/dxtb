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

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tad_mctc import storch
from torch.func import jacfwd, jvp, vmap

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, labels
from dxtb._src.calculators.singlepoint import _interaction_data
from dxtb._src.components.base import Component
from dxtb._src.components.interactions import Interaction, InteractionList
from dxtb._src.components.interactions.base import InteractionCache
from dxtb._src.components.interactions.container import Charges
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.list import InteractionListCache
from dxtb._src.constants import defaults
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional import (
    ChargeLayout,
    SCFCharges,
    SCFPotential,
    pack_charges,
    potential_from_charges,
    scf_map,
    setup_scf_inputs,
    unpack_charges,
    validate_functional_scf_interactions,
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
    charge: float = 0.0,
    spin: float = 0.0,
    field: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    par: ParamModule | None = None,
    int_level: int | None = None,
    scf_options: dict[str, object] | None = None,
) -> SimpleNamespace:
    sample = samples["H2O"]
    numbers = sample["numbers"].to(device="cpu")
    if positions is None:
        positions = (
            sample["positions"].to(device="cpu", dtype=torch.float64).clone()
        )
    if par is None:
        par = get_param_module(
            "gfn1" if method == GFN1_XTB else "gfn2",
            device="cpu",
            dtype=torch.float64,
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
        dtype=torch.float64,
        opts=opts,
    )
    system = calc.model.setup(numbers, dd=calc.dd)
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
        inputs=inputs,
    )


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-13)


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
        "charge": case.charge,
        "spin": case.spin,
        "field": case.field,
        "positions": case.positions.clone(),
        "par": case.par,
        "int_level": case.int_level,
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
    with pytest.raises(NotImplementedError, match="charge raw map only"):
        setup_scf_inputs(
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

    warm_config = replace(
        case.system.config.scf,
        fermi=replace(case.system.config.scf.fermi, etemp=1000.0),
    )
    with pytest.raises(
        NotImplementedError, match="zero electronic temperature"
    ):
        setup_scf_inputs(
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
    after = scf_map(case.x, case.inputs)
    _assert_tensor_equal(after.mapped_x, baseline.mapped_x)


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
