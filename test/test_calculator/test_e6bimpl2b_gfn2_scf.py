# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

"""Mixed GFN2 padded interaction and fixed-trip SCF acceptance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten

from dxtb import GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.padded_integrals import evaluate_padded_integrals
from dxtb._src.calculators.singlepoint import (
    _evaluate_classical_descriptor,
    _interaction_data,
)
from dxtb._src.scf.functional.core import FunctionalInteractionSet
from dxtb._src.scf.functional.data import SCFCharges
from dxtb._src.scf.functional.interactions import (
    interaction_energy_from_charges,
    potential_from_charges,
    prepare_functional_interactions,
)
from dxtb._src.scf.functional.padded import (
    _prepare_padded_interactions,
    evaluate_padded_fixed_trip,
    evaluate_padded_systems,
    setup_padded_scf_inputs,
)
from dxtb._src.scf.guess import get_eeq_guess

DTYPE = torch.float64

_GEOMETRIES = {
    (1, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]],
    (3, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.62]],
    (8, 1, 1): [
        [0.0, 0.0, 0.0],
        [0.82, 0.0, 0.61],
        [-0.82, 0.0, 0.61],
    ],
    (6, 1, 1, 1, 1): [
        [0.0, 0.0, 0.0],
        [0.65, 0.60, 0.66],
        [-0.65, -0.61, 0.59],
        [-0.61, 0.67, -0.62],
        [0.66, -0.64, -0.59],
    ],
}


def _model(
    *,
    maxiter: int = 12,
    temperature: float = 0.0,
    differentiable_h0: bool = False,
    mixer: str = "linear",
    scp_mode: str = "charge",
    differentiable_path: tuple[str, ...] | None = None,
    fermi_maxiter: int = 200,
    fermi_thresh: float | None = None,
) -> Model:
    """Construct a production GFN2 Model with the requested fixed-trip setup."""
    parameters = ParamModule(GFN2_XTB, dtype=DTYPE)
    if differentiable_h0:
        parameters.set_differentiable("hamiltonian", "xtb", "wexp")
    if differentiable_path is not None:
        parameters.set_differentiable(*differentiable_path)
    return Model(
        par=parameters,
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_QUADRUPOLE,
            mixer=mixer,
            scp_mode=scp_mode,
            maxiter=maxiter,
            fermi_etemp=temperature,
            fermi_maxiter=fermi_maxiter,
            fermi_thresh=fermi_thresh,
        ),
        auto_int_level=False,
    )


def _positions(
    compositions: Sequence[tuple[int, ...]],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.tensor(_GEOMETRIES[numbers], dtype=DTYPE)
        for numbers in compositions
    )


def _systems(model: Model, compositions: Sequence[tuple[int, ...]]):
    systems = tuple(
        model.setup(torch.tensor(numbers, dtype=torch.int64))
        for numbers in compositions
    )
    assert all(system.exact_core is not None for system in systems)
    return systems


def _prepare(model: Model, compositions: Sequence[tuple[int, ...]]):
    systems = _systems(model, compositions)
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, lane)
        for lane, core in enumerate(cores)
    )
    source_positions = _positions(compositions)
    positions = tuple(
        canonicalize_positions(position, bucket, lane)
        for lane, position in enumerate(source_positions)
    )
    zeros = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    return systems, bucket, padded, source_positions, positions, zeros


def _core_with_eeq_cutoff(core, cutoff: float):
    """Return an immutable exact-core copy with one D4 EEQ cutoff value."""
    terms = []
    for descriptor in core.classical_terms:
        if descriptor.kind != "d4":
            terms.append(descriptor)
            continue
        setup = descriptor.setup
        cutoff_value = torch.as_tensor(
            cutoff,
            dtype=setup.cutoff.cn_eeq.dtype,
            device=setup.cutoff.cn_eeq.device,
        )
        setup = setup.replace(cutoff=setup.cutoff.replace(cn_eeq=cutoff_value))
        terms.append(descriptor.replace(setup=setup))
    if not any(term.kind == "d4" for term in terms):
        raise AssertionError("GFN2 fixture did not install classical D4.")
    return core.replace(classical_terms=tuple(terms))


def _assert_result_parity(actual, expected) -> None:
    assert tree_flatten(actual)[1] == tree_flatten(expected)[1]
    assert tuple(name for name, _ in actual.classical) == tuple(
        name for name, _ in expected.classical
    )
    for name in (
        "energy",
        "scf",
        "fenergy",
        "density",
        "emo",
        "hamiltonian",
        "overlap",
        "hcore",
        "dipole_integrals",
        "quadrupole_integrals",
        "overlap_norm",
        "iterations",
        "converged",
        "residual",
    ):
        left, right = getattr(actual, name), getattr(expected, name)
        if right is None:
            assert left is None
        else:
            assert left is not None
            try:
                torch.testing.assert_close(left, right, atol=5e-8, rtol=5e-8)
            except AssertionError as exc:
                raise AssertionError(f"Result.{name} differs: {exc}") from exc
    if expected.occupation is None:
        assert actual.occupation is None
    else:
        assert actual.occupation is not None
        torch.testing.assert_close(
            actual.occupation.sum(-1), expected.occupation.sum(-1)
        )
        # In a degenerate eigenspace, an Aufbau occupation may be distributed
        # over an arbitrary basis. Compare total occupation per energy cluster.
        energies = expected.emo.reshape(-1, expected.emo.shape[-1])
        actual_occupations = actual.occupation.reshape(
            -1, actual.occupation.shape[-1]
        )
        expected_occupations = expected.occupation.reshape(
            -1, expected.occupation.shape[-1]
        )
        for lane_energies, lane_actual, lane_expected in zip(
            energies, actual_occupations, expected_occupations
        ):
            groups: list[list[int]] = [[0]]
            for orbital in range(1, lane_energies.numel()):
                if (
                    abs(
                        (
                            lane_energies[orbital] - lane_energies[orbital - 1]
                        ).item()
                    )
                    < 5e-8
                ):
                    groups[-1].append(orbital)
                else:
                    groups.append([orbital])
            for group in groups:
                indices = torch.tensor(group, device=lane_energies.device)
                torch.testing.assert_close(
                    lane_actual.index_select(0, indices).sum(),
                    lane_expected.index_select(0, indices).sum(),
                    atol=5e-8,
                    rtol=5e-8,
                )
    if expected.potential is None:
        assert actual.potential is None
    else:
        assert actual.potential is not None
        for name in ("mono", "dipole", "quadrupole"):
            left = getattr(actual.potential, name)
            right = getattr(expected.potential, name)
            if right is None:
                assert left is None
            else:
                assert left is not None
                torch.testing.assert_close(
                    left,
                    right,
                    atol=5e-8,
                    rtol=5e-8,
                    msg=lambda message: f"Result.{name}: {message}",
                )
    if expected.status is None:
        assert actual.status is None
    else:
        assert actual.status is not None
        for name in ("trajectory_failure", "final_failure"):
            torch.testing.assert_close(
                getattr(actual.status, name),
                getattr(expected.status, name),
            )
    for (name_a, value_a), (name_b, value_b) in zip(
        actual.classical, expected.classical
    ):
        assert name_a == name_b
        torch.testing.assert_close(value_a, value_b, atol=5e-8, rtol=5e-8)
    assert (actual.charges is None) == (expected.charges is None)
    if actual.charges is None or expected.charges is None:
        return
    for name in ("mono", "dipole", "quadrupole"):
        left = getattr(actual.charges, name)
        right = getattr(expected.charges, name)
        if right is None:
            assert left is None
        else:
            assert left is not None
            torch.testing.assert_close(left, right, atol=5e-8, rtol=5e-8)


@pytest.mark.parametrize(
    "compositions",
    [
        ((1, 1), (1, 1)),
        ((1, 1), (3, 1)),
        ((1, 1), (8, 1, 1)),
        ((8, 1, 1), (1, 1)),
        ((8, 1, 1), (6, 1, 1, 1, 1)),
    ],
    ids=["h2-h2", "h2-lih", "h2-water", "water-h2", "water-methane"],
)
def test_mixed_gfn2_fixed_trip_matches_independent_systems(
    compositions: tuple[tuple[int, ...], ...],
) -> None:
    """Mapped default GFN2 Results match independent fixed-trip Systems."""
    # The zero-temperature Aufbau occupation is gauge-dependent inside the
    # symmetry-degenerate methane frontier. Finite temperature gives a stable
    # per-orbital occupation comparison while preserving the production model.
    temperature = 300.0 if (6, 1, 1, 1, 1) in compositions else 0.0
    systems, bucket, padded, source_positions, positions, zeros = _prepare(
        _model(temperature=temperature), compositions
    )
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, zeros, zeros
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, zeros, zeros
        )
    )
    assert mapped.energy.shape == (len(systems), bucket.atoms)
    assert mapped.classical[0][0] == "DispersionD4"
    assert mapped.classical[1][0] == "Repulsion"
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)


@pytest.mark.parametrize("source_ids", [(4, 19), (19, 4)])
def test_gfn2_noncontiguous_bucket_source_ids_preserve_lane_order(
    source_ids: tuple[int, int],
) -> None:
    """Bucket source identifiers select lanes without acting as tensor offsets."""
    compositions = ((1, 1), (8, 1, 1))
    systems = _systems(_model(maxiter=2), compositions)
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    base_bucket = bucket_canonical_exact_cores(cores)
    bucket = replace(
        base_bucket,
        indices=source_ids,
        source_maps=tuple(
            (source_id, source_map)
            for source_id, (_, source_map) in zip(
                source_ids, base_bucket.source_maps
            )
        ),
        source_pair_shells=tuple(
            (source_id, pair_map)
            for source_id, (_, pair_map) in zip(
                source_ids, base_bucket.source_pair_shells
            )
        ),
    )
    padded = tuple(
        pad_canonical_exact_core(core, bucket, source_id)
        for core, source_id in zip(cores, source_ids)
    )
    source_positions = _positions(compositions)
    positions = tuple(
        canonicalize_positions(position, bucket, source_id)
        for position, source_id in zip(source_positions, source_ids)
    )
    zeros = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    _, decoded = evaluate_padded_systems(
        padded, bucket, positions, zeros, zeros
    )
    serial = tuple(
        system.singlepoint(position, chrg=zero, spin=zero)
        for system, position, zero in zip(systems, source_positions, zeros)
    )
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)


def _component_tensors(
    core, positions: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    entries = _prepare_padded_interactions(core, positions)
    return tuple(
        tensor for entry in entries.entries for _, tensor in entry.data
    )


def _interaction_observables(
    core,
    positions: torch.Tensor,
    charge: torch.Tensor,
    mono: torch.Tensor,
    dipole: torch.Tensor,
    quadrupole: torch.Tensor,
):
    integrals = evaluate_padded_integrals(core, positions, charge)
    inputs = setup_padded_scf_inputs(
        core, integrals, positions, charge, torch.zeros_like(charge)
    )
    trial = SCFCharges(mono=mono, dipole=dipole, quad=quadrupole)
    outputs = []
    for entry in inputs.interactions.entries:
        one = FunctionalInteractionSet((entry,))
        potential = potential_from_charges(trial, one, inputs.index)
        energy = interaction_energy_from_charges(trial, one, inputs.index)
        outputs.append(
            tuple(
                value
                for value in (
                    potential.mono,
                    potential.dipole,
                    potential.quad,
                    energy,
                )
                if value is not None
            )
        )
    return tuple(outputs)


def _serial_interaction_observables(
    system,
    positions: torch.Tensor,
    charge: torch.Tensor,
    mono: torch.Tensor,
    dipole: torch.Tensor,
    quadrupole: torch.Tensor,
):
    """Prepare interactions through the ordinary unpadded System boundary."""
    call_data = _interaction_data(system, positions)
    functional = prepare_functional_interactions(
        system.interactions,
        call_data,
        system.exact_core.numbers,
        d4sc_setup=system.exact_core.interaction_terms[-1].setup,
        kind_by_label=call_data.functional_kinds,
    )
    trial = SCFCharges(mono=mono, dipole=dipole, quad=quadrupole)
    outputs = []
    for entry in functional.entries:
        one = FunctionalInteractionSet((entry,))
        potential = potential_from_charges(trial, one, system.exact_core.index)
        energy = interaction_energy_from_charges(
            trial, one, system.exact_core.index
        )
        outputs.append(
            tuple(
                value
                for value in (
                    potential.mono,
                    potential.dipole,
                    potential.quad,
                    energy,
                )
                if value is not None
            )
        )
    return tuple(outputs)


@pytest.mark.parametrize(
    "compositions",
    [((1, 1), (3, 1)), ((1, 1), (8, 1, 1)), ((8, 1, 1), (6, 1, 1, 1, 1))],
    ids=["h2-lih", "h2-water", "water-methane"],
)
def test_mapped_gfn2_interaction_call_data_matches_single_core(
    compositions: tuple[tuple[int, ...], ...],
) -> None:
    """ES2, ES3, AES2, and D4SC call tensors survive canonical padding."""
    _, bucket, padded, _, positions, _ = _prepare(
        _model(maxiter=1), compositions
    )
    mapped = torch.func.vmap(_component_tensors)(
        stack(padded), torch.stack(positions)
    )
    for lane, (core, position) in enumerate(zip(padded, positions)):
        expected = _component_tensors(core, position)
        assert len(mapped) == len(expected)
        for actual, reference in zip(mapped, expected):
            assert actual.shape[0] == len(padded)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual[lane], reference)


def test_gfn2_nonzero_trial_multipoles_preserve_interaction_potentials_and_energy():
    """Mapped ES2/ES3/AES2/D4SC kernels retain trial-state contributions."""
    compositions = ((1, 1), (8, 1, 1))
    _, _, padded, _, positions, zero_charges = _prepare(
        _model(maxiter=1), compositions
    )
    monopoles = []
    dipoles = []
    quadrupoles = []
    for core in padded:
        atom_mask = core.validity.atoms
        orbital_mask = core.validity.orbitals
        mono = torch.linspace(-0.06, 0.08, orbital_mask.numel(), dtype=DTYPE)
        mono = torch.where(orbital_mask, mono, torch.zeros_like(mono))
        dipole = 0.01 * torch.arange(atom_mask.numel() * 3, dtype=DTYPE)
        dipole = dipole.reshape(-1, 3) + 0.02
        dipole = torch.where(
            atom_mask[:, None], dipole, torch.zeros_like(dipole)
        )
        quadrupole = 0.003 * torch.arange(atom_mask.numel() * 6, dtype=DTYPE)
        quadrupole = quadrupole.reshape(-1, 6) - 0.01
        quadrupole = torch.where(
            atom_mask[:, None], quadrupole, torch.zeros_like(quadrupole)
        )
        monopoles.append(mono)
        dipoles.append(dipole)
        quadrupoles.append(quadrupole)

    mapped = torch.func.vmap(_interaction_observables)(
        stack(padded),
        torch.stack(positions),
        torch.stack(zero_charges),
        torch.stack(monopoles),
        torch.stack(dipoles),
        torch.stack(quadrupoles),
    )
    for lane, core in enumerate(padded):
        serial = _interaction_observables(
            core,
            positions[lane],
            zero_charges[lane],
            monopoles[lane],
            dipoles[lane],
            quadrupoles[lane],
        )
        assert tuple(item.kind for item in core.core.interaction_terms) == (
            "es2",
            "aes2",
            "es3",
            "d4sc",
        )
        assert len(mapped) == len(serial)
        for mapped_entry, serial_entry in zip(mapped, serial):
            assert torch.linalg.vector_norm(serial_entry[-1]) > 0
            for mapped_value, serial_value in zip(mapped_entry, serial_entry):
                torch.testing.assert_close(
                    mapped_value[lane], serial_value, atol=5e-10, rtol=5e-10
                )


@pytest.mark.parametrize(
    "compositions",
    [
        ((1, 1), (3, 1)),
        ((1, 1), (8, 1, 1)),
        ((8, 1, 1), (1, 1)),
    ],
    ids=["h2-lih", "h2-water", "water-h2"],
)
def test_gfn2_interactions_match_independent_unpadded_references(
    compositions: tuple[tuple[int, ...], ...],
) -> None:
    """Each mapped interaction matches independent ordinary-System call data."""
    systems, bucket, padded, source_positions, positions, charges = _prepare(
        _model(maxiter=1), compositions
    )
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    mono_batch = []
    dipole_batch = []
    quadrupole_batch = []
    for core in padded:
        mono = torch.linspace(-0.04, 0.07, core.validity.orbitals.numel())
        mono = mono.to(dtype=DTYPE)
        mono = torch.where(core.validity.orbitals, mono, torch.zeros_like(mono))
        dipole = torch.linspace(-0.03, 0.04, core.validity.atoms.numel() * 3)
        dipole = dipole.to(dtype=DTYPE).reshape(-1, 3)
        dipole = torch.where(
            core.validity.atoms[:, None], dipole, torch.zeros_like(dipole)
        )
        quadrupole = torch.linspace(
            -0.02, 0.03, core.validity.atoms.numel() * 6
        )
        quadrupole = quadrupole.to(dtype=DTYPE).reshape(-1, 6)
        quadrupole = torch.where(
            core.validity.atoms[:, None],
            quadrupole,
            torch.zeros_like(quadrupole),
        )
        mono_batch.append(mono)
        dipole_batch.append(dipole)
        quadrupole_batch.append(quadrupole)

    mapped = torch.func.vmap(_interaction_observables)(
        stack(padded),
        torch.stack(positions),
        torch.stack(charges),
        torch.stack(mono_batch),
        torch.stack(dipole_batch),
        torch.stack(quadrupole_batch),
    )
    mapped_data = torch.func.vmap(_component_tensors)(
        stack(padded), torch.stack(positions)
    )
    for lane, (system, source_position, core) in enumerate(
        zip(systems, source_positions, padded)
    ):
        mapping = dict(bucket.source_maps)[bucket.indices[lane]]
        atom_indices = torch.tensor(mapping.atoms, dtype=torch.int64)
        shell_indices = torch.tensor(mapping.shells, dtype=torch.int64)
        orbital_indices = torch.tensor(mapping.orbitals, dtype=torch.int64)
        serial_cache = _interaction_data(system, source_position)
        serial_functional = prepare_functional_interactions(
            system.interactions,
            serial_cache,
            system.exact_core.numbers,
            d4sc_setup=next(
                entry.setup
                for entry in system.exact_core.interaction_terms
                if entry.kind == "d4sc"
            ),
            kind_by_label=serial_cache.functional_kinds,
        )
        tensor_index = 0
        for serial_entry in serial_functional.entries:
            for name, serial_tensor in serial_entry.data:
                mapped_tensor = mapped_data[tensor_index][lane]
                tensor_index += 1
                if serial_entry.kind in ("es2", "es3"):
                    if serial_entry.option("shell_resolved"):
                        indices = shell_indices
                    else:
                        indices = atom_indices
                    axes = (0, 1) if name == "mat" else (0,)
                elif serial_entry.kind == "aes2":
                    indices = atom_indices
                    axes = (0,) if name in ("dkernel", "qkernel") else (0, 1)
                elif serial_entry.kind == "d4sc":
                    indices = atom_indices
                    axes = (0,) if name == "cn" else (0, 1)
                else:
                    raise AssertionError(serial_entry.kind)
                physical_tensor = mapped_tensor
                for axis in axes:
                    physical_tensor = physical_tensor.index_select(
                        axis, indices
                    )
                torch.testing.assert_close(
                    physical_tensor,
                    serial_tensor,
                    atol=5e-9,
                    rtol=5e-9,
                    msg=lambda message: f"{serial_entry.kind}.{name}: {message}",
                )
        reference = _serial_interaction_observables(
            system,
            source_position,
            charges[lane],
            mono_batch[lane].index_select(0, orbital_indices),
            dipole_batch[lane].index_select(0, atom_indices),
            quadrupole_batch[lane].index_select(0, atom_indices),
        )
        assert len(reference) == len(mapped)
        for mapped_entry, serial_entry in zip(mapped, reference):
            assert (
                sum(torch.linalg.vector_norm(value) for value in serial_entry)
                > 0
            )
            for mapped_value, serial_value in zip(mapped_entry, serial_entry):
                lane_value = mapped_value[lane]
                if lane_value.shape == (bucket.orbitals,):
                    physical = lane_value.index_select(0, orbital_indices)
                elif lane_value.shape[0] == bucket.atoms:
                    physical = lane_value.index_select(0, atom_indices)
                else:
                    physical = lane_value
                torch.testing.assert_close(
                    physical, serial_value, atol=5e-9, rtol=5e-9
                )


def test_gfn2_interaction_potentials_are_energy_derivatives() -> None:
    """Nonzero interaction potentials agree with energy charge derivatives."""
    compositions = ((1, 1), (8, 1, 1))
    _, _, padded, _, positions, charges = _prepare(
        _model(maxiter=1), compositions
    )
    core = padded[1]
    integrals = evaluate_padded_integrals(core, positions[1], charges[1])
    inputs = setup_padded_scf_inputs(
        core, integrals, positions[1], charges[1], torch.zeros_like(charges[1])
    )
    mono = torch.linspace(-0.06, 0.08, core.validity.orbitals.numel())
    mono = mono.to(DTYPE).requires_grad_(True)
    dipole = torch.linspace(-0.03, 0.04, core.validity.atoms.numel() * 3)
    dipole = dipole.to(DTYPE).reshape(-1, 3).requires_grad_(True)
    quadrupole = torch.linspace(-0.02, 0.03, core.validity.atoms.numel() * 6)
    quadrupole = quadrupole.to(DTYPE).reshape(-1, 6).requires_grad_(True)
    for entry in inputs.interactions.entries:
        interaction = FunctionalInteractionSet((entry,))

        def energy(qmono, qdipole, qquadrupole):
            trial = SCFCharges(mono=qmono, dipole=qdipole, quad=qquadrupole)
            return interaction_energy_from_charges(
                trial, interaction, inputs.index
            ).sum()

        gradients = torch.func.grad(energy, argnums=(0, 1, 2))(
            mono, dipole, quadrupole
        )
        trial = SCFCharges(mono=mono, dipole=dipole, quad=quadrupole)
        potential = potential_from_charges(trial, interaction, inputs.index)
        for gradient, expected in zip(
            gradients,
            (potential.mono, potential.dipole, potential.quad),
        ):
            if expected is not None:
                torch.testing.assert_close(
                    gradient,
                    expected,
                    atol=2e-8,
                    rtol=2e-8,
                    msg=f"{entry.kind} potential is not the energy derivative",
                )


def test_gfn2_classical_d4_uses_custom_eeq_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default and custom D4 EEQ cutoffs affect serial and mapped energies."""
    composition = ((8, 1, 1),)
    system = _systems(_model(maxiter=0), composition)[0]
    source_position = _positions(composition)[0]
    source_charge = torch.zeros((), dtype=DTYPE)
    cutoffs = (0.8, 4.0, 25.0)
    observed_charges = []
    observed_energies = []
    cutoff_cores = []

    for cutoff in cutoffs:
        core = _core_with_eeq_cutoff(system.exact_core, cutoff)
        cutoff_cores.append(core)
        bucket = bucket_canonical_exact_cores((core,))
        padded = pad_canonical_exact_core(core, bucket, 0)
        position = canonicalize_positions(source_position, bucket, 0)
        charge = torch.zeros((), dtype=DTYPE)
        spin = torch.zeros((), dtype=DTYPE)
        mapped, decoded = evaluate_padded_systems(
            (padded,), bucket, (position,), (charge,), (spin,)
        )
        descriptor = next(
            term for term in core.classical_terms if term.kind == "d4"
        )
        eeq = get_eeq_guess(
            core.numbers,
            source_position,
            source_charge,
            cutoff=torch.tensor(cutoff, dtype=DTYPE),
        )
        reference_energy = _evaluate_classical_descriptor(
            descriptor, source_position, source_charge, q=eeq
        )
        mapped_d4 = dict(decoded[0].classical)["DispersionD4"]
        torch.testing.assert_close(
            mapped_d4, reference_energy, atol=2e-8, rtol=2e-8
        )
        assert mapped.status is not None

        # Diagnostic: expose the q value delivered by the mapped production
        # boundary to the D4 energy adapter. This makes cutoff forwarding
        # observable even though this GFN2 parameterization has nearly zero
        # classical D4 energy sensitivity to EEQ charge.
        from dxtb._src.calculators import singlepoint as singlepoint_module

        original_classical = singlepoint_module._evaluate_classical_descriptor

        def return_eeq(_descriptor, _positions, _charge, *, q=None):
            if _descriptor.kind == "d4":
                assert q is not None
                return q
            return original_classical(_descriptor, _positions, _charge, q=q)

        with monkeypatch.context() as context:
            context.setattr(
                singlepoint_module,
                "_evaluate_classical_descriptor",
                return_eeq,
            )
            _, captured = evaluate_padded_systems(
                (padded,), bucket, (position,), (charge,), (spin,)
            )
        mapped_eeq = dict(captured[0].classical)["DispersionD4"]
        torch.testing.assert_close(mapped_eeq, eeq, atol=2e-10, rtol=2e-10)
        observed_charges.append(eeq)
        observed_energies.append(reference_energy.sum())
        if cutoff == 4.0:

            def mapped_d4_energy(position):
                output, _ = evaluate_padded_systems(
                    (padded,),
                    bucket,
                    (position,),
                    (charge,),
                    (spin,),
                )
                return dict(output.classical)["DispersionD4"].sum()

            def serial_d4_energy(position):
                local_eeq = get_eeq_guess(
                    core.numbers,
                    position,
                    source_charge,
                    cutoff=torch.tensor(cutoff, dtype=DTYPE),
                )
                return _evaluate_classical_descriptor(
                    descriptor, position, source_charge, q=local_eeq
                ).sum()

            mapped_gradient = torch.func.jacrev(mapped_d4_energy)(position)
            serial_gradient = torch.func.jacrev(serial_d4_energy)(
                source_position
            )
            torch.testing.assert_close(
                mapped_gradient,
                serial_gradient,
                atol=2e-10,
                rtol=2e-8,
            )
            _, mapped_tangent = torch.func.jvp(
                mapped_d4_energy,
                (position,),
                (torch.ones_like(position),),
            )
            torch.testing.assert_close(
                mapped_tangent,
                (mapped_gradient * torch.ones_like(mapped_gradient)).sum(),
                atol=2e-10,
                rtol=2e-8,
            )

    assert any(
        not torch.allclose(left, right, atol=1e-8, rtol=1e-8)
        for left, right in zip(observed_charges, observed_charges[1:])
    )
    assert all(torch.isfinite(value) for value in observed_energies)
    assert all(torch.abs(value) > 0 for value in observed_energies)
    with pytest.raises(TypeError, match="matching static configuration"):
        bucket_canonical_exact_cores(tuple(cutoff_cores))


@pytest.mark.parametrize(
    "compositions",
    [
        ((1, 1), (3, 1)),
        ((1, 1), (8, 1, 1)),
        ((8, 1, 1), (1, 1)),
        ((8, 1, 1), (6, 1, 1, 1, 1)),
    ],
    ids=["h2-lih", "h2-water", "water-h2", "water-methane"],
)
def test_gfn2_finite_temperature_mapped_status_matches_serial(
    compositions: tuple[tuple[int, ...], ...],
) -> None:
    """Finite-temperature GFN2 filling and public status remain lane-local."""
    frontier_h2 = (1, 1) in compositions and (8, 1, 1) in compositions
    # A high but finite temperature yields a robust fractional-occupation
    # control across the supported Torch versions, including H2/LiH.
    temperature = 10_000.0
    systems, bucket, padded, source_positions, positions, zeros = _prepare(
        _model(maxiter=5, temperature=temperature), compositions
    )
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, zeros, zeros
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, zeros, zeros
        )
    )
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)
        assert actual.status is not None
        assert actual.status.trajectory_failure.item() == 0
        assert actual.status.final_failure.item() == 0
    assert mapped.status is not None
    assert mapped.status.trajectory_failure.shape == (2,)
    assert mapped.status.final_failure.shape == (2,)
    assert mapped.occupation is not None
    assert mapped.emo is not None
    from tad_mctc.units.energy import KELVIN2AU

    from dxtb._src.wavefunction import filling

    kt = torch.tensor(temperature * KELVIN2AU, dtype=DTYPE)
    for lane, core in enumerate(padded):
        valid_count = int(core.validity.orbitals.sum())
        torch.testing.assert_close(
            mapped.occupation[lane, ..., valid_count:],
            torch.zeros_like(mapped.occupation[lane, ..., valid_count:]),
            atol=0,
            rtol=0,
        )
        assert torch.all(mapped.emo[lane, ..., valid_count:] > 9_000.0)
        torch.testing.assert_close(
            mapped.occupation[lane].sum(), serial[lane].occupation.sum()
        )
        mapped_entropy = filling.get_fermi_entropy_finite(
            mapped.occupation[lane],
            mapped.emo[lane],
            kt,
            mask=torch.arange(bucket.orbitals) < valid_count,
        ).sum()
        serial_entropy = filling.get_fermi_entropy_finite(
            serial[lane].occupation, serial[lane].emo, kt
        ).sum()
        torch.testing.assert_close(
            mapped_entropy, serial_entropy, atol=2e-10, rtol=2e-8
        )

        fermi_config = systems[lane].config.scf.fermi

        def fermi_state(occupation, energies, mask):
            return filling._get_fermi_occupation_core_state(
                occupation.sum(-1),
                energies.unsqueeze(-2).expand(2, -1),
                kt,
                mask=mask,
                thr=fermi_config.thresh,
                maxiter=fermi_config.maxiter,
                diff_order=fermi_config.diff_order,
            )

        padded_mask = (torch.arange(bucket.orbitals) < valid_count).expand(
            2, -1
        )
        padded_fermi = fermi_state(
            mapped.occupation[lane],
            mapped.emo[lane],
            padded_mask,
        )
        serial_fermi = fermi_state(
            serial[lane].occupation, serial[lane].emo, None
        )
        padded_mu = (
            padded_fermi.chemical_potential_ref
            + padded_fermi.chemical_potential_offset
        )
        serial_mu = (
            serial_fermi.chemical_potential_ref
            + serial_fermi.chemical_potential_offset
        )
        torch.testing.assert_close(padded_mu, serial_mu, atol=2e-10, rtol=2e-8)
        torch.testing.assert_close(
            padded_fermi.result.status.converged,
            serial_fermi.result.status.converged,
        )
    if frontier_h2:
        h2_lane = compositions.index((1, 1))
        assert mapped.emo[h2_lane, 1] > 0
        assert 0 < mapped.occupation[h2_lane, :, 1].min() < 1

        def mapped_thermodynamic_energy(batch_positions):
            output, _ = evaluate_padded_systems(
                padded, bucket, batch_positions, zeros, zeros
            )
            return (output.energy + output.fenergy).sum()

        mapped_gradient = torch.func.grad(mapped_thermodynamic_energy)(
            positions
        )
        for lane, (system, source_position) in enumerate(
            zip(systems, source_positions)
        ):

            def serial_thermodynamic_energy(position):
                result = system.singlepoint(
                    position, chrg=zeros[lane], spin=zeros[lane]
                )
                return (result.energy + result.fenergy).sum()

            serial_gradient = torch.func.grad(serial_thermodynamic_energy)(
                source_position
            )
            mapping = dict(bucket.source_maps)[bucket.indices[lane]]
            atom_indices = torch.tensor(mapping.atoms, dtype=torch.int64)
            torch.testing.assert_close(
                mapped_gradient[lane].index_select(0, atom_indices),
                serial_gradient,
                atol=3e-6,
                rtol=3e-6,
            )


def test_gfn2_zero_trip_result_matches_serial() -> None:
    """The padded GFN2 route retains the shared one-shot contract."""
    compositions = ((1, 1), (8, 1, 1))
    systems, bucket, padded, source_positions, positions, zeros = _prepare(
        _model(maxiter=0), compositions
    )
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, zeros, zeros
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, zeros, zeros
        )
    )
    assert mapped.iterations.shape == (2,)
    assert torch.all(mapped.iterations == 1)
    assert not torch.any(mapped.converged)
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)


def test_gfn2_lane_local_convergence_and_iteration_metadata() -> None:
    """One converged lane cannot terminate the other lane's fixed-trip work."""
    compositions = ((1, 1), (8, 1, 1))
    systems, bucket, padded, source_positions, positions, zeros = _prepare(
        _model(maxiter=18), compositions
    )
    mapped, decoded = evaluate_padded_systems(
        padded, bucket, positions, zeros, zeros
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, zeros, zeros
        )
    )
    assert mapped.converged.tolist() == [True, False]
    assert mapped.iterations.tolist() == [16, 19]
    assert decoded[0].converged.item()
    assert not decoded[1].converged.item()
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)
        assert actual.status is not None
        assert actual.status.trajectory_failure.item() == 0
        assert actual.status.final_failure.item() == 0


def test_gfn2_marginal_lane_is_invariant_to_bucket_padding_capacity() -> None:
    """The same marginal H2 lane is unchanged when its partner enlarges padding."""
    h2_results = []
    for compositions in (((1, 1), (3, 1)), ((1, 1), (8, 1, 1))):
        systems, bucket, padded, source_positions, positions, zeros = _prepare(
            _model(maxiter=18), compositions
        )
        mapped, decoded = evaluate_padded_systems(
            padded, bucket, positions, zeros, zeros
        )
        serial = tuple(
            system.singlepoint(position, chrg=zero, spin=zero)
            for system, position, zero in zip(systems, source_positions, zeros)
        )
        _assert_result_parity(decoded[0], serial[0])
        assert decoded[0].converged.item()
        assert decoded[0].iterations.item() == 16
        h2_results.append(decoded[0])
    _assert_result_parity(h2_results[0], h2_results[1])


def test_gfn2_mixed_charged_open_shell_lanes_match_serial() -> None:
    """Distinct valid positive/negative charges and open-shell spins stay local."""
    compositions = ((1, 1), (1, 1))
    systems, bucket, padded, source_positions, positions, _ = _prepare(
        _model(maxiter=4), compositions
    )
    charges = (
        torch.tensor(1.0, dtype=DTYPE),
        torch.tensor(-1.0, dtype=DTYPE),
    )
    spins = (
        torch.tensor(1.0, dtype=DTYPE),
        torch.tensor(1.0, dtype=DTYPE),
    )
    _, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)
    assert not torch.allclose(decoded[0].energy, decoded[1].energy)


def test_gfn2_too_many_electrons_status_is_lane_local() -> None:
    """A Fermi electron-count failure stays in its own mixed-System lane."""
    compositions = ((1, 1), (1, 1))
    systems, bucket, padded, source_positions, positions, _ = _prepare(
        _model(maxiter=2, temperature=300.0), compositions
    )
    charges = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(-3.0, dtype=DTYPE))
    spins = (torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
    _, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)
        assert actual.status is not None and expected.status is not None
        torch.testing.assert_close(
            actual.status.trajectory_failure, expected.status.trajectory_failure
        )
        torch.testing.assert_close(
            actual.status.final_failure, expected.status.final_failure
        )
    assert decoded[0].status is not None and decoded[1].status is not None
    assert decoded[0].status.trajectory_failure.item() == 0
    assert decoded[0].status.final_failure.item() == 0
    assert decoded[1].status.trajectory_failure.item() == 2
    assert decoded[1].status.final_failure.item() == 2


def test_gfn2_mixed_position_derivatives_match_serial() -> None:
    """Mapped gradients retain physical lanes and mask padded coordinates."""
    compositions = ((1, 1), (3, 1))
    systems, bucket, padded, source_positions, positions, charges = _prepare(
        _model(maxiter=3), compositions
    )
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)

    def mapped_energy(batch_positions):
        mapped, _ = evaluate_padded_systems(
            padded, bucket, batch_positions, charges, spins
        )
        return mapped.energy.sum()

    mapped_gradient = torch.func.grad(mapped_energy)(positions)
    for lane, (system, source_position, lane_gradient) in enumerate(
        zip(systems, source_positions, mapped_gradient)
    ):
        serial_gradient = torch.func.grad(
            lambda position: system.singlepoint(
                position,
                chrg=charges[lane],
                spin=spins[lane],
            ).energy.sum()
        )(source_position)
        source_map = dict(bucket.source_maps)[bucket.indices[lane]]
        atom_indices = torch.tensor(source_map.atoms, dtype=torch.int64)
        torch.testing.assert_close(
            lane_gradient.index_select(0, atom_indices),
            serial_gradient,
            atol=2e-6,
            rtol=2e-6,
        )
        inactive = ~padded[lane].validity.atoms
        torch.testing.assert_close(
            lane_gradient[inactive], torch.zeros_like(lane_gradient[inactive])
        )
        assert torch.linalg.vector_norm(serial_gradient) > 0

    tangent = tuple(torch.ones_like(position) for position in positions)
    _, forward = torch.func.jvp(mapped_energy, (positions,), (tangent,))
    reverse = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(mapped_gradient, tangent)
    )
    torch.testing.assert_close(forward, reverse, atol=2e-6, rtol=2e-6)

    def mapped_energy_vector(batch_positions):
        mapped, _ = evaluate_padded_systems(
            padded, bucket, batch_positions, charges, spins
        )
        return mapped.energy.sum(-1)

    jacobian = torch.stack(
        torch.func.jacfwd(mapped_energy_vector)(positions), dim=1
    )
    assert jacobian.shape == (2, 2, bucket.atoms, 3)
    torch.testing.assert_close(
        jacobian[0, 1], torch.zeros_like(jacobian[0, 1]), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(
        jacobian[1, 0], torch.zeros_like(jacobian[1, 0]), atol=2e-6, rtol=0
    )


def test_gfn2_mixed_h0_parameter_gradient_matches_serial_sum() -> None:
    """Mapped total-energy gradients remain attached to Model-owned GFN2 H0."""
    compositions = ((1, 1), (3, 1))
    model = _model(maxiter=3, differentiable_h0=True)
    systems, bucket, padded, source_positions, positions, charges = _prepare(
        model, compositions
    )
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    parameter = model.par.get("hamiltonian.xtb.wexp")
    identity = id(parameter)
    before = parameter.detach().clone()
    mapped, _ = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    mapped_gradient = torch.autograd.grad(
        mapped.energy.sum(), parameter, retain_graph=True
    )[0]
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    serial_gradient = torch.autograd.grad(
        torch.stack(tuple(result.energy.sum() for result in serial)).sum(),
        parameter,
    )[0]
    torch.testing.assert_close(
        mapped_gradient, serial_gradient, atol=2e-6, rtol=2e-6
    )
    assert torch.isfinite(mapped_gradient).all()
    assert torch.count_nonzero(mapped_gradient) > 0
    assert id(model.par.get("hamiltonian.xtb.wexp")) == identity
    torch.testing.assert_close(parameter.detach(), before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "path,energy_field",
    [
        (("element", "O", "gam"), "energy"),
        (("element", "O", "gam3"), "energy"),
        (("multipole", "damped", "dmp3"), "energy"),
        (("dispersion", "d4", "a1"), "scf"),
        (("dispersion", "d4", "a1"), "DispersionD4"),
        (("element", "O", "arep"), "Repulsion"),
    ],
    ids=["es2", "es3", "aes2", "d4sc", "classical-d4", "repulsion"],
)
def test_gfn2_component_parameter_gradients_match_serial_sum(
    path: tuple[str, ...], energy_field: str
) -> None:
    """Active Model-owned interaction/classical leaves retain mapped gradients."""
    compositions = ((1, 1), (8, 1, 1))
    model = _model(maxiter=3, differentiable_path=path)
    systems, bucket, padded, source_positions, positions, charges = _prepare(
        model, compositions
    )
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    parameter = model.par.get(*path)
    assert parameter.requires_grad
    identity = id(parameter)
    before = parameter.detach().clone()

    def select(result):
        if energy_field in ("DispersionD4", "Repulsion"):
            return dict(result.classical)[energy_field].sum()
        return getattr(result, energy_field).sum()

    mapped, _ = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    mapped_gradient = torch.autograd.grad(
        select(mapped), parameter, retain_graph=True
    )[0]
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    serial_gradient = torch.autograd.grad(
        torch.stack(tuple(select(result) for result in serial)).sum(), parameter
    )[0]
    torch.testing.assert_close(
        mapped_gradient, serial_gradient, atol=2e-7, rtol=2e-6
    )
    assert torch.isfinite(mapped_gradient).all()
    assert torch.count_nonzero(mapped_gradient) > 0
    assert id(model.par.get(*path)) == identity
    torch.testing.assert_close(parameter.detach(), before, atol=0, rtol=0)


def test_gfn2_mixed_position_es2_parameter_derivative_matches_serial() -> None:
    """A mixed position/ES2-parameter derivative stays attached through vmap."""
    compositions = ((1, 1), (8, 1, 1))
    path = ("element", "O", "gam")
    model = _model(maxiter=2, differentiable_path=path)
    systems, bucket, padded, source_positions, positions, charges = _prepare(
        model, compositions
    )
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    position_leaves = tuple(
        position.clone().requires_grad_(True) for position in positions
    )
    parameter = model.par.get(*path)
    mapped, _ = evaluate_padded_systems(
        padded, bucket, position_leaves, charges, spins
    )
    mapped_total = (mapped.energy + mapped.fenergy).sum()
    mapped_position_gradients = torch.autograd.grad(
        mapped_total,
        position_leaves,
        create_graph=True,
        retain_graph=True,
    )
    direction = tuple(torch.ones_like(position) for position in position_leaves)
    mapped_directional = sum(
        (gradient * vector).sum()
        for gradient, vector in zip(mapped_position_gradients, direction)
    )
    mapped_mixed = torch.autograd.grad(
        mapped_directional, parameter, retain_graph=True
    )[0]

    serial_position_leaves = tuple(
        position.clone().requires_grad_(True) for position in source_positions
    )
    serial_results = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, serial_position_leaves, charges, spins
        )
    )
    serial_position_gradients = tuple(
        torch.autograd.grad(
            result.energy.sum() + result.fenergy.sum(),
            position,
            create_graph=True,
            retain_graph=True,
        )[0]
        for result, position in zip(serial_results, serial_position_leaves)
    )
    serial_directional = sum(
        (gradient * torch.ones_like(gradient)).sum()
        for gradient in serial_position_gradients
    )
    serial_mixed = torch.autograd.grad(serial_directional, parameter)[0]
    torch.testing.assert_close(mapped_mixed, serial_mixed, atol=3e-6, rtol=3e-5)
    assert torch.isfinite(mapped_mixed).all()
    assert torch.count_nonzero(mapped_mixed) > 0


def test_gfn2_mixed_energy_hessian_vector_matches_serial() -> None:
    """A full mixed-GFN2 energy HVP matches independent serial Systems."""
    compositions = ((1, 1), (3, 1))
    systems, bucket, padded, source_positions, positions, charges = _prepare(
        _model(maxiter=2), compositions
    )
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)

    def mapped_energy(batch_positions):
        result, _ = evaluate_padded_systems(
            padded, bucket, batch_positions, charges, spins
        )
        return result.energy.sum()

    directions = []
    source_directions = []
    mappings = dict(bucket.source_maps)
    for lane, (position, source_position) in enumerate(
        zip(positions, source_positions)
    ):
        source_direction = torch.linspace(
            -0.3, 0.4, source_position.numel(), dtype=DTYPE
        ).reshape_as(source_position)
        source_directions.append(source_direction)
        padded_direction = torch.zeros_like(position)
        atom_indices = torch.tensor(
            mappings[bucket.indices[lane]].atoms, dtype=torch.int64
        )
        padded_direction = padded_direction.index_copy(
            0, atom_indices, source_direction
        )
        directions.append(padded_direction)

    _, mapped_hvp = torch.func.jvp(
        torch.func.grad(mapped_energy), (positions,), (tuple(directions),)
    )
    for lane, (system, source_position, source_direction) in enumerate(
        zip(systems, source_positions, source_directions)
    ):

        def serial_energy(position):
            return system.singlepoint(
                position,
                chrg=charges[lane],
                spin=spins[lane],
            ).energy.sum()

        _, serial_hvp = torch.func.jvp(
            torch.func.grad(serial_energy),
            (source_position,),
            (source_direction,),
        )
        atom_indices = torch.tensor(
            mappings[bucket.indices[lane]].atoms, dtype=torch.int64
        )
        torch.testing.assert_close(
            mapped_hvp[lane].index_select(0, atom_indices),
            serial_hvp,
            atol=2e-12,
            rtol=2e-12,
        )
        assert torch.linalg.vector_norm(serial_hvp) > 0


@pytest.mark.parametrize("mixer", ["linear", "anderson"])
@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
def test_gfn2_supported_scp_mixer_pairs_match_serial(
    mixer: str, scp_mode: str
) -> None:
    """The existing fixed-trip solver remains shared across GFN2 modes."""
    compositions = ((1, 1), (3, 1))
    systems, bucket, padded, source_positions, positions, charges = _prepare(
        _model(maxiter=3, mixer=mixer, scp_mode=scp_mode), compositions
    )
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    _, decoded = evaluate_padded_systems(
        padded, bucket, positions, charges, spins
    )
    serial = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    for actual, expected in zip(decoded, serial):
        _assert_result_parity(actual, expected)


def test_gfn2_solver_trace_matches_independent_serial_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapped intermediate matrices and solver controls match serial traces."""
    from tad_mctc.tree import stack as tree_stack

    from dxtb._src.scf.functional import eager as eager_functional

    compositions = ((1, 1), (8, 1, 1))
    model = _model(maxiter=30)
    systems = _systems(model, compositions)
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, lane)
        for lane, core in enumerate(cores)
    )
    source_positions = _positions(compositions)
    positions = tuple(
        canonicalize_positions(position, bucket, lane)
        for lane, position in enumerate(source_positions)
    )
    charges = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    spins = tuple(torch.zeros((), dtype=DTYPE) for _ in systems)
    d4_cutoff = float(
        next(
            descriptor.setup.cutoff.cn_eeq
            for descriptor in padded[0].core.classical_terms
            if descriptor.kind == "d4"
        )
    )
    original_solve = eager_functional._solve_representation
    serial_traces = []

    def traced_serial_solve(*args, **kwargs):
        solved = original_solve(*args, record_trace=True, **kwargs)
        serial_traces.append(solved.trace)
        return solved

    monkeypatch.setattr(
        eager_functional, "_solve_representation", traced_serial_solve
    )
    serial_results = tuple(
        system.singlepoint(position, chrg=charge, spin=spin)
        for system, position, charge, spin in zip(
            systems, source_positions, charges, spins
        )
    )
    mapped_results, mapped_trace = torch.func.vmap(
        lambda core, position, charge, spin: evaluate_padded_fixed_trip(
            core,
            position,
            charge,
            spin,
            record_trace=True,
            d4_eeq_cutoff=d4_cutoff,
        )
    )(
        tree_stack(padded),
        torch.stack(positions),
        torch.stack(charges),
        torch.stack(spins),
    )
    assert torch.any(
        mapped_trace.active_after[0] != mapped_trace.active_after[1]
    ), mapped_trace.residuals
    assert len(serial_traces) == len(systems)
    assert all(trace is not None for trace in serial_traces)
    for lane, (serial_result, serial_trace) in enumerate(
        zip(serial_results, serial_traces)
    ):
        assert serial_trace is not None
        mapping = dict(bucket.source_maps)[bucket.indices[lane]]
        atom_indices = torch.tensor(mapping.atoms, dtype=torch.int64)
        orbital_indices = torch.tensor(mapping.orbitals, dtype=torch.int64)
        torch.testing.assert_close(
            mapped_results.energy[lane].index_select(0, atom_indices),
            serial_result.energy,
            atol=5e-8,
            rtol=5e-8,
        )
        for name in ("hamiltonians", "densities"):
            physical = getattr(mapped_trace, name)[lane]
            physical = physical.index_select(-2, orbital_indices).index_select(
                -1, orbital_indices
            )
            torch.testing.assert_close(
                physical,
                getattr(serial_trace, name),
                atol=2e-8,
                rtol=2e-8,
            )
        for name in (
            "residuals",
            "active_before",
            "active_after",
            "iterations",
            "damping",
        ):
            torch.testing.assert_close(
                getattr(mapped_trace, name)[lane],
                getattr(serial_trace, name),
                atol=2e-10,
                rtol=2e-10,
            )
        for name in (
            "fermi_converged",
            "fermi_electron_count_ok",
            "fermi_invalid_temperature",
            "fermi_too_many_electrons",
            "fermi_residual",
        ):
            mapped_status = getattr(mapped_trace, name)
            serial_status = getattr(serial_trace, name)
            assert (mapped_status is None) == (serial_status is None)
            if mapped_status is not None and serial_status is not None:
                torch.testing.assert_close(
                    mapped_status[lane],
                    serial_status,
                    atol=2e-10,
                    rtol=2e-10,
                )
