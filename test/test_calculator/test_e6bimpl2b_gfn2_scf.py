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
from dxtb._src.scf.functional.core import FunctionalInteractionSet
from dxtb._src.scf.functional.data import SCFCharges
from dxtb._src.scf.functional.interactions import (
    interaction_energy_from_charges,
    potential_from_charges,
)
from dxtb._src.scf.functional.padded import (
    _prepare_padded_interactions,
    evaluate_padded_fixed_trip,
    evaluate_padded_systems,
    setup_padded_scf_inputs,
)

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
) -> Model:
    """Construct a production GFN2 Model with the requested fixed-trip setup."""
    parameters = ParamModule(GFN2_XTB, dtype=DTYPE)
    if differentiable_h0:
        parameters.set_differentiable("hamiltonian", "xtb", "wexp")
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


def test_gfn2_finite_temperature_mapped_status_matches_serial() -> None:
    """Finite-temperature GFN2 filling and public status remain lane-local."""
    compositions = ((1, 1), (8, 1, 1))
    systems, bucket, padded, source_positions, positions, zeros = _prepare(
        _model(maxiter=5, temperature=300.0), compositions
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
