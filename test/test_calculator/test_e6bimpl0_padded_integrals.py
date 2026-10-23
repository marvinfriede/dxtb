# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

"""Numerical setup parity for canonically padded exact Systems."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.padded_integrals import (
    PaddedIntegralEvaluation,
    evaluate_padded_integrals,
)
from dxtb._src.integral.evaluation import build_integral_matrices

DTYPE = torch.float64


def _positions(numbers: tuple[int, ...]) -> torch.Tensor:
    coordinates = {
        (1, 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        (1, 1, 1): [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [-0.3, 0.8, 0.7],
        ],
        (8, 1, 1): [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.6],
            [-0.8, 0.0, 0.6],
        ],
        (1, 1, 8): [
            [0.8, 0.0, 0.6],
            [-0.8, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ],
    }
    return torch.tensor(coordinates[numbers], dtype=DTYPE)


def _model(
    *, level: int = labels.INTLEVEL_HCORE, differentiable_h0: bool = False
) -> Model:
    parameters = ParamModule(GFN1_XTB, dtype=DTYPE)
    if differentiable_h0:
        parameters.set_differentiable("hamiltonian", "xtb", "kpol")
    return Model(
        par=parameters,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=level,
        ),
        auto_int_level=False,
    )


def _sources(
    numbers: tuple[tuple[int, ...], ...], *, level: int, differentiable=False
):
    model = _model(level=level, differentiable_h0=differentiable)
    systems = tuple(
        model.setup(torch.tensor(entry, dtype=torch.int64)) for entry in numbers
    )
    assert all(system.exact_core is not None for system in systems)
    return model, systems


def _mapped(systems, numbers, positions):
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, i)
        for i, core in enumerate(cores)
    )
    canonical_positions = tuple(
        canonicalize_positions(position, bucket, i)
        for i, position in enumerate(positions)
    )
    mapped = torch.func.vmap(evaluate_padded_integrals)(
        stack(padded),
        torch.stack(canonical_positions),
        torch.zeros(len(systems), dtype=DTYPE),
    )
    return bucket, padded, canonical_positions, mapped


@pytest.mark.parametrize(
    "numbers",
    [
        ((1, 1), (1, 1, 1)),
        ((1, 1), (8, 1, 1)),
        ((8, 1, 1), (1, 1)),
        ((1, 1), (1, 1, 8)),
    ],
    ids=["h2-h3", "h2-water", "water-h2", "h2-water-reordered"],
)
@pytest.mark.parametrize(
    "level",
    [
        labels.INTLEVEL_HCORE,
        labels.INTLEVEL_DIPOLE,
        labels.INTLEVEL_QUADRUPOLE,
    ],
    ids=["hcore", "dipole", "quadrupole"],
)
def test_padded_mapped_integrals_match_independent_systems(numbers, level):
    """Mapped padded matrices reproduce each source's physical AO block."""
    _, systems = _sources(numbers, level=level)
    positions = tuple(_positions(entry) for entry in numbers)
    bucket, padded, _, mapped = _mapped(systems, numbers, positions)

    assert mapped.hcore.shape == (
        len(systems),
        bucket.orbitals,
        bucket.orbitals,
    )
    assert mapped.overlap.shape == mapped.hcore.shape
    for source_index, (system, position, padded_system) in enumerate(
        zip(systems, positions, padded)
    ):
        result = PaddedIntegralEvaluation(
            hcore=mapped.hcore[source_index],
            overlap=mapped.overlap[source_index],
            dipole=(
                None if mapped.dipole is None else mapped.dipole[source_index]
            ),
            quadrupole=(
                None
                if mapped.quadrupole is None
                else mapped.quadrupole[source_index]
            ),
            reference_occupations=mapped.reference_occupations[source_index],
            overlap_norm=mapped.overlap_norm[source_index],
        )
        assert system.integral_setup is not None
        assert system.h0_setup is not None
        reference, reference_occupations, reference_norm = (
            build_integral_matrices(
                system.integral_setup,
                system.h0_setup,
                position,
                torch.zeros((), dtype=DTYPE),
            )
        )
        mapping = dict(bucket.source_maps)[source_index]
        ao = torch.tensor(mapping.orbitals, dtype=torch.int64)
        shells = torch.tensor(mapping.shell_types, dtype=torch.int64)
        torch.testing.assert_close(
            result.hcore.index_select(-2, ao).index_select(-1, ao),
            reference.hcore,
            atol=2e-12,
            rtol=2e-12,
        )
        torch.testing.assert_close(
            result.overlap.index_select(-2, ao).index_select(-1, ao),
            reference.overlap,
            atol=2e-12,
            rtol=2e-12,
        )
        torch.testing.assert_close(
            result.reference_occupations.index_select(0, shells),
            reference_occupations,
        )
        torch.testing.assert_close(
            result.overlap_norm.index_select(0, ao), reference_norm
        )
        assert (result.dipole is not None) == (reference.dipole is not None)
        assert (result.quadrupole is not None) == (
            reference.quadrupole is not None
        )
        if reference.dipole is not None:
            torch.testing.assert_close(
                result.dipole.index_select(-2, ao).index_select(-1, ao),
                reference.dipole,
                atol=2e-12,
                rtol=2e-12,
            )
        if reference.quadrupole is not None:
            torch.testing.assert_close(
                result.quadrupole.index_select(-2, ao).index_select(-1, ao),
                reference.quadrupole,
                atol=2e-12,
                rtol=2e-12,
            )
        inactive = ~padded_system.validity.orbitals
        inactive_shells = ~padded_system.validity.unique_shell_types
        assert torch.count_nonzero(result.hcore[inactive, :]) == 0
        assert torch.count_nonzero(result.hcore[:, inactive]) == 0
        assert torch.count_nonzero(result.overlap[inactive, :]) == 0
        assert torch.count_nonzero(result.overlap[:, inactive]) == 0
        assert (
            torch.count_nonzero(result.reference_occupations[inactive_shells])
            == 0
        )
        assert torch.count_nonzero(result.overlap_norm[inactive]) == 0
        assert torch.isfinite(result.hcore).all()
        assert torch.isfinite(result.overlap).all()


def test_mapped_position_derivatives_match_serial_and_have_no_cross_lane_terms():
    """Mapped H0 derivatives match independent sources lane by lane."""
    numbers = ((1, 1), (8, 1, 1))
    _, systems = _sources(numbers, level=labels.INTLEVEL_HCORE)
    source_positions = tuple(
        position.requires_grad_() for position in map(_positions, numbers)
    )
    bucket, padded, canonical_positions, _ = _mapped(
        systems, numbers, source_positions
    )
    stacked_core = stack(padded)
    positions = torch.stack(canonical_positions)
    charges = torch.zeros(len(systems), dtype=DTYPE)

    def lane_values(position_batch):
        mapped = torch.func.vmap(evaluate_padded_integrals)(
            stacked_core, position_batch, charges
        )
        return mapped.hcore.sum(dim=(-2, -1))

    jacobian = torch.func.jacrev(lane_values)(positions)
    forward = torch.func.jacfwd(lane_values)(positions)
    torch.testing.assert_close(jacobian, forward, atol=2e-10, rtol=2e-10)
    assert jacobian.shape == (
        len(systems),
        len(systems),
        bucket.atoms,
        3,
    )

    for i, (system, position) in enumerate(zip(systems, source_positions)):
        assert system.integral_setup is not None and system.h0_setup is not None

        def serial_value(x):
            matrices, _, _ = build_integral_matrices(
                system.integral_setup,
                system.h0_setup,
                x,
                torch.zeros((), dtype=DTYPE),
            )
            return matrices.hcore.sum()

        serial_gradient = torch.func.grad(serial_value)(position)
        mapping = dict(bucket.source_maps)[i]
        diagonal = jacobian[i, i].index_select(
            0, torch.tensor(mapping.atoms, dtype=torch.int64)
        )
        torch.testing.assert_close(
            diagonal, serial_gradient, atol=2e-9, rtol=2e-9
        )
        inactive_atoms = sorted(set(range(bucket.atoms)) - set(mapping.atoms))
        if inactive_atoms:
            assert (
                torch.count_nonzero(
                    jacobian[i, i].index_select(
                        0, torch.tensor(inactive_atoms, dtype=torch.int64)
                    )
                )
                == 0
            )
        for j in range(len(systems)):
            if i != j:
                assert torch.count_nonzero(jacobian[i, j]) == 0

    direction = torch.randn_like(positions)
    _, tangent = torch.func.jvp(lane_values, (positions,), (direction,))
    torch.testing.assert_close(
        tangent,
        torch.einsum("bijc,bjc->b", jacobian, direction),
        atol=2e-9,
        rtol=2e-9,
    )


def test_mapped_h0_parameter_gradient_reaches_single_model_owner():
    """The shared Model-owned kpol gradient survives setup, padding and vmap."""
    numbers = ((1, 1), (8, 1, 1))
    model, systems = _sources(
        numbers, level=labels.INTLEVEL_HCORE, differentiable=True
    )
    parameter = model.par.get("hamiltonian.xtb.kpol")
    source_snapshots = tuple(
        (
            tuple(id(value) for value in tree_flatten(system.exact_core)[0]),
            tuple(
                value.detach().clone()
                for value in tree_flatten(system.exact_core)[0]
            ),
        )
        for system in systems
    )
    parameter_snapshot = parameter.detach().clone()
    positions = tuple(_positions(entry) for entry in numbers)
    bucket, padded, canonical_positions, mapped = _mapped(
        systems, numbers, positions
    )
    assert mapped.hcore.requires_grad
    mapped_gradient = torch.autograd.grad(
        mapped.hcore.sum(), parameter, retain_graph=True
    )[0]

    serial_energies = []
    for system, position in zip(systems, positions):
        assert system.integral_setup is not None and system.h0_setup is not None
        matrices, _, _ = build_integral_matrices(
            system.integral_setup,
            system.h0_setup,
            position,
            torch.zeros((), dtype=DTYPE),
        )
        serial_energies.append(matrices.hcore.sum())
    serial_gradient = torch.autograd.grad(sum(serial_energies), parameter)[0]
    torch.testing.assert_close(mapped_gradient, serial_gradient)
    assert tuple(model.par.parameters())
    assert all(system.exact_core is not None for system in systems)
    assert id(model.par.get("hamiltonian.xtb.kpol")) == id(parameter)
    torch.testing.assert_close(parameter, parameter_snapshot)
    for system, (identities, values) in zip(systems, source_snapshots):
        assert system.exact_core is not None
        leaves = tuple(tree_flatten(system.exact_core)[0])
        assert tuple(id(value) for value in leaves) == identities
        for current, before in zip(leaves, values):
            torch.testing.assert_close(current, before)
        assert not hasattr(system.exact_core, "par")
    assert bucket.orbitals == mapped.hcore.shape[-1]


def test_padded_integral_evaluator_rejects_invalid_inputs():
    """Malformed core setup and call tensors fail before numerical kernels."""
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    system = _model().setup(numbers)
    assert system.exact_core is not None
    bucket = bucket_canonical_exact_cores((system.exact_core,))
    padded = pad_canonical_exact_core(system.exact_core, bucket, 0)
    positions = canonicalize_positions(_positions((1, 1)), bucket, 0)
    charge = torch.zeros((), dtype=DTYPE)

    with pytest.raises(ValueError, match="Positions do not match"):
        evaluate_padded_integrals(padded, positions[:1], charge)
    with pytest.raises(TypeError, match="effective dtype"):
        evaluate_padded_integrals(padded, positions.float(), charge.float())
    with pytest.raises(ValueError, match="scalar charge"):
        evaluate_padded_integrals(padded, positions, charge.reshape(1))

    with pytest.raises(ValueError, match="H0 setup"):
        evaluate_padded_integrals(
            padded.replace(core=padded.core.replace(h0_setup=None)),
            positions,
            charge,
        )
    with pytest.raises(ValueError, match="integral setup"):
        evaluate_padded_integrals(
            padded.replace(core=padded.core.replace(integral=None)),
            positions,
            charge,
        )
    with pytest.raises(ValueError, match="HCORE"):
        evaluate_padded_integrals(
            padded.replace(
                core=padded.core.replace(
                    integral=padded.core.integral.replace(
                        intlevel=labels.INTLEVEL_OVERLAP
                    )
                )
            ),
            positions,
            charge,
        )
    with pytest.raises(TypeError, match="does not match driver"):
        padded.core.integral.replace(driver_type=labels.INTDRIVER_LIBCINT)

    masks = padded.validity.pair_entries
    malformed = (
        torch.zeros(masks[0].numel() + 1, dtype=torch.bool),
        *masks[1:],
    )
    with pytest.raises(ValueError, match="entry mask has an invalid shape"):
        evaluate_padded_integrals(
            padded.replace(
                validity=padded.validity.replace(pair_entries=malformed)
            ),
            positions,
            charge,
        )


def test_gfn2_canonical_integral_path_builds_the_registered_core():
    """GFN2 exact cores enter the canonical numerical setup path."""
    model = Model(
        par=ParamModule(GFN2_XTB, dtype=DTYPE),
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_HCORE,
        ),
        auto_int_level=False,
    )
    systems = tuple(
        model.setup(torch.tensor(numbers, dtype=torch.int64))
        for numbers in ((1, 1), (8, 1, 1))
    )
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, index)
        for index, core in enumerate(cores)
    )
    positions = tuple(_positions(numbers) for numbers in ((1, 1), (8, 1, 1)))
    canonical = torch.stack(
        tuple(
            canonicalize_positions(position, bucket, index)
            for index, position in enumerate(positions)
        )
    )
    result = torch.func.vmap(evaluate_padded_integrals)(
        stack(padded), canonical, torch.zeros(2, dtype=DTYPE)
    )
    assert result.hcore.shape[-1] == bucket.orbitals
