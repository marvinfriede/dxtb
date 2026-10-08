# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

from __future__ import annotations

import pytest
import torch

from dxtb import Calculator, GFN2_XTB, labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.typing import Tensor

from .samples import samples


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_integral_matrices_are_history_independent(driver: int) -> None:
    """One System setup produces current-geometry matrices on every call."""
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": driver},
    )
    setup = calc.system.integral_setup
    h0_setup = calc.system.h0_setup
    assert setup is not None and h0_setup is not None
    if setup.libcint is not None:
        assert not hasattr(setup.libcint, "par")
        assert all(
            not hasattr(basis, "par") for basis in setup.libcint.basis_setups
        )

    moved = positions.clone()
    moved[1, 0] += 0.08

    first, refocc1, norm1 = build_integral_matrices(
        setup, h0_setup, positions, charge=0.0
    )
    second, refocc2, norm2 = build_integral_matrices(
        setup, h0_setup, moved, charge=0.0
    )
    # Mutating compatibility builders in a different order must not affect
    # the next pure evaluation from this System setup.
    calc.integrals.build_quadrupole(moved)
    calc.integrals.build_overlap(positions)
    calc.integrals.build_dipole(positions)
    repeated, refocc3, norm3 = build_integral_matrices(
        setup, h0_setup, positions.clone(), charge=0.0
    )

    for name in ("overlap", "dipole", "quadrupole", "hcore"):
        first_matrix = getattr(first, name)
        second_matrix = getattr(second, name)
        repeated_matrix = getattr(repeated, name)
        assert first_matrix is not None
        assert second_matrix is not None
        assert repeated_matrix is not None
        torch.testing.assert_close(repeated_matrix, first_matrix)
        assert not torch.equal(second_matrix, first_matrix)
    torch.testing.assert_close(refocc1, refocc2)
    torch.testing.assert_close(refocc1, refocc3)
    torch.testing.assert_close(norm1, norm3)
    assert torch.isfinite(norm1).all()
    assert torch.isfinite(norm2).all()

    def loss(geometry: Tensor) -> Tensor:
        matrices, refocc, _ = build_integral_matrices(
            setup, h0_setup, geometry, charge=0.0
        )
        assert matrices.dipole is not None
        assert matrices.quadrupole is not None
        return (
            matrices.overlap.sum()
            + matrices.dipole.sum()
            + matrices.quadrupole.sum()
            + matrices.hcore.sum()
            + refocc.sum()
        )

    leaf1 = positions.clone().requires_grad_()
    grad1 = torch.autograd.grad(loss(leaf1), leaf1)[0]
    leaf2 = positions.clone().requires_grad_()
    grad2 = torch.autograd.grad(loss(leaf2), leaf2)[0]
    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()
    torch.testing.assert_close(grad1, grad2)


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_pure_integral_matrices_match_legacy_adapter(driver: int) -> None:
    """The legacy adapter agrees with the pure evaluation outputs.

    This is an adapter-consistency check. Independent physics values are
    covered by the existing end-to-end reference baseline.
    """
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={
            "verbosity": 0,
            "int_driver": driver,
            "int_level": labels.INTLEVEL_QUADRUPOLE,
        },
    )
    setup = calc.system.integral_setup
    h0_setup = calc.system.h0_setup
    assert setup is not None and h0_setup is not None

    legacy_overlap = calc.integrals.build_overlap(positions)
    assert calc.integrals.overlap is not None
    legacy_norm = calc.integrals.overlap.norm.clone()
    calc.integrals.build_dipole(positions)
    legacy_quadrupole = calc.integrals.build_quadrupole(positions)
    # At quadrupole level the legacy adapter intentionally returns the
    # origin-centered dipole from build_dipole and shifts its stored value
    # only while building the quadrupole.
    assert calc.integrals.dipole is not None
    legacy_dipole = calc.integrals.dipole.matrix
    legacy_hcore = calc.integrals.build_hcore(
        positions, with_overlap=True, charge=0.0
    )
    matrices, refocc, overlap_norm = build_integral_matrices(
        setup, h0_setup, positions, charge=0.0
    )

    torch.testing.assert_close(matrices.overlap, legacy_overlap)
    torch.testing.assert_close(overlap_norm, legacy_norm)
    assert matrices.dipole is not None and calc.integrals.dipole is not None
    torch.testing.assert_close(matrices.dipole, legacy_dipole)
    assert (
        matrices.quadrupole is not None
        and calc.integrals.quadrupole is not None
    )
    torch.testing.assert_close(matrices.quadrupole, legacy_quadrupole)
    torch.testing.assert_close(matrices.hcore, legacy_hcore)
    torch.testing.assert_close(refocc, calc.integrals.hcore.refocc)
