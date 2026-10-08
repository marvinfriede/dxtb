# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from dxtb._src.integral.container import IntegralMatrices


def test_integral_matrices_are_immutable_and_convert_by_value() -> None:
    hcore = torch.eye(3, dtype=torch.float64)
    value = IntegralMatrices(hcore=hcore, overlap=hcore.clone())

    with pytest.raises(FrozenInstanceError):
        value.hcore = torch.zeros_like(hcore)  # type: ignore[misc]

    converted = value.to(torch.device("cpu"), torch.float32)
    assert converted is not value
    assert converted.dtype == torch.float32
    assert value.dtype == torch.float64
    assert value.type(torch.float32).dtype == torch.float32


def test_integral_matrices_validate_shapes_and_components() -> None:
    hcore = torch.eye(2)
    overlap = hcore.clone()
    with pytest.raises(ValueError, match="match the hcore shape"):
        IntegralMatrices(hcore=hcore, overlap=torch.eye(3))

    for ncomp in (6, 9):
        quad = torch.zeros(ncomp, 2, 2)
        assert (
            IntegralMatrices(
                hcore=hcore, overlap=overlap, quadrupole=quad
            ).quadrupole
            is quad
        )
        batched_quad = quad.unsqueeze(0).expand(2, -1, -1, -1)
        batch = torch.eye(2).expand(2, -1, -1)
        assert (
            IntegralMatrices(
                hcore=batch, overlap=batch, quadrupole=batched_quad
            ).quadrupole
            is batched_quad
        )

    with pytest.raises(ValueError, match="incompatible shape"):
        IntegralMatrices(
            hcore=hcore, overlap=overlap, quadrupole=torch.zeros(8, 2, 2)
        )


def test_integral_matrices_slice_returns_valid_value() -> None:
    hcore = torch.eye(3).expand(2, -1, -1)
    dipole = torch.zeros(2, 3, 3, 3)
    quad = torch.zeros(2, 6, 3, 3)
    value = IntegralMatrices(
        hcore=hcore, overlap=hcore, dipole=dipole, quadrupole=quad
    )
    sliced = value.slice(
        (torch.tensor([True, False]), slice(None, 2), slice(None, 2)),
        (torch.tensor([True, False]), Ellipsis, slice(None, 2), slice(None, 2)),
    )
    assert sliced.hcore.shape == (1, 2, 2)
    assert sliced.dipole is not None and sliced.dipole.shape == (1, 3, 2, 2)
    assert sliced.quadrupole is not None and sliced.quadrupole.shape == (
        1,
        6,
        2,
        2,
    )
    assert value.hcore.shape == (2, 3, 3)
