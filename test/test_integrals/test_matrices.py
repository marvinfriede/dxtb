# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from dxtb._src.integral.container import IntegralMatrices


def test_integral_matrices_is_minimal_frozen_value_container() -> None:
    """IntegralMatrices stores hcore, overlap, dipole, and quadrupole without extra methods."""
    hcore = torch.eye(3, dtype=torch.float64)
    overlap = hcore.clone()
    value = IntegralMatrices(hcore=hcore, overlap=overlap)

    assert value.hcore is hcore
    assert value.overlap is overlap
    assert value.dipole is None
    assert value.quadrupole is None

    with pytest.raises(FrozenInstanceError):
        value.hcore = torch.zeros_like(hcore)  # type: ignore[misc]

    for removed_attr in ("to", "type", "slice", "device", "dtype"):
        assert not hasattr(value, removed_attr)


def test_integral_matrices_stores_optional_multipole_tensors() -> None:
    """IntegralMatrices preserves exact tensor identity for optional multipoles."""
    hcore = torch.eye(2, dtype=torch.float64)
    overlap = hcore.clone()
    dipole = torch.zeros(3, 2, 2, dtype=torch.float64)
    quad = torch.zeros(6, 2, 2, dtype=torch.float64)

    value = IntegralMatrices(
        hcore=hcore, overlap=overlap, dipole=dipole, quadrupole=quad
    )
    assert value.hcore is hcore
    assert value.overlap is overlap
    assert value.dipole is dipole
    assert value.quadrupole is quad
