# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""
Run tests for overlap of diatomic systems.
References calculated with tblite 0.3.0.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.basis.slater import slater_to_gauss
from dxtb._src.typing.exceptions import (
    CGTOAzimuthalQuantumNumberError,
    CGTOPrimitivesError,
    CGTOPrincipalQuantumNumberError,
    CGTOQuantumNumberError,
    CGTOSlaterExponentsError,
    IntegralTransformError,
)

from ..utils import overlap_1d


def test_fail_number_primitives() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOPrimitivesError):
        slater_to_gauss(torch.tensor(7), n, l, torch.tensor(1.2))


def test_fail_slater_exponent() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOSlaterExponentsError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(-1.2))


def test_fail_max_principal() -> None:
    # principal and azimuthal quantum number of 7s-orbital
    n, l = torch.tensor(7), torch.tensor(0)

    with pytest.raises(CGTOPrincipalQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital() -> None:
    # principal and azimuthal quantum number of 5h-orbital
    n, l = torch.tensor(5), torch.tensor(5)

    with pytest.raises(CGTOAzimuthalQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_quantum_number() -> None:
    # principal and azimuthal quantum number of 2f-orbital
    n, l = torch.tensor(2), torch.tensor(3)

    with pytest.raises(CGTOQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital_trafo():
    """No higher orbitals than d-orbitals allowed."""
    vec = torch.tensor([0.0, 0.0, 1.4])

    # arbitrary element (Rn)
    number = torch.tensor([86])

    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)
    bas = Basis(number, GFN1_XTB, ihelp)
    alpha, coeff = bas.create_cgtos()

    j = torch.tensor(5)
    for i in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_1d(
                (torch.tensor(i), j),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )
    i = torch.tensor(5)
    for j in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_1d(
                (i, torch.tensor(j)),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )


# Device and Dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    """Test changing the `dtype` of the Basis class."""
    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)

    bas = Basis(number, GFN1_XTB, ihelp)
    assert bas is not None

    cls = bas.type(dtype)
    assert cls.dtype == dtype


def test_change_type_fail() -> None:
    """Test failure upon changing `dtype` incorrectly."""
    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)

    bas = Basis(number, GFN1_XTB, ihelp)
    assert bas is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        bas.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        bas.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    """Test changing the `device` of the class."""
    device = str_to_device(device_str)

    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)

    bas = Basis(number, GFN1_XTB, ihelp)
    assert bas is not None

    bas = bas.to(device)
    assert bas.device == device


def test_change_device_fail() -> None:
    """Test failure upon changing `device` incorrectly."""
    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)

    bas = Basis(number, GFN1_XTB, ihelp)
    assert bas is not None

    # trying to use setter
    with pytest.raises(AttributeError):
        bas.device = "cpu"


def test_non_valence_orthogonalization_is_metadata_independent() -> None:
    """Non-valence shell orthogonalization depends on valence flags, not meta."""
    numbers = torch.tensor([8, 1, 1])
    custom = GFN1_XTB.model_copy(deep=True)
    custom.meta = None

    bas_ref = Basis(
        numbers, GFN1_XTB, IndexHelper.from_numbers(numbers, GFN1_XTB)
    )
    bas_custom = Basis(
        numbers, custom, IndexHelper.from_numbers(numbers, custom)
    )

    alphas_ref, coeffs_ref = bas_ref.create_cgtos()
    alphas_cus, coeffs_cus = bas_custom.create_cgtos()
    for a_ref, a_cus in zip(alphas_ref, alphas_cus):
        torch.testing.assert_close(a_cus, a_ref, atol=0.0, rtol=0.0)
    for c_ref, c_cus in zip(coeffs_ref, coeffs_cus):
        torch.testing.assert_close(c_cus, c_ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("batched", [False, True])
def test_create_cgtos_handles_zero_padded_numbers(batched: bool) -> None:
    """Zero-padded atomic numbers produce 1D placeholder CGTO slots."""
    unpadded = torch.tensor([8, 1])
    bas_ref = Basis(
        unpadded, GFN1_XTB, IndexHelper.from_numbers(unpadded, GFN1_XTB)
    )
    alphas_ref, coeffs_ref = bas_ref.create_cgtos()

    padded = torch.tensor([[8, 1, 0]]) if batched else torch.tensor([8, 1, 0])
    ihelp = IndexHelper.from_numbers(padded, GFN1_XTB, batch_mode=int(batched))
    bas_pad = Basis(padded, GFN1_XTB, ihelp)
    alphas_pad, coeffs_pad = bas_pad.create_cgtos()

    assert len(alphas_pad) == len(alphas_ref) + 1
    assert alphas_pad[0].shape == (1,)
    assert coeffs_pad[0].shape == (1,)
    assert torch.count_nonzero(alphas_pad[0]) == 0
    assert torch.count_nonzero(coeffs_pad[0]) == 0

    for a_ref, a_act in zip(alphas_ref, alphas_pad[1:]):
        torch.testing.assert_close(a_act, a_ref, atol=0.0, rtol=0.0)
    for c_ref, c_act in zip(coeffs_ref, coeffs_pad[1:]):
        torch.testing.assert_close(c_act, c_ref, atol=0.0, rtol=0.0)
