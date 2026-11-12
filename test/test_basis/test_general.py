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
    from dxtb import ParamModule, labels
    from dxtb._src.integral.evaluation import setup_integral_evaluation
    from dxtb.integrals.wrappers import overlap

    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
    )
    custom = GFN1_XTB.model_copy(deep=True)
    custom.meta = None
    custom_pm = ParamModule(custom, dtype=torch.float64)

    bas_ref = Basis(
        numbers,
        GFN1_XTB,
        IndexHelper.from_numbers(numbers, GFN1_XTB),
        dtype=torch.float64,
    )
    bas_custom = Basis(
        numbers,
        custom,
        IndexHelper.from_numbers(numbers, custom),
        dtype=torch.float64,
    )
    bas_custom_pm = Basis(
        numbers, custom_pm, IndexHelper.from_numbers(numbers, custom_pm)
    )

    alphas_ref, coeffs_ref = bas_ref.create_cgtos()
    alphas_cus, coeffs_cus = bas_custom.create_cgtos()
    alphas_pm, coeffs_pm = bas_custom_pm.create_cgtos()
    for a_ref, a_cus, a_pm in zip(alphas_ref, alphas_cus, alphas_pm):
        torch.testing.assert_close(a_cus, a_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(a_pm, a_ref, atol=0.0, rtol=0.0)
    for c_ref, c_cus, c_pm in zip(coeffs_ref, coeffs_cus, coeffs_pm):
        torch.testing.assert_close(c_cus, c_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(c_pm, c_ref, atol=0.0, rtol=0.0)

    setup = setup_integral_evaluation(
        numbers,
        custom_pm,
        IndexHelper.from_numbers(numbers, custom_pm),
        driver_type=labels.INTDRIVER_PYTORCH,
        intlevel=labels.INTLEVEL_OVERLAP,
    )
    assert setup.backend is not None

    s_ref = overlap(
        numbers, positions, GFN1_XTB, driver=labels.INTDRIVER_PYTORCH
    )
    s_cus = overlap(numbers, positions, custom, driver=labels.INTDRIVER_PYTORCH)
    torch.testing.assert_close(s_cus, s_ref, atol=0.0, rtol=0.0)


def test_basis_and_wrappers_honor_target_dtype_with_param_module() -> None:
    """Basis and integral wrappers cast existing ParamModule CGTOs to dtype."""
    from dxtb import ParamModule, labels
    from dxtb._src.exlibs.available import has_libcint
    from dxtb.integrals.wrappers import dipint, overlap, quadint

    numbers = torch.tensor([8, 1, 1])
    positions_f32 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float32,
    )
    pm_f64 = ParamModule(GFN1_XTB, dtype=torch.float64)
    ihelp = IndexHelper.from_numbers(numbers, pm_f64)

    bas_default = Basis(numbers, pm_f64, ihelp)
    assert bas_default.dtype == torch.float64
    assert bas_default.slater.dtype == torch.float64

    bas_f32 = Basis(numbers, pm_f64, ihelp, dtype=torch.float32)
    alphas_f32, coeffs_f32 = bas_f32.create_cgtos()
    assert bas_f32.dtype == torch.float32
    assert all(a.dtype == torch.float32 for a in alphas_f32)
    assert all(c.dtype == torch.float32 for c in coeffs_f32)

    drivers = [labels.INTDRIVER_PYTORCH]
    if has_libcint:
        drivers.append(labels.INTDRIVER_LIBCINT)
    for drv in drivers:
        assert (
            overlap(numbers, positions_f32, pm_f64, driver=drv).dtype
            == torch.float32
        )
        assert (
            dipint(numbers, positions_f32, pm_f64, driver=drv).dtype
            == torch.float32
        )
        assert (
            quadint(numbers, positions_f32, pm_f64, driver=drv).dtype
            == torch.float32
        )


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
