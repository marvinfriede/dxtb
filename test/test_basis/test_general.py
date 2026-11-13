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

from ..conftest import DEVICE
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
    """Test changing the `dtype` of the Basis class without corrupting int/bool slots."""
    number = torch.tensor([8, 1, 1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)

    bas = Basis(number, GFN1_XTB, ihelp, dtype=torch.float64)
    assert bas is not None

    for converted in (bas.type(dtype), bas.to(dtype=dtype)):
        assert converted.dtype == dtype
        assert converted.slater.dtype == dtype
        assert converted.numbers.dtype == bas.numbers.dtype
        assert converted.unique.dtype == bas.unique.dtype
        assert converted.ngauss.dtype == bas.ngauss.dtype
        assert converted.pqn.dtype == bas.pqn.dtype
        assert converted.valence.dtype == torch.bool
        if dtype in (torch.float32, torch.float64):
            alphas, coeffs = converted.create_cgtos()
            assert all(a.dtype == dtype for a in alphas)
            assert all(c.dtype == dtype for c in coeffs)

    meta_bas = bas.to(device=torch.device("meta"), dtype=dtype)
    assert meta_bas.device == torch.device("meta")
    assert meta_bas.dtype == dtype
    assert meta_bas.ihelp.device == torch.device("meta")
    assert meta_bas.slater.dtype == dtype
    assert meta_bas.ngauss.dtype == bas.ngauss.dtype
    assert meta_bas.pqn.dtype == bas.pqn.dtype
    assert meta_bas.valence.dtype == torch.bool
    assert meta_bas.numbers.dtype == bas.numbers.dtype
    assert meta_bas.unique.dtype == bas.unique.dtype


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

    bas_orig = Basis(number, GFN1_XTB, ihelp)
    assert bas_orig is not None

    bas = bas_orig.to(device)
    assert bas.device == device
    assert bas.ihelp.device == device
    assert bas.ngauss.dtype == bas_orig.ngauss.dtype
    assert bas.pqn.dtype == bas_orig.pqn.dtype
    assert bas.valence.dtype == torch.bool
    alphas, coeffs = bas.create_cgtos()
    assert all(a.device == device for a in alphas)
    assert all(c.device == device for c in coeffs)


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
    from dxtb import GFN0_XTB, ParamModule, labels
    from dxtb._src.exlibs.available import has_libcint
    from dxtb.integrals.wrappers import dipint, hcore, overlap, quadint

    numbers = torch.tensor([8, 1, 1])
    positions_f32 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float32,
    )
    pm_f64 = ParamModule(GFN1_XTB, dtype=torch.float64)
    pm_gfn0_f64 = ParamModule(GFN0_XTB, dtype=torch.float64)
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
        assert (
            hcore(numbers, positions_f32, pm_f64, charge=0.0, driver=drv).dtype
            == torch.float32
        )
        assert (
            hcore(
                numbers, positions_f32, pm_gfn0_f64, charge=0.0, driver=drv
            ).dtype
            == torch.float32
        )
    assert pm_f64.dtype == torch.float64
    assert pm_gfn0_f64.dtype == torch.float64


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


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_canonical_cgtos_all_supported_elements(method: str) -> None:
    """WP2: canonical CGTO setup across all 86 elements in GFN0, GFN1, GFN2."""
    from tad_mctc.data import pse

    from dxtb import GFN0_XTB, GFN2_XTB, ParamModule
    from dxtb._src.basis.ortho import gaussian_integral

    par_raw = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}[method]
    assert par_raw.element is not None
    zs = sorted(
        pse.S2Z[sym] for sym in par_raw.element.keys() if sym in pse.S2Z
    )
    assert zs == list(range(1, 87))

    expected_dev = torch.device("cpu") if DEVICE is None else DEVICE
    par = ParamModule(par_raw, dtype=torch.float64, device=DEVICE)
    tm_dsp_seen = 0

    for z in zs:
        num = torch.tensor([z], device=DEVICE)
        ihelp = IndexHelper.from_numbers(num, par)
        bas = Basis(num, par, ihelp, device=DEVICE, dtype=torch.float64)
        alphas, coeffs = bas.create_cgtos()

        n_shells = int(ihelp.ushells_per_unique[0].item())
        assert len(alphas) == n_shells == len(coeffs)
        angs = [int(x) for x in ihelp.unique_angular.tolist()]
        vals = [int(x) for x in bas.valence.tolist()]
        if angs[:1] == [2] and 0 in angs[1:]:
            tm_dsp_seen += 1

        for l_val in set(angs):
            nv = sum(1 for a_l, v in zip(angs, vals) if a_l == l_val and v == 0)
            assert nv <= 1

        for s_idx, (alpha, coeff) in enumerate(zip(alphas, coeffs)):
            assert alpha.ndim == 1 and coeff.ndim == 1
            assert alpha.shape == coeff.shape
            assert alpha.dtype == torch.float64 and coeff.dtype == torch.float64
            assert alpha.device == expected_dev and coeff.device == expected_dev
            assert torch.isfinite(alpha).all() and torch.isfinite(coeff).all()
            if angs[s_idx] == 0:
                self_ovlp = gaussian_integral(alpha, alpha, coeff, coeff)
                torch.testing.assert_close(
                    self_ovlp,
                    torch.ones_like(self_ovlp),
                    atol=1e-13,
                    rtol=1e-13,
                )

        if z == 1 and method in ("gfn0", "gfn1"):
            assert vals == [1, 0] and angs == [0, 0]
            cross = gaussian_integral(
                alphas[0], alphas[1], coeffs[0], coeffs[1]
            )
            torch.testing.assert_close(
                cross, torch.zeros_like(cross), atol=1e-14, rtol=0.0
            )

    assert tm_dsp_seen == 41


def test_custom_multi_nonvalence_s_shells_orthogonalize_to_primary() -> None:
    """WP2: multiple non-valence s shells each orthogonalize to valence 1s."""
    import copy

    from dxtb import ParamModule
    from dxtb._src.basis.ortho import gaussian_integral

    custom = copy.deepcopy(GFN1_XTB)
    assert custom.element is not None
    h_el = custom.element["H"]
    h_el.shells = ["1s", "2s", "3s"]
    h_el.ngauss = [4, 4, 4]
    h_el.slater = [1.2, 0.9, 0.6]
    h_el.levels = [-10.0, -5.0, -2.0]
    h_el.shpoly = [0.0, 0.0, 0.0]
    h_el.kcn = [0.0, 0.0, 0.0]
    h_el.lgam = [1.0, 1.0, 1.0]

    par = ParamModule(custom, dtype=torch.float64, device=DEVICE)
    num = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers(num, par)
    bas = Basis(num, par, ihelp, device=DEVICE, dtype=torch.float64)
    alphas, coeffs = bas.create_cgtos()

    assert [int(x) for x in bas.valence.tolist()] == [1, 0, 0]
    for idx in range(3):
        diag = gaussian_integral(
            alphas[idx], alphas[idx], coeffs[idx], coeffs[idx]
        )
        torch.testing.assert_close(
            diag, torch.ones_like(diag), atol=1e-14, rtol=1e-14
        )
    s12 = gaussian_integral(alphas[0], alphas[1], coeffs[0], coeffs[1])
    s13 = gaussian_integral(alphas[0], alphas[2], coeffs[0], coeffs[2])
    s23 = gaussian_integral(alphas[1], alphas[2], coeffs[1], coeffs[2])
    torch.testing.assert_close(s12, torch.zeros_like(s12), atol=1e-14, rtol=0.0)
    torch.testing.assert_close(s13, torch.zeros_like(s13), atol=1e-14, rtol=0.0)
    assert s23.abs() > 0.1
