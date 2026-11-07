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
Tests for GFN1-xTB 4sp and 5sp STO-6G basis expansion restoration (tblite #423).

Verifies:
- Exact primitive exponents and contraction coefficients matching tblite #423.
- Exponent sharing between 4s/4p and 5s/5p shells under `sto="gfn1"`.
- Contracted Gaussian function normalization.
- Invariance of all unaffected shells and standard Stewart coefficients under `sto="gfn2"`.
- Explicit STO expansion selection via parameter model.
- Basis construction paths through `Basis` (PyTorch, libcint, export, Slater derivatives).
"""

from __future__ import annotations

import copy
import pytest
import torch

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis, BasisSetup
from dxtb._src.basis.slater import (
    pAlpha6sp,
    pCoeff6sp_p,
    pCoeff6sp_s,
    slater_to_gauss,
)
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.param import GFN1_XTB, GFN2_XTB, Param, ParamModule
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..utils import overlap_1d

# Reference values directly from tblite src/tblite/basis/slater.f90 (commit 7c935fb)
TBLITE_ALPHA_4SP = [
    1.365346e00,
    4.393213e-01,
    1.877069e-01,
    9.360270e-02,
    5.052263e-02,
    2.809354e-02,
]
TBLITE_COEFF_4S = [
    3.775056e-03,
    -5.585965e-02,
    -3.192946e-01,
    -2.764780e-02,
    9.049199e-01,
    3.406258e-01,
]
TBLITE_COEFF_4P = [
    -7.052075e-03,
    -5.259505e-02,
    -3.773450e-02,
    3.874773e-01,
    5.791672e-01,
    1.221817e-01,
]

TBLITE_ALPHA_5SP = [
    7.701420258e-01,
    2.756268915e-01,
    1.301847480e-01,
    6.953441940e-02,
    4.002545502e-02,
    2.348388309e-02,
]
TBLITE_COEFF_5S = [
    1.267447151e-02,
    3.266734789e-03,
    -4.307553999e-01,
    -3.231998963e-01,
    1.104322879e00,
    4.368498703e-01,
]
TBLITE_COEFF_5P = [
    -1.105673292e-3,
    -6.243132446e-2,
    -1.628476766e-1,
    3.210328714e-1,
    6.964579592e-1,
    1.493146125e-1,
]


def test_tblite_raw_coefficient_tables() -> None:
    """Verify internal coefficient tables match tblite #423 to machine precision."""
    torch.testing.assert_close(
        pAlpha6sp[4],
        torch.tensor(TBLITE_ALPHA_4SP, dtype=torch.double),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        pCoeff6sp_s[4],
        torch.tensor(TBLITE_COEFF_4S, dtype=torch.double),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        pCoeff6sp_p[4],
        torch.tensor(TBLITE_COEFF_4P, dtype=torch.double),
        atol=0.0,
        rtol=0.0,
    )

    torch.testing.assert_close(
        pAlpha6sp[5],
        torch.tensor(TBLITE_ALPHA_5SP, dtype=torch.double),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        pCoeff6sp_s[5],
        torch.tensor(TBLITE_COEFF_5S, dtype=torch.double),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        pCoeff6sp_p[5],
        torch.tensor(TBLITE_COEFF_5P, dtype=torch.double),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("zeta_val", [0.8, 1.25, 2.1])
def test_gfn1_shared_exponents_and_differing_coeffs(
    dtype: torch.dtype, zeta_val: float
) -> None:
    """Verify 4s/4p and 5s/5p share exponents but have distinct contraction coefficients."""
    zeta = torch.tensor(zeta_val, dtype=dtype, device=DEVICE)

    for n in (4, 5):
        alpha_s, coeff_s = slater_to_gauss(6, n, 0, zeta, norm=True, sto="gfn1")
        alpha_p, coeff_p = slater_to_gauss(6, n, 1, zeta, norm=True, sto="gfn1")

        # Exponents must be exactly identical
        torch.testing.assert_close(alpha_s, alpha_p, atol=0.0, rtol=0.0)

        # Contraction coefficients must be distinct
        assert not torch.allclose(coeff_s, coeff_p)

        # GFN2 (Stewart) exponents for s and p must differ from GFN1
        alpha_s_gfn2, coeff_s_gfn2 = slater_to_gauss(
            6, n, 0, zeta, norm=True, sto="gfn2"
        )
        assert not torch.allclose(alpha_s, alpha_s_gfn2)
        assert not torch.allclose(coeff_s, coeff_s_gfn2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [4, 5])
@pytest.mark.parametrize("l", [0, 1])
def test_gfn1_expansion_normalization(
    dtype: torch.dtype, n: int, l: int
) -> None:
    """Verify contracted GFN1 Gaussians are normalized to 1."""
    dd: DD = {"dtype": dtype, "device": DEVICE}
    zeta = torch.tensor(1.23, **dd)
    alpha, coeff = slater_to_gauss(6, n, l, zeta, norm=True, sto="gfn1")

    ang = torch.tensor(l, device=DEVICE)
    vec = torch.zeros(3, **dd)
    s = overlap_1d((ang, ang), (alpha, alpha), (coeff, coeff), vec)
    ref = torch.eye(2 * l + 1, **dd)

    atol = 1.0e-5 if dtype == torch.float32 else 1.0e-14
    assert pytest.approx(ref.cpu(), abs=atol) == s.cpu()


def test_unaffected_shells_retain_stewart() -> None:
    """Verify all shells other than 6g 4sp and 5sp retain Stewart coefficients."""
    zeta = torch.tensor(1.2, dtype=torch.float64, device=DEVICE)

    for ng in range(1, 7):
        for n in range(1, 6):
            for l in range(0, n):
                if ng == 6 and n in (4, 5) and l in (0, 1):
                    continue
                a_gfn1, c_gfn1 = slater_to_gauss(
                    ng, n, l, zeta, norm=True, sto="gfn1"
                )
                a_gfn2, c_gfn2 = slater_to_gauss(
                    ng, n, l, zeta, norm=True, sto="gfn2"
                )
                torch.testing.assert_close(a_gfn1, a_gfn2, atol=0.0, rtol=0.0)
                torch.testing.assert_close(c_gfn1, c_gfn2, atol=0.0, rtol=0.0)


def test_invalid_sto_expansion_raises() -> None:
    """Verify unknown STO selector raises ValueError."""
    zeta = torch.tensor(1.0, dtype=torch.float64, device=DEVICE)
    with pytest.raises(ValueError, match="Unknown STO-NG expansion"):
        slater_to_gauss(6, 4, 0, zeta, sto="invalid")  # type: ignore[arg-type]


def test_custom_param_sto_selection() -> None:
    """Verify custom parametrization can explicitly select either expansion."""
    par_dict = GFN1_XTB.clean_model_dump()

    # Explicit gfn2 in GFN1 param
    par_dict["hamiltonian"]["xtb"]["sto"] = "gfn2"
    par_gfn2 = Param.model_validate(par_dict)
    assert par_gfn2.hamiltonian.xtb.sto == "gfn2"  # type: ignore[union-attr]

    # Explicit gfn1
    par_dict["hamiltonian"]["xtb"]["sto"] = "gfn1"
    par_gfn1 = Param.model_validate(par_dict)
    assert par_gfn1.hamiltonian.xtb.sto == "gfn1"  # type: ignore[union-attr]

    # Basis reflects setting
    numbers = torch.tensor([26], device=DEVICE)  # Fe (contains 4s, 4p)
    ih = IndexHelper.from_numbers(numbers, par_gfn1)
    b_gfn1 = Basis(numbers, par_gfn1, ih)
    b_gfn2 = Basis(numbers, par_gfn2, ih)
    assert b_gfn1.sto == "gfn1"
    assert b_gfn2.sto == "gfn2"

    alphas_gfn1, coeffs_gfn1 = b_gfn1.create_cgtos()
    alphas_gfn2, coeffs_gfn2 = b_gfn2.create_cgtos()

    # 3d shell (index 0) is unaffected
    torch.testing.assert_close(
        alphas_gfn1[0], alphas_gfn2[0], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        coeffs_gfn1[0], coeffs_gfn2[0], atol=0.0, rtol=0.0
    )

    # 4s (index 1) and 4p (index 2) differ
    assert not torch.allclose(alphas_gfn1[1], alphas_gfn2[1])
    assert not torch.allclose(coeffs_gfn1[1], coeffs_gfn2[1])
    assert not torch.allclose(alphas_gfn1[2], alphas_gfn2[2])
    assert not torch.allclose(coeffs_gfn1[2], coeffs_gfn2[2])

    # In GFN1, 4s and 4p share primitive unscaled exponents
    torch.testing.assert_close(
        alphas_gfn1[1] / (b_gfn1.slater[1] ** 2),
        alphas_gfn1[2] / (b_gfn1.slater[2] ** 2),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        alphas_gfn1[1] / (b_gfn1.slater[1] ** 2),
        pAlpha6sp[4].to(alphas_gfn1[1].dtype),
        atol=1e-12,
        rtol=1e-12,
    )


def test_basis_construction_and_export() -> None:
    """Verify Basis methods (create_cgtos, create_libcint_cgtos, to_bse) use sto."""
    # Fe (Z=26, 4s/4p) and Rb (Z=37, 5s/5p)
    numbers = torch.tensor([26, 37], device=DEVICE)
    ih = IndexHelper.from_numbers(numbers, GFN1_XTB)
    b_gfn1 = Basis(numbers, GFN1_XTB, ih)
    b_gfn2 = Basis(numbers, GFN2_XTB, IndexHelper.from_numbers(numbers, GFN2_XTB))

    assert b_gfn1.sto == "gfn1"
    assert b_gfn2.sto == "gfn2"

    # PyTorch CGTOs
    alphas_pt, coeffs_pt = b_gfn1.create_cgtos()

    # Libcint CGTOs via BasisSetup
    setup = b_gfn1.setup_data()
    assert setup.sto == "gfn1"
    alphas_lc, coeffs_lc = setup.create_libcint_cgtos()

    for a_pt, a_lc in zip(alphas_pt, alphas_lc):
        torch.testing.assert_close(a_pt, a_lc, atol=0.0, rtol=0.0)
    for c_pt, c_lc in zip(coeffs_pt, coeffs_lc):
        torch.testing.assert_close(c_pt, c_lc, atol=0.0, rtol=0.0)

    # BSE Export
    nwchem_txt = b_gfn1.to_bse(qcformat="nwchem")
    assert "Fe    s" in nwchem_txt
    assert "Fe    p" in nwchem_txt
    assert "Rb    s" in nwchem_txt
    assert "Rb    p" in nwchem_txt

    g94_txt = b_gfn1.to_bse(qcformat="gaussian94")
    assert "s    6    1.00" in g94_txt
    assert "p    6    1.00" in g94_txt


def test_slater_exponent_derivatives() -> None:
    """Verify autograd derivatives with respect to Slater exponent remain valid."""
    z = torch.tensor(1.45, dtype=torch.float64, requires_grad=True)
    a, c = slater_to_gauss(6, 4, 0, z, norm=True, sto="gfn1")
    loss = (a * 0.5 + c * 2.0).sum()
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad)
    assert z.grad.item() != 0.0
