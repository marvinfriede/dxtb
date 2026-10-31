# This file is part of tad-dftd.
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
Tests for atomic and pair data tables in ``tad_dftd.data``.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd import data


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_element_tables_shapes_and_values(dtype: torch.dtype) -> None:
    """Verify element data tables have index-0 padding and expected reference entries."""
    rcov = data.COV_D3(dtype=dtype)
    r4r2 = data.R4R2(dtype=dtype)
    vdw_d3 = data.VDW_D3(dtype=dtype)
    vdw_pair = data.VDW_PAIRWISE(dtype=dtype)
    d1_radii = data.D1_RADII(dtype=dtype)
    d2_radii = data.D2_RADII(dtype=dtype)
    en = data.PAULING_EN(dtype=dtype)
    gam = data.GAM(dtype=dtype)
    zeff = data.ZEFF(dtype=dtype)
    d1_c6 = data.D1_C6(dtype=dtype)
    d2_c6 = data.D2_C6(dtype=dtype)

    for tbl in (rcov, r4r2, en, gam, zeff):
        assert tbl.ndim == 1
        assert tbl.shape[0] == 119
        assert tbl[0].item() == 0.0
        assert tbl.dtype == dtype

    # VDW_D3 is (95,) and VDW_PAIRWISE is (104, 104)
    assert vdw_d3.shape == (95,)
    assert vdw_d3[0].item() == 0.0
    assert vdw_pair.shape == (104, 104)
    assert vdw_pair[0, 0].item() == 0.0
    assert torch.allclose(vdw_pair, vdw_pair.T)

    # D1 tables (padded to 119)
    assert d1_radii.shape == (119,)
    assert d1_c6.shape == (119,)
    assert d1_radii[0].item() == 0.0
    assert d1_c6[0].item() == 0.0
    assert d1_c6[6].item() > 0.0  # Carbon

    # D2 tables (padded to 119)
    assert d2_radii.shape == (119,)
    assert d2_c6.shape == (119,)
    assert d2_radii[0].item() == 0.0
    assert d2_c6[0].item() == 0.0
    assert d2_c6[6].item() > 0.0  # Carbon

    # Check specific physical values from Fortran dftd/data/
    assert pytest.approx(en[1].item(), rel=1e-6) == 2.20  # H Pauling EN
    assert pytest.approx(en[6].item(), rel=1e-6) == 2.55  # C Pauling EN
    assert pytest.approx(en[8].item(), rel=1e-6) == 3.44  # O Pauling EN
    assert pytest.approx(zeff[1].item(), rel=1e-6) == 1.0
    assert pytest.approx(zeff[6].item(), rel=1e-6) == 6.0


def test_tanh_parameters() -> None:
    """Verify D4Srev tanh charge-scaling parameter tables."""
    tbl = data.TANH_PARAMS(dtype=torch.float64)
    assert tbl.shape == (119, 4)
    assert torch.all(tbl[0] == 0.0)

    numbers = torch.tensor([0, 1, 6, 8], dtype=torch.long)
    a, b, c, d = data.get_tanh_params(numbers, dtype=torch.float64)
    for t in (a, b, c, d):
        assert t.shape == (4,)
        assert t[0].item() == 0.0
    # Hydrogen (Z=1) has non-zero parameters
    assert a[1].item() > 0.0
    assert c[1].item() > 0.0


def test_wfpair_tables() -> None:
    """Verify pairwise Gaussian weighting factor tables for D3S and D4S."""
    wf_d3s = data.WFPAIR_D3S(dtype=torch.float64)
    wf_d4s = data.WFPAIR_D4S(dtype=torch.float64)

    assert wf_d3s.shape == (95, 95)
    assert wf_d4s.shape == (119, 119)

    # Padding row/column is 0
    assert torch.all(wf_d3s[0, :] == 0.0)
    assert torch.all(wf_d3s[:, 0] == 0.0)
    assert torch.all(wf_d4s[0, :] == 0.0)
    assert torch.all(wf_d4s[:, 0] == 0.0)

    # Symmetric
    assert torch.allclose(wf_d3s, wf_d3s.T)
    assert torch.allclose(wf_d4s, wf_d4s.T)

    # Positive on real element pairs
    assert torch.all(wf_d3s[1:, 1:] > 0.0)
    assert torch.all(wf_d4s[1:, 1:] > 0.0)


def test_element_table_validation() -> None:
    """Verify element_table resolves callables and validates tensor shapes."""
    like = torch.zeros(3, dtype=torch.float64)
    resolved = data.element_table(data.R4R2, "r4r2_table", like)
    assert resolved.dtype == torch.float64
    assert resolved.shape == (119,)

    custom = torch.ones(119, dtype=torch.float32)
    resolved_custom = data.element_table(custom, "r4r2_table", like)
    assert resolved_custom.dtype == torch.float64
    assert resolved_custom.shape == (119,)

    with pytest.raises(ValueError, match="table indexed by atomic number"):
        data.element_table(torch.ones(10), "r4r2_table", like)
