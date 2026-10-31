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
Tests for ``tad_dftd.model.catalogue`` and ``tad_dftd.param``.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd import param
from tad_dftd.damping import DampingParam
from tad_dftd.model.catalogue import (
    DISPERSION_MODELS,
    ChargeModelId,
    DispersionModelId,
    get_model_metadata,
    resolve_charge_model_id,
    resolve_model_id,
    validate_model_damping,
)


def test_model_catalogue_and_damping_validation() -> None:
    """Verify all 8 dispersion models and their valid damping combinations."""
    expected_models = {"d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev", "d5"}
    assert set(DISPERSION_MODELS.keys()) == expected_models

    assert resolve_model_id("dft-d3") == DispersionModelId.D3
    assert resolve_model_id("D4Srev") == DispersionModelId.D4SREV
    assert resolve_charge_model_id("eeqbc", model="d4srev") == ChargeModelId.EEQBC

    with pytest.raises(ValueError, match="Unknown dispersion model"):
        resolve_model_id("d6")

    # D1 allows only Fermi damping
    validate_model_damping("d1", "fermi")
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d1", "rational")

    # D2 allows only Fermi damping
    validate_model_damping("d2", "fermi")
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d2", "rational")

    # D3 allows rational, zero, mzero, optimizedpower, cso, z, koide, screened
    for d in ("rational", "zero", "mzero", "optimizedpower", "cso", "z", "koide", "screened"):
        validate_model_damping("d3", d)
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d3", "fermi")

    # D4 allows rational, zero, mzero, optimizedpower, screened
    for d in ("rational", "zero", "mzero", "optimizedpower", "screened"):
        validate_model_damping("d4", d)
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d4", "cso")

    # D5 allows screened, rational, zero, koide
    validate_model_damping("d5", "screened")
    validate_model_damping("d5", "bj")
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d5", "cso")

    meta_d4srev = get_model_metadata("d4srev")
    assert meta_d4srev.charge_dep_2b is True
    assert meta_d4srev.default_qmodel == ChargeModelId.EEQ


def test_functional_parameter_lookup() -> None:
    """Verify functional alias normalization and parameter lookup across models."""
    # D3 rational (BJ) parameters for PBE
    p_d3_pbe = param.get_params("PBE", model="d3", damping="rational")
    assert pytest.approx(p_d3_pbe["s6"].item(), rel=1e-6) == 1.0
    assert pytest.approx(p_d3_pbe["s8"].item(), rel=1e-6) == 0.7875
    assert pytest.approx(p_d3_pbe["a1"].item(), rel=1e-6) == 0.4289
    assert pytest.approx(p_d3_pbe["a2"].item(), rel=1e-6) == 4.4407

    # D4 rational parameters for PBE
    p_d4_pbe = param.get_params("PBE", model="d4", damping="rational")
    assert pytest.approx(p_d4_pbe["s6"].item(), rel=1e-6) == 1.0
    assert p_d4_pbe["s8"].item() > 0.0
    assert p_d4_pbe["a1"].item() > 0.0
    assert p_d4_pbe["a2"].item() > 0.0

    # D1 and D2 parameters for PBE
    p_d1_pbe = param.get_params("PBE", model="d1")
    assert pytest.approx(p_d1_pbe["s6"].item(), rel=1e-6) == 0.7
    p_d2_pbe = param.get_params("PBE", model="d2")
    assert pytest.approx(p_d2_pbe["s6"].item(), rel=1e-6) == 0.75

    # DampingParam.from_functional constructs DampingParam
    dp = DampingParam.from_functional("PBE", model="d3", damping="rational", dtype=torch.float64)
    assert isinstance(dp, DampingParam)
    assert isinstance(dp.a1, torch.Tensor)
    assert pytest.approx(dp.a1.item(), rel=1e-6) == 0.4289
