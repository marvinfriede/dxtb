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
Dispersion Models
=================

Model catalogue, weighting kernels, and model implementations for DFT-D1 through
DFT-D5.
"""
from __future__ import annotations

from .base import DispersionModel, DispersionProperties
from .catalogue import (
    CHARGE_MODELS,
    DISPERSION_MODELS,
    MODEL_ALLOWED_DAMPINGS,
    ChargeModelId,
    DispersionModelId,
    ModelMetadata,
    get_model_metadata,
    resolve_charge_model_id,
    resolve_model_id,
    validate_model_damping,
)
from .weights import (
    WeightingFunction,
    gaussian_log_weight,
    gaussian_weight,
    limit_avg_deriv,
    quintic_hermite_basis,
    quintic_hermite_weight,
    smooth_weight,
    weight_cn,
    weight_references,
    zeta,
    zeta_tanh,
)

__all__ = [
    "CHARGE_MODELS",
    "DISPERSION_MODELS",
    "MODEL_ALLOWED_DAMPINGS",
    "ChargeModelId",
    "DispersionModel",
    "DispersionModelId",
    "DispersionProperties",
    "ModelMetadata",
    "WeightingFunction",
    "gaussian_log_weight",
    "gaussian_weight",
    "get_model_metadata",
    "limit_avg_deriv",
    "quintic_hermite_basis",
    "quintic_hermite_weight",
    "resolve_charge_model_id",
    "resolve_model_id",
    "smooth_weight",
    "validate_model_damping",
    "weight_cn",
    "weight_references",
    "zeta",
    "zeta_tanh",
]
