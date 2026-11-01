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

Model catalogue, weighting kernels, C6 factorization, and model implementations
for DFT-D1 through DFT-D4Srev.
"""
from __future__ import annotations

from .base import DispersionModel, DispersionProperties, ModelProperties
from .c6 import (
    AtomicC6,
    atomic_c6,
    atomic_c6_d3s,
    atomic_c6_d4s,
    contract_rc6,
    factored_c6,
    factored_c6_d3s,
    factored_c6_d4s,
)
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
from .d1 import D1Model
from .d2 import D2Model
from .d3 import D3Model, D3SModel, Environment, default_damping
from .d4 import (
    D4Model,
    D4SModel,
    D4SrevModel,
    d4sc_atom_energy,
    d4sc_atom_potential,
    d4sc_dispersion_matrix,
)
from .weights import (
    WeightingFunction,
    dzeta,
    dzeta_tanh,
    gaussian_log_weight,
    gaussian_weight,
    limit_avg_deriv,
    quintic_hermite_basis,
    quintic_hermite_weight,
    smooth_weight,
    weight_cn,
    weight_references,
    weight_references_d3s,
    weight_references_d4,
    weight_references_d4s,
    zeta,
    zeta_tanh,
)

__all__ = [
    "CHARGE_MODELS",
    "DISPERSION_MODELS",
    "MODEL_ALLOWED_DAMPINGS",
    "AtomicC6",
    "ChargeModelId",
    "D1Model",
    "D2Model",
    "D3Model",
    "D3SModel",
    "D4Model",
    "D4SModel",
    "D4SrevModel",
    "DispersionModel",
    "DispersionModelId",
    "DispersionProperties",
    "Environment",
    "ModelMetadata",
    "ModelProperties",
    "WeightingFunction",
    "atomic_c6",
    "atomic_c6_d3s",
    "atomic_c6_d4s",
    "contract_rc6",
    "d4sc_atom_energy",
    "d4sc_atom_potential",
    "d4sc_dispersion_matrix",
    "default_damping",
    "dzeta",
    "dzeta_tanh",
    "factored_c6",
    "factored_c6_d3s",
    "factored_c6_d4s",
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
    "weight_references_d3s",
    "weight_references_d4",
    "weight_references_d4s",
    "zeta",
    "zeta_tanh",
]
