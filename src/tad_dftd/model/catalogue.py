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
Dispersion Model Catalogue
==========================

Catalogue of supported dispersion models, charge models, model metadata, and
model-damping validation rules matching Grimme-lab ``dftd/model/catalogue.f90``
and ``dftd/param.f90``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .. import defaults
from ..cutoff import Cutoff

__all__ = [
    "CHARGE_MODELS",
    "DISPERSION_MODELS",
    "MODEL_ALLOWED_DAMPINGS",
    "ChargeModelId",
    "DispersionModelId",
    "ModelMetadata",
    "get_model_metadata",
    "resolve_charge_model_id",
    "resolve_model_id",
    "validate_model_damping",
]


class DispersionModelId(IntEnum):
    """
    Integer identifiers for DFT-D models matching ``enum_dftd_models`` in
    ``dftd/model/type.f90``.
    """

    D1 = 10
    D2 = 20
    D3 = 30
    D3S = 31
    D4 = 40
    D4S = 41
    D4SREV = 42
    D5 = 50


class ChargeModelId(IntEnum):
    """
    Integer identifiers for partial charge models matching ``enum_dftd_qmodels``
    in ``dftd/model/type.f90``.
    """

    EEQ = 1
    GFN2 = 2
    EEQBC = 3
    GXTB = 4


@dataclass(frozen=True)
class ModelMetadata:
    """
    Static metadata and default configuration of a DFT-D dispersion model
    (matching ``set_model_defaults`` in ``dftd/model/catalogue.f90``).
    """

    model_id: DispersionModelId
    name: str
    label: str
    max_elem: int
    charge_dep_2b: bool
    charge_dep_3b: bool
    has_polarizabilities: bool
    default_qmodel: ChargeModelId | None
    default_damping_2b: str
    default_damping_3b: str | None
    default_cutoff: Cutoff
    allowed_dampings: tuple[str, ...]
    allowed_qmodels: tuple[ChargeModelId, ...]


DISPERSION_MODELS: dict[str, ModelMetadata] = {
    "d1": ModelMetadata(
        model_id=DispersionModelId.D1,
        name="d1",
        label="D1",
        max_elem=defaults.MAX_ELEM_D1,
        charge_dep_2b=False,
        charge_dep_3b=False,
        has_polarizabilities=False,
        default_qmodel=None,
        default_damping_2b="fermi",
        default_damping_3b=None,
        default_cutoff=Cutoff.from_model("d1"),
        allowed_dampings=("fermi", "d1"),
        allowed_qmodels=(),
    ),
    "d2": ModelMetadata(
        model_id=DispersionModelId.D2,
        name="d2",
        label="D2",
        max_elem=defaults.MAX_ELEM_D2,
        charge_dep_2b=False,
        charge_dep_3b=False,
        has_polarizabilities=False,
        default_qmodel=None,
        default_damping_2b="fermi",
        default_damping_3b=None,
        default_cutoff=Cutoff.from_model("d2"),
        allowed_dampings=("fermi", "d2"),
        allowed_qmodels=(),
    ),
    "d3": ModelMetadata(
        model_id=DispersionModelId.D3,
        name="d3",
        label="D3",
        max_elem=defaults.MAX_ELEM_D3,
        charge_dep_2b=False,
        charge_dep_3b=False,
        has_polarizabilities=False,
        default_qmodel=None,
        default_damping_2b="rational",
        default_damping_3b="zero_d3",
        default_cutoff=Cutoff.from_model("d3"),
        allowed_dampings=(
            "rational",
            "bj",
            "bjm",
            "zero",
            "chg",
            "mzero",
            "zerom",
            "optimizedpower",
            "optimisedpower",
            "op",
            "cso",
            "screened",
            "koide",
            "z",
        ),
        allowed_qmodels=(),
    ),
    "d3s": ModelMetadata(
        model_id=DispersionModelId.D3S,
        name="d3s",
        label="D3S",
        max_elem=defaults.MAX_ELEM_D3S,
        charge_dep_2b=False,
        charge_dep_3b=False,
        has_polarizabilities=False,
        default_qmodel=None,
        default_damping_2b="rational",
        default_damping_3b="zero_d3",
        default_cutoff=Cutoff.from_model("d3s"),
        allowed_dampings=(
            "rational",
            "bj",
            "bjm",
            "zero",
            "chg",
            "mzero",
            "zerom",
            "optimizedpower",
            "optimisedpower",
            "op",
            "cso",
            "screened",
            "koide",
        ),
        allowed_qmodels=(),
    ),
    "d4": ModelMetadata(
        model_id=DispersionModelId.D4,
        name="d4",
        label="D4",
        max_elem=defaults.MAX_ELEM_D4,
        charge_dep_2b=True,
        charge_dep_3b=False,
        has_polarizabilities=True,
        default_qmodel=ChargeModelId.EEQ,
        default_damping_2b="rational",
        default_damping_3b="zero_d4",
        default_cutoff=Cutoff.from_model("d4"),
        allowed_dampings=(
            "rational",
            "bj",
            "bjm",
            "zero",
            "chg",
            "mzero",
            "zerom",
            "optimizedpower",
            "optimisedpower",
            "op",
            "screened",
            "koide",
        ),
        allowed_qmodels=(
            ChargeModelId.EEQ,
            ChargeModelId.GFN2,
        ),
    ),
    "d4s": ModelMetadata(
        model_id=DispersionModelId.D4S,
        name="d4s",
        label="D4S",
        max_elem=defaults.MAX_ELEM_D4,
        charge_dep_2b=True,
        charge_dep_3b=False,
        has_polarizabilities=True,
        default_qmodel=ChargeModelId.EEQ,
        default_damping_2b="rational",
        default_damping_3b="zero_d4",
        default_cutoff=Cutoff.from_model("d4s"),
        allowed_dampings=(
            "rational",
            "bj",
            "bjm",
            "zero",
            "chg",
            "mzero",
            "zerom",
            "optimizedpower",
            "optimisedpower",
            "op",
            "screened",
            "koide",
        ),
        allowed_qmodels=(
            ChargeModelId.EEQ,
            ChargeModelId.GFN2,
        ),
    ),
    "d4srev": ModelMetadata(
        model_id=DispersionModelId.D4SREV,
        name="d4srev",
        label="D4Srev",
        max_elem=defaults.MAX_ELEM_D4SREV,
        charge_dep_2b=True,
        charge_dep_3b=False,
        has_polarizabilities=True,
        default_qmodel=ChargeModelId.EEQ,
        default_damping_2b="rational",
        default_damping_3b="zero_d4",
        default_cutoff=Cutoff.from_model("d4srev"),
        allowed_dampings=(
            "rational",
            "bj",
            "bjm",
            "zero",
            "chg",
            "mzero",
            "zerom",
            "optimizedpower",
            "optimisedpower",
            "op",
            "screened",
            "koide",
        ),
        allowed_qmodels=(
            ChargeModelId.EEQ,
            ChargeModelId.GFN2,
            ChargeModelId.GXTB,
        ),
    ),
}

CHARGE_MODELS: dict[str, ChargeModelId] = {
    "eeq": ChargeModelId.EEQ,
    "eeq2019": ChargeModelId.EEQ,
    "gfn2": ChargeModelId.GFN2,
    "gxtb": ChargeModelId.GXTB,
}

MODEL_ALLOWED_DAMPINGS: dict[str, tuple[str, ...]] = {
    k: v.allowed_dampings for k, v in DISPERSION_MODELS.items()
}

_MODEL_METADATA_BY_ID: dict[DispersionModelId, ModelMetadata] = {
    v.model_id: v for v in DISPERSION_MODELS.values()
}


class _InvalidModelTypeError(TypeError, ValueError):
    """Exception raised when a boolean or invalid type is passed as a model ID."""


def resolve_model_id(
    model: str | int | DispersionModelId,
) -> DispersionModelId:
    """Resolve a model name, integer ID, or enum to a :class:`DispersionModelId`."""
    if isinstance(model, bool):
        raise _InvalidModelTypeError(
            f"Boolean value '{model}' is not a valid dispersion model identifier."
        )
    if isinstance(model, int):
        try:
            model_id = (
                model
                if isinstance(model, DispersionModelId)
                else DispersionModelId(model)
            )
        except ValueError as exc:
            valid = ", ".join(
                f"{m.value} ({m.name})"
                for m in DispersionModelId
                if m != DispersionModelId.D5
            )
            raise ValueError(
                f"Unknown dispersion model ID '{model}'; expected one of: {valid}."
            ) from exc
        if model_id == DispersionModelId.D5:
            raise NotImplementedError(
                "Dispersion model 'D5' is not yet implemented because the "
                "upstream reference is not finalized."
            )
        return model_id
    if not isinstance(model, str):
        raise _InvalidModelTypeError(
            f"Expected str, int, or DispersionModelId for 'model', got "
            f"'{type(model).__name__}'."
        )
    key = model.lower().strip()
    if key.startswith("dft-"):
        key = key[4:]
    elif key.startswith("dft"):
        key = key[3:]
    if key == "d5":
        raise NotImplementedError(
            "Dispersion model 'D5' is not yet implemented because the "
            "upstream reference is not finalized."
        )
    meta = DISPERSION_MODELS.get(key)
    if meta is None:
        known = ", ".join(sorted(DISPERSION_MODELS))
        raise ValueError(
            f"Unknown dispersion model '{model}'; expected one of: {known}."
        )
    return meta.model_id


def get_model_metadata(model: str | int | DispersionModelId) -> ModelMetadata:
    """Return the :class:`ModelMetadata` for ``model``."""
    return _MODEL_METADATA_BY_ID[resolve_model_id(model)]


def resolve_charge_model_id(
    qmodel: str | int | ChargeModelId | None,
    *,
    model: str | int | DispersionModelId = "d4",
) -> ChargeModelId | None:
    """
    Resolve a charge model name, integer ID, or enum for ``model`` and validate
    that the dispersion model supports it.
    """
    meta = get_model_metadata(model)
    if qmodel is None:
        return meta.default_qmodel

    if isinstance(qmodel, bool):
        raise _InvalidModelTypeError(
            f"Boolean value '{qmodel}' is not a valid charge model identifier."
        )
    qid: ChargeModelId
    if isinstance(qmodel, int):
        try:
            qid = (
                qmodel
                if isinstance(qmodel, ChargeModelId)
                else ChargeModelId(qmodel)
            )
        except ValueError as exc:
            raise ValueError(f"Unknown charge model ID '{qmodel}'.") from exc
        if qid == ChargeModelId.EEQBC:
            raise NotImplementedError(
                "Charge model 'eeqbc' is not supported in this release "
                "because the upstream model is not finalized."
            )
    elif isinstance(qmodel, str):
        key = qmodel.lower().strip()
        if key in ("eeqbc", "eeqbc2025"):
            raise NotImplementedError(
                f"Charge model '{qmodel}' is not supported in this release "
                "because the upstream model is not finalized."
            )
        if key not in CHARGE_MODELS:
            known = ", ".join(sorted(CHARGE_MODELS))
            raise ValueError(
                f"Unknown charge model '{qmodel}'; expected one of: {known}."
            )
        qid = CHARGE_MODELS[key]
    else:
        raise _InvalidModelTypeError(
            "Expected str, int, ChargeModelId, or None for 'qmodel', got "
            f"'{type(qmodel).__name__}'."
        )

    if not meta.allowed_qmodels:
        raise ValueError(
            f"Dispersion model '{meta.label}' does not use a charge model."
        )
    if qid not in meta.allowed_qmodels:
        allowed = ", ".join(q.name.lower() for q in meta.allowed_qmodels)
        raise ValueError(
            f"Charge model '{qid.name.lower()}' is not supported by "
            f"dispersion model '{meta.label}' (allowed: {allowed})."
        )
    return qid


def validate_model_damping(
    model: str | int | DispersionModelId,
    damping: str | object,
) -> None:
    """
    Validate that ``damping`` is supported for ``model``.
    """
    meta = get_model_metadata(model)
    if not isinstance(damping, str):
        from ..damping.base import ThreeBodyDamping, TwoBodyDamping
        from ..damping.select import identify_damping

        matched = (
            identify_damping(damping)
            if isinstance(damping, (TwoBodyDamping, ThreeBodyDamping))
            else None
        )
        if matched is None:
            raise ValueError(
                f"Unknown two-body damping '{type(damping).__name__}' for "
                f"dispersion model '{meta.label}'."
            )
        key = matched
    else:
        from ..damping.select import get_damping_id

        key = damping.lower().strip()
        try:
            key = get_damping_id(key, three_body=False)
        except ValueError:
            pass
    if key not in meta.allowed_dampings:
        allowed = ", ".join(meta.allowed_dampings)
        raise ValueError(
            f"Damping '{damping}' is not supported for dispersion model "
            f"'{meta.label}' (allowed: {allowed})."
        )
