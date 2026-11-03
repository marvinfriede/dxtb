# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
"""
ASE Support
===========

`ASE calculator <https://wiki.fysik.dtu.dk/ase/>`_ implementation for the
unified PyTorch dispersion library ``tad_dftd``.

This module provides a native PyTorch ASE :class:`~ase.calculators.calculator.Calculator`
that evaluates DFT-D dispersion energies, nuclear forces, and periodic stress
tensors without requiring a compiled Fortran library. To combine dispersion
with a baseline electronic structure or machine-learning calculator, use
:class:`~ase.calculators.mixing.SumCalculator` or :meth:`DFTD.add_calculator`.

The generic :class:`DFTD` calculator evaluates all supported dispersion models
in the library, selected with the ``model`` keyword (default ``"d4"``). The
model-specific convenience classes only change this default:

======================== ===================================================
 Calculator               Default model
======================== ===================================================
 ``DFTD``                 ``"d4"``, generic dispersion interface
 ``DFTD4``                ``"d4"``, DFT-D4 with rational (BJ) + ATM damping
 ``DFTD3``                ``"d3"``, DFT-D3 with rational (BJ) 2-body damping
======================== ===================================================

Supported ASE properties:

- ``energy`` (and ``free_energy``)
- ``energies`` (per-atom dispersion energies)
- ``forces``
- ``stress`` (for full-rank 3D periodic cells when ``compute_stress=True``)

Supported calculator keywords:

======================== ================== ============================================
 Keyword                  Default            Description
======================== ================== ============================================
 ``method``               ``None``           Density functional name for parameter lookup
 ``model``                ``"d4"``           Dispersion model (D1, D2, D3, D3S, D4, D4S, D4Srev)
 ``params_tweaks``        ``None``           Optional dict of explicit damping parameters
 ``damping_hint``         ``None``           Optional dict selecting ``"2b"`` / ``"3b"`` damping
 ``realspace_cutoff``     ``None``           Optional real-space cutoffs in Angstrom
 ``charge``               ``None``           Optional explicit total system charge for EEQ
 ``charges_key``          ``None``           Optional ``atoms.arrays`` key for atomic charges
 ``qmodel``               ``"eeq"``          Reference charge model for D4/D4S/D4Srev
 ``device``               ``"cpu"``          PyTorch execution device
 ``dtype``                ``torch.float64``  PyTorch floating-point precision
 ``compute_stress``       ``True``           Enable analytical stress for 3D periodic cells
======================== ================== ============================================

Example
-------
>>> from ase.build import molecule
>>> from tad_dftd.ase import DFTD
>>> atoms = molecule("H2O")
>>> atoms.calc = DFTD(method="PBE", model="d4")
>>> energy = atoms.get_potential_energy()
>>> forces = atoms.get_forces()
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

import numpy as np
import torch

try:
    from ase.atoms import Atoms  # type: ignore[import-untyped]
    from ase.calculators.calculator import (  # type: ignore[import-untyped]
        Calculator,
        InputError,
        PropertyNotImplementedError,
        all_changes,
    )
    from ase.calculators.mixing import (  # type: ignore[import-untyped]
        SumCalculator,
    )
    from ase.stress import (  # type: ignore[import-untyped]
        full_3x3_to_voigt_6_stress,
    )
    from ase.units import Bohr, Hartree  # type: ignore[import-untyped]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'tad_dftd.ase' submodule requires ASE to be installed."
    ) from exc

from tad_mctc import Structure

from .cutoff import Cutoff
from .damping import (
    THREE_BODY_DAMPINGS,
    TWO_BODY_DAMPINGS,
    Damping,
    DampingParam,
    PairData,
    ThreeBodyDamping,
    TripleData,
    TwoBodyDamping,
    damping_from_name,
    get_damping_id,
    identify_damping,
)
from .disp import dftd
from .model.catalogue import (
    ChargeModelId,
    DispersionModelId,
    ModelMetadata,
    get_model_metadata,
    resolve_charge_model_id,
    resolve_model_id,
    validate_model_damping,
)
from .param import get_params

__all__ = ["DFTD", "DFTD3", "DFTD4"]


class _ConfigurationError(InputError, ValueError):
    """Exception raised when calculator configuration or parameters are invalid."""


class _InvalidTypeConfigurationError(InputError, TypeError, ValueError):
    """Exception raised when a configuration option has an invalid type."""


class _StressNotDefinedError(PropertyNotImplementedError, ValueError):
    """Exception raised when stress is requested but not defined for the system."""


_ALLOWED_DAMPING_HINT_KEYS = frozenset({"2b", "3b"})

_ALLOWED_CUTOFF_KEYS = frozenset(f.name for f in fields(Cutoff))

_ALLOWED_PARAM_KEYS = frozenset(DampingParam._child_names)

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
    "torch.float64": torch.float64,
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "torch.float32": torch.float32,
}

_STATELESS_2B_TYPES: frozenset[type[TwoBodyDamping]] = frozenset(
    cls_2b for _, cls_2b in TWO_BODY_DAMPINGS
)

_STATELESS_3B_TYPES: frozenset[type[ThreeBodyDamping]] = frozenset(
    cls_3b for _, cls_3b in THREE_BODY_DAMPINGS
)


def _is_stateless_builtin_damping(damping: Damping) -> bool:
    """
    Return ``True`` if ``damping`` and its two-body/three-body components are
    unmodified instances of the built-in stateless catalogue damping classes.
    """
    if type(damping) is not Damping:
        return False
    two = damping.two
    if type(two) not in _STATELESS_2B_TYPES or bool(
        getattr(two, "__dict__", None)
    ):
        return False
    three = damping.three
    if three is not None and (
        type(three) not in _STATELESS_3B_TYPES
        or bool(getattr(three, "__dict__", None))
    ):
        return False
    return True


def _snapshot_damping_param(
    param: DampingParam,
) -> tuple[tuple[str, float], ...]:
    """Capture a canonical sorted ``(name, float_value)`` snapshot of ``param``."""
    items: list[tuple[str, float]] = []
    for name in sorted(param._child_names):
        val = getattr(param, name)
        if val is not None:
            fval = (
                float(val.detach().cpu().item())
                if isinstance(val, torch.Tensor)
                else float(val)
            )
            items.append((name, fval))
    return tuple(items)


@dataclass(frozen=True)
class _ResolvedConfig:
    """Validated, immutable calculator configuration resolved from parameters."""

    model_id: DispersionModelId
    meta: ModelMetadata
    damping: Damping
    param: DampingParam
    cutoff: Cutoff
    qmodel: str | None
    device: torch.device
    dtype: torch.dtype
    charge: float | None
    charges_key: str | None
    compute_stress: bool
    tweaks_snapshot: tuple[tuple[str, float], ...]
    param_snapshot: tuple[tuple[str, float], ...]

    @property
    def cacheable_damping(self) -> bool:
        """Return whether ``self.damping`` is an unmodified stateless built-in damping."""
        return _is_stateless_builtin_damping(self.damping)

    def has_param_drift(self, parameters: Mapping[str, Any]) -> bool:
        """Return whether ``params_tweaks`` or ``self.param`` has drifted since resolution."""
        return _has_mutable_param_drift(parameters, self)


def _resolve_device(device: str | torch.device | Any) -> torch.device:
    """Validate and resolve a PyTorch execution device."""
    if isinstance(device, bool) or not isinstance(device, (str, torch.device)):
        raise _InvalidTypeConfigurationError(
            f"Expected 'device' to be a str or torch.device, got '{type(device).__name__}'."
        )

    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise _ConfigurationError(
            f"Invalid PyTorch device '{device}'."
        ) from exc

    if resolved.type == "cpu":
        return resolved

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise _ConfigurationError(
                f"CUDA device '{device}' requested, but CUDA is not available."
            )
        if (
            resolved.index is not None
            and resolved.index >= torch.cuda.device_count()
        ):
            raise _ConfigurationError(
                f"CUDA device index {resolved.index} is out of range "
                f"(available devices: {torch.cuda.device_count()})."
            )
        return resolved

    if resolved.type == "mps":
        if not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise _ConfigurationError(
                f"MPS device '{device}' requested, but MPS is not available."
            )
        return resolved

    raise _ConfigurationError(
        f"Unsupported PyTorch device '{device}'; expected 'cpu' or 'cuda'."
    )


def _resolve_dtype(dtype: str | torch.dtype | Any) -> torch.dtype:
    """Validate and resolve a PyTorch floating-point dtype."""
    if isinstance(dtype, torch.dtype):
        if dtype in (torch.float64, torch.float32):
            return dtype
        raise _ConfigurationError(
            f"Unsupported dtype '{dtype}'; expected torch.float64 or torch.float32."
        )

    if isinstance(dtype, str):
        key = dtype.lower().strip()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        raise _ConfigurationError(
            f"Unknown dtype '{dtype}'; expected 'float64' or 'float32'."
        )

    raise _InvalidTypeConfigurationError(
        f"Expected 'dtype' to be a torch.dtype or str, got '{type(dtype).__name__}'."
    )


def _validate_tweaks_mapping(
    params_tweaks: Mapping[str, Any] | DampingParam | None,
) -> dict[str, float]:
    """Validate explicit damping parameter overrides and convert values to floats."""
    if params_tweaks is None:
        return {}

    if isinstance(params_tweaks, DampingParam):
        raw_items = {
            name: getattr(params_tweaks, name)
            for name in params_tweaks._child_names
            if getattr(params_tweaks, name) is not None
        }
    elif isinstance(params_tweaks, Mapping):
        raw_items = dict(params_tweaks)
    else:
        raise _InvalidTypeConfigurationError(
            "Expected 'params_tweaks' to be a mapping of damping parameters or None, "
            f"got '{type(params_tweaks).__name__}'."
        )

    unknown = sorted(set(raw_items) - _ALLOWED_PARAM_KEYS)
    if unknown:
        raise _InvalidTypeConfigurationError(
            f"Unsupported damping parameter(s) in 'params_tweaks': {', '.join(unknown)}."
        )

    cleaned: dict[str, float] = {}
    for key, value in raw_items.items():
        if isinstance(value, bool):
            raise _InvalidTypeConfigurationError(
                f"Damping parameter '{key}' must be a real number, not bool."
            )
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not torch.is_floating_point(value):
                raise _ConfigurationError(
                    f"Tensor damping parameter '{key}' must be a scalar floating-point tensor."
                )
            fval = float(value.detach().cpu().item())
        elif isinstance(value, Real):
            fval = float(value)
        else:
            raise _InvalidTypeConfigurationError(
                f"Damping parameter '{key}' must be a real number, got '{type(value).__name__}'."
            )

        if not math.isfinite(fval):
            raise _ConfigurationError(
                f"Damping parameter '{key}' must be finite, got {fval}."
            )
        cleaned[key] = fval

    return cleaned


def _resolve_damping(
    model_id: DispersionModelId,
    meta: ModelMetadata,
    method: str | None,
    damping_hint: Mapping[str, Any] | Damping | None,
    tweaks: Mapping[str, float],
) -> tuple[Damping, str | None, str | bool, str | None]:
    """
    Resolve the two-body and three-body damping objects from ``damping_hint``,
    ``model``, ``method``, and ``params_tweaks``.

    Preserves user-supplied :class:`Damping`, :class:`TwoBodyDamping`, and
    :class:`ThreeBodyDamping` instances while identifying their canonical
    damping families for model validation and functional parameter lookup.

    Returns ``(damping, canonical_2b, canonical_3b_or_false, raw_2b_hint)``.
    """
    if isinstance(damping_hint, Damping):
        if not isinstance(damping_hint.two, TwoBodyDamping):
            raise _InvalidTypeConfigurationError(
                "Expected 'damping_hint.two' to be a TwoBodyDamping instance, "
                f"got '{type(damping_hint.two).__name__}'."
            )
        if damping_hint.three is not None and not isinstance(
            damping_hint.three, ThreeBodyDamping
        ):
            raise _InvalidTypeConfigurationError(
                "Expected 'damping_hint.three' to be a ThreeBodyDamping instance "
                f"or None, got '{type(damping_hint.three).__name__}'."
            )
        raw_2b: Any = damping_hint.two
        raw_3b: Any = (
            False if damping_hint.three is None else damping_hint.three
        )
    elif damping_hint is None:
        raw_2b = None
        raw_3b = None
    elif isinstance(damping_hint, Mapping):
        unknown = sorted(set(damping_hint) - _ALLOWED_DAMPING_HINT_KEYS)
        if unknown:
            raise _ConfigurationError(
                f"Unknown key(s) in 'damping_hint': {', '.join(unknown)}; "
                "expected '2b' and/or '3b'."
            )
        raw_2b = damping_hint.get("2b")
        raw_3b = damping_hint.get("3b")
    else:
        raise _InvalidTypeConfigurationError(
            "Expected 'damping_hint' to be a mapping with keys '2b'/'3b', a "
            f"Damping instance, or None, got '{type(damping_hint).__name__}'."
        )

    two_obj: TwoBodyDamping | None = None
    raw_2b_clean: str | None = None

    if raw_2b is None:
        d2_name: str | None = meta.default_damping_2b
        try:
            validate_model_damping(model_id, d2_name)
        except ValueError as exc:
            raise _ConfigurationError(str(exc)) from exc
    elif not isinstance(raw_2b, bool) and isinstance(raw_2b, str):
        raw_2b_clean = raw_2b.lower().strip()
        try:
            validate_model_damping(model_id, raw_2b_clean)
            d2_name = get_damping_id(raw_2b_clean, three_body=False)
        except ValueError as exc:
            raise _ConfigurationError(str(exc)) from exc
    elif not isinstance(raw_2b, bool) and isinstance(raw_2b, TwoBodyDamping):
        two_obj = raw_2b
        d2_name = identify_damping(raw_2b)
        if d2_name is not None:
            try:
                validate_model_damping(model_id, d2_name)
            except ValueError as exc:
                raise _ConfigurationError(str(exc)) from exc
        elif meta.name in ("d1", "d2"):
            raise _ConfigurationError(
                f"Custom two-body damping '{type(raw_2b).__name__}' is "
                f"not supported for dispersion model '{meta.label}'."
            )
    else:
        raise _InvalidTypeConfigurationError(
            f"Invalid '2b' damping hint '{raw_2b}'; expected a damping name "
            "string or TwoBodyDamping instance."
        )

    three_obj: ThreeBodyDamping | None = None
    default_3b = meta.default_damping_3b or "zero_d4"

    if raw_3b is not None:
        if raw_3b is False or (
            isinstance(raw_3b, str)
            and raw_3b.lower().strip() in ("none", "false", "off")
        ):
            d3_spec: str | bool = False
        elif raw_3b is True or (
            isinstance(raw_3b, str)
            and raw_3b.lower().strip() in ("zero_avg", "zero-avg")
        ):
            d3_spec = default_3b
        elif not isinstance(raw_3b, bool) and isinstance(raw_3b, str):
            try:
                d3_spec = get_damping_id(
                    raw_3b.lower().strip(), three_body=True
                )
            except ValueError as exc:
                raise _ConfigurationError(str(exc)) from exc
        elif not isinstance(raw_3b, bool) and isinstance(
            raw_3b, ThreeBodyDamping
        ):
            three_obj = raw_3b
            d3_name = identify_damping(raw_3b)
            d3_spec = d3_name if d3_name is not None else "custom"
        else:
            raise _InvalidTypeConfigurationError(
                f"Invalid '3b' damping hint '{raw_3b}'; expected a damping name "
                "string, ThreeBodyDamping instance, or 'none'."
            )
    else:
        # When "3b" is not explicitly hinted:
        # - If "s9" is explicitly provided in params_tweaks, enable ATM when s9 != 0.
        # - Else if "2b" was explicitly hinted without "3b", disable ATM (matching Fortran DampingFunction(damping_2b=...)).
        # - Else if a functional method is provided, enable ATM by default for D4/D4S/D4Srev and disable for D1/D2/D3/D3S (matching Fortran catalogue.f90).
        # - Else (explicit params_tweaks without method and without s9), disable ATM.
        if "s9" in tweaks:
            d3_spec = False if tweaks["s9"] == 0.0 else default_3b
        elif raw_2b is not None:
            d3_spec = False
        elif method is not None and meta.name in ("d4", "d4s", "d4srev"):
            d3_spec = default_3b
        else:
            d3_spec = False

    if isinstance(damping_hint, Damping):
        return damping_hint, d2_name, d3_spec, raw_2b_clean

    if two_obj is None and three_obj is None:
        assert d2_name is not None
        assert isinstance(d3_spec, (str, bool))
        try:
            resolved = damping_from_name(d2_name, d3_spec, model=meta.name)
        except ValueError as exc:
            raise _ConfigurationError(str(exc)) from exc
        return resolved, d2_name, d3_spec, raw_2b_clean

    if two_obj is None:
        assert d2_name is not None
        try:
            two_obj = damping_from_name(d2_name, False, model=meta.name).two
        except ValueError as exc:
            raise _ConfigurationError(str(exc)) from exc

    if three_obj is None and d3_spec is not False:
        assert isinstance(d3_spec, str)
        try:
            three_obj = damping_from_name(
                meta.default_damping_2b, d3_spec, model=meta.name
            ).three
        except ValueError as exc:
            raise _ConfigurationError(str(exc)) from exc

    return Damping(two=two_obj, three=three_obj), d2_name, d3_spec, raw_2b_clean


def _resolve_parameters(
    meta: ModelMetadata,
    method: str | None,
    tweaks: Mapping[str, float],
    damping: Damping,
    d2_name: str | None,
    d3_spec: str | bool,
    raw_2b_hint: str | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DampingParam:
    """
    Resolve damping parameters from functional name and/or explicit parameter
    overrides and validate completeness against the selected ``damping``.
    """
    if method is not None:
        if not isinstance(method, str):
            raise _InvalidTypeConfigurationError(
                f"Expected 'method' to be a functional name string or None, got '{type(method).__name__}'."
            )
        method_clean = method.strip()
        if not method_clean:
            raise _ConfigurationError(
                "Expected 'method' to be a non-empty functional name string or None."
            )
        has_method = True
    else:
        method_clean = ""
        has_method = False

    if has_method:
        if meta.model_id == DispersionModelId.D4SREV:
            raise _ConfigurationError(
                "No functional damping parameter tables are available for model "
                "'d4srev'; provide explicit parameters via 'params_tweaks'."
            )

        if d2_name is None:
            raise _ConfigurationError(
                f"No {meta.label} functional parameters are available for custom "
                f"two-body damping '{type(damping.two).__name__}'; provide "
                "explicit parameters via 'params_tweaks' with 'method=None'."
            )

        if meta.name in ("d1", "d2"):
            if d3_spec is not False and "s9" not in tweaks:
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    "three-body dispersion."
                )
            damping_lookup: str | list[str] | None = "fermi"

        elif meta.name in ("d3", "d3s"):
            if d2_name not in (
                "rational",
                "zero",
                "mzero",
                "optimizedpower",
                "cso",
            ):
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    f"two-body damping '{d2_name}'."
                )
            if d3_spec not in (False, "zero_d3") and "s9" not in tweaks:
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    f"three-body damping '{d3_spec}'."
                )
            damping_lookup = "bjm" if raw_2b_hint == "bjm" else d2_name

        elif meta.name in ("d4", "d4s"):
            if d2_name != "rational" or (
                raw_2b_hint is not None
                and raw_2b_hint not in ("rational", "bj")
            ):
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    f"two-body damping '{raw_2b_hint or d2_name}'."
                )
            if d3_spec not in (False, "zero_d4") and "s9" not in tweaks:
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    f"three-body damping '{d3_spec}'."
                )
            damping_lookup = (
                ["bj-eeq-two", "bj-eeq-atm"]
                if d3_spec is False
                else ["bj-eeq-atm"]
            )
        else:
            damping_lookup = None

        try:
            merged_params = get_params(
                method_clean,
                model=meta.name,
                damping=damping_lookup,
                keep_meta=False,
                dtype=dtype,
                device=device,
            )
        except (KeyError, ValueError) as exc:
            raise _ConfigurationError(str(exc)) from exc

        if d3_spec is False:
            merged_params.pop("s9", None)
            merged_params.pop("rs9", None)

        for key, val in tweaks.items():
            merged_params[key] = torch.tensor(val, dtype=dtype, device=device)

        dparam = DampingParam.from_dict(
            merged_params, device=device, dtype=dtype
        )
    else:
        if not tweaks:
            raise _ConfigurationError(
                "Either a functional 'method' name or explicit 'params_tweaks' "
                "must be provided."
            )
        dparam = DampingParam.from_dict(tweaks, device=device, dtype=dtype)

    if (dparam.sxc is None) != (dparam.rsxc is None):
        raise _ConfigurationError(
            "Both 'sxc' and 'rsxc' are required when using XC screening."
        )

    # Validate that all parameters required by the selected 2B and 3B damping
    # functions are present.
    one = torch.ones((1,), dtype=dtype, device=device)
    dummy_pairs = PairData(
        distances=one,
        qq=one,
        c6=one,
        rvdw=one,
        znum=one,
        rdamp=one,
    )
    try:
        damping.two(dummy_pairs, dparam)
        if damping.three is not None:
            dummy_triples = TripleData(
                r=one,
                r2ij=one,
                r2ik=one,
                r2jk=one,
                rdampij=one,
                rdampik=one,
                rdampjk=one,
                rvdwij=one,
                rvdwik=one,
                rvdwjk=one,
            )
            damping.three(dummy_triples, dparam)
    except ValueError as exc:
        raise _ConfigurationError(str(exc)) from exc

    return dparam


def _resolve_cutoff(
    realspace_cutoff: Mapping[str, Any] | Cutoff | None,
    meta: ModelMetadata,
) -> Cutoff:
    """
    Resolve real-space cutoffs, converting ASE inputs from Angstrom to Bohr.
    """
    if realspace_cutoff is None:
        return meta.default_cutoff

    if isinstance(realspace_cutoff, Cutoff):
        return realspace_cutoff

    if not isinstance(realspace_cutoff, Mapping):
        raise _InvalidTypeConfigurationError(
            "Expected 'realspace_cutoff' to be a mapping of cutoff values in "
            f"Angstrom or None, got '{type(realspace_cutoff).__name__}'."
        )

    if not realspace_cutoff:
        return meta.default_cutoff

    unknown = sorted(set(realspace_cutoff) - _ALLOWED_CUTOFF_KEYS)
    if unknown:
        raise _ConfigurationError(
            f"Unknown key(s) in 'realspace_cutoff': {', '.join(unknown)}; "
            f"expected a subset of {sorted(_ALLOWED_CUTOFF_KEYS)}."
        )

    converted: dict[str, float] = {}
    for key, val in realspace_cutoff.items():
        if isinstance(val, bool) or not isinstance(val, Real):
            raise _InvalidTypeConfigurationError(
                f"Cutoff '{key}' must be a real number in Angstrom, got '{type(val).__name__}'."
            )
        fval = float(val)
        if not math.isfinite(fval):
            raise _ConfigurationError(
                f"Cutoff '{key}' must be finite, got {fval}."
            )
        if key in ("width2", "width3"):
            if fval < 0.0:
                raise _ConfigurationError(
                    f"Cutoff smoothing width '{key}' must be non-negative, got {fval}."
                )
        elif fval <= 0.0:
            raise _ConfigurationError(
                f"Cutoff '{key}' must be strictly positive, got {fval}."
            )
        converted[key] = fval / Bohr

    return Cutoff.from_model(meta.name, **converted)


def _resolve_qmodel(
    qmodel: str | int | ChargeModelId | None,
    model_id: DispersionModelId,
    meta: ModelMetadata,
) -> str | None:
    """Resolve and validate the reference charge model for ``model_id``."""
    try:
        if meta.charge_dep_2b:
            qid = resolve_charge_model_id(qmodel, model=model_id)
            return qid.name.lower() if qid is not None else "eeq"

        # For charge-independent models (D1, D2, D3, D3S), ignore the default "eeq"
        # setting, but reject excluded ("eeqbc") or explicitly non-default charge models.
        if qmodel in (
            None,
            "eeq",
            "eeq2019",
            ChargeModelId.EEQ,
            1,
        ) and not isinstance(qmodel, bool):
            return None

        resolve_charge_model_id(qmodel, model=model_id)
        return None
    except NotImplementedError:
        raise
    except TypeError as exc:
        raise _InvalidTypeConfigurationError(str(exc)) from exc
    except ValueError as exc:
        raise _ConfigurationError(str(exc)) from exc


def _resolve_charge_options(
    charge: Any,
    charges_key: Any,
    qmodel: str | None,
    meta: ModelMetadata,
) -> tuple[float | None, str | None]:
    """Validate total charge and external atomic partial charges key."""
    if charge is None:
        resolved_charge = None
    elif isinstance(charge, bool) or not isinstance(charge, Real):
        raise _InvalidTypeConfigurationError(
            f"Expected 'charge' to be a real number or None, got '{type(charge).__name__}'."
        )
    else:
        resolved_charge = float(charge)
        if not math.isfinite(resolved_charge):
            raise _ConfigurationError(
                f"Total 'charge' must be finite, got {resolved_charge}."
            )

    if charges_key is None:
        resolved_key = None
    elif not isinstance(charges_key, str) or not charges_key.strip():
        raise _InvalidTypeConfigurationError(
            "Expected 'charges_key' to be a non-empty string or None."
        )
    else:
        if not meta.charge_dep_2b:
            raise _ConfigurationError(
                f"Dispersion model '{meta.label}' does not use atomic partial "
                "charges ('charges_key' is only supported for D4, D4S, and D4Srev)."
            )
        resolved_key = charges_key.strip()

    if qmodel not in (None, "eeq") and resolved_key is None:
        raise _ConfigurationError(
            f"Charge model '{qmodel}' requires external atomic partial charges "
            "via 'charges_key'."
        )

    return resolved_charge, resolved_key


def _resolve_config(
    parameters: Mapping[str, Any],
    *,
    default_model: str | int | DispersionModelId = "d4",
) -> _ResolvedConfig:
    """Validate and resolve a calculator parameter mapping into an immutable :class:`_ResolvedConfig`."""
    with torch.inference_mode(False):
        compute_stress = parameters.get("compute_stress", True)
        if not isinstance(compute_stress, bool):
            raise _InvalidTypeConfigurationError(
                f"Expected 'compute_stress' to be a bool, got '{type(compute_stress).__name__}'."
            )

        device = _resolve_device(parameters.get("device", "cpu"))
        dtype = _resolve_dtype(parameters.get("dtype", torch.float64))

        raw_model = parameters.get("model")
        if raw_model is None:
            raw_model = default_model
        try:
            model_id = resolve_model_id(raw_model)
        except NotImplementedError:
            raise
        except TypeError as exc:
            raise _InvalidTypeConfigurationError(str(exc)) from exc
        except ValueError as exc:
            raise _ConfigurationError(str(exc)) from exc

        meta = get_model_metadata(model_id)

        qmodel = _resolve_qmodel(
            parameters.get("qmodel", "eeq"), model_id, meta
        )
        charge, charges_key = _resolve_charge_options(
            parameters.get("charge"),
            parameters.get("charges_key"),
            qmodel,
            meta,
        )
        cutoff = _resolve_cutoff(parameters.get("realspace_cutoff"), meta)
        tweaks = _validate_tweaks_mapping(parameters.get("params_tweaks"))

        method = parameters.get("method")
        damping, d2_name, d3_spec, raw_2b_hint = _resolve_damping(
            model_id,
            meta,
            method if isinstance(method, str) and method.strip() else None,
            parameters.get("damping_hint"),
            tweaks,
        )
        dparam = _resolve_parameters(
            meta,
            method,
            tweaks,
            damping,
            d2_name,
            d3_spec,
            raw_2b_hint,
            device=device,
            dtype=dtype,
        )

        return _ResolvedConfig(
            model_id=model_id,
            meta=meta,
            damping=damping,
            param=dparam,
            cutoff=cutoff,
            qmodel=qmodel,
            device=device,
            dtype=dtype,
            charge=charge,
            charges_key=charges_key,
            compute_stress=compute_stress,
            tweaks_snapshot=tuple(sorted(tweaks.items())),
            param_snapshot=_snapshot_damping_param(dparam),
        )


def _has_mutable_param_drift(
    parameters: Mapping[str, Any],
    cfg: _ResolvedConfig,
) -> bool:
    """
    Return ``True`` if ``params_tweaks`` (e.g., a :class:`DampingParam` or
    mapping with tensor values) or ``cfg.param`` has been modified in place
    since ``cfg`` was resolved.
    """
    try:
        current_tweaks = tuple(
            sorted(
                _validate_tweaks_mapping(
                    parameters.get("params_tweaks")
                ).items()
            )
        )
        if current_tweaks != cfg.tweaks_snapshot:
            return True
        if _snapshot_damping_param(cfg.param) != cfg.param_snapshot:
            return True
    except Exception:
        return True
    return False


def _copy_parameter_value(value: Any) -> Any:
    """Copy mutable mapping containers while preserving object instances."""
    if isinstance(value, Mapping):
        return {k: _copy_parameter_value(v) for k, v in value.items()}
    return value


_ALWAYS_UNEQUAL_PARAM_TYPES = (
    Damping,
    TwoBodyDamping,
    ThreeBodyDamping,
    Cutoff,
    DampingParam,
    torch.Tensor,
)


def _parameters_equal(old: Any, new: Any) -> bool:
    """Compare two calculator parameter values safely for change detection."""
    if isinstance(old, _ALWAYS_UNEQUAL_PARAM_TYPES) or isinstance(
        new, _ALWAYS_UNEQUAL_PARAM_TYPES
    ):
        return False
    if old is new:
        return True
    if type(old) is not type(new):
        return False
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        if old.keys() != new.keys():
            return False
        return all(_parameters_equal(old[k], new[k]) for k in old)
    try:
        return bool(old == new)
    except Exception:
        return False


def _resolve_total_charge(
    atoms: Atoms,
    cfg: _ResolvedConfig,
) -> torch.Tensor:
    """
    Resolve the total system charge used by automatic EEQ evaluation.
    """
    if cfg.charge is not None:
        total_q = float(cfg.charge)
    else:
        init_q = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
        if not np.all(np.isfinite(init_q)):
            raise ValueError(
                "ASE 'initial_charges' array contains non-finite values."
            )
        total_q = float(init_q.sum()) if init_q.size > 0 else 0.0

    return torch.tensor(total_q, dtype=cfg.dtype, device=cfg.device)


def _resolve_external_charges(
    atoms: Atoms,
    cfg: _ResolvedConfig,
    num_atoms: int,
) -> torch.Tensor | None:
    """
    Validate and extract externally supplied atomic partial charges from
    ``atoms.arrays[cfg.charges_key]``.
    """
    if cfg.charges_key is None:
        return None

    if cfg.charges_key not in atoms.arrays:
        raise ValueError(
            f"External atomic partial charges key '{cfg.charges_key}' was not "
            "found in 'atoms.arrays'."
        )

    raw_arr = np.asarray(atoms.arrays[cfg.charges_key])
    if raw_arr.dtype.kind in (
        "b",
        "c",
        "O",
        "S",
        "U",
        "V",
    ) or not np.issubdtype(raw_arr.dtype, np.number):
        raise ValueError(
            f"External atomic partial charges '{cfg.charges_key}' must have a "
            f"real numeric dtype, got '{raw_arr.dtype}'."
        )

    if raw_arr.ndim != 1 or raw_arr.shape != (num_atoms,):
        raise ValueError(
            f"External atomic partial charges '{cfg.charges_key}' must have "
            f"1D shape ({num_atoms},), got {raw_arr.shape}."
        )

    q_np = raw_arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(q_np)):
        raise ValueError(
            f"External atomic partial charges '{cfg.charges_key}' contain "
            "non-finite values."
        )

    return torch.as_tensor(q_np, dtype=cfg.dtype, device=cfg.device)


def _build_structure(
    atoms: Atoms,
    cfg: _ResolvedConfig,
    *,
    need_forces: bool = False,
    need_stress: bool = False,
) -> tuple[
    Structure,
    torch.Tensor,
    torch.Tensor | None,
    float | None,
    torch.Tensor | None,
]:
    """
    Convert an ASE :class:`~ase.atoms.Atoms` object into a fresh
    :class:`~tad_mctc.Structure` in atomic units (Bohr) without mutating ``atoms``.
    """
    raw_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    if raw_numbers.ndim != 1 or raw_numbers.size == 0:
        raise ValueError("ASE Atoms object must contain at least one atom.")

    if np.any(raw_numbers <= 0) or np.any(raw_numbers > cfg.meta.max_elem):
        bad_mask = (raw_numbers <= 0) | (raw_numbers > cfg.meta.max_elem)
        bad = int(raw_numbers[bad_mask][0])
        raise ValueError(
            f"Unsupported atomic number Z={bad} for {cfg.meta.label} "
            f"(supported range: 1..{cfg.meta.max_elem})."
        )

    numbers = torch.as_tensor(raw_numbers, dtype=torch.long, device=cfg.device)

    raw_positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    if raw_positions.shape != (raw_numbers.size, 3):
        raise ValueError(
            f"Expected positions of shape ({raw_numbers.size}, 3), got {raw_positions.shape}."
        )
    if not np.all(np.isfinite(raw_positions)):
        raise ValueError("Atomic positions contain non-finite values.")

    positions_bohr = (
        torch.as_tensor(raw_positions, dtype=cfg.dtype, device=cfg.device)
        / Bohr
    )
    if need_forces:
        positions_bohr = positions_bohr.detach().requires_grad_(True)

    pbc_flags = np.asarray(atoms.get_pbc(), dtype=bool)
    if pbc_flags.shape != (3,):
        raise ValueError(
            f"Expected 3 periodic boundary flags, got shape {pbc_flags.shape}."
        )

    has_any_pbc = bool(pbc_flags.any())
    is_3d_pbc = bool(pbc_flags.all())

    if need_stress and not cfg.compute_stress:
        raise _StressNotDefinedError(
            "Stress calculation is disabled because 'compute_stress=False'."
        )

    if need_stress and not is_3d_pbc:
        raise _StressNotDefinedError(
            "Stress is only defined for 3D periodic structures with "
            f"pbc=[True, True, True] (got pbc={pbc_flags.tolist()})."
        )

    strain: torch.Tensor | None = None
    volume_ang3: float | None = None

    if has_any_pbc:
        raw_cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
        if raw_cell.shape != (3, 3) or not np.all(np.isfinite(raw_cell)):
            raise ValueError(
                "Periodic cell vectors must be a finite (3, 3) array."
            )
        det_cell = float(np.linalg.det(raw_cell))
        if is_3d_pbc and abs(det_cell) <= 1.0e-12:
            if need_stress:
                raise _StressNotDefinedError(
                    "Stress is not defined for a degenerate 3D periodic cell "
                    "with zero volume."
                )
            raise ValueError(
                "3D periodic cell vectors are singular (cell volume is zero)."
            )

        lattice_bohr = (
            torch.as_tensor(raw_cell, dtype=cfg.dtype, device=cfg.device) / Bohr
        )
        periodic: torch.Tensor | None = torch.as_tensor(
            pbc_flags, dtype=torch.bool, device=cfg.device
        )

        if need_stress:
            volume_ang3 = abs(det_cell)
            strain = torch.zeros(
                (3, 3), dtype=cfg.dtype, device=cfg.device, requires_grad=True
            )
            scale = torch.eye(
                3, dtype=cfg.dtype, device=cfg.device
            ) + strain.transpose(0, 1)
            positions_eval = positions_bohr @ scale
            lattice: torch.Tensor | None = lattice_bohr @ scale
        else:
            positions_eval = positions_bohr
            lattice = lattice_bohr
    else:
        positions_eval = positions_bohr
        lattice = None
        periodic = None

    total_charge = _resolve_total_charge(atoms, cfg)
    q_external = _resolve_external_charges(atoms, cfg, int(raw_numbers.size))

    structure = Structure(
        numbers=numbers,
        positions=positions_eval,
        charge=total_charge,
        lattice=lattice,
        periodic=periodic,
    )
    return structure, positions_bohr, strain, volume_ang3, q_external


def _calculate_energy_and_derivatives(
    atoms: Atoms,
    cfg: _ResolvedConfig,
    properties: set[str],
) -> dict[str, Any]:
    """
    Evaluate dispersion energies, nuclear forces, and periodic stress in atomic
    units and convert outputs to ASE units (eV, eV/Angstrom, and eV/Angstrom^3).
    """
    need_forces = "forces" in properties
    need_stress = "stress" in properties
    need_grad = need_forces or need_stress

    with torch.inference_mode(False), torch.set_grad_enabled(need_grad):
        structure, positions_bohr, strain, volume_ang3, q_external = (
            _build_structure(
                atoms,
                cfg,
                need_forces=need_forces,
                need_stress=need_stress,
            )
        )

        extra_kwargs: dict[str, Any] = {}
        if cfg.qmodel is not None:
            extra_kwargs["qmodel"] = cfg.qmodel
        if q_external is not None:
            extra_kwargs["q"] = q_external

        atom_energies = dftd(
            structure,
            cfg.param,
            model=cfg.model_id,
            damping=cfg.damping,
            cutoff=cfg.cutoff,
            **extra_kwargs,
        )
        total_energy_eh = atom_energies.sum()

        grad_pos: torch.Tensor | None = None
        virial_eh: torch.Tensor | None = None

        if need_forces and need_stress:
            assert strain is not None
            grad_pos, virial_eh = torch.autograd.grad(
                total_energy_eh,
                (positions_bohr, strain),
                retain_graph=False,
                create_graph=False,
            )
        elif need_forces:
            grad_pos = torch.autograd.grad(
                total_energy_eh,
                positions_bohr,
                retain_graph=False,
                create_graph=False,
            )[0]
        elif need_stress:
            assert strain is not None
            virial_eh = torch.autograd.grad(
                total_energy_eh,
                strain,
                retain_graph=False,
                create_graph=False,
            )[0]

    energy_ev = float(total_energy_eh.detach().cpu().item()) * Hartree
    energies_ev = (
        atom_energies.detach().cpu().numpy().astype(np.float64) * Hartree
    )

    results: dict[str, Any] = {
        "energy": energy_ev,
        "free_energy": energy_ev,
        "energies": energies_ev,
    }

    if grad_pos is not None:
        forces_ev_ang = (
            -grad_pos.detach().cpu().numpy().astype(np.float64)
        ) * (Hartree / Bohr)
        results["forces"] = forces_ev_ang

    if virial_eh is not None:
        assert volume_ang3 is not None and volume_ang3 > 0.0
        stress_3x3_ev_ang3 = virial_eh.detach().cpu().numpy().astype(
            np.float64
        ) * (Hartree / volume_ang3)
        results["stress"] = np.asarray(
            full_3x3_to_voigt_6_stress(stress_3x3_ev_ang3), dtype=np.float64
        )

    return results


class DFTD(Calculator):
    """
    ASE calculator for Grimme-lab DFT-D dispersion corrections backed by
    ``tad_dftd``.

    Parameters
    ----------
    atoms : Atoms | None, optional
        Optional ASE :class:`~ase.atoms.Atoms` object to attach the calculator to.
    method : str | None, optional
        Density functional name for parameter lookup (e.g. ``"PBE"``, ``"B3LYP"``).
    model : str | int | DispersionModelId, optional
        Dispersion model identifier: string name (``"d1"``, ``"d2"``, ``"d3"``,
        ``"d3s"``, ``"d4"``, ``"d4s"``, ``"d4srev"``), enum member, or integer
        model ID (``D1=10``, ``D2=20``, ``D3=30``, ``D3S=31``, ``D4=40``,
        ``D4S=41``, ``D4Srev=42``; ``D5=50`` is excluded). Defaults to ``"d4"``.
    params_tweaks : Mapping[str, Any] | DampingParam | None, optional
        Explicit damping parameters or overrides applied on top of ``method``.
    damping_hint : Mapping[str, Any] | Damping | None, optional
        Optional dictionary with ``"2b"`` (damping name or :class:`TwoBodyDamping`
        instance) and/or ``"3b"`` (damping name, :class:`ThreeBodyDamping`
        instance, or ``"none"`` to disable ATM), or a direct :class:`Damping`
        instance. Built-in stateless catalogue damping classes retain normal ASE
        result caching across unchanged :class:`~ase.atoms.Atoms` states, whereas
        custom or stateful damping subclasses bypass ASE result reuse so in-place
        mutations of damping state never return stale energies or derivatives.
    realspace_cutoff : Mapping[str, float] | Cutoff | None, optional
        Optional real-space cutoffs (``disp2``, ``disp3``, ``cn``, ``cn_eeq``,
        ``width2``, ``width3``) in Angstrom, or a native :class:`Cutoff`
        instance in Bohr.
    charge : float | int | None, optional
        Explicit total system charge for automatic EEQ evaluation. If ``None``,
        determined from ``atoms.get_initial_charges().sum()`` (defaulting to ``0.0``).
    charges_key : str | None, optional
        Name of an entry in ``atoms.arrays`` supplying external atomic partial
        charges for charge-dependent models (D4, D4S, D4Srev).
    qmodel : str | int | ChargeModelId | None, optional
        Reference charge model for D4/D4S/D4Srev (default ``"eeq"``).
    device : str | torch.device, optional
        PyTorch device for dispersion evaluation. Defaults to ``"cpu"``.
    dtype : str | torch.dtype, optional
        PyTorch floating-point dtype. Defaults to ``torch.float64``.
    compute_stress : bool, optional
        Whether to advertise and evaluate the periodic stress tensor. Defaults
        to ``True``.
    """

    implemented_properties: list[str] = [
        "energy",
        "energies",
        "free_energy",
        "forces",
        "stress",
    ]

    default_parameters: dict[str, Any] = {
        "method": None,
        "model": "d4",
        "params_tweaks": None,
        "damping_hint": None,
        "realspace_cutoff": None,
        "charge": None,
        "charges_key": None,
        "qmodel": "eeq",
        "device": "cpu",
        "dtype": torch.float64,
        "compute_stress": True,
    }

    _UNSET = object()

    def __init__(
        self,
        atoms: Atoms | None = None,
        method: str | None | object = _UNSET,
        model: str | int | DispersionModelId | object = _UNSET,
        params_tweaks: (
            Mapping[str, Any] | DampingParam | None | object
        ) = _UNSET,
        damping_hint: Mapping[str, Any] | Damping | None | object = _UNSET,
        realspace_cutoff: Mapping[str, float] | Cutoff | None | object = _UNSET,
        *,
        charge: float | int | None | object = _UNSET,
        charges_key: str | None | object = _UNSET,
        qmodel: str | int | ChargeModelId | None | object = _UNSET,
        device: str | torch.device | object = _UNSET,
        dtype: str | torch.dtype | object = _UNSET,
        compute_stress: bool | object = _UNSET,
        **kwargs: Any,
    ) -> None:
        """Initialize the DFTD ASE calculator."""
        self._resolved_config: _ResolvedConfig | None = None
        self._last_external_charges: (
            np.ndarray[Any, np.dtype[np.float64]] | None
        ) = None
        self.results: dict[str, Any] = {}
        self.implemented_properties = list(type(self).implemented_properties)

        init_kwargs: dict[str, Any] = dict(kwargs)
        explicit_args = {
            "method": method,
            "model": model,
            "params_tweaks": params_tweaks,
            "damping_hint": damping_hint,
            "realspace_cutoff": realspace_cutoff,
            "charge": charge,
            "charges_key": charges_key,
            "qmodel": qmodel,
            "device": device,
            "dtype": dtype,
            "compute_stress": compute_stress,
        }
        for key, value in explicit_args.items():
            if value is not self._UNSET:
                init_kwargs[key] = value

        super().__init__(atoms=None, **init_kwargs)
        if atoms is not None:
            atoms.calc = self

    def add_calculator(self, other: Calculator) -> SumCalculator:
        """
        Combine this dispersion calculator with another ASE calculator via
        :class:`~ase.calculators.mixing.SumCalculator`.
        """
        return SumCalculator([self, other])

    def _resolve_current_config(self) -> _ResolvedConfig:
        """Validate and resolve the current ``self.parameters`` dictionary."""
        return _resolve_config(
            self.parameters,
            default_model=self.default_parameters["model"],
        )

    def set(self, **kwargs: Any) -> dict[str, Any]:
        """
        Transactionally validate and apply calculator parameter updates,
        defensively copying mutable mappings and resetting cached results on change.
        """
        unknown = sorted(set(kwargs) - set(self.default_parameters))
        if unknown:
            raise _InvalidTypeConfigurationError(
                f"Unexpected keyword argument(s) for DFTD calculator: {', '.join(unknown)}."
            )

        copied_kwargs = {
            key: _copy_parameter_value(value) for key, value in kwargs.items()
        }
        current_params = (
            dict(self.parameters)
            if getattr(self, "parameters", None) is not None
            else dict(self.default_parameters)
        )
        candidate_params = {
            **self.default_parameters,
            **current_params,
            **copied_kwargs,
        }

        new_resolved_config = _resolve_config(
            candidate_params,
            default_model=self.default_parameters["model"],
        )

        had_prev_config = self._resolved_config is not None
        changed_parameters: dict[str, Any] = {}
        for key, val in copied_kwargs.items():
            old_val = self.parameters.get(key)
            if (
                not had_prev_config
                or key not in self.parameters
                or not _parameters_equal(old_val, val)
            ):
                changed_parameters[key] = val
            self.parameters[key] = val

        self._resolved_config = new_resolved_config

        base_props = list(type(self).implemented_properties)
        if not new_resolved_config.compute_stress:
            self.implemented_properties = [
                p for p in base_props if p != "stress"
            ]
        else:
            self.implemented_properties = base_props

        if changed_parameters:
            self.reset()

        return changed_parameters

    def reset(self) -> None:
        """Clear all cached calculation results and external charge state."""
        super().reset()
        self._last_external_charges = None

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        """
        Check for any system changes since the last calculation, including
        changes to ``atoms.arrays[charges_key]`` when external charges are enabled,
        in-place tensor edits in ``params_tweaks``, and custom/stateful damping objects.
        """
        changes = list(super().check_state(atoms, tol=tol))
        cfg = self._resolved_config
        if self.atoms is not None and cfg is not None:
            if not cfg.cacheable_damping and "damping_state" not in changes:
                changes.append("damping_state")
            if (
                cfg.has_param_drift(self.parameters)
                and "params_tweaks" not in changes
            ):
                changes.append("params_tweaks")

        charges_key = (
            cfg.charges_key
            if cfg is not None
            else self.parameters.get("charges_key")
        )
        if (
            isinstance(charges_key, str)
            and charges_key.strip()
            and self.atoms is not None
        ):
            key = charges_key.strip()
            change_tag = f"arrays:{key}"
            if change_tag not in changes:
                has_in_self = key in self.atoms.arrays
                has_in_atoms = key in atoms.arrays
                if has_in_self != has_in_atoms:
                    changes.append(change_tag)
                elif has_in_atoms:
                    curr = np.asarray(atoms.arrays[key])
                    cached_ref = self._last_external_charges
                    ref = (
                        cached_ref
                        if cached_ref is not None
                        else np.asarray(self.atoms.arrays[key])
                    )
                    if (
                        curr.shape != ref.shape
                        or curr.dtype.kind in ("b", "c", "O", "S", "U", "V")
                        or not np.issubdtype(curr.dtype, np.number)
                    ):
                        changes.append(change_tag)
                    else:
                        curr_f64 = curr.astype(np.float64, copy=False)
                        if cached_ref is not None:
                            ref_f64 = cached_ref
                            ref_finite = True
                        else:
                            ref_f64 = ref.astype(np.float64, copy=False)
                            ref_finite = bool(np.all(np.isfinite(ref_f64)))
                        if not (
                            np.all(np.isfinite(curr_f64))
                            and ref_finite
                            and np.allclose(
                                curr_f64, ref_f64, rtol=0.0, atol=tol
                            )
                        ):
                            changes.append(change_tag)

        return changes

    def get_property(
        self,
        name: str,
        atoms: Atoms | None = None,
        allow_calculation: bool = True,
    ) -> Any:
        """Return a calculated property, checking state even when ``atoms`` is omitted."""
        if atoms is None:
            atoms = self.atoms
        return super().get_property(
            name, atoms=atoms, allow_calculation=allow_calculation
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] | None = None,
        system_changes: list[str] | tuple[str, ...] = all_changes,
    ) -> None:
        """
        Evaluate the requested ASE properties for ``atoms`` and store them in
        ``self.results``.
        """
        if properties is None:
            target_atoms = atoms if atoms is not None else self.atoms
            if target_atoms is not None and bool(
                np.asarray(target_atoms.get_pbc(), dtype=bool).all()
            ):
                requested_props = list(self.implemented_properties)
            else:
                requested_props = [
                    p for p in self.implemented_properties if p != "stress"
                ]
        else:
            requested_props = list(properties)

        cfg = self._resolved_config
        param_drift = (
            cfg is None
            or "params_tweaks" in system_changes
            or cfg.has_param_drift(self.parameters)
        )
        had_cached_results = (
            bool(self.results)
            and not system_changes
            and cfg is not None
            and cfg.cacheable_damping
            and not param_drift
        )
        super().calculate(
            atoms=atoms,
            properties=requested_props,
            system_changes=list(system_changes),
        )

        if self.atoms is None:
            raise _ConfigurationError(
                "No ASE Atoms object is associated with the calculator."
            )

        try:
            if param_drift:
                self._resolved_config = self._resolve_current_config()
            assert self._resolved_config is not None

            new_results = _calculate_energy_and_derivatives(
                self.atoms,
                self._resolved_config,
                set(requested_props),
            )
        except _StressNotDefinedError:
            if not had_cached_results:
                self.reset()
            raise
        except Exception:
            self.reset()
            raise

        if had_cached_results:
            self.results.update(new_results)
        else:
            self.results = new_results

        key = self._resolved_config.charges_key
        if key is not None and key in self.atoms.arrays:
            self._last_external_charges = np.asarray(
                self.atoms.arrays[key], dtype=np.float64
            ).copy()
        else:
            self._last_external_charges = None


class DFTD4(DFTD):
    """
    ASE calculator for the DFT-D4 dispersion model.
    Identical to :class:`DFTD` with ``model="d4"`` as default.
    """

    default_parameters: dict[str, Any] = {
        **DFTD.default_parameters,
        "model": "d4",
    }


class DFTD3(DFTD):
    """
    ASE calculator for the DFT-D3 dispersion model.
    Identical to :class:`DFTD` with ``model="d3"`` as default (rational
    Becke-Johnson two-body damping without ATM three-body term unless enabled
    via ``damping_hint`` or ``params_tweaks``).
    """

    default_parameters: dict[str, Any] = {
        **DFTD.default_parameters,
        "model": "d3",
    }
