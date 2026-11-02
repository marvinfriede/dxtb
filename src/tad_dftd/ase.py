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

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
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
    Damping,
    DampingParam,
    PairData,
    TripleData,
    TwoBodyDamping,
    damping_from_name,
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

_ALLOWED_CUTOFF_KEYS = frozenset(
    {"disp2", "disp3", "cn", "cn_eeq", "width2", "width3"}
)

_ALLOWED_PARAM_KEYS = frozenset(
    {
        "s6",
        "s8",
        "s9",
        "s10",
        "a1",
        "a2",
        "a3",
        "a4",
        "a13",
        "a23",
        "rs6",
        "rs8",
        "rs9",
        "alp",
        "alp3",
        "bet",
        "sxc",
        "rsxc",
    }
)

_TWO_BODY_NAME_ALIASES: dict[str, str] = {
    "bj": "rational",
    "bjm": "rational",
    "chg": "zero",
    "d1": "fermi",
    "d2": "fermi",
    "zerom": "mzero",
    "m-zero": "mzero",
    "op": "optimizedpower",
    "optpower": "optimizedpower",
    "optimisedpower": "optimizedpower",
    "opt-power": "optimizedpower",
    "optimized-power": "optimizedpower",
    "optimised-power": "optimizedpower",
    "tang_toennies": "screened",
    "z-damping": "z",
    "koide-bjm": "koide",
}

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


def _resolve_device(device: str | torch.device | Any) -> torch.device:
    """Validate and resolve a PyTorch execution device."""
    if isinstance(device, bool) or not isinstance(device, (str, torch.device)):
        raise _InvalidTypeConfigurationError(
            f"Expected 'device' to be a str or torch.device, got '{type(device).__name__}'."
        )

    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise _ConfigurationError(f"Invalid PyTorch device '{device}'.") from exc

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
        if value is None:
            continue
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
) -> tuple[Damping, str, str | bool, str | None]:
    """
    Resolve the two-body and three-body damping objects from ``damping_hint``,
    ``model``, ``method``, and ``params_tweaks``.

    Returns ``(damping, canonical_2b, canonical_3b_or_false, raw_2b_hint)``.
    """
    if isinstance(damping_hint, Damping):
        validate_model_damping(model_id, damping_hint.two)
        d2_name = meta.default_damping_2b
        d3_spec: str | bool = damping_hint.three is not None
        return damping_hint, d2_name, d3_spec, None

    if damping_hint is None:
        hint: dict[str, Any] = {}
    elif isinstance(damping_hint, Mapping):
        unknown = sorted(set(damping_hint) - _ALLOWED_DAMPING_HINT_KEYS)
        if unknown:
            raise _ConfigurationError(
                f"Unknown key(s) in 'damping_hint': {', '.join(unknown)}; "
                "expected '2b' and/or '3b'."
            )
        hint = dict(damping_hint)
    else:
        raise _InvalidTypeConfigurationError(
            "Expected 'damping_hint' to be a mapping with keys '2b'/'3b' or None, "
            f"got '{type(damping_hint).__name__}'."
        )

    raw_2b = hint.get("2b")
    raw_3b = hint.get("3b")

    if raw_2b is not None:
        if isinstance(raw_2b, bool) or not isinstance(
            raw_2b, (str, TwoBodyDamping)
        ):
            raise _InvalidTypeConfigurationError(
                f"Invalid '2b' damping hint '{raw_2b}'; expected a damping name string."
            )
        validate_model_damping(model_id, raw_2b)
        if isinstance(raw_2b, str):
            raw_2b_clean = raw_2b.lower().strip()
            d2_name = _TWO_BODY_NAME_ALIASES.get(raw_2b_clean, raw_2b_clean)
        else:
            d2_name = meta.default_damping_2b
            raw_2b_clean = None
    else:
        raw_2b_clean = None
        d2_name = meta.default_damping_2b
        validate_model_damping(model_id, d2_name)

    if raw_3b is not None:
        if raw_3b is False or (
            isinstance(raw_3b, str)
            and raw_3b.lower().strip() in ("none", "false", "off")
        ):
            d3_spec = False
        elif raw_3b is True or (
            isinstance(raw_3b, str)
            and raw_3b.lower().strip() in ("zero_avg", "zero-avg")
        ):
            d3_spec = "zero_d3" if meta.name in ("d3", "d3s") else "zero_d4"
        elif isinstance(raw_3b, str):
            d3_spec = raw_3b.lower().strip()
        else:
            raise _InvalidTypeConfigurationError(
                f"Invalid '3b' damping hint '{raw_3b}'; expected a damping name string or 'none'."
            )
    else:
        # When "3b" is not explicitly hinted:
        # - If "s9" is explicitly provided in params_tweaks, enable ATM when s9 != 0.
        # - Else if "2b" was explicitly hinted without "3b", disable ATM (matching Fortran DampingFunction(damping_2b=...)).
        # - Else if a functional method is provided, enable ATM by default for D4/D4S/D4Srev and disable for D1/D2/D3/D3S (matching Fortran catalogue.f90).
        # - Else (explicit params_tweaks without method and without s9), disable ATM.
        if "s9" in tweaks:
            if tweaks["s9"] == 0.0:
                d3_spec = False
            else:
                d3_spec = "zero_d3" if meta.name in ("d3", "d3s") else "zero_d4"
        elif raw_2b is not None:
            d3_spec = False
        elif method is not None:
            d3_spec = "zero_d4" if meta.name in ("d4", "d4s", "d4srev") else False
        else:
            d3_spec = False

    try:
        damping = damping_from_name(d2_name, d3_spec, model=meta.name)
    except ValueError as exc:
        raise _ConfigurationError(str(exc)) from exc

    return damping, d2_name, d3_spec, raw_2b_clean


def _resolve_parameters(
    meta: ModelMetadata,
    method: str | None,
    tweaks: Mapping[str, float],
    damping: Damping,
    d2_name: str,
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
    if method is not None and not isinstance(method, str):
        raise _InvalidTypeConfigurationError(
            f"Expected 'method' to be a functional name string or None, got '{type(method).__name__}'."
        )

    method_clean = method.strip() if isinstance(method, str) else ""
    has_method = bool(method_clean)

    if has_method:
        if meta.model_id == DispersionModelId.D4SREV:
            raise _ConfigurationError(
                "No functional damping parameter tables are available for model "
                "'d4srev'; provide explicit parameters via 'params_tweaks'."
            )

        if meta.name in ("d1", "d2"):
            if d3_spec is not False and "s9" not in tweaks:
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    "three-body dispersion."
                )
            damping_lookup: str | list[str] | None = "fermi"

        elif meta.name in ("d3", "d3s"):
            if d2_name not in ("rational", "zero", "mzero", "optimizedpower", "cso"):
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    f"two-body damping '{d2_name}'."
                )
            if d3_spec not in (False, "zero_d3") and "s9" not in tweaks:
                raise _ConfigurationError(
                    f"No {meta.label} functional parameters are available for "
                    f"three-body damping '{d3_spec}'."
                )
            damping_lookup = raw_2b_hint if raw_2b_hint is not None else d2_name

        elif meta.name in ("d4", "d4s"):
            if d2_name != "rational" or (
                raw_2b_hint is not None and raw_2b_hint not in ("rational", "bj")
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
        if val is None:
            continue
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
    if meta.charge_dep_2b:
        qid = resolve_charge_model_id(qmodel, model=model_id)
        return qid.name.lower() if qid is not None else "eeq"

    # For charge-independent models (D1, D2, D3, D3S), ignore the default "eeq"
    # setting, but reject excluded ("eeqbc") or explicitly non-default charge models.
    if qmodel in (None, "eeq", "eeq2019", ChargeModelId.EEQ, 1) and not isinstance(
        qmodel, bool
    ):
        return None

    resolve_charge_model_id(qmodel, model=model_id)
    return None


def _resolve_charge_options(
    charge: Any,
    charges_key: Any,
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

    return resolved_charge, resolved_key


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
    if (
        raw_arr.dtype.kind in ("b", "c", "O", "S", "U", "V")
        or not np.issubdtype(raw_arr.dtype, np.number)
    ):
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
            torch.as_tensor(raw_cell, dtype=cfg.dtype, device=cfg.device)
            / Bohr
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
        forces_ev_ang = (-grad_pos.detach().cpu().numpy().astype(np.float64)) * (
            Hartree / Bohr
        )
        results["forces"] = forces_ev_ang

    if virial_eh is not None:
        assert volume_ang3 is not None and volume_ang3 > 0.0
        stress_3x3_ev_ang3 = (
            virial_eh.detach().cpu().numpy().astype(np.float64)
            * (Hartree / volume_ang3)
        )
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
        Dispersion model identifier (``"d1"``, ``"d2"``, ``"d3"``, ``"d3s"``,
        ``"d4"``, ``"d4s"``, ``"d4srev"``). Defaults to ``"d4"``.
    params_tweaks : Mapping[str, Any] | None, optional
        Explicit damping parameters or overrides applied on top of ``method``.
    damping_hint : Mapping[str, Any] | None, optional
        Optional dictionary with ``"2b"`` and/or ``"3b"`` damping names
        (use ``{"3b": "none"}`` to disable the three-body ATM term).
    realspace_cutoff : Mapping[str, float] | Cutoff | None, optional
        Optional real-space cutoffs (``disp2``, ``disp3``, ``cn``, ``cn_eeq``,
        ``width2``, ``width3``) in Angstrom.
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
        params_tweaks: Mapping[str, Any] | DampingParam | None | object = _UNSET,
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
        self._last_external_charges: np.ndarray[Any, np.dtype[np.float64]] | None = None
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

        super().__init__(atoms=atoms, **init_kwargs)

    def add_calculator(self, other: Calculator) -> SumCalculator:
        """
        Combine this dispersion calculator with another ASE calculator via
        :class:`~ase.calculators.mixing.SumCalculator`.
        """
        return SumCalculator([self, other])

    def _resolve_current_config(self) -> _ResolvedConfig:
        """Validate and resolve the current ``self.parameters`` dictionary."""
        with torch.inference_mode(False):
            params = self.parameters
            compute_stress = params.get("compute_stress", True)
            if not isinstance(compute_stress, bool):
                raise _InvalidTypeConfigurationError(
                    f"Expected 'compute_stress' to be a bool, got '{type(compute_stress).__name__}'."
                )

            device = _resolve_device(params.get("device", "cpu"))
            dtype = _resolve_dtype(params.get("dtype", torch.float64))

            raw_model = params.get("model")
            if raw_model is None:
                raw_model = self.default_parameters["model"]
            model_id = resolve_model_id(raw_model)
            meta = get_model_metadata(model_id)

            qmodel = _resolve_qmodel(
                params.get("qmodel", "eeq"), model_id, meta
            )
            charge, charges_key = _resolve_charge_options(
                params.get("charge"), params.get("charges_key"), meta
            )
            cutoff = _resolve_cutoff(params.get("realspace_cutoff"), meta)
            tweaks = _validate_tweaks_mapping(params.get("params_tweaks"))

            method = params.get("method")
            damping, d2_name, d3_spec, raw_2b_hint = _resolve_damping(
                model_id,
                meta,
                method if isinstance(method, str) and method.strip() else None,
                params.get("damping_hint"),
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
            )

    def set(self, **kwargs: Any) -> dict[str, Any]:
        """Set calculator parameters, defensively copying mutable mappings and resetting results."""
        unknown = sorted(set(kwargs) - set(self.default_parameters))
        if unknown:
            raise _InvalidTypeConfigurationError(
                f"Unexpected keyword argument(s) for DFTD calculator: {', '.join(unknown)}."
            )

        copied_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, Mapping):
                copied_kwargs[key] = copy.deepcopy(dict(value))
            else:
                copied_kwargs[key] = copy.deepcopy(value)

        # Force change detection if a mapping parameter is updated, even if the
        # user mutated an previously passed dict in place before calling set().
        prev_snapshot = {
            k: copy.deepcopy(self.parameters.get(k)) for k in copied_kwargs
        }
        changed_parameters = super().set(**copied_kwargs)
        for key, val in copied_kwargs.items():
            if key not in changed_parameters and prev_snapshot.get(key) != val:
                changed_parameters[key] = val

        # Always copy any mapping stored in self.parameters so external mutation
        # of user dicts cannot silently alter internal state.
        for key in ("params_tweaks", "damping_hint", "realspace_cutoff"):
            val = self.parameters.get(key)
            if isinstance(val, Mapping):
                self.parameters[key] = copy.deepcopy(dict(val))

        self._resolved_config = self._resolve_current_config()

        base_props = list(type(self).implemented_properties)
        if not self._resolved_config.compute_stress:
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
        changes to ``atoms.arrays[charges_key]`` when external charges are enabled.
        """
        changes = list(super().check_state(atoms, tol=tol))
        cfg = self._resolved_config
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
                    ref = (
                        self._last_external_charges
                        if self._last_external_charges is not None
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
                        ref_f64 = ref.astype(np.float64, copy=False)
                        if not (
                            np.all(np.isfinite(curr_f64))
                            and np.all(np.isfinite(ref_f64))
                            and np.allclose(
                                curr_f64, ref_f64, rtol=0.0, atol=tol
                            )
                        ):
                            changes.append(change_tag)

        return changes

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
            if (
                target_atoms is not None
                and bool(np.asarray(target_atoms.get_pbc(), dtype=bool).all())
            ):
                requested_props = list(self.implemented_properties)
            else:
                requested_props = [
                    p for p in self.implemented_properties if p != "stress"
                ]
        else:
            requested_props = list(properties)

        had_cached_results = bool(self.results) and not system_changes
        super().calculate(
            atoms=atoms,
            properties=requested_props,
            system_changes=list(system_changes),
        )

        if self.atoms is None:
            raise _ConfigurationError(
                "No ASE Atoms object is associated with the calculator."
            )

        if self._resolved_config is None:
            self._resolved_config = self._resolve_current_config()

        try:
            new_results = _calculate_energy_and_derivatives(
                self.atoms,
                self._resolved_config,
                set(requested_props),
            )
        except Exception:
            if not had_cached_results:
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
