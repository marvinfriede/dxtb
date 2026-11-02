# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
"""
Tests for the native ASE calculator interface ``tad_dftd.ase``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ase = pytest.importorskip("ase")
from ase.calculators.calculator import (  # type: ignore[import-untyped]
    Calculator,
    InputError,
)
from ase.units import Bohr  # type: ignore[import-untyped]

from tad_dftd.ase import DFTD, DFTD3, DFTD4
from tad_dftd.damping import (
    FermiTwoBody,
    ModifiedZeroTwoBody,
    OptimizedPowerTwoBody,
    RationalTwoBody,
    ZeroThreeBodyD3,
    ZeroThreeBodyD4,
    ZeroTwoBody,
)
from tad_dftd.model.catalogue import DispersionModelId

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"


def test_public_interface_and_convenience_subclasses() -> None:
    """Verify DFTD, DFTD3, and DFTD4 subclass ASE Calculator and set default models."""
    assert issubclass(DFTD, Calculator)
    assert issubclass(DFTD3, DFTD)
    assert issubclass(DFTD4, DFTD)

    calc_default = DFTD(method="PBE")
    assert calc_default.parameters["model"] == "d4"
    assert calc_default.parameters["device"] == "cpu"
    assert calc_default.parameters["dtype"] == torch.float64
    assert calc_default.parameters["compute_stress"] is True
    assert "stress" in calc_default.implemented_properties

    calc_d4 = DFTD4(method="PBE")
    assert calc_d4.parameters["model"] == "d4"
    assert calc_d4._resolved_config is not None
    assert calc_d4._resolved_config.model_id == DispersionModelId.D4

    calc_d3 = DFTD3(method="PBE")
    assert calc_d3.parameters["model"] == "d3"
    assert calc_d3._resolved_config is not None
    assert calc_d3._resolved_config.model_id == DispersionModelId.D3


def test_parameter_modes() -> None:
    """Verify parameter resolution modes 1 (method), 2 (tweaks), 3 (method+tweaks), and 4 (damping_hint)."""
    # Mode 1: Functional name
    calc1 = DFTD(method="PBE", model="d4")
    cfg1 = calc1._resolved_config
    assert cfg1 is not None
    assert isinstance(cfg1.damping.two, RationalTwoBody)
    assert isinstance(cfg1.damping.three, ZeroThreeBodyD4)
    assert cfg1.param.s6 is not None and pytest.approx(float(cfg1.param.s6)) == 1.0
    assert cfg1.param.s9 is not None and pytest.approx(float(cfg1.param.s9)) == 1.0

    # Mode 2: Explicit parameters without method
    calc2 = DFTD(
        model="d3",
        params_tweaks={
            "s6": 1.0,
            "s8": 0.7875,
            "a1": 0.4289,
            "a2": 4.4407,
        },
    )
    cfg2 = calc2._resolved_config
    assert cfg2 is not None
    assert isinstance(cfg2.damping.two, RationalTwoBody)
    assert cfg2.damping.three is None
    assert cfg2.param.s8 is not None and pytest.approx(float(cfg2.param.s8)) == 0.7875

    # Mode 3: Functional parameters with explicit overrides (including s9=0.0)
    calc3 = DFTD(
        method="PBE",
        model="d4",
        params_tweaks={"s9": 0.0, "s8": 0.5},
    )
    cfg3 = calc3._resolved_config
    assert cfg3 is not None
    assert cfg3.damping.three is None
    assert cfg3.param.s9 is not None and float(cfg3.param.s9) == 0.0
    assert cfg3.param.s8 is not None and pytest.approx(float(cfg3.param.s8)) == 0.5

    # Mode 4: Explicit damping choice
    calc4 = DFTD(
        method="PBE",
        model="d3",
        damping_hint={"2b": "zero", "3b": "none"},
    )
    cfg4 = calc4._resolved_config
    assert cfg4 is not None
    assert isinstance(cfg4.damping.two, ZeroTwoBody)
    assert cfg4.damping.three is None
    assert cfg4.param.rs6 is not None and pytest.approx(float(cfg4.param.rs6)) == 1.217

    # Mode 4b: D3 with explicit 3B zero-avg enabled
    calc4b = DFTD(
        method="PBE",
        model="d3",
        damping_hint={"2b": "rational", "3b": "zero-avg"},
    )
    cfg4b = calc4b._resolved_config
    assert cfg4b is not None
    assert isinstance(cfg4b.damping.two, RationalTwoBody)
    assert isinstance(cfg4b.damping.three, ZeroThreeBodyD3)


@pytest.mark.parametrize(
    ("model", "method", "tweaks", "expected_2b"),
    [
        ("d1", "BLYP", None, FermiTwoBody),
        ("d2", "PBE", None, FermiTwoBody),
        ("d3", "PBE", None, RationalTwoBody),
        ("d3s", "BLYP", None, RationalTwoBody),
        ("d4", "PBE", None, RationalTwoBody),
        ("d4s", "PBE", None, RationalTwoBody),
        (
            "d4srev",
            None,
            {"s6": 1.0, "s8": 0.9, "a1": 0.4, "a2": 4.5, "s9": 1.0},
            RationalTwoBody,
        ),
    ],
)
def test_all_seven_supported_models_configuration(
    model: str,
    method: str | None,
    tweaks: dict[str, float] | None,
    expected_2b: type[object],
) -> None:
    """Verify all 7 supported dispersion models resolve configuration cleanly."""
    calc = DFTD(method=method, model=model, params_tweaks=tweaks)
    assert calc._resolved_config is not None
    assert isinstance(calc._resolved_config.damping.two, expected_2b)


def test_damping_aliases_and_cutoff_conversion() -> None:
    """Verify damping name aliases and Angstrom-to-Bohr cutoff conversion."""
    calc_mzero = DFTD(
        method="PBE",
        model="d3",
        damping_hint={"2b": "mzero", "3b": "none"},
    )
    assert calc_mzero._resolved_config is not None
    assert isinstance(calc_mzero._resolved_config.damping.two, ModifiedZeroTwoBody)

    calc_op = DFTD(
        method="PBE",
        model="d3",
        damping_hint={"2b": "optimizedpower", "3b": "none"},
    )
    assert calc_op._resolved_config is not None
    assert isinstance(calc_op._resolved_config.damping.two, OptimizedPowerTwoBody)

    calc_cut = DFTD(
        method="PBE",
        model="d4",
        realspace_cutoff={"disp2": 25.0, "disp3": 15.0, "cn": 10.0, "cn_eeq": 12.0},
    )
    cutoff = calc_cut._resolved_config.cutoff  # type: ignore[union-attr]
    assert pytest.approx(float(cutoff.disp2)) == 25.0 / Bohr
    assert pytest.approx(float(cutoff.disp3)) == 15.0 / Bohr
    assert pytest.approx(float(cutoff.cn)) == 10.0 / Bohr
    assert pytest.approx(float(cutoff.cn_eeq)) == 12.0 / Bohr


def test_configuration_validation_errors() -> None:
    """Verify clear exceptions for invalid methods, parameters, dampings, models, and devices."""
    # Missing both method and params_tweaks
    with pytest.raises((InputError, ValueError), match="Either a functional 'method'"):
        DFTD(model="d4")

    # Unknown functional method
    with pytest.raises((InputError, ValueError), match="(?i)unknown|no .* parameters"):
        DFTD(method="not-a-real-functional", model="d4")

    # Missing required parameters in Mode 2
    with pytest.raises((InputError, ValueError), match="requires the parameter"):
        DFTD(model="d3", params_tweaks={"s6": 1.0, "s8": 0.5})

    # Unsupported parameter name in params_tweaks
    with pytest.raises((InputError, TypeError, ValueError), match="Unsupported damping parameter"):
        DFTD(method="PBE", model="d4", params_tweaks={"not_a_param": 1.0})

    # Incompatible model and damping combination (e.g. D1 with rational)
    with pytest.raises((InputError, ValueError), match="not supported"):
        DFTD(method="BLYP", model="d1", damping_hint={"2b": "rational"})

    # Incompatible model and damping combination (e.g. D4 with cso)
    with pytest.raises((InputError, ValueError), match="not supported"):
        DFTD(
            model="d4",
            damping_hint={"2b": "cso"},
            params_tweaks={"s6": 1.0, "a1": 1.0, "a2": 2.5, "a3": 0.0, "a4": 6.0},
        )

    # D4 with method and non-rational damping that has no D4 functional table
    with pytest.raises((InputError, ValueError), match="No D4 functional parameters"):
        DFTD(method="PBE", model="d4", damping_hint={"2b": "zero"})

    # D4Srev with functional name (no published method table)
    with pytest.raises((InputError, ValueError), match="d4srev"):
        DFTD(method="PBE", model="d4srev")

    # Excluded D5 model
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        DFTD(method="PBE", model="d5")

    # Excluded EEQBC charge model
    with pytest.raises(NotImplementedError, match="not supported in this release"):
        DFTD(method="PBE", model="d4", qmodel="eeqbc")

    with pytest.raises(NotImplementedError, match="not supported in this release"):
        DFTD(method="PBE", model="d4", qmodel="eeqbc2025")

    # Unknown dispersion model
    with pytest.raises(ValueError, match="Unknown dispersion model"):
        DFTD(method="PBE", model="d99")

    # charges_key on charge-independent model (D3)
    with pytest.raises((InputError, ValueError), match="does not use atomic partial charges"):
        DFTD(method="PBE", model="d3", charges_key="q")

    # Unknown cutoff key
    with pytest.raises((InputError, ValueError), match="Unknown key"):
        DFTD(method="PBE", model="d4", realspace_cutoff={"bad_cutoff": 10.0})

    # Non-positive cutoff value
    with pytest.raises((InputError, ValueError), match="strictly positive"):
        DFTD(method="PBE", model="d4", realspace_cutoff={"disp2": -5.0})

    # Invalid device or dtype
    with pytest.raises((InputError, ValueError), match="Invalid PyTorch device|Unsupported PyTorch device"):
        DFTD(method="PBE", model="d4", device="opencl:0")

    with pytest.raises((InputError, ValueError), match="Unsupported dtype|Unknown dtype"):
        DFTD(method="PBE", model="d4", dtype=torch.int64)

    # Unexpected keyword argument in set() / __init__()
    with pytest.raises((InputError, TypeError, ValueError), match="Unexpected keyword argument"):
        DFTD(method="PBE", model="d4", cache_api=True)


def test_optional_ase_dependency_isolation() -> None:
    """Verify tad_dftd imports and runs without ASE, while tad_dftd.ase raises a clear ModuleNotFoundError."""
    code = """
import sys
import torch
import tad_dftd
from tad_dftd import dftd

# Ensure 'ase' was not imported eagerly and is blocked
assert "ase" not in sys.modules
assert "tad_dftd.ase" not in sys.modules

# Run a native tad_dftd calculation without ASE present
from tad_mctc import Structure
s = Structure(
    numbers=torch.tensor([8, 1, 1]),
    positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]], dtype=torch.float64),
)
e = dftd(s, tad_dftd.param.get_params("PBE", model="d4"), model="d4")
assert e.shape == (3,)

# Now verify importing tad_dftd.ase raises a clear ModuleNotFoundError
try:
    import tad_dftd.ase
except ModuleNotFoundError as exc:
    assert "requires ASE" in str(exc)
else:
    raise AssertionError("Expected ModuleNotFoundError when importing tad_dftd.ase without ASE")
"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "PYTHONPATH"
    }
    env["PYTHONPATH"] = str(SRC_DIR)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path = [p for p in sys.path if 'ase' not in p.lower()]; sys.path.insert(0, {str(SRC_DIR)!r}); {code}",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
