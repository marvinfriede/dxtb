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
from typing import Any

import numpy as np
import pytest
import torch

ase = pytest.importorskip("ase")
from ase.atoms import Atoms  # type: ignore[import-untyped]
from ase.build import molecule  # type: ignore[import-untyped]
from ase.calculators.calculator import (  # type: ignore[import-untyped]
    Calculator,
    InputError,
    PropertyNotImplementedError,
)
from ase.stress import (  # type: ignore[import-untyped]
    full_3x3_to_voigt_6_stress,
)
from ase.units import Bohr, Hartree  # type: ignore[import-untyped]

from tad_mctc import Structure
from tad_mctc.data.structures import get_structure

from tad_dftd import dftd, param
from tad_dftd.ase import DFTD, DFTD3, DFTD4
from tad_dftd.damping import (
    FermiTwoBody,
    ModifiedZeroTwoBody,
    OptimizedPowerTwoBody,
    RationalTwoBody,
    ZeroThreeBodyD3,
    ZeroThreeBodyD4,
    ZeroTwoBody,
    damping_from_name,
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


@pytest.mark.parametrize(
    ("model", "method", "tweaks"),
    [
        ("d1", "BLYP", None),
        ("d2", "PBE", None),
        ("d3", "PBE", None),
        ("d3s", "BLYP", None),
        ("d4", "PBE", None),
        ("d4s", "PBE", None),
        (
            "d4srev",
            None,
            {"s6": 1.0, "s8": 0.9589, "a1": 0.3857, "a2": 4.8069, "s9": 1.0},
        ),
    ],
)
def test_all_seven_models_energy_and_forces_evaluation(
    model: str,
    method: str | None,
    tweaks: dict[str, float] | None,
) -> None:
    """Verify energy, per-atom energies, and forces for all 7 supported dispersion models."""
    atoms = molecule("H2O")
    orig_pos = atoms.get_positions().copy()

    atoms.calc = DFTD(method=method, model=model, params_tweaks=tweaks)
    energy = atoms.get_potential_energy()
    energies = atoms.get_potential_energies()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert energy < 0.0
    assert isinstance(energies, np.ndarray)
    assert energies.shape == (3,)
    assert pytest.approx(float(energies.sum()), rel=1e-14, abs=1e-14) == energy
    assert pytest.approx(atoms.calc.results["free_energy"], rel=1e-14, abs=1e-14) == energy

    assert isinstance(forces, np.ndarray)
    assert forces.shape == (3, 3)
    assert np.all(np.isfinite(forces))
    # Net translational force on an isolated molecule must vanish
    assert np.max(np.abs(forces.sum(axis=0))) < 1e-12
    # Input Atoms positions must not be mutated
    np.testing.assert_allclose(atoms.get_positions(), orig_pos, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("model", "method", "tweaks", "invalid_z"),
    [
        ("d1", "BLYP", None, 11),
        ("d2", "PBE", None, 87),
        ("d3", "PBE", None, 104),
        ("d3s", "BLYP", None, 95),
        ("d4", "PBE", None, 119),
    ],
)
def test_element_range_validation(
    model: str,
    method: str | None,
    tweaks: dict[str, float] | None,
    invalid_z: int,
) -> None:
    """Verify element-range limits are enforced via the catalogue without duplication."""
    atoms = Atoms(
        numbers=[1, invalid_z],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
    )
    atoms.calc = DFTD(method=method, model=model, params_tweaks=tweaks)
    with pytest.raises(ValueError, match="Unsupported|unsupported"):
        atoms.get_potential_energy()

    # Dummy Z=0 atom must also be rejected
    atoms_zero = Atoms(
        numbers=[0, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    atoms_zero.calc = DFTD(method="PBE", model="d4")
    with pytest.raises(ValueError, match="Unsupported atomic number"):
        atoms_zero.get_potential_energy()


@pytest.mark.parametrize(
    ("model", "method", "damping_hint", "tweaks"),
    [
        ("d1", "BLYP", None, None),
        ("d2", "PBE", None, None),
        ("d3", "PBE", None, None),
        ("d3", "PBE", {"2b": "zero", "3b": "none"}, None),
        ("d3", "PBE", {"2b": "rational", "3b": "zero-avg"}, None),
        ("d3s", "BLYP", None, None),
        ("d4", "PBE", None, None),
        ("d4", "PBE", {"3b": "none"}, None),
        ("d4s", "PBE", None, None),
        (
            "d4srev",
            None,
            None,
            {"s6": 1.0, "s8": 0.9589, "a1": 0.3857, "a2": 4.8069, "s9": 1.0},
        ),
    ],
)
def test_direct_numerical_parity_with_native_dftd(
    model: str,
    method: str | None,
    damping_hint: dict[str, str] | None,
    tweaks: dict[str, float] | None,
) -> None:
    """Verify exact numerical parity between tad_dftd.ase.DFTD and native tad_dftd.dftd()."""
    atoms = molecule("CH3CH2OH")
    calc = DFTD(
        method=method,
        model=model,
        damping_hint=damping_hint,
        params_tweaks=tweaks,
    )
    atoms.calc = calc
    e_ase = atoms.get_potential_energy()
    es_ase = atoms.get_potential_energies()
    f_ase = atoms.get_forces()

    cfg = calc._resolved_config
    assert cfg is not None
    pos_bohr = (
        torch.tensor(atoms.get_positions(), dtype=torch.float64) / Bohr
    ).requires_grad_(True)
    numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    structure = Structure(
        numbers=numbers,
        positions=pos_bohr,
        charge=torch.tensor(0.0, dtype=torch.float64),
    )
    extra_kwargs: dict[str, Any] = {}
    if cfg.qmodel is not None:
        extra_kwargs["qmodel"] = cfg.qmodel
    es_native = dftd(
        structure,
        cfg.param,
        model=cfg.model_id,
        damping=cfg.damping,
        cutoff=cfg.cutoff,
        **extra_kwargs,
    )
    e_native = es_native.sum()
    grad_native = torch.autograd.grad(e_native, pos_bohr)[0]
    f_native = (-grad_native.detach().numpy()) * (Hartree / Bohr)

    assert pytest.approx(float(e_native.item()) * Hartree, rel=1e-14, abs=1e-14) == e_ase
    np.testing.assert_allclose(
        es_ase, es_native.detach().numpy() * Hartree, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(f_ase, f_native, rtol=1e-13, atol=1e-13)


def test_fortran_dftd_ase_reference_parity() -> None:
    """Verify against independent Fortran grimme-lab/dftd python/dftd/test_ase.py reference values."""
    # 1. SCAN-D4 on methylenecyclopropane (energy + forces)
    atoms1 = molecule("methylenecyclopropane")
    atoms1.calc = DFTD4(method="SCAN")
    ref_forces1 = np.array(
        [
            [-7.90552684e-21, -1.15811595e-19, -2.80061133e-05],
            [+4.61502216e-20, +4.26735028e-04, +4.94269127e-04],
            [+2.28682834e-19, -4.26735028e-04, +4.94269127e-04],
            [+1.39725405e-20, -9.71924142e-20, -7.95205193e-04],
            [-1.08042249e-04, -8.23929519e-05, +2.31098749e-05],
            [+1.08042249e-04, -8.23929519e-05, +2.31098749e-05],
            [+1.08042249e-04, +8.23929519e-05, +2.31098749e-05],
            [-1.08042249e-04, +8.23929519e-05, +2.31098749e-05],
            [+1.07391220e-20, -4.98420762e-05, -1.28883224e-04],
            [+5.97028977e-21, +4.98420762e-05, -1.28883224e-04],
        ]
    )
    assert pytest.approx(-0.021665446836610567, abs=1e-9) == atoms1.get_potential_energy()
    np.testing.assert_allclose(atoms1.get_forces(), ref_forces1, atol=1e-6)

    # 2. PBE-D4S on bicyclobutane (energy + forces)
    atoms2 = molecule("bicyclobutane")
    atoms2.calc = DFTD(method="PBE", model="d4s")
    ref_forces2 = np.array(
        [
            [-1.05900228e-18, -1.47186433e-03, -2.08505399e-03],
            [-8.94454333e-19, 1.47186433e-03, -2.08505399e-03],
            [-2.48990627e-03, -1.06729670e-18, 2.15375860e-03],
            [2.48990627e-03, -1.91047493e-19, 2.15375860e-03],
            [-4.33024525e-22, -2.06112648e-03, -6.95481292e-05],
            [2.47579260e-21, -2.26182709e-03, -5.56758674e-04],
            [3.90654502e-20, 2.06112648e-03, -6.95481292e-05],
            [4.57489295e-20, 2.26182709e-03, -5.56758674e-04],
            [-1.57814347e-03, 6.44754156e-19, 5.57602199e-04],
            [1.57814347e-03, 5.62735538e-19, 5.57602199e-04],
        ]
    )
    assert pytest.approx(-0.16377494406788423, abs=1e-12) == atoms2.get_potential_energy()
    np.testing.assert_allclose(atoms2.get_forces(), ref_forces2, atol=1e-6)

    # 3. BLYP-D3S on OH2LiNi
    bohr_codata2018 = 0.529177210903
    atoms3 = Atoms(
        "OH2LiNi",
        positions=bohr_codata2018
        * np.array(
            [
                [+0.00000000000000, +0.00000000000000, -0.73578586109551],
                [+1.44183152868459, +0.00000000000000, +0.36789293054775],
                [-1.44183152868459, +0.00000000000000, +0.36789293054775],
                [+0.00000000000000, +0.00000000000000, +3.90000000000000],
                [+0.00000000000000, +5.00000000000000, +0.00000000000000],
            ]
        ),
    )
    atoms3.calc = DFTD(method="BLYP", model="d3s")
    assert pytest.approx(-1.0101127823439129e-02 * Hartree, abs=1e-9) == atoms3.get_potential_energy()
    np.testing.assert_allclose(np.sum(atoms3.get_forces(), axis=0), np.zeros(3), atol=1e-12)

    # 4. PBE-D4 without ATM (3b="none") on bicyclobutane
    atoms4 = molecule("bicyclobutane")
    atoms4.calc = DFTD(method="PBE", model="d4", damping_hint={"3b": "none"})
    assert pytest.approx(-0.15427642282371362, abs=1e-12) == atoms4.get_potential_energy()

    # 5. TPSS-D4 on C2H6CHOH (energy + forces)
    atoms5 = molecule("C2H6CHOH")
    atoms5.calc = DFTD4(method="tpss")
    ref_forces5 = np.array(
        [
            [-8.27697238e-04, -1.74116189e-02, +9.50315402e-05],
            [-3.00615825e-04, +3.51410800e-04, -3.12339518e-03],
            [-9.55691674e-04, -4.05325537e-03, +7.33934018e-05],
            [+2.70525425e-04, -1.91784287e-03, -6.13796425e-04],
            [+1.56444473e-02, +5.71643192e-03, +6.29706049e-04],
            [-1.43399413e-02, +7.36397630e-03, +7.87584027e-04],
            [+4.41551907e-03, +4.04705396e-04, +1.42098826e-03],
            [-4.17670039e-03, +1.57923335e-03, +1.60488604e-03],
            [+4.66065256e-03, +7.97764912e-05, -3.60249901e-05],
            [-4.95386767e-03, +1.38115911e-03, -1.63079386e-05],
            [+5.36717422e-03, +2.78165913e-03, -4.68647341e-04],
            [-4.80380448e-03, +3.72436466e-03, -3.53417437e-04],
        ]
    )
    assert pytest.approx(-0.24206732765720423, abs=1e-8) == atoms5.get_potential_energy()
    np.testing.assert_allclose(atoms5.get_forces(), ref_forces5, atol=1e-6)

    # 6. PBE-D4 on H2O
    atoms6 = molecule("H2O")
    atoms6.calc = DFTD4(method="pbe")
    assert pytest.approx(-0.005358475432239303, abs=1e-10) == atoms6.get_potential_energy()

    # 7. PBE-D3(BJ) on MB16-43-01 (from Fortran test_ase_dftd3_bj)
    atoms7 = Atoms(
        numbers=[11, 1, 8, 1, 9, 1, 1, 8, 7, 1, 1, 17, 5, 5, 7, 13],
        positions=(
            np.array(
                [
                    [-1.85528263484662, +3.58670515364616, -2.41763729306344],
                    [+4.40178023537845, +0.02338844412653, -4.95457749372945],
                    [-2.98706033463438, +4.76252065456814, +1.27043301573532],
                    [+0.79980886075526, +1.41103455609189, -5.04655321620119],
                    [-4.20647469409936, +1.84275767548460, +4.55038084858449],
                    [-3.54356121843970, -3.18835665176557, +1.46240021785588],
                    [+2.70032160109941, +1.06818452504054, -1.73234650374438],
                    [+3.73114088824361, -2.07001543363453, +2.23160937604731],
                    [-1.75306819230397, +0.35951417150421, +1.05323406177129],
                    [+5.41755788583825, -1.57881830078929, +1.75394002750038],
                    [-2.23462868255966, -2.13856505054269, +4.10922285746451],
                    [+1.01565866207568, -3.21952154552768, -3.36050963020778],
                    [+2.42119255723593, +0.26626435093114, -3.91862474360560],
                    [-3.02526098819107, +2.53667889095925, +2.31664984740423],
                    [-2.00438948664892, -2.29235136977220, +2.19782807357059],
                    [+1.12226554109716, -1.36942007032045, +0.48455055461782],
                ]
            )
            * Bohr
        ),
    )
    atoms7.calc = DFTD3(method="PBE")
    assert pytest.approx(-1.7882220155186028e-02 * Hartree, abs=1e-9) == atoms7.get_potential_energy()

    # 8. RPBE-D3(0) on MB16-43-09 (from Fortran test_ase_dftd3_zero)
    atoms8 = Atoms(
        numbers=[1, 1, 1, 1, 3, 1, 6, 5, 1, 1, 14, 1, 17, 9, 1, 5],
        positions=(
            np.array(
                [
                    [+3.97360649552839, +1.71723751297383, -0.51862929250676],
                    [+0.16903666216522, +1.73154352333176, -0.40099024352959],
                    [-3.94463844105182, -1.24346369608005, +0.09565841726334],
                    [+2.21647168119803, +4.10625979391554, +2.61391340002321],
                    [-0.04488993380842, -2.16288302687041, +4.48488595610432],
                    [+3.52287141817194, -0.90500888687059, -5.00916337263077],
                    [+1.95336082370762, -0.83849036872324, -3.65515970516029],
                    [+2.05706981818495, +1.70095588601056, -2.06303335904159],
                    [-6.40097100472159, -1.71072935987273, +3.14621771036234],
                    [+2.04751538182937, -2.55691868000982, -2.49926722310562],
                    [+2.03251078714394, +1.35094356516468, +2.02150308748654],
                    [+0.20477572129201, -0.93291693232462, -4.76431390827476],
                    [-2.67673272939098, +1.40764602033672, +4.10347165469140],
                    [-2.75901984658887, -3.73954809548334, +3.19373273207227],
                    [+1.96938102642596, +3.74070925169244, -3.03185101883736],
                    [-4.32034786008576, -1.66533650719069, +2.28302516508337],
                ]
            )
            * Bohr
        ),
    )
    atoms8.calc = DFTD3(method="RPBE", damping_hint={"2b": "zero", "3b": "none"})
    assert pytest.approx(-2.0178760785797962e-02 * Hartree, abs=1e-9) == atoms8.get_potential_energy()


@pytest.mark.parametrize(
    ("model", "method", "damping_hint"),
    [
        ("d3", "PBE", {"2b": "rational", "3b": "none"}),
        ("d3", "PBE", {"2b": "rational", "3b": "zero-avg"}),
        ("d3", "PBE", {"2b": "zero", "3b": "none"}),
        ("d3", "PBE", {"2b": "mzero", "3b": "none"}),
        ("d3", "PBE", {"2b": "optimizedpower", "3b": "none"}),
        ("d4", "PBE", {"2b": "rational", "3b": "zero-avg"}),
        ("d4", "PBE", {"2b": "rational", "3b": "none"}),
    ],
)
def test_molecular_forces_central_finite_differences(
    model: str,
    method: str,
    damping_hint: dict[str, str],
) -> None:
    """Compare analytical autograd forces against central finite differences of ASE energy."""
    atoms = molecule("H2O")
    # Perturb slightly to break exact symmetry so all force components are non-trivial
    pos0 = atoms.get_positions().copy()
    pos0[0] += np.array([0.02, -0.01, 0.03])
    pos0[1] += np.array([-0.01, 0.03, -0.02])
    atoms.set_positions(pos0)

    atoms.calc = DFTD(
        method=method,
        model=model,
        damping_hint=damping_hint,
        dtype=torch.float64,
    )
    f_analytic = atoms.get_forces().copy()

    eps = 1.0e-6
    f_num = np.zeros_like(f_analytic)
    for idx in range(len(atoms)):
        for alpha in range(3):
            pos_plus = pos0.copy()
            pos_plus[idx, alpha] += eps
            atoms.set_positions(pos_plus)
            e_plus = atoms.get_potential_energy()

            pos_minus = pos0.copy()
            pos_minus[idx, alpha] -= eps
            atoms.set_positions(pos_minus)
            e_minus = atoms.get_potential_energy()

            f_num[idx, alpha] = -(e_plus - e_minus) / (2.0 * eps)

    np.testing.assert_allclose(f_analytic, f_num, rtol=1e-7, atol=1e-8)


def test_dtype_and_gradient_contexts() -> None:
    """Verify float64/float32 execution and safe force evaluation under no_grad and inference_mode."""
    default_dtype_before = torch.get_default_dtype()

    atoms64 = molecule("H2O")
    atoms64.calc = DFTD(method="PBE", model="d4", dtype=torch.float64)
    e64 = atoms64.get_potential_energy()
    f64 = atoms64.get_forces()

    atoms32 = molecule("H2O")
    atoms32.calc = DFTD(method="PBE", model="d4", dtype=torch.float32)
    e32 = atoms32.get_potential_energy()
    f32 = atoms32.get_forces()

    assert pytest.approx(e64, rel=1e-5, abs=1e-5) == e32
    np.testing.assert_allclose(f32, f64, rtol=1e-4, atol=1e-5)
    assert torch.get_default_dtype() == default_dtype_before

    # Test under torch.no_grad()
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        atoms_ng = molecule("H2O")
        atoms_ng.calc = DFTD(method="PBE", model="d4")
        e_ng = atoms_ng.get_potential_energy()
        f_ng = atoms_ng.get_forces()
        assert not torch.is_grad_enabled()
    assert pytest.approx(e64, rel=1e-14, abs=1e-14) == e_ng
    np.testing.assert_allclose(f_ng, f64, rtol=1e-13, atol=1e-13)

    # Test under torch.inference_mode()
    with torch.inference_mode():
        assert torch.is_inference_mode_enabled()
        atoms_im = molecule("H2O")
        atoms_im.calc = DFTD(method="PBE", model="d4")
        e_im = atoms_im.get_potential_energy()
        f_im = atoms_im.get_forces()
        assert torch.is_inference_mode_enabled()
    assert pytest.approx(e64, rel=1e-14, abs=1e-14) == e_im
    np.testing.assert_allclose(f_im, f64, rtol=1e-13, atol=1e-13)


def test_periodic_fortran_reference_parity_all_models() -> None:
    """Verify 1D/2D/3D periodic energies, forces, and 3D Voigt stress against Fortran libdftd.so references."""
    from .test_periodic_and_transforms import (
        _FORTRAN_PERIODIC_REFS,
        _fortran_ref_cell,
    )

    q_zero2 = np.zeros(2, dtype=np.float64)
    for pbc_key, pbc_flags in (
        ("1d", (True, False, False)),
        ("2d", (True, True, False)),
        ("3d", (True, True, True)),
    ):
        cell, cutoffs, params = _fortran_ref_cell(pbc_flags)
        assert cell.lattice is not None
        pos_ang = cell.positions.numpy() * Bohr
        lat_ang = cell.lattice.numpy() * Bohr
        nums = cell.numbers.numpy()
        vol_ang3 = abs(float(np.linalg.det(lat_ang)))

        for model_name in ("d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"):
            ref = _FORTRAN_PERIODIC_REFS[pbc_key][model_name]
            cut = cutoffs[model_name]
            p = params[model_name]

            atoms = Atoms(
                numbers=nums,
                positions=pos_ang,
                cell=lat_ang,
                pbc=pbc_flags,
            )
            charges_key: str | None = None
            if model_name in ("d4", "d4s", "d4srev") and pbc_key != "3d":
                atoms.set_array("q_ext", q_zero2.copy())
                charges_key = "q_ext"

            cut_ang = {
                "disp2": float(cut.disp2) * Bohr,
                "disp3": float(cut.disp3) * Bohr,
                "cn": float(cut.cn) * Bohr,
                "cn_eeq": float(cut.cn_eeq) * Bohr,
            }
            atoms.calc = DFTD(
                model=model_name,
                params_tweaks=p,
                realspace_cutoff=cut_ang,
                charges_key=charges_key,
            )

            e_ase = atoms.get_potential_energy()
            f_ase = atoms.get_forces()

            ref_e_ev = float(np.asarray(ref["energy"])) * Hartree
            ref_f0_ev_ang = -np.asarray(ref["grad0"], dtype=np.float64) * (
                Hartree / Bohr
            )
            assert pytest.approx(ref_e_ev, rel=1e-12, abs=1e-12) == e_ase
            np.testing.assert_allclose(
                f_ase[0], ref_f0_ev_ang, rtol=1e-11, atol=1e-11
            )

            if pbc_key == "3d":
                s_ase = atoms.get_stress()
                ref_sigma_3x3 = np.asarray(ref["sigma"], dtype=np.float64) * (
                    Hartree / vol_ang3
                )
                ref_voigt = np.asarray(
                    full_3x3_to_voigt_6_stress(ref_sigma_3x3), dtype=np.float64
                )
                assert s_ase.shape == (6,)
                np.testing.assert_allclose(
                    s_ase, ref_voigt, rtol=1e-11, atol=1e-11
                )

    # Also verify X23 3D molecular crystal references via ASE calculator
    cut_x23_d12 = {"cn": 30.0 * Bohr, "disp2": 60.0 * Bohr, "disp3": 15.0 * Bohr}
    cut_x23_d3 = {"cn": 40.0 * Bohr, "disp2": 60.0 * Bohr, "disp3": 15.0 * Bohr}

    for dataset_name, model_name, method_name, hint, cut_dict, ref_eh in [
        ("ammonia", "d1", "PBE", None, cut_x23_d12, -2.1111064217077948e-02),
        ("ammonia", "d2", "PBE", None, cut_x23_d12, -2.9729958659375735e-02),
        ("acetic", "d3", "PBE", None, cut_x23_d3, -0.066727625533349),
        ("acetic", "d3s", "PBE", None, cut_x23_d3, -0.066728412990681),
        (
            "benzene",
            "d3",
            "BLYP",
            {"2b": "zero", "3b": "none"},
            cut_x23_d3,
            -0.16644876006925308,
        ),
    ]:
        s_x23 = get_structure("x23", dataset_name, dtype=torch.float64)
        assert s_x23.lattice is not None
        atoms_x23 = Atoms(
            numbers=s_x23.numbers.numpy(),
            positions=s_x23.positions.numpy() * Bohr,
            cell=s_x23.lattice.numpy() * Bohr,
            pbc=[True, True, True],
        )
        atoms_x23.calc = DFTD(
            method=method_name,
            model=model_name,
            damping_hint=hint,
            realspace_cutoff=cut_dict,
        )
        assert (
            pytest.approx(ref_eh * Hartree, rel=1e-11, abs=1e-11)
            == atoms_x23.get_potential_energy()
        )


@pytest.mark.parametrize("model_name", ["d3", "d4"])
def test_periodic_stress_central_finite_differences_and_virial_identity(
    model_name: str,
) -> None:
    """Compare analytical ASE 3D periodic Voigt stress against central finite differences under homogeneous strain."""
    numbers = [14, 8, 6, 1]
    positions_ang = (
        np.array(
            [
                [0.10, 0.20, 0.30],
                [2.40, 2.10, 1.90],
                [1.30, 3.60, 0.80],
                [3.70, 1.20, 3.80],
            ],
            dtype=np.float64,
        )
        * Bohr
    )
    cell_ang = (
        np.array(
            [
                [7.5, 0.2, 0.0],
                [0.1, 7.8, 0.1],
                [0.0, 0.1, 8.0],
            ],
            dtype=np.float64,
        )
        * Bohr
    )
    cut_ang = {
        "cn": 9.0 * Bohr,
        "disp2": 10.0 * Bohr,
        "disp3": 8.0 * Bohr,
        "width2": 2.0 * Bohr,
        "width3": 2.0 * Bohr,
        "cn_eeq": 9.0 * Bohr,
    }
    q_fixed = np.array([0.20, -0.25, 0.15, -0.10], dtype=np.float64)

    atoms = Atoms(
        numbers=numbers,
        positions=positions_ang.copy(),
        cell=cell_ang.copy(),
        pbc=[True, True, True],
    )
    charges_key: str | None = None
    if model_name == "d4":
        atoms.set_array("q_ext", q_fixed.copy())
        charges_key = "q_ext"

    calc = DFTD(
        method="PBE",
        model=model_name,
        damping_hint={"2b": "rational", "3b": "zero-avg"},
        realspace_cutoff=cut_ang,
        charges_key=charges_key,
    )
    atoms.calc = calc
    stress_voigt = atoms.get_stress().copy()
    forces_ase = atoms.get_forces().copy()
    assert stress_voigt.shape == (6,)

    # Verify agreement with (dE/dr)^T r + (dE/dL)^T L from native tad_dftd
    cfg = calc._resolved_config
    assert cfg is not None
    pos_t = (
        torch.tensor(positions_ang, dtype=torch.float64) / Bohr
    ).requires_grad_(True)
    lat_t = (
        torch.tensor(cell_ang, dtype=torch.float64) / Bohr
    ).requires_grad_(True)
    s_native = Structure(
        numbers=torch.tensor(numbers, dtype=torch.long),
        positions=pos_t,
        lattice=lat_t,
        periodic=torch.tensor([True, True, True]),
    )
    extra_kw: dict[str, Any] = {}
    if model_name == "d4":
        extra_kw["q"] = torch.tensor(q_fixed, dtype=torch.float64)
    e_nat = dftd(
        s_native,
        cfg.param,
        model=cfg.model_id,
        damping=cfg.damping,
        cutoff=cfg.cutoff,
        **extra_kw,
    ).sum()
    g_pos, g_lat = torch.autograd.grad(e_nat, (pos_t, lat_t))
    virial_ident = (g_pos.T @ pos_t.detach() + g_lat.T @ lat_t.detach()).numpy()
    vol_ang3 = abs(float(np.linalg.det(cell_ang)))
    voigt_ident = np.asarray(
        full_3x3_to_voigt_6_stress(virial_ident * (Hartree / vol_ang3)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(stress_voigt, voigt_ident, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        forces_ase, -g_pos.detach().numpy() * (Hartree / Bohr), rtol=1e-13, atol=1e-13
    )

    # Central finite differences over all 3x3 strain components (diagonal and off-diagonal)
    eps = 1.0e-6
    stress_3x3_fd = np.zeros((3, 3), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    for alpha in range(3):
        for beta in range(3):
            scale_p = eye3.copy()
            scale_p[alpha, beta] += eps
            atoms.set_positions(positions_ang @ scale_p.T)
            atoms.set_cell(cell_ang @ scale_p.T, scale_atoms=False)
            e_p = atoms.get_potential_energy()

            scale_m = eye3.copy()
            scale_m[alpha, beta] -= eps
            atoms.set_positions(positions_ang @ scale_m.T)
            atoms.set_cell(cell_ang @ scale_m.T, scale_atoms=False)
            e_m = atoms.get_potential_energy()

            stress_3x3_fd[alpha, beta] = (e_p - e_m) / (2.0 * eps * vol_ang3)

    stress_voigt_fd = np.asarray(
        full_3x3_to_voigt_6_stress(stress_3x3_fd), dtype=np.float64
    )
    # Check Voigt ordering explicitly: xx, yy, zz, yz, xz, xy
    assert pytest.approx(stress_3x3_fd[0, 0], rel=1e-14) == stress_voigt_fd[0]
    assert pytest.approx(stress_3x3_fd[1, 1], rel=1e-14) == stress_voigt_fd[1]
    assert pytest.approx(stress_3x3_fd[2, 2], rel=1e-14) == stress_voigt_fd[2]
    assert pytest.approx(0.5 * (stress_3x3_fd[1, 2] + stress_3x3_fd[2, 1]), rel=1e-14) == stress_voigt_fd[3]
    assert pytest.approx(0.5 * (stress_3x3_fd[0, 2] + stress_3x3_fd[2, 0]), rel=1e-14) == stress_voigt_fd[4]
    assert pytest.approx(0.5 * (stress_3x3_fd[0, 1] + stress_3x3_fd[1, 0]), rel=1e-14) == stress_voigt_fd[5]

    np.testing.assert_allclose(stress_voigt, stress_voigt_fd, rtol=1e-6, atol=1e-8)


def test_stress_restrictions_and_errors() -> None:
    """Verify stress raises a clear error for 0D, 1D, 2D, degenerate 3D cells, and compute_stress=False."""
    # 0D molecular system
    mol = molecule("H2O")
    mol.calc = DFTD(method="PBE", model="d4")
    assert mol.get_potential_energy() < 0.0
    with pytest.raises((PropertyNotImplementedError, ValueError), match="3D periodic"):
        mol.get_stress()

    # 1D periodic system
    chain = Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[3.5, 10.0, 10.0],
        pbc=[True, False, False],
    )
    chain.calc = DFTD(method="PBE", model="d3")
    assert chain.get_potential_energy() < 0.0
    assert chain.get_forces().shape == (2, 3)
    with pytest.raises((PropertyNotImplementedError, ValueError), match="3D periodic"):
        chain.get_stress()

    # 2D periodic slab
    slab = Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 0.0]],
        cell=[3.5, 3.5, 10.0],
        pbc=[True, True, False],
    )
    slab.calc = DFTD(method="PBE", model="d3")
    assert slab.get_potential_energy() < 0.0
    assert slab.get_forces().shape == (2, 3)
    with pytest.raises((PropertyNotImplementedError, ValueError), match="3D periodic"):
        slab.get_stress()

    # Degenerate 3D periodic cell (zero volume)
    degen = Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[[3.5, 0.0, 0.0], [3.5, 0.0, 0.0], [0.0, 0.0, 3.5]],
        pbc=[True, True, True],
    )
    degen.calc = DFTD(method="PBE", model="d3")
    with pytest.raises((PropertyNotImplementedError, ValueError), match="degenerate|zero volume"):
        degen.get_stress()

    # 3D periodic cell with compute_stress=False
    cryst = Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]],
        cell=[3.5, 3.5, 3.5],
        pbc=[True, True, True],
    )
    cryst.calc = DFTD(method="PBE", model="d3", compute_stress=False)
    assert "stress" not in cryst.calc.implemented_properties
    with pytest.raises(PropertyNotImplementedError):
        cryst.get_stress()


def test_charge_workflows_and_validation() -> None:
    """Verify automatic EEQ total charge, external charges_key, invalid arrays, and 1D/2D EEQ restrictions."""
    # 1. Molecular automatic EEQ: neutral vs nonzero charge vs initial_charges
    atoms_neutral = molecule("H2O")
    atoms_neutral.calc = DFTD(method="PBE", model="d4")
    e_neutral = atoms_neutral.get_potential_energy()

    atoms_cation = molecule("H2O")
    atoms_cation.calc = DFTD(method="PBE", model="d4", charge=1.0)
    e_cation = atoms_cation.get_potential_energy()
    assert abs(e_cation - e_neutral) > 1e-4

    # Initial charges sum to +1.0 when charge=None
    atoms_initq = molecule("H2O")
    atoms_initq.set_initial_charges([0.4, 0.3, 0.3])
    atoms_initq.calc = DFTD(method="PBE", model="d4")
    assert pytest.approx(e_cation, rel=1e-14, abs=1e-14) == atoms_initq.get_potential_energy()

    # Explicit charge=0.0 overrides nonzero initial_charges
    atoms_override = molecule("H2O")
    atoms_override.set_initial_charges([0.4, 0.3, 0.3])
    atoms_override.calc = DFTD(method="PBE", model="d4", charge=0.0)
    assert pytest.approx(e_neutral, rel=1e-14, abs=1e-14) == atoms_override.get_potential_energy()

    # 2. 3D periodic automatic EEQ
    cryst = Atoms(
        numbers=[14, 6],
        positions=[[0.0, 0.0, 0.0], [1.9, 1.9, 1.9]],
        cell=[[4.0, 0.1, 0.0], [0.1, 4.1, 0.0], [0.0, 0.1, 4.2]],
        pbc=[True, True, True],
    )
    cryst.calc = DFTD(
        method="PBE",
        model="d4",
        realspace_cutoff={"disp2": 10.0, "disp3": 8.0, "cn": 8.0, "cn_eeq": 8.0},
    )
    assert cryst.get_potential_energy() < 0.0
    assert cryst.get_forces().shape == (2, 3)
    assert cryst.get_stress().shape == (6,)

    # 3. External partial charges via charges_key on 0D, 1D, 2D, and 3D
    q_custom = np.array([-0.55, 0.30, 0.25], dtype=np.float64)
    atoms_ext = molecule("H2O")
    atoms_ext.set_array("dftd_partial_charges", q_custom)
    atoms_ext.calc = DFTD(
        method="PBE",
        model="d4",
        charges_key="dftd_partial_charges",
    )
    e_ext = atoms_ext.get_potential_energy()
    f_ext = atoms_ext.get_forces()

    # Compare directly against native tad_dftd.dftd(..., q=q_custom)
    cfg = atoms_ext.calc._resolved_config
    assert cfg is not None
    pos_t = (
        torch.tensor(atoms_ext.get_positions(), dtype=torch.float64) / Bohr
    ).requires_grad_(True)
    s_ext = Structure(
        numbers=torch.tensor(atoms_ext.get_atomic_numbers(), dtype=torch.long),
        positions=pos_t,
        charge=torch.tensor(0.0, dtype=torch.float64),
    )
    e_nat_ext = dftd(
        s_ext,
        cfg.param,
        model="d4",
        damping=cfg.damping,
        cutoff=cfg.cutoff,
        q=torch.tensor(q_custom, dtype=torch.float64),
    ).sum()
    g_nat_ext = torch.autograd.grad(e_nat_ext, pos_t)[0]
    assert pytest.approx(float(e_nat_ext.item()) * Hartree, rel=1e-14, abs=1e-14) == e_ext
    np.testing.assert_allclose(
        f_ext, -g_nat_ext.detach().numpy() * (Hartree / Bohr), rtol=1e-13, atol=1e-13
    )

    # 4. Automatic 1D and 2D periodic EEQ is rejected, while external charges_key works
    for pbc_partial in ([True, False, False], [True, True, False]):
        partial_atoms = Atoms(
            numbers=[14, 6],
            positions=[[0.0, 0.0, 0.0], [1.9, 1.9, 0.5]],
            cell=[4.5, 4.5, 10.0],
            pbc=pbc_partial,
        )
        partial_atoms.calc = DFTD(
            method="PBE",
            model="d4",
            realspace_cutoff={"disp2": 10.0, "disp3": 8.0, "cn": 8.0, "cn_eeq": 8.0},
        )
        with pytest.raises((NotImplementedError, ValueError), match="1D|2D|3D"):
            partial_atoms.get_potential_energy()

        # Supplying external charges makes 1D/2D D4 work
        partial_atoms.set_array("q_ext", np.array([0.15, -0.15], dtype=np.float64))
        partial_atoms.calc = DFTD(
            method="PBE",
            model="d4",
            charges_key="q_ext",
            realspace_cutoff={"disp2": 10.0, "disp3": 8.0, "cn": 8.0, "cn_eeq": 8.0},
        )
        assert partial_atoms.get_potential_energy() < 0.0
        assert partial_atoms.get_forces().shape == (2, 3)

    # 5. Invalid external charge arrays raise clear ValueErrors
    atoms_missing = molecule("H2O")
    atoms_missing.calc = DFTD(method="PBE", model="d4", charges_key="missing_q")
    with pytest.raises(ValueError, match="not found in 'atoms.arrays'"):
        atoms_missing.get_potential_energy()

    atoms_badshape = molecule("H2O")
    atoms_badshape.set_array("bad_q", np.zeros((3, 2), dtype=np.float64))
    atoms_badshape.calc = DFTD(method="PBE", model="d4", charges_key="bad_q")
    with pytest.raises(ValueError, match="1D shape"):
        atoms_badshape.get_potential_energy()

    atoms_baddtype = molecule("H2O")
    atoms_baddtype.set_array("bool_q", np.array([True, False, True]))
    atoms_baddtype.calc = DFTD(method="PBE", model="d4", charges_key="bool_q")
    with pytest.raises(ValueError, match="real numeric dtype"):
        atoms_baddtype.get_potential_energy()

    atoms_nan = molecule("H2O")
    atoms_nan.set_array("nan_q", np.array([0.0, np.nan, 0.0], dtype=np.float64))
    atoms_nan.calc = DFTD(method="PBE", model="d4", charges_key="nan_q")
    with pytest.raises(ValueError, match="non-finite"):
        atoms_nan.get_potential_energy()


