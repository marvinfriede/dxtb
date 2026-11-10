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
Live reference validation against ``tblite.interface.Calculator`` (tblite 0.8.0).

Executes upstream tblite dynamically without reading static reference files.
Covers:
- WP0 live reference test matrix (GFN1 H2O, CH4; GFN2 H2O, non-planar and
  mixed-element molecules).
- WP3 live reference validation for restored GFN1 4sp (Fe(CO)5) and 5sp (RbH, SrH2)
  basis expansions.
- WP1.6 C live tblite comparison for GFN2-xTB on original and rotated
  geometries (total energy, Cartesian gradient, atomic charges, and dipole).
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from ..tblite import (
    Options,
    Structure,
    get_tblite_charges,
    get_tblite_dipole,
    get_tblite_energy,
    get_tblite_forces,
    get_tblite_gradient,
    get_tblite_hamiltonian_matrix,
    get_tblite_overlap_matrix,
    get_tblite_quadrupole,
    tblite_singlepoint,
)
from .molecules import SYSTEMS
from .refdata import Tolerance
from .tblite_live import (
    DEFAULT_CONVERGENCE_LIVE_TOLERANCES,
    EXPECTED_TBLITE_VERSION,
    TIGHT_LIVE_TOLERANCES,
    assert_tblite_agrees,
    evaluate_dxtb_reference,
    get_tblite_version,
    run_tblite_live,
)

pytestmark = [pytest.mark.tblite, pytest.mark.physics_values]


def _nontrivial_rotation_matrix(
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Return a deterministic non-axis-aligned proper rotation R in SO(3)."""
    axis = torch.tensor([0.37, -0.81, 0.45], dtype=dtype, device=device)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = 1.23456789
    kx, ky, kz = axis[0], axis[1], axis[2]
    z = torch.zeros((), dtype=dtype, device=device)
    skew = torch.stack(
        (
            torch.stack((z, -kz, ky)),
            torch.stack((kz, z, -kx)),
            torch.stack((-ky, kx, z)),
        )
    )
    eye = torch.eye(3, dtype=dtype, device=device)
    return (
        eye + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def test_tblite_version() -> None:
    """Verify that the active native tblite library matches version 0.8.0."""
    version = get_tblite_version()
    assert (
        version == EXPECTED_TBLITE_VERSION
    ), f"Expected tblite {EXPECTED_TBLITE_VERSION}, got {version}."


@pytest.mark.parametrize("name", ["H2O", "CH4"])
def test_gfn1_live_reference(name: str) -> None:
    """Compare GFN1-xTB energy, gradient, charges, and dipole against live tblite."""
    sys = SYSTEMS[name]
    live = run_tblite_live(
        "GFN1-xTB",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        uhf=sys.spin,
        accuracy=1.0e-6,
        max_iter=300,
    )
    computed = evaluate_dxtb_reference(
        "gfn1",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        spin=sys.spin,
        x_atol=1.0e-10,
    )
    assert_tblite_agrees(
        method="GFN1-xTB",
        structure=name,
        live=live,
        computed=computed,
        tolerances=TIGHT_LIVE_TOLERANCES,
    )


def test_gfn1_live_4sp_fe_co_5() -> None:
    """Compare GFN1-xTB Fe(CO)5 (4sp expansion) energy, gradient, charges, and dipole against live tblite."""
    sys = SYSTEMS["Fe(CO)5"]
    live = run_tblite_live(
        "GFN1-xTB",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        uhf=sys.spin,
        accuracy=1.0e-6,
        max_iter=300,
    )
    computed = evaluate_dxtb_reference(
        "gfn1",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        spin=sys.spin,
        x_atol=1.0e-10,
    )
    # Tolerances accommodating the 11-atom transition-metal complex
    fe_tolerances = {
        **TIGHT_LIVE_TOLERANCES,
        "energy": Tolerance(atol=2.0e-6, rtol=0.0),
        "gradient": Tolerance(atol=2.0e-8, rtol=0.0),
    }
    assert_tblite_agrees(
        method="GFN1-xTB",
        structure="Fe(CO)5",
        live=live,
        computed=computed,
        tolerances=fe_tolerances,
    )


@pytest.mark.parametrize(
    "name,numbers,positions",
    [
        (
            "RbH",
            torch.tensor([37, 1]),
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 4.47]], dtype=torch.float64
            ),
        ),
        (
            "SrH2",
            torch.tensor([38, 1, 1]),
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, -4.0]],
                dtype=torch.float64,
            ),
        ),
    ],
)
def test_gfn1_live_5sp_reference(
    name: str, numbers: torch.Tensor, positions: torch.Tensor
) -> None:
    """Compare GFN1-xTB 5sp systems (RbH, SrH2) against live tblite."""
    live = run_tblite_live(
        "GFN1-xTB",
        numbers,
        positions,
        charge=0.0,
        uhf=0,
        accuracy=1.0e-6,
        max_iter=300,
    )
    computed = evaluate_dxtb_reference(
        "gfn1",
        numbers,
        positions,
        charge=0.0,
        spin=0,
        x_atol=1.0e-10,
    )
    assert_tblite_agrees(
        method="GFN1-xTB",
        structure=name,
        live=live,
        computed=computed,
        tolerances=TIGHT_LIVE_TOLERANCES,
    )


@pytest.mark.parametrize(
    "name",
    [
        "H2O",
        "CH4",
        "C6H6",
        "glycine",
        "NO2",
        "OH-",
        "NH4+",
        "PbH4-BiH3",
    ],
)
def test_gfn2_live_reference(name: str) -> None:
    """Compare GFN2-xTB energy, gradient, charges, and dipole against live tblite (tight SCF)."""
    sys = SYSTEMS[name]
    live = run_tblite_live(
        "GFN2-xTB",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        uhf=sys.spin,
        accuracy=1.0e-6,
        max_iter=300,
    )
    computed = evaluate_dxtb_reference(
        "gfn2",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        spin=sys.spin,
        x_atol=1.0e-10,
    )
    assert_tblite_agrees(
        method="GFN2-xTB",
        structure=name,
        live=live,
        computed=computed,
        tolerances=TIGHT_LIVE_TOLERANCES,
    )


@pytest.mark.parametrize(
    "name",
    ["H2O", "CH4", "glycine", "PbH4-BiH3"],
)
def test_gfn2_live_default_convergence(name: str) -> None:
    """Compare GFN2-xTB at default SCF tolerance (x_atol=1e-4) against live tblite (accuracy=1.0)."""
    sys = SYSTEMS[name]
    live = run_tblite_live(
        "GFN2-xTB",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        uhf=sys.spin,
        accuracy=1.0,
        max_iter=300,
    )
    computed = evaluate_dxtb_reference(
        "gfn2",
        sys.numbers,
        sys.positions,
        charge=sys.charge,
        spin=sys.spin,
        x_atol=1.0e-4,
    )
    assert_tblite_agrees(
        method="GFN2-xTB[default_scf]",
        structure=name,
        live=live,
        computed=computed,
        tolerances=DEFAULT_CONVERGENCE_LIVE_TOLERANCES,
    )


@pytest.mark.parametrize("name", ["H2O", "glycine", "PbH4-BiH3"])
@pytest.mark.parametrize("variable", ["potential", "charge"])
def test_gfn2_live_rotational_invariance_and_agreement(
    name: str, variable: str
) -> None:
    """Compare GFN2-xTB against live tblite 0.8.0 on original and rotated geometries."""
    sys = SYSTEMS[name]
    rot = _nontrivial_rotation_matrix(
        dtype=sys.positions.dtype, device=sys.positions.device
    )
    assert torch.allclose(
        rot.T @ rot,
        torch.eye(3, dtype=rot.dtype, device=rot.device),
        atol=1.0e-14,
    )
    assert math.isclose(
        float(torch.linalg.det(rot).item()), 1.0, abs_tol=1.0e-14
    )

    pos_orig = sys.positions.clone()
    pos_rot = pos_orig @ rot.T

    live_orig = run_tblite_live(
        "GFN2-xTB",
        sys.numbers,
        pos_orig,
        charge=sys.charge,
        uhf=sys.spin,
        accuracy=1.0e-6,
        max_iter=300,
    )
    live_rot = run_tblite_live(
        "GFN2-xTB",
        sys.numbers,
        pos_rot,
        charge=sys.charge,
        uhf=sys.spin,
        accuracy=1.0e-6,
        max_iter=300,
    )

    dxtb_orig = evaluate_dxtb_reference(
        "gfn2",
        sys.numbers,
        pos_orig,
        charge=sys.charge,
        spin=sys.spin,
        variable=variable,
        x_atol=1.0e-9,
    )
    dxtb_rot = evaluate_dxtb_reference(
        "gfn2",
        sys.numbers,
        pos_rot,
        charge=sys.charge,
        spin=sys.spin,
        variable=variable,
        x_atol=1.0e-9,
    )

    assert_tblite_agrees(
        method=f"GFN2-xTB[{variable}]",
        structure=f"{name}-orig",
        live=live_orig,
        computed=dxtb_orig,
        tolerances=TIGHT_LIVE_TOLERANCES,
    )
    assert_tblite_agrees(
        method=f"GFN2-xTB[{variable}]",
        structure=f"{name}-rotated",
        live=live_rot,
        computed=dxtb_rot,
        tolerances=TIGHT_LIVE_TOLERANCES,
    )

    # Verify dxtb rotational invariance between original and rotated frames.
    torch.testing.assert_close(
        dxtb_rot["energy"], dxtb_orig["energy"], atol=1.0e-12, rtol=0.0
    )
    torch.testing.assert_close(
        dxtb_rot["gradient"] @ rot,
        dxtb_orig["gradient"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        dxtb_rot["charges"], dxtb_orig["charges"], atol=1.0e-10, rtol=1.0e-10
    )
    torch.testing.assert_close(
        dxtb_rot["dipole"] @ rot,
        dxtb_orig["dipole"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_tblite_live_failure_reporting() -> None:
    """Verify that assertion failures report both dxtb and tblite values and diffs."""
    sys = SYSTEMS["H2O"]
    live = run_tblite_live("GFN2-xTB", sys.numbers, sys.positions)
    corrupted = {
        "energy": torch.tensor(live.energy + 0.125, dtype=torch.float64),
        "gradient": torch.as_tensor(live.gradient, dtype=torch.float64),
        "charges": torch.as_tensor(live.charges, dtype=torch.float64),
        "dipole": torch.as_tensor(live.dipole, dtype=torch.float64),
    }
    strict_tol = {
        "energy": Tolerance(atol=1.0e-10, rtol=1.0e-10),
        "gradient": Tolerance(atol=1.0e-6, rtol=1.0e-5),
        "charges": Tolerance(atol=1.0e-6, rtol=1.0e-5),
        "dipole": Tolerance(atol=1.0e-6, rtol=1.0e-5),
    }
    with pytest.raises(AssertionError) as excinfo:
        assert_tblite_agrees(
            method="GFN2-xTB",
            structure="H2O",
            live=live,
            computed=corrupted,
            tolerances=strict_tol,
        )
    msg = str(excinfo.value)
    assert "0.8.0" in msg
    assert "dxtb[energy]" in msg
    assert "tblite[energy]" in msg
    assert "max_abs=" in msg
    assert "max_rel=" in msg


def test_tblite_live_executes_singlepoint_dynamically() -> None:
    """Confirm each live call invokes tblite.interface.Calculator.singlepoint() dynamically."""
    import tblite.interface

    sys = SYSTEMS["H2O"]
    orig_singlepoint = tblite.interface.Calculator.singlepoint
    with patch.object(
        tblite.interface.Calculator,
        "singlepoint",
        autospec=True,
        side_effect=orig_singlepoint,
    ) as spy_singlepoint:
        res_eq = run_tblite_live("GFN2-xTB", sys.numbers, sys.positions)
        stretched_pos = sys.positions.clone()
        stretched_pos[1, 0] += 0.15
        res_stretched = run_tblite_live("GFN2-xTB", sys.numbers, stretched_pos)

    assert spy_singlepoint.call_count == 2
    assert abs(res_stretched.energy - res_eq.energy) > 1.0e-4


def test_tblite_structure_and_property_getters() -> None:
    """Verify the Structure, Options, and get_tblite_* testsuite interface."""
    sys = SYSTEMS["H2O"]
    mol = Structure(sys.numbers, sys.positions, charge=sys.charge, uhf=sys.spin)
    opts = Options(method="GFN2-xTB", accuracy=1.0e-6, max_iter=300)

    res = tblite_singlepoint(mol, options=opts)
    energy = get_tblite_energy(mol, options=opts)
    gradient = get_tblite_gradient(mol, options=opts)
    forces = get_tblite_forces(mol, options=opts)
    charges = get_tblite_charges(mol, options=opts)
    dipole = get_tblite_dipole(mol, options=opts)
    quadrupole = get_tblite_quadrupole(mol, options=opts)
    overlap = get_tblite_overlap_matrix(mol, options=opts)
    hcore = get_tblite_hamiltonian_matrix(mol, options=opts)

    assert isinstance(energy, torch.Tensor)
    torch.testing.assert_close(energy, res.energy, atol=0.0, rtol=0.0)
    torch.testing.assert_close(gradient, res.gradient, atol=0.0, rtol=0.0)
    torch.testing.assert_close(forces, -gradient, atol=0.0, rtol=0.0)
    torch.testing.assert_close(charges, res.charges, atol=0.0, rtol=0.0)
    torch.testing.assert_close(dipole, res.dipole, atol=0.0, rtol=0.0)
    torch.testing.assert_close(quadrupole, res.quadrupole, atol=0.0, rtol=0.0)
    assert overlap.shape == (6, 6)
    assert hcore.shape == (6, 6)
    torch.testing.assert_close(
        torch.diagonal(overlap),
        torch.ones(6, dtype=overlap.dtype, device=overlap.device),
        atol=1.0e-9,
        rtol=0.0,
    )

    # Also verify hyphenated tblite option dict keys
    energy_dict = get_tblite_energy(
        mol,
        "GFN2-xTB",
        options={"accuracy": 1.0e-6, "max-iter": 300, "mixer-damping": 0.4},
    )
    torch.testing.assert_close(energy_dict, energy, atol=1.0e-12, rtol=0.0)

    # Verify electric_field interaction via CFFI pointer conversion
    efield = torch.tensor([0.0, 0.0, 1.0e-3], dtype=torch.float64)
    energy_field = get_tblite_energy(
        mol,
        "GFN2-xTB",
        accuracy=1.0e-6,
        electric_field=efield,
    )
    assert abs(float((energy_field - energy).item())) > 1.0e-4


def test_tblite_interface_dtype_temperature_uhf_and_immutability() -> None:
    """Verify WP3 contract: float64 default dtype, temperature/uhf validation, and input immutability."""
    from tad_mctc.io.structure import Structure as MctcStructure
    from tad_mctc.units import KELVIN2AU

    sys = SYSTEMS["H2O"]
    pos_f32 = sys.positions.to(dtype=torch.float32)
    pos_f32_copy = pos_f32.clone()
    num_copy = sys.numbers.clone()

    mol_f32 = Structure(sys.numbers, pos_f32, charge=0.0, uhf=0)

    # 1. Default return dtype is float64 even when Structure.positions is float32.
    energy_default = get_tblite_energy(mol_f32, "GFN2-xTB", accuracy=1.0e-6)
    grad_default = get_tblite_gradient(mol_f32, "GFN2-xTB", accuracy=1.0e-6)
    assert energy_default.dtype == torch.float64
    assert grad_default.dtype == torch.float64

    # Explicit dtype=torch.float32 casts the output when requested.
    energy_f32 = get_tblite_energy(
        mol_f32, "GFN2-xTB", accuracy=1.0e-6, dtype=torch.float32
    )
    assert energy_f32.dtype == torch.float32

    # Caller tensors are never mutated.
    assert torch.equal(pos_f32, pos_f32_copy)
    assert torch.equal(sys.numbers, num_copy)

    # 2. tad_mctc.io.structure.Structure compatibility.
    mctc_mol = MctcStructure(
        numbers=sys.numbers, positions=sys.positions, charge=sys.charge
    )
    energy_mctc = get_tblite_energy(mctc_mol, "GFN2-xTB", accuracy=1.0e-6)
    torch.testing.assert_close(
        energy_mctc, energy_default, atol=1.0e-6, rtol=0.0
    )

    # 3. Temperature in Hartree vs Kelvin agreement and conflict rejection.
    temp_k = 450.0
    temp_au = temp_k * KELVIN2AU
    e_kelvin = get_tblite_energy(
        mctc_mol, "GFN2-xTB", accuracy=1.0e-6, temperature_kelvin=temp_k
    )
    e_au = get_tblite_energy(
        mctc_mol, "GFN2-xTB", accuracy=1.0e-6, temperature=temp_au
    )
    torch.testing.assert_close(e_kelvin, e_au, atol=1.0e-14, rtol=0.0)

    with pytest.raises(ValueError, match="Specify at most one"):
        Options(temperature=temp_au, temperature_kelvin=temp_k)
    with pytest.raises(ValueError, match="Specify at most one"):
        get_tblite_energy(
            mctc_mol,
            "GFN2-xTB",
            temperature=temp_au,
            temperature_kelvin=temp_k,
        )

    # 4. Open-shell uhf/spin handling and non-integer rejection.
    no2 = SYSTEMS["NO2"]
    e_uhf = get_tblite_energy(
        Structure(no2.numbers, no2.positions, charge=0.0, uhf=1),
        "GFN2-xTB",
        accuracy=1.0e-6,
    )
    e_spin = get_tblite_energy(
        Structure(no2.numbers, no2.positions, charge=0.0, spin=1),
        "GFN2-xTB",
        accuracy=1.0e-6,
    )
    torch.testing.assert_close(e_uhf, e_spin, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="Conflicting"):
        Structure(no2.numbers, no2.positions, uhf=1, spin=2)
    with pytest.raises(ValueError, match="integer number of unpaired electrons"):
        Structure(no2.numbers, no2.positions, uhf=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        Structure(no2.numbers, no2.positions, uhf=-1)
