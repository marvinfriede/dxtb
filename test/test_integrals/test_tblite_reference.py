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
GFN2-xTB single points of the PyTorch integral driver, for every algorithm,
against tblite (the reference xTB implementation).

tblite's Python API exposes energy, gradient, dipole, quadrupole and orbital
energies but not the overlap matrix, so the integrals are checked through
these observables. dxtb and tblite differ by about 1e-7 Eh in the energy for
reasons unrelated to the integrals (the libcint driver shows the same offset),
hence the tolerances below; the integral-level agreement with libcint is
tested in ``test_pytorch_gfn2.py``.
"""

from __future__ import annotations

import pytest
import torch

import dxtb
from dxtb._src.typing import DD

from ..conftest import DEVICE
from ..tblite import Structure, get_tblite_quadrupole, tblite_singlepoint
from .samples import samples

pytestmark = [pytest.mark.tblite]

ALGOS = ["md", "os"]


def _calculator(numbers, algorithm: str, dd: DD):
    # the dipole is a derivative with respect to a (zero) field
    field = torch.zeros(3, **dd, requires_grad=True)
    calc = dxtb.calculators.GFN2Calculator(
        numbers,
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "int_algorithm": algorithm,
            "int_level": dxtb.labels.INTLEVEL_DIPOLE,
        },
        **dd,
    )
    return calc, field


@pytest.mark.parametrize("algorithm", ALGOS)
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_energy_gradient_dipole_match_tblite(name: str, algorithm: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    mol = Structure(numbers, positions)
    res = tblite_singlepoint(mol, "GFN2-xTB", accuracy=1e-4)

    def fresh():
        return positions.detach().clone().requires_grad_(True)

    calc, field = _calculator(numbers, algorithm, dd)
    energy = calc.get_energy(fresh(), field=field)
    calc, field = _calculator(numbers, algorithm, dd)
    forces = calc.get_forces(fresh(), field=field)
    calc, field = _calculator(numbers, algorithm, dd)
    dipole = calc.get_dipole(fresh(), field=field)

    assert energy.detach().cpu() == pytest.approx(float(res.energy), abs=1e-6)
    assert torch.allclose(-forces, res.gradient, atol=1e-7, rtol=0.0)
    assert torch.allclose(dipole, res.dipole, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("algorithm", ALGOS)
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_quadrupole_matches_tblite(
    name: str, algorithm: str, rotate: bool
) -> None:
    """Molecular traceless quadrupole (``xx, yx, yy, zx, zy, zz``) of the
    public ``get_quadrupole`` against tblite's, for every algorithm. The
    rotated geometry gives nonzero off-diagonal elements (the samples are
    symmetric), including tblite's own off-diagonal packing."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    if rotate:
        rot = torch.tensor(
            [
                [-0.299179, -0.587620, -0.751794],
                [0.410265, -0.790554, 0.454650],
                [-0.861496, -0.172413, 0.477597],
            ],
            **dd,
        )
        rot = torch.linalg.qr(rot)[0]  # exactly orthogonal
        positions = positions @ rot.T

    mol = Structure(numbers, positions)
    expected = get_tblite_quadrupole(mol, "GFN2-xTB", accuracy=1e-6)

    field_grad = torch.zeros((3, 3), **dd, requires_grad=True)
    calc = dxtb.calculators.GFN2Calculator(
        numbers,
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "int_algorithm": algorithm,
            "differentiation": "unrolled",
            "x_atol": 1e-11,
            "int_level": dxtb.labels.INTLEVEL_QUADRUPOLE,
        },
        **dd,
    )
    quad = calc.get_quadrupole(positions, field_grad=field_grad)

    assert quad.shape == (6,)
    # Diagonal elements (xx, yy, zz) agree directly with tblite; off-diagonal
    # elements (yx, zx, zy) differ by 1.5 * Q_ij of the point charges and
    # dipoles, which tblite's analytical expression counts twice.
    diag = [0, 2, 5]
    offdiag = [1, 3, 4]
    assert torch.allclose(quad[diag], expected[diag], atol=1e-7, rtol=0.0)
    sp = calc.singlepoint(positions)
    qat = calc.ihelp.reduce_orbital_to_atom(sp.charges.mono)
    mu = sp.charges.dipole
    assert mu is not None
    q_pt = (
        torch.einsum("...a,...ai,...aj->...ij", qat, positions, positions)
        + torch.einsum("...ai,...aj->...ij", positions, mu)
        + torch.einsum("...ai,...aj->...ij", mu, positions)
    ).detach()
    off = torch.stack([q_pt[1, 0], q_pt[2, 0], q_pt[2, 1]])
    assert torch.allclose(
        expected[offdiag] - quad[offdiag], 1.5 * off, atol=1e-7, rtol=0.0
    )
    # traceless
    assert abs(float((quad[0] + quad[2] + quad[5]).detach().item())) < 1e-10
