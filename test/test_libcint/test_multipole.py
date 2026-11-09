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
Test overlap from libcint.

We cannot compare with our internal integrals or the reference integrals from
tblite, because the sorting of the p-orbitals are different.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.convert import numpy_to_tensor
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint, has_pyscf
from dxtb._src.integral.driver.libcint import (
    LibcintIntegralSetup,
    build_multipole_libcint,
    build_overlap_libcint,
    setup_libcint,
)
from dxtb._src.integral.types.quadrupole import (
    _reduce_9_to_6,
    make_quadrupole_traceless,
    reduce_quadrupole_9_to_6,
    shift_quadrupole_origin,
)
from dxtb._src.integral.utils import normalize_integral_matrix
from dxtb._src.integral.utils import snorm as overlap_norm
from dxtb._src.param import ParamModule
from dxtb._src.typing import DD, Tensor

if has_pyscf is True:
    from dxtb._src.exlibs.pyscf.mol import M
if has_libcint is True:
    from dxtb._src.exlibs import libcint

from ..conftest import DEVICE
from ..utils import get_param_module
from .ao_order import reorder_pyscf_ao
from .samples import samples

slist = ["H2", "LiH", "Li2", "H2O", "S"]
slist_large = ["SiH4", "MB16_43_01", "C60"]


def snorm(overlap: Tensor) -> Tensor:
    return torch.pow(overlap.diagonal(dim1=-1, dim2=-2), -0.5)


def libcint_moment(
    numbers: Tensor,
    positions: Tensor,
    par: ParamModule,
    dd: DD,
    intstring: str,
) -> tuple[Tensor, Tensor]:
    """Multipole integral normalized with the overlap norm, and that norm."""
    ihelp = IndexHelper.from_numbers(numbers, par)
    setup = LibcintIntegralSetup(
        numbers=numbers,
        ihelp=ihelp,
        basis_setups=(Basis(numbers, par, ihelp, **dd).setup_data(),),
        force_cpu=True,
    )
    call_data = setup_libcint(setup, positions)
    norm = overlap_norm(build_overlap_libcint(call_data))
    matrix = build_multipole_libcint(call_data, intstring)
    return normalize_integral_matrix(matrix, norm), norm


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_single(dtype: torch.dtype, name: str, gfn) -> None:
    run_single(dtype, name, gfn)


@pytest.mark.large
@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist_large)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_large(dtype: torch.dtype, name: str, gfn: str) -> None:
    run_single(dtype, name, gfn)


def run_single(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    par = get_param_module(gfn, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = Basis(numbers, par, ihelp, **dd)
    libcint_basis, structure = bas.create_libcint(positions)

    def _run(i: int, atol: float) -> None:
        # Native j/jj integrals are already ket-centered. Keep all nine
        # quadrupole components here for direct comparison with PySCF.
        dxtb_mpint = libcint.int1e(
            i * "j", libcint_basis, structure, packed=False
        )

        # pyscf reference integral
        assert M is not False
        mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
        pyscf_mpint = reorder_pyscf_ao(
            numpy_to_tensor(mol.intor(f"int1e_{i*'r'}_origj"), **dd), ihelp
        )

        assert dxtb_mpint.shape == pyscf_mpint.shape
        assert pytest.approx(pyscf_mpint.cpu(), abs=atol) == dxtb_mpint.cpu()

        # normalize
        norm = snorm(libcint.overlap(libcint_basis, structure))
        dxtb_mpint = einsum("xij,i,j->xij", dxtb_mpint, norm, norm)
        pyscf_mpint = einsum("xij,i,j->xij", pyscf_mpint, norm, norm)

        assert dxtb_mpint.shape == pyscf_mpint.shape
        assert pytest.approx(pyscf_mpint.cpu(), abs=tol) == dxtb_mpint.cpu()

    _run(1, atol=tol * 1e-2)  # dipole
    _run(2, atol=tol)  # quadrupole


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_shift_r0_rj(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    par = get_param_module(gfn, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    ##########################################################################

    dipole, norm = libcint_moment(numbers, positions, par, dd, "j")

    # Native libcint j is already centered at the ket atom.
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_rj = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(
            numpy_to_tensor(mol.intor("int1e_r_origj"), **dd), ihelp
        ),
        norm,
        norm,
    )

    torch.testing.assert_close(
        dipole.cpu(), pyscf_rj.cpu(), atol=tol, rtol=1e-6
    )


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_no_shift_r0r0_rjrj(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    par = get_param_module(gfn, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    ##########################################################################

    quadrupole, norm = libcint_moment(numbers, positions, par, dd, "r0r0")

    ##########################################################################

    # PySCF reference integral, centered at r0
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_r0r0 = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(numpy_to_tensor(mol.intor("int1e_rr"), **dd), ihelp),
        norm,
        norm,
    )
    pyscf_r0r0 = _reduce_9_to_6(pyscf_r0r0)

    # Compare with PySCF reference
    assert pyscf_r0r0.shape == quadrupole.shape
    assert pytest.approx(pyscf_r0r0.cpu(), abs=tol) == quadrupole.cpu()


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_shift_r0r0_rjrj(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    par = get_param_module(gfn, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    ##########################################################################

    quadrupole, norm = libcint_moment(numbers, positions, par, dd, "jj")

    # Native libcint jj is already packed and centered at the ket atom.
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_qpint = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(
            numpy_to_tensor(mol.intor("int1e_rr_origj"), **dd), ihelp
        ),
        norm,
        norm,
    )
    pyscf_qpint = _reduce_9_to_6(pyscf_qpint)

    torch.testing.assert_close(
        quadrupole.cpu(), pyscf_qpint.cpu(), atol=tol, rtol=0.0
    )


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_traceless(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    par = get_param_module(gfn, **dd)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)

    ##########################################################################

    quadrupole, norm = libcint_moment(numbers, positions, par, dd, "jj")
    quadrupole = make_quadrupole_traceless(quadrupole)

    # PySCF reference integral, centered at rj
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_qp = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(
            numpy_to_tensor(mol.intor("int1e_rr_origj"), **dd), ihelp
        ),
        norm,
        norm,
    )

    # Remove symmetry-equivalent components
    pyscf_qp = _reduce_9_to_6(pyscf_qp)

    # Make traceless
    tr = 0.5 * (pyscf_qp[0] + pyscf_qp[2] + pyscf_qp[5])
    pyscf_qp = torch.stack(
        [
            1.5 * pyscf_qp[0] - tr,
            1.5 * pyscf_qp[1],
            1.5 * pyscf_qp[2] - tr,
            1.5 * pyscf_qp[3],
            1.5 * pyscf_qp[4],
            1.5 * pyscf_qp[5] - tr,
        ]
    )

    # Compare with PySCF reference
    assert pyscf_qp.shape == quadrupole.shape
    assert pytest.approx(pyscf_qp.cpu(), abs=tol) == quadrupole.cpu()


@pytest.mark.skipif(
    has_pyscf is False or has_libcint is False,
    reason="PySCF or libcint interface not installed",
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_reduce_9_to_6(dtype: torch.dtype, name: str, gfn: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_r0r0 = numpy_to_tensor(mol.intor("int1e_rr"), **dd)

    # 0
    # 3 4    ->  0, 3, 4, 6, 7, 8
    # 6 7 8
    int_l = _reduce_9_to_6(pyscf_r0r0, uplo="L").cpu()

    # 0 1 2
    #   4 5  ->  0, 1, 2, 4, 5, 8
    #     8
    int_u = _reduce_9_to_6(pyscf_r0r0, uplo="U").cpu()

    assert int_l.shape == int_u.shape

    # equivalent indices: l0-u0, l3-u1, l4-l4, u6-l2, u7-l5, u8-l8
    assert pytest.approx(int_l[0], abs=tol) == int_u[0]  # l0-u0
    assert pytest.approx(int_l[1], abs=tol) == int_u[1]  # l3-u1
    assert pytest.approx(int_l[2], abs=tol) == int_u[3]  # l4-l4
    assert pytest.approx(int_l[3], abs=tol) == int_u[2]  # u6-l2
    assert pytest.approx(int_l[4], abs=tol) == int_u[4]  # u7-l5
    assert pytest.approx(int_l[5], abs=tol) == int_u[5]  # u8-l8


@pytest.mark.skipif(
    has_libcint is False, reason="libcint interface not installed"
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_shift_shape_pos(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    fake_dipint = torch.zeros(3, 2, 2, **dd)
    fake_ovlp = torch.eye(2, **dd)
    pos = torch.zeros(1, 3, **dd)  # wrong shape

    qpint = reduce_quadrupole_9_to_6(torch.zeros(9, 2, 2, **dd))

    with pytest.raises(RuntimeError) as e:
        shift_quadrupole_origin(qpint, fake_dipint, fake_ovlp, pos)

    assert "Shape mismatch between positions and overlap" in str(e.value)


@pytest.mark.skipif(
    has_libcint is False, reason="libcint interface not installed"
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_shift_shape_qpint(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    fake_dipint = torch.zeros(3, 2, 2, **dd)
    fake_ovlp = torch.eye(2, **dd)
    pos = torch.zeros(2, 3, **dd)

    qpint = torch.zeros(8, 2, 2, **dd)  # wrong shape

    with pytest.raises(RuntimeError) as e:
        shift_quadrupole_origin(qpint, fake_dipint, fake_ovlp, pos)

    assert "Quadrupole integral must be a tensor of shape" in str(e.value)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_reduce_shape(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    qpint = torch.zeros(8, 2, 2, **dd)
    with pytest.raises(RuntimeError):
        _reduce_9_to_6(qpint)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_fail_reduce_uplo(dtype: torch.dtype) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}

    qpint = torch.zeros(9, 2, 2, **dd)
    with pytest.raises(ValueError):
        _reduce_9_to_6(qpint, uplo="X")  # type: ignore
