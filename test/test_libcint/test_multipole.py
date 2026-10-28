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

from dxtb import IndexHelper, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint, has_pyscf
from dxtb._src.integral.driver.manager import DriverManager
from dxtb._src.integral.types.quadrupole import _reduce_9_to_6
from dxtb._src.typing import DD, Tensor
from dxtb.integrals import Integrals
from dxtb.integrals.factories import new_quadint

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

    # Setup the driver
    mgr = DriverManager(labels.INTDRIVER_LIBCINT, **dd)
    mgr.create_driver(numbers, par, ihelp)
    i = Integrals(mgr, intlevel=labels.INTLEVEL_DIPOLE, **dd)
    i.build_dipole(positions, shift=True)
    assert i.dipole is not None
    assert i.overlap is not None

    # Native libcint j is already centered at the ket atom.
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_rj = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(
            numpy_to_tensor(mol.intor("int1e_r_origj"), **dd), ihelp
        ),
        i.overlap.norm,
        i.overlap.norm,
    )

    assert i.dipole.matrix is not None
    torch.testing.assert_close(
        i.dipole.matrix.cpu(), pyscf_rj.cpu(), atol=tol, rtol=1e-6
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

    # Setup the driver
    mgr = DriverManager(labels.INTDRIVER_LIBCINT, **dd)
    mgr.create_driver(numbers, par, ihelp)

    i = Integrals(mgr, intlevel=labels.INTLEVEL_QUADRUPOLE, **dd)
    i.build_quadrupole(positions, shift=False, traceless=False)
    assert i.quadrupole is not None

    # We always normalize, i.e., the overlap is always present
    assert i.overlap is not None

    ##########################################################################

    # PySCF reference integral, centered at r0
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_r0r0 = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(numpy_to_tensor(mol.intor("int1e_rr"), **dd), ihelp),
        i.overlap.norm,
        i.overlap.norm,
    )
    pyscf_r0r0 = _reduce_9_to_6(pyscf_r0r0)

    # Compare with PySCF reference
    assert pyscf_r0r0.shape == i.quadrupole.matrix.shape
    assert pytest.approx(pyscf_r0r0.cpu(), abs=tol) == i.quadrupole.matrix.cpu()


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

    # Setup the driver
    mgr = DriverManager(labels.INTDRIVER_LIBCINT, **dd)
    mgr.create_driver(numbers, par, ihelp)
    i = Integrals(mgr, intlevel=labels.INTLEVEL_QUADRUPOLE, **dd)
    i.build_quadrupole(positions, shift=True, traceless=False)
    assert i.quadrupole is not None
    assert i.overlap is not None

    # Native libcint jj is already packed and centered at the ket atom.
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_qpint = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(
            numpy_to_tensor(mol.intor("int1e_rr_origj"), **dd), ihelp
        ),
        i.overlap.norm,
        i.overlap.norm,
    )
    pyscf_qpint = _reduce_9_to_6(pyscf_qpint)

    assert i.quadrupole.matrix is not None
    torch.testing.assert_close(
        i.quadrupole.matrix.cpu(), pyscf_qpint.cpu(), atol=tol, rtol=0.0
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

    # Setup the driver
    mgr = DriverManager(
        labels.INTDRIVER_LIBCINT, force_cpu_for_libcint=True, **dd
    )
    mgr.create_driver(numbers, par, ihelp)
    mgr.setup_driver(positions)

    i = Integrals(mgr, intlevel=labels.INTLEVEL_QUADRUPOLE, **dd)
    i.build_overlap(positions)
    i.build_dipole(positions)
    i.build_quadrupole(positions)

    assert i.overlap is not None
    assert i.dipole is not None
    assert i.quadrupole is not None

    # PySCF reference integral, centered at rj
    assert M is not False
    mol = M(numbers, positions, xtb_version=gfn, parse_arg=False)
    pyscf_qp = einsum(
        "xij,i,j->xij",
        reorder_pyscf_ao(
            numpy_to_tensor(mol.intor("int1e_rr_origj"), **dd), ihelp
        ),
        i.overlap.norm,
        i.overlap.norm,
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
    assert pyscf_qp.shape == i.quadrupole.matrix.shape
    assert pytest.approx(pyscf_qp.cpu(), abs=tol) == i.quadrupole.matrix.cpu()


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
def test_integrals_rebuild_multipole_inputs_for_new_geometry() -> None:
    """Repeated quadrupole builds use overlap from the current geometry."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples["H2"]
    numbers = sample["numbers"].to(DEVICE)
    positions_a = sample["positions"].to(**dd)
    positions_b = positions_a.clone()
    positions_b[1, 2] += 0.25
    par = get_param_module("gfn1", **dd)

    def make_integrals() -> Integrals:
        ihelp = IndexHelper.from_numbers(numbers, par)
        manager = DriverManager(
            labels.INTDRIVER_LIBCINT,
            force_cpu_for_libcint=True,
            **dd,
        )
        manager.create_driver(numbers, par, ihelp)
        return Integrals(manager, intlevel=labels.INTLEVEL_QUADRUPOLE, **dd)

    reused = make_integrals()
    reused.build_quadrupole(positions_a)
    reused_quadrupole = reused.quadrupole
    assert reused_quadrupole is not None
    quadrupole_a = reused_quadrupole.matrix.clone()

    reused.build_quadrupole(positions_b)
    fresh = make_integrals()
    fresh.build_quadrupole(positions_b)
    assert fresh.quadrupole is not None
    assert torch.allclose(reused.quadrupole.matrix, fresh.quadrupole.matrix)
    assert not torch.allclose(quadrupole_a, reused.quadrupole.matrix)


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

    qpint = new_quadint(labels.INTDRIVER_LIBCINT, **dd)
    qpint._matrix = torch.zeros(9, 2, 2, **dd)

    with pytest.raises(RuntimeError) as e:
        qpint.shift_r0r0_rjrj(fake_dipint, fake_ovlp, pos)

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

    qpint = new_quadint(labels.INTDRIVER_LIBCINT, **dd)
    qpint._matrix = torch.zeros(8, 2, 2, **dd)  # wrong shape

    with pytest.raises(RuntimeError) as e:
        qpint.shift_r0r0_rjrj(fake_dipint, fake_ovlp, pos)

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
