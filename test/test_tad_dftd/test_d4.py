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
Tests for Milestone C: DFT-D4, DFT-D4S, and DFT-D4Srev dispersion models,
reference tables, EEQ and EEQBC charge models, C6 and polarizability
conformance, analytical charge derivatives (dgwdq), periodic cells, batches,
autograd, and functional transforms.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import pack_structures

from tad_dftd import (
    Cutoff,
    D4Model,
    D4Reference,
    D4SModel,
    D4SrevModel,
    DensePairs,
    Environment,
    SparsePairs,
    charge,
    damping,
    data,
    dftd,
    dftd4,
    dftd4s,
    dftd4srev,
    dispersion,
    dispersion2,
    dispersion3,
    get_properties,
    model,
    ncoord,
    reference,
)


def _conformance_param(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    """
    Complete damping parameters from ``dftd/test/unit/util.f90:get_param``.
    """
    return {
        "s6": torch.tensor(1.0, dtype=dtype),
        "s8": torch.tensor(0.95948085, dtype=dtype),
        "s9": torch.tensor(1.0, dtype=dtype),
        "a1": torch.tensor(0.38574991, dtype=dtype),
        "a2": torch.tensor(4.80688534, dtype=dtype),
        "a3": torch.tensor(1.0, dtype=dtype),
        "a4": torch.tensor(1.0, dtype=dtype),
        "alp": torch.tensor(16.0, dtype=dtype),
        "rs6": torch.tensor(1.0, dtype=dtype),
        "rs8": torch.tensor(1.0, dtype=dtype),
        "rs9": torch.tensor(1.0, dtype=dtype),
        "a13": torch.tensor(0.38574991, dtype=dtype),
        "a23": torch.tensor(4.80688534, dtype=dtype),
        "alp3": torch.tensor(16.0, dtype=dtype),
    }


def _amf3_structure(dtype: torch.dtype = torch.float64) -> Structure:
    """
    AmF3 structure from ``dftd/test/unit/test_dftd.f90:test_tpsshd4atm_amf3``.
    """
    numbers = torch.tensor([95, 9, 9, 9], dtype=torch.long)
    xyz = torch.tensor(
        [
            [-1.13163973200000, -2.17446990100000, +1.10012477100000],
            [-4.66377948900000, -3.12947883400000, -0.36987606800000],
            [-0.19032564300000, +1.36339950600000, -0.36521789300000],
            [+1.46283310800000, -4.75734549200000, -0.36503081000000],
        ],
        dtype=dtype,
    )
    return Structure(numbers=numbers, positions=xyz)


def _actinides_d4(dtype: torch.dtype = torch.float64) -> Structure:
    """
    17-atom actinide cluster (Z=87..103) from
    ``dftd/test/unit/test_dftd.f90:test_actinides_d4``.
    """
    numbers = torch.arange(87, 104, dtype=torch.long)
    xyz = torch.tensor(
        [
            [0.98692316414074, 6.12727238368797, -6.67861597188102],
            [3.63898862390869, 5.12109301182962, 3.01908613326278],
            [5.14503571563551, -3.97172984617710, 3.82011791828867],
            [6.71986847575494, 1.71382138402812, 3.92749159076307],
            [4.13783589704826, -2.10695793491818, 0.19753203068899],
            [8.97685097698326, -3.08813636191844, -4.45568615593938],
            [12.5486412940776, -1.77128765259458, 0.59261498922861],
            [7.82051475868325, -3.97159756604558, -0.53637703616916],
            [-0.43444574624893, -1.69696511583960, -1.65898182093050],
            [-4.71270645149099, -0.11534827468942, 2.84863373521297],
            [-2.52061680335614, 1.82937752749537, -2.10366982879172],
            [0.13551154616576, 7.99805359235043, -1.55508522619903],
            [3.91594542499717, -1.72975169129597, -5.07944366756113],
            [-1.03393930231679, 4.69307230054046, 0.02656940927472],
            [6.20675384557240, 4.24490721493632, -0.71004195169885],
            [7.04586341131562, 5.20053667939076, -7.51972863675876],
            [2.01082807362334, 1.34838807211157, -4.70482633508447],
        ],
        dtype=dtype,
    )
    return Structure(numbers=numbers, positions=xyz)


# ===========================================================================
# 1. Reference tables and charge models
# ===========================================================================


def test_d4_reference_tables() -> None:
    ref_d4 = reference.d4.load_d4(dtype=torch.float64)
    ref_srev = reference.d4.load_d4srev(dtype=torch.float64)

    assert isinstance(ref_d4, D4Reference)
    assert isinstance(ref_srev, D4Reference)
    assert ref_d4.n_elem == 119
    assert ref_d4.n_ref == 7
    assert ref_d4.cn.shape == (119, 7)
    assert ref_d4.q.shape == (119, 7)
    assert ref_d4.c6.shape == (119, 119, 7, 7)
    assert ref_d4.aiw.shape == (119, 7, 23)
    assert ref_d4.refn.shape == (119,)
    assert ref_d4.ngw.shape == (119, 7)

    # Padding row/col at Z=0 must be zero
    assert torch.all(ref_d4.cn[0] == 0.0)
    assert torch.all(ref_d4.q[0] == 0.0)
    assert torch.all(ref_d4.c6[0] == 0.0)
    assert torch.all(ref_d4.c6[:, 0] == 0.0)

    # Gap Z=104..111 has refn == 0, superheavies Z=112..118 have refn > 0
    assert torch.all(ref_d4.refn[104:112] == 0)
    assert torch.all(ref_d4.refn[112:119] > 0)

    # D4 and D4Srev differ in their reference polarizabilities/C6
    assert not torch.allclose(ref_d4.aiw, ref_srev.aiw)
    assert not torch.allclose(ref_d4.c6, ref_srev.c6)


# ===========================================================================
# 2. C6 matrix conformance against Fortran test_model.f90
# ===========================================================================


def test_c6_d4_mb01() -> None:
    """
    Reference C6 matrix for MB16-43 01 with D4 (with_cn=True, with_q=False)
    from ``dftd/test/unit/test_model.f90:test_c6_d4_mb01``.
    """
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    cn = ncoord.cn_d4(mol)
    q = torch.zeros_like(cn)

    d4 = D4Model()
    gw = d4.weights(mol, Environment(cn=cn, q=q))
    c6 = d4.coefficients(mol, gw).dense()

    ref_diag = torch.tensor(
        [
            1.0673751057011e02,
            3.0564966218613e00,
            1.4581513138206e01,
            3.0376708717485e00,
            9.4188339482987e00,
            3.0787340802721e00,
            3.0376112296738e00,
            1.4455069876704e01,
            1.9888865465619e01,
            3.1063343888377e00,
            3.0787899812795e00,
            9.7300368597444e01,
            3.3522232984596e01,
            3.6486691424878e01,
            1.9888583164001e01,
            1.5002113554136e02,
        ],
        dtype=torch.float64,
    )
    ref_row0 = torch.tensor(
        [
            1.0673751057011e02,
            1.3341905517525e01,
            2.5653153287381e01,
            1.3286256032550e01,
            1.8172837654743e01,
            1.3407379807008e01,
            1.3286079405123e01,
            2.5572059731044e01,
            3.2413191615239e01,
            1.3488257086895e01,
            1.3407544045253e01,
            7.3738556418539e01,
            4.4740018054061e01,
            4.7377208046689e01,
            3.2412969775949e01,
            1.0847087588239e02,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(torch.diagonal(c6), ref_diag, rtol=1e-12, atol=1e-12)
    assert torch.allclose(c6[0], ref_row0, rtol=1e-12, atol=1e-12)


def test_c6_d4s_mb01() -> None:
    """
    Reference C6 matrix for MB16-43 01 with D4S (with_cn=True, with_q=False)
    from ``dftd/test/unit/test_model.f90:test_c6_d4s_mb01``.
    """
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    cn = ncoord.cn_d4(mol)
    q = torch.zeros_like(cn)

    d4s = D4SModel()
    c6 = d4s.coefficients(mol, Environment(cn=cn, q=q)).dense()

    ref_row0 = torch.tensor(
        [
            2.4365306222463e02,
            2.0332338587140e01,
            3.0502619153371e01,
            1.8298596791412e01,
            2.0742419322199e01,
            2.0716929307831e01,
            1.8190372677731e01,
            3.0488441450911e01,
            4.0457627599127e01,
            2.0967433681204e01,
            2.0717596735073e01,
            9.9580698415550e01,
            6.1846685567225e01,
            7.1021885273123e01,
            4.0423599454699e01,
            1.8347543119346e02,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(c6[0], ref_row0, rtol=1e-12, atol=1e-12)
    assert torch.allclose(c6, c6.T, rtol=1e-14, atol=1e-14)


def test_c6_d4_eeqbc_mb02() -> None:
    """
    Reference C6 matrix for MB16-43 02 with D4 + EEQBC (with_cn=False, with_q=True)
    from ``dftd/test/unit/test_model.f90:test_c6_d4_eeqbc_mb02``.
    """
    mol = get_structure("mb16_43", "02", dtype=torch.float64)
    cn = torch.zeros(mol.numbers.shape, dtype=torch.float64)
    q = charge.get_charges_eeqbc(mol.numbers, mol.positions)

    d4 = D4Model(qmodel="eeqbc")
    gw = d4.weights(mol, Environment(cn=cn, q=q))
    c6 = d4.coefficients(mol, gw).dense()

    ref_row0 = torch.tensor(
        [
            8.3927698731514e00,
            3.5694618468534e01,
            2.9939267651186e01,
            1.4756740088150e01,
            6.3400848091147e01,
            4.0968861748150e00,
            7.7602731112588e00,
            8.0399475458183e00,
            4.9479521337131e01,
            4.1163947967514e00,
            2.9811861798045e01,
            6.3182463323600e01,
            1.0790044024388e01,
            7.6731828442285e00,
            8.3342633309769e00,
            3.6347058629589e01,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(c6[0], ref_row0, rtol=1e-11, atol=1e-11)


def test_c6_d4_mb03() -> None:
    """
    Reference C6 matrix for MB16-43 03 with D4 + EEQ (with_cn=True, with_q=True)
    from ``dftd/test/unit/test_model.f90:test_c6_d4_mb03``.
    """
    mol = get_structure("mb16_43", "03", dtype=torch.float64)
    cn = ncoord.cn_d4(mol)
    q = charge.get_charges_eeq(mol.numbers, mol.positions)

    d4 = D4Model()
    gw = d4.weights(mol, Environment(cn=cn, q=q))
    c6 = d4.coefficients(mol, gw).dense()

    ref_row0 = torch.tensor(
        [
            2.6969645804632e01,
            2.8729803918609e01,
            8.0783557426273e00,
            1.2572598284646e01,
            3.9383726731882e01,
            6.0100611314024e01,
            2.8877446590256e01,
            7.9285493229103e00,
            7.5097624231199e00,
            7.1772575509757e00,
            1.9756649427858e01,
            6.3114791359146e01,
            2.9607901769799e01,
            5.3279952989770e00,
            3.4651507628618e01,
            8.1388552398630e00,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(c6[0], ref_row0, rtol=1e-11, atol=1e-11)


def test_c6_d4_gfn2_mb06() -> None:
    """
    Reference C6 matrix for MB16-43 06 with D4 + GFN2 external charges
    from ``dftd/test/unit/test_model.f90:test_c6_d4_mb07``.
    """
    mol = get_structure("mb16_43", "06", dtype=torch.float64)
    cn = ncoord.cn_d4(mol)
    qat = torch.tensor(
        [
            -1.57324183192355e-1,
            1.65228028672395e-1,
            3.22320366812437e-1,
            3.63579860576008e-2,
            4.85677294229959e-2,
            -3.59193331069774e-1,
            -1.93844259127416e-1,
            -3.86492630088218e-1,
            3.10104713332147e-1,
            8.34803229863654e-2,
            -3.62667644019899e-1,
            3.64142434058147e-1,
            3.34644499696670e-1,
            -4.69889877462762e-1,
            -1.89224201365947e-1,
            4.53790045287620e-1,
        ],
        dtype=torch.float64,
    )

    d4 = D4Model(qmodel="gfn2")
    gw = d4.weights(mol, Environment(cn=cn, q=qat))
    c6 = d4.coefficients(mol, gw).dense()

    ref_row0 = torch.tensor(
        [
            2.5899776752056e01,
            2.1718975287903e01,
            4.0896559404825e00,
            1.9068415125987e01,
            2.3793938067778e01,
            3.0452162987514e01,
            5.0890472676690e01,
            3.4184801377206e01,
            2.9794164793817e01,
            5.0038725736932e01,
            3.1096288315400e01,
            3.7613777716825e00,
            3.9882410743378e00,
            4.8634793174522e01,
            2.4997216709159e01,
            3.1670568222466e00,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(c6[0], ref_row0, rtol=1e-11, atol=1e-11)


# ===========================================================================
# 3. Polarizability conformance against Fortran test_model.f90
# ===========================================================================


def test_pol_d4_and_d4s_mb09() -> None:
    """
    Reference static dipole-dipole and quadrupole-quadrupole polarizabilities
    for MB16-43 09 from ``dftd/test/unit/test_model.f90:test_pol_d4_mb09``
    and ``test_pol_d4s_mb09``.
    """
    mol = get_structure("mb16_43", "09", dtype=torch.float64)
    cn = ncoord.cn_d4(mol)

    # D4: with_cn=True, with_q=False
    d4 = D4Model()
    env_d4 = Environment(cn=cn, q=torch.zeros_like(cn))
    alphadd_d4, alphaqq_d4 = d4.get_polarizabilities(mol, env_d4)

    ref_dd_d4 = torch.tensor(
        [
            2.7207807469636e00,
            2.7207697669106e00,
            2.7340295240471e00,
            2.7373201241818e00,
            2.0051462957754e01,
            2.7348043044018e00,
            6.9923962318719e00,
            9.2345547677329e00,
            2.7338175995796e00,
            2.7347959316428e00,
            2.3018401095949e01,
            2.7347926176379e00,
            1.5079046616252e01,
            3.5315073387745e00,
            2.7340548152913e00,
            9.2442543133947e00,
        ],
        dtype=torch.float64,
    )
    ref_qq_d4 = torch.tensor(
        [
            5.4615565063498e00,
            5.4615344655514e00,
            5.4881514257539e00,
            5.4947568086372e00,
            1.0065572291228e02,
            5.4897066803227e00,
            2.1710888390047e01,
            3.3655020315897e01,
            5.4877260193864e00,
            5.4896898732732e00,
            1.1239537836978e02,
            5.4896832209087e00,
            5.6234642849808e01,
            8.4341312278194e00,
            5.4882021941076e00,
            3.3690369979688e01,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(alphadd_d4, ref_dd_d4, rtol=1e-11, atol=1e-11)
    assert torch.allclose(alphaqq_d4, ref_qq_d4, rtol=1e-11, atol=1e-11)

    # D4S: with_cn=True, with_q=True
    q_eeq = charge.get_charges_eeq(mol.numbers, mol.positions)
    d4s = D4SModel()
    env_d4s = Environment(cn=cn, q=q_eeq)
    alphadd_d4s, alphaqq_d4s = d4s.get_polarizabilities(mol, env_d4s)

    ref_dd_d4s = torch.tensor(
        [
            2.0796638857659e00,
            2.0317560362159e00,
            2.1362479991008e00,
            2.4840420351444e00,
            3.5144617579624e01,
            2.1798399760895e00,
            8.7285804463012e00,
            1.0754720091309e01,
            2.1068749047750e00,
            2.2167964014782e00,
            2.4223007635619e01,
            2.1626725547918e00,
            1.5342832501948e01,
            4.0831999012216e00,
            2.2319948533110e00,
            1.0723200212288e01,
        ],
        dtype=torch.float64,
    )
    ref_qq_d4s = torch.tensor(
        [
            4.1746112173873e00,
            4.0784434435940e00,
            4.2881952805960e00,
            4.9863392903782e00,
            1.7642138613056e02,
            4.3756995919275e00,
            2.7101615753611e01,
            3.9195211059822e01,
            4.2292332291312e00,
            4.4498840354034e00,
            1.1827729029096e02,
            4.3412385859865e00,
            5.7218385751360e01,
            9.7517123688813e00,
            4.4803926324619e00,
            3.9080337934321e01,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(alphadd_d4s, ref_dd_d4s, rtol=1e-11, atol=1e-11)
    assert torch.allclose(alphaqq_d4s, ref_qq_d4s, rtol=1e-11, atol=1e-11)

    # Also check get_properties high-level helper
    props = get_properties(mol, model="d4s")
    assert props.alphadd is not None and props.alphaqq is not None
    assert torch.allclose(props.alphadd, ref_dd_d4s, rtol=1e-11, atol=1e-11)
    assert torch.allclose(props.alphaqq, ref_qq_d4s, rtol=1e-11, atol=1e-11)


# ===========================================================================
# 4. Analytical charge derivatives (with_dgwdq=True) vs autograd
# ===========================================================================


@pytest.mark.parametrize("model_cls", [D4Model, D4SModel, D4SrevModel])
@pytest.mark.parametrize("qmod", ["eeq", "gfn2"])
def test_dgwdq_analytical_vs_autograd(
    model_cls: type[D4Model | D4SModel | D4SrevModel], qmod: str
) -> None:
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    cn = ncoord.cn_d4(mol)
    q = charge.get_charges_eeq(mol.numbers, mol.positions).detach().requires_grad_(True)

    disp = model_cls(qmodel=qmod)
    gw, dgwdq = disp.weights(mol, Environment(cn=cn, q=q), with_dgwdq=True)

    # Check every reference slot derivative against autograd
    for iref in range(gw.shape[-1]):
        if gw.ndim == 2:
            loss = gw[..., iref].sum()
            (grad_q,) = torch.autograd.grad(
                loss, q, retain_graph=True, allow_unused=True
            )
            if grad_q is None:
                grad_q = torch.zeros_like(q)
            assert torch.allclose(dgwdq[..., iref], grad_q, rtol=1e-11, atol=1e-11)
        else:
            # D4S / D4Srev: gw has shape (..., nat, nel, 7)
            loss = gw[..., iref].sum()
            (grad_q,) = torch.autograd.grad(
                loss, q, retain_graph=True, allow_unused=True
            )
            if grad_q is None:
                grad_q = torch.zeros_like(q)
            assert torch.allclose(
                dgwdq[..., iref].sum(dim=-1), grad_q, rtol=1e-11, atol=1e-11
            )

    # Also verify get_atomic_c6(..., with_dc6dq=True) against autograd
    c6, dc6dq = disp.get_atomic_c6(mol, Environment(cn=cn, q=q), with_dc6dq=True)
    c6_plain = disp.get_atomic_c6(mol, Environment(cn=cn, q=q), with_dc6dq=False)
    assert torch.allclose(c6, c6_plain, rtol=1e-13, atol=1e-13)

    # For off-diagonal entries, d(sum_j C6_ij)/d(q_i) == sum_{j != i} dc6dq[i, j] + 2 * dc6dq[i, i]
    w_test = torch.linspace(0.5, 1.5, mol.numbers.shape[0], dtype=torch.float64)
    loss_c6 = (c6 * w_test.unsqueeze(0)).sum()
    (grad_c6_q,) = torch.autograd.grad(loss_c6, q)
    # d/dq_i of sum_{a,b} C6_{ab} w_b = sum_b dc6dq[i,b]*w_b + sum_a dc6dq[i,a]*w_i
    expected_grad = (dc6dq * w_test.unsqueeze(0)).sum(dim=-1) + dc6dq.sum(dim=-1) * w_test
    assert torch.allclose(grad_c6_q, expected_grad, rtol=1e-11, atol=1e-11)


def test_charge_model_aliases() -> None:
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    param = _conformance_param(torch.float64)

    e_eeq = dftd4(mol, param, qmodel="eeq")
    e_eeq2019 = dftd4(mol, param, qmodel="eeq2019")
    assert torch.allclose(e_eeq, e_eeq2019, rtol=1e-14, atol=1e-14)

    e_eeqbc = dftd4(mol, param, qmodel="eeqbc")
    e_eeqbc2025 = dftd4(mol, param, qmodel="eeqbc2025")
    assert torch.allclose(e_eeqbc, e_eeqbc2025, rtol=1e-14, atol=1e-14)


# ===========================================================================
# 5. Fortran dispersion energy regressions (D4, D4S, D4Srev)
# ===========================================================================


@pytest.mark.parametrize(
    "mb_id,model_name,qmod,param_dict,ref_energy",
    [
        (
            "01",
            "d4",
            "eeq",
            {"s6": 1.0, "alp": 16.0, "s8": 0.95948085, "a1": 0.38574991, "a2": 4.80688534},
            -1.8578752883363366e-02,
        ),
        (
            "01",
            "d4",
            "eeqbc",
            {"s6": 1.0, "alp": 16.0, "s8": 0.95948085, "a1": 0.38574991, "a2": 4.80688534},
            -1.8738354839506779e-02,
        ),
        (
            "01",
            "d4s",
            "eeq",
            {"s6": 1.0, "alp": 16.0, "s8": 0.95948085, "a1": 0.38574991, "a2": 4.80688534},
            -1.9870451633183694e-02,
        ),
        (
            "01",
            "d4s",
            "eeqbc",
            {"s6": 1.0, "alp": 16.0, "s8": 0.95948085, "a1": 0.38574991, "a2": 4.80688534},
            -1.9901335934314583e-02,
        ),
        (
            "02",
            "d4",
            "eeq",
            {"s6": 1.0, "alp": 16.0, "s8": 1.69460052, "a1": 0.28904684, "a2": 4.13407323},
            -8.9181168937810723e-02,
        ),
        (
            "02",
            "d4s",
            "eeq",
            {"s6": 1.0, "alp": 16.0, "s8": 1.69460052, "a1": 0.28904684, "a2": 4.13407323},
            -0.10354699373838475,
        ),
        (
            "03",
            "d4",
            "eeq",
            {"s6": 1.0, "alp": 16.0, "s8": 1.91130849, "a1": 0.43332851, "a2": 4.56986797},
            -2.4695638764787930e-02,
        ),
        (
            "03",
            "d4s",
            "eeq",
            {"s6": 1.0, "alp": 16.0, "s8": 1.91130849, "a1": 0.43332851, "a2": 4.56986797},
            -2.9321970435591389e-02,
        ),
        (
            "04",
            "d4",
            "eeq",
            {"s6": 0.82, "alp": 16.0, "s8": -0.34639127, "a1": 0.41080636, "a2": 3.83878274},
            -9.5128100471706181e-03,
        ),
        (
            "04",
            "d4s",
            "eeq",
            {"s6": 0.82, "alp": 16.0, "s8": -0.34639127, "a1": 0.41080636, "a2": 3.83878274},
            -1.0003412660121589e-02,
        ),
        (
            "09",
            "d4",
            "eeq",
            {
                "s6": 1.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 1.31183787,
                "a1": 0.46169493,
                "a2": 3.15711757,
                "a13": 0.46169493,
                "a23": 3.15711757,
                "alp3": 16.0,
            },
            -4.5140422485299259e-02,
        ),
        (
            "09",
            "d4s",
            "eeq",
            {
                "s6": 1.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 1.31183787,
                "a1": 0.46169493,
                "a2": 3.15711757,
                "a13": 0.46169493,
                "a23": 3.15711757,
                "alp3": 16.0,
            },
            -4.7626123029052128e-02,
        ),
        (
            "10",
            "d4",
            "eeq",
            {
                "s6": 0.56,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 0.94633372,
                "a1": 0.42907301,
                "a2": 5.18802602,
                "a13": 0.42907301,
                "a23": 5.18802602,
                "alp3": 16.0,
            },
            -9.6812427202205668e-03,
        ),
        (
            "10",
            "d4s",
            "eeq",
            {
                "s6": 0.56,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 0.94633372,
                "a1": 0.42907301,
                "a2": 5.18802602,
                "a13": 0.42907301,
                "a23": 5.18802602,
                "alp3": 16.0,
            },
            -1.1490632914927183e-02,
        ),
        (
            "11",
            "d4",
            "eeq",
            {
                "s6": 1.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 1.27677253,
                "a1": 0.38128670,
                "a2": 4.91698883,
                "a13": 0.38128670,
                "a23": 4.91698883,
                "alp3": 16.0,
            },
            -1.7460015867914524e-02,
        ),
        (
            "11",
            "d4s",
            "eeq",
            {
                "s6": 1.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 1.27677253,
                "a1": 0.38128670,
                "a2": 4.91698883,
                "a13": 0.38128670,
                "a23": 4.91698883,
                "alp3": 16.0,
            },
            -2.0329508047867848e-02,
        ),
        (
            "12",
            "d4",
            "eeq",
            {
                "s6": 1.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 1.27701162,
                "a1": 0.40554715,
                "a2": 4.63323074,
                "a13": 0.40554715,
                "a23": 4.63323074,
                "alp3": 16.0,
            },
            -2.5712178361964221e-02,
        ),
        (
            "12",
            "d4s",
            "eeq",
            {
                "s6": 1.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "s8": 1.27701162,
                "a1": 0.40554715,
                "a2": 4.63323074,
                "a13": 0.40554715,
                "a23": 4.63323074,
                "alp3": 16.0,
            },
            -2.6528352555725724e-02,
        ),
    ],
)
def test_d4_d4s_fortran_regressions(
    mb_id: str,
    model_name: str,
    qmod: str,
    param_dict: dict[str, float],
    ref_energy: float,
) -> None:
    mol = get_structure("mb16_43", mb_id, dtype=torch.float64)
    param = {k: torch.tensor(v, dtype=torch.float64) for k, v in param_dict.items()}
    energy = dftd(
        mol,
        param,
        model=model_name,
        qmodel=qmod,
    )
    assert torch.allclose(
        energy.sum(),
        torch.tensor(ref_energy, dtype=torch.float64),
        rtol=1e-11,
        atol=1e-12,
    )


def test_r2scan3c_d4_and_d4s() -> None:
    """
    r2SCAN-3c custom ga=2.0, gc=1.0 regression from
    ``dftd/test/unit/test_dftd.f90:test_r2scan3c_mb01`` and ``test_r2scan3c_d4s_mb01``.
    """
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    param = {
        "s6": torch.tensor(1.00, dtype=torch.float64),
        "s9": torch.tensor(2.00, dtype=torch.float64),
        "rs9": torch.tensor(1.0, dtype=torch.float64),
        "alp": torch.tensor(16.0, dtype=torch.float64),
        "s8": torch.tensor(0.00, dtype=torch.float64),
        "a1": torch.tensor(0.42, dtype=torch.float64),
        "a2": torch.tensor(5.65, dtype=torch.float64),
        "a13": torch.tensor(0.42, dtype=torch.float64),
        "a23": torch.tensor(5.65, dtype=torch.float64),
        "alp3": torch.tensor(16.0, dtype=torch.float64),
    }

    e_d4 = dftd4(mol, param, ga=2.0, gc=1.0)
    assert torch.allclose(
        e_d4.sum(),
        torch.tensor(-5.7825025556386862e-03, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )

    e_d4s = dftd4s(mol, param, ga=2.0, gc=1.0)
    assert torch.allclose(
        e_d4s.sum(),
        torch.tensor(-6.1176284425895639e-03, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_amf3_and_actinides_d4_d4s() -> None:
    """
    AmF3 and 17-atom actinide cluster (Z=87..103) regressions from
    ``dftd/test/unit/test_dftd.f90:test_tpsshd4atm_amf3``,
    ``test_tpsshd4satm_amf3``, ``test_actinides_d4``, and ``test_actinides_d4s``.
    """
    amf3 = _amf3_structure(torch.float64)
    param_tpssh = {
        "s6": torch.tensor(1.0, dtype=torch.float64),
        "s9": torch.tensor(1.0, dtype=torch.float64),
        "rs9": torch.tensor(1.0, dtype=torch.float64),
        "alp": torch.tensor(16.0, dtype=torch.float64),
        "s8": torch.tensor(1.85897750, dtype=torch.float64),
        "a1": torch.tensor(0.44286966, dtype=torch.float64),
        "a2": torch.tensor(4.60230534, dtype=torch.float64),
        "a13": torch.tensor(0.44286966, dtype=torch.float64),
        "a23": torch.tensor(4.60230534, dtype=torch.float64),
        "alp3": torch.tensor(16.0, dtype=torch.float64),
    }
    e_amf3_d4 = dftd4(amf3, param_tpssh)
    assert torch.allclose(
        e_amf3_d4.sum(),
        torch.tensor(-2.4882226918209061e-03, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )

    e_amf3_d4s = dftd4s(amf3, param_tpssh)
    assert torch.allclose(
        e_amf3_d4s.sum(),
        torch.tensor(-2.5523952752362009e-03, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )

    act = _actinides_d4(torch.float64)
    param_pbe = {
        "s6": torch.tensor(1.0, dtype=torch.float64),
        "alp": torch.tensor(16.0, dtype=torch.float64),
        "s8": torch.tensor(0.95948085, dtype=torch.float64),
        "a1": torch.tensor(0.38574991, dtype=torch.float64),
        "a2": torch.tensor(4.80688534, dtype=torch.float64),
    }
    e_act_d4 = dftd4(act, param_pbe)
    assert torch.allclose(
        e_act_d4.sum(),
        torch.tensor(-0.17966420554540324, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )

    e_act_d4s = dftd4s(act, param_pbe)
    assert torch.allclose(
        e_act_d4s.sum(),
        torch.tensor(-0.18578133252612403, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "model_name,ref_mol_energy,ref_pbc_energy",
    [
        ("d4", -1.8528083872689883e-02, -6.3491389917830363e-02),
        ("d4s", -1.9813838966238666e-02, -6.4968999904905331e-02),
        ("d4srev", -1.9073487468982180e-02, -6.5163658273002273e-02),
    ],
)
def test_conformance_d4_d4s_d4srev_mol_and_pbc(
    model_name: str, ref_mol_energy: float, ref_pbc_energy: float
) -> None:
    """
    Frozen molecular (MB16-43 01) and 3D periodic (X23 acetic) reference
    energies for D4, D4S, and D4Srev from ``dftd/test/unit/test_conformance.f90``.
    """
    param = _conformance_param(torch.float64)
    cutoff = Cutoff(cn=20.0, disp2=40.0, disp3=15.0)

    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    e_mol = dftd(
        mol,
        param,
        model=model_name,
        cutoff=cutoff,
    )
    assert torch.allclose(
        e_mol.sum(),
        torch.tensor(ref_mol_energy, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )

    pbc = get_structure("x23", "acetic", dtype=torch.float64)
    e_pbc = dftd(
        pbc,
        param,
        model=model_name,
        cutoff=cutoff,
    )
    # Note: tad_multicharge wraps periodic positions into the unit cell before
    # building Wigner-Seitz Ewald sums (unlike multicharge 0.5.0, which changes
    # periodic EEQ charges by ~5e-7 and energies by ~1e-8..4e-8).
    assert torch.allclose(
        e_pbc.sum(),
        torch.tensor(ref_pbc_energy, dtype=torch.float64),
        rtol=1e-6,
        atol=5e-8,
    )


# ===========================================================================
# 6. Three-body charge independence and external charges
# ===========================================================================


@pytest.mark.parametrize(
    "model_cls,model_name",
    [
        (D4Model, "d4"),
        (D4SModel, "d4s"),
        (D4SrevModel, "d4srev"),
    ],
)
def test_threebody_charge_independence(
    model_cls: type[D4Model | D4SModel | D4SrevModel], model_name: str
) -> None:
    """
    In D4, D4S, and D4Srev, the three-body ATM term is evaluated at neutral
    partial charges (q = 0), so dE_3b / dq == 0 while dE_2b / dq != 0.
    """
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    param_3b = _conformance_param(torch.float64)
    param_2b = {k: v for k, v in param_3b.items() if k not in ("s9", "rs9", "a13", "a23", "alp3")}
    cutoff = Cutoff(cn=20.0, disp2=40.0, disp3=15.0)

    q1 = torch.zeros(mol.numbers.shape, dtype=torch.float64, requires_grad=True)
    q2 = torch.full(mol.numbers.shape, 0.25, dtype=torch.float64, requires_grad=True)

    e_tot_q1 = dftd(mol, param_3b, model=model_name, q=q1, cutoff=cutoff)
    e_2b_q1 = dftd(mol, param_2b, model=model_name, q=q1, cutoff=cutoff)
    e_3b_q1 = e_tot_q1 - e_2b_q1

    e_tot_q2 = dftd(mol, param_3b, model=model_name, q=q2, cutoff=cutoff)
    e_2b_q2 = dftd(mol, param_2b, model=model_name, q=q2, cutoff=cutoff)
    e_3b_q2 = e_tot_q2 - e_2b_q2

    assert torch.allclose(e_3b_q1, e_3b_q2, rtol=1e-14, atol=1e-14)

    # Total energy with external charges has non-zero dE/dq from 2-body term
    (dedq,) = torch.autograd.grad(e_tot_q1.sum(), q1)
    assert torch.any(dedq.abs() > 1e-6)


# ===========================================================================
# 7. Element and charge-model validation
# ===========================================================================


@pytest.mark.parametrize("model_cls", [D4Model, D4SModel, D4SrevModel])
def test_d4_element_and_charge_validation(
    model_cls: type[D4Model | D4SModel | D4SrevModel],
) -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float64)
    param = _conformance_param(torch.float64)
    disp = model_cls()

    # Gap elements Z=104..111 (e.g., Rf=104, Db=105) must raise ValueError
    with pytest.raises(ValueError, match="Unsupported atomic number"):
        disp(Structure(numbers=torch.tensor([104, 105]), positions=pos), param)

    # Elements beyond Og (Z > 118) must raise ValueError
    with pytest.raises(ValueError, match="Unsupported atomic number"):
        disp(Structure(numbers=torch.tensor([1, 119]), positions=pos), param)

    # Superheavy Z=118 (Og2) is supported by D4 and D4S with external charges,
    # whereas D4Srev only defines tanh charge-scaling parameters up to Z=103
    og2 = Structure(numbers=torch.tensor([118, 118], dtype=torch.long), positions=pos)
    og2_q = torch.zeros(2, dtype=torch.float64)
    if model_cls is D4SrevModel:
        with pytest.raises(ValueError, match="Unsupported atomic number"):
            disp(og2, param, q=og2_q)
    else:
        e_og2 = disp(og2, param, q=og2_q)
        assert torch.isfinite(e_og2).all() and (e_og2.sum() < 0.0)

        # Superheavy Z=118 without external charges fails in EEQ/EEQBC (max Z=103)
        with pytest.raises(ValueError, match="unsupported element"):
            disp(og2, param)

    # qmodel="gfn2" without external charges raises ValueError("requires external charges")
    h2 = Structure(numbers=torch.tensor([1, 1], dtype=torch.long), positions=pos)
    with pytest.raises(ValueError, match="requires external charges"):
        model_cls(qmodel="gfn2")(h2, param)


# ===========================================================================
# 8. Dense vs Sparse NeighborList, Batching, Autograd & Transforms
# ===========================================================================


@pytest.mark.parametrize("model_name", ["d4", "d4s", "d4srev"])
def test_d4_dense_vs_sparse_and_batching(model_name: str) -> None:
    mol1 = get_structure("mb16_43", "01", dtype=torch.float64)
    amf3 = _amf3_structure(torch.float64)
    param = _conformance_param(torch.float64)
    cutoff = Cutoff(cn=20.0, disp2=40.0, disp3=15.0)

    e_dense = dftd(
        mol1,
        param,
        model=model_name,
        cutoff=cutoff,
        pairs=DensePairs(),
    )
    e_sparse = dftd(
        mol1,
        param,
        model=model_name,
        cutoff=cutoff,
        pairs=SparsePairs.build(mol1, cutoff, three_body=True),
    )
    assert torch.allclose(e_dense, e_sparse, rtol=1e-13, atol=1e-13)

    # Batched padded calculation matches individual evaluations
    batch = pack_structures([mol1, amf3])
    e_batch = dftd(
        batch,
        param,
        model=model_name,
        cutoff=cutoff,
    )
    e_amf3 = dftd(
        amf3,
        param,
        model=model_name,
        cutoff=cutoff,
    )
    assert torch.allclose(
        e_batch[0, : mol1.numbers.shape[0]], e_dense, rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(e_batch[1, :4], e_amf3, rtol=1e-12, atol=1e-12)
    assert torch.all(e_batch[1, 4:] == 0.0)


@pytest.mark.parametrize("model_name", ["d4", "d4s", "d4srev"])
def test_d4_autograd_and_transforms(model_name: str) -> None:
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.00000000, 0.00000000, 0.22170000],
            [0.00000000, 1.43090000, -0.88670000],
            [0.00000000, -1.43090000, -0.88670000],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    q = torch.tensor([-0.6, 0.3, 0.3], dtype=torch.float64, requires_grad=True)
    param = _conformance_param(torch.float64)

    def fn(pos: torch.Tensor, q_in: torch.Tensor) -> torch.Tensor:
        return dftd(
            Structure(numbers=numbers, positions=pos),
            param,
            model=model_name,
            q=q_in,
        ).sum()

    assert torch.autograd.gradcheck(fn, (positions, q), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(
        fn, (positions, q), eps=1e-5, atol=1e-4, rtol=1e-3
    )

    # Also check gradcheck with internal EEQ charges
    def fn_eeq(pos: torch.Tensor) -> torch.Tensor:
        return dftd(
            Structure(numbers=numbers, positions=pos),
            param,
            model=model_name,
            qmodel="eeq",
        ).sum()

    assert torch.autograd.gradcheck(fn_eeq, (positions,), eps=1e-6, atol=1e-5, rtol=1e-4)

    # Check jacrev, jacfwd, vmap
    j_rev = torch.func.jacrev(fn_eeq)(positions)
    j_fwd = torch.func.jacfwd(fn_eeq)(positions)
    assert torch.allclose(j_rev, j_fwd, rtol=1e-10, atol=1e-10)

    pos_batch = torch.stack([positions.detach(), positions.detach() * 1.05], dim=0)
    vmap_e = torch.func.vmap(fn_eeq)(pos_batch)
    direct_e = dftd(
        Structure(
            numbers=numbers.unsqueeze(0).expand(2, -1),
            positions=pos_batch,
        ),
        param,
        model=model_name,
    ).sum(dim=-1)
    assert torch.allclose(vmap_e, direct_e, rtol=1e-12, atol=1e-12)
