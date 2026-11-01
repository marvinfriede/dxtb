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
Tests for Milestone B: DFT-D3 and DFT-D3S models, reference tables, C6
factorization, damping variants, periodic cells, batches, autograd, and
transforms.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.units import AA2AU

from tad_dftd import (
    Cutoff,
    D3Model,
    D3SModel,
    DensePairs,
    Reference,
    SparsePairs,
    damping,
    data,
    dftd,
    dftd3,
    dftd3s,
    dispersion,
    dispersion2,
    dispersion3,
    model,
)


def _metal_complex(dtype: torch.dtype = torch.float64) -> Structure:
    """
    14-atom transition-metal and heavy main-group complex from
    ``dftd/test/unit/test_dftd.f90:get_metal_complex``.
    """
    numbers = torch.tensor(
        [46, 72, 3, 28, 74, 79, 6, 7, 8, 1, 1, 16, 17, 35], dtype=torch.long
    )
    xyz_aa = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [3.600, 0.400, -0.300],
            [-2.100, 1.000, 0.500],
            [0.300, -3.000, 0.800],
            [-0.500, 3.400, -1.200],
            [3.900, -3.100, 1.900],
            [1.600, 1.500, 1.900],
            [-1.700, -1.600, 1.400],
            [2.000, -1.800, -1.700],
            [0.200, 0.900, 1.900],
            [-1.000, -0.200, -2.000],
            [-3.400, 0.200, -1.800],
            [1.500, 4.200, 0.900],
            [-3.000, -3.500, -0.700],
        ],
        dtype=dtype,
    )
    return Structure(numbers=numbers, positions=xyz_aa * AA2AU)


def _actinides_d3(dtype: torch.dtype = torch.float64) -> Structure:
    """
    17-atom actinide cluster (Z=87..103) from
    ``dftd/test/unit/test_dftd.f90:test_actinides_d3``.
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


def _actinides_d3s(dtype: torch.dtype = torch.float64) -> Structure:
    """
    10-atom Fr-Pu + C + H cluster from
    ``dftd/test/unit/test_dftd.f90:test_actinides_d3s``.
    """
    numbers = torch.tensor(
        [87, 88, 89, 90, 91, 92, 93, 94, 6, 1], dtype=torch.long
    )
    xyz_aa = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [3.600, 0.400, -0.300],
            [-2.100, 1.000, 0.500],
            [0.300, -3.000, 0.800],
            [-0.500, 3.400, -1.200],
            [3.900, -3.100, 1.900],
            [1.600, 1.500, 1.900],
            [-1.700, -1.600, 1.400],
            [2.000, -1.800, -1.700],
            [0.200, 0.900, 1.900],
        ],
        dtype=dtype,
    )
    return Structure(numbers=numbers, positions=xyz_aa * AA2AU)


def test_reference_load_d3_and_d3s() -> None:
    ref_default = Reference()
    assert ref_default.cn.shape == (104, 7)
    assert ref_default.c6.shape == (104, 104, 7, 7)

    ref_d3 = Reference.load(dtype=torch.float64)
    ref_d3s = Reference.load_d3s(dtype=torch.float64)

    assert ref_d3.cn.shape == (104, 7)
    assert ref_d3.c6.shape == (104, 104, 7, 7)
    assert ref_d3s.cn.shape == (104, 7)
    assert ref_d3s.c6.shape == (104, 104, 7, 7)

    # Up to Rn (Z=86), D3 and D3S reference tables are identical
    assert torch.allclose(ref_d3.cn[:87], ref_d3s.cn[:87])
    assert torch.allclose(ref_d3.c6[:87, :87], ref_d3s.c6[:87, :87])

    # For Fr-Pu (Z=87..94), D3S uses the original D3 publication references
    assert not torch.allclose(ref_d3.c6[87:95, 87:95], ref_d3s.c6[87:95, 87:95])
    # Symmetry of D3S C6 reference table
    assert torch.allclose(
        ref_d3s.c6, ref_d3s.c6.permute(1, 0, 3, 2), atol=1e-14
    )


@pytest.mark.parametrize(
    "record,param,damping_obj,ref_energy",
    [
        (
            "01",
            {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
            None,
            -1.7882220155186028e-2,
        ),
        (
            "02",
            {"s6": 1.0, "s8": 2.2609, "a1": 0.5545, "a2": 3.2297},
            None,
            -5.9503371130070551e-2,
        ),
        (
            "03",
            {"s6": 1.0, "s8": 1.9435, "a1": 0.4535, "a2": 4.4752},
            None,
            -2.8684059240803347e-2,
        ),
        (
            "04",
            {"s6": 0.82, "s8": 0.2904, "a1": 0.0, "a2": 7.3141},
            None,
            -1.2270948604225146e-2,
        ),
        (
            "09",
            {
                "s6": 1.0,
                "s8": 0.514,
                "rs6": 0.872,
                "rs8": 1.0,
                "alp": 14.0,
                "damping": "zero",
            },
            None,
            -2.0178760785797962e-2,
        ),
        (
            "17",
            {
                "s6": 0.5,
                "s8": 0.2130,
                "a1": 0.0,
                "a2": 6.0519,
                "s9": 1.0,
                "rs9": 4.0 / 3.0,
                "alp": 14.0,
                "a13": 1.0,
                "a23": 0.0,
                "alp3": 16.0,
            },
            None,
            -1.3592755832923201e-2,
        ),
        (
            "25",
            {
                "s6": 1.0,
                "s8": 1.682,
                "rs6": 1.094,
                "rs8": 1.0,
                "s9": 1.0,
                "rs9": 4.0 / 3.0,
                "alp": 14.0,
                "a13": 1.0,
                "a23": 0.0,
                "alp3": 16.0,
                "damping": "zero",
            },
            None,
            -1.8876259384422459e-2,
        ),
        (
            "33",
            {
                "s6": 1.0,
                "s8": 0.0,
                "rs6": 2.340218,
                "rs8": 1.0,
                "bet": 0.129434,
                "s9": 1.0,
                "rs9": 4.0 / 3.0,
                "alp": 14.0,
                "a13": 1.0,
                "a23": 0.0,
                "alp3": 16.0,
                "damping": "mzero",
            },
            None,
            -6.4839784232922359e-2,
        ),
        (
            "37",
            {
                "s6": 0.97388,
                "s8": 0.0,
                "a1": 0.150,
                "a2": 4.25,
                "bet": 6.0,
                "s9": 1.0,
                "rs9": 4.0 / 3.0,
                "alp": 14.0,
                "a13": 1.0,
                "a23": 0.0,
                "alp3": 16.0,
                "damping": "optpower",
            },
            None,
            -2.7952861273999385e-2,
        ),
        (
            "01",
            {
                "s6": 1.0,
                "s8": 0.0,
                "a2": 6.25,
                "a3": 0.86,
                "a4": 2.5,
                "damping": "cso",
            },
            None,
            -3.8002950817452329e-2,
        ),
        (
            "02",
            {
                "s6": 1.0,
                "s8": 0.0,
                "a2": 6.25,
                "a3": 0.24,
                "a4": 2.5,
                "s9": 1.0,
                "rs9": 4.0 / 3.0,
                "alp": 14.0,
                "a13": 1.0,
                "a23": 0.0,
                "alp3": 16.0,
                "damping": "cso",
            },
            None,
            -3.8229908827789316e-2,
        ),
    ],
)
def test_d3_fortran_regression(
    record: str,
    param: dict[str, float | str],
    damping_obj: damping.Damping | None,
    ref_energy: float,
) -> None:
    mol = get_structure("mb16_43", record, dtype=torch.float64)
    e_atoms = dftd3(mol, param, damping=damping_obj)
    e_disp = dftd(mol, param, model="d3", damping=damping_obj)
    assert torch.allclose(e_atoms, e_disp, atol=1e-14, rtol=1e-14)
    assert pytest.approx(ref_energy, rel=1e-11, abs=1e-12) == float(
        e_atoms.sum()
    )


def test_actinides_d3_fortran_regression() -> None:
    mol = _actinides_d3()
    param = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}
    e_atoms = dftd3(mol, param)
    assert pytest.approx(-1.4131143363689097e-1, rel=1e-11, abs=1e-12) == float(
        e_atoms.sum()
    )


def test_d3s_metals_fortran_regression() -> None:
    mol = _metal_complex()

    # BLYP-D3S(BJ)
    param_bj = {"s6": 1.0, "s8": 2.6996, "a1": 0.4298, "a2": 4.2359}
    e_bj = dftd3s(mol, param_bj)
    e_bj_dispatch = dftd(mol, param_bj, model="d3s")
    assert torch.allclose(e_bj, e_bj_dispatch, atol=1e-14, rtol=1e-14)
    assert pytest.approx(
        -9.1416161240587700e-2, rel=1e-9, abs=1e-9
    ) == float(e_bj.sum())

    # BLYP-D3S(0)
    param_zero = {
        "s6": 1.0,
        "s8": 1.682,
        "rs6": 1.094,
        "rs8": 1.0,
        "alp": 14.0,
        "damping": "zero",
    }
    e_zero = dftd3s(mol, param_zero)
    assert pytest.approx(
        -8.1471285735904840e-2, rel=1e-9, abs=1e-9
    ) == float(e_zero.sum())


def test_actinides_d3s_fortran_regression() -> None:
    mol = _actinides_d3s()

    # BLYP-D3S(BJ)
    param_bj = {"s6": 1.0, "s8": 2.6996, "a1": 0.4298, "a2": 4.2359}
    e_bj = dftd3s(mol, param_bj)
    assert pytest.approx(
        -9.3937017995884e-2, rel=1e-9, abs=1e-9
    ) == float(e_bj.sum())

    # BLYP-D3S(0)
    param_zero = {
        "s6": 1.0,
        "s8": 1.682,
        "rs6": 1.094,
        "rs8": 1.0,
        "alp": 14.0,
        "damping": "zero",
    }
    e_zero = dftd3s(mol, param_zero)
    assert pytest.approx(
        -4.5984628022162e-2, rel=1e-9, abs=1e-9
    ) == float(e_zero.sum())


def test_d3s_default_wf_matches_d3() -> None:
    """
    D3S with all weighting factors set to 4.0 and D3 reference table is
    identical to D3 in energy and nuclear gradient (``test_d3s_default_wf``).
    """
    mol = _metal_complex()
    pos3 = mol.positions.clone().detach().requires_grad_(True)
    pos3s = mol.positions.clone().detach().requires_grad_(True)
    mol3 = Structure(numbers=mol.numbers, positions=pos3)
    mol3s = Structure(numbers=mol.numbers, positions=pos3s)

    param = {
        "s6": 1.0,
        "s8": 2.6996,
        "a1": 0.4298,
        "a2": 4.2359,
        "s9": 1.0,
    }
    wf4 = torch.full((95, 95), 4.0, dtype=torch.float64)
    wf4[0, :] = 0.0
    wf4[:, 0] = 0.0

    e3 = dftd3(mol3, param).sum()
    e3s = dftd3s(
        mol3s, param, wfpair_table=wf4, ref=Reference.load(dtype=torch.float64)
    ).sum()

    e3.backward()
    e3s.backward()

    assert torch.allclose(e3, e3s, atol=1e-13, rtol=1e-13)
    assert pos3.grad is not None and pos3s.grad is not None
    assert torch.allclose(pos3.grad, pos3s.grad, atol=1e-12, rtol=1e-12)


def test_d3s_pair_weights_and_properties() -> None:
    """
    Verify symmetry, positivity, and Pd-Hf difference of D3S C6 coefficients
    (``test_d3s_pair_weights``), plus ``get_properties`` and ``AtomicC6``.
    """
    mol = _metal_complex()
    d3_mod = D3Model()
    d3s_mod = D3SModel()

    props3 = d3_mod.get_properties(mol)
    props3s = d3s_mod.get_properties(mol)

    assert props3.cn is not None and props3s.cn is not None
    assert torch.allclose(props3.cn, props3s.cn, atol=1e-14)

    c6_3 = props3.c6
    c6_3s = props3s.c6
    assert c6_3 is not None and c6_3s is not None
    assert torch.allclose(c6_3s, c6_3s.T, atol=1e-12)
    assert torch.all(c6_3s > 0.0)

    # Atom 0 is Pd (Z=46), Atom 1 is Hf (Z=72)
    assert abs(float(c6_3s[0, 1] - c6_3[0, 1])) > 1e-3

    # Check AtomicC6.pair() vs AtomicC6.dense() for both D3 and D3S
    env = d3s_mod.environment(mol)
    ac6_3s = d3s_mod.coefficients(mol, env)
    idx_i, idx_j = torch.meshgrid(
        torch.arange(mol.numbers.shape[0]),
        torch.arange(mol.numbers.shape[0]),
        indexing="ij",
    )
    pair_vals = ac6_3s.pair(idx_i.reshape(-1), idx_j.reshape(-1)).reshape_as(
        c6_3s
    )
    assert torch.allclose(pair_vals, c6_3s, atol=1e-12)

    # Also test AtomicC6.dense() on > 16 unique elements (nel > 16 branch)
    mol_17 = _actinides_d3s()
    nums_many = torch.arange(1, 20, dtype=torch.long)
    pos_many = torch.arange(57, dtype=torch.float64).reshape(19, 3)
    mol_many = Structure(numbers=nums_many, positions=pos_many)
    env_many = d3s_mod.environment(mol_many)
    ac6_many = d3s_mod.coefficients(mol_many, env_many)
    dense_many = ac6_many.dense()
    ii, jj = torch.meshgrid(
        torch.arange(19), torch.arange(19), indexing="ij"
    )
    pair_many = ac6_many.pair(ii.reshape(-1), jj.reshape(-1)).reshape(19, 19)
    assert torch.allclose(dense_many, pair_many, atol=1e-12)
    assert torch.allclose(dense_many, dense_many.T, atol=1e-12)
    assert mol_17.numbers.shape[0] == 10


def test_unsupported_elements() -> None:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float64
    )
    param = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}

    # D3 supports up to Z=103; Z=105 raises ValueError
    with pytest.raises(ValueError, match="Unsupported atomic number"):
        dftd3(
            Structure(
                numbers=torch.tensor([1, 105]), positions=positions
            ),
            param,
        )

    # D3S supports up to Z=94 (Pu); Z=95 (Am) raises ValueError
    with pytest.raises(ValueError, match="Unsupported atomic number"):
        dftd3s(
            Structure(
                numbers=torch.tensor([1, 95]), positions=positions
            ),
            param,
        )


def test_batch_and_padding_invariance() -> None:
    mol1 = get_structure("mb16_43", "01", dtype=torch.float64)
    mol2 = _actinides_d3s(dtype=torch.float64)
    batch = pack_structures([mol1, mol2])
    batch_pos = batch.positions.clone().detach().requires_grad_(True)
    batch_grad = Structure(numbers=batch.numbers, positions=batch_pos)

    param = {
        "s6": 1.0,
        "s8": 0.7875,
        "a1": 0.4289,
        "a2": 4.4407,
        "s9": 1.0,
    }

    for fn in (dftd3, dftd3s):
        e1 = fn(mol1, param)
        e2 = fn(mol2, param)
        eb = fn(batch_grad, param)

        assert torch.allclose(eb[0, : mol1.numbers.shape[0]], e1, atol=1e-12)
        assert torch.allclose(eb[1, : mol2.numbers.shape[0]], e2, atol=1e-12)
        # Padding atoms (Z=0) must have exactly zero energy
        assert torch.all(eb[1, mol2.numbers.shape[0] :] == 0.0)

        if batch_pos.grad is not None:
            batch_pos.grad.zero_()
        eb.sum().backward()
        assert batch_pos.grad is not None
        assert not torch.isnan(batch_pos.grad).any()
        # Padding atoms (Z=0) must have exactly zero gradient
        assert torch.all(batch_pos.grad[1, mol2.numbers.shape[0] :] == 0.0)


def test_periodic_dense_vs_sparse_and_checkpoint() -> None:
    """
    Periodic cell evaluation: DensePairs and SparsePairs (and checkpoint=True)
    must agree for both D3 and D3S, with 2-body and 3-body ATM terms.
    """
    numbers = torch.tensor([14, 14, 6, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5],
            [1.2, 3.8, 0.9],
            [3.9, 1.1, 4.1],
        ],
        dtype=torch.float64,
    )
    lattice = torch.tensor(
        [
            [8.0, 0.2, 0.0],
            [0.0, 8.2, 0.1],
            [0.1, 0.0, 8.5],
        ],
        dtype=torch.float64,
    )
    cell = Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=torch.tensor([True, True, True]),
    )
    cutoff = Cutoff(cn=12.0, disp2=14.0, disp3=10.0)
    param = {
        "s6": 1.0,
        "s8": 0.7875,
        "a1": 0.4289,
        "a2": 4.4407,
        "s9": 1.0,
    }

    dense_pairs = DensePairs().fill(cell, cutoff, three_body=True)
    sparse_pairs = SparsePairs.build(cell, cutoff, three_body=True)

    for fn in (dftd3, dftd3s):
        pos_d = positions.clone().detach().requires_grad_(True)
        pos_s = positions.clone().detach().requires_grad_(True)
        pos_c = positions.clone().detach().requires_grad_(True)

        cell_d = cell.replace(positions=pos_d)
        cell_s = cell.replace(positions=pos_s)
        cell_c = cell.replace(positions=pos_c)

        ed = fn(cell_d, param, pairs=dense_pairs, cutoff=cutoff)
        es = fn(cell_s, param, pairs=sparse_pairs, cutoff=cutoff)
        ec = fn(
            cell_c,
            param,
            pairs=sparse_pairs,
            cutoff=cutoff,
            checkpoint=True,
        )

        assert torch.allclose(ed, es, atol=1e-12, rtol=1e-12)
        assert torch.allclose(es, ec, atol=1e-12, rtol=1e-12)

        ed.sum().backward()
        es.sum().backward()
        ec.sum().backward()
        assert pos_d.grad is not None and pos_s.grad is not None and pos_c.grad is not None
        assert torch.allclose(pos_d.grad, pos_s.grad, atol=1e-11, rtol=1e-11)
        assert torch.allclose(pos_s.grad, pos_c.grad, atol=1e-12, rtol=1e-12)


def test_differentiable_s9_at_zero() -> None:
    """
    A tensor s9 = 0.0 with requires_grad=True still evaluates the three-body
    term so that dE/ds9 equals the unscaled ATM energy.
    """
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    s9 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    param = {
        "s6": torch.tensor(1.0, dtype=torch.float64),
        "s8": torch.tensor(0.7875, dtype=torch.float64),
        "a1": torch.tensor(0.4289, dtype=torch.float64),
        "a2": torch.tensor(4.4407, dtype=torch.float64),
        "s9": s9,
    }
    e_total = dftd3(mol, param).sum()
    e_total.backward()

    param_atm1 = dict(param, s9=torch.tensor(1.0, dtype=torch.float64))
    param_no_atm = {k: v for k, v in param.items() if k != "s9"}
    e_2b = dftd3(mol, param_no_atm).sum()
    e_3b_only = dftd3(mol, param_atm1).sum() - e_2b

    assert torch.allclose(e_total.detach(), e_2b.detach(), atol=1e-14)
    assert s9.grad is not None
    assert torch.allclose(s9.grad, e_3b_only.detach(), atol=1e-13)


def test_gradcheck_and_gradgradcheck_d3_and_d3s() -> None:
    numbers = torch.tensor([6, 8, 46, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.3, 0.1, -0.2],
            [-1.8, 2.4, 0.5],
            [0.9, -1.9, 1.4],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    a1 = torch.tensor(0.4289, dtype=torch.float64, requires_grad=True)
    s8 = torch.tensor(0.7875, dtype=torch.float64, requires_grad=True)
    cutoff = Cutoff(cn=15.0, disp2=12.0, disp3=10.0, width2=3.0, width3=3.0)

    def eval_d3(pos: torch.Tensor, a1_t: torch.Tensor, s8_t: torch.Tensor) -> torch.Tensor:
        mol = Structure(numbers=numbers, positions=pos)
        p = {"s6": 1.0, "s8": s8_t, "a1": a1_t, "a2": 4.4407, "s9": 1.0}
        return dftd3(mol, p, cutoff=cutoff).sum()

    def eval_d3s(pos: torch.Tensor, a1_t: torch.Tensor, s8_t: torch.Tensor) -> torch.Tensor:
        mol = Structure(numbers=numbers, positions=pos)
        p = {"s6": 1.0, "s8": s8_t, "a1": a1_t, "a2": 4.4407, "s9": 1.0}
        return dftd3s(mol, p, cutoff=cutoff).sum()

    assert torch.autograd.gradcheck(
        eval_d3, (positions, a1, s8), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    assert torch.autograd.gradgradcheck(
        eval_d3, (positions, a1, s8), eps=1e-5, atol=1e-4, rtol=1e-3
    )
    assert torch.autograd.gradcheck(
        eval_d3s, (positions, a1, s8), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    assert torch.autograd.gradgradcheck(
        eval_d3s, (positions, a1, s8), eps=1e-5, atol=1e-4, rtol=1e-3
    )


def test_func_transforms_and_compile_d3_and_d3s() -> None:
    nums = torch.tensor(
        [[6, 8, 1, 1], [14, 8, 1, 0]], dtype=torch.long
    )
    pos = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.2, 0.1, 0.0],
                [-1.1, 1.8, 0.2],
                [-1.0, -1.7, -0.3],
            ],
            [
                [0.0, 0.0, 0.0],
                [2.8, -0.2, 0.1],
                [-1.5, 2.1, -0.4],
                [0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float64,
    )
    param = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407, "s9": 1.0}

    for fn in (dftd3, dftd3s):
        def single_e(n: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return fn(Structure(numbers=n, positions=p), param).sum(-1)

        # jacrev vs jacfwd on sample 0
        rev_g = torch.func.jacrev(single_e, argnums=1)(nums[0], pos[0])
        fwd_g = torch.func.jacfwd(single_e, argnums=1)(nums[0], pos[0])
        assert torch.allclose(rev_g, fwd_g, atol=1e-11, rtol=1e-11)

        # vmap over energy and gradient
        vmap_e = torch.func.vmap(single_e, in_dims=(0, 0))(nums, pos)
        ref_e0 = single_e(nums[0], pos[0])
        ref_e1 = single_e(nums[1, :3], pos[1, :3])
        assert torch.allclose(vmap_e[0], ref_e0, atol=1e-12)
        assert torch.allclose(vmap_e[1], ref_e1, atol=1e-12)

        vmap_g = torch.func.vmap(
            torch.func.jacrev(single_e, argnums=1), in_dims=(0, 0)
        )(nums, pos)
        assert torch.allclose(vmap_g[0], rev_g, atol=1e-11)
        assert torch.all(vmap_g[1, 3] == 0.0)

        # torch.compile with fullgraph=True
        compiled_fn = torch.compile(
            single_e, backend="aot_eager", fullgraph=True
        )
        comp_e0 = compiled_fn(nums[0], pos[0])
        assert torch.allclose(comp_e0, ref_e0, atol=1e-12)


def test_dispersion_from_c6_helpers() -> None:
    mol = get_structure("mb16_43", "01", dtype=torch.float64)
    param = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407, "s9": 1.0}
    d3 = D3Model()
    env = d3.environment(mol)
    w = d3.weights(mol, env)
    c6 = d3.coefficients(mol, w)

    e_full = dftd3(mol, param)
    e_from_c6 = dispersion(mol, param, c6)
    e2 = dispersion2(mol, param, c6)
    e3 = dispersion3(mol, param, c6)

    assert torch.allclose(e_full, e_from_c6, atol=1e-14)
    assert torch.allclose(e_full, e2 + e3, atol=1e-14)

    # Also test gaussian_weight passed to weight_references
    w_gw = model.weight_references(
        mol.numbers, env.cn, Reference(), model.gaussian_weight
    )
    assert torch.allclose(w, w_gw, atol=1e-14)
    assert data.R4R2(dtype=torch.float64).shape == (119,)

    # Test D3SModel.weights(), D3SModel.coefficients(mol, w), and atomic_c6_d3s
    d3s = D3SModel()
    env_d3s = d3s.environment(mol)
    w_d3s = d3s.weights(mol, env_d3s)
    c6_d3s = d3s.coefficients(mol, w_d3s).dense()
    c6_d3s_direct = model.atomic_c6_d3s(
        mol.numbers,
        env_d3s.cn,
        Reference.load_d3s(dtype=torch.float64),
        data.WFPAIR_D3S(dtype=torch.float64),
    )
    assert torch.allclose(c6_d3s, c6_d3s_direct, atol=1e-14)

    # Test catalogue damping validation in D3 and D3S
    with pytest.raises(Exception):
        dftd3(mol, dict(param, damping="fermi"))
    with pytest.raises(Exception):
        dftd3s(mol, dict(param, damping="z"))
