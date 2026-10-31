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
Tests for Milestone D: DFT-D1 (`D1Model`) and DFT-D2 (`D2Model`) dispersion
models, regression against Grimme-lab `dftd/test/unit/test_dftd.f90`, dense vs
sparse consistency, batching/padding invariance, periodic cells, autograd, and
`torch.func` / `torch.compile` transforms.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import pack_structures

from tad_dftd import (
    Cutoff,
    D1Model,
    D2Model,
    DensePairs,
    SparsePairs,
    damping,
    data,
    dftd,
    dftd1,
    dftd2,
    get_properties,
    param,
)


def _d1_cluster(dtype: torch.dtype = torch.float64) -> Structure:
    """
    15-atom cluster of H, C, N, O, F, and Ne from
    ``dftd/test/unit/test_dftd.f90:get_d1_cluster``.
    """
    numbers = torch.tensor(
        [1, 6, 7, 9, 1, 8, 1, 1, 10, 6, 1, 1, 1, 1, 7], dtype=torch.long
    )
    xyz = torch.tensor(
        [
            [0.00000000000000, 0.00000000000000, 0.00000000000000],
            [2.00310969210332, 0.00000000000000, 0.00000000000000],
            [4.17629473542295, 0.00000000000000, 0.00000000000000],
            [0.37794522492515, 5.85815098633989, 0.75589044985031],
            [2.07869873708835, 5.66917837387731, 0.56691783738773],
            [0.94486306231289, -5.29123314895216, 2.07869873708835],
            [2.26767134955092, -4.72431531156443, 3.02356179940123],
            [0.56691783738773, -6.99198666111535, 2.45664396201350],
            [7.18095927357793, 3.59047963678896, -4.72431531156443],
            [8.50376756081597, -2.83458918693866, 1.51178089970062],
            [9.63760323559143, -1.88972612462577, 2.83458918693866],
            [7.55890449850308, -4.34637008663927, 2.45664396201350],
            [9.63760323559143, -3.77945224925154, 0.00000000000000],
            [7.18095927357793, -1.70075351216319, 0.37794522492515],
            [-4.72431531156443, 1.88972612462577, 3.77945224925154],
        ],
        dtype=dtype,
    )
    return Structure(numbers=numbers, positions=xyz)


def _d2_heavy_cluster(dtype: torch.dtype = torch.float64) -> Structure:
    """
    12-atom heavy-element cluster (Z up to 86) from
    ``dftd/test/unit/test_dftd.f90:test_pbed2_heavy``.
    """
    numbers = torch.tensor(
        [55, 56, 57, 72, 79, 80, 82, 83, 86, 54, 36, 30], dtype=torch.long
    )
    xyz = torch.tensor(
        [
            [-0.09524089298036, 0.11954477216099, 0.84842116804746],
            [6.93130014019955, 0.01568254612454, 0.17476965769979],
            [13.36932068770975, 0.02381727808361, 0.25976544043360],
            [0.58595374503991, 6.18824691245844, -0.39319747475095],
            [6.18134107498368, 7.61928906873436, 0.38687696508248],
            [13.08376067273969, 7.96438684159756, 0.92951556225113],
            [0.30784506706768, 0.23112540915714, 6.31498819028032],
            [6.03000147389921, 0.05676253234096, 6.11910221033771],
            [13.38041652559586, -0.51611397266957, 6.06016517844958],
            [-0.07213107755343, 6.88106223331331, 7.68485425703706],
            [7.03824822952815, 7.28058341583835, 6.99954630441358],
            [14.32489906378074, 6.91465976319911, 6.55632579932777],
        ],
        dtype=dtype,
    )
    return Structure(numbers=numbers, positions=xyz)


def test_d1_fortran_regression() -> None:
    """
    Regression against ``test_blypd1_cluster`` and ``test_pbed1_cluster`` in
    ``dftd/test/unit/test_dftd.f90``.
    """
    mol = _d1_cluster(dtype=torch.float64)

    # BLYP-D1 (from functional parameter lookup)
    blyp_param = param.get_params(
        model="d1", functional="blyp", dtype=torch.float64
    )
    e_blyp = dftd1(mol, blyp_param).sum()
    assert torch.allclose(
        e_blyp,
        torch.tensor(-5.0291775699253e-3, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    # Same via unified dftd(..., model="d1")
    e_blyp_disp = dftd(mol, blyp_param, model="d1").sum()
    assert torch.allclose(e_blyp, e_blyp_disp, atol=1e-14, rtol=1e-14)

    # PBE-D1 with explicit parameters
    pbe_param = {"s6": 0.7, "rs6": 1.0, "alp": 23.0}
    e_pbe = dftd1(mol, pbe_param).sum()
    assert torch.allclose(
        e_pbe,
        torch.tensor(-2.5145887849626e-3, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    # Dense vs SparsePairs agreement
    sparse = SparsePairs.build(mol, Cutoff(disp2=90.0))
    e_pbe_sparse = dftd1(mol, pbe_param, pairs=sparse).sum()
    assert torch.allclose(e_pbe, e_pbe_sparse, atol=1e-14, rtol=1e-14)


def test_d1_harmonic_mean_and_properties() -> None:
    """
    Regression against ``test_d1_harmonic_mean`` in
    ``dftd/test/unit/test_dftd.f90``.
    """
    mol = Structure(
        numbers=torch.tensor([1, 6], dtype=torch.long),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float64
        ),
    )
    d1 = D1Model()
    c6 = d1.coefficients(mol)
    c6_tab = data.D1_C6(dtype=torch.float64)
    expected_12 = 2.0 * c6_tab[1] * c6_tab[6] / (c6_tab[1] + c6_tab[6])
    assert torch.allclose(c6[0, 1], expected_12, atol=1e-12, rtol=1e-12)
    assert torch.allclose(c6[1, 0], expected_12, atol=1e-12, rtol=1e-12)
    assert torch.allclose(c6[1, 1], c6_tab[6], atol=1e-12, rtol=1e-12)

    props = get_properties(mol, model="d1")
    assert torch.allclose(props.c6, c6)
    assert torch.all(props.cn == 0.0)
    assert torch.all(props.c8 == 0.0)
    assert props.q is None


def test_d1_unsupported_element_validation() -> None:
    """
    Regression against ``test_unsupported_d1`` in
    ``dftd/test/unit/test_dftd.f90``.
    """
    for bad_z in (2, 3, 4, 5, 11, 16, -1):
        mol = Structure(
            numbers=torch.tensor([bad_z, 1], dtype=torch.long),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float64
            ),
        )
        with pytest.raises(ValueError, match="unsupported element"):
            dftd1(mol, {"s6": 0.7, "rs6": 1.0, "alp": 23.0})


@pytest.mark.parametrize(
    "record,param_dict,ref_energy",
    [
        (
            "01",
            param.get_params(model="d2", functional="pbe", dtype=torch.float64),
            -2.1689889055763276e-2,
        ),
        (
            "02",
            {"s6": 1.25, "rs6": 1.1, "alp": 20.0},
            -3.8203096588666829e-2,
        ),
        (
            "03",
            param.get_params(
                model="d2", functional="dsd-blyp", dtype=torch.float64
            ),
            -9.6695603361320100e-3,
        ),
    ],
)
def test_d2_fortran_mb16_43_regression(
    record: str, param_dict: dict[str, float | str], ref_energy: float
) -> None:
    """
    Regression against ``test_pbed2_mb01``, ``test_b97d2_mb02``, and
    ``test_dsdblypd2_mb03`` in ``dftd/test/unit/test_dftd.f90``.
    """
    mol = get_structure("mb16_43", record).to(dtype=torch.float64)
    e_dense = dftd2(mol, param_dict).sum()
    assert torch.allclose(
        e_dense,
        torch.tensor(ref_energy, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    e_disp = dftd(mol, param_dict, model="d2").sum()
    assert torch.allclose(e_dense, e_disp, atol=1e-14, rtol=1e-14)

    sparse = SparsePairs.build(mol, Cutoff.from_model("d2"))
    e_sparse = dftd2(mol, param_dict, pairs=sparse).sum()
    assert torch.allclose(e_dense, e_sparse, atol=1e-13, rtol=1e-13)


def test_d2_heavy_elements_and_actinide_validation() -> None:
    """
    Regression against ``test_pbed2_heavy`` and ``test_actinides_d2`` in
    ``dftd/test/unit/test_dftd.f90``.
    """
    mol = _d2_heavy_cluster(dtype=torch.float64)
    p = {"s6": 0.75, "rs6": 1.1, "alp": 20.0}
    e = dftd2(mol, p).sum()
    assert torch.allclose(
        e,
        torch.tensor(-1.2371225385416840e-1, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    props = get_properties(mol, model="d2")
    d2 = D2Model()
    assert torch.allclose(props.c6, d2.coefficients(mol))
    assert torch.allclose(d2.get_atomic_c6(mol), d2.coefficients(mol), atol=1e-12)

    # Actinides (Z >= 87) must raise ValueError
    mol_fr = Structure(
        numbers=torch.tensor([87, 1], dtype=torch.long),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float64
        ),
    )
    with pytest.raises(ValueError, match="unsupported element"):
        dftd2(mol_fr, p)


def test_d1_and_d2_three_body_atm_and_smooth_cutoff() -> None:
    """
    Regression against ``test_atm_d1_d2`` and ``test_d2_smooth_cutoff`` in
    ``dftd/test/unit/test_dftd.f90``.
    """
    mol = _d1_cluster(dtype=torch.float64)
    p_d1 = {
        "s6": 0.7,
        "rs6": 1.0,
        "alp": 23.0,
        "s9": 1.0,
        "rs9": 4.0 / 3.0,
        "alp3": 14.0,
        "a13": 1.0,
        "a23": 0.0,
    }
    damp_3b = damping.new_damping("fermi", "zero_avg", model="d1")
    damp_2b = damping.new_damping("fermi", False, model="d1")

    e_3b = dftd1(mol, p_d1, damping=damp_3b).sum()
    e_2b = dftd1(mol, p_d1, damping=damp_2b).sum()
    assert torch.abs(e_3b - e_2b) > 1e-8

    # Default damping when s9 is set uses zero_d4 / zero_avg
    e_3b_default = dftd1(mol, p_d1).sum()
    assert torch.allclose(e_3b, e_3b_default, atol=1e-14, rtol=1e-14)

    # Sparse vs dense with 3-body ATM for D1 and D2
    cut = Cutoff(disp2=50.0, disp3=40.0)
    sparse = SparsePairs.build(mol, cut, three_body=True)
    e_3b_sparse = dftd1(mol, p_d1, damping=damp_3b, pairs=sparse, cutoff=cut).sum()
    assert torch.allclose(e_3b, e_3b_sparse, atol=1e-13, rtol=1e-13)

    p_d2 = {**p_d1, "rs6": 1.1, "alp": 20.0}
    e_d2_3b = dftd2(mol, p_d2, damping=damp_3b, cutoff=cut).sum()
    e_d2_3b_sparse = dftd2(
        mol, p_d2, damping=damp_3b, pairs=sparse, cutoff=cut
    ).sum()
    assert torch.allclose(e_d2_3b, e_d2_3b_sparse, atol=1e-13, rtol=1e-13)

    # Smooth cutoff gradcheck on D2
    pos = mol.positions.clone().detach().requires_grad_(True)
    cut_smooth = Cutoff(disp2=8.0, width2=4.0)

    def _fn(x: torch.Tensor) -> torch.Tensor:
        s = Structure(numbers=mol.numbers, positions=x)
        return dftd2(
            s, {"s6": 0.75, "rs6": 1.1, "alp": 20.0}, cutoff=cut_smooth
        ).sum()

    assert torch.autograd.gradcheck(_fn, (pos,), eps=1e-6, atol=1e-5)


def test_d1_and_d2_batching_padding_and_periodic() -> None:
    """
    Batching, padding (`Z == 0`) invariance, and periodic boundary conditions
    for D1 and D2.
    """
    m1 = _d1_cluster(dtype=torch.float64)
    m2 = Structure(
        numbers=torch.tensor([6, 8, 1, 1], dtype=torch.long),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.3],
                [1.8, 0.0, -1.0],
                [-1.8, 0.0, -1.0],
            ],
            dtype=torch.float64,
        ),
    )
    batch = pack_structures([m1, m2])

    p1 = {"s6": 0.7, "rs6": 1.0, "alp": 23.0}
    e_batch_d1 = dftd1(batch, p1)
    assert torch.allclose(e_batch_d1[0].sum(), dftd1(m1, p1).sum(), atol=1e-14)
    assert torch.allclose(e_batch_d1[1, :4].sum(), dftd1(m2, p1).sum(), atol=1e-14)
    assert torch.all(e_batch_d1[1, 4:] == 0.0)

    p2 = {"s6": 0.75, "rs6": 1.1, "alp": 20.0}
    e_batch_d2 = dftd2(batch, p2)
    assert torch.allclose(e_batch_d2[0].sum(), dftd2(m1, p2).sum(), atol=1e-14)
    assert torch.allclose(e_batch_d2[1, :4].sum(), dftd2(m2, p2).sum(), atol=1e-14)
    assert torch.all(e_batch_d2[1, 4:] == 0.0)

    # Periodic cell dense vs sparse
    cell = Structure(
        numbers=torch.tensor([6, 1], dtype=torch.long),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], dtype=torch.float64
        ),
        lattice=torch.eye(3, dtype=torch.float64) * 8.0,
        periodic=torch.tensor([True, True, True]),
    )
    cut_pbc = Cutoff(disp2=15.0)
    dense_pbc = DensePairs.build(cell, cut_pbc)
    sparse_pbc = SparsePairs.build(cell, cut_pbc)
    assert torch.allclose(
        dftd1(cell, p1, pairs=dense_pbc, cutoff=cut_pbc),
        dftd1(cell, p1, pairs=sparse_pbc, cutoff=cut_pbc),
        atol=1e-13,
    )
    assert torch.allclose(
        dftd2(cell, p2, pairs=dense_pbc, cutoff=cut_pbc),
        dftd2(cell, p2, pairs=sparse_pbc, cutoff=cut_pbc),
        atol=1e-13,
    )


def test_d1_and_d2_autograd_func_and_compile() -> None:
    """
    First and second derivatives (`gradcheck`, `gradgradcheck`), `torch.func`
    (`jacrev`, `jacfwd`, `vmap`), and `torch.compile(fullgraph=True)` for D1
    and D2.
    """
    mol = Structure(
        numbers=torch.tensor([6, 8, 1, 7], dtype=torch.long),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.4],
                [1.9, 0.5, -0.9],
                [-1.8, -0.4, 1.2],
            ],
            dtype=torch.float64,
        ),
    )
    pos = mol.positions.clone().detach().requires_grad_(True)
    s6_t = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)

    for model_fn, alp in ((dftd1, 23.0), (dftd2, 20.0)):
        def _energy(x: torch.Tensor, s6_val: torch.Tensor) -> torch.Tensor:
            s = Structure(numbers=mol.numbers, positions=x)
            return model_fn(
                s,
                {"s6": s6_val, "rs6": 1.05, "alp": alp, "s9": 0.5},
            ).sum()

        assert torch.autograd.gradcheck(
            _energy, (pos, s6_t), eps=1e-6, atol=1e-5
        )
        assert torch.autograd.gradgradcheck(
            _energy, (pos, s6_t), eps=1e-5, atol=1e-4
        )

        j_rev = torch.func.jacrev(_energy)(pos.detach(), s6_t.detach())
        j_fwd = torch.func.jacfwd(_energy)(pos.detach(), s6_t.detach())
        assert torch.allclose(j_rev, j_fwd, atol=1e-11)

        batched_pos = torch.stack([pos.detach(), pos.detach() * 1.05], dim=0)
        vmap_res = torch.func.vmap(lambda x: _energy(x, s6_t.detach()))(
            batched_pos
        )
        assert vmap_res.shape == (2,)

        compiled = torch.compile(_energy, fullgraph=True)
        e_eager = _energy(pos, s6_t)
        e_comp = compiled(pos, s6_t)
        assert torch.allclose(e_eager, e_comp, atol=1e-13)
