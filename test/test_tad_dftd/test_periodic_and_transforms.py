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
Periodic dense vs sparse parity, chunking invariance, and PyTorch transform
matrix tests across all implemented models (D1, D2, D3, D3S, D4, D4S, D4Srev).
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.io.structure import Structure

from tad_dftd import (
    Cutoff,
    DensePairs,
    SparsePairs,
    dftd,
    dftd3,
)
from tad_dftd.damping.param import DampingParam
from tad_dftd.model.d3 import D3Model
from tad_dftd.threebody import dispersion_atm_periodic, dispersion_atm_sparse


def _make_periodic_cell(
    dtype: torch.dtype = torch.float64, *, d1_compatible: bool = False
) -> Structure:
    numbers = (
        torch.tensor([8, 6, 7, 1], dtype=torch.long)
        if d1_compatible
        else torch.tensor([14, 8, 6, 1], dtype=torch.long)
    )
    positions = torch.tensor(
        [
            [0.10, 0.20, 0.30],
            [2.40, 2.10, 1.90],
            [1.30, 3.60, 0.80],
            [3.70, 1.20, 3.80],
        ],
        dtype=dtype,
    )
    lattice = torch.tensor(
        [
            [7.5, 0.2, 0.0],
            [0.1, 7.8, 0.1],
            [0.0, 0.1, 8.0],
        ],
        dtype=dtype,
    )
    return Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=torch.tensor([True, True, True]),
    )


def _model_param(model_name: str, dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    if model_name == "d1":
        return {
            "s6": torch.tensor(1.0, dtype=dtype),
            "rs6": torch.tensor(1.0, dtype=dtype),
            "alp": torch.tensor(23.0, dtype=dtype),
        }
    if model_name == "d2":
        return {
            "s6": torch.tensor(0.75, dtype=dtype),
            "rs6": torch.tensor(1.0, dtype=dtype),
            "alp": torch.tensor(20.0, dtype=dtype),
        }
    if model_name in ("d3", "d3s"):
        return {
            "s6": torch.tensor(1.0, dtype=dtype),
            "s8": torch.tensor(0.7875, dtype=dtype),
            "a1": torch.tensor(0.4289, dtype=dtype),
            "a2": torch.tensor(4.4407, dtype=dtype),
            "s9": torch.tensor(1.0, dtype=dtype),
        }
    return {
        "s6": torch.tensor(1.0, dtype=dtype),
        "s8": torch.tensor(0.95948085, dtype=dtype),
        "a1": torch.tensor(0.38574991, dtype=dtype),
        "a2": torch.tensor(4.80688534, dtype=dtype),
        "s9": torch.tensor(1.0, dtype=dtype),
        "rs9": torch.tensor(1.0, dtype=dtype),
        "alp": torch.tensor(16.0, dtype=dtype),
        "a13": torch.tensor(0.38574991, dtype=dtype),
        "a23": torch.tensor(4.80688534, dtype=dtype),
        "alp3": torch.tensor(16.0, dtype=dtype),
    }


@pytest.mark.parametrize(
    "model_name", ["d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"]
)
def test_periodic_dense_vs_sparse_energies_forces_and_strain(
    model_name: str,
) -> None:
    """
    Verify that periodic DensePairs and SparsePairs yield identical atom-resolved
    energies, atomic forces (-dE/dr), and lattice/strain derivatives (dE/dlattice)
    across all implemented models (2B and 3B).
    """
    cell = _make_periodic_cell(
        torch.float64, d1_compatible=(model_name == "d1")
    )
    cutoff = Cutoff(cn=11.0, disp2=13.0, disp3=9.5, cn_eeq=11.0)
    param = _model_param(model_name, torch.float64)
    has_3b = model_name not in ("d1", "d2")
    with_cn_eeq = model_name in ("d4", "d4s", "d4srev")
    extra_kw = (
        {"q": torch.tensor([0.25, -0.30, 0.15, -0.10], dtype=torch.float64)}
        if with_cn_eeq
        else {}
    )

    dense_pairs = DensePairs.build(
        cell, cutoff, three_body=has_3b, with_cn_eeq=with_cn_eeq
    )
    sparse_pairs = SparsePairs.build(
        cell, cutoff, three_body=has_3b, with_cn_eeq=with_cn_eeq
    )

    pos_d = cell.positions.clone().detach().requires_grad_(True)
    lat_d = cell.lattice.clone().detach().requires_grad_(True)
    cell_d = cell.replace(positions=pos_d, lattice=lat_d)

    pos_s = cell.positions.clone().detach().requires_grad_(True)
    lat_s = cell.lattice.clone().detach().requires_grad_(True)
    cell_s = cell.replace(positions=pos_s, lattice=lat_s)

    e_dense = dftd(
        cell_d,
        param,
        model=model_name,
        cutoff=cutoff,
        pairs=dense_pairs,
        **extra_kw,
    )
    e_sparse = dftd(
        cell_s,
        param,
        model=model_name,
        cutoff=cutoff,
        pairs=sparse_pairs,
        **extra_kw,
    )

    assert torch.allclose(e_dense, e_sparse, atol=1e-12, rtol=1e-12)

    grad_pos_d, grad_lat_d = torch.autograd.grad(e_dense.sum(), (pos_d, lat_d))
    grad_pos_s, grad_lat_s = torch.autograd.grad(e_sparse.sum(), (pos_s, lat_s))

    assert torch.allclose(grad_pos_d, grad_pos_s, atol=1e-11, rtol=1e-11)
    assert torch.allclose(grad_lat_d, grad_lat_s, atol=1e-11, rtol=1e-11)


def test_periodic_threebody_chunking_invariance_and_transforms() -> None:
    """
    Verify that ``max_triples`` chunking in both ``dispersion_atm_periodic``
    and ``dispersion_atm_sparse`` preserves energies and gradients in eager mode,
    under ``vmap``/``jacrev``/``jacfwd``, and under ``torch.compile``.
    """
    cell = _make_periodic_cell(torch.float64)
    cutoff = Cutoff(cn=10.0, disp2=12.0, disp3=9.0)
    dense_pairs = DensePairs.build(cell, cutoff, three_body=True)
    sparse_pairs = SparsePairs.build(
        cell, cutoff, three_body=True, chunk_size=13
    )
    assert dense_pairs.disp3 is not None
    assert sparse_pairs.disp3 is not None

    d3_mod = D3Model(cutoff=cutoff)
    env = d3_mod.environment(cell, pairs=dense_pairs.cn)
    w = d3_mod.weights(cell, env)
    c6 = d3_mod.coefficients(cell, w).dense()
    dparam = DampingParam(
        s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0
    )

    # 1. Eager chunking invariance (max_triples=13 vs default 2_000_000)
    e_pbc_full = dispersion_atm_periodic(
        cell, c6, dparam, dense_pairs.disp3, cutoff=cutoff.disp3
    )
    e_pbc_chunk = dispersion_atm_periodic(
        cell,
        c6,
        dparam,
        dense_pairs.disp3,
        cutoff=cutoff.disp3,
        max_triples=13,
    )
    e_sp_full = dispersion_atm_sparse(
        cell, c6, dparam, sparse_pairs.disp3, cutoff=cutoff.disp3
    )
    e_sp_chunk = dispersion_atm_sparse(
        cell,
        c6,
        dparam,
        sparse_pairs.disp3,
        cutoff=cutoff.disp3,
        max_triples=13,
    )

    assert torch.allclose(e_pbc_full, e_pbc_chunk, atol=1e-13, rtol=1e-13)
    assert torch.allclose(e_sp_full, e_sp_chunk, atol=1e-13, rtol=1e-13)
    assert torch.allclose(e_pbc_full, e_sp_chunk, atol=1e-12, rtol=1e-12)

    # 2. Periodic dense & sparse under jacrev, jacfwd, vmap, and torch.compile
    param_dict = {
        "s6": 1.0,
        "s8": 0.7875,
        "a1": 0.4289,
        "a2": 4.4407,
        "s9": 1.0,
    }

    def eval_dense(pos: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        return dftd3(
            st,
            param_dict,
            cutoff=cutoff,
            pairs=dense_pairs,
            max_triples=120_000,
        ).sum(-1)

    def eval_sparse(pos: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        return dftd3(
            st,
            param_dict,
            cutoff=cutoff,
            pairs=sparse_pairs,
            max_triples=17,
        ).sum(-1)

    jrev_pos_d, jrev_lat_d = torch.func.jacrev(eval_dense, argnums=(0, 1))(
        cell.positions, cell.lattice
    )
    jfwd_pos_d, jfwd_lat_d = torch.func.jacfwd(eval_dense, argnums=(0, 1))(
        cell.positions, cell.lattice
    )
    jrev_pos_s, jrev_lat_s = torch.func.jacrev(eval_sparse, argnums=(0, 1))(
        cell.positions, cell.lattice
    )
    jfwd_pos_s, jfwd_lat_s = torch.func.jacfwd(eval_sparse, argnums=(0, 1))(
        cell.positions, cell.lattice
    )

    assert torch.allclose(jrev_pos_d, jfwd_pos_d, atol=1e-11, rtol=1e-11)
    assert torch.allclose(jrev_lat_d, jfwd_lat_d, atol=1e-11, rtol=1e-11)
    assert torch.allclose(jrev_pos_d, jrev_pos_s, atol=1e-11, rtol=1e-11)
    assert torch.allclose(jrev_lat_d, jrev_lat_s, atol=1e-11, rtol=1e-11)
    assert torch.allclose(jrev_pos_s, jfwd_pos_s, atol=1e-11, rtol=1e-11)
    assert torch.allclose(jrev_lat_s, jfwd_lat_s, atol=1e-11, rtol=1e-11)

    # vmap over periodic positions (both with pre-built pairs and on-the-fly DensePairs)
    pos_batch = torch.stack(
        [cell.positions, cell.positions + 0.03], dim=0
    )
    vmap_dense = torch.func.vmap(
        lambda p: eval_dense(p, cell.lattice)
    )(pos_batch)
    vmap_auto = torch.func.vmap(
        lambda p: dftd3(
            cell.replace(positions=p),
            param_dict,
            cutoff=cutoff,
            max_triples=120_000,
        ).sum(-1)
    )(pos_batch)
    ref0 = eval_dense(pos_batch[0], cell.lattice)
    ref1 = eval_dense(pos_batch[1], cell.lattice)
    assert torch.allclose(vmap_dense, torch.stack([ref0, ref1]), atol=1e-12)
    assert torch.allclose(vmap_auto, torch.stack([ref0, ref1]), atol=1e-12)

    # Nested vmap(jacrev) over periodic positions
    vmap_grad_d = torch.func.vmap(
        torch.func.jacrev(lambda p: eval_dense(p, cell.lattice))
    )(pos_batch)
    assert torch.allclose(vmap_grad_d[0], jrev_pos_d, atol=1e-11, rtol=1e-11)

    # torch.compile with chunking enabled (max_triples=120_000) on both dense and sparse periodic
    comp_dense = torch.compile(eval_dense, backend="aot_eager", fullgraph=True)
    comp_sparse = torch.compile(
        eval_sparse, backend="aot_eager", fullgraph=True
    )
    assert torch.allclose(
        comp_dense(cell.positions, cell.lattice), ref0, atol=1e-12
    )
    assert torch.allclose(
        comp_sparse(cell.positions, cell.lattice), ref0, atol=1e-12
    )


@pytest.mark.parametrize("model_name", ["d4", "d4s", "d4srev"])
def test_periodic_d4_family_transforms_and_compile(model_name: str) -> None:
    """
    Verify periodic D4, D4S, and D4Srev under jacrev, jacfwd, vmap, and
    torch.compile (both DensePairs and SparsePairs, with 2B and 3B ATM).
    """
    cell = _make_periodic_cell(torch.float64)
    cutoff = Cutoff(cn=10.0, disp2=12.0, disp3=9.0, cn_eeq=10.0)
    param = _model_param(model_name, torch.float64)
    q = torch.tensor([0.25, -0.30, 0.15, -0.10], dtype=torch.float64)

    dense_pairs = DensePairs.build(
        cell, cutoff, three_body=True, with_cn_eeq=True
    )
    sparse_pairs = SparsePairs.build(
        cell, cutoff, three_body=True, with_cn_eeq=True, chunk_size=19
    )

    def eval_dense(
        pos: torch.Tensor, lat: torch.Tensor, q_in: torch.Tensor
    ) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        return dftd(
            st,
            param,
            model=model_name,
            q=q_in,
            cutoff=cutoff,
            pairs=dense_pairs,
            max_triples=120_000,
        ).sum(-1)

    def eval_sparse(
        pos: torch.Tensor, lat: torch.Tensor, q_in: torch.Tensor
    ) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        return dftd(
            st,
            param,
            model=model_name,
            q=q_in,
            cutoff=cutoff,
            pairs=sparse_pairs,
            max_triples=19,
        ).sum(-1)

    jrev_d = torch.func.jacrev(eval_dense, argnums=(0, 1, 2))(
        cell.positions, cell.lattice, q
    )
    jfwd_d = torch.func.jacfwd(eval_dense, argnums=(0, 1, 2))(
        cell.positions, cell.lattice, q
    )
    jrev_s = torch.func.jacrev(eval_sparse, argnums=(0, 1, 2))(
        cell.positions, cell.lattice, q
    )

    for gd_rev, gd_fwd, gs_rev in zip(jrev_d, jfwd_d, jrev_s):
        assert torch.allclose(gd_rev, gd_fwd, atol=1e-10, rtol=1e-10)
        assert torch.allclose(gd_rev, gs_rev, atol=1e-10, rtol=1e-10)

    pos_batch = torch.stack(
        [cell.positions, cell.positions + 0.02], dim=0
    )
    vmap_e = torch.func.vmap(
        lambda p: eval_dense(p, cell.lattice, q)
    )(pos_batch)
    ref0 = eval_dense(pos_batch[0], cell.lattice, q)
    ref1 = eval_dense(pos_batch[1], cell.lattice, q)
    assert torch.allclose(vmap_e, torch.stack([ref0, ref1]), atol=1e-12)

    comp_dense = torch.compile(eval_dense, backend="aot_eager", fullgraph=True)
    comp_sparse = torch.compile(
        eval_sparse, backend="aot_eager", fullgraph=True
    )
    assert torch.allclose(
        comp_dense(cell.positions, cell.lattice, q), ref0, atol=1e-12
    )
    assert torch.allclose(
        comp_sparse(cell.positions, cell.lattice, q), ref0, atol=1e-12
    )

