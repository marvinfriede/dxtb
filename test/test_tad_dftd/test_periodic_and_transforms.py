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
import tad_dftd.threebody as _threebody_mod
from tad_dftd.threebody import (
    _periodic_block_energy,
    _periodic_tile_shape,
    dispersion_atm_periodic,
    dispersion_atm_sparse,
)
from torch.utils._python_dispatch import TorchDispatchMode


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

    def eval_dense_pos(p: torch.Tensor) -> torch.Tensor:
        return eval_dense(p, cell.lattice)

    def eval_auto_pos(p: torch.Tensor) -> torch.Tensor:
        return dftd3(
            cell.replace(positions=p),
            param_dict,
            cutoff=cutoff,
            max_triples=120_000,
        ).sum(-1)

    vmap_dense = torch.func.vmap(eval_dense_pos)(pos_batch)
    vmap_auto = torch.func.vmap(eval_auto_pos)(pos_batch)
    ref0 = eval_dense(pos_batch[0], cell.lattice)
    ref1 = eval_dense(pos_batch[1], cell.lattice)
    assert torch.allclose(vmap_dense, torch.stack([ref0, ref1]), atol=1e-12)
    assert torch.allclose(vmap_auto, torch.stack([ref0, ref1]), atol=1e-12)

    # Nested vmap(jacrev) over periodic positions
    grad_dense_pos = torch.func.jacrev(eval_dense_pos)
    vmap_grad_d = torch.func.vmap(grad_dense_pos)(pos_batch)
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

    def eval_dense_pos(p: torch.Tensor) -> torch.Tensor:
        return eval_dense(p, cell.lattice, q)

    vmap_e = torch.func.vmap(eval_dense_pos)(pos_batch)
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


class _MaxElementTracker(TorchDispatchMode):
    """Track maximum tensor element count and 2D/3D shapes allocated during dispatch."""

    def __init__(self) -> None:
        super().__init__()
        self.max_2d_elements: int = 0
        self.max_3d_elements: int = 0
        self.shapes: list[tuple[int, ...]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        tensors = (
            out
            if isinstance(out, (tuple, list))
            else (out,)
            if isinstance(out, torch.Tensor)
            else ()
        )
        for t in tensors:
            if isinstance(t, torch.Tensor):
                sh = tuple(int(s) for s in t.shape)
                self.shapes.append(sh)
                if len(sh) == 2:
                    self.max_2d_elements = max(
                        self.max_2d_elements, sh[0] * sh[1]
                    )
                elif len(sh) == 3:
                    self.max_3d_elements = max(
                        self.max_3d_elements, sh[0] * sh[1] * sh[2]
                    )
        return out


def test_periodic_atm_2d_tile_shape_and_memory_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify H2.1 2D tiling bounds (`tile_rows * tile_cols <= max_triples`),
    intermediate tensor allocation bounds via TorchDispatchMode, and energy /
    nuclear gradient / lattice derivative / parameter derivative invariance
    when `n_leg > max_triples`, `max_triples = 1`, non-square `max_triples`,
    incomplete final tiles, padded atoms, nonorthogonal cell, multiple images
    of the same atom, and empty candidate sets.
    """
    with pytest.raises(ValueError, match="max_triples"):
        _periodic_tile_shape(10, 0)
    assert _periodic_tile_shape(0, 5) == (0, 0)

    for n_leg in (1, 2, 5, 13, 27, 81, 108):
        for mt in (1, 2, 3, 5, 7, 11, 16, 17, 50, 2_000_000):
            r_blk, c_blk = _periodic_tile_shape(n_leg, mt)
            assert 1 <= r_blk <= n_leg
            assert 1 <= c_blk <= n_leg
            assert r_blk * c_blk <= mt

    # Nonorthogonal compact triclinic cell with padding (Z=0 at slot 2) and
    # multiple periodic images of the same atom within cutoff.
    numbers = torch.tensor([14, 8, 0, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.15, 0.25, 0.35],
            [2.10, 1.85, 1.65],
            [1.00, 1.00, 1.00],  # padded slot (Z=0)
            [3.15, 0.95, 2.85],
        ],
        dtype=torch.float64,
    )
    lattice = torch.tensor(
        [
            [4.6, 0.35, -0.15],
            [0.25, 4.8, 0.20],
            [-0.10, 0.15, 5.0],
        ],
        dtype=torch.float64,
    )
    cell = Structure(
        numbers=numbers,
        positions=positions,
        lattice=lattice,
        periodic=torch.tensor([True, True, True]),
    )
    # Cutoff 6.2 > lattice lengths (~4.6..5.0) ensures multiple images of the
    # same atom (including self-images shift != 0) are active, while shifts
    # built for 8.5 ensure most candidate periodic images are inactive.
    shifts_large = DensePairs.build(
        cell, Cutoff(cn=8.5, disp2=8.5, disp3=8.5), three_body=True
    )
    cutoff_eval = Cutoff(cn=6.2, disp2=6.2, disp3=6.2)
    sparse_pairs = SparsePairs.build(cell, cutoff_eval, three_body=True)
    assert shifts_large.disp3 is not None
    assert sparse_pairs.disp3 is not None

    d3_mod = D3Model(cutoff=cutoff_eval)
    env = d3_mod.environment(cell, pairs=shifts_large.cn)
    w = d3_mod.weights(cell, env)
    c6 = d3_mod.coefficients(cell, w).dense()
    dparam = DampingParam(
        s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0, alp=14.0
    )

    def eval_atm(
        pos: torch.Tensor,
        lat: torch.Tensor,
        s9_t: torch.Tensor,
        alp_t: torch.Tensor,
        mt: int,
        ckpt: bool = False,
    ) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        dp = DampingParam(
            s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=s9_t, alp=alp_t
        )
        return dispersion_atm_periodic(
            st,
            c6,
            dp,
            shifts_large.disp3,
            cutoff=cutoff_eval.disp3,
            max_triples=mt,
            checkpoint=ckpt,
        )

    s9_ref = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    alp_ref = torch.tensor(14.0, dtype=torch.float64, requires_grad=True)
    pos_ref = positions.clone().detach().requires_grad_(True)
    lat_ref = lattice.clone().detach().requires_grad_(True)

    e_ref = eval_atm(pos_ref, lat_ref, s9_ref, alp_ref, mt=2_000_000)
    # Padded atom (index 2) has exact zero energy
    assert e_ref[2].item() == 0.0
    assert torch.all(e_ref[[0, 1, 3]] != 0.0)

    # Dense vs sparse ATM parity on padded nonorthogonal multi-image cell
    e_sparse = dispersion_atm_sparse(
        cell, c6, dparam, sparse_pairs.disp3, cutoff=cutoff_eval.disp3
    )
    torch.testing.assert_close(e_ref.detach(), e_sparse, atol=1e-12, rtol=1e-12)

    g_pos_ref, g_lat_ref, g_s9_ref, g_alp_ref = torch.autograd.grad(
        e_ref.sum(), (pos_ref, lat_ref, s9_ref, alp_ref)
    )
    # Padded atom has zero nuclear force
    assert torch.all(g_pos_ref[2] == 0.0)

    # Direct spy + TorchDispatchMode verification for max_triples in (1, 3, 5, 7, 11)
    # Note: in eager mode, active n_leg is ~18..24 > max_triples; in static-leg mode
    # (under jacrev/jacfwd/compile), n_leg = nat * n_shift > 100 >> max_triples.
    for mt in (1, 3, 5, 7, 11):
        recorded_tiles: list[tuple[int, int]] = []
        max_tile_2d_alloc = 0
        max_tile_3d_alloc = 0

        def _spied_block(*args: object, **kwargs: object) -> torch.Tensor:
            nonlocal max_tile_2d_alloc, max_tile_3d_alloc
            vec_a = args[0]
            vec_b = args[1]
            assert isinstance(vec_a, torch.Tensor)
            assert isinstance(vec_b, torch.Tensor)
            ta, tb = int(vec_a.shape[0]), int(vec_b.shape[0])
            recorded_tiles.append((ta, tb))
            assert 1 <= ta <= mt
            assert 1 <= tb <= mt
            assert ta * tb <= mt
            with _MaxElementTracker() as tracker:
                res = _periodic_block_energy(*args, **kwargs)  # type: ignore[arg-type]
            # Inside _periodic_block_energy, c6_sys is (nat, nat) = (4, 4) = 16;
            # every leg-pair 2D tensor has ta * tb <= mt elements, and vec_jk has
            # ta * tb * 3 <= 3 * mt elements.
            max_tile_2d_alloc = max(max_tile_2d_alloc, tracker.max_2d_elements)
            max_tile_3d_alloc = max(max_tile_3d_alloc, tracker.max_3d_elements)
            return res

        monkeypatch.setattr(
            _threebody_mod, "_periodic_block_energy", _spied_block
        )

        pos_t = positions.clone().detach().requires_grad_(True)
        lat_t = lattice.clone().detach().requires_grad_(True)
        s9_t = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        alp_t = torch.tensor(14.0, dtype=torch.float64, requires_grad=True)

        e_t = eval_atm(pos_t, lat_t, s9_t, alp_t, mt=mt, ckpt=(mt == 7))
        assert len(recorded_tiles) > 10
        assert all(r * c <= mt for r, c in recorded_tiles)
        # Verify incomplete final tiles occur when n_leg is not divisible by block size
        if mt in (7, 11):
            assert len({(r, c) for r, c in recorded_tiles}) > 1
        assert max_tile_2d_alloc <= max(mt, int(numbers.shape[0]) ** 2)
        assert max_tile_3d_alloc <= 3 * mt

        torch.testing.assert_close(e_t, e_ref, atol=1e-13, rtol=1e-13)
        g_pos_t, g_lat_t, g_s9_t, g_alp_t = torch.autograd.grad(
            e_t.sum(), (pos_t, lat_t, s9_t, alp_t)
        )
        torch.testing.assert_close(g_pos_t, g_pos_ref, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(g_lat_t, g_lat_ref, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(g_s9_t, g_s9_ref, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(g_alp_t, g_alp_ref, atol=1e-12, rtol=1e-12)

    # Also verify static-leg path (jacrev & jacfwd) with non-square max_triples < n_leg
    recorded_static_tiles: list[tuple[int, int]] = []
    n_leg_static = int(numbers.shape[0]) * int(shifts_large.disp3.shifts.shape[-2])
    mt_static = max(11, n_leg_static - 5)  # strictly < n_leg_static so even 1 full row exceeds mt_static

    def _spied_static_block(*args: object, **kwargs: object) -> torch.Tensor:
        vec_a = args[0]
        vec_b = args[1]
        assert isinstance(vec_a, torch.Tensor)
        assert isinstance(vec_b, torch.Tensor)
        ta, tb = int(vec_a.shape[0]), int(vec_b.shape[0])
        recorded_static_tiles.append((ta, tb))
        assert 1 <= ta < n_leg_static
        assert 1 <= tb < n_leg_static
        assert ta * tb <= mt_static
        return _periodic_block_energy(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        _threebody_mod, "_periodic_block_energy", _spied_static_block
    )

    def eval_atm_total(
        pos: torch.Tensor,
        lat: torch.Tensor,
        s9_in: torch.Tensor,
        alp_in: torch.Tensor,
    ) -> torch.Tensor:
        return eval_atm(pos, lat, s9_in, alp_in, mt=mt_static).sum()

    jrev = torch.func.jacrev(eval_atm_total, argnums=(0, 1, 2, 3))(
        positions, lattice, s9_ref.detach(), alp_ref.detach()
    )
    assert len(recorded_static_tiles) > 100
    assert all(r * c <= mt_static for r, c in recorded_static_tiles)
    jfwd = torch.func.jacfwd(eval_atm_total, argnums=(0, 1, 2, 3))(
        positions, lattice, s9_ref.detach(), alp_ref.detach()
    )
    for g_r, g_f, g_auto in zip(
        jrev, jfwd, (g_pos_ref, g_lat_ref, g_s9_ref, g_alp_ref)
    ):
        torch.testing.assert_close(g_r, g_f, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(g_r, g_auto, atol=1e-12, rtol=1e-12)

    monkeypatch.setattr(
        _threebody_mod, "_periodic_block_energy", _periodic_block_energy
    )

    # Empty candidate set (cutoff smaller than all interatomic distances -> n_leg == 0)
    pos_empty = positions.clone().detach().requires_grad_(True)
    lat_empty = lattice.clone().detach().requires_grad_(True)
    e_empty = dispersion_atm_periodic(
        cell.replace(positions=pos_empty, lattice=lat_empty),
        c6,
        dparam,
        shifts_large.disp3,
        cutoff=0.5,
        max_triples=1,
    )
    assert e_empty.shape == numbers.shape
    assert torch.isfinite(e_empty).all()
    assert torch.all(e_empty == 0.0)
    g_pos_empty, g_lat_empty = torch.autograd.grad(
        e_empty.sum(), (pos_empty, lat_empty)
    )
    assert torch.isfinite(g_pos_empty).all() and torch.all(g_pos_empty == 0.0)
    assert torch.isfinite(g_lat_empty).all() and torch.all(g_lat_empty == 0.0)

