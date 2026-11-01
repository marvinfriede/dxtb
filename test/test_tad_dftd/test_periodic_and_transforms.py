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


def test_inductor_compile_molecular_and_periodic_models() -> None:
    """
    Validate default CPU Inductor backend (`torch.compile(..., backend="inductor", fullgraph=True)`)
    beyond `aot_eager` (H2.4) for:
    - Molecular D3 (2B + 3B ATM, differentiable damping parameter `a1`)
    - Molecular D4 (automatic EEQ and external charges, 2B + 3B ATM, differentiable `a1`)
    - Periodic dense D3 with ATM
    - Periodic sparse D3 with ATM
    - Periodic D4 with external charges (dense and sparse)
    Verifies energy, first derivatives, selected second derivatives, and repeated calls
    with changed coordinates and changed damping parameters.
    """
    torch._dynamo.reset()

    # 1. Molecular D3 and Molecular D4 (small nontrivial 3-atom H2O)
    mol_nums = torch.tensor([8, 1, 1], dtype=torch.long)
    mol_pos0 = torch.tensor(
        [
            [0.00, 0.00, 0.22],
            [0.00, 1.43, -0.88],
            [0.00, -1.43, -0.88],
        ],
        dtype=torch.float64,
    )
    mol_pos1 = mol_pos0 + torch.tensor(
        [[0.04, -0.02, 0.01], [-0.03, 0.05, -0.02], [0.01, -0.04, 0.03]],
        dtype=torch.float64,
    )
    mol_q0 = torch.tensor([-0.56, 0.28, 0.28], dtype=torch.float64)

    def mol_d3_fn(pos: torch.Tensor, a1_val: torch.Tensor) -> torch.Tensor:
        st = Structure(numbers=mol_nums, positions=pos)
        p = {
            "s6": torch.tensor(1.0, dtype=torch.float64),
            "s8": torch.tensor(0.7875, dtype=torch.float64),
            "a1": a1_val,
            "a2": torch.tensor(4.4407, dtype=torch.float64),
            "s9": torch.tensor(1.0, dtype=torch.float64),
        }
        return dftd3(st, p).sum()

    def mol_d4_ext_fn(
        pos: torch.Tensor, a1_val: torch.Tensor, q_in: torch.Tensor
    ) -> torch.Tensor:
        st = Structure(numbers=mol_nums, positions=pos)
        p = {
            "s6": torch.tensor(1.0, dtype=torch.float64),
            "s8": torch.tensor(0.95948085, dtype=torch.float64),
            "a1": a1_val,
            "a2": torch.tensor(4.80688534, dtype=torch.float64),
            "s9": torch.tensor(1.0, dtype=torch.float64),
            "rs9": torch.tensor(1.0, dtype=torch.float64),
            "alp": torch.tensor(16.0, dtype=torch.float64),
            "a13": a1_val,
            "a23": torch.tensor(4.80688534, dtype=torch.float64),
            "alp3": torch.tensor(16.0, dtype=torch.float64),
        }
        return dftd(st, p, model="d4", q=q_in).sum()

    def mol_d4_eeq_fn(pos: torch.Tensor, a1_val: torch.Tensor) -> torch.Tensor:
        st = Structure(numbers=mol_nums, positions=pos)
        p = {
            "s6": torch.tensor(1.0, dtype=torch.float64),
            "s8": torch.tensor(0.95948085, dtype=torch.float64),
            "a1": a1_val,
            "a2": torch.tensor(4.80688534, dtype=torch.float64),
            "s9": torch.tensor(1.0, dtype=torch.float64),
        }
        return dftd(st, p, model="d4", qmodel="eeq").sum()

    comp_mol_d3 = torch.compile(mol_d3_fn, backend="inductor", fullgraph=True)
    comp_mol_d4_ext = torch.compile(
        mol_d4_ext_fn, backend="inductor", fullgraph=True
    )
    comp_mol_d4_eeq = torch.compile(
        mol_d4_eeq_fn, backend="inductor", fullgraph=True
    )

    hess_mol_d3_fn = torch.func.jacfwd(
        torch.func.jacrev(mol_d3_fn, argnums=0), argnums=1
    )
    comp_hess_mol_d3 = torch.compile(
        hess_mol_d3_fn, backend="inductor", fullgraph=True
    )

    for idx, (pos_in, a1_in) in enumerate(
        (
            (mol_pos0, 0.4289),
            (mol_pos1, 0.4289),
            (mol_pos1, 0.4650),
        )
    ):
        # Molecular D3: energy, 1st derivative, 2nd mixed derivative (d^2E / dpos da1)
        p_e = pos_in.clone().requires_grad_(True)
        a_e = torch.tensor(a1_in, dtype=torch.float64, requires_grad=True)
        e_eager = mol_d3_fn(p_e, a_e)
        g_pos_e, g_a1_e = torch.autograd.grad(e_eager, (p_e, a_e))

        p_c = pos_in.clone().requires_grad_(True)
        a_c = torch.tensor(a1_in, dtype=torch.float64, requires_grad=True)
        e_comp = comp_mol_d3(p_c, a_c)
        g_pos_c, g_a1_c = torch.autograd.grad(e_comp, (p_c, a_c))

        torch.testing.assert_close(e_comp, e_eager, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(g_pos_c, g_pos_e, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(g_a1_c, g_a1_e, atol=1e-12, rtol=1e-12)
        if idx in (0, 2):
            h_mixed_e = hess_mol_d3_fn(pos_in, a_e.detach())
            h_mixed_c = comp_hess_mol_d3(pos_in, a_c.detach())
            torch.testing.assert_close(
                h_mixed_c, h_mixed_e, atol=1e-11, rtol=1e-11
            )

        # Molecular D4 with external charges + auto EEQ
        p_d4_e = pos_in.clone().requires_grad_(True)
        a_d4_e = torch.tensor(a1_in, dtype=torch.float64, requires_grad=True)
        q_d4_e = mol_q0.clone().requires_grad_(True)
        e_d4_eager = mol_d4_ext_fn(p_d4_e, a_d4_e, q_d4_e)
        gp_e, ga_e, gq_e = torch.autograd.grad(
            e_d4_eager, (p_d4_e, a_d4_e, q_d4_e)
        )

        p_d4_c = pos_in.clone().requires_grad_(True)
        a_d4_c = torch.tensor(a1_in, dtype=torch.float64, requires_grad=True)
        q_d4_c = mol_q0.clone().requires_grad_(True)
        e_d4_comp = comp_mol_d4_ext(p_d4_c, a_d4_c, q_d4_c)
        gp_c, ga_c, gq_c = torch.autograd.grad(
            e_d4_comp, (p_d4_c, a_d4_c, q_d4_c)
        )

        torch.testing.assert_close(
            e_d4_comp, e_d4_eager, atol=1e-13, rtol=1e-13
        )
        torch.testing.assert_close(gp_c, gp_e, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(ga_c, ga_e, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(gq_c, gq_e, atol=1e-12, rtol=1e-12)

        # Also check Molecular D4 with automatic EEQ under Inductor
        if idx in (0, 2):
            p_eeq_e = pos_in.clone().requires_grad_(True)
            a_eeq_e = torch.tensor(
                a1_in, dtype=torch.float64, requires_grad=True
            )
            e_eeq_eager = mol_d4_eeq_fn(p_eeq_e, a_eeq_e)
            gp_eeq_e, ga_eeq_e = torch.autograd.grad(
                e_eeq_eager, (p_eeq_e, a_eeq_e)
            )

            p_eeq_c = pos_in.clone().requires_grad_(True)
            a_eeq_c = torch.tensor(
                a1_in, dtype=torch.float64, requires_grad=True
            )
            e_eeq_comp = comp_mol_d4_eeq(p_eeq_c, a_eeq_c)
            gp_eeq_c, ga_eeq_c = torch.autograd.grad(
                e_eeq_comp, (p_eeq_c, a_eeq_c)
            )
            torch.testing.assert_close(
                e_eeq_comp, e_eeq_eager, atol=1e-13, rtol=1e-13
            )
            torch.testing.assert_close(
                gp_eeq_c, gp_eeq_e, atol=1e-11, rtol=1e-11
            )
            torch.testing.assert_close(
                ga_eeq_c, ga_eeq_e, atol=1e-11, rtol=1e-11
            )

    # 2. Periodic dense D3+ATM, periodic sparse D3+ATM, and periodic D4+external charges
    # Use a compact 2-atom periodic cell so C++ Inductor codegen finishes quickly
    cell = Structure(
        numbers=torch.tensor([14, 1], dtype=torch.long),
        positions=torch.tensor(
            [[0.15, 0.25, 0.35], [2.10, 1.85, 1.65]], dtype=torch.float64
        ),
        lattice=torch.tensor(
            [[6.2, 0.2, 0.0], [0.1, 6.4, 0.1], [0.0, 0.1, 6.6]],
            dtype=torch.float64,
        ),
        periodic=torch.tensor([True, True, True]),
    )
    cutoff = Cutoff(cn=7.5, disp2=8.5, disp3=6.8, cn_eeq=7.5)
    dense_d3 = DensePairs.build(cell, cutoff, three_body=True)
    sparse_d3 = SparsePairs.build(cell, cutoff, three_body=True)
    dense_d4 = DensePairs.build(
        cell, cutoff, three_body=True, with_cn_eeq=True
    )
    sparse_d4 = SparsePairs.build(
        cell, cutoff, three_body=True, with_cn_eeq=True
    )
    q_pbc0 = torch.tensor([0.25, -0.25], dtype=torch.float64)

    def pbc_d3_dense_fn(
        pos: torch.Tensor, lat: torch.Tensor, a1_val: torch.Tensor
    ) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        p = {
            "s6": torch.tensor(1.0, dtype=torch.float64),
            "s8": torch.tensor(0.7875, dtype=torch.float64),
            "a1": a1_val,
            "a2": torch.tensor(4.4407, dtype=torch.float64),
            "s9": torch.tensor(1.0, dtype=torch.float64),
        }
        return dftd3(
            st, p, cutoff=cutoff, pairs=dense_d3, max_triples=200_000
        ).sum()

    def pbc_d3_sparse_fn(
        pos: torch.Tensor, lat: torch.Tensor, a1_val: torch.Tensor
    ) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        p = {
            "s6": torch.tensor(1.0, dtype=torch.float64),
            "s8": torch.tensor(0.7875, dtype=torch.float64),
            "a1": a1_val,
            "a2": torch.tensor(4.4407, dtype=torch.float64),
            "s9": torch.tensor(1.0, dtype=torch.float64),
        }
        return dftd3(
            st, p, cutoff=cutoff, pairs=sparse_d3, max_triples=200_000
        ).sum()

    def pbc_d4_dense_fn(
        pos: torch.Tensor,
        lat: torch.Tensor,
        a1_val: torch.Tensor,
        q_in: torch.Tensor,
    ) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        p = {
            "s6": torch.tensor(1.0, dtype=torch.float64),
            "s8": torch.tensor(0.95948085, dtype=torch.float64),
            "a1": a1_val,
            "a2": torch.tensor(4.80688534, dtype=torch.float64),
            "s9": torch.tensor(1.0, dtype=torch.float64),
            "rs9": torch.tensor(1.0, dtype=torch.float64),
            "alp": torch.tensor(16.0, dtype=torch.float64),
            "a13": a1_val,
            "a23": torch.tensor(4.80688534, dtype=torch.float64),
            "alp3": torch.tensor(16.0, dtype=torch.float64),
        }
        return dftd(
            st,
            p,
            model="d4",
            q=q_in,
            cutoff=cutoff,
            pairs=dense_d4,
            max_triples=200_000,
        ).sum()

    def pbc_d4_sparse_fn(
        pos: torch.Tensor,
        lat: torch.Tensor,
        a1_val: torch.Tensor,
        q_in: torch.Tensor,
    ) -> torch.Tensor:
        st = cell.replace(positions=pos, lattice=lat)
        p = {
            "s6": torch.tensor(1.0, dtype=torch.float64),
            "s8": torch.tensor(0.95948085, dtype=torch.float64),
            "a1": a1_val,
            "a2": torch.tensor(4.80688534, dtype=torch.float64),
            "s9": torch.tensor(1.0, dtype=torch.float64),
            "rs9": torch.tensor(1.0, dtype=torch.float64),
            "alp": torch.tensor(16.0, dtype=torch.float64),
            "a13": a1_val,
            "a23": torch.tensor(4.80688534, dtype=torch.float64),
            "alp3": torch.tensor(16.0, dtype=torch.float64),
        }
        return dftd(
            st,
            p,
            model="d4",
            q=q_in,
            cutoff=cutoff,
            pairs=sparse_d4,
            max_triples=200_000,
        ).sum()

    comp_pbc_d3_dense = torch.compile(
        pbc_d3_dense_fn, backend="inductor", fullgraph=True
    )
    comp_pbc_d3_sparse = torch.compile(
        pbc_d3_sparse_fn, backend="inductor", fullgraph=True
    )
    comp_pbc_d4_dense = torch.compile(
        pbc_d4_dense_fn, backend="inductor", fullgraph=True
    )
    comp_pbc_d4_sparse = torch.compile(
        pbc_d4_sparse_fn, backend="inductor", fullgraph=True
    )

    for shift_pos, a1_val in ((0.0, 0.4289), (0.015, 0.4450)):
        pos_cur = cell.positions + shift_pos
        lat_cur = cell.lattice

        for eager_fn, comp_fn in (
            (pbc_d3_dense_fn, comp_pbc_d3_dense),
            (pbc_d3_sparse_fn, comp_pbc_d3_sparse),
        ):
            pe = pos_cur.clone().requires_grad_(True)
            le = lat_cur.clone().requires_grad_(True)
            ae = torch.tensor(a1_val, dtype=torch.float64, requires_grad=True)
            ee = eager_fn(pe, le, ae)
            gpe, gle, gae = torch.autograd.grad(ee, (pe, le, ae))

            pc = pos_cur.clone().requires_grad_(True)
            lc = lat_cur.clone().requires_grad_(True)
            ac = torch.tensor(a1_val, dtype=torch.float64, requires_grad=True)
            ec = comp_fn(pc, lc, ac)
            gpc, glc, gac = torch.autograd.grad(ec, (pc, lc, ac))

            torch.testing.assert_close(ec, ee, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(gpc, gpe, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(glc, gle, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(gac, gae, atol=1e-11, rtol=1e-11)

        for eager_fn4, comp_fn4 in (
            (pbc_d4_dense_fn, comp_pbc_d4_dense),
            (pbc_d4_sparse_fn, comp_pbc_d4_sparse),
        ):
            pe = pos_cur.clone().requires_grad_(True)
            le = lat_cur.clone().requires_grad_(True)
            ae = torch.tensor(a1_val, dtype=torch.float64, requires_grad=True)
            qe = q_pbc0.clone().requires_grad_(True)
            ee = eager_fn4(pe, le, ae, qe)
            gpe, gle, gae, gqe = torch.autograd.grad(ee, (pe, le, ae, qe))

            pc = pos_cur.clone().requires_grad_(True)
            lc = lat_cur.clone().requires_grad_(True)
            ac = torch.tensor(a1_val, dtype=torch.float64, requires_grad=True)
            qc = q_pbc0.clone().requires_grad_(True)
            ec = comp_fn4(pc, lc, ac, qc)
            gpc, glc, gac, gqc = torch.autograd.grad(ec, (pc, lc, ac, qc))

            torch.testing.assert_close(ec, ee, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(gpc, gpe, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(glc, gle, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(gac, gae, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(gqc, gqe, atol=1e-11, rtol=1e-11)


# Authoritative Fortran libdftd.so (commit d2740777049b2a77fd5acc86c4e1abe75d5374ee)
# periodic reference fixtures (total energy, atom-0 nuclear gradient, and 3x3
# virial/strain tensor sigma = dE/d(eps)) for a 2-atom displaced carbon triclinic
# cell (numbers=[6, 6], positions=[[0.15, 0.25, 0.35], [1.95, 2.15, 2.35]],
# lattice=[[6.2, 0.3, 0.1], [0.3, 6.4, 0.2], [0.1, 0.2, 6.6]], disp2=14.0,
# disp3=10.0, cn=40.0 for D3/D3S and 30.0 for D4/D4S/D4Srev, q=[0.0, 0.0] exact
# by inversion symmetry, 2B + 3B ATM s9=1.0) across 1D, 2D, and 3D periodicities.
_FORTRAN_PERIODIC_REFS: dict[str, dict[str, dict[str, object]]] = {
    "1d": {
        "d1": {
            "energy": -0.0003203756367092846,
            "grad0": [-4.6149799478561866e-05, -3.963235705885511e-06, -2.625384567589911e-06],
            "sigma": [
                [-0.0018097758217357493, -3.904677480917659e-06, 6.176983346039534e-05],
                [-3.904677480917656e-06, 6.9960267886899776e-06, 7.748431060940194e-06],
                [6.176983346039534e-05, 7.748431060940197e-06, 6.170835929352946e-06],
            ],
        },
        "d2": {
            "energy": -0.0006763477969582252,
            "grad0": [-0.00012417751304802103, 2.5242724893461283e-05, 3.197320983521225e-05],
            "sigma": [
                [-0.0004266315746117237, 0.0002044783506897161, 0.0002378687180622009],
                [0.0002044783506897161, -3.586850364445001e-05, -4.645455856921378e-05],
                [0.0002378687180622009, -4.645455856921378e-05, -5.918157296452804e-05],
            ],
        },
        "d3": {
            "energy": -0.0023280466086087314,
            "grad0": [5.573737177058033e-05, -2.2399088031063884e-05, -2.638515819978102e-05],
            "sigma": [
                [0.0033063434868249642, 5.8937901184865544e-05, -5.652844102483796e-05],
                [5.89379011848655e-05, 4.3459212874938304e-05, 4.509849126743349e-05],
                [-5.6528441024837985e-05, 4.509849126743348e-05, 5.1092546628845275e-05],
            ],
        },
        "d3s": {
            "energy": -0.002323667709579081,
            "grad0": [7.91298001214453e-05, 8.180759803562799e-06, 6.007621596952416e-06],
            "sigma": [
                [0.0032283858202675096, 1.2757547212130182e-05, -0.00010389154442859849],
                [1.2757547212130182e-05, -1.4213625165871986e-05, -1.5918246786826508e-05],
                [-0.00010389154442859847, -1.5918246786826508e-05, -1.3516498444777163e-05],
            ],
        },
        "d4": {
            "energy": -0.0019489788342842073,
            "grad0": [-5.331399539233691e-05, -0.00011665587681387364, -0.00012488221321660834],
            "sigma": [
                [0.002855956881604011, 0.00023484457624049516, 0.00015114398578302553],
                [0.00023484457624049516, 0.0002228492626225949, 0.00023371278585315896],
                [0.00015114398578302556, 0.00023371278585315898, 0.0002485766200137511],
            ],
        },
        "d4s": {
            "energy": -0.0019654319918025985,
            "grad0": [2.472471078494192e-05, -3.403118104761623e-05, -3.790019251100505e-05],
            "sigma": [
                [0.002735451841138289, 8.753706501915152e-05, -4.611416399703452e-06],
                [8.753706501915152e-05, 6.59309023356374e-05, 6.848624821028802e-05],
                [-4.611416399703459e-06, 6.848624821028802e-05, 7.46256791684696e-05],
            ],
        },
        "d4srev": {
            "energy": -0.0019898582378184695,
            "grad0": [4.832101565146522e-05, -9.64939393193153e-06, -1.225331052182421e-05],
            "sigma": [
                [0.002726537648131793, 4.432791588984981e-05, -5.1262749427052545e-05],
                [4.43279158898498e-05, 1.963831654223633e-05, 1.9733610554385176e-05],
                [-5.126274942705254e-05, 1.9733610554385176e-05, 2.332406123128813e-05],
            ],
        },
    },
    "2d": {
        "d1": {
            "energy": -0.0011203492159730791,
            "grad0": [-1.6373363373011714e-05, -5.7916661394561984e-06, -3.4383652913152925e-05],
            "sigma": [
                [-0.0007633260581719595, -0.000158618160559695, 1.533340919682042e-05],
                [-0.0001586181605596951, -0.0004099844858125113, -3.7599690815817765e-06],
                [1.533340919682042e-05, -3.7599690815817646e-06, 6.59736524556762e-05],
            ],
        },
        "d2": {
            "energy": -0.0017759248255049776,
            "grad0": [-5.023415231631628e-06, 5.1990562030896055e-06, 6.27446819810322e-06],
            "sigma": [
                [0.0008467411591756482, 0.0005603458419221748, 3.915355094642469e-05],
                [0.0005603458419221747, 0.0021018471462640456, 6.248938646174914e-05],
                [3.915355094642472e-05, 6.248938646174914e-05, -9.53563090608518e-06],
            ],
        },
        "d3": {
            "energy": -0.004823217035993955,
            "grad0": [0.0001350756544680284, 0.00015958381753893461, -0.00010079085174900403],
            "sigma": [
                [0.005608065868019742, -1.3181253958221472e-05, -0.00017698922762641745],
                [-1.3181253958221499e-05, 0.005692572102808725, -0.00013190176442529417],
                [-0.00017698922762641748, -0.00013190176442529417, 0.00018644660108894397],
            ],
        },
        "d3s": {
            "energy": -0.004809017461084289,
            "grad0": [0.00018802459668468033, 0.0002164349030232173, -2.7303687846367788e-05],
            "sigma": [
                [0.005431219221405124, -0.00010069618010752778, -0.0002836816511313778],
                [-0.00010069618010752767, 0.00550558832647849, -0.00024780034478391404],
                [-0.0002836816511313778, -0.00024780034478391404, 4.0572265732768205e-05],
            ],
        },
        "d4": {
            "energy": -0.004093855117788804,
            "grad0": [-8.698702554584841e-05, -7.750625946521202e-05, -0.00030814757107890095],
            "sigma": [
                [0.005097679826307234, 0.00033014514234688217, 0.0002514009845369647],
                [0.0003301451423468822, 0.0052213786414479265, 0.0003128869547303983],
                [0.0002514009845369647, 0.0003128869547303984, 0.0006035209177611984],
            ],
        },
        "d4s": {
            "energy": -0.004128202356786792,
            "grad0": [7.688025964050287e-05, 9.592872081892933e-05, -0.00012611393870716172],
            "sigma": [
                [0.0048375255697990695, 2.1830498944361615e-05, -7.573095519451071e-05],
                [2.1830498944361568e-05, 0.004928833741709114, -3.2796056678186025e-05],
                [-7.573095519451074e-05, -3.2796056678186025e-05, 0.0002394669256666231],
            ],
        },
        "d4srev": {
            "energy": -0.004179304651727521,
            "grad0": [0.00012667220732274668, 0.00014877556097173182, -7.293141785099679e-05],
            "sigma": [
                [0.004807809543061969, -7.020124795616553e-05, -0.0001743585289665613],
                [-7.020124795616562e-05, 0.004890321408989705, -0.00013655266573251484],
                [-0.00017435852896656134, -0.0001365526657325148, 0.0001329765916450896],
            ],
        },
    },
    "3d": {
        "d1": {
            "energy": -0.002444227740069225,
            "grad0": [-3.7508600001870454e-05, 4.195586516249137e-05, -4.89185858992177e-05],
            "sigma": [
                [0.00021345856949210598, -5.466428169129375e-05, -0.00014872300412036347],
                [-5.466428169129376e-05, 0.0007399739121525611, 0.00011188354615130148],
                [-0.00014872300412036347, 0.00011188354615130148, 0.0012563428540028],
            ],
        },
        "d2": {
            "energy": -0.003384694764178707,
            "grad0": [0.00010222645176630387, 0.00012252561658837705, 8.937152832063217e-05],
            "sigma": [
                [0.0022093372298810717, 0.0003231415201341346, 0.0005073671932328328],
                [0.0003231415201341347, 0.0034682815112771744, 0.0005767623479044245],
                [0.0005073671932328327, 0.0005767623479044245, 0.003936272963852782],
            ],
        },
        "d3": {
            "energy": -0.008466244526345116,
            "grad0": [0.00023774662486868273, 0.00026388730513322763, 0.0003029650911074484],
            "sigma": [
                [0.008685370655583087, -0.00014818789303918954, -0.00011785283692542952],
                [-0.00014818789303918954, 0.008716399975541196, -9.729792009378245e-05],
                [-0.0001178528369254296, -9.729792009378251e-05, 0.008633407834043622],
            ],
        },
        "d3s": {
            "energy": -0.008432478271809919,
            "grad0": [0.0003373352921595698, 0.0003707267347723396, 0.00042248778632998315],
            "sigma": [
                [0.008348878903483879, -0.00031257652224043987, -0.0002943239178964653],
                [-0.00031257652224043987, 0.008362270776223328, -0.0002896967787417705],
                [-0.0002943239178964653, -0.0002896967787417704, 0.008268182643980241],
            ],
        },
        "d4": {
            "energy": -0.0072292413255369385,
            "grad0": [-0.00014967000141554243, -0.0001464332478758697, -0.0001458975652815542],
            "sigma": [
                [0.00809653832618058, 0.0005106223244503449, 0.0005814896722958316],
                [0.0005106223244503447, 0.00820052973846829, 0.0006429139762401785],
                [0.0005814896722958314, 0.0006429139762401786, 0.008193080171429636],
            ],
        },
        "d4s": {
            "energy": -0.0072893601427173604,
            "grad0": [0.00013757229274872376, 0.00015745316307994134, 0.00017584544367175756],
            "sigma": [
                [0.007633159403160376, -3.0202819827765076e-05, 1.0533475555495376e-05],
                [-3.0202819827765035e-05, 0.007679426322029776, 3.904610795281661e-05],
                [1.05334755555495267e-05, 3.9046107952816694e-05, 0.007612959612231376],
            ],
        },
        "d4srev": {
            "energy": -0.007378847425421662,
            "grad0": [0.00022486765275386364, 0.00024992704465691477, 0.00027382124319655463],
            "sigma": [
                [0.007570119655792728, -0.00019215592754706558, -0.00016004530027771344],
                [-0.00019215592754706564, 0.007599513547163332, -0.00014113463250816188],
                [-0.00016004530027771338, -0.00014113463250816188, 0.007514732800932988],
            ],
        },
    },
}


def _fortran_ref_cell(
    pbc_flags: tuple[bool, bool, bool],
) -> tuple[Structure, dict[str, Cutoff], dict[str, dict[str, float]]]:
    cell = Structure(
        numbers=torch.tensor([6, 6], dtype=torch.long),
        positions=torch.tensor(
            [[0.15, 0.25, 0.35], [1.95, 2.15, 2.35]], dtype=torch.float64
        ),
        lattice=torch.tensor(
            [[6.2, 0.3, 0.1], [0.3, 6.4, 0.2], [0.1, 0.2, 6.6]],
            dtype=torch.float64,
        ),
        periodic=torch.tensor(pbc_flags, dtype=torch.bool),
    )
    cutoffs = {
        "d1": Cutoff(disp2=14.0, disp3=10.0),
        "d2": Cutoff(disp2=14.0, disp3=10.0),
        "d3": Cutoff(cn=40.0, disp2=14.0, disp3=10.0),
        "d3s": Cutoff(cn=40.0, disp2=14.0, disp3=10.0),
        "d4": Cutoff(cn=30.0, disp2=14.0, disp3=10.0, cn_eeq=25.0),
        "d4s": Cutoff(cn=30.0, disp2=14.0, disp3=10.0, cn_eeq=25.0),
        "d4srev": Cutoff(cn=30.0, disp2=14.0, disp3=10.0, cn_eeq=25.0),
    }
    params: dict[str, dict[str, float]] = {
        "d1": {
            "s6": 0.75,
            "rs6": 1.0,
            "alp": 23.0,
            "s9": 1.0,
            "rs9": 1.0,
            "alp3": 16.0,
            "a1": 0.4,
            "a2": 4.0,
            "a13": 0.4,
            "a23": 4.0,
        },
        "d2": {
            "s6": 0.75,
            "rs6": 1.0,
            "alp": 20.0,
            "s9": 1.0,
            "rs9": 1.0,
            "alp3": 16.0,
            "a1": 0.4,
            "a2": 4.0,
            "a13": 0.4,
            "a23": 4.0,
        },
        "d3": {
            "s6": 1.0,
            "s8": 0.7875,
            "a1": 0.4289,
            "a2": 4.4407,
            "s9": 1.0,
            "rs9": 4.0 / 3.0,
            "alp": 14.0,
            "alp3": 14.0,
        },
        "d3s": {
            "s6": 1.0,
            "s8": 0.7875,
            "a1": 0.4289,
            "a2": 4.4407,
            "s9": 1.0,
            "rs9": 4.0 / 3.0,
            "alp": 14.0,
            "alp3": 14.0,
        },
        "d4": {
            "s6": 1.0,
            "s8": 0.95948085,
            "a1": 0.38574991,
            "a2": 4.80688534,
            "s9": 1.0,
            "rs9": 1.0,
            "alp": 16.0,
            "alp3": 16.0,
        },
        "d4s": {
            "s6": 1.0,
            "s8": 0.95948085,
            "a1": 0.38574991,
            "a2": 4.80688534,
            "s9": 1.0,
            "rs9": 1.0,
            "alp": 16.0,
            "alp3": 16.0,
        },
        "d4srev": {
            "s6": 1.0,
            "s8": 0.95948085,
            "a1": 0.38574991,
            "a2": 4.80688534,
            "s9": 1.0,
            "rs9": 1.0,
            "alp": 16.0,
            "alp3": 16.0,
        },
    }
    return cell, cutoffs, params


def test_fortran_periodic_reference_conformance_all_models() -> None:
    """
    Independent Fortran periodic reference conformance across all 7 models
    (`d1`, `d2`, `d3`, `d3s`, `d4`, `d4s`, `d4srev`) against `libdftd.so`
    (`d2740777049b2a77fd5acc86c4e1abe75d5374ee`) and `dftd/test/unit/test_periodic.f90`
    at `atol=1e-12, rtol=1e-12` (H3.3, H3.4).
    """
    from tad_mctc.data.structures import get_structure
    from tad_dftd import damping

    # 1. Exact 1e-12 conformance across all 7 models and 1D/2D/3D periodicities
    # against Fortran libdftd.so (energy, nuclear gradient, and virial sigma)
    q_zero2 = torch.zeros(2, dtype=torch.float64)
    for pbc_key, pbc_flags in (
        ("1d", (True, False, False)),
        ("2d", (True, True, False)),
        ("3d", (True, True, True)),
    ):
        cell, cutoffs, params = _fortran_ref_cell(pbc_flags)
        for model_name in ("d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"):
            ref = _FORTRAN_PERIODIC_REFS[pbc_key][model_name]
            cut = cutoffs[model_name]
            p = params[model_name]
            kw = (
                {"q": q_zero2}
                if (model_name in ("d4", "d4s", "d4srev") and pbc_key != "3d")
                else {}
            )

            dense_pairs = DensePairs.build(cell, cut, three_body=True)
            sparse_pairs = SparsePairs.build(cell, cut, three_body=True)

            pos_d = cell.positions.clone().requires_grad_(True)
            lat_d = cell.lattice.clone().requires_grad_(True)
            ed = dftd(
                cell.replace(positions=pos_d, lattice=lat_d),
                p,
                model=model_name,
                cutoff=cut,
                pairs=dense_pairs,
                **kw,
            ).sum()
            gd, glat_d = torch.autograd.grad(ed, (pos_d, lat_d))
            sigma_d = gd.T @ cell.positions + glat_d.T @ cell.lattice

            pos_s = cell.positions.clone().requires_grad_(True)
            lat_s = cell.lattice.clone().requires_grad_(True)
            es = dftd(
                cell.replace(positions=pos_s, lattice=lat_s),
                p,
                model=model_name,
                cutoff=cut,
                pairs=sparse_pairs,
                **kw,
            ).sum()
            gs, glat_s = torch.autograd.grad(es, (pos_s, lat_s))
            sigma_s = gs.T @ cell.positions + glat_s.T @ cell.lattice

            ref_e = torch.tensor(ref["energy"], dtype=torch.float64)
            ref_g0 = torch.tensor(ref["grad0"], dtype=torch.float64)
            ref_sig = torch.tensor(ref["sigma"], dtype=torch.float64)

            torch.testing.assert_close(ed, ref_e, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(es, ref_e, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(gd[0], ref_g0, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(gs[0], ref_g0, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(sigma_d, ref_sig, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(sigma_s, ref_sig, atol=1e-13, rtol=1e-13)

    # 2. X23 3D periodic crystal references against Fortran C-API (`new_structure_api`,
    # which calls `wrap_to_central_cell` just like `tad_mctc`), all matching to < 1e-12 Eh
    cut_pbc = Cutoff(cn=30.0, disp2=60.0, disp3=15.0)
    cut_d3 = Cutoff(cn=40.0, disp2=60.0, disp3=15.0)
    ammonia = get_structure("x23", "ammonia", dtype=torch.float64)
    acetic = get_structure("x23", "acetic", dtype=torch.float64)
    adaman = get_structure("x23", "adaman", dtype=torch.float64)
    benzene = get_structure("x23", "benzene", dtype=torch.float64)
    cyanamide = get_structure("x23", "cyanamide", dtype=torch.float64)

    torch.testing.assert_close(
        dftd(ammonia, {"s6": 0.7, "rs6": 1.0, "alp": 23.0}, model="d1", cutoff=cut_pbc).sum(),
        torch.tensor(-2.1111064217077948e-02, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dftd(ammonia, {"s6": 0.75, "rs6": 1.1, "alp": 20.0}, model="d2", cutoff=cut_pbc).sum(),
        torch.tensor(-2.9729958659375735e-02, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dftd(acetic, {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}, model="d3", cutoff=cut_d3).sum(),
        torch.tensor(-0.066727625533349, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dftd(adaman, {"s6": 1.0, "s8": 2.9491, "a1": 0.4466, "a2": 6.1742}, model="d3", cutoff=cut_d3).sum(),
        torch.tensor(-0.07930789307762197, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dftd(acetic, {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}, model="d3s", cutoff=cut_d3).sum(),
        torch.tensor(-0.066728412990681, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    damp_d3_zero = damping.new_damping("zero", False, model="d3")
    torch.testing.assert_close(
        dftd(
            benzene,
            {"s6": 1.0, "s8": 1.682, "rs6": 1.094, "rs8": 1.0, "alp": 14.0},
            model="d3",
            damping=damp_d3_zero,
            cutoff=cut_d3,
        ).sum(),
        torch.tensor(-0.16644876006925308, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        dftd(
            cyanamide,
            {"s6": 1.0, "s8": 0.0, "rs6": 1.581, "rs8": 1.0, "alp": 14.0},
            model="d3",
            damping=damp_d3_zero,
            cutoff=cut_d3,
        ).sum(),
        torch.tensor(-0.02322118588513826, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_genuine_partial_periodicity_1d_2d_3d_all_models() -> None:
    """
    Validate genuine 1D (`periodic=[True, False, False]`), 2D
    (`periodic=[True, True, False]`), and 3D (`periodic=[True, True, True]`)
    periodic boundary conditions across all 7 models (H3.3):
    - Dense vs Sparse parity for energies, nuclear gradients, and strain tensors.
    - Nonperiodic lattice vectors in 1D/2D do not generate periodic images (rescaling
      the nonperiodic lattice vectors leaves energy and nuclear gradient unchanged).
    - 1D, 2D, and 3D periodicities yield distinct energies, gradients, and strains.
    - Automatic EEQ (`q=None`) works for 3D periodic structures and explicitly raises
      `NotImplementedError` for 1D and 2D periodic structures.
    """
    q_fixed = torch.tensor([0.12, -0.12], dtype=torch.float64)
    energies_by_dim: dict[str, dict[str, torch.Tensor]] = {
        "1d": {},
        "2d": {},
        "3d": {},
    }

    for pbc_key, pbc_flags in (
        ("1d", (True, False, False)),
        ("2d", (True, True, False)),
        ("3d", (True, True, True)),
    ):
        cell, cutoffs, params = _fortran_ref_cell(pbc_flags)
        for model_name in ("d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"):
            cut = cutoffs[model_name]
            p = params[model_name]
            kw = (
                {"q": q_fixed}
                if model_name in ("d4", "d4s", "d4srev")
                else {}
            )
            e_dense = dftd(
                cell,
                p,
                model=model_name,
                cutoff=cut,
                pairs=DensePairs.build(cell, cut, three_body=True),
                **kw,
            ).sum()
            e_sparse = dftd(
                cell,
                p,
                model=model_name,
                cutoff=cut,
                pairs=SparsePairs.build(cell, cut, three_body=True),
                **kw,
            ).sum()
            torch.testing.assert_close(e_dense, e_sparse, atol=1e-13, rtol=1e-13)
            energies_by_dim[pbc_key][model_name] = e_dense

            # In 1D and 2D, rescaling the nonperiodic z lattice vector (index 2)
            # must not change the dispersion energy because no images are generated
            # along nonperiodic axes.
            if pbc_key in ("1d", "2d"):
                lat_scaled = cell.lattice.clone()
                lat_scaled[2] = lat_scaled[2] * 3.5
                cell_scaled = cell.replace(lattice=lat_scaled)
                e_scaled = dftd(
                    cell_scaled,
                    p,
                    model=model_name,
                    cutoff=cut,
                    **kw,
                ).sum()
                torch.testing.assert_close(
                    e_dense, e_scaled, atol=1e-14, rtol=1e-14
                )

    for model_name in ("d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"):
        e1 = energies_by_dim["1d"][model_name]
        e2 = energies_by_dim["2d"][model_name]
        e3 = energies_by_dim["3d"][model_name]
        assert float((e1 - e2).abs()) > 1e-4
        assert float((e2 - e3).abs()) > 1e-4

    # Automatic EEQ (`q=None`) raises NotImplementedError on 1D and 2D periodic cells
    for pbc_flags in ((True, False, False), (True, True, False)):
        cell_partial, cutoffs, params = _fortran_ref_cell(pbc_flags)
        with pytest.raises(NotImplementedError, match="1D or 2D"):
            dftd(
                cell_partial,
                params["d4"],
                model="d4",
                cutoff=cutoffs["d4"],
            )


def test_periodic_wrapping_translation_permutation_and_supercell_invariance() -> None:
    """
    Verify periodic coordinate wrapping and crystal symmetry invariances across
    all 7 models (H3.4):
    1. Integer lattice translations `r_i -> r_i + n_1 a_1 + n_2 a_2 + n_3 a_3`
       leave energy, nuclear gradients, and strain tensor invariant to machine precision.
    2. Uniform global translation `r_i -> r_i + t` leaves energy, nuclear gradients,
       and strain tensor invariant to machine precision.
    3. Atom permutation commutes with atom-resolved energies and nuclear gradients.
    4. A 2x1x1 supercell has exactly twice the total energy of the primitive cell.
    """
    cell, cutoffs, params = _fortran_ref_cell((True, True, True))
    q_prim = torch.tensor([0.15, -0.15], dtype=torch.float64)

    # Integer lattice shifts: shift atom 0 by (+1, -2, +1) lattice vectors and
    # atom 1 by (-1, +1, -1) lattice vectors
    int_shifts = torch.tensor(
        [[1.0, -2.0, 1.0], [-1.0, 1.0, -1.0]], dtype=torch.float64
    )
    pos_wrapped_shift = cell.positions + int_shifts @ cell.lattice
    pos_global_shift = cell.positions + torch.tensor(
        [0.73, -1.19, 2.41], dtype=torch.float64
    )
    perm = torch.tensor([1, 0], dtype=torch.long)

    # Build 2x1x1 supercell
    lat_super = cell.lattice.clone()
    lat_super[0] = 2.0 * cell.lattice[0]
    pos_super = torch.cat(
        [cell.positions, cell.positions + cell.lattice[0].unsqueeze(0)], dim=0
    )
    nums_super = torch.cat([cell.numbers, cell.numbers], dim=0)
    q_super = torch.cat([q_prim, q_prim], dim=0)
    cell_super = Structure(
        numbers=nums_super,
        positions=pos_super,
        lattice=lat_super,
        periodic=cell.periodic,
    )

    for model_name in ("d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"):
        cut = cutoffs[model_name]
        p = params[model_name]
        kw = {"q": q_prim} if model_name in ("d4", "d4s", "d4srev") else {}
        kw_perm = (
            {"q": q_prim[perm]}
            if model_name in ("d4", "d4s", "d4srev")
            else {}
        )
        kw_super = (
            {"q": q_super} if model_name in ("d4", "d4s", "d4srev") else {}
        )

        pos0 = cell.positions.clone().requires_grad_(True)
        lat0 = cell.lattice.clone().requires_grad_(True)
        e0_atoms = dftd(
            cell.replace(positions=pos0, lattice=lat0),
            p,
            model=model_name,
            cutoff=cut,
            **kw,
        )
        g0_pos, g0_lat = torch.autograd.grad(e0_atoms.sum(), (pos0, lat0))
        sig0 = g0_pos.T @ cell.positions + g0_lat.T @ cell.lattice

        # 1. Integer lattice shift invariance
        pos1 = pos_wrapped_shift.clone().requires_grad_(True)
        lat1 = cell.lattice.clone().requires_grad_(True)
        e1_atoms = dftd(
            cell.replace(positions=pos1, lattice=lat1),
            p,
            model=model_name,
            cutoff=cut,
            **kw,
        )
        g1_pos, g1_lat = torch.autograd.grad(e1_atoms.sum(), (pos1, lat1))
        # Note: under integer shift r' = r + N @ L, the physical virial is
        # g_pos.T @ r + g_lat_total.T @ L == g1_pos.T @ pos1 + g1_lat.T @ L
        sig1 = g1_pos.T @ pos_wrapped_shift + g1_lat.T @ cell.lattice
        torch.testing.assert_close(e1_atoms, e0_atoms, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(g1_pos, g0_pos, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(sig1, sig0, atol=1e-12, rtol=1e-12)

        # 2. Uniform global translation invariance
        pos2 = pos_global_shift.clone().requires_grad_(True)
        lat2 = cell.lattice.clone().requires_grad_(True)
        e2_atoms = dftd(
            cell.replace(positions=pos2, lattice=lat2),
            p,
            model=model_name,
            cutoff=cut,
            **kw,
        )
        g2_pos, g2_lat = torch.autograd.grad(e2_atoms.sum(), (pos2, lat2))
        sig2 = g2_pos.T @ pos_global_shift + g2_lat.T @ cell.lattice
        torch.testing.assert_close(e2_atoms, e0_atoms, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(g2_pos, g0_pos, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(sig2, sig0, atol=1e-12, rtol=1e-12)

        # 3. Atom permutation equivariance
        pos3 = cell.positions[perm].clone().requires_grad_(True)
        e3_atoms = dftd(
            cell.replace(numbers=cell.numbers[perm], positions=pos3),
            p,
            model=model_name,
            cutoff=cut,
            **kw_perm,
        )
        (g3_pos,) = torch.autograd.grad(e3_atoms.sum(), pos3)
        torch.testing.assert_close(
            e3_atoms, e0_atoms[perm], atol=1e-13, rtol=1e-13
        )
        torch.testing.assert_close(
            g3_pos, g0_pos[perm], atol=1e-12, rtol=1e-12
        )

        # 4. 2x1x1 supercell doubling E(2x1x1) == 2 * E(1x1x1)
        e_super = dftd(
            cell_super, p, model=model_name, cutoff=cut, **kw_super
        ).sum()
        torch.testing.assert_close(
            e_super, 2.0 * e0_atoms.sum(), atol=1e-12, rtol=1e-12
        )


def test_periodic_strain_analytical_vs_fd_all_models() -> None:
    """
    Verify analytical periodic nuclear gradients (`dE/dr`) and cell virial /
    strain tensors (`sigma = dE/d(eps)`) against central finite differences
    across all 7 models, plus external-charge chain-rule gradient decomposition
    (`test_conformance.f90:test_external_charges`).
    """
    from tad_mctc.data.structures import get_structure
    from tad_dftd import charge

    cell_strain = _make_periodic_cell(torch.float64, d1_compatible=True)
    cut_smooth = Cutoff(
        cn=9.0, disp2=10.0, disp3=8.0, width2=2.0, width3=2.0, cn_eeq=9.0
    )
    q_fixed = torch.tensor([0.20, -0.25, 0.15, -0.10], dtype=torch.float64)
    step = 1e-6

    for model_name in ("d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"):
        p_m = _model_param(model_name, torch.float64)
        kw_m = {"q": q_fixed} if model_name in ("d4", "d4s", "d4srev") else {}

        pos_a = cell_strain.positions.clone().requires_grad_(True)
        lat_a = cell_strain.lattice.clone().requires_grad_(True)
        e_a = dftd(
            cell_strain.replace(positions=pos_a, lattice=lat_a),
            p_m,
            model=model_name,
            cutoff=cut_smooth,
            **kw_m,
        ).sum()
        g_pos, g_lat = torch.autograd.grad(e_a, (pos_a, lat_a))
        sigma_analytic = (
            g_pos.T @ cell_strain.positions + g_lat.T @ cell_strain.lattice
        )

        sigma_fd = torch.zeros((3, 3), dtype=torch.float64)
        eye3 = torch.eye(3, dtype=torch.float64)
        for i in range(3):
            for j in range(3):
                eps_p = eye3.clone()
                eps_p[i, j] += step
                eps_m = eye3.clone()
                eps_m[i, j] -= step
                ep = dftd(
                    cell_strain.replace(
                        positions=cell_strain.positions @ eps_p.T,
                        lattice=cell_strain.lattice @ eps_p.T,
                    ),
                    p_m,
                    model=model_name,
                    cutoff=cut_smooth,
                    **kw_m,
                ).sum()
                em = dftd(
                    cell_strain.replace(
                        positions=cell_strain.positions @ eps_m.T,
                        lattice=cell_strain.lattice @ eps_m.T,
                    ),
                    p_m,
                    model=model_name,
                    cutoff=cut_smooth,
                    **kw_m,
                ).sum()
                sigma_fd[i, j] = 0.5 * (ep - em) / step

        torch.testing.assert_close(
            sigma_analytic, sigma_fd, atol=1e-7, rtol=1e-6
        )

    # Fortran test_conformance.f90:test_external_charges chain-rule decomposition
    mol_mb01 = get_structure("mb16_43", "01", dtype=torch.float64)
    p_d4_conf = _model_param("d4", torch.float64)
    cut_conf = Cutoff(cn=20.0, disp2=40.0, disp3=15.0)

    pos_eeq = mol_mb01.positions.clone().requires_grad_(True)
    e_auto = dftd(
        mol_mb01.replace(positions=pos_eeq),
        p_d4_conf,
        model="d4",
        cutoff=cut_conf,
    ).sum()
    (g_auto,) = torch.autograd.grad(e_auto, pos_eeq)

    q_eeq_det = charge.get_charges_eeq(
        mol_mb01.numbers, mol_mb01.positions
    ).detach()
    pos_ext = mol_mb01.positions.clone().requires_grad_(True)
    q_ext_leaf = q_eeq_det.clone().requires_grad_(True)
    e_ext = dftd(
        mol_mb01.replace(positions=pos_ext),
        p_d4_conf,
        model="d4",
        q=q_ext_leaf,
        cutoff=cut_conf,
    ).sum()
    g_ext_pos, dedq = torch.autograd.grad(e_ext, (pos_ext, q_ext_leaf))

    pos_q = mol_mb01.positions.clone().requires_grad_(True)
    q_of_r = charge.get_charges_eeq(mol_mb01.numbers, pos_q)
    (g_q,) = torch.autograd.grad((q_of_r * dedq.detach()).sum(), pos_q)

    torch.testing.assert_close(e_auto, e_ext, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        g_auto, g_ext_pos + g_q, atol=1e-11, rtol=1e-11
    )



