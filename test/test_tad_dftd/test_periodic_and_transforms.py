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


def test_fortran_periodic_reference_energies_and_strain_all_models() -> None:
    """
    Independent periodic and molecular Fortran reference regressions across all
    7 models (`d1`, `d2`, `d3`, `d3s`, `d4`, `d4s`, `d4srev`) from
    `dftd/test/unit/test_periodic.f90` and `dftd/test/unit/test_conformance.f90`,
    plus periodic finite-difference nuclear gradient and cell strain tensor (`sigma`)
    verification and D4 external-charge chain-rule gradient decomposition (H2.6).
    """
    from tad_mctc.data.structures import get_structure
    from tad_dftd import charge, damping

    cut_pbc = Cutoff(cn=30.0, disp2=60.0, disp3=15.0)
    ammonia = get_structure("x23", "ammonia", dtype=torch.float64)
    acetic = get_structure("x23", "acetic", dtype=torch.float64)
    adaman = get_structure("x23", "adaman", dtype=torch.float64)
    benzene = get_structure("x23", "benzene", dtype=torch.float64)
    cyanamide = get_structure("x23", "cyanamide", dtype=torch.float64)

    # 1. D1 periodic Fortran reference (test_pbed1_ammonia)
    e_d1_nh3 = dftd(
        ammonia,
        {"s6": 0.7, "rs6": 1.0, "alp": 23.0},
        model="d1",
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_d1_nh3,
        torch.tensor(-2.1111064217077948e-02, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    # 2. D2 periodic Fortran reference (test_pbed2_ammonia)
    e_d2_nh3 = dftd(
        ammonia,
        {"s6": 0.75, "rs6": 1.1, "alp": 20.0},
        model="d2",
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_d2_nh3,
        torch.tensor(-2.9729958659375735e-02, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    # 3. D3(BJ) periodic Fortran references (test_pbed3_acetic, test_pbesold3_adaman)
    # Note: X23 acetic, adaman, and cyanamide in mstore have unwrapped coordinates
    # outside [0, 1)^3; tad_mctc folds periodic coordinates into the central cell via
    # wrap_to_central_cell so periodic sums are invariant under integer lattice shifts,
    # shifting finite-box boundary tails near r=30..60 Bohr by ~8e-10..1.6e-8 Eh.
    e_d3_acetic = dftd(
        acetic,
        {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
        model="d3",
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_d3_acetic,
        torch.tensor(-6.6732836815486210e-02, dtype=torch.float64),
        atol=2e-9,
        rtol=5e-8,
    )
    e_d3_adaman = dftd(
        adaman,
        {"s6": 1.0, "s8": 2.9491, "a1": 0.4466, "a2": 6.1742},
        model="d3",
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_d3_adaman,
        torch.tensor(-7.9313514714713124e-02, dtype=torch.float64),
        atol=1e-8,
        rtol=2e-7,
    )

    # 4. D3(0) nondefault zero-damping periodic Fortran references
    # (test_d3_zero_pbc, test_blypd3_zero_benzene, test_m06ld3_zero_cyanamide)
    damp_d3_zero = damping.new_damping("zero", False, model="d3")
    e_d3zero_acetic = dftd(
        acetic,
        {"s6": 1.0, "s8": 0.722, "rs6": 1.217, "rs8": 1.0, "alp": 14.0},
        model="d3",
        damping=damp_d3_zero,
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_d3zero_acetic,
        torch.tensor(-6.20361256858119672e-02, dtype=torch.float64),
        atol=2e-9,
        rtol=5e-8,
    )
    # X23 benzene lies inside the central cell -> exact machine-precision match (< 1e-12)
    e_d3zero_benzene = dftd(
        benzene,
        {"s6": 1.0, "s8": 1.682, "rs6": 1.094, "rs8": 1.0, "alp": 14.0},
        model="d3",
        damping=damp_d3_zero,
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_d3zero_benzene,
        torch.tensor(-0.16647358662571451, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    e_d3zero_cyanamide = dftd(
        cyanamide,
        {"s6": 1.0, "s8": 0.0, "rs6": 1.581, "rs8": 1.0, "alp": 14.0},
        model="d3",
        damping=damp_d3_zero,
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_d3zero_cyanamide,
        torch.tensor(-2.3222556050310025e-02, dtype=torch.float64),
        atol=2e-8,
        rtol=1e-6,
    )

    # 5. D4 & D4S 2D-periodic revPBE-D4-ATM with zero external charges (test_revpbed4_2d)
    # Exact match to Fortran (1e-12) because zero external charges bypass Ewald Wigner-Seitz wrapping
    xyz_2d = torch.tensor(
        [
            [-0.12918412100093, 0.06210659750976, -2.13384498734326],
            [0.12856915667443, -0.07403227791901, 4.02358027265954],
            [-0.12317720857511, 2.75170732207802, -2.13345350602279],
            [2.44816466162280, 1.28612566399214, 4.02317048854901],
        ],
        dtype=torch.float64,
    )
    lat_2d_vac = torch.tensor(
        [
            [4.68837849314507, 0.0, 0.0],
            [-2.36282788044783, 4.04978545156612, 0.0],
            [0.0, 0.0, 100.0],
        ],
        dtype=torch.float64,
    )
    cell_2d = Structure(
        numbers=torch.tensor([6, 6, 6, 6], dtype=torch.long),
        positions=xyz_2d,
        lattice=lat_2d_vac,
        periodic=torch.tensor([True, True, True]),
    )
    param_revpbe_d4 = {
        "s6": 1.0,
        "s9": 1.0,
        "alp": 16.0,
        "rs9": 1.0,
        "s8": 1.74676530,
        "a1": 0.53634900,
        "a2": 3.07261485,
        "a13": 0.53634900,
        "a23": 3.07261485,
        "alp3": 16.0,
    }
    e_revpbe_2d = dftd(
        cell_2d,
        param_revpbe_d4,
        model="d4",
        q=torch.zeros(4, dtype=torch.float64),
        cutoff=cut_pbc,
    ).sum()
    torch.testing.assert_close(
        e_revpbe_2d,
        torch.tensor(-4.9174828244048341e-02, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    # 6. Analytical periodic nuclear gradient and strain tensor (`sigma`) vs central FD
    # across all 7 models (matching Fortran test_util.f90:test_numgrad & test_numsigma)
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
        # Analytical virial/strain derivative dE/d(eps) at eps = I:
        # pos(eps) = pos @ eps.T, lat(eps) = lat @ eps.T => sigma = g_pos.T @ pos + g_lat.T @ lat
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

    # 7. Fortran test_conformance.f90:test_external_charges chain-rule decomposition
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

    # Indirect charge-response force g_q = sum_j (dq_j / dr) * (dE / dq_j)
    pos_q = mol_mb01.positions.clone().requires_grad_(True)
    q_of_r = charge.get_charges_eeq(mol_mb01.numbers, pos_q)
    (g_q,) = torch.autograd.grad((q_of_r * dedq.detach()).sum(), pos_q)

    torch.testing.assert_close(e_auto, e_ext, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(
        g_auto, g_ext_pos + g_q, atol=1e-11, rtol=1e-11
    )


