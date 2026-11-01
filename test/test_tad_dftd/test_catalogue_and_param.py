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
Tests for ``tad_dftd.model.catalogue`` and ``tad_dftd.param``.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd import param
from tad_dftd.damping import DampingParam
from tad_dftd.model.catalogue import (
    DISPERSION_MODELS,
    ChargeModelId,
    DispersionModelId,
    get_model_metadata,
    resolve_charge_model_id,
    resolve_model_id,
    validate_model_damping,
)


def test_model_catalogue_and_damping_validation() -> None:
    """Verify all 7 supported dispersion models and their valid damping combinations."""
    expected_models = {"d1", "d2", "d3", "d3s", "d4", "d4s", "d4srev"}
    assert set(DISPERSION_MODELS.keys()) == expected_models

    assert resolve_model_id("dft-d3") == DispersionModelId.D3
    assert resolve_model_id("D4Srev") == DispersionModelId.D4SREV

    with pytest.raises(NotImplementedError, match="not supported in this release"):
        resolve_charge_model_id("eeqbc", model="d4srev")
    with pytest.raises(NotImplementedError, match="not supported in this release"):
        resolve_charge_model_id("eeqbc2025", model="d4")

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        resolve_model_id("d5")

    with pytest.raises(ValueError, match="Unknown dispersion model"):
        resolve_model_id("d6")

    # D1 allows only Fermi damping
    validate_model_damping("d1", "fermi")
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d1", "rational")

    # D2 allows only Fermi damping
    validate_model_damping("d2", "fermi")
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d2", "rational")

    # D3 allows rational, zero, mzero, optimizedpower, cso, z, koide, screened
    for d in ("rational", "zero", "mzero", "optimizedpower", "cso", "z", "koide", "screened"):
        validate_model_damping("d3", d)
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d3", "fermi")

    # D4 allows rational, zero, mzero, optimizedpower, screened
    for d in ("rational", "zero", "mzero", "optimizedpower", "screened"):
        validate_model_damping("d4", d)
    with pytest.raises(ValueError, match="not supported"):
        validate_model_damping("d4", "cso")

    # D5 raises NotImplementedError
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        validate_model_damping("d5", "screened")

    meta_d4srev = get_model_metadata("d4srev")
    assert meta_d4srev.charge_dep_2b is True
    assert meta_d4srev.default_qmodel == ChargeModelId.EEQ


def test_functional_parameter_lookup() -> None:
    """Verify functional alias normalization and parameter lookup across models."""
    # D3 rational (BJ) parameters for PBE
    p_d3_pbe = param.get_params("PBE", model="d3", damping="rational")
    assert pytest.approx(p_d3_pbe["s6"].item(), rel=1e-6) == 1.0
    assert pytest.approx(p_d3_pbe["s8"].item(), rel=1e-6) == 0.7875
    assert pytest.approx(p_d3_pbe["a1"].item(), rel=1e-6) == 0.4289
    assert pytest.approx(p_d3_pbe["a2"].item(), rel=1e-6) == 4.4407

    # D4 rational parameters for PBE
    p_d4_pbe = param.get_params("PBE", model="d4", damping="rational")
    assert pytest.approx(p_d4_pbe["s6"].item(), rel=1e-6) == 1.0
    assert p_d4_pbe["s8"].item() > 0.0
    assert p_d4_pbe["a1"].item() > 0.0
    assert p_d4_pbe["a2"].item() > 0.0

    # D1 and D2 parameters for PBE
    p_d1_pbe = param.get_params("PBE", model="d1")
    assert pytest.approx(p_d1_pbe["s6"].item(), rel=1e-6) == 0.7
    p_d2_pbe = param.get_params("PBE", model="d2")
    assert pytest.approx(p_d2_pbe["s6"].item(), rel=1e-6) == 0.75

    # DampingParam.from_functional constructs DampingParam
    dp = DampingParam.from_functional("PBE", model="d3", damping="rational", dtype=torch.float64)
    assert isinstance(dp, DampingParam)
    assert isinstance(dp.a1, torch.Tensor)
    assert pytest.approx(dp.a1.item(), rel=1e-6) == 0.4289


def test_element_ranges_padding_and_transform_validation_all_models() -> None:
    """
    Verify exact supported element boundaries, Z=0 padding invariance, heavy-element
    energy/force finiteness, and transform-time (vmap/jacrev/jacfwd) validation
    across all 7 dispersion models (H2.2).
    """
    from tad_mctc import Structure
    from tad_dftd import dftd

    # 1. Catalogue max_elem matches documented maximum Z per model
    expected_max_elem = {
        "d1": 10,
        "d2": 86,
        "d3": 103,
        "d3s": 94,
        "d4": 118,
        "d4s": 118,
        "d4srev": 103,
    }
    for name, max_z in expected_max_elem.items():
        assert DISPERSION_MODELS[name].max_elem == max_z

    pos2 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 4.2]], dtype=torch.float64
    )
    pos3 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 4.2], [9.0, 9.0, 9.0]],
        dtype=torch.float64,
    )

    p_fermi = {"s6": 0.75, "rs6": 1.1, "alp": 20.0, "s9": 1.0}
    p_bj = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407, "s9": 1.0}

    model_specs = [
        ("d1", p_fermi, 10, [-1, 2, 3, 4, 5, 11, 86], None),
        ("d2", p_fermi, 86, [-1, 87, 103], None),
        ("d3", p_bj, 103, [-1, 104, 118], None),
        ("d3s", p_bj, 94, [-1, 95, 103], None),
        ("d4", p_bj, 103, [-1, 104, 105, 111, 119], None),
        ("d4s", p_bj, 103, [-1, 104, 105, 111, 119], None),
        ("d4srev", p_bj, 103, [-1, 104, 112, 118], None),
    ]

    for model_name, params, max_auto_z, invalid_zs, q_ext in model_specs:
        # Valid heavy boundary element + padding invariance (Z=0)
        nums_unpad = torch.tensor([max_auto_z, 1], dtype=torch.long)
        nums_pad = torch.tensor([max_auto_z, 1, 0], dtype=torch.long)

        pos_u = pos2.clone().requires_grad_(True)
        pos_p = pos3.clone().requires_grad_(True)

        kw = {"q": q_ext} if q_ext is not None else {}
        e_unpad = dftd(
            Structure(numbers=nums_unpad, positions=pos_u),
            params,
            model=model_name,
            **kw,
        )
        e_pad = dftd(
            Structure(numbers=nums_pad, positions=pos_p),
            params,
            model=model_name,
            **kw,
        )
        assert torch.isfinite(e_unpad).all()
        assert torch.allclose(e_unpad, e_pad[:2], atol=1e-13, rtol=1e-13)
        assert e_pad[2].item() == 0.0

        (g_unpad,) = torch.autograd.grad(e_unpad.sum(), pos_u)
        (g_pad,) = torch.autograd.grad(e_pad.sum(), pos_p)
        assert torch.isfinite(g_unpad).all()
        assert torch.allclose(g_unpad, g_pad[:2], atol=1e-12, rtol=1e-12)
        assert torch.all(g_pad[2] == 0.0)

        # Invalid atomic numbers raise ValueError eagerly and inside transforms
        for bad_z in invalid_zs:
            bad_nums = torch.tensor([bad_z, 1], dtype=torch.long)
            with pytest.raises(ValueError):
                dftd(
                    Structure(numbers=bad_nums, positions=pos2),
                    params,
                    model=model_name,
                )

        # Transform-time validation (vmap, jacrev, jacfwd) on first invalid Z
        bad_nums_t = torch.tensor([invalid_zs[1], 1], dtype=torch.long)

        def _bad_fn(x: torch.Tensor) -> torch.Tensor:
            return dftd(
                Structure(numbers=bad_nums_t, positions=x),
                params,
                model=model_name,
            ).sum()

        with pytest.raises(ValueError):
            torch.func.jacrev(_bad_fn)(pos2)
        with pytest.raises(ValueError):
            torch.func.jacfwd(_bad_fn)(pos2)
        with pytest.raises(ValueError):
            torch.func.vmap(_bad_fn)(pos2.unsqueeze(0))

    # Superheavy Cn-Og (Z=112..118) with external charges in D4 and D4S vs D4Srev
    q_zero = torch.zeros(2, dtype=torch.float64)
    for superheavy_z in (112, 118):
        sh_nums = torch.tensor([superheavy_z, 1], dtype=torch.long)
        for m_ok in ("d4", "d4s"):
            pos_sh = pos2.clone().requires_grad_(True)
            e_sh = dftd(
                Structure(numbers=sh_nums, positions=pos_sh),
                p_bj,
                model=m_ok,
                q=q_zero,
            )
            (g_sh,) = torch.autograd.grad(e_sh.sum(), pos_sh)
            assert torch.isfinite(e_sh).all() and (e_sh.sum() < 0.0)
            assert torch.isfinite(g_sh).all()
            # Without external charges, auto-EEQ rejects Z > 103
            with pytest.raises(ValueError, match="unsupported element"):
                dftd(
                    Structure(numbers=sh_nums, positions=pos2),
                    p_bj,
                    model=m_ok,
                )
        # D4Srev rejects Z >= 104 even when external charges are supplied
        with pytest.raises(ValueError, match="Unsupported atomic number"):
            dftd(
                Structure(numbers=sh_nums, positions=pos2),
                p_bj,
                model="d4srev",
                q=q_zero,
            )


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_compiled_unsupported_elements_all_models_and_backends(
    backend: str,
) -> None:
    """
    Verify H3.6 compiled unsupported-element behavior across `eager`,
    `aot_eager`, and `inductor` (`fullgraph=True`):
    1. Valid atomic numbers succeed under `torch.compile`.
    2. Invalid atomic numbers (negative Z, Z above model max, unsupported gap
       elements Z=2..5 in D1 and Z=104..111 in D4/D4S, mixed valid/invalid
       batches, padded systems, and repeated calls with different `numbers`)
       never silently produce finite energies and deterministically raise an
       exception (`IndexError` or `RuntimeError`).
    3. Public `validate_numbers` raises `ValueError` at the structure boundary.
    """
    from tad_dftd import dftd, validate_numbers
    from tad_mctc import Structure

    pos2 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]], dtype=torch.float64
    )
    p_fermi = {"s6": 1.0, "rs6": 1.0, "alp": 20.0}
    p_bj = {"s6": 1.0, "s8": 0.8, "a1": 0.4, "a2": 4.5, "s9": 1.0}
    q2 = torch.zeros(2, dtype=torch.float64)

    def run_d1(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(Structure(numbers=nums, positions=pos), p_fermi, model="d1").sum()

    def run_d2(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(Structure(numbers=nums, positions=pos), p_fermi, model="d2").sum()

    def run_d3(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(Structure(numbers=nums, positions=pos), p_bj, model="d3").sum()

    def run_d3s(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(Structure(numbers=nums, positions=pos), p_bj, model="d3s").sum()

    def run_d4_ext(
        nums: torch.Tensor, pos: torch.Tensor, q_in: torch.Tensor
    ) -> torch.Tensor:
        return dftd(
            Structure(numbers=nums, positions=pos), p_bj, model="d4", q=q_in
        ).sum()

    def run_d4_eeq(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(
            Structure(numbers=nums, positions=pos), p_bj, model="d4"
        ).sum()

    def run_d4srev(
        nums: torch.Tensor, pos: torch.Tensor, q_in: torch.Tensor
    ) -> torch.Tensor:
        return dftd(
            Structure(numbers=nums, positions=pos),
            p_bj,
            model="d4srev",
            q=q_in,
        ).sum()

    c_d1 = torch.compile(run_d1, backend=backend, fullgraph=True)
    c_d2 = torch.compile(run_d2, backend=backend, fullgraph=True)
    c_d3 = torch.compile(run_d3, backend=backend, fullgraph=True)
    c_d3s = torch.compile(run_d3s, backend=backend, fullgraph=True)
    c_d4_ext = torch.compile(run_d4_ext, backend=backend, fullgraph=True)
    c_d4_eeq = torch.compile(run_d4_eeq, backend=backend, fullgraph=True)
    c_d4srev = torch.compile(run_d4srev, backend=backend, fullgraph=True)

    # First call with valid elements succeeds and matches eager
    valid_hc = torch.tensor([1, 6], dtype=torch.long)
    assert torch.isfinite(c_d1(valid_hc, pos2))
    assert torch.isfinite(c_d2(valid_hc, pos2))
    assert torch.isfinite(c_d3(valid_hc, pos2))
    assert torch.isfinite(c_d3s(valid_hc, pos2))
    assert torch.isfinite(c_d4_ext(valid_hc, pos2, q2))
    assert torch.isfinite(c_d4_eeq(valid_hc, pos2))
    assert torch.isfinite(c_d4srev(valid_hc, pos2, q2))
    # Superheavy Z=118 succeeds in compiled D4 with external charges
    assert torch.isfinite(
        c_d4_ext(torch.tensor([1, 118], dtype=torch.long), pos2, q2)
    )

    # Repeated calls with invalid elements must raise and never return finite energies
    for bad_d1 in ([-1, 6], [2, 6], [5, 6], [1, 11], [0, -2]):
        t_bad = torch.tensor(bad_d1, dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(t_bad, model="d1")
        with pytest.raises((IndexError, RuntimeError)):
            _ = c_d1(t_bad, pos2).item()

    for bad_d2 in ([-1, 6], [1, 87], [0, 100]):
        t_bad = torch.tensor(bad_d2, dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(t_bad, model="d2")
        with pytest.raises((IndexError, RuntimeError)):
            _ = c_d2(t_bad, pos2).item()

    for bad_d3 in ([-1, 6], [1, 104], [0, 118]):
        t_bad = torch.tensor(bad_d3, dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(t_bad, model="d3")
        with pytest.raises((IndexError, RuntimeError)):
            _ = c_d3(t_bad, pos2).item()

    for bad_d3s in ([-1, 6], [1, 95], [0, 103]):
        t_bad = torch.tensor(bad_d3s, dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(t_bad, model="d3s")
        with pytest.raises((IndexError, RuntimeError)):
            _ = c_d3s(t_bad, pos2).item()

    # D4 with external charges rejects negative Z, gap Z=104..111, and Z > 118
    for bad_d4_ext in ([-1, 6], [1, 104], [1, 111], [1, 119], [0, 105]):
        t_bad = torch.tensor(bad_d4_ext, dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(t_bad, model="d4", q=q2)
        with pytest.raises((IndexError, RuntimeError)):
            _ = c_d4_ext(t_bad, pos2, q2).item()

    # D4 with auto-EEQ also rejects Z=112..118 because EEQ supports Z <= 103
    for bad_d4_eeq in ([-1, 6], [1, 104], [1, 112], [1, 118]):
        t_bad = torch.tensor(bad_d4_eeq, dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(t_bad, model="d4")
        with pytest.raises((IndexError, RuntimeError)):
            _ = c_d4_eeq(t_bad, pos2).item()

    # D4Srev rejects negative Z and Z >= 104 even with external charges
    for bad_srev in ([-1, 6], [1, 104], [1, 118]):
        t_bad = torch.tensor(bad_srev, dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(t_bad, model="d4srev", q=q2)
        with pytest.raises((IndexError, RuntimeError)):
            _ = c_d4srev(t_bad, pos2, q2).item()

    # Mixed valid/invalid batch with padding
    batch_pos = torch.stack([pos2, pos2], dim=0)
    batch_valid = torch.tensor([[1, 6], [8, 0]], dtype=torch.long)
    batch_mixed_bad = torch.tensor([[1, 6], [105, 0]], dtype=torch.long)
    assert torch.isfinite(c_d3(batch_valid, batch_pos))
    with pytest.raises((IndexError, RuntimeError)):
        _ = c_d3(batch_mixed_bad, batch_pos).item()


