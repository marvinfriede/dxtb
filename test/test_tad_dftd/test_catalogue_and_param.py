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

    with pytest.raises(
        NotImplementedError, match="not supported in this release"
    ):
        resolve_charge_model_id("eeqbc", model="d4srev")
    with pytest.raises(
        NotImplementedError, match="not supported in this release"
    ):
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
    for d in (
        "rational",
        "zero",
        "mzero",
        "optimizedpower",
        "cso",
        "z",
        "koide",
        "screened",
    ):
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
    dp = DampingParam.from_functional(
        "PBE", model="d3", damping="rational", dtype=torch.float64
    )
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

    pos2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 4.2]], dtype=torch.float64)
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


@pytest.mark.parametrize(
    "backend",
    ["eager", "aot_eager", pytest.param("inductor", marks=pytest.mark.slow)],
)
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
    from tad_mctc import Structure

    from tad_dftd import dftd, validate_numbers

    pos2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]], dtype=torch.float64)
    p_fermi = {"s6": 1.0, "rs6": 1.0, "alp": 20.0}
    p_bj = {"s6": 1.0, "s8": 0.8, "a1": 0.4, "a2": 4.5, "s9": 1.0}
    q2 = torch.zeros(2, dtype=torch.float64)

    def run_d1(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(
            Structure(numbers=nums, positions=pos), p_fermi, model="d1"
        ).sum()

    def run_d2(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(
            Structure(numbers=nums, positions=pos), p_fermi, model="d2"
        ).sum()

    def run_d3(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(
            Structure(numbers=nums, positions=pos), p_bj, model="d3"
        ).sum()

    def run_d3s(nums: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return dftd(
            Structure(numbers=nums, positions=pos), p_bj, model="d3s"
        ).sum()

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


def test_integer_model_id_dispatch_and_validation() -> None:
    """
    Verify H3-F.1 integer model-ID dispatch (10, 20, 30, 31, 40, 41, 42),
    agreement with string and enum selections, D5 (50) NotImplementedError,
    invalid integer ValueError, and explicit boolean rejection across
    resolve_model_id, get_model_metadata, dftd, get_properties, and
    validate_numbers.
    """
    from tad_mctc import Structure

    from tad_dftd import dftd, get_properties, validate_numbers

    mol = Structure(
        numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.43, -1.10], [0.0, -1.43, -1.10]],
            dtype=torch.float64,
        ),
    )
    p_fermi = {"s6": 0.75, "rs6": 1.1, "alp": 20.0, "s9": 1.0}
    p_bj = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407, "s9": 1.0}

    cases = [
        (10, "d1", DispersionModelId.D1, p_fermi, 11),
        (20, "d2", DispersionModelId.D2, p_fermi, 87),
        (30, "d3", DispersionModelId.D3, p_bj, 104),
        (31, "d3s", DispersionModelId.D3S, p_bj, 95),
        (40, "d4", DispersionModelId.D4, p_bj, 104),
        (41, "d4s", DispersionModelId.D4S, p_bj, 104),
        (42, "d4srev", DispersionModelId.D4SREV, p_bj, 104),
    ]

    for int_id, str_id, enum_id, params, bad_z in cases:
        assert int(enum_id) == int_id
        assert resolve_model_id(int_id) == enum_id
        assert resolve_model_id(str_id) == enum_id
        assert resolve_model_id(enum_id) == enum_id

        meta_int = get_model_metadata(int_id)
        meta_str = get_model_metadata(str_id)
        meta_enum = get_model_metadata(enum_id)
        assert meta_int == meta_str == meta_enum

        e_int = dftd(mol, params, model=int_id)
        e_str = dftd(mol, params, model=str_id)
        e_enum = dftd(mol, params, model=enum_id)
        torch.testing.assert_close(e_int, e_str, atol=0.0, rtol=0.0)
        torch.testing.assert_close(e_int, e_enum, atol=0.0, rtol=0.0)

        props_int = get_properties(mol, model=int_id)
        props_str = get_properties(mol, model=str_id)
        props_enum = get_properties(mol, model=enum_id)
        torch.testing.assert_close(
            props_int.c6, props_str.c6, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            props_int.c6, props_enum.c6, atol=0.0, rtol=0.0
        )

        v_int = validate_numbers(mol.numbers, model=int_id)
        v_str = validate_numbers(mol.numbers, model=str_id)
        v_enum = validate_numbers(mol.numbers, model=enum_id)
        assert torch.equal(v_int, v_str)
        assert torch.equal(v_int, v_enum)

        bad_nums = torch.tensor([bad_z, 1], dtype=torch.long)
        with pytest.raises(ValueError):
            validate_numbers(bad_nums, model=int_id)

    # Excluded D5 identifier (50) must continue raising NotImplementedError
    for d5_id in (50, DispersionModelId.D5, "d5", "dft-d5"):
        with pytest.raises(NotImplementedError, match="D5"):
            resolve_model_id(d5_id)
        with pytest.raises(NotImplementedError, match="D5"):
            get_model_metadata(d5_id)
        with pytest.raises(NotImplementedError, match="D5"):
            dftd(mol, p_bj, model=d5_id)
        with pytest.raises(NotImplementedError, match="D5"):
            get_properties(mol, model=d5_id)
        with pytest.raises(NotImplementedError, match="D5"):
            validate_numbers(mol.numbers, model=d5_id)

    # Unknown integer IDs raise clear ValueError
    for invalid_int in (-10, -1, 0, 1, 9, 11, 29, 32, 39, 43, 49, 51, 99):
        with pytest.raises(ValueError, match="Unknown dispersion model ID"):
            resolve_model_id(invalid_int)
        with pytest.raises(ValueError, match="Unknown dispersion model ID"):
            get_model_metadata(invalid_int)
        with pytest.raises(ValueError, match="Unknown dispersion model ID"):
            dftd(mol, p_bj, model=invalid_int)
        with pytest.raises(ValueError, match="Unknown dispersion model ID"):
            get_properties(mol, model=invalid_int)
        with pytest.raises(ValueError, match="Unknown dispersion model ID"):
            validate_numbers(mol.numbers, model=invalid_int)

    # Booleans are explicitly rejected (both as TypeError and ValueError)
    for bad_bool in (True, False):
        with pytest.raises(TypeError, match="Boolean value"):
            resolve_model_id(bad_bool)
        with pytest.raises(ValueError, match="Boolean value"):
            resolve_model_id(bad_bool)
        with pytest.raises(ValueError, match="Boolean value"):
            get_model_metadata(bad_bool)
        with pytest.raises(ValueError, match="Boolean value"):
            dftd(mol, p_bj, model=bad_bool)
        with pytest.raises(ValueError, match="Boolean value"):
            get_properties(mol, model=bad_bool)
        with pytest.raises(ValueError, match="Boolean value"):
            validate_numbers(mol.numbers, model=bad_bool)
        with pytest.raises(ValueError, match="Boolean value"):
            resolve_charge_model_id(bad_bool, model=40)


def test_a1h_dispatch_and_parameter_resolution_equivalence() -> None:
    """
    Verify A1-H.1 and A1-H.2:
    1. _DFTD_DISPATCH and _MODEL_CLASSES equivalence across all 7 models and
       forwarded arguments (structure, param, damping, pairs, cutoff,
       max_triples, checkpoint, model-specific kwargs, external charges,
       and both string/int IDs and instantiated model objects in get_properties).
    2. Authoritative parameter and damping alias resolution across all
       representative functionals and damping variants, including BJM vs BJ.
    """
    from tad_mctc import Structure

    from tad_dftd import (
        Cutoff,
        D1Model,
        D2Model,
        D3Model,
        D3SModel,
        D4Model,
        D4SModel,
        D4SrevModel,
        DensePairs,
        damping_from_name,
        dftd,
        dftd1,
        dftd2,
        dftd3,
        dftd3s,
        dftd4,
        dftd4s,
        dftd4srev,
        get_properties,
    )

    mol = Structure(
        numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.43, -1.10], [0.0, -1.43, -1.10]],
            dtype=torch.float64,
        ),
    )
    cut = Cutoff(cn=25.0, disp2=45.0, disp3=20.0, cn_eeq=22.0)
    pairs = DensePairs.build(mol, cut)
    q_ext = torch.tensor([-0.6, 0.3, 0.3], dtype=torch.float64)

    p_d1 = param.get_params(
        "BLYP", model="d1", damping="fermi", dtype=torch.float64
    )
    p_d2 = param.get_params(
        "PBE", model="d2", damping="fermi", dtype=torch.float64
    )
    p_d3_bj = param.get_params(
        "PBE", model="d3", damping="bj", dtype=torch.float64
    )
    p_d3_zero = param.get_params(
        "PBE", model="d3", damping="zero", dtype=torch.float64
    )
    p_d3_bjm = param.get_params(
        "BLYP", model="d3", damping="bjm", dtype=torch.float64
    )
    p_d3_blyp_bj = param.get_params(
        "BLYP", model="d3", damping="bj", dtype=torch.float64
    )
    p_d3_mzero = param.get_params(
        "BLYP", model="d3", damping="mzero", dtype=torch.float64
    )
    p_d3_op = param.get_params(
        "BLYP", model="d3", damping="op", dtype=torch.float64
    )
    p_d3s = param.get_params(
        "BLYP", model="d3s", damping="rational", dtype=torch.float64
    )
    p_d4 = param.get_params(
        "PBE", model="d4", damping="bj", dtype=torch.float64
    )
    p_d4s = param.get_params(
        "PBE", model="d4s", damping="bj", dtype=torch.float64
    )
    p_d4srev = {"s6": 1.0, "s8": 0.9, "a1": 0.4, "a2": 4.5, "s9": 1.0}

    # Check BJM vs BJ distinct parameters on BLYP
    assert not torch.allclose(p_d3_bjm["s8"], p_d3_blyp_bj["s8"])
    assert pytest.approx(p_d3_bjm["s8"].item(), rel=1e-14) == 1.875007
    assert pytest.approx(p_d3_bjm["a1"].item(), rel=1e-14) == 0.448486
    assert pytest.approx(p_d3_bjm["a2"].item(), rel=1e-14) == 3.610679

    # Check all damping aliases return identical parameter dicts
    for alias in ("fermi", "d1"):
        p_al = param.get_params(
            "BLYP", model="d1", damping=alias, dtype=torch.float64
        )
        for k in p_d1:
            torch.testing.assert_close(p_al[k], p_d1[k], atol=0.0, rtol=0.0)
    for alias in ("fermi", "d2"):
        p_al = param.get_params(
            "PBE", model="d2", damping=alias, dtype=torch.float64
        )
        for k in p_d2:
            torch.testing.assert_close(p_al[k], p_d2[k], atol=0.0, rtol=0.0)
    for alias in ("bj", "rational"):
        p_al = param.get_params(
            "PBE", model="d3", damping=alias, dtype=torch.float64
        )
        for k in p_d3_bj:
            torch.testing.assert_close(p_al[k], p_d3_bj[k], atol=0.0, rtol=0.0)
    for alias in ("zero", "chg"):
        p_al = param.get_params(
            "PBE", model="d3", damping=alias, dtype=torch.float64
        )
        for k in p_d3_zero:
            torch.testing.assert_close(
                p_al[k], p_d3_zero[k], atol=0.0, rtol=0.0
            )
    for alias in ("mzero", "zerom", "m-zero"):
        p_al = param.get_params(
            "BLYP", model="d3", damping=alias, dtype=torch.float64
        )
        for k in p_d3_mzero:
            torch.testing.assert_close(
                p_al[k], p_d3_mzero[k], atol=0.0, rtol=0.0
            )
    for alias in (
        "op",
        "optimizedpower",
        "optimisedpower",
        "optpower",
        "opt-power",
        "optimized-power",
        "optimised-power",
    ):
        p_al = param.get_params(
            "BLYP", model="d3", damping=alias, dtype=torch.float64
        )
        for k in p_d3_op:
            torch.testing.assert_close(p_al[k], p_d3_op[k], atol=0.0, rtol=0.0)

    # Check _DFTD_DISPATCH equivalence with direct dftd1..dftd4srev functions
    damp_fermi = damping_from_name("fermi", False, model="d1")
    damp_bj_d3 = damping_from_name("rational", True, model="d3")
    damp_bj_d4 = damping_from_name("rational", True, model="d4")

    torch.testing.assert_close(
        dftd(
            mol,
            p_d1,
            model="d1",
            damping=damp_fermi,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
        ),
        dftd1(
            mol,
            p_d1,
            damping=damp_fermi,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
        ),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dftd(
            mol,
            p_d2,
            model="d2",
            damping=damp_fermi,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
        ),
        dftd2(
            mol,
            p_d2,
            damping=damp_fermi,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
        ),
        atol=0.0,
        rtol=0.0,
    )
    from tad_dftd import ncoord

    torch.testing.assert_close(
        dftd(
            mol,
            p_d3_bj,
            model="d3",
            damping=damp_bj_d3,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
            counting_function=ncoord.exp_count,
        ),
        dftd3(
            mol,
            p_d3_bj,
            damping=damp_bj_d3,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
            counting_function=ncoord.exp_count,
        ),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dftd(
            mol,
            p_d3s,
            model="d3s",
            damping=damp_bj_d3,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
        ),
        dftd3s(
            mol,
            p_d3s,
            damping=damp_bj_d3,
            pairs=pairs,
            cutoff=cut,
            max_triples=1000,
            checkpoint=True,
        ),
        atol=0.0,
        rtol=0.0,
    )
    for m_str, fn_direct, p_cur in (
        ("d4", dftd4, p_d4),
        ("d4s", dftd4s, p_d4s),
        ("d4srev", dftd4srev, p_d4srev),
    ):
        torch.testing.assert_close(
            dftd(
                mol,
                p_cur,
                model=m_str,
                damping=damp_bj_d4,
                pairs=pairs,
                cutoff=cut,
                max_triples=1000,
                checkpoint=True,
                q=q_ext,
            ),
            fn_direct(
                mol,
                p_cur,
                damping=damp_bj_d4,
                pairs=pairs,
                cutoff=cut,
                max_triples=1000,
                checkpoint=True,
                q=q_ext,
            ),
            atol=0.0,
            rtol=0.0,
        )

    # Check get_properties with string/int IDs vs instantiated model objects
    for m_str, cls in (
        ("d1", D1Model),
        ("d2", D2Model),
        ("d3", D3Model),
        ("d3s", D3SModel),
    ):
        inst = cls(cutoff=cut)
        pr_str = get_properties(mol, model=m_str, pairs=pairs, cutoff=cut)
        pr_inst = get_properties(mol, model=inst, pairs=pairs)
        torch.testing.assert_close(pr_str.cn, pr_inst.cn, atol=0.0, rtol=0.0)
        torch.testing.assert_close(pr_str.c6, pr_inst.c6, atol=0.0, rtol=0.0)
        torch.testing.assert_close(pr_str.c8, pr_inst.c8, atol=0.0, rtol=0.0)

    for m_str, cls in (
        ("d4", D4Model),
        ("d4s", D4SModel),
        ("d4srev", D4SrevModel),
    ):
        inst = cls(cutoff=cut)
        pr_str = get_properties(
            mol, model=m_str, pairs=pairs, cutoff=cut, q=q_ext
        )
        pr_inst = get_properties(mol, model=inst, pairs=pairs, q=q_ext)
        torch.testing.assert_close(pr_str.cn, pr_inst.cn, atol=0.0, rtol=0.0)
        torch.testing.assert_close(pr_str.c6, pr_inst.c6, atol=0.0, rtol=0.0)
        torch.testing.assert_close(pr_str.c8, pr_inst.c8, atol=0.0, rtol=0.0)
        assert pr_str.q is not None and pr_inst.q is not None
        torch.testing.assert_close(pr_str.q, pr_inst.q, atol=0.0, rtol=0.0)
        assert pr_str.alphadd is not None and pr_inst.alphadd is not None
        torch.testing.assert_close(
            pr_str.alphadd, pr_inst.alphadd, atol=0.0, rtol=0.0
        )
        assert pr_str.alphaqq is not None and pr_inst.alphaqq is not None
        torch.testing.assert_close(
            pr_str.alphaqq, pr_inst.alphaqq, atol=0.0, rtol=0.0
        )
