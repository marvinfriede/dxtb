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
Run tests for overlap of diatomic systems.
References calculated with tblite 0.3.0.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.convert import str_to_device

from dxtb import GFN1_XTB, IndexHelper, ParamModule
from dxtb._src.basis.bas import Basis, CGTOExpansion
from dxtb._src.basis.slater import slater_to_gauss
from dxtb._src.constants.defaults import DEFAULT_BASIS_INT
from dxtb._src.typing.exceptions import (
    CGTOAzimuthalQuantumNumberError,
    CGTOPrimitivesError,
    CGTOPrincipalQuantumNumberError,
    CGTOQuantumNumberError,
    CGTOSlaterExponentsError,
    IntegralTransformError,
)

from ..conftest import DEVICE
from ..utils import libcint_setup, make_basis, overlap_1d, pytorch_setup


def test_fail_number_primitives() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOPrimitivesError):
        slater_to_gauss(torch.tensor(7), n, l, torch.tensor(1.2))


def test_fail_slater_exponent() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOSlaterExponentsError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(-1.2))


def test_fail_max_principal() -> None:
    # principal and azimuthal quantum number of 7s-orbital
    n, l = torch.tensor(7), torch.tensor(0)

    with pytest.raises(CGTOPrincipalQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital() -> None:
    # principal and azimuthal quantum number of 5h-orbital
    n, l = torch.tensor(5), torch.tensor(5)

    with pytest.raises(CGTOAzimuthalQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_quantum_number() -> None:
    # principal and azimuthal quantum number of 2f-orbital
    n, l = torch.tensor(2), torch.tensor(3)

    with pytest.raises(CGTOQuantumNumberError):
        slater_to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital_trafo():
    """No higher orbitals than d-orbitals allowed."""
    vec = torch.tensor([0.0, 0.0, 1.4])

    # arbitrary element (Rn)
    number = torch.tensor([86])

    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)
    bas = make_basis(number, GFN1_XTB, ihelp)
    expansion = bas.expand()
    alpha = expansion.alphas
    coeff = expansion.coeffs

    j = torch.tensor(5)
    for i in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_1d(
                (torch.tensor(i), j),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )
    i = torch.tensor(5)
    for j in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_1d(
                (i, torch.tensor(j)),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )


# Device and Dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_dtype(dtype: torch.dtype) -> None:
    """``dtype`` sets the Slater exponents; integer and bool fields keep theirs."""
    number = torch.tensor([8, 1, 1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)
    pm = ParamModule(GFN1_XTB, dtype=torch.float64)

    bas = Basis(number, pm, ihelp, dtype=dtype)
    assert bas.slater.dtype == dtype
    assert bas.unique.dtype == number.dtype
    assert bas.ngauss.dtype == DEFAULT_BASIS_INT
    assert not bas.pqn.is_floating_point()
    assert bas.valence.dtype == torch.bool

    if dtype in (torch.float32, torch.float64):
        expansion = bas.expand()
        assert all(a.dtype == dtype for a in expansion.alphas)
        assert all(c.dtype == dtype for c in expansion.coeffs)

    # without `dtype`, the dtype of the parametrization is used
    assert Basis(number, pm, ihelp).slater.dtype == torch.float64


def test_requires_param_module() -> None:
    """Basis is internal and takes a ParamModule only."""
    number = torch.tensor([1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)

    with pytest.raises(TypeError, match="ParamModule"):
        Basis(number, GFN1_XTB, ihelp)  # type: ignore[arg-type]


def test_rejects_inputs_on_different_devices() -> None:
    """Basis makes no device moves; inputs on different devices raise."""
    number = torch.tensor([8, 1, 1])
    ihelp = IndexHelper.from_numbers(number, GFN1_XTB)
    pm = ParamModule(GFN1_XTB)

    with pytest.raises(ValueError, match="one device"):
        Basis(number, pm, ihelp.to(device=torch.device("meta")))

    pm_meta = ParamModule(GFN1_XTB, device=torch.device("meta"))
    with pytest.raises(ValueError, match="one device"):
        Basis(number, pm_meta, ihelp)

    # no conversion methods that could move or cast fields behind the caller
    bas = Basis(number, pm, ihelp)
    assert not hasattr(bas, "to")
    assert not hasattr(bas, "type")


def test_non_valence_orthogonalization_is_metadata_independent() -> None:
    """Non-valence shell orthogonalization depends on valence flags, not meta."""
    from dxtb import ParamModule, labels
    from dxtb._src.integral.evaluation import setup_integral_evaluation
    from dxtb.integrals.wrappers import overlap

    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
    )
    custom = GFN1_XTB.model_copy(deep=True)
    custom.meta = None
    custom_pm = ParamModule(custom, dtype=torch.float64)

    bas_ref = make_basis(
        numbers,
        GFN1_XTB,
        IndexHelper.from_numbers(numbers, GFN1_XTB),
        dtype=torch.float64,
    )
    bas_custom = make_basis(
        numbers,
        custom,
        IndexHelper.from_numbers(numbers, custom),
        dtype=torch.float64,
    )
    bas_custom_pm = make_basis(
        numbers, custom_pm, IndexHelper.from_numbers(numbers, custom_pm)
    )

    exp_ref = bas_ref.expand()
    exp_cus = bas_custom.expand()
    exp_pm = bas_custom_pm.expand()
    for a_ref, a_cus, a_pm in zip(
        exp_ref.alphas, exp_cus.alphas, exp_pm.alphas
    ):
        torch.testing.assert_close(a_cus, a_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(a_pm, a_ref, atol=0.0, rtol=0.0)
    for c_ref, c_cus, c_pm in zip(
        exp_ref.coeffs, exp_cus.coeffs, exp_pm.coeffs
    ):
        torch.testing.assert_close(c_cus, c_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(c_pm, c_ref, atol=0.0, rtol=0.0)

    setup = setup_integral_evaluation(
        numbers,
        custom_pm,
        IndexHelper.from_numbers(numbers, custom_pm),
        driver_type=labels.INTDRIVER_PYTORCH,
        intlevel=labels.INTLEVEL_OVERLAP,
    )
    assert setup.backend is not None

    s_ref = overlap(
        Structure(numbers=numbers, positions=positions),
        GFN1_XTB,
        driver=labels.INTDRIVER_PYTORCH,
    )
    s_cus = overlap(
        Structure(numbers=numbers, positions=positions),
        custom,
        driver=labels.INTDRIVER_PYTORCH,
    )
    torch.testing.assert_close(s_cus, s_ref, atol=0.0, rtol=0.0)


def test_basis_and_wrappers_honor_target_dtype_with_param_module() -> None:
    """Basis and integral wrappers cast existing ParamModule CGTOs to dtype."""
    from dxtb import GFN0_XTB, ParamModule, labels
    from dxtb._src.exlibs.available import has_libcint
    from dxtb.integrals.wrappers import dipint, hcore, overlap, quadint

    numbers = torch.tensor([8, 1, 1])
    positions_f32 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float32,
    )
    pm_f64 = ParamModule(GFN1_XTB, dtype=torch.float64)
    pm_gfn0_f64 = ParamModule(GFN0_XTB, dtype=torch.float64)
    ihelp = IndexHelper.from_numbers(numbers, pm_f64)

    bas_default = Basis(numbers, pm_f64, ihelp)
    assert bas_default.slater.dtype == torch.float64

    bas_f32 = Basis(numbers, pm_f64, ihelp, dtype=torch.float32)
    exp_f32 = bas_f32.expand()
    assert bas_f32.slater.dtype == torch.float32
    assert all(a.dtype == torch.float32 for a in exp_f32.alphas)
    assert all(c.dtype == torch.float32 for c in exp_f32.coeffs)

    drivers = [labels.INTDRIVER_PYTORCH]
    if has_libcint:
        drivers.append(labels.INTDRIVER_LIBCINT)
    for drv in drivers:
        assert (
            overlap(
                Structure(numbers=numbers, positions=positions_f32),
                pm_f64,
                driver=drv,
            ).dtype
            == torch.float32
        )
        assert (
            dipint(
                Structure(numbers=numbers, positions=positions_f32),
                pm_f64,
                driver=drv,
            ).dtype
            == torch.float32
        )
        assert (
            quadint(
                Structure(numbers=numbers, positions=positions_f32),
                pm_f64,
                driver=drv,
            ).dtype
            == torch.float32
        )
        assert (
            hcore(
                Structure(numbers=numbers, positions=positions_f32),
                pm_f64,
                driver=drv,
            ).dtype
            == torch.float32
        )
        assert (
            hcore(
                Structure(numbers=numbers, positions=positions_f32),
                pm_gfn0_f64,
                driver=drv,
            ).dtype
            == torch.float32
        )
    assert pm_f64.dtype == torch.float64
    assert pm_gfn0_f64.dtype == torch.float64


@pytest.mark.parametrize("batched", [False, True])
def test_expand_handles_zero_padded_numbers(batched: bool) -> None:
    """Zero-padded atomic numbers produce 1D placeholder CGTO slots."""
    unpadded = torch.tensor([8, 1])
    bas_ref = make_basis(
        unpadded, GFN1_XTB, IndexHelper.from_numbers(unpadded, GFN1_XTB)
    )
    exp_ref = bas_ref.expand()

    padded = torch.tensor([[8, 1, 0]]) if batched else torch.tensor([8, 1, 0])
    ihelp = IndexHelper.from_numbers(padded, GFN1_XTB, batch_mode=int(batched))
    bas_pad = make_basis(padded, GFN1_XTB, ihelp)
    exp_pad = bas_pad.expand()

    assert len(exp_pad.alphas) == len(exp_ref.alphas) + 1
    assert exp_pad.alphas[0].shape == (1,)
    assert exp_pad.coeffs[0].shape == (1,)
    assert torch.count_nonzero(exp_pad.alphas[0]) == 0
    assert torch.count_nonzero(exp_pad.coeffs[0]) == 0

    for a_ref, a_act in zip(exp_ref.alphas, exp_pad.alphas[1:]):
        torch.testing.assert_close(a_act, a_ref, atol=0.0, rtol=0.0)
    for c_ref, c_act in zip(exp_ref.coeffs, exp_pad.coeffs[1:]):
        torch.testing.assert_close(c_act, c_ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_canonical_cgtos_all_supported_elements(method: str) -> None:
    """WP2: canonical CGTO setup across all 86 elements in GFN0, GFN1, GFN2."""
    from tad_mctc.data import pse

    from dxtb import GFN0_XTB, GFN2_XTB, ParamModule
    from dxtb._src.basis.ortho import gaussian_integral

    par_raw = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}[method]
    assert par_raw.element is not None
    zs = sorted(
        pse.S2Z[sym] for sym in par_raw.element.keys() if sym in pse.S2Z
    )
    assert zs == list(range(1, 87))

    expected_dev = torch.device("cpu") if DEVICE is None else DEVICE
    par = ParamModule(par_raw, dtype=torch.float64, device=DEVICE)
    tm_dsp_seen = 0

    for z in zs:
        num = torch.tensor([z], device=DEVICE)
        ihelp = IndexHelper.from_numbers(num, par)
        bas = make_basis(num, par, ihelp, device=DEVICE, dtype=torch.float64)
        expansion = bas.expand()
        alphas, coeffs = expansion.alphas, expansion.coeffs

        n_shells = int(ihelp.ushells_per_unique[0].item())
        assert len(alphas) == n_shells == len(coeffs)
        angs = [int(x) for x in ihelp.unique_angular.tolist()]
        vals = [int(x) for x in bas.valence.tolist()]
        if angs[:1] == [2] and 0 in angs[1:]:
            tm_dsp_seen += 1

        for l_val in set(angs):
            nv = sum(1 for a_l, v in zip(angs, vals) if a_l == l_val and v == 0)
            assert nv <= 1

        for s_idx, (alpha, coeff) in enumerate(zip(alphas, coeffs)):
            assert alpha.ndim == 1 and coeff.ndim == 1
            assert alpha.shape == coeff.shape
            assert alpha.dtype == torch.float64 and coeff.dtype == torch.float64
            assert alpha.device == expected_dev and coeff.device == expected_dev
            assert torch.isfinite(alpha).all() and torch.isfinite(coeff).all()
            if angs[s_idx] == 0:
                self_ovlp = gaussian_integral(alpha, alpha, coeff, coeff)
                torch.testing.assert_close(
                    self_ovlp,
                    torch.ones_like(self_ovlp),
                    atol=1e-13,
                    rtol=1e-13,
                )

        if z == 1 and method in ("gfn0", "gfn1"):
            assert vals == [1, 0] and angs == [0, 0]
            cross = gaussian_integral(
                alphas[0], alphas[1], coeffs[0], coeffs[1]
            )
            torch.testing.assert_close(
                cross, torch.zeros_like(cross), atol=1e-14, rtol=0.0
            )

    assert tm_dsp_seen == 41


def test_custom_multi_nonvalence_s_shells_orthogonalize_to_primary() -> None:
    """WP2: multiple non-valence s shells each orthogonalize to valence 1s."""
    import copy

    from dxtb import ParamModule
    from dxtb._src.basis.ortho import gaussian_integral

    custom = copy.deepcopy(GFN1_XTB)
    assert custom.element is not None
    h_el = custom.element["H"]
    h_el.shells = ["1s", "2s", "3s"]
    h_el.ngauss = [4, 4, 4]
    h_el.slater = [1.2, 0.9, 0.6]
    h_el.levels = [-10.0, -5.0, -2.0]
    h_el.shpoly = [0.0, 0.0, 0.0]
    h_el.kcn = [0.0, 0.0, 0.0]
    h_el.lgam = [1.0, 1.0, 1.0]

    par = ParamModule(custom, dtype=torch.float64, device=DEVICE)
    num = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers(num, par)
    bas = make_basis(num, par, ihelp, device=DEVICE, dtype=torch.float64)
    expansion = bas.expand()
    alphas, coeffs = expansion.alphas, expansion.coeffs

    assert [int(x) for x in bas.valence.tolist()] == [1, 0, 0]
    for idx in range(3):
        diag = gaussian_integral(
            alphas[idx], alphas[idx], coeffs[idx], coeffs[idx]
        )
        torch.testing.assert_close(
            diag, torch.ones_like(diag), atol=1e-14, rtol=1e-14
        )
    s12 = gaussian_integral(alphas[0], alphas[1], coeffs[0], coeffs[1])
    s13 = gaussian_integral(alphas[0], alphas[2], coeffs[0], coeffs[2])
    s23 = gaussian_integral(alphas[1], alphas[2], coeffs[1], coeffs[2])
    torch.testing.assert_close(s12, torch.zeros_like(s12), atol=1e-14, rtol=0.0)
    torch.testing.assert_close(s13, torch.zeros_like(s13), atol=1e-14, rtol=0.0)
    assert s23.abs() > 0.1


def _scale_tensor_leaf(leaf: torch.Tensor) -> torch.Tensor:
    """Scale a PyTree tensor leaf without breaking autograd."""
    return leaf * 1.25


def test_cgto_expansion_container_pytree_and_autograd() -> None:
    """CGTOExpansion is immutable, PyTree-compatible, differentiable, and uncached."""
    from torch.utils import _pytree as pytree

    from dxtb import ParamModule
    from dxtb._src.basis import CGTOExpansion as ReExportedCGTOExpansion

    assert ReExportedCGTOExpansion is CGTOExpansion
    assert not hasattr(Basis, "create_cgtos")

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, dtype=torch.float64, device=DEVICE)
    named = dict(par.named_parameters())
    h_slater = named["parameter_tree.element.H.slater.param"]
    o_slater = named["parameter_tree.element.O.slater.param"]
    h_slater.requires_grad_(True)
    o_slater.requires_grad_(True)

    ihelp = IndexHelper.from_numbers(numbers, par)
    bas = make_basis(numbers, par, ihelp, device=DEVICE, dtype=torch.float64)
    assert not hasattr(bas, "create_cgtos")
    assert not hasattr(bas, "__dict__")

    expansion = bas.expand()
    assert isinstance(expansion, CGTOExpansion)
    assert expansion._fields == ("alphas", "coeffs")
    assert len(expansion.alphas) == len(expansion.coeffs) == 4

    with pytest.raises(AttributeError):
        expansion.alphas = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        expansion.extra = 1  # type: ignore[attr-defined]

    leaves, spec = pytree.tree_flatten(expansion)
    assert len(leaves) == 8
    for expected, actual in zip((*expansion.alphas, *expansion.coeffs), leaves):
        assert actual is expected
        assert actual.requires_grad
        assert actual.grad_fn is not None

    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert type(rebuilt) is CGTOExpansion
    for a_orig, a_reb in zip(expansion.alphas, rebuilt.alphas):
        torch.testing.assert_close(a_reb, a_orig, atol=0.0, rtol=0.0)
    for c_orig, c_reb in zip(expansion.coeffs, rebuilt.coeffs):
        torch.testing.assert_close(c_reb, c_orig, atol=0.0, rtol=0.0)

    mapped = pytree.tree_map(_scale_tensor_leaf, expansion)
    assert type(mapped) is CGTOExpansion
    loss = sum(
        (a.square().sum() for a in mapped.alphas),
        torch.zeros((), dtype=torch.float64, device=DEVICE),
    ) + sum(
        (c.square().sum() for c in mapped.coeffs),
        torch.zeros((), dtype=torch.float64, device=DEVICE),
    )
    grad_h, grad_o = torch.autograd.grad(loss, (h_slater, o_slater))
    assert torch.isfinite(grad_h).all() and grad_h.abs().sum() > 0
    assert torch.isfinite(grad_o).all() and grad_o.abs().sum() > 0

    # No cached expansion or consumed graph lives on Basis across fresh calls
    expansion2 = bas.expand()
    assert expansion2 is not expansion
    mapped2 = pytree.tree_map(_scale_tensor_leaf, expansion2)
    loss2 = sum(
        (a.square().sum() for a in mapped2.alphas),
        torch.zeros((), dtype=torch.float64, device=DEVICE),
    ) + sum(
        (c.square().sum() for c in mapped2.coeffs),
        torch.zeros((), dtype=torch.float64, device=DEVICE),
    )
    grad_h2, grad_o2 = torch.autograd.grad(loss2, (h_slater, o_slater))
    torch.testing.assert_close(grad_h2, grad_h, atol=0.0, rtol=0.0)
    torch.testing.assert_close(grad_o2, grad_o, atol=0.0, rtol=0.0)


def test_basis_and_setups_keep_parametrization_in_place() -> None:
    """
    Basis reads the ParamModule without moving or copying it, and gradients
    reach the Slater exponents through both integral setups.
    """
    import sys

    import dxtb._src.calculators.types.base as base_calc_mod
    from dxtb import Calculator, labels
    from dxtb._src.components.classicals.dispersion import new_dispersion
    from dxtb._src.exlibs.available import has_libcint
    from dxtb._src.integral.driver.pytorch import (
        build_overlap as build_overlap_pytorch,
    )
    from dxtb._src.integral.evaluation import setup_integral_evaluation
    from dxtb._src.typing.exceptions import DeviceError
    from dxtb._src.xtb.h0 import setup_h0, setup_h0_core

    raw_numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
    )
    ihelp = IndexHelper.from_numbers(raw_numbers, GFN1_XTB)

    # 1. Basis keeps the ParamModule's storage and the IndexHelper identity
    pm = ParamModule(GFN1_XTB, dtype=torch.float64, device=torch.device("cpu"))
    pm.set_differentiable("element.O.slater")
    pm.set_differentiable("element.H.slater")
    o_slater = pm.get("element.O.slater")
    h_slater = pm.get("element.H.slater")
    o_ptr_before = o_slater.data_ptr()
    h_ptr_before = h_slater.data_ptr()

    bas = Basis(raw_numbers, pm, ihelp)
    assert bas.ihelp is ihelp
    assert torch.equal(bas.unique, torch.tensor([1, 8]))
    assert bas.slater.dtype == torch.float64
    assert o_slater.data_ptr() == o_ptr_before
    assert h_slater.data_ptr() == h_ptr_before

    # 2. setup_integral_evaluation needs the numbers on the requested device
    with pytest.raises(ValueError, match="requested device"):
        setup_integral_evaluation(
            raw_numbers,
            pm,
            ihelp,
            driver_type=labels.INTDRIVER_PYTORCH,
            intlevel=labels.INTLEVEL_OVERLAP,
            device=torch.device("meta"),
        )

    # 3. Dispersion & Calculator cross-dtype/device boundary invariants
    with torch.device("meta"):
        disp = new_dispersion(raw_numbers, pm)
    assert disp is not None and disp.device == torch.device("cpu")
    with pytest.raises(DeviceError):
        new_dispersion(raw_numbers, pm, device="wrong")  # type: ignore[arg-type]

    calc_f32 = Calculator(
        raw_numbers,
        pm,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
        device="cpu",  # type: ignore[arg-type]
        dtype=torch.float32,
    )
    orig_pm_dd = ParamModule.dd.fget  # type: ignore[union-attr]
    assert orig_pm_dd is not None
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            ParamModule,
            "dd",
            property(
                lambda _self: (
                    {"device": torch.device("meta"), "dtype": _self.dtype}
                    if sys._getframe(1).f_code.co_filename
                    == base_calc_mod.__file__
                    else orig_pm_dd(_self)
                )
            ),
        )
        eval_sys = calc_f32._prepare_eval_system()
    assert eval_sys.h0_setup.selfenergy.dtype == torch.float32
    assert eval_sys.h0_setup.selfenergy.device == torch.device("cpu")

    # 4. H0 setups, PyTorch and libcint setups, and gradients
    h0 = setup_h0(raw_numbers, pm, ihelp)
    h0_core = setup_h0_core(raw_numbers, pm, ihelp)
    assert h0.selfenergy.requires_grad is False
    assert h0_core.hscale.requires_grad is False

    s_pt = build_overlap_pytorch(pytorch_setup(bas), positions)
    grad_pt_o, grad_pt_h = torch.autograd.grad(
        s_pt.pow(2).sum(), (o_slater, h_slater)
    )
    assert torch.isfinite(grad_pt_o).all() and grad_pt_o.abs().sum() > 0
    assert torch.isfinite(grad_pt_h).all() and grad_pt_h.abs().sum() > 0

    if has_libcint:
        from dxtb._src.integral.driver.libcint import (
            build_overlap as build_overlap_libcint,
        )

        lc_setup = libcint_setup(raw_numbers, Basis(raw_numbers, pm, ihelp))
        s_lc = build_overlap_libcint(lc_setup, positions)
        torch.testing.assert_close(s_lc, s_pt, atol=1e-12, rtol=1e-12)
        grad_lc_o, grad_lc_h = torch.autograd.grad(
            s_lc.pow(2).sum(), (o_slater, h_slater)
        )
        torch.testing.assert_close(grad_lc_o, grad_pt_o, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(grad_lc_h, grad_pt_h, atol=1e-10, rtol=1e-10)


@pytest.mark.cuda
def test_cuda_system_libcint_moves_only_the_gaussians() -> None:
    """
    A CUDA System with libcint builds the Basis on CUDA; only the libcint
    setup moves to the CPU, and energies and gradients match PyTorch.
    """
    pytest.importorskip("tad_libcint")
    from dxtb import labels
    from dxtb._src.calculators.model import Model
    from dxtb.config import Config

    cuda_dev = str_to_device("cuda")
    numbers = torch.tensor([8, 1, 1], device=cuda_dev)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.8, 0.0]],
        dtype=torch.float64,
        device=cuda_dev,
        requires_grad=True,
    )

    pm_probe = ParamModule(GFN1_XTB, dtype=torch.float64, device=cuda_dev)
    ihelp_cuda = IndexHelper.from_numbers(numbers, pm_probe)
    bas = Basis(numbers, pm_probe, ihelp_cuda)
    assert bas.unique.device == cuda_dev
    assert all(a.device == cuda_dev for a in bas.expand().alphas)
    lc_setup = libcint_setup(numbers, bas)
    assert all(a.device.type == "cpu" for a in lc_setup.alphas)

    pm_cuda = ParamModule(GFN1_XTB, dtype=torch.float64, device=cuda_dev)
    pm_cuda.set_differentiable("element.O.slater")
    pm_cuda.set_differentiable("element.H.slater")
    model_lc = Model(
        par=pm_cuda, config=Config.create(int_driver=labels.INTDRIVER_LIBCINT)
    )
    sys_lc = model_lc.setup(numbers)

    pm_pt = ParamModule(GFN1_XTB, dtype=torch.float64, device=cuda_dev)
    pm_pt.set_differentiable("element.O.slater")
    pm_pt.set_differentiable("element.H.slater")
    model_pt = Model(
        par=pm_pt, config=Config.create(int_driver=labels.INTDRIVER_PYTORCH)
    )
    sys_pt = model_pt.setup(numbers)
    positions_pt = positions.detach().clone().requires_grad_(True)
    e_lc = sys_lc.energy(positions)
    e_pt = sys_pt.energy(positions_pt)
    torch.testing.assert_close(e_lc, e_pt, atol=1e-10, rtol=1e-10)
    (g_pos_lc,) = torch.autograd.grad(e_lc, positions, retain_graph=True)
    (g_pos_pt,) = torch.autograd.grad(e_pt, positions_pt, retain_graph=True)
    torch.testing.assert_close(g_pos_lc, g_pos_pt, atol=1e-9, rtol=1e-9)
    g_par_lc = torch.autograd.grad(
        e_lc, (pm_cuda.get("element.O.slater"), pm_cuda.get("element.H.slater"))
    )
    g_par_pt = torch.autograd.grad(
        e_pt, (pm_pt.get("element.O.slater"), pm_pt.get("element.H.slater"))
    )
    torch.testing.assert_close(g_par_lc[0], g_par_pt[0], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(g_par_lc[1], g_par_pt[1], atol=1e-9, rtol=1e-9)
