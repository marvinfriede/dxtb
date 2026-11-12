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
Gradient of the total energy with respect to the basis exponents against
finite differences. For GFN2, the gradient also flows through the multipole
integrals.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import Calculator
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Literal, Tensor

from ..conftest import DEVICE
from ..utils import get_param_module
from .samples import samples


def energy_of_scaled_exponents(
    gfn: Literal["gfn1", "gfn2"],
    numbers: Tensor,
    positions: Tensor,
    scale: Tensor,
    driver: str,
    dd: DD,
) -> Tensor:
    """Single point with every primitive exponent multiplied by ``scale``,
    standing in for a fitted basis parameter."""
    original = Basis.create_cgtos

    def patched(self):
        alphas, coeffs = original(self)
        return tuple(a * scale for a in alphas), coeffs

    Basis.create_cgtos = patched  # type: ignore[method-assign]
    try:
        opts = {
            "verbosity": 0,
            "int_driver": driver,
            "differentiation": "unrolled",
            "x_atol": 1e-11,
        }
        par = get_param_module(gfn, **dd)
        calc = Calculator(numbers, par, opts=opts, **dd)
        return calc.get_energy(positions)
    finally:
        Basis.create_cgtos = original  # type: ignore[method-assign]


@pytest.mark.parametrize("driver", ["pytorch", "libcint"])
@pytest.mark.parametrize(
    "gfn, name", [("gfn1", "LiH"), ("gfn1", "H2O"), ("gfn2", "LiH")]
)
def test_exponent_gradient_matches_fd(
    driver: str, gfn: Literal["gfn1", "gfn2"], name: str
) -> None:
    """Analytical SCF exponent gradients match finite differences on drivers."""
    if driver == "libcint" and not has_libcint:
        pytest.skip("libcint not available")
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    scale = torch.tensor(1.0, **dd, requires_grad=True)
    energy = energy_of_scaled_exponents(
        gfn, numbers, positions, scale, driver, dd
    )
    (grad,) = torch.autograd.grad(energy, scale)

    step = 1e-4
    plus = energy_of_scaled_exponents(
        gfn, numbers, positions, torch.tensor(1.0 + step, **dd), driver, dd
    )
    minus = energy_of_scaled_exponents(
        gfn, numbers, positions, torch.tensor(1.0 - step, **dd), driver, dd
    )
    grad_fd = (plus - minus) / (2 * step)

    torch.testing.assert_close(grad, grad_fd, atol=1e-6, rtol=1e-5)


def _integral_objective(matrices) -> Tensor:
    """Scalar objective combining overlap, hcore, and any multipole tensors."""
    val = (matrices.overlap * 1.3 + matrices.hcore * 0.7).square().sum()
    if matrices.dipole is not None:
        val = val + (matrices.dipole * 0.5).square().sum()
    if matrices.quadrupole is not None:
        val = val + (matrices.quadrupole * 0.3).square().sum()
    return val


@pytest.mark.parametrize("driver", ["pytorch", "libcint"])
def test_reused_setup_independent_position_gradients(driver: str) -> None:
    """Case A: one reused setup supports independent position backward passes."""
    if driver == "libcint" and not has_libcint:
        pytest.skip("libcint not available")

    from dxtb import GFN2_XTB, IndexHelper, ParamModule, labels
    from dxtb._src.integral.evaluation import (
        build_integral_matrices,
        setup_integral_evaluation,
    )
    from dxtb._src.xtb.h0 import setup_h0_core

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    pos1 = samples["H2O"]["positions"].to(**dd)
    pos2 = pos1 + torch.tensor(
        [[0.0, 0.0, 0.04], [0.06, -0.03, 0.08], [-0.05, 0.04, -0.06]], **dd
    )
    driver_id = (
        labels.INTDRIVER_LIBCINT
        if driver == "libcint"
        else labels.INTDRIVER_PYTORCH
    )

    par = ParamModule(GFN2_XTB, **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    isetup = setup_integral_evaluation(
        numbers,
        par,
        ihelp,
        driver_type=driver_id,
        intlevel=labels.INTLEVEL_QUADRUPOLE,
        **dd,
    )
    assert isetup is not None
    h0 = setup_h0_core(numbers, par, ihelp, **dd)

    p1 = pos1.clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(
        _integral_objective(build_integral_matrices(isetup, h0, p1, 0.0)[0]), p1
    )
    p2 = pos2.clone().requires_grad_(True)
    (g2,) = torch.autograd.grad(
        _integral_objective(build_integral_matrices(isetup, h0, p2, 0.0)[0]), p2
    )
    assert torch.isfinite(g1).all() and torch.isfinite(g2).all()

    eps = 1e-6
    fd2 = torch.zeros_like(pos2)
    for ia in range(pos2.shape[0]):
        for ic in range(3):
            dp = torch.zeros_like(pos2)
            dp[ia, ic] = eps
            vp = _integral_objective(
                build_integral_matrices(isetup, h0, pos2 + dp, 0.0)[0]
            )
            vm = _integral_objective(
                build_integral_matrices(isetup, h0, pos2 - dp, 0.0)[0]
            )
            fd2[ia, ic] = (vp - vm) / (2 * eps)
    torch.testing.assert_close(g2, fd2, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("driver", ["pytorch", "libcint"])
def test_reused_setup_second_parameter_backward_contract(driver: str) -> None:
    """Case B: persistent setup consumes setup-time parameter graph on 1st backward."""
    if driver == "libcint" and not has_libcint:
        pytest.skip("libcint not available")

    from dxtb import GFN1_XTB, IndexHelper, ParamModule, labels
    from dxtb._src.integral.evaluation import (
        build_integral_matrices,
        setup_integral_evaluation,
    )
    from dxtb._src.xtb.h0 import setup_h0_core

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    pos1 = samples["H2O"]["positions"].to(**dd)
    pos2 = pos1 * 1.03
    pkey = "parameter_tree.element.H.slater.param"
    driver_id = (
        labels.INTDRIVER_LIBCINT
        if driver == "libcint"
        else labels.INTDRIVER_PYTORCH
    )

    par = ParamModule(GFN1_XTB, **dd)
    h_slater = dict(par.named_parameters())[pkey]
    h_slater.requires_grad_(True)
    ihelp = IndexHelper.from_numbers(numbers, par)
    isetup = setup_integral_evaluation(
        numbers,
        par,
        ihelp,
        driver_type=driver_id,
        intlevel=labels.INTLEVEL_HCORE,
        **dd,
    )
    assert isetup is not None
    h0 = setup_h0_core(numbers, par, ihelp, **dd)

    (g_first,) = torch.autograd.grad(
        _integral_objective(build_integral_matrices(isetup, h0, pos1, 0.0)[0]),
        h_slater,
    )
    assert torch.isfinite(g_first).all() and g_first.abs().sum() > 0

    with pytest.raises(
        RuntimeError, match="backward through the graph a second time"
    ):
        _ = torch.autograd.grad(
            _integral_objective(
                build_integral_matrices(isetup, h0, pos2, 0.0)[0]
            ),
            h_slater,
        )


@pytest.mark.parametrize("driver", ["pytorch", "libcint"])
@pytest.mark.parametrize("gfn", ["gfn1", "gfn2"])
def test_functional_setup_repeated_slater_gradients_and_transforms(
    driver: str, gfn: Literal["gfn1", "gfn2"]
) -> None:
    """Cases C & D: fresh functional setup supports repeated Slater grads and transforms."""
    if driver == "libcint" and not has_libcint:
        pytest.skip("libcint not available")

    from dataclasses import replace
    from functools import partial

    from torch.func import jacfwd, jacrev, jvp

    from dxtb._src.calculators.config.main import Config
    from dxtb._src.calculators.model import Model, ModelParameterNode
    from dxtb._src.calculators.properties.functional import (
        system_energy,
        system_parameter_force_jacobian,
    )

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["LiH"]["numbers"].to(DEVICE)
    pos1 = samples["LiH"]["positions"].to(**dd)
    pos2 = pos1 * 1.04

    par = get_param_module(gfn, **dd)
    cfg = Config.create(
        method=gfn,
        int_driver=driver,
        exclude=("disp", "hal", "rep"),
        maxiter=20,
        x_atol=1e-11,
    )
    model = Model(par=par, config=cfg)
    base_tree = model.parameter_tree
    skey = next(
        k
        for k in base_tree.parameters.params
        if k.endswith("element.H.slater.param") or k == "element.H.slater"
    )
    base_slater = base_tree.parameters.params[skey].detach().clone()

    def energy_for_slater_and_geom(slater_val: Tensor, geom: Tensor) -> Tensor:
        """Total SCF energy from a fresh functional System setup."""
        updated = dict(base_tree.parameters.params)
        updated[skey] = slater_val
        tree = ModelParameterNode(
            parameters=replace(base_tree.parameters, params=updated),
            config=base_tree.config,
            auto_int_level=base_tree.auto_int_level,
        )
        sys_obj = model.setup_functional(numbers, tree)
        return system_energy(sys_obj, geom)

    energy_of_slater = partial(energy_for_slater_and_geom, geom=pos1)

    leaf1 = base_slater.clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(energy_for_slater_and_geom(leaf1, pos1), leaf1)
    leaf2 = base_slater.clone().requires_grad_(True)
    (g2,) = torch.autograd.grad(energy_for_slater_and_geom(leaf2, pos2), leaf2)
    leaf_pert = (base_slater + 0.03).clone().requires_grad_(True)
    (g_pert,) = torch.autograd.grad(
        energy_for_slater_and_geom(leaf_pert, pos1), leaf_pert
    )
    leaf3 = base_slater.clone().requires_grad_(True)
    (g3,) = torch.autograd.grad(energy_for_slater_and_geom(leaf3, pos1), leaf3)

    torch.testing.assert_close(g1, g3, atol=0.0, rtol=0.0)
    assert (g2 - g1).abs().max() > 1e-4
    assert (g_pert - g1).abs().max() > 1e-4

    eps = 1e-6
    fd_grad = torch.zeros_like(base_slater)
    for idx in range(base_slater.numel()):
        ds = torch.zeros_like(base_slater)
        ds[idx] = eps
        ep = energy_of_slater(base_slater + ds)
        em = energy_of_slater(base_slater - ds)
        fd_grad[idx] = (ep - em) / (2 * eps)
    torch.testing.assert_close(g1, fd_grad, atol=1e-7, rtol=1e-6)

    jr = jacrev(energy_of_slater)(base_slater)
    jf = jacfwd(energy_of_slater)(base_slater)
    _, jv = jvp(
        energy_of_slater, (base_slater,), (torch.ones_like(base_slater),)
    )
    torch.testing.assert_close(jr, g1, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(jf, g1, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(jv, g1.sum(), atol=1e-10, rtol=1e-10)

    # Mixed 2nd-order geometry/parameter Jacobian dF(R, theta)/d(theta) vs FD
    mixed_jac = system_parameter_force_jacobian(model, numbers, pos1, skey)
    fd_forces = torch.zeros(pos1.shape + base_slater.shape, **dd)
    for idx in range(base_slater.numel()):
        ds = torch.zeros_like(base_slater)
        ds[idx] = eps
        p_plus = pos1.clone().requires_grad_(True)
        (f_plus,) = torch.autograd.grad(
            -energy_for_slater_and_geom(base_slater + ds, p_plus), p_plus
        )
        p_minus = pos1.clone().requires_grad_(True)
        (f_minus,) = torch.autograd.grad(
            -energy_for_slater_and_geom(base_slater - ds, p_minus), p_minus
        )
        fd_forces[..., idx] = (f_plus - f_minus) / (2 * eps)
    torch.testing.assert_close(mixed_jac, fd_forces, atol=1e-7, rtol=1e-6)
