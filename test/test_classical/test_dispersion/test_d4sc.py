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
Test for self-consistent D4 dispersion energy
Reference values obtained with tblite 0.4.0 disabling all other interactions.
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import Calculator
from dxtb._src.components.interactions.container import Charges
from dxtb._src.calculators.result import Result
from dxtb._src.typing import DD
from dxtb.components.dispersion import new_dispersion

from ...conftest import DEVICE
from ...utils import get_param_module, scf_strategy
from .samples import samples

slist = ["LiH", "SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]

opts = {
    "verbosity": 0,
    "maxiter": 50,
    "exclude": ["aes2", "es2", "es3"],
    "differentiation": "implicit",
    "strategy": scf_strategy("anderson", "potential"),
}


def _scf_charges(result: Result, batch_mode: int) -> Charges:
    """Adapt public Result charge tensors for this legacy SCF operation."""
    charge_result = result.charges
    assert charge_result is not None
    return Charges(
        mono=charge_result.mono,
        dipole=charge_result.dipole,
        quad=charge_result.quadrupole,
        batch_mode=batch_mode,
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
def test_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["edisp_d4sc"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    options = opts
    calc = Calculator(
        numbers,
        get_param_module("gfn2", **dd),
        opts=options,
        **dd,
        auto_int_level=False,
    )

    result = calc.singlepoint(positions, charges)
    d4sc = calc.interactions.get_interaction("DispersionD4SC")
    cache = d4sc.get_cache(numbers=numbers, positions=positions)

    edisp = d4sc.get_energy(
        cache, _scf_charges(result, calc.ihelp.batch_mode), calc.ihelp
    )
    assert pytest.approx(ref.cpu(), abs=10 * tol, rel=tol) == edisp.cpu()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", slist)
@pytest.mark.parametrize("name2", ["LiH", "MB16_43_01"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = (samples[name1], samples[name2])
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["edisp_d4sc"].to(**dd),
            sample2["edisp_d4sc"].to(**dd),
        )
    )

    options = opts
    calc = Calculator(
        numbers,
        get_param_module("gfn2", **dd),
        opts=options,
        **dd,
        auto_int_level=False,
    )

    result = calc.singlepoint(positions)
    d4sc = calc.interactions.get_interaction("DispersionD4SC")
    cache = d4sc.get_cache(numbers=numbers, positions=positions)

    edisp = d4sc.get_energy(
        cache, _scf_charges(result, calc.ihelp.batch_mode), calc.ihelp
    )
    assert pytest.approx(ref.cpu(), abs=10 * tol, rel=tol) == edisp.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", slist)
def test_classical(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["edisp_d4atm"].to(**dd)
    charges = torch.tensor(0.0, **dd)

    disp = new_dispersion(
        numbers,
        get_param_module("gfn2", **dd),
        charge=charges,
        ref_charges="gfn2",
        **dd,
    )
    assert disp is not None

    cache = disp.get_cache(numbers=numbers, ihelp=None)
    e = disp.get_energy(positions, cache)

    assert pytest.approx(ref.cpu(), abs=tol) == e.cpu()


def test_h3_7_d4sc_casimir_polder_derivatives_and_compatibility() -> None:
    """
    Verify H3.7 D4SC Casimir-Polder `3/pi` integration factor consistency:
    1. Analytical D4SC atom potential matches autograd `dE_disp/dq`.
    2. Second derivatives (`d^2E/dq^2` and `d^2E/dr dq`) match finite differences
       and `torch.func` Jacobians.
    3. Custom `D4Model` subclasses are preserved in `setup_d4sc` / `D4SCCoreSetup`,
       while non-`RationalDamping` strategies are explicitly rejected with `TypeError`.
    """
    from dxtb._src.components.classicals.dispersion import _d4_adapter as d4_compat
    from dxtb._src.components.interactions.dispersion.d4sc import (
        DispersionD4SC,
        build_d4sc_data,
        core_d4sc_setup,
        new_d4sc,
        setup_d4sc,
    )

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    sample = samples["MB16_43_01"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    par = get_param_module("gfn2", **dd)
    comp = new_d4sc(numbers, par, **dd)
    assert isinstance(comp, DispersionD4SC)
    q0 = torch.linspace(-0.2, 0.2, numbers.numel(), **dd)

    core_setup = core_d4sc_setup(setup_d4sc(comp, numbers))

    def energy_fn(pos_in: torch.Tensor, q_in: torch.Tensor) -> torch.Tensor:
        cache = build_d4sc_data(core_setup, pos_in)
        return comp.get_monopole_atom_energy(cache, q_in).sum()

    def pot_fn(pos_in: torch.Tensor, q_in: torch.Tensor) -> torch.Tensor:
        cache = build_d4sc_data(core_setup, pos_in)
        return comp.get_monopole_atom_potential(cache, q_in)

    pot_analytic = pot_fn(positions, q0)
    pot_auto = torch.func.jacrev(energy_fn, argnums=1)(positions, q0)
    torch.testing.assert_close(
        pot_analytic, pot_auto, rtol=1e-13, atol=1e-14
    )

    # Second derivatives d^2E/dq^2 and d^2E/dr dq
    d2e_dq2_pot = torch.func.jacrev(pot_fn, argnums=1)(positions, q0)
    d2e_dq2_ene = torch.func.jacfwd(
        torch.func.jacrev(energy_fn, argnums=1), argnums=1
    )(positions, q0)
    torch.testing.assert_close(
        d2e_dq2_pot, d2e_dq2_ene, rtol=1e-12, atol=1e-13
    )

    d2e_dqdr_pot = torch.func.jacrev(pot_fn, argnums=0)(positions, q0)
    d2e_dqdr_ene = torch.func.jacrev(
        torch.func.jacrev(energy_fn, argnums=1), argnums=0
    )(positions, q0)
    torch.testing.assert_close(
        d2e_dqdr_pot, d2e_dqdr_ene, rtol=1e-12, atol=1e-13
    )

    # Custom D4Model subclass preserved; non-RationalDamping rejected
    class _ScaledD4Model(d4_compat.model.D4Model):
        scale: float = 1.25

        def weight_references(self, cn, q, with_dgwdcn=False, with_dgwdq=False):
            res = super().weight_references(
                cn, q, with_dgwdcn=with_dgwdcn, with_dgwdq=with_dgwdq
            )
            if isinstance(res, tuple):
                return tuple(r * self.scale for r in res)
            return res * self.scale

    comp_custom_mod = DispersionD4SC(
        comp.param,
        model=_ScaledD4Model(numbers, ref_charges="gfn2", **dd),
        rcov=comp.rcov,
        r4r2=comp.r4r2,
        cutoff=comp.cutoff,
        **dd,
    )
    setup_custom = core_d4sc_setup(setup_d4sc(comp_custom_mod, numbers))
    cache_custom = build_d4sc_data(setup_custom, positions)
    e_custom = comp_custom_mod.get_monopole_atom_energy(cache_custom, q0).sum()
    torch.testing.assert_close(
        e_custom, (1.25**2) * energy_fn(positions, q0), rtol=1e-13, atol=1e-14
    )

    comp_bad_damp = DispersionD4SC(
        comp.param,
        model=comp.model,
        rcov=comp.rcov,
        r4r2=comp.r4r2,
        cutoff=comp.cutoff,
        damping_function=d4_compat.ZeroDamping(),
        **dd,
    )
    with pytest.raises(TypeError, match="RationalDamping"):
        setup_d4sc(comp_bad_damp, numbers)
