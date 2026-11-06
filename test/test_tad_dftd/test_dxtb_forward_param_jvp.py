# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Regression coverage for forward-mode AD (jvp) over DFT-D parameters in dxtb."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor

DD_CPU: DD = {"device": torch.device("cpu"), "dtype": torch.float64}

NUMBERS = torch.tensor([8, 1, 1], device=DD_CPU["device"])
POSITIONS = torch.tensor(
    [
        [0.00000000, 0.00000000, 0.22170000],
        [0.00000000, 1.43090000, -0.88670000],
        [0.00000000, -1.43090000, -0.88670000],
    ],
    **DD_CPU,
)

DISP_KEYS = ("s6", "s8", "a1", "a2", "s9")


def _set_param(par: ParamModule, name: str, value: Tensor) -> None:
    mod = par.get_submodule(name.rsplit(".", 1)[0])
    if "param" in mod._parameters:  # pylint: disable=protected-access
        del mod._parameters["param"]  # pylint: disable=protected-access
    mod.param = value


def _energy_from_disp_params(
    par: ParamModule,
    method: str,
    driver: str,
    scf_mode: str,
    disp_values: tuple[Tensor, ...],
) -> Tensor:
    dkey = "d3" if method == "gfn1" else "d4"
    for key, val in zip(DISP_KEYS, disp_values):
        _set_param(par, f"parameter_tree.dispersion.{dkey}.{key}.param", val)

    opts = {
        "int_driver": driver,
        "scf_mode": scf_mode,
        "mixer": "anderson" if scf_mode == "full" else "broyden",
        "x_atol": 1.0e-10,
        "maxiter": 100,
        "verbosity": 0,
    }
    calc = Calculator(NUMBERS, par, opts=opts, **DD_CPU)
    return calc.singlepoint(POSITIONS).total.sum()


@pytest.mark.parametrize(
    ("method", "scf_mode", "driver"),
    [
        ("gfn1", "full", "pytorch"),
        ("gfn1", "full", "libcint"),
        ("gfn1", "implicit", "libcint"),
        ("gfn2", "full", "libcint"),
        ("gfn2", "implicit", "libcint"),
    ],
)
def test_dftd_forward_jvp_matches_reverse_and_fd(
    method: str, scf_mode: str, driver: str
) -> None:
    """Forward-mode jvp over DFT-D parameters matches reverse AD and central FD."""
    if driver == "libcint" and not has_libcint:
        driver = "pytorch"

    raw_par = GFN1_XTB if method == "gfn1" else GFN2_XTB
    base_par = ParamModule(raw_par, **DD_CPU)
    dkey = "d3" if method == "gfn1" else "d4"
    named = dict(base_par.named_parameters())
    primals = tuple(
        named[f"parameter_tree.dispersion.{dkey}.{k}.param"]
        .detach()
        .clone()
        .requires_grad_(True)
        for k in DISP_KEYS
    )

    def f(*vals: Tensor) -> Tensor:
        return _energy_from_disp_params(base_par, method, driver, scf_mode, vals)

    # Reverse-mode gradient across all 5 dispersion parameters
    energy = f(*primals)
    reverse_grads = torch.autograd.grad(energy, primals)

    # Non-trivial joint tangent across all 5 dispersion parameters
    weights = (0.7, -1.1, 0.5, 0.3, -0.9)
    tangents = tuple(
        torch.full_like(p, w) for p, w in zip(primals, weights)
    )
    _, jvp_directional = torch.func.jvp(f, primals, tangents)
    rev_directional = sum(
        (r * t).sum() for r, t in zip(reverse_grads, tangents)
    )
    torch.testing.assert_close(
        jvp_directional, rev_directional, atol=1.0e-9, rtol=1.0e-7
    )

    # Single-parameter basis JVP (s8, index 1)
    s8_tangents = tuple(
        torch.ones_like(p) if i == 1 else torch.zeros_like(p)
        for i, p in enumerate(primals)
    )
    _, jvp_s8 = torch.func.jvp(f, primals, s8_tangents)
    torch.testing.assert_close(
        jvp_s8, reverse_grads[1], atol=1.0e-9, rtol=1.0e-7
    )

    # Central finite-difference reference along the joint direction
    step = 1.0e-6
    detached = tuple(p.detach() for p in primals)
    plus = tuple(p + step * t for p, t in zip(detached, tangents))
    minus = tuple(p - step * t for p, t in zip(detached, tangents))
    with torch.no_grad():
        fd_directional = (f(*plus) - f(*minus)) / (2.0 * step)
    torch.testing.assert_close(
        jvp_directional, fd_directional, atol=1.0e-7, rtol=1.0e-5
    )
