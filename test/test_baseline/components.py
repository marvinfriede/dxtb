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
Component-level derivative checks (T0.4).

Every target is a function of one float64 tensor (positions, a parameter,
charges or a matrix) on a small system. Checks:

=========== ===============================================================
check        what
=========== ===============================================================
order1       ``torch.autograd.gradcheck``
order2       ``torch.autograd.gradgradcheck``
order3       ``gradgradcheck`` of the projected gradient ``x -> d(f.r)/dx``
forward      ``gradcheck(check_forward_ad=True)`` (``torch.autograd.forward_ad``)
jvp          ``torch.func.jvp`` against finite differences
vmap         ``torch.func.vmap`` over two inputs against a loop, with vmap
             fallback warnings turned into errors
=========== ===============================================================

``OverlapAG`` and ``EFunction`` of the plan (at ``46af7bc``) no longer
exist: since #268/#270 the PyTorch integrals are plain torch code (pair
builder with McMurchie-Davidson and Obara-Saika kernels). They are checked
as ``overlap_md``, ``overlap_os``, ``dipint`` and ``quadint`` instead.
"""

from __future__ import annotations

import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Tuple

import torch

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.typing import Tensor

from ..molecules import mols

__all__ = ["TARGETS", "CHECKS", "ComponentResult", "run_check"]

DD = {"device": torch.device("cpu"), "dtype": torch.float64}

CHECKS = ("order1", "order2", "order3", "forward", "jvp", "vmap")


@dataclass
class ComponentResult:
    status: str  # pass, fail, error
    seconds: float | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Target = Callable[[], Tuple[Callable[[Tensor], Tensor], Tensor]]


@lru_cache(maxsize=None)
def _water() -> tuple[Tensor, Tensor]:
    return mols["H2O"]["numbers"].clone(), mols["H2O"]["positions"].to(**DD)


@lru_cache(maxsize=None)
def _lih() -> tuple[Tensor, Tensor]:
    # small system with s and p shells on both atoms (quick integrals)
    return mols["LiH"]["numbers"].clone(), mols["LiH"]["positions"].to(**DD)


def _par(method: str) -> ParamModule:
    return ParamModule(GFN1_XTB if method == "gfn1" else GFN2_XTB, **DD)


def _leaf(x: Tensor) -> Tensor:
    return x.detach().clone().requires_grad_(True)


###############################################################################
# classical terms (positions and parameters)


def _repulsion(method: str, wrt: str) -> Target:
    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb._src.components.classicals import new_repulsion

        numbers, positions = _water()
        par = _par(method)
        ihelp = IndexHelper.from_numbers(numbers, par)
        rep = new_repulsion(
            torch.unique(numbers),
            par,
            **DD,
        )
        assert rep is not None
        cache = rep.get_cache(numbers, ihelp)

        if wrt == "positions":
            return (lambda x: rep.get_energy(x, cache)), _leaf(positions)

        arep0 = cache.arep.detach().clone()

        def f(arep: Tensor) -> Tensor:
            c = rep.get_cache(numbers, ihelp)
            c.arep = arep
            return rep.get_energy(positions, c)

        return f, _leaf(arep0)

    return build


def _classical(name: str, method: str) -> Target:
    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb._src.components import classicals as cl
        from dxtb._src.constants import defaults

        numbers, positions = _water()
        par = _par(method)
        ihelp = IndexHelper.from_numbers(numbers, par)
        unique = torch.unique(numbers)
        if name == "halogen":
            # needs a halogen bond donor/acceptor pair
            numbers = torch.tensor([35, 35, 8, 1, 1])
            positions = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.3],
                    [0.0, 0.0, 9.6],
                    [1.4, 0.0, 10.7],
                    [-1.4, 0.0, 10.7],
                ],
                **DD,
            )
            ihelp = IndexHelper.from_numbers(numbers, par)
            comp = cl.new_halogen(torch.unique(numbers), par, **DD)
        elif name == "dispersion":
            comp = cl.new_dispersion(
                numbers, par, charge=torch.tensor(defaults.CHRG, **DD), **DD
            )
        elif name == "repulsion":
            comp = cl.new_repulsion(unique, par, **DD)
        else:
            raise KeyError(name)
        assert comp is not None
        cache = comp.get_cache(numbers, ihelp)
        return (lambda x: comp.get_energy(x, cache)), _leaf(positions)

    return build


###############################################################################
# ES2 Coulomb matrix


def _es2_coulomb(wrt: str) -> Target:
    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb._src.components.interactions.coulomb import (
            build_es2_coulomb,
            new_es2,
            setup_es2,
        )

        numbers, positions = _water()
        par = _par("gfn1")
        ihelp = IndexHelper.from_numbers(numbers, par)
        es2 = new_es2(torch.unique(numbers), par, **DD)
        assert es2 is not None

        if wrt == "positions":
            setup = setup_es2(
                numbers,
                es2.hubbard,
                ihelp,
                lhubbard=es2.lhubbard,
                gexp=es2.gexp,
                average=es2.average,
                shell_resolved=True,
            )
            return (lambda x: build_es2_coulomb(setup, x)), _leaf(positions)

        hubbard0 = es2.hubbard.detach().clone()

        def f(hubbard: Tensor) -> Tensor:
            setup = setup_es2(
                numbers,
                hubbard,
                ihelp,
                lhubbard=es2.lhubbard,
                gexp=es2.gexp,
                average=es2.average,
                shell_resolved=True,
            )
            return build_es2_coulomb(setup, positions)

        return f, _leaf(hubbard0)

    return build


###############################################################################
# interactions: energy as a function of the charges


def _interaction(name: str) -> Target:
    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb._src.components.interactions import coulomb, dispersion
        from dxtb._src.components.interactions.container import Charges

        numbers, positions = _water()
        method = "gfn1" if name in ("es2", "es3") else "gfn2"
        par = _par(method)
        ihelp = IndexHelper.from_numbers(numbers, par)
        unique = torch.unique(numbers)
        if name == "es2":
            comp = coulomb.new_es2(unique, par, **DD)
        elif name == "es3":
            comp = coulomb.new_es3(unique, par, **DD)
        elif name == "aes2":
            comp = coulomb.new_aes2(unique, par, **DD)
        elif name == "d4sc":
            comp = dispersion.new_d4sc(numbers, par, **DD)
        else:
            raise KeyError(name)
        assert comp is not None
        cache = comp.get_cache(
            numbers=numbers, positions=positions, ihelp=ihelp
        )

        norb = ihelp.nao
        gen = torch.Generator().manual_seed(5)
        q0 = 0.1 * torch.randn(norb, generator=gen, **DD)

        if name == "aes2":
            # atomic dipoles as the variable, fixed monopoles
            dp0 = 0.05 * torch.randn((len(numbers), 3), generator=gen, **DD)
            qp = 0.05 * torch.randn((len(numbers), 6), generator=gen, **DD)

            def f(dp: Tensor) -> Tensor:
                charges = Charges(mono=q0, dipole=dp, quad=qp)
                return comp.get_energy(cache, charges, ihelp)

            return f, _leaf(dp0)

        def g(q: Tensor) -> Tensor:
            return comp.get_energy(cache, Charges(mono=q), ihelp)

        return g, _leaf(q0)

    return build


def _interaction_positions(name: str) -> Target:
    """ES2 energy at fixed charges as a function of positions (cache)."""

    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb._src.components.interactions import coulomb
        from dxtb._src.components.interactions.container import Charges

        numbers, positions = _water()
        par = _par("gfn1")
        ihelp = IndexHelper.from_numbers(numbers, par)
        comp = coulomb.new_es2(torch.unique(numbers), par, **DD)
        assert comp is not None
        gen = torch.Generator().manual_seed(5)
        q = 0.1 * torch.randn(ihelp.nao, generator=gen, **DD)

        def f(x: Tensor) -> Tensor:
            comp.cache_invalidate()
            cache = comp.get_cache(numbers=numbers, positions=x, ihelp=ihelp)
            return comp.get_energy(cache, Charges(mono=q), ihelp)

        return f, _leaf(positions)

    return build


###############################################################################
# integrals and core Hamiltonian


def _integral(kind: str, algorithm: str = "os", method: str = "gfn2") -> Target:
    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb import labels
        from dxtb._src.integral.driver.manager import DriverManager
        from dxtb.integrals import factories

        numbers, positions = _lih()
        par = _par(method)
        ihelp = IndexHelper.from_numbers(numbers, par)

        def f(x: Tensor) -> Tensor:
            mgr = DriverManager(
                labels.INTDRIVER_PYTORCH, algorithm=algorithm, **DD
            )
            mgr.create_driver(numbers, par, ihelp)
            mgr.driver.setup(x)
            if kind == "overlap":
                integral = factories.new_overlap(mgr.driver_type, **DD)
            elif kind == "dipint":
                integral = factories.new_dipint(mgr.driver_type, **DD)
            elif kind == "quadint":
                integral = factories.new_quadint(mgr.driver_type, **DD)
            else:
                raise KeyError(kind)
            integral.build(mgr.driver)
            return integral.matrix

        return f, _leaf(positions)

    return build


def _hcore(method: str) -> Target:
    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb._src.basis.bas import Basis
        from dxtb._src.integral.driver.pytorch import (
            build_overlap,
            setup_integrals,
        )
        from dxtb._src.xtb.h0 import build_hcore, setup_h0

        numbers, positions = _lih()
        par = _par(method)
        ihelp = IndexHelper.from_numbers(numbers, par)
        h0_setup = setup_h0(numbers, par, ihelp)
        integral_setup = setup_integrals(Basis(numbers, par, ihelp, **DD))

        def f(x: Tensor) -> Tensor:
            overlap = build_overlap(integral_setup, x)
            return build_hcore(h0_setup, x, overlap)[0]

        return f, _leaf(positions)

    return build


def _basis_exponent() -> Target:
    """Overlap as a function of the Slater exponents (basis parameter)."""

    def build():
        # pylint: disable=import-outside-toplevel
        from dxtb import labels
        from dxtb._src.integral.driver.manager import DriverManager
        from dxtb.integrals import factories

        numbers, positions = _lih()
        par = _par("gfn1")
        ihelp = IndexHelper.from_numbers(numbers, par)
        slater = par.get("element.Li.slater")
        x0 = slater.detach().clone()

        def f(x: Tensor) -> Tensor:
            p = _par("gfn1")
            mod = p.get("element.Li.slater", unwrapped=False)
            del mod._parameters["param"]  # pylint: disable=protected-access
            mod.param = x
            mgr = DriverManager(labels.INTDRIVER_PYTORCH, **DD)
            mgr.create_driver(numbers, p, ihelp)
            mgr.driver.setup(positions)
            ovlp = factories.new_overlap(mgr.driver_type, **DD)
            ovlp.build(mgr.driver)
            return ovlp.matrix

        return f, _leaf(x0)

    return build


###############################################################################
# eigensolver


def _eigensolver(spectrum: str, filling: str) -> Target:
    """
    Density matrix of a generalized eigenproblem through the solver of the
    unrolled SCF (``storch.eighb``, Lorentzian broadening with factor
    ``sqrt(eps)``), as a function of the Hamiltonian matrix. Two electrons
    pairs in five orbitals:

    - ``degenerate``: exactly degenerate pair within the occupied orbitals
      (``-0.5, -0.5``), gap 0.8 Eh; the density is well defined.
    - ``small_gap``: HOMO-LUMO gap of 1e-3 Eh.

    ``filling``: ``aufbau`` (integer occupations) or ``fermi`` (Fermi
    smearing at 300 K).
    """

    def build():
        # pylint: disable=import-outside-toplevel
        from tad_mctc import storch

        from dxtb._src.wavefunction.filling import get_fermi_occupation

        n = 5
        gen = torch.Generator().manual_seed(9)
        q, _ = torch.linalg.qr(torch.randn((n, n), generator=gen, **DD))
        if spectrum == "degenerate":
            evals = torch.tensor([-1.0, -0.5, -0.5, 0.3, 0.8], **DD)
            nocc = 3
        else:
            evals = torch.tensor([-1.0, -0.6, -0.3, -0.299, 0.8], **DD)
            nocc = 3
        h0 = q @ torch.diag(evals) @ q.T
        s = torch.eye(n, **DD) + 0.05 * (q[:, :1] @ q[:, :1].T)

        def f(h: Tensor) -> Tensor:
            h = 0.5 * (h + h.T)
            e, c = storch.eighb(
                a=h,
                b=s,
                is_posdef=True,
                factor=torch.finfo(h.dtype).eps ** 0.5,
                broadening_method="lorn",
            )
            if filling == "aufbau":
                occ = torch.tensor([2.0] * nocc + [0.0] * (n - nocc), **DD)
            else:
                kt = torch.tensor(9.5e-4, **DD)
                nel = torch.tensor(float(nocc), **DD)
                occ = 2 * get_fermi_occupation(nel, e, kt)
            return (c * occ) @ c.T

        return f, _leaf(h0)

    return build


TARGETS: dict[str, Target] = {
    "repulsion.positions": _repulsion("gfn1", "positions"),
    "repulsion.arep": _repulsion("gfn1", "arep"),
    "es2_coulomb.positions": _es2_coulomb("positions"),
    "es2_coulomb.hubbard": _es2_coulomb("hubbard"),
    "overlap_md.positions": _integral("overlap", "md"),
    "overlap_os.positions": _integral("overlap", "os"),
    "dipint_os.positions": _integral("dipint", "os"),
    "quadint_os.positions": _integral("quadint", "os"),
    "overlap.slater": _basis_exponent(),
    "hcore_gfn1.positions": _hcore("gfn1"),
    "hcore_gfn2.positions": _hcore("gfn2"),
    "halogen.positions": _classical("halogen", "gfn1"),
    "dispersion_d3.positions": _classical("dispersion", "gfn1"),
    "dispersion_d4.positions": _classical("dispersion", "gfn2"),
    "repulsion_gfn2.positions": _classical("repulsion", "gfn2"),
    "repulsion_gfn2.arep": _repulsion("gfn2", "arep"),
    "es2.charges": _interaction("es2"),
    "es3.charges": _interaction("es3"),
    "aes2.dipoles": _interaction("aes2"),
    "d4sc.charges": _interaction("d4sc"),
    "es2.positions": _interaction_positions("es2"),
    "eigensolver.degenerate_aufbau": _eigensolver("degenerate", "aufbau"),
    "eigensolver.degenerate_fermi": _eigensolver("degenerate", "fermi"),
    "eigensolver.small_gap_aufbau": _eigensolver("small_gap", "aufbau"),
    "eigensolver.small_gap_fermi": _eigensolver("small_gap", "fermi"),
}
"""Targets by name (``<function>.<variable>``)."""


###############################################################################
# checks


def _projected_grad(f: Callable[[Tensor], Tensor], x: Tensor) -> Callable:
    gen = torch.Generator().manual_seed(3)
    r = torch.randn(f(x).shape, generator=gen, **DD)

    def g(y: Tensor) -> Tensor:
        (d,) = torch.autograd.grad((f(y) * r).sum(), y, create_graph=True)
        return d

    return g


def _jvp(f: Callable[[Tensor], Tensor], x: Tensor) -> None:
    # pylint: disable=import-outside-toplevel
    from torch.func import jvp

    gen = torch.Generator().manual_seed(4)
    v = torch.randn(x.shape, generator=gen, **DD)
    x0 = x.detach()
    _, tangent = jvp(f, (x0,), (v,))
    h = 1e-6
    fd = (f(x0 + h * v) - f(x0 - h * v)) / (2 * h)
    err = (tangent - fd.detach()).abs().max().item()
    scale = max(1.0, fd.abs().max().item())
    if not err <= 1e-5 * scale:
        raise AssertionError(
            f"jvp differs from finite differences by {err:.2e}"
        )


def _vmap(f: Callable[[Tensor], Tensor], x: Tensor) -> None:
    # pylint: disable=import-outside-toplevel
    from torch.func import vmap

    gen = torch.Generator().manual_seed(6)
    x0 = x.detach()
    xs = torch.stack(
        [x0, x0 + 1e-3 * torch.randn(x0.shape, generator=gen, **DD)]
    )

    # pylint: disable=protected-access
    torch._C._functorch._set_vmap_fallback_warning_enabled(True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*fallback.*")
            warnings.filterwarnings("error", message=".*vmap.*")
            out = vmap(f)(xs)
    finally:
        torch._C._functorch._set_vmap_fallback_warning_enabled(False)

    ref = torch.stack([f(xi) for xi in xs]).detach()
    err = (out.detach() - ref).abs().max().item()
    if not err <= 1e-10 * max(1.0, ref.abs().max().item()):
        raise AssertionError(f"vmap differs from a loop by {err:.2e}")


def _run(check: str, f: Callable[[Tensor], Tensor], x: Tensor) -> None:
    gc = torch.autograd.gradcheck
    ggc = torch.autograd.gradgradcheck
    if check == "order1":
        assert gc(f, (x,), atol=1e-6, rtol=1e-5)
    elif check == "order2":
        assert ggc(f, (x,), atol=1e-6, rtol=1e-5)
    elif check == "order3":
        assert ggc(_projected_grad(f, x), (x,), atol=1e-6, rtol=1e-5)
    elif check == "forward":
        assert gc(
            f,
            (x,),
            atol=1e-6,
            rtol=1e-5,
            check_forward_ad=True,
            check_backward_ad=False,
            check_undefined_grad=False,
            check_batched_grad=False,
        )
    elif check == "jvp":
        _jvp(f, x)
    elif check == "vmap":
        _vmap(f, x)
    else:
        raise KeyError(check)


def run_check(target: str, check: str) -> ComponentResult:
    """Run one check of one target."""
    start = time.perf_counter()
    try:
        f, x = TARGETS[target]()
        _run(check, f, x)
    except Exception as e:  # pylint: disable=broad-except
        tb = traceback.extract_tb(e.__traceback__)
        where = ""
        for frame in reversed(tb):
            if (
                "/dxtb/" in frame.filename
                and "test_baseline" not in frame.filename
            ):
                where = f"{frame.filename.split('/src/')[-1]}:{frame.lineno}"
                break
        status = (
            "fail"
            if isinstance(e, AssertionError)
            or type(e).__name__ == "GradcheckError"
            else "error"
        )
        first = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
        msg = f"{type(e).__name__}: {first}" + (f" [{where}]" if where else "")
        return ComponentResult(status, time.perf_counter() - start, msg)
    return ComponentResult("pass", time.perf_counter() - start)
