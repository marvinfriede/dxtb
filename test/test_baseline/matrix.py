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
Derivative status matrix (T0.3).

Every cell computes one quantity along one derivative path for one method,
integral driver, SCF mode and input, and compares it with a finite
difference reference (``reference/matrix/<method>-<driver>.npz``, written by
``status.py``). The finite differences are taken of the quantity one order
lower, computed by autograd through the unrolled SCF with tight convergence:

========================== ============================================
quantity                    reference
========================== ============================================
forces                      -dE/dR, central differences of energies
dipole                      -dE/dF, central differences of energies
dE_dparam                   dE/dp along a random direction, per leaf
hessian                     central differences of forces
polarizability              central differences of dipoles (field)
dipole_deriv                central differences of dipoles (positions)
dforces_dparam              u.(dF/dp)v, central differences of forces
third_order                 d/dR (w.H.v), central differences of w.H.v
hyperpolarizability         central differences of polarizabilities
pol_deriv                   central differences of polarizabilities
========================== ============================================

Paths: ``analytical`` (dxtb's analytical first derivatives, and quantities
derived from them), ``autograd`` (``torch.autograd``, the default of the
Calculator API), ``functorch`` (``torch.func.jacrev``, via the Calculator
API where it offers it), ``forward`` (``torch.func.jacfwd``/``jvp``) and
``numerical`` (dxtb's numerical Calculator methods). Batches are compared
entry by entry with the references of the single systems.
"""

from __future__ import annotations

import time
import traceback
import zlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import torch
from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, ParamModule
from dxtb._src.typing import Tensor
from dxtb.components.field import new_efield

from .molecules import get_system
from .refdata import REFERENCE_DIR

__all__ = [
    "Cell",
    "CellResult",
    "CELLS",
    "METHODS",
    "DRIVERS",
    "SCF_MODES",
    "INPUTS",
    "QUANTITIES",
    "run_cell",
]

DD = {"device": torch.device("cpu"), "dtype": torch.float64}

METHODS = ("gfn1", "gfn2")
DRIVERS = ("pytorch", "libcint")
SCF_MODES = ("full", "implicit", "nonpure")
INPUTS = ("single", "padded", "conformer")

PARAMS = {"gfn1": GFN1_XTB, "gfn2": GFN2_XTB}

SCF_OPTIONS: dict[str, Any] = {
    "maxiter": 300,
    "f_atol": 1e-10,
    "x_atol": 1e-10,
    "cache_enabled": False,
    "verbosity": 0,
}

# (quantity, order, paths, inputs)
QUANTITIES: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {
    "forces": (
        1,
        ("analytical", "autograd", "functorch", "forward", "numerical"),
        INPUTS,
    ),
    "dipole": (
        1,
        ("analytical", "autograd", "functorch", "forward", "numerical"),
        INPUTS,
    ),
    "dE_dparam": (1, ("autograd", "forward"), ("single",)),
    "hessian": (2, ("autograd", "functorch", "forward", "numerical"), INPUTS),
    "polarizability": (
        2,
        ("analytical", "autograd", "functorch", "forward", "numerical"),
        INPUTS,
    ),
    "dipole_deriv": (
        2,
        ("analytical", "autograd", "functorch", "forward", "numerical"),
        INPUTS,
    ),
    "dforces_dparam": (2, ("autograd", "forward"), ("single",)),
    "third_order": (3, ("autograd", "functorch", "forward"), ("single",)),
    "hyperpolarizability": (
        3,
        ("autograd", "functorch", "forward", "numerical"),
        INPUTS,
    ),
    "pol_deriv": (
        3,
        ("autograd", "functorch", "forward", "numerical"),
        INPUTS,
    ),
}

TOLERANCES: dict[str, tuple[float, float]] = {
    # (atol, rtol) against the finite difference reference
    "forces": (1e-6, 0.0),
    "dipole": (1e-6, 0.0),
    "dE_dparam": (1e-6, 1e-5),
    "hessian": (1e-5, 0.0),
    "polarizability": (1e-5, 1e-5),
    "dipole_deriv": (1e-5, 0.0),
    "dforces_dparam": (1e-5, 1e-4),
    "third_order": (1e-4, 1e-4),
    "hyperpolarizability": (1e-2, 1e-3),
    "pol_deriv": (1e-3, 1e-3),
}

STEPS = {
    "positions": 1e-4,
    "field": 1e-4,
    "param": 1e-5,
}
"""Steps of the finite difference references."""


###############################################################################
# Cells


@dataclass(frozen=True)
class Cell:
    quantity: str
    path: str
    method: str
    driver: str
    scf: str
    input: str

    @property
    def order(self) -> int:
        return QUANTITIES[self.quantity][0]

    @property
    def id(self) -> str:
        return (
            f"{self.quantity}-{self.path}-{self.method}-{self.driver}-"
            f"{self.scf}-{self.input}"
        )


@dataclass
class CellResult:
    status: str  # pass, fail, error, nonfinite
    max_abs: float | None = None
    max_rel: float | None = None
    seconds: float | None = None
    peak_rss_mb: float | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cells() -> list[Cell]:
    out = []
    for quantity, (_, paths, inputs) in QUANTITIES.items():
        for path in paths:
            for method in METHODS:
                for driver in DRIVERS:
                    for scf in SCF_MODES:
                        for inp in inputs:
                            out.append(
                                Cell(quantity, path, method, driver, scf, inp)
                            )
    return out


CELLS: list[Cell] = _cells()
"""All cells of the matrix."""


###############################################################################
# Systems


@lru_cache(maxsize=None)
def single_system(name: str) -> tuple[Tensor, Tensor, Tensor]:
    """Numbers, positions and charge of the single systems of the matrix."""
    if name == "H2O":
        numbers = mols["H2O"]["numbers"].clone()
        positions = mols["H2O"]["positions"].to(**DD).clone()
        return numbers, positions, torch.tensor(0.0, **DD)
    if name == "H2O*":
        numbers, positions, charge = single_system("H2O")
        gen = torch.Generator().manual_seed(7)
        disp = torch.randn(positions.shape, generator=gen, **DD)
        return numbers, positions + 0.05 * disp, charge
    if name == "OH-":
        s = get_system("OH-")
        return s.numbers.clone(), s.positions.clone(), s.charge.clone()
    raise KeyError(name)


INPUT_SYSTEMS = {
    "single": ("H2O",),
    "padded": ("H2O", "OH-"),
    "conformer": ("H2O", "H2O*"),
}
"""Single systems that make up each input."""


def input_system(inp: str) -> tuple[Tensor, Tensor, Tensor, int]:
    """Numbers, positions, charge and batch mode of an input."""
    names = INPUT_SYSTEMS[inp]
    if inp == "single":
        return (*single_system(names[0]), 0)

    parts = [single_system(n) for n in names]
    numbers = pack([p[0] for p in parts])
    positions = pack([p[1] for p in parts])
    charge = torch.stack([p[2] for p in parts])
    return numbers, positions, charge, 1 if inp == "padded" else 2


###############################################################################
# Calculators and building blocks (also used for the references)


def calculator(
    method: str,
    driver: str,
    scf: str,
    numbers: Tensor,
    batch_mode: int = 0,
    field: Tensor | None = None,
    par: ParamModule | None = None,
) -> Calculator:
    opts = dict(
        SCF_OPTIONS,
        scf_mode=scf,
        mixer="anderson" if scf == "full" else "broyden",
        int_driver=driver,
        batch_mode=batch_mode,
    )
    interaction = [new_efield(field)] if field is not None else None
    return Calculator(
        numbers,
        PARAMS[method] if par is None else par,
        opts=opts,
        interaction=interaction,
        **DD,
    )


def zero_field(requires_grad: bool = True) -> Tensor:
    return torch.zeros(3, **DD, requires_grad=requires_grad)


def energy(method, driver, scf, numbers, positions, charge, bm=0) -> Tensor:
    calc = calculator(method, driver, scf, numbers, bm)
    return calc.energy(positions, charge)


def energy_in_field(method, driver, scf, numbers, positions, charge, field):
    calc = calculator(method, driver, scf, numbers, field=field)
    return calc.energy(positions, charge)


def forces_ad(method, driver, scf, numbers, positions, charge) -> Tensor:
    pos = positions.detach().clone().requires_grad_(True)
    e = energy(method, driver, scf, numbers, pos, charge)
    (g,) = torch.autograd.grad(e, pos)
    return -g


def dipole_ad(method, driver, scf, numbers, positions, charge, field=None):
    f = zero_field() if field is None else field.detach().clone()
    f.requires_grad_(True)
    e = energy_in_field(method, driver, scf, numbers, positions, charge, f)
    (g,) = torch.autograd.grad(e, f)
    return -g


def polarizability_ad(
    method, driver, scf, numbers, positions, charge, field=None
):
    f = zero_field() if field is None else field.detach().clone()
    f.requires_grad_(True)
    e = energy_in_field(method, driver, scf, numbers, positions, charge, f)
    (g,) = torch.autograd.grad(e, f, create_graph=True)
    rows = [
        torch.autograd.grad(g[i], f, retain_graph=True)[0] for i in range(3)
    ]
    return -torch.stack(rows)


def hvp_scalar(method, driver, scf, numbers, positions, charge, v, w) -> Tensor:
    """``w . H v`` by double backward."""
    pos = positions.detach().clone().requires_grad_(True)
    e = energy(method, driver, scf, numbers, pos, charge)
    (g,) = torch.autograd.grad(e, pos, create_graph=True)
    (hv,) = torch.autograd.grad((g * v).sum(), pos)
    return (hv * w).sum()


###############################################################################
# Parameters


def param_module(method: str) -> ParamModule:
    return ParamModule(PARAMS[method], **DD)


def param_leaves(method: str, numbers: Tensor) -> list[str]:
    """
    Floating point leaves of the parametrization relevant for ``numbers``:
    global leaves (except ``meta``), leaves of the elements present and
    element pairs of present elements.
    """
    # pylint: disable=import-outside-toplevel
    from tad_mctc.data import pse

    symbols = {pse.Z2S[int(z)] for z in torch.unique(numbers) if z > 0}
    out = []
    for name, p in param_module(method).named_parameters():
        if not p.is_floating_point() or ".meta." in name:
            continue
        if ".element." in name:
            if name.split(".element.")[1].split(".")[0] not in symbols:
                continue
        if ".kpair." in name:
            pair = name.split(".kpair.")[1].split(".")[0]
            if not set(pair.split("-")) <= symbols:
                continue
        out.append(name)
    return out


def _direction(shape: torch.Size, seed: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(shape, generator=gen, **DD)
    return v / v.norm().clamp(min=1e-12)


def param_direction(method: str, name: str) -> Tensor:
    p = dict(param_module(method).named_parameters())[name]
    # not `hash()`, which is randomized per process
    return _direction(p.shape, zlib.crc32(name.encode()))


def _set_param(par: ParamModule, name: str, value: Tensor) -> None:
    """
    Replace a leaf by a plain tensor (which may be a ``torch.func``
    wrapper); the parametrization reads ``ParameterModule.param``.
    """
    mod = par.get_submodule(name.rsplit(".", 1)[0])
    if "param" in mod._parameters:  # pylint: disable=protected-access
        del mod._parameters["param"]  # pylint: disable=protected-access
    mod.param = value


def param_energy(method, driver, scf, numbers, positions, charge, name, value):
    """Energy with one leaf replaced by ``value`` (may require grad)."""
    par = param_module(method)
    _set_param(par, name, value)
    calc = calculator(method, driver, scf, numbers, par=par)
    return calc.energy(positions, charge)


def param_forces(method, driver, scf, numbers, positions, charge, name, value):
    par = param_module(method)
    _set_param(par, name, value)
    calc = calculator(method, driver, scf, numbers, par=par)
    pos = positions.detach().clone().requires_grad_(True)
    e = calc.energy(pos, charge)
    (g,) = torch.autograd.grad(e, pos, create_graph=value.requires_grad)
    return -g


###############################################################################
# References


def reference_file(method: str, driver: str):
    return REFERENCE_DIR / "matrix" / f"{method}-{driver}.npz"


@lru_cache(maxsize=None)
def load_references(method: str, driver: str) -> dict[str, np.ndarray]:
    with np.load(reference_file(method, driver), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def reference_for(cell: Cell) -> np.ndarray | dict[str, np.ndarray]:
    refs = load_references(cell.method, cell.driver)
    names = INPUT_SYSTEMS[cell.input]

    if cell.quantity in ("dE_dparam", "dforces_dparam"):
        prefix = f"{names[0]}.{cell.quantity}."
        return {
            k[len(prefix) :]: v for k, v in refs.items() if k.startswith(prefix)
        }

    values = [refs[f"{n}.{cell.quantity}"] for n in names]
    if cell.input == "single":
        return values[0]

    # pad atom dimensions to the batch size
    nat = max(len(single_system(n)[0]) for n in names)
    return np.stack([_pad_atoms(v, cell.quantity, nat) for v in values])


def _pad_atoms(v: np.ndarray, quantity: str, nat: int) -> np.ndarray:
    atom_axes = {
        "forces": (0,),
        "hessian": (0, 2),
        "dipole_deriv": (1,),
        "pol_deriv": (2,),
    }.get(quantity, ())
    pad = [(0, 0)] * v.ndim
    for ax in atom_axes:
        pad[ax] = (0, nat - v.shape[ax])
    return np.pad(v, pad)


###############################################################################
# Paths


def _func_energy(
    calc: Calculator, charge: Tensor
) -> Callable[[Tensor], Tensor]:
    """Energy of positions, summed over a batch (independent systems)."""

    def f(pos: Tensor) -> Tensor:
        return calc.energy(pos, charge).sum()

    return f


def _func_field_energy(
    calc: Calculator, pos: Tensor, charge: Tensor
) -> Callable[[Tensor], Tensor]:
    def f(field: Tensor) -> Tensor:
        calc.interactions.update_efield(field=field)
        return calc.energy(pos, charge)

    return f


def _block_diag(x: Tensor, nb: int, lead: int) -> Tensor:
    """
    Extract per-system blocks from the derivative of a batch-summed or
    batched function computed with ``torch.func``: ``x`` has shape
    ``(*out, *in)`` with ``in = (nb, ...)``; ``lead`` is the number of
    output dimensions per system.
    """
    if nb == 0:
        return x
    return torch.stack(
        [x[(b,) + (slice(None),) * lead + (b,)] for b in range(nb)]
    )


def _run(cell: Cell) -> Tensor | dict[str, Tensor]:
    numbers, positions, charge, bm = input_system(cell.input)
    nb = numbers.shape[0] if bm > 0 else 0
    m, d, s = cell.method, cell.driver, cell.scf
    q, path = cell.quantity, cell.path

    def calc(field: Tensor | None = None) -> Calculator:
        return calculator(m, d, s, numbers, bm, field=field)

    def pos_grad() -> Tensor:
        return positions.detach().clone().requires_grad_(True)

    # pylint: disable=import-outside-toplevel
    from torch.func import jacfwd, jacrev, jvp

    if q == "forces":
        if path == "analytical":
            return calc().forces_analytical(pos_grad(), charge)
        if path == "autograd":
            return calc().forces(pos_grad(), charge)
        if path == "functorch":
            return calc().forces(pos_grad(), charge, grad_mode="functorch")
        if path == "forward":
            c = calc()
            return -jacfwd(_func_energy(c, charge))(positions.clone())
        if path == "numerical":
            return calc().forces_numerical(positions.clone(), charge)

    if q == "dipole":
        if path == "analytical":
            return calc(zero_field()).dipole_analytical(
                positions.clone(), charge
            )
        if path == "autograd":
            return calc(zero_field()).dipole(positions.clone(), charge)
        if path == "functorch":
            return calc(zero_field()).dipole(
                positions.clone(), charge, use_functorch=True
            )
        if path == "forward":
            c = calc(zero_field(False))
            f = _func_field_energy(c, positions.clone(), charge)
            return -jacfwd(f)(zero_field(False))
        if path == "numerical":
            return calc(zero_field(False)).dipole_numerical(
                positions.clone(), charge
            )

    if q == "hessian":
        if path == "autograd":
            return calc().hessian(pos_grad(), charge)
        if path == "functorch":
            return calc().hessian(pos_grad(), charge, use_functorch=True)
        if path == "forward":
            c = calc()
            h = jacfwd(jacrev(_func_energy(c, charge)))(positions.clone())
            return _block_diag(h, nb, 2) if nb else h
        if path == "numerical":
            return calc().hessian_numerical(positions.clone(), charge)

    if q == "polarizability":
        if path in ("analytical", "autograd", "functorch"):
            return calc(zero_field()).polarizability(
                positions.clone(),
                charge,
                use_functorch=path == "functorch",
                use_analytical=path == "analytical",
            )
        if path == "forward":
            c = calc(zero_field(False))
            f = _func_field_energy(c, positions.clone(), charge)
            return -jacfwd(jacrev(f))(zero_field(False))
        if path == "numerical":
            return calc(zero_field(False)).polarizability_numerical(
                positions.clone(), charge
            )

    if q == "dipole_deriv":
        if path in ("analytical", "autograd", "functorch"):
            return calc(zero_field()).dipole_deriv(
                pos_grad(),
                charge,
                use_analytical_dipmom=path == "analytical",
                use_functorch=path == "functorch",
            )
        if path == "forward":
            c = calc(zero_field(False))

            def mu(pos: Tensor) -> Tensor:
                f = _func_field_energy(c, pos, charge)
                return -jacrev(f)(zero_field(False))

            out = jacfwd(mu)(positions.clone())
            return _block_diag(out, nb, 1) if nb else out
        if path == "numerical":
            return calc(zero_field(False)).dipole_deriv_numerical(
                positions.clone(), charge
            )

    if q == "hyperpolarizability":
        if path in ("autograd", "functorch"):
            return calc(zero_field()).hyperpolarizability(
                positions.clone(), charge, use_functorch=path == "functorch"
            )
        if path == "forward":
            c = calc(zero_field(False))
            f = _func_field_energy(c, positions.clone(), charge)
            return -jacfwd(jacfwd(jacrev(f)))(zero_field(False))
        if path == "numerical":
            return calc(zero_field(False)).hyperpolarizability_numerical(
                positions.clone(), charge
            )

    if q == "pol_deriv":
        if path in ("autograd", "functorch"):
            return calc(zero_field()).pol_deriv(
                pos_grad(), charge, use_functorch=path == "functorch"
            )
        if path == "forward":
            c = calc(zero_field(False))

            def alpha(pos: Tensor) -> Tensor:
                f = _func_field_energy(c, pos, charge)
                return -jacfwd(jacrev(f))(zero_field(False))

            out = jacfwd(alpha)(positions.clone())
            return _block_diag(out, nb, 2) if nb else out
        if path == "numerical":
            return calc(zero_field(False)).pol_deriv_numerical(
                positions.clone(), charge
            )

    if q == "third_order":
        v, w = third_order_directions(positions)
        if path == "autograd":
            pos = pos_grad()
            e = calc().energy(pos, charge)
            (g,) = torch.autograd.grad(e, pos, create_graph=True)
            (hv,) = torch.autograd.grad((g * v).sum(), pos, create_graph=True)
            (t,) = torch.autograd.grad((hv * w).sum(), pos)
            return t
        c = calc()
        grad = jacrev(_func_energy(c, charge))

        def whv(pos: Tensor) -> Tensor:
            return (jvp(grad, (pos,), (v,))[1] * w).sum()

        if path == "functorch":
            return jacrev(whv)(positions.clone())
        if path == "forward":
            return jacfwd(whv)(positions.clone())

    if q == "dE_dparam":
        return _param_derivatives(cell, numbers, positions, charge, order=1)

    if q == "dforces_dparam":
        return _param_derivatives(cell, numbers, positions, charge, order=2)

    raise NotImplementedError(cell.id)


def third_order_directions(positions: Tensor) -> tuple[Tensor, Tensor]:
    return _direction(positions.shape, 11), _direction(positions.shape, 12)


def force_direction(positions: Tensor) -> Tensor:
    return _direction(positions.shape, 13)


def _param_derivatives(
    cell, numbers, positions, charge, order
) -> dict[str, Tensor]:
    """Directional derivatives per leaf, all leaves at once."""
    # pylint: disable=import-outside-toplevel
    from torch.func import grad as torch_grad
    from torch.func import jvp

    m, d, s = cell.method, cell.driver, cell.scf
    names = param_leaves(m, numbers)
    par = param_module(m)
    params = dict(par.named_parameters())
    u = force_direction(positions)

    if cell.path == "autograd":
        # the user's way: switch on gradients of the parameters themselves
        leaves = {name: params[name].requires_grad_(True) for name in names}
        calc = calculator(m, d, s, numbers, par=par)
        if order == 1:
            out = calc.energy(positions, charge)
        else:
            pos = positions.detach().clone().requires_grad_(True)
            e = calc.energy(pos, charge)
            (g,) = torch.autograd.grad(e, pos, create_graph=True)
            out = (-g * u).sum()
        grads = torch.autograd.grad(
            out, list(leaves.values()), allow_unused=True
        )
        result = {}
        for name, g in zip(names, grads):
            v = param_direction(m, name)
            result[name] = torch.zeros((), **DD) if g is None else (g * v).sum()
        return result

    if cell.path == "forward":
        result = {}
        for name in names:
            v = param_direction(m, name)
            p0 = params[name].detach().clone()

            def f(p: Tensor) -> Tensor:
                if order == 1:
                    return param_energy(
                        m, d, s, numbers, positions, charge, name, p
                    )

                # forces with torch.func (no `requires_grad_` under jvp)
                def e(pos: Tensor) -> Tensor:
                    return param_energy(m, d, s, numbers, pos, charge, name, p)

                return (-torch_grad(e)(positions) * u).sum()

            result[name] = jvp(f, (p0,), (v,))[1]
        return result

    raise NotImplementedError(cell.id)


###############################################################################
# Comparison


def _compare(
    cell: Cell, value: Tensor | dict[str, Tensor]
) -> tuple[str, float, float, str]:
    ref = reference_for(cell)
    atol, rtol = TOLERANCES[cell.quantity]

    if isinstance(ref, dict):
        assert isinstance(value, dict)
        new = np.array([float(value[k].detach()) for k in ref])
        old = np.array([float(ref[k]) for k in ref])
        names = list(ref)
    else:
        assert isinstance(value, Tensor)
        new = value.detach().cpu().numpy()
        old = ref
        names = []
        if new.shape != old.shape:
            return "fail", np.inf, np.inf, f"shape {new.shape} != {old.shape}"

    if not np.all(np.isfinite(new)):
        return "nonfinite", np.inf, np.inf, "non-finite values"

    diff = np.abs(new - old)
    scale = np.abs(old)
    ok = diff <= atol + rtol * scale
    max_abs = float(diff.max()) if diff.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 1e-8, diff / scale, 0.0)
    max_rel = float(rel.max()) if rel.size else 0.0

    msg = ""
    if names and not ok.all():
        bad = [
            n.replace("parameter_tree.", "").replace(".param", "")
            for n, o in zip(names, ok.ravel())
            if not o
        ]
        msg = "leaves: " + ", ".join(bad)
    return ("pass" if ok.all() else "fail"), max_abs, max_rel, msg


def _peak_rss_mb() -> float:
    # pylint: disable=import-outside-toplevel
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_cell(cell: Cell) -> CellResult:
    """Run one cell and classify the outcome."""
    torch.manual_seed(0)
    start = time.perf_counter()
    try:
        value = _run(cell)
    except Exception as e:  # pylint: disable=broad-except
        tb = traceback.extract_tb(e.__traceback__)
        where = (
            f"{tb[-1].filename.split('/src/')[-1]}:{tb[-1].lineno}"
            if tb
            else ""
        )
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300]} [{where}]"
        return CellResult(
            "error",
            seconds=time.perf_counter() - start,
            peak_rss_mb=_peak_rss_mb(),
            message=msg,
        )
    seconds = time.perf_counter() - start
    status, max_abs, max_rel, msg = _compare(cell, value)
    return CellResult(status, max_abs, max_rel, seconds, _peak_rss_mb(), msg)
