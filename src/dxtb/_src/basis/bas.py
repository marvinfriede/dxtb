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
Basis: Main Class
=================

Main basis set class for creating the contracted Gaussian type orbitals (CGTOs)
from the parametrization. The basis set can also be printed in various formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tad_mctc.data import pse

from dxtb._src.constants.defaults import DEFAULT_BASIS_INT
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import Literal, Self, Tensor, TensorLike, override

from .indexhelper import IndexHelper
from .ortho import orthogonalize
from .slater import slater_to_gauss

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = ["Basis"]


def _to_int_list(tensor: Tensor) -> list[int]:
    """Convert a 1D integer tensor to Python ints, including inside transforms."""
    flat = tensor.reshape(-1)
    try:
        return [int(x) for x in flat.tolist()]
    except RuntimeError:
        return [int(v.item()) for v in flat]


angular2label = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
}


class Basis(TensorLike):
    """Atomic orbital basis set."""

    ngauss: Tensor
    """Number of Gaussians used in expansion from Slater orbital."""

    shells: dict[str, list[str]]
    """Shells for each atom."""

    slater: Tensor
    """Exponent of Slater function."""

    pqn: Tensor
    """Principal quantum number of each shell"""

    valence: Tensor
    """Whether the shell is part of the valence shell."""

    sto: Literal["gfn1", "gfn2"]
    """STO-NG expansion set for Slater-type orbitals."""

    __slots__ = [
        "numbers",
        "unique",
        "meta",
        "ihelp",
        "ngauss",
        "shells",
        "slater",
        "pqn",
        "valence",
        "sto",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        resolved_dtype = (
            par.dtype
            if dtype is None and isinstance(par, ParamModule)
            else dtype
        )
        super().__init__(
            numbers.device if device is None else device, resolved_dtype
        )
        self.numbers = numbers
        self.unique: Tensor = torch.unique(numbers)
        self.meta = (
            None
            if isinstance(par, ParamModule) and par.is_none("meta")
            else par.meta
        )
        self.ihelp = ihelp

        if not isinstance(par, ParamModule):
            par = ParamModule(par, **self.dd)

        if par.is_none("hamiltonian.xtb.sto"):
            self.sto: Literal["gfn1", "gfn2"] = "gfn2"
        else:
            self.sto = par.get("hamiltonian.xtb.sto")

        # Integer data types for `ngauss` and `pqn`
        self.ngauss = par.get_elem_param(
            self.unique, "ngauss", dtype=DEFAULT_BASIS_INT
        )
        self.pqn = par.get_elem_pqn(self.unique)

        self.slater = par.get_elem_param(self.unique, "slater")
        self.valence = par.get_elem_valence(self.unique)
        self.shells = par.get_elem_shells(self.unique)

        # When a CUDA device is used, the parametrization remains on CPU, even
        # when the libcint library is requested via the `force_cpu_for_libcint`
        # flag. This flag only moves the positions and the IndexHelper (see
        # `setup_integral_evaluation`). We are moving here, because we do not
        # want to move the whole parametrization back and forth.
        self.ngauss = self.ngauss.to(device=self.device)
        self.slater = self.slater.to(device=self.device, dtype=self.dtype)
        self.pqn = self.pqn.to(device=self.device)
        self.valence = self.valence.to(device=self.device)

    @override
    def to(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        obj = super().to(device=device, dtype=dtype)
        if device is not None:
            obj.ihelp = self.ihelp.to(device=obj.device)
        return obj

    def create_cgtos(self) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """
        Expand Slater basis parameters into normalized contracted Gaussians.

        Iterates unique elements and their unique shells in ``IndexHelper``
        order. Within each unique element, secondary shells marked with
        ``valence == False`` (such as hydrogen's polarizing ``2s`` shell in
        GFN0-xTB and GFN1-xTB) are Gram-Schmidt orthonormalized against the
        primary valence shell of the same angular momentum on that element,
        independent of parametrization metadata names. Padding or zero-shell
        placeholder slots in ``unique_angular`` emit 1D zero tensors so unique
        shell indices remain aligned with ``IndexHelper.unique_angular``.

        Returns
        -------
        tuple[tuple[Tensor, ...], tuple[Tensor, ...]]
            Per-unique-shell primitive Gaussian exponents and contraction
            coefficients.
        """
        unique_numbers = _to_int_list(self.unique)
        shell_counts = _to_int_list(self.ihelp.ushells_per_unique)
        angular = _to_int_list(self.ihelp.unique_angular)
        valence_flags = [bool(v) for v in _to_int_list(self.valence)]
        n_slots = len(angular)

        alphas: list[Tensor] = []
        coeffs: list[Tensor] = []
        shell = 0

        for number, count in zip(unique_numbers, shell_counts):
            if count == 0:
                if shell < n_slots and angular[shell] < 0:
                    zero = torch.zeros(
                        (1,), dtype=self.slater.dtype, device=self.slater.device
                    )
                    alphas.append(zero)
                    coeffs.append(zero)
                    shell += 1
                continue

            valence_by_angmom: dict[int, int] = {}
            for _ in range(count):
                angmom = angular[shell]
                if number <= 0 or angmom < 0:
                    zero = torch.zeros(
                        (1,), dtype=self.slater.dtype, device=self.slater.device
                    )
                    alphas.append(zero)
                    coeffs.append(zero)
                    shell += 1
                    continue

                alpha, coeff = slater_to_gauss(
                    self.ngauss[shell],
                    self.pqn[shell],
                    self.ihelp.unique_angular[shell],
                    self.slater[shell],
                    sto=self.sto,
                )
                if valence_flags[shell]:
                    valence_by_angmom[angmom] = len(alphas)
                else:
                    ref_shell = valence_by_angmom.get(angmom)
                    if ref_shell is None:
                        raise ValueError(
                            f"Non-valence shell (Z={number}, l={angmom}) has "
                            "no preceding valence shell of the same angular "
                            "momentum."
                        )
                    if angmom != 0:
                        raise NotImplementedError(
                            "Gram-Schmidt orthogonalization is only "
                            f"implemented for s shells (l=0), got l={angmom}."
                        )
                    alpha, coeff = orthogonalize(
                        (alphas[ref_shell], alpha),
                        (coeffs[ref_shell], coeff),
                    )

                alphas.append(alpha)
                coeffs.append(coeff)
                shell += 1

        return tuple(alphas), tuple(coeffs)

    def to_bse(
        self,
        qcformat: Literal["gaussian94", "nwchem"] = "nwchem",
        save: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
        with_header: bool = False,
    ) -> str:
        """
        Convert the basis set to a format suitable for basis set exchange.

        Parameters
        ----------
        qcformat : Literal["gaussian94", "nwchem"], optional
            Format of the basis set. Defaults to ``"nwchem"``.
        save : bool, optional
            Whether to save the basis set to a file. Defaults to ``False``.
        overwrite : bool, optional
            Whether to overwrite existing files. Defaults to ``False``.
        verbose : bool, optional
            Whether to print the basis set to the console.
            Defaults to ``False``.
        with_header : bool, optional
            Whether to include the header in the basis set.
            Defaults to ``False``.

        Returns
        -------
        str
            Basis set in the specified format.

        Raises
        ------
        RuntimeError
            If no meta data is found in the parametrization, or if the meta
            data is incomplete (name missing), or if the basis set is batched.
        ValueError
            If no atoms for basis set printout are found, or if the basis set
            format is not supported.
        """
        if self.meta is None:
            raise RuntimeError("No meta data found in the parametrization.")

        if self.unique.ndim > 1:
            raise RuntimeError("Basis set printing does not work batched.")

        if len(self.unique) == 0:
            raise ValueError("No atoms for basis set printout found.")

        allowed_formats = ("gaussian94", "nwchem")
        if qcformat not in allowed_formats:
            raise ValueError(
                f"Basis set format '{qcformat}' not supported. "
                f"Available options are: {allowed_formats}."
            )

        header = ""
        if with_header is True:
            meta_name = getattr(self.meta, "name", None)
            meta_version = getattr(self.meta, "version", None)
            line = 70 * "-"
            header = (
                f"!{line}\n"
                "! Basis Set Exchange\n"
                "! Version v0.9\n"
                "! https://www.basissetexchange.org\n"
                f"!{line}\n"
                f"!   Basis set: {meta_name}\n"
                f"! Description: Orthonormalized {meta_name} Basis\n"
                "!        Role: orbital\n"
                f"!     Version: {meta_version}\n"
                f"!{line}\n\n\n"
            )

        if qcformat == "nwchem":
            header += 'BASIS "ao basis" SPHERICAL PRINT\n'

        alphas, coeffs = self.create_cgtos()
        unique_numbers = _to_int_list(self.unique)
        shell_counts = _to_int_list(self.ihelp.ushells_per_unique)
        angular = _to_int_list(self.ihelp.unique_angular)
        n_slots = len(angular)

        s = 0
        fulltxt = ""

        for number, shells in zip(unique_numbers, shell_counts):
            if number <= 0:
                if shells == 0 and s < n_slots and angular[s] < 0:
                    s += 1
                else:
                    s += shells
                continue

            txt = header
            symbol = pse.Z2S[number]
            if symbol not in self.shells:
                raise ValueError(
                    f"Element '{symbol}' not found in the basis set."
                )

            if qcformat == "gaussian94":
                txt += f"{symbol}\n"
            elif qcformat == "nwchem":
                f = format_contraction(
                    self.shells[symbol], self.ngauss[s : s + shells]
                )
                txt += f"#BASIS SET: {f}\n"

            for _ in range(shells):
                alpha = alphas[s]
                coeff = coeffs[s]
                l = angular2label[angular[s]]
                if qcformat == "gaussian94":
                    txt += f"{l}    {len(alpha)}    1.00\n"
                elif qcformat == "nwchem":
                    txt += f"{symbol}    {l}\n"

                for a, c in zip(alpha, coeff):
                    txt += f"      {a}      {c}\n"

                s += 1

            if qcformat == "gaussian94":
                txt += "****\n"
            elif qcformat == "nwchem":
                txt += "END\n"

            if save is True:
                meta_name = getattr(self.meta, "name", None)
                if meta_name is None:
                    raise RuntimeError("Meta data incomplete (name missing).")

                target = f"mol/external/basis/{meta_name.casefold()}"
                dpath = Path(__file__).parents[1] / target
                Path(dpath).mkdir(parents=True, exist_ok=True)

                fpath = f"{number:02d}.nwchem"
                file_path = Path(dpath) / fpath

                if overwrite is False and file_path.exists():
                    print(
                        f"The file '{fpath}' already exists in the "
                        f"directory '{dpath}'. It will not be overwritten."
                    )
                    continue

                with open(file_path, "w", encoding="utf8") as file:
                    file.write(txt)

            if verbose is True:
                print(txt)

            fulltxt += txt

        if not fulltxt and not save:
            raise ValueError("No atoms for basis set printout found.")

        return fulltxt

    def create_libcint(
        self, positions: Tensor, mask: Tensor | None = None
    ) -> tuple[libcint.Basis, libcint.Structure]:
        """Create call-local libcint basis and structure for positions."""
        if self.ihelp.batch_mode != 0 or self.unique.ndim > 1:
            raise NotImplementedError(
                "Libcint Basis conversion requires a single Structure."
            )

        # pylint: disable=import-outside-toplevel
        from dxtb._src.integral.driver.libcint.driver import setup_libcint

        return setup_libcint(self, force_cpu=False).create_libcint(
            positions, mask=mask
        )


def format_contraction(shells: list[str], ngauss: Tensor) -> str:
    """
    Format the contraction of the basis set.

    Parameters
    ----------
    shells : list[str]
        List of shells.
    ngauss : Tensor
        Number of Gaussian primitives for each shell.

    Returns
    -------
    str
        Formatted contraction string.

    Note
    ----
    For H in GFN1-xTB, the contraction will be (7s) -> [2s]. However, the
    exported basis set will included the orthogonalized 2s shell, which would
    make the contraction (11s) -> [2s].
    """
    type_order = ["s", "p", "d", "f", "g"]

    # 1) tally up primitives by angular momentum
    prim_counts: dict[str, int] = {}
    for shell, n in zip(shells, ngauss):
        l = shell[-1]  # last char: 's','p',...
        prim_counts[l] = prim_counts.get(l, 0) + int(n)

    # 2) build the "(...)" part for nonzero primitives
    prim_parts = [
        f"{prim_counts[l]}{l}" for l in type_order if prim_counts.get(l, 0) > 0
    ]

    # 3) build the "[...]" part for nonzero contracted fns
    counts = {t: 0 for t in type_order}

    for sh in shells:
        shell_type = sh[-1]
        if shell_type not in counts:
            raise ValueError(f"Unknown shell type '{shell_type}'.")

        counts[shell_type] += 1

    cont_parts = [f"{counts[l]}{l}" for l in type_order if counts.get(l, 0) > 0]

    return f"({','.join(prim_parts)}) -> [{','.join(cont_parts)}]"
