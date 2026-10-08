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

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.data import pse

from dxtb._src.constants.defaults import DEFAULT_BASIS_INT
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import Literal, Tensor, TensorLike

from .indexhelper import IndexHelper
from .ortho import orthogonalize
from .slater import slater_to_gauss

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

__all__ = ["Basis", "BasisSetup"]


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
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.numbers = numbers
        self.unique: Tensor = torch.unique(numbers)
        self.meta = par.meta
        self.ihelp = ihelp

        if not isinstance(par, ParamModule):
            par = ParamModule(par, **self.dd)

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
        # `DriverManager.create_driver`). We are moving here, because we do not
        # want to move the whole parametrization back and forth.
        self.ngauss = self.ngauss.to(device=self.device)
        self.slater = self.slater.to(device=self.device)
        self.pqn = self.pqn.to(device=self.device)
        self.valence = self.valence.to(device=self.device)

    def create_cgtos(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Create contracted Gaussian type orbitals from parametrization.

        Returns
        -------
        tuple[list[Tensor], list[Tensor]]
            List of primitive Gaussian exponents and contraction coefficients
            for the orthonormalized basis functions for each shell.
        """
        coeffs = []
        alphas = []

        # maybe this can be batched too, but this loop is rather small
        # so it is probably not worth it
        for i in range(self.ihelp.unique_angular.size(0)):
            alpha, coeff = slater_to_gauss(
                self.ngauss[i],
                self.pqn[i],
                self.ihelp.unique_angular[i],
                self.slater[i],
            )

            # NOTE:
            # This only works for GFN0/GFN1 with H being the only element
            # with a non-valence shell. Otherwise the generation of
            # self.valence is not correct.
            # The correct way would be a map to show which orbital needs
            # to be orthogonalized w.r.t. another one.
            # Example: Si with 3s, 3p, 3d, 4p
            # angular = [0, 1, 2, 1]; ortho = [None, None, None, 1]
            # GFN0 and GFN1 both contain this duplicate hydrogen s shell.
            if self.meta is not None and self.meta.name is not None:
                method = self.meta.name.casefold()
                if "gfn0" in method or "gfn1" in method:
                    if self.valence[i].item() is False:
                        alpha, coeff = orthogonalize(
                            (alphas[i - 1], alpha),
                            (coeffs[i - 1], coeff),
                        )

            alphas.append(alpha)
            coeffs.append(coeff)

        return alphas, coeffs

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
            l = 70 * "-"
            header = (
                f"!{l}\n"
                "! Basis Set Exchange\n"
                "! Version v0.9\n"
                "! https://www.basissetexchange.org\n"
                f"!{l}\n"
                f"!   Basis set: {self.meta.name}\n"
                f"! Description: Orthonormalized {self.meta.name} Basis\n"
                "!        Role: orbital\n"
                f"!     Version: {self.meta.version}\n"
                f"!{l}\n\n\n"
            )

        coeffs = []
        alphas = []
        s = 0
        fulltxt = ""

        if qcformat == "nwchem":
            header += 'BASIS "ao basis" SPHERICAL PRINT\n'

        for i, number in enumerate(self.unique.tolist()):
            txt = header

            symbol = pse.Z2S[number]
            if symbol not in self.shells:
                raise ValueError(
                    f"Element '{symbol}' not found in the basis set."
                )

            if qcformat == "gaussian94":
                txt += f"{symbol}\n"
            elif qcformat == "nwchem":
                f = format_contraction(self.shells[symbol], self.ngauss)
                txt += f"#BASIS SET: {f}\n"

            shells = self.ihelp.shells_per_atom[i]
            for _ in range(shells):
                alpha, coeff = slater_to_gauss(
                    self.ngauss[s],
                    self.pqn[s],
                    self.ihelp.angular[s],
                    self.slater[s],
                )
                if self.valence[s].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[s - 1], alpha),
                        (coeffs[s - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)

                l = angular2label[self.ihelp.angular.tolist()[s]]
                if qcformat == "gaussian94":
                    txt += f"{l}    {len(alpha)}    1.00\n"
                elif qcformat == "nwchem":
                    txt += f"{pse.Z2S[number]}    {l}\n"

                # write exponents and coefficients
                for a, c in zip(alpha, coeff):
                    txt += f"      {a}      {c}\n"

                s += 1

            # final separator in gaussian94 format
            if qcformat == "gaussian94":
                txt += "****\n"
            elif qcformat == "nwchem":
                txt += f"END\n"

            # always save to src/dxtb/mol/external/basis
            if save is True:
                if self.meta.name is None:
                    raise RuntimeError("Meta data incomplete (name missing).")

                # Create the directory if it doesn't exist
                target = f"mol/external/basis/{self.meta.name.casefold()}"
                dpath = Path(__file__).parents[1] / target
                Path(dpath).mkdir(parents=True, exist_ok=True)

                # Create the file path
                fpath = f"{number:02d}.nwchem"
                file_path = Path(dpath) / fpath

                # Check if the file already exists
                if overwrite is False:
                    if file_path.exists():
                        print(
                            f"The file '{fpath}' already exists in the "
                            f"directory '{dpath}'. It will not be overwritten."
                        )
                        continue

                # Save the file
                with open(file_path, "w", encoding="utf8") as file:
                    file.write(txt)

            if verbose is True:
                print(txt)

            fulltxt += txt

        return fulltxt

    def create_libcint(
        self, positions: Tensor, mask: Tensor | None = None
    ) -> list[libcint.AtomCGTOBasis] | list[list[libcint.AtomCGTOBasis]]:
        """Create the geometry-local libcint basis wrappers."""
        return self.setup_data().create_libcint(positions, mask=mask)

    def setup_data(self) -> BasisSetup:
        """Return the gathered, geometry-free basis data."""
        return BasisSetup(
            numbers=self.numbers,
            unique=self.unique,
            ihelp=self.ihelp,
            ngauss=self.ngauss,
            pqn=self.pqn,
            slater=self.slater,
            valence=self.valence,
        )

    def create_libcint_cgtos(
        self,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Create geometry-independent Gaussian basis data for libcint."""
        return self.setup_data().create_libcint_cgtos()


@dataclass(frozen=True, eq=False)
class BasisSetup:
    """Narrow composition data required to create libcint basis wrappers."""

    numbers: Tensor
    unique: Tensor
    ihelp: IndexHelper
    ngauss: Tensor
    pqn: Tensor
    slater: Tensor
    valence: Tensor

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BasisSetup:
        """Return converted gathered basis data."""
        return replace(
            self,
            numbers=self.numbers.to(device=device),
            unique=self.unique.to(device=device),
            ihelp=self.ihelp.to(device=device),
            ngauss=self.ngauss.to(device=device),
            pqn=self.pqn.to(device=device),
            slater=self.slater.to(device=device, dtype=dtype),
            valence=self.valence.to(device=device),
        )

    def create_libcint(
        self, positions: Tensor, mask: Tensor | None = None
    ) -> list[libcint.AtomCGTOBasis] | list[list[libcint.AtomCGTOBasis]]:
        """Create libcint wrappers for the supplied current geometry."""
        if self.unique.ndim > 1:
            raise NotImplementedError("Batch mode not implemented.")

        from dxtb._src.exlibs import libcint

        alphas, coeffs = self.create_libcint_cgtos()
        if self.ihelp.batch_mode == 0:
            atom_basis: list[libcint.AtomCGTOBasis] = []
            shell = 0
            for i, number in enumerate(self.numbers):
                bases = []
                for _ in range(self.ihelp.shells_per_atom[i]):
                    index = self.ihelp.shells_to_ushell[shell]
                    bases.append(
                        libcint.CGTOBasis(
                            angmom=int(self.ihelp.angular[shell]),
                            alphas=alphas[index],
                            coeffs=coeffs[index],
                            normalized=True,
                        )
                    )
                    shell += 1
                atom_basis.append(
                    libcint.AtomCGTOBasis(
                        atomz=number, bases=bases, pos=positions[i, :]
                    )
                )
            return atom_basis

        batches: list[list[libcint.AtomCGTOBasis]] = []
        for batch_index in range(self.numbers.shape[0]):
            shell = 0
            atom_basis = []
            shell_indices = self.ihelp.shells_to_ushell[batch_index]
            shells_per_atom = self.ihelp.shells_per_atom[batch_index]
            angular = tensor_to_numpy(self.ihelp.angular[batch_index])
            batch_positions = positions[batch_index]

            for atom_index, number in enumerate(self.numbers[batch_index]):
                if number == 0:
                    continue
                bases = []
                for _ in range(shells_per_atom[atom_index]):
                    index = shell_indices[shell]
                    bases.append(
                        libcint.CGTOBasis(
                            angmom=angular[shell],
                            alphas=alphas[index],
                            coeffs=coeffs[index],
                            normalized=True,
                        )
                    )
                    shell += 1

                if self.ihelp.batch_mode == 1:
                    if mask is None:
                        from tad_mctc.batch import deflate

                        atom_positions = deflate(batch_positions, value=float("nan"))
                    else:
                        atom_positions = torch.masked_select(
                            batch_positions, mask[batch_index]
                        ).reshape((-1, 3))
                elif self.ihelp.batch_mode == 2:
                    atom_positions = batch_positions
                else:
                    raise ValueError(
                        f"Unknown batch mode '{self.ihelp.batch_mode}'."
                    )

                atom_basis.append(
                    libcint.AtomCGTOBasis(
                        atomz=number,
                        bases=bases,
                        pos=atom_positions[atom_index, :],
                    )
                )
            batches.append(atom_basis)
        return batches

    def create_libcint_cgtos(
        self,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Create parameter-differentiable Gaussian contractions."""
        alphas: list[Tensor] = []
        coeffs: list[Tensor] = []
        shell = 0

        for unique_index in range(self.unique.size(0)):
            shell_count = self.ihelp.ushells_per_unique[unique_index]
            if shell_count == 0:
                zero = torch.tensor(
                    0.0, dtype=self.slater.dtype, device=self.slater.device
                )
                alphas.append(zero)
                coeffs.append(zero)
                shell += 1
                continue

            for _ in range(shell_count):
                alpha, coeff = slater_to_gauss(
                    self.ngauss[shell],
                    self.pqn[shell],
                    self.ihelp.unique_angular[shell],
                    self.slater[shell],
                )
                if self.valence[shell].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[shell - 1], alpha),
                        (coeffs[shell - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)
                shell += 1
        return tuple(alphas), tuple(coeffs)


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
    prim_counts = {}
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
