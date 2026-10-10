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
Calculators: Analytical
=======================

Calculator for the extended tight-binding model with analytical gradients.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb import OutputHandler
from dxtb import integrals as ints
from dxtb import labels
from dxtb._src import ncoord, scf
from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.constants import defaults
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor

from ..result import Result
from . import decorators as cdec
from .energy import _UNSET, EnergyCalculator

__all__ = ["AnalyticalCalculator"]


class AnalyticalCalculator(EnergyCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides analytical formulas and/or gradients for certain
    properties.
    """

    implemented_properties = EnergyCalculator.implemented_properties + [
        "forces",
        "dipole",
        "quadrupole",
    ]
    """Names of implemented methods of the Calculator."""

    @cdec.requires_positions_grad
    def forces_analytical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None | object = _UNSET,
        field_grad: Tensor | None | object = _UNSET,
        **kwargs: Any,
    ) -> Tensor:
        """Calculate analytical nuclear forces from one local Result."""
        field, field_grad = self._resolve_fields(field, field_grad)
        if field_grad is not None:
            raise NotImplementedError(
                "Analytical forces for an electric field gradient are not "
                "implemented; use autograd forces."
            )
        result = self.singlepoint(
            positions,
            chrg,
            spin,
            field=field,
            field_grad=field_grad,
            **kwargs,
        )
        total_grad = torch.zeros(positions.shape, **self.dd)

        if result.classical:
            cgradients = self.classicals.get_gradient(
                dict(result.classical), positions
            )
            total_grad += torch.stack(tuple(cgradients.values())).sum(0)

        if {"all", "scf"} & set(self.opts.exclude):
            return -total_grad

        timer.start("Interaction Cache", parent_uid="SCF")
        icaches = self.interactions.get_cache(
            numbers=self.numbers, positions=positions, ihelp=self.ihelp
        )
        timer.stop("Interaction Cache")

        charges = result.charges
        if charges is None:
            raise RuntimeError("SCF charges are missing from the Result.")
        if self.interactions.components:
            # The legacy interaction gradient reads the three charge tensors
            # from the mutable SCF container shape. Keep this adapter local to
            # the call; Result retains only its immutable output value.
            legacy_charges = Charges(
                mono=charges.mono,
                dipole=charges.dipole,
                quad=charges.quadrupole,
                batch_mode=self.ihelp.batch_mode,
            )
            total_grad += self.interactions.get_gradient(
                legacy_charges, positions, icaches, self.ihelp
            )

        if field is not None and result.charges is not None:
            qat = self.ihelp.reduce_orbital_to_atom(
                result.charges.mono.detach()
            )
            total_grad -= qat.unsqueeze(-1) * field

        if result.overlap is None or result.density is None:
            raise RuntimeError(
                "SCF overlap or density is missing from the Result."
            )
        if result.coefficients is None or result.emo is None:
            raise RuntimeError("SCF orbital data is missing from the Result.")
        if result.occupation is None or result.potential is None:
            raise RuntimeError(
                "SCF occupation or potential is missing from the Result."
            )
        if self.integrals.hcore is None:
            raise RuntimeError("Legacy H0 gradient adapter is not initialized.")

        legacy_potential = Potential(
            mono=result.potential.mono,
            dipole=result.potential.dipole,
            quad=result.potential.quadrupole,
            batch_mode=self.ihelp.batch_mode,
        )

        overlap_grad = self.integrals.grad_overlap(positions)
        wmat = scf.get_density(
            result.coefficients,
            result.occupation.sum(-2),
            emo=result.emo,
        )
        cn = ncoord.cn_d3(self.numbers, positions)
        dedcn, dedr = self.integrals.hcore.get_gradient(
            positions,
            result.overlap,
            overlap_grad,
            result.density,
            wmat,
            legacy_potential,
            cn,
        )
        dcndr = ncoord.cn_d3_gradient(self.numbers, positions)
        total_grad += dedr + ncoord.get_dcn(dcndr, dedcn)

        if (
            result.potential.dipole is not None
            and result.dipole_integrals is not None
        ):
            vdp = self.ihelp.spread_atom_to_orbital(
                result.potential.dipole.detach(), dim=-2, extra=True
            )
            edp = -einsum(
                "...ij,...kij,...jk->...",
                result.density.detach(),
                result.dipole_integrals,
                vdp,
            )
            (dipole_grad,) = torch.autograd.grad(
                edp.sum(),
                positions,
                retain_graph=True,
                create_graph=torch.is_grad_enabled(),
            )
            total_grad += dipole_grad

        return -total_grad

    def dipole_analytical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *_,  # absorb stuff
        field: Tensor | None | object = _UNSET,
        field_grad: Tensor | None | object = _UNSET,
        **kwargs: Any,
    ) -> Tensor:
        r"""
        Analytically calculate the electric dipole moment :math:`\mu`.


        .. math::

            \mu = \dfrac{\partial E}{\partial F}

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        Tensor
            Electric dipole moment of shape `(..., 3)`.
        """
        # Keep the SCF and integral outputs local to this calculation.
        result = self.singlepoint(
            positions,
            chrg,
            spin,
            field=field,
            field_grad=field_grad,
            **kwargs,
        )
        if result.dipole_integrals is None:
            raise RuntimeError(
                "Dipole moment requires a dipole integral. They should "
                "be available when the System is constructed with dipole "
                "integrals."
            )

        # pylint: disable=import-outside-toplevel
        from ..properties.moments.dip import dipole

        qat = self.ihelp.reduce_orbital_to_atom(result.charges.mono)
        dip = dipole(qat, positions, result.density, result.dipole_integrals)
        return dip

    def quadrupole_analytical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *_,  # absorb stuff
        **kwargs: Any,
    ) -> Tensor:
        r"""
        Analytically calculate the traceless electric quadrupole moment
        :math:`\Theta` (nuclear and electronic contributions), following the
        `tblite` implementation.

        The atom-resolved monopole, dipole and quadrupole populations that the
        SCF builds from the density matrix and the (atom-centered) integrals
        are combined with the nuclear positions.

        Requires the quadrupole integral, i.e., an integral level of at least
        ``labels.INTLEVEL_QUADRUPOLE`` (the default for GFN2-xTB).

        .. note::

            This reproduces `tblite` element by element. The off-diagonal
            elements of the packed result mix two scalings (``3 Q_ij`` for
            the point-charge/dipole part, ``1.5 Q_ij`` for the atomic
            quadrupoles), so they are not rotation-covariant, whereas the
            diagonal elements are. The derivative of the energy with respect
            to the electric field gradient
            (:meth:`~dxtb.Calculator.quadrupole`,
            :meth:`~dxtb.Calculator.quadrupole_numerical`) is consistent and
            agrees with this result for the diagonal elements.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        Tensor
            Traceless quadrupole moment of shape ``(..., 6)`` in the order
            ``xx, yx, yy, zx, zy, zz``.
        """
        if self.opts.ints.level < labels.INTLEVEL_QUADRUPOLE:
            raise RuntimeError(
                "The quadrupole moment requires the quadrupole integral, i.e., "
                f"an integral level of at least {labels.INTLEVEL_QUADRUPOLE} "
                f"(current: {self.opts.ints.level}). GFN2-xTB does this by "
                "default."
            )

        result = self.singlepoint(positions, chrg, spin, **kwargs)

        charges = result.charges
        if charges.dipole is None or charges.quadrupole is None:
            raise RuntimeError(
                "The SCF did not produce atom-resolved dipole and quadrupole "
                "populations. This is probably a bug."
            )

        # pylint: disable=import-outside-toplevel
        from ..properties.moments.quad import quadrupole

        qat = self.ihelp.reduce_orbital_to_atom(charges.mono)
        return quadrupole(qat, charges.dipole, charges.quadrupole, positions)

    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Calculate requested properties and return them explicitly."""
        values = EnergyCalculator.calculate(
            self, properties, positions, chrg, spin, **kwargs
        )
        if "forces" in properties:
            force_kwargs = dict(kwargs)
            force_kwargs.pop("grad_mode", None)
            values["forces"] = self.forces_analytical(
                positions, chrg, spin, **force_kwargs
            )
        if "dipole" in properties:
            values["dipole"] = self.dipole_analytical(
                positions, chrg, spin, **kwargs
            )
        if "quadrupole" in properties:
            values["quadrupole"] = self.quadrupole_analytical(
                positions, chrg, spin, **kwargs
            )
        return values
