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
"""
xTB Hamiltonians: GFN0-xTB
==========================

The GFN0-xTB Hamiltonian.
"""

from __future__ import annotations

from functools import partial

import torch
from tad_mctc import Structure
from tad_mctc.convert import any_to_tensor
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper
from dxtb._src.components.interactions import Potential
from dxtb._src.ncoord import coordination_number, erf_count
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import Any, Self, Tensor, override
from dxtb._src.utils.tensors import structure_charge

from .base import PAD, BaseHamiltonian
from .h0 import build_hcore, gather_hscale

__all__ = ["GFN0Hamiltonian"]


class GFN0Hamiltonian(BaseHamiltonian):
    """Charge- and coordination-dependent GFN0-xTB Hamiltonian."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(par, ParamModule):
            par = ParamModule(par, device=device, dtype=dtype)

        super().__init__(
            numbers, par, ihelp, device, dtype, setup=kwargs.get("setup")
        )

        setup = self.setup
        assert setup.kq is not None and setup.kqat is not None
        assert setup.h0rad is not None and setup.kdiff is not None
        assert setup.enshell is not None and setup.enscale4 is not None
        assert setup.eeq_model is not None and setup.cn_radii is not None
        self.kq = setup.kq
        self.kqat = setup.kqat
        self.h0rad = setup.h0rad
        self.kdiff = setup.kdiff
        self.enshell = setup.enshell
        self.enscale4 = setup.enscale4
        self.eeq_model = setup.eeq_model
        self.cn_radii = setup.cn_radii
        self.cn_cutoff = setup.cn_cutoff
        self.cn_max = setup.cn_max
        self.cn_kcn = setup.cn_kcn
        self._set_cn_callable()

    def _set_cn_callable(self) -> None:
        """Configure the local GFN0 coordination-number callable."""
        self.cn = partial(
            coordination_number,
            counting_function=erf_count,
            rcov=self.cn_radii,
            cutoff=self.cn_cutoff,
            cn_max=self.cn_max,
            kcn=self.cn_kcn,
        )

    @override
    def _clone_tensorlike(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Clone GFN0-owned tensors and its EEQ model with the base fields."""
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        new = super()._clone_tensorlike(device=device, dtype=dtype)

        # TensorLike treats every plain tensor as floating-point. Restore the
        # integer and boolean indexing tensors with only a device conversion.
        new.numbers = self.numbers.to(device=target_device)
        new.unique = self.unique.to(device=target_device)
        new.valence = self.valence.to(device=target_device)
        new.ihelp = self.ihelp.to(device=target_device)
        if self.matrix is not None:
            new.matrix = self.matrix.to(
                device=target_device, dtype=target_dtype
            )

        owned_tensors = (
            "kq",
            "kqat",
            "h0rad",
            "kdiff",
            "enshell",
            "enscale4",
            "cn_radii",
            "cn_cutoff",
            "cn_max",
            "cn_kcn",
        )
        for name in owned_tensors:
            value = getattr(self, name)
            setattr(
                new,
                name,
                value.to(device=target_device, dtype=target_dtype),
            )

        model = self.eeq_model
        new.eeq_model = EEQModel(
            chi=model.chi.to(device=target_device, dtype=target_dtype),
            kcn=model.kcn.to(device=target_device, dtype=target_dtype),
            eta=model.eta.to(device=target_device, dtype=target_dtype),
            rad=model.rad.to(device=target_device, dtype=target_dtype),
        )
        new._set_cn_callable()
        return new

    @override
    def _get_elem_valence(self, par: ParamModule) -> Tensor:
        """Return the GFN0 valence-shell mask, including duplicate H shells."""
        return par.get_elem_valence(self.unique, pad_val=PAD)

    @override
    def _get_hscale(self, par: ParamModule) -> Tensor:
        if par.is_none("hamiltonian"):
            raise RuntimeError("No Hamiltonian specified.")
        return gather_hscale(
            "gfn0", self.unique, self.ihelp, self.valence, par, self.dd
        )

    def get_coordination_number(self, positions: Tensor) -> Tensor:
        """Evaluate capped GFN0 coordination numbers."""
        if self.cn is None:  # pragma: no cover - initialized above
            raise RuntimeError("GFN0 coordination-number function is missing.")
        return self.cn(self.numbers, positions)

    def get_eeq_charges(
        self, positions: Tensor, charge: Tensor | float | int, cn: Tensor
    ) -> Tensor:
        """Solve the coordinate-local main GFN0 EEQ model."""
        total_charge = any_to_tensor(charge, **self.dd)
        structure = Structure(
            numbers=self.numbers,
            positions=positions,
            charge=structure_charge(total_charge, self.numbers),
        )
        charges = self.eeq_model.solve(structure, cn)
        assert isinstance(charges, Tensor)
        return charges

    def get_selfenergy(self, cn: Tensor, charges: Tensor) -> Tensor:
        """Return environment-shifted self-energies for every atom shell."""
        eps0 = self.ihelp.spread_ushell_to_shell(self.selfenergy)
        kcn = self.ihelp.spread_ushell_to_shell(self.kcn)
        kq = self.ihelp.spread_ushell_to_shell(self.kq)
        shell_cn = self.ihelp.spread_atom_to_shell(cn)
        shell_q = self.ihelp.spread_atom_to_shell(charges)

        kqat = self.ihelp.spread_uspecies_to_atom(self.kqat)
        shell_q2 = self.ihelp.spread_atom_to_shell(kqat * charges**2)
        return eps0 - kcn * shell_cn - kq * shell_q - shell_q2

    @override
    def get_gradient(
        self,
        positions: Tensor,
        overlap: Tensor,
        doverlap: Tensor,
        pmat: Tensor,
        wmat: Tensor,
        pot: Potential,
        cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "GFN0 analytical gradient is not implemented."
        )

    @override
    def build(
        self,
        positions: Tensor,
        overlap: Tensor | None = None,
        charge: Tensor | float | int | None = None,
    ) -> Tensor:
        """Build the GFN0 H0 matrix from local CN and main EEQ charges."""
        h0, _ = build_hcore(self.setup, positions, overlap, charge)
        self.matrix = h0
        return h0
