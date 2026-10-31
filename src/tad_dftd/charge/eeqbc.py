# This file is part of tad-dftd.
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
Bond-capacitor electronegativity equilibration charge model (EEQ_BC).

Port of ``multicharge_model_eeqbc`` (``src/multicharge/model/eeqbc.f90``) and
``multicharge_param::new_eeqbc2025_model``.
"""
from __future__ import annotations

import math
from typing import Literal, overload

import torch
from tad_mctc import storch
from tad_mctc.batch import real_atoms, real_pairs
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor import wignerseitz_cell, wrap_to_central_cell
from tad_mctc.ncoord.common import CNModel
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import DD, Tensor, get_default_dtype

from ..ncoord import cn_eeqbc, cn_eeqbc_en
from . import eeqbc_param

__all__ = ["EEQBCModel", "get_eeqbc_charges"]

_REP = (2, 2, 2)
_IMAGE_REP = (1, 1, 1)
_EPS = math.sqrt(torch.finfo(torch.float64).eps)
_SQRT2PI = math.sqrt(2.0 / math.pi)


def _combined_shifts(images: Tensor) -> tuple[Tensor, Tensor]:
    """
    Combine Wigner-Seitz nearest-image shifts ``[-1, 1]^3`` with real-space
    lattice shifts ``[-2, 2]^3``.
    """
    reach = [a + b for a, b in zip(_IMAGE_REP, _REP)]
    axes = [torch.arange(-r, r + 1, device=images.device) for r in reach]
    shifts = torch.cartesian_prod(*axes)  # (ns, 3)

    rep = torch.tensor(_REP, device=images.device)
    reached = ((shifts[None, :, :] - images[:, None, :]).abs() <= rep).all(-1)
    return shifts, reached


class EEQBCModel(Node):
    """
    Bond-capacitor electronegativity equilibration charge model (EEQ_BC 2025).
    """

    chi: Tensor = child()
    eta: Tensor = child()
    rad: Tensor = child()
    kcnchi: Tensor = child()
    kqchi: Tensor = child()
    kqeta: Tensor = child()
    kcnrad: Tensor = child()
    cap: Tensor = child()
    avg_cn: Tensor = child()
    rvdw: Tensor = child()

    kbc: float = context(default=eeqbc_param.KBC_DEFAULT)
    kqeta_pre: float = context(default=eeqbc_param.KQETA_PRE_DEFAULT)
    efield_scale: float = context(default=eeqbc_param.EFIELD_SCALE_DEFAULT)

    cn: CNModel = child(default_factory=lambda: cn_eeqbc)
    cn_en: CNModel = child(default_factory=lambda: cn_eeqbc_en)

    @classmethod
    def param2025(
        cls,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> EEQBCModel:
        """
        Instantiate the standard 2025 EEQ_BC parametrization.
        """
        dt = get_default_dtype() if dtype is None else dtype
        dd: DD = {"device": device, "dtype": dt}
        return cls(
            chi=eeqbc_param.chi.to(**dd),
            eta=eeqbc_param.eta.to(**dd),
            rad=eeqbc_param.rad.to(**dd),
            kcnchi=eeqbc_param.kcnchi.to(**dd),
            kqchi=eeqbc_param.kqchi.to(**dd),
            kqeta=eeqbc_param.kqeta.to(**dd),
            kcnrad=eeqbc_param.kcnrad.to(**dd),
            cap=eeqbc_param.cap.to(**dd),
            avg_cn=eeqbc_param.avg_cn.to(**dd),
            rvdw=eeqbc_param.get_eeqbc_rvdw(device=device, dtype=dt),
        )

    @overload
    def __call__(
        self,
        structure: Structure,
        *,
        return_energy: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def __call__(
        self,
        structure: Structure,
        *,
        return_energy: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def __call__(
        self,
        structure: Structure,
        *,
        return_energy: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Evaluate partial charges (and optional electrostatic energies) for
        ``structure``.
        """
        cn = self.cn(structure)
        qloc = self.cn_en(structure)
        return self.solve(structure, cn, qloc, return_energy=return_energy)

    @overload
    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        qloc: Tensor,
        *,
        return_energy: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        qloc: Tensor,
        *,
        return_energy: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def solve(
        self,
        structure: Structure,
        cn: Tensor,
        qloc: Tensor,
        *,
        return_energy: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Solve the EEQ_BC linear system given coordination numbers ``cn`` and
        electronegativity-weighted coordination numbers ``qloc``.
        """
        numbers = structure.numbers
        positions = structure.positions
        real = real_atoms(numbers)
        pair_mask = real_pairs(numbers, mask_diagonal=True)

        # Safe atomic numbers clamped to [0, 103]
        z_safe = torch.clamp(numbers, min=0, max=eeqbc_param.MAX_ELEM_EEQBC)

        # Element parameters per atom
        chi_i = self.chi[z_safe]
        eta_i = self.eta[z_safe]
        rad_i = self.rad[z_safe]
        kcnchi_i = self.kcnchi[z_safe]
        kqchi_i = self.kqchi[z_safe]
        kqeta_i = self.kqeta[z_safe]
        kcnrad_i = self.kcnrad[z_safe]
        cap_i = self.cap[z_safe]
        avg_cn_i = torch.where(real, self.avg_cn[z_safe], torch.ones_like(cn))

        # Effective charge width of atom i
        norm_cn = torch.where(real, cn / avg_cn_i, torch.zeros_like(cn))
        radi = torch.where(
            real,
            rad_i * torch.exp(-kcnrad_i * norm_cn),
            torch.ones_like(cn),
        )

        # Effective chemical hardness of atom i
        eta_eff = (
            eta_i
            + self.kqeta_pre * torch.tanh(kqeta_i * qloc)
            + _SQRT2PI / radi
        )

        # Unweighted RHS vector xtmp of shape (..., nat + 1)
        xtmp_atoms = torch.where(
            real,
            -chi_i + kcnchi_i * cn + kqchi_i * qloc,
            torch.zeros_like(cn),
        )
        batch_shape = positions.shape[:-2]
        nat = positions.shape[-2]
        if structure.charge is None:
            total_charge = torch.zeros(
                (*batch_shape, 1), device=positions.device, dtype=positions.dtype
            )
        else:
            total_charge = structure.charge.to(
                device=positions.device, dtype=positions.dtype
            ).reshape(*batch_shape, 1)
        xtmp = torch.cat([xtmp_atoms, total_charge], dim=-1)

        # Pairwise vdW radii and capacitance prefactors
        rvdw_ij = self.rvdw[z_safe[..., :, None], z_safe[..., None, :]]
        cap_ij = torch.sqrt(cap_i[..., :, None] * cap_i[..., None, :])

        is_periodic = (
            structure.periodic is not None and bool(structure.periodic.any())
        )
        if is_periodic:
            if not bool(structure.periodic.all()):
                raise NotImplementedError(
                    "EEQ_BC charge model only supports 0D molecules and 3D periodic cells."
                )
            assert structure.lattice is not None
            amat, xvec = self._build_3d(
                positions=positions,
                lattice=structure.lattice,
                periodic=structure.periodic,
                real=real,
                pair_mask=pair_mask,
                radi=radi,
                eta_eff=eta_eff,
                cap_ij=cap_ij,
                rvdw_ij=rvdw_ij,
                xtmp=xtmp,
            )
        else:
            amat, xvec = self._build_0d(
                positions=positions,
                real=real,
                pair_mask=pair_mask,
                radi=radi,
                eta_eff=eta_eff,
                cap_ij=cap_ij,
                rvdw_ij=rvdw_ij,
                xtmp=xtmp,
            )

        vrhs = torch.linalg.solve(amat, xvec.unsqueeze(-1)).squeeze(-1)
        q = torch.where(real, vrhs[..., :nat], torch.zeros_like(cn))

        if return_energy:
            jmat = amat[..., :nat, :nat]
            jq = torch.einsum("...ij,...j->...i", jmat, q)
            energy = q * (0.5 * jq - xvec[..., :nat])
            energy = torch.where(real, energy, torch.zeros_like(energy))
            return q, energy

        return q

    def _build_0d(
        self,
        positions: Tensor,
        real: Tensor,
        pair_mask: Tensor,
        radi: Tensor,
        eta_eff: Tensor,
        cap_ij: Tensor,
        rvdw_ij: Tensor,
        xtmp: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Construct the `(..., nat + 1, nat + 1)` bordered Coulomb matrix ``amat``
        and right-hand side ``xvec`` for a 0D molecule.
        """
        batch_shape = positions.shape[:-2]
        nat = positions.shape[-2]
        dd: DD = {"device": positions.device, "dtype": positions.dtype}

        rij = storch.cdist(positions, positions, p=2)
        rvdw_safe = torch.where(pair_mask & (rvdw_ij > 0.0), rvdw_ij, torch.ones_like(rvdw_ij))
        arg = -self.kbc * (rij - rvdw_safe) / rvdw_safe
        cpair = torch.where(
            pair_mask,
            cap_ij * 0.5 * (1.0 + torch.erf(arg)),
            torch.zeros_like(rij),
        )

        # Maxwell capacitance matrix cmat of shape (..., nat + 1, nat + 1)
        cmat = torch.zeros((*batch_shape, nat + 1, nat + 1), **dd)
        cmat_diag = cpair.sum(dim=-1)
        cmat[..., :nat, :nat] = -cpair + torch.diag_embed(cmat_diag)
        cmat[..., nat, nat] = 1.0

        xvec = torch.einsum("...ij,...j->...i", cmat, xtmp)

        # Coulomb interaction of Gaussian charges
        gam = 1.0 / torch.sqrt(radi[..., :, None] ** 2 + radi[..., None, :] ** 2)
        rij_safe = torch.where(pair_mask, rij, torch.ones_like(rij))
        coulomb_off = torch.where(
            pair_mask,
            (torch.erf(gam * rij_safe) / rij_safe) * cmat[..., :nat, :nat],
            torch.zeros_like(rij),
        )
        coulomb_diag = torch.where(
            real,
            eta_eff * cmat_diag + 1.0,
            torch.ones_like(eta_eff),
        )

        amat = torch.zeros((*batch_shape, nat + 1, nat + 1), **dd)
        amat[..., :nat, :nat] = coulomb_off + torch.diag_embed(coulomb_diag)
        constraint = torch.where(real, torch.ones_like(eta_eff), torch.zeros_like(eta_eff))
        amat[..., nat, :nat] = constraint
        amat[..., :nat, nat] = constraint
        amat[..., nat, nat] = 0.0

        return amat, xvec

    def _build_3d(
        self,
        positions: Tensor,
        lattice: Tensor,
        periodic: Tensor,
        real: Tensor,
        pair_mask: Tensor,
        radi: Tensor,
        eta_eff: Tensor,
        cap_ij: Tensor,
        rvdw_ij: Tensor,
        xtmp: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Construct the `(..., nat + 1, nat + 1)` bordered Coulomb matrix ``amat``
        and right-hand side ``xvec`` for a 3D periodic structure.
        """
        batch_shape = positions.shape[:-2]
        nat = positions.shape[-2]
        dd: DD = {"device": positions.device, "dtype": positions.dtype}

        pos_wrapped, _ = wrap_to_central_cell(
            positions, lattice, periodic.unsqueeze(-2)
        )
        wsc = wignerseitz_cell(pos_wrapped, lattice, periodic)
        shifts, reached = _combined_shifts(wsc.shifts)
        coefficients = wsc.weights @ reached.to(wsc.weights.dtype)  # (..., nat, nat, ns)

        # Pair vectors x_j - x_i
        vectors = pos_wrapped[..., None, :, :] - pos_wrapped[..., :, None, :]
        translations = shifts.to(vectors.dtype) @ lattice  # (..., ns, 3)
        diff = vectors[..., None, :] + translations[..., None, None, :, :]
        dist2 = (diff * diff).sum(dim=-1)

        real_pair_full = real_pairs(
            real.to(torch.long), mask_diagonal=False
        )
        valid = (
            real_pair_full[..., None]
            & (coefficients > 0.0)
            & (dist2 >= _EPS * _EPS)
        )
        dist = torch.where(valid, dist2, torch.ones_like(dist2)).sqrt()

        rvdw_safe = torch.where(
            real_pair_full & (rvdw_ij > 0.0),
            rvdw_ij,
            torch.ones_like(rvdw_ij),
        )
        arg = -self.kbc * (dist - rvdw_safe[..., None]) / rvdw_safe[..., None]
        cpair_terms = cap_ij[..., None] * 0.5 * (1.0 + torch.erf(arg))
        cpair_sum = torch.where(
            valid,
            coefficients * cpair_terms,
            torch.zeros_like(cpair_terms),
        ).sum(dim=-1)  # (..., nat, nat)

        cpair_off = torch.where(pair_mask, cpair_sum, torch.zeros_like(cpair_sum))
        cpair_self_img = torch.where(
            real,
            torch.diagonal(cpair_sum, dim1=-2, dim2=-1),
            torch.zeros_like(eta_eff),
        )
        cmat_diag = cpair_off.sum(dim=-1) + cpair_self_img

        cmat = torch.zeros((*batch_shape, nat + 1, nat + 1), **dd)
        cmat[..., :nat, :nat] = -cpair_off + torch.diag_embed(cmat_diag)
        cmat[..., nat, nat] = 1.0

        # In periodic EEQ_BC (get_xvec lines 310-322), self-image capacitance is
        # subtracted from xvec(iat)
        xvec = torch.einsum("...ij,...j->...i", cmat, xtmp)
        xvec_atoms = xvec[..., :nat] - cpair_self_img * xtmp[..., :nat]
        xvec = torch.cat([xvec_atoms, xvec[..., nat:]], dim=-1)

        # Direct-space Coulomb terms: -cpair * erf(gam * r) / r
        gam = 1.0 / torch.sqrt(radi[..., :, None] ** 2 + radi[..., None, :] ** 2)
        amat_terms = -cpair_terms * torch.erf(gam[..., None] * dist) / dist
        amat_dir = torch.where(
            valid,
            coefficients * amat_terms,
            torch.zeros_like(amat_terms),
        ).sum(dim=-1)

        amat_off = torch.where(pair_mask, amat_dir, torch.zeros_like(amat_dir))
        amat_self_img = torch.where(
            real,
            torch.diagonal(amat_dir, dim1=-2, dim2=-1),
            torch.zeros_like(eta_eff),
        )
        amat_diag = torch.where(
            real,
            amat_self_img + eta_eff * cmat_diag + 1.0,
            torch.ones_like(eta_eff),
        )

        amat = torch.zeros((*batch_shape, nat + 1, nat + 1), **dd)
        amat[..., :nat, :nat] = amat_off + torch.diag_embed(amat_diag)
        constraint = torch.where(real, torch.ones_like(eta_eff), torch.zeros_like(eta_eff))
        amat[..., nat, :nat] = constraint
        amat[..., :nat, nat] = constraint
        amat[..., nat, nat] = 0.0

        return amat, xvec


def get_eeqbc_charges(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor | float | int | None = None,
    lattice: Tensor | None = None,
    periodic: Tensor | None = None,
    cutoff: Tensor | float | None = None,
) -> Tensor:
    """
    Calculate atomic partial charges using the EEQ_BC (2025) charge model.
    """
    if isinstance(charge, (float, int)):
        chrg = torch.tensor(charge, device=positions.device, dtype=positions.dtype)
    elif charge is None:
        chrg = torch.zeros(
            positions.shape[:-2], device=positions.device, dtype=positions.dtype
        )
    else:
        chrg = charge.to(device=positions.device, dtype=positions.dtype)

    struct = Structure(
        numbers=numbers,
        positions=positions,
        charge=chrg,
        lattice=lattice,
        periodic=periodic,
    )
    model = EEQBCModel.param2025(device=positions.device, dtype=positions.dtype)
    if cutoff is not None:
        model = model.replace(
            cn=model.cn.replace(cutoff=cutoff),
            cn_en=model.cn_en.replace(cutoff=cutoff),
        )
    return model(struct, return_energy=False)
