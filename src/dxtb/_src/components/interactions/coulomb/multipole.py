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
Coulomb: Anisotropic second-order electrostatics (AES2)
=======================================================

This module implements the anisotropic second-order electrostatic energy
that uses a damped multipole expansion.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.exceptions import DeviceError
from tad_mctc.math import einsum
from tad_mctc.tree import Node, child

from dxtb import IndexHelper
from dxtb._src.param import Param, ParamModule, as_param_module
from dxtb._src.typing import DD, Tensor, override
from dxtb._src.utils.tensors import normalize_device

from ..base import Interaction, InteractionCache

__all__ = [
    "AES2",
    "AES2Setup",
    "LABEL_AES2",
    "build_aes2_cache",
    "new_aes2",
    "setup_aes2",
]


LABEL_AES2 = "AES2"
"""Label for the 'AES2' interaction, coinciding with the class name."""


class AES2Setup(Node):
    """Numbers- and parameter-derived data for one AES2 interaction."""

    numbers: Tensor = child(keep_dtype=True)
    dmp3: Tensor = child()
    dmp5: Tensor = child()
    shift: Tensor = child()
    kexp: Tensor = child()
    rmax: Tensor = child()
    dkernel: Tensor = child()
    qkernel: Tensor = child()
    rad: Tensor = child()
    vcn: Tensor = child()
    pair_mask: Tensor = child(keep_dtype=True)


def setup_aes2(aes2: AES2, numbers: Tensor, ihelp: IndexHelper) -> AES2Setup:
    """Resolve AES2 parameters and structural masks to atom resolution."""
    return AES2Setup(
        numbers=numbers,
        dmp3=aes2.dmp3.clone(),
        dmp5=aes2.dmp5.clone(),
        shift=aes2.shift.clone(),
        kexp=aes2.kexp.clone(),
        rmax=aes2.rmax.clone(),
        dkernel=ihelp.spread_uspecies_to_atom(aes2.dkernel),
        qkernel=ihelp.spread_uspecies_to_atom(aes2.qkernel),
        rad=ihelp.spread_uspecies_to_atom(aes2.rad),
        vcn=ihelp.spread_uspecies_to_atom(aes2.vcn),
        pair_mask=real_pairs(numbers, mask_diagonal=True),
    )


def build_aes2_cache(setup: AES2Setup, positions: Tensor) -> AES2Cache:
    """Build fresh geometry-dependent AES2 data from setup and positions."""
    if positions.shape[-2:] != (setup.numbers.shape[-1], 3):
        raise ValueError("positions must have shape (..., nat, 3).")
    if positions.device != setup.dmp3.device:
        raise RuntimeError("AES2 setup and positions must share a device.")
    if positions.dtype != setup.dmp3.dtype:
        raise RuntimeError("AES2 setup and positions must share a dtype.")

    from tad_mctc import Structure
    from tad_mctc.ncoord import cn_gfn2

    cn = cn_gfn2(
        Structure(
            numbers=setup.numbers.expand(positions.shape[:-1]),
            positions=positions,
        )
    )
    t1 = torch.exp(-setup.kexp * (cn - setup.vcn - setup.shift))
    t2 = (setup.rmax - setup.rad) / (1.0 + t1)
    mrad = setup.rad + t2
    amat_sd, amat_dd, amat_sq = _build_atom_coulomb_matrix(
        setup.numbers,
        positions,
        mrad,
        setup.dmp3,
        setup.dmp5,
        setup.pair_mask,
    )
    return AES2Cache(
        mrad=mrad,
        dkernel=setup.dkernel.unsqueeze(-1),
        qkernel=setup.qkernel.unsqueeze(-1),
        amat_sd=amat_sd,
        amat_dd=amat_dd,
        amat_sq=amat_sq,
    )


class AES2Cache(InteractionCache):
    """
    Cache for AES2 interaction.
    """

    mrad: Tensor = child()
    """Multipole damping radii for all atoms."""

    dkernel: Tensor = child()
    """Kernel for on-site dipole exchange-correlation."""

    qkernel: Tensor = child()
    """Kernel for on-site quadrupole exchange-correlation."""

    amat_sd: Tensor = child()
    """
    Interation matrix for charges and dipoles
    (shape: ``(..., nat, nat, 3)``).
    """

    amat_dd: Tensor = child()
    """
    Interation matrix for dipoles and dipoles
    (shape: ``(..., nat, nat, 3, 3)``).
    """

    amat_sq: Tensor = child()
    """
    Interation matrix for charges and quadrupoles
    (shape: ``(..., nat, nat, 6)``).
    """


class AES2(Interaction):
    """
    Isotropic second-order electrostatic energy (ES2).
    """

    dmp3: Tensor = child()
    """Damping function for inverse quadratic contributions."""

    dmp5: Tensor = child()
    """Damping function for inverse cubic contributions."""

    dkernel: Tensor = child()
    """Kernel for on-site dipole exchange-correlation."""

    qkernel: Tensor = child()
    """Kernel for on-site quadrupole exchange-correlation."""

    shift: Tensor = child()
    """Shift for the generation of the multipolar damping radii."""

    kexp: Tensor = child()
    """Exponent for the generation of the multipolar damping radii."""

    rmax: Tensor = child()
    """Maximum radius for the multipolar damping radii."""

    rad: Tensor = child()
    """Base radii for the multipolar damping radii."""

    vcn: Tensor = child()
    """Valence coordination number."""

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> AES2Cache:
        """Build fresh AES2 data for direct component calls.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        AES2Cache
            Cache object for anisotropic second order electrostatics.

        """
        if numbers is None:
            raise ValueError("Atomic numbers are required for AES2 cache.")
        if positions is None:
            raise ValueError("Positions are required for AES2 cache.")
        if ihelp is None:
            raise ValueError("IndexHelper is required for AES2 cache creation.")

        return build_aes2_cache(setup_aes2(self, numbers, ihelp), positions)

    def get_atom_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, rad: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the atom-resolved interaction matrices.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Interaction matrices for:
            - charges and dipoles (shape: ``(..., nat, nat, 3)``),
            - dipoles and dipoles (shape: ``(..., nat, nat, 3, 3)``),
            - charges and quadrupoles (shape: ``(..., nat, nat, 6)``).
        """
        return _build_atom_coulomb_matrix(
            numbers,
            positions,
            rad,
            self.dmp3,
            self.dmp5,
            real_pairs(numbers, mask_diagonal=True),
        )

    @override
    def get_dipole_atom_energy(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved dipolar energy.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved shadow charges (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved dipolar energy.
        """
        if qdp is None:
            raise RuntimeError(
                "Dipole moments are required for dipolar energy calculation."
            )

        # tblite: coulomb/multipole.f90::get_energy
        vdp_sd = einsum("...ijx,...i->...jx", cache.amat_sd, qat)
        vdp_dd = 0.5 * einsum("...ijxy,...ix->...jy", cache.amat_dd, qdp)

        # tblite: coulomb/multipole.f90::get_kernel_energy
        # (ke = dk * dot(qdp, qdp))
        # Remember: cache.dkernel was unsqueezed in `get_cache`!
        ke = einsum("...ix,...ix,...ix->...i", cache.dkernel, qdp, qdp)

        return ke + einsum("...ix,...ix->...i", vdp_sd + vdp_dd, qdp)

    @override
    def get_quadrupole_atom_energy(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved dipolar energy.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved shadow charges (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved dipolar energy.
        """
        assert qqp is not None

        # tblite: coulomb/multipole.f90::get_energy
        vqp = einsum("...ijx,...i->...jx", cache.amat_sq, qat)

        # tblite: coulomb/multipole.f90::get_kernel_energy
        # (ke = dk * dot(qdp * scale, qdp))
        # Remember: cache.dkernel was unsqueezed in `get_cache`!
        scale = qqp.new_tensor([1, 2, 1, 2, 2, 1])
        ke = einsum("...ix,...ix,x,...ix->...i", cache.qkernel, qqp, scale, qqp)

        return ke + einsum("...ix,...ix->...i", vqp, qqp)

    @override
    def get_monopole_atom_potential(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved potential.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved monopolar potential.
        """
        assert qdp is not None
        assert qqp is not None

        vat_sd = einsum("...ijx,...jx->...i", cache.amat_sd, qdp)
        vat_sq = einsum("...ijx,...jx->...i", cache.amat_sq, qqp)

        return vat_sd + vat_sq

    @override
    def get_dipole_atom_potential(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate atom-resolved dipolar potential.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved monopolar potential.
        """
        assert qdp is not None

        vdp_sd = einsum("...ijx,...i->...jx", cache.amat_sd, qat)
        vdp_dd = einsum("...ijxy,...ix->...jy", cache.amat_dd, qdp)

        # tblite: coulomb/multipole.f90::get_kernel_potential
        kernel_pot_dp = 2 * cache.dkernel * qdp

        return vdp_sd + vdp_dd + kernel_pot_dp

    @override
    def get_quadrupole_atom_potential(
        self,
        cache: AES2Cache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        r"""
        Calculate atom-resolved quadrupolar potential.

        Parameters
        ----------
        cache : ComponentCache
            Restart data for the interaction.
        qat : Tensor
            Atom-resolved partial charges (shape: ``(..., nat)``).
        qdp : Tensor
            Atom-resolved dipole moments (shape: ``(..., nat, 3)``).
        qqp : Tensor
            Atom-resolved quadrupole moments (shape: ``(..., nat, 6)``).

        Returns
        -------
        Tensor
            Atom-resolved monopolar potential.
        """
        assert qqp is not None

        vqp = einsum("...ijx,...i->...jx", cache.amat_sq, qat)

        # tblite: coulomb/multipole.f90::get_kernel_potential
        scale = qqp.new_tensor([1, 2, 1, 2, 2, 1])
        kernel_pot_dp = 2 * cache.qkernel * qqp * scale

        return vqp + kernel_pot_dp


def _build_atom_coulomb_matrix(
    numbers: Tensor,
    positions: Tensor,
    rad: Tensor,
    dmp3: Tensor,
    dmp5: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct atom-resolved AES2 matrices from explicit tensor inputs."""
    # Invalid atom pairs include padded atoms and the physical diagonal. Make
    # those displacement vectors nonsingular before taking a norm. Masking a
    # zero distance after ``cdist`` is too late for higher derivatives at
    # coincident padded coordinates.
    pair_mask = mask.to(dtype=torch.bool, device=positions.device)
    displacement = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    safe_displacement = torch.zeros_like(displacement) + positions.new_tensor(
        (1.0, 0.0, 0.0)
    )
    displacement = torch.where(
        pair_mask.unsqueeze(-1), displacement, safe_displacement
    )
    dist = torch.linalg.vector_norm(displacement, dim=-1)
    dist = torch.where(pair_mask, dist, torch.ones_like(dist))

    # (nb, nat, nat)
    g1 = storch.safe_reciprocal(dist)
    g3 = g1 * g1 * g1
    g5 = g3 * g1 * g1

    # (nb, nat, nat)
    rr = 0.5 * (rad.unsqueeze(-1) + rad.unsqueeze(-2)) * g1
    fdmp3 = 1.0 / (1.0 + 6.0 * rr**dmp3)
    fdmp5 = 1.0 / (1.0 + 6.0 * rr**dmp5)

    # (nb, nat, nat, 3)
    rij = torch.where(pair_mask.unsqueeze(-1), displacement, 0.0)

    # Monopole / Dipole

    # (nb, nat, nat, 1)
    _g3 = g3.unsqueeze(-1)
    _fdmp3 = fdmp3.unsqueeze(-1)

    # (nb, nat, nat, 3) * (nb, nat, nat, 1) -> (nb, nat, nat, 3)
    sd = rij * _g3 * _fdmp3

    # Dipole / Dipole

    # (nb, nat, nat, 1)
    _g5 = g5.unsqueeze(-1)
    _fdmp5 = fdmp5.unsqueeze(-1)
    g5_fdmp5 = _g5 * _fdmp5

    # (nb, nat, nat, 1, 1)
    _g5_fdmp5 = g5_fdmp5.unsqueeze(-1)

    # (nb, 1, 1, 3, 3)
    unity = torch.eye(3, device=positions.device, dtype=positions.dtype)
    unity = unity.unsqueeze(-3).unsqueeze(-3)

    # (nb, nat, nat, 3, 3)
    dd = (
        unity * _g3.unsqueeze(-1) * _fdmp5.unsqueeze(-1)
        - rij.unsqueeze(-1) * rij.unsqueeze(-2) * 3 * _g5_fdmp5
    )

    # Monopole / Quadrupole

    # (nb, nat, nat, 6)
    sq = torch.stack(
        (
            rij[..., 0] * rij[..., 0],
            2 * rij[..., 0] * rij[..., 1],
            rij[..., 1] * rij[..., 1],
            2 * rij[..., 0] * rij[..., 2],
            2 * rij[..., 1] * rij[..., 2],
            rij[..., 2] * rij[..., 2],
        ),
        dim=-1,
    )

    # (nb, nat, nat, 6) * (nb, nat, nat, 1) -> (nb, nat, nat, 6)
    sq = sq * g5_fdmp5

    # Explicitly clear every invalid pair. In particular, the isotropic part
    # of ``dd`` is nonzero even when ``rij`` is zero, so masking only the
    # displacement does not suppress padded dipole couplings.
    sd = torch.where(pair_mask.unsqueeze(-1), sd, 0.0)
    dd = torch.where(pair_mask[..., None, None], dd, 0.0)
    sq = torch.where(pair_mask.unsqueeze(-1), sq, 0.0)

    return sd, dd, sq


def new_aes2(
    unique: Tensor,
    par: Param | ParamModule,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> AES2 | None:
    """
    Create new instance of :class:`.AES2`.

    Parameters
    ----------
    unique : Tensor
        Unique elements in the Structure (shape: ``(nunique,)``).
    par : Param | ParamModule
        Parametrization of an extended tight-binding (xTB) Method.

    Returns
    -------
    AES2 | None
        Instance of the AES2 class or ``None`` if no AES2 is used.
    """
    par = as_param_module(par, device=device, dtype=dtype)
    dd: DD = {"device": device, "dtype": dtype or par.dtype}

    if "multipole" not in par or par.is_none("multipole"):
        return None

    if device is not None:
        if normalize_device(device) != unique.device:
            raise DeviceError(
                f"Passed device ({device}) and device of `unique` tensor "
                f"({unique.device}) do not match."
            )

    dkernel = par.get_elem_param(unique, "dkernel")
    qkernel = par.get_elem_param(unique, "qkernel")
    rad = par.get_elem_param(unique, "mprad")
    vcn = par.get_elem_param(unique, "mpvcn")

    return AES2(
        dmp3=par.get("multipole.damped.dmp3").to(**dd),
        dmp5=par.get("multipole.damped.dmp5").to(**dd),
        dkernel=dkernel.to(**dd),
        qkernel=qkernel.to(**dd),
        shift=par.get("multipole.damped.shift").to(**dd),
        kexp=par.get("multipole.damped.kexp").to(**dd),
        rmax=par.get("multipole.damped.rmax").to(**dd),
        rad=rad.to(**dd),
        vcn=vcn.to(**dd),
    )
