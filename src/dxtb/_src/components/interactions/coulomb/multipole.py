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

from dataclasses import dataclass

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.exceptions import DeviceError
from tad_mctc.math import einsum

from dxtb import IndexHelper
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import (
    DD,
    Any,
    Slicers,
    Tensor,
    TensorLike,
    get_default_dtype,
    override,
)
from dxtb._src.utils.tensors import normalize_device

from ..base import Interaction, InteractionCache

__all__ = [
    "AES2",
    "AES2Setup",
    "LABEL_AES2",
    "build_aes2_data",
    "new_aes2",
    "setup_aes2",
]


LABEL_AES2 = "AES2"
"""Label for the 'AES2' interaction, coinciding with the class name."""


@dataclass(frozen=True, eq=False)
class AES2Setup:
    """Numbers- and parameter-derived data for one AES2 interaction."""

    numbers: Tensor
    dmp3: Tensor
    dmp5: Tensor
    shift: Tensor
    kexp: Tensor
    rmax: Tensor
    dkernel: Tensor
    qkernel: Tensor
    rad: Tensor
    vcn: Tensor
    pair_mask: Tensor


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


def build_aes2_data(setup: AES2Setup, positions: Tensor) -> AES2Cache:
    """Build fresh geometry-dependent AES2 data from setup and positions."""
    if positions.shape[-2:] != (setup.numbers.shape[-1], 3):
        raise ValueError("positions must have shape (..., nat, 3).")
    if positions.device != setup.dmp3.device:
        raise RuntimeError("AES2 setup and positions must share a device.")
    if positions.dtype != setup.dmp3.dtype:
        raise RuntimeError("AES2 setup and positions must share a dtype.")

    from dxtb._src.ncoord import cn_d3, gfn2_count

    cn = cn_d3(setup.numbers, positions, counting_function=gfn2_count)
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


class AES2Cache(InteractionCache, TensorLike):
    """
    Cache for AES2 interaction.
    """

    __store: Store | None
    """Storage for cache (required for culling)."""

    mrad: Tensor
    """Multipole damping radii for all atoms."""

    dkernel: Tensor
    """Kernel for on-site dipole exchange-correlation."""

    qkernel: Tensor
    """Kernel for on-site quadrupole exchange-correlation."""

    amat_sd: Tensor
    """
    Interation matrix for charges and dipoles
    (shape: ``(..., nat, nat, 3)``).
    """

    amat_dd: Tensor
    """
    Interation matrix for dipoles and dipoles
    (shape: ``(..., nat, nat, 3, 3)``).
    """

    amat_sq: Tensor
    """
    Interation matrix for charges and quadrupoles
    (shape: ``(..., nat, nat, 6)``).
    """

    __slots__ = [
        "__store",
        "mrad",
        "dkernel",
        "qkernel",
        "amat_sd",
        "amat_dd",
        "amat_sq",
    ]

    def __init__(
        self,
        mrad: Tensor,
        dkernel: Tensor,
        qkernel: Tensor,
        amat_sd: Tensor,
        amat_dd: Tensor,
        amat_sq: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is None else mrad.device,
            dtype=dtype if dtype is None else mrad.dtype,
        )
        self.mrad = mrad
        self.dkernel = dkernel
        self.qkernel = qkernel
        self.amat_sd = amat_sd
        self.amat_dd = amat_dd
        self.amat_sq = amat_sq

        self.__store = None

    class Store:
        """
        Storage container for cache containing ``__slots__`` before culling.
        """

        mrad: Tensor
        """Multipole damping radii for all atoms."""

        dkernel: Tensor
        """Kernel for on-site dipole exchange-correlation."""

        qkernel: Tensor
        """Kernel for on-site quadrupole exchange-correlation."""

        amat_sd: Tensor
        """Interation matrix for charges and dipoles."""

        amat_dd: Tensor
        """Interation matrix for dipoles and dipoles."""

        amat_sq: Tensor
        """Interation matrix for charges and quadrupoles."""

        def __init__(
            self,
            mrad: Tensor,
            dkernel: Tensor,
            qkernel: Tensor,
            amat_sd: Tensor,
            amat_dd: Tensor,
            amat_sq: Tensor,
        ) -> None:
            self.mrad = mrad
            self.dkernel = dkernel
            self.qkernel = qkernel
            self.amat_sd = amat_sd
            self.amat_dd = amat_dd
            self.amat_sq = amat_sq

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        if self.__store is None:
            self.__store = self.Store(
                self.mrad,
                self.dkernel,
                self.qkernel,
                self.amat_sd,
                self.amat_dd,
                self.amat_sq,
            )

        _slicer = slicers["atom"]
        _slicer_one = tuple([~conv, *_slicer])
        _slicer_two = tuple([~conv, *_slicer, *_slicer])

        self.mrad = self.mrad[_slicer_one]
        self.dkernel = self.dkernel[_slicer_one]
        self.qkernel = self.qkernel[_slicer_one]
        self.amat_sd = self.amat_sd[_slicer_two]
        self.amat_dd = self.amat_dd[_slicer_two]
        self.amat_sq = self.amat_sq[_slicer_two]

    def restore(self) -> None:
        if self.__store is None:
            raise RuntimeError("Nothing to restore. Store is empty.")

        self.mrad = self.__store.mrad
        self.dkernel = self.__store.dkernel
        self.qkernel = self.__store.qkernel
        self.amat_sd = self.__store.amat_sd
        self.amat_dd = self.__store.amat_dd
        self.amat_sq = self.__store.amat_sq


class AES2(Interaction):
    """
    Isotropic second-order electrostatic energy (ES2).
    """

    dmp3: Tensor
    """Damping function for inverse quadratic contributions."""

    dmp5: Tensor
    """Damping function for inverse cubic contributions."""

    dkernel: Tensor
    """Kernel for on-site dipole exchange-correlation."""

    qkernel: Tensor
    """Kernel for on-site quadrupole exchange-correlation."""

    shift: Tensor
    """Shift for the generation of the multipolar damping radii."""

    kexp: Tensor
    """Exponent for the generation of the multipolar damping radii."""

    rmax: Tensor
    """Maximum radius for the multipolar damping radii."""

    rad: Tensor
    """Base radii for the multipolar damping radii."""

    vcn: Tensor
    """Valence coordination number."""

    __slots__ = [
        "dmp3",
        "dmp5",
        "dkernel",
        "qkernel",
        "shift",
        "kexp",
        "rmax",
        "rad",
        "vcn",
    ]

    def __init__(
        self,
        dmp3: Tensor,
        dmp5: Tensor,
        dkernel: Tensor,
        qkernel: Tensor,
        shift: Tensor,
        kexp: Tensor,
        rmax: Tensor,
        rad: Tensor,
        vcn: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        # scalar parameters
        self.dmp3 = dmp3.to(**self.dd)
        self.dmp5 = dmp5.to(**self.dd)
        self.shift = shift.to(**self.dd)
        self.kexp = kexp.to(**self.dd)
        self.rmax = rmax.to(**self.dd)

        # element-wise parameters
        self.dkernel = dkernel.to(**self.dd)
        self.qkernel = qkernel.to(**self.dd)
        self.rad = rad.to(**self.dd)
        self.vcn = vcn.to(**self.dd)

    # pylint: disable=unused-argument
    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> AES2Cache:
        """Build fresh AES2 data for the compatibility API.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
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

        return build_aes2_data(setup_aes2(self, numbers, ihelp), positions)

    def get_atom_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, rad: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the atom-resolved interaction matrices.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
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

    def update(self, **kwargs: Any) -> None:
        """Reject updates for exact setup-migrated AES2 interactions."""
        if type(self) is AES2:
            raise RuntimeError(
                "AES2 parameters are setup-derived and cannot be updated. "
                "Create a new Model/System/Calculator with changed parameters."
            )
        super().update(**kwargs)

    def reset(self) -> None:
        """Reject resets for exact setup-migrated AES2 interactions."""
        if type(self) is AES2:
            raise RuntimeError(
                "AES2 parameters are setup-derived and cannot be reset. "
                "Create a new Model/System/Calculator with changed parameters."
            )
        super().reset()

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
    eps = positions.new_tensor(torch.finfo(positions.dtype).eps)

    dist = storch.cdist(positions, positions, p=2)

    # (nb, nat, nat)
    g1 = storch.safe_reciprocal(dist)
    g3 = g1 * g1 * g1
    g5 = g3 * g1 * g1

    # (nb, nat, nat)
    rr = 0.5 * (rad.unsqueeze(-1) + rad.unsqueeze(-2)) * g1
    fdmp3 = 1.0 / (1.0 + 6.0 * rr**dmp3)
    fdmp5 = 1.0 / (1.0 + 6.0 * rr**dmp5)

    # (nb, nat, nat, 3)
    rij = torch.where(
        mask.unsqueeze(-1),
        positions.unsqueeze(-2) - positions.unsqueeze(-3),
        eps,
    )

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
        Unique elements in the system (shape: ``(nunique,)``).
    par : Param | ParamModule
        Representation of an extended tight-binding model.

    Returns
    -------
    AES2 | None
        Instance of the AES2 class or ``None`` if no AES2 is used.
    """
    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    # compatibility with previous version based on `Param`
    if not isinstance(par, ParamModule):
        par = ParamModule(par, **dd)

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
        dmp3=par.get("multipole.damped.dmp3"),
        dmp5=par.get("multipole.damped.dmp5"),
        dkernel=dkernel,
        qkernel=qkernel,
        shift=par.get("multipole.damped.shift"),
        kexp=par.get("multipole.damped.kexp"),
        rmax=par.get("multipole.damped.rmax"),
        rad=rad,
        vcn=vcn,
        **dd,
    )
