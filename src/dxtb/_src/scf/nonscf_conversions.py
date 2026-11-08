"""
Conversions
===========

Charge and density conversions used by the non-self-consistent GFN0 path.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum
from tad_mctc.units.energy import KELVIN2AU

from dxtb._src.components.interactions.container import Charges
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import Callable, Tensor
from dxtb._src.wavefunction import filling
from dxtb.config import ConfigSCF

from .utils import get_density
from .nonscf_data import _Data
from .nonscf_ovlp_diag import diagonalize

__all__ = [
    "density_to_charges",
    "hamiltonian_to_density",
]

Diagonalizer = Callable[[Tensor], "tuple[Tensor, Tensor]"]


@timer_decorator("Charges", "SCF")
def density_to_charges(density: Tensor, data: _Data, cfg: ConfigSCF) -> Charges:
    """
    Compute the orbital charges from the density matrix.

    Parameters
    ----------
    density : Tensor
        Density matrix.
    data: _Data
        Data cache for intermediary storage during self-consistency.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges vector.
    """

    # Calculate diagonal directly by using index "i" twice on left side.
    # The slower but more readable approach would instead compute the full
    # matrix with "...ik,...kj->...ij" and only extract the diagonal
    # afterwards with `torch.diagonal(tensor, dim1=-2, dim2=-1)`.
    data.energy = einsum("...ik,...ki->...i", density, data.ints.hcore)

    # monopolar charges
    populations = einsum("...ik,...ki->...i", density, data.ints.overlap)
    charges = Charges(
        mono=data.n0 - populations, batch_mode=data.ihelp.batch_mode
    )

    # Atomic dipole moments (dipole charges)
    if data.ints.dipole is not None:
        # Again, the diagonal is directly calculated instead of full matrix
        # ("...ik,...mkj->...ijm") as `torch.diagonal` behaves weirdly for
        # more than 2D tensors. Additionally, we move the multipole
        # dimension to the back, which is required for the reduction to
        # atom-resolution.
        charges.dipole = data.ihelp.reduce_orbital_to_atom(
            -einsum("...ik,...mki->...im", density, data.ints.dipole),
            extra=True,
            dim=-2,
        )

    # Atomic quadrupole moments (quadrupole charges)
    if data.ints.quadrupole is not None:
        charges.quad = data.ihelp.reduce_orbital_to_atom(
            -einsum("...ik,...mki->...im", density, data.ints.quadrupole),
            extra=True,
            dim=-2,
        )

    return charges


def hamiltonian_to_density(
    hamiltonian: Tensor,
    data: _Data,
    cfg: ConfigSCF,
    diagonalizer: Diagonalizer | None = None,
) -> Tensor:
    """
    Compute the density matrix from the Hamiltonian.

    Parameters
    ----------
    hamiltonian : Tensor
        Hamiltonian matrix.
    data: _Data
        Data cache for intermediary storage during self-consistency.
    cfg: SCFConfig
        Configuration for SCF settings.
    diagonalizer : Callable[[Tensor], tuple[Tensor, Tensor]] | None, optional
        Diagonalizer of the Hamiltonian (returns eigenvalues and
        eigenvectors). Defaults to the generalized eigensolver of
        :mod:`.ovlp_diag` .

    Returns
    -------
    Tensor
        Density matrix.
    """

    if diagonalizer is None:
        data.evals, data.evecs = diagonalize(hamiltonian, data.ints.overlap)
    else:
        data.evals, data.evecs = diagonalizer(hamiltonian)

    # fixed number of alpha and beta electrons from the setup (as in
    # tblite); it is never re-derived from the previous occupation and
    # keeps the graph of the total charge
    nel = data.nel

    # expand emo/mask to second dim (for alpha/beta electrons)
    emo = data.evals.unsqueeze(-2).expand([*nel.shape, -1])
    mask = data.ihelp.spread_shell_to_orbital(data.ihelp.orbitals_per_shell)
    mask = mask.unsqueeze(-2).expand([*nel.shape, -1])

    # Fermi smearing only for non-zero electronic temperature
    kt = data.ints.hcore.new_tensor(cfg.fermi.etemp * KELVIN2AU)
    if not torch.all(kt < 3e-7):  # 0.1 Kelvin * K2AU
        data.occupation = filling.get_fermi_occupation(
            nel,
            emo,
            kt=kt,
            mask=mask,
            maxiter=cfg.fermi.maxiter,
            thr=cfg.fermi.thresh,
            diff_order=cfg.fermi.diff_order,
        )

        # check if number of electrons is still correct
        _nel = data.occupation.sum(-1)
        if torch.any(torch.abs(nel - _nel) > 1e-4):
            raise RuntimeError(
                f"Number of electrons changed during Fermi smearing "
                f"({nel} -> {_nel})."
            )

    return get_density(data.evecs, data.occupation.sum(-2))
