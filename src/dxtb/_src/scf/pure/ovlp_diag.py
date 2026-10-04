from __future__ import annotations

import torch

from dxtb._src.exlibs import xitorch as xt
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import Tensor

__all__ = ["get_overlap", "diagonalize", "symeig_padded"]

PADDING_SHIFT = 1e3
"""
Diagonal of the padding block of the Hamiltonian (Hartree) during the
diagonalization, which moves the eigenvalues of the padding orbitals behind
all physical ones.
"""


def symeig_padded(
    hamiltonian: Tensor, smat: Tensor, eigen_options: dict
) -> tuple[Tensor, Tensor]:
    """
    Generalized eigendecomposition of zero-padded (batched) matrices.

    The occupation is masked by orbital position, i.e., it requires the
    eigenvalues of the padding orbitals at the end, as
    :func:`tad_mctc.storch.eighb` does for the unrolled SCF. With a zero
    padding block, the padding eigenvalues are 0 and are sorted between the
    physical ones, which gives electrons to padding orbitals whenever an
    orbital close to zero is (partially) occupied, e.g., for anions in a
    padded batch. The padding block of the Hamiltonian is therefore shifted
    for the diagonalization and the padding eigenvalues are reported as 0.

    Parameters
    ----------
    hamiltonian : Tensor
        Current Hamiltonian matrix.
    smat : Tensor
        Current overlap matrix.
    eigen_options : dict
        Options for calculating EVs.

    Returns
    -------
    evals : Tensor
        Eigenvalues of the Hamiltonian.
    evecs : Tensor
        Eigenvectors of the Hamiltonian.
    """
    zeros = torch.eq(smat, 0)
    mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

    h_op = xt.LinearOperator.m(
        hamiltonian + torch.diag_embed(PADDING_SHIFT * mask.type(smat.dtype))
    )
    o_op = get_overlap(smat)
    evals, evecs = xt.linalg.lsymeig(A=h_op, M=o_op, **eigen_options)

    # padding orbitals are at the end, as in the (padded) AO order
    return torch.where(mask, torch.zeros_like(evals), evals), evecs


def get_overlap(smat: Tensor) -> xt.LinearOperator:
    """
    Get the overlap matrix.

    Parameters
    ----------
    smat : Tensor
        Current overlap matrix as tensor.

    Returns
    -------
    LinearOperator
        Overlap matrix as linear operator.
    """

    zeros = torch.eq(smat, 0)
    mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

    return xt.LinearOperator.m(
        smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)
    )


@timer_decorator("Diagonalize", "SCF")
def diagonalize(
    hamiltonian: Tensor, ovlp: Tensor, eigen_options: dict
) -> tuple[Tensor, Tensor]:
    """
    Diagonalize the Hamiltonian.

    Parameters
    ----------
    hamiltonian : Tensor
        Current Hamiltonian matrix.
    ovlp : Tensor
        Current overlap matrix.
    eigen_options : dict
        Options for calculating EVs.

    Returns
    -------
    evals : Tensor
        Eigenvalues of the Hamiltonian.
    evecs : Tensor
        Eigenvectors of the Hamiltonian.
    """

    return symeig_padded(hamiltonian, ovlp, eigen_options)
