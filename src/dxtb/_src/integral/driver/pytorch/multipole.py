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
Implementation: Overlap and Multipoles (PyTorch)
================================================

Overlap, dipole and quadrupole integrals built from the interchangeable
``compute_1d`` kernels (see
:mod:`dxtb._src.integral.driver.pytorch.impls.kernels`). The overlap is the
zeroth moment of the same pipeline.

The pipeline computes every shell pair once and mirrors it into the other
triangle, which requires a common origin for all moments. The internal xTB
moments are centered on the ket atom instead, so they are shifted from the
common origin to the ket atom after assembly.
"""

from __future__ import annotations

from tad_mctc.math import einsum

from dxtb._src.constants import labels
from dxtb._src.typing import Tensor
from dxtb._src.utils import wrap_gather

from .impls.kernels import get_kernel
from .impls.pairs import assemble_matrix
from .impls.pipeline import (
    DIPOLE_COMPONENTS,
    OVERLAP_COMPONENTS,
    QUADRUPOLE_COMPONENTS,
    QUADRUPOLE_LOWER_COMPONENTS,
)
from .setup import PytorchIntegralSetup

__all__ = [
    "build_overlap",
    "build_dipole",
    "build_quadrupole",
    "evaluate_moments",
]


def _build_moment(
    setup: PytorchIntegralSetup,
    positions: Tensor,
    components: tuple[tuple[int, int, int], ...],
) -> Tensor:
    """Build an origin-centered moment matrix of shape ``(ncomp, nao, nao)``."""
    return assemble_matrix(
        get_kernel(setup.algorithm),
        None,
        setup.alphas,
        setup.coeffs,
        positions,
        components,
        plan=setup.pair_plan,
    )


def build_overlap(setup: PytorchIntegralSetup, positions: Tensor) -> Tensor:
    """Build the normalized overlap matrix of shape ``(nao, nao)``."""
    return _build_moment(setup, positions, OVERLAP_COMPONENTS)[0]


def build_dipole(setup: PytorchIntegralSetup, positions: Tensor) -> Tensor:
    """Build the origin-centered dipole matrix of shape ``(3, nao, nao)``."""
    return _build_moment(setup, positions, DIPOLE_COMPONENTS)


def build_quadrupole(setup: PytorchIntegralSetup, positions: Tensor) -> Tensor:
    """Build the origin-centered quadrupole matrix of shape ``(9, nao, nao)``."""
    return _build_moment(setup, positions, QUADRUPOLE_COMPONENTS)


# Cartesian axes (a, b) of the six lower-triangle quadrupole components
# (x = 0, y = 1, z = 2), in the order of QUADRUPOLE_LOWER_COMPONENTS.
#              xx yx yy zx zy zz
_AXIS_A = [0, 1, 1, 2, 2, 2]
_AXIS_B = [0, 0, 1, 0, 1, 2]


def _shift_dipole_to_ket(
    dipole: Tensor, overlap: Tensor, ket_positions: Tensor
) -> Tensor:
    r"""
    Shift the dipole integral from the common origin to the ket atom.

    With the ket atom at :math:`R_j`, the shifted operator is
    :math:`r_a - R_{a,j}`:

    .. math::

        D_{a,ij} = \langle i | r_a - R_{a,j} | j \rangle
                 = D^0_{a,ij} - R_{a,j} S_{ij}

    Parameters
    ----------
    dipole : Tensor
        Dipole integral :math:`D^0` about the common origin, shape
        ``(..., 3, nao, nao)``.
    overlap : Tensor
        Overlap integral :math:`S`, shape ``(..., nao, nao)``.
    ket_positions : Tensor
        Position of the atom of each orbital, shape ``(..., nao, 3)``.

    Returns
    -------
    Tensor
        Ket-centered dipole integral, shape ``(..., 3, nao, nao)``.
    """
    # R_{a,j} S_ij for every axis a
    shift = einsum("...ja,...ij->...aij", ket_positions, overlap)

    # D_a = D^0_a - R_{a,j} S_ij
    return dipole - shift


def _shift_quadrupole_to_ket(
    quadrupole: Tensor,
    dipole: Tensor,
    overlap: Tensor,
    ket_positions: Tensor,
) -> Tensor:
    r"""
    Shift the six-component quadrupole integral from the common origin to the
    ket atom.

    Expanding the shifted operator gives, for every component :math:`ab`,

    .. math::

        \begin{align}
            Q_{ab,ij} &= \langle i | (r_a - R_{a,j})(r_b - R_{b,j}) | j \rangle \\
            &= Q^0_{ab,ij} - R_{b,j} D^0_{a,ij} - R_{a,j} D^0_{b,ij}
               + R_{a,j} R_{b,j} S_{ij}
        \end{align}

    For the diagonal components (:math:`a = b`), the two dipole terms are
    equal and the expression becomes
    :math:`Q^0_{aa} - 2 R_{a,j} D^0_a + R_{a,j}^2 S`.

    Parameters
    ----------
    quadrupole : Tensor
        Quadrupole integral :math:`Q^0` about the common origin, shape
        ``(..., 6, nao, nao)``, in the order xx, yx, yy, zx, zy, zz.
    dipole : Tensor
        Dipole integral :math:`D^0` about the *same* common origin (not the
        ket-centered one), shape ``(..., 3, nao, nao)``.
    overlap : Tensor
        Overlap integral :math:`S`, shape ``(..., nao, nao)``.
    ket_positions : Tensor
        Position of the atom of each orbital, shape ``(..., nao, 3)``.

    Returns
    -------
    Tensor
        Ket-centered quadrupole integral, shape ``(..., 6, nao, nao)``.
    """
    # Ket-atom coordinates R_{a,j} and R_{b,j} for each of the six components
    # (c), shape (..., nao, 6)
    r_a = ket_positions[..., _AXIS_A]
    r_b = ket_positions[..., _AXIS_B]

    # Dipole components D^0_a and D^0_b for each of the six components (c),
    # shape (..., 6, nao, nao)
    d_a = dipole[..., _AXIS_A, :, :]
    d_b = dipole[..., _AXIS_B, :, :]

    # R_{b,j} D^0_{a,ij}
    rb_da = einsum("...jc,...cij->...cij", r_b, d_a)

    # R_{a,j} D^0_{b,ij}
    ra_db = einsum("...jc,...cij->...cij", r_a, d_b)

    # R_{a,j} R_{b,j} S_ij
    ra_rb_s = einsum("...jc,...jc,...ij->...cij", r_a, r_b, overlap)

    # Q_ab = Q^0_ab - R_b D^0_a - R_a D^0_b + R_a R_b S
    return quadrupole - rb_da - ra_db + ra_rb_s


def evaluate_moments(
    setup: PytorchIntegralSetup,
    positions: Tensor,
    intlevel: int,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """
    Evaluate the normalized overlap and the ket-centered dipole and
    quadrupole (not yet traceless, see
    :func:`~dxtb._src.integral.evaluation.evaluate_integrals`).
    """
    # The overlap and the ket-centered moments do not depend on where the
    # molecule sits. The raw moments about the common origin, however, grow
    # with the distance to that origin, and the shift to the ket atom then
    # subtracts large, nearly equal terms (in single precision, the
    # quadrupole loses about 1e-2 for a molecule 100 Bohr from the origin).
    # Using the centroid as the common origin keeps these terms small.
    positions = positions - positions.mean(dim=-2, keepdim=True)

    overlap = build_overlap(setup, positions)
    dipole = None
    quadrupole = None

    if intlevel >= labels.INTLEVEL_DIPOLE:
        raw_dipole = build_dipole(setup, positions)
        ket_positions = wrap_gather(
            positions, -2, setup.pair_plan.orbitals_to_atom, extra=True
        )
        dipole = _shift_dipole_to_ket(raw_dipole, overlap, ket_positions)

        if intlevel >= labels.INTLEVEL_QUADRUPOLE:
            raw_quadrupole = _build_moment(
                setup, positions, QUADRUPOLE_LOWER_COMPONENTS
            )
            quadrupole = _shift_quadrupole_to_ket(
                raw_quadrupole, raw_dipole, overlap, ket_positions
            )

    return overlap, dipole, quadrupole
