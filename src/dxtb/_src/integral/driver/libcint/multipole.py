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
Implementation: Overlap and Multipoles (Libcint)
================================================

Standalone origin-centered and internal xTB ket-centered integrals for the
`libcint` backend.
"""

from __future__ import annotations

from dxtb._src.constants import labels
from dxtb._src.typing import Tensor

from .driver import LibcintIntegralSetup

__all__ = [
    "build_overlap",
    "build_dipole",
    "build_quadrupole",
    "evaluate_moments",
]


def build_overlap(setup: LibcintIntegralSetup, positions: Tensor) -> Tensor:
    """Build the normalized overlap matrix of shape ``(nao, nao)``."""
    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    return libcint.overlap(*setup.create_libcint(positions))


def build_dipole(setup: LibcintIntegralSetup, positions: Tensor) -> Tensor:
    """Build the origin-centered dipole matrix of shape ``(3, nao, nao)``."""
    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    return libcint.int1e("r0", *setup.create_libcint(positions))


def build_quadrupole(setup: LibcintIntegralSetup, positions: Tensor) -> Tensor:
    """Build the origin-centered quadrupole matrix of shape ``(9, nao, nao)``."""
    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    return libcint.int1e("r0r0", *setup.create_libcint(positions), packed=False)


def evaluate_moments(
    setup: LibcintIntegralSetup,
    positions: Tensor,
    intlevel: int,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """
    Evaluate the normalized overlap and the ket-centered dipole and
    quadrupole (not yet traceless, see
    :func:`~dxtb._src.integral.evaluation.evaluate_integrals`).
    """
    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    basis, structure = setup.create_libcint(positions)
    overlap = libcint.overlap(basis, structure)
    dipole = None
    quadrupole = None

    if intlevel >= labels.INTLEVEL_DIPOLE:
        # tad-libcint 0.4 returns native ket-centered j/jj moments;
        # jj is already packed as xx, yx, yy, zx, zy, zz.
        dipole = libcint.int1e("j", basis, structure)
        if intlevel >= labels.INTLEVEL_QUADRUPOLE:
            quadrupole = libcint.int1e("jj", basis, structure)

    return overlap, dipole, quadrupole
