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
Implementation: Multipole (Libcint)
===================================

Standalone origin-centered and internal xTB ket-centered multipole evaluation
for the `libcint` integral backend.
"""

from __future__ import annotations

from dxtb._src.constants import labels
from dxtb._src.integral.types.quadrupole import make_quadrupole_traceless
from dxtb._src.typing import Tensor

from .driver import LibcintIntegralSetup

__all__ = [
    "build_dipole",
    "build_dipole_libcint",
    "build_multipole_libcint",
    "build_quadrupole",
    "build_quadrupole_libcint",
    "evaluate_moments",
]

ALLOWED_MULTIPOLES = ("r0", "j", "r0r0", "jj", "jjj")


def build_multipole_libcint(
    setup: LibcintIntegralSetup,
    positions: Tensor,
    intstring: str,
    *,
    packed: bool = True,
) -> Tensor:
    """Build a raw libcint multipole matrix for the requested operator."""
    # pylint: disable=import-outside-toplevel
    from dxtb._src.exlibs import libcint

    if intstring not in ALLOWED_MULTIPOLES:
        raise ValueError(
            f"Unknown integral string '{intstring}' provided. "
            f"Only '{', '.join(ALLOWED_MULTIPOLES)}' are allowed."
        )

    basis, structure = setup.create_libcint(positions)
    return libcint.int1e(intstring, basis, structure, packed=packed)


def build_dipole(
    setup: LibcintIntegralSetup,
    positions: Tensor,
) -> Tensor:
    """Build the standalone origin-centered three-component dipole matrix."""
    return build_multipole_libcint(setup, positions, "r0")


def build_quadrupole(
    setup: LibcintIntegralSetup,
    positions: Tensor,
) -> Tensor:
    """Build the standalone origin-centered nine-component quadrupole matrix."""
    return build_multipole_libcint(setup, positions, "r0r0", packed=False)


build_dipole_libcint = build_dipole
build_quadrupole_libcint = build_quadrupole


def evaluate_moments(
    setup: LibcintIntegralSetup,
    positions: Tensor,
    intlevel: int,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Evaluate normalized overlap and internal xTB ket-centered moments."""
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
            raw_quadrupole = libcint.int1e("jj", basis, structure)
            quadrupole = make_quadrupole_traceless(raw_quadrupole)

    return overlap, dipole, quadrupole
