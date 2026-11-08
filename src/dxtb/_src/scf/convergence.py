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
"""Current raw-residual convergence boundary for functional SCF."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from dxtb._src.typing import Tensor

__all__ = ["SCFConvergence", "SCFResidual"]


@dataclass(frozen=True, eq=False)
class SCFResidual:
    """L2 and max-absolute norms of a projected raw physical residual."""

    l2: Tensor
    max_abs: Tensor

    def as_tensor(self) -> Tensor:
        """Return norms in the public residual order."""
        return torch.stack((self.l2, self.max_abs))


@dataclass(frozen=True, eq=False)
class SCFConvergence:
    """Existing two-norm stopping predicate, exposed as an S1 boundary.

    Notes
    -----
    With the potential SCF variable and ``sqrt(2)`` off-diagonal quadrupole
    weighting, ``residual.l2`` is strictly invariant under :math:`\\mathrm{SO}(3)`
    rotations because shell monopole potentials are rotational scalars,
    Cartesian dipoles carry the Euclidean norm, and weighted packed quadrupoles
    carry the Frobenius norm.

    By contrast, ``residual.max_abs`` (``delta.abs().amax()``) is a
    coordinate-axis-dependent :math:`\\ell_\\infty` component norm: rotating a
    3D dipole residual can change its largest Cartesian component by up to a
    factor of :math:`\\sqrt{3}`, and rotating a traceless quadrupole residual
    in a 2D plane can change its largest orthonormal component by up to a
    factor of :math:`\\sqrt{2}` (:math:`\\sqrt{6}` in 3D). Consequently, while
    every fixed-iteration state and mixer step in potential mode is
    rotationally equivariant to machine precision, ``x_atol_max`` can cause two
    orientations to stop one iteration apart when ``l2 < x_atol`` is already
    satisfied and the dominant multipole residual lies within that geometric
    factor of ``x_atol_max``.
    """

    x_atol: float
    x_atol_max: float

    @staticmethod
    def residual(
        delta: Tensor,
        projection: Callable[[Tensor], Tensor] | None = None,
    ) -> SCFResidual:
        """Measure the raw fixed-point delta, optionally after projection."""
        if projection is not None:
            delta = projection(delta)
        return SCFResidual(
            l2=torch.linalg.vector_norm(delta),
            max_abs=delta.abs().amax(),
        )

    def converged(self, residual: SCFResidual) -> Tensor:
        """Apply the unchanged strict L2 and max-absolute predicate."""
        return (residual.l2 < self.x_atol) & (
            residual.max_abs < self.x_atol_max
        )
