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
Integral container
==================

Immutable container for the integral matrices of one evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from dxtb._src.typing import Tensor

__all__ = ["IntegralMatrices"]


@dataclass(frozen=True, eq=False)
class IntegralMatrices:
    """Matrices produced by one electronic integral evaluation.

    Notes
    -----
    ``frozen=True`` prevents reassigning dataclass attributes on an instance;
    it does not prevent in-place mutation of the underlying ``torch.Tensor``
    storage.

    In the internal evaluation pipeline (``build_integral_matrices`` and
    ``Result.integrals``), ``quadrupole`` holds the 6 ket-centered, traceless
    lower-triangular components with shape ``(..., 6, nao, nao)``. By contrast,
    the standalone wrapper :func:`dxtb.integrals.wrappers.quadint` returns the
    9 origin-centered Cartesian components with shape ``(..., 9, nao, nao)``.

    Parameters
    ----------
    hcore : Tensor
        Core Hamiltonian matrix of shape ``(..., nao, nao)``.
    overlap : Tensor
        Normalized overlap matrix of shape ``(..., nao, nao)``.
    dipole : Tensor | None, optional
        Ket-centered dipole integral tensor of shape ``(..., 3, nao, nao)``.
    quadrupole : Tensor | None, optional
        Ket-centered traceless quadrupole integral tensor of shape
        ``(..., 6, nao, nao)``.
    """

    hcore: Tensor
    overlap: Tensor
    dipole: Tensor | None = None
    quadrupole: Tensor | None = None
