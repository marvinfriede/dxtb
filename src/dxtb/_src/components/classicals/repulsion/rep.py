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
Repulsion: Classes
==================

This module implements the classical repulsion energy term using ordinary
PyTorch operations, so reverse- and forward-mode derivatives use one path.

Note
----
The Repulsion class has a cache scope that goes beyond single-point
calculations (geometry optimization, numerical gradients). The atomic numbers
are set upon instantiation (``numbers`` is a property), and the parameters in
the cache are created for only those atomic numbers. The positions, however,
must be supplied to the ``get_energy`` method. Hence, the cache does not become
invalid for different geometries, but only for different atomic numbers.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Any, Tensor, override

from .base import (
    BaseRepulsion,
    BaseRepulsionCache,
    repulsion_energy,
)

__all__ = ["LABEL_REPULSION", "Repulsion"]


LABEL_REPULSION = "Repulsion"
"""
Label for the :class:`.Repulsion` component, coinciding with the class name.
"""


class Repulsion(BaseRepulsion):
    """
    Representation of the classical repulsion.
    """

    @override
    def get_energy(
        self, positions: Tensor, cache: BaseRepulsionCache, **kwargs: Any
    ) -> Tensor:
        """
        Get repulsion energy.

        Parameters
        ----------
        cache : Repulsion.Cache
            Cache for repulsion.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        atom_resolved : bool
            Whether to return atom-resolved energy (True) or full matrix
            (False).

        Returns
        -------
        Tensor
            (Atom-resolved) repulsion energy.
        """
        e = repulsion_energy(
            positions,
            cache.mask,
            cache.arep,
            cache.kexp,
            cache.zeff,
            self.cutoff,
        )

        if kwargs.get("atom_resolved", True) is True:
            return 0.5 * torch.sum(e, dim=-1)
        return e
