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
Repulsion
=========

This module implements the classical repulsion energy term.

Note
----
Element pair parameters are resolved from unique atomic numbers, and
composition-dependent setup data is built via ``get_cache``. Positions are
supplied to ``get_energy`` (or ``get_gradient``).

Example
-------

.. code-block:: python

    import torch
    from dxtb import GFN1_XTB, IndexHelper
    from dxtb.components.repulsion import new_repulsion

    # Define atomic numbers and positions
    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [+0.00000000000000, +0.00000000000000, +0.00000000000000],
        [+1.61768389755830, +1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [+1.61768389755830, -1.61768389755830, +1.61768389755830],
        [-1.61768389755830, +1.61768389755830, +1.61768389755830],
    ])

    # Initialize the repulsion and IndexHelper objects
    rep = new_repulsion(torch.unique(numbers), GFN1_XTB)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # Get the cache and calculate the energy
    cache = rep.get_cache(numbers, ihelp)
    energy = rep.get_energy(positions, cache)

    # Output the summed energy across all atoms
    print(energy.sum(-1))
"""

from .base import (
    RepulsionSetup,
    setup_repulsion,
)
from .factory import new_repulsion
from .rep import LABEL_REPULSION, Repulsion

__all__ = [
    "LABEL_REPULSION",
    "new_repulsion",
    "Repulsion",
    "RepulsionSetup",
    "setup_repulsion",
]
