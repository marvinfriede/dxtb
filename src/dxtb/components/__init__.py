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
Components
==========

The tight-binding components are the central building blocks of each Model.

The components are split into :class:`~dxtb.components.base.Interaction` and
:class:`~dxtb.components.base.Classical` components. The
:class:`~dxtb.components.base.Interaction` components define the Hamiltonian
and, correspondingly, depend on the SCF charges and enter the SCF loop. The
:class:`~dxtb.components.base.Classical` components are classical corrections
that do not depend on the SCF charges.

The built-in components are initialized automatically from a parametrization
and the atomic numbers of the Structure within :class:`~dxtb.Calculator` and
:class:`~dxtb.calculators.Model` (which also accept custom
:class:`~dxtb.components.base.Classical` and
:class:`~dxtb.components.base.Interaction` subclasses). For explicit
instantiation of built-in components, factory functions are provided.

.. code-block:: python

    import torch
    from dxtb import GFN1_XTB
    from dxtb.components.dispersion import new_dispersion

    numbers = torch.tensor([3, 1])
    disp = new_dispersion(numbers, GFN1_XTB)

    print(disp.label)  # DispersionD3
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dxtb.components import base as base
    from dxtb.components import coulomb as coulomb
    from dxtb.components import dispersion as dispersion
    from dxtb.components import field as field
    from dxtb.components import halogen as halogen
    from dxtb.components import ies as ies
    from dxtb.components import repulsion as repulsion
    from dxtb.components import shortrangebond as shortrangebond
    from dxtb.components import solvation as solvation
else:
    import dxtb._src.loader.lazy as _lazy

    __getattr__, __dir__, __all__ = _lazy.attach_module(
        __name__,
        [
            "base",
            "coulomb",
            "field",
            "solvation",
            #
            "dispersion",
            "halogen",
            "ies",
            "repulsion",
            "shortrangebond",
        ],
    )
    del _lazy

del TYPE_CHECKING
