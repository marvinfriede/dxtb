# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
"""
Dispersion Model Base Classes
=============================

Abstract base class and property containers for all DFT-D dispersion models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tad_mctc import Structure
from tad_mctc.tree import Node
from tad_mctc.typing import Tensor

from ..cutoff import Cutoff
from ..pairs import Pairs

__all__ = ["DispersionModel", "DispersionProperties", "ModelProperties"]


@dataclass(frozen=True)
class DispersionProperties:
    """
    Physical properties evaluated by a dispersion model (matching
    ``get_properties`` in Grimme-lab ``dftd/disp.f90``).

    Attributes
    ----------
    cn : Tensor
        Atomic coordination numbers of shape ``(..., nat)`` (zeros for models
        without coordination numbers).
    q : Tensor | None
        Atomic partial charges of shape ``(..., nat)``, or ``None`` for
        charge-independent models.
    c6 : Tensor
        Pairwise :math:`C_6` dispersion coefficients of shape ``(..., nat, nat)``.
    c8 : Tensor | None
        Pairwise :math:`C_8` dispersion coefficients of shape ``(..., nat, nat)``.
    alphadd : Tensor | None
        Static dipole-dipole polarizabilities :math:`\alpha_{dd}(0)` of shape
        ``(..., nat)``, or ``None`` if not available for the model.
    alphaqq : Tensor | None
        Static quadrupole-quadrupole polarizabilities :math:`\alpha_{qq}(0)` of
        shape ``(..., nat)``, or ``None`` if not available for the model.
    """

    cn: Tensor
    c6: Tensor
    c8: Tensor | None = None
    q: Tensor | None = None
    alphadd: Tensor | None = None
    alphaqq: Tensor | None = None


ModelProperties = DispersionProperties


class DispersionModel(Node):
    """
    Base class for DFT-D dispersion models.
    """

    def get_properties(
        self,
        structure: Structure,
        pairs: Pairs | None = None,
        *,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> DispersionProperties:
        """
        Evaluate coordination numbers, partial charges, pairwise :math:`C_6` and
        :math:`C_8` coefficients, and static polarizabilities for ``structure``.
        """
        raise NotImplementedError
