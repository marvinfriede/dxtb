# This file is part of tad-dftd.
#
# SPDX-Identifier: Apache-2.0 AND LGPL-3.0-or-later
# Copyright (C) 2024 Grimme Group
#
# Derived from tad-dftd3 / tad-dftd4 (Apache-2.0) and grimme-lab/dftd
# (LGPL-3.0-or-later). See PROVENANCE.md, LICENSE, COPYING, and COPYING.LESSER.
r"""
Real-space cutoffs
==================

One real-space cutoff per part of a dispersion model, in Bohr, plus optional
quintic :math:`C^2` smooth switching widths ``width2`` and ``width3``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real

import torch
from tad_mctc.typing import Tensor

from . import defaults

__all__ = ["Cutoff", "smooth_cutoff"]


@dataclass(frozen=True)
class Cutoff:
    """
    Real-space cutoffs for the individual parts of a dispersion model, in Bohr.

    Plain floats, like the ``cutoff`` of the ``tad-mctc`` coordination number
    models: they only mask distances, so they are constants to
    :func:`torch.compile`, :func:`torch.vmap` and :func:`torch.func.jacrev`
    and need no device or dtype. Any real number, including NumPy scalars,
    is converted to :class:`float`; tensors are rejected.
    """

    cn: float = defaults.D3_CN_CUTOFF
    """Coordination number cutoff."""

    disp2: float = defaults.D3_DISP2_CUTOFF
    """Two-body dispersion interaction cutoff."""

    disp3: float = defaults.D3_DISP3_CUTOFF
    """Three-body dispersion interaction cutoff."""

    cn_eeq: float = defaults.D4_CN_EEQ_CUTOFF
    """EEQ charge-model coordination number cutoff."""

    width2: float = defaults.D3_DISP2_WIDTH
    """
    Width of the smooth two-body cutoff. Zero (default) is a hard cutoff;
    a width of at least `disp2` is treated as zero, as in s-dftd3.
    """

    width3: float = defaults.D3_DISP3_WIDTH
    """
    Width of the smooth three-body cutoff, like `width2` for `disp3`.
    """

    def __post_init__(self) -> None:
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"Cutoff '{name}' must be a plain number, not "
                    f"'{type(value).__name__}'."
                )
            object.__setattr__(self, name, float(value))

        for name in ("width2", "width3"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"Cutoff '{name}' must not be negative.")

    @classmethod
    def from_model(
        cls,
        model: str = "d3",
        *,
        cn: float | None = None,
        disp2: float | None = None,
        disp3: float | None = None,
        cn_eeq: float | None = None,
        width2: float = defaults.D3_DISP2_WIDTH,
        width3: float = defaults.D3_DISP3_WIDTH,
    ) -> Cutoff:
        """
        Construct default cutoffs for a specific dispersion model matching
        Grimme-lab ``dftd/cutoff.f90``.
        """
        m = model.lower().strip()
        if m in ("d1", "d2"):
            def_disp2 = defaults.D1_DISP2_CUTOFF
            def_disp3 = defaults.D3_DISP3_CUTOFF
            def_cn = defaults.D3_CN_CUTOFF
            def_cn_eeq = defaults.D4_CN_EEQ_CUTOFF
        elif m in ("d3", "d3s"):
            def_disp2 = defaults.D3_DISP2_CUTOFF
            def_disp3 = defaults.D3_DISP3_CUTOFF
            def_cn = defaults.D3_CN_CUTOFF
            def_cn_eeq = defaults.D4_CN_EEQ_CUTOFF
        elif m in ("d4", "d4s", "d4srev"):
            def_disp2 = defaults.D4_DISP2_CUTOFF
            def_disp3 = defaults.D4_DISP3_CUTOFF
            def_cn = defaults.D4_CN_CUTOFF
            def_cn_eeq = defaults.D4_CN_EEQ_CUTOFF
        elif m in ("d5", "dft-d5", "dftd5"):
            raise NotImplementedError(
                "Dispersion model 'D5' is not yet implemented because the "
                "upstream reference is not finalized."
            )
        else:
            raise ValueError(f"Unknown dispersion model '{model}'.")

        return cls(
            cn=def_cn if cn is None else cn,
            disp2=def_disp2 if disp2 is None else disp2,
            disp3=def_disp3 if disp3 is None else disp3,
            cn_eeq=def_cn_eeq if cn_eeq is None else cn_eeq,
            width2=width2,
            width3=width3,
        )


def smooth_cutoff(distance: Tensor, cutoff: float, width: float) -> Tensor:
    r"""
    Smooth switch-off of a contribution at the real-space `cutoff`, as in
    s-dftd3's ``smooth_cutoff``: 1 up to ``cutoff - width``, 0 from `cutoff`,
    and the quintic :math:`x^3 (10 - 15 x + 6 x^2)` of
    :math:`x = (\text{cutoff} - r) / \text{width}` in between.
    """
    if width <= 0.0 or width >= cutoff:
        return torch.ones_like(distance)

    x = torch.clamp((cutoff - distance) / width, min=0.0, max=1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))
