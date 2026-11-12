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
Integrals: Wrappers/Shortcuts
=============================

Stateless convenience functions for the integral matrices of a Structure
(or a padded batch of Structures).

Conventions (the same for both drivers):

- The multipole operators are centred on the origin ``(0, 0, 0)``, not on
  the ket as inside the xTB methods.
- The dipole integral has 3 components, the quadrupole integral all 9
  Cartesian components (``xx, xy, xz, yx, ...``), not traceless.
- Every shell is normalized to unit self-overlap, so the overlap matrix has
  unit diagonal.

Example
-------

.. code-block:: python

    from dxtb.integrals.wrappers import overlap, dipint, quadint
    from dxtb import GFN1_XTB as par
    import torch

    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, 0.00000000000000, 0.00000000000000],
        [1.61768389755830, 1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [1.61768389755830, -1.61768389755830, 1.61768389755830],
        [-1.61768389755830, 1.61768389755830, 1.61768389755830],
    ])

    s = overlap(numbers, positions, par)
    print(s.shape)  # Output: torch.Size([17, 17])

    d = dipint(numbers, positions, par)
    print(d.shape)  # Output: torch.Size([3, 17, 17])

    q = quadint(numbers, positions, par)
    print(q.shape)  # Output: torch.Size([9, 17, 17])
"""

from __future__ import annotations

import torch
from tad_mctc.batch import pack

from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import Literal, Tensor

from .driver.libcint import build_dipole as build_dipole_libcint
from .driver.libcint import build_overlap as build_overlap_libcint
from .driver.libcint import build_quadrupole as build_quadrupole_libcint
from .driver.pytorch import build_dipole as build_dipole_pytorch
from .driver.pytorch import build_overlap as build_overlap_pytorch
from .driver.pytorch import build_quadrupole as build_quadrupole_pytorch
from .evaluation import setup_integral_evaluation

__all__ = ["dipint", "hcore", "overlap", "quadint"]

_Kind = Literal["overlap", "dipole", "quadrupole"]


def overlap(
    numbers: Tensor,
    positions: Tensor,
    par: Param | ParamModule,
    *,
    driver: int | None = None,
) -> Tensor:
    """
    Overlap integral.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the Structure (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    par : Param | ParamModule
        Parametrization of an extended tight-binding (xTB) Method.
    driver : int | None, optional
        Integral driver (:data:`~dxtb.labels.INTDRIVER_LIBCINT` or
        :data:`~dxtb.labels.INTDRIVER_PYTORCH`). Defaults to libcint if it is
        installed, PyTorch otherwise.

    Returns
    -------
    Tensor
        Overlap matrix of shape ``(..., nao, nao)``.
    """
    return _integral("overlap", numbers, positions, par, driver)


def dipint(
    numbers: Tensor,
    positions: Tensor,
    par: Param | ParamModule,
    *,
    driver: int | None = None,
) -> Tensor:
    """
    Dipole integral about the origin.

    Parameters are those of :func:`overlap`.

    Returns
    -------
    Tensor
        Dipole integral of shape ``(..., 3, nao, nao)``.
    """
    return _integral("dipole", numbers, positions, par, driver)


def quadint(
    numbers: Tensor,
    positions: Tensor,
    par: Param | ParamModule,
    *,
    driver: int | None = None,
) -> Tensor:
    """
    Quadrupole integral about the origin (all nine Cartesian components).

    Parameters are those of :func:`overlap`.

    Returns
    -------
    Tensor
        Quadrupole integral of shape ``(..., 9, nao, nao)``.
    """
    return _integral("quadrupole", numbers, positions, par, driver)


def hcore(
    numbers: Tensor,
    positions: Tensor,
    par: Param | ParamModule,
    charge: Tensor | float | int | None = None,
    *,
    driver: int | None = None,
) -> Tensor:
    """
    Core Hamiltonian of a single Structure.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the Structure (shape: ``(nat,)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(nat, 3)``).
    par : Param | ParamModule
        Parametrization of an extended tight-binding (xTB) Method.
    charge : Tensor | float | int | None, optional
        Total charge (required by GFN0-xTB).
    driver : int | None, optional
        Integral driver of the overlap (see :func:`overlap`).

    Returns
    -------
    Tensor
        Core Hamiltonian of shape ``(nao, nao)``.
    """
    # pylint: disable=import-outside-toplevel
    from dxtb._src.xtb.h0 import build_hcore, setup_h0

    _validate_inputs(numbers, positions, allow_batch=False)
    numbers, positions = _extract_active_atoms(numbers, positions)

    dd = {"device": positions.device, "dtype": positions.dtype}
    pm = par if isinstance(par, ParamModule) else ParamModule(par, **dd)
    ihelp = IndexHelper.from_numbers(numbers, pm)
    # set up H0 first: it validates the method name of the parametrization
    setup = setup_h0(numbers, pm, ihelp, **dd)
    s = overlap(numbers, positions, pm, driver=driver)
    return build_hcore(setup, positions, s.to(**dd), charge)[0]


def _validate_inputs(
    numbers: Tensor, positions: Tensor, *, allow_batch: bool
) -> None:
    """Validate atomic numbers and Cartesian coordinates for integral wrappers."""
    if not isinstance(numbers, Tensor) or not isinstance(positions, Tensor):
        raise ValueError(
            "numbers and positions must be torch.Tensor instances."
        )
    if (
        numbers.is_floating_point()
        or numbers.is_complex()
        or numbers.dtype == torch.bool
    ):
        raise ValueError(
            f"numbers must have an integer dtype, got {numbers.dtype}."
        )
    if not positions.is_floating_point():
        raise ValueError(
            f"positions must have a floating-point dtype, got {positions.dtype}."
        )
    allowed_ndims = (1, 2) if allow_batch else (1,)
    if numbers.ndim not in allowed_ndims:
        expected = "1D or 2D" if allow_batch else "1D"
        raise ValueError(
            f"Expected {expected} atomic numbers tensor, "
            f"got shape {tuple(numbers.shape)}."
        )
    if (
        positions.ndim != numbers.ndim + 1
        or positions.shape[:-1] != numbers.shape
        or positions.shape[-1] != 3
    ):
        raise ValueError(
            "Expected positions of shape (*numbers.shape, 3) matching "
            f"numbers shape {tuple(numbers.shape)}, "
            f"got {tuple(positions.shape)}."
        )
    if numbers.numel() == 0:
        raise ValueError("Atomic numbers tensor must not be empty.")
    if bool((numbers < 0).any()):
        raise ValueError("Atomic numbers must be non-negative.")
    if not bool((numbers > 0).any(dim=-1).all()):
        raise ValueError(
            "Every structure must contain at least one positive atomic number."
        )


def _extract_active_atoms(
    numbers: Tensor, positions: Tensor
) -> tuple[Tensor, Tensor]:
    """Select active atoms (Z > 0) and their corresponding coordinates."""
    mask = numbers > 0
    if bool(mask.all()):
        return numbers, positions
    return numbers[mask], positions[mask]


def _integral(
    kind: _Kind,
    numbers: Tensor,
    positions: Tensor,
    par: Param | ParamModule,
    driver: int | None,
) -> Tensor:
    """Build one integral for a Structure or each Structure of a batch."""
    if driver is None:
        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs.available import has_libcint

        driver = (
            labels.INTDRIVER_LIBCINT
            if has_libcint is True
            else labels.INTDRIVER_PYTORCH
        )
    if driver not in (labels.INTDRIVER_LIBCINT, labels.INTDRIVER_PYTORCH):
        raise ValueError(f"Unknown integral driver '{driver}'.")

    _validate_inputs(numbers, positions, allow_batch=True)

    dd = {"device": positions.device, "dtype": positions.dtype}
    pm = par if isinstance(par, ParamModule) else ParamModule(par, **dd)

    if numbers.ndim == 1:
        n_active, p_active = _extract_active_atoms(numbers, positions)
        return _single(kind, n_active, p_active, pm, driver)

    matrices = []
    for batch_idx in range(numbers.shape[0]):
        n_active, p_active = _extract_active_atoms(
            numbers[batch_idx], positions[batch_idx]
        )
        matrices.append(_single(kind, n_active, p_active, pm, driver))
    return pack(matrices)


def _single(
    kind: _Kind,
    numbers: Tensor,
    positions: Tensor,
    par: ParamModule,
    driver: int,
) -> Tensor:
    """Build one integral of a single Structure."""
    ihelp = IndexHelper.from_numbers(numbers, par)
    eval_setup = setup_integral_evaluation(
        numbers,
        par,
        ihelp,
        driver_type=driver,
        intlevel=labels.INTLEVEL_OVERLAP,
        force_cpu_for_libcint=True,
        device=positions.device,
        dtype=positions.dtype,
    )
    assert eval_setup is not None

    if driver == labels.INTDRIVER_PYTORCH:
        assert eval_setup.pytorch is not None
        pt_setup = eval_setup.pytorch
        if kind == "dipole":
            matrix = build_dipole_pytorch(pt_setup, positions)
        elif kind == "quadrupole":
            matrix = build_quadrupole_pytorch(pt_setup, positions)
        else:
            matrix = build_overlap_pytorch(pt_setup, positions)
    else:
        assert eval_setup.libcint is not None
        lc_setup = eval_setup.libcint
        if kind == "dipole":
            matrix = build_dipole_libcint(lc_setup, positions)
        elif kind == "quadrupole":
            matrix = build_quadrupole_libcint(lc_setup, positions)
        else:
            matrix = build_overlap_libcint(lc_setup, positions)

    return matrix.to(device=positions.device)
