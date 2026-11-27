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
(or a padded batch of Structures). Every function takes a
:class:`tad_mctc.Structure`; its ``uhf`` field does not enter the integrals,
and periodic structures (``lattice`` set) are not supported.

Conventions (the same for both drivers):

- The multipole operators are centred on the origin ``(0, 0, 0)``, not on
  the ket as inside the xTB methods.
- The dipole integral has 3 components, the quadrupole integral all 9
  Cartesian components (``xx, xy, xz, yx, ...``), not traceless.
- Every shell is normalized to unit self-overlap, so the overlap matrix has
  unit diagonal.

.. warning::

   **Breaking change**: The ``normalize`` keyword argument (including
   ``normalize=False``) has been removed from :func:`overlap`, :func:`dipint`,
   and :func:`quadint`. Contracted Gaussian shells are now always normalized to
   unit self-overlap at construction across both backends. Internal xTB
   evaluations (:func:`~dxtb._src.integral.evaluation.evaluate_integrals`)
   retain ket-centered 3-component dipole and 6-component traceless quadrupole
   conventions, whereas these standalone wrappers always return origin-centered
   3-component dipole and 9-component Cartesian quadrupole matrices.

Example
-------

.. code-block:: python

    import torch
    from tad_mctc import Structure

    from dxtb import GFN1_XTB as par
    from dxtb.integrals.wrappers import dipint, overlap, quadint

    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, 0.00000000000000, 0.00000000000000],
        [1.61768389755830, 1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [1.61768389755830, -1.61768389755830, 1.61768389755830],
        [-1.61768389755830, 1.61768389755830, 1.61768389755830],
    ])
    structure = Structure(numbers=numbers, positions=positions)

    s = overlap(structure, par)
    print(s.shape)  # Output: torch.Size([17, 17])

    d = dipint(structure, par)
    print(d.shape)  # Output: torch.Size([3, 17, 17])

    q = quadint(structure, par)
    print(q.shape)  # Output: torch.Size([9, 17, 17])
"""

from __future__ import annotations

import torch
from tad_mctc import Structure
from tad_mctc.batch import pack

from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.param import Param, ParamModule
from dxtb._src.typing import DD, Callable, Literal, Tensor

from .driver import libcint, pytorch
from .evaluation import setup_integral_evaluation

__all__ = ["dipint", "hcore", "overlap", "quadint"]

_Kind = Literal["overlap", "dipole", "quadrupole"]

_BUILDERS: dict[tuple[int, _Kind], Callable[..., Tensor]] = {
    (labels.INTDRIVER_PYTORCH, "overlap"): pytorch.build_overlap,
    (labels.INTDRIVER_PYTORCH, "dipole"): pytorch.build_dipole,
    (labels.INTDRIVER_PYTORCH, "quadrupole"): pytorch.build_quadrupole,
    (labels.INTDRIVER_LIBCINT, "overlap"): libcint.build_overlap,
    (labels.INTDRIVER_LIBCINT, "dipole"): libcint.build_dipole,
    (labels.INTDRIVER_LIBCINT, "quadrupole"): libcint.build_quadrupole,
}


def overlap(
    structure: Structure,
    par: Param | ParamModule,
    *,
    driver: int | None = None,
) -> Tensor:
    """
    Overlap integral.

    Parameters
    ----------
    structure : Structure
        One Structure or a padded batch of Structures. Padding atoms (atomic
        number zero) may sit anywhere in a Lane.
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

    Raises
    ------
    TypeError
        ``structure`` is not a :class:`tad_mctc.Structure`.
    ValueError
        Unknown driver, periodic structure, or invalid atomic numbers or
        positions.
    """
    return _integral("overlap", structure, par, driver)


def dipint(
    structure: Structure,
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
    return _integral("dipole", structure, par, driver)


def quadint(
    structure: Structure,
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
    return _integral("quadrupole", structure, par, driver)


def hcore(
    structure: Structure,
    par: Param | ParamModule,
    *,
    driver: int | None = None,
) -> Tensor:
    """
    Core Hamiltonian of a single Structure.

    Parameters
    ----------
    structure : Structure
        One unbatched Structure. Its total charge enters GFN0-xTB; an unset
        charge means neutral.
    par : Param | ParamModule
        Parametrization of an extended tight-binding (xTB) Method.
    driver : int | None, optional
        Integral driver of the overlap (see :func:`overlap`).

    Returns
    -------
    Tensor
        Core Hamiltonian of shape ``(nao, nao)``.
    """
    # pylint: disable=import-outside-toplevel
    from dxtb._src.xtb.h0 import build_hcore, setup_h0

    _validate_structure(structure, allow_batch=False)
    numbers, positions = _extract_active_atoms(
        structure.numbers, structure.positions
    )
    # an unset charge means neutral, as everywhere for a Structure
    charge = 0 if structure.charge is None else structure.charge

    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    pm = par if isinstance(par, ParamModule) else ParamModule(par, **dd)
    ihelp = IndexHelper.from_numbers(numbers, pm)
    # set up H0 first: it validates the method name of the parametrization
    setup = setup_h0(numbers, pm, ihelp, **dd).to(**dd)
    s = overlap(structure, pm, driver=driver)
    return build_hcore(setup, positions, s.to(**dd), charge)[0]


def _validate_structure(structure: Structure, *, allow_batch: bool) -> None:
    """
    Check what the wrappers need beyond the checks run by ``Structure``
    itself (integer numbers, matching shapes, one device).
    """
    if not isinstance(structure, Structure):
        raise TypeError(
            "The integral wrappers take a tad_mctc.Structure, got "
            f"{type(structure).__name__}. Use "
            "Structure(numbers=numbers, positions=positions)."
        )
    if structure.lattice is not None:
        raise ValueError("Periodic structures are not supported.")

    numbers, positions = structure.numbers, structure.positions
    if not positions.is_floating_point():
        raise ValueError(
            f"positions must have a floating-point dtype, got {positions.dtype}."
        )
    if positions.shape[-1] != 3:
        raise ValueError(
            "Expected positions of shape (..., nat, 3), "
            f"got {tuple(positions.shape)}."
        )
    if not allow_batch and numbers.ndim != 1:
        raise ValueError(
            f"Expected an unbatched Structure, got atomic numbers of shape "
            f"{tuple(numbers.shape)}."
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
    structure: Structure,
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

    _validate_structure(structure, allow_batch=True)
    numbers, positions = structure.numbers, structure.positions

    dd: DD = {"device": positions.device, "dtype": positions.dtype}
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
        device=positions.device,
        dtype=positions.dtype,
    )
    assert eval_setup is not None and eval_setup.backend is not None

    build = _BUILDERS[(driver, kind)]
    return build(eval_setup.backend, positions).to(device=positions.device)
