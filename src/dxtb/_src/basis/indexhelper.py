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
Basis: IndexHelper
==================

Index helper utility to create index maps between atomic, shell-resolved, and
orbital resolved representations of quantities.

Example
-------

.. code-block:: python

    import torch
    from dxtb import IndexHelper

    # Define atomic numbers and angular momentum for each element
    numbers = torch.tensor([6, 1, 1, 1, 1])
    angular = {1: [0], 6: [0, 1]}

    # Create an IndexHelper instance with angular momentum specifications
    ihelp = IndexHelper.from_numbers_angular(numbers, angular)

    # Count the number of entries in the angular momentum tensor
    result = torch.sum(ihelp.angular >= 0)
    print(result)  # torch.tensor(6)
"""

from __future__ import annotations

import torch
from tad_mctc.batch import pack
from tad_mctc.math import einsum
from tad_mctc.tree import Node, child, context

from dxtb._src.typing import Any, Tensor, override

from ..param import Param, ParamModule
from ..param.element import get_elem_angular
from ..utils import t2int, wrap_gather, wrap_scatter_reduce

__all__ = ["IndexHelper"]


PAD = -999

_INDEX_TENSOR_FIELDS: tuple[str, ...] = (
    "unique_angular",
    "angular",
    "atom_to_unique",
    "ushells_to_unique",
    "ushells_per_unique",
    "shells_to_ushell",
    "shells_per_atom",
    "shell_index",
    "shells_to_atom",
    "orbitals_per_shell",
    "orbital_index",
    "orbitals_to_shell",
)
_ALLOWED_INT_DTYPES: tuple[torch.dtype, ...] = (
    torch.int16,
    torch.int32,
    torch.int64,
)


def _fill(index: Tensor, repeat: Tensor) -> Tensor:
    """
    Fill an index map using index offsets and number of repeats
    """
    index_map = torch.zeros(
        int(torch.sum(repeat).item()), device=index.device, dtype=index.dtype
    )

    for idx, offset, count in zip(torch.arange(index.shape[-1]), index, repeat):
        index_map[offset : offset + count] = idx
    return index_map


def _expand(index: Tensor, repeat: Tensor) -> Tensor:
    """
    Expand an index map using index offsets and number of repeats
    """

    return torch.tensor(
        [
            idx
            for offset, count in zip(index, repeat)
            for idx in torch.arange(offset.item(), (offset + count).item(), 1)
        ],
        device=index.device,
        dtype=index.dtype,
    )


class IndexHelper(Node):
    """
    Index helper for basis set.

    All 12 index tensors are registered as integer pytree children with
    ``keep_dtype=True``. Under the ``tad_mctc.tree.Node`` contract, integer
    children preserve their integer dtype across :meth:`to` and :meth:`type`
    calls, and :attr:`dtype` raises :exc:`AttributeError` because the node
    contains no floating-point tensor leaves.
    """

    unique_angular: Tensor = child(keep_dtype=True)
    """Angular momenta of all unique shells"""

    angular: Tensor = child(keep_dtype=True)
    """Angular momenta for all shells"""

    atom_to_unique: Tensor = child(keep_dtype=True)
    """Mapping of atoms to unique species"""

    ushells_to_unique: Tensor = child(keep_dtype=True)
    """Mapping of unique shells to unique species"""

    ushells_per_unique: Tensor = child(keep_dtype=True)
    """Number of unique shells per unqiue atoms."""

    shells_to_ushell: Tensor = child(keep_dtype=True)
    """Mapping of shells to unique atoms"""

    shells_per_atom: Tensor = child(keep_dtype=True)  # nsh_at
    """Number of shells for each atom"""

    shell_index: Tensor = child(keep_dtype=True)  # ish_at
    """Offset index for starting the next shell block"""

    shells_to_atom: Tensor = child(keep_dtype=True)
    """Mapping of shells to atoms"""

    orbitals_per_shell: Tensor = child(keep_dtype=True)
    """Number of orbitals for each shell"""

    orbital_index: Tensor = child(keep_dtype=True)
    """Offset index for starting the next orbital block"""

    orbitals_to_shell: Tensor = child(keep_dtype=True)
    """Mapping of orbitals to shells"""

    batch_mode: int = context(default=0)
    """
    Whether multiple systems or a single one are handled:

    - 0: Single system
    - 1: Multiple systems with padding
    - 2: Multiple systems with no padding (conformer ensemble)
    """

    def _normalize(self) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if isinstance(self.batch_mode, bool):
            updates["batch_mode"] = int(self.batch_mode)

        first_device: torch.device | None = None

        for name in _INDEX_TENSOR_FIELDS:
            tensor = getattr(self, name)
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"IndexHelper.{name} must be a torch.Tensor, "
                    f"got {type(tensor).__name__}."
                )

            if first_device is None:
                first_device = tensor.device
            elif tensor.device != first_device:
                raise ValueError("All tensors must be on the same device")

        return updates

    def _validate(self) -> None:
        if (
            isinstance(self.batch_mode, bool)
            or not isinstance(self.batch_mode, int)
            or self.batch_mode not in (0, 1, 2)
        ):
            raise ValueError(
                "IndexHelper.batch_mode must be 0, 1, or 2, "
                f"got {self.batch_mode!r}."
            )

        expected_dtype = self.unique_angular.dtype
        expected_device = self.unique_angular.device
        for name in _INDEX_TENSOR_FIELDS:
            tensor = getattr(self, name)
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"IndexHelper.{name} must be a torch.Tensor, "
                    f"got {type(tensor).__name__}."
                )
            if tensor.dtype not in _ALLOWED_INT_DTYPES:
                raise ValueError(
                    f"IndexHelper.{name} must have an integer dtype in "
                    f"{_ALLOWED_INT_DTYPES}, got {tensor.dtype}."
                )
            if tensor.dtype != expected_dtype:
                raise ValueError("All tensors must have same dtype")
            if tensor.device != expected_device:
                raise ValueError("All tensors must be on the same device")
            if tensor.ndim < 1:
                raise ValueError(
                    f"IndexHelper.{name} must be at least 1D, "
                    f"got shape {tuple(tensor.shape)}."
                )

        if self.unique_angular.shape != self.ushells_to_unique.shape:
            raise ValueError(
                "IndexHelper.unique_angular and ushells_to_unique must "
                "have the same shape."
            )
        if (
            self.atom_to_unique.shape != self.shells_per_atom.shape
            or self.atom_to_unique.shape != self.shell_index.shape
        ):
            raise ValueError(
                "IndexHelper atom-level tensors (atom_to_unique, "
                "shells_per_atom, shell_index) must have the same shape."
            )
        if (
            self.angular.shape != self.shells_to_ushell.shape
            or self.angular.shape != self.shells_to_atom.shape
            or self.angular.shape != self.orbitals_per_shell.shape
            or self.angular.shape != self.orbital_index.shape
        ):
            raise ValueError(
                "IndexHelper shell-level tensors (angular, "
                "shells_to_ushell, shells_to_atom, orbitals_per_shell, "
                "orbital_index) must have the same shape."
            )

        if self.batch_mode == 0:
            batch_shape = self.unique_angular.shape[:-1]
            for name in _INDEX_TENSOR_FIELDS:
                tensor = getattr(self, name)
                if tensor.shape[:-1] != batch_shape:
                    raise ValueError(
                        f"IndexHelper.{name} leading dimensions "
                        f"{tuple(tensor.shape[:-1])} do not match "
                        f"{tuple(batch_shape)}."
                    )
        else:
            for name in (
                "unique_angular",
                "ushells_to_unique",
                "ushells_per_unique",
            ):
                tensor = getattr(self, name)
                if tensor.ndim != 1:
                    raise ValueError(
                        f"IndexHelper.{name} must be 1D for batched mode, "
                        f"got shape {tuple(tensor.shape)}."
                    )
            batch_size = self.atom_to_unique.shape[0]
            for name in (
                "angular",
                "atom_to_unique",
                "shells_to_ushell",
                "shells_per_atom",
                "shell_index",
                "shells_to_atom",
                "orbitals_per_shell",
                "orbital_index",
                "orbitals_to_shell",
            ):
                tensor = getattr(self, name)
                if tensor.ndim != 2 or tensor.shape[0] != batch_size:
                    raise ValueError(
                        f"IndexHelper.{name} must be 2D with leading batch "
                        f"size {batch_size}, got shape {tuple(tensor.shape)}."
                    )

    @classmethod
    def from_numbers(
        cls,
        numbers: Tensor,
        par: Param | ParamModule,
        batch_mode: int | None = None,
        move_to_numbers_device: bool = True,
    ) -> IndexHelper:
        """
        Construct an index helper instance from atomic numbers and a
        parametrization.

        Note that this always runs on CPU to avoid inefficient communication
        between devices. Only the resulting tensors are transfered to the GPU.
        This is necessary because of complex data look up that is not
        vectorizable and requires native for-loops. Furthermore, the method
        frequently uses the :meth:`torch.Tensor.item` method, which forces
        CPU-GPU synchronization because it converts a GPU tensor to a Python
        scalar.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        par : Param
            Representation of an extended tight-binding model.
        batch_mode : int
            Whether multiple systems or a single one are handled:

            - 0: Single system
            - 1: Multiple systems with padding
            - 2: Multiple systems with no padding (conformer ensemble)
        move_to_numbers_device : bool
            Move the resulting tensors to the device of the ``numbers`` tensor.
            This should be switched off for GPU calculations that use `libcint`
            for integrals as the :class:`.IndexHelper` has to be on the CPU
            for this step.

        Returns
        -------
        IndexHelper
            Instance of index helper for given basis set.
        """
        # PERF: Only the angular momenta (plain Python ints) are required
        # here. Hence, a plain `Param` is read directly instead of building
        # the full differentiable parameter tree (~0.1 s).
        return cls.from_numbers_angular(
            numbers,
            angular=get_elem_angular(par.element),
            batch_mode=batch_mode,
            move_to_numbers_device=move_to_numbers_device,
        )

    @classmethod
    def from_numbers_angular(
        cls,
        numbers: Tensor,
        angular: dict[int, list[int]],
        batch_mode: int | None = None,
        move_to_numbers_device: bool = True,
    ) -> IndexHelper:
        """
        Construct an index helper instance from atomic numbers and their
        angular momenta. If you are not sure about the angular momenta, use
        :meth:`.IndexHelper.from_numbers` instead, which simply takes a
        parametrization.

        Note that this always runs on CPU to avoid inefficient communication
        between devices. Only the resulting tensors are transfered to the GPU.
        This is necessary because of complex data look up that is not
        vectorizable and requires native for-loops. Furthermore, the method
        frequently uses the :meth:`torch.Tensor.item` method, which forces
        CPU-GPU synchronization because it converts a GPU tensor to a Python
        scalar.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        angular : dict[int, Tensor]
            Map between atomic numbers and angular momenta of all shells.
        batch_mode : int
            Whether multiple systems or a single one are handled:

            - 0: Single system
            - 1: Multiple systems with padding
            - 2: Multiple systems with no padding (conformer ensemble)
        move_to_numbers_device : bool
            Move the resulting tensors to the device of the ``numbers`` tensor.
            This should be switched off for GPU calculations that use `libcint`
            for integrals as the :class:`.IndexHelper` has to be on the CPU
            for this step.

        Returns
        -------
        IndexHelper
            Instance of index helper for given basis set.
        """
        cpu = torch.device("cpu")
        device = numbers.device if move_to_numbers_device else cpu

        # Ensure that all tensors are moved to CPU to avoid inefficient
        # memory transfers between devices (.item() and native for-loops).
        numbers = numbers.to(cpu)

        if batch_mode is None:
            batch_mode = int(numbers.ndim > 1)
        else:
            batch_mode = int(batch_mode)

        unique, atom_to_unique = torch.unique(numbers, return_inverse=True)

        unique_angular = torch.tensor(
            [l for number in unique for l in angular.get(number.item(), [-1])],
            device=cpu,
        )

        # note that padding (i.e., when number = 0) is assigned one shell
        ushells_per_unique = torch.tensor(
            [len(angular.get(number.item(), [-1])) for number in unique],
            device=cpu,
        )

        ushell_index = (
            torch.cumsum(ushells_per_unique, dim=-1) - ushells_per_unique
        )
        ushells_to_unique = _fill(ushell_index, ushells_per_unique)

        if batch_mode > 0:
            # remove the single shell assigned to the padding value in order to
            # avoid an additional count in the expansion as this will cause
            # errors in certain situations
            # (see https://github.com/grimme-lab/dxtb/issues/67)
            if (unique == 0.0).any():
                ushells_per_unique[0] = 0

            shells_to_ushell = pack(
                [
                    _expand(
                        ushell_index[atom_to_unique[_batch, :]],
                        ushells_per_unique[atom_to_unique[_batch, :]],
                    )
                    for _batch in range(numbers.shape[0])
                ],
                value=-1,
            )
        else:
            shells_to_ushell = _expand(
                ushell_index[atom_to_unique],
                ushells_per_unique[atom_to_unique],
            )

        shells_per_atom = ushells_per_unique[atom_to_unique]
        shell_index = torch.cumsum(shells_per_atom, -1) - shells_per_atom
        shell_index[shells_per_atom == 0] = PAD

        if batch_mode > 0:
            shells_to_atom = pack(
                [
                    _fill(shell_index[_batch, :], shells_per_atom[_batch, :])
                    for _batch in range(numbers.shape[0])
                ],
                value=PAD,
            )
        else:
            shells_to_atom = _fill(shell_index, shells_per_atom)

        lsh = torch.where(
            shells_to_ushell >= 0,
            unique_angular[shells_to_ushell],
            PAD,
        )

        orbitals_per_shell = torch.where(
            lsh >= 0, 2 * lsh + 1, torch.tensor(0, device=cpu)
        )

        orbital_index = (
            torch.cumsum(orbitals_per_shell, -1) - orbitals_per_shell
        )
        orbital_index[orbitals_per_shell == 0] = PAD

        if batch_mode > 0:
            orbitals_to_shell = pack(
                [
                    _fill(
                        orbital_index[_batch, :], orbitals_per_shell[_batch, :]
                    )
                    for _batch in range(numbers.shape[0])
                ],
                value=PAD,
            )
        else:
            orbitals_to_shell = _fill(orbital_index, orbitals_per_shell)

        return cls(
            unique_angular=unique_angular.to(device),
            angular=lsh.to(device),
            atom_to_unique=atom_to_unique.to(device),
            ushells_to_unique=ushells_to_unique.to(device),
            ushells_per_unique=ushells_per_unique.to(device),
            shells_to_ushell=shells_to_ushell.to(device),
            shells_per_atom=shells_per_atom.to(device),
            shell_index=shell_index.to(device),
            shells_to_atom=shells_to_atom.to(device),
            orbitals_per_shell=orbitals_per_shell.to(device),
            orbital_index=orbital_index.to(device),
            orbitals_to_shell=orbitals_to_shell.to(device),
            batch_mode=batch_mode,
        )

    def reduce_orbital_to_shell(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        """
        Reduce orbital-resolved tensor to shell-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Orbital-resolved tensor.
        dim : int | (int, int)
            Dimension to reduce over, defaults to -1.
        reduce : str
            Reduction method, defaults to "sum".
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Shell-resolved tensor.
        """

        shell_count = self.shells_to_atom.shape[-1]
        output_size = (
            (shell_count, shell_count)
            if isinstance(dim, tuple)
            else shell_count
        )
        return wrap_scatter_reduce(
            x,
            dim,
            self.orbitals_to_shell,
            reduce,
            extra=extra,
            output_size=output_size,
        )

    def reduce_shell_to_atom(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        """
        Reduce shell-resolved tensor to atom-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Shell-resolved tensor
        dim : int | (int, int)
            Dimension to reduce over, defaults to -1.
        reduce : str
            Reduction method, defaults to "sum".
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Atom-resolved tensor.
        """

        atom_count = self.atom_to_unique.shape[-1]
        output_size = (
            (atom_count, atom_count) if isinstance(dim, tuple) else atom_count
        )
        return wrap_scatter_reduce(
            x,
            dim,
            self.shells_to_atom,
            reduce,
            extra=extra,
            output_size=output_size,
        )

    def reduce_orbital_to_atom(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        reduce: str = "sum",
        extra: bool = False,
    ) -> Tensor:
        """
        Reduce orbital-resolved tensor to atom-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Orbital-resolved tensor.
        dim : int | (int, int)
            Dimension to reduce over, defaults to -1.
        reduce : str
            Reduction method, defaults to "sum".
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Atom-resolved tensor.
        """

        return self.reduce_shell_to_atom(
            self.reduce_orbital_to_shell(
                x, dim=dim, reduce=reduce, extra=extra
            ),
            dim=dim,
            reduce=reduce,
            extra=extra,
        )

    def spread_atom_to_shell(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        """
        Spread atom-resolved tensor to shell-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Atom-resolved tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Shell-resolved tensor.
        """

        return wrap_gather(x, dim, self.shells_to_atom, extra=extra)

    def spread_shell_to_orbital(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        """
        Spread shell-resolved tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Shell-resolved tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return wrap_gather(x, dim, self.orbitals_to_shell, extra=extra)

    def spread_shell_to_orbital_cart(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        """
        Spread shell-resolved tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Shell-resolved tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return wrap_gather(x, dim, self.orbitals_to_shell_cart, extra=extra)

    def spread_atom_to_orbital(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        """
        Spread atom-resolved tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Atom-resolved tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return self.spread_shell_to_orbital(
            self.spread_atom_to_shell(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    def spread_atom_to_orbital_cart(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread atom-resolved tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Atom-resolved tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return self.spread_shell_to_orbital_cart(
            self.spread_atom_to_shell(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    def spread_uspecies_to_atom(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread unique species tensor to atom-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Unique specie tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Atom-resolved tensor.
        """

        return wrap_gather(x, dim, self.atom_to_unique, extra=extra)

    def spread_uspecies_to_shell(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread unique species tensor to shell-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Unique specie tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Shell-resolved tensor.
        """

        return self.spread_atom_to_shell(
            self.spread_uspecies_to_atom(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    def spread_uspecies_to_orbital(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread unique species tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Unique specie tensor.
        dim : int
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return self.spread_atom_to_orbital(
            self.spread_uspecies_to_atom(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    def spread_uspecies_to_orbital_cart(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread unique species tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Unique specie tensor.
        dim : int
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return self.spread_atom_to_orbital_cart(
            self.spread_uspecies_to_atom(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    def spread_ushell_to_shell(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread unique shell tensor to shell-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Unique shell tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
            Tensor to reduce contains a extra dimension of arbitrary size.
            Defaults to ``False``.

        Returns
        -------
        Tensor
            Shell-resolved tensor.
        """

        return wrap_gather(x, dim, self.shells_to_ushell, extra=extra)

    def spread_ushell_to_orbital(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread unique shell tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Unique shell tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
             Tensor to reduce contains a extra dimension of arbitrary size.
             Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return self.spread_shell_to_orbital(
            self.spread_ushell_to_shell(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    def spread_ushell_to_orbital_cart(
        self, x: Tensor, dim: int | tuple[int, int] = -1, extra: bool = False
    ) -> Tensor:
        """
        Spread unique shell tensor to orbital-resolved tensor.

        Parameters
        ----------
        x : Tensor
            Unique shell tensor.
        dim : int | (int, int)
            Dimension to spread over, defaults to -1.
        extra : bool
             Tensor to reduce contains a extra dimension of arbitrary size.
             Defaults to ``False``.

        Returns
        -------
        Tensor
            Orbital-resolved tensor.
        """

        return self.spread_shell_to_orbital_cart(
            self.spread_ushell_to_shell(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    @property
    def orbitals_to_atom(self) -> Tensor:
        return self._orbitals_to_atom()

    def _orbitals_to_atom(self) -> Tensor:
        return self.spread_shell_to_orbital(self.shells_to_atom)

    @property
    def orbitals_per_shell_cart(self) -> Tensor:
        return self._orbitals_per_shell_cart()

    def _orbitals_per_shell_cart(self) -> Tensor:
        l = self.angular
        ls = torch.div((l + 1) * (l + 2), 2, rounding_mode="floor")
        return torch.where(l >= 0, ls, torch.tensor(0, device=self.device))

    @property
    def orbital_index_cart(self) -> Tensor:
        return self._orbital_index_cart()

    def _orbital_index_cart(self) -> Tensor:
        orb_per_shell = self.orbitals_per_shell_cart
        return torch.cumsum(orb_per_shell, -1) - orb_per_shell

    @property
    def orbitals_to_shell_cart(self) -> Tensor:
        return self._orbitals_to_shell_cart()

    def _orbitals_to_shell_cart(self) -> Tensor:
        orbital_index = self.orbital_index_cart
        orbitals_per_shell = self.orbitals_per_shell_cart
        if self.batch_mode > 0:
            orbitals_to_shell = pack(
                [
                    _fill(
                        orbital_index[_batch, :], orbitals_per_shell[_batch, :]
                    )
                    for _batch in range(self.angular.shape[0])
                ],
                value=PAD,
            )
        else:
            orbitals_to_shell = _fill(orbital_index, orbitals_per_shell)

        return orbitals_to_shell

    @property
    def orbitals_to_atom_cart(self) -> Tensor:
        return self._orbitals_to_atom_cart()

    def _orbitals_to_atom_cart(self) -> Tensor:
        return self.spread_shell_to_orbital_cart(self.shells_to_atom)

    def get_shell_indices(self, atom_idx: int) -> Tensor:
        """
        Get shell indices belong to given atom.

        Parameters
        ----------
        atom_idx : int
            Index of given atom.

        Returns
        -------
        Tensor
            Index list of shells belonging to given atom.
        """
        return (self.shells_to_atom == atom_idx).nonzero(as_tuple=True)[0]

    def get_orbital_indices(self, shell_idx: int) -> Tensor:
        """
        Get orbital indices belong to given shell.

        Parameters
        ----------
        shell_idx : int
            Index of given shell.

        Returns
        -------
        Tensor
            Index list of orbitals belonging to given shell.
        """
        return (self.orbitals_to_shell == shell_idx).nonzero(as_tuple=True)[0]

    def orbital_atom_mapping(self, idx: int) -> Tensor:
        """
        Mapping of atom index to orbital index, i.e., return indices of orbitals
        belonging to given atom. The orbital order is given by
        :meth:`.IndexHelper.orbitals_to_shell`.

        Parameters
        ----------
        idx : int
            Index of target atom.

        Returns
        -------
        Tensor
            1d-Tensor containing the indices of the orbitals.
        """
        # FIXME: batched mode
        if self.batch_mode > 0:
            raise NotImplementedError(
                "Currently, `orbital_atom_mapping` only supports a single sample."
            )

        return torch.tensor(
            [
                oidx
                for sidx in self.get_shell_indices(idx)
                for oidx in self.get_orbital_indices(t2int(sidx)).tolist()
            ]
        )

    @property
    def orbitals_per_atom(self) -> Tensor:
        """
        Number of orbitals for each atom.

        Returns
        -------
        Tensor
            Atom indices for each orbital.
        """

        try:
            # batch mode
            pad = torch.nn.utils.rnn.pad_sequence(
                [self.shells_to_atom.mT, self.orbitals_to_shell.T],
                padding_value=PAD,
            )
            pad = einsum("ijk->kji", pad)  # [2, bs, norb_max]
        except RuntimeError:
            # single mode
            pad = torch.nn.utils.rnn.pad_sequence(
                [self.shells_to_atom, self.orbitals_to_shell], padding_value=PAD
            ).T  # [2, norb_max]

        if len(pad.shape) > 2:
            # gathering over subentries to avoid padded value (PAD) in index tensor
            return pack(
                [torch.gather(a[b != PAD], 0, b[b != PAD]) for a, b in pad],
                value=PAD,
            )
            # TODO:
            # masked_tensor could be a vectorised solution (though only
            # available in pytorch 1.13)
            # alternatively write all values into extra column
        else:
            return torch.gather(pad[0], 0, pad[1])

    @property
    def nat(self) -> int:
        return self.atom_to_unique.shape[-1]

    @property
    def nsh(self) -> int:
        return int(self.shells_per_atom.sum(-1).max())

    @property
    def nao(self) -> int:
        return int(self.orbitals_per_shell.sum(-1).max())

    @property
    def nbatch(self) -> int | None:
        return self.atom_to_unique.ndim if self.batch_mode > 0 else None

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"IndexHelper(\n"
            f"  unique_angular={self.unique_angular},\n"
            f"  angular={self.angular},\n"
            f"  atom_to_unique={self.atom_to_unique},\n"
            f"  ushells_to_unique={self.ushells_to_unique},\n"
            f"  ushells_per_unique={self.ushells_per_unique},\n"
            f"  shells_to_ushell={self.shells_to_ushell},\n"
            f"  shells_per_atom={self.shells_per_atom},\n"
            f"  shell_index={self.shell_index},\n"
            f"  shells_to_atom={self.shells_to_atom},\n"
            f"  orbitals_per_shell={self.orbitals_per_shell},\n"
            f"  orbital_index={self.orbital_index},\n"
            f"  orbitals_to_shell={self.orbitals_to_shell},\n"
            f"  batch_mode={self.batch_mode},\n"
            f"  device={self.device},\n"
            f"  dtype={self.unique_angular.dtype}\n"
            ")"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


class IndexHelperGFN1(IndexHelper):
    """
    Index helper for GFN1 basis set.
    """

    @override
    @classmethod
    def from_numbers(
        cls,
        numbers: Tensor,
        batch_mode: int | None = None,
        move_to_numbers_device: bool = True,
    ) -> IndexHelper:
        """
        Construct an index helper instance from atomic numbers and their
        angular momenta. The latter are collected from the GFN1 parametrization.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        batch_mode : int
            Whether multiple systems or a single one are handled:

            - 0: Single system
            - 1: Multiple systems with padding
            - 2: Multiple systems with no padding (conformer ensemble)
        move_to_numbers_device : bool
            Move the resulting tensors to the device of the ``numbers`` tensor.
            This should be switched off for GPU calculations that use `libcint`
            for integrals as the :class:`.IndexHelper` has to be on the CPU
            for this step.

        Returns
        -------
        IndexHelper
            Instance of index helper for given basis set.
        """
        # pylint: disable=import-outside-toplevel
        from dxtb import GFN1_XTB

        return super().from_numbers(
            numbers,
            GFN1_XTB,
            batch_mode=batch_mode,
            move_to_numbers_device=move_to_numbers_device,
        )


class IndexHelperGFN2(IndexHelper):
    """
    Index helper for GFN2 basis set.
    """

    @override
    @classmethod
    def from_numbers(
        cls,
        numbers: Tensor,
        batch_mode: int | None = None,
        move_to_numbers_device: bool = True,
    ) -> IndexHelper:
        """
        Construct an index helper instance from atomic numbers and their
        angular momenta. The latter are collected from the GFN1 parametrization.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        batch_mode : int
            Whether multiple systems or a single one are handled:

            - 0: Single system
            - 1: Multiple systems with padding
            - 2: Multiple systems with no padding (conformer ensemble)
        move_to_numbers_device : bool
            Move the resulting tensors to the device of the ``numbers`` tensor.
            This should be switched off for GPU calculations that use `libcint`
            for integrals as the :class:`.IndexHelper` has to be on the CPU
            for this step.

        Returns
        -------
        IndexHelper
            Instance of index helper for given basis set.
        """
        # pylint: disable=import-outside-toplevel
        from dxtb import GFN2_XTB

        return super().from_numbers(
            numbers,
            GFN2_XTB,
            batch_mode=batch_mode,
            move_to_numbers_device=move_to_numbers_device,
        )
