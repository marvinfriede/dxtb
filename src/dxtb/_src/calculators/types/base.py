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
Calculators: Base Class
=======================

A base class for all calculators. All calculators should inherit from this
class and implement the :meth:`calculate` method and the corresponding methods
to calculate the properties specified within this :meth:`calculate`, as well as
the :attr:`implemented_properties` attribute.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace

import torch
from tad_mctc.exceptions import DeviceError, DtypeError

from dxtb import IndexHelper, OutputHandler, labels
from dxtb import integrals as ints
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.properties.vibration import (
    IRResult,
    RamanResult,
    VibResult,
)
from dxtb._src.components.classicals import (
    Classical,
    ClassicalList,
)
from dxtb._src.components.interactions import Interaction, InteractionList
from dxtb._src.components.interactions.container import Charges, Potential
from dxtb._src.constants import defaults
from dxtb._src.param import Param, ParamModule
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Self, Tensor, TensorLike, override
from dxtb._src.utils.tensors import normalize_device
from dxtb.config import Config
from dxtb.integrals import Integrals

from .abc import GetPropertiesMixin, PropertyNotImplementedError

_REMOVED_STORE_OPTIONS = {
    "store_charges",
    "store_coefficients",
    "store_density",
    "store_iterations",
    "store_mo_energies",
    "store_occupation",
    "store_potential",
    "store_fock",
    "store_hcore",
    "store_overlap",
    "store_dipole",
    "store_quadrupole",
}


def reject_removed_store_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject the removed Calculator result-retention arguments."""
    removed = sorted(_REMOVED_STORE_OPTIONS.intersection(kwargs))
    if removed:
        raise TypeError(
            "Calculator result caching has been removed; retain the returned "
            "Result instead. Unsupported arguments: " + ", ".join(removed)
        )


class BaseCalculator(GetPropertiesMixin, TensorLike):
    """
    Base calculator for the extended tight-binding (xTB) models.

    .. warning::

        ``Calculator.to(device)`` does not move all internal state (e.g., the
        :class:`~dxtb.IndexHelper`). Create the calculator on the target
        device instead (``device=...``). See :ref:`help_known_issues`.
    """

    numbers: Tensor
    """Atomic numbers for all atoms in the system (shape: ``(..., nat)``)."""

    classicals: ClassicalList
    """Classical contributions."""

    interactions: InteractionList
    """Interactions to minimize in self-consistent iterations."""

    integrals: Integrals
    """Integrals for the extended tight-binding model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    opts: Config
    """Calculator configuration."""

    _ncalcs: int
    """
    Number of calculations performed with the calculator.
    """

    def __init__(
        self,
        numbers: Tensor,
        par: Param | ParamModule,
        *,
        classical: list[Classical] | tuple[Classical] | Classical | None = None,
        interaction: (
            list[Interaction] | tuple[Interaction] | Interaction | None
        ) = None,
        opts: dict[str, Any] | Config | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Instantiate the Calculator object with the following parameters:

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
        par : Param
            Representation of an extended tight-binding model (full xTB
            parametrization). Decides energy contributions.
        classical : Sequence[Classical] | None, optional
            Additional classical contributions. Defaults to ``None``.
        interaction : Sequence[Interaction] | None, optional
            Additional self-consistent contributions (interactions).
            Defaults to ``None``.
        opts : dict[str, Any] | None, optional
            Calculator options. If ``None`` (default) is given, default options
            are used automatically.
        device : torch.device | None, optional
            Device to store the tensor on. If ``None`` (default), the default
            device is used.
        dtype : torch.dtype | None, optional
            Data type of the tensor. If ``None`` (default), the data type is
            inferred.
        kwargs : Any
            Additional keyword arguments.

            - ``auto_int_level``: Automatically set the integral level based on
              the parametrization. Defaults to ``True``. This should only be
              turned off for testing purposes.
            - ``timer``: Enable the timer. Defaults to ``False``. The global
              timer can also be enabled by setting the environment variable
              ``DXTB_TIMER`` to ``1``.

        """
        if "cache" in kwargs:
            raise TypeError(
                "Calculator result caching has been removed; retain the "
                "returned Result instead."
            )

        if not timer.enabled and kwargs.pop("timer", False):
            timer.enable()

        timer.start("Calculator", parent_uid="Setup")

        # setup verbosity first
        opts = opts if opts is not None else {}
        batch_mode = kwargs.pop("batch_mode", None)
        if isinstance(opts, dict):
            opts = dict(opts)
            OutputHandler.verbosity = opts.pop("verbosity", 1)
            batch_mode = opts.pop("batch_mode", batch_mode)

        OutputHandler.write_stdout("===========", v=4)
        OutputHandler.write_stdout("CALCULATION", v=4)
        OutputHandler.write_stdout("===========", v=4)
        OutputHandler.write_stdout("", v=4)
        OutputHandler.write_stdout("Setup Calculator", v=4)

        allowed_dtypes = (torch.int16, torch.int32, torch.int64)
        if numbers.dtype not in allowed_dtypes:
            raise DtypeError(
                "Dtype of atomic numbers must be one of the following to allow "
                f"indexing: '{', '.join([str(x) for x in allowed_dtypes])}', "
                f"but is '{numbers.dtype}'"
            )
        self.numbers = numbers

        # "cpu" or "cuda" would not compare equal to `tensor.device`
        super().__init__(normalize_device(device), dtype)
        dd = {"device": self.device, "dtype": self.dtype}

        # Internally, we will always use the differentiable parameter model.
        if not isinstance(par, ParamModule):
            par = ParamModule(par, **self.dd)

        # If method not explicitly set in options, we try to get it from the
        # parametrization.
        if isinstance(opts, dict) and "method" not in opts:
            if par.meta is not None:
                if par.meta.name is not None:
                    opts["method"] = par.meta.name.casefold()

        # setup calculator options
        if isinstance(opts, dict):
            opts = Config.create(**opts)
        self.opts = opts

        # Everything that depends on the atomic numbers only is set up by
        # the model; the calculator keeps the parts as attributes (B5 removes
        # the calculator state).
        if classical is None:
            classical = ()
        elif isinstance(classical, Classical):
            classical = (classical,)
        elif isinstance(classical, (list, tuple)):
            classical = tuple(classical)
        else:
            raise TypeError(
                "Expected 'classical' to be 'None' or of type 'Classical', "
                "'list[Classical]', or 'tuple[Classical]', but got "
                f"'{type(classical).__name__}'."
            )

        if interaction is None:
            interaction = ()
        elif isinstance(interaction, Interaction):
            interaction = (interaction,)
        elif isinstance(interaction, (list, tuple)):
            interaction = tuple(interaction)
        else:
            raise TypeError(
                "Expected 'interaction' to be 'None' or of type 'Interaction', "
                "'list[Interaction]', or 'tuple[Interaction]', but got "
                f"'{type(interaction).__name__}'."
            )

        self.model = Model(
            par=par,
            config=self.opts,
            classical=classical,
            interaction=interaction,
            auto_int_level=kwargs.pop("auto_int_level", True),
        )
        self.system = self.model.setup(numbers, batch_mode=batch_mode, dd=dd)

        self.opts = self.system.config
        self.ihelp = self.system.ihelp
        self.classicals = self.system.classicals
        self.interactions = self.system.interactions
        self.integrals = self._create_legacy_integral_adapter()

        self._ncalcs = 0
        timer.stop("Calculator")

    def _create_legacy_integral_adapter(self) -> Integrals:
        """Create Calculator-local mutable integral compatibility objects."""
        manager = ints.DriverManager(
            self.opts.ints.driver,
            algorithm=self.opts.ints.algorithm,
            **self.dd,
        )
        manager.create_driver(self.numbers, self.model.par, self.ihelp)
        integrals = ints.Integrals(
            manager, intlevel=self.opts.ints.level, **self.dd
        )

        if self.opts.ints.level >= labels.INTLEVEL_OVERLAP:
            if self.system.h0_setup is None:
                raise RuntimeError("Core Hamiltonian setup is missing.")
            integrals.hcore = ints.factories.new_hcore(
                self.numbers,
                self.model.par,
                self.ihelp,
                setup=self.system.h0_setup,
                **self.dd,
            )
            integrals.overlap = ints.factories.new_overlap(
                driver=manager.driver_type, **self.dd
            )

        if self.opts.ints.level >= labels.INTLEVEL_DIPOLE:
            integrals.dipole = ints.factories.new_dipint(
                driver=manager.driver_type, **self.dd
            )

        if self.opts.ints.level >= labels.INTLEVEL_QUADRUPOLE:
            integrals.quadrupole = ints.factories.new_quadint(
                driver=manager.driver_type, **self.dd
            )

        return integrals

    def reset(self) -> None:
        """
        Reset the calculator to its initial state.

        .. warning::

            The tensors of the components are replaced by detached copies,
            which cuts gradients to tensors passed in by the user (e.g., an
            electric field with ``requires_grad=True``). Pass them again
            after the reset. See :ref:`help_known_issues`.
        """
        self.classicals.reset_all()
        self.interactions.reset_all()
        self.integrals.reset_all()

    @abstractmethod
    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ):
        """
        Calculate the requested properties. This is more of a dispatcher method
        that calls the appropriate methods of the Calculator.

        Parameters
        ----------
        properties : list[str]
            List of properties to calculate.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        """

    def get_property(
        self,
        name: str,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        allow_calculation: bool = True,
        return_clone: bool = False,
        **kwargs: Any,
    ) -> Tensor | Charges | Potential | VibResult | IRResult | RamanResult | None:
        """
        Get the named property.

        Parameters
        ----------
        name : str
            Name of the property to get.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        allow_calculation : bool, optional
            If ``False``, return ``None`` without evaluating. Since previous
            outputs are not retained by Calculator, this does not retrieve an
            earlier value. Defaults to ``True``.
        return_clone : bool, optional
            If True and the value is a tensor, return a local clone.
            Defaults to ``False``.
        """
        reject_removed_store_kwargs(kwargs)
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(
                f"Property '{name}' not implemented. Use one of: "
                f"{self.implemented_properties}."
            )

        # Without a persistent property cache, calculation is the only source.
        if allow_calculation is False:
            return None

        # Before calculating, let's do some device and dtype checks.
        if self.device != positions.device:
            raise DeviceError(
                f"Device mismatch: Calculator is on '{self.device}', but "
                f"positions are on '{positions.device}'."
            )
        if self.dtype != positions.dtype:
            raise DtypeError(
                f"Dtype mismatch: Calculator is of type '{self.dtype}', but "
                f"positions are of type '{positions.dtype}'."
            )
        if isinstance(chrg, Tensor):
            if self.dtype != chrg.dtype:
                raise DtypeError(
                    f"Dtype mismatch: Calculator is of type '{self.dtype}', "
                    f"but charge is of type '{chrg.dtype}'."
                )
            if self.device != chrg.device:
                raise DeviceError(
                    f"Device mismatch: Calculator is on '{self.device}', but "
                    f"charge is on '{chrg.device}'."
                )
        if isinstance(spin, Tensor):
            if self.dtype != spin.dtype:
                raise DtypeError(
                    f"Dtype mismatch: Calculator is of type '{self.dtype}', "
                    f"but spin is of type '{spin.dtype}'."
                )
            if self.device != spin.device:
                raise DeviceError(
                    f"Device mismatch: Calculator is on '{self.device}', but "
                    f"spin is on '{spin.device}'."
                )

        values = self.calculate(
            [name], positions, chrg=chrg, spin=spin, **kwargs
        )
        if not isinstance(values, dict) or name not in values:
            raise PropertyNotImplementedError(
                f"Property '{name}' was not returned by the Calculator."
            )
        result = values[name]
        return (
            result.clone()
            if return_clone and isinstance(result, Tensor)
            else result
        )

    @override
    def type(self, dtype: torch.dtype) -> Self:
        """
        Returns a copy of the class instance with specified floating point type.

        This method overrides the usual approach because the
        :class:`dxtb.Calculator`'s arguments and slots differ significantly.
        Hence, it is not practical to instantiate a new copy.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        Self
            A copy of the class instance with the specified dtype.

        Raises
        ------
        RuntimeError
            If the ``__slots__`` attribute is not set in the class.
        DtypeError
            If the specified dtype is not allowed.
        """

        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting ``__slots__`` in the "
                f"'{self.__class__.__name__}' class."
            )

        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )

        self.classicals = self.classicals.type(dtype)
        self.interactions = self.interactions.type(dtype)
        self.integrals = self.integrals.type(dtype)

        # Keep the system consistent with the converted components. Its
        # classical data is not converted (running the calculator after
        # `type` is a known issue, see the class docstring).
        self.system = replace(
            self.system,
            classicals=self.classicals,
            interactions=self.interactions,
            integral_setup=(
                None
                if self.system.integral_setup is None
                else self.system.integral_setup.to(dtype=dtype)
            ),
            h0_setup=(
                None
                if self.system.h0_setup is None
                else self.system.h0_setup.to(dtype=dtype)
            ),
            dd={**self.system.dd, "dtype": dtype},
        )
        # hard override of the dtype in TensorLike
        self.override_dtype(dtype)

        return self

    def __str__(self) -> str:  # pragma: no cover
        """
        Return a string representation of the instance.
        """
        num_atoms = self.numbers.shape[-1]
        device = self.device
        dtype = self.dtype
        num_calculations = self._ncalcs
        num_classicals = len(self.classicals)
        num_interactions = len(self.interactions)
        num_integrals = len(self.integrals)

        return (
            f"{self.__class__.__name__}(\n"
            f"  Number of Atoms: {num_atoms}\n"
            f"  Device: {device}\n"
            f"  Data Type: {dtype}\n"
            f"  Calculations Performed: {num_calculations}\n"
            f"  Classical Contributions: {num_classicals} {self.classicals.labels}\n"
            f"  Interactions: {num_interactions} {self.interactions.labels}\n"
            f"  Integrals: {num_integrals} {self.integrals.labels}\n"
            f")"
        )

    def __repr__(self) -> str:  # pragma: no cover
        """Return a representation of the instance."""
        return str(self)
