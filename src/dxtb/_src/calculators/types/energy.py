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
Calculators: Energy
===================

Base calculator for the energy calculation of an extended tight-binding Method.
"""

from __future__ import annotations

from collections.abc import Sequence
import inspect

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.io.checks import content_checks, shape_checks

from dxtb import OutputHandler, labels
from dxtb._src.calculators.singlepoint import (
    _call_tensor,
    _singlepoint_eager,
)
from dxtb._src.constants import defaults
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor

from ..result import Result
from .base import BaseCalculator

SINGLEPOINT_OPTIONS = frozenset(
    {
        "write_overlap",
        "write_dipole",
        "write_quadrupole",
        "write_hcore",
        "cuda_sync_in_scf",
    }
)
"""Keyword options that a Calculator single point accepts besides its inputs."""

_ENERGY_KWARGS = SINGLEPOINT_OPTIONS | {"field", "field_grad", "keepdim"}
"""Keyword arguments of :meth:`EnergyCalculator.energy`."""


def check_singlepoint_options(kwargs: dict[str, Any], owner: str) -> None:
    """Raise a ``TypeError`` for keyword arguments that no single point takes."""
    unknown = sorted(set(kwargs) - SINGLEPOINT_OPTIONS)
    if unknown:
        raise TypeError(
            f"{owner}() got an unexpected keyword argument {unknown[0]!r}"
        )


def accepted_kwargs(method: Any) -> frozenset[str]:
    """
    Keyword arguments that a property method takes.

    These are its named parameters besides the positions, charge and spin,
    and, if it forwards ``**kwargs`` (down to the energy), the keyword
    arguments of the energy.
    """
    params = inspect.signature(method).parameters.values()
    names = {
        p.name
        for p in params
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    } - {"self", "positions", "chrg", "spin"}
    if any(p.kind is p.VAR_KEYWORD for p in params):
        names |= _ENERGY_KWARGS
    return frozenset(names)


def kwargs_for(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Select the keyword arguments that a property method takes."""
    names = accepted_kwargs(method)
    return {k: v for k, v in kwargs.items() if k in names}


__all__ = ["EnergyCalculator"]



class EnergyCalculator(BaseCalculator):
    """
    Parametrized calculator defining the extended tight-binding Method.

    This class provides the basic functionality for the extended tight-binding
    model. It provides methods for single point calculations, nuclear
    gradients, Hessians, molecular properties, and spectra.
    """

    # all from an SCF
    implemented_properties: list[str] = [
        "bond_orders",
        "energy",
        "coefficients",
        "charges",
        "density",
        "iterations",
        "mo_energies",
        "occupation",
        "potential",
    ]
    """Names of implemented methods of the Calculator."""

    __slots__ = [
        "numbers",
        "opts",
        "classicals",
        "interactions",
        "ihelp",
    ]

    def singlepoint(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | Sequence[Tensor | None] | None = None,
        field_grad: Tensor | Sequence[Tensor | None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Entry point for performing single point calculations.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0 (also for batched calculations). May
            be fractional. At finite electronic temperature, the energy is
            differentiable with respect to it.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to 0.
        options : Any
            ``write_overlap``, ``write_dipole``, ``write_quadrupole`` and
            ``write_hcore`` write the matrix to the given path;
            ``cuda_sync_in_scf`` synchronizes CUDA in the SCF timer. Any other
            keyword argument raises a ``TypeError``.
        """
        check_singlepoint_options(kwargs, "singlepoint")
        field, field_grad = self._check_fields(field, field_grad)

        # shape checks
        assert shape_checks(
            self.numbers,
            positions,
            allow_batched=True,
        )
        assert content_checks(
            self.numbers,
            positions,
            self.opts.max_element,
            allow_batched=True,
        )

        OutputHandler.write_stdout("Singlepoint ", v=3)

        if self.numbers.ndim > 1:
            from dxtb._src.calculators.packed_result import (
                evaluate_calculator_packed_singlepoint,
            )

            return evaluate_calculator_packed_singlepoint(
                self,
                positions,
                chrg=chrg,
                spin=spin,
                field=field,
                field_grad=field_grad,
                **kwargs,
            )

        _chrg: Tensor = torch.atleast_1d(any_to_tensor(chrg, **self.dd))
        if spin is not None:
            _spin = torch.atleast_1d(any_to_tensor(spin, **self.dd))
        else:
            _spin = None

        field = _call_tensor("field", field, positions, (3,))
        field_grad = _call_tensor("field_grad", field_grad, positions, (3, 3))
        if field is not None and self.opts.ints.level < labels.INTLEVEL_DIPOLE:
            raise RuntimeError(
                "An electric field requires dipole integrals. Construct the "
                "Calculator with an integral level of at least "
                f"{labels.INTLEVEL_DIPOLE}."
            )
        if (
            field_grad is not None
            and self.opts.ints.level < labels.INTLEVEL_QUADRUPOLE
        ):
            raise RuntimeError(
                "An electric field gradient requires quadrupole integrals. "
                "Construct the Calculator with an integral level of at least "
                f"{labels.INTLEVEL_QUADRUPOLE}."
            )

        return self._singlepoint_system(
            positions, _chrg, _spin, field, field_grad, kwargs
        )

    def _singlepoint_system(
        self,
        positions: Tensor,
        charge: Tensor,
        spin: Tensor | None,
        field: Tensor | None,
        field_grad: Tensor | None,
        kwargs: dict[str, Any],
    ) -> Result:
        """Adapt a System result to the Calculator state and options."""
        eval_system = self._prepare_eval_system()
        old_cuda_sync = timer.cuda_sync
        timer.cuda_sync = kwargs.get(
            "cuda_sync_in_scf", False if self.device.type == "cpu" else True
        )
        try:
            result = _singlepoint_eager(
                eval_system,
                positions,
                charge,
                spin,
                field,
                field_grad,
            )
        finally:
            timer.cuda_sync = old_cuda_sync

        from dxtb._src.calculators.packed_result import (
            _write_packed_integrals,
        )

        _write_packed_integrals(result, eval_system, kwargs)

        self._ncalcs += 1
        return result

    def energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | Sequence[Tensor | None] | None = None,
        field_grad: Tensor | Sequence[Tensor | None] | None = None,
        keepdim: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the total energy :math:`E` of the Structure.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        keepdim : bool, optional
            Keep the summed atom dimension. Defaults to ``False``.
        kwargs : Any
            Single-point options (see :meth:`singlepoint`).

        Returns
        -------
        Tensor
            Total energy of the Structure (scalar value).
        """
        result = self.singlepoint(
            positions, chrg, spin, field=field, field_grad=field_grad, **kwargs
        )
        return result.energy.sum(-1, keepdim=keepdim)

    def _validate_resolved_perturbation(
        self,
        value: Any,
        *,
        name: str,
        display_name: str,
        expected_shape: tuple[int, ...],
    ) -> None:
        if value is None:
            return
        allow_seq = self.numbers.ndim > 1
        if not (
            isinstance(value, Tensor)
            or (
                allow_seq
                and isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
            )
        ):
            raise TypeError(f"{name} must be a Tensor or None.")
        if not isinstance(value, Tensor):
            return
        if self.numbers.ndim == 1:
            if value.shape != expected_shape:
                raise ValueError(
                    f"{display_name} must have shape {expected_shape}, got "
                    f"{tuple(value.shape)}."
                )
        else:
            batch_size = int(self.numbers.shape[0])
            if value.shape not in (expected_shape, (batch_size, *expected_shape)):
                raise ValueError(
                    f"{name} must have shape {expected_shape} or "
                    f"{(batch_size, *expected_shape)}, got {tuple(value.shape)}."
                )

    def _check_fields(
        self,
        field: Tensor | Sequence[Tensor | None] | None = None,
        field_grad: Tensor | Sequence[Tensor | None] | None = None,
    ) -> tuple[
        Tensor | Sequence[Tensor | None] | None,
        Tensor | Sequence[Tensor | None] | None,
    ]:
        """Validate the per-call electric field and field gradient."""
        self._validate_resolved_perturbation(
            field,
            name="field",
            display_name="Electric field",
            expected_shape=(3,),
        )
        self._validate_resolved_perturbation(
            field_grad,
            name="field_grad",
            display_name="Electric field gradient",
            expected_shape=(3, 3),
        )
        return field, field_grad

    def _stack_response_perturbation(
        self,
        value: Any,
        *,
        name: str,
        response_name: str,
        expected_shape: tuple[int, ...],
    ) -> Any:
        if not (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        ):
            return value
        batch_size = int(self.numbers.shape[0])
        if len(value) != batch_size:
            raise ValueError(
                f"{name} sequence length {len(value)} must match "
                f"batch size {batch_size}."
            )
        if any(
            item is None
            or not isinstance(item, Tensor)
            or item.shape != expected_shape
            for item in value
        ):
            raise ValueError(
                f"{name} sequence for {response_name} response must contain "
                f"{expected_shape} Tensors for every lane."
            )
        return torch.stack(list(value), dim=0)

    def _response_fields(
        self,
        field: Tensor | Sequence[Tensor | None] | None = None,
        field_grad: Tensor | Sequence[Tensor | None] | None = None,
        *,
        needs_field: bool = False,
        needs_field_grad: bool = False,
    ) -> tuple[Any, Any]:
        """Validate the fields of a response and stack per-Lane sequences."""
        resolved_field, resolved_grad = self._check_fields(field, field_grad)

        if needs_field:
            resolved_field = self._stack_response_perturbation(
                resolved_field,
                name="field",
                response_name="dipole",
                expected_shape=(3,),
            )
        if needs_field_grad:
            resolved_grad = self._stack_response_perturbation(
                resolved_grad,
                name="field_grad",
                response_name="quadrupole",
                expected_shape=(3, 3),
            )
        return resolved_field, resolved_grad

    def bond_orders(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the (Wiberg) bond order matrix.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.

        Returns
        -------
        Tensor
            Bond order matrix.
        """
        result = self.singlepoint(positions, chrg, spin, **kwargs)
        if result.overlap is None or result.density is None:
            raise RuntimeError(
                "Bond orders require overlap and density matrices."
            )

        # pylint: disable=import-outside-toplevel
        from dxtb._src.wavefunction.wiberg import get_bond_order

        return get_bond_order(result.overlap, result.density, self.ihelp)

    def _calculate_methods(self, properties: list[str]) -> list[str]:
        """Names of the property methods that `calculate` calls."""
        return []

    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
        kwargs : Any
            Keyword arguments of the property methods. Each method (and the
            single point for the energy and the Result fields) receives the
            ones it takes; every keyword argument must be taken by at least
            one requested property.

        Returns
        -------
        dict[str, Any]
            Values for the requested properties, returned explicitly.

        Raises
        ------
        TypeError
            A keyword argument is taken by none of the requested properties.
        """
        values: dict[str, Any] = {}
        result_fields = {
            "charges": "charges",
            "coefficients": "coefficients",
            "density": "density",
            "iterations": "iterations",
            "mo_energies": "emo",
            "occupation": "occupation",
            "potential": "potential",
        }
        needs_result = bool(
            set(properties) & (set(result_fields) | {"energy", "bond_orders"})
        )

        accepted = set(_ENERGY_KWARGS) if needs_result else set()
        for name in self._calculate_methods(properties):
            accepted |= accepted_kwargs(getattr(self, name))
        unknown = sorted(set(kwargs) - accepted)
        if unknown:
            raise TypeError(
                f"calculate() got an unexpected keyword argument "
                f"{unknown[0]!r} (none of the requested properties takes it)"
            )
        result = (
            self.singlepoint(
                positions,
                chrg,
                spin,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in _ENERGY_KWARGS and k != "keepdim"
                },
            )
            if needs_result
            else None
        )
        if result is not None:
            for name in properties:
                if name in result_fields:
                    values[name] = getattr(result, result_fields[name])
                elif name == "energy":
                    values[name] = result.energy.sum(
                        -1, keepdim=kwargs.get("keepdim", False)
                    )
        if "bond_orders" in properties:
            assert result is not None
            if result.overlap is None or result.density is None:
                raise RuntimeError(
                    "Bond orders require overlap and density matrices."
                )
            from dxtb._src.wavefunction.wiberg import get_bond_order

            values["bond_orders"] = get_bond_order(
                result.overlap, result.density, self.ihelp
            )
        return values
