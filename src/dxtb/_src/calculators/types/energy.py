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

Base calculator for the energy calculation of an extended tight-binding model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.io.checks import content_checks, shape_checks

from dxtb import OutputHandler, labels
from dxtb._src.calculators.singlepoint import (
    _call_tensor,
    _singlepoint_eager,
)
from dxtb._src.components.interactions.field.efield import LABEL_EFIELD
from dxtb._src.components.interactions.field.efieldgrad import LABEL_EFIELD_GRAD
from dxtb._src.constants import defaults
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor

from ..result import Result
from .base import BaseCalculator, reject_removed_store_kwargs

__all__ = ["EnergyCalculator"]

_UNSET = object()


class EnergyCalculator(BaseCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

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
        "_integrals",
        "ihelp",
    ]

    def singlepoint(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
        field_grad: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
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
        """
        reject_removed_store_kwargs(kwargs)
        field, field_grad = self._resolve_fields(field, field_grad)

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
        """Adapt a pure System result to legacy Calculator state and options."""
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
                use_captured_extension_cache=True,
            )
        finally:
            timer.cuda_sync = old_cuda_sync

        if self._integrals is not None:
            for key, matrix, integral in (
                ("write_overlap", result.overlap, self._integrals.overlap),
                ("write_dipole", result.dipole_integrals, self._integrals.dipole),
                (
                    "write_quadrupole",
                    result.quadrupole_integrals,
                    self._integrals.quadrupole,
                ),
                ("write_hcore", result.hcore, self._integrals.hcore),
            ):
                path = kwargs.get(key, False)
                if path is False or matrix is None:
                    continue
                if path is None or path is True:
                    if integral is None:
                        raise RuntimeError(
                            f"No legacy label is available for {key}."
                        )
                    path = integral.label.casefold() + ".pt"
                torch.save(matrix, path)
        else:
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
        field: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
        field_grad: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate the total energy :math:`E` of the system.

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
            Total energy of the system (scalar value).
        """
        result = self.singlepoint(
            positions, chrg, spin, field=field, field_grad=field_grad, **kwargs
        )
        return result.energy.sum(-1, keepdim=kwargs.get("keepdim", False))

    def _resolve_fields(
        self,
        field: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
        field_grad: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
    ) -> tuple[
        Tensor | Sequence[Tensor | None] | None,
        Tensor | Sequence[Tensor | None] | None,
    ]:
        """Resolve omitted Calculator fields against constructor defaults."""
        resolved_field = self._field_default if field is _UNSET else field
        resolved_grad = (
            self._field_grad_default if field_grad is _UNSET else field_grad
        )
        allow_seq = self.numbers.ndim > 1
        if resolved_field is not None and not (
            isinstance(resolved_field, Tensor)
            or (
                allow_seq
                and isinstance(resolved_field, Sequence)
                and not isinstance(resolved_field, (str, bytes))
            )
        ):
            raise TypeError("field must be a Tensor or None.")
        if resolved_grad is not None and not (
            isinstance(resolved_grad, Tensor)
            or (
                allow_seq
                and isinstance(resolved_grad, Sequence)
                and not isinstance(resolved_grad, (str, bytes))
            )
        ):
            raise TypeError("field_grad must be a Tensor or None.")
        if isinstance(resolved_field, Tensor):
            if self.numbers.ndim == 1:
                if resolved_field.shape != (3,):
                    raise ValueError(
                        "Electric field must have shape (3,), got "
                        f"{tuple(resolved_field.shape)}."
                    )
            else:
                batch_size = int(self.numbers.shape[0])
                if resolved_field.shape not in ((3,), (batch_size, 3)):
                    raise ValueError(
                        "field must have shape (3,) or "
                        f"({batch_size}, 3), got {tuple(resolved_field.shape)}."
                    )

        if isinstance(resolved_grad, Tensor):
            if self.numbers.ndim == 1:
                if resolved_grad.shape != (3, 3):
                    raise ValueError(
                        "Electric field gradient must have shape (3, 3), got "
                        f"{tuple(resolved_grad.shape)}."
                    )
            else:
                batch_size = int(self.numbers.shape[0])
                if resolved_grad.shape not in ((3, 3), (batch_size, 3, 3)):
                    raise ValueError(
                        "field_grad must have shape (3, 3) or "
                        f"({batch_size}, 3, 3), got "
                        f"{tuple(resolved_grad.shape)}."
                    )

        return resolved_field, resolved_grad

    def _resolve_response_fields(
        self,
        field: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
        field_grad: Tensor | Sequence[Tensor | None] | None | object = _UNSET,
        *,
        needs_field: bool = False,
        needs_field_grad: bool = False,
    ) -> tuple[Any, Any]:
        """Resolve standard response inputs and explain custom extensions."""
        resolved_field, resolved_grad = self._resolve_fields(field, field_grad)
        component_labels = {
            component.label for component in self.interactions.components
        }

        if (
            needs_field
            and resolved_field is None
            and LABEL_EFIELD in component_labels
        ):
            raise RuntimeError(
                "This Calculator contains a custom ElectricField-labelled "
                "interaction. Generic dipole/polarizability response "
                "differentiates the standard explicit field= input and cannot "
                "infer a perturbation variable from a custom interaction. "
                "Pass/use an exact ElectricField compatibility default or "
                "provide a custom response implementation."
            )
        if (
            needs_field_grad
            and resolved_grad is None
            and LABEL_EFIELD_GRAD in component_labels
        ):
            raise RuntimeError(
                "This Calculator contains a custom ElectricFieldGrad-labelled "
                "interaction. Generic quadrupole response differentiates the "
                "standard explicit field_grad= input and cannot infer a "
                "perturbation variable from a custom interaction. Pass/use an "
                "exact ElectricFieldGrad compatibility default or provide a "
                "custom response implementation."
            )

        if (
            needs_field
            and isinstance(resolved_field, Sequence)
            and not isinstance(resolved_field, (str, bytes))
        ):
            batch_size = int(self.numbers.shape[0])
            if len(resolved_field) != batch_size:
                raise ValueError(
                    f"field sequence length {len(resolved_field)} must match "
                    f"batch size {batch_size}."
                )
            if any(
                item is None or not isinstance(item, Tensor) or item.shape != (3,)
                for item in resolved_field
            ):
                raise ValueError(
                    "field sequence for dipole response must contain (3,) "
                    "Tensors for every lane."
                )
            resolved_field = torch.stack(list(resolved_field), dim=0)

        if (
            needs_field_grad
            and isinstance(resolved_grad, Sequence)
            and not isinstance(resolved_grad, (str, bytes))
        ):
            batch_size = int(self.numbers.shape[0])
            if len(resolved_grad) != batch_size:
                raise ValueError(
                    f"field_grad sequence length {len(resolved_grad)} must "
                    f"match batch size {batch_size}."
                )
            if any(
                item is None
                or not isinstance(item, Tensor)
                or item.shape != (3, 3)
                for item in resolved_grad
            ):
                raise ValueError(
                    "field_grad sequence for quadrupole response must contain "
                    "(3, 3) Tensors for every lane."
                )
            resolved_grad = torch.stack(list(resolved_grad), dim=0)

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

        Returns
        -------
        dict[str, Any]
            Values for the requested properties, returned explicitly.
        """
        reject_removed_store_kwargs(kwargs)
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
        result = (
            self.singlepoint(positions, chrg, spin, **kwargs)
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
