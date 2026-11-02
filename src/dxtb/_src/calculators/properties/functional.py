# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Pure functional 1st, 2nd, and 3rd order property evaluators (B8a/B8b/E8).

All functions in this module operate on immutable
:class:`~dxtb._src.calculators.model.System` and
:class:`~dxtb._src.calculators.model.Model` values via :mod:`torch.func`
transforms without mutating or resetting component caches.

Conventions and Units
---------------------
All evaluators use atomic units (Hartree ``Eh``, Bohr ``a0``, elementary charge
``e``) and match the sign and packing conventions of
:class:`~dxtb._src.calculators.types.autograd.AutogradCalculator`:

- ``system_energy``: Scalar total energy :math:`E` in :math:`\\text{E}_\\text{h}`
  (includes the Mermin electronic free-energy term :math:`F_\\text{el} = -T S_\\text{el}`
  by default when ``include_free_energy=True``).
- ``system_forces``: Nuclear forces :math:`F = -\\partial E / \\partial R` in
  :math:`\\text{E}_\\text{h}/\\text{a}_0`, shape ``(nat, 3)``.
- ``system_hessian``: Nuclear Hessian :math:`H = \\partial^2 E / \\partial R^2`
  in :math:`\\text{E}_\\text{h}/\\text{a}_0^2`, shape ``(nat, 3, nat, 3)`` or
  ``(3*nat, 3*nat)`` when ``matrix=True``.
- ``system_dipole``: Electric dipole moment :math:`\\mu = -\\partial E / \\partial F`
  in :math:`\\text{e}\\cdot\\text{a}_0`, shape ``(3,)``.
- ``system_quadrupole``: Traceless electric quadrupole moment
  :math:`\\Theta_{ij} = -3\\,\\partial E / \\partial G_{ij}` in
  :math:`\\text{e}\\cdot\\text{a}_0^2`, packed in lower-triangular order
  ``(xx, yx, yy, zx, zy, zz)`` of shape ``(6,)``.
- ``system_polarizability``: Static dipole polarizability
  :math:`\\alpha = \\partial \\mu / \\partial F = -\\partial^2 E / \\partial F^2`
  in :math:`\\text{a}_0^3`, shape ``(3, 3)``.
- ``system_dipole_deriv``: Cartesian dipole derivative
  :math:`\\mu' = \\partial \\mu / \\partial R = -\\partial^2 E / (\\partial F \\partial R)`
  in :math:`\\text{e}`, shape ``(3, nat, 3)``.
- ``system_pol_deriv``: Cartesian polarizability derivative
  :math:`\\chi = \\partial \\alpha / \\partial R = -\\partial^3 E / (\\partial F^2 \\partial R)`
  in :math:`\\text{a}_0^2`, shape ``(3, 3, nat, 3)``.
- ``system_hyperpolarizability``: Static first hyperpolarizability
  :math:`\\beta = \\partial \\alpha / \\partial F = -\\partial^3 E / \\partial F^3`
  in :math:`\\text{a}_0^5 / \\text{e}`, shape ``(3, 3, 3)``.
- ``system_third_order_directional``: Third directional nuclear energy
  derivative :math:`D^3 E(R)[v_1, v_2, v_3]`, scalar ``()``.
- ``system_parameter_force_jacobian``: Mixed parameter-force Jacobian
  :math:`\\partial F(R, \\theta) / \\partial \\theta = -\\partial^2 E(R, \\theta) / (\\partial R \\partial \\theta)`.
- ``system_bond_orders``: Wiberg bond order matrix (unitless), shape ``(nat, nat)``.
- ``system_vibration``, ``system_ir``, ``system_raman``: Eager spectroscopic
  analyzers returning :class:`VibResult`, :class:`IRResult` (``ints`` convertible
  to ``km/mol``), and :class:`RamanResult` (``ints`` convertible to ``A^4/amu``).
  Physical normal-mode interpretation requires a stationary (equilibrium)
  geometry where :math:`\\partial E / \\partial R = 0`; because :func:`vib_analysis`
  uses data-dependent linearity detection, rank thresholding, QR projection, and
  eigendecomposition, these three spectroscopic wrappers are eager single-System
  post-processors over the transform-compatible derivative tensors.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from dxtb._src.calculators.model import Model, ModelParameterNode, System
from dxtb._src.calculators.properties.vibration import (
    IRResult,
    RamanResult,
    VibResult,
    ir_ints,
    raman_ints_depol,
    vib_analysis,
)
from dxtb._src.constants import defaults
from dxtb._src.typing import Tensor
from dxtb._src.wavefunction.wiberg import get_bond_order

__all__ = [
    "system_bond_orders",
    "system_dipole",
    "system_dipole_deriv",
    "system_energy",
    "system_forces",
    "system_hessian",
    "system_hyperpolarizability",
    "system_ir",
    "system_parameter_force_jacobian",
    "system_pol_deriv",
    "system_polarizability",
    "system_quadrupole",
    "system_raman",
    "system_third_order_directional",
    "system_vibration",
]


def _default_field(positions: Tensor, field: Tensor | None) -> Tensor:
    return (
        positions.new_zeros((3,))
        if field is None
        else field.to(device=positions.device, dtype=positions.dtype)
    )


def _default_field_grad(positions: Tensor, field_grad: Tensor | None) -> Tensor:
    return (
        positions.new_zeros((3, 3))
        if field_grad is None
        else field_grad.to(device=positions.device, dtype=positions.dtype)
    )


def system_energy(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    include_free_energy: bool = True,
) -> Tensor:
    """Return the scalar total energy of ``system`` at ``positions``.

    Note: ``Result.energy`` already includes ``Result.fenergy`` (the Mermin
    electronic free-energy contribution ``-T * S_el``). Setting
    ``include_free_energy=False`` subtracts ``Result.fenergy`` to return the
    internal energy ``E_classical + E_scf``.
    """
    res = system.singlepoint(
        positions, chrg=chrg, spin=spin, field=field, field_grad=field_grad
    )
    if not include_free_energy and res.fenergy is not None:
        return (res.energy - res.fenergy).sum(-1)
    return res.energy.sum(-1)


def system_forces(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return nuclear forces ``-dE/dR`` of shape ``(nat, 3)`` via ``torch.func``."""

    def energy_fn(pos: Tensor) -> Tensor:
        return system_energy(
            system,
            pos,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    return -torch.func.grad(energy_fn)(positions).contiguous()


def system_hessian(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    matrix: bool = False,
) -> Tensor:
    """Return the nuclear Hessian ``d^2E/dR^2`` via ``jacfwd(grad(E))``."""

    def energy_fn(pos: Tensor) -> Tensor:
        return system_energy(
            system,
            pos,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    hess = torch.func.jacfwd(torch.func.grad(energy_fn))(positions).contiguous()
    if matrix:
        dof = int(positions.shape[-2]) * 3
        return hess.reshape(*positions.shape[:-2], dof, dof)
    return hess


def system_dipole(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return electric dipole moment ``-dE/dF`` of shape ``(3,)``."""
    f0 = _default_field(positions, field)

    def energy_of_field(f: Tensor) -> Tensor:
        return system_energy(
            system,
            positions,
            chrg=chrg,
            spin=spin,
            field=f,
            field_grad=field_grad,
        )

    return -torch.func.jacrev(energy_of_field)(f0).contiguous()


def system_quadrupole(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return packed lower-triangular electric quadrupole ``-3 dE/dG`` of shape ``(6,)``."""
    g0 = _default_field_grad(positions, field_grad)

    def energy_of_fgrad(g: Tensor) -> Tensor:
        return system_energy(
            system,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=g,
        )

    deriv = torch.func.jacrev(energy_of_fgrad)(g0)
    rows, cols = torch.tril_indices(3, 3, device=deriv.device).unbind()
    return (-3.0 * deriv[..., rows, cols]).contiguous()


def system_polarizability(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return static electric polarizability ``-d^2E/dF^2`` of shape ``(3, 3)``."""
    f0 = _default_field(positions, field)

    def dipole_of_field(f: Tensor) -> Tensor:
        return system_dipole(
            system,
            positions,
            chrg=chrg,
            spin=spin,
            field=f,
            field_grad=field_grad,
        )

    return torch.func.jacfwd(dipole_of_field)(f0).contiguous()


def system_dipole_deriv(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return Cartesian dipole derivative ``d(mu)/dR`` of shape ``(3, nat, 3)``."""

    def dipole_of_pos(pos: Tensor) -> Tensor:
        return system_dipole(
            system,
            pos,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    return torch.func.jacrev(dipole_of_pos)(positions).contiguous()


def system_pol_deriv(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return Cartesian polarizability derivative ``d(alpha)/dR`` of shape ``(3, 3, nat, 3)``."""

    def pol_of_pos(pos: Tensor) -> Tensor:
        return system_polarizability(
            system,
            pos,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    return torch.func.jacrev(pol_of_pos)(positions).contiguous()


def system_hyperpolarizability(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return static first hyperpolarizability ``beta = -d^3E/dF^3`` of shape ``(3, 3, 3)``."""
    f0 = _default_field(positions, field)

    def pol_of_field(f: Tensor) -> Tensor:
        return system_polarizability(
            system,
            positions,
            chrg=chrg,
            spin=spin,
            field=f,
            field_grad=field_grad,
        )

    return torch.func.jacfwd(pol_of_field)(f0).contiguous()


def system_third_order_directional(
    system: System,
    positions: Tensor,
    v1: Tensor,
    v2: Tensor | None = None,
    v3: Tensor | None = None,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return the third directional energy derivative along ``(v1, v2, v3)``."""
    d2 = v1 if v2 is None else v2
    d3 = v1 if v3 is None else v3

    def energy_fn(pos: Tensor) -> Tensor:
        return system_energy(
            system,
            pos,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def hvp_contraction(pos: Tensor) -> Tensor:
        _, hvp = torch.func.jvp(torch.func.grad(energy_fn), (pos,), (d2,))
        return (hvp * d3).sum()

    _, third = torch.func.jvp(hvp_contraction, (positions,), (v1,))
    return third


def system_parameter_force_jacobian(
    model: Model,
    numbers: Tensor,
    positions: Tensor,
    parameter_key: str,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return mixed parameter-force Jacobian ``dF(R, theta)/d(theta)``."""
    base_tree = model.parameter_tree
    if parameter_key not in base_tree.parameters.params:
        raise KeyError(
            f"Parameter key '{parameter_key}' not found in Model.parameter_tree."
        )
    base_param = base_tree.parameters.params[parameter_key]

    def forces_of_param(param_val: Tensor) -> Tensor:
        updated = dict(base_tree.parameters.params)
        updated[parameter_key] = param_val
        tree = ModelParameterNode(
            parameters=replace(base_tree.parameters, params=updated),
            config=base_tree.config,
            auto_int_level=base_tree.auto_int_level,
        )
        sys = model.setup_functional(numbers, tree)
        return system_forces(
            sys,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    return torch.func.jacfwd(forces_of_param)(base_param).contiguous()


def system_bond_orders(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Tensor:
    """Return the Wiberg bond order matrix ``(nat, nat)``."""
    res = system.singlepoint(
        positions, chrg=chrg, spin=spin, field=field, field_grad=field_grad
    )
    if res.overlap is None or res.density is None:
        raise RuntimeError("Bond orders require overlap and density matrices.")
    return get_bond_order(res.overlap, res.density, system.ihelp)


def system_vibration(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    project_translational: bool = True,
    project_rotational: bool = True,
) -> VibResult:
    """Perform vibrational analysis purely via ``System`` transforms."""
    hess = system_hessian(
        system,
        positions,
        chrg=chrg,
        spin=spin,
        field=field,
        field_grad=field_grad,
        matrix=False,
    )
    return vib_analysis(
        system.numbers,
        positions,
        hess,
        project_translational=project_translational,
        project_rotational=project_rotational,
    )


def system_ir(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    project_translational: bool = True,
    project_rotational: bool = True,
) -> IRResult:
    """Calculate IR frequencies and intensities without mutable cache resets."""
    vib_res = system_vibration(
        system,
        positions,
        chrg=chrg,
        spin=spin,
        field=field,
        field_grad=field_grad,
        project_translational=project_translational,
        project_rotational=project_rotational,
    )
    dmu_dr = system_dipole_deriv(
        system,
        positions,
        chrg=chrg,
        spin=spin,
        field=field,
        field_grad=field_grad,
    )
    intensities = ir_ints(dmu_dr, vib_res.modes)
    return IRResult(vib_res.freqs, intensities)


def system_raman(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
    *,
    project_translational: bool = True,
    project_rotational: bool = True,
) -> RamanResult:
    """Calculate Raman frequencies, intensities, and depolarization ratios."""
    vib_res = system_vibration(
        system,
        positions,
        chrg=chrg,
        spin=spin,
        field=field,
        field_grad=field_grad,
        project_translational=project_translational,
        project_rotational=project_rotational,
    )
    da_dr = system_pol_deriv(
        system,
        positions,
        chrg=chrg,
        spin=spin,
        field=field,
        field_grad=field_grad,
    )
    intensities, depol = raman_ints_depol(da_dr, vib_res.modes)
    return RamanResult(vib_res.freqs, intensities, depol)
