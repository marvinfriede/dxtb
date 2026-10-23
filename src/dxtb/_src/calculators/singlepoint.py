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
Calculators: singlepoint core
=============================

Single-system evaluation from explicit System and call inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tad_mctc.exceptions import DeviceError, DtypeError

from dxtb import labels
from dxtb._src import scf
from dxtb._src.calculators.model import ClassicalTermDescriptor, System
from dxtb._src.calculators.result import ChargeLike, PotentialLike, Result
from dxtb._src.components.classicals.dispersion.d3 import (
    DispersionD3,
    DispersionD3CoreSetup,
    dispersion_d3_energy,
)
from dxtb._src.components.classicals.dispersion.d4 import (
    DispersionD4,
    DispersionD4CoreSetup,
    dispersion_d4_energy,
)
from dxtb._src.components.classicals.halogen import (
    Halogen,
    HalogenCoreSetup,
    halogen_energy,
)
from dxtb._src.components.classicals.ies import IES, IESCoreSetup, ies_energy
from dxtb._src.components.classicals.repulsion.base import repulsion_energy
from dxtb._src.components.classicals.repulsion.rep import (
    Repulsion,
    RepulsionCoreSetup,
)
from dxtb._src.components.classicals.shortrangebond import (
    ShortRangeBond,
    ShortRangeBondCoreSetup,
    short_range_bond_energy,
)
from dxtb._src.components.interactions.coulomb.multipole import (
    AES2CoreSetup,
    build_aes2_data,
)
from dxtb._src.components.interactions.coulomb.secondorder import (
    ES2Cache,
    ES2CoreSetup,
    build_es2_coulomb,
)
from dxtb._src.components.interactions.coulomb.thirdorder import (
    ES3Cache,
    ES3CoreSetup,
)
from dxtb._src.components.interactions.dispersion.d4sc import (
    D4SCCoreSetup,
    build_d4sc_data,
)
from dxtb._src.components.interactions.field.efield import (
    LABEL_EFIELD,
    ElectricField,
    build_electric_field_data,
)
from dxtb._src.components.interactions.field.efieldgrad import (
    LABEL_EFIELD_GRAD,
    ElectricFieldGrad,
    build_electric_field_grad_data,
)
from dxtb._src.components.interactions.list import InteractionListCache
from dxtb._src.components.interactions.solvation.alpb import (
    GeneralizedBorn,
    GeneralizedBornCoreSetup,
    build_generalized_born_data,
)
from dxtb._src.constants import defaults
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.scf.status import SCFStatus
from dxtb._src.timing import timer
from dxtb._src.typing import Any, Tensor

__all__ = ["singlepoint"]


def _iteration_tensor(value: int | Tensor, like: Tensor) -> Tensor:
    """Preserve functional iteration tensors and normalize legacy integers."""
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.int64, device=like.device)


@dataclass(frozen=True, eq=False)
class _SCFOutput:
    """Common read-only view over the legacy and functional result contracts."""

    energy: Tensor
    fenergy: Tensor
    charges: ChargeLike
    density: Tensor
    coefficients: Tensor
    emo: Tensor
    occupation: Tensor
    potential: PotentialLike
    hamiltonian: Tensor
    iterations: Tensor
    converged: Tensor | None
    residual: Tensor | None
    status: SCFStatus | None


def _normalize_scf_output(result: Any, like: Tensor) -> _SCFOutput:
    """Read SCF outputs once without copying functional tensors."""
    from dxtb._src.scf.functional.eager import (
        FunctionalCoreResult,
        public_scf_status,
    )

    if isinstance(result, FunctionalCoreResult):
        return _SCFOutput(
            energy=result.energy,
            fenergy=result.fenergy,
            charges=result.charges,
            density=result.density,
            coefficients=result.coefficients,
            emo=result.emo,
            occupation=result.occupation,
            potential=result.potential,
            hamiltonian=result.hamiltonian,
            iterations=_iteration_tensor(result.iterations, like),
            converged=result.converged,
            residual=result.residual,
            status=public_scf_status(
                result.solve_result.trajectory_status,
                result.solve_result.final_evaluation.fermi_status,
                like=result.converged,
            ),
        )

    return _SCFOutput(
        energy=result["energy"],
        fenergy=result["fenergy"],
        charges=result["charges"],
        density=result["density"],
        coefficients=result["coefficients"],
        emo=result["emo"],
        occupation=result["occupation"],
        potential=result["potential"],
        hamiltonian=result["hamiltonian"],
        iterations=_iteration_tensor(result["iterations"], like),
        converged=None,
        residual=None,
        status=None,
    )


def _evaluate_classical_descriptor(
    descriptor: ClassicalTermDescriptor,
    positions: Tensor,
    charge: Tensor,
    *,
    q: Tensor | None = None,
) -> Tensor:
    """Evaluate a captured built-in term without component dispatch."""
    setup = descriptor.setup
    if descriptor.kind == "d3":
        if not isinstance(setup, DispersionD3CoreSetup):
            raise TypeError("D3 descriptor has an invalid setup value.")
        return dispersion_d3_energy(setup, positions)
    if descriptor.kind == "repulsion":
        if not isinstance(setup, RepulsionCoreSetup):
            raise TypeError("Repulsion descriptor has an invalid setup value.")
        energy = repulsion_energy(
            positions,
            setup.mask,
            setup.arep,
            setup.kexp,
            setup.zeff,
            setup.cutoff,
        )
        return 0.5 * torch.sum(energy, dim=-1)
    if descriptor.kind == "srb":
        if not isinstance(setup, ShortRangeBondCoreSetup):
            raise TypeError("SRB descriptor has an invalid setup value.")
        return short_range_bond_energy(setup, positions)
    if descriptor.kind == "d4":
        if not isinstance(setup, DispersionD4CoreSetup):
            raise TypeError("D4 descriptor has an invalid setup value.")
        return dispersion_d4_energy(setup, positions, charge, q=q)
    if descriptor.kind == "ies":
        if not isinstance(setup, IESCoreSetup):
            raise TypeError("IES descriptor has an invalid setup value.")
        return ies_energy(setup, positions, charge)
    if descriptor.kind == "halogen":
        if not isinstance(setup, HalogenCoreSetup):
            raise TypeError("Halogen descriptor has an invalid setup value.")
        return halogen_energy(setup, positions)
    raise RuntimeError(f"Unknown classical term kind '{descriptor.kind}'.")


class _SystemInteractionData(InteractionListCache):
    """Call-local interaction caches with their captured D4SC authority."""

    def __init__(
        self,
        d4sc_setup: D4SCCoreSetup | None,
        functional_kinds: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.d4sc_setup = d4sc_setup
        self.functional_kinds = (
            {} if functional_kinds is None else functional_kinds
        )


def _interaction_data(
    system: System, positions: Tensor, interactions=None
) -> InteractionListCache:
    """Build call-local interaction data for one SCF evaluation.

    Exact ES2, ES3, AES2, D4SC, and GeneralizedBorn use explicit System setup values. Other
    interactions keep their current cache APIs until their B6a packages, but
    returned data is local to this call.
    """
    interactions = system.interactions if interactions is None else interactions
    core = system._active_exact_core()
    descriptors = (
        {term.position: term for term in core.interaction_terms}
        if core is not None
        else {}
    )
    functional_kinds: dict[str, str] = {}
    d4sc_descriptor = next(
        (term for term in descriptors.values() if term.kind == "d4sc"), None
    )
    owns_selected_d4sc = system.d4sc_interaction is not None and any(
        interaction is system.d4sc_interaction
        for interaction in interactions.components
    )
    authoritative_d4sc_setup = (
        system.interaction_setup(system.d4sc_interaction)
        if d4sc_descriptor is not None
        and owns_selected_d4sc
        and system.interaction_setup(system.d4sc_interaction)
        is d4sc_descriptor.setup
        else system.d4sc_setup if owns_selected_d4sc else None
    )
    data = _SystemInteractionData(authoritative_d4sc_setup, functional_kinds)
    for position, interaction in enumerate(interactions.components):
        descriptor = descriptors.get(position)
        if descriptor is not None:
            trusted_setup = system.interaction_setup(interaction)
            if (
                trusted_setup is not descriptor.setup
                or interaction.label != descriptor.label
            ):
                descriptor = None
        if descriptor is not None:
            setup = descriptor.setup
            functional_kinds[descriptor.label] = descriptor.kind
            if descriptor.kind == "es2":
                if not isinstance(setup, ES2CoreSetup):
                    raise TypeError("ES2 descriptor has an invalid setup.")
                matrix = build_es2_coulomb(setup, positions)
                data[descriptor.label] = ES2Cache(
                    matrix, shell_resolved=setup.shell_resolved
                )
            elif descriptor.kind == "es3":
                if not isinstance(setup, ES3CoreSetup):
                    raise TypeError("ES3 descriptor has an invalid setup.")
                data[descriptor.label] = ES3Cache(
                    setup.hubbard_derivs, shell_resolved=setup.shell_resolved
                )
            elif descriptor.kind == "aes2":
                if not isinstance(setup, AES2CoreSetup):
                    raise TypeError("AES2 descriptor has an invalid setup.")
                data[descriptor.label] = build_aes2_data(setup, positions)
            elif descriptor.kind == "d4sc":
                if not isinstance(setup, D4SCCoreSetup):
                    raise TypeError("D4SC descriptor has an invalid setup.")
                data[descriptor.label] = build_d4sc_data(setup, positions)
            elif descriptor.kind == "generalized_born":
                if not isinstance(setup, GeneralizedBornCoreSetup):
                    raise TypeError(
                        "GeneralizedBorn descriptor has an invalid setup."
                    )
                data[descriptor.label] = build_generalized_born_data(
                    setup, positions
                )
            else:
                raise RuntimeError(
                    f"Unknown interaction term kind '{descriptor.kind}'."
                )
            continue

        captured_setup = (
            system.interaction_setup(interaction) if core is not None else None
        )
        if type(interaction) is ElectricField:
            data[interaction.label] = build_electric_field_data(
                positions, interaction.field
            )
        elif type(interaction) is ElectricFieldGrad:
            data[interaction.label] = build_electric_field_grad_data(
                positions, interaction.field_grad
            )
        elif isinstance(captured_setup, ES2CoreSetup) or (
            captured_setup is None and interaction is system.es2_interaction
        ):
            setup = (
                captured_setup
                if isinstance(captured_setup, ES2CoreSetup)
                else system.es2_setup
            )
            if setup is None:
                raise RuntimeError("Single-system ES2 setup is missing.")
            matrix = build_es2_coulomb(setup, positions)
            data[interaction.label] = ES2Cache(
                matrix, shell_resolved=setup.shell_resolved
            )
        elif isinstance(captured_setup, ES3CoreSetup) or (
            captured_setup is None and interaction is system.es3_interaction
        ):
            setup = (
                captured_setup
                if isinstance(captured_setup, ES3CoreSetup)
                else system.es3_setup
            )
            if setup is None:
                raise RuntimeError("Single-system ES3 setup is missing.")
            data[interaction.label] = ES3Cache(
                setup.hubbard_derivs,
                shell_resolved=setup.shell_resolved,
            )
        elif isinstance(captured_setup, AES2CoreSetup) or (
            captured_setup is None and interaction is system.aes2_interaction
        ):
            setup = (
                captured_setup
                if isinstance(captured_setup, AES2CoreSetup)
                else system.aes2_setup
            )
            if setup is None:
                raise RuntimeError("Single-system AES2 setup is missing.")
            data[interaction.label] = build_aes2_data(setup, positions)
        elif isinstance(captured_setup, D4SCCoreSetup) or (
            captured_setup is None and interaction is system.d4sc_interaction
        ):
            setup = (
                captured_setup
                if isinstance(captured_setup, D4SCCoreSetup)
                else system.d4sc_setup
            )
            if setup is None:
                raise RuntimeError("Single-system D4SC setup is missing.")
            data[interaction.label] = build_d4sc_data(setup, positions)
        elif isinstance(captured_setup, GeneralizedBornCoreSetup) or (
            captured_setup is None
            and interaction is system.generalized_born_interaction
        ):
            setup = (
                captured_setup
                if isinstance(captured_setup, GeneralizedBornCoreSetup)
                else system.generalized_born_setup
            )
            if setup is None:
                raise RuntimeError(
                    "Single-system GeneralizedBorn setup is missing."
                )
            data[interaction.label] = build_generalized_born_data(
                setup, positions
            )
        elif (
            isinstance(interaction, GeneralizedBorn)
            and type(interaction) is not GeneralizedBorn
        ):
            data[interaction.label] = interaction.get_cache(
                numbers=(core.numbers if core is not None else system.numbers),
                positions=positions,
                ihelp=system.ihelp,
            )
        else:
            data[interaction.label] = interaction.get_cache(
                numbers=(core.numbers if core is not None else system.numbers),
                positions=positions,
                ihelp=system.ihelp,
            )
    return data


def _floating_setup_tensor(system: System) -> Tensor | None:
    """Return a gathered floating tensor that defines the System dtype."""
    if system.h0_setup is not None:
        return system.h0_setup.hscale

    setup = system.integral_setup
    if setup is None:
        return None
    if setup.pytorch is not None and setup.pytorch.alphas:
        return setup.pytorch.alphas[0]
    if setup.libcint is not None and setup.libcint.basis_setups:
        return setup.libcint.basis_setups[0].slater
    return None


def _call_scalar(
    name: str,
    value: Tensor | float | int,
    positions: Tensor,
) -> Tensor:
    """Validate a scalar call input without copying tensor inputs."""
    if isinstance(value, Tensor):
        if value.device != positions.device:
            raise DeviceError(
                f"Device mismatch: positions are on '{positions.device}', "
                f"but {name} is on '{value.device}'."
            )
        if value.dtype != positions.dtype:
            raise DtypeError(
                f"Dtype mismatch: positions are of type '{positions.dtype}', "
                f"but {name} is of type '{value.dtype}'."
            )
        if value.ndim == 0:
            return value.unsqueeze(0)
        if value.ndim == 1 and value.shape[0] == 1:
            return value
        raise ValueError(
            f"Core singlepoint expects {name} to be a scalar or a "
            "one-element tensor."
        )

    return torch.tensor([value], dtype=positions.dtype, device=positions.device)


def _call_tensor(
    name: str, value: Tensor | None, positions: Tensor, shape: tuple[int, ...]
) -> Tensor | None:
    """Validate a perturbation tensor without copying or severing its graph."""
    if value is None:
        return None
    if value.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(value.shape)}."
        )
    if value.device != positions.device:
        raise DeviceError(
            f"Device mismatch: positions are on '{positions.device}', "
            f"but {name} is on '{value.device}'."
        )
    if value.dtype != positions.dtype:
        raise DtypeError(
            f"Dtype mismatch: positions are of type '{positions.dtype}', "
            f"but {name} is of type '{value.dtype}'."
        )
    return value


def singlepoint(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Result:
    """Run the internal eager singlepoint adapter retained for compatibility."""
    return _singlepoint_impl(
        system, positions, chrg, spin, field, field_grad, fixed_trip=False
    )


def system_singlepoint(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int = defaults.CHRG,
    spin: Tensor | float | int | None = defaults.SPIN,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Result:
    """Evaluate one System with fixed trips and tensor-valued status."""
    return _singlepoint_impl(
        system, positions, chrg, spin, field, field_grad, fixed_trip=True
    )


_singlepoint_eager = singlepoint


def _singlepoint_impl(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int,
    spin: Tensor | float | int | None,
    field: Tensor | None,
    field_grad: Tensor | None,
    *,
    fixed_trip: bool,
) -> Result:
    """Share setup, physical kernels, and Result construction by controller."""
    core = system._active_exact_core()
    numbers = core.numbers if core is not None else system.numbers
    config = core.config if core is not None else system.config
    # Keep the legacy IntegralSetup facade at this evaluation boundary. The
    # active core has already verified that its selected backend Node is the
    # same authoritative child held by this facade.
    integral_setup = system.integral_setup
    h0_setup = core.h0_setup if core is not None else system.h0_setup

    if numbers.ndim != 1:
        raise ValueError("Core singlepoint accepts one system at a time.")
    if positions.shape != (numbers.shape[0], 3):
        raise ValueError(
            "Core singlepoint expects positions with shape (nat, 3)."
        )
    system_device = numbers.device
    if positions.device != system_device:
        raise DeviceError(
            f"Device mismatch: System is on '{system_device}', but positions "
            f"are on '{positions.device}'."
        )

    floating_setup = _floating_setup_tensor(system)
    expected_dtype = (
        core.effective_dtype
        if core is not None
        else (
            system.dd["dtype"]
            if floating_setup is None
            else floating_setup.dtype
        )
    )
    if floating_setup is not None:
        if floating_setup.device != system_device:
            raise DeviceError(
                "System floating setup is on "
                f"'{floating_setup.device}', but System numbers are on "
                f"'{system_device}'."
            )
        if core is not None and floating_setup.dtype != core.effective_dtype:
            raise DtypeError(
                "Dtype mismatch: System floating setup is of type "
                f"'{floating_setup.dtype}', but exact core declares "
                f"'{core.effective_dtype}'."
            )
    if positions.dtype != expected_dtype:
        raise DtypeError(
            "Dtype mismatch: System floating setup is of type "
            f"'{expected_dtype}', but positions are of type "
            f"'{positions.dtype}'."
        )

    charge = _call_scalar("charge", chrg, positions)
    spin_tensor = (
        None if spin is None else _call_scalar("spin", spin, positions)
    )

    field = _call_tensor("field", field, positions, (3,))
    field_grad = _call_tensor("field_grad", field_grad, positions, (3, 3))
    if field is not None and config.ints.level < labels.INTLEVEL_DIPOLE:
        raise RuntimeError(
            "An electric field requires dipole integrals. Construct the "
            "System with an integral level of at least "
            f"{labels.INTLEVEL_DIPOLE}."
        )
    if (
        field_grad is not None
        and config.ints.level < labels.INTLEVEL_QUADRUPOLE
    ):
        raise RuntimeError(
            "An electric field gradient requires quadrupole integrals. "
            "Construct the System with an integral level of at least "
            f"{labels.INTLEVEL_QUADRUPOLE}."
        )

    # B5 provides the pure evaluation boundary. B6a removes persistent
    # component state behind this boundary.
    if system.classicals.components:
        classical: list[tuple[str, Tensor]] = []
        classical_terms = (
            {term.position: term for term in core.classical_terms}
            if core is not None
            else {}
        )
        for position, component in enumerate(system.classicals.components):
            term = classical_terms.get(position)
            if term is not None:
                timer.start(term.label, parent_uid="Classicals")
                energy = _evaluate_classical_descriptor(term, positions, charge)
                classical.append((term.label, energy))
                timer.stop(term.label)
                continue

            timer.start(component.label, parent_uid="Classicals")
            captured_setup = (
                system.classical_setup(component) if core is not None else None
            )
            d3_setup = (
                captured_setup
                if isinstance(captured_setup, DispersionD3CoreSetup)
                else next(
                    (
                        setup
                        for d3_component, setup in system.d3_setups
                        if component is d3_component
                    ),
                    None,
                )
            )
            if type(component) is DispersionD3:
                if d3_setup is None:
                    raise RuntimeError("Single-system D3 setup is missing.")
                classical.append(
                    (
                        component.label,
                        dispersion_d3_energy(d3_setup, positions),
                    )
                )
            elif type(component) is DispersionD4:
                d4_setup = (
                    captured_setup
                    if isinstance(captured_setup, DispersionD4CoreSetup)
                    else (
                        system.d4_setup
                        if component is system.d4_classical
                        else None
                    )
                )
                if d4_setup is None:
                    raise RuntimeError("Single-system D4 setup is missing.")
                classical.append(
                    (
                        component.label,
                        dispersion_d4_energy(d4_setup, positions, charge),
                    )
                )
            elif type(component) is Halogen:
                halogen_setup = (
                    captured_setup
                    if isinstance(captured_setup, HalogenCoreSetup)
                    else (
                        system.halogen_setup
                        if component is system.halogen_classical
                        else None
                    )
                )
                if halogen_setup is None:
                    raise RuntimeError(
                        "Single-system halogen setup is missing."
                    )
                classical.append(
                    (component.label, halogen_energy(halogen_setup, positions))
                )
            else:
                if isinstance(captured_setup, RepulsionCoreSetup):
                    setup = captured_setup
                elif isinstance(captured_setup, ShortRangeBondCoreSetup):
                    setup = captured_setup
                elif component is system.repulsion_classical:
                    setup = system.repulsion_setup
                elif component is system.srb_classical:
                    setup = system.srb_setup
                elif (
                    isinstance(component, Repulsion)
                    and type(component) is not Repulsion
                ):
                    # Custom subclasses remain on their legacy extension path.
                    # Refresh their call data so supported update/reset
                    # behavior is observed; exact Repulsion uses System setup.
                    setup = component.get_cache(system.numbers, system.ihelp)
                elif (
                    isinstance(component, ShortRangeBond)
                    and type(component) is not ShortRangeBond
                ):
                    # Custom subclasses retain their legacy extension hooks.
                    setup = component.get_cache(system.numbers, system.ihelp)
                elif (
                    isinstance(component, DispersionD4)
                    and type(component) is not DispersionD4
                ):
                    # D4 subclasses are legacy extensions. Refresh their
                    # cache each evaluation so updates affect the next call.
                    setup = component.get_cache(system.numbers, system.ihelp)
                elif (
                    isinstance(component, Halogen)
                    and type(component) is not Halogen
                ):
                    # Custom subclasses retain cache-driven legacy behavior.
                    setup = component.get_cache(system.numbers, system.ihelp)
                elif isinstance(captured_setup, IESCoreSetup) or (
                    captured_setup is None and component is system.ies_classical
                ):
                    ies_setup = (
                        captured_setup
                        if isinstance(captured_setup, IESCoreSetup)
                        else system.ies_setup
                    )
                    if ies_setup is None:
                        raise RuntimeError(
                            "Single-system IES setup is missing."
                        )
                    classical.append(
                        (
                            component.label,
                            ies_energy(ies_setup, positions, charge),
                        )
                    )
                    timer.stop(component.label)
                    continue
                elif isinstance(component, IES) and type(component) is not IES:
                    # Custom IES subclasses retain their cache-driven hooks.
                    setup = component.get_cache(system.numbers, system.ihelp)
                else:
                    setup = system._classical_extension_cache[component.label]
                classical.append(
                    (
                        component.label,
                        component.get_energy(positions, setup, charge=charge),
                    )
                )
            timer.stop(component.label)
        classical_energy = torch.stack(
            tuple(value for _, value in classical)
        ).sum(0)
    else:
        classical = []
        classical_energy = positions.new_zeros(positions.shape[:-1])
    zero_energy = torch.zeros_like(classical_energy)

    if {"all", "scf"} & set(config.exclude):
        return Result.snapshot(
            energy=classical_energy,
            scf=zero_energy,
            classical=tuple(classical),
            fenergy=zero_energy,
            iterations=torch.zeros(
                (), dtype=torch.int64, device=positions.device
            ),
        )

    if h0_setup is None:
        raise NotImplementedError("Core Hamiltonian setup is missing.")
    if integral_setup is None:
        raise NotImplementedError(
            "Single-system integral setup is missing. Core evaluation does "
            "not support legacy batched Calculator setup."
        )
    if config.ints.level < labels.INTLEVEL_HCORE:
        raise NotImplementedError(
            "Core Hamiltonian missing. Skipping the Core Hamiltonian in "
            "the SCF is currently not supported. Please increase the "
            "integral level to at least '2'. Currently, the level is set "
            f"to '{config.ints.level}'."
        )

    matrices, refocc, overlap_norm = build_integral_matrices(
        integral_setup,
        h0_setup,
        positions,
        charge,
    )
    call_components = list(system.interactions.components)
    if field is not None:
        if any(
            component.label == LABEL_EFIELD for component in call_components
        ):
            raise ValueError(
                "Explicit field= cannot be combined with a custom interaction "
                f"using the '{LABEL_EFIELD}' label."
            )
        call_components.append(ElectricField(field))
    if field_grad is not None:
        if any(
            component.label == LABEL_EFIELD_GRAD
            for component in call_components
        ):
            raise ValueError(
                "Explicit field_grad= cannot be combined with a custom "
                f"interaction using the '{LABEL_EFIELD_GRAD}' label."
            )
        call_components.append(ElectricFieldGrad(field_grad))
    from dxtb._src.components.interactions.list import InteractionList

    call_interactions = InteractionList(
        *call_components, device=positions.device, dtype=positions.dtype
    )
    interaction_data = _interaction_data(system, positions, call_interactions)
    candidate_index = core.index if core is not None else None
    core_index = (
        candidate_index
        if candidate_index is not None
        and candidate_index.shells_to_atom.device
        == system.ihelp.shells_to_atom.device
        and candidate_index.orbitals_to_shell.device
        == system.ihelp.orbitals_to_shell.device
        else None
    )

    scf_solve = (
        scf.solve_with_core_index
        if core is not None and core_index is not None
        else scf.solve
    )
    scf_kwargs = (
        {
            "core_index": core_index,
            "core_index_owner": system,
        }
        if core is not None and core_index is not None
        else {}
    )
    if fixed_trip:
        from dxtb._src.scf.functional.eager import run_system_fixed_trip

        scf_results = run_system_fixed_trip(
            numbers,
            positions,
            charge,
            spin_tensor,
            call_interactions,
            interaction_data,
            system.ihelp,
            config.scf,
            matrices,
            refocc,
            core_index=core_index,
            core_index_owner=(system if core_index is not None else None),
        )
    else:
        scf_results = scf_solve(
            numbers,
            positions,
            charge,
            spin_tensor,
            call_interactions,
            interaction_data,
            system.ihelp,
            config.scf,
            matrices,
            refocc,
            **scf_kwargs,
        )
    scf_output = _normalize_scf_output(scf_results, positions)

    total_energy = classical_energy + scf_output.energy + scf_output.fenergy
    return Result.snapshot(
        energy=total_energy,
        scf=scf_output.energy,
        classical=tuple(classical),
        fenergy=scf_output.fenergy,
        charges=scf_output.charges,
        density=scf_output.density,
        coefficients=scf_output.coefficients,
        emo=scf_output.emo,
        occupation=scf_output.occupation,
        potential=scf_output.potential,
        hamiltonian=scf_output.hamiltonian,
        overlap=matrices.overlap,
        hcore=matrices.hcore,
        dipole_integrals=matrices.dipole,
        quadrupole_integrals=matrices.quadrupole,
        overlap_norm=overlap_norm,
        iterations=scf_output.iterations,
        converged=scf_output.converged,
        residual=scf_output.residual,
        status=scf_output.status,
    )
