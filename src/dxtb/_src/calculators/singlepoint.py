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
from dxtb._src.components.classicals.ies import IESCoreSetup, ies_energy
from dxtb._src.components.classicals.repulsion.base import repulsion_energy
from dxtb._src.components.classicals.repulsion.rep import (
    RepulsionCoreSetup,
)
from dxtb._src.components.classicals.shortrangebond import (
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
    """Keep iteration tensors and convert Python integer counts to tensors."""
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.int64, device=like.device)


@dataclass(frozen=True, eq=False)
class _SCFOutput:
    """Common read-only view over the non-SCF and SCF result contracts."""

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
    """Read SCF outputs once without copying tensors."""
    from dxtb._src.scf.eager import (
        SCFCoreResult,
        public_scf_status,
    )

    if isinstance(result, SCFCoreResult):
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
        return dispersion_d3_energy(setup, positions)  # type: ignore[arg-type]
    if descriptor.kind == "repulsion":
        energy = repulsion_energy(
            positions,
            setup.mask,  # type: ignore[union-attr]
            setup.arep,  # type: ignore[union-attr]
            setup.kexp,  # type: ignore[union-attr]
            setup.zeff,  # type: ignore[union-attr]
            setup.cutoff,  # type: ignore[union-attr]
        )
        return 0.5 * torch.sum(energy, dim=-1)
    if descriptor.kind == "srb":
        return short_range_bond_energy(setup, positions)  # type: ignore[arg-type]
    if descriptor.kind == "d4":
        return dispersion_d4_energy(setup, positions, charge, q=q)  # type: ignore[arg-type]
    if descriptor.kind == "ies":
        return ies_energy(setup, positions, charge)  # type: ignore[arg-type]
    if descriptor.kind == "halogen":
        return halogen_energy(setup, positions)  # type: ignore[arg-type]
    raise RuntimeError(f"Unknown classical term kind '{descriptor.kind}'.")


def _build_builtin_interaction_cache(setup: Any, positions: Tensor) -> Any:
    """Build call-local cache data from a captured built-in interaction setup."""
    if isinstance(setup, ES2CoreSetup):
        return ES2Cache(
            build_es2_coulomb(setup, positions),
            shell_resolved=setup.shell_resolved,
        )
    if isinstance(setup, ES3CoreSetup):
        return ES3Cache(
            setup.hubbard_derivs,
            shell_resolved=setup.shell_resolved,
        )
    if isinstance(setup, AES2CoreSetup):
        return build_aes2_data(setup, positions)
    if isinstance(setup, D4SCCoreSetup):
        return build_d4sc_data(setup, positions)
    if isinstance(setup, GeneralizedBornCoreSetup):
        return build_generalized_born_data(setup, positions)
    raise TypeError(f"Unsupported built-in interaction setup {type(setup)}.")


class _SystemInteractionData(InteractionListCache):
    """Call-local interaction caches with their captured D4SC authority."""

    def __init__(
        self,
        d4sc_setup: D4SCCoreSetup | None,
        scf_kinds: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.d4sc_setup = d4sc_setup
        self.scf_kinds = {} if scf_kinds is None else scf_kinds


def _interaction_data(
    system: System, positions: Tensor, interactions=None
) -> InteractionListCache:
    """Build call-local interaction data for one SCF evaluation."""
    interactions = system.interactions if interactions is None else interactions
    core = system._active_exact_core()
    descriptors = (
        {term.position: term for term in core.interaction_terms}
        if core is not None
        else {}
    )
    scf_kinds: dict[str, str] = {}
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
    data = _SystemInteractionData(authoritative_d4sc_setup, scf_kinds)
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
            scf_kinds[descriptor.label] = descriptor.kind
            data[descriptor.label] = _build_builtin_interaction_cache(
                descriptor.setup, positions
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
        elif isinstance(
            captured_setup,
            (
                ES2CoreSetup,
                ES3CoreSetup,
                AES2CoreSetup,
                D4SCCoreSetup,
                GeneralizedBornCoreSetup,
            ),
        ):
            data[interaction.label] = _build_builtin_interaction_cache(
                captured_setup, positions
            )
        elif captured_setup is None and interaction in (
            system.es2_interaction,
            system.es3_interaction,
            system.aes2_interaction,
            system.d4sc_interaction,
            system.generalized_born_interaction,
        ):
            setup_pairs = (
                (system.es2_interaction, system.es2_setup, "ES2"),
                (system.es3_interaction, system.es3_setup, "ES3"),
                (system.aes2_interaction, system.aes2_setup, "AES2"),
                (system.d4sc_interaction, system.d4sc_setup, "D4SC"),
                (
                    system.generalized_born_interaction,
                    system.generalized_born_setup,
                    "GeneralizedBorn",
                ),
            )
            setup, name = next(
                (s, label)
                for owner, s, label in setup_pairs
                if interaction is owner
            )
            if setup is None:
                raise RuntimeError(f"Single-system {name} setup is missing.")
            data[interaction.label] = _build_builtin_interaction_cache(
                setup, positions
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
    if setup is not None and setup.backend is not None and setup.backend.alphas:
        return setup.backend.alphas[0]
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
    """Evaluate one System with the Eager controller."""
    return _singlepoint_impl(
        system,
        positions,
        chrg,
        spin,
        field,
        field_grad,
        fixed_trip=False,
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


@dataclass(frozen=True, eq=False)
class _PreparedSinglepoint:
    """Evaluated non-SCF terms and the inputs of one System's SCF call."""

    positions: Tensor
    classical: tuple[tuple[str, Tensor], ...]
    classical_energy: Tensor
    matrices: Any
    refocc: Tensor
    numbers: Tensor
    charge: Tensor
    spin: Tensor | None
    interactions: Any
    interaction_data: Any
    ihelp: Any
    config: Any
    core_index: Any
    core_index_owner: System | None

    def scf_arguments(self) -> tuple[Any, ...]:
        """Positional arguments shared by every SCF entry point."""
        return (
            self.numbers,
            self.positions,
            self.charge,
            self.spin,
            self.interactions,
            self.interaction_data,
            self.ihelp,
            self.config.scf,
            self.matrices,
            self.refocc,
        )


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
    prepared = _prepare_singlepoint(
        system,
        positions,
        chrg,
        spin,
        field,
        field_grad,
    )
    if isinstance(prepared, Result):
        return prepared

    if fixed_trip and prepared.config.scf.differentiation.value != "implicit":
        from dxtb._src.scf.eager import run_system_fixed_trip

        scf_results = run_system_fixed_trip(
            *prepared.scf_arguments(),
            core_index=prepared.core_index,
            core_index_owner=prepared.core_index_owner,
        )
    elif prepared.core_index is not None:
        scf_results = scf.solve_with_core_index(
            *prepared.scf_arguments(),
            core_index=prepared.core_index,
            core_index_owner=prepared.core_index_owner,
        )
    else:
        scf_results = scf.solve(*prepared.scf_arguments())
    return _singlepoint_result(prepared, scf_results)


def _singlepoint_result(
    prepared: _PreparedSinglepoint, scf_results: Any
) -> Result:
    """Assemble the public Result from non-SCF terms and SCF outputs."""
    matrices = prepared.matrices
    scf_output = _normalize_scf_output(scf_results, prepared.positions)
    total_energy = (
        prepared.classical_energy + scf_output.energy + scf_output.fenergy
    )
    return Result.snapshot(
        energy=total_energy,
        scf=scf_output.energy,
        classical=prepared.classical,
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
        iterations=scf_output.iterations,
        converged=scf_output.converged,
        residual=scf_output.residual,
        status=scf_output.status,
    )


def _prepare_singlepoint(
    system: System,
    positions: Tensor,
    chrg: Tensor | float | int,
    spin: Tensor | float | int | None,
    field: Tensor | None,
    field_grad: Tensor | None,
) -> _PreparedSinglepoint | Result:
    """Evaluate everything before the SCF and collect the SCF inputs.

    Returns the final Result directly when the SCF is excluded.
    """
    core = system._active_exact_core()
    numbers = core.numbers if core is not None else system.numbers
    config = core.config if core is not None else system.config
    # Keep the IntegralSetup facade at this evaluation boundary. The
    # active core has already verified that its selected backend Node is the
    # same authoritative child held by this facade.
    integral_setup = system.integral_setup
    h0_setup = core.h0_setup if core is not None else system.h0_setup

    if numbers.ndim != 1:
        raise ValueError("Core singlepoint accepts one Structure at a time.")
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
    if core is None and not system._ihelp_compatible():
        raise ValueError(
            "System IndexHelper is incompatible with the captured System setup."
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
                else:
                    raise RuntimeError(
                        f"No setup for the Classical '{component.label}'."
                    )
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
            "not support a padded or conformer System."
        )
    if config.ints.level < labels.INTLEVEL_HCORE:
        raise NotImplementedError(
            "Core Hamiltonian missing. Skipping the Core Hamiltonian in "
            "the SCF is currently not supported. Please increase the "
            "integral level to at least '2'. Currently, the level is set "
            f"to '{config.ints.level}'."
        )

    matrices, refocc = build_integral_matrices(
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

    return _PreparedSinglepoint(
        positions=positions,
        classical=tuple(classical),
        classical_energy=classical_energy,
        matrices=matrices,
        refocc=refocc,
        numbers=numbers,
        charge=charge,
        spin=spin_tensor,
        interactions=call_interactions,
        interaction_data=interaction_data,
        ihelp=system.ihelp,
        config=config,
        core_index=core_index if core is not None else None,
        core_index_owner=(
            system if core is not None and core_index is not None else None
        ),
    )
