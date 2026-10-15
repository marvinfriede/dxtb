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
"""Pure interaction dispatch over frozen single-System core values."""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb._src.components.interactions import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.components.interactions.coulomb.multipole import AES2
from dxtb._src.components.interactions.coulomb.secondorder import ES2
from dxtb._src.components.interactions.coulomb.thirdorder import ES3
from dxtb._src.components.interactions.dispersion.d4sc import (
    D4SCCoreSetup,
    DispersionD4SC,
    D4SCSetup,
    _d4sc_setup_view,
    _build_d4sc_model,
    setup_d4sc,
)
from dxtb._src.components.interactions.field.efield import ElectricField
from dxtb._src.components.interactions.field.efieldgrad import ElectricFieldGrad
from dxtb._src.components.interactions.solvation.alpb import GeneralizedBorn
from dxtb._src.typing import Tensor

from .core import (
    CoreIndexData,
    FunctionalInteractionEntry,
    FunctionalInteractionSet,
)
from .data import SCFCharges, SCFPotential


def prepare_functional_interactions(
    interactions: InteractionList,
    interaction_data: InteractionListCache,
    numbers: Tensor,
    *,
    d4sc_setup: D4SCCoreSetup | None = None,
) -> FunctionalInteractionSet:
    """Convert supported components/caches to frozen core entries.

    ``d4sc_setup`` is supplied by the single-System call-data boundary so
    its functional reference weights use the same setup snapshot as the
    geometry-dependent D4SC cache. Direct compatibility callers may omit it.
    """
    entries: list[FunctionalInteractionEntry] = []
    for component in interactions.components:
        cache = interaction_data[component.label]
        kind = type(component)
        if kind is ES2:
            entries.append(
                FunctionalInteractionEntry(
                    "es2",
                    (("mat", cache.mat),),
                    (("shell_resolved", cache.shell_resolved),),
                )
            )
        elif kind is ES3:
            entries.append(
                FunctionalInteractionEntry(
                    "es3",
                    (("hd", cache.hd),),
                    (("shell_resolved", cache.shell_resolved),),
                )
            )
        elif kind is AES2:
            entries.append(
                FunctionalInteractionEntry(
                    "aes2",
                    tuple(
                        (name, getattr(cache, name))
                        for name in (
                            "dkernel",
                            "qkernel",
                            "amat_sd",
                            "amat_dd",
                            "amat_sq",
                        )
                    ),
                )
            )
        elif kind is DispersionD4SC:
            setup = (
                _d4sc_setup_view(d4sc_setup)
                if d4sc_setup is not None
                else setup_d4sc(component, numbers)
            )
            entries.append(
                FunctionalInteractionEntry(
                    "d4sc",
                    (("cn", cache.cn), ("dispmat", cache.dispmat)),
                    (("setup", setup),),
                )
            )
        elif kind is GeneralizedBorn:
            entries.append(
                FunctionalInteractionEntry(
                    "generalized_born", (("mat", cache.mat),)
                )
            )
        elif kind is ElectricField:
            entries.append(
                FunctionalInteractionEntry(
                    "electric_field",
                    (("vat", cache.vat), ("vdp", cache.vdp)),
                )
            )
        elif kind is ElectricFieldGrad:
            entries.append(
                FunctionalInteractionEntry(
                    "electric_field_grad",
                    (
                        ("vat", cache.vat),
                        ("vdp", cache.vdp),
                        ("vqp", cache.vqp),
                    ),
                )
            )
        else:  # guarded by exact-type eligibility before conversion
            raise TypeError(f"Unsupported functional interaction {kind!r}.")
    return FunctionalInteractionSet(tuple(entries))


def _optional_add(left: Tensor | None, right: Tensor | None) -> Tensor | None:
    if right is None:
        return left
    return right if left is None else left + right


def _potential_parts(
    entry: FunctionalInteractionEntry,
    charges: SCFCharges,
    index: CoreIndexData,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Return shell/atom monopole and optional multipole potentials."""
    mono = charges.mono
    shell_charges = index.reduce_orbital_to_shell(mono)
    atom_charges = index.reduce_shell_to_atom(shell_charges)
    shell = torch.zeros_like(shell_charges)
    atom = torch.zeros_like(atom_charges)
    dipole = None
    quadrupole = None

    if entry.kind == "es2":
        resolved = entry.option("shell_resolved")
        if resolved:
            shell = einsum(
                "...ik,...k->...i", entry.tensor("mat"), shell_charges
            )
        else:
            atom = einsum("...ik,...k->...i", entry.tensor("mat"), atom_charges)
    elif entry.kind == "es3":
        resolved = entry.option("shell_resolved")
        if resolved:
            shell = entry.tensor("hd") * shell_charges.square()
        else:
            atom = entry.tensor("hd") * atom_charges.square()
    elif entry.kind == "aes2":
        if charges.dipole is None or charges.quad is None:
            raise RuntimeError("AES2 requires dipole and quadrupole charges.")
        dipole_charges = charges.dipole
        quad_charges = charges.quad
        atom = einsum(
            "...ijx,...jx->...i", entry.tensor("amat_sd"), dipole_charges
        )
        atom = atom + einsum(
            "...ijx,...jx->...i", entry.tensor("amat_sq"), quad_charges
        )
        dipole = (
            einsum("...ijx,...i->...jx", entry.tensor("amat_sd"), atom_charges)
            + einsum(
                "...ijxy,...ix->...jy", entry.tensor("amat_dd"), dipole_charges
            )
            + 2 * entry.tensor("dkernel") * dipole_charges
        )
        scale = quad_charges.new_tensor([1, 2, 1, 2, 2, 1])
        quadrupole = (
            einsum("...ijx,...i->...jx", entry.tensor("amat_sq"), atom_charges)
            + 2 * entry.tensor("qkernel") * quad_charges * scale
        )
    elif entry.kind == "d4sc":
        setup = entry.option("setup")
        assert isinstance(setup, D4SCSetup)
        model = _build_d4sc_model(setup)
        weights, derivative = model.weight_references(
            entry.tensor("cn"), atom_charges, with_dgwdq=True
        )
        atom = einsum(
            "...ijab,...jb,...ia->...i",
            entry.tensor("dispmat"),
            weights,
            derivative,
        )
    elif entry.kind == "generalized_born":
        atom = einsum("...ik,...k->...i", entry.tensor("mat"), atom_charges)
    elif entry.kind == "electric_field":
        atom = -entry.tensor("vat")
        dipole = -entry.tensor("vdp")
    elif entry.kind == "electric_field_grad":
        atom = entry.tensor("vat")
        dipole = entry.tensor("vdp")
        quadrupole = entry.tensor("vqp")
    else:
        raise TypeError(f"Unknown functional interaction kind {entry.kind!r}.")

    return shell, atom, dipole, quadrupole


def potential_from_charges(
    charges: SCFCharges,
    interactions: FunctionalInteractionSet,
    index: CoreIndexData,
) -> SCFPotential:
    """Evaluate total potential by pure addition of immutable contributions."""
    mono = torch.zeros_like(charges.mono)
    dipole = None
    quadrupole = None
    for entry in interactions.entries:
        shell, atom, dipole_part, quad_part = _potential_parts(
            entry, charges, index
        )
        shell = shell + index.spread_atom_to_shell(atom)
        mono = mono + index.spread_shell_to_orbital(shell)
        dipole = _optional_add(dipole, dipole_part)
        quadrupole = _optional_add(quadrupole, quad_part)
    return SCFPotential(mono=mono, dipole=dipole, quad=quadrupole)


def _entry_energy(
    entry: FunctionalInteractionEntry,
    charges: SCFCharges,
    shell_charges: Tensor,
    atom_charges: Tensor,
) -> Tensor:
    """Return one entry's atom-resolved interaction energy."""
    result = torch.zeros_like(atom_charges)
    if entry.kind == "es2":
        resolved = entry.option("shell_resolved")
        if not resolved:
            atom_potential = einsum(
                "...ik,...k->...i", entry.tensor("mat"), atom_charges
            )
            result = result + 0.5 * atom_charges * atom_potential
    elif entry.kind == "es3":
        resolved = entry.option("shell_resolved")
        if not resolved:
            result = result + entry.tensor("hd") * atom_charges.pow(3) / 3
    elif entry.kind == "aes2":
        if charges.dipole is not None:
            qdp = charges.dipole
            vdp_sd = einsum(
                "...ijx,...i->...jx", entry.tensor("amat_sd"), atom_charges
            )
            vdp_dd = 0.5 * einsum(
                "...ijxy,...ix->...jy", entry.tensor("amat_dd"), qdp
            )
            ke = einsum(
                "...ix,...ix,...ix->...i", entry.tensor("dkernel"), qdp, qdp
            )
            result = (
                result + ke + einsum("...ix,...ix->...i", vdp_sd + vdp_dd, qdp)
            )
        if charges.quad is not None:
            qqp = charges.quad
            scale = qqp.new_tensor([1, 2, 1, 2, 2, 1])
            vqp = einsum(
                "...ijx,...i->...jx", entry.tensor("amat_sq"), atom_charges
            )
            ke = einsum(
                "...ix,...ix,x,...ix->...i",
                entry.tensor("qkernel"),
                qqp,
                scale,
                qqp,
            )
            result = result + ke + einsum("...ix,...ix->...i", vqp, qqp)
    elif entry.kind == "d4sc":
        setup = entry.option("setup")
        assert isinstance(setup, D4SCSetup)
        weights = _build_d4sc_model(setup).weight_references(
            entry.tensor("cn"), atom_charges
        )
        result = result + 0.5 * einsum(
            "...ijab,...ia,...jb->...j",
            entry.tensor("dispmat"),
            weights,
            weights,
            optimize=[(0, 1), (0, 1)],
        )
    elif entry.kind == "generalized_born":
        potential = einsum(
            "...ik,...k->...i", entry.tensor("mat"), atom_charges
        )
        result = result + 0.5 * atom_charges * potential
    elif entry.kind == "electric_field":
        result = result - entry.tensor("vat") * atom_charges
        if charges.dipole is not None:
            result = result + einsum(
                "...ix,...ix->...i", -entry.tensor("vdp"), charges.dipole
            )
    elif entry.kind == "electric_field_grad":
        result = result + entry.tensor("vat") * atom_charges
        if charges.dipole is not None:
            result = result + einsum(
                "...ix,...ix->...i", entry.tensor("vdp"), charges.dipole
            )
        if charges.quad is not None:
            result = result + einsum(
                "...ix,...ix->...i", entry.tensor("vqp"), charges.quad
            )
    return result


def interaction_energy_from_charges(
    charges: SCFCharges,
    interactions: FunctionalInteractionSet,
    index: CoreIndexData,
) -> Tensor:
    """Evaluate atom-resolved energy using immutable core interaction data."""
    shell_charges = index.reduce_orbital_to_shell(charges.mono)
    atom_charges = index.reduce_shell_to_atom(shell_charges)
    result = atom_charges.new_zeros(atom_charges.shape)
    for entry in interactions.entries:
        local = _entry_energy(entry, charges, shell_charges, atom_charges)
        if entry.kind == "es2" and entry.option("shell_resolved"):
            shell_energy = entry.tensor("mat")
            shell_energy = (
                0.5
                * shell_charges
                * einsum("...ik,...k->...i", shell_energy, shell_charges)
            )
            local = local + index.reduce_shell_to_atom(shell_energy)
        if entry.kind == "es3" and entry.option("shell_resolved"):
            shell_energy = entry.tensor("hd") * shell_charges.pow(3) / 3
            local = local + index.reduce_shell_to_atom(shell_energy)
        result = result + local
    return result
