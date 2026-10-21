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
"""Validity-bearing structural padding for compatible exact cores.

This first padding slice supports same-method PyTorch GFN1 cores whose
shell-class and descriptor structures agree. It intentionally does not alter
ordinary ``ExactSystemCore`` values or claim that padded SCF/integral
evaluation is supported. The padded pair plan uses the larger core's ordered
shell-pair structure; entries absent from a smaller source are identified by
the accompanying validity masks and occupy padded AO rows or columns only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tad_mctc.tree import Node, child
from torch.utils._pytree import tree_flatten

from dxtb._src.calculators.model import ExactSystemCore
from dxtb._src.components.classicals.dispersion.d3 import (
    DispersionD3CoreSetup,
)
from dxtb._src.components.classicals.halogen.hal import HalogenCoreSetup
from dxtb._src.components.classicals.repulsion.base import RepulsionCoreSetup
from dxtb._src.components.interactions.coulomb.secondorder import ES2CoreSetup
from dxtb._src.components.interactions.coulomb.thirdorder import ES3CoreSetup
from dxtb._src.constants import labels
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    PairClassNode,
    PairPlanNode,
)
from dxtb._src.integral.driver.pytorch.setup import PytorchIntegralSetupNode
from dxtb._src.scf.functional.core import CoreIndexData
from dxtb._src.typing import Tensor
from dxtb._src.xtb.h0 import H0IndexNode

__all__ = [
    "CoreBucket",
    "PaddedCoreValidity",
    "PaddedExactSystemCore",
    "bucket_exact_cores",
    "compatibility_signature",
    "pad_exact_core",
    "pad_positions",
]


class PaddedCoreValidity(Node):
    """Dynamic masks for structural entries that are physically present."""

    atoms: Tensor = child(keep_dtype=True)
    shells: Tensor = child(keep_dtype=True)
    orbitals: Tensor = child(keep_dtype=True)
    unique_elements: Tensor = child(keep_dtype=True)
    unique_shell_types: Tensor = child(keep_dtype=True)
    pair_entries: tuple[Tensor, ...] = child()


class PaddedExactSystemCore(Node):
    """Complete registered core plus explicit structural validity."""

    core: ExactSystemCore = child()
    validity: PaddedCoreValidity = child()


@dataclass(frozen=True)
class CoreBucket:
    """A deterministic group of compatible input cores and its capacities."""

    indices: tuple[int, ...]
    signature: tuple[object, ...]
    atoms: int
    shells: int
    orbitals: int
    unique_elements: int
    unique_shell_types: int
    pair_entries: tuple[int, ...]


def _backend(core: ExactSystemCore) -> PytorchIntegralSetupNode:
    if core.integral is None or not isinstance(
        core.integral.backend, PytorchIntegralSetupNode
    ):
        raise TypeError(
            "Mixed-size padding currently requires a PyTorch backend."
        )
    return core.integral.backend


def _callable_signature(value: object) -> tuple[str, str]:
    function = getattr(value, "function", value)
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise TypeError("Unsupported opaque callable in exact-core metadata.")
    return module, name


def _setup_static_signature(setup: Node) -> tuple[object, ...]:
    if isinstance(setup, DispersionD3CoreSetup):
        return (
            _callable_signature(setup.counting_function),
            _callable_signature(setup.weighting_function),
            _callable_signature(setup.damping_function),
            tuple(
                (entry.name, entry.value.kind, entry.value.static_value)
                for entry in setup.parameter_entries
            ),
        )
    if isinstance(setup, ES2CoreSetup):
        return (
            setup.shell_resolved,
            _callable_signature(setup.average),
            setup.atom_hubbard is not None,
            setup.shell_pair_mask is not None,
            setup.shell_mask is not None,
            setup.shell_hubbard is not None,
            setup.shells_to_atom is not None,
        )
    if isinstance(setup, ES3CoreSetup):
        return (setup.shell_resolved,)
    if isinstance(setup, (HalogenCoreSetup, RepulsionCoreSetup)):
        return ()
    raise TypeError(f"Unsupported exact-core setup {type(setup).__name__}.")


def compatibility_signature(core: ExactSystemCore) -> tuple[object, ...]:
    """Return a value-based structural signature, excluding capacities."""
    backend = _backend(core)
    plan = backend.pair_plan
    if (
        core.integral is None
        or core.integral.logical_device != core.numbers.device
    ):
        raise ValueError("Core logical device disagrees with its atom data.")
    if plan.nao != backend.index.orbitals_to_shell.numel():
        raise ValueError("PairPlan AO stride disagrees with backend index.")
    if not torch.equal(plan.atom, backend.index.shells_to_atom):
        raise ValueError("PairPlan atom map disagrees with backend index.")
    if core.h0_setup is not None and (
        not torch.equal(core.h0_setup.numbers, core.numbers)
        or not torch.equal(
            core.h0_setup.index.shells_to_atom,
            backend.index.shells_to_atom,
        )
        or not torch.equal(
            core.h0_setup.index.orbitals_to_shell,
            backend.index.orbitals_to_shell,
        )
        or not torch.equal(
            core.h0_setup.index.unique_angular,
            torch.tensor(
                plan.unique_angular,
                dtype=core.h0_setup.index.unique_angular.dtype,
                device=core.h0_setup.index.unique_angular.device,
            ),
        )
    ):
        raise ValueError("H0 mappings disagree with the selected backend.")
    if core.config.method != labels.GFN1_XTB:
        raise TypeError(
            "This padding slice currently supports GFN1 cores only."
        )

    leaves, _ = tree_flatten(core)
    tensor_policy = tuple(
        (leaf.dtype, leaf.device, leaf.layout)
        for leaf in leaves
        if isinstance(leaf, torch.Tensor)
    )
    return (
        core.config,
        core.effective_dtype,
        core.numbers.device,
        core.integral.driver_type if core.integral is not None else None,
        core.integral.intlevel if core.integral is not None else None,
        core.integral.logical_device if core.integral is not None else None,
        type(backend),
        backend.algorithm,
        tuple(plan.unique_angular),
        tuple((item.ub, item.uk, item.lb, item.lk) for item in plan.classes),
        tuple((tuple(a.shape), a.dtype, a.device) for a in backend.alphas),
        tuple((tuple(c.shape), c.dtype, c.device) for c in backend.coeffs),
        tensor_policy,
        (
            None
            if core.h0_setup is None
            else (
                core.h0_setup.method,
                core.h0_setup.cn_kind,
                core.h0_setup.kq is not None,
                core.h0_setup.eeq_model is not None,
                core.h0_setup.unique.numel(),
                core.h0_setup.index.unique_angular.numel(),
            )
        ),
        tuple(
            (
                term.kind,
                term.label,
                term.position,
                term.geometry_inputs,
                type(term.setup),
                _setup_static_signature(term.setup),
            )
            for term in core.classical_terms
        ),
        tuple(
            (
                term.kind,
                term.label,
                term.position,
                term.geometry_inputs,
                type(term.setup),
                _setup_static_signature(term.setup),
            )
            for term in core.interaction_terms
        ),
    )


def _capacities(
    core: ExactSystemCore,
) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    backend = _backend(core)
    return (
        core.numbers.numel(),
        backend.index.shells_to_atom.numel(),
        backend.index.orbitals_to_shell.numel(),
        0 if core.h0_setup is None else core.h0_setup.unique.numel(),
        len(backend.pair_plan.unique_angular),
        tuple(item.ib.numel() for item in backend.pair_plan.classes),
    )


def bucket_exact_cores(
    cores: tuple[ExactSystemCore, ...],
) -> tuple[CoreBucket, ...]:
    """Partition cores deterministically by structure and report max capacity."""
    groups: dict[tuple[object, ...], list[int]] = {}
    for i, core in enumerate(cores):
        groups.setdefault(compatibility_signature(core), []).append(i)

    result = []
    for signature, indices in groups.items():
        sizes = [_capacities(cores[i]) for i in indices]
        nclass = len(sizes[0][5])
        if any(len(size[5]) != nclass for size in sizes):
            raise ValueError("Compatible cores disagree on pair-class count.")
        result.append(
            CoreBucket(
                indices=tuple(indices),
                signature=signature,
                atoms=max(size[0] for size in sizes),
                shells=max(size[1] for size in sizes),
                orbitals=max(size[2] for size in sizes),
                unique_elements=max(size[3] for size in sizes),
                unique_shell_types=max(size[4] for size in sizes),
                pair_entries=tuple(
                    max(size[5][j] for size in sizes) for j in range(nclass)
                ),
            )
        )
    return tuple(result)


def _pad_vector(
    value: Tensor, size: int, fill: int | float | bool = 0
) -> Tensor:
    if value.ndim != 1 or value.shape[0] > size:
        raise ValueError(
            "Expected a one-dimensional value within bucket capacity."
        )
    if value.shape[0] == size:
        return value
    padding = value.new_full((size - value.shape[0],), fill)
    return torch.cat((value, padding))


def _pad_matrix(
    value: Tensor, size: int, fill: int | float | bool = 0
) -> Tensor:
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or value.shape[0] > size
    ):
        raise ValueError("Expected a square matrix within bucket capacity.")
    if value.shape[0] == size:
        return value
    out = value.new_full((size, size), fill)
    return out.index_put(
        (
            torch.arange(value.shape[0], device=value.device)[:, None],
            torch.arange(value.shape[1], device=value.device)[None, :],
        ),
        value,
    )


def _pad_indices(source: Tensor, template: Tensor, real: int) -> Tensor:
    """Keep real mappings and use validated template values after them."""
    if (
        source.ndim != 1
        or template.ndim != 1
        or source.numel() > template.numel()
    ):
        raise ValueError("Index mapping does not fit the selected capacity.")
    if source.numel() != real:
        raise ValueError(
            "Index mapping real extent disagrees with validity mask."
        )
    return torch.cat((source, template[real:]))


def _pad_pair_plan(
    source: PairPlanNode,
    template: PairPlanNode,
    old_nao: int,
) -> tuple[PairPlanNode, tuple[Tensor, ...]]:
    if source.unique_angular != template.unique_angular:
        raise ValueError(
            "Unique angular classes cannot be remapped in this slice."
        )
    source_context = tuple((c.ub, c.uk, c.lb, c.lk) for c in source.classes)
    template_context = tuple((c.ub, c.uk, c.lb, c.lk) for c in template.classes)
    if source_context != template_context or source.nao > template.nao:
        raise ValueError("Pair-plan classes are incompatible with the bucket.")

    classes: list[PairClassNode] = []
    masks: list[Tensor] = []
    real_destinations: list[Tensor] = []
    invalid_destinations: list[Tensor] = []
    for old, full in zip(source.classes, template.classes):
        old_pairs = {
            (int(i), int(j)): n for n, (i, j) in enumerate(zip(old.ib, old.jk))
        }
        mask = torch.tensor(
            [(int(i), int(j)) in old_pairs for i, j in zip(full.ib, full.jk)],
            dtype=torch.bool,
            device=old.ib.device,
        )
        positions = torch.tensor(
            [
                old_pairs.get((int(i), int(j)), 0)
                for i, j in zip(full.ib, full.jk)
            ],
            dtype=torch.long,
            device=old.ib.device,
        )
        if int(mask.sum()) != old.ib.numel():
            raise ValueError(
                "Source pair entries are not a subset of bucket pairs."
            )
        if mask.any() and not torch.equal(
            old.ib[positions[mask]], full.ib[mask]
        ):
            raise ValueError("Real pair ordering changed during padding.")
        if mask.any() and not torch.equal(
            old.jk[positions[mask]], full.jk[mask]
        ):
            raise ValueError("Real pair ordering changed during padding.")

        old_idx = old.idx.index_select(0, positions)
        rows = torch.div(old_idx, old_nao, rounding_mode="floor")
        cols = old_idx.remainder(old_nao)
        reindexed = rows * template.nao + cols
        if mask.any() and not torch.equal(reindexed[mask], full.idx[mask]):
            raise ValueError(
                "Real PairPlan AO indices do not match the canonical padded stride."
            )
        real_destinations.append(reindexed[mask].reshape(-1))
        invalid_destinations.append(full.idx[~mask].reshape(-1))
        idx = torch.where(mask[:, None, None], reindexed, full.idx)
        weight = torch.where(
            mask, old.weight.index_select(0, positions), full.weight
        )
        classes.append(
            PairClassNode(
                ub=full.ub,
                uk=full.uk,
                lb=full.lb,
                lk=full.lk,
                ib=full.ib,
                jk=full.jk,
                idx=idx,
                weight=weight,
            )
        )
        masks.append(mask)

    real_flat = torch.cat(real_destinations)
    invalid_flat = torch.cat(invalid_destinations)
    if torch.unique(real_flat).numel() != real_flat.numel():
        raise ValueError("Pair-plan real AO destinations are not unique.")
    if torch.unique(invalid_flat).numel() != invalid_flat.numel():
        raise ValueError("Padded pair entries have colliding AO destinations.")
    if torch.isin(invalid_flat, real_flat).any():
        raise ValueError(
            "Padded pair destinations collide with real AO entries."
        )
    invalid_rows = torch.div(invalid_flat, template.nao, rounding_mode="floor")
    invalid_cols = invalid_flat.remainder(template.nao)
    if not ((invalid_rows >= old_nao) | (invalid_cols >= old_nao)).all():
        raise ValueError("Invalid pair entry targets the real AO submatrix.")

    return (
        PairPlanNode(
            nao=template.nao,
            unique_angular=template.unique_angular,
            atom=template.atom,
            classes=tuple(classes),
        ),
        tuple(masks),
    )


def _pad_core(
    core: ExactSystemCore, template: ExactSystemCore
) -> tuple[ExactSystemCore, PaddedCoreValidity]:
    backend = _backend(core)
    template_backend = _backend(template)
    nreal = core.numbers.numel()
    nshell = backend.index.shells_to_atom.numel()
    nao = backend.index.orbitals_to_shell.numel()
    ntarget, nshell_target, nao_target, _, _, _ = _capacities(template)
    if nreal > ntarget or nshell > nshell_target or nao > nao_target:
        raise ValueError("Core exceeds the selected bucket capacity.")
    if not torch.equal(core.numbers, template.numbers[:nreal]):
        raise ValueError(
            "Mixed-size padding requires the smaller atom sequence to be a "
            "prefix of the bucket representative."
        )
    if (
        backend.pair_plan.nao != nao
        or template_backend.pair_plan.nao != nao_target
    ):
        raise ValueError("PairPlan AO stride disagrees with orbital capacity.")
    if not torch.equal(
        backend.index.shells_to_atom,
        template_backend.index.shells_to_atom[:nshell],
    ):
        raise ValueError(
            "Shell-to-atom mapping is not a prefix of the bucket map."
        )
    if not torch.equal(
        backend.index.orbitals_to_shell,
        template_backend.index.orbitals_to_shell[:nao],
    ):
        raise ValueError(
            "Orbital-to-shell mapping is not a prefix of the bucket map."
        )
    if not torch.equal(backend.pair_plan.atom, backend.index.shells_to_atom):
        raise ValueError(
            "Source PairPlan atom mapping disagrees with backend index."
        )
    if core.h0_setup is not None:
        template_h0_check = template.h0_setup
        if template_h0_check is None:
            raise ValueError("Optional H0 setup differs within bucket.")
        if not torch.equal(core.h0_setup.numbers, core.numbers):
            raise ValueError("H0 atom numbers disagree with core numbers.")
        if not torch.equal(core.h0_setup.unique, template_h0_check.unique):
            raise ValueError("Unique element mapping differs within bucket.")
        if not torch.equal(
            core.h0_setup.index.atom_to_unique,
            template_h0_check.index.atom_to_unique[:nreal],
        ):
            raise ValueError("Atom-to-unique mapping is not a bucket prefix.")
        if not torch.equal(
            core.h0_setup.index.shells_to_ushell,
            template_h0_check.index.shells_to_ushell[:nshell],
        ):
            raise ValueError("Shell-to-unique mapping is not a bucket prefix.")
        if not torch.equal(
            core.h0_setup.index.unique_angular,
            template_h0_check.index.unique_angular,
        ):
            raise ValueError("H0 unique shell classes differ within bucket.")
        if not torch.equal(
            core.h0_setup.index.shells_to_atom,
            backend.index.shells_to_atom,
        ) or not torch.equal(
            core.h0_setup.index.orbitals_to_shell,
            backend.index.orbitals_to_shell,
        ):
            raise ValueError("H0 and integral index ownership disagree.")

    index = CoreIndexData(
        shells_to_atom=_pad_indices(
            backend.index.shells_to_atom,
            template_backend.index.shells_to_atom,
            nshell,
        ),
        orbitals_to_shell=_pad_indices(
            backend.index.orbitals_to_shell,
            template_backend.index.orbitals_to_shell,
            nao,
        ),
    )
    plan, pair_masks = _pad_pair_plan(
        backend.pair_plan, template_backend.pair_plan, backend.pair_plan.nao
    )
    plan = plan.replace(atom=index.shells_to_atom)
    padded_backend = backend.replace(index=index, pair_plan=plan)

    h0 = core.h0_setup
    if h0 is not None:
        template_h0 = template.h0_setup
        if template_h0 is None:
            raise ValueError("Optional H0 setup differs within bucket.")
        h0_index = H0IndexNode(
            atom_to_unique=_pad_vector(
                h0.index.atom_to_unique,
                ntarget,
                fill=0,
            ),
            shells_to_ushell=_pad_indices(
                h0.index.shells_to_ushell,
                template_h0.index.shells_to_ushell,
                nshell,
            ),
            unique_angular=h0.index.unique_angular,
            shells_to_atom=index.shells_to_atom,
            orbitals_to_shell=index.orbitals_to_shell,
        )
        h0 = h0.replace(
            index=h0_index, numbers=_pad_vector(h0.numbers, ntarget)
        )

    classical = []
    for term in core.classical_terms:
        setup = term.setup
        if isinstance(setup, HalogenCoreSetup):
            setup = setup.replace(
                numbers=_pad_vector(setup.numbers, ntarget),
                xbond=_pad_vector(setup.xbond, ntarget),
                atomic_radii=_pad_vector(setup.atomic_radii, ntarget),
                halogen_mask=_pad_vector(setup.halogen_mask, ntarget, False),
                base_mask=_pad_vector(setup.base_mask, ntarget, False),
                valid_atom_mask=_pad_vector(
                    setup.valid_atom_mask, ntarget, False
                ),
                pair_type_mask=_pad_matrix(
                    setup.pair_type_mask, ntarget, False
                ),
                neighbor_type_mask=_pad_matrix(
                    setup.neighbor_type_mask, ntarget, False
                ),
            )
        elif isinstance(setup, DispersionD3CoreSetup):
            setup = setup.replace(
                setup_numbers=_pad_vector(setup.setup_numbers, ntarget),
                component_numbers=_pad_vector(setup.component_numbers, ntarget),
                rcov=_pad_vector(setup.rcov, ntarget),
                r4r2=_pad_vector(setup.r4r2, ntarget),
                rvdw=_pad_matrix(setup.rvdw, ntarget),
            )
        elif isinstance(setup, RepulsionCoreSetup):
            setup = setup.replace(
                arep=_pad_matrix(setup.arep, ntarget),
                zeff=_pad_matrix(setup.zeff, ntarget),
                kexp=_pad_matrix(setup.kexp, ntarget),
                mask=_pad_matrix(setup.mask, ntarget, False),
            )
        else:
            raise TypeError(
                f"Unsupported classical setup {type(setup).__name__}."
            )
        classical.append(term.replace(setup=setup))

    interactions = []
    for term in core.interaction_terms:
        setup = term.setup
        if isinstance(setup, ES2CoreSetup):
            shell_hubbard = setup.shell_hubbard
            shells_to_atom = setup.shells_to_atom
            if shell_hubbard is not None:
                shell_hubbard = _pad_vector(shell_hubbard, nshell_target)
            if shells_to_atom is not None:
                shells_to_atom = _pad_indices(
                    shells_to_atom,
                    template_backend.index.shells_to_atom,
                    nshell,
                )
            setup = setup.replace(
                atom_pair_mask=_pad_matrix(
                    setup.atom_pair_mask, ntarget, False
                ),
                atom_mask=_pad_matrix(setup.atom_mask, ntarget, False),
                atom_hubbard=(
                    None
                    if setup.atom_hubbard is None
                    else _pad_vector(setup.atom_hubbard, ntarget)
                ),
                shell_pair_mask=(
                    None
                    if setup.shell_pair_mask is None
                    else _pad_matrix(
                        setup.shell_pair_mask, nshell_target, False
                    )
                ),
                shell_mask=(
                    None
                    if setup.shell_mask is None
                    else _pad_matrix(setup.shell_mask, nshell_target, False)
                ),
                shell_hubbard=shell_hubbard,
                shells_to_atom=shells_to_atom,
            )
        elif isinstance(setup, ES3CoreSetup):
            setup = setup.replace(
                hubbard_derivs=_pad_vector(setup.hubbard_derivs, ntarget)
            )
        else:
            raise TypeError(
                f"Unsupported interaction setup {type(setup).__name__}."
            )
        interactions.append(term.replace(setup=setup))

    padded = core.replace(
        numbers=_pad_vector(core.numbers, ntarget),
        integral=core.integral.replace(backend=padded_backend),
        h0_setup=h0,
        classical_terms=tuple(classical),
        interaction_terms=tuple(interactions),
    )
    validity = PaddedCoreValidity(
        atoms=torch.arange(ntarget, device=core.numbers.device) < nreal,
        shells=torch.arange(nshell_target, device=core.numbers.device) < nshell,
        orbitals=torch.arange(nao_target, device=core.numbers.device) < nao,
        unique_elements=torch.ones(
            h0.unique.numel() if h0 is not None else 0,
            dtype=torch.bool,
            device=core.numbers.device,
        ),
        unique_shell_types=torch.ones(
            plan.atom.new_tensor(plan.unique_angular).numel(),
            dtype=torch.bool,
            device=core.numbers.device,
        ),
        pair_entries=pair_masks,
    )
    return padded, validity


def pad_exact_core(
    core: ExactSystemCore, bucket: CoreBucket, template: ExactSystemCore
) -> PaddedExactSystemCore:
    """Pad one supported core to a previously computed bucket template."""
    signature = compatibility_signature(core)
    if signature != bucket.signature or signature != compatibility_signature(
        template
    ):
        raise ValueError(
            "Core does not belong to the requested compatibility bucket."
        )
    if _capacities(template) != (
        bucket.atoms,
        bucket.shells,
        bucket.orbitals,
        bucket.unique_elements,
        bucket.unique_shell_types,
        bucket.pair_entries,
    ):
        raise ValueError(
            "Template core does not realize the bucket capacities."
        )
    capacities = _capacities(core)
    if (
        capacities[0] > bucket.atoms
        or capacities[1] > bucket.shells
        or capacities[2] > bucket.orbitals
        or capacities[3] > bucket.unique_elements
        or capacities[4] > bucket.unique_shell_types
        or any(a > b for a, b in zip(capacities[5], bucket.pair_entries))
    ):
        raise ValueError("Core exceeds the requested bucket capacities.")
    if capacities == _capacities(template):
        backend = _backend(core)
        validity = PaddedCoreValidity(
            atoms=torch.ones_like(core.numbers, dtype=torch.bool),
            shells=torch.ones_like(
                backend.index.shells_to_atom, dtype=torch.bool
            ),
            orbitals=torch.ones_like(
                backend.index.orbitals_to_shell, dtype=torch.bool
            ),
            unique_elements=torch.ones(
                0 if core.h0_setup is None else core.h0_setup.unique.numel(),
                dtype=torch.bool,
                device=core.numbers.device,
            ),
            unique_shell_types=torch.ones(
                len(backend.pair_plan.unique_angular),
                dtype=torch.bool,
                device=core.numbers.device,
            ),
            pair_entries=tuple(
                torch.ones_like(pair.ib, dtype=torch.bool)
                for pair in backend.pair_plan.classes
            ),
        )
        return PaddedExactSystemCore(core=core, validity=validity)
    padded, validity = _pad_core(core, template)
    return PaddedExactSystemCore(core=padded, validity=validity)


def pad_positions(positions: Tensor, validity: PaddedCoreValidity) -> Tensor:
    """Append zero coordinate rows for invalid atoms without breaking graphs."""
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("Expected unbatched positions with shape (nat, 3).")
    if int(validity.atoms.sum()) != positions.shape[0]:
        raise ValueError("Position count disagrees with atom validity.")
    missing = validity.atoms.numel() - positions.shape[0]
    if missing == 0:
        return positions
    return torch.cat((positions, positions.new_zeros((missing, 3))))
