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
"""Canonical structural padding for mixed-element exact-core buckets.

This bounded extension constructs a canonical union of element and shell
types for independently captured GFN1/PyTorch exact cores. Its padded values
remain outside ordinary System and SCF evaluation; the accepted numerical
consumer is the validity-aware Repulsion kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten_with_path

from dxtb._src.calculators.model import ExactSystemCore
from dxtb._src.calculators.padding import (
    PaddedCoreValidity,
    PaddedExactSystemCore,
    _backend,
    _setup_static_signature,
    _validate_d3_reference_shapes,
)
from dxtb._src.components.classicals.dispersion.d3 import (
    DispersionD3CoreSetup,
)
from dxtb._src.components.classicals.halogen.hal import HalogenCoreSetup
from dxtb._src.components.classicals.repulsion.base import RepulsionCoreSetup
from dxtb._src.components.interactions.coulomb.secondorder import ES2CoreSetup
from dxtb._src.components.interactions.coulomb.thirdorder import ES3CoreSetup
from dxtb._src.integral.driver.pytorch.impls.kernels import get_kernel
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    PairClassNode,
    PairPlanNode,
)
from dxtb._src.scf.functional.core import CoreIndexData
from dxtb._src.typing import Tensor
from dxtb._src.xtb.h0 import H0IndexNode, H0SetupNode

ShellType = tuple[int, int, int]
PairClassKey = tuple[int, int, int, int]
PairShells = tuple[int, int]


@dataclass(frozen=True)
class CanonicalCoreMap:
    """Source-to-canonical index maps for one exact core."""

    atoms: tuple[int, ...]
    shells: tuple[int, ...]
    orbitals: tuple[int, ...]
    elements: tuple[int, ...]
    shell_types: tuple[int, ...]


@dataclass(frozen=True)
class _SourceLayout:
    """Validated local shell identity and angular structure."""

    element_values: tuple[int, ...]
    shell_type_keys: tuple[ShellType, ...]
    atom_elements: tuple[int, ...]
    shell_atoms: tuple[int, ...]
    shells_to_type: tuple[int, ...]
    orbitals_to_shell: tuple[int, ...]
    alpha_shapes: tuple[tuple[int, ...], ...]
    coeff_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CanonicalCoreBucket:
    """Value-based canonical layout for a mixed-element core group."""

    indices: tuple[int, ...]
    signature: tuple[object, ...]
    elements: tuple[int, ...]
    shell_types: tuple[ShellType, ...]
    shell_type_angular: tuple[int, ...]
    element_counts: tuple[tuple[int, int], ...]
    atoms: int
    shells: int
    orbitals: int
    atom_elements: tuple[int, ...]
    shells_to_atom: tuple[int, ...]
    shells_to_type: tuple[int, ...]
    orbitals_to_shell: tuple[int, ...]
    ao_offsets: tuple[int, ...]
    pair_classes: tuple[PairClassKey, ...]
    pair_shells: tuple[tuple[PairShells, ...], ...]
    source_maps: tuple[tuple[int, CanonicalCoreMap], ...]
    source_pair_shells: tuple[
        tuple[int, tuple[tuple[PairShells, ...], ...]], ...
    ]
    alpha_shapes: tuple[tuple[int, ...], ...]
    coeff_shapes: tuple[tuple[int, ...], ...]


def _tensor_ints(value: Tensor) -> tuple[int, ...]:
    if value.ndim != 1:
        raise ValueError("Expected a one-dimensional structural tensor.")
    return tuple(int(item) for item in value.tolist())


def _source_layout(core: ExactSystemCore) -> _SourceLayout:
    """Read and validate one source's element/shell/basis identities."""
    backend = _backend(core)
    h0 = core.h0_setup
    if h0 is None:
        raise TypeError("Canonical mixed-element padding requires H0 setup.")

    elements = _tensor_ints(h0.unique)
    numbers = _tensor_ints(core.numbers)
    atom_to_unique = _tensor_ints(h0.index.atom_to_unique)
    shell_atoms = _tensor_ints(backend.index.shells_to_atom)
    shells_to_type = _tensor_ints(h0.index.shells_to_ushell)
    orbital_to_shell = _tensor_ints(backend.index.orbitals_to_shell)
    angular = _tensor_ints(h0.index.unique_angular)
    if _tensor_ints(h0.numbers) != numbers:
        raise ValueError(
            "H0 atomic numbers disagree with ExactSystemCore.numbers."
        )
    if len(set(elements)) != len(elements) or set(elements) != set(numbers):
        raise ValueError("H0 unique elements do not match the source atoms.")
    if len(numbers) != len(atom_to_unique) or len(shell_atoms) != len(
        shells_to_type
    ):
        raise ValueError("H0 and backend structural index lengths disagree.")
    if _tensor_ints(h0.index.shells_to_atom) != shell_atoms:
        raise ValueError("H0 and backend shell-to-atom mappings disagree.")
    if _tensor_ints(h0.index.orbitals_to_shell) != orbital_to_shell:
        raise ValueError("H0 and backend orbital-to-shell mappings disagree.")
    if tuple(backend.pair_plan.unique_angular) != angular:
        raise ValueError("PairPlan and H0 shell angular classes disagree.")
    if backend.pair_plan.nao != len(orbital_to_shell):
        raise ValueError(
            "PairPlan AO stride disagrees with the backend AO map."
        )
    if _tensor_ints(backend.pair_plan.atom) != shell_atoms:
        raise ValueError(
            "PairPlan shell-to-atom mapping disagrees with the backend."
        )
    if any(index < 0 or index >= len(elements) for index in atom_to_unique):
        raise ValueError(
            "H0 atom-to-element mapping contains an invalid index."
        )
    if any(index < 0 or index >= len(angular) for index in shells_to_type):
        raise ValueError("H0 shell-to-type mapping contains an invalid index.")
    if any(index < 0 or index >= len(numbers) for index in shell_atoms):
        raise ValueError(
            "Backend shell-to-atom mapping contains an invalid index."
        )
    if any(
        numbers[i] != elements[atom_to_unique[i]] for i in range(len(numbers))
    ):
        raise ValueError("H0 atom-to-element mapping disagrees with numbers.")

    shell_types_by_element: dict[int, list[int]] = {}
    shell_type_element: dict[int, int] = {}
    for atom, z in enumerate(numbers):
        for shell, shell_atom in enumerate(shell_atoms):
            if shell_atom != atom:
                continue
            shell_type = shells_to_type[shell]
            previous = shell_type_element.setdefault(shell_type, z)
            if previous != z:
                raise ValueError(
                    "A local shell type maps to multiple elements."
                )
            values = shell_types_by_element.setdefault(z, [])
            if shell_type not in values:
                values.append(shell_type)

    type_keys: list[ShellType | None] = [None] * len(angular)
    for z, type_ids in shell_types_by_element.items():
        for ordinal, type_id in enumerate(type_ids):
            if type_id >= len(angular):
                raise ValueError("Shell-to-unique-shell index is out of range.")
            type_keys[type_id] = (z, ordinal, angular[type_id])
    if any(value is None for value in type_keys):
        raise ValueError("A registered unique shell type has no real shell.")
    if len(backend.alphas) != len(type_keys) or len(backend.coeffs) != len(
        type_keys
    ):
        raise ValueError("Basis tuples do not match the unique shell types.")

    expected_orbitals: list[int] = []
    for shell, type_id in enumerate(shells_to_type):
        expected_orbitals.extend([shell] * (2 * angular[type_id] + 1))
    if tuple(expected_orbitals) != orbital_to_shell:
        raise ValueError("Orbital-to-shell mapping is not basis-consistent.")
    if len(expected_orbitals) != len(orbital_to_shell):
        raise ValueError("AO count does not match the shell angular classes.")

    # Every atom of one element must have the same ordered shell identities.
    patterns: dict[int, tuple[ShellType, ...]] = {}
    for atom, z in enumerate(numbers):
        pattern = tuple(
            type_keys[shells_to_type[shell]]
            for shell, shell_atom in enumerate(shell_atoms)
            if shell_atom == atom
        )
        if z in patterns and patterns[z] != pattern:
            raise TypeError(
                f"Element {z} has incompatible per-atom shell layouts."
            )
        patterns[z] = pattern  # type: ignore[assignment]

    # Shell ordinal and angular identity must be stable across the bucket.
    return _SourceLayout(
        element_values=elements,
        shell_type_keys=tuple(type_keys),  # type: ignore[arg-type]
        atom_elements=numbers,
        shell_atoms=shell_atoms,
        shells_to_type=shells_to_type,
        orbitals_to_shell=orbital_to_shell,
        alpha_shapes=tuple(tuple(value.shape) for value in backend.alphas),
        coeff_shapes=tuple(tuple(value.shape) for value in backend.coeffs),
    )


def _base_signature(core: ExactSystemCore) -> tuple[object, ...]:
    """Static data that must agree before canonicalizing dimensions."""
    backend = _backend(core)
    get_kernel(backend.algorithm)
    integral = core.integral
    if integral is None:
        raise TypeError("Canonical mixed-element padding needs an integral.")
    h0 = core.h0_setup
    if h0 is None:
        raise TypeError("Canonical mixed-element padding needs H0 setup.")
    if h0.enscale.ndim != 0:
        raise ValueError(
            "Canonical padding requires scalar H0 enscale, got shape "
            f"{tuple(h0.enscale.shape)}."
        )
    if h0.method != "gfn1":
        raise TypeError(
            "Canonical mixed-element padding currently supports GFN1 only."
        )
    optional_h0 = (
        "kq",
        "kqat",
        "h0rad",
        "kdiff",
        "enshell",
        "enscale4",
        "eeq_model",
        "cn_radii",
        "cn_cutoff",
        "cn_max",
        "cn_kcn",
    )
    if any(getattr(h0, name) is not None for name in optional_h0):
        raise TypeError(
            "Canonical GFN1 padding does not support optional H0 data."
        )
    for term in core.classical_terms:
        if isinstance(term.setup, DispersionD3CoreSetup):
            _validate_d3_reference_shapes(term.setup)
            if term.setup.cutoff.ndim != 0:
                raise ValueError("Canonical padding requires scalar D3 cutoff.")
        elif isinstance(term.setup, HalogenCoreSetup):
            if term.setup.damp.ndim != 0 or term.setup.cutoff.ndim != 0:
                raise ValueError(
                    "Canonical padding requires scalar Halogen damp/cutoff."
                )
        elif isinstance(term.setup, RepulsionCoreSetup):
            if term.setup.cutoff.ndim != 0:
                raise ValueError(
                    "Canonical padding requires scalar Repulsion cutoff."
                )
    for term in core.interaction_terms:
        if isinstance(term.setup, ES2CoreSetup) and term.setup.gexp.ndim != 0:
            raise ValueError("Canonical padding requires scalar ES2 gexp.")
    h0_optional = tuple(
        getattr(h0, name) is not None
        for name in (
            "kq",
            "kqat",
            "h0rad",
            "kdiff",
            "enshell",
            "enscale4",
            "eeq_model",
            "cn_radii",
            "cn_cutoff",
            "cn_max",
            "cn_kcn",
        )
    )
    dynamic_tensors = tuple(
        leaf
        for leaf in torch.utils._pytree.tree_flatten(core)[0]
        if isinstance(leaf, torch.Tensor)
    )
    for leaf in dynamic_tensors:
        if leaf.layout is not torch.strided:
            raise TypeError("Canonical padding requires strided tensor leaves.")
        if leaf.device != core.numbers.device:
            raise TypeError(
                "Canonical padding requires one logical tensor device."
            )
        if leaf.is_complex():
            raise TypeError(
                "Canonical padding does not support complex setup leaves."
            )
        if leaf.is_floating_point() and leaf.dtype is not core.effective_dtype:
            raise TypeError(
                "Floating setup leaves must match the ExactSystemCore effective dtype."
            )
        if not leaf.is_floating_point() and leaf.dtype != core.numbers.dtype:
            if leaf.dtype is not torch.bool:
                raise TypeError(
                    "Structural integer setup leaves must match the atomic-number dtype."
                )
    dtypes = tuple(
        sorted(
            {
                (leaf.dtype, leaf.device, leaf.layout)
                for leaf in dynamic_tensors
                if isinstance(leaf, torch.Tensor)
            },
            key=lambda item: (str(item[0]), str(item[1]), str(item[2])),
        )
    )
    classical = tuple(
        (
            term.kind,
            term.label,
            term.position,
            term.geometry_inputs,
            term.charge_dependent,
            type(term.setup),
            _setup_static_signature(term.setup),
            (
                (
                    tuple(term.setup.reference_cn.shape),
                    tuple(term.setup.reference_c6.shape),
                    tuple(
                        (
                            tuple(entry.value.tensor_value.shape)
                            if entry.value.tensor_value is not None
                            else None
                        )
                        for entry in term.setup.parameter_entries
                    ),
                )
                if isinstance(term.setup, DispersionD3CoreSetup)
                else None
            ),
        )
        for term in core.classical_terms
    )
    interactions = tuple(
        (
            term.kind,
            term.label,
            term.position,
            term.geometry_inputs,
            type(term.setup),
            _setup_static_signature(term.setup),
        )
        for term in core.interaction_terms
    )
    retained_shapes = _retained_tensor_shapes(core)
    return (
        core.config,
        core.effective_dtype,
        core.numbers.device,
        integral.driver_type,
        integral.intlevel,
        integral.logical_device,
        type(backend),
        backend.algorithm,
        h0.method,
        h0.cn_kind,
        h0_optional,
        dtypes,
        classical,
        interactions,
        retained_shapes,
    )


def _path_parts(path: tuple[object, ...]) -> tuple[str | int, ...]:
    """Convert registered pytree paths to stable attribute/index values."""
    parts: list[str | int] = []
    for key in path:
        name = getattr(key, "name", None)
        index = getattr(key, "idx", None)
        if isinstance(name, str):
            parts.append(name)
        elif isinstance(index, int):
            parts.append(index)
        else:
            raise TypeError(
                "Canonical padding cannot classify pytree path key "
                f"{type(key).__name__}."
            )
    return tuple(parts)


def _is_canonicalized_path(parts: tuple[str | int, ...]) -> bool:
    """Whether canonical padding explicitly remaps a tensor's shape."""
    if parts in {
        ("numbers",),
        ("integral", "backend", "index", "shells_to_atom"),
        ("integral", "backend", "index", "orbitals_to_shell"),
        ("integral", "backend", "pair_plan", "atom"),
        ("h0_setup", "index", "atom_to_unique"),
        ("h0_setup", "index", "shells_to_ushell"),
        ("h0_setup", "index", "unique_angular"),
        ("h0_setup", "index", "shells_to_atom"),
        ("h0_setup", "index", "orbitals_to_shell"),
        ("h0_setup", "numbers"),
        ("h0_setup", "unique"),
        ("h0_setup", "hscale"),
        ("h0_setup", "kcn"),
        ("h0_setup", "kpair"),
        ("h0_setup", "refocc"),
        ("h0_setup", "selfenergy"),
        ("h0_setup", "shpoly"),
        ("h0_setup", "valence"),
        ("h0_setup", "en"),
        ("h0_setup", "rad"),
    }:
        return True
    if (
        len(parts) == 4
        and parts[:3] == ("integral", "backend", "alphas")
        and isinstance(parts[3], int)
    ) or (
        len(parts) == 4
        and parts[:3] == ("integral", "backend", "coeffs")
        and isinstance(parts[3], int)
    ):
        # Their shape is checked against the source shell type in _source_layout
        # and tuple positions are rebuilt from the canonical shell map.
        return True
    if (
        len(parts) == 6
        and parts[:4] == ("integral", "backend", "pair_plan", "classes")
        and isinstance(parts[4], int)
        and parts[5] in {"ib", "jk", "idx", "weight"}
    ):
        return True
    if len(parts) != 4 or not isinstance(parts[1], int) or parts[2] != "setup":
        return False
    field = parts[3]
    if parts[0] == "classical_terms":
        return field in {
            "numbers",
            "xbond",
            "atomic_radii",
            "halogen_mask",
            "base_mask",
            "valid_atom_mask",
            "pair_type_mask",
            "neighbor_type_mask",
            "setup_numbers",
            "component_numbers",
            "rcov",
            "r4r2",
            "rvdw",
            "arep",
            "zeff",
            "kexp",
            "mask",
        }
    if parts[0] == "interaction_terms":
        return field in {
            "atom_pair_mask",
            "atom_mask",
            "atom_hubbard",
            "shell_pair_mask",
            "shell_mask",
            "shell_hubbard",
            "shells_to_atom",
            "hubbard_derivs",
        }
    return False


def _retained_tensor_shapes(core: ExactSystemCore) -> tuple[object, ...]:
    """Record every dynamic shape not normalized by canonical padding."""
    return tuple(
        (parts, tuple(value.shape))
        for path, value in tree_flatten_with_path(core)[0]
        if isinstance(value, torch.Tensor)
        and not _is_canonicalized_path(parts := _path_parts(path))
    )


def _make_source_map(
    layout: _SourceLayout,
    elements: tuple[int, ...],
    shell_types: tuple[ShellType, ...],
    element_counts: dict[int, int],
    element_offsets: dict[int, int],
    target_shell_for_atom: dict[tuple[int, ShellType], int],
    target_ao_offsets: tuple[int, ...],
) -> CanonicalCoreMap:
    """Build source index maps into the canonical element-grouped layout."""
    atom_seen: dict[int, int] = {}
    atom_map: list[int] = []
    for z in layout.atom_elements:
        ordinal = atom_seen.get(z, 0)
        atom_seen[z] = ordinal + 1
        if ordinal >= element_counts[z]:
            raise ValueError("Canonical atom capacity is smaller than source.")
        atom_map.append(element_offsets[z] + ordinal)

    source_shell_type_map = tuple(
        shell_types.index(key) for key in layout.shell_type_keys
    )
    shell_map: list[int] = []
    for shell, atom in enumerate(layout.shell_atoms):
        z = layout.atom_elements[atom]
        key = layout.shell_type_keys[layout.shells_to_type[shell]]
        if key[0] != z:
            raise ValueError("Shell element identity disagrees with its atom.")
        target_atom = atom_map[atom]
        target = target_shell_for_atom.get((target_atom, key))
        if target is None:
            raise ValueError("Source shell has no canonical shell destination.")
        shell_map.append(target)

    source_ao_offsets: list[int] = []
    offset = 0
    for shell in range(len(layout.shell_atoms)):
        source_ao_offsets.append(offset)
        key = layout.shell_type_keys[layout.shells_to_type[shell]]
        offset += 2 * key[2] + 1
    if offset != len(layout.orbitals_to_shell):
        raise ValueError("Source AO offsets do not cover the orbital map.")
    orbital_map = [-1] * len(layout.orbitals_to_shell)
    for source_shell, target_shell in enumerate(shell_map):
        angular = layout.shell_type_keys[layout.shells_to_type[source_shell]][2]
        width = 2 * angular + 1
        for component in range(width):
            orbital_map[source_ao_offsets[source_shell] + component] = (
                target_ao_offsets[target_shell] + component
            )
    if any(index < 0 for index in orbital_map):
        raise ValueError("A source AO has no canonical destination.")

    element_map = tuple(elements.index(z) for z in layout.element_values)
    return CanonicalCoreMap(
        atoms=tuple(atom_map),
        shells=tuple(shell_map),
        orbitals=tuple(orbital_map),
        elements=element_map,
        shell_types=source_shell_type_map,
    )


def _expected_pair_idx(
    ib: int,
    jk: int,
    angular_i: int,
    angular_j: int,
    ao_offsets: tuple[int, ...],
    nao: int,
    *,
    device: torch.device,
) -> Tensor:
    """Construct a flattened AO block destination for one shell pair."""
    rows = ao_offsets[ib] + torch.arange(
        2 * angular_i + 1, dtype=torch.int64, device=device
    )
    cols = ao_offsets[jk] + torch.arange(
        2 * angular_j + 1, dtype=torch.int64, device=device
    )
    return rows[:, None] * nao + cols[None, :]


def _mapped_pair_class(
    core: ExactSystemCore,
    mapping: CanonicalCoreMap,
    layout: _SourceLayout,
    target_ao_offsets: tuple[int, ...],
    target_nao: int,
) -> tuple[
    dict[PairClassKey, dict[PairShells, float]],
    dict[PairClassKey, set[PairShells]],
]:
    """Map pair entries and prove their canonical AO block destinations."""
    pair_plan = _backend(core).pair_plan
    source_pairs: dict[PairClassKey, set[PairShells]] = {}
    union_entries: dict[PairClassKey, dict[PairShells, float]] = {}
    angular_by_type = tuple(key[2] for key in layout.shell_type_keys)
    for pair_class in pair_plan.classes:
        source_type_i = pair_class.ub
        source_type_j = pair_class.uk
        if source_type_i >= len(layout.shell_type_keys) or source_type_j >= len(
            layout.shell_type_keys
        ):
            raise ValueError("PairPlan shell type is outside its source map.")
        if (
            pair_class.lb != angular_by_type[source_type_i]
            or pair_class.lk != angular_by_type[source_type_j]
        ):
            raise ValueError(
                "PairPlan angular metadata disagrees with shell types."
            )
        for entry in range(pair_class.ib.numel()):
            old_i = int(pair_class.ib[entry])
            old_j = int(pair_class.jk[entry])
            if not (0 <= old_i < len(mapping.shells)) or not (
                0 <= old_j < len(mapping.shells)
            ):
                raise ValueError(
                    "PairPlan shell index is outside the source map."
                )
            if (
                layout.shells_to_type[old_i] != source_type_i
                or layout.shells_to_type[old_j] != source_type_j
            ):
                raise ValueError(
                    "PairPlan entry shell types disagree with context."
                )
            new_i = mapping.shells[old_i]
            new_j = mapping.shells[old_j]
            type_i = mapping.shell_types[source_type_i]
            type_j = mapping.shell_types[source_type_j]
            angular_i = pair_class.lb
            angular_j = pair_class.lk
            flipped = new_i < new_j
            if flipped:
                new_i, new_j = new_j, new_i
                type_i, type_j = type_j, type_i
                angular_i, angular_j = angular_j, angular_i
            key = (type_i, type_j, angular_i, angular_j)
            old_idx = pair_class.idx[entry]
            if bool(torch.any(old_idx < 0)) or bool(
                torch.any(old_idx >= pair_plan.nao * pair_plan.nao)
            ):
                raise ValueError(
                    "PairPlan AO destination is outside its source stride."
                )
            mapped_rows = [
                [
                    mapping.orbitals[int(value) // pair_plan.nao] * target_nao
                    + mapping.orbitals[int(value) % pair_plan.nao]
                    for value in row
                ]
                for row in old_idx.tolist()
            ]
            if flipped:
                mapped_rows = [
                    [
                        (value % target_nao) * target_nao + value // target_nao
                        for value in row
                    ]
                    for row in zip(*mapped_rows)
                ]
            expected = _expected_pair_idx(
                new_i,
                new_j,
                angular_i,
                angular_j,
                target_ao_offsets,
                target_nao,
                device=old_idx.device,
            )
            if mapped_rows != expected.tolist():
                raise ValueError(
                    "Mapped PairPlan AO destinations are invalid for "
                    f"source shells {(old_i, old_j)} mapped to {(new_i, new_j)}: "
                    f"source idx {old_idx.tolist()} with nao={pair_plan.nao}, "
                    f"got {mapped_rows}, expected {expected.tolist()}."
                )
            pair = (new_i, new_j)
            weight = float(pair_class.weight[entry])
            if weight != (0.5 if new_i == new_j else 1.0):
                raise ValueError(
                    "Mapped PairPlan weight disagrees with shell pair."
                )
            values = union_entries.setdefault(key, {})
            if pair in values:
                raise ValueError("Duplicate canonical PairPlan shell pair.")
            values[pair] = weight
            source_pairs.setdefault(key, set()).add(pair)
    expected_count = len(mapping.shells) * (len(mapping.shells) + 1) // 2
    actual_count = sum(len(entries) for entries in union_entries.values())
    if actual_count != expected_count:
        raise ValueError(
            "PairPlan does not contain every lower-triangular source shell pair."
        )
    return union_entries, source_pairs


def bucket_canonical_exact_cores(
    cores: tuple[ExactSystemCore, ...],
) -> tuple[CanonicalCoreBucket, ...]:
    """Build one deterministic canonical GFN1/PyTorch element-union bucket."""
    if not cores:
        return ()
    signatures = tuple(_base_signature(core) for core in cores)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise TypeError(
            "Canonical cores require matching static configuration."
        )
    layouts = tuple(_source_layout(core) for core in cores)

    elements = tuple(
        sorted({z for layout in layouts for z in layout.element_values})
    )

    per_element_angular: dict[tuple[int, int], int] = {}
    per_element_shell_count: dict[int, int] = {}
    alpha_shape_by_type: dict[ShellType, tuple[int, ...]] = {}
    coeff_shape_by_type: dict[ShellType, tuple[int, ...]] = {}
    element_counts: dict[int, int] = {z: 0 for z in elements}
    for layout in layouts:
        for z in layout.atom_elements:
            element_counts[z] = max(
                element_counts[z], layout.atom_elements.count(z)
            )
        types_by_element: dict[int, list[ShellType]] = {}
        for type_id, key in enumerate(layout.shell_type_keys):
            types_by_element.setdefault(key[0], []).append(key)
            previous = per_element_angular.setdefault((key[0], key[1]), key[2])
            if previous != key[2]:
                raise TypeError(
                    f"Element {key[0]} shell ordinal {key[1]} has different angular classes."
                )
            for shape, shape_by_type, name in (
                (layout.alpha_shapes[type_id], alpha_shape_by_type, "alpha"),
                (
                    layout.coeff_shapes[type_id],
                    coeff_shape_by_type,
                    "coefficient",
                ),
            ):
                old = shape_by_type.setdefault(key, shape)
                if old != shape:
                    raise TypeError(
                        f"Canonical shell type {key} has incompatible {name} shape."
                    )
        for z, type_keys in types_by_element.items():
            count = len(type_keys)
            previous_count = per_element_shell_count.setdefault(z, count)
            if previous_count != count:
                raise TypeError(f"Element {z} has incompatible shell counts.")

    shell_types = tuple(
        sorted(
            {key for layout in layouts for key in layout.shell_type_keys},
            key=lambda key: (key[0], key[1], key[2]),
        )
    )
    if set(per_element_shell_count) != set(elements):
        raise ValueError("An element has no canonical shell-type definition.")
    element_offsets: dict[int, int] = {}
    atom_elements: list[int] = []
    offset = 0
    for z in elements:
        element_offsets[z] = offset
        atom_elements.extend([z] * element_counts[z])
        offset += element_counts[z]

    target_shell_for_atom: dict[tuple[int, ShellType], int] = {}
    shells_to_atom: list[int] = []
    shells_to_type: list[int] = []
    angular_by_type = tuple(key[2] for key in shell_types)
    for atom, z in enumerate(atom_elements):
        for key in (key for key in shell_types if key[0] == z):
            target_shell_for_atom[(atom, key)] = len(shells_to_atom)
            shells_to_atom.append(atom)
            shells_to_type.append(shell_types.index(key))

    ao_offsets: list[int] = []
    orbitals_to_shell: list[int] = []
    ao_offset = 0
    for shell, type_id in enumerate(shells_to_type):
        ao_offsets.append(ao_offset)
        width = 2 * angular_by_type[type_id] + 1
        orbitals_to_shell.extend([shell] * width)
        ao_offset += width
    ao_offsets.append(ao_offset)

    source_maps: list[tuple[int, CanonicalCoreMap]] = []
    for index, (core, layout) in enumerate(zip(cores, layouts)):
        mapping = _make_source_map(
            layout,
            elements,
            shell_types,
            element_counts,
            element_offsets,
            target_shell_for_atom,
            tuple(ao_offsets),
        )
        source_maps.append((index, mapping))

    pair_union: dict[PairClassKey, dict[PairShells, float]] = {}
    mapped_pairs: list[dict[PairClassKey, set[PairShells]]] = []
    for core, layout, (_, mapping) in zip(cores, layouts, source_maps):
        entries, source_pairs = _mapped_pair_class(
            core,
            mapping,
            layout,
            tuple(ao_offsets),
            len(orbitals_to_shell),
        )
        mapped_pairs.append(source_pairs)
        for key, values in entries.items():
            union = pair_union.setdefault(key, {})
            for pair, weight in values.items():
                if pair in union and union[pair] != weight:
                    raise ValueError(
                        "Canonical pair structure is inconsistent."
                    )
                union[pair] = weight

    pair_classes = tuple(sorted(pair_union))
    pair_shells = tuple(tuple(sorted(pair_union[key])) for key in pair_classes)
    source_pair_shells = tuple(
        (
            index,
            tuple(
                tuple(sorted(source_pairs.get(key, set())))
                for key in pair_classes
            ),
        )
        for index, source_pairs in enumerate(mapped_pairs)
    )
    signature = (
        signatures[0],
        elements,
        shell_types,
        tuple(sorted(element_counts.items())),
        pair_classes,
        pair_shells,
    )
    return CanonicalCoreBucket(
        indices=tuple(range(len(cores))),
        signature=signature,
        elements=elements,
        shell_types=shell_types,
        shell_type_angular=angular_by_type,
        element_counts=tuple(sorted(element_counts.items())),
        atoms=len(atom_elements),
        shells=len(shells_to_atom),
        orbitals=len(orbitals_to_shell),
        atom_elements=tuple(atom_elements),
        shells_to_atom=tuple(shells_to_atom),
        shells_to_type=tuple(shells_to_type),
        orbitals_to_shell=tuple(orbitals_to_shell),
        ao_offsets=tuple(ao_offsets),
        pair_classes=pair_classes,
        pair_shells=pair_shells,
        source_maps=tuple(source_maps),
        source_pair_shells=source_pair_shells,
        alpha_shapes=tuple(alpha_shape_by_type[key] for key in shell_types),
        coeff_shapes=tuple(coeff_shape_by_type[key] for key in shell_types),
    )


def _scatter_vector(
    value: Tensor,
    mapping: tuple[int, ...],
    size: int,
    *,
    fill: int | float | bool = 0,
) -> Tensor:
    """Place source entries at canonical positions without detaching them."""
    if value.ndim != 1 or value.shape[0] != len(mapping):
        raise ValueError(
            "Source vector does not match its canonical index map."
        )
    indices = torch.tensor(mapping, dtype=torch.int64, device=value.device)
    output = torch.full((size,), fill, dtype=value.dtype, device=value.device)
    return output.index_copy(0, indices, value)


def _scatter_square(
    value: Tensor,
    mapping: tuple[int, ...],
    size: int,
    *,
    fill: int | float | bool = 0,
) -> Tensor:
    """Place both source axes at canonical positions, preserving autograd."""
    if value.ndim != 2 or value.shape != (len(mapping), len(mapping)):
        raise ValueError(
            "Source matrix does not match its canonical index map."
        )
    indices = torch.tensor(mapping, dtype=torch.int64, device=value.device)
    rows = value.new_full((size, value.shape[1]), fill).index_copy(
        0, indices, value
    )
    output = value.new_full((size, size), fill)
    return output.index_copy(1, indices, rows)


def _remap_h0(
    h0: H0SetupNode,
    core: ExactSystemCore,
    bucket: CanonicalCoreBucket,
    mapping: CanonicalCoreMap,
) -> H0SetupNode:
    """Remap the supported GFN1 H0 element and shell tables explicitly."""
    optional = (
        "kq",
        "kqat",
        "h0rad",
        "kdiff",
        "enshell",
        "enscale4",
        "eeq_model",
        "cn_radii",
        "cn_cutoff",
        "cn_max",
        "cn_kcn",
    )
    if any(getattr(h0, name) is not None for name in optional):
        raise TypeError(
            "Canonical GFN1 padding does not support optional H0 data."
        )

    n_elements = len(bucket.elements)
    n_shell_types = len(bucket.shell_types)
    shell_types = mapping.shell_types
    elements = mapping.elements
    element_tensor = torch.tensor(
        bucket.elements, dtype=h0.unique.dtype, device=h0.unique.device
    )
    angular = torch.tensor(
        bucket.shell_type_angular,
        dtype=h0.index.unique_angular.dtype,
        device=h0.index.unique_angular.device,
    )
    atom_to_unique = torch.tensor(
        [bucket.elements.index(z) for z in bucket.atom_elements],
        dtype=h0.index.atom_to_unique.dtype,
        device=h0.index.atom_to_unique.device,
    )
    shells_to_type = torch.tensor(
        bucket.shells_to_type,
        dtype=h0.index.shells_to_ushell.dtype,
        device=h0.index.shells_to_ushell.device,
    )
    h0_index = H0IndexNode(
        atom_to_unique=atom_to_unique,
        shells_to_ushell=shells_to_type,
        unique_angular=angular,
        shells_to_atom=torch.tensor(
            bucket.shells_to_atom,
            dtype=h0.index.shells_to_atom.dtype,
            device=h0.index.shells_to_atom.device,
        ),
        orbitals_to_shell=torch.tensor(
            bucket.orbitals_to_shell,
            dtype=h0.index.orbitals_to_shell.dtype,
            device=h0.index.orbitals_to_shell.device,
        ),
    )
    shell_vectors = ("kcn", "refocc", "selfenergy", "shpoly", "valence")
    element_vectors = ("en", "rad")
    updates: dict[str, object] = {
        "index": h0_index,
        "numbers": _scatter_vector(core.numbers, mapping.atoms, bucket.atoms),
        "unique": element_tensor,
        "hscale": _scatter_square(h0.hscale, shell_types, n_shell_types),
        "kpair": _scatter_square(h0.kpair, elements, n_elements),
    }
    for name in shell_vectors:
        updates[name] = _scatter_vector(
            getattr(h0, name), shell_types, n_shell_types
        )
    for name in element_vectors:
        updates[name] = _scatter_vector(getattr(h0, name), elements, n_elements)
    updates["enscale"] = h0.enscale
    return h0.replace(**updates)


def _remap_family_setup(
    setup: object,
    bucket: CanonicalCoreBucket,
    mapping: CanonicalCoreMap,
) -> object:
    """Apply the explicit atom/shell map for each supported setup family."""
    n_atoms = bucket.atoms
    n_shells = bucket.shells
    if isinstance(setup, HalogenCoreSetup):
        values = {
            name: _scatter_vector(getattr(setup, name), mapping.atoms, n_atoms)
            for name in (
                "numbers",
                "xbond",
                "atomic_radii",
                "halogen_mask",
                "base_mask",
                "valid_atom_mask",
            )
        }
        for name in ("pair_type_mask", "neighbor_type_mask"):
            values[name] = _scatter_square(
                getattr(setup, name), mapping.atoms, n_atoms
            )
        return setup.replace(**values)
    if isinstance(setup, DispersionD3CoreSetup):
        return setup.replace(
            setup_numbers=_scatter_vector(
                setup.setup_numbers, mapping.atoms, n_atoms
            ),
            component_numbers=_scatter_vector(
                setup.component_numbers, mapping.atoms, n_atoms
            ),
            rcov=_scatter_vector(setup.rcov, mapping.atoms, n_atoms),
            r4r2=_scatter_vector(setup.r4r2, mapping.atoms, n_atoms),
            rvdw=_scatter_square(setup.rvdw, mapping.atoms, n_atoms),
        )
    if isinstance(setup, RepulsionCoreSetup):
        return setup.replace(
            arep=_scatter_square(setup.arep, mapping.atoms, n_atoms),
            zeff=_scatter_square(setup.zeff, mapping.atoms, n_atoms),
            kexp=_scatter_square(setup.kexp, mapping.atoms, n_atoms),
            mask=_scatter_square(
                setup.mask, mapping.atoms, n_atoms, fill=False
            ),
        )
    if isinstance(setup, ES2CoreSetup):
        atom_count = len(mapping.atoms)
        shell_count = len(mapping.shells)
        if setup.atom_pair_mask.shape != (
            atom_count,
            atom_count,
        ) or setup.atom_mask.shape != (
            atom_count,
            atom_count,
        ):
            raise ValueError("ES2 atom masks do not match source atom count.")
        if setup.shell_resolved:
            if (
                setup.atom_hubbard is not None
                or setup.shell_pair_mask is None
                or setup.shell_mask is None
                or setup.shell_hubbard is None
                or setup.shells_to_atom is None
            ):
                raise ValueError(
                    "Shell-resolved ES2 has inconsistent optional children."
                )
            if (
                setup.shell_pair_mask.shape != (shell_count, shell_count)
                or setup.shell_mask.shape != (shell_count, shell_count)
                or setup.shell_hubbard.shape != (shell_count,)
                or setup.shells_to_atom.shape != (shell_count,)
            ):
                raise ValueError(
                    "Shell-resolved ES2 children do not match source shells."
                )
            inverse_atoms = {
                target: source for source, target in enumerate(mapping.atoms)
            }
            expected_source_map = tuple(
                inverse_atoms[bucket.shells_to_atom[target_shell]]
                for target_shell in mapping.shells
            )
            actual_source_map = tuple(
                int(value) for value in setup.shells_to_atom.tolist()
            )
            if actual_source_map != expected_source_map:
                raise ValueError(
                    "ES2 shell-to-atom mapping disagrees with source structure."
                )
        elif (
            setup.atom_hubbard is None
            or setup.atom_hubbard.shape != (atom_count,)
            or setup.shell_pair_mask is not None
            or setup.shell_mask is not None
            or setup.shell_hubbard is not None
            or setup.shells_to_atom is not None
        ):
            raise ValueError(
                "Atom-resolved ES2 has inconsistent optional children."
            )
        shell_hubbard = (
            None
            if setup.shell_hubbard is None
            else _scatter_vector(setup.shell_hubbard, mapping.shells, n_shells)
        )
        atom_hubbard = (
            None
            if setup.atom_hubbard is None
            else _scatter_vector(setup.atom_hubbard, mapping.atoms, n_atoms)
        )
        shells_to_atom = (
            None
            if setup.shells_to_atom is None
            else torch.tensor(
                bucket.shells_to_atom,
                dtype=setup.shells_to_atom.dtype,
                device=setup.shells_to_atom.device,
            )
        )
        return setup.replace(
            atom_pair_mask=_scatter_square(
                setup.atom_pair_mask, mapping.atoms, n_atoms, fill=False
            ),
            atom_mask=_scatter_square(
                setup.atom_mask, mapping.atoms, n_atoms, fill=False
            ),
            atom_hubbard=atom_hubbard,
            shell_pair_mask=(
                None
                if setup.shell_pair_mask is None
                else _scatter_square(
                    setup.shell_pair_mask, mapping.shells, n_shells, fill=False
                )
            ),
            shell_mask=(
                None
                if setup.shell_mask is None
                else _scatter_square(
                    setup.shell_mask, mapping.shells, n_shells, fill=False
                )
            ),
            shell_hubbard=shell_hubbard,
            shells_to_atom=shells_to_atom,
        )
    if isinstance(setup, ES3CoreSetup):
        if setup.shell_resolved:
            return setup.replace(
                hubbard_derivs=_scatter_vector(
                    setup.hubbard_derivs, mapping.shells, n_shells
                )
            )
        return setup.replace(
            hubbard_derivs=_scatter_vector(
                setup.hubbard_derivs, mapping.atoms, n_atoms
            )
        )
    raise TypeError(
        f"Canonical padding does not support {type(setup).__name__}."
    )


def pad_canonical_exact_core(
    core: ExactSystemCore,
    bucket: CanonicalCoreBucket,
    input_index: int,
) -> PaddedExactSystemCore:
    """Map one complete exact core to the bucket's canonical structural tree."""
    if _base_signature(core) != bucket.signature[0]:
        raise ValueError("Core does not match the canonical bucket metadata.")
    try:
        mapping = dict(bucket.source_maps)[input_index]
        pair_source = dict(bucket.source_pair_shells)[input_index]
    except KeyError as exc:
        raise ValueError("Input index is not a member of this bucket.") from exc
    layout = _source_layout(core)
    if len(mapping.atoms) != core.numbers.numel():
        raise ValueError("Canonical source map does not match the core.")
    if tuple(bucket.atom_elements[index] for index in mapping.atoms) != (
        layout.atom_elements
    ):
        raise ValueError(
            "Canonical atom map does not describe this source core."
        )
    if tuple(bucket.elements[index] for index in mapping.elements) != (
        layout.element_values
    ):
        raise ValueError(
            "Canonical element map does not describe this source core."
        )
    if tuple(bucket.shell_types[index] for index in mapping.shell_types) != (
        layout.shell_type_keys
    ):
        raise ValueError(
            "Canonical shell-type map does not describe this source core."
        )
    if tuple(bucket.shells_to_atom[index] for index in mapping.shells) != tuple(
        mapping.atoms[atom] for atom in layout.shell_atoms
    ):
        raise ValueError(
            "Canonical shell map does not describe this source core."
        )
    if tuple(bucket.shells_to_type[index] for index in mapping.shells) != tuple(
        mapping.shell_types[type_index] for type_index in layout.shells_to_type
    ):
        raise ValueError(
            "Canonical shell-type map does not describe this source core."
        )
    if tuple(
        bucket.orbitals_to_shell[index] for index in mapping.orbitals
    ) != tuple(mapping.shells[shell] for shell in layout.orbitals_to_shell):
        raise ValueError("Canonical AO map does not describe this source core.")

    backend = _backend(core)
    current_pair_entries, current_source_pairs = _mapped_pair_class(
        core,
        mapping,
        layout,
        bucket.ao_offsets,
        bucket.orbitals,
    )
    expected_source_pairs = {
        key: set(pairs)
        for key, pairs in zip(bucket.pair_classes, pair_source)
        if pairs
    }
    if current_source_pairs != expected_source_pairs:
        raise ValueError(
            "Current PairPlan shell-pair structure does not match the "
            "canonical bucket source mapping."
        )
    if set(current_pair_entries) - set(bucket.pair_classes):
        raise ValueError("Current PairPlan has an unrecorded canonical class.")
    expected_classes = {
        key: set(pairs)
        for key, pairs in zip(bucket.pair_classes, bucket.pair_shells)
    }
    if any(
        not pairs.issubset(expected_classes[key])
        for key, pairs in current_source_pairs.items()
    ):
        raise ValueError("Current PairPlan has an unrecorded canonical pair.")

    source_type_to_target = mapping.shell_types
    alpha_by_target: list[Tensor | None] = [None] * len(bucket.shell_types)
    coeff_by_target: list[Tensor | None] = [None] * len(bucket.shell_types)
    for source_type, target_type in enumerate(source_type_to_target):
        alpha_by_target[target_type] = backend.alphas[source_type]
        coeff_by_target[target_type] = backend.coeffs[source_type]
    alphas = tuple(
        (
            value
            if value is not None
            else torch.ones(
                bucket.alpha_shapes[i],
                dtype=core.effective_dtype,
                device=core.numbers.device,
            )
        )
        for i, value in enumerate(alpha_by_target)
    )
    coeffs = tuple(
        (
            value
            if value is not None
            else torch.ones(
                bucket.coeff_shapes[i],
                dtype=core.effective_dtype,
                device=core.numbers.device,
            )
        )
        for i, value in enumerate(coeff_by_target)
    )

    pair_masks: list[Tensor] = []
    pair_nodes: list[PairClassNode] = []
    for class_index, (key, pairs) in enumerate(
        zip(bucket.pair_classes, bucket.pair_shells)
    ):
        angular_i, angular_j = key[2], key[3]
        idx = torch.stack(
            [
                _expected_pair_idx(
                    ib,
                    jk,
                    angular_i,
                    angular_j,
                    bucket.ao_offsets,
                    bucket.orbitals,
                    device=core.numbers.device,
                )
                for ib, jk in pairs
            ]
        )
        pair_tensor = torch.tensor(
            pairs, dtype=torch.int64, device=core.numbers.device
        )
        weights = torch.where(
            pair_tensor[:, 0] == pair_tensor[:, 1],
            pair_tensor.new_full(
                (len(pairs),), 0.5, dtype=core.effective_dtype
            ),
            pair_tensor.new_ones((len(pairs),), dtype=core.effective_dtype),
        )
        pair_nodes.append(
            PairClassNode(
                ub=key[0],
                uk=key[1],
                lb=key[2],
                lk=key[3],
                ib=pair_tensor[:, 0],
                jk=pair_tensor[:, 1],
                idx=idx,
                weight=weights,
            )
        )
        source_pairs = set(pair_source[class_index])
        pair_masks.append(
            torch.tensor(
                [pair in source_pairs for pair in pairs],
                dtype=torch.bool,
                device=core.numbers.device,
            )
        )

    index = CoreIndexData(
        shells_to_atom=torch.tensor(
            bucket.shells_to_atom,
            dtype=backend.index.shells_to_atom.dtype,
            device=backend.index.shells_to_atom.device,
        ),
        orbitals_to_shell=torch.tensor(
            bucket.orbitals_to_shell,
            dtype=backend.index.orbitals_to_shell.dtype,
            device=backend.index.orbitals_to_shell.device,
        ),
    )
    pair_plan = PairPlanNode(
        nao=bucket.orbitals,
        unique_angular=bucket.shell_type_angular,
        atom=index.shells_to_atom,
        classes=tuple(pair_nodes),
    )
    padded_backend = backend.replace(
        index=index,
        pair_plan=pair_plan,
        alphas=alphas,
        coeffs=coeffs,
    )
    padded_h0 = _remap_h0(core.h0_setup, core, bucket, mapping)
    classical = tuple(
        term.replace(setup=_remap_family_setup(term.setup, bucket, mapping))
        for term in core.classical_terms
    )
    interactions = tuple(
        term.replace(setup=_remap_family_setup(term.setup, bucket, mapping))
        for term in core.interaction_terms
    )
    padded_core = core.replace(
        numbers=_scatter_vector(core.numbers, mapping.atoms, bucket.atoms),
        integral=core.integral.replace(backend=padded_backend),
        h0_setup=padded_h0,
        classical_terms=classical,
        interaction_terms=interactions,
    )
    validity = PaddedCoreValidity(
        atoms=torch.zeros(
            bucket.atoms, dtype=torch.bool, device=core.numbers.device
        ).index_fill(
            0,
            torch.tensor(
                mapping.atoms, dtype=torch.int64, device=core.numbers.device
            ),
            True,
        ),
        shells=torch.zeros(
            bucket.shells, dtype=torch.bool, device=core.numbers.device
        ).index_fill(
            0,
            torch.tensor(
                mapping.shells, dtype=torch.int64, device=core.numbers.device
            ),
            True,
        ),
        orbitals=torch.zeros(
            bucket.orbitals, dtype=torch.bool, device=core.numbers.device
        ).index_fill(
            0,
            torch.tensor(
                mapping.orbitals, dtype=torch.int64, device=core.numbers.device
            ),
            True,
        ),
        unique_elements=torch.zeros(
            len(bucket.elements), dtype=torch.bool, device=core.numbers.device
        ).index_fill(
            0,
            torch.tensor(
                mapping.elements, dtype=torch.int64, device=core.numbers.device
            ),
            True,
        ),
        unique_shell_types=torch.zeros(
            len(bucket.shell_types),
            dtype=torch.bool,
            device=core.numbers.device,
        ).index_fill(
            0,
            torch.tensor(
                mapping.shell_types,
                dtype=torch.int64,
                device=core.numbers.device,
            ),
            True,
        ),
        pair_entries=tuple(pair_masks),
    )
    return PaddedExactSystemCore(core=padded_core, validity=validity)


def canonicalize_positions(
    positions: Tensor,
    bucket: CanonicalCoreBucket,
    input_index: int,
) -> Tensor:
    """Place source coordinates in canonical atom slots with zero padding."""
    try:
        mapping = dict(bucket.source_maps)[input_index]
    except KeyError as exc:
        raise ValueError("Input index is not a member of this bucket.") from exc
    if positions.ndim != 2 or positions.shape != (len(mapping.atoms), 3):
        raise ValueError("Positions do not match the source atom mapping.")
    indices = torch.tensor(
        mapping.atoms, dtype=torch.int64, device=positions.device
    )
    return positions.new_zeros((bucket.atoms, 3)).index_copy(
        0, indices, positions
    )
