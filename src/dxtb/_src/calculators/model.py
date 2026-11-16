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
Calculators: Model and System
=============================

The calculator is split into a molecule-independent :class:`Model` and a
:class:`System`, which :meth:`Model.setup` builds for given atomic numbers.

- The :class:`Model` holds the parameters, the configuration and additional
  components. It is independent of any molecule.
- The :class:`System` holds everything that depends on the atomic numbers
  only: the index helper, the components with their parameters gathered from
  the element tables, the data of the classical contributions and immutable
  integral/H0 setup values. It does not depend on positions, charge or spin.

:meth:`Model.setup` is a function of the parameters and the numbers.
Gradients flow from everything computed in the System to the parameter
leaves, i.e., it can be called inside a differentiated function.

.. note::

    Component lists and classical setup remain transitional mutable objects
    until later component/cache packages. Integral matrix production uses
    explicit setup values and does not store per-call outputs on System.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import wraps
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any, Callable, Union
from weakref import WeakKeyDictionary

import torch
from tad_mctc.tree import ModuleNode, Node, child, context
from torch import nn

try:
    from torch.utils._python_dispatch import _get_current_dispatch_mode
except ImportError:  # pragma: no cover
    _get_current_dispatch_mode = None  # type: ignore[assignment]

from dxtb import IndexHelper, OutputHandler, labels
from dxtb._src.components.classicals import (
    IES,
    Classical,
    ClassicalList,
    DispersionD3,
    DispersionD3Cache,
    DispersionD3CoreSetup,
    DispersionD4,
    DispersionD4CoreSetup,
    DispersionD4Setup,
    Halogen,
    HalogenCoreSetup,
    HalogenSetup,
    IESCoreSetup,
    IESSetup,
    Repulsion,
    RepulsionCoreSetup,
    RepulsionSetup,
    ShortRangeBond,
    ShortRangeBondCoreSetup,
    ShortRangeBondSetup,
    core_dispersion_d3_setup,
    core_dispersion_d4_setup,
    core_halogen_setup,
    core_ies_setup,
    core_repulsion_setup,
    core_srb_setup,
    new_dispersion,
    new_halogen,
    new_ies,
    new_repulsion,
    new_srb,
)
from dxtb._src.components.classicals.list import ClassicalListCache
from dxtb._src.components.custom import (
    ComponentMutationSnapshot,
    CustomClassicalCoreSetup,
    build_custom_classical_core_setup,
    can_build_custom_classical_core_setup,
    snapshot_component_state,
)
from dxtb._src.components.interactions import Interaction, InteractionList
from dxtb._src.components.interactions.coulomb import (
    AES2,
    ES2,
    ES3,
    AES2CoreSetup,
    ES2CoreSetup,
    ES3CoreSetup,
    core_aes2_setup,
    core_es2_setup,
    core_es3_setup,
    new_aes2,
    new_es2,
    new_es3,
    setup_aes2,
    setup_es2,
    setup_es3,
)
from dxtb._src.components.interactions.dispersion import (
    D4SCCoreSetup,
    DispersionD4SC,
    core_d4sc_setup,
    new_d4sc,
    setup_d4sc,
)
from dxtb._src.components.interactions.field import efield
from dxtb._src.components.interactions.field import efieldgrad as efield_grad
from dxtb._src.components.interactions.solvation.alpb import (
    GeneralizedBorn,
    GeneralizedBornCoreSetup,
    core_generalized_born_setup,
    setup_generalized_born,
)
from dxtb._src.constants import defaults
from dxtb._src.integral.driver.libcint.driver import (
    LibcintIntegralSetup,
    ShellLayout,
    _build_shell_layout,
)
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    prepare as prepare_pair_plan,
)
from dxtb._src.integral.driver.pytorch.setup import PytorchIntegralSetup
from dxtb._src.integral.evaluation import (
    IntegralSetup,
    setup_integral_evaluation,
)
from dxtb._src.param import ParamModule
from dxtb._src.timing import timer
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils.tensors import disable_functorch, normalize_device
from dxtb._src.xtb.h0 import H0Setup, H0SetupNode, setup_h0, setup_h0_core
from dxtb.config import Config

if TYPE_CHECKING:
    from dxtb._src.calculators.result import Result

__all__ = ["Model", "ModelParameterNode", "System"]


# ``functional_call`` temporarily installs state on the original ParamModule.
# Keep one reentrant setup lock per owner so distinct Model façades that share
# that module cannot observe another call's substituted values.  A weak-key
# registry avoids making the lock part of either the parameter tree or System.
_PARAM_SETUP_LOCKS: WeakKeyDictionary[ParamModule, RLock] = WeakKeyDictionary()
_PARAM_SETUP_LOCKS_GUARD = Lock()


BUILTIN_CLASSICAL_TYPES = (
    DispersionD3,
    DispersionD4,
    Halogen,
    IES,
    Repulsion,
    ShortRangeBond,
)
"""Exact built-in Classical Component types."""

BUILTIN_INTERACTION_TYPES = (AES2, ES2, ES3, DispersionD4SC, GeneralizedBorn)
"""Exact built-in SCF Interaction types (External fields are call inputs)."""


def _require_component_instances(
    components: Sequence[Any], base_cls: type, kind: str
) -> None:
    """Reject entries that do not inherit from ``base_cls``."""
    invalid = [c for c in components if not isinstance(c, base_cls)]
    if invalid:
        names = ", ".join(
            f"{type(c).__module__}.{type(c).__qualname__}" for c in invalid
        )
        raise TypeError(
            f"All {kind} Components must inherit from {base_cls.__name__}; "
            f"got {names}."
        )


def _custom_interaction_required_int_level(
    interaction: Interaction,
) -> int:
    """Infer multipole integral level required by overridden custom hooks."""
    cls = type(interaction)
    if (
        cls.get_quadrupole_atom_potential
        is not Interaction.get_quadrupole_atom_potential
        or cls.get_quadrupole_atom_energy
        is not Interaction.get_quadrupole_atom_energy
    ):
        return labels.INTLEVEL_QUADRUPOLE
    if (
        cls.get_dipole_atom_potential
        is not Interaction.get_dipole_atom_potential
        or cls.get_dipole_atom_energy is not Interaction.get_dipole_atom_energy
    ):
        return labels.INTLEVEL_DIPOLE
    return 0


def _parameter_setup_lock(par: ParamModule) -> RLock:
    with _PARAM_SETUP_LOCKS_GUARD:
        lock = _PARAM_SETUP_LOCKS.get(par)
        if lock is None:
            lock = RLock()
            _PARAM_SETUP_LOCKS[par] = lock
        return lock


def _serialize_parameter_setup(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    """Serialize setup reads against functional parameter substitution."""

    @wraps(function)
    def wrapped(self: Model, *args: Any, **kwargs: Any) -> Any:
        with _parameter_setup_lock(self.par):
            return function(self, *args, **kwargs)

    return wrapped


CoreSetup = Union[
    RepulsionCoreSetup,
    ShortRangeBondCoreSetup,
    DispersionD3CoreSetup,
    DispersionD4CoreSetup,
    IESCoreSetup,
    HalogenCoreSetup,
    CustomClassicalCoreSetup,
    ES2CoreSetup,
    ES3CoreSetup,
    AES2CoreSetup,
    D4SCCoreSetup,
    GeneralizedBornCoreSetup,
]


class ClassicalTermDescriptor(Node):
    """Ordered exact classical term without a live component reference."""

    setup: CoreSetup = child()
    kind: str = context()
    label: str = context()
    position: int = context()
    charge_dependent: bool = context()
    geometry_inputs: tuple[str, ...] = context()


class InteractionTermDescriptor(Node):
    """Ordered exact interaction term without a live component reference."""

    setup: CoreSetup = child()
    kind: str = context()
    label: str = context()
    position: int = context()
    geometry_inputs: tuple[str, ...] = context()


class ExactSystemCore(Node):
    """Registered exact single-System setup, separate from the component setups.

    The canonical :class:`~dxtb.IndexHelper`, selected integral backend, and
    all family setup Nodes are dynamic children.
    """

    numbers: Tensor = child(keep_dtype=True)
    index: IndexHelper = child()
    effective_dtype: torch.dtype = context()
    config: Config = context()
    integral: IntegralSetup | None = child()
    h0_setup: H0SetupNode | None = child()
    classical_terms: tuple[ClassicalTermDescriptor, ...] = child()
    interaction_terms: tuple[InteractionTermDescriptor, ...] = child()

    @property
    def integral_setup(self) -> IntegralSetup | None:
        """Return the selected integral setup Node."""
        return self.integral

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ExactSystemCore:
        """Convert dynamic setup leaves and update the effective dtype."""
        result = super().to(device=device, dtype=dtype)
        if dtype is not None and result.effective_dtype is not dtype:
            result = result.replace(effective_dtype=dtype)
        if (
            result.integral is not None
            and result.integral.logical_device != result.numbers.device
        ):
            result = result.replace(
                integral=result.integral.replace(
                    logical_device=result.numbers.device
                )
            )
        return result

    def classical_term_at(
        self, position: int
    ) -> ClassicalTermDescriptor | None:
        """Return an ordered descriptor without consulting components."""
        return next(
            (
                entry
                for entry in self.classical_terms
                if entry.position == position
            ),
            None,
        )

    def interaction_term_at(
        self, position: int
    ) -> InteractionTermDescriptor | None:
        """Return an ordered descriptor without consulting components."""
        return next(
            (
                entry
                for entry in self.interaction_terms
                if entry.position == position
            ),
            None,
        )


class ModelParameterNode(Node):
    """Registered Model-level parameter and buffer tree."""

    parameters: ModuleNode = child()
    config: Config = context()
    auto_int_level: bool = context()


class _ModelSetupAdapter(nn.Module):
    """Temporarily apply functional ParamModule values during Model.setup."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.par = model.par
        # Keep the façade outside the registered module tree.
        self.__dict__["_model"] = model

    def forward(
        self,
        numbers: Tensor,
        *,
        batch_mode: int | None = None,
        dd: DD | None = None,
    ) -> System:
        return self._model.setup(numbers, batch_mode=batch_mode, dd=dd)


def _classical_term_kind(
    component: Classical, setup: CoreSetup | None
) -> tuple[str, bool] | None:
    """Describe built-in and tensor-only custom Classical core setups."""
    if type(component) not in BUILTIN_CLASSICAL_TYPES and isinstance(
        setup, CustomClassicalCoreSetup
    ):
        return ("custom", True)
    kinds: dict[type, tuple[type, str, bool]] = {
        Repulsion: (RepulsionCoreSetup, "repulsion", False),
        ShortRangeBond: (ShortRangeBondCoreSetup, "srb", False),
        DispersionD3: (DispersionD3CoreSetup, "d3", False),
        DispersionD4: (DispersionD4CoreSetup, "d4", True),
        IES: (IESCoreSetup, "ies", True),
        Halogen: (HalogenCoreSetup, "halogen", False),
    }
    spec = kinds.get(type(component))
    if spec is None or not isinstance(setup, spec[0]):
        return None
    return spec[1], spec[2]


def _interaction_term_kind(
    interaction: Interaction, setup: CoreSetup | None
) -> str | None:
    """Describe only exact built-in interactions with captured setup data."""
    kinds: dict[type, tuple[type, str]] = {
        ES2: (ES2CoreSetup, "es2"),
        ES3: (ES3CoreSetup, "es3"),
        AES2: (AES2CoreSetup, "aes2"),
        DispersionD4SC: (D4SCCoreSetup, "d4sc"),
        GeneralizedBorn: (GeneralizedBornCoreSetup, "generalized_born"),
    }
    spec = kinds.get(type(interaction))
    if spec is None or not isinstance(setup, spec[0]):
        return None
    return spec[1]


def _classical_terms_match(
    captured: tuple[ClassicalTermDescriptor, ...],
    expected: list[tuple[str, str, int, CoreSetup, bool, tuple[str, ...]]],
) -> bool:
    """Compare captured descriptors without constructing temporary Nodes."""
    return len(captured) == len(expected) and all(
        entry.kind == kind
        and entry.label == label
        and entry.position == position
        and entry.setup is setup
        and entry.charge_dependent == charge_dependent
        and entry.geometry_inputs == geometry_inputs
        for entry, (
            kind,
            label,
            position,
            setup,
            charge_dependent,
            geometry_inputs,
        ) in zip(captured, expected)
    )


def _interaction_terms_match(
    captured: tuple[InteractionTermDescriptor, ...],
    expected: list[tuple[str, str, int, CoreSetup, tuple[str, ...]]],
) -> bool:
    """Compare captured descriptors without constructing temporary Nodes."""
    return len(captured) == len(expected) and all(
        entry.kind == kind
        and entry.label == label
        and entry.position == position
        and entry.setup is setup
        and entry.geometry_inputs == geometry_inputs
        for entry, (kind, label, position, setup, geometry_inputs) in zip(
            captured, expected
        )
    )


_FUNCTORCH_C = getattr(torch._C, "_functorch", None)
_IS_FUNCTORCH_WRAPPED = getattr(
    _FUNCTORCH_C, "is_functorch_wrapped_tensor", None
)


def _unwrap_concrete_structural_tensor(tensor: Tensor) -> Tensor | None:
    """Peel non-batched AD wrappers to a plain non-meta Tensor, or return None."""
    if (
        torch._C._get_tracing_state() is not None
        or (
            _get_current_dispatch_mode is not None
            and _get_current_dispatch_mode() is not None
        )
        or torch.overrides.has_torch_function_unary(tensor)
    ):
        return None
    ft = _FUNCTORCH_C
    is_wrapped = _IS_FUNCTORCH_WRAPPED
    raw = tensor
    if ft is not None and is_wrapped is not None and is_wrapped(raw):
        if not callable(getattr(torch._C, "_DisableFuncTorch", None)):
            return None
        for _ in range(16):
            if ft.is_functionaltensor(raw) or ft.is_batchedtensor(raw):
                return None
            if ft.is_gradtrackingtensor(raw):
                raw = ft.get_unwrapped(raw)
                continue
            break
        if is_wrapped(raw):
            return None
    if type(raw) is not Tensor or raw.device.type == "meta":
        return None
    return raw


def _unwrap_concrete_index_helper(ihelp: IndexHelper) -> IndexHelper | None:
    """Return a concrete unwrapped IndexHelper, or None if meta or vmap-wrapped."""
    raw_fields: dict[str, Tensor] = {}
    changed = False
    for name in ihelp._child_names:
        child_tensor = getattr(ihelp, name)
        raw = _unwrap_concrete_structural_tensor(child_tensor)
        if raw is None:
            return None
        raw_fields[name] = raw
        if raw is not child_tensor:
            changed = True
    if not changed:
        return ihelp
    return IndexHelper(**raw_fields, batch_mode=ihelp.batch_mode)


def _clone_detached_concrete(tensor: Tensor) -> Tensor | None:
    """Clone a detached plain structural tensor when concrete values exist."""
    raw = _unwrap_concrete_structural_tensor(tensor)
    if raw is None:
        return None
    with disable_functorch(), torch.no_grad():
        return raw.detach().clone()


@dataclass(frozen=True, kw_only=True, eq=False)
class _SetupIndexToken:
    """Setup-time ownership and tensor-version snapshot for ``System.ihelp``."""

    numbers: Tensor
    numbers_snapshot: Tensor | None
    ihelp: IndexHelper
    integral_driver_type: int | None
    integral_backend: PytorchIntegralSetup | LibcintIntegralSetup | None
    integral_logical_device: torch.device | None
    libcint_shell_layout: ShellLayout | None
    h0_setup: H0Setup | H0SetupNode | None
    es2_shells_to_atom: Tensor | None
    tensor_states: tuple[
        tuple[Tensor, int | None, torch.Size, torch.dtype, torch.device], ...
    ]
    unique_angular_snapshot: Tensor | None
    ushells_per_unique_snapshot: Tensor | None


def _backend_structural_tensors(
    backend: PytorchIntegralSetup | LibcintIntegralSetup | None,
) -> list[Tensor]:
    """Collect backend structural tensors whose mutation invalidates setup."""
    if isinstance(backend, PytorchIntegralSetup):
        plan = backend.pair_plan
        tensors: list[Tensor] = [plan.atom, plan.orbitals_to_atom]
        for cls in plan.classes:
            tensors.extend((cls.ib, cls.jk, cls.idx, cls.weight))
        return tensors
    if isinstance(backend, LibcintIntegralSetup):
        tensors = [backend.numbers]
        if backend.active_atoms is not None:
            tensors.append(backend.active_atoms)
        return tensors
    return []


def _h0_structural_tensors(
    h0_setup: H0Setup | H0SetupNode | None,
) -> list[Tensor]:
    """Collect H0 structural tensors whose mutation invalidates setup."""
    if h0_setup is None:
        return []
    tensors = [h0_setup.numbers, h0_setup.unique, h0_setup.valence]
    if isinstance(h0_setup, H0Setup) and isinstance(
        h0_setup.ihelp, IndexHelper
    ):
        tensors.extend(
            getattr(h0_setup.ihelp, name)
            for name in h0_setup.ihelp._child_names
        )
    return tensors


def _snapshot_tensor_states(
    tensors: Sequence[Tensor],
) -> tuple[
    tuple[Tensor, int | None, torch.Size, torch.dtype, torch.device], ...
]:
    """Record deduplicated ``(tensor, _version, shape, dtype, device)`` states."""
    seen_ids: set[int] = set()
    states: list[
        tuple[Tensor, int | None, torch.Size, torch.dtype, torch.device]
    ] = []
    is_wrapped = _IS_FUNCTORCH_WRAPPED
    for leaf in tensors:
        leaf_id = id(leaf)
        if leaf_id in seen_ids:
            continue
        seen_ids.add(leaf_id)
        dev = leaf.device
        ver = (
            leaf._version
            if (
                type(leaf) is Tensor
                and dev.type != "meta"
                and not leaf.is_inference()
                and (is_wrapped is None or not is_wrapped(leaf))
            )
            else None
        )
        states.append((leaf, ver, leaf.shape, leaf.dtype, dev))
    return tuple(states)


def _build_setup_index_token(
    numbers: Tensor,
    ihelp: IndexHelper,
    integral_setup: IntegralSetup | None,
    h0_setup: H0Setup | H0SetupNode | None,
    es2_setup: ES2CoreSetup | None,
) -> _SetupIndexToken:
    """Capture setup-time identity and version metadata for fast validation."""
    driver_type = None if integral_setup is None else integral_setup.driver_type
    backend = None if integral_setup is None else integral_setup.backend
    logical_device = (
        None if integral_setup is None else integral_setup.logical_device
    )
    shell_layout = (
        backend.shell_layout
        if isinstance(backend, LibcintIntegralSetup)
        else None
    )
    es2_shells = es2_setup.shells_to_atom if es2_setup is not None else None

    tracked: list[Tensor] = [
        numbers,
        *(getattr(ihelp, name) for name in ihelp._child_names),
        *_backend_structural_tensors(backend),
        *_h0_structural_tensors(h0_setup),
    ]
    if es2_shells is not None:
        tracked.append(es2_shells)

    return _SetupIndexToken(
        numbers=numbers,
        numbers_snapshot=_clone_detached_concrete(numbers),
        ihelp=ihelp,
        integral_driver_type=driver_type,
        integral_backend=backend,
        integral_logical_device=logical_device,
        libcint_shell_layout=shell_layout,
        h0_setup=h0_setup,
        es2_shells_to_atom=es2_shells,
        tensor_states=_snapshot_tensor_states(tracked),
        unique_angular_snapshot=_clone_detached_concrete(ihelp.unique_angular),
        ushells_per_unique_snapshot=_clone_detached_concrete(
            ihelp.ushells_per_unique
        ),
    )


def _refresh_setup_index_token(
    token: _SetupIndexToken,
    integral_setup: IntegralSetup | None,
    h0_setup: H0Setup | H0SetupNode | None,
    es2_setup: ES2CoreSetup | None,
) -> _SetupIndexToken:
    """Refresh backend/H0/ES2 tracking after dtype conversion of an intact System."""
    driver_type = None if integral_setup is None else integral_setup.driver_type
    backend = None if integral_setup is None else integral_setup.backend
    logical_device = (
        None if integral_setup is None else integral_setup.logical_device
    )
    shell_layout = (
        backend.shell_layout
        if isinstance(backend, LibcintIntegralSetup)
        else None
    )
    es2_shells = es2_setup.shells_to_atom if es2_setup is not None else None
    tracked: list[Tensor] = [
        token.numbers,
        *(getattr(token.ihelp, name) for name in token.ihelp._child_names),
        *_backend_structural_tensors(backend),
        *_h0_structural_tensors(h0_setup),
    ]
    if es2_shells is not None:
        tracked.append(es2_shells)

    return _SetupIndexToken(
        numbers=token.numbers,
        numbers_snapshot=token.numbers_snapshot,
        ihelp=token.ihelp,
        integral_driver_type=driver_type,
        integral_backend=backend,
        integral_logical_device=logical_device,
        libcint_shell_layout=shell_layout,
        h0_setup=h0_setup,
        es2_shells_to_atom=es2_shells,
        tensor_states=_snapshot_tensor_states(tracked),
        unique_angular_snapshot=token.unique_angular_snapshot,
        ushells_per_unique_snapshot=token.ushells_per_unique_snapshot,
    )


@torch._dynamo.disable
def _eager_index_helpers_equal(a: IndexHelper, b: IndexHelper) -> bool:
    """Compare concrete IndexHelper tensor values outside Dynamo graph capture."""
    if (
        not isinstance(a, IndexHelper)
        or not isinstance(b, IndexHelper)
        or a.batch_mode != b.batch_mode
    ):
        return False
    with disable_functorch(), torch.no_grad():
        raw_a = _unwrap_concrete_index_helper(a)
        if raw_a is None:
            return False
        if a is b:
            return True
        raw_b = _unwrap_concrete_index_helper(b)
        if raw_b is None or raw_a.batch_mode != raw_b.batch_mode:
            return False
        if raw_a is raw_b:
            return True
        a_dev = raw_a.unique_angular.device
        for name in raw_a._child_names:
            ta = getattr(raw_a, name)
            tb = getattr(raw_b, name)
            if (
                ta.device != a_dev
                or tb.device != a_dev
                or ta.shape != tb.shape
                or ta.dtype != tb.dtype
            ):
                return False
        return bool(
            torch.equal(
                torch.cat(
                    [
                        getattr(raw_a, name).reshape(-1)
                        for name in raw_a._child_names
                    ]
                ),
                torch.cat(
                    [
                        getattr(raw_b, name).reshape(-1)
                        for name in raw_b._child_names
                    ]
                ),
            )
        )


def _index_helpers_equal(a: IndexHelper, b: IndexHelper) -> bool:
    """Return whether two IndexHelpers share identical metadata and concrete values."""
    if (
        not isinstance(a, IndexHelper)
        or not isinstance(b, IndexHelper)
        or a.batch_mode != b.batch_mode
    ):
        return False
    a_dev = a.unique_angular.device
    if a_dev.type == "meta":
        return False
    for name in a._child_names:
        ta = getattr(a, name)
        tb = getattr(b, name)
        if (
            ta.device != a_dev
            or tb.device != a_dev
            or ta.shape != tb.shape
            or ta.dtype != tb.dtype
        ):
            return False
    return _eager_index_helpers_equal(a, b)


def _backend_schema_valid(
    backend: PytorchIntegralSetup | LibcintIntegralSetup,
    ihelp_device: torch.device,
) -> bool:
    """Validate local backend Node schemas, including unvalidated tree_unflatten."""
    try:
        backend._validate()
        expected_cgto_dev = (
            torch.device("cpu")
            if isinstance(backend, LibcintIntegralSetup)
            else ihelp_device
        )
        for alpha, coeff in zip(backend.alphas, backend.coeffs):
            if (
                alpha.device != expected_cgto_dev
                or coeff.device != expected_cgto_dev
                or not alpha.is_floating_point()
                or not coeff.is_floating_point()
                or alpha.dtype != coeff.dtype
            ):
                return False
        if isinstance(backend, PytorchIntegralSetup):
            plan = backend.pair_plan
            plan._validate()
            if (
                plan.atom.device != ihelp_device
                or plan.orbitals_to_atom.device != ihelp_device
            ):
                return False
            for cls in plan.classes:
                cls._validate()
                if (
                    cls.ib.device != ihelp_device
                    or cls.jk.device != ihelp_device
                    or cls.idx.device != ihelp_device
                    or cls.weight.device != ihelp_device
                    or not cls.weight.is_floating_point()
                ):
                    return False
        return True
    except (TypeError, ValueError):
        return False


def _h0_float_leaves_valid(
    h0_setup: H0Setup | H0SetupNode,
    n_ushells: int,
    n_unique: int,
    target_device: torch.device,
) -> bool:
    """Validate device, floating-point dtype, and shapes of H0 parameter leaves."""
    if (
        h0_setup.selfenergy.shape != (n_ushells,)
        or h0_setup.shpoly.shape != (n_ushells,)
        or h0_setup.refocc.shape != (n_ushells,)
        or h0_setup.kcn.shape != (n_ushells,)
        or h0_setup.hscale.shape != (n_ushells, n_ushells)
        or h0_setup.rad.shape != (n_unique,)
        or h0_setup.en.shape != (n_unique,)
        or h0_setup.kpair.shape != (n_unique, n_unique)
    ):
        return False
    for leaf in (
        h0_setup.selfenergy,
        h0_setup.shpoly,
        h0_setup.refocc,
        h0_setup.kcn,
        h0_setup.hscale,
        h0_setup.rad,
        h0_setup.en,
        h0_setup.enscale,
        h0_setup.kpair,
    ):
        if leaf.device != target_device or not leaf.is_floating_point():
            return False
    for opt_name in (
        "kq",
        "kqat",
        "h0rad",
        "kdiff",
        "enshell",
        "enscale4",
        "cn_radii",
        "cn_max",
        "cn_kcn",
    ):
        opt_leaf = getattr(h0_setup, opt_name, None)
        if opt_leaf is not None and (
            opt_leaf.device != target_device or not opt_leaf.is_floating_point()
        ):
            return False
    if (
        (h0_setup.kq is not None and h0_setup.kq.shape != (n_ushells,))
        or (h0_setup.kqat is not None and h0_setup.kqat.shape != (n_unique,))
        or (h0_setup.h0rad is not None and h0_setup.h0rad.shape != (n_unique,))
    ):
        return False
    eeq = h0_setup.eeq_model
    if eeq is not None:
        for eeq_leaf in (eeq.chi, eeq.kcn, eeq.eta, eeq.rad):
            if (
                eeq_leaf.device != target_device
                or not eeq_leaf.is_floating_point()
            ):
                return False
    return True


def _h0_concrete_values_match(
    h0_setup: H0Setup | H0SetupNode,
    raw_numbers: Tensor,
    unique: Tensor,
    raw_ihelp: IndexHelper,
) -> bool:
    """Verify H0 structural tensor values against unwrapped numbers and ihelp."""
    raw_h0_numbers = _unwrap_concrete_structural_tensor(h0_setup.numbers)
    raw_h0_unique = _unwrap_concrete_structural_tensor(h0_setup.unique)
    raw_h0_valence = _unwrap_concrete_structural_tensor(h0_setup.valence)
    if (
        raw_h0_numbers is None
        or raw_h0_unique is None
        or raw_h0_valence is None
        or not torch.equal(
            raw_numbers,
            raw_h0_numbers.to(dtype=raw_numbers.dtype),
        )
        or not torch.equal(unique, raw_h0_unique.to(dtype=unique.dtype))
    ):
        return False
    if h0_setup.method == "gfn2":
        expected_valence = torch.ones_like(
            raw_ihelp.unique_angular, dtype=torch.bool
        )
    else:
        same_elem_and_l = (
            raw_ihelp.ushells_to_unique[:, None]
            == raw_ihelp.ushells_to_unique[None, :]
        ) & (
            raw_ihelp.unique_angular[:, None]
            == raw_ihelp.unique_angular[None, :]
        )
        expected_valence = ~same_elem_and_l.tril(diagonal=-1).any(dim=-1)
    if not torch.equal(raw_h0_valence, expected_valence):
        return False
    if isinstance(h0_setup, H0Setup):
        raw_h0_ihelp = _unwrap_concrete_index_helper(h0_setup.ihelp)
        if raw_h0_ihelp is None or (
            raw_h0_ihelp is not raw_ihelp
            and not _index_helpers_equal(raw_h0_ihelp, raw_ihelp)
        ):
            return False
    return True


def _pytorch_backend_matches_ihelp(
    backend: PytorchIntegralSetup, raw_ihelp: IndexHelper
) -> bool:
    """Verify compiled PairPlan metadata and concrete values against canonical ihelp."""
    plan = backend.pair_plan
    raw_atom = _unwrap_concrete_structural_tensor(plan.atom)
    raw_orb_to_atom = _unwrap_concrete_structural_tensor(plan.orbitals_to_atom)
    if raw_atom is None or raw_orb_to_atom is None:
        return False

    weight_dtype = (
        plan.classes[0].weight.dtype if plan.classes else torch.float64
    )
    expected = prepare_pair_plan(
        raw_ihelp, raw_ihelp.device, dtype=weight_dtype
    )
    if (
        plan.nao != expected.nao
        or plan.unique_angular != expected.unique_angular
        or len(plan.classes) != len(expected.classes)
        or not torch.equal(raw_atom, expected.atom)
        or not torch.equal(raw_orb_to_atom, expected.orbitals_to_atom)
    ):
        return False

    for actual_cls, expected_cls in zip(plan.classes, expected.classes):
        raw_ib = _unwrap_concrete_structural_tensor(actual_cls.ib)
        raw_jk = _unwrap_concrete_structural_tensor(actual_cls.jk)
        raw_idx = _unwrap_concrete_structural_tensor(actual_cls.idx)
        raw_weight = _unwrap_concrete_structural_tensor(actual_cls.weight)
        if (
            raw_ib is None
            or raw_jk is None
            or raw_idx is None
            or raw_weight is None
            or (actual_cls.ub, actual_cls.uk, actual_cls.lb, actual_cls.lk)
            != (
                expected_cls.ub,
                expected_cls.uk,
                expected_cls.lb,
                expected_cls.lk,
            )
            or not torch.equal(raw_ib, expected_cls.ib)
            or not torch.equal(raw_jk, expected_cls.jk)
            or not torch.equal(raw_idx, expected_cls.idx)
            or not torch.equal(raw_weight, expected_cls.weight)
        ):
            return False
    return True


def _libcint_backend_matches_ihelp(
    backend: LibcintIntegralSetup,
    raw_numbers: Tensor,
    raw_unique: Tensor,
    raw_ihelp: IndexHelper,
) -> bool:
    """Verify CPU libcint setup numbers, active_atoms, and nested ShellLayout."""
    raw_backend_nums = _unwrap_concrete_structural_tensor(backend.numbers)
    if raw_backend_nums is None:
        return False
    expected_numbers = raw_numbers.to(
        device=raw_backend_nums.device, dtype=raw_backend_nums.dtype
    )
    if not torch.equal(raw_backend_nums, expected_numbers):
        return False

    has_padding = bool((raw_backend_nums <= 0).any())
    if has_padding:
        if backend.active_atoms is None:
            return False
        raw_active = _unwrap_concrete_structural_tensor(backend.active_atoms)
        if raw_active is None:
            return False
        expected_active = (
            torch.nonzero(raw_backend_nums > 0, as_tuple=False)
            .squeeze(-1)
            .to(dtype=raw_active.dtype)
        )
        if not torch.equal(raw_active, expected_active):
            return False
    elif backend.active_atoms is not None:
        return False

    try:
        expected_layout = _build_shell_layout(
            raw_unique, raw_ihelp.ushells_per_unique, raw_ihelp.unique_angular
        )
    except ValueError:
        return False
    return backend.shell_layout == expected_layout


@dataclass(frozen=True, kw_only=True, eq=False)
class System:
    """
    Everything that depends on the atomic numbers only (see :class:`Model`).

    Instances are created by :meth:`Model.setup`.
    """

    numbers: Tensor
    """Atomic numbers (shape: ``(..., nat)``)."""

    config: Config
    """
    Effective configuration. It differs from the one of the model in the
    integral level, which is derived from the parametrization and the
    interactions.
    """

    batch_mode: int
    """
    Batch mode (0: one Structure, 1: Padded batch, 2: Conformer batch). The
    default follows the shape of the atomic numbers.
    """

    ihelp: IndexHelper
    """Index helper (basis, mapping of unique species, shells and orbitals)."""

    classicals: ClassicalList
    """Classical contributions with their parameters gathered."""

    interactions: InteractionList
    """Self-consistent contributions with their parameters gathered."""

    es2_setup: ES2CoreSetup | None
    """Numbers-only setup for pure single-system ES2 matrix construction."""

    es2_interaction: ES2 | None
    """Exact interaction associated with :attr:`es2_setup`, if any."""

    es3_setup: ES3CoreSetup | None
    """Numbers-only parameter data for per-call ES3 interaction data."""

    es3_interaction: ES3 | None
    """Exact interaction associated with :attr:`es3_setup`, if any."""

    aes2_setup: AES2CoreSetup | None
    """Numbers/parameter setup for per-call AES2 geometry data."""

    aes2_interaction: AES2 | None
    """Exact AES2 interaction associated with :attr:`aes2_setup`."""

    d4sc_setup: D4SCCoreSetup | None
    """Numbers/parameter setup for the exact migrated D4SC term."""

    d4sc_interaction: DispersionD4SC | None
    """Exact D4SC interaction associated with :attr:`d4sc_setup`."""

    generalized_born_setup: GeneralizedBornCoreSetup | None
    """Explicit setup for one exact GeneralizedBorn interaction."""

    generalized_born_interaction: GeneralizedBorn | None
    """Exact GeneralizedBorn interaction associated with its setup."""

    integral_setup: IntegralSetup | None
    """Immutable backend setup used by pure integral evaluation."""

    h0_setup: H0Setup | H0SetupNode | None
    """Composition-dependent H0 data for pure matrix construction."""

    _packed_classical_setups: ClassicalListCache = field(repr=False)
    """Setups of a padded or conformer System's D3 Classicals."""

    repulsion_setup: RepulsionCoreSetup | RepulsionSetup | None
    """Numbers-only setup associated with the exact migrated Repulsion."""

    repulsion_classical: Repulsion | None
    """Exact migrated Repulsion associated with :attr:`repulsion_setup`."""

    srb_setup: ShortRangeBondCoreSetup | ShortRangeBondSetup | None
    """Numbers-only setup associated with the exact migrated SRB term."""

    srb_classical: ShortRangeBond | None
    """Exact migrated SRB associated with :attr:`srb_setup`."""

    d4_setup: DispersionD4CoreSetup | DispersionD4Setup | None
    """Numbers-only setup associated with the exact classical D4 term."""

    d4_classical: DispersionD4 | None
    """Exact classical D4 associated with :attr:`d4_setup`."""

    ies_setup: IESCoreSetup | IESSetup | None
    """Numbers/parameter setup for the exact migrated IES term."""

    ies_classical: IES | None
    """Exact IES associated with :attr:`ies_setup`."""

    halogen_setup: HalogenCoreSetup | HalogenSetup | None
    """Numbers/parameter setup for the exact migrated halogen term."""

    halogen_classical: Halogen | None
    """Exact Halogen associated with :attr:`halogen_setup`."""

    dd: DD
    """Device and data type of the tensors of the System."""

    d3_setups: tuple[tuple[DispersionD3, DispersionD3CoreSetup], ...] = ()
    """Captured setup for each exact single-System classical D3 component."""

    exact_core: ExactSystemCore | None = field(default=None, repr=False)
    """Exact unbatched values; None for padded or conformer Systems."""

    _setup_index_token: _SetupIndexToken | None = field(
        default=None, repr=False
    )
    """Setup-time ownership and tensor-version snapshot for fast validation."""

    _custom_classical_snapshots: tuple[
        tuple[Classical, ComponentMutationSnapshot, Any], ...
    ] = field(default=(), repr=False)
    """Setup-time ``(component, mutation_snapshot, setup)`` entries for custom Classicals."""

    _auto_classical_ids: dict[int, str] = field(
        default_factory=dict, repr=False
    )
    """Setup-time ``id(component) -> exclude_token`` map for auto-built Classicals."""

    _auto_interaction_ids: dict[int, str] = field(
        default_factory=dict, repr=False
    )
    """Setup-time ``id(interaction) -> exclude_token`` map for auto-built Interactions."""

    def singlepoint(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Result:
        """Evaluate this System through the fixed-trip transform-safe core."""
        from dxtb._src.calculators.singlepoint import system_singlepoint

        return system_singlepoint(
            self, positions, chrg, spin, field, field_grad
        )

    def energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
        *,
        include_free_energy: bool = True,
    ) -> Tensor:
        """Evaluate the scalar total energy of this System."""
        from dxtb._src.calculators.properties.functional import system_energy

        return system_energy(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
            include_free_energy=include_free_energy,
        )

    def forces(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate nuclear forces ``-dE/dR`` of this System via ``torch.func``."""
        from dxtb._src.calculators.properties.functional import system_forces

        return system_forces(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def hessian(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
        *,
        matrix: bool = False,
    ) -> Tensor:
        """Evaluate the nuclear Hessian ``d^2E/dR^2`` via ``torch.func``."""
        from dxtb._src.calculators.properties.functional import system_hessian

        return system_hessian(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
            matrix=matrix,
        )

    def dipole(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the electric dipole moment ``-dE/dF`` via ``torch.func``."""
        from dxtb._src.calculators.properties.functional import system_dipole

        return system_dipole(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def quadrupole(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the packed electric quadrupole moment ``-3 dE/dG``."""
        from dxtb._src.calculators.properties.functional import (
            system_quadrupole,
        )

        return system_quadrupole(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def polarizability(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the static polarizability tensor ``-d^2E/dF^2``."""
        from dxtb._src.calculators.properties.functional import (
            system_polarizability,
        )

        return system_polarizability(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def dipole_deriv(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the Cartesian dipole derivative ``d(mu)/dR``."""
        from dxtb._src.calculators.properties.functional import (
            system_dipole_deriv,
        )

        return system_dipole_deriv(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def pol_deriv(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the Cartesian polarizability derivative ``d(alpha)/dR``."""
        from dxtb._src.calculators.properties.functional import system_pol_deriv

        return system_pol_deriv(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def hyperpolarizability(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the static first hyperpolarizability ``beta = -d^3E/dF^3``."""
        from dxtb._src.calculators.properties.functional import (
            system_hyperpolarizability,
        )

        return system_hyperpolarizability(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def third_order_directional(
        self,
        positions: Tensor,
        v1: Tensor,
        v2: Tensor | None = None,
        v3: Tensor | None = None,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the third directional energy derivative along ``(v1, v2, v3)``."""
        from dxtb._src.calculators.properties.functional import (
            system_third_order_directional,
        )

        return system_third_order_directional(
            self,
            positions,
            v1,
            v2=v2,
            v3=v3,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def bond_orders(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the Wiberg bond order matrix ``(nat, nat)``."""
        from dxtb._src.calculators.properties.functional import (
            system_bond_orders,
        )

        return system_bond_orders(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
        )

    def vibration(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
        *,
        project_translational: bool = True,
        project_rotational: bool = True,
    ):
        """Perform vibrational analysis purely via ``System`` transforms."""
        from dxtb._src.calculators.properties.functional import system_vibration

        return system_vibration(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    def ir(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
        *,
        project_translational: bool = True,
        project_rotational: bool = True,
    ):
        """Calculate IR frequencies and intensities without mutable cache resets."""
        from dxtb._src.calculators.properties.functional import system_ir

        return system_ir(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    def raman(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
        *,
        project_translational: bool = True,
        project_rotational: bool = True,
    ):
        """Calculate Raman frequencies, intensities, and depolarization ratios."""
        from dxtb._src.calculators.properties.functional import system_raman

        return system_raman(
            self,
            positions,
            chrg=chrg,
            spin=spin,
            field=field,
            field_grad=field_grad,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    @property
    def d3_setup(self) -> DispersionD3CoreSetup | None:
        """Return the sole exact D3 setup when the System has one."""
        return self.d3_setups[0][1] if len(self.d3_setups) == 1 else None

    def classical_setup(self, component: Classical) -> CoreSetup | None:
        """Return the active core classical setup Node for ``component``."""
        setup = self._builtin_classical_setup(component)
        return (
            setup
            if isinstance(
                setup,
                (
                    RepulsionCoreSetup,
                    ShortRangeBondCoreSetup,
                    DispersionD3CoreSetup,
                    DispersionD4CoreSetup,
                    IESCoreSetup,
                    HalogenCoreSetup,
                ),
            )
            else None
        )

    def interaction_setup(self, component: Interaction) -> CoreSetup | None:
        """Return the active core interaction setup Node for ``component``."""
        setup = self._builtin_interaction_setup(component)
        return (
            setup
            if isinstance(
                setup,
                (
                    ES2CoreSetup,
                    ES3CoreSetup,
                    AES2CoreSetup,
                    D4SCCoreSetup,
                    GeneralizedBornCoreSetup,
                ),
            )
            else None
        )

    def _builtin_classical_setup(self, component: Classical) -> Any:
        if component is self.repulsion_classical:
            return self.repulsion_setup
        if component is self.srb_classical:
            return self.srb_setup
        if component is self.d4_classical:
            return self.d4_setup
        if component is self.ies_classical:
            return self.ies_setup
        if component is self.halogen_classical:
            return self.halogen_setup
        return next(
            (value for owner, value in self.d3_setups if owner is component),
            None,
        )

    def _builtin_interaction_setup(self, interaction: Interaction) -> Any:
        if interaction is self.es2_interaction:
            return self.es2_setup
        if interaction is self.es3_interaction:
            return self.es3_setup
        if interaction is self.aes2_interaction:
            return self.aes2_setup
        if interaction is self.d4sc_interaction:
            return self.d4sc_setup
        if interaction is self.generalized_born_interaction:
            return self.generalized_born_setup
        return None

    @torch._dynamo.disable
    def _setup_token_without_ihelp_intact(self) -> bool:
        """Return whether captured setup state matches token apart from ``self.ihelp``."""
        token = self._setup_index_token
        if token is None:
            return False
        driver_type = (
            None
            if self.integral_setup is None
            else self.integral_setup.driver_type
        )
        backend = (
            None if self.integral_setup is None else self.integral_setup.backend
        )
        logical_device = (
            None
            if self.integral_setup is None
            else self.integral_setup.logical_device
        )
        shell_layout = (
            backend.shell_layout
            if isinstance(backend, LibcintIntegralSetup)
            else None
        )
        es2_shells = (
            self.es2_setup.shells_to_atom
            if self.es2_setup is not None
            else None
        )
        is_wrapped = _IS_FUNCTORCH_WRAPPED
        return (
            self.numbers is token.numbers
            and driver_type == token.integral_driver_type
            and backend is token.integral_backend
            and logical_device == token.integral_logical_device
            and shell_layout == token.libcint_shell_layout
            and self.h0_setup is token.h0_setup
            and es2_shells is token.es2_shells_to_atom
            and all(
                ver is not None
                and type(t) is Tensor
                and t.device == dev
                and not t.is_inference()
                and (is_wrapped is None or not is_wrapped(t))
                and t._version == ver
                and t.shape == shape
                and t.dtype is dtype
                for t, ver, shape, dtype, dev in token.tensor_states
            )
        )

    def _has_intact_setup_token(self) -> bool:
        """Return whether setup-time structural tensors match their version token."""
        token = self._setup_index_token
        return (
            token is not None
            and self.ihelp is token.ihelp
            and self._setup_token_without_ihelp_intact()
        )

    def _fast_token_compatible(self) -> bool:
        """Validate intact setup-token state in O(N_leaves) without device sync."""
        if not self._has_intact_setup_token() or self.batch_mode not in (
            0,
            1,
            2,
        ):
            return False
        ihelp = self.ihelp
        n_ushells = ihelp.unique_angular.shape[0]
        n_unique = ihelp.ushells_per_unique.shape[0]
        if self.numbers.numel() == 0 or n_ushells == 0 or n_unique == 0:
            return False
        if self.h0_setup is not None:
            if not _h0_float_leaves_valid(
                self.h0_setup, n_ushells, n_unique, self.numbers.device
            ):
                return False
            if (
                isinstance(self.h0_setup, H0Setup)
                and self.h0_setup.ihelp is not ihelp
                and not _index_helpers_equal(self.h0_setup.ihelp, ihelp)
            ):
                return False
        if self.batch_mode != 0:
            return (
                self.integral_setup is None
                and self.es2_setup is None
                and self.es3_setup is None
            )
        nat = self.numbers.shape[0]
        n_shells = ihelp.shells_to_atom.shape[0]
        if self.integral_setup is not None:
            backend = self.integral_setup.backend
            if (
                backend is None
                or len(backend.alphas) != n_ushells
                or len(backend.coeffs) != n_ushells
            ):
                return False
            expected_cgto_dev = (
                torch.device("cpu")
                if isinstance(backend, LibcintIntegralSetup)
                else ihelp.device
            )
            for alpha, coeff in zip(backend.alphas, backend.coeffs):
                if (
                    alpha.device != expected_cgto_dev
                    or coeff.device != expected_cgto_dev
                    or not alpha.is_floating_point()
                    or not coeff.is_floating_point()
                    or alpha.dtype != coeff.dtype
                    or alpha.ndim < 1
                    or alpha.shape != coeff.shape
                ):
                    return False
            if isinstance(backend, PytorchIntegralSetup) and any(
                cls.weight.device != ihelp.device
                or not cls.weight.is_floating_point()
                for cls in backend.pair_plan.classes
            ):
                return False
        if self.es2_setup is not None:
            if (
                self.es2_setup.shell_resolved
                and self.es2_setup.shells_to_atom is None
            ):
                return False
            if self.es2_setup.shells_to_atom is not None and (
                self.es2_setup.shells_to_atom.device != self.numbers.device
                or self.es2_setup.shells_to_atom.shape != (n_shells,)
                or self.es2_setup.shells_to_atom.is_floating_point()
                or self.es2_setup.shells_to_atom.dtype is torch.bool
            ):
                return False
        if self.es3_setup is not None and (
            self.es3_setup.hubbard_derivs.device != self.numbers.device
            or not self.es3_setup.hubbard_derivs.is_floating_point()
            or self.es3_setup.hubbard_derivs.shape
            != ((n_shells,) if self.es3_setup.shell_resolved else (nat,))
        ):
            return False
        return True

    def _ihelp_schema_compatible(self) -> bool:
        """Return whether ``ihelp`` and setup shapes/dtypes/devices match schema."""
        if (
            self.numbers.numel() == 0
            or self.numbers.is_floating_point()
            or self.numbers.dtype is torch.bool
        ):
            return False
        ihelp = self.ihelp
        if (
            not isinstance(ihelp, IndexHelper)
            or ihelp.batch_mode != self.batch_mode
        ):
            return False
        try:
            ihelp._validate()
        except (TypeError, ValueError):
            return False
        if (
            ihelp.unique_angular.numel() == 0
            or ihelp.ushells_per_unique.numel() == 0
            or ihelp.unique_angular.dtype is not torch.int64
            or ihelp.unique_angular.device != self.numbers.device
        ):
            return False

        n_ushells = ihelp.unique_angular.shape[0]
        n_unique = ihelp.ushells_per_unique.shape[0]
        if self.h0_setup is not None:
            if (
                self.h0_setup.numbers.device != self.numbers.device
                or self.h0_setup.numbers.shape != self.numbers.shape
                or self.h0_setup.numbers.is_floating_point()
                or self.h0_setup.numbers.dtype is torch.bool
                or self.h0_setup.unique.device != self.numbers.device
                or self.h0_setup.unique.shape != (n_unique,)
                or self.h0_setup.unique.is_floating_point()
                or self.h0_setup.unique.dtype is torch.bool
                or self.h0_setup.valence.device != self.numbers.device
                or self.h0_setup.valence.shape != (n_ushells,)
                or self.h0_setup.valence.dtype is not torch.bool
                or not _h0_float_leaves_valid(
                    self.h0_setup, n_ushells, n_unique, self.numbers.device
                )
            ):
                return False
            if isinstance(self.h0_setup, H0Setup):
                h0_ihelp = self.h0_setup.ihelp
                if (
                    not isinstance(h0_ihelp, IndexHelper)
                    or h0_ihelp.batch_mode != self.batch_mode
                ):
                    return False
                if h0_ihelp is not ihelp:
                    try:
                        h0_ihelp._validate()
                    except (TypeError, ValueError):
                        return False
                    if (
                        h0_ihelp.unique_angular.dtype is not torch.int64
                        or h0_ihelp.unique_angular.device != self.numbers.device
                        or any(
                            getattr(h0_ihelp, name).shape
                            != getattr(ihelp, name).shape
                            for name in ihelp._child_names
                        )
                    ):
                        return False

        if self.batch_mode != 0:
            return (
                self.numbers.ndim == 2
                and ihelp.atom_to_unique.shape == self.numbers.shape
                and self.integral_setup is None
                and self.es2_setup is None
                and self.es3_setup is None
            )

        if self.numbers.ndim != 1 or ihelp.unique_angular.ndim != 1:
            return False
        nat = self.numbers.shape[0]
        if ihelp.atom_to_unique.shape != (nat,):
            return False

        n_shells = ihelp.shells_to_atom.shape[0]
        n_orbitals = ihelp.orbitals_to_shell.shape[0]

        if self.integral_setup is not None:
            backend = self.integral_setup.backend
            if (
                self.integral_setup.logical_device != self.numbers.device
                or backend is None
            ):
                return False
            if self.integral_setup.driver_type == labels.INTDRIVER_PYTORCH:
                if not isinstance(backend, PytorchIntegralSetup):
                    return False
                plan = backend.pair_plan
                if (
                    not _backend_schema_valid(backend, ihelp.device)
                    or plan.nao != n_orbitals
                    or len(plan.unique_angular) != n_ushells
                    or plan.atom.shape != (n_shells,)
                ):
                    return False
            elif self.integral_setup.driver_type == labels.INTDRIVER_LIBCINT:
                if not isinstance(backend, LibcintIntegralSetup):
                    return False
                if (
                    not _backend_schema_valid(backend, ihelp.device)
                    or backend.numbers.shape != (nat,)
                    or len(backend.alphas) != n_ushells
                    or len(backend.shell_layout) > n_unique
                ):
                    return False
            else:
                return False

        if self.es2_setup is not None:
            if (
                self.es2_setup.shell_resolved
                and self.es2_setup.shells_to_atom is None
            ):
                return False
            if self.es2_setup.shells_to_atom is not None and (
                self.es2_setup.shells_to_atom.device != self.numbers.device
                or self.es2_setup.shells_to_atom.shape != (n_shells,)
                or self.es2_setup.shells_to_atom.is_floating_point()
                or self.es2_setup.shells_to_atom.dtype is torch.bool
            ):
                return False
        if self.es3_setup is not None and (
            self.es3_setup.hubbard_derivs.device != self.numbers.device
            or not self.es3_setup.hubbard_derivs.is_floating_point()
            or self.es3_setup.hubbard_derivs.shape
            != ((n_shells,) if self.es3_setup.shell_resolved else (nat,))
        ):
            return False
        return True

    def _ihelp_compatible(self) -> bool:
        """Return whether ``ihelp`` concretely matches this System's structural setup."""
        if self.numbers.device.type == "meta":
            return False
        if self._fast_token_compatible():
            return True
        if not self._ihelp_schema_compatible():
            return False
        if self.batch_mode != 0:
            return self._eager_ihelp_compatible_batched(self.ihelp)
        backend = (
            None if self.integral_setup is None else self.integral_setup.backend
        )
        return self._eager_ihelp_compatible(
            self.ihelp,
            backend,
            self.numbers.shape[0],
            self.ihelp.shells_to_atom.shape[0],
        )

    @torch._dynamo.disable
    def _eager_ihelp_compatible_batched(self, ihelp: IndexHelper) -> bool:
        """Verify concrete setup tensor values for padded/conformer Systems."""
        token = self._setup_index_token
        if token is not None and self._setup_token_without_ihelp_intact():
            return _index_helpers_equal(ihelp, token.ihelp)

        with disable_functorch(), torch.no_grad():
            raw_numbers = _unwrap_concrete_structural_tensor(self.numbers)
            raw_ihelp = _unwrap_concrete_index_helper(ihelp)
            if (
                raw_numbers is None
                or raw_ihelp is None
                or bool((raw_numbers < 0).any())
            ):
                return False

            if token is not None and (
                token.numbers_snapshot is None
                or token.unique_angular_snapshot is None
                or token.ushells_per_unique_snapshot is None
                or not torch.equal(
                    raw_numbers,
                    token.numbers_snapshot.to(
                        device=raw_numbers.device, dtype=raw_numbers.dtype
                    ),
                )
                or not torch.equal(
                    raw_ihelp.unique_angular,
                    token.unique_angular_snapshot.to(
                        device=raw_ihelp.device,
                        dtype=raw_ihelp.unique_angular.dtype,
                    ),
                )
                or not torch.equal(
                    raw_ihelp.ushells_per_unique,
                    token.ushells_per_unique_snapshot.to(
                        device=raw_ihelp.device,
                        dtype=raw_ihelp.ushells_per_unique.dtype,
                    ),
                )
            ):
                return False

            unique, expected_atom_to_unique = torch.unique(
                raw_numbers, return_inverse=True
            )
            if (
                raw_ihelp.ushells_per_unique.shape != unique.shape
                or not torch.equal(
                    raw_ihelp.atom_to_unique, expected_atom_to_unique
                )
                or bool((raw_ihelp.ushells_per_unique < 0).any())
            ):
                return False

            if self.h0_setup is not None and not _h0_concrete_values_match(
                self.h0_setup, raw_numbers, unique, raw_ihelp
            ):
                return False

            angular: dict[int, list[int]] = {}
            for idx, z_val in enumerate(unique.tolist()):
                z = int(z_val)
                shells_z = [
                    int(v)
                    for v in raw_ihelp.unique_angular[
                        raw_ihelp.ushells_to_unique == idx
                    ].tolist()
                ]
                if z <= 0:
                    if shells_z != [-1]:
                        return False
                else:
                    if not shells_z or any(ang < 0 for ang in shells_z):
                        return False
                    angular[z] = shells_z

            expected_ihelp = IndexHelper.from_numbers_angular(
                raw_numbers,
                angular=angular,
                batch_mode=self.batch_mode,
                move_to_numbers_device=True,
            )
            return _index_helpers_equal(raw_ihelp, expected_ihelp)

    @torch._dynamo.disable
    def _eager_ihelp_compatible(
        self,
        ihelp: IndexHelper,
        backend: PytorchIntegralSetup | LibcintIntegralSetup | None,
        nat: int,
        n_shells: int,
    ) -> bool:
        """Verify concrete setup tensor values outside Dynamo graph capture."""
        token = self._setup_index_token
        if token is not None and self._setup_token_without_ihelp_intact():
            return _index_helpers_equal(ihelp, token.ihelp)

        with disable_functorch(), torch.no_grad():
            raw_numbers = _unwrap_concrete_structural_tensor(self.numbers)
            raw_ihelp = _unwrap_concrete_index_helper(ihelp)
            if (
                raw_numbers is None
                or raw_ihelp is None
                or bool((raw_numbers < 0).any())
            ):
                return False

            if token is not None and (
                token.numbers_snapshot is None
                or token.unique_angular_snapshot is None
                or token.ushells_per_unique_snapshot is None
                or not torch.equal(
                    raw_numbers,
                    token.numbers_snapshot.to(
                        device=raw_numbers.device, dtype=raw_numbers.dtype
                    ),
                )
                or not torch.equal(
                    raw_ihelp.unique_angular,
                    token.unique_angular_snapshot.to(
                        device=raw_ihelp.device,
                        dtype=raw_ihelp.unique_angular.dtype,
                    ),
                )
                or not torch.equal(
                    raw_ihelp.ushells_per_unique,
                    token.ushells_per_unique_snapshot.to(
                        device=raw_ihelp.device,
                        dtype=raw_ihelp.ushells_per_unique.dtype,
                    ),
                )
            ):
                return False

            if (
                self.es2_setup is not None
                and self.es2_setup.shells_to_atom is not None
            ):
                raw_es2_shells = _unwrap_concrete_structural_tensor(
                    self.es2_setup.shells_to_atom
                )
                if (
                    raw_es2_shells is None
                    or raw_es2_shells.is_floating_point()
                    or raw_es2_shells.dtype is torch.bool
                    or not torch.equal(
                        raw_ihelp.shells_to_atom,
                        raw_es2_shells.to(dtype=raw_ihelp.shells_to_atom.dtype),
                    )
                ):
                    return False

            unique, expected_atom_to_unique = torch.unique(
                raw_numbers, return_inverse=True
            )
            if (
                raw_ihelp.ushells_per_unique.shape != unique.shape
                or not torch.equal(
                    raw_ihelp.atom_to_unique, expected_atom_to_unique
                )
                or bool((raw_ihelp.ushells_per_unique < 0).any())
            ):
                return False

            device = raw_ihelp.device
            dtype = raw_ihelp.unique_angular.dtype
            expected_ushells_to_unique = torch.repeat_interleave(
                torch.arange(unique.shape[0], device=device, dtype=dtype),
                raw_ihelp.ushells_per_unique,
            )
            if not torch.equal(
                raw_ihelp.ushells_to_unique, expected_ushells_to_unique
            ):
                return False

            if bool((unique > 0).any()):
                active_ushells = unique[raw_ihelp.ushells_to_unique] > 0
                if bool(
                    (raw_ihelp.ushells_per_unique[unique > 0] <= 0).any()
                ) or bool((raw_ihelp.unique_angular[active_ushells] < 0).any()):
                    return False
            if bool((unique <= 0).any()):
                inactive_ushells = unique[raw_ihelp.ushells_to_unique] <= 0
                if bool(
                    (raw_ihelp.ushells_per_unique[unique <= 0] != 1).any()
                ) or bool(
                    (raw_ihelp.unique_angular[inactive_ushells] != -1).any()
                ):
                    return False

            if self.h0_setup is not None and not _h0_concrete_values_match(
                self.h0_setup, raw_numbers, unique, raw_ihelp
            ):
                return False

            expected_shells_per_atom = raw_ihelp.ushells_per_unique[
                raw_ihelp.atom_to_unique
            ]
            if not torch.equal(
                raw_ihelp.shells_per_atom, expected_shells_per_atom
            ):
                return False

            expected_shell_index = (
                torch.cumsum(raw_ihelp.shells_per_atom, -1)
                - raw_ihelp.shells_per_atom
            )
            expected_shell_index = torch.where(
                raw_ihelp.shells_per_atom == 0, -999, expected_shell_index
            )
            if not torch.equal(raw_ihelp.shell_index, expected_shell_index):
                return False

            expected_shells_to_atom = torch.repeat_interleave(
                torch.arange(nat, device=device, dtype=dtype),
                raw_ihelp.shells_per_atom,
            )
            if not torch.equal(
                raw_ihelp.shells_to_atom, expected_shells_to_atom
            ):
                return False

            ushell_offset = (
                torch.cumsum(raw_ihelp.ushells_per_unique, -1)
                - raw_ihelp.ushells_per_unique
            )
            expected_shells_to_ushell = (
                ushell_offset[
                    raw_ihelp.atom_to_unique[raw_ihelp.shells_to_atom]
                ]
                + torch.arange(n_shells, device=device, dtype=dtype)
                - raw_ihelp.shell_index[raw_ihelp.shells_to_atom]
            )
            if not torch.equal(
                raw_ihelp.shells_to_ushell, expected_shells_to_ushell
            ):
                return False

            if bool(
                (
                    raw_ihelp.shells_to_ushell
                    >= raw_ihelp.unique_angular.shape[0]
                ).any()
            ):
                return False
            expected_angular = torch.where(
                raw_ihelp.shells_to_ushell >= 0,
                raw_ihelp.unique_angular[
                    raw_ihelp.shells_to_ushell.clamp(min=0)
                ],
                -999,
            )
            if not torch.equal(raw_ihelp.angular, expected_angular):
                return False

            expected_orbitals_per_shell = torch.where(
                raw_ihelp.angular >= 0, 2 * raw_ihelp.angular + 1, 0
            )
            if not torch.equal(
                raw_ihelp.orbitals_per_shell, expected_orbitals_per_shell
            ):
                return False

            expected_orbital_index = (
                torch.cumsum(raw_ihelp.orbitals_per_shell, -1)
                - raw_ihelp.orbitals_per_shell
            )
            expected_orbital_index = torch.where(
                raw_ihelp.orbitals_per_shell == 0, -999, expected_orbital_index
            )
            if not torch.equal(raw_ihelp.orbital_index, expected_orbital_index):
                return False

            expected_orbitals_to_shell = torch.repeat_interleave(
                torch.arange(n_shells, device=device, dtype=dtype),
                raw_ihelp.orbitals_per_shell,
            )
            if not torch.equal(
                raw_ihelp.orbitals_to_shell, expected_orbitals_to_shell
            ):
                return False

            if isinstance(backend, PytorchIntegralSetup):
                if not _pytorch_backend_matches_ihelp(backend, raw_ihelp):
                    return False
            elif isinstance(backend, LibcintIntegralSetup):
                if not _libcint_backend_matches_ihelp(
                    backend, raw_numbers, unique, raw_ihelp
                ):
                    return False

            return True

    def _active_exact_core(self) -> ExactSystemCore | None:
        """Return the captured core while its façade setup references agree."""
        core = self.exact_core
        if (
            self.batch_mode != 0
            or core is None
            or core.numbers is not self.numbers
            or core.config is not self.config
            or core.h0_setup is not self.h0_setup
            or not self._ihelp_compatible()
        ):
            return None

        if core.integral is None:
            if self.integral_setup is not None:
                return None
        elif self.integral_setup is None or (
            core.integral is not self.integral_setup
            and (
                core.integral.driver_type != self.integral_setup.driver_type
                or core.integral.intlevel != self.integral_setup.intlevel
                or core.integral.backend is not self.integral_setup.backend
            )
        ):
            return None
        if core.integral is not None and (
            type(core.integral.backend)
            is not {
                labels.INTDRIVER_PYTORCH: PytorchIntegralSetup,
                labels.INTDRIVER_LIBCINT: LibcintIntegralSetup,
            }.get(core.integral.driver_type)
        ):
            return None

        d3_components = tuple(
            component
            for component in self.classicals.components
            if type(component) is DispersionD3
        )
        d3_owners = tuple(owner for owner, _ in self.d3_setups)
        if len(d3_owners) != len(d3_components) or any(
            owner is not component
            for owner, component in zip(d3_owners, d3_components)
        ):
            return None

        expected_classical_terms: list[
            tuple[str, str, int, CoreSetup, bool, tuple[str, ...]]
        ] = []
        for position, component in enumerate(self.classicals.components):
            setup = self._builtin_classical_setup(component)
            if isinstance(
                setup,
                (
                    RepulsionCoreSetup,
                    ShortRangeBondCoreSetup,
                    DispersionD3CoreSetup,
                    DispersionD4CoreSetup,
                    IESCoreSetup,
                    HalogenCoreSetup,
                ),
            ):
                kind = _classical_term_kind(component, setup)
                if kind is None:
                    return None
                expected_classical_terms.append(
                    (
                        kind[0],
                        component.label,
                        position,
                        setup,
                        kind[1],
                        ("positions",),
                    )
                )
            elif type(component) not in BUILTIN_CLASSICAL_TYPES:
                term = core.classical_term_at(position)
                if (
                    term is not None
                    and term.kind == "custom"
                    and isinstance(term.setup, CustomClassicalCoreSetup)
                    and term.setup.matches_component_state(component)
                ):
                    expected_classical_terms.append(
                        (
                            "custom",
                            component.label,
                            position,
                            term.setup,
                            True,
                            ("positions",),
                        )
                    )

        expected_interaction_terms: list[
            tuple[str, str, int, CoreSetup, tuple[str, ...]]
        ] = []
        for position, interaction in enumerate(self.interactions.components):
            setup = self._builtin_interaction_setup(interaction)
            if isinstance(
                setup,
                (
                    ES2CoreSetup,
                    ES3CoreSetup,
                    AES2CoreSetup,
                    D4SCCoreSetup,
                    GeneralizedBornCoreSetup,
                ),
            ):
                kind = _interaction_term_kind(interaction, setup)
                if kind is None:
                    return None
                expected_interaction_terms.append(
                    (
                        kind,
                        interaction.label,
                        position,
                        setup,
                        ("positions",),
                    )
                )

        if not _classical_terms_match(
            core.classical_terms, expected_classical_terms
        ):
            return None
        if not _interaction_terms_match(
            core.interaction_terms, expected_interaction_terms
        ):
            return None

        token = self._setup_index_token
        if core.index is not self.ihelp and not (
            token is not None
            and core.index is token.ihelp
            and self._setup_token_without_ihelp_intact()
        ):
            if not _index_helpers_equal(core.index, self.ihelp):
                return None
        return core

    @property
    def classical_cache(self) -> ClassicalListCache:
        """Build a view over captured setup values on demand."""
        cache = ClassicalListCache()
        for component in self.classicals.components:
            setup = self._builtin_classical_setup(component)
            if setup is None:
                if component.label not in self._packed_classical_setups:
                    continue
                setup = self._packed_classical_setups[component.label]
            if setup is not None:
                cache[component.label] = setup
        return cache


@dataclass(frozen=True, kw_only=True, eq=False)
class Model:
    """
    Parametrization, configuration and additional components of an xTB
    method, independent of any molecule.
    """

    par: ParamModule
    """Parameters (the leaves that gradients flow to)."""

    config: Config = field(default_factory=Config)
    """Configuration."""

    classical: tuple[Classical, ...] = ()
    """Additional Classical Components."""

    interaction: tuple[Interaction, ...] = ()
    """Additional Interaction Components."""

    auto_int_level: bool = True
    """
    Set the integral level to the maximum that the parametrization requires
    (it is never lowered). Turned off in tests that need no integrals.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.par, ParamModule):
            raise TypeError(
                "The parameters must be a `ParamModule` (the calculator "
                f"converts a `Param`), but '{type(self.par)}' was given."
            )
        if not isinstance(self.classical, tuple):
            raise TypeError("The additional classicals must be a tuple.")
        if not isinstance(self.interaction, tuple):
            raise TypeError("The additional interactions must be a tuple.")

    @property
    @_serialize_parameter_setup
    def parameter_tree(self) -> ModelParameterNode:
        """Return registered dynamic leaves for this Model's parameters."""
        return ModelParameterNode(
            parameters=ModuleNode.from_module(self.par),
            config=self.config,
            auto_int_level=self.auto_int_level,
        )

    @_serialize_parameter_setup
    def setup_functional(
        self,
        numbers: Tensor,
        parameter_tree: ModelParameterNode,
        *,
        batch_mode: int | None = None,
        dd: DD | None = None,
    ) -> System:
        """Build a System while substituted Model parameter leaves are active.

        ``parameter_tree`` must originate from this Model. Its dynamic
        parameter and buffer values are installed for the duration of the
        existing :meth:`setup` implementation, then restored by
        ``torch.func.functional_call``. Model parameter-tree snapshots and
        setup calls through façades sharing this ParamModule are serialized
        during that interval; direct concurrent ParamModule access is outside
        this synchronization boundary.
        """
        if not isinstance(parameter_tree, ModelParameterNode):
            raise TypeError(
                "Functional parameter tree must be a ModelParameterNode."
            )
        if not isinstance(parameter_tree.parameters, ModuleNode):
            raise TypeError(
                "Functional parameter tree parameters must be a ModuleNode."
            )
        if parameter_tree.parameters.module is not self.par:
            raise ValueError(
                "Functional parameter tree does not belong to this Model: "
                "it uses a different ParamModule owner."
            )
        if parameter_tree.config is not self.config:
            raise ValueError(
                "Functional parameter tree uses a different Config object."
            )
        if parameter_tree.auto_int_level != self.auto_int_level:
            raise ValueError(
                "Functional parameter tree has incompatible auto_int_level."
            )

        # Take one shallow snapshot so validation and functional_call consume
        # the same tensor references, even if the caller later edits the
        # ModuleNode's mutable mapping.
        state_values = dict(parameter_tree.parameters.params)
        expected_state = {
            name: (tuple(value.shape), value.dtype)
            for name, value in (
                *self.par.named_parameters(),
                *self.par.named_buffers(),
            )
        }
        actual_names = set(state_values)
        expected_names = set(expected_state)
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing or unexpected:
            raise ValueError(
                "Functional parameter tree state keys do not match the "
                f"ParamModule (missing={missing}, unexpected={unexpected})."
            )
        state_device: torch.device | None = None
        device_owner: str | None = None
        floating_dtype: torch.dtype | None = None
        dtype_owner: str | None = None
        for name, value in state_values.items():
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Functional state entry '{name}' must be a Tensor."
                )

            if state_device is None:
                state_device = value.device
                device_owner = name
            elif value.device != state_device:
                raise ValueError(
                    "Functional parameter tree state tensors must share one "
                    f"device; '{device_owner}' is on {state_device}, but "
                    f"'{name}' is on {value.device}."
                )

            expected_shape, expected_dtype = expected_state[name]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Functional state entry '{name}' has shape "
                    f"{tuple(value.shape)}, expected {expected_shape}."
                )
            if expected_dtype.is_floating_point:
                if not value.is_floating_point() or value.is_complex():
                    raise TypeError(
                        f"Functional state entry '{name}' must have a real "
                        f"floating dtype, got {value.dtype}."
                    )
                if floating_dtype is None:
                    floating_dtype = value.dtype
                    dtype_owner = name
                elif value.dtype != floating_dtype:
                    raise TypeError(
                        "Functional parameter tree floating state tensors "
                        "must share one dtype; "
                        f"'{dtype_owner}' has {floating_dtype}, but '{name}' "
                        f"has {value.dtype}. Convert the complete tree "
                        "instead of mutating an individual state entry."
                    )
            elif value.dtype != expected_dtype:
                raise TypeError(
                    f"Functional state entry '{name}' must preserve its "
                    f"structural dtype {expected_dtype}, got {value.dtype}."
                )

        adapter = _ModelSetupAdapter(self)
        state = {f"par.{name}": value for name, value in state_values.items()}
        return torch.func.functional_call(
            adapter,
            state,
            (numbers,),
            {"batch_mode": batch_mode, "dd": dd},
            strict=True,
        )

    @_serialize_parameter_setup
    def setup(
        self,
        numbers: Tensor,
        *,
        batch_mode: int | None = None,
        dd: DD | None = None,
    ) -> System:
        """
        Set up the System for the atomic numbers.

        The result is a function of the parameters and the numbers, i.e.,
        everything that is computed here (including the gathering of the
        element parameters) can be differentiated with respect to the
        parameters.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers (shape: ``(..., nat)``).
        batch_mode : int | None, optional
            Batch mode. The default follows the shape of `numbers` (0 for one
            Structure, 1 for a Padded batch). Mode 2 (Conformer batch) has
            to be requested.
        dd : DD | None, optional
            Device and data type of the tensors. Default: the ones of the
            parameters.

        Returns
        -------
        System
            Data of the System that depends on `numbers` only.
        """
        par = self.par
        if not torch.is_inference_mode_enabled() and any(
            t.is_inference() for t in (*par.parameters(), *par.buffers())
        ):
            for name, val in (
                *par.named_parameters(),
                *par.named_buffers(),
            ):
                if val.is_inference() and val.requires_grad:
                    raise RuntimeError(
                        f"ParamModule entry '{name}' is an inference-mode "
                        "tensor with requires_grad=True and cannot participate "
                        "in autograd. Construct or replace the parameter "
                        "outside torch.inference_mode()."
                    )
            state = {
                f"par.{name}": (
                    val.detach().clone() if val.is_inference() else val
                )
                for name, val in (
                    *par.named_parameters(),
                    *par.named_buffers(),
                )
            }
            return torch.func.functional_call(
                _ModelSetupAdapter(self),
                state,
                (numbers,),
                {"batch_mode": batch_mode, "dd": dd},
                strict=True,
            )

        config = self.config
        exclude = set(config.exclude)
        dd = par.dd if dd is None else dd
        unique = torch.unique(numbers)

        if batch_mode is None or (batch_mode == 0 and numbers.ndim > 1):
            batch_mode = 1 if numbers.ndim > 1 else 0
        if batch_mode not in (0, 1, 2):
            raise ValueError(
                f"Invalid batch mode '{batch_mode}'. Must be one of [0, 1, 2]."
            )

        # Set integral level based on parametrization. For the tests, we want
        # to turn this off. Otherwise, all GFN2-xTB tests will fail without
        # `libcint`, even when integrals are not tested (e.g. D4SC).
        if self.auto_int_level:
            if par.meta is not None and par.meta.name is not None:
                name = par.meta.name.casefold()
                if any(method in name for method in ("gfn0", "gfn1")):
                    config = _with_int_level(
                        config, max(labels.INTLEVEL_HCORE, config.ints.level)
                    )
                elif "gfn2" in name:
                    config = _with_int_level(
                        config,
                        max(labels.INTLEVEL_QUADRUPOLE, config.ints.level),
                    )

        # PERF: The IndexHelper is created on CPU and moved to the device of
        # the `number` tensor. This is required for the instantiation of the
        # integral classes later. However, if the `libcint` interface is used,
        # we need to transfer the IndexHelper to the CPU again.
        ihelp = IndexHelper.from_numbers(numbers, par, batch_mode)

        ################
        # INTERACTIONS #
        ################

        OutputHandler.write_stdout_nf(" - Interactions      ... ", v=4)

        _require_component_instances(
            self.interaction, Interaction, "Interaction"
        )
        if any(
            type(interaction)
            in (efield.ElectricField, efield_grad.ElectricFieldGrad)
            for interaction in self.interaction
        ):
            raise ValueError(
                "Electric fields are evaluation inputs. Pass field= or "
                "field_grad= to System.singlepoint instead of storing an "
                "exact field interaction on Model."
            )

        es2 = (
            new_es2(unique, par, **dd) if not {"all", "es2"} & exclude else None
        )
        aes2 = (
            new_aes2(unique, par, **dd)
            if not {"all", "aes2"} & exclude
            else None
        )
        es3 = (
            new_es3(unique, par, **dd) if not {"all", "es3"} & exclude else None
        )
        d4sc = (
            new_d4sc(numbers, par, **dd)
            if not {"all", "d4sc", "disp"} & exclude
            else None
        )
        auto_interaction_ids = {
            id(inter): token
            for inter, token in (
                (es2, "es2"),
                (aes2, "aes2"),
                (es3, "es3"),
                (d4sc, "d4sc"),
            )
            if inter is not None
        }

        interactions = InteractionList(
            es2, aes2, es3, d4sc, *self.interaction, **dd
        )
        _require_component_instances(
            interactions.components, Interaction, "Interaction"
        )
        if batch_mode == 0:
            interaction_labels_by_name: dict[str, Interaction] = {}
            for interaction in interactions.components:
                previous_inter = interaction_labels_by_name.get(
                    interaction.label
                )
                if (
                    previous_inter is not None
                    and previous_inter is not interaction
                ):
                    raise ValueError(
                        "Distinct interactions in a single System share the "
                        f"label '{interaction.label}'."
                    )
                interaction_labels_by_name[interaction.label] = interaction

        # Preserve automatic integral promotion for field-labelled subclasses
        # and custom multipole interactions.
        if any(
            interaction.label == efield.LABEL_EFIELD
            and type(interaction) is not efield.ElectricField
            for interaction in interactions.components
        ):
            config = _with_int_level(
                config, max(labels.INTLEVEL_DIPOLE, config.ints.level)
            )
        if any(
            interaction.label == efield_grad.LABEL_EFIELD_GRAD
            and type(interaction) is not efield_grad.ElectricFieldGrad
            for interaction in interactions.components
        ):
            config = _with_int_level(
                config, max(labels.INTLEVEL_QUADRUPOLE, config.ints.level)
            )
        if self.auto_int_level:
            custom_int_level = max(
                (
                    _custom_interaction_required_int_level(interaction)
                    for interaction in interactions.components
                    if type(interaction) not in BUILTIN_INTERACTION_TYPES
                ),
                default=0,
            )
            if custom_int_level > config.ints.level:
                config = _with_int_level(config, custom_int_level)
        es2_interaction = next(
            (i for i in interactions.components if type(i) is ES2), None
        )
        es3_interaction = next(
            (i for i in interactions.components if type(i) is ES3), None
        )
        aes2_interaction = next(
            (i for i in interactions.components if type(i) is AES2), None
        )
        d4sc_interaction = next(
            (i for i in interactions.components if type(i) is DispersionD4SC),
            None,
        )
        generalized_born_interaction = next(
            (i for i in interactions.components if type(i) is GeneralizedBorn),
            None,
        )
        es2_setup = (
            core_es2_setup(
                setup_es2(
                    numbers,
                    es2_interaction.hubbard,
                    ihelp,
                    lhubbard=es2_interaction.lhubbard,
                    gexp=es2_interaction.gexp,
                    average=es2_interaction.average,
                    shell_resolved=es2_interaction.shell_resolved,
                )
            )
            if es2_interaction is not None and batch_mode == 0
            else None
        )
        es3_setup = (
            core_es3_setup(
                setup_es3(
                    es3_interaction.hubbard_derivs,
                    ihelp,
                    shell_scale=es3_interaction.shell_scale,
                )
            )
            if es3_interaction is not None and batch_mode == 0
            else None
        )
        aes2_setup = (
            core_aes2_setup(setup_aes2(aes2_interaction, numbers, ihelp))
            if aes2_interaction is not None and batch_mode == 0
            else None
        )
        d4sc_setup = (
            core_d4sc_setup(setup_d4sc(d4sc_interaction, numbers))
            if d4sc_interaction is not None and batch_mode == 0
            else None
        )
        generalized_born_setup = (
            core_generalized_born_setup(
                setup_generalized_born(generalized_born_interaction, numbers)
            )
            if generalized_born_interaction is not None and batch_mode == 0
            else None
        )

        OutputHandler.write_stdout("done", v=4)

        ##############
        # CLASSICALS #
        ##############

        OutputHandler.write_stdout_nf(" - Classicals        ... ", v=4)

        _require_component_instances(self.classical, Classical, "Classical")
        halogen = (
            new_halogen(unique, par, **dd)
            if not {"all", "hal"} & exclude
            else None
        )
        ies = (
            new_ies(numbers, par, **dd)
            if not {"all", "ies"} & exclude
            else None
        )
        dispersion = (
            new_dispersion(
                numbers,
                par,
                charge=torch.tensor(defaults.CHRG, **dd),
                **dd,
            )
            if not {"all", "disp"} & exclude
            else None
        )
        repulsion = (
            new_repulsion(unique, par, **dd)
            if not {"all", "rep"} & exclude
            else None
        )
        srb = (
            new_srb(unique, par, **dd) if not {"all", "srb"} & exclude else None
        )
        auto_classical_ids = {
            id(comp): token
            for comp, token in (
                (halogen, "hal"),
                (ies, "ies"),
                (dispersion, "disp"),
                (repulsion, "rep"),
                (srb, "srb"),
            )
            if comp is not None
        }

        classicals = ClassicalList(
            halogen, ies, dispersion, repulsion, srb, *self.classical, **dd
        )
        _require_component_instances(
            classicals.components, Classical, "Classical"
        )

        if batch_mode == 0:
            labels_by_name: dict[str, Classical] = {}
            for component in classicals.components:
                previous = labels_by_name.get(component.label)
                if previous is not None and previous is not component:
                    raise ValueError(
                        "Distinct classical contributions in a single System "
                        f"share the label '{component.label}'."
                    )
                labels_by_name[component.label] = component

            # These family-owned System fields intentionally represent one
            # setup per family. D3 is the exception: it has an identity-keyed
            # sequence of setups. Reject multiple exact instances of the
            # other migrated families instead of allowing the later setup to
            # replace the earlier one.
            single_setup_types = (
                Repulsion,
                ShortRangeBond,
                IES,
                Halogen,
                DispersionD4,
            )
            for component_type in single_setup_types:
                matches = [
                    component
                    for component in classicals.components
                    if type(component) is component_type
                ]
                if len(matches) > 1:
                    raise ValueError(
                        "A single System supports at most one exact "
                        f"{component_type.__name__} component; found "
                        f"{len(matches)} with distinct labels."
                    )

        # The single-System core captures each built-in and tensor-only custom
        # setup directly; custom and packed setups are also retained in
        # ``packed_classical_setups``.
        packed_classical_setups = ClassicalListCache()
        custom_classical_snapshots: list[
            tuple[Classical, ComponentMutationSnapshot, Any]
        ] = []
        d3_setups: list[tuple[DispersionD3, DispersionD3CoreSetup]] = []
        repulsion_classical = next(
            (c for c in classicals.components if type(c) is Repulsion), None
        )
        repulsion_setup = None
        srb_classical = next(
            (c for c in classicals.components if type(c) is ShortRangeBond),
            None,
        )
        srb_setup = None
        d4_classical = next(
            (c for c in classicals.components if type(c) is DispersionD4),
            None,
        )
        d4_setup = None
        ies_classical = next(
            (c for c in classicals.components if type(c) is IES), None
        )
        ies_setup = None
        halogen_classical = next(
            (c for c in classicals.components if type(c) is Halogen), None
        )
        halogen_setup = None

        classical_terms: list[ClassicalTermDescriptor] = []
        if batch_mode == 0:
            for position, component in enumerate(classicals.components):
                timer.start(component.label, parent_uid="Classicals")
                component_setup = component.get_cache(numbers, ihelp)
                setup: CoreSetup | None
                if type(component) is DispersionD3:
                    assert isinstance(component_setup, DispersionD3Cache)
                    setup = core_dispersion_d3_setup(
                        component_setup,
                        numbers,
                        component.numbers,
                        component.param,
                        snapshot=True,
                    )
                    d3_setups.append((component, setup))
                elif type(component) is Repulsion:
                    assert isinstance(component_setup, RepulsionSetup)
                    setup = core_repulsion_setup(component_setup)
                    repulsion_setup = setup
                elif type(component) is ShortRangeBond:
                    assert isinstance(component_setup, ShortRangeBondSetup)
                    setup = core_srb_setup(component_setup)
                    srb_setup = setup
                elif type(component) is DispersionD4:
                    assert isinstance(component_setup, DispersionD4Setup)
                    setup = core_dispersion_d4_setup(component_setup)
                    d4_setup = setup
                elif type(component) is IES:
                    assert isinstance(component_setup, IESSetup)
                    setup = core_ies_setup(component_setup)
                    ies_setup = setup
                elif type(component) is Halogen:
                    assert isinstance(component_setup, HalogenSetup)
                    setup = core_halogen_setup(component_setup)
                    halogen_setup = setup
                else:
                    packed_classical_setups[component.label] = component_setup
                    comp_snap = snapshot_component_state(component)
                    custom_classical_snapshots.append(
                        (
                            component,
                            comp_snap,
                            component_setup,
                        )
                    )
                    setup = (
                        build_custom_classical_core_setup(
                            component,
                            component_setup,
                            dd,
                            mutation_snapshot=comp_snap,
                        )
                        if can_build_custom_classical_core_setup(
                            component, component_setup, dd
                        )
                        else None
                    )
                timer.stop(component.label)

                kind = _classical_term_kind(component, setup)
                if kind is not None and setup is not None:
                    classical_terms.append(
                        ClassicalTermDescriptor(
                            kind=kind[0],
                            label=component.label,
                            position=position,
                            setup=setup,
                            charge_dependent=kind[1],
                            geometry_inputs=("positions",),
                        )
                    )
        else:
            for component in classicals.components:
                timer.start(component.label, parent_uid="Classicals")
                batched_setup = component.get_cache(numbers, ihelp)
                if component is repulsion_classical:
                    repulsion_setup = batched_setup
                elif component is srb_classical:
                    srb_setup = batched_setup
                elif component is d4_classical:
                    d4_setup = batched_setup
                elif component is ies_classical:
                    ies_setup = batched_setup
                elif component is halogen_classical:
                    halogen_setup = batched_setup
                else:
                    packed_classical_setups[component.label] = batched_setup
                    if type(component) not in BUILTIN_CLASSICAL_TYPES:
                        custom_classical_snapshots.append(
                            (
                                component,
                                snapshot_component_state(component),
                                batched_setup,
                            )
                        )
                timer.stop(component.label)

        OutputHandler.write_stdout("done", v=4)

        #############
        # INTEGRALS #
        #############

        OutputHandler.write_stdout_nf(" - Integrals         ... ", v=4)

        h0_setup = None
        integral_setup = None
        if config.ints.level >= labels.INTLEVEL_OVERLAP:
            h0_setup = (
                setup_h0_core(numbers, par, ihelp, **dd)
                if batch_mode == 0
                else setup_h0(numbers, par, ihelp, **dd)
            )
            if batch_mode == 0:
                integral_setup = setup_integral_evaluation(
                    numbers,
                    par,
                    ihelp,
                    driver_type=config.ints.driver,
                    intlevel=config.ints.level,
                    algorithm=config.ints.algorithm,
                    device=dd["device"],
                    dtype=dd["dtype"],
                )

        OutputHandler.write_stdout("done\n", v=4)

        exact_core = None
        if batch_mode == 0:
            interaction_terms: list[InteractionTermDescriptor] = []
            for position, interaction in enumerate(interactions.components):
                if interaction is es2_interaction:
                    setup = es2_setup
                elif interaction is es3_interaction:
                    setup = es3_setup
                elif interaction is aes2_interaction:
                    setup = aes2_setup
                elif interaction is d4sc_interaction:
                    setup = d4sc_setup
                elif interaction is generalized_born_interaction:
                    setup = generalized_born_setup
                else:
                    setup = None
                if setup is not None:
                    kind = _interaction_term_kind(interaction, setup)
                    if kind is None:
                        raise RuntimeError(
                            "Exact interaction setup has no built-in descriptor."
                        )
                    interaction_terms.append(
                        InteractionTermDescriptor(
                            kind=kind,
                            label=interaction.label,
                            position=position,
                            setup=setup,
                            geometry_inputs=("positions",),
                        )
                    )

            exact_core = ExactSystemCore(
                numbers=numbers,
                index=ihelp,
                effective_dtype=dd["dtype"],
                config=config,
                integral=integral_setup,
                h0_setup=(
                    h0_setup if isinstance(h0_setup, H0SetupNode) else None
                ),
                classical_terms=tuple(classical_terms),
                interaction_terms=tuple(interaction_terms),
            )

        setup_index_token = _build_setup_index_token(
            numbers, ihelp, integral_setup, h0_setup, es2_setup
        )

        return System(
            numbers=numbers,
            config=config,
            batch_mode=batch_mode,
            ihelp=ihelp,
            classicals=classicals,
            interactions=interactions,
            integral_setup=integral_setup,
            h0_setup=h0_setup,
            _packed_classical_setups=packed_classical_setups,
            repulsion_setup=repulsion_setup,
            repulsion_classical=repulsion_classical,
            srb_setup=srb_setup,
            srb_classical=srb_classical,
            d4_setup=d4_setup,
            d4_classical=d4_classical,
            ies_setup=ies_setup,
            ies_classical=ies_classical,
            halogen_setup=halogen_setup,
            halogen_classical=halogen_classical,
            d3_setups=tuple(d3_setups),
            es2_setup=es2_setup,
            es2_interaction=es2_interaction,
            es3_setup=es3_setup,
            es3_interaction=es3_interaction,
            aes2_setup=aes2_setup,
            aes2_interaction=aes2_interaction,
            d4sc_setup=d4sc_setup,
            d4sc_interaction=d4sc_interaction,
            generalized_born_setup=generalized_born_setup,
            generalized_born_interaction=generalized_born_interaction,
            dd=dd,
            exact_core=exact_core,
            _setup_index_token=setup_index_token,
            _custom_classical_snapshots=tuple(custom_classical_snapshots),
            _auto_classical_ids=auto_classical_ids,
            _auto_interaction_ids=auto_interaction_ids,
        )


def _with_int_level(config: Config, level: int) -> Config:
    """Configuration with a different integral level."""
    return replace(config, ints=replace(config.ints, level=level))
