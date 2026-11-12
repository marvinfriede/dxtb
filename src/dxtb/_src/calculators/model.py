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

from dataclasses import dataclass, field, replace
from functools import wraps
from threading import Lock, RLock
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable
from weakref import WeakKeyDictionary

import torch
from tad_mctc.tree import ModuleNode, Node, child, context
from torch import nn

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
from dxtb._src.integral.driver.libcint.driver import LibcintIntegralSetup
from dxtb._src.integral.driver.pytorch.setup import PytorchIntegralSetup
from dxtb._src.integral.evaluation import (
    IntegralSetup,
    setup_integral_evaluation,
)
from dxtb._src.param import ParamModule
from dxtb._src.timing import timer
from dxtb._src.typing import DD, Tensor
from dxtb._src.utils.tensors import normalize_device
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
"""The Classical Components a Model can set up."""

BUILTIN_INTERACTION_TYPES = (AES2, ES2, ES3, DispersionD4SC, GeneralizedBorn)
"""The Interactions a Model can set up (External fields are call inputs)."""


def _require_builtin_components(
    components: Sequence[Any], builtin: tuple[type, ...], kind: str
) -> None:
    """Reject Components whose exact type is not a built-in one."""
    custom = [c for c in components if type(c) not in builtin]
    if custom:
        names = ", ".join(
            f"{type(c).__module__}.{type(c).__qualname__}" for c in custom
        )
        supported = ", ".join(t.__name__ for t in builtin)
        raise TypeError(
            f"Only the built-in {kind} Components ({supported}) are "
            f"supported; got {names}."
        )


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


CoreSetup = (
    RepulsionCoreSetup
    | ShortRangeBondCoreSetup
    | DispersionD3CoreSetup
    | DispersionD4CoreSetup
    | IESCoreSetup
    | HalogenCoreSetup
    | ES2CoreSetup
    | ES3CoreSetup
    | AES2CoreSetup
    | D4SCCoreSetup
    | GeneralizedBornCoreSetup
)


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

    The selected backend and all family setup Nodes are dynamic children.
    ``index`` and ``integral_setup`` are views derived from
    the backend, so the backend index is not registered a second time.
    """

    numbers: Tensor = child(keep_dtype=True)
    effective_dtype: torch.dtype = context()
    config: Config = context()
    integral: IntegralSetup | None = child()
    h0_setup: H0SetupNode | None = child()
    classical_terms: tuple[ClassicalTermDescriptor, ...] = child()
    interaction_terms: tuple[InteractionTermDescriptor, ...] = child()

    @property
    def index(self) -> IndexHelper | None:
        """Return the index owned by the selected backend, if present."""
        return None if self.integral is None else self.integral.index

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
    """Describe only exact built-ins whose setup is already captured."""
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


@dataclass(frozen=True, kw_only=True, eq=False)
class _SetupIndexToken:
    """Setup-time ownership and tensor-version snapshot for ``System.ihelp``."""

    numbers: Tensor
    numbers_snapshot: Tensor
    ihelp: IndexHelper
    backend_index: IndexHelper | None
    es2_shells_to_atom: Tensor | None
    tensor_states: tuple[
        tuple[Tensor, int | None, torch.Size, torch.dtype, torch.device], ...
    ]
    unique_angular_snapshot: Tensor
    ushells_per_unique_snapshot: Tensor


def _build_setup_index_token(
    numbers: Tensor,
    ihelp: IndexHelper,
    integral_setup: IntegralSetup | None,
    es2_setup: ES2CoreSetup | None,
) -> _SetupIndexToken:
    """Capture setup-time identity and version metadata for fast validation."""
    backend_index = None if integral_setup is None else integral_setup.index

    es2_shells = (
        es2_setup.shells_to_atom if es2_setup is not None else None
    )

    tracked: list[Tensor] = [
        numbers,
        *(getattr(ihelp, name) for name in ihelp._child_names),
    ]
    if backend_index is not None and backend_index is not ihelp:
        tracked.extend(
            getattr(backend_index, name) for name in backend_index._child_names
        )
    if es2_shells is not None and es2_shells is not ihelp.shells_to_atom:
        tracked.append(es2_shells)

    return _SetupIndexToken(
        numbers=numbers,
        numbers_snapshot=numbers.detach().clone(),
        ihelp=ihelp,
        backend_index=backend_index,
        es2_shells_to_atom=es2_shells,
        tensor_states=tuple(
            (
                t,
                None if t.is_inference() else t._version,
                t.shape,
                t.dtype,
                t.device,
            )
            for t in tracked
        ),
        unique_angular_snapshot=ihelp.unique_angular.detach().clone(),
        ushells_per_unique_snapshot=ihelp.ushells_per_unique.detach().clone(),
    )


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

    def _ihelp_compatible(self) -> bool:
        """Return whether ``ihelp`` matches this System's structural setup."""
        ihelp = self.ihelp
        if (
            not isinstance(ihelp, IndexHelper)
            or ihelp.batch_mode != self.batch_mode
            or ihelp.device != self.numbers.device
        ):
            return False
        if self.batch_mode != 0:
            return ihelp.atom_to_unique.shape == self.numbers.shape

        if self.numbers.ndim != 1:
            return False
        nat = self.numbers.shape[0]
        if (
            ihelp.atom_to_unique.shape != (nat,)
            or ihelp.shells_per_atom.shape != (nat,)
            or ihelp.shell_index.shape != (nat,)
            or ihelp.unique_angular.ndim != 1
            or ihelp.ushells_to_unique.shape != ihelp.unique_angular.shape
            or ihelp.ushells_per_unique.ndim != 1
            or ihelp.shells_to_atom.ndim != 1
        ):
            return False

        n_shells = ihelp.shells_to_atom.shape[0]
        if (
            ihelp.angular.shape != (n_shells,)
            or ihelp.shells_to_ushell.shape != (n_shells,)
            or ihelp.orbitals_per_shell.shape != (n_shells,)
            or ihelp.orbital_index.shape != (n_shells,)
            or ihelp.orbitals_to_shell.ndim != 1
        ):
            return False
        n_orbitals = ihelp.orbitals_to_shell.shape[0]

        if self.h0_setup is not None and (
            self.h0_setup.numbers.shape != (nat,)
            or self.h0_setup.selfenergy.shape != ihelp.unique_angular.shape
            or self.h0_setup.shpoly.shape != ihelp.unique_angular.shape
            or self.h0_setup.rad.shape != ihelp.ushells_per_unique.shape
            or self.h0_setup.valence.shape != ihelp.unique_angular.shape
            or self.h0_setup.refocc.shape != ihelp.unique_angular.shape
        ):
            return False

        backend_index = (
            None if self.integral_setup is None else self.integral_setup.index
        )
        if backend_index is not None and (
            backend_index.atom_to_unique.shape != (nat,)
            or backend_index.shells_to_ushell.shape != (n_shells,)
            or backend_index.unique_angular.shape != ihelp.unique_angular.shape
            or backend_index.shells_to_atom.shape != (n_shells,)
            or backend_index.orbitals_to_shell.shape != (n_orbitals,)
        ):
            return False

        if (
            self.es2_setup is not None
            and self.es2_setup.shells_to_atom is not None
            and self.es2_setup.shells_to_atom.shape != (n_shells,)
        ):
            return False
        if self.es3_setup is not None and (
            self.es3_setup.hubbard_derivs.shape[-1]
            != (n_shells if self.es3_setup.shell_resolved else nat)
        ):
            return False

        if (
            self.numbers.device.type == "meta"
            or torch.compiler.is_compiling()
            or torch._C._functorch.is_functorch_wrapped_tensor(self.numbers)
        ):
            return True

        token = self._setup_index_token
        es2_shells = (
            self.es2_setup.shells_to_atom
            if self.es2_setup is not None
            else None
        )
        token_valid = (
            token is not None
            and self.numbers is token.numbers
            and backend_index is token.backend_index
            and es2_shells is token.es2_shells_to_atom
            and all(
                ver is not None
                and not t.is_inference()
                and t._version == ver
                and t.shape == shape
                and t.dtype == dtype
                and t.device == dev
                for t, ver, shape, dtype, dev in token.tensor_states
            )
        )
        if token_valid:
            assert token is not None
            if ihelp is token.ihelp:
                return True
            if token.ihelp.device == ihelp.device and all(
                getattr(ihelp, name).shape == getattr(token.ihelp, name).shape
                and getattr(ihelp, name).dtype
                == getattr(token.ihelp, name).dtype
                for name in ihelp._child_names
            ):
                return torch.equal(
                    torch.cat(
                        [getattr(ihelp, name) for name in ihelp._child_names]
                    ),
                    torch.cat(
                        [
                            getattr(token.ihelp, name)
                            for name in ihelp._child_names
                        ]
                    ),
                )

        if token is not None and (
            not torch.equal(
                self.numbers,
                token.numbers_snapshot.to(
                    device=self.numbers.device, dtype=self.numbers.dtype
                ),
            )
            or not torch.equal(
                ihelp.unique_angular,
                token.unique_angular_snapshot.to(
                    device=ihelp.device, dtype=ihelp.unique_angular.dtype
                ),
            )
            or not torch.equal(
                ihelp.ushells_per_unique,
                token.ushells_per_unique_snapshot.to(
                    device=ihelp.device,
                    dtype=ihelp.ushells_per_unique.dtype,
                ),
            )
        ):
            return False

        if (
            self.h0_setup is not None
            and self.h0_setup.numbers.device == self.numbers.device
            and not torch.equal(
                self.numbers,
                self.h0_setup.numbers.to(dtype=self.numbers.dtype),
            )
        ):
            return False

        if (
            backend_index is not None
            and backend_index.shells_to_atom.device == ihelp.device
            and backend_index is not ihelp
            and not all(
                torch.equal(getattr(ihelp, name), getattr(backend_index, name))
                for name in ihelp._child_names
            )
        ):
            return False

        if (
            self.es2_setup is not None
            and self.es2_setup.shells_to_atom is not None
            and self.es2_setup.shells_to_atom.device == ihelp.device
            and not torch.equal(
                ihelp.shells_to_atom, self.es2_setup.shells_to_atom
            )
        ):
            return False

        unique, expected_atom_to_unique = torch.unique(
            self.numbers, return_inverse=True
        )
        if (
            ihelp.ushells_per_unique.shape != unique.shape
            or not torch.equal(ihelp.atom_to_unique, expected_atom_to_unique)
            or bool((ihelp.ushells_per_unique < 0).any())
        ):
            return False

        device = ihelp.device
        dtype = ihelp.unique_angular.dtype
        expected_ushells_to_unique = torch.repeat_interleave(
            torch.arange(unique.shape[0], device=device, dtype=dtype),
            ihelp.ushells_per_unique,
        )
        if not torch.equal(ihelp.ushells_to_unique, expected_ushells_to_unique):
            return False

        expected_shells_per_atom = ihelp.ushells_per_unique[
            ihelp.atom_to_unique
        ]
        if not torch.equal(ihelp.shells_per_atom, expected_shells_per_atom):
            return False

        expected_shell_index = (
            torch.cumsum(ihelp.shells_per_atom, -1) - ihelp.shells_per_atom
        )
        expected_shell_index = torch.where(
            ihelp.shells_per_atom == 0, -999, expected_shell_index
        )
        if not torch.equal(ihelp.shell_index, expected_shell_index):
            return False

        expected_shells_to_atom = torch.repeat_interleave(
            torch.arange(nat, device=device, dtype=dtype),
            ihelp.shells_per_atom,
        )
        if not torch.equal(ihelp.shells_to_atom, expected_shells_to_atom):
            return False

        ushell_offset = (
            torch.cumsum(ihelp.ushells_per_unique, -1)
            - ihelp.ushells_per_unique
        )
        expected_shells_to_ushell = (
            ushell_offset[ihelp.atom_to_unique[ihelp.shells_to_atom]]
            + torch.arange(n_shells, device=device, dtype=dtype)
            - ihelp.shell_index[ihelp.shells_to_atom]
        )
        if not torch.equal(ihelp.shells_to_ushell, expected_shells_to_ushell):
            return False

        if bool(
            (ihelp.shells_to_ushell >= ihelp.unique_angular.shape[0]).any()
        ):
            return False
        expected_angular = torch.where(
            ihelp.shells_to_ushell >= 0,
            ihelp.unique_angular[ihelp.shells_to_ushell.clamp(min=0)],
            -999,
        )
        if not torch.equal(ihelp.angular, expected_angular):
            return False

        expected_orbitals_per_shell = torch.where(
            ihelp.angular >= 0, 2 * ihelp.angular + 1, 0
        )
        if not torch.equal(
            ihelp.orbitals_per_shell, expected_orbitals_per_shell
        ):
            return False

        expected_orbital_index = (
            torch.cumsum(ihelp.orbitals_per_shell, -1)
            - ihelp.orbitals_per_shell
        )
        expected_orbital_index = torch.where(
            ihelp.orbitals_per_shell == 0, -999, expected_orbital_index
        )
        if not torch.equal(ihelp.orbital_index, expected_orbital_index):
            return False

        expected_orbitals_to_shell = torch.repeat_interleave(
            torch.arange(n_shells, device=device, dtype=dtype),
            ihelp.orbitals_per_shell,
        )
        if not torch.equal(ihelp.orbitals_to_shell, expected_orbitals_to_shell):
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

        expected_index = (
            None if self.integral_setup is None else self.integral_setup.index
        )
        if core.index is not expected_index:
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

        interactions = InteractionList(
            es2, aes2, es3, d4sc, *self.interaction, **dd
        )
        _require_builtin_components(
            interactions.components, BUILTIN_INTERACTION_TYPES, "Interaction"
        )
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

        classicals = ClassicalList(
            halogen, ies, dispersion, repulsion, srb, *self.classical, **dd
        )
        _require_builtin_components(
            classicals.components, BUILTIN_CLASSICAL_TYPES, "Classical"
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

        # The single-System core captures each built-in setup directly. A
        # padded or conformer System keeps the D3 setups here instead.
        packed_classical_setups = ClassicalListCache()
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
                else:
                    assert isinstance(component_setup, HalogenSetup)
                    setup = core_halogen_setup(component_setup)
                    halogen_setup = setup
                timer.stop(component.label)

                kind = _classical_term_kind(component, setup)
                if kind is not None:
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
                setup = component.get_cache(numbers, ihelp)
                if component is repulsion_classical:
                    repulsion_setup = setup
                elif component is srb_classical:
                    srb_setup = setup
                elif component is d4_classical:
                    d4_setup = setup
                elif component is ies_classical:
                    ies_setup = setup
                elif component is halogen_classical:
                    halogen_setup = setup
                else:
                    packed_classical_setups[component.label] = setup
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
                    force_cpu_for_libcint=(
                        config.ints.driver == labels.INTDRIVER_LIBCINT
                    ),
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
                effective_dtype=dd["dtype"],
                config=config,
                integral=integral_setup,
                h0_setup=(
                    h0_setup if isinstance(h0_setup, H0SetupNode) else None
                ),
                classical_terms=tuple(classical_terms),
                interaction_terms=tuple(interaction_terms),
            )

        setup_index_token = (
            _build_setup_index_token(
                numbers, ihelp, integral_setup, es2_setup
            )
            if batch_mode == 0
            else None
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
        )


def _with_int_level(config: Config, level: int) -> Config:
    """Configuration with a different integral level."""
    return replace(config, ints=replace(config.ints, level=level))
