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
Gradients flow from everything computed in the system to the parameter
leaves, i.e., it can be called inside a differentiated function.

.. note::

    Component lists and classical setup remain transitional mutable objects
    until later component/cache packages. Integral matrix production uses
    explicit setup values and does not store per-call outputs on System.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from tad_mctc.tree import Node, child, context

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
from dxtb._src.integral.driver.libcint.driver import LibcintIntegralSetupNode
from dxtb._src.integral.driver.pytorch.setup import PytorchIntegralSetupNode
from dxtb._src.integral.evaluation import (
    IntegralSetup,
    setup_integral_evaluation,
)
from dxtb._src.param import ParamModule
from dxtb._src.timing import timer
from dxtb._src.typing import DD, Tensor
from dxtb._src.xtb.h0 import H0Setup, H0SetupNode, setup_h0, setup_h0_core
from dxtb.config import Config

if TYPE_CHECKING:
    from dxtb._src.calculators.result import Result
    from dxtb._src.scf.functional.core import CoreIndexData

__all__ = ["Model", "System"]


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


@dataclass(frozen=True, eq=False)
class CoreSetupRef:
    """Compatibility association for validating one captured setup."""

    component_identity: int
    label: str
    position: int
    setup: CoreSetup


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


@dataclass(frozen=True, eq=False)
class ExactSystemCore:
    """Immutable exact single-System values shared with the compatibility façade.

    Setup objects are references to the already captured family Nodes. This
    record introduces no second tensor storage and intentionally excludes
    packed-mode selectors, mutable dtype dictionaries, IndexHelper, and
    mutable component containers.
    """

    numbers: Tensor
    effective_dtype: torch.dtype
    config: Config
    integral_setup: IntegralSetup | None
    h0_setup: H0SetupNode | None
    index: CoreIndexData | None
    classical_terms: tuple[ClassicalTermDescriptor, ...]
    interaction_terms: tuple[InteractionTermDescriptor, ...]

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


@dataclass(frozen=True, eq=False)
class ExactSystemCompatibility:
    """Identity associations used only at the mutable System boundary."""

    index_helper_identity: int
    classical_setups: tuple[CoreSetupRef, ...]
    interaction_setups: tuple[CoreSetupRef, ...]


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
    expected: list[ClassicalTermDescriptor],
) -> bool:
    """Compare immutable descriptors by metadata and setup identity."""
    return len(captured) == len(expected) and all(
        left.kind == right.kind
        and left.label == right.label
        and left.position == right.position
        and left.setup is right.setup
        and left.charge_dependent == right.charge_dependent
        and left.geometry_inputs == right.geometry_inputs
        for left, right in zip(captured, expected)
    )


def _interaction_terms_match(
    captured: tuple[InteractionTermDescriptor, ...],
    expected: list[InteractionTermDescriptor],
) -> bool:
    """Compare immutable descriptors by metadata and setup identity."""
    return len(captured) == len(expected) and all(
        left.kind == right.kind
        and left.label == right.label
        and left.position == right.position
        and left.setup is right.setup
        and left.geometry_inputs == right.geometry_inputs
        for left, right in zip(captured, expected)
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
    Batch mode (0: single system, 1: padded batch, 2: conformers without
    padding). The default follows the shape of the atomic numbers.
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

    _classical_extension_cache: ClassicalListCache = field(repr=False)
    """Setup-time compatibility data for custom classical extensions only."""

    _packed_classical_cache: ClassicalListCache | None = field(repr=False)
    """Full legacy cache retained only for packed Calculator systems."""

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
    """Device and data type of the tensors of the system."""

    d3_setups: tuple[tuple[DispersionD3, DispersionD3CoreSetup], ...] = ()
    """Captured setup for each exact single-System classical D3 component."""

    exact_core: ExactSystemCore | None = field(default=None, repr=False)
    """Exact unbatched values; None for packed compatibility Systems."""

    _exact_compatibility: ExactSystemCompatibility | None = field(
        default=None, repr=False
    )
    """Identity tokens kept outside the future stackable core value."""

    def singlepoint(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        field: Tensor | None = None,
        field_grad: Tensor | None = None,
    ) -> Result:
        """Evaluate this single system through the pure calculator core."""
        from dxtb._src.calculators.singlepoint import singlepoint

        return singlepoint(self, positions, chrg, spin, field, field_grad)

    @property
    def d3_setup(self) -> DispersionD3CoreSetup | None:
        """Return the sole exact D3 setup when the System has one."""
        return self.d3_setups[0][1] if len(self.d3_setups) == 1 else None

    def classical_setup(self, component: Classical) -> CoreSetup | None:
        """Compatibility lookup isolated from exact numerical dispatch."""
        return self._compat_setup(
            (
                self._exact_compatibility.classical_setups
                if self._exact_compatibility is not None
                else ()
            ),
            component,
        )

    def interaction_setup(self, component: Interaction) -> CoreSetup | None:
        """Compatibility lookup isolated from exact numerical dispatch."""
        return self._compat_setup(
            (
                self._exact_compatibility.interaction_setups
                if self._exact_compatibility is not None
                else ()
            ),
            component,
        )

    @staticmethod
    def _compat_setup(
        entries: tuple[CoreSetupRef, ...], component: Classical | Interaction
    ) -> CoreSetup | None:
        identity = id(component)
        for entry in entries:
            if entry.component_identity == identity:
                return entry.setup
        return None

    def _active_exact_core(self) -> ExactSystemCore | None:
        """Return the captured core while its façade setup references agree."""
        core = self.exact_core
        compatibility = self._exact_compatibility
        if (
            core is None
            or compatibility is None
            or core.numbers is not self.numbers
            or core.config is not self.config
            or core.integral_setup is not self.integral_setup
            or core.h0_setup is not self.h0_setup
            or compatibility.index_helper_identity != id(self.ihelp)
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

        expected_classical: list[tuple[int, int, str, CoreSetup]] = []
        expected_classical_terms: list[ClassicalTermDescriptor] = []
        for position, component in enumerate(self.classicals.components):
            setup = None
            if component is self.repulsion_classical:
                setup = self.repulsion_setup
            elif component is self.srb_classical:
                setup = self.srb_setup
            elif component is self.d4_classical:
                setup = self.d4_setup
            elif component is self.ies_classical:
                setup = self.ies_setup
            elif component is self.halogen_classical:
                setup = self.halogen_setup
            else:
                setup = next(
                    (
                        value
                        for owner, value in self.d3_setups
                        if owner is component
                    ),
                    None,
                )
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
                expected_classical.append(
                    (position, id(component), component.label, setup)
                )
                kind = _classical_term_kind(component, setup)
                if kind is None:
                    return None
                expected_classical_terms.append(
                    ClassicalTermDescriptor(
                        kind=kind[0],
                        label=component.label,
                        position=position,
                        setup=setup,
                        charge_dependent=kind[1],
                        geometry_inputs=("positions",),
                    )
                )

        expected_interactions: list[tuple[int, int, str, CoreSetup]] = []
        expected_interaction_terms: list[InteractionTermDescriptor] = []
        for position, interaction in enumerate(self.interactions.components):
            if interaction is self.es2_interaction:
                setup = self.es2_setup
            elif interaction is self.es3_interaction:
                setup = self.es3_setup
            elif interaction is self.aes2_interaction:
                setup = self.aes2_setup
            elif interaction is self.d4sc_interaction:
                setup = self.d4sc_setup
            elif interaction is self.generalized_born_interaction:
                setup = self.generalized_born_setup
            else:
                setup = None
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
                expected_interactions.append(
                    (position, id(interaction), interaction.label, setup)
                )
                kind = _interaction_term_kind(interaction, setup)
                if kind is None:
                    return None
                expected_interaction_terms.append(
                    InteractionTermDescriptor(
                        kind=kind,
                        label=interaction.label,
                        position=position,
                        setup=setup,
                        geometry_inputs=("positions",),
                    )
                )

        if not self._setup_refs_match(
            compatibility.classical_setups, expected_classical
        ):
            return None
        if not self._setup_refs_match(
            compatibility.interaction_setups, expected_interactions
        ):
            return None
        if not _classical_terms_match(
            core.classical_terms, expected_classical_terms
        ):
            return None
        if not _interaction_terms_match(
            core.interaction_terms, expected_interaction_terms
        ):
            return None

        backend_setup = (
            self.integral_setup.pytorch
            if self.integral_setup is not None
            and self.integral_setup.pytorch is not None
            else (
                self.integral_setup.libcint
                if self.integral_setup is not None
                else None
            )
        )
        expected_index = (
            backend_setup.index
            if type(backend_setup)
            in (PytorchIntegralSetupNode, LibcintIntegralSetupNode)
            else None
        )
        if core.index is not expected_index:
            return None
        return core

    @staticmethod
    def _setup_refs_match(
        captured: tuple[CoreSetupRef, ...],
        expected: list[tuple[int, int, str, CoreSetup]],
    ) -> bool:
        """Check ordered setup references without tensor equality/sync."""
        if len(captured) != len(expected):
            return False
        return all(
            entry.position == position
            and entry.component_identity == component_identity
            and entry.label == label
            and entry.setup is setup
            for entry, (position, component_identity, label, setup) in zip(
                captured, expected
            )
        )

    @property
    def classical_cache(self) -> ClassicalListCache:
        """Build a compatibility view over captured setup values on demand.

        Single-System core setup Nodes remain the sole persistent owners for
        exact built-in terms. Custom extension setup data is retained in a
        separate compatibility cache. Packed systems keep their established
        complete legacy cache.
        """
        if self.batch_mode != 0:
            if self._packed_classical_cache is None:
                return ClassicalListCache()
            return self._packed_classical_cache

        cache = ClassicalListCache()
        for component in self.classicals.components:
            if component is self.repulsion_classical:
                setup = self.repulsion_setup
            elif component is self.srb_classical:
                setup = self.srb_setup
            elif component is self.d4_classical:
                setup = self.d4_setup
            elif component is self.ies_classical:
                setup = self.ies_setup
            elif component is self.halogen_classical:
                setup = self.halogen_setup
            else:
                setup = next(
                    (
                        captured
                        for d3_component, captured in self.d3_setups
                        if component is d3_component
                    ),
                    None,
                )
                if setup is None:
                    if component.label not in self._classical_extension_cache:
                        continue
                    setup = self._classical_extension_cache[component.label]

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
    """Additional classical contributions (instances for a given system)."""

    interaction: tuple[Interaction, ...] = ()
    """Additional self-consistent contributions (instances for a given system)."""

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

    def setup(
        self,
        numbers: Tensor,
        *,
        batch_mode: int | None = None,
        dd: DD | None = None,
    ) -> System:
        """
        Set up the system for the atomic numbers.

        The result is a function of the parameters and the numbers, i.e.,
        everything that is computed here (including the gathering of the
        element parameters) can be differentiated with respect to the
        parameters.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers (shape: ``(..., nat)``).
        batch_mode : int | None, optional
            Batch mode. The default follows the shape of `numbers` (0 for a
            single system, 1 for a padded batch). Mode 2 (conformers without
            padding) has to be requested.
        dd : DD | None, optional
            Device and data type of the tensors. Default: the ones of the
            parameters.

        Returns
        -------
        System
            Data of the system that depends on `numbers` only.
        """
        par = self.par
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
        # Exact field adapters are removed by the Calculator before setup.
        # Preserve automatic integral promotion for legacy field subclasses.
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

        # The single-System core captures each exact built-in setup directly.
        # ClassicalList.get_cache remains the compatibility boundary for
        # packed systems; custom components retain their legacy setup hooks.
        classical_extension_cache = ClassicalListCache()
        packed_classical_cache = None
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

        if batch_mode == 0:
            for component in classicals.components:
                timer.start(component.label, parent_uid="Classicals")
                if type(component) is DispersionD3:
                    legacy_setup = component.get_cache(numbers, ihelp)
                    assert isinstance(legacy_setup, DispersionD3Cache)
                    setup = core_dispersion_d3_setup(
                        legacy_setup,
                        numbers,
                        component.numbers,
                        component.param,
                        snapshot=True,
                    )
                    d3_setups.append((component, setup))
                elif type(component) is Repulsion:
                    legacy_setup = component.get_cache(numbers, ihelp)
                    assert isinstance(legacy_setup, RepulsionSetup)
                    setup = core_repulsion_setup(legacy_setup)
                    repulsion_setup = setup
                elif type(component) is ShortRangeBond:
                    legacy_setup = component.get_cache(numbers, ihelp)
                    assert isinstance(legacy_setup, ShortRangeBondSetup)
                    setup = core_srb_setup(legacy_setup)
                    srb_setup = setup
                elif type(component) is DispersionD4:
                    legacy_setup = component.get_cache(numbers, ihelp)
                    assert isinstance(legacy_setup, DispersionD4Setup)
                    setup = core_dispersion_d4_setup(legacy_setup)
                    d4_setup = setup
                elif type(component) is IES:
                    legacy_setup = component.get_cache(numbers, ihelp)
                    assert isinstance(legacy_setup, IESSetup)
                    setup = core_ies_setup(legacy_setup)
                    ies_setup = setup
                elif type(component) is Halogen:
                    legacy_setup = component.get_cache(numbers, ihelp)
                    assert isinstance(legacy_setup, HalogenSetup)
                    setup = core_halogen_setup(legacy_setup)
                    halogen_setup = setup
                else:
                    setup = component.get_cache(numbers, ihelp)
                if type(component) not in (
                    DispersionD3,
                    Repulsion,
                    ShortRangeBond,
                    DispersionD4,
                    IES,
                    Halogen,
                ):
                    classical_extension_cache[component.label] = setup
                timer.stop(component.label)
        else:
            packed_classical_cache = classicals.get_cache(numbers, ihelp)
            if repulsion_classical is not None:
                repulsion_setup = packed_classical_cache[
                    repulsion_classical.label
                ]
            if srb_classical is not None:
                srb_setup = packed_classical_cache[srb_classical.label]
            if d4_classical is not None:
                d4_setup = packed_classical_cache[d4_classical.label]
            if ies_classical is not None:
                ies_setup = packed_classical_cache[ies_classical.label]
            if halogen_classical is not None:
                halogen_setup = packed_classical_cache[halogen_classical.label]

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
            # B4's pure evaluation core handles one system. Keep existing
            # Calculator batching on its legacy adapter until the later
            # batching package supplies stacked-System/vmap evaluation.
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
        exact_compatibility = None
        if batch_mode == 0:
            classical_setup_refs: list[CoreSetupRef] = []
            classical_terms: list[ClassicalTermDescriptor] = []
            for position, component in enumerate(classicals.components):
                if type(component) is DispersionD3:
                    setup = next(
                        (
                            captured
                            for owner, captured in d3_setups
                            if owner is component
                        ),
                        None,
                    )
                elif component is repulsion_classical:
                    setup = repulsion_setup
                elif component is srb_classical:
                    setup = srb_setup
                elif component is d4_classical:
                    setup = d4_setup
                elif component is ies_classical:
                    setup = ies_setup
                elif component is halogen_classical:
                    setup = halogen_setup
                else:
                    setup = None
                if setup is not None and isinstance(
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
                    classical_setup_refs.append(
                        CoreSetupRef(
                            id(component), component.label, position, setup
                        )
                    )
                    kind = _classical_term_kind(component, setup)
                    if kind is None:
                        raise RuntimeError(
                            "Exact classical setup has no built-in descriptor."
                        )
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

            interaction_setup_refs: list[CoreSetupRef] = []
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
                    interaction_setup_refs.append(
                        CoreSetupRef(
                            id(interaction), interaction.label, position, setup
                        )
                    )
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

            backend_setup = (
                integral_setup.pytorch
                if integral_setup is not None
                and integral_setup.pytorch is not None
                else (
                    integral_setup.libcint
                    if integral_setup is not None
                    else None
                )
            )
            core_index = (
                backend_setup.index
                if type(backend_setup)
                in (PytorchIntegralSetupNode, LibcintIntegralSetupNode)
                else None
            )
            exact_core = ExactSystemCore(
                numbers=numbers,
                effective_dtype=dd["dtype"],
                config=config,
                integral_setup=integral_setup,
                h0_setup=(
                    h0_setup if isinstance(h0_setup, H0SetupNode) else None
                ),
                index=core_index,
                classical_terms=tuple(classical_terms),
                interaction_terms=tuple(interaction_terms),
            )
            exact_compatibility = ExactSystemCompatibility(
                index_helper_identity=id(ihelp),
                classical_setups=tuple(classical_setup_refs),
                interaction_setups=tuple(interaction_setup_refs),
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
            _classical_extension_cache=classical_extension_cache,
            _packed_classical_cache=packed_classical_cache,
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
            _exact_compatibility=exact_compatibility,
        )


def _with_int_level(config: Config, level: int) -> Config:
    """Configuration with a different integral level."""
    return replace(config, ints=replace(config.ints, level=level))
