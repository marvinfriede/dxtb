.. _indepth_components_new:

Creating New Components
=======================

``dxtb`` supports both custom **classical** contributions (geometry-dependent,
SCF-charge-independent energy terms) and custom self-consistent **interactions**
(charge- or multipole-dependent terms coupled into the SCF cycle). Custom
components can be passed to :class:`~dxtb.calculators.Model` or
:class:`~dxtb.Calculator`, including direct subclasses of
:class:`~dxtb.components.base.Classical` /
:class:`~dxtb.components.base.Interaction` as well as
subclasses of built-in components.

.. note::

   Uniform external electric fields and electric field gradients are evaluation
   inputs rather than setup-time components: pass ``field=`` or ``field_grad=``
   to :meth:`~dxtb.calculators.System.singlepoint` or
   :meth:`~dxtb.Calculator.singlepoint`.


Custom Classical Components
---------------------------

A custom classical contribution inherits from
:class:`~dxtb.components.base.Classical` and implements:

1. :meth:`~dxtb.components.base.Classical.get_cache`: builds
   composition-dependent setup data from ``numbers`` and
   :class:`~dxtb.IndexHelper` during :meth:`~dxtb.calculators.Model.setup`.
2. :meth:`~dxtb.components.base.Classical.get_energy`: evaluates
   atom-resolved energies of shape ``(..., nat)`` from ``positions`` (shape
   ``(..., nat, 3)``) and the setup cache. Single-system evaluation forwards the
   total molecular charge via ``charge=charge``, so ``get_energy`` should accept
   ``charge: Tensor | None = None`` or ``**kwargs: Any``.

.. code-block:: python

    from typing import Any
    import torch
    from dxtb import Calculator, GFN1_XTB, IndexHelper
    from dxtb.components.base import Classical, ClassicalCache
    from dxtb.typing import DD, Tensor, override


    class HarmonicRestraintCache(ClassicalCache):
        """Composition-dependent setup data for the harmonic restraint."""

        force_constant: Tensor
        __slots__ = ["force_constant"]

        def __init__(
            self,
            force_constant: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(
                device=force_constant.device if device is None else device,
                dtype=force_constant.dtype if dtype is None else dtype,
            )
            self.force_constant = force_constant


    class HarmonicRestraint(Classical):
        """Isotropic harmonic restraint around the origin."""

        force_constant: Tensor
        __slots__ = ["force_constant"]

        def __init__(
            self,
            force_constant: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(
                device=force_constant.device if device is None else device,
                dtype=force_constant.dtype if dtype is None else dtype,
            )
            self.force_constant = force_constant

        @override
        def get_cache(
            self,
            numbers: Tensor,
            ihelp: IndexHelper | None = None,
        ) -> HarmonicRestraintCache:
            """Create composition-dependent setup data."""
            return HarmonicRestraintCache(self.force_constant)

        @override
        def get_energy(
            self,
            positions: Tensor,
            cache: HarmonicRestraintCache,
            **_: Any,
        ) -> Tensor:
            """Return atom-resolved harmonic restraint energies."""
            return 0.5 * cache.force_constant * positions.square().sum(dim=-1)


    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    restraint = HarmonicRestraint(torch.tensor(0.05, **dd), **dd)

    numbers = torch.tensor([3, 1], device=dd["device"])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], **dd)

    calc = Calculator(numbers, GFN1_XTB, classical=(restraint,), **dd)
    result = calc.singlepoint(positions)
    print(result.energy.sum(-1))


Custom Self-Consistent Interactions
-----------------------------------

A custom SCF interaction inherits from
:class:`~dxtb.components.base.Interaction` and implements
:meth:`~dxtb.components.base.Interaction.get_cache` together
with the relevant potential and energy hooks:

- Shell-resolved monopoles: :meth:`get_monopole_shell_potential` and
  :meth:`get_monopole_shell_energy` (shape ``(..., nsh)``).
- Atom-resolved monopoles: :meth:`get_monopole_atom_potential` and
  :meth:`get_monopole_atom_energy` (shape ``(..., nat)``).
- Atomic dipoles: :meth:`get_dipole_atom_potential` (shape ``(..., nat, 3)``)
  and :meth:`get_dipole_atom_energy` (shape ``(..., nat)``).
- Atomic quadrupoles: :meth:`get_quadrupole_atom_potential` (shape
  ``(..., nat, 6)``) and :meth:`get_quadrupole_atom_energy` (shape
  ``(..., nat)``).

Each potential hook must return the partial derivative
:math:`\partial E_{\text{int}} / \partial q` of the corresponding energy hook
so that the variational SCF fixed point and implicit differentiation
(``differentiation="implicit"``) remain consistent with unrolled differentiation
(``differentiation="unrolled"``). Monopole hooks may omit ``qdp`` and ``qqp``
from their signatures when unused; overriding dipole or quadrupole hooks
automatically promotes the integral level when ``auto_int_level=True``.
Non-``keep_dtype`` floating-point tensor leaves on a custom interaction or its
cache must match the active SCF floating ``dtype``. Unlike classical setups,
:meth:`~dxtb.components.base.Interaction.get_cache` is geometry-dependent and is
evaluated fresh on each singlepoint call.

.. code-block:: python

    import torch
    from dxtb import Calculator, GFN1_XTB, IndexHelper
    from dxtb.components.base import Interaction, InteractionCache
    from dxtb.typing import DD, Tensor, override


    class OnsiteShiftCache(InteractionCache):
        """Per-geometry interaction cache."""

        shift: Tensor
        hardness: Tensor
        __slots__ = ["shift", "hardness"]

        def __init__(self, shift: Tensor, hardness: Tensor) -> None:
            super().__init__(device=shift.device, dtype=shift.dtype)
            self.shift = shift
            self.hardness = hardness


    class OnsiteShift(Interaction):
        """Quadratic atomic charge penalty coupled self-consistently."""

        scale: Tensor
        hardness: Tensor
        __slots__ = ["scale", "hardness"]

        def __init__(self, scale: Tensor, hardness: Tensor) -> None:
            super().__init__(device=scale.device, dtype=scale.dtype)
            self.scale = scale
            self.hardness = hardness

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> OnsiteShiftCache:
            assert positions is not None
            radial = self.scale * positions.square().sum(dim=-1)
            return OnsiteShiftCache(radial, self.hardness)

        @override
        def get_monopole_atom_potential(
            self,
            cache: OnsiteShiftCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            return cache.shift + cache.hardness * qat

        @override
        def get_monopole_atom_energy(
            self,
            cache: OnsiteShiftCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            return cache.shift * qat + 0.5 * cache.hardness * qat.square()


    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    interaction = OnsiteShift(
        torch.tensor(0.02, **dd),
        torch.tensor(0.10, **dd),
    )

    numbers = torch.tensor([8, 1, 1], device=dd["device"])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, -0.4]],
        **dd,
    )

    calc = Calculator(
        numbers,
        GFN1_XTB,
        interaction=(interaction,),
        opts={"int_driver": "pytorch"},
        **dd,
    )
    result = calc.singlepoint(positions)
    print(result.energy.sum(-1))


Transforms, ``vmap``, and Differentiation
-----------------------------------------

- **Component labels**:
  - *Interactions*: Distinct :class:`~dxtb.components.base.Interaction`
    instances in a single ``System`` or ``Calculator`` (including
    ``batch_mode=1`` padded batches and ``batch_mode=2`` conformer batches) must
    have unique ``label`` strings so their per-geometry SCF caches remain
    unambiguous; duplicate interaction labels raise :class:`ValueError`.
  - *Classicals*: Reserved built-in classical labels (``Repulsion``,
    ``ShortRangeBond``, ``DispersionD3``, ``DispersionD4``, ``IES``, ``Halogen``)
    cannot be shared by distinct instances (:class:`ValueError`), and direct
    single-``System`` :meth:`~dxtb.calculators.Model.setup` (``batch_mode=0``,
    as well as initial 1D :class:`~dxtb.Calculator` setup) rejects duplicate
    classical labels because ``System.classical_cache`` is label-keyed. Batched
    ``Model.setup`` (``batch_mode=1`` or ``2``) and ``Calculator`` evaluations
    support multiple custom :class:`~dxtb.components.base.Classical` instances
    sharing a non-reserved ``label``: each component sees its real
    ``self.label`` inside ``get_cache``, retains an independent setup snapshot
    or core descriptor, and contributes a separate ``(label, energy)`` entry in
    ``Result.classical``.
- **Supported state, containers, and object-graph contract**: Custom components
  and their caches may store floating and integer/boolean PyTorch ``Tensor``
  leaves inside ``__slots__``, ``__dict__``, ``ComponentCache``,
  :class:`~dxtb.typing.TensorLike`, ``tad_mctc.tree.Node`` (using ``child()``,
  or ``child(keep_dtype=True)`` for fixed-precision tensors), ``tuple``,
  ``NamedTuple``, ``list``, ``dict``, mutable ``@dataclass`` instances, and
  ``@dataclass(frozen=True)`` instances whose ``init=False`` fields hold only
  non-tensor static metadata. Acyclic shared tensor and container references
  (aliases) are preserved across ``.to()``, ``.type()``, ``.reset()`` (for
  direct component tensor leaves), and ``Calculator.type()``. Cyclic component
  or cache object graphs raise :class:`ValueError`; frozen dataclasses with
  tensor-valued ``init=False`` fields and ``Node`` instances storing tensors in
  ``context()`` fields raise :class:`TypeError`. Subclasses with custom
  ``.to()`` / ``.type()`` / ``_convert_child()`` overrides or helper
  ``TensorLike`` objects are honored during conversion and routed to independent
  (non-``vmap``) evaluation.
- **Opaque external objects**: Arbitrary third-party objects and non-transparent
  standard-library containers (such as ``set``, ``frozenset``,
  ``collections.deque``, ``types.MappingProxyType``, and ``functools.partial``)
  are not recursively traversed for dtype/device conversion or functional leaf
  extraction. Opaque objects holding only non-tensor state are supported and
  route custom components to independent evaluation; opaque objects storing any
  reachable PyTorch ``Tensor`` raise :class:`TypeError` (and cyclic graphs raise
  :class:`ValueError`) so conversion and SCF differentiation never silently skip
  active tensor parameters.
- **Hook purity and automatic ``vmap`` fallback**: ``get_cache``,
  ``get_energy``, and interaction potential/energy hooks must be pure functions
  of their arguments and component tensors without mutating component state,
  cache state, ``numbers``, or ``ihelp`` (mutating ``numbers`` or ``ihelp`` in
  place raises :class:`ValueError`). Components holding non-tensor opaque
  objects are routed directly to independent per-lane evaluation. Tensor-only
  custom components that hit a genuine PyTorch ``vmap`` operator limitation
  (such as ``.item()`` or data-dependent host branching) automatically fall back
  to independent evaluation provided neither component nor cache state was
  mutated during the mapped attempt; mutating state during a failed mapped
  attempt raises :class:`RuntimeError`. Unrelated user exceptions propagate
  unmodified, and lane evaluation errors carry ``exc.source_id`` identifying the
  failing source lane.
- **Differentiation and mutation invalidation**: Tensor parameters stored in
  supported containers on a component or its cache participate in autograd
  under both ``differentiation="unrolled"`` and ``differentiation="implicit"``.
  Any between-call change to a custom ``Classical``'s (or its captured
  cache's) tensor or non-tensor attributes invalidates that term's captured
  setup and triggers a fresh ``get_cache`` call on the next evaluation without
  discarding unrelated built-in setups. This invalidation is preserved across
  ``Calculator.type(dtype)`` and call-local dtype conversion so stale caches are
  purged rather than revalidated, while unmutated custom caches convert in
  lockstep with their components.
