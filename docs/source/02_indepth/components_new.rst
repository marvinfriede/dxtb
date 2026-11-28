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
    from tad_mctc.tree import child
    from dxtb import Calculator, GFN1_XTB, IndexHelper
    from dxtb.components.base import Classical, ClassicalCache
    from dxtb.typing import DD, Tensor, override


    class HarmonicRestraintCache(ClassicalCache):
        """Composition-dependent setup data for the harmonic restraint."""

        force_constant: Tensor = child()


    class HarmonicRestraint(Classical):
        """Isotropic harmonic restraint around the origin."""

        force_constant: Tensor = child()

        @override
        def get_cache(
            self,
            numbers: Tensor,
            ihelp: IndexHelper | None = None,
        ) -> HarmonicRestraintCache:
            """Create composition-dependent setup data."""
            return HarmonicRestraintCache(force_constant=self.force_constant)

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
    restraint = HarmonicRestraint(force_constant=torch.tensor(0.05, **dd))

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
Unlike classical setups,
:meth:`~dxtb.components.base.Interaction.get_cache` is geometry-dependent and is
evaluated fresh on each singlepoint call.

.. code-block:: python

    import torch
    from tad_mctc.tree import child
    from dxtb import Calculator, GFN1_XTB, IndexHelper
    from dxtb.components.base import Interaction, InteractionCache
    from dxtb.typing import DD, Tensor, override


    class OnsiteShiftCache(InteractionCache):
        """Per-geometry interaction cache."""

        shift: Tensor = child()
        hardness: Tensor = child()


    class OnsiteShift(Interaction):
        """Quadratic atomic charge penalty coupled self-consistently."""

        scale: Tensor = child()
        hardness: Tensor = child()

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
            return OnsiteShiftCache(shift=radial, hardness=self.hardness)

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
        scale=torch.tensor(0.02, **dd),
        hardness=torch.tensor(0.10, **dd),
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
    ``self.label`` inside ``get_cache``, keeps an independent setup
    or core descriptor, and contributes a separate ``(label, energy)`` entry in
    ``Result.classical``.
- **Components are Nodes**: Components and their caches are
  :class:`tad_mctc.tree.Node` objects: frozen dataclasses whose fields are
  declared with :func:`~tad_mctc.tree.child` (tensors, Nodes, lists, tuples
  or dicts of them, or static hashable values; ``child(keep_dtype=True)`` for
  fixed-precision tensors) and :func:`~tad_mctc.tree.context` (static
  hashable values). They are constructed with keyword arguments, must not
  define ``__init__`` (use ``_normalize`` to fill in derived fields, or a
  classmethod constructor), and convert with ``.to()`` and ``.type()``. All
  floating-point tensors of a component share one dtype, and all its tensors
  one device.
- **Immutability**: A component's fields cannot be reassigned. To change a
  parameter, create a new component with ``component.replace(...)`` and build
  a new Model, System or :class:`~dxtb.Calculator` from it; a System keeps
  the setup computed from the components it was built with. Changing the
  contents of a field in place (an in-place tensor operation, or assigning to
  an entry of a dict field) is not supported and gives undefined results.
- **Setup data**: ``get_cache`` must return a ``Node`` (such as a
  ``ClassicalCache`` or ``InteractionCache``), a tensor, ``None``, or a list,
  tuple (including a ``NamedTuple``) or dict of them; anything else raises
  :class:`TypeError` at setup. Its floating-point dtype may differ from the
  component's.
- **Hook purity and automatic ``vmap`` fallback**: ``get_cache``,
  ``get_energy``, and interaction potential/energy hooks must be pure
  functions of their arguments and the component's fields. Custom components
  that hit a genuine PyTorch ``vmap`` operator limitation (such as ``.item()``
  or data-dependent host branching) automatically fall back to independent
  per-lane evaluation. Unrelated user exceptions propagate unmodified, and lane
  evaluation errors carry ``exc.source_id`` identifying the failing source
  lane.
- **Differentiation**: Tensor fields of a component or its cache participate
  in autograd under both ``differentiation="unrolled"`` and
  ``differentiation="implicit"``.
