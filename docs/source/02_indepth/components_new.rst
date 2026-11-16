.. _indepth_components_new:

Creating New Components
=======================

``dxtb`` supports both custom **classical** contributions (geometry-dependent,
density-independent energy terms) and custom self-consistent **interactions**
(charge- or multipole-dependent terms coupled into the SCF cycle). Custom
components can be passed to :class:`~dxtb._src.calculators.model.Model` or
:class:`~dxtb.Calculator`, including direct subclasses of
:class:`~dxtb.components.base.Classical` /
:class:`~dxtb.components.base.Interaction` as well as
subclasses of built-in components.

.. note::

   Uniform external electric fields and electric field gradients are evaluation
   inputs rather than setup-time components: pass ``field=`` or ``field_grad=``
   to :meth:`System.singlepoint` or :meth:`Calculator.singlepoint`.


Custom Classical Components
---------------------------

A custom classical contribution inherits from
:class:`~dxtb.components.base.Classical` and implements:

1. :meth:`~dxtb.components.base.Classical.get_cache`: builds
   composition-dependent setup data from ``numbers`` and
   :class:`~dxtb.IndexHelper` during :meth:`Model.setup`.
2. :meth:`~dxtb.components.base.Classical.get_energy`: evaluates
   atom-resolved energies of shape ``(..., nat)`` from ``positions`` (shape
   ``(..., nat, 3)``) and the setup cache.

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
(``differentiation="unrolled"``).

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

- **Pure PyTorch state for Mapped evaluation**: When a custom component and its
  cache hold only PyTorch ``Tensor`` leaves and static Python metadata (scalars,
  strings, tuples, lists, dicts, dataclasses, or ``ComponentCache``), ``System``
  transforms (``forces``, ``hessian``, ``dipole``, ``polarizability``,
  ``vibration``, ``ir``, ``raman``) and conformer batch ``vmap`` groups run
  through the transform-compatible kernel automatically.
- **Fallback to Independent evaluation**: Components that hold opaque Python
  objects or data-dependent host control flow are evaluated independently per
  lane on ``Calculator``.
- **Differentiation**: Tensor parameters stored on the component or its cache
  participate in autograd under both ``differentiation="unrolled"`` and
  ``differentiation="implicit"``. Do not mutate component attributes in place
  inside ``get_cache``, ``get_energy``, or potential/energy hooks.
