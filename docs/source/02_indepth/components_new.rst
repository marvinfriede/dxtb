.. _indepth_components_new:

Creating New Components
=======================

In the following, we explain how to create a custom classical tight-binding
component that can be passed to :class:`~dxtb._src.calculators.model.Model` or
:class:`~dxtb.Calculator`. Custom classical contributions inherit from
:class:`~dxtb._src.components.classicals.base.Classical` and implement two
methods: :meth:`get_cache` (for composition-dependent setup data) and
:meth:`get_energy` (for geometry-dependent energy evaluation).

.. note::

   Self-consistent electronic interactions are evaluated by the functional SCF
   solver using immutable built-in descriptors (:class:`ES2`, :class:`ES3`,
   :class:`AES2`, :class:`DispersionD4SC`, :class:`GeneralizedBorn`). External
   electric fields and field gradients are passed directly as evaluation inputs
   (``field=`` and ``field_grad=``) to :meth:`System.singlepoint` or
   :meth:`Calculator.singlepoint`. Custom :class:`Interaction` subclasses are
   not supported by the functional SCF solver and raise
   :class:`NotImplementedError`.

Step 1: Create the classical component class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A custom classical contribution inherits from
:class:`~dxtb._src.components.classicals.base.Classical`, which itself inherits
from :class:`~dxtb.typing.TensorLike` to support :meth:`~torch.Tensor.to` and
:meth:`~torch.Tensor.type`. List all tensor attributes in ``__slots__``.

.. code-block:: python

    from typing import Any
    import torch
    from dxtb import IndexHelper
    from dxtb._src.components.classicals.base import Classical, ClassicalCache
    from dxtb.typing import Tensor, override


    class HarmonicRestraintCache(ClassicalCache):
        """Setup data for the harmonic restraint."""

        force_constant: Tensor
        __slots__ = ["force_constant"]

        def __init__(
            self,
            force_constant: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(
                device=device if device is None else force_constant.device,
                dtype=dtype if dtype is None else force_constant.dtype,
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
                device=device if device is None else force_constant.device,
                dtype=dtype if dtype is None else force_constant.dtype,
            )
            self.force_constant = force_constant

Step 2: Implement setup data creation (``get_cache``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During :meth:`Model.setup`, :meth:`get_cache` is called with the atomic numbers
and :class:`~dxtb.IndexHelper` to produce an immutable setup object for the
system composition. No persistent cache state or batch culling methods are stored
on the component itself.

.. code-block:: python

        @override
        def get_cache(
            self,
            numbers: Tensor,
            ihelp: IndexHelper | None = None,
        ) -> HarmonicRestraintCache:
            """Create composition-dependent setup data."""
            return HarmonicRestraintCache(self.force_constant)

Step 3: Implement the energy evaluation (``get_energy``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During evaluation, :meth:`get_energy` receives the atomic ``positions`` tensor
(shape ``(..., nat, 3)``) and the setup object returned by :meth:`get_cache`,
and returns atom-resolved energies of shape ``(..., nat)``.

.. code-block:: python

        @override
        def get_energy(
            self,
            positions: Tensor,
            cache: HarmonicRestraintCache,
            **_: Any,
        ) -> Tensor:
            """Return atom-resolved harmonic restraint energies."""
            return 0.5 * cache.force_constant * positions.square().sum(dim=-1)

Step 4: Pass the component to ``Model`` or ``Calculator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass an instance of the custom classical component via the ``classical`` tuple
on :class:`~dxtb._src.calculators.model.Model` or :class:`~dxtb.Calculator`:

.. code-block:: python

    import torch
    from dxtb import Calculator, GFN1_XTB
    from dxtb.typing import DD

    dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
    restraint = HarmonicRestraint(torch.tensor(0.05, **dd), **dd)

    numbers = torch.tensor([3, 1], device=dd["device"])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], **dd)

    calc = Calculator(numbers, GFN1_XTB, classical=(restraint,), **dd)
    result = calc.singlepoint(positions)
    print(result.total)

