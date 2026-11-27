.. _indepth_integrals_drivers_pytorch:

PyTorch
=======

.. automodule:: dxtb._src.integral.driver.pytorch
   :no-index:

Overlap, dipole and quadrupole integrals
----------------------------------------

The PyTorch driver builds the overlap, dipole and quadrupole integrals (and
thereby GFN2-xTB, which needs the multipoles) without libcint. All integrals
are assembled from one-dimensional integrals, which can be computed with one
of two interchangeable algorithms, selected with the ``int_algorithm`` option
(command line: ``--int-algorithm``):

- ``os`` (default): Obara-Saika, three-index vertical recursion
- ``md``: McMurchie-Davidson with Hermite moments

Both algorithms agree with libcint to ``1e-12`` in double precision, support
CPU and CUDA, derivatives of any order (also with ``torch.func``) with respect
to the positions and the basis parameters, and work with ``torch.compile``.
Low-level :class:`~dxtb._src.integral.driver.pytorch.setup.PytorchIntegralSetup`
and ``build_*`` functions operate on one single ``System`` (composable with
``torch.vmap``, including ``Z == 0`` atom padding), while multi-system padded
batches are handled automatically by :class:`~dxtb.Calculator` and
:mod:`dxtb.integrals.wrappers`.

The molecular traceless quadrupole moment is evaluated as the derivative of the
energy with respect to an electric field gradient (``-3 dE / dG``, six packed
components ``xx, yx, yy, zx, zy, zz``), requiring the quadrupole integral:

- On :class:`~dxtb.Calculator`: via autograd
  (``calc.quadrupole(positions, field_grad=field_grad)`` or
  ``calc.get_quadrupole(positions, field_grad=field_grad)``) and finite
  differences (``calc.quadrupole_numerical(positions, field_grad=field_grad)``),
  both taking an explicit ``field_grad=`` tensor (with ``requires_grad=True``
  for autograd when ``use_functorch=False``).
- On :class:`~dxtb.calculators.System`: via
  ``system.quadrupole(positions)`` (evaluating the zero-field-gradient response
  with ``torch.func``).

These field-gradient derivatives agree with the analytical diagonal elements in
``tblite``; ``tblite``'s analytical off-diagonal formula (available standalone
in :func:`dxtb._src.calculators.properties.moments.quad.quadrupole`) counts the
point-charge and dipole contributions twice.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}
    numbers = torch.tensor([8, 1, 1], device=dd["device"])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], **dd
    )

    opts = {"int_driver": "pytorch", "int_algorithm": "os"}
    calc = dxtb.calculators.GFN2Calculator(numbers, opts=opts, **dd)
    energy = calc.get_energy(positions)

The option is only valid for the PyTorch driver; it is rejected for libcint.

On the CPU, libcint is faster than the PyTorch driver, in particular for the
multipole integrals of large systems.

Setup
-----

.. automodule:: dxtb._src.integral.driver.pytorch.setup
   :members:

Implementations
---------------

.. automodule:: dxtb._src.integral.driver.pytorch.multipole
   :members:
   :show-inheritance:
