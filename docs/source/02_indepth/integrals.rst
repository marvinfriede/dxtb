.. _indepth_integrals:

Integrals
=========

.. toctree::
   :maxdepth: 1

   Drivers <integrals_drivers>
   Shortcuts <../_autosummary/dxtb.integrals.wrappers>
   Types <integrals_types>


.. automodule:: dxtb.integrals
   :no-index:

.. note::

   :class:`~dxtb.Calculator` and :class:`~dxtb._src.calculators.model.System`
   evaluate integrals purely during :meth:`~dxtb.Calculator.singlepoint` and
   return the resulting matrices on :class:`~dxtb._src.calculators.result.Result`
   (``result.overlap``, ``result.hcore``, ``result.dipole_integrals``,
   ``result.quadrupole_integrals``); ``Calculator.integrals`` has been retired.
   For standalone integral matrices outside a calculator evaluation, use the
   stateless shortcut functions in :mod:`dxtb.integrals.wrappers`. The stateful
   integral classes, drivers and the ``DriverManager`` have been removed.

   **Breaking change**: The ``normalize`` keyword argument (including
   ``normalize=False``) on :func:`~dxtb.integrals.wrappers.overlap`,
   :func:`~dxtb.integrals.wrappers.dipint`, and
   :func:`~dxtb.integrals.wrappers.quadint` (as well as ``Result.overlap_norm``)
   has been removed; contracted Gaussian shells are always normalized to unit
   self-overlap at construction.

   **Parameter-autograd lifecycle**: Persistent :class:`~dxtb._src.integral.evaluation.IntegralSetup`
   and :class:`~dxtb._src.calculators.model.System` instances capture setup-time
   CGTO ``alphas`` and ``coeffs`` and support repeated geometry evaluations and
   independent position backward passes without ``retain_graph=True``. Repeated
   independent backward passes or :mod:`torch.func` transforms with respect to
   basis parameters (e.g. ``element.*.slater``) must rebuild a fresh setup per
   evaluation via :meth:`~dxtb._src.calculators.model.Model.setup_functional`,
   :class:`~dxtb.Calculator`, or :func:`~dxtb._src.integral.evaluation.setup_integral_evaluation`.

For all available keywords and settings, see
:ref:`here <indepth_calculators_config_labels>`.

.. automodule:: dxtb._src.integral.wrappers

.. automodule:: dxtb._src.integral.driver

.. automodule:: dxtb._src.integral.types
