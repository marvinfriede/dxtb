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

For all available keywords and settings, see
:ref:`here <indepth_calculators_config_labels>`.

.. automodule:: dxtb._src.integral.wrappers

.. automodule:: dxtb._src.integral.driver

.. automodule:: dxtb._src.integral.types
