.. _indepth_components_interactions:

Components: Interactions
========================

.. automodule:: dxtb._src.components.interactions
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


.. note::

   Uniform external electric fields and field gradients are passed per
   evaluation via ``field=`` and ``field_grad=`` on ``singlepoint`` rather than
   stored as setup-time ``Model`` or ``Calculator`` interactions.

Below, you can find the detailed documentation.

.. toctree::
   :maxdepth: 1

   Coulomb <../_autosummary/dxtb.components.coulomb>
   Dispersion (D4SC) <../_autosummary/dxtb.components.dispersion>
   Fields <../_autosummary/dxtb.components.field>
   Solvation <../_autosummary/dxtb.components.solvation>
