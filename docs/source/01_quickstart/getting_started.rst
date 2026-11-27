.. _quickstart-getting-started:

Getting Started
===============

The main object of *dxtb* is the :class:`~dxtb.Calculator` class, which is used
to perform calculations on a given system.

Note that all quantities are in atomic units.

Creating a Calculator
---------------------

The constructor always requires the atomic numbers of the system(s) and a
tight-binding parametrization.
Currently, we provide the :data:`~dxtb.GFN0_XTB`,
:data:`~dxtb.GFN1_XTB`, and :data:`~dxtb.GFN2_XTB` parametrizations out of
the box.
If you directly use a corresponding method wrapper such as
:class:`~dxtb.calculators.GFN0Calculator`,
:class:`~dxtb.calculators.GFN1Calculator`, or
:class:`~dxtb.calculators.GFN2Calculator`, only the atomic numbers are required.

.. code-block:: python

    import torch
    import dxtb

    numbers = torch.tensor([3, 1])  # LiH
    calc1 = dxtb.calculators.GFN1Calculator(numbers)

    # equivalent
    calc2 = dxtb.Calculator(numbers, dxtb.GFN1_XTB)

We recommend to always pass the (floating point) :class:`~torch.dtype` and
:class:`~torch.device` arguments to the constructor to ensure consistency.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

The parametrization is converted into a differentiable
:class:`~dxtb.ParamModule` every time a calculator is created.
When creating many calculators, you can convert it once and pass the
:class:`~dxtb.ParamModule` instead.
Note that all calculators then share the same parameters.

.. code-block:: python

    numbers1 = torch.tensor([3, 1], device=dd["device"])
    numbers2 = torch.tensor([8, 1, 1], device=dd["device"])

    par = dxtb.ParamModule(dxtb.GFN1_XTB, **dd)
    calc1 = dxtb.Calculator(numbers1, par, **dd)
    calc2 = dxtb.Calculator(numbers2, par, **dd)

Using the Calculator
--------------------

Now, you can request various properties of the system. The most common one is
the total energy.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    energy = calc.energy(positions)

    print(energy)

If your system is charged or has unpaired electrons, you can supply ``chrg``
and/or ``spin`` as optional keyword arguments to :meth:`~dxtb.Calculator.energy`.

.. code-block:: python

    energy = calc.energy(positions, chrg=0, spin=0)

Instead of calling the :meth:`~dxtb.Calculator.energy` method, you can also
use corresponding getters :meth:`~dxtb.Calculator.get_energy`:

.. code-block:: python

    energy = calc.get_energy(positions, chrg=0, spin=0)

We recommend using the getters, as they provide the familiar
`ASE <https://wiki.fysik.dtu.dk/ase/>`_-like interface.


Gradients
---------

To calculate the gradients of the energy with respect to the atomic positions,
you can use the standard :func:`torch.autograd.grad` function.
Remember to set the ``requires_grad`` attribute of the positions tensor to
``True``.

.. code-block:: python

    import torch
    import dxtb

    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    numbers = torch.tensor([3, 1], device=dd["device"])
    calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    positions.requires_grad_(True)

    energy = calc.energy(positions)
    (g,) = torch.autograd.grad(energy, positions)

    print(g)

For convenience, you can use the :meth:`~dxtb.Calculator.forces` or
:meth:`~dxtb.Calculator.get_forces` method directly (with
``positions.requires_grad_(True)`` set as above).

.. code-block:: python

    forces = calc.forces(positions)
    forces = calc.get_forces(positions)

The equivalency of the two methods (except for the sign) can be verified by
the example `here <https://github.com/grimme-lab/dxtb/blob/main/examples/forces.py>`_.


.. note::

    Single-system and batched :class:`~dxtb.Calculator` evaluations are
    stateless across calls: you can evaluate the same :class:`~dxtb.Calculator`
    repeatedly with geometry or parameter gradient tracking enabled without
    calling :meth:`~dxtb.Calculator.reset` between calls. A persistent
    :class:`~dxtb.calculators.System` likewise supports repeated geometry
    differentiation; use :meth:`~dxtb.calculators.Model.setup_functional` when
    differentiating repeatedly with respect to setup-time parameters.


More Properties
---------------

Besides :meth:`~dxtb.Calculator.get_energy` / :meth:`~dxtb.Calculator.energy`
and :meth:`~dxtb.Calculator.get_forces` / :meth:`~dxtb.Calculator.forces`,
the :class:`~dxtb.Calculator` class provides methods to calculate various other
quantities:

- :meth:`~dxtb.Calculator.energy`: Total energy.
- :meth:`~dxtb.Calculator.forces`: Nuclear forces (negative gradient; requires ``positions.requires_grad_(True)``).
- :meth:`~dxtb.Calculator.dipole`: Electric dipole moment (requires explicit ``field=``).
- :meth:`~dxtb.Calculator.quadrupole`: Traceless electric quadrupole moment (requires explicit ``field_grad=``).
- :meth:`~dxtb.Calculator.dipole_deriv`: Derivative of electric dipole moment w.r.t. nuclear positions (requires ``field=``).
- :meth:`~dxtb.Calculator.polarizability`: Electric dipole polarizability (requires ``field=``).
- :meth:`~dxtb.Calculator.pol_deriv`: Derivative of electric dipole polarizability w.r.t. nuclear positions (requires ``field=``).
- :meth:`~dxtb.Calculator.hyperpolarizability`: Electric hyperpolarizability (requires ``field=``).
- :meth:`~dxtb.Calculator.hessian`: Hessian matrix (requires ``positions.requires_grad_(True)``).
- :meth:`~dxtb.Calculator.vibration`: Vibrational frequencies and normal modes (requires ``positions.requires_grad_(True)``).
- :meth:`~dxtb.Calculator.ir`: Infrared intensities (requires ``positions.requires_grad_(True)`` and ``field=``).
- :meth:`~dxtb.Calculator.raman`: Raman intensities (requires ``positions.requires_grad_(True)`` and ``field=``).
- :meth:`~dxtb.Calculator.bond_orders`: Wiberg bond orders.

Each method has a corresponding ``get_*`` getter, and additional SCF and
vibrational outputs are also accessible via getters:

- :meth:`~dxtb.Calculator.get_normal_modes`: Normal modes from vibrational analysis.
- :meth:`~dxtb.Calculator.get_frequencies`: Vibrational frequencies.
- :meth:`~dxtb.Calculator.get_ir_intensities`: Infrared intensities.
- :meth:`~dxtb.Calculator.get_raman_intensities`: Raman intensities.
- :meth:`~dxtb.Calculator.get_raman_depol`: Raman depolarization ratios.
- :meth:`~dxtb.Calculator.get_charges` /
  :meth:`~dxtb.Calculator.get_mulliken_charges`: Atomic partial charges from SCF.
- :meth:`~dxtb.Calculator.get_bond_orders`: Wiberg bond orders.
- :meth:`~dxtb.Calculator.get_density`, :meth:`~dxtb.Calculator.get_coefficients`,
  :meth:`~dxtb.Calculator.get_mo_energies`, :meth:`~dxtb.Calculator.get_occupation`,
  :meth:`~dxtb.Calculator.get_potential`: Converged wavefunction and SCF quantities.
- :meth:`~dxtb.Calculator.get_iterations`: Number of SCF iterations.

Note that electric-field response properties on :class:`~dxtb.Calculator` take
an explicit ``field=`` (shape ``(3,)`` or ``(nbatch, 3)``, with ``int_level >=
dxtb.labels.INTLEVEL_DIPOLE``) or ``field_grad=`` (shape ``(3, 3)`` or
``(nbatch, 3, 3)``, with ``int_level >= dxtb.labels.INTLEVEL_QUADRUPOLE``)
evaluation input, with ``requires_grad_(True)`` set on differentiated tensors
when ``use_functorch=False``. All derivative methods (everything above except
:meth:`~dxtb.Calculator.energy` and :meth:`~dxtb.Calculator.bond_orders`)
utilize automatic differentiation and also provide a finite-difference
counterpart, e.g., :meth:`~dxtb.Calculator.forces_numerical`.

.. note:: Reusing calculation outputs

    Calculator property calls evaluate their inputs each time. To reuse several
    outputs from one evaluation, retain the returned value from
    :meth:`~dxtb.Calculator.singlepoint` and read its fields, for example
    ``result.energy`` and ``result.density``.

For more details, please see :ref:`here <indepth_calculators>`.
