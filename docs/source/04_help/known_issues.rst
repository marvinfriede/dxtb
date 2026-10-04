.. _help_known_issues:

Known Issues
============

The following problems are known and will be fixed by the restructuring of
dxtb (see ``docs/plan`` in the repository), which removes the affected code.
Until then, use the workarounds given here. The list was compiled during the
baseline of the restructuring (Track 0); the tests marked
``xfail(strict=True)`` in ``test/test_baseline`` and
``test/test_calculator/test_cache/test_stale.py`` document every item.


Higher derivatives of systems with degenerate orbitals are wrong
----------------------------------------------------------------

Second- and higher-order derivatives (Hessian, polarizability, dipole and
polarizability derivatives, hyperpolarizability) are wrong when the system
has exactly degenerate orbitals and the perturbation splits them, e.g., the
in-plane polarizability of benzene (GFN2-xTB: 24 and -33 instead of 57
a.u.). Energies and first derivatives (forces, dipole moment) are not
affected.

The derivatives are taken through the eigendecomposition in the SCF, whose
backward pass contains :math:`1/(\epsilon_i - \epsilon_j)`. The broadening
that keeps it finite for degenerate pairs does not give the correct limit at
higher orders.

**Workaround:** use finite differences of first derivatives (e.g.,
``Calculator.hessian_numerical``), or break the symmetry slightly (a
displacement of 0.01 bohr already gives correct derivatives).


Cached results after a call without gradient tracking (fixed)
-------------------------------------------------------------

Up to dxtb 0.4.0, forces could silently miss contributions (errors up to
0.1 Eh/bohr) if the same calculator was used before with positions that did
not require gradients, or with another tensor of the same values. The
component caches and the integral driver now also check the gradient
tracking state of the positions.


Repeated derivatives on the same tensor raise an error
------------------------------------------------------

Calling ``calc.get_forces(positions)`` twice with the same ``positions``
tensor raises ``RuntimeError: Trying to backward through the graph a second
time``, because the second call reuses cached intermediate results whose
graph the first call has freed.

**Workaround:** create a new calculator, call ``calc.reset()`` (but see
below), or pass a fresh tensor (``positions.detach().clone()``).


``Calculator.forces`` fails for batched input
---------------------------------------------

For a batch of systems, ``calc.forces(positions)`` (default
``grad_mode="autograd"`` and ``"backward"``) raises ``RuntimeError: grad can
be implicitly created only for scalar outputs``.

**Workaround:** differentiate the sum of the energies, which gives the forces
of every system because the systems are independent:

.. code-block:: python

    energy = calc.energy(positions)
    (grad,) = torch.autograd.grad(energy.sum(), positions)
    forces = -grad


Result cache: numerical field derivatives and views of a batch
--------------------------------------------------------------

With ``opts={"cache_enabled": True}`` (default: ``False``):

- numerical derivatives with respect to the electric field (e.g.,
  ``dipole_numerical``) return zero, because the field updates do not
  invalidate the cached energies;
- the result cache identifies a tensor by its storage, so a view of a batch
  that was calculated before (e.g., ``positions[0]``) returns the cached
  result of the whole batch instead of being validated.

**Workaround:** keep the result cache disabled for these calculations.


``reset()`` cuts gradients to user-supplied tensors
---------------------------------------------------

``calc.reset()``, ``reset_all()`` and the ``reset_*`` methods of the
interaction and classical lists replace the tensors of the components with
detached copies. Gradients with respect to a tensor the user passed in, e.g.,
an electric field with ``requires_grad=True``, are then ``None``.

**Workaround:** pass the tensor again after a reset, e.g.,
``calc.interactions.update_efield(field=field)``.


``Calculator.to(device)`` does not move all state
-------------------------------------------------

``Calculator.to`` (and ``.type``) convert the components, but not every
internal object (e.g., the :class:`~dxtb.IndexHelper` stays on the original
device).

**Workaround:** create the calculator on the target device
(``Calculator(..., device=device)``).


``"cuda"`` and ``"cuda:0"`` are different devices
-------------------------------------------------

Device checks compare ``torch.device`` objects, and
``torch.device("cuda") != torch.device("cuda:0")``. A calculator created with
``device="cuda"`` can therefore reject positions on ``cuda:0`` (which is what
tensors on the default GPU report) with a ``DeviceError``.

**Workaround:** always pass an explicit index (``device="cuda:0"``).


Analytical repulsion: parameter gradients do not terminate
----------------------------------------------------------

The repulsion with the custom backward function
(``new_repulsion(..., with_analytical_gradient=True)``, not used by the
calculators) recurses without bound when differentiated with respect to its
parameters and runs out of memory. Position gradients are correct.

**Workaround:** use the default repulsion for parameter gradients.
