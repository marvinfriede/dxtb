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

Forward-mode differentiation (``torch.func.jacfwd``, ``jvp``) through the
SCF gives ``NaN`` for such systems (e.g., the dipole moment of a linear
molecule like OH-).

A related limitation: the broadening also biases derivatives when the
HOMO-LUMO gap is very small (about 1.5% at a gap of 1e-3 Hartree).

**Workaround:** use finite differences of first derivatives (e.g.,
``Calculator.hessian_numerical``), or break the symmetry slightly (a
displacement of 0.01 bohr already gives correct derivatives).


The implicit SCF mode does not work with ``torch.func`` transforms
-------------------------------------------------------------------

With ``scf_mode="implicit"``, the SCF is differentiated with the implicit
function theorem (fixed-point solver). Derivatives with ``torch.autograd``
are correct up to third order (Hessian, polarizability, dipole derivatives,
hyperpolarizability). In dxtb 0.4.0, second and higher derivatives were
wrong with the implicit modes, and ``scf_mode="nonpure"`` existed; that mode
was removed.

Transforms of ``torch.func`` (``jacrev``, ``jacfwd``, ``vmap``, ...) fail
with the implicit mode, e.g., ``You are attempting to call
Tensor.requires_grad_() ... inside of a function being transformed``. This
includes the ``functorch`` and ``forward`` paths of the calculator.

**Workaround:** use the default ``scf_mode="full"`` with transforms.


Derivative with respect to ``refocc`` (fixed)
---------------------------------------------

Between the introduction of the fractional number of electrons and the fix,
the gradient of the energy with respect to the reference occupation of an
element (``element.<X>.refocc``) was wrong for molecules with a wide
HOMO-LUMO gap (water, GFN1-xTB: -0.0036 instead of -0.2259 for hydrogen). The
derivative of the Fermi occupations with respect to the number of electrons
was dropped in such gaps.


Padded batches with the implicit SCF mode (fixed)
-------------------------------------------------

Up to dxtb 0.4.0, the implicit SCF modes put electrons into the padding
orbitals of a batch if a system had an occupied orbital close to zero
energy (e.g., an anion), which gave wrong energies (OH- next to water:
8e-6 Eh too low with GFN2-xTB). The default ``scf_mode="full"`` was not
affected.


Cached results after a call without gradient tracking (fixed)
-------------------------------------------------------------

Up to dxtb 0.4.0, forces could silently miss contributions (errors up to
0.1 Eh/bohr) if the same calculator was used before with positions that did
not require gradients, or with another tensor of the same values. The
component caches and the integral driver now also check the gradient
tracking state of the positions.


Batched calculations hang with several CPU threads (fixed)
----------------------------------------------------------

Up to dxtb 0.4.0, batched SCF calculations (``batch_mode`` 1 or 2) on CPU with
more than one PyTorch thread could stall in the eigensolver: batched LU
factorizations (``torch.linalg.solve``) of matrices of a few hundred orbitals
did not finish with four threads in the baseline environment (PyTorch 2.14
wheel, MKL 2024.2), while they take a fraction of a second with one thread.
The unrolled SCF (``scf_mode="full"``) now computes the inverse Cholesky
factor of the overlap matrix once and passes it to the eigensolver, which
avoids the solve (requires tad-mctc 0.9.1). A batched ``torch.linalg.solve``
in your own code can still stall in such an environment.

Running the test suite with several ``pytest-xdist`` workers is also much
slower if each worker uses several threads; set ``OMP_NUM_THREADS=1``.


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


Device given as a string (fixed)
--------------------------------

Up to dxtb 0.4.0, ``Calculator(..., device="cpu")`` (a string instead of a
``torch.device``) and ``device="cuda"`` (without index, while tensors report
``cuda:0``) raised a ``DeviceError``. Devices are now normalized.


Analytical repulsion: parameter gradients do not terminate
----------------------------------------------------------

The repulsion with the custom backward function
(``new_repulsion(..., with_analytical_gradient=True)``, not used by the
calculators) recurses without bound when differentiated with respect to its
parameters and runs out of memory. Position gradients are correct.

**Workaround:** use the default repulsion for parameter gradients.
