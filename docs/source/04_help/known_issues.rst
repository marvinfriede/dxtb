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


Implicit SCF uses a prepared physical-root boundary
--------------------------------------------------

``differentiation="implicit"`` uses the functional physical-root derivative. Root
acquisition and physical validation happen eagerly; the prepared-root response
supports ordinary ``torch.autograd`` and its documented ``torch.func``
transforms. Both single-System and batched iterative routes use the functional
SCF controller (canonical padded ``vmap`` for eligible PyTorch single-stage
strategies and independent functional evaluation otherwise). Unsupported
interaction extensions and retired solver overrides fail explicitly instead of
selecting another engine. The experimental single-shot mode is retired because
its derivatives were not exact. An implicit route rejects a nonconverged primal
with ``SCFConvergenceError`` for an uncertified primal root. The removed global
``strict`` configuration flag does not alter SCF policy.

In dxtb 0.4.0, second and higher derivatives were wrong with the implicit
modes, and ``scf_mode="nonpure"`` existed; that mode was removed.


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
not require gradients, or with another tensor of the same values. Persistent
per-evaluation component caches and mutable ``Calculator.integrals`` have been
retired; integral matrices, classical setups, and SCF iterations are evaluated
purely without cross-call state.


Retired legacy APIs and migration guide
---------------------------------------

The functional ``Model`` / ``System`` / ``Result`` and ``S13`` SCF architecture
intentionally retired the following stateful legacy APIs:

- ``Calculator.integrals``: integral matrices produced during a calculation are
  returned on ``Result`` (``result.overlap``, ``result.hcore``,
  ``result.dipole_integrals``, ``result.quadrupole_integrals``). Standalone
  integral objects and shortcut functions remain available through the public
  :mod:`dxtb.integrals` package (:class:`dxtb.integrals.Integrals`,
  :mod:`dxtb.integrals.wrappers`, :mod:`dxtb.integrals.types`).
- Stateful SCF and mixer classes (``SelfConsistentField``,
  ``SelfConsistentFieldFull``, ``SelfConsistentFieldSingleShot``, ``BaseSCF``,
  ``Anderson``, ``Broyden``, ``Simple``, ``BaseMixer``): use functional SCF
  strategies in :mod:`dxtb.scf` (:class:`~dxtb.scf.SCFStrategy`,
  :class:`~dxtb.scf.SCFStage`, :class:`~dxtb.scf.LinearMethod`,
  :class:`~dxtb.scf.AndersonMethod`, :class:`~dxtb.scf.BroydenMethod`,
  :class:`~dxtb.scf.CDIISMethod`, :class:`~dxtb.scf.SOSCFMethod`) via
  :class:`~dxtb.config.ConfigSCF`.
- Component cache-control and batch-culling methods (``cache_disable``,
  ``cache_enable``, ``cache_invalidate``, ``cache_is_setup``,
  ``cache_is_latest``, ``IndexHelper.cull`` / ``restore``, and cache ``cull`` /
  ``restore``): ``Model.setup`` captures immutable setup data on ``System``, and
  batched evaluations use canonical fixed-shape padding without dynamic batch
  culling.
- Legacy SCF option and composition wrappers (``LegacySCFOptions``,
  ``strict_e6bcompat``, low-level ``scf_results`` override kwargs,
  ``LegacyCompositionResult``, ``LegacySystemRoute``, ``classify_legacy_system``):
  configure SCF via :class:`~dxtb.config.Config` / :class:`~dxtb.config.ConfigSCF`,
  and use :func:`dxtb._src.calculators.packed_result.evaluate_packed_systems`
  and :func:`dxtb._src.calculators.legacy_composition.can_pad_system` for
  internal multi-System routing.


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


``reset()`` cuts gradients to user-supplied tensors
---------------------------------------------------

``calc.reset()``, ``reset_all()`` and the ``reset_*`` methods of the
interaction and classical lists replace the tensors of custom components with
detached copies. Gradients with respect to a tensor the user passed into a
custom component, e.g., ``requires_grad=True`` component parameters, are then
``None``. Electric fields are per-evaluation inputs and are not component
state; their graph is preserved across ``Calculator.reset()``. ``Model.par``
parameters are re-evaluated on each call and remain differentiable across
``Calculator.reset()``.

**Workaround:** for custom component parameters, pass a fresh component or
rebuild the Calculator after reset if gradients to the original leaves matter.


``Calculator.to(device)`` does not move all state
-------------------------------------------------

``Calculator.to`` converts the components, but not every internal object
(e.g., the :class:`~dxtb.IndexHelper` stays on the original device). Legacy
constructor-field defaults are also retained at their original device because
they are compatibility call inputs rather than interaction components.
(Changing floating-point precision via ``Calculator.type(dtype)`` is supported.)

**Workaround:** create the calculator on the target device
(``Calculator(..., device=device)``).


Device given as a string (fixed)
--------------------------------

Up to dxtb 0.4.0, ``Calculator(..., device="cpu")`` (a string instead of a
``torch.device``) and ``device="cuda"`` (without index, while tensors report
``cuda:0``) raised a ``DeviceError``. Devices are now normalized.
