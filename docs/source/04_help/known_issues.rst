.. _help_known_issues:

Known Issues
============

The sections below document remaining limitations and workarounds alongside
historical issues from earlier releases (marked ``(fixed)``) and APIs retired
by the functional ``Model`` / ``System`` / ``Result`` architecture.


Higher-order derivatives of systems with degenerate orbitals
------------------------------------------------------------

``dxtb`` evaluates the SCF density matrix with gauge-invariant spectral matrix
functions (finite-temperature Fermi–Dirac divided differences and
zero-temperature matrix-sign projectors), which resolves the broadened
eigendecomposition ``1 / (e_i - e_j)`` singularity that previously produced
erroneous second derivatives on degenerate closed-shell systems (such as the
in-plane polarizability of benzene) or ``NaN`` under forward-mode AD
(``torch.func.jacfwd`` / ``jvp``) on linear molecules.

Remaining caveats when differentiating degenerate or near-degenerate systems:

- **SCF convergence tolerance**: Standard SCF tolerances (``x_atol_max=1e-5``)
  are too loose for second- and third-order derivatives of degenerate systems
  (e.g., benzene's in-plane polarizability requires tight stationarity). Pass
  ``convergence="higher_order"`` (or tighten both ``x_atol`` and
  ``x_atol_max``) when computing Hessians, polarizabilities, or
  hyperpolarizabilities.
- **Zero-temperature open-shell frontier degeneracy**: At ``fermi_etemp=0.0``,
  an exact orbital degeneracy across differently occupied frontier orbitals
  makes the zero-temperature projector non-smooth; use nonzero electronic
  temperature (``fermi_etemp > 0``) for degenerate open-shell frontiers.
- **Implicit SCF forward-mode boundaries**: For higher-order ``torch.func``
  compositions involving forward-mode AD (``jvp`` / ``jacfwd``), use
  ``differentiation="unrolled"`` with ``int_driver="pytorch"`` and
  ``convergence="higher_order"``.


Implicit SCF uses a prepared physical-root boundary
--------------------------------------------------

``differentiation="implicit"`` uses the functional physical-root derivative. Root
acquisition and physical validation happen eagerly; the prepared-root CPSCF solve
supports ordinary ``torch.autograd`` and its documented ``torch.func``
transforms. Batched evaluations partition a batch into unpadded per-Composition
single-``System`` lanes: lanes sharing the same ``System`` (and optional input
presence) run together under ``torch.func.vmap`` over the fixed-trip kernel,
while all other lanes use independent evaluation, padding heterogeneous shapes
only when packing ``Result``. Custom :class:`~dxtb.components.base.Interaction`
subclasses are supported in both ``differentiation="unrolled"`` and
``differentiation="implicit"``; non-``Interaction`` objects and retired solver
overrides fail explicitly instead of selecting another engine. The experimental
single-shot mode is retired because its derivatives were not exact. An implicit
route rejects a nonconverged primal with ``SCFConvergenceError`` for an
uncertified primal root. The removed global ``strict`` configuration flag does
not alter SCF policy.

In dxtb 0.4.0, second and higher derivatives were wrong with the implicit
modes, and ``scf_mode="nonpure"`` existed; that mode was removed. The
``scf_mode`` option itself is no longer accepted: select
``differentiation="unrolled"`` or ``differentiation="implicit"`` instead (see
:doc:`../02_indepth/scf_strategy`).


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
8e-6 Eh too low with GFN2-xTB). The default ``scf_mode="full"`` (now
``differentiation="unrolled"``; the ``scf_mode`` option is no longer accepted)
was not affected.


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
  integral matrices are available through the stateless shortcut functions
  in :mod:`dxtb.integrals.wrappers`.
- Stateful integral layer (``dxtb.integrals.Integrals``, ``DriverManager``,
  the integral drivers and classes such as ``OverlapPytorch`` or
  ``DipoleLibcint``, and ``dxtb.integrals.factories`` /
  ``dxtb.integrals.types``): use :mod:`dxtb.integrals.wrappers`. The multipole
  wrappers return origin-centred integrals (nine quadrupole components, not
  traceless) for both drivers.
- Stateful SCF and mixer classes (``SelfConsistentField``,
  ``SelfConsistentFieldFull``, ``SelfConsistentFieldSingleShot``, ``BaseSCF``,
  ``Anderson``, ``Broyden``, ``Simple``, ``BaseMixer``): use functional SCF
  strategies in :mod:`dxtb.scf` (:class:`~dxtb.scf.SCFStrategy`,
  :class:`~dxtb.scf.SCFStage`, :class:`~dxtb.scf.LinearConverger`,
  :class:`~dxtb.scf.AndersonConverger`, :class:`~dxtb.scf.BroydenConverger`,
  :class:`~dxtb.scf.CDIISConverger`, :class:`~dxtb.scf.SOSCFConverger`) via
  :class:`~dxtb.config.ConfigSCF`.
- Component cache-control and batch-culling methods (``cache_disable``,
  ``cache_enable``, ``cache_invalidate``, ``cache_is_setup``,
  ``cache_is_latest``, ``IndexHelper.cull`` / ``restore``, and cache ``cull`` /
  ``restore``): ``Model.setup`` captures immutable setup data on ``System``,
  and batched evaluations partition batches into unpadded single-``System``
  lanes rather than culling padded batches.
- Component mutation (``Component.update`` / ``reset``, the list methods
  ``update``, ``reset``, ``reset_all``, ``update_halogen`` and
  ``reset_halogen``, and ``Calculator.reset``): components are immutable
  :class:`tad_mctc.tree.Node` objects; create a changed copy with
  ``component.replace(...)`` and build a new Model or Calculator from it.
- Positional component construction and the ``device`` / ``dtype`` arguments
  of component constructors: components take keyword arguments
  (``ES2(hubbard=..., gexp=...)``), and the inputs must already share one
  dtype and device (use ``.to(...)`` on the component afterwards, or the
  ``new_*`` factories, which convert for you).
  ``GeneralizedBorn.create``, ``D4Model.create`` and ``Cutoff.create`` keep
  the previous argument lists.
- Legacy SCF option and composition wrappers (``LegacySCFOptions``,
  ``strict_e6bcompat``, low-level ``scf_results`` override kwargs,
  ``LegacyCompositionResult``, ``LegacySystemRoute``, ``classify_legacy_system``):
  configure SCF via :class:`~dxtb.config.Config` / :class:`~dxtb.config.ConfigSCF`,
  and use :func:`dxtb._src.calculators.packed_result.evaluate_packed_systems`
  and :func:`dxtb._src.calculators.lane_evaluation.can_vmap_system` for
  internal multi-System routing.


Batched calculations hang with several CPU threads (fixed)
----------------------------------------------------------

Up to dxtb 0.4.0, batched SCF calculations (``batch_mode`` 1 or 2) on CPU with
more than one PyTorch thread could stall in the eigensolver: batched LU
factorizations (``torch.linalg.solve``) of matrices of a few hundred orbitals
did not finish with four threads in the baseline environment (PyTorch 2.14
wheel, MKL 2024.2), while they take a fraction of a second with one thread.
The functional SCF now precomputes the symmetric inverse square root
:math:`S^{-1/2}` of the overlap matrix via Newton–Schulz iteration and passes it
to the eigensolver, avoiding ``torch.linalg.solve``. A batched
``torch.linalg.solve`` in your own code can still stall in such an environment.

Running the test suite with several ``pytest-xdist`` workers is also much
slower if each worker uses several threads; set ``OMP_NUM_THREADS=1``.


``Calculator.to(device)`` does not move all state
-------------------------------------------------

``Calculator.to`` moves :class:`~dxtb.IndexHelper`, but does not move
``Calculator.model`` (:class:`~dxtb.ParamModule`), ``Calculator.system``, or
component lists to the new device. (Changing floating-point precision in place
via ``Calculator.type(dtype)`` is supported across built-in and custom
components.)

**Workaround:** create the calculator directly on the target device
(``Calculator(..., device=device)``).


Device given as a string (fixed)
--------------------------------

Up to dxtb 0.4.0, ``Calculator(..., device="cpu")`` (a string instead of a
``torch.device``) and ``device="cuda"`` (without index, while tensors report
``cuda:0``) raised a ``DeviceError``. Devices are now normalized.
