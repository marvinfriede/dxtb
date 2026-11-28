.. _help_errors:

Common Errors
=============

The following sections summarize common runtime and validation errors in
``dxtb`` and how to resolve them.


SCF convergence and certification errors
----------------------------------------

``SCFConvergenceError`` (or ``SCFConvergenceWarning`` when
``force_convergence=False`` on unrolled eager runs) indicates that the SCF loop
did not reach a certified stationary SCF solution within ``maxiter`` iterations,
or encountered a numerical failure recorded in ``Result.status``
(``SCFFailureCode``). In batched evaluations, raised exceptions carry
``exc.source_id`` identifying the failing batch lane. Try switching to a more
robust SCF strategy (such as ``strategy="broyden-cdiis"`` or
``strategy="anderson-cdiis"``), increasing ``maxiter``, or enabling finite
electronic temperature (``fermi_etemp``) for small-gap systems.


Non-finite Broyden converger updates
------------------------------------

``BroydenConverger`` checks secant updates before storing them. A zero or
ill-conditioned secant denominator clears the bounded correction history and
restarts from the scalar initial inverse approximation. The SCF controller then
continues with its ordinary SCF map and physical certification rules. This is a
converger-history restart, not an SCF convergence or Fermi failure.


Missing ``requires_grad`` or perturbation inputs on ``Calculator``
------------------------------------------------------------------

Autograd property methods on :class:`~dxtb.Calculator` raise ``RuntimeError``
when their differentiated input tensor is missing or does not track gradients:

- Nuclear derivative methods (``forces``, ``hessian``, ``vibration``, ``ir``,
  ``raman``) require ``positions.requires_grad_(True)`` when
  ``use_functorch=False``.
- Electric-field response methods (``dipole``, ``dipole_deriv``,
  ``polarizability``, ``pol_deriv``, ``hyperpolarizability``, ``ir``, ``raman``)
  require passing an explicit ``field=`` tensor (and ``int_level >=
  dxtb.labels.INTLEVEL_DIPOLE``); ``quadrupole`` requires passing an explicit
  ``field_grad=`` tensor (and ``int_level >= dxtb.labels.INTLEVEL_QUADRUPOLE``).
  Set ``requires_grad_(True)`` on ``field`` / ``field_grad`` when
  ``use_functorch=False``. (By contrast, functional :class:`~dxtb.calculators.System`
  property methods construct ``torch.func`` transforms internally at zero field
  when ``field`` / ``field_grad`` is omitted.)


``DeviceError`` and ``DtypeError``
----------------------------------

All floating-point call inputs (``positions``, tensor ``chrg`` / ``spin``,
``field``, ``field_grad``) and non-``keep_dtype`` floating tensors on custom
``Interaction`` components must match the active floating ``dtype`` and
``device`` of the ``System`` or ``Calculator``; mismatches raise ``DtypeError``
or ``DeviceError``. Construct inputs with a shared ``dd = {"device": ...,
"dtype": ...}`` dictionary or convert the calculator precision via
``calc.type(dtype)``.


Custom component validation errors
----------------------------------

Custom :class:`~dxtb.components.base.Classical` and
:class:`~dxtb.components.base.Interaction` components are
:class:`tad_mctc.tree.Node` objects (see :ref:`indepth_components_new`):

- ``ValueError``: duplicate ``Interaction`` labels, or duplicate ``Classical``
  labels in a direct single-``System`` ``Model.setup(batch_mode=0)``.
- ``NodeLayoutError``: a component or cache class defines ``__init__`` or
  declares a field without ``child()`` / ``context()``.
- ``TypeError``: a tensor in a ``context()`` field, an unhashable static value,
  floating-point tensors of different dtypes in one component, or setup data
  that is neither a ``Node``, a tensor, a container of them, nor ``None``.
- ``dataclasses.FrozenInstanceError``: assigning to a field of a component; use
  ``component.replace(...)`` instead.
