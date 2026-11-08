SCF strategies
==============

SCF configuration separates the numerical strategy from how derivatives are
computed. The default is Anderson in potential coordinates with unrolled
fixed-trip differentiation.

.. code-block:: python

   from dxtb.config import Config

   config = Config.create()
   high_order = Config.create(
       strategy="anderson",
       differentiation="implicit",
       convergence="higher_order",
   )

Named strategies are ``linear``, ``anderson``, ``broyden``, ``cdiis``,
``broyden-cdiis``, ``anderson-cdiis``, ``cdiis-soscf``, and
``broyden-cdiis-soscf``. Broyden is a native limited-memory method available
in charge, potential, and Fock coordinates. CDIIS is a Fock-space method.
SOSCF is a real orbital second-order method, but it is restricted to
zero-temperature fixed occupations and eager primal execution. It is intended
as an explicit near-root stage, not a general startup method.

The following examples select supported policies explicitly:

.. code-block:: python

   from dxtb.config import Config

   broyden = Config.create(strategy="broyden")
   diis = Config.create(strategy="cdiis")
   second_order = Config.create(
       strategy="cdiis-soscf",
       differentiation="implicit",
       fermi_etemp=0.0,
   )
   thermal = Config.create(
       strategy="anderson",
       differentiation="implicit",
       fermi_etemp=2000.0,
   )

A custom staged strategy can select each method's representation and switch
criterion directly:

.. code-block:: python

   from dxtb import scf
   from dxtb.config import Config

   robust = scf.SCFStrategy(
       (
           scf.SCFStage(
               scf.BroydenMethod(
                   scf.BroydenMethodSettings(max_rank=8)
               ),
               representation=scf.SCFRepresentation.POTENTIAL,
               switch_policy=scf.SCFSwitchPolicy(fixed_point_l2=1.0e-2),
           ),
           scf.SCFStage(
               scf.CDIISMethod(scf.CDIISMethodSettings(max_history=8)),
               representation=scf.SCFRepresentation.FOCK,
           ),
       )
   )
   staged = Config.create(strategy=robust, differentiation="implicit")

The command line accepts the same named policy:

.. code-block:: console

   dxtb input.xyz --scf-strategy anderson --scf-differentiation unrolled \
       --scf-convergence standard

Fermi smearing at nonzero electronic temperature supports Linear, Anderson,
Broyden, and CDIIS. SOSCF and every SOSCF-containing strategy reject it. Thermal
occupations are never frozen to enable orbital SOSCF.

The ``standard``, ``tight``, and ``higher_order`` convergence profiles set
physical fixed-point and stationarity accuracy. Explicit ``x_atol`` and
``x_atol_max`` values override the profile. Profile selection is explicit and
does not depend on whether a tensor requires gradients.

Unrolled execution is the transform-compatible default for a single stage.
Same-shape fixed-trip ``vmap`` is supported for Linear, Anderson, Broyden, and
Fock CDIIS. Dynamic multi-stage strategies and SOSCF require eager primal
execution and are normally paired with ``differentiation="implicit"``.
Implicit differentiation attaches to a certified physical root and excludes
solver history. The implicit response uses potential coordinates by default.

Automatic ``torch.compile`` is not enabled. Current tests retain graph breaks
in raw maps and full solvers; an optimized-backend speedup has not been
established. Small same-shape ``vmap`` batches are supported, while streaming
is the memory-bounded option for large derivative batches.

The result's convergence flag and residual describe primal S2 physical
certification and controller progress. A later implicit response failure is
a derivative failure and does not change the primal convergence result.

Removed options
---------------

The older flat SCF options were removed, not converted. ``Config.create``,
``ConfigSCF.create`` and ``Calculator(..., opts={...})`` no longer accept
``mixer``, ``scp_mode``, ``scf_mode``, ``damp_init``, ``damp_soft_start``,
``damp_generations``, ``damp_diagonal_offset`` or ``f_atol``; passing any of
them raises a ``TypeError``. The integer label families ``SCF_MODE_*``,
``SCP_MODE_*`` and ``MIXER_*`` were removed from :mod:`dxtb.labels` as well.
A strategy whose Stages own their SCF converger and SCF variable replaces the
global converger and representation options, and ``differentiation`` replaces
the SCF mode. SCF convergence uses physical fixed-point and stationarity
checks; there is no residual-norm (``f_atol``) or ``strict`` switch.

Rewrite old configurations as follows:

.. list-table::
   :header-rows: 1

   * - Removed option
     - Replacement
   * - ``mixer="simple"`` (or ``"linear"``)
     - ``strategy="linear"``
   * - ``mixer="anderson"`` / ``mixer="broyden"``
     - ``strategy="anderson"`` / ``strategy="broyden"``
   * - ``scp_mode="charge"`` / ``"potential"`` / ``"fock"``
     - Set ``SCFStage.representation`` to ``SCFRepresentation.CHARGE`` /
       ``POTENTIAL`` / ``FOCK`` (see below)
   * - ``damp_init``, ``damp_generations``, ``damp_soft_start``,
       ``damp_diagonal_offset``
     - ``AndersonMethodSettings(damp_init=..., generations=...,
       soft_start=..., diagonal_offset=...)`` on an Anderson Stage
   * - ``scf_mode="full"`` / ``scf_mode="implicit"``
     - ``differentiation="unrolled"`` / ``differentiation="implicit"``
   * - ``f_atol`` or ``strict``
     - Use an explicit convergence profile; neither option controls SCF now

A named preset keeps its own SCF variable (potential coordinates for
``linear``, ``anderson`` and ``broyden``). To run a preset converger on
another SCF variable, replace the representation of its Stage, for example
the old ``mixer="anderson", scp_mode="charge", scf_mode="full"``:

.. code-block:: python

   from dataclasses import replace

   from dxtb import scf
   from dxtb.config import Config

   stage = replace(
       scf.strategy_preset("anderson").stages[0],
       representation=scf.SCFRepresentation.CHARGE,
   )
   config = Config.create(
       strategy=scf.SCFStrategy((stage,)),
       differentiation="unrolled",
   )

The functional configuration stores no converger or global representation
field: read them from the strategy (``config.scf.initial_representation`` and
``config.scf.strategy.stages[0].method``) and the derivative policy from
``config.scf.differentiation``. Experimental single-shot SCF and legacy SCF
engine overrides no longer exist.
