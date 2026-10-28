# SCF solver, convergence, and optimization redesign

**Project:** dxtb  
**Status:** architectural overview for future implementation plans  
**Reviewed branch:** `claude/tender-dijkstra-cda42s`  
**Reviewed HEAD:** `4546417c4a1174ce9320a0c18014eb552c678f7a`  
**Implementation kickoff base:** `f8297423a363305d0233307bc92d1c6c947b63a8`  

The reviewed HEAD above records the architecture review point; implementation
starts from this newer refactor commit.
**Date:** 2026-10-28

This document defines the intended high-level direction for the next SCF redesign.
It is deliberately an architectural roadmap rather than an implementation plan.
Each major step below should later receive its own focused implementation plan
with exact files, tests, migration details, and acceptance commands.

The redesign may break Python APIs. Compatibility with the old SCF/mixer API is
not a design goal. Compatibility with the ongoing functional dxtb refactor,
higher-order differentiation, `torch.func`, `vmap`, and the single-System core
is a design goal.

---

## 1. Executive decision

The target is one authoritative functional SCF implementation with:

1. one stateless physical electronic-structure core;
2. multiple interchangeable SCF numerical methods, including Broyden, Anderson,
   linear mixing, CDIIS and second-order methods;
3. a strategy controller that can switch methods during one SCF solve;
4. convergence assessment that is independent of the numerical method;
5. explicit final physical convergence certification;
6. differentiation that is independent of how the primal root was found;
7. removal of the vendored xitorch SCF implementation and, ultimately, the
   vendored xitorch package;
8. continued support for Broyden through a new native functional
   implementation;
9. a design that can support DIIS -> SOSCF, Broyden -> SOSCF, and other staged
   strategies without special-case SCF drivers.

The conceptual separation is:

```text
physical problem
      |
      v
electronic state <-------------------------+
      |                                    |
      v                                    |
SCF strategy controller                    |
      |                                    |
      v                                    |
current SCF method                          |
(linear / Anderson / Broyden / CDIIS /     |
 EDIIS / SOSCF / future methods)            |
      |                                    |
      v                                    |
trial state -> physical evaluation -> progress/convergence
      |                                    |
      +---------- continue/switch ----------+
      |
      v
final physical convergence certification
      |
      v
validated physical root
      |
      +---------------------+
      |                     |
      v                     v
 implicit response      unrolled/full differentiation
      |                     |
      +----------+----------+
                 |
                 v
             SCF result
```

The current refactor already supplies much of the correct physical foundation.
The redesign should preserve that work rather than start over.

---

## 2. Re-reviewed decisions

### 2.1 Decisions that remain

The following decisions are retained.

- **The functional SCF core is the foundation.** The stateless charge,
  potential, and Fock maps introduced by the refactor are the correct
  architectural direction.
- **The physical fixed-point equation must not contain mixing history.**
  Solver state, damping, DIIS history, Broyden updates, switch decisions, and
  iteration counters stay outside the physical map.
- **Implicit differentiation attaches to a validated physical root.**
  It must not differentiate through Broyden updates, DIIS coefficient solves,
  SOSCF switching logic, or the primal iteration trajectory.
- **Convergence must be independent of the mixer/solver.** A method may decide
  that it has produced a promising candidate, but final SCF acceptance is a
  physical property of the electronic state.
- **The vendored xitorch SCF path should be retired.** It is currently a
  compatibility implementation, not the desired numerical core.
- **Breaking API changes are acceptable.** The redesign should remove
  confusing legacy concepts rather than reproduce them behind new names.
- **Single-System physics remains authoritative.** Batching should be built
  around `vmap` and fixed-shape single-System operations, in agreement with the
  broader dxtb refactor.
- **Failure must be observable.** An unconverged primal root or response solve
  must never silently masquerade as a valid result.

### 2.2 Decisions that are revised

Two earlier ideas are refined.

#### Do not make `Mixer` or `Accelerator` the universal abstraction

Linear mixing and Anderson are naturally fixed-point accelerators. Broyden is
a quasi-Newton root method. CDIIS extrapolates Fock matrices using orbital
stationarity errors. SOSCF optimizes orbitals with second-order information.

Forcing all of these into an interface such as

```python
mixer.iter(new, old)
```

would make the abstraction too weak and would push method-specific physics
into the driver.

The common abstraction should instead be a numerical **SCF method** or
**SCF stage**. A method may maintain method-specific immutable state and may
operate in its own natural representation.

#### Do not force one working representation on the entire solve

A compact Broyden step may naturally work in charge or potential space.
CDIIS naturally works with Fock matrices and commutator/orbital-gradient
errors. SOSCF naturally works with orbital rotations.

Therefore the *working representation* belongs to the method/stage, not to the
whole SCF calculation.

A separate *response representation* may still be used to define the physical
fixed point for implicit differentiation. The existing charge, potential, and
Fock raw maps should be retained until benchmarks establish whether one should
become the preferred response representation.

---

## 3. Current refactor baseline to preserve

At the reviewed branch HEAD, the refactor already provides important pieces
that should survive the redesign.

### 3.1 Stateless physical maps

The functional SCF implementation has explicit raw maps for:

```text
charge:     x -> G_charge(x)
potential:  x -> G_potential(x)
Fock:       x -> G_fock(x)
```

The map result carries the mapped variable and the corresponding spectral
state. Mixing, convergence decisions, solver history, and finalization remain
outside the raw map.

This separation is a core invariant and should become permanent.

### 3.2 Immutable SCF input data

`SCFInputs` already separates immutable physical input data from solver
configuration. It includes structural mappings, integrals, occupations,
interactions, overlap treatment, orbital masks, and Fermi settings.

The redesign should keep this separation. Solver and strategy configuration
should not be stored in the physical problem data.

### 3.3 Functional spectral evaluation

The new functional path uses `tad_mctc.storch`-based overlap treatment and
spectral/density machinery rather than relying on xitorch's generalized
eigensolver for the new path.

This is the path that should become authoritative.

### 3.4 Functional simple and Anderson solvers

The refactor already has functional linear/simple mixing and functional
Anderson mixing with explicit immutable history and a shared solver-control
loop.

These should be migrated into the new method/strategy interfaces rather than
discarded.

### 3.5 Prepared-root implicit differentiation

The new implicit route validates the primal SCF/Fermi result before attaching
implicit derivatives and uses matrix-free response solves through
`tad_mctc.storch`.

This is a strong architectural boundary and should remain:

```text
primal solve -> validate physical root -> attach IFT -> finalize observables
```

The nonlinear solver trajectory is only a mechanism for finding the root.

### 3.6 Existing numerical evidence

The current E3.1 reference work is important for the redesign:

- charge, potential, and Fock fixed-point formulations have executable
  references;
- tight unrolling, dense IFT, implicit response, and physical finite
  differences have been compared;
- GFN2/NO2 demonstrates that ordinary SCF tolerances can be insufficient for
  third-order response;
- a root tolerance around `1e-12` enters the existing third-order reference
  bounds for the tested NO2 case, while tighter values such as `1e-13` provide
  a more stable reference.

The redesign must preserve these reference tests and must not weaken the
prepared-root semantics.

---

## 4. Target conceptual interfaces

The exact Python API is intentionally not fixed here. The focused
implementation plans should decide concrete names and signatures. The
following concepts, however, should be explicit.

### 4.1 `SCFProblem`

Represents the physical electronic-structure problem for one System.

Responsibilities:

- hold or reference immutable `SCFInputs`;
- evaluate the physical map in supported representations;
- construct Hamiltonian, density, charges, potential, occupations, and
  energies/free energies;
- provide conversions between representations;
- provide physical residual and orbital-gradient quantities needed by
  numerical methods;
- provide differentiable energy/free-energy functions needed by second-order
  methods.

It must not contain iteration history or algorithm-specific state.

### 4.2 `SCFState`

Represents one physically interpretable electronic state.

It should provide, directly or through a well-defined evaluation object, the
quantities needed by different methods:

- charges/multipoles;
- potential;
- Hamiltonian/Fock matrix;
- density matrix;
- coefficients and orbital energies;
- occupations;
- electronic energy and, where applicable, electronic free energy;
- Fermi subsolver status;
- structural masks.

A method may use only a subset of these quantities.

The state contract must make consistency explicit. The driver must not combine
a density from one physical evaluation with a Fock matrix from another and
silently call that a state.

### 4.3 `SCFMethod`

A numerical method that advances the SCF calculation.

Examples:

- linear mixing;
- Anderson;
- Broyden;
- CDIIS;
- EDIIS/ADIIS;
- SOSCF;
- future Newton-Krylov or augmented-Hessian methods.

A method owns its method-specific state but not the global SCF lifecycle.

The conceptual operation is:

```text
(current physical state, method state, progress information)
    -> proposed next state + updated method state + method diagnostics
```

A method is allowed to use its natural representation.

### 4.4 `SCFMethodState`

Immutable/tensor-friendly history owned by one method.

Examples:

- Anderson residual/history vectors;
- Broyden inverse-Jacobian or limited-memory updates;
- DIIS Fock/error histories;
- EDIIS density/energy history;
- SOSCF trust radius, preconditioner information, or microiteration state.

Method state must be reset or converted explicitly when switching to another
method. Incompatible histories must never leak across a strategy transition.

### 4.5 `SCFStrategy`

An ordered sequence or policy for selecting SCF methods.

Examples:

```text
linear -> CDIIS
EDIIS -> CDIIS
CDIIS -> SOSCF
Broyden -> SOSCF
Anderson -> CDIIS -> SOSCF
```

A strategy defines:

- which methods may be used;
- entry conditions;
- switch conditions;
- stage-specific iteration limits;
- fallback/restart rules;
- method compatibility requirements.

The strategy does **not** define physical convergence.

### 4.6 `SCFController`

Owns the global SCF lifecycle:

- global iteration/evaluation budget;
- current method/stage;
- physical evaluations;
- progress metrics;
- convergence checks;
- stage transitions;
- restart/fallback decisions;
- final certification;
- status and diagnostics.

There should be one authoritative controller rather than separate physical SCF
implementations for Broyden, DIIS, SOSCF, implicit mode, and full mode.

Two execution controllers may be justified:

1. an ordinary eager early-exit controller;
2. a fixed-trip/tensor-masked controller for transform/compile-oriented
   execution.

They must share the same physical problem, method definitions, convergence
semantics, and result schema.

### 4.7 `SCFConvergence`

A method-independent convergence evaluator.

It should distinguish:

- cheap iteration stopping criteria;
- progress/stagnation metrics used for method switching;
- final physical certification.

### 4.8 `SCFDifferentiation`

Differentiation is a separate policy from the primal solver:

```text
none
unrolled/full
implicit
```

The choice of Broyden, CDIIS, or SOSCF must not implicitly select the
differentiation mode.

---

## 5. Configuration redesign

The current `scf_mode`, `scp_mode`, `mixer`, generic `damp_*` fields, and
solver-specific legacy overrides mix several concerns.

A future configuration should instead make the independent choices explicit.

Illustrative only:

```python
SCFOptions(
    strategy=SCFStrategy(
        stages=(
            SCFStage(
                method=CDIISOptions(history=8),
                switch=When(orbital_gradient_below=1e-3),
            ),
            SCFStage(
                method=SOSCFOptions(
                    trust_region=True,
                    max_microiterations=20,
                ),
            ),
        ),
    ),
    differentiation="implicit",
    response_representation="potential",
    convergence=SCFConvergenceOptions(...),
    maxiter=100,
    fermi=FermiOptions(...),
)
```

The exact public API should be designed later, but the following principles are
mandatory:

- no silent substitution of one requested method for another;
- no hidden dependence of differentiation on the primal solver;
- no single generic damping configuration pretending to apply identically to
  every method;
- no untyped bag of legacy `fwd_options`/`kwargs`;
- method-specific options live with the method;
- global iteration and convergence policy lives at the strategy/controller
  level;
- effective tolerances are reported explicitly if dtype limitations require a
  numerical floor.

---

## 6. Convergence and correctness model

The redesign should separate **solver progress** from **physical convergence**.

### 6.1 Raw fixed-point residual

For a selected response representation,

\[
r(x) = G(x) - x.
\]

At minimum, retain both:

\[
\|r\|_2,\qquad \|r\|_\infty.
\]

This is the correct quantity for validating the physical fixed point used by
implicit differentiation.

Residual thresholds are representation-dependent: a charge residual and a
potential residual have different units and should not share tolerances merely
because both happen to be tensors.

### 6.2 Fock/density stationarity

For a nonorthogonal AO basis, use the SCF commutator/orbital-gradient
condition, e.g.

\[
R = F P S - S P F.
\]

The final convergence check must use a **freshly rebuilt Fock matrix
corresponding to the density being certified**.

Using the same Fock matrix that was diagonalized to generate the density would
make the commutator artificially uninformative.

An orthogonalized residual may be used for numerically stable norms, but the
metric and transformation must be explicit and tested.

### 6.3 Energy/free-energy consistency

Energy change is a useful progress and certification diagnostic but should not
be the sole convergence criterion.

At finite electronic temperature, the relevant thermodynamic quantity is the
electronic free energy rather than only the zero-temperature electronic
energy.

### 6.4 Fermi and occupation validity

Final certification must preserve the existing fail-closed behavior for:

- Fermi subsolver convergence;
- electron-count consistency;
- invalid temperature/settings;
- non-finite states.

### 6.5 Final independent certification

After the iterative method decides to stop, perform a final physical
certification based on the accepted state.

Conceptually:

```text
method-local stopping condition
        |
        v
rebuild/evaluate consistent physical state
        |
        +-- fixed-point residual
        +-- Fock-density/orbital-gradient residual
        +-- Fermi/electron-count validity
        +-- finite-value checks
        +-- energy/free-energy consistency diagnostics
        |
        v
accept / continue / restart / fail
```

This avoids accepting a state solely because one numerical method's internal
error estimate became small.

### 6.6 Tolerance profiles

Do not infer tighter SCF tolerances from `requires_grad`,
`torch.is_grad_enabled`, or transform mode.

Instead, expose explicit user-selectable convergence profiles or explicit
tolerances.

The current higher-order reference evidence should be used to calibrate
profiles such as:

```text
standard
tight
higher-order/reference
```

Exact defaults must be benchmark-driven and are not fixed by this document.

---

## 7. Method portfolio

### 7.1 Linear mixing

Purpose:

- simple robust baseline;
- early damping;
- debugging/reference behavior.

It should remain available but should not define the common solver interface.

### 7.2 Anderson

Retain the new functional Anderson implementation conceptually.

Required hardening:

- immutable fixed-size history;
- no accidental batch-lane coupling;
- robust regularization/conditioning;
- explicit restart behavior;
- compatibility with the shared strategy controller.

### 7.3 Broyden — mandatory first-class method

Broyden stays.

The algorithm should be reimplemented natively in the functional framework
rather than kept through xitorch.

Initial target:

- charge and/or potential-space root solve;
- limited-memory or otherwise bounded history;
- safeguarded updates;
- detection of singular/poor updates;
- restart policy;
- optional damping/trust safeguards;
- explicit status and diagnostics;
- exact raw physical residual as the convergence input.

The implementation should be benchmarked against the historical Broyden
behavior, Anderson, and DIIS, but the historical xitorch implementation is not
the numerical specification.

### 7.4 CDIIS

CDIIS should be the first DIIS variant implemented.

Its error vector should be based on the Fock/density stationarity residual.
The implementation should explicitly support the nonorthogonal AO basis and
define whether the stored error is raw AO or orthogonalized.

Required features:

- Fock/error history;
- constrained Pulay coefficient solve;
- conditioning checks and regularization;
- history pruning/restart;
- robust handling of nearly dependent histories;
- no cross-System history coupling;
- physical final certification independent of the DIIS error.

"AO-DIIS" should be treated as a representation/metric choice for DIIS rather
than as an unrelated solver family unless a genuinely distinct algorithm is
introduced.

### 7.5 EDIIS/ADIIS

EDIIS/ADIIS are useful candidates for robust early SCF, but they are not a
prerequisite for the first redesign milestone.

They require model-consistent energy expressions. The standard Hartree-Fock
quadratic formulas must not be copied blindly into GFN-xTB, especially for
GFN2 with nonlinear charge/multipole terms and finite-temperature free energy.

Therefore:

1. implement and validate CDIIS first;
2. derive the correct xTB energy/free-energy interpolation/objective;
3. add EDIIS/ADIIS only after the mathematical formulation is documented and
   numerically checked.

Whether EDIIS/ADIIS becomes part of the default strategy should be decided by
benchmarking.

### 7.6 SOSCF — genuine second-order orbital optimization

SOSCF must be implemented as a real orbital optimizer, not as a renamed
fixed-point Newton step.

For orthonormal orbitals parameterized by an antisymmetric rotation
\(\kappa\),

\[
C_{\mathrm{new}} = C \exp(\kappa),
\]

the local second-order model is

\[
E(\kappa) pprox E(0)
+ g^\mathrm{T}\kappa
+ rac{1}{2}\kappa^\mathrm{T}H_{\mathrm{orb}}\kappa.
\]

The second-order step solves or approximately solves

\[
H_{\mathrm{orb}}\Delta\kappa = -g.
\]

The xTB orbital Hessian must include the response of the actual GFN-xTB
Hamiltonian. It must not be approximated as a Hartree-Fock Hessian merely
because the outer algorithm resembles conventional SOSCF.

Preferred initial implementation:

- matrix-free orbital Hessian-vector products;
- iterative inner solve;
- orbital-energy-difference or related preconditioner;
- step control;
- trust-region or augmented-Hessian-ready interface;
- method-local microiterations;
- explicit acceptance/rejection diagnostics.

A raw undamped Newton step should not be the long-term design because the
orbital Hessian may be indefinite away from the quadratic regime.

### 7.7 Finite-temperature second-order behavior

This is a major design constraint.

The current dxtb configuration uses finite electronic temperature by default.
Conventional occupied/virtual SOSCF assumes fixed occupation structure and is
not automatically a correct optimizer of the finite-temperature free energy.

Therefore the implementation must **not** silently:

- disable smearing;
- freeze occupations;
- reinterpret a finite-temperature calculation as a zero-temperature SOSCF
  calculation.

Initial acceptable policy:

- support orbital SOSCF first for fixed/Aufbau occupation cases;
- make SOSCF stage eligibility explicit;
- for finite-temperature cases, retain Broyden/CDIIS/Anderson until a
  mathematically correct second-order free-energy method exists.

A future compact-space Newton-Krylov method or a coupled finite-temperature
orbital/free-energy second-order method may fill this role. The strategy
architecture must support this without another redesign.

---

## 8. Hybrid and switchable SCF strategies

Switching methods is a core requirement, not an afterthought.

### 8.1 Example strategies

```text
DIIS -> SOSCF
Broyden -> SOSCF
EDIIS -> CDIIS -> SOSCF
Anderson -> CDIIS
Broyden -> CDIIS
```

A default strategy is **not** selected by this document.

### 8.2 Switch criteria

The controller should be able to base transitions on physical progress metrics,
for example:

- fixed-point residual;
- commutator/orbital-gradient norm;
- free-energy/energy progress;
- stagnation over a window;
- oscillation/divergence indicators;
- minimum stage iterations;
- maximum stage iterations;
- method-specific health diagnostics.

SOSCF should normally be entered only in an appropriate local regime.
Experience in established quantum-chemistry programs supports using DIIS or a
robust first-order/quasi-Newton method first and switching to SOSCF near
convergence rather than starting with SOSCF far from the solution.

### 8.3 Global iteration ownership

The controller owns:

- the total iteration/evaluation budget;
- convergence status;
- switch decisions;
- restart/fallback decisions.

A method owns only its local step and local history.

### 8.4 Stage transitions

Every transition must define:

- how the current physical state is converted to the next method's working
  representation;
- which method history is discarded;
- which information is transferred;
- how counters are updated;
- how failures are reported.

For example, Broyden -> SOSCF should reconstruct a consistent spectral/orbital
state from the accepted compact-space iterate before initializing SOSCF.

### 8.5 Fallbacks

Fallbacks should be explicit strategy rules, never silent substitutions.

Examples:

```text
SOSCF rejected repeatedly -> restart CDIIS
CDIIS history singular -> prune/restart CDIIS
Broyden update unhealthy -> restart Broyden or switch Anderson
```

The result diagnostics should record the path taken.

---

## 9. Differentiation architecture

### 9.1 Implicit differentiation

The current prepared-root architecture should remain the preferred way to
differentiate sophisticated primal strategies.

The sequence is:

```text
hybrid primal strategy
        |
        v
final physical certification
        |
        v
convert to selected response representation
        |
        v
verify G(x*) - x*
        |
        v
attach implicit fixed-point response
        |
        v
finalize energy/observables
```

This means that a primal solve may use CDIIS and SOSCF while the derivative is
still defined by the physical charge/potential/Fock fixed-point equation.

The implicit derivative must not depend on the DIIS coefficients, Broyden
history, orbital trust radius, or stage-switch path.

### 9.2 Response representation

Do not yet force one canonical response representation.

The current branch has validated charge, potential, and Fock formulations.
Benchmark their:

- conditioning;
- response-solve iteration counts;
- memory;
- higher-order derivative stability;
- scaling with system size.

A later package may choose a preferred default while retaining alternatives if
valuable.

### 9.3 Unrolled/full differentiation

Do not require every advanced primal method to be differentiable through its
internal iterations immediately.

Phased support is acceptable:

1. eager primal + implicit differentiation;
2. transform-compatible prepared-root response;
3. fixed-trip/unrolled support for methods where it is useful and maintainable;
4. full compile/vmap support after correctness.

Linear/Anderson may remain easier to unroll than CDIIS/SOSCF hybrid strategies.
That is acceptable as long as the public capability is explicit.

### 9.4 Response-solve convergence

The tangent/adjoint linear solves have their own convergence problem.

Their status must remain separate from primal SCF convergence. A converged
primal root with an unconverged response solve is not a valid derivative.

The current fail-closed/NaN behavior under transforms should be preserved or
replaced only by an equally explicit failure contract.

---

## 10. xitorch retirement

The goal is to remove the vendored xitorch implementation, not Broyden.

### 10.1 Why retire it

The legacy xitorch path creates several problems:

- separate root-solver convergence semantics;
- known root-solver endpoint correctness concerns;
- historical memory-leak failures;
- legacy generalized eigensolver machinery;
- a default implicit route whose Broyden behavior is disconnected from the
  configured functional mixer;
- duplicated implicit-SCF architecture beside the new `tad_mctc.storch`
  prepared-root implementation.

### 10.2 Removal sequence

Do not delete xitorch first. Remove its consumers.

1. implement the new functional Broyden method;
2. move all intended single-System primal routes to the functional controller;
3. complete or deliberately retire legacy batch/override/fallback behaviors;
4. ensure the functional spectral path covers all intended production cases;
5. replace any remaining non-SCF xitorch consumers or prove they are unused;
6. remove the legacy `BaseSCF`/implicit/unrolling paths that only exist for the
   old engine;
7. delete `src/dxtb/_src/exlibs/xitorch`;
8. add architectural tests preventing its reintroduction.

The xitorch deletion should not wait for SOSCF if all existing production
functionality has already been migrated.

---

## 11. Batch, `vmap`, and compile implications

### 11.1 Single-System core remains authoritative

Do not reintroduce a permanently batched SCF implementation merely because
individual methods have histories.

Method state should be a pytree-compatible single-System value. Conformer and
mixed-System batching should be expressed through the broader `vmap` design.

### 11.2 Per-lane convergence and switching

Dynamic per-lane switching is straightforward in ordinary eager single-System
execution but harder inside a vmapped/fixed-trip controller.

Therefore distinguish:

- **eager strategy execution:** Python early exit and stage switching are
  allowed;
- **fixed-trip strategy execution:** use static schedules or tensor masks where
  required.

Do not distort the eager numerical architecture solely to make all switching
decisions compilable from day one.

### 11.3 Compile comes after correctness

`torch.compile` support is a performance feature, not the numerical
specification.

First establish:

- correct physics;
- correct convergence;
- deterministic method switching;
- correct derivatives;
- batch independence.

Then optimize the dominant execution paths.

---

## 12. Diagnostics and result contract

The new SCF result should provide enough information to diagnose convergence
without exposing mutable solver objects.

At minimum retain/report:

- converged/not converged;
- failure category;
- total physical map/evaluation count;
- final fixed-point residual norms;
- final commutator/orbital-gradient norms where applicable;
- energy/free-energy convergence diagnostics;
- Fermi/electron-count status;
- selected response representation;
- method/stage sequence used;
- per-stage iteration/evaluation counts;
- switch/restart/fallback events;
- effective tolerances;
- response-solve convergence information when derivatives are requested.

Dynamic tensor data should remain pytree/transform friendly. Human-readable
strings can be constructed at an outer eager reporting boundary.

---

## 13. Numerical validation program

The redesign must be benchmark-driven. Do not choose a default solver strategy
from iteration-count anecdotes.

### 13.1 Physical systems

The validation set should include at least:

- GFN1 and GFN2;
- closed-shell systems;
- open-shell systems;
- charged species;
- the existing NO2/GFN2 difficult higher-order case;
- stretched bonds;
- small-gap/near-degenerate cases;
- finite-temperature cases;
- systems that historically oscillate or converge slowly;
- conformer batches;
- mixed-composition padded Systems when the production batch route is ready.

### 13.2 Quantities

Compare:

- final energy/free energy;
- charges and multipoles;
- density and Hamiltonian;
- orbital energies/occupations where physically comparable;
- fixed-point residual;
- commutator/orbital-gradient residual;
- gradients;
- Hessians;
- selected third-order derivatives;
- implicit response versus tight unrolling and physical finite differences.

### 13.3 Method comparisons

Benchmark at least:

```text
linear
Anderson
new Broyden
CDIIS
hybrid CDIIS -> SOSCF
hybrid Broyden -> SOSCF
```

Add EDIIS/ADIIS and finite-temperature second-order methods when implemented.

Measure:

- physical map evaluations;
- wall time;
- memory;
- failure/restart rate;
- final certified residual;
- derivative accuracy.

A method requiring more expensive macrosteps must not be judged only by the
number of outer iterations.

### 13.4 Serial/batch independence

For every method that claims batch/vmap support:

- serial and mapped results must agree;
- method histories must not couple independent Systems;
- convergence/switching of one lane must not alter another lane's numerical
  state.

---

## 14. Major implementation steps

The following are the high-level work packages for which separate detailed
agent plans should be written later.

### S0 — Freeze architecture and reference matrix

**Purpose:** establish the redesign contract before code migration.

Deliverables:

- agree on the concepts in this document;
- inventory every legacy/xitorch SCF consumer;
- inventory current functional routes and fallbacks;
- freeze the numerical/reference test matrix;
- define which legacy behaviors are intentionally retired rather than migrated.

**Exit criterion:** no ambiguity about ownership of physical maps,
methods/strategies, convergence, differentiation, or xitorch removal.

---

### S1 — Introduce the new SCF problem/state/method contracts

**Purpose:** create the common architecture without changing numerical defaults.

Deliverables:

- `SCFProblem`/equivalent physical evaluation boundary;
- consistent `SCFState`/evaluation representation;
- `SCFMethod` and immutable method-state contract;
- strategy/stage data structures;
- global controller interface;
- redesigned typed configuration;
- redesigned result/diagnostic schema.

Migrate no advanced algorithm merely to prove the interface.

**Exit criterion:** linear mixing can run end-to-end through the new contracts
without changing physical results.

---

### S2 — Unified convergence and final physical certification

**Purpose:** make convergence independent of the selected method.

Deliverables:

- representation-aware raw fixed-point residuals;
- rebuilt Fock/density commutator or orbital-gradient diagnostic;
- energy/free-energy progress diagnostics;
- Fermi/electron-count validity integration;
- final certification step;
- explicit tolerance handling and reporting;
- preserved tight-root higher-order references.

**Exit criterion:** the same state receives the same final convergence verdict
regardless of which method produced it.

---

### S3 — Migrate linear and Anderson to the common method/controller

**Purpose:** prove that existing functional solvers fit the new architecture.

Deliverables:

- linear method;
- Anderson method;
- immutable histories;
- eager controller;
- fixed-trip controller where currently required;
- parity with the accepted functional branch behavior.

**Exit criterion:** the old functional simple/Anderson solver-specific driver
can be retired in favor of the common controller.

---

### S4 — Implement native functional Broyden

**Purpose:** retain Broyden while removing dependence on xitorch.

Deliverables:

- compact-space Broyden method;
- bounded/safeguarded history;
- restart and conditioning logic;
- charge/potential support as justified by benchmarks;
- diagnostics;
- parity/reference tests;
- implicit differentiation through the validated physical root.

**Exit criterion:** all intended Broyden production uses can run without the
xitorch root solver.

---

### S5 — Implement CDIIS

**Purpose:** add conventional commutator-based DIIS as a first-class method.

Deliverables:

- AO/Fock history;
- well-defined commutator/orbital-gradient error;
- orthogonalized/raw metric decision and tests;
- robust Pulay solve;
- history conditioning/pruning/restart;
- standalone CDIIS convergence benchmarks.

**Exit criterion:** CDIIS is reliable enough to serve as a strategy stage and
matches certified physical roots from the compact-space methods.

---

### S6 — Implement the strategy controller and method switching

**Purpose:** support multi-stage SCF calculations.

Deliverables:

- stage entry/switch/fallback policies;
- physical progress monitor;
- global iteration/evaluation budget;
- explicit state conversion at stage boundaries;
- history reset rules;
- stage diagnostics;
- configured Broyden -> CDIIS and/or CDIIS -> later-method transitions.

**Exit criterion:** switching methods does not change the physical convergence
definition or implicit derivative definition.

---

### S7 — Retire legacy SCF routing and xitorch

**Purpose:** make the functional controller authoritative.

Deliverables:

- migrate or explicitly retire every remaining legacy fallback;
- remove legacy Broyden routing;
- remove legacy xitorch generalized-eigensolver consumers;
- remove old SCF class hierarchy where no longer needed;
- delete vendored xitorch;
- update docs/config/tests;
- add architecture guard tests.

**Exit criterion:** production SCF and its derivatives have no xitorch
dependency.

This step need not wait for SOSCF.

---

### S8 — Derive and implement SOSCF for fixed/Aufbau occupations

**Purpose:** add a true second-order orbital optimization stage.

Deliverables:

- orbital parameterization and gradient;
- xTB-consistent Hessian-vector product;
- iterative inner solve;
- preconditioning;
- step acceptance/control;
- trust-region or augmented-Hessian-compatible design;
- microiteration diagnostics;
- explicit stage eligibility;
- comparisons with CDIIS/Broyden near convergence.

**Exit criterion:** a configured DIIS -> SOSCF or Broyden -> SOSCF strategy
converges certified zero/fixed-temperature reference cases and produces the
same physical root and derivatives as tight reference calculations.

---

### S9 — Finite-temperature second-order strategy

**Purpose:** avoid a permanent incompatibility between default Fermi smearing
and second-order SCF.

This is initially a research/design package.

Possible outcomes:

- a correct free-energy orbital SOSCF formulation;
- a coupled orbital/occupation second-order method;
- a compact-space Newton-Krylov refinement;
- a deliberate decision that finite-temperature calculations remain on
  Broyden/CDIIS/Anderson while SOSCF is restricted to fixed occupations.

**Exit criterion:** the supported policy is mathematically documented,
explicit in configuration, and never silently disables or freezes smearing.

---

### S10 — EDIIS/ADIIS and robust startup strategies

**Purpose:** determine whether energy-based DIIS materially improves difficult
xTB starts.

Deliverables:

- derive xTB-consistent energy/free-energy formulation;
- implement EDIIS/ADIIS if justified;
- benchmark EDIIS -> CDIIS and EDIIS/ADIIS -> SOSCF strategies;
- compare against Broyden/Anderson startup.

**Exit criterion:** either adopt the method with evidence or explicitly record
that it does not justify the complexity.

This package is intentionally after CDIIS and can be deprioritized if Broyden
and CDIIS already provide robust startup.

---

### S11 — Differentiation hardening across all strategies

**Purpose:** ensure solver sophistication does not compromise derivative
correctness.

Deliverables:

- implicit response after Broyden/CDIIS/SOSCF/hybrid primal solves;
- response-representation benchmarks;
- first/second/third derivative references;
- fail-closed tangent/adjoint solve behavior;
- no dependence of derivatives on strategy history;
- clearly scoped unrolled/full support.

**Exit criterion:** different primal strategies that reach the same physical
root yield matching derivatives within the established tight-root bounds.

---

### S12 — `vmap`, fixed-trip, compile, and performance hardening

**Purpose:** integrate the redesign with the broader dxtb transform and
performance goals.

Deliverables:

- single-System method-state pytrees;
- serial/vmap independence tests;
- fixed-trip semantics for supported methods;
- per-lane status handling;
- compile experiments driven by profiles;
- conformer and mixed-System performance measurements.

**Exit criterion:** supported transformed/batched routes are correct and no
performance optimization changes the numerical specification.

---

### S13 — Benchmark-driven defaults and public stabilization

**Purpose:** choose defaults only after the method portfolio is mature.

Decide from evidence:

- default primal strategy;
- whether Broyden or DIIS is the initial default stage;
- whether automatic SOSCF transition is enabled by default;
- default response representation;
- convergence profiles/tolerances;
- history sizes and switch thresholds;
- default finite-temperature behavior.

Then:

- freeze the new public API;
- remove transitional compatibility names;
- document strategy presets;
- document failure/status semantics.

**Exit criterion:** defaults are justified by a reproducible benchmark matrix,
not inherited from the legacy implementation.

---

## 15. Dependency/order summary

Recommended dependency flow:

```text
S0 architecture/reference freeze
 |
 v
S1 common contracts
 |
 v
S2 convergence/certification
 |
 v
S3 linear + Anderson migration
 |
 +--------------------+
 |                    |
 v                    v
S4 Broyden          S5 CDIIS
 |                    |
 +---------+----------+
           |
           v
S6 strategy switching
 |
 +-------------------------+
 |                         |
 v                         v
S7 legacy/xitorch       S8 SOSCF
retirement                 |
                            +-----------+
                            |           |
                            v           v
                           S9          S10
                    finite-T 2nd   EDIIS/ADIIS
                    order policy
                            \           /
                             \         /
                              v       v
                               S11
                    differentiation hardening
                                |
                                v
                               S12
                     vmap/compile/performance
                                |
                                v
                               S13
                         benchmarked defaults
```

S4 and S5 can be developed in parallel after S2/S3.

S7 should proceed as soon as existing functionality no longer requires the
legacy engine; it does not depend on completing SOSCF.

S8 can begin once the common state/method/strategy and convergence contracts
are stable.

---

## 16. Explicit non-decisions

The following should **not** be fixed prematurely.

### Default SCF method

Do not yet declare Broyden, CDIIS, Anderson, or a hybrid as the default.
Benchmark first.

### Canonical response representation

Do not yet choose charge, potential, or Fock universally for implicit
differentiation. Existing references support all three; benchmark conditioning
and higher-order behavior.

### Exact convergence thresholds

The NO2 evidence proves that higher-order derivatives can require much tighter
roots than ordinary energy calculations, but it does not establish universal
public tolerances for every model and property.

### EDIIS/ADIIS as mandatory components

They are candidates, not architectural prerequisites.

### Finite-temperature SOSCF formulation

Do not claim conventional occupied/virtual SOSCF is valid under Fermi
smearing. Resolve this in S9.

### Full unrolling through every advanced method

Implicit differentiation of the certified physical root is the priority.
Unrolling through DIIS/SOSCF switching is optional unless a concrete use case
requires it.

---

## 17. Architectural invariants for every future implementation plan

Every lower-level work package should preserve these invariants.

1. **One physical SCF model.** Numerical methods must not implement different
   electronic equations.
2. **No hidden mutable SCF state.** Histories are explicit immutable values.
3. **Method != convergence.** A method proposes steps; common logic certifies
   the physical result.
4. **Method != differentiation.** The primal strategy is independent of the
   derivative strategy.
5. **No silent solver substitution.** Fallbacks and switches are explicit and
   diagnosable.
6. **Broyden remains supported.** xitorch removal does not mean Broyden removal.
7. **SOSCF is genuinely second order.** It uses the xTB orbital
   gradient/Hessian response, not a relabeled fixed-point mixer.
8. **Finite-temperature physics is preserved.** No hidden disabling/freezing
   of Fermi occupations.
9. **Implicit derivatives attach only to a certified root.**
10. **Response failures are explicit.**
11. **Single-System core remains authoritative.**
12. **Serial and mapped calculations are independent.**
13. **Higher-order derivative references are acceptance tests, not optional
    diagnostics.**
14. **Compile/performance work follows numerical correctness.**
15. **Legacy behavior is migrated only when scientifically/API valuable;
    otherwise it is explicitly retired.**

---

## 18. References and design context

Repository context:

- Refactor branch:
  <https://github.com/marvinfriede/dxtb/tree/claude/tender-dijkstra-cda42s>
- Existing root/implicit design record:
  `docs/E3.1-root-strategy.md`
- Existing top-level restructuring plan:
  `docs/plan/00-overview.md`
- Existing current-state/dependency plan:
  `docs/plan/07-current-state-and-execution-plan.md`

External reference behavior considered during this review:

- PySCF SCF documentation describes CDIIS/EDIIS/ADIIS and a second-order
  augmented-Hessian/Newton SCF implementation:
  <https://pyscf.org/user/scf.html>
- ORCA 6.1 documents switching from DIIS to SOSCF only near convergence,
  trust-region augmented-Hessian SCF for difficult cases, and the current
  restriction between its finite-temperature smearing and SOSCF:
  <https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/scf.html>

These external implementations are references for numerical strategy and
failure modes, not specifications for dxtb. The dxtb implementation must be
derived from the GFN-xTB energy/free-energy and the functional physical core.

