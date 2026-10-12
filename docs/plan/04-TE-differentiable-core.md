# Track E: Differentiable core

**Purpose.** Make the per-call path differentiable to third order (reverse and forward mode), batchable with `vmap`, and free of data-dependent shapes. This track carries goals G1 and G2.

**Starting point.** See `07-current-state-and-execution-plan.md`.

- Remaining custom autograd functions: `CoulombMatrixAG` and optional `RepulsionAG`; none has a `jvp` rule.
- SCF: default mode unrolls all iterations (`SCF_MODE_FULL`); implicit modes use a vendored copy of xitorch (`scf/implicit/base.py`).
- GFN2 multipole integrals exist in PyTorch (E5, done).
- The PyTorch overlap uses the PairPlan builder (E2-core).
- The legacy production batch FULL solver culls converged systems in place (`IndexHelper.cull`/`restore`, cache `cull`/`restore`); the new functional single-System path is separately fixed-shape.

---

## E0 Decision note: SCF differentiation

**Goal.** Decide, from measurements, how the SCF is differentiated. E0 records two numerical decisions and one ownership boundary:

```text
E0a canonical SCF differentiation path
E0b degenerate/small-gap eigensolver response
E0c NO2 handling boundary (diagnosis in E3.0; any fix in E3.1)
```

Changing the Lorentzian broadening parameter is not an acceptable resolution of E0b.

E0 remains a hard design gate for G1.

**Investigations (using the T0.3/T0.4 harness).**

1. **Unrolled SCF.** Accuracy of orders 1–3 as a function of the convergence threshold; memory growth with iteration count.
2. **Vendored xitorch implicit SCF.** Does it support second and third order? Does it work under `vmap`? Does `EditableModule` interfere with `torch.func`?
3. **torch.func-native implicit differentiation.** Prototype on a toy fixed-point problem: a custom autograd function (with `setup_context`, `jvp` and vmap rules) that solves the linear system of the implicit function theorem; check orders 1–3 and `vmap`.
4. **libcint (via tad-libcint).** Highest derivative order available for overlap and multipole integrals; forward mode; `vmap`; GPU. Record what dxtb calls today (`int1e("ipovlp", ...)` for the overlap gradient).
5. **Eigensolver.** Behaviour of third derivatives near degeneracies, with and without Fermi smearing.

**Decisions to record.**

- SCF differentiation strategy for each order (unrolled, implicit, or mixed: implicit for the fixed point, unrolled where cheaper).
- Whether xitorch is replaced.
- E0b: how the eigensolver response at degeneracies and small gaps is handled (not by changing the broadening parameter).
- E0c: NO2 remains an unresolved G1 correctness blocker; E3.0 owns diagnosis
  and E3.1 owns any mathematical/spectral fix. E0 does not infer the cause or
  change NO2 tolerances, references, or markers.

**Done when.** The note is agreed.

**Needs.** T0.3, T0.4. **Unblocks.** E3.0 and, after full E1, full E3.
**Size.** M.

## G1 transform-driver contract

PyTorch is the canonical integral driver for higher-order `torch.func`
composition and full E3/E8 transform acceptance. This follows the fixed-shape
PyTorch `PairPlan` path (E2-core) and PyTorch GFN2 multipole path (E5).
`libcint` remains supported for eager, reference, and analytical use within
its documented capabilities; its lack of JVP/`jacfwd`/`vmap` support does not
block the G1 transform contract. This policy does not remove libcint from
dxtb or weaken any existing libcint value/reference regression.

This is the canonical driver-scope statement for Track E. B4.3, E3.0, E3.1,
full E3, and E8 follow it; E0 selects SCF differentiation and spectral
semantics, not an integral driver. Any future requirement for libcint
transform parity needs a separately scoped implementation package before it
can become an acceptance prerequisite.

**G2 batching-driver scope.** PyTorch is the authoritative transform and
batching driver for E6a, E6bimpl/E6broute, E6c1/E6c2, and G2. The public API shape remains
backend-independent, but E6a's transform guarantee is tested on Systems using
the canonical PyTorch driver. Libcint remains an eager/reference/analytical
backend within documented capabilities; no whole-System `torch.func`
guarantee is implied. A retained libcint batch compatibility API may compose
independent single-System calls with an outer Python loop. That is not the
authoritative G2 route and is excluded from G2 no-loop and throughput
acceptance.

---

## E1 Custom autograd functions

**Goal.** Every custom autograd function supports reverse mode to third order, forward mode and `vmap`; or is deleted.

**Starting point.** `OverlapAG` and `EFunction` are already gone. Remaining dxtb-local shortcuts
include `CoulombMatrixAG` and optional `RepulsionAG`; external blockers include
tad-dftd3 and the generalized eigensolver path.

Prefer removal in favor of plain torch rather than adding custom rules unless
profiling proves a need.

| Function | Location | Today |
| --- | --- | --- |
| `RepulsionAG` | `components/classicals/repulsion/rep.py` | `setup_context`, `generate_vmap_rule`, no `jvp`; optional |
| `CoulombMatrixAG` | `components/interactions/coulomb/secondorder.py` | no `jvp`, no vmap rule |

**Steps per function.**

1. Measure plain-torch performance (autograd through the forward code) against the custom function, for first-order cost and memory (T0.8 workloads).
2. If plain torch is within an agreed margin: delete the custom function.
3. Otherwise: convert to the `setup_context` form; add `jvp`; make `backward` use only differentiable operations; add a vmap rule (`generate_vmap_rule = True` where the body is pure torch).
4. Tests: orders 1–3 by `gradcheck` chains, forward mode, `vmap` with fallback warnings as errors (from T0.4).

**Done when.** Both pass T0.4's checks at orders 1–3, forward mode and `vmap`, or are deleted.

**Needs.** T0.4. **Unblocks.** E3.0 through the completed E1.1/E1.2 subset;
full E3, E6a, E8, and D2 still require full E1. **Size.** M.

---

## E2-core Fixed-shape single-System pair kernels

**Goal.** Integral and Hamiltonian pair computations with shapes fixed by the composition, no per-pair Python loops, and no geometry-dependent grouping.

**Starting point.** The current PairPlan/pair-builder implementation.

**Scope.** The completed E2-core milestone is the fixed-shape single-System
integral path: setup owns a structural PairPlan; pure overlap/dipole/
quadrupole builders consume it; core evaluation has no position-dependent
pair count or dynamic screening; parameter-connected setup tensors remain
connected. The packed legacy driver's per-entry setup loop is not part of
E2-core and remains E2-legacy until E6broute retires that execution path.

Apply the distance cutoff as a multiplicative mask (double-`where`, P8), not as a shape decision; replace any remaining in-place `fill_diagonal_` with a pure operation.

**Tests.** T0.2 overlap and H0 values reproduced; T0.4 checks at orders 1–3; `vmap` over positions with no fallbacks; profile against T0.8 W1 and W2.

**Done when.** PairPlan ownership and fixed-shape behavior are established for
single-System use, reference values and derivatives are reproduced, and
parameter gradients remain connected. Packed legacy-driver cleanup is tracked
separately as E2-legacy.

**Needs.** T0.2, B3a, B4. **Unblocks.** E6a, E3.1, D1. **Size.** L.

### E2-legacy packed-driver cleanup

The legacy packed Calculator path still performs per-entry setup in its
driver loop. Keep this compatibility behavior until E6broute makes the
mixed-System route authoritative. **Needs.** E6broute. This is the remaining
item under the partial E2 umbrella; no single-System transform consumer waits
for it.

---

## E3 SCF differentiation

E3 has a runnable preflight milestone and a separate full-completion gate.

### E3.0 Diagnosis-only spectral derivative / NO2 preflight

**Scope.** Reproduce the two user-visible open-shell GFN2 NO2 failures and
isolate their first failing layer before implementation. This package is
diagnosis-only: it may add focused characterization/regression tests and
record evidence, but it makes no mathematical, spectral, or eigensolver
implementation change. Inspect `src/dxtb/_src/scf/functional/map.py`
(`_transitional_diagonalize`, `evaluate_hamiltonian`),
`src/dxtb/_src/wavefunction/filling.py` (`get_fermi_occupation_core`,
density/occupation response and entropy), the `storch.eighb` generalized
eigensolver path, and these tests: `test/test_scf/test_functional_map.py`,
`test/test_scf/test_fermi_derivatives.py`,
`test/test_wavefunction/test_functional_filling.py`,
`test/test_wavefunction/test_fermi_derivatives.py`,
`test/test_wavefunction/test_filling.py`,
`test/test_baseline/components.py`, `test/test_baseline/test_derivatives.py`,
and `test/test_baseline/test_reference.py`. Do not include DFT-D3/tad-dftd3
dispersion in this preflight.

#### E3.0 acceptance: reproduce both historical NO2 failures first

Use `test/molecules.py::samples["NO2"]` geometry unchanged and the existing
open-shell `spin=1` setup in `test/test_baseline/molecules.py::SYSTEMS["NO2"]`.
First reproduce both exact entries in `KNOWN_WRONG[("NO2", "gfn2")]`,
individually, for every driver covered by that method-wide marker:
`pytorch` and `libcint`. The four cases are GFN2/PyTorch
hyperpolarizability, GFN2/libcint hyperpolarizability, GFN2/PyTorch
`pol_deriv`, and GFN2/libcint `pol_deriv`. Do not diagnose a lower layer
until all four cases reproduce. If drivers or quantities fail at
different layers, report each separately.

For transform experiments in the diagnosis ladder, use the canonical PyTorch
driver defined above. The libcint cases remain mandatory historical
reference/derivative reproductions because the existing marker covers them;
they do not make libcint JVP/`jacfwd`/`vmap` part of the G1 transform contract.

A local environment without libcint may skip those cases, but this is not
package-completion evidence. E3.0 cannot be marked complete until all four
cases currently covered by the method-wide NO2 marker and supported by the
baseline reference suite have been run in supported environments with the
corresponding driver available. The G1 transform-driver contract does not
change this marker/reference scope. Only a separately reviewed change to the
libcint eager/reference API scope, baseline reference inventory, or marker
scope can remove a libcint case. Record Python, PyTorch, dxtb, and
driver/library versions and environment for each case; evidence from separate
supported environments may be combined.

For these historical reproductions, reuse the T0.2 calculation path in
`test/test_baseline/compute.py::SCF_OPTIONS` and
`test/test_baseline/test_reference.py::test_reference`, including the ordinary
`compute(...)` helper and its `third` group. Its exact settings are
`scf_mode="full"`, `mixer="anderson"`, `maxiter=300`, `f_atol=1e-10`,
`x_atol=1e-10`, `verbosity=0`, default potential representation, and stored
300 K configuration. Keep the reference driver's `pytorch` or `libcint`
selection. `test/test_baseline/matrix.py` is the separate T0.3
derivative-methodology/status matrix, not the T0.2 reference configuration.
For each observable and driver record
the current nested-autograd value, stored dxtb reference, dxtb central
finite-difference result, tblite reference/finite-difference result, maximum
absolute and relative disagreements for autograd-versus-stored-dxtb,
autograd-versus-dxtb-finite-difference, and dxtb-finite-difference-versus-
tblite, plus driver, SCF settings, and actual derivative composition. Reuse
the T0.2 reference helper for historical reproduction; use T0.3 separately
for derivative-path sweeps.

Separately exercise the current functional production route as Case B: FULL,
potential representation, configured non-strict Broyden, effective Anderson,
300 K. Compare Case A and Case B at converged electronic energy, charges,
orbital energies, occupation, density, Hamiltonian, and both relevant field
derivatives (`hyperpolarizability` and `pol_deriv`). Investigate differences
that could explain either failure; do not infer equivalence from the configured
Broyden fallback alone.

Only after both endpoints reproduce, descend through the diagnosis ladder:
field-response endpoint; Fermi/occupation response with frozen
NO2-representative electron counts, spin channels, temperatures, and orbital
spectra; spectral-density construction with frozen Hamiltonian and overlap;
one raw SCF map at a fixed converged input; and finally the complete fixed
point. Include zero-temperature cases where occupations are defined,
finite-temperature response, exact/near degeneracies, and gauge-invariant
density, occupation, and energy outputs. Test reverse and forward first
derivatives plus nested second and third derivatives on affected smooth paths.
Compare against central finite differences over a double-precision step sweep
from `1e-2` through `1e-5`, recording stable regions and truncation/roundoff
behavior. T0.3 supplies derivative methodology/tolerances and matrix evidence;
the stored NO2 reference belongs to T0.2, while T0.4 supplies
component/eigensolver characterization.

For every GFN1 comparison intended to be independent of E1.3, construct the
System using the supported dispersion exclusion (`exclude=("disp",)` or its
exact current Config API equivalent) and assert that no DFT-D3 component is
present. Do not invoke DFT-D3 and merely omit its energy from differentiation.
An end-to-end GFN1 case that includes DFT-D3 is explicitly expected-blocked by
E1.3 and is not an E3.0 success criterion.

E3.0 is complete only when both exact NO2 endpoints reproduce and the first
failing layer of each is recorded with the evidence above. E3.0 cannot land a
mathematical/spectral fix. It must not introduce a new custom
spectral/autograd primitive, broadening formula, degeneracy convention, or
eigensolver derivative policy. If diagnosis identifies a need for a new
invariant derivative primitive in `tad_mctc.storch.eighb`, stop at the
reproducible diagnosis and create a separately reviewed upstream design and
implementation package. That package must specify and test exact/near
degeneracy, zero/finite temperature, reverse orders 1–3, JVP, jacfwd, and vmap
before dxtb consumes the pinned dependency.

**Needs.** E0, E4core, T0.2/T0.3/T0.4 baselines, and completed E1.1/E1.2
primitives used by the selected non-DFT-D3 path. E3.0 does not require E1.3
or full E1. **Unblocks.** E3.1 SCF differentiation implementation.
**Size.** M.

### E3.1 SCF differentiation implementation

**Scope and implementation contract.** Implement the E0-selected SCF
differentiation strategy on supported single-System paths after E3.0 isolates
the NO2 failures. The differentiated physical equation is the raw-map
residual `F(x, p) = G(x, p) - x = 0`; mixer history, dynamic damping, and
solver bookkeeping are not part of that implicit equation. The finite
functional unrolled composition of the E4core raw map and solver transitions
is the reference differentiation path.

E3.1 owns the concrete implementation:

1. Build and retain the differentiable unrolled reference from the shared
   E4core physical map, without legacy mutable solver state in its graph.
2. Implement the derivative strategy authorized by E0. If E0 selects
   implicit differentiation, define the fixed-point primal, adjoint backward
   solve, tangent/JVP solve, and transform rules against `G(x,p)-x=0`; do not
   make mixer history the fixed-point definition. If E0 selects unrolled
   differentiation, ordinary differentiable composition is the implementation
   and no implicit custom function is added.
3. Keep backward and JVP operations differentiable for required nested
   composition and validate higher orders.
4. Define derivative convergence tolerances/configuration only as authorized
   by E0, and apply E0's explicit policy to nonconverged implicit primals.
5. Fix spectral/Fermi defects demonstrated by E3.0 under E0's decided
   numerical contract. A new custom spectral primitive still requires a
   separately reviewed upstream mathematical/design package.
6. Add focused tests for raw-map derivatives, finite unrolled derivatives,
   implemented adjoint/JVP solves, higher-order composition, and
   unrolled/implicit agreement where both strategies remain. Compare with
   double-precision central finite differences over the T0.3 step sweep
   (`1e-2` through `1e-5`) and record the stable region and roundoff behavior.
7. Remove vendored xitorch only if E3.1 proves it has no remaining consumer
   and the replacement passes compatibility/derivative tests; otherwise
   defer deletion until its final consumer is retired.

E3.1 uses the canonical PyTorch transform driver. It excludes outer classical
DFT-D3: SCF differentiation can proceed without E1.3 because DFT-D3 is not
part of the self-consistent map. Existing libcint eager/reference behavior
remains covered separately within its supported scope.

There are two legal paths for a T0.2 NO2 fix. If the fix is common to both
drivers, validate PyTorch and libcint, regenerate both affected stored
references, remove the quantity from the method-wide marker, and land code,
references, and marker update together. If only one driver is fixed, extend
the marker schema in the same change so lookup receives the driver (for
example, `known_wrong(name, method, driver)`) and the strict unexpected-fix
sentinel is independent for each driver. For this driver-specific transition,
atomically remove the affected quantity from the broad NO2 method marker and
add an explicit wrong entry only for any still-failing driver. For example,
if the broad NO2 marker initially lists `{hyperpolarizability, pol_deriv}`,
then after a PyTorch-only hyperpolarizability fix the broad marker lists only
`{pol_deriv}`, while a libcint-specific marker lists
`{hyperpolarizability}`. PyTorch has no wrong entry for that quantity and is
normally enforced against its reference. Do not use tombstones or retain a
broad marker for a quantity fixed on one driver. Unaffected quantities
continue inheriting broad markers, so existing C6H6 and Fe(CO)5 system-wide
markers retain their current method/driver scope. Test broad system- and
method-marker inheritance, driver-specific wrong entries, corrected
PyTorch/still-wrong libcint, the inverse split, strict failure when a
still-marked result unexpectedly becomes correct, normal enforcement of an
unmarked quantity, unrelated-quantity inheritance, and final removal after
both drivers are corrected. Reference maintenance is atomic over every affected
`(method, driver, system, quantity)` and every affected stored file (the
`gfn2-pytorch/NO2.npz` and `gfn2-libcint/NO2.npz` files when both change).
Enumerate affected files; recompute through the ordinary reference-generation
path; confirm only expected quantities change; compare each corrected value
with tblite and dxtb finite differences; preserve unaffected quantities; and
update the marker only when its entire current scope is correct. Never patch
NPZ values manually or weaken the strict sentinel. Required evidence is
nested autograd, supported forward-mode, a stable central finite-difference
sweep, tblite/tolerance agreement, and no unaffected driver/method regressions.

Any T0.2 NO2 reference/marker change follows the driver-complete atomic rule
above.

**Needs.** E3.0, E0, E4core, E2-core, and the SCF-relevant completed
transform primitives (E1.1/E1.2); no outer DFT-D3 dependency. **Unblocks.** Full E3
end-to-end acceptance after full E1 is complete. **Size.** L.

### Full E3 end-to-end acceptance

**Goal.** Acceptance-only validation of E3.1 in the complete supported
end-to-end model. Full E3 does not introduce a second implementation.

**Steps.**

1. Validate complete GFN1 and GFN2 transform paths on the T0.2 reference set
   using the canonical PyTorch transform driver. Separately retain eager and
   reference/value regression coverage for PyTorch and libcint wherever each
   is supported; libcint is not required to pass higher-order transforms.
2. Validate DFT-D3-enabled GFN1 after E1.3; until then, that end-to-end row is
   blocked rather than approximated by silently omitting the classical term.
3. Check reverse orders 1–3, JVP, `jacfwd`, and `vmap` composition of the E3
   transform-safe single-System differentiation path, including parameter,
   geometry, field, and mixed derivatives required by T0.3. The exact B1
   public `torch.func.vmap(system.singlepoint, in_dims=(0,))(positions_batch)`
   composition remains E6a ownership and is not required to mark E3 complete.
4. Where both strategies remain, compare unrolled and implicit physical
   derivatives at the same converged root within solver convergence error.
5. Confirm the canonical PyTorch transform-driver contract and no regression
   in unaffected methods or derivative quantities. Preserve libcint
   eager/reference regressions within its supported scope.

**Done when.** The supported complete GFN1/GFN2 end-to-end transform contract
passes T0.3 tolerances on the canonical PyTorch transform driver, with
libcint eager/reference regressions retained where supported; the
DFT-D3-enabled GFN1 row waits for E1.3.

**Needs.** E3.1, E0, full E1 (including E1.3), E2-core, and E4core. Full E3
cannot be marked complete until the supported end-to-end GFN1/GFN2 transform
contract on PyTorch, including outer DFT-D3 where applicable, passes and
libcint eager/reference regressions remain covered where supported. **Unblocks.** E8,
F2d, F3, D2. **Size.** L.

---

## E4core Pure SCF step and masked convergence loop

**Note.** The currently named `scf/pure` functions are not the final pure design because `_Data` is mutable. E4core requires functional SCF state and functional mixer state. E4core is the accepted single-System implementation milestone; E4.6a/E4.6b are separate legacy compatibility cleanup packages and are not prerequisites of E4core or its consumers.

**Goal.** The functional single-System SCF iteration is a pure function of its
state and uses fixed-shape solver control. Legacy production batch culling
remains isolated until E6broute retires that route and E4.6a removes its
compatibility machinery after E6c1 W2/W3 and batch-critical E7 clearances.

**Steps.**

1. Define the SCF state (charges or potential, mixer history) and the step `state -> state` as a pure function of system, call inputs and state.
2. Fixed-trip transform controller: use per-system convergence flags and
   freeze converged entries with `torch.where`; do not use tensor-dependent
   Python control flow.
3. Mixer state per system, part of the step's state (so it can be vmapped) or handled in the driver.
4. Keep the functional core free of cull/restore calls. Do not remove those
   methods from a shared legacy adapter while production batch FULL still uses
   it; that deletion is E4.6a after E6broute retires the production batch route
   and the E6c1 W2/W3 plus batch-critical E7 gates pass.
5. Benchmark fixed-trip masking against eager early-stop and legacy batch
   culling on the relevant T0.8 workloads.

**Done when.** The single-System functional path is mutation-free, matches
T0.2, and its fixed-trip solver passes transform checks. This does not require
repository-wide culling deletion or migration of legacy production batches.

**Needs.** T0.2, B5, B6a-SCF-core. **Unblocks.** E3.0, E6a after its transform-safe
entrypoint, B6b-core, C5, D1, F4. **Size.** L.

---

## E5 PyTorch dipole and quadrupole integrals

**Status: done upstream (#270).** PyTorch dipole and quadrupole integral paths
exist and pass the T0 component checks through third order, forward mode and
vmap. Keep regression tests; do not reimplement.


---

## E6a Conformers: `vmap` against hand-written batching

**Public API.** Preserve the B1 call:
`torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)`. E6a changes
the implementation beneath this call; it does not introduce a public
conformer function. The API shape is backend-independent; the transform
guarantee is for Systems configured with the canonical PyTorch transform
driver. Retain libcint eager/reference regressions where supported, without
implying whole-System libcint transform support. C8a Result pytree support is
a separate prerequisite, so E6a does not mix result-container migration into
the physics change.

**Goal.** Confirm that one fixed System mapped over conformer positions matches
a Python loop and the legacy reference path.

**Steps.**

1. `System.singlepoint` statically uses the fixed-trip transform-safe
   controller, returns tensor-valued convergence/subsolve diagnostics, and
   performs no host warning/error inspection. A direct eager call to
   `System.singlepoint` intentionally follows those same fixed-trip semantics:
   it evaluates the configured trip count after convergence, returns invalid
   subsolve status instead of raising for Fermi failures, and does not emit
   nonconvergence warnings or errors. This is an explicit B1 behavior change,
   not an accidental side effect of making `vmap` work.
2. `Calculator.singlepoint` no longer obtains its ordinary eager result merely
   by delegating to `System.singlepoint`. Its eager compatibility adapter owns
   eager early stop and host warning/error interpretation and produces the
   same public `Result` contract. No functorch or tensor-wrapper detection is
   used.
3. Both paths share the same `scf_map`, spectral/Fermi numerical primitive,
   solver-step implementation, mixer formulas, and finalization. Only the
   controller and eager policy differ.
4. Compare one fixed System over real GFN1 and GFN2 conformer sets against a
   Python loop, with vmap fallback warnings as errors, using Systems
   configured with the canonical PyTorch transform driver. Mixed padded
   compositions belong to E6bimpl. Keep libcint eager/reference coverage where
   supported; it is not part of the transform guarantee.
5. Record time and memory on T0.8 W2.

**Result pytree prerequisite (C8a).** C8a independently proves the closed
Result schema can flatten/unflatten, vmap a tensor-built Result probe with
default `out_dims=0`, reconstruct nested values and static metadata, and
preserve autograd. It does not call `System.singlepoint` or exercise SCF
control. The exact public call belongs only to the E6a integration acceptance
below.

**Eager compatibility acceptance.** Test direct eager `System.singlepoint`
and `Calculator.singlepoint` separately for converged positive-budget,
nonconverged warning, `force_convergence=True`, `maxiter=0`, finite-T Fermi
subsolve failures and chronological error policy, dynamic damping, iteration
count, and selected physical result. Calculator retains legacy eager policy;
direct System follows the explicit fixed-trip/status-only behavior above.

**Decision rule.**

- If performance is acceptable, proceed.
- If one operation is responsible, profile it and add a transform/batching
  rule where justified.
- If performance is broadly unacceptable, record an E6 performance blocker
  and require explicit architecture review before retiring the legacy route.
  Do not silently create or preserve a second permanent batched physics core.

**Done when.** Results and decision recorded.

**Needs.** E1, E2-core, E4core, B6b-core, C8a Result pytree representation, T0.8,
the B5.6 public solver-status schema, and the private fixed-trip boundary
under the B1 public method. It does not need C5/C8b mixed-System stackability
or E4.6a. **Size.** M.

---

## E6b Mixed-System production batching through `vmap`

**Goal.** For the authoritative PyTorch production route, mixed-System
batching is stack plus `vmap` over single-System physics; `batch_mode`
disappears (P2). Integral and SCF implementations do not handle `batch_mode`
or per-entry physics loops. The final PyTorch composition is conformer
`vmap` plus stacked padded Systems. A retained libcint eager/reference batch
compatibility API may compose independent single-System evaluations with an
outer Python loop. That compatibility layer does not retain driver-internal
batch state, is not the authoritative G2 route, and is excluded from E6c/G2
performance and no-loop acceptance.

**Steps.**

1. Mixed sets: per-molecule `setup`, padding to size buckets, stacking with a tree map, `vmap` with `in_dims=(0, 0)` (stacking needs C8b).
2. A pad-and-stack helper, with configurable bucket sizes.
3. Batch entry points from B1 (no shape-based guessing).
4. Delete `batch_mode`, driver-internal per-entry batch construction/physics
   loops (including those in `integral/driver/pytorch/driver.py` and
   `libcint/driver.py`), and batched/unbatched implementation branches (for
   example, the two einsum strings in `BaseIntegral.normalize`). This does not
   prohibit an outer compatibility API loop over independent libcint
   single-System calls.
5. Authoritative batched GFN2 uses the PyTorch multipole integrals (E5).
   Libcint remains eager/reference/analytical compatibility, and any outer-loop
   batch composition is not used to satisfy G2 acceptance.

**Done when.** The PyTorch mixed-System implementation composes under
stack/vmap; T0.2 batched references are reproduced and per-System Hessians
have no cross-system blocks. E6bimpl owns mixed-System construction,
composition, and correctness. E6broute owns authoritative production
activation and retirement of the legacy batch FULL route. Keep implementation
and route activation distinct: E6bimpl can be completed and measured first;
E6broute proceeds only after the E6c1 W2/W3 clearances and batch-critical E7
gate.

**E6bcompat-policy (before E6bimpl scope is frozen).** Use the accepted B1
public batch API contract together with E4.5b's legacy FULL dispatch contract
to decide retain or retire for every currently supported legacy FULL
batch-facing behavior listed below. This policy determines what E6bimpl must
replace; it cannot first be decided during post-implementation validation.

**E6bcompat validation.** Validate the policy selected above. For each
retained behavior, prove the replacement preserves its supported values and
API behavior. For each retired behavior, prove the documented migration or
error contract and absence of accidental legacy fallback. In particular, if
libcint batch compatibility is retained, E6bimpl provides an outer
composition of independent libcint single-System evaluations, and this gate
proves its eager/reference values and API behavior. No driver-internal packed
batch state or per-entry setup loop may remain. This is a correctness/API
gate, not an E6c/G2 performance or no-loop criterion.

| Legacy FULL batch-facing behavior | Current legacy behavior | Future batch behavior and replacement owner | Retain/retire decision and acceptance |
| --- | --- | --- | --- |
| Standard PyTorch FULL batch | Reaches packed legacy FULL and its batch-aware integral/SCF path | Canonical stacked/padded `vmap`; E6bimpl | Compare supported values, status, and public batch shape with legacy. |
| Standard libcint batch | Reaches packed legacy FULL and driver batch setup | Outer independent single-System composition if retained; E6bimpl | Preserve supported eager/reference values and API behavior, or explicitly retire batch API. |
| Custom/unsupported interaction batch | Legacy FULL receives the extension and preserves its current behavior/error | Outer legacy single-System FULL calls if retained; E6bimpl composition over the E4.5b fallback | Prove extension behavior is preserved, or document explicit batch retirement/error. |
| FULL batch with `unexpected_option`, `solver_options`, any other unknown/nonempty `**kwargs`, positional engine args, `fwd_options`, `bck_options`, or `eigen_options` | E4.5b forwards all nonempty `args`/`kwargs` unchanged to the legacy constructor; unknown keywords raise its characterized `TypeError` family | Forward every argument/keyword unchanged to each retained single-System legacy FULL call; do not filter, normalize, or silently discard arbitrary kwargs | Test batch+`unexpected_option`, batch+`solver_options`, positional extra, and each named option. Unsupported keywords preserve the characterized exception family; supported options preserve constructor semantics, or document explicit API retirement/error. |
| Negative configured electronic temperature | Legacy warning order and eager exception family/message are authoritative | Outer legacy single-System FULL calls while policy remains supported | Compare warning order and exception/message family, or explicitly retire batch behavior. |
| Negative Fermi `maxiter` | Legacy Fermi result/error behavior is authoritative | Outer legacy single-System FULL calls while policy remains supported | Preserve characterized result/error, or explicitly retire batch behavior. |
| Negative outer SCF `maxiter` | Batch is legacy-owned before the functional-only E0 rejection | Preserve characterized per-entry legacy behavior if retained | Record the actual batch result/error, or explicitly retire batch behavior. |
| Missing/malformed `hcore` or overlap | Legacy FULL owns the current validation/error path | Preserve the single-System legacy error contract for each entry if retained | Assert exception family/message, or document an explicit validation/API change. |

E6bcompat-policy has no performance-gate role. The batch inventory is about
not losing currently supported behavior when the shared packed legacy FULL
route is removed. Final single-System fallback policy remains E4.6b-owned.

**Needs.** E6a, B3b, E4core, C5/C8b, defined padding, and
E6bcompat-policy. The policy is completed before implementation scope is
frozen. E7 owns padding validation; only its batch-critical subset gates route
activation. The broader E7 audit may continue afterward.
**Unblocks.** E6c1 W3 measurement and batch-critical E7 validation on the new
route; after those gates, E6bcompat validation, and E6c1 W2/W3 clearance,
E6broute may become authoritative. E6bimpl alone does
not unblock E4.6a or Track D3. **Size.** L.

---

## E7 Padding-safe masking audit

**Goal.** Derivatives stay finite to third order on padded systems (P8).

**Steps.**

1. Inventory every masked singular operation in the per-call path (`1/r`, `sqrt`, `exp` of large arguments, divisions by counts).
2. Apply double-`where`: mask the input before the singular operation, mask the output after.
3. Tests: padded systems in T0.3 finite to order 3, including under `vmap`.

E7 owns padding-safety validation and hardening; E6bimpl owns mixed-System
construction/composition/correctness, and E6broute owns production activation
and legacy-route retirement. E6broute requires only the E7 checks for padded
operations used by its production route, not completion of every eventual E7
audit item.

**Done when.** T0.3 padded cells finite at every order that passes for single systems.

**Needs.** T0.3. **Size.** M. E6broute activation is gated by the
batch-critical E7 subset, not by the entire eventual audit.

## E6c1 Pre-retirement performance gates

E6c1 is the explicit performance clearance required before E6broute retires the
shared legacy production batch route. Both measurements use the authoritative
PyTorch route. It includes both batching branches:

- **E6a/W2 conformers:** same-System positions mapped with the exact B1
  `vmap(system.singlepoint)` call;
- **E6bimpl/W3 mixed Systems:** padded/stacked compositions compared with the
  recorded legacy W3 mixed/training-like baseline.

For both, record CPU wall time, peak memory, CPU threads, and padding/bucket
costs where applicable. W2 clearance compares conformer throughput with the
legacy `batch_mode=2` target. W3 clearance compares mixed-System forward time,
force evaluation where applicable, peak memory, padding overhead, and bucket
count/size distribution with the legacy mixed-batch baseline. Parameter-
gradient timing is recorded when the C8b training tree exists.

Each clearance requires an acceptable measured result or an explicit
architecture-review decision accepting a regression. Missing measurements or
silence do not count as clearance. The batch-critical E7 safety checks and
E6bimpl value/derivative correctness are also hard prerequisites. Without W2,
W3, and E7 clearance, keep the legacy batch route. A broadly unacceptable
result blocks retirement until explicit review; it cannot silently authorize
a second permanent batched physics core.

An architecture-review waiver permits route retirement only. It does not
satisfy G2's conformer or mixed-System throughput target. E6c2/G2 remains open
until the documented performance target is met or its acceptance criterion is
explicitly amended; missing measurements are never clearance.

## E6c2 Final performance acceptance

Record the complete W2 and W3 result, including the C8b training-tree
parameter-gradient timing when available, together with PyTorch-route GPU
throughput and memory against the deferred T0.8 GPU baseline. E6c2 is the final
G2 performance sign-off; it may follow route migration because CPU W2/W3
clearances were already required by E6c1. A missing GPU baseline blocks final
GPU/G2 performance acceptance, not the CPU correctness decision or the E6broute
migration after E6c1 review.

**Needs.** E6c1, E6broute, T0.8 W2/W3, and the T0.8 GPU baseline for GPU
acceptance. **Size.** M.

---

## E8 Third-order acceptance (G1)

**Goal.** End-to-end validation of goal G1.

E8 uses the same canonical PyTorch transform-driver scope as full E3.
Libcint's supported eager/reference regressions remain enforced, but E8 does
not add a requirement for libcint higher-order transform parity.

**Steps.**

1. Third-order force constants for GFN1 and GFN2 on the T0.2 set: forward-over-reverse against finite differences of Hessians.
2. First hyperpolarizability: autograd against finite differences of polarizabilities.
3. Raman (polarizability derivatives) and IR (dipole derivatives) against T0.2.
4. Cost: time and memory against T0.8 W4/W5; record whether compiling (D2) is worth pursuing.

**Done when.** Third-order transforms pass T0.3 tolerances on the canonical
PyTorch transform driver and results are added to the reference data.
Libcint remains subject to its separate eager/reference regression contract.

**Needs.** E1, E2-core, E3, B8. **Size.** M.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| E0 | Decision note: SCF differentiation and NO2 ownership boundary (E0a/E0b/E0c) | T0.3, T0.4 | M |
| E1 | Custom autograd functions | T0.4 | M |
| E2-core | Fixed-shape single-System PairPlan/integral path | T0.2, B3a, B4 | L |
| E2-legacy | Retire packed legacy driver's per-entry setup loop | E6broute | S |
| E3.0 | Diagnosis-only spectral derivative / NO2 preflight | E0, E4core, T0.2/T0.3/T0.4, E1.1/E1.2 supported subset | M |
| E3.1 | SCF differentiation implementation and fixes | E3.0, E0, E4core, E2-core, SCF-relevant E1.1/E1.2 | L |
| E3 | Full end-to-end SCF differentiation acceptance, including DFT-D3 where applicable | E3.1, E0, full E1, E2-core, E4core | L |
| E4core | Pure single-System SCF path, fixed-trip control and tensor status | T0.2, B5, B6a-SCF-core | L |
| E5 | PyTorch multipole integrals (done, #270) | - | - |
| B5.6 | Public tensor-valued solver-status Result schema | B5, E4core | S |
| C8a | Result pytree output via tensor-built probe | B5, B5.6, C1 | M |
| E6a | B1 conformer `vmap(system.singlepoint)` over one fixed System | E1, E2-core, E4core, B6b-core, B5.6, C8a, T0.8, private fixed-trip boundary | M |
| E6bcompat-policy | Retain/retire policy for each legacy FULL batch-facing behavior using both the public batch API and legacy dispatch contracts | B1 batch API contract, E4.5b legacy FULL compatibility contract | S; freezes E6bimpl scope |
| E6bimpl | PyTorch mixed-System construction, stack/vmap, correctness; replacements required by E6bcompat-policy | E6a, B3b, E4core, C5/C8b, defined padding, E2-core, E6bcompat-policy | L; unblocks E6c1 W3, batch-critical E7, and E6bcompat validation |
| E6bcompat | Validate retained batch replacements or explicit API retirement/migration/error behavior | E6bcompat-policy, E6bimpl | S; required before E6broute |
| E6c1 | Pre-retirement W2 conformer and W3 mixed-System performance gates | E6a/W2, E6bimpl/W3 | M |
| E6broute | Authoritative PyTorch mixed-System production route; retire shared legacy batch FULL | E6bimpl, E6bcompat-policy, E6bcompat, E6c1 W2/W3 clearances, batch-critical E7 gate | M; unblocks E4.6a and Track D3 |
| E6c2 | Final W2/W3/GPU G2 performance acceptance | E6c1, E6broute, T0.8 GPU | M |
| E4.6a | Remove legacy batch cull/restore after batch fallback retirement | E6broute, E6c1 W2/W3, batch-critical E7 | M |
| E4.6b | Delete remaining legacy FULL loop after all fallback owners retire | E4.6a, fallback migrations/policy decisions | M |
| E7 | Padding-safety validation; batch-critical subset gates E6broute activation | T0.3 | M |
| E8 | Third-order acceptance | E1, E2-core, E3, B8a, B8b | M |
