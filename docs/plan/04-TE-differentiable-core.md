# Track E: Differentiable core

**Purpose.** Make the per-call path differentiable to third order (reverse and forward mode), batchable with `vmap`, and free of data-dependent shapes. This track carries goals G1 and G2.

**Starting point.** See `07-current-state-and-execution-plan.md`.

- Remaining custom autograd functions: `CoulombMatrixAG` and optional `RepulsionAG`; none has a `jvp` rule.
- SCF: default mode unrolls all iterations (`SCF_MODE_FULL`); implicit modes use a vendored copy of xitorch (`scf/implicit/base.py`).
- GFN2 multipole integrals exist in PyTorch (E5, done).
- The PyTorch overlap uses the PairPlan builder (E2).
- The legacy production batch FULL solver culls converged systems in place (`IndexHelper.cull`/`restore`, cache `cull`/`restore`); the new functional single-System path is separately fixed-shape.

---

## E0 Decision note: SCF differentiation

**Goal.** Decide, from measurements, how the SCF is differentiated. E0 is split into three explicit decisions:

```text
E0a canonical SCF differentiation path
E0b degenerate/small-gap eigensolver response
E0c open-shell GFN2 NO2 third-order failure
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
- E0c: the cause and fix of the open-shell GFN2 NO2 third-order failure.

**Done when.** The note is agreed.

**Needs.** T0.3, T0.4. **Unblocks.** E3.0 and, after full E1, full E3.
**Size.** M.

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

## E2 Fixed-shape pair kernels

**Goal.** Integral and Hamiltonian pair computations with shapes fixed by the composition, no per-pair Python loops, and no geometry-dependent grouping.

**Starting point.** The current PairPlan/pair-builder implementation.

**Scope.**

```text
move PairPlan to setup;
make the driver single-system;
remove per-entry batch construction;
keep transform-critical pair shapes static;
only change grouping/packing further if profiling justifies it.
```

Apply the distance cutoff as a multiplicative mask (double-`where`, P8), not as a shape decision; replace any remaining in-place `fill_diagonal_` with a pure operation.

**Tests.** T0.2 overlap and H0 values reproduced; T0.4 checks at orders 1–3; `vmap` over positions with no fallbacks; profile against T0.8 W1 and W2.

**Done when.** No Python loop over pairs; no data-dependent operation in the per-call overlap path; reference reproduced.

**Needs.** T0.2, B3a, B4. **Unblocks.** E6a, D1. **Size.** L.

---

## E3 SCF differentiation

E3 has a runnable preflight milestone and a separate full-completion gate.

### E3.0 Spectral derivative / NO2 preflight and invariant eigensolver audit

**Scope.** Audit the spectral/Fermi derivative seam and isolate the open-shell
GFN2 NO2 third-order failure before starting the full E3 strategy. Inspect
`src/dxtb/_src/scf/functional/map.py` (`_transitional_diagonalize`,
`evaluate_hamiltonian`), `src/dxtb/_src/wavefunction/filling.py`
(`get_fermi_occupation_core`, density/occupation response and entropy), the
`storch.eighb` generalized eigensolver path, and these tests:
`test/test_scf/test_functional_map.py`,
`test/test_scf/test_fermi_derivatives.py`,
`test/test_wavefunction/test_functional_filling.py`,
`test/test_wavefunction/test_fermi_derivatives.py`,
`test/test_wavefunction/test_filling.py`,
`test/test_baseline/components.py`,
`test/test_baseline/test_derivatives.py`, and
`test/test_baseline/test_reference.py`. Do not include DFT-D3/tad-dftd3
dispersion in this preflight.

Use the repository's `test/molecules.py::samples["NO2"]` geometry unchanged,
GFN2 with the existing open-shell `spin=1` setup recorded in
`test/test_baseline/molecules.py::SYSTEMS["NO2"]`, and the T0.2/T0.3 baseline
property configuration. For the functional production-path comparison, record
the explicit setup as FULL SCF, potential representation, non-strict configured
Broyden (effective Anderson), and 300 K; retain the T0.3 driver/mode cell
settings for the isolated comparisons. Add a zero-T spectral/filling case where
the occupations are defined. Retain a GFN1 comparison around the same converged
state. First test Fermi/occupation response with frozen NO2-representative
electron counts, spin channels, temperatures, and orbital-energy spectra;
then test the common spectral-density primitive with frozen Hamiltonian and
overlap; then one pure SCF map at a fixed converged input; only then compare
the full fixed point. Include zero-T and finite-T cases where defined,
exact/near degeneracies, and gauge-invariant density, occupation, and energy
outputs. Test reverse and forward first derivatives plus nested second and
third derivatives on the affected smooth paths, and compare against central
finite differences over a double-precision step sweep from `1e-2` through
`1e-5`, recording the stable region and truncation/roundoff behavior.

E3.0 is complete when it records the first layer that fails, distinguishes
Fermi, eigensolver, density, pure-map, and full-fixed-point causes, and gives
a reproducible diagnosis. A narrowly isolated spectral/eigensolver fix may
land in E3.0 only when a regression identifies that primitive as the cause;
full SCF implicit/unrolled differentiation remains E3 work. Do not change the
`KNOWN_WRONG` NO2 entry, references, or tolerances in E3.0. If a fix makes the
stored result agree with nested derivatives and a stable finite-difference
region, record that evidence and leave marker/reference maintenance to an
explicit follow-up.

**Needs.** E0, E4core, T0.2/T0.3/T0.4 baselines, and the completed E1.1/E1.2
primitives used by the selected non-DFT-D3 path. It does not require E1.3 or
full E1 completion. The overview DAG names this milestone E3.0 and makes each
completed prerequisite explicit. **Unblocks.** Full E3 diagnosis/implementation.
**Size.** M.

### Full E3 implementation and acceptance

**Goal.** Implement the strategy chosen in E0, so SCF results are differentiable to third order, in forward mode and under `vmap`.

**Steps (assuming E0 chooses torch.func-native implicit differentiation; adjust otherwise).**

1. A fixed-point solve function taking the pure SCF step (E4core) and returning the converged state, with a custom autograd function implementing the implicit function theorem: backward solves the adjoint linear system; `jvp` solves the tangent system; vmap rule batches both.
2. Higher orders: the backward and `jvp` use only differentiable operations, so nesting works; verify orders 2 and 3 against T0.3.
3. Convergence settings for derivatives (linear-solver tolerance) exposed in configuration.
4. Remove the vendored xitorch if no longer used.

**Done when.** All T0.3 SCF cells pass at orders 1–3 for GFN1 and GFN2 (with the driver chosen in E0); forward mode works; `vmap` over conformers works.

**Needs.** E3.0, E0, full E1 (including E1.3), and E4core. **Unblocks.** E8,
F2d, F3, D2. **Size.** L.

---

## E4core Pure SCF step and masked convergence loop

**Note.** The currently named `scf/pure` functions are not the final pure design because `_Data` is mutable. E4core requires functional SCF state and functional mixer state. E4core is the accepted single-System implementation milestone; E4.6a/E4.6b are separate legacy compatibility cleanup packages and are not prerequisites of E4core or its consumers.

**Goal.** The functional single-System SCF iteration is a pure function of its
state and uses fixed-shape solver control. Legacy production batch culling
remains isolated until E6b retires that route and E4.6a removes its
compatibility machinery after E6c1 W2/W3 and batch-critical E7 clearances.

**Steps.**

1. Define the SCF state (charges or potential, mixer history) and the step `state -> state` as a pure function of system, call inputs and state.
2. Fixed-trip transform controller: use per-system convergence flags and
   freeze converged entries with `torch.where`; do not use tensor-dependent
   Python control flow.
3. Mixer state per system, part of the step's state (so it can be vmapped) or handled in the driver.
4. Keep the functional core free of cull/restore calls. Do not remove those
   methods from a shared legacy adapter while production batch FULL still uses
   it; that deletion is E4.6a after E6b retires the production batch route
   and the E6c1 W2/W3 plus batch-critical E7 gates pass.
5. Benchmark fixed-trip masking against eager early-stop and legacy batch
   culling on the relevant T0.8 workloads.

**Done when.** The single-System functional path is mutation-free, matches
T0.2, and its fixed-trip solver passes transform checks. This does not require
repository-wide culling deletion or migration of legacy production batches.

**Needs.** T0.2, B5, B6a-SCF. **Unblocks.** E3.0, E6a after its transform-safe
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
conformer function. C8a Result pytree support is a separate prerequisite, so
E6a does not mix result-container migration into the physics change.

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
   Python loop, with vmap fallback warnings as errors. Mixed padded
   compositions belong to E6b.
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

**Needs.** E1, E2, E4core, B6b-core, C8a Result pytree representation, T0.8,
the B5.6 public solver-status schema, and the private fixed-trip boundary
under the B1 public method. It does not need C5/C8b mixed-System stackability
or E4.6a. **Size.** M.

---

## E6b Mixed-System production batching through `vmap`

**Goal.** All batching goes through `vmap`; `batch_mode` disappears (P2). Integral and SCF code do not handle `batch_mode` themselves; the only final batching composition is single-System `vmap` plus stacked padded Systems.

**Steps.**

1. Mixed sets: per-molecule `setup`, padding to size buckets, stacking with a tree map, `vmap` with `in_dims=(0, 0)` (stacking needs C8b).
2. A pad-and-stack helper, with configurable bucket sizes.
3. Batch entry points from B1 (no shape-based guessing).
4. Delete `batch_mode`, the per-entry loops in driver setup (`integral/driver/pytorch/driver.py`, `libcint/driver.py`) and batched/unbatched code branches (for example, the two einsum strings in `BaseIntegral.normalize`).
5. Batched GFN2 runs through the PyTorch multipole integrals (E5); a libcint path is a plain loop over systems.

**Done when.** The mixed-System implementation composes under stack/vmap;
T0.2 batched references are reproduced and per-System Hessians have no
cross-system blocks. E6b owns batch construction, composition, production
routing, and retirement of the legacy batch FULL route. Keep implementation
and route activation distinct: implementation can be completed and measured
first; the route becomes authoritative only after E6c1's W2/W3 clearances and
the batch-critical E7 gate.

**Needs.** E6a, B3b, E4core, C5/C8b, and defined padding. E7 owns padding
validation; only its batch-critical subset gates route activation. The broader
E7 audit may continue afterward.
**Unblocks.** E6c1 W3 measurement and batch-critical E7 validation on the new
route; after those gates and E6c1 W2/W3 clearance, E6broute may become
authoritative. E6bimpl alone does not unblock E4.6a or Track D3. **Size.** L.

---

## E7 Padding-safe masking audit

**Goal.** Derivatives stay finite to third order on padded systems (P8).

**Steps.**

1. Inventory every masked singular operation in the per-call path (`1/r`, `sqrt`, `exp` of large arguments, divisions by counts).
2. Apply double-`where`: mask the input before the singular operation, mask the output after.
3. Tests: padded systems in T0.3 finite to order 3, including under `vmap`.

E7 owns padding-safety validation and hardening; E6b owns batch-route
migration. E6b requires only the E7 checks for padded operations used by its
production route, not completion of every eventual E7 audit item.

**Done when.** T0.3 padded cells finite at every order that passes for single systems.

**Needs.** T0.3. **Size.** M. The E6b production route is gated by the
batch-critical E7 subset, not by the entire eventual audit.

## E6c1 Pre-retirement performance gates

E6c1 is the explicit performance clearance required before E6b retires the
shared legacy production batch route. It includes both batching branches:

- **E6a/W2 conformers:** same-System positions mapped with the exact B1
  `vmap(system.singlepoint)` call;
- **E6b/W3 mixed Systems:** padded/stacked compositions compared with the
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
E6b value/derivative correctness are also hard prerequisites. Without W2,
W3, and E7 clearance, keep the legacy batch route. A broadly unacceptable
result blocks retirement until explicit review; it cannot silently authorize
a second permanent batched physics core.

An architecture-review waiver permits route retirement only. It does not
satisfy G2's conformer or mixed-System throughput target. E6c2/G2 remains open
until the documented performance target is met or its acceptance criterion is
explicitly amended; missing measurements are never clearance.

## E6c2 Final performance acceptance

Record the complete W2 and W3 result, including the C8b training-tree
parameter-gradient timing when available, together with GPU throughput and
memory against the deferred T0.8 GPU baseline. E6c2 is the final G2
performance sign-off; it may follow route migration because CPU W2/W3
clearances were already required by E6c1. A missing GPU baseline blocks final
GPU/G2 performance acceptance, not the CPU correctness decision or the E6b
route migration after E6c1 review.

**Needs.** E6c1, E6broute, T0.8 W2/W3, and the T0.8 GPU baseline for GPU
acceptance. **Size.** M.

---

## E8 Third-order acceptance (G1)

**Goal.** End-to-end validation of goal G1.

**Steps.**

1. Third-order force constants for GFN1 and GFN2 on the T0.2 set: forward-over-reverse against finite differences of Hessians.
2. First hyperpolarizability: autograd against finite differences of polarizabilities.
3. Raman (polarizability derivatives) and IR (dipole derivatives) against T0.2.
4. Cost: time and memory against T0.8 W4/W5; record whether compiling (D2) is worth pursuing.

**Done when.** All within T0.3 tolerances; results added to the reference data.

**Needs.** E1, E2, E3, B8. **Size.** M.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| E0 | Decision note: SCF differentiation (E0a/E0b/E0c) | T0.3, T0.4 | M |
| E1 | Custom autograd functions | T0.4 | M |
| E2 | Fixed-shape pair kernels (from PairPlan) | T0.2, B3a, B4 | L |
| E3.0 | Spectral derivative / NO2 preflight and invariant eigensolver audit | E0, E4core, T0.2/T0.3/T0.4, E1.1/E1.2 supported subset | M |
| E3 | Full SCF differentiation and GFN1/GFN2 transform acceptance | E3.0, E0, full E1, E4core | L |
| E4core | Pure single-System SCF path, fixed-trip control and tensor status | T0.2, B5, B6a-SCF | L |
| E5 | PyTorch multipole integrals (done, #270) | - | - |
| B5.6 | Public tensor-valued solver-status Result schema | B5, E4core | S |
| C8a | Result pytree output via tensor-built probe | B5, B5.6, C1 | M |
| E6a | B1 conformer `vmap(system.singlepoint)` over one fixed System | E1, E2, E4core, B6b-core, B5.6, C8a, T0.8, private fixed-trip boundary | M |
| E6bimpl | Mixed-System construction, stack/vmap, correctness | E6a, B3b, E4core, C5/C8b, defined padding | L; unblocks E6c1 W3 and batch-critical E7 validation |
| E6c1 | Pre-retirement W2 conformer and W3 mixed-System performance gates | E6a/W2, E6bimpl/W3 | M |
| E6broute | Authoritative mixed-System production route; retire legacy batch FULL | E6bimpl, E6c1 W2/W3 clearances, batch-critical E7 gate | M; unblocks E4.6a and Track D3 |
| E6c2 | Final W2/W3/GPU G2 performance acceptance | E6c1, E6broute, T0.8 GPU | M |
| E4.6a | Remove legacy batch cull/restore after batch fallback retirement | E6broute, E6c1 W2/W3, batch-critical E7 | M |
| E4.6b | Delete remaining legacy FULL loop after all fallback owners retire | E4.6a, fallback migrations/policy decisions | M |
| E7 | Padding-safety validation; batch-critical subset gates E6b production routing | T0.3 | M |
| E8 | Third-order acceptance | E1, E2, E3, B8a, B8b | M |
