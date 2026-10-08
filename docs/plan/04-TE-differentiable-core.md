# Track E: Differentiable core

**Purpose.** Make the per-call path differentiable to third order (reverse and forward mode), batchable with `vmap`, and free of data-dependent shapes. This track carries goals G1 and G2.

**Starting point.** See `07-current-state-and-execution-plan.md`.

- Remaining custom autograd functions: `CoulombMatrixAG` and optional `RepulsionAG`; none has a `jvp` rule.
- SCF: default mode unrolls all iterations (`SCF_MODE_FULL`); implicit modes use a vendored copy of xitorch (`scf/implicit/base.py`).
- GFN2 multipole integrals exist in PyTorch (E5, done).
- The PyTorch overlap uses the PairPlan builder (E2).
- Batched SCF culls converged systems in place (`IndexHelper.cull`/`restore`, cache `cull`/`restore`).

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

**Needs.** T0.3, T0.4. **Unblocks.** E3. **Size.** M.

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

**Needs.** T0.4. **Unblocks.** E3, E6a, E8, D2. **Size.** M.

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

**Goal.** Implement the strategy chosen in E0, so SCF results are differentiable to third order, in forward mode and under `vmap`.

**Steps (assuming E0 chooses torch.func-native implicit differentiation; adjust otherwise).**

1. A fixed-point solve function taking the pure SCF step (E4) and returning the converged state, with a custom autograd function implementing the implicit function theorem: backward solves the adjoint linear system; `jvp` solves the tangent system; vmap rule batches both.
2. Higher orders: the backward and `jvp` use only differentiable operations, so nesting works; verify orders 2 and 3 against T0.3.
3. Convergence settings for derivatives (linear-solver tolerance) exposed in configuration.
4. Remove the vendored xitorch if no longer used.

**Done when.** All T0.3 SCF cells pass at orders 1–3 for GFN1 and GFN2 (with the driver chosen in E0); forward mode works; `vmap` over conformers works.

**Needs.** E0, E1, E4. **Unblocks.** E8, F2d, F3, D2. **Size.** L.

---

## E4 Pure SCF step and masked convergence loop

**Note.** The currently named `scf/pure` functions are not the final pure design because `_Data` is mutable. E4 requires functional SCF state and functional mixer state.

**Goal.** The SCF iteration is a pure function of its state; the convergence loop runs outside it and handles batches by masking, not culling.

**Steps.**

1. Define the SCF state (charges or potential, mixer history) and the step `state -> state` as a pure function of system, call inputs and state.
2. Driver loop: run the step on the whole batch; per-system convergence flags; converged entries frozen with `torch.where`; stop when all have converged or the iteration limit is reached.
3. Mixer state per system, part of the step's state (so it can be vmapped) or handled in the driver.
4. Delete `IndexHelper.cull`/`restore`, the cache `cull`/`restore` methods and their `__store` slots.
5. Benchmark masking against today's culling on T0.8 W2 and W3 (masking wastes compute on converged systems).

**Done when.** No in-place culling remains; T0.2 reproduced, including batched cases; benchmark recorded.

**Needs.** T0.2, B5, B6a. **Unblocks.** E3, E6a, B6b, C5, D1, F4. **Size.** L.

---

## E5 PyTorch dipole and quadrupole integrals

**Status: done upstream (#270).** PyTorch dipole and quadrupole integral paths
exist and pass the T0 component checks through third order, forward mode and
vmap. Keep regression tests; do not reimplement.


---

## E6a Measurement: `vmap` against hand-written batching

**Goal.** Confirm that a single-system core with `vmap` matches today's batched code before deleting it.

**Steps.**

1. Port one path (GFN1 energy and forces, PyTorch driver) to a single-system core, using E1, E2 and E4.
2. Compare against `batch_mode = 2` on a real conformer set and `batch_mode = 1` on a padded mixed set (T0.8 W2, W3), with `vmap` fallback warnings as errors.
3. Record time and memory.

**Decision rule.**

- Close to today's code: proceed with E6.
- One operation slow: give it a batching rule; proceed.
- Broadly slower: keep an explicitly batched core and record the cost for higher-order derivatives (cross-system blocks).

**Done when.** Results and decision recorded.

**Needs.** E1, E2, E4, T0.8. **Size.** M.

---

## E6 Batching through `vmap`

**Goal.** All batching goes through `vmap`; `batch_mode` disappears (P2). Integral and SCF code do not handle `batch_mode` themselves; the only final batching composition is single-System `vmap` plus stacked padded Systems.

**Steps.**

1. Conformers: one system, `vmap` over positions (`in_dims=(None, 0)`).
2. Mixed sets: per-molecule `setup`, padding to size buckets, stacking with a tree map, `vmap` with `in_dims=(0, 0)` (stacking needs C8).
3. A pad-and-stack helper, with configurable bucket sizes.
4. Batch entry points from B1 (no shape-based guessing).
5. Delete `batch_mode`, the per-entry loops in driver setup (`integral/driver/pytorch/driver.py`, `libcint/driver.py`) and batched/unbatched code branches (for example, the two einsum strings in `BaseIntegral.normalize`).
6. Batched GFN2 runs through the PyTorch multipole integrals (E5); a libcint path is a plain loop over systems.

**Done when.** Both batching modes run through `vmap` with no fallbacks; T0.2 batched references reproduced; per-system Hessians computed with `vmap(hessian)` without cross-system blocks.

**Needs.** E6a, B3, E4; C8 for stacked mixed batches. **Unblocks.** D1, D3. **Size.** L.

---

## E7 Padding-safe masking audit

**Goal.** Derivatives stay finite to third order on padded systems (P8).

**Steps.**

1. Inventory every masked singular operation in the per-call path (`1/r`, `sqrt`, `exp` of large arguments, divisions by counts).
2. Apply double-`where`: mask the input before the singular operation, mask the output after.
3. Tests: padded systems in T0.3 finite to order 3, including under `vmap`.

**Done when.** T0.3 padded cells finite at every order that passes for single systems.

**Needs.** T0.3. **Size.** M.

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
| E3 | SCF differentiation | E0, E1, E4 | L |
| E4 | Pure SCF step, masked loop | T0.2, B5, B6a | L |
| E5 | PyTorch multipole integrals (done, #270) | - | - |
| E6a | `vmap` versus hand-written batching | E1, E2, E4, T0.8 | M |
| E6 | Batching through `vmap` | E6a, B3, E4 (C8) | L |
| E7 | Padding-safe masking | T0.3 | M |
| E8 | Third-order acceptance | E1, E2, E3, B8a, B8b | M |
