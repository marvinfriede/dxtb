# Track E: Differentiable core

**Purpose.** Make the per-call path differentiable to third order (reverse and forward mode), batchable with `vmap`, and free of data-dependent shapes. This track carries goals G1 and G2.

**Starting point (at `46af7bc`).**

- Four custom autograd functions, none with a `jvp` rule; two with `generate_vmap_rule` (`RepulsionAG`, `OverlapAG`).
- SCF: default mode unrolls all iterations (`SCF_MODE_FULL`); implicit modes use a vendored copy of xitorch (`scf/implicit/base.py`, `BaseXSCF(BaseSCF, xt.EditableModule)`).
- GFN2 needs dipole and quadrupole integrals; the PyTorch driver raises `NotImplementedError` for them (`integral/driver/pytorch/dipole.py`, `quadrupole.py`), so GFN2 runs on libcint (the default driver), which is CPU-only and a C extension.
- The PyTorch overlap groups shell pairs with a geometry-dependent mask (`unique_shell_pairs`) and writes every pair in a Python loop (`integral/driver/pytorch/impls/overlap.py:192–229`).
- Batched SCF culls converged systems in place (`IndexHelper.cull`/`restore`, cache `cull`/`restore`).

---

## E0 Decision note: SCF differentiation and integrals

**Goal.** Decide, from measurements, how the SCF is differentiated and whether PyTorch multipole integrals are needed.

**Investigations (using the T0.3/T0.4 harness).**

1. **Unrolled SCF.** Accuracy of orders 1–3 as a function of the convergence threshold; memory growth with iteration count.
2. **Vendored xitorch implicit SCF.** Does it support second and third order? Does it work under `vmap`? Does `EditableModule` interfere with `torch.func`?
3. **torch.func-native implicit differentiation.** Prototype on a toy fixed-point problem: a custom autograd function (with `setup_context`, `jvp` and vmap rules) that solves the linear system of the implicit function theorem; check orders 1–3 and `vmap`.
4. **libcint (via tad-libcint).** Highest derivative order available for overlap and multipole integrals; forward mode; `vmap`; GPU. Record what dxtb calls today (`int1e("ipovlp", ...)` for the overlap gradient).
5. **Eigensolver.** Behaviour of third derivatives near degeneracies, with and without Fermi smearing.

**Decisions to record.**

- SCF differentiation strategy for each order (unrolled, implicit, or mixed: implicit for the fixed point, unrolled where cheaper).
- Whether xitorch is replaced.
- Whether E5 (PyTorch multipole integrals) is needed, and for which goals (higher order, GPU, `vmap`).

**Done when.** The note is agreed.

**Needs.** T0.3, T0.4. **Unblocks.** E3, E5. **Size.** M.

---

## E1 Custom autograd functions

**Goal.** Every custom autograd function supports reverse mode to third order, forward mode and `vmap`; or is deleted.

| Function | Location | Today |
| --- | --- | --- |
| `RepulsionAG` | `components/classicals/repulsion/rep.py:145` | `setup_context`, `generate_vmap_rule`, no `jvp` |
| `CoulombMatrixAG` | `components/interactions/coulomb/secondorder.py:930` | no `jvp`, no vmap rule |
| `OverlapAG` | `integral/driver/pytorch/impls/overlap.py:40` | `setup_context`, `generate_vmap_rule`, no `jvp` |
| `EFunction` | `integral/driver/pytorch/impls/md/recursion.py:227` | no `jvp`, no vmap rule |

**Steps per function.**

1. Measure plain-torch performance (autograd through the forward code) against the custom function, for first-order cost and memory (T0.8 workloads).
2. If plain torch is within an agreed margin: delete the custom function.
3. Otherwise: convert to the `setup_context` form; add `jvp`; make `backward` use only differentiable operations; add a vmap rule (`generate_vmap_rule = True` where the body is pure torch).
4. Tests: orders 1–3 by `gradcheck` chains, forward mode, `vmap` with fallback warnings as errors (from T0.4).

**Done when.** All four pass T0.4's checks at orders 1–3, forward mode and `vmap`, or are deleted.

**Needs.** T0.4. **Unblocks.** E3, E6a, E8, D2. **Size.** M.

---

## E2 Fixed-shape pair kernels

**Goal.** Integral and Hamiltonian pair computations with shapes fixed by the composition, no per-pair Python loops, and no geometry-dependent grouping.

**Today.** `overlap()` calls `bas.unique_shell_pairs(mask=...)`, where the mask comes from a distance cutoff; it then calls `overlap_gto` once per unique pair type and writes each pair's block in a Python loop (`for r, pair in enumerate(upairs)`), one slice assignment and one autograd node per shell pair.

**Steps.**

1. Group shell pairs by angular-momentum combination (at most 10 for s–f), not by species pair.
2. Build the group index lists in `setup` from `numbers` alone (B3).
3. Gather each pair's exponents and coefficients, padded to the maximum number of primitives.
4. Apply the distance cutoff as a multiplicative mask (with double-`where`, P8), not as a shape decision.
5. Write each group's blocks with one scatter (`index_put`/`scatter`) instead of per-pair assignments.
6. Apply the same pattern to the gradient path, the multipole integrals (E5, if built) and the H0 build if it follows the same structure.
7. Delete `unique_shell_pairs`, `get_pairs`, `get_subblock_start` and the per-pair loop; replace the in-place `fill_diagonal_` with a pure operation.

**Tests.** T0.2 overlap and H0 values reproduced; T0.4 checks at orders 1–3; `vmap` over positions with no fallbacks; profile against T0.8 W1 and W2.

**Done when.** No Python loop over pairs; no data-dependent operation in the per-call overlap path; reference reproduced.

**Needs.** T0.2, B3 (index lists in setup), B4. **Unblocks.** E6a, E5, D1. **Size.** L.

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

**Goal.** The SCF iteration is a pure function of its state; the convergence loop runs outside it and handles batches by masking, not culling.

**Steps.**

1. Define the SCF state (charges or potential, mixer history) and the step `state -> state` as a pure function of system, call inputs and state.
2. Driver loop: run the step on the whole batch; per-system convergence flags; converged entries frozen with `torch.where`; stop when all have converged or the iteration limit is reached.
3. Mixer state per system, part of the step's state (so it can be vmapped) or handled in the driver.
4. Delete `IndexHelper.cull`/`restore`, the cache `cull`/`restore` methods and their `__store` slots.
5. Benchmark masking against today's culling on T0.8 W2 and W3 (masking wastes compute on converged systems).

**Done when.** No in-place culling remains; T0.2 reproduced, including batched cases; benchmark recorded.

**Needs.** T0.2, B5, B6. **Unblocks.** E3, E6a, C5, D1, F4. **Size.** L.

---

## E5 PyTorch dipole and quadrupole integrals (conditional)

**Goal.** Remove libcint from GFN2's critical path, if E0 shows libcint blocks higher-order derivatives, GPU execution or `vmap`.

**Steps.**

1. Implement dipole and quadrupole integrals in the McMurchie–Davidson code (which already supports up to f shells), using the fixed-shape pair structure from E2.
2. Validate against libcint values for the T0.2 set.
3. Make the PyTorch driver usable for GFN2; decide the default driver per method.

**Done when.** GFN2 runs entirely in PyTorch with T0.2 reproduced, and T0.3 GFN2 cells pass with the PyTorch driver.

**Needs.** E0, E2. **Unblocks.** D1 for GFN2 on GPU. **Size.** L.

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

**Goal.** All batching goes through `vmap`; `batch_mode` disappears (P2).

**Steps.**

1. Conformers: one system, `vmap` over positions (`in_dims=(None, 0)`).
2. Mixed sets: per-molecule `setup`, padding to size buckets, stacking with a tree map, `vmap` with `in_dims=(0, 0)` (stacking needs C8).
3. A pad-and-stack helper, with configurable bucket sizes.
4. Batch entry points from B1 (no shape-based guessing).
5. Delete `batch_mode`, the per-entry loops in driver setup (`integral/driver/pytorch/driver.py`, `libcint/driver.py`) and batched/unbatched code branches (for example, the two einsum strings in `BaseIntegral.normalize`).
6. Batched GFN2 on libcint: a plain loop over systems, or E5.

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
| E0 | Decision note: SCF differentiation, integrals | T0.3, T0.4 | M |
| E1 | Custom autograd functions | T0.4 | M |
| E2 | Fixed-shape pair kernels | T0.2, B3, B4 | L |
| E3 | SCF differentiation | E0, E1, E4 | L |
| E4 | Pure SCF step, masked loop | T0.2, B5, B6 | L |
| E5 | PyTorch multipole integrals (conditional) | E0, E2 | L |
| E6a | `vmap` versus hand-written batching | E1, E2, E4, T0.8 | M |
| E6 | Batching through `vmap` | E6a, B3, E4 (C8) | L |
| E7 | Padding-safe masking | T0.3 | M |
| E8 | Third-order acceptance | E1, E2, E3, B8 | M |
