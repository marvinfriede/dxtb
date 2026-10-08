# Track D: Performance and compiling

**Purpose.** Speed up the workloads behind G1–G3 where measurements show a benefit (G5). Nothing here changes the design; every package starts from the T0.8 profile and ends with a before/after comparison on the same workloads.

**Expected value of compiling (to be confirmed by T0.8).**

| Area | Expected gain | Reason |
| --- | --- | --- |
| SCF step, many small molecules on GPU | largest relative | launch and Python overhead dominate; a fixed-shape step can be captured as a CUDA graph |
| Integrals and H0 | high | long elementwise chains over shell pairs; fusion reduces memory traffic |
| Pairwise terms (Coulomb, multipoles, ALPB, repulsion, D3/D4) | high | memory-bound pair intermediates avoided by fusion |
| Higher-order derivatives | high, mostly memory | very large graphs of small operations |
| Training loss | medium | same kernels; recompiles per shape unless bucketed |
| `eigh`, setup, convergence loop, libcint | none | library call, one-time or data-dependent code, C extension |

**Rules.**

- Compile the SCF step, not the SCF loop.
- Compile the outermost transformed function (for example, `compile(jacfwd(jacrev(energy)))`); never rely on `.backward(create_graph=True)` through a compiled region (double backward through compiled code isn't supported, as far as known).
- Fixed shapes: masking instead of culling (E4); size buckets for mixed sets (E6).

**Gates.**

- Final GPU performance decisions require completion of the missing T0.8 GPU
  baseline.
- No compile-specific architectural workaround may be introduced before the
  same function works correctly in eager mode under the required transforms.
- Compile work stays downstream of E4/E6.

---

## D1 Compiled SCF step and CUDA graphs for conformers

**Goal.** Fast conformer ensembles on GPU (G2).

**Steps.**

1. Compile the pure SCF step (E4) under `vmap` over conformers (E6).
2. Try `mode="reduce-overhead"` (CUDA graphs) and `fullgraph=True`.
3. Compile the per-call build stages (integrals, H0, Coulomb) as one function per call.
4. Compare with T0.8 W2 (eager, current code) and with eager E6.

**Done when.** Speed-up and compile time recorded; enabled by default for batch entry points if worthwhile.

**Needs.** E4, E6 (E5, done, for GFN2 on GPU), T0.8 including the GPU baseline. **Size.** M.

---

## D2 Compiled derivative functions

**Goal.** Faster and leaner Hessians, third-order force constants and hyperpolarizabilities (G1).

**Steps.**

1. Compile the property functions from B8 as whole transformed functions.
2. Measure time, peak memory and compile time against eager on T0.8 W4/W5 and the E8 cases.
3. Decide per property whether compiling is enabled by default.

**Done when.** Results recorded; defaults set.

**Needs.** E1, E3, B8. **Size.** M.

---

## D3 Compiled training loss with size buckets

**Goal.** Faster parameter training (G3).

**Steps.**

1. Compile `grad(loss)` per size bucket.
2. Tune bucket sizes against padding waste and number of compiled graphs.
3. Compare with eager on T0.8 W3 and the F2d workflow.

**Done when.** Results recorded; bucket defaults set.

**Needs.** F2d, E6. **Size.** M.

---

## D4 Compile support policy and tooling

**Goal.** Decide and enforce which torch versions support compiled use, once D1–D3 show it is worth supporting.

**Steps.**

1. Run the compile probe suite (from the design review) plus D1–D3 cases on each torch minor from the candidate floor upward.
2. Decide the minimum torch version for compiled use.
3. Enforce it at a boundary dxtb controls (for example, a `dxtb.compile()` wrapper checking the version before calling `torch.compile`), not inside traced code.
4. CI: full suite at the eager floor and newest torch; compile probe suite on every supported minor.

**Done when.** Policy documented; wrapper and CI in place.

**Needs.** D1. **Size.** S.

---

## D5 Concatenated batching for pair stages (conditional)

**Goal.** Remove padding waste for very mixed training sets, if E6a or D3 show it matters.

**Steps.**

1. For pair stages only (integrals, Coulomb, dispersion): concatenate all atoms and pairs of all molecules with segment indices, grouped by angular-momentum combination.
2. Feed padded dense blocks to the SCF and `eigh`.
3. Compare with stacked padded `vmap` on T0.8 W3.

**Done when.** Measured; adopted only if clearly better.

**Needs.** E6, D3. **Size.** L. **Start only if** padding waste is shown to dominate.

---

## D6 Eigensolver for large systems (conditional)

**Goal.** Address `eigh` cost if T0.8 W1 shows it dominates the large-system workloads that matter.

**Options to evaluate.** Batched solvers for many small matrices on GPU (already used by torch for batched input), mixed strategies for large matrices, reuse of previous eigenvectors as starting guesses. Compiling doesn't help here.

**Done when.** Evaluation recorded; adopted only with measured benefit and unchanged T0.3 status.

**Needs.** T0.8. **Size.** M. **Start only if** `eigh` dominates relevant workloads.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| D1 | Compiled SCF step and CUDA graphs | E4, E6 (E5), T0.8 | M |
| D2 | Compiled derivative functions | E1, E3, B8 | M |
| D3 | Compiled training loss with buckets | F2d, E6 | M |
| D4 | Compile policy and tooling | D1 | S |
| D5 | Concatenated pair batching (conditional) | E6, D3 | L |
| D6 | Eigensolver (conditional) | T0.8 | M |
