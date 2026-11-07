# Grouped-vmap heterogeneous batching

Status: **current architecture** for batched `Calculator.singlepoint` and
`evaluate_packed_systems`. It replaces the canonical cross-composition padding
route (C8b.3, E6bimpl.0–2, E7 padding safety for mixed buckets). Documents
describing that route are historical.

## Contract

```text
Public heterogeneous batching  !=  one heterogeneous vmap
```

1. Batched input (`numbers` `(B, Nmax)`, trailing or non-contiguous zero
   padding) is split into ordinary single-System lanes.
   `build_calculator_lane_systems` deduplicates identical lane compositions, so
   repeated compositions and conformers share one immutable `System` object
   and one `Model.setup` call per call.
2. `group_lanes` (`dxtb._src.calculators.lane_evaluation`) partitions lanes:
   - lanes that reference **the same `System` object** and agree on whether
     `spin`, `field` and `field_grad` are given form one group;
   - a group of two or more lanes whose System passes `can_vmap_system` runs
     through `torch.func.vmap` over the authoritative fixed-trip single-System
     kernel (`system_singlepoint`);
   - every other lane (groups of one, libcint, implicit differentiation,
     multi-stage/eager-only strategies, classical extensions without exact
     setup, SCF-free calculations) is evaluated independently with the eager
     single-System route.
3. Results are restored to source order and combined by
   `pack_system_results`.

Different molecules are never padded to a common atom/shell/AO layout for
physics. Each lane keeps its own numbers, index maps, pair plan, basis, H0 and
component setup. The only cross-lane padding is the representation of
differently sized per-lane Results in the public batched `Result`.

### Grouping rule

Grouping uses `System` identity only. Separately constructed but structurally
equal Systems are *not* merged; no tensor values or setup tensors are
compared. A missed batching opportunity is acceptable; structural remapping is
not. Lanes whose `spin`, `field` or `field_grad` presence differs are separate
groups because the mapped call differs. Within a group, per-lane tensors are
stacked; a shared tensor is stacked as repeated references and autograd
accumulates its gradient across lanes.

### Eligibility and errors

`can_vmap_system` returns `False` for valid configurations that need the eager
controller and raises for configurations that are invalid on every route
(unsupported interaction extensions, negative temperature or iteration
limits, SOSCF with active smearing, eager-only strategies under unrolled
differentiation, unknown integral drivers). All lanes are classified before
any lane is evaluated. Because `vmap` cannot raise data-dependent errors,
electron-count and spin consistency is validated eagerly for every mapped lane
with the single-System `get_refocc` check; errors carry the lane's
`source_id`. Under `legacy_compat`, mapped lanes raise the eager convergence
diagnostics from their tensor status.

### Fixed-trip behavior inside a group

Mapped groups use the transform-compatible fixed-trip controller: every lane
executes `maxiter` map evaluations and converged lanes are frozen by masks.
With autograd enabled, all trips are recorded on the tape. This is unchanged
by this package; it is the main remaining cost of large homogeneous groups and
is left to a separate SCF optimization. Groups of one avoid it by using the
eager controller, which stops at convergence.

## Removed machinery

Commit `8da3ed0d` removed `calculators/canonical_padding.py`,
`calculators/padding.py`, `calculators/padded_integrals.py` and
`scf/functional/padded.py` (canonical buckets and source maps, PairPlan
reconstruction, H0/setup remapping, `PaddedExactSystemCore`,
`PaddedCoreValidity`, `PaddedCoreIndexData`, padded SCF inputs, padded
fixed-trip evaluation and Result decoding), the `pair_validity` masks in the
PyTorch overlap/multipole drivers, the `padded` flag of
`_solve_representation`, and the `PaddedExactSystemCore` branches of the
experimental converged-root batch prototype. Source lines: +469/−4005; tests:
+959/−6977 (relative to `74d94a1c`).

## Benchmark

`benchmarks/grouped_vmap_batching.py` (public batched `Calculator`, CPU only,
GFN1, PyTorch integrals, float64, `OMP_NUM_THREADS=1`, default SCF settings,
best of two runs, every case in its own subprocess with a 12 GiB resident
memory watchdog). Raw results:
`benchmarks/results/grouped-vmap-baseline-74d94a1c.json` (canonical route)
and `benchmarks/results/grouped-vmap-final.json` (grouped vmap). Total
energies agree between both routes to ~1e-14 Eh.

| Case | Lanes (System groups) | Real AO slots / canonical AO slots | Mode | Canonical (74d94a1c) | Grouped vmap |
| --- | --- | --- | --- | --- | --- |
| A conformers | 32 × H2O (32) | 256 / 256 | energy | 8.96 s, 505 MiB | 8.56 s, 369 MiB |
| | | | energy + forces | 14.84 s, 4.1 GiB | 14.20 s, 4.0 GiB |
| B repeated | 8 H2O, 8 CH4, 4 NH3 (8, 8, 4) | 200 / 400 | energy | 11.12 s, 474 MiB | 23.31 s, 374 MiB |
| | | | energy + forces | killed > 12 GiB | 35.86 s, 3.8 GiB |
| C heterogeneous | H2, H2O, NH3, CH4, SiH4, C4H5NCS (all singletons) | 94 / 336 | energy | 15.54 s, 429 MiB | 2.41 s, 383 MiB |
| | | | energy + forces | killed > 12 GiB | 4.30 s, 0.9 GiB |
| D mixed sizes | 7 × H2O + 1 × C4H5NCS (7, 1) | 99 / 376 | energy | 15.79 s, 420 MiB | 7.93 s, 368 MiB |
| | | | energy + forces | killed > 12 GiB | 12.57 s, 1.8 GiB |

Interpretation:

- Homogeneous conformers (A) are unchanged: both routes map one 32-lane group
  without padding.
- Cross-composition padding inflated the AO extent 2–4× (B–D). With autograd,
  every mapped lane records all fixed SCF trips at that inflated size, which
  exceeded 12 GiB for B, C and D; the grouped route stays below 4 GiB.
- Singletons and small groups use the eager controller or smaller mapped
  groups, which is much faster for heterogeneous batches (C: 6.4×, D: 2.0×).
- For many tiny molecules split into several groups (B, energy only) the
  grouped route is **2.1× slower**: three 100-trip mapped executions replace
  one, and per-trip Python overhead dominates at 4–12 AOs. Reducing the
  fixed-trip cost inside mapped groups is the remaining optimization.

GPU throughput was not measured (CPU-only environment; CUDA work skipped).
