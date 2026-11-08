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
electron-count and spin consistency of a mapped group is validated eagerly
before the group runs:

1. one vectorized evaluation of the single-System rules for all lanes of the
   group (`reference_electron_count`, `spin_electron_violations`, shared with
   `get_refocc`);
2. one host read of the per-lane verdicts;
3. only if a lane is invalid, the first invalid lane is re-run through the
   authoritative single-System `get_refocc` check, solely to raise its exact
   exception and message; the error carries that lane's `source_id`.

Without an explicit spin no value check is needed.

**Failure ordering.** Execution groups run in the order of their first source
lane. Within a group, electron validation, mapped status inspection
(`legacy_compat`) and warnings follow source order. A failure in a group that
starts earlier is therefore raised before a failure at a smaller source id in
a later group: for interleaved lanes `A0, B1, A2, B3` with invalid `B1` and
`A2`, the error carries `source_id == 2`. The canonical bucketed route
(`74d94a1c`) had the same per-bucket ordering, and no public contract promises
the globally first failing source lane, so this is kept and pinned by
`test_interleaved_*_follows_group_order` (electron/spin input, mapped Fermi
failure, forced nonconvergence). Under `legacy_compat`, mapped lanes raise the eager convergence
diagnostics from their tensor status.

### Execution schedule inside a group

Mapped groups use the transform-compatible fixed-trip controller
(`FixedTripControl` in `scf/functional/solver.py`). It is resumable: `start`
evaluates the initial map (and the optional initial mix), `advance(state,
start, stop)` runs map trips `start <= index < stop` of the configured
schedule, and `finish` certifies the selected state. Advancing through
consecutive ranges is bitwise the same computation as one range over
`maxiter` (`test_functional_resumable_fixed_trip.py`: solver state, Result,
`record_trace=True` traces with observations carried between chunks, the
`maxiter = 0 / 1` and `record_initial_status` boundaries, and the
representation-owned charge/potential projection); the monolithic
`_solve_map_control` is `start`, `advance(0, maxiter)`, `finish`.
`fixed_trip_problem_control` is the resumable form of the standard problem
solve only (representation projection, default options); it does not expose
the caller projections of `solve_scf_problem`. There is one
transition implementation (`_solver_control_step`) for eager, static and
adaptive execution.

Two schedules use it:

- **Adaptive (default for eager and autograd calls).** The group runs one
  `vmap` call that prepares every lane (classicals, integrals, SCF inputs and
  problem) and starts its controller, then one `vmap` call per chunk of
  `_ADAPTIVE_CHUNK_TRIPS` trips, then one `vmap` call that finishes the
  controller and assembles the lane Results. After each chunk the host reads
  "is any lane active?" once for the whole group; the group stops when no
  lane is active (every lane certified or failed terminally) or when
  `maxiter` trips have run. Lanes that finish inside a chunk are frozen by the
  shared masked transition, exactly as in the full schedule, so values,
  derivatives, iterations, status, warnings and errors equal the full static
  schedule (`test_grouped_adaptive_control.py`, bitwise for Results and
  derivatives). Lane-dependent tensors cross the `vmap` calls as leaves of
  the immutable dataclass trees (prepared non-SCF terms, `SCFProblem`, solver
  state) via `tensor_leaves`/`replace_tensor_leaves`; lane-independent
  tensors (structural indices, model-parameter-derived setup) stay in the
  template unchanged. A guard rejects lane tensors held outside these trees.
- **Static (inside `torch.func` transforms and `torch.compile`).** When a
  functorch level is active (`maybe_current_level()`) or `torch.compile` is
  tracing, the group is part of a traced computation and cannot stop on
  tensor values; it keeps the single `vmap` over `system_singlepoint` with
  all `maxiter` trips.

Counts are distinct:

| Quantity | Meaning |
| --- | --- |
| configured `maxiter` | user setting; upper bound of the schedule, unchanged |
| per-lane `iterations` | physical count, 1 (initial map) + trips while the lane was active; identical in both schedules and in eager evaluation |
| mapped map evaluations | adaptive: 1 + trips run until the chunk in which the last lane finished (at most `chunk - 1` trips beyond it); static: 1 + `maxiter` |
| chunks | adaptive host checks, `ceil(trips run / chunk)` |

The chunk length is a performance parameter only; it never changes the
physical schedule or any result (`test_chunk_length_does_not_change_results`).
Groups of one keep the eager controller, which stops at convergence.

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
  one, and the per-trip map cost dominates at 4–12 AOs (the profile below
  attributes it to the SCF solve, not to integrals or the eigensolver).
  Reducing the fixed-trip cost inside mapped groups was the remaining
  optimization; see "Adaptive grouped SCF" below.

GPU throughput was not measured (CPU-only environment; CUDA work skipped).

## Post-cutover hardening

Package on top of `17b2bfe4`: dispatcher host-synchronization audit,
dead converged-root padding branches, broad-suite memory-test diagnosis and
vmap-vs-eager characterization. The architecture above is unchanged.

### Structural host decisions (CUDA synchronization audit)

No CUDA device was available; every source below was removed by inspection
and checked on CPU with a proxy that counts Python-level host decisions
(`item`, `tolist`, `bool`/`int`/`float` of tensors, `torch.equal`,
boolean-mask indexing) through the public batched `Calculator`
(`test_grouped_dispatch_hardening.py`).

| Source (17b2bfe4) | Frequency | Replacement |
| --- | --- | --- |
| `int(x.item())` over `numbers` rows in `build_calculator_lane_systems` | per atom per lane | one `.cpu().tolist()` of `numbers` (and of each component-owned 2D `numbers`); keys built on the host; device tensors still used for `Model.setup` |
| `can_vmap_system` in `group_lanes` and again in the independent evaluator | per lane, twice for singletons | once per distinct System per call (call-local `id` cache); private prevalidated evaluator for groups from `group_lanes` |
| `_validate_electrons` (`get_refocc` + aufbau `.item()`) | per mapped lane | one vectorized check per group with the shared single-System rules (`reference_electron_count`, `spin_electron_violations`) and one host transfer; `spin=None` needs no value check; first invalid source lane re-raises the exact eager error |
| `_enforce_eager_status_policy` `bool(...)` checks | up to 3 per mapped lane | one stacked `.tolist()` per group, eager precedence applied per lane on the host |
| `_is_prefix_mask` (`.item()` + `torch.equal`) | per lane per atom-resolved Result field | atom masks resolved once to index placements (one D2H, one H2D); one `pad + index_select` gather places prefix and non-contiguous layouts alike |
| `positions[idx][atom_mask]` | per lane | `index_select` with the resolved placement |
| `torch.tensor(float(chrg))` for Python scalar charge/spin | per lane | one tensor per device/dtype |

Public conformer call, explicit spin, non-contiguous padding: counted host
decisions grew from 328 (B=2) to 604 (B=8) for H2O at `17b2bfe4`; they are
243 for both B=2 and B=8 now (`torch.equal`: 16 → 64 before, 0 now;
boolean-mask indexing: 4 → 16 before, 1 per distinct composition now). The
remaining constant part is `Model.setup` of the one distinct System.
Deliberate synchronization is unchanged: under `legacy_compat` on CUDA,
`cuda_sync_in_scf` defaults to `True` and synchronizes the SCF timer.

### Converged-root padding branches

`batch_implicit` contained `lane_padded` projections, a `padding_valid`
check (`BatchRootFailureCode.INVALID_PADDING = 8`) and a padded potential
projection in the response path. After `PaddedExactSystemCore` was removed
they were unreachable: `prepare_converged_root_batch` takes a
`batch_mode=0` System; a zero atomic number fails integral setup (pytorch
driver: `Model.setup`; libcint: `prepare_converged_root_batch`, both
`CGTOPrimitivesError`) before any branch; `SCFInputs.orbital_mask` is
`orbitals_per_shell > 0` and therefore all-True for every real shell
(probed for GFN1/GFN2, zero slot leading, inner and trailing). The branches
and `eager._padded_*_projections` were removed and code 8 retired (other
codes keep their values). The structural PADNZ potential projection is a
different mechanism and stays.

### Broad-suite memory-test discrepancy

`17b2bfe4` broad run (`-n 8`, seed 4711) failed two tests more than the
baseline: `test_a_memory_leak/test_scf.py::test_functional_implicit[False-False-dtype1]`
(worker gw7, 24th test of the worker) and
`::test_functional_implicit_energy[False-False-dtype1]` (gw1, 571st). Both
are single-System LiH tests (no packed-route call; checked with a spy) with
`run_gc=False`, i.e. no `gc.collect()` between measured repeats. A verbose
rerun reproduced the identical failure set. Controlled single-worker
reproductions (`--random-order` off, both revisions):

| Sequence | 74d94a1c | 17b2bfe4 |
| --- | --- | --- |
| target alone (command line) | pass | pass |
| target alone after full-suite collection | pass | pass |
| 20 repeated calls in one process (both tests) | 20/20 pass | 20/20 pass |
| after the batching modules (canonical modules / grouped module) | pass | pass |
| after packed lifecycle + differential modules | pass | pass |
| gw7 history (23 tests) on the command line | pass | pass |
| gw7 history after full-suite collection | **fail** | **fail** |
| gw1 history (570 tests; 4 grouped-only nodes absent at 74d94a1c) after full-suite collection | **fail** | **fail** |

No single gw7 predecessor reproduces the failure. Instrumenting the check
in the failing state shows, identically on both revisions, +14 tensors
(0-d `bool`, 13 bytes) after the first measured repeat, all reclaimed by a
single `gc.collect()`; the automatic collector ran 102/9/1 generation
collections during that call. The test flags unreachable cyclic garbage
whose collection time depends on process history (here: which test modules
were imported during collection). It is not retained memory and not a
regression. The final hardening broad run (`88e812d6`, same command) no
longer fails `test_functional_implicit` (the collection set changed) and
still fails `test_functional_implicit_energy`; replaying its worker history
(gw0, 372 predecessors) at the final revision shows the same signature:
two flat repeats, then +14 0-d `bool` tensors that a single `gc.collect()`
reclaims. Repeated grouped evaluation (H2O conformer group; H2O + CH4 +
H2O mixed batch; energy, forces, forces with `create_graph=True`) reaches a
plateau on both revisions; without `gc` the mixed energy case holds a
constant +112 tensors of uncollected garbage. The new
`test_a_memory_leak/test_grouped_batching.py` therefore measures with
`gccollect=True` only.

### Grouped vmap versus independent eager lanes (CPU)

`benchmarks/vmap_vs_eager_groups.py`: one homogeneous group (one shared
System, B distorted conformers) evaluated either as one mapped group
(fixed-trip controller, default `maxiter = 100`) or lane by lane with the
eager single-System route (stops at convergence). GFN1, PyTorch integrals,
float64, `OMP_NUM_THREADS=1`, best of two, each case in its own process,
12 GiB RSS watchdog, nothing else running. Cells: wall time / peak RSS of
the process. `iter` is the physical convergence iteration (all lanes
converged in every case). `energy` runs under `no_grad`; `energy_ad` with
positions requiring grad but no backward; `forces` adds the position
gradient; `param` the gradient of a shared `element.<heavy>.gam`. Raw data:
`benchmarks/results/vmap-vs-eager-cpu.json`.

| Molecule (nat/nao, iter) | Mode | Route | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2O (3/8, 9) | energy | vmap | 7.12 s / 0.4 G | 7.15 s / 0.4 G | 7.21 s / 0.4 G | 7.34 s / 0.4 G | 7.75 s / 0.4 G | 8.11 s / 0.4 G |
|  |  | eager | 0.34 s / 0.4 G | 0.69 s / 0.4 G | 1.38 s / 0.4 G | 2.72 s / 0.4 G | 5.43 s / 0.4 G | 10.81 s / 0.4 G |
| H2O (3/8, 9) | energy_ad | vmap | 7.78 s / 0.7 G | 7.80 s / 0.7 G | 7.76 s / 0.8 G | 8.17 s / 1.0 G | 8.30 s / 1.5 G | 9.03 s / 2.7 G |
|  |  | eager | 0.37 s / 0.4 G | 0.74 s / 0.4 G | 1.58 s / 0.4 G | 3.09 s / 0.5 G | 5.92 s / 0.7 G | 12.36 s / 1.1 G |
| H2O (3/8, 9) | forces | vmap | 10.44 s / 0.7 G | 10.27 s / 0.7 G | 10.52 s / 0.8 G | 10.76 s / 1.0 G | 11.51 s / 1.6 G | 12.73 s / 2.8 G |
|  |  | eager | 0.55 s / 0.4 G | 1.10 s / 0.4 G | 2.33 s / 0.5 G | 4.66 s / 0.6 G | 9.32 s / 0.8 G | 19.32 s / 1.2 G |
| H2O (3/8, 9) | param | vmap | 10.32 s / 0.7 G | 11.11 s / 0.7 G | 10.90 s / 0.8 G | 11.02 s / 1.0 G | 11.75 s / 1.5 G | 13.20 s / 2.7 G |
|  |  | eager | 0.56 s / 0.4 G | 1.14 s / 0.4 G | 2.43 s / 0.5 G | 4.79 s / 0.6 G | 9.52 s / 0.8 G | 19.33 s / 1.2 G |
| CH4 (5/12, 10) | energy | vmap | 7.26 s / 0.4 G | 7.48 s / 0.4 G | 7.48 s / 0.4 G | 7.85 s / 0.4 G | 8.08 s / 0.4 G | 9.04 s / 0.4 G |
|  |  | eager | 0.40 s / 0.4 G | 0.75 s / 0.4 G | 1.52 s / 0.4 G | 2.99 s / 0.4 G | 5.84 s / 0.4 G | 12.03 s / 0.4 G |
| CH4 (5/12, 10) | energy_ad | vmap | 7.93 s / 0.7 G | 8.00 s / 0.8 G | 8.05 s / 1.0 G | 8.35 s / 1.7 G | 9.22 s / 3.0 G | 10.69 s / 5.7 G |
|  |  | eager | 0.43 s / 0.4 G | 0.81 s / 0.4 G | 1.72 s / 0.5 G | 3.33 s / 0.6 G | 6.77 s / 0.8 G | 13.65 s / 1.4 G |
| CH4 (5/12, 10) | forces | vmap | 10.74 s / 0.8 G | 10.99 s / 0.9 G | 10.97 s / 1.0 G | 11.30 s / 1.7 G | 12.42 s / 3.0 G | 14.00 s / 5.7 G |
|  |  | eager | 0.63 s / 0.4 G | 1.24 s / 0.4 G | 2.67 s / 0.5 G | 5.10 s / 0.6 G | 10.42 s / 0.9 G | 21.74 s / 1.5 G |
| CH4 (5/12, 10) | param | vmap | 10.58 s / 0.8 G | 11.04 s / 0.8 G | 10.86 s / 1.0 G | 11.60 s / 1.7 G | 12.48 s / 3.0 G | 15.05 s / 5.7 G |
|  |  | eager | 0.63 s / 0.4 G | 1.23 s / 0.4 G | 2.62 s / 0.5 G | 5.04 s / 0.6 G | 10.24 s / 0.9 G | 21.29 s / 1.4 G |
| SiH4 (5/17, 10) | energy | vmap | 7.89 s / 0.4 G | 8.00 s / 0.4 G | 7.99 s / 0.4 G | 8.69 s / 0.4 G | 9.84 s / 0.4 G | 11.39 s / 0.4 G |
|  |  | eager | 0.40 s / 0.4 G | 0.82 s / 0.4 G | 1.65 s / 0.4 G | 3.27 s / 0.4 G | 6.57 s / 0.4 G | 13.30 s / 0.4 G |
| SiH4 (5/17, 10) | energy_ad | vmap | 8.29 s / 0.8 G | 8.12 s / 1.0 G | 8.73 s / 1.7 G | 9.43 s / 3.0 G | 11.35 s / 5.7 G | 14.93 s / 11.1 G |
|  |  | eager | 0.44 s / 0.4 G | 0.86 s / 0.4 G | 1.82 s / 0.5 G | 3.56 s / 0.7 G | 7.34 s / 1.1 G | 14.43 s / 1.8 G |
| SiH4 (5/17, 10) | forces | vmap | 10.95 s / 0.9 G | 11.24 s / 1.0 G | 11.78 s / 1.7 G | 13.11 s / 3.0 G | 15.22 s / 5.7 G | 22.46 s / 11.1 G |
|  |  | eager | 0.68 s / 0.4 G | 1.37 s / 0.4 G | 2.85 s / 0.5 G | 5.61 s / 0.7 G | 11.45 s / 1.1 G | 23.28 s / 1.9 G |
| SiH4 (5/17, 10) | param | vmap | 10.91 s / 0.8 G | 11.22 s / 1.0 G | 11.62 s / 1.7 G | 13.24 s / 3.0 G | 16.21 s / 5.7 G | 21.99 s / 11.0 G |
|  |  | eager | 0.67 s / 0.4 G | 1.31 s / 0.4 G | 2.89 s / 0.5 G | 5.68 s / 0.7 G | 11.29 s / 1.1 G | 22.95 s / 1.8 G |
| C4H5NCS (12/43, 14) | energy | vmap | 8.22 s / 0.4 G | 9.05 s / 0.4 G | 10.59 s / 0.4 G | 13.90 s / 0.4 G | 19.98 s / 0.4 G | 32.01 s / 0.4 G |
|  |  | eager | 0.66 s / 0.4 G | 1.30 s / 0.4 G | 2.67 s / 0.4 G | 5.31 s / 0.4 G | 10.05 s / 0.4 G | 20.59 s / 0.4 G |
| C4H5NCS (12/43, 14) | energy_ad | vmap | 8.99 s / 2.5 G | 9.94 s / 3.1 G | 13.95 s / 8.9 G | killed | killed | killed |
|  |  | eager | 0.81 s / 0.7 G | 1.63 s / 1.1 G | 3.16 s / 1.8 G | 6.11 s / 3.2 G | 11.75 s / 5.9 G | 23.26 s / 11.4 G |
| C4H5NCS (12/43, 14) | forces | vmap | 12.74 s / 2.5 G | 14.68 s / 3.1 G | 18.72 s / 8.9 G | killed | killed | killed |
|  |  | eager | 1.22 s / 0.7 G | 2.53 s / 1.1 G | 5.22 s / 1.8 G | 10.34 s / 3.2 G | 19.99 s / 5.9 G | 39.47 s / 11.5 G |
| C4H5NCS (12/43, 14) | param | vmap | 12.54 s / 2.5 G | 14.43 s / 3.1 G | 20.51 s / 8.8 G | killed | killed | killed |
|  |  | eager | 1.18 s / 0.7 G | 2.60 s / 1.1 G | 5.13 s / 1.8 G | 10.07 s / 3.2 G | 20.15 s / 5.9 G | 40.99 s / 11.4 G |

On CPU, independent eager evaluation is faster for every group size up to
16, for every molecule and mode. The mapped group wins only at B = 32 for the
small molecules (1.1–1.5×) and never for C4H5NCS, whose mapped autograd runs
exceed 12 GiB from B = 8 (eager: 3.2 GiB at B = 8). With autograd the mapped
tape is 2–6× larger than the sum of eager lanes.

### Where the mapped time goes

`benchmarks/grouped_vmap_phase_profile.py` (case B composition, energy,
`no_grad`; raw data `benchmarks/results/grouped-vmap-phase-profile.json`):

| Case | Route | Total | SCF solve | Integrals | SCF prep | Split + pack | Eigensolver (calls) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H2O × 8 | vmap | 7.37 s | 7.34 s | < 0.01 s | < 0.01 s | < 0.01 s | 0.03 s (204) |
| H2O × 8 | eager | 2.74 s | 2.68 s | 0.02 s | < 0.01 s | < 0.01 s | 0.01 s (160) |
| CH4 × 8 | vmap | 7.64 s | 7.60 s | < 0.01 s | < 0.01 s | < 0.01 s | 0.04 s (204) |
| CH4 × 8 | eager | 2.91 s | 2.85 s | 0.02 s | < 0.01 s | < 0.01 s | 0.01 s (166) |
| NH3 × 4 | vmap | 7.43 s | 7.40 s | < 0.01 s | < 0.01 s | < 0.01 s | 0.03 s (204) |
| NH3 × 4 | eager | 1.50 s | 1.46 s | 0.01 s | < 0.01 s | < 0.01 s | 0.01 s (86) |
| B (all three) | vmap | 22.41 s | 22.35 s | 0.01 s | < 0.01 s | < 0.01 s | 0.09 s (612) |
| B (all three) | eager | 7.17 s | 7.03 s | 0.05 s | 0.01 s | < 0.01 s | 0.03 s (416) |

More than 99 % of the time is the SCF solve; integrals, preparation,
splitting and packing are negligible, and so is the eigensolver itself. The
per-trip cost is the rest of the map evaluation, which this profile does not
break down further. The 2.1× case-B
regression is three fixed 100-trip loops (≈ 7.4 s each, independent of
group size) for lanes that converge physically after 9–10 iterations.

The trip count is the lever. Same benchmark, mapped route only, with the
configured `maxiter` lowered (all lanes still converged):

| Molecule | Mode | B | vmap, 12 trips | vmap, 25 trips | vmap, 100 trips | eager |
| --- | --- | --- | --- | --- | --- | --- |
| H2O | energy | 1 | 1.00 s | 1.92 s | 7.12 s | 0.34 s |
| H2O | energy | 4 | 1.01 s | 1.94 s | 7.21 s | 1.38 s |
| H2O | energy | 16 | 1.09 s | 2.12 s | 7.75 s | 5.43 s |
| H2O | forces | 1 | 1.40 s | 2.68 s | 10.44 s | 0.55 s |
| H2O | forces | 4 | 1.47 s | 2.74 s | 10.52 s | 2.33 s |
| H2O | forces | 16 | 1.58 s | 3.02 s | 11.51 s | 9.32 s |
| CH4 | energy | 1 | 1.03 s | 1.92 s | 7.26 s | 0.40 s |
| CH4 | energy | 4 | 1.05 s | 1.95 s | 7.48 s | 1.52 s |
| CH4 | energy | 16 | 1.14 s | 2.17 s | 8.08 s | 5.84 s |
| CH4 | forces | 1 | 1.43 s | 2.73 s | 10.74 s | 0.63 s |
| CH4 | forces | 4 | 1.48 s | 2.81 s | 10.97 s | 2.67 s |
| CH4 | forces | 16 | 1.71 s | 3.25 s | 12.42 s | 10.42 s |

Mapped time is linear in the trip count and almost flat in B. With a trip
count near physical convergence the mapped group beats eager lanes from
B = 4 (energy and forces) and by about 5× at B = 16, and the autograd memory
drops with it (forces, B = 16: 1.6 → 0.54 GiB for H2O, 3.0 → 0.77 GiB for
CH4). Lowering `maxiter` is not a fix (it only works because these lanes
converge within 12 trips); it isolates the cause.

No execution-policy change was made: on CPU the data would support "eager
for every group", but that rule is device-dependent and CUDA was not
measured. A CUDA run of the same script (`--device cuda`, synchronized
timings, peak allocated/reserved memory) remains open.

### Recommendation

Redesign the fixed-trip SCF controller (option D): the measured cost of
mapped groups is the unconditional `maxiter` trip schedule, both in time
(≈ 10× more trips than needed) and in autograd memory; at matched trip
counts grouped vmap is already the faster route. A dispatch threshold
(option B) would only route around this cost on CPU and cannot be grounded
for CUDA yet; native homogeneous batching (option C) would keep the same
fixed-trip schedule.

## Adaptive grouped SCF

The recommendation above was implemented without a second SCF stack: the
fixed-trip controller became resumable (`FixedTripControl`), and mapped
groups of eager/autograd calls advance in chunks and stop after their last
active lane (see "Execution schedule inside a group"). The configured
`maxiter` and convergence tolerances are unchanged; the public per-lane
`iterations` remain the physical counts.

### Chunk length

`benchmarks/vmap_vs_eager_groups.py --strategies vmap-adaptive --chunk 1 2 4
8 16` (H2O, CH4, C4H5NCS; energy and forces; B = 4, 16; raw data
`benchmarks/results/adaptive-chunk-sweep-cpu.json`). One extra `vmap` call
and host read costs a few milliseconds, one map trip 50–600 ms, so overshoot
trips dominate: chunk 1 was fastest or within noise everywhere (e.g. CH4
forces B = 16: 1.35 / 1.45 / 1.70 / 2.21 / 2.24 s for 1 / 2 / 4 / 8 / 16)
and records the fewest trips (C4H5NCS forces B = 16: 5.9 GiB at chunk 1,
6.9 GiB at chunk 4). The internal default is `_ADAPTIVE_CHUNK_TRIPS = 1`.
Chunk length 1 is the measured CPU optimum; the CUDA optimum is unknown.
On CUDA every chunk boundary is a device-to-host read of the group's
`active` flags, so if CUDA is revisited the chunk length must be benchmarked
there independently; no device-specific default is set in advance.

### Static vs adaptive vs eager (CPU)

Same environment as above (GFN1, PyTorch integrals, float64,
`OMP_NUM_THREADS=1`, default `maxiter = 100`, two repeats, minimum wall time,
peak RSS of the case process, watchdog 12 GiB; raw data
`benchmarks/results/adaptive-vs-static-vs-eager-cpu.json`). `static` is the
full fixed-trip schedule (used inside transforms), `adaptive` the default
mapped route, `eager` independent eager lanes. Cells: wall time / peak RSS.

Both JSON files record commit `b0f44881`, the pre-rebase tree with the
same package code (the branch was later rebased onto the WP0–WP3 commits,
whose only solver change, the charge-padding projection, is shared by both
schedules). A spot check at the rebased tree reproduced the counts exactly
and the timings within 5 % (H2O energy B = 8: static 7.26 s / 101 maps,
adaptive 0.76 s / 9 maps / 8 chunks; CH4 forces B = 16: adaptive 1.35 s,
0.7 GiB, 10 maps). The grid below is therefore historical (pre-rebase)
evidence; the accepted rebased implementation is verified separately in
"Final-HEAD verification" below.

| Molecule | Mode | Route | B=2 | B=4 | B=8 | B=16 | B=32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H2O (3/8) | energy | static | 7.22 s / 0.4 G | 7.31 s / 0.4 G | 7.51 s / 0.4 G | 8.00 s / 0.4 G | 8.31 s / 0.4 G |
| H2O (3/8) | energy | adaptive | 0.77 s / 0.4 G | 0.76 s / 0.4 G | 0.78 s / 0.4 G | 0.82 s / 0.4 G | 0.87 s / 0.4 G |
| H2O (3/8) | energy | eager | 0.70 s / 0.4 G | 1.39 s / 0.4 G | 2.81 s / 0.4 G | 5.59 s / 0.4 G | 11.10 s / 0.4 G |
| H2O (3/8) | forces | static | 10.74 s / 0.7 G | 10.61 s / 0.8 G | 10.65 s / 1.0 G | 11.39 s / 1.6 G | 12.81 s / 2.8 G |
| H2O (3/8) | forces | adaptive | 1.03 s / 0.4 G | 1.10 s / 0.4 G | 1.10 s / 0.4 G | 1.21 s / 0.5 G | 1.31 s / 0.6 G |
| H2O (3/8) | forces | eager | 1.12 s / 0.4 G | 2.38 s / 0.5 G | 4.72 s / 0.6 G | 9.79 s / 0.8 G | 19.64 s / 1.2 G |
| H2O (3/8) | param | static | 10.62 s / 0.7 G | 10.82 s / 0.8 G | 11.08 s / 1.0 G | 11.73 s / 1.5 G | 12.91 s / 2.7 G |
| H2O (3/8) | param | adaptive | 1.04 s / 0.4 G | 1.07 s / 0.4 G | 1.09 s / 0.4 G | 1.17 s / 0.5 G | 1.30 s / 0.6 G |
| H2O (3/8) | param | eager | 1.11 s / 0.4 G | 2.39 s / 0.5 G | 4.57 s / 0.6 G | 9.73 s / 0.8 G | 18.81 s / 1.2 G |
| CH4 (5/12) | energy | static | 7.31 s / 0.4 G | 7.71 s / 0.4 G | 8.08 s / 0.4 G | 8.23 s / 0.4 G | 8.75 s / 0.4 G |
| CH4 (5/12) | energy | adaptive | 0.85 s / 0.4 G | 0.85 s / 0.4 G | 0.90 s / 0.4 G | 0.93 s / 0.4 G | 1.04 s / 0.4 G |
| CH4 (5/12) | energy | eager | 0.75 s / 0.4 G | 1.47 s / 0.4 G | 2.99 s / 0.4 G | 5.71 s / 0.4 G | 12.01 s / 0.4 G |
| CH4 (5/12) | forces | static | 10.89 s / 0.9 G | 11.17 s / 1.0 G | 11.52 s / 1.7 G | 12.17 s / 3.0 G | 14.34 s / 5.7 G |
| CH4 (5/12) | forces | adaptive | 1.18 s / 0.4 G | 1.21 s / 0.4 G | 1.29 s / 0.5 G | 1.43 s / 0.7 G | 1.58 s / 1.0 G |
| CH4 (5/12) | forces | eager | 1.25 s / 0.4 G | 2.61 s / 0.5 G | 5.04 s / 0.6 G | 10.36 s / 0.9 G | 21.11 s / 1.5 G |
| CH4 (5/12) | param | static | 10.63 s / 0.8 G | 10.88 s / 1.0 G | 11.40 s / 1.7 G | 12.75 s / 3.0 G | 15.19 s / 5.7 G |
| CH4 (5/12) | param | adaptive | 1.20 s / 0.4 G | 1.24 s / 0.4 G | 1.26 s / 0.5 G | 1.44 s / 0.7 G | 1.74 s / 1.0 G |
| CH4 (5/12) | param | eager | 1.22 s / 0.4 G | 2.48 s / 0.5 G | 5.20 s / 0.6 G | 9.94 s / 0.9 G | 20.63 s / 1.4 G |
| SiH4 (5/17) | energy | static | 7.83 s / 0.4 G | 8.07 s / 0.4 G | 8.35 s / 0.4 G | 9.44 s / 0.4 G | 11.36 s / 0.4 G |
| SiH4 (5/17) | energy | adaptive | 0.87 s / 0.4 G | 0.89 s / 0.4 G | 0.96 s / 0.4 G | 1.08 s / 0.4 G | 1.34 s / 0.4 G |
| SiH4 (5/17) | energy | eager | 0.80 s / 0.4 G | 1.62 s / 0.4 G | 3.16 s / 0.4 G | 6.71 s / 0.4 G | 13.01 s / 0.4 G |
| SiH4 (5/17) | forces | static | 11.16 s / 1.0 G | 12.09 s / 1.7 G | 13.14 s / 3.0 G | 15.67 s / 5.7 G | 23.21 s / 11.1 G |
| SiH4 (5/17) | forces | adaptive | 1.20 s / 0.4 G | 1.43 s / 0.5 G | 1.54 s / 0.7 G | 1.74 s / 1.0 G | 2.60 s / 1.6 G |
| SiH4 (5/17) | forces | eager | 1.41 s / 0.4 G | 2.90 s / 0.5 G | 5.61 s / 0.7 G | 11.54 s / 1.1 G | 23.21 s / 1.9 G |
| SiH4 (5/17) | param | static | 11.35 s / 1.0 G | 11.71 s / 1.7 G | 13.27 s / 3.0 G | 16.33 s / 5.7 G | 22.23 s / 11.0 G |
| SiH4 (5/17) | param | adaptive | 1.26 s / 0.4 G | 1.32 s / 0.5 G | 1.50 s / 0.7 G | 1.89 s / 1.0 G | 2.55 s / 1.6 G |
| SiH4 (5/17) | param | eager | 1.33 s / 0.4 G | 2.74 s / 0.5 G | 5.43 s / 0.7 G | 11.03 s / 1.1 G | 23.09 s / 1.8 G |
| C4H5NCS (12/43) | energy | static | 9.13 s / 0.4 G | 10.47 s / 0.4 G | 13.60 s / 0.4 G | 19.74 s / 0.4 G | 31.83 s / 0.4 G |
| C4H5NCS (12/43) | energy | adaptive | 1.42 s / 0.4 G | 1.65 s / 0.4 G | 2.13 s / 0.4 G | 3.04 s / 0.4 G | 5.02 s / 0.4 G |
| C4H5NCS (12/43) | energy | eager | 1.30 s / 0.4 G | 2.57 s / 0.4 G | 5.08 s / 0.4 G | 10.06 s / 0.4 G | 20.03 s / 0.4 G |
| C4H5NCS (12/43) | forces | static | 14.65 s / 3.1 G | 19.21 s / 8.9 G | killed | killed | killed |
| C4H5NCS (12/43) | forces | adaptive | 2.26 s / 0.8 G | 3.01 s / 1.7 G | 4.29 s / 3.0 G | 8.09 s / 5.7 G | 14.66 s / 11.2 G |
| C4H5NCS (12/43) | forces | eager | 2.40 s / 1.1 G | 4.98 s / 1.8 G | 10.67 s / 3.2 G | 19.92 s / 5.9 G | 38.54 s / 11.5 G |
| C4H5NCS (12/43) | param | static | 14.35 s / 3.1 G | 20.49 s / 8.8 G | killed | killed | killed |
| C4H5NCS (12/43) | param | adaptive | 2.20 s / 0.8 G | 3.27 s / 1.7 G | 4.79 s / 3.0 G | 8.10 s / 5.6 G | 13.60 s / 11.0 G |
| C4H5NCS (12/43) | param | eager | 2.58 s / 1.1 G | 5.19 s / 1.8 G | 10.25 s / 3.1 G | 20.64 s / 5.9 G | 41.15 s / 11.4 G |

Map evaluations per group (energy, B = 8): static always 101 (1 + `maxiter`);
adaptive equals the slowest lane's physical count (H2O 9, CH4 10, SiH4 10,
C4H5NCS 14, with 8 / 9 / 9 / 13 chunks); eager lanes need the sum over lanes
(72 / 75 / 80 / 111).

- Adaptive is 6–10× faster than the static schedule at every size and its
  autograd memory drops accordingly (SiH4 forces B = 32: 11.1 → 1.6 GiB;
  C4H5NCS forces/param, where static exceeded 12 GiB from B = 8, now run up
  to B = 32 at the memory of the eager lanes).
- Compared with the hand-lowered 12-trip experiment above, adaptive is
  faster without changing `maxiter` (H2O energy B = 4: 0.76 s vs 1.01 s;
  forces B = 16: 1.21 s vs 1.58 s; CH4 forces B = 16: 1.43 s vs 1.71 s),
  because it stops after 9–10 trips instead of 12.
- Against eager lanes, adaptive ties at B = 2 (0.89–1.17×) and wins from
  B = 4 (1.6–2.2×), up to 9–15× at B = 32 for the small molecules and
  2.6–4× for C4H5NCS. Peak memory is at most that of the eager lanes.

### Final-HEAD verification

A representative subset was rerun at the accepted rebased commit
`4c014d36` from a clean detached worktree (no uncommitted changes), with the
same settings (`--maxiter 100 --repeats 2 --mem-gib 12 --chunk 1`,
`OMP_NUM_THREADS=1`; raw data
`benchmarks/results/adaptive-final-head-smoke-cpu.json`). Cells: minimum
wall time / peak RSS; maps = SCF map evaluations of the group.

| Case | nat / nao | static | adaptive | eager | iterations (max) | maps static / adaptive / eager |
| --- | --- | --- | --- | --- | --- | --- |
| H2O energy, B = 8 | 3 / 8 | 7.52 s / 0.36 G | 0.78 s / 0.35 G | 2.73 s / 0.35 G | 9 | 101 / 9 / 72 |
| H2O forces, B = 16 | 3 / 8 | 11.82 s / 1.56 G | 1.18 s / 0.48 G | 9.92 s / 0.79 G | 9 | 101 / 9 / 144 |
| SiH4 forces, B = 32 | 5 / 17 | 22.08 s / 11.1 G | 2.74 s / 1.63 G | 23.67 s / 1.88 G | 10 | 101 / 10 / 320 |
| C4H5NCS forces, B = 16 | 12 / 43 | killed above 12 GiB | 6.89 s / 5.73 G | 19.86 s / 5.91 G | 14 | – / 14 / 218 |

For chunk length 1 the adaptive map evaluations equal the group's maximum
physical iteration count in every case (`adaptive_maps_equal_max_iterations`
in the JSON), i.e. no overshoot trip is run. Counts reproduce the pre-rebase
grid exactly; timings agree within 6 % except C4H5NCS forces B = 16,
where this adaptive run was 15 % faster (H2O energy B = 8 adaptive 0.78 vs
0.78 s; SiH4 forces B = 32 adaptive 2.74 vs 2.60 s, static 22.1 vs 23.2 s;
C4H5NCS forces B = 16 adaptive 6.89 vs 8.09 s, eager 19.9 vs 19.9 s).
