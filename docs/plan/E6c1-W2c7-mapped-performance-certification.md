# E6c1-W2c7 — mapped performance certification

## Decision

**W2 remains blocked by an architectural mismatch.** The committed W2c6
production recurrence passes a controlled three-process one-conformer
implicit-root energy/force comparison against the pre-W2c6 recurrence, and
the two-conformer private mapped implicit route passes position and shared
Model-parameter gradient parity. The authoritative
`vmap(System.singlepoint)` route remains a fixed-trip unrolled calculation.
Its bounded one-conformer, one-trip energy/force diagnostic did not return
within 180 seconds. No full 300-trip or eight-conformer mapped calculation
was launched. A faster private implicit route does not clear the public W2
gate.

No SCF production source, public Calculator routing, dependency, tolerance,
or frozen benchmark input changed in this package.

## Revision and environment

- Starting/accepted SHA: `d943e6dea42e552e9451c1e89bf38d1e564bc0ef`.
- Production source measured: `d943e6dea42e552e9451c1e89bf38d1e564bc0ef`.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote: `origin/claude/tender-dijkstra-cda42s`; it was fetched and verified
  at the required starting SHA before measurement.
- Source path: this worktree's `src/dxtb`.
- Host: `m199`, AMD EPYC 7763, Linux x86_64.
- A/B environment: Python 3.11.15, PyTorch 2.8.0+cpu, float64, one intra-op
  and one inter-op thread. `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
  `OPENBLAS_NUM_THREADS=1` were required and asserted.
- Dependencies: `tad-mctc 0.11.3`, `tad-multicharge 0.8.0`,
  `tad-libcint 0.4.0`, `tad-dftd3 0.7.0`, `tad-dftd4 0.8.0`.
- Each physical A/B observation ran in a fresh process with an 8,000 MiB
  diagnostic memory monitor and an external wall-time limit.

The initial sandbox `git fetch` was rejected by a local SSH configuration
ownership check. A retry in the approved Git runtime fetched the branch and
verified the required remote SHA. No remote state changed during that check.

## Workload fidelity

The W2 harness and the frozen baseline generator produced exactly equal
atomic-number and position tensors for conformers 0 and 1:
`torch.equal(numbers)` and `torch.equal(positions)` both returned `True`.
The shape was `(2, 69, 3)`, atom order was retained, conformer 0 was
undisplaced, and conformer 1 used the baseline generator's seed 0 and 0.1
bohr displacement. The full frozen workload remains tmpda, eight conformers,
GFN1-xTB, PyTorch integrals, float64, FULL SCF, charge SCP for the functional
route, Anderson mixing, 300 K, `maxiter=300`, `x_atol=f_atol=1e-8`, and the
default `x_atol_max=1e-5`. Default classical terms were retained.

The authoritative observable is `result.energy.sum()`. In the production
Result constructor, `energy` already contains the classical, SCF, and Fermi
free-energy contributions; `fenergy` remains a separately reported field.
The private diagnostic used the same total `Result.energy` once. Its optional
`energy + fenergy` value is not the W2 observable and was not used for the
reported force comparisons.

## Execution-route audit

| Route | Solver and differentiation | Convergence and Result behavior | Status |
|---|---|---|---|
| `Calculator(batch_mode=2)` | Retained legacy packed `SelfConsistentFieldFull`; ordinary unrolled differentiation over executed steps. | Legacy eager controller culls converged batch lanes. The recorded eight-lane run stopped after 205 iterations. The original W2 options do not specify `scp_mode`, so Calculator uses the default potential SCP mode. Result construction includes `scf + fenergy + classical` in `energy`. | Public legacy reference. |
| `torch.func.vmap(System.singlepoint, in_dims=(0,))(positions)` | `System.singlepoint` selects `system_singlepoint(..., fixed_trip=True)`, then the functional fixed-trip controller and full trajectory autograd. The W2 harness configures charge SCP, FULL, Anderson, and the frozen tolerances. | The controller executes every configured trip. Its active/converged masks project state and status, but do not skip the raw-map evaluation. Result finalization and the Fermi free-energy path run after the fixed-trip trajectory. The same System and Model are shared across geometry lanes. | Authoritative public E6a mapped route. |
| W2c2a/W2c7 private mapped implicit route | Each conformer gets an independent eager physical-root solve without retaining the trajectory graph. After independent residual/Fermi/electron-count checks, the validated roots are mapped through the existing implicit `G(x,p)-x=0` attachment and BiCGSTAB response. | Roots stop after their own convergence counts; this is not differentiation of the finite 300-step trajectory. Geometry/integral tensors are lane-specific; one Model parameter owner is shared. Full Result finalization follows attachment. | Private feasibility diagnostic; not public dispatch. |

The functional route uses the accepted invariant matrix-Fermi density and the
functional free-energy implementation. The private implicit diagnostic uses
those same production map/finalization functions. Legacy SCF uses a separate
legacy iterator and stopping/culling policy; it is not a matched controller
comparison. The current Calculator batch options also leave the default
potential SCP mode in place, unlike the mapped W2 functional configuration's
charge SCP mode. These route differences prevent describing legacy-versus-
mapped timing as a pure kernel speed comparison.

`W2c6`'s direct-RHS tanh-doubling recurrence is present in production
`spectral.py`; neither this benchmark nor the private route substitutes a
different matrix-Fermi function in production mode. The A/B reference mode
scopes the test-only pre-W2c6 identity-RHS recurrence to its process and
restores the production function in `finally`.

## Direct production and controlled A/B

The unpatched production recurrence was measured three times on conformer 0
using the private converged-root energy/position diagnostic. The comparison
method was the actual pre-W2c6 identity-RHS recurrence, also three fresh
processes. All six runs converged in 66 physical map evaluations and five
BiCGSTAB iterations; response status was zero in every run.

The first direct-RHS production observation is the earlier
`private-implicit-production-one-lane-py311.jsonl` trace. It uses the
W2c2-era JSONL schema and therefore lacks the wrapper's
`w2c7_kernel_run`/`source_revision` header; the two subsequent production
observations use the W2c7 wrapper. Its environment, workload configuration,
and source path are recorded, and it was run from the verified W2c6 source
SHA above. All three observations are retained separately rather than
rewritten into a synthetic common trace.

| Metric | Pre-W2c6 identity RHS | W2c6 production direct RHS |
|---|---:|---:|
| Total time, median (range) | 164.617 s (163.339–164.987) | 143.937 s (143.763–144.011) |
| Root preparation, median | 102.934 s | 89.902 s |
| Response `linearize` time, median | 38.055 s | 33.308 s |
| Peak RSS, median | 2,532.84 MiB | 2,364.53 MiB |
| SCF map evaluations | 66 | 66 |
| BiCGSTAB iterations | 5 | 5 |

The direct-RHS production run was 12.56% faster end to end, with 12.48% less
median measured linearization time and 6.65% lower median peak RSS. This is a
matched three-versus-three comparison on the private converged-root route,
not the public mapped W2 route.

Numerical comparison was invariant across all three paired observations:

- Energy delta (direct RHS minus identity RHS): `+6.04e-14 Eh`.
- Separate `fenergy` field delta: `-1.106e-13 Eh`; this field is already
  included once in `Result.energy`.
- Maximum force-component difference: `1.88e-13 Eh/bohr`.
- Position-gradient norm: `41.2947363357 Eh/bohr` in both methods to the
  displayed precision.
- Independent raw-map residual: `6.0120e-9`; electron-count error zero;
  Fermi status converged.

This establishes the W2c6 optimization on its tested production and private
implicit-response boundary. It does not establish full-W2 throughput.

## Two-conformer mapped physical and derivative validation

Two independent conformer roots were prepared under the frozen functional
configuration and checked against their own raw map:

| Conformer | Iterations | Residual norm | Maximum residual | Electron-count error | Fermi |
|---:|---:|---:|---:|---:|---|
| 0 | 66 | `6.0120e-9` | `4.3560e-9` | 0 | converged |
| 1 | 96 | `6.4161e-9` | `2.7394e-9` | 0 | converged |

Both densities and physical observables were finite. Mapped per-lane total
energies were `[-29.75843746217631, -11.312642703835364] Eh`; separate Fermi
free-energy fields were `[-0.001391311654408853, -0.001886992801543350] Eh`.
Each lane matched an independently evaluated serial value reference within
the benchmark assertions.

The mapped-only positions process took 428.62 s and peaked at 3,706.12 MiB.
It performed two independent roots in 226.15 s, mapped finalization in 6.53
s, and position backward in 195.91 s. No serial reference graph was created
in this memory measurement. Each lane had zero cross-conformer position
derivative block. Independent serial position gradients differed by at most
`6.22e-15` and `6.60e-14 Eh/bohr` for lanes 0 and 1.

A separate positions-plus-Model mapped/serial parity process completed in
653.50 s with a 4,618.43 MiB process peak. It used the same Model owner for
both lanes and compared all 2,261 floating parameter leaves against the sum
of independent serial gradients:

- 77 connected leaves; 4 of those had connected zero gradients.
- 2,184 leaves were disconnected for this physical composition.
- Maximum absolute difference: `1.2790e-13`.
- Difference norm: `2.4677e-13`; relative norm difference:
  `5.3299e-16`.
- Mapped and summed-serial gradient norms: `462.98230451394`.
- Position gradients remained finite and matched serial per lane.
- Mapped response status was `[0, 0]`; lane-specific position responses
  converged in 5 and 7 iterations, with residual norms `3.565e-13` and
  `9.232e-14`. The joint Model response converged in `[5, 7]` iterations.

The 4.62 GiB parity-process peak includes serial derivative work and is not
the isolated mapped memory result. The mapped-only position result is the
authoritative two-lane mapped RSS measurement.

## Authoritative mapped-route feasibility and scaling decision

The frozen public harness was run directly, without monkey-patching, on one
conformer with one fixed trip, one thread, and a 180-second wall-time bound.
The process completed setup but produced no Result or gradient before the
bound expired. Its output artifact has no completion event. The run was
stopped at this bound; it was not treated as a numerical pass or a full W2
measurement.

This is consistent with the earlier W2 stage trace, which localized the
one-trip finalization cost to the 512-point matrix-Fermi entropy quadrature.
The current solver also executes `maxiter=300` map calls even after its
convergence mask becomes inactive, and reverse-mode differentiation retains
the fixed-trip trajectory. W2b measured approximately 0.64 GiB of added
differentiated RSS per trip over trips 2–16 and a 106.4 GiB one-conformer
full diagnostic before timeout. W2c6 removes matrix multiplications inside
each matrix-Fermi doubling step; it does not change the number of fixed SCF
trips, early-stop policy, or trajectory graph lifetime.

The staged stop condition therefore applies: four/eight-lane public mapped
calculations were not launched, and the private implicit route was not
substituted for the authoritative mapped route to claim W2. The existing
legacy eight-conformer historical reference (36.215 s at one thread,
16.998 s at four threads, approximately 8.16 GiB peak) remains context only.
It differs in controller/stopping policy and the legacy default SCP mode;
no comparable same-configuration public mapped result exists. The legacy
comparison is **qualified**, not a speedup ratio.

### Architecture conclusion

The W2c2a/W2c7 private implicit-root implementation is not a performance
representative for the documented W2 gate. It changes the derivative
contract from a finite fixed-trip trajectory to the derivative of a
validated converged physical root. It completes one conformer in about 144 s
and two mapped conformers with position gradients in about 429 s, but those
results do not show that `vmap(System.singlepoint)` passes W2.

W2 is **BLOCKED — ARCHITECTURAL MISMATCH**. The smallest justified next task
is a separately reviewed architecture package that assesses integration of
the existing certified physical-root/implicit-response boundary into the
authoritative functional mapped API. It must define convergence/failure and
Result/status behavior, per-lane root preparation, shared Model ownership,
padding, and compatibility with accepted E6a/E6bcompat behavior before any
public routing change. No routing change is made here. The fixed-trip route's
entropy finalization and trajectory memory remain measured blockers; the
W2c6 kernel saving alone is not sufficient.

## Validation, review and artifacts

- W2c6 recurrence test module: 27 passed on Python 3.11/PyTorch 2.8 and 27
  passed on Python 3.14/PyTorch 2.14.1.
- The W2c6 accepted E6a/E7a/E6bcompat and mapped GFN1/GFN2 focused results
  remain as recorded in the W2c6 report. This package changed no production
  code and did not rerun those large selections.
- The new A/B runner passed Python byte compilation and its CLI help check.
- Two-conformer results and assertions are from the existing W2c2a
  diagnostic run against the committed W2c6 production code; no benchmark
  implementation was changed for those measurements.
- Direct matrix-free operator-vector parity was not separately emitted as a
  standalone artifact. Instead, the full implicit observable gradients and
  BiCGSTAB statuses/residuals were compared between recurrences and against
  independent serial lane references. This limitation is retained; no claim
  of isolated raw operator-vector equality is made.
- Independent read-only review: **no blocking or major findings**. The
  reviewer confirmed workload/observable fidelity, A/B consistency, isolated
  mapped RSS methodology, route distinction, and the W2 disposition. The
  minor provenance observation about the first production timing trace is
  addressed above.
- Public Calculator routing: unchanged. Dependencies and numerical
  tolerances: unchanged. `spectral.py`, fixed-trip solver, and implicit
  production code: unchanged.

Raw traces, including the bounded public one-trip timeout and all six
one-conformer A/B processes, are under
[`benchmarks/results/e6c1-w2c7/`](../../benchmarks/results/e6c1-w2c7/).
The two-conformer JSONL traces retain complete per-lane position and Model
gradient evidence. Historical W2c2a and W2c6 files were not overwritten.

## Final disposition

- W2c6 production recurrence: preserved and independently exercised.
- Controlled one-conformer private-route speedup: verified at 12.56% median.
- Two-conformer private mapped positions and shared Model gradients: pass.
- Authoritative public mapped W2 route: not completed even at one trip within
  the bounded diagnostic; 300-trip and eight-conformer results are not
  verified.
- W2 performance clearance: **OPEN; architecture review required**.
- W3 performance clearance: **OPEN**.
- E6broute: **BLOCKED**.
