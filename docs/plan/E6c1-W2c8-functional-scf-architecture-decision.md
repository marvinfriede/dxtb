# E6c1-W2c8 — Functional SCF architecture decision

## Decision summary

**Recommendation: preserve the existing finite-trip `System.singlepoint()`
contract and develop a separately named explicit converged-root batch API, if
the project owner approves that derivative contract and amends the W2 route
acceptance.** Do not change `System.singlepoint()` dispatch in this package.

The literal `torch.func.vmap(System.singlepoint)` entrypoint is transform-safe
because its SCF controller has a static trip count. That same property means it
cannot stop individual lanes from executing map evaluations after convergence.
The private implicit-root path avoids retaining the primal trajectory, but its
eager lane-local root preparation cannot simply run inside `vmap`. W2c7's
one-trip timeout attribution was incorrect: the timeout is not reproduced by
the current route, and its incomplete artifact does not identify a cause.

This is an architecture recommendation, not an API or acceptance-policy
change. W2 and W3 remain open; E6broute remains blocked.

## Revision, environment, and scope

- Base SHA: `124a464e952e230ac8489ced48c5ef9e9295dcb6`.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote branch was fetched before changes and matched the base SHA.
- SCF coordination was read; S2b.1 owns SCF architecture production work.
  W2c8 changed only private benchmark code and documentation.
- CPU: AMD EPYC 7763.
- Python 3.11.15 / PyTorch 2.8.0+cpu and Python 3.14.8 /
  PyTorch 2.14.1+cpu.
- float64, one PyTorch intra-op thread, one inter-op thread;
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.
- Dependencies: tad-mctc 0.11.3, tad-multicharge 0.8.0,
  tad-libcint 0.4.0, tad-dftd3 0.7.0, tad-dftd4 0.8.0.
- Source import was asserted to resolve to this worktree's `src/dxtb`.
- No production code, dependency, tolerance, Result schema, public route, or
  frozen workload input changed.

Raw traces are in
[`benchmarks/results/e6c1-w2c8/`](../../benchmarks/results/e6c1-w2c8/).
The phase-profiler entrypoint is
[`benchmarks/e6c1_w2c8_phase_profile.py`](../../benchmarks/e6c1_w2c8_phase_profile.py).

## Correction to W2c7

The original W2 report discussed a 512-point entropy quadrature. That
diagnosis predates the current equal-partition trace implementation and is not
evidence for this workload. At positive electronic temperature and the
default `FERMI_PARTITION_EQUAL`, the current functional finalizer calls:

```text
free_energy_from_spectral
  -> fermi_free_energy_trace
  -> _fermi_free_energy_setup
```

It does not call `fermi_free_energy_matrix()` or
`_binary_entropy_matrix()` in this route. W2c8 measured one trace free-energy
call and zero binary-entropy-matrix calls for the one-trip case. The
512-point diagnosis is historical only.

The W2c7 one-trip artifact contains only progress text for Interactions,
Classicals, and Integrals setup. It has no stage completion, environment
header, or process result. W2c7 did not localize its 180-second timeout.
W2c8 reran the current source path successfully in both primary environments;
the prior timeout is unreproduced, and no cause is assigned to it.

W2c7's inherited E7a/E6bcompat results are inherited evidence, not tests
rerun by W2c7. W2c7 itself reran the W2c6 recurrence module on the two primary
profiles. W2c8 did not rerun E7a or E6bcompat.

## Frozen one-trip profiler

The harness uses conformer 0 from the frozen tmpda generator (69 atoms,
seed 0, sigma 0.1 bohr with conformer 0 undisplaced), GFN1-xTB, PyTorch
integrals, float64, charge SCP, FULL, Anderson, 300 K, `x_atol=f_atol=1e-8`,
`x_atol_max=1e-5`, and position-only gradients. It sets `maxiter=1` for the
bounded diagnostic. This state is not converged and its gradient is not a
physical force reference.

Each process compares the direct single-System call and the literal
one-lane entrypoint:

```python
torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
```

It measures forward-only and energy-plus-position-gradient runs. Timing
wrappers are scoped and restored with `finally`. Setup and result metadata
are emitted as JSONL. `Result.energy.sum()` is the observable; `fenergy` is
reported separately and is not added again.

| Python/Torch | Route | Forward | Backward | Total | Peak RSS |
|---|---|---:|---:|---:|---:|
| 3.14 / 2.14.1 | direct, no gradient | 4.23 s | — | 4.23 s | 425 MiB |
| 3.14 / 2.14.1 | one-lane vmap, no gradient | 4.59 s | — | 4.59 s | 434 MiB |
| 3.14 / 2.14.1 | direct, position gradient | 5.87 s | 7.93 s | 13.80 s | 3,334 MiB |
| 3.14 / 2.14.1 | one-lane vmap, position gradient | 6.11 s | 8.14 s | 14.25 s | 3,045 MiB |
| 3.11 / 2.8.0 | one-lane vmap, position gradient | 6.13 s | 7.90 s | 14.03 s | 2,997 MiB |

The direct and one-lane vmap PyTorch 2.14.1 runs had identical energy
(`-26.59355133377614 Eh`), Fermi field (`-0.00024347170545983355 Eh`),
density norm, iteration metadata, and residual. The maximum component
difference between stored position gradients was `4.996e-15 Eh/bohr`.
The 3.11/PyTorch 2.8 mapped run returned the same displayed energy and
residual. Timings are single diagnostic runs, not a performance distribution.

At one configured trip the fixed controller performs **two** raw maps: one
to initialize the solver state, then one for the configured update. The
reported iteration count is 2. The second map result is the selected result
used in finalization; there is no separate third SCF map in charge mode.
Both maps construct a spectrum and matrix-Fermi density. A single
equal-partition trace free-energy calculation follows the SCF controller.

| Forward stage, direct no-gradient run | Calls | Inclusive time |
|---|---:|---:|
| Integral preparation | 1 | 0.066 s |
| Interaction preparation | 1 | 0.0017 s |
| Initial raw SCF map | 1 | 1.37 s |
| Configured update raw SCF map | 1 | 1.39 s |
| Matrix-Fermi density | 2 | 2.66 s |
| Spectral diagonalization | 2 | 0.026 s |
| Fermi occupation state | 2 | 0.077 s |
| Equal-partition free-energy trace | 1 | 1.30 s |
| Result assembly | 1 | 0.00003 s |

Stage timings are nested and must not be summed. The solver-controller
inclusive timer contains the raw maps and free-energy finalization. For the
position-gradient run, density construction took about 3.7–3.8 s total, the
trace path about 1.8–1.9 s, forward completion about 5.9–6.1 s, and backward
about 7.9–8.1 s. Classical energy construction was under 0.02 s, integral
preparation about 0.10–0.13 s, and Result assembly negligible. Thus the
current measurable one-trip forward cost is concentrated in matrix-Fermi
density and trace finalization; the old 180-second timeout is not explained
by these completed runs.

## Fixed-trip trajectory and memory behavior

The public `System.singlepoint()` calls `system_singlepoint()`, which enters
`_singlepoint_impl(..., fixed_trip=True)`. It prepares integrals and
interaction data once, calls `run_system_fixed_trip()`, and constructs the
public Result. For the functional charge route, `run_system_fixed_trip()`
builds the initial charge guess and calls `run_prepared_functional_full()`.
The functional solver initializes from `G(x0)` and then loops over exactly
`maxiter` configured updates. Its active/converged tensor selects solver state
and returned values but does not skip `G(x)` evaluation. This is why one trip
has two raw map calls and `N` configured trips have `N+1` raw maps.

The returned gradient differentiates the finite executed map/mixing
trajectory. The graph includes states required by that trajectory and the
final observable. No BiCGSTAB implicit response is used on this route. The
fixed controller is transform-safe precisely because trip count is static
and no tensor-valued convergence status is converted to a Python boolean.

Bounded direct position-gradient measurements on Python 3.14/PyTorch 2.14.1:

| Configured trips | Raw maps | Forward | Backward | Total | Peak RSS | Converged |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 5.87 s | 7.93 s | 13.80 s | 3,334 MiB | no |
| 2 | 3 | 7.95 s | 10.56 s | 18.51 s | 4,251 MiB | no |
| 4 | 5 | 12.20 s | 17.55 s | 29.75 s | 6,070 MiB | no |
| 8 | — | not run | not run | not run | — | — |

Peak RSS grows by about 0.9 GiB per added configured trip across these
short observations. This is a small-run trend, not a reliable long-range
linear law. An 8-trip projection is approximately 9–10 GiB; the 8-trip test
was not launched because that estimate already exceeds the historical
8.16-GiB legacy eight-conformer peak used as a conservative resource
reference. The 300-trip workload was not run. The independent W2b historical
memory curve is additional evidence of severe growth, but came from its own
fixed-trip measurement conditions.

No live graph-node count was collected; process RSS is the available
measurement. The forward-only run peaked at about 0.42 GiB while the
one-trip differentiated run peaked at about 3.3 GiB. This confirms substantial
backward-graph ownership but does not assign every allocated byte to a
specific tensor family.

## Execution routes and derivative contracts

| Route | Primal controller | Derivative meaning | Public status |
|---|---|---|---|
| `Calculator(batch_mode=2)` | Legacy packed eager SCF; converged lanes can be culled by its controller | Derivative of its executed finite legacy trajectory | Public legacy compatibility/reference route |
| `torch.func.vmap(System.singlepoint)` | Functional fixed-trip controller; all configured maps execute | Derivative of the finite fixed-trip map/mixer trajectory | Authoritative E6a mapped single-System route |
| W2c2a/W2c7 private implicit route | Eager physical-root preparation per System, followed by mapped implicit attachment and Result finalization | Derivative of validated root equation `G(x*,p)-x*=0` | Private feasibility diagnostic |

The documented W2 legacy route uses Calculator's defaults, including
potential SCP unless overridden; the functional W2 construction explicitly
uses charge SCP. The frozen W2 contract and `torch.func.vmap(System.singlepoint)`
route remain authoritative for the existing gate. The private implicit route
does not clear that gate.

The finite-trajectory and converged-root derivatives are different
mathematical contracts. They can be expected to agree only when the finite
trajectory has converged sufficiently close to the same physical root, the
same physical map and parameters are used, and finite solver / linear-response
errors are below the comparison tolerance. At unconverged states they must
not be compared as though they represented the same derivative.

## Architecture alternatives

| Candidate | Derivative contract | Transform and API compatibility | Performance evidence | Decision |
|---|---|---|---|---|
| Keep finite-trip `System.singlepoint()` and optimize it | Exact current finite trajectory | Preserves API and its eager/`vmap`/AD contract. Masked lanes still compute maps. Skipping work requires a transform-compatible conditional with the same trajectory derivative. | Measured fixed-trip graph growth; no accepted dynamic-skip primitive demonstrated | Keep as existing contract; continue only targeted optimization with independent proof |
| Put converged-root stopping inside literal `vmap(System.singlepoint)` | Would change to root derivative | Eager tensor-to-Python convergence decisions are invalid in `vmap`; fixed maximum trips retain the same wasted evaluations. | Not established | Not feasible as a direct insertion under current constraints |
| Add explicit converged-root batch operation | Explicitly uses validated physical roots and implicit response | Eager lane-local preparation outside transforms; map explicit lane tensors through existing `_SCFMapTensorTree`/implicit boundary. New API semantics, status and failure behavior need acceptance. | Numerically validated on two tmpda conformers, but 429 s for two positions lanes; not competitive with historical W2 target yet | Preferred architecture to assess, not accepted for W2 |
| Retain routes and grant performance waiver | Preserves all existing behavior | No transform/API change | W2 mapped benchmark remains incomplete and historical target not matched | Requires explicit project-owner approval; agent cannot grant it |

### Literal `vmap(System.singlepoint)` feasibility

**Conclusion: blocked for lane-local early stopping under the current
entrypoint and transform contract.** `torch.func.vmap` can map the static
fixed-trip function, but a host-controlled `bool(converged)` branch cannot
inspect each lane inside that transform. A tensor mask only prevents a lane
from changing the selected state; it does not avoid computing the raw map.
Running every lane for a static maximum and attaching implicit derivatives
afterward would still pay the complete fixed-trip primal cost and would
silently change derivative semantics if applied to the existing method.

### Explicit batch API proposal

If approved, a separately named `System.singlepoint_batch(...)` (name is
provisional) should accept a shared single-structure System and explicit
batched positions `[B, nat, 3]`, with lane-valued charge/spin and external
field tensors represented by documented shapes. Model parameters remain
owned by the one shared Model and are never cloned per lane. The method
returns the existing Result pytree with a leading lane dimension and
lane-specific SCF status.

The call consists of two phases:

1. **Prepare roots eagerly per lane.** Reuse existing integral/setup
   construction, eager controller, functional SCF map, root certification,
   Fermi/electron-count validation, and the `root_x`/map-result pairing.
   Store only call-local roots, lane tensors, iteration metadata, statuses,
   and explicit SCF inputs. Do not retain trajectory graphs or put roots in a
   System/Model cache.
2. **Attach and finalize mapped roots.** Stack lane-varying integral,
   interaction, electron-count and geometry tensors; keep Model parameter
   tensors shared. Use the existing `_SCFMapTensorTree` and implicit-root
   response mechanism under `vmap`, then reuse functional finalization and
   Result construction. Preserve a lane dimension for response convergence
   metadata.

Existing reusable pieces include `build_integral_matrices()`,
`setup_scf_inputs()`, the eager solver/controller, `_prepare_scf_root()`,
`implicit_root()`, `_SCFMapTensorTree`, `finalize_scf*()`,
`free_energy_from_spectral()`, and `Result.snapshot()`. The new work should
orchestrate these once; it must not add a second SCF physical map.

Root preparation itself is host-controlled and is not promised to work under
`torch.func.vmap`, `grad`, `jacfwd`, or `jacrev`. The first proposed API
contract is an eager batch operation whose returned tensor outputs support
ordinary reverse-mode differentiation through the implicit attachment; its
internal per-lane response is mapped. Any whole-operation `torch.func`
support, especially forward mode or nested transforms, requires separate
proof and must not be implied by low-level implicit transform support.

### Proposed failure contract for owner review

No root that fails convergence, raw-map residual, final Fermi status, or
electron-count validation is attached to the implicit derivative. Before
attachment, root preparation records for each lane: iteration count, residual
norm and max component, electron-count error, Fermi status, and failure
code.

For a proposed non-strict batch API, valid lanes should continue independently;
invalid lanes should retain explicit failure status and have NaN differentiable
observables rather than a fabricated finite zero response. Strict/Fermi
failures and `force_convergence` should preserve existing exception behavior,
with a lane identifier added to the diagnostic. A failed BiCGSTAB response
remains a failed response with the existing NaN policy and per-lane status;
it is never accepted merely because a gradient tensor is finite. The
implementation should prepare only valid lanes for attachment outside
`vmap`, then reconstruct the result pytree with failed-lane NaNs/status, so
invalid lanes do not share a solver response or contaminate valid lanes.

These are proposed semantics for a new operation, not changes to current
`System.singlepoint()` behavior. The project owner must accept or revise them
before implementation.

## Model ownership, padding, and derivative support

For conformers of the same System, geometry-dependent integral and
interaction tensors vary per lane. GFN parameters remain the same tensor
leaves owned by one Model. Position gradients are lane-independent; shared
parameter gradients sum valid lane contributions. The existing W2c2a test
compared 2,261 floating leaves: 77 connected (4 connected-zero) and 2,184
disconnected, with maximum mapped-versus-serial-sum difference `1.279e-13`.
That ownership and connectivity metadata is a required invariant.

Conformer batches have common atom/orbital structure and therefore do not
introduce structural padding between lanes. The proposed API must still use
the accepted AO/orbital masks and must preserve zeros/NaNs/status for padded
or inactive entries. A future mixed-composition batch is a separate E6bimpl
boundary; it must not be implied by same-structure conformer batching.

The low-level implicit root and response have accepted `vmap` and nested
derivative controls at their tested boundary. The public E6a
`System.singlepoint()` path retains its existing eager, `vmap`, reverse/forward
transform and higher-order contract. For an explicit batch API, ordinary
position and shared-parameter gradients are required initially; JVP,
`jacfwd`, `jacrev`, HVP, third order, nested `vmap`, and `torch.compile` must
each be stated and tested before being advertised.

## Performance feasibility estimate

Measured private implicit-route observations from W2c7 remain:

- One conformer: median about 144 s, peak about 2.36 GiB.
- Two conformer positions-only mapped result: about 429 s, peak about
  3.62 GiB; independent root preparation was 226 s for 66/96 iterations,
  and mapped backward was about 196 s.
- Two-conformer positions-plus-Model parity process: about 653 s / 4.51 GiB,
  including serial derivative graphs; it is not isolated mapped memory.

The implicit architecture is numerically promising at the accepted
observable boundary but not performance-competitive with the historical
eight-conformer legacy result (36.215 s at one thread, 16.998 s at four
threads, about 8.16 GiB). Solver policies differ, so those numbers are
context rather than a matched speedup ratio.

Using the observed two-lane root preparation (226 s for 162 reported
iterations, about 1.40 s per reported iteration across those two systems)
only as a rough calibration, four lane roots with similar convergence
histories would require about 450–600 s for root preparation. Eight would
require roughly 900–1,200 s before finalization/backward. These estimates
exclude nonlinear effects from batched linear response, memory pressure, and
different conformer convergence histories. Based on the 3.62-GiB two-lane
mapped peak and the 4.62-GiB mixed mapped/serial parity process, four mapped
lanes plausibly require 6–9 GiB and eight 10–18 GiB; uncertainty is high.
No four/eight private implicit run is justified before the API and memory
ownership design are accepted.

For public fixed-trip mode, the measured 1–4 trip process peak rises from
3.33 to 6.07 GiB. That is not a direct projection to 300 trips, but both the
short-run trend and W2b's separately measured graph growth make the full
unrolled path a serious memory risk. No full 300-trip rerun was performed.

## Smallest next implementation package

The next package should be conditional on project-owner approval and should:

1. Add an internal call-local `prepare_root_batch` orchestration layer, with
   no public dispatch change.
2. Validate roots independently against their matching SCF inputs and
   preserve per-lane statuses.
3. Add a narrow explicit batch evaluation prototype using existing
   `_SCFMapTensorTree`/implicit response and shared Model parameters.
4. Test two small same-composition Systems, valid/invalid lane separation,
   position isolation, shared parameter accumulation, lane ordering, and
   response failure propagation.
5. Exercise one complete tmpda conformer and then two only if bounded
   resource measurements justify it.
6. Obtain a separate route-acceptance decision before changing whether the
   documented W2 gate may use this new API.

Risk is moderate-to-high: mathematical root and observable parity exist,
but public statuses, non-strict invalid-lane representation, input ownership,
whole-operation transform support, and W2 runtime remain unaccepted.

## Project-owner decision required

The current W2 acceptance specifically names
`torch.func.vmap(System.singlepoint)`. The efficient architecture under
consideration requires eager root preparation outside that transform and
therefore a distinct public batch operation or an explicit two-phase API.
The owner must choose whether to:

1. Keep W2 tied to literal `vmap(System.singlepoint)` and leave it blocked
   until that route meets the existing contract;
2. amend W2 to evaluate the supported converged-root batch API, with its
   derivative/failure semantics stated separately; or
3. preserve both APIs and maintain separate performance and compatibility
   expectations.

No choice is made by this report. `System.singlepoint()`, Calculator
routing, legacy batch FULL, E6bcompat, and E6broute are unchanged.

## Validation and review

- One-trip direct and one-lane vmap numerical summaries agree; a separate
  JSON comparison asserted exact energy/density-norm agreement and maximum
  gradient-component difference below `1e-10`.
- W2c6 production recurrence tests: inherited W2c7 evidence, 27 passed in
  Python 3.11/Torch 2.8 and 27 passed in Python 3.14/Torch 2.14.1.
- Focused route controls in Python 3.11/Torch 2.8:
  `test_e6a0_system_vmap` value and gradient controls plus implicit mapped-root
  and failed-root controls: 6 passed.
- Same focused selection in Python 3.14/Torch 2.14.1: 6 passed.
- No E7a or E6bcompat suite was rerun; their accepted status is inherited.
- Static checks: Python byte compilation, Black, isort, and
  `git diff --check` passed. Ruff was unavailable in the selected Python 3.14
  environment (`No module named ruff`) and was not run.
- Independent read-only review: **no blocker or major finding**. The reviewer
  confirmed route attribution, equal-partition trace dispatch, fixed-trip map
  counts, derivative-contract distinction, transform constraints, API and
  W2 acceptance isolation, and the bounded resource rationale. Its minor
  finding about Ruff availability is resolved by the explicit statement
  above. Runtime did not expose the reviewer model identity. The exact final
  commit SHA will be recorded in the package handoff.

## Status

- Public fixed-trip route: preserved.
- W2c7 timeout attribution: corrected; old timeout unreproduced.
- Current one-trip dominant forward stages: matrix-Fermi density and the
  equal-partition trace free-energy path.
- Literal vmap lane-local converged stopping: blocked by transform/control
  constraints.
- Explicit converged-root batch: recommended for owner review, not
  implemented.
- W2 / W3: open.
- E6broute: blocked.
- Public Calculator routing: unchanged.
