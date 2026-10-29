# E6c1-W2c2a — mapped gradients and performance attribution

## Scope and status

This package corrects and measures the private W2c2 diagnostic. It does not
change production SCF code, the public Calculator route, E6bcompat, dependencies,
or the fixed-trip derivative contract.

- Starting SHA: `5365b16ca48ee2d7bac0326eeaad779993f244f6`.
- Diagnostic, test and trace commit: `31678bd6063a624f3c50b2d0de4705b12c1f4026`.
- Documentation and roadmap commit follows the reviewed diagnostic commit.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote: `origin`, `git@github.com:marvinfriede/dxtb.git`.
- The remote branch was fetched successfully at the starting SHA before the
  diagnostic commit; publication is verified after the final push.
- CPU host: `m199`, AMD EPYC 7763; Linux x86_64; one PyTorch/BLAS thread.
- Main performance measurements: Python 3.11.15, PyTorch 2.8.0+cpu.
- Focused test profiles: Python 3.11 / PyTorch 2.8 and Python 3.14 /
  PyTorch 2.14.1.
- Dependencies: `tad-mctc==0.11.3`, `tad-multicharge==0.8.0`,
  `tad-libcint==0.4.0`, `tad-dftd3==0.7.0`, `tad-dftd4==0.8.0`.

The first two frozen tmpda conformers, with seed 0 and 0.1 bohr displacement,
are taken directly from `benchmarks.baseline.workloads.conformers`. Atomic
number and position tensors are checked with exact `torch.equal` against that
generator. The original eight-conformer W2 benchmark and its historical result
files were not modified.

## Configuration reconciliation

The diagnostic now explicitly selects charge SCP, FULL SCF, Anderson mixing,
300 K, `maxiter=300`, `x_atol=f_atol=1e-8`, and
`force_convergence=False`. It sets `x_atol_max` to
`defaults.X_ATOL_MAX == 1e-5`, which is the effective setting for the frozen
mapped W2 benchmark because that benchmark does not override the Config default.
The configuration is logged both as requested and read back from
`ConfigSCF`. The original frozen mapped benchmark explicitly uses
`SCP_MODE_CHARGE`; the diagnostic no longer creates a default potential-mode
Config and then manually calls the charge map.

The earlier W2c2 runs used `x_atol_max=1e-8`. Those measurements are retained
as a stricter convergence experiment, not represented as exact W2-config
measurements. The same-host legacy Calculator command does not pass an SCP-mode
override; it therefore uses the Calculator default (`SCP_MODE_POTENTIAL`). Its
timing and energy are context only, not a like-for-like W2 performance result.

Two gradient modes are configured before System setup:

- `positions-only` leaves all floating Model parameter tensors with
  `requires_grad=False`, matching the frozen energy/force workload.
- `positions-and-model` enables floating Model parameter tensors before setup.

The configuration/geometry regression passed in both primary profiles:

```text
Python 3.11 / Torch 2.8:       1 passed
Python 3.14 / Torch 2.14.1:    1 passed
```

This matches the actual original benchmark setup: `ParamModule` wraps numeric
parameter leaves with `requires_grad=False` by default, and
`benchmarks/e6c1_w2.py` does not enable them.

## Position-only one-conformer result

Conformer 0 converged in 66 SCF map evaluations. The independently evaluated
raw-map residual was `6.01e-9` in norm and `4.36e-9` maximum component; both
spin-channel electron-count errors were zero, Fermi status was converged, and
density and energy were finite.

The exact frozen W2 observable is `result.energy.sum()`. The separately
reported `energy + fenergy` quantity is not substituted for it.

| Quantity | Position-only |
|---|---:|
| Energy observable, Eh | -29.758437462176378 |
| Separate Fermi free-energy field sum, Eh | -0.001391311654298254 |
| Energy plus free-energy field, Eh | -29.759828773830677 |
| Root preparation | 102.14 s, 66 iterations |
| Finalization and response setup | about 11.0 s before BiCGSTAB |
| Backward | 51.34 s |
| Total | 164.44 s |
| Peak RSS | 2,534.72 MiB |
| Position gradient norm / max absolute component | 41.2947 / 11.4680 Eh/bohr |
| BiCGSTAB | status 0, 5 iterations, residual `3.53e-13` |

The corresponding `positions-and-model` one-conformer run produced the same
energy, free energy, density, iterations, and position gradient at logged
precision. It completed in 163.02 s and peaked at 2,572.75 MiB. The measured
peak increase was about 38 MiB. In this mode all 2,261 floating Model leaves
were trainable; 77 had connected finite gradients. This is a mode comparison,
not a broad claim that all disconnected parameter leaves should be connected.

## Two-conformer mapped measurements

Each lane's physical root was prepared independently with its own geometry and
input tensors, then independently checked before mapped implicit attachment.
The roots required 66 and 96 evaluations and passed residual checks:

| Lane | Iterations | Raw-map residual norm | Max component |
|---|---:|---:|---:|
| 0 | 66 | `6.01e-9` | `4.36e-9` |
| 1 | 96 | `6.42e-9` | `2.74e-9` |

The authoritative isolated position-only run performed no serial reference
calculation in its process:

| Phase | Measurement |
|---|---:|
| Two independent roots | 249.8 s total; peak 607.9 MiB |
| Mapped finalization | 7.39 s; RSS rose from about 580 MiB to 2,945 MiB |
| Mapped backward | 223.29 s |
| Total mapped-only process | 486.00 s |
| Mapped-only lifetime peak RSS | 4,198.95 MiB |

The two mapped physical energies were `[-29.75843746217637,
-11.31264270383536] Eh`; Fermi free-energy fields were
`[-0.001391311654298254, -0.001886992801653950] Eh`. Both matched the
independent serial values within the assertions.

Separate differentiation of each lane energy gave per-lane gradients. They
matched independently prepared serial implicit gradients with maximum absolute
differences `5.9952e-15` and `5.1514e-14 Eh/bohr`. The two cross-conformer
position blocks were exactly zero in the measured tensors. Reversed conformer
ordering is additionally checked by a separate mapped-only run; no public
batching route is involved.

For components with serial magnitude greater than `1e-3 Eh/bohr`, maximum
componentwise relative differences were `4.21e-13` (lane 0) and `3.94e-13`
(lane 1). The full gradient norm differences were `2.22e-14` and
`1.74e-13 Eh/bohr`; mapped/serial gradient norms were `41.2947363357` and
`66.2949602146 Eh/bohr` respectively.

The explicit reversed run recorded lane-to-conformer indices `[1, 0]` and
independent roots in 96/66 iterations. After reversing the output order back,
energy and free-energy differences from the forward run were at most
`5.26e-13 Eh` and `5.53e-13 Eh`. Reordered position gradients differed by at
most `4.00e-14` and `5.36e-14 Eh/bohr`; both cross-lane blocks remained zero.
The reversed mapped-only process peaked at 4,181.72 MiB.

The mapped response results retained the lane dimension:

| Output lane | RHS norm by response lane | Iterations | Status | Residual norm |
|---|---|---|---|---|
| 0 | `[3.5973e-7, 0]` | `[5, 0]` | `[0, 0]` | `[3.5334e-13, 0]` |
| 1 | `[0, 2.8908e-7]` | `[0, 7]` | `[0, 0]` | `[0, 1.0471e-13]` |

Status 0 denotes successful response convergence. There is no lane-status
reduction before recording.

### Shared Model parameter gradients

The mapped parameter-gradient run uses the same Model parameter owner for both
lanes. Two independent serial Systems are set up over that same owner, and the
sum of the independent serial gradients is compared against the mapped total
gradient for each named leaf. All 2,261 floating leaves were compared and the
assertions passed. Of these, 77 were connected; 73 had nonzero finite gradient
norms and 4 had connected zero gradients in this workload. The other 2,184
leaves were disconnected, principally because their element/pair parameters
are not involved in this tmpda composition. This inventory is a workload
observation, not a requirement that unused parameters receive gradients.
The 73 nonzero leaves were in Hamiltonian, D3, repulsion, charge, halogen, and
the H/C/N/F/I element parameter groups. Across all leaves, mapped versus
summed-serial gradients had maximum absolute difference `9.2371e-14`,
Euclidean difference norm `1.9001e-13`, and relative norm difference
`4.1041e-16`; their gradient norms were `462.98230451394` and
`462.98230451394` respectively.

For the numerical sum comparison, disconnected serial leaves are represented
as zero contributions. Therefore the connected/zero/disconnected inventory
describes the mapped total gradient; it does not claim lane-by-lane identity
of PyTorch's `None` versus connected-zero metadata. The numerical gradient
sum itself is asserted for each parameter leaf.

The successful derivative comparison process includes serial backward graphs
and is therefore not used as the mapped memory result. A separate two-lane
`positions-and-model`, mapped-only run was also collected to report its
uncontaminated memory peak. It completed in 612.19 s and peaked at 4,320.57
MiB. The two active response lanes converged in 5 and 7 iterations (status
`[0, 0]`); 77 connected Model leaves had finite gradients. Its peak is the
isolated mapped-plus-Model-gradient result; the larger 5.05 GiB parity-process
peak remains a mixed mapped-plus-serial-reference measurement.

Two early parity attempts are retained in the trace directory. They failed
with PyTorch's “backward through the graph a second time” error because the
serial reference reused graph-bearing System setup state. The corrected
diagnostic creates fresh serial System setup graphs over the same Model
parameter owner, re-prepares and validates each serial input/root, and then
compares responses. The corrected parity run completed; the initial errors
were harness graph-lifetime defects, not production SCF failures.

### Corrected interpretation of the previous memory figure

The prior W2c2 two-lane process prepared serial derivative references before
the mapped backward. Those reference graphs remained live, and the process
peaked at about 8.37 GiB. That is a **combined diagnostic-process** peak, not
the isolated mapped cost. The corrected mapped-only position run peaks at
4.10 GiB (4,198.95 MiB). Thus serial-reference graph contamination is
eliminated from the primary measurement. RSS remains a process-level measure;
it does not identify live tensor ownership or separate allocator high-water
behavior exactly.

The mapped finalization stage itself coincides with an RSS increase before
backward. Existing traces do not attribute that increase to a single owner.
It includes implicit attachment and differentiable observable finalization;
the benchmark does not claim that all of it is implicit-root solver storage.

## Cost attribution

A one-map diagnostic profile took 5.45 s with profiler instrumentation. Named
component timings were:

| Component | Time |
|---|---:|
| Interaction potential | 0.0013 s |
| Hamiltonian construction | 0.0012 s |
| Diagonalization | 0.0141 s |
| Fermi occupation state | 0.0696 s |
| Fermi density construction | 1.5894 s |
| Density to charges | 0.0008 s |

This is one instrumented map, not an uncontaminated wall-time estimate; the
profiler adds overhead. The operator profile attributes the largest measured
share to `aten::bmm` (287 calls, 0.631 s self CPU time), followed by 90 LU
solve calls (0.423 s) and 90 LU factorizations (0.146 s). These operations
occur within the Fermi-density path. This identifies Fermi density
construction and its batched matrix products/solves as the dominant measured
primal-map cost; it does not justify changing the accepted Fermi algorithm.

For one-conformer energy-only implicit response, the raw-map linearization
setup before the solver took about 5.98 s. BiCGSTAB completed in 5 iterations;
13 `linearize` evaluations took 38.72 s of a 51.34 s backward. For the two
mapped lanes, each lane's response took 79.06 s and 101.14 s, with the full
mapped backward taking 223.29 s. The response operator is therefore a major
cost center. The current logs do not provide a complete per-operator allocation
profile; RSS samples are process RSS before and after a call, not allocation
bytes. A next optimization should first profile one matrix-free response
application and identify repeated raw-map/Fermi work before considering
call-local reuse.

## Same-host legacy context

A one-conformer legacy Calculator run on the same host and Python/Torch profile
completed energy plus force in 3.69 s, peaking at 837 MiB. Its energy was
`-29.758437462176353 Eh`, numerically matching the converged implicit result.
This is a context measurement only: the legacy Calculator options omit an SCP
mode and therefore use the potential-mode default, while the mapped functional
configuration explicitly uses charge mode; their controllers also differ
(legacy early stopping versus independent root acquisition plus implicit
response). It does not provide an accepted performance ratio or establish
equivalent derivative contracts. The historical eight-conformer W2 baseline
remains the only performance reference.

## Validation and review

Focused tests:

| Selection | Python 3.11 / Torch 2.8 | Python 3.14 / Torch 2.14.1 |
|---|---:|---:|
| Frozen workload/configuration/gradient-mode test | 1 passed | 1 passed |
| W2c1a mapped physical-root tests | 3 passed | 3 passed |

The W2c1a selection took 63.29 s and 59.42 s respectively. Python 3.14
reported the existing PyTorch JIT deprecation warning. No production numerical
source changed, so the complete E1.3/E7a/E6bcompat selections were not repeated;
their accepted results remain separate evidence. `py_compile` and
`git diff --check` pass. Ruff is unavailable in the active environment.
Black is also unavailable in the active environment.

An independent read-only source review found no blocker. It confirmed frozen
configuration and geometry ownership, explicit gradient-mode setup, mapped-
only isolation, per-lane position differentiation, cross-lane checks, shared
Model ownership, serial gradient comparisons, and lane-wise solver status.
Nonblocking review suggestions were handled in this report: the `energy` and
`serial` CLI stages are aliases, and parameter leaves are classified as
connected nonzero, connected zero, or disconnected. RSS deltas are described
as process measurements rather than allocation bytes.

The versioned raw JSONL and stderr records are in
`benchmarks/results/e6c1-w2c2a/`. The original W2 benchmark result remains
unchanged. Ruff and Black could not be run (`No module named ruff` / `No
module named black`).

## Verdict

- Frozen W2 configuration parity: **verified** for the mapped functional
  configuration and effective defaults.
- Position-only benchmark mode: **verified**.
- Model-parameter benchmark mode: **verified** for the measured one- and
  two-conformer derivative checks.
- Isolated mapped peak RSS: **measured** at 4,198.95 MiB (position-only) and
  recorded separately for position-plus-Model mode.
- Serial-reference memory contamination: **eliminated** from mapped-only
  measurements; the old 8.37 GiB figure is retained as combined-process
  history.
- Per-lane mapped/serial position gradients: **pass**.
- Cross-conformer position derivative blocks: **pass**, exactly zero in the
  measured result.
- Shared Model parameter-gradient accumulation: **pass** for all named leaves.
- Per-lane implicit response status: **verified**, all active lane solves
  converged with status 0.
- Primal SCF cost: **identified** at Fermi density construction and its batched
  matrix operations.
- Linear-response cost: **identified** as a major cost center; a more detailed
  per-operator profile is the next measurement.
- W2c2a: **accepted at the diagnostic/derivative boundary**.
- W2 architecture viability: **not demonstrated for performance**; the
  implicit response is correct in this two-lane case but expensive.
- W2 and W3: **open**. E6broute remains **blocked**.
- E6bcompat: **preserved**. Public Calculator routing: **unchanged**.
- Remote publication is reported after the final push verification.

## Next focused work

Use the profile evidence to isolate a single matrix-free response application
and determine whether it rebuilds Fermi-density quantities that could be
reused call-locally without changing autograd ownership. Measure that target
before proposing an optimization. Keep any production optimization separately
gated on the S0.1b numerical and higher-order transform contract.
