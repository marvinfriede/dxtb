# E6c1-W2c2 — tmpda implicit-root feasibility

## Scope and status

This is a private diagnostic of the existing converged-root implicit-SCF
implementation. It does not change production SCF, public Calculator routing,
the dependency stack, or the fixed-trip derivative contract.

- Starting SHA: `2b26d6f456477bba18a31bb6d9e1422c64a7a5f0`.
- Validated harness and trace SHA: `04ad6c4f32ed3fd50292b42cfccd9a84eac53c5a`.
- Final report/roadmap commit follows the validated diagnostic commit.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote: `origin` (`git@github.com:marvinfriede/dxtb.git`); publication is
  reported only after push verification.
- Dependencies: mctc 0.11.3, multicharge 0.8.0, libcint 0.4.0, D3 0.7.0,
  D4 0.8.0.
- Environment: Linux x86_64, host `m199`, Python 3.11.15, PyTorch
  2.8.0+cpu, CPU-only, OMP/MKL/OpenBLAS and PyTorch threads set to 1.
- Each calculation ran in a fresh process with a 16,000 MiB RSS watchdog and
  a bounded external timeout. The original W2 workload settings were retained:
  69-atom tmpda, GFN1, float64, PyTorch integrals, 300 K, FULL, Anderson,
  maxiter 300, `f_atol=x_atol=1e-8`, production classical terms. The frozen
  mapped W2 benchmark does not specify `x_atol_max`; its effective value is
  `defaults.X_ATOL_MAX` (1e-5). The original W2c2 harness explicitly used
  `x_atol_max=1e-8`, so those results are a stricter-convergence diagnostic,
  not an exact frozen-W2 configuration measurement. See the reconciliation in
  [`E6c1-W2c2a-mapped-gradient-and-performance.md`](E6c1-W2c2a-mapped-gradient-and-performance.md).

The principal commands used the Python 3.11 / Torch 2.8 environment and one
thread. Each command redirected stdout and stderr to a unique file under the
versioned trace directory:

```bash
timeout 1200s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u benchmarks/e6c1_w2c2_implicit.py \
  --stage root --threads 1 --memory-limit-mb 16000

timeout 1200s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u benchmarks/e6c1_w2c2_implicit.py \
  --stage energy --observable energy --threads 1 --memory-limit-mb 16000

timeout 1200s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u benchmarks/e6c1_w2c2_implicit.py \
  --stage energy --observable energy-plus-free-energy --threads 1 \
  --memory-limit-mb 16000

timeout 1200s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u benchmarks/e6c1_w2c2_implicit.py \
  --stage mapped --nconformers 2 --threads 1 --memory-limit-mb 16000
```

The four finite-difference roots used the same `--stage root` command with
`--displace-atom 0 --displace-axis 0` and offsets `+/-0.001` and `+/-0.002`.
The focused test command was
`python -m pytest -q test/test_scf/test_e6c1_w2c1_mapped_implicit.py -vv --tb=short`
in each primary environment.

The report accepts the diagnostic as a bounded feasibility result. It does
**not** accept the implicit architecture for W2 and does not clear W2/W3.

## Workload and observable definitions

The harness reuses `benchmarks/baseline/workloads.py`'s tmpda geometry. It
uses the frozen seed 0, 0.1 bohr displacement generator and takes conformer 0
with its prescribed zero displacement. The setup includes Repulsion, D3 0.7,
Halogen, ES2 and ES3.

Two observables were evaluated separately, as required:

1. W2's energy-only target is `result.energy.sum()`. In this production Result,
   `energy` contains the atomwise SCF energy, Fermi free energy and classical
   terms.
2. The separate `result.energy.sum() + result.fenergy.sum()` diagnostic adds
   the Fermi free-energy field once more. It is not substituted for the W2
   target.

At conformer 0, the respective values were `-29.758437462176378 Eh` and
`-29.759828773830677 Eh`. The summed `fenergy` field is
`-0.0013913116542982537 Eh`.

## One-conformer physical root

The primal root was prepared without an SCF trajectory differentiation graph.
The raw physical map was evaluated independently using the exact SCF tensor
inputs later supplied to implicit attachment. The root passed all gates:

| Diagnostic | Result |
|---|---:|
| SCF map evaluations / iterations | 66 / 66 |
| Solver residual components | `6.0119432e-9`, `4.3558941e-9` |
| Independent raw-map residual L2 | `6.0119432e-9` |
| Independent raw-map max component | `4.3558941e-9` |
| Electron-count error | 0 for both spin channels |
| Fermi convergence / residual | converged / 0 for both channels |
| Trajectory and final failure code | 0 / 0 |
| Density and energy finiteness | finite |
| Root solve time / peak RSS | about 102 s / 515 MiB |

The independently validated root used the same position-dependent integral
and interaction tensors for implicit attachment. There was no unrolled SCF
graph to detach accidentally. The Model's floating parameter leaves were
enabled before setup, and setup-derived tensors retained their ownership path.

## Implicit energy and force response

The response instrumentation was scoped and restored in `finally`. The
primary energy-only run completed in 163.6 s total (165.3 s in the separate
position-only instrumentation run) at 2.58 GiB peak RSS. It spent 5.91 s
before BiCGSTAB and 39.6 s in the response solve. BiCGSTAB converged in 5
iterations with status 0 and residual norm `3.53e-13`. The position gradient
was finite, with norm `41.2947 Eh/bohr` and maximum absolute component
`11.4680 Eh/bohr`. With actual Model parameters enabled, 77 Model-owned leaves
received finite gradients; 2,349 floating parameter leaves were available.

The separate energy-plus-free-energy gradient completed in 232.5 s at 2.59
GiB peak RSS. Its larger RHS norm (`0.28409`, compared with `3.60e-7` for the
energy-only observable) required 17 BiCGSTAB iterations and 110.0 s in the
response solve. Status was 0 and residual norm was `2.84e-11`. Its position
gradient was finite, with norm `41.2932` and maximum absolute component
`11.4677 Eh/bohr`; 77 Model-owned leaves received finite gradients.

The two response costs are separate and must not be combined or represented as
one observable. Both measured costs are dominated by the response work after a
roughly 100 s root preparation. No 300-trip unrolled-gradient run was made.

## Independent tmpda force check

Atom 0, x coordinate, was displaced by `+/-0.001` and `+/-0.002` bohr. Each
displaced geometry was independently converged and checked against its own raw
map; no displaced point reused the undisplaced root. The four roots converged
in 65, 66, 67 and 68 iterations. Their independent residual norms were below
`8e-9`, and each had valid Fermi and electron-count status.

The implicit gradients and central finite differences were:

| Observable | Implicit dE/dx (Eh/bohr) | FD, h=0.001 | FD, h=0.002 |
|---|---:|---:|---:|
| `result.energy.sum()` | `0.15392710` | `0.15391600` | `0.15388269` |
| `result.energy.sum()+result.fenergy.sum()` | `0.15304889` | `0.15303778` | `0.15300445` |

The forces are the negatives of these derivatives. The finite-difference
estimates move toward the implicit value by a factor consistent with the
second-order central-difference truncation when the step is halved. Richardson
extrapolation gives `0.15392709978` and `0.15304888850 Eh/bohr`, respectively,
matching the implicit values at this coordinate. This is a selected-coordinate
check, not a full force-vector reference or an exhaustive parameter-gradient
reference.

## Two-conformer mapped diagnostic

The first two frozen conformers were independently prepared using one shared
Model owner and lane-specific positions, integrals and interaction data. Their
roots passed independent residual validation:

| Lane | Iterations | Independent residual L2 | Max component |
|---|---:|---:|---:|
| 0 | 66 | `6.0119432e-9` | `4.3558941e-9` |
| 1 | 96 | `6.4160559e-9` | `2.7393936e-9` |

Mapped energy and free-energy values were explicitly asserted against the
independently finalized serial lanes. Energy differences were at most
`5.3e-10 Eh`; free-energy differences were below `6e-13 Eh`. The summed
energy-only mapped position gradient was finite (norm `78.1043`, maximum
`20.5749 Eh/bohr`). The mapped forward evaluation took 7.32 s and the backward
pass 123.1 s; full root, mapping, serial-reference and backward time was
394.7 s. Peak RSS was 8,371.6 MiB, after sequential root preparation had used
about 630 MiB.

This two-lane result is useful feasibility evidence, but its derivative
acceptance is incomplete. This run did not record mapped BiCGSTAB status and
per-lane response residuals, did not compare each lane's mapped gradient with
an independently computed serial implicit gradient, did not evaluate
cross-lane Jacobian blocks, and did not measure shared Model-parameter
gradient accumulation across lanes. No assertion of cross-lane derivative
independence is made from the summed gradient alone. These gaps prevent a
claim that the mapped implicit architecture is accepted for W2.

Four- and eight-conformer calculations were not attempted. The original
combined two-lane diagnostic included serial derivative-reference graphs in
its 8.37 GiB process peak. W2c2a later isolated mapped execution at 4.10 GiB,
and added lane-wise response status, per-lane serial position-gradient parity,
zero cross-lane position blocks, and shared Model parameter-gradient
accumulation. The old peak is retained as combined-process history, not an
isolated mapped measurement. Stopping at two lanes avoids an unsafe
extrapolation or a misleading throughput claim; see
[`E6c1-W2c2a-mapped-gradient-and-performance.md`](E6c1-W2c2a-mapped-gradient-and-performance.md).

## Comparison and limitations

The historical packed legacy W2 reference is 36.215 s for eight conformers at
one thread, 16.998 s at four threads, and about 8.16 GiB peak RSS. It was
measured on its recorded hardware, not this host, and it uses legacy
early-stopping SCF. The current prototype separately prepares converged
roots and computes implicit responses. Therefore these timings do not support
a direct speed ratio. They do show that this prototype has not established a
practical W2 performance advantage: a single conformer's energy-plus-force
calculation took 165 s, and two mapped conformers took 395 s with 8.37 GiB
peak RSS. Same-host legacy comparison, four/eight-lane evaluation, complete
mapped derivative blocks and response-status instrumentation remain open.

The two-conformer harness initially stopped before mapped evaluation due to a
missing local `implicit_root` import. The corrected retry completed; the
initial harness error is retained in the raw log and is not classified as a
numerical failure. An earlier root run also exposed an editable-install path
pointing to an older checkout; it was discarded, the harness was fixed to
import this worktree, and the accepted candidate runs report the active source
path. Raw stdout includes package setup messages in addition to JSON records;
clean event-only JSONL extracts are saved next to the raw logs.

No production numerical source changed. The existing W2c1a mapped-root,
mismatch-rejection and finite-temperature tests passed in both Python 3.11 /
Torch 2.8 (`3 passed in 53.90 s`) and Python 3.14 / Torch 2.14.1 (`3 passed in
49.83 s`, one upstream PyTorch JIT deprecation warning). The accepted E6bcompat
and E7a boundaries were not rerun because this package changes only a private
diagnostic harness; their earlier accepted evidence remains the source of
those claims.

## Files and evidence

- Private harness: `benchmarks/e6c1_w2c2_implicit.py`.
- Versioned raw and event-only traces:
  `benchmarks/results/e6c1-w2c2/`.
- Original W2 JSON and benchmark were not modified.
- `black`, Python byte compilation and `git diff --check` pass.
- Independent review found no production correctness issue. It confirmed the
  validated root/parameter pairs and the single-lane results, and identified
  the mapped-response and cross-lane derivative gaps listed above.
- The independent review confirmed that the finite-difference errors scale
  by approximately four as the central step doubles and that the Richardson
  estimates match the implicit derivatives at the selected coordinate.

## Verdict

- One-conformer primal root: **converged and independently validated**.
- Energy-only implicit position response: **completed, finite, solver passed**.
- Separate energy-plus-free-energy response: **completed, finite, solver
  passed**.
- Active Model parameter connection: **verified finite for 77 leaves**; no
  tmpda-specific independent parameter finite-difference comparison.
- Selected tmpda force coordinate: **consistent with two independently
  rooted finite-difference step sizes**.
- Two-conformer mapped values and summed position gradient: **completed and
  finite**; mapped response metadata, per-lane gradient parity, cross-lane
  blocks and shared parameter accumulation are **not verified**.
- W2c2 diagnostic: **accepted as a bounded feasibility diagnosis**.
- Implicit architecture viability for W2: **blocked by incomplete mapped
  derivative validation and poor measured cost/memory at two lanes**.
- W2 and W3: **open**. E6broute remains **blocked**.
- E6bcompat: **preserved**; no production files changed.
- Public Calculator routing: **unchanged**.
