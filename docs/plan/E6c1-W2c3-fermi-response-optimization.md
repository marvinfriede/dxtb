# E6c1-W2c3 — Spectral Cost Diagnosis and Candidate Assessment

## Status

**Performance diagnosis: COMPLETE.** **Production optimization: NOT IMPLEMENTED.**
The accepted matrix-Fermi kernel remains unchanged. The isolated Cholesky
candidate was rejected because it made the full tmpda position response slower,
despite modestly faster forward-only matrix-function calls.

W2 and W3 remain open; E6broute remains blocked. Public Calculator routing is
unchanged.

## Revisions and environment

- Starting SHA: `a0f35c92e7357752e2f4cc8ec2ba2b3f16ba4423`
- Candidate diagnostic SHA: same source baseline; no production SCF change
- Branch: `claude/tender-dijkstra-cda42s`
- Initial remote SHA: `a0f35c92e7357752e2f4cc8ec2ba2b3f16ba4423`
- Primary host: AMD EPYC 7763, CPU, one intra-op and one inter-op thread
- Profiles: Python 3.11.15 / Torch 2.8.0+cpu and Python 3.14.8 / Torch
  2.14.1+cpu
- Dependencies: tad-mctc 0.11.3, tad-multicharge 0.8.0, tad-libcint 0.4.0,
  tad-dftd3 0.7.0, tad-dftd4 0.8.0
- Workload: frozen tmpda conformer 0, GFN1, PyTorch integrals, float64, charge
  SCP, FULL SCF, Anderson, 300 K, 300 maximum iterations, `f_atol=x_atol=1e-8`,
  `x_atol_max=1e-5`

No dependency, physical setting, reference, production route, or derivative
strategy changed.

## Commands and validation

The main raw-map and convergence-state profile was run in a fresh process with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp1/friede/dxtb-agentic/envs/py3.11-torch2.8.0/bin/python -u \
  benchmarks/e6c1_w2c3_fermi.py --mode tmpda-profile
```

The isolated Fermi measurements used the same harness with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp1/friede/dxtb-agentic/envs/py3.11-torch2.8.0/bin/python -u \
  benchmarks/e6c1_w2c3_fermi.py --mode solver-microbench --repeats 5
```

The same command was run with the Python 3.14 interpreter. Boundary checks used
`--sizes 32 --powers 14,15,21,22,23,29,30 --repeats 1`; control-flow and
mixed-power lane checks used `--mode control-flow-probe` and
`--mode mixed-power-vmap` in both primary environments.

The full tmpda candidate-response run used the accepted W2c2a
`run_diagnostic(stage="energy", observable="energy")` entry point in a fresh
Python 3.11 process, with charge/FULL/Anderson settings and a 12,000 MiB memory
cap. A scoped wrapper replaced `spectral.dense_solve` with
`spectral._solve_positive_definite` and restored the original in `finally`.
No production code or public route was changed by the probe. Its JSONL output
is `tmpda-cholesky-response-py311-torch280.jsonl`.

Python compilation passed on both primary interpreters. Black and isort checks
passed in the Py3.14 environment; Ruff was unavailable. `git diff --check`
passed. No pytest suite was rerun because the production spectral and solver
code remained unchanged.

## Scaling and discarded work

The new harness captured actual states from the eager W2 controller: the
initial input, map evaluations 25 and 50, and the converged root. All four
states selected power 16 for both spin channels. Each raw SCF map invoked the
matrix-Fermi function three times, each on two 255-orbital spin matrices.

| State | Calls | Required/selected power | Active outer doublings per lane | Masked outer doublings per lane | Dense solves |
|---|---:|---:|---:|---:|---:|
| Initial | 3 | 16 / 16 | 15 of 29 | 14 of 29 | 90 total; 42 masked |
| Map evaluation 25 | 3 | 16 / 16 | 15 of 29 | 14 of 29 | 90 total; 42 masked |
| Map evaluation 50 | 3 | 16 / 16 | 15 of 29 | 14 of 29 | 90 total; 42 masked |
| Converged root | 3 | 16 / 16 | 15 of 29 | 14 of 29 | 90 total; 42 masked |

Each call makes one initial solve and executes all 29 doubling solves. At power
16, 15 doubling results are retained and 14 are masked away. Thus `42/90`
dense-solve results per map (`46.7%`) are discarded; this is `14/29` of the
doubling solve work. Each inner `_matrix_expm1()` selects power 5 while its
fixed loop executes 8 squaring products, so 3 inner products per Fermi call are
also masked. The outer loop computes two matrix products for every doubling,
including the 14 masked results.

The scoped dense-solve timer measured 0.64–0.79 s across the 90 solve calls in
the profiled raw map. The profiler separately reported about 0.42–0.44 s in LU
solves and 0.15 s in LU factorization. These instrumentation timings include
profiler/wrapper overhead and are not used as clean speed benchmarks.

The map-wide profiler observed 573 `bmm`/`mm`/`matmul` operators and 0.64–0.68
seconds of aggregate self time for them. **That is an aggregate for the full
`scf_map`, not a Fermi-only matrix-product time.** Isolated Fermi forward times
below provide the clean kernel timing boundary. The map profiler also saw 90
LU factorizations and solves. The existing W2c2a map profile measured the
Fermi-density component at about 1.59–1.65 seconds for the full raw map, which
includes more than `_matrix_fermi()` alone.

The converged root required 66 map evaluations, with final residual components
`6.01e-9` and `4.36e-9`. The one-map profiler confirmed finite density and the
scaling metadata above. No new numerical instability was observed.

## Candidate assessment

### Transform-compatible conditional execution

Small probes of `torch.cond` and `torch.while_loop` were run on both primary
profiles. In the normal eager execution path, `torch.cond` failed because it
requires complete capture by `torch.compile`; its grad, JVP, and vmap probes
also failed. `torch.while_loop` ran its primal eager example, but its grad and
JVP probes failed, and its vmap probe failed. These controls therefore do not
provide a viable drop-in conditional for the current eager and
`torch.func`-transformed kernel. This did not test an end-to-end compiled SCF
route; the project does not currently execute this path through such a route.

### Static specialization

A global static loop limit must still cover the documented supported power 30.
The tmpda observation of power 16 cannot justify lowering that limit for other
valid inputs. No safe static specialization was identified.

### Alternative matrix functions

No rational/Pade or eigendecomposition formulation was prototyped. Each would
replace the accepted invariant matrix-function approximation and require a
separate range, padding, and first-through-third-derivative acceptance. The
current evidence does not justify that broader change.

### Cholesky substitution for the dense solve

The existing `_solve_positive_definite()` helper was used only as a scoped
benchmark substitution for the current `dense_solve`. The Fermi denominators
are symmetric positive definite over the tested inputs. The diagonal test
fixture used an independent fixed-chemical-potential scalar Fermi-Dirac
oracle, including analytical first, second, and third derivatives. Powers
14, 15, 21, 22, 23, 29, and 30 passed in both primary profiles. Third-order
absolute errors were approximately `1e-16`. A `vmap` with powers 14, 22, and
30 matched serial values and had exactly zero cross-lane Jacobian blocks.

Five-run median forward timings (seconds) were:

| Profile | Orbitals | `dense_solve` median (min–max) | Cholesky median (min–max) | Change |
|---|---:|---:|---:|---:|
| Py3.11 / Torch 2.8 | 32 | 0.009896 (0.009830–0.010165) | 0.004966 (0.004903–0.004995) | -49.8% |
| Py3.11 / Torch 2.8 | 128 | 0.042832 (0.042138–0.043865) | 0.038667 (0.038419–0.040376) | -9.7% |
| Py3.11 / Torch 2.8 | 255 | 0.200862 (0.200170–0.203728) | 0.184245 (0.182930–0.184625) | -8.3% |
| Py3.14 / Torch 2.14.1 | 32 | 0.009602 (0.009498–0.009628) | 0.004574 (0.004572–0.004624) | -52.4% |
| Py3.14 / Torch 2.14.1 | 128 | 0.042725 (0.041828–0.043687) | 0.037630 (0.037501–0.038456) | -11.9% |
| Py3.14 / Torch 2.14.1 | 255 | 0.203319 (0.200420–0.203529) | 0.181885 (0.181545–0.182082) | -10.5% |

The 255-orbital derivative timings were less favorable. On Py3.11, the
Cholesky first derivative was about 6% faster and its JVP about 4% slower; the
second and third derivatives were about 28% and 57% slower. On Py3.14, first,
JVP, second, and third derivatives were about 34%, 23%, 43%, and 42% slower.
The third-order oracle still passed; this is a performance rejection, not a
correctness failure.

## tmpda end-to-end candidate response

The candidate was evaluated in a fresh process by temporarily substituting the
Cholesky helper for `dense_solve` and running the accepted W2c2a energy-only,
positions-only diagnostic. The monkey patch was restored in `finally`; no
production file changed.

| Measure | Accepted baseline | Cholesky experiment |
|---|---:|---:|
| Total time | 164.44 s | 170.13 s |
| Peak RSS | 2534.72 MiB | 2501.84 MiB |
| Root preparation | 102.14 s | 95.82 s |
| Implicit response solve | 40.41 s | 50.84 s |
| Backward | 51.34 s | 64.04 s |
| `linearize` time | 38.72 s | 49.32 s |
| Response iterations / status | 5 / success | 5 / success |
| Energy | -29.758437462176378 | -29.758437462176396 |
| Fermi free energy | -0.0013913116542982552 | -0.0013913116545194542 |

The energy difference was `1.78e-14 Ha`; maximum force-component difference
was `2.98e-13 Ha/bohr`. The candidate reduced RSS by about 33 MiB (1.3%) but
increased total runtime by 5.69 s (3.5%) and `linearize` time by 27%. Its
BiCGSTAB status succeeded with five iterations; its final response residual
was `3.59e-13` (`9.99e-7` relative). The accepted baseline response residual
was `3.53e-13`. This is direct evidence that forward-only solve savings do
not transfer to the implicit response path.

## Decision and next work

No production optimization was selected. The Cholesky substitution is
numerically sound in the tested matrix-function cases, but it is rejected
because it regresses end-to-end response time and higher-order derivative
costs, while providing only a small memory reduction. Conditional loops are
not supported by the present eager plus transform execution path, and a
global static schedule cannot reduce work while preserving power 30.

The next viable investigation is a separately scoped compiled-control-flow
prototype or a new matrix-function approximation with a complete independent
derivative/range proof. It must first demonstrate `vmap`, JVP, reverse mode,
and third-order compatibility at scaling boundaries before touching production
code. Do not lower the scaling cap or switch the accepted solve in the interim.

S0.1b and E6bcompat remain at their accepted starting SHA and are preserved.

## Independent review

An independent read-only review was completed after the report and roadmap
were updated. It found no remaining issue. The reviewer confirmed the scaling
and discarded-solve accounting, scalar-oracle and mixed-power lane checks,
candidate-versus-production distinction, and the end-to-end basis for
rejecting Cholesky. The two initial minor review notes (aggregate labeling and
candidate oracle/JVP assertion quality) were corrected. Requested reviewer:
Luna / high; actual model identity is not exposed by the runtime.

## Artifacts

The diagnostic harness is [`benchmarks/e6c1_w2c3_fermi.py`](../../benchmarks/e6c1_w2c3_fermi.py).
Versioned JSONL traces are under `benchmarks/results/e6c1-w2c3/`, including:

- `baseline-tmpda-profile-py311-torch280.jsonl`
- `solver-microbench-py311-torch280.jsonl`
- `solver-microbench-py314-torch2141.jsonl`
- `solver-derivative-profile-py311-torch280.jsonl`
- `solver-boundaries-final-py311.jsonl` and `solver-boundaries-final-py314.jsonl`
- `control-flow-py311.jsonl` and `control-flow-py314.jsonl`
- `mixed-power-vmap-py311.jsonl` and `mixed-power-vmap-py314.jsonl`
- `tmpda-cholesky-response-py311-torch280.jsonl`

The map-wide matrix-product profile is aggregate, as described above; the
dense-solve count/timing is scoped to `_matrix_fermi`.
