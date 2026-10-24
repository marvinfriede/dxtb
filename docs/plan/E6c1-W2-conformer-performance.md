# E6c1-W2 — CPU conformer performance and memory

## Decision

**W2 is BLOCKED.** The current legacy packed Calculator was measured on the
recorded 69-atom, eight-conformer `tmpda` workload. The accepted mapped E6a
path could not finish even a one-conformer, one-trip energy-plus-gradient
diagnostic within roughly 90 seconds at one thread or 120 seconds at four
threads. A full 300-trip run, serial fixed-trip comparison, conformer scaling,
and mapped peak-memory measurement therefore remain unavailable. No mapped
speedup ratio or performance clearance is claimed.

The mapped attempt also exposed a genuine GFN1 finite-temperature spectral
failure. The narrow correction and regression are included in this package;
they do not resolve the separate W2 performance blocker.

## Revision and changed files

- Starting SHA: `ee96ffb85ddb8b8e86fcc1fb212d40eab2f860cf`.
- Final SHA: recorded after the local package commit.
- Production fix: `src/dxtb/_src/scf/functional/spectral.py`.
- Regression: `test/test_scf/test_invariant_spectral_density.py`.
- Harness: `benchmarks/e6c1_w2.py`.
- Results: `benchmarks/results/E6c1-W2-cpu.json`.
- Roadmap: `docs/plan/07-current-state-and-execution-plan.md`.
- `AGENTS.md` remains untouched and unstaged.

The public Calculator route was not changed. The historical T0.8 JSON and
baseline workload generator remain unchanged.

## Workload and environment

The harness reuses `benchmarks/baseline/workloads.py`'s geometry source and
random policy: `tmpda`, 69 atoms, eight conformers, displacement standard
deviation 0.1 bohr, seed 0. It uses GFN1-xTB, PyTorch integrals, float64, full
SCF, Anderson mixing, 300 maximum iterations, 300 K electronic temperature,
and `f_atol=x_atol=1e-8`. Default configured classical terms are retained.
Energy and position gradients are evaluated from full production Results.

Primary host: AMD EPYC 7763 64-Core Processor, 128 logical CPUs, 503.7 GiB RAM,
Linux 6.4.0-150600.23-default, Python 3.14.8, PyTorch 2.14.1+cpu,
`tad-mctc==0.9.1`, official `tad-dftd3==0.7.0` wheel SHA256
`44bb0ddec5fdce42f4c50f0f8270a52a996390a25db827ef2e370110bd1bde7`, and
`tad-dftd4==0.8.0`. Each benchmark route ran in a fresh process. PyTorch
interop threads were fixed at one; intra-op threads were one or four, with
matching OMP/MKL thread settings and OpenBLAS fixed at one.

## Spectral correctness issue found before timing

The first serial and mapped fixed-trip calls failed in the 300 K Fermi path.
The initial GFN1 spectral matrix ranged from approximately -1.2419 to
+70.1037 Eh. At ordinary temperature, `_matrix_fermi` used an exponential
scale of 2048. For this spectrum, `exp_scaled + I` reached eigenvalues of
about -8.3e15 and +2.0e31 after matrix-exponential roundoff, so the required
Cholesky solve failed. The legacy eager route completed.

The production path now uses scale `2**22` and 21 fixed tanh-doubling steps
at all temperatures. This keeps the exponential close to the identity and
evaluates the same matrix Fermi function. `_binary_entropy_matrix` uses the
same helper and retained its existing 512-point quadrature. A dense 32-state
analytic regression fails with the old scale (minimum eigenvalue of the
Cholesky input approximately -1.06e15) and passes with the correction. No
physical overlap regularization, solver-tolerance change, or public routing
change was made.

The correction passed 86 spectral/free-energy/E7a tests and 27 E6a
conformer/transform tests in the primary environment. The new dense-spectrum
test checks the analytic matrix-Fermi response with `grad`, JVP, `jacfwd` and
`jacrev`; it and an existing nested third-order Fermi test passed in all five
supported CPU profiles. Details are in the JSON artifact.

## Current legacy packed measurements

Five warm measurements were taken for each thread count after one untimed
energy-plus-force warm-up. Forward and backward passes are separately timed.
The cold figure includes setup and the first energy-plus-force call.

| Threads | Setup | Cold setup + E/F | Warm forward median | Backward median | E/F median (IQR) | E/F range |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.133 s | 38.091 s | 18.191 s | 17.918 s | 36.215 s (0.236 s) | 35.965–36.457 s |
| 4 | 0.132 s | 18.160 s | 10.467 s | 6.531 s | 16.998 s (0.092 s) | 16.872–18.623 s |

All eight conformers returned the same 205 legacy iterations. The total energy
was -181.3355700136383 Eh at one thread and -181.3355700136381 Eh at four
threads. Across thread counts, the maximum per-conformer energy difference was
6.40e-14 Eh and the maximum force difference was 6.14e-11 Eh/bohr. Legacy
Results do not expose the functional tensor SCFStatus fields.

Process-isolated peak RSS was 8.157 GiB at one thread and 8.162 GiB at four
threads, from roughly 0.35 GiB starting RSS. The historical T0.8 record's
5.53 GiB peak was measured after multiple approaches in one process and is not
a per-route measurement.

## Mapped and serial status

The mapped harness uses exactly
`torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)`, with full
Results and gradient of the sum of atom-resolved energies.

| Route | Outcome |
|---|---|
| Legacy `Calculator(batch_mode=2)` | Complete; timings and process RSS above. |
| Independent serial `System.singlepoint` | The pre-fix attempt reproduced the same Fermi Cholesky failure as mapped. It was not timed after the scaling fix because the mapped one-trip diagnostic did not complete. |
| Mapped E6a `vmap(System.singlepoint)` | One conformer, one trip did not return within about 90 s at one thread or about 120 s at four threads; interrupted. No energy, force, status, or peak-RSS result was produced. |

Because the mapped route did not produce a result, energy/force parity and the
ratios `legacy/mapped` and `serial/mapped` are undefined. No mapped-versus-
serial result parity is claimed. The W2 path did not reach a usable point for
the required five-repetition timing, process-memory comparison, or 1/2/4/8
conformer scaling. The bounded run is evidence of an unresolved cost, not a
full W2 timing sample.

## Bottleneck evidence

A bounded one-trip run with repeated Python stack snapshots consistently
stopped in:

```text
solver.finalize_scf
  -> free_energy_from_spectral
  -> fermi_free_energy_matrix
  -> _binary_entropy_matrix
  -> _matrix_fermi
  -> _solve_positive_definite
```

The entropy routine evaluates 512 quadrature points and applies the 21-step
matrix Fermi doubling routine to batched 255-by-255 matrices. This is the
observed mapped fixed-trip bottleneck. One-thread timing attempts did not
complete; therefore no exact stage duration, mapped peak memory, or profiler
table is available. The profiler sample for the completed legacy one-conformer
four-thread call recorded 2.805 s E/F wall time; its largest self CPU entries
were `aten::bmm` (0.735 s), `aten::_linalg_eigh` (0.537 s), `aten::mul`
(0.124 s), and `aten::where` (0.112 s). Profiler overhead is excluded from the
benchmark table.

The legacy eight-conformer call used 205 iterations, while the functional
route is configured for 300 fixed trips, a 1.46x trip-count ratio before
accounting for its final spectral/free-energy work. This is a solver-policy
difference, but it does not explain the observed one-trip delay by itself.
The final entropy matrix calculation is the directly localized bottleneck.

## Historical comparison

| Threads | Historical legacy packed (setup included) | Current cold setup + E/F | Current warm E/F | Historical serial loop |
|---:|---:|---:|---:|---:|
| 1 | 32.487 s | 38.091 s | 36.215 s | 40.655 s |
| 4 | 17.274 s | 18.160 s | 16.998 s | 31.241 s |

Against the historical legacy figures, the current cold setup-plus-E/F times
are 17.3% slower at one thread and 5.1% slower at four threads. The current
warm E/F medians are 11.5% slower and 1.6% faster, respectively. These
historical comparisons remain uncontrolled across hardware and software
versions; the cold comparison is the closer match because the historical
timing included Calculator setup.

Historical measurements used dxtb 0.4.0 and PyTorch 2.14.0+cu130 on an x86_64
four-vCPU profile; the CPU model and affinity were not recorded. Current runs
used PyTorch 2.14.1+cpu on an EPYC 7763 host. The warm one-thread median is
about 11.5% slower than the historical time, while the warm four-thread
median is about 1.6% faster. These are not controlled same-hardware historical
comparisons. Current same-host thread scaling is 2.13x from one to four
threads for the legacy path.

The reproducible route commands were:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/e6c1_w2.py --route legacy --nconformers 8 \
  --threads 1 --repeats 5
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/e6c1_w2.py --route legacy --nconformers 8 \
  --threads 4 --repeats 5
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/e6c1_w2.py --route mapped --nconformers 1 \
  --threads 1 --maxiter 1 --repeats 1
```

The mapped diagnostic was also attempted at four threads and with 30-second
repeated `faulthandler` snapshots. For each of the five supported profiles,
the large-spectrum value/derivative regression (`grad`, JVP, `jacfwd`,
`jacrev`) and existing nested third-order Fermi test ran with
`pytest -q -n logical --dist worksteal`. The primary environment also ran the
86-test spectral/free-energy/E7a selection and the 27-test E6a conformer/
transform selection using the same worker policy. Benchmark timing runs were
not run under pytest-xdist.

Black, isort, compileall, `git diff --check`, and JSON parsing pass. Ruff and
mypy are not installed in this environment. No CUDA or no-libcint claim is
made; W2's measured routes use the supported CPU PyTorch backend.

## Required next action and verdict

W2 remains **OPEN / BLOCKED** pending an approved performance disposition or a
narrow, correctness-preserving optimization of the mapped finite-temperature
free-energy path. The next proposal should target `_binary_entropy_matrix`
quadrature cost, with first- through third-order derivative, permutation,
and padding controls retained. It must not remove the free-energy term, lower
the configured fixed-trip budget, or revert the E7a spectral correction.

No 100–1000 conformer result is claimed. Counts 1, 2, 4, 8, 16, and 32 were
not characterized for the mapped route. W3 remains a separate gate. Public
Calculator routing remains unchanged, and E6broute remains blocked by W2,
W3, broader route requirements, and the public packed Result adapter.
