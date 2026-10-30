# E6c1-W2c5a — Physical Fermi Correction Ranges and Performance Qualification

## Scope and disposition

Starting SHA: `4a6d8100d28332aedc455065962e7ed76b506a60`.

The accepted invariant matrix-Fermi implementation remains unchanged. The
trace-squared and gauge-invariant matrix-shift methods remain diagnostic-only.
The shift candidate is promising on sampled physical tmpda states, but its
complete S0.1b operating domain and free-energy correction path are not
established. Production integration is rejected for this package.

The first independent review found missing response-base correction telemetry,
instrumentation-contaminated timing comparison, and missing memory/report
artifacts. The revised physical harness probes the validated root's raw map
under `no_grad` outside `torch.func` and records those base-point correction
values. Matched baseline/candidate runs use one lightweight harness in fresh
processes with identical thread configuration and process peak RSS reporting.

## Environment and workload

- AMD EPYC 7763; Python 3.11.15; PyTorch 2.8.0+cpu; float64.
- Torch intra/inter-op, OMP, MKL and OpenBLAS threads: 1.
- tad-mctc 0.11.3, tad-multicharge 0.8.0, tad-libcint 0.4.0,
  tad-dftd3 0.7.0, tad-dftd4 0.8.0.
- Source imported from this worktree's `src/dxtb`.
- tmpda uses the frozen generator and W2 physical settings.
- High-temperature H2O and GFN2/Fock rows are diagnostic cases, not W2.

## Physical correction distribution

The candidate guard is `abs(delta / kT) <= 9.0109133473`. The response-base
probe evaluates the same raw physical map at the independently validated root
outside transforms. It records primal correction values, not tangent-specific
quantities inside the linear operator.

| Physical case / stage | Active channel corrections | Max `abs(delta/kT)` | Guard misses | Negative slopes |
|---|---:|---:|---:|---:|
| tmpda conformer 0, primal SCF | 264 | 1.6334e-12 | 0 | 0 |
| tmpda conformer 0, response base point | 4 | 3.1067e-13 | 0 | 0 |
| tmpda conformer 1, primal SCF | 408 | 1.7106e-12 | 0 | 0 |
| H2O GFN1, 300 K | no enabled Newton corrections | 0 | 0 | 0 |
| H2O GFN1, 10,000 K diagnostic | 1,204 | 1.0433e-13 | 0 | 0 |
| H2O GFN2/Fock, 10,000 K diagnostic | 1,208 | 1.8433e-12 | 0 | 0 |

Maximum count residuals were below `8.9e-16` for high-temperature H2O and
`5.7e-13` for tmpda. tmpda-0 converged in 66 map evaluations; its independent
residual L2/max were `6.01194e-9`/`4.35589e-9`. tmpda-1 converged in 96
iterations with residual L2/max `6.41606e-9`/`2.73939e-9`.

These sampled shifts are more than twelve orders of magnitude inside the
guard. This establishes coverage only for sampled trajectories and the
tmpda-0 response base point, not every accepted S0.1b input or failure state.

## Density spectrum and conditioning

Selected tmpda density matrices have 255 orbitals. Their eigenvalues are
within about `3.6e-15` of `[0,1]`; they are not strictly inside the interval.
For selected tmpda response-base matrices, the denominator condition number
was at most `1.0000000000018`. Shifted densities matched direct matrix-Fermi
values within `3e-15`. High-temperature H2O denominators had condition number
effectively one.

This supports the sampled cases, but not arbitrary approximate densities
outside `[0,1]`. No fallback is established that preserves existing production
behavior when the shift exceeds the candidate guard. This blocks production.

## Numerical and derivative checks

The diagnostic shift helper sanitizes invalid operands before division,
`expm1` and dense solve; inactive masked entries are excluded; invalid active
lanes propagate NaNs while valid lanes retain finite values and derivatives.
Checks include zero/negative/nonfinite temperature, nonfinite shift/density,
over-guard corrections and mixed valid/invalid lanes.

The asserted W2c5 diagnostic passed on Python 3.11/PyTorch 2.8 and Python
3.14/PyTorch 2.14.1 for tested matrix-shift values and first-through-third
derivatives on rotated, degenerate, near-degenerate, generalized-overlap,
padded and mixed-lane fixtures. Trace-squared passed its tested value and
derivative checks. The physical N=255 core benchmark adds first-directional
derivative parity, not a physical third-order test. Generalized-overlap and
padded free-energy shift behavior was not qualified.

## Density correction-core performance

The benchmark uses the same detached 255-orbital tmpda input for production,
trace-squared, matrix-shift and combined sequences. It excludes diagonalization,
AO transformation and Result construction. Nominal dense solve counts are 90
for the three-call production sequence and 32 for one Fermi evaluation plus
two shift solves.

The three-repeat physical N=255 run measured these medians (min/max in the
versioned JSONL):

| Correction core | Forward | Reverse | JVP |
|---|---:|---:|---:|
| Production | 1.195 s | 3.162 s | 3.137 s |
| Trace-squared | 1.162 s | 3.153 s | 3.067 s |
| Matrix-shift | 0.412 s | 1.086 s | 1.091 s |
| Combined | 0.401 s | 1.065 s | 1.074 s |

All four returned identical physical density values in this core fixture;
first directional derivatives agreed to floating-point noise. The process
lifetime peak RSS was 1,933 MiB across the sequential methods; it is not an
isolated per-method memory value. Trace-squared-only changes were small and
mixed in this complete correction core, so no trace-squared production
optimization is proposed.

## Matched tmpda implicit energy response

Fresh processes used the same lightweight harness, W2 configuration, dtype,
source, threads and observable (`result.energy.sum()`). The matrix-shift run
was asserted against the accepted production energy, free energy and position
gradient references.

| Metric | Production | Matrix-shift diagnostic |
|---|---:|---:|
| Root iterations / map evaluations | 66 / 66 | 66 / 66 |
| Root preparation | 102.36 s | 50.46 s |
| Finalization | 4.21 s | 3.07 s |
| Implicit response backward | 50.17 s | 41.54 s |
| Total root + finalization + backward | 159.82 s | 96.58 s |
| Peak process RSS | 2,505.10 MiB | 1,972.08 MiB |
| BiCGSTAB iterations / status | 5 / 0 | 8 / 0 |
| Response residual | 3.5334e-13 | 1.5427e-13 |
| Linearize calls / operator time | 13 / 37.75 s | 19 / 33.19 s |
| Energy | -29.758437462176378 Eh | -29.758437462176182 Eh |
| Free-energy field sum | -0.001391311654298254 Eh | -0.001391311654519453 Eh |
| Max position-gradient difference | reference | 1.1940e-10 |

This is one matched run per implementation, not a repeated timing distribution.
It is promising evidence for tmpda-0 but does not qualify the candidate's
complete numerical range. No two-conformer run was attempted. The earlier
instrumented captures remain historical; the comparable outputs are
`tmpda0-matched-production-py311.jsonl` and
`tmpda0-matched-matrix-shift-py311.jsonl`.

## Production decision

- Matrix-shift identity: established and tested in W2c5 fixtures.
- Sampled physical correction range: comfortably inside the guard.
- Full S0.1b numerical-domain preservation: not established.
- Trace-squared slope: mathematically equivalent; no compelling complete
  density performance benefit demonstrated.
- Matrix-shift tmpda energy/position response: close parity and converged
  response; one matched run shows reduced time and peak RSS.
- Generalized-overlap/padded free-energy correction path: not qualified.
- Production spectral source and public Calculator routing: unchanged.
- W2/W3 remain open; E6broute remains blocked.

Do not widen candidate fixtures indefinitely. The next decision must first
define a transform-compatible full-domain policy for approximate density
matrices and shifts beyond the guard. If that policy cannot preserve current
behavior, close matrix-shift reuse as diagnostic-only and assess a distinct
gauge-invariant matrix-function approach in a separately gated package.

## Independent review status

The first read-only review's three findings prompted the response-base probe,
matched fresh-process timing runs, RSS reporting and this report. Per-method
microbenchmark memory remains cumulative rather than isolated. A follow-up
review must inspect the revised code and traces before this package is marked
accepted; the performance figures remain one-run evidence.

## Reproduction and validation

The final asserted matrix-shift diagnostic completed on both primary profiles:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 180s \
  <py311-torch2.8>/bin/python -u benchmarks/e6c1_w2c5_fermi_shift.py \
  --output benchmarks/results/e6c1-w2c5a/w2c5-diagnostic-final-py311.jsonl

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 180s \
  <py314-torch2.14.1>/bin/python -u benchmarks/e6c1_w2c5_fermi_shift.py \
  --output benchmarks/results/e6c1-w2c5a/w2c5-diagnostic-final-py314.jsonl
```

The physical-only correction benchmark used three repeats and the captured
tmpda-0 255-orbital input. The matched response command was run twice in fresh
processes with `--mode production` and `--mode matrix-shift`; both used the
same accepted baseline JSONL for numerical assertions. Their full commands
and environment are recorded in the scripts and output traces.

Black, isort, Python byte compilation and `git diff --check` passed. Ruff was
not installed in the available environments. No production SCF/spectral code
changed, so E7a/E6bcompat pytest selections were not rerun; their previously
accepted evidence remains the applicable production boundary.
