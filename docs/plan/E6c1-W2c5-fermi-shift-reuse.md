# E6c1-W2c5 — Gauge-Invariant Matrix-Fermi Shift Reuse

## Status

W2c5 completed as a diagnostic-only feasibility study. No production
`spectral.py` changes were made. W2 and W3 remain open, E6broute remains
blocked, and public Calculator routing is unchanged.

| Field | Value |
|---|---|
| Starting source SHA | `573f706df2047d7ee96fdfeee25f8a0378cc5966` |
| Final diagnostic/source SHA | `cc6512b5c6d94049264cf0dbe8bf863f16354bae` |
| Branch | `claude/tender-dijkstra-cda42s` |
| Remote at start | `origin` = `git@github.com:marvinfriede/dxtb.git` |
| Dependency stack | mctc 0.11.3; multicharge 0.8.0; libcint 0.4.0; D3 0.7.0; D4 0.8.0 |
| Source changes | Diagnostic only; no production numerics changed |

The S1/S1.1 coordination boundary was read before work. S2a is active on
physical certification primitives and explicitly does not plan to edit
`spectral.py`. This package did not modify SCF-owned production or test files.
Any future production adoption still requires coordination at a clean boundary.

## Matrix-shift identity

Let `X=(H-μI)/(kT)` be the dimensionless symmetric spectral matrix and
`D=(I+exp(X))⁻¹`. Increasing the chemical potential by `δ` changes the
argument to `X' = X - (δ/kT)I`. Because the shift is scalar, it commutes with
`X`, `exp(X)`, and `D`. Set `a=exp(-δ/kT)`. Then

```text
exp(X') = a exp(X)
         = a (D⁻¹ - I)
I + exp(X') = D⁻¹ [aI + (1-a)D]
D' = [aI + (1-a)D]⁻¹ D
   = [I + expm1(-δ/kT)(I-D)]⁻¹ D.
```

This is an exact real-arithmetic identity for finite temperature and a Fermi
matrix produced by the same Hermitian spectral operator. The denominator
eigenvalues are `a(1-d)+d` for eigenvalues `d` of `D`. If `D` is symmetric with
spectrum in `[0,1]`, they are positive for finite `δ`, and the condition
number is bounded by `exp(abs(δ/kT))`. The identity carries derivatives through
both `D` and `δ`; no eigenvectors or detached chemical potential are used.
The zero-shift limit is the identity map and the `expm1` form avoids
small-shift subtraction cancellation.

That spectral condition is an assumption about the input density, not a
general guarantee for arbitrary approximate matrices. Saturation and finite
precision can move computed eigenvalues slightly outside `[0,1]`; then the
conditioning bound does not apply. Large positive and negative shifts can
amplify density errors or overflow. The diagnostic therefore uses a
conservative ratio guard `abs(δ/kT) <= -0.25 log(eps)` and substitutes safe
operands before solving invalid lanes, returning NaNs for them afterward. For
float64 this bounds the ideal-spectrum condition number by `eps**(-1/4) =
8192`. This guard is a tested experimental domain, not a proven production
bound for all SCF Newton corrections.

For the response slope, `Tr(D²) = sum_ij D_ij D_ji`, so the cubic matrix
product can be replaced by an elementwise contraction:

```python
count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
trace_square = (density * density.transpose(-2, -1)).sum((-2, -1))
slope = (count - trace_square) / safe_kt
```

This identity holds for symmetric `D`; the test also compares the reduction
and its first through third derivatives.

## Diagnostic and validation

`benchmarks/e6c1_w2c5_fermi_shift.py` is standalone and does not change
production dispatch. It compares the current two differentiable chemical
potential corrections (three full matrix-Fermi evaluations) with one full
matrix-Fermi evaluation followed by two invariant matrix-shift solves. It
also tests the trace-squared slope expression. The free-energy trace
experiment uses a scoped `_matrix_fermi` substitution restored in `finally`;
it is not a production implementation.

Complete asserted diagnostic runs are stored in:

- `benchmarks/results/e6c1-w2c5/spectral-py311-torch280-accepted.jsonl`
- `benchmarks/results/e6c1-w2c5/spectral-py314-torch2141-accepted.jsonl`

The separate `spectral-py311-torch280-partial.jsonl` preserves an earlier
interrupted invocation; it is not included in the accepted-run evidence.

Both ran on AMD EPYC 7763 with Python 3.11.15 / PyTorch 2.8.0+cpu and Python
3.14.8 / PyTorch 2.14.1+cpu, respectively. Each asserted one intra-op,
inter-op, OMP, MKL, and OpenBLAS thread and verified the source path and
accepted dependency versions.

The matrix-shift value identity and the reconstructed two-correction density
passed against the accepted invariant implementation for smooth, near- and
exact-degenerate, wide-gap, low-temperature partial-occupation,
noncontiguous-padded, and generalized-overlap fixtures. The generalized
overlap test checked the metric electron count. Matrix, temperature, electron
count, and scalar-shift derivatives were compared through third order,
including reverse-over-forward checks; the largest reported tested-fixture
errors were at floating-point noise scale (up to about `5e-12` in the
low-temperature third derivative). Exact degeneracy and rotated matrices
passed without eigenvector differentiation.

Mapped tests covered different temperatures, shifts, spectra, and structural
masks. Lane permutation preserved outputs, cross-lane derivative blocks were
zero, and an invalid over-limit lane returned NaNs without corrupting its
valid neighbor. The shift guard was tested immediately below, at, and above
both signed boundaries. At and below the guard the direct matrix-Fermi
comparison had maximum absolute error `6.1e-13`; above it the candidate
returned NaNs. Unguarded high-shift probes produced visibly inaccurate or
nonphysical values, confirming the need for the guard.

The free-energy trace candidate matched the current trace path for matrix,
temperature, and electron-count directional derivatives through third order
in the tested orthonormal fixture. Reverse-over-forward third-order checks
were additionally run for the explicit scalar shift and a scalar Hamiltonian
direction, rather than separately for each temperature and electron-count
direction. Generalized-overlap free-energy parity was not tested, and no
end-to-end physical or implicit-response calculation used the candidate.

## Trace-squared microbenchmark

Median microseconds per slope evaluation, single-thread CPU; the same
float64 matrix and derivative mode were used for each pair:

| N | Torch | Matmul forward | Trace-square forward | Forward reduction | Matmul reverse | Trace-square reverse | Reverse reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 2.8 | 27.5 | 21.2 | 23% | 475 | 437 | 8% |
| 32 | 2.14.1 | 22.6 | 19.0 | 16% | 475 | 413 | 13% |
| 128 | 2.8 | 188 | 54.7 | 71% | 992 | 526 | 47% |
| 128 | 2.14.1 | 183 | 52.1 | 72% | 964 | 506 | 48% |
| 255 | 2.8 | 1,133 | 99.6 | 91% | 3,895 | 694 | 82% |
| 255 | 2.14.1 | 1,162 | 90.8 | 92% | 3,965 | 691 | 83% |

Values and derivatives through third order passed. These timings isolate only
the slope kernel. They do not establish a full SCF or implicit-response
improvement.

The matrix-shift candidate is expected to reduce three full matrix-Fermi
calls to one full call plus two dense shift solves. Given the fixed 30 dense
solve calls per matrix-Fermi invocation in the accepted implementation, this
is a nominal reduction from 90 solves to 32 (about 64%). This is a call-count
estimate, not a measured timing result. No matched matrix-shift runtime,
matrix-shift derivative timing, implicit-response operator, or tmpda run was
performed.

## Review and decision

The independent read-only review found no algebraic or trace-squared blocker.
It required that the matrix-shift result be described as narrow feasibility,
not production acceptance; that the `[0,1]` spectrum assumption and signed
guard boundary be explicit; and that ignored diagnostics be included
deliberately. The report now records those limits and the boundary results.
Reviewer model identity was not exposed by the runtime.

**Decision:** retain the accepted invariant production implementation. The
matrix-shift identity is proved and numerically feasible in the guarded test
range, but its admissible range has not been established for all physical
Newton updates and its response-performance benefit is unmeasured. The
trace-squared identity is correct and has a promising isolated slope-kernel
benefit, but production integration is deferred until the SCF owner
coordinates it and an actual implicit-response/end-to-end measurement
confirms value. W2/W3 remain open; E6broute remains blocked.

No E7a or E6bcompat suites were rerun because no production code changed. No
production pytest regression claim is made. Black, isort, Python bytecode
compilation, and `git diff --check` were run for the diagnostic. Ruff was not
available in the active environment.
